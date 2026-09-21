"""Convex clipping in direction space, with sign tests a meshed enclosure cannot upset.

Cutting a convex region with a half-space is the one primitive every closed-form visibility
answer in this package is built from: a triangle cut to the half-space a receiver can see, a
blocker cut to the near side of a source's plane, a source's angular extent cut by each edge of
a blocker's silhouette. Sutherland and Hodgman's (1974) algorithm does it in one pass per plane,
and it is written here as a **fixed** pass -- every vertex and every edge crossing emitted
unconditionally and selected with a mask -- because a traced, compiled program cannot carry a
vertex count that depends on the values.

⚠️ **THE HARD PART IS NOT THE CLIP, IT IS THE SIGN TESTS UNDER IT.** Sutherland and Hodgman ask
only two questions per edge: is this vertex on the kept side, and does this edge change sides.
Both are the sign of a height ``h = (x - c) . n``. In a meshed enclosure those heights are
**exactly zero far more often than not**: neighbouring triangles share vertices and edges, two
triangles of one flat wall are coplanar, and a receiver sits in the plane of every triangle on
its own facet. A height that is zero in exact arithmetic is computed as a few parts in ``1e19``
of noise whose *sign* is set by how the compiler happened to fuse the multiplies and adds -- and
that changes with the batch shape, with whether the code is compiled at all, and even with the
presence of an arithmetically dead term such as ``0.0 * x`` elsewhere in the expression. The
consequence is not a rounding: a vertex judged one side rather than the other changes which
polygon comes out, so the same geometry gives different answers depending on how it was
evaluated, and a real occluder can vanish from the result without any error being raised.

**Two filters remove it, one on each quantity a sign is read from, and neither suffices alone.**
Each is the first stage of the standard construction for exact geometric predicates (Shewchuk
1997): compute the quantity in floating point, bound how far that can be from the exact value,
and trust the sign only when the magnitude clears the bound.

* **Heights** (:func:`decidable_heights`). What the standard construction does when the bound is
  not cleared is recompute in exact arithmetic. **Here it does not have to**, which is what makes
  this cheap: the undecidable case is answered **zero**, and zero is a correct and consistent
  answer for a clip. The kept half-space is closed, so a vertex on the boundary belongs to it; and
  an edge with a zero at one end does not cross, because its crossing *is* that endpoint, already
  emitted. A triangulation cannot do this -- it must pick a side -- which is why the exact stage
  is unavoidable there and avoidable here.
* **Planes** (:func:`spanning_plane`). A clip emits a crossing at the very end of an edge as
  ``a + 1.0 * (b - a)``, which is ``b`` to within a rounding but not bit for bit, so a loop can
  carry two corners that are the same direction without being equal. The plane through them is
  mathematically zero and computationally noise, and clipping by that noise empties a loop that
  should have been left whole. Snapping the plane to zero makes the stage the no-op it is meant to
  be, and the remaining edges still bound the polygon.

**Why both.** Without the plane filter the heights alone are not enough, because near-coincident
corners arise from ordinary rounding and not only from degenerate heights: on a sleeved box of
364 facets, a triangle covering 0.37 of a source read as covering nothing, on 25 pairs. Without
the height filter the planes alone are not enough, because not every undecided sign produces a
pair of corners for the plane filter to catch -- a sign decided while cutting the source by a
blocker's edge selects which part of the source every later cut sees. On the same box, one
triangle covering 0.0020 of a source read that eagerly and 0.0 once compiled. Smaller versions of
the same body, at 80 and 156 facets, were equally right with either filter removed, which is
worth knowing before trusting any fixture to show that one of them can go.
"""

from __future__ import annotations

import jax.numpy as jnp

from aquaflux.vectors import dot, norm_squared

__all__ = [
    "clip_to_halfspace",
    "decidable_heights",
    "spanning_plane",
]

#: The double-precision unit roundoff: half a unit in the last place of a number near one.
_ROUNDING = 2.0**-53

#: How many units of roundoff a quantity must clear before its sign is trusted, as a multiple of
#: the sum of the magnitudes that went into it.
#:
#: Four would suffice for the arithmetic alone -- one rounding for a subtraction, one per
#: product, one per addition of the three terms -- and fusing a multiply into an add only ever
#: reduces that. The margin over four is for the operands: a clip's output feeds the next clip's
#: input, so the vertices tested here are often crossing points that already carry a few units
#: of their own. Being generous costs only the width of the band in which a genuine height is
#: declared to be a zero, and at this width that band is parts in ``1e15`` of a solid angle.
_SLACK = 16.0 * _ROUNDING


def spanning_plane(left, right):
    """The plane through the origin containing two directions, or **exactly zero** if there is none.

    Two directions span a plane unless they are parallel, and a fixed-width loop is full of
    parallel pairs by construction: a slot repeated to pad the loop out gives a zero-length edge
    whose two endpoints are the same direction. The cross product of a direction with itself is
    mathematically zero and computationally a vector of last-bit noise, and that noise is not
    harmless -- fed to :func:`clip_to_halfspace` it cuts a loop that the real, zero plane would
    have left untouched, which can empty a loop that should have been kept whole.

    So the magnitude is compared against what the cross product's own rounding could have
    produced from parallel inputs, and anything within it is returned as the zero vector, which
    :func:`clip_to_halfspace` treats as an exact no-op. Comparison is on squared magnitudes, so
    no square root is taken.

    A *non*-degenerate plane is left as the cross product computes it, a few units in the last
    place off the exact one, and nothing here tries to do better. It does not need to: the only
    thing this module ever reads a sign from is a height, and :func:`decidable_heights` allows
    for operands carrying a few units of their own, so a plane that is slightly off cannot make
    a sign test answer a degenerate configuration wrongly. Protecting the predicate that is
    consumed is stronger than making every quantity that feeds it reproducible.

    Parameters
    ----------
    left, right : jnp.ndarray, shape ``(..., 3)``
        Two directions from the origin.

    Returns
    -------
    jnp.ndarray, shape ``(..., 3)``
        Their cross product, or zeros where the two are indistinguishable in direction.
    """
    plane = jnp.cross(left, right)
    bound = _SLACK**2 * norm_squared(left) * norm_squared(right)
    return jnp.where((norm_squared(plane) <= bound)[..., None], 0.0, plane)


def decidable_heights(loop, plane_normal, through=None):
    """Signed heights of a loop above a plane, with an undecidable sign resolved to exact zero.

    The height of vertex ``x`` above the plane through ``c`` with normal ``n`` is ``(x - c) . n``,
    positive on the side ``n`` points into. Where the true height is zero -- a shared vertex, a
    coplanar triangle, a receiver in a blocker's own plane -- the computed one is noise of either
    sign, so it is snapped to the zero it mathematically is. See this module's own description
    for why zero is not merely a safe answer but the correct one.

    Parameters
    ----------
    loop : jnp.ndarray, shape ``(..., n, 3)``
        Vertices, as directions from the receiver.
    plane_normal : jnp.ndarray, shape ``(..., 3)``
        Normal of the plane, pointing into the half-space that counts as positive. The zero
        vector gives all-zero heights, which every consumer here treats as a no-op.
    through : jnp.ndarray, shape ``(..., 3)``, optional
        A point the plane passes through. Omitted, the plane passes through the origin, which
        for a receiver-relative loop means through the receiver -- the case for every plane
        built from two directions. A plane that does **not** contain the receiver, such as a
        source's own supporting plane, must pass its offset here, or the heights silently
        answer for a parallel plane through the receiver instead.

    Returns
    -------
    jnp.ndarray, shape ``(..., n)``
    """
    offset = loop if through is None else loop - through[..., None, :]
    spread = plane_normal[..., None, :]
    height = dot(offset, spread)
    bound = _SLACK * dot(jnp.abs(offset), jnp.abs(spread))
    return jnp.where(jnp.abs(height) <= bound, 0.0, height)


def _candidates(loop, height):
    """Every vertex the clip could keep, and whether it keeps it.

    Walking the loop, each edge contributes two candidates: the vertex it starts at, kept when
    that vertex is inside, and the crossing point on the edge, kept when the edge changes side.
    Emitting both unconditionally and selecting with a mask is what keeps the shape static.

    The two tests are read off the same heights, so they cannot contradict each other: a vertex
    at height zero is kept, and neither edge meeting it is recorded as crossing, because the
    crossing would be that vertex and it has already been emitted.
    """
    width = loop.shape[-2]
    candidates, kept = [], []
    for k in range(width):
        following = (k + 1) % width
        here, there = height[..., k], height[..., following]
        candidates.append(loop[..., k, :])
        kept.append(here >= 0.0)
        gap = here - there
        # Guarded against a zero-length edge, which a fixed-width loop is full of by design.
        crossing = jnp.clip(here / jnp.where(gap == 0.0, 1.0, gap), 0.0, 1.0)
        candidates.append(
            loop[..., k, :] + crossing[..., None] * (loop[..., following, :] - loop[..., k, :])
        )
        kept.append(here * there < 0.0)
    return candidates, jnp.stack(kept, axis=-1)


def clip_to_halfspace(loop, height, width: int) -> jnp.ndarray:
    """Intersect a convex loop with the half-space its ``height`` is non-negative in.

    The survivors are compacted **in order** into exactly ``width`` slots, which is the part that
    has to be expressible without data-dependent shapes. It is done with a **rank**: the running
    count of survivors up to each candidate, so the ``j``-th output vertex is the candidate of
    rank ``j + 1``. Slots past the last survivor repeat it, giving zero-length edges that subtend
    no angle -- so a loop with fewer real vertices than ``width`` is represented exactly, not
    approximately.

    Parameters
    ----------
    loop : jnp.ndarray, shape ``(..., n, 3)``
        Directions from the receiver to the region's vertices, in order.
    height : jnp.ndarray, shape ``(..., n)``
        Signed height of each vertex above the clipping plane, from
        :func:`decidable_heights`. Taken rather than computed here so that a caller which also
        needs the heights -- to ask whether anything at all lies in front of the plane, say --
        asks the same question once and gets one answer.
    width : int
        Output vertex count. Must be at least ``n + 1`` for the result to be exact.

    Returns
    -------
    jnp.ndarray, shape ``(..., width, 3)``

    Notes
    -----
    All-zero heights are an exact no-op: every vertex is kept, no edge crosses. That is relied
    on -- a zero-length edge has no plane (:func:`spanning_plane` returns zeros for it), and the
    clip stage for it must do nothing.
    """
    candidates, survived = _candidates(loop, height)
    stacked = jnp.stack(candidates, axis=-2)
    rank = jnp.cumsum(survived, axis=-1)
    # Past the last survivor, keep asking for the last one -- which repeats it.
    wanted = jnp.minimum(jnp.arange(width), (rank[..., -1] - 1)[..., None])
    picked = survived[..., None, :] & (rank[..., None, :] - 1 == wanted[..., None])
    return jnp.einsum("...wk,...kd->...wd", picked.astype(stacked.dtype), stacked)
