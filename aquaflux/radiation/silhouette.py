"""Exact occlusion by clipping a source's angular extent against a blocker's silhouette.

A ray test answers "is the line from this facet's centre to that one blocked" with a bit. A
facet only half in shadow gets the whole of one answer or the whole of the other, and refining
the mesh changes how *many* pairs straddle a shadow edge rather than how wrong a straddling one
is. Sampling more points -- on the receiver, or over the source -- only moves that boundary
around, because what is being sampled is a step function.

This module does not sample. It computes the fraction of a source's view that a blocker covers,
in closed form, by **clipping**: project the blocker's silhouette from the receiver, intersect it
with the source's angular extent, and evaluate the same contour integral the unoccluded transfer
already uses. For a receiver on a surface that is the projected solid angle ``∫ cos θ dω`` an
irradiance weights by; for a point in the fluid, which has no normal, it is the plain solid angle
``∫ dω`` a fluence rate weights by. The clip is the same for both, since it works in direction
space and never reads the normal; only the integral taken of its result changes. The treatment is the classical analytic form factor --
Nishita and Nakamae (1983), and Baum, Rushmeier and Winget (*Computer Graphics* 23(3), 1989),
who project blockers onto the source's supporting plane and clip away the occluded part.

**Four properties make it expressible as a traced program**, and each is load-bearing:

1. **The contour integral is signed and additive** (:func:`_signed_loop_solid_angle`, and
   :func:`_signed_loop_area` for the plain solid angle), so the *visible* region never has to be
   constructed. It is the whole minus the covered part, and the covered part is what clipping
   produces directly.
2. **The covered part is an intersection of convex regions**, hence convex, hence has a
   statically bounded vertex count -- which is what a shape-polymorphic compiler needs.
   Intersecting a convex region with a half-space adds **at most one** vertex, so a triangle
   clipped by a triangle's three edge planes runs 4, 5, 6, 7 wide rather than doubling to 48.
3. **Working in direction space** rather than on the source's supporting plane avoids the
   perspective divide, so a blocker straddling that plane raises no infinity.
4. **Every sign the clip reads is decidable**, so a configuration that is degenerate in exact
   arithmetic -- and in a meshed enclosure most are -- gets one answer however the batch is
   shaped and whether or not it is compiled. See :mod:`aquaflux.radiation.clipping`: without its
   two filters the answer depended on the compiler's choices, and a real occluder could vanish.

⚠️ **WHERE IT OVER-COUNTS, STATED EXACTLY.** Each blocker is clipped against the **source**, not
against what is still unblocked, so the method adds *areas*: two blockers covering the same cone
contribute ``0.54 + 0.54 = 1.08`` where a ray test's ``blocked or blocked`` is idempotent. A
tiling of one surface is safe -- its pieces do not overlap in projection, so per-triangle
fractions simply add -- and so is any sight line crossing front-facing geometry once, which
covers a sleeve, a wall, and the bent duct this package exists for, and a zero-thickness baffle
once it is declared two-sided (see
:class:`~aquaflux.radiation.self_occlusion.SilhouetteOcclusion`). It errs **dark**,
and only where two *separate* front-facing silhouettes overlap in angle: one sleeve behind
another in a multi-lamp bundle, or a serpentine channel seen across two walls. The exact repair
is to clip each blocker against the remaining unblocked region, depth-sorted and progressive --
the hidden-surface algorithm, whose per-pair vertex count is not static and so cannot be traced.
:func:`covered_fraction` therefore reports a **count** alongside the fraction, and a count of one
proves that pair exact.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from aquaflux.radiation.clipping import (
    clip_to_halfspace,
    decidable_heights,
    spanning_plane,
)
from aquaflux.radiation.solid_angle import _signed_loop_area, _signed_loop_solid_angle
from aquaflux.vectors import dot

__all__ = [
    "SourceView",
    "angular_cone",
    "beyond_source_plane",
    "cones_may_overlap",
    "covered_by",
    "covered_fraction",
    "covers_nothing",
    "enclosing_cone",
    "may_occlude",
    "source_plane",
    "source_view",
]

#: Below this, a cone's axis or half-angle is not a usable bound and the cone is flagged.
_CONE_FLOOR = 1e-12

#: Slack on the cone-disjoint comparison, so a pair that is borderline in floating point is
#: kept rather than dropped. The cull may keep too much at no cost but must never drop.
_CONE_SLACK = 1e-9

#: Relative floor for the degeneracy tests below: a quantity smaller than this times its own
#: natural scale is treated as the exact zero it is mathematically. Degenerate configurations
#: are the NORM in a meshed enclosure -- two triangles of one flat wall are exactly coplanar,
#: and a receiver sits in the plane of every triangle sharing its facet -- so these are the
#: common case and not an edge case.
_DEGENERATE = 1e-12

#: Below this many steradians, a source's projected solid angle is the rounding dust of the
#: contour sum that produced it rather than a quantity: at most eight edge terms, each at most
#: pi, so a few parts in 1e15 of pi. A source at that level subtends nothing -- it is being seen
#: edge-on -- and no fraction of it can be hidden. Far above the dust and far below any solid
#: angle that carries light.
_EXTENT_FLOOR = 1e-12

#: Below this share of a source, a blocker is not counted as having covered anything. It exists
#: for the COUNT and not for the fraction, which is summed whatever its size: a near-tangent
#: blocker legitimately covers a few parts in a quadrillion, and counting those as contributors
#: made the overlap report fire on three quarters of all pairs and mean nothing.
_COVERAGE_FLOOR = 1e-12

#: Loop widths through the clip: the source's three corners clipped to the front half-space (or,
#: for a receiver in the volume, padded to the same width), then one edge plane of the blocker at
#: a time. Each intersection with a half-space adds at most one
#: vertex, which is the whole reason these are small numbers and not ``6, 12, 24, 48, 96``. There
#: are four edge planes rather than three because the blocker has itself been cut to the near side
#: of the source's plane first, which can turn its triangle into a quadrilateral.
_WIDTHS = (4, 5, 6, 7, 8)

#: Slots for the blocker after that cut: a triangle clipped by one half-space has at most four
#: vertices.
_BLOCKER_WIDTH = 4


class SourceView(eqx.Module):
    """Everything about a (receiver, source) pair that does not depend on the blocker.

    Hoisted out of :func:`covered_by` because a build evaluates many blockers against each pair,
    and recomputing the source's clipped loop and total solid angle for every one of them is the
    same work done over and over. It is a value object rather than four loose arrays for the
    usual reason: the four travel together, and a caller that assembled three of them correctly
    and the fourth from a different pair would get a plausible wrong answer.

    Attributes
    ----------
    loop : jnp.ndarray, shape ``(..., 4, 3)``
        The source's corners as directions from the receiver, clipped to the half-space the
        receiver faces -- or, for a receiver in the volume, which faces every way, unclipped and
        padded with a repeated corner.
    whole : jnp.ndarray, shape ``(...)``
        The signed measure of that loop -- projected solid angle on a surface, plain solid angle
        in the volume -- which is the denominator of the fraction.
    support : jnp.ndarray, shape ``(..., 3)``
        The source's supporting plane, oriented towards the receiver.
    through : jnp.ndarray, shape ``(..., 3)``
        A point on that plane, relative to the receiver.
    """

    loop: jnp.ndarray
    whole: jnp.ndarray
    support: jnp.ndarray
    through: jnp.ndarray

    def subtends(self):
        """Whether the source's measure is a quantity rather than the rounding dust of a zero.

        Below :data:`_EXTENT_FLOOR` the source is being seen edge-on: it subtends nothing, and no
        fraction of it can be hidden. Asked here, once, because both the cull and the clip must
        give the same answer to it.

        Returns
        -------
        jnp.ndarray of bool, shape ``(...)``
        """
        return jnp.abs(self.whole) > _EXTENT_FLOOR

    def take(self, index):
        """The views of the pairs ``index`` picks along the leading axis.

        Parameters
        ----------
        index : array of int, shape ``(k,)``

        Returns
        -------
        SourceView
            With leading dimension ``k``.
        """
        return SourceView(
            loop=self.loop[index],
            whole=self.whole[index],
            support=self.support[index],
            through=self.through[index],
        )


def _measure(receiver_normal, loop):
    """The signed measure a share is taken of: projected on a surface, plain in the volume.

    A receiver on a surface responds to ``∫ cos θ dω`` about its normal, a point in the fluid to
    ``∫ dω``; the two are different integrals over the same clipped region, and ``None`` for the
    normal is how a volume receiver says which one it wants. Choosing here, once, is what keeps
    the whole, the covered part and the blocker's extent in the same measure -- a fraction of one
    taken against the other would be a number with no meaning.
    """
    if receiver_normal is None:
        return _signed_loop_area(loop)
    return _signed_loop_solid_angle(receiver_normal, loop)


def _in_view(relative, receiver_normal, width: int):
    """A loop cut to what the receiver can see, in ``width`` slots.

    A surface receiver sees only the half-space its normal points into, so the loop is clipped to
    it. A volume receiver sees in every direction, so nothing is cut and the loop is only padded
    with its last vertex -- a zero-length edge, which contributes nothing to either measure.
    """
    if receiver_normal is None:
        pad = width - relative.shape[-2]
        return jnp.concatenate([relative, *[relative[..., -1:, :]] * pad], axis=-2)
    return clip_to_halfspace(relative, decidable_heights(relative, receiver_normal), width)


def source_view(receiver, receiver_normal, source) -> SourceView:
    """Build the per-pair quantities :func:`covered_by` needs.

    ``receiver_normal`` is ``None`` for a receiver in the volume, which takes its share of the
    plain solid angle rather than the projected one.
    """
    receiver = jnp.asarray(receiver, dtype=float)
    if receiver_normal is not None:
        receiver_normal = jnp.asarray(receiver_normal, dtype=float)
    to_source = jnp.asarray(source, dtype=float) - receiver[..., None, :]
    loop = _in_view(to_source, receiver_normal, _WIDTHS[0])
    support, through = source_plane(receiver, source)
    return SourceView(
        loop=loop,
        whole=_measure(receiver_normal, loop),
        support=support,
        through=through,
    )


def covered_by(view: SourceView, receiver_normal, to_blocker):
    """Fraction of a source's view one blocker covers, given the pair's view.

    The share is of the projected solid angle for a receiver on a surface and of the plain solid
    angle for one in the volume -- the measure each gathers its light in. The clip itself is the
    same for both: it works in direction space and never reads the normal.

    Parameters
    ----------
    view : SourceView
        From :func:`source_view`, for this (receiver, source) pair, built with the same
        ``receiver_normal``.
    receiver_normal : jnp.ndarray, shape ``(..., 3)``, or None
        Unit normal at the receiver, or ``None`` for a receiver in the volume.
    to_blocker : jnp.ndarray, shape ``(..., 3, 3)``
        The blocker's corners **relative to the receiver**.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        The covered fraction in ``[0, 1]``, and whether this blocker covers anything at all --
        which a caller sums to detect the overlapping-silhouette case where fractions may be
        added twice.
    """
    if receiver_normal is not None:
        receiver_normal = jnp.asarray(receiver_normal, dtype=float)

    # ⚠️ THE CLIP LIVES IN DIRECTION SPACE AND HAS NO NOTION OF DEPTH, so the part of a blocker
    # lying BEYOND the source must be removed before its silhouette is taken -- otherwise a
    # triangle poking through the source's plane occludes with the whole of itself. Measured on
    # a straddling blocker: 1.0000 against a sampled 0.6396. For a planar source the cut is
    # exact: a point occludes a direction precisely when it is on the receiver's side of the
    # source's supporting plane.
    height = _depth_heights(view, to_blocker)
    near = clip_to_halfspace(to_blocker, height, _BLOCKER_WIDTH)
    in_front = _in_front(to_blocker, height)
    volume, edge_on = _orientation(to_blocker)
    facing = jnp.sign(volume)[..., None]
    # A four-wide loop holding a triangle carries a repeated slot, and the cut can leave a second
    # one; the edge through a repeat has no plane, and `spanning_plane` returns the exact zero
    # that makes its clip stage a no-op rather than the last-bit noise a cross product of a
    # direction with itself would give.
    corner = [near[..., k, :] for k in range(_BLOCKER_WIDTH)]
    edges = [
        spanning_plane(corner[k], corner[(k + 1) % _BLOCKER_WIDTH]) * facing
        for k in range(_BLOCKER_WIDTH)
    ]

    loop = view.loop
    for width, plane in zip(_WIDTHS[1:], edges, strict=True):
        loop = clip_to_halfspace(loop, decidable_heights(loop, plane), width)
    covered = _measure(receiver_normal, loop)

    # ⚠️ THE COVERED REGION CANNOT EXCEED THE BLOCKER'S OWN EXTENT, AND CLAMPING TO IT IS WHAT
    # MAKES A DEGENERATE CUT SAFE. A zero clipping plane is a no-op by construction -- correct
    # for the repeated slot a triangular blocker leaves in a four-wide loop, and catastrophic
    # when the cut degenerates the blocker ENTIRELY: every edge plane is then zero, every clip
    # does nothing, and the source comes back reported as wholly covered by a blocker that
    # covers none of it. Measured on a reactor facet whose source lay in the very plane the
    # blocker was cut to: 1.0000 against a sampled 0.0. The bound holds geometrically for every
    # pair, so clamping is a no-op wherever the clip was already right.
    extent = jnp.abs(_measure(receiver_normal, _in_view(near, receiver_normal, _BLOCKER_WIDTH + 1)))
    # ⚠️ A SOURCE THAT SUBTENDS NOTHING CANNOT BE HIDDEN, AND THE RATIO BELOW CANNOT SAY SO.
    # A source coplanar with the receiver's own facet is seen exactly edge-on, so its projected
    # solid angle is mathematically zero and computationally a few parts in 1e16 of dust -- and
    # the covered part is dust of the same size, so their quotient is an arbitrary number
    # between zero and one rather than the zero it should be. In a meshed enclosure this is not
    # a corner case: every other triangle of the receiver's own wall is coplanar with it.
    # Measured without this guard on a sleeved reactor: 13 pairs of coplanar sleeve triangles
    # read anywhere from 0.06 to a fully hidden 1.0, where the truth is that they exchange no
    # light at all.
    subtends = view.subtends()
    fraction = jnp.minimum(jnp.abs(covered), extent) / jnp.where(subtends, jnp.abs(view.whole), 1.0)
    # Nothing of the blocker strictly in front of the source, or a blocker seen exactly edge-on:
    # either way it covers nothing, and saying so here is what keeps the degenerate cases out of
    # the sign tests above rather than at their mercy.
    fraction = jnp.where(in_front & ~edge_on & subtends, fraction, 0.0)
    return jnp.clip(fraction, 0.0, 1.0), fraction > _COVERAGE_FLOOR


def _depth_heights(view: SourceView, to_blocker):
    """Heights of a blocker's corners above the source's supporting plane, towards the receiver.

    Filtered (:func:`~aquaflux.radiation.clipping.decidable_heights`), so a corner lying in the
    source's plane -- every shared vertex of a meshed wall -- reads an exact zero.
    """
    return decidable_heights(to_blocker, view.support, through=view.through)


def _in_front(to_blocker, height):
    """Whether any of a blocker lies strictly in front of the source's plane.

    ⚠️ A BLOCKER COPLANAR WITH THE SOURCE OCCLUDES NOTHING, AND IN A MESHED ENCLOSURE THAT IS THE
    COMMON CASE -- every other triangle of the same flat wall. Every height is then a
    mathematical zero, which the filtered heights report as an exact zero rather than as noise of
    arbitrary sign; this reads that report. Measured before the two together: 119 of 2304 pairs
    of a closed box changed with the chunk size alone, one of them by the whole of its value.
    """
    scale = jnp.max(jnp.linalg.norm(to_blocker, axis=-1), axis=-1)
    return jnp.max(height, axis=-1) > _DEGENERATE * scale


def _orientation(triangle):
    """A triangle's winding as seen from the receiver, and whether it is seen edge-on.

    The winding decides which side of each edge plane is inside, so it is read off rather than
    assumed: an imported file's winding is whatever its exporter wrote, and a triangle's apparent
    winding flips as the receiver crosses its plane in any case. Taken from the uncut triangle,
    whose triple product is well conditioned.

    The triple product is six times the volume of the tetrahedron on the receiver and the
    triangle, so it vanishes when the receiver lies in the triangle's plane -- the common case,
    for every triangle sharing the receiver's own facet. Its sign is then zero, every edge plane
    is zero and every clip a no-op, so a triangle seen exactly edge-on would read as covering the
    whole source. It is compared against its own scale rather than against zero.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        The triple product, and whether it is indistinguishable from zero, both shape ``(...)``.
    """
    volume = dot(triangle[..., 0, :], jnp.cross(triangle[..., 1, :], triangle[..., 2, :]))
    edge_on = jnp.abs(volume) <= _DEGENERATE * jnp.prod(jnp.linalg.norm(triangle, axis=-1), axis=-1)
    return volume, edge_on


def _outside_an_edge_plane(triangle, other):
    """Whether every corner of ``other`` lies strictly outside one edge plane of ``triangle``.

    The directions a triangle occupies, seen from the receiver, are the intersection of the three
    half-spaces through the receiver bounded by its edge planes, each oriented inwards by the
    triangle's winding. Corners of the other triangle all strictly outside one of them put the
    whole of its directions -- every positive combination of its corners -- in the open half-space
    beyond that plane, so the two share no direction and no clip of one by the other can keep
    anything. A separating-axis test restricted to the face planes: conservative, because it can
    miss a separation, never invent one.

    Heights are filtered and the comparison is strict, so a corner on the plane -- a shared edge --
    never separates, and a triangle seen edge-on, whose planes mean nothing, separates nothing.
    """
    volume, edge_on = _orientation(triangle)
    facing = jnp.sign(volume)[..., None]
    outside = [
        jnp.all(
            decidable_heights(
                other, spanning_plane(triangle[..., k, :], triangle[..., (k + 1) % 3, :]) * facing
            )
            < 0.0,
            axis=-1,
        )
        for k in range(3)
    ]
    return ~edge_on & (outside[0] | outside[1] | outside[2])


def covers_nothing(view: SourceView, to_source, to_blocker):
    """Certainly: this blocker covers none of this source, so clipping the pair would give zero.

    The **second stage of the cull**, after :func:`may_occlude`'s bounding cones and planes, and
    run on the pairs those keep: it answers exactly where they only bound, so it rejects most of
    what they let through -- on a sleeved box, nine pairs in ten of the cone cull's survivors cover
    nothing -- at a fraction of what clipping them costs. Four reasons, each **exact** in the sense
    that the clip would have returned zero for the pair:

    * the source subtends nothing (it is seen edge-on), which :func:`covered_by` maps to zero;
    * nothing of the blocker lies strictly in front of the source's plane -- every neighbour of
      the source on its own flat wall, and every blocker merely touching that plane, which the
      cone cull's raw-sign plane test lets through;
    * the blocker is seen edge-on, which :func:`covered_by` also maps to zero;
    * a face plane of either triangle separates their directions.

    The first three read the very predicates :func:`covered_by` zeroes a pair by, so they cannot
    disagree with it except within rounding of their own thresholds, where the coverage is a few
    parts in ``1e12``. The fourth reads filtered heights with a strict sign, so a configuration
    degenerate in exact arithmetic -- a shared edge, a coplanar pair -- is kept for the clip to
    decide.

    Parameters
    ----------
    view : SourceView
        From :func:`source_view`, for this (receiver, source) pair.
    to_source : jnp.ndarray, shape ``(..., 3, 3)``
        The source's corners relative to the receiver, uncut. The directions the clip measures
        are a subset of these, so separating from the uncut triangle is conservative.
    to_blocker : jnp.ndarray, shape ``(..., 3, 3)``
        The blocker's corners relative to the receiver.

    Returns
    -------
    jnp.ndarray of bool, shape ``(...)``
        ``True`` where the pair certainly covers nothing and need not be clipped.
    """
    _, edge_on = _orientation(to_blocker)
    return (
        ~view.subtends()
        | ~_in_front(to_blocker, _depth_heights(view, to_blocker))
        | edge_on
        | _outside_an_edge_plane(to_source, to_blocker)
        | _outside_an_edge_plane(to_blocker, to_source)
    )


def covered_fraction(receiver, receiver_normal, source, blocker):
    """Fraction of a source triangle's view that a blocker triangle covers.

    Composes :func:`source_view` and :func:`covered_by` for one-off use and for tests. A build
    should call the two separately, so the per-pair half is not rebuilt for every blocker.

    Parameters
    ----------
    receiver : jnp.ndarray, shape ``(..., 3)``
        Where the light is being gathered.
    receiver_normal : jnp.ndarray, shape ``(..., 3)``, or None
        Unit normal there. Only the half-space it faces contributes, which is the same clamp the
        unoccluded transfer applies, and the fraction is of the projected solid angle. ``None``
        for a receiver in the volume, whose fraction is of the plain solid angle.
    source : jnp.ndarray, shape ``(..., 3, 3)``
        The emitting triangle's corners.
    blocker : jnp.ndarray, shape ``(..., 3, 3)``
        The occluding triangle's corners.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray)
        The covered fraction in ``[0, 1]`` and whether it is non-zero, both shape ``(...)``.
    """
    receiver = jnp.asarray(receiver, dtype=float)
    source = jnp.asarray(source, dtype=float)
    blocker = jnp.asarray(blocker, dtype=float)
    # Broadcast to one batch shape before building the view: the per-pair half and the
    # per-triple half must agree on it, and a caller passing one source against many blockers is
    # the ordinary way to use this.
    batch = jnp.broadcast_shapes(receiver.shape[:-1], source.shape[:-2], blocker.shape[:-2])
    receiver = jnp.broadcast_to(receiver, (*batch, 3))
    source = jnp.broadcast_to(source, (*batch, 3, 3))
    blocker = jnp.broadcast_to(blocker, (*batch, 3, 3))
    if receiver_normal is not None:
        receiver_normal = jnp.broadcast_to(jnp.asarray(receiver_normal, dtype=float), (*batch, 3))
    view = source_view(receiver, receiver_normal, source)
    return covered_by(view, receiver_normal, blocker - receiver[..., None, :])


def angular_cone(relative):
    """A bounding cone, from the receiver, of the directions a triangle occupies.

    The cull this feeds has to answer a **direction-space** question -- can this blocker's
    silhouette overlap this source's -- and answering it with planes through the receiver is
    what made the first version of the cull 66x looser than necessary: such a plane cuts the
    whole enclosure, so a triangle far from the narrow pyramid between a pair still straddles it
    and survives. A cone is the shape of the question.

    The cone is the spherical cap about the mean of the vertex directions that reaches the
    furthest of them. For a triangle spanning less than a hemisphere this contains the whole
    projected triangle, because a cap of half-angle below a right angle is geodesically convex
    and therefore contains the spherical convex hull of the three vertices -- which is exactly
    what the triangle projects to.

    Parameters
    ----------
    relative : jnp.ndarray, shape ``(..., 3, 3)``
        Triangle corners **relative to the receiver**.

    Returns
    -------
    tuple of (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray)
        Unit axis ``(..., 3)``; cosine and sine of the half-angle ``(...)``; and a boolean
        ``(...)`` marking cones that do **not** bound anything usable -- a degenerate axis, or a
        cap reaching a right angle or beyond, which happens when the receiver lies near the
        triangle's own plane. Those are flagged rather than fixed, and a caller must treat a
        flagged cone as overlapping everything, or the cull stops being conservative.
    """
    relative = jnp.asarray(relative, dtype=float)
    length = jnp.linalg.norm(relative, axis=-1, keepdims=True)
    unit = relative / jnp.where(length == 0.0, 1.0, length)

    total = jnp.sum(unit, axis=-2)
    reach = jnp.linalg.norm(total, axis=-1)
    axis = total / jnp.where(reach == 0.0, 1.0, reach)[..., None]
    cos_half = jnp.clip(jnp.min(dot(unit, axis[..., None, :]), axis=-1), -1.0, 1.0)

    # A vertex at the receiver, a vanishing mean direction, or a cap at or past a right angle:
    # in each the cap is not a bound on the triangle, so the cone is unusable.
    unusable = (
        (reach <= _CONE_FLOOR) | (cos_half <= _CONE_FLOOR) | jnp.any(length[..., 0] == 0.0, axis=-1)
    )
    return axis, cos_half, jnp.sqrt(jnp.clip(1.0 - cos_half**2, 0.0, 1.0)), unusable


def cones_may_overlap(source_cone, blocker_cone):
    """Conservatively: can these two cones share a direction?

    Two caps are disjoint exactly when the angle between their axes exceeds the sum of their
    half-angles. Compared through cosines so no arc-cosine is needed: with both half-angles
    below a right angle their sum is at most ``pi``, where the cosine is still monotone, so the
    comparison is valid without a wrap check.

    ⚠️ **An unusable cone overlaps everything.** Dropping that rule is what made the first
    version of this cull discard 4,941 genuine occluders on a 480-facet reactor -- silently,
    since a missing occluder is a slightly brighter answer and not an error.
    """
    source_axis, source_cos, source_sin, source_bad = source_cone
    blocker_axis, blocker_cos, blocker_sin, blocker_bad = blocker_cone
    between = dot(source_axis, blocker_axis)
    limit = source_cos * blocker_cos - source_sin * blocker_sin
    return source_bad | blocker_bad | (between >= limit - _CONE_SLACK)


def enclosing_cone(cone, members, eligible):
    """A cone containing the cones of every eligible member of each group, and which groups have one.

    For a cull over groups of triangles: two groups whose enclosing cones cannot overlap hold no
    pair of members whose cones can, so every pair between them is rejected by one test. The
    enclosing cap is centred on the mean of the members' axes and reaches, for each member, the
    angle to its axis plus its own half-angle -- by the triangle inequality on the sphere, every
    direction in a member's cap is then inside it. Angles are taken with ``arctan2``, which stays
    accurate for the near-parallel axes a compact group has.

    ⚠️ **An unusable member makes its group unusable**, and so does an enclosing cap reaching a
    right angle -- the same rule as for a single cone, and for the same reason: a flagged cone
    overlaps everything, or the cull stops being conservative.

    Parameters
    ----------
    cone : tuple of jnp.ndarray
        Per-triangle cones from :func:`angular_cone`, each of leading dimension ``n``.
    members : jnp.ndarray of int, shape ``(m, k)``
        Triangle indices of each of ``m`` groups, ``-1`` for an empty slot.
    eligible : jnp.ndarray of bool, shape ``(n,)``
        Which triangles take part; the rest are left out of every group's bound.

    Returns
    -------
    tuple of (tuple of jnp.ndarray, jnp.ndarray)
        The enclosing cones in the form :func:`angular_cone` returns, leading dimension ``m``,
        and a boolean ``(m,)`` marking the groups with any eligible member at all -- a group
        without one has no pairs to keep and its cone means nothing.
    """
    axis, cos_half, sin_half, unusable = cone
    slot = jnp.maximum(members, 0)
    valid = (members >= 0) & jnp.take(eligible, slot)
    member_axis = jnp.take(axis, slot, axis=0)
    total = jnp.sum(jnp.where(valid[..., None], member_axis, 0.0), axis=-2)
    reach = jnp.linalg.norm(total, axis=-1)
    centre = total / jnp.where(reach == 0.0, 1.0, reach)[..., None]
    apart = jnp.arctan2(
        jnp.linalg.norm(jnp.cross(centre[..., None, :], member_axis), axis=-1),
        dot(centre[..., None, :], member_axis),
    )
    own = jnp.arctan2(jnp.take(sin_half, slot), jnp.take(cos_half, slot))
    half = jnp.max(jnp.where(valid, apart + own, -jnp.inf), axis=-1) + _CONE_SLACK
    bad = (
        jnp.any(valid & jnp.take(unusable, slot), axis=-1)
        | (reach <= _CONE_FLOOR)
        | (half >= 0.5 * jnp.pi - _CONE_FLOOR)
    )
    half = jnp.clip(half, 0.0, 0.5 * jnp.pi)
    return (centre, jnp.cos(half), jnp.sin(half), bad), jnp.any(valid, axis=-1)


def source_plane(receiver, source):
    """The source's supporting plane, oriented towards the receiver, and a point on it.

    Depth is the one thing a direction-space cull cannot see: a blocker beyond the source covers
    the same directions and must be rejected on position. Returned as a plane and an offset
    because, unlike everything else here, it does **not** pass through the receiver.
    """
    receiver = jnp.asarray(receiver, dtype=float)
    to_source = jnp.asarray(source, dtype=float) - receiver[..., None, :]
    support = jnp.cross(
        to_source[..., 1, :] - to_source[..., 0, :], to_source[..., 2, :] - to_source[..., 0, :]
    )
    support = support * jnp.sign(dot(support, -to_source[..., 0, :]))[..., None]
    # Normalized, so a height taken against it is a DISTANCE and can be compared to a length
    # scale. Left unnormalized, the only available tolerance is absolute, and it would mean
    # different things on a reactor in metres and a lamp in millimetres.
    length = jnp.linalg.norm(support, axis=-1, keepdims=True)
    return support / jnp.where(length == 0.0, 1.0, length), to_source[..., 0, :]


def beyond_source_plane(to_blocker, support, through):
    """Whether every corner of a blocker lies further away than the source's plane."""
    return jnp.all(dot(to_blocker - through[..., None, :], support[..., None, :]) < 0.0, axis=-1)


def may_occlude(receiver, receiver_normal, source, blocker):
    """Conservatively: could this blocker cover any of this source, seen from this receiver?

    **Conservative in one direction only.** A ``True`` may be wrong -- the blocker may turn out
    to cover nothing -- but a ``False`` never is, so nothing that could occlude is dropped. That
    asymmetry is what makes it legal to run the expensive clip on the survivors alone, and it is
    the property to protect in any change here: a cull that drops an occluder produces a
    slightly brighter field and no error of any kind.

    Three independent reasons a blocker cannot matter, in the space each belongs to: its
    directions miss the source's (a cone test), it lies beyond the source (a plane test), or it
    lies behind the receiver (a plane test). The last needs the receiver to face one way, so a
    receiver in the volume -- ``receiver_normal`` of ``None`` -- is culled by the first two.

    This composes the pieces for one-off use and for tests. A build should call them separately,
    because the cones are per ``(receiver, triangle)`` and the plane is per ``(receiver,
    source)`` -- computing either per triple throws away the saving they exist for.
    """
    receiver = jnp.asarray(receiver, dtype=float)
    to_source = jnp.asarray(source, dtype=float) - receiver[..., None, :]
    to_blocker = jnp.asarray(blocker, dtype=float) - receiver[..., None, :]

    support, through = source_plane(receiver, source)
    keep = cones_may_overlap(angular_cone(to_source), angular_cone(to_blocker)) & (
        ~beyond_source_plane(to_blocker, support, through)
    )
    if receiver_normal is None:
        return keep
    behind = jnp.all(
        dot(to_blocker, jnp.asarray(receiver_normal, dtype=float)[..., None, :]) < 0.0, axis=-1
    )
    return keep & ~behind
