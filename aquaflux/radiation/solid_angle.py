"""Exact closed-form solid angles of a triangle seen from a point.

Two different quantities live here, and keeping them apart is the whole reason this
module exists as a separate leaf.

:func:`solid_angle` is the plain solid angle ``Ω`` — the area of the triangle's image on
the unit sphere about the receiver. It is what a *fluence rate* needs: the total radiant
power arriving at a point from all directions, with no weighting by direction, because a
freely tumbling particle in a flow is not oriented.

:func:`projected_solid_angle` is ``Ω_proj = ∫ cos(θ) dω`` over the same triangle, with
``θ`` measured from a receiving **surface**'s normal. It is what an *irradiance* needs,
and what a surface-to-surface transfer factor needs, because light arriving obliquely at
a surface spreads over more of it.

These are different integrals of the same geometry, not two normalizations of one number:
no scalar converts one into the other, since the obliquity varies across the triangle. A
plain solid angle used where a projected one belongs overestimates the transfer, and over a
closed enclosure it does so by exactly a factor of two — every row of the resulting transfer
matrix sums to 2 rather than 1, at every refinement, which reads as a clean result rather
than as an error. Reach for :func:`solid_angle` when the receiver is a point in a volume and
for :func:`projected_solid_angle` when the receiver is a surface element.

Both are exact at every distance, including inside the near field where the elementary
``A·cos(θ)/r²`` point approximation diverges and the truth stays bounded. That approximation
is accurate to about 1% only beyond roughly eight times the square root of the emitting area
— on axis it is 331% high at a quarter of that distance, 24% high at one, and 1.6% high at
four — so in any geometry whose receivers sit within a few source widths of the source, which
is where the field is strongest, the closed forms below are not a refinement but a
correction.

Numerically, every angle here is formed with :func:`jax.numpy.arctan2` rather than by an
inverse cosine or by summing interior angles. Both alternatives lose accuracy exactly where
a refined mesh puts most of its work — see each function's own docstring.
"""

from __future__ import annotations

import jax.numpy as jnp

from aquaflux.radiation.clipping import clip_to_halfspace, decidable_heights
from aquaflux.vectors import dot

__all__ = ["projected_solid_angle", "signed_solid_angle", "solid_angle"]


def _unit(v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Normalize vectors along the last axis, reporting which ones had no direction.

    Returns the unit vectors and a mask that is ``True`` where the input had zero length —
    a receiver sitting exactly on a vertex, where no direction to that vertex exists. The
    zero rows are divided by one rather than by zero so the caller can substitute a value
    for them instead of propagating a NaN through an entire gather.
    """
    squared = dot(v, v)
    degenerate = squared == 0.0
    # The substitution happens *inside* the square root, not after it. Dividing by a
    # patched-up zero keeps the forward value finite, but reverse-mode differentiation still
    # visits the square root of zero, whose derivative is infinite, and a NaN from the branch
    # that was not taken propagates through the selection regardless. Feeding the root a one
    # in place of the zero keeps both the value and the gradient finite.
    length = jnp.sqrt(jnp.where(degenerate, 1.0, squared))[..., None]
    return v / length, degenerate


def solid_angle(point: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    """Solid angle ``Ω`` subtended by a triangle at a point, in steradians.

    Girard's theorem gives the area of a spherical triangle as its angular excess
    ``α + β + γ − π``. Evaluated that way it is inaccurate for small triangles, where the
    three angles each approach their own limit and the excess is the small difference of
    large quantities — which is the common case, since a refined facet far from a receiver
    is exactly a small spherical triangle. The equivalent tangent form of van Oosterom and
    Strackee (1983) computes the same area as a single ``arctan2`` of two quantities that
    are themselves small, and so loses nothing:

    .. math::
        \\Omega = 2 \\left| \\arctan2\\bigl(
            \\mathbf{a}\\cdot(\\mathbf{b}\\times\\mathbf{c}),\\;
            1 + \\mathbf{a}\\cdot\\mathbf{b} + \\mathbf{a}\\cdot\\mathbf{c}
              + \\mathbf{b}\\cdot\\mathbf{c} \\bigr) \\right|

    for unit vectors ``a``, ``b``, ``c`` from the point to the three vertices. The magnitude
    is taken because the sign of the numerator records only the order the vertices happen to
    be stored in, which for an imported surface file is whatever the file says; a triangle's
    orientation is a property of its normal, decided by the caller, not of its vertex order.

    Two configurations are worth knowing about, neither of them an error:

    * A point lying **in the triangle's own plane but outside it** returns zero, correctly:
      every direction to the triangle lies in one plane, a set of directions with no area.
    * A point lying **inside** the triangle returns ``2π``. That is the oriented spherical
      area — the image on the sphere is a great circle traversed once, an entire hemisphere
      by area. Physically a surface element contributes nothing to a receiver lying on it, so
      a caller assembling surface-to-surface transfer must exclude each element from its own
      row rather than relying on this function to return zero. Left in place it adds exactly
      one unit to every row sum.
    * A point coinciding with a **vertex** has no defined solid angle and returns zero, so
      that one degenerate pair cannot turn an entire gather into NaN.

    Parameters
    ----------
    point : jnp.ndarray
        Receiver positions, shape ``(..., 3)``.
    vertices : jnp.ndarray
        Triangle vertices, shape ``(..., 3, 3)`` — the second-to-last axis indexes the three
        vertices and the last holds the spatial components. Broadcast against ``point``, so a
        ``(n_points, 1, 3)`` point array against a ``(1, n_triangles, 3, 3)`` vertex array
        gives every pair.

    Returns
    -------
    jnp.ndarray
        Solid angle in steradians, shape ``(...)``, in ``[0, 4π]``.
    """
    return jnp.abs(signed_solid_angle(point, vertices))


def signed_solid_angle(point: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    """The same spherical area as :func:`solid_angle`, keeping the sign of the winding.

    ⚠️ **On a single triangle this sign means nothing**, which is exactly why
    :func:`solid_angle` discards it: it records the order the three vertices happen to be
    stored in, and for an imported surface file that is whatever the exporter wrote. Reach for
    this function only over a **consistently wound, closed** surface, where the signs of all
    its facets agree with one another and their sum therefore says something the individual
    terms do not.

    What it says is the *winding number*. Summed over such a surface and divided by ``4π``, the
    result is ``±1`` at a point enclosed by it and ``0`` at a point outside, with the overall
    sign fixed by whether the surface is wound outward or inward — so the enclosure test is on
    the magnitude. That is the one quantity here that distinguishes a cell of fluid from a cell
    embedded in metal, and it is why this function is exposed rather than kept private.

    Two properties make it the right test for surfaces that arrive from a file:

    * **It is exact, not asymptotic.** The measured winding number of a unit box is ``1`` to
      within about ``1e-14`` at a point a thousandth of a box-width from a wall, at every
      refinement — rounding in the sum, with no near-field regime where it degrades.
    * **It degrades continuously on an open surface** rather than answering confidently. A bare
      disc reads ``±0.45`` just off its face — nowhere near either ``0`` or ``±1`` — so a value
      in between is itself the diagnostic that the surface is not closed. A ray-parity test has
      no such reading: it returns a clean, wrong bit.

    Parameters
    ----------
    point : jnp.ndarray
        Receiver positions, shape ``(..., 3)``.
    vertices : jnp.ndarray
        Triangle vertices, shape ``(..., 3, 3)``, broadcast against ``point`` as in
        :func:`solid_angle`.

    Returns
    -------
    jnp.ndarray
        Signed solid angle in steradians, shape ``(...)``, in ``(-2π, 2π]``. Zero where the
        receiver coincides with a vertex.
    """
    direction, degenerate = _unit(vertices - point[..., None, :])
    a, b, c = direction[..., 0, :], direction[..., 1, :], direction[..., 2, :]
    numerator = dot(a, jnp.cross(b, c))
    denominator = 1.0 + dot(a, b) + dot(a, c) + dot(b, c)
    # A two-argument arctangent rather than a one-argument one: the quotient alone loses the
    # quadrant, and a triangle subtending most of a hemisphere has a denominator that changes
    # sign, so the single-argument form wraps to the wrong branch exactly where the term is
    # largest -- which is the near-field pair that matters most to the sum.
    omega = 2.0 * jnp.arctan2(numerator, denominator)
    return jnp.where(jnp.any(degenerate, axis=-1), 0.0, omega)


def _clip_to_front(relative: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
    """Clip a triangle to the half-space a receiving surface can see.

    ``relative`` holds the three vertices measured from the receiver, ``(..., 3, 3)``; the
    half-space kept is the one the receiver's ``normal`` points into. Clipping is not an
    optimization here, it is what makes :func:`projected_solid_angle` correct: the contour
    formula it uses is *signed*, so a triangle straddling the receiver's plane returns a
    partially cancelled value and one behind the plane returns a negative one. Summed over a
    closed enclosure those cancel to approximately zero, which looks like a broken mesh
    rather than a missing clip.

    Cutting a convex region with one half-space adds at most one vertex, so four slots hold a
    clipped triangle exactly; a triangle wholly behind the plane keeps nothing and collapses to
    the zero loop. Both the cut and the sign tests under it are shared with the occlusion clip
    -- see :mod:`aquaflux.radiation.clipping`, whose heights are filtered so that a vertex lying
    *in* the receiver's own tangent plane, which every triangle sharing that facet does, is
    judged by the exact zero it mathematically is rather than by the sign of its rounding.
    """
    return clip_to_halfspace(relative, decidable_heights(relative, normal), 4)


def projected_solid_angle(
    point: jnp.ndarray, normal: jnp.ndarray, vertices: jnp.ndarray
) -> jnp.ndarray:
    """Projected solid angle ``∫ cos(θ) dω`` of a triangle at an oriented surface point.

    This is the obliquity-weighted counterpart of :func:`solid_angle`, and the quantity a
    receiving *surface* responds to. Lambert's contour formula evaluates the integral exactly,
    as a sum over the edges of the polygon's image on the unit sphere:

    .. math::
        \\Omega_{\\mathrm{proj}} = \\tfrac{1}{2} \\left|
            \\sum_k \\theta_k \\, (\\mathbf{n} \\cdot \\mathbf{u}_k) \\right|

    where ``u_k`` is the unit normal of the plane through the receiver and edge ``k``, and
    ``θ_k`` is the angle that edge subtends. Dividing the result by π gives the fraction of a
    Lambertian receiver's hemisphere the triangle accounts for — the differential-to-finite
    transfer factor — and over a closed enclosure those fractions sum to one exactly, which is
    what bounds the spectral radius of a reflection system and makes it converge.

    ``θ_k`` is formed as ``arctan2(|u × u'|, u · u')`` and **not** as ``arccos(u · u')``. The
    inverse cosine is ill-conditioned as its argument approaches one, which is precisely the
    regime of a refined mesh, where consecutive edge directions are nearly parallel: summed
    over a closed enclosure the inverse-cosine form is wrong by about the square root of the
    machine epsilon, roughly 1e-8, and — the diagnostic detail — that error does **not**
    shrink under refinement, because it is a property of the formula and not an accumulation.
    The tangent form is exact to a few units in the last place at every refinement tested.

    The magnitude is taken for the same reason as in :func:`solid_angle`: after clipping, the
    projected solid angle is a non-negative quantity, and the sign of the contour sum records
    only the order the vertices are stored in.

    A receiver lying inside the triangle returns ``π``, a full hemisphere, by the same
    oriented-area convention described there; a caller assembling surface-to-surface transfer
    must exclude each element from its own row.

    Parameters
    ----------
    point : jnp.ndarray
        Receiver positions, shape ``(..., 3)``.
    normal : jnp.ndarray
        Unit outward normals of the receiving surface at ``point``, shape ``(..., 3)``. Only
        the half-space this points into contributes.
    vertices : jnp.ndarray
        Triangle vertices, shape ``(..., 3, 3)``, broadcast against ``point`` as in
        :func:`solid_angle`.

    Returns
    -------
    jnp.ndarray
        Projected solid angle, shape ``(...)``, in ``[0, π]``.
    """
    return jnp.abs(
        _signed_loop_solid_angle(normal, _clip_to_front(vertices - point[..., None, :], normal))
    )


def _signed_loop_solid_angle(normal: jnp.ndarray, loop: jnp.ndarray) -> jnp.ndarray:
    """The contour form of the projected solid angle of a closed loop of directions, **signed**.

    Split out of :func:`projected_solid_angle`, which is its magnitude, because two properties
    are lost the moment the magnitude is taken and both are load-bearing elsewhere:

    * **It is additive over a partition.** Cut a region into pieces and the signed values sum to
      the whole, exactly -- so a region can be built as a difference without ever constructing
      it. That is what lets an occluded source be evaluated as *whole minus covered*.
    * **It negates with the winding**, so a loop traversed the other way subtracts.

    Parameters
    ----------
    normal : jnp.ndarray, shape ``(..., 3)``
        Unit normal of the receiving surface. Only the half-space it points into contributes,
        and the loop is assumed already clipped to that half-space.
    loop : jnp.ndarray, shape ``(..., n, 3)``
        Directions from the receiver to each vertex in order, not necessarily unit length. A
        clipped loop repeats vertices wherever the clip dropped a candidate; such zero-length
        edges subtend no angle and contribute nothing, which is what makes a fixed-width loop a
        legal representation of a shorter one.

    Returns
    -------
    jnp.ndarray, shape ``(...)``
        The signed projected solid angle, in ``[-pi, pi]``.
    """
    direction, _ = _unit(loop)
    following = jnp.roll(direction, -1, axis=-2)
    edge_normal = jnp.cross(direction, following)
    # Every clipped loop carries repeated vertices wherever the clip dropped a candidate, so
    # zero-length edges are the normal case here rather than a degenerate one; the square
    # root is guarded the same way as in :func:`_unit`, and for the same gradient reason.
    span_squared = dot(edge_normal, edge_normal)
    flat = span_squared == 0.0
    span = jnp.sqrt(jnp.where(flat, 1.0, span_squared))
    axis = edge_normal / span[..., None]
    span = jnp.where(flat, 0.0, span)
    angle = jnp.arctan2(span, dot(direction, following))
    return 0.5 * jnp.sum(angle * dot(axis, normal[..., None, :]), axis=-1)


def _signed_loop_area(loop: jnp.ndarray) -> jnp.ndarray:
    """The plain solid angle of a closed, convex loop of directions, **signed**.

    The unprojected counterpart of :func:`_signed_loop_solid_angle`: the area of the loop's image
    on the unit sphere, with no weighting by any receiving normal. It is what a share of a
    *volume* receiver's view needs, because a point in the fluid has no normal to project onto.

    Evaluated as a fan of triangles from the loop's first vertex, each by the same closed form as
    :func:`signed_solid_angle`. That keeps the two properties the projected form is used for:

    * **It is additive over a partition**, and it negates with the winding, since each fan term
      is the signed area of its own triangle. So a covered region can still be subtracted from a
      whole without the visible region ever being constructed.
    * **Every term is a two-argument arctangent**, so a thin loop -- a sliver left by a clip -- is
      as well conditioned as a small triangle, rather than the small difference of interior
      angles an angle-excess formula would give.

    The loop must be convex and lie within an open hemisphere, which a triangle seen from any
    point off its plane does, and so does everything a clip of it by half-spaces through the
    receiver leaves: each fan triangle is then inside the loop and smaller than a hemisphere,
    where the closed form's branch is the right one.

    Parameters
    ----------
    loop : jnp.ndarray, shape ``(..., n, 3)``
        Directions from the receiver to each vertex in order, not necessarily unit length.
        Repeated vertices, as a fixed-width clipped loop carries, make degenerate fan triangles
        that contribute nothing; an emptied loop of zero vectors contributes nothing at all.

    Returns
    -------
    jnp.ndarray, shape ``(...)``
        The signed solid angle, in ``(-2 pi, 2 pi)``.
    """
    apex = jnp.broadcast_to(loop[..., :1, :], loop[..., 1:-1, :].shape)
    fan = jnp.stack([apex, loop[..., 1:-1, :], loop[..., 2:, :]], axis=-2)
    return jnp.sum(signed_solid_angle(jnp.zeros(3), fan), axis=-1)
