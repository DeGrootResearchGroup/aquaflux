"""Solid bodies, and the two questions asked of them: does a segment pass through, is a point inside.

A :class:`Body` answers those two questions and nothing else. It says nothing about what the
body *does* to whatever crosses it — how much light a sleeve lets through, say — because that is
the consumer's business and usually a quantity it wants to vary and differentiate, while the
geometry is fixed. Keeping them apart is what lets the expensive, discontinuous half (which
segments cross which bodies) be computed once and the smooth half stay live.

**The bodies here are analytic primitives, and that is a structural choice rather than a cheaper
one.** A cylinder answers "does this segment hit me" with a formula: a few dozen arithmetic
operations, the same ones for every ray, no branching and nothing to search. A triangulated
surface turns the same question into a search over however many triangles describe it, and every
acceleration for a search is a walk with a data-dependent stopping rule — which cannot be traced
without pricing the work it skips at the same rate as the work it does. So a primitive does not
merely make each test cheaper: it removes the search, leaving one array expression that fuses and
compiles. A primitive is also *exact* where a triangulation is not — a round pipe described as a
fifteen-sided inscribed polygon is wrong by the sliver between the polygon and the circle — and it
carries real parameters, a radius and a length and a position, where a frozen triangle soup
carries none.

**Three layers, each doing one thing.**

:class:`Body` is the contract a consumer sees: :meth:`~Body.blocks` and :meth:`~Body.contains`.
Anything answering those two questions can stand in the way, including a body answered on the
host from triangles rather than by a formula.

:class:`Solid` narrows it to a body whose inside, seen along a line, is a **bounded union of
intervals** in the line's parameter. That is the property every operation here rests on: blocking
is then "does any interval overlap the segment", and combining bodies is interval arithmetic
rather than a new intersection routine per shape.

:class:`ConvexSolid` narrows it further to the case where that union is a **single** interval,
which is what convexity means for a line. A convex body is an intersection of simple inequalities
— a plane, a tube, a ball, a taper — and the primitives below are each written as the handful of
inequalities that bound them. That is why there is one implementation of the plane test rather
than one per body that happens to have a flat end, and why :meth:`~Solid.contains` and the
interval can never disagree about where a body's surface is: they are two readings of the same
inequalities.

**Composition is CSG.** :class:`Union`, :class:`Intersection` and :class:`Difference` build a
vessel out of primitives instead of hand-deriving it, and each is exact — interval arithmetic,
not a sampled approximation. :class:`Outside` is the one that matters most in practice: a reactor
wall is most naturally described by the *fluid it holds*, and a segment is then clear exactly when
the fluid's regions cover it end to end.

**A body can also vouch for a whole region of space at once** (:meth:`~Body.clearance`). A
segment test answers for one segment; a consumer testing millions of segments between two
compact groups of points -- a block of mesh cells and a patch of a lamp, say -- would rather ask
once whether the body can come between the groups at all. Every segment between two points of a
set lies in the set's convex hull, so a body that provably misses the hull blocks none of them.
The evidence is a handful of *witness* functions whose negative region lies in a convex set
disjoint from the body -- a separating half-space, or a convex region of fluid -- and it composes
by taking maxima, which is what lets a consumer summarize each group once rather than each pair.

⚠️ **Do not represent one surface twice.** A body whose surface is also, elsewhere, a set of
triangles the segments start from — a lamp sleeve that is also the emitting surface of a
radiation model — must not be passed as a body as well: every segment would then leave a point
lying exactly on a body's surface, the degenerate configuration ray-tracing tools warn about.
"""

from __future__ import annotations

import abc
import functools
import numbers
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, norm_squared

__all__ = [
    "Body",
    "Box",
    "Cone",
    "ConvexSolid",
    "Cylinder",
    "Difference",
    "HalfSpace",
    "Intersection",
    "Outside",
    "Solid",
    "Sphere",
    "Union",
]


class Body(eqx.Module):
    """A solid body, and the test for whether a segment passes through it."""

    #: Whether this body's answers are a pure array expression, so a caller may compile them.
    #: **Declared rather than assumed, because the alternative is a crash.** A body built from
    #: triangles answers on the host, walking a spatial structure and dropping rays as they are
    #: settled — a search whose whole value is the work it skips, which tracing would price at
    #: the same rate as the work it does. Handed a traced argument it raises rather than being
    #: slow. Primitives are the opposite case and want compiling badly: a body assembled from
    #: several inequalities forms several receiver-by-facet intermediates, and eagerly every one
    #: of them is written to memory. Since the two kinds are meant to stand in one scene
    #: together, whoever builds the mask has to be told which it is holding. The default is the
    #: safe answer, so a bespoke body is merely not compiled rather than broken.
    traceable: ClassVar[bool] = False

    @abc.abstractmethod
    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        """Whether the segment from ``origin`` to ``target`` meets this body.

        Parameters
        ----------
        origin, target : jnp.ndarray
            Segment endpoints, shape ``(..., 3)``, broadcast against each other. By convention
            ``origin`` is the source and ``target`` the receiver.
        min_distance : jnp.ndarray
            How far from ``origin`` a hit must be before it counts, in length units, shape
            ``(...)``. A facet lying on or near a body would otherwise shadow itself: the
            intersection sits at essentially zero distance, and whether it is found at all comes
            down to rounding.

        Returns
        -------
        jnp.ndarray of bool
            ``True`` where the body lies across the segment.
        """

    @abc.abstractmethod
    def contains(self, position) -> jnp.ndarray:
        """Whether each position lies inside the solid.

        Used at build time to refuse a scene in which a facet or a receiver has been placed
        inside a solid body. Such a point is not merely shadowed, it is embedded in metal or in
        quartz, and every answer computed there is meaningless rather than small.

        Parameters
        ----------
        position : jnp.ndarray, shape ``(..., 3)``

        Returns
        -------
        jnp.ndarray of bool, shape ``(...)``
        """

    def clearance(self, position) -> jnp.ndarray:
        """Witnesses that a convex hull of positions misses this body.

        Each column ``i`` is a function ``w_i`` with one guarantee: **if every position of a set
        reads** ``w_i < 0`` **for the same** ``i``, **no point of the set's convex hull lies in the
        body**, so no segment between two of those positions meets it. A column that reads
        ``>= 0`` for some position says nothing -- the hull may or may not miss the body.

        That guarantee is what makes the answer cheap to use in bulk. A set's summary is the
        largest value each column takes over its positions, the summary of two sets together
        is the larger of their two summaries, and the union is certified clear where any column
        of the combined summary is negative. So a consumer asking about every pair of a block of
        receivers and a cluster of sources summarizes each group once and never visits a pair.

        Conservative by construction: a witness may fail to certify a hull that does miss the
        body, never the other way round, and each is offset by a margin sized from the
        magnitudes it compares, so a hull that misses the body only by a rounding is not
        certified either. A body that cannot vouch for anything -- the default, and the right
        answer for a triangulated surface answered on the host -- has no columns.

        Parameters
        ----------
        position : jnp.ndarray, shape ``(..., 3)``

        Returns
        -------
        jnp.ndarray, shape ``(..., n_witnesses)``
        """
        position = jnp.asarray(position, dtype=float)
        return jnp.zeros((*position.shape[:-1], 0))


#: Directions along which a separating plane between a convex hull and a body is looked for:
#: the 26 neighbours of a cell in a 3 x 3 x 3 stencil, normalized. A fixed set, so a group of
#: positions is summarized once against every body rather than once per pair of groups.
_SEPARATING_AXES = jnp.asarray(
    [
        (x, y, z)
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
        for z in (-1.0, 0.0, 1.0)
        if (x, y, z) != (0.0, 0.0, 0.0)
    ]
)
_SEPARATING_AXES = _SEPARATING_AXES / jnp.sqrt(norm_squared(_SEPARATING_AXES))[:, None]

#: How far, relative to the magnitudes compared, a witness must clear its bound. Far above the
#: rounding of a projection or a distance (a few units in 1e16) and far below any geometric gap
#: worth certifying, so a hull that touches a body, or misses it only by a rounding, is left to
#: the exact segment test rather than vouched for.
_CLEARANCE_MARGIN = 1e-10


def _exceeds(value, bound, magnitude):
    """A witness that ``value > bound`` by more than a rounding of ``magnitude``: negative when so."""
    return bound - value + _CLEARANCE_MARGIN * magnitude


def _separating_witnesses(position, support):
    """Witnesses from separating planes along :data:`_SEPARATING_AXES`, given the body's support.

    A position with ``n . x > h(n)``, where ``h`` bounds the body's extent along ``n`` from above,
    lies strictly beyond the body along ``n``; so does the hull of any set of such positions.
    An infinite bound -- a direction the body is unbounded in -- never certifies anything.
    """
    position = jnp.asarray(position, dtype=float)[..., None, :]
    along = dot(position, _SEPARATING_AXES)
    return _exceeds(
        along, support, _projection_magnitude(position, _SEPARATING_AXES) + jnp.abs(support)
    )


def _projection_magnitude(position, direction):
    """What the rounding of ``position . direction`` is relative to: the sum of the terms' sizes."""
    return dot(jnp.abs(position), jnp.abs(direction))


def _disc_support(centre, axis, radius, direction):
    """How far a flat disc reaches along ``direction``: its centre, plus its radius times the
    length of the direction's part lying in the disc's plane."""
    along = dot(direction, axis)
    in_plane = jnp.sqrt(jnp.maximum(norm_squared(direction) - along**2, 0.0))
    return dot(direction, centre) + radius * in_plane


def _segment_parameters(origin, target):
    """Direction, squared length and length of each segment, with zero length made safe."""
    direction = jnp.asarray(target, dtype=float) - jnp.asarray(origin, dtype=float)
    squared = norm_squared(direction)
    length = jnp.sqrt(jnp.where(squared == 0.0, 1.0, squared))
    return direction, squared, length


def _unit(vector):
    """``vector`` as a length-3 unit vector."""
    vector = jnp.broadcast_to(jnp.asarray(vector, dtype=float), (3,))
    return vector / jnp.sqrt(norm_squared(vector))


def _point(position):
    """``position`` as a length-3 point."""
    return jnp.broadcast_to(jnp.asarray(position, dtype=float), (3,))


# ---------------------------------------------------------------------------------------------
# The inequalities convex bodies are built from
# ---------------------------------------------------------------------------------------------


class _Constraint(eqx.Module):
    """One inequality ``f(x) <= 0``, in the two views a convex body needs of it.

    A convex primitive is an intersection of a few of these — a cylinder is a tube and two
    planes, a box is six planes, a cone is a taper and two planes — so each inequality is written
    once and every body that has a flat end, a round side or a spherical cap shares it. The two
    views must describe the same surface, which they do by construction: they are the same
    ``f`` read at a point and along a line.
    """

    @abc.abstractmethod
    def signed_distance(self, position) -> jnp.ndarray:
        """``f(x)`` scaled as a length: negative inside, zero on the surface, positive outside.

        Exact for a plane, a tube and a ball; for a taper it is the exact distance to the
        infinite conical surface, which is what a caller comparing against a small tolerance
        needs.
        """

    @abc.abstractmethod
    def span(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Where the **line** ``origin + t direction`` satisfies the inequality.

        Returns ``(enter, exit)``, the closed interval of ``t``. Both ends may be infinite — a
        half-space keeps one of them — and an inequality the line never satisfies returns
        ``(+inf, -inf)``, which is empty under every comparison and so needs no flag of its own.
        """


def _positive_definite_span(a, b, discriminant, inside_when_degenerate):
    """Where ``a t² + 2 b t + c <= 0`` holds, for ``a >= 0``, given the discriminant directly.

    Shared by the tube and the ball, which differ only in what ``a``, ``b`` and the discriminant
    are made of. The caller passes the discriminant rather than ``c`` because **how it is formed
    is the whole numerical story**: see :class:`_Tube`.

    ``a == 0`` is the line lying along the quadratic's null direction — parallel to a tube's
    axis, or a segment of zero length — where the inequality is the same everywhere on the line
    and ``inside_when_degenerate`` decides which way.
    """
    degenerate = a == 0.0
    safe_a = jnp.where(degenerate, 1.0, a)
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    enter = (-b - root) / safe_a
    exit_ = (-b + root) / safe_a
    everywhere = jnp.where(inside_when_degenerate, -jnp.inf, jnp.inf)
    enter = jnp.where(degenerate, everywhere, jnp.where(discriminant < 0.0, jnp.inf, enter))
    exit_ = jnp.where(degenerate, -everywhere, jnp.where(discriminant < 0.0, -jnp.inf, exit_))
    return enter, exit_


class _Plane(eqx.Module):
    """``(x - point) . normal <= 0`` — the solid side is the one the normal points away from."""

    point: jnp.ndarray
    normal: jnp.ndarray

    def __init__(self, point, normal):
        self.point = _point(point)
        self.normal = _unit(normal)

    def signed_distance(self, position) -> jnp.ndarray:
        """Exact: the normal is a unit vector, so the projection is the distance."""
        return dot(jnp.asarray(position, dtype=float) - self.point, self.normal)

    def span(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """A half-line, or the whole line, or none of it when the line runs along the plane."""
        at_origin = self.signed_distance(origin)
        rate = dot(jnp.asarray(direction, dtype=float), self.normal)
        crossing = -at_origin / jnp.where(rate == 0.0, 1.0, rate)
        parallel = jnp.where(at_origin <= 0.0, -jnp.inf, jnp.inf)
        enter = jnp.where(rate < 0.0, crossing, jnp.where(rate > 0.0, -jnp.inf, parallel))
        exit_ = jnp.where(rate > 0.0, crossing, jnp.where(rate < 0.0, jnp.inf, -parallel))
        return enter, exit_


class _Ball(eqx.Module):
    """``|x - centre| <= radius``."""

    centre: jnp.ndarray
    radius: jnp.ndarray

    def __init__(self, centre, radius):
        self.centre = _point(centre)
        self.radius = jnp.asarray(radius, dtype=float)

    def signed_distance(self, position) -> jnp.ndarray:
        """Exact everywhere: distance from the centre, less the radius."""
        offset = jnp.asarray(position, dtype=float) - self.centre
        return jnp.sqrt(norm_squared(offset)) - self.radius

    def span(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The chord, formed the way a grazing ray survives — see :class:`_Tube`."""
        direction = jnp.asarray(direction, dtype=float)
        offset = jnp.asarray(origin, dtype=float) - self.centre
        a = norm_squared(direction)
        b = dot(offset, direction)
        safe_a = jnp.where(a == 0.0, 1.0, a)
        closest = offset - (b / safe_a)[..., None] * direction
        discriminant = safe_a * (self.radius**2 - norm_squared(closest))
        inside = norm_squared(offset) <= self.radius**2
        return _positive_definite_span(a, b, discriminant, inside)


class _Tube(eqx.Module):
    """``|radial part of (x - centre)| <= radius`` — an infinite circular cylinder.

    ⚠️ **The discriminant is formed as ``a (r² - h²)``, never as ``b² - a c``.** The naive form
    subtracts two nearly equal numbers exactly when the ray almost touches the surface, which at
    *any* precision throws away most of the significant digits. Writing it with ``h``, the
    distance of closest approach, formed as a vector difference, keeps the subtraction between
    quantities that are genuinely different sizes. This is not a corner case here: a sleeve sits
    at essentially the radius of the lamp facets it surrounds, so nearly tangential rays are the
    *ordinary* geometry rather than the exception.

    Attributes
    ----------
    centre : jnp.ndarray, shape ``(3,)``
        Any point on the axis.
    axis : jnp.ndarray, shape ``(3,)``
        Unit vector along the axis.
    radius : jnp.ndarray
    """

    centre: jnp.ndarray
    axis: jnp.ndarray
    radius: jnp.ndarray

    def __init__(self, centre, axis, radius):
        self.centre = _point(centre)
        self.axis = _unit(axis)
        self.radius = jnp.asarray(radius, dtype=float)

    def _radial(self, vector):
        """The part of ``vector`` perpendicular to the axis."""
        return vector - dot(vector, self.axis)[..., None] * self.axis

    def signed_distance(self, position) -> jnp.ndarray:
        """Exact: distance from the axis, less the radius."""
        radial = self._radial(jnp.asarray(position, dtype=float) - self.centre)
        return jnp.sqrt(norm_squared(radial)) - self.radius

    def span(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The chord across the circle the axis is seen as, in the perpendicular plane."""
        radial_direction = self._radial(jnp.asarray(direction, dtype=float))
        radial_offset = self._radial(jnp.asarray(origin, dtype=float) - self.centre)

        a = norm_squared(radial_direction)
        b = dot(radial_offset, radial_direction)
        safe_a = jnp.where(a == 0.0, 1.0, a)
        # Closest approach of the line to the axis, as a vector difference rather than as the
        # difference of two squares -- see the class docstring.
        closest = radial_offset - (b / safe_a)[..., None] * radial_direction
        discriminant = safe_a * (self.radius**2 - norm_squared(closest))
        # A line running along the axis never crosses the curved surface: it is inside the tube
        # for its whole length, or outside it for the whole length.
        inside = norm_squared(radial_offset) <= self.radius**2
        return _positive_definite_span(a, b, discriminant, inside)


def _intersect(first, second):
    """The overlap of two intervals, empty if they do not meet."""
    return jnp.maximum(first[0], second[0]), jnp.minimum(first[1], second[1])


class _Taper(eqx.Module):
    """The solid infinite cone whose radius grows linearly along an axis.

    The surface is ``|radial part of (x - start)| = R(u)`` with ``u`` the distance along the axis
    and ``R(u) = start_radius + slope * u``. Writing it from a radius *profile* rather than from
    an apex and a half-angle is deliberate: a real reducer is given as two radii and a length,
    the apex may be far outside the body, and the profile form degenerates gracefully toward a
    tube as the two radii approach each other, where an apex runs off to infinity.

    ⚠️ **Only the nappe where ``R(u) >= 0`` is solid.** The quadratic is satisfied on the mirror
    cone beyond the apex as well, where ``R`` has gone negative and ``|radial| <= |R|`` — a
    perfectly good solution of the equation and a body that is not there. The span below cuts to
    the valid side explicitly rather than relying on a caller's end caps to hide it, so that the
    inequality means the same thing on its own as it does inside a body.

    ⚠️ **The quadratic here is indefinite**, unlike a tube's or a ball's: a line steeper than the
    cone's own slope is inside the double cone *outside* its two roots rather than between them.
    That is why this cannot share :func:`_positive_definite_span`, and why the closest-approach
    reformulation that protects a grazing tube does not apply — there is no positive-definite
    form to build it from. Near-tangential accuracy on a taper is therefore the ordinary
    ``b² - a c``, and a body whose surface a ray must graze is better described as a tube.
    """

    start: jnp.ndarray
    axis: jnp.ndarray
    slope: jnp.ndarray
    start_radius: jnp.ndarray

    def __init__(self, start, axis, slope, start_radius):
        self.start = _point(start)
        self.axis = _unit(axis)
        self.slope = jnp.asarray(slope, dtype=float)
        self.start_radius = jnp.asarray(start_radius, dtype=float)

    def _split(self, vector):
        """The radial and axial parts of ``vector`` about the axis."""
        along = dot(vector, self.axis)
        return vector - along[..., None] * self.axis, along

    def signed_distance(self, position) -> jnp.ndarray:
        """Distance to the conical surface, exact on the solid nappe.

        The radial excess is measured perpendicular to the axis, so it overstates the distance
        to a *tilted* surface by exactly the cosine of the taper's half-angle, which is divided
        out. Beyond the apex the profile radius has gone negative and this is positive — the
        point is outside the body, which is what a caller needs there — but it is no longer the
        distance to anything.
        """
        radial, along = self._split(jnp.asarray(position, dtype=float) - self.start)
        excess = jnp.sqrt(norm_squared(radial)) - (self.start_radius + self.slope * along)
        return excess / jnp.sqrt(1.0 + self.slope**2)

    def span(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Where the line is inside the solid nappe."""
        offset_radial, offset_along = self._split(jnp.asarray(origin, dtype=float) - self.start)
        step_radial, step_along = self._split(jnp.asarray(direction, dtype=float))
        # The profile radius along the line is itself affine: a value at the origin and a rate.
        offset_radius = self.start_radius + self.slope * offset_along
        step_radius = self.slope * step_along

        a = norm_squared(step_radial) - step_radius**2
        b = dot(offset_radial, step_radial) - offset_radius * step_radius
        c = norm_squared(offset_radial) - offset_radius**2
        discriminant = b * b - a * c
        root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        safe_a = jnp.where(a == 0.0, 1.0, a)
        lower = jnp.minimum((-b - root) / safe_a, (-b + root) / safe_a)
        upper = jnp.maximum((-b - root) / safe_a, (-b + root) / safe_a)

        # A line shallower than the cone enters and leaves one nappe: inside between the roots.
        misses = discriminant < 0.0
        shallow = (jnp.where(misses, jnp.inf, lower), jnp.where(misses, -jnp.inf, upper))
        # A line steeper than the cone is inside OUTSIDE the roots, one branch per nappe; the
        # growing side of the profile says which branch is the solid one. With no real roots it
        # never leaves the double cone at all.
        growing = step_radius > 0.0
        steep = (
            jnp.where(misses, -jnp.inf, jnp.where(growing, upper, -jnp.inf)),
            jnp.where(misses, jnp.inf, jnp.where(growing, jnp.inf, lower)),
        )
        # Exactly along the cone's own slope the quadratic collapses to a line.
        crossing = -0.5 * c / jnp.where(b == 0.0, 1.0, b)
        parallel = jnp.where(c <= 0.0, -jnp.inf, jnp.inf)
        flat = (
            jnp.where(b < 0.0, crossing, jnp.where(b > 0.0, -jnp.inf, parallel)),
            jnp.where(b > 0.0, crossing, jnp.where(b < 0.0, jnp.inf, -parallel)),
        )

        quadratic = tuple(
            jnp.where(a > 0.0, s, jnp.where(a < 0.0, t, f))
            for s, t, f in zip(shallow, steep, flat, strict=True)
        )
        # Cut to the nappe the body is on: R(u) >= 0 along the line.
        #
        # ⚠️ Through `Cone` this changes nothing, and that is measured rather than assumed: with
        # both end radii non-negative the profile radius is non-negative across the body, so the
        # mirror nappe always lies beyond an end cap and the caps discard it anyway. Removing
        # this line moved not one answer in 160,000 rays over four cone shapes -- a true cone, a
        # frustum whose apex is far outside it, a tilted narrowing one and a steep one.
        #
        # It stays because the inequality has to mean on its own what it says it means. Without
        # it, `span` returns the mirror branch whenever the line meets only that one, and this
        # helper would silently require its caller to supply end caps: on the bare taper the two
        # differ on 7.2% of the same rays. What is NOT redundant is the branch selection above --
        # picking the wrong one of the two branches reports a solid cone as clear, and a test
        # pins that.
        apex = -offset_radius / jnp.where(step_radius == 0.0, 1.0, step_radius)
        whole = jnp.where(offset_radius >= 0.0, -jnp.inf, jnp.inf)
        solid_side = (
            jnp.where(growing, apex, jnp.where(step_radius < 0.0, -jnp.inf, whole)),
            jnp.where(step_radius < 0.0, apex, jnp.where(growing, jnp.inf, -whole)),
        )
        return _intersect(quadratic, solid_side)


# ---------------------------------------------------------------------------------------------
# The two bases: a body whose inside along a line is a set of intervals, and the convex case
# ---------------------------------------------------------------------------------------------


class Solid(Body):
    """A body whose inside, seen along a line, is a bounded union of intervals.

    Everything else here is written against that one property. Blocking is "does any interval
    overlap the segment"; composing bodies is interval arithmetic; and describing a vessel by the
    fluid it holds is "do the intervals cover the segment". None of those needs to know what
    shape produced the intervals, which is what keeps one implementation of each rather than one
    per primitive.

    The count of intervals is a property of the *body*, fixed when it is built and the same for
    every ray — a convex body has one, a union of three has three — so nothing here is a search
    with a data-dependent length. That is what lets the whole mask stay a single array
    expression.
    """

    traceable: ClassVar[bool] = True

    @property
    @abc.abstractmethod
    def interval_count(self) -> int:
        """How many intervals :meth:`intervals` returns, the same for every ray."""

    @abc.abstractmethod
    def intervals(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Where the **line** ``origin + t direction`` is inside this body.

        Parameters
        ----------
        origin, direction : jnp.ndarray, shape ``(..., 3)``
            Broadcast against each other. ``t = 0`` is at ``origin`` and ``t = 1`` one
            ``direction`` further on, so a segment is the part of the line in ``[0, 1]``.

        Returns
        -------
        enter, exit : jnp.ndarray, shape ``(..., interval_count)``
            The intervals, which need not be sorted or disjoint. An interval with
            ``enter > exit`` is empty — a miss is written ``(+inf, -inf)`` — so a caller never
            has to ask whether a given interval is real.
        """

    @abc.abstractmethod
    def signed_distance(self, position) -> jnp.ndarray:
        """How far each position is from the surface: negative inside, positive outside.

        Exact for a single primitive. For a composed body it is the usual constructive estimate
        — the larger of two distances for an intersection, the smaller for a union — which is
        exact in magnitude on the surface and a lower bound on how far away an outside point is.
        That is the conservative direction for the one thing it is used for: deciding whether a
        point is inside a body, or outside it by more than a tolerance.

        Parameters
        ----------
        position : jnp.ndarray, shape ``(..., 3)``

        Returns
        -------
        jnp.ndarray, shape ``(...)``
        """

    def support(self, direction) -> jnp.ndarray:
        """An upper bound on how far the body reaches along each direction.

        The support function ``h(n) = max over the body of n . x``, exact for a primitive and a
        bound for a composition (an intersection reaches no further than its nearest member, a
        difference no further than the body it is cut from). Infinite where no bound is known or
        the body is unbounded that way -- the default, which is always true.

        Parameters
        ----------
        direction : jnp.ndarray, shape ``(..., 3)``

        Returns
        -------
        jnp.ndarray, shape ``(...)``
        """
        direction = jnp.asarray(direction, dtype=float)
        return jnp.full(direction.shape[:-1], jnp.inf)

    def clearance(self, position) -> jnp.ndarray:
        """Separating planes along a fixed set of directions, placed by :meth:`support`."""
        return _separating_witnesses(position, self.support(_SEPARATING_AXES))

    def contains(self, position) -> jnp.ndarray:
        """True where the position is on or inside the surface."""
        return self.signed_distance(position) <= 0.0

    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        """True where any of the body's intervals overlaps the segment.

        The segment is ``t`` in ``[0, 1]``, less a sliver next to the origin: a facet lying on a
        body would otherwise shadow itself, because the intersection sits at essentially zero
        distance and whether it is found comes down to rounding.
        """
        direction, _, length = _segment_parameters(origin, target)
        enter, exit_ = self.intervals(origin, direction)
        near = (jnp.asarray(min_distance) / length)[..., None]
        return jnp.any((exit_ >= jnp.maximum(enter, near)) & (enter <= 1.0), axis=-1)


class ConvexSolid(Solid):
    """A body bounded by a handful of inequalities, all of which hold inside it.

    A line meets a convex body in a single interval, so the intersection of the inequalities'
    own intervals *is* the body's — no case analysis, no sorting, and one interval however many
    inequalities there are. Each primitive below is therefore only a list of inequalities and
    the parameters that build them.
    """

    @property
    @abc.abstractmethod
    def constraints(self) -> tuple:
        """The inequalities bounding this body, all of which hold inside it."""

    @property
    def interval_count(self) -> int:
        """One, which is what convexity means for a line."""
        return 1

    def signed_distance(self, position) -> jnp.ndarray:
        """The inequality the position violates worst, or the one it satisfies least."""
        return functools.reduce(
            jnp.maximum, [bound.signed_distance(position) for bound in self.constraints]
        )

    def intervals(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The overlap of the inequalities' intervals, as a single interval."""
        enter, exit_ = functools.reduce(
            _intersect, [bound.span(origin, direction) for bound in self.constraints]
        )
        return enter[..., None], exit_[..., None]

    def clearance(self, position) -> jnp.ndarray:
        """The body's own flat faces, then separating planes along the fixed directions.

        Beyond a flat face is outside the whole body, since every inequality holds inside it, and
        that region is a half-space -- convex -- so each face is a witness as it stands. It is the
        only witness a :class:`HalfSpace` has, being unbounded along every fixed direction. A
        curved inequality is not one: the outside of a tube is not convex.
        """
        position = jnp.asarray(position, dtype=float)
        faces = [
            _exceeds(
                dot(position, bound.normal),
                dot(bound.point, bound.normal),
                _projection_magnitude(position, bound.normal)
                + _projection_magnitude(bound.point, bound.normal),
            )
            for bound in self.constraints
            if isinstance(bound, _Plane)
        ]
        return jnp.concatenate(
            [jnp.stack(faces, axis=-1), super().clearance(position)]
            if faces
            else [super().clearance(position)],
            axis=-1,
        )


# ---------------------------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------------------------


class HalfSpace(ConvexSolid):
    """Everything on one side of a plane — a wall, a floor, a large flat baffle.

    The solid is the side the normal points **away** from: a point ``x`` is inside when
    ``(x - point) . normal <= 0``.

    That includes a receiver sitting inside the solid, and it should: a cell on the far side of
    a wall is inside metal, and a model that reported light there would be reporting light
    inside metal.

    Attributes
    ----------
    point : jnp.ndarray, shape ``(3,)``
        Any point on the bounding plane.
    normal : jnp.ndarray, shape ``(3,)``
        Unit normal pointing out of the solid.
    """

    point: jnp.ndarray
    normal: jnp.ndarray

    def __init__(self, point, normal):
        self.point = _point(point)
        self.normal = _unit(normal)

    @property
    def constraints(self) -> tuple:
        """The one plane."""
        return (_Plane(self.point, self.normal),)


class Sphere(ConvexSolid):
    """A solid ball — a float, a probe head, the rounded tip of a lamp.

    Attributes
    ----------
    centre : jnp.ndarray, shape ``(3,)``
    radius : jnp.ndarray
    """

    centre: jnp.ndarray
    radius: jnp.ndarray

    def __init__(self, centre, radius):
        self.centre = _point(centre)
        self.radius = jnp.asarray(radius, dtype=float)

    @property
    def constraints(self) -> tuple:
        """The one ball."""
        return (_Ball(self.centre, self.radius),)

    def support(self, direction) -> jnp.ndarray:
        """The centre's reach, plus the radius times the direction's length."""
        direction = jnp.asarray(direction, dtype=float)
        return dot(direction, self.centre) + self.radius * jnp.sqrt(norm_squared(direction))


class Cylinder(ConvexSolid):
    """A finite solid cylinder with flat ends — the shape of a lamp sleeve.

    The curved side and the two ends are three inequalities; their intervals overlap in the
    chord through the body.

    Attributes
    ----------
    centre : jnp.ndarray, shape ``(3,)``
        Midpoint of the axis.
    axis : jnp.ndarray, shape ``(3,)``
        Unit vector along the axis.
    radius : jnp.ndarray
        Cylinder radius.
    half_length : jnp.ndarray
        Half the axial extent, measured from ``centre``.
    """

    centre: jnp.ndarray
    axis: jnp.ndarray
    radius: jnp.ndarray
    half_length: jnp.ndarray

    def __init__(self, centre, axis, radius, half_length):
        self.centre = _point(centre)
        self.axis = _unit(axis)
        self.radius = jnp.asarray(radius, dtype=float)
        self.half_length = jnp.asarray(half_length, dtype=float)

    @property
    def constraints(self) -> tuple:
        """The curved side, then the two flat ends."""
        end = self.half_length * self.axis
        return (
            _Tube(self.centre, self.axis, self.radius),
            _Plane(self.centre + end, self.axis),
            _Plane(self.centre - end, -self.axis),
        )

    def support(self, direction) -> jnp.ndarray:
        """The further of the two end discs: the body is their convex hull."""
        direction = jnp.asarray(direction, dtype=float)
        disc = _disc_support(self.centre, self.axis, self.radius, direction)
        return disc + self.half_length * jnp.abs(dot(direction, self.axis))


class Cone(ConvexSolid):
    """A solid cone or truncated cone — a reducer, a diffuser, a conical baffle.

    Described by a radius at each end rather than by an apex and a half-angle, because that is
    how a reducer is drawn, and because the apex of a shallow taper lies far outside the body
    and may not exist at all when the two radii are equal.

    ⚠️ **Equal radii are refused** wherever they can be seen at construction: that body is a
    :class:`Cylinder`, and a cylinder's curved side is written in the form that survives a ray
    grazing it, which a taper's cannot be. A taper whose two radii are merely *nearly* equal
    inherits the same weakness near tangency and is better described as a cylinder too.

    Attributes
    ----------
    centre : jnp.ndarray, shape ``(3,)``
        Midpoint of the axis.
    axis : jnp.ndarray, shape ``(3,)``
        Unit vector along the axis, pointing from the base end toward the tip end.
    half_length : jnp.ndarray
        Half the axial extent, measured from ``centre``.
    base_radius : jnp.ndarray
        Radius at the ``-axis`` end.
    tip_radius : jnp.ndarray
        Radius at the ``+axis`` end. Zero, the default, makes a cone with its point along
        ``+axis``.
    """

    centre: jnp.ndarray
    axis: jnp.ndarray
    half_length: jnp.ndarray
    base_radius: jnp.ndarray
    tip_radius: jnp.ndarray

    def __init__(self, centre, axis, half_length, base_radius, tip_radius=0.0):
        self.centre = _point(centre)
        self.axis = _unit(axis)
        self.half_length = jnp.asarray(half_length, dtype=float)
        self.base_radius = jnp.asarray(base_radius, dtype=float)
        self.tip_radius = jnp.asarray(tip_radius, dtype=float)
        # Only where the radii are plain numbers, which is every scene built from a drawing. A
        # radius arriving as an array cannot be compared without forcing a value out of a trace,
        # and the degenerate taper is correct there anyway -- merely less robust near tangency.
        if (
            isinstance(base_radius, numbers.Real)
            and isinstance(tip_radius, numbers.Real)
            and base_radius == tip_radius
        ):
            msg = (
                "a Cone with equal end radii is a Cylinder -- use one. A cylinder's curved side "
                "is written in the form that survives a ray grazing it, which a taper's cannot "
                "be, so the taper is the worse-conditioned spelling of the same body."
            )
            raise ValueError(msg)

    @property
    def constraints(self) -> tuple:
        """The tapered side, then the two flat ends."""
        end = self.half_length * self.axis
        slope = (self.tip_radius - self.base_radius) / (2.0 * self.half_length)
        return (
            _Taper(self.centre - end, self.axis, slope, self.base_radius),
            _Plane(self.centre + end, self.axis),
            _Plane(self.centre - end, -self.axis),
        )

    def support(self, direction) -> jnp.ndarray:
        """The further of the two end discs: a truncated cone is their convex hull."""
        direction = jnp.asarray(direction, dtype=float)
        end = self.half_length * self.axis
        return jnp.maximum(
            _disc_support(self.centre - end, self.axis, self.base_radius, direction),
            _disc_support(self.centre + end, self.axis, self.tip_radius, direction),
        )


class Box(ConvexSolid):
    """A solid box — a baffle, a plate, a rectangular duct wall, a bounding volume.

    Attributes
    ----------
    centre : jnp.ndarray, shape ``(3,)``
    half_sizes : jnp.ndarray, shape ``(3,)``
        Half the extent along each of ``axes``.
    axes : jnp.ndarray, shape ``(3, 3)``
        The box's own directions, one per row, each normalized on construction. Mutually
        perpendicular rows give a rectangular box; merely independent ones give the
        parallelepiped bounded by the three pairs of planes perpendicular to them. Defaults to
        the coordinate axes.
    """

    centre: jnp.ndarray
    half_sizes: jnp.ndarray
    axes: jnp.ndarray

    def __init__(self, centre, half_sizes, axes=None):
        self.centre = _point(centre)
        self.half_sizes = jnp.broadcast_to(jnp.asarray(half_sizes, dtype=float), (3,))
        axes = jnp.eye(3) if axes is None else jnp.asarray(axes, dtype=float)
        self.axes = axes / jnp.sqrt(norm_squared(axes))[:, None]

    @property
    def constraints(self) -> tuple:
        """Six planes, a facing pair per axis."""
        return tuple(
            _Plane(self.centre + sign * self.half_sizes[i] * self.axes[i], sign * self.axes[i])
            for i in range(3)
            for sign in (1.0, -1.0)
        )

    def support(self, direction) -> jnp.ndarray:
        """The furthest corner.

        A corner is where one plane of each facing pair meets, ``centre + E (s * half_sizes)``
        for a choice of signs ``s``, where the columns of ``E`` -- the inverse of the matrix whose
        rows are :attr:`axes` -- are the box's edge directions. Along ``n`` a corner reaches
        ``n . centre + sum_j s_j half_sizes_j (n . E_j)``, and each sign is chosen independently,
        so the furthest takes every term at its magnitude. Exact for a parallelepiped as well as
        a rectangular box.
        """
        direction = jnp.asarray(direction, dtype=float)
        edges = jnp.linalg.inv(self.axes)
        reach = jnp.abs(direction @ edges) @ self.half_sizes
        return dot(direction, self.centre) + reach


# ---------------------------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------------------------


def _as_solids(bodies, what):
    """The bodies of a combinator, checked for the one thing that cannot be recovered from."""
    bodies = tuple(bodies)
    if not bodies:
        msg = f"{what} needs at least one body; an empty one is a shape with no meaning"
        raise ValueError(msg)
    for body in bodies:
        if not isinstance(body, Solid):
            msg = (
                f"{what} composes Solid bodies by interval arithmetic, and "
                f"{type(body).__name__} does not report intervals. Only a body that can say "
                "where a line is inside it can be combined exactly with another."
            )
            raise TypeError(msg)
    return bodies


class Union(Solid):
    """Everything inside any of several bodies — a lamp is a cylinder and a hemispherical tip.

    A line is inside the union exactly where it is inside any one of them, so the intervals are
    simply pooled. They may overlap and are not sorted; nothing downstream needs them to be.

    Attributes
    ----------
    bodies : tuple of Solid
    """

    bodies: tuple

    def __init__(self, *bodies):
        self.bodies = _as_solids(bodies, "Union")

    @property
    def interval_count(self) -> int:
        """The bodies' counts added."""
        return sum(body.interval_count for body in self.bodies)

    def intervals(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Every body's intervals, pooled."""
        spans = [body.intervals(origin, direction) for body in self.bodies]
        return (
            jnp.concatenate([enter for enter, _ in spans], axis=-1),
            jnp.concatenate([exit_ for _, exit_ in spans], axis=-1),
        )

    def signed_distance(self, position) -> jnp.ndarray:
        """The nearest body's distance."""
        return functools.reduce(
            jnp.minimum, [body.signed_distance(position) for body in self.bodies]
        )

    def support(self, direction) -> jnp.ndarray:
        """The furthest reach of any body."""
        return functools.reduce(jnp.maximum, [body.support(direction) for body in self.bodies])


class Intersection(Solid):
    """Only what is inside all of several bodies — a plate cut to the shape of a duct.

    A line is inside the intersection where it is inside every body at once, so each of one
    body's intervals is overlapped with each of another's. The count multiplies, which is why an
    intersection of convex bodies — one interval each — costs nothing extra and is itself convex.

    Attributes
    ----------
    bodies : tuple of Solid
    """

    bodies: tuple

    def __init__(self, *bodies):
        self.bodies = _as_solids(bodies, "Intersection")

    @property
    def interval_count(self) -> int:
        """The bodies' counts multiplied."""
        return functools.reduce(lambda a, b: a * b, (body.interval_count for body in self.bodies))

    def intervals(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Every pairing of one body's intervals with another's, overlapped."""

        def overlap(first, second):
            enter, exit_ = _intersect(
                (first[0][..., :, None], first[1][..., :, None]),
                (second[0][..., None, :], second[1][..., None, :]),
            )
            flat = (*enter.shape[:-2], enter.shape[-2] * enter.shape[-1])
            return jnp.reshape(enter, flat), jnp.reshape(exit_, flat)

        return functools.reduce(
            overlap, [body.intervals(origin, direction) for body in self.bodies]
        )

    def signed_distance(self, position) -> jnp.ndarray:
        """The body the position is furthest outside, or least far inside."""
        return functools.reduce(
            jnp.maximum, [body.signed_distance(position) for body in self.bodies]
        )

    def support(self, direction) -> jnp.ndarray:
        """No further than the nearest reach of any body -- a bound, not exact."""
        return functools.reduce(jnp.minimum, [body.support(direction) for body in self.bodies])

    def clearance(self, position) -> jnp.ndarray:
        """Every body's witnesses: a hull that misses any one of them misses the intersection."""
        return jnp.concatenate([body.clearance(position) for body in self.bodies], axis=-1)


class Difference(Solid):
    """A body with a piece taken out of it — a sleeve is a cylinder less its bore.

    Removing a **convex** hole cuts each of the body's intervals into the part before the hole
    and the part after it, so the count doubles. That is the whole implementation, and it is why
    the hole must be convex: a hole in several pieces would have to be sorted and merged before
    its complement could be written down at all.

    ⚠️ **Subtracting several holes is repeated subtraction**, not one difference with a union:
    ``A - (B + C)`` is written ``Difference(Difference(A, B), C)``. Each step doubles the
    interval count, which is the honest price of the cut.

    Attributes
    ----------
    body : Solid
        What the material is.
    hole : Solid
        What is taken out of it. Must be convex, in the sense of reporting exactly one interval.
    """

    body: Solid
    hole: Solid

    def __init__(self, body, hole):
        (self.body, self.hole) = _as_solids((body, hole), "Difference")
        if self.hole.interval_count != 1:
            msg = (
                f"Difference removes a convex hole, and {type(hole).__name__} reports "
                f"{self.hole.interval_count} intervals. Subtract the pieces one at a time -- "
                "A - (B + C) is Difference(Difference(A, B), C) -- so that each cut is a single "
                "interval taken out of what is left."
            )
            raise ValueError(msg)

    @property
    def interval_count(self) -> int:
        """Twice the body's: each interval is cut into a piece before the hole and one after."""
        return 2 * self.body.interval_count

    def intervals(self, origin, direction) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Each of the body's intervals, split by the hole's."""
        enter, exit_ = self.body.intervals(origin, direction)
        cut_enter, cut_exit = self.hole.intervals(origin, direction)
        before = (enter, jnp.minimum(exit_, cut_enter))
        after = (jnp.maximum(enter, cut_exit), exit_)
        return (
            jnp.concatenate([before[0], after[0]], axis=-1),
            jnp.concatenate([before[1], after[1]], axis=-1),
        )

    def signed_distance(self, position) -> jnp.ndarray:
        """Inside the body and outside the hole, which is the hole's distance negated."""
        return jnp.maximum(
            self.body.signed_distance(position), -self.hole.signed_distance(position)
        )

    def support(self, direction) -> jnp.ndarray:
        """No further than the body the hole is cut from."""
        return self.body.support(direction)

    def clearance(self, position) -> jnp.ndarray:
        """The body's witnesses: a hull that misses the body misses what is left of it."""
        return self.body.clearance(position)


def _covers(enter, exit_, lower, upper) -> jnp.ndarray:
    """Whether a pool of intervals covers ``[lower, upper]`` with no gap anywhere in it.

    The intervals may overlap, repeat, be empty, and arrive in any order. Sorting them would be
    the textbook approach and is the wrong one here: a sort over the last axis of a
    receivers-by-facets array materializes a permutation the size of the whole mask, where the
    test below is a handful of comparisons that fuse into the reduction and never form anything.

    **Why comparing against each interval's far end is enough.** Coverage can only first fail at
    ``lower``, or at the right-hand end of some interval — anywhere else, whatever covers a point
    covers its neighbourhood too. So at each of those places the pool must not merely reach: some
    interval must *continue past* it, strictly. Both halves are load-bearing, and dropping either
    reports a gap as covered rather than the other way round. Without the endpoint candidates,
    only ``lower`` is examined and a gap beyond it is never looked at. Without the strict
    inequality, an interval's own far end is trivially satisfied by that same interval, so every
    candidate passes and no gap is ever found.

    Intervals that meet exactly, ``[a, b]`` and ``[b, c]``, do cover ``[a, c]``: they are closed
    and share the point. That is the right answer and the strict test gives it, because the
    second interval continues past ``b``.

    Parameters
    ----------
    enter, exit_ : jnp.ndarray, shape ``(..., k)``
        The intervals.
    lower, upper : jnp.ndarray, shape ``(...)``
        The stretch to cover. An empty one, ``lower >= upper``, is covered by anything.

    Returns
    -------
    jnp.ndarray of bool, shape ``(...)``
    """
    candidate = jnp.concatenate([lower[..., None], exit_], axis=-1)
    inside = (candidate >= lower[..., None]) & (candidate < upper[..., None])
    continued = jnp.any(
        (enter[..., None, :] <= candidate[..., :, None])
        & (candidate[..., :, None] < exit_[..., None, :]),
        axis=-1,
    )
    return jnp.all(continued | ~inside, axis=-1)


class Outside(Body):
    """Everything that is not fluid — a vessel described by what it holds rather than by its wall.

    This is the cheapest and most accurate way to shadow a real reactor, and it is a different
    idea from the bodies above rather than another one of them. A chamber with a pipe off it has
    a wall that is awkward to write down as a solid; the *fluid* is two cylinders. A segment
    between two points in that fluid is clear exactly when the regions cover it end to end, and
    nothing else has to be tested — no opening has to be identified, no shadow edge derived.

    **Convexity is what makes it cheap, and it is not needed for it to be right.** Any
    :class:`Solid` may be a region: the test pools every region's intervals and asks whether
    together they leave a gap. A convex region contributes one interval, which is why a chain of
    chambers and pipes costs a few comparisons per ray. It is also where the shortcut everyone
    reaches for comes from: two points inside one convex region have a clear segment between them
    by definition, with nothing to test at all.

    ⚠️ **Neighbouring regions must overlap or touch — a gap between them reads as solid**, and
    reads that way silently, as a shadow rather than as an error. A pipe standing on a chamber is
    described by extending the pipe's cylinder *into* the chamber, not by stopping it at the
    chamber's surface, where a curved junction would leave slivers of neither region.

    ⚠️ **This cannot be nested inside :class:`Union`, :class:`Intersection` or
    :class:`Difference`.** The complement of a pool of intervals is not a pool of intervals
    without first sorting and merging them, which is the cost this class is written to avoid.
    Use :class:`Difference` to take material out of a body.

    Attributes
    ----------
    regions : tuple of Solid
        The fluid. Their union is everything this body is *not*.
    tolerance : jnp.ndarray
        How far outside every region a point must be, as a length, before
        :meth:`~Body.contains` calls it embedded in the wall. A mesh never lands exactly on
        the surface its cells were snapped to, so a cell centre a rounding outside a region is a
        discretization, not a cell in the metal. Applies only to that test: where a *segment* is
        clear is decided by the geometry with no slack, and the margin it needs at its two ends
        comes from ``min_distance`` instead.
    """

    traceable: ClassVar[bool] = True

    regions: tuple
    tolerance: jnp.ndarray

    def __init__(self, *regions, tolerance=0.0):
        self.regions = _as_solids(regions, "Outside")
        self.tolerance = jnp.asarray(tolerance, dtype=float)

    @property
    def fluid(self) -> Solid:
        """The regions as one body — what this body is the outside of."""
        return Union(*self.regions)

    def contains(self, position) -> jnp.ndarray:
        """True where the position is outside every region by more than the tolerance."""
        return self.fluid.signed_distance(position) > self.tolerance

    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        """True where the segment leaves the fluid anywhere between its two ends.

        The two ends are excluded by ``min_distance``, and **both** of them: a facet centroid
        used as a receiver sits exactly on the wall the regions are bounded by, so the far end
        needs the same margin the near end does. Without it a segment between two facets that
        can see each other reads as blocked by a gap of one rounding, which is a shadow over a
        whole enclosure rather than a small error.
        """
        direction, _, length = _segment_parameters(origin, target)
        enter, exit_ = self.fluid.intervals(origin, direction)
        near = jnp.asarray(min_distance) / length
        return ~_covers(enter, exit_, near, 1.0 - near)

    def clearance(self, position) -> jnp.ndarray:
        """One witness per convex region: every position strictly inside it.

        A convex region holds the hull of any positions inside it, and a region is fluid, which
        this body is the outside of -- so positions all inside one region have every segment
        between them clear, the shortcut the class docstring describes, read off for a whole
        group at once. The margin is sized from the position's own magnitude, since that is what
        the rounding of a distance to a surface is relative to. A region that is not a
        :class:`ConvexSolid` gives no witness: the hull of positions inside a non-convex region
        can leave it.
        """
        position = jnp.asarray(position, dtype=float)
        scale = jnp.sqrt(norm_squared(position))
        witnesses = [
            _exceeds(0.0, region.signed_distance(position), scale)
            for region in self.regions
            if isinstance(region, ConvexSolid)
        ]
        if not witnesses:
            return super().clearance(position)
        return jnp.stack(witnesses, axis=-1)
