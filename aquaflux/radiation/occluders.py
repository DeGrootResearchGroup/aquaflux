"""Solid bodies that stand between a source and a receiver.

An occluder answers one question about a straight segment: does the segment pass through this
body? That is all the gather needs, because a body's *effect* on the light — opaque, or a
partly transmitting sleeve — is carried separately as a transmittance, and the two are split
deliberately. The geometry is fixed when the model is built and the intersection test is a hard
yes or no; the transmittance is a number that varies from one evaluation to the next and that a
design study differentiates with respect to. Keeping them apart is what lets the expensive,
discontinuous half be computed once and the smooth half stay live.

The bodies here are **analytic primitives**, described by a few numbers rather than by a
surface: a lamp sleeve is a cylinder, a wall or a baffle is a half-space. They are not the only
occluders — the emitting geometry occludes too, through its own triangles — but they are the
cheap case, and a handful of them is tested by brute force because a hierarchy over four bodies
is one leaf node.

⚠️ **Do not represent a body twice.** A sleeve that is already an emitting surface must not also
be added here: every ray would then leave a facet lying exactly on an occluder, which is the
degenerate configuration that ray-tracing tools warn about, and the emitter's own convexity
already makes the source-side cosine clamp an exact visibility test for it.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, norm_squared

__all__ = ["Cylinder", "HalfSpace", "Occluder"]


class Occluder(eqx.Module):
    """A solid body, and the test for whether a segment passes through it."""

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


def _segment_parameters(origin, target):
    """Direction, squared length and length of each segment, with zero length made safe."""
    direction = jnp.asarray(target, dtype=float) - jnp.asarray(origin, dtype=float)
    squared = norm_squared(direction)
    length = jnp.sqrt(jnp.where(squared == 0.0, 1.0, squared))
    return direction, squared, length


class HalfSpace(Occluder):
    """Everything on one side of a plane — a wall, a floor, a large flat baffle.

    The solid is the side the normal points **away** from: a point ``x`` is inside when
    ``(x - point) . normal <= 0``. A segment meets the body when any part of it is inside,
    which, the body being convex and the two endpoints bounding the segment, means when either
    endpoint is.

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
        self.point = jnp.broadcast_to(jnp.asarray(point, dtype=float), (3,))
        normal = jnp.broadcast_to(jnp.asarray(normal, dtype=float), (3,))
        self.normal = normal / jnp.sqrt(norm_squared(normal))

    def contains(self, position) -> jnp.ndarray:
        """True on the solid side of the plane."""
        return dot(jnp.asarray(position, dtype=float) - self.point, self.normal) <= 0.0

    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        """True where either endpoint lies in the solid, beyond the near-origin exclusion."""
        direction, _, length = _segment_parameters(origin, target)
        origin = jnp.asarray(origin, dtype=float)
        excluded = jnp.asarray(min_distance) / length
        entry = origin + direction * excluded[..., None]
        at_entry = dot(entry - self.point, self.normal)
        at_target = dot(jnp.asarray(target, dtype=float) - self.point, self.normal)
        return (at_entry <= 0.0) | (at_target <= 0.0)


class Cylinder(Occluder):
    """A finite solid cylinder with flat ends — the shape of a lamp sleeve.

    The intersection is solved in the plane perpendicular to the axis, where it is a ray against
    a circle, and then clipped to the axial extent.

    ⚠️ **The quadratic is written in the form that survives a grazing hit.** Taking the
    discriminant as ``b^2 - a c`` subtracts two nearly equal numbers exactly when the ray almost
    touches the surface, which at *any* precision throws away most of the significant digits.
    Writing it as ``a (r^2 - h^2)``, with ``h`` the distance of closest approach formed as a
    vector difference, keeps the subtraction between quantities that are genuinely different
    sizes. This is not a corner case here: a sleeve sits at essentially the radius of the lamp
    facets it surrounds, so nearly tangential rays are the *ordinary* geometry rather than the
    exception.

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
        self.centre = jnp.broadcast_to(jnp.asarray(centre, dtype=float), (3,))
        axis = jnp.broadcast_to(jnp.asarray(axis, dtype=float), (3,))
        self.axis = axis / jnp.sqrt(norm_squared(axis))
        self.radius = jnp.asarray(radius, dtype=float)
        self.half_length = jnp.asarray(half_length, dtype=float)

    def contains(self, position) -> jnp.ndarray:
        """True inside the curved surface and between the two ends."""
        offset = jnp.asarray(position, dtype=float) - self.centre
        along = dot(offset, self.axis)
        radial = offset - along[..., None] * self.axis
        return (norm_squared(radial) <= self.radius**2) & (jnp.abs(along) <= self.half_length)

    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        """True where the segment passes through the solid cylinder."""
        direction, _, length = _segment_parameters(origin, target)
        origin = jnp.asarray(origin, dtype=float)
        offset = origin - self.centre

        along_axis = dot(direction, self.axis)
        offset_along = dot(offset, self.axis)
        radial_direction = direction - along_axis[..., None] * self.axis
        radial_offset = offset - offset_along[..., None] * self.axis

        a = norm_squared(radial_direction)
        b = dot(radial_offset, radial_direction)
        parallel = a == 0.0
        safe_a = jnp.where(parallel, 1.0, a)

        # Closest approach of the segment's line to the axis, as a vector difference rather
        # than as the difference of two squares -- see the class docstring.
        closest = radial_offset - (b / safe_a)[..., None] * radial_direction
        discriminant = safe_a * (self.radius**2 - norm_squared(closest))
        misses_radially = (discriminant < 0.0) | parallel
        root = jnp.sqrt(jnp.where(misses_radially, 0.0, discriminant))
        enter = (-b - root) / safe_a
        exit_ = (-b + root) / safe_a

        # A segment running parallel to the axis never crosses the curved surface; it is inside
        # the infinite cylinder for its whole length, or outside it for the whole length.
        inside_radially = norm_squared(radial_offset) <= self.radius**2
        enter = jnp.where(parallel, jnp.where(inside_radially, -jnp.inf, jnp.inf), enter)
        exit_ = jnp.where(parallel, jnp.where(inside_radially, jnp.inf, -jnp.inf), exit_)

        # Clip to the flat ends, which are two more parallel planes on the axis.
        moving_along = along_axis != 0.0
        safe_along = jnp.where(moving_along, along_axis, 1.0)
        cap_one = (self.half_length - offset_along) / safe_along
        cap_two = (-self.half_length - offset_along) / safe_along
        cap_enter = jnp.where(moving_along, jnp.minimum(cap_one, cap_two), -jnp.inf)
        cap_exit = jnp.where(moving_along, jnp.maximum(cap_one, cap_two), jnp.inf)
        within_ends = jnp.abs(offset_along) <= self.half_length
        cap_enter = jnp.where(moving_along, cap_enter, jnp.where(within_ends, -jnp.inf, jnp.inf))
        cap_exit = jnp.where(moving_along, cap_exit, jnp.where(within_ends, jnp.inf, -jnp.inf))

        enter = jnp.maximum(enter, cap_enter)
        exit_ = jnp.minimum(exit_, cap_exit)

        # The hit must overlap the segment itself, excluding the sliver next to the source.
        near = jnp.asarray(min_distance) / length
        return (~misses_radially | parallel) & (exit_ >= jnp.maximum(enter, near)) & (enter <= 1.0)
