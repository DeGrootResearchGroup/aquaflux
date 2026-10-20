"""Projections onto analytic surfaces: each moves points to the nearest point of its surface.

A block face is moved onto a true surface by any callable taking points ``(n, 3)`` and returning
the points on the surface. These are the analytic ones; a face of a computer-aided design (CAD)
model is reached through the model's own projection.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["ConeSurface", "CylinderSurface", "PlaneSurface", "SphereSurface"]


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


@dataclasses.dataclass(frozen=True)
class PlaneSurface:
    """The projection onto the plane through ``point`` with ``normal``, along the normal.

    Attributes
    ----------
    point : np.ndarray, shape ``(3,)``
    normal : np.ndarray, shape ``(3,)``
        Any length.
    """

    point: np.ndarray
    normal: np.ndarray

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """The points moved onto the plane, shape ``(n, 3)``."""
        normal = _unit(self.normal)
        points = np.asarray(points, dtype=np.float64)
        height = (points - np.asarray(self.point, dtype=np.float64)) @ normal
        return points - height[:, None] * normal


@dataclasses.dataclass(frozen=True)
class CylinderSurface:
    """The projection onto the infinite circular cylinder of ``radius`` about an axis.

    Each point moves straight away from (or toward) the axis, keeping its position along it.

    Attributes
    ----------
    radius : float
    centre : np.ndarray, shape ``(3,)``
        A point on the axis. Unset, the origin.
    axis : np.ndarray, shape ``(3,)``
        The axis direction, any length. Unset, ``z``.
    """

    radius: float
    centre: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """The points moved onto the cylinder, shape ``(n, 3)``."""
        axis = _unit(self.axis)
        centre = np.asarray(self.centre, dtype=np.float64)
        relative = np.asarray(points, dtype=np.float64) - centre
        along = (relative @ axis)[:, None] * axis
        across = relative - along
        return centre + along + self.radius * across / np.linalg.norm(across, axis=1)[:, None]


@dataclasses.dataclass(frozen=True)
class ConeSurface:
    """The projection onto the cone through two circles about one axis, to the nearest point.

    The cone (a frustum's side, extended without end) is the surface swept about the axis by the
    straight line joining radius ``first_radius`` at ``first`` to ``second_radius`` at ``second``.
    Each point moves to the nearest point of that line in its own half-plane through the axis.

    Attributes
    ----------
    first, second : np.ndarray, shape ``(3,)``
        Two distinct points on the axis.
    first_radius, second_radius : float
        The cone's radius at each.
    """

    first: np.ndarray
    second: np.ndarray
    first_radius: float
    second_radius: float

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """The points moved onto the cone, shape ``(n, 3)``."""
        first = np.asarray(self.first, dtype=np.float64)
        span = np.asarray(self.second, dtype=np.float64) - first
        length = float(np.linalg.norm(span))
        axis = span / length
        relative = np.asarray(points, dtype=np.float64) - first
        along = relative @ axis
        across = relative - along[:, None] * axis
        distance = np.linalg.norm(across, axis=1)
        outward = across / distance[:, None]
        # The generator, in the half-plane's (along, distance) coordinates.
        rise = self.second_radius - self.first_radius
        direction = np.array([length, rise]) / np.hypot(length, rise)
        reach = along * direction[0] + (distance - self.first_radius) * direction[1]
        foot_along = reach * direction[0]
        foot_distance = self.first_radius + reach * direction[1]
        return first + foot_along[:, None] * axis + foot_distance[:, None] * outward


@dataclasses.dataclass(frozen=True)
class SphereSurface:
    """The projection onto the sphere of ``radius`` about ``centre``, along each point's radius.

    Attributes
    ----------
    radius : float
    centre : np.ndarray, shape ``(3,)``
        Unset, the origin.
    """

    radius: float
    centre: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """The points moved onto the sphere, shape ``(n, 3)``."""
        centre = np.asarray(self.centre, dtype=np.float64)
        relative = np.asarray(points, dtype=np.float64) - centre
        return centre + self.radius * relative / np.linalg.norm(relative, axis=1)[:, None]
