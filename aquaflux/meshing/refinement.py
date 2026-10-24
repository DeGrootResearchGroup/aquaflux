"""Parts of a region meshed finer than the rest: refinement regions.

A :class:`Refinement` names a part of the region and the size wanted in it: a volume
(:class:`RefinedBox`, :class:`RefinedSphere`, :class:`RefinedCylinder`) or the fluid within a
distance of some boundary patches (:class:`RefinedNear`). Sizes are measured as the mesher's own
``max_size`` is -- the edge of a tetrahedron or of a cube of the hexahedral core -- so the cells,
each the dual of a point, come out somewhat larger.

**One setting serves every part of the mesh that sizes cells.** Located on the region's walls
(:meth:`Refinement.located`), a refinement becomes a :class:`LocatedRefinement`, which is

- a refinement criterion of the octree the hexahedral core is cut from: a cube reaching into the
  part is split until it is no larger than the size, so the cubes there are the largest
  ``size / 2**k`` not over it, and balance grades them back out;
- points for Netgen (:meth:`LocatedRefinement.local_sizes`): the centres of the cubes of exactly the
  size reaching into the part, over the region's box, so the surface triangles and the tetrahedra
  there are no larger than the size either, grading away from it.

The two must agree because the octree core always meets the walls through a band of tetrahedra:
refining one alone would leave a jump in size where they meet.

**Whether a cube reaches into the part errs towards yes.** A volume is asked through its signed
distance -- a box reaches it when the distance at the box's centre is within half its diagonal,
which is exact for a ball and conservative for anything else, since the distance a solid reports
outside it is never more than the true one -- and the fluid near patches through the patches'
triangles registered in a grid of voxels, the box grown by the distance (so a box near a patch's
corner may be reached a little beyond the distance, along the diagonal).
"""

from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from aquaflux.octree import Octree, signed_distance
from aquaflux.solids import Box, Cylinder, Solid, Sphere
from aquaflux.triangle_grid import TriangleGrid

from .netgen_process import LocalSizes
from .regions import grid_meets, triangle_grid

__all__ = [
    "LocatedRefinement",
    "RefinedBox",
    "RefinedCylinder",
    "RefinedNear",
    "RefinedSphere",
    "Refinement",
]

#: How far a cube's edge may exceed a size, relatively, and still count as that size.
_SIZE_TOLERANCE = 1e-9


class _Reach(Protocol):
    """Which boxes reach into a part of space."""

    def meets(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Whether each axis-aligned box from ``lower`` to ``upper`` (each ``(m, 3)``) is taken
        to reach into the part, ``(m,)``; never False where it does."""


@dataclasses.dataclass(frozen=True)
class _InsideBody:
    """The boxes reaching into a solid: the solid's signed distance at the box's centre is within
    half the box's diagonal."""

    body: Solid

    def meets(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        centre = 0.5 * (lower + upper)
        half_diagonal = 0.5 * np.linalg.norm(upper - lower, axis=1)
        return signed_distance(self.body, centre) <= half_diagonal


@dataclasses.dataclass(frozen=True)
class _NearTriangles:
    """The boxes reaching within ``distance`` of some triangles: the box grown by the distance on
    every side holds a voxel a triangle is registered in."""

    grid: TriangleGrid
    distance: float

    def meets(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return grid_meets(self.grid, lower - self.distance, upper + self.distance)


@dataclasses.dataclass(frozen=True)
class LocatedRefinement:
    """A refinement placed in the region: the part of space it covers, and the size wanted there.

    Attributes
    ----------
    reach : object
        Which boxes reach into the part: ``reach.meets(lower, upper)`` for boxes ``(m, 3)``,
        never False for one that does.
    size : float
        The size wanted in the part.
    """

    reach: _Reach
    size: float

    def refines(self, lower: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Split each cube larger than :attr:`size` that reaches into the part -- a refinement
        criterion of :meth:`aquaflux.octree.Octree.refined_by`.

        Parameters
        ----------
        lower : np.ndarray, shape ``(n, 3)``
            Each cube's lower corner.
        size : np.ndarray, shape ``(n,)``
            Each cube's edge.

        Returns
        -------
        np.ndarray of bool, shape ``(n,)``
        """
        large = size > self.size * (1.0 + _SIZE_TOLERANCE)
        refines = np.zeros(size.shape, dtype=bool)
        if np.any(large):
            refines[large] = self.reach.meets(lower[large], lower[large] + size[large, None])
        return refines

    def local_sizes(self, lower, upper) -> LocalSizes:
        """The size at the centre of each cube of exactly :attr:`size` reaching into the part,
        over the box from ``lower`` to ``upper``.

        The cubes are the leaves of an octree over the box whose root cubes are :attr:`size`
        doubled as often as the box needs, refined by :meth:`refines`: so only the cubes the part
        reaches are ever split, and the points cost the part's volume over the size cubed, not the
        box's.

        Parameters
        ----------
        lower, upper : array_like, shape ``(3,)``
            Opposite corners of the region's box.

        Returns
        -------
        LocalSizes
            Points about :attr:`size` apart covering the part where it lies in the box; none where
            it lies wholly outside.
        """
        lower = np.asarray(lower, dtype=float)
        extent = float(np.max(np.asarray(upper, dtype=float) - lower))
        doublings = max(0, math.ceil(math.log2(extent / self.size) - _SIZE_TOLERANCE))
        tree = Octree.lattice(lower, upper, self.size * 2.0**doublings).refined_by([self])
        smallest = np.flatnonzero(tree.size <= self.size * (1.0 + _SIZE_TOLERANCE))
        corner = tree.lower[smallest]
        reached = smallest[self.reach.meets(corner, corner + tree.size[smallest, None])]
        centres = tree.lower[reached] + 0.5 * tree.size[reached, None]
        return LocalSizes(centres, np.full(len(reached), self.size))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Refinement(abc.ABC):
    """A part of the region meshed finer than the rest.

    Attributes
    ----------
    size : float
        The size wanted in the part, in the model's units: the edge of a tetrahedron or of a cube
        of the hexahedral core, as the mesher's ``max_size`` is.
    """

    size: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.size) and self.size > 0):
            raise ValueError(
                f"{type(self).__name__}.size must be a positive, finite number, got {self.size!r}."
            )

    def patches_named(self) -> tuple[str, ...]:
        """The boundary patches this refinement is placed by, so a name that is not a patch can be
        refused before anything is meshed; none unless it says otherwise."""
        return ()

    @abc.abstractmethod
    def located(self, walls: Mapping[str, np.ndarray]) -> LocatedRefinement:
        """This refinement placed in the region.

        Parameters
        ----------
        walls : mapping of {str: np.ndarray}
            Each boundary patch's triangles, by name, as their corners: shape ``(n, 3, 3)``.

        Returns
        -------
        LocatedRefinement

        Raises
        ------
        ValueError
            If it names a patch the region does not have.
        """


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RefinedVolume(Refinement):
    """A refinement of the fluid inside a solid body."""

    @abc.abstractmethod
    def body(self) -> Solid:
        """The body whose inside is refined."""

    def located(self, walls: Mapping[str, np.ndarray]) -> LocatedRefinement:
        """The fluid inside :meth:`body` -- see :meth:`Refinement.located`."""
        return LocatedRefinement(_InsideBody(self.body()), self.size)


def _point(value, owner: str, name: str) -> np.ndarray:
    """``value`` as a finite point, refused by name otherwise."""
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{owner}.{name} must be three finite coordinates, got {value!r}.")
    return point


def _positive(value: float, owner: str, name: str) -> None:
    """Refuse ``value`` unless it is a positive, finite number."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{owner}.{name} must be a positive, finite number, got {value!r}.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class RefinedBox(_RefinedVolume):
    """The fluid inside an axis-aligned box, meshed at ``size``.

    Attributes
    ----------
    lower, upper : tuple of float
        Opposite corners of the box, three coordinates each, ``upper`` above ``lower`` along every
        axis.
    size : float
        As :attr:`Refinement.size`.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        lower = _point(self.lower, "RefinedBox", "lower")
        upper = _point(self.upper, "RefinedBox", "upper")
        if not np.all(upper > lower):
            raise ValueError(
                f"RefinedBox.upper {self.upper!r} must lie above lower {self.lower!r} along every "
                "axis."
            )

    def body(self) -> Solid:
        """The box."""
        lower, upper = np.asarray(self.lower, float), np.asarray(self.upper, float)
        return Box(0.5 * (lower + upper), 0.5 * (upper - lower))


@dataclasses.dataclass(frozen=True, kw_only=True)
class RefinedSphere(_RefinedVolume):
    """The fluid inside a ball, meshed at ``size``.

    Attributes
    ----------
    centre : tuple of float
        The ball's centre, three coordinates.
    radius : float
        Its radius.
    size : float
        As :attr:`Refinement.size`.
    """

    centre: tuple[float, ...]
    radius: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _point(self.centre, "RefinedSphere", "centre")
        _positive(self.radius, "RefinedSphere", "radius")

    def body(self) -> Solid:
        """The ball."""
        return Sphere(np.asarray(self.centre, float), self.radius)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RefinedCylinder(_RefinedVolume):
    """The fluid inside a cylinder, meshed at ``size``: around a lamp, along a jet.

    Attributes
    ----------
    start, end : tuple of float
        The centres of the cylinder's two flat ends, three coordinates each, apart.
    radius : float
        Its radius.
    size : float
        As :attr:`Refinement.size`.
    """

    start: tuple[float, ...]
    end: tuple[float, ...]
    radius: float

    def __post_init__(self) -> None:
        super().__post_init__()
        start = _point(self.start, "RefinedCylinder", "start")
        end = _point(self.end, "RefinedCylinder", "end")
        _positive(self.radius, "RefinedCylinder", "radius")
        if not np.linalg.norm(end - start) > 0:
            raise ValueError("RefinedCylinder.start and end must be apart.")

    def body(self) -> Solid:
        """The cylinder."""
        start, end = np.asarray(self.start, float), np.asarray(self.end, float)
        axis = end - start
        length = float(np.linalg.norm(axis))
        return Cylinder(0.5 * (start + end), axis / length, self.radius, 0.5 * length)


@dataclasses.dataclass(frozen=True, kw_only=True)
class RefinedNear(Refinement):
    """The fluid within ``distance`` of some boundary patches, meshed at ``size``: a lamp's
    surroundings, a baffle's wake.

    The distance is measured from the patches as triangulated, before any layers are grown, so a
    layered patch's refinement covers its layers and the fluid beyond them.

    Attributes
    ----------
    patches : tuple of str
        The patches, by name.
    distance : float
        How far from them the fluid is refined.
    size : float
        As :attr:`Refinement.size`.
    """

    patches: tuple[str, ...]
    distance: float

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "patches", tuple(self.patches))
        if not self.patches:
            raise ValueError("RefinedNear names the patches the fluid near is refined.")
        _positive(self.distance, "RefinedNear", "distance")

    def patches_named(self) -> tuple[str, ...]:
        """:attr:`patches` -- see :meth:`Refinement.patches_named`."""
        return self.patches

    def located(self, walls: Mapping[str, np.ndarray]) -> LocatedRefinement:
        """The fluid within :attr:`distance` of the patches -- see :meth:`Refinement.located`."""
        unknown = sorted(set(self.patches) - set(walls))
        if unknown:
            raise ValueError(
                f"RefinedNear names patch(es) {unknown}, which are not patches; the patches are "
                f"{sorted(walls)}"
            )
        corners = np.concatenate([walls[name] for name in self.patches]).reshape(-1, 3, 3)
        if not len(corners):
            raise ValueError(f"RefinedNear's patches {list(self.patches)} have no faces")
        return LocatedRefinement(
            _NearTriangles(triangle_grid(corners, self.size), self.distance), self.size
        )
