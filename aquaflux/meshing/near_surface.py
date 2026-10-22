"""Refining an octree towards a region's boundary: the cubes it passes near are split.

:class:`NearSurface` is a refinement criterion for :meth:`aquaflux.octree.Octree.refined_by`; it asks the
region which cubes, grown by a margin, its boundary meets, and :func:`cube_samples` is the lattice of
points a region samples a cube by when it has no exact answer.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .regions import Region

__all__ = ["NearSurface"]


def cube_samples(lower: np.ndarray, size: np.ndarray, per_edge: int, margin: float) -> np.ndarray:
    """A lattice of ``per_edge`` points along each axis of each cube grown by ``margin`` on every
    side, corners included; shape ``(n, per_edge**3, 3)``."""
    ticks = np.linspace(0.0, 1.0, per_edge)
    unit = np.array(list(itertools.product(ticks, repeat=3)))
    grown = size[:, None, None] + 2.0 * margin
    return lower[:, None, :] - margin + grown * unit[None, :, :]


@dataclasses.dataclass(frozen=True)
class NearSurface:
    """Split every cube a region's boundary passes near until it is no larger than ``size``.

    Attributes
    ----------
    region : Region
        The region (see :class:`~aquaflux.meshing.ClosedSurface`,
        :class:`~aquaflux.meshing.SampledRegion`); a cube is split where its boundary is taken to
        pass through the cube grown by ``margin`` on each side.
    size : float
        The largest cube kept where the boundary passes.
    margin : float
        How far beyond each cube to look for the boundary.
    """

    region: Region
    size: float
    margin: float = 0.0

    def refines(self, lower: np.ndarray, size: np.ndarray) -> np.ndarray:
        large = size > self.size * (1.0 + 1e-9)
        refines = np.zeros(size.shape, dtype=bool)
        if np.any(large):
            low = lower[large] - self.margin
            refines[large] = self.region.meets(low, low + size[large, None] + 2.0 * self.margin)
        return refines
