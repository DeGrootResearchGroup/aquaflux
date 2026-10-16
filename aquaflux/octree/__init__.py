"""Adaptive sampling of a field on an octree, and a continuous interpolant of what was sampled.

A field that is costly to evaluate and smooth almost everywhere -- the fluence rate of a lamp in an
absorbing medium, whose value at each point is an independent sum over the lamp -- is wasted on a
uniform grid: most points land where the field is nearly linear, and the error sits in a thin layer
where it is not. This package samples such a field on an octree instead:

- :class:`Octree` -- a box of cubic root cells, each halved as often as it is asked, every position
  on one integer lattice so that cells share corners exactly;
- :class:`Hierarchy` -- the interpolant, a sum of hat functions in which each node carries its
  **surplus**: how far the coarser tree missed its value. Continuous wherever a fine cell meets a
  coarse one, and through every sampled node;
- :func:`leaf_cuts` -- the points where cell edges cross the boundary of the region the field lives
  in, a standoff inside it, so the layer beside a boundary is sampled rather than extrapolated;
- :func:`sample_adaptively` -- the refinement: split, sample only the new points, measure each one's
  miss against the tree before it, and split again where a miss exceeds the tolerance. Returns a
  :class:`SampledField`, which interpolates to any points in the box.

The region is a solid in the sense of :mod:`aquaflux.solids`, used only through its methods, and the
field through a callable, so nothing here knows what is being sampled. Build-time numpy throughout:
nothing is traced or differentiated.
"""

from __future__ import annotations

from .cuts import LeafCuts, crossing_points, exit_points, leaf_cuts, merged_intervals
from .hierarchy import Hierarchy
from .ghosts import ghost_values
from .sampling import Linear, Logarithm, Round, SampledField, sample_adaptively
from .tree import Octree

__all__ = [
    "Hierarchy",
    "LeafCuts",
    "Linear",
    "Logarithm",
    "Octree",
    "Round",
    "SampledField",
    "crossing_points",
    "exit_points",
    "ghost_values",
    "leaf_cuts",
    "merged_intervals",
    "sample_adaptively",
]
