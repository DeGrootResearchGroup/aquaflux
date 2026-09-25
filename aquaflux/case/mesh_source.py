"""Where a case's mesh comes from.

A case file names its mesh by source rather than holding it: a mesh file to read
(:class:`OpenFOAMMesh`), or a structured grid to generate (:class:`StructuredGrid`). Relative paths are
relative to the directory the case file sits in, so a case and its mesh can be moved together.
"""

from __future__ import annotations

import abc
import dataclasses
import math
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np

from aquaflux.io import OpenFOAMReader
from aquaflux.mesh import Mesh, graded_nodes, structured_grid_2d

__all__ = ["AxisGrading", "GeometricGrading", "MeshSource", "OpenFOAMMesh", "StructuredGrid"]

#: The axes of a two-dimensional structured grid, in coordinate order.
_AXES_2D = ("x", "y")


@dataclasses.dataclass(frozen=True)
class MeshSource(abc.ABC):
    """A mesh a case reads or generates."""

    @abc.abstractmethod
    def read(self, directory: Path) -> Mesh:
        """The mesh, with any relative path taken from ``directory``.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the case file sits in.

        Returns
        -------
        Mesh
            The mesh.
        """


@dataclasses.dataclass(frozen=True)
class OpenFOAMMesh(MeshSource):
    """An OpenFOAM polyMesh, read with :class:`~aquaflux.io.OpenFOAMReader`.

    A case that is one cell thick between ``empty`` patches reads as a two-dimensional mesh, and those
    patches are gone from it -- so a case file never names them.

    Attributes
    ----------
    path : str
        The polyMesh directory, or an OpenFOAM case directory holding ``constant/polyMesh``; relative to
        the case file's directory unless absolute.
    cyclic_match_tolerance : float or None
        The tolerance for matching the faces of a ``cyclic`` patch pair; unset, the reader's own
        default.

    Raises
    ------
    ValueError
        If ``path`` is empty, or the tolerance is not a positive, finite number.
    """

    path: str
    cyclic_match_tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("an OpenFOAM mesh needs the path to its polyMesh or case directory.")
        tolerance = self.cyclic_match_tolerance
        if tolerance is not None and not (math.isfinite(tolerance) and tolerance > 0):
            raise ValueError(
                f"cyclic_match_tolerance must be a positive, finite number, got {tolerance!r}."
            )

    def reader(self, directory: Path) -> OpenFOAMReader:
        """The reader for this mesh, with a relative :attr:`path` taken from ``directory``.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the case file sits in.

        Returns
        -------
        OpenFOAMReader
            The reader; nothing is read until its ``read()`` is called.
        """
        options = (
            {}
            if self.cyclic_match_tolerance is None
            else {"cyclic_match_tolerance": self.cyclic_match_tolerance}
        )
        return OpenFOAMReader(Path(directory) / self.path, **options)

    def read(self, directory: Path) -> Mesh:
        """The polyMesh, read -- see :meth:`MeshSource.read`."""
        return self.reader(directory).read()


@dataclasses.dataclass(frozen=True)
class AxisGrading(abc.ABC):
    """How the cells along one axis of a structured grid are sized. One implementation: :class:`GeometricGrading`."""

    @abc.abstractmethod
    def nodes(self, n: int, length: float) -> np.ndarray:
        """The node coordinates along the axis.

        Parameters
        ----------
        n : int
            Number of cells along the axis.
        length : float
            The axis length.

        Returns
        -------
        np.ndarray
            ``n + 1`` increasing positions spanning ``[0, length]``.
        """


@dataclasses.dataclass(frozen=True)
class GeometricGrading(AxisGrading):
    """Cells growing geometrically away from the wall(s): finest at the ends, coarsest inside.

    The wall-normal grading of a wall-resolved boundary layer, by :func:`~aquaflux.mesh.graded_nodes`.

    Attributes
    ----------
    growth : float
        The size ratio of adjacent cells, ``> 0``; ``1`` is uniform, and ``> 1`` clusters cells toward
        the wall(s).
    both_sides : bool or None
        Graded toward both ends of the axis (a channel between two walls) or toward its start only;
        unset, both.

    Raises
    ------
    ValueError
        If ``growth`` is not a positive, finite number.
    """

    growth: float
    both_sides: bool | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.growth) and self.growth > 0):
            raise ValueError(
                f"GeometricGrading.growth must be a positive, finite number, got {self.growth!r}."
            )

    def nodes(self, n: int, length: float) -> np.ndarray:
        """The graded node coordinates -- see :meth:`AxisGrading.nodes`."""
        options = {} if self.both_sides is None else {"both_sides": self.both_sides}
        return graded_nodes(n, length, self.growth, **options)


@dataclasses.dataclass(frozen=True)
class StructuredGrid(MeshSource):
    """A structured quadrilateral grid, generated rather than read: the box ``[0, lx] x [0, ly]``.

    Its boundary patches are named by side -- ``left`` (x = 0), ``right`` (x = lx), ``bottom``
    (y = 0) and ``top`` (y = ly) -- except on a periodic axis, whose two sides are fused into an
    interior seam and so are not patches at all: a grid periodic in ``x`` has only ``bottom`` and
    ``top``.

    Attributes
    ----------
    cells : tuple of int
        The number of cells along x and y.
    lengths : tuple of float
        The box's extent along x and y.
    periodic : tuple of {"x"}
        The axes that wrap around; a streamwise-periodic channel is periodic in ``x``. A periodic axis
        needs at least two cells.
    grading : mapping of {str: AxisGrading}
        How the cells are sized along each named axis (``"x"`` or ``"y"``); an axis left out is
        uniform.

    Raises
    ------
    ValueError
        If the grid is not two-dimensional, a count or length is not positive, an axis is named twice
        as periodic, a periodic axis has fewer than two cells, or a grading names an axis the grid
        does not have.
    """

    cells: tuple[int, ...]
    lengths: tuple[float, ...]
    periodic: tuple[Literal["x"], ...] = ()
    grading: Mapping[str, AxisGrading] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.cells) != 2 or len(self.lengths) != 2:
            raise ValueError(
                "a structured grid is two-dimensional, so it takes two cell counts and two lengths, "
                f"got {self.cells!r} and {self.lengths!r}."
            )
        # A reader may hand back a whole number as a float; the count is still a count.
        object.__setattr__(
            self,
            "cells",
            tuple(int(n) if isinstance(n, float) and n.is_integer() else n for n in self.cells),
        )
        if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in self.cells):
            raise ValueError(f"a structured grid's cell counts must be >= 1, got {self.cells!r}.")
        if not all(math.isfinite(length) and length > 0 for length in self.lengths):
            raise ValueError(
                f"a structured grid's lengths must be positive, finite numbers, got {self.lengths!r}."
            )
        if len(set(self.periodic)) != len(self.periodic):
            raise ValueError(f"an axis is named twice as periodic: {self.periodic!r}.")
        for axis in self.periodic:
            if self.cells[_AXES_2D.index(axis)] < 2:
                raise ValueError(
                    f"a periodic axis needs at least two cells, or its one cell would couple itself; "
                    f"{axis} has {self.cells[_AXES_2D.index(axis)]}."
                )
        unknown = sorted(set(self.grading) - set(_AXES_2D))
        if unknown:
            raise ValueError(f"a structured grid's grading names axes {_AXES_2D}, got {unknown}.")
        for axis, grading in self.grading.items():
            if not isinstance(grading, AxisGrading):
                raise TypeError(
                    f"StructuredGrid.grading[{axis!r}] must be an axis grading such as "
                    f"GeometricGrading(growth), got {grading!r}."
                )
        object.__setattr__(self, "grading", types.MappingProxyType(dict(self.grading)))

    def read(self, directory: Path) -> Mesh:
        """The generated grid -- see :meth:`MeshSource.read`. ``directory`` is unused: nothing is read."""
        del directory
        (nx, ny), (lx, ly) = self.cells, self.lengths
        nodes = {
            f"{axis}_nodes": grading.nodes(n, length)
            for axis, n, length in zip(_AXES_2D, self.cells, self.lengths, strict=True)
            if (grading := self.grading.get(axis)) is not None
        }
        return structured_grid_2d(
            nx, ny, lx, ly, named_boundaries=True, periodic=tuple(self.periodic), **nodes
        )
