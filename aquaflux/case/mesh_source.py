"""Where a case's mesh comes from.

A case file names its mesh by source rather than holding it: a mesh file to read
(:class:`OpenFOAMMesh`), a structured grid to generate (:class:`StructuredGrid`), or a region of a STEP
drawing to mesh as polyhedra (:class:`CadMesh`). Relative paths are relative to the directory the case
file sits in, so a case and its mesh can be moved together.
"""

from __future__ import annotations

import abc
import dataclasses
import math
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from aquaflux.io import OpenFOAMReader
from aquaflux.io.cad import CadRegion, read_step
from aquaflux.mesh import Mesh, graded_nodes, structured_grid_2d, structured_grid_3d
from aquaflux.meshing import (
    CadMesher,
    FacePlanarizer,
    FaceSelection,
    NetgenProcess,
    PipeCore,
    PrismLayers,
)

from .drawing import CadPlacement

__all__ = [
    "AxisGrading",
    "CadMesh",
    "GeometricGrading",
    "MeshSource",
    "OpenFOAMMesh",
    "StructuredGrid",
]

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

    path_fields: ClassVar[tuple[str, ...]] = ("path",)

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

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (graded_nodes,)

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
    """A structured grid, generated rather than read: the box ``[0, lx] x [0, ly]``, or in three
    dimensions ``[0, lx] x [0, ly] x [0, lz]``.

    Its boundary patches are named by side -- ``left`` (x = 0), ``right`` (x = lx), ``bottom``
    (y = 0) and ``top`` (y = ly), and in three dimensions ``back`` (z = 0) and ``front`` (z = lz) --
    except on a periodic axis, whose two sides are fused into an interior seam and so are not patches
    at all: a grid periodic in ``x`` has only ``bottom`` and ``top``.

    A three-dimensional grid is uniform: it takes neither a periodic axis nor a grading.

    Attributes
    ----------
    cells : tuple of int
        The number of cells along each axis: two counts, or three.
    lengths : tuple of float
        The box's extent along each axis, one per count.
    periodic : tuple of {"x"}
        The axes that wrap around; a streamwise-periodic channel is periodic in ``x``. A periodic axis
        needs at least two cells. Two dimensions only.
    grading : mapping of {str: AxisGrading}
        How the cells are sized along each named axis (``"x"`` or ``"y"``); an axis left out is
        uniform. Two dimensions only.

    Raises
    ------
    ValueError
        If the grid is not two- or three-dimensional, a count or length is not positive, an axis is
        named twice as periodic, a periodic axis has fewer than two cells, a grading names an axis the
        grid does not have, or a three-dimensional grid is given a periodic axis or a grading.
    """

    cells: tuple[int, ...]
    lengths: tuple[float, ...]
    periodic: tuple[Literal["x"], ...] = ()
    grading: Mapping[str, AxisGrading] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.cells) not in (2, 3) or len(self.lengths) != len(self.cells):
            raise ValueError(
                "a structured grid takes two cell counts and two lengths, or three of each, got "
                f"{self.cells!r} and {self.lengths!r}."
            )
        if len(self.cells) == 3 and (self.periodic or self.grading):
            raise ValueError(
                "a three-dimensional structured grid is uniform, so it takes neither a periodic axis "
                "nor a grading."
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
        if len(self.cells) == 3:
            return structured_grid_3d(*self.cells, *self.lengths, named_boundaries=True)
        (nx, ny), (lx, ly) = self.cells, self.lengths
        nodes = {
            f"{axis}_nodes": grading.nodes(n, length)
            for axis, n, length in zip(_AXES_2D, self.cells, self.lengths, strict=True)
            if (grading := self.grading.get(axis)) is not None
        }
        return structured_grid_2d(
            nx, ny, lx, ly, named_boundaries=True, periodic=tuple(self.periodic), **nodes
        )


@dataclasses.dataclass(frozen=True)
class CadMesh(MeshSource):
    """A region of a STEP drawing, meshed as polyhedra (:class:`~aquaflux.meshing.CadMesher`).

    The region is the union of some solids less others -- the water in a vessel and its pipes, less a
    lamp sleeve standing in it. Netgen fills it with tetrahedra, and their polyhedral dual is the mesh:
    one cell per tetrahedron point, so the cells are larger than ``max_size``. Its boundary patches are
    named for the faces chosen by each selection (:class:`~aquaflux.meshing.FacesNear`, the face
    nearest a point; :class:`~aquaflux.meshing.FacesOn`, the faces on some solids' surface), and the
    faces left over make one more patch. The mesh is generated each time the case is read.

    Needs the optional CAD kernel and mesher (``pip install aquaflux[cad,meshing]``).

    Attributes
    ----------
    file : str
        The STEP file, relative to the case file.
    solids : tuple of str
        The solids whose union is the region, by name in the drawing.
    max_size : float
        The largest tetrahedron edge wanted, in metres.
    feature_angle : float
        The fold, in radians, beyond which an edge of the boundary is kept sharp.
    without : tuple of str
        The solids taken out of the region. Default: none.
    patches : mapping of {str: FaceSelection}
        The boundary patches, each named for the faces its selection chooses. Default: none.
    others : str or None
        The patch for the faces no selection chooses; unset, ``walls``.
    placement : CadPlacement or None
        Where the drawing sits in the case's frame; unset, where it was drawn.
    min_size : float or None
        The smallest triangle wanted, in metres; unset, Netgen's.
    grading : float or None
        How fast the size may change between neighbouring cells, in ``(0, 1]``; unset, Netgen's.
    layers : PrismLayers or None
        Prism layers grown on the ``layered`` patches, sized for the solver's cells; unset, none.
    layered : tuple of str
        The patches the layers grow on. Where one meets a patch without layers, the layers slide
        along it. Default: none.
    mapped : tuple of str
        The patches whose faces are meshed with structured quadrilaterals rather than triangles,
        each face a band (a cylinder, a cone, a ring) or four-sided; layered, they carry hexahedra.
        Default: none.
    pipes : tuple of str or None
        The patches whose faces are pipes -- each a cylinder the fluid fills, with nothing inside --
        their walls meshed as grids and their cores as structured O-grids. Unset, every face that
        is one; empty, none.
    pipe_core : PipeCore or None
        How the pipes' cores are filled; unset, the defaults of
        :class:`~aquaflux.meshing.PipeCore`.
    planarizer : FacePlanarizer or None
        Flattens the mesh's faces by moving the cells' interior vertices, never making a cell less
        convex or a face worse than the worst the mesh started with, nor a convex cell concave;
        unset, the faces are left as made.

    Raises
    ------
    ValueError
        If no solid is named; a size, the angle or the grading is not a positive, finite number; or
        layers are given without a patch to grow on, or patches are named without layers.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("file",)

    file: str
    solids: tuple[str, ...]
    max_size: float
    feature_angle: float
    without: tuple[str, ...] = ()
    patches: Mapping[str, FaceSelection] = dataclasses.field(default_factory=dict)
    others: str | None = None
    placement: CadPlacement | None = None
    min_size: float | None = None
    grading: float | None = None
    layers: PrismLayers | None = None
    layered: tuple[str, ...] = ()
    mapped: tuple[str, ...] = ()
    pipes: tuple[str, ...] | None = None
    pipe_core: PipeCore | None = None
    planarizer: FacePlanarizer | None = None

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (CadMesher.mesh, NetgenProcess.triangulate)

    def __post_init__(self) -> None:
        if not self.solids:
            raise ValueError("a CadMesh names the solids whose union is the region to mesh.")
        for name, selection in self.patches.items():
            if not isinstance(selection, FaceSelection):
                raise TypeError(
                    f"CadMesh.patches[{name!r}] must be a face selection such as FacesNear(points) "
                    f"or FacesOn(solids), got {selection!r}."
                )
        object.__setattr__(self, "patches", types.MappingProxyType(dict(self.patches)))
        object.__setattr__(self, "layered", tuple(self.layered))
        object.__setattr__(self, "mapped", tuple(self.mapped))
        if self.pipes is not None:
            object.__setattr__(self, "pipes", tuple(self.pipes))
        if (self.layers is None) != (not self.layered):
            raise ValueError(
                "a CadMesh grows layers on the patches it names in layered, so it takes both "
                "layers and layered, or neither."
            )
        self.mesher()  # refuses a size, angle or grading out of range

    def mesher(self) -> CadMesher:
        """The mesher these settings describe."""
        options = {
            name: value
            for name in ("min_size", "grading")
            if (value := getattr(self, name)) is not None
        }
        if self.pipe_core is not None:
            options["pipe_core"] = self.pipe_core
        return CadMesher(
            max_size=self.max_size, feature_angle=self.feature_angle, layers=self.layers,
            planarizer=self.planarizer, **options,
        )  # fmt: skip

    def read(self, directory: Path) -> Mesh:
        """The region, read from its drawing and meshed -- see :meth:`MeshSource.read`."""
        model = read_step(
            Path(directory) / self.file,
            None if self.placement is None else self.placement.placement(),
        )
        options = {} if self.others is None else {"others": self.others}
        return self.mesher().mesh(
            model, CadRegion(self.solids, self.without), dict(self.patches),
            layered=self.layered, mapped=self.mapped, pipes=self.pipes, **options,
        )  # fmt: skip
