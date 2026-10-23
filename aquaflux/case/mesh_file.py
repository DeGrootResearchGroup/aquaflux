"""A mesh file: how to mesh a region of a STEP drawing, and where to write the mesh.

A mesh is made once, by ``aquaflux mesh mesh.yaml``, and read by any number of case files after
(:class:`~aquaflux.case.AquafluxMesh`): a flow case and a radiation case on one mesh, say. The file
holds what the mesher needs -- the drawing, the region of it to mesh, the patches, the sizes, the
layers, the walls meshed as structured quadrilaterals, the pipes, a hex core and the planarizer --
and where the mesh is written: aquaflux's own ``.aqmesh`` file, and an OpenFOAM polyMesh too if
asked for. Relative paths are relative to the directory the mesh file sits in.

::

    drawing: {file: reactor.step}
    region: {solids: [vessel, inlet_pipe, outlet_pipe], without: [lamp]}
    sizes: {max_size: 0.01, feature_angle: 0.785}
    patches:
      inlet: {kind: FacesNear, points: [[0.05, 0.0, 0.89]]}
      lamp: {kind: FacesNear, points: [[0.4, 0.01, 0.0]]}
    layers: {kind: PrismLayers, first_height: 0.0005, growth: 1.2, count: 5}
    layered: [walls, lamp]
    mapped: [lamp]
    output: {path: reactor.aqmesh}

The drawing, region, sizes and output sections have one form each, so they name no ``kind``.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar

from aquaflux.io import AQMESH_SUFFIX
from aquaflux.io.cad import CadRegion, read_step
from aquaflux.mesh import Mesh
from aquaflux.meshing import (
    CadMesher,
    FacePlanarizer,
    FaceSelection,
    NetgenProcess,
    OctreeCore,
    PipeCore,
    PrismLayers,
    Stage,
)

from .drawing import CadPlacement

__all__ = ["Drawing", "DrawingRegion", "MeshOutput", "MeshSizes", "MeshSpec"]


@dataclasses.dataclass(frozen=True)
class Drawing:
    """The STEP drawing a mesh is made from.

    Attributes
    ----------
    file : str
        The STEP file, relative to the mesh file's directory unless absolute.
    placement : CadPlacement or None
        Where the drawing sits in the mesh's frame; unset, where it was drawn.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("file",)

    file: str
    placement: CadPlacement | None = None

    def __post_init__(self) -> None:
        if not self.file:
            raise ValueError("a mesh file's drawing names its STEP file.")


@dataclasses.dataclass(frozen=True)
class DrawingRegion:
    """The region of the drawing to mesh: the union of some solids, less others.

    The water in a vessel and its pipes, less a lamp sleeve standing in it.

    Attributes
    ----------
    solids : tuple of str
        The solids whose union is the region, by name in the drawing.
    without : tuple of str
        The solids taken out of it. Default: none.
    """

    solids: tuple[str, ...]
    without: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.solids:
            raise ValueError("a mesh file's region names the solids whose union is the region.")
        object.__setattr__(self, "solids", tuple(self.solids))
        object.__setattr__(self, "without", tuple(self.without))

    def cad_region(self) -> CadRegion:
        """The drawing's region these name."""
        return CadRegion(self.solids, self.without)


@dataclasses.dataclass(frozen=True)
class MeshSizes:
    """How large the mesh's cells are, and which edges stay sharp.

    Netgen fills the region with tetrahedra, and their polyhedral dual is the mesh: one cell per
    tetrahedron point, so the cells are larger than ``max_size``.

    Attributes
    ----------
    max_size : float
        The largest tetrahedron edge wanted, in metres.
    feature_angle : float
        The fold, in radians, beyond which an edge of the boundary is kept sharp.
    min_size : float or None
        The smallest triangle wanted, in metres; unset, Netgen's.
    grading : float or None
        How fast the size may change between neighbouring cells, in ``(0, 1]``; unset, Netgen's.
    """

    max_size: float
    feature_angle: float
    min_size: float | None = None
    grading: float | None = None

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (NetgenProcess.triangulate,)


@dataclasses.dataclass(frozen=True)
class MeshOutput:
    """Where ``aquaflux mesh`` writes the mesh.

    Attributes
    ----------
    path : str
        The ``.aqmesh`` file, relative to the mesh file's directory unless absolute.
    openfoam : str or None
        An OpenFOAM case directory to write the mesh into as well, as its ``constant/polyMesh``;
        unset, none.

    Raises
    ------
    ValueError
        If ``path`` does not name a ``.aqmesh`` file.
    """

    path_fields: ClassVar[tuple[str, ...]] = ("path", "openfoam")

    path: str = "mesh" + AQMESH_SUFFIX
    openfoam: str | None = None

    def __post_init__(self) -> None:
        if Path(self.path).suffix != AQMESH_SUFFIX:
            raise ValueError(f"a mesh is written to a {AQMESH_SUFFIX} file, got {self.path!r}.")


@dataclasses.dataclass(frozen=True)
class MeshSpec:
    """A mesh file: a region of a STEP drawing meshed as polyhedra (:class:`~aquaflux.meshing.CadMesher`).

    Its boundary patches are named for the faces chosen by each selection
    (:class:`~aquaflux.meshing.FacesNear`, the face nearest a point;
    :class:`~aquaflux.meshing.FacesOn`, the faces on some solids' surface), and the faces left over
    make one more patch.

    Needs the optional CAD kernel and mesher (``pip install aquaflux[cad,meshing]``).

    Attributes
    ----------
    drawing : Drawing
        The STEP drawing.
    region : DrawingRegion
        The region of it to mesh.
    sizes : MeshSizes
        The cells' sizes and the feature angle.
    patches : mapping of {str: FaceSelection}
        The boundary patches, each named for the faces its selection chooses. Default: none.
    others : str or None
        The patch for the faces no selection chooses; unset, ``walls``.
    layers : PrismLayers or None
        Prism layers grown on the ``layered`` patches, sized for the solver's cells; unset, none.
    layered : tuple of str
        The patches the layers grow on. Where one meets a patch without layers, the layers slide
        along it. Default: none.
    mapped : tuple of str
        The patches whose faces are meshed with structured quadrilaterals rather than triangles,
        each face a band (a pipe's wall, a cone, a ring) or four-sided; layered, they carry hexahedra.
        Default: none.
    pipes : tuple of str or None
        The patches whose faces are pipes -- each a tube the fluid fills (straight, bent, coiled,
        tapering or swept along any curve), with nothing inside -- their walls meshed as grids and their cores as structured O-grids,
        one O-grid through pipes joined end to end. Unset, every face that
        is one; empty, none.
    pipe_core : PipeCore or None
        How the pipes' cores are filled; unset, the defaults of
        :class:`~aquaflux.meshing.PipeCore`.
    planarizer : FacePlanarizer or None
        Flattens the mesh's faces by moving the cells' interior vertices, never making a cell less
        convex, or its volume more dependent on how its faces are triangulated, or a face worse,
        than the worst the mesh started with, nor a convex cell concave; unset, the faces are left
        as made.
    core : OctreeCore or None
        A core of hexahedra from an octree inside the region, kept clear of the walls and of the
        layers' top, joined to the rest through polyhedra; its unset wall size is ``max_size``.
        Unset, the core is polyhedral throughout.
    output : MeshOutput
        Where the mesh is written; by default ``mesh.aqmesh`` beside the mesh file.

    Raises
    ------
    ValueError
        If a size, the angle or the grading is not a positive, finite number; layers are given
        without a patch to grow on, or patches are named without layers; or the core's size is
        below an unset wall size.
    """

    drawing: Drawing
    region: DrawingRegion
    sizes: MeshSizes
    patches: Mapping[str, FaceSelection] = dataclasses.field(default_factory=dict)
    others: str | None = None
    layers: PrismLayers | None = None
    layered: tuple[str, ...] = ()
    mapped: tuple[str, ...] = ()
    pipes: tuple[str, ...] | None = None
    pipe_core: PipeCore | None = None
    planarizer: FacePlanarizer | None = None
    core: OctreeCore | None = None
    output: MeshOutput = dataclasses.field(default_factory=MeshOutput)

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (CadMesher.mesh,)

    def __post_init__(self) -> None:
        for name, selection in self.patches.items():
            if not isinstance(selection, FaceSelection):
                raise TypeError(
                    f"MeshSpec.patches[{name!r}] must be a face selection such as FacesNear(points) "
                    f"or FacesOn(solids), got {selection!r}."
                )
        object.__setattr__(self, "patches", types.MappingProxyType(dict(self.patches)))
        object.__setattr__(self, "layered", tuple(self.layered))
        object.__setattr__(self, "mapped", tuple(self.mapped))
        if self.pipes is not None:
            object.__setattr__(self, "pipes", tuple(self.pipes))
        if (self.layers is None) != (not self.layered):
            raise ValueError(
                "a mesh file grows layers on the patches it names in layered, so it takes both "
                "layers and layered, or neither."
            )
        self.mesher()  # refuses a size, angle or grading out of range

    def mesher(self) -> CadMesher:
        """The mesher these settings describe."""
        options = {
            name: value
            for name in ("min_size", "grading")
            if (value := getattr(self.sizes, name)) is not None
        }
        if self.pipe_core is not None:
            options["pipe_core"] = self.pipe_core
        return CadMesher(
            max_size=self.sizes.max_size, feature_angle=self.sizes.feature_angle,
            layers=self.layers, planarizer=self.planarizer, core=self.core, **options,
        )  # fmt: skip

    def mesh(self, directory: Path, on_stage: Stage | None = None) -> Mesh:
        """The region, read from its drawing and meshed.

        Parameters
        ----------
        directory : pathlib.Path
            The directory the mesh file sits in, which relative paths are taken from.
        on_stage : callable, optional
            Told the name of each stage as it starts: reading the drawing, then the mesher's (see
            :meth:`~aquaflux.meshing.CadMesher.mesh`).

        Returns
        -------
        Mesh
        """
        if on_stage is not None:
            on_stage("reading the drawing")
        placement = self.drawing.placement
        model = read_step(
            Path(directory) / self.drawing.file,
            None if placement is None else placement.placement(),
        )
        options = {} if self.others is None else {"others": self.others}
        return self.mesher().mesh(
            model, self.region.cad_region(), dict(self.patches),
            layered=self.layered, mapped=self.mapped, pipes=self.pipes, on_stage=on_stage,
            **options,
        )  # fmt: skip
