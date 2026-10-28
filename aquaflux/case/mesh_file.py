"""A mesh file: how to mesh a region of a STEP drawing, and where to write the mesh.

A mesh is made once, by ``aquaflux mesh mesh.yaml``, and read by any number of case files after
(:class:`~aquaflux.case.AquafluxMesh`): a flow case and a radiation case on one mesh, say. The file
holds what the mesher needs -- the drawing, the region of it to mesh, the patches, the sizes, the
layers, the walls meshed as structured quadrilaterals, the pipes and ducts, a hex core, the parts
meshed finer (refinement regions) and the planarizer --
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
    refinements:
      - {kind: RefinedNear, patches: [lamp], distance: 0.02, size: 0.004}
      - {kind: RefinedBox, lower: [0.2, -0.1, 0.0], upper: [0.4, 0.1, 0.2], size: 0.005}
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
from aquaflux.mesh import Mesh, PeriodicJoin
from aquaflux.meshing import (
    CadMesher,
    DuctCore,
    FacePlanarizer,
    FaceSelection,
    NetgenProcess,
    OctreeCore,
    PipeCore,
    PipeGrading,
    PrismLayers,
    Refinement,
    Stage,
)

from .drawing import CadPlacement

__all__ = ["Drawing", "DrawingRegion", "DrawingSections", "MeshOutput", "MeshSizes", "MeshSpec"]


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
    """The region of the drawing to mesh: the union of some solids, less others, with baffles
    standing in it.

    The water in a vessel and its pipes, less a lamp sleeve standing in it.

    Attributes
    ----------
    solids : tuple of str
        The solids whose union is the region, by name in the drawing.
    without : tuple of str
        The solids taken out of it. Default: none.
    baffles : tuple of str, or None
        The sheets of the drawing standing in the region as baffles -- surfaces of no thickness,
        the fluid on both sides. Unset, every sheet standing in it; empty, none.
    """

    solids: tuple[str, ...]
    without: tuple[str, ...] = ()
    baffles: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.solids:
            raise ValueError("a mesh file's region names the solids whose union is the region.")
        object.__setattr__(self, "solids", tuple(self.solids))
        object.__setattr__(self, "without", tuple(self.without))
        if self.baffles is not None:
            object.__setattr__(self, "baffles", tuple(self.baffles))

    def cad_region(self) -> CadRegion:
        """The drawing's region these name."""
        return CadRegion(self.solids, self.without, self.baffles)


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
class DrawingSections:
    """The parts of a mesh file its patches are chosen by, read while the rest may be unfinished.

    Attributes
    ----------
    drawing : Drawing
    region : DrawingRegion or None
        Unset until the file names a solid.
    patches : mapping of {str: FaceSelection}
    others : str or None
        The patch for the faces no selection chooses; unset, the mesher's default.
    feature_angle : float or None
        The file's ``sizes.feature_angle``, which decides which edges are a duct's corners; unset
        until the file states a positive one.
    """

    drawing: Drawing
    region: DrawingRegion | None
    patches: Mapping[str, FaceSelection]
    others: str | None = None
    feature_angle: float | None = None


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
    pipe_grading : PipeGrading or None
        How a straight pipe's cross-sections grow longer away from what it joins -- a bend,
        another pipe, a vessel -- toward an open end or its middle; unset, every pipe's
        cross-sections are about as far apart as the cells round it are wide.
    ducts : tuple of str or None
        The patches whose faces are the sides of rectangular ducts -- four four-sided faces round
        each stretch, along a straight line or a curve, with nothing inside -- their walls meshed as
        grids and their cores as hexahedra, cross-section by cross-section. Unset, every duct found;
        empty, none.
    duct_core : DuctCore or None
        How the ducts' cores are filled; unset, the defaults of
        :class:`~aquaflux.meshing.DuctCore`.
    planarizer : FacePlanarizer or None
        Flattens the mesh's faces by moving the cells' interior vertices, never making a cell less
        convex, or its volume more dependent on how its faces are triangulated, or a face worse,
        than the worst the mesh started with, nor a convex cell concave; unset, the faces are left
        as made.
    core : OctreeCore or None
        A core of hexahedra from an octree inside the region, kept clear of the walls and of the
        layers' top, joined to the rest through polyhedra; its unset wall size is ``max_size``.
        Unset, the core is polyhedral throughout.
    refinements : tuple of Refinement
        Parts of the region meshed finer than ``max_size``, each at its own size: inside a box, a
        ball or a cylinder (:class:`~aquaflux.meshing.RefinedBox`,
        :class:`~aquaflux.meshing.RefinedSphere`, :class:`~aquaflux.meshing.RefinedCylinder`), or
        within a distance of some patches (:class:`~aquaflux.meshing.RefinedNear`). The surface,
        the tetrahedra and the core's cubes are all made smaller there. Default: none.
    periodic : PeriodicJoin or None
        Two patches joined into one periodic seam once the region is meshed -- the two ends of a
        length of pipe whose outflow is its own inflow, say -- so neither is a patch of the mesh
        written. They must be meshed alike, each the other displaced by the period: the ends of a
        straight pipe are. Unset, the mesh is not periodic.
    output : MeshOutput
        Where the mesh is written; by default ``mesh.aqmesh`` beside the mesh file.

    Raises
    ------
    ValueError
        If a size, the angle or the grading is not a positive, finite number; layers are given
        without a patch to grow on, or patches are named without layers; the core's size is
        below an unset wall size; or a refinement's size is not below ``max_size``, or is below a
        set ``min_size``; or a periodic mesh is to be written as an OpenFOAM polyMesh, which holds
        no periodic seam.
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
    pipe_grading: PipeGrading | None = None
    ducts: tuple[str, ...] | None = None
    duct_core: DuctCore | None = None
    planarizer: FacePlanarizer | None = None
    core: OctreeCore | None = None
    refinements: tuple[Refinement, ...] = ()
    periodic: PeriodicJoin | None = None
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
        object.__setattr__(self, "refinements", tuple(self.refinements))
        for name in ("pipes", "ducts"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(getattr(self, name)))
        if (self.layers is None) != (not self.layered):
            raise ValueError(
                "a mesh file grows layers on the patches it names in layered, so it takes both "
                "layers and layered, or neither."
            )
        if self.periodic is not None and self.output.openfoam is not None:
            raise ValueError(
                "output.openfoam: an OpenFOAM polyMesh is written without periodic seams, so a "
                "periodic mesh is written only as an .aqmesh file. Remove one of the two."
            )
        self.mesher()  # refuses a size, angle or grading out of range

    def mesher(self) -> CadMesher:
        """The mesher these settings describe."""
        options = {
            name: value
            for name in ("min_size", "grading")
            if (value := getattr(self.sizes, name)) is not None
        }
        for name in ("pipe_core", "pipe_grading", "duct_core"):
            if (value := getattr(self, name)) is not None:
                options[name] = value
        return CadMesher(
            max_size=self.sizes.max_size, feature_angle=self.sizes.feature_angle,
            layers=self.layers, planarizer=self.planarizer, core=self.core,
            refinements=self.refinements, **options,
        )  # fmt: skip

    def mesh(self, directory: Path, on_stage: Stage | None = None) -> Mesh:
        """The region, read from its drawing and meshed; its periodic patches are still patches
        (:meth:`joined` joins them).

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
            layered=self.layered, mapped=self.mapped, pipes=self.pipes, ducts=self.ducts,
            on_stage=on_stage,
            **options,
        )  # fmt: skip

    def joined(self, mesh: Mesh) -> Mesh:
        """``mesh`` with :attr:`periodic`'s patches joined, or as it is when unset.

        Parameters
        ----------
        mesh : Mesh
            The mesh :meth:`mesh` made.

        Returns
        -------
        Mesh
        """
        return mesh if self.periodic is None else self.periodic.join(mesh)
