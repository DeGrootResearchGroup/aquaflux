"""A polyhedral mesh of a region of a computer-aided design (CAD) model.

:class:`CadMesher` meshes a region (some solids' union, less others) in these steps: the region is
written as a boundary-representation (BREP) file; Netgen triangulates its faces, each triangle
tagged by its face; the faces chosen are meshed again as structured quadrilaterals; layers are
grown inward from the walls chosen; each quadrilateral left facing the rest is capped with a
pyramid; a core of hexahedra may be placed inside, clear of the walls, its own quadrilaterals
capped too; and Netgen fills what is left with tetrahedra; all the cells become a mesh whose boundary
faces are in patches chosen by face (:class:`FaceSelection`); that mesh's polyhedral dual is built,
the region's sharp edges kept as features; and the dual's boundary vertices, which sit at the
boundary faces' centroids and the midpoints of their edges, are moved onto the region's faces and
sharp edges.
"""

from __future__ import annotations

import abc
import dataclasses
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux.io.cad import CadModel, CadRegion
from aquaflux.mesh import RESERVED_PATCH_NAMES, Mesh

from .apexes import ApexRelaxation
from .dual import PolyhedralDual
from .elements import element_mesh, face_keys, search_rows
from .hex_core import OctreeCore
from .layers import PrismLayers
from .mapped import MappedQuads, MappedSurface
from .netgen_process import NetgenProcess, SurfaceTriangulation
from .pipe_core import PipeCore
from .pipes import fill_pipes, find_pipes, pipe_ends
from .polygons import Polygons, PyramidCaps

__all__ = ["CadMesher", "FaceSelection", "FacesNear", "FacesOn", "RegionFaces"]


@dataclasses.dataclass(frozen=True)
class RegionFaces:
    """A region's faces as a selection sees them.

    Attributes
    ----------
    model : CadModel
    region : CadRegion
    samples : np.ndarray, shape ``(n_faces, 3)``
        One point on each face, in the order :meth:`~aquaflux.io.cad.CadModel.topology` numbers
        them.
    """

    model: CadModel
    region: CadRegion
    samples: np.ndarray

    @property
    def n_faces(self) -> int:
        """Number of faces."""
        return self.samples.shape[0]


@dataclasses.dataclass(frozen=True)
class FaceSelection(abc.ABC):
    """Some faces of a region, for a boundary patch."""

    @abc.abstractmethod
    def faces(self, region: RegionFaces) -> np.ndarray:
        """The faces selected, as indices into the region's faces, ascending and distinct.

        Raises
        ------
        ValueError
            If the selection finds no face.
        """


@dataclasses.dataclass(frozen=True)
class FacesNear(FaceSelection):
    """The face of the region nearest each of some points: a pipe's end, by a point on it.

    Attributes
    ----------
    points : tuple of tuple of float
        The points, three coordinates each, in the model's frame.
    """

    points: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError(f"FacesNear takes points of three coordinates, got {self.points!r}.")

    def faces(self, region: RegionFaces) -> np.ndarray:
        """The nearest face to each point -- see :meth:`FaceSelection.faces`."""
        landed = region.model.project(region.region, np.asarray(self.points, dtype=float))
        return np.unique(landed.face)


@dataclasses.dataclass(frozen=True)
class FacesOn(FaceSelection):
    """The faces of the region lying on the surface of some solids: a lamp taken out of the water.

    Attributes
    ----------
    solids : tuple of str
        The solids, by name in the model.
    tolerance : float or None
        How far a point of a face may lie from the solids' surface and still be on it, in the
        model's units. Unset, a millionth of the region's extent.
    """

    solids: tuple[str, ...]
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.solids:
            raise ValueError("FacesOn names the solids whose surface the faces lie on.")
        if self.tolerance is not None and not (
            math.isfinite(self.tolerance) and self.tolerance > 0
        ):
            raise ValueError(
                f"FacesOn.tolerance must be a positive, finite number, got {self.tolerance!r}."
            )

    def faces(self, region: RegionFaces) -> np.ndarray:
        """The faces on the solids' surface -- see :meth:`FaceSelection.faces`."""
        tolerance = self.tolerance
        if tolerance is None:
            extent = region.samples.max(axis=0) - region.samples.min(axis=0)
            tolerance = 1e-6 * float(np.linalg.norm(extent))
        distance = region.model.project(CadRegion(self.solids), region.samples).distance
        found = np.flatnonzero(distance <= tolerance)
        if not found.size:
            raise ValueError(f"no face of the region lies on the surface of {list(self.solids)}")
        return found


@dataclasses.dataclass(frozen=True)
class CadMesher:
    """Mesh a region of a CAD model as polyhedra: the dual of a Netgen tetrahedral fill, about an
    optional core of hexahedra.

    Each cell is the dual cell of one of the primal mesh's points (cut in pieces at a sharp concave
    edge), so the cells come out larger than the tetrahedra; ``max_size`` bounds the tetrahedra.

    Attributes
    ----------
    max_size : float
        The largest tetrahedron edge wanted, in the model's units: Netgen's target, which its
        triangles and tetrahedra exceed somewhat.
    feature_angle : float
        The fold, in radians, beyond which an edge of the region, or of the triangulated surface,
        is kept sharp (see :class:`~aquaflux.meshing.PolyhedralDual`).
    min_size : float or None
        The smallest triangle wanted; unset, Netgen's.
    grading : float or None
        How fast the size may change between neighbours, in ``(0, 1]``; unset, Netgen's.
    layers : PrismLayers or None
        The prism layers grown on the patches :meth:`mesh` is asked to layer; unset, none.
    cap_height : float
        The height of the pyramids that cap each quadrilateral the tetrahedra meet, over the
        quadrilateral's first edge (round a structured grid), in ``(0, 1/2)`` (see
        :class:`~aquaflux.meshing.PyramidCaps`), the hex core's caps included.
    pipe_core : PipeCore
        How the cores of pipes (see :meth:`mesh`) are filled.
    core : OctreeCore or None
        A core of hexahedra inside the fill, kept clear of the layers' top and the other walls,
        its unset wall size ``max_size``; unset, none, and the whole core is the tetrahedra's
        polyhedral dual.
    apex_relaxation : ApexRelaxation or None
        How the caps' apexes, wall and core alike, are moved once the tetrahedra are in; unset,
        they stay where the caps put them.
    netgen : NetgenProcess
        The Netgen it runs.
    """

    max_size: float
    feature_angle: float
    min_size: float | None = None
    grading: float | None = None
    layers: PrismLayers | None = None
    cap_height: float = 0.25
    pipe_core: PipeCore = dataclasses.field(default_factory=PipeCore)
    core: OctreeCore | None = None
    apex_relaxation: ApexRelaxation | None = None
    netgen: NetgenProcess = dataclasses.field(default_factory=NetgenProcess)

    def __post_init__(self) -> None:
        for name in ("max_size", "feature_angle", "min_size", "grading"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"CadMesher.{name} must be a positive, finite number, got {value!r}."
                )
        PyramidCaps.check_height(self.cap_height, "CadMesher.cap_height")
        if self.core is not None:
            self.core.resolved(self.max_size)  # refuses a core smaller than an unset wall size

    def mesh(
        self,
        model: CadModel,
        region: CadRegion,
        patches: Mapping[str, FaceSelection] | None = None,
        others: str = "walls",
        layered: Sequence[str] = (),
        mapped: Sequence[str] = (),
        pipes: Sequence[str] | None = None,
    ) -> Mesh:
        """The polyhedral mesh of ``region``.

        Parameters
        ----------
        model : CadModel
        region : CadRegion
        patches : mapping of {str: FaceSelection}, optional
            The boundary patches named for the faces selected.
        others : str
            The patch for the faces no selection names.
        layered : sequence of str
            The patches :attr:`layers` are grown on; where one meets a patch without layers, the
            layers slide along it.
        mapped : sequence of str
            The patches whose faces are meshed with structured quadrilaterals in place of triangles
            (see :class:`~aquaflux.meshing.MappedQuads`): each face a band or four-sided. Layered,
            they carry hexahedra.
        pipes : sequence of str, or None
            The patches whose faces are pipes: each a cylinder the fluid fills, bounded by two
            loops of equal counts (a multiple of four), with nothing else inside it. A pipe's face
            is meshed as a structured grid (as ``mapped``), layered if its patch is, and its core
            filled with a structured O-grid (:attr:`pipe_core`; see
            :mod:`~aquaflux.meshing.pipes`). Unset, every face that is one; empty, none.

        Returns
        -------
        Mesh
            Its boundary in the patches named, and in ``others`` where any face is left.

        Raises
        ------
        ValueError
            If a patch name is reserved or given twice, a selection finds no face, two selections
            share one, a layered or mapped patch is not a patch, a patch is layered without
            :attr:`layers`, a mapped patch's face is neither a band nor four-sided, or a face of a
            patch named in ``pipes`` is not a pipe.
        MeshingError
            If Netgen fails.
        """
        patches = dict(patches or {})
        names = [*patches, others]
        _check_names(patches, others)
        layered = list(layered)
        if layered and self.layers is None:
            raise ValueError("patches are named to be layered, but the mesher has no layers")
        mapped = list(mapped)
        named_pipes = None if pipes is None else list(pipes)
        for role, chosen in (
            ("layered", layered),
            ("mapped", mapped),
            ("pipes", named_pipes or ()),
        ):
            unknown = sorted(set(chosen) - set(names))
            if unknown:
                raise ValueError(
                    f"{role} patch(es) {unknown} are not patches; the patches are {names}"
                )
        topology = model.topology(region)
        with tempfile.TemporaryDirectory() as directory:
            surface = self.netgen.triangulate(
                model.write_brep(region, Path(directory) / "region.brep"),
                max_size=self.max_size,
                **self._netgen_options(min_size=self.min_size),
            )
        patch_of_face = _patch_of_face(model, region, surface, topology.n_faces, patches, others)
        pipe_faces = _pipe_faces(topology, surface, patch_of_face, names, named_pipes)
        chosen = np.union1d(_faces_of(patch_of_face, names, mapped), pipe_faces)
        walls = self._mapped(model, region, topology, surface, chosen)
        on_layered = np.isin(patch_of_face[walls.face], [names.index(n) for n in layered])
        prisms = hexahedra = stacked = None
        sides = side_face = None
        if on_layered.any():
            grown = self.layers.extrude(walls.points, walls.faces, on_layered)
            points, fill, wall_faces = grown.points, grown.inner, grown.walls
            prisms, hexahedra, stacked = grown.prisms, grown.hexahedra, grown.columns
            sides, side_face = grown.sides, walls.face[grown.side_face]
        else:
            points, fill, wall_faces = walls.points, walls.faces, walls.faces
        wall_tag = walls.face
        if len(pipe_faces):
            pipes_in = self._pipes(topology, walls, pipe_faces, on_layered, stacked)
            points, cores, fill, wall_faces, wall_tag = fill_pipes(
                self.pipe_core, points, fill, wall_faces, wall_tag, pipes_in
            )
            hexahedra = cores if hexahedra is None else np.concatenate([hexahedra, cores])
        boundary = _Boundary.of(wall_faces, wall_tag, sides, side_face)
        # The tetrahedra meet triangles only: each quadrilateral left for them is capped on their side.
        caps = PyramidCaps.on(points, fill.quads[:, ::-1], self.cap_height)
        apexes = points.shape[0] + np.arange(caps.apexes.shape[0])
        points = np.vstack([points, caps.apexes])
        points, surface, core = self._cored(points, np.vstack([fill.triangles, caps.sides]))
        points, tetrahedra = self._filled(points, surface, core.fill_size)
        tetrahedra = np.vstack([tetrahedra, core.tetrahedra])
        # The layers' hexahedra first, so a column's cells are numbered as the layers grew them.
        hexahedra = core.hexahedra if hexahedra is None else np.vstack([hexahedra, core.hexahedra])
        pyramids = np.vstack([caps.pyramids, core.pyramids])
        if self.apex_relaxation is not None:
            points = self.apex_relaxation.relaxed(
                points, tetrahedra, pyramids, np.concatenate([apexes, core.apexes])
            ).points
        layered_walls = walls.faces.where(on_layered)
        if len(pipe_faces):
            # An open pipe end's own points inside the layers' ring are left on no cell.
            cells = [tetrahedra, prisms, hexahedra, pyramids]
            points, renumber = _compacted(points, [c for c in cells if c is not None])
            tetrahedra, prisms, hexahedra, pyramids = (
                None if c is None else renumber(c) for c in cells
            )
            boundary = boundary.renumbered(renumber)
            layered_walls = layered_walls.renumbered(renumber(np.arange(walls.points.shape[0])))
            stacked = None if stacked is None else renumber(stacked)
        primal = element_mesh(
            points, tetrahedra, prisms, hexahedra, pyramids,
            patches=boundary.patches(patch_of_face, names),
        )  # fmt: skip
        columns = None
        if prisms is not None:
            columns = _Columns.of(
                primal, layered_walls, tetrahedra.shape[0], prisms.shape[0], self.layers
            )
        edges = _SurfaceEdges.of(boundary)
        feature_pairs = {
            tuple(sorted(topology.faces_of_edge(edge)[:2].tolist()))
            for edge in topology.feature_edges(self.feature_angle)
            if topology.faces_of_edge(edge).size >= 2
        }
        dual = PolyhedralDual(
            self.feature_angle, feature_edges=edges.between(feature_pairs), columns=stacked
        )
        traced = dual.build_traced(primal)
        return _onto_the_drawing(traced, primal, boundary, edges, model, region, topology, columns)

    def _pipes(self, topology, walls: MappedSurface, pipe_faces, on_layered, stacked):
        """Each pipe's ends, and its rings round the core: the layers' top over its grid, where it
        is layered, else the grid itself."""
        top_of = np.arange(walls.points.shape[0])
        if stacked is not None:
            top_of[stacked[:, 0]] = stacked[:, -1]
        pipes = []
        for f in np.asarray(pipe_faces).tolist():
            grid = walls.grids[f]
            ends = pipe_ends(topology, grid, f, walls.faces, walls.face, on_layered)
            pipes.append((ends, top_of[grid]))
        return pipes

    @staticmethod
    def _mapped(model, region, topology, surface, chosen) -> MappedSurface:
        """The triangulated surface, with the ``chosen`` faces remeshed as structured
        quadrilaterals, each new point moved onto its face."""
        if not chosen.size:
            return MappedSurface(surface.points, Polygons(surface.triangles), surface.face)
        return MappedQuads().remesh(
            surface.points, surface.triangles, surface.face, chosen, topology.vertices,
            lambda points, faces: model.project(region, points, faces=faces).points,
        )  # fmt: skip

    def _cored(self, points: np.ndarray, surface: np.ndarray):
        """``points`` and the fill's closed ``surface``, with the hex core inside the surface:
        its points after ``points``, its capped surface added to the fill's as a hole, and its
        cells (:class:`_CoreCells`, numbered after ``points``)."""
        if self.core is None:
            return points, surface, _CoreCells.none(self.max_size)
        core = self.core.within(
            points, surface, cap_height=self.cap_height, boundary_size=self.max_size
        )
        wall, _ = self.core.resolved(self.max_size)
        start = points.shape[0]
        cells = _CoreCells(
            tetrahedra=start + core.tetrahedra,
            hexahedra=start + core.hexahedra,
            pyramids=start + np.vstack([core.pyramids, core.caps]),
            apexes=start + core.caps[:, 4],
            fill_size=max(self.max_size, wall),
        )
        return np.vstack([points, core.points]), np.vstack([surface, start + core.surface]), cells

    def _filled(
        self, points: np.ndarray, triangles: np.ndarray, max_size: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """``points`` with the fill's new points after them, and the fill's tetrahedra, none
        over ``max_size``.

        Netgen is given only the points the surface uses (it renumbers when one is not), and its
        tetrahedra are put back in the numbering of ``points``. With no surface left (every cell
        structured), Netgen is not run.
        """
        if not len(triangles):
            return points, np.zeros((0, 4), dtype=np.int64)  # nothing left for tetrahedra
        used = np.unique(triangles)
        position = np.full(points.shape[0], -1, dtype=np.int64)
        position[used] = np.arange(used.size)
        filled = self.netgen.fill(
            points[used], position[triangles], max_size=max_size, **self._netgen_options()
        )
        added = filled.points[used.size :]
        back = np.concatenate([used, points.shape[0] + np.arange(added.shape[0])])
        return np.vstack([points, added]), back[filled.tetrahedra]

    def _netgen_options(self, **extra) -> dict:
        """The Netgen settings that are set, to pass through."""
        options = {"grading": self.grading, **extra}
        return {name: value for name, value in options.items() if value is not None}


@dataclasses.dataclass(frozen=True)
class _CoreCells:
    """The hex core's cells in the mesher's numbering, and the largest tetrahedron the fill
    around it may have: the core's wall cubes, where they are larger than the mesher's size, so
    the tetrahedra next to the core are not held below its cubes.

    Attributes
    ----------
    tetrahedra : np.ndarray of int, shape ``(n, 4)``
    hexahedra : np.ndarray of int, shape ``(n, 8)``
    pyramids : np.ndarray of int, shape ``(n, 5)``
        The transition cubes' pyramids, then the caps on the core's boundary.
    apexes : np.ndarray of int, shape ``(n_caps,)``
        The caps' apexes.
    fill_size : float
    """

    tetrahedra: np.ndarray
    hexahedra: np.ndarray
    pyramids: np.ndarray
    apexes: np.ndarray
    fill_size: float

    @classmethod
    def none(cls, fill_size: float) -> _CoreCells:
        """No core: no cells, and the fill at ``fill_size``."""
        empty = np.zeros((0,), np.int64)
        return cls(empty.reshape(0, 4), empty.reshape(0, 8), empty.reshape(0, 5), empty, fill_size)


def _compacted(points: np.ndarray, cells: list[np.ndarray]):
    """``points`` without those no cell uses, and the map from the old numbering to the new."""
    used = np.zeros(points.shape[0], dtype=bool)
    for corners in cells:
        used[np.asarray(corners).reshape(-1)] = True
    position = np.cumsum(used) - 1

    def renumber(indices):
        return position[np.asarray(indices, dtype=np.int64)]

    return points[used], renumber


def _faces_of(patch_of_face: np.ndarray, names, chosen) -> np.ndarray:
    """The faces of the patches named in ``chosen``."""
    return np.flatnonzero(np.isin(patch_of_face, [names.index(n) for n in chosen]))


def _pipe_faces(topology, surface: SurfaceTriangulation, patch_of_face, names, named) -> np.ndarray:
    """The pipes' faces: every face that is one, or (``named`` given) those of the patches named,
    each of which must be one."""
    candidates = (
        np.arange(topology.n_faces) if named is None else _faces_of(patch_of_face, names, named)
    )
    found, refused = find_pipes(
        topology, surface.points, surface.triangles, surface.face, candidates
    )
    if named is not None and refused:
        face, reason = next(iter(refused.items()))
        patch = names[patch_of_face[face]]
        raise ValueError(f"face {face} of patch {patch!r} is not a pipe: {reason}")
    return np.asarray(found, dtype=np.int64)


def _check_names(patches, others: str) -> None:
    """Refuse a reserved patch name, or ``others`` naming a selection too."""
    reserved = sorted(({*patches, others}) & set(RESERVED_PATCH_NAMES))
    if reserved:
        raise ValueError(f"patch name(s) {reserved} are reserved for the mesh's own groups")
    if others in patches:
        raise ValueError(f"{others!r} names both a selection and the faces no selection names")


def _patch_of_face(model, region, surface: SurfaceTriangulation, n_faces, patches, others):
    """Each face's patch, as its position among the selections (``others`` last)."""
    # A point on each face: the centroid of one of its triangles, moved onto the face.
    first = np.full(n_faces, -1, dtype=np.int64)
    first[surface.face[::-1]] = np.arange(surface.face.size)[::-1]
    if np.any(first < 0):
        raise ValueError(f"{int(np.count_nonzero(first < 0))} face(s) were given no triangle")
    centroids = surface.points[surface.triangles[first]].mean(axis=1)
    samples = model.project(region, centroids, faces=np.arange(n_faces)).points
    faces = RegionFaces(model, region, samples)
    patch_of_face = np.full(n_faces, len(patches), dtype=np.int64)
    taken = np.zeros(n_faces, dtype=bool)
    for index, (name, selection) in enumerate(patches.items()):
        chosen = selection.faces(faces)
        if np.any(taken[chosen]):
            raise ValueError(f"patch {name!r} selects a face another patch has already")
        taken[chosen] = True
        patch_of_face[chosen] = index
    return patch_of_face


@dataclasses.dataclass(frozen=True)
class _Boundary:
    """The primal's boundary faces, triangles and quadrilaterals, with the drawing's face under each.

    The quadrilaterals are a mapped wall's, and the layer cells' sides lying on a face without
    layers.
    """

    triangles: np.ndarray
    triangle_face: np.ndarray
    quads: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 4), np.int64))
    quad_face: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, np.int64))

    @classmethod
    def of(cls, faces: Polygons, face: np.ndarray, sides=None, side_face=None) -> _Boundary:
        """The boundary of ``faces``, tagged by ``face``, and of any further quadrilaterals."""
        triangle_face, quad_face = faces.split(face)
        quads, under = [faces.quads], [quad_face]
        if sides is not None:
            quads.append(sides)
            under.append(side_face)
        return cls(faces.triangles, triangle_face, np.concatenate(quads), np.concatenate(under))

    def renumbered(self, renumber) -> _Boundary:
        """The same faces with their points renumbered by ``renumber``."""
        return _Boundary(
            renumber(self.triangles), self.triangle_face, renumber(self.quads), self.quad_face
        )

    def rings(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each kind of face with the drawing's faces under it."""
        return [(np.asarray(self.triangles, np.int64), np.asarray(self.triangle_face, np.int64)),
                (np.asarray(self.quads, np.int64), np.asarray(self.quad_face, np.int64))]  # fmt: skip

    def patches(self, patch_of_face: np.ndarray, names: list[str]) -> dict[str, list[np.ndarray]]:
        """The faces of each patch that has any, by the patch of the drawing's face under each."""
        found = {}
        for index, name in enumerate(names):
            mine = [faces[patch_of_face[under] == index] for faces, under in self.rings()]
            if any(len(part) for part in mine):
                found[name] = mine
        return found


@dataclasses.dataclass(frozen=True)
class _SurfaceEdges:
    """Each edge of a closed boundary once, with the drawing's faces of the two faces beside it.

    Attributes
    ----------
    points : np.ndarray of int, shape ``(n_edges, 2)``
        The edge's two points, ascending; rows in ascending order.
    faces : np.ndarray of int, shape ``(n_edges, 2)``
        The drawing's faces under the two faces beside it, ascending (equal inside a face).
    """

    points: np.ndarray
    faces: np.ndarray

    @classmethod
    def of(cls, boundary: _Boundary) -> _SurfaceEdges:
        corners, face = [], []
        for rings, under in boundary.rings():
            if not len(rings):
                continue
            corners.append(np.stack([rings, np.roll(rings, -1, axis=1)], axis=2).reshape(-1, 2))
            face.append(np.repeat(under, rings.shape[1]))
        corners, face = np.concatenate(corners), np.concatenate(face)
        keys = np.sort(corners, axis=1)
        order = np.lexsort(keys.T[::-1])
        keys, face = keys[order], face[order]
        # A closed boundary has each edge on exactly two faces, so sorted keys come in pairs.
        pair = np.sort(np.stack([face[0::2], face[1::2]], axis=1), axis=1)
        return cls(keys[0::2], pair)

    def between(self, pairs: set[tuple[int, int]]) -> np.ndarray:
        """The edges between two faces whose (ascending) pair is in ``pairs``, as point pairs."""
        bound = int(self.faces.max()) + 1
        wanted = np.array([a * bound + b for a, b in pairs], dtype=np.int64)
        chosen = np.isin(self.faces[:, 0] * bound + self.faces[:, 1], wanted)
        return self.points[chosen & (self.faces[:, 0] != self.faces[:, 1])]

    def faces_of(self, points: np.ndarray) -> np.ndarray:
        """The two faces beside each edge given by its points (ascending), shape ``(n, 2)``."""
        bound = int(self.points.max()) + 1
        keys = self.points[:, 0] * bound + self.points[:, 1]
        position = np.searchsorted(keys, points[:, 0] * bound + points[:, 1])
        return self.faces[position]


def _onto_the_drawing(
    traced, primal, boundary, edges, model, region, topology, columns=None
) -> Mesh:
    """The dual with each boundary vertex the primal boundary approximates moved onto the drawing.

    A boundary face's centroid goes onto the drawing's face under it; a vertex on a boundary edge,
    onto the face under its two boundary faces or, where they lie on two faces, onto the drawing's
    edge between them (the nearest, where they share several). A primal point is on the drawing
    already and stays. Above a layered wall face, the layer cells' centroids -- the dual's vertices
    in the layers -- move with the face's centroid, by ``columns.weight`` of its displacement, so
    the dual's layers keep their heights from the true wall rather than from the face's chord.
    """
    coordinates = np.array(traced.mesh.node_coords, dtype=np.float64)
    of_face = _drawing_faces(primal, boundary)
    at_face = np.flatnonzero(traced.vertex_face >= 0)
    at_edge = np.flatnonzero(traced.vertex_edge[:, 0] >= 0)
    beside = edges.faces_of(traced.vertex_edge[at_edge])
    inside = beside[:, 0] == beside[:, 1]
    onto_faces = np.concatenate([at_face, at_edge[inside]])
    landed = model.project(
        region,
        coordinates[onto_faces],
        faces=np.concatenate([of_face[traced.vertex_face[at_face]], beside[inside, 0]]),
    )
    if columns is not None:
        moved = landed.points[: at_face.size] - coordinates[at_face]
        column = columns.of_face[traced.vertex_face[at_face]]
        layered = column >= 0
        for k, weight in enumerate(columns.weight):
            coordinates[columns.cells[column[layered], k]] += weight * moved[layered]
    coordinates[onto_faces] = landed.points
    crossing, pairs = at_edge[~inside], beside[~inside]
    shared = _edges_shared(topology)
    for pair in np.unique(pairs, axis=0):
        mine = crossing[np.all(pairs == pair, axis=1)]
        candidates = shared.get(tuple(pair.tolist()), [])
        if not candidates:
            continue  # faces meeting only at a point: the vertex stays on its chord
        # Each candidate from where the vertex was, not from where an earlier one put it.
        start, best = coordinates[mine].copy(), np.full(mine.size, np.inf)
        for candidate in candidates:
            moved = model.project(region, start, edges=np.full(mine.size, candidate))
            nearer = moved.distance < best
            best[nearer] = moved.distance[nearer]
            coordinates[mine[nearer]] = moved.points[nearer]
    return eqx.tree_at(lambda mesh: mesh.node_coords, traced.mesh, jnp.asarray(coordinates))


@dataclasses.dataclass(frozen=True)
class _Columns:
    """The layer cells above each layered wall face, and how much of the face's move each follows.

    Attributes
    ----------
    of_face : np.ndarray of int, shape ``(n_faces,)``
        The column of each primal face that is a layered wall face, else ``-1``.
    cells : np.ndarray of int, shape ``(n_columns, n_layers)``
        The primal cell of each layer of each column, wall first.
    weight : np.ndarray, shape ``(n_layers,)``
        The share of the wall face's move each cell's centroid takes: ``1 - (c / t) ** 2`` for a
        centroid ``c`` above the wall in layers ``t`` thick, so the layers next to the wall move
        nearly with it and the top of the layers, which the tetrahedra meet, barely at all.
    """

    of_face: np.ndarray
    cells: np.ndarray
    weight: np.ndarray

    @classmethod
    def of(cls, primal, walls: Polygons, first_prism: int, n_prisms: int, layers) -> _Columns:
        """The columns over ``walls`` (the layered faces, in the order the layers were grown),
        whose cells are the ``n_prisms`` prisms from cell ``first_prism`` on, then the hexahedra,
        each kind layer by layer."""
        n_layers = layers.count
        n_triangles, n_quads = walls.triangles.shape[0], walls.quads.shape[0]
        first_hexahedron = first_prism + n_prisms
        cells = np.concatenate([
            first_prism + np.arange(n_layers * n_triangles).reshape(n_layers, n_triangles).T,
            first_hexahedron + np.arange(n_layers * n_quads).reshape(n_layers, n_quads).T,
        ])  # fmt: skip
        neighbour = np.asarray(primal.face_cells.neighbour)
        on_boundary = np.flatnonzero(neighbour < 0)
        rings = _boundary_rings(primal, on_boundary)
        keys = face_keys(rings, primal.n_nodes)
        order = np.lexsort(keys.T[::-1])
        found = order[search_rows(keys[order], face_keys(walls.padded(), primal.n_nodes))]
        of_face = np.full(primal.n_faces, -1, dtype=np.int64)
        of_face[on_boundary[found]] = np.arange(walls.n_faces)
        tops = layers.offsets()
        centroids = 0.5 * (np.concatenate([[0.0], tops[:-1]]) + tops)
        return cls(of_face, cells, 1.0 - (centroids / tops[-1]) ** 2)


def _boundary_rings(primal: Mesh, faces: np.ndarray) -> np.ndarray:
    """The rings of some primal faces, padded to four corners with ``-1``."""
    offsets = np.asarray(primal.face_nodes.offsets)
    nodes = np.asarray(primal.face_nodes.face_node_indices, dtype=np.int64)
    count = offsets[faces + 1] - offsets[faces]
    rings = np.full((faces.size, 4), -1, dtype=np.int64)
    for size in (3, 4):
        chosen = np.flatnonzero(count == size)
        rows = offsets[faces[chosen]][:, None] + np.arange(size)
        rings[chosen, :size] = nodes[rows]
    return rings


def _drawing_faces(primal: Mesh, boundary: _Boundary) -> np.ndarray:
    """The drawing's face under each primal boundary face (``-1`` inside), matched by corners."""
    on_boundary = np.flatnonzero(np.asarray(primal.face_cells.neighbour) < 0)
    rings = _boundary_rings(primal, on_boundary)
    n_points = primal.n_nodes
    given, under = [], []
    for faces, face in boundary.rings():
        padded = np.full((faces.shape[0], 4), -1, dtype=np.int64)
        padded[:, : faces.shape[1]] = faces
        given.append(face_keys(padded, n_points))
        under.append(face)
    given, under = np.concatenate(given), np.concatenate(under)
    order = np.lexsort(given.T[::-1])
    found = order[search_rows(given[order], face_keys(rings, n_points))]
    faces = np.full(primal.n_faces, -1, dtype=np.int64)
    faces[on_boundary] = under[found]
    return faces


def _edges_shared(topology) -> dict[tuple[int, int], list[int]]:
    """The drawing's edges between each pair of faces (ascending), seams and slits left out."""
    shared: dict[tuple[int, int], list[int]] = {}
    for edge in range(topology.n_edges):
        faces = np.unique(topology.faces_of_edge(edge))
        if faces.size == 2:
            shared.setdefault(tuple(faces.tolist()), []).append(edge)
    return shared
