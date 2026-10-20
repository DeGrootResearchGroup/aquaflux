"""A polyhedral mesh of a region of a computer-aided design (CAD) model.

:class:`CadMesher` meshes a region (some solids' union, less others) in five steps: the region is
written as a boundary-representation (BREP) file; Netgen triangulates its faces, each triangle
tagged by its face, and fills the surface with tetrahedra; the tetrahedra become a mesh whose
boundary faces are in patches chosen by face (:class:`FaceSelection`); that mesh's polyhedral dual
is built, the region's sharp edges kept as features; and the dual's boundary vertices, which sit at
the triangles' centroids and the midpoints of their edges, are moved onto the region's faces and
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

from .dual import PolyhedralDual
from .elements import element_mesh, face_keys, search_rows
from .layers import PrismLayers
from .netgen_process import NetgenProcess, SurfaceTriangulation

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
    """Mesh a region of a CAD model as polyhedra: the dual of a Netgen tetrahedral fill.

    Each cell is the dual cell of one of the tetrahedra's points (cut in pieces at a sharp concave
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
    netgen : NetgenProcess
        The Netgen it runs.
    """

    max_size: float
    feature_angle: float
    min_size: float | None = None
    grading: float | None = None
    layers: PrismLayers | None = None
    netgen: NetgenProcess = dataclasses.field(default_factory=NetgenProcess)

    def __post_init__(self) -> None:
        for name in ("max_size", "feature_angle", "min_size", "grading"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"CadMesher.{name} must be a positive, finite number, got {value!r}."
                )

    def mesh(
        self,
        model: CadModel,
        region: CadRegion,
        patches: Mapping[str, FaceSelection] | None = None,
        others: str = "walls",
        layered: Sequence[str] = (),
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

        Returns
        -------
        Mesh
            Its boundary in the patches named, and in ``others`` where any face is left.

        Raises
        ------
        ValueError
            If a patch name is reserved or given twice, a selection finds no face, two selections
            share one, or a layered patch is not a patch or is named without :attr:`layers`.
        MeshingError
            If Netgen fails.
        """
        patches = dict(patches or {})
        names = [*patches, others]
        _check_names(patches, others)
        layered = list(layered)
        if layered and self.layers is None:
            raise ValueError("patches are named to be layered, but the mesher has no layers")
        unknown = sorted(set(layered) - set(names))
        if unknown:
            raise ValueError(
                f"layered patch(es) {unknown} are not patches; the patches are {names}"
            )
        topology = model.topology(region)
        with tempfile.TemporaryDirectory() as directory:
            surface = self.netgen.triangulate(
                model.write_brep(region, Path(directory) / "region.brep"),
                max_size=self.max_size,
                **self._netgen_options(min_size=self.min_size),
            )
        patch_of_face = _patch_of_face(model, region, surface, topology.n_faces, patches, others)
        on_layered = np.isin(patch_of_face[surface.face], [names.index(n) for n in layered])
        if on_layered.any():
            grown = self.layers.extrude(surface.points, surface.triangles, on_layered)
            points, tetrahedra = self._filled(grown.points, grown.inner)
            prisms = grown.prisms
            boundary = _Boundary(
                grown.walls, surface.face, grown.sides, surface.face[grown.side_triangle]
            )
        else:
            points, tetrahedra = self._filled(surface.points, surface.triangles)
            prisms = None
            boundary = _Boundary(surface.triangles, surface.face)
        primal = element_mesh(
            points, tetrahedra, prisms, patches=boundary.patches(patch_of_face, names)
        )
        columns = None
        if prisms is not None:
            columns = _Columns.of(
                primal, surface.triangles[on_layered], tetrahedra.shape[0], self.layers
            )
        edges = _SurfaceEdges.of(boundary)
        feature_pairs = {
            tuple(sorted(topology.faces_of_edge(edge)[:2].tolist()))
            for edge in topology.feature_edges(self.feature_angle)
            if topology.faces_of_edge(edge).size >= 2
        }
        dual = PolyhedralDual(self.feature_angle, feature_edges=edges.between(feature_pairs))
        traced = dual.build_traced(primal)
        return _onto_the_drawing(traced, primal, boundary, edges, model, region, topology, columns)

    def _filled(self, points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``points`` with the fill's new points after them, and the fill's tetrahedra.

        Netgen is given only the points the surface uses (it renumbers when one is not), and its
        tetrahedra are put back in the numbering of ``points``.
        """
        used = np.unique(triangles)
        position = np.full(points.shape[0], -1, dtype=np.int64)
        position[used] = np.arange(used.size)
        filled = self.netgen.fill(
            points[used], position[triangles], max_size=self.max_size, **self._netgen_options()
        )
        added = filled.points[used.size :]
        back = np.concatenate([used, points.shape[0] + np.arange(added.shape[0])])
        return np.vstack([points, added]), back[filled.tetrahedra]

    def _netgen_options(self, **extra) -> dict:
        """The Netgen settings that are set, to pass through."""
        options = {"grading": self.grading, **extra}
        return {name: value for name, value in options.items() if value is not None}


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

    The quadrilaterals are prisms' sides lying on a face without layers.
    """

    triangles: np.ndarray
    triangle_face: np.ndarray
    quads: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 4), np.int64))
    quad_face: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, np.int64))

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

    A boundary face's centroid goes onto the face its triangle lies on; a vertex on a boundary
    edge, onto the face of its two triangles or, where they lie on two faces, onto the drawing's
    edge between them (the nearest, where they share several). A primal point is on the drawing
    already and stays. Above a layered wall face, the prisms' centroids -- the dual's vertices in
    the layers -- move with the face's centroid, by ``columns.weight`` of its displacement, so the
    dual's layers keep their heights from the true wall rather than from the triangle's chord.
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
    """The prisms above each layered wall face, and how much of the face's move each follows.

    Attributes
    ----------
    of_face : np.ndarray of int, shape ``(n_faces,)``
        The column of each primal face that is a layered wall triangle, else ``-1``.
    cells : np.ndarray of int, shape ``(n_columns, n_layers)``
        The primal cell of each prism of each column, wall first.
    weight : np.ndarray, shape ``(n_layers,)``
        The share of the wall face's move each prism's centroid takes: ``1 - (c / t) ** 2`` for a
        centroid ``c`` above the wall in layers ``t`` thick, so the layers next to the wall move
        nearly with it and the top of the layers, which the tetrahedra meet, barely at all.
    """

    of_face: np.ndarray
    cells: np.ndarray
    weight: np.ndarray

    @classmethod
    def of(cls, primal, walls, n_tetrahedra, layers) -> _Columns:
        """The columns over ``walls`` (the layered triangles, in the order the prisms were grown),
        whose prisms are the cells after the ``n_tetrahedra`` tetrahedra, layer by layer."""
        n_columns, n_layers = walls.shape[0], layers.count
        cells = n_tetrahedra + np.arange(n_layers * n_columns).reshape(n_layers, n_columns).T
        neighbour = np.asarray(primal.face_cells.neighbour)
        on_boundary = np.flatnonzero(neighbour < 0)
        rings = _boundary_rings(primal, on_boundary)
        keys = face_keys(rings, primal.n_nodes)
        order = np.lexsort(keys.T[::-1])
        padded = np.full((n_columns, 4), -1, dtype=np.int64)
        padded[:, :3] = walls
        found = order[search_rows(keys[order], face_keys(padded, primal.n_nodes))]
        of_face = np.full(primal.n_faces, -1, dtype=np.int64)
        of_face[on_boundary[found]] = np.arange(n_columns)
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
