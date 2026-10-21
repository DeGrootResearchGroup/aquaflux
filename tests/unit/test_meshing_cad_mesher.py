"""The CAD mesher's own work, with the CAD model and Netgen replaced by a unit cube's.

The model is a unit cube of six faces (numbered ``2 axis + side``: ``x-, x+, y-, y+, z-, z+``)
whose twelve edges fold by an angle the test chooses; Netgen hands back Kuhn tetrahedra of the cube,
their boundary triangles tagged by the face each lies on. So what is checked is the mesher's part:
patches chosen by face, the CAD's sharp edges reaching the dual, and the refusals -- each against
the cube's areas and volume.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.io.cad import CadRegion, CadTopology, Projection
from aquaflux.mesh import closed_cell_residual, face_warp
from aquaflux.meshing import (
    CadMesher,
    FacePlanarizer,
    FacesNear,
    FacesOn,
    SurfaceTriangulation,
    TetrahedralMesh,
    check_closed_surface,
)

from tests.support.meshes import tetrahedral_grid_3d

REGION = CadRegion(("cube",))


def _wall(points: np.ndarray) -> np.ndarray:
    """The face of the cube nearest each point."""
    distance = np.concatenate([points, 1.0 - points], axis=1)  # x-, y-, z-, x+, y+, z+
    nearest = np.argmin(distance, axis=1)
    return 2 * (nearest % 3) + nearest // 3


class _Cube:
    """The unit cube as a CAD model: its topology, a projection, and a named solid ``floor``
    standing in for whatever the face ``z-`` is shared with."""

    def __init__(self, fold: float, push: float = 0.0, split: bool = False) -> None:
        corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        edges, faces = [], []
        for a in range(8):
            for b in range(a + 1, 8):
                differ = np.flatnonzero(corners[a] != corners[b])
                if differ.size == 1:
                    fixed = [axis for axis in range(3) if axis != differ[0]]
                    edges.append((a, b))
                    faces.append([2 * axis + int(corners[a][axis]) for axis in fixed])
        if split:
            # The edge x = 1, y = 1 drawn as two edges, its halves, the upper half listed first.
            corners = np.vstack([corners, [1.0, 1.0, 0.5]])
            at = edges.index((6, 7))
            edges[at : at + 1] = [(8, 7), (6, 8)]
            faces[at : at + 1] = [faces[at], faces[at]]
        self.push = push
        self._topology = CadTopology(
            faces=tuple(f"face {i}" for i in range(6)),
            face_area=np.ones(6),
            vertices=corners,
            edge_vertices=np.array(edges),
            edge_length=np.ones(len(edges)),
            edge_face_offsets=np.arange(len(edges) + 1) * 2,
            edge_face_indices=np.array(faces).reshape(-1),
            edge_angle=np.full(len(edges), fold),
            seam=np.zeros(len(edges), dtype=bool),
        )

    def topology(self, region):
        assert region == REGION
        return self._topology

    def write_brep(self, region, path):
        return path

    def project(self, region, points, faces=None, edges=None):
        points = np.asarray(points, dtype=float)
        if region == CadRegion(("floor",)):
            distance = np.abs(points[:, 2])
            return Projection(points, np.zeros(len(points), int), np.zeros_like(points), distance)
        if edges is not None:
            # The closest point of each edge's segment.
            ends = self._topology.vertices[self._topology.edge_vertices[np.asarray(edges)]]
            along = ends[:, 1] - ends[:, 0]
            t = np.clip(
                np.sum((points - ends[:, 0]) * along, axis=1) / np.sum(along**2, axis=1), 0, 1
            )
            landed = ends[:, 0] + t[:, None] * along
            face = self._topology.edge_face_indices[2 * np.asarray(edges)]
            distance = np.linalg.norm(landed - points, axis=1)
            return Projection(landed, face, np.zeros_like(points), distance)
        face = _wall(points) if faces is None else np.asarray(faces)
        # The drawing's faces stand ``push`` outside the cube's, so a projection shows which face.
        outward = np.zeros_like(points)
        outward[np.arange(len(face)), face // 2] = np.where(face % 2, 1.0, -1.0)
        landed = points + self.push * outward
        return Projection(landed, face, outward, np.full(len(points), abs(self.push)))


class _Netgen:
    """Kuhn tetrahedra of the cube, as a triangulation and a fill; counts its calls.

    ``n`` cubes a side; ``perturb`` moves the interior points (the boundary stays the cube's).
    """

    def __init__(self, n: int = 2, perturb: float = 0.0) -> None:
        mesh = tetrahedral_grid_3d(n, perturb=perturb)
        self.points = np.asarray(mesh.node_coords)
        offsets = np.asarray(mesh.face_nodes.offsets)
        nodes = np.asarray(mesh.face_nodes.face_node_indices).reshape(-1, 3)
        owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
        assert np.all(np.diff(offsets) == 3)
        corners = [set() for _ in range(mesh.n_cells)]
        for face, ring in enumerate(nodes):
            for cell in (owner[face], neighbour[face]):
                if cell >= 0:
                    corners[cell].update(ring.tolist())
        self.tetrahedra = np.array([sorted(c) for c in corners])
        self.triangles = nodes[neighbour < 0]
        self.calls = 0

    def triangulate(self, shape, **options):
        # As Netgen does: the surface's own points only, every one used by a triangle.
        self.calls += 1
        used, triangles = np.unique(self.triangles, return_inverse=True)
        triangles = triangles.reshape(-1, 3)
        face = _wall(self.points[used][triangles].mean(axis=1))
        return SurfaceTriangulation(self.points[used], triangles, face)

    def fill(self, points, triangles, **options):
        # As Netgen does: the given points first, in their order, then the points it adds.
        self.calls += 1
        points = np.asarray(points)
        given = {tuple(p): i for i, p in enumerate(points.round(12).tolist())}
        number = np.empty(len(self.points), dtype=np.int64)
        added = []
        for i, p in enumerate(self.points.round(12).tolist()):
            if tuple(p) in given:
                number[i] = given[tuple(p)]
            else:
                number[i] = len(points) + len(added)
                added.append(self.points[i])
        return TetrahedralMesh(np.vstack([points, *added]), number[self.tetrahedra])


def _areas(mesh) -> dict[str, float]:
    area = np.asarray(mesh.geometry().face.area)
    return {
        name: float(area[np.asarray(mesh.face_patches.indices(name))].sum())
        for name in mesh.face_patches.names
        if name not in ("interior", "boundary")
    }


def test_the_faces_each_selection_chooses_make_its_patch_and_the_rest_make_others():
    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=_Netgen())
    mesh = mesher.mesh(
        _Cube(fold=math.pi / 2),
        REGION,
        {"lid": FacesNear(((0.5, 0.5, 1.0),)), "floor": FacesOn(("floor",))},
        others="sides",
    )
    assert _areas(mesh) == pytest.approx({"lid": 1.0, "floor": 1.0, "sides": 4.0}, rel=1e-12)
    assert float(np.asarray(mesh.geometry().cell.volume).sum()) == pytest.approx(1.0, rel=1e-12)
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-14
    # One cell per point of the tetrahedra: their polyhedral dual, not the tetrahedra themselves.
    assert mesh.n_cells == 27


def test_the_cads_sharp_edges_are_kept_whatever_the_triangles_fold_by():
    # One patch over the whole cube, and a feature angle above the triangles' 90 degree folds:
    # only the CAD's say-so keeps the cube's edges, and with them its corners and its volume.
    def volume(fold):
        mesher = CadMesher(max_size=0.5, feature_angle=math.radians(100), netgen=_Netgen())
        mesh = mesher.mesh(_Cube(fold=fold), REGION)
        return float(np.asarray(mesh.geometry().cell.volume).sum())

    assert volume(math.radians(120)) == pytest.approx(1.0, rel=1e-12)
    assert volume(math.radians(90)) < 1.0 - 1e-3


def test_each_boundary_face_centroid_is_moved_onto_the_face_its_triangle_lies_on():
    # Each of the drawing's faces stands 0.01 outside the cube's. The lid's dual vertices at the
    # centroids of its eight triangles move onto the lid's own plane; nothing else reaches it.
    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=_Netgen())
    mesh = mesher.mesh(_Cube(fold=math.pi / 2, push=0.01), REGION)
    z = np.asarray(mesh.node_coords)[:, 2]
    assert np.count_nonzero(np.isclose(z, 1.01)) == 8
    assert float(z.max()) == pytest.approx(1.01)
    assert np.count_nonzero(np.isclose(z, -0.01)) == 8


def test_a_vertex_on_an_edge_drawn_in_two_pieces_lands_on_the_nearer_piece():
    # The cube's edge x = y = 1 drawn as two halves: a dual vertex on its lower half must stay
    # there (the mesh unchanged), not be carried to the halves' shared end by the upper half.
    def coordinates(split):
        mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=_Netgen())
        return np.asarray(mesher.mesh(_Cube(fold=math.pi / 2, split=split), REGION).node_coords)

    np.testing.assert_allclose(coordinates(split=True), coordinates(split=False), atol=1e-14)


def test_two_selections_choosing_one_face_are_refused():
    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=_Netgen())
    with pytest.raises(ValueError, match="'top' selects a face another patch has already"):
        mesher.mesh(
            _Cube(fold=math.pi / 2),
            REGION,
            {"lid": FacesNear(((0.5, 0.5, 1.0),)), "top": FacesNear(((0.2, 0.3, 1.0),))},
        )


def test_a_solid_no_face_lies_on_is_refused():
    # The floor solid sits at z = 0; moved off the cube, no face of the cube lies on it.
    class _Away(_Cube):
        def project(self, region, points, faces=None):
            landed = super().project(region, points, faces)
            if region == CadRegion(("floor",)):
                return Projection(landed.points, landed.face, landed.normal, landed.distance + 0.1)
            return landed

    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=_Netgen())
    with pytest.raises(
        ValueError, match=r"no face of the region lies on the surface of \['floor'\]"
    ):
        mesher.mesh(_Away(fold=math.pi / 2), REGION, {"floor": FacesOn(("floor",))})


@pytest.mark.parametrize(
    ("patches", "others", "message"),
    [
        ({"boundary": FacesNear(((0.5, 0.5, 1.0),))}, "walls", r"\['boundary'\] are reserved"),
        ({}, "interior", r"\['interior'\] are reserved"),
        ({"walls": FacesNear(((0.5, 0.5, 1.0),))}, "walls", "'walls' names both a selection"),
    ],
)
def test_a_patch_name_that_cannot_be_used_is_refused_before_meshing(patches, others, message):
    netgen = _Netgen()
    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=netgen)
    with pytest.raises(ValueError, match=message):
        mesher.mesh(_Cube(fold=math.pi / 2), REGION, patches, others=others)
    assert netgen.calls == 0


@pytest.mark.parametrize("name", ["max_size", "feature_angle", "min_size", "grading"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_a_size_angle_or_grading_out_of_range_is_refused(name, value):
    settings = {"max_size": 0.5, "feature_angle": 0.5, name: value}
    with pytest.raises(ValueError, match=f"CadMesher.{name} must be a positive, finite number"):
        CadMesher(**settings)


def test_facesnear_takes_points_of_three_coordinates():
    with pytest.raises(ValueError, match="points of three coordinates"):
        FacesNear(((0.5, 0.5),))


def test_a_vertex_on_an_edge_inside_one_face_is_moved_onto_that_face():
    # A Kuhn L cut at its inside edge: a cut cell's boundary fan is split away from any feature,
    # through a vertex on a boundary edge between two triangles of one wall. Each wall of this
    # drawing stands 0.01 off the L's, along its own axis, so every vertex the projection reaches
    # -- each boundary-face centroid and each such vertex -- moves by exactly that.
    from aquaflux.mesh import Mesh
    from aquaflux.meshing import PolyhedralDual
    from aquaflux.meshing.cad_mesher import _Boundary, _onto_the_drawing, _SurfaceEdges

    from tests.unit.test_meshing_dual import _cube_walls, _without_corner

    block = tetrahedral_grid_3d(3)
    stretched = Mesh.from_csr(
        np.asarray(block.node_coords) * [1.0, 1.0, 1.5], block.face_nodes.offsets,
        block.face_nodes.face_node_indices, block.face_cells.owner, block.face_cells.neighbour,
        block.n_cells,
    )  # fmt: skip
    primal = _without_corner(_cube_walls(stretched), (1 / 3, 0.5), [0, 2])
    traced = PolyhedralDual(math.radians(45)).build_traced(primal)
    points = np.asarray(primal.node_coords)
    boundary = np.asarray(primal.face_cells.neighbour) < 0
    triangles = np.asarray(primal.face_nodes.face_node_indices).reshape(-1, 3)[boundary]
    # A wall per plane: its axis is the one along which the triangle's corners agree.
    corners = points[triangles]
    axis = np.argmax(np.all(np.isclose(corners, corners[:, :1]), axis=1), axis=1)
    level = np.round(corners[np.arange(len(axis)), 0, axis], 9)
    walls, face = np.unique(np.stack([axis, level], axis=1), axis=0, return_inverse=True)
    boundary = _Boundary(triangles, face.reshape(-1))

    class _Drawing:
        def project(self, region, points, faces=None, edges=None):
            landed = np.array(points, dtype=float)
            if faces is not None:
                landed[np.arange(len(faces)), walls[faces, 0].astype(int)] += 0.01
            return Projection(landed, faces, np.zeros_like(landed), np.full(len(landed), 0.01))

    class _NoEdges:
        n_edges = 0

    moved = _onto_the_drawing(
        traced, primal, boundary, _SurfaceEdges.of(boundary), _Drawing(), REGION, _NoEdges()
    )
    shift = np.linalg.norm(
        np.asarray(moved.node_coords) - np.asarray(traced.mesh.node_coords), axis=1
    )
    beside = _SurfaceEdges.of(boundary).faces_of(traced.vertex_edge[traced.vertex_edge[:, 0] >= 0])
    in_one_wall = int(np.count_nonzero(beside[:, 0] == beside[:, 1]))
    assert in_one_wall > 0
    expected = np.count_nonzero(traced.vertex_face >= 0) + in_one_wall
    assert np.count_nonzero(np.isclose(shift, 0.01)) == expected
    assert np.count_nonzero(shift > 1e-12) == expected


def test_a_planarizer_flattens_the_dual_moving_only_its_cell_centroids():
    # The perturbed Kuhn tetrahedra's dual has warped faces. Flattened, its boundary vertices -- on the
    # drawing, 0.01 off the cube -- must not have moved, while its cell centroids have.
    def mesh(planarizer):
        mesher = CadMesher(
            max_size=0.5, feature_angle=math.radians(45), planarizer=planarizer,
            netgen=_Netgen(n=3, perturb=0.2),
        )  # fmt: skip
        return mesher.mesh(_Cube(fold=math.pi / 2, push=0.01), REGION)

    plain, flat = mesh(None), mesh(FacePlanarizer())
    before, after = np.asarray(plain.node_coords), np.asarray(flat.node_coords)
    moved = np.linalg.norm(after - before, axis=1) > 1e-12
    offsets = np.asarray(plain.face_nodes.offsets)
    nodes = np.asarray(plain.face_nodes.face_node_indices)
    on_boundary = np.zeros(plain.n_nodes, dtype=bool)
    for face in np.flatnonzero(np.asarray(plain.face_cells.neighbour) < 0):
        on_boundary[nodes[offsets[face] : offsets[face + 1]]] = True
    assert moved.any() and not np.any(moved & on_boundary)
    interior = np.asarray(plain.face_cells.interior)

    def total_warp(mesh):
        return float(np.sum(np.asarray(face_warp(mesh))[interior] ** 2))

    assert total_warp(flat) < 0.5 * total_warp(plain)
    assert float(np.asarray(flat.geometry().cell.volume).sum()) == pytest.approx(
        float(np.asarray(plain.geometry().cell.volume).sum()), rel=1e-12
    )
    assert float(np.max(np.asarray(closed_cell_residual(flat)))) < 1e-14


def test_the_layer_cells_centroids_are_not_free_to_move():
    # Vertices 0-3 are the centroids of primal cells 0-3, of which 1 and 3 are layer cells; vertex 4
    # is no cell's centroid.
    from types import SimpleNamespace

    from aquaflux.meshing.cad_mesher import _free_centroids

    traced = SimpleNamespace(vertex_cell=np.array([0, 1, 2, 3, -1]))
    columns = SimpleNamespace(cells=np.array([[1], [3]]))
    assert _free_centroids(traced, columns).tolist() == [True, False, True, False, False]
    assert _free_centroids(traced, None).tolist() == [True, True, True, True, False]


def test_the_core_is_a_hole_in_the_fills_surface_and_its_cells_are_numbered_after_its_points():
    # A box of edge 2 meshed at 0.2, a core of 0.5 cubes graded to 0.25 at its walls: the core's
    # capped surface joins the box's as a hole, the cells it brings fill exactly that hole, and
    # the tetrahedra round it may be as large as its wall cubes.
    from aquaflux.meshing import OctreeCore, element_mesh

    from tests.support.meshes import box_surface

    points, surface = box_surface(0.0, 1.0, 8, into_box=False)
    points = 2.0 * points
    core = OctreeCore(size=0.5, wall_size=0.25, clearance=0.3)
    mesher = CadMesher(max_size=0.2, feature_angle=math.radians(45), core=core, netgen=_Netgen())
    joined, holed, cells = mesher._cored(points, surface)
    np.testing.assert_array_equal(joined[: len(points)], points)
    np.testing.assert_array_equal(holed[: len(surface)], surface)
    used, local = np.unique(holed, return_inverse=True)
    check_closed_surface(joined[used], local.reshape(-1, 3))
    corner = joined[holed]
    left = np.einsum("ij,ij->i", corner[:, 0], np.cross(corner[:, 1], corner[:, 2])).sum() / 6.0
    filled = element_mesh(
        joined, tetrahedra=cells.tetrahedra, hexahedra=cells.hexahedra, pyramids=cells.pyramids
    )
    volume = float(np.asarray(filled.geometry().cell.volume).sum())
    assert cells.hexahedra.shape[0] > 0 and 0.0 < left < 8.0
    np.testing.assert_allclose(volume + left, 8.0, rtol=1e-12)
    # The cells' outer faces are the hole's triangles, and nothing else.
    offsets = np.asarray(filled.face_nodes.offsets)
    nodes = np.asarray(filled.face_nodes.face_node_indices)
    rings = [
        nodes[offsets[f] : offsets[f + 1]]
        for f in np.flatnonzero(np.asarray(filled.face_cells.neighbour) < 0)
    ]
    hole = holed[len(surface) :]
    assert {tuple(sorted(r.tolist())) for r in rings} == {tuple(sorted(r)) for r in hole.tolist()}
    assert cells.fill_size == 0.25
    assert CadMesher(max_size=0.2, feature_angle=1.0)._cored(points, surface)[2].fill_size == 0.2
