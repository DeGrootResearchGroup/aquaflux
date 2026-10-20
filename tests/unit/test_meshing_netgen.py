"""Netgen's triangulation and fill, driven through its child process from a process that holds OCP.

The whole module skips when Netgen is not installed. It is found without importing it: importing it
here would load its geometry kernel into the test process beside OCP's, which is the crash the child
process exists to avoid. The solids are written by OCP in this process, so every test also exercises
the arrangement the mesher depends on -- the CAD kernel in the parent, Netgen in a child, a file
between them.

Each result is checked against the geometry it should reproduce (the plane or cylinder a triangle
lies on, the volume a fill encloses, the surface it must keep), not against a recorded mesh.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.meshing import (
    MeshingError,
    NetgenProcess,
    check_closed_surface,
    netgen_available,
)

pytestmark = pytest.mark.skipif(not netgen_available(), reason="netgen-mesher is not installed")

BOX = (1.0, 2.0, 3.0)
RADIUS, HEIGHT = 0.5, 2.0


@pytest.fixture(scope="module")
def solids(tmp_path_factory):
    """A box ``[0, 1] x [0, 2] x [0, 3]`` and a cylinder (radius 0.5 on the z axis, ``z`` in
    ``[0, 2]``), written by OCP as BREP files, with each one's faces in OCP's explorer order."""
    pytest.importorskip("OCP")
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer

    directory = tmp_path_factory.mktemp("solids")
    shapes = {
        "box": BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), *BOX).Shape(),
        "cylinder": BRepPrimAPI_MakeCylinder(RADIUS, HEIGHT).Shape(),
    }
    found = {}
    for name, shape in shapes.items():
        path = directory / f"{name}.brep"
        BRepTools.Write_s(shape, str(path))
        centres = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            properties = GProp_GProps()
            BRepGProp.SurfaceProperties_s(explorer.Current(), properties)
            centre = properties.CentreOfMass()
            centres.append((centre.X(), centre.Y(), centre.Z()))
            explorer.Next()
        found[name] = (path, np.array(centres))
    return found


def _normals(points, triangles):
    corner = points[triangles]
    return np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])


def _box_face(centre):
    """The axis and side of the box face whose centre is ``centre``."""
    for axis, extent in enumerate(BOX):
        for side in (0.0, extent):
            if np.isclose(centre[axis], side):
                return axis, side
    raise AssertionError(f"no box face is centred at {centre}")


def test_each_box_triangle_lies_on_the_face_ocp_numbers_it_by_and_points_out(solids):
    path, centres = solids["box"]
    surface = NetgenProcess().triangulate(path, max_size=0.4)
    assert sorted(np.unique(surface.face)) == list(range(len(centres)))
    normals = _normals(surface.points, surface.triangles)
    for face, centre in enumerate(centres):
        axis, side = _box_face(centre)
        mine = surface.face == face
        np.testing.assert_allclose(surface.points[surface.triangles[mine]][..., axis], side)
        outward = 1.0 if side > 0.0 else -1.0
        assert np.all(normals[mine][:, axis] * outward > 0.0)


def test_cylinder_triangles_lie_on_their_faces_and_point_out(solids):
    path, centres = solids["cylinder"]
    surface = NetgenProcess().triangulate(path, max_size=0.15)
    normals = _normals(surface.points, surface.triangles)
    corner = surface.points[surface.triangles]
    for face, centre in enumerate(centres):
        mine = surface.face == face
        if np.isclose(centre[2], HEIGHT / 2):  # the side
            np.testing.assert_allclose(np.hypot(corner[mine][..., 0], corner[mine][..., 1]), RADIUS)
            middle = corner[mine].mean(axis=1)
            assert np.all(np.einsum("ij,ij->i", normals[mine][:, :2], middle[:, :2]) > 0.0)
        else:  # a cap, at z = 0 or z = HEIGHT
            np.testing.assert_allclose(corner[mine][..., 2], centre[2])
            assert np.all(normals[mine][:, 2] * np.sign(centre[2] - HEIGHT / 2) > 0.0)


def test_a_finer_size_gives_smaller_triangles(solids):
    path, _ = solids["box"]
    coarse = NetgenProcess().triangulate(path, max_size=0.5)
    fine = NetgenProcess().triangulate(path, max_size=0.2)

    def longest(surface):
        corner = surface.points[surface.triangles]
        return np.linalg.norm(corner - np.roll(corner, 1, axis=1), axis=-1).max()

    assert longest(fine) < longest(coarse)
    # The size is Netgen's target, not a hard bound; an ignored size would leave box-length edges.
    assert longest(fine) <= 2 * 0.2


def test_the_triangulation_is_a_closed_outward_surface(solids):
    for name in ("box", "cylinder"):
        path, _ = solids[name]
        surface = NetgenProcess().triangulate(path, max_size=0.3)
        check_closed_surface(surface.points, surface.triangles)


def _boundary_triangles(tetrahedra):
    """The faces of a tetrahedral mesh used by one tetrahedron, as sorted vertex triples."""
    faces = np.concatenate([tetrahedra[:, [1, 2, 3]], tetrahedra[:, [0, 3, 2]],
                            tetrahedra[:, [0, 1, 3]], tetrahedra[:, [0, 2, 1]]])  # fmt: skip
    keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    return {tuple(key) for key in keys[counts == 1].tolist()}


def _signed_volumes(points, tetrahedra):
    corner = points[tetrahedra]
    edges = corner[:, 1:] - corner[:, :1]
    return np.einsum("ij,ij->i", np.cross(edges[:, 0], edges[:, 1]), edges[:, 2]) / 6.0


def _enclosed_volume(points, triangles):
    corner = points[triangles]
    return np.einsum("ij,ij->i", corner[:, 0], np.cross(corner[:, 1], corner[:, 2])).sum() / 6.0


@pytest.mark.parametrize("name", ["box", "cylinder"])
def test_the_fill_keeps_the_surface_and_fills_exactly_what_it_encloses(solids, name):
    path, _ = solids[name]
    netgen = NetgenProcess()
    surface = netgen.triangulate(path, max_size=0.3)
    filled = netgen.fill(surface.points, surface.triangles, max_size=0.3)
    np.testing.assert_array_equal(filled.points[: len(surface.points)], surface.points)
    assert _boundary_triangles(filled.tetrahedra) == {
        tuple(t) for t in np.sort(surface.triangles, axis=1).tolist()
    }
    volumes = _signed_volumes(filled.points, filled.tetrahedra)
    assert np.all(volumes > 0.0)
    np.testing.assert_allclose(
        volumes.sum(), _enclosed_volume(surface.points, surface.triangles), rtol=1e-12
    )


def _box_surface(lo, hi, n, *, into_box, offset=0):
    """The six faces of the cube ``[lo, hi]^3``, ``n`` x ``n`` squares each split in two.

    Wound out of the cube, or into it with ``into_box`` (the walls of a hole). Point indices start
    at ``offset``.
    """
    ticks = np.linspace(lo, hi, n + 1)
    points, index, triangles = [], {}, []

    def point(xyz):
        key = tuple(np.round(xyz, 12))
        if key not in index:
            index[key] = offset + len(points)
            points.append(xyz)
        return index[key]

    for axis in range(3):
        a, b = (axis + 1) % 3, (axis + 2) % 3
        for side in (lo, hi):
            for i in range(n):
                for j in range(n):
                    ring = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        xyz = np.zeros(3)
                        xyz[axis], xyz[a], xyz[b] = side, ticks[i + di], ticks[j + dj]
                        ring.append(point(xyz))
                    if (side == lo) != into_box:
                        ring = ring[::-1]
                    triangles += [ring[:3], [ring[0], ring[2], ring[3]]]
    return np.array(points), np.array(triangles)


def test_the_fill_meshes_the_gap_between_two_surfaces():
    outer_points, outer = _box_surface(0.0, 1.0, 4, into_box=False)
    inner_points, inner = _box_surface(0.35, 0.65, 2, into_box=True, offset=len(outer_points))
    points = np.vstack([outer_points, inner_points])
    triangles = np.vstack([outer, inner])
    filled = NetgenProcess().fill(points, triangles, max_size=0.3)
    np.testing.assert_array_equal(filled.points[: len(points)], points)
    volumes = _signed_volumes(filled.points, filled.tetrahedra)
    assert np.all(volumes > 0.0)
    np.testing.assert_allclose(volumes.sum(), 1.0 - 0.3**3, rtol=1e-12)


def test_a_fill_of_the_wrong_region_is_refused():
    # The hole's walls wound out of the hole rather than into it: still closed and of positive total
    # volume, so the surface check passes it, and Netgen fills straight through the hole, keeping
    # every supplied point. Only the check that the tetrahedra are bounded by the supplied surface
    # catches it.
    outer_points, outer = _box_surface(0.0, 1.0, 4, into_box=False)
    inner_points, inner = _box_surface(0.35, 0.65, 2, into_box=False, offset=len(outer_points))
    with pytest.raises(MeshingError, match="bounded by 192 triangles, not the 240 supplied"):
        NetgenProcess().fill(
            np.vstack([outer_points, inner_points]), np.vstack([outer, inner]), max_size=0.3
        )


def test_a_file_netgen_cannot_read_is_reported(tmp_path):
    missing = tmp_path / "missing.brep"
    with pytest.raises(MeshingError, match="Netgen's triangulate failed"):
        NetgenProcess().triangulate(missing, max_size=0.3)


def test_netgen_numbers_a_written_regions_faces_as_its_topology_does(tmp_path):
    # The contract the mesher rests on: a face index from CadModel.topology is the face Netgen
    # reports for the file CadModel.write_brep wrote. Checked by area, face by face, on the Sozzi
    # water, whose nine faces have areas from 2.9e-4 to 0.25 square metres.
    pytest.importorskip("OCP")
    from pathlib import Path

    from aquaflux.io.cad import CadRegion, Placement, read_step

    drawing = (
        Path(__file__).resolve().parents[2]
        / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
    )
    model = read_step(drawing, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    water = CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
    topology = model.topology(water)
    surface = NetgenProcess().triangulate(
        model.write_brep(water, tmp_path / "water.brep"), max_size=0.01
    )
    areas = np.bincount(
        surface.face,
        weights=0.5 * np.linalg.norm(_normals(surface.points, surface.triangles), axis=1),
        minlength=topology.n_faces,
    )
    # The smallest faces (pipe caps, the lamp's tip) are under-resolved at this size by up to 7 %.
    np.testing.assert_allclose(areas, topology.face_area, rtol=0.1)
    # Area cannot tell the two pipe caps apart (they are equal), so locate them too: a triangle's
    # centroid is nearest the face the triangle lies on.
    rng = np.random.default_rng(0)
    sample = np.concatenate(
        [rng.choice(np.flatnonzero(surface.face == f), 20) for f in range(topology.n_faces)]
    )
    centroids = surface.points[surface.triangles[sample]].mean(axis=1)
    np.testing.assert_array_equal(model.project(water, centroids).face, surface.face[sample])


def test_the_dual_of_a_netgen_l_is_cut_at_its_inside_edge_into_nearly_convex_cells(tmp_path):
    # The case that needs the cut: on Netgen's tetrahedra of an L, a cell around the inside edge,
    # uncut, wraps round it -- one of its vertices lies 1.2 cell sizes beyond one of its own faces
    # (at this size). Cut with the vertices between the pieces left at face centroids it is 0.56,
    # and with the vertices on its cut edges left at their midpoints 1.1. Cut and placed, every
    # cell is within 0.35 of convex.
    pytest.importorskip("OCP")
    from aquaflux.mesh import cell_non_convexity, closed_cell_residual
    from aquaflux.meshing import PolyhedralDual, element_mesh
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt

    shape = BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.2, 0.1, 0.1).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0.1), 0.1, 0.1, 0.1).Shape(),
    ).Shape()
    BRepTools.Write_s(shape, str(tmp_path / "l.brep"))
    netgen = NetgenProcess()
    surface = netgen.triangulate(tmp_path / "l.brep", max_size=0.01)
    filled = netgen.fill(surface.points, surface.triangles, max_size=0.01)
    primal = element_mesh(filled.points, filled.tetrahedra)
    dual = PolyhedralDual(feature_angle=np.radians(45)).build(primal)
    on_edge = np.isclose(filled.points[:, 0], 0.1) & np.isclose(filled.points[:, 2], 0.1)
    assert dual.n_cells == primal.n_nodes + np.count_nonzero(on_edge)
    volume = np.asarray(dual.geometry().cell.volume)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(volume.sum(), 0.2 * 0.1 * 0.1 + 0.1 * 0.1 * 0.1, rtol=1e-12)
    scale = float(np.max(np.asarray(dual.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(dual)))) < 1e-12 * scale
    assert float(cell_non_convexity(dual).max()) < 0.4


#: The Sozzi water's faces, as its topology numbers them (pinned above by area): the vessel's
#: cylinder and end plates, the two pipes' cylinders, the lamp's cylinder and hemispherical tip, and
#: the two pipes' end caps.
SOZZI_PATCH_FACES = {"inlet": {6}, "outlet": {8}, "lamp": {4, 7}, "walls": {0, 1, 2, 3, 5}}


def test_a_case_files_step_mesh_is_written_as_a_polymesh_with_its_patches_on_their_faces(
    tmp_path, monkeypatch
):
    # The whole path a user takes: a case file's CadMesh section, meshed by `aquaflux mesh --case
    # --openfoam`, read back from the polyMesh. Every boundary vertex must lie on the drawing, every
    # boundary face on a CAD face of its patch with nearly all of its faces' area, and every cell
    # must close.
    pytest.importorskip("OCP")
    import contextlib
    import io
    import json
    import os
    from pathlib import Path

    import yaml
    from aquaflux.__main__ import main
    from aquaflux.io import read_openfoam
    from aquaflux.io.cad import CadRegion, Placement, read_step
    from aquaflux.mesh import closed_cell_residual

    drawing = (
        Path(__file__).resolve().parents[2]
        / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
    )
    swap = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    section = {
        "kind": "CadMesh",
        "file": os.path.relpath(
            drawing, tmp_path
        ),  # relative to the case file, as a user writes it
        "solids": ["reactor_body", "inlet_pipe", "outlet_pipe"],
        "without": ["lamp"],
        "placement": {"kind": "CadPlacement", "matrix": swap},
        "max_size": 0.03,
        "feature_angle": 0.785,
        "patches": {
            "inlet": {"kind": "FacesNear", "points": [[0.04765, 0.0, 0.8945]]},
            "outlet": {"kind": "FacesNear", "points": [[1.739, 0.0, 0.0]]},
            "lamp": {"kind": "FacesOn", "solids": ["lamp"]},
        },
    }
    (tmp_path / "case.yaml").write_text(yaml.safe_dump({"mesh": section}))
    # Run from elsewhere: the drawing is found from the case file's directory, not from here.
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = main(["mesh", str(tmp_path / "out"), "--case", str(tmp_path / "case.yaml"),
                       "--openfoam"])  # fmt: skip
    reply = json.loads(out.getvalue())
    assert status == 0, reply
    assert {patch["name"] for patch in reply["patches"]} == set(SOZZI_PATCH_FACES)
    mesh = read_openfoam(tmp_path / "out")
    assert mesh.n_cells == reply["cells"] > 1000
    scale = float(np.max(np.asarray(mesh.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale

    model = read_step(drawing, Placement(matrix=swap))
    water = CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
    topology = model.topology(water)
    geometry = mesh.geometry().face
    for name, faces in SOZZI_PATCH_FACES.items():
        mine = np.asarray(mesh.face_patches.indices(name))
        landed = model.project(water, np.asarray(geometry.centroid)[mine]).face
        assert set(landed.tolist()) <= faces, name
        # Flat polygons between points of a curved surface, or of a circular rim: short by little.
        area = float(np.asarray(geometry.area)[mine].sum())
        assert area == pytest.approx(topology.face_area[sorted(faces)].sum(), rel=0.025), name
    offsets = np.asarray(mesh.face_nodes.offsets)
    nodes = np.asarray(mesh.face_nodes.face_node_indices)
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    on_boundary = np.unique(np.concatenate([nodes[offsets[f] : offsets[f + 1]] for f in boundary]))
    off = model.project(water, np.asarray(mesh.node_coords)[on_boundary]).distance
    assert float(off.max()) < 1e-9


def test_prism_layers_on_the_sozzi_walls_keep_their_first_height_from_the_true_wall():
    # Layers on the vessel's walls and the lamp, none on the pipes' ends. Every cell must close
    # with positive volume and every boundary vertex lie on the drawing; and the dual's first layer
    # boundary -- the centroids of the prisms against the wall, the only dual vertices between
    # 0.2 and 0.8 mm from it -- must sit at the first height from the true, curved wall, not from
    # the triangles' chords (which on the lamp lie up to about 0.4 mm inside it at this size).
    pytest.importorskip("OCP")
    from pathlib import Path

    from aquaflux.io.cad import CadRegion, Placement, read_step
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import CadMesher, FacesNear, FacesOn, PrismLayers

    drawing = (
        Path(__file__).resolve().parents[2]
        / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
    )
    model = read_step(drawing, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    water = CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
    patches = {
        "inlet": FacesNear(((0.04765, 0.0, 0.8945),)),
        "outlet": FacesNear(((1.739, 0.0, 0.0),)),
        "lamp": FacesOn(("lamp",)),
    }
    bare = CadMesher(max_size=0.03, feature_angle=0.785).mesh(model, water, patches)
    layers = PrismLayers(first_height=0.0005, growth=1.2, count=5)
    mesh = CadMesher(max_size=0.03, feature_angle=0.785, layers=layers).mesh(
        model, water, patches, layered=("walls", "lamp")
    )
    # Five rows of layer points on most of the surface: several times the bare mesh's cells.
    assert mesh.n_cells > 2 * bare.n_cells
    geometry = mesh.geometry()
    assert np.all(np.asarray(geometry.cell.volume) > 0.0)
    scale = float(np.max(np.asarray(geometry.face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    coords = np.asarray(mesh.node_coords)
    offsets = np.asarray(mesh.face_nodes.offsets)
    nodes = np.asarray(mesh.face_nodes.face_node_indices)
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    on_boundary = np.unique(np.concatenate([nodes[offsets[f] : offsets[f + 1]] for f in boundary]))
    assert float(model.project(water, coords[on_boundary]).distance.max()) < 1e-9
    radius = np.hypot(coords[:, 1], coords[:, 2])
    along = (coords[:, 0] > 0.1) & (coords[:, 0] < 0.7)
    # Within a millimetre of the lamp (radius 0.01) or the vessel (0.0445), then by true distance.
    for name, near in (
        ("lamp", np.abs(radius - 0.0105) < 0.001),
        ("vessel", np.abs(radius - 0.044) < 0.001),
    ):
        away = model.project(water, coords[along & near]).distance
        first = away[(away > 0.0002) & (away < 0.0008)]
        assert first.size > 100, name
        assert float(np.median(first)) == pytest.approx(0.0005, rel=0.03), name


def test_the_dual_of_an_octree_core_filled_out_to_a_box_is_closed_and_hexahedral_inside():
    # A core of 0.1 cubes, 0.05 near a sphere at the centre, kept 0.15 clear of the box's walls;
    # Netgen fills the gap between its capped surface and the walls, keeping both. The dual of the
    # whole primal mesh must fill the box exactly, and every point the core's hexahedra surround
    # on all sides must become a cube of their size (the dual of a block of equal cubes is the
    # same block shifted half a cube).
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import NearSurface, OctreeCore, PolyhedralDual, element_mesh

    def ball(points):
        return np.linalg.norm(points - 0.5, axis=1) < 0.15

    def box(points):
        return np.all((points >= 0.0) & (points <= 1.0), axis=1)

    core = OctreeCore(size=0.1, clearance=0.15, refinements=(NearSurface(ball, 0.05),)).mesh(
        box, [0, 0, 0], [1, 1, 1]
    )
    assert set(np.unique(core.octree.level[core.kept]).tolist()) == {0, 1}
    wall_points, wall = _box_surface(0.0, 1.0, 8, into_box=False)
    # The fill wants the surface's own points only; the core's inner points follow its result.
    on_surface = np.unique(core.surface)
    number = np.empty(core.points.shape[0], dtype=np.int64)
    number[on_surface] = len(wall_points) + np.arange(on_surface.size)
    filled = NetgenProcess().fill(
        np.vstack([wall_points, core.points[on_surface]]),
        np.vstack([wall, number[core.surface]]),
        max_size=0.1,
    )
    inner = np.setdiff1d(np.arange(core.points.shape[0]), on_surface)
    number[inner] = len(filled.points) + np.arange(inner.size)
    primal = element_mesh(
        np.vstack([filled.points, core.points[inner]]),
        tetrahedra=np.vstack([filled.tetrahedra, number[core.tetrahedra]]),
        hexahedra=number[core.hexahedra],
        pyramids=number[np.vstack([core.pyramids, core.caps])],
    )
    np.testing.assert_allclose(np.asarray(primal.geometry().cell.volume).sum(), 1.0, rtol=1e-12)
    dual = PolyhedralDual(feature_angle=np.radians(45)).build(primal)
    assert dual.n_cells == primal.n_nodes  # a box has no concave edge to cut cells at
    volume = np.asarray(dual.geometry().cell.volume)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(volume.sum(), 1.0, rtol=1e-12)
    scale = float(np.max(np.asarray(dual.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(dual)))) < 1e-12 * scale
    surrounded = np.bincount(number[core.hexahedra].ravel(), minlength=primal.n_nodes) == 8
    assert np.count_nonzero(surrounded) > 100
    size = np.zeros(primal.n_nodes)
    hexahedra = number[core.hexahedra]
    size[hexahedra] = np.linalg.norm(
        primal.node_coords[hexahedra[:, 1]] - primal.node_coords[hexahedra[:, 0]], axis=1
    )[:, None]
    np.testing.assert_allclose(volume[surrounded], size[surrounded] ** 3, rtol=1e-12)
    face_size = np.diff(np.asarray(dual.face_nodes.offsets))
    faces_of = np.zeros(dual.n_cells, dtype=np.int64)
    quads_of = np.zeros(dual.n_cells, dtype=np.int64)
    for side in (dual.face_cells.owner, dual.face_cells.neighbour):
        side = np.asarray(side)
        np.add.at(faces_of, side[side >= 0], 1)
        np.add.at(quads_of, side[side >= 0], face_size[side >= 0] == 4)
    assert np.all(faces_of[surrounded] == 6) and np.all(quads_of[surrounded] == 6)


def _hexahedral_walls(mesh, patch: str) -> np.ndarray:
    """Which of a patch's faces are quadrilaterals of hexahedral dual cells."""
    size = np.diff(np.asarray(mesh.face_nodes.offsets))
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    faces = np.zeros(mesh.n_cells, dtype=np.int64)
    quads = np.zeros(mesh.n_cells, dtype=np.int64)
    for side in (owner, neighbour):
        np.add.at(faces, side[side >= 0], 1)
        np.add.at(quads, side[side >= 0], size[side >= 0] == 4)
    mine = np.asarray(mesh.face_patches.indices(patch))
    return (size[mine] == 4) & (faces[owner[mine]] == 6) & (quads[owner[mine]] == 6)


def _hexahedral_wall_share(mesh, patch: str) -> float:
    """The share of a patch's faces that are quadrilaterals of hexahedral dual cells."""
    return float(np.mean(_hexahedral_walls(mesh, patch)))


def test_mapped_walls_layered_round_an_l_give_hexahedra_that_close_at_its_inside_edge(tmp_path):
    # The L's six rectangular faces meshed as structured grids, its two L-shaped ends left to
    # triangles, and layers grown on all eight. At the inside edge the two walls' hexahedra meet
    # and the layer points above the edge are cut with it; the dual must close everywhere and fill
    # the L exactly, and the cells against the grids must be hexahedra.
    pytest.importorskip("OCP")
    from aquaflux.io.cad import CadRegion, read_step
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import CadMesher, FacesNear, PrismLayers
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    # Drawn in millimetres, which is what a STEP file without units is read in.
    fused = BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 200, 100, 100).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 100), 100, 100, 100).Shape(),
    ).Shape()
    unified = ShapeUpgrade_UnifySameDomain(fused)
    unified.Build()
    writer = STEPControl_Writer()
    writer.Transfer(unified.Shape(), STEPControl_AsIs)
    writer.Write(str(tmp_path / "l.step"))
    model = read_step(tmp_path / "l.step")
    region = CadRegion((model.names[0],))
    ends = {"ends": FacesNear(((0.05, 0.0, 0.05), (0.05, 0.1, 0.05)))}
    layers = PrismLayers(first_height=0.002, growth=1.2, count=3)
    mesh = CadMesher(max_size=0.02, feature_angle=np.radians(45), layers=layers).mesh(
        model, region, ends, layered=("walls", "ends"), mapped=("walls",)
    )
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(volume.sum(), 0.2 * 0.1 * 0.1 + 0.1 * 0.1 * 0.1, rtol=1e-12)
    scale = float(np.max(np.asarray(mesh.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    # A wall face with no vertex on one of the L's edges is the dual face of a grid point inside
    # its CAD face: its cell, and the one above it in every layer, a hexahedron.
    offsets = np.asarray(mesh.face_nodes.offsets)
    nodes = np.asarray(mesh.face_nodes.face_node_indices)
    walls = np.asarray(mesh.face_patches.indices("walls"))
    used = np.unique(np.concatenate([nodes[offsets[f] : offsets[f + 1]] for f in walls]))
    topology = model.topology(region)
    ends = topology.vertices[topology.edge_vertices]  # the L's edges are straight
    along = ends[:, 1] - ends[:, 0]
    offset = np.asarray(mesh.node_coords)[used][:, None] - ends[None, :, 0]
    t = np.clip(np.einsum("pek,ek->pe", offset, along) / np.sum(along**2, axis=1), 0.0, 1.0)
    gap = np.linalg.norm(offset - t[..., None] * along[None], axis=2).min(axis=1)
    on_edge = np.zeros(mesh.n_nodes, dtype=bool)
    on_edge[used[gap < 1e-9]] = True
    inside = np.array([not on_edge[nodes[offsets[f] : offsets[f + 1]]].any() for f in walls])
    assert np.count_nonzero(inside) > 100
    assert np.all(_hexahedral_walls(mesh, "walls")[inside])


def test_the_sozzi_lamp_and_pipes_meshed_as_grids_carry_hexahedral_layers():
    # The lamp's cylinder and the two pipes' (each a band) as structured grids, layered with the
    # vessel and the lamp's tip. The pipes meet the vessel at concave edges -- the riser on its
    # cylinder, a saddle, and the outlet on its end plate. Every cell must close; the cells against
    # the grids must be hexahedra; and the dual's first layer boundary on the lamp must sit at the
    # first height from the true wall.
    pytest.importorskip("OCP")
    from pathlib import Path

    from aquaflux.io.cad import CadRegion, Placement, read_step
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import CadMesher, FacesNear, PrismLayers

    drawing = (
        Path(__file__).resolve().parents[2]
        / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
    )
    model = read_step(drawing, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    water = CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
    patches = {
        "inlet": FacesNear(((0.04765, 0.0, 0.8945),)),
        "outlet": FacesNear(((1.739, 0.0, 0.0),)),
        "lamp": FacesNear(((0.4, 0.01, 0.0),)),
        "lamp_tip": FacesNear(((0.81, 0.0, 0.0),)),
        "pipes": FacesNear(((0.0572, 0.0, 0.5), (1.3, 0.0096, 0.0))),
    }
    layers = PrismLayers(first_height=0.0005, growth=1.2, count=5)
    mesh = CadMesher(max_size=0.02, feature_angle=0.785, layers=layers).mesh(
        model, water, patches, layered=("walls", "lamp", "lamp_tip", "pipes"),
        mapped=("lamp", "pipes"),
    )  # fmt: skip
    geometry = mesh.geometry()
    assert np.all(np.asarray(geometry.cell.volume) > 0.0)
    scale = float(np.max(np.asarray(geometry.face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    # All but the faces at the grids' rims, where the dual's faces meet the CAD's edges.
    assert _hexahedral_wall_share(mesh, "lamp") > 0.95
    assert _hexahedral_wall_share(mesh, "pipes") > 0.95
    coords = np.asarray(mesh.node_coords)
    radius = np.hypot(coords[:, 1], coords[:, 2])
    near = (coords[:, 0] > 0.1) & (coords[:, 0] < 0.7) & (np.abs(radius - 0.0105) < 0.001)
    away = model.project(water, coords[near]).distance
    first = away[(away > 0.0002) & (away < 0.0008)]
    assert first.size > 100
    assert float(np.median(first)) == pytest.approx(0.0005, rel=0.03)
