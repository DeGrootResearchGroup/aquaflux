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
    Polygons,
    PrismLayers,
    PyramidCaps,
    check_closed_surface,
    element_mesh,
    netgen_available,
)

from tests.support.meshes import box_surface, voxel_surface

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


def test_the_fill_meshes_the_gap_between_two_surfaces():
    outer_points, outer = box_surface(0.0, 1.0, 4, into_box=False)
    inner_points, inner = box_surface(0.35, 0.65, 2, into_box=True, offset=len(outer_points))
    points = np.vstack([outer_points, inner_points])
    triangles = np.vstack([outer, inner])
    filled = NetgenProcess().fill(points, triangles, max_size=0.3)
    np.testing.assert_array_equal(filled.points[: len(points)], points)
    volumes = _signed_volumes(filled.points, filled.tetrahedra)
    assert np.all(volumes > 0.0)
    np.testing.assert_allclose(volumes.sum(), 1.0 - 0.3**3, rtol=1e-12)


@pytest.mark.parametrize("min_top_share", [0.0, None])
def test_capped_hexahedral_layers_meeting_at_an_edge_leave_a_surface_netgen_fills(min_top_share):
    # A cube of quadrilaterals an eighth wide, hexahedral layers almost as thick: where two walls
    # meet, the top face beside the edge collapses to a sliver, and the pyramid capping it crosses
    # the other wall's top. Netgen stops on the overlapping surface; with the top's edges kept to a
    # share of the wall's (the default) it fills it, and the cells close and fill the cube.
    from aquaflux.mesh import closed_cell_residual

    points, quads = voxel_surface([(0, 0, 0)], 8)
    share = {} if min_top_share is None else {"min_top_share": min_top_share}
    layers = PrismLayers(first_height=0.06125, growth=1.2, count=1, **share)
    grown = layers.extrude(points, Polygons(quads=quads), np.ones(len(quads), dtype=bool))
    caps = PyramidCaps.on(grown.points, grown.inner.quads[:, ::-1], 0.25)
    points = np.vstack([grown.points, caps.apexes])
    used = np.unique(caps.sides)
    position = np.full(points.shape[0], -1)
    position[used] = np.arange(used.size)
    if min_top_share == 0.0:
        with pytest.raises(MeshingError, match="boundary mesh is overlapping"):
            NetgenProcess().fill(points[used], position[caps.sides], max_size=0.125)
        return
    filled = NetgenProcess().fill(points[used], position[caps.sides], max_size=0.125)
    back = np.concatenate([used, points.shape[0] + np.arange(filled.points.shape[0] - used.size)])
    mesh = element_mesh(
        np.vstack([points, filled.points[used.size :]]),
        tetrahedra=back[filled.tetrahedra],
        hexahedra=grown.hexahedra,
        pyramids=caps.pyramids,
    )
    assert float(np.max(np.abs(np.asarray(closed_cell_residual(mesh))))) < 1e-14
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    assert volume.sum() == pytest.approx(1.0, rel=1e-12)


def test_a_fill_of_the_wrong_region_is_refused():
    # The hole's walls wound out of the hole rather than into it: still closed and of positive total
    # volume, so the surface check passes it, and Netgen fills straight through the hole, keeping
    # every supplied point. Only the check that the tetrahedra are bounded by the supplied surface
    # catches it.
    outer_points, outer = box_surface(0.0, 1.0, 4, into_box=False)
    inner_points, inner = box_surface(0.35, 0.65, 2, into_box=False, offset=len(outer_points))
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


def test_a_mesh_files_mesh_is_written_with_its_patches_on_their_faces(tmp_path, monkeypatch):
    # The whole path a user takes: a mesh file, meshed by `aquaflux mesh mesh.yaml` into a .aqmesh
    # file and an OpenFOAM polyMesh, read back from each. Every boundary vertex must lie on the
    # drawing, every boundary face on a CAD face of its patch with nearly all of its faces' area,
    # and every cell must close.
    pytest.importorskip("OCP")
    import contextlib
    import io
    import os
    from pathlib import Path

    import yaml
    from aquaflux.__main__ import main
    from aquaflux.io import read_aqmesh, read_openfoam
    from aquaflux.io.cad import CadRegion, Placement, read_step
    from aquaflux.mesh import closed_cell_residual

    drawing = (
        Path(__file__).resolve().parents[2]
        / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
    )
    swap = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    document = {
        "drawing": {
            # Relative to the mesh file, as a user writes it.
            "file": os.path.relpath(drawing, tmp_path),
            "placement": {"kind": "CadPlacement", "matrix": swap},
        },
        "region": {"solids": ["reactor_body", "inlet_pipe", "outlet_pipe"], "without": ["lamp"]},
        "sizes": {"max_size": 0.03, "feature_angle": 0.785},
        "patches": {
            "inlet": {"kind": "FacesNear", "points": [[0.04765, 0.0, 0.8945]]},
            "outlet": {"kind": "FacesNear", "points": [[1.739, 0.0, 0.0]]},
            "lamp": {"kind": "FacesOn", "solids": ["lamp"]},
        },
        "output": {"path": "out/sozzi.aqmesh", "openfoam": "out"},
    }
    (tmp_path / "mesh.yaml").write_text(yaml.safe_dump(document))
    # Run from elsewhere: the drawing is found from the mesh file's directory, not from here.
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = main(["mesh", str(tmp_path / "mesh.yaml")])
    assert status == 0, out.getvalue()
    mesh = read_aqmesh(tmp_path / "out" / "sozzi.aqmesh")
    assert {
        name for name in mesh.face_patches.names
        if mesh.face_patches.size(name) and mesh.face_patches.is_boundary_patch(name, mesh.face_cells)
    } == set(SOZZI_PATCH_FACES)  # fmt: skip
    assert read_openfoam(tmp_path / "out").n_cells == mesh.n_cells > 1000
    assert f"{mesh.n_cells} cells (3D)" in out.getvalue()
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
    # A core of 0.2 cubes, halved to 0.1 towards the box's walls and kept 0.15 clear of them;
    # Netgen fills the gap between its capped surface and the walls, keeping both. The dual of the
    # whole primal mesh must fill the box exactly, and every point the core's hexahedra surround
    # on all sides must become a cube of their size (the dual of a block of equal cubes is the
    # same block shifted half a cube).
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import OctreeCore, PolyhedralDual, element_mesh

    wall_points, wall = box_surface(0.0, 1.0, 8, into_box=False)
    core = OctreeCore(size=0.2, wall_size=0.1, clearance=0.15).within(
        wall_points, wall, cap_height=0.25, boundary_size=0.125
    )
    assert set(np.unique(core.octree.level[core.kept]).tolist()) == {0, 1}
    # The fill wants the surface's own points only; the core's inner points follow its result.
    on_surface = np.unique(core.surface)
    number = np.empty(core.points.shape[0], dtype=np.int64)
    number[on_surface] = len(wall_points) + np.arange(on_surface.size)
    filled = NetgenProcess().fill(
        np.vstack([wall_points, core.points[on_surface]]),
        np.vstack([wall, number[core.surface]]),
        max_size=0.125,
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
    assert np.count_nonzero(surrounded) > 50
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


def _l_drawing(tmp_path):
    """An L of two boxes, 0.2 x 0.1 x 0.1 m and a 0.1 m cube standing on its first half, drawn by
    OCP and read back from a STEP file: the model and the region of its one solid."""
    pytest.importorskip("OCP")
    from aquaflux.io.cad import CadRegion, read_step
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
    return model, CadRegion((model.names[0],))


def _depth_in_the_l(points):
    """How far each point inside the L lies from its boundary: from the L's box, or from the box
    cut out of it above its second half."""
    to_box = np.min(np.concatenate([points, [0.2, 0.1, 0.2] - points], axis=1), axis=1)
    cut_low, cut_high = np.array([0.1, -1.0, 0.1]), np.array([1.0, 1.0, 1.0])
    outside_cut = np.maximum(np.maximum(cut_low - points, points - cut_high), 0.0)
    return np.minimum(to_box, np.linalg.norm(outside_cut, axis=1))


def test_mapped_walls_layered_round_an_l_give_hexahedra_that_close_at_its_inside_edge(tmp_path):
    # The L's six rectangular faces meshed as structured grids, its two L-shaped ends left to
    # triangles, and layers grown on all eight. At the inside edge the two walls' hexahedra meet
    # and the layer points above the edge are cut with it; the dual must close everywhere and fill
    # the L exactly, and the cells against the grids must be hexahedra.
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import CadMesher, FacesNear, PrismLayers

    model, region = _l_drawing(tmp_path)
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


def test_a_hex_core_inside_the_layers_of_an_l_joins_them_through_polyhedra(tmp_path):
    # Prism layers on every wall of the L, and inside them a core of 0.02 m cubes halved to
    # 0.01 m towards the layers' top and kept 0.02 m clear of it (the L is only 0.1 m across, so
    # its core is all wall cubes). The dual must close everywhere
    # and fill the L exactly; the core's interior points become cubes of their hexahedra's size
    # with six quadrilateral faces; and no cube stands nearer the wall than the clearance plus
    # half the layers' thickness.
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import CadMesher, OctreeCore, PrismLayers

    model, region = _l_drawing(tmp_path)
    layers = PrismLayers(first_height=0.002, growth=1.2, count=3)
    core = OctreeCore(size=0.02, wall_size=0.01, clearance=0.02)
    mesh = CadMesher(max_size=0.01, feature_angle=np.radians(45), layers=layers, core=core).mesh(
        model, region, layered=("walls",)
    )
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(volume.sum(), 0.2 * 0.1 * 0.1 + 0.1 * 0.1 * 0.1, rtol=1e-12)
    scale = float(np.max(np.asarray(mesh.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    size = np.diff(np.asarray(mesh.face_nodes.offsets))
    faces_of = np.zeros(mesh.n_cells, dtype=np.int64)
    quads_of = np.zeros(mesh.n_cells, dtype=np.int64)
    for side in (mesh.face_cells.owner, mesh.face_cells.neighbour):
        side = np.asarray(side)
        np.add.at(faces_of, side[side >= 0], 1)
        np.add.at(quads_of, side[side >= 0], size[side >= 0] == 4)
    cubes = (faces_of == 6) & (quads_of == 6) & np.isclose(volume, 0.01**3, rtol=1e-9)
    assert np.count_nonzero(cubes) > 50
    centroid = np.asarray(mesh.geometry().cell.centroid)
    thickness = float(layers.offsets()[-1])
    assert _depth_in_the_l(centroid[cubes]).min() >= 0.02 + 0.5 * thickness


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


def test_the_sozzi_pipes_cores_are_structured_o_grids_with_a_junction_at_each_vessel():
    # Both pipes found as pipes (a cylinder the water fills, nothing inside), their walls mapped and
    # layered and their cores filled with O-grids: open at the inlet and outlet, stopping short of
    # the junctions with the vessel (the riser's saddle on its cylinder, the outlet on its end
    # plate). Without the cores the pipes' faces reach 87 degrees non-orthogonal at the 99th
    # percentile 47 (five draws, 2026-10-20); with them their cells along the axis are hexahedra.
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
    # The riser's grid runs from its saddle; unflattened, its rows near the junction stay saddle
    # shaped and the caps on the junction's disc cross (Netgen: "boundary mesh is overlapping").
    mesh = CadMesher(max_size=0.02, feature_angle=0.785, layers=layers).mesh(
        model, water, patches, layered=("walls", "lamp", "lamp_tip", "pipes"), mapped=("lamp",),
        pipes=("pipes",),
    )  # fmt: skip
    geometry = mesh.geometry()
    assert np.all(np.asarray(geometry.cell.volume) > 0.0)
    scale = float(np.max(np.asarray(geometry.face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    inside = neighbour >= 0
    centroid = np.asarray(geometry.cell.centroid)
    joining = centroid[neighbour[inside]] - centroid[owner[inside]]
    normal = np.asarray(geometry.face.normal)[inside]
    cosine = np.abs(np.einsum("ij,ij->i", joining, normal)) / np.linalg.norm(joining, axis=1)
    angle = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    where = np.asarray(geometry.face.centroid)[inside]
    in_pipes = (where[:, 0] > 0.9) | (where[:, 2] > 0.07)
    assert float(np.percentile(angle[in_pipes], 99)) < 7.0  # 6.0; 8.2 with saddle-shaped rows
    # Along each pipe's axis, away from its junction, every cell is a hexahedron.
    faces = np.bincount(owner, minlength=mesh.n_cells) + np.bincount(
        neighbour[inside], minlength=mesh.n_cells
    )
    riser = (np.hypot(centroid[:, 0] - 0.0476, centroid[:, 1]) < 0.002) & (centroid[:, 2] > 0.1)
    outlet = (np.hypot(centroid[:, 1], centroid[:, 2]) < 0.002) & (centroid[:, 0] > 0.95)
    for along in (riser, outlet):
        assert np.count_nonzero(along) > 50
        assert np.all(faces[along] == 6)
    area = np.asarray(geometry.face.area)
    for name in ("inlet", "outlet"):
        end = float(area[np.asarray(mesh.face_patches.indices(name))].sum())
        assert end == pytest.approx(np.pi * 0.0096**2, rel=0.03), name


def test_the_planarizer_in_the_mesher_leaves_the_layers_at_their_heights_from_the_wall():
    # Layers on the lamp. The flattening pass may move the tetrahedra's centroids only: every dual
    # vertex inside the layers, and every boundary vertex, must be where the mesher put it without
    # the pass. Two iterations are enough to show which vertices move.
    pytest.importorskip("OCP")
    from pathlib import Path

    from aquaflux.io.cad import CadRegion, Placement, read_step
    from aquaflux.meshing import CadMesher, FacePlanarizer, FacesNear, FacesOn, PrismLayers

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
    layers = PrismLayers(first_height=0.0005, growth=1.2, count=5)

    def mesh(planarizer):
        mesher = CadMesher(max_size=0.03, feature_angle=0.785, layers=layers, planarizer=planarizer)
        return mesher.mesh(model, water, patches, layered=("lamp",))

    plain, flat = mesh(None), mesh(FacePlanarizer(max_iterations=2))
    before, after = np.asarray(plain.node_coords), np.asarray(flat.node_coords)
    moved = np.linalg.norm(after - before, axis=1) > 1e-15
    # The layers fill the 0.0005 * (1 + 1.2 + ... + 1.2^4) = 3.72 mm round the lamp (radius 10.5 mm).
    radius = np.hypot(before[:, 1], before[:, 2])
    lengthwise = (before[:, 0] > 0.1) & (before[:, 0] < 0.7)
    in_layers = lengthwise & (radius > 0.0105) & (radius < 0.0105 + 0.0037)
    assert in_layers.sum() > 100
    assert moved.sum() > 100 and not np.any(moved & in_layers)
    offsets = np.asarray(plain.face_nodes.offsets)
    nodes = np.asarray(plain.face_nodes.face_node_indices)
    boundary = np.flatnonzero(np.asarray(plain.face_cells.neighbour) < 0)
    on_boundary = np.unique(np.concatenate([nodes[offsets[f] : offsets[f + 1]] for f in boundary]))
    assert not np.any(moved[on_boundary])
    assert np.all(np.asarray(flat.geometry().cell.volume) > 0.0)


def test_netgen_divides_the_sozzi_pipes_rims_as_asked_so_both_are_pipes(tmp_path):
    # At maxh 0.0035 Netgen divides every rim of both pipes into 17 (2026-10-21), which no O-grid
    # joins. Asked for the rim edges' divisions -- numbered as CadModel.topology numbers edges,
    # the riser's inlet rim drawn as two edges -- it gives both rims 20 points, equally spaced: by
    # arc length to 1.0002 on the riser's saddle, whose curve parameter is not its arc length (equal
    # steps in the parameter give 1.004).
    pytest.importorskip("OCP")
    from pathlib import Path

    from aquaflux.io.cad import CadRegion, Placement, read_step
    from aquaflux.meshing.mapped import _loops
    from aquaflux.meshing.pipes import find_pipes, rim_divisions

    drawing = (
        Path(__file__).resolve().parents[2]
        / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
    )
    model = read_step(drawing, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    water = CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
    topology = model.topology(water)
    brep = model.write_brep(water, tmp_path / "water.brep")
    netgen = NetgenProcess()
    candidates = np.arange(topology.n_faces)
    own = netgen.triangulate(brep, max_size=0.0035)
    assert find_pipes(topology, own.points, own.triangles, own.face, candidates)[0] == []
    divisions = rim_divisions(topology, own.triangles, own.face, candidates)
    assert sorted(divisions.values()) == [5, 15, 20, 20, 20]  # the inlet rim's edges 15 and 5
    surface = netgen.triangulate(brep, max_size=0.0035, edge_divisions=divisions)
    pipes, _ = find_pipes(topology, surface.points, surface.triangles, surface.face, candidates)
    assert len(pipes) == 2
    for pipe in pipes:
        for loop in _loops(surface.triangles[surface.face == pipe]):
            assert loop.size == 20
            step = np.linalg.norm(np.diff(surface.points[np.append(loop, loop[0])], axis=0), axis=1)
            assert step.max() < 1.001 * step.min()


def test_netgen_places_an_edges_points_at_the_fractions_asked_from_the_end_named(tmp_path):
    # The box's edges are straight, so a point at a fraction of one's length is at that fraction of
    # the way along it: measured from the end nearer the point named, whichever end Netgen starts.
    pytest.importorskip("OCP")
    from aquaflux.io.cad import CadRegion, read_step
    from aquaflux.meshing import EdgePoints
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100, 100, 100).Shape(), STEPControl_AsIs)
    writer.Write(str(tmp_path / "box.step"))
    model = read_step(tmp_path / "box.step")
    region = CadRegion((model.names[0],))
    topology = model.topology(region)
    brep = model.write_brep(region, tmp_path / "box.brep")
    fractions = (0.1, 0.15, 0.5, 0.9)
    for edge in (0, 5):
        a, b = topology.vertices[topology.edge_vertices[edge]]
        for start, end in ((a, b), (b, a)):
            asked = {edge: EdgePoints(fractions, tuple(start))}
            surface = NetgenProcess().triangulate(brep, max_size=0.03, edge_divisions=asked)
            offset = surface.points - start
            along = offset @ (end - start) / np.sum((end - start) ** 2)
            off = np.linalg.norm(offset - along[:, None] * (end - start), axis=1)
            on = np.sort(along[(off < 1e-9) & (along > 1e-9) & (along < 1 - 1e-9)])
            np.testing.assert_allclose(on, fractions, atol=1e-6)


def _bent_duct(tmp_path):
    """A duct 0.2 m wide and 0.1 m high: a 0.3 m leg along ``y``, a quarter bend of centreline
    radius 0.4 m about ``z``, and another 0.3 m leg along ``-x``, drawn by OCP as two boxes and a
    revolved rectangle, so its sides are planes and, in the bend, two cylinders and two flat
    sectors."""
    pytest.importorskip("OCP")
    from aquaflux.io.cad import CadRegion, read_step
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    inner, outer, height, leg = 300.0, 500.0, 100.0, 300.0  # millimetres
    rectangle = BRepBuilderAPI_MakePolygon(
        gp_Pnt(inner, 0, 0), gp_Pnt(outer, 0, 0), gp_Pnt(outer, 0, height),
        gp_Pnt(inner, 0, height), True,
    ).Wire()  # fmt: skip
    face = BRepBuilderAPI_MakeFace(rectangle).Face()
    axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    bend = BRepPrimAPI_MakeRevol(face, axis, np.pi / 2).Shape()
    first = BRepPrimAPI_MakeBox(gp_Pnt(inner, -leg, 0), gp_Pnt(outer, 0, height)).Shape()
    second = BRepPrimAPI_MakeBox(gp_Pnt(-leg, inner, 0), gp_Pnt(0, outer, height)).Shape()
    shape = BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(first, bend).Shape(), second).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(tmp_path / "duct.step"))
    model = read_step(tmp_path / "duct.step")
    return model, CadRegion((model.names[0],))


def test_a_bent_duct_is_found_and_meshed_as_hexahedra_round_its_bend(tmp_path):
    # Its three segments are found as one duct, its sides gridded, layered, and its core filled
    # cross-section by cross-section, open at its inlet and outlet: no tetrahedra at all. Every
    # cell is a hexahedron's six faces but at the layers' top's corners (a triangular prism and a
    # seven-faced cell beside it, per corner per row); the cross-sections round the bend are radial,
    # so the faces are as nearly orthogonal as in the straight legs.
    from aquaflux.mesh import closed_cell_residual, face_non_orthogonality
    from aquaflux.meshing import CadMesher, FacesNear, PrismLayers, find_ducts

    model, region = _bent_duct(tmp_path)
    topology = model.topology(region)
    patches = {"inlet": FacesNear(((0.4, -0.3, 0.05),)), "outlet": FacesNear(((-0.3, 0.4, 0.05),))}
    layers = PrismLayers(first_height=0.002, growth=1.2, count=3)
    mesher = CadMesher(max_size=0.03, feature_angle=np.radians(45), layers=layers)
    mesh = mesher.mesh(model, region, patches, layered=("walls",))
    geometry = mesh.geometry()
    volume = np.asarray(geometry.cell.volume)
    assert np.all(volume > 0.0)
    scale = float(np.max(np.asarray(geometry.face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    # The legs exactly, the bend between the polygon through its rows and the true arc.
    legs, bend = 2 * 0.2 * 0.1 * 0.3, 0.25 * np.pi * (0.5**2 - 0.3**2) * 0.1
    assert legs + 0.995 * bend < volume.sum() < legs + bend
    for name in ("inlet", "outlet"):
        area = np.asarray(geometry.face.area)[np.asarray(mesh.face_patches.indices(name))].sum()
        assert area == pytest.approx(0.2 * 0.1, rel=1e-12)
    owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
    faces = np.bincount(owner, minlength=mesh.n_cells)
    faces += np.bincount(neighbour[neighbour >= 0], minlength=mesh.n_cells)
    assert set(np.unique(faces).tolist()) == {5, 6, 7}
    assert np.count_nonzero(faces == 5) == np.count_nonzero(faces == 7)
    assert np.mean(faces == 6) > 0.9
    assert float(np.max(np.asarray(face_non_orthogonality(mesh)))) < 20.0
    # It was one duct of three segments: a leg, the bend, the other leg.
    surface = NetgenProcess().triangulate(
        model.write_brep(region, tmp_path / "duct.brep"), max_size=0.03
    )
    ducts, _ = find_ducts(
        topology, surface.points, surface.triangles, surface.face, np.arange(topology.n_faces),
        np.radians(45), edge_counts=False,
    )  # fmt: skip
    assert [duct.sides.shape for duct in ducts] == [(3, 4)]
