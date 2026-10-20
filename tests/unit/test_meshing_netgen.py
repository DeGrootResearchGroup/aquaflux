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


def test_the_dual_of_a_netgen_l_is_cut_at_its_inside_edge_and_closes(tmp_path):
    # The case that needs the cut: on Netgen's tetrahedra of an L, cells around the inside edge,
    # uncut or cut with their added vertices off the plane between the pieces, were folded enough
    # that the mesh turned a face of each inward (measured at this size). Every cell must close.
    pytest.importorskip("OCP")
    from aquaflux.mesh import closed_cell_residual
    from aquaflux.meshing import PolyhedralDual
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt

    from tests.support.meshes import tetrahedral_mesh

    shape = BRepAlgoAPI_Fuse(
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.2, 0.1, 0.1).Shape(),
        BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0.1), 0.1, 0.1, 0.1).Shape(),
    ).Shape()
    BRepTools.Write_s(shape, str(tmp_path / "l.brep"))
    netgen = NetgenProcess()
    surface = netgen.triangulate(tmp_path / "l.brep", max_size=0.01)
    filled = netgen.fill(surface.points, surface.triangles, max_size=0.01)
    primal = tetrahedral_mesh(filled.points, filled.tetrahedra)
    dual = PolyhedralDual(feature_angle=np.radians(45)).build(primal)
    on_edge = np.isclose(filled.points[:, 0], 0.1) & np.isclose(filled.points[:, 2], 0.1)
    assert dual.n_cells == primal.n_nodes + np.count_nonzero(on_edge)
    volume = np.asarray(dual.geometry().cell.volume)
    assert np.all(volume > 0.0)
    np.testing.assert_allclose(volume.sum(), 0.2 * 0.1 * 0.1 + 0.1 * 0.1 * 0.1, rtol=1e-12)
    scale = float(np.max(np.asarray(dual.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(dual)))) < 1e-12 * scale
