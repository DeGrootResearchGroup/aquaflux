"""A CAD region's topology, face-tagged triangles, file and projection, through the kernel.

Each answer is checked against geometry known independently of the kernel: a box's faces are planes
whose equations and outward normals are written down here, an L-shaped solid has exactly one
concave edge and it is where the two blocks meet, the Sozzi vessel's concave edges are where its
pipes join it (told apart by their radii), and so on.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")

from aquaflux.io.cad import CadModel, CadRegion, Placement, PlaneFace, read_step
from aquaflux.io.cad.kernel import OpenCascade
from OCP.BRep import BRep_Builder
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Shape

SOZZI = (
    Path(__file__).resolve().parents[2]
    / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
)
SWAP_XY = Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
WATER = CadRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), without=("lamp",))
R_BODY, R_PIPE, R_LAMP = 0.0445, 0.00955, 0.010

#: The box [0, 2] x [0, 1] x [0, 1]: its six planes as (axis, coordinate, outward sign).
BOX_PLANES = [(0, 0.0, -1), (0, 2.0, 1), (1, 0.0, -1), (1, 1.0, 1), (2, 0.0, -1), (2, 1.0, 1)]

BOX, L_SHAPE, CYLINDER = CadRegion(("box",)), CadRegion(("box", "top")), CadRegion(("cylinder",))


@pytest.fixture(scope="module")
def model():
    """A box ``[0, 2] x [0, 1] x [0, 1]``, a cube on its ``x < 1`` half (together an L), and a
    cylinder of radius 0.5 and height 1 on the z axis."""
    shapes = {
        "box": BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 2.0, 1.0, 1.0).Shape(),
        "top": BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 1), 1.0, 1.0, 1.0).Shape(),
        "cylinder": BRepPrimAPI_MakeCylinder(0.5, 1.0).Shape(),
    }
    return CadModel(shapes, OpenCascade())


@pytest.fixture(scope="module")
def sozzi():
    return read_step(SOZZI, SWAP_XY)


def _plane_of(face) -> tuple[int, float, int]:
    """The (axis, coordinate, outward sign) of an axis-aligned plane face record."""
    assert isinstance(face, PlaneFace)
    axis = int(np.argmax(np.abs(face.outward_normal)))
    return axis, float(face.point[axis]), int(np.sign(face.outward_normal[axis]))


def _midpoint(topology, edge):
    return topology.vertices[topology.edge_vertices[edge]].mean(axis=0)


def test_a_box_has_six_planes_twelve_convex_edges_and_eight_corners(model):
    topology = model.topology(BOX)
    assert (topology.n_faces, topology.n_edges, len(topology.vertices)) == (6, 12, 8)
    assert sorted(_plane_of(face) for face in topology.faces) == sorted(BOX_PLANES)
    np.testing.assert_allclose(
        topology.edge_angle, math.pi / 2, rtol=1e-12
    )  # every edge of a box is convex
    assert np.all(np.diff(topology.edge_face_offsets) == 2)
    expected_area = {0: 1.0, 1: 2.0, 2: 2.0}
    for face, area in zip(topology.faces, topology.face_area, strict=True):
        assert area == pytest.approx(expected_area[_plane_of(face)[0]])


def test_faces_come_in_the_order_the_solid_describes_them(model):
    # A single solid's region is the solid itself, so its faces are its own, in its own order.
    described = model.describe("box").faces
    region = model.topology(BOX).faces
    assert len(region) == len(described)
    for mine, theirs in zip(region, described, strict=True):
        np.testing.assert_allclose(mine.point, theirs.point)
        np.testing.assert_allclose(mine.outward_normal, theirs.outward_normal)


def test_an_l_has_one_concave_edge_where_the_blocks_meet(model):
    topology = model.topology(L_SHAPE)
    concave = np.flatnonzero(topology.edge_angle < -1e-9)
    assert concave.size == 1
    np.testing.assert_allclose(topology.edge_angle[concave], -math.pi / 2, rtol=1e-12)
    middle = _midpoint(topology, concave[0])
    np.testing.assert_allclose(middle[[0, 2]], [1.0, 1.0])  # along y at x = 1, z = 1
    # The fused blocks keep the faces their shared plane cut; where two of those meet, nothing folds.
    flat = np.flatnonzero(np.abs(topology.edge_angle) < 1e-9)
    assert flat.size == 3
    features = topology.feature_edges(math.radians(30))
    assert set(features) == set(range(topology.n_edges)) - set(flat)


def test_a_cylinders_seam_is_marked_and_is_not_a_feature(model):
    topology = model.topology(CYLINDER)
    (seam,) = np.flatnonzero(topology.seam)
    assert topology.faces_of_edge(seam).size == 1
    assert math.isnan(topology.edge_angle[seam])
    rims = np.flatnonzero(~topology.seam)
    np.testing.assert_allclose(topology.edge_angle[rims], math.pi / 2, rtol=1e-12)
    assert seam not in topology.feature_edges(math.radians(30))
    np.testing.assert_allclose(topology.edge_length[rims], math.pi, rtol=1e-12)  # circles r = 0.5
    (side,) = [f for f, face in enumerate(topology.faces) if not isinstance(face, PlaneFace)]
    assert set(topology.edges_of_face(side)) == set(range(topology.n_edges))


def test_the_vessels_concave_edges_are_where_its_pipes_join_it(sozzi):
    # One per pipe: the inlet riser meets the vessel's cylinder, the outlet its end plate. Every
    # other fold of the water is convex, the lamp's passage through the far end plate included.
    topology = sozzi.topology(WATER)
    concave = np.flatnonzero(topology.edge_angle < -1e-9)
    pipes = []
    for edge in concave:
        faces = topology.faces_of_edge(edge)
        radii = [getattr(topology.faces[f], "radius", None) for f in faces]
        (pipe,) = [f for f, r in zip(faces, radii, strict=True) if r == pytest.approx(R_PIPE)]
        (vessel,) = [f for f in faces if f != pipe]
        assert getattr(topology.faces[vessel], "radius", None) in (pytest.approx(R_BODY), None)
        pipes.append(pipe)
    assert len(pipes) == 2 and len(set(pipes)) == 2
    pairs = {tuple(p) for p in topology.face_neighbours().tolist()}
    assert all(tuple(sorted(topology.faces_of_edge(e).tolist())) in pairs for e in concave)


@pytest.mark.parametrize("facet_size", [None, 0.3])
def test_each_triangle_lies_on_the_face_it_is_tagged_with(model, facet_size):
    topology = model.topology(BOX)
    surface = model.face_triangles(BOX, chord=1e-3, facet_size=facet_size)
    corners = surface.triangles
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    for f, face in enumerate(topology.faces):
        axis, coordinate, sign = _plane_of(face)
        mine = surface.face == f
        np.testing.assert_allclose(corners[mine][..., axis], coordinate)
        assert np.all(normals[mine][:, axis] * sign > 0.0)
        area = 0.5 * np.linalg.norm(normals[mine], axis=1).sum()
        assert area == pytest.approx(topology.face_area[f], rel=1e-12)


def test_the_vessels_triangles_cover_each_face_to_its_area(sozzi):
    topology = sozzi.topology(WATER)
    surface = sozzi.face_triangles(WATER, chord=2e-5, facet_size=0.03)
    corners = surface.triangles
    doubled = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    areas = np.bincount(surface.face, weights=0.5 * doubled, minlength=topology.n_faces)
    np.testing.assert_allclose(areas, topology.face_area, rtol=2e-3)


def test_the_file_keeps_the_face_order(model, sozzi, tmp_path):
    # A mesher reads the file and numbers faces as it explores them; that must be this order.
    for owner, region in ((model, L_SHAPE), (sozzi, WATER)):
        path = owner.write_brep(region, tmp_path / "region.brep")
        shape = TopoDS_Shape()
        assert BRepTools.Read_s(shape, str(path), BRep_Builder())
        areas, explorer = [], TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            properties = GProp_GProps()
            BRepGProp.SurfaceProperties_s(explorer.Current(), properties)
            areas.append(properties.Mass())
            explorer.Next()
        np.testing.assert_allclose(areas, owner.topology(region).face_area, rtol=1e-9)


def test_points_land_on_the_nearest_face_with_its_outward_normal(model):
    topology = model.topology(BOX)
    planes = [_plane_of(face) for face in topology.faces]
    points = np.array([[2.4, 0.5, 0.5], [1.0, -0.3, 0.6], [0.7, 0.4, 0.9]])
    projection = model.project(BOX, points)
    expected = [(0, 2.0, 1), (1, 0.0, -1), (2, 1.0, 1)]
    for i, (axis, coordinate, sign) in enumerate(expected):
        assert planes[projection.face[i]] == (axis, coordinate, sign)
        landed = points[i].copy()
        landed[axis] = coordinate
        np.testing.assert_allclose(projection.points[i], landed, atol=1e-12)
        np.testing.assert_allclose(projection.normal[i], sign * np.eye(3)[axis], atol=1e-12)
    np.testing.assert_allclose(projection.distance, [0.4, 0.3, 0.1], rtol=1e-12)


def test_a_point_is_moved_onto_the_face_asked_for_not_the_nearest(model):
    topology = model.topology(BOX)
    planes = [_plane_of(face) for face in topology.faces]
    far = planes.index((0, 0.0, -1))
    projection = model.project(BOX, [[1.9, 0.5, 0.5]], faces=[far])
    np.testing.assert_allclose(projection.points[0], [0.0, 0.5, 0.5], atol=1e-12)
    assert projection.face[0] == far
    assert projection.distance[0] == pytest.approx(1.9)


def test_a_point_nearest_an_edge_takes_the_first_face_meeting_there(model):
    topology = model.topology(BOX)
    planes = [_plane_of(face) for face in topology.faces]
    projection = model.project(BOX, [[2.3, 0.5, 1.4]])
    np.testing.assert_allclose(projection.points[0], [2.0, 0.5, 1.0], atol=1e-12)
    assert projection.face[0] == min(planes.index((0, 2.0, 1)), planes.index((2, 1.0, 1)))
    assert projection.distance[0] == pytest.approx(0.5)


def test_a_point_is_moved_onto_the_edge_asked_for_and_given_the_first_face_there(model):
    # The edge where x = 2 meets z = 1, asked for from a point far nearer the face y = 0.
    topology = model.topology(BOX)
    planes = [_plane_of(face) for face in topology.faces]
    right, top = planes.index((0, 2.0, 1)), planes.index((2, 1.0, 1))
    edge = next(
        e
        for e in range(topology.n_edges)
        if set(topology.faces_of_edge(e).tolist()) == {right, top}
    )
    projection = model.project(BOX, [[1.5, 0.3, 0.2]], edges=[edge])
    np.testing.assert_allclose(projection.points[0], [2.0, 0.3, 1.0], atol=1e-12)
    assert projection.face[0] == min(right, top)
    assert projection.distance[0] == pytest.approx(math.hypot(0.5, 0.8))


def test_a_point_is_moved_onto_a_circular_edge(model):
    # The cylinder's top rim: radius 0.5 at z = 1, reached along the point's own direction.
    topology = model.topology(CYLINDER)
    rims = [
        e
        for e in range(topology.n_edges)
        if not topology.seam[e]
        and topology.edge_length[e] > 0
        and abs(_midpoint(topology, e)[2] - 1.0) < 1e-9
    ]
    assert len(rims) == 1
    projection = model.project(CYLINDER, [[0.3, 0.1, 0.9]], edges=rims)
    expected = np.array([0.3, 0.1]) / math.hypot(0.3, 0.1) * 0.5
    np.testing.assert_allclose(projection.points[0], [*expected, 1.0], atol=1e-9)


def test_an_edge_index_out_of_range_or_faces_and_edges_together_are_refused(model):
    n_edges = model.topology(BOX).n_edges
    with pytest.raises(ValueError, match=rf"edge index is outside \[0, {n_edges}\)"):
        model.project(BOX, [[0.5, 0.5, 0.5]], edges=[n_edges])
    with pytest.raises(ValueError, match="onto faces or onto edges, not both"):
        model.project(BOX, [[0.5, 0.5, 0.5]], faces=[0], edges=[0])


def test_near_the_lamp_the_outward_normal_points_into_the_lamp(sozzi):
    # Out of the water is into the lamp sleeve: toward the lamp's axis, the x axis.
    projection = sozzi.project(WATER, [[0.4, 0.0, 0.015]])
    np.testing.assert_allclose(projection.points[0], [0.4, 0.0, R_LAMP], atol=1e-9)
    np.testing.assert_allclose(projection.normal[0], [0.0, 0.0, -1.0], atol=1e-9)


def test_a_face_index_out_of_range_is_refused(model):
    with pytest.raises(ValueError, match=r"outside \[0, 6\)"):
        model.project(BOX, [[0.5, 0.5, 0.5]], faces=[6])


def test_a_region_needs_a_solid_and_cannot_take_one_out_of_itself():
    with pytest.raises(ValueError, match="at least one solid"):
        CadRegion(())
    with pytest.raises(ValueError, match="both in the region and taken out"):
        CadRegion(("a", "b"), without=("b",))
