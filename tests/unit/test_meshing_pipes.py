"""Pipes: found among a region's faces, and their cores filled with structured O-grids.

The O-grid is checked inside rings round a regular polygon's prism, which it must fill exactly,
meeting the rings' own quadrilaterals. The search for pipes is checked on a drawn cylinder and its
two ends, each refusal on a shape that has only that fault. The whole mesher is run on that
cylinder with its two ends open (an inlet and an outlet): every cell is then structured, so no
tetrahedra are needed and Netgen is not run, and the solver's mesh must be hexahedra throughout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.io.cad import CadRegion, CadTopology, CylinderFace, PlaneFace, Projection, TorusFace
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import (
    CadMesher,
    FacesNear,
    PipeCore,
    PrismLayers,
    SurfaceTriangulation,
    element_mesh,
)
from aquaflux.meshing.pipes import find_pipes, rim_divisions

N, RADIUS, HEIGHT, ROWS = 12, 4.0, 10.0, 4


def _rings(n=N, radius=RADIUS, rows=ROWS, height=HEIGHT):
    angle = 2 * np.pi * np.arange(n) / n
    points = np.array(
        [
            [radius * np.cos(t), radius * np.sin(t), j * height / rows]
            for j in range(rows + 1)
            for t in angle
        ]
    )
    return points, np.arange((rows + 1) * n).reshape(rows + 1, n)


def _polygon_prism_volume(n=N, radius=RADIUS, height=HEIGHT):
    return 0.5 * n * radius**2 * np.sin(2 * np.pi / n) * height


@pytest.mark.parametrize("core", [PipeCore(), PipeCore(core_bow=0.6, shell=3, core_size=0.4)])
def test_the_o_grid_fills_the_rings_exactly_and_meets_their_quadrilaterals(core):
    points, rings = _rings()
    filled = core.fill(points, rings)
    mesh = element_mesh(filled.points, hexahedra=filled.hexahedra)
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    assert volume.sum() == pytest.approx(_polygon_prism_volume(), rel=1e-12)
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12
    # The O-grid's faces round it are the rings' quadrilaterals, and its ends its cross-sections.
    offsets = np.asarray(mesh.face_nodes.offsets)
    nodes = np.asarray(mesh.face_nodes.face_node_indices)
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    found = {tuple(sorted(nodes[offsets[f] : offsets[f + 1]])) for f in boundary}
    closed = np.concatenate([rings, rings[:, :1]], axis=1)
    ring_quads = {
        tuple(sorted((closed[j, i], closed[j, i + 1], closed[j + 1, i + 1], closed[j + 1, i])))
        for j in range(ROWS) for i in range(N)
    }  # fmt: skip
    ends = {tuple(sorted(q)) for q in np.concatenate([filled.first, filled.last]).tolist()}
    assert found == ring_quads | ends


def test_the_core_has_a_corner_toward_every_fourth_ring_point():
    # The square core's first corner lies on the line from the axis to the ring's first point, at
    # sqrt(2) core_size times the radius; the core takes the points after the ring's.
    points, rings = _rings()
    filled = PipeCore(core_size=0.4).fill(points, rings[:2])
    core = filled.points[points.shape[0] : points.shape[0] + 16]  # the first section's 4 x 4
    distance = np.linalg.norm(core[:, :2], axis=1)
    corners = np.isclose(distance, math.sqrt(2) * 0.4 * RADIUS)
    assert np.count_nonzero(corners) == 4
    angles = np.sort(np.degrees(np.arctan2(core[corners, 1], core[corners, 0])) % 360)
    np.testing.assert_allclose(angles, [0, 90, 180, 270], atol=1e-9)


@pytest.mark.parametrize("bow", [0.0, 0.8])
def test_each_ring_points_radial_line_runs_to_the_core_node_across_from_it(bow):
    # In a cross-section each ring point is joined in to the core by one line of the shell, which
    # must end within half a ring step of its own angle: a shell joined one point round is twisted.
    points, rings = _rings()
    filled = PipeCore(core_bow=bow).fill(points, rings[:2])
    ring = set(rings[0].tolist())
    step = 2 * np.pi / N
    for quad in filled.first.tolist():
        for a, b in zip(quad, quad[1:] + quad[:1], strict=True):
            if (a in ring) == (b in ring):
                continue
            outside, inside = (a, b) if a in ring else (b, a)
            turn = np.arctan2(*filled.points[inside, 1::-1]) - np.arctan2(
                *filled.points[outside, 1::-1]
            )
            assert abs((turn + np.pi) % (2 * np.pi) - np.pi) < 0.5 * step


def test_a_bowed_core_moves_its_sides_middle_out_toward_the_corners_circle():
    points, rings = _rings(n=8)
    for bow in (0.0, 0.5):
        filled = PipeCore(core_size=0.5, core_bow=bow).fill(points, rings[:2])
        core = filled.points[points.shape[0] : points.shape[0] + 9]  # 3 x 3, its middles on sides
        distance = np.sort(np.linalg.norm(core[:, :2], axis=1))
        half = 0.5 * RADIUS
        middle = half + bow * (math.sqrt(2) - 1) * half
        np.testing.assert_allclose(distance[1:5], middle, rtol=1e-12)


@pytest.mark.parametrize(("n", "rows"), [(10, 3), (12, 1)])
def test_rings_an_o_grid_cannot_fill_are_refused(n, rows):
    points, rings = _rings(n=n)
    with pytest.raises(ValueError, match="at least two rings of a multiple of four"):
        PipeCore().fill(points, rings[:rows])


# A drawn cylinder (face 0) and its two ends (faces 1 at z = 0, 2 at z = HEIGHT).


def _tube_surface(bottom=N, top=N, rows=5):
    """The cylinder's surface, triangulated: rings stacked up its wall and fans across its ends."""
    counts = np.round(np.linspace(bottom, top, rows + 1)).astype(int)
    points, rings = [], []
    for j, n in enumerate(counts):
        angle = 2 * np.pi * (np.arange(n) + 0.25 * j) / n
        rings.append(len(points) + np.arange(n))
        points += [[RADIUS * np.cos(t), RADIUS * np.sin(t), HEIGHT * j / rows] for t in angle]
    points += [[0.0, 0.0, 0.0], [0.0, 0.0, HEIGHT]]
    points = np.array(points)
    low, high = len(points) - 2, len(points) - 1
    triangles, face = [], []
    for j in range(rows):
        a, b = rings[j], rings[j + 1]
        a2, b2 = np.append(a, a[0]), np.append(b, b[0])
        i = k = 0
        while i < a.size or k < b.size:
            if k == b.size or (i < a.size and (i + 1) / a.size <= (k + 1) / b.size):
                triangles.append([a2[i], a2[i + 1], b2[k]])
                i += 1
            else:
                triangles.append([a2[i], b2[k + 1], b2[k]])
                k += 1
            face.append(0)
    first, last = rings[0], rings[-1]
    for i in range(first.size):
        triangles.append([low, first[(i + 1) % first.size], first[i]])
        face.append(1)
    for i in range(last.size):
        triangles.append([high, last[i], last[(i + 1) % last.size]])
        face.append(2)
    triangles = np.array(triangles)
    corner = points[triangles]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    centre = corner.mean(axis=1)
    out = np.where(
        np.array(face)[:, None] == 0, np.c_[centre[:, :2], np.zeros(len(centre))],
        np.c_[np.zeros((len(centre), 2)), centre[:, 2:] - HEIGHT / 2],
    )  # fmt: skip
    inward = np.einsum("ij,ij->i", normal, out) < 0
    triangles[inward] = triangles[inward][:, ::-1]
    return points, triangles, np.array(face)


def _tube_topology(wall=None) -> CadTopology:
    """The cylinder's topology: its wall, its two ends, and the two rims and the seam between."""
    wall = wall or CylinderFace(
        origin=np.zeros(3), axis=np.array([0.0, 0.0, 1.0]), radius=RADIUS,
        axial_range=(0.0, HEIGHT), angle=2 * np.pi, solid_inside=True,
    )  # fmt: skip
    ends = (
        PlaneFace(point=np.zeros(3), outward_normal=np.array([0.0, 0.0, -1.0])),
        PlaneFace(point=np.array([0.0, 0.0, HEIGHT]), outward_normal=np.array([0.0, 0.0, 1.0])),
    )
    vertices = np.array([[RADIUS, 0.0, 0.0], [RADIUS, 0.0, HEIGHT]])
    return CadTopology(
        faces=(wall, *ends),
        face_area=np.array([2 * np.pi * RADIUS * HEIGHT, np.pi * RADIUS**2, np.pi * RADIUS**2]),
        vertices=vertices,
        edge_vertices=np.array([[0, 0], [1, 1], [0, 1]]),
        edge_length=np.array([2 * np.pi * RADIUS, 2 * np.pi * RADIUS, HEIGHT]),
        edge_face_offsets=np.array([0, 2, 4, 6]),
        edge_face_indices=np.array([0, 1, 0, 2, 0, 0]),
        edge_angle=np.array([np.pi / 2, np.pi / 2, np.nan]),
        seam=np.array([False, False, True]),
    )


def test_a_cylinder_the_fluid_fills_with_nothing_inside_is_a_pipe():
    points, triangles, face = _tube_surface()
    found, refused = find_pipes(_tube_topology(), points, triangles, face, np.arange(3))
    assert found == [0]
    assert set(refused) == {1, 2}
    assert "bounded by 1 loops, not the two of a band" in refused[1]


def _bored(triangles, face):
    """The cylinder's wall wound into the cylinder: the fluid round it, as round a bore."""
    triangles = triangles.copy()
    triangles[face == 0] = triangles[face == 0][:, ::-1]
    return triangles


def test_a_cylinder_the_fluid_is_outside_is_not_a_pipe():
    points, triangles, face = _tube_surface()
    _, refused = find_pipes(_tube_topology(), points, _bored(triangles, face), face, np.array([0]))
    assert "the fluid is outside it" in refused[0]


def test_a_flat_ring_is_not_a_pipe():
    # A band between two concentric circles in one plane: two rims, but no tube between them.
    angle = 2 * np.pi * np.arange(N) / N
    inner = np.c_[np.cos(angle), np.sin(angle), np.zeros(N)]
    points = np.vstack([inner, 2.0 * inner])
    triangles = []
    for i in range(N):
        j = (i + 1) % N
        triangles += [[i, N + j, j], [i, N + i, N + j]]
    triangles = np.array(triangles)
    normal = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                      points[triangles[:, 2]] - points[triangles[:, 0]])  # fmt: skip
    triangles[normal[:, 2] < 0] = triangles[normal[:, 2] < 0][:, ::-1]
    face = np.zeros(len(triangles), dtype=np.int64)
    _, refused = find_pipes(_tube_topology(), points, triangles, face, np.array([0]))
    assert "holds no tube's volume" in refused[0]


def test_a_cylinder_whose_rims_differ_is_not_a_pipe():
    points, triangles, face = _tube_surface(bottom=12, top=16)
    _, refused = find_pipes(_tube_topology(), points, triangles, face, np.array([0]))
    assert "its rims have 12 and 16 points" in refused[0]


def test_a_cylinder_whose_rims_differ_is_a_pipe_when_its_rims_are_to_be_divided_again():
    # The drawing view asks before any rim is divided: a second triangulation sets the counts.
    points, triangles, face = _tube_surface(bottom=12, top=14)
    found, refused = find_pipes(
        _tube_topology(), points, triangles, face, np.array([0]), rim_counts=False
    )
    assert (found, refused) == ([0], {})


def test_a_cylinder_with_something_inside_is_not_a_pipe():
    # A lamp down the middle: a few triangles of another face inside the cylinder.
    points, triangles, face = _tube_surface()
    lamp = len(points) + np.arange(3)
    points = np.vstack([points, [[0.5, 0.0, 4.0], [0.0, 0.5, 5.0], [-0.5, 0.0, 6.0]]])
    triangles = np.vstack([triangles, lamp[None]])
    face = np.append(face, 3)
    topology = _tube_topology()
    _, refused = find_pipes(topology, points, triangles, face, np.array([0]))
    assert "something else lies inside it" in refused[0]


class _Tube:
    """The cylinder as a CAD model: its topology, and projections onto its wall, ends and rims."""

    def __init__(self) -> None:
        self._topology = _tube_topology()

    def topology(self, region):
        return self._topology

    def write_brep(self, region, path):
        return path

    def project(self, region, points, faces=None, edges=None):
        points = np.asarray(points, dtype=float)
        if edges is not None:
            edges = np.asarray(edges)
            z = np.where(edges == 0, 0.0, HEIGHT)
            radial = points[:, :2] / np.linalg.norm(points[:, :2], axis=1, keepdims=True)
            landed = np.c_[RADIUS * radial, z]
            face = np.where(edges == 0, 1, 2)
        else:
            if faces is None:
                to_wall = np.abs(np.linalg.norm(points[:, :2], axis=1) - RADIUS)
                distance = np.stack([to_wall, np.abs(points[:, 2]), np.abs(points[:, 2] - HEIGHT)])
                faces = np.argmin(distance, axis=0)
            face = np.asarray(faces)
            landed = points.copy()
            on_wall = face == 0
            radial = points[on_wall, :2] / np.linalg.norm(
                points[on_wall, :2], axis=1, keepdims=True
            )
            landed[on_wall, :2] = RADIUS * radial
            landed[face == 1, 2] = 0.0
            landed[face == 2, 2] = HEIGHT
        distance = np.linalg.norm(landed - points, axis=1)
        return Projection(landed, face, np.zeros_like(points), distance)


class _NoFill:
    """Netgen's triangulation of the cylinder; its fill must not be needed."""

    def triangulate(self, shape, **options):
        return SurfaceTriangulation(*_tube_surface(rows=8))

    def fill(self, points, triangles, **options):
        raise AssertionError("a pipe open at both ends leaves nothing for tetrahedra")


def test_a_pipe_open_at_both_ends_is_meshed_as_hexahedra_throughout_without_tetrahedra():
    layers = PrismLayers(first_height=0.05, growth=1.2, count=3)
    mesher = CadMesher(max_size=1.0, feature_angle=np.radians(45), layers=layers, netgen=_NoFill())
    patches = {"inlet": FacesNear(((0.0, 0.0, 0.0),)), "outlet": FacesNear(((0.0, 0.0, HEIGHT),))}
    mesh = mesher.mesh(_Tube(), CadRegion(("tube",)), patches, layered=("walls",))
    geometry = mesh.geometry()
    volume = np.asarray(geometry.cell.volume)
    assert np.all(volume > 0.0)
    scale = float(np.max(np.asarray(geometry.face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    # Its boundary vertices on the drawing, so its volume the cylinder's but for the flat faces
    # between them round the wall: twelve sides, within the polygon's area deficit.
    assert _polygon_prism_volume() < volume.sum() < np.pi * RADIUS**2 * HEIGHT
    # Every cell has six faces, every face between two cells is a quadrilateral, and every one on
    # the boundary too but at the rims, pentagons where the sharp edge's points join their faces --
    # but for the dual cells of the core's four corners in each cross-section: three primal cells
    # meet there, so they are triangular prisms, at the core's corners' distance from the axis.
    size = np.diff(np.asarray(mesh.face_nodes.offsets))
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    faces = np.bincount(owner, minlength=mesh.n_cells)
    faces += np.bincount(neighbour[neighbour >= 0], minlength=mesh.n_cells)
    prisms = faces == 5
    assert np.all(faces[~prisms] == 6)
    inside = neighbour >= 0
    in_prism = prisms[owner] | prisms[np.maximum(neighbour, 0)] & inside
    assert np.all(size[inside & ~in_prism] == 4)
    assert np.all(size[inside & in_prism & (size != 4)] == 3)
    outer = ~inside
    centroid = np.asarray(geometry.face.centroid)
    on_end = np.isclose(centroid[:, 2], 0.0) | np.isclose(centroid[:, 2], HEIGHT)
    assert np.all(size[outer & prisms[owner]] == 3)
    assert set(np.unique(size[outer & ~prisms[owner]]).tolist()) == {4, 5}
    # The pentagons are the rim points' faces, on the ends and the wall beside each rim.
    pentagons = outer & (size == 5)
    near_rim = np.minimum(centroid[:, 2], HEIGHT - centroid[:, 2]) < 1.0
    assert np.all(near_rim[pentagons])
    assert np.all(np.hypot(centroid[pentagons, 0], centroid[pentagons, 1]) > 0.9 * RADIUS)
    assert np.count_nonzero(pentagons & on_end) == np.count_nonzero(pentagons & ~on_end)
    cells = np.asarray(geometry.cell.centroid)
    from_axis = np.hypot(cells[prisms, 0], cells[prisms, 1])
    np.testing.assert_allclose(from_axis, from_axis.mean(), rtol=1e-9)
    sections = np.unique(np.round(cells[prisms, 2], 9)).size
    assert np.count_nonzero(prisms) == 4 * sections and sections > 2
    for name in ("inlet", "outlet"):
        area = float(
            np.asarray(geometry.face.area)[np.asarray(mesh.face_patches.indices(name))].sum()
        )
        assert _polygon_prism_volume() / HEIGHT < area <= np.pi * RADIUS**2 * (1 + 1e-12)


@pytest.mark.parametrize(
    ("bottom", "top", "count"), [(12, 13, 16), (17, 17, 20), (13, 12, 16), (11, 8, 12)]
)
def test_a_candidates_rims_are_both_asked_for_the_larger_count_rounded_up_to_four(
    bottom, top, count
):
    # Netgen's rim counts as on Sozzi (12 and 13 at maxh 0.005, 17 and 17 at 0.0035): an O-grid
    # needs one count on both rims, a multiple of four, and never coarser than Netgen chose.
    points, triangles, face = _tube_surface(bottom=bottom, top=top)
    divisions = rim_divisions(_tube_topology(), points, triangles, face, np.arange(3))
    assert divisions == {0: count, 1: count}


def test_rims_an_o_grid_already_joins_are_left_as_netgen_divided_them():
    points, triangles, face = _tube_surface(bottom=16, top=16)
    assert rim_divisions(_tube_topology(), points, triangles, face, np.arange(3)) == {}


def test_a_cylinder_the_fluid_is_outside_is_not_asked_for_rim_divisions():
    points, triangles, face = _tube_surface(bottom=12, top=13)
    bored = _bored(triangles, face)
    assert rim_divisions(_tube_topology(), points, bored, face, np.arange(3)) == {}


def test_a_rim_drawn_in_two_edges_shares_its_count_between_them_by_length():
    # As Sozzi's inlet rim: three quarters of the bottom circle is one edge, the last quarter
    # another, so sixteen segments go twelve and four.
    topology = _tube_topology()
    circle = 2 * np.pi * RADIUS
    topology = CadTopology(
        faces=topology.faces,
        face_area=topology.face_area,
        vertices=np.array([[RADIUS, 0.0, 0.0], [RADIUS, 0.0, HEIGHT], [0.0, -RADIUS, 0.0]]),
        edge_vertices=np.array([[0, 2], [2, 0], [1, 1], [0, 1]]),
        edge_length=np.array([0.75 * circle, 0.25 * circle, circle, HEIGHT]),
        edge_face_offsets=np.array([0, 2, 4, 6, 7]),
        edge_face_indices=np.array([0, 1, 0, 1, 0, 2, 0]),
        edge_angle=np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.nan]),
        seam=np.array([False, False, False, True]),
    )
    points, triangles, face = _tube_surface(bottom=12, top=13)
    assert rim_divisions(topology, points, triangles, face, np.arange(3)) == {0: 12, 1: 4, 2: 16}


class _RimsAsAsked(_NoFill):
    """Netgen's triangulation of the cylinder with rims of 12 and 13 points, or as many as asked."""

    def __init__(self) -> None:
        self.asked = []

    def triangulate(self, shape, edge_divisions=None, **options):
        self.asked.append(edge_divisions)
        if edge_divisions is None:
            return SurfaceTriangulation(*_tube_surface(bottom=12, top=13, rows=8))
        bottom, top = edge_divisions[0], edge_divisions[1]
        return SurfaceTriangulation(*_tube_surface(bottom=bottom, top=top, rows=8))


def test_a_pipe_whose_rims_netgen_divided_unequally_is_triangulated_again_and_cored():
    # Without the second triangulation the cylinder would be refused as a pipe and its core left
    # to tetrahedra, which this Netgen refuses to make.
    netgen = _RimsAsAsked()
    mesher = CadMesher(max_size=1.0, feature_angle=np.radians(45), netgen=netgen)
    patches = {"inlet": FacesNear(((0.0, 0.0, 0.0),)), "outlet": FacesNear(((0.0, 0.0, HEIGHT),))}
    mesh = mesher.mesh(_Tube(), CadRegion(("tube",)), patches)
    assert netgen.asked == [None, {0: 16, 1: 16}]
    scale = float(np.max(np.asarray(mesh.geometry().face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale


# A bend: a quarter torus (face 0) of tube radius RADIUS about the z axis through (0, BEND, 0), from
# the plane x = 0 to the plane y = BEND, and a straight pipe (face 1) on along +y from its far rim.
BEND = 3.0 * RADIUS


def _section(angle, n):
    """The bend's cross-section at ``angle`` (``n`` points), or past its end for ``angle > pi/2``:
    the straight pipe's, ``angle - pi/2`` bend radii along it."""
    turn = min(angle, 0.5 * np.pi)
    along = np.array([np.sin(turn), -np.cos(turn), 0.0])
    centre = np.array([0.0, BEND, 0.0]) + BEND * along
    centre = centre + max(0.0, angle - 0.5 * np.pi) * BEND * np.array([0.0, 1.0, 0.0])
    t = 2 * np.pi * np.arange(n) / n
    return centre + RADIUS * (np.cos(t)[:, None] * along + np.sin(t)[:, None] * [0.0, 0.0, 1.0])


def _bend_surface(counts):
    """The bend and the straight pipe after it, triangulated as bands between rings whose counts
    are ``counts`` (from the bend's first rim, two rows per face)."""
    angles = np.linspace(0.0, np.pi, len(counts))
    points, rings = [], []
    for angle, n in zip(angles, counts, strict=True):
        rings.append(len(points) + np.arange(n))
        points += list(_section(angle, n))
    points = np.array(points)
    triangles, face = [], []
    for j in range(len(counts) - 1):
        a, b = rings[j], rings[j + 1]
        a2, b2 = np.append(a, a[0]), np.append(b, b[0])
        i = k = 0
        while i < a.size or k < b.size:
            if k == b.size or (i < a.size and (i + 1) / a.size <= (k + 1) / b.size):
                triangles.append([a2[i], a2[i + 1], b2[k]])
                i += 1
            else:
                triangles.append([a2[i], b2[k + 1], b2[k]])
                k += 1
            face.append(0 if angles[j] < 0.5 * np.pi - 1e-9 else 1)
    triangles = np.array(triangles)
    # Wound out of the fluid: each triangle's normal away from the centreline beside it.
    corner = points[triangles]
    centroid = corner.mean(axis=1)
    offset = centroid - np.array([0.0, BEND, 0.0])
    turn = np.clip(np.arctan2(offset[:, 0], -offset[:, 1]), 0.0, 0.5 * np.pi)
    beside = np.array([0.0, BEND, 0.0]) + BEND * np.c_[np.sin(turn), -np.cos(turn), 0 * turn]
    past = centroid[:, 1] > BEND
    beside[past] = np.c_[np.full(past.sum(), BEND), centroid[past, 1], np.zeros(past.sum())]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    inward = np.einsum("ij,ij->i", normal, centroid - beside) < 0
    triangles[inward] = triangles[inward][:, ::-1]
    return points, triangles, np.array(face)


def _bend_topology() -> CadTopology:
    """The bend (face 0) and the straight pipe (face 1): rims 0, 1 (shared), 2, and two seams."""
    torus = TorusFace(
        centre=np.array([0.0, BEND, 0.0]), axis=np.array([0.0, 0.0, 1.0]), major_radius=BEND,
        minor_radius=RADIUS, angle=0.5 * np.pi, solid_inside=True,
    )  # fmt: skip
    pipe = CylinderFace(
        origin=np.array([BEND, BEND, 0.0]), axis=np.array([0.0, 1.0, 0.0]), radius=RADIUS,
        axial_range=(0.0, 0.5 * np.pi * BEND), angle=2 * np.pi, solid_inside=True,
    )  # fmt: skip
    vertices = np.array([[RADIUS, 0.0, 0.0], [BEND, BEND - RADIUS, 0.0], [BEND, 2 * BEND, 0.0]])
    circle = 2 * np.pi * RADIUS
    return CadTopology(
        faces=(torus, pipe),
        face_area=np.array([circle * BEND * 0.5 * np.pi, circle * BEND * 0.5 * np.pi]),
        vertices=vertices,
        edge_vertices=np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 2]]),
        edge_length=np.full(5, circle),
        edge_face_offsets=np.array([0, 1, 3, 4, 6, 8]),
        edge_face_indices=np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        edge_angle=np.array([np.nan, 0.0, np.nan, np.nan, np.nan]),
        seam=np.array([False, False, False, True, True]),
    )


def _lamp_at(points, triangles, face, centre):
    """The surface with a small triangle of another face (2) about ``centre``."""
    lamp = centre + 0.2 * RADIUS * np.eye(3)
    return (
        np.vstack([points, lamp]),
        np.vstack([triangles, len(points) + np.arange(3)[None]]),
        np.append(face, 2),
    )


def _on_centreline(angle):
    return np.array([0.0, BEND, 0.0]) + BEND * np.array([np.sin(angle), -np.cos(angle), 0.0])


def test_a_torus_the_fluid_fills_is_a_pipe_and_one_with_something_in_its_tube_is_not():
    points, triangles, face = _bend_surface([N] * 5)
    found, refused = find_pipes(_bend_topology(), points, triangles, face, np.array([0, 1]))
    assert (found, refused) == ([0, 1], {})
    # A triangle on the bend's centreline half way round it is something inside the bend.
    lamp = _lamp_at(points, triangles, face, _on_centreline(0.25 * np.pi))
    _, refused = find_pipes(_bend_topology(), *lamp, np.array([0]))
    assert "something else lies inside it" in refused[0]


def test_what_lies_on_a_bends_circle_beyond_its_ends_is_not_inside_it():
    # On the torus's centreline circle a half turn round from the bend's start -- inside the whole
    # torus, but not the bend's quarter of it -- a triangle leaves the bend a pipe.
    points, triangles, face = _bend_surface([N] * 5)
    lamp = _lamp_at(points, triangles, face, _on_centreline(np.pi))
    found, _ = find_pipes(_bend_topology(), *lamp, np.array([0]))
    assert found == [0]


def test_pipes_joined_at_a_rim_are_divided_alike_all_along():
    # The bend's rims are 12 and 12 points, the straight pipe's 12 and 14: alone the bend would
    # ask nothing, but the shared rim must take the straight pipe's 16, so the bend's far rim too,
    # or the bend's two rims would differ.
    points, triangles, face = _bend_surface([12, 12, 12, 13, 14])
    divisions = rim_divisions(_bend_topology(), points, triangles, face, np.array([0, 1]))
    assert divisions == {0: 16, 1: 16, 2: 16}


def _grid(rows, n, start=0):
    return start + np.arange(rows * n).reshape(rows, n)


def _grid_edges(grid):
    """A grid's edges round its rows and along its columns, each as a sorted pair."""
    closed = np.concatenate([grid, grid[:, :1]], axis=1)
    around = np.stack([closed[:, :-1], closed[:, 1:]], axis=-1).reshape(-1, 2)
    along = np.stack([grid[:-1], grid[1:]], axis=-1).reshape(-1, 2)
    return {tuple(sorted(e)) for e in around.tolist()}, {tuple(sorted(e)) for e in along.tolist()}


@pytest.mark.parametrize("reverse_rows", [False, True])
@pytest.mark.parametrize("reverse_columns", [False, True])
@pytest.mark.parametrize("roll", [0, 5])
def test_pipes_sharing_a_rim_are_one_run_whose_grid_joins_them_point_for_point(
    reverse_rows, reverse_columns, roll
):
    # The second pipe's grid starts at the first's last rim, however its rows and columns run:
    # the run's grid holds every point once, row after row, and every edge round its rows and
    # along its columns is one of the two pipes' own.
    from aquaflux.meshing.pipes import pipe_runs

    first = _grid(4, 8)
    second = np.vstack([first[-1], _grid(3, 8, start=100)])
    if reverse_columns:
        second = second[:, ::-1]
    second = np.roll(second, roll, axis=1)
    if reverse_rows:
        second = second[::-1]
    ((faces, grid),) = pipe_runs({3: first, 7: second})
    assert grid.shape == (7, 8)
    assert sorted(faces) == [3, 7]
    np.testing.assert_array_equal(
        np.sort(grid, axis=None), np.r_[np.arange(32), 100 + np.arange(24)]
    )
    around, along = _grid_edges(grid)
    for pipe in (first, second):
        own_around, own_along = _grid_edges(pipe)
        assert own_around <= around
        assert own_along <= along
    assert len(along) == 6 * 8


def test_three_pipes_at_one_rim_or_a_loop_of_pipes_have_no_run():
    from aquaflux.meshing.pipes import pipe_runs

    first = _grid(3, 4)
    branch = np.vstack([first[-1], _grid(2, 4, start=50)])
    other = np.vstack([first[-1], _grid(2, 4, start=80)])
    with pytest.raises(ValueError, match="all meet at one rim"):
        pipe_runs({0: first, 1: branch, 2: other})
    back = np.vstack([first[-1], _grid(1, 4, start=50), first[0]])
    with pytest.raises(ValueError, match="in a loop"):
        pipe_runs({0: first, 1: back})
