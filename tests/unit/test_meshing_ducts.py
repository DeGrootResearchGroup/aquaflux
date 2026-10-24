"""Rectangular ducts: found among a region's faces, divided for a structured core, and cored.

The duct here is a box along ``z`` drawn as a chain of segments -- four sides each, cut across at
heights in between -- closed by two end faces, so its triangulation and its topology are written
down without a CAD kernel. The cross-sections' core is checked to fill four-sided rings exactly; the
search for ducts on that drawing and on drawings with one fault each; the rails' points against
cross-sections normal to a curved duct's centreline; and the whole mesher on the box open at both
ends, which leaves nothing for tetrahedra.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.io.cad import CadRegion, CadTopology, PlaneFace, Projection
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import (
    CadMesher,
    DuctCore,
    EdgePoints,
    FacesNear,
    PrismLayers,
    SurfaceTriangulation,
    element_mesh,
    find_ducts,
)
from aquaflux.meshing.ducts import _stations, duct_divisions
from scipy.spatial import Delaunay

W, H, L = 2.0, 1.0, 6.0
FEATURE = math.radians(45)

# ---------------------------------------------------------------------------------------------
# The core: cross-sections of four-sided rings
# ---------------------------------------------------------------------------------------------


def _ring(width, height, w, h, z, bend=None):
    """A ``w x h`` rectangle's ring of points at height ``z`` (or turned about the ``y`` axis by
    ``bend`` radians, its ``x`` then a radius), from its corner at the origin round ``x`` first."""
    sides = [
        [(width * i / w, 0.0) for i in range(w)],
        [(width, height * j / h) for j in range(h)],
        [(width * i / w, height) for i in range(w, 0, -1)],
        [(0.0, height * j / h) for j in range(h, 0, -1)],
    ]
    flat = np.array([p for side in sides for p in side])
    if bend is None:
        return np.c_[flat, np.full(len(flat), z)]
    radius = 3.0 + flat[:, 0]
    return np.c_[radius * np.cos(bend), flat[:, 1], radius * np.sin(bend)]


def _rings(w=3, h=2, rows=4, bent=False):
    points, rings = [], []
    for k in range(rows + 1):
        ring = _ring(W, H, w, h, L * k / rows, bend=(0.5 * math.pi * k / rows) if bent else None)
        rings.append(len(points) + np.arange(len(ring)))
        points += ring.tolist()
    return np.array(points), np.array(rings)


@pytest.mark.parametrize(("w", "h"), [(3, 2), (1, 1), (4, 1)])
def test_the_core_fills_the_rings_exactly_and_meets_their_quadrilaterals(w, h):
    points, rings = _rings(w, h)
    filled = DuctCore().fill(points, rings, width=w)
    assert filled.hexahedra.shape[0] == w * h * (rings.shape[0] - 1)
    mesh = element_mesh(filled.points, hexahedra=filled.hexahedra)
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    assert volume.sum() == pytest.approx(W * H * L, rel=1e-12)
    # Each hexahedron is one cell of the box's grid: its eight corners those of a grid cell.
    corners = filled.points[filled.hexahedra]
    low, high = corners.min(axis=1), corners.max(axis=1)
    np.testing.assert_allclose(
        high - low, np.tile([W / w, H / h, L / 4], (len(low), 1)), atol=1e-12
    )
    on_box = np.isclose(corners, low[:, None]) | np.isclose(corners, high[:, None])
    assert np.all(on_box)
    assert all(len({tuple(c) for c in cell.round(12).tolist()}) == 8 for cell in corners)
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12
    # Its faces round it are the rings' quadrilaterals, and its ends its cross-sections.
    offsets = np.asarray(mesh.face_nodes.offsets)
    nodes = np.asarray(mesh.face_nodes.face_node_indices)
    boundary = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    found = {tuple(sorted(nodes[offsets[f] : offsets[f + 1]])) for f in boundary}
    closed = np.concatenate([rings, rings[:, :1]], axis=1)
    sides = {
        tuple(sorted((closed[j, i], closed[j, i + 1], closed[j + 1, i + 1], closed[j + 1, i])))
        for j in range(rings.shape[0] - 1)
        for i in range(rings.shape[1])
    }
    ends = {tuple(sorted(q)) for q in np.concatenate([filled.first, filled.last]).tolist()}
    assert found == sides | ends


def test_a_bent_ducts_core_follows_its_rings_round_the_bend():
    # Each ring a rectangle in a plane through the bend's axis: the core's points of a ring lie in
    # its plane, and its volume is the polygonal bend's, between the inscribed and the true one.
    points, rings = _rings(bent=True, rows=6)
    filled = DuctCore().fill(points, rings, width=3)
    mesh = element_mesh(filled.points, hexahedra=filled.hexahedra)
    volume = np.asarray(mesh.geometry().cell.volume)
    assert np.all(volume > 0.0)
    true = 0.5 * math.pi * H * ((3.0 + W) ** 2 - 3.0**2) / 2
    assert true * math.cos(math.pi / 24) ** 2 < volume.sum() < true
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12
    angle = np.arctan2(filled.points[:, 2], filled.points[:, 0])
    np.testing.assert_allclose(
        np.unique(np.round(angle[points.shape[0] :], 12)),
        0.5 * math.pi * np.arange(7) / 6,
        atol=1e-12,
    )


def test_rings_that_are_not_four_sided_are_refused():
    points, rings = _rings(3, 2)
    with pytest.raises(ValueError, match="at least two rings of four sides"):
        DuctCore().fill(points, rings, width=5)  # leaves the second side no segment
    with pytest.raises(ValueError, match="at least two rings of four sides"):
        DuctCore().fill(points, rings[:1], width=3)


@pytest.mark.parametrize("rows", [0, -1, 1.5, True])
def test_junction_rows_must_be_a_whole_number_of_at_least_one(rows):
    with pytest.raises(ValueError, match="junction_rows"):
        DuctCore(junction_rows=rows)


# ---------------------------------------------------------------------------------------------
# A box duct drawn as a chain of segments
# ---------------------------------------------------------------------------------------------


class _BoxDuct:
    """The box ``[0, W] x [0, H] x [0, L]`` as a duct of ``segments`` along ``z``: its topology,
    a face-tagged triangulation wound outward, and projections onto its faces and edges.

    Faces: segment ``k``'s sides ``4 k + i`` (``i``: ``y = 0``, ``x = W``, ``y = H``, ``x = 0``),
    then the ends ``z = 0`` and ``z = L``. Edges: each level's four rim edges (``4 l + i``, along
    side ``i``), then each segment's four rails (between sides ``i - 1`` and ``i``, at corner ``i``).
    """

    def __init__(self, segments=2, rail_fold=math.pi / 2) -> None:
        self.n = segments
        self.levels = L * np.arange(segments + 1) / segments
        corners = np.array([[0.0, 0.0], [W, 0.0], [W, H], [0.0, H]])
        self.vertices = np.array([[*corners[i], z] for z in self.levels for i in range(4)])
        rims = [(4 * lv + i, 4 * lv + (i + 1) % 4) for lv in range(segments + 1) for i in range(4)]
        rails = [(4 * k + i, 4 * (k + 1) + i) for k in range(segments) for i in range(4)]
        self.edges = np.array(rims + rails)
        self.n_sides = 4 * segments
        incident, angle = [], []
        for lv in range(segments + 1):
            for i in range(4):
                below = 4 * (lv - 1) + i if lv > 0 else self.n_sides
                above = 4 * lv + i if lv < segments else self.n_sides + 1
                incident.append([below, above])
                angle.append(0.0 if 0 < lv < segments else math.pi / 2)
        for k in range(segments):
            for i in range(4):
                incident.append([4 * k + (i - 1) % 4, 4 * k + i])
                angle.append(rail_fold)
        records = []
        for f in range(self.n_sides):
            axis = 1 if f % 4 in (0, 2) else 0
            value = {0: 0.0, 1: W, 2: H, 3: 0.0}[f % 4]
            sign = 1.0 if f % 4 in (1, 2) else -1.0
            normal = np.zeros(3)
            normal[axis] = sign
            point = np.zeros(3)
            point[axis] = value
            records.append(PlaneFace(point=point, outward_normal=normal))
        records.append(PlaneFace(point=np.zeros(3), outward_normal=np.array([0.0, 0.0, -1.0])))
        records.append(PlaneFace(point=np.array([0, 0, L]), outward_normal=np.array([0, 0, 1.0])))
        self.records = tuple(records)
        length = np.linalg.norm(
            self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]], axis=1
        )
        self._topology = CadTopology(
            faces=self.records,
            face_area=np.ones(len(records)),
            vertices=self.vertices,
            edge_vertices=self.edges,
            edge_length=length,
            edge_face_offsets=np.arange(len(incident) + 1) * 2,
            edge_face_indices=np.array(incident).reshape(-1),
            edge_angle=np.array(angle),
            seam=np.zeros(len(incident), dtype=bool),
        )

    # -- the CAD model's side --------------------------------------------------------------------

    def topology(self, region):
        return self._topology

    def write_brep(self, region, path):
        return path

    def project(self, region, points, faces=None, edges=None):
        points = np.asarray(points, dtype=float)
        if edges is not None:
            ends = self.vertices[self.edges[np.asarray(edges)]]
            along = ends[:, 1] - ends[:, 0]
            t = np.clip(
                np.sum((points - ends[:, 0]) * along, axis=1) / np.sum(along**2, axis=1), 0, 1
            )
            landed = ends[:, 0] + t[:, None] * along
            face = self._topology.edge_face_indices[2 * np.asarray(edges)]
        else:
            face = self._nearest(points) if faces is None else np.asarray(faces)
            landed = points.copy()
            for f in np.unique(face):
                record = self.records[f]
                axis = int(np.argmax(np.abs(record.outward_normal)))
                landed[face == f, axis] = record.point[axis]
        return Projection(
            landed, face, np.zeros_like(points), np.linalg.norm(landed - points, axis=1)
        )

    def _nearest(self, points):
        distance = np.stack([
            np.abs((points - r.point) @ r.outward_normal)
            + (0.0 if f >= self.n_sides else 10.0 * self._outside(points, f))
            for f, r in enumerate(self.records)
        ])  # fmt: skip
        return np.argmin(distance, axis=0)

    def _outside(self, points, f):
        k = f // 4
        return (points[:, 2] < self.levels[k] - 1e-9) | (points[:, 2] > self.levels[k + 1] + 1e-9)

    # -- Netgen's side ---------------------------------------------------------------------------

    def triangulate(self, rails=(4, 5, 4, 5), across=(3, 2), divisions=None, extra=None):
        """The surface, each rail of every segment divided into ``rails[i]`` segments (corner
        ``i``) and each side's rim edges into ``across[i % 2]``, unless ``divisions`` says
        otherwise, as Netgen would; ``extra`` adds triangles of another face (points, a face)."""
        divisions = divisions or {}
        topology = self._topology

        def edge_points(e):
            a, b = self.vertices[self.edges[e]]
            division = divisions.get(e)
            if isinstance(division, EdgePoints):
                t = np.array(division.fractions)
                if np.linalg.norm(np.array(division.start) - b) < np.linalg.norm(
                    np.array(division.start) - a
                ):
                    t = 1.0 - t[::-1]
            else:
                rim = e < 4 * (self.n + 1)
                count = division or (
                    across[(e % 4) % 2] if rim else rails[(e - 4 * (self.n + 1)) % 4]
                )
                t = np.linspace(0, 1, count + 1)[1:-1]
            return np.vstack([a, a + t[:, None] * (b - a), b])

        points, triangles, face = [], [], []
        for f, record in enumerate(self.records):
            mine = np.flatnonzero(np.any(topology.edge_face_indices.reshape(-1, 2) == f, axis=1))
            boundary = np.vstack([edge_points(e) for e in mine])
            axis = int(np.argmax(np.abs(record.outward_normal)))
            plane = [a for a in range(3) if a != axis]
            low, high = boundary.min(axis=0), boundary.max(axis=0)
            u, v = np.meshgrid(np.linspace(0, 1, 6)[1:-1], np.linspace(0, 1, 6)[1:-1])
            inner = np.repeat(low[None], u.size, axis=0)
            inner[:, plane[0]] += u.reshape(-1) * (high - low)[plane[0]]
            inner[:, plane[1]] += v.reshape(-1) * (high - low)[plane[1]]
            mine_points = np.unique(np.round(np.vstack([boundary, inner]), 12), axis=0)
            # Triangulated with its boundary bowed out a little, so every point on a straight edge
            # is a corner of the hull and no triangle is a sliver along it.
            flat = mine_points[:, plane]
            middle = 0.5 * (flat.min(axis=0) + flat.max(axis=0))
            half = 0.5 * (flat.max(axis=0) - flat.min(axis=0))
            local = (flat - middle) / half
            bow = 1.0 + 0.05 * (1.0 - local[:, ::-1] ** 2)
            simplices = Delaunay(local * bow).simplices
            corner = mine_points[simplices]
            normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
            flip = normal @ record.outward_normal < 0
            simplices[flip] = simplices[flip][:, ::-1]
            triangles.append(len(points) + simplices)
            points += mine_points.tolist()
            face.append(np.full(len(simplices), f))
        if extra is not None:
            extra_points, extra_face = extra
            triangles.append(len(points) + np.array([[0, 1, 2]]))
            points += extra_points
            face.append(np.array([extra_face]))
        points = np.array(points)
        unique, inverse = np.unique(np.round(points, 10), axis=0, return_inverse=True)
        return SurfaceTriangulation(
            unique, inverse.reshape(-1)[np.concatenate(triangles)], np.concatenate(face)
        )


def _found(box, surface, candidates=None, **options):
    topology = box.topology(None)
    if candidates is None:
        candidates = np.arange(topology.n_faces)
    return find_ducts(
        topology, surface.points, surface.triangles, surface.face, candidates, FEATURE, **options
    )


def test_the_box_is_one_duct_of_chained_segments_along_its_length():
    box = _BoxDuct(segments=3)
    ducts, refused = _found(box, box.triangulate(rails=(4, 4, 4, 4)))
    assert len(ducts) == 1
    duct = ducts[0]
    assert duct.sides.shape == (3, 4)
    # Side i of each segment continues side i of the one before, round the duct in one order.
    first = duct.sides[0, 0] % 4
    step = (duct.sides[0, 1] - duct.sides[0, 0]) % 4
    for k in range(3):
        np.testing.assert_array_equal(duct.sides[k] // 4, k)
        np.testing.assert_array_equal(duct.sides[k] % 4, (first + step * np.arange(4)) % 4)
    # The rims are the levels' edges, in side order, from one end to the other.
    levels = duct.rims // 4
    assert (levels[:, 0] == np.arange(4)).all() or (levels[:, 0] == np.arange(3, -1, -1)).all()
    assert all(len(set(row.tolist())) == 1 for row in levels)
    # The ends are no duct's sides: across the box, the sides of the duct's segments meet each
    # other smoothly, so no four faces close round it that way.
    assert set(refused) == {12, 13}
    assert "do not make it one of four faces round a duct" in refused[12]


def test_a_ducts_edges_divided_unalike_are_refused_until_divided_again():
    box = _BoxDuct(segments=2)
    surface = box.triangulate(rails=(4, 5, 4, 5))
    ducts, refused = _found(box, surface)
    assert ducts == [] and "not divided alike" in refused[0]
    ducts, _ = _found(box, surface, edge_counts=False)
    divisions = duct_divisions(
        box.topology(None), surface.points, surface.triangles, surface.face, ducts
    )
    rails = np.arange(4 * 3, 4 * 3 + 8)
    assert all(isinstance(divisions[e], EdgePoints) for e in rails)
    # The rails get the most segments any was given, at equal steps: the box's cross-sections.
    for e in rails:
        np.testing.assert_allclose(divisions[e].fractions, np.arange(1, 5) / 5, atol=1e-9)
    rims = np.arange(12)
    assert {divisions[e] for e in rims if e % 2 == 0} == {3}
    assert {divisions[e] for e in rims if e % 2 == 1} == {2}
    again = box.triangulate(divisions=divisions)
    ducts, refused = _found(box, again)
    assert len(ducts) == 1 and set(refused) == {8, 9}


def test_a_duct_with_something_inside_is_not_a_duct():
    box = _BoxDuct(segments=1)
    inside = [[0.5, 0.4, 2.0], [1.0, 0.6, 3.0], [1.5, 0.4, 4.0]]
    ducts, refused = _found(box, box.triangulate(rails=(4,) * 4, extra=(inside, 6)))
    assert ducts == []
    assert refused[0] == "something else lies inside its duct"


def test_a_duct_with_something_beyond_its_end_is_still_a_duct():
    box = _BoxDuct(segments=1)
    beyond = [[0.5, 0.4, L + 1.0], [1.0, 0.6, L + 2.0], [1.5, 0.4, L + 1.5]]
    ducts, _ = _found(box, box.triangulate(rails=(4,) * 4, extra=(beyond, 6)))
    assert len(ducts) == 1


def test_four_faces_whose_corner_edges_are_not_sharp_are_no_duct():
    box = _BoxDuct(segments=1, rail_fold=math.radians(30))
    ducts, refused = _found(box, box.triangulate(rails=(4,) * 4), candidates=np.arange(4))
    assert ducts == []
    assert "do not make it one of four faces round a duct" in refused[0]


def test_only_candidates_are_sides():
    # Three of the four sides named: no ring of four candidates closes round the duct.
    box = _BoxDuct(segments=1)
    ducts, refused = _found(box, box.triangulate(rails=(4,) * 4), candidates=np.arange(3))
    assert ducts == [] and set(refused) == {0, 1, 2}


def test_a_face_of_other_than_four_edges_is_not_a_side():
    box = _BoxDuct(segments=1)
    topology = box.topology(None)
    # Drop one of face 0's edges from the topology's count by marking it a seam.
    seam = topology.seam.copy()
    seam[0] = True
    import dataclasses

    box._topology = dataclasses.replace(topology, seam=seam)
    _, refused = _found(box, box.triangulate(rails=(4,) * 4), candidates=np.array([0]))
    assert refused[0] == "it is bounded by 3 edges, not the four of a duct's side"


# ---------------------------------------------------------------------------------------------
# Where the rails' points go: cross-sections normal to a curved duct
# ---------------------------------------------------------------------------------------------


def _parabolic_rails(rows=400):
    """A square duct's four rails along the parabola ``x = z**2 / 4``, ``z`` in ``[0, 4]``: each
    corner offset from it across the duct, so the rails differ in length unevenly along it."""
    z = np.linspace(0.0, 4.0, rows)
    centre = np.c_[z**2 / 4, np.zeros_like(z), z]
    tangent = np.gradient(centre, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    across = np.cross(tangent, [0.0, 1.0, 0.0])
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0])
    half = 0.4
    return [centre + half * (a * across + b * up) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _tilt(rails, fractions):
    """The worst angle, in degrees, between each interior cross-section's plane through the four
    rails' points and the centreline there."""
    from aquaflux.meshing.mapped import _resampled

    at = [
        _resampled(r, np.concatenate([[0], f, [1]]), closed=False)
        for r, f in zip(rails, fractions, strict=True)
    ]
    sections = np.stack(at, axis=1)  # (rows, 4, 3)
    centre = sections.mean(axis=1)
    worst = 0.0
    for k in range(1, len(centre) - 1):
        normal = np.linalg.svd(sections[k] - centre[k])[2][-1]
        along = centre[k + 1] - centre[k - 1]
        along /= np.linalg.norm(along)
        worst = max(worst, math.degrees(math.acos(min(1.0, abs(normal @ along)))))
    return worst


def test_the_rails_points_make_cross_sections_normal_to_a_curved_ducts_centreline():
    rails = _parabolic_rails()
    segments = 16
    equal = [np.arange(1, segments) / segments] * 4
    placed = _stations(rails, segments)
    for fractions in placed:
        assert np.all(np.diff(np.concatenate([[0], fractions, [1]])) > 0)
    # At equal arc length the inner and outer rails' points drift apart along the bend.
    assert _tilt(rails, equal) > 5.0
    assert _tilt(rails, placed) < 0.5


def test_cross_sections_lean_toward_a_rim_cut_obliquely_and_are_normal_away_from_it():
    # A straight square duct along z whose first rim is cut at 30 degrees about y.
    z = np.linspace(0.0, 6.0, 200)
    lean = math.tan(math.radians(30))
    rails = []
    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        start = a * 0.5 * lean
        rails.append(
            np.c_[
                np.full_like(z, 0.5 * a), np.full_like(z, 0.5 * b), start + (6.0 - start) * z / 6.0
            ]
        )
    placed = _stations(rails, 12)
    points = np.stack([
        np.interp(np.concatenate([[0], f, [1]]), np.linspace(0, 1, 200), r[:, 2])
        for r, f in zip(rails, placed, strict=True)
    ])  # fmt: skip
    # The first interior cross-section, half a width from the rim, leans half as far as it does;
    # from a width on, each is square across the duct.
    first = points[:, 1]
    assert first[1] - first[0] == pytest.approx(0.5 * lean, rel=0.05)
    np.testing.assert_allclose(points[:, 4:-1] - points[:, 4:-1].mean(axis=0), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------------------------
# The whole mesher on the box, open at both ends
# ---------------------------------------------------------------------------------------------


class _Netgen:
    """The box's triangulation, its rails first divided unalike, then as asked; its fill must
    not be needed."""

    def __init__(self, box) -> None:
        self.box = box
        self.asked = []

    def triangulate(self, shape, edge_divisions=None, **options):
        self.asked.append(edge_divisions)
        return self.box.triangulate(divisions=edge_divisions)

    def fill(self, points, triangles, **options):
        raise AssertionError("a duct open at both ends leaves nothing for tetrahedra")


@pytest.mark.parametrize("layered", [False, True])
def test_a_duct_open_at_both_ends_is_meshed_as_hexahedra_without_tetrahedra(layered):
    box = _BoxDuct(segments=2)
    netgen = _Netgen(box)
    layers = PrismLayers(first_height=0.02, growth=1.2, count=3) if layered else None
    mesher = CadMesher(max_size=1.0, feature_angle=FEATURE, layers=layers, netgen=netgen)
    patches = {"inlet": FacesNear(((1.0, 0.5, 0.0),)), "outlet": FacesNear(((1.0, 0.5, L),))}
    mesh = mesher.mesh(box, CadRegion(("duct",)), patches, layered=("walls",) if layered else ())
    # The rails were divided unalike at first, so the box was triangulated again.
    assert len(netgen.asked) == 2 and netgen.asked[0] is None
    geometry = mesh.geometry()
    volume = np.asarray(geometry.cell.volume)
    assert np.all(volume > 0.0)
    assert volume.sum() == pytest.approx(W * H * L, rel=1e-12)
    scale = float(np.max(np.asarray(geometry.face.area)))
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-12 * scale
    for name in ("inlet", "outlet"):
        faces = np.asarray(mesh.face_patches.indices(name))
        assert float(np.asarray(geometry.face.area)[faces].sum()) == pytest.approx(W * H, rel=1e-12)
    owner = np.asarray(mesh.face_cells.owner)
    neighbour = np.asarray(mesh.face_cells.neighbour)
    faces = np.bincount(owner, minlength=mesh.n_cells)
    faces += np.bincount(neighbour[neighbour >= 0], minlength=mesh.n_cells)
    if not layered:
        assert np.all(faces == 6)  # the dual of a block of hexahedra is one
        return
    # With layers, every cell has a hexahedron's six faces but at the layers' top's four corners,
    # where three primal cells meet along the duct: there each row of cells has a triangular prism
    # (five faces) and beside it a cell of seven, in the corner of the core's first cell.
    centroid = np.asarray(geometry.cell.centroid)
    rows = np.unique(np.round(centroid[:, 2], 9)).size
    assert np.count_nonzero(faces == 5) == np.count_nonzero(faces == 7) == 4 * rows
    assert np.all((faces >= 5) & (faces <= 7))
    odd = faces != 6
    thickness = 0.02 * sum(1.2**k for k in range(3))
    from_side = np.minimum(centroid[:, 0], W - centroid[:, 0])
    from_floor = np.minimum(centroid[:, 1], H - centroid[:, 1])
    assert np.all(from_side[odd] < thickness + 0.5 * W / 3)  # within its first core cell
    assert np.all(from_floor[odd] < thickness + 0.5 * H / 2)


@pytest.mark.parametrize("seed", range(6))
def test_a_ducts_rings_are_read_round_it_whichever_way_each_sides_grid_runs(seed):
    # A duct of two segments of four rows, its sides 3, 2, 3 and 2 points across: point ``r n + p``
    # at row ``r`` and position ``p`` round it, side 0 starting at position 1. Each side's grid is
    # handed over turned and flipped at random, as mapped meshing may give it.
    from aquaflux.meshing.ducts import Duct, duct_rings

    rng = np.random.default_rng(seed)
    counts, n, rows = (3, 2, 3, 2), 10, 8
    ids = np.arange((rows + 1) * n).reshape(rows + 1, n)
    starts = 1 + np.concatenate([[0], np.cumsum(counts)[:-1]])
    grids = {}
    for k in range(2):
        for i in range(4):
            grid = ids[4 * k : 4 * k + 5][:, (starts[i] + np.arange(counts[i] + 1)) % n]
            if rng.random() < 0.5:
                grid = grid.T
            grid = grid[:: rng.choice([-1, 1]), :: rng.choice([-1, 1])]
            grids[4 * k + i] = grid
    duct = Duct(
        sides=np.arange(8).reshape(2, 4),
        rails=np.zeros((2, 4), dtype=np.int64),
        rims=np.zeros((3, 4), dtype=np.int64),
    )
    rings, width = duct_rings(duct, grids)
    expected = ids[:, (starts[0] + np.arange(n)) % n]
    # Round it from side 0 to side 1 and on, from side 0's first corner, row 0 first.
    assert width == 3
    np.testing.assert_array_equal(rings, expected)


@pytest.mark.parametrize(
    ("fractions", "start"),
    [((0.5, 0.4), (0, 0, 0)), ((0.0, 0.5), (0, 0, 0)), ((0.5, 1.0), (0, 0, 0)), ((0.5,), (0, 0))],
)
def test_an_edges_points_must_increase_within_it_from_a_point(fractions, start):
    with pytest.raises(ValueError, match="an edge's points"):
        EdgePoints(fractions, start)
