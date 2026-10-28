"""Structured quadrilaterals in place of a face's triangles, checked against the face they cover.

The faces are a band (an open cylinder, its triangles stacked in rings whose counts change from one
rim to the other, so the rims can be given different counts) and a unit square triangulated from
graded points on its sides. A grid must keep every point of the face's boundary and nothing else of
the triangulation, lie on the face, cover its area, be wound as the triangles were, and join the
rest of the surface edge for edge -- so the surface's edges used once are exactly the face's rims.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from aquaflux.meshing import MappedQuads, PipeGrading, Polygons
from scipy.spatial import Delaunay, cKDTree


def _rings(points: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Triangles joining two closed rings of points, each step along the shorter diagonal."""
    first, second = np.append(first, first[0]), np.append(second, second[0])
    triangles, i, j = [], 0, 0
    while i < first.size - 1 or j < second.size - 1:
        if j == second.size - 1 or (
            i < first.size - 1
            and np.linalg.norm(points[first[i + 1]] - points[second[j]])
            <= np.linalg.norm(points[second[j + 1]] - points[first[i]])
        ):
            triangles.append([first[i], first[i + 1], second[j]])
            i += 1
        else:
            triangles.append([first[i], second[j + 1], second[j]])
            j += 1
    return np.array(triangles)


def _band(bottom: int, top: int, rows: int = 5):
    """An open cylinder of radius 1, ``z`` in ``[0, 2]``, triangulated outward, with ``bottom``
    and ``top`` points on its rims."""
    counts = np.round(np.linspace(bottom, top, rows + 1)).astype(int)
    points, rings = [], []
    for j, n in enumerate(counts):
        angle = 2 * np.pi * (np.arange(n) + 0.3 * j) / n
        rings.append(len(points) + np.arange(n))
        points += [[np.cos(a), np.sin(a), 2.0 * j / rows] for a in angle]
    points = np.array(points)
    triangles = np.concatenate([_rings(points, rings[j], rings[j + 1]) for j in range(rows)])
    return points, _outward(points, triangles, lambda c: np.c_[c[:, :2], np.zeros(len(c))])


def _square(counts):
    """The unit square in ``z = 0``, ``counts`` segments on its sides (bottom, right, top, left),
    graded along each, triangulated with normals ``+z``."""
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    boundary = []
    for k, n in enumerate(counts):
        t = (np.arange(n) / n) ** 1.3
        boundary += list(corners[k] + t[:, None] * (corners[(k + 1) % 4] - corners[k]))
    inside = np.random.default_rng(0).uniform(0.1, 0.9, (30, 2))
    plane = np.vstack([boundary, inside])
    points = np.c_[plane, np.zeros(len(plane))]
    triangles = Delaunay(plane).simplices
    return points, _outward(points, triangles, lambda c: np.tile([0.0, 0.0, 1.0], (len(c), 1)))


def _outward(points, triangles, outward):
    corner = points[triangles]
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    inward = np.einsum("ij,ij->i", normal, outward(corner.mean(axis=1))) < 0
    triangles[inward] = triangles[inward][:, ::-1]
    return triangles


def _onto_cylinder(points, faces):
    radius = np.linalg.norm(points[:, :2], axis=1, keepdims=True)
    landed = np.c_[points[:, :2] / radius, points[:, 2]]
    return landed, np.c_[landed[:, :2], np.zeros(len(points))]


def _onto_plane(points, faces):
    normal = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    return np.c_[points[:, :2], np.zeros(len(points))], normal


SQUARE_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)


def _open_edges(faces: Polygons) -> set[tuple[int, int]]:
    """The edges used once, as directed pairs: an edge used twice is used once each way."""
    pairs, _ = faces.edges()
    used = Counter(map(tuple, pairs.tolist()))
    assert all(count == 1 for count in used.values())
    return {(a, b) for (a, b) in used if (b, a) not in used}


def _area(points, faces: Polygons) -> float:
    corner = points[faces.as_triangles()]
    return 0.5 * float(
        np.linalg.norm(
            np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]), axis=1
        ).sum()
    )


@pytest.mark.parametrize(("bottom", "top"), [(24, 24), (24, 30), (30, 24)])
def test_a_band_becomes_a_grid_between_its_rims(bottom, top):
    points, triangles = _band(bottom, top)
    face = np.zeros(len(triangles), dtype=np.int64)
    rims = _open_edges(Polygons(triangles))
    mapped = MappedQuads().remesh(
        points, triangles, face, np.array([0]), np.zeros((0, 3)), _onto_cylinder
    )
    faces = mapped.faces
    # The rims are kept point for point: the boundary is the same polyline, renumbered.
    kept = {tuple(np.round(points[[a, b]], 12).ravel()) for a, b in rims}
    new = {tuple(np.round(mapped.points[[a, b]], 12).ravel()) for a, b in _open_edges(faces)}
    assert new == kept
    # Every point on the cylinder, each face wound outward like the triangles.
    np.testing.assert_allclose(np.linalg.norm(mapped.points[:, :2], axis=1), 1.0, atol=1e-14)
    centroid = faces.centroids(mapped.points)
    radial = np.c_[centroid[:, :2], np.zeros(len(centroid))]
    assert np.all(np.einsum("ij,ij->i", faces.normals(mapped.points), radial) > 0.9)
    # Rows about as tall as the cells are wide: the band's height over the chord between the
    # points of the rim with more of them.
    n = max(bottom, top)
    rows = round(2.0 / (2.0 * np.sin(np.pi / n)))
    if bottom == top:
        assert faces.triangles.shape == (0, 3)
        assert faces.quads.shape[0] == n * rows
    else:
        # One row short of the smaller rim, then a strip of triangles: one per point of the two.
        assert faces.quads.shape[0] == n * (rows - 1)
        assert faces.triangles.shape[0] == bottom + top
    assert _area(mapped.points, faces) == pytest.approx(4 * np.pi, rel=5e-3)
    # Each quadrilateral starts along the grid's rows, round the band: its first edge is level.
    corner = mapped.points[faces.quads]
    np.testing.assert_allclose(corner[:, 0, 2], corner[:, 1, 2], atol=1e-12)
    np.testing.assert_array_equal(mapped.face, np.zeros(faces.n_faces))


@pytest.mark.parametrize(
    ("counts", "quads", "strip", "inside"),
    [((6, 4, 6, 4), 24, 0, 15), ((6, 4, 8, 4), 18, 14, 15), ((6, 5, 6, 3), 25, 8, 20)],
)
def test_a_four_sided_face_becomes_a_coons_grid(counts, quads, strip, inside):
    # Equal opposite sides: a whole grid. One pair unequal: the grid stops a row short of the
    # longer-counted side's opposite, and a strip of triangles joins it there.
    points, triangles = _square(counts)
    face = np.zeros(len(triangles), dtype=np.int64)
    mapped = MappedQuads().remesh(
        points, triangles, face, np.array([0]), SQUARE_CORNERS, _onto_plane
    )
    assert mapped.faces.quads.shape[0] == quads
    assert mapped.faces.triangles.shape[0] == strip
    # The sides' points and the grid's own, the triangulation's inner points gone.
    assert mapped.points.shape[0] == sum(counts) + inside
    assert len(_open_edges(mapped.faces)) == sum(counts)
    assert _area(mapped.points, mapped.faces) == pytest.approx(1.0, rel=1e-12)
    assert np.all(mapped.faces.normals(mapped.points)[:, 2] > 0.99)


def test_a_face_with_neither_pair_of_sides_equal_is_refused():
    points, triangles = _square((6, 5, 7, 3))
    face = np.zeros(len(triangles), dtype=np.int64)
    with pytest.raises(ValueError, match=r"sides of \[6, 5, 7, 3\] segments"):
        MappedQuads().remesh(points, triangles, face, np.array([0]), SQUARE_CORNERS, _onto_plane)


def test_a_face_with_other_than_four_corners_is_refused():
    points, triangles = _square((6, 4, 6, 4))
    face = np.zeros(len(triangles), dtype=np.int64)
    with pytest.raises(ValueError, match="one loop with 3 corner"):
        MappedQuads().remesh(
            points, triangles, face, np.array([0]), SQUARE_CORNERS[:3], _onto_plane
        )


def test_faces_not_chosen_keep_their_triangles():
    # The band's lower half is face 0, its upper half face 1; only face 1 is meshed again.
    points, triangles = _band(24, 24, rows=6)
    face = (points[triangles][:, :, 2].mean(axis=1) > 1.0).astype(np.int64)
    mapped = MappedQuads().remesh(
        points, triangles, face, np.array([1]), np.zeros((0, 3)), _onto_cylinder
    )
    kept = mapped.faces.split(mapped.face)
    assert np.all(kept[0] == 0)
    assert np.all(kept[1] == 1)
    assert mapped.faces.triangles.shape[0] == np.count_nonzero(face == 0)
    # Joined edge for edge: the only edges used once are the band's two rims.
    assert len(_open_edges(mapped.faces)) == 48


def test_a_bands_rows_flatten_within_two_rows_of_a_rim_that_is_not_flat():
    # A cylinder from a saddle-shaped rim (z = 0.3 cos 2 theta, as a pipe meets a vessel) to a flat
    # one at z = 2: the rows run from the saddle's shape to flat rings, flat from the second row in,
    # so a pipe's cross-sections away from its junction are flat.
    n, rows = 24, 6
    angle = 2 * np.pi * np.arange(n) / n
    rings, points = [], []
    for j in range(rows + 1):
        t = j / rows
        z = (1 - t) * 0.3 * np.cos(2 * angle) + 2.0 * t
        rings.append(len(points) + np.arange(n))
        points += [[np.cos(a), np.sin(a), h] for a, h in zip(angle, z, strict=True)]
    points = np.array(points)
    triangles = np.concatenate([_rings(points, rings[j], rings[j + 1]) for j in range(rows)])
    triangles = _outward(points, triangles, lambda c: np.c_[c[:, :2], np.zeros(len(c))])
    face = np.zeros(len(triangles), dtype=np.int64)
    mapped = MappedQuads().remesh(
        points, triangles, face, np.array([0]), np.zeros((0, 3)), _onto_cylinder
    )
    grid = mapped.grids[0]
    z = mapped.points[grid][:, :, 2]
    bow = z.max(axis=1) - z.min(axis=1)
    saddle_first = bow[0] > bow[-1]
    if not saddle_first:
        bow = bow[::-1]
    assert bow[0] == pytest.approx(0.6, rel=1e-9)  # the saddle itself
    assert 0.1 < bow[1] < 0.5  # half way out of it
    np.testing.assert_allclose(bow[2:], 0.0, atol=1e-12)


# A quarter bend: a torus of centreline radius BEND about the z axis through (0, BEND, 0), its tube
# of radius 1, turning from the plane x = 0 (angle 0) to the plane y = BEND (a quarter turn).
BEND = 3.0


def _bend_section(angle: float, n: int) -> np.ndarray:
    """The bend's cross-section at ``angle`` as ``n`` points round it, the first on top (+z)."""
    along = np.array([np.sin(angle), -np.cos(angle), 0.0])  # away from the bend's axis
    centre = np.array([0.0, BEND, 0.0]) + BEND * along
    t = 0.5 * np.pi + 2 * np.pi * np.arange(n) / n
    return centre + np.cos(t)[:, None] * along + np.sin(t)[:, None] * np.array([0.0, 0.0, 1.0])


def _bend_band(n: int = 24, rows: int = 4):
    """The bend's wall, triangulated outward: ``rows`` rings of ``n`` points between its rims."""
    points, rings = [], []
    for j in range(rows + 1):
        rings.append(len(points) + np.arange(n))
        points += list(_bend_section(0.5 * np.pi * j / rows, n))
    points = np.array(points)
    triangles = np.concatenate([_rings(points, rings[j], rings[j + 1]) for j in range(rows)])
    return points, _outward(points, triangles, lambda c: c - _onto_centreline(c))


def _bend_angle(points: np.ndarray) -> np.ndarray:
    offset = points - np.array([0.0, BEND, 0.0])
    return np.arctan2(offset[:, 0], -offset[:, 1])


def _onto_centreline(points: np.ndarray) -> np.ndarray:
    angle = _bend_angle(points)
    return np.array([0.0, BEND, 0.0]) + BEND * np.c_[np.sin(angle), -np.cos(angle), 0 * angle]


def _onto_torus(points, faces):
    centre = _onto_centreline(points)
    normal = points - centre
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return centre + normal, normal


def test_a_bends_band_becomes_a_grid_of_cross_sections_turned_round_its_axis():
    # Each row of the grid is one cross-section of the bend -- every point at one angle round the
    # bend's axis, all on the unit circle about the centreline there -- at equal steps from rim to
    # rim, as many as keep the cells as long round the bend as they are wide. Interpolated straight
    # from rim to rim instead, a row's points spread over a range of angles.
    n = 24
    points, triangles = _bend_band(n)
    face = np.zeros(len(triangles), dtype=np.int64)
    mapped = MappedQuads().remesh(
        points, triangles, face, np.array([0]), np.zeros((0, 3)), _onto_torus
    )
    grid = mapped.grids[0]
    rows = round(BEND * 0.5 * np.pi / (2 * np.sin(np.pi / n)))
    assert grid.shape == (rows + 1, n)
    angle = _bend_angle(mapped.points[grid].reshape(-1, 3)).reshape(grid.shape)
    steps = 0.5 * np.pi * np.arange(rows + 1) / rows
    if angle[0, 0] > angle[-1, 0]:
        steps = steps[::-1]
    np.testing.assert_allclose(angle, np.repeat(steps[:, None], n, axis=1), atol=1e-12)
    # Unprojected, the rows are already the circles: the motion carries the rim along the wall.
    built = mapped.points[grid[1:-1]].reshape(-1, 3)
    radius = np.linalg.norm(built - _onto_centreline(built), axis=1)
    np.testing.assert_allclose(radius, 1.0, atol=1e-12)
    # The columns run along the bend: column i keeps its place round each cross-section.
    top = mapped.points[grid[:, 0]]
    centre = _onto_centreline(top)
    heights = (top - centre)[:, 2]
    assert np.all(np.abs(heights - heights[0]) < 1e-12)


def test_a_rim_motion_fitted_between_a_loop_and_its_screwed_image_is_that_screw():
    from aquaflux.meshing.mapped import RimMotion

    rng = np.random.default_rng(3)
    loop = rng.normal(size=(17, 3))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    screw = RimMotion(axis, rng.normal(size=3), 1.1, 0.4 * axis)
    fitted = RimMotion.fit(loop, screw.at(1.0, loop))
    assert fitted.angle == pytest.approx(1.1, abs=1e-12)
    for t in (0.25, 0.5, 0.8):
        np.testing.assert_allclose(fitted.at(t, loop), screw.at(t, loop), atol=1e-12)
    # A point a distance rho from the line travels the helix of that radius.
    offset = loop - screw.pivot
    rho = np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1)
    np.testing.assert_allclose(fitted.path_length(loop), np.hypot(1.1 * rho, 0.4), rtol=1e-12)


def test_a_rim_motion_between_two_rims_of_a_straight_pipe_is_a_slide_despite_rounding():
    # Netgen's points on two rims of a straight pipe fit with a turn of rounding error; about its
    # line every point is far away, and taken as a turn the rows' length would be that distance.
    from aquaflux.meshing.mapped import RimMotion

    t = 2 * np.pi * np.arange(36) / 36
    rim = np.c_[np.zeros(36), np.cos(t), np.sin(t)]
    other = (
        rim
        + np.array([3.0, 0.0, 0.0])
        + np.random.default_rng(0).normal(scale=1e-12, size=rim.shape)
    )
    motion = RimMotion.fit(rim, other)
    assert motion.angle == 0.0
    np.testing.assert_allclose(motion.path_length(rim), 3.0, rtol=1e-9)
    np.testing.assert_allclose(motion.at(0.5, rim), rim + np.array([1.5, 0.0, 0.0]), atol=1e-9)


# An S-bend in one face: a quarter turn of radius TURN one way, then a quarter turn back, in the
# plane z = 0, its tube of radius 1. Its rims are parallel, so the one rigid motion between them
# is a slide, and only a march along it finds its cross-sections.
TURN = 4.0


def _s_bend_centreline(s: np.ndarray):
    """The centreline at arc length ``s`` (from 0 to pi TURN) and its unit tangent there."""
    half = 0.5 * np.pi * TURN
    first = s <= half
    angle = np.where(first, s / TURN, (2 * half - s) / TURN)
    centre_first = np.c_[TURN * np.sin(angle), TURN * (1 - np.cos(angle)), 0 * s]
    # The second turn is the first turned half a circle about the S's middle.
    middle = np.array([TURN, TURN, 0.0])
    centre_second = 2 * middle - centre_first
    point = np.where(first[:, None], centre_first, centre_second)
    tangent = np.c_[np.cos(angle), np.sin(angle), 0 * s]
    return point, tangent


def _s_bend_band(n: int = 24, rings: int = 64):
    points, loops = [], []
    s = np.linspace(0.0, np.pi * TURN, rings + 1)
    centre, tangent = _s_bend_centreline(s)
    t = 0.5 * np.pi + 2 * np.pi * np.arange(n) / n
    for c, tan in zip(centre, tangent, strict=True):
        side = np.cross([0.0, 0.0, 1.0], tan)
        loops.append(len(points) + np.arange(n))
        points += list(c + np.cos(t)[:, None] * side + np.sin(t)[:, None] * [0.0, 0.0, 1.0])
    points = np.array(points)
    triangles = np.concatenate([_rings(points, loops[j], loops[j + 1]) for j in range(rings)])
    return points, _outward(points, triangles, lambda c: c - _s_bend_nearest(c)[0])


_S_SAMPLES = np.linspace(0.0, np.pi * TURN, 20001)
_S_CENTRE, _S_TANGENT = _s_bend_centreline(_S_SAMPLES)
_S_TREE = cKDTree(_S_CENTRE)


def _s_bend_nearest(points):
    """The nearest centreline point to each point (to 6e-4 of the tube's radius), the
    centreline's tangent there and its arc length."""
    _, at = _S_TREE.query(points)
    return _S_CENTRE[at], _S_TANGENT[at], _S_SAMPLES[at]


def _onto_s_bend(points, faces):
    centre = _s_bend_nearest(points)[0]
    normal = points - centre
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return centre + normal, normal


def test_an_s_bends_band_becomes_a_grid_of_its_cross_sections_marched_along_it():
    # Each row is a cross-section: its plane square to the centreline where it crosses it, the
    # rows evenly spaced along it. Swept by the slide between its parallel rims instead, the rows
    # in the middle of the S would stand at 45 degrees to it; with the wall's normals taken from
    # its facets (averaged at their vertices), the row at the S's inflection leant 1.5 degrees.
    n = 24
    points, triangles = _s_bend_band(n)
    face = np.zeros(len(triangles), dtype=np.int64)
    mapped = MappedQuads().remesh(
        points, triangles, face, np.array([0]), np.zeros((0, 3)), _onto_s_bend
    )
    grid = mapped.grids[0]
    rows = round(np.pi * TURN / (2 * np.sin(np.pi / n)))
    assert grid.shape[0] - 1 == pytest.approx(rows, abs=1)
    ring = mapped.points[grid]
    centroid = ring.mean(axis=1)
    _, tangent, arc = _s_bend_nearest(centroid)
    offset = ring - centroid[:, None]
    normal = np.cross(offset, np.roll(offset, -1, axis=1)).sum(axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", normal, tangent)), 0, 1)))
    assert tilt.max() < 0.1
    spacing = np.diff(np.sort(arc))
    assert spacing.max() / spacing.min() < 1.02


@pytest.mark.parametrize("open_rim", ["top", "bottom"])
def test_a_straight_bands_rows_grow_away_from_its_joined_rim_toward_its_open_one(open_rim):
    # The cylinder z in [0, 2], its top rim (or bottom) an open end: the rows start a spacing
    # long at the other rim and grow by the grading's ratio toward it.
    n = 24
    points, triangles = _band(n, n)
    face = np.zeros(len(triangles), dtype=np.int64)
    z = points[:, 2]
    open_points = np.flatnonzero(np.isclose(z, 2.0 if open_rim == "top" else 0.0))
    grading = PipeGrading(growth=1.15)
    mapped = MappedQuads().remesh(
        points, triangles, face, np.array([0]), np.zeros((0, 3)), _onto_cylinder,
        graded={0: (grading, open_points)},
    )  # fmt: skip
    heights = mapped.points[mapped.grids[0]][:, :, 2].mean(axis=1)
    if heights[0] > heights[-1]:
        heights = heights[::-1]
    rows = np.diff(heights)
    if open_rim == "bottom":
        rows = rows[::-1]
    np.testing.assert_allclose(rows[1:] / rows[:-1], 1.15, rtol=1e-9)
    assert rows[0] == pytest.approx(2 * np.sin(np.pi / n), rel=0.15)


def test_a_bends_rows_stay_even_under_a_grading():
    # Grading is for straight lengths: round a bend the flow changes all along it.
    n = 24
    points, triangles = _bend_band(n)
    face = np.zeros(len(triangles), dtype=np.int64)
    rim = np.flatnonzero(np.isclose(_bend_angle(points), 0.5 * np.pi))
    mapped = MappedQuads().remesh(
        points, triangles, face, np.array([0]), np.zeros((0, 3)), _onto_torus,
        graded={0: (PipeGrading(growth=1.3), rim)},
    )  # fmt: skip
    angle = _bend_angle(mapped.points[mapped.grids[0]][:, 0])
    np.testing.assert_allclose(np.abs(np.diff(angle)), np.abs(np.diff(angle)).mean(), rtol=1e-9)


def test_triangles_wound_unalike_are_refused_rather_than_walked_for_ever():
    from aquaflux.meshing.mapped import _loops

    # A square of two triangles, the second wound the other way: its boundary edges do not chain.
    with pytest.raises(ValueError, match="not wound alike"):
        _loops(np.array([[0, 1, 2], [0, 3, 2]]))
    assert [loop.tolist() for loop in _loops(np.array([[0, 1, 2], [0, 2, 3]]))] == [[0, 1, 2, 3]]
