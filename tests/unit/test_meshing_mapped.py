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
from aquaflux.meshing import MappedQuads, Polygons
from scipy.spatial import Delaunay


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
    return np.c_[points[:, :2] / radius, points[:, 2]]


def _onto_plane(points, faces):
    return np.c_[points[:, :2], np.zeros(len(points))]


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
