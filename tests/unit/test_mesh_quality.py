"""Mesh-quality metrics, each against a value worked out by hand on a small mesh.

Non-orthogonality and skewness follow OpenFOAM's ``checkMesh``: on a sheared pair of boxes the
centre line's tilt and the offset of where it crosses the shared face are known in closed form, and
an elongated pair makes the length floors of the skewness definition the binding term. Face warp
and the volume a re-triangulation can add are checked on a cube with one corner raised, the second
against volumes of explicitly triangulated polyhedra. Convexity is checked on a cube and an
L-shaped prism, whose excursions and face-centroid cosines are plain arithmetic.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.mesh import (
    Mesh,
    cell_concavity,
    cell_non_convexity,
    face_non_orthogonality,
    face_skewness,
    face_warp,
    stored_ring_is_outward,
    structured_grid_2d,
    structured_grid_3d,
    triangulation_volume_change,
)
from aquaflux.meshing import PolyhedralDual

from tests.support.meshes import tetrahedral_grid_3d

SHEAR = 0.4


def _sheared_pair(length: float = 2.0) -> Mesh:
    """Two boxes along x, ``length`` long in all, the far end's nodes moved ``SHEAR`` along y.

    The second box is the image of an axis-aligned box under ``y -> y + SHEAR (x - x_mid) / half``,
    an affine map, so its centroid is the image of the box's: it sits ``SHEAR / 2`` up in y.
    """
    mesh = structured_grid_3d(2, 1, 1, length, 1.0, 1.0)
    coords = np.asarray(mesh.node_coords).copy()
    coords[np.isclose(coords[:, 0], length), 1] += SHEAR
    return eqx.tree_at(lambda m: m.node_coords, mesh, jnp.asarray(coords))


def _face_at(mesh: Mesh, point) -> int:
    centroid = np.asarray(mesh.geometry().face.centroid)
    return int(np.argmin(np.linalg.norm(centroid - np.asarray(point), axis=1)))


def test_the_shared_face_of_a_sheared_pair_tilts_by_the_centroid_offset():
    # Owner centroid (0.5, 0.5, 0.5), neighbour (1.5, 0.5 + SHEAR/2, 0.5): the line between them
    # leans atan(SHEAR / 2) off the shared face's normal (x).
    mesh = _sheared_pair()
    angle = np.asarray(face_non_orthogonality(mesh))
    assert angle[_face_at(mesh, (1.0, 0.5, 0.5))] == pytest.approx(
        math.degrees(math.atan(SHEAR / 2)), abs=1e-10
    )
    # The far wall is still the plane x = 2, its centroid moved SHEAR up: from the owner's centroid
    # (half a box away in x, SHEAR/2 in y) the angle is atan(SHEAR).
    far = _face_at(mesh, (2.0, 0.5 + SHEAR, 0.5))
    assert angle[far] == pytest.approx(math.degrees(math.atan(SHEAR)), abs=1e-10)
    assert angle[_face_at(mesh, (0.0, 0.5, 0.5))] == pytest.approx(0.0, abs=1e-6)


def test_an_orthogonal_periodic_grid_has_no_non_orthogonality_across_its_seam():
    # The seam's raw neighbour centroid lies a period away; only its periodic image is in line.
    mesh = structured_grid_2d(4, 3, 2.0, 1.0, periodic=("x",))
    assert float(np.max(np.asarray(face_non_orthogonality(mesh)))) < 1e-6


def test_skewness_is_the_offset_of_the_crossing_over_the_faces_extent():
    # The centre line crosses x = 1 at y = 0.5 + SHEAR/4; the face centroid is at y = 0.5, so the
    # skewness vector is SHEAR/4 long, over the face's half-width 0.5 in y.
    mesh = _sheared_pair()
    skew = np.asarray(face_skewness(mesh))
    assert skew[_face_at(mesh, (1.0, 0.5, 0.5))] == pytest.approx(SHEAR / 2, rel=1e-12)
    # On the far wall the face centroid sits SHEAR/2 tangentially off the owner's: over 0.5.
    assert skew[_face_at(mesh, (2.0, 0.5 + SHEAR, 0.5))] == pytest.approx(SHEAR, rel=1e-12)


def test_on_an_elongated_pair_the_length_floors_set_the_skewness():
    # Boxes 5 long: the interior floor 0.2 |d| (|d| = sqrt(25 + SHEAR^2/4)) exceeds the half-width,
    # and on the far wall the floor 0.4 |d| with |d| = 2.5, the owner centroid's distance to it.
    mesh = _sheared_pair(length=10.0)
    skew = np.asarray(face_skewness(mesh))
    across = math.hypot(5.0, SHEAR / 2)
    assert skew[_face_at(mesh, (5.0, 0.5, 0.5))] == pytest.approx(
        (SHEAR / 4) / (0.2 * across), rel=1e-12
    )
    assert skew[_face_at(mesh, (10.0, 0.5 + SHEAR, 0.5))] == pytest.approx(
        (SHEAR / 2) / (0.4 * 2.5), rel=1e-12
    )


RAISE = 0.2


def _cube_with_a_raised_corner() -> tuple[np.ndarray, list[list[int]]]:
    """A unit cube, node (1, 1, 1) raised by ``RAISE``; faces wound outward."""
    coords = np.array(
        [(x, y, z) for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)], dtype=float
    )
    coords[7, 2] += RAISE
    faces = [
        [0, 2, 3, 1],  # z = 0
        [4, 5, 7, 6],  # z = 1, the warped one
        [0, 1, 5, 4],  # y = 0
        [2, 6, 7, 3],  # y = 1
        [0, 4, 6, 2],  # x = 0
        [1, 3, 7, 5],  # x = 1
    ]
    return coords, faces


def _single_cell(coords, faces) -> Mesh:
    n = len(faces)
    return Mesh.from_faces(coords, faces, np.zeros(n, int), -np.ones(n, int), 1)


def test_the_warp_of_a_quadrilateral_with_one_corner_raised():
    # Top face (0,0,1) (1,0,1) (1,1,1+h) (0,1,1): area vector (-h, -h, 2) / 2, vertex mean
    # (1/2, 1/2, 1 + h/4); every corner stands (h/2) / sqrt(2 h^2 + 4) off that plane.
    coords, faces = _cube_with_a_raised_corner()
    mesh = _single_cell(coords, faces)
    warp = np.asarray(face_warp(mesh))
    height = (RAISE / 2) / math.sqrt(2 * RAISE**2 + 4)
    area = math.sqrt(2 * RAISE**2 + 4) / 2
    top = _face_at(mesh, (0.5, 0.5, 1.0 + RAISE / 4))
    assert warp[top] == pytest.approx(height / math.sqrt(area), rel=1e-12)
    # The faces through the raised corner stay planar (each lies in x = 1 or y = 1).
    assert float(np.max(np.delete(warp, top))) < 1e-12


def _triangulated_volume(coords, faces, fans) -> float:
    """Volume of the polyhedron whose faces are fanned from ``fans[f]`` (``None``: vertex mean)."""
    total = 0.0
    for ring, apex in zip(faces, fans, strict=True):
        points = coords[ring]
        hub = points.mean(axis=0) if apex is None else points[apex]
        for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
            total += np.dot(hub, np.cross(a, b)) / 6.0
    return total


def test_the_volume_change_a_retriangulation_can_make_matches_explicit_triangulations():
    coords, faces = _cube_with_a_raised_corner()
    mesh = _single_cell(coords, faces)
    centre = _triangulated_volume(coords, faces, [None] * 6)
    # Only the top face is warped; fan it from each of its corners in turn.
    changes = [
        _triangulated_volume(coords, faces, [None, k, None, None, None, None]) - centre
        for k in range(4)
    ]
    volume = float(np.asarray(mesh.geometry().cell.volume)[0])
    assert max(abs(c) for c in changes) > 1e-3  # a real dependence on the triangulation
    assert float(triangulation_volume_change(mesh)[0]) == pytest.approx(
        max(abs(c) for c in changes) / volume, rel=1e-10
    )


def test_on_a_dual_the_volume_change_is_the_explicit_triangulations_largest():
    # Faces of five and more vertices, owned and neighboured: per cell, re-fan each face from each
    # of its vertices as explicit triangles (wound out of the cell) and take, independently per
    # face, the change that adds most and the one that removes most.
    dual = PolyhedralDual(feature_angle=math.radians(45)).build(tetrahedral_grid_3d(2, perturb=0.2))
    coords = np.asarray(dual.node_coords)
    offsets = np.asarray(dual.face_nodes.offsets)
    nodes = np.asarray(dual.face_nodes.face_node_indices)
    outward = stored_ring_is_outward(dual)
    owner, neighbour = np.asarray(dual.face_cells.owner), np.asarray(dual.face_cells.neighbour)
    gain, loss = np.zeros(dual.n_cells), np.zeros(dual.n_cells)
    for face in range(dual.n_faces):
        ring = nodes[offsets[face] : offsets[face + 1]]
        for cell, flip in ((owner[face], not outward[face]), (neighbour[face], outward[face])):
            if cell < 0:
                continue
            points = coords[ring[::-1] if flip else ring]
            centre = _fan_volume(points, points.mean(axis=0))
            change = [_fan_volume(points, apex) - centre for apex in points]
            gain[cell] += max(change)
            loss[cell] += max(-c for c in change)
    assert np.unique(np.diff(offsets)).size > 1 and np.max(np.diff(offsets)) >= 5
    expected = np.maximum(gain, loss) / np.asarray(dual.geometry().cell.volume)
    assert float(expected.max()) > 1e-3
    np.testing.assert_allclose(
        np.asarray(triangulation_volume_change(dual)), expected, rtol=1e-9, atol=1e-12
    )


def _fan_volume(points: np.ndarray, apex: np.ndarray) -> float:
    """The volume a ring's fan of triangles from ``apex`` contributes, seen from the origin."""
    return sum(
        np.dot(apex, np.cross(a, b)) / 6.0
        for a, b in zip(points, np.roll(points, -1, axis=0), strict=True)
    )


def _l_prism() -> Mesh:
    """One cell: the L of three unit squares, (0,0)-(2,1) and (0,1)-(1,2), extruded 1 in z."""
    ring = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    coords = np.array([(x, y, z) for z in (0.0, 1.0) for x, y in ring], dtype=float)
    faces = [list(range(5, -1, -1)), list(range(6, 12))]
    faces += [[i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6] for i in range(6)]
    return _single_cell(coords, faces)


def test_a_cubes_vertices_are_all_behind_its_faces_and_an_ls_reach_a_unit_past_two():
    cube = structured_grid_3d(1, 1, 1, 1.0, 1.0, 1.0)
    assert float(cell_non_convexity(cube)[0]) == pytest.approx(0.0, abs=1e-12)
    # The walls meeting at the inside corner (1, 1): the far vertices of the other arm lie one
    # unit beyond each; the cell's size is the cube root of its volume, 3.
    assert float(cell_non_convexity(_l_prism())[0]) == pytest.approx(1 / 3 ** (1 / 3), rel=1e-12)


def test_neighbouring_cells_see_their_shared_face_from_their_own_side():
    # Each box of a pair is convex: viewed from the neighbour the shared face's normal is reversed.
    pair = structured_grid_3d(2, 1, 1, 2.0, 1.0, 1.0)
    np.testing.assert_allclose(cell_non_convexity(pair), 0.0, atol=1e-12)
    np.testing.assert_allclose(cell_concavity(pair), -1 / math.sqrt(2), rtol=1e-12)


def test_the_concavity_cosine_of_a_cube_and_of_an_l():
    # In a cube the nearest other face centroid is an adjacent face's, at 45 degrees behind.
    cube = structured_grid_3d(1, 1, 1, 1.0, 1.0, 1.0)
    assert float(cell_concavity(cube)[0]) == pytest.approx(-1 / math.sqrt(2), rel=1e-12)
    # From the wall y = 1 (x in [1, 2]) the centroid of the far wall y = 2 (x in [0, 1]) lies
    # 45 degrees in front: checkMesh calls the L concave.
    assert float(cell_concavity(_l_prism())[0]) == pytest.approx(1 / math.sqrt(2), rel=1e-12)


def test_the_cell_checks_are_the_same_formed_a_few_cells_at_a_time(monkeypatch):
    # A large mesh's cell-face pairs are formed in blocks; a block boundary must lose no pair.
    primal = tetrahedral_grid_3d(3, perturb=0.2)
    dual = PolyhedralDual(feature_angle=math.radians(45)).build(primal)
    whole_convexity, whole_concavity = cell_non_convexity(dual), cell_concavity(dual)
    import aquaflux.mesh.quality as quality

    monkeypatch.setattr(quality._CellSides, "_PAIRS_PER_BLOCK", 300)
    np.testing.assert_array_equal(cell_non_convexity(dual), whole_convexity)
    np.testing.assert_array_equal(cell_concavity(dual), whole_concavity)


def test_cell_nodes_are_each_cells_distinct_nodes():
    # The middle box of 3 x 3 x 3 has only interior faces, three of which it does not own.
    mesh = structured_grid_3d(3, 3, 3, 3.0, 3.0, 3.0)
    offsets, nodes = mesh.cell_nodes()
    coords = np.asarray(mesh.node_coords)
    centroid = np.asarray(mesh.geometry().cell.centroid)
    assert np.all(np.diff(offsets) == 8)
    for cell in range(mesh.n_cells):
        mine = nodes[offsets[cell] : offsets[cell + 1]]
        assert np.all(np.diff(mine) > 0)
        corner = np.all(np.abs(coords - centroid[cell]) < 0.5 + 1e-12, axis=1)
        np.testing.assert_array_equal(mine, np.flatnonzero(corner))
