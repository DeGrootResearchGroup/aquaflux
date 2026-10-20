"""The face planarizer, checked against meshes whose answer is known or whose guards must bite.

A block of hexahedra with its interior nodes displaced has a known flat solution, the lattice. On
duals of perturbed Kuhn tetrahedra the faces cannot all be flattened, and the unguarded iteration
is known to make a cube's faces more non-orthogonal and skewed, and an L's cut cells less convex:
those are the fixtures that show each bound holding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.mesh import (
    CONCAVE_COSINE,
    Mesh,
    cell_concavity,
    cell_non_convexity,
    closed_cell_residual,
    face_non_orthogonality,
    face_skewness,
    face_warp,
    structured_grid_2d,
    structured_grid_3d,
)
from aquaflux.meshing import FacePlanarizer, PolyhedralDual

from tests.support.meshes import perturbed_grid_3d, tetrahedral_grid_3d, without_cells

SHARP = PolyhedralDual(feature_angle=math.radians(45))


def _off_the_boundary(mesh: Mesh) -> np.ndarray:
    """The nodes on no boundary face."""
    offsets = np.asarray(mesh.face_nodes.offsets)
    nodes = np.asarray(mesh.face_nodes.face_node_indices)
    on = np.zeros(mesh.n_nodes, dtype=bool)
    for face in np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0):
        on[nodes[offsets[face] : offsets[face + 1]]] = True
    return ~on


def _dual_of(primal: Mesh) -> tuple[Mesh, np.ndarray]:
    """The dual of ``primal`` and its free vertices: the primal cell centroids, numbered first."""
    dual = SHARP.build(primal)
    return dual, np.arange(dual.n_nodes) < primal.n_cells


def test_a_displaced_hex_block_is_flattened_back_and_its_boundary_kept():
    mesh = perturbed_grid_3d(3, 3, 3, perturb=0.15, seed=4)
    movable = _off_the_boundary(mesh)
    assert float(np.max(np.asarray(face_warp(mesh)))) > 0.02
    result = FacePlanarizer(min_decrease=1e-6).planarize(mesh, movable)
    assert result.converged
    assert float(np.max(np.asarray(face_warp(result.mesh)))) < 1e-3
    before, after = np.asarray(mesh.node_coords), np.asarray(result.mesh.node_coords)
    np.testing.assert_array_equal(after[~movable], before[~movable])
    assert float(np.max(np.asarray(closed_cell_residual(result.mesh)))) < 1e-14


def test_a_flat_mesh_is_left_alone():
    mesh = structured_grid_3d(2, 2, 2, 1.0, 1.0, 1.0)
    result = FacePlanarizer().planarize(mesh, _off_the_boundary(mesh))
    assert result.converged and result.sweeps == 0
    np.testing.assert_array_equal(np.asarray(result.mesh.node_coords), np.asarray(mesh.node_coords))


def test_no_face_ends_more_non_orthogonal_or_skewed_than_the_worst_it_started_with():
    # Each bound bites here: without the non-orthogonality bound the worst face rises by about 3
    # degrees, without the skewness bound the worst skewness by about 0.4.
    dual, movable = _dual_of(tetrahedral_grid_3d(3, perturb=0.25))
    result = FacePlanarizer().planarize(dual, movable)
    assert result.warp[-1] < 0.5 * result.warp[0]
    for measure in (face_non_orthogonality, face_skewness):
        start = float(np.max(np.asarray(measure(dual))))
        assert float(np.max(np.asarray(measure(result.mesh)))) <= start + 1e-12


def _kuhn_l(n: int, perturb: float, seed: int) -> Mesh:
    block = tetrahedral_grid_3d(n, perturb=perturb, seed=seed)
    centroid = np.asarray(block.geometry().cell.centroid)
    corner = (centroid[:, 0] > 0.5) & (centroid[:, 2] > 0.5)
    return without_cells(block, np.flatnonzero(corner), patch="cut")


def test_no_cell_ends_less_convex_or_more_concave_than_it_started():
    # Each bound bites here: without the convexity bound five cells end less convex, without the
    # concavity bound two more concave.
    dual, movable = _dual_of(_kuhn_l(4, 0.25, 0))
    result = FacePlanarizer().planarize(dual, movable)
    assert result.warp[-1] < 0.6 * result.warp[0]
    assert np.all(cell_non_convexity(result.mesh) <= cell_non_convexity(dual) + 1e-12)
    bound = np.maximum(cell_concavity(dual), CONCAVE_COSINE)
    assert np.all(cell_concavity(result.mesh) <= bound + 1e-12)
    assert np.all(np.asarray(result.mesh.geometry().cell.volume) > 0.0)


def test_the_pass_stops_on_its_convergence_test_not_its_cap():
    dual, movable = _dual_of(tetrahedral_grid_3d(3, perturb=0.25))
    loose = FacePlanarizer(min_decrease=1e-2).planarize(dual, movable)
    tight = FacePlanarizer(min_decrease=1e-4).planarize(dual, movable)
    assert loose.converged and tight.converged
    assert loose.sweeps < tight.sweeps < FacePlanarizer().max_sweeps
    # The last sweep is the first to fall short of its decrease, and none before it did.
    warp = np.asarray(loose.warp)
    decrease = (warp[:-1] - warp[1:]) / warp[:-1]
    assert decrease[-1] < 1e-2 and np.all(decrease[:-1] >= 1e-2)
    capped = FacePlanarizer(min_decrease=1e-4, max_sweeps=3).planarize(dual, movable)
    assert capped.sweeps == 3 and not capped.converged


def test_the_planarizer_refuses_a_flat_mesh_and_a_mask_of_the_wrong_shape():
    with pytest.raises(ValueError, match="three-dimensional"):
        FacePlanarizer().planarize(structured_grid_2d(2, 2), np.ones(9, dtype=bool))
    mesh = structured_grid_3d(1, 1, 1)
    with pytest.raises(ValueError, match="movable must have shape"):
        FacePlanarizer().planarize(mesh, np.ones(3, dtype=bool))
