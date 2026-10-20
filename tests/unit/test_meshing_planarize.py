"""The face planarizer, checked against meshes whose answer is known or whose guards must bite.

A block of hexahedra with its interior nodes displaced has a known flat solution, the lattice. On
duals of perturbed Kuhn tetrahedra the faces cannot all be flattened, and the unguarded iteration
is known to make a cube's faces more non-orthogonal and skewed, and an L's cut cells less convex:
those are the fixtures that show each bound holding. The derivatives the steps are solved with are
checked against finite differences of the warp as the guard computes it, a separate numpy path,
and the preconditioner against the blocks of an explicitly formed frozen-plane Jacobian.
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
from aquaflux.meshing.planarize import _block_inverse, _gradient, _product, _State, _Warp

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
    assert result.converged and result.iterations == 0
    np.testing.assert_array_equal(np.asarray(result.mesh.node_coords), np.asarray(mesh.node_coords))


def test_no_face_ends_more_non_orthogonal_or_skewed_than_the_worst_it_started_with():
    # Each bound bites here: without the non-orthogonality bound the worst face rises by about 3
    # degrees, without the skewness bound the worst skewness by about 0.4.
    dual, movable = _dual_of(tetrahedral_grid_3d(3, perturb=0.25))
    result = FacePlanarizer().planarize(dual, movable)
    # Pinning both cells of a face that breaks a bound reaches 7 % of the start; pinning only its
    # owner stops at about 13 %.
    assert result.warp[-1] < 0.1 * result.warp[0]
    assert np.all(np.diff(result.warp) < 0.0)  # every kept step lowered the warp
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
    assert loose.iterations < tight.iterations < FacePlanarizer().max_iterations
    # The last sweep is the first to fall short of its decrease, and none before it did.
    warp = np.asarray(loose.warp)
    decrease = (warp[:-1] - warp[1:]) / warp[:-1]
    assert decrease[-1] < 1e-2 and np.all(decrease[:-1] >= 1e-2)
    capped = FacePlanarizer(min_decrease=1e-4, max_iterations=3).planarize(dual, movable)
    assert capped.iterations == 3 and not capped.converged


def test_the_planarizer_refuses_a_flat_mesh_and_a_mask_of_the_wrong_shape():
    with pytest.raises(ValueError, match="three-dimensional"):
        FacePlanarizer().planarize(structured_grid_2d(2, 2), np.ones(9, dtype=bool))
    mesh = structured_grid_3d(1, 1, 1)
    with pytest.raises(ValueError, match="movable must have shape"):
        FacePlanarizer().planarize(mesh, np.ones(3, dtype=bool))


def _small_dual() -> tuple[Mesh, np.ndarray, np.ndarray]:
    dual, movable = _dual_of(tetrahedral_grid_3d(2, perturb=0.2))
    free = np.flatnonzero(movable)
    return dual, free, np.asarray(dual.node_coords, dtype=np.float64)


def _numpy_residual(dual: Mesh, coords: np.ndarray, free: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Each incidence's height over the root of its face's area, as the guard's state computes it."""
    moved = coords.copy()
    moved[free] = x.reshape(-1, 3)
    state = _State.at(dual, moved)
    return state.height / np.sqrt(state.area[state.face])


def test_the_steps_derivatives_are_those_of_the_warp_with_its_planes_moving():
    # A face's plane (its vertex mean and normal) moves with its vertices; a derivative that held
    # either still would differ from the finite difference by first order.
    dual, free, coords = _small_dual()
    warp = _Warp.of(dual, free)
    x = coords[free].reshape(-1)
    rng = np.random.default_rng(1)
    for _ in range(3):
        v = rng.standard_normal(x.size)
        eps = 1e-6
        ahead = _numpy_residual(dual, coords, free, x + eps * v)
        behind = _numpy_residual(dual, coords, free, x - eps * v)
        directional = (ahead - behind) / (2 * eps)  # J v
        energy = (np.sum(ahead**2) - np.sum(behind**2)) / (2 * eps)  # d|r|^2 along v
        gradient = np.asarray(_gradient(warp, x))
        assert float(gradient @ v) == pytest.approx(0.5 * energy, rel=1e-6)
        curvature = float(v @ np.asarray(_product(warp, x, v, 0.0)))
        assert curvature == pytest.approx(float(directional @ directional), rel=1e-6)


def test_the_preconditioner_is_the_inverse_of_the_frozen_plane_jacobians_blocks():
    # With each face's plane held, a height depends on its own vertex only, so J^T J's diagonal
    # blocks are formed here column by column from the residual's finite differences.
    dual, free, coords = _small_dual()
    state = _State.at(dual, coords)
    mean = np.asarray(dual.face_nodes.vertex_mean(dual.node_coords))
    slot = np.full(dual.n_nodes, -1)
    slot[free] = np.arange(free.size)

    def frozen(x):
        moved = coords.copy()
        moved[free] = x.reshape(-1, 3)
        lift = np.einsum("ij,ij->i", moved[state.node] - mean[state.face], state.normal[state.face])
        return lift / np.sqrt(state.area[state.face])

    x = coords[free].reshape(-1)
    jacobian = np.stack(
        [(frozen(x + 1e-6 * e) - frozen(x - 1e-6 * e)) / 2e-6 for e in np.eye(x.size)], axis=1
    )
    lam = 0.3
    normal = jacobian.T @ jacobian + lam * np.eye(x.size)
    pinned = np.zeros(free.size, dtype=bool)
    pinned[::5] = True
    inverse = _block_inverse(state, slot, free.size, lam, pinned)
    for k in range(free.size):
        expected = (
            np.eye(3) if pinned[k] else np.linalg.inv(normal[3 * k : 3 * k + 3, 3 * k : 3 * k + 3])
        )
        np.testing.assert_allclose(inverse[k], expected, rtol=1e-6, atol=1e-9)
