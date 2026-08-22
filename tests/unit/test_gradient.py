"""Verification of gradient reconstruction schemes against exact analytic gradients.

Physics-free: reconstruct the gradient of a known field and compare, cell-by-cell, to its
analytic gradient. Errors are measured on *interior* cells (boundary cells reconstruct at
lower order and would otherwise pollute the observed rate). This is the exact oracle that
lets the gradient — the highest-risk numerics — be de-risked before any solver exists.

`CompactGreenGauss` baseline behaviour, confirmed here:
  - orthogonal grid:  linear reconstructed exactly, smooth fields at 2nd order;
  - irregular (randomly-skewed) grid:  **inconsistent** (order ~0, error does not vanish) —
    the classic Green–Gauss deficiency that the non-orthogonal correction must fix.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.mesh import structured_grid_2d, structured_grid_3d
from aquaflux.mesh.quality import closed_cell_residual, face_planarity
from aquaflux.schemes import (
    CellBlockJacobi,
    CompactGreenGauss,
    CorrectedGreenGauss,
    ExactCellBlock,
    GmresGradientSolve,
    GradientSolve,
    GradientSystem,
    HessianCorrectedGradient,
    InverseCellVolume,
    InverseVolume,
    SweepCalibration,
    SweptGradientSolve,
    cell_diagonal_block,
    contraction_rate,
    narrow_gradient_sweeps,
)
from aquaflux.schemes import gradient as gradient_module
from aquaflux.vectors import dot, scale

from tests.support.meshes import (
    columnwise_perturbed_grid_3d,
    perturbed_grid_2d,
    perturbed_grid_3d,
)

# --- analytic fields: (value, gradient) ------------------------------------------------


def _linear(x):
    return 2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.0


def _linear_grad(x):
    return jnp.stack([2.0 * jnp.ones(x.shape[0]), -3.0 * jnp.ones(x.shape[0])], axis=1)


def _quadratic(x):
    return x[..., 0] ** 2 + x[..., 0] * x[..., 1] + x[..., 1] ** 2


def _quadratic_grad(x):
    return jnp.stack([2.0 * x[..., 0] + x[..., 1], x[..., 0] + 2.0 * x[..., 1]], axis=1)


def _trig(x):
    return jnp.sin(jnp.pi * x[..., 0]) * jnp.sin(jnp.pi * x[..., 1])


def _trig_grad(x):
    return jnp.stack(
        [
            jnp.pi * jnp.cos(jnp.pi * x[..., 0]) * jnp.sin(jnp.pi * x[..., 1]),
            jnp.pi * jnp.sin(jnp.pi * x[..., 0]) * jnp.cos(jnp.pi * x[..., 1]),
        ],
        axis=1,
    )


# --- harness ---------------------------------------------------------------------------


def _interior_mask(mesh) -> np.ndarray:
    """Cells that do not own a boundary face."""
    boundary = np.asarray(mesh.face_cells.neighbour) < 0
    boundary_cells = set(np.asarray(mesh.face_cells.owner)[boundary].tolist())
    return np.array([c not in boundary_cells for c in range(mesh.n_cells)])


def _interior_gradient_error(scheme, n, func, grad_func, perturb) -> float:
    """L2 gradient error over interior cells for an n x n structured grid."""
    mesh = perturbed_grid_2d(n, n, perturb=perturb)
    geom = mesh.geometry()
    grad = scheme.gradients(func(geom.cell.centroid), mesh, geom, func(geom.face.centroid))
    per_cell = jnp.sqrt(jnp.sum((grad - grad_func(geom.cell.centroid)) ** 2, axis=1))
    keep = _interior_mask(mesh)
    return float(jnp.sqrt(jnp.mean(per_cell[keep] ** 2)))


def _orders(errors) -> list[float]:
    return [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]


# --- tests -----------------------------------------------------------------------------


def test_compact_gg_reconstructs_linear_exactly_on_orthogonal() -> None:
    err = _interior_gradient_error(CompactGreenGauss(), 16, _linear, _linear_grad, perturb=0.0)
    assert err < 1e-12


def test_compact_gg_is_second_order_on_orthogonal() -> None:
    errs = [
        _interior_gradient_error(CompactGreenGauss(), n, _trig, _trig_grad, 0.0)
        for n in (8, 16, 32)
    ]
    assert min(_orders(errs)) > 1.8


def test_compact_gg_is_inconsistent_on_irregular_grids() -> None:
    """The known Green–Gauss deficiency: the error does not vanish under refinement."""
    errs = [
        _interior_gradient_error(CompactGreenGauss(), n, _linear, _linear_grad, 0.2)
        for n in (8, 16, 32)
    ]
    assert errs[-1] > 0.1  # still large at the finest resolution
    assert errs[-1] / errs[0] > 0.5  # barely decreased — order ~ 0


def test_compact_gg_is_differentiable() -> None:
    """`jax.grad` flows through the reconstruction without NaNs."""
    mesh = perturbed_grid_2d(8, 8, perturb=0.2)
    geom = mesh.geometry()
    scheme = CompactGreenGauss()
    bvals = _trig(geom.face.centroid)

    def loss(field):
        grad = scheme.gradients(field, mesh, geom, bvals)
        return jnp.sum(grad**2)

    sens = jax.grad(loss)(_trig(geom.cell.centroid))
    assert sens.shape == (mesh.n_cells,)
    assert not bool(jnp.any(jnp.isnan(sens)))


@pytest.mark.parametrize("func,grad_func", [(_linear, _linear_grad), (_quadratic, _quadratic_grad)])
def test_compact_gg_polynomials_exact_interior_on_orthogonal(func, grad_func) -> None:
    """On a uniform orthogonal grid, compact Green–Gauss is exact for low-order polynomials."""
    err = _interior_gradient_error(CompactGreenGauss(), 16, func, grad_func, perturb=0.0)
    assert err < 1e-12


# --- corrected Green–Gauss (the non-orthogonal correction) -----------------------------


def test_corrected_gg_reconstructs_linear_exactly_on_irregular() -> None:
    """The fix: corrected Green–Gauss is linear-exact even on irregular grids, where compact
    Green–Gauss is inconsistent.

    This asserts a machine-precision property of the *discretization*, so it pins the exact
    :class:`GmresGradientSolve`; the default :class:`SweptGradientSolve` (fixed 4 sweeps) reaches this
    only to within its sweep residual on an irregular mesh (its own accuracy is tested separately)."""
    scheme = CorrectedGreenGauss(solver=GmresGradientSolve())
    err = _interior_gradient_error(scheme, 16, _linear, _linear_grad, perturb=0.2)
    assert err < 1e-10


def test_corrected_gg_reduces_to_compact_on_orthogonal() -> None:
    """On an orthogonal grid the skewness offset is zero, so it matches compact Green–Gauss."""
    mesh = structured_grid_2d(12, 12)
    geom = mesh.geometry()
    phi = _trig(geom.cell.centroid)
    bvals = _trig(geom.face.centroid)
    corrected = CorrectedGreenGauss().gradients(phi, mesh, geom, bvals)
    compact = CompactGreenGauss().gradients(phi, mesh, geom, bvals)
    assert jnp.allclose(corrected, compact, atol=1e-9)


def test_corrected_gg_is_consistent_on_irregular() -> None:
    """Consistency restored: on irregular grids the error converges (compact's was order ~0),
    though capped near 1st order (the accuracy ceiling that motivates the implicit gradient)."""
    errs = [
        _interior_gradient_error(CorrectedGreenGauss(), n, _quadratic, _quadratic_grad, 0.2)
        for n in (8, 16, 32)
    ]
    assert errs[0] > errs[1] > errs[2]  # monotonically decreasing
    assert errs[0] / errs[-1] > 2.5  # clearly converging, unlike compact Green–Gauss
    assert min(_orders(errs)) > 0.8


def test_corrected_gg_is_differentiable() -> None:
    """`jax.grad` flows through the implicit (lineax) solve without NaNs."""
    mesh = perturbed_grid_2d(8, 8, perturb=0.2)
    geom = mesh.geometry()
    scheme = CorrectedGreenGauss()
    bvals = _trig(geom.face.centroid)

    def loss(field):
        return jnp.sum(scheme.gradients(field, mesh, geom, bvals) ** 2)

    sens = jax.grad(loss)(_trig(geom.cell.centroid))
    assert sens.shape == (mesh.n_cells,)
    assert not bool(jnp.any(jnp.isnan(sens)))


# --- swept corrected Green–Gauss (fixed matrix-free Richardson sweeps) ------------------


def test_swept_matches_corrected_at_sufficient_sweeps() -> None:
    """Enough preconditioned-Richardson sweeps reproduce the exact corrected gradient — the
    fixed-depth solve is the same reconstruction, just applied matrix-free."""
    mesh = perturbed_grid_2d(16, 16, perturb=0.2)
    geom = mesh.geometry()
    phi = _trig(geom.cell.centroid)
    bvals = _trig(geom.face.centroid)
    # The exact reference is the GMRES solve (the default is now the fixed-sweep swept solver).
    exact = CorrectedGreenGauss(solver=GmresGradientSolve()).gradients(phi, mesh, geom, bvals)
    swept = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=20)).gradients(
        phi, mesh, geom, bvals
    )
    assert jnp.allclose(swept, exact, atol=1e-11)


def test_swept_convergence_is_mesh_independent() -> None:
    """The scalability property: at a fixed sweep count the error is the same on a coarse and a
    fine mesh, so the iteration count (hence O(n) cost) does not grow with refinement."""

    def err(n):
        mesh = perturbed_grid_2d(n, n, perturb=0.2)
        geom = mesh.geometry()
        phi = _trig(geom.cell.centroid)
        bvals = _trig(geom.face.centroid)
        exact = CorrectedGreenGauss(solver=GmresGradientSolve()).gradients(phi, mesh, geom, bvals)
        swept = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=12)).gradients(
            phi, mesh, geom, bvals
        )
        return float(jnp.max(jnp.abs(swept - exact)) / jnp.max(jnp.abs(exact)))

    coarse, fine = err(16), err(32)
    # Both partially converged at 12 sweeps, to within a small factor of each other (not growing
    # with n) — the mesh-independent rate that makes the fixed sweep count scalable.
    assert 0.2 < fine / coarse < 5.0


def test_swept_is_differentiable() -> None:
    """`jax.grad` flows through the unrolled sweeps (no implicit-diff solve) without NaNs."""
    mesh = perturbed_grid_2d(8, 8, perturb=0.2)
    geom = mesh.geometry()
    scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=12))
    bvals = _trig(geom.face.centroid)

    def loss(field):
        return jnp.sum(scheme.gradients(field, mesh, geom, bvals) ** 2)

    sens = jax.grad(loss)(_trig(geom.cell.centroid))
    assert sens.shape == (mesh.n_cells,)
    assert not bool(jnp.any(jnp.isnan(sens)))


def test_swept_default_sweeps_is_four() -> None:
    """The default sweep count is the validated, cheap 4 — the well-conditioned A_g converges in a
    few sweeps, so the earlier 16 was over-provisioned."""
    assert SweptGradientSolve().sweeps == 4


def test_swept_warns_once_when_underresolved() -> None:
    """The free residual check warns (a single host-side message) only when the sweeps are
    under-resolved; a converged solve, or ``warn_tol=None``, is silent."""
    import warnings as _warnings

    from aquaflux.schemes import gradient as _gradient

    mesh = perturbed_grid_2d(12, 12, perturb=0.2)
    geom = mesh.geometry()
    phi = _trig(geom.cell.centroid)
    bvals = _trig(geom.face.centroid)

    def warnings_emitted(warn_tol) -> int:
        _gradient._GRADIENT_UNCONVERGED_WARNED = False  # reset the once-per-process guard
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=4, warn_tol=warn_tol))
            jax.block_until_ready(scheme.gradients(phi, mesh, geom, bvals))
        return sum("SweptGradientSolve" in str(w.message) for w in caught)

    assert warnings_emitted(1e-12) == 1  # unreachable tolerance -> exactly one warning
    assert warnings_emitted(5e-2) == 0  # converged at 4 sweeps -> silent (the default)
    assert warnings_emitted(None) == 0  # check disabled -> silent


def test_swept_operator_hook_is_applied_before_each_apply() -> None:
    """``operator_hook`` transforms the iterate before every operator apply — the seam a
    domain-decomposed solve uses to refresh ghost rows each sweep. Identity is a no-op; a zeroing
    hook makes the operator see 0 on every sweep, so the unit-volume Richardson update accumulates
    the right-hand side once per sweep."""
    solver = SweptGradientSolve(sweeps=3, warn_tol=None)
    unit = InverseVolume(jnp.ones(4))
    rhs = jnp.arange(1.0, 5.0)

    def operator(v):
        return 3.0 * v  # a diagonal the bare (un-hooked) iteration would diverge on

    plain = solver.solve(unit, operator, rhs)
    identity = solver.solve(unit, operator, rhs, operator_hook=lambda x: x)
    assert jnp.allclose(plain, identity)  # an identity hook changes nothing

    zeroed = solver.solve(unit, operator, rhs, operator_hook=lambda x: jnp.zeros_like(x))
    assert jnp.allclose(
        zeroed, solver.sweeps * rhs
    )  # operator sees 0 -> x += V^{-1} rhs each sweep


def test_gmres_gradient_solve_refuses_distributed_operator_hook() -> None:
    """GMRES forms whole-vector inner products, so it cannot honour a per-apply ghost exchange and
    must raise rather than silently return a wrong owned gradient."""
    solver = GmresGradientSolve()
    with pytest.raises(NotImplementedError, match="SweptGradientSolve"):
        solver.solve(
            InverseVolume(jnp.ones(3)), lambda v: v, jnp.arange(3.0), operator_hook=lambda x: x
        )


def test_hessian_gradient_refuses_distributed_operator_hook() -> None:
    """The Hessian-corrected gradient's nested Schur/A_HH solves read ghost data the outer exchange
    does not refresh, so it refuses a distributed ``operator_hook``."""
    mesh = perturbed_grid_2d(4, 4, perturb=0.1)
    geom = mesh.geometry()
    phi = _trig(geom.cell.centroid)
    bvals = _trig(geom.face.centroid)
    with pytest.raises(NotImplementedError, match="domain-decomposed"):
        HessianCorrectedGradient().gradients(phi, mesh, geom, bvals, operator_hook=lambda x: x)


# --- Hessian-corrected gradient (Hessian Schur-eliminated) -----------------------------


def test_hessian_reconstructs_quadratic_exactly_on_irregular() -> None:
    """The Hessian-corrected scheme captures the exact 2nd derivative, so quadratics are exact on any mesh —
    where compact GG is inconsistent and corrected GG is only ~1st order."""
    err = _interior_gradient_error(
        HessianCorrectedGradient(), 16, _quadratic, _quadratic_grad, perturb=0.2
    )
    assert err < 1e-10


def test_hessian_is_second_order_on_irregular() -> None:
    """Full-range order on a smooth (trig) field is ~2 — removing corrected GG's ~1st-order cap.
    (Measured over the full 8->32 range: per-step orders are noisy because each random mesh is
    an independent realization, but the full-range slope is robust.)"""
    e_coarse = _interior_gradient_error(HessianCorrectedGradient(), 8, _trig, _trig_grad, 0.2)
    e_fine = _interior_gradient_error(HessianCorrectedGradient(), 32, _trig, _trig_grad, 0.2)
    order = float(np.log2(e_coarse / e_fine) / np.log2(32 / 8))
    assert order > 1.8


def test_hessian_schur_matches_coupled_solve() -> None:
    """Schur-eliminating the Hessian gives the identical gradient to the full [g, H] solve —
    the elimination is exact."""
    mesh = perturbed_grid_2d(16, 16, perturb=0.2)
    geom = mesh.geometry()
    phi = _trig(geom.cell.centroid)
    bvals = _trig(geom.face.centroid)
    # Both paths pinned to an exact solve: the claim is that ELIMINATING the Hessian changes nothing,
    # which is a property of the discretization, not of how either system is inverted. The coupled
    # path is preconditioned by the cell volume, and that system's per-cell block couples the gradient
    # to the Hessian at leading order — so a fixed sweep does not converge on it, and comparing a
    # converged Schur solve against an unconverged coupled one would measure the solver.
    exact = dict(solver=GmresGradientSolve(), hessian_solver=GmresGradientSolve())
    schur = HessianCorrectedGradient(schur=True, **exact).gradients(phi, mesh, geom, bvals)
    coupled = HessianCorrectedGradient(schur=False, **exact).gradients(phi, mesh, geom, bvals)
    assert jnp.allclose(schur, coupled, atol=1e-10)


def test_hessian_accepts_an_injected_solve_strategy() -> None:
    """The linear solve is an injected `GradientSolve`, exactly as in `CorrectedGreenGauss`: a fixed
    sweep and a Krylov solve reach the same reconstruction, so the strategy is orthogonal to the
    discretization.

    The **outer** default is a fixed sweep, because a Krylov solve is differentiated by the implicit
    function theorem — every Jacobian-vector product then solves a second Schur system — where an
    unrolled sweep is not. The Krylov path stays available and is what the exactness tests pin.
    """
    mesh = perturbed_grid_2d(16, 16, perturb=0.2)
    geom = mesh.geometry()
    phi = _quadratic(geom.cell.centroid)
    bvals = _quadratic(geom.face.centroid)
    default = HessianCorrectedGradient().solver
    assert isinstance(default, SweptGradientSolve) and default.sweeps == 20
    gmres = HessianCorrectedGradient(solver=GmresGradientSolve()).gradients(phi, mesh, geom, bvals)
    swept = HessianCorrectedGradient(solver=SweptGradientSolve(sweeps=80, warn_tol=None)).gradients(
        phi, mesh, geom, bvals
    )
    assert jnp.allclose(gmres, swept, atol=1e-8)


def test_hessian_beats_compact_and_corrected_on_irregular() -> None:
    """The three schemes separate cleanly on the same irregular quadratic field."""
    mesh = perturbed_grid_2d(16, 16, perturb=0.2)
    geom = mesh.geometry()
    phi = _quadratic(geom.cell.centroid)
    bvals = _quadratic(geom.face.centroid)
    exact = _quadratic_grad(geom.cell.centroid)

    def rms(scheme):
        grad = scheme.gradients(phi, mesh, geom, bvals)
        return float(jnp.sqrt(jnp.mean(jnp.sum((grad - exact) ** 2, axis=1))))

    compact = rms(CompactGreenGauss())
    corrected = rms(CorrectedGreenGauss())
    hessian = rms(HessianCorrectedGradient())
    assert hessian < corrected < compact
    assert compact > 0.05  # compact is inconsistent
    assert hessian < 1e-10  # hessian is exact for quadratics


def test_hessian_is_differentiable() -> None:
    """`jax.grad` flows through the nested Schur solve without NaNs."""
    mesh = perturbed_grid_2d(8, 8, perturb=0.2)
    geom = mesh.geometry()
    scheme = HessianCorrectedGradient()
    bvals = _trig(geom.face.centroid)

    def loss(field):
        return jnp.sum(scheme.gradients(field, mesh, geom, bvals) ** 2)

    sens = jax.grad(loss)(_trig(geom.cell.centroid))
    assert sens.shape == (mesh.n_cells,)
    assert not bool(jnp.any(jnp.isnan(sens)))


# --- Hessian-corrected gradient in 3D (planar-faced skewed hex mesh) --------------------


def _quad_3d(x):
    return (
        x[..., 0] ** 2
        + 2.0 * x[..., 1] ** 2
        + 3.0 * x[..., 2] ** 2
        + x[..., 0] * x[..., 1]
        + x[..., 1] * x[..., 2]
    )


def _quad_3d_grad(x):
    return jnp.stack(
        [
            2.0 * x[..., 0] + x[..., 1],
            4.0 * x[..., 1] + x[..., 0] + x[..., 2],
            6.0 * x[..., 2] + x[..., 1],
        ],
        axis=1,
    )


def _interior_grad_error_3d(scheme, mesh, func, grad_func) -> float:
    geom = mesh.geometry()
    grad = scheme.gradients(func(geom.cell.centroid), mesh, geom, func(geom.face.centroid))
    per_cell = jnp.abs(grad - grad_func(geom.cell.centroid))
    return float(jnp.max(per_cell[_interior_mask(mesh)]))


def test_hessian_reconstructs_quadratic_exactly_in_3d() -> None:
    """The 3D reconstruction (dimension-general, Betchen Eq. 7) is exact for quadratics on a
    genuinely skewed hex mesh — where CorrectedGreenGauss, being only linear-exact, is not. The
    grid is skewed *in-plane* so its faces stay planar (the Green–Gauss face integral is then exact
    for a quadratic; a warped-face grid would break that for every Green–Gauss scheme)."""
    mesh = columnwise_perturbed_grid_3d(6, 6, 6, perturb=0.25, seed=1)
    assert float(jnp.min(face_planarity(mesh))) > 1.0 - 1e-9  # planar faces by construction
    hessian_err = _interior_grad_error_3d(HessianCorrectedGradient(), mesh, _quad_3d, _quad_3d_grad)
    corrected_err = _interior_grad_error_3d(CorrectedGreenGauss(), mesh, _quad_3d, _quad_3d_grad)
    assert hessian_err < 1e-10  # exact for quadratics in 3D
    assert corrected_err > 1e-3  # the mesh is genuinely skewed; corrected Green–Gauss is not exact


def test_hessian_3d_is_differentiable() -> None:
    """`jax.grad` flows through the 3D nested Schur solve without NaNs."""
    mesh = columnwise_perturbed_grid_3d(4, 4, 4, perturb=0.2, seed=2)
    geom = mesh.geometry()
    scheme = HessianCorrectedGradient()
    bvals = _quad_3d(geom.face.centroid)

    def loss(field):
        return jnp.sum(scheme.gradients(field, mesh, geom, bvals) ** 2)

    sens = jax.grad(loss)(_quad_3d(geom.cell.centroid))
    assert sens.shape == (mesh.n_cells,)
    assert not bool(jnp.any(jnp.isnan(sens)))


# --- narrowing the sweep count, and what it does to the Jacobian's reach ----------------


def _cell_graph_distance(mesh) -> np.ndarray:
    """All-pairs graph distance over the interior-face cell graph (breadth-first per source)."""
    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)
    adjacency: list[list[int]] = [[] for _ in range(mesh.n_cells)]
    for a, b in zip(owner, nb, strict=True):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    distance = np.full((mesh.n_cells, mesh.n_cells), mesh.n_cells, dtype=int)
    for source in range(mesh.n_cells):
        distance[source, source] = 0
        frontier, step = [source], 0
        while frontier:
            step += 1
            reached = []
            for u in frontier:
                for v in adjacency[u]:
                    if distance[source, v] > step:
                        distance[source, v] = step
                        reached.append(v)
            frontier = reached
    return distance


def test_narrow_gradient_sweeps_only_ever_narrows() -> None:
    """A cap at or above a solve's own count leaves it alone, and returns the tree by identity."""
    scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=4, warn_tol=None))
    assert narrow_gradient_sweeps(scheme, 2).solver.sweeps == 2
    assert (
        narrow_gradient_sweeps(scheme, 2).solver.warn_tol is None
    )  # the rest of the solve survives
    assert narrow_gradient_sweeps(scheme, 4) is scheme
    assert narrow_gradient_sweeps(scheme, 8) is scheme


def test_narrow_gradient_sweeps_leaves_other_solves_untouched() -> None:
    """Only the swept solve has a sweep count; a Krylov or one-shot reconstruction is returned as is."""
    gmres = CorrectedGreenGauss(solver=GmresGradientSolve())
    compact = CompactGreenGauss()
    assert narrow_gradient_sweeps(gmres, 1) is gmres
    assert narrow_gradient_sweeps(compact, 1) is compact


def test_narrow_gradient_sweeps_reaches_into_a_tree() -> None:
    """It rewrites every swept solve it finds, through Module fields and through sequences."""
    nested = (CorrectedGreenGauss(), CompactGreenGauss())
    narrowed = narrow_gradient_sweeps(nested, 1)
    assert narrowed[0].solver.sweeps == 1
    assert isinstance(narrowed[1], CompactGreenGauss)


def test_narrow_gradient_sweeps_rejects_a_count_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        narrow_gradient_sweeps(CorrectedGreenGauss(), 0)


@pytest.mark.parametrize("sweeps", [1, 2, 3, 4])
def test_each_sweep_couples_one_further_ring_on_a_skewed_mesh(sweeps: int) -> None:
    """The property the cap exists to control: the reconstruction reaches exactly ``sweeps`` cells.

    Each Richardson sweep applies the correction operator once, and that operator couples a cell to
    its face neighbours — so the reconstruction widens by one ring per sweep wherever the skewness
    correction is live. Measured here on the reconstruction alone (no physics involved): the graph
    distance of the furthest nonzero of ``d(gradient)/d(field)``. A residual built on it is one ring
    wider again, since a face flux reads the gradient of the cells on both sides.
    """
    mesh = perturbed_grid_2d(6, 6, perturb=0.25, seed=1)
    geom = mesh.geometry()
    scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=sweeps, warn_tol=None))
    bvals = _linear(geom.face.centroid)

    def reconstruct(field):
        return scheme.gradients(field, mesh, geom, bvals)[:, 0]

    jacobian = np.abs(np.asarray(jax.jacfwd(reconstruct)(_linear(geom.cell.centroid))))
    live = jacobian > 1e-13 * jacobian.max()
    assert int(_cell_graph_distance(mesh)[live].max()) == sweeps


def test_a_narrowed_reconstruction_costs_nothing_on_an_orthogonal_mesh() -> None:
    """Where the mesh is orthogonal the correction vanishes, so the extra sweeps have nothing to add.

    Agreement is to round-off rather than bit-exact: the sweeps converge immediately, and each further
    one re-applies an update that is zero only up to the floating-point error of forming it.
    """
    mesh = structured_grid_2d(6, 6)
    geom = mesh.geometry()
    field, bvals = _linear(geom.cell.centroid), _linear(geom.face.centroid)
    full = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=4, warn_tol=None))
    narrowed = narrow_gradient_sweeps(full, 2)
    assert narrowed.solver.sweeps == 2  # the arms really do differ
    np.testing.assert_allclose(
        np.asarray(narrowed.gradients(field, mesh, geom, bvals)),
        np.asarray(full.gradients(field, mesh, geom, bvals)),
        rtol=0,
        atol=1e-14,
    )


def test_the_underresolved_warning_does_not_fire_at_one_sweep() -> None:
    """At one sweep the check measures ``rhs / rhs`` — exactly 1 on any mesh, so it says nothing.

    The residual the diagnostic reads is the one entering the *final* update, which at a single sweep
    is the initial ``rhs - A·0``. Left ungated it warned that the sweeps were under-resolved on a
    perfectly orthogonal grid, where the reconstruction is exact and the correction is identically
    zero. A single sweep is the uncorrected Green–Gauss reconstruction, so there is no correction
    left under-resolved to report.
    """
    mesh = structured_grid_2d(6, 6)  # exactly orthogonal: the skewness offset is identically zero
    geom = mesh.geometry()
    field, bvals = _linear(geom.cell.centroid), _linear(geom.face.centroid)

    def warnings_from(sweeps):
        gradient_module._GRADIENT_UNCONVERGED_WARNED = False
        scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=sweeps))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scheme.gradients(field, mesh, geom, bvals)
        return [str(w.message) for w in caught]

    assert warnings_from(1) == []
    assert warnings_from(4) == []  # and a resolved multi-sweep solve is silent, as it always was


def test_the_swept_solve_spends_one_apply_fewer_than_its_sweep_count() -> None:
    """The first sweep's operator apply is against a zero iterate, so it is peeled — exactly.

    ``sweeps`` sweeps cost ``sweeps - 1`` applies, because the iteration starts at ``x = 0`` where
    ``rhs - A·0`` is ``rhs`` outright. Nothing downstream removes that apply on its own: the compiler
    folds the gathers against the zero constant but not the scatters, so left in it costs a whole
    operator apply out of ``sweeps`` on a solve that runs inside every residual evaluation and every
    Jacobian--vector product.

    Counting applies rather than timing is what makes this a *test*: it fails if the peel is ever
    reverted or an extra apply creeps back in, where a timing assertion would only wobble.

    ⚠️ **Counted from the traced program, not with a Python counter.** The sweeps run as a
    ``lax.scan``, whose body is traced ONCE however many times it executes, so a counter incremented
    inside the operator reports ``1`` at every sweep count -- it would pass whatever the peel did,
    which is worse than no test. The trip count and the applies inside one body give the same total
    and remain visible.
    """
    unit = InverseVolume(jnp.ones(4))
    rhs = jnp.arange(1.0, 5.0)
    applies = []

    def operator(v):
        applies.append(1)
        return 0.25 * v  # contracting, so the iteration is well-posed

    for sweeps in (1, 2, 4, 7):
        applies.clear()
        solve = SweptGradientSolve(sweeps=sweeps, warn_tol=None)
        jaxpr = jax.make_jaxpr(lambda r, _s=solve: _s.solve(unit, operator, r))(rhs)
        scans = [e for e in jaxpr.eqns if e.primitive.name == "scan"]
        trips = sum(int(e.params["length"]) for e in scans)
        # One apply per body execution, and the body runs `length` times -- so the total applies are
        # `length`, which must be one fewer than the sweep count.
        assert trips == sweeps - 1, f"expected {sweeps - 1} applies, program runs {trips}"
        # One apply per body, so the trip count IS the apply count. At a single sweep there is no
        # body at all -- the peel is returned directly, which is what the operator below proves.
        assert len(applies) == len(scans), "the body should hold exactly one operator apply"

    # AND THE PEEL ITSELF, checked where it is unambiguous: at one sweep the operator must never be
    # applied at all, which an operator that raises proves outright rather than by counting.
    def forbidden(_v):
        raise AssertionError("the peeled first sweep must not apply the operator")

    SweptGradientSolve(sweeps=1, warn_tol=None).solve(unit, forbidden, rhs)


def test_peeling_the_zero_apply_leaves_the_answer_BIT_identical() -> None:
    """Exact, not merely close — the peel replaces ``rhs - A·0`` with ``rhs``, which it equals.

    Checked against the unpeeled iteration written out here rather than against a tolerance, so a
    change that alters the arithmetic (reordering the update, folding the volume differently) fails
    even though it would stay well inside any sensible tolerance.

    ⚠️ **The reference runs the same loop construct as the solve.** Both are a ``lax.scan``, so the
    only difference between the two arms is the peel, which is what this test is about. Written as a
    Python loop instead it fails by ~1 unit in the last place -- not because the peel is inexact
    (``A·0`` is exactly zero, so ``rhs - A·0`` is ``rhs`` bit for bit) but because XLA contracts the
    multiply-adds differently in an unrolled chain than in a scan body. Comparing across constructs
    would test the compiler, not the peel.
    """
    mesh = perturbed_grid_2d(12, 12, perturb=0.2)
    geometry = mesh.geometry()
    field = jnp.sin(3.0 * geometry.cell.centroid[:, 0]) * geometry.cell.centroid[:, 1]

    for sweeps in (1, 2, 4, 9):
        solver = SweptGradientSolve(sweeps=sweeps, warn_tol=None)
        captured = {}

        def capture(preconditioner, operator, rhs, *, operator_hook=None, _s=solver, _c=captured):
            # The iteration exactly as it stood before the peel, over the same operator and rhs, and
            # in the same loop construct -- so the peel is the only difference between the arms.
            def sweep(x, _):
                return x + preconditioner.apply(rhs - operator(x)), None

            _c["unpeeled"], _ = jax.lax.scan(sweep, jnp.zeros_like(rhs), None, length=_s.sweeps)
            return _s.solve(preconditioner, operator, rhs, operator_hook=operator_hook)

        peeled = CorrectedGreenGauss(
            solver=type("_Capture", (), {"solve": staticmethod(capture)})()
        ).gradients(field, mesh, geometry, jnp.zeros(mesh.n_faces))

        assert jnp.array_equal(peeled, captured["unpeeled"]), f"differs at sweeps={sweeps}"


# --------------------------------------------------------------------------------------
# Preconditioning the reconstruction systems
# --------------------------------------------------------------------------------------


def _dense(operator, shape) -> np.ndarray:
    """Dense matrix of a linear matvec on arrays of ``shape``, flattened row-major."""
    n = int(np.prod(shape))
    columns = jax.vmap(operator)(jnp.eye(n).reshape(n, *shape))
    return np.asarray(columns.reshape(n, n)).T


def _per_cell_blocks(dense: np.ndarray, n_cells: int, size: int) -> np.ndarray:
    """The ``n_cells`` diagonal blocks of a dense operator laid out cell-major."""
    return np.stack(
        [dense[i * size : (i + 1) * size, i * size : (i + 1) * size] for i in range(n_cells)]
    )


def test_inverse_volume_and_cell_block_apply_the_inverse_they_are_given() -> None:
    """Both preconditioners are exactly ``P⁻¹·r``, over whichever component rank the unknown carries.

    The Hessian unknown is rank-3 ``(n_cells, dim, dim)`` where the gradient is rank-2, and the two
    are preconditioned by the same objects — so the broadcast (``InverseVolume``) and the contraction
    (``CellBlockJacobi``) are checked against an explicit per-cell reference at both ranks.
    """
    rng = np.random.default_rng(0)
    n_cells, dim = 5, 3
    volume = jnp.asarray(rng.uniform(0.5, 2.0, n_cells))
    blocks = jnp.asarray(rng.normal(size=(n_cells, dim, dim)) + 4.0 * np.eye(dim))
    inverse = jnp.linalg.inv(blocks)

    vector = jnp.asarray(rng.normal(size=(n_cells, dim)))
    tensor = jnp.asarray(rng.normal(size=(n_cells, dim, dim)))

    scaled = InverseVolume(1.0 / volume).apply(vector)
    assert jnp.allclose(scaled, vector / volume[:, None])
    assert jnp.allclose(InverseVolume(1.0 / volume).apply(tensor), tensor / volume[:, None, None])

    # A gradient residual is a per-cell matrix-vector product with the inverse.
    got = CellBlockJacobi(inverse).apply(vector)
    want = np.stack([np.asarray(inverse)[c] @ np.asarray(vector)[c] for c in range(n_cells)])
    assert jnp.allclose(got, want)
    # A Hessian residual is that same product applied to each of its rows (the `I ⊗ C` structure).
    got = CellBlockJacobi(inverse).apply(tensor)
    want = np.stack(
        [
            np.stack([np.asarray(inverse)[c] @ np.asarray(tensor)[c, i] for i in range(dim)])
            for c in range(n_cells)
        ]
    )
    assert jnp.allclose(got, want)


def test_cell_diagonal_block_recovers_the_true_diagonal_block_exactly() -> None:
    """The sided probe is the operator's real per-cell block, not an approximation of it.

    Checked against a densely materialized ``A_g`` on a skewed mesh, so it fails if the two halves
    of the probe ever stop summing to the whole block (a missed boundary contribution, a scatter
    half attributed to the wrong side).
    """
    mesh = perturbed_grid_2d(6, 6, perturb=0.3, seed=0)
    geometry = mesh.geometry()
    terms = CorrectedGreenGauss.terms(mesh, geometry)
    face_cells = terms.face_cells
    dim, n_cells = mesh.dim, mesh.n_cells
    zero_face = jnp.zeros((mesh.n_faces, dim))

    def sided(owner_field, neighbour_field):
        """``A_g``'s face contribution with each side read from its own field (see its `operator`)."""
        w = (1.0 - terms.g) * dot(terms.skew, owner_field[face_cells.owner]) + terms.g * dot(
            terms.skew, neighbour_field[face_cells.safe_neighbour]
        )
        return face_cells.combine_face_values(scale(terms.area_vector, w), 0.0)

    block = cell_diagonal_block(
        lambda probe: face_cells.scatter(sided(probe, jnp.zeros_like(probe)), zero_face),
        lambda probe: face_cells.scatter(zero_face, -sided(jnp.zeros_like(probe), probe)),
        terms.volume,
        n_cells,
        dim,
    )
    truth = _per_cell_blocks(
        _dense(CorrectedGreenGauss.operator(terms), (n_cells, dim)), n_cells, dim
    )
    assert np.abs(np.asarray(block) - truth).max() / np.abs(truth).max() < 1e-13


def test_the_hessian_block_is_a_kronecker_product_so_it_is_stored_dim_by_dim() -> None:
    """``A_HH``'s per-cell block is exactly ``I_dim ⊗ C``, which is why one ``(dim, dim)`` matrix per
    cell suffices where the block is nominally ``(dim², dim²)``.

    This is a memory claim with teeth at scale — ``dim²×dim²`` would be nine times the storage in 3D
    — so it is checked as an exact identity rather than assumed. The Hessian enters its own equation
    only as ``H·a`` for per-face vectors ``a``, which contracts ``H``'s second index and leaves the
    first untouched; if that ever stops being true this fails.
    """
    for mesh in (
        perturbed_grid_2d(6, 6, perturb=0.3, seed=0),
        columnwise_perturbed_grid_3d(3, 3, 3, perturb=0.3, seed=0),
    ):
        geometry = mesh.geometry()
        dim, n_cells = mesh.dim, mesh.n_cells
        operator, preconditioner = _inner_system(mesh, geometry)
        kronecker_factor = np.linalg.inv(np.asarray(preconditioner.inverse))

        truth = _per_cell_blocks(_dense(operator, (n_cells, dim, dim)), n_cells, dim * dim).reshape(
            n_cells, dim, dim, dim, dim
        )
        rebuilt = np.einsum("ik,cjl->cijkl", np.eye(dim), kronecker_factor)
        assert np.abs(rebuilt - truth).max() / np.abs(truth).max() < 1e-13


class _CaptureSolve(GradientSolve):
    """A ``GradientSolve`` that records what it was handed and solves nothing.

    The scheme passes its inner operator and preconditioner to the injected ``hessian_solver``, so
    injecting this recovers exactly the pair the scheme really uses — no second copy of the formula
    to drift, and no reaching past the interface to get it.
    """

    seen: list = eqx.field(static=True, default_factory=list)

    def solve(self, preconditioner, operator, rhs, *, operator_hook=None):
        self.seen.append((operator, preconditioner))
        return jnp.zeros_like(rhs)


def _inner_system(mesh, geometry):
    """The scheme's inner ``A_HH`` operator and the preconditioner it pairs with it."""
    capture = _CaptureSolve()
    HessianCorrectedGradient(hessian_solver=capture).gradients(
        jnp.zeros(mesh.n_cells), mesh, geometry, jnp.zeros(mesh.n_faces)
    )
    return capture.seen[0]


def test_the_hessian_system_needs_a_block_preconditioner_not_the_volume() -> None:
    """The inner system is what forced a Krylov solve, and the per-cell block is why it no longer
    does — measured as the Richardson contraction rate on the actual operator.

    The decisive part is the **orthogonal** mesh: there the skewness coupling is identically zero, so
    the inverse-volume iteration's poor rate cannot be blamed on mesh quality. It is the gradient and
    Hessian coupling to each other inside a cell, which ``1/V`` cannot represent at all and the
    per-cell block represents exactly.
    """
    for perturb, block_bound in ((0.0, 1e-12), (0.3, 0.25)):
        mesh = perturbed_grid_2d(8, 8, perturb=perturb, seed=0)
        geometry = mesh.geometry()
        dim, n_cells = mesh.dim, mesh.n_cells
        operator, _ = _inner_system(mesh, geometry)
        dense = _dense(operator, (n_cells, dim, dim))
        size = dim * dim
        identity = np.eye(dense.shape[0])

        volume = np.repeat(np.asarray(geometry.cell.volume), size)
        rate_volume = np.abs(
            np.linalg.eigvals(identity - np.linalg.solve(np.diag(volume), dense))
        ).max()
        blocks = np.zeros_like(dense)
        for c, block in enumerate(_per_cell_blocks(dense, n_cells, size)):
            blocks[c * size : (c + 1) * size, c * size : (c + 1) * size] = block
        rate_block = np.abs(np.linalg.eigvals(identity - np.linalg.solve(blocks, dense))).max()

        assert rate_volume > 0.45, f"inverse-volume rate {rate_volume} at perturb={perturb}"
        assert rate_block < block_bound, f"block rate {rate_block} at perturb={perturb}"


def test_the_default_inner_sweeps_reach_the_exactly_solved_reconstruction() -> None:
    """The shipped inner sweep count is enough to keep the scheme's defining property — exactness for
    a quadratic — rather than trading it for speed.

    Pinned against the exactly-solved inner system on the same mesh, so it fails if the default sweep
    count is lowered without the accuracy consequence being faced.
    """
    mesh = columnwise_perturbed_grid_3d(5, 5, 5, perturb=0.25, seed=1)
    geometry = mesh.geometry()
    field = _quad_3d(geometry.cell.centroid)
    bvals = _quad_3d(geometry.face.centroid)

    default = HessianCorrectedGradient().gradients(field, mesh, geometry, bvals)
    exact = HessianCorrectedGradient(hessian_solver=GmresGradientSolve()).gradients(
        field, mesh, geometry, bvals
    )
    assert float(jnp.max(jnp.abs(default - exact))) < 1e-11


def test_a_swept_outer_solve_reaches_the_same_gradient_as_the_krylov_one() -> None:
    """The outer Schur system is well enough conditioned to be swept rather than Krylov-solved, which
    is what removes the last inner product (and the last implicit-diff tangent) from the scheme.

    Both outer strategies solve the same system, so given enough sweeps they must agree to machine
    precision; if they do not, the block-preconditioned outer iteration is not converging to ``S⁻¹``.
    """
    mesh = perturbed_grid_2d(8, 8, perturb=0.3, seed=0)
    geometry = mesh.geometry()
    field = _quadratic(geometry.cell.centroid)
    bvals = _quadratic(geometry.face.centroid)

    # Named explicitly rather than taken from the default: this asserts that the two OUTER strategies
    # agree, so it must keep comparing those two whatever the default later becomes.
    krylov = HessianCorrectedGradient(solver=GmresGradientSolve()).gradients(
        field, mesh, geometry, bvals
    )
    swept = HessianCorrectedGradient(solver=SweptGradientSolve(sweeps=60, warn_tol=None)).gradients(
        field, mesh, geometry, bvals
    )
    assert float(jnp.max(jnp.abs(krylov - swept))) < 1e-12


def test_a_diagnostic_emitting_inner_solver_is_refused_with_an_explanation() -> None:
    """The inner solver sits inside an operator the outer Krylov solve transposes, and a convergence
    diagnostic is nonlinear — so the combination cannot work.

    Left alone it fails with a bare ``AssertionError`` from inside the linear solver, which says
    nothing about the cause; this pins the explanatory error, and pins that the *same* inner solver
    is fine under a fixed-sweep outer solve, which transposes nothing.
    """
    mesh = perturbed_grid_2d(4, 4, perturb=0.2)
    geometry = mesh.geometry()
    field = _quadratic(geometry.cell.centroid)
    bvals = _quadratic(geometry.face.centroid)
    chatty = SweptGradientSolve(sweeps=6, warn_tol=5e-2)

    # A Krylov OUTER solver is what makes this invalid — it forms its tangent by transposing the
    # operator the inner solver sits inside. The default outer is a fixed sweep, which transposes
    # nothing, so the condition has to be asked for explicitly rather than inherited.
    with pytest.raises(ValueError, match="strictly linear"):
        HessianCorrectedGradient(solver=GmresGradientSolve(), hessian_solver=chatty).gradients(
            field, mesh, geometry, bvals
        )

    swept_outer = HessianCorrectedGradient(
        solver=SweptGradientSolve(sweeps=40, warn_tol=None), hessian_solver=chatty
    ).gradients(field, mesh, geometry, bvals)
    assert not bool(jnp.any(jnp.isnan(swept_outer)))


# --- build-time sweep calibration ------------------------------------------------------
#
# The sweep count is a mesh property spanning some eighty-fold across the meshes this solver runs
# on, and a fixed count carries no convergence test — so being short of one is silent. These pin the
# estimator that measures it, on synthetic operators whose rate is known in closed form, and the
# factories that turn it into a scheme.


def _known_rate_system(rate: float, n: int = 12, spread: float = 3.0) -> GradientSystem:
    """A system whose preconditioned Richardson iteration contracts at exactly ``rate``.

    ``P⁻¹A = I - M`` with ``M`` diagonal, its largest entry ``rate`` and the rest geometrically
    smaller, so ``rho(I - P⁻¹A) = rate`` exactly and the estimate has a closed-form answer to be
    checked against. Written through the real :class:`InverseVolume` preconditioner (volumes of one,
    so it is the identity) rather than a stub, so the estimator is exercised on the interface it
    consumes in production.
    """
    diagonal = rate / spread ** jnp.arange(n, dtype=jnp.float64)

    def operator(v: jnp.ndarray) -> jnp.ndarray:
        return v - diagonal[:, None] * v

    return GradientSystem(InverseVolume(jnp.ones(n)), operator, (n, 2))


@pytest.mark.parametrize("rate", [0.9, 0.5, 0.25, 0.05, 1e-6])
def test_the_estimator_recovers_a_known_contraction_rate(rate: float) -> None:
    """On an operator whose rate is known exactly, the estimate must return it — and from below.

    The Gelfand form takes the budget-th root of the accumulated growth, which carries the starting
    vector's deficiency in the dominant eigendirection: a random start contributes a factor
    ``|c|^(1/iters)`` slightly under one, so the estimate rises toward the rate rather than
    overshooting it. That direction is asserted here because the sweep count is built on it — an
    estimate that could sit *above* the rate would under-count sweeps, while one below over-counts,
    and only the first is a silent loss of accuracy.
    """
    measured = contraction_rate(_known_rate_system(rate))
    assert measured.rate <= rate * (1.0 + 1e-9), "the estimate must not exceed the true rate"
    assert measured.rate >= 0.85 * rate, "...nor fall so far short that the count is wrong"


def test_the_estimator_survives_a_complex_dominant_pair() -> None:
    """A rotation-like operator has no dominant *real* eigenvalue, and the root form is why that is fine.

    ``M`` here rotates one plane by a quarter turn while scaling by ``rate``, so its dominant
    eigenvalues are a complex-conjugate pair. The ratio of successive norms oscillates on such an
    operator indefinitely and never settles; the budget-th root of the accumulated growth averages
    the oscillation out, which is the reason the estimate is taken that way rather than as a ratio.
    """
    rate = 0.4

    def operator(v: jnp.ndarray) -> jnp.ndarray:
        # M v = rate * (rotate v by a quarter turn in the component plane); A = I - M.
        return v - rate * jnp.stack([-v[:, 1], v[:, 0]], axis=1)

    measured = contraction_rate(GradientSystem(InverseVolume(jnp.ones(8)), operator, (8, 2)))
    assert measured.rate == pytest.approx(rate, rel=0.05)


def test_the_settledness_check_reports_a_short_budget() -> None:
    """The half-budget estimate is the self-check, and it has to move when the budget is too short.

    The estimate rises toward the true rate, so a budget that has not settled reports a ratio well
    above one. A short budget must show that and the default budget must not — a check that reads
    the same either way would report nothing.
    """
    system = _known_rate_system(0.6)
    short = contraction_rate(system, iters=2)
    settled = contraction_rate(system, iters=24)

    assert short.settling_ratio > settled.settling_ratio
    assert settled.settling_ratio == pytest.approx(1.0, abs=0.2)
    assert settled.rate > short.rate, "the estimate approaches the true rate from below"


def test_an_annihilating_iteration_reports_a_rate_of_zero_not_one() -> None:
    """An exactly-solved system has rate zero, and reporting one would cap the sweep count.

    Where the skewness correction vanishes the preconditioner is the exact inverse, so the iteration
    sends its probe to zero outright. Renormalizing that by one would leave every further step
    contributing no growth at all and report a *perfect* mesh as a non-converging one — the worst
    possible direction for the error to run in.
    """
    system = GradientSystem(InverseVolume(jnp.ones(6)), lambda v: v, (6, 2))
    assert contraction_rate(system).rate < 1e-100
    assert SweepCalibration().sweeps(system) == 1


def test_calibration_refuses_traced_geometry_by_name() -> None:
    """Under a trace the count cannot be an int, and the failure has to say so where it happened.

    ``sweeps`` is static configuration, so a tracer can never become one. Left to itself the tracer
    surfaces as a concretization error from inside a logarithm several frames down, naming neither
    the geometry nor the way out.
    """
    mesh = perturbed_grid_2d(4, 4, perturb=0.2)

    def calibrate(volumes: jnp.ndarray) -> int:
        geometry = mesh.geometry()
        scheme = CorrectedGreenGauss()
        terms = scheme.terms(mesh, geometry)
        system = GradientSystem(
            InverseVolume(1.0 / volumes), scheme.operator(terms), (mesh.n_cells, mesh.dim)
        )
        return SweepCalibration().sweeps(system)

    with pytest.raises(ValueError, match="geometry is traced"):
        jax.jit(calibrate)(mesh.geometry().cell.volume)


def test_the_floor_and_cap_bound_the_calibrated_count() -> None:
    """The bounds have to hold at both ends, including where the rate formula gives no answer.

    A rate at or above one does not converge at any count and must return the cap rather than a
    negative or infinite one; a rate of zero must return the floor rather than zero sweeps, since a
    solve runs at least once.
    """
    calibration = SweepCalibration(tol=1e-4, floor=3, cap=8)

    assert calibration.sweeps_for(0.0) == 3
    assert calibration.sweeps_for(1e-30) == 3, "one sweep would do; the floor overrides it"
    assert calibration.sweeps_for(0.5) == 8, "13 sweeps would be needed; the cap holds"
    assert calibration.sweeps_for(1.0) == 8, "a non-converging iteration returns the cap"
    assert calibration.sweeps_for(2.0) == 8
    assert SweepCalibration(tol=1e-4).sweeps_for(0.5) == 14


def test_the_calibrated_count_is_the_smallest_reaching_the_tolerance() -> None:
    """The count is ``ceil(log tol / log rate)`` and must be tight — one sweep fewer must miss."""
    calibration = SweepCalibration(tol=1e-6)
    for rate in (0.1, 0.3, 0.55, 0.8):
        k = calibration.sweeps_for(rate)
        # `ceil` lands exactly on the tolerance where the logarithms divide evenly, and `0.1 ** 6`
        # is 1e-6 only to within a rounding error, so the reach is compared with that slack.
        assert rate**k <= calibration.tol * (1.0 + 1e-9)
        assert rate ** (k - 1) > calibration.tol, "a smaller count would also have done"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tol": 0.0}, "tol must be"),
        ({"tol": 1.0}, "tol must be"),
        ({"iters": 1}, "iters must be"),
        ({"floor": 0}, "floor must be"),
        ({"cap": 2, "floor": 4}, "must be at least floor"),
    ],
)
def test_an_inconsistent_calibration_cannot_be_constructed(kwargs: dict, match: str) -> None:
    """Rejected at construction, so no path through it can reach the count with bad settings."""
    with pytest.raises(ValueError, match=match):
        SweepCalibration(**kwargs)


def test_the_estimator_needs_a_budget_it_can_halve() -> None:
    with pytest.raises(ValueError, match="iters >= 2"):
        contraction_rate(_known_rate_system(0.5), iters=1)


def test_calibration_gives_an_orthogonal_mesh_a_single_sweep() -> None:
    """Where the correction vanishes the system is diagonal and one sweep is already exact.

    This is the over-resolved end of the range the fixed count cannot cover: four sweeps here spend
    three operator applies to refine an answer that was exact after the first.
    """
    mesh = structured_grid_2d(6, 6, 1.0, 1.0)
    scheme = CorrectedGreenGauss.calibrated(mesh, mesh.geometry())

    assert scheme.solver.sweeps == 1


@pytest.mark.parametrize("perturb, tol", [(0.2, 1e-4), (0.3, 1e-4), (0.2, 1e-8)])
def test_a_calibrated_reconstruction_reaches_the_tolerance_it_promises(
    perturb: float, tol: float
) -> None:
    """The whole point: the count returned must actually deliver the accuracy asked for.

    Measured against the same system solved exactly, in the L2 norm over all cells, which is the
    norm the tolerance is stated in. The estimate approaches the rate from below, which would
    under-count, while the iteration's early transient reduces the error faster than the asymptotic
    rate, which over-delivers — this asserts the second is at least as large as the first.
    """
    mesh = perturbed_grid_2d(14, 14, perturb=perturb, seed=3)
    geometry = mesh.geometry()
    field = _trig(geometry.cell.centroid)
    bvals = _trig(geometry.face.centroid)

    calibrated = CorrectedGreenGauss.calibrated(mesh, geometry, tol=tol)
    exact = CorrectedGreenGauss(solver=GmresGradientSolve()).gradients(field, mesh, geometry, bvals)
    got = calibrated.gradients(field, mesh, geometry, bvals)

    error = float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact))
    assert error <= tol
    # ...and the count is not simply enormous: one sweep fewer should be the one that misses.
    fewer = CorrectedGreenGauss(
        solver=SweptGradientSolve(sweeps=calibrated.solver.sweeps - 1, warn_tol=None)
    ).gradients(field, mesh, geometry, bvals)
    assert float(jnp.linalg.norm(fewer - exact) / jnp.linalg.norm(exact)) > 0.1 * tol


def test_a_skewer_mesh_calibrates_to_more_sweeps() -> None:
    """The count has to track the mesh, which is the property a fixed count cannot have."""
    counts = [
        CorrectedGreenGauss.calibrated(mesh, mesh.geometry()).solver.sweeps
        for mesh in (
            structured_grid_2d(12, 12, 1.0, 1.0),
            perturbed_grid_2d(12, 12, perturb=0.1, seed=3),
            perturbed_grid_2d(12, 12, perturb=0.3, seed=3),
        )
    ]
    assert counts == sorted(counts) and counts[0] < counts[-1]


def test_a_tighter_tolerance_calibrates_to_more_sweeps() -> None:
    mesh = perturbed_grid_2d(10, 10, perturb=0.25, seed=1)
    geometry = mesh.geometry()
    loose = CorrectedGreenGauss.calibrated(mesh, geometry, tol=1e-3).solver.sweeps
    tight = CorrectedGreenGauss.calibrated(mesh, geometry, tol=1e-10).solver.sweeps
    assert loose < tight


def test_calibrating_the_hessian_scheme_sizes_both_of_its_systems() -> None:
    """Both systems are measured, and the inner one keeps the diagnostic its position requires.

    The inner solve runs inside the outer operator, so its convergence diagnostic would make that
    operator nonlinear — which the scheme rejects outright when an outer Krylov solve would have to
    transpose it. A calibrated scheme must not walk into that.
    """
    mesh = perturbed_grid_2d(8, 8, perturb=0.25, seed=3)
    geometry = mesh.geometry()
    scheme = HessianCorrectedGradient.calibrated(mesh, geometry)

    assert scheme.solver.sweeps >= 1 and scheme.hessian_solver.sweeps >= 1
    assert scheme.hessian_solver.warn_tol is None
    assert (
        not scheme.solver.emits_host_diagnostics or not scheme.hessian_solver.emits_host_diagnostics
    )

    field = _quadratic(geometry.cell.centroid)
    bvals = _quadratic(geometry.face.centroid)
    exact = HessianCorrectedGradient(
        solver=GmresGradientSolve(), hessian_solver=SweptGradientSolve(sweeps=30, warn_tol=None)
    ).gradients(field, mesh, geometry, bvals)
    got = scheme.gradients(field, mesh, geometry, bvals)
    assert float(jnp.linalg.norm(got - exact) / jnp.linalg.norm(exact)) <= 1e-4


def test_the_hessian_outer_rate_barely_moves_with_mesh_quality() -> None:
    """The outer system's difficulty is intra-cell coupling, not skewness — so its count is nearly fixed.

    The gradient and the Hessian couple to each other *within* a cell at the same order as the
    volume term, and that coupling is there on any mesh. The inner system is the one that tracks
    skewness in the usual way. Pinned because it is what makes the outer default defensible across
    meshes while the inner one is not.
    """
    rates = {}
    for perturb in (0.1, 0.3):
        mesh = perturbed_grid_2d(8, 8, perturb=perturb, seed=3)
        systems = HessianCorrectedGradient._systems(mesh, mesh.geometry())
        inner = systems.inner()
        rates[perturb] = (
            contraction_rate(inner).rate,
            contraction_rate(
                systems.outer(SweptGradientSolve(sweeps=8, warn_tol=None), inner)
            ).rate,
        )

    inner_growth = rates[0.3][0] / rates[0.1][0]
    outer_growth = rates[0.3][1] / rates[0.1][1]
    assert outer_growth < 1.5, "the outer rate is nearly mesh-independent"
    assert inner_growth > 2.0, "the inner rate tracks the skewness"


def test_the_two_calibrated_factories_expose_one_calibration_surface() -> None:
    """Two builders of one thing drift a keyword at a time, and no single change to either looks wrong.

    Both factories answer the same question — what sweep count does this mesh need — and differ only
    in which systems they have to measure. Their calibration keywords are therefore one surface, and
    this derives what that surface is from the settings object itself, so adding a setting and wiring
    it into only one of them fails here rather than in whichever case reaches for it first.
    """
    expected = {f.name: f.default for f in dataclasses.fields(SweepCalibration)}
    # `preconditioner` is genuinely one scheme's property and not drift: the corrected scheme's
    # default preconditioner is a cheap approximation of its diagonal block, which can fail outright
    # on a degenerate cell, so it offers the exact block as an escape. The Hessian-corrected scheme
    # already inverts its exact block on both of its systems and so has no cheaper option to choose
    # between. ⚠️ That reasoning is about today's code: if that scheme ever gains a preconditioner
    # choice of its own, this exemption stops being justified and the keyword belongs on both.
    # `local_schur_block` is one scheme's property for a different reason, and a structural one:
    # only the Hessian-corrected scheme eliminates a block, so only it has a Schur complement whose
    # diagonal its preconditioner can be built from.
    scheme_specific = {"mesh", "geometry", "schur", "preconditioner", "local_schur_block"}

    for factory in (CorrectedGreenGauss.calibrated, HessianCorrectedGradient.calibrated):
        parameters = inspect.signature(factory).parameters
        offered = {name: p.default for name, p in parameters.items() if name not in scheme_specific}
        assert offered == expected, f"{factory.__qualname__} does not carry the calibration surface"


def _quadratic_3d(x):
    return (
        0.7
        + 1.3 * x[..., 0]
        - 0.9 * x[..., 1]
        + 0.8 * x[..., 2]
        + 0.5 * x[..., 0] ** 2
        + 0.4 * x[..., 0] * x[..., 1]
        - 0.3 * x[..., 1] * x[..., 2]
    )


def _quadratic_3d_grad(x):
    return jnp.stack(
        [
            1.3 + 1.0 * x[..., 0] + 0.4 * x[..., 1],
            -0.9 + 0.4 * x[..., 0] - 0.3 * x[..., 2],
            0.8 - 0.3 * x[..., 1],
        ],
        axis=1,
    )


def _median_gradient_error(mesh) -> float:
    """Median per-cell relative gradient error for the Hessian-corrected scheme on a quadratic."""
    geom = mesh.geometry()
    grad = HessianCorrectedGradient(
        solver=GmresGradientSolve(),
        hessian_solver=SweptGradientSolve(sweeps=12, warn_tol=None),
    ).gradients(_quadratic_3d(geom.cell.centroid), mesh, geom, _quadratic_3d(geom.face.centroid))
    exact = _quadratic_3d_grad(geom.cell.centroid)
    relative = jnp.linalg.norm(grad - exact, axis=-1) / jnp.linalg.norm(exact, axis=-1)
    return float(jnp.median(relative))


def test_hessian_corrected_reconstructs_a_quadratic_on_WARPED_faces() -> None:
    """The scheme stays exact when the faces are non-planar — the reason it carries a warp term.

    A fully perturbed hex grid warps its quad faces, which breaks the constant-face-normal assumption
    the second-order Green–Gauss derivation rests on. Without the face's warp moment the
    reconstruction of a quadratic degrades to ~4e-02 here — no better than the corrected Green–Gauss
    scheme this one exists to improve on, so the whole cost of solving for a Hessian buys nothing.

    The threshold is a machine-precision one deliberately: with the moment carried in BOTH equations
    the scheme is exact for a quadratic on a warped mesh exactly as it is on a planar one, so this
    asserts the defining property rather than a tolerance someone chose.
    """
    assert _median_gradient_error(perturbed_grid_3d(8, 8, 8, perturb=0.3, seed=1)) < 1e-10


def test_hessian_corrected_is_unaffected_on_PLANAR_faces() -> None:
    """The warp term is identically zero on planar faces, so this path must stay near-exact.

    Paired with the warped case above deliberately: it is what says the correction is a correction
    and not a tuning, since a mesh with nothing to correct must be untouched by it.
    """
    assert (
        _median_gradient_error(columnwise_perturbed_grid_3d(8, 8, 8, perturb=0.3, seed=1)) < 1e-10
    )


def test_hessian_corrected_survives_a_sliver_cell_without_going_non_finite() -> None:
    """A near-degenerate cell degrades accuracy but must not produce NaN or inf.

    An automatic mesher produces slivers, and the per-cell blocks are inverted with an unguarded
    ``jnp.linalg.inv`` — a single non-finite row would poison the whole Krylov solve rather than the
    one bad cell, since the reconstruction is a global solve. Measured up to a volume ratio of 3.6e8,
    the error plateaus instead of diverging and no entry goes non-finite.
    """
    mesh = _sliver_mesh(1e-6)
    geom = mesh.geometry()
    grad = HessianCorrectedGradient(
        solver=GmresGradientSolve(),
        hessian_solver=SweptGradientSolve(sweeps=12, warn_tol=None),
    ).gradients(_quadratic_3d(geom.cell.centroid), mesh, geom, _quadratic_3d(geom.face.centroid))
    assert bool(jnp.all(jnp.isfinite(grad)))


def _sliver_mesh(squash: float, cell: tuple[int, int, int] = (4, 4, 4), n: int = 8):
    """A warped grid with ONE cell squashed flat -- an automatic mesher's sliver, on purpose.

    ⚠️ Only that cell's four top nodes move, so every cell stays **closed**. Squashing a whole node
    band instead is much easier to write and produces an INVALID mesh: its face area-vectors stop
    summing to zero (measured: a closure residual of 6.6e-01 against 1e-16 here). Green–Gauss *is*
    the divergence theorem, so on such a mesh the operator is wrong and no solver recovers it -- an
    exact solve leaves 3e+01 where on this fixture it reaches 1e-11. A test built on that mesh is
    measuring the fixture, not the scheme.

    ``aquaflux.mesh.quality.closed_cell_residual`` is the check; note ``face_planarity`` does not
    reliably flag it, so planarity is not a substitute.
    """

    def node(i: int, j: int, k: int) -> int:
        return (k * (n + 1) + j) * (n + 1) + i

    i, j, k = cell
    base = perturbed_grid_3d(n, n, n, perturb=0.2, seed=5)
    coords = np.asarray(base.node_coords).copy()
    for a in (i, i + 1):
        for b in (j, j + 1):
            top, bottom = node(a, b, k + 1), node(a, b, k)
            coords[top] = coords[bottom] + squash * (coords[top] - coords[bottom])
    return eqx.tree_at(lambda m: m.node_coords, base, jnp.asarray(coords))


def test_the_sliver_fixture_is_a_valid_mesh() -> None:
    """The fixture must close, or every test built on it measures a broken mesh instead of a scheme.

    Pinned because the invalid variant is the natural thing to write and fails silently: it looks
    like a sliver, reports a plausible volume ratio, and quietly breaks the divergence theorem the
    whole discretization rests on.
    """
    for squash in (1e-4, 1e-8):
        assert float(jnp.max(closed_cell_residual(_sliver_mesh(squash)))) < 1e-12


def _corrected_error(mesh, preconditioner):
    geom = mesh.geometry()
    grad = CorrectedGreenGauss(preconditioner=preconditioner).gradients(
        _quadratic_3d(geom.cell.centroid), mesh, geom, _quadratic_3d(geom.face.centroid)
    )
    exact = _quadratic_3d_grad(geom.cell.centroid)
    relative = jnp.linalg.norm(grad - exact, axis=-1) / jnp.linalg.norm(exact, axis=-1)
    return relative, jnp.asarray(geom.cell.volume)


def test_inverse_cell_volume_is_the_default() -> None:
    """The cheap preconditioner stays the default -- this option is opt-in, not a migration."""
    assert isinstance(CorrectedGreenGauss().preconditioner, InverseCellVolume)


def test_exact_cell_block_bounds_the_error_on_a_sliver_that_1_over_V_does_not() -> None:
    """``1/V`` diverges with the volume ratio on a degenerate cell; the true block does not.

    This is the reason the option exists. ``A_g``'s per-cell block is the volume less a coupling that
    scales with face area, so as a cell flattens the volume vanishes while the coupling does not and
    ``1/V`` stops approximating the block at all. The error it leaves grows without bound -- ~1.7e+10
    at a volume ratio of 1.9e6 and ~1.7e+18 at 1.9e8 -- while the exact block holds ~1.7e-02 at both,
    which is the scheme's own discretization error on such a cell rather than a diverging iteration
    stacked on top of it.
    """
    for squash, floor in ((1e-6, 1e6), (1e-8, 1e12)):
        mesh = _sliver_mesh(squash)
        volume_error, volume = _corrected_error(mesh, InverseCellVolume())
        block_error, _ = _corrected_error(mesh, ExactCellBlock())
        sliver = volume < 10 * volume.min()
        assert float(volume_error[sliver].max()) > floor  # the default really does diverge here
        assert (
            float(block_error[sliver].max()) < 1e-1
        )  # and the block holds it at the scheme's own error


def test_exact_cell_block_does_not_change_a_healthy_mesh() -> None:
    """On a mesh of reasonable quality the two agree, so this buys robustness and not accuracy.

    Pinned because it is what makes the option safe to reach for: a user switching to it on a case
    that diverges must not find their good cells answered differently.
    """
    for mesh in (structured_grid_3d(8, 8, 8), perturbed_grid_3d(8, 8, 8, perturb=0.3, seed=1)):
        volume_error, _ = _corrected_error(mesh, InverseCellVolume())
        block_error, _ = _corrected_error(mesh, ExactCellBlock())
        # Compared in AGGREGATE, not cell by cell. At a finite sweep count the two preconditioners
        # produce slightly different iterates while converging to the same solution, so a
        # per-cell equality would pin the iteration path rather than the accuracy, and would fail on
        # cells whose error is near zero for reasons that say nothing about either choice.
        for statistic in (jnp.median, lambda a: jnp.percentile(a, 99), jnp.max):
            assert statistic(block_error) == pytest.approx(statistic(volume_error), rel=0.02)


def test_calibration_carries_the_preconditioner_it_measured() -> None:
    """The sweep count belongs to the (operator, preconditioner) pairing it was measured on.

    Calibrating under one preconditioner and running under another would report a count for a system
    that never runs, which is the failure the bundled ``GradientSystem`` exists to prevent.
    """
    mesh = structured_grid_3d(4, 4, 4)
    scheme = CorrectedGreenGauss.calibrated(mesh, mesh.geometry(), preconditioner=ExactCellBlock())
    assert isinstance(scheme.preconditioner, ExactCellBlock)


def _hessian_error(mesh, *, local_schur_block: bool):
    geom = mesh.geometry()
    grad = HessianCorrectedGradient(local_schur_block=local_schur_block).gradients(
        _quadratic_3d(geom.cell.centroid), mesh, geom, _quadratic_3d(geom.face.centroid)
    )
    exact = _quadratic_3d_grad(geom.cell.centroid)
    relative = jnp.linalg.norm(grad - exact, axis=-1) / jnp.linalg.norm(exact, axis=-1)
    return relative, jnp.asarray(geom.cell.volume)


def test_the_local_schur_block_rescues_the_sweep_on_a_sliver() -> None:
    """The outer operator is the Schur complement, so its preconditioner should approximate that.

    Preconditioning with ``A_gg``'s block instead omits the elimination term
    ``A_gH A_HH⁻¹ A_Hg``. On a well-shaped cell that term is a small perturbation and the omission is
    harmless; on a flattened one the cell's volume vanishes while its face couplings do not, so the
    neglected term becomes the dominant part of that row and the sweep stops converging there.
    Measured at a volume ratio of 1.9e4: the sliver cell goes from 8.1e-01 to 1.4e-05.
    """
    error, volume = _hessian_error(_sliver_mesh(1e-4), local_schur_block=False)
    block_error, _ = _hessian_error(_sliver_mesh(1e-4), local_schur_block=True)
    sliver = volume < 10 * volume.min()
    assert float(error[sliver].max()) > 1e-2  # the shipped block really does struggle here
    assert float(block_error[sliver].max()) < 1e-3  # and the Schur block really does rescue it


def test_the_local_schur_block_does_not_regress_a_healthy_mesh() -> None:
    """It must not cost exactness where the omitted term was harmless — it is slightly better there.

    Pinned because two earlier attempts at this block DID regress clean meshes, both by building
    something that was not the diagonal block: one summed only the owner-side columns, dropping half
    of every cell's own diagonal, and one composed the operators rather than the per-cell blocks, so
    a second scatter reached into neighbouring cells' Hessians. Either way the result was plausible,
    slightly worse, and easy to mistake for the idea itself failing.
    """
    for mesh in (
        structured_grid_3d(8, 8, 8),
        columnwise_perturbed_grid_3d(8, 8, 8, perturb=0.3, seed=1),
        perturbed_grid_3d(8, 8, 8, perturb=0.3, seed=1),
    ):
        plain, _ = _hessian_error(mesh, local_schur_block=False)
        block, _ = _hessian_error(mesh, local_schur_block=True)
        assert float(jnp.median(block)) < 1e-10
        assert float(jnp.median(block)) <= 3.0 * float(jnp.median(plain))
