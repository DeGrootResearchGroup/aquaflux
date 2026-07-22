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

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.mesh import structured_grid_2d
from aquaflux.mesh.quality import face_planarity
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    GmresGradientSolve,
    HessianCorrectedGradient,
    SweptGradientSolve,
)

from tests.support.meshes import columnwise_perturbed_grid_3d, perturbed_grid_2d

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
    volume = jnp.ones(4)
    rhs = jnp.arange(1.0, 5.0)

    def operator(v):
        return 3.0 * v  # a diagonal the bare (un-hooked) iteration would diverge on

    plain = solver.solve(volume, operator, rhs)
    identity = solver.solve(volume, operator, rhs, operator_hook=lambda x: x)
    assert jnp.allclose(plain, identity)  # an identity hook changes nothing

    zeroed = solver.solve(volume, operator, rhs, operator_hook=lambda x: jnp.zeros_like(x))
    assert jnp.allclose(
        zeroed, solver.sweeps * rhs
    )  # operator sees 0 -> x += V^{-1} rhs each sweep


def test_gmres_gradient_solve_refuses_distributed_operator_hook() -> None:
    """GMRES forms whole-vector inner products, so it cannot honour a per-apply ghost exchange and
    must raise rather than silently return a wrong owned gradient."""
    solver = GmresGradientSolve()
    with pytest.raises(NotImplementedError, match="SweptGradientSolve"):
        solver.solve(jnp.ones(3), lambda v: v, jnp.arange(3.0), operator_hook=lambda x: x)


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
    schur = HessianCorrectedGradient(schur=True).gradients(phi, mesh, geom, bvals)
    coupled = HessianCorrectedGradient(schur=False).gradients(phi, mesh, geom, bvals)
    assert jnp.allclose(schur, coupled, atol=1e-10)


def test_hessian_accepts_an_injected_solve_strategy() -> None:
    """The linear solve is an injected `GradientSolve`, exactly as in `CorrectedGreenGauss`: the
    default is GMRES, and a (sufficiently-swept) `SweptGradientSolve` reaches the same reconstruction
    on both the Schur and the full-coupled paths — the solve strategy is orthogonal to the
    discretization. (The sweep needs many iterations here because the gradient-Hessian coupling is
    strong; GMRES is the practical default.)"""
    mesh = perturbed_grid_2d(16, 16, perturb=0.2)
    geom = mesh.geometry()
    phi = _quadratic(geom.cell.centroid)
    bvals = _quadratic(geom.face.centroid)
    assert isinstance(HessianCorrectedGradient().solver, GmresGradientSolve)
    gmres = HessianCorrectedGradient().gradients(phi, mesh, geom, bvals)
    for schur in (True, False):
        swept = HessianCorrectedGradient(
            solver=SweptGradientSolve(sweeps=80, warn_tol=None), schur=schur
        ).gradients(phi, mesh, geom, bvals)
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
