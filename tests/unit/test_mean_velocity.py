"""Unit: the bulk-velocity-constrained flow solve holds the bulk velocity by construction.

The body force is a scalar Lagrange multiplier enforcing ``<U_dir> = target``, appended to the flow
state and solved jointly with the flow by the production Newton
(:func:`aquaflux.flow.bulk_velocity_flow_solve`). Tested in isolation on a laminar Stokes channel --
no turbulence, a small mesh -- where the exact answer is known: a plane channel of full height ``H``
driven by a uniform force ``beta`` has bulk velocity ``beta H^2 / (12 mu)``, so the force that hits a
target ``U_b`` is ``beta = 12 mu U_b / H^2``.

The solve is convergence-gated, so the constraint is met to the residual tolerance (machine precision
here), and the recovered force matches the analytic value to the discretization error alone.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import (
    BlockPreconditioner,
    BoundaryDriven,
    MassFlow,
    MomentumContinuity,
    NoSlipWall,
    PinnedPoint,
    bulk_velocity_flow_solve,
)
from aquaflux.flow.mean_velocity import _bordered_preconditioner
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import RootSolveSettings, solve_linear

H, MU, RHO, U_TARGET = 2.0, 0.1, 1.0, 1.0  # h = H/2 = 1; beta_analytic = 12 mu U_b / H^2 = 3 mu
_DIRECT = lx.AutoLinearSolver(well_posed=True)


def _channel(beta_initial: float) -> MomentumContinuity:
    """A laminar streamwise-periodic channel with a (deliberately wrong) initial body force."""
    mesh = structured_grid_2d(4, 48, lx=1.0, ly=H, periodic=("x",), named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(RHO * MU), "density": Constant(RHO)}),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        gradient_scheme=CompactGreenGauss(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
        drive=MassFlow(target=U_TARGET, force=beta_initial),
    )


def _bulk_velocity(momentum: MomentumContinuity, state: jnp.ndarray, direction: int) -> float:
    velocity, _ = momentum.unpack(state)
    volume = momentum.geometry.cell.volume
    return float(jnp.sum(velocity[:, direction] * volume) / jnp.sum(volume))


def test_constraint_is_met_to_machine_precision() -> None:
    """The converged solve hits the target bulk velocity to the residual tolerance."""
    momentum = _channel(beta_initial=0.05)  # far from the analytic 0.3
    solve = bulk_velocity_flow_solve(momentum, root_solve=RootSolveSettings(linear_solver=_DIRECT))
    solved_momentum, flow = solve(momentum, momentum.initial_state())
    assert _bulk_velocity(solved_momentum, flow, 0) == pytest.approx(U_TARGET, abs=1e-10)


def test_recovers_the_analytic_body_force() -> None:
    """The converged multiplier matches ``beta = 12 mu U_b / H^2`` to the discretization error."""
    momentum = _channel(beta_initial=0.05)
    solve = bulk_velocity_flow_solve(momentum, root_solve=RootSolveSettings(linear_solver=_DIRECT))
    solved_momentum, _ = solve(momentum, momentum.initial_state())
    beta_analytic = 12.0 * MU * U_TARGET / H**2  # = 3 mu = 0.3
    # The constraint is met on the *discrete* problem exactly; the gap to the continuum beta is the FVM
    # discretization on 48 wall-normal cells (a fraction of a percent), not a solver error.
    assert float(solved_momentum.drive.force) == pytest.approx(beta_analytic, rel=2e-3)


def test_initial_force_does_not_change_the_result() -> None:
    """The constraint fixes the outcome: a different starting force reaches the same converged force."""
    forces = []
    for beta_initial in (0.02, 0.5):
        momentum = _channel(beta_initial)
        solve = bulk_velocity_flow_solve(
            momentum, root_solve=RootSolveSettings(linear_solver=_DIRECT)
        )
        solved_momentum, _ = solve(momentum, momentum.initial_state())
        forces.append(float(solved_momentum.drive.force))
    assert forces[0] == pytest.approx(forces[1], rel=1e-8)


def test_bordered_preconditioner_inverts_the_augmented_jacobian() -> None:
    """With an exact flow inverse ``M = J^{-1}``, the bordered preconditioner is exactly ``J_aug^{-1}``.

    This checks two things at once: that AD assembles the border ``[[J, a], [c^T, 0]]`` the hand-built
    ``(a, c)`` claim (the ``e_beta`` column and the constraint row), and that the Schur elimination in
    :func:`_bordered_preconditioner` inverts it (right formula, right signs).
    """
    momentum = _channel(beta_initial=0.05)
    flow0 = momentum.initial_state()
    beta0 = 0.05
    drive, fields = momentum.drive, momentum.layout
    augmented0 = jnp.append(flow0, beta0)
    volume = momentum.geometry.cell.volume

    def augmented_residual(augmented: jnp.ndarray) -> jnp.ndarray:
        flow, beta = augmented[:-1], augmented[-1]
        forced = drive.forced(momentum, beta)
        velocity, _ = forced.unpack(flow)
        bulk = jnp.sum(velocity[:, 0] * volume) / jnp.sum(volume)
        return jnp.append(forced.residual(flow), bulk - U_TARGET)

    def j_augmented(v: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(augmented_residual, (augmented0,), (v,))[1]

    force, average = drive.constraint_vectors(momentum)

    # AD's border matches the hand-built (a, c): the beta column is [a; 0], and the constraint row is c.
    beta_column = j_augmented(jnp.append(jnp.zeros_like(flow0), 1.0))
    assert jnp.allclose(beta_column[:-1], force, atol=1e-10)
    assert float(beta_column[-1]) == pytest.approx(0.0, abs=1e-12)
    probe = jax.random.normal(jax.random.PRNGKey(0), flow0.shape)
    row = j_augmented(jnp.append(probe, 0.0))[-1]
    assert float(row) == pytest.approx(float(jnp.dot(average, probe)), abs=1e-8)

    # Exact flow inverse -> the bordered preconditioner is exactly J_aug^{-1}: M_aug(J_aug v) == v.
    forced0 = drive.forced(momentum, beta0)

    def flow_inverse(_augmented_flow: jnp.ndarray) -> object:
        def solve_flow(residual: jnp.ndarray) -> jnp.ndarray:
            matvec = lambda v: jax.jvp(forced0.residual, (flow0,), (v,))[1]  # noqa: E731
            return solve_linear(matvec, residual, _DIRECT)[0]

        return solve_flow

    m_augmented = _bordered_preconditioner(flow_inverse, drive, fields, force, average)(augmented0)
    v = jax.random.normal(jax.random.PRNGKey(1), augmented0.shape)
    assert jnp.allclose(m_augmented(j_augmented(v)), v, atol=1e-6)


def test_preconditioned_iterative_solve_matches_the_direct_solve() -> None:
    """The block-preconditioned augmented Krylov solve reaches the same root as a direct solve.

    Exercises the production path -- the block-SIMPLE AMG wrapped by :func:`_bordered_preconditioner`,
    driving an iterative (GMRES) augmented solve -- and checks it lands on the direct solve's answer
    (the constraint met, the same body force).
    """
    mesh = structured_grid_2d(16, 48, lx=1.0, ly=H, periodic=("x",), named_boundaries=True)
    momentum = MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(RHO * MU), "density": Constant(RHO)}),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        gradient_scheme=CompactGreenGauss(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
        drive=MassFlow(target=U_TARGET, force=0.05),
    )
    preconditioner = BlockPreconditioner.build(momentum).factory()
    gmres = lx.GMRES(rtol=1e-8, atol=1e-10)
    solve_iterative = bulk_velocity_flow_solve(
        momentum,
        root_solve=RootSolveSettings(linear_solver=gmres),
        preconditioner=preconditioner,
    )
    solve_direct = bulk_velocity_flow_solve(
        momentum, root_solve=RootSolveSettings(linear_solver=_DIRECT)
    )

    momentum_it, flow_it = solve_iterative(momentum, momentum.initial_state())
    momentum_di, flow_di = solve_direct(momentum, momentum.initial_state())

    assert _bulk_velocity(momentum_it, flow_it, 0) == pytest.approx(U_TARGET, abs=1e-8)
    assert float(momentum_it.drive.force) == pytest.approx(float(momentum_di.drive.force), rel=1e-6)
    assert jnp.allclose(flow_it, flow_di, atol=1e-6)


def test_a_solve_refuses_an_assembler_that_is_not_mass_flow_driven() -> None:
    """Without a ``MassFlow`` drive there is no target to hold and no leaf to write the iterate into.

    The refusal is at build, not inside the traced Newton step: the residual writes each iterate into
    ``momentum.drive.force``, which on a boundary-driven assembler is a path that does not exist, so
    the failure would otherwise surface as a tree-path error from inside ``equinox.tree_at``.
    """
    mesh = structured_grid_2d(4, 8, lx=1.0, ly=H, periodic=("x",), named_boundaries=True)
    momentum = MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(RHO * MU), "density": Constant(RHO)}),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        gradient_scheme=CompactGreenGauss(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
    )
    assert isinstance(momentum.drive, BoundaryDriven)
    with pytest.raises(TypeError, match="MassFlow"):
        bulk_velocity_flow_solve(momentum)


def _laminar_channel(mu: float) -> MomentumContinuity:
    """The laminar constrained channel at viscosity ``mu``, on the gradient test's own mesh."""
    mesh = structured_grid_2d(16, 24, lx=1.0, ly=H, periodic=("x",), named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(RHO * mu), "density": Constant(RHO)}),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        gradient_scheme=CompactGreenGauss(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
        drive=MassFlow(target=U_TARGET, force=0.05),
    )


def _laminar_body_force(mu: float, solve) -> float:
    """The converged body force (the multiplier) of the laminar channel at viscosity ``mu``."""
    momentum = _laminar_channel(mu)
    solved_momentum, _ = solve(momentum, momentum.initial_state())
    return solved_momentum.drive.force


def test_constrained_solve_is_reverse_differentiable() -> None:
    """``jax.grad`` through the constrained solve matches finite differences.

    Differentiate the converged body force ``beta`` (the multiplier) w.r.t. viscosity: for the laminar
    channel ``<U> = beta H^2 / (12 mu) = U_b`` gives ``beta = 12 mu U_b / H^2``, so ``d beta / d mu =
    12 U_b / H^2 = 3``. This exercises the implicit-function-theorem adjoint (the assembler threaded as
    the Newton parameter) -- without it the solve raises rather than returning a gradient.
    """
    solve = bulk_velocity_flow_solve(
        _laminar_channel(MU), root_solve=RootSolveSettings(linear_solver=_DIRECT)
    )
    ad = float(jax.grad(lambda mu: _laminar_body_force(mu, solve))(MU))
    eps = 1e-6
    fd = (_laminar_body_force(MU + eps, solve) - _laminar_body_force(MU - eps, solve)) / (2 * eps)
    assert ad == pytest.approx(float(fd), rel=1e-5)
    assert ad == pytest.approx(12.0 * U_TARGET / H**2, rel=2e-2)  # analytic 3.0 to discretization


def test_preconditioned_adjoint_matches_the_unpreconditioned_adjoint() -> None:
    """The reverse-mode gradient is identical with and without the constraint preconditioner.

    The block-preconditioned iterative solve reuses the bordered preconditioner **transposed** (formed
    by :func:`jax.linear_transpose`) for the adjoint transpose solve; the preconditioner only
    accelerates the adjoint Krylov iteration, so the gradient must equal the direct (unpreconditioned)
    adjoint's -- the guard rail that the adjoint preconditioner is correct, not just fast.
    """
    reference = _laminar_channel(MU)
    preconditioner = BlockPreconditioner.build(reference).factory()  # built off-grad, concrete
    solve_direct = bulk_velocity_flow_solve(
        reference, root_solve=RootSolveSettings(linear_solver=_DIRECT)
    )
    solve_pre = bulk_velocity_flow_solve(
        reference,
        root_solve=RootSolveSettings(linear_solver=lx.GMRES(rtol=1e-9, atol=1e-11)),
        preconditioner=preconditioner,
    )
    grad_direct = float(jax.grad(lambda mu: _laminar_body_force(mu, solve_direct))(MU))
    grad_pre = float(jax.grad(lambda mu: _laminar_body_force(mu, solve_pre))(MU))
    assert grad_pre == pytest.approx(grad_direct, abs=1e-9)
