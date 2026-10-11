"""Integration: the staged flow march holding a bulk velocity (a ``MassFlow`` drive).

A laminar streamwise-periodic channel driven to a target bulk velocity is marched by
:func:`~aquaflux.flow.solve_flow_march` -- the same march every other flow solve runs on, with the force
appended to its state -- and judged against the bordered Newton solve the segregated loop uses and the
analytic Poiseuille force, and differentiated through.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from aquaflux.flow import bulk_velocity_flow_solve, mass_flow_border, solve_flow_march
from aquaflux.solve import (
    CompleteLu,
    Convergence,
    DualTimeLoop,
    Globalization,
    MaterializedJacobian,
    RootSolveSettings,
    relative_residual_gmres,
)

from tests.unit.test_mean_velocity import MU, U_TARGET, H, _bulk_velocity, _channel

#: The march's opening shift. On this Stokes channel the default schedule (``beta0 = 2``) crawls through
#: the viscous spin-up from the plug start whether or not the force is held -- the same with a prescribed
#: force -- while a light shift converges quadratically in about a dozen steps.
LIGHT = Globalization(beta0=0.1)


def _held_force(momentum, flow) -> jnp.ndarray:
    return mass_flow_border(momentum, momentum.layout).force_holding(momentum.residual(flow))


@pytest.fixture(scope="module")
def channel():
    return _channel(beta_initial=0.05)


@pytest.fixture(scope="module")
def marched(channel):
    return solve_flow_march(channel, globalization=LIGHT)


def test_the_march_reaches_the_root_the_bordered_newton_solve_reaches(channel, marched) -> None:
    """The bulk velocity is the target, and the fields and force are the segregated loop's flow solve's."""
    solve = bulk_velocity_flow_solve(
        channel, root_solve=RootSolveSettings(linear_solver=lx.AutoLinearSolver(well_posed=True))
    )
    solved, reference = solve(channel, channel.initial_state())
    assert marched.shape == (channel.layout.size,)
    assert _bulk_velocity(channel, marched, 0) == pytest.approx(U_TARGET, abs=1e-9)
    assert float(jnp.linalg.norm(marched - reference) / jnp.linalg.norm(reference)) < 1e-7
    assert float(_held_force(channel, marched)) == pytest.approx(
        float(solved.drive.force), rel=1e-7
    )
    # And the discrete force is the analytic Poiseuille one to the discretization error.
    assert float(_held_force(channel, marched)) == pytest.approx(
        12.0 * MU * U_TARGET / H**2, rel=2e-3
    )


def test_a_dual_time_march_reaches_the_same_root_and_reports_the_constraint(
    channel, marched
) -> None:
    """Every setting of the march reaches a held bulk velocity: here a dual-time loop and its control.

    The row-scaled measure reports the constraint row beside the flow's equations, and a checkpoint is
    handed the fields alone, so a case's checkpoint writer reads it as it reads any other.
    """
    residuals, checkpoints = [], []
    flow = solve_flow_march(
        channel,
        dual_time=DualTimeLoop(inner_steps=3),
        convergence=Convergence(rtol=1e-8),
        max_steps=80,
        on_residuals=residuals.append,
        on_checkpoint=lambda report, state: checkpoints.append(state.shape),
    )
    assert float(jnp.linalg.norm(flow - marched) / jnp.linalg.norm(marched)) < 1e-6
    assert residuals and list(residuals[-1])[-1] == "bulk_velocity"
    assert checkpoints and set(checkpoints) == {(channel.layout.size,)}


def test_the_gradient_through_the_march_is_the_sensitivity_at_fixed_bulk_velocity(
    channel, marched
) -> None:
    """``d beta / d mu`` through the march matches finite differences and the analytic ``12 U_b / H^2``."""

    # A warm start carries the force as well as the fields: the drive's force is where the solve for it
    # starts, so the converged one is handed on beside the converged fields.
    warm = eqx.tree_at(lambda m: m.drive.force, channel, _held_force(channel, marched))

    def force(scale):
        scaled = warm.with_scaled_molecular_viscosity(scale)
        flow = solve_flow_march(
            scaled,
            marched,
            globalization=LIGHT,
            # Warm-started, its own starting residual is already small, so the stop is an absolute one,
            # well past what a finite difference at eps = 1e-4 can resolve.
            convergence=Convergence(rtol=0.0, atol=1e-9),
            adjoint_solver=relative_residual_gmres(1e-10, restart=120),
        )
        return _held_force(scaled, flow)

    analytic = float(jax.grad(force)(1.0))
    eps = 1e-4
    finite_difference = float((force(1.0 + eps) - force(1.0 - eps)) / (2 * eps))
    assert analytic == pytest.approx(finite_difference, rel=1e-5)
    # beta = 12 mu U_b / H^2 is linear in mu, so d beta / d(scale) is beta itself.
    assert analytic == pytest.approx(float(_held_force(channel, marched)), rel=1e-6)


def test_a_materialized_preconditioner_is_refused_beside_a_held_bulk_velocity(channel) -> None:
    with pytest.raises(TypeError, match="holds a bulk velocity"):
        solve_flow_march(channel, preconditioner=MaterializedJacobian(CompleteLu()))
