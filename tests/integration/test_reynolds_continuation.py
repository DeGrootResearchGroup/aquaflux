"""Integration: Reynolds-number continuation on the coupled RANS solve.

Reynolds continuation walks a homotopy in Reynolds number -- a sequence of lower-Re solves seeded from
one another up to the target -- to reach the target-Re root more robustly than a cold direct solve.
Because it **dissolves at the target** (the final solve is the true physical problem), it must change
only the path, never the root or its exact adjoint. These check exactly that on the small turbulent
channel (Re = 2500): the same converged fields as a direct solve, a no-continuation identity, and the
implicit-function-theorem adjoint matching a direct solve and finite differences, independent of how
many continuation points were used.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import pytest
from aquaflux.turbulence import solve_reynolds_continuation
from aquaflux.turbulence.coupled import CoupledRANS, coupled_continuation, solve_coupled

# The 560-cell turbulent channel (Re = U H / nu = 2500) and its constants, reused verbatim so the case
# is defined once.
from test_coupled_rans import PRECONDITIONER, _channel

MAX_STEPS = 60


@pytest.fixture(scope="module")
def channel():
    _, momentum, turbulence, _, _ = _channel()
    return CoupledRANS.build(momentum, turbulence)


@pytest.mark.slow
def test_reaches_the_same_root_as_a_direct_solve(channel) -> None:
    """Two continuation points (Re 25 -> 250 -> 2500) land on the direct solve's converged fields."""
    flow_c, k_c, omega_c = solve_reynolds_continuation(
        channel, n_points=2, method="twolevel", max_steps=MAX_STEPS, rtol=1e-10, **PRECONDITIONER
    )
    flow_d, k_d, omega_d = solve_coupled(
        channel, method="twolevel", max_steps=MAX_STEPS, rtol=1e-10, **PRECONDITIONER
    )

    # The continuation dissolves at the target, so the root is the direct solve's root. The k bound is
    # looser than flow/omega because both solves stop on the same omega-dominated total residual, which
    # pins the k block less tightly in relative terms -- the two paths agree on k to ~1e-6, well inside
    # the 1e-3 the coupled-vs-segregated test allows for the same reason.
    assert float(jnp.linalg.norm(flow_c - flow_d) / jnp.linalg.norm(flow_d)) < 1e-6
    assert float(jnp.linalg.norm(k_c - k_d) / jnp.linalg.norm(k_d)) < 1e-5
    assert float(jnp.linalg.norm(omega_c - omega_d) / jnp.linalg.norm(omega_d)) < 1e-6
    # And it is as converged as the direct solve and strictly positive. The continuation's final solve
    # is warm-started, so its residual is measured against a small ||R0||; comparing it to the direct
    # solve's residual (rather than an absolute bar) keeps the check honest for both.
    residual_c = float(jnp.linalg.norm(channel.residual(channel.pack_state(flow_c, k_c, omega_c))))
    residual_d = float(jnp.linalg.norm(channel.residual(channel.pack_state(flow_d, k_d, omega_d))))
    assert residual_c < max(1e-6, 10.0 * residual_d)
    assert float(jnp.min(k_c)) >= 0.0 and float(jnp.min(omega_c)) > 0.0


@pytest.mark.slow
def test_zero_points_is_a_plain_direct_solve(channel) -> None:
    """``n_points = 0`` runs no continuation -- bit-for-bit the direct solve from the same hybrid IC."""
    flow_c, k_c, omega_c = solve_reynolds_continuation(
        channel, n_points=0, method="twolevel", max_steps=MAX_STEPS, rtol=1e-10, **PRECONDITIONER
    )
    flow_d, k_d, omega_d = solve_coupled(
        channel, method="twolevel", max_steps=MAX_STEPS, rtol=1e-10, **PRECONDITIONER
    )
    assert jnp.array_equal(flow_c, flow_d)
    assert jnp.array_equal(k_c, k_d)
    assert jnp.array_equal(omega_c, omega_d)


@pytest.mark.slow
def test_adjoint_matches_a_direct_solve_and_is_point_count_independent(channel) -> None:
    """``jax.grad`` through the continuation equals the direct-solve adjoint and finite differences.

    The lower-Re ramp only makes an initial guess (its result is ``stop_gradient``-ed), so the gradient
    is the target solve's implicit-function-theorem adjoint -- the same single transpose solve a direct
    solve yields, independent of the number of continuation points. Build the target continuation once
    outside ``jax.grad`` on concrete parameters (the block preconditioner must not be traced) and hand
    it to the final solve.
    """
    # Build the target-viscosity continuation once, outside jax.grad (the block preconditioner must be
    # constructed on concrete parameters).
    continuation = coupled_continuation(
        channel, channel.pack_state(*_hybrid(channel)), method="twolevel", **PRECONDITIONER
    )

    def objective(nu_scale, n_points):
        scaled = channel.with_scaled_molecular_viscosity(nu_scale)
        _, k, _ = solve_reynolds_continuation(
            scaled,
            n_points=n_points,
            method="twolevel",
            max_steps=MAX_STEPS,
            continuation=continuation,
            **PRECONDITIONER,
        )
        return jnp.sum(k**2)

    grad_direct = float(jax.grad(objective)(1.0, 0))  # n_points = 0: the plain direct solve
    grad_ramped = float(jax.grad(objective)(1.0, 1))  # one continuation point
    eps = 1e-4
    finite_difference = float((objective(1.0 + eps, 0) - objective(1.0 - eps, 0)) / (2 * eps))

    assert abs(grad_direct - finite_difference) / abs(finite_difference) < 1e-5
    # The adjoint is independent of the continuation depth (it is the target solve's IFT adjoint).
    assert abs(grad_ramped - grad_direct) / abs(grad_direct) < 1e-6


def _hybrid(coupled):
    from aquaflux.turbulence import hybrid_initialize

    return hybrid_initialize(coupled.momentum, coupled.turbulence)
