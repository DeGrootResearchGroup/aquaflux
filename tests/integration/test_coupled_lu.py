"""Integration: the monolithic complete-LU-preconditioned coupled RANS Newton solve on a turbulent channel.

The complete-LU counterpart of :mod:`tests.integration.test_coupled_ilut`. The coupled continuation's
block-triangular SIMPLE preconditioner is replaced by a single *complete* LU factorization of the assembled
coupled Jacobian (:func:`~aquaflux.turbulence.coupled_lu_continuation`). These check the same two
properties that make it a usable drop-in: handed to ``solve_coupled`` it converges the monolithic Newton
to the **same** fixed point the block preconditioner reaches, and -- built once outside ``jax.grad`` on
concrete parameters -- it yields the exact coupled adjoint matching finite differences. The channel setup
is shared with the ILUT integration test (one source of truth).

Run under the always-available SciPy (SuperLU) backend so no optional dependency is needed; the complete
factorization is exact regardless of backend, so these correctness/adjoint properties are backend-independent
(the UMFPACK backend only changes the factorization speed).
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.turbulence import (
    CoupledRANS,
    coupled_lu_continuation,
    coupled_lu_refreshing_continuation,
    hybrid_initialize,
    solve_coupled,
)

from tests.integration.test_coupled_ilut import PRECONDITIONER, _channel

BACKEND = "scipy"  # always available; exact, so backend-independent correctness


@pytest.fixture(scope="module")
def case():
    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    start = hybrid_initialize(momentum, turbulence)
    return {"coupled": coupled, "start": start}


@pytest.mark.slow
def test_lu_continuation_builds_the_right_step_types(case) -> None:
    """``inner_steps > 1`` builds a complete-LU-preconditioned dual-time step; ``1`` a single-step."""
    from aquaflux.solve import DualTimeStep, PseudoTransientStep

    coupled = case["coupled"]
    flow, k, omega = case["start"]
    reference_state = coupled.pack_state(flow, k, omega)

    single = coupled_lu_continuation(coupled, reference_state, backend=BACKEND)
    assert isinstance(single, PseudoTransientStep)
    dual = coupled_lu_continuation(
        coupled, reference_state, backend=BACKEND, inner_steps=5, inner_tol=1e-3
    )
    assert isinstance(dual, DualTimeStep)
    assert dual.inner_steps == 5


@pytest.mark.slow
def test_lu_solve_converges_and_matches_the_block_preconditioned_solve(case) -> None:
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)

    lu = coupled_lu_continuation(coupled, reference_state, backend=BACKEND)
    flow_l, k_l, omega_l = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, continuation=lu, max_steps=40
    )
    residual_norm = float(
        jnp.linalg.norm(coupled.residual(coupled.pack_state(flow_l, k_l, omega_l)))
    )
    assert residual_norm < 1e-8
    assert float(jnp.min(k_l)) >= 0.0
    assert float(jnp.min(omega_l)) > 0.0
    assert float(jnp.max(k_l)) > 10.0 * float(jnp.min(jnp.abs(k_l)) + 1e-30)  # genuinely turbulent

    # Same fixed point as the block-triangular preconditioner reaches.
    flow_b, k_b, omega_b = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )
    assert float(jnp.linalg.norm(flow_l - flow_b) / jnp.linalg.norm(flow_b)) < 1e-4
    assert float(jnp.linalg.norm(k_l - k_b) / jnp.linalg.norm(k_b)) < 1e-3
    assert float(jnp.linalg.norm(omega_l - omega_b) / jnp.linalg.norm(omega_b)) < 1e-4


@pytest.mark.slow
def test_lu_adjoint_matches_finite_difference(case) -> None:
    """The coupled implicit-function-theorem adjoint is exact through the complete-LU-preconditioned solve.

    The preconditioner is ``stop_gradient``-ed (it only accelerates the Krylov solve -- here to a single
    exact iteration), so the gradient is the single transpose solve on the unfrozen coupled residual and
    matches finite differences. Built once outside ``jax.grad`` on concrete parameters.
    """
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)
    continuation = coupled_lu_continuation(coupled, reference_state, backend=BACKEND)

    def objective(nu_scale):
        scaled = eqx.tree_at(
            lambda c: c.turbulence.molecular_viscosity,
            coupled,
            coupled.turbulence.molecular_viscosity * nu_scale,
        )
        _, k, _ = solve_coupled(
            scaled, flow_ws, k_ws, omega_ws, continuation=continuation, max_steps=40
        )
        return jnp.sum(k**2)

    analytic = float(jax.grad(objective)(1.0))
    eps = 1e-4
    finite_difference = float((objective(1.0 + eps) - objective(1.0 - eps)) / (2 * eps))
    assert abs(analytic - finite_difference) / abs(finite_difference) < 1e-5


@pytest.mark.slow
def test_lu_refreshing_continuation_refreshes_the_same_step_in_place(case) -> None:
    """The refreshing builder re-factors the SAME continuation in place (the cache-hit object identity)."""
    from aquaflux.solve import DualTimeStep

    coupled = case["coupled"]
    flow, k, omega = case["start"]
    state0 = coupled.pack_state(flow, k, omega)
    state1 = state0 * 1.05

    rb = coupled_lu_refreshing_continuation(coupled, backend=BACKEND, inner_steps=5, inner_tol=1e-3)
    step0 = rb(state0)
    assert isinstance(step0, DualTimeStep)
    backend0 = step0.shift_policy.preconditioner.factors.backend
    step1 = rb(state1)
    assert step1 is step0  # same continuation object -> jitted march-step is a cache hit
    assert step1.shift_policy.preconditioner.factors.backend is backend0  # refactored in place
