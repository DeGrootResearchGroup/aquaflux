"""Integration: the monolithic-AMG-preconditioned coupled RANS Newton solve on a turbulent channel.

The coupled continuation's block-triangular SIMPLE preconditioner is replaced by a single
algebraic-multigrid V-cycle of the assembled coupled Jacobian (:func:`coupled_amg_continuation`) -- the
scaling path for large three-dimensional meshes, where the complete LU's fill is out of memory and the
threshold-ILU's factorization is prohibitively slow to build. These check the two properties that make it
a usable drop-in: handed to ``solve_coupled`` it converges the monolithic Newton to the **same** fixed
point the block preconditioner reaches, and -- built once outside ``jax.grad`` on concrete parameters --
it yields the exact coupled adjoint (a single transpose solve on the unfrozen residual, preconditioned by
the V-cycle's *transpose*, which the multigrid supplies directly), matching finite differences. The
V-cycle needs PETSc, so the module is skipped where ``petsc4py`` is unavailable.

Genuinely turbulent (Re = U H / nu = 2500), so ``k`` stays above its floor and the floor plays no part in
the converged state or its sensitivity.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("petsc4py")

from aquaflux.solve import DualTimeStep, PseudoTransientStep
from aquaflux.turbulence import (
    CoupledRANS,
    coupled_amg_continuation,
    solve_coupled,
)

from tests.integration.test_coupled_ilut import PRECONDITIONER, _channel


@pytest.fixture(scope="module")
def case():
    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    from aquaflux.turbulence import hybrid_initialize

    start = hybrid_initialize(momentum, turbulence)
    return {"coupled": coupled, "start": start}


@pytest.mark.slow
def test_amg_continuation_inner_steps_builds_a_dual_time_step(case) -> None:
    """``inner_steps`` selects a dual-time step, like the factorization builders -- a fast structural check."""
    coupled = case["coupled"]
    flow, k, omega = case["start"]
    reference_state = coupled.pack_state(flow, k, omega)

    single = coupled_amg_continuation(coupled, reference_state)
    assert isinstance(single, PseudoTransientStep)

    dual = coupled_amg_continuation(coupled, reference_state, inner_steps=5, inner_tol=1e-3)
    assert isinstance(dual, DualTimeStep)
    assert dual.inner_steps == 5


@pytest.mark.slow
def test_amg_solve_converges_and_matches_the_block_preconditioned_solve(case) -> None:
    """Handed to ``solve_coupled`` the AMG V-cycle converges to the block preconditioner's fixed point."""
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)

    amg = coupled_amg_continuation(coupled, reference_state)
    flow_a, k_a, omega_a = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, continuation=amg, max_steps=40
    )

    residual_norm = float(
        jnp.linalg.norm(coupled.residual(coupled.pack_state(flow_a, k_a, omega_a)))
    )
    assert residual_norm < 1e-8
    assert float(jnp.min(k_a)) >= 0.0
    assert float(jnp.min(omega_a)) > 0.0
    assert float(jnp.max(k_a)) > 10.0 * float(jnp.min(jnp.abs(k_a)) + 1e-30)  # genuinely turbulent

    flow_b, k_b, omega_b = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )
    assert float(jnp.linalg.norm(flow_a - flow_b) / jnp.linalg.norm(flow_b)) < 1e-4
    assert float(jnp.linalg.norm(k_a - k_b) / jnp.linalg.norm(k_b)) < 1e-3
    assert float(jnp.linalg.norm(omega_a - omega_b) / jnp.linalg.norm(omega_b)) < 1e-4


@pytest.mark.slow
def test_amg_adjoint_matches_finite_difference(case) -> None:
    """The coupled implicit-function-theorem adjoint is exact through the AMG-preconditioned solve.

    The V-cycle is ``stop_gradient``-ed (it only accelerates the Krylov iteration), so the gradient is the
    single transpose solve on the unfrozen coupled residual, preconditioned by the V-cycle's own transpose
    -- ``jax.grad`` through the AMG solve matches finite differences, exactly as for the factorizations.
    The continuation is built once outside ``jax.grad`` on concrete parameters (it must not be traced).
    """
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)
    continuation = coupled_amg_continuation(coupled, reference_state)

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
def test_amg_beta_floor_builds_the_preconditioner_above_the_marchs_own_beta(
    case, monkeypatch
) -> None:
    """``beta_floor`` clamps the PRECONDITIONER's shift while the march keeps solving at its own β.

    As β falls the shift's diagonal dominance vanishes and the frozen V-cycle degrades, but the operator
    needs the small β to make pseudo-transient progress -- so the floor applies to the preconditioner's
    copy only. Asserts both halves: the refresh receives ``max(β, beta_floor) · d``, and the step's own
    relaxation schedule (what the march actually solves) is untouched.
    """
    import numpy as np
    from aquaflux.solve import DualTimeControl
    from aquaflux.turbulence import amg_beta_tracking_refresh

    coupled = case["coupled"]
    flow, k, omega = case["start"]
    state = coupled.pack_state(flow, k, omega)

    beta, floor = 0.01, 0.05  # β well below the floor, so the clamp is active
    dual = coupled_amg_continuation(coupled, state, inner_steps=5)
    active, _ = DualTimeControl(beta_start=beta).next_step(dual, None, None)

    seen: dict[str, np.ndarray] = {}
    pc_type = type(active.shift_policy.preconditioner)
    monkeypatch.setattr(
        pc_type,
        "refresh_shift_in_place",
        lambda _self, shift: seen.__setitem__("shift", np.asarray(shift)),
    )
    monkeypatch.setattr(
        pc_type,
        "refresh_in_place",
        lambda _self, _mv, _col, _nf, shift, **_kw: seen.__setitem__("shift", np.asarray(shift)),
    )

    amg_beta_tracking_refresh(coupled, beta_floor=floor)(active, state)

    diagonal = np.asarray(active.shift_policy.base.shift_term(state).diagonal)
    assert np.allclose(seen["shift"], floor * diagonal)  # built at the FLOOR, not at β
    assert not np.allclose(seen["shift"], beta * diagonal)
    # ...and the march's own shift strength is untouched: the operator still carries the small β.
    assert float(active.relaxation_schedule.beta) == pytest.approx(beta)
