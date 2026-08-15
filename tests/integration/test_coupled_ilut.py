"""Integration: the monolithic-ILUT-preconditioned coupled RANS Newton solve on a turbulent channel.

The coupled continuation's block-triangular SIMPLE preconditioner is replaced by a single monolithic
incomplete-LU factorization of the assembled coupled Jacobian (:func:`coupled_ilut_continuation`).
These check the two properties that make it a usable drop-in: handed to ``solve_coupled`` it converges
the monolithic Newton to the **same** fixed point the block preconditioner reaches, and -- built once
outside ``jax.grad`` on concrete parameters -- it yields the exact coupled adjoint (a single transpose
solve on the unfrozen residual, preconditioned by the factorization's transpose), matching finite
differences. Genuinely turbulent (Re = U H / nu = 2500), so ``k`` stays above its floor and the floor
plays no part in the converged state or its sensitivity.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import RefreshPolicy
from aquaflux.turbulence import (
    CoupledRANS,
    SSTModel,
    SSTTurbulence,
    coupled_ilut_continuation,
    coupled_ilut_refreshing_continuation,
    hybrid_initialize,
    ilut_beta_tracking_refresh,
    inlet_k,
    inlet_omega,
    solve_coupled,
)

RHO, U_IN, H, L = 1.0, 1.0, 1.0, 4.0
NU = 4e-4  # Re = U H / nu = 2500
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * H
PRECONDITIONER = {"schur_scaling": "msimpler", "velocity": "convection"}


def _channel(nx=20, ny=14, growth=1.2):
    y_nodes = graded_nodes(ny, H, growth)
    mesh = structured_grid_2d(nx, ny, lx=L, ly=H, named_boundaries=True, y_nodes=y_nodes)
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, model))
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(U_IN, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(k_in),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(omega_in),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return momentum, turbulence


@pytest.fixture(scope="module")
def case():
    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    start = hybrid_initialize(momentum, turbulence)
    return {"coupled": coupled, "start": start}


@pytest.mark.slow
def test_ilut_continuation_inner_steps_builds_a_dual_time_step(case) -> None:
    """``inner_steps > 1`` swaps the single-step ILUT step for an ILUT-preconditioned dual-time step.

    The dual-time branch mirrors ``coupled_continuation``: an inner Newton loop per outer pseudo-timestep,
    whose implicit step the ILUT preconditions -- the path whose true-inverse conditioning at low shift is
    what lets the pseudo-timestep grow past the block preconditioner's wall.
    """
    from aquaflux.solve import DualTimeStep, PseudoTransientStep

    coupled = case["coupled"]
    flow, k, omega = case["start"]
    reference_state = coupled.pack_state(flow, k, omega)

    single = coupled_ilut_continuation(coupled, reference_state)
    assert isinstance(single, PseudoTransientStep)

    dual = coupled_ilut_continuation(coupled, reference_state, inner_steps=5, inner_tol=1e-3)
    assert isinstance(dual, DualTimeStep)
    assert dual.inner_steps == 5


@pytest.mark.slow
def test_ilut_refreshing_continuation_refreshes_the_same_step_in_place(case) -> None:
    """The refreshing builder returns a ``refresh_builder`` whose later calls re-factor the SAME
    continuation in place -- the object identity that makes ``solve_coupled``'s jitted march-step a
    compilation cache hit, so a mid-march refresh pays only the materialize + factor, not a recompile.
    """
    from aquaflux.solve import DualTimeStep

    coupled = case["coupled"]
    flow, k, omega = case["start"]
    state0 = coupled.pack_state(flow, k, omega)
    state1 = state0 * 1.05  # a mildly "developed" state to refresh at

    rb = coupled_ilut_refreshing_continuation(coupled, inner_steps=5, inner_tol=1e-3)
    step0 = rb(state0)  # first call builds the continuation
    assert isinstance(step0, DualTimeStep)
    factors0 = step0.shift_policy.preconditioner.factors

    step1 = rb(state1)  # later call refreshes in place
    # Same continuation object -> the jitted march-step sees an unchanged pytree (cache hit)...
    assert step1 is step0
    # ...but the ILUT was genuinely re-factored at the new state (a fresh factorization object).
    assert step1.shift_policy.preconditioner.factors is not factors0


def test_staleness_beta_gate_fires_on_first_call_beta_move_and_staleness_cap() -> None:
    """The β-tracking gate re-factors on the first step, on a β move past the threshold, or at the
    staleness cap -- and skips otherwise, so the ILUT's expensive re-factor is paid only when it pays off.

    Pure logic, no solver: the gate is the whole novelty of the ILUT β-tracking refresh (the refactor
    mechanism itself is shared with the drift path), so it earns a fast, isolated test.
    """
    from aquaflux.turbulence.coupled import _staleness_beta_gate

    gate = _staleness_beta_gate(refresh_every=3, beta_rel_change=0.25)
    assert gate(1.0) is True  # first call always fires (nothing factored yet)
    assert gate(1.1) is False  # +10% < 25%, 1 step since -> reuse
    assert gate(1.2) is False  # +20% < 25% (vs last-refresh 1.0), 2 steps since -> reuse
    assert gate(1.0) is True  # 3 steps since -> staleness cap fires (state-development bound)
    assert (
        gate(1.4) is True
    )  # +40% > 25% vs last-refresh 1.0 -> β-move fires (the anti-stall trigger)
    assert gate(1.4) is False  # unchanged, 1 step since -> reuse


def test_materialize_gate_fires_on_drift_and_the_step_cap() -> None:
    """The β-diagonal split's materialize gate re-materializes the Jacobian only when the coefficient has
    drifted past the threshold since the last materialize, or at the step cap -- so the expensive full
    re-probe is reserved for a genuinely stale Jacobian and the cheap shift-only refresh carries the rest.

    Pure logic with an injected synthetic drift measure (``drift = |state - reference|``): the gate's
    decision -- first-call seeding without a redundant materialize, drift-move, step-cap, and re-basing the
    reference at every materialize -- is the whole novelty; the materialize itself is shared machinery.
    """
    import jax.numpy as jnp
    from aquaflux.turbulence.coupled import _materialize_gate

    def drift_factory(reference):
        ref = float(reference)
        return lambda state: abs(float(state) - ref)

    # Drift only: seed at the first state (no redundant materialize), then fire on a >0.5 move, re-basing.
    gate = _materialize_gate(drift_factory, materialize_drift=0.5, materialize_every=None)
    assert (
        gate(jnp.asarray(0.0)) is False
    )  # first call seeds the reference; Jacobian is fresh from build
    assert gate(jnp.asarray(0.3)) is False  # drift 0.3 < 0.5 -> shift-only
    assert (
        gate(jnp.asarray(0.6)) is True
    )  # drift 0.6 > 0.5 -> materialize, re-base reference to 0.6
    assert gate(jnp.asarray(0.7)) is False  # drift 0.1 vs 0.6 -> shift-only (re-based, not vs 0.0)
    assert gate(jnp.asarray(1.2)) is True  # drift 0.6 vs 0.6 -> materialize again

    # Step cap only (no drift trigger): fire every 3rd refresh regardless of state.
    cap = _materialize_gate(drift_factory, materialize_drift=None, materialize_every=3)
    assert [cap(jnp.asarray(0.0)) for _ in range(6)] == [False, False, True, False, False, True]


@pytest.mark.slow
def test_ilut_beta_tracking_refresh_gates_the_in_place_refactor(case) -> None:
    """The precondition_step re-factors the ILUT in place on its first call and skips within the cap.

    Confirms the gate is wired to the real preconditioner: the first call re-factors (a fresh factorization
    object), a second call at the same β within ``refresh_every`` reuses the standing factorization.
    """
    from aquaflux.solve import DualTimeControl

    coupled = case["coupled"]
    flow, k, omega = case["start"]
    state = coupled.pack_state(flow, k, omega)

    dual = coupled_ilut_continuation(coupled, state, ilut_beta=0.05, inner_steps=5)
    active, _ = DualTimeControl(beta_start=0.7).next_step(
        dual, None, None
    )  # ConstantRelaxation(β=0.7)
    precondition_step = ilut_beta_tracking_refresh(coupled, refresh_every=5)

    factors_built = active.shift_policy.preconditioner.factors
    precondition_step(active, state)  # first call -> gate fires, re-factor at (state, β=0.7)
    factors_tracked = active.shift_policy.preconditioner.factors
    assert factors_tracked is not factors_built  # genuinely re-factored at the current β

    precondition_step(
        active, state
    )  # same β, 1 step since -> gate skips, reuse the standing factor
    assert active.shift_policy.preconditioner.factors is factors_tracked


@pytest.mark.slow
def test_ilut_beta_tracking_forward_march_converges_to_the_same_fixed_point(case) -> None:
    """solve_coupled with the ILUT precondition_step + DualTimeControl reaches the block PC's root."""
    from aquaflux.solve import DualTimeControl

    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)

    ilut = coupled_ilut_continuation(
        coupled, reference_state, ilut_beta=0.5, inner_steps=5, inner_tol=1e-3
    )
    flow_l, k_l, _ = solve_coupled(
        coupled,
        flow_ws,
        k_ws,
        omega_ws,
        continuation=ilut,
        step_control=DualTimeControl(beta_start=0.5, beta_min=0.02),
        refresh=RefreshPolicy(precondition_step=ilut_beta_tracking_refresh(coupled)),
        scaled_norm=True,
        max_steps=60,
    )
    flow_b, k_b, _ = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )
    assert float(jnp.linalg.norm(flow_l - flow_b) / jnp.linalg.norm(flow_b)) < 1e-4
    assert float(jnp.linalg.norm(k_l - k_b) / jnp.linalg.norm(k_b)) < 1e-3


@pytest.mark.slow
def test_ilut_solve_converges_and_matches_the_block_preconditioned_solve(case) -> None:
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)

    # Monolithic ILUT continuation, built off the jit path at the cold reference state.
    ilut = coupled_ilut_continuation(coupled, reference_state)
    flow_i, k_i, omega_i = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, continuation=ilut, max_steps=40
    )

    residual_norm = float(
        jnp.linalg.norm(coupled.residual(coupled.pack_state(flow_i, k_i, omega_i)))
    )
    assert residual_norm < 1e-8
    assert float(jnp.min(k_i)) >= 0.0
    assert float(jnp.min(omega_i)) > 0.0
    assert float(jnp.max(k_i)) > 10.0 * float(jnp.min(jnp.abs(k_i)) + 1e-30)  # genuinely turbulent

    # Same fixed point as the block-triangular preconditioner reaches.
    flow_b, k_b, omega_b = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )
    assert float(jnp.linalg.norm(flow_i - flow_b) / jnp.linalg.norm(flow_b)) < 1e-4
    assert float(jnp.linalg.norm(k_i - k_b) / jnp.linalg.norm(k_b)) < 1e-3
    assert float(jnp.linalg.norm(omega_i - omega_b) / jnp.linalg.norm(omega_b)) < 1e-4


@pytest.mark.slow
def test_ilut_adjoint_matches_finite_difference(case) -> None:
    """The coupled implicit-function-theorem adjoint is exact through the ILUT-preconditioned solve.

    The preconditioner is ``stop_gradient``-ed (it only accelerates the Krylov iteration), so the
    gradient is the single transpose solve on the unfrozen coupled residual -- ``jax.grad`` through the
    ILUT solve matches finite differences, exactly as for the block preconditioner. The continuation is
    built once outside ``jax.grad`` on concrete parameters (the factorization must not be traced).
    """
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)
    continuation = coupled_ilut_continuation(coupled, reference_state)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
