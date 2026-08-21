"""Integration: the monolithic-AMG-preconditioned coupled RANS Newton solve on a turbulent channel.

The coupled continuation's block-triangular SIMPLE preconditioner is replaced by a single
algebraic-multigrid V-cycle of the assembled coupled Jacobian (:func:`coupled_amg_continuation`) -- the
scaling path for large three-dimensional meshes, where the complete LU's fill is out of memory. These
check the two properties that make it a usable drop-in: handed to ``solve_coupled`` it converges the
monolithic Newton to the **same** fixed point the block preconditioner reaches, and -- built once
outside ``jax.grad`` on concrete parameters --
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

from tests.integration.test_coupled_lu import PRECONDITIONER, _channel

#: One extra level of smoother fill for this fixture, and why it is not the library default.
#:
#: The V-cycle's level smoother is a stationary incomplete-LU sweep, and an incomplete factorization
#: takes its fill pattern from the entries that are **stored**, not from the ones that are nonzero --
#: which is why ``AmgVCycle._live`` drops exactly-zero entries before handing the operator over.
#:
#: This channel starts from ``hybrid_initialize``, whose wall-normal velocity is zero to rounding, so
#: the wall-normal face mass fluxes are ~0 and the first-order upwind switch on the ``omega`` transport
#: sits exactly on its kink. At that state ~24 500 of the block-stencil positions carry *identically*
#: zero coupling, so pruning removes them and the fine-level ILU(1) pattern loses the fill it was
#: relying on: the V-cycle goes from **1 restart cycle to ~39**, and the Newton march wanders and
#: stalls instead of converging. One more level of fill restores it (measured on this fixture: fill 1
#: fails at 39-43 cycles; fill 2 and fill 3 both converge in 24 steps at 1 cycle, and raising the
#: smoother *sweeps* instead does not help at all -- it is the fill that was lost, not the effort).
#:
#: Deliberately set here rather than in the builder: the flagship three-dimensional cases run the
#: ILU(1) default on operators whose couplings are not degenerate, and moving a shipped default to suit
#: a small fixture would be a change to them measured on something else entirely.
SMOOTHER_FILL = 2


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

    amg = coupled_amg_continuation(coupled, reference_state, smoother_fill_levels=SMOOTHER_FILL)
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
    continuation = coupled_amg_continuation(
        coupled, reference_state, smoother_fill_levels=SMOOTHER_FILL
    )

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
        lambda _self, _mv, _plan, shift, **_kw: seen.__setitem__("shift", np.asarray(shift)),
    )

    amg_beta_tracking_refresh(coupled, beta_floor=floor)(active, state)

    diagonal = np.asarray(active.shift_policy.base.shift_term(state).diagonal)
    assert np.allclose(seen["shift"], floor * diagonal)  # built at the FLOOR, not at β
    assert not np.allclose(seen["shift"], beta * diagonal)
    # ...and the march's own shift strength is untouched: the operator still carries the small β.
    assert float(active.relaxation_schedule.beta) == pytest.approx(beta)


def test_inner_refresh_rebuilds_at_the_iterate_it_is_handed(case, monkeypatch) -> None:
    """The mid-step hook materializes where the inner loop actually got to, not at the step's start.

    That is the point of refreshing inside a step at all: the march's expensive inner solves are
    stale-preconditioner effects, and rebuilding at the step's start would reproduce the staleness it is
    meant to remove. *When* it fires is the dual-time loop's decision (``refresh_on_cycles``), so that
    one rule both triggers the refresh and forgives the abort; this asserts only where it builds.
    """
    import numpy as np
    from aquaflux.solve import DualTimeControl
    from aquaflux.turbulence import amg_beta_tracking_refresh

    coupled = case["coupled"]
    flow, k, omega = case["start"]
    state = coupled.pack_state(flow, k, omega)
    dual = coupled_amg_continuation(coupled, state, inner_steps=5)
    active, _ = DualTimeControl(beta_start=0.5).next_step(dual, None, None)

    built_at: list[np.ndarray] = []
    monkeypatch.setattr(
        type(active.shift_policy.preconditioner),
        "refresh_in_place",
        lambda _self, _mv, _plan, shift, **_kw: built_at.append(np.asarray(shift)),
    )
    refresh = amg_beta_tracking_refresh(coupled)
    refresh(active, state)  # the march calls this before each step; it is what binds the hook
    built_at.clear()  # that binding call also does the step's own refresh, which is not under test

    iterate = state * 1.05  # somewhere the inner loop has moved to, away from the step's start
    refresh.refresh_at(iterate)
    assert len(built_at) == 1
    assert np.allclose(
        built_at[0], 0.5 * np.asarray(active.shift_policy.base.shift_term(iterate).diagonal)
    )
    assert not np.allclose(
        built_at[0], 0.5 * np.asarray(active.shift_policy.base.shift_term(state).diagonal)
    )


_RUNG_TRACES: list[int] = []


class _CountingRans(eqx.Module):
    """The coupled assembler, wrapped so each TRACE of its residual is countable.

    A bound method of an ``equinox.Module`` is a pytree, exactly as ``coupled.residual`` is, so this
    wrapper has the same cache-key behaviour as the real thing while making compilations observable --
    which ``equinox``'s jit wrapper exposes no handle for.
    """

    inner: CoupledRANS

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        _RUNG_TRACES.append(1)
        return self.inner.residual(state)


def _continuation_ready(momentum):
    """The same assembler with an ARRAY viscosity -- the precondition a Reynolds continuation needs.

    A Python float is not a JAX array, so it rides on the *static* side of a jitted function and is
    compared by value; a rung that rescales it is a fresh compilation key for everything taking the
    assembler as an argument, the coupled solve included. As an array the rungs differ in a leaf value
    and share the compilation. Left to the caller rather than forced by the library, which deliberately
    keeps property values plain scalars.
    """
    from aquaflux.properties import Constant

    viscosity = momentum.properties.properties["viscosity"]
    return eqx.tree_at(
        lambda m: m.properties.properties["viscosity"],
        momentum,
        Constant(jnp.asarray(viscosity.value)),
        is_leaf=lambda leaf: leaf is viscosity,
    )


@pytest.mark.slow
def test_sharing_one_preconditioner_makes_a_new_rung_a_march_step_cache_hit() -> None:
    """A Reynolds rung that reuses the V-cycle OBJECT must not recompile the coupled solve.

    The preconditioner rides in a static field of the forward step, so it is part of the compiled
    step's cache key and is compared by identity: a rung that fits its own V-cycle hands ``_march_step``
    a new key and pays a full compilation of the coupled solve. That was the largest fixed overhead of
    the three-dimensional march -- the three most expensive steps of every archived run were exactly the
    three rung-first steps, at cycle counts no higher than their cheap ones.

    Sharing the object is necessary and, on its own, **not sufficient**, which is why this test drives
    the real builder rather than comparing two policies. Two further things had to hold, and both are
    exercised here: the shift policy must not carry a block preconditioner it never applies (its
    multigrid coarsening reads the operator's values, so its array shapes moved with the viscosity), and
    the viscosity must be an array rather than a float (see :func:`_continuation_ready`).
    """
    from aquaflux.solve.march import _march_step
    from aquaflux.turbulence import CoupledJacobianProbe, hybrid_initialize

    momentum, turbulence = _channel()
    momentum = _continuation_ready(momentum)
    coupled = CoupledRANS.build(momentum, turbulence)
    state = coupled.pack_state(*hybrid_initialize(momentum, turbulence))
    # The next rung of a ramp: the same case at a tenth of the Reynolds number.
    companion = coupled.with_scaled_molecular_viscosity(10.0)
    probe = CoupledJacobianProbe.build(coupled)

    def build(assembler, preconditioner=None):
        return coupled_amg_continuation(
            assembler, state, inner_steps=2, probe=probe, preconditioner=preconditioner
        )

    def run(assembler, step) -> int:
        before = len(_RUNG_TRACES)
        _march_step(
            step,
            _CountingRans(inner=assembler).residual,
            state,
            jnp.asarray(1.0),
            step.default_solver(),
        )
        return len(_RUNG_TRACES) - before

    _RUNG_TRACES.clear()
    # The control: a fresh V-cycle per rung, which is what the driver used to do.
    assert run(coupled, build(coupled)) > 0  # the first rung compiles either way
    assert run(companion, build(companion)) > 0  # ...and so does the second, for the V-cycle alone

    # Now the same two rungs sharing one V-cycle. The first still compiles (a different object again
    # from the control's), and the second is the assertion this test exists for.
    shared = build(coupled)
    assert run(coupled, shared) > 0
    assert run(companion, build(companion, preconditioner=shared.shift_policy.preconditioner)) == 0
