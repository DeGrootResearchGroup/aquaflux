"""Unit tests for the observed-march step controls (AlphaTargetingControl).

A step control is stateful but its decision is a pure function of the previous report, so these test
it on synthetic reports with no solve — the same replayability the refresh trigger has. The
load-bearing structural check is that the controlled step differs from the base only in a dynamic β
leaf (a :class:`ConstantRelaxation`), so the eager march stays a compilation-cache hit.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
from aquaflux.solve import (
    AlphaTargetingControl,
    ConstantRelaxation,
    PseudoTransientStep,
    ShiftTerm,
    StepReport,
    SwitchedEvolutionRelaxation,
)


class _TrivialShiftPolicy(eqx.Module):
    def shift_term(self, phi):
        return ShiftTerm(diagonal=jnp.ones_like(phi), make_preconditioner=lambda _relaxation: None)


def _base_step() -> PseudoTransientStep:
    return PseudoTransientStep(
        _TrivialShiftPolicy(), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=2.0)
    )


def _report(alpha: float) -> StepReport:
    return StepReport(step=0, cycles=10, residual_norm=1.0, residual_ratio=1.0, alpha=alpha)


def test_first_step_uses_beta_start() -> None:
    """With no previous report, the control sets β to beta_start and reports it as its state."""
    control = AlphaTargetingControl(beta_start=2.0)
    step, beta = control.next_step(_base_step(), None, None)
    assert beta == 2.0
    assert isinstance(step.relaxation_schedule, ConstantRelaxation)
    assert jnp.allclose(step.relaxation_schedule.beta, 2.0)


def test_a_clipped_step_raises_beta_toward_the_boundary() -> None:
    """α < 1 (the step was clipped) drives β up: β ← β/α, capped."""
    control = AlphaTargetingControl(growth_cap=3.0)
    # α = 0.5 at β = 2 -> β/α = 4, within the ×3 cap (=6), so β becomes 4.
    _, beta = control.next_step(_base_step(), _report(alpha=0.5), 2.0)
    assert jnp.isclose(beta, 4.0)


def test_the_raise_is_capped() -> None:
    """A tiny α cannot fling β to the ceiling in one step — the growth cap bounds it."""
    control = AlphaTargetingControl(growth_cap=3.0)
    # α = 0.01 -> β/α = 200, but the ×3 cap holds it to 6.
    _, beta = control.next_step(_base_step(), _report(alpha=0.01), 2.0)
    assert jnp.isclose(beta, 6.0)


def test_a_full_step_eases_beta_down() -> None:
    """α = 1 (the full step descended) eases β gently, to probe a larger productive step."""
    control = AlphaTargetingControl(ease=1.1)
    _, beta = control.next_step(_base_step(), _report(alpha=1.0), 2.0)
    assert jnp.isclose(beta, 2.0 / 1.1)


def test_beta_is_clamped() -> None:
    control = AlphaTargetingControl(beta_min=0.1, beta_max=5.0, growth_cap=100.0)
    _, high = control.next_step(_base_step(), _report(alpha=0.01), 4.0)  # would exceed beta_max
    assert high == 5.0
    low_control = AlphaTargetingControl(beta_min=0.5, ease=10.0)
    _, low = low_control.next_step(_base_step(), _report(alpha=1.0), 1.0)  # 1/10 < beta_min
    assert low == 0.5


def test_controlled_step_differs_from_base_only_in_a_dynamic_beta_leaf() -> None:
    """The control replaces just the schedule with ConstantRelaxation(β) on a dynamic leaf.

    This is what keeps the eager march a compilation-cache hit across steps: two controlled steps have
    identical static structure (a ConstantRelaxation schedule) and differ only in the traced β value.
    """
    control = AlphaTargetingControl()
    step_a, _ = control.next_step(_base_step(), _report(alpha=0.5), 2.0)
    step_b, _ = control.next_step(_base_step(), _report(alpha=1.0), 2.0)
    # Same static (non-array) structure ...
    static_a = eqx.partition(step_a, eqx.is_array)[1]
    static_b = eqx.partition(step_b, eqx.is_array)[1]
    assert eqx.tree_equal(static_a, static_b) is True
    # ... but different β values (the dynamic leaf).
    assert not jnp.allclose(step_a.relaxation_schedule.beta, step_b.relaxation_schedule.beta)
    # The non-schedule configuration of the base step is untouched.
    assert step_a.max_escalations == _base_step().max_escalations
