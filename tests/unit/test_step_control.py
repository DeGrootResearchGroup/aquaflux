"""Unit tests for the observed-march step controls.

A step control is stateful but its decision is a pure function of the previous report, so these test
it on synthetic reports with no solve — the same replayability the refresh trigger has. The
load-bearing structural check is that the controlled step differs from the base only in a dynamic β
leaf (a :class:`ConstantRelaxation`), so the eager march stays a compilation-cache hit.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import pytest
from aquaflux.solve import (
    CflResidualDualTimeControl,
    ConstantRelaxation,
    DualTimeControl,
    DualTimeStep,
    PseudoTransientStep,
    ResidualRatioDualTimeControl,
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


def _dual_step() -> DualTimeStep:
    return DualTimeStep(
        _TrivialShiftPolicy(),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=2.0),
        inner_steps=4,
    )


def test_dual_time_control_first_step_uses_beta_start() -> None:
    """With no previous report, DualTimeControl sets β to beta_start and carries it as state."""
    control = DualTimeControl(beta_start=2.0)
    step, (beta, memo) = control.next_step(_dual_step(), None, None)
    assert (beta, memo) == (2.0, None)  # memoryless rule: alpha alone drives it
    assert isinstance(step.relaxation_schedule, ConstantRelaxation)
    assert jnp.allclose(step.relaxation_schedule.beta, 2.0)


def test_dual_time_control_grows_the_timestep_when_comfortable() -> None:
    """α ≥ grow_above (inner loop comfortable) grows the pseudo-timestep: β ← β/grow."""
    control = DualTimeControl(grow=1.5, grow_above=0.5)
    _, (beta, _) = control.next_step(_dual_step(), _report(alpha=1.0), (1.5, None))
    assert jnp.isclose(beta, 1.0)  # 1.5 / 1.5


def test_dual_time_control_backs_off_when_clipped() -> None:
    """α < backoff_below (an inner step clipped hard) shrinks the pseudo-timestep: β ← β*backoff."""
    control = DualTimeControl(backoff=2.0, backoff_below=0.25)
    _, (beta, _) = control.next_step(_dual_step(), _report(alpha=0.1), (0.5, None))
    assert jnp.isclose(beta, 1.0)  # 0.5 * 2.0


def test_dual_time_control_holds_beta_across_a_refresh() -> None:
    """The first step of a refresh segment (previous is None, β carried) holds β, not beta_start.

    Without this the Courant ramp would reset to beta_start on every preconditioner refresh -- which
    fires every few steps on a developing flow -- so the pseudo-timestep would sawtooth and never grow.
    """
    control = DualTimeControl(beta_start=2.0)
    _, (beta, _) = control.next_step(_dual_step(), None, (0.12, None))  # carried, segment boundary
    assert jnp.isclose(beta, 0.12)  # continues the ramp, does not reset to beta_start


def test_dual_time_control_holds_in_the_dead_band() -> None:
    """A moderate clip (backoff_below ≤ α < grow_above) neither grows nor shrinks β."""
    control = DualTimeControl(grow_above=0.5, backoff_below=0.25)
    _, (beta, _) = control.next_step(_dual_step(), _report(alpha=0.4), (0.8, None))
    assert jnp.isclose(beta, 0.8)


def test_dual_time_control_clamps_beta() -> None:
    control = DualTimeControl(beta_min=0.1, beta_max=4.0, backoff=100.0)
    _, (high, _) = control.next_step(_dual_step(), _report(alpha=0.0), (3.0, None))  # over beta_max
    assert high == 4.0
    low_control = DualTimeControl(beta_min=0.5, grow=10.0)
    _, (low, _) = low_control.next_step(_dual_step(), _report(alpha=1.0), (1.0, None))  # < beta_min
    assert low == 0.5


def test_dual_time_control_step_differs_from_base_only_in_a_dynamic_beta_leaf() -> None:
    """The control replaces just the schedule with ConstantRelaxation(β) on a dynamic leaf.

    Two controlled steps share static structure and differ only in the traced β, so the eager march
    does not recompile per step.
    """
    control = DualTimeControl()
    step_a, _ = control.next_step(_dual_step(), _report(alpha=1.0), (1.0, None))
    step_b, _ = control.next_step(_dual_step(), _report(alpha=0.1), (1.0, None))
    static_a = eqx.partition(step_a, eqx.is_array)[1]
    static_b = eqx.partition(step_b, eqx.is_array)[1]
    assert eqx.tree_equal(static_a, static_b) is True
    assert not jnp.allclose(step_a.relaxation_schedule.beta, step_b.relaxation_schedule.beta)
    # The non-schedule configuration (e.g. inner_steps) is untouched.
    assert step_a.inner_steps == _dual_step().inner_steps


def _res_report(alpha: float, residual: float) -> StepReport:
    return StepReport(
        step=0, cycles=6, residual_norm=residual, residual_ratio=residual, alpha=alpha
    )


def test_residual_ratio_first_step_uses_beta_start() -> None:
    """With no state, β is beta_start and the carried state is (beta_start, no residual yet)."""
    control = ResidualRatioDualTimeControl(beta_start=0.5)
    step, state = control.next_step(_dual_step(), None, None)
    assert state == (0.5, None)
    assert float(step.relaxation_schedule.beta) == 0.5


def test_residual_ratio_grows_the_timestep_when_the_residual_falls() -> None:
    """A residual drop (ratio < 1) lowers β (grows the pseudo-timestep): β ← β·ratio."""
    control = ResidualRatioDualTimeControl(beta_start=0.5, max_change=2.0, backoff_below=0.0)
    _, (beta, prev) = control.next_step(
        _dual_step(), _res_report(alpha=1.0, residual=0.9), (0.5, 1.0)
    )
    assert beta == 0.45  # 0.5 * (0.9 / 1.0)
    assert prev == 0.9


def test_residual_ratio_shrinks_the_timestep_when_the_residual_rises() -> None:
    """A residual rise (ratio > 1) raises β (shrinks the pseudo-timestep) -- the anti-runaway property."""
    control = ResidualRatioDualTimeControl(beta_start=0.5, max_change=2.0, backoff_below=0.0)
    _, (beta, _prev) = control.next_step(
        _dual_step(), _res_report(alpha=1.0, residual=1.2), (0.5, 1.0)
    )
    assert beta == pytest.approx(0.6)  # 0.5 * (1.2 / 1.0)


def test_residual_ratio_change_is_clipped() -> None:
    """One anomalous ratio cannot fling the timestep: the change is clipped to [1/max_change, max_change]."""
    control = ResidualRatioDualTimeControl(
        beta_start=0.5, max_change=1.3, backoff_below=0.0, beta_max=10.0
    )
    _, (beta, _p) = control.next_step(
        _dual_step(), _res_report(alpha=1.0, residual=5.0), (0.5, 1.0)
    )
    assert beta == pytest.approx(0.65)  # clipped to 0.5 * 1.3, not 0.5 * 5


def test_residual_ratio_hard_inner_clip_forces_a_shrink() -> None:
    """An inner-loop clip (α < backoff_below) shrinks the step regardless of the residual ratio."""
    control = ResidualRatioDualTimeControl(
        beta_start=0.5, max_change=1.3, backoff=2.0, backoff_below=0.6
    )
    # Residual flat (ratio 1) but the inner loop clipped hard -> β multiplied by backoff.
    _, (beta, _p) = control.next_step(
        _dual_step(), _res_report(alpha=0.5, residual=1.0), (0.5, 1.0)
    )
    assert beta == pytest.approx(1.0)  # 0.5 * 1 (ratio) * 2 (backoff)


def test_residual_ratio_holds_beta_across_a_refresh() -> None:
    """The first step of a refresh segment (previous is None, state carried) holds β, not beta_start."""
    control = ResidualRatioDualTimeControl(beta_start=0.5)
    _, (beta, prev) = control.next_step(_dual_step(), None, (0.12, 0.3))
    assert beta == 0.12  # continues the ramp, does not reset to beta_start
    assert prev == 0.3


def test_residual_ratio_clamps_beta() -> None:
    """β is clamped to [beta_min, beta_max] after the update."""
    low = ResidualRatioDualTimeControl(beta_min=0.1, max_change=10.0, backoff_below=0.0)
    _, (beta, _p) = low.next_step(_dual_step(), _res_report(alpha=1.0, residual=0.01), (0.2, 1.0))
    assert beta == 0.1  # 0.2 * 0.1 clipped-change would go below beta_min


def _cfl_res_control(**kwargs: float) -> CflResidualDualTimeControl:
    base = dict(
        grow=1.5, backoff=2.0, grow_above=0.5, backoff_below=0.25, hold_ratio=1.05, rise_ratio=1.10
    )
    base.update(kwargs)
    return CflResidualDualTimeControl(**base)


def test_cfl_residual_first_step_uses_beta_start() -> None:
    """With no state, β is beta_start and the carried state is (beta_start, no residual yet)."""
    control = _cfl_res_control(beta_start=0.5)
    step, state = control.next_step(_dual_step(), None, None)
    assert state == (0.5, None)
    assert float(step.relaxation_schedule.beta) == 0.5


def test_cfl_residual_grows_on_alpha_when_the_residual_is_flat() -> None:
    """The point of the combination: α comfortable + a FLAT residual (ratio ≤ hold_ratio) still grows the
    step (β ← β/grow), where the residual-only rule would stall on the β×travel plateau."""
    control = _cfl_res_control(beta_start=0.5)
    _, (beta, prev) = control.next_step(
        _dual_step(),
        _res_report(alpha=1.0, residual=1.0),
        (0.6, 1.0),  # ratio = 1.0, flat
    )
    assert beta == pytest.approx(0.4)  # 0.6 / 1.5, grown on α despite no residual drop
    assert prev == 1.0


def test_cfl_residual_brakes_on_a_rising_residual_even_at_full_alpha() -> None:
    """The overshoot governor α lacks: a rising residual (ratio > rise_ratio) shrinks the step even when
    the inner loop is perfectly comfortable (α = 1) -- the case that NaNs the α-only control."""
    control = _cfl_res_control(beta_start=0.5)
    _, (beta, _prev) = control.next_step(
        _dual_step(),
        _res_report(alpha=1.0, residual=1.2),
        (0.5, 1.0),  # ratio = 1.2 > rise_ratio
    )
    assert beta == pytest.approx(1.0)  # 0.5 * backoff(2.0), braked despite α = 1


def test_cfl_residual_brakes_on_an_inner_clip() -> None:
    """The local wall: a hard inner clip (α < backoff_below) shrinks the step regardless of the residual."""
    control = _cfl_res_control(beta_start=0.5)
    _, (beta, _prev) = control.next_step(
        _dual_step(),
        _res_report(alpha=0.1, residual=1.0),
        (0.5, 1.0),  # α clipped, residual flat
    )
    assert beta == pytest.approx(1.0)  # 0.5 * backoff(2.0)


def test_cfl_residual_holds_in_the_ratio_band() -> None:
    """Between hold_ratio and rise_ratio the step holds -- the band that keeps a noisy plateau from
    oscillating between grow and brake."""
    control = _cfl_res_control(beta_start=0.5)
    _, (beta, _prev) = control.next_step(
        _dual_step(),
        _res_report(alpha=1.0, residual=1.07),
        (0.5, 1.0),  # 1.05 < 1.07 ≤ 1.10
    )
    assert beta == pytest.approx(0.5)  # unchanged


def test_cfl_residual_holds_beta_across_a_refresh() -> None:
    """The first step of a refresh segment (previous is None, state carried) holds β, not beta_start."""
    control = _cfl_res_control(beta_start=2.0)
    _, (beta, prev) = control.next_step(_dual_step(), None, (0.12, 0.3))
    assert beta == 0.12
    assert prev == 0.3


def test_cfl_residual_clamps_beta() -> None:
    """β is clamped to [beta_min, beta_max] after the update."""
    control = _cfl_res_control(beta_start=0.5, beta_min=0.1, grow=10.0)
    _, (beta, _p) = control.next_step(
        _dual_step(),
        _res_report(alpha=1.0, residual=1.0),
        (0.2, 1.0),  # grow would go below beta_min
    )
    assert beta == 0.1


def test_residual_ratio_step_differs_from_base_only_in_a_dynamic_beta_leaf() -> None:
    """The control replaces just the schedule with ConstantRelaxation(β) on a dynamic leaf."""
    control = ResidualRatioDualTimeControl()
    step_a, _ = control.next_step(_dual_step(), _res_report(alpha=1.0, residual=0.9), (0.5, 1.0))
    step_b, _ = control.next_step(_dual_step(), _res_report(alpha=1.0, residual=1.2), (0.5, 1.0))
    static_a = eqx.partition(step_a, eqx.is_array)[1]
    static_b = eqx.partition(step_b, eqx.is_array)[1]
    assert eqx.tree_equal(static_a, static_b) is True
    assert not jnp.allclose(step_a.relaxation_schedule.beta, step_b.relaxation_schedule.beta)


def test_step_report_restart_cycles_and_matvecs_correct_the_num_steps_offset() -> None:
    """`restart_cycles` strips lineax's +2-per-inner-solve offset; `matvecs` scales by the restart.

    lineax's `num_steps` (StepReport.cycles) reports 3 for any solve within one 120-restart cycle and is
    summed over the inner Newton iterations for a dual-time step, so the raw number conflates the
    nonlinear inner work with the linear cost. The offset-corrected accessors report them honestly.
    """
    single = StepReport(step=0, cycles=3, residual_norm=1.0, residual_ratio=1.0, alpha=1.0)
    assert single.inner_iterations == 1  # default: a single-step march has no inner loop
    assert single.restart_cycles == 1  # 3 - 2*1: one ideal restart cycle
    assert single.matvecs(120) == 120

    dual = StepReport(
        step=0, cycles=6, residual_norm=1.0, residual_ratio=1.0, alpha=1.0, inner_iterations=2
    )
    assert dual.restart_cycles == 2  # 6 - 2*2: two inner iters, each an ideal 1-cycle solve
    assert dual.matvecs(120) == 240

    rejected = StepReport(step=0, cycles=0, residual_norm=1.0, residual_ratio=1.0, alpha=1.0)
    assert rejected.restart_cycles == 0  # a no-measurement step stays 0, not negative


def test_carry_beta_seeds_the_carried_state() -> None:
    """`carry_beta` replaces the control's carried β with an externally-chosen (escalated) value, keeping
    any carried residual so the ratio signal is unbroken -- the hook `forward_march` uses to carry an
    escalated β forward so a persistently hard region is not re-escalated every step."""
    from aquaflux.solve import (
        CflResidualDualTimeControl,
        DualTimeControl,
        ResidualRatioDualTimeControl,
    )

    # One implementation on the shared base serves all three, so none can be missing it -- which is
    # exactly what went wrong before: one control had no `carry_beta` at all, and `forward_march`
    # probes for it with `hasattr`, so its escalation feedback was dropped in silence.
    for ctrl in (DualTimeControl(), CflResidualDualTimeControl(), ResidualRatioDualTimeControl()):
        assert ctrl.carry_beta((0.02, 3.5), 0.16) == (0.16, 3.5)  # β replaced, memo preserved
        assert ctrl.carry_beta(None, 0.16) == (0.16, None)  # first-step state has no memo yet


def test_the_combined_control_reduces_to_the_courant_one_at_infinite_ratio_thresholds() -> None:
    """The superset claim, checked rather than asserted in prose.

    `CflResidualDualTimeControl` is documented as `DualTimeControl` plus a residual-rise brake. With
    both ratio thresholds at infinity neither ratio clause can fire, so the two must agree step for
    step -- and if someone edits one rule without the other, this is what says so. Driven over a
    sequence that exercises all three bands (grow, hold, back off) and a rising residual, which is the
    one input that would separate them if the reduction were not exact.
    """
    from aquaflux.solve import CflResidualDualTimeControl, DualTimeControl

    shared = dict(beta_start=1.0, grow=1.5, backoff=2.0, grow_above=0.5, backoff_below=0.25)
    courant = DualTimeControl(**shared)
    combined = CflResidualDualTimeControl(
        **shared, hold_ratio=float("inf"), rise_ratio=float("inf")
    )

    courant_state = combined_state = None
    for alpha, residual in ((1.0, 1.0), (1.0, 5.0), (0.4, 4.0), (0.1, 9.0), (1.0, 0.1)):
        report = _res_report(alpha=alpha, residual=residual)
        _, courant_state = courant.next_step(_dual_step(), report, courant_state)
        _, combined_state = combined.next_step(_dual_step(), report, combined_state)
        assert courant_state[0] == pytest.approx(combined_state[0])


def test_every_control_shares_one_body_and_supplies_only_its_rule() -> None:
    """The seam itself: a capability added to the base reaches all three controls.

    Structural rather than behavioural, because the failure it guards against is a future divergence --
    someone re-adding a private `next_step`, `carry_beta` or clamp to one class, which is precisely how
    these drifted before: `carry_beta` ended up byte-identical in two of them and absent from a third,
    and one class reset β at a refresh boundary where the others held it.
    """
    from aquaflux.solve import (
        CflResidualDualTimeControl,
        DualTimeControl,
        ResidualRatioDualTimeControl,
        ShiftStrengthControl,
    )

    for cls in (DualTimeControl, ResidualRatioDualTimeControl, CflResidualDualTimeControl):
        assert issubclass(cls, ShiftStrengthControl)
        assert set(vars(cls)) & {"next_step", "carry_beta", "_clamp"} == set(), (
            f"{cls.__name__} overrides shared behaviour the base owns"
        )
        assert "_adapt" in vars(cls), f"{cls.__name__} must supply its own rule"


def test_a_refresh_boundary_holds_beta_for_every_control() -> None:
    """One rule, so no control can quietly reset the ramp at a segment boundary.

    A refresh fires every few steps on a developing flow, so a control that resets to `beta_start`
    there sawtooths β and the pseudo-timestep never grows. That was fixed once for the Courant control
    and, being a per-class `next_step`, never reached the others.
    """
    from aquaflux.solve import (
        CflResidualDualTimeControl,
        DualTimeControl,
        ResidualRatioDualTimeControl,
    )

    for ctrl in (
        DualTimeControl(beta_start=2.0),
        ResidualRatioDualTimeControl(beta_start=0.5),
        CflResidualDualTimeControl(beta_start=2.0),
    ):
        _, (beta, memo) = ctrl.next_step(_dual_step(), None, (0.12, 0.3))
        assert beta == 0.12, f"{type(ctrl).__name__} reset β at a refresh boundary"
        assert memo == 0.3, f"{type(ctrl).__name__} dropped its memo at a refresh boundary"
