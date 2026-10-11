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
    DampedNewtonStep,
    DualTimeControl,
    DualTimeStep,
    PseudoTransientStep,
    ResidualRatioDualTimeControl,
    ShiftTerm,
    StepReport,
    SwitchedEvolutionRelaxation,
    newton_march,
)


class _TrivialShiftPolicy(eqx.Module):
    def shift_term(self, phi, residual=None):
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


def test_a_control_resumed_at_a_shift_holds_it_for_the_first_step_then_adapts_from_it() -> None:
    """The resumed march opens at the shift the interrupted one had reached, not at ``beta_start``.

    Wrong answers this catches: a resume that reseeds ``beta_start`` (the ramp walked again), one that
    adapts the shift before running a step it has no report for, and one that keeps a stale memo.
    """
    control = DualTimeControl(beta_start=2.0, beta_min=0.02, beta_max=4.0, grow=1.5)
    state = control.resumed_at(0.3)
    assert state == (0.3, None)
    first, state = control.next_step(_dual_step(), None, state)
    assert jnp.allclose(first.relaxation_schedule.beta, 0.3)
    # With a step to adapt from it grows the timestep as it would have, a comfortable alpha lowering beta.
    second, _ = control.next_step(_dual_step(), _report(alpha=1.0), state)
    assert jnp.allclose(second.relaxation_schedule.beta, 0.3 / 1.5)


@pytest.mark.parametrize(("shift", "held"), [(1.0e-6, 0.02), (50.0, 4.0)])
def test_a_resumed_shift_is_held_inside_this_controls_bounds(shift, held) -> None:
    """The resumed march's control may be bounded differently from the interrupted one's."""
    assert DualTimeControl(beta_min=0.02, beta_max=4.0).resumed_at(shift) == (held, None)


def test_a_residual_keyed_control_resumes_with_no_remembered_residual() -> None:
    """The residual its ratio is formed against belongs to the interrupted march; alpha alone drives
    the first adaptation, the path a rule with no reference already takes."""
    control = ResidualRatioDualTimeControl()
    assert control.resumed_at(0.3) == (0.3, None)


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


def test_step_report_restart_cycles_corrects_the_num_steps_offset() -> None:
    """`restart_cycles` strips lineax's +2-per-inner-solve offset.

    lineax's `num_steps` (StepReport.cycles) reports 3 for any solve within one 120-restart cycle and is
    summed over the inner Newton iterations for a dual-time step, so the raw number conflates the
    nonlinear inner work with the linear cost. The offset-corrected accessors report them honestly.
    """
    single = StepReport(step=0, cycles=3, residual_norm=1.0, residual_ratio=1.0, alpha=1.0)
    assert single.inner_iterations == 1  # default: a single-step march has no inner loop
    assert single.restart_cycles == 1  # 3 - 2*1: one ideal restart cycle

    dual = StepReport(
        step=0, cycles=6, residual_norm=1.0, residual_ratio=1.0, alpha=1.0, inner_iterations=2
    )
    assert dual.restart_cycles == 2  # 6 - 2*2: two inner iters, each an ideal 1-cycle solve

    rejected = StepReport(step=0, cycles=0, residual_norm=1.0, residual_ratio=1.0, alpha=1.0)
    assert rejected.restart_cycles == 0  # a no-measurement step stays 0, not negative


def test_carry_beta_seeds_the_carried_state() -> None:
    """`carry_beta` replaces the control's carried β with an externally-chosen (escalated) value, keeping
    any carried residual so the ratio signal is unbroken -- the hook `newton_march` uses to carry an
    escalated β forward so a persistently hard region is not re-escalated every step."""
    from aquaflux.solve import (
        CflResidualDualTimeControl,
        DualTimeControl,
        ResidualRatioDualTimeControl,
    )

    # One implementation on the shared base serves all three, so none can be missing it -- which is
    # exactly what went wrong before: one control had no `carry_beta` at all, and `newton_march`
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


@pytest.mark.parametrize(
    "control", [DualTimeControl(), ResidualRatioDualTimeControl(), CflResidualDualTimeControl()]
)
def test_a_shift_control_refuses_a_step_with_no_shift_before_the_march_takes_a_step(
    control,
) -> None:
    """A damped-Newton step has no shift to drive, so the control says so rather than failing inside.

    The march calls the control before its first step, so the refusal arrives there. Until the control
    checked, the swap reached for a field the step does not have and the march died on an
    ``AttributeError`` naming neither the control nor what it needs.
    """
    calls = []

    def residual(phi):
        calls.append(phi)
        return phi**3 - 1.0

    with pytest.raises(TypeError, match=r"DampedNewtonStep has none"):
        newton_march(
            DampedNewtonStep(),
            residual,
            jnp.array([2.0]),
            max_steps=3,
            rtol=0.0,
            atol=1e-12,
            step_control=control,
        )
    # Only the march's own opening measurement: no step was taken before the refusal.
    assert len(calls) == 1


# --------------------------------------------------------------------------------------------------
# Releasing the floor once the march has settled at the target (`release_floor`).
# --------------------------------------------------------------------------------------------------

_FLOOR, _RELEASE = 0.005, 1.0e-4


def _releasing(kind, **kwargs):
    """Each control, at a floor of 0.005 and releasing to 1e-4 unless told otherwise."""
    settings = dict(beta_start=0.5, beta_min=_FLOOR, release_floor=_RELEASE) | kwargs
    return kind(**settings)


def _settled(shift: float, alpha: float, residual: float = 0.5, arrived: bool = True) -> StepReport:
    """A step that ran at ``shift`` with line-search factor ``alpha``, its residual half its memo's."""
    return StepReport(
        step=20,
        cycles=10,
        residual_norm=residual,
        residual_ratio=residual,
        alpha=alpha,
        shift=shift,
        arrived=arrived,
    )


_CONTROLS = (DualTimeControl, ResidualRatioDualTimeControl, CflResidualDualTimeControl)


@pytest.mark.parametrize("kind", _CONTROLS)
def test_a_full_step_at_the_floor_on_the_target_releases_the_shift_for_every_control(kind) -> None:
    """The release is the base's, so every control reaches it -- and none does without asking for it.

    Wrong answers this catches: a release written into one rule only, a release that ignores its
    setting (the unset control must stay at the floor), and one that lowers β by a rule step instead of
    dropping it to ``release_floor``.
    """
    released = _releasing(kind)
    step, (beta, _memo) = released.next_step(_dual_step(), _settled(_FLOOR, 1.0), (_FLOOR, 1.0))
    assert beta == _RELEASE
    assert jnp.allclose(step.relaxation_schedule.beta, _RELEASE)
    unset = _releasing(kind, release_floor=None)
    _step, (beta, _memo) = unset.next_step(_dual_step(), _settled(_FLOOR, 1.0), (_FLOOR, 1.0))
    assert beta == _FLOOR


@pytest.mark.parametrize(
    "previous",
    [
        pytest.param(_settled(_FLOOR, 1.0, arrived=False), id="a station on the way to the target"),
        pytest.param(_settled(_FLOOR, 0.5), id="a clipped step at the floor"),
        pytest.param(_settled(0.01, 1.0), id="a full step above the floor"),
    ],
)
def test_the_shift_is_not_released_until_the_target_step_at_the_floor_was_full_length(
    previous,
) -> None:
    """Each of the three conditions is necessary on its own.

    Measured on pitzDaily, a zero-shift step from the state the viscosity ramp arrived at did not
    descend at all, and one from the state after the first full-length step at the floor reached the
    stopping tolerance -- so neither arrival alone nor a clipped step at the floor licenses it.
    """
    control = _releasing(CflResidualDualTimeControl)
    _step, (beta, _memo) = control.next_step(_dual_step(), previous, (previous.shift, 1.0))
    assert beta >= _FLOOR


def test_a_released_march_stays_released_while_the_rule_does_not_back_off() -> None:
    """A released step need not be full-length to stay released: one at alpha 0.3 (the rule's dead band)
    or 0.5 cut the pitzDaily residual 60-fold. Only a back-off sends it back to the floor.

    Wrong answer caught: a release that holds for one step and then snaps back to ``beta_min``.
    """
    control = _releasing(CflResidualDualTimeControl)
    for alpha in (0.3, 0.5):
        _step, (beta, _memo) = control.next_step(
            _dual_step(), _settled(_RELEASE, alpha), (_RELEASE, 1.0)
        )
        assert beta == _RELEASE


@pytest.mark.parametrize(
    "previous",
    [
        pytest.param(_settled(_RELEASE, 0.1), id="a clipped released step"),
        pytest.param(
            _settled(_RELEASE, 1.0, residual=2.0), id="a released step whose residual rose"
        ),
    ],
)
def test_a_released_march_returns_to_the_floor_when_the_rule_backs_off(previous) -> None:
    """The way back is the rule's own back-off, raised to the floor rather than doubled from the release.

    Wrong answers caught: a release that persists through a clipped or diverging step (the residual
    case also has alpha 1, so it would re-enter the release were the back-off not checked first), and a
    return that only doubles the released shift, which would take six bad steps to climb back.
    """
    control = _releasing(CflResidualDualTimeControl)
    _step, (beta, _memo) = control.next_step(_dual_step(), previous, (_RELEASE, 1.0))
    assert beta == _FLOOR


def test_without_a_release_the_rule_never_runs_below_its_floor() -> None:
    """The lowered clamp must not leak: before the release fires, β stops at ``beta_min`` as it always did."""
    control = _releasing(DualTimeControl)
    state = (0.006, None)
    _step, (beta, _memo) = control.next_step(_dual_step(), _settled(0.006, 1.0), state)
    assert beta == _FLOOR


@pytest.mark.parametrize(
    ("shift", "alpha", "settle_alpha", "expected"),
    [
        pytest.param(_FLOOR, 1.0, 1.0, True, id="full length at the floor"),
        pytest.param(_FLOOR * 0.5, 1.0, 1.0, True, id="full length below the floor"),
        pytest.param(_FLOOR * 1.01, 1.0, 1.0, False, id="full length just above the floor"),
        pytest.param(_FLOOR, 0.5, 1.0, False, id="clipped at the floor"),
        pytest.param(_FLOOR, 0.5, 0.5, True, id="clipped to the settle alpha"),
    ],
)
def test_settled_is_a_step_at_the_floor_that_reached_the_settle_alpha(
    shift, alpha, settle_alpha, expected
) -> None:
    """One definition, read by the release on the target and by a ramp that ends on it.

    It looks at the step's shift and line-search factor only -- not at whether the step was on the
    target, which is the release's own further condition and not part of being settled.
    """
    control = _releasing(DualTimeControl, settle_alpha=settle_alpha)
    assert control.settled(_settled(shift, alpha, arrived=False)) is expected
    assert control.settled(_settled(shift, alpha, arrived=True)) is expected


@pytest.mark.parametrize(
    ("settings", "names"),
    [
        (dict(release_floor=0.0), "release_floor"),
        (dict(release_floor=_FLOOR), "release_floor"),
        (dict(settle_alpha=0.0), "settle_alpha"),
        (dict(settle_alpha=1.5), "settle_alpha"),
    ],
)
def test_a_release_outside_its_range_is_refused(settings, names) -> None:
    with pytest.raises(ValueError, match=names):
        _releasing(DualTimeControl, **settings)
