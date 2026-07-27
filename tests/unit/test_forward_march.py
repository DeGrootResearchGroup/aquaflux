"""Unit tests for the observed forward march and the preconditioner-staleness trigger.

Two things are pinned here. The **trigger** is a pure function of a step history, so it is tested on
synthetic histories with no solve at all -- which is also what makes it calibratable by replaying a
logged march offline. The **march** is tested on a small analytic residual: that it takes the same
path as the traced Newton march when nothing interrupts it, that it stops where an injected trigger
says and does *not* raise for stopping short, and that stepping it repeatedly is a compilation-cache
hit (the property without which an eager march would recompile its linear solve every step).
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.solve import (
    AlphaTargetingControl,
    CoefficientDriftTrigger,
    CycleGrowthTrigger,
    DampedNewtonStep,
    ImplicitNewtonSolver,
    PseudoTransientStep,
    ShiftTerm,
    StepReport,
    SwitchedEvolutionRelaxation,
    forward_march,
)

# Incremented on every *trace* of the residual below, so a test can assert that repeated steps
# reuse a compiled march step instead of retracing it.
_TRACES: list[int] = []


class _Cubic(eqx.Module):
    """The residual ``phi**3 - theta``, whose root is the elementwise cube root of ``theta``.

    A module (not a closure) so it is a pytree: its array leaves ride as dynamic arguments through
    ``filter_jit``, which is what the march requires of a residual.
    """

    theta: jnp.ndarray

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        # Only trace-time invocations are counted: an eager evaluation runs this too, and counting
        # those would confuse "was it compiled again?" with "was it called again?".
        if isinstance(phi, jax.core.Tracer):
            _TRACES.append(1)
        return phi**3 - self.theta


def _report(step: int, cycles: int, ratio: float) -> StepReport:
    """A synthetic report; only ``cycles`` and ``residual_ratio`` drive the trigger."""
    return StepReport(
        step=step, cycles=cycles, residual_norm=ratio, residual_ratio=ratio, alpha=1.0
    )


def _history(cycles: list[int], ratio: float = 1e-3) -> list[StepReport]:
    return [_report(i, c, ratio) for i, c in enumerate(cycles)]


def test_trigger_fires_on_a_sustained_cost_rise_once_the_residual_has_fallen() -> None:
    """The trigger's purpose: a sustained rise in linear-solve cost at a developed state."""
    trigger = CycleGrowthTrigger(growth=2.0, max_residual_ratio=1e-2, warmup=2, patience=2)
    # Cheapest measured step is 10; the last two are >= 2x that, at a ratio inside the gate.
    assert trigger.should_refresh(_history([10, 11, 10, 21, 22], ratio=1e-3))


def test_trigger_does_not_fire_before_the_flow_has_developed() -> None:
    """The residual gate, not the cost, is what keeps an early refresh from firing.

    The cost rise here is identical to the firing case above -- only the residual ratio differs. The
    damping schedule raises the cycle count on its own as it ramps down, so without this gate the
    trigger would fire from damping alone, at a state where rebuilding was measured to *cost* rather
    than pay.
    """
    trigger = CycleGrowthTrigger(growth=2.0, max_residual_ratio=1e-2, warmup=2, patience=2)
    assert not trigger.should_refresh(_history([10, 11, 10, 21, 22], ratio=0.5))


def test_trigger_does_not_fire_on_a_single_expensive_step() -> None:
    """One spike must not buy a rebuild and a recompilation -- that is what ``patience`` is for."""
    trigger = CycleGrowthTrigger(growth=2.0, max_residual_ratio=1e-2, warmup=2, patience=2)
    assert not trigger.should_refresh(_history([10, 11, 10, 12, 40], ratio=1e-3))


def test_trigger_ignores_unmeasured_steps() -> None:
    """A ``0`` count is "no measurement", not "free".

    A pseudo-transient step records its count only on acceptance, so a step whose every damping
    attempt was rejected reports ``0``. Were that allowed to set the running-minimum baseline, every
    later step would count as "grown" and the trigger would latch on permanently.
    """
    trigger = CycleGrowthTrigger(growth=2.0, max_residual_ratio=1e-2, warmup=2, patience=2)
    # A zero among otherwise flat costs must neither set the baseline nor satisfy the growth test.
    assert not trigger.should_refresh(_history([10, 11, 0, 10, 11], ratio=1e-3))
    # A history with no measurement at all can never fire.
    assert not trigger.should_refresh(_history([0, 0, 0, 0, 0], ratio=1e-3))


def test_trigger_respects_the_warmup() -> None:
    """The opening steps run at the largest damping from a fresh preconditioner: not representative."""
    trigger = CycleGrowthTrigger(growth=2.0, max_residual_ratio=1e-2, warmup=5, patience=2)
    assert not trigger.should_refresh(_history([10, 30, 30], ratio=1e-3))


def _march_and_solver_inputs():
    theta = jnp.array([8.0, 27.0, 64.0])
    return _Cubic(theta), jnp.ones_like(theta), jnp.cbrt(theta)


def test_march_without_a_trigger_reaches_the_same_root_as_the_newton_solver() -> None:
    """The eager march and the traced Newton march take the same path on the same problem.

    They share the step strategy, the residual measure, and the tolerance test; only the loop differs
    (a Python ``for`` versus a ``lax.while_loop``). This pins that the one duplicated piece -- the loop
    shell -- has not drifted.
    """
    residual, phi0, root = _march_and_solver_inputs()
    step = DampedNewtonStep(line_search=10)

    marched = forward_march(step, residual, phi0, max_steps=50, rtol=1e-10, atol=1e-12)
    solved = ImplicitNewtonSolver(rtol=1e-10, atol=1e-12, max_steps=50, forward_step=step).solve(
        lambda p, r: r(p), phi0, residual
    )

    assert marched.converged
    assert jnp.allclose(marched.state, root, atol=1e-8)
    assert jnp.allclose(marched.state, solved, atol=1e-8)


def test_march_stops_where_the_trigger_says_without_raising() -> None:
    """A triggered march stops short of a root and reports it -- deliberately without raising.

    This is the contrast with the Newton solver, which raises whenever it ends away from a root
    (its adjoint would otherwise be silently wrong). Stopping short is this march's whole purpose,
    so the guard lives only in the solver that produces the actual result.
    """

    class _AfterTwoSteps(eqx.Module):
        def should_refresh(self, history):
            return len(history) >= 2

    residual, phi0, _ = _march_and_solver_inputs()
    result = forward_march(
        DampedNewtonStep(line_search=10),
        residual,
        phi0,
        max_steps=50,
        rtol=1e-10,
        atol=1e-12,
        trigger=_AfterTwoSteps(),
    )

    assert result.triggered
    assert not result.converged  # stopped short, and said so rather than raising
    assert len(result.reports) == 2


def test_march_reports_every_step_to_an_observer() -> None:
    """The observer sees each step as it happens; the returned history is the same data."""
    residual, phi0, _ = _march_and_solver_inputs()
    seen: list[StepReport] = []
    result = forward_march(
        DampedNewtonStep(line_search=10),
        residual,
        phi0,
        max_steps=6,
        rtol=1e-14,
        atol=1e-16,
        observer=seen.append,
    )

    assert seen == list(result.reports)
    assert [report.step for report in seen] == list(range(len(seen)))
    # The ratio is measured against the march's reference, so it starts at or below 1 and falls.
    assert seen[-1].residual_ratio < seen[0].residual_ratio
    # Every report carries a valid line-search factor (this base line-searches, so α ∈ (0, 1]).
    assert all(0.0 < report.alpha <= 1.0 for report in seen)


class _UnitShiftPolicy(eqx.Module):
    """A trivial pseudo-transient shift policy (unit diagonal, no preconditioner) for a scalar root."""

    def shift_term(self, phi):
        return ShiftTerm(diagonal=jnp.ones_like(phi), make_preconditioner=lambda _relaxation: None)


def test_step_control_drives_the_march_and_stays_a_cache_hit() -> None:
    """A ``step_control`` reshapes the step each iteration, and the controlled march does not retrace.

    The α-targeting control replaces the base step's schedule with a ``ConstantRelaxation`` on a
    dynamic β leaf. Two things must hold: β actually changes across steps (the control is doing
    something), and ``_march_step`` compiles once despite a fresh controlled step object each iteration
    (the load-bearing cache-hit property — a per-step recompile would dominate a real march).
    """
    # A state size no other test uses, so the compiled step cannot already be a module-level cache
    # hit from another test (the compilation cache lives for the whole process).
    theta = 1.0 + jnp.arange(7, dtype=float)
    residual = _Cubic(theta)
    phi0 = jnp.full(7, 0.5)
    base = PseudoTransientStep(
        _UnitShiftPolicy(),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=2.0),
        line_search=8,
    )
    control = AlphaTargetingControl(beta_start=2.0)
    common = dict(rtol=1e-12, atol=1e-14, step_control=control)

    # One controlled step pays the compilation (which invokes the residual several times per trace).
    _TRACES.clear()
    one = forward_march(base, residual, phi0, max_steps=1, **common)
    compiled = len(_TRACES)
    assert compiled > 0 and len(one.reports) == 1

    # Several controlled steps: β adapts each step (the control is live), yet no step recompiles --
    # the controlled steps differ only in a dynamic β leaf, so _march_step stays a cache hit.
    _TRACES.clear()
    several = forward_march(base, residual, phi0, max_steps=5, **common)
    assert len(several.reports) > 1
    assert len({round(r.residual_ratio, 12) for r in several.reports}) > 1  # β is doing something
    assert len(_TRACES) == compiled  # extra controlled steps added no traces


class _CountingControl:
    """A stub step control that leaves the step unchanged and counts its calls in its state.

    Isolates the state-threading seam from any real controller dynamics: it returns ``base_step``
    untouched (so the march runs exactly as an uncontrolled one) and its state is just the number of
    times it has been called.
    """

    def next_step(self, base_step, previous, state):
        return base_step, (0 if state is None else state) + 1


def test_forward_march_threads_the_step_control_state_across_calls() -> None:
    """``forward_march`` returns the control state and resumes from a passed-in one (issue #156).

    ``solve_coupled`` runs one ``forward_march`` per preconditioner refresh, so without this a stateful
    control (the α-targeting shift climb) restarts every segment. ``rtol = atol = 0`` makes each march
    take exactly ``max_steps`` steps, so the counter is deterministic.
    """
    residual = _Cubic(1.0 + jnp.arange(6, dtype=float))
    phi0 = jnp.full(6, 0.5)
    base = DampedNewtonStep()
    common = dict(rtol=0.0, atol=0.0, step_control=_CountingControl())

    first = forward_march(base, residual, phi0, max_steps=3, **common)
    assert first.control_state == 3  # one count per step, started from None

    # Threading the returned state continues the count instead of restarting ...
    threaded = forward_march(
        base, residual, first.state, max_steps=2, control_state=first.control_state, **common
    )
    assert threaded.control_state == 5  # 3 carried in + 2 more steps

    # ... whereas omitting it restarts from None (the pre-fix per-segment reset).
    restarted = forward_march(base, residual, first.state, max_steps=2, **common)
    assert restarted.control_state == 2
    assert threaded.control_state > restarted.control_state


def test_checkpoint_receives_the_state_behind_each_report() -> None:
    """``checkpoint`` pairs each report with the state that produced it.

    Separate from ``observer`` on purpose: the report history a :class:`RefreshTrigger` reads stays
    purely numeric, which is what lets a trigger be replayed offline against a logged march. Here the
    two callbacks must agree step for step, and the final checkpointed state must be the one the
    march returns.
    """
    residual, phi0, _ = _march_and_solver_inputs()
    seen: list[StepReport] = []
    saved: list[tuple[StepReport, jnp.ndarray]] = []

    result = forward_march(
        DampedNewtonStep(line_search=10),
        residual,
        phi0,
        max_steps=4,
        rtol=1e-14,
        atol=1e-16,
        observer=seen.append,
        checkpoint=lambda report, state: saved.append((report, state)),
    )

    assert [report for report, _ in saved] == seen
    assert jnp.allclose(saved[-1][1], result.state)
    # Each checkpointed state really is that step's, not a shared reference to the last one.
    assert not jnp.allclose(saved[0][1], saved[-1][1])


def test_repeated_steps_reuse_the_compiled_march_step() -> None:
    """Stepping the march must be a compilation-cache hit, or an eager march is unusable.

    Each step's shifted linear solve is expensive to compile, so retracing per step would dominate
    the march it is meant to accelerate. The step is compiled with the strategy and the residual as
    *arguments*, so a second march over the same objects adds no traces at all.
    """
    # A state size no other test uses, so the compiled step cannot already be cached from one of
    # them -- the compilation cache is module-level and lives for the whole process.
    residual = _Cubic(jnp.array([8.0, 27.0, 64.0, 125.0, 216.0]))
    phi0 = jnp.ones(5)
    step = DampedNewtonStep(line_search=10)
    common = dict(rtol=1e-14, atol=1e-16)

    # One step, which pays the compilation. (A single trace invokes the residual several times --
    # the step, the line-search ladder, the norm -- so the trace count is not the compile count;
    # what matters is whether *further* steps add any.)
    _TRACES.clear()
    first = forward_march(step, residual, phi0, max_steps=1, **common)
    compiled = len(_TRACES)
    assert compiled > 0 and len(first.reports) == 1

    # Several more steps over the same strategy and residual: all cache hits, no tracing at all.
    more = forward_march(step, residual, first.state, max_steps=4, **common)
    assert len(more.reports) > 1  # it really did take several steps
    assert len(_TRACES) == compiled  # ...and none of them recompiled


def _drift_history(drifts: list[float]) -> list[StepReport]:
    """A synthetic history in which only the drift varies -- the drift trigger reads nothing else."""
    return [
        StepReport(step=i, cycles=10, residual_norm=1.0, residual_ratio=1.0, alpha=1.0, drift=d)
        for i, d in enumerate(drifts)
    ]


def test_drift_trigger_fires_once_the_coefficients_have_moved_past_the_threshold() -> None:
    trigger = CoefficientDriftTrigger(threshold=0.5, warmup=2)
    assert not trigger.should_refresh(_drift_history([0.0, 0.1, 0.2]))
    assert trigger.should_refresh(_drift_history([0.0, 0.1, 0.2, 0.6]))


def test_drift_trigger_ignores_its_warmup_however_large_the_drift() -> None:
    """A segment's opening steps run against a preconditioner that is fresh by construction."""
    trigger = CoefficientDriftTrigger(threshold=0.5, warmup=3)
    assert not trigger.should_refresh(_drift_history([9.0, 9.0, 9.0]))
    assert trigger.should_refresh(_drift_history([9.0, 9.0, 9.0, 9.0]))


def test_drift_trigger_never_fires_without_a_drift_measure() -> None:
    """Reports default to ``drift = 0.0`` -- "not measured" -- so the trigger fails closed."""
    trigger = CoefficientDriftTrigger(threshold=0.5, warmup=0)
    history = [
        StepReport(step=i, cycles=10, residual_norm=1.0, residual_ratio=1.0, alpha=1.0)
        for i in range(6)
    ]
    assert not trigger.should_refresh(history)


def test_drift_trigger_is_independent_of_the_damping_confound() -> None:
    """The residual ratio does not gate this trigger, unlike the cost-growth one.

    The cycle count rises both with staleness and with the damping ``beta`` ramping toward zero as
    the residual falls, so a cost trigger needs a residual gate to separate them. Drift responds only
    to the coefficients, so an identical drift history fires the same way at any residual level --
    which is the whole reason to prefer it.
    """
    trigger = CoefficientDriftTrigger(threshold=0.5, warmup=1)
    for ratio in (1.0, 1e-1, 1e-4):
        history = [
            StepReport(
                step=i, cycles=10, residual_norm=ratio, residual_ratio=ratio, alpha=1.0, drift=d
            )
            for i, d in enumerate([0.0, 0.2, 0.9])
        ]
        assert trigger.should_refresh(history)


def test_march_reports_the_injected_drift_measure() -> None:
    """``forward_march`` evaluates the measure once per step and puts the scalar on the report."""
    residual = _Cubic(theta=jnp.asarray(1.0))
    step = DampedNewtonStep(line_search=10)
    seen: list[float] = []

    result = forward_march(
        step,
        residual.__call__,
        jnp.asarray(2.0),
        max_steps=3,
        rtol=1e-12,
        atol=1e-14,
        drift_measure=lambda state: jnp.abs(state - 2.0),
        observer=lambda report: seen.append(report.drift),
    )
    assert len(seen) == len(result.reports) > 0
    assert all(
        report.drift == pytest.approx(abs(float(s)))
        for report, s in zip(result.reports, seen, strict=True)
    )
    # The measure is zero only at the state it was based on, and the march has moved away from it.
    assert seen[-1] > 0.0


def test_the_march_rebuilds_the_measure_each_outer_iteration_and_holds_it_within_one():
    """``norm_builder`` re-derives the residual measure per outer iteration, not per trial step.

    The measure's scales depend on the state, so they have to move as the solve does -- but they must
    be constant *within* an iteration, or the line search stops comparing like with like: a trial step
    could be preferred for shrinking its own denominator rather than its residual. This records the
    state each rebuild is asked about, and checks there is exactly one rebuild per step (not one per
    trial step) and that each is asked about that step's starting state.
    """
    asked = []

    def norm_builder(state):
        asked.append(jnp.asarray(state))
        # A measure that varies with the state, so a per-trial-step rebuild would be observable.
        scale = float(jnp.maximum(jnp.mean(jnp.abs(state)), 1e-12))
        return lambda residual: jnp.linalg.norm(residual) / scale

    def residual_fn(phi):
        return phi - 1.0

    phi0 = jnp.array([4.0, 4.0])
    result = forward_march(
        DampedNewtonStep(),
        residual_fn,
        phi0,
        max_steps=3,
        rtol=1e-12,
        atol=1e-12,
    )
    baseline_states = len(result.reports)

    asked.clear()
    controlled = forward_march(
        DampedNewtonStep(),
        residual_fn,
        phi0,
        max_steps=3,
        rtol=1e-12,
        atol=1e-12,
        norm_builder=norm_builder,
    )
    # One rebuild to establish the segment reference, then exactly one per outer iteration -- and no
    # more. A rebuild per *trial* step would show up here as many more calls than steps.
    assert len(controlled.reports) == baseline_states
    assert len(asked) == baseline_states + 1
    # The segment reference is measured in the march's own measure (the setup call), and the first
    # iteration is asked about the march's starting state rather than some trial point.
    assert jnp.allclose(asked[0], phi0)
    assert jnp.allclose(asked[1], phi0)
