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
    ConstantRelaxation,
    CycleGrowthTrigger,
    DampedNewtonStep,
    ImplicitNewtonSolver,
    PseudoTransientStep,
    ShiftTerm,
    StepOutcome,
    StepReport,
    SwitchedEvolutionRelaxation,
    forward_march,
)

# Incremented on every *trace* of the residual below, so a test can assert that repeated steps
# reuse a compiled march step instead of retracing it.
_TRACES: list[int] = []


def _outcome(phi, cycles, alpha=1.0, inner=1, reached=False, binding=1.0):
    """A `StepOutcome` for a test double, so a fake step matches the real protocol in one place.

    The doubles used to build the tuple inline; when the protocol grew a field every one of them broke
    separately, which is the argument for a single constructor rather than five literal tuples.

    ``reached`` defaults to **False** because these doubles model steps that do not converge -- which
    is what makes them useful for testing the escalation, and what the escalation now requires: a step
    that met its own target is not redone on cost alone. ``binding`` defaults to **1.0**, i.e. no
    constraint was in play, so a double models a step whose length the descent test alone decided.
    """
    cycles = jnp.asarray(cycles, dtype=jnp.int32)
    return StepOutcome(
        phi,
        cycles,
        jnp.asarray(alpha),
        jnp.asarray(inner, dtype=jnp.int32),
        jnp.asarray(reached),
        jnp.maximum(cycles - 2, 0),
        jnp.asarray(binding),
    )


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


class _PoisonUnlessTight(eqx.Module):
    """A step that poisons the state (non-finite) unless solved with the ``"tight"`` solver.

    Models an inexact preconditioner whose loose Krylov solve returns a non-finite correction on a
    stiff operator while a tighter solve recovers it -- so the march's divergence retry is pinned
    without a real threshold-ILU. The ``solver`` rides through ``filter_jit`` as a static argument, so
    the branch on it is resolved at trace time (a distinct compilation per solver, as for a real one).
    """

    def stepper(self):
        def step(residual_fn, phi, residual_norm_0, solver):
            if solver == "tight":  # the recovered step lands on the root, at a higher cycle cost
                return _outcome(jnp.zeros_like(phi), 6, reached=True)  # lands on the root
            return _outcome(jnp.full_like(phi, jnp.inf), 3)

        return step

    def norm(self):
        return jnp.linalg.norm

    def default_solver(self):
        return "loose"


def test_march_retries_a_diverged_step_with_the_tighter_solver() -> None:
    """A step that diverges under the loose default is redone with ``retry_solver`` and the recovered
    (finite) step is what the march accepts -- the reactive divergence retry for an inexact PC.

    Without a retry solver the poisoned step breaks the march (the pre-existing behaviour, which lets
    the finishing solve diagnose the failure); with one, the same step is redone at the tighter solver,
    lands on the root, and converges -- and the accepted step's reported cost is the tight retry's.
    """
    residual = _Cubic(jnp.zeros((1,)))  # root at phi = 0; a finite phi gives a finite residual
    phi0 = jnp.ones((1,))
    step = _PoisonUnlessTight()

    poisoned = forward_march(step, residual, phi0, max_steps=5, rtol=1e-10, atol=1e-12)
    assert not poisoned.converged
    assert not bool(jnp.isfinite(poisoned.reports[-1].residual_norm))

    recovered = forward_march(
        step, residual, phi0, max_steps=5, rtol=1e-10, atol=1e-12, retry_solver="tight"
    )
    assert recovered.converged
    assert jnp.allclose(recovered.state, 0.0, atol=1e-8)
    assert recovered.reports[-1].cycles == 6  # accepted the tight retry, not the loose attempt


def test_march_does_not_retry_a_finite_step() -> None:
    """A retry solver set but never a divergence: the march runs exactly as without one.

    The retry fires only on a diverged step, so on a healthy march ``retry_solver`` is inert -- the
    same path, the same root, the same reports -- which is what keeps it a safety net rather than an
    every-step cost.
    """
    residual, phi0, root = _march_and_solver_inputs()
    step = DampedNewtonStep(line_search=10)
    common = dict(max_steps=50, rtol=1e-10, atol=1e-12)

    plain = forward_march(step, residual, phi0, **common)
    with_retry = forward_march(step, residual, phi0, retry_solver="tight", **common)

    assert with_retry.converged and plain.converged
    assert jnp.allclose(with_retry.state, root, atol=1e-8)
    assert len(with_retry.reports) == len(plain.reports)


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


class _CyclesFromBeta(eqx.Module):
    """A step whose reported solve count is ~``base / β`` and that makes no progress -- a stand-in for the
    stiff low-β operator the cycle-count bailout escalates β against. It carries a ``ConstantRelaxation``
    β leaf so the escalation (``eqx.tree_at`` on ``relaxation_schedule.beta``) has something to raise."""

    relaxation_schedule: ConstantRelaxation
    base: float = eqx.field(static=True, default=40.0)

    def stepper(self):
        schedule, base = self.relaxation_schedule, self.base

        def step(residual_fn, phi, residual_norm_0, solver):
            cyc = jnp.round(base / jnp.maximum(schedule.beta, 1e-6)).astype(jnp.int32)
            return _outcome(phi, cyc)  # phi held: never converges

        return step

    def norm(self):
        return jnp.linalg.norm

    def default_solver(self):
        return None


def test_march_escalates_beta_on_a_cycle_count_spike() -> None:
    """A step whose count exceeds ``retry_on_cycles`` is redone from the pre-step state with β escalated
    (×``retry_beta_factor``) until the count drops or the limit is hit -- the hard-operator bailout. Here
    cyc ≈ 40/β, so β = 1 → 40 escalates to β = 4 → 10 over two ×2 escalations."""
    residual = _Cubic(
        jnp.zeros((1,))
    )  # residual(1) = 1: the held phi never converges, so the retry fires
    phi0 = jnp.ones((1,))
    step = _CyclesFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    result = forward_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry_on_cycles=10,
        retry_beta_factor=2.0,
        retry_cycles_limit=2,
    )
    assert int(result.reports[0].cycles) == 10  # 40 -> 20 -> 10 over two escalations


def test_march_does_not_escalate_below_the_cycle_cap() -> None:
    """A count under ``retry_on_cycles`` never escalates -- the bailout is inert on a comfortable step,
    so it is a safety net, not an every-step cost."""
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    step = _CyclesFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))  # cyc = 40
    result = forward_march(
        step, residual, phi0, max_steps=1, rtol=1e-10, atol=1e-12, retry_on_cycles=100
    )
    assert int(result.reports[0].cycles) == 40  # under the cap -> no escalation


def test_a_forced_escalation_adds_no_march_step_compilations() -> None:
    """A β-escalation retry must be a compilation-cache hit, not a recompile of the whole step.

    A retried step redoes ``_march_step`` at an escalated β. The escalation must not change the
    compiled step's cache key -- else a stiff region that retries every step recompiles the (in the
    coupled case, minutes-long) solve each time, which was ~half the march wall. Escalating by
    *scaling* the existing β leaf keeps its abstract value (dtype/weak_type) identical, so the step is
    a cache hit for whatever β dtype the control set. The sensitive case is a **strong-typed** β leaf:
    rebuilding β from a Python float (``jnp.asarray(escalated)``) yields a *weak*-typed leaf, a distinct
    aval that recompiled every escalation. ``_CyclesFromBeta`` never calls the residual, so
    ``_march_step`` traces it exactly once per compile -- the trace count is the compile count.
    """
    # A unique state size, so the escalation's would-be recompile cannot be a cache hit from another
    # test's compiled step (the compilation cache is process-global).
    residual = _Cubic(
        jnp.zeros((13,))
    )  # residual(1) = 1: the held phi never converges -> retry fires
    phi0 = jnp.ones((13,))
    # A strong-typed (non-weak) β leaf -- the case the old `jnp.asarray(float)` escalation recompiled on.
    step = _CyclesFromBeta(
        relaxation_schedule=ConstantRelaxation(jnp.array(1.0, dtype=jnp.float64))
    )

    # One step under the cap compiles `_march_step` once (no escalation).
    _TRACES.clear()
    baseline = forward_march(
        step, residual, phi0, max_steps=1, rtol=1e-10, atol=1e-12, retry_on_cycles=100
    )
    compiled = len(_TRACES)
    assert compiled == 1 and int(baseline.reports[0].cycles) == 40

    # The same step, now escalating β = 1 -> 2 -> 4 (cyc 40 -> 20 -> 10): both escalation `_march_step`
    # calls must reuse the compiled step -- zero further compilations -- while still recovering the step.
    _TRACES.clear()
    escalated = forward_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry_on_cycles=10,
        retry_beta_factor=2.0,
        retry_cycles_limit=2,
    )
    assert len(_TRACES) == 0  # the escalation retries added no recompiles
    assert int(escalated.reports[0].cycles) == 10  # ...and still escalated β to recover the step


class _NaNUntilDamped(eqx.Module):
    """A step whose correction is non-finite while β is below ``threshold`` and finite + on-root once β is
    escalated past it -- the coupled-AMG failure where a NaN'd low-β step recovers at a larger β. If ever
    handed the tight ``"tight"`` retry solver it returns a distinct marker cost (99), so a test can tell
    whether the cheap β-escalation recovered the step or the expensive divergence fallback fired."""

    relaxation_schedule: ConstantRelaxation
    threshold: float = eqx.field(static=True, default=1.0)

    def stepper(self):
        sched, thr = self.relaxation_schedule, self.threshold

        def step(residual_fn, phi, residual_norm_0, solver):
            if (
                solver == "tight"
            ):  # the divergence fallback -- marked so the test can detect it fired
                return _outcome(jnp.zeros_like(phi), 6, reached=True)
            phi_out = jnp.where(sched.beta >= thr, jnp.zeros_like(phi), jnp.full_like(phi, jnp.inf))
            return _outcome(phi_out, 3)

        return step

    def norm(self):
        return jnp.linalg.norm

    def default_solver(self):
        return "loose"


def test_march_escalates_beta_before_the_tight_divergence_retry() -> None:
    """A non-finite step that recovers at escalated β is fixed by the CHEAP β-escalation first -- the
    tight ``retry_solver`` (the expensive fallback) never fires. This is the reorder: on the stiff low-β
    saddle a NaN is cured by more damping, so grinding the tight Krylov solve before escalating (the old
    order) was wasted work that the escalation then re-damped away anyway."""
    residual = _Cubic(
        jnp.zeros((1,))
    )  # root at phi = 0; a non-finite phi gives a non-finite residual
    phi0 = jnp.ones((1,))
    step = _NaNUntilDamped(relaxation_schedule=ConstantRelaxation(jnp.asarray(0.5)), threshold=1.0)
    result = forward_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry_solver="tight",
        retry_on_cycles=10,
        retry_beta_factor=2.0,
        retry_cycles_limit=2,
    )
    assert result.converged
    assert jnp.allclose(result.state, 0.0, atol=1e-8)
    assert (
        result.reports[0].cycles == 3
    )  # β=0.5 -> 1.0 recovered it (cost 3); the tight retry (99/6) never fired


def test_march_falls_back_to_the_tight_retry_when_escalation_cannot_fix_divergence() -> None:
    """If β-escalation does not lift the step out of the non-finite regime -- the inexact-PC failure only a
    tighter Krylov solve fixes -- the divergence retry still fires as the fallback, so the reorder never
    loses the ILUT recovery it front-runs."""
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    # threshold unreachable by 0.5 -> 1.0 -> 2.0, so every escalation stays non-finite; only "tight" recovers.
    step = _NaNUntilDamped(relaxation_schedule=ConstantRelaxation(jnp.asarray(0.5)), threshold=1e9)
    result = forward_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry_solver="tight",
        retry_on_cycles=10,
        retry_beta_factor=2.0,
        retry_cycles_limit=2,
    )
    assert result.converged
    assert jnp.allclose(result.state, 0.0, atol=1e-8)
    assert (
        result.reports[0].cycles == 6
    )  # escalation exhausted -> fell back to the tight retry (cost 6)


def test_march_carries_the_escalated_beta_into_the_control() -> None:
    """After a β-escalation the control's carried β is SEEDED with the escalated value, so the next step
    continues from the discovered-safe β instead of re-deriving the control's own (floor-ward) β and
    re-paying the escalation every step -- the low-β reachability-tail fix that lets a static β floor be
    dropped (escalation + carry discover how low is safe). Here cyc ≈ 40/β, so β = 1 escalates 1→2→4 to
    clear the cap of 10, and 4.0 is what the control carries out (a plain reset would carry the control's
    own β)."""
    from aquaflux.solve import DualTimeControl

    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    step = _CyclesFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    control = DualTimeControl(beta_start=1.0, beta_min=0.01)
    result = forward_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        step_control=control,
        retry_on_cycles=10,
        retry_beta_factor=2.0,
        retry_cycles_limit=3,
    )
    assert (
        float(result.control_state) == 4.0
    )  # the escalated β, carried; not the control's beta_start


class _AlphaFromBeta(eqx.Module):
    """A step whose line-search factor collapses until β is large enough, making no progress meanwhile.

    The stand-in for the failure the cycle count cannot see: the solves are *cheap* (a fixed small
    count) and the step still achieves nothing, because the correction cannot be followed. ``alpha``
    is 0 below ``beta_needed`` and 1 at or above it, which is the shape the real march shows -- a
    positivity cap or a non-descending direction that a larger shift steps short enough to satisfy.
    """

    relaxation_schedule: ConstantRelaxation
    beta_needed: float = eqx.field(static=True, default=4.0)

    def stepper(self):
        schedule, needed = self.relaxation_schedule, self.beta_needed

        def step(residual_fn, phi, residual_norm_0, solver):
            alpha = jnp.where(schedule.beta >= needed, 1.0, 0.0)
            return _outcome(phi, 3, alpha=alpha)  # cheap every time; phi held, so never converges

        return step

    def norm(self):
        return jnp.linalg.norm

    def default_solver(self):
        return None


def test_march_escalates_beta_on_a_collapsed_step_length() -> None:
    """A step whose line-search factor collapsed is redone at a larger β, though its solves were cheap.

    This is the trigger's whole point: the cost bailout cannot see this failure, because 3 cycles is
    under any sane threshold. Here α = 0 until β reaches 4, so β = 1 escalates twice (×2) to 4.
    """
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    step = _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    result = forward_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry_on_alpha=0.0,
        retry_on_cycles=1000,  # far above the step's 3 cycles: only the α trigger can fire
        retry_beta_factor=2.0,
        retry_cycles_limit=2,
    )
    assert int(result.reports[0].escalations) == 2
    assert float(result.reports[0].alpha) == 1.0


def test_march_does_not_escalate_on_a_healthy_step_length() -> None:
    """A step taking its full length never escalates -- the trigger is a safety net, not a per-step cost."""
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    # beta_needed = 0 => alpha is 1 from the start.
    step = _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)), beta_needed=0.0)
    result = forward_march(
        step, residual, phi0, max_steps=1, rtol=1e-10, atol=1e-12, retry_on_alpha=0.5
    )
    assert int(result.reports[0].escalations) == 0


def test_the_alpha_trigger_reports_its_own_reason() -> None:
    """``on_retry`` must say ``"alpha"``, not ``"cycles"`` -- the two call for different responses, and a
    log that cannot tell them apart is why the reason is reported at all."""
    reasons: list[str] = []
    residual = _Cubic(jnp.zeros((1,)))
    step = _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    forward_march(
        step,
        residual,
        jnp.ones((1,)),
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry_on_alpha=0.0,
        retry_on_cycles=1000,
        retry_cycles_limit=2,
        on_retry=lambda reason, attempt, beta: reasons.append(reason),
    )
    assert reasons == ["alpha", "alpha"]


def test_the_alpha_trigger_never_bins_a_step_that_reached_its_target() -> None:
    """A collapsed α on a step that met its own stopping criterion is NOT a reason to redo it.

    Same guard the cost trigger carries: redoing a step that converged discards a good iterate and
    replaces it with a shorter one. Only a step that was cut short is escalated.
    """
    from aquaflux.solve.march import _escalation_reason

    converged = _outcome(jnp.zeros((1,)), 3, alpha=0.0, reached=True)
    cut_short = _outcome(jnp.zeros((1,)), 3, alpha=0.0, reached=False)
    kwargs = dict(retry_on_cycles=None, retry_on_alpha=0.0, divergence_cap=float("inf"))
    assert _escalation_reason(converged, jnp.asarray(1.0), 1.0, **kwargs) is None
    assert _escalation_reason(cut_short, jnp.asarray(1.0), 1.0, **kwargs) == "alpha"


def test_a_diverged_step_outranks_the_other_escalation_reasons() -> None:
    """Divergence is reported first, and unlike the other two it fires whatever the step's target says --
    a non-finite residual is not a result to be kept because the loop happened to meet its tolerance."""
    from aquaflux.solve.march import _escalation_reason

    outcome = _outcome(jnp.zeros((1,)), 3, alpha=0.0, reached=True)
    reason = _escalation_reason(
        outcome,
        jnp.asarray(jnp.nan),
        1.0,
        retry_on_cycles=1,
        retry_on_alpha=0.0,
        divergence_cap=float("inf"),
    )
    assert reason == "diverged"


def test_no_escalation_reason_when_neither_threshold_is_set() -> None:
    """Both thresholds ``None`` disables escalation entirely -- the default path, byte-identical."""
    from aquaflux.solve.march import _escalation_reason

    outcome = _outcome(jnp.zeros((1,)), 10_000, alpha=0.0, reached=False)
    assert (
        _escalation_reason(
            outcome,
            jnp.asarray(jnp.nan),
            1.0,
            retry_on_cycles=None,
            retry_on_alpha=None,
            divergence_cap=float("inf"),
        )
        is None
    )


def test_the_alpha_trigger_fires_whatever_collapsed_the_step_length() -> None:
    """A constraint-bound collapse escalates too -- gating this on ``binding_limit`` is a regression.

    Tempting and wrong: more damping shrinks the correction, so it *widens* a fraction-to-the-boundary
    cap rather than tightening it. The one escalation of an entire coupled RANS march fired at a step
    whose cap was 4.37e-10, and doubling the shift there was worth 8 steps and 199 s end to end. The
    failure damping cannot fix is a cell already pinned on the boundary, and that is caught by the
    stall bailout below, not here.
    """
    from aquaflux.solve.march import _escalation_reason

    kwargs = dict(retry_on_cycles=None, retry_on_alpha=0.01, divergence_cap=float("inf"))
    search = _outcome(jnp.zeros((1,)), 3, alpha=0.001)  # binding 1.0: the ladder chose this length
    capped = _outcome(jnp.zeros((1,)), 3, alpha=0.001, binding=0.001)  # the cap chose it
    assert _escalation_reason(search, jnp.asarray(1.0), 1.0, **kwargs) == "alpha"
    assert _escalation_reason(capped, jnp.asarray(1.0), 1.0, **kwargs) == "alpha"


def test_a_constraint_bound_step_is_only_a_stall_when_it_narrows_and_gains_nothing() -> None:
    """``_limit_collapsing`` needs all three conditions; each alone names an ordinary step.

    A capped step is a legitimate short step and a pseudo-transient path may be non-monotone, so
    neither a tight cap nor a risen residual is a failure on its own. The lock-up is the conjunction:
    the cap narrowing while the residual does not fall.
    """
    from aquaflux.solve.march import _limit_collapsing

    def report(binding, residual):
        return StepReport(
            step=0,
            cycles=1,
            residual_norm=residual,
            residual_ratio=residual,
            alpha=binding,
            binding_limit=binding,
        )

    previous = report(1e-5, 1.0)
    assert _limit_collapsing(previous, report(1e-7, 1.0))  # narrowing, no gain -> a stall
    assert not _limit_collapsing(previous, report(1e-7, 0.5))  # the residual fell
    assert not _limit_collapsing(previous, report(1e-3, 1.0))  # the cap widened again
    assert not _limit_collapsing(previous, report(1.0, 1.0))  # no constraint bound this step
    assert not _limit_collapsing(None, report(1e-7, 1.0))  # nothing to continue


class _PinnedOnItsLimit(eqx.Module):
    """A step held against a constraint: the cap never widens and the iterate never moves.

    The stand-in for a fraction-to-the-boundary lock-up. In the real march the cap *narrows* every step
    -- taking ``tau`` of the distance to a constraint leaves the binding entry at ``1 - tau`` of its
    value, so the next step's room is a fixed fraction of this one's for as long as the direction keeps
    pointing at the boundary. A constant cap is the same case for the predicate under test (which asks
    only that the cap not widen) and keeps this double a pure function of its inputs, which the march
    requires. The iterate is held, so the residual is frozen and no stopping test can ever fire --
    which is the whole failure.
    """

    cap: float = eqx.field(static=True, default=0.01)

    def stepper(self):
        cap = self.cap

        def step(residual_fn, phi, residual_norm_0, solver):
            return _outcome(phi, 1, alpha=0.0, binding=cap)

        return step

    def norm(self):
        return jnp.linalg.norm

    def default_solver(self):
        return None


def test_a_collapsing_constraint_cap_ends_the_segment() -> None:
    """A march pinned on a cap stops after ``stop_on_limit_stall`` steps, not at ``max_steps``.

    Without this the segment spends its entire budget on arithmetically null steps -- measured on a
    coupled RANS march as ninety-six consecutive steps at a frozen residual, the cap falling by exactly
    100x each one. Ending the segment hands back an honestly unconverged state instead.
    """
    residual = _Cubic(jnp.zeros((1,)))
    result = forward_march(
        _PinnedOnItsLimit(),
        residual,
        jnp.ones((1,)),
        max_steps=50,
        rtol=1e-10,
        atol=1e-12,
        stop_on_limit_stall=3,
    )
    # One step establishes the sequence; three more continue it and trip the count.
    assert len(result.reports) == 4
    assert not result.converged and not result.triggered


def test_the_stall_bailout_is_off_when_unset() -> None:
    """``stop_on_limit_stall=None`` disables the test, so the march runs its whole budget as before."""
    residual = _Cubic(jnp.zeros((1,)))
    result = forward_march(
        _PinnedOnItsLimit(),
        residual,
        jnp.ones((1,)),
        max_steps=7,
        rtol=1e-10,
        atol=1e-12,
        stop_on_limit_stall=None,
    )
    assert len(result.reports) == 7
