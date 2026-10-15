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
    NO_RETRIES,
    CflResidualDualTimeControl,
    CoefficientDriftTrigger,
    ConstantRelaxation,
    CycleGrowthTrigger,
    DampedNewtonStep,
    DualTimeControl,
    DualTimeStep,
    LinearSolverSpec,
    PseudoTransientStep,
    RetryPolicy,
    ShiftControlState,
    ShiftTerm,
    StepOutcome,
    StepReport,
    SwitchedEvolutionRelaxation,
    newton_march,
)
from aquaflux.solve.implicit import backtracking_line_search
from aquaflux.solve.march import _march_step
from aquaflux.solve.norm import norm_of_residual

# Incremented on every *trace* of the residual below, so a test can assert that repeated steps
# reuse a compiled march step instead of retracing it.
_TRACES: list[int] = []


def _outcome(phi, cycles, alpha=1.0, inner=1, reached=False, residual_fn=None, refresh_due=False):
    """A `StepOutcome` for a test double, so a fake step matches the real protocol in one place.

    The doubles used to build the tuple inline; when the protocol grew a field every one of them broke
    separately, which is the argument for a single constructor rather than five literal tuples.

    ``residual_fn`` is the double's own residual, used to fill :attr:`StepOutcome.residual_norm` with
    ``norm(R(phi))`` at the stepped iterate -- what a real step reports, and what the march used to
    evaluate for itself. A double built outside a step (to be handed straight to a retry policy, which
    reads only the cost fields) may leave it ``None``; the norm is then a stand-in.

    ``reached`` defaults to **False** because these doubles model steps that do not converge -- which
    is what makes them useful for testing the escalation, and what the escalation now requires: a step
    that met its own target is not redone on cost alone.
    """
    cycles = jnp.asarray(cycles, dtype=jnp.int32)
    measured = phi if residual_fn is None else residual_fn(phi)
    return StepOutcome(
        phi,
        jnp.linalg.norm(measured),
        cycles,
        jnp.asarray(alpha),
        jnp.asarray(inner, dtype=jnp.int32),
        jnp.asarray(reached),
        jnp.maximum(cycles - 2, 0),
        jnp.asarray(jnp.nan),
        jnp.asarray(refresh_due),
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


def _warm_the_march_norm(step, residual, phi0) -> None:
    """Compile the march's starting-norm program before a test counts the step's traces.

    The march reads its starting residual norm through a compiled program of its own, which traces the
    residual once the first time it meets a residual of that structure. A test that takes the first
    march's trace count as the compile count of the STEP would count that trace too.
    """
    norm_of_residual(step.norm(), residual, phi0)


def _report(step: int, cycles: int, ratio: float) -> StepReport:
    """A synthetic report; only ``cycles`` and ``residual_ratio`` drive the trigger."""
    return StepReport(
        step=step, raw_cycles=cycles, residual_norm=ratio, residual_ratio=ratio, alpha=1.0
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


def test_march_without_a_trigger_reaches_the_analytic_root() -> None:
    """Left alone, the march converges to the root and says so.

    It used to compare itself against a second, traced Newton loop, which no longer exists -- there is
    one march, and `RootSolver` runs it -- so the comparison would now be against itself.
    The analytic cube root is an independent reference and is what this checks against.
    """
    residual, phi0, root = _march_and_solver_inputs()

    marched = newton_march(
        DampedNewtonStep(line_search=10), residual, phi0, max_steps=50, rtol=1e-10, atol=1e-12
    )

    assert marched.converged
    assert jnp.allclose(marched.state, root, atol=1e-8)


class _AlreadyInfinite(eqx.Module):
    """A residual that is ``+inf`` at every state: a march handed one has diverged before it began."""

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        # A genuine function of `phi` (1/0), not a bare `inf` fill, so the value the march reads is
        # one the arithmetic produced rather than a constant folded in.
        return 1.0 / phi


def test_a_march_whose_residual_is_already_infinite_does_not_report_converged() -> None:
    """The tolerance test alone cannot reject this, and the march must not step it either.

    ``within_tolerance`` compares with ``<=``. When the residual norm and the threshold it is judged
    against have both run away to ``+inf``, ``inf <= inf`` is ``True`` -- so a march that only asked
    that question would report a state solving nothing as converged, and its caller would attach an
    adjoint to it. (A ``NaN`` fails the comparison on its own; ``+inf`` is the case needing the
    explicit finiteness test.)

    It must also take **no steps**: the first step's Krylov solve would be handed a non-finite
    right-hand side and raise from inside ``lineax``, reporting a bug upstream of itself rather than
    the fact the caller needs.
    """
    result = newton_march(
        DampedNewtonStep(), _AlreadyInfinite(), jnp.zeros(3), max_steps=5, rtol=1e-10, atol=1e-12
    )

    # The fixture really is the case under test: non-finite before a single step is taken.
    assert not jnp.all(jnp.isfinite(_AlreadyInfinite()(jnp.zeros(3))))
    assert not result.converged
    assert result.reports == ()


class _TightSolve(LinearSolverSpec):
    """Names the ``"tight"`` marker solver the test steps below branch on, in place of a real GMRES."""

    def build(self):
        return "tight"


_TIGHT = _TightSolve()


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
                return _outcome(jnp.zeros_like(phi), 6, reached=True, residual_fn=residual_fn)
            return _outcome(jnp.full_like(phi, jnp.inf), 3, residual_fn=residual_fn)

        return step

    def norm(self):
        return jnp.linalg.norm

    def linear_solver(self):
        return "loose"


def test_march_retries_a_diverged_step_with_the_tighter_solver() -> None:
    """A step that diverges under the loose default is redone with ``retry.solver`` and the recovered
    (finite) step is what the march accepts -- the reactive divergence retry for an inexact PC.

    Without a retry solver the poisoned step breaks the march (the pre-existing behaviour, which lets
    the finishing solve diagnose the failure); with one, the same step is redone at the tighter solver,
    lands on the root, and converges -- and the accepted step's reported cost is the tight retry's.
    """
    residual = _Cubic(jnp.zeros((1,)))  # root at phi = 0; a finite phi gives a finite residual
    phi0 = jnp.ones((1,))
    step = _PoisonUnlessTight()

    poisoned = newton_march(step, residual, phi0, max_steps=5, rtol=1e-10, atol=1e-12)
    assert not poisoned.converged
    assert not bool(jnp.isfinite(poisoned.reports[-1].residual_norm))

    recovered = newton_march(
        step, residual, phi0, max_steps=5, rtol=1e-10, atol=1e-12, retry=RetryPolicy(solver=_TIGHT)
    )
    assert recovered.converged
    assert jnp.allclose(recovered.state, 0.0, atol=1e-8)
    assert recovered.reports[-1].raw_cycles == 6  # accepted the tight retry, not the loose attempt


def test_march_does_not_retry_a_finite_step() -> None:
    """A retry solver set but never a divergence: the march runs exactly as without one.

    The retry fires only on a diverged step, so on a healthy march ``retry.solver`` is inert -- the
    same path, the same root, the same reports -- which is what keeps it a safety net rather than an
    every-step cost.
    """
    residual, phi0, root = _march_and_solver_inputs()
    step = DampedNewtonStep(line_search=10)
    common = dict(max_steps=50, rtol=1e-10, atol=1e-12)

    plain = newton_march(step, residual, phi0, **common)
    with_retry = newton_march(step, residual, phi0, retry=RetryPolicy(solver=_TIGHT), **common)

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
    result = newton_march(
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
    result = newton_march(
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

    def shift_term(self, phi, residual=None):
        return ShiftTerm(
            diagonal=jnp.ones_like(phi), make_preconditioner=lambda _shift_strength: None
        )


def test_step_control_drives_the_march_and_stays_a_cache_hit() -> None:
    """A ``step_control`` reshapes the step each iteration, and the controlled march does not retrace.

    A control replaces the base step's schedule with a ``ConstantRelaxation`` on a dynamic β leaf.
    Two things must hold: β actually changes across steps (the control is doing something), and
    ``_march_step`` compiles once despite a fresh controlled step object each iteration (the
    load-bearing cache-hit property — a per-step recompile would dominate a real march).

    Driven here on a ``PseudoTransientStep`` rather than the dual-time step the Courant control is
    written for: what is under test is the march's caching across a *reshaped* step, and any control
    reshapes it the same way, so the simpler base keeps the test about compilation and not physics.
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
    control = DualTimeControl(beta_start=2.0)
    common = dict(rtol=1e-12, atol=1e-14, step_control=control)

    # One controlled step pays the compilation (which invokes the residual several times per trace).
    _warm_the_march_norm(base, residual, phi0)
    _TRACES.clear()
    one = newton_march(base, residual, phi0, max_steps=1, **common)
    compiled = len(_TRACES)
    assert compiled > 0 and len(one.reports) == 1

    # Several controlled steps: β adapts each step (the control is live), yet no step recompiles --
    # the controlled steps differ only in a dynamic β leaf, so _march_step stays a cache hit.
    _TRACES.clear()
    several = newton_march(base, residual, phi0, max_steps=5, **common)
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
    """``newton_march`` returns the control state and resumes from a passed-in one (issue #156).

    ``solve_coupled`` runs one ``newton_march`` per preconditioner refresh, so without this a stateful
    control (the α-targeting shift climb) restarts every segment. ``rtol = atol = 0`` makes each march
    take exactly ``max_steps`` steps, so the counter is deterministic.
    """
    residual = _Cubic(1.0 + jnp.arange(6, dtype=float))
    phi0 = jnp.full(6, 0.5)
    base = DampedNewtonStep()
    common = dict(rtol=0.0, atol=0.0, step_control=_CountingControl())

    first = newton_march(base, residual, phi0, max_steps=3, **common)
    assert first.control_state == 3  # one count per step, started from None

    # Threading the returned state continues the count instead of restarting ...
    threaded = newton_march(
        base, residual, first.state, max_steps=2, control_state=first.control_state, **common
    )
    assert threaded.control_state == 5  # 3 carried in + 2 more steps

    # ... whereas omitting it restarts from None (the pre-fix per-segment reset).
    restarted = newton_march(base, residual, first.state, max_steps=2, **common)
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

    result = newton_march(
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
    first = newton_march(step, residual, phi0, max_steps=1, **common)
    compiled = len(_TRACES)
    assert compiled > 0 and len(first.reports) == 1

    # Several more steps over the same strategy and residual: all cache hits, no tracing at all.
    more = newton_march(step, residual, first.state, max_steps=4, **common)
    assert len(more.reports) > 1  # it really did take several steps
    assert len(_TRACES) == compiled  # ...and none of them recompiled


def _drift_history(drifts: list[float]) -> list[StepReport]:
    """A synthetic history in which only the drift varies -- the drift trigger reads nothing else."""
    return [
        StepReport(step=i, raw_cycles=10, residual_norm=1.0, residual_ratio=1.0, alpha=1.0, drift=d)
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
        StepReport(step=i, raw_cycles=10, residual_norm=1.0, residual_ratio=1.0, alpha=1.0)
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
                step=i, raw_cycles=10, residual_norm=ratio, residual_ratio=ratio, alpha=1.0, drift=d
            )
            for i, d in enumerate([0.0, 0.2, 0.9])
        ]
        assert trigger.should_refresh(history)


def test_march_reports_the_injected_drift_measure() -> None:
    """``newton_march`` evaluates the measure once per step and puts the scalar on the report."""
    residual = _Cubic(theta=jnp.asarray(1.0))
    step = DampedNewtonStep(line_search=10)
    seen: list[float] = []

    result = newton_march(
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


def test_each_report_carries_its_steps_full_step_ratio() -> None:
    """The report's ``full_step_ratio`` is the step's own full-length trial, the one it REJECTED here.

    From ``phi = 0.5`` on ``R = phi**3 - 1`` the Newton correction ``-R / (3 phi**2) = 7/6`` overshoots to
    ``phi = 5/3``, where ``|R| = 98/27`` against ``|R(0.5)| = 7/8``: a ratio of ``112/27 > 1``, so the
    search halves the step. The kept rung's ratio is below one, so a march reporting that instead -- or
    dropping the field -- fails here.
    """
    residual = _Cubic(theta=jnp.asarray(1.0))
    result = newton_march(
        DampedNewtonStep(line_search=10),
        residual.__call__,
        jnp.asarray(0.5),
        max_steps=1,
        rtol=1e-12,
        atol=1e-14,
    )

    (report,) = result.reports
    assert report.full_step_ratio == pytest.approx(112.0 / 27.0, rel=1e-12)
    assert report.alpha < 1.0


def test_the_march_rebuilds_the_measure_each_outer_iteration_and_holds_it_within_one():
    """``norm_builder`` re-derives the residual measure per outer iteration, not per trial step.

    The measure's scales depend on the state, so they have to move as the solve does -- but they must
    be constant *within* an iteration, or the line search stops comparing like with like: a trial step
    could be preferred for shrinking its own denominator rather than its residual. This records the
    state each rebuild is asked about, and checks there is exactly one rebuild per step (not one per
    trial step) and that each is asked about that step's starting state.
    """
    asked = []

    def norm_builder(step, state):
        # Handed the step the iteration runs, for a measure that reads its scales from the step's shift.
        assert isinstance(step, DampedNewtonStep)
        asked.append(jnp.asarray(state))
        # A measure that varies with the state, so a per-trial-step rebuild would be observable.
        scale = float(jnp.maximum(jnp.mean(jnp.abs(state)), 1e-12))
        return lambda residual: jnp.linalg.norm(residual) / scale

    def residual_fn(phi):
        return phi - 1.0

    phi0 = jnp.array([4.0, 4.0])
    result = newton_march(
        DampedNewtonStep(),
        residual_fn,
        phi0,
        max_steps=3,
        rtol=1e-12,
        atol=1e-12,
    )
    baseline_states = len(result.reports)

    asked.clear()
    controlled = newton_march(
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
            return _outcome(phi, cyc, residual_fn=residual_fn)  # phi held: never converges

        return step

    def norm(self):
        return jnp.linalg.norm

    def linear_solver(self):
        return None


def test_a_cycle_spike_redoes_the_step_ONCE_and_does_NOT_escalate() -> None:
    """A costly solve is a statement about the PRECONDITIONER, so the step is redone at the SAME β.

    The cure for an expensive solve is a fresh factorization -- which the march asks its refresh hook
    for before the redo -- not a stiffer operator. Escalating here
    additionally assumes a larger shift makes the block easier, which is false on at least one case,
    where it closed a loop that ran the wrong way and drove a working march to a non-finite residual.

    And exactly ONE redo: a second at the same shift on the same factorization would repeat the first
    attempt exactly, burning the escalation budget on identical work.

    Here cyc ≈ 40/β, so an escalating policy would drop 40 → 20 → 10; holding β pins it at 40.
    """
    residual = _Cubic(
        jnp.zeros((1,))
    )  # residual(1) = 1: the held phi never converges, so the retry fires
    phi0 = jnp.ones((1,))
    seen: list = []
    step = _CyclesFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    result = newton_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(abort_above_cycles=10, beta_factor=2.0, max_retries=2),
        on_retry=lambda reason, attempt, beta: seen.append((reason, float(beta))),
    )
    assert int(result.reports[0].raw_cycles) == 40  # β never moved, so the count never dropped
    assert seen == [("cycles", 1.0)]  # one redo, at the shift it already had


def test_a_cost_redo_asks_the_refresh_hook_for_a_refit() -> None:
    """The redo exists to run on a FRESH preconditioner, so the march must ask for one, not offer.

    A hook gating its own cadence (a multigrid session re-fits only when asked) would otherwise reuse
    the standing inverse, and the redo would repeat the first attempt exactly. Before the first step the
    refresh is only offered (``due`` false); before the redo it is asked for.
    """
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    asked: list[bool] = []
    newton_march(
        _CyclesFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0))),
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(abort_above_cycles=10, beta_factor=2.0, max_retries=2),
        refresh_preconditioner=lambda step, state, *, due: asked.append(due),
    )
    assert asked == [False, True]


class _CostFromData(eqx.Module):
    """A step whose solve cost is a data leaf and that makes no progress: a stale preconditioner as data."""

    cost: jnp.ndarray
    relaxation_schedule: ConstantRelaxation

    def stepper(self):
        def step(residual_fn, phi, residual_norm_0, solver):
            return _outcome(phi, self.cost, residual_fn=residual_fn)

        return step

    def norm(self):
        return jnp.linalg.norm

    def linear_solver(self):
        return None


def test_an_update_returned_before_a_cost_redo_reaches_the_redo_and_every_later_step() -> None:
    """The redo is asked a fresh preconditioner; one carried as data must reach it and outlast it.

    The first attempt costs 40 cycles against a threshold of 10, so it is redone after the hook is
    asked for a re-fit, which comes back as an update cutting the cost to 1. Applied to the redo's step
    alone, the next step would revert to 40 and be redone again: the counts are the evidence.
    """
    retried: list[str] = []

    def refresh(step, state, *, due):
        return (lambda s: eqx.tree_at(lambda t: t.cost, s, jnp.asarray(1))) if due else None

    result = newton_march(
        _CostFromData(jnp.asarray(40), ConstantRelaxation(jnp.asarray(1.0))),
        _Cubic(jnp.zeros((1,))),
        jnp.ones((1,)),
        max_steps=2,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(abort_above_cycles=10, max_retries=2),
        on_retry=lambda reason, attempt, beta: retried.append(reason),
        refresh_preconditioner=refresh,
    )
    assert retried == ["cycles"]
    assert [int(report.raw_cycles) for report in result.reports] == [1, 1]
    assert int(result.strategy.cost) == 1


class _StopsForRefit(eqx.Module):
    """A step that always ends early for a preconditioner re-fit, after an expensive solve.

    It halves the state each step, so progress is visible, and reports a solve far over any cost
    threshold -- which, on a step that did NOT stop for a re-fit, the march would redo.
    """

    relaxation_schedule: ConstantRelaxation
    refresh_on_cycles: int = eqx.field(static=True, default=3)

    def stepper(self):
        def step(residual_fn, phi, residual_norm_0, solver):
            return _outcome(0.5 * phi, 40, residual_fn=residual_fn, refresh_due=True)

        return step

    def norm(self):
        return jnp.linalg.norm

    def linear_solver(self):
        return None


def test_a_step_that_stopped_for_a_refit_is_kept_and_the_next_step_asks_for_it() -> None:
    """Stopping for a re-fit keeps the step's work; the re-fit is taken before the next step.

    The step's solve cost (38 corrected cycles) is far over ``abort_above_cycles``, so without the
    exemption the march would redo it at the same shift -- discarding the iterate the refresh exists to
    keep. Instead nothing is retried, each step's halving survives, and the hook is asked (``due``) before
    every step after the first, at the state the previous one kept.
    """
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    asked: list[tuple[bool, float]] = []
    retried: list[str] = []
    result = newton_march(
        _StopsForRefit(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0))),
        residual,
        phi0,
        max_steps=3,
        rtol=0.0,
        atol=0.0,
        retry=RetryPolicy(abort_above_cycles=10),
        on_retry=lambda reason, attempt, beta: retried.append(reason),
        refresh_preconditioner=lambda step, state, *, due: asked.append((due, float(state[0]))),
    )
    assert retried == []
    assert float(result.state[0]) == 0.125
    assert asked == [(False, 1.0), (True, 0.5), (True, 0.25)]


class _ScalesByFactor(eqx.Module):
    """A step whose preconditioner stand-in is DATA: it multiplies the state by an array leaf.

    The shape of a traced field split riding in a step: a refresh cannot mutate it, so the hook hands
    back an update and the march has to carry it into every later step.
    """

    factor: jnp.ndarray
    relaxation_schedule: ConstantRelaxation

    def stepper(self):
        def step(residual_fn, phi, residual_norm_0, solver):
            return _outcome(self.factor * phi, 1, residual_fn=residual_fn)

        return step

    def norm(self):
        return jnp.linalg.norm

    def linear_solver(self):
        return None


def test_an_update_the_refresh_hook_returns_reaches_every_later_step_and_the_result() -> None:
    """A preconditioner carried as data comes back from the hook as a step update, once, at a re-fit.

    The march rebuilds the active step from its base step at the top of every iteration, so an update
    applied to the active step alone would last one step and the next would run on the stale data --
    here, three steps at the re-fitted factor give 0.25**3 where one step of it gives 0.25 * 0.5**2.
    The driver adjoints with the step the march reports, so that must carry the update too.
    """
    updates = []

    def refresh(step, state, *, due):
        if updates:
            return None
        updates.append(1)
        return lambda s: eqx.tree_at(lambda t: t.factor, s, jnp.asarray(0.25))

    result = newton_march(
        _ScalesByFactor(jnp.asarray(0.5), ConstantRelaxation(jnp.asarray(1.0))),
        _Cubic(jnp.zeros((1,))),
        jnp.ones((1,)),
        max_steps=3,
        rtol=0.0,
        atol=0.0,
        refresh_preconditioner=refresh,
    )
    assert float(result.state[0]) == 0.25**3
    assert float(result.strategy.factor) == 0.25


class _JudgedOnItsScale(eqx.Module):
    """A step whose measure is built from a leaf of the step, as a row-scaled one reads its shift.

    It halves the state and reports the residual in the measure it was handed, so a report says which
    measure the step was judged in.
    """

    scale: jnp.ndarray
    residual_norm: object
    relaxation_schedule: ConstantRelaxation

    def stepper(self):
        def step(residual_fn, phi, residual_norm_0, solver):
            stepped = 0.5 * phi
            return _outcome(stepped, 1)._replace(
                residual_norm=self.residual_norm(residual_fn(stepped))
            )

        return step

    def norm(self):
        return self.residual_norm

    def with_norm(self, norm):
        return eqx.tree_at(lambda s: s.residual_norm, self, norm)

    def linear_solver(self):
        return None


def test_a_step_an_update_moves_is_judged_in_the_measure_built_from_the_updated_step() -> None:
    """An update can carry what the measure is built from (a re-pointed problem's shift), so the march
    builds the measure again from the updated step.

    The update on the first step raises the scale from 1 to 10. A march that kept the measure it built
    before the hook ran would judge that step in the old scale and report ``|R|`` instead of ``|R| / 10``.
    """
    updated = []

    def refresh(step, state, *, due):
        if updated:
            return None
        updated.append(1)
        return lambda s: eqx.tree_at(lambda t: t.scale, s, jnp.asarray(10.0))

    def norm_builder(step, state):
        return lambda residual: jnp.linalg.norm(residual) / step.scale

    result = newton_march(
        _JudgedOnItsScale(jnp.asarray(1.0), jnp.linalg.norm, ConstantRelaxation(jnp.asarray(1.0))),
        _Cubic(jnp.zeros((1,))),
        jnp.ones((1,)),
        max_steps=2,
        rtol=0.0,
        atol=0.0,
        norm_builder=norm_builder,
        refresh_preconditioner=refresh,
    )
    assert [float(r.residual_norm) for r in result.reports] == [0.5**3 / 10.0, 0.25**3 / 10.0]


def _scaled_by_step(step, state):
    """A measure built from the step: the residual's 2-norm over the step's ``scale``."""
    del state
    return lambda residual: jnp.linalg.norm(residual) / step.scale


def _judged_on(scale):
    """The stopping step of a target whose own scale is ``scale``, for :class:`_JudgedOnItsScale`."""
    return lambda step: eqx.tree_at(lambda t: t.scale, step, jnp.asarray(scale))


def _march_judged(steering_scale, *, max_steps=6, rtol=0.0, atol=0.0, **kwargs):
    """March ``phi**3 = 0`` from one, halving each step, in a measure scaled by ``steering_scale``."""
    return newton_march(
        _JudgedOnItsScale(
            jnp.asarray(steering_scale), jnp.linalg.norm, ConstantRelaxation(jnp.asarray(1.0))
        ),
        _Cubic(jnp.zeros((1,))),
        jnp.ones((1,)),
        max_steps=max_steps,
        rtol=rtol,
        atol=atol,
        norm_builder=_scaled_by_step,
        **kwargs,
    )


def test_a_stopping_step_decides_when_the_march_stops_and_the_step_s_measure_how_it_steps() -> None:
    """The stop is judged in the stopping step's measure; each step is still reported in its own.

    Steered in a measure ten times looser than the target's, the march would stop after one step
    (``|R| / 10 = 0.0125 <= 0.02``); judged in the target's, it needs a second (``0.125 > 0.02``,
    ``0.0156 <= 0.02``).
    """
    assert len(_march_judged(10.0, atol=0.02).reports) == 1
    judged = _march_judged(10.0, atol=0.02, stopping_step=_judged_on(1.0))
    assert judged.converged
    assert [float(r.residual_norm) for r in judged.reports] == [0.5**3 / 10.0, 0.25**3 / 10.0]


def test_a_stopping_step_s_relative_tolerance_is_relative_to_the_residual_in_its_own_measure() -> (
    None
):
    """``rtol`` scales the stopping step's measure of the starting residual, not the step's own.

    Judged in the target's measure against a reference taken in it (``|R0| = 1``), ``rtol = 0.05`` is
    met at the second step (0.0156). A reference taken in the ten-times-looser steering measure (0.1)
    would demand 0.005 and a third step.
    """
    judged = _march_judged(10.0, rtol=0.05, stopping_step=_judged_on(1.0))
    assert judged.converged
    assert len(judged.reports) == 2


def test_each_report_carries_the_residual_the_stopping_test_read() -> None:
    """``stopping_norm`` / ``stopping_ratio`` are what the stop compared, ``residual_norm`` what steered.

    Judged in the target's measure (ten times the steering one, ``|R0| = 1`` there), the reports read
    ``0.125`` and ``0.0156`` with ratios equal to them; without a stopping step the two pairs agree.
    """
    judged = _march_judged(10.0, atol=0.02, stopping_step=_judged_on(1.0))
    assert [r.stopping_norm for r in judged.reports] == pytest.approx([0.5**3, 0.25**3])
    assert [r.stopping_ratio for r in judged.reports] == pytest.approx([0.5**3, 0.25**3])
    plain = _march_judged(10.0, atol=0.002)
    assert [r.stopping_norm for r in plain.reports] == [r.residual_norm for r in plain.reports]
    assert [r.stopping_ratio for r in plain.reports] == [r.residual_ratio for r in plain.reports]


def test_a_stopping_step_needs_a_measure_built_from_a_step() -> None:
    """Without ``norm_builder`` there is no measure to build from the stopping step."""
    with pytest.raises(ValueError, match="needs a norm_builder"):
        newton_march(
            _JudgedOnItsScale(
                jnp.asarray(1.0), jnp.linalg.norm, ConstantRelaxation(jnp.asarray(1.0))
            ),
            _Cubic(jnp.zeros((1,))),
            jnp.ones((1,)),
            max_steps=1,
            rtol=0.0,
            atol=0.0,
            stopping_step=_judged_on(1.0),
        )


class _ScaledByStep:
    """Measures for :func:`~aquaflux.solve.staged_march` whose row scale is the step's ``scale``."""

    def row_scaled(self, step, state):
        return _scaled_by_step(step, state)

    def block_scaled(self, state):
        raise TypeError("not used")


def test_the_staged_march_judges_its_root_in_the_measure_the_march_stopped_by() -> None:
    """The driver's final convergence test reads the stopping step's measure, as the march did.

    Judged in a target measure ten times looser than the steering one, the march stops after one step
    (``0.0125 <= 0.02``). Judged again in the steering measure (0.125) the driver would refuse that
    state as not a root.
    """
    from aquaflux.solve import NO_REFRESH, Convergence, RowScaled, staged_march
    from aquaflux.solve.driver import FinishedSource

    result = staged_march(
        _Cubic(jnp.zeros((1,))),
        jnp.ones((1,)),
        strategy=_JudgedOnItsScale(
            jnp.asarray(1.0), jnp.linalg.norm, ConstantRelaxation(jnp.asarray(1.0))
        ),
        source=FinishedSource(),
        refresh=NO_REFRESH,
        convergence=Convergence(measure=RowScaled(), rtol=0.0, atol=0.02),
        measures=_ScaledByStep(),
        max_steps=6,
        stopping_step=_judged_on(10.0),
    )
    assert float(result.state[0]) == 0.5


def test_the_staged_march_scales_its_relative_tolerance_by_the_stopping_measure() -> None:
    """``rtol`` at the driver's final test scales the starting residual in the stopping step's measure.

    Steered ten times looser than the target, the march stops at the second step, where the target's
    measure reads 0.0156 against ``0.05 * |R0| = 0.05``. Scaled instead by the starting residual in the
    steering measure (0.1), the driver would demand 0.005 and refuse the state.
    """
    from aquaflux.solve import NO_REFRESH, Convergence, RowScaled, staged_march
    from aquaflux.solve.driver import FinishedSource

    result = staged_march(
        _Cubic(jnp.zeros((1,))),
        jnp.ones((1,)),
        strategy=_JudgedOnItsScale(
            jnp.asarray(10.0), jnp.linalg.norm, ConstantRelaxation(jnp.asarray(1.0))
        ),
        source=FinishedSource(),
        refresh=NO_REFRESH,
        convergence=Convergence(measure=RowScaled(), rtol=0.05, atol=0.0),
        measures=_ScaledByStep(),
        max_steps=6,
        stopping_step=_judged_on(1.0),
    )
    assert float(result.state[0]) == 0.25


def test_a_step_that_can_stop_for_a_refit_is_refused_without_a_hook_to_take_it() -> None:
    """Every hard step would end early and the preconditioner would never be refreshed: refuse it."""
    step = _StopsForRefit(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    with pytest.raises(ValueError, match="refresh_on_cycles=3"):
        newton_march(step, _Cubic(jnp.zeros((1,))), jnp.ones((1,)), max_steps=1, rtol=0.0, atol=0.0)


def test_march_does_not_escalate_below_the_cycle_cap() -> None:
    """A count under ``retry.abort_above_cycles`` never escalates -- the bailout is inert on a comfortable step,
    so it is a safety net, not an every-step cost."""
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    step = _CyclesFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))  # cyc = 40
    result = newton_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(abort_above_cycles=100),
    )
    assert int(result.reports[0].raw_cycles) == 40  # under the cap -> no escalation


def test_a_forced_escalation_adds_no_march_step_compilations() -> None:
    """A β-escalation retry must be a compilation-cache hit, not a recompile of the whole step.

    A retried step redoes ``_march_step`` at an escalated β. The escalation must not change the
    compiled step's cache key -- else a stiff region that retries every step recompiles the (in the
    coupled case, minutes-long) solve each time, which was ~half the march wall. Escalating by
    *scaling* the existing β leaf keeps its abstract value (dtype/weak_type) identical, so the step is
    a cache hit for whatever β dtype the control set. The sensitive case is a **strong-typed** β leaf:
    rebuilding β from a Python float (``jnp.asarray(escalated)``) yields a *weak*-typed leaf, a distinct
    aval that recompiled every escalation. ``_AlphaFromBeta`` never calls the residual, so
    ``_march_step`` traces it exactly once per compile -- the trace count is the compile count.
    """
    # A unique state size, so the escalation's would-be recompile cannot be a cache hit from another
    # test's compiled step (the compilation cache is process-global).
    residual = _Cubic(
        jnp.zeros((13,))
    )  # residual(1) = 1: the held phi never converges -> retry fires
    phi0 = jnp.ones((13,))
    # A strong-typed (non-weak) β leaf -- the case the old `jnp.asarray(float)` escalation recompiled on.
    step = _AlphaFromBeta(
        relaxation_schedule=ConstantRelaxation(jnp.array(1.0, dtype=jnp.float64)), beta_needed=4.0
    )

    # One step with escalation disabled compiles `_march_step` once, and collapses (alpha 0 at β = 1).
    _warm_the_march_norm(step, residual, phi0)
    # Escalation is driven by a COLLAPSED STEP LENGTH, the only thing that escalates: a costly solve
    # redoes at unchanged β, so it would make this test pass without an escalation ever happening.
    _TRACES.clear()
    baseline = newton_march(
        step, residual, phi0, max_steps=1, rtol=1e-10, atol=1e-12, retry=RetryPolicy()
    )
    compiled = len(_TRACES)
    assert compiled == 1 and float(baseline.reports[0].alpha) == 0.0

    # The same step, now escalating β = 1 -> 2 -> 4 (cyc 40 -> 20 -> 10): both escalation `_march_step`
    # calls must reuse the compiled step -- zero further compilations -- while still recovering the step.
    _TRACES.clear()
    escalated = newton_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(on_alpha=0.5, beta_factor=2.0, max_retries=2),
    )
    assert len(_TRACES) == 0  # the escalation retries added no recompiles
    assert float(escalated.reports[0].alpha) == 1.0  # ...and still escalated β to recover the step


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
                return _outcome(jnp.zeros_like(phi), 6, reached=True, residual_fn=residual_fn)
            phi_out = jnp.where(sched.beta >= thr, jnp.zeros_like(phi), jnp.full_like(phi, jnp.inf))
            return _outcome(phi_out, 3, residual_fn=residual_fn)

        return step

    def norm(self):
        return jnp.linalg.norm

    def linear_solver(self):
        return "loose"


def test_march_escalates_beta_before_the_tight_divergence_retry() -> None:
    """A non-finite step that recovers at escalated β is fixed by the CHEAP β-escalation first -- the
    tight ``retry.solver`` (the expensive fallback) never fires. This is the reorder: on the stiff low-β
    saddle a NaN is cured by more damping, so grinding the tight Krylov solve before escalating (the old
    order) was wasted work that the escalation then re-damped away anyway."""
    residual = _Cubic(
        jnp.zeros((1,))
    )  # root at phi = 0; a non-finite phi gives a non-finite residual
    phi0 = jnp.ones((1,))
    step = _NaNUntilDamped(relaxation_schedule=ConstantRelaxation(jnp.asarray(0.5)), threshold=1.0)
    result = newton_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(solver=_TIGHT, on_alpha=0.5, beta_factor=2.0, max_retries=2),
    )
    assert result.converged
    assert jnp.allclose(result.state, 0.0, atol=1e-8)
    assert (
        result.reports[0].raw_cycles == 3
    )  # β=0.5 -> 1.0 recovered it (cost 3); the tight retry (99/6) never fired


def test_march_falls_back_to_the_tight_retry_when_escalation_cannot_fix_divergence() -> None:
    """If β-escalation does not lift the step out of the non-finite regime -- the inexact-PC failure only a
    tighter Krylov solve fixes -- the divergence retry still fires as the fallback, so the reorder never
    loses the recovery it front-runs."""
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    # threshold unreachable by 0.5 -> 1.0 -> 2.0, so every escalation stays non-finite; only "tight" recovers.
    step = _NaNUntilDamped(relaxation_schedule=ConstantRelaxation(jnp.asarray(0.5)), threshold=1e9)
    result = newton_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(solver=_TIGHT, on_alpha=0.5, beta_factor=2.0, max_retries=2),
    )
    assert result.converged
    assert jnp.allclose(result.state, 0.0, atol=1e-8)
    assert (
        result.reports[0].raw_cycles == 6
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
    step = _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    control = DualTimeControl(beta_start=1.0, beta_min=0.01)
    result = newton_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        step_control=control,
        retry=RetryPolicy(on_alpha=0.5, beta_factor=2.0, max_retries=3),
    )
    beta = result.control_state.beta
    assert beta == 4.0  # the escalated β, carried; not the control's beta_start


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
            return _outcome(
                phi, 3, alpha=alpha, residual_fn=residual_fn
            )  # cheap; phi held, never converges

        return step

    def norm(self):
        return jnp.linalg.norm

    def linear_solver(self):
        return None


def test_march_escalates_beta_on_a_collapsed_step_length() -> None:
    """A step whose line-search factor collapsed is redone at a larger β, though its solves were cheap.

    This is the trigger's whole point: the cost bailout cannot see this failure, because 3 cycles is
    under any sane threshold. Here α = 0 until β reaches 4, so β = 1 escalates twice (×2) to 4.
    """
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    step = _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    result = newton_march(
        step,
        residual,
        phi0,
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        # abort_above_cycles far above the step's 3 cycles: only the α trigger can fire
        retry=RetryPolicy(on_alpha=0.0, abort_above_cycles=1000, beta_factor=2.0, max_retries=2),
    )
    assert int(result.reports[0].escalations) == 2
    assert float(result.reports[0].alpha) == 1.0


def test_march_does_not_escalate_on_a_healthy_step_length() -> None:
    """A step taking its full length never escalates -- the trigger is a safety net, not a per-step cost."""
    residual = _Cubic(jnp.zeros((1,)))
    phi0 = jnp.ones((1,))
    # beta_needed = 0 => alpha is 1 from the start.
    step = _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)), beta_needed=0.0)
    result = newton_march(
        step, residual, phi0, max_steps=1, rtol=1e-10, atol=1e-12, retry=RetryPolicy(on_alpha=0.5)
    )
    assert int(result.reports[0].escalations) == 0


def test_the_alpha_trigger_reports_its_own_reason() -> None:
    """``on_retry`` must say ``"alpha"``, not ``"cycles"`` -- the two call for different responses, and a
    log that cannot tell them apart is why the reason is reported at all."""
    reasons: list[str] = []
    residual = _Cubic(jnp.zeros((1,)))
    step = _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)))
    newton_march(
        step,
        residual,
        jnp.ones((1,)),
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(on_alpha=0.0, abort_above_cycles=1000, max_retries=2),
        on_retry=lambda reason, attempt, beta: reasons.append(reason),
    )
    assert reasons == ["alpha", "alpha"]


def test_on_retry_reports_the_beta_the_retried_attempt_will_run_at() -> None:
    """The reported shift is the escalated one, at whatever ``retry.beta_factor`` the march was given.

    ``on_retry`` used to fire *before* the escalation and hand over the abandoned attempt's β, leaving
    its one consumer to reconstruct the real value by multiplying -- which it did, by a literal 2. That
    is right only at the default factor, so this runs at 3: the reported β must be 1 -> 3 -> 9, matching
    the shifts the retried attempts actually run at, and never 2 or 4.
    """
    seen: list[float] = []
    residual = _Cubic(jnp.zeros((1,)))
    step = _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0)), beta_needed=9.0)
    newton_march(
        step,
        residual,
        jnp.ones((1,)),
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        retry=RetryPolicy(on_alpha=0.5, beta_factor=3.0, max_retries=2),
        on_retry=lambda reason, attempt, beta: seen.append(beta),
    )
    assert seen == pytest.approx([3.0, 9.0])


def test_the_alpha_trigger_never_bins_a_step_that_reached_its_target() -> None:
    """A collapsed α on a step that met its own stopping criterion is NOT a reason to redo it.

    Same guard the cost trigger carries: redoing a step that converged discards a good iterate and
    replaces it with a shorter one. Only a step that was cut short is escalated.
    """
    converged = _outcome(jnp.zeros((1,)), 3, alpha=0.0, reached=True)
    cut_short = _outcome(jnp.zeros((1,)), 3, alpha=0.0, reached=False)
    policy = RetryPolicy(on_alpha=0.0)
    assert policy.retry_reason(converged, jnp.asarray(1.0), 1.0) is None
    assert policy.retry_reason(cut_short, jnp.asarray(1.0), 1.0) == "alpha"


def test_the_default_retry_policy_is_the_inert_one() -> None:
    """``NO_RETRIES`` must stay exactly the settings the march had before the policy existed.

    Pinned as *values*, not as "whatever the class defaults to": every march in the tree runs this
    policy unless it says otherwise, so a default moved here silently changes the trajectory of every
    solve -- and a measurement taken under the old one cannot be told apart from one taken under the
    new. The march-level consequence is covered separately (a default policy escalates nothing); this
    is the numbers.
    """
    assert NO_RETRIES == RetryPolicy()
    assert NO_RETRIES.solver is None
    assert NO_RETRIES.divergence_cap == float("inf")
    assert NO_RETRIES.abort_above_cycles is None
    assert NO_RETRIES.on_alpha is None
    assert NO_RETRIES.beta_factor == 2.0
    assert NO_RETRIES.max_retries == 2
    # The whole point of the default: nothing can fire.
    assert not NO_RETRIES.escalates


def test_escalates_reports_whether_the_step_length_threshold_is_set() -> None:
    """Only the STEP-LENGTH threshold arms escalation; the cost budget never raises the shift.

    A cost-only policy still redoes a truncated step, but at unchanged β -- so it needs no readable
    shift leaf, and `escalates` (which gates that requirement) must be False for it.
    """
    assert not RetryPolicy(abort_above_cycles=10).escalates
    assert RetryPolicy(on_alpha=0.01).escalates
    assert RetryPolicy(abort_above_cycles=10, on_alpha=0.01).escalates
    assert not RetryPolicy().escalates
    # A tighter solver is the divergence FALLBACK, not an escalation trigger.
    assert not RetryPolicy(solver=_TIGHT).escalates


def test_escalate_preserves_the_shift_leafs_dtype_and_weak_type() -> None:
    """Escalation scales the leaf, so the escalated step stays a compilation-cache hit.

    Rebuilding the leaf from a Python float (``jnp.asarray(float(beta) * factor)``) yields a weak-typed
    float64 whose abstract value need not match what the step carries, and any mismatch recompiles the
    whole coupled solve on every retry. A strong-typed leaf is the case that catches it -- the shipped
    controls happen to set weak float64, so a weak-only test passes against the defect.
    """
    policy = RetryPolicy(beta_factor=2.0)
    for beta in (jnp.asarray(0.5), jnp.asarray(0.5, dtype=jnp.float64), jnp.float64(0.5)):
        escalated = policy.escalate(beta)
        assert escalated.dtype == jnp.asarray(beta).dtype
        assert jnp.asarray(escalated).weak_type == jnp.asarray(beta).weak_type
        assert float(escalated) == 1.0


def test_a_diverged_step_outranks_the_other_retry_reasons() -> None:
    """Divergence is reported first, and unlike the other two it fires whatever the step's target says --
    a non-finite residual is not a result to be kept because the loop happened to meet its tolerance."""
    outcome = _outcome(jnp.zeros((1,)), 3, alpha=0.0, reached=True)
    reason = RetryPolicy(abort_above_cycles=1, on_alpha=0.0).retry_reason(
        outcome, jnp.asarray(jnp.nan), 1.0
    )
    assert reason == "diverged"


def test_no_retry_reason_when_no_threshold_is_set() -> None:
    """Every threshold ``None`` disables retrying entirely -- the default path, byte-identical."""
    outcome = _outcome(jnp.zeros((1,)), 10_000, alpha=0.0, reached=False)
    assert RetryPolicy().retry_reason(outcome, jnp.asarray(jnp.nan), 1.0) is None


def test_a_step_control_need_not_drive_the_shift() -> None:
    """The protocol asks a control for a ready-to-run step, not for a beta ramp.

    So the escalation's requirement must not be levied on every control: a step-agnostic one is
    legitimate, and gating it would reject the very thing the protocol permits. A control that *does*
    drive beta and is handed a step without one already fails loudly from its own ``tree_at``.
    """
    residual, phi0, _ = _march_and_solver_inputs()
    result = newton_march(
        DampedNewtonStep(line_search=6),
        residual,
        phi0,
        max_steps=2,
        rtol=1e-10,
        atol=1e-12,
        step_control=_CountingControl(),
    )
    assert result.control_state == 2  # ran, on a step carrying no shift at all


def test_the_escalation_refuses_a_step_that_has_no_shift_to_escalate() -> None:
    """The silent failure the gate replaces: a conforming step that simply never escalated.

    ``DampedNewtonStep`` satisfies ``NewtonStrategy`` in full -- all four declared methods -- so a march
    configured to escalate accepted one and then skipped every escalation, because a ``hasattr`` deep
    in the loop came back False. From the log that is indistinguishable from a march that never needed
    to escalate, which is the worst way for a safety net to be missing.

    Asked of the policy, because it is the policy's requirement: the tight-solver fallback next door
    has no such need (see the test below).
    """
    residual = _Cubic(jnp.zeros((1,)))
    # Cost-only is deliberately absent: it redoes at unchanged β, so it needs no shift to read.
    for policy in (RetryPolicy(on_alpha=0.5), RetryPolicy(on_alpha=0.01, abort_above_cycles=3)):
        with pytest.raises(TypeError, match="relaxation_schedule"):
            newton_march(
                DampedNewtonStep(),
                residual,
                jnp.ones((1,)),
                max_steps=1,
                rtol=1e-10,
                atol=1e-12,
                retry=policy,
            )


def test_the_divergence_retry_works_on_a_step_with_no_shift() -> None:
    """It re-solves at a tighter tolerance and never touches beta, so it must not demand one.

    This is the path that used to *raise*: announcing the retry read
    ``active_step.relaxation_schedule`` unguarded, so a step satisfying the forward-step contract in
    full crashed mid-march. The same contract gap as the test above, failing the opposite way -- one
    silently did nothing, this one crashed -- which is why gating the escalation must not gate this.

    ``_PoisonUnlessTight`` carries no ``relaxation_schedule`` at all and genuinely diverges under the
    loose solver, so the retry and its announcement actually fire.
    """
    seen: list[tuple[str, float]] = []
    result = newton_march(
        _PoisonUnlessTight(),
        _Cubic(jnp.zeros((1,))),
        jnp.ones((1,)),
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        solver="loose",
        retry=RetryPolicy(solver=_TIGHT),
        on_retry=lambda reason, attempt, beta: seen.append((reason, beta)),
    )

    assert seen == [("solver", 0.0)]  # announced, with no shift invented for a step that has none
    assert bool(result.reports[0].diverged_retry)
    assert result.reports[0].shift == 0.0


class _UnshiftedStep(eqx.Module):
    """A step whose schedule carries no readable ``beta`` -- what a builder hands over BEFORE a control.

    This is not a contrived shape: the dual-time builders return exactly this, a
    ``SwitchedEvolutionRelaxation`` with no ``beta``, and rely on a ``StepControl`` to swap in a
    ``ConstantRelaxation`` per iteration.
    """

    relaxation_schedule: SwitchedEvolutionRelaxation = SwitchedEvolutionRelaxation()

    def stepper(self):
        def step(residual_fn, phi, residual_norm_0, solver):
            return _outcome(
                phi, 40, residual_fn=residual_fn
            )  # held: never converges, escalation fires

        return step

    def norm(self):
        return jnp.linalg.norm

    def linear_solver(self):
        return None


class _InstallsBeta(eqx.Module):
    """A ``StepControl`` that installs a readable ``beta``, as the dual-time controls do."""

    def next_step(self, base, previous, state):
        return _CyclesFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0))), state


def test_the_escalation_guard_accepts_a_shift_a_step_control_installs() -> None:
    """A base step with no readable ``beta`` is fine when a ``StepControl`` installs one.

    The regression this pins: the guard was applied to the BASE step before the loop, where the
    dual-time family always fails it -- its builder hands over a ``SwitchedEvolutionRelaxation`` and
    the control swaps in a ``ConstantRelaxation`` per iteration. That rejected the shipped coupled
    configuration outright, and the flagship 3D validation case died on its first step with a
    ``TypeError`` naming a step type the message itself listed as acceptable.

    Checking the step that will actually run -- after the control has shaped it -- is what makes the
    guard mean what it says.
    """
    # The base step genuinely FAILS the guard -- which is what makes this a regression test rather
    # than a tautology: applied here, as it was, it rejects the march outright.
    with pytest.raises(TypeError, match="relaxation_schedule"):
        RetryPolicy(abort_above_cycles=10).require_shifted(_UnshiftedStep())

    residual = _Cubic(jnp.zeros((1,)))
    result = newton_march(
        _UnshiftedStep(),
        residual,
        jnp.ones((1,)),
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        step_control=_InstallsBeta(),
        retry=RetryPolicy(abort_above_cycles=10, beta_factor=2.0, max_retries=2),
    )
    # It ran, and the escalation actually fired -- the guard let through a step that can escalate.
    assert int(result.reports[0].escalations) > 0


def test_the_escalation_guard_still_rejects_a_step_that_can_never_escalate() -> None:
    """With no control to install one, a step carrying no ``beta`` is rejected -- loudly, not silently.

    This is the failure the guard exists for: a ``DampedNewtonStep`` satisfies ``NewtonStrategy`` in full,
    so a march configured to escalate used to accept one and then never escalate, indistinguishable in
    the log from a march that never needed to.
    """
    with pytest.raises(TypeError, match="relaxation_schedule"):
        newton_march(
            DampedNewtonStep(line_search=0),
            _Cubic(jnp.zeros((1,))),
            jnp.ones((1,)),
            max_steps=1,
            rtol=1e-10,
            atol=1e-12,
            retry=RetryPolicy(on_alpha=0.5),
        )


def test_a_march_step_does_not_re_evaluate_the_residual_it_was_just_handed() -> None:
    """The step reports its own measure, so the march forms no residual of its own.

    A globalized step ends in a line search that evaluated ``norm(R(phi + alpha delta))`` at every rung
    it walked, so the value at the rung it kept is in hand when the step returns. The march used to
    throw it away and evaluate the residual again at that same iterate -- a full residual per step.

    Counted at TRACE time, which is what makes this deterministic: ``_march_step`` is compiled once, so
    how many times the residual appears in it is a static property of the program rather than a timing.
    Three is the floor here -- the Newton correction's own ``R(phi)``, its Jacobian--vector product, and
    the line-search ladder's body -- and a fourth would be the re-evaluation this pins out.
    """
    for line_search in (0, 6):
        _TRACES.clear()
        step = DampedNewtonStep(line_search=line_search)
        residual = _Cubic(jnp.array([8.0, 27.0]))
        _march_step(step, residual, jnp.array([1.5, 2.5]), jnp.asarray(10.0), step.linear_solver())
        assert len(_TRACES) == 3, f"line_search={line_search} traced {len(_TRACES)} residuals"


def test_every_strategy_reports_the_measure_at_the_iterate_it_returns() -> None:
    """The contract both drivers now rely on, checked on all three shipped strategies.

    ``StepOutcome.residual_norm`` must be ``norm(R(phi))`` at the step's OWN returned iterate. A
    stepper that reported the measure of a rung it did not keep -- the line search's fallback rung is
    the one that can differ from the last it evaluated -- would give the march a residual history
    describing steps it never took, and the convergence test would run on it.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    residual_fn = _Cubic(theta)
    phi0 = jnp.array([1.2, 2.2, 3.2])

    for step in (
        DampedNewtonStep(line_search=0),
        DampedNewtonStep(line_search=8),
        PseudoTransientStep(
            shift_policy=_UnitShiftPolicy(),
            relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0, exponent=1.0),
            line_search=8,
        ),
    ):
        r0 = step.norm()(residual_fn(phi0))
        outcome = step.stepper()(residual_fn, phi0, r0, step.linear_solver())
        expected = step.norm()(residual_fn(outcome.phi))
        assert jnp.allclose(outcome.residual_norm, expected, rtol=1e-12, atol=0.0), type(step)


def test_the_line_search_reports_the_fallback_rungs_own_measure() -> None:
    """When nothing is admissible the kept rung is not the last one walked -- and the pair must agree.

    The ladder walks longest-first and stops at the first admissible rung; with none, it falls back to
    the longest rung that is finite and within ``fallback_growth`` of ``reference_norm`` -- here the
    full step, at 5x a ``reference_norm`` of 1 -- which it passed several evaluations earlier. Carrying
    the accepted rung's measure alone would then describe a step that was not taken, so the fallback
    carries its own.
    """
    phi, delta, reference = jnp.array([1.0]), jnp.array([4.0]), jnp.asarray(1.0)

    def residual(p):  # every rung increases the residual, so nothing is admissible
        return p

    step = backtracking_line_search(residual, phi, delta, reference, steps=4)

    assert (
        float(step.alpha) == 1.0
    )  # the fallback: the longest qualifying rung, capped at the full step
    assert jnp.allclose(step.phi, phi + delta)
    assert jnp.allclose(step.residual_norm, jnp.linalg.norm(residual(step.phi)))


def test_a_search_with_no_finite_rung_reports_a_NON_FINITE_measure() -> None:
    """The one path that reads the carry's seed, and the one place its value matters.

    When not a single rung is finite the ladder has no measure to report, so the carry's seed is what
    comes out. It must stay non-finite: every consumer of this number tests it with ``isfinite`` --
    the divergence guard, the acceptance policy, and the Newton driver's convergence guard -- and a
    finite seed (``0.0``, say) would present a poisoned step to all three as a converged one.

    The step itself is unchanged either way: ``alpha`` is still the shortest rung, which the ladder
    evaluated and found non-finite, so the residual a caller would have recomputed there is non-finite
    too. Only the route to that conclusion changed.
    """
    phi, delta, reference = jnp.array([1.0]), jnp.array([1.0]), jnp.asarray(1.0)

    def poisoned(p):
        return jnp.full_like(p, jnp.nan)

    step = backtracking_line_search(poisoned, phi, delta, reference, steps=4)

    assert not bool(jnp.isfinite(step.residual_norm))
    assert float(step.alpha) == 0.5**4  # the shortest rung, exactly as before
    # ...and it agrees with what a caller recomputing at that iterate would have found.
    assert not bool(jnp.isfinite(jnp.linalg.norm(poisoned(step.phi))))


def test_the_fallback_refuses_a_rung_that_is_finite_but_an_excursion() -> None:
    """ "Finite" is far too weak a bound on a fallback rung, and unbounded it poisons the march.

    Measured on a coupled march: with nothing admissible, the search returned a rung it had itself
    evaluated at ~1.5e+129 times the reference. A dual-time step has no acceptance policy and the
    march's divergence test defaults to catching only non-finiteness, so that rung became the next
    anchor. The fallback exists to avoid a null step, not to license an excursion.
    """
    phi, delta, reference = jnp.array([1.0]), jnp.array([1.0]), jnp.asarray(1.0)

    def blows_up(p):  # every rung is finite, and every rung is enormous
        return jnp.exp(30.0 * p)

    unbounded = backtracking_line_search(
        blows_up, phi, delta, reference, steps=4, fallback_growth=jnp.inf
    )
    bounded = backtracking_line_search(blows_up, phi, delta, reference, steps=4)

    # Unbounded, the search hands back the longest finite rung however vast its measure.
    assert float(unbounded.alpha) == 1.0
    assert float(unbounded.residual_norm) > 1e20
    # Bounded, no finite rung is under the ceiling, so no step is taken at all -- and the reported
    # measure is the untouched iterate's own, not the excursion's.
    assert float(bounded.alpha) == 0.0
    assert jnp.allclose(bounded.phi, phi)
    assert jnp.allclose(bounded.residual_norm, reference)


def test_the_bounded_fallback_prefers_a_shorter_rung_to_no_step() -> None:
    """The ceiling shortens the fallback where it can, and only gives up when nothing fits under it."""
    phi, delta, reference = jnp.array([1.0]), jnp.array([1.0]), jnp.asarray(1.0)

    def steep(p):  # alpha=1 -> 1.2e5 (over a ceiling of 100), alpha=1/2 -> 3.5e2, alpha=1/4 -> ~19
        return jnp.exp(11.7 * p) / jnp.exp(11.7)

    step = backtracking_line_search(steep, phi, delta, reference, steps=6, fallback_growth=100.0)

    assert 0.0 < float(step.alpha) < 1.0  # a real step, shorter than the full one
    assert float(step.residual_norm) <= 100.0 * float(reference)
    assert jnp.allclose(step.residual_norm, jnp.linalg.norm(steep(step.phi)))


def test_a_healthy_fallback_is_untouched_by_the_ceiling() -> None:
    """The ceiling must not disturb an ordinary fallback, which is the common non-admissible case.

    Across two healthy marches of the coupled case the fallback fired on 0 of 360 inner iterations,
    and the largest benign fallback observed anywhere was 4.5x the reference -- more than twenty
    decades below the default ceiling. This pins that a modest fallback still takes the full step.
    """
    phi, delta, reference = jnp.array([1.0]), jnp.array([4.0]), jnp.asarray(1.0)

    def residual(p):  # every rung increases the residual; the full step lands at 5x the reference
        return p

    assert float(backtracking_line_search(residual, phi, delta, reference, steps=4).alpha) == 1.0


# --- the within-march homotopy -----------------------------------------------------------------


class _CubicRamp:
    """A ``ResidualHomotopy`` over ``_Cubic``: walk ``theta`` from a start value down to the target.

    The stations are cube roots of the states they converge to, so a station's root is a *different*
    point from the target's -- which is what makes "did the march stop at the wrong station?"
    observable rather than a distinction only the bookkeeping can see.
    """

    def __init__(self, target, start, stations, steps_per_station, redamping=1.0):
        self.target = target
        self.start = start
        self.stations = stations
        self.steps_per_station = steps_per_station
        # Defaults to 1.0 -- NO re-damping -- so a test of the rebase isolates the rebase. The two
        # mechanisms fire on adjacent steps for different reasons, and a fixture that switched both on
        # could not tell which one moved beta.
        self.redamping = redamping
        self.entered: list[int] = []  # one entry per `enter` call: the station it was asked for
        self.previous: list[
            int | None
        ] = []  # one per `enter` call: the step of the report handed in
        self.changes: list[float] = []  # one entry per station CHANGE: the theta it moved to
        self._station = -1
        self._residual = _Cubic(target)

    def station(self, step: int) -> int:
        return min(step // self.steps_per_station, self.stations)

    def enter(self, step: int, previous):
        station = self.station(step)
        self.entered.append(station)
        self.previous.append(None if previous is None else previous.step)
        if station != self._station:
            self._station = station
            frac = 1.0 - station / self.stations
            theta = (
                self.target * (self.start / self.target) ** frac
                if station < self.stations
                else self.target
            )
            self._residual = _Cubic(theta)
            self.changes.append(float(jnp.asarray(theta).ravel()[0]))
        return self._residual

    def arrived(self, step: int) -> bool:
        return self.station(step) >= self.stations

    def shift_factor(self, step: int) -> float:
        del step
        return self.redamping


def test_the_station_hook_reshapes_the_step_for_the_station_it_is_about_to_run() -> None:
    """The seam by which a change of PROBLEM can be matched by a change to how the step is DAMPED.

    Every signal a shift policy can read for itself -- the shift strength, the residual -- measures
    *progress*, which is a different quantity from *which station this is*; on a short ramp they come
    apart completely, since progress saturates long before the last station begins. So the station
    index has to arrive from the march, which is the only thing that knows it.
    """
    residual, phi0, _root = _march_and_solver_inputs()
    ramp = _CubicRamp(target=8.0, start=64.0, stations=3, steps_per_station=2)
    seen: list[tuple[int, bool]] = []

    def station_step(step, station, arrived):
        seen.append((station, arrived))
        return step

    newton_march(
        DampedNewtonStep(line_search=10),
        residual,
        phi0,
        max_steps=8,
        rtol=1e-10,
        atol=1e-12,
        homotopy=ramp,
        station_step=station_step,
    )

    assert seen, "the hook was never called"
    # One call per outer step, carrying that step's own station. Compared against `station(i)` rather
    # than against the homotopy's `entered` log, which carries one extra leading entry: the march
    # enters station 0 once BEFORE the loop, to take the reference norm.
    assert [station for station, _ in seen] == [ramp.station(i) for i in range(len(seen))]
    assert [arrived for _, arrived in seen] == [station >= 3 for station, _ in seen]


def test_the_step_the_station_hook_RETURNS_is_the_one_that_is_taken() -> None:
    """A hook whose result were discarded would look identical from the outside -- the march would run,
    and the reshaping would simply never happen. That is the failure mode the per-block damping already
    hit twice, so pin the return value rather than the call."""
    residual, phi0, _root = _march_and_solver_inputs()
    ramp = _CubicRamp(target=8.0, start=64.0, stations=3, steps_per_station=2)

    def crippled(step, station, arrived):
        del step, station, arrived
        return DampedNewtonStep(line_search=0)  # no line search at all

    plain = newton_march(
        DampedNewtonStep(line_search=10),
        residual,
        phi0,
        max_steps=8,
        rtol=1e-10,
        atol=1e-12,
        homotopy=_CubicRamp(8.0, 64.0, 3, 2),
    )
    reshaped = newton_march(
        DampedNewtonStep(line_search=10),
        residual,
        phi0,
        max_steps=8,
        rtol=1e-10,
        atol=1e-12,
        homotopy=ramp,
        station_step=crippled,
    )

    assert not jnp.array_equal(plain.state, reshaped.state)


def test_the_station_hook_is_ignored_without_a_homotopy_and_is_byte_identical_when_absent() -> None:
    """There are no stations without a homotopy, so there is nothing for the hook to key on."""
    residual, phi0, _root = _march_and_solver_inputs()
    step = DampedNewtonStep(line_search=10)
    kwargs = dict(max_steps=50, rtol=1e-10, atol=1e-12)

    def explode(_step, _station, _arrived):
        raise AssertionError("the hook must not be consulted without a homotopy")

    without = newton_march(step, residual, phi0, **kwargs)
    with_hook = newton_march(step, residual, phi0, station_step=explode, **kwargs)

    assert jnp.array_equal(without.state, with_hook.state)
    assert len(without.reports) == len(with_hook.reports)


def test_a_march_with_no_homotopy_is_unchanged() -> None:
    """The default must be byte-identical -- an added seam that moves the incumbent path is a defect."""
    residual, phi0, _root = _march_and_solver_inputs()
    step = DampedNewtonStep(line_search=10)
    kwargs = dict(max_steps=50, rtol=1e-10, atol=1e-12)

    without = newton_march(step, residual, phi0, **kwargs)
    explicit_none = newton_march(step, residual, phi0, homotopy=None, **kwargs)

    assert without.converged and explicit_none.converged
    assert jnp.array_equal(without.state, explicit_none.state)
    assert len(without.reports) == len(explicit_none.reports)


def test_the_march_reaches_the_TARGET_root_through_the_ramp() -> None:
    """The homotopy dissolves: the state the march returns solves the target residual, not a station's.

    The stations converge to visibly different roots (theta walks a decade), so landing on the target's
    cube root is not something an unwired homotopy could produce by accident.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    homotopy = _CubicRamp(target, start=target * 100.0, stations=3, steps_per_station=2)
    step = DampedNewtonStep(line_search=10)

    result = newton_march(
        step,
        _Cubic(target),
        jnp.ones_like(target),
        max_steps=60,
        rtol=1e-10,
        atol=1e-12,
        homotopy=homotopy,
    )

    assert result.converged
    assert jnp.allclose(result.state, jnp.cbrt(target), atol=1e-8)


def test_the_march_genuinely_solves_each_station_not_only_the_target() -> None:
    """Reachability: the residual the homotopy hands back is the one the steps actually drive.

    An unwired ``homotopy`` leaves the march bit-identical to one that ignored it, and every property
    asserted of it would then hold for a reason unrelated to the seam. Marching a ramp whose stations
    are far from the target and stopping the moment the ramp ends shows the state has been carried to
    the LAST STATION's root -- which only happens if the stations were the residuals being solved.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    # Three ramp stations of six steps each, and a budget of exactly those eighteen: the march is cut
    # off the moment the ramp ends, so what it holds is the ramp's work rather than the target's. Six
    # steps is comfortably enough for a damped Newton to settle this cubic at each station, so the
    # state is AT a station's root and not somewhere between two of them.
    homotopy = _CubicRamp(target, start=target * 1000.0, stations=3, steps_per_station=6)
    step = DampedNewtonStep(line_search=10)

    ramped = newton_march(
        step,
        _Cubic(target),
        jnp.ones_like(target),
        max_steps=18,
        rtol=1e-10,
        atol=1e-12,
        homotopy=homotopy,
    )
    ignored = newton_march(
        step,
        _Cubic(target),
        jnp.ones_like(target),
        max_steps=18,
        rtol=1e-10,
        atol=1e-12,
    )

    # The two marches spent the same budget on demonstrably different problems: the one that ignored
    # the ramp sat on the target's root the whole time.
    assert not ramped.converged
    assert jnp.allclose(ignored.state, jnp.cbrt(target), atol=1e-8)
    # The LAST RAMP STATION's root is where the ramped march is -- theta a decade above the target.
    last_station_theta = target * (1000.0 ** (1.0 - 2 / 3))
    assert jnp.allclose(ramped.state, jnp.cbrt(last_station_theta), rtol=1e-6)


def test_the_march_may_not_stop_on_tolerance_before_the_ramp_arrives() -> None:
    """A converged intermediate station is not a converged answer, and the stopping test must know it.

    The march is started AT the first station's exact root, so its residual is zero on step one and the
    tolerance test is satisfied immediately. Without the arrival gate the march would return there and
    report ``converged`` -- at a state that does not solve the target at all.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    start = target * 100.0
    homotopy = _CubicRamp(target, start=start, stations=3, steps_per_station=2)
    step = DampedNewtonStep(line_search=10)

    result = newton_march(
        step,
        _Cubic(target),
        jnp.cbrt(start),
        max_steps=60,
        rtol=1e-10,
        atol=1e-12,
        homotopy=homotopy,
    )

    assert len(result.reports) > homotopy.stations * homotopy.steps_per_station
    assert result.converged
    assert jnp.allclose(result.state, jnp.cbrt(target), atol=1e-8)


def test_a_march_cut_off_inside_the_ramp_does_not_report_convergence() -> None:
    """``converged`` is the caller's guard, and mid-ramp it must be False however small the residual is.

    Same construction as above -- started on a station's own root, so the residual test passes on every
    ramp step -- but the budget ends before arrival. A ``True`` here would hand a driver an
    intermediate-viscosity state as if it were the answer.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    start = target * 100.0
    homotopy = _CubicRamp(target, start=start, stations=4, steps_per_station=2)
    step = DampedNewtonStep(line_search=10)

    result = newton_march(
        step,
        _Cubic(target),
        jnp.cbrt(start),
        max_steps=3,
        rtol=1e-10,
        atol=1e-12,
        homotopy=homotopy,
    )

    assert not result.converged
    assert len(result.reports) == 3


def test_the_homotopy_is_entered_once_per_step_and_changes_once_per_station() -> None:
    """A station spans several steps: whatever a change costs downstream is paid per change, not per step.

    This is the property that makes the design affordable -- the motivating consumer refits a frozen
    preconditioner on each change -- so it is pinned rather than left to the implementation's goodwill.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    homotopy = _CubicRamp(target, start=target * 100.0, stations=3, steps_per_station=4)
    step = DampedNewtonStep(line_search=10)

    result = newton_march(
        step,
        _Cubic(target),
        jnp.ones_like(target),
        max_steps=12,
        rtol=1e-10,
        atol=1e-12,
        homotopy=homotopy,
    )

    steps = len(result.reports)
    # `enter` is also called once before the loop, for the damping anchor at the first station.
    assert homotopy.entered == [0] + [homotopy.station(i) for i in range(steps)]
    # Four steps per station over 12 steps: stations 0, 1, 2 -- three changes, not twelve.
    assert len(homotopy.changes) == 3


def test_each_station_is_entered_with_the_report_of_the_step_before_it() -> None:
    """A homotopy may choose where a step runs from how the march went, so it is told -- in order.

    The first step is entered twice (once for the damping anchor, once to take it), with no report;
    every later step is handed the report of the step just before it, never an older one.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    homotopy = _CubicRamp(target, start=target * 100.0, stations=3, steps_per_station=2)

    result = newton_march(
        DampedNewtonStep(line_search=10),
        _Cubic(target),
        jnp.ones_like(target),
        max_steps=10,
        rtol=1e-10,
        atol=1e-12,
        homotopy=homotopy,
    )

    steps = len(result.reports)
    assert steps > 3
    assert homotopy.previous == [None, None, *range(steps - 1)]


def test_each_report_says_which_station_its_step_drove_and_whether_that_was_the_target() -> None:
    """A record of a march can mark where the path ended and the case began, from the reports alone."""
    target = jnp.array([8.0, 27.0, 64.0])
    homotopy = _CubicRamp(target, start=target * 100.0, stations=3, steps_per_station=2)
    result = newton_march(
        DampedNewtonStep(line_search=10),
        _Cubic(target),
        jnp.ones_like(target),
        max_steps=10,
        rtol=1e-10,
        atol=1e-12,
        homotopy=homotopy,
    )
    steps = range(len(result.reports))
    assert len(result.reports) > 6
    assert [report.station for report in result.reports] == [homotopy.station(i) for i in steps]
    assert [report.arrived for report in result.reports] == [homotopy.arrived(i) for i in steps]
    assert not result.reports[0].arrived and result.reports[-1].arrived

    plain = newton_march(
        DampedNewtonStep(line_search=10),
        _Cubic(target),
        jnp.ones_like(target),
        max_steps=3,
        rtol=1e-10,
        atol=1e-12,
    )
    assert {(report.station, report.arrived) for report in plain.reports} == {(0, True)}


# --- rebasing a residual-keyed control at a station boundary ------------------------------------


class _RecordingCfl(CflResidualDualTimeControl):
    """A ``CflResidualDualTimeControl`` that records the ``(beta, memo)`` it is handed each call.

    The rebase happens to the state *before* ``next_step`` sees it, so the memo the control actually
    adapts from is the observable that distinguishes a rebased march from one that is not. Recording
    it here reads that directly rather than inferring it from the beta trajectory, which several other
    effects also move.
    """

    seen: list = eqx.field(static=True, default_factory=list)

    def next_step(self, base_step, previous, state):
        self.seen.append(None if state is None else state[1])
        return super().next_step(base_step, previous, state)


class _NoRebase:
    """A step control that delegates ``next_step`` and deliberately exposes no ``rebase``.

    Reproduces the march exactly as it ran before the rebase existed, which is what makes the
    comparison below a test of the fix rather than of the fixture. It is also a real case and not only
    a stub: :func:`~aquaflux.solve.newton_march` calls the rebase only on a
    :class:`~aquaflux.solve.ShiftCarryingControl`, so a third-party control that is not one degrades to
    precisely this behaviour.
    """

    def __init__(self, inner):
        self.inner = inner

    def next_step(self, base_step, previous, state):
        return self.inner.next_step(base_step, previous, state)


def _ramp_march(homotopy, control, steps, start=1000.0):
    """Run a short controlled march over ``homotopy``'s stations on a shifted step.

    ⚠️ Starts **near** the first station's root, not on it. Landing exactly on it makes ``‖R₀‖``
    zero, and that anchor is what the step's line search measures descent against -- with a zero
    reference nothing is admissible, every step falls through to the non-descent fallback, and the
    march freezes with ``alpha = 1.000`` and an unchanging residual. A test built on that fixture
    passes without exercising anything, which is the failure easiest to mistake for a result.
    """
    theta = jnp.array([8.0, 27.0, 64.0])
    base = PseudoTransientStep(
        _UnitShiftPolicy(),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=1.0),
        line_search=8,
    )
    return newton_march(
        base,
        _Cubic(theta),
        jnp.cbrt(theta * start) * 1.05,
        max_steps=steps,
        rtol=1e-12,
        atol=1e-14,
        step_control=control,
        homotopy=homotopy,
    )


def test_rebase_drops_the_remembered_residual_and_keeps_the_shift() -> None:
    """The mirror of ``carry_beta``: β survives, the reference does not."""
    control = CflResidualDualTimeControl(beta_start=0.5)

    assert control.rebase((0.25, 3.5)) == ShiftControlState(0.25)
    assert control.carry_beta((0.25, 3.5), 0.5) == ShiftControlState(
        0.5, 3.5
    )  # the sibling, unchanged


def test_rebasing_a_memoryless_control_changes_nothing() -> None:
    """``DualTimeControl`` keys on α alone, so it carries no cross-problem reference to invalidate."""
    control = DualTimeControl(beta_start=0.5)

    assert control.rebase((0.25, None)) == ShiftControlState(0.25)


def test_the_march_rebases_the_control_at_a_station_change() -> None:
    """Reachability: the control adapts from ``memo is None`` on the step after a station change.

    Read off the state the control is actually handed, rather than off β -- β is moved by the grow
    rule, the back-off rule and the clamp as well, so asserting on it would pass for reasons unrelated
    to the rebase.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    homotopy = _CubicRamp(target, start=target * 1000.0, stations=3, steps_per_station=2)
    control = _RecordingCfl(beta_start=0.5, seen=[])

    result = _ramp_march(homotopy, control, steps=6)

    steps = len(result.reports)
    # Station changes at steps 2 and 4, so the calls that STRADDLE one are those for steps 3 and 5:
    # each divides step (n-1)'s residual by step (n-2)'s across the boundary.
    straddling = [
        n for n in range(steps) if n >= 2 and homotopy.station(n - 1) != homotopy.station(n - 2)
    ]
    assert straddling, "the fixture must actually cross a station boundary"
    for n in straddling:
        assert control.seen[n] is None, f"step {n} adapted across a station change"


def test_a_station_change_does_not_brake_the_shift() -> None:
    """The defect this fixes: a harder new station raises the residual, and the control read it as
    an overshoot and doubled β.

    The unrebased arm is the control, and it is what makes this a test of the fix rather than of the
    fixture -- the same march, the same stations, differing only in whether the reference is dropped.
    """
    target = jnp.array([8.0, 27.0, 64.0])

    def run(rebasing):
        homotopy = _CubicRamp(target, start=target * 1000.0, stations=3, steps_per_station=2)
        control = CflResidualDualTimeControl(beta_start=0.5, backoff=2.0)
        return _ramp_march(homotopy, control if rebasing else _NoRebase(control), steps=6)

    braked = run(rebasing=False)
    fixed = run(rebasing=True)

    # Every step of the rebased arm was comfortable, so nothing legitimately justified a brake.
    assert all(r.alpha >= 0.99 for r in fixed.reports)
    assert max(r.shift for r in fixed.reports) <= max(r.shift for r in braked.reports)


def test_no_station_change_brakes_the_shift_however_far_the_residual_JUMPS() -> None:
    """The strong form, at EVERY boundary rather than at the ones that happened to fire.

    The weaker test above compares two arms on aggregate shift, which a fix that ran at only some
    boundaries would still pass -- and on the real march only two of four station changes braked, so
    "it did not brake here" is indistinguishable from "the rebase silently did not run here" without
    forcing the issue. This gives every station enough steps to converge, so each change lands a
    residual jump of many orders (a converged station's residual is ~0; the next station's, at the
    same state, is the whole gap between their roots). Every boundary then *must* brake if the
    reference is not re-based.

    Both directions are asserted at each boundary, which is what makes the fixture honest: the
    unrebased arm proves the jump really does trip the brake, and the rebased arm proves the fix
    catches all of them and not a subset.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    stations, per = 3, 4  # four steps is ample for a damped Newton to settle this cubic

    def run(rebasing):
        homotopy = _CubicRamp(
            target, start=target * 1000.0, stations=stations, steps_per_station=per
        )
        control = CflResidualDualTimeControl(beta_start=0.5, backoff=2.0)
        return homotopy, _ramp_march(
            homotopy, control if rebasing else _NoRebase(control), steps=stations * per
        )

    _h_braked, braked = run(rebasing=False)
    homotopy, fixed = run(rebasing=True)

    # The step that ADAPTS across a boundary is the one after the first step of a new station.
    straddling = [
        n
        for n in range(len(fixed.reports))
        if n >= 2 and homotopy.station(n - 1) != homotopy.station(n - 2)
    ]
    assert len(straddling) >= 2, "the fixture must cross at least two station boundaries"

    for n in straddling:
        assert braked.reports[n].shift > braked.reports[n - 1].shift, (
            f"step {n} did not brake without the rebase -- the fixture is not tripping the rule, "
            f"so the rebased arm passing below would prove nothing"
        )
        assert fixed.reports[n].shift <= fixed.reports[n - 1].shift, (
            f"step {n} braked despite the rebase -- the reference survived a station change"
        )


# --- deliberate re-damping on entering a station ------------------------------------------------


def test_entering_a_station_re_damps_the_shift_by_the_homotopy_s_factor() -> None:
    """The shift is multiplied by ``shift_factor`` on the step that first faces a new station.

    Exactly the factor, not the factor further adapted: the control is HELD for that step, so the
    re-damping is the step's whole adaptation rather than something the grow rule immediately erodes.
    """
    target = jnp.array([8.0, 27.0, 64.0])
    homotopy = _CubicRamp(
        target, start=target * 1000.0, stations=3, steps_per_station=2, redamping=2.0
    )
    control = DualTimeControl(beta_start=0.5, beta_min=1e-6, beta_max=1e6, grow=1.5)

    result = _ramp_march(homotopy, control, steps=6)

    shifts = [r.shift for r in result.reports]
    entering = [n for n in range(1, len(shifts)) if homotopy.station(n) != homotopy.station(n - 1)]
    assert entering, "the fixture must cross a station boundary"
    for n in entering:
        assert shifts[n] == pytest.approx(shifts[n - 1] * 2.0), (
            f"step {n} entered a station without the re-damping being applied cleanly"
        )


def test_re_damping_is_off_by_default_so_a_plain_homotopy_march_is_unchanged() -> None:
    """``shift_factor`` of 1.0 must leave the march byte-identical to one that never asks.

    The protocol's own default is 1.0, so a homotopy that does not care about the shift pays nothing
    and behaves exactly as it did before this seam existed.
    """
    target = jnp.array([8.0, 27.0, 64.0])

    def run(redamping):
        homotopy = _CubicRamp(
            target, start=target * 1000.0, stations=3, steps_per_station=2, redamping=redamping
        )
        return _ramp_march(homotopy, DualTimeControl(beta_start=0.5), steps=6)

    assert [r.shift for r in run(1.0).reports] == [r.shift for r in run(1.0).reports]
    assert [r.shift for r in run(2.0).reports] != [r.shift for r in run(1.0).reports]


def test_re_damping_keeps_the_shift_above_a_level_the_undamped_ramp_walks_through() -> None:
    """The property the pitzDaily failure is about: the undamped ramp descends far further.

    Within a station the control divides the shift by ``grow`` each step, so the descent per station
    is ``grow ** steps_per_station``; re-damping by ``factor`` leaves a net of
    ``grow ** steps / factor``. This pins that the damped arm's *minimum* shift stays well above the
    undamped arm's -- which on the real case is the difference between approaching a wall at
    ``beta ~ 0.012`` and stopping short of it.
    """
    target = jnp.array([8.0, 27.0, 64.0])

    def run(redamping):
        homotopy = _CubicRamp(
            target, start=target * 1000.0, stations=4, steps_per_station=3, redamping=redamping
        )
        control = DualTimeControl(beta_start=0.5, beta_min=1e-9, beta_max=1e6, grow=1.5)
        return _ramp_march(homotopy, control, steps=12)

    undamped = min(r.shift for r in run(1.0).reports)
    damped = min(r.shift for r in run(2.0).reports)

    # Three station changes over four stations. Each is worth ``factor * grow``, not ``factor``: the
    # control is HELD on an entering step, so that step also forgoes its usual division by ``grow``.
    # That is not an accident of the implementation -- it is what reproduces the behaviour the
    # accidental brake had, which replaced a grow division rather than adding to one, and which is the
    # arm that converged in 36 steps.
    assert damped > undamped
    assert damped / undamped == pytest.approx((2.0 * 1.5) ** 3, rel=1e-6)


class _NormUnderAnotherName(eqx.Module):
    """A conforming strategy that keeps its measure somewhere other than a ``residual_norm`` field.

    It delegates to a :class:`DampedNewtonStep` held one level down, so the measure lives at
    ``inner.residual_norm``. :meth:`with_norm` is the declared way to swap it; nothing else about the
    step tells the march where it is.
    """

    inner: DampedNewtonStep

    def stepper(self):
        return self.inner.stepper()

    def linear_solver(self):
        return self.inner.linear_solver()

    def norm(self):
        return self.inner.norm()

    def with_norm(self, norm):
        return eqx.tree_at(lambda s: s.inner.residual_norm, self, norm)

    def adjoint_preconditioner(self):
        return None


def test_the_rebuilt_measure_reaches_a_strategy_through_with_norm_not_a_field_name():
    """The march swaps the per-iteration measure in through ``with_norm``, whatever the field is called.

    It used to write a ``residual_norm`` field directly, so a strategy satisfying the protocol in full
    but keeping its measure under another name failed inside ``eqx.tree_at`` with an error naming a
    lambda rather than the contract. Here the measure triples the Euclidean norm, and the cubic is not
    solved in one step, so the report reads three times the plain residual exactly when the swap
    reached the step that ran.
    """

    def tripled(residual):
        return 3.0 * jnp.linalg.norm(residual)

    residual = _Cubic(jnp.array([8.0]))
    result = newton_march(
        _NormUnderAnotherName(DampedNewtonStep()),
        residual,
        jnp.array([4.0]),
        max_steps=1,
        rtol=1e-12,
        atol=1e-12,
        norm_builder=lambda step, state: tripled,
    )
    plain = float(jnp.linalg.norm(residual(result.state)))
    assert plain > 1e-3  # not at the root, so the two measures are distinguishable
    assert result.reports[0].residual_norm == pytest.approx(3.0 * plain)


class _ForwardingControl:
    """Delegates to a :class:`DualTimeControl`, exposing exactly the hooks it is given.

    Records every hook the march calls on it, so a test can see which ones reached the control.
    """

    def __init__(self, inner, hooks):
        self.inner = inner
        self.called = []
        for name in hooks:
            setattr(self, name, self._recording(name))

    def _recording(self, name):
        def hook(*args):
            self.called.append(name)
            return getattr(self.inner, name)(*args)

        return hook

    def next_step(self, base_step, previous, state):
        return self.inner.next_step(base_step, previous, state)


_SHIFT_HOOKS = ("carry_beta", "resumed_at", "redamp", "rebase")


@pytest.mark.parametrize("hooks", [_SHIFT_HOOKS, ("carry_beta",)], ids=["all four", "one of four"])
def test_the_shift_hooks_reach_a_control_offering_all_four_and_none_offering_some(hooks):
    """The four shift hooks are one capability, so a control offering only some of them gets none.

    A control answering some boundaries and not others would carry a shift that is right across one
    kind of boundary and silently wrong across another. Here a beta escalation (1 -> 2 -> 4) is the
    boundary: a control with all four hooks is told and carries 4.0 out; one with only ``carry_beta`` is
    not a :class:`~aquaflux.solve.ShiftCarryingControl`, so it is not told and carries its own beta.
    """
    control = _ForwardingControl(DualTimeControl(beta_start=1.0, beta_min=0.01), hooks)
    result = newton_march(
        _AlphaFromBeta(relaxation_schedule=ConstantRelaxation(jnp.asarray(1.0))),
        _Cubic(jnp.zeros((1,))),
        jnp.ones((1,)),
        max_steps=1,
        rtol=1e-10,
        atol=1e-12,
        step_control=control,
        retry=RetryPolicy(on_alpha=0.5, beta_factor=2.0, max_retries=3),
    )
    beta = result.control_state.beta
    if len(hooks) == len(_SHIFT_HOOKS):
        assert control.called == ["carry_beta"]
        assert beta == 4.0
    else:
        assert control.called == []
        assert beta == 1.0


class _SlowMode(eqx.Module):
    """``R(phi) = lam * phi - b``: a linear residual whose first mode is slow against a unit shift."""

    lam: jnp.ndarray
    b: jnp.ndarray

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        return self.lam * phi - self.b


def test_a_released_shift_turns_a_linear_tail_into_newton_steps() -> None:
    """At a fixed shift a dual-time step contracts a mode by ``beta / (lam + beta)``; released, it is Newton.

    With ``lam = 1e-3`` against a floor of 5e-3 the slow mode keeps 5/6 of its error per step, so the
    floored march needs ~100 steps to fall 8 decades, and twenty steps leave it well short. Released
    after its first full step at the floor, the same march keeps 1e-6/1e-3 per step and converges in a
    handful. Both arms get the same twenty-step budget. Wrong answers caught: a release that never
    reaches the march (both arms then stall alike) and one that reaches it at the wrong shift.
    """
    residual = _SlowMode(jnp.array([1.0e-3, 1.0]), jnp.array([1.0, 1.0]))
    phi0 = jnp.zeros(2)
    base = DualTimeStep(
        _UnitShiftPolicy(), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=2.0)
    )
    common = dict(max_steps=20, rtol=0.0, atol=1.0e-8)
    floored = newton_march(
        base, residual, phi0, step_control=DualTimeControl(beta_start=5e-3, beta_min=5e-3), **common
    )
    released = newton_march(
        base,
        residual,
        phi0,
        step_control=DualTimeControl(beta_start=5e-3, beta_min=5e-3, release_floor=1e-6),
        **common,
    )
    assert not floored.converged
    assert released.converged and len(released.reports) <= 5
    assert released.reports[0].shift == 5e-3 and released.reports[1].shift == 1e-6


# Incremented on EVERY call of the residual below, traced or eager, so a test can tell a residual
# evaluated op by op (one call per evaluation) from one evaluated through a compiled program (calls
# only while it is traced).
_CALLS: list[int] = []


class _CountedCubic(eqx.Module):
    """``phi**3 - theta``, counting every call -- what the march's norms must not evaluate op by op."""

    theta: jnp.ndarray

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        _CALLS.append(1)
        return phi**3 - self.theta


class _ScaledNorm(eqx.Module):
    """The residual's 2-norm over a scale held as an array leaf, as a row-scaled measure holds its rows."""

    scale: jnp.ndarray

    def __call__(self, residual: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(residual) / self.scale


class _ModuleScaledByStep:
    """Measures for :func:`~aquaflux.solve.staged_march` built as modules from the step's ``scale``."""

    def row_scaled(self, step, state):
        del state
        return _ScaledNorm(step.scale)

    def block_scaled(self, state):
        raise TypeError("not used")


def test_the_march_reads_its_norms_through_compiled_programs_not_op_by_op() -> None:
    """A second march over the same objects evaluates the residual no more than its compiled step does.

    Outside its compiled step the march reads three norms of the residual: the starting one its damping
    ramps against, the starting one in the stopping measure, and the stopping measure at every step.
    Each was evaluated op by op, one Python call of the residual per evaluation -- on a coupled problem
    a dispatch per operation, and the first march's largest cost outside its linear solves. Compiled,
    a march repeated over the same objects (new measure values, same structure) calls it not at all.
    """
    step = _JudgedOnItsScale(
        jnp.asarray(10.0), jnp.linalg.norm, ConstantRelaxation(jnp.asarray(1.0))
    )
    residual = _CountedCubic(jnp.zeros((3,)))

    def march():
        return newton_march(
            step,
            residual,
            jnp.ones((3,)),
            max_steps=4,
            rtol=0.0,
            atol=0.0,
            norm_builder=lambda strategy, state: _ScaledNorm(strategy.scale),
            stopping_step=_judged_on(1.0),
        )

    first = march()
    assert len(first.reports) == 4  # the stopping norm was read at every one of them
    _CALLS.clear()
    second = march()
    assert [r.stopping_norm for r in second.reports] == [r.stopping_norm for r in first.reports]
    assert _CALLS == []


def test_the_staged_march_reads_its_norms_through_compiled_programs_not_op_by_op() -> None:
    """The driver's references and its final root test are compiled too.

    :func:`~aquaflux.solve.staged_march` takes the starting residual in the steering measure and in the
    stopping measure before it marches, and the residual at the state reached afterwards. Repeated over
    the same objects, the whole solve -- driver and march -- calls the residual not at all.
    """
    from aquaflux.solve import NO_REFRESH, Convergence, RowScaled, staged_march
    from aquaflux.solve.driver import FinishedSource

    residual = _CountedCubic(jnp.zeros((2,)))

    def solve():
        return staged_march(
            residual,
            jnp.ones((2,)),
            strategy=_JudgedOnItsScale(
                jnp.asarray(10.0), jnp.linalg.norm, ConstantRelaxation(jnp.asarray(1.0))
            ),
            source=FinishedSource(),
            refresh=NO_REFRESH,
            convergence=Convergence(measure=RowScaled(), rtol=0.0, atol=0.02),
            measures=_ModuleScaledByStep(),
            max_steps=6,
            stopping_step=_judged_on(1.0),
        )

    first = solve()
    _CALLS.clear()
    second = solve()
    assert jnp.array_equal(second.state, first.state)
    assert _CALLS == []
