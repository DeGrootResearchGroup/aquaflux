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
from aquaflux.solve import (
    CycleGrowthTrigger,
    DampedNewtonStep,
    ImplicitNewtonSolver,
    StepReport,
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
    return StepReport(step=step, cycles=cycles, residual_norm=ratio, residual_ratio=ratio)


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
