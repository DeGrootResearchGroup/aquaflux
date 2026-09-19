"""The staged solve driver: march a residual in segments, re-freezing the step between them.

:func:`~aquaflux.solve.newton_march` steps one segment. A solve whose preconditioner goes stale as the
flow develops runs a *sequence* of segments -- each stops when a refresh trigger fires, the step is
re-frozen at the state reached, and the next segment continues -- and none of the rules that sequence
obeys mention what residual is being driven:

* the stopping target ``atol + rtol * reference`` is measured **once**, at the initial state, and held
  across every segment, so a refreshed solve stops exactly where an unrefreshed one does;
* the residual measure is built once by a builder and handed to every segment, so steering and judging
  are one definition, and a measure that holds the initial state's scales keeps holding them;
* the last segment has no refresh left to spend, so it marches without the trigger -- with no second
  solve after the march, a segment stopped where its trigger fires would end the solve short of a root;
* a stateful step control is threaded through the segments, while the damping reference and the drift
  measure restart with each;
* the state the last segment reaches is judged in the measure it was steered by, and refused if it is
  not a root, because the implicit-function-theorem adjoint is valid only at one.

:func:`staged_march` is those rules, taking everything residual-specific as an argument: the residual,
the measure source, the drift measure, and a :class:`ContinuationSource` that builds and re-freezes the
Newton step. The coupled RANS solve and the laminar flow solve both run on it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax.numpy as jnp

from .convergence import Convergence, ResidualMeasures
from .march import ResidualHomotopy, newton_march
from .materialized_session import PreconditionerSession
from .refresh import RefreshPolicy
from .retry import NO_RETRIES, RetryPolicy
from .step_control import default_dual_time_control
from .strategy import NewtonStrategy, StepControl, StepReport

__all__ = [
    "CallerBuiltSource",
    "ContinuationSource",
    "FinishedSource",
    "SessionSource",
    "StagedResult",
    "explicit_source",
    "refuse_unforwardable_settings",
    "staged_march",
]


class ContinuationSource(Protocol):
    """Where a staged solve's :class:`~aquaflux.solve.NewtonStrategy` comes from, and how it re-freezes.

    A solve needs a strategy twice: once at the start, and again at each refresh, from a developed
    state. Those are one decision -- *which* strategy this solve runs -- so they live behind one
    interface rather than two independent branches that drift.

    Attributes
    ----------
    refresh_preconditioner : callable or None
        The per-step refresh hook this source brings with it, or ``None``.
    """

    refresh_preconditioner: Callable[[NewtonStrategy, jnp.ndarray], None] | None

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        """The strategy to start the march with, frozen at ``state``."""
        ...

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        """Re-freeze at the developed ``state``; ``previous`` is the step being replaced."""
        ...


@dataclasses.dataclass(frozen=True)
class CallerBuiltSource:
    """A strategy the caller builds from the state, and rebuilds the same way at every refresh.

    The builder owns the whole configuration, so a setting that could only reach a strategy this driver
    builds has nowhere to go and must be refused by whoever accepts it.
    """

    builder: Callable[[jnp.ndarray], NewtonStrategy]
    #: A caller-built strategy brings its own refresh hook, if any, on its ``RefreshPolicy``.
    refresh_preconditioner = None

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        return self.builder(state)

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        del previous  # the builder re-derives everything from the state
        return self.builder(state)


@dataclasses.dataclass(frozen=True)
class SessionSource:
    """A strategy a preconditioner session builds and re-freezes -- the source that has configuration.

    The preconditioner chose the session, and the march keywords are handed to every build and refresh
    the session makes. The session's own per-step refresh hook, if it has one, rides with it.

    Attributes
    ----------
    session : PreconditionerSession
        Builds each step and keeps its preconditioner current.
    reference_state : jnp.ndarray or None
        The state the first build is frozen at, or ``None`` for the state the march starts from.
    march : dict
        The march keywords handed to every build and refresh.
    """

    session: PreconditionerSession
    reference_state: jnp.ndarray | None
    march: dict

    @property
    def refresh_preconditioner(self) -> Callable[[NewtonStrategy, jnp.ndarray], None] | None:
        return self.session.refresh_preconditioner

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        reference = state if self.reference_state is None else self.reference_state
        return self.session.build(reference, **self.march)

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        return self.session.refresh(state, previous, **self.march)


@dataclasses.dataclass(frozen=True)
class FinishedSource:
    """The caller handed over a finished step and no builder, so nothing here can re-freeze it.

    :meth:`~aquaflux.solve.RefreshPolicy.require_rebuildable` already refuses that combination when a
    refresh would run, so both methods are unreachable through the driver. The source exists so it is
    never ``None``, and so that if one is ever reached it says what is missing.
    """

    refresh_preconditioner = None

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        raise TypeError(
            "no strategy to build: a finished `strategy` was supplied. This is a driver bug -- the "
            "supplied step should have been used directly."
        )

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        raise TypeError(
            "a refresh triggered but the explicit `strategy` cannot be rebuilt: pass "
            "`RefreshPolicy(builder=...)` so the solve can re-freeze it at each developed state."
        )


def explicit_source(
    strategy: NewtonStrategy | None,
    refresh: RefreshPolicy,
    given: dict[str, object],
    *,
    caller: str,
) -> ContinuationSource | None:
    """The source for a solve whose caller supplied the strategy or the builder, or ``None``.

    ``given`` are the settings that configure a strategy the solve builds itself. Where the caller
    supplied a finished ``strategy`` or a ``RefreshPolicy(builder=...)``, that object owns its whole
    configuration and there is nothing for them to reach, so they are **refused rather than dropped**:
    ``**kwargs`` accepts every keyword and checks none, and a solve asked for a dual-time loop that ran
    the library defaults would look exactly like the one configured.

    Parameters
    ----------
    strategy : NewtonStrategy or None
        A finished step, or ``None``.
    refresh : RefreshPolicy
        Its ``builder``, if any, rebuilds the strategy at each developed state.
    given : dict
        The strategy-configuring settings the caller passed, by name.
    caller : str
        The public entry point's name, for the error.

    Returns
    -------
    ContinuationSource or None
        :class:`FinishedSource` or :class:`CallerBuiltSource`; ``None`` when the solve builds the strategy
        itself and the settings apply.

    Raises
    ------
    TypeError
        If a setting is given beside a strategy or builder that cannot receive it.
    """
    if strategy is not None:
        refuse_unforwardable_settings(
            given, "`strategy`", "the step you passed already carries them", caller
        )
        return FinishedSource() if refresh.builder is None else CallerBuiltSource(refresh.builder)
    if refresh.builder is not None:
        refuse_unforwardable_settings(
            given, "`RefreshPolicy(builder=...)`", "the builder owns its own configuration", caller
        )
        return CallerBuiltSource(refresh.builder)
    return None


def refuse_unforwardable_settings(
    given: dict[str, object], owner: str, why: str, caller: str
) -> None:
    """Raise if any continuation setting was passed to a solve that cannot forward it."""
    if not given:
        return
    raise TypeError(
        f"{sorted(given)} configure the continuation `{caller}` builds, and {owner} was given, so "
        f"{why}. These would have been dropped silently. Pass them where the continuation is built "
        f"instead, or drop {owner}."
    )


@dataclasses.dataclass(frozen=True)
class StagedResult:
    """What a staged march ends with.

    Attributes
    ----------
    state : jnp.ndarray
        The converged state (the driver raises rather than return one that is not a root).
    strategy : NewtonStrategy
        The step of the last segment, whose ``adjoint_preconditioner`` the adjoint reuses.
    """

    state: jnp.ndarray
    strategy: NewtonStrategy


def staged_march(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    state: jnp.ndarray,
    *,
    strategy: NewtonStrategy | None,
    source: ContinuationSource,
    refresh: RefreshPolicy,
    convergence: Convergence,
    measures: ResidualMeasures,
    drift_measure: Callable[[jnp.ndarray], Callable | None] | None = None,
    max_steps: int,
    step_control: StepControl | None = None,
    on_step: Callable[[StepReport], None] | None = None,
    on_checkpoint: Callable[[StepReport, jnp.ndarray], None] | None = None,
    retry: RetryPolicy = NO_RETRIES,
    on_retry: Callable[[str, int, float], None] | None = None,
    homotopy: ResidualHomotopy | None = None,
    station_step: Callable[[NewtonStrategy, int, bool], NewtonStrategy] | None = None,
    caller: str = "staged_march",
) -> StagedResult:
    """March ``residual_fn`` to convergence in refresh segments, and refuse a state that is not a root.

    Parameters
    ----------
    residual_fn : callable
        The single-argument residual. Pass a bound method of a module (a pytree) rather than a lambda,
        so its arrays ride as dynamic leaves and every step within a segment is a compilation-cache hit.
    state : jnp.ndarray
        The state to march from, in the solved variables.
    strategy : NewtonStrategy or None
        A finished strategy, or ``None`` to build one from ``source`` at ``state``.
    source : ContinuationSource
        Builds and re-freezes the strategy.
    refresh : RefreshPolicy
        The between-segment trigger, its budget, and the per-step refresh hook. ``max_steps`` applies to
        **each** segment, so a refreshed solve may take ``refresh.segments * max_steps`` steps.
    convergence : Convergence
        The stopping test with **every field set** (the caller fills unset fields from its own base).
    measures : ResidualMeasures
        What the residual can be measured in. The measure ``convergence`` names is built once from it,
        for the initial state, and handed to every outer iteration of every segment.
    drift_measure : callable or None
        ``state -> drift(state)``, re-based at each segment's starting state, since a refresh re-freezes
        there and a measure carried across would keep reporting drift the refresh had absorbed. ``None``
        for a residual with no coefficient drift to watch (a refresh trigger then needs to read costs).
    max_steps, step_control, on_step, on_checkpoint, retry, on_retry, homotopy, station_step
        Forwarded to :func:`~aquaflux.solve.newton_march` on every segment.
    caller : str
        The public entry point's name, for the error a non-converged march raises.

    Returns
    -------
    StagedResult
        The converged state and the last segment's strategy.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the march ends short of its target: it exhausted ``max_steps`` in its last segment, stopped on
        a stalled positivity cap, went non-finite, or a homotopy never reached its target.
    """
    # A refresh rebuilds the step; a caller-supplied step with no builder leaves it nothing to rebuild
    # WITH, so the refresh would silently never happen. The policy owns that check.
    refresh.require_rebuildable(strategy)
    # A per-step re-fit hook: a materialized session re-fits its inverse before every step; a
    # caller-built step brings its own on the policy.
    refresh_preconditioner = refresh.refresh_preconditioner or source.refresh_preconditioner
    if strategy is None:
        strategy = source.build(state)
    # A dual-time march with no caller control defaults to the Courant ramp, carried across the refreshes
    # below.
    step_control = default_dual_time_control(step_control, strategy)

    # The measure is built once per outer iteration by the march, from the one builder made here, and the
    # global progress reference is taken in it at the initial state. A `BlockScaled` builder holds the
    # initial state's scales for the whole solve, so a refresh cannot re-base it (which would put the
    # target, measured once here, out of reach); a `RowScaled` one re-reads the step it is handed.
    norm_builder = convergence.measure._builder(measures, state)
    reference_norm = float(norm_builder(strategy, state)(residual_fn(state)))
    # `refresh.limit` refreshes means `refresh.segments` segments: the segment *after* the last refresh
    # must still be marched, or the newly-refreshed preconditioner would never be used.
    control_state: object = None
    for segment in range(refresh.segments):
        result = newton_march(
            strategy,
            residual_fn,
            state,
            max_steps=max_steps,
            rtol=convergence.rtol,
            atol=convergence.atol,
            reference_norm=reference_norm,
            # The last segment has no refresh left to spend, so it marches to convergence or to
            # `max_steps` rather than stopping where the trigger fires.
            trigger=None if refresh.is_last_segment(segment) else refresh.trigger,
            step_control=step_control,
            # Threaded across segments so a stateful control (the alpha-targeting shift climb)
            # continues past each refresh rather than restarting -- the same global-lifetime carry as
            # `reference_norm`, unlike the per-segment damping reference and drift measure.
            control_state=control_state,
            observer=on_step,
            checkpoint=on_checkpoint,
            drift_measure=None if drift_measure is None else drift_measure(state),
            norm_builder=norm_builder,
            refresh_preconditioner=refresh_preconditioner,
            retry=retry,
            on_retry=on_retry,
            homotopy=homotopy,
            station_step=station_step,
        )
        state = result.state
        control_state = result.control_state
        if not result.triggered or refresh.is_last_segment(segment):
            break
        # Re-freeze at the developed state -- how is the source's, the same choice made at the initial
        # build above rather than a second one made here.
        strategy = source.refresh(state, strategy)

    # Judge convergence in the measure the march steered by, built at the state reached.
    residual_norm = float(norm_builder(strategy, state)(residual_fn(state)))
    target = convergence.atol + convergence.rtol * reference_norm
    arrived = homotopy is None or result.converged
    if not (math.isfinite(residual_norm) and residual_norm <= target and arrived):
        raise eqx.EquinoxRuntimeError(
            f"{caller} did not converge: the march ended at residual {residual_norm:.3e} against "
            f"a target of {target:.3e} (atol + rtol*||R0||)"
            + ("" if arrived else ", and its homotopy never reached the target problem")
            + ". The implicit-function-theorem adjoint is only valid at a converged root, so the fields "
            "and any gradient built on them would be silently wrong. Raise max_steps, loosen the "
            "tolerances, or strengthen the globalization (a dual-time loop, a retry policy)."
        )
    return StagedResult(state, strategy)
