"""When to redo a march step, and how — the observed march's retry policy.

A forward step goes bad in three ways, and on a stiff low-shift saddle all three have the same cheap
cure: **more damping**. A larger shift lifts a non-finite correction back into the finite regime, cuts
the linear solve's cycle count, and shortens the implicit step until it fits inside whatever bound was
clipping it. :class:`RetryPolicy` holds the thresholds that detect those three cases and the knobs that
govern the response, together with the decisions taken from them — so the reason a step is redone and
the numbers that decided it cannot drift apart.

The one case damping does **not** cure is a correction that is merely under-solved: an incomplete
factorization can return a non-finite correction where a tighter Krylov tolerance returns a finite one.
That is what :attr:`RetryPolicy.solver` is for, and it is deliberately the *fallback* — it runs only on
a step still diverged after escalation, because the tight solve is far more expensive than a doubling.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import lineax as lx

from .forward_step import ForwardStep, StepOutcome

#: Retry reasons whose response is to RAISE the pseudo-transient shift. The others redo the step at the
#: shift it already had: ``"cycles"`` because the cure for an expensive solve is a fresh preconditioner
#: rather than a stiffer operator, and ``"solver"`` because a tighter Krylov tolerance is the cure for an
#: under-solved correction. Kept here, in one place, because the march decides whether to escalate and
#: the log decides whether to print ``beta -> x`` or ``beta x unchanged``, and the two disagreeing would
#: put a shift in the log that the march never ran.
ESCALATING_REASONS = frozenset({"diverged", "alpha"})


@dataclasses.dataclass(frozen=True)
class RetryPolicy:
    """When an observed march redoes a step, and how.

    The default policy retries nothing and aborts nothing (every threshold ``None``, no tighter
    solver), which is byte-identical to a march that never retries. Set ``on_alpha`` to enable the
    shift escalation; set ``abort_above_cycles`` to bound what one step may spend; set ``solver`` to
    enable the tight-Krylov divergence fallback. All three are independent and compose.

    ⚠️ **A COST GUARD AND AN ESCALATION TRIGGER ARE DIFFERENT RESPONSES AND ARE DELIBERATELY NOT ONE
    NUMBER.** They were, once: a single ``on_cycles`` both stopped the inner loop and escalated the
    shift, and that conflation is a recorded defect rather than a tidiness complaint. A restart-cycle
    count is ``preconditioner strength × operator difficulty``, so any constant encodes an assumption
    about *which preconditioner is installed* — and the march already has the right response to a
    growing cycle count, which is to **refresh the frozen preconditioner** (a ``RefreshPolicy``
    trigger, fired mid-step). Escalating the shift on the same observation additionally assumed a
    bigger shift makes the block easier, which is not even true on every case: measured on a
    two-dimensional coupled saddle the flow block wants **140** Krylov applications at ``beta`` 0.5
    against **32** at 0.05, so "expensive, therefore stiffen" closed a loop that ran the wrong way and
    drove a working march to a non-finite residual in three steps. Stiffness is what ``on_alpha``
    measures, and a step length is dimensionless — it needs no per-arm calibration.

    A **value object**, not a bundle of loose numbers: the six settings are meaningless apart (a
    ``beta_factor`` with no threshold escalates nothing; a ``cycles_limit`` bounds a loop that never
    runs), they were previously threaded as six parallel arguments through two signatures, and the
    decisions taken from them — :meth:`escalation_reason`, :meth:`has_diverged` — read three or four of
    them each. Keeping the data and those decisions in one place is what stops a caller reproducing the
    predicate with one threshold left out.

    Attributes
    ----------
    solver : lineax.AbstractLinearSolver or None
        A **tighter** linear solver for redoing a step that is still diverged after any shift
        escalation. With an *inexact* preconditioner the loose default Krylov tolerance can leave the
        correction non-finite on the stiff operator an aggressive step produces, where the same solve
        taken tightly is finite. The step is redone from the same pre-step state, so only the Krylov
        tolerance changes -- the preconditioner is already matched to this state and shift. ``None``
        (default) never retries this way; an exact factorization never needs it.
    divergence_cap : float
        A step counts as diverged when its residual norm is non-finite **or**, if this cap is finite,
        exceeds ``divergence_cap * reference``. Defaults to ``inf``, so only non-finiteness counts. A
        tight cap is the wrong default because the residual legitimately *rises* while a flow develops
        under a pseudo-time march, so a finite cap false-fires on exactly the progress it should leave
        alone.
    abort_above_cycles : int or None
        Per-solve restart-cycle budget. Once a solve exceeds it the step stops taking further inner
        iterations and is judged on what it has: a **cost guard**, not a diagnosis. It does **not**
        escalate the shift and does not by itself discard the step — an aborted step that is finite and
        whose line search held is accepted, and the step control adapts to the rate it achieved.
        ``None`` (default) leaves the inner loop bounded only by its own iteration count.

        The abort is what stops a hopeless step spending its whole inner budget before ``on_alpha``
        catches it. ⚠️ But set it **above** what the installed preconditioner costs when it is
        *healthy*, or it truncates convergence a step is in the middle of achieving: on the case above,
        a threshold of 10 against an arm whose healthy cost was 12 cut the step off after three inner
        iterations, where a fourth would have accepted it.

        **Per solve, never summed.** A summed threshold grows with how many times the step solved, so
        the same per-solve difficulty trips it or not depending on an inner-iteration count that says
        nothing about conditioning -- roughly six times more sensitive for a five-iteration step than a
        one-iteration one.
    on_alpha : float or None
        Step-length threshold. A step whose line-search factor falls to this or below, without reaching
        its stopping criterion, is redone at a larger shift. ``None`` (default) disables it.

        It catches the failure a cycle count cannot see. A step length of essentially zero is as dead
        as an expensive one and is invisible to the cost trigger, because the solves are *cheap* -- the
        correction simply cannot be followed, either because it does not descend or because a
        positivity cap admits almost none of it. Measured on a three-dimensional coupled march: four
        consecutive steps each ran a full inner loop at 5-12 cycles, moved the residual not at all, and
        escaped only once the step control's own backoff had doubled the shift four times, one step per
        doubling. Escalating instead does those doublings as discarded attempts of a single step.

        Also pushed into an inner loop that can act on it (see :meth:`with_inner_abort`), so the
        attempt exits at the collapse rather than iterating on inside a loop that cannot move.
    beta_factor : float
        The factor the shift strength is multiplied by on each escalation (default ``2``).
    cycles_limit : int
        The most successive escalations one step may take (default ``2``).

    Notes
    -----
    Escalation needs a readable shift leaf on the forward step (a constant relaxation set by a step
    control). Without one it no-ops, and a diverged step falls through to :attr:`solver` as though no
    thresholds were set.
    """

    solver: lx.AbstractLinearSolver | None = None
    divergence_cap: float = float("inf")
    abort_above_cycles: int | None = None
    on_alpha: float | None = None
    beta_factor: float = 2.0
    cycles_limit: int = 2

    @property
    def escalates(self) -> bool:
        """Whether escalation can fire at all -- i.e. whether the step-length threshold is set.

        The cycle budget is deliberately absent: it bounds cost and never escalates (see the class
        docstring). A policy with only ``abort_above_cycles`` set still needs no readable shift.
        """
        return self.on_alpha is not None

    def require_shifted(self, forward_step: ForwardStep) -> None:
        """Reject a step this policy cannot escalate, at the seam rather than mid-march.

        Escalation raises the pseudo-transient shift, so it needs a step carrying a
        ``relaxation_schedule`` with a readable ``beta`` -- what
        :class:`~aquaflux.solve.ShiftedForwardStep` declares and :class:`~aquaflux.solve.ForwardStep`
        does not. The distinction was once enforced by ``hasattr`` deep in the march loop, which fails
        **silently**: a :class:`~aquaflux.solve.DampedNewtonStep` satisfies ``ForwardStep`` in full, so
        a march configured to escalate accepted one and then never escalated -- indistinguishable, from
        the log, from a march that never needed to.

        It belongs on the policy rather than on the march because it is *this policy's* requirement:
        the tight-solver fallback next door has no such need, and gating it too would reject steps that
        work perfectly well with it.

        ⚠️ **Apply it to the step that will actually run, after any ``StepControl`` has shaped it --
        never to the base step before the march.** A control is what *installs* the readable shift on
        the dual-time family: the builder hands over a schedule with no ``beta`` and the control swaps
        in one that has it, per iteration. Checked too early, this rejects the shipped coupled
        configuration outright.

        Raises
        ------
        TypeError
            If ``forward_step`` carries no ``relaxation_schedule`` with a readable ``beta``.
        """
        # Checked against the SHIFT specifically, not `isinstance(..., ShiftedForwardStep)`. The
        # argument is already typed `ForwardStep`, so re-testing those four methods at runtime would
        # reject a legitimate duck-typed step for a reason that has nothing to do with escalation.
        schedule = getattr(forward_step, "relaxation_schedule", None)
        if schedule is None or not hasattr(schedule, "beta"):
            raise TypeError(
                "the beta-escalation retry (RetryPolicy.on_alpha) drives the "
                "pseudo-transient shift strength, so it needs a forward step whose "
                "`relaxation_schedule` exposes a readable `beta` -- a ConstantRelaxation, which a "
                "StepControl swaps onto a PseudoTransientStep or a DualTimeStep each iteration. The "
                "default SwitchedEvolutionRelaxation those steps are built with exposes none, so "
                f"constructing one is not enough on its own. {type(forward_step).__name__} has no "
                "readable `beta`, so the retry would silently do nothing. Either run the step under "
                "a step control, or leave `on_alpha` unset."
            )

    def with_inner_abort(self, forward_step: ForwardStep) -> ForwardStep:
        """Give ``forward_step`` this policy's stopping thresholds, if it can act on them.

        A step that runs an inner loop can stop the moment it crosses one, rather than iterating on
        inside a loop that will not help: a solve costing more than :attr:`abort_above_cycles`, or a
        step length fallen to :attr:`on_alpha`. A step with no inner loop has nothing to stop and is
        returned unchanged -- as is any step when neither threshold is set, so the default path is
        byte-identical.

        ⚠️ **The two mean different things once the loop has stopped, and that asymmetry is the point
        of keeping them separate.** Crossing :attr:`on_alpha` is a diagnosis: the step is going nowhere
        and the march discards the attempt and escalates. Crossing :attr:`abort_above_cycles` is only a
        budget: the step keeps whatever it achieved and is judged on its merits, because a cycle count
        says how hard the *preconditioner* is finding this operator, not how stiff the step is.

        Apply once per march rather than per iteration: the thresholds are constant for the segment,
        and a step whose static fields are rewritten every iteration would be a fresh compilation key
        each time.

        Parameters
        ----------
        forward_step : ForwardStep
            The march's base step.

        Returns
        -------
        ForwardStep
            The step carrying the thresholds it can act on, or ``forward_step`` itself.
        """
        # `dataclasses.replace`, not `eqx.tree_at`: the thresholds are STATIC fields, so they live in
        # the treedef rather than among the leaves and `tree_at` (which addresses leaves) cannot reach
        # them.
        fields = {}
        if self.abort_above_cycles is not None and hasattr(
            forward_step, "abort_above_inner_cycles"
        ):
            fields["abort_above_inner_cycles"] = self.abort_above_cycles
        if self.on_alpha is not None and hasattr(forward_step, "abort_below_alpha"):
            fields["abort_below_alpha"] = self.on_alpha
        return dataclasses.replace(forward_step, **fields) if fields else forward_step

    def has_diverged(self, residual_norm: jnp.ndarray, reference: float) -> bool:
        """Whether a step's residual norm signals a diverged step this policy should redo.

        True if the norm is non-finite, or -- when a finite :attr:`divergence_cap` is set -- if it
        exceeds ``divergence_cap * reference``. A non-finite norm is the load-bearing case (an inexact
        preconditioner returning a poisoned correction on a stiff operator); the cap is an optional
        extra for a finite blow-up.

        Parameters
        ----------
        residual_norm : jnp.ndarray
            The residual measure at the state the attempt produced, a scalar.
        reference : float
            The march's global reference norm, which the cap is a multiple of.

        Returns
        -------
        bool
            Whether the step counts as diverged.
        """
        if not bool(jnp.isfinite(residual_norm)):
            return True
        return (
            self.divergence_cap < float("inf")
            and reference > 0.0
            and (float(residual_norm) > self.divergence_cap * reference)
        )

    def retry_reason(
        self, outcome: StepOutcome, residual_norm: jnp.ndarray, reference: float
    ) -> str | None:
        """Why this step should be redone, or ``None`` to accept it as taken.

        The three ways a step goes bad, and **they do not share one response**: a **diverged**
        correction (non-finite, or past :attr:`divergence_cap`), a step length **collapsed** by the
        descent test, and a **truncated** solve that hit :attr:`abort_above_cycles` without reaching
        its target.

        The first two are stiffness, and more damping is the cure -- a larger shift lifts the
        correction out of the non-finite regime and shortens the implicit step until it fits inside
        whatever bound was clipping it. They are in :data:`ESCALATING_REASONS`.

        The third is **not** stiffness, and this is the distinction the class docstring is about. A
        solve that ran long says the frozen preconditioner is struggling with this operator, and the
        cure is a **fresh preconditioner**, which the dual-time loop's mid-step refresh has already
        built by the time the step returns. So ``"cycles"`` redoes the step at the shift it already
        had, on the factorization that refresh produced. Raising the shift instead assumes a stiffer
        operator is an easier one, which is not true on every case, and closed a divergent loop on the
        one where it is false.

        Cost and step length are only reasons when the step **missed its own stopping criterion**.
        Redoing a step that met it discards a good iterate and replaces it with a shorter one, whatever
        it cost and however hard the ladder had to work to get there.

        The reason is returned as a short string, which is also what the march reports through its
        retry seam -- so the decision and its explanation cannot disagree.

        Parameters
        ----------
        outcome : StepOutcome
            The attempt's record; reads ``max_inner_cycles``, ``reached_target`` and ``alpha``.
        residual_norm : jnp.ndarray
            The residual measure at the state the attempt produced, a scalar.
        reference : float
            The march's global reference norm, for the divergence cap.

        Returns
        -------
        str or None
            ``"diverged"``, ``"alpha"``, or ``None`` to keep the step. There is deliberately no
            ``"cycles"`` reason: a cycle count is answered by refreshing the preconditioner and by
            :attr:`abort_above_cycles`, never by stiffening the shift (see the class docstring).
            ``on_alpha`` unset disables escalation entirely, leaving a diverged step to the
            tight-Krylov retry.
        """
        if not self.escalates and self.abort_above_cycles is None:
            return None
        if self.escalates and self.has_diverged(residual_norm, reference):
            return "diverged"
        if bool(outcome.reached_target):
            return None
        # ...and step length is a reason WHATEVER collapsed it, including an injected constraint. That
        # is not an oversight: gating it on `binding_limit == 1` was tried and is a regression.
        # MEASURED -- the one escalation of an entire coupled RANS march fired at a step whose cap was
        # 4.37e-10, and suppressing it cost 8 steps and 199 s end to end. (The mechanism usually offered
        # for the gate, that more damping tightens such a cap, is not measurable from a march LOG --
        # only the accepted attempt's cap is recorded there. It IS measurable if the limiter's inputs
        # are dumped per call, and when that was done the answer was that `d(cap)/d(shift)` has NO FIXED
        # SIGN at a fixed state: doubling the shift narrowed the cap ~9x on one state and widened it
        # ~3x on another, and on a third the binding cell changed between attempts, so consecutive caps
        # did not measure the same quantity. Rest the decision on the A/B above, not on a mechanism --
        # in either direction.)
        #
        # What damping cannot do is un-pin a cell already ON the boundary, WITH AN UNFLOORED LIMITER:
        # the room is `phi_i / |delta_i|`, so once the constrained entry is driven to (almost) zero the
        # cap is small however small the correction gets, and the fraction-to-the-boundary rule then
        # shrinks it by a fixed factor per step. Note this is exactly the unfloored case -- give the
        # limiter a `floor` and the room becomes `(phi_i + floor)/|delta_i|`, which a smaller correction
        # DOES widen. It does not rescue the cell, though: the floored rule leaves `(phi_i + floor)`
        # decaying by that same fixed factor per clipped step, so the collapse restarts a fixed number
        # of decades later. That failure is not this predicate's to catch -- the march's stall guard
        # ends the segment instead.
        if self.escalates and self.on_alpha is not None and float(outcome.alpha) <= self.on_alpha:
            return "alpha"
        # Last, and deliberately last: a step that merely cost too much is the weakest of the three
        # signals, and the other two are about the iterate rather than about its price. Redone at the
        # SAME shift (see this method's docstring and `ESCALATING_REASONS`).
        if (
            self.abort_above_cycles is not None
            and int(outcome.max_inner_cycles) > self.abort_above_cycles
        ):
            return "cycles"
        return None

    def escalate(self, beta: jnp.ndarray) -> jnp.ndarray:
        """The escalated shift strength: ``beta`` scaled by :attr:`beta_factor`.

        **Scales the existing leaf rather than rebuilding one**, and that is the whole reason this is a
        method. Writing ``jnp.asarray(float(beta) * factor)`` instead yields a fresh weak-typed float64
        array whose abstract value (dtype and weak type) need not match the shift leaf the step already
        carries -- and any mismatch makes the escalated step a compilation-cache **miss**, recompiling
        the whole solve on every retry. Multiplying the leaf preserves its dtype and weak type exactly,
        so the escalated step is a cache hit for whatever the step control set.

        Parameters
        ----------
        beta : jnp.ndarray
            The shift-strength leaf the step currently carries, a scalar.

        Returns
        -------
        jnp.ndarray
            The escalated leaf, with ``beta``'s dtype and weak type preserved.
        """
        return beta * self.beta_factor


#: The policy that retries nothing -- the default for every march, and byte-identical to a march with
#: no retry machinery at all. A module-level singleton rather than a fresh ``RetryPolicy()`` per
#: signature default: the class is immutable, so one shared instance is safe, and naming it says what
#: the default *means* where a constructor call would only say how it is spelled.
NO_RETRIES = RetryPolicy()
