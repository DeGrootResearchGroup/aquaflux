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

from .implicit import ForwardStep, StepOutcome


@dataclasses.dataclass(frozen=True)
class RetryPolicy:
    """When an observed march redoes a step, and how.

    The default policy retries nothing (both thresholds ``None``, no tighter solver), which is
    byte-identical to a march that never retries. Set ``on_cycles`` and/or ``on_alpha`` to enable the
    shift escalation; set ``solver`` to enable the tight-Krylov divergence fallback. The two are
    independent and compose — with both set, escalation is tried first and the tighter solve catches
    what it could not fix.

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
    on_cycles : int or None
        Per-solve restart-cycle threshold. A step whose most expensive single solve exceeds it, without
        having reached its own stopping criterion, is redone at a larger shift. ``None`` (default)
        disables the cost reason.

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
    on_cycles: int | None = None
    on_alpha: float | None = None
    beta_factor: float = 2.0
    cycles_limit: int = 2

    @property
    def escalates(self) -> bool:
        """Whether either escalation threshold is set, i.e. whether escalation can fire at all."""
        return self.on_cycles is not None or self.on_alpha is not None

    def with_inner_abort(self, forward_step: ForwardStep) -> ForwardStep:
        """Give ``forward_step`` this policy's discard thresholds, if it can act on them.

        A step that runs an inner loop can stop the moment it crosses one of them, because crossing
        either with the target unmet is exactly what makes the march discard the attempt: a solve
        costing more than :attr:`on_cycles`, or a step length fallen to :attr:`on_alpha`. Pushing them
        down is what keeps each threshold **one** number rather than two that must be kept in step. A
        step with no inner loop has nothing to stop and is returned unchanged -- as is any step when
        neither threshold is set, so the default path is byte-identical.

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
        if self.on_cycles is not None and hasattr(forward_step, "abort_above_inner_cycles"):
            fields["abort_above_inner_cycles"] = self.on_cycles
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

    def escalation_reason(
        self, outcome: StepOutcome, residual_norm: jnp.ndarray, reference: float
    ) -> str | None:
        """Why this step should be redone at a larger shift, or ``None`` to accept it as taken.

        The three ways a step goes bad, and the reason they share one response: a **diverged**
        correction (non-finite, or past :attr:`divergence_cap`), a **costly** solve that did not reach
        its target, and a step length **collapsed** by the descent test. All three are cured by more
        damping -- a larger shift lifts the correction out of the non-finite regime, cuts the cycle
        count, and shortens the implicit step until it fits inside whatever bound was clipping it.

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
            ``"diverged"``, ``"cycles"``, ``"alpha"``, or ``None`` to keep the step. Both thresholds
            ``None`` disables escalation entirely, leaving a diverged step to the tight-Krylov retry.
        """
        if not self.escalates:
            return None
        if self.has_diverged(residual_norm, reference):
            return "diverged"
        if bool(outcome.reached_target):
            return None
        if self.on_cycles is not None and int(outcome.max_inner_cycles) > self.on_cycles:
            return "cycles"
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
        if self.on_alpha is not None and float(outcome.alpha) <= self.on_alpha:
            return "alpha"
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
