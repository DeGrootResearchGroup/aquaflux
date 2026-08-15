"""How a frozen preconditioner is kept current during a solve — the refresh policy.

A preconditioner frozen at one state goes stale as the flow develops, and there are two quite
different remedies for it. **Between segments**, a driver can stop the march, rebuild the
preconditioner at the state reached, and continue — expensive, off the jit path, and worth it only
when the operator has genuinely moved. **Within a step**, a host factorization can be re-derived in
place at the current state and shift, cheap enough to do every step and the only thing that keeps an
*exact* factorization exact as the shift ramps.

:class:`RefreshPolicy` carries both, because they are the same concern — keeping the frozen operator
close to the one being solved — and because a driver choosing one is choosing against the other.

**Forward-only.** Every mechanism here re-derives the preconditioner from a *mid-march* state, which
is a tracer under differentiation; a refreshed preconditioner would capture it and escape the
converged solve's reverse rule as a leaked tracer. A driver must reject a policy that would refresh
under a JAX transform rather than let the leak surface later as an opaque error. Nothing is lost: the
adjoint is refresh-independent, since the preconditioner only accelerates the Krylov iteration and
both marches reach the same converged state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from .forward_step import ForwardStep
from .march import RefreshTrigger


@dataclasses.dataclass(frozen=True)
class RefreshPolicy:
    """When and how a solve re-derives its frozen preconditioner.

    The default policy refreshes nothing, which is byte-identical to a single-stage solve. A value
    object rather than four loose arguments: they are meaningless apart -- a ``limit`` with no
    ``trigger`` bounds a loop that never runs, and a ``builder`` with no ``trigger`` is only ever
    called once, for the initial build -- and the decisions taken from them read two or three each.

    Attributes
    ----------
    trigger : RefreshTrigger or None
        Judges, from the march's own per-step reports, when the frozen preconditioner has gone stale
        enough to be worth rebuilding. ``None`` (default) never refreshes between segments.

        Prefer a trigger that watches how far the operator's own coefficients have moved since they
        were frozen: that movement **is** the staleness. Inferring it from the linear solve's cost
        instead also works, but the cost rises as the damping schedule ramps down as well as from
        staleness -- on a separating flow, by more -- so such a trigger needs a residual gate that a
        coefficient-drift one does not.
    limit : int
        The most refreshes one solve may perform (default ``1``). Each costs a preconditioner rebuild
        and a recompilation of the shifted solve, so this bounds that expense independently of how
        eager the trigger is. ``0`` disables refreshing entirely, whatever the trigger says.
    builder : callable or None
        ``state -> ForwardStep``, rebuilding the whole forward step at a given state. Supplying it
        replaces the driver's own rebuild on **both** the initial build and every refresh, which is
        what lets a preconditioner the driver does not know how to construct -- a host factorization
        materialized off the jit path -- refresh at all. It is also what lifts the restriction that a
        caller-supplied step cannot be refreshed: the builder is *how* the refresh rebuilds.
    precondition_step : callable or None
        ``(active_step, state) -> None``, called before **each** observed step to re-derive that
        step's frozen host preconditioner from the state and shift strength the step is about to run
        at. It mutates in place, so the compiled step stays a cache hit.

        This is the per-step counterpart of the between-segment rebuild above, and the two suit
        opposite kinds of preconditioner. An *exact* factorization is exact only for the operator it
        factored, so once the shift ramps away from that value it mis-preconditions and must be
        re-derived every step -- cheap enough to do, because it is exact. An *approximate* one
        tolerates the mismatch at the cost of a few extra iterations, so it is better refreshed
        occasionally on evidence than unconditionally.

    Notes
    -----
    **A driver unpacks this; it is not passed on wholesale.** The eager march needs only
    :attr:`trigger` and :attr:`precondition_step` -- :attr:`limit` and :attr:`builder` govern the
    *sequence* of segments, which is the driver's loop and not the march's business. Handing the
    march the whole object would give it two fields it cannot use and could not act on correctly.
    """

    trigger: RefreshTrigger | None = None
    limit: int = 1
    builder: Callable[[Any], ForwardStep] | None = None
    precondition_step: Callable[[ForwardStep, Any], None] | None = None

    @property
    def refreshes(self) -> bool:
        """Whether a between-segment refresh can actually happen: a trigger, and a budget to spend."""
        return self.trigger is not None and self.limit > 0

    @property
    def observes(self) -> bool:
        """Whether this policy requires the driver to run the observed (eager) march at all.

        True when anything here needs per-step evidence or a per-step hook. A policy carrying only a
        ``builder`` does **not** make a march observed: without a trigger the builder is called once,
        for the initial build, which a single-stage solve does just as well.
        """
        return self.refreshes or self.precondition_step is not None

    @property
    def segments(self) -> int:
        """How many march segments the driver runs: one more than the refresh budget.

        ``limit`` refreshes means ``limit + 1`` segments -- the segment *after* the last refresh must
        still be marched, or the newly-refreshed preconditioner would only ever be used by the
        finishing solve and its steps would go unobserved.
        """
        return self.limit + 1

    def is_last_segment(self, segment: int) -> bool:
        """Whether ``segment`` (0-based) is the final one, so a fired trigger must not refresh again."""
        return segment >= self.limit

    def require_rebuildable(self, continuation: ForwardStep | None) -> None:
        """Raise if this policy would refresh a step it has no way to rebuild.

        A between-segment refresh works by *reconstructing* the forward step at the developed state.
        When the caller supplies its own step and no :attr:`builder`, there is nothing to reconstruct
        it with -- so the refresh would silently never happen, and a solve asking for one would
        quietly not get it. Fail with the three ways out named instead.

        Parameters
        ----------
        continuation : ForwardStep or None
            The caller-supplied forward step, or ``None`` to let the driver build its own.

        Raises
        ------
        ValueError
            If a refresh is configured, a step was supplied, and no builder was.
        """
        if continuation is not None and self.refreshes and self.builder is None:
            raise ValueError(
                "a refresh trigger needs the solve to (re)build the forward step, but an explicit "
                "`continuation` was supplied with no `RefreshPolicy(builder=...)`. Pass a builder "
                "(state -> ForwardStep) that rebuilds it at each developed state, drop the explicit "
                "`continuation` so the solve builds it, or stage the refresh yourself."
            )


#: The policy that refreshes nothing -- the default, and byte-identical to a single-stage solve. A
#: module-level singleton rather than a fresh ``RefreshPolicy()`` per signature default: the class is
#: immutable, so one shared instance is safe, and naming it says what the default *means*.
NO_REFRESH = RefreshPolicy()
