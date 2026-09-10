"""Reynolds-number continuation: reach a high-Reynolds coupled root through easier lower-Re ones.

The coupled k-omega SST steady solve is slow (or fails) to reach its root from a cold start at a high
Reynolds number, because the convective nonlinearity is strong. Raising the molecular viscosity (a
lower Reynolds number) weakens that nonlinearity and makes the solve far easier -- measured, a cold
step that diverges at the target converges comfortably an order of magnitude lower in ``Re``. This
module walks a **homotopy in Reynolds number**: solve a sequence of lower-Re problems from an easy
anchor up to the true target, each seeded by the previous converged solution. The continuation
**dissolves at the target** -- the final solve is the true physical problem at the case's own
viscosity -- so it changes only the *path* to the root, never the root itself or its exact adjoint.

The user surface is a single integer, ``n_points``: the number of lower-Reynolds solves to run before
the target. Everything else is automatic -- the anchor Reynolds number, the intermediate values, the
initial condition for each (the lowest self-starts from the hybrid initialization; each converged
solution seeds the next higher Re), and the final target solve.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import equinox as eqx
import jax

from .coupled import solve_coupled
from .initialization import hybrid_initialize

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

    from .coupled import CoupledRANS


@dataclass(frozen=True)
class ReynoldsPoint:
    """Which continuation point a ``point_setup`` is being asked to configure.

    A per-point builder needs to know *where in the ramp* it is -- to label a log, to pick a tolerance,
    to vary a preconditioner between the seed rungs and the target. That is the continuation's own
    bookkeeping, so it is passed rather than re-derived: a caller counting its own invocations would be
    duplicating the loop's index, and could not know the total or the viscosity scaling at all.

    Attributes
    ----------
    index : int
        Which point this is, **1-based** (``1`` is the lowest-Reynolds anchor).
    total : int or None
        How many points the ramp has, including the target, or ``None`` when that is not knowable in
        advance. A schedule that may retreat onto a gentler step after a rung fails does not know how
        many rungs it will take until it has taken them, so it reports ``None`` and the label renders
        the index alone.
    viscosity_scale : float
        The factor the molecular viscosity is multiplied by at this point (``1.0`` at the target, larger
        below it). The Reynolds number is reduced by the same factor.
    """

    index: int
    total: int | None
    viscosity_scale: float

    @property
    def is_target(self) -> bool:
        """Whether this is the final, true-viscosity point (the one whose root is returned)."""
        return self.viscosity_scale == 1.0

    @property
    def label(self) -> str:
        """A short human label, e.g. ``"point 2/3 (Re/10)"`` -- so every driver need not format one.

        Renders the index alone (``"point 2 (Re/10)"``) when the ramp's length is not known ahead of
        time, which is the case for any schedule that may retreat onto a gentler step.
        """
        scaling = "target Re" if self.is_target else f"Re/{self.viscosity_scale:g}"
        position = f"{self.index}" if self.total is None else f"{self.index}/{self.total}"
        return f"point {position} ({scaling})"


class ReynoldsSchedule(Protocol):
    """The Reynolds numbers a continuation visits, as molecular-viscosity scale factors.

    A schedule decides where the ramp starts and, one rung at a time, where it goes next. Each factor
    ``> 1`` is a lower-Reynolds companion problem (``Re`` reduced by that factor); ``1.0`` is the true
    target and ends the ramp.

    **It is asked for the next scale one rung at a time rather than for the whole ladder up front**,
    which is what lets a schedule react to a rung that failed. A schedule that does not care can ignore
    the history and emit a fixed ladder (:class:`GeometricReynoldsSchedule`); one that does can shorten
    its step and try again (:class:`AdaptiveReynoldsSchedule`). Every method is a pure function of the
    arguments it is given -- no mesh, no state, no field -- so a schedule is trivially unit-testable and
    the loop that drives it owns all the bookkeeping.
    """

    def anchor(self, n_points: int) -> float:
        """The viscosity scale of the first, lowest-Reynolds rung.

        Parameters
        ----------
        n_points : int
            The user's one number: how deep below the target the ramp is anchored (``>= 0``). ``0``
            anchors at the target itself, i.e. no continuation.
        """
        ...

    def next_scale(self, converged: tuple[float, ...], failed: float | None) -> float | None:
        """The next viscosity scale to attempt, ``1.0`` for the target, or ``None`` to give up.

        Parameters
        ----------
        converged : tuple of float
            The scales of the rungs that have converged so far, in the order they were solved; empty
            before the first has. ``converged[-1]`` is the scale the current seed is a root of, and
            ``converged[-2] / converged[-1]`` is the step the ramp last took successfully.
        failed : float or None
            The scale that has just **failed** to converge, or ``None`` when the previous attempt
            succeeded (so this call is being asked for an ordinary next rung).

        Returns
        -------
        float or None
            The scale to attempt, which must be strictly less than ``converged[-1]`` and at least
            ``1.0``; or ``None`` to abandon the continuation, which the caller reports as a failure.
        """
        ...

    def planned_total(self, n_points: int) -> int | None:
        """How many points the ramp will visit including the target, or ``None`` if not knowable.

        Only used to label a rung for a human. A schedule that may retreat cannot answer, and says so.
        """
        ...


class GeometricReynoldsSchedule(eqx.Module):
    """Geometric Reynolds spacing: each up-step multiplies the Reynolds number by a fixed ratio.

    With ``ratio = 10`` (the default, one decade per step) the anchor sits at ``Re_target / 10 ** N``
    and every step raises ``Re`` by a decade, so ``n_points`` is the number of decades of continuation:
    ``n_points = 1`` anchors one decade below the target and visits ``(10.0, 1.0)``; ``n_points = 2``
    anchors two decades below and visits ``(100.0, 10.0, 1.0)``. Geometric spacing is the natural
    choice because the convective
    nonlinearity scales multiplicatively with ``Re``, and each up-step is seeded by a *converged*
    neighbour, so a factor-``ratio`` jump from a converged solution is far easier than the same jump
    from a cold start. A harder target is reached by raising ``n_points`` (deeper anchor, more rungs).

    Attributes
    ----------
    ratio : float
        The Reynolds-number multiplier per up-step (equivalently the molecular-viscosity divisor).
        Default ``10.0``.
    """

    ratio: float = eqx.field(static=True, default=10.0)

    def anchor(self, n_points: int) -> float:
        """``ratio ** n_points`` -- the anchor sits that many steps below the target."""
        return float(self.ratio**n_points)

    def next_scale(self, converged: tuple[float, ...], failed: float | None) -> float | None:
        """One step down from the last converged rung, or ``None`` if a rung failed.

        **A failed rung ends the continuation**, which is this schedule's defining property: the ladder
        is fixed in advance, so there is nothing gentler to fall back to and the caller is told to
        re-run with a larger ``n_points``. :class:`AdaptiveReynoldsSchedule` is the one that retreats.
        """
        if failed is not None:
            return None
        return _step_down(converged[-1], self.ratio)

    def planned_total(self, n_points: int) -> int:
        """``n_points + 1`` -- the ladder is fixed, so its length is known before it is walked."""
        return n_points + 1


class AdaptiveReynoldsSchedule(eqx.Module):
    """Geometric spacing that **retreats onto a gentler step** when a rung fails, rather than giving up.

    The fixed ladder's weakness is that its step is chosen before anything is known: a rung that turns
    out to be too big a jump ends the whole continuation, discarding every rung already converged, and
    the caller is told to re-run from the beginning with a larger ``n_points``. That retry loop is the
    standard step-size control of numerical continuation with the human as the controller and a full
    restart as the retry -- so this closes it, keeping the converged rungs and shortening only the step
    that failed.

    On a failure the next attempt is the **geometric mean** of the root already in hand and the scale
    that failed from it, which halves the step in the log of the viscosity scale -- the parameterization
    the ramp is geometric in, so a bisection there is a bisection of the step. Repeated failures bisect
    again, and the schedule gives up once the step has shrunk below :attr:`min_ratio`, since a step that
    small is evidence the difficulty is not the jump size. After a retreat the step is allowed to grow
    back by :attr:`recovery` per successful rung, capped at :attr:`ratio`: without that the ramp would
    carry one hard rung's caution all the way to the target, and with it unbounded the rung after a
    retreat would jump straight back to the step that had just failed.

    **This adapts the spacing, not the aggression.** It reacts only to whether a rung converged, which
    is the one signal available without observing the march -- deliberately, because three independent
    attempts to *predict* a bad continuation step from cheaper signals have all failed here (a static
    census on the three-dimensional case, a line-search-trend rule, and the curvature of the solution
    path, which ranked the step that diverged as the safest of three). Reacting to the outcome is what
    is left, and it is what continuation codes do.

    Attributes
    ----------
    ratio : float
        The Reynolds-number multiplier per rung when nothing has gone wrong -- the step this schedule
        returns to, and never exceeds. Default ``10.0`` (static).
    recovery : float
        How much of the step is won back per successful rung after a retreat, as a multiplier on the
        step last achieved. ``1.0`` never recovers; large values return to ``ratio`` immediately.
        Default ``1.5`` (static).
    min_ratio : float
        The step below which the schedule stops retreating and reports failure, as a ratio of the last
        converged scale to the proposed one. Default ``1.05`` (static).
    """

    ratio: float = eqx.field(static=True, default=10.0)
    recovery: float = eqx.field(static=True, default=1.5)
    min_ratio: float = eqx.field(static=True, default=1.05)

    def anchor(self, n_points: int) -> float:
        """``ratio ** n_points`` -- the same anchor as the fixed ladder, since only the path adapts."""
        return float(self.ratio**n_points)

    def next_scale(self, converged: tuple[float, ...], failed: float | None) -> float | None:
        """A gentler scale after a failure, otherwise one step down from the last converged rung."""
        if failed is not None:
            if not converged:
                # The anchor itself failed. There is no converged root to retreat toward, and a
                # gentler step is not the remedy: the ramp has to START lower, which is `n_points`.
                return None
            root = converged[-1]
            retreat = math.sqrt(root * float(failed))
            return None if root / retreat < self.min_ratio else retreat
        root = converged[-1]
        achieved = converged[-2] / root if len(converged) >= 2 else self.ratio
        return _step_down(root, min(self.ratio, achieved * self.recovery))

    def planned_total(self, n_points: int) -> None:
        """``None`` -- how many rungs this takes is not known until it has taken them."""
        return None


#: A hard cap on rung ATTEMPTS, so a schedule that never descends fails loudly instead of running for
#: ever. It bounds a defect, not a hard case: a schedule's own ``min_ratio`` is what limits how far it
#: retreats, and a ramp that legitimately needs this many rungs is one whose step is far too small.
_MAX_ATTEMPTS = 100

#: How close to ``1.0`` a computed scale must be to count as having ARRIVED at the target. A ladder
#: built by repeated division lands on ``1.0`` exactly only when the ratio divides the anchor exactly,
#: so without this a ramp at, say, ``ratio = 3.1623`` would finish with a spurious extra rung at
#: ``1.0000000000000002`` -- a companion microscopically different from the target, solved for nothing
#: and reported as its own point.
_TARGET_TOLERANCE = 1e-9


def _step_down(scale: float, factor: float) -> float:
    """``scale / factor``, snapped to exactly ``1.0`` once it has reached the target.

    The one home for "take a step down the ladder", so the two schedules cannot disagree about when a
    ramp has arrived.
    """
    stepped = float(scale) / float(factor)
    return 1.0 if stepped <= 1.0 + _TARGET_TOLERANCE else stepped


#: The keywords that drive the *solve* rather than configure a continuation, derived from
#: :func:`~aquaflux.turbulence.solve_coupled`'s own signature so the two cannot drift apart. Everything
#: else a caller passes -- ``method``, ``reference_state``, and every keyword bound for
#: ``coupled_continuation`` behind ``**continuation_kwargs`` -- describes a continuation, and reaches the
#: target solve only when that solve is the one building it.
_SOLVE_ONLY = frozenset(inspect.signature(solve_coupled).parameters) - {
    "coupled",
    "flow",
    "k",
    "omega",
    "method",
    "reference_state",
    # `inspect.signature` names the `**kwargs` parameter itself; it is not a keyword anyone passes.
    "continuation_kwargs",
}


def solve_reynolds_continuation(
    coupled: CoupledRANS,
    n_points: int,
    *,
    schedule: ReynoldsSchedule | None = None,
    intermediate_rtol: float | None = 1e-2,
    intermediate_atol: float | None = None,
    point_setup: Callable[[CoupledRANS, jnp.ndarray, ReynoldsPoint], dict] | None = None,
    seed_projection: Callable[[CoupledRANS, jnp.ndarray, ReynoldsPoint], jnp.ndarray] | None = None,
    **solve_kwargs: object,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the coupled RANS system by Reynolds-number continuation, returning the target-Re root.

    Runs ``n_points`` lower-Reynolds solves from an easy anchor up to the target, each seeded by the
    previous converged solution, then the final solve at the case's true viscosity. The lowest-Re
    point self-starts from the hybrid initialization; the returned ``(flow, k, omega)`` is the target-Re
    root, **identical** to what a direct :func:`~aquaflux.turbulence.solve_coupled` would reach -- the
    continuation only changes the path.

    The whole thing is an outer wrapper around :func:`~aquaflux.turbulence.solve_coupled` and is
    agnostic to the per-Re globalization: every keyword in ``solve_kwargs`` is forwarded to each per-Re
    solve, so the pseudo-transient march, the dual-time march (``inner_steps > 1``, whose observed rungs
    default to the :class:`~aquaflux.solve.DualTimeControl` Courant ramp), the preconditioner options and
    the observers all compose here unchanged.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler at the **true target** viscosity; the differentiable parameter
        pytree for the adjoint. The lower-Re companions are built from it by scaling the molecular
        viscosity (:meth:`~aquaflux.turbulence.CoupledRANS.with_scaled_molecular_viscosity`), so the
        case is never restated.
    n_points : int
        The number of lower-Reynolds continuation points before the target (``>= 0``). ``0`` is a plain
        direct solve (no continuation). This is the whole user surface.
    schedule : ReynoldsSchedule, optional
        The Reynolds spacing; defaults to :class:`GeometricReynoldsSchedule` (one decade per step,
        anchor at ``Re_target / 10 ** n_points``), whose ladder is fixed in advance and whose failure
        ends the continuation. Pass :class:`AdaptiveReynoldsSchedule` to have a rung that fails
        **retreat onto a gentler step** and carry on from the rungs already converged, instead of
        discarding them and asking for a re-run at a larger ``n_points``.
    intermediate_rtol : float or None
        The relative residual tolerance for the **lower-Re** points, overriding ``rtol`` from
        ``solve_kwargs`` for those solves only (the target solve always uses the caller's ``rtol``).
        The intermediate solutions are only initial guesses for the next Reynolds number, so converging
        them to the tight target tolerance is wasted work -- a loose value develops the field enough to
        seed the next point at a fraction of the cost. Default ``1e-2``. Pass ``None`` to converge every
        point to the caller's ``rtol`` (no loosening).
    intermediate_atol : float or None
        The **absolute** residual tolerance for the lower-Re points, overriding ``atol`` for those solves
        only. The stopping test is ``‖R‖ <= atol + rtol·‖R₀‖``, so pairing this with ``rtol=0`` converges
        each seed point to a fixed level rather than to a fraction of its own starting residual. Prefer it
        for a self-normalizing residual measure (the default row-equilibrated one already reports a
        fractional change per equation): every point re-bases its own ``‖R₀‖``, and a Reynolds jump makes
        the inherited field a *worse* seed, so a purely relative bar can let a later point stop at a worse
        absolute residual than an earlier point already reached. ``None`` (default) leaves ``atol``
        untouched.
    point_setup : callable, optional
        ``(companion, seed_state, point) -> dict``, a **per-Reynolds-point** builder of extra ``solve_coupled``
        keyword arguments (merged over ``solve_kwargs`` for that point). It exists for a preconditioner that
        is both **per-companion and per-state** — chiefly the complete-LU β-tracking hook, whose
        ``continuation`` is frozen at the point's own viscosity *and* seed state and whose
        ``precondition_step`` closes over the point's own residual — which the single target-specific
        ``continuation`` cannot express across the whole ramp. It is called for **every** point (lower-Re
        and target) with that point's companion assembler and its **packed seed coupled state**
        (:meth:`~aquaflux.turbulence.CoupledRANS.state_from_physical` of the seed fields; the lowest point's
        seed is materialized from :func:`~aquaflux.turbulence.hybrid_initialize` here, so the built
        continuation freezes at the same state the solve starts from). Typical use::

            point_setup=lambda comp, state, point: {
                "continuation": coupled_lu_continuation(comp, state, inner_steps=..., inner_tol=...),
                "refresh": RefreshPolicy(precondition_step=lu_beta_tracking_refresh(comp)),
            }

        ⚠️ ``precondition_step`` lives on :class:`~aquaflux.solve.RefreshPolicy`, not on
        ``solve_coupled``. This example passed it bare until 2026-08-20, where it was absorbed by
        ``**continuation_kwargs`` and the hook simply never ran; ``solve_coupled`` now rejects it
        instead. **A key here that is neither a ``solve_coupled`` parameter nor a setting for a
        continuation it builds is an error, not a no-op.**

        **Forward-only** (the ``precondition_step`` it returns raises under ``jax.grad``, like the other
        observed-march hooks), so leave it ``None`` when differentiating and use the ``continuation`` path
        instead. ``None`` (default) leaves the ramp byte-identical: each point builds its own continuation
        from :func:`~aquaflux.turbulence.solve_coupled`'s defaults, and the seed is passed through as-is
        (the lowest point self-starts inside ``solve_coupled``). When set, its keys **override** any
        ``continuation`` / ``reference_state`` in ``solve_kwargs`` (they are mutually exclusive uses).
    seed_projection : callable, optional
        ``(companion, seed_state, point) -> state``, a **per-point correction of the seed itself**,
        applied to the packed coupled state before that point's solve begins. It exists because a rung
        inherits the previous rung's converged root, and **not everything in that state is a solved
        quantity**: some of it is imposed by the model from the case parameters, and those parts are
        wrong for the new rung by construction.
        :func:`~aquaflux.turbulence.wall_consistent_state` is the correction written for it::

            seed_projection=lambda comp, state, point: wall_consistent_state(comp, state)

        It is the seam ``point_setup`` cannot be: that hook returns *keyword arguments* and the seed
        travels positionally, so a correction to the state has nowhere to go through it.

        **Forward-only, and it cannot move the answer.** It changes where a point's march starts and
        nothing else; the root each point converges to and the target adjoint are properties of the
        residual. Every point's result is ``stop_gradient``-ed before it seeds the next in any case.

        ⚠️ **Returning the argument unchanged is a true no-op, and a correction that declines a point
        should do exactly that.** The state is only unpacked and repacked when the projection returns
        something different, because that round trip inverts the scalar transforms and is not the
        identity in floating point under a log-solved field -- and a declined rung has to stay
        bit-identical to the arm it is being compared against.
    **solve_kwargs
        Forwarded to every per-Re :func:`~aquaflux.turbulence.solve_coupled`. ``continuation`` and
        ``reference_state`` are **target-specific** (a preconditioner frozen at the target viscosity),
        so they are applied to the final solve only; each lower-Re point builds its own continuation at
        its own viscosity. The split runs the other way too: ``method`` and every keyword bound for
        :func:`~aquaflux.turbulence.coupled_continuation` describe a continuation *this function builds*,
        so when ``continuation`` is supplied they reach the **ramp** only — the target is not building
        one. ⚠️ A ``point_setup`` that returns a ``continuation`` supplies one to **every** point, which
        leaves such settings dead everywhere; ``solve_coupled`` then rejects them rather than dropping
        them, so pass them to the builder inside ``point_setup`` instead.

    Returns
    -------
    tuple of jnp.ndarray
        The converged target-Re ``(flow, k, omega)``.

    Raises
    ------
    ValueError
        If ``n_points < 0``.
    RuntimeError
        If a lower-Reynolds continuation point fails to converge (naming the point and its scale, and
        suggesting a larger ``n_points``). The final target solve fails the usual way
        (:func:`~aquaflux.turbulence.solve_coupled` raises directly).

    Notes
    -----
    **Differentiability.** The lower-Re ramp only produces an initial guess: the companions are built
    from a ``stop_gradient`` copy of ``coupled`` and each intermediate result is ``stop_gradient``-ed
    before it seeds the next, so the ramp never tapes. The final solve runs on the **live** ``coupled``
    (the user's true differentiable parameters) from a stopped seed, so ``jax.grad`` through this
    function is **identical** to differentiating a direct :func:`~aquaflux.turbulence.solve_coupled` --
    exact and independent of ``n_points``. As with a direct solve, to differentiate, pass a
    ``continuation`` built on concrete parameters outside ``jax.grad`` (used by the final solve) and no
    forward-only keywords (``refresh`` / ``on_step`` / ``step_control`` / ``point_setup``).
    """
    if n_points < 0:
        raise ValueError(f"n_points must be >= 0, got {n_points}")
    schedule = schedule or GeometricReynoldsSchedule()

    # Build the companions from a stopped copy so the ramp -- which only makes an initial guess --
    # never tapes onto the target-Re adjoint.
    frozen = jax.lax.stop_gradient(coupled)
    # The split runs BOTH ways, and this is the whole of it. `continuation` / `reference_state` freeze a
    # preconditioner at the TARGET viscosity, so they belong to the final solve only and each lower-Re
    # point builds its own at its own viscosity. The mirror image is that everything which *configures*
    # a continuation this function builds -- `method`, and every keyword bound for `coupled_continuation`
    # -- belongs to the RAMP only, because the final solve is not building one when the caller supplied
    # it. Passing both at once is the ordinary case, not a mistake: the ramp needs the settings and the
    # target needs the pre-built step. Only the ramp strip existed, so a caller who did that reached
    # `solve_coupled` with a continuation *and* the settings for one -- which used to be silently
    # ignored there and is now a `TypeError`.
    ramp_kwargs = {
        key: value
        for key, value in solve_kwargs.items()
        if key not in ("continuation", "reference_state")
    }
    target_kwargs = (
        solve_kwargs
        if solve_kwargs.get("continuation") is None
        else {key: value for key, value in solve_kwargs.items() if key in _SOLVE_ONLY}
    )
    # The lower-Re points are only seeds for the next Reynolds number, so converge them loosely --
    # over-converging them to the target tolerance is wasted work.
    if intermediate_rtol is not None:
        ramp_kwargs["rtol"] = intermediate_rtol
    # The absolute counterpart. The stopping test is ``‖R‖ <= atol + rtol·‖R₀‖``, so a purely ABSOLUTE
    # target is ``rtol=0`` with ``atol`` the level to reach -- which is the meaningful form for a
    # self-normalizing residual measure (the default row-equilibrated one already reports a fractional
    # change per equation, so dividing it again by ‖R₀‖ makes the bar a property of the initial guess).
    # It matters most here: every point re-bases its own ‖R₀‖, and a Reynolds jump makes the inherited
    # field a WORSE seed, so a relative bar lets a later point stop at a worse absolute residual than an
    # earlier one already reached.
    if intermediate_atol is not None:
        ramp_kwargs["atol"] = intermediate_atol

    def _point_solve(assembler, seed_fields, base_kwargs, point):
        # One Reynolds point. Without `point_setup` this is the plain solve (byte-identical to before,
        # seed passed through — the lowest point self-starts inside solve_coupled). With it, materialize
        # the seed (hybrid start for the lowest point) so the per-point continuation freezes at the same
        # state the solve begins from, then merge the point's own continuation / precondition_step over
        # the base kwargs.
        if point_setup is None and seed_projection is None:
            return solve_coupled(assembler, *seed_fields, **base_kwargs)
        if seed_fields[0] is None:
            seed_fields = hybrid_initialize(assembler.momentum, assembler.turbulence)
        packed = assembler.state_from_physical(*seed_fields)
        # The projection runs BEFORE `point_setup`, so a per-point continuation is frozen at the state
        # the solve will actually begin from rather than at the one it was handed. The two hooks would
        # otherwise disagree about where the point starts, which is exactly the kind of mismatch the
        # seed materialization above exists to prevent.
        if seed_projection is not None:
            projected = seed_projection(assembler, packed, point)
            # ⚠️ Identity-checked, so returning the state unchanged is genuinely a no-op. Unpacking is
            # not free of consequence: `physical_fields` inverts the scalar transforms, so a state that
            # round-trips through `state_from_physical` and back differs from the original in its last
            # bits under a log-solved field (`exp(log(w))` is not the identity in floating point). A
            # projection that declines to act on a given point -- the usual shape, since the correction
            # belongs to a *handover* and the anchor inherits nothing -- must leave that point's march
            # bit-identical, or the arms it is being compared against are not matched where they agree.
            if projected is not packed:
                packed = projected
                seed_fields = assembler.physical_fields(packed)
        extra = {} if point_setup is None else point_setup(assembler, packed, point)
        return solve_coupled(assembler, *seed_fields, **{**base_kwargs, **extra})

    seed: tuple[jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None] = (None, None, None)
    converged: list[float] = []
    total = schedule.planned_total(n_points)
    attempt = schedule.anchor(n_points)

    for _ in range(_MAX_ATTEMPTS):
        # A rung and the target take the SAME path, differing only in which assembler and which
        # keywords they get. That is what lets a failed TARGET retreat too -- it is the hardest point
        # on the ramp, so it is the one most worth being able to insert a rung in front of, and a loop
        # that special-cased it could not. The target still runs on the live `coupled`, so its adjoint
        # is the direct solve's; every lower-Re rung runs on the stopped copy and never tapes.
        is_target = attempt <= 1.0 + _TARGET_TOLERANCE
        assembler = coupled if is_target else frozen.with_scaled_molecular_viscosity(attempt)
        point = ReynoldsPoint(len(converged) + 1, total, 1.0 if is_target else float(attempt))
        try:
            flow, k, omega = _point_solve(
                assembler, seed, target_kwargs if is_target else ramp_kwargs, point
            )
        except eqx.EquinoxRuntimeError as exc:
            retreat = schedule.next_scale(tuple(converged), float(attempt))
            if retreat is None:
                raise RuntimeError(
                    f"Reynolds-continuation {point.label} (molecular viscosity scaled by "
                    f"{attempt:g}, i.e. Reynolds number reduced by {attempt:g}x) failed to converge, "
                    f"and {type(schedule).__name__} offered no gentler step to retreat onto. Increase "
                    f"n_points for a deeper anchor, use AdaptiveReynoldsSchedule to shorten a failed "
                    f"step automatically, or check the case at this Reynolds number directly."
                ) from exc
            attempt = retreat
            continue

        if is_target:
            return flow, k, omega
        converged.append(float(attempt))
        # Stop the seed so the next companion's solve (and, ultimately, the target adjoint) does not
        # tape onto this intermediate root.
        seed = tuple(jax.lax.stop_gradient(field) for field in (flow, k, omega))

        following = schedule.next_scale(tuple(converged), None)
        if following is None or following >= attempt:
            raise RuntimeError(
                f"{type(schedule).__name__} did not advance the ramp after converging at viscosity "
                f"scale {attempt:g}: it returned {following!r}, which is not a smaller scale. A "
                f"schedule must descend strictly toward 1.0 so the continuation terminates."
            )
        attempt = following

    raise RuntimeError(
        f"Reynolds continuation did not reach the target in {_MAX_ATTEMPTS} rung attempts "
        f"(last scale {attempt:g}, {len(converged)} rungs converged). This is a runaway schedule "
        f"rather than a hard case: raise the schedule's min_ratio so it stops retreating sooner."
    )


#: The exponent relating a station's re-damping to the station's own viscosity ratio,
#: ``redamping = ratio ** _REDAMPING_EXPONENT``.
#:
#: **An empirical fit through ONE calibrated point, not a derived law.** ``0.6`` is the value that
#: reproduces the measured-good four-station setting on pitzDaily -- ``3.162 ** 0.6 = 2.0`` -- and it
#: interpolates sensibly, giving ``1.12`` at 24 stations where the viscosity moves only 1.21x per
#: station. What justifies the *shape* rather than the number is that a station warrants damping in
#: proportion to how far it moves the problem, which a constant cannot express: the same constant is
#: simultaneously too little at a coarse ramp and a runaway at a fine one.
_REDAMPING_EXPONENT = 0.6


class ViscosityRampHomotopy:
    """Walk the molecular viscosity down to the case's own value **within a single march**.

    The :class:`~aquaflux.solve.ResidualHomotopy` counterpart of :func:`solve_reynolds_continuation`,
    and the difference between them is what it exists for. The continuation solves each Reynolds rung
    as its **own** march: every rung is converged to a stopping tolerance and every rung restarts its
    step control. Both are pure loss on a rung whose only job is to seed the next one. Measured on a
    backward-facing step at ``Re ~ 25000`` with a two-decade ladder, the first rung spent 12 of its 28
    outer steps taking the residual from ``4.5e-04`` to ``7.2e-06`` -- and the decade of viscosity that
    followed put it straight back to ``4.5e-02``, six thousand times worse, so every one of those 12
    steps was discarded. Over the same handover the pseudo-timestep ramp restarted at its opening shift
    and spent a further 12 steps walking back down, a cost the docstring of that ramp notes is the same
    whether the seed it was handed was good or not.

    Marching the stations in one loop keeps the state, the shift and the preconditioner across every
    parameter change, and converges the target and nothing else.

    **⚠️ A station was believed to need several steps, and on this case that is REFUTED (2026-09-10).**
    The argument was that a station change re-points the refresh hook through ``rebind`` and forces a
    full preconditioner rebuild -- the most expensive single operation in the march -- so moving the
    viscosity every step would cost more than the steps it saves. Measured on pitzDaily the rebuild is
    **1.2-1.6 s** against a **~9 s** outer step, about a sixth of one, and per-step viscosity is the
    *cheapest* schedule tried: 24 stations x 1 step at ``redamping = 1.0`` costs **191** restart cycles
    against **261** for 4 x 3 at 2.0. :attr:`steps_per_station` is how many outer steps a station is
    held for, and on this case the answer is one.

    **⚠️ The cost balance inverts on a larger case, so treat that as a pitzDaily calibration rather than
    a general rule.** On the three-dimensional sibling a full re-materialize is recorded at ~36 s
    against a ~34 s outer step -- a whole step per rebuild instead of a sixth of one -- so a fine ramp
    would roughly double it there. Measure before carrying these numbers across.

    The ramp is **geometric**, for the same reason the ladder is: the convective nonlinearity scales
    multiplicatively with the Reynolds number, so equal ratios are equal difficulty. Station ``s`` of
    ``stations`` runs the viscosity scaled by ``anchor ** (1 - s / stations)``, from ``anchor`` at
    ``s = 0`` down to the target's own viscosity at ``s = stations``, which is the last station and the
    only one that may satisfy the march's stopping test.

    **⚠️ A station change is a compilation-cache hit only if the momentum viscosity is an ARRAY.**
    :meth:`~aquaflux.properties.Constant.scaled` on a plain Python ``float`` produces a value that
    rides on the *static* side of a jitted function and is compared by value, so every station would be
    a fresh cache key and recompile the whole coupled solve -- which on a real case is minutes per
    station and would swamp everything this class saves. Build the property as
    ``Constant(jnp.asarray(rho * nu))``. This is the same trap the rung ladder carries, but it bites
    harder here because a ramp changes the viscosity more often than a ladder does.

    **The target station is the case's own assembler**, not a copy scaled by one. The march's root and
    its adjoint therefore belong to the target problem exactly, as they do under the ladder -- the
    homotopy dissolves at the target rather than leaving a rescaled residual behind.

    This is a plain mutable object rather than an :class:`equinox.Module` because it caches the station
    it is currently on and re-points a preconditioner refresh as a side effect. It runs only on the
    eager, forward-only march (like the refresh hook it drives) and must never be on a differentiated
    path.

    Parameters
    ----------
    coupled : CoupledRANS
        The **target** assembler -- the case at its own molecular viscosity.
    anchor : float
        The molecular-viscosity multiplier the ramp starts at, i.e. the Reynolds number is divided by
        this at the first station. Must be ``>= 1``.
    stations : int
        How many stations the ramp is walked in before the target. ``stations`` ramp stations are
        visited (``s = 0 .. stations - 1``) and the target is station ``stations``, so the ratio
        between neighbours is ``anchor ** (1 / stations)``. Must be ``>= 1``.
    steps_per_station : int
        Outer steps each ramp station is held for. The target station is held for as long as the march
        needs, so this bounds only the ramp: it occupies ``stations * steps_per_station`` steps.
    redamping : float, optional
        What to multiply the pseudo-transient shift by on entering each ramp station, asking the march
        to re-damp for a problem that just got harder. ``1.0`` disables it.

        **Defaults to the station's own viscosity ratio raised to :data:`_REDAMPING_EXPONENT`**, so a
        station that barely moves the problem barely damps -- and to exactly ``1.0`` at
        ``steps_per_station == 1``, which is a requirement rather than a rounding (see Raises). A
        constant cannot serve both ends of the range: the value calibrated for a coarse ramp is a
        runaway on a fine one.

        **Not a tuning knob but the ramp's answer to a measured failure.** Within a station the shift
        control divides β by its own ``grow`` each step, so a station of ``s`` steps divides it by
        ``grow ** s`` -- ``1.5 ** 3 = 3.375`` at the defaults. Left unopposed on a pitzDaily
        four-station ramp that walks β 0.5 → 0.148 → 0.044 → **0.013**, and the case has a wall at
        **β ≈ 0.012**: the line search collapsed to ``alpha = 0``, ``|R|`` went 4.4e-03 → 4.4e-01, and
        the retry ladder had to escalate β to ~2 to recover -- 14 outer steps and 32 restart cycles
        worse than the same march re-damped. The default ``2.0`` leaves a net ``3.375 / 2 = 1.69`` per
        station, so β roughly halves per station and reaches 0.062 rather than 0.004.

        ⚠️ It is a **damping while the problem moves**, not a floor under β. The same march converged
        comfortably at β = 0.005 once the ramp had arrived and the state settled, at a shift that was
        fatal mid-ramp -- so the wall belongs to ``(state, β)`` and a learned floor would forbid the
        low-shift convergence that finishes the march.
    rebind : callable, optional
        ``assembler -> None``, called on every station **change** to re-point whatever must follow the
        viscosity -- in practice a preconditioner refresh hook's ``rebind``, so the operator each step
        is preconditioned by is fitted to the station that step solves. ``None`` leaves the
        preconditioner alone, which is correct only if it is rebuilt by some other means.

    Examples
    --------
    A two-decade ramp in four stations of three steps each, re-pointing an AMG refresh::

        homotopy = ViscosityRampHomotopy(
            coupled, anchor=100.0, stations=4, steps_per_station=3, rebind=refresh.rebind
        )

    Raises
    ------
    ValueError
        If ``anchor < 1``, or ``stations < 1``, or ``steps_per_station < 1``, or ``redamping < 1``, or
        ``redamping != 1`` when ``steps_per_station == 1`` -- every step then *enters* a station, so the
        march holds the step control on every step and it never adapts at all, leaving the shift to
        run away as ``beta_start * redamping ** n``.
    """

    def __init__(
        self,
        coupled: CoupledRANS,
        *,
        anchor: float,
        stations: int,
        steps_per_station: int,
        redamping: float | None = None,
        rebind: Callable[[CoupledRANS], None] | None = None,
    ) -> None:
        if anchor < 1.0:
            raise ValueError(
                f"anchor must be >= 1 (the ramp walks DOWN to the target), got {anchor}"
            )
        if stations < 1:
            raise ValueError(f"stations must be >= 1, got {stations}")
        if steps_per_station < 1:
            raise ValueError(f"steps_per_station must be >= 1, got {steps_per_station}")
        ratio = float(anchor) ** (1.0 / int(stations))
        if redamping is None:
            # Track the station's own size rather than sit at a constant: a station that moves the
            # viscosity a little warrants little damping. `1.0` at one step per station is not a
            # rounding of that rule but a requirement of it -- see the guard below.
            redamping = 1.0 if int(steps_per_station) == 1 else ratio**_REDAMPING_EXPONENT
        if redamping < 1.0:
            raise ValueError(
                f"redamping must be >= 1 (a station change damps, never accelerates), got {redamping}"
            )
        if int(steps_per_station) == 1 and redamping != 1.0:
            raise ValueError(
                "redamping must be exactly 1.0 when steps_per_station == 1: every step then enters a "
                "station, so the step control is held on every step and never adapts at all, and the "
                f"shift becomes beta_start * redamping ** n. Got redamping={redamping}."
            )
        self.coupled = coupled
        self.anchor = float(anchor)
        self.stations = int(stations)
        self.steps_per_station = int(steps_per_station)
        self.ratio = ratio
        self.redamping = float(redamping)
        self.rebind = rebind
        # The station whose assembler `_assembler` currently holds. -1 is "none entered yet", which is
        # not a station index, so the first `enter` always counts as a change and rebinds.
        self._station = -1
        self._assembler = coupled

    def station(self, step: int) -> int:
        """The station index outer step ``step`` runs, saturating at :attr:`stations` (the target)."""
        return min(step // self.steps_per_station, self.stations)

    def scale(self, station: int) -> float:
        """The molecular-viscosity multiplier at ``station``; exactly ``1.0`` at the target."""
        if station >= self.stations:
            return 1.0
        return float(self.anchor ** (1.0 - station / self.stations))

    def enter(self, step: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Make this step's station current -- rebinding on a change -- and return its residual.

        Returns ``assembler.residual`` as a **bound method**, so its arrays ride as dynamic leaves of a
        pytree and the compiled step stays a cache hit across a station change rather than recompiling
        the whole coupled solve at each one.
        """
        station = self.station(step)
        if station != self._station:
            self._station = station
            scale = self.scale(station)
            # The target station is the caller's own assembler, not `with_scaled_molecular_viscosity(1)`
            # -- the root and adjoint must belong to the case, and a rescale by one is a different
            # object holding a multiplied array rather than the original.
            self._assembler = (
                self.coupled
                if station >= self.stations
                else self.coupled.with_scaled_molecular_viscosity(scale)
            )
            if self.rebind is not None:
                self.rebind(self._assembler)
        return self._assembler.residual

    def arrived(self, step: int) -> bool:
        """Whether ``step`` runs the target problem, which is the only station that may stop the march."""
        return self.station(step) >= self.stations

    def shift_factor(self, step: int) -> float:
        """:attr:`redamping` -- the march re-damps by the same factor at every station change.

        Uniform across stations because the ramp is geometric: every change is the same multiplicative
        step in viscosity, so every change is the same increment in difficulty and warrants the same
        response. A non-geometric schedule would want this to track its own step sizes.

        The march reads it only on a step that enters a new station, so the constant is not applied
        anywhere else.
        """
        del step
        return self.redamping

    @property
    def ramp_steps(self) -> int:
        """Outer steps the ramp occupies before the target station begins."""
        return self.stations * self.steps_per_station
