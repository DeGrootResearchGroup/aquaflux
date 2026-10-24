"""A materialized-Jacobian preconditioner, kept current across every step of a march.

A field split, or one block inverse over the whole state, preconditions a shifted solve by inverting the
**assembled** Jacobian, recovered off the jit path by coloured probing. It is built at one state and one
shift and goes stale as the march moves both, so a march that uses one must also re-fit it -- on
evidence, since a multigrid re-fit is expensive -- and must hand the *same objects* to every step it
builds, because the inverse and the refresh hooks ride in static fields of the
Newton step and a new object recompiles the whole solve.

:class:`MaterializedSession` is that lifecycle: one probe, one inverse and one refresh hook, each created
at most once. It is written against :class:`MaterializedProblem`, which is everything the lifecycle needs
from a residual and nothing more: the assembler whose Jacobian is materialized, how to probe its graph,
which base shift policy to fit against, how to cut its fields for a split, and how to assemble the step.
The coupled RANS solve and the laminar flow solve each supply one.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..allocator import GlibcTrim, MemoryTrim
from ..solve import FieldLayout, LinearSolveRegime, NewtonStrategy, RefreshTiming, shift_of
from .block_inverse import BlockInverse
from .block_preconditioner import MaterializedBlockPreconditioner
from .field_split import FieldGroups, FieldSplitInverse, FieldSplitPreconditioner
from .jacobian_probe import JacobianProbe
from .materialized_spec import FieldSplit, MaterializedJacobian
from .monolithic_policy import (
    MonolithicFactorShiftPolicy,
    TracedFormShiftPolicy,
    with_traced_form,
)
from .pivot_floor import PivotFloor
from .sparse_jacobian import FieldwiseMatvec
from .traced_cycle import OffersTracedCycle, TracedForm, TracedSingleBlock
from .traced_field_split import offers_traced_split, traced_field_split

__all__ = [
    "BUILD_BETA",
    "PROBE_BATCH_SIZE",
    "VCYCLE_LINEAR_SOLVE",
    "BetaTrackingRefresh",
    "FieldwiseResidual",
    "MaterializedProblem",
    "MaterializedSession",
    "batched_field_jacobian_matvec",
    "batched_jacobian_matvec",
    "frozen_shift_diagonal",
    "jacobian_matvec",
]

#: How many coloured tangents share one vmapped jvp pass when materializing a Jacobian. Larger amortizes
#: dispatch over more probes; each column field's coloured probes run in ``ceil(n_colours / this)`` fused
#: passes instead of one call per probe.
#:
#: Measured on the 3D backward-facing step (399 probes; 47.2M structural nonzeros in the fixed sparsity
#: pattern, of which ~38.7-39.0M are live at any one state), wall / peak against the batch:
#:
#:     batch      1      2      4      8     16     32
#:     wall    11.7    8.4    6.7    5.9    5.8    6.5   s
#:     peak     380    383    388    399    419    460   MB
#:
#: Two things decide the eight. The curve has an interior optimum and **turns** -- 32 is slower than 16 --
#: so this is not "as large as memory allows"; and the memory it costs is ~2.5 MB per unit of batch, which
#: is nothing against the matrix being built, so the peak is not what picks the value. Eight is where the
#: processor time bottoms; sixteen is a hair faster in wall and slower in processor time, which on a
#: shared machine is the less trustworthy of the two.
#:
#: An earlier default of four came from a measurement -- "16 vs 4: ~2.2 GB against ~0.7 GB" -- taken when
#: the seed set and the response array dominated the peak, and NEITHER of those ever scaled with the
#: batch. Both are built a chunk at a time now, so that trade no longer exists.
PROBE_BATCH_SIZE = 8

#: The shift strength a materialized preconditioner's first build is fitted at when its spec leaves
#: ``build_beta`` unset. A frozen coarse space is chosen at that build and reused by every later refit, so
#: this is not only the first step's operator.
BUILD_BETA = 2.0

#: A multigrid V-cycle -- a field split's blocks, or one block inverse over the whole state. Restart 15
#: is the measured sweet spot for a one-V-cycle preconditioner: enough Arnoldi history for its convergence while checking the stop often enough not to
#: overshoot the loose tolerance deep into the next cycle (a larger restart costs ~2x the expensive host
#: V-cycle applies for the same trajectory).
#:
#: ⚠️ Two constraints bind ``max_restarts`` against the march's retry threshold, and both bite silently.
#: It is in raw ``lineax`` restarts, which carry a fixed ``+2`` per solve, while
#: ``retry.abort_above_cycles`` is in corrected cycles (:func:`~aquaflux.solve.restart_cycles`), so a
#: corrected cap of ``c`` is ``max_restarts = c + 2``. And the corrected count must stay **strictly
#: above** ``retry.abort_above_cycles``: the march's test is ``max_inner_cycles >
#: retry.abort_above_cycles``, so a cap landing exactly on the threshold does not trip the redo, and the
#: step accepts the truncated, non-converged direction instead of re-running it on a fresh
#: preconditioner.
VCYCLE_LINEAR_SOLVE = LinearSolveRegime(rtol=0.3, restart=15, max_restarts=60)


@eqx.filter_jit
def jacobian_matvec(assembler, state: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """``J(state) @ tangent`` -- the matrix-free Jacobian-vector product, compiled once.

    Everything it needs is an **argument**, including the assembler. A locally-defined ``jax.jit`` closure
    over the assembler is a fresh cache entry per closure, so each continuation rung -- which rebuilds the
    assembler at its own viscosity -- would recompile the probe from scratch, even though a scaled
    viscosity changes only leaf *values* and leaves the pytree structure identical. As an argument the
    assembler's arrays are ordinary traced leaves and every rung is a cache hit.

    Parameters
    ----------
    assembler : object
        Anything with a ``residual(state)`` method that is a pytree.
    state : jnp.ndarray
        The state to linearize at.
    tangent : jnp.ndarray
        The direction, same shape as ``state``.
    """
    return jax.jvp(assembler.residual, (state,), (tangent,))[1]


@eqx.filter_jit
def batched_jacobian_matvec(assembler, state: jnp.ndarray, tangents: jnp.ndarray) -> jnp.ndarray:
    """``J(state) @ tangents`` for a stack of tangents -- the batched form the coloured probe uses.

    The same directional derivative as :func:`jacobian_matvec` applied to each row; running them as a few
    fused passes amortizes dispatch. The responses agree with a per-tangent loop to rounding, not bit for
    bit: the batched program is compiled differently, and the compiler's choices (which operations it
    fuses, where it contracts a multiply and an add) move the last bits. Takes the assembler as an
    argument for the same reason.

    Parameters
    ----------
    assembler : object
        Anything with a ``residual(state)`` method that is a pytree.
    state : jnp.ndarray
        The state to linearize at.
    tangents : jnp.ndarray
        The stack of directions, shape ``(n_tangents, state.size)``.
    """
    return jax.vmap(lambda tangent: jax.jvp(assembler.residual, (state,), (tangent,))[1])(tangents)


@eqx.filter_jit
def batched_field_jacobian_matvec(
    assembler, layout: FieldLayout, state: jnp.ndarray, field: int, tangents: jnp.ndarray
) -> jnp.ndarray:
    """``J(state) @ tangents`` for tangents confined to one column field -- one program per field.

    Differentiates ``assembler.residual_with_field(state, field, values)`` in ``values`` alone, with
    every other field held at ``state`` as a constant of the derivative. Those fields carry no tangent
    at all, so nothing they alone reach is evaluated in the derivative; seeding the whole state with
    zeros outside the field (:func:`batched_jacobian_matvec`) evaluates all of it on zeros. The
    responses are those of :func:`batched_jacobian_matvec` for the same seeds embedded in the field, to
    rounding: the two programs are compiled differently.

    ``field`` is a static argument, so each column field compiles its own program once; the assembler
    is an argument for the reason :func:`jacobian_matvec` gives. The tangents' floating type is part of
    each program's compilation key too.

    Parameters
    ----------
    assembler : object
        A pytree with a ``residual_with_field(state, field, values)`` method (:class:`FieldwiseResidual`).
    layout : FieldLayout
        The state's layout, which says where ``field`` lies.
    state : jnp.ndarray
        The state to linearize at, shape ``(layout.size,)``.
    field : int
        The column field, counted over the layout's per-cell fields.
    tangents : jnp.ndarray
        The stack of directions within that field, shape ``(n_tangents, n_cells)``.

    Returns
    -------
    jnp.ndarray
        The responses, shape ``(n_tangents, layout.size)``.
    """
    values = state[layout.field_dofs(field) : layout.field_dofs(field + 1)]

    def residual(of_field):
        return assembler.residual_with_field(state, field, of_field)

    return jax.vmap(lambda tangent: jax.jvp(residual, (values,), (tangent,))[1])(tangents)


class FieldwiseResidual(Protocol):
    """An assembler whose residual can be evaluated with one per-cell field replaced.

    ``residual_with_field(state, field, values)`` must equal ``residual`` of ``state`` with column field
    ``field`` (counted over the layout's per-cell fields, the probe's field-major order) set to
    ``values``, shape ``(n_cells,)``, and must reach the operators through the fields as separate
    arrays -- never by repacking a flat state, whose concatenation would hand every other field an
    explicit zero tangent and undo the point of it (see :func:`batched_field_jacobian_matvec`).
    """

    def residual_with_field(
        self, state: jnp.ndarray, field: int, values: jnp.ndarray
    ) -> jnp.ndarray:
        """The residual at ``state`` with ``field`` replaced by ``values``."""


def _probe_products(
    probed, state: jnp.ndarray, by_field: FieldLayout | None
) -> tuple[Callable, Callable]:
    """The single and batched Jacobian products a materialize probes ``probed`` with, at ``state``.

    With ``by_field`` (the layout of a :class:`FieldwiseResidual`) the batched product takes one column
    field's seeds at a time (:class:`~aquaflux.precondition.sparse_jacobian.FieldwiseMatvec`); without
    it, the whole state's.
    """

    def matvec(v):
        return jacobian_matvec(probed, state, v)

    if by_field is None:
        return matvec, lambda seeds: batched_jacobian_matvec(probed, state, seeds)
    return matvec, FieldwiseMatvec(
        lambda field, seeds: batched_field_jacobian_matvec(probed, by_field, state, field, seeds)
    )


def frozen_shift_diagonal(base, beta: float, state: jnp.ndarray) -> jnp.ndarray:
    """The frozen pseudo-transient shift diagonal a factorization is built against, at ``state``.

    ``beta`` scales the base policy's shift diagonal; the ``stop_gradient`` keeps the frozen factorization
    off the differentiation path. Shared by the initial build and every in-place refresh, and left on the
    device, where a probed operator's shift is added.

    It asks the term for the shift at ``beta`` rather than scaling the diagonal itself, so a policy that
    runs a block at its own pseudo-timestep preconditions the operator it actually forms. Open-coding
    ``beta * diagonal`` here would drop that factor silently -- the march would run, and the
    preconditioner would simply be fitted to a different operator than the one being solved.

    Parameters
    ----------
    base : ShiftPolicy
        The policy whose shift diagonal is read.
    beta : float
        The shift strength.
    state : jnp.ndarray
        The state the diagonal is taken at.
    """
    return _frozen_shift_and_scale(base, beta, state)[0]


def _frozen_shift_and_scale(
    base, beta: float, state: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """:func:`frozen_shift_diagonal` and the base diagonal ``d`` it scales, from one shift term.

    The base diagonal is each row's pseudo-time scale independent of ``beta`` -- what a
    :class:`~aquaflux.precondition.PivotFloor` measures a pivot against -- so it is read off the same
    term rather than recovered by dividing the shift by ``beta``, which would fold in any per-row factor
    of ``beta`` the policy applies.
    """
    term = base.shift_term(state)
    return (
        jax.lax.stop_gradient(jnp.asarray(term.shift(beta))),
        jax.lax.stop_gradient(jnp.asarray(term.diagonal)),
    )


def _is_traced(pytree: object) -> bool:
    """Whether ``pytree`` holds a JAX tracer.

    A materialized preconditioner is assembled off the jit path from concrete arrays, so a tracer leaf
    means the caller has wrapped the solve in ``jax.grad`` / ``jvp`` / ``vmap``.
    """
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(pytree))


def _traced_form(inverse, precision: str) -> TracedForm | None:
    """The form of a fitted frozen inverse that a step applies on device, or ``None`` if it has none.

    A field split has one when both of its blocks offer a traced cycle; a single block inverse over the
    whole state has one when it offers a traced cycle itself. Anything else -- a block that is a host
    factorization -- is applied through a host callback.

    Parameters
    ----------
    inverse : FrozenInverse
        The fitted host inverse, as it is now; take a new form after each re-fit.
    precision : {"float64", "float32"}
        The precision the form is applied in.
    """
    if isinstance(inverse, FieldSplitInverse):
        return traced_field_split(inverse, precision) if offers_traced_split(inverse) else None
    if isinstance(inverse, OffersTracedCycle):
        form = TracedSingleBlock(inverse.traced_cycle(), inverse.n_dofs)
        return form if precision == "float64" else form.in_precision(precision)
    return None


def _release_superseded(superseded, current, inverse) -> None:
    """Free the arrays only a superseded traced form holds, once a re-fit has replaced it.

    Every step the session hands out is re-pointed at the re-fitted form (the march applies the update
    this hook returns; the session re-points its own stopping steps), so the superseded form's arrays
    are dead by contract. They are still referenced, though: a march's caller holds the step it started
    the march with for as long as the march runs (``solve.staged_march`` keeps the step it built as the
    argument of ``newton_march``, and a caller always holds its arguments for the call), and on a
    three-dimensional case that pins a whole hierarchy -- most of a gigabyte -- for the rest of the
    march. Freeing them here is what reclaims it; the alternative is a march that takes its starting
    step through a hand-off its caller does not keep, a change to the march's interface. A step kept
    across a re-fit and then applied fails loudly (JAX refuses a deleted array) rather than applying a
    stale preconditioner.

    Arrays the re-fitted form or the inverse still holds -- a frozen coarsening's index arrays, which a
    re-fit reuses, or a hierarchy shared with a form applied in its own precision -- are kept.
    """
    if superseded is None:
        return
    held = {id(leaf) for leaf in jax.tree.leaves(current)}
    held.update(id(leaf) for leaf in jax.tree.leaves(_held_state(inverse)))
    for leaf in jax.tree.leaves(superseded):
        if isinstance(leaf, jax.Array) and id(leaf) not in held and not leaf.is_deleted():
            leaf.delete()


def _held_state(inverse) -> list:
    """The traced state each block of an inverse currently holds, for :func:`_release_superseded`."""
    blocks = (
        (inverse.leading, inverse.trailing)
        if isinstance(inverse, FieldSplitInverse)
        else (inverse,)
    )
    return [block.traced_cycle().state for block in blocks]


class BetaTrackingRefresh:
    """The ``refresh_preconditioner`` hook that re-fits a monolithic inverse as the shift strength moves.

    Called as ``refresh(active_step, state, due=False)`` before every step, it reads ``β`` from the step's
    :class:`~aquaflux.solve.ConstantRelaxation` schedule and re-fits its materialized preconditioner in
    place at ``J(state) + β·d(state)``, with ``d`` the step's base shift diagonal, on its first call, after
    each :meth:`rebind`, and whenever the march passes ``due``. It returns whether it re-fitted, so a
    caller carrying the inverse in the step as data knows when to hand the march a new step. A multigrid re-materialize is too expensive to pay every step, so between those the
    re-fit is left to the cost of the step's own solves: the dual-time loop's ``refresh_on_cycles`` ends
    a step whose solve ran long, and the march then calls this with ``due`` at the iterate that step
    kept.

    It is a plain mutable object, deliberately not an ``equinox.Module``: it re-fits a host
    preconditioner in place, so it runs only on the eager forward march and must never be on a
    differentiated path, and a Newton step holds it in a *static* field compared by identity -- which is
    what lets every continuation rung share one hook, and with it one compiled step.

    Parameters
    ----------
    assembler : object
        The residual assembler (supplies the Jacobian-vector product); a pytree with ``residual``.
    probe : JacobianProbe
        The shared colouring plan, de-compression map and assembler stand-in.
    preconditioner : MaterializedJacobianPreconditioner
        The host preconditioner it re-fits (``refresh_in_place``). Held here rather than read off the
        step, because a step may carry the inverse in a traced form derived from it
        (:class:`~aquaflux.precondition.monolithic_policy.TracedFormShiftPolicy`), which has nothing to re-fit.
    refit_beta_floor : float
        A lower bound on the shift strength the **preconditioner** is refreshed at: it is built at
        ``max(beta, refit_beta_floor)`` while the march keeps solving at its own ``beta``. ``0.0``
        (default) tracks ``beta`` exactly. It is not the march's own shift floor
        (:attr:`~aquaflux.solve.Globalization.beta_floor`), which bounds the shift the solve runs at.
    observer : callable, optional
        ``(timing: RefreshTiming) -> None``, called on each refresh with which branch ran, its total
        seconds, and its per-phase costs. ``None`` (default) elides the call.
    pivot_floor : PivotFloor, optional
        Fit each re-fit with its near-zero pivots moved out to their rows' pseudo-time diagonals,
        measured on the diagonal the fit would otherwise see (the Jacobian's and the shift). It changes the fit only; the operator the steps solve is left as it is. ``None``
        (default) fits that diagonal unchanged.
    floorable_rows : np.ndarray, optional
        Boolean, shape ``(n_dofs,)``: the only rows ``pivot_floor`` may change
        (:attr:`MaterializedProblem.floorable_fields`). ``None`` (default) lets it change any shifted row.
    by_field : FieldLayout, optional
        The state's layout when the assembler is a :class:`FieldwiseResidual`, so each column field is
        probed by its own program (:func:`batched_field_jacobian_matvec`); ``None`` (default) probes
        with whole-state seeds.
    memory_trim : MemoryTrim, optional
        Called just before each re-fit's probe, to give the allocator's kept, freed memory back to the
        operating system (:class:`~aquaflux.allocator.GlibcTrim`), so a re-fit's own transient lands on
        a lower resident set. It changes no value. ``None`` (default) leaves the allocator alone.

    Attributes
    ----------
    assembler : object
        The assembler this hook currently refreshes for; :meth:`rebind` changes it.
    floored_rows : np.ndarray
        The rows ``pivot_floor`` changed at the last re-fit, shape ``(n_floored,)``; empty without one.
    """

    def __init__(
        self,
        assembler: object,
        probe: JacobianProbe,
        preconditioner: object,
        *,
        refit_beta_floor: float = 0.0,
        observer: Callable[[RefreshTiming], None] | None = None,
        pivot_floor: PivotFloor | None = None,
        floorable_rows: np.ndarray | None = None,
        by_field: FieldLayout | None = None,
        memory_trim: MemoryTrim | None = None,
    ) -> None:
        self._probe = probe
        self._memory_trim = memory_trim
        self._by_field = by_field
        self._pivot_floor = pivot_floor
        # On the device, where the floor is applied, once rather than at every re-fit.
        self._floorable_rows = None if floorable_rows is None else jnp.asarray(floorable_rows)
        self.floored_rows = np.zeros(0, dtype=np.int64)
        self._preconditioner = preconditioner
        self._refit_beta_floor = refit_beta_floor
        self._observer = observer
        # Pending on the first call -- the build froze the preconditioner at its own shift, not the
        # march's -- and again after `rebind`, since the standing preconditioner then describes the
        # PREVIOUS companion.
        self._full_rebuild_pending = True
        self.assembler = assembler
        # What the coloured probe differentiates: the assembler itself unless the probe carries a
        # stand-in (`JacobianProbe.narrow`). Stored rather than derived per call so a rebind narrows once.
        self._probed = probe.narrow(assembler)

    def __call__(
        self, active_step: NewtonStrategy, state: jnp.ndarray, *, due: bool = False
    ) -> bool:
        """Re-fit the preconditioner at ``state`` when this cadence calls for it, or when ``due``.

        Returns whether it re-fitted.

        The march passes ``due`` before the step after one that ended because an inner solve reached
        ``refresh_on_cycles``. The march's expensive inner solves are **stale-preconditioner** effects,
        not hard operators: at the hardest solve of a three-dimensional coupled march a preconditioner
        rebuilt at that very iterate converged in an order of magnitude fewer cycles than the march's
        own. Ending the step there and re-fitting before the next keeps the step's progress, where the
        alternative reaction (abort the step and escalate β) discards both the work and the
        pseudo-timestep. Reacting is also what makes this a *replacement* for a scheduled refresh rather
        than an addition to one: the right interval is regime-dependent in a way no fixed cadence can
        track (one step of staleness is nearly free at a large shift and dominates the solve at a small
        one).
        """
        beta = shift_of(active_step)
        if beta is None:
            raise ValueError(
                "a β-tracking refresh needs the step's shift strength as a readable constant -- pair it "
                "with a DualTimeControl (which sets a ConstantRelaxation β), not the default "
                f"switched-evolution schedule ({type(active_step).__name__} has no readable β)."
            )
        started = time.perf_counter()
        if not (self._full_rebuild_pending or due):
            self._report("none", started)
            return False
        # Named for WHY it ran: "cost" when only a step's solve cost asked for it, so a log can tell
        # the re-fits the march's own cadence pays for from the ones its solves provoked.
        kind = "full" if self._full_rebuild_pending else "cost"
        self._full_rebuild_pending = False
        if self._memory_trim is not None:
            self._memory_trim()
        self._report(kind, started, self._materialize_at(active_step, state, float(beta)))
        return True

    def rebind(self, companion: object) -> None:
        """Point this hook at another companion of the same case, and force the next refresh to be full.

        A continuation solves a sequence of companions differing only in their molecular viscosity. Each
        is a separate solve, and rebuilding a preconditioner per solve recompiles the whole step, because
        the preconditioner rides in a *static* field of the Newton step and is compared by identity.
        Rebinding one hook instead lets every solve share a single preconditioner object -- so the
        compiled step is a cache hit across a rung boundary -- while each rung's inverse is still fitted
        to its own problem, at its own state and shift, by the refresh the march runs before that
        segment's first step.

        The standing preconditioner describes the previous companion, so the next refresh is forced to a
        **full** re-materialize at the new one. Forward-only, like everything else on this hook. The
        companion must be the same case -- same mesh, same layout, same schemes -- since the colouring
        plan and the gather map are not rebuilt.

        Parameters
        ----------
        companion : object
            The assembler the following segment solves.
        """
        self.assembler = companion
        self._probed = self._probe.narrow(companion)
        self._full_rebuild_pending = True

    def _materialize_at(
        self, step: NewtonStrategy, state: jnp.ndarray, beta: float
    ) -> tuple[tuple[str, float], ...]:
        """Re-materialize the preconditioner at ``state``, shifted at the floored ``beta``.

        The preconditioner's shift is floored independently of the march's own ``beta``. As ``beta -> 0``
        the shift's diagonal dominance vanishes and a frozen V-cycle degrades, but the OPERATOR must keep
        the small ``beta`` to make pseudo-transient progress. Flooring only the preconditioner's copy
        keeps the V-cycle in a regime it inverts well while the solved system is untouched, so the
        converged root and its adjoint are unchanged. The resulting mismatch SATURATES at
        ``refit_beta_floor * d`` rather than growing without bound the way a stale (never-refreshed)
        preconditioner's does.

        The preconditioner materializes through the coloured probe in its batched form, one column field
        at a time when the problem's assembler offers it (``by_field``). Every probe takes the assembler
        as an argument to a module-level jitted function, so a :meth:`rebind` changes no compilation key
        of theirs.
        """
        frozen = jax.lax.stop_gradient(jnp.asarray(state))
        shift, scale = _frozen_shift_and_scale(
            step.shift_policy.base, max(beta, self._refit_beta_floor), frozen
        )
        probed = self._probed
        self.floored_rows = np.zeros(0, dtype=np.int64)
        floor = self._pivot_floor

        def correction(jacobian_diagonal: jnp.ndarray) -> jnp.ndarray:
            change = floor.correction(jacobian_diagonal + shift, scale, rows=self._floorable_rows)
            # The one value read back: which rows were floored, for the march's log.
            self.floored_rows = np.flatnonzero(np.asarray(change))
            return change

        matvec, batched_matvec = _probe_products(probed, frozen, self._by_field)
        return self._preconditioner.refresh_in_place(
            matvec,
            self._probe.plan,
            shift,
            batched_matvec=batched_matvec,
            probe_batch_size=PROBE_BATCH_SIZE,
            structure=self._probe.structure,
            diagonal_correction=None if floor is None else correction,
        )

    def _report(
        self, kind: str, started: float, phases: tuple[tuple[str, float], ...] | None = None
    ) -> None:
        """Tell an injected observer which branch ran and what each part of it cost.

        The total alone cannot be acted on: a refresh dominated by the coloured jvp probe and one
        dominated by the multigrid setup take the same wall time and call for opposite fixes.
        """
        if self._observer is not None:
            self._observer(RefreshTiming(kind, time.perf_counter() - started, tuple(phases or ())))


class MaterializedProblem(abc.ABC):
    """What a :class:`MaterializedSession` needs from a residual, and nothing more.

    A residual supplies the assembler whose Jacobian is materialized and four decisions that are its own:
    how its graph is probed, which base shift policy the inverse is fitted against, how its fields are cut
    for a field split, and how its Newton step is finally assembled around the fitted inverse. Everything
    else -- the probe's lifecycle, the inverse family, the refresh hook, the rebind -- is the session's.

    Implementations are immutable values: :meth:`with_assembler` returns a new problem rather than
    mutating one, so a session can rebind across continuation rungs without aliasing.
    """

    #: The fields, by layout name, whose shifted diagonal pivot a source term can take through zero: the
    #: only rows a :class:`~aquaflux.precondition.PivotFloor` may change. A pivot is judged against its
    #: row's pseudo-time diagonal, and a row whose pivot is merely small beside that diagonal without being
    #: anywhere near singular (a momentum row) would be changed for nothing. Empty (default): no field,
    #: and a session refuses a pivot floor for this problem rather than ignore it.
    floorable_fields: tuple[str, ...] = ()

    #: Whether :attr:`assembler`, and the stand-in the probe narrows it to, is a
    #: :class:`FieldwiseResidual`, so that the coloured probe differentiates it in one column field at a
    #: time and the other fields' tangents are never formed. ``False`` (default): the probe seeds the
    #: whole state, as for any residual.
    probes_by_field: bool = False

    @property
    @abc.abstractmethod
    def assembler(self) -> object:
        """The residual assembler: a pytree with a ``residual(state)`` method."""

    @property
    @abc.abstractmethod
    def layout(self) -> FieldLayout:
        """The flat state layout, whose ``size`` and ``n_fields`` a rebind must preserve."""

    @abc.abstractmethod
    def with_assembler(self, assembler: object) -> MaterializedProblem:
        """This problem on another companion assembler of the same case."""

    @abc.abstractmethod
    def probe(self, settings: dict, groups: FieldGroups | None) -> JacobianProbe:
        """The coloured probe for this problem's graph.

        Parameters
        ----------
        settings : dict
            The spec's set probe fields (``stencil_reach``, ``column_reach``, ``gradient_sweeps``).
        groups : FieldGroups or None
            The partition of the field split the probe feeds, or ``None`` for a probe of the whole
            matrix (see :meth:`~aquaflux.precondition.JacobianProbe.build`).
        """

    @abc.abstractmethod
    def groups(self) -> FieldGroups | None:
        """How the state's fields are cut into a leading and a trailing group, or ``None``.

        ``None`` means there is nothing to split (a single group of fields), and a
        :class:`~aquaflux.precondition.FieldSplit` inverse is refused for the problem.
        """

    @abc.abstractmethod
    def bind_march(self, march: dict) -> dict:
        """The march keywords validated and completed with their defaults.

        Everything in it is the problem's to interpret in :meth:`shift_source` and :meth:`build_step`.
        """

    @abc.abstractmethod
    def shift_source(self, state: jnp.ndarray, march: dict):
        """The base shift policy the inverse is fitted against, at ``state``."""

    @abc.abstractmethod
    def build_step(
        self,
        state: jnp.ndarray,
        policy,
        march: dict,
        base_regime: LinearSolveRegime,
    ) -> NewtonStrategy:
        """The Newton step around ``policy``, configured by ``march``.

        ``policy`` is the base shift policy glued to the fitted inverse -- a
        :class:`~aquaflux.precondition.MonolithicFactorShiftPolicy` for a host inverse, a
        :class:`~aquaflux.precondition.monolithic_policy.TracedFormShiftPolicy` for one whose blocks are all
        traced. Both offer ``shift_term`` and ``adjoint_factory()``, which is all a step needs. The
        session chooses between them; the problem never does. ``base_regime`` is the Krylov regime of
        the inverse family; the march's ``linear_solve`` moves it.
        """


#: The trim every session hands its refresh hook unless told otherwise. A re-fit's transient lands on top
#: of the freed memory the previous step's program left in the allocator's arenas, so returning that
#: memory first lowers the peak a re-fit sets; it changes no value.
_RETURN_FREED_MEMORY = GlibcTrim()


class MaterializedSession:
    """The materialized-Jacobian family's session: one probe, one inverse and one refresh hook.

    Everything expensive or identity-bearing is created at most once. The probe (a colouring plan and
    its de-compression map, the largest allocation a three-dimensional case makes) and the refresh hook
    are created on first use; the inverse is fitted on the first :meth:`build`, at that build's
    assembler, state and ``build_beta``, and every later build glues that same object in. The callable
    handed to the march, :attr:`refresh_preconditioner`, is created when the session is opened, so every
    march served by it calls the identical object.

    **An inverse whose blocks all offer a traced cycle is applied on device** -- a field split of two
    such blocks, or one such block over the whole state: the step carries its
    :class:`~aquaflux.precondition.traced_cycle.TracedForm` as data through a
    :class:`~aquaflux.precondition.monolithic_policy.TracedFormShiftPolicy`, so a Krylov iteration emits no
    host callback. The host inverse is still what is fitted and re-fitted; after each re-fit the hook
    hands the march a step carrying the traced form of the re-fitted inverse
    (:func:`~aquaflux.precondition.monolithic_policy.with_traced_form`). The traced form is taken once
    per fit and shared by every step the session builds, the stopping steps included: each is a copy of
    the whole hierarchy in the inverse's precision, so a copy per step would keep one alive per step
    held. Swapping in a re-fitted form is a compilation-cache hit only while the hierarchies keep their
    array shapes across a re-fit, so a block whose coarsening reads
    operator values must freeze it (``frozen_coarsening``) or pour it into a fixed ladder
    (``shape_headroom``); otherwise every re-fit recompiles the step. Any other inverse is applied
    through a host callback and re-fitted in place.

    **A re-pointed session re-fits the inverse and leaves the shift alone.** :meth:`rebind` points the
    session at another companion (a station of a viscosity ramp, say) and forces the next re-fit, which
    is fitted to the new companion's Jacobian plus the shift the march's step already carries. That
    shift's diagonal stays the one the step was built with: a step control adapts the shift strength on
    the premise that the diagonal it scales is fixed, and swapping the diagonal under it at each station
    changes the damping by a different amount on every row, which the control can neither see nor undo.
    The shift vanishes at the root, so which fixed diagonal is held changes the path and not the answer.
    What must be the new problem's is how the march is *judged*: :meth:`stopping_step` gives the step a
    march's stopping test reads its measure from, built on the problem the march is meant to solve.

    Parameters
    ----------
    spec : MaterializedJacobian
        Which inverse, how it is probed, the shift its first build is fitted at and its refit floor.
    problem : MaterializedProblem
        The residual the session serves.
    observer : callable, optional
        ``(timing: RefreshTiming) -> None``, told what each refresh did and what it cost.
    reports : dict, optional
        Where each field-split block inverse sends its build record, keyed ``"leading"`` /
        ``"trailing"``.
    on_build : callable, optional
        ``step -> step``, applied to every step the session builds, for instrumenting a driver.
    precondition_wrapper : callable, optional
        ``hook -> hook``, wrapping the per-step :attr:`refresh_preconditioner` once.
    inverse_wrapper : callable, optional
        ``(role, factory) -> factory``, wrapping a field-split block-inverse factory before it is used.
    memory_trim : MemoryTrim or None, optional
        Handed to the refresh hook, which calls it before each re-fit (:class:`BetaTrackingRefresh`).
        The default is a :class:`~aquaflux.allocator.GlibcTrim`, which gives glibc's kept, freed memory
        back to the operating system and does nothing without glibc; ``None`` leaves the allocator alone.

    Raises
    ------
    TypeError
        If the spec asks for a field split and the problem has no fields to split.
    """

    def __init__(
        self,
        spec: MaterializedJacobian,
        problem: MaterializedProblem,
        *,
        observer: Callable[[RefreshTiming], None] | None = None,
        reports: dict[str, Callable[[str], None]] | None = None,
        on_build: Callable[[NewtonStrategy], NewtonStrategy] | None = None,
        precondition_wrapper: Callable[[Callable], Callable] | None = None,
        inverse_wrapper: Callable[[str, Callable], Callable] | None = None,
        memory_trim: MemoryTrim | None = _RETURN_FREED_MEMORY,
    ) -> None:
        if isinstance(spec.inverse, FieldSplit) and problem.groups() is None:
            raise TypeError(
                "a FieldSplit inverse needs a leading and a trailing group of fields, and this problem "
                "has a single group -- there is nothing to split. Use a block inverse such as "
                "SimpleSmoothed() over the whole state."
            )
        if isinstance(spec.inverse, BlockInverse) and problem.groups() is not None:
            raise TypeError(
                "a bare block inverse is fitted to the WHOLE state, which is only meaningful when the "
                "fields form a single group (a laminar flow); this problem has a leading and a trailing "
                "group. Wrap block inverses in FieldSplit(leading=..., trailing=...)."
            )
        if spec.pivot_floor is not None and not problem.floorable_fields:
            raise ValueError(
                f"a pivot floor changes only the rows of the fields a problem names as ones whose pivot can "
                f"pass through zero, and {type(problem).__name__} names none, so it would change nothing. "
                "Leave pivot_floor unset for this problem."
            )
        self._spec = spec
        self._problem = problem
        self._observer = observer
        self._reports = dict(reports or {})
        self._on_build = on_build
        self._inverse_wrapper = inverse_wrapper
        self._memory_trim = memory_trim
        self._probe: JacobianProbe | None = None
        self._hook: BetaTrackingRefresh | None = None
        self._preconditioner: object | None = None
        # The traced form of the fitted inverse, or `None` when it is applied on the host; taken once per
        # fit and carried by every step built from it. Each traced form is a copy of the whole hierarchy
        # in the inverse's precision, so a step holding its own would keep one such copy alive per step
        # built, for as long as that step is.
        self._form: TracedForm | None = None
        # What the last build was made at and with, for building a step on another problem alike.
        self._reference_state: jnp.ndarray | None = None
        self._march: dict = {}
        # The steps built on other problems for `stopping_step`, keyed by the problem's identity and
        # dropped whenever the session builds again, since they are made at the last build's state.
        self._stopping_steps: dict[int, NewtonStrategy] = {}

        def refresh_preconditioner(
            active_step: NewtonStrategy, state: jnp.ndarray, *, due: bool = False
        ) -> Callable[[NewtonStrategy], NewtonStrategy] | None:
            refitted = self._refresh_hook()(active_step, state, due=due)
            if not (refitted and self._form is not None):
                return None
            superseded, self._form = self._form, None
            self._form = form = _traced_form(self._preconditioner.inverse, self._precision)
            # The stopping steps share it too: they are read for their measure alone, and otherwise
            # each would keep the fit it was built at alive for the rest of the build.
            self._stopping_steps = {
                key: with_traced_form(step, form) for key, step in self._stopping_steps.items()
            }
            _release_superseded(superseded, form, self._preconditioner.inverse)
            return lambda step: with_traced_form(step, form)

        self.refresh_preconditioner = (
            refresh_preconditioner
            if precondition_wrapper is None
            else precondition_wrapper(refresh_preconditioner)
        )

    @property
    def problem(self) -> MaterializedProblem:
        """The residual the session currently serves (it changes on :meth:`rebind`)."""
        return self._problem

    @property
    def preconditioner(self) -> object | None:
        """The fitted host preconditioner the session re-fits, or ``None`` before the first build.

        What a step carries may be a traced form derived from it (see the class summary), so this is the
        one place to reach the host object itself -- its ``inverse``, or its callback ``matvec()``.
        """
        return self._preconditioner

    def build(self, state: jnp.ndarray, **march: object) -> NewtonStrategy:
        return self._build(state, march)

    def stopping_step(self, target: object) -> Callable[[NewtonStrategy], NewtonStrategy]:
        """The map from a step the march holds to the step its stopping test is measured with.

        That step is the held one with ``target``'s shift policy in place of its own
        (:meth:`~aquaflux.solve.ShiftedStep.preconditioned_as`). A row-scaled measure takes its row
        scales from the step's shift policy, so the march is then judged in ``target``'s scales while it
        is still stepped with the shift it holds (see the class summary).

        The step on ``target`` is built once, on first use after a build, at the state and with the march
        keywords of that build, so only the problem differs from the steps in use: a shift policy is
        frozen at a state, and building it at the march's iterate would move everything it freezes
        there as well as the problem.

        Parameters
        ----------
        target : object
            An assembler of the same case -- in practice the case itself, after a continuation's
            stations.

        Returns
        -------
        callable
            ``held -> step``, for the march's stopping test only; the march never steps with it.
        """

        def stopping(held: NewtonStrategy) -> NewtonStrategy:
            key = id(target)
            if key not in self._stopping_steps:
                if self._reference_state is None:
                    raise RuntimeError("stopping_step was used before the session built a step")
                problem = self._problem.with_assembler(target)
                self._stopping_steps[key] = self._step(self._reference_state, self._march, problem)
            return held.preconditioned_as(self._stopping_steps[key])

        return stopping

    def refresh(
        self, state: jnp.ndarray, previous: NewtonStrategy, **march: object
    ) -> NewtonStrategy:
        del previous  # the shared inverse is re-fitted in place, not re-derived from the old step
        step = self._build(state, march)
        if self._hook is not None:
            # The standing inverse was fitted before the march moved; force the next refresh to be full.
            self._hook.rebind(self._problem.assembler)
        return step

    def rebind(self, assembler: object) -> None:
        current = self._problem.layout
        rebound = self._problem.with_assembler(assembler)
        if rebound.layout.size != current.size or rebound.layout.n_fields != current.n_fields:
            raise ValueError(
                "a session can only be re-pointed at another companion of the SAME case: its colouring "
                f"probe was built for a {current.n_fields}-field state of size {current.size}, and this "
                f"assembler has {rebound.layout.n_fields} fields and size {rebound.layout.size}."
            )
        self._problem = rebound
        if self._hook is not None:
            self._hook.rebind(assembler)

    def _build(self, state: jnp.ndarray, march: dict) -> NewtonStrategy:
        """The step at ``state``, instrumented by ``on_build`` as every step handed to a march is."""
        self._reference_state, self._march, self._stopping_steps = state, march, {}
        step = self._step(state, march, self._problem)
        return step if self._on_build is None else self._on_build(step)

    def _step(
        self, state: jnp.ndarray, march: dict, problem: MaterializedProblem
    ) -> NewtonStrategy:
        """Fit the inverse (once) and assemble a step on ``problem`` around it."""
        bound = problem.bind_march(march)
        if _is_traced((problem.assembler, state)):
            raise ValueError(
                "a materialized-Jacobian preconditioner is assembled off the jit path from concrete "
                "arrays, so it cannot be built under jax.grad (or any JAX transform). Build the step "
                "with concrete parameters outside the transform and pass it as `strategy`; the "
                "adjoint reuses the same frozen inverse, so the gradient is unchanged."
            )
        base = problem.shift_source(state, bound)
        if self._preconditioner is None:
            self._preconditioner = self._fit(state, base)
            self._form = _traced_form(self._preconditioner.inverse, self._precision)
            if self._precision != "float64" and self._form is None:
                raise ValueError(
                    f"FieldSplit(precision={self._precision!r}) needs a split applied on the device, "
                    "and this one has a block with no traced cycle, so it is applied on the host in "
                    "double precision."
                )
        return problem.build_step(state, self._policy(base), bound, VCYCLE_LINEAR_SOLVE)

    def _policy(self, base) -> MonolithicFactorShiftPolicy | TracedFormShiftPolicy:
        """``base`` glued to the fitted inverse, in whichever form the step carries it."""
        if self._form is not None:
            return TracedFormShiftPolicy(base, self._form)
        return MonolithicFactorShiftPolicy(base, self._preconditioner)

    @property
    def _precision(self) -> str:
        """The precision the fitted split is applied in; a single block inverse is always double."""
        inverse = self._spec.inverse
        return inverse.precision if isinstance(inverse, FieldSplit) else "float64"

    def _probe_for(self) -> JacobianProbe:
        if self._probe is None:
            self._probe = self._problem.probe(
                self._spec.probe.settings(),
                # A probe feeding a split stores neither the triangle the split never reads nor the whole
                # matrix: its entries are laid out by the split's blocks.
                self._problem.groups() if isinstance(self._spec.inverse, FieldSplit) else None,
            )
        return self._probe

    def _refresh_hook(self) -> BetaTrackingRefresh:
        if self._hook is None:
            if self._preconditioner is None:
                raise RuntimeError(
                    "the refresh hook was called before the session built a step: there is no "
                    "preconditioner to re-fit yet."
                )
            refit_beta_floor = self._spec.refit_beta_floor
            self._hook = BetaTrackingRefresh(
                self._problem.assembler,
                self._probe_for(),
                self._preconditioner,
                observer=self._observer,
                pivot_floor=self._spec.pivot_floor,
                floorable_rows=self._floorable_row_mask(),
                by_field=self._by_field,
                memory_trim=self._memory_trim,
                **({} if refit_beta_floor is None else {"refit_beta_floor": refit_beta_floor}),
            )
        return self._hook

    @property
    def _by_field(self) -> FieldLayout | None:
        """The layout to probe one column field at a time over, or ``None`` for whole-state seeds."""
        return self._problem.layout if self._problem.probes_by_field else None

    def _floorable_row_mask(self) -> np.ndarray | None:
        """Which rows the pivot floor may change, shape ``(n_dofs,)``, or ``None`` without a floor."""
        if self._spec.pivot_floor is None:
            return None
        layout = self._problem.layout
        rows = np.zeros(layout.size, dtype=bool)
        for name in self._problem.floorable_fields:
            rows[layout.slice_of(name)] = True
        return rows

    def _fit(self, state: jnp.ndarray, base) -> object:
        probe = self._probe_for()
        probed = probe.narrow(self._problem.assembler)
        frozen = jax.lax.stop_gradient(state)
        matvec, batched_matvec = _probe_products(probed, frozen, self._by_field)
        build_beta = BUILD_BETA if self._spec.build_beta is None else self._spec.build_beta
        shift = frozen_shift_diagonal(base, build_beta, state)
        inverse = self._spec.inverse
        probing = {
            "batched_matvec": batched_matvec,
            "probe_batch_size": PROBE_BATCH_SIZE,
            "structure": probe.structure,
        }
        if isinstance(inverse, BlockInverse):
            return MaterializedBlockPreconditioner.build(
                matvec,
                probe.plan,
                shift,
                inverse=inverse,
                n_fields=self._problem.layout.n_fields,
                **probing,
            )
        return FieldSplitPreconditioner.build(
            matvec,
            probe.plan,
            shift,
            self._problem.groups(),
            leading_inverse=self._block_inverse("leading"),
            trailing_inverse=self._block_inverse("trailing"),
            # Held as the traced split applies it, like the blocks' hierarchies.
            stored_precision=self._precision,
            **probing,
        )

    def _block_inverse(self, role: str) -> Callable:
        spec: BlockInverse = getattr(self._spec.inverse, role)
        # A block applied in single precision is held in it too, where its family can be: otherwise a
        # double copy of its hierarchy would sit beside the single one the step applies, unread.
        stored = self._precision if spec._stores_precision else "float64"
        factory = spec.bound(report=self._reports.get(role), stored_precision=stored)
        return factory if self._inverse_wrapper is None else self._inverse_wrapper(role, factory)
