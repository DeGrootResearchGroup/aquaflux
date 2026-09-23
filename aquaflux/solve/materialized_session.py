"""A materialized-Jacobian preconditioner, kept current across every step of a march.

A complete LU, a multigrid V-cycle or a field split preconditions a shifted solve by inverting the
**assembled** Jacobian, recovered off the jit path by coloured probing. It is built at one state and one
shift and goes stale as the march moves both, so a march that uses one must also re-fit it -- every step
for a cheap exact factorization, on evidence for an expensive multigrid -- and must hand the *same
objects* to every step it builds, because the inverse and the refresh hooks ride in static fields of the
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

from .amg_preconditioner import MaterializedJacobianPreconditioner, MonolithicAmgPreconditioner
from .block_inverse import BlockInverse
from .block_preconditioner import MaterializedBlockPreconditioner
from .field_split import FieldGroups, FieldSplitAmgPreconditioner
from .jacobian_probe import JacobianProbe
from .lu_preconditioner import MonolithicLuPreconditioner
from .materialized_spec import CompleteLu, FieldSplit, MaterializedJacobian, MonolithicVCycle
from .refresh_timing import RefreshTiming
from .shifted_step import LinearSolveRegime
from .state import FieldLayout
from .strategy import NewtonStrategy

__all__ = [
    "BUILD_BETA",
    "FACTORIZATION_LINEAR_SOLVE",
    "PROBE_BATCH_SIZE",
    "VCYCLE_LINEAR_SOLVE",
    "MaterializedProblem",
    "MaterializedSession",
    "PreconditionerSession",
    "batched_jacobian_matvec",
    "beta_tracking_refresh",
    "frozen_shift_diagonal",
    "jacobian_matvec",
]

#: How many coloured tangents share one vmapped jvp pass when materializing a Jacobian. Larger amortizes
#: dispatch over more probes; the coloured probes run in ``ceil(n_probes / this)`` fused passes instead of
#: an ``n_probes``-call Python loop.
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

#: A monolithic complete-LU factorization. It factors the whole saddle exactly, so the preconditioned
#: operator's spectrum collapses to a single point at the state and shift it was factored at -- the Krylov
#: solve stops within a handful of vectors there, and a large subspace is pure waste: with ``restart =
#: 120`` it would build ~120 matrix-vector products (each paying the factorization's triangular
#: back-solve) before it could stop. ``max_restarts`` is kept generous so a transiently harder (e.g.
#: drifted-reference) solve still completes before the next refactor.
FACTORIZATION_LINEAR_SOLVE = LinearSolveRegime(rtol=0.3, restart=10, max_restarts=40)

#: A monolithic multigrid V-cycle. Restart 15 is the measured sweet spot for a one-V-cycle
#: preconditioner: enough Arnoldi history for its convergence while checking the stop often enough not to
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

    The same directional derivative as :func:`jacobian_matvec` applied to each row, so the responses are
    bit-identical to a per-tangent loop; running them as a few fused passes only amortizes dispatch. Takes
    the assembler as an argument for the same reason.

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


def frozen_shift_diagonal(base, beta: float, state: jnp.ndarray) -> np.ndarray:
    """The frozen pseudo-transient shift diagonal a factorization is built against, at ``state``.

    ``beta`` scales the base policy's shift diagonal; the ``stop_gradient`` keeps the frozen factorization
    off the differentiation path. Shared by the initial build and every in-place refresh, for the
    complete-LU and multigrid preconditioners alike.

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
    return np.asarray(jax.lax.stop_gradient(base.shift_term(state).shift(beta)))


def _is_traced(pytree: object) -> bool:
    """Whether ``pytree`` holds a JAX tracer.

    A materialized preconditioner is assembled off the jit path from concrete arrays, so a tracer leaf
    means the caller has wrapped the solve in ``jax.grad`` / ``jvp`` / ``vmap``.
    """
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(pytree))


class PreconditionerSession(Protocol):
    """One preconditioner, kept current across every step it serves.

    A session is what a march holds on to between its steps: the frozen inverse, the colouring probe it was
    materialized with, and the per-step refresh hook. Those must outlive a single Newton step -- a
    continuation builds a step per rung, a refresh builds one per segment -- and must be the *same
    objects* each time, because the inverse and the hooks ride in static fields of the step and a new
    object recompiles the whole solve.

    Attributes
    ----------
    refresh_preconditioner : callable or None
        ``(step, state) -> None``, called by the march before every step to re-fit the inverse at that
        step's shift; ``None`` for a family with nothing to re-fit.
    """

    refresh_preconditioner: Callable[[NewtonStrategy, jnp.ndarray], None] | None

    def build(self, state: jnp.ndarray, **march: object) -> NewtonStrategy:
        """The Newton step at ``state``, configured by the march keywords."""
        ...

    def refresh(
        self, state: jnp.ndarray, previous: NewtonStrategy, **march: object
    ) -> NewtonStrategy:
        """Re-freeze at the developed ``state``; ``previous`` is the step it replaces."""
        ...

    def rebind(self, assembler: object) -> None:
        """Point the session at another companion of the same case, such as the next continuation rung."""
        ...


def beta_tracking_refresh(
    assembler: object,
    probe: JacobianProbe,
    *,
    every_step: bool,
    refit_beta_floor: float = 0.0,
    observer: Callable[[RefreshTiming], None] | None = None,
) -> Callable[[NewtonStrategy, jnp.ndarray], None]:
    """The ``refresh_preconditioner`` hook that re-fits a monolithic inverse as the shift strength moves.

    Returns a ``refresh_preconditioner(active_step, state)`` that reads ``β`` from the step's
    :class:`~aquaflux.solve.ConstantRelaxation` schedule and re-factors the step's
    :class:`~aquaflux.solve.MonolithicFactorShiftPolicy` preconditioner in place at
    ``J(state) + β·d(state)``. With ``every_step`` it does so on every step (the cheap exact-LU cadence);
    without, only on its first call and after each ``rebind`` -- a multigrid re-materialize is too
    expensive to pay every step, so between those the rebuild is left to the dual-time loop's cost
    trigger, through ``refresh_at``.

    Parameters
    ----------
    assembler : object
        The residual assembler (supplies the Jacobian-vector product); a pytree with ``residual``.
    probe : JacobianProbe
        The shared colouring plan, de-compression map and assembler stand-in.
    every_step : bool
        Re-factor on every step (``True``), or only on the first call and after each ``rebind``
        (``False``).
    refit_beta_floor : float
        A lower bound on the shift strength the **preconditioner** is refreshed at: it is built at
        ``max(beta, refit_beta_floor)`` while the march keeps solving at its own ``beta``. ``0.0``
        (default) tracks ``beta`` exactly. It is not the march's own shift floor
        (:attr:`~aquaflux.solve.Globalization.beta_floor`), which bounds the shift the solve runs at.
    observer : callable, optional
        ``(timing: RefreshTiming) -> None``, called on each refresh with which branch ran, its total
        seconds, and its per-phase costs. ``None`` (default) elides the call.

    Returns
    -------
    callable
        ``refresh_preconditioner(active_step, state) -> None``, carrying ``refresh_at`` (the inner-loop
        hook) and ``rebind`` (point it at another companion of the same case).
    """
    plan, structure = probe.plan, probe.structure

    # WHICH case this hook currently refreshes for, in a mutable binding rather than closed over, so
    # `rebind` can point it at another companion of the same case. A continuation solves a sequence of
    # companions that differ only in their molecular viscosity, and rebinding one hook lets them all share
    # ONE preconditioner -- which is what keeps the compiled step a cache hit across a rung boundary,
    # since the preconditioner rides in a static field compared by identity. Both probes below take the
    # assembler as an argument to a module-level jitted function, so swapping it changes no compilation
    # key of theirs either. `"probed"` is what the coloured probe differentiates, which is the assembler
    # itself unless the probe carries a stand-in (`JacobianProbe.narrow`); it is stored rather than
    # derived per call so a rebind narrows once.
    bound = {"assembler": assembler, "probed": probe.narrow(assembler)}

    # `frozen` a traced argument (not closed over) so the jvp-matvec compiles once and every refactor
    # reuses it, rather than a fresh lambda recompiling each step.
    def matvec_at(frozen, v):
        return jacobian_matvec(bound["probed"], frozen, v)

    # Batched form (vmapped over the tangent) so the coloured probes of a full materialize run as a few
    # fused passes rather than a Python loop of separate calls. Used only by the AMG preconditioner's
    # `refresh_in_place`.
    def batched_matvec_at(frozen, seeds):
        return batched_jacobian_matvec(bound["probed"], frozen, seeds)

    # Pending on the first call -- the build froze the preconditioner at its own shift, not the march's --
    # and again after `rebind`, since the standing preconditioner then describes the PREVIOUS companion.
    forced_full = {"pending": True}

    def _report_refresh(
        kind: str, started: float, phases: tuple[tuple[str, float], ...] | None = None
    ) -> None:
        """Tell an injected observer which branch ran and what each part of it cost.

        The total alone cannot be acted on: a refresh dominated by the coloured jvp probe and one
        dominated by the multigrid setup take the same wall time and call for opposite fixes.
        """
        if observer is not None:
            observer(RefreshTiming(kind, time.perf_counter() - started, tuple(phases or ())))

    # Which step the inner-loop hook is refreshing, kept current by `refresh_preconditioner` below.
    bound_step: dict[str, NewtonStrategy] = {}

    def refresh_preconditioner(active_step: NewtonStrategy, state: jnp.ndarray) -> None:
        # The march calls this immediately before every step and again on every retry, always with the
        # CURRENT step -- so this is also where the inner-loop hook learns which step it is refreshing.
        # Binding once at construction cannot work: the step the builder returns still carries the
        # default schedule, and the march replaces it each iteration with one the control has set β on.
        bound_step["step"] = active_step
        schedule = active_step.relaxation_schedule
        beta = getattr(schedule, "beta", None)
        if beta is None:
            raise ValueError(
                "a β-tracking refresh needs the step's shift strength as a readable constant -- pair it "
                "with a DualTimeControl (which sets a ConstantRelaxation β), not the default "
                f"switched-evolution schedule ({type(schedule).__name__})."
            )
        beta = float(beta)
        started = time.perf_counter()
        # The preconditioner's shift is floored independently of the march's own beta. As beta -> 0 the
        # shift's diagonal dominance vanishes and the frozen V-cycle degrades, but the OPERATOR must keep
        # the small beta to make pseudo-transient progress. Flooring only the preconditioner's copy keeps
        # the V-cycle in a regime it inverts well while the solved system is untouched, so the converged
        # root and its adjoint are unchanged. The resulting mismatch SATURATES at `refit_beta_floor * d`
        # rather
        # than growing without bound the way a stale (never-refreshed) preconditioner's does.
        pc_beta = max(beta, refit_beta_floor)
        policy = active_step.shift_policy
        pc = policy.preconditioner
        if not (every_step or forced_full["pending"]):
            _report_refresh("none", started)
            return
        forced_full["pending"] = False
        frozen = jax.lax.stop_gradient(state)
        shift = frozen_shift_diagonal(policy.base, pc_beta, state)
        _report_refresh("full", started, _materialize_at(pc, frozen, shift))

    def _materialize_at(pc, frozen, shift) -> tuple[tuple[str, float], ...]:
        """Re-materialize the preconditioner at ``frozen`` with shift diagonal ``shift``.

        The AMG preconditioner materializes via the coloured probe and takes the batched form; the
        complete-LU preconditioner does not, so pass it only on the AMG path.
        """
        extra = (
            {
                "batched_matvec": lambda seeds: batched_matvec_at(frozen, seeds),
                "probe_batch_size": PROBE_BATCH_SIZE,
                "structure": structure,
            }
            if isinstance(pc, MaterializedJacobianPreconditioner)
            else {}
        )
        return pc.refresh_in_place(lambda v: matvec_at(frozen, v), plan, shift, **extra) or ()

    def refresh_at(iterate) -> None:
        """``inner_refresh`` hook: rebuild the preconditioner at this mid-step iterate.

        *When* to fire is decided by the dual-time loop (``DualTimeStep.refresh_on_cycles``), not here,
        so that the rule which triggers the refresh is the same one that forgives the abort it would
        otherwise be discarded by.

        The march's expensive inner solves are **stale-preconditioner** effects, not hard operators: at
        the hardest solve of a three-dimensional coupled march a preconditioner rebuilt at that very
        iterate converged in an order of magnitude fewer cycles than the march's own. Refreshing here --
        between inner iterations, after the line search and before the next solve -- keeps the step's
        progress, where the alternative reaction (abort the step and escalate β) discards both the work
        and the pseudo-timestep.

        Reacting is also what makes this worth doing as a *replacement* for a scheduled refresh rather
        than an addition to one: a fixed cadence pays on every step to protect the minority that needs
        it, and the right interval is regime-dependent in a way no fixed cadence can track (one step of
        staleness is nearly free at a large shift and dominates the solve at a small one).
        """
        if "step" not in bound_step:
            return
        started = time.perf_counter()
        step = bound_step["step"]
        beta = max(float(step.relaxation_schedule.beta), refit_beta_floor)
        frozen = jax.lax.stop_gradient(jnp.asarray(iterate))
        shift = frozen_shift_diagonal(step.shift_policy.base, beta, frozen)
        _report_refresh(
            "inner", started, _materialize_at(step.shift_policy.preconditioner, frozen, shift)
        )

    def rebind(companion) -> None:
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
        bound["assembler"] = companion
        bound["probed"] = probe.narrow(companion)
        forced_full["pending"] = True

    refresh_preconditioner.refresh_at = refresh_at
    refresh_preconditioner.rebind = rebind
    return refresh_preconditioner


class MaterializedProblem(abc.ABC):
    """What a :class:`MaterializedSession` needs from a residual, and nothing more.

    A residual supplies the assembler whose Jacobian is materialized and four decisions that are its own:
    how its graph is probed, which base shift policy the inverse is fitted against, how its fields are cut
    for a field split, and how its Newton step is finally assembled around the fitted inverse. Everything
    else -- the probe's lifecycle, the inverse family, the refresh hook, the rebind -- is the session's.

    Implementations are immutable values: :meth:`with_assembler` returns a new problem rather than
    mutating one, so a session can rebind across continuation rungs without aliasing.
    """

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
    def probe(self, settings: dict, active_rows: np.ndarray | None) -> JacobianProbe:
        """The coloured probe for this problem's graph.

        Parameters
        ----------
        settings : dict
            The spec's set probe fields (``stencil_reach``, ``column_reach``, ``gradient_sweeps``).
        active_rows : np.ndarray or None
            Field-pair blocks a split never reads, or ``None`` to want every block.
        """

    @abc.abstractmethod
    def groups(self) -> FieldGroups | None:
        """How the state's fields are cut into a leading and a trailing group, or ``None``.

        ``None`` means there is nothing to split (a single group of fields), and a
        :class:`~aquaflux.solve.FieldSplit` inverse is refused for the problem.
        """

    @abc.abstractmethod
    def bind_march(self, march: dict) -> dict:
        """The march keywords validated and completed with their defaults.

        The result **must** carry ``dual_time`` (a :class:`~aquaflux.solve.DualTimeLoop` or ``None``) and
        ``inner_refresh`` (a callable or ``None``), which the session reads and may set; everything else
        is the problem's to interpret in :meth:`shift_source` and :meth:`build_step`.
        """

    @abc.abstractmethod
    def shift_source(self, state: jnp.ndarray, march: dict):
        """The base shift policy the inverse is fitted against, at ``state``."""

    @abc.abstractmethod
    def build_step(
        self,
        state: jnp.ndarray,
        base,
        preconditioner: object,
        march: dict,
        base_regime: LinearSolveRegime,
    ) -> NewtonStrategy:
        """The Newton step: ``base`` glued to the fitted ``preconditioner``, configured by ``march``.

        ``base_regime`` is the Krylov regime of the inverse family; the march's ``linear_solve`` moves it.
        """


class MaterializedSession:
    """The materialized-Jacobian family's session: one probe, one inverse and one refresh hook.

    Everything expensive or identity-bearing is created at most once. The probe (a colouring plan and
    its de-compression map, the largest allocation a three-dimensional case makes) and the refresh hook
    are created on first use; the inverse is fitted on the first :meth:`build`, at that build's
    assembler, state and ``build_beta``, and every later build glues that same object in. The two
    callables handed to the march -- :attr:`refresh_preconditioner` and the mid-step refresh -- are
    created when the session is opened, so every step built from it carries the identical objects.

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
    ) -> None:
        if isinstance(spec.inverse, FieldSplit) and problem.groups() is None:
            raise TypeError(
                "a FieldSplit inverse needs a leading and a trailing group of fields, and this problem "
                "has a single group -- there is nothing to split. Use a block inverse such as "
                "SimpleSmoothed() over the whole state, or CompleteLu() / MonolithicVCycle()."
            )
        if isinstance(spec.inverse, BlockInverse) and problem.groups() is not None:
            raise TypeError(
                "a bare block inverse is fitted to the WHOLE state, which is only meaningful when the "
                "fields form a single group (a laminar flow); this problem has a leading and a trailing "
                "group. Wrap block inverses in FieldSplit(leading=..., trailing=...), or use CompleteLu() "
                "/ MonolithicVCycle()."
            )
        self._spec = spec
        self._problem = problem
        self._observer = observer
        self._reports = dict(reports or {})
        self._on_build = on_build
        self._inverse_wrapper = inverse_wrapper
        self._probe: JacobianProbe | None = None
        self._hook: Callable | None = None
        self._preconditioner: object | None = None

        def refresh_preconditioner(active_step: NewtonStrategy, state: jnp.ndarray) -> None:
            self._refresh_hook()(active_step, state)

        def refresh_at(iterate: jnp.ndarray) -> None:
            self._refresh_hook().refresh_at(iterate)

        self._refresh_at = refresh_at
        self.refresh_preconditioner = (
            refresh_preconditioner
            if precondition_wrapper is None
            else precondition_wrapper(refresh_preconditioner)
        )

    @property
    def problem(self) -> MaterializedProblem:
        """The residual the session currently serves (it changes on :meth:`rebind`)."""
        return self._problem

    def build(self, state: jnp.ndarray, **march: object) -> NewtonStrategy:
        return self._build(state, march, track=True)

    def refresh(
        self, state: jnp.ndarray, previous: NewtonStrategy, **march: object
    ) -> NewtonStrategy:
        del previous  # the shared inverse is re-fitted in place, not re-derived from the old step
        step = self._build(state, march, track=True)
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

    def _build(self, state: jnp.ndarray, march: dict, *, track: bool) -> NewtonStrategy:
        """Fit (once) and assemble; ``track`` wires the mid-step refresh, which a frozen step never has."""
        problem = self._problem
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
        dual_time = bound["dual_time"]
        if (
            track
            and dual_time is not None
            and dual_time.refresh_on_cycles is not None
            and bound["inner_refresh"] is None
        ):
            bound["inner_refresh"] = self._refresh_at
        step = problem.build_step(
            state,
            base,
            self._preconditioner,
            bound,
            FACTORIZATION_LINEAR_SOLVE
            if isinstance(self._spec.inverse, CompleteLu)
            else VCYCLE_LINEAR_SOLVE,
        )
        return step if self._on_build is None else self._on_build(step)

    def _probe_for(self) -> JacobianProbe:
        if self._probe is None:
            self._probe = self._problem.probe(
                self._spec.probe.settings(),
                # A split never reads the leading-by-trailing triangle, so its probe need not store it.
                self._problem.groups().active_rows()
                if isinstance(self._spec.inverse, FieldSplit)
                else None,
            )
        return self._probe

    def _refresh_hook(self) -> Callable:
        if self._hook is None:
            refit_beta_floor = self._spec.refit_beta_floor
            self._hook = beta_tracking_refresh(
                self._problem.assembler,
                self._probe_for(),
                every_step=isinstance(self._spec.inverse, CompleteLu),
                observer=self._observer,
                **({} if refit_beta_floor is None else {"refit_beta_floor": refit_beta_floor}),
            )
        return self._hook

    def _fit(self, state: jnp.ndarray, base) -> object:
        probe = self._probe_for()
        probed = probe.narrow(self._problem.assembler)
        frozen = jax.lax.stop_gradient(state)

        def matvec(v):
            return jacobian_matvec(probed, frozen, v)

        build_beta = BUILD_BETA if self._spec.build_beta is None else self._spec.build_beta
        shift = frozen_shift_diagonal(base, build_beta, state)
        inverse = self._spec.inverse
        if isinstance(inverse, CompleteLu):
            return MonolithicLuPreconditioner.build(matvec, probe.plan, shift, **inverse.settings())

        def batched_matvec(seeds):
            return batched_jacobian_matvec(probed, frozen, seeds)

        probing = {
            "batched_matvec": batched_matvec,
            "probe_batch_size": PROBE_BATCH_SIZE,
            "structure": probe.structure,
        }
        if isinstance(inverse, MonolithicVCycle):
            return MonolithicAmgPreconditioner.build(
                matvec, probe.plan, shift, **inverse.settings(), **probing
            )
        if isinstance(inverse, BlockInverse):
            return MaterializedBlockPreconditioner.build(
                matvec,
                probe.plan,
                shift,
                inverse=inverse,
                n_fields=self._problem.layout.n_fields,
                **probing,
            )
        return FieldSplitAmgPreconditioner.build(
            matvec,
            probe.plan,
            shift,
            self._problem.groups(),
            leading_inverse=self._block_inverse("leading"),
            trailing_inverse=self._block_inverse("trailing"),
            **probing,
        )

    def _block_inverse(self, role: str) -> Callable:
        spec: BlockInverse = getattr(self._spec.inverse, role)
        factory = spec.bound(report=self._reports[role]) if role in self._reports else spec
        return factory if self._inverse_wrapper is None else self._inverse_wrapper(role, factory)
