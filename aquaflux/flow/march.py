"""The staged flow march: the coupled ``(u, p)`` solve on the same robust machinery as any other march.

:func:`~aquaflux.flow.momentum_continuation` builds a bare pseudo-transient step for
:class:`~aquaflux.solve.RootSolver`. That is enough for a well-behaved laminar case, and it offers
nothing when the case is not: no dual-time inner loop, no retry policy, no row-equilibrated residual
measure, no preconditioner refresh, no step control. :func:`solve_flow_march` runs the flow residual on
the residual-agnostic staged driver (:func:`~aquaflux.solve.staged_march`) that every coupled solve
uses, so a laminar problem is *configured* like a turbulent one rather than solved by a weaker path --
which is what makes it usable as a control when a solver question is being asked.

What is flow-specific here is small: the shift policy (the velocity ``a_P`` shift and its matching
block-SIMPLE preconditioner), the measures of the flow residual, and the potential-flow starting state.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Mapping

import jax.numpy as jnp
import lineax as lx
import numpy as np

from aquaflux.schemes import narrow_gradient_sweeps
from aquaflux.solve import (
    DEFAULT_GLOBALIZATION,
    NO_REFRESH,
    NO_RETRIES,
    Convergence,
    DualTimeLoop,
    FieldGroups,
    Globalization,
    JacobianProbe,
    LinearSolveRegime,
    LinearSolveSettings,
    MaterializedJacobian,
    MaterializedProblem,
    MaterializedSession,
    MonolithicFactorShiftPolicy,
    NewtonStrategy,
    PreconditionerSession,
    RefreshPolicy,
    ResidualHomotopy,
    RetryPolicy,
    RowScaled,
    SessionSource,
    ShiftBasis,
    StepControl,
    StepReport,
    assembler_residual,
    explicit_source,
    refuse_a_transform_the_march_cannot_run_in,
    resolve_linear_solve,
    root_adjoint,
    shifted_step,
    staged_march,
    stop_array_gradients,
)

from .continuation import momentum_shift_only_policy, momentum_shift_policy
from .initialization import potential_flow
from .measures import FlowMeasures
from .momentum import MomentumContinuity

__all__ = ["flow_march_step", "open_flow_session", "solve_flow_march"]

#: The flow march's default stopping test: row-scaled, because the Euclidean norm of a ``(u, p)`` residual
#: is dominated by whichever block is largest and judges nothing else.
_FLOW_CONVERGENCE = Convergence(measure=RowScaled(), rtol=1e-10, atol=1e-12)

#: The Krylov regime of the shifted forward solve. This is the block-diagonal coupled family's
#: calibration, carried across because the stop is a property of the measure the march judges by and the
#: preconditioner is the same block-SIMPLE one; it has **not** been re-measured on a flow-only residual.
_FLOW_LINEAR_SOLVE = LinearSolveRegime(rtol=0.3, restart=120, max_restarts=15)


def flow_march_step(
    momentum: MomentumContinuity,
    reference_state: jnp.ndarray,
    *,
    preconditioner_options: Mapping[str, object] | None = None,
    globalization: Globalization = DEFAULT_GLOBALIZATION,
    dual_time: DualTimeLoop | None = None,
    linear_solve: LinearSolveSettings | lx.AbstractLinearSolver | None = None,
    shift_basis: ShiftBasis | None = None,
    inner_observer: Callable[..., None] | None = None,
    inner_refresh: Callable[[jnp.ndarray], None] | None = None,
    jacobian_gradient_sweeps: int | None = None,
) -> NewtonStrategy:
    """The flow march's Newton step, with its preconditioner **frozen** at ``reference_state``.

    Parameters
    ----------
    momentum : MomentumContinuity
        The coupled flow residual assembler.
    reference_state : jnp.ndarray
        The flow state the block preconditioner is frozen at, shape ``((dim + 1) n_cells,)``.
    preconditioner_options : mapping, optional
        The block-SIMPLE preconditioner's settings -- the keywords of
        :meth:`~aquaflux.flow.BlockPreconditioner.build` (``velocity``, ``schur_scaling``,
        ``composition``, ...); ``reference_state`` is this function's own. An unknown name raises.
    globalization : Globalization
        How hard the march damps and what it does when a step misbehaves. Only the fields it sets are
        applied; an unset ``line_search`` takes the step class's own default, the full shifted step,
        since on the flow residual the shift is the globalization. Beside a ``dual_time`` loop the
        escalation-ladder fields are refused.
    dual_time : DualTimeLoop or None
        Given, the march is dual-time (backward-Euler); ``None`` is the single shifted step.
    linear_solve : LinearSolveSettings, lineax.AbstractLinearSolver or None
        The shifted solve: a regime whose unset fields take ``rtol 0.3, restart 120, max_restarts 15``
        (measured in the march's progress measure, not the Euclidean norm), or a whole solver, which
        replaces the regime **and** the stopping measure.
    shift_basis : ShiftBasis, optional
        How the velocity shift diagonal is built from the momentum diagonal's parts (see
        :class:`~aquaflux.flow.MomentumShiftPolicy`).
    inner_observer, inner_refresh : callable, optional
        The dual-time loop's per-inner-iteration hook and mid-step rebuild. Forward-only, and refused
        without ``dual_time``.
    jacobian_gradient_sweeps : int or None
        Cap the gradient reconstruction's sweeps in the copy of the residual the **forward Jacobian** is
        differentiated from, leaving the residual itself -- and so the root and the adjoint -- untouched.
        ``None`` (default) differentiates the residual as it stands. It changes how fast the inexact-Newton
        iteration reaches the root, never which root; see
        :func:`~aquaflux.schemes.narrow_gradient_sweeps`.

    Returns
    -------
    NewtonStrategy
        A :class:`~aquaflux.solve.DualTimeStep` when ``dual_time`` is given, else a
        :class:`~aquaflux.solve.PseudoTransientStep`.
    """
    policy = momentum_shift_policy(
        momentum,
        shift_basis,
        reference_state=reference_state,
        **dict(preconditioner_options or {}),
    )
    return _flow_shifted_step(
        momentum,
        policy,
        policy.preconditioner.factory(),
        base_regime=_FLOW_LINEAR_SOLVE,
        globalization=globalization,
        dual_time=dual_time,
        linear_solve=linear_solve,
        inner_observer=inner_observer,
        inner_refresh=inner_refresh,
        jacobian_gradient_sweeps=jacobian_gradient_sweeps,
    )


def _flow_shifted_step(
    momentum: MomentumContinuity,
    policy: object,
    adjoint_preconditioner_factory: Callable,
    *,
    base_regime: LinearSolveRegime,
    globalization: Globalization,
    dual_time: DualTimeLoop | None,
    linear_solve: LinearSolveSettings | lx.AbstractLinearSolver | None,
    inner_observer: Callable[..., None] | None,
    inner_refresh: Callable[[jnp.ndarray], None] | None,
    jacobian_gradient_sweeps: int | None,
) -> NewtonStrategy:
    """The flow's step around a composed policy: what every flow preconditioner shares.

    The block-SIMPLE step and the materialized-Jacobian step differ in the policy, the adjoint
    preconditioner and the family's default Krylov regime; the regime override, the Jacobian stand-in
    and the step tail are this function's.
    """
    regime, krylov_solver = resolve_linear_solve(linear_solve, base_regime)
    return shifted_step(
        policy,
        globalization=globalization,
        dual_time=dual_time,
        regime=regime,
        krylov_solver=krylov_solver,
        adjoint_preconditioner_factory=adjoint_preconditioner_factory,
        inner_observer=inner_observer,
        inner_refresh=inner_refresh,
        jacobian_residual=(
            None
            if jacobian_gradient_sweeps is None
            else narrow_gradient_sweeps(momentum, jacobian_gradient_sweeps).residual
        ),
    )


@dataclasses.dataclass(frozen=True)
class _GradientSweepCap:
    """The assembler stand-in a flow Jacobian probe differentiates: gradient sweeps capped, as a value."""

    sweeps: int

    def __call__(self, momentum: MomentumContinuity) -> MomentumContinuity:
        return narrow_gradient_sweeps(momentum, self.sweeps)


#: The march settings a flow session builds a step from: the keywords of :func:`flow_march_step`, less the
#: three that belong to the block-SIMPLE preconditioner it builds itself.
_SESSION_NOT_MARCH = ("momentum", "reference_state", "preconditioner_options")


@dataclasses.dataclass(frozen=True, eq=False)
class _FlowProblem(MaterializedProblem):
    """The laminar flow residual as a :class:`~aquaflux.solve.MaterializedProblem`.

    What a materialized-Jacobian session asks of a residual, answered for ``(u, p)``: the assembler, its
    mesh graph and layout, no field groups (a single group of fields has nothing to split), the
    shift-only momentum policy the inverse is fitted against, and the flow step around the fitted inverse.

    Attributes
    ----------
    momentum : MomentumContinuity
        The assembler the session builds on until it is re-pointed with ``rebind``.
    """

    momentum: MomentumContinuity

    @property
    def assembler(self) -> MomentumContinuity:
        return self.momentum

    @property
    def layout(self):
        return self.momentum.layout

    def with_assembler(self, assembler: MomentumContinuity) -> _FlowProblem:
        return dataclasses.replace(self, momentum=assembler)

    def probe(self, settings: dict, active_rows: np.ndarray | None) -> JacobianProbe:
        settings = dict(settings)
        sweeps = settings.pop("gradient_sweeps", None)
        return JacobianProbe.build(
            self.momentum.mesh.face_cells,
            self.momentum.mesh.n_cells,
            self.momentum.layout.n_fields,
            **settings,
            active_rows=active_rows,
            narrowing=None if sweeps is None else _GradientSweepCap(sweeps),
        )

    def groups(self) -> FieldGroups | None:
        return None

    def bind_march(self, march: dict) -> dict:
        signature = inspect.signature(flow_march_step)
        unknown = sorted(set(march) - (set(signature.parameters) - set(_SESSION_NOT_MARCH)))
        if unknown:
            raise TypeError(
                f"{unknown} are not march settings of a materialized-Jacobian flow march. A "
                "preconditioner setting belongs on the MaterializedJacobian spec; the block-SIMPLE "
                "`preconditioner_options` do not apply to it."
            )
        bound = signature.bind(None, None, **march)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        for name in _SESSION_NOT_MARCH:
            arguments.pop(name)
        return arguments

    def shift_source(self, state: jnp.ndarray, march: dict):
        return momentum_shift_only_policy(self.momentum, march["shift_basis"])

    def build_step(
        self,
        state: jnp.ndarray,
        base,
        preconditioner: object,
        march: dict,
        base_regime: LinearSolveRegime,
    ) -> NewtonStrategy:
        policy = MonolithicFactorShiftPolicy(base, preconditioner)
        return _flow_shifted_step(
            self.momentum,
            policy,
            policy.adjoint_factory(),
            base_regime=base_regime,
            globalization=march["globalization"],
            dual_time=march["dual_time"],
            linear_solve=march["linear_solve"],
            inner_observer=march["inner_observer"],
            inner_refresh=march["inner_refresh"],
            jacobian_gradient_sweeps=march["jacobian_gradient_sweeps"],
        )


def open_flow_session(
    preconditioner: MaterializedJacobian,
    momentum: MomentumContinuity,
    *,
    observer: Callable | None = None,
    on_build: Callable[[NewtonStrategy], NewtonStrategy] | None = None,
    precondition_wrapper: Callable[[Callable], Callable] | None = None,
) -> PreconditionerSession:
    """Open a materialized-Jacobian preconditioner session for a laminar flow, doing no array work yet.

    The flow counterpart of :func:`~aquaflux.turbulence.open_session`. Pass the session as
    ``solve_flow_march(preconditioner=session)`` to share one inverse and its refresh hook across several
    solves, such as the rungs of a viscosity continuation.

    Parameters
    ----------
    preconditioner : MaterializedJacobian
        Which inverse and how it is probed. A :class:`~aquaflux.solve.CompleteLu` or
        :class:`~aquaflux.solve.MonolithicVCycle`; a :class:`~aquaflux.solve.FieldSplit` is refused,
        because a ``(u, p)`` state is a single group of fields and has nothing to split.
    momentum : MomentumContinuity
        The assembler the session builds on until it is re-pointed with ``rebind``.
    observer : callable, optional
        ``(timing: RefreshTiming) -> None``, told what each refresh did and what it cost.
    on_build : callable, optional
        ``step -> step``, applied to every step the session builds, for instrumenting a driver.
    precondition_wrapper : callable, optional
        ``hook -> hook``, wrapping the per-step refresh hook once.

    Returns
    -------
    PreconditionerSession
        The session.

    Raises
    ------
    TypeError
        If ``preconditioner`` is not a :class:`~aquaflux.solve.MaterializedJacobian`, or asks for a field
        split.
    """
    if not isinstance(preconditioner, MaterializedJacobian):
        raise TypeError(
            "a flow session is opened for a MaterializedJacobian(...); the block-SIMPLE "
            "preconditioner is built by the march itself. Got "
            f"{type(preconditioner).__name__}."
        )
    return MaterializedSession(
        preconditioner,
        _FlowProblem(momentum),
        observer=observer,
        on_build=on_build,
        precondition_wrapper=precondition_wrapper,
    )


@dataclasses.dataclass(frozen=True)
class _FlowSource:
    """The flow march's continuation source: a step frozen at a state, re-frozen at a developed one.

    It brings no per-step refresh hook: the block-SIMPLE preconditioner is rebuilt between segments, not
    re-fitted every step.
    """

    momentum: MomentumContinuity
    reference_state: jnp.ndarray | None
    preconditioner_options: Mapping[str, object] | None
    march: dict

    refresh_preconditioner = None

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        reference = state if self.reference_state is None else self.reference_state
        return self._step(reference)

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        del previous  # the preconditioner is rebuilt from the developed state
        return self._step(state)

    def _step(self, reference: jnp.ndarray) -> NewtonStrategy:
        return flow_march_step(
            self.momentum,
            reference,
            preconditioner_options=self.preconditioner_options,
            **self.march,
        )


def _flow_source(
    momentum: MomentumContinuity,
    preconditioner: MaterializedJacobian | PreconditionerSession | None,
    preconditioner_options: Mapping[str, object] | None,
    reference_state: jnp.ndarray | None,
    refresh: RefreshPolicy,
    march: dict,
):
    """The continuation source for a solve that builds its own step: a session, or block-SIMPLE.

    Refuses configuration the chosen family cannot receive, rather than dropping it: the block-SIMPLE
    settings beside a materialized preconditioner reach nothing, and two refresh hooks re-fitting one
    inverse would disagree.
    """
    if preconditioner is None:
        return _FlowSource(momentum, reference_state, preconditioner_options, march)
    if preconditioner_options is not None:
        raise TypeError(
            "preconditioner_options configure the block-SIMPLE preconditioner, and a materialized-"
            "Jacobian preconditioner was given, which has no such settings. Configure it on the "
            "MaterializedJacobian spec."
        )
    session = (
        open_flow_session(preconditioner, momentum)
        if isinstance(preconditioner, MaterializedJacobian)
        else preconditioner
    )
    if refresh.refresh_preconditioner is not None and session.refresh_preconditioner is not None:
        raise TypeError(
            "RefreshPolicy(refresh_preconditioner=...) was given beside a materialized-Jacobian "
            "preconditioner, whose session already re-fits its inverse. Drop one."
        )
    return SessionSource(session, reference_state, march)


def solve_flow_march(
    momentum: MomentumContinuity,
    state: jnp.ndarray | None = None,
    *,
    strategy: NewtonStrategy | None = None,
    reference_state: jnp.ndarray | None = None,
    preconditioner: MaterializedJacobian | PreconditionerSession | None = None,
    preconditioner_options: Mapping[str, object] | None = None,
    max_steps: int = 60,
    convergence: Convergence | None = None,
    adjoint_solver: lx.AbstractLinearSolver | None = None,
    refresh: RefreshPolicy = NO_REFRESH,
    step_control: StepControl | None = None,
    on_step: Callable[[StepReport], None] | None = None,
    on_checkpoint: Callable[[StepReport, jnp.ndarray], None] | None = None,
    retry: RetryPolicy = NO_RETRIES,
    on_retry: Callable[[str, int, float], None] | None = None,
    homotopy: ResidualHomotopy | None = None,
    station_step: Callable[[NewtonStrategy, int, bool], NewtonStrategy] | None = None,
    **march: object,
) -> jnp.ndarray:
    """Solve the coupled flow system ``R(u, p) = 0`` on the staged march every coupled solve uses.

    The flow counterpart of :func:`~aquaflux.turbulence.solve_coupled`, and the same march: dual-time
    stepping, a retry policy, a row-equilibrated residual measure rebuilt every outer iteration, a
    preconditioner refresh between segments, a step control, a homotopy and per-step observation are all
    configured the same way and mean the same thing. The root it reaches is handed to
    :func:`~aquaflux.solve.root_adjoint`, so the result is reverse-differentiable by one transpose solve
    at the converged state, whatever path the march took.

    The march runs on ``stop_gradient`` copies, so everything it does between steps is ordinary Python
    on concrete values and works under ``jax.grad``; for the same reason it cannot run inside a traced
    program (``jax.jit``, ``jax.vmap``, ``jax.lax.scan``).

    Parameters
    ----------
    momentum : MomentumContinuity
        The flow residual assembler; **the differentiable parameter pytree** for the adjoint.
    state : jnp.ndarray or None
        The initial flow state ``[vel..., pressure]``, shape ``((dim + 1) n_cells,)``. ``None`` starts from
        :func:`~aquaflux.flow.potential_flow`. Never differentiated through.
    strategy : NewtonStrategy or None
        A pre-built step (from :func:`flow_march_step`); ``None`` builds one from the initial state.
    reference_state : jnp.ndarray or None
        The state to freeze the internally built preconditioner at; defaults to the initial state.
    preconditioner : MaterializedJacobian, PreconditionerSession or None
        What preconditions the internally built step. ``None`` (default) is the block-SIMPLE
        preconditioner, configured by ``preconditioner_options``. A
        :class:`~aquaflux.solve.MaterializedJacobian` spec opens a session private to this solve, inverting
        the materialized Jacobian by a complete LU or a multigrid V-cycle (a field split is refused: a
        ``(u, p)`` state has nothing to split); pass a session from :func:`open_flow_session` to share one
        inverse across several solves. Like ``preconditioner_options`` it is refused beside a ``strategy``
        or a ``RefreshPolicy(builder=...)``.
    preconditioner_options : mapping, optional
        The block-SIMPLE preconditioner's settings; see :func:`flow_march_step`. Refused with a
        materialized-Jacobian ``preconditioner``, which has no such settings.
    max_steps : int
        Outer-step cap for **each** march segment.
    convergence : Convergence or None
        The stopping test ``measure(R) <= atol + rtol * measure(R0)``. Unset fields take
        :class:`~aquaflux.solve.RowScaled`, ``rtol = 1e-10`` and ``atol = 1e-12``. The measure both steers
        the march and judges it.
    adjoint_solver : lineax.AbstractLinearSolver, optional
        The solver of the transpose solve behind ``jax.grad``, taken once at the converged state on the
        unshifted operator. ``None`` uses :func:`~aquaflux.solve.default_linear_solver`.
    refresh : RefreshPolicy
        How the frozen preconditioner is kept current. There is no coefficient to watch drift on a
        constant-viscosity flow, so use a cost trigger (:class:`~aquaflux.solve.CycleGrowthTrigger`).
    step_control, on_step, on_checkpoint, retry, on_retry, homotopy, station_step
        As for :func:`~aquaflux.turbulence.solve_coupled`. A dual-time march given no ``step_control``
        defaults to the Courant ramp.
    **march
        The settings of :func:`flow_march_step` (``globalization``, ``dual_time``, ``linear_solve``,
        ``shift_basis``, ...), handed to every build and refresh; an unknown keyword raises.
        ⚠️ Like ``preconditioner_options`` and ``reference_state`` they are accepted only when this
        function builds the step, and are refused beside a ``strategy`` or a ``RefreshPolicy(builder=...)``.

    Returns
    -------
    jnp.ndarray
        The converged flow state.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the march ends short of its stopping target.
    ValueError
        If called from inside a traced program.
    TypeError
        If a step-configuring setting is passed where the step is not built here.
    """
    refuse_a_transform_the_march_cannot_run_in((momentum, state), caller="solve_flow_march")
    frozen = stop_array_gradients(momentum)
    given = dict(march)
    if preconditioner_options is not None:
        given["preconditioner_options"] = preconditioner_options
    if preconditioner is not None:
        given["preconditioner"] = preconditioner
    if reference_state is not None:
        given["reference_state"] = reference_state
    source = explicit_source(strategy, refresh, given, caller="solve_flow_march")
    if source is None:
        source = _flow_source(
            frozen,
            preconditioner,
            preconditioner_options,
            stop_array_gradients(reference_state),
            refresh,
            march,
        )
    state = potential_flow(frozen) if state is None else stop_array_gradients(state)
    staged = staged_march(
        frozen.residual,
        state,
        strategy=strategy,
        source=source,
        refresh=refresh,
        convergence=(
            _FLOW_CONVERGENCE if convergence is None else convergence.filled_from(_FLOW_CONVERGENCE)
        ),
        measures=FlowMeasures(frozen),
        max_steps=max_steps,
        step_control=step_control,
        on_step=on_step,
        on_checkpoint=on_checkpoint,
        retry=retry,
        on_retry=on_retry,
        homotopy=homotopy,
        station_step=station_step,
        caller="solve_flow_march",
    )
    return root_adjoint(
        assembler_residual,
        staged.state,
        momentum,
        adjoint_solver=adjoint_solver,
        adjoint_preconditioner=staged.strategy.adjoint_preconditioner(),
    )
