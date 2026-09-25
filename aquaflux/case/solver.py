"""How a case is solved: the march, its preconditioner, its stopping test and its continuation.

A case file's ``solver`` section names one of three solves, each the library solve of the same shape:

* :class:`CoupledMarch` -- a Reynolds-averaged case marched as one coupled system of the flow and the
  closure (:func:`~aquaflux.turbulence.solve_coupled`), optionally along a viscosity ramp
  (:class:`ViscosityRamp`, by :func:`~aquaflux.turbulence.solve_reynolds_ramp`);
* :class:`FlowMarch` -- a laminar case marched as the flow system (:func:`~aquaflux.flow.solve_flow_march`);
* :class:`Segregated` -- a Reynolds-averaged case solved by alternating the flow and the closure
  (:func:`~aquaflux.turbulence.solve_segregated`), which is also the solve that holds a
  :class:`~aquaflux.case.BulkVelocity` drive.

Every setting is optional unless its solve cannot run without it, and an unset one leaves the library
solve's own default in force. What a solver section holds is settings, never a built solve: a march
re-fits its preconditioner from states the file never sees, at each continuation station and each
refresh, and it does so from the same settings every time.

A solve can be observed without being reconfigured. :meth:`SolverSpec.solve` takes the problem the case
built and, optionally, observers -- a logger, a checkpointer -- which it passes to the library solve
beside the file's settings. It refuses an observer keyword that is also a setting, so a script that
runs a case cannot change what the case says.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal

from aquaflux.flow import MassFlow, bulk_velocity_flow_solve, reused_flow_solve, solve_flow_march
from aquaflux.solve import (
    Convergence,
    DualTimeLoop,
    LinearSolverSpec,
    LinearSolveSettings,
    MaterializedJacobian,
    RetryPolicy,
    RootSolveSettings,
    ShiftStrengthControl,
)
from aquaflux.turbulence import (
    BlockDiagonal,
    CoupledShiftSettings,
    ScalarBlock,
    open_session,
    scalar_pseudo_transient_solve,
    scale_both_blocks,
    scale_momentum_only,
    solve_coupled,
    solve_reynolds_ramp,
    solve_segregated,
    sst_initial_fields,
)

from .forcing import DriveSpec
from .physics import RANS, Laminar, Physics, _set

__all__ = [
    "CoupledMarch",
    "FlowMarch",
    "RootSolve",
    "Segregated",
    "SolverSpec",
    "ViscosityRamp",
]


@dataclasses.dataclass(frozen=True)
class SolverSpec(abc.ABC):
    """How a case is solved: :class:`CoupledMarch`, :class:`FlowMarch` or :class:`Segregated`."""

    @abc.abstractmethod
    def refuse_for(self, physics: Physics, drive: DriveSpec | None) -> None:
        """Refuse a case this solve cannot solve.

        Parameters
        ----------
        physics : Physics
            The case's physics.
        drive : DriveSpec or None
            The case's drive.

        Raises
        ------
        ValueError
            If the solve is not one for this physics, or cannot hold this drive.
        """

    @abc.abstractmethod
    def solve(self, problem: object, **observers: object) -> object:
        """Solve ``problem``, the problem the case built.

        Parameters
        ----------
        problem : object
            What :meth:`~aquaflux.case.CheckedCase.build` returned.
        **observers
            Keywords of the library solve that observe it without changing it -- ``on_checkpoint``,
            ``on_retry``, ``inner_observer`` and the like. See each solve for which it accepts.

        Returns
        -------
        object
            The converged fields, as the library solve returns them.

        Raises
        ------
        TypeError
            If an observer keyword is one of the solve's settings.
        """


def _refuse_settings_as_observers(owner: str, owned: frozenset[str], observers) -> None:
    """Refuse an observer keyword that is one of the solve's settings, whether the case sets it or not.

    A setting the case leaves unset is still the case's: its default is part of what the case says.
    """
    clashes = sorted(set(observers) & owned)
    if clashes:
        raise TypeError(
            f"{owner}: {', '.join(clashes)} {'is a setting' if len(clashes) == 1 else 'are settings'} "
            "of the case, so a script running the case cannot pass "
            f"{'it' if len(clashes) == 1 else 'them'}. Set it in the case file."
        )


def _refuse_physics(owner: str, physics: Physics, wanted: type, other: str) -> None:
    if not isinstance(physics, wanted):
        raise ValueError(
            f"solver: {owner} solves a {wanted.__name__} case, but the physics is "
            f"{type(physics).__name__}; use {other}."
        )


def _refuse_a_held_bulk_velocity(owner: str, drive: DriveSpec | None, other: str) -> None:
    """Refuse a drive the march cannot hold: it marches the fields alone, so the force would stay fixed."""
    if drive is not None:
        raise ValueError(
            f"solver: {owner} marches the fields alone, so it cannot hold the bulk velocity the "
            f"case's {type(drive).__name__} drive asks for -- the force would stay at its starting "
            f"guess. {other}"
        )


@dataclasses.dataclass(frozen=True)
class _March(SolverSpec):
    """The settings every observed march takes, whatever residual it marches.

    Attributes
    ----------
    max_steps : int or None
        The outer-step cap of each march segment; unset, the solve's own.
    convergence : Convergence or None
        The stopping test -- a measure and its tolerances; an unset part takes the solve's own.
    preconditioner : BlockDiagonal, MaterializedJacobian or None
        What preconditions each step's linear solve; unset, the solve's own.
    dual_time : DualTimeLoop or None
        Run each outer step as a dual-time inner loop; unset, a single pseudo-transient step.
    linear_solve : LinearSolveSettings or None
        The Krylov regime of each step's linear solve; an unset part takes the preconditioner's own.
    step_control : ShiftStrengthControl or None
        How the pseudo-time shift adapts from step to step; unset, the solve's own (a Courant ramp for a
        dual-time march).
    retry : RetryPolicy or None
        When and how a bad step is redone; unset, never.
    """

    max_steps: int | None = None
    convergence: Convergence | None = None
    preconditioner: BlockDiagonal | MaterializedJacobian | None = None
    dual_time: DualTimeLoop | None = None
    linear_solve: LinearSolveSettings | None = None
    step_control: ShiftStrengthControl | None = None
    retry: RetryPolicy | None = None

    def _owned(self) -> frozenset[str]:
        """The library solve's keywords that are this solve's settings, set or not.

        Each march field is passed under its own name, and ``homotopy`` belongs to the solve too: it is
        the continuation a case states, or none.
        """
        return frozenset(field.name for field in dataclasses.fields(_March)) | {"homotopy"}

    def __post_init__(self) -> None:
        for name, family in (
            ("convergence", Convergence),
            ("preconditioner", (BlockDiagonal, MaterializedJacobian)),
            ("dual_time", DualTimeLoop),
            ("linear_solve", LinearSolveSettings),
            ("step_control", ShiftStrengthControl),
            ("retry", RetryPolicy),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, family):
                raise TypeError(f"{type(self).__name__}.{name} got {value!r}.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(
                f"{type(self).__name__}.max_steps must be >= 1, got {self.max_steps!r}."
            )

    def _march_settings(self) -> dict[str, object]:
        """The set march settings, by the library solve's keyword -- each field's own name."""
        return _set(
            **{field.name: getattr(self, field.name) for field in dataclasses.fields(_March)}
        )


#: The viscosity a ramp station scales, by the name a file gives it; unset leaves the ramp's own.
_RAMP_SCALINGS = {"flow": scale_momentum_only, "both": scale_both_blocks, None: None}


@dataclasses.dataclass(frozen=True)
class ViscosityRamp:
    """Reach the case's viscosity along a geometric ramp inside one march, from a larger one.

    A Reynolds-averaged march from a cold start at a high Reynolds number takes a long transient to
    develop; started at a viscosity ``anchor`` times the case's and walked down to it station by
    station, it keeps its state, its shift and its preconditioner across every change and converges the
    case's own problem only. The intermediate stations are a path, not answers.

    Attributes
    ----------
    anchor : float
        The factor the viscosity starts at, ``> 1``.
    stations : int
        How many geometric steps the viscosity is walked down in, ``>= 1``.
    steps_per_station : int
        Outer steps held at each station, ``>= 1``.
    scale : {"flow", "both"} or None
        Which viscosity a station scales: the flow's only, leaving the closure at the case's own
        (``flow``), or both (``both``, a genuine lower-Reynolds problem at each station); unset, both.
    redamping : float or None
        How much the shift is raised on entering each station; unset, the ramp's own derived value.

    Raises
    ------
    ValueError
        If a count is not ``>= 1`` or the anchor is not above one.
    """

    anchor: float
    stations: int
    steps_per_station: int
    scale: Literal["flow", "both"] | None = None
    redamping: float | None = None

    def __post_init__(self) -> None:
        if self.scale not in _RAMP_SCALINGS:
            raise ValueError(
                f"ViscosityRamp.scale is one of {sorted(k for k in _RAMP_SCALINGS if k)} or unset, "
                f"got {self.scale!r}."
            )
        if not self.anchor > 1.0:
            raise ValueError(f"ViscosityRamp.anchor must be above 1, got {self.anchor!r}.")
        for name in ("stations", "steps_per_station"):
            if getattr(self, name) < 1:
                raise ValueError(f"ViscosityRamp.{name} must be >= 1, got {getattr(self, name)!r}.")

    def solve(self, coupled: object, options: dict[str, object], point_setup: Callable) -> object:
        """March ``coupled`` along this ramp, by :func:`~aquaflux.turbulence.solve_reynolds_ramp`.

        Parameters
        ----------
        coupled : CoupledRANS
            The case's coupled problem, at the case's own viscosity; the march ends on it.
        options : dict
            The march's other keywords -- its settings and any observers.
        point_setup : callable
            The hook the ramp calls once, for its anchor station; it returns no settings.

        Returns
        -------
        tuple of jnp.ndarray
            The converged ``(flow, k, omega)``.
        """
        companion = _RAMP_SCALINGS[self.scale]
        return solve_reynolds_ramp(
            coupled,
            anchor=self.anchor,
            stations=self.stations,
            steps_per_station=self.steps_per_station,
            point_setup=point_setup,
            **_set(redamping=self.redamping, companion=companion),
            **options,
        )


def _no_point_setup(companion: object, seed_state: object, point: object) -> dict[str, object]:
    """A ramp's per-station configuration: none, since every setting is the case's."""
    del companion, seed_state, point
    return {}


def _observing_point_setup(observe: Callable) -> Callable:
    """``observe`` as a ramp's point hook, refused if it returns settings rather than only observing."""

    def point_setup(companion: object, seed_state: object, point: object) -> dict[str, object]:
        returned = observe(companion, seed_state, point)
        if returned:
            raise TypeError(
                f"a point_setup passed to a case's solve may only observe, but it returned settings "
                f"{sorted(returned)}; set them in the case file."
            )
        return {}

    return point_setup


@dataclasses.dataclass(frozen=True)
class CoupledMarch(_March):
    """March a Reynolds-averaged case as one coupled system of the flow and the closure.

    The march settings are :class:`FlowMarch`'s; these add what belongs to the closure. Without a
    ``continuation`` it is :func:`~aquaflux.turbulence.solve_coupled`, self-starting from a hybrid
    initial condition; with one it is :func:`~aquaflux.turbulence.solve_reynolds_ramp`.

    Attributes
    ----------
    turbulence_damping : float or None
        How much harder the ``k`` and ``omega`` rows are damped than the flow's, as a multiplier on
        their shift strength; unset, ``1``.
    positivity_floor : float or None
        The least ``k`` a step may leave; unset, the solve's own.
    positivity_projection : bool or None
        Project ``k`` back to the floor after a step rather than shortening the step to respect it;
        unset, the solve's own.
    continuation : ViscosityRamp or None
        Reach the case's viscosity along a ramp; unset, march at it from the start.

    Raises
    ------
    ValueError
        If the case is not Reynolds-averaged, or holds a bulk velocity (see :class:`Segregated`).
    """

    turbulence_damping: float | None = None
    positivity_floor: float | None = None
    positivity_projection: bool | None = None
    continuation: ViscosityRamp | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.continuation is not None and not isinstance(self.continuation, ViscosityRamp):
            raise TypeError(f"CoupledMarch.continuation got {self.continuation!r}.")

    def _owned(self) -> frozenset[str]:
        """The march's keywords and the closure's -- see :meth:`_March._owned`."""
        return super()._owned() | {"positivity_floor", "positivity_projection", "shift"}

    def refuse_for(self, physics: Physics, drive: DriveSpec | None) -> None:
        """Refuse a laminar case, or one holding a bulk velocity -- see :meth:`SolverSpec.refuse_for`."""
        _refuse_physics("CoupledMarch", physics, RANS, "FlowMarch")
        _refuse_a_held_bulk_velocity(
            "CoupledMarch", drive, "Solve it with a Segregated solver, which holds it."
        )

    def settings(self) -> dict[str, object]:
        """The keywords of :func:`~aquaflux.turbulence.solve_coupled` this march sets.

        Returns
        -------
        dict
            By keyword; a setting left unset is absent, so the solve's own default applies.
        """
        settings = self._march_settings() | _set(
            positivity_floor=self.positivity_floor,
            positivity_projection=self.positivity_projection,
        )
        if self.turbulence_damping is not None:
            settings["shift"] = CoupledShiftSettings(turbulence_damping=self.turbulence_damping)
        return settings

    def solve(
        self,
        problem: object,
        *,
        session_options: Mapping[str, object] | None = None,
        point_setup: Callable | None = None,
        **observers: object,
    ) -> object:
        """March ``problem`` -- see :meth:`SolverSpec.solve`.

        A materialized-Jacobian preconditioner is opened here as one session
        (:func:`~aquaflux.turbulence.open_session`), which the ramp re-points at each station.

        Parameters
        ----------
        problem : CoupledRANS
            The case's coupled problem.
        session_options : mapping, optional
            Observers for that session -- ``observer``, ``reports``, ``on_build`` and the other
            keywords of :func:`~aquaflux.turbulence.open_session` beyond the spec. Refused for a
            block-diagonal preconditioner, which is not opened here.
        point_setup : callable, optional
            ``(companion, seed_state, point) -> {}``, called for the ramp's anchor station to observe
            it. It may only observe: returning a setting is refused. Refused without a continuation.
        **observers
            Further observer keywords of :func:`~aquaflux.turbulence.solve_coupled` (``on_checkpoint``,
            ``on_retry``, ``inner_observer``, ``station_step``, ...).

        Returns
        -------
        tuple of jnp.ndarray
            The converged ``(flow, k, omega)``.
        """
        settings = self.settings()
        _refuse_settings_as_observers("CoupledMarch", self._owned(), observers)
        if point_setup is not None and self.continuation is None:
            raise TypeError(
                "point_setup observes a continuation's stations, and this march has none."
            )
        preconditioner = settings.get("preconditioner")
        if isinstance(preconditioner, MaterializedJacobian):
            settings["preconditioner"] = open_session(
                preconditioner, problem, **dict(session_options or {})
            )
        elif session_options:
            raise TypeError(
                "session_options observe a materialized-Jacobian preconditioner's session, and this "
                "march's preconditioner is not one."
            )
        options = settings | observers
        if self.continuation is None:
            return solve_coupled(problem, **options)
        return self.continuation.solve(
            problem,
            options,
            _no_point_setup if point_setup is None else _observing_point_setup(point_setup),
        )


@dataclasses.dataclass(frozen=True)
class FlowMarch(_March):
    """March a laminar case as its coupled flow system, by :func:`~aquaflux.flow.solve_flow_march`.

    It self-starts from a potential-flow initial condition. Its preconditioner, when set, is a
    :class:`~aquaflux.solve.MaterializedJacobian`; the block-diagonal family is a Reynolds-averaged
    one.

    Raises
    ------
    ValueError
        If the case is Reynolds-averaged, or holds a bulk velocity.
    TypeError
        If the preconditioner is block-diagonal.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.preconditioner, BlockDiagonal):
            raise TypeError(
                "FlowMarch.preconditioner is a MaterializedJacobian; BlockDiagonal preconditions a "
                "Reynolds-averaged march."
            )

    def refuse_for(self, physics: Physics, drive: DriveSpec | None) -> None:
        """Refuse a Reynolds-averaged case, or one holding a bulk velocity -- see :meth:`SolverSpec.refuse_for`."""
        _refuse_physics("FlowMarch", physics, Laminar, "CoupledMarch or Segregated")
        _refuse_a_held_bulk_velocity(
            "FlowMarch", drive, "A laminar case cannot hold one from a case file yet."
        )

    def settings(self) -> dict[str, object]:
        """The keywords of :func:`~aquaflux.flow.solve_flow_march` this march sets.

        Returns
        -------
        dict
            By keyword; a setting left unset is absent, so the solve's own default applies.
        """
        return self._march_settings()

    def solve(self, problem: object, **observers: object) -> object:
        """March ``problem`` -- see :meth:`SolverSpec.solve`.

        Parameters
        ----------
        problem : MomentumContinuity
            The case's flow problem.
        **observers
            Observer keywords of :func:`~aquaflux.flow.solve_flow_march` (``on_step``,
            ``on_checkpoint``, ``on_retry``, ...).

        Returns
        -------
        jnp.ndarray
            The converged flow state.
        """
        settings = self.settings()
        _refuse_settings_as_observers("FlowMarch", self._owned(), observers)
        return solve_flow_march(problem, **settings, **observers)


@dataclasses.dataclass(frozen=True)
class RootSolve:
    """How one Newton solve inside a segregated sweep is run: its step cap, its stop, its linear solver.

    The settings of :class:`~aquaflux.solve.RootSolveSettings` a case file can state, each unset one
    left to the solve's own default.

    Attributes
    ----------
    max_steps : int or None
        The Newton step cap.
    convergence : Convergence or None
        The stopping test.
    linear_solver : LinearSolverSpec or None
        The forward linear solver.
    adjoint_solver : LinearSolverSpec or None
        The linear solver of the transpose solve a gradient through the solve takes.
    """

    max_steps: int | None = None
    convergence: Convergence | None = None
    linear_solver: LinearSolverSpec | None = None
    adjoint_solver: LinearSolverSpec | None = None

    def root_solve_settings(self) -> RootSolveSettings:
        """The library value these settings describe.

        Returns
        -------
        RootSolveSettings
            With each linear solver built from its settings.
        """
        return RootSolveSettings(
            max_steps=self.max_steps,
            convergence=self.convergence,
            linear_solver=None if self.linear_solver is None else self.linear_solver.build(),
            adjoint_solver=None if self.adjoint_solver is None else self.adjoint_solver.build(),
        )


@dataclasses.dataclass(frozen=True)
class Segregated(SolverSpec):
    """Solve a Reynolds-averaged case by alternating the flow and the closure, by :func:`~aquaflux.turbulence.solve_segregated`.

    Each sweep solves the flow at the current eddy viscosity and then ``k`` and ``omega`` at the new
    flow, under-relaxing the closure's update. The flow solve holds the case's bulk velocity when its
    drive is a :class:`~aquaflux.case.BulkVelocity` (:func:`~aquaflux.flow.bulk_velocity_flow_solve`);
    otherwise it is :func:`~aquaflux.flow.reused_flow_solve`. It starts from
    :func:`~aquaflux.turbulence.sst_initial_fields`.

    Attributes
    ----------
    sweeps : int
        The most sweeps it may take, ``>= 1``; it stops earlier once a sweep changes the fields by less
        than ``increment_tol``.
    relaxation : float or None
        The under-relaxation of the closure's update, in ``(0, 1]``; unset, the solve's own.
    relaxation_max : float or None
        The largest relaxation it may grow to as the sweeps converge; unset, no growth.
    increment_tol : float or None
        The largest per-field relative change over a sweep at which it stops; unset, the solve's own.
    flow_solve, scalar_solve : RootSolve or None
        How the flow solve and each of the ``k`` and ``omega`` solves are run; unset, their own.
    scalar_preconditioner : ScalarBlock or None
        What preconditions the ``k`` and ``omega`` solves; unset, none.

    Raises
    ------
    ValueError
        If the case is laminar, or ``sweeps`` is not ``>= 1``.
    """

    sweeps: int
    relaxation: float | None = None
    relaxation_max: float | None = None
    increment_tol: float | None = None
    flow_solve: RootSolve | None = None
    scalar_solve: RootSolve | None = None
    scalar_preconditioner: ScalarBlock | None = None

    def __post_init__(self) -> None:
        if self.sweeps < 1:
            raise ValueError(f"Segregated.sweeps must be >= 1, got {self.sweeps!r}.")
        for name, family in (
            ("flow_solve", RootSolve),
            ("scalar_solve", RootSolve),
            ("scalar_preconditioner", ScalarBlock),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, family):
                raise TypeError(f"Segregated.{name} got {value!r}.")

    def refuse_for(self, physics: Physics, drive: DriveSpec | None) -> None:
        """Refuse a laminar case -- see :meth:`SolverSpec.refuse_for`."""
        del drive
        _refuse_physics("Segregated", physics, RANS, "FlowMarch")

    def solve(self, problem: object, **observers: object) -> object:
        """Solve ``problem`` -- see :meth:`SolverSpec.solve`. It takes no observers.

        Parameters
        ----------
        problem : CoupledRANS
            The case's coupled problem, whose flow and closure are solved in turn.

        Returns
        -------
        tuple of jnp.ndarray
            The converged ``(flow, k, omega)``.
        """
        if observers:
            raise TypeError(f"Segregated takes no observers, got {sorted(observers)}.")
        momentum, turbulence = problem.momentum, problem.turbulence
        flow_root = (
            {} if self.flow_solve is None else {"root_solve": self.flow_solve.root_solve_settings()}
        )
        build_flow_solve = (
            bulk_velocity_flow_solve if isinstance(momentum.drive, MassFlow) else reused_flow_solve
        )
        scalar_root = (
            {}
            if self.scalar_solve is None
            else {"root_solve": self.scalar_solve.root_solve_settings()}
        )
        return solve_segregated(
            momentum,
            turbulence,
            build_flow_solve(momentum, **flow_root),
            scalar_pseudo_transient_solve(**scalar_root),
            *sst_initial_fields(momentum, turbulence),
            max_sweeps=self.sweeps,
            **_set(
                relaxation=self.relaxation,
                relaxation_max=self.relaxation_max,
                increment_tol=self.increment_tol,
                scalar_preconditioner=self.scalar_preconditioner,
            ),
        )


def solver_for(spec: object) -> SolverSpec:
    """The solver a case runs: the one it states, or its physics' march with every setting unset.

    Parameters
    ----------
    spec : CaseSpec
        The case.

    Returns
    -------
    SolverSpec
        :attr:`~aquaflux.case.CaseSpec.solver`, or unset, :class:`CoupledMarch` for a
        Reynolds-averaged case and :class:`FlowMarch` for a laminar one.

    Raises
    ------
    ValueError
        If the case states no solver and that default cannot solve it -- a case holding a bulk
        velocity must name its solve.
    """
    if spec.solver is not None:
        return spec.solver
    solver = CoupledMarch() if isinstance(spec.physics, RANS) else FlowMarch()
    try:
        solver.refuse_for(spec.physics, spec.drive)
    except ValueError as error:
        raise ValueError(
            f"the case states no solver, and its default cannot solve it: {error}"
        ) from error
    return solver
