"""The pseudo-transient / dual-time Newton step a march runs, assembled around a composed shift policy.

Every march in the package -- the coupled RANS one and the laminar flow-only one -- ends the same way:
a :class:`~aquaflux.solve.ShiftPolicy` (the pseudo-time shift and the preconditioner matched to it) is
handed to a :class:`~aquaflux.solve.Globalization`, which builds either a single shifted step
(:class:`~aquaflux.solve.PseudoTransientStep`) or a dual-time inner loop
(:class:`~aquaflux.solve.DualTimeStep`). :func:`shifted_step` is that tail, so what the two share --
the refusal of hooks a single step has nothing to attach to, the refusal of a refresh with nothing to
fire, the choice between the two step classes -- is written once. What differs per residual is passed
in: the shift policy, the default line search, the linear solve and the operator the Krylov solve
differentiates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
import lineax as lx

from .continuation import DualTimeLoop, Globalization
from .linear import relative_residual_gmres
from .settings_value import SettingsValue
from .strategy import NewtonStrategy

__all__ = ["LinearSolveRegime", "LinearSolveSettings", "resolve_linear_solve", "shifted_step"]


class LinearSolveRegime(NamedTuple):
    """One preconditioner family's default Krylov regime for the shifted forward solve.

    The stop is the progress measure the march hands the step (the solve's
    :class:`~aquaflux.solve.Convergence` measure), so a solve cannot converge in a quantity the march
    does not read, including after the march rebuilds that measure at a new state. That half of the
    decision is shared by every march; only the restart regime differs per preconditioner.

    Attributes
    ----------
    rtol : float
        Relative tolerance, **in the progress measure the march hands the step**. The same number is a
        different tightness under a different measure.
    restart : int
        Arnoldi restart length. A restarted GMRES tests its stop only at each restart boundary, so the
        subspace should match how many vectors the preconditioner actually needs.
    max_restarts : int
        Restart-cycle cap -- the only bound on a single running solve, since a cycle budget and the
        march's abort threshold are tested *between* inner iterations.
    """

    rtol: float
    restart: int
    max_restarts: int


@dataclasses.dataclass(frozen=True)
class LinearSolveSettings(SettingsValue):
    """The shifted forward solve's Krylov regime, as one value.

    Each field is resolved against the chosen preconditioner family's own regime when unset. A builder
    takes this value or a whole ``lineax`` solver in the same parameter, never both, because a solver
    replaces the regime -- and the stopping measure with it.

    ⚠️ **``rtol`` is measured in the solve's progress measure** -- the measure of its
    :class:`~aquaflux.solve.Convergence` -- not in the Euclidean norm. A coupled residual's 2-norm can be
    almost entirely one block, so a Euclidean stop halts while the rest of the step is still coarse. The
    same number is therefore not the same tightness under a different measure.

    Attributes
    ----------
    rtol : float or None
        The relative tolerance each shifted solve stops at, in the progress measure.
    restart : int or None
        The Arnoldi restart length.
    max_restarts : int or None
        The restart-cycle cap, in raw ``lineax`` restarts -- the only bound on a single running solve.
    """

    rtol: float | None = None
    restart: int | None = None
    max_restarts: int | None = None


def resolve_linear_solve(
    linear_solve: LinearSolveSettings | lx.AbstractLinearSolver | None, base: LinearSolveRegime
) -> tuple[LinearSolveRegime, lx.AbstractLinearSolver | None]:
    """The forward solve's regime and explicit solver, from a builder's ``linear_solve``.

    Parameters
    ----------
    linear_solve : LinearSolveSettings, lineax.AbstractLinearSolver or None
        A regime whose unset fields take ``base``; a whole solver, which replaces the regime; or
        ``None`` for ``base`` itself.
    base : LinearSolveRegime
        The chosen preconditioner family's own regime.

    Returns
    -------
    tuple
        ``(regime, krylov_solver)``, the solver ``None`` unless one was given.
    """
    if linear_solve is None:
        return base, None
    if isinstance(linear_solve, LinearSolveSettings):
        return (
            LinearSolveRegime(
                base.rtol if linear_solve.rtol is None else linear_solve.rtol,
                base.restart if linear_solve.restart is None else linear_solve.restart,
                base.max_restarts
                if linear_solve.max_restarts is None
                else linear_solve.max_restarts,
            ),
            None,
        )
    return base, linear_solve


def shifted_step(
    policy: object,
    *,
    globalization: Globalization,
    dual_time: DualTimeLoop | None,
    regime: LinearSolveRegime | None,
    krylov_solver: lx.AbstractLinearSolver | None,
    adjoint_preconditioner_factory: Callable | None,
    inner_observer: Callable[..., None] | None = None,
    inner_refresh: Callable[[jnp.ndarray], None] | None = None,
    step_limit: Callable[..., jnp.ndarray] | None = None,
    step_projection: Callable[..., jnp.ndarray] | None = None,
    jacobian_residual: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    line_search: int | None = None,
) -> NewtonStrategy:
    """Assemble the shifted step around an already-composed shift policy.

    Parameters
    ----------
    policy : ShiftPolicy
        The composed shift-and-preconditioner policy.
    globalization : Globalization
        How hard the march damps and what it does when a step misbehaves. Its unset ``line_search``
        takes ``line_search`` below, and every other unset field the step class's own default. Beside a
        ``dual_time`` loop the escalation-ladder fields are refused, since a dual-time step has no
        ladder for them to reach.
    dual_time : DualTimeLoop or None
        The dual-time inner loop, or ``None`` for the single shifted step.
    regime : LinearSolveRegime or None
        The Krylov tolerance and restart regime of the default forward solve, used when
        ``krylov_solver`` is ``None``. The default stops in the progress measure the march hands the step
        at every outer iteration (``relative_residual_gmres(norm=None)``). ``None`` with no
        ``krylov_solver`` leaves the step class's own default solve.
    krylov_solver : lineax.AbstractLinearSolver or None
        A whole forward solver, replacing the regime **and** the stopping measure.
    adjoint_preconditioner_factory : callable or None
        Builds the preconditioner of the converged adjoint solve. Not the policy's to supply: a flow-only
        policy has no ``adjoint_factory`` method, its block preconditioner does.
    inner_observer, inner_refresh, step_limit, step_projection, jacobian_residual
        The dual-time loop's hooks and the per-step guards, forwarded to the step class. Forward-only.
    line_search : int or None
        The line-search rungs an unset ``globalization.line_search`` takes for this residual. ``None``
        leaves an unset one at the step class's own default (the full shifted step).

    Returns
    -------
    NewtonStrategy
        A :class:`~aquaflux.solve.DualTimeStep` when ``dual_time`` is given, else a
        :class:`~aquaflux.solve.PseudoTransientStep`.

    Raises
    ------
    TypeError
        If a dual-time hook is given without a dual-time loop, or the loop's ``refresh_on_cycles`` has no
        ``inner_refresh`` to fire.
    """
    if krylov_solver is None and regime is not None:
        krylov_solver = relative_residual_gmres(
            regime.rtol,
            norm=None,
            restart=regime.restart,
            stagnation_iters=40,
            max_restarts=regime.max_restarts,
        )
    if line_search is not None:
        globalization = globalization.with_defaults(line_search=line_search)
    if dual_time is None:
        hooks = sorted(
            name
            for name, hook in (("inner_observer", inner_observer), ("inner_refresh", inner_refresh))
            if hook is not None
        )
        if hooks:
            raise TypeError(
                f"{hooks} are hooks of the dual-time inner loop, and this march has none: the single "
                "shifted step runs no inner iterations to observe or refresh. Give "
                "dual_time=DualTimeLoop(...) to march in dual time, or leave them unset."
            )
        # The positivity guard is passed on BOTH branches: an escalation ladder is no substitute for
        # it, because the divergence guard fires on a non-finite residual, which is already the
        # poisoned state.
        return globalization.step(
            policy,
            krylov_solver=krylov_solver,
            adjoint_preconditioner_factory=adjoint_preconditioner_factory,
            step_limit=step_limit,
            step_projection=step_projection,
            jacobian_residual=jacobian_residual,
        )
    if dual_time.refresh_on_cycles is not None and inner_refresh is None:
        raise TypeError(
            f"DualTimeLoop(refresh_on_cycles={dual_time.refresh_on_cycles}) has nothing to fire: no "
            "inner_refresh was given, and only a materialized-Jacobian preconditioner's session supplies "
            "one of its own -- a frozen step and a block-diagonal session do not. Open a session for a "
            "MaterializedJacobian, pass inner_refresh, or leave refresh_on_cycles unset."
        )
    # Dual-time (backward-Euler) march: an inner Newton loop per outer timestep on the transient
    # residual, so the measured steady residual is the honest discrete time derivative rather than
    # beta x travel, and a larger pseudo-timestep (smaller beta, driven by a step control) stays
    # stable. The inner loop replaces the escalation ladder, so `dual_time_step` refuses the
    # escalation/acceptance settings and the line search's growth rung and rule rather than dropping
    # them.
    return globalization.dual_time_step(
        policy,
        **dual_time.settings(),
        krylov_solver=krylov_solver,
        adjoint_preconditioner_factory=adjoint_preconditioner_factory,
        inner_observer=inner_observer,
        inner_refresh=inner_refresh,
        step_limit=step_limit,
        step_projection=step_projection,
        jacobian_residual=jacobian_residual,
    )
