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
    total : int
        How many points the ramp has, including the target (``n_points + 1``).
    viscosity_scale : float
        The factor the molecular viscosity is multiplied by at this point (``1.0`` at the target, larger
        below it). The Reynolds number is reduced by the same factor.
    """

    index: int
    total: int
    viscosity_scale: float

    @property
    def is_target(self) -> bool:
        """Whether this is the final, true-viscosity point (the one whose root is returned)."""
        return self.viscosity_scale == 1.0

    @property
    def label(self) -> str:
        """A short human label, e.g. ``"point 2/3 (Re/10)"`` -- so every driver need not format one."""
        scaling = "target Re" if self.is_target else f"Re/{self.viscosity_scale:g}"
        return f"point {self.index}/{self.total} ({scaling})"


class ReynoldsSchedule(Protocol):
    """The intermediate Reynolds numbers a continuation visits, as molecular-viscosity scale factors.

    A schedule maps the one integer ``n_points`` to the multiplicative factors applied to the case's
    molecular viscosity, from the lowest-Re anchor down to the target. Each factor ``> 1`` is a
    lower-Reynolds companion problem (``Re`` reduced by that factor); the sequence ends at ``1.0`` (the
    true target). It is a pure function of ``n_points`` -- no mesh, no state -- so it is trivially
    unit-testable.
    """

    def scales(self, n_points: int) -> tuple[float, ...]:
        """The ``n_points + 1`` viscosity-scale factors, descending to ``1.0`` (the target).

        Parameters
        ----------
        n_points : int
            The number of lower-Reynolds continuation points (``>= 0``).
        """
        ...


class GeometricReynoldsSchedule(eqx.Module):
    """Geometric Reynolds spacing: each up-step multiplies the Reynolds number by a fixed ratio.

    With ``ratio = 10`` (the default, one decade per step) the anchor sits at ``Re_target / 10 ** N``
    and every step raises ``Re`` by a decade, so ``n_points`` is the number of decades of continuation:
    ``scales(1) == (10.0, 1.0)`` anchors one decade below the target, ``scales(2) == (100.0, 10.0,
    1.0)`` two decades below. Geometric spacing is the natural choice because the convective
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

    def scales(self, n_points: int) -> tuple[float, ...]:
        """The viscosity-scale factors ``(ratio ** N, ..., ratio, 1.0)`` for ``N = n_points``.

        Parameters
        ----------
        n_points : int
            The number of lower-Reynolds continuation points (``>= 0``); ``0`` yields ``(1.0,)`` -- the
            target alone, i.e. no continuation.
        """
        return tuple(float(self.ratio ** (n_points - i)) for i in range(n_points + 1))


def solve_reynolds_continuation(
    coupled: CoupledRANS,
    n_points: int,
    *,
    schedule: ReynoldsSchedule | None = None,
    intermediate_rtol: float | None = 1e-2,
    intermediate_atol: float | None = None,
    point_setup: Callable[[CoupledRANS, jnp.ndarray, ReynoldsPoint], dict] | None = None,
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
        The intermediate Reynolds spacing; defaults to :class:`GeometricReynoldsSchedule` (one decade
        per step, anchor at ``Re_target / 10 ** n_points``).
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
                "precondition_step": lu_beta_tracking_refresh(comp),
            }

        **Forward-only** (the ``precondition_step`` it returns raises under ``jax.grad``, like the other
        observed-march hooks), so leave it ``None`` when differentiating and use the ``continuation`` path
        instead. ``None`` (default) leaves the ramp byte-identical: each point builds its own continuation
        from :func:`~aquaflux.turbulence.solve_coupled`'s defaults, and the seed is passed through as-is
        (the lowest point self-starts inside ``solve_coupled``). When set, its keys **override** any
        ``continuation`` / ``reference_state`` in ``solve_kwargs`` (they are mutually exclusive uses).
    **solve_kwargs
        Forwarded to every per-Re :func:`~aquaflux.turbulence.solve_coupled`. ``continuation`` and
        ``reference_state`` are **target-specific** (a preconditioner frozen at the target viscosity),
        so they are applied to the final solve only; each lower-Re point builds its own continuation at
        its own viscosity.

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
    scales = schedule.scales(n_points)

    # Build the companions from a stopped copy so the ramp -- which only makes an initial guess --
    # never tapes onto the target-Re adjoint.
    frozen = jax.lax.stop_gradient(coupled)
    # continuation / reference_state freeze a preconditioner at the target viscosity, so they belong to
    # the final solve only; each lower-Re point builds its own at its own viscosity.
    ramp_kwargs = {
        key: value
        for key, value in solve_kwargs.items()
        if key not in ("continuation", "reference_state")
    }
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
        if point_setup is None:
            return solve_coupled(assembler, *seed_fields, **base_kwargs)
        if seed_fields[0] is None:
            seed_fields = hybrid_initialize(assembler.momentum, assembler.turbulence)
        packed = assembler.state_from_physical(*seed_fields)
        return solve_coupled(
            assembler, *seed_fields, **{**base_kwargs, **point_setup(assembler, packed, point)}
        )

    seed: tuple[jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None] = (None, None, None)
    for index, scale in enumerate(scales[:-1]):
        companion = frozen.with_scaled_molecular_viscosity(scale)
        try:
            point = ReynoldsPoint(index + 1, len(scales), float(scale))
            flow, k, omega = _point_solve(companion, seed, ramp_kwargs, point)
        except eqx.EquinoxRuntimeError as exc:
            raise RuntimeError(
                f"Reynolds-continuation point {index + 1} of {n_points} "
                f"(molecular viscosity scaled by {scale:g}, i.e. Reynolds number reduced by "
                f"{scale:g}x) failed to converge. Increase n_points for a gentler ramp, or check the "
                f"case at this Reynolds number directly."
            ) from exc
        # Stop the seed so the next companion's solve (and, ultimately, the target adjoint) does not
        # tape onto this intermediate root.
        seed = tuple(jax.lax.stop_gradient(field) for field in (flow, k, omega))

    # The final point is the true target: the live `coupled` (so its adjoint is the direct solve's),
    # seeded by the last converged lower-Re solution, with the full solve_kwargs.
    target = ReynoldsPoint(len(scales), len(scales), float(scales[-1]))
    return _point_solve(coupled, seed, solve_kwargs, target)
