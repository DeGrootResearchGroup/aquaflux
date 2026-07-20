"""The segregated outer loop coupling the flow solve to the k-omega SST turbulence model.

The coupled RANS system is solved by Picard iteration rather than as one monolithic block: each
outer sweep freezes the eddy viscosity for the flow solve, then freezes the flow for the turbulence
solve. A sweep is

1. eddy viscosity ``nu_t`` from the current velocity gradient and ``(k, omega)``;
2. the momentum viscosity set to the effective ``mu_eff = rho (nu + nu_t)`` and the flow solved;
3. the closure fields recomputed from the new flow, and the k then omega equations solved on the
   flow's Rhie--Chow mass flux;
4. ``k`` and ``omega`` under-relaxed towards the new values and floored to keep them positive.

The relaxation and the floor are the segregated iteration's stabilisers; the floor is a
nonlinear-iteration safeguard that should be inactive once the fields have converged (a converged
turbulent field is strictly positive). Constant density (see :mod:`~aquaflux.turbulence.transport`).

The outer loop **stops on convergence**, not a fixed sweep count: each sweep's coupled Picard
increment -- the largest per-field relative change ``max(||dflow||/||flow||, ||dk||/||k||,
||domega||/||omega||)`` -- is the residual-agnostic fixed-point measure, and the loop exits once it
drops below ``rtol`` (``max_sweeps`` only caps the work; a warning fires if it is hit without
converging). The outer under-relaxation is **adaptive**: it opens from its floor toward
``relaxation_max`` as that increment falls, a switched-evolution-relaxation ramp
``w = clip(w0 (r0/r)^p, w0, w_max)`` -- conservative while the coupling is still moving, then close
to a full step near the fixed point, so a safe floor need not throttle the whole march. With
``relaxation_max`` left at the floor the relaxation is constant (the plain Picard loop).

The flow and scalar solvers are **injected** rather than chosen here (the preconditioner, iteration
counts, and adjoint are the caller's to set): ``solve_flow(momentum, state)`` solves the momentum
residual for the (re-viscosified) assembler, and ``solve_scalar(residual, state, policy)`` solves a
scalar residual. Both stiff reactive scalars are globalized by pseudo-transient continuation -- the
driver builds the per-sweep :class:`~aquaflux.turbulence.continuation.ScalarShiftPolicy` and the
caller wires :func:`~aquaflux.turbulence.scalar_pseudo_transient_solve`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.properties import FieldProperty, PropertyModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from aquaflux.flow import MomentumContinuity

    from .transport import SSTTurbulence


def _with_viscosity(
    momentum: MomentumContinuity, effective_viscosity: jnp.ndarray
) -> MomentumContinuity:
    """Return ``momentum`` with its ``viscosity`` property replaced by a per-cell field."""
    properties = PropertyModel(
        {**momentum.properties.properties, "viscosity": FieldProperty(effective_viscosity)}
    )
    return eqx.tree_at(lambda m: m.properties, momentum, properties)


def _relax(old: jnp.ndarray, new: jnp.ndarray, factor: float) -> jnp.ndarray:
    """Under-relax: ``old + factor (new - old)``."""
    return old + factor * (new - old)


def _relative_change(*field_pairs: tuple[jnp.ndarray, jnp.ndarray]) -> float:
    """Largest per-field relative L2 change ``max_i ||new_i - old_i|| / ||new_i||``.

    The scale-free Picard-increment measure that drives both the outer convergence test and the
    relaxation ramp: taking the max over the fields (rather than one combined norm) keeps the three
    disparate scales -- velocity O(1), ``k`` O(1e-3), ``omega`` O(1e2) -- comparable, so a stalled
    ``omega`` cannot hide behind a converged velocity. The tiny floor guards the first-sweep divide
    when a field starts at exactly zero.
    """
    worst = 0.0
    for old, new in field_pairs:
        change = float(jnp.linalg.norm(new - old) / jnp.maximum(jnp.linalg.norm(new), 1e-30))
        worst = max(worst, change)
    return worst


def _sweep_relaxation(
    increment: float | None,
    increment_0: float | None,
    relaxation: float,
    relaxation_ceiling: float,
    ser_exponent: float,
) -> float:
    """The adaptive outer under-relaxation for a sweep: the SER ramp opened by the increment drop.

    ``clip(relaxation (increment_0 / increment)^ser_exponent, relaxation, relaxation_ceiling)`` -- the
    floor ``relaxation`` on the first sweep (no increment yet) and whenever the coupling is not yet
    contracting, ramping up toward ``relaxation_ceiling`` as the increment falls below its first value.
    With ``relaxation_ceiling == relaxation`` this is a constant floor (the plain Picard loop).
    """
    if increment is None or increment_0 is None:
        return relaxation
    ramp = (increment_0 / max(increment, 1e-30)) ** ser_exponent
    return float(jnp.clip(relaxation * ramp, relaxation, relaxation_ceiling))


def bulk_velocity(
    momentum: MomentumContinuity, flow: jnp.ndarray, direction: int = 0
) -> jnp.ndarray:
    """Volume-averaged velocity component ``Sigma(u_dir V) / Sigma(V)`` for a flow state.

    The mean (bulk) velocity a mass-flow-controlled periodic channel targets. Reads the cell
    volumes from the assembler's geometry, so it is the same average the controller drives.
    """
    velocity, _ = momentum.unpack(flow)
    volume = momentum.geometry.cell.volume
    return jnp.sum(velocity[:, direction] * volume) / jnp.sum(volume)


def solve_segregated(
    momentum: MomentumContinuity,
    turbulence: SSTTurbulence,
    solve_flow: Callable[[MomentumContinuity, jnp.ndarray], jnp.ndarray],
    solve_scalar: Callable[..., jnp.ndarray],
    flow: jnp.ndarray,
    k: jnp.ndarray,
    omega: jnp.ndarray,
    *,
    density: float,
    max_sweeps: int,
    rtol: float = 1e-6,
    relaxation: float = 0.7,
    relaxation_max: float | None = None,
    ser_exponent: float = 1.0,
    k_floor: float = 1e-8,
    omega_floor: float = 1e-8,
    scalar_preconditioner: str | None = None,
    bulk_velocity_target: float | None = None,
    flow_direction: int = 0,
    bulk_velocity_gain: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the coupled RANS system by the segregated Picard loop.

    Parameters
    ----------
    momentum : MomentumContinuity
        The flow assembler; its viscosity property is replaced with the effective viscosity each
        sweep (its molecular value is ignored -- the molecular viscosity comes from ``turbulence``).
    turbulence : SSTTurbulence
        The k-omega SST closure and equation assembler.
    solve_flow : callable
        ``solve_flow(momentum, state) -> state`` solves the momentum residual of the (re-viscosified)
        assembler from ``state`` (e.g. a preconditioned Newton solve).
    solve_scalar : callable
        ``solve_scalar(residual, state, policy) -> state`` solves a scalar residual function from
        ``state``, globalizing it by pseudo-transient continuation. ``policy`` is the per-sweep
        :class:`~aquaflux.turbulence.continuation.ScalarShiftPolicy` (the transport shift diagonal plus
        the optional AMG) the driver builds; wire it with
        :func:`~aquaflux.turbulence.scalar_pseudo_transient_solve`.
    flow : jnp.ndarray
        The initial flat flow state ``[vel..., pressure]``, shape ``((dim + 1) n_cells,)``.
    k, omega : jnp.ndarray
        The initial turbulence fields, shape ``(n_cells,)`` (e.g. the inlet values, uniform).
    density : float
        The (constant) fluid density, forming ``mu_eff = rho (nu + nu_t)``.
    max_sweeps : int
        Upper bound on outer Picard sweeps. The loop normally stops earlier, when the coupled
        increment drops below ``rtol``; hitting this cap without converging emits a warning.
    rtol : float
        Outer convergence tolerance on the coupled Picard increment (the largest per-field relative
        change over a sweep). The loop exits the first sweep whose increment is below it.
    relaxation : float
        Under-relaxation factor for ``k`` and ``omega`` in ``(0, 1]``. This is the **floor** of the
        adaptive ramp -- the value used on the first sweep and never dropped below -- so set it to the
        largest factor that is safe from a cold start.
    relaxation_max : float or None
        Ceiling the adaptive relaxation ramps up to as the coupled increment falls (the SER schedule
        ``clip(relaxation (r0/r)^ser_exponent, relaxation, relaxation_max)``). ``None`` pins it to
        ``relaxation`` -- a constant under-relaxation, the plain Picard loop. A value above
        ``relaxation`` (up to ``1.0`` for a full step near convergence) accelerates the tail without
        risking the early sweeps.
    ser_exponent : float
        Exponent ``p`` on the increment-drop ratio in the relaxation ramp; larger opens the
        relaxation up faster as the coupling settles. Ignored when ``relaxation_max is None``.
    k_floor, omega_floor : float
        Positive floors applied to ``k`` and ``omega`` after each sweep.
    scalar_preconditioner : {"twolevel", "air"} or None
        When set, the per-sweep :class:`~aquaflux.turbulence.continuation.ScalarShiftPolicy` carries a
        convection-diffusion AMG (the given multigrid method) for its shifted-operator solve -- the
        mesh-independent scalar solve a high-Reynolds case needs. ``None`` is a shift-only
        (unpreconditioned) continuation solve.
    bulk_velocity_target : float or None
        When set, drive a streamwise-periodic channel to this **bulk velocity** by a mass-flow
        controller: after each sweep's flow solve the body force is rescaled toward the
        linear-response estimate that would hit the target,
        ``beta <- beta + gain (beta U_target / U_bulk - beta)`` (the standard mass-flow feedback for a
        periodic pressure drop, cast for the body-force formulation, where it is scale-free -- no gain
        tuning per Reynolds number). Requires
        ``momentum`` to carry a nonzero initial ``body_force`` along ``flow_direction`` and a
        ``pressure_pin``. ``None`` leaves the body force fixed.
    flow_direction : int
        The streamwise axis the bulk velocity is measured and the body force is applied along.
    bulk_velocity_gain : float
        Relaxation on the controller update in ``(0, 1]``: ``1.0`` takes the full linear-response
        step (exact in one sweep for Stokes flow), lower it if the bulk velocity oscillates under the
        nonlinear closure.

    Returns
    -------
    tuple of jnp.ndarray
        The ``(flow, k, omega)`` at the sweep that met ``rtol`` -- or at ``max_sweeps`` if it was not
        reached, in which case a warning is emitted and the fields may be under-converged.
    """
    molecular = turbulence.molecular_viscosity
    relaxation_ceiling = relaxation if relaxation_max is None else relaxation_max
    increment_0: float | None = None
    increment: float | None = None
    converged = False
    for _ in range(max_sweeps):
        # Adaptive under-relaxation: open from the floor toward the ceiling as the coupled increment
        # falls (SER ramp). The first sweep, and a constant-relaxation run, use the floor unchanged.
        sweep_relaxation = _sweep_relaxation(
            increment, increment_0, relaxation, relaxation_ceiling, ser_exponent
        )

        flow_prev, k_prev, omega_prev = flow, k, omega
        nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
        momentum = _with_viscosity(momentum, density * (molecular + nu_t))
        flow = solve_flow(momentum, flow)

        if bulk_velocity_target is not None:
            u_bulk = bulk_velocity(momentum, flow, flow_direction)
            beta = momentum.body_force[flow_direction]
            # Linear-response estimate of the force that hits the target, relaxed for the closure's
            # nonlinearity; scale-free, so no per-Reynolds gain tuning (unlike a raw additive step).
            beta_target = beta * bulk_velocity_target / jnp.maximum(u_bulk, 1e-12)
            new_beta = beta + bulk_velocity_gain * (beta_target - beta)
            new_force = momentum.body_force.at[flow_direction].set(new_beta)
            momentum = eqx.tree_at(lambda m: m.body_force, momentum, new_force)

        closure = turbulence.closure_fields(momentum.velocity_gradient(flow), k, omega)
        mdot = momentum.mass_flux(flow)
        # Each stiff reactive scalar is globalized by pseudo-transient continuation: the shift policy
        # bundles the transport shift diagonal with the (optional) AMG for the injected solve to use.
        k_policy = turbulence.k_shift_policy(mdot, closure, k, method=scalar_preconditioner)
        k_solved = solve_scalar(turbulence.k_residual(mdot, closure), k, k_policy)
        k = jnp.maximum(_relax(k, k_solved, sweep_relaxation), k_floor)
        omega_policy = turbulence.omega_shift_policy(
            mdot, closure, omega, method=scalar_preconditioner
        )
        omega_solved = solve_scalar(turbulence.omega_residual(mdot, closure), omega, omega_policy)
        omega = jnp.maximum(_relax(omega, omega_solved, sweep_relaxation), omega_floor)

        # Coupled Picard increment: the residual-agnostic outer convergence signal, also the ramp's
        # drop ratio. The first sweep sets the reference the ramp opens relative to.
        increment = _relative_change((flow_prev, flow), (k_prev, k), (omega_prev, omega))
        if increment_0 is None:
            increment_0 = increment
        if increment < rtol:
            converged = True
            break

    if not converged:
        last = "n/a" if increment is None else f"{increment:g}"
        warnings.warn(
            f"segregated coupling did not reach rtol={rtol:g} within max_sweeps={max_sweeps} "
            f"(last increment {last}); the returned fields may be under-converged.",
            stacklevel=2,
        )
    return flow, k, omega
