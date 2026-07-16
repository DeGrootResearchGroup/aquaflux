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

The flow and scalar solvers are **injected** rather than chosen here: the coupled p--U flow needs a
preconditioned Newton solve while a single k/omega equation is better conditioned, and the choice
(preconditioner, iteration counts, adjoint) is the caller's to make. ``solve_flow(momentum, state)``
solves the momentum residual for the (re-viscosified) assembler, and ``solve_scalar(residual,
state)`` solves a scalar residual.
"""

from __future__ import annotations

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


def solve_segregated(
    momentum: MomentumContinuity,
    turbulence: SSTTurbulence,
    solve_flow: Callable[[MomentumContinuity, jnp.ndarray], jnp.ndarray],
    solve_scalar: Callable[[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray], jnp.ndarray],
    flow: jnp.ndarray,
    k: jnp.ndarray,
    omega: jnp.ndarray,
    *,
    density: float,
    sweeps: int,
    relaxation: float = 0.7,
    k_floor: float = 1e-8,
    omega_floor: float = 1e-8,
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
        ``solve_scalar(residual, state) -> state`` solves a scalar residual function from ``state``.
    flow : jnp.ndarray
        The initial flat flow state ``[vel..., pressure]``, shape ``((dim + 1) n_cells,)``.
    k, omega : jnp.ndarray
        The initial turbulence fields, shape ``(n_cells,)`` (e.g. the inlet values, uniform).
    density : float
        The (constant) fluid density, forming ``mu_eff = rho (nu + nu_t)``.
    sweeps : int
        Number of outer Picard sweeps.
    relaxation : float
        Under-relaxation factor for ``k`` and ``omega`` in ``(0, 1]``.
    k_floor, omega_floor : float
        Positive floors applied to ``k`` and ``omega`` after each sweep.

    Returns
    -------
    tuple of jnp.ndarray
        The converged ``(flow, k, omega)``.
    """
    molecular = turbulence.molecular_viscosity
    for _ in range(sweeps):
        nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
        momentum = _with_viscosity(momentum, density * (molecular + nu_t))
        flow = solve_flow(momentum, flow)

        closure = turbulence.closure_fields(momentum.velocity_gradient(flow), k, omega)
        mdot = momentum.mass_flux(flow)
        k_solved = solve_scalar(turbulence.k_residual(mdot, closure), k)
        k = jnp.maximum(_relax(k, k_solved, relaxation), k_floor)
        omega_solved = solve_scalar(turbulence.omega_residual(mdot, closure), omega)
        omega = jnp.maximum(_relax(omega, omega_solved, relaxation), omega_floor)
    return flow, k, omega
