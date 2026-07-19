"""Hybrid initial condition for the coupled RANS solve -- potential velocity + smoothed turbulence.

The monolithic coupled Newton solve (:func:`~aquaflux.turbulence.solve_coupled`) is a local method: it
converges quadratically once in the basin, but from a raw ``u=0, k=k_in, omega=omega_in`` cold start it
stalls -- the near-wall ``omega`` fixation alone puts a ``~60 nu / (beta_1 d^2)`` jump into the residual,
and a uniform interior is far from a consistent field. This module builds a **cheap, physical** initial
state (a few linear Laplace solves) that lands directly in the basin, so the coupled solve self-starts
from nothing. It is the Fluent-style "hybrid initialization" specialized to the k-omega SST fields:

- **velocity** -- potential flow ``u = grad phi`` (:func:`~aquaflux.flow.potential_flow`), respecting the
  through-flow and geometry;
- **k** -- the harmonic interpolant of its boundary values (``k_in`` at the inlet decaying to ``0`` at
  the walls);
- **omega** -- its boundary-propagated interior value with the near-wall cells set to the analytical
  wall value ``omega_wall = 60 nu / (beta_1 d^2)`` (a Laplace-smoothed ``omega`` instead over-diffuses
  that large wall value into the interior and slows the solve, so only the wall cells are set).

Each field is one linear SPD solve -- together far cheaper than a single coupled Newton iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquaflux.flow.initialization import laplace_field, potential_flow
from aquaflux.schemes import CompactGreenGauss

from .boundary import omega_wall_value

if TYPE_CHECKING:
    from aquaflux.flow import MomentumContinuity
    from aquaflux.schemes import GradientScheme

    from .transport import SSTTurbulence


def hybrid_initialize(
    momentum: MomentumContinuity,
    turbulence: SSTTurbulence,
    *,
    gradient_scheme: GradientScheme | None = None,
    k_floor: float = 1e-8,
    omega_floor: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a hybrid initial ``(flow, k, omega)`` that lets the coupled RANS solve self-start.

    Parameters
    ----------
    momentum : MomentumContinuity
        The flow assembler (supplies the potential-flow boundary data and mesh).
    turbulence : SSTTurbulence
        The SST closure (supplies the k/omega boundary conditions, the near-wall cells, and the wall
        distance / viscosity that set the analytical ``omega`` wall value).
    gradient_scheme : GradientScheme or None
        The scheme for the Laplace solves' gradient reconstruction (defaults to
        :class:`~aquaflux.schemes.CompactGreenGauss`).
    k_floor, omega_floor : float
        Positive floors applied to the smoothed fields, matching the solver's realizability floors.

    Returns
    -------
    tuple of jnp.ndarray
        ``(flow, k, omega)`` -- the flat flow state ``((dim + 1) n_cells,)`` and the two fields
        ``(n_cells,)`` -- ready to hand to :func:`~aquaflux.turbulence.solve_coupled`.
    """
    mesh, geometry = momentum.mesh, momentum.geometry
    gradient_scheme = gradient_scheme or CompactGreenGauss()

    flow = potential_flow(momentum, gradient_scheme=gradient_scheme)

    k, _ = laplace_field(mesh, geometry, turbulence.k_boundary, gradient_scheme=gradient_scheme)
    omega, _ = laplace_field(
        mesh, geometry, turbulence.omega_boundary, gradient_scheme=gradient_scheme
    )
    wall_values = omega_wall_value(
        turbulence.molecular_viscosity[turbulence.wall_cells],
        turbulence.wall_distance[turbulence.wall_cells],
        turbulence.model,
    )
    omega = omega.at[turbulence.wall_cells].set(wall_values)

    k = jnp.maximum(k, k_floor)
    omega = jnp.maximum(omega, omega_floor)
    return flow, k, omega
