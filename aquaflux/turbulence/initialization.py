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

**A body-force-driven domain has no boundary values to interpolate.** A streamwise-periodic channel is
driven by a uniform force with every patch a wall, so the k interpolant is harmonic between all-zero
wall values (identically zero) and the omega solve is pure-Neumann with nothing in its interior. Taken
literally that starts the solve at ``k = 0`` -- not merely far from the answer but in the *laminar*
regime, which for a turbulent case is the wrong problem. Such a domain still has one velocity scale,
the friction velocity fixed by the global force balance
(:func:`~aquaflux.flow.scales.friction_velocity`), and the equilibrium levels it implies
(:func:`~aquaflux.turbulence.equilibrium_k` and a length scale) replace the degenerate interpolants.
An inlet-driven domain is unaffected: its friction velocity is zero, so the estimate never applies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquaflux.flow.initialization import laplace_field, potential_flow
from aquaflux.flow.scales import friction_velocity, hydraulic_length
from aquaflux.schemes import CompactGreenGauss

from .boundary import equilibrium_k, inlet_omega, omega_wall_value

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
    length_scale_factor: float = 0.09,
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
    length_scale_factor : float
        Outer turbulent mixing length as a fraction of the hydraulic length, used only for the
        body-force-driven equilibrium levels described above. ``0.09`` is the standard outer mixing
        length of wall-bounded turbulence (numerically equal to ``beta_star``, but unrelated to it),
        and it puts the initial eddy viscosity at the ``~0.09 u_tau h`` a developed channel carries.

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

    # A body-force-driven domain has no inlet, so both interpolants above are degenerate: k is
    # harmonic between all-zero wall values (identically zero), and omega is a pure-Neumann solve
    # whose interior carries nothing. Lifting k alone would be worse than either -- nu_t = k/omega
    # with omega at its floor is enormous -- so both levels come from the same place: the friction
    # velocity the force balance fixes, which is the one velocity scale such a domain always has.
    # Zero for an inlet-driven domain, where `maximum` then leaves the interpolants untouched.
    u_tau = friction_velocity(momentum)
    k_equilibrium = equilibrium_k(u_tau, turbulence.model)
    # The outer mixing length of wall-bounded turbulence, so k/omega lands at the ~0.09 u_tau h
    # eddy viscosity a developed channel carries rather than several times it.
    length_scale = length_scale_factor * hydraulic_length(momentum)
    if float(u_tau) > 0.0:
        k = jnp.maximum(k, k_equilibrium)
        # The wall cells keep their analytical value, which is far larger than this core level.
        omega = jnp.maximum(omega, inlet_omega(k_equilibrium, length_scale, turbulence.model))

    k = jnp.maximum(k, k_floor)
    omega = jnp.maximum(omega, omega_floor)
    return flow, k, omega
