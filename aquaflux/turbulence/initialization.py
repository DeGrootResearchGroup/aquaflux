"""Hybrid initial condition for the coupled RANS solve -- potential velocity + smoothed turbulence.

The monolithic coupled Newton solve (:func:`~aquaflux.turbulence.solve_coupled`) is a local method: it
converges quadratically once in the basin, but from a raw ``u=0, k=k_in, omega=omega_in`` cold start it
stalls -- the near-wall ``omega`` fixation alone puts a ``~6 nu / (beta_1 d^2)`` jump into the residual,
and a uniform interior is far from a consistent field. This module builds a **cheap, physical** initial
state (a few linear Laplace solves) that lands directly in the basin, so the coupled solve self-starts
from nothing. It is the Fluent-style "hybrid initialization" specialized to the k-omega SST fields:

- **velocity** -- potential flow ``u = grad phi`` (:func:`~aquaflux.flow.potential_flow`), respecting the
  through-flow and geometry;
- **k** -- the harmonic interpolant of its boundary values, then **floored to a turbulent level** so the
  interior does not start laminar (the interpolant otherwise collapses toward the wall ``k = 0`` -- see
  below);
- **omega** -- its boundary-propagated interior value raised, at every cell's own wall distance, to the
  same near-wall closure the residual imposes at the wall: the adaptive blend
  ``sqrt(omega_vis^2 + omega_log^2)`` (``omega_vis = 6 nu / (beta_1 y^2)``). That profile is the exact
  solution of the near-wall balance ``nu d2(omega)/dy2 = beta_1 omega^2``, so each near-wall cell starts
  on the same analytical decay curve; a Laplace-smoothed ``omega`` instead over-diffuses the large wall
  value into the interior, while setting only the wall cells leaves a cliff to the flat interpolant that
  concentrates almost all of the initial ``omega`` residual in the wall-adjacent cell (seeding the
  profile everywhere roughly halves that initial residual). It is also smooth in the log variable
  ``w = log omega``, where the profile is the ramp ``w(y) = log(6 nu / beta_1) - 2 log(y)`` rather than
  the ``~log(omega_wall / omega_core)`` cliff a wall-cells-only seed leaves -- the ramp's largest
  cross-cell step is set by the mesh growth ratio, where the cliff grows with Reynolds number as the
  wall spacing shrinks.

Each field is one linear SPD solve -- together far cheaper than a single coupled Newton iteration.

**A body-force-driven domain has no boundary values to interpolate.** A streamwise-periodic channel is
driven by a uniform force with every patch a wall, so the k interpolant is harmonic between all-zero
wall values (identically zero) and the omega solve is pure-Neumann with nothing in its interior. Taken
literally that starts the solve at ``k = 0`` -- not merely far from the answer but in the *laminar*
regime, which for a turbulent case is the wrong problem. Such a domain still has one velocity scale,
the friction velocity fixed by the global force balance
(:func:`~aquaflux.flow.scales.friction_velocity`), and the equilibrium levels it implies
(:func:`~aquaflux.turbulence.equilibrium_k` and a length scale) replace the degenerate interpolants.

An **inlet-driven wall-bounded domain has the same failure for a subtler reason**: it does have an
inlet, but the walls carry ``k = 0`` over the whole domain and dominate the small inlet patch by area,
so the harmonic interpolant still collapses toward zero a few channel heights downstream (median ``k``
orders of magnitude below ``k_in``). Its friction velocity is zero, so the equilibrium estimate does not
apply; instead ``k`` is floored at the inlet turbulence level -- the interpolant's own maximum, which by
the maximum principle is the peak boundary (inlet) value. ``omega`` needs no such floor in either case:
its walls are zero-gradient, not Dirichlet-``0``, so its interpolant stays at ~``omega_in`` and does not
collapse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquaflux.flow.initialization import laplace_field, potential_flow
from aquaflux.flow.scales import friction_velocity, hydraulic_length
from aquaflux.schemes import CompactGreenGauss

from .boundary import equilibrium_k, inlet_omega, omega_wall

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
        The SST closure (supplies the k/omega boundary conditions and the per-cell wall distance /
        viscosity that set the analytical near-wall ``omega`` profile).
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
    # Give the interior a turbulent-level k, or the coupled solve starts laminar. The k interpolant
    # is pulled toward its wall Dirichlet(0) values, and because the walls dominate a wall-bounded
    # domain by area the interior k collapses toward zero -- a channel only a few heights long already
    # has a median k orders of magnitude below the inlet level. That is the *laminar* field
    # (nu_t = k/omega ~ 0), so a turbulent case must then grow k across the whole interior, a swing the
    # coupled Newton must absorb. (omega does not need the same treatment: its walls are zero-gradient,
    # not Dirichlet-0, so its interpolant stays at ~omega_in and does not collapse.) The turbulent level
    # comes from whatever drives the flow:
    u_tau = friction_velocity(momentum)
    if float(u_tau) > 0.0:
        # Body-force-driven domain: no inlet to read a level from, and every wall is k = 0, so the
        # interpolant is ~0 everywhere. Set k -- and omega with it, to keep nu_t = k/omega sane -- from
        # the equilibrium turbulence the friction velocity the force balance fixes implies. The outer
        # mixing length 0.09 h lands k/omega at the ~0.09 u_tau h eddy viscosity a developed channel
        # carries rather than several times it.
        k_equilibrium = equilibrium_k(u_tau, turbulence.model)
        length_scale = length_scale_factor * hydraulic_length(momentum)
        k = jnp.maximum(k, k_equilibrium)
        # The near-wall cells keep their analytical profile, far larger than this core level.
        omega = jnp.maximum(omega, inlet_omega(k_equilibrium, length_scale, turbulence.model))
    else:
        # Inlet-driven domain: the interpolant carries the inlet k only near the inlet and collapses
        # downstream. Floor the whole interior at the inlet turbulence level -- by the maximum principle
        # the peak boundary k is the interpolant's maximum -- so nu_t starts turbulent across the
        # domain. With omega already at ~omega_in in the core, (k_in, omega_in) is the consistent
        # inlet-level eddy viscosity. A domain with no inlet turbulence (all-zero k boundaries) has a
        # ~0 interpolant, so this lifts nothing and the k_floor below carries positivity.
        k = jnp.maximum(k, jnp.max(k))

    k = jnp.maximum(k, k_floor)

    # Seed the near-wall omega profile on EVERY cell, at its own wall distance, using the SAME closure
    # the residual imposes at the wall (`omega_wall`) -- the adaptive blend
    # sqrt(omega_vis^2 + omega_log^2), not the viscous branch alone. Matching the closure is the whole
    # point: where the seed and the wall condition disagree, the wall-adjacent cells start off their own
    # boundary condition and carry a large initial omega residual, which is exactly what this seeding is
    # meant to remove. The two agree as y+ -> 0 (the blend reduces to the viscous 6 nu / (beta_1 y^2)),
    # so a wall-resolved mesh is unaffected; they diverge on a wall-function mesh, where the log branch
    # dominates -- ~3x at y+ = 30 and ~10x at y+ = 100.
    #
    # Seeded everywhere rather than only on the wall-adjacent cells: setting only those leaves a cliff
    # between the fixed wall cell (large omega) and its neighbour on the flat interpolant, and that
    # neighbour's omega equation then carries almost the entire initial residual. The profile falls off
    # with distance, so a few cells out it drops below the interpolant and the maximum leaves the core
    # untouched. It is also smooth in the log variable w = log omega -- a ramp rather than the
    # ~log(omega_wall / omega_core) cliff a wall-cells-only seed leaves across the first cell.
    #
    # This must come after k is settled: the blend's log branch reads sqrt(k), so seeding it against the
    # bare Laplace interpolant (k ~ 0 in the interior of a wall-bounded domain) would evaluate the
    # closure at a k the solve never sees.
    near_wall_omega = omega_wall(
        turbulence.molecular_viscosity, turbulence.wall_distance, k, turbulence.model
    )
    omega = jnp.maximum(omega, near_wall_omega)
    omega = jnp.maximum(omega, omega_floor)
    return flow, k, omega
