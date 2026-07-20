"""Validation: a fully-developed periodic channel reproduces the law of the wall.

The streamwise-periodic channel driven to a target bulk velocity by the mass-flow controller is the
canonical fully-developed case (unlike a spatially-developing inlet-to-outlet channel, whose
boundary layers never merge). At a wall-resolved, converged, turbulent state this asserts the three
law-of-the-wall facts:

1. the controller reaches the target bulk velocity (the body force self-adjusts);
2. the **viscous sublayer collapses onto** ``u+ = y+``;
3. the **log-law indicator** ``Xi = y+ dU+/dy+ = 1/kappa`` has a genuine plateau, with a realized
   ``kappa`` near the accepted value (a few percent below the nominal 0.41 -- the realized-vs-nominal
   gap of standard k-omega SST; see ``validation/turbulent_channel``).

Kept modest (``nx = 4`` since the flow is x-homogeneous; Re_tau ~ 1000) so it runs in a few minutes.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import NewtonSolver
from aquaflux.turbulence import (
    SSTModel,
    SSTTurbulence,
    bulk_velocity,
    hybrid_initialize,
    scalar_pseudo_transient_solve,
    solve_segregated,
)

# The scalar march's cap is a backstop, not a cost: the solver exits on tolerance, so a generous
# value only bounds the worst case (measured identical physics and wall time at 200 vs 500).
SCALAR_MAX_STEPS = 200

RHO, U_B, H = 1.0, 1.0, 2.0  # half-height h = 1


def _solve(Re_b=45000, ny=120, growth=1.075, beta0=0.0035, sweeps=100):
    nu = U_B * H / Re_b
    y = graded_nodes(ny, H, growth)
    mesh = structured_grid_2d(
        4, ny, lx=1.0, ly=H, periodic=("x",), named_boundaries=True, y_nodes=y
    )
    geom = mesh.geometry()
    model = SSTModel()
    momentum = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(RHO * nu), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
        body_force=(beta0, 0.0),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geom,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, nu),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions({"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)}),
        omega_boundary=BoundaryConditions({"bottom": ZeroGradient(), "top": ZeroGradient()}),
    )
    direct = lx.AutoLinearSolver(well_posed=True)

    def solve_flow(mom, state):
        return NewtonSolver(iterations=15, solver=direct).solve(mom.residual, state)

    # A uniform k leaves the first sweep's residual essentially unchanged for ~30 pseudo-transient
    # steps, so the SER schedule's beta never relaxes and the march exhausts its budget before the
    # residual moves. The hybrid IC descends from the first step (>200 steps -> 37 here).
    flow0, k0, omega0 = hybrid_initialize(momentum, turbulence)
    flow, k, omega = solve_segregated(
        momentum,
        turbulence,
        solve_flow,
        scalar_pseudo_transient_solve(max_steps=SCALAR_MAX_STEPS),
        flow0,
        k0,
        omega0,
        density=RHO,
        max_sweeps=sweeps,
        relaxation=0.9,
        bulk_velocity_target=U_B,
        flow_direction=0,
        bulk_velocity_gain=0.5,
    )
    return mesh, geom, momentum, turbulence, flow, k, omega, nu


@pytest.mark.validation
def test_periodic_channel_reproduces_the_law_of_the_wall() -> None:
    mesh, geom, momentum, turbulence, flow, k, omega, nu = _solve()

    # 1. The mass-flow controller reached the target bulk velocity.
    assert np.isclose(float(bulk_velocity(momentum, flow, 0)), U_B, atol=0.02)

    velocity, _ = momentum.unpack(flow)
    c = np.asarray(geom.cell.centroid)
    idx = np.arange(0, mesh.n_cells, 4)
    yc, u = c[idx, 1], np.asarray(velocity[:, 0])[idx]
    half = H / 2.0
    u_tau = float(np.sqrt(nu * u[0] / yc[0]))
    Re_tau = u_tau * half / nu
    assert Re_tau > 500.0  # a genuinely turbulent, wall-resolved channel

    below = yc <= half
    yplus, uplus = yc[below] * u_tau / nu, u[below] / u_tau
    assert yplus[0] < 1.0  # wall-resolved

    # 2. Viscous sublayer collapses onto u+ = y+ (checked below y+ = 4, above the pinned first cell).
    sub = (yplus > 1.0) & (yplus < 4.0)
    assert sub.sum() >= 2
    assert np.allclose(uplus[sub], yplus[sub], rtol=0.06)

    # 3. The log-law indicator has a genuine plateau with a realized kappa near the accepted value.
    xi = yplus * np.gradient(uplus, yplus)
    plateau = (yplus > 50.0) & (yplus < 0.2 * Re_tau)
    assert plateau.sum() >= 3
    kappa = 1.0 / float(np.min(xi[plateau]))
    assert 0.33 < kappa < 0.42  # realized SST value, a few percent below the nominal 0.41

    # Genuinely turbulent closure.
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    assert float(jnp.max(nu_t) / nu) > 10.0
