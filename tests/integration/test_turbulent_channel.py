"""Validation: the SST eddy viscosity is active and couples to the flow on an open channel.

A plane channel (uniform velocity inlet, pressure outlet, no-slip walls) solved with the segregated
driver. The open-channel flow solver is robust at a moderate Reynolds number (Re ~ 100), where a
turbulent boundary layer is not yet developed, so this validates the model's *coupling* rather than
a high-Re profile: with elevated free-stream turbulence the closure produces a significant eddy
viscosity (nu_t >> nu), and that eddy viscosity measurably diffuses the momentum -- the turbulent
velocity departs from the laminar one and does not exceed its peak. A quantitative high-Re turbulent
profile awaits a flow solver that converges the open channel at higher Reynolds number.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    BlockPreconditioner,
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import ImplicitNewtonSolver, NewtonSolver
from aquaflux.turbulence import SSTModel, SSTTurbulence, inlet_k, inlet_omega, solve_segregated

RHO, NU, U_IN, H, L = 1.0, 1e-2, 1.0, 1.0, 4.0  # Re = rho U H / mu = 100
INTENSITY, LENGTH_SCALE = 0.4, 0.3  # elevated free-stream turbulence so nu_t is significant


def _solve_flow(momentum, state):
    """The globalized, preconditioned Newton solve of the coupled open-channel flow."""
    preconditioner = BlockPreconditioner.build(momentum).factory()
    solver = ImplicitNewtonSolver(max_steps=40, preconditioner=preconditioner)
    return solver.solve(lambda s, m: m.residual(s), state, momentum)


def _solve_scalar(residual, state):
    return NewtonSolver(iterations=6).solve(residual, state)


def _channel():
    mesh = structured_grid_2d(16, 8, lx=L, ly=H, named_boundaries=True)
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, model))
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(U_IN, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(k_in),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(omega_in),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return mesh, momentum, turbulence, k_in, omega_in


@pytest.mark.validation
def test_channel_eddy_viscosity_is_active_and_couples_to_the_flow() -> None:
    mesh, momentum, turbulence, k_in, omega_in = _channel()
    laminar = _solve_flow(momentum, momentum.initial_state())
    flow, k, omega = solve_segregated(
        momentum,
        turbulence,
        _solve_flow,
        _solve_scalar,
        momentum.initial_state(),
        jnp.full(mesh.n_cells, k_in),
        jnp.full(mesh.n_cells, omega_in),
        density=RHO,
        sweeps=12,
    )
    # Self-consistent: finite and positive.
    assert not bool(jnp.any(jnp.isnan(flow)))
    assert float(jnp.min(k)) >= 0.0
    assert float(jnp.min(omega)) > 0.0

    # The closure produces a significant eddy viscosity (turbulent mixing, not a rounding effect).
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    assert float(jnp.max(nu_t) / NU) > 1.0

    # That eddy viscosity couples to the momentum: the turbulent velocity departs measurably from the
    # laminar one, and the added diffusion does not accelerate the peak above laminar.
    u_laminar = momentum.unpack(laminar)[0][:, 0]
    u_turbulent = momentum.unpack(flow)[0][:, 0]
    assert float(jnp.max(jnp.abs(u_turbulent - u_laminar))) > 1e-3
    assert float(jnp.max(u_turbulent)) <= float(jnp.max(u_laminar)) + 1e-6
