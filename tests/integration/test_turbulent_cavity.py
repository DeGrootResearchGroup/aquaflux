"""Integration: the segregated k-omega SST driver on a lid-driven cavity.

A stability / plumbing check that the whole model solves end to end. The lid-driven cavity is the
validated, robust nonlinear flow case; here the lid shear generates the turbulence (all four
boundaries are walls), so the driver runs with no inlet. It checks that the outer loop stays stable,
the turbulence fields stay positive and finite, the eddy viscosity is active, and the flow develops.
Quantitative accuracy against a turbulent profile is a separate validation concern.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import BlockPreconditioner, MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver
from aquaflux.turbulence import (
    SSTModel,
    SSTTurbulence,
    scalar_pseudo_transient_solve,
    solve_segregated,
)

RHO, NU, U_LID = 1.0, 1e-2, 1.0
WALLS = ("top", "bottom", "left", "right")


def _solve_flow(momentum, state):
    """The validated preconditioned Newton solve of the coupled cavity flow.

    The driver's flow seam returns ``(assembler, state)``; this unconstrained solve leaves the
    assembler unchanged, so it is passed through.
    """
    preconditioner = BlockPreconditioner.build(momentum).factory()
    solver = ImplicitNewtonSolver(
        max_steps=30, forward_step=DampedNewtonStep(preconditioner=preconditioner)
    )
    return momentum, solver.solve(lambda s, m: m.residual(s), state, momentum)


def _cavity():
    mesh = structured_grid_2d(16, 16, lx=1.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(U_LID, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=list(WALLS),
        k_boundary=BoundaryConditions({w: Dirichlet(0.0) for w in WALLS}),
        omega_boundary=BoundaryConditions({w: ZeroGradient() for w in WALLS}),
    )
    return mesh, momentum, turbulence


@pytest.mark.slow
def test_segregated_cavity_is_stable_and_active() -> None:
    """The segregated driver solves the coupled cavity end to end, the stiff k/omega solves globalized
    by pseudo-transient continuation -- stable, positive/finite fields, active eddy viscosity, developed
    flow."""
    mesh, momentum, turbulence = _cavity()
    flow, k, omega = solve_segregated(
        momentum,
        turbulence,
        _solve_flow,
        scalar_pseudo_transient_solve(max_steps=40),
        momentum.initial_state(),
        jnp.full(mesh.n_cells, 1e-4),  # seed k > 0 so the shear production can start
        jnp.full(mesh.n_cells, 1.0),
        density=RHO,
        max_sweeps=10,
        scalar_preconditioner="twolevel",
    )
    assert not bool(jnp.any(jnp.isnan(flow)))
    assert not bool(jnp.any(jnp.isnan(k)))
    assert not bool(jnp.any(jnp.isnan(omega)))
    assert float(jnp.min(k)) >= 0.0
    assert float(jnp.min(omega)) > 0.0
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    assert float(jnp.max(nu_t)) > 0.0
    velocity, _ = momentum.unpack(flow)
    assert float(jnp.max(jnp.abs(velocity[:, 0]))) > 0.3 * U_LID
