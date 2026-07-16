"""Open-channel flow: the coupled p--U solve on an inlet/outlet domain with convection.

A plane channel driven by a **uniform** velocity inlet against no-slip walls, closed by a pressure
outlet, with first-order-upwind convection (so the coupled residual is nonlinear). Unlike the closed
lid-driven cavity (:mod:`tests.integration.test_cavity`) and the Stokes Poiseuille channel
(:mod:`tests.integration.test_poiseuille`, no advection), this exercises the two things an *open*
convective domain needs that a closed one hides:

* a **globalized** Newton step — the undamped full step overshoots from the uniform initial field
  and diverges, so :class:`ImplicitNewtonSolver` damps it with a backtracking line search; and
* a **non-singular pressure Schur** in the block preconditioner — its interior part is a pure-Neumann
  Laplacian, de-singularised here by the pressure-outlet boundary coupling
  (:meth:`~aquaflux.flow.PressureOutlet.pressure_schur_coefficient`).

Both are required: with either missing the solve stagnates in the inner GMRES.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
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
from aquaflux.solve import ImplicitNewtonSolver

H, L, U_IN, RHO = 1.0, 4.0, 1.0, 1.0
MU = 0.01  # Re = rho U H / mu = 100


def _channel(nx=16, ny=8, mu=MU):
    mesh = structured_grid_2d(nx, ny, lx=L, ly=H, named_boundaries=True)
    geom = mesh.geometry()
    return MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
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


def _solve(assembler, precond=None, **kwargs):
    if precond is None:
        precond = BlockPreconditioner.build(assembler).factory()
    solver = ImplicitNewtonSolver(max_steps=30, preconditioner=precond, **kwargs)
    return solver.solve(lambda s, a: a.residual(s), assembler.initial_state(), assembler)


def test_open_channel_converges() -> None:
    """The convective open channel drives to a converged flow (residual ~ 0)."""
    assembler = _channel()
    state = _solve(assembler)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-8


def test_open_channel_conserves_mass() -> None:
    """Net boundary mass flux is zero: what the inlet delivers leaves through the outlet."""
    assembler = _channel()
    state = _solve(assembler)
    mdot = np.asarray(assembler.mass_flux(state))  # owner-outward per face
    patches = assembler.mesh.face_patches
    inlet = mdot[np.asarray(patches.indices("left"))].sum()
    outlet = mdot[np.asarray(patches.indices("right"))].sum()
    # Owner-outward: the inlet flux is negative (into the domain), the outlet positive; they balance.
    assert inlet < 0.0 < outlet
    assert abs(inlet + outlet) < 1e-6  # global continuity (a sum over cells of the ~1e-8 residual)
    assert abs(abs(inlet) - RHO * U_IN * H) < 1e-8  # inlet delivers rho U H (prescribed exactly)


def test_line_search_is_necessary() -> None:
    """The globalization is *necessary*: the undamped full Newton step fails on this open convective
    flow — it overshoots from the uniform initial field and either diverges or stagnates the inner
    linear solve at the overshot iterate — and the backtracking line search recovers convergence."""
    assembler = _channel()
    try:
        undamped = _solve(assembler, line_search=0)  # pure full Newton
        residual = float(jnp.linalg.norm(assembler.residual(undamped)))
        undamped_failed = (not np.isfinite(residual)) or residual > 1e-3  # diverged
    except Exception:
        undamped_failed = True  # the overshot iterate stagnated the inner GMRES
    assert undamped_failed

    damped = _solve(assembler)  # default line search on
    assert float(jnp.linalg.norm(assembler.residual(damped))) < 1e-8


def test_open_channel_solve_is_differentiable() -> None:
    """Reverse-mode gradient of a scalar objective through the open-channel solve is finite (the IFT
    adjoint flows through the globalized, preconditioned solve)."""

    # Build the (stop_gradient-ed) preconditioner once from a concrete-viscosity assembler and reuse
    # it across mu, as the cavity adjoint test does — it only accelerates the Krylov iteration.
    precond = BlockPreconditioner.build(_channel()).factory()

    def mean_speed(mu):
        assembler = _channel(mu=mu)
        state = _solve(assembler, precond=precond)
        velocity, _ = assembler.unpack(state)
        return jnp.mean(jnp.abs(velocity[:, 0]))

    grad = float(jax.grad(mean_speed)(MU))
    assert np.isfinite(grad)


@pytest.mark.validation
def test_open_channel_accelerates_core_flow() -> None:
    """Physical sanity: the no-slip walls slow the near-wall fluid, so continuity accelerates the
    core above the uniform inlet speed (peak centreline velocity exceeds U_in)."""
    assembler = _channel(nx=32, ny=16)
    state = _solve(assembler)
    velocity, _ = assembler.unpack(state)
    assert float(jnp.max(velocity[:, 0])) > 1.1 * U_IN
