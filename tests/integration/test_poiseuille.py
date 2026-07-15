"""Coupled p--U validation: Poiseuille flow in a plane channel.

Fully-developed laminar flow between parallel no-slip walls, driven by a parabolic velocity
inlet and a pressure outlet, has the closed-form solution

    u(y) = u_max [ 1 - ((y - H/2)/(H/2))^2 ],   v = 0,   dp/dx = -8 mu u_max / H^2 (linear p).

This is the coupled analogue of the plane-wall gate for diffusion: it exercises the whole flow
substrate — momentum, the pressure-gradient coupling, and Rhie--Chow continuity — against an
analytical field. Because fully-developed flow has no convection, the system is linear (Stokes),
so Newton converges in one step; the velocity converges at second order and the reverse-mode
gradient flows through the coupled solve.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.materials import Constant, MaterialModel
from aquaflux.mesh import structured_grid_2d
from aquaflux.schemes import CorrectedGreenGauss
from aquaflux.solve import NewtonSolver

H, L, MU, RHO, UMAX = 1.0, 3.0, 0.1, 1.0, 1.0
DPDX = -8.0 * MU * UMAX / H**2


def _parabola(centroid):
    y = centroid[:, 1]
    u = UMAX * (1.0 - ((y - H / 2.0) / (H / 2.0)) ** 2)
    return jnp.stack([u, jnp.zeros_like(u)], axis=1)


def _solve(nx, ny, mu=MU):
    mesh = structured_grid_2d(nx, ny, lx=L, ly=H, named_boundaries=True)
    geom = mesh.geometry()
    cell_geometry = geom.cell
    assembler = MomentumContinuity.build(
        mesh,
        geom,
        MaterialModel({"viscosity": Constant(mu), "density": Constant(RHO)}),
        CorrectedGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=_parabola),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
    )
    state = NewtonSolver(iterations=3).solve(assembler.residual, assembler.initial_state())
    return mesh, cell_geometry, assembler, state


def test_poiseuille_reproduces_analytical_profile() -> None:
    """Velocity is parabolic, cross-flow ~0, and pressure is linear with the correct slope."""
    _, cell_geometry, assembler, state = _solve(32, 24)
    velocity, pressure = assembler.unpack(state)
    x = np.asarray(cell_geometry.centroid)[:, 0]
    y = np.asarray(cell_geometry.centroid)[:, 1]
    u_exact = UMAX * (1.0 - ((y - H / 2.0) / (H / 2.0)) ** 2)
    p_exact = (L - x) * (-DPDX)
    assert np.max(np.abs(np.asarray(velocity[:, 0]) - u_exact)) < 1e-2
    assert np.max(np.abs(np.asarray(velocity[:, 1]))) < 5e-3  # v ~ 0
    assert np.max(np.abs(np.asarray(pressure) - p_exact)) < 3e-2


def test_poiseuille_stokes_converges_in_one_newton_step() -> None:
    """Fully-developed flow has no convection, so the coupled system is linear (one step)."""
    _, _, assembler, _ = _solve(24, 16)
    one_step = NewtonSolver(iterations=1).solve(assembler.residual, assembler.initial_state())
    assert float(jnp.linalg.norm(assembler.residual(one_step))) < 1e-9


def test_poiseuille_solve_is_differentiable() -> None:
    """Reverse-mode gradient of a flow functional w.r.t. viscosity flows through the coupled solve."""

    def mean_speed(mu):
        _, _, assembler, state = _solve(16, 12, mu=mu)
        velocity, _ = assembler.unpack(state)
        return jnp.mean(velocity[:, 0])

    grad = float(jax.grad(mean_speed)(MU))
    assert np.isfinite(grad)


@pytest.mark.validation
def test_poiseuille_velocity_is_second_order() -> None:
    """The velocity field converges at second order under grid refinement."""
    errors = []
    for nx, ny in ((16, 12), (32, 24), (64, 48)):
        _, cell_geometry, assembler, state = _solve(nx, ny)
        velocity, _ = assembler.unpack(state)
        y = np.asarray(cell_geometry.centroid)[:, 1]
        u_exact = UMAX * (1.0 - ((y - H / 2.0) / (H / 2.0)) ** 2)
        errors.append(float(np.sqrt(np.mean((np.asarray(velocity[:, 0]) - u_exact) ** 2))))
    order = np.log2(errors[0] / errors[-1]) / np.log2(64 / 16)
    assert order > 1.8
