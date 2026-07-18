"""Integration: a streamwise-periodic channel driven by a body force is fully-developed Poiseuille.

Two no-slip walls, streamwise-periodic in x, driven by a uniform body force ``beta`` per unit
volume (the mean pressure gradient, ``G = -beta``, carried as a force rather than a boundary
pressure drop). The exact fully-developed laminar solution is

    u(y) = (beta / 2 mu) y (H - y),   u_max = beta H^2 / (8 mu),   v = 0,

with the flow **independent of x** and the periodic pressure ``p_tilde`` streamwise-homogeneous
(the entire drop lives in the body force). Fully-developed flow has no convection, so the system is
linear (Stokes) and Newton converges in one step. This exercises the whole periodic-connectivity
path -- the seam face's cell geometry, diffusion, Rhie--Chow continuity, and the body-force source
-- against an analytical field, and is the laminar prerequisite for the turbulent-channel validation.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import MomentumContinuity, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss
from aquaflux.solve import NewtonSolver

H, LX, MU, RHO, BETA = 1.0, 2.0, 0.1, 1.0, 0.1
U_MAX = BETA * H**2 / (8.0 * MU)


def _solve(nx, ny):
    mesh = structured_grid_2d(nx, ny, lx=LX, ly=H, periodic=("x",), named_boundaries=True)
    geometry = mesh.geometry()
    assembler = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(MU), "density": Constant(RHO)}),
        CorrectedGreenGauss(),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        pressure_pin=0,  # periodic + walls is a closed domain: fix the pressure datum
        body_force=(BETA, 0.0),
    )
    state = NewtonSolver(iterations=3).solve(assembler.residual, assembler.initial_state())
    return mesh, geometry, assembler, state


@pytest.mark.validation
def test_periodic_channel_is_fully_developed_poiseuille() -> None:
    nx, ny = 6, 32
    _, geometry, assembler, state = _solve(nx, ny)
    velocity, pressure = assembler.unpack(state)
    y = np.asarray(geometry.cell.centroid)[:, 1]
    u = np.asarray(velocity[:, 0])
    v = np.asarray(velocity[:, 1])

    u_exact = (BETA / (2.0 * MU)) * y * (H - y)
    assert np.max(np.abs(u - u_exact)) < 1e-3  # second-order accurate parabola
    assert np.isclose(float(u.max()), U_MAX, atol=2e-3)  # analytic peak beta H^2 / 8 mu
    assert np.max(np.abs(v)) < 1e-10  # no cross-flow

    # Fully developed: u and the periodic pressure are x-homogeneous (the drop is in the body force,
    # not in p_tilde). Cells are row-major, so a column at fixed y spans x as u.reshape(ny, nx)[j].
    u_grid = u.reshape(ny, nx)
    p_grid = np.asarray(pressure).reshape(ny, nx)
    assert np.max(u_grid.max(axis=1) - u_grid.min(axis=1)) < 1e-10  # u independent of x
    assert np.max(p_grid.max(axis=1) - p_grid.min(axis=1)) < 1e-10  # dp_tilde/dx ~ 0


@pytest.mark.validation
def test_periodic_channel_converges_second_order() -> None:
    """Refining the wall-normal resolution halves the profile error at ~second order."""
    errors = []
    for ny in (16, 32):
        _, geometry, assembler, state = _solve(4, ny)
        velocity, _ = assembler.unpack(state)
        y = np.asarray(geometry.cell.centroid)[:, 1]
        u_exact = (BETA / (2.0 * MU)) * y * (H - y)
        errors.append(float(np.max(np.abs(np.asarray(velocity[:, 0]) - u_exact))))
    assert errors[1] < errors[0] / 3.0  # ~4x drop for 2x refinement; 3x is a safe floor
