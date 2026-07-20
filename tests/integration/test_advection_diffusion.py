"""First-order upwind advection in the solve: the 1-D advection-diffusion boundary layer.

Steady advection-diffusion ``U dphi/dx = Gamma d^2phi/dx^2`` on ``[0, 1]`` with ``phi(0) = 0``,
``phi(1) = 1`` has the exponential boundary-layer solution

    phi(x) = (exp(Pe x) - 1) / (exp(Pe) - 1),   Pe = U L / Gamma,

a standard verification case for advection schemes. First-order upwind is linear (so the
residual stays affine and Newton converges in one step) and unconditionally bounded (monotone,
no over/undershoot), at the cost of first-order accuracy from its numerical diffusion.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import (
    AdvectionFlux,
    DiffusionFlux,
    FirstOrderUpwind,
    ResidualAssembler,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.solve import newton_step

from tests.support.fields import face_mass_flux

U = 1.0
GAMMA = 0.1
PE = U / GAMMA


def _exact(x: np.ndarray) -> np.ndarray:
    return (np.exp(PE * x) - 1.0) / (np.exp(PE) - 1.0)


def _solve(nx):
    mesh = structured_grid_2d(nx, 1, lx=1.0, ly=1.0 / nx, named_boundaries=True)
    geom = mesh.geometry()
    mdot = face_mass_flux(geom.face, jnp.array([U, 0.0]))
    assembler = ResidualAssembler.build(
        mesh,
        geom,
        PropertyModel({"diffusivity": Constant(GAMMA)}),
        (AdvectionFlux(mass_flux=mdot, scheme=FirstOrderUpwind()), DiffusionFlux()),
        BoundaryConditions(
            {
                "left": Dirichlet(0.0),  # inflow
                "right": Dirichlet(1.0),  # outflow (imposed weakly through diffusion)
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    phi = eqx.filter_jit(newton_step)(assembler.residual, jnp.zeros(mesh.n_cells))
    return mesh, geom.cell, assembler, phi


def test_upwind_one_newton_step_and_bounded() -> None:
    """One Newton step converges (linear), and the solution is bounded in [0, 1] (monotone)."""
    _, _, assembler, phi = _solve(40)
    assert float(jnp.linalg.norm(assembler.residual(phi))) < 1e-9
    assert float(jnp.min(phi)) >= -1e-9
    assert float(jnp.max(phi)) <= 1.0 + 1e-9


def test_upwind_matches_exponential_solution() -> None:
    """The solution tracks the analytical boundary-layer profile (to first-order accuracy)."""
    _, cell_geometry, _, phi = _solve(80)
    x = np.asarray(cell_geometry.centroid)[:, 0]
    err = float(np.sqrt(np.mean((np.asarray(phi) - _exact(x)) ** 2)))
    assert err < 2e-2


def test_upwind_solve_is_differentiable() -> None:
    """jax.grad flows through the advection-diffusion solve without NaNs."""

    def mean_phi(gamma):
        mesh = structured_grid_2d(40, 1, lx=1.0, ly=1.0 / 40, named_boundaries=True)
        geom = mesh.geometry()
        mdot = face_mass_flux(geom.face, jnp.array([U, 0.0]))
        assembler = ResidualAssembler.build(
            mesh,
            geom,
            PropertyModel({"diffusivity": Constant(gamma)}),
            (AdvectionFlux(mass_flux=mdot, scheme=FirstOrderUpwind()), DiffusionFlux()),
            BoundaryConditions(
                {
                    "left": Dirichlet(0.0),
                    "right": Dirichlet(1.0),
                    "bottom": ZeroGradient(),
                    "top": ZeroGradient(),
                }
            ),
        )
        return jnp.mean(eqx.filter_jit(newton_step)(assembler.residual, jnp.zeros(mesh.n_cells)))

    grad = jax.grad(mean_phi)(GAMMA)
    assert np.isfinite(float(grad))


@pytest.mark.validation
def test_upwind_is_first_order_accurate() -> None:
    """Grid refinement gives first-order convergence — the upwind numerical-diffusion cap."""
    errors = []
    for nx in (20, 40, 80, 160):
        _, cell_geometry, _, phi = _solve(nx)
        x = np.asarray(cell_geometry.centroid)[:, 0]
        errors.append(float(np.sqrt(np.mean((np.asarray(phi) - _exact(x)) ** 2))))
    order = np.log2(errors[0] / errors[-1]) / np.log2(160 / 20)
    assert 0.8 < order < 1.3  # first order, approached from below at finite Pe
