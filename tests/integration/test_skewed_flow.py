"""Analytical validation of the coupled flow on a skewed (non-orthogonal) mesh.

A linear Couette velocity ``u = (y, 0)`` with constant pressure is an exact Stokes solution
(``div u = 0``, ``grad^2 u = 0``, so ``grad p = 0``). Because the Rhie--Chow mass flux and the
momentum pressure force reconstruct face values to the integration point — carrying the
``grad·(x_ip − x_g)`` skewness correction rather than stopping at the projection foot ``x_g`` — a
linear velocity is represented exactly, so the discrete divergence and the pressure force vanish at
the exact field and the solver reproduces it to solver tolerance even on a non-orthogonal mesh.

A plain owner/neighbour blend (the pre-correction behaviour) leaves an ``O(skew)`` error on such a
mesh, so the tight tolerance here is what distinguishes the integration-point reconstruction. All
boundaries are Dirichlet velocity, so the deferred boundary tangential correction does not enter.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall, VelocityInlet
from aquaflux.materials import Constant, MaterialModel
from aquaflux.schemes import CorrectedGreenGauss
from aquaflux.solve import NewtonSolver

from tests.support.meshes import perturbed_grid_2d


def _couette(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([x[:, 1], jnp.zeros(x.shape[0])], axis=1)  # u = (y, 0)


def _solve_couette(n: int = 8, perturb: float = 0.2, seed: int = 2):
    mesh = perturbed_grid_2d(
        n, n, lx=1.0, ly=1.0, perturb=perturb, seed=seed, named_boundaries=True
    )
    geom = mesh.geometry()
    assembler = MomentumContinuity.build(
        mesh,
        geom,
        MaterialModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CorrectedGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),  # u = (1, 0) at y = 1
                "bottom": NoSlipWall(),  # u = (0, 0) at y = 0
                "left": VelocityInlet(velocity=_couette),  # u = (y, 0)
                "right": VelocityInlet(velocity=_couette),  # u = (y, 0)
            }
        ),
        pressure_pin=0,  # closed domain (all velocity Dirichlet): fix the pressure level
    )
    # Stokes (no advection) so the residual is affine: one Newton step is exact.
    state = NewtonSolver(iterations=1).solve(assembler.residual, assembler.initial_state())
    return geom, assembler, state


def test_stokes_couette_is_exact_on_a_skewed_mesh() -> None:
    """The linear Couette velocity is reproduced to solver tolerance on a non-orthogonal mesh."""
    geom, assembler, state = _solve_couette()
    velocity, _ = assembler.unpack(state)
    u_exact = np.asarray(_couette(geom.cell.centroid))
    error = np.max(np.abs(np.asarray(velocity) - u_exact))
    assert error < 1e-6
