"""Unit tests for the flow coupling accessors used by scalar transport.

``MomentumContinuity.mass_flux`` must return exactly the Rhie--Chow mass flux the continuity
residual is built from (so a transported scalar advects on the same conservative flux), and
``velocity_gradient`` must return the tensor a turbulence model reads. Both are checked without a
solve -- they are functions of the state.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss


def _uniform_inlet(centroid):
    n = centroid.shape[0]
    return jnp.stack([jnp.ones(n), jnp.zeros(n)], axis=1)


def _assembler():
    mesh = structured_grid_2d(4, 3, lx=3.0, ly=1.0, named_boundaries=True)
    geom = mesh.geometry()
    asm = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(0.1), "density": Constant(1.0)}),
        CorrectedGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=_uniform_inlet),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
    )
    return mesh, asm


def _arbitrary_state(mesh):
    """A deterministic non-trivial flat state (mass-flux consistency holds for any state)."""
    n = (mesh.dim + 1) * mesh.n_cells
    return 0.2 + 0.1 * jnp.sin(jnp.arange(n, dtype=jnp.float64))


def test_mass_flux_is_the_continuity_mdot() -> None:
    """mass_flux(state) is exactly the mdot the continuity residual scatters (no pin here)."""
    mesh, asm = _assembler()
    state = _arbitrary_state(mesh)
    _, pressure_residual = asm.unpack(asm.residual(state))
    mdot = asm.mass_flux(state)
    assert jnp.allclose(pressure_residual, mesh.face_cells.scatter_conservative(mdot))


def test_velocity_gradient_shape_and_differentiable() -> None:
    """The tensor has shape (n_cells, dim, dim), is finite, and grad flows through it."""
    mesh, asm = _assembler()
    state = _arbitrary_state(mesh)
    grad = asm.velocity_gradient(state)
    assert grad.shape == (mesh.n_cells, mesh.dim, mesh.dim)
    assert not bool(jnp.any(jnp.isnan(grad)))
    g = jax.grad(lambda s: jnp.sum(asm.velocity_gradient(s) ** 2))(state)
    assert not bool(jnp.any(jnp.isnan(g)))
