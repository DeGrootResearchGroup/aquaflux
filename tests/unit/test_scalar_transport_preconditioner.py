"""The convection-diffusion AMG preconditioner for a scalar transport equation.

Exercises :func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner` on a synthetic
convection-dominated advection-diffusion scalar (a uniform streamwise flux, no turbulence needed), the
same operator family the k/omega transport linearizes to. The preconditioner must reconstruct that
operator well enough that a single V-cycle strongly contracts the error, and the contraction must stay
bounded as the mesh refines (mesh-independent) -- the property that keeps the scalar solve's iteration
count from growing with problem size at high cell Peclet.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import (
    AdvectionFlux,
    DiffusionFlux,
    FirstOrderUpwind,
    ResidualAssembler,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.turbulence.preconditioner import scalar_transport_preconditioner

U, GAMMA = 1.0, 1e-3  # cell Peclet ~ U dx / Gamma is large -> convection-dominated


def _transport(nx, ny):
    """A scalar advected by a uniform streamwise flux and diffusing with constant ``GAMMA``."""
    mesh = structured_grid_2d(nx, ny, lx=4.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    volume_flux = U * geometry.face.normal[:, 0] * geometry.face.area  # uniform u = (U, 0)
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": Constant(GAMMA)}),
        (AdvectionFlux(volume_flux, FirstOrderUpwind()), DiffusionFlux()),
        BoundaryConditions(
            {
                "left": Dirichlet(1.0),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
    )
    return mesh, geometry, volume_flux, assembler.residual


def _contraction(nx, ny, method, *, seed=0):
    mesh, geometry, volume_flux, residual = _transport(nx, ny)
    reference = jnp.ones(mesh.n_cells)
    m = scalar_transport_preconditioner(
        mesh,
        geometry,
        jnp.full(mesh.n_cells, GAMMA),
        volume_flux,
        residual,
        reference,
        method=method,
    )(reference)

    def jacobian(v):
        return jax.jvp(residual, (reference,), (v,))[1]

    rng = np.random.default_rng(seed)
    factors = []
    for _ in range(3):
        v = jnp.asarray(rng.standard_normal(mesh.n_cells))
        factors.append(float(jnp.linalg.norm(jacobian(m(v)) - v) / jnp.linalg.norm(v)))
    return float(np.mean(factors))


def test_preconditioner_strongly_contracts_the_convection_diffusion_error() -> None:
    """A single V-cycle brings ``J M v`` close to ``v`` -- i.e. ``M`` approximates ``J^{-1}`` well --
    on the convection-dominated operator, for both the two-level and the reduction-based hierarchy."""
    assert _contraction(32, 16, "twolevel") < 0.5
    assert _contraction(32, 16, "air") < 0.2  # lAIR is near-exact per cycle


def test_preconditioner_contraction_is_mesh_independent() -> None:
    """The contraction stays bounded as the mesh refines -- the scalable-iteration property (an
    unpreconditioned convection solve would instead need O(N) iterations)."""
    coarse = _contraction(24, 12, "twolevel")
    fine = _contraction(48, 24, "twolevel")
    assert coarse < 0.5 and fine < 0.5
    assert fine < 1.6 * coarse  # bounded, not growing with size


def test_preconditioner_apply_is_linear() -> None:
    """The frozen V-cycle is a constant linear operator in its argument (a valid left preconditioner
    for a plain-GMRES step)."""
    mesh, geometry, volume_flux, residual = _transport(16, 8)
    reference = jnp.ones(mesh.n_cells)
    m = scalar_transport_preconditioner(
        mesh, geometry, jnp.full(mesh.n_cells, GAMMA), volume_flux, residual, reference
    )(reference)
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.standard_normal(mesh.n_cells))
    b = jnp.asarray(rng.standard_normal(mesh.n_cells))
    assert jnp.allclose(m(2.0 * a - 3.0 * b), 2.0 * m(a) - 3.0 * m(b), atol=1e-9)
