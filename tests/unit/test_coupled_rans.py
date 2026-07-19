"""Unit: the monolithic coupled RANS residual -- layout, jit-safety, and Jacobian correctness.

Fast checks that do not run a full coupled solve: the state layout in isolation, that the residual
assembles under jit (the regression guard for the boundary-resolve fix), and that its automatic
Jacobian matches finite differences on a healthy (well-positive) state. The full coupled Newton
convergence, its agreement with the segregated loop, and the coupled adjoint are the slow integration
tests (:mod:`tests.integration.test_coupled_rans`).
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import SSTModel, SSTTurbulence
from aquaflux.turbulence.coupled import CoupledRANS, CoupledRANSLayout

RHO, NU, U_LID = 1.0, 1e-2, 1.0
WALLS = ("top", "bottom", "left", "right")


def test_layout_round_trips_and_sizes() -> None:
    layout = CoupledRANSLayout(dim=2, n_cells=5)
    assert layout.flow_size == (2 + 1) * 5
    assert layout.size == (2 + 3) * 5
    flow = jnp.arange(layout.flow_size, dtype=float)
    k = 10.0 + jnp.arange(5, dtype=float)
    omega = 100.0 + jnp.arange(5, dtype=float)
    state = layout.pack(flow, k, omega)
    assert state.shape == (layout.size,)
    f, kk, oo = layout.unpack(state)
    assert jnp.array_equal(f, flow)
    assert jnp.array_equal(kk, k)
    assert jnp.array_equal(oo, omega)


def _cavity(n=6):
    mesh = structured_grid_2d(n, n, lx=1.0, ly=1.0, named_boundaries=True)
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
    return mesh, CoupledRANS.build(momentum, turbulence, RHO)


def _healthy_state(mesh, coupled, seed=0):
    """A well-positive coupled state: modest random flow, k ~ 0.05, omega ~ 10 (floor inactive)."""
    n = mesh.n_cells
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    velocity = 0.1 * jax.random.normal(keys[0], (n, mesh.dim))
    pressure = 0.1 * jax.random.normal(keys[1], (n,))
    flow = coupled.momentum.pack(velocity, pressure)
    k = 0.05 + 0.01 * jax.random.uniform(keys[2], (n,))
    omega = 10.0 + jax.random.uniform(keys[3], (n,))
    return coupled.pack_state(flow, k, omega)


def test_coupled_build_resolves_boundaries_so_the_residual_jits() -> None:
    # Regression: the turbulence residual rebuilds its assembler each call; without the pre-resolved
    # boundaries (CoupledRANS.build) that rebuild re-runs a dynamic-shape nonzero on the mesh labels
    # and a jitted residual raises ConcretizationTypeError. jit + eval must succeed and stay finite.
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    residual = eqx.filter_jit(coupled.residual)(state)
    assert residual.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(residual)))


def test_residual_jacobian_matches_finite_difference() -> None:
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    direction = jax.random.normal(jax.random.PRNGKey(3), (state.shape[0],))
    direction = direction / jnp.linalg.norm(direction)
    jvp = jax.jvp(coupled.residual, (state,), (direction,))[1]
    assert bool(jnp.all(jnp.isfinite(jvp)))
    eps = 1e-5
    fd = (coupled.residual(state + eps * direction) - coupled.residual(state - eps * direction)) / (
        2 * eps
    )
    rel = float(jnp.linalg.norm(fd - jvp) / jnp.linalg.norm(jvp))
    assert rel < 1e-6


def test_layout_matches_the_assembler_dimensions() -> None:
    mesh, coupled = _cavity()
    assert coupled.layout.dim == mesh.dim
    assert coupled.layout.n_cells == mesh.n_cells
    assert coupled.pack_state(
        coupled.momentum.initial_state(),
        jnp.ones(mesh.n_cells),
        jnp.ones(mesh.n_cells),
    ).shape == (coupled.layout.size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
