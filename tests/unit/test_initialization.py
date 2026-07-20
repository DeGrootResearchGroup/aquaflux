"""Unit: the cheap field initializers -- scalar Laplace, potential flow, and the hybrid RANS IC.

Fast checks (linear solves on small meshes, no nonlinear/coupled solve): the Laplace solve reproduces
an analytic harmonic field; the potential-flow velocity matches the through-flow with no wall
penetration and works as a standalone flow-only initializer (and degrades to zero on a closed domain);
and the hybrid IC yields positive fields with the analytical near-wall omega and a small momentum-block
residual. The coupled solve self-starting from this IC is the slow integration test.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    laplace_field,
    potential_flow,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    SSTModel,
    SSTTurbulence,
    hybrid_initialize,
    inlet_k,
    inlet_omega,
    omega_wall_value,
)

RHO, U_IN, NU = 1.0, 1.0, 1e-2


def _channel(nx=16, ny=12, lx=3.0, ly=1.0):
    mesh = structured_grid_2d(nx, ny, lx=lx, ly=ly, named_boundaries=True)
    geometry = mesh.geometry()
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
    return mesh, geometry, momentum


def test_laplace_field_reproduces_a_linear_harmonic() -> None:
    # phi linear from 0 (left) to 1 (right), zero-gradient top/bottom -> phi = x / lx exactly.
    lx = 3.0
    mesh = structured_grid_2d(12, 8, lx=lx, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    boundary = BoundaryConditions(
        {
            "left": Dirichlet(0.0),
            "right": Dirichlet(1.0),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    phi, _ = laplace_field(mesh, geometry, boundary, gradient_scheme=CompactGreenGauss())
    expected = geometry.cell.centroid[:, 0] / lx
    assert float(jnp.max(jnp.abs(phi - expected))) < 1e-8


def test_potential_flow_matches_the_through_flow_without_wall_penetration() -> None:
    _, _, momentum = _channel()
    flow = potential_flow(momentum)  # flow-only: no turbulence involved
    velocity, _ = momentum.unpack(flow)
    assert bool(jnp.all(jnp.isfinite(flow)))
    # straight channel: streamwise velocity == inlet, no transverse penetration
    assert float(jnp.max(jnp.abs(velocity[:, 0] - U_IN))) < 1e-6
    assert float(jnp.max(jnp.abs(velocity[:, 1]))) < 1e-6


def test_potential_flow_is_zero_on_a_closed_domain() -> None:
    # Lid-driven cavity: all walls, no outlet -> no potential through-flow -> zero velocity.
    mesh = structured_grid_2d(8, 8, lx=1.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(U_IN, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )
    velocity, _ = momentum.unpack(potential_flow(momentum))
    assert float(jnp.max(jnp.abs(velocity))) < 1e-8


def _turbulence(mesh, geometry, k_in, omega_in):
    return SSTTurbulence.build(
        SSTModel(),
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


def test_hybrid_initialize_is_positive_with_analytical_wall_omega() -> None:
    mesh, geometry, momentum = _channel()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), 0.05))
    omega_in = float(inlet_omega(jnp.array(k_in), 0.07, model))
    turbulence = _turbulence(mesh, geometry, k_in, omega_in)

    flow, k, omega = hybrid_initialize(momentum, turbulence)

    assert flow.shape == (momentum.mesh.n_cells * (mesh.dim + 1),)
    assert bool(jnp.all(jnp.isfinite(flow)))
    assert float(jnp.min(k)) > 0.0
    assert float(jnp.min(omega)) > 0.0
    assert float(jnp.max(k)) <= k_in + 1e-12  # harmonic interpolant is bounded by its boundary data

    # the near-wall cells carry the analytical omega_wall = 60 nu / (beta_1 d^2)
    expected_wall = omega_wall_value(
        turbulence.molecular_viscosity[turbulence.wall_cells],
        turbulence.wall_distance[turbulence.wall_cells],
        model,
    )
    assert jnp.allclose(omega[turbulence.wall_cells], expected_wall)
    # the interior (non-wall) omega is the boundary-propagated inlet value
    assert float(jnp.min(omega)) == pytest.approx(omega_in, rel=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _periodic_channel(beta=0.0035, mu_factor=1.0):
    """A streamwise-periodic channel driven by a body force: no inlet, every patch a wall."""
    mesh = structured_grid_2d(4, 24, lx=1.0, ly=2.0, periodic=("x",), named_boundaries=True)
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU * mu_factor), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
        body_force=(beta, 0.0),
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions({"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)}),
        omega_boundary=BoundaryConditions({"bottom": ZeroGradient(), "top": ZeroGradient()}),
    )
    return mesh, momentum, turbulence


def test_hybrid_initialize_starts_a_body_force_channel_in_the_turbulent_regime() -> None:
    """With no inlet, the k interpolant is harmonic between zero wall values -- identically zero.

    Started there the closure has no turbulence at all (``nu_t = 0``), which is not merely a poor
    guess but the *laminar* problem. The equilibrium level from the friction velocity replaces it.
    """
    _, momentum, turbulence = _periodic_channel(beta=0.0035)
    _, k, omega = hybrid_initialize(momentum, turbulence)

    u_tau = (0.0035 * 1.0 / RHO) ** 0.5  # h = V/A_wall = 1 for this ly = 2 channel
    assert float(jnp.min(k)) == pytest.approx(u_tau**2 / SSTModel().beta_star ** 0.5)
    assert float(jnp.min(omega)) > 0.0  # a pure-Neumann omega solve would leave the interior empty


def test_hybrid_initialize_gives_a_developed_channel_eddy_viscosity() -> None:
    """The equilibrium levels must land at the ``~0.09 u_tau h`` eddy viscosity a channel carries.

    Getting k right but omega wrong would be worse than either: ``nu_t = k / omega`` with omega at
    its floor is enormous, so both come from the same friction velocity.
    """
    _, momentum, turbulence = _periodic_channel(beta=0.0035)
    flow, k, omega = hybrid_initialize(momentum, turbulence)
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)

    u_tau = (0.0035 * 1.0 / RHO) ** 0.5
    assert float(jnp.max(nu_t)) == pytest.approx(0.09 * u_tau * 1.0, rel=0.05)
    # The degenerate interpolant would leave k at its 1e-8 floor, i.e. nu_t ~ 1e-9 -- no
    # turbulence at all. Pin the gap so a regression to that start cannot pass.
    assert float(jnp.max(nu_t)) > 1e-4
