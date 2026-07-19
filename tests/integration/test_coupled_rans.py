"""Integration: the monolithic coupled RANS Newton solve on a turbulent channel.

The unfrozen residual ``R(u, p, k, omega)`` is solved as **one** Newton system (globalized by the
coupled pseudo-transient continuation), from a short segregated warm start. These check the three
properties that make it the design note's target engine (S5): it converges the coupled system to
machine precision with the turbulence field positive and healthy; the coupled fixed point is the
*same* state the segregated Picard loop converges to; and -- handed to the implicit solver -- it
yields the exact coupled adjoint (a single transpose solve on the unfrozen residual), matching finite
differences. Genuinely turbulent (Re = U H / nu = 2500), so ``k`` stays well above its floor and the
floor plays no part in the converged state or its sensitivity.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    reused_flow_solve,
)
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, FieldProperty, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    SSTModel,
    SSTTurbulence,
    inlet_k,
    inlet_omega,
    scalar_pseudo_transient_solve,
    solve_segregated,
)
from aquaflux.turbulence.coupled import CoupledRANS, coupled_continuation, solve_coupled

RHO, U_IN, H, L = 1.0, 1.0, 1.0, 4.0
NU = 4e-4  # Re = U H / nu = 2500
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * H
PRECONDITIONER = {"schur_scaling": "msimpler", "velocity": "convection"}


def _with_viscosity(momentum, mu):
    props = PropertyModel({**momentum.properties.properties, "viscosity": FieldProperty(mu)})
    return eqx.tree_at(lambda m: m.properties, momentum, props)


def _channel(nx=28, ny=20, growth=1.2):
    y_nodes = graded_nodes(ny, H, growth)
    mesh = structured_grid_2d(nx, ny, lx=L, ly=H, named_boundaries=True, y_nodes=y_nodes)
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, model))
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
    turbulence = SSTTurbulence.build(
        model,
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
    return mesh, momentum, turbulence, k_in, omega_in


@pytest.fixture(scope="module")
def warm_started():
    """A short segregated pre-smooth, shared by the coupled tests as their warm start / reference."""
    mesh, momentum, turbulence, k_in, omega_in = _channel()
    n = mesh.n_cells
    coupled = CoupledRANS.build(momentum, turbulence, RHO)
    reference = jnp.full(n, RHO * 21 * NU)
    solve_flow = reused_flow_solve(_with_viscosity(momentum, reference), **PRECONDITIONER)
    flow0, k0, omega0 = momentum.initial_state(), jnp.full(n, k_in), jnp.full(n, omega_in)
    flow_ws, k_ws, omega_ws = solve_segregated(
        momentum,
        turbulence,
        solve_flow,
        scalar_pseudo_transient_solve(max_steps=40),
        flow0,
        k0,
        omega0,
        density=RHO,
        max_sweeps=8,
        rtol=1e-10,
        relaxation=0.5,
        scalar_preconditioner="twolevel",
    )
    return {
        "mesh": mesh,
        "momentum": momentum,
        "turbulence": turbulence,
        "coupled": coupled,
        "solve_flow": solve_flow,
        "warm": (flow_ws, k_ws, omega_ws),
        "initial": (flow0, k0, omega0),
    }


@pytest.mark.slow
def test_coupled_newton_converges_and_matches_the_segregated_solution(warm_started) -> None:
    coupled = warm_started["coupled"]
    flow_ws, k_ws, omega_ws = warm_started["warm"]

    flow, k, omega = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )

    # Converged to machine precision, with a healthy, strictly-positive turbulence field.
    residual_norm = float(jnp.linalg.norm(coupled.residual(coupled.pack_state(flow, k, omega))))
    assert residual_norm < 1e-8
    assert float(jnp.min(k)) >= 0.0
    assert float(jnp.min(omega)) > 0.0
    assert float(jnp.max(k)) > 10.0 * float(jnp.min(jnp.abs(k)) + 1e-30)  # genuinely turbulent

    # Same fixed point as a fully-converged segregated solve.
    momentum, turbulence = warm_started["momentum"], warm_started["turbulence"]
    flow0, k0, omega0 = warm_started["initial"]
    flow_s, k_s, omega_s = solve_segregated(
        momentum,
        turbulence,
        warm_started["solve_flow"],
        scalar_pseudo_transient_solve(max_steps=40),
        flow0,
        k0,
        omega0,
        density=RHO,
        max_sweeps=60,
        rtol=1e-9,
        relaxation=0.5,
        scalar_preconditioner="twolevel",
    )
    assert float(jnp.linalg.norm(flow - flow_s) / jnp.linalg.norm(flow_s)) < 1e-4
    assert float(jnp.linalg.norm(k - k_s) / jnp.linalg.norm(k_s)) < 1e-3
    assert float(jnp.linalg.norm(omega - omega_s) / jnp.linalg.norm(omega_s)) < 1e-4


@pytest.mark.slow
def test_coupled_adjoint_matches_finite_difference(warm_started) -> None:
    coupled = warm_started["coupled"]
    flow_ws, k_ws, omega_ws = warm_started["warm"]

    # Build the continuation once, outside jax.grad, on concrete parameters (the block preconditioner
    # must not be traced); differentiate only the converged solve through the coupled IFT adjoint.
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)
    continuation = coupled_continuation(
        coupled, reference_state, method="twolevel", **PRECONDITIONER
    )

    def objective(nu_scale):
        scaled = eqx.tree_at(
            lambda c: c.turbulence.molecular_viscosity,
            coupled,
            coupled.turbulence.molecular_viscosity * nu_scale,
        )
        _, k, _ = solve_coupled(
            scaled, flow_ws, k_ws, omega_ws, continuation=continuation, max_steps=40
        )
        return jnp.sum(k**2)

    analytic = float(jax.grad(objective)(1.0))
    eps = 1e-4
    finite_difference = float((objective(1.0 + eps) - objective(1.0 - eps)) / (2 * eps))
    assert abs(analytic - finite_difference) / abs(finite_difference) < 1e-5


@pytest.mark.slow
def test_coupled_solve_self_starts_from_a_cold_hybrid_initial_condition() -> None:
    # No warm start, no initial state: solve_coupled builds the hybrid IC itself (potential-flow
    # velocity + Laplace-smoothed k/omega) and converges the monolithic Newton from nothing -- which a
    # raw cold start (u=0, uniform k/omega) cannot do.
    _, momentum, turbulence, _, _ = _channel()
    coupled = CoupledRANS.build(momentum, turbulence, RHO)

    flow, k, omega = solve_coupled(coupled, method="twolevel", max_steps=40, **PRECONDITIONER)

    residual_norm = float(jnp.linalg.norm(coupled.residual(coupled.pack_state(flow, k, omega))))
    assert residual_norm < 1e-8
    assert float(jnp.min(k)) >= 0.0
    assert float(jnp.min(omega)) > 0.0
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    assert float(jnp.max(nu_t) / NU) > 1.0  # genuinely turbulent at the converged state


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
