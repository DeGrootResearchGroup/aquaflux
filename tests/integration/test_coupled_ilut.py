"""Integration: the monolithic-ILUT-preconditioned coupled RANS Newton solve on a turbulent channel.

The coupled continuation's block-triangular SIMPLE preconditioner is replaced by a single monolithic
incomplete-LU factorization of the assembled coupled Jacobian (:func:`coupled_ilut_continuation`).
These check the two properties that make it a usable drop-in: handed to ``solve_coupled`` it converges
the monolithic Newton to the **same** fixed point the block preconditioner reaches, and -- built once
outside ``jax.grad`` on concrete parameters -- it yields the exact coupled adjoint (a single transpose
solve on the unfrozen residual, preconditioned by the factorization's transpose), matching finite
differences. Genuinely turbulent (Re = U H / nu = 2500), so ``k`` stays above its floor and the floor
plays no part in the converged state or its sensitivity.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    CoupledRANS,
    SSTModel,
    SSTTurbulence,
    coupled_ilut_continuation,
    hybrid_initialize,
    inlet_k,
    inlet_omega,
    solve_coupled,
)

RHO, U_IN, H, L = 1.0, 1.0, 1.0, 4.0
NU = 4e-4  # Re = U H / nu = 2500
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * H
PRECONDITIONER = {"schur_scaling": "msimpler", "velocity": "convection"}


def _channel(nx=20, ny=14, growth=1.2):
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
    return momentum, turbulence


@pytest.fixture(scope="module")
def case():
    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    start = hybrid_initialize(momentum, turbulence)
    return {"coupled": coupled, "start": start}


@pytest.mark.slow
def test_ilut_solve_converges_and_matches_the_block_preconditioned_solve(case) -> None:
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)

    # Monolithic ILUT continuation, built off the jit path at the cold reference state.
    ilut = coupled_ilut_continuation(coupled, reference_state)
    flow_i, k_i, omega_i = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, continuation=ilut, max_steps=40
    )

    residual_norm = float(
        jnp.linalg.norm(coupled.residual(coupled.pack_state(flow_i, k_i, omega_i)))
    )
    assert residual_norm < 1e-8
    assert float(jnp.min(k_i)) >= 0.0
    assert float(jnp.min(omega_i)) > 0.0
    assert float(jnp.max(k_i)) > 10.0 * float(jnp.min(jnp.abs(k_i)) + 1e-30)  # genuinely turbulent

    # Same fixed point as the block-triangular preconditioner reaches.
    flow_b, k_b, omega_b = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )
    assert float(jnp.linalg.norm(flow_i - flow_b) / jnp.linalg.norm(flow_b)) < 1e-4
    assert float(jnp.linalg.norm(k_i - k_b) / jnp.linalg.norm(k_b)) < 1e-3
    assert float(jnp.linalg.norm(omega_i - omega_b) / jnp.linalg.norm(omega_b)) < 1e-4


@pytest.mark.slow
def test_ilut_adjoint_matches_finite_difference(case) -> None:
    """The coupled implicit-function-theorem adjoint is exact through the ILUT-preconditioned solve.

    The preconditioner is ``stop_gradient``-ed (it only accelerates the Krylov iteration), so the
    gradient is the single transpose solve on the unfrozen coupled residual -- ``jax.grad`` through the
    ILUT solve matches finite differences, exactly as for the block preconditioner. The continuation is
    built once outside ``jax.grad`` on concrete parameters (the factorization must not be traced).
    """
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["start"]
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)
    continuation = coupled_ilut_continuation(coupled, reference_state)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
