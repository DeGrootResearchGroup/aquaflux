"""Integration: the monolithic coupled RANS Newton solve on a turbulent channel.

The unfrozen residual ``R(u, p, k, omega)`` is solved as **one** Newton system (globalized by the
coupled pseudo-transient continuation), self-started from the hybrid initial condition -- no
segregated pre-smooth. These check the three properties that make it the design note's target
engine (S5): it converges the coupled system to
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
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    SSTModel,
    SSTTurbulence,
    hybrid_initialize,
    inlet_k,
    inlet_omega,
    scalar_pseudo_transient_solve,
    solve_segregated,
)
from aquaflux.turbulence.coupled import CoupledRANS, coupled_continuation, solve_coupled

# Step caps are backstops, not costs: the solvers' while_loop exits on tolerance, so a generous cap
# is free (measured identical wall time and physics at 200 vs 500). These are sized to clear the
# pseudo-transient march's slow phase rather than truncate it mid-descent, which the convergence
# guard rightly rejects.
SCALAR_MAX_STEPS = 200
FLOW_MAX_STEPS = 300

RHO, U_IN, H, L = 1.0, 1.0, 1.0, 4.0
NU = 4e-4  # Re = U H / nu = 2500
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * H
PRECONDITIONER = {"schur_scaling": "msimpler", "velocity": "convection"}


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
def case():
    """The channel and the segregated reference's starting state, shared by the coupled tests.

    The coupled solve needs **no** segregated pre-smooth: `solve_coupled` self-starts from its own
    hybrid initial condition and converges in a third the wall clock the pre-smoothed path took.

    The segregated reference keeps its flow at **rest** on purpose -- the two engines want opposite
    velocity starts. The Picard flow block converges in 20 steps from rest against 70 from the
    potential field (a smaller ||R0|| tightens the relative stopping target faster than it shortens
    the march), while the coupled Newton wants the developed potential field: a state at rest is far
    from the answer, and its scalars still come from the hybrid IC, without which a uniform k leaves
    the first sweep's residual unchanged for ~30 pseudo-transient steps -- it briefly *rises* -- so
    the SER schedule's beta never relaxes and the march burns its budget. (An exactly-uniform start
    is itself fine for the coupled solve now that the strain magnitude's sqrt is guarded at S = 0;
    see `test_coupled_periodic_channel`, which self-starts from the symmetric plug.)
    """
    mesh, momentum, turbulence, _, _ = _channel()
    n = mesh.n_cells
    coupled = CoupledRANS.build(momentum, turbulence)
    # Freeze the preconditioner at a representative turbulent viscosity: nu_t = 20 nu, so the
    # effective mu the frozen a_P sees is 21x molecular.
    reference_nu_t = jnp.full(n, 20.0 * NU)
    solve_flow = reused_flow_solve(
        momentum.with_eddy_viscosity(reference_nu_t), max_steps=FLOW_MAX_STEPS, **PRECONDITIONER
    )
    hybrid = hybrid_initialize(momentum, turbulence)
    _, k0, omega0 = hybrid
    return {
        "mesh": mesh,
        "momentum": momentum,
        "turbulence": turbulence,
        "coupled": coupled,
        # The driver's flow seam returns (assembler, state); an unconstrained solve leaves the
        # assembler unchanged, so pass it through.
        "solve_flow": lambda m, s: (m, solve_flow(m, s)),
        # The coupled Newton's start: the full hybrid IC, potential-flow velocity included. Built
        # once here so the adjoint test can hand the solve a *concrete* state -- an IC constructed
        # inside jax.grad would trace the preconditioner it seeds.
        "coupled_start": hybrid,
        # The segregated reference's start: the same scalars, but the flow at rest.
        "initial": (momentum.initial_state(), k0, omega0),
    }


@pytest.mark.slow
def test_coupled_newton_converges_and_matches_the_segregated_solution(case) -> None:
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["coupled_start"]

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
    momentum, turbulence = case["momentum"], case["turbulence"]
    flow0, k0, omega0 = case["initial"]
    flow_s, k_s, omega_s = solve_segregated(
        momentum,
        turbulence,
        case["solve_flow"],
        scalar_pseudo_transient_solve(max_steps=SCALAR_MAX_STEPS),
        flow0,
        k0,
        omega0,
        max_sweeps=60,
        rtol=1e-9,
        relaxation=0.9,
        scalar_preconditioner="twolevel",
    )
    assert float(jnp.linalg.norm(flow - flow_s) / jnp.linalg.norm(flow_s)) < 1e-4
    assert float(jnp.linalg.norm(k - k_s) / jnp.linalg.norm(k_s)) < 1e-3
    assert float(jnp.linalg.norm(omega - omega_s) / jnp.linalg.norm(omega_s)) < 1e-4


@pytest.mark.slow
def test_coupled_adjoint_matches_finite_difference(case) -> None:
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["coupled_start"]

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
    coupled = CoupledRANS.build(momentum, turbulence)

    flow, k, omega = solve_coupled(coupled, method="twolevel", max_steps=40, **PRECONDITIONER)

    residual_norm = float(jnp.linalg.norm(coupled.residual(coupled.pack_state(flow, k, omega))))
    assert residual_norm < 1e-8
    assert float(jnp.min(k)) >= 0.0
    assert float(jnp.min(omega)) > 0.0
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    assert float(jnp.max(nu_t) / NU) > 1.0  # genuinely turbulent at the converged state


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
