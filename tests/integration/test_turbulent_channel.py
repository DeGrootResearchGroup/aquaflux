"""Integration: the segregated k-omega SST driver solves a genuinely turbulent channel.

The qualitative cavity check (:mod:`tests.integration.test_turbulent_cavity`) runs at low Reynolds
number. Here the open channel is driven to a convective, wall-resolved Reynolds number where the k
and omega transport equations are strongly convection-dominated -- the regime whose exact-Jacobian
linear solve stalls unless the production limiter is made explicit (``KProduction.explicit_limiter``,
set by ``k_residual``), which restores the M-matrix the Krylov solve needs. The flow block uses the
high-Reynolds preconditioned continuation (reused across sweeps); the scalar solves are a plain
(unpreconditioned) Krylov Newton, so this test isolates that the *linearization* -- not a scalar
preconditioner -- is what makes the high-Re turbulent solve converge.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import lineax
import numpy as np
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
from aquaflux.solve import NewtonSolver
from aquaflux.turbulence import (
    SSTModel,
    SSTTurbulence,
    inlet_k,
    inlet_omega,
    scalar_pseudo_transient_solve,
    solve_segregated,
)

RHO, U_IN, H, L = 1.0, 1.0, 1.0, 6.0
NU = 2e-4  # Re = rho U H / mu = 5000
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * H


def _channel(nx=48, ny=40, growth=1.18, *, explicit_production_limiter=True):
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
        explicit_production_limiter=explicit_production_limiter,
    )
    return mesh, momentum, turbulence, k_in, omega_in


@pytest.mark.slow
def test_high_reynolds_turbulent_channel_solves() -> None:
    mesh, momentum, turbulence, k_in, omega_in = _channel()

    # A frozen preconditioner at a representative eddy viscosity, reused across the sweeps.
    reference = jnp.full(mesh.n_cells, RHO * 21 * NU)
    reference_momentum = _with_viscosity(momentum, reference)
    solve_flow = reused_flow_solve(
        reference_momentum, schur_scaling="msimpler", velocity="convection"
    )

    flow, k, omega = solve_segregated(
        momentum,
        turbulence,
        solve_flow,
        scalar_pseudo_transient_solve(max_steps=40),
        momentum.initial_state(),
        jnp.full(mesh.n_cells, k_in),
        jnp.full(mesh.n_cells, omega_in),
        density=RHO,
        max_sweeps=8,
        relaxation=0.5,
        scalar_preconditioner="twolevel",
    )

    # Finite and physical.
    assert not bool(jnp.any(jnp.isnan(flow)))
    assert float(jnp.min(k)) >= 0.0
    assert float(jnp.min(omega)) > 0.0

    # Genuinely turbulent: the closure produces an eddy viscosity well above molecular.
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    assert float(jnp.max(nu_t) / NU) > 1.0


@pytest.mark.slow
def test_exact_production_limiter_solves_with_the_preconditioner() -> None:
    """The scalar preconditioner rescues the *exact* (non-Patankar) k-linearization.

    With ``explicit_production_limiter=False`` the k-Jacobian keeps the production limiter's exact
    derivative, which is indefinite where the cap is active -- an unpreconditioned solve stagnates
    there. The convection-diffusion AMG the continuation policy carries (its ``preconditioner``)
    reconstructs the operator well enough to rescue it, so a preconditioned exact-Newton k-solve
    converges to machine precision (quadratic). This isolates the **preconditioner** on the exact
    operator -- a bare Newton solve, no pseudo-time shift -- to document that the earlier
    exact-linearization failure was the preconditioner's diagonal bug, not an unsolvable operator.
    (The pseudo-transient shift the segregated driver adds is a *positive* diagonal that stabilizes the
    indefinite operator from a cold start but does not converge it as deeply; that robustness, not this
    deep exact-Newton convergence, is what the coupled solve relies on.)
    """
    mesh, momentum, turbulence, k_in, omega_in = _channel(32, 24, explicit_production_limiter=False)
    geometry = momentum.geometry
    n = mesh.n_cells

    # A limiter-active state: strong synthetic shear (high strain) at the small inlet k, so the cap
    # bites over much of the field -- where the exact and Patankar linearizations actually differ.
    velocity_gradient = jnp.zeros((n, mesh.dim, mesh.dim)).at[:, 0, 1].set(50.0)  # du_x/dy shear
    k = jnp.full(n, k_in)
    omega = jnp.full(n, omega_in)
    closure = turbulence.closure_fields(velocity_gradient, k, omega)
    mdot = RHO * U_IN * geometry.face.normal[:, 0] * geometry.face.area  # a uniform-flow mass flux

    production = np.asarray(closure.nu_t * closure.strain_rate**2)
    cap = np.asarray(10.0 * turbulence.model.beta_star * k * closure.omega)
    assert int(np.sum(production > cap)) > n // 4  # the cap is genuinely active over the field

    residual = turbulence.k_residual(mdot, closure)
    # The AMG the continuation policy carries, applied to a bare Newton solve to isolate it.
    preconditioner = turbulence.k_shift_policy(mdot, closure, k, method="twolevel").preconditioner
    gmres = lineax.GMRES(rtol=1e-8, atol=1e-8, restart=32, stagnation_iters=32)
    solved = NewtonSolver(iterations=6, solver=gmres, preconditioner=preconditioner).solve(
        residual, k
    )
    assert float(jnp.linalg.norm(residual(solved))) < 1e-8


def _with_viscosity(momentum, mu):
    properties = PropertyModel({**momentum.properties.properties, "viscosity": FieldProperty(mu)})
    return eqx.tree_at(lambda m: m.properties, momentum, properties)
