"""Integration: the monolithic coupled RANS solve with a mass-flow (bulk-velocity) constraint.

A streamwise-periodic channel driven to a **target bulk velocity** by a body force that is itself a
solve unknown -- a scalar Lagrange multiplier appended to the coupled state, with the coupled residual
bordered by the constraint ``<U_dir> - U_bar = 0``. This is the companion of
:mod:`test_coupled_periodic_channel` (fixed body force, floating bulk velocity): here the force floats
and the bulk velocity is held fixed.

It checks the three properties that make the constrained coupled solve the design-note target engine:

* it holds the bulk velocity **by construction** while converging the unfrozen coupled residual to
  machine precision, from the exactly-uniform hybrid initial condition -- with a genuinely turbulent,
  floor-free field (the realizability floors are strictly inactive at the root, so the adjoint is
  honest);
* the fixed point is **AMG-method-independent** -- the ``air`` and ``twolevel`` coarsenings land on the
  same root (and the same recovered force). The segregated Picard loop does not converge on this
  body-force channel (the failure the monolithic engine exists to overcome), so this independent-solver
  cross-check stands in for the inlet case's coupled-vs-segregated agreement;
* the coupled implicit-function-theorem adjoint **carries the constraint** -- the point of putting the
  bulk-velocity constraint *inside* the coupled residual: ``jax.grad`` through the converged constrained
  solve is a single transpose solve on the bordered residual at the fixed point, matching finite
  differences. This is the sensitivity of the turbulent flow *at fixed bulk velocity*.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall
from aquaflux.flow.mean_velocity import _with_body_force
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import SSTModel, SSTTurbulence
from aquaflux.turbulence.coupled import (
    CoupledRANS,
    mass_flow_coupled_continuation,
    solve_coupled_mass_flow,
)

RHO, U_B, H = 1.0, 1.0, 2.0  # target bulk velocity U_B along x
RE_B, NY, GROWTH, BETA0 = 20000, 48, 1.13, 0.004
NU = U_B * H / RE_B  # 1e-4
K_FLOOR = 1e-8  # the hybrid IC's floor; asserted strictly inactive at the converged state
PRECONDITIONER = {"schur_scaling": "msimpler", "velocity": "convection"}
MAX_STEPS = 300


def _periodic_channel():
    """Build the periodic body-force turbulent channel (mesh, momentum, turbulence)."""
    y_nodes = graded_nodes(NY, H, GROWTH)
    mesh = structured_grid_2d(
        4, NY, lx=1.0, ly=H, periodic=("x",), named_boundaries=True, y_nodes=y_nodes
    )
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
        body_force=(
            BETA0,
            0.0,
        ),  # only the initial guess for the multiplier; the constraint sets it
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


def _bulk_velocity(momentum: MomentumContinuity, flow: jnp.ndarray) -> float:
    velocity, _ = momentum.unpack(flow)
    volume = momentum.geometry.cell.volume
    return float(jnp.sum(velocity[:, 0] * volume) / jnp.sum(volume))


@pytest.fixture(scope="module")
def case():
    """The channel and the ``air``-preconditioned constrained solution, self-started from the hybrid IC."""
    mesh, momentum, turbulence = _periodic_channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    # No initial state: the constrained solve self-starts from the exactly-symmetric uniform plug
    # (u_y == 0), so this exercises the guarded-sqrt strain fix directly.
    flow, k, omega, beta = solve_coupled_mass_flow(
        coupled, target=U_B, flow_direction=0, method="air", max_steps=MAX_STEPS, **PRECONDITIONER
    )
    return {
        "mesh": mesh,
        "momentum": momentum,
        "turbulence": turbulence,
        "coupled": coupled,
        "solution": (flow, k, omega, beta),
    }


@pytest.mark.slow
def test_constrained_solve_holds_bulk_velocity_and_converges(case) -> None:
    """The bulk velocity is held at the target while the bordered coupled residual reaches machine zero."""
    coupled, momentum, turbulence = case["coupled"], case["momentum"], case["turbulence"]
    flow, k, omega, beta = case["solution"]

    # The constraint holds by construction: the volume-averaged streamwise velocity equals the target.
    assert _bulk_velocity(momentum, flow) == pytest.approx(U_B, abs=1e-8)

    # The bordered residual [R_coupled(state; beta); <U> - U_b] is a genuine root to machine precision.
    forced = eqx.tree_at(lambda c: c.momentum, coupled, _with_body_force(coupled.momentum, 0, beta))
    r_coupled = forced.residual(coupled.pack_state(flow, k, omega))
    constraint = _bulk_velocity(momentum, flow) - U_B
    assert float(jnp.linalg.norm(jnp.append(r_coupled, constraint))) < 1e-8

    # The recovered multiplier is the mean pressure gradient the force balance needs: strictly positive
    # (drives the flow) and finite.
    assert float(beta) > 0.0
    assert jnp.isfinite(beta)

    # Genuinely turbulent, not the laminar (nu_t = 0) branch the degenerate IC would have started.
    nu_t = turbulence.eddy_viscosity(momentum.velocity_gradient(flow), k, omega)
    assert float(jnp.max(nu_t) / NU) > 10.0

    # The realizability floors are strictly inactive at the converged state (adjoint honesty): the
    # coupled residual carries no in-residual floor, so these are not tautological.
    assert float(jnp.min(k)) > 100.0 * K_FLOOR
    assert float(jnp.min(omega)) > 0.0


@pytest.mark.slow
def test_constrained_fixed_point_is_amg_method_independent(case) -> None:
    """Two independent AMG coarsenings reach the same constrained root -- and the same recovered force.

    The segregated loop does not converge on this body-force channel, so the constrained fixed point is
    cross-validated the way the periodic fixed-force case is: an independent solve (``twolevel`` in place
    of ``air``) must land on the same state, and the floating multiplier must agree, to solver tolerance.
    """
    coupled = case["coupled"]
    flow_air, k_air, omega_air, beta_air = case["solution"]

    flow_tl, k_tl, omega_tl, beta_tl = solve_coupled_mass_flow(
        coupled, target=U_B, flow_direction=0, method="twolevel", max_steps=400, **PRECONDITIONER
    )

    assert float(jnp.linalg.norm(flow_tl - flow_air) / jnp.linalg.norm(flow_air)) < 1e-6
    assert float(jnp.linalg.norm(k_tl - k_air) / jnp.linalg.norm(k_air)) < 1e-6
    assert float(jnp.linalg.norm(omega_tl - omega_air) / jnp.linalg.norm(omega_air)) < 1e-6
    assert float(beta_tl) == pytest.approx(float(beta_air), rel=1e-6)


@pytest.mark.slow
def test_constrained_coupled_adjoint_matches_finite_difference(case) -> None:
    """``jax.grad`` through the converged constrained solve is the exact coupled adjoint carrying the constraint.

    The constraint lives *inside* the coupled residual, so the implicit-function-theorem adjoint at the
    fixed point transposes the bordered Jacobian -- the sensitivity of the turbulent field at fixed bulk
    velocity. Differentiating a scalar objective through the warm-started solve must match a central
    finite difference. Iteration-count independence is inherent: the adjoint is a single transpose solve
    on the converged residual, not the forward march unrolled.
    """
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws, _ = case["solution"]

    # Build the continuation once, outside jax.grad, on concrete parameters (the block preconditioner
    # and the constraint border must not be traced); differentiate only the converged solve.
    reference_state = coupled.pack_state(flow_ws, k_ws, omega_ws)
    continuation = mass_flow_coupled_continuation(
        coupled, reference_state, flow_direction=0, method="twolevel", **PRECONDITIONER
    )

    def objective(nu_scale):
        scaled = eqx.tree_at(
            lambda c: c.turbulence.molecular_viscosity,
            coupled,
            coupled.turbulence.molecular_viscosity * nu_scale,
        )
        _, k, _, _ = solve_coupled_mass_flow(
            scaled,
            target=U_B,
            flow_direction=0,
            flow=flow_ws,
            k=k_ws,
            omega=omega_ws,
            continuation=continuation,
            max_steps=40,
        )
        return jnp.sum(k**2)

    analytic = float(jax.grad(objective)(1.0))
    eps = 1e-4
    finite_difference = float((objective(1.0 + eps) - objective(1.0 - eps)) / (2 * eps))
    assert abs(analytic - finite_difference) / abs(finite_difference) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
