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
    LogScalars,
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
def test_coupled_log_omega_converges_to_the_same_positive_fixed_point(case) -> None:
    """The omega-log parametrization reaches the *same* coupled fixed point as the direct form, with
    ``omega`` strictly positive by construction.

    ``omega = e^w`` is a smooth bijection onto the positives, so the root of ``R(u, p, k, e^w) = 0`` is
    the direct root; only the Newton iterate space changes. This confirms the reparametrization is
    physics-preserving on a case both forms solve -- the payoff (positivity a full step cannot violate)
    is what lets it solve the stiff separating cases the direct form cannot.
    """
    momentum, turbulence, coupled = case["momentum"], case["turbulence"], case["coupled"]
    flow_ws, k_ws, omega_ws = case["coupled_start"]
    log_omega = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())

    flow_l, k_l, omega_l = solve_coupled(
        log_omega, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )
    residual_norm = float(
        jnp.linalg.norm(log_omega.residual(log_omega.state_from_physical(flow_l, k_l, omega_l)))
    )
    assert residual_norm < 1e-8
    assert float(jnp.min(omega_l)) > 0.0  # structural: e^w > 0 for every w

    flow_d, k_d, omega_d = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )
    assert float(jnp.linalg.norm(flow_l - flow_d) / jnp.linalg.norm(flow_d)) < 1e-4
    assert float(jnp.linalg.norm(k_l - k_d) / jnp.linalg.norm(k_d)) < 1e-3
    assert float(jnp.linalg.norm(omega_l - omega_d) / jnp.linalg.norm(omega_d)) < 1e-4


@pytest.mark.slow
def test_coupled_log_omega_adjoint_matches_finite_difference(case) -> None:
    """The coupled implicit-function-theorem adjoint is exact through the omega-log reparametrization.

    At the converged state the realizability floor is inactive and ``e^w`` is smooth, so the adjoint is
    the same single transpose solve on the unfrozen residual -- ``jax.grad`` through the omega-log solve
    matches finite differences, exactly as for the direct form.
    """
    momentum, turbulence = case["momentum"], case["turbulence"]
    flow_ws, k_ws, omega_ws = case["coupled_start"]
    log_omega = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    continuation = coupled_continuation(
        log_omega,
        log_omega.state_from_physical(flow_ws, k_ws, omega_ws),
        method="twolevel",
        **PRECONDITIONER,
    )

    def objective(nu_scale):
        scaled = eqx.tree_at(
            lambda c: c.turbulence.molecular_viscosity,
            log_omega,
            log_omega.turbulence.molecular_viscosity * nu_scale,
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


@pytest.mark.slow
def test_staged_preconditioner_refresh_reaches_the_same_fixed_point(case) -> None:
    """``refresh_rtol`` re-freezes the preconditioner mid-march without moving the converged state.

    The refresh is a forward-path device: both stages drive the *same* residual, and the preconditioner
    is ``stop_gradient``-ed whichever state it is frozen at. So a staged march must land on exactly the
    single-stage fixed point -- if it does not, the refresh has leaked into the physics.
    """
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["coupled_start"]

    single = solve_coupled(
        coupled, flow_ws, k_ws, omega_ws, method="twolevel", max_steps=40, **PRECONDITIONER
    )
    staged = solve_coupled(
        coupled,
        flow_ws,
        k_ws,
        omega_ws,
        method="twolevel",
        max_steps=40,
        refresh_rtol=1e-2,  # loose first stage, then re-freeze and finish
        **PRECONDITIONER,
    )

    for name, one, two in zip(("flow", "k", "omega"), single, staged, strict=True):
        rel = float(jnp.linalg.norm(two - one) / jnp.linalg.norm(one))
        assert rel < 1e-6, f"{name} moved by {rel:.2e} under a staged refresh"


@pytest.mark.slow
def test_staged_refresh_stops_at_the_same_tolerance(case) -> None:
    """``rtol`` must mean the same thing with and without ``refresh_rtol``.

    Each solver measures its own reference residual from the state it is handed, so stage two -- which
    starts from stage one's result -- would stop at ``rtol * refresh_rtol * ||R0||`` if its relative
    tolerance were not compensated: a factor ``refresh_rtol`` tighter than the caller asked for, which
    silently turns a converging solve into far more work or a ``max_steps`` failure. Both paths are
    driven to a loose ``rtol`` here so each stops *on tolerance* rather than overshooting to machine
    zero, which is what makes the comparison able to detect the difference.
    """
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["coupled_start"]
    start = coupled.pack_state(flow_ws, k_ws, omega_ws)
    reference_norm = float(jnp.linalg.norm(coupled.residual(start)))
    rtol = 1e-3

    common = dict(method="twolevel", max_steps=40, rtol=rtol, **PRECONDITIONER)
    single = solve_coupled(coupled, flow_ws, k_ws, omega_ws, **common)
    staged = solve_coupled(coupled, flow_ws, k_ws, omega_ws, refresh_rtol=1e-1, **common)

    def terminal_residual(fields):
        return float(jnp.linalg.norm(coupled.residual(coupled.pack_state(*fields))))

    single_residual = terminal_residual(single)
    staged_residual = terminal_residual(staged)
    target = rtol * reference_norm

    # Both must satisfy the *requested* tolerance ...
    assert single_residual <= target
    assert staged_residual <= target
    # ... and the staged one must not be dramatically over-solved, which is what an uncompensated
    # relative tolerance would produce (it would chase `rtol * refresh_rtol * ||R0||`, 10x tighter).
    assert staged_residual > target * 1e-2, (
        f"staged solve stopped at {staged_residual:.3e} against a target of {target:.3e} -- far "
        "tighter than requested, so the second stage's tolerance is not being compensated"
    )


@pytest.mark.slow
def test_staged_refresh_keeps_the_coupled_adjoint_exact(case) -> None:
    """The coupled implicit-function-theorem adjoint is unchanged by a mid-march refresh.

    The final stage is an ordinary converged solve, and its adjoint returns a zero cotangent to its
    initial guess, so the earlier stage (and the refresh between them) cannot enter the gradient. This
    checks that against finite differences, the same gate the unstaged solve passes.
    """
    coupled = case["coupled"]
    flow_ws, k_ws, omega_ws = case["coupled_start"]

    def objective(nu_scale, **kwargs):
        scaled = eqx.tree_at(
            lambda c: c.turbulence.molecular_viscosity,
            coupled,
            coupled.turbulence.molecular_viscosity * nu_scale,
        )
        _, k, _ = solve_coupled(
            scaled,
            flow_ws,
            k_ws,
            omega_ws,
            method="twolevel",
            max_steps=40,
            **PRECONDITIONER,
            **kwargs,
        )
        return jnp.sum(k**2)

    analytic = float(jax.grad(objective)(1.0, refresh_rtol=1e-2))
    eps = 1e-4
    finite_difference = float(
        (objective(1.0 + eps, refresh_rtol=1e-2) - objective(1.0 - eps, refresh_rtol=1e-2))
        / (2 * eps)
    )
    assert abs(analytic - finite_difference) / abs(finite_difference) < 1e-5
