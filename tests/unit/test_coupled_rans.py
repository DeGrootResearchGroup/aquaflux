"""Unit: the monolithic coupled RANS residual -- layout, jit-safety, and Jacobian correctness.

Fast checks that do not run a full coupled solve: the state layout in isolation, that the residual
assembles under jit (the regression guard for the boundary-resolve fix), and that its automatic
Jacobian matches finite differences on a healthy (well-positive) state. The full coupled Newton
convergence, its agreement with the segregated loop, and the coupled adjoint are the slow integration
tests (:mod:`tests.integration.test_coupled_rans`).
"""

from __future__ import annotations

import dataclasses
import inspect

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import DifferenceRow, FirstOrderUpwind, LogRatioRow
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss, CorrectedGreenGauss, SweptGradientSolve
from aquaflux.solve import (
    NO_REFRESH,
    CycleGrowthTrigger,
    PseudoTransientStep,
    RefreshPolicy,
    RowScaledNorm,
    ShiftTerm,
)
from aquaflux.turbulence import (
    DirectScalars,
    LogScalars,
    SSTModel,
    SSTTurbulence,
    coupled_equation_names,
    coupled_fields,
    coupled_residuals,
    eddy_viscosity_drift,
    hybrid_initialize,
    production_and_limit,
    production_cap_active,
)
from aquaflux.turbulence.coupled import (
    _BLOCK_FORWARD,
    _FACTORIZATION_FORWARD,
    CoupledJacobianProbe,
    CoupledRANS,
    CoupledRANSLayout,
    LiveViscosityVelocityParts,
    _row_jacobian_scale,
    coupled_amg_continuation,
    coupled_continuation,
    coupled_lu_continuation,
    coupled_scaled_norm,
    mass_flow_coupled_continuation,
    solve_coupled,
)

from tests.support.meshes import perturbed_grid_2d
from tests.unit.test_gradient import _cell_graph_distance

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


def _cavity(n=6, mesh=None, gradient=None):
    mesh = structured_grid_2d(n, n, lx=1.0, ly=1.0, named_boundaries=True) if mesh is None else mesh
    gradient = CompactGreenGauss() if gradient is None else gradient
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        gradient,
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
        gradient,
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=list(WALLS),
        k_boundary=BoundaryConditions({w: Dirichlet(0.0) for w in WALLS}),
        omega_boundary=BoundaryConditions({w: ZeroGradient() for w in WALLS}),
    )
    return mesh, CoupledRANS.build(momentum, turbulence)


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


def test_coupled_build_rejects_a_turbulence_density_that_disagrees_with_the_flow_assembler() -> (
    None
):
    """SSTTurbulence.density and the flow PropertyModel's density are two independent numbers.

    Nothing else checks that a caller supplied the same value to both -- if they disagree, the
    k/omega volume flux (mdot / density) is silently wrong by that ratio in every SST consumer
    while the flow block solves fine, since it never reads SSTTurbulence.density at all.
    """
    mesh = structured_grid_2d(4, 4, lx=1.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    gradient = CompactGreenGauss()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        gradient,
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
        gradient,
        FirstOrderUpwind(),
        density=998.0,  # deliberately different from the flow assembler's RHO = 1.0
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=list(WALLS),
        k_boundary=BoundaryConditions({w: Dirichlet(0.0) for w in WALLS}),
        omega_boundary=BoundaryConditions({w: ZeroGradient() for w in WALLS}),
    )
    with pytest.raises(ValueError, match="does not match the flow assembler's density"):
        CoupledRANS.build(momentum, turbulence)


def test_lu_and_block_continuations_use_oppositely_tuned_restart_sizes() -> None:
    """The complete-LU continuation defaults to a small-restart GMRES; the block one keeps the large one.

    A restarted GMRES tests its stop only at each restart boundary, so the restart size should match how
    many vectors the preconditioner actually needs. The monolithic complete LU is the operator's exact
    inverse, so the 1% stop is reached within a handful of vectors and it uses a small restart; the
    block-triangular preconditioner needs a large subspace per cycle. The two must not share a default.
    """
    assert _FACTORIZATION_FORWARD.restart == 10
    assert _BLOCK_FORWARD.restart == 120

    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    lu_step = coupled_lu_continuation(coupled, state, backend="scipy")
    block_step = coupled_continuation(coupled, state, method=None)
    # Each built step carries the solver it will run; the LU's is the small-restart one by default.
    assert lu_step.forward_solver.restart == 10
    assert block_step.forward_solver.restart == 120
    # An explicit forward_solver still overrides the LU default.
    # ...and the restart alone can be moved without also replacing the stopping measure.
    assert (
        coupled_lu_continuation(
            coupled, state, backend="scipy", forward_restart=120
        ).forward_solver.restart
        == 120
    )


def test_every_continuation_builder_installs_the_same_globalization() -> None:
    """The march's globalization must not depend on which preconditioner it was built around.

    The three builders differ in exactly one thing — the preconditioner they freeze — and they route
    through one shared step builder for everything else. This pins that, because the alternative is not
    hypothetical: the block-diagonal and monolithic builders each grew their own copy of the tail and the
    copies drifted, in both directions and in ways unrelated to preconditioning. The monolithic one
    gained the k-positivity step limit, the cycle budget and the inner refresh; the block-diagonal one
    gained the line search's growth and descent-backoff rungs; neither gained the other's.

    The k-positivity limit is the one that mattered. Without it a step that drives ``k`` through zero
    makes ``sqrt(k)`` — and so the eddy viscosity — non-finite, which is a failure that has actually
    stopped a march; it shipped on one path only, so any recorded comparison between the two was
    comparing a guarded march against an unguarded one.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    # Both branches: the single-step one needs the cap as much as the dual-time one, because its
    # escalation ladder cannot catch `k < 0` -- the divergence guard fires on a residual that is already
    # non-finite, by which point `sqrt(k)` has poisoned the closure.
    built = {
        "block": coupled_continuation(coupled, state, method=None),
        "lu": coupled_lu_continuation(coupled, state, backend="scipy"),
        "block dual-time": coupled_continuation(coupled, state, method=None, inner_steps=2),
    }
    for name, step in built.items():
        assert step.step_limit is not None, f"{name} has no k-positivity limit"

    # ...and the surfaces themselves agree, so a knob cannot reappear on one side. Everything here is a
    # property of the coupled march rather than of any preconditioner, which is the test: a parameter
    # belongs on all four or on none, and the ones that reached only the builder being worked on at the
    # time are exactly how this went wrong before -- twice with the tail already extracted, so nothing
    # in the bodies looked duplicated and no single commit looked wrong.
    shared = {
        # The pseudo-transient schedule, the divergence guard and the line search.
        "beta0",
        "exponent",
        "beta_floor",
        "max_escalations",
        "escalation_factor",
        "divergence_cap",
        "line_search",
        "inner_steps",
        "inner_tol",
        "grow",
        "descent_backoff",
        "descent_test",
        # The shifted forward solve. `forward_rtol` / `forward_restart` / `forward_max_restarts` sat on
        # the multigrid builder alone, although the argument for them is about the *coupled residual*
        # (~100% omega under a plain 2-norm, so the flow block goes unresolved) and not about multigrid.
        "forward_solver",
        "forward_rtol",
        "forward_restart",
        "forward_max_restarts",
        # The progress measure and the shift.
        "block_scaled_norm",
        "shift_basis",
        # ...and this one was on two of the four, absent from both monolithic builders although the
        # configuration it was built for -- a dual-time low-shift march whose shift must track the
        # developing eddy viscosity -- is a monolithic one.
        "velocity_shift_parts",
        # The per-step guards.
        "inner_observer",
        "cycle_budget",
        "refresh_on_cycles",
        "inner_refresh",
        "positivity_floor",
        "positivity_projection",
    }
    builders = (
        coupled_continuation,
        coupled_lu_continuation,
        coupled_amg_continuation,
        mass_flow_coupled_continuation,
    )
    for builder in builders:
        missing = shared - set(inspect.signature(builder).parameters)
        assert not missing, f"{builder.__name__} cannot be given {sorted(missing)}"

    # One deliberate carve-out, pinned so it stays deliberate: the mass-flow builder takes no explicit
    # `residual_norm`, because the constrained path has no staged-refresh driver to inject a frozen
    # measure, and it supplies its own constraint-aware one. Every other builder takes it.
    assert "residual_norm" not in inspect.signature(mass_flow_coupled_continuation).parameters
    for builder in builders[:-1]:
        assert "residual_norm" in inspect.signature(builder).parameters


def test_every_builder_stops_the_forward_solve_in_the_march_s_own_measure() -> None:
    """The forward solve's stopping measure is the march's progress measure, on every builder.

    Two separate rules meet here. The march must be steered by and judged by one definition, so the
    linear solve cannot converge in a quantity the march never reads. And the *reason* the coupled
    residual needs a row-scaled stop is about the residual, not about any preconditioner: a plain
    2-norm of it is ~100% ``omega``, whose residual sits orders above the flow's, so a solve stops once
    ``omega`` is resolved while the flow-dominated part of the Newton step is still coarse. That
    argument reached only the builder being worked on at the time, leaving the default path -- what
    ``solve_coupled`` builds when nothing is passed -- stopping on the norm its own docstring calls
    effectively blind.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    for name, step in {
        "block": coupled_continuation(coupled, state, method=None),
        "lu": coupled_lu_continuation(coupled, state, backend="scipy"),
    }.items():
        assert step.forward_solver.norm is step.residual_norm, (
            f"{name} steers on one measure and stops its linear solve on another"
        )
        # ...and that measure is the row-equilibrated one, not the Euclidean norm it used to be.
        assert isinstance(step.residual_norm, RowScaledNorm), name

    # An explicit measure is honoured all the way through, so the two cannot come apart there either --
    # which is what `solve_coupled` relies on when it re-injects the march's initial measure at every
    # refresh rather than letting a self-normalising one re-base at the developed state.
    base = coupled_continuation(coupled, state, method=None)
    explicit = coupled_continuation(coupled, state, method=None, residual_norm=base.residual_norm)
    assert explicit.forward_solver.norm is explicit.residual_norm is base.residual_norm


def test_the_constrained_builder_keeps_a_euclidean_stop_for_a_stated_reason() -> None:
    """The bordered mass-flow path is the one that genuinely differs, and it differs consistently.

    The row-equilibrated measure has no constraint-aware form: it would scale the border row by a
    diagonal the constraint does not have. So that march is judged in the Euclidean norm — and its
    forward solve therefore stops there too, at a Euclidean tolerance, because the tolerance and the
    norm it is measured in are one decision. This is a property of the path, not a surface that drifted.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    step = mass_flow_coupled_continuation(coupled, state, method=None)
    assert step.residual_norm is jnp.linalg.norm
    assert step.forward_solver.norm is step.residual_norm


def test_a_monolithic_builder_takes_the_injected_velocity_shift_source() -> None:
    """``velocity_shift_parts`` reaches the monolithic paths, which is where it was wanted.

    It says where the velocity shift's two diagonal buckets come from — a property of the *shift*, not
    of the preconditioner — and a live-viscosity source needs only momentum, the closure and the two
    variable transforms, so nothing about a monolithic build excludes it. It nonetheless existed on the
    two block builders only, and the configuration it was written for (a dual-time low-shift march whose
    shift must track the developing eddy viscosity) is a monolithic one.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    live = LiveViscosityVelocityParts(
        coupled.momentum, coupled.turbulence, coupled.k_transform, coupled.omega_transform
    )
    step = coupled_lu_continuation(coupled, state, backend="scipy", velocity_shift_parts=live)
    assert step.shift_policy.base.velocity_shift_parts is live
    # ...and it is genuinely live: away from the state the assembler was frozen at, the shift it
    # produces differs from the frozen one, which is the whole reason the source is injected. At the
    # freeze state the two coincide by construction, so a check there would pass on a dead wire.
    frozen = coupled_lu_continuation(coupled, state, backend="scipy")
    flow_p, k_p, omega_p = coupled.layout.unpack(state)
    developed = coupled.layout.pack(flow_p, k_p * 4.0, omega_p)
    assert not np.allclose(
        np.asarray(step.shift_policy.shift_term(developed).diagonal),
        np.asarray(frozen.shift_policy.shift_term(developed).diagonal),
    )


def test_the_live_shift_source_honours_its_protocol_arity() -> None:
    """It must be callable the way the protocol declares, and say so when it cannot do the job.

    The protocol gives the turbulence blocks defaults, because a frozen-viscosity source ignores them.
    This one declared them required, so it did not satisfy the arity its own protocol promises — latent
    until something handed it to a policy that calls ``parts(flow)``, and then a ``TypeError`` from deep
    inside a shift policy. It now accepts the call and refuses it in terms that name the alternative.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    flow, k, omega = coupled.layout.unpack(state)
    live = LiveViscosityVelocityParts(
        coupled.momentum, coupled.turbulence, coupled.k_transform, coupled.omega_transform
    )
    assert len(live.parts(flow, k, omega)) == 2
    with pytest.raises(TypeError, match="FrozenViscosityVelocityParts"):
        live.parts(flow)


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


def test_scalar_variable_transforms() -> None:
    """DirectScalars is the identity; LogScalars is ``e^w`` with derivative ``e^w`` (physics-free)."""
    w = jnp.array([-3.0, 0.0, 2.5])
    direct = DirectScalars()
    assert jnp.array_equal(direct.to_physical(w), w)
    assert jnp.array_equal(direct.to_solved(w), w)
    assert jnp.array_equal(direct.jacobian_scale(w), jnp.ones_like(w))

    log_scalars = LogScalars()
    phi = log_scalars.to_physical(w)
    assert jnp.allclose(phi, jnp.exp(w))
    assert bool(jnp.all(phi > 0.0))  # positive for any real w -- the structural guarantee
    assert jnp.allclose(log_scalars.to_solved(phi), w)  # round trip
    assert jnp.allclose(log_scalars.jacobian_scale(phi), phi)  # d(e^w)/dw = e^w = phi


def test_log_omega_reparametrization_preserves_the_transport_residual() -> None:
    """omega-log reparametrizes the Newton *unknown*, not the physics.

    Every **transport** row of the coupled residual at the log-mapped state equals the direct
    residual at the same physical fields, and it stays differentiable through the ``e^w`` map.

    The near-wall **fixation** rows are deliberately *not* identical: each transform writes the
    fixation in its own solved variable (``omega - omega_wall`` directly, ``log(omega/omega_wall)``
    under the log map), which is what keeps that row linear in the unknown actually being stepped.
    Both vanish on exactly the same set, so the two forms still share a root -- which the companion
    test pins.
    """
    mesh, direct = _cavity()
    log_omega = CoupledRANS.build(direct.momentum, direct.turbulence, omega_transform=LogScalars())
    physical = _healthy_state(mesh, direct)
    flow, k, omega = direct.layout.unpack(physical)
    solved = log_omega.state_from_physical(flow, k, omega)

    _, _, direct_omega = direct.layout.unpack(direct.residual(physical))
    _, _, log_omega_rows = log_omega.layout.unpack(log_omega.residual(solved))
    interior = jnp.setdiff1d(jnp.arange(direct_omega.shape[0]), direct.turbulence.wall_cells)
    assert jnp.allclose(direct_omega[interior], log_omega_rows[interior], atol=1e-10)

    direction = jax.random.normal(jax.random.PRNGKey(4), (solved.shape[0],))
    jvp = jax.jvp(log_omega.residual, (solved,), (direction,))[1]
    assert bool(jnp.all(jnp.isfinite(jvp)))


def test_the_two_fixation_row_forms_share_a_root_and_the_log_form_is_linear() -> None:
    """The transform-matched fixation rows vanish together, and the log form is linear in ``w``.

    Linearity is the point: the difference row's Newton correction in the log variable is
    ``target/phi - 1`` (the linearization of an exponential, which overshoots by ``e**(r-1)`` at a
    target ratio ``r``), while the log-ratio row's is ``log(target/phi)`` -- exact at any ratio, so a
    full step lands on the constraint however far off it starts.
    """
    phi = jnp.array([1.0, 5.0, 1.0e4])
    target = jnp.array([1.0, 5.0, 5.0])  # first two already satisfied, the third far off
    difference = DifferenceRow().row(phi, target)
    log_ratio = LogRatioRow().row(phi, target)
    # Same root: both vanish exactly where phi == target, and nowhere else.
    assert jnp.array_equal(difference == 0.0, log_ratio == 0.0)
    assert bool(jnp.all(difference[:2] == 0.0)) and bool(jnp.all(log_ratio[:2] == 0.0))

    # The log row is exactly linear in w = log(phi): its derivative is 1 regardless of the ratio.
    slope = jax.grad(lambda w: jnp.sum(LogRatioRow().row(jnp.exp(w), target)))(jnp.log(phi))
    assert jnp.allclose(slope, jnp.ones_like(slope))


def _count_rhie_chow_assemblies(monkeypatch):
    """A mutable ``[count]`` incremented on each lagged-``a_P`` Rhie--Chow assembly (see the seam)."""
    calls = [0]
    original = MomentumContinuity.momentum_matrix_diagonal

    def counted(self, *args, **kwargs):
        calls[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MomentumContinuity, "momentum_matrix_diagonal", counted)
    return calls


def test_residual_assembles_the_flow_fields_once(monkeypatch) -> None:
    """The coupled residual re-derives the Rhie--Chow flow fields a single time per evaluation.

    The residual, the mass flux the scalars advect on, and the velocity gradient the closure reads all
    come from one :meth:`~aquaflux.flow.MomentumContinuity.flow_fields` assembly (the gradient is the
    lightweight one that does no ``a_P`` work), so the expensive lagged-``a_P`` Rhie--Chow assembly runs
    exactly once -- not once each for the residual, the mass flux, and the gradient.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    calls = _count_rhie_chow_assemblies(monkeypatch)
    calls[0] = 0
    coupled.residual(state)
    assert calls[0] == 1


def test_segregated_prologues_match_the_eager_assembly() -> None:
    """The jitted sweep prologues equal the eager accessor expressions they replace.

    ``_sweep_eddy_viscosity`` is the pre-solve ``nu_t`` from the velocity gradient; ``_sweep_closure``
    is the post-solve ``(mdot, closure)`` from a single flow-field assembly. Jitting and fusing them
    must not change the numbers (the driver's per-sweep assembly savings come for free). That the
    fused path assembles the Rhie--Chow flow fields only once is pinned by the eager
    ``test_residual_assembles_the_flow_fields_once`` and the momentum seam tests.
    """
    from aquaflux.turbulence.driver import _sweep_closure, _sweep_eddy_viscosity

    mesh, coupled = _cavity()
    momentum, turbulence = coupled.momentum, coupled.turbulence
    flow, k, omega = coupled.layout.unpack(_healthy_state(mesh, coupled))

    nu_t = _sweep_eddy_viscosity(momentum, turbulence, flow, k, omega)
    expected_nu_t = turbulence.eddy_viscosity(momentum.velocity_fields(flow).gradient, k, omega)
    assert jnp.allclose(nu_t, expected_nu_t)

    mdot, closure = _sweep_closure(momentum, turbulence, flow, k, omega)
    assert jnp.allclose(mdot, momentum.mass_flux(flow))
    expected_closure = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega)
    assert jnp.allclose(closure.nu_t, expected_closure.nu_t)
    assert jnp.allclose(closure.strain_rate, expected_closure.strain_rate)


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


def test_refresh_trigger_is_rejected_under_differentiation() -> None:
    """``refresh_trigger`` is a forward-only accelerator: it raises under ``jax.grad``, not leaks.

    The refresh re-derives the preconditioner from the mid-march state, which is a tracer when
    differentiating; the refreshed preconditioner would then capture that tracer and escape the
    converged solve's ``custom_vjp`` as an opaque ``UnexpectedTracerError``. A refresh also forbids an
    explicit (concrete) ``continuation``, so there is no way to build the preconditioner outside the
    trace -- the only honest behaviour is a clear up-front error. The guard fires before any solve, so
    this stays a fast test. Differentiating the single-stage solve (no trigger) remains the supported
    path and is exercised by the integration adjoint gate.
    """
    mesh, coupled = _cavity()
    flow, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))

    def objective(nu_scale):
        scaled = eqx.tree_at(
            lambda c: c.turbulence.molecular_viscosity,
            coupled,
            coupled.turbulence.molecular_viscosity * nu_scale,
        )
        f, _, _ = solve_coupled(
            scaled, flow, k, omega, rtol=1e-2, refresh=RefreshPolicy(trigger=CycleGrowthTrigger())
        )
        return jnp.sum(f**2)

    with pytest.raises(ValueError, match="forward-only eager march"):
        jax.grad(objective)(1.0)


class _TrivialShiftPolicy(eqx.Module):
    """A shift policy with a unit diagonal and no preconditioner -- enough to build a step object."""

    def shift_term(self, phi):
        return ShiftTerm(diagonal=jnp.ones_like(phi), make_preconditioner=lambda _relaxation: None)


def test_refresh_trigger_with_an_explicit_continuation_and_no_builder_is_rejected() -> None:
    """A refresh must rebuild the continuation, so an explicit one needs a ``refresh_builder``.

    Without a builder the refresh has no way to re-freeze the preconditioner, so the combination is
    rejected and the error names the supported alternatives. The guard is on the argument combination
    and fires before the continuation is ever stepped, so a trivial step object is sufficient here --
    no preconditioner needs to be built. (Supplying ``refresh_builder`` lifts the restriction, since the
    builder is how the refresh rebuilds -- exercised by the complete-LU refresh integration tests.)
    """
    mesh, coupled = _cavity()
    flow, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))
    with pytest.raises(ValueError, match="supplied with no"):
        solve_coupled(
            coupled,
            flow,
            k,
            omega,
            rtol=1e-2,
            continuation=PseudoTransientStep(_TrivialShiftPolicy()),
            refresh=RefreshPolicy(trigger=CycleGrowthTrigger()),
        )


def test_refreshing_the_policy_rebuilds_transport_and_carries_the_coordinate_factor() -> None:
    """A ``reuse=`` refresh rebuilds the shift's transport time scale and carries its coordinate factor.

    The shift diagonal is ``transport_diagonal * jacobian_scale``. Rebuilding the *product* at a
    developed state over-damps and freezes the coupled log-omega march, because under ``LogScalars``
    ``jacobian_scale(omega) = omega`` drags the field's growing range into the shift. Storing the two
    factors separately (issue #156) lets a refresh rebuild the transport time scale -- physics that
    should track the flow -- while carrying the coordinate factor frozen, so the temporal ratio
    ``transport(state)/transport(reference)`` has that range cancel. Pinned here at the mechanism,
    without a full separating march. The flow block is carried too; the scalar AMG refresh itself is
    pinned in ``test_scalar_transport_preconditioner``. ``method=None`` and the symmetric viscous
    velocity block keep the policy build robust to the two synthetic states.
    """
    from aquaflux.turbulence.coupled import _coupled_shift_policy

    mesh, base_coupled = _cavity()
    coupled = CoupledRANS.build(
        base_coupled.momentum, base_coupled.turbulence, omega_transform=LogScalars()
    )
    cold = _healthy_state(mesh, coupled, seed=0)
    developed = _healthy_state(mesh, coupled, seed=1)  # a *different*, more-developed reference

    kw = dict(velocity="smoothed")
    base = _coupled_shift_policy(coupled, cold, None, **kw)
    refreshed = _coupled_shift_policy(coupled, developed, None, base, **kw)
    rebuilt = _coupled_shift_policy(coupled, developed, None, **kw)

    # The coordinate factor (jacobian_scale) is carried from `base` frozen ...
    assert jnp.array_equal(refreshed.k_jacobian_scale, base.k_jacobian_scale)
    assert jnp.array_equal(refreshed.omega_jacobian_scale, base.omega_jacobian_scale)
    # ... while the transport time scale is rebuilt at the developed state (== a from-scratch build
    # there, and genuinely different from the cold-reference one, so the refresh actually tracks it).
    assert jnp.array_equal(refreshed.k_shift_transport, rebuilt.k_shift_transport)
    assert jnp.array_equal(refreshed.omega_shift_transport, rebuilt.omega_shift_transport)
    assert not jnp.allclose(refreshed.k_shift_transport, base.k_shift_transport)
    assert not jnp.allclose(refreshed.omega_shift_transport, base.omega_shift_transport)
    # The flow block is carried over (the expensive half; measured no help to re-freeze).
    assert refreshed.flow_preconditioner is base.flow_preconditioner


def test_refresh_carries_the_block_scaled_progress_norm_fixed_at_the_initial_state() -> None:
    """A refresh reuses the initial residual measure, so block-scaled progress does not re-base (#156 s4).

    ``BlockScaledNorm`` is self-normalising: at the state its per-block scales were built at it returns
    ``sqrt(n_blocks)``. If a refresh rebuilt it at the developed state, every ``residual_ratio`` would
    jump back toward one and the convergence test become unreachable. ``coupled_continuation`` with an
    explicit ``residual_norm`` (what ``solve_coupled`` passes on every refresh) uses it verbatim instead
    of rebuilding, so the measure stays fixed at the state the global progress reference was measured at.
    """
    mesh, coupled = _cavity()
    cold = _healthy_state(mesh, coupled, seed=0)
    developed = _healthy_state(mesh, coupled, seed=1)
    kw = dict(method=None, block_scaled_norm=True, velocity="smoothed")

    base = coupled_continuation(coupled, cold, **kw)
    base_norm = base.norm()
    refreshed = coupled_continuation(
        coupled, developed, reuse=base.shift_policy, residual_norm=base_norm, **kw
    )
    # The refreshed continuation measures progress with the *same* norm object, not a re-based one.
    assert refreshed.norm() is base_norm
    # And that carry matters: a from-scratch rebuild at the developed state re-bases the per-block
    # scales, so it scores the same residual differently (it self-normalises to sqrt(n_blocks) there).
    rebuilt = coupled_continuation(coupled, developed, **kw)
    r_dev = coupled.residual(developed)
    assert not jnp.allclose(base_norm(r_dev), rebuilt.norm()(r_dev))


def test_fixation_rows_take_their_own_derivative_not_the_chain_factor() -> None:
    """The per-row Jacobian scale is ``phi`` on transport rows but **one** on the fixation rows.

    Regression test. The scalar block's frozen preconditioner is built for the physical operator and
    rescaled by ``1 / (d(row)/d(w))``; the frozen operator carries a unit identity row at each fixed
    cell. Rescaling those rows by the block-wide chain factor ``d(phi)/d(w) = phi`` instead of the
    fixation row's own unit derivative leaves the preconditioned operator with a cluster of ``1/phi``
    eigenvalues -- measured at ~1e-5 on a wall-resolved mesh -- which stalls the Krylov solve.
    """
    omega = jnp.array([10.0, 1.0e5, 3.0, 250.0])
    fixed = jnp.array([1, 3])
    transport = jnp.array([0, 2])

    scale = _row_jacobian_scale(LogScalars(), omega, fixed)

    assert jnp.allclose(scale[transport], omega[transport])
    assert jnp.allclose(scale[fixed], 1.0)


def test_row_jacobian_scale_is_all_ones_for_a_directly_solved_scalar() -> None:
    """The directly-solved path keeps a unit scale with or without fixed cells, so it is unchanged."""
    omega = jnp.array([10.0, 1.0e5, 3.0, 250.0])
    assert jnp.allclose(_row_jacobian_scale(DirectScalars(), omega), 1.0)
    assert jnp.allclose(_row_jacobian_scale(DirectScalars(), omega, jnp.array([1, 3])), 1.0)


def test_eddy_viscosity_drift_is_zero_at_its_reference_and_grows_away_from_it() -> None:
    """The staleness measure a drift trigger fires on: relative movement of ``nu_t``.

    Zero at the state the preconditioner was frozen at (nothing has gone stale yet) and positive
    once the turbulence field moves, which is what makes it a direct staleness signal rather than an
    inference from solver cost.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    drift = eddy_viscosity_drift(coupled, state)

    assert float(drift(state)) == pytest.approx(0.0, abs=1e-12)

    # Raise k, which raises nu_t = k / omega: the frozen scalar operators no longer describe this
    # state, and the measure must say so.
    flow, k, omega = coupled.physical_fields(state)
    moved = coupled.state_from_physical(flow, 1.5 * k, omega)
    assert float(drift(moved)) > 0.1


def test_eddy_viscosity_drift_matches_its_definition() -> None:
    """The measure is exactly the relative L2 movement of ``nu_t`` -- pinned against a direct compute.

    Stated as the definition rather than as an invariance: ``nu_t`` is **not** proportional to ``k``
    once the shear limiter ``a1 k / max(a1 omega, S F2)`` engages, so properties that assume
    homogeneity in ``k`` do not hold, and asserting one would be testing the closure rather than the
    measure.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    flow, k, omega = coupled.physical_fields(state)
    moved = coupled.state_from_physical(flow, 1.3 * k, 0.8 * omega)

    reference_nu_t = coupled.eddy_viscosity(state)
    expected = float(
        jnp.linalg.norm(coupled.eddy_viscosity(moved) - reference_nu_t)
        / jnp.linalg.norm(reference_nu_t)
    )
    assert float(eddy_viscosity_drift(coupled, state)(moved)) == pytest.approx(expected, rel=1e-10)
    assert expected > 0.0


def test_live_velocity_shift_parts_use_the_current_eddy_viscosity() -> None:
    """The injected live source reproduces the momentum diagonal at the state's own ``nu_t``.

    The velocity shift's buckets are a *local time scale* and must describe the operator being solved,
    so they have to see the current effective viscosity rather than one frozen at a reference state.
    Pinned by composing the closure by hand and comparing.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    flow, k, omega = coupled.physical_fields(state)

    closure = coupled.turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega)
    live_assembler = coupled.momentum.with_eddy_viscosity(
        closure.nu_t, coupled.turbulence.wall_face_eddy_viscosity(k)
    )
    velocity, _pressure = live_assembler.unpack(flow)
    expected = live_assembler.momentum_matrix_diagonal_parts(velocity)

    source = LiveViscosityVelocityParts(
        coupled.momentum, coupled.turbulence, coupled.k_transform, coupled.omega_transform
    )
    flow_block, k_solved, omega_solved = coupled.layout.unpack(state)
    got = source.parts(flow_block, k_solved, omega_solved)

    for mine, theirs in zip(got, expected, strict=True):
        assert jnp.allclose(mine, theirs)


def test_live_velocity_shift_parts_map_the_solved_unknown_back_to_physical() -> None:
    """The buckets are the same whether omega is solved directly or in log form.

    The source receives the block **as solved**, so under a log parametrization it must exponentiate
    before forming the closure. Without that it would build the shift from ``log(omega)`` as if it were
    ``omega`` -- a silent, badly wrong local time scale.
    """
    mesh, direct = _cavity()
    log_omega = CoupledRANS.build(direct.momentum, direct.turbulence, omega_transform=LogScalars())
    physical = _healthy_state(mesh, direct)
    flow, k, omega = direct.layout.unpack(physical)

    direct_parts = LiveViscosityVelocityParts(
        direct.momentum, direct.turbulence, direct.k_transform, direct.omega_transform
    ).parts(flow, k, omega)

    solved = log_omega.state_from_physical(flow, k, omega)
    flow_l, k_l, omega_l = log_omega.layout.unpack(solved)
    log_parts = LiveViscosityVelocityParts(
        log_omega.momentum, log_omega.turbulence, log_omega.k_transform, log_omega.omega_transform
    ).parts(flow_l, k_l, omega_l)

    for a, b in zip(direct_parts, log_parts, strict=True):
        assert jnp.allclose(a, b)


class _TrivialShiftPolicy(eqx.Module):
    """A minimal shift policy for constructing a step without a mesh (see test_step_control.py)."""

    def shift_term(self, phi):
        return ShiftTerm(diagonal=jnp.ones_like(phi), make_preconditioner=lambda _r: None)


def _dual_time_step():
    from aquaflux.solve import DualTimeStep, SwitchedEvolutionRelaxation

    return DualTimeStep(
        _TrivialShiftPolicy(),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=2.0),
        inner_steps=4,
    )


def _single_step():
    from aquaflux.solve import SwitchedEvolutionRelaxation

    return PseudoTransientStep(
        _TrivialShiftPolicy(), relaxation_schedule=SwitchedEvolutionRelaxation(beta0=2.0)
    )


def test_dual_time_observed_march_defaults_to_the_courant_step_control() -> None:
    """A dual-time march that is observing but was given no control defaults to ``DualTimeControl``."""
    from aquaflux.solve import DualTimeControl, default_dual_time_control

    control = default_dual_time_control(None, observing=True, continuation=_dual_time_step())
    assert isinstance(control, DualTimeControl)


def test_single_step_observed_march_gets_no_default_control() -> None:
    """A single-step (pseudo-transient) march is not a dual-time step, so no control is injected."""
    from aquaflux.solve import default_dual_time_control

    assert default_dual_time_control(None, observing=True, continuation=_single_step()) is None


def test_a_caller_supplied_control_is_never_overridden() -> None:
    """An explicit control on a dual-time observed march is returned unchanged (the override path)."""
    from aquaflux.solve import ResidualRatioDualTimeControl, default_dual_time_control

    explicit = ResidualRatioDualTimeControl(beta_start=0.5)
    assert (
        default_dual_time_control(explicit, observing=True, continuation=_dual_time_step())
        is explicit
    )


def test_a_non_observing_dual_time_march_gets_no_default_control() -> None:
    """Not observing (the differentiable single-stage path) => no control injected, so no forward-only
    control can make the grad path raise the observe-under-trace guard."""
    from aquaflux.solve import default_dual_time_control

    assert default_dual_time_control(None, observing=False, continuation=_dual_time_step()) is None


def test_the_equation_names_follow_the_flat_state_layout() -> None:
    """One home for these names, so a per-block residual and a per-field change cannot label the same
    equation differently -- which is exactly how a log stops being joinable."""
    assert coupled_equation_names(3) == ("u", "v", "w", "p", "k", "omega")
    assert coupled_equation_names(2) == ("u", "v", "p", "k", "omega")
    with pytest.raises(ValueError, match="exceeds the named velocity components"):
        coupled_equation_names(4)


def test_coupled_fields_splits_velocity_per_component_and_names_omega() -> None:
    """A single vector entry averages the components, hiding one that has stopped moving behind two
    that have not -- and each component has its own momentum equation to line up against."""
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)

    fields = coupled_fields(coupled)(state)

    assert list(fields) == ["u", "v", "p", "k", "omega", "nut"]
    assert fields["u"].shape == (mesh.n_cells,)  # a component, not the (n, dim) vector
    velocity, _ = coupled.momentum.unpack(coupled.physical_fields(state)[0])
    assert jnp.allclose(fields["v"], velocity[:, 1])


def test_the_per_equation_residuals_compose_into_the_march_s_own_measure() -> None:
    """They are read on the same scale as the scalar residual beside them, which only holds if they
    are the very numbers that scalar is built from -- not a separately-scaled lookalike."""
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    engine = coupled_continuation(coupled, state, method=None)

    reported = coupled_residuals(coupled, engine)(state)

    assert list(reported) == list(coupled_equation_names(mesh.dim))
    measure = coupled_scaled_norm(coupled, engine.shift_policy, state)
    assert float(jnp.linalg.norm(jnp.array(list(reported.values())))) == pytest.approx(
        float(measure(coupled.residual(state))), rel=1e-10
    )


def test_the_per_equation_rows_add_up_to_the_residual_the_march_reports() -> None:
    """`forward_march` equilibrates at the state each outer iteration STARTS from and holds that
    measure for the whole iteration -- so the step it reports is ``norm_at_start(R(state_at_end))``.
    Scaling at the end state instead would measure the right residual in the wrong scales, and the
    rows would not add up to the number printed above them.
    """
    mesh, coupled = _cavity()
    start = _healthy_state(mesh, coupled, seed=0)
    end = _healthy_state(mesh, coupled, seed=1)
    engine = coupled_continuation(coupled, start, method=None)
    reported_by_march = float(
        coupled_scaled_norm(coupled, engine.shift_policy, start)(coupled.residual(end))
    )

    residuals = coupled_residuals(coupled, engine, start)
    rows = residuals(end)  # the first observed step: starts at `start`, ends at `end`

    assert float(jnp.linalg.norm(jnp.array(list(rows.values())))) == pytest.approx(
        reported_by_march, rel=1e-12
    )


def test_each_step_equilibrates_at_the_state_it_started_from() -> None:
    """The seed covers only the first step; from then on the previous state IS the start state, which
    is what keeps a whole march's rows consistent rather than just its opening step."""
    mesh, coupled = _cavity()
    first = _healthy_state(mesh, coupled, seed=0)
    second = _healthy_state(mesh, coupled, seed=1)
    third = _healthy_state(mesh, coupled, seed=2)
    engine = coupled_continuation(coupled, first, method=None)

    residuals = coupled_residuals(coupled, engine, first)
    residuals(second)  # step 1 consumes the seed and records `second`
    rows = residuals(third)  # step 2 must equilibrate at `second`, not at `third`

    expected = float(
        coupled_scaled_norm(coupled, engine.shift_policy, second)(coupled.residual(third))
    )
    assert float(jnp.linalg.norm(jnp.array(list(rows.values())))) == pytest.approx(
        expected, rel=1e-12
    )


def test_state_drift_forces_a_full_refresh_whatever_the_beta_gate_says() -> None:
    """The two staleness gates are combined, NOT nested -- the regression this function was extracted for.

    Below the preconditioner's shift floor the clamped β never moves, so the β gate answers "no change"
    on every step forever. When the state gate was asked only *after* the β gate had already said yes,
    that made eddy-viscosity drift unable to trigger anything at all in exactly the low-shift tail where
    the flow develops fastest.
    """
    from aquaflux.turbulence.coupled import _refresh_branch

    assert _refresh_branch(stale_state=True, moved_beta=False, split=True) == "full"
    assert _refresh_branch(stale_state=True, moved_beta=True, split=True) == "full"


def test_a_moved_beta_alone_takes_the_cheap_shift_branch() -> None:
    """A matching Jacobian with a mismatched shift needs only the diagonal re-added, not a re-probe."""
    from aquaflux.turbulence.coupled import _refresh_branch

    assert _refresh_branch(stale_state=False, moved_beta=True, split=True) == "shift"


def test_without_the_split_any_trigger_is_a_full_refresh() -> None:
    """A preconditioner with no shift-only path (the factorization ones) has a single branch."""
    from aquaflux.turbulence.coupled import _refresh_branch

    assert _refresh_branch(stale_state=False, moved_beta=True, split=False) == "full"
    assert _refresh_branch(stale_state=False, moved_beta=False, split=False) == "none"


def test_neither_gate_firing_reuses_the_standing_factorization() -> None:
    """The gates exist to skip work; both quiet must still mean no refresh."""
    from aquaflux.turbulence.coupled import _refresh_branch

    assert _refresh_branch(stale_state=False, moved_beta=False, split=True) == "none"


_DRIFT_TRACES: list[int] = []


class _CountingEddyViscosity(eqx.Module):
    """A stand-in for the coupled assembler that records each TRACE of its eddy viscosity.

    The body runs at trace time only, so the recorded count is the compilation count -- the repo's
    trace-counting idiom, used here because ``equinox``'s jit wrapper exposes no cache-clearing handle.
    """

    gain: jnp.ndarray

    def eddy_viscosity(self, state: jnp.ndarray) -> jnp.ndarray:
        _DRIFT_TRACES.append(1)
        return self.gain * state


def test_rebasing_the_drift_measure_is_a_compilation_cache_hit() -> None:
    """Re-basing the staleness reference must change a VALUE, not build a new compiled function.

    ``_materialize_gate`` re-bases this measure at every materialize, so a per-reference compilation is
    paid on every full preconditioner refresh. Measured on a three-dimensional coupled march before the
    fix: ~3.8 s each time, ~21 % of the refresh, for a number that is one norm of an already-computed
    field. The reference therefore rides as an argument to a module-level jitted function rather than as
    a captured constant of a locally-defined one, which ``filter_jit`` caches per closure.
    """
    from aquaflux.turbulence.coupled import _eddy_viscosity_drift

    # A unique size, so a would-be recompile cannot be a cache hit from another test (the cache is
    # process-global) and a genuine hit cannot be manufactured by one.
    coupled = _CountingEddyViscosity(gain=jnp.asarray(2.0))
    state = jnp.linspace(1.0, 2.0, 37)
    scale = jnp.asarray(1.0)
    # Every reference is built the SAME way, as production's `coupled.eddy_viscosity(...)` is. Mixing
    # constructors here would compare two abstract values -- `jnp.zeros(n)` is strongly typed while
    # `jnp.full(n, 1.0)` is weakly typed -- and a weak/strong mismatch is itself a cache miss, so the
    # test would fail for a reason that has nothing to do with what it is checking.
    references = [jnp.linspace(0.0, offset, 37) for offset in (0.5, 1.0, 2.0, 3.0)]

    _DRIFT_TRACES.clear()
    float(_eddy_viscosity_drift(coupled, state, references[0], scale))
    compiled = len(_DRIFT_TRACES)
    assert compiled == 1

    for reference in references[1:]:  # three re-bases, as three materializes would do
        float(_eddy_viscosity_drift(coupled, state, reference, scale))

    assert len(_DRIFT_TRACES) == compiled


def test_the_drift_measure_is_zero_at_its_own_reference() -> None:
    """A re-based measure reports no movement until the state actually moves -- else it re-fires at once."""
    from aquaflux.turbulence import eddy_viscosity_drift

    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled, seed=0)

    assert float(eddy_viscosity_drift(coupled, state)(state)) == pytest.approx(0.0, abs=1e-12)


def test_scaling_the_viscosity_leaves_the_pytree_structure_identical() -> None:
    """A Reynolds-continuation rung changes leaf VALUES only -- which is what makes a cache hit possible.

    Every jitted quantity derived from the assembler can therefore be shared across the whole ramp, and
    anything that recompiles per rung is capturing the assembler rather than taking it as an argument.
    """
    _, coupled = _cavity()
    scaled = coupled.with_scaled_molecular_viscosity(0.1)

    assert jax.tree_util.tree_structure(coupled) == jax.tree_util.tree_structure(scaled)
    before = jax.tree_util.tree_leaves(coupled)
    after = jax.tree_util.tree_leaves(scaled)
    assert [jnp.shape(x) for x in before] == [jnp.shape(x) for x in after]
    assert [jnp.result_type(x) for x in before] == [jnp.result_type(x) for x in after]
    assert any(not jnp.array_equal(x, y) for x, y in zip(before, after, strict=True))


_PROBE_TRACES: list[int] = []


class _CountingResidual(eqx.Module):
    """A stand-in assembler recording each TRACE of its residual, so recompiles are countable."""

    gain: jnp.ndarray

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        _PROBE_TRACES.append(1)
        return self.gain * state**2


def test_the_jacobian_probe_is_a_cache_hit_across_reynolds_rungs() -> None:
    """The coloured probe must not recompile when the ramp rebuilds the assembler at a new viscosity.

    Each rung's first step was measured at 112/102/145 s more than that rung's median step *at an
    identical cycle count* -- compilation, repeated per rung. The probe is one contributor: written as a
    local ``jax.jit`` closure over the assembler it is a fresh cache entry per rung, so it takes the
    assembler as an argument instead. A rung differs only in leaf values (pinned by the test above), so
    a probe that takes the assembler as an argument is a hit.
    """
    from aquaflux.turbulence.coupled import _batched_jacobian_matvec, _jacobian_matvec

    state = jnp.linspace(1.0, 2.0, 29)  # a unique size; the compilation cache is process-global
    tangent = jnp.ones_like(state)
    seeds = jnp.stack([tangent, 0.5 * tangent])

    _PROBE_TRACES.clear()
    first = _CountingResidual(gain=jnp.asarray(1.0))
    _jacobian_matvec(first, state, tangent)
    _batched_jacobian_matvec(first, state, seeds)
    compiled = len(_PROBE_TRACES)
    assert compiled > 0  # sanity: the stub really is being traced

    for scale in (0.1, 0.01):  # two further rungs of a Reynolds ramp
        rung = _CountingResidual(gain=jnp.asarray(scale))
        _jacobian_matvec(rung, state, tangent)
        _batched_jacobian_matvec(rung, state, seeds)

    assert len(_PROBE_TRACES) == compiled


def test_the_adjoint_transpose_factory_compares_by_the_preconditioner_it_wraps() -> None:
    """Two engines sharing one preconditioner must produce EQUAL adjoint factories.

    The factory rides in the forward step's ``adjoint_preconditioner_factory``, a static field and so
    part of the compiled step's cache key. As a lambda it compared by identity, which meant a rung that
    rebuilt its engine recompiled the whole coupled solve even when it was reusing the very same
    preconditioner -- defeating the point of reusing it.
    """
    from aquaflux.solve import TransposedPreconditioner
    from aquaflux.turbulence.coupled import FrozenTransposeFactory

    class _Pc:
        def matvec(self, *, transpose: bool = False):
            return lambda v: v

    first, second = _Pc(), _Pc()
    assert FrozenTransposeFactory(first) == FrozenTransposeFactory(first)
    assert FrozenTransposeFactory(first) != FrozenTransposeFactory(second)
    # ...and the wrapper must not throw that equality away again.
    assert TransposedPreconditioner(FrozenTransposeFactory(first)) == TransposedPreconditioner(
        FrozenTransposeFactory(first)
    )
    assert TransposedPreconditioner(FrozenTransposeFactory(first)) != TransposedPreconditioner(
        FrozenTransposeFactory(second)
    )


def test_the_frozen_transpose_factory_ignores_the_state_it_is_given() -> None:
    """The factorization is frozen, so the same transpose serves every state -- which is what lets this
    be a value object at all."""
    from aquaflux.turbulence.coupled import FrozenTransposeFactory

    class _Pc:
        def __init__(self):
            self.calls = 0

        def matvec(self, *, transpose: bool = False):
            self.calls += 1
            assert transpose
            return lambda v: 2.0 * v

    pc = _Pc()
    factory = FrozenTransposeFactory(pc)
    assert float(factory(jnp.ones(3))(jnp.ones(3))[0]) == 2.0
    assert float(factory(jnp.zeros(3))(jnp.ones(3))[0]) == 2.0


def test_the_k_positivity_builders_address_the_k_block_and_defer_to_the_transform() -> None:
    """Both positivity constructions target the ``k`` slice, and both stand down for a log variable.

    Worth its own test for two reasons. The slice ``((dim + 1) n, (dim + 2) n)`` is block-order
    knowledge, so a builder that computed it independently would drift silently when the order
    changed -- here both read one helper, and this pins the answer. And these builders are the only
    place the projection is constructed for a coupled case, so a missing import in the module would
    otherwise surface for the first time in the middle of a march rather than here.

    The builders read only the transform and the block layout, so a stub carrying those two is a
    sufficient collaborator -- no mesh, no assembled case.
    """
    from types import SimpleNamespace

    from aquaflux.turbulence import positive_k_limit, positive_k_projection
    from aquaflux.turbulence.coupled import DirectScalars, LogScalars

    n, dim = 7, 3
    direct = SimpleNamespace(
        k_transform=DirectScalars(), layout=SimpleNamespace(n_cells=n, dim=dim)
    )

    cap = positive_k_limit(direct)
    project = positive_k_projection(direct)
    assert (cap.start, cap.stop) == ((dim + 1) * n, (dim + 2) * n)
    assert (project.start, project.stop) == (cap.start, cap.stop)

    # ...and each acts on that slice only. One dead k cell, one healthy, velocities driven hard.
    phi = jnp.ones((dim + 3) * n)
    phi = phi.at[cap.start].set(1.0e-12)
    delta = -jnp.ones_like(phi)
    assert float(cap(phi, delta)) < 1.0e-9  # the dead cell throttles the whole step
    clipped = project(phi, delta)
    assert float(clipped[cap.start]) == pytest.approx(-0.99e-12)  # ...held back alone
    assert float(clipped[0]) == -1.0  # a velocity entry is untouched
    assert float(cap(phi, clipped)) == pytest.approx(1.0)  # the cap now finds nothing binding

    # A log variable is positive by construction, so neither constrains it.
    logged = SimpleNamespace(k_transform=LogScalars(), layout=SimpleNamespace(n_cells=n, dim=dim))
    assert positive_k_limit(logged) is None
    assert positive_k_projection(logged) is None


def test_the_probe_is_the_same_for_every_reynolds_rung() -> None:
    """The colouring plan and its de-compression map depend on the MESH, never on the viscosity.

    That is the whole licence for building one :class:`CoupledJacobianProbe` and handing it to every
    continuation rung's step and to the refresh hook beside it. Without it a three-rung ramp built six
    copies of the largest allocation a three-dimensional case makes, and the assertion that they would
    all have been identical was never checked.
    """
    import numpy as np
    from aquaflux.turbulence import CoupledJacobianProbe

    _, coupled = _cavity()
    probe = CoupledJacobianProbe.build(coupled, stencil_reach=2)
    scaled = CoupledJacobianProbe.build(coupled.with_scaled_molecular_viscosity(100.0), 2)

    assert probe.plan.n_probes == scaled.plan.n_probes
    assert probe.plan.n_fields == scaled.plan.n_fields
    assert np.array_equal(probe.structure.indptr, scaled.structure.indptr)
    assert np.array_equal(probe.structure.indices, scaled.structure.indices)


def test_staleness_beta_gate_fires_on_first_call_beta_move_and_staleness_cap() -> None:
    """The β-tracking gate re-factors on the first step, on a β move past the threshold, or at the
    staleness cap -- and skips otherwise, so an expensive re-factor is paid only when it pays off.

    Pure logic, no solver: the gate is the whole novelty of a gated β-tracking refresh (the refactor
    mechanism itself is shared machinery), so it earns a fast, isolated test.
    """
    from aquaflux.turbulence.coupled import _staleness_beta_gate

    gate = _staleness_beta_gate(refresh_every=3, beta_rel_change=0.25)
    assert gate(1.0) is True  # first call always fires (nothing factored yet)
    assert gate(1.1) is False  # +10% < 25%, 1 step since -> reuse
    assert gate(1.2) is False  # +20% < 25% (vs last-refresh 1.0), 2 steps since -> reuse
    assert gate(1.0) is True  # 3 steps since -> staleness cap fires (state-development bound)
    assert (
        gate(1.4) is True
    )  # +40% > 25% vs last-refresh 1.0 -> β-move fires (the anti-stall trigger)
    assert gate(1.4) is False  # unchanged, 1 step since -> reuse


def test_materialize_gate_fires_on_drift_and_the_step_cap() -> None:
    """The β-diagonal split's materialize gate re-materializes the Jacobian only when the coefficient has
    drifted past the threshold since the last materialize, or at the step cap -- so the expensive full
    re-probe is reserved for a genuinely stale Jacobian and the cheap shift-only refresh carries the rest.

    Pure logic with an injected synthetic drift measure (``drift = |state - reference|``): the gate's
    decision -- first-call seeding without a redundant materialize, drift-move, step-cap, and re-basing the
    reference at every materialize -- is the whole novelty; the materialize itself is shared machinery.
    """
    from aquaflux.turbulence.coupled import _materialize_gate

    def drift_factory(reference):
        ref = float(reference)
        return lambda state: abs(float(state) - ref)

    # Drift only: seed at the first state (no redundant materialize), then fire on a >0.5 move, re-basing.
    gate = _materialize_gate(drift_factory, materialize_drift=0.5, materialize_every=None)
    assert (
        gate(jnp.asarray(0.0)) is False
    )  # first call seeds the reference; Jacobian is fresh from build
    assert gate(jnp.asarray(0.3)) is False  # drift 0.3 < 0.5 -> shift-only
    assert (
        gate(jnp.asarray(0.6)) is True
    )  # drift 0.6 > 0.5 -> materialize, re-base reference to 0.6
    assert gate(jnp.asarray(0.7)) is False  # drift 0.1 vs 0.6 -> shift-only (re-based, not vs 0.0)
    assert gate(jnp.asarray(1.2)) is True  # drift 0.6 vs 0.6 -> materialize again

    # Step cap only (no drift trigger): fire every 3rd refresh regardless of state.
    cap = _materialize_gate(drift_factory, materialize_drift=None, materialize_every=3)
    assert [cap(jnp.asarray(0.0)) for _ in range(6)] == [False, False, True, False, False, True]


def test_the_materialize_gate_forgets_its_reference_on_reset() -> None:
    """``reset`` discards the drift reference, so a rebound hook cannot compare across two problems.

    The gate measures how far the eddy viscosity has moved since the Jacobian was last probed. Point the
    hook at the next Reynolds rung's companion and that reference belongs to a different Reynolds
    number, so the drift it reports is a viscosity difference rather than flow development. Resetting
    makes the next call re-seed against the state it is actually handed.
    """
    from aquaflux.turbulence.coupled import _materialize_gate

    seen: list[float] = []

    def drift_factory(reference):
        seen.append(float(reference))
        return lambda state: jnp.abs(state - reference)

    gate = _materialize_gate(drift_factory, materialize_drift=0.5, materialize_every=None)

    assert gate(jnp.asarray(1.0)) is False  # seeds at 1.0; zero drift against its own reference
    assert seen == [1.0]
    assert gate(jnp.asarray(1.2)) is False  # still inside the threshold, reference unchanged
    assert seen == [1.0]

    gate.reset()
    assert gate(jnp.asarray(1.2)) is False  # re-seeded at 1.2 rather than fired against the old 1.0
    assert seen == [1.0, 1.2]
    assert gate(jnp.asarray(2.0)) is True  # ...and it still fires on a genuine move from there


class _ScalarRans(eqx.Module):
    """A one-line stand-in assembler whose Jacobian is a scalar, so a rebind is visible in one apply."""

    gain: jnp.ndarray

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.gain * state


class _RecordingPreconditioner:
    """A frozen inverse that records what each refresh was asked to build, and builds nothing.

    Deliberately does NOT expose ``refresh_shift_in_place``: without a cheap branch every refresh is a
    full re-materialize, which is the decision under test here.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def refresh_in_place(self, matvec, plan, shift_diagonal, **_kwargs):
        self.calls.append({"matvec": matvec, "shift": shift_diagonal})
        return ()


def _stub_step(preconditioner, beta, diagonal):
    """The smallest forward step the refresh hook reads: a shift strength, a policy and its diagonal."""
    from types import SimpleNamespace

    from aquaflux.solve import ShiftTerm

    base = SimpleNamespace(shift_term=lambda _phi: ShiftTerm(diagonal, lambda _relaxation: None))
    return SimpleNamespace(
        relaxation_schedule=SimpleNamespace(beta=beta),
        shift_policy=SimpleNamespace(preconditioner=preconditioner, base=base),
    )


def test_rebinding_the_refresh_swaps_the_case_and_forces_a_full_rebuild() -> None:
    """One refresh hook can serve a whole Reynolds ramp, which is what lets the ramp share one V-cycle.

    A rung boundary is invisible to both gates -- one watches the shift strength, the other the eddy
    viscosity's drift *within* one case -- so a hook whose gate had gone quiet would leave the next rung
    solving against a V-cycle fitted to the previous rung's viscosity. ``rebind`` therefore does two
    things, and both are asserted: the Jacobian probe starts reporting the NEW companion's derivative,
    and the next refresh is a full re-materialize whatever the gates make of it.
    """

    import numpy as np
    from aquaflux.turbulence.coupled import _beta_tracking_refresh, _staleness_beta_gate

    state = jnp.linspace(1.0, 2.0, 5)
    diagonal = jnp.full(5, 2.0)
    tangent = jnp.ones(5)
    # The real probe (its plan and gather map are unused here), not a lookalike: `_beta_tracking_refresh`
    # asks it which assembler to differentiate, which only the class itself can answer.
    probe = CoupledJacobianProbe(plan=object(), structure=object())

    pc = _RecordingPreconditioner()
    step = _stub_step(pc, beta=0.5, diagonal=diagonal)
    # A gate that fires once (its initializing call) and then never again, which is the shipped bfs3d
    # configuration: the cost trigger replaces the schedule, so nothing else may rebuild the V-cycle.
    refresh = _beta_tracking_refresh(
        _ScalarRans(gain=jnp.asarray(3.0)),
        stencil_reach=2,
        probe=probe,
        gate=_staleness_beta_gate(refresh_every=10**9, beta_rel_change=float("inf")),
    )

    refresh(step, state)  # the initializing call
    assert len(pc.calls) == 1
    assert np.allclose(pc.calls[0]["shift"], 0.5 * np.asarray(diagonal))
    assert np.allclose(pc.calls[0]["matvec"](tangent), 3.0 * tangent)

    refresh(step, state)  # the gate has gone quiet, as it does for the rest of a rung
    assert len(pc.calls) == 1

    refresh.rebind(_ScalarRans(gain=jnp.asarray(7.0)))
    refresh(step, state)
    assert len(pc.calls) == 2  # forced, though the gate is still quiet
    assert np.allclose(pc.calls[1]["matvec"](tangent), 7.0 * tangent)  # ...at the new companion

    refresh(step, state)  # and the force is spent: one rebuild per rebind, not a stuck flag
    assert len(pc.calls) == 2


def test_the_default_refresh_policy_is_the_inert_one() -> None:
    """``NO_REFRESH`` must stay exactly the settings a single-stage solve had before the policy existed.

    Pinned as *values*: every solve runs this policy unless it says otherwise, so a default moved here
    silently turns an unobserved solve into an observed one (or the reverse), which changes how
    ``max_steps`` is spent and whether the solve can be differentiated at all.
    """
    assert NO_REFRESH == RefreshPolicy()
    assert NO_REFRESH.trigger is None
    assert NO_REFRESH.limit == 1
    assert NO_REFRESH.builder is None
    assert NO_REFRESH.precondition_step is None
    # The default must not refresh and must not force the observed march.
    assert not NO_REFRESH.refreshes
    assert not NO_REFRESH.observes


def test_a_refresh_needs_both_a_trigger_and_a_budget() -> None:
    """``refreshes`` is the conjunction: a trigger with no budget refreshes nothing, and vice versa.

    ``limit=0`` is the documented way to disable refreshing while leaving a trigger in place, so
    reading this as "a trigger is set" would quietly ignore that.
    """
    assert RefreshPolicy(trigger=object()).refreshes
    assert not RefreshPolicy(trigger=object(), limit=0).refreshes
    assert not RefreshPolicy(limit=5).refreshes


def test_a_builder_alone_does_not_make_a_march_observed() -> None:
    """A builder with no trigger is called once, for the initial build -- which needs no eager march.

    Getting this wrong would silently force the observed path (and its doubled ``max_steps`` budget,
    and its ban under ``jax.grad``) on a solve that only wanted a custom way to construct its step.
    """
    assert not RefreshPolicy(builder=lambda state: state).observes
    assert RefreshPolicy(trigger=object()).observes
    assert RefreshPolicy(precondition_step=lambda step, state: None).observes


def test_segments_is_one_more_than_the_refresh_budget() -> None:
    """``limit`` refreshes means ``limit + 1`` segments; the last one must still be marched.

    Off by one here and the freshly refreshed preconditioner is never used by an observed step -- only
    by the finishing solve -- so the refresh it just paid for buys nothing.
    """
    assert RefreshPolicy(limit=0).segments == 1
    assert RefreshPolicy(limit=3).segments == 4
    policy = RefreshPolicy(limit=2)
    assert [policy.is_last_segment(i) for i in range(policy.segments)] == [False, False, True]


def test_a_supplied_step_with_no_builder_is_rejected_when_a_refresh_is_configured() -> None:
    """A refresh rebuilds the step, so a caller-supplied step with no builder leaves nothing to rebuild.

    Silently not refreshing would be the harmful outcome: the solve would run as a single stage while
    the caller believed it was re-freezing the preconditioner.
    """
    step = object()
    with pytest.raises(ValueError, match="builder"):
        RefreshPolicy(trigger=object()).require_rebuildable(step)

    # Every way out the message names must actually work.
    RefreshPolicy(trigger=object(), builder=lambda s: s).require_rebuildable(step)
    RefreshPolicy(trigger=object()).require_rebuildable(None)
    RefreshPolicy().require_rebuildable(step)
    RefreshPolicy(trigger=object(), limit=0).require_rebuildable(step)


def test_globalization_knobs_still_reach_the_continuation_builder(monkeypatch) -> None:
    """``grow`` / ``descent_backoff`` / ``descent_test`` are no longer named on ``solve_coupled``, and
    still arrive at :func:`coupled_continuation` unchanged -- they ride ``**continuation_kwargs``.

    They used to be declared on ``solve_coupled`` *and* forwarded explicitly, while the very same call
    sites already splatted ``**continuation_kwargs`` into the same function -- so the declarations were
    pure duplication, costing three parameters on an already-wide signature to buy nothing. Deleting
    them is call-for-call identical, and this pins that: it is the only thing standing between the
    deletion and a silently dropped knob.
    """
    from aquaflux.turbulence import coupled as coupled_module

    _, coupled = _cavity(4)
    seen: dict = {}

    def spy(assembler, reference_state, **kwargs):
        seen.update(kwargs)
        raise _StopBuild

    monkeypatch.setattr(coupled_module, "coupled_continuation", spy)
    with pytest.raises(_StopBuild):
        solve_coupled(coupled, grow=2, descent_backoff=3, descent_test=True, beta0=1.5)

    assert seen["grow"] == 2
    assert seen["descent_backoff"] == 3
    assert seen["descent_test"] is True
    # An ordinary continuation knob rides the same path, so the mechanism is not special-cased.
    assert seen["beta0"] == 1.5


class _StopBuild(Exception):
    """Aborts ``solve_coupled`` once the continuation build has been observed."""


def test_the_production_limiter_defaults_to_the_exact_operator() -> None:
    """``explicit_production_limiter`` is OFF by default, so the coupled adjoint is exact by default.

    It defaulted to ``True``, which put a ``stop_gradient`` inside the residual of every coupled solve
    and contradicted ``KProduction``'s own documented contract ("the coupled sensitivity residual uses
    the exact operator so the adjoint stays exact"). Pinned as a value: a default moved back silently
    re-arms a hazard whose whole character is that it is invisible -- finite gradients, wrong.
    """
    _, coupled = _cavity(4)
    assert coupled.turbulence.explicit_production_limiter is False
    assert (
        inspect.signature(SSTTurbulence.build).parameters["explicit_production_limiter"].default
        is False
    )


def test_a_root_the_frozen_cap_invalidates_is_refused() -> None:
    """With the limiter opted into AND the cap active at the root, the solve refuses to return.

    The forward fields would be perfectly good, so nothing else would ever surface this -- the damage
    is confined to a gradient that comes back finite and wrong. The guard is the only thing standing
    between that and a published sensitivity.
    """
    from aquaflux.turbulence.coupled import _reject_a_root_the_frozen_cap_invalidates

    _, exact = _cavity(4)
    # `dataclasses.replace`, not `eqx.tree_at`: the flag is a STATIC field, so it lives in the
    # treedef rather than among the leaves and `tree_at` (which addresses leaves) cannot reach it.
    frozen = dataclasses.replace(
        exact,
        turbulence=dataclasses.replace(exact.turbulence, explicit_production_limiter=True),
    )
    flow, k, omega = hybrid_initialize(exact.momentum, exact.turbulence)
    quiet = exact.state_from_physical(flow, k, omega)
    # Shrinking omega raises S/omega, which is what the cap actually keys on -- the ratio must clear
    # sqrt(10 beta*) = 0.949 against an equilibrium value of 0.3, so a hundredfold is what it takes.
    binding = exact.state_from_physical(flow, k, omega * 1e-2)
    assert not bool(jnp.any(production_cap_active(exact, quiet)))
    assert bool(jnp.any(production_cap_active(exact, binding)))

    # The exact operator is never guarded, whatever the cap is doing -- there is nothing frozen.
    assert _reject_a_root_the_frozen_cap_invalidates(exact, binding) is binding

    # With the limiter opted into: an inactive cap leaves the root alone...
    jax.block_until_ready(_reject_a_root_the_frozen_cap_invalidates(frozen, quiet))
    # ...and an active one refuses it, because the gradient through it would be finite and wrong.
    with pytest.raises(eqx.EquinoxRuntimeError, match="production cap"):
        jax.block_until_ready(_reject_a_root_the_frozen_cap_invalidates(frozen, binding))


def test_the_cap_predicate_is_the_one_the_residual_uses() -> None:
    """``production_cap_active`` must agree with ``KProduction``'s own ``min``, cell for cell.

    Two spellings of "is the cap active" is exactly how a validity guard comes to clear a state the
    residual actually caps. They share ``production_and_limit`` so they cannot drift; this pins that
    they really do agree, rather than trusting the shared call.
    """
    _, coupled = _cavity(4)
    state = coupled.state_from_physical(*hybrid_initialize(coupled.momentum, coupled.turbulence))
    flow, k, omega = coupled.physical_fields(state)
    closure = coupled.turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega)

    production, limit = production_and_limit(
        closure.nu_t, closure.strain_rate, closure.omega, k, coupled.turbulence.model
    )
    # Where the mask is True the `min` must take the LIMIT; where False, the production.
    mask = production_cap_active(coupled, state)
    assert jnp.array_equal(mask, production > limit)
    assert jnp.allclose(jnp.where(mask, limit, production), jnp.minimum(production, limit))


# --- the probe's gradient-sweep cap ------------------------------------------------------


def _skewed_corrected_cavity(n=6, sweeps=4):
    """A cavity on a skewed mesh whose gradients carry the non-orthogonal correction."""
    mesh = perturbed_grid_2d(n, n, perturb=0.25, seed=1, named_boundaries=True)
    scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=sweeps, warn_tol=None))
    return _cavity(mesh=mesh, gradient=scheme)


def test_an_uncapped_probe_differentiates_the_assembler_itself() -> None:
    """The default is the residual as it stands -- returned by identity, so nothing downstream moves."""
    _, coupled = _skewed_corrected_cavity()
    probe = CoupledJacobianProbe.build(coupled, stencil_reach=2)
    assert probe.gradient_sweeps is None
    assert probe.narrow(coupled) is coupled


def test_a_capped_probe_narrows_every_gradient_solve_in_the_case() -> None:
    """Both blocks reconstruct gradients, and the cap has to reach the momentum block and the closure."""
    _, coupled = _skewed_corrected_cavity(sweeps=4)
    probed = CoupledJacobianProbe.build(coupled, stencil_reach=2, gradient_sweeps=2).narrow(coupled)
    assert coupled.momentum.gradient_scheme.solver.sweeps == 4  # the case itself is untouched
    assert probed.momentum.gradient_scheme.solver.sweeps == 2
    assert probed.turbulence.gradient_scheme.solver.sweeps == 2


def test_capping_the_probe_shrinks_the_reach_of_the_jacobian_it_materializes() -> None:
    """The point of the cap: the residual the probe differentiates fits inside a shorter reach.

    A colouring recovers couplings only out to the distance it was built at, and folds anything
    further onto the entries it does keep. Measured on a velocity column of the coupled Jacobian:
    capping the sweeps at two takes its stencil from six cells to four. The sweeps are not the only
    term feeding that reach -- this cavity's remaining terms carry two rings of their own -- which is
    why a cap is chosen by measuring the case rather than by subtracting one from a target.
    """
    mesh, coupled = _skewed_corrected_cavity(sweeps=4)
    state = _healthy_state(mesh, coupled)
    probed = CoupledJacobianProbe.build(coupled, stencil_reach=4, gradient_sweeps=2).narrow(coupled)
    distance = _cell_graph_distance(mesh)

    def reach(case):
        seed = jnp.zeros_like(state).at[0].set(1.0)  # perturb cell 0's u, read where it lands
        response = jnp.abs(jax.jvp(case.residual, (state,), (seed,))[1][: mesh.n_cells])
        live = np.asarray(response > 1e-13 * response.max())
        return int(distance[0][live].max())

    assert reach(coupled) == 6
    assert reach(probed) == 4


def test_the_cap_leaves_the_residual_itself_alone() -> None:
    """It is the preconditioner's stand-in, so the solved equations must not move."""
    mesh, coupled = _skewed_corrected_cavity(sweeps=4)
    state = _healthy_state(mesh, coupled)
    probed = CoupledJacobianProbe.build(coupled, stencil_reach=3, gradient_sweeps=2).narrow(coupled)
    assert not bool(jnp.array_equal(coupled.residual(state), probed.residual(state)))  # arms differ
    np.testing.assert_array_equal(
        np.asarray(coupled.residual(state)),
        np.asarray(
            _cavity(mesh=mesh, gradient=coupled.momentum.gradient_scheme)[1].residual(state)
        ),
    )
