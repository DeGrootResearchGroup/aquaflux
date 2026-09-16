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
from aquaflux.flow import MomentumContinuity, MovingWall, NoSlipWall, ViscousMultilevel
from aquaflux.flow.state import flow_state_layout
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss, CorrectedGreenGauss, SweptGradientSolve
from aquaflux.solve import (
    NO_REFRESH,
    CycleGrowthTrigger,
    DualTimeLoop,
    Globalization,
    PseudoTransientStep,
    RefreshPolicy,
    RowScaledNorm,
    ShiftTerm,
)
from aquaflux.turbulence import (
    BlockDiagonal,
    CompleteLu,
    DirectScalars,
    LinearSolveSettings,
    LogScalars,
    MaterializedJacobian,
    MonolithicVCycle,
    ShiftSettings,
    SSTModel,
    SSTTurbulence,
    coupled_equation_names,
    coupled_fields,
    coupled_residuals,
    coupled_step,
    eddy_viscosity_drift,
    hybrid_initialize,
    open_session,
    production_and_limit,
    production_cap_active,
)
from aquaflux.turbulence import coupled as coupled_module
from aquaflux.turbulence.coupled import (
    _BLOCK_LINEAR_SOLVE,
    _FACTORIZATION_LINEAR_SOLVE,
    CoupledJacobianProbe,
    CoupledRANS,
    LiveViscosityVelocityParts,
    _k_positivity_guards,
    _row_jacobian_scale,
    coupled_rans_layout,
    coupled_scaled_norm,
    frozen_production_viscosity,
    mass_flow_coupled_continuation,
    solve_coupled,
    solve_coupled_mass_flow,
    wall_consistent_state,
)

from tests.support.meshes import perturbed_grid_2d
from tests.unit.test_gradient import _cell_graph_distance

RHO, NU, U_LID = 1.0, 1e-2, 1.0
WALLS = ("top", "bottom", "left", "right")


def test_layout_round_trips_and_sizes() -> None:
    layout = coupled_rans_layout(flow_state_layout(dim=2, n_cells=5))
    flow_size = layout.sizes[0]
    assert flow_size == (2 + 1) * 5
    assert layout.size == (2 + 3) * 5
    assert layout.names == ("flow", "k", "omega")
    flow = jnp.arange(flow_size, dtype=float)
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
    properties = PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)})
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        properties,
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(U_LID, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        gradient_scheme=gradient,
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        gradient_scheme=gradient,
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
    """SSTTurbulence and the flow assembler take separate PropertyModels with no shared source.

    Nothing else checks that a caller supplied the same density to both -- if they disagree, the
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
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(U_LID, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        gradient_scheme=gradient,
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        FirstOrderUpwind(),
        # deliberately a different density from the flow assembler's RHO = 1.0
        PropertyModel({"viscosity": Constant(998.0 * NU), "density": Constant(998.0)}),
        gradient_scheme=gradient,
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
    assert _FACTORIZATION_LINEAR_SOLVE.restart == 10
    assert _BLOCK_LINEAR_SOLVE.restart == 120

    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    lu_step = coupled_step(
        coupled, state, preconditioner=MaterializedJacobian(CompleteLu(backend="scipy"))
    )
    block_step = coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None))
    # Each built step carries the solver it will run; the LU's is the small-restart one by default.
    assert lu_step.krylov_solver.restart == 10
    assert block_step.krylov_solver.restart == 120
    # An explicit krylov_solver still overrides the LU default.
    # ...and the restart alone can be moved without also replacing the stopping measure.
    assert (
        coupled_step(
            coupled,
            state,
            preconditioner=MaterializedJacobian(CompleteLu(backend="scipy")),
            linear_solve=LinearSolveSettings(restart=120),
        ).krylov_solver.restart
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
        "block": coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None)),
        "lu": coupled_step(
            coupled, state, preconditioner=MaterializedJacobian(CompleteLu(backend="scipy"))
        ),
        "block dual-time": coupled_step(
            coupled,
            state,
            preconditioner=BlockDiagonal(method=None),
            dual_time=DualTimeLoop(inner_steps=2),
        ),
    }
    for name, step in built.items():
        assert step.step_limit is not None, f"{name} has no k-positivity limit"

    # ...and the surfaces themselves agree, so a knob cannot reappear on one side. Everything here is a
    # property of the coupled march rather than of any preconditioner, which is the test: a parameter
    # belongs on all four or on none, and the ones that reached only the builder being worked on at the
    # time are exactly how this went wrong before -- twice with the tail already extracted, so nothing
    # in the bodies looked duplicated and no single commit looked wrong.
    shared = {
        # The pseudo-transient schedule, the divergence guard and the line search -- eight keywords
        # apiece until 2026-09-13, now one `Globalization` shared with the flow-only and scalar
        # builders as well (issue #372); `test_globalization_reach.py` is where that wider claim is
        # pinned, and this entry keeps the four coupled builders inside it.
        "globalization",
        "dual_time",
        # The shifted forward solve. `forward_rtol` / `forward_restart` / `forward_max_restarts` sat on
        # the multigrid builder alone, although the argument for them is about the *coupled residual*
        # (~100% omega under a plain 2-norm, so the flow block goes unresolved) and not about multigrid.
        "linear_solve",
        # The progress measure and the shift.
        "block_scaled_norm",
        # ...one value since #387, so the velocity parts -- once on two builders of four -- cannot fall
        # off one again.
        "shift",
        # The per-step guards.
        "inner_observer",
        "inner_refresh",
        "positivity_floor",
        "positivity_projection",
    }
    # One builder for every preconditioner family now, plus the bordered mass-flow sibling -- so the
    # surfaces that used to drift across four builders are one signature and one sibling.
    builders = (coupled_step, mass_flow_coupled_continuation)
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
        "block": coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None)),
        "lu": coupled_step(
            coupled, state, preconditioner=MaterializedJacobian(CompleteLu(backend="scipy"))
        ),
    }.items():
        assert step.krylov_solver.norm is step.residual_norm, (
            f"{name} steers on one measure and stops its linear solve on another"
        )
        # ...and that measure is the row-equilibrated one, not the Euclidean norm it used to be.
        assert isinstance(step.residual_norm, RowScaledNorm), name

    # An explicit measure is honoured all the way through, so the two cannot come apart there either --
    # which is what `solve_coupled` relies on when it re-injects the march's initial measure at every
    # refresh rather than letting a self-normalising one re-base at the developed state.
    base = coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None))
    explicit = coupled_step(
        coupled, state, preconditioner=BlockDiagonal(method=None), residual_norm=base.residual_norm
    )
    assert explicit.krylov_solver.norm is explicit.residual_norm is base.residual_norm


def test_the_constrained_builder_keeps_a_euclidean_stop_for_a_stated_reason() -> None:
    """The bordered mass-flow path is the one that genuinely differs, and it differs consistently.

    The row-equilibrated measure has no constraint-aware form: it would scale the border row by a
    diagonal the constraint does not have. So that march is judged in the Euclidean norm — and its
    forward solve therefore stops there too, at a Euclidean tolerance, because the tolerance and the
    norm it is measured in are one decision. This is a property of the path, not a surface that drifted.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    step = mass_flow_coupled_continuation(coupled, state, preconditioner=BlockDiagonal(method=None))
    assert step.residual_norm is jnp.linalg.norm
    assert step.krylov_solver.norm is step.residual_norm


def test_the_constrained_builder_refuses_a_materialized_preconditioner() -> None:
    """The bordered solve eliminates ``beta`` around a block-diagonal preconditioner, and only that.

    A materialized Jacobian has no constraint row, so its inverse would precondition a system other than
    the one solved. Refused before anything is built, so this needs no factorization.
    """
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    with pytest.raises(TypeError, match="must be a BlockDiagonal, not MaterializedJacobian"):
        mass_flow_coupled_continuation(
            coupled, state, preconditioner=MaterializedJacobian(CompleteLu(backend="scipy"))
        )


@pytest.mark.parametrize(
    "configuration",
    [
        {"preconditioner": BlockDiagonal(method="air")},
        {"reference_state": "state"},
        {"dual_time": DualTimeLoop(inner_steps=3)},
    ],
    ids=["preconditioner", "reference_state", "march setting"],
)
def test_the_constrained_solve_refuses_configuration_beside_a_finished_continuation(
    configuration: dict,
) -> None:
    """Configuration for a step the solve is not building is refused, not dropped.

    A finished ``strategy`` already carries its preconditioner, reference and march settings, so
    passing any of them beside it used to reach nothing: ``method="air"`` beside a twolevel step ran
    twolevel, with no error. The refusal comes before the initial condition is built.
    """
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    continuation = mass_flow_coupled_continuation(
        coupled, state, preconditioner=BlockDiagonal(method=None)
    )
    given = {name: state if value == "state" else value for name, value in configuration.items()}
    with pytest.raises(TypeError, match=r"configure the continuation `solve_coupled_mass_flow`"):
        solve_coupled_mass_flow(coupled, 1.0, strategy=continuation, **given)


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
    step = coupled_step(
        coupled,
        state,
        preconditioner=MaterializedJacobian(CompleteLu(backend="scipy")),
        shift=ShiftSettings(velocity_parts=live),
    )
    assert step.shift_policy.base.velocity_shift_parts is live
    # ...and it is genuinely live: away from the state the assembler was frozen at, the shift it
    # produces differs from the frozen one, which is the whole reason the source is injected. At the
    # freeze state the two coincide by construction, so a check there would pass on a dead wire.
    frozen = coupled_step(
        coupled, state, preconditioner=MaterializedJacobian(CompleteLu(backend="scipy"))
    )
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


def test_continuation_settings_are_refused_where_they_would_be_dropped() -> None:
    """A setting the solve cannot forward is an error, not a silent no-op.

    ``preconditioner`` / ``reference_state`` / ``**strategy_kwargs`` configure the continuation
    ``solve_coupled`` builds. On the two paths where it builds none -- an explicit ``strategy``, or a
    ``RefreshPolicy(builder=...)`` -- they reached nothing at all: a solve asked for a dual-time loop
    and ``positivity_floor=1e-6`` ran the library defaults, with no error and no log line. ``**kwargs``
    is what made it quiet, since it accepts every keyword and checks none, and that door is the main
    entry point's.

    The message must name the keywords, because the whole failure mode is not knowing they were lost.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    flow, k, omega = coupled.physical_fields(state)
    step = coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None))

    for kwargs in (
        {"strategy": step, "dual_time": DualTimeLoop(inner_steps=3)},
        {"strategy": step, "positivity_floor": 1e-6},
        {"strategy": step, "preconditioner": BlockDiagonal()},
        {"strategy": step, "reference_state": state},
        {
            "refresh": RefreshPolicy(builder=lambda s: step),
            "dual_time": DualTimeLoop(inner_steps=3),
        },
        {
            "refresh": RefreshPolicy(builder=lambda s: step),
            "preconditioner": BlockDiagonal(method=None),
        },
    ):
        offender = next(iter(set(kwargs) - {"strategy", "refresh"}))
        with pytest.raises(TypeError, match=offender):
            solve_coupled(coupled, flow, k, omega, max_steps=1, **kwargs)


def test_the_settings_are_still_accepted_where_the_solve_does_build_the_continuation() -> None:
    """The guard must not fire on the path the settings are for -- including under a builder-less refresh.

    A ``RefreshPolicy`` with a trigger but no builder still leaves ``solve_coupled`` building the
    continuation, so it keeps receiving the configuration. Getting that wrong would break every staged
    march in the suite, which is why it is pinned beside the rejection rather than assumed.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    for refresh in (NO_REFRESH, RefreshPolicy(trigger=CycleGrowthTrigger())):
        source = coupled_module._continuation_source(
            coupled=coupled,
            strategy=None,
            refresh=refresh,
            preconditioner=BlockDiagonal(method=None),
            reference_state=state,
            kwargs={"dual_time": DualTimeLoop(inner_steps=2)},
        )
        assert isinstance(source, coupled_module._SessionContinuation)
        # ...and it carries them, rather than accepting and then dropping them one layer down.
        assert source.march == {"dual_time": DualTimeLoop(inner_steps=2)}
        assert source.reference_state is state
        assert source.session._spec == BlockDiagonal(method=None)


def test_an_unnamed_preconditioner_is_the_default_block_diagonal_family() -> None:
    """With nothing named the solve builds the block-diagonal family at its defaults.

    The resolved scalar method must not move: ``None`` selects no scalar preconditioner at all, so the
    unset default and an explicit ``None`` are different choices and are kept apart on the spec.
    """
    assert inspect.signature(solve_coupled).parameters["preconditioner"].default is None
    source = coupled_module._continuation_source(
        coupled=None,
        strategy=None,
        refresh=NO_REFRESH,
        preconditioner=None,
        reference_state=None,
        kwargs={},
    )
    assert source.session._spec == BlockDiagonal()
    assert source.session._spec.resolved_method() == "twolevel"
    assert source.refresh_preconditioner is None


def test_a_session_owned_setting_and_a_second_refresh_hook_are_refused() -> None:
    """Each would otherwise be a silent disagreement between two owners of one decision."""
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    flow, k, omega = coupled.physical_fields(state)
    session = open_session(MaterializedJacobian(CompleteLu(backend="scipy")), coupled)
    with pytest.raises(TypeError, match="belongs to the preconditioner session"):
        solve_coupled(
            coupled, flow, k, omega, preconditioner=session, jacobian_production_viscosity=True
        )
    with pytest.raises(TypeError, match="already re-fits its inverse"):
        solve_coupled(
            coupled,
            flow,
            k,
            omega,
            preconditioner=MaterializedJacobian(CompleteLu()),
            refresh=RefreshPolicy(refresh_preconditioner=lambda step, s: None),
        )
    with pytest.raises(TypeError, match="belongs on the spec"):
        solve_coupled(coupled, flow, k, omega, max_steps=1, velocity="convection")


def test_the_continuation_source_is_one_decision_for_the_build_and_every_refresh() -> None:
    """The initial build and the refresh rebuild come from one object, not two parallel branches.

    They are the same decision -- which continuation this solve runs -- and were written as two
    independent two-way branches, one at the build and one inside the refresh loop. That is the shape
    that drifts: a change has to be made twice and nothing fails when it is made once.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    built = []

    def builder(s):
        built.append(s)
        return coupled_step(coupled, s, preconditioner=BlockDiagonal(method=None))

    source = coupled_module._continuation_source(
        coupled=coupled,
        strategy=None,
        refresh=RefreshPolicy(trigger=CycleGrowthTrigger(), builder=builder),
        preconditioner=None,
        reference_state=None,
        kwargs={},
    )
    first = source.build(state)
    assert len(built) == 1
    # A refresh re-invokes the SAME builder, and re-injects the march's measure rather than rebuilding
    # it -- a self-normalising measure rebuilt at a developed state would re-base the convergence test.
    measure = coupled_scaled_norm(coupled, first.shift_policy, state)
    refreshed = source.refresh(state, first, measure)
    assert len(built) == 2
    # Value, not identity: the builder path re-injects the measure with `tree_at`, which rebuilds
    # the pytree around the substituted leaf. What must hold is that the scales did not move.
    assert bool(eqx.tree_equal(refreshed.residual_norm, measure))


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
    assert coupled.momentum.mesh.dim == mesh.dim
    assert coupled.layout.n_cells == mesh.n_cells
    assert coupled.pack_state(
        coupled.momentum.initial_state(),
        jnp.ones(mesh.n_cells),
        jnp.ones(mesh.n_cells),
    ).shape == (coupled.layout.size,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("transform", ["jit", "vmap"])
def test_the_coupled_solve_refuses_jit_and_vmap_before_any_work(transform) -> None:
    """The march steps in Python on concrete values, which ``jit`` and ``vmap`` never provide.

    Without the up-front check the solve would get part way -- building a preconditioner, say -- and
    fail on a tracer conversion that names neither the transform nor what to do instead. ``jax.grad``
    is not refused: a stopped copy of its inputs is concrete, and the integration tests differentiate
    through the solve.
    """
    mesh, coupled = _cavity()
    flow, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))

    with pytest.raises(ValueError, match=r"cannot run inside a traced program"):
        if transform == "jit":
            eqx.filter_jit(lambda c: solve_coupled(c, flow, k, omega, rtol=1e-2))(coupled)
        else:
            jax.vmap(lambda f: solve_coupled(coupled, f, k, omega, rtol=1e-2))(
                jnp.stack([flow, flow])
            )


def test_the_march_is_handed_the_homotopy_and_the_same_arguments_whether_or_not_it_is_observed(
    monkeypatch,
) -> None:
    """Observing a solve must not change what is solved, and a homotopy must always reach the march.

    Before the solve had one march, ``on_step`` decided which of two it ran, and a ``homotopy`` was
    handed only to the observed one -- so a ramp with nothing else forcing observation silently skipped
    its ramp. Recording what the march receives pins both: the arguments with and without an observer
    differ in the observer alone, and the homotopy arrives with no observer at all.
    """
    from aquaflux.solve import MarchResult

    mesh, coupled = _cavity()
    flow, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))
    calls: list[dict] = []

    def recording_march(step, residual_fn, state, **kwargs):
        calls.append(kwargs)
        return MarchResult(state, (), True, False, None)

    monkeypatch.setattr(coupled_module, "newton_march", recording_march)
    homotopy = object()
    # A target no state can miss, so the recorded march's untouched state is accepted as the root; a
    # pre-built step with a plain Euclidean measure keeps the test to the wiring.
    loose = dict(rtol=1.0, atol=1e30, strategy=_single_step())

    solve_coupled(coupled, flow, k, omega, homotopy=homotopy, **loose)
    solve_coupled(coupled, flow, k, omega, homotopy=homotopy, on_step=print, **loose)

    unobserved, observed = calls
    assert unobserved["homotopy"] is homotopy
    assert unobserved["observer"] is None and observed["observer"] is print
    differing = {
        key
        for key in unobserved
        if key not in ("observer", "drift_measure", "norm_builder")
        and not _same_argument(unobserved[key], observed[key])
    }
    assert differing == set()


@pytest.mark.parametrize(
    ("converged", "homotopy", "atol", "match"),
    [
        (True, None, 0.0, "did not converge: the march ended at residual"),
        (False, object(), 1e30, "homotopy never reached the target problem"),
    ],
    ids=["above-target", "homotopy-not-arrived"],
)
def test_a_march_that_ends_short_of_a_root_is_refused_rather_than_returned(
    monkeypatch, converged, homotopy, atol, match
) -> None:
    """The adjoint is only valid at a root, so a state that is not one must raise, not be returned.

    Two ways to end short: a residual above the target (here the untouched starting state against a
    zero tolerance), and a homotopy that never reached its target, where a small residual belongs to an
    intermediate problem and says nothing about this one.
    """
    from aquaflux.solve import MarchResult

    mesh, coupled = _cavity()
    flow, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))
    monkeypatch.setattr(
        coupled_module,
        "newton_march",
        lambda step, residual_fn, state, **kwargs: MarchResult(state, (), converged, False, None),
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match=match):
        solve_coupled(
            coupled,
            flow,
            k,
            omega,
            rtol=0.0,
            atol=atol,
            homotopy=homotopy,
            strategy=_single_step(),
        )


def test_the_last_refresh_segment_marches_without_the_trigger(monkeypatch) -> None:
    """With no refresh left to spend, the last segment must not stop where the trigger fires.

    There is no second solve after the march, so a last segment stopped by its trigger would end the
    whole solve short of the root. Every earlier segment still gets the trigger, which is how a refresh
    happens at all. The recorded march fires its trigger whenever it is given one, and the refresh is
    stubbed to hand the same step back.
    """
    from aquaflux.solve import MarchResult

    mesh, coupled = _cavity()
    flow, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))
    triggers: list[object] = []

    def recording_march(step, residual_fn, state, **kwargs):
        triggers.append(kwargs["trigger"])
        return MarchResult(state, (), True, kwargs["trigger"] is not None, None)

    monkeypatch.setattr(coupled_module, "newton_march", recording_march)
    trigger = object()
    step = _single_step()
    solve_coupled(
        coupled,
        flow,
        k,
        omega,
        rtol=1.0,
        atol=1e30,
        strategy=step,
        refresh=RefreshPolicy(trigger=trigger, limit=2, builder=lambda state: step),
    )

    assert triggers == [trigger, trigger, None]


def _same_argument(one, two) -> bool:
    """Equality for a recorded march argument, treating equal arrays as equal."""
    if isinstance(one, jnp.ndarray) or isinstance(two, jnp.ndarray):
        return bool(jnp.array_equal(one, two))
    return one is two or one == two


class _TrivialShiftPolicy(eqx.Module):
    """A shift policy with a unit diagonal and no preconditioner -- enough to build a step object."""

    def shift_term(self, phi, residual=None):
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
            strategy=PseudoTransientStep(_TrivialShiftPolicy()),
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

    kw = dict(velocity=ViscousMultilevel())
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
    jump back toward one and the convergence test become unreachable. A session's refresh, handed the
    march's measure (what ``solve_coupled`` passes on every refresh), uses it verbatim instead of
    rebuilding, so the measure stays fixed at the state the global progress reference was measured at.
    """
    mesh, coupled = _cavity()
    cold = _healthy_state(mesh, coupled, seed=0)
    developed = _healthy_state(mesh, coupled, seed=1)
    spec = BlockDiagonal(method=None, velocity=ViscousMultilevel())
    session = open_session(spec, coupled)

    base = session.build(cold, block_scaled_norm=True)
    base_norm = base.norm()
    refreshed = session.refresh(developed, base, base_norm, block_scaled_norm=True)
    # The refreshed continuation measures progress with the *same* norm object, not a re-based one.
    assert refreshed.norm() is base_norm
    # And that carry matters: a from-scratch rebuild at the developed state re-bases the per-block
    # scales, so it scores the same residual differently (it self-normalises to sqrt(n_blocks) there).
    rebuilt = coupled_step(coupled, developed, preconditioner=spec, block_scaled_norm=True)
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

    def shift_term(self, phi, residual=None):
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


def test_a_dual_time_march_given_no_control_defaults_to_the_courant_step_control() -> None:
    """A dual-time march given no control defaults to ``DualTimeControl``, observed or not."""
    from aquaflux.solve import DualTimeControl, default_dual_time_control

    control = default_dual_time_control(None, strategy=_dual_time_step())
    assert isinstance(control, DualTimeControl)


def test_a_single_step_march_gets_no_default_control() -> None:
    """A single-step (pseudo-transient) march is not a dual-time step, so no control is injected."""
    from aquaflux.solve import default_dual_time_control

    assert default_dual_time_control(None, strategy=_single_step()) is None


def test_a_caller_supplied_control_is_never_overridden() -> None:
    """An explicit control on a dual-time march is returned unchanged (the override path)."""
    from aquaflux.solve import ResidualRatioDualTimeControl, default_dual_time_control

    explicit = ResidualRatioDualTimeControl(beta_start=0.5)
    assert default_dual_time_control(explicit, strategy=_dual_time_step()) is explicit


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
    engine = coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None))

    reported = coupled_residuals(coupled, engine)(state)

    assert list(reported) == list(coupled_equation_names(mesh.dim))
    measure = coupled_scaled_norm(coupled, engine.shift_policy, state)
    assert float(jnp.linalg.norm(jnp.array(list(reported.values())))) == pytest.approx(
        float(measure(coupled.residual(state))), rel=1e-10
    )


def test_the_per_equation_rows_add_up_to_the_residual_the_march_reports() -> None:
    """`newton_march` equilibrates at the state each outer iteration STARTS from and holds that
    measure for the whole iteration -- so the step it reports is ``norm_at_start(R(state_at_end))``.
    Scaling at the end state instead would measure the right residual in the wrong scales, and the
    rows would not add up to the number printed above them.
    """
    mesh, coupled = _cavity()
    start = _healthy_state(mesh, coupled, seed=0)
    end = _healthy_state(mesh, coupled, seed=1)
    engine = coupled_step(coupled, start, preconditioner=BlockDiagonal(method=None))
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
    engine = coupled_step(coupled, first, preconditioner=BlockDiagonal(method=None))

    residuals = coupled_residuals(coupled, engine, first)
    residuals(second)  # step 1 consumes the seed and records `second`
    rows = residuals(third)  # step 2 must equilibrate at `second`, not at `third`

    expected = float(
        coupled_scaled_norm(coupled, engine.shift_policy, second)(coupled.residual(third))
    )
    assert float(jnp.linalg.norm(jnp.array(list(rows.values())))) == pytest.approx(
        expected, rel=1e-12
    )


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

    ``solve_coupled`` re-bases this measure at every refresh segment, so a per-reference compilation is paid on every one. Measured on a three-dimensional coupled march before the
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

    The factory rides in the strategy's ``adjoint_preconditioner_factory``, a static field and so
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
    changed -- here both ask the layout for the block by name, and this pins the answer. And these
    builders are the only place the projection is constructed for a coupled case, so a missing import
    in the module would otherwise surface for the first time in the middle of a march rather than here.

    The builders read only the transform and the block layout, and a layout is mesh-free, so a stub
    carrying those two is a sufficient collaborator -- no mesh, no assembled case.
    """
    from types import SimpleNamespace

    from aquaflux.turbulence import positive_k_limit, positive_k_projection
    from aquaflux.turbulence.coupled import DirectScalars, LogScalars

    n, dim = 7, 3
    direct = SimpleNamespace(
        k_transform=DirectScalars(),
        layout=coupled_rans_layout(flow_state_layout(dim=dim, n_cells=n)),
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
    logged = SimpleNamespace(k_transform=LogScalars(), layout=direct.layout)
    assert positive_k_limit(logged) is None
    assert positive_k_projection(logged) is None


def test_k_positivity_guards_refuses_a_floor_the_default_projection_would_silence() -> None:
    """``positivity_floor`` fed only to the limiter is provably inert once the projection runs first.

    The projection clips every k correction to within ``tau`` of its own boundary; the limiter then
    sees room ``>= 1/tau`` on every entry whatever floor it carries, so it always reports
    ``alpha_max = 1``. A non-zero floor under the default ``positivity_projection=True`` therefore
    protects nothing -- refused here rather than left as a step-length surprise (#365).
    """
    from types import SimpleNamespace

    from aquaflux.turbulence.coupled import DirectScalars, LogScalars

    direct = SimpleNamespace(
        k_transform=DirectScalars(),
        layout=coupled_rans_layout(flow_state_layout(dim=2, n_cells=5)),
    )

    with pytest.raises(ValueError, match=r"positivity_floor.*has no effect"):
        _k_positivity_guards(direct, 1e-6, True)
    with pytest.raises(ValueError, match=r"positivity_floor.*has no effect"):
        _k_positivity_guards(direct, 1e-6, True)  # not consumed by the first call: pure function

    # The floor is genuinely usable paired with the plain global cap.
    step_limit, step_projection = _k_positivity_guards(direct, 1e-6, False)
    assert step_limit.floor == 1e-6
    assert step_projection is None

    # A zero floor is never refused, under either setting -- it is what the projection already does.
    for projection in (True, False):
        step_limit, step_projection = _k_positivity_guards(direct, 0.0, projection)
        assert step_limit.floor == 0.0
        assert (step_projection is not None) is projection

    # Inert for a different, already-documented reason (the transform, not the projection) when k is
    # solved in log form -- no limiter is built at all, so there is nothing for the floor to feed.
    logged = SimpleNamespace(k_transform=LogScalars(), layout=direct.layout)
    assert _k_positivity_guards(logged, 1e-6, True) == (None, None)


def _block_step(coupled, state, **march):
    return coupled_step(coupled, state, **march)


def _lu_step(coupled, state, **march):
    return coupled_step(
        coupled, state, preconditioner=MaterializedJacobian(CompleteLu(backend="scipy")), **march
    )


def _vcycle_step(coupled, state, **march):
    return coupled_step(
        coupled, state, preconditioner=MaterializedJacobian(MonolithicVCycle()), **march
    )


@pytest.mark.parametrize(
    "builder",
    [_block_step, _lu_step, _vcycle_step, mass_flow_coupled_continuation],
    ids=["block", "lu", "vcycle", "mass flow"],
)
def test_every_coupled_continuation_builder_refuses_the_same_inert_combination(builder) -> None:
    """Every preconditioner family routes through :func:`_k_positivity_guards`, and it is checked first.

    First specifically so a real caller's mistake -- and this test -- never pays for building the
    preconditioner the raise makes moot. That matters beyond cost for the multigrid V-cycle: it needs
    ``petsc4py`` (not installed by CI, see ``tests/unit/test_optional_dependency_skips.py``), so checking
    the raise here would break under that dependency's absence if the guard ran any later than it does.
    """
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    with pytest.raises(ValueError, match=r"positivity_floor.*has no effect"):
        builder(coupled, state, positivity_floor=1e-6)
    with pytest.raises(ValueError, match=r"positivity_floor.*has no effect"):
        builder(coupled, state, positivity_floor=1e-6, positivity_projection=True)


@pytest.mark.parametrize(
    "builder",
    [_block_step, _lu_step, mass_flow_coupled_continuation],
    ids=["block", "lu", "mass flow"],
)
def test_the_inert_combination_is_the_only_thing_refused(builder) -> None:
    """Every other pairing of the two settings still builds -- including a floor that now matters.

    The multigrid V-cycle is excluded here (unlike the raise check above): building its real
    preconditioner needs ``petsc4py``, which this file does not gate on, so only the checks that
    never reach it may run unconditionally.
    """
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    builder(coupled, state, positivity_floor=1e-6, positivity_projection=False)
    builder(coupled, state, positivity_floor=0.0)
    builder(coupled, state, positivity_floor=0.0, positivity_projection=False)


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


class _ScalarRans(eqx.Module):
    """A one-line stand-in assembler whose Jacobian is a scalar, so a rebind is visible in one apply."""

    gain: jnp.ndarray

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.gain * state


class _RecordingPreconditioner:
    """A frozen inverse that records what each refresh was asked to build, and builds nothing.

    It builds nothing, so what is under test is only when the hook asks for a rebuild, and of what.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def refresh_in_place(self, matvec, plan, shift_diagonal, **_kwargs):
        self.calls.append({"matvec": matvec, "shift": shift_diagonal})
        return ()


def _stub_step(preconditioner, beta, diagonal):
    """The smallest Newton step the refresh hook reads: a shift strength, a policy and its diagonal."""
    from types import SimpleNamespace

    from aquaflux.solve import ShiftTerm

    # Accepts the optional residual even though the refresh hook does not pass one: a stand-in that
    # is narrower than the protocol breaks silently the day a caller starts supplying it.
    base = SimpleNamespace(
        shift_term=lambda _phi, _residual=None: ShiftTerm(diagonal, lambda _relaxation: None)
    )
    return SimpleNamespace(
        relaxation_schedule=SimpleNamespace(beta=beta),
        shift_policy=SimpleNamespace(preconditioner=preconditioner, base=base),
    )


def test_rebinding_the_refresh_swaps_the_case_and_forces_a_full_rebuild() -> None:
    """One refresh hook can serve a whole Reynolds ramp, which is what lets the ramp share one V-cycle.

    Nothing else in the hook watches for a rung boundary, so a hook that had stopped rebuilding would leave the next rung
    solving against a V-cycle fitted to the previous rung's viscosity. ``rebind`` therefore does two
    things, and both are asserted: the Jacobian probe starts reporting the NEW companion's derivative,
    and the next refresh is a full re-materialize.
    """

    import numpy as np
    from aquaflux.turbulence.coupled import _beta_tracking_refresh

    state = jnp.linspace(1.0, 2.0, 5)
    diagonal = jnp.full(5, 2.0)
    tangent = jnp.ones(5)
    # The real probe (its plan and gather map are unused here), not a lookalike: `_beta_tracking_refresh`
    # asks it which assembler to differentiate, which only the class itself can answer.
    probe = CoupledJacobianProbe(plan=object(), structure=object())

    pc = _RecordingPreconditioner()
    step = _stub_step(pc, beta=0.5, diagonal=diagonal)
    # The multigrid cadence: a full rebuild on the first call and after a rebind, none otherwise --
    # between those only the dual-time loop's cost trigger rebuilds the V-cycle.
    refresh = _beta_tracking_refresh(
        _ScalarRans(gain=jnp.asarray(3.0)), stencil_reach=2, probe=probe, every_step=False
    )

    refresh(step, state)  # the initializing call
    assert len(pc.calls) == 1
    assert np.allclose(pc.calls[0]["shift"], 0.5 * np.asarray(diagonal))
    assert np.allclose(pc.calls[0]["matvec"](tangent), 3.0 * tangent)

    refresh(step, state)  # no rebuild between rebinds, as for the rest of a rung
    assert len(pc.calls) == 1

    refresh.rebind(_ScalarRans(gain=jnp.asarray(7.0)))
    refresh(step, state)
    assert len(pc.calls) == 2  # forced by the rebind
    assert np.allclose(pc.calls[1]["matvec"](tangent), 7.0 * tangent)  # ...at the new companion

    refresh(step, state)  # and the force is spent: one rebuild per rebind, not a stuck flag
    assert len(pc.calls) == 2


def test_the_factorization_cadence_rebuilds_on_every_step() -> None:
    """``every_step=True`` is the complete-LU cadence: an exact factorization is cheap, and exact only
    at the shift it was built at, so it is re-factored before every step rather than on a rebind."""
    from aquaflux.turbulence.coupled import _beta_tracking_refresh

    state = jnp.linspace(1.0, 2.0, 5)
    diagonal = jnp.full(5, 2.0)
    pc = _RecordingPreconditioner()
    refresh = _beta_tracking_refresh(
        _ScalarRans(gain=jnp.asarray(3.0)),
        stencil_reach=2,
        probe=CoupledJacobianProbe(plan=object(), structure=object()),
        every_step=True,
    )

    for beta in (0.5, 0.25, 0.125):
        refresh(_stub_step(pc, beta=beta, diagonal=diagonal), state)
    assert [float(call["shift"][0]) for call in pc.calls] == [1.0, 0.5, 0.25]


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
    assert NO_REFRESH.refresh_preconditioner is None
    # The default must not refresh.
    assert not NO_REFRESH.refreshes


def test_a_refresh_needs_both_a_trigger_and_a_budget() -> None:
    """``refreshes`` is the conjunction: a trigger with no budget refreshes nothing, and vice versa.

    ``limit=0`` is the documented way to disable refreshing while leaving a trigger in place, so
    reading this as "a trigger is set" would quietly ignore that.
    """
    assert RefreshPolicy(trigger=object()).refreshes
    assert not RefreshPolicy(trigger=object(), limit=0).refreshes
    assert not RefreshPolicy(limit=5).refreshes


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
    """The globalization is not named on ``solve_coupled`` and still arrives at the shared step tail
    unchanged -- it rides ``**strategy_kwargs`` through the preconditioner session.

    ``grow`` used to be declared on ``solve_coupled`` *and* forwarded explicitly, while the very same
    call sites already splatted ``**strategy_kwargs`` into the builder -- so the declaration was
    pure duplication, costing a parameter on an already-wide signature to buy nothing. Deleting it was
    call-for-call identical, and this pins that: it is the only thing standing between the deletion
    and a silently dropped knob. The knob itself now lives on the ``Globalization``, so what rides the
    path is one object rather than eight keywords, and the pin is the same. It is spied on
    ``_coupled_step`` because every family's session reaches the march through that one tail.
    """
    from aquaflux.turbulence import coupled as coupled_module

    _, coupled = _cavity(4)
    seen: dict = {}

    def spy(assembler, reference_state, policy, **kwargs):
        seen.update(kwargs)
        raise _StopBuild

    monkeypatch.setattr(coupled_module, "_coupled_step", spy)
    asked = Globalization(grow=2, beta0=1.5)
    with pytest.raises(_StopBuild):
        solve_coupled(coupled, globalization=asked)

    assert seen["globalization"] is asked


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


def test_freezing_the_production_viscosity_changes_the_operator_and_not_the_residual() -> None:
    """The two properties that make ``explicit_production_viscosity`` a legal Jacobian stand-in.

    It exists because ``nu_t`` is proportional to ``k``, so the production is too, and differentiating
    that puts a negative term on the k row's own Jacobian diagonal which the pseudo-time shift then
    has to cancel. Freezing it is only sound if it changes the **derivative** and nothing else, so
    both halves are pinned here:

    * the residual is **bit-identical**, which is what leaves the converged root and the
      implicit-function-theorem adjoint untouched when the copy is used as an operator; and
    * the Jacobian genuinely moves, which is the guard against a freeze that silently does nothing --
      a flag that had stopped taking effect would pass a residual check alone.

    Off by default, and byte-identical off.
    """
    _, exact = _cavity(4)
    assert exact.turbulence.explicit_production_viscosity is False
    assert (
        inspect.signature(SSTTurbulence.build).parameters["explicit_production_viscosity"].default
        is False
    )

    frozen = frozen_production_viscosity(exact)
    assert frozen.turbulence.explicit_production_viscosity is True
    assert exact.turbulence.explicit_production_viscosity is False  # the original is not mutated

    flow, k, omega = hybrid_initialize(exact.momentum, exact.turbulence)
    state = exact.state_from_physical(flow, k, omega)
    tangent = jax.random.normal(jax.random.PRNGKey(3), state.shape, dtype=state.dtype)

    assert jnp.array_equal(frozen.residual(state), exact.residual(state))
    exact_action = jax.jvp(exact.residual, (state,), (tangent,))[1]
    frozen_action = jax.jvp(frozen.residual, (state,), (tangent,))[1]
    assert not jnp.allclose(frozen_action, exact_action)

    # And the difference is confined to the k block: the momentum closure, the omega equation and the
    # k diffusivity all keep the live eddy viscosity, so only the production's own derivative moved.
    layout = exact.layout
    flow_block, omega_block = layout.slice_of("flow"), layout.slice_of("omega")
    assert jnp.array_equal(frozen_action[flow_block], exact_action[flow_block])
    assert jnp.array_equal(frozen_action[omega_block], exact_action[omega_block])


def test_every_continuation_builder_defaults_to_the_per_entry_positivity_projection() -> None:
    """The default is a value, pinned, because the global cap it replaces loses a march.

    The plain fraction-to-the-boundary cap scales the whole step by the worst entry of the
    smallest-magnitude block, so one numerically-dead cell sets the step length for every degree of
    freedom -- and then ratchets, the capped entry decaying by ``1 - tau`` per step whatever the floor
    is. Measured on a separating coupled-RANS benchmark, that loses the march outright while the
    per-entry projection completes it *and* speeds up the arm that already worked.

    Pinned on every builder together: a default that reverts on one of them reverts silently, and
    the failure it re-arms shows up as a step length rather than as an error.
    """
    for builder in (coupled_step, mass_flow_coupled_continuation):
        got = inspect.signature(builder).parameters["positivity_projection"].default
        assert got is True, f"{builder.__name__} defaults positivity_projection to {got!r}"


def test_the_probe_materializes_the_operator_the_solve_applies() -> None:
    """A preconditioner must be assembled from the matrix the Krylov iteration applies, not another.

    ``CoupledJacobianProbe.narrow`` is the single place that decides which assembler gets
    materialized, and every consumer routes through it -- the initial build, the refresh hook, and the
    rebind across a Reynolds-continuation rung. So the stand-in belongs on the probe, and this pins
    that it lands there rather than on the colouring plan.

    That distinction is the whole point: the plan is a graph pass over the cell adjacency and is
    **identical** whichever assembler it is derived from, so deriving it from the stand-in changes
    nothing at all. Getting that wrong leaves a preconditioner built for a matrix nobody is solving,
    which is invisible except as a cycle count.
    """
    _, coupled = _cavity(4)
    plain = CoupledJacobianProbe.build(coupled, 3)
    frozen = CoupledJacobianProbe.build(coupled, 3, production_viscosity_frozen=True)

    # The colouring and its de-compression are untouched -- this axis is about values, not structure.
    assert np.array_equal(np.asarray(plain.structure.indices), np.asarray(frozen.structure.indices))

    assert plain.narrow(coupled).turbulence.explicit_production_viscosity is False
    assert frozen.narrow(coupled).turbulence.explicit_production_viscosity is True
    # And it survives being pointed at another companion of the same case, which is what a Reynolds
    # rung boundary does -- the failure would otherwise appear only after the first rung.
    companion = coupled.with_scaled_molecular_viscosity(10.0)
    assert frozen.narrow(companion).turbulence.explicit_production_viscosity is True
    assert plain.narrow(companion).turbulence.explicit_production_viscosity is False


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


def test_effective_momentum_is_the_assembler_the_residual_itself_uses() -> None:
    """The shared re-viscosification seam reproduces the residual's own flow block, bit for bit.

    ``effective_momentum`` exists so the coupled residual and the callers that want only the momentum
    block at the current ``mu_eff`` read one implementation. That is worth nothing unless what it
    returns really is what the residual assembles with, so compare the flow block it produces against
    the flow block of the coupled residual rather than re-deriving the closure here and asserting the
    two derivations agree.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    flow, k, omega = coupled.physical_fields(state)
    closure, momentum = coupled.effective_momentum(flow, k, omega)

    # Both halves are checked against an INDEPENDENT derivation, never against the coupled residual --
    # `CoupledRANS.residual` now calls this method, so comparing the two would compare it with itself
    # and would pass for any nu_t whatsoever. That tautology is what an earlier version of this test
    # asserted, and it would have been blind to the regressions the consolidation could actually cause.
    independent = coupled.momentum.with_eddy_viscosity(
        coupled.turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega).nu_t,
        coupled.turbulence.wall_face_eddy_viscosity(k),
    )
    assert bool(jnp.array_equal(momentum.residual(flow), independent.residual(flow)))
    assert bool(jnp.array_equal(closure.nu_t, coupled.eddy_viscosity(state)))

    # ⚠️ And the WALL-FACE half specifically, which the eddy-viscosity comparison above cannot see: it
    # rides a separate leaf and reaches only the boundary diffusion coefficient. Dropping it is a real
    # and recorded failure mode on a wall-function mesh (the wall model's viscosity and the wall cell's
    # log-layer `k/omega` value differ by a large factor there), so pin that it is carried.
    without_wall = coupled.momentum.with_eddy_viscosity(closure.nu_t)
    assert not bool(jnp.array_equal(momentum.residual(flow), without_wall.residual(flow)))


def test_wall_omega_repair_drives_the_fixation_rows_to_zero_after_a_viscosity_change() -> None:
    """The repaired value is the one the RESIDUAL fixes, checked against the residual and not against
    a second copy of the formula.

    This is the property that matters: a near-wall ``omega`` row is a value fixation, so a state holding
    the right value has *zero* residual there. Asserting that the function reproduces ``omega_wall`` would
    only check one transcription against another; driving the residual to zero checks it against the
    equation being solved.
    """
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    cells = coupled.turbulence.wall_cells
    n = mesh.n_cells
    fixed = slice((coupled.momentum.mesh.dim + 2) * n, (coupled.momentum.mesh.dim + 3) * n)

    # Converge the fixation rows at THIS viscosity first, so the only thing the rung change introduces
    # is the viscosity itself -- otherwise the "before" is polluted by the state never having satisfied
    # them at all.
    at_home = wall_consistent_state(coupled, state)
    assert float(jnp.max(jnp.abs(coupled.residual(at_home)[fixed][cells]))) < 1e-12

    # Hand that state to a companion at one tenth the viscosity, exactly as a Reynolds rung does.
    companion = coupled.with_scaled_molecular_viscosity(0.1)
    carried = float(jnp.max(jnp.abs(companion.residual(at_home)[fixed][cells])))
    repaired = wall_consistent_state(companion, at_home)
    after = float(jnp.max(jnp.abs(companion.residual(repaired)[fixed][cells])))

    assert carried > 1.0  # the carried value really is wrong at the new viscosity
    assert after < 1e-12  # and the repair lands exactly on the row


def test_wall_omega_repair_touches_only_the_rows_the_model_prescribes() -> None:
    """Flow, ``k`` and interior ``omega`` are a converged field and must come through untouched -- a
    repair that moved them would be changing the answer, not restoring a boundary condition."""
    mesh, coupled = _cavity()
    state = _healthy_state(mesh, coupled)
    companion = coupled.with_scaled_molecular_viscosity(0.1)
    repaired = wall_consistent_state(companion, state)

    flow, k, omega = coupled.physical_fields(state)
    flow_after, k_after, omega_after = coupled.physical_fields(repaired)
    assert bool(jnp.array_equal(flow_after, flow))
    assert bool(jnp.array_equal(k_after, k))

    moved = jnp.where(omega_after != omega)[0]
    assert bool(jnp.array_equal(jnp.sort(moved), jnp.sort(coupled.turbulence.wall_cells)))


def test_wall_omega_repair_is_a_no_op_at_an_unchanged_viscosity() -> None:
    """It corrects a *parameter* change, so with no parameter change there is nothing to correct. This
    is what makes it safe to apply unconditionally at every continuation point."""
    mesh, coupled = _cavity()
    repaired_once = wall_consistent_state(coupled, _healthy_state(mesh, coupled))
    assert bool(jnp.array_equal(wall_consistent_state(coupled, repaired_once), repaired_once))
