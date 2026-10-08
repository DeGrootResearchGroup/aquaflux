"""The flow-only march: a laminar ``(u, p)`` solve on the staged machinery every coupled solve uses.

A laminar problem used to have only ``momentum_continuation`` -- a bare pseudo-transient step -- so a
laminar case could not be marched "exactly like the turbulent one", and a control experiment that asked
whether a solver behaviour needs turbulence had to use a weaker driver. These tests pin that the flow
residual runs on the shared driver, that its root is the one the bare step reaches, and that the
implicit-function-theorem adjoint at that root does not depend on which march reached it.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.flow import (
    FlowMeasures,
    flow_march_step,
    momentum_continuation,
    open_flow_session,
    solve_flow_march,
)
from aquaflux.flow import march as flow_march_module
from aquaflux.solve import (
    BlockScaled,
    CompleteLu,
    Convergence,
    DualTimeLoop,
    DualTimeStep,
    Euclidean,
    FieldSplit,
    Globalization,
    JacobiSmoothed,
    MaterializedJacobian,
    MonolithicFactorShiftPolicy,
    PseudoTransientStep,
    RefreshPolicy,
    RetryPolicy,
    RootSolver,
    RowScaled,
    SimpleSmoothed,
    assembler_residual,
    newton_step,
)

from .test_cavity import _cavity
from .test_channel_high_reynolds import _channel

MU = 5e-3  # Re = 200 on the 24 x 16 channel: past the floor a bare Newton step reaches
TIGHT = Convergence(measure=RowScaled(), rtol=0.0, atol=1e-9)
#: Newton from rest reaches the 8 x 8, Re = 100 cavity's root in five steps; one more is the margin.
NEWTON_STEPS_TO_THE_CAVITY_ROOT = 6


def _reference_root(assembler):
    """The root the bare pseudo-transient step reaches, the one every march here must agree with."""
    solver = RootSolver(max_steps=120, strategy=momentum_continuation(assembler))
    return solver.solve(assembler_residual, assembler.initial_state(), assembler)


@pytest.fixture(scope="module")
def channel():
    assembler = _channel(24, 16, MU)
    return assembler, _reference_root(assembler)


def test_the_march_reaches_the_root_the_bare_continuation_reaches(channel) -> None:
    assembler, root = channel
    state = solve_flow_march(assembler, convergence=TIGHT, max_steps=120)
    assert float(jnp.linalg.norm(assembler.residual(state))) < 1e-7
    # A wrong root would have a large residual; the same root differs only by the tolerances.
    assert float(jnp.linalg.norm(state - root) / jnp.linalg.norm(root)) < 1e-6


def test_a_dual_time_march_with_a_retry_policy_reaches_the_same_root(channel) -> None:
    """The capabilities the flow path lacked -- dual time, retries, a step control -- are reachable.

    The dual-time step is asserted to be the class the builder returned, so this cannot pass by the
    single-step path silently having run instead.
    """
    assembler, root = channel
    dual_time = DualTimeLoop(inner_steps=3)
    assert isinstance(
        flow_march_step(assembler, assembler.initial_state(), dual_time=dual_time),
        DualTimeStep,
    )
    reports = []
    state = solve_flow_march(
        assembler,
        convergence=TIGHT,
        max_steps=150,
        dual_time=dual_time,
        retry=RetryPolicy(),
        on_step=reports.append,
    )
    assert reports, "the march must report its steps"
    assert float(jnp.linalg.norm(state - root) / jnp.linalg.norm(root)) < 1e-6


class _FiresAtOnce:
    """A trigger that judges the preconditioner stale after two steps (a plain object: the protocol is structural)."""

    def should_refresh(self, history) -> bool:
        return len(history) >= 2


def test_a_refreshed_march_rebuilds_the_step_between_segments_and_reaches_the_same_root(
    channel, monkeypatch
) -> None:
    """A trigger that fires early re-freezes the preconditioner once, and the solve still converges.

    The build count is the evidence the refresh happened: a solve whose refresh silently never ran
    would build the step once and still converge.
    """
    assembler, root = channel
    builds = []
    real = flow_march_module.flow_march_step

    def counting(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(flow_march_module, "flow_march_step", counting)
    state = solve_flow_march(
        assembler,
        convergence=TIGHT,
        max_steps=150,
        refresh=RefreshPolicy(trigger=_FiresAtOnce(), limit=1),
    )
    assert len(builds) == 2
    assert float(jnp.linalg.norm(state - root) / jnp.linalg.norm(root)) < 1e-6


def test_the_gradient_through_the_march_matches_finite_differences_and_the_path(channel) -> None:
    """``jax.grad`` is the transpose solve at the root, so it cannot depend on the forward path.

    A finite-difference comparison, because a severed adjoint returns a finite zero. Two marches that
    take different paths -- the single shifted step and a dual-time loop -- must give the same gradient.
    """
    assembler, _ = channel

    def outlet_momentum(scale, **march):
        scaled = assembler.with_scaled_molecular_viscosity(scale)
        state = solve_flow_march(scaled, convergence=TIGHT, max_steps=150, **march)
        return jnp.sum(scaled.unpack(state)[0][:, 0] ** 2)

    single = jax.grad(outlet_momentum)(1.0)
    dual = jax.grad(lambda s: outlet_momentum(s, dual_time=DualTimeLoop(inner_steps=3)))(1.0)
    eps = 1e-4
    fd = (outlet_momentum(1.0 + eps) - outlet_momentum(1.0 - eps)) / (2 * eps)
    assert abs(float(single)) > 1e-6, "a severed adjoint would be zero"
    assert float(single) == pytest.approx(float(fd), rel=1e-4)
    assert float(dual) == pytest.approx(float(single), rel=1e-6)


def test_settings_that_configure_a_step_are_refused_beside_a_finished_step(channel) -> None:
    """A dual-time loop given beside a finished strategy would run the strategy's own, in silence."""
    assembler, _ = channel
    finished = flow_march_step(assembler, assembler.initial_state())
    assert isinstance(finished, PseudoTransientStep)
    with pytest.raises(TypeError, match=r"dual_time.*already carries them"):
        solve_flow_march(assembler, strategy=finished, dual_time=DualTimeLoop(inner_steps=3))


def test_an_unknown_march_setting_is_refused(channel) -> None:
    assembler, _ = channel
    with pytest.raises(TypeError, match="not_a_setting"):
        solve_flow_march(assembler, max_steps=1, not_a_setting=1)


def test_a_march_that_ends_short_of_the_root_is_refused_rather_than_returned(channel) -> None:
    """The adjoint is valid only at a root, so an exhausted step budget must raise."""
    assembler, _ = channel
    with pytest.raises(eqx.EquinoxRuntimeError, match="solve_flow_march did not converge"):
        solve_flow_march(assembler, convergence=TIGHT, max_steps=1)


def test_the_march_refuses_a_traced_program(channel) -> None:
    assembler, _ = channel
    with pytest.raises(ValueError, match="cannot run inside a traced program"):
        eqx.filter_jit(lambda a: solve_flow_march(a, max_steps=1))(assembler)


def test_the_block_scaled_measure_of_a_state_is_one_per_block_at_that_state(channel) -> None:
    """Every block's norm divided by its own magnitude at the reference state is exactly one.

    So the norm there is ``sqrt(n_blocks)``: a wrong scale (one block divided by another's magnitude)
    moves it off that value.
    """
    assembler, _ = channel
    state = assembler.initial_state() + 0.1
    norm = FlowMeasures(assembler).block_scaled(state)
    assert float(norm(assembler.residual(state))) == pytest.approx(
        len(norm.sizes) ** 0.5, rel=1e-12
    )


def test_the_row_scaled_measure_divides_each_row_by_the_scale_its_units_call_for(channel) -> None:
    """A unit error in one row reads as the fractional change that row's own scale implies.

    Expected values are written from the definition rather than read back from the measure: a momentum
    row's scale is its diagonal in the shift, its field scale the mean speed; the continuity row is a
    mass imbalance, scaled by the cell's mass throughput, with field scale one. The two errors carry
    different units, so a Euclidean norm -- which reads both as ``sqrt(n)`` -- cannot pass this.
    """
    assembler, root = channel
    step = flow_march_step(assembler, root)
    norm = FlowMeasures(assembler).row_scaled(step, root)
    n = assembler.mesh.n_cells
    velocity, _ = assembler.unpack(root)
    diagonal, _ = assembler.unpack(step.shift_policy.shift_term(root).diagonal)
    throughput, _ = assembler.momentum_matrix_diagonal_parts(velocity)

    momentum_error = jnp.zeros_like(root).at[:n].set(1.0)  # the first velocity component's rows
    continuity_error = jnp.zeros_like(root).at[-n:].set(1.0)
    expected_momentum = jnp.mean(1.0 / jnp.abs(diagonal[:, 0])) / jnp.mean(jnp.abs(velocity))
    expected_continuity = jnp.mean(1.0 / jnp.abs(throughput))
    assert float(norm(momentum_error)) == pytest.approx(float(expected_momentum), rel=1e-10)
    assert float(norm(continuity_error)) == pytest.approx(float(expected_continuity), rel=1e-10)
    assert float(jnp.linalg.norm(momentum_error)) == float(jnp.linalg.norm(continuity_error))


@pytest.mark.parametrize("measure", [Euclidean(), BlockScaled(), RowScaled()])
def test_every_measure_a_convergence_can_name_is_supplied_and_converges(channel, measure) -> None:
    """Each measure is built against the flow and drives the march to the root.

    ``BlockScaled`` starts from rest and the others from potential flow. A self-normalizing ``BlockScaled`` measure takes
    each block's scale from the start state's own residual, and potential flow is divergence-free, so its
    continuity block starts ~0 and every later step reads as an enormous relative increase and is
    rejected. That is a property of the coarse measure on a start that already satisfies one block, not
    of the march; the default ``RowScaled`` measure has no such sensitivity. ``RowScaled`` can start
    from rest too, since its scales are floored by the driving speed (the cavity test below).
    """
    assembler, root = channel
    start = assembler.initial_state() if isinstance(measure, BlockScaled) else None
    state = solve_flow_march(
        assembler,
        start,
        convergence=Convergence(measure=measure, rtol=1e-8, atol=0.0),
        max_steps=150,
    )
    assert float(jnp.linalg.norm(state - root) / jnp.linalg.norm(root)) < 1e-5


def test_a_lid_driven_cavity_marches_from_rest_under_the_default_measure() -> None:
    """A state at rest is a legitimate start, and the default row-scaled measure must accept it.

    Both of the measure's scales vanish at rest. Unfloored, the reference measure was NaN and the solve
    failed at step 0. Floored only while the state was exactly at rest, the step after it rebuilt the
    measure at about a hundredth of the lid speed and read 114x the start, past the divergence guard,
    and every later step was rejected. The floors the characteristic speed sets hold through the
    spin-up and release as the flow develops; the march must reach the root a plain Newton iteration
    from rest reaches.
    """
    cavity = _cavity(8)
    step = jax.jit(lambda state: newton_step(cavity.residual, state))  # one compile, not six
    reference = cavity.initial_state()
    for _ in range(NEWTON_STEPS_TO_THE_CAVITY_ROOT):
        reference = step(reference)
    assert float(jnp.linalg.norm(cavity.residual(reference))) < 1e-10

    state = solve_flow_march(cavity, cavity.initial_state(), max_steps=40)
    assert float(jnp.linalg.norm(state - reference) / jnp.linalg.norm(reference)) < 1e-6


def test_globalization_reaches_the_flow_march_step(channel) -> None:
    assembler, _ = channel
    step = flow_march_step(
        assembler, assembler.initial_state(), globalization=Globalization(line_search=3)
    )
    assert step.line_search == 3


def test_the_jacobian_gradient_sweeps_reach_the_step_and_only_the_step() -> None:
    """The narrowed copy is what the Krylov operator differentiates; the residual it drives is unchanged.

    Unset, the step differentiates the residual as it stands (``None``). Set, it carries a residual
    that is the narrowed copy's -- the assembler's own is never replaced, which is what keeps the root and
    the adjoint where they were.
    """
    assembler = _channel(8, 6, MU)
    state = assembler.initial_state()
    assert flow_march_step(assembler, state).jacobian_residual is None
    narrowed = flow_march_step(assembler, state, jacobian_gradient_sweeps=1).jacobian_residual
    assert narrowed is not None
    # A channel is orthogonal, so narrowing is free in value: the copy's residual is the same function.
    assert jnp.allclose(narrowed(state + 0.1), assembler.residual(state + 0.1), rtol=1e-12)


def test_a_complete_lu_march_reaches_the_root_the_block_simple_march_reaches(channel) -> None:
    """A laminar flow accepts the materialized-Jacobian preconditioner and lands on the same root.

    The complete LU is exact at the state and shift it was factored at, so this exercises the whole
    session -- probe, factorization, per-step re-fit under a dual-time march -- against the block-SIMPLE
    path's root, not against a transcription of it.
    """
    assembler, root = channel
    built = []
    session = open_flow_session(
        MaterializedJacobian(CompleteLu(backend="scipy")),
        assembler,
        on_build=lambda step: built.append(step) or step,
    )
    state = solve_flow_march(
        assembler,
        preconditioner=session,
        convergence=TIGHT,
        max_steps=150,
        dual_time=DualTimeLoop(inner_steps=3),
    )
    # The march ran on the materialized inverse, not on a block-SIMPLE step that happened to converge.
    assert built and isinstance(built[0].shift_policy, MonolithicFactorShiftPolicy)
    assert float(jnp.linalg.norm(state - root) / jnp.linalg.norm(root)) < 1e-6


def test_a_field_split_is_refused_for_a_flow_with_nothing_to_split(channel) -> None:
    assembler, _ = channel
    split = MaterializedJacobian(FieldSplit(leading=SimpleSmoothed(), trailing=JacobiSmoothed()))
    with pytest.raises(TypeError, match="single group"):
        open_flow_session(split, assembler)


def test_block_simple_settings_are_refused_beside_a_materialized_preconditioner(channel) -> None:
    assembler, _ = channel
    with pytest.raises(TypeError, match="no such settings"):
        solve_flow_march(
            assembler,
            preconditioner=MaterializedJacobian(CompleteLu(backend="scipy")),
            preconditioner_options={"schur_scaling": "msimple"},
        )


def test_a_preconditioner_is_refused_beside_a_finished_step(channel) -> None:
    assembler, _ = channel
    finished = flow_march_step(assembler, assembler.initial_state())
    with pytest.raises(TypeError, match=r"preconditioner.*already carries them"):
        solve_flow_march(
            assembler,
            strategy=finished,
            preconditioner=MaterializedJacobian(CompleteLu(backend="scipy")),
        )


def test_the_shift_settings_reach_the_flow_policy_field_for_field() -> None:
    """``shift`` carries both of its fields into the policy, and a field left unset keeps the default.

    Distinct sentinel values, so a swapped or dropped field shows up as the wrong object rather than
    coinciding with a default.
    """
    from aquaflux.solve import LocalCourantBasis, ShiftSettings

    assembler = _channel(8, 6, MU)
    state = assembler.initial_state()
    basis = LocalCourantBasis(dissipative_weight=0.0)
    parts = object()  # only stored here, never called

    both = flow_march_step(assembler, state, shift=ShiftSettings(basis=basis, velocity_parts=parts))
    assert both.shift_policy.shift_basis is basis
    assert both.shift_policy.velocity_shift_parts is parts

    only_basis = flow_march_step(assembler, state, shift=ShiftSettings(basis=basis))
    assert only_basis.shift_policy.shift_basis is basis
    assert only_basis.shift_policy.velocity_shift_parts is None

    unset = flow_march_step(assembler, state)
    assert unset.shift_policy.shift_basis == LocalCourantBasis()


def test_the_coupled_shift_settings_are_the_flow_ones_plus_the_closures_damping() -> None:
    """One flow part, defined once: the coupled value IS a ``ShiftSettings`` with one more field."""
    import dataclasses

    from aquaflux.solve import ShiftSettings
    from aquaflux.turbulence import CoupledShiftSettings

    assert issubclass(CoupledShiftSettings, ShiftSettings)
    flow_fields = {f.name for f in dataclasses.fields(ShiftSettings)}
    assert {f.name for f in dataclasses.fields(CoupledShiftSettings)} == flow_fields | {
        "turbulence_damping"
    }


def test_a_simple_smoothed_march_needs_no_petsc_and_reaches_the_same_root(channel) -> None:
    """A laminar flow takes the traced ``SimpleSmoothed`` hierarchy over its whole ``(u, p)`` saddle.

    This is the inverse the field split uses for the saddle, with no optional dependency, so it runs in
    CI. The refresh observer is the evidence the in-place refit ran (a solve that never refit would still
    converge on the state it was first fitted at), and the isinstance check that the march ran on it.
    """
    from aquaflux.solve import MaterializedBlockPreconditioner

    assembler, root = channel
    built, timings = [], []
    session = open_flow_session(
        MaterializedJacobian(SimpleSmoothed()),
        assembler,
        on_build=lambda step: built.append(step) or step,
        observer=timings.append,
    )
    state = solve_flow_march(
        assembler,
        preconditioner=session,
        convergence=TIGHT,
        max_steps=150,
        dual_time=DualTimeLoop(inner_steps=3),
    )
    assert built and isinstance(
        built[0].shift_policy.preconditioner, MaterializedBlockPreconditioner
    )
    assert any(
        t.kind == "full" and any(name == "refactor" for name, _ in t.phases) for t in timings
    )
    assert float(jnp.linalg.norm(state - root) / jnp.linalg.norm(root)) < 1e-6


def test_the_block_preconditioner_is_an_approximate_inverse_and_transposes_exactly() -> None:
    """``M`` inverts the shifted Jacobian it was fitted to, and ``M^T`` is its exact transpose.

    The adjoint's transpose solve applies ``M^T`` and a non-flexible Krylov solve needs ``M`` fixed and
    linear, so both are properties of the object and not of any one march. Judged against the operator
    itself (``M (J + D) v`` should return ``v`` up to the approximation of a one-cycle hierarchy),
    which a preconditioner fitted to the wrong matrix -- unshifted, say -- would not do.
    """
    import numpy as np
    from aquaflux.solve import (
        MaterializedBlockPreconditioner,
        jacobian_matvec,
        jacobian_probe_plan,
    )

    assembler = _channel(8, 6, MU)
    state = assembler.initial_state() + 0.05
    plan = jacobian_probe_plan(
        assembler.mesh.face_cells, assembler.mesh.n_cells, assembler.layout.n_fields, 3
    )
    shift = np.full(
        state.shape, 2.0
    )  # a strong pseudo-time shift keeps the saddle well conditioned
    pc = MaterializedBlockPreconditioner.build(
        lambda v: jacobian_matvec(assembler, state, v),
        plan,
        shift,
        inverse=SimpleSmoothed(),
        n_fields=assembler.layout.n_fields,
    )
    rng = np.random.default_rng(0)
    x, y = (jnp.asarray(rng.standard_normal(state.shape)) for _ in range(2))
    apply, apply_t = pc.matvec(), pc.matvec(transpose=True)
    assert float(jnp.vdot(y, apply(x))) == pytest.approx(float(jnp.vdot(apply_t(y), x)), rel=1e-9)

    v = jnp.asarray(rng.standard_normal(state.shape))
    shifted_v = jacobian_matvec(assembler, state, v) + jnp.asarray(shift) * v
    assert float(jnp.linalg.norm(apply(shifted_v) - v) / jnp.linalg.norm(v)) < 0.7


@pytest.fixture(scope="module")
def shared_step(channel):
    """One finished step every march below runs, so each is a cache hit on the first one's compile."""
    assembler, _ = channel
    return flow_march_step(assembler, assembler.initial_state())


def _march_with_states(assembler, step, state=None, **march):
    """Marches ``assembler`` with ``step`` and returns the reports and the state after each step."""
    reports, states = [], []

    def record(report, step_state):
        reports.append(report)
        states.append(jnp.asarray(step_state))

    solve_flow_march(
        assembler, state, strategy=step, on_checkpoint=record, convergence=TIGHT, **march
    )
    return reports, states


def test_a_march_resumed_with_its_reference_residual_continues_the_one_it_resumes(
    channel, shared_step
) -> None:
    """Resuming at a step's state with the original ``|R0|`` reproduces the steps that followed it.

    The wrong answers this catches: a resume that re-bases the damping ramp at the resumed state (the
    steps then differ from the original's), one that ignores the reference in the stopping target (the
    resumed march stops at a different residual), and one that applies it to the ramp but not to the
    stopping bar.
    """
    assembler, _ = channel
    fresh_reports, fresh_states = _march_with_states(assembler, shared_step)
    resumed_at = 3
    last = fresh_reports[resumed_at - 1]
    reference = float(last.residual_norm) / float(last.residual_ratio)

    control_reports, _ = _march_with_states(assembler, shared_step, fresh_states[resumed_at - 1])
    reports, states = _march_with_states(
        assembler, shared_step, fresh_states[resumed_at - 1], reference_residual=reference
    )

    expected = fresh_reports[resumed_at:]
    # Without the reference the resumed march is a different one: this is what the carry repairs.
    assert [r.residual_norm for r in control_reports] != pytest.approx(
        [r.residual_norm for r in expected], rel=1e-6
    )
    assert len(reports) == len(expected)
    assert [r.residual_norm for r in reports] == pytest.approx(
        [r.residual_norm for r in expected], rel=1e-8
    )
    assert [r.residual_ratio for r in reports] == pytest.approx(
        [r.residual_ratio for r in expected], rel=1e-8
    )
    assert float(jnp.max(jnp.abs(states[-1] - fresh_states[-1]))) < 1e-9


def test_a_reference_residual_is_the_stopping_targets_scale(channel, shared_step) -> None:
    """A march stops against ``atol + rtol * reference``, so the reference moves where it stops."""
    assembler, _ = channel
    reports, states = _march_with_states(assembler, shared_step)
    start = states[1]
    loose = Convergence(measure=RowScaled(), rtol=1e-2, atol=0.0)
    tight = Convergence(measure=RowScaled(), rtol=1e-6, atol=0.0)

    def steps(reference, convergence):
        taken = []
        solve_flow_march(
            assembler,
            start,
            strategy=shared_step,
            convergence=convergence,
            max_steps=120,
            reference_residual=reference,
            on_checkpoint=lambda report, _: taken.append(report),
        )
        return taken

    reference = float(reports[0].residual_norm) / float(reports[0].residual_ratio)
    # Against a reference 1e4 times larger the target (here 100 x the starting residual) is already met,
    # so the march takes no step at all -- and with no step the damping anchor cannot be what differs,
    # so this isolates the stopping scale. Against the original reference it has work to do.
    assert steps(reference * 1e4, loose) == []
    assert len(steps(reference, loose)) > 0
    assert len(steps(reference, tight)) > len(steps(reference, loose))


def test_a_march_resumed_at_its_root_takes_no_step_whatever_reference_it_is_given(
    channel, shared_step
) -> None:
    """The march starts from the residual it measures at the state it is handed, not from the reference.

    A resume at a converged state has nothing left to do; a march that took its starting residual
    from the reference (here far above the target) would think the state unconverged and step.
    """
    assembler, _ = channel
    _, states = _march_with_states(assembler, shared_step)
    reports, _ = _march_with_states(assembler, shared_step, states[-1], reference_residual=1.0e3)
    assert reports == []


def test_a_reference_residual_beside_a_homotopy_or_that_is_not_positive_is_refused(channel) -> None:
    assembler, _ = channel
    with pytest.raises(ValueError, match="positive finite"):
        solve_flow_march(assembler, reference_residual=0.0, max_steps=1)
    with pytest.raises(ValueError, match="cannot be combined with a homotopy"):
        solve_flow_march(assembler, reference_residual=1.0, homotopy=object(), max_steps=1)
