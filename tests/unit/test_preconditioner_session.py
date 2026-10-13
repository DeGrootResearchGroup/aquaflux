"""The preconditioner session and ``coupled_step``: built once, shared, and refusing what they cannot use.

When the session replaced the three preconditioner-specific builders, its steps were first pinned
array-identical to theirs for every family; those comparisons went with the builders. What stays is
what a session adds: one inverse and one set of hooks per session, the probe following the operator,
the driver seams, and the refusals.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.flow.block_preconditioner import _characteristic_reference_state
from aquaflux.precondition import (
    AirReduction,
    FieldSplit,
    FieldSplitPreconditioner,
    JacobianProbeSpec,
    JacobiSmoothed,
    MaterializedJacobian,
    MonolithicFactorShiftPolicy,
    SimpleSmoothed,
)
from aquaflux.precondition.monolithic_policy import TracedSplitShiftPolicy
from aquaflux.precondition.traced_field_split import contains_host_callback
from aquaflux.solve import ConstantRelaxation, DualTimeLoop
from aquaflux.turbulence import BlockDiagonal, ScalarAir, ScalarTwoLevel, open_session

from tests.unit.test_coupled_rans import NU, RHO, _cavity, _healthy_state

_SPLIT = FieldSplit(SimpleSmoothed(), JacobiSmoothed())


@pytest.fixture(scope="module")
def case():
    mesh, coupled = _cavity()
    return coupled, _healthy_state(mesh, coupled)


def test_a_block_session_refresh_carries_the_flow_block(case) -> None:
    """A refresh re-derives the scalar blocks on the reused coarsening and carries the flow block over."""
    coupled, state = case
    session = open_session(BlockDiagonal(), coupled)
    first = session.build(state, dual_time=DualTimeLoop(inner_steps=3))
    refreshed = session.refresh(state * 1.01, first, dual_time=DualTimeLoop(inner_steps=3))
    assert refreshed.shift_policy.flow_preconditioner is first.shift_policy.flow_preconditioner


def test_a_re_pointed_block_session_hands_the_march_the_new_problems_shift(case) -> None:
    """Re-pointed at a companion, the next step is shifted and preconditioned for the companion.

    The shift's velocity block reads the momentum diagonal, whose viscous part scales with the
    molecular viscosity, so a step still built on the original assembler shifts a station with a
    hundredfold viscosity by the wrong amount. The march's own shift strength must survive the swap.
    """
    coupled, state = case
    companion = coupled.with_scaled_molecular_viscosity(100.0)
    loop = DualTimeLoop(inner_steps=3)
    session = open_session(BlockDiagonal(), coupled)
    held = eqx.tree_at(
        lambda s: s.relaxation_schedule,
        session.build(state, dual_time=loop),
        ConstantRelaxation(jnp.asarray(0.125)),
    )
    assert session.refresh_preconditioner(held, state) is None
    session.rebind(companion)
    updated = session.refresh_preconditioner(held, state)(held)
    expected = open_session(BlockDiagonal(), companion).build(state, dual_time=loop)
    shift = updated.shift_policy.shift_term(state).diagonal
    assert jnp.allclose(shift, expected.shift_policy.shift_term(state).diagonal, rtol=1e-12)
    original = held.shift_policy.shift_term(state).diagonal
    assert not jnp.allclose(shift, original, rtol=1e-3)
    assert float(updated.relaxation_schedule.beta) == 0.125
    # The flow block keeps its frozen hierarchy but preconditions the companion: the momentum diagonal
    # it rescales that hierarchy to is the companion's.
    flow = coupled.layout.unpack(state)[0]
    a_p = updated.shift_policy.flow_preconditioner.frozen_momentum_diagonal(flow)
    companion_a_p = expected.shift_policy.flow_preconditioner.frozen_momentum_diagonal(flow)
    assert jnp.allclose(a_p, companion_a_p, rtol=1e-12)
    anchor_a_p = held.shift_policy.flow_preconditioner.frozen_momentum_diagonal(flow)
    assert not jnp.allclose(a_p, anchor_a_p, rtol=1e-3)


@pytest.mark.parametrize("scalar", [ScalarTwoLevel(), ScalarAir()], ids=["twolevel", "air"])
def test_a_re_pointed_block_session_step_is_a_compilation_cache_hit(scalar) -> None:
    """A station change hands the march a step it has already compiled, for either scalar block.

    The march's step is compiled against the whole step it holds, so a re-pointed step that differs in
    anything but array values -- a new preconditioner object, a coarse grid of another size -- recompiles
    the coupled solve at every station. The flow block's strength-of-connection aggregation reads the
    operator's values, and lAIR's coarse/fine split does too, so a step built afresh on the companion is
    a miss (the control below); the re-pointed one keeps every hierarchy's coarsening and is a hit.
    """
    mesh, coupled = _cavity(viscosity=jnp.asarray(RHO * NU))  # an array leaf; a float is static
    # The flow block is frozen at this state's flow, so the flow must carry the lid's convective scale
    # for its coarsening to depend on the viscosity at all: at the healthy state's small random flow the
    # operator is viscous-dominated and the control's fresh build would coarsen the same way.
    _, k, omega = coupled.physical_fields(_healthy_state(mesh, coupled))
    state = coupled.pack_state(_characteristic_reference_state(coupled.momentum), k, omega)
    loop = DualTimeLoop(inner_steps=3)
    traces: list[None] = []

    @eqx.filter_jit
    def apply(step, state):
        traces.append(None)
        term = step.shift_policy.shift_term(state)
        shifted = term.make_preconditioner(jnp.asarray(0.5))(state)
        return shifted + step.adjoint_preconditioner_factory(state)(state)

    session = open_session(BlockDiagonal(scalar=scalar), coupled)
    held = session.build(state, dual_time=loop)
    apply(held, state)
    for scale in (100.0, 10.0):
        companion = coupled.with_scaled_molecular_viscosity(scale)
        session.rebind(companion)
        held = session.refresh_preconditioner(held, state)(held)
        apply(held, state)
    assert len(traces) == 1
    rebuilt = open_session(BlockDiagonal(scalar=scalar), companion).build(state, dual_time=loop)
    apply(rebuilt, state)
    assert len(traces) == 2


def _materialized_step_held_on(coupled, anchor, state, loop, beta=0.125):
    """A materialized session opened on ``coupled``, re-pointed at ``anchor``, and its step as a march holds it.

    That is how a viscosity ramp opens its session: on the target, re-pointed at the anchor before the
    first build, so the first step is fitted to the problem it solves. The held step carries the shift
    strength a step control set, and has been through the march's first refresh, the forced first fit.
    """
    session = open_session(MaterializedJacobian(_SPLIT), coupled)
    session.rebind(anchor)
    held = eqx.tree_at(
        lambda s: s.relaxation_schedule,
        session.build(state, dual_time=loop),
        ConstantRelaxation(jnp.asarray(beta)),
    )
    update = session.refresh_preconditioner(held, state)
    return session, held if update is None else update(held)


def test_a_re_pointed_materialized_session_hands_the_march_the_new_problems_shift(case) -> None:
    """Re-pointed at the target, the held step is shifted, and measured, as the target's own step is.

    The step's base shift diagonal carries the momentum ``a_P``, whose viscous part scales with the
    molecular viscosity, so a step left on the anchor shifts the target by the anchor's hundredfold
    diagonal. The row-scaled measure takes its row scales from the same policy, so it would judge the
    target in the anchor's scales. The march's own shift strength must survive the swap.
    """
    from aquaflux.turbulence.measures import coupled_scaled_norm

    target, state = case
    anchor = target.with_scaled_molecular_viscosity(100.0)
    loop = DualTimeLoop(inner_steps=3)
    session, held = _materialized_step_held_on(target, anchor, state, loop)
    assert (
        session.refresh_preconditioner(held, state) is None
    )  # not re-pointed: nothing to hand back
    session.rebind(target)
    update = session.refresh_preconditioner(held, state)
    assert update is not None
    updated = update(held)

    expected = open_session(MaterializedJacobian(_SPLIT), target).build(state, dual_time=loop)
    shift = updated.shift_policy.base.shift_term(state).diagonal
    assert jnp.allclose(shift, expected.shift_policy.base.shift_term(state).diagonal, rtol=1e-12)
    assert not jnp.allclose(shift, held.shift_policy.base.shift_term(state).diagonal, rtol=1e-3)
    assert float(updated.relaxation_schedule.beta) == 0.125

    residual = target.residual(state)
    measured = coupled_scaled_norm(target, updated.shift_policy, state)(residual)
    in_target_scales = coupled_scaled_norm(target, expected.shift_policy, state)(residual)
    in_anchor_scales = coupled_scaled_norm(target, held.shift_policy, state)(residual)
    assert float(measured) == pytest.approx(float(in_target_scales), rel=1e-12)
    assert float(measured) != pytest.approx(float(in_anchor_scales), rel=1e-6)


def test_a_re_pointing_moves_the_problem_and_not_the_state_the_shift_was_built_at(case) -> None:
    """The graft is built on the new companion at the HELD steps' build state, not the march's iterate.

    A policy frozen at a state carries that state's eddy viscosity, transport time scale and (for the
    log-solved omega) coordinate factor. Re-freezing them at the iterate a station is entered at moves
    the shift on every row, the closure's included, where the companion moves only what its viscosity
    reaches; that is a refresh's job, not a re-pointing's.
    """
    target, state = case
    anchor = target.with_scaled_molecular_viscosity(100.0)
    loop = DualTimeLoop(inner_steps=3)
    session, held = _materialized_step_held_on(target, anchor, state, loop)
    iterate = state * 1.05
    session.rebind(target)
    updated = session.refresh_preconditioner(held, iterate)(held)

    def base_diagonal(step):
        return step.shift_policy.base.shift_term(state).diagonal

    at_build_state = open_session(MaterializedJacobian(_SPLIT), target).build(state, dual_time=loop)
    at_iterate = open_session(MaterializedJacobian(_SPLIT), target).build(iterate, dual_time=loop)
    assert jnp.allclose(base_diagonal(updated), base_diagonal(at_build_state), rtol=1e-12)
    assert not jnp.allclose(base_diagonal(updated), base_diagonal(at_iterate), rtol=1e-6)


def test_a_re_pointed_materialized_session_refits_against_the_new_problems_shift(
    case, monkeypatch
) -> None:
    """The re-fit at a station change is of the station's Jacobian plus the STATION'S shift.

    The inverse is re-fitted at ``J + beta * d``; with ``d`` read off a step still built on the anchor,
    it fits the target's Jacobian plus the anchor's shift, an operator the march never solves.
    """
    target, state = case
    anchor = target.with_scaled_molecular_viscosity(100.0)
    session, held = _materialized_step_held_on(target, anchor, state, DualTimeLoop(inner_steps=3))
    built_at: list[jnp.ndarray] = []
    monkeypatch.setattr(
        FieldSplitPreconditioner,
        "refresh_in_place",
        lambda _self, _mv, _plan, shift, **_kw: built_at.append(jnp.asarray(shift)),
    )
    session.rebind(target)
    session.refresh_preconditioner(held, state)
    assert len(built_at) == 1
    expected = open_session(MaterializedJacobian(_SPLIT), target).build(
        state, dual_time=DualTimeLoop(inner_steps=3)
    )
    at_target = 0.125 * expected.shift_policy.base.shift_term(state).diagonal
    at_anchor = 0.125 * held.shift_policy.base.shift_term(state).diagonal
    assert jnp.allclose(built_at[0], at_target, rtol=1e-12)
    assert not jnp.allclose(built_at[0], at_anchor, rtol=1e-3)


def test_a_re_pointed_materialized_session_step_is_a_compilation_cache_hit() -> None:
    """A station change hands the march a step of the structure it has already compiled.

    The march's step is compiled against the whole step it holds, so a re-pointed step that differs in
    anything but array values recompiles the coupled solve at every station of a ramp.
    """
    mesh, target = _cavity(viscosity=jnp.asarray(RHO * NU))  # an array leaf; a float is static
    state = _healthy_state(mesh, target)
    loop = DualTimeLoop(inner_steps=3)
    traces: list[None] = []

    @eqx.filter_jit
    def apply(step, state):
        traces.append(None)
        term = step.shift_policy.shift_term(state)
        shifted = term.make_preconditioner(jnp.asarray(0.5))(state)
        return shifted + step.adjoint_preconditioner_factory(state)(state)

    session, held = _materialized_step_held_on(
        target, target.with_scaled_molecular_viscosity(100.0), state, loop
    )
    apply(held, state)
    for scale in (10.0, 1.0):
        session.rebind(target.with_scaled_molecular_viscosity(scale))
        held = session.refresh_preconditioner(held, state)(held)
        apply(held, state)
    assert len(traces) == 1


def test_a_re_pointing_does_not_instrument_a_step_the_march_never_takes(case) -> None:
    """``on_build`` sees every step handed to a march, and not the step a re-pointing grafts from.

    The graft takes three fields of that step onto a held step ``on_build`` has already instrumented; a
    driver that records something per step it builds (a residual reporter, say) would otherwise gain a
    record per station for a step no march ever runs.
    """
    target, state = case
    built: list[object] = []
    session = open_session(
        MaterializedJacobian(_SPLIT), target, on_build=lambda step: built.append(step) or step
    )
    session.rebind(target.with_scaled_molecular_viscosity(100.0))
    held = eqx.tree_at(
        lambda s: s.relaxation_schedule,
        session.build(state, dual_time=DualTimeLoop(inner_steps=3)),
        ConstantRelaxation(jnp.asarray(0.125)),
    )
    session.refresh_preconditioner(held, state)
    session.rebind(target)
    assert session.refresh_preconditioner(held, state) is not None
    assert len(built) == 1


def test_the_session_hook_refits_when_the_march_asks_and_not_otherwise(case) -> None:
    """A multigrid session re-fits only on its first call, after a rebind, and when ``due`` is passed.

    ``due`` is how the march asks for the re-fit a step stopped for (``refresh_on_cycles``), so a hook
    that ignored it would leave every such step to run again on the preconditioner it reported as
    struggling. The observer records what each call did: the first is the forced first fit, a call that
    is only offered reuses the standing inverse, and an asked-for one re-fits and says why.
    """
    coupled, state = case
    kinds: list[str] = []
    session = open_session(
        MaterializedJacobian(_SPLIT), coupled, observer=lambda timing: kinds.append(timing.kind)
    )
    loop = DualTimeLoop(inner_steps=3, refresh_on_cycles=3)
    step = session.build(state, dual_time=loop)
    assert step.refresh_on_cycles == 3
    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(0.5))
    session.refresh_preconditioner(active, state)
    session.refresh_preconditioner(active, state)
    session.refresh_preconditioner(active, state, due=True)
    session.refresh_preconditioner(active, state, due=False)
    assert kinds == ["full", "none", "cost", "none"]


def test_an_asked_for_refit_is_built_at_the_state_the_march_hands_it(case, monkeypatch) -> None:
    """The re-fit a step stopped for is materialized where that step got to, at the next step's shift.

    The march hands the hook the iterate the stopped step kept, so a re-fit built anywhere else -- at
    the state the session was first built at, say -- would reproduce the staleness it exists to remove.
    The shift is the step's own ``beta`` times the base diagonal AT THAT STATE, which differs from the
    diagonal at the build state, so the assertion cannot be met by building at the wrong state.
    """
    coupled, state = case
    session = open_session(MaterializedJacobian(_SPLIT), coupled)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(0.5))
    session.refresh_preconditioner(active, state)  # the forced first fit, not under test

    built_at: list[jnp.ndarray] = []
    monkeypatch.setattr(
        FieldSplitPreconditioner,
        "refresh_in_place",
        lambda _self, _mv, _plan, shift, **_kw: built_at.append(jnp.asarray(shift)),
    )
    iterate = state * 1.05
    session.refresh_preconditioner(active, iterate, due=True)
    assert len(built_at) == 1
    at_iterate = 0.5 * active.shift_policy.base.shift_term(iterate).diagonal
    at_start = 0.5 * active.shift_policy.base.shift_term(state).diagonal
    assert jnp.allclose(built_at[0], at_iterate)
    assert not jnp.allclose(built_at[0], at_start)


def test_every_build_of_a_session_shares_one_inverse_and_one_set_of_hooks(case) -> None:
    """A second build glues in the inverse the first one fitted, and keeps the compiled step's key.

    A host inverse rides in a static field compared by identity, so it must be the same object; a
    traced split rides as data, so it must hold the same values over the same structure. Either way a
    second build fits nothing and a march built from it is a compilation-cache hit.
    """
    coupled, state = case
    session = open_session(MaterializedJacobian(_SPLIT), coupled)
    hook = session.refresh_preconditioner
    first = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    second = session.build(state * 1.01, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    first_leaves, first_def = jax.tree.flatten(eqx.filter(first.shift_policy.split, eqx.is_array))
    second_leaves, second_def = jax.tree.flatten(
        eqx.filter(second.shift_policy.split, eqx.is_array)
    )
    assert first_def == second_def
    assert all(jnp.array_equal(a, b) for a, b in zip(first_leaves, second_leaves, strict=True))
    assert session.refresh_preconditioner is hook


def test_a_split_of_two_traced_blocks_is_carried_in_the_step_and_each_refit_hands_back_a_new_one(
    case,
) -> None:
    """Both blocks offer a traced cycle, so the step applies the split on device and a re-fit is a new step.

    Three things a wrong wiring would get wrong, each asserted. The step must carry the traced form (no
    host callback in a Krylov iteration). A re-fit must come back as an update -- the traced split is
    data and cannot be mutated in place -- while an offered refresh that re-fits nothing returns none.
    And the update must reach BOTH places the split rides, the shift policy and the adjoint factory, at
    an unchanged structure (the compiled step's key), applying the host split's re-fitted values.
    """
    coupled, state = case
    session = open_session(MaterializedJacobian(_SPLIT), coupled)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    assert isinstance(step.shift_policy, TracedSplitShiftPolicy)
    n = state.shape[0]
    assert not contains_host_callback(step.shift_policy.split.matvec(), jnp.zeros(n))

    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(0.5))
    assert session.refresh_preconditioner(active, state) is not None  # the forced first fit
    assert session.refresh_preconditioner(active, state) is None  # offered, nothing re-fitted

    update = session.refresh_preconditioner(active, state * 1.05, due=True)
    assert update is not None
    refitted = update(active)
    r = jax.random.normal(jax.random.PRNGKey(0), (n,))
    host = session.preconditioner.inverse.apply(jnp.asarray(r))
    applied = refitted.shift_policy.split.apply(r)
    assert jnp.allclose(applied, host, rtol=1e-12, atol=1e-14)
    assert not jnp.allclose(applied, active.shift_policy.split.apply(r))
    assert jnp.array_equal(refitted.adjoint_preconditioner_factory(state)(r), applied)
    assert jax.tree.structure(eqx.filter(refitted, eqx.is_array)) == jax.tree.structure(
        eqx.filter(active, eqx.is_array)
    )


class _HostOnly:
    """A block inverse with the host interface and no traced cycle, as a factorization would be."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.n_dofs = inner.n_dofs

    def apply(self, residual, *, transpose: bool = False):
        return self._inner.apply(residual, transpose=transpose)

    def refactor_block(self, block) -> None:
        self._inner.refactor_block(block)


def test_a_split_with_a_host_block_is_applied_through_a_callback_and_refit_in_place(case) -> None:
    """A block with no traced cycle has no on-device form, so the split stays a host object.

    The control for the test above: the traced form is chosen by what the blocks offer, so a block that
    offers nothing must leave the session on the host path, where a re-fit happens in place and the
    step needs no update.
    """
    coupled, state = case

    def host_trailing(role, factory):
        if role == "leading":
            return factory
        return lambda block, n_fields: _HostOnly(factory(block, n_fields))

    session = open_session(MaterializedJacobian(_SPLIT), coupled, inverse_wrapper=host_trailing)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    assert isinstance(step.shift_policy, MonolithicFactorShiftPolicy)
    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(0.5))
    assert session.refresh_preconditioner(active, state) is None
    assert session.refresh_preconditioner(active, state * 1.05, due=True) is None


def test_the_session_probe_follows_the_operator_stand_in(case) -> None:
    coupled, state = case
    session = open_session(
        MaterializedJacobian(_SPLIT), coupled, jacobian_production_viscosity=True
    )
    session.build(state, dual_time=DualTimeLoop(inner_steps=3))
    probe = session._probe_for()
    assert probe.narrowing.production_viscosity_frozen


def test_a_probe_spec_reaches_the_probe(case) -> None:
    coupled, _ = case
    session = open_session(
        MaterializedJacobian(_SPLIT, probe=JacobianProbeSpec(gradient_sweeps=1)), coupled
    )
    assert session._probe_for().narrowing.gradient_sweeps == 1


def test_reports_and_the_inverse_wrapper_reach_the_field_split_blocks(case) -> None:
    coupled, state = case
    messages: list[str] = []
    wrapped: list[str] = []

    def wrapper(role, factory):
        wrapped.append(role)
        return factory

    session = open_session(
        MaterializedJacobian(_SPLIT),
        coupled,
        reports={"leading": messages.append},
        inverse_wrapper=wrapper,
    )
    session.build(state, dual_time=DualTimeLoop(inner_steps=3))
    assert messages, "the leading inverse's build record did not reach its sink"
    assert sorted(wrapped) == ["leading", "trailing"]


def test_the_precondition_wrapper_wraps_the_hook_the_march_calls(case) -> None:
    coupled, _ = case
    calls: list[object] = []

    def wrapper(hook):
        def recorded(step, state, *, due=False):
            calls.append(step)

        return recorded

    session = open_session(MaterializedJacobian(_SPLIT), coupled, precondition_wrapper=wrapper)
    session.refresh_preconditioner("step", None)
    assert calls == ["step"]


def test_the_block_family_refuses_settings_it_cannot_use(case) -> None:
    coupled, _ = case
    with pytest.raises(TypeError, match="nothing to act on in the block-diagonal family"):
        open_session(BlockDiagonal(), coupled, observer=print)


def test_block_inverse_settings_are_refused_without_a_field_split(case) -> None:
    coupled, _ = case
    with pytest.raises(TypeError, match="SimpleSmoothed has none"):
        open_session(MaterializedJacobian(SimpleSmoothed()), coupled, reports={"leading": print})


def test_a_report_for_an_inverse_with_no_record_is_refused_when_the_session_opens(case) -> None:
    coupled, _ = case
    with pytest.raises(TypeError, match="no build record"):
        open_session(
            MaterializedJacobian(FieldSplit(SimpleSmoothed(), AirReduction())),
            coupled,
            reports={"trailing": print},
        )


def test_a_build_refuses_keywords_the_session_owns_and_keywords_that_do_not_exist(case) -> None:
    coupled, state = case
    session = open_session(BlockDiagonal(), coupled)
    with pytest.raises(TypeError, match="belong to the session"):
        session.build(state, jacobian_production_viscosity=True)
    with pytest.raises(TypeError):
        session.build(state, beta0=1.5)


def test_a_session_cannot_be_re_pointed_at_a_different_case(case) -> None:
    coupled, _ = case
    _, other = _cavity(n=7)
    session = open_session(MaterializedJacobian(_SPLIT), coupled)
    with pytest.raises(ValueError, match="SAME case"):
        session.rebind(other)


def test_a_materialized_build_refuses_a_traced_state(case) -> None:
    coupled, state = case
    session = open_session(MaterializedJacobian(_SPLIT), coupled)

    def objective(s):
        session.build(s)
        return jnp.sum(s)

    with pytest.raises(ValueError, match=r"cannot be built under jax\.grad"):
        jax.grad(objective)(state)


def test_a_bare_block_inverse_is_refused_for_a_problem_with_two_groups(case) -> None:
    """A bare block inverse is fitted to the whole state, which is only meaningful for one group of fields.

    Coupled RANS has a leading saddle and trailing scalars, so a saddle hierarchy over all six fields
    would be fitted to the wrong operator; it must be wrapped in a FieldSplit.
    """
    coupled, _ = case
    with pytest.raises(TypeError, match="single group"):
        open_session(MaterializedJacobian(SimpleSmoothed()), coupled)
