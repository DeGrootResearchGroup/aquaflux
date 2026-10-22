"""The preconditioner session and ``coupled_step``: built once, shared, and refusing what they cannot use.

When the session replaced the three preconditioner-specific builders, its steps were first pinned
array-identical to theirs for every family; those comparisons went with the builders. What stays is
what a session adds: one inverse and one set of hooks per session, the probe following the operator,
the driver seams, and the refusals.
"""

from __future__ import annotations

import dataclasses

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


def test_a_traced_split_step_is_one_compiled_program_across_refits_and_stations() -> None:
    """A cost refit and a station change each hand the march a step it has already compiled.

    The march compiles its step against the WHOLE step, static fields included, so what a refit or a
    re-pointing hands back must differ from the held step in array values only: a static value moved by
    a refit (a precision, a level count, a hook rebuilt) recompiles the coupled solve once per refit, and
    one moved by a station once per station. Measured on the shipped pitzDaily march (21 refits, 12
    stations), nothing compiles after step 1, which is what this pins on a small case. The traced
    program takes the whole step as an argument, as the march's own does, and applies what a Krylov
    iteration and the adjoint apply.
    """
    mesh, target = _cavity(viscosity=jnp.asarray(RHO * NU))  # an array leaf; a float is static
    state = _healthy_state(mesh, target)
    split = FieldSplit(
        SimpleSmoothed(frozen_coarsening=True, max_coarse=10), JacobiSmoothed(max_coarse=10)
    )
    loop = DualTimeLoop(inner_steps=3, refresh_on_cycles=3)
    traces: list[None] = []

    @eqx.filter_jit
    def apply(step, state):
        traces.append(None)
        term = step.shift_policy.shift_term(state)
        shifted = term.make_preconditioner(step.relaxation_schedule.beta)(state)
        return shifted + step.adjoint_preconditioner_factory(state)(state)

    session = open_session(MaterializedJacobian(split), target)
    session.rebind(target.with_scaled_molecular_viscosity(100.0))
    held = eqx.tree_at(
        lambda s: s.relaxation_schedule,
        session.build(state, dual_time=loop),
        ConstantRelaxation(jnp.asarray(0.5)),
    )
    held = session.refresh_preconditioner(held, state)(held)  # the forced first fit
    first = apply(held, state)
    held = session.refresh_preconditioner(held, state * 1.05, due=True)(held)  # a cost refit
    for scale in (10.0, 1.0):  # two stations, the second the target
        session.rebind(target.with_scaled_molecular_viscosity(scale))
        held = session.refresh_preconditioner(held, state)(held)
    last = apply(held, state)
    assert not jnp.allclose(last, first)  # the refits reached the step
    assert len(traces) == 1


def _base_diagonal(step, state):
    return step.shift_policy.base.shift_term(state).diagonal


def test_a_re_pointed_materialized_session_keeps_the_shift_the_held_step_carries(case) -> None:
    """Re-pointed at the target, the held step keeps the anchor's shift diagonal and shift strength.

    The step control adapts the shift strength on the premise that the diagonal it scales is fixed, so
    a station change must not swap the diagonal under it. What the re-pointing hands back, if anything,
    is the re-fitted inverse.
    """
    target, state = case
    anchor = target.with_scaled_molecular_viscosity(100.0)
    session, held = _materialized_step_held_on(target, anchor, state, DualTimeLoop(inner_steps=3))
    session.rebind(target)
    update = session.refresh_preconditioner(held, state)
    stepped = held if update is None else update(held)

    on_target = open_session(MaterializedJacobian(_SPLIT), target).build(
        state, dual_time=DualTimeLoop(inner_steps=3)
    )
    assert jnp.array_equal(_base_diagonal(stepped, state), _base_diagonal(held, state))
    assert not jnp.allclose(
        _base_diagonal(stepped, state), _base_diagonal(on_target, state), rtol=1e-3
    )
    assert float(stepped.relaxation_schedule.beta) == 0.125


def test_a_re_pointed_materialized_session_refits_against_the_shift_the_step_carries(
    case, monkeypatch
) -> None:
    """The re-fit at a station change is of the station's Jacobian plus the shift the march solves with.

    That shift is the held step's, so an inverse fitted to the new problem's own diagonal would
    precondition an operator the march never solves.
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
    assert jnp.allclose(built_at[0], 0.125 * _base_diagonal(held, state), rtol=1e-12)


def test_the_stopping_step_measures_the_target_in_its_own_row_scales(case) -> None:
    """``stopping_step(target)`` hands the held step the target's shift policy, for a measure only.

    It is built at the state the session's steps were built at, so only the problem differs: the march
    is judged in the row scales the target's own step would have from that state.
    """
    from aquaflux.turbulence.measures import coupled_scaled_norm

    target, state = case
    anchor = target.with_scaled_molecular_viscosity(100.0)
    loop = DualTimeLoop(inner_steps=3)
    session, held = _materialized_step_held_on(target, anchor, state, loop)
    judged = session.stopping_step(target)(held)

    on_target = open_session(MaterializedJacobian(_SPLIT), target).build(state, dual_time=loop)
    assert jnp.allclose(_base_diagonal(judged, state), _base_diagonal(on_target, state), rtol=1e-12)
    residual = target.residual(state)
    in_target_scales = coupled_scaled_norm(target, on_target.shift_policy, state)(residual)
    in_anchor_scales = coupled_scaled_norm(target, held.shift_policy, state)(residual)
    measured = coupled_scaled_norm(target, judged.shift_policy, state)(residual)
    assert float(measured) == pytest.approx(float(in_target_scales), rel=1e-12)
    assert float(measured) != pytest.approx(float(in_anchor_scales), rel=1e-6)


def test_the_stopping_step_follows_the_state_a_refresh_rebuilds_at(case) -> None:
    """After a refresh re-freezes the session's steps at a new state, the stopping step is rebuilt there.

    A stopping step built at the old state would judge the march in scales frozen at a state none of
    its steps are built at any more.
    """
    target, state = case
    anchor = target.with_scaled_molecular_viscosity(100.0)
    loop = DualTimeLoop(inner_steps=3)
    session, held = _materialized_step_held_on(target, anchor, state, loop)
    stopping = session.stopping_step(target)
    before = _base_diagonal(stopping(held), state)
    developed = state * 1.05
    refreshed = session.refresh(developed, held, dual_time=loop)
    after = _base_diagonal(stopping(refreshed), state)

    at_developed = open_session(MaterializedJacobian(_SPLIT), target).build(
        developed, dual_time=loop
    )
    assert jnp.allclose(after, _base_diagonal(at_developed, state), rtol=1e-12)
    assert not jnp.allclose(after, before, rtol=1e-6)


def test_a_stopping_step_is_not_a_step_handed_to_a_march(case) -> None:
    """``on_build`` sees every step a march takes, and not the one the stopping test measures with.

    A driver recording something per step it builds (a residual reporter, say) would otherwise gain a
    record for a step no march runs.
    """
    target, state = case
    built: list[object] = []
    session = open_session(
        MaterializedJacobian(_SPLIT), target, on_build=lambda step: built.append(step) or step
    )
    held = session.build(state, dual_time=DualTimeLoop(inner_steps=3))
    session.stopping_step(target.with_scaled_molecular_viscosity(10.0))(held)
    assert len(built) == 1


def test_a_stopping_step_needs_a_built_session(case) -> None:
    """Its step is built at the state of the session's last build, so there must have been one."""
    target, _ = case
    session = open_session(MaterializedJacobian(_SPLIT), target)
    with pytest.raises(RuntimeError, match="before the session built a step"):
        session.stopping_step(target)(None)


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
    first = session.refresh_preconditioner(active, state)  # the forced first fit
    assert first is not None
    active = first(active)  # as the march does: the build's split is released by the re-fit
    assert session.refresh_preconditioner(active, state) is None  # offered, nothing re-fitted

    r = jax.random.normal(jax.random.PRNGKey(0), (n,))
    # Applied before the re-fit: the re-fit releases the arrays only the superseded split holds.
    before = active.shift_policy.split.apply(r)
    update = session.refresh_preconditioner(active, state * 1.05, due=True)
    assert update is not None
    refitted = update(active)
    host = session.preconditioner.inverse.apply(jnp.asarray(r))
    applied = refitted.shift_policy.split.apply(r)
    assert jnp.allclose(applied, host, rtol=1e-12, atol=1e-14)
    assert not jnp.allclose(applied, before)
    assert jnp.array_equal(refitted.adjoint_preconditioner_factory(state)(r), applied)
    assert jax.tree.structure(eqx.filter(refitted, eqx.is_array)) == jax.tree.structure(
        eqx.filter(active, eqx.is_array)
    )


def _same_arrays(first, second) -> bool:
    """Whether two pytrees hold the very same array objects, leaf for leaf (not merely equal values).

    Identity of the arrays, not of the modules: ``equinox.tree_at`` rebuilds a module it inserts, so
    two steps sharing one split hold equal-but-distinct split objects over the same buffers.
    """
    first, second = jax.tree.leaves(first), jax.tree.leaves(second)
    return len(first) == len(second) and all(a is b for a, b in zip(first, second, strict=True))


def test_every_step_built_from_one_fit_carries_the_one_traced_split_it_made(case) -> None:
    """A traced split is a copy of the whole fitted hierarchy, so the session makes one per fit.

    Every step built from a fit -- a later build, and the stopping step, which is read only for its
    measure -- must carry that one split's arrays rather than a fresh copy, or each would keep its own
    copy alive as long as it is held; on a three-dimensional case each copy costs most of a gigabyte.
    After a re-fit the session's stopping step is re-pointed at the new split as the march's step is,
    so no step the session holds keeps the old one alive.
    """
    target, state = case
    anchor = target.with_scaled_molecular_viscosity(100.0)
    loop = DualTimeLoop(inner_steps=3)
    session, held = _materialized_step_held_on(target, anchor, state, loop)
    split = held.shift_policy.split
    assert _same_arrays(
        session.refresh(state * 1.01, held, dual_time=loop).shift_policy.split, split
    )
    # Built (and cached) after that build, as a march's stopping test builds it, so the re-fit below
    # meets it in the session's cache.
    stopping = session.stopping_step(target)
    assert _same_arrays(stopping(held).shift_policy.split, split)

    update = session.refresh_preconditioner(held, state * 1.05, due=True)
    assert update is not None
    refitted = update(held).shift_policy.split
    assert not _same_arrays(refitted, split)
    # The stopping step lends the held step its own policy and adjoint factory, so these are the
    # stopping step's split, not the held step's.
    judged = stopping(held)
    assert _same_arrays(judged.shift_policy.split, refitted)
    assert _same_arrays(judged.adjoint_preconditioner_factory.split, refitted)


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


def test_a_single_precision_split_reaches_the_step_and_a_host_split_refuses_it(case) -> None:
    """The precision the spec names is the precision the step's split is applied in.

    A split that has to run on the host is applied in double precision there, so asking for single
    precision of it is refused rather than silently ignored.
    """
    coupled, state = case
    single = dataclasses.replace(_SPLIT, precision="float32")
    session = open_session(MaterializedJacobian(single), coupled)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    assert step.shift_policy.split.precision == "float32"
    assert step.shift_policy.split.coupling.values.dtype == jnp.float32
    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(0.5))
    session.refresh_preconditioner(active, state)  # the forced first fit
    refitted = session.refresh_preconditioner(active, state * 1.05, due=True)(active)
    assert refitted.shift_policy.split.precision == "float32"
    assert refitted.shift_policy.split.coupling.values.dtype == jnp.float32

    def host_trailing(role, factory):
        if role == "leading":
            return factory
        return lambda block, n_fields: _HostOnly(factory(block, n_fields))

    session = open_session(MaterializedJacobian(single), coupled, inverse_wrapper=host_trailing)
    with pytest.raises(ValueError, match="no traced cycle, so it is applied on the host"):
        session.build(state, dual_time=DualTimeLoop(inner_steps=3))


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_re_fit_releases_the_arrays_only_the_superseded_split_held(case, precision) -> None:
    """A re-fit frees what only the split it replaced held, and nothing the new split or inverse uses.

    The step a march's caller started the march with keeps the first split referenced for the whole
    march, so without the release that copy of a hierarchy stays resident. What the new split shares
    with the old (a frozen coarsening's index arrays) and what the inverse holds must survive, or the
    march's own step would fail on its next apply.
    """
    coupled, state = case
    spec = dataclasses.replace(_SPLIT, precision=precision)
    session = open_session(MaterializedJacobian(spec), coupled)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(0.5))
    old = session.refresh_preconditioner(active, state)(active)  # the forced first fit
    superseded = jax.tree.leaves(old.shift_policy.split)
    new = session.refresh_preconditioner(old, state * 1.05, due=True)(old)
    current = {id(leaf) for leaf in jax.tree.leaves(new.shift_policy.split)}
    shared = [leaf for leaf in superseded if id(leaf) in current]
    only_old = [leaf for leaf in superseded if id(leaf) not in current]
    assert only_old and all(leaf.is_deleted() for leaf in only_old)
    assert shared and not any(leaf.is_deleted() for leaf in shared)
    r = jax.random.normal(jax.random.PRNGKey(1), (state.shape[0],))
    assert bool(jnp.all(jnp.isfinite(new.shift_policy.split.apply(r))))
    assert bool(jnp.all(jnp.isfinite(session.preconditioner.inverse.apply(r))))
    with pytest.raises(RuntimeError, match="deleted"):
        old.shift_policy.split.apply(r)


def test_a_single_precision_split_applies_the_hierarchies_its_inverses_hold_not_a_copy(
    case,
) -> None:
    """With ``precision="float32"`` each block inverse holds its hierarchy narrow, and the step uses it.

    The split's hierarchies are the inverses' own arrays, so no double-precision hierarchy is held
    beside the single one the step applies -- at the fit and after a re-fit. Only the prolongation
    values, which a refit reads back in double, are narrowed into a copy of their own.
    """
    coupled, state = case
    single = dataclasses.replace(_SPLIT, precision="float32")
    session = open_session(MaterializedJacobian(single), coupled)
    step = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(0.5))
    # The forced first fit, then an asked-for re-fit; each split checked before the next re-fit.
    for at, due in ((state, False), (state * 1.05, True)):
        split = session.refresh_preconditioner(active, at, due=due)(active).shift_policy.split
        inverse = session.preconditioner.inverse
        for applied, held in (
            (split.leading_cycle, inverse.leading.traced_cycle()),
            (split.trailing_cycle, inverse.trailing.traced_cycle()),
        ):
            pairs = list(zip(jax.tree.leaves(applied), jax.tree.leaves(held), strict=True))
            floating = [(a, h) for a, h in pairs if jnp.issubdtype(h.dtype, jnp.floating)]
            narrow = [(a, h) for a, h in floating if h.dtype == jnp.float32]
            assert len(narrow) > len(floating) // 2, "the inverse holds its hierarchy in double"
            assert all(a is h for a, h in narrow), "the step applies a copy of the held hierarchy"


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
