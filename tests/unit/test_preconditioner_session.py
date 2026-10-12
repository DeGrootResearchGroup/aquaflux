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
from aquaflux.turbulence import BlockDiagonal, open_session

from tests.unit.test_coupled_rans import _cavity, _healthy_state

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
