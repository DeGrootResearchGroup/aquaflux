"""The preconditioner session and ``coupled_step``: built once, shared, and refusing what they cannot use.

When the session replaced the three preconditioner-specific builders, its steps were first pinned
array-identical to theirs for every family; those comparisons went with the builders. What stays is
what a session adds: one inverse and one set of hooks per session, the probe following the operator,
the driver seams, and the refusals. Nothing here needs ``petsc4py``.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import pytest
from aquaflux.solve import AirReduction, DualTimeLoop, JacobiSmoothed, SimpleSmoothed
from aquaflux.turbulence import (
    BlockDiagonal,
    CompleteLu,
    FieldSplit,
    JacobianProbeSpec,
    MaterializedJacobian,
    coupled_step,
    open_session,
)

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


def test_a_frozen_step_refuses_a_refresh_count_but_a_session_build_wires_one(case) -> None:
    """A refresh count with nothing to fire is refused; as a loose keyword it was accepted and ignored."""
    coupled, state = case
    spec = MaterializedJacobian(CompleteLu(backend="scipy"))
    loop = DualTimeLoop(inner_steps=3, refresh_on_cycles=3)
    with pytest.raises(TypeError, match="refresh_on_cycles"):
        coupled_step(coupled, state, preconditioner=spec, dual_time=loop)
    session = open_session(spec, coupled)
    assert session.build(state, dual_time=DualTimeLoop(inner_steps=3)).inner_refresh is None
    assert session.build(state, dual_time=loop).inner_refresh is not None


def test_every_build_of_a_session_shares_one_inverse_and_one_set_of_hooks(case) -> None:
    """A new object in a static field recompiles the coupled solve, so a session must hand back the same ones."""
    coupled, state = case
    session = open_session(MaterializedJacobian(CompleteLu(backend="scipy")), coupled)
    hook = session.refresh_preconditioner
    first = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    second = session.build(state * 1.01, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    assert first.shift_policy.preconditioner is second.shift_policy.preconditioner
    assert first.inner_refresh is second.inner_refresh
    assert session.refresh_preconditioner is hook


def test_the_session_probe_follows_the_operator_stand_in(case) -> None:
    coupled, state = case
    session = open_session(
        MaterializedJacobian(_SPLIT), coupled, jacobian_production_viscosity=True
    )
    session.build(state, dual_time=DualTimeLoop(inner_steps=3))
    probe = session._probe_for()
    assert probe.production_viscosity_frozen


def test_a_probe_spec_reaches_the_probe(case) -> None:
    coupled, _ = case
    session = open_session(
        MaterializedJacobian(CompleteLu(), probe=JacobianProbeSpec(gradient_sweeps=1)), coupled
    )
    assert session._probe_for().gradient_sweeps == 1


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
        def recorded(step, state):
            calls.append(step)

        return recorded

    session = open_session(
        MaterializedJacobian(CompleteLu()), coupled, precondition_wrapper=wrapper
    )
    session.refresh_preconditioner("step", None)
    assert calls == ["step"]


def test_the_block_family_refuses_settings_it_cannot_use(case) -> None:
    coupled, _ = case
    with pytest.raises(TypeError, match="nothing to act on in the block-diagonal family"):
        open_session(BlockDiagonal(), coupled, observer=print)


def test_block_inverse_settings_are_refused_without_a_field_split(case) -> None:
    coupled, _ = case
    with pytest.raises(TypeError, match="CompleteLu has none"):
        open_session(MaterializedJacobian(CompleteLu()), coupled, reports={"leading": print})


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
    session = open_session(MaterializedJacobian(CompleteLu()), coupled)
    with pytest.raises(ValueError, match="SAME case"):
        session.rebind(other)


def test_a_materialized_build_refuses_a_traced_state(case) -> None:
    coupled, state = case
    session = open_session(MaterializedJacobian(CompleteLu(backend="scipy")), coupled)

    def objective(s):
        session.build(s)
        return jnp.sum(s)

    with pytest.raises(ValueError, match=r"cannot be built under jax\.grad"):
        jax.grad(objective)(state)
