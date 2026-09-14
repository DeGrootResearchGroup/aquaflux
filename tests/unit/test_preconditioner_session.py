"""The preconditioner session and ``coupled_step``: the same steps as the builders, built once and shared.

Every family is checked against the builder it replaces, on the same case and state, by comparing the
built step's array leaves and one application of its preconditioner. Nothing here needs ``petsc4py``:
the monolithic V-cycle is the one family that does, and its comparison lives with the other
PETSc-gated coupled tests.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import AirReduction, JacobiSmoothed, SimpleSmoothed
from aquaflux.turbulence import (
    BlockDiagonal,
    CompleteLu,
    FieldSplit,
    JacobianProbeSpec,
    MaterializedJacobian,
    coupled_amg_continuation,
    coupled_continuation,
    coupled_lu_continuation,
    coupled_step,
    open_session,
)

from tests.unit.test_coupled_rans import _cavity, _healthy_state

_SPLIT = FieldSplit(SimpleSmoothed(), JacobiSmoothed())


@pytest.fixture(scope="module")
def case():
    mesh, coupled = _cavity()
    return coupled, _healthy_state(mesh, coupled)


def _assert_same_step(built, reference, apply_vector) -> None:
    assert type(built) is type(reference)
    built_leaves = jax.tree_util.tree_leaves(eqx.filter(built, eqx.is_array))
    reference_leaves = jax.tree_util.tree_leaves(eqx.filter(reference, eqx.is_array))
    assert len(built_leaves) == len(reference_leaves)
    for mine, theirs in zip(built_leaves, reference_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    np.testing.assert_array_equal(
        np.asarray(apply_vector(built)), np.asarray(apply_vector(reference))
    )


def _vector(coupled) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(0).normal(size=coupled.layout.size))


def _block_apply(coupled, state):
    v = _vector(coupled)
    return lambda step: step.shift_policy.shift_term(state).make_preconditioner(jnp.asarray(1.0))(v)


def _frozen_apply(coupled):
    v = _vector(coupled)
    return lambda step: step.shift_policy.preconditioner.matvec()(v)


@pytest.mark.parametrize("inner_steps", [1, 3])
def test_the_block_diagonal_step_is_the_block_builders_step(case, inner_steps) -> None:
    coupled, state = case
    _assert_same_step(
        coupled_step(
            coupled,
            state,
            preconditioner=BlockDiagonal(method="air", v_cycles=2),
            inner_steps=inner_steps,
        ),
        coupled_continuation(coupled, state, method="air", v_cycles=2, inner_steps=inner_steps),
        _block_apply(coupled, state),
    )


def test_a_block_session_refresh_is_the_builders_reuse_refresh(case) -> None:
    coupled, state = case
    session = open_session(BlockDiagonal(), coupled)
    first = session.build(state, inner_steps=3)
    reference_first = coupled_continuation(coupled, state, inner_steps=3)
    measure = first.norm()
    moved = state * 1.01
    _assert_same_step(
        session.refresh(moved, first, measure, inner_steps=3),
        coupled_continuation(
            coupled,
            moved,
            inner_steps=3,
            reuse=reference_first.shift_policy,
            residual_norm=measure,
        ),
        _block_apply(coupled, moved),
    )


def test_the_complete_lu_step_is_the_lu_builders_step(case) -> None:
    coupled, state = case
    _assert_same_step(
        coupled_step(
            coupled,
            state,
            preconditioner=MaterializedJacobian(CompleteLu(backend="scipy")),
            inner_steps=3,
        ),
        coupled_lu_continuation(coupled, state, backend="scipy", inner_steps=3),
        _frozen_apply(coupled),
    )


def test_the_field_split_step_is_the_split_builders_step(case) -> None:
    coupled, state = case
    _assert_same_step(
        coupled_step(
            coupled,
            state,
            preconditioner=MaterializedJacobian(_SPLIT, build_beta=1.5),
            inner_steps=3,
            refresh_on_cycles=3,
        ),
        coupled_amg_continuation(
            coupled,
            state,
            amg_beta=1.5,
            field_split=True,
            leading_inverse=SimpleSmoothed(),
            trailing_inverse=JacobiSmoothed(),
            inner_steps=3,
            refresh_on_cycles=3,
        ),
        _frozen_apply(coupled),
    )


def test_a_frozen_step_wires_no_refresh_but_a_session_build_does(case) -> None:
    coupled, state = case
    spec = MaterializedJacobian(CompleteLu(backend="scipy"))
    assert (
        coupled_step(
            coupled, state, preconditioner=spec, inner_steps=3, refresh_on_cycles=3
        ).inner_refresh
        is None
    )
    session = open_session(spec, coupled)
    assert session.build(state, inner_steps=3).inner_refresh is None
    assert session.build(state, inner_steps=3, refresh_on_cycles=3).inner_refresh is not None


def test_every_build_of_a_session_shares_one_inverse_and_one_set_of_hooks(case) -> None:
    """A new object in a static field recompiles the coupled solve, so a session must hand back the same ones."""
    coupled, state = case
    session = open_session(MaterializedJacobian(CompleteLu(backend="scipy")), coupled)
    hook = session.precondition_step
    first = session.build(state, inner_steps=3, refresh_on_cycles=3)
    second = session.build(state * 1.01, inner_steps=3, refresh_on_cycles=3)
    assert first.shift_policy.preconditioner is second.shift_policy.preconditioner
    assert first.inner_refresh is second.inner_refresh
    assert session.precondition_step is hook


def test_the_session_probe_follows_the_operator_stand_in(case) -> None:
    coupled, state = case
    session = open_session(
        MaterializedJacobian(_SPLIT), coupled, jacobian_production_viscosity=True
    )
    session.build(state, inner_steps=3)
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
    session.build(state, inner_steps=3)
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
    session.precondition_step("step", None)
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
