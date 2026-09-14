"""The pseudo-time shift settings as one value: they resolve and build exactly as the keywords did.

The three shift settings -- the basis, the velocity parts and the closure's damping ratio -- were loose
keywords on every coupled builder. The value that groups them must change nothing numerically, must
refuse being given beside the keywords it replaces, and must merge field by field where a continuation
combines shared options with one point's own, which is how the same settings as separate keywords
combine.
"""

from __future__ import annotations

import dataclasses
import inspect

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import LocalCourantBasis
from aquaflux.turbulence import (
    BlockDiagonal,
    CompleteLu,
    ConstantDamping,
    MaterializedJacobian,
    ShiftSettings,
    coupled_step,
    open_session,
    solve_reynolds_continuation,
)
from aquaflux.turbulence.coupled import (
    _DEFAULT_SHIFT_BASIS,
    _resolved_shift,
    mass_flow_coupled_continuation,
)
from aquaflux.turbulence.march_settings import merged_march_options

from tests.unit.test_coupled_rans import _cavity, _healthy_state
from tests.unit.test_reynolds_continuation import _record_solves, _tiny_coupled

#: Each field of the value beside the keyword it replaces.
_KEYWORD_OF = {
    "basis": "shift_basis",
    "velocity_parts": "velocity_shift_parts",
    "turbulence_damping": "turbulence_damping",
}
_BASIS = LocalCourantBasis(dissipative_weight=0.0)


def _loose(**given):
    return {"shift": None, **{name: None for name in _KEYWORD_OF.values()}, **given}


def _valued(**fields):
    return {"shift": ShiftSettings(**fields), **{name: None for name in _KEYWORD_OF.values()}}


def test_the_value_names_exactly_the_shift_keywords_of_every_coupled_builder() -> None:
    assert {field.name for field in dataclasses.fields(ShiftSettings)} == set(_KEYWORD_OF)
    for builder in (coupled_step, mass_flow_coupled_continuation):
        parameters = inspect.signature(builder).parameters
        assert {"shift", *_KEYWORD_OF.values()} <= set(parameters), builder.__name__


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"basis": _BASIS},
        {"turbulence_damping": 2.0},
        {"turbulence_damping": ConstantDamping(3.0)},
        {"basis": _BASIS, "turbulence_damping": 2.0},
    ],
    ids=["unset", "basis", "float-damping", "strategy-damping", "basis-and-damping"],
)
def test_a_value_resolves_to_the_very_objects_its_keywords_resolved_to(fields) -> None:
    from_keywords = _resolved_shift(_loose(**{_KEYWORD_OF[name]: v for name, v in fields.items()}))
    from_value = _resolved_shift(_valued(**fields))
    for a, b in zip(from_keywords, from_value, strict=True):
        assert a is b
    assert from_value[0] is fields.get("basis", _DEFAULT_SHIFT_BASIS)


def test_resolving_consumes_exactly_the_four_shift_names() -> None:
    keywords = {**_loose(), "inner_steps": 3}
    _resolved_shift(keywords)
    assert keywords == {"inner_steps": 3}


def _assert_same_shift(policy_a, policy_b, state) -> None:
    term_a, term_b = policy_a.shift_term(state), policy_b.shift_term(state)
    np.testing.assert_array_equal(np.asarray(term_a.diagonal), np.asarray(term_b.diagonal))
    relaxation = jnp.asarray(0.5)
    assert (term_a.row_relaxation is None) == (term_b.row_relaxation is None)
    if term_a.row_relaxation is not None:
        np.testing.assert_array_equal(
            np.asarray(term_a.row_relaxation(relaxation)),
            np.asarray(term_b.row_relaxation(relaxation)),
        )
    v = jnp.asarray(np.random.default_rng(0).standard_normal(state.shape))
    np.testing.assert_array_equal(
        np.asarray(term_a.make_preconditioner(relaxation)(v)),
        np.asarray(term_b.make_preconditioner(relaxation)(v)),
    )


def test_a_block_diagonal_step_built_from_the_value_matches_the_keywords() -> None:
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    spec = BlockDiagonal(method=None)
    loose = coupled_step(
        coupled, state, preconditioner=spec, shift_basis=_BASIS, turbulence_damping=2.0
    )
    valued = coupled_step(
        coupled,
        state,
        preconditioner=spec,
        shift=ShiftSettings(basis=_BASIS, turbulence_damping=2.0),
    )
    _assert_same_shift(loose.shift_policy, valued.shift_policy, state)


def test_a_materialized_session_builds_the_same_shift_from_the_value() -> None:
    """Both forms from ONE session, so the fitted inverse is the same object and only the shift varies."""
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    session = open_session(MaterializedJacobian(CompleteLu(backend="scipy")), coupled)
    loose = session.build(state, shift_basis=_BASIS, turbulence_damping=2.0)
    valued = session.build(state, shift=ShiftSettings(basis=_BASIS, turbulence_damping=2.0))
    assert loose.shift_policy.preconditioner is valued.shift_policy.preconditioner
    base_a, base_b = loose.shift_policy.base, valued.shift_policy.base
    np.testing.assert_array_equal(
        np.asarray(base_a.shift_term(state).diagonal), np.asarray(base_b.shift_term(state).diagonal)
    )
    relaxation = jnp.asarray(0.5)
    np.testing.assert_array_equal(
        np.asarray(base_a.shift_term(state).row_relaxation(relaxation)),
        np.asarray(base_b.shift_term(state).row_relaxation(relaxation)),
    )


def test_the_mass_flow_builder_builds_the_same_shift_from_the_value() -> None:
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    spec = BlockDiagonal(method=None)
    loose = mass_flow_coupled_continuation(
        coupled, state, preconditioner=spec, shift_basis=_BASIS, turbulence_damping=2.0
    )
    valued = mass_flow_coupled_continuation(
        coupled,
        state,
        preconditioner=spec,
        shift=ShiftSettings(basis=_BASIS, turbulence_damping=2.0),
    )
    augmented = jnp.append(state, 0.1)
    _assert_same_shift(loose.shift_policy, valued.shift_policy, augmented)


@pytest.mark.parametrize("keyword", list(_KEYWORD_OF.values()))
def test_the_value_beside_a_keyword_it_replaces_is_refused(keyword) -> None:
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    value = {"shift_basis": _BASIS, "velocity_shift_parts": object(), "turbulence_damping": 2.0}
    with pytest.raises(TypeError, match=keyword):
        coupled_step(
            coupled,
            state,
            preconditioner=BlockDiagonal(method=None),
            shift=ShiftSettings(),
            **{keyword: value[keyword]},
        )


def test_a_point_s_shift_value_is_merged_field_by_field_over_the_shared_one() -> None:
    base = {"shift": ShiftSettings(basis=_BASIS, turbulence_damping=3.0), "rtol": 1e-6}
    override = {"shift": ShiftSettings(turbulence_damping=2.0), "rtol": 1e-8}
    merged = merged_march_options(base, override)
    assert merged["shift"].basis is _BASIS  # kept from the shared value
    assert merged["shift"].turbulence_damping == 2.0  # the point's own wins where it sets one
    assert merged["rtol"] == 1e-8  # a loose keyword is a plain override
    assert merged_march_options(base, {})["shift"] is base["shift"]


def test_the_reynolds_continuation_merges_a_point_s_shift_over_the_shared_one(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(
        _tiny_coupled(),
        n_points=0,
        shift=ShiftSettings(basis=_BASIS),
        point_setup=lambda companion, state, point: {
            "shift": ShiftSettings(turbulence_damping=2.0)
        },
    )
    (call,) = calls
    assert call["kwargs"]["shift"].basis is _BASIS
    assert call["kwargs"]["shift"].turbulence_damping == 2.0
