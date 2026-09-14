"""The dual-time loop and the forward-solve regime as values: they resolve and build as the keywords did.

The coupled march's inner loop (``inner_steps``, ``inner_tol``, ``cycle_budget``, ``refresh_on_cycles``)
and its Krylov regime (``forward_solver`` and the ``forward_*`` trio) were loose keywords, several of
them inert without another. The two values that replace them must change nothing numerically, must be
refused beside the keywords they replace, and must merge field by field where a continuation combines
shared options with one point's own.
"""

from __future__ import annotations

import dataclasses

import aquaflux  # noqa: F401  (enables x64)
import pytest
from aquaflux.solve import DualTimeLoop, DualTimeStep, PseudoTransientStep, relative_residual_gmres
from aquaflux.turbulence import (
    BlockDiagonal,
    CompleteLu,
    ForwardSolve,
    MaterializedJacobian,
    coupled_step,
    open_session,
)
from aquaflux.turbulence.coupled import (
    _BLOCK_FORWARD,
    _CONSTRAINED_FORWARD,
    _resolved_march,
    mass_flow_coupled_continuation,
)
from aquaflux.turbulence.march_settings import merged_march_options

from tests.unit.test_coupled_rans import _cavity, _healthy_state

_LOOP_KEYWORDS = ("inner_steps", "inner_tol", "cycle_budget", "refresh_on_cycles")
_TRIO = {
    "rtol": "forward_rtol",
    "restart": "forward_restart",
    "max_restarts": "forward_max_restarts",
}


def _keywords(**given):
    return {
        "dual_time": None,
        "forward": None,
        "forward_solver": None,
        **{name: None for name in _LOOP_KEYWORDS},
        **{name: None for name in _TRIO.values()},
        **given,
    }


def test_the_loop_value_names_exactly_its_dual_time_step_fields() -> None:
    fields = {field.name for field in dataclasses.fields(DualTimeLoop)}
    assert fields == set(_LOOP_KEYWORDS)
    assert fields <= {field.name for field in dataclasses.fields(DualTimeStep)}


def test_the_forward_value_names_exactly_the_regime_s_fields() -> None:
    assert {field.name for field in dataclasses.fields(ForwardSolve)} == set(_TRIO)
    assert set(_BLOCK_FORWARD._fields) == set(_TRIO)


def test_a_loop_of_fewer_than_two_inner_steps_is_refused_and_points_at_the_single_step() -> None:
    with pytest.raises(ValueError, match="dual_time=None"):
        DualTimeLoop(inner_steps=1)


@pytest.mark.parametrize(
    "loop",
    [
        {"inner_steps": 3},
        {"inner_steps": 3, "inner_tol": 1e-3, "cycle_budget": 40, "refresh_on_cycles": 3},
    ],
    ids=["steps", "every-field"],
)
def test_a_loop_value_resolves_to_the_loop_its_keywords_resolved_to(loop) -> None:
    _, _, from_keywords = _resolved_march(_keywords(**loop), _BLOCK_FORWARD)
    _, _, from_value = _resolved_march(_keywords(dual_time=DualTimeLoop(**loop)), _BLOCK_FORWARD)
    assert from_keywords == from_value == DualTimeLoop(**loop)


@pytest.mark.parametrize("inner_steps", [None, 1])
def test_one_inner_step_or_none_is_the_single_shifted_step_as_before(inner_steps) -> None:
    _, _, loop = _resolved_march(
        _keywords(inner_steps=inner_steps, cycle_budget=40), _BLOCK_FORWARD
    )
    assert loop is None


@pytest.mark.parametrize(
    "base", [_BLOCK_FORWARD, _CONSTRAINED_FORWARD], ids=["block", "constrained"]
)
@pytest.mark.parametrize(
    "trio", [{}, {"restart": 15}, {"rtol": 0.1, "restart": 30, "max_restarts": 9}], ids=str
)
def test_a_forward_value_resolves_to_the_regime_its_keywords_resolved_to(base, trio) -> None:
    from_keywords = _resolved_march(
        _keywords(**{_TRIO[name]: value for name, value in trio.items()}), base
    )
    from_value = _resolved_march(_keywords(forward=ForwardSolve(**trio)), base)
    assert from_keywords[:2] == from_value[:2]
    assert from_value[0] == base._replace(**trio)


def test_a_solver_given_as_the_forward_value_replaces_the_regime_as_forward_solver_did() -> None:
    solver = relative_residual_gmres(1e-4)
    from_keywords = _resolved_march(_keywords(forward_solver=solver), _BLOCK_FORWARD)
    from_value = _resolved_march(_keywords(forward=solver), _BLOCK_FORWARD)
    assert from_keywords[0] == from_value[0] == _BLOCK_FORWARD
    assert from_keywords[1] is from_value[1] is solver


def test_resolving_consumes_exactly_the_loop_and_forward_names() -> None:
    keywords = {**_keywords(), "globalization": object()}
    _resolved_march(keywords, _BLOCK_FORWARD)
    assert set(keywords) == {"globalization"}


@pytest.mark.parametrize(
    ("value", "keyword"),
    [
        ({"dual_time": DualTimeLoop(inner_steps=3)}, {"inner_tol": 1e-3}),
        ({"forward": ForwardSolve(restart=15)}, {"forward_rtol": 0.1}),
        ({"forward": relative_residual_gmres(1e-4)}, {"forward_max_restarts": 9}),
    ],
    ids=["loop", "regime", "solver"],
)
def test_a_value_beside_a_keyword_it_replaces_is_refused(value, keyword) -> None:
    (name,) = keyword
    with pytest.raises(TypeError, match=name):
        _resolved_march(_keywords(**value, **keyword), _BLOCK_FORWARD)


def _loop_fields(step) -> tuple:
    return tuple(getattr(step, name) for name in _LOOP_KEYWORDS)


def test_a_block_step_built_from_the_values_matches_the_keywords() -> None:
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    spec = BlockDiagonal(method=None)
    loose = coupled_step(
        coupled, state, preconditioner=spec, inner_steps=3, inner_tol=1e-3, forward_restart=30
    )
    valued = coupled_step(
        coupled,
        state,
        preconditioner=spec,
        dual_time=DualTimeLoop(inner_steps=3, inner_tol=1e-3),
        forward=ForwardSolve(restart=30),
    )
    assert type(loose) is type(valued) is DualTimeStep
    assert _loop_fields(loose) == _loop_fields(valued)
    single = coupled_step(coupled, state, preconditioner=spec)
    assert type(single) is PseudoTransientStep


def test_a_materialized_session_wires_its_refresh_from_either_form() -> None:
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    session = open_session(MaterializedJacobian(CompleteLu(backend="scipy")), coupled)
    loose = session.build(state, inner_steps=3, refresh_on_cycles=3)
    valued = session.build(state, dual_time=DualTimeLoop(inner_steps=3, refresh_on_cycles=3))
    assert _loop_fields(loose) == _loop_fields(valued)
    assert loose.inner_refresh is not None
    assert loose.inner_refresh is valued.inner_refresh


def test_the_mass_flow_builder_resolves_the_values_as_its_keywords() -> None:
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    spec = BlockDiagonal(method=None)
    loose = mass_flow_coupled_continuation(coupled, state, preconditioner=spec, inner_steps=3)
    valued = mass_flow_coupled_continuation(
        coupled, state, preconditioner=spec, dual_time=DualTimeLoop(inner_steps=3)
    )
    assert type(loose) is type(valued) is DualTimeStep
    assert _loop_fields(loose) == _loop_fields(valued)


def test_a_point_s_loop_and_forward_values_merge_field_by_field_over_the_shared_ones() -> None:
    base = {
        "dual_time": DualTimeLoop(inner_steps=5, cycle_budget=40),
        "forward": ForwardSolve(restart=15),
    }
    override = {"dual_time": DualTimeLoop(cycle_budget=20), "forward": ForwardSolve(rtol=0.1)}
    merged = merged_march_options(base, override)
    assert merged["dual_time"] == DualTimeLoop(inner_steps=5, cycle_budget=20)
    assert merged["forward"] == ForwardSolve(rtol=0.1, restart=15)
