"""The dual-time loop and the forward-solve regime as values, and the combinations they refuse.

The coupled march's inner loop (``inner_steps``, ``inner_tol``, ``cycle_budget``, ``refresh_on_cycles``)
and its Krylov regime (``krylov_solver`` and a ``forward_*`` trio) were loose keywords, several of them
inert without another: loop settings on a single shifted step, a regime beside an explicit solver, a
refresh count with nothing to fire. Each was accepted and reached nothing. As values the first two
cannot be written at all, and the third is refused where the march knows whether a refresh exists.
"""

from __future__ import annotations

import dataclasses
import inspect

import aquaflux  # noqa: F401  (enables x64)
import pytest
from aquaflux.solve import (
    DualTimeLoop,
    DualTimeStep,
    LinearSolveSettings,
    PseudoTransientStep,
    relative_residual_gmres,
    resolve_linear_solve,
)
from aquaflux.turbulence import (
    BlockDiagonal,
    CompleteLu,
    MaterializedJacobian,
    UnpreconditionedScalars,
    coupled_step,
    open_session,
)
from aquaflux.turbulence.coupled import (
    _BLOCK_LINEAR_SOLVE,
    _CONSTRAINED_LINEAR_SOLVE,
    mass_flow_coupled_continuation,
)
from aquaflux.turbulence.march_settings import merged_march_options

from tests.unit.test_coupled_rans import _cavity, _healthy_state

_LOOP_FIELDS = ("inner_steps", "inner_tol", "cycle_budget", "refresh_on_cycles")
#: The loose keywords the two values replaced -- none may reappear on a builder beside them.
_RETIRED = (
    *_LOOP_FIELDS,
    "krylov_solver",
    "forward_rtol",
    "forward_restart",
    "forward_max_restarts",
)


@pytest.fixture(scope="module")
def case():
    mesh, coupled = _cavity(4)
    return coupled, _healthy_state(mesh, coupled)


def test_every_coupled_builder_takes_the_values_and_none_of_the_keywords_they_replaced() -> None:
    for builder in (coupled_step, mass_flow_coupled_continuation):
        parameters = set(inspect.signature(builder).parameters)
        assert {"dual_time", "linear_solve"} <= parameters, builder.__name__
        assert not parameters & set(_RETIRED), builder.__name__


def test_the_loop_value_names_exactly_its_dual_time_step_fields() -> None:
    fields = {field.name for field in dataclasses.fields(DualTimeLoop)}
    assert fields == set(_LOOP_FIELDS)
    assert fields <= {field.name for field in dataclasses.fields(DualTimeStep)}


def test_the_forward_value_names_exactly_the_regime_s_fields() -> None:
    assert {field.name for field in dataclasses.fields(LinearSolveSettings)} == set(
        _BLOCK_LINEAR_SOLVE._fields
    )


def test_a_loop_of_fewer_than_two_inner_steps_is_refused_and_points_at_the_single_step() -> None:
    with pytest.raises(ValueError, match="dual_time=None"):
        DualTimeLoop(inner_steps=1)


@pytest.mark.parametrize(
    "base", [_BLOCK_LINEAR_SOLVE, _CONSTRAINED_LINEAR_SOLVE], ids=["block", "constrained"]
)
@pytest.mark.parametrize(
    "fields", [{}, {"restart": 15}, {"rtol": 0.1, "restart": 30, "max_restarts": 9}], ids=str
)
def test_a_forward_value_resolves_each_unset_field_to_the_family_s_regime(base, fields) -> None:
    regime, solver = resolve_linear_solve(LinearSolveSettings(**fields), base)
    assert regime == base._replace(**fields)
    assert solver is None
    assert resolve_linear_solve(None, base) == (base, None)


def test_a_solver_given_as_the_forward_value_replaces_the_regime() -> None:
    solver = relative_residual_gmres(1e-4)
    assert resolve_linear_solve(solver, _BLOCK_LINEAR_SOLVE) == (_BLOCK_LINEAR_SOLVE, solver)


def test_the_loop_selects_the_step_shape_and_reaches_its_fields(case) -> None:
    coupled, state = case
    spec = BlockDiagonal(scalar=UnpreconditionedScalars())
    loop = DualTimeLoop(inner_steps=3, inner_tol=1e-3, cycle_budget=40)
    dual = coupled_step(
        coupled,
        state,
        preconditioner=spec,
        dual_time=loop,
        linear_solve=LinearSolveSettings(restart=30),
    )
    assert type(dual) is DualTimeStep
    assert (dual.inner_steps, dual.inner_tol, dual.cycle_budget) == (3, 1e-3, 40)
    assert dual.krylov_solver.restart == 30
    assert type(coupled_step(coupled, state, preconditioner=spec)) is PseudoTransientStep
    mass_flow = mass_flow_coupled_continuation(coupled, state, preconditioner=spec, dual_time=loop)
    assert type(mass_flow) is DualTimeStep
    assert mass_flow.inner_steps == 3


@pytest.mark.parametrize("hook", ["inner_observer", "inner_refresh"])
def test_a_loop_hook_without_a_loop_is_refused(case, hook) -> None:
    coupled, state = case
    with pytest.raises(TypeError, match=hook):
        coupled_step(
            coupled,
            state,
            preconditioner=BlockDiagonal(scalar=UnpreconditionedScalars()),
            **{hook: lambda *a: None},
        )


def test_a_refresh_count_with_nothing_to_fire_is_refused_but_a_materialized_session_fires_it(
    case,
) -> None:
    coupled, state = case
    loop = DualTimeLoop(inner_steps=3, refresh_on_cycles=3)
    with pytest.raises(
        TypeError, match="refresh_on_cycles"
    ):  # a block-diagonal session has no refresh
        open_session(BlockDiagonal(scalar=UnpreconditionedScalars()), coupled).build(
            state, dual_time=loop
        )
    spec = MaterializedJacobian(CompleteLu(backend="scipy"))
    with pytest.raises(TypeError, match="refresh_on_cycles"):  # nor does a frozen step
        coupled_step(coupled, state, preconditioner=spec, dual_time=loop)
    session = open_session(spec, coupled)
    first, second = session.build(state, dual_time=loop), session.build(state, dual_time=loop)
    assert first.inner_refresh is not None
    assert first.inner_refresh is second.inner_refresh
    # ...and a caller's own refresh is enough on its own.
    coupled_step(
        coupled,
        state,
        preconditioner=BlockDiagonal(scalar=UnpreconditionedScalars()),
        dual_time=loop,
        inner_refresh=lambda iterate: None,
    )


def test_a_point_s_loop_and_forward_values_merge_field_by_field_over_the_shared_ones() -> None:
    base = {
        "dual_time": DualTimeLoop(inner_steps=5, cycle_budget=40),
        "linear_solve": LinearSolveSettings(restart=15),
    }
    override = {
        "dual_time": DualTimeLoop(cycle_budget=20),
        "linear_solve": LinearSolveSettings(rtol=0.1),
    }
    merged = merged_march_options(base, override)
    assert merged["dual_time"] == DualTimeLoop(inner_steps=5, cycle_budget=20)
    assert merged["linear_solve"] == LinearSolveSettings(rtol=0.1, restart=15)


def test_a_point_s_globalization_merges_field_by_field_too() -> None:
    """The oldest march value merges like the three newer ones, rather than being replaced whole."""
    from aquaflux.solve import Globalization

    merged = merged_march_options(
        {"globalization": Globalization(beta0=2.0, line_search=10)},
        {"globalization": Globalization(line_search=3)},
    )["globalization"]
    assert (merged.beta0, merged.line_search) == (2.0, 3)


def test_an_unset_field_takes_the_shared_setting_rather_than_the_family_default() -> None:
    """The documented limit of the merge: ``None`` cannot ask for the default back over a shared setting."""
    merged = merged_march_options(
        {"linear_solve": LinearSolveSettings(rtol=1e-2)},
        {"linear_solve": LinearSolveSettings(restart=30)},
    )["linear_solve"]
    assert merged == LinearSolveSettings(rtol=1e-2, restart=30)
