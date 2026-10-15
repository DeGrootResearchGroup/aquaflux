"""Unit tests for the adaptive inexact-Newton tolerance a march may solve its steps to.

The rule (`AdaptiveLinearTolerance.next_tolerance`) is a pure function of how the step just taken went,
so it is pinned on hand-computed numbers with no solve. The march's side is pinned on two small
nonlinear systems solved by a residual-stop GMRES: that the tolerance each report carries is the one the
rule chose from the step before it, that a march whose steps are limited by their linear solve converges
in fewer steps with it, and that varying the tolerance from step to step does not recompile the step.
"""

from __future__ import annotations

import math

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from aquaflux.solve import (
    AdaptiveLinearTolerance,
    DampedNewtonStep,
    PseudoTransientStep,
    ShiftTerm,
    SwitchedEvolutionRelaxation,
    newton_march,
)
from aquaflux.solve.linear import (
    relative_residual_gmres,
    relative_tolerance,
    residual_stop_gmres,
    with_relative_tolerance,
)

_TRACES: list[int] = []

# --- the rule --------------------------------------------------------------------------------------


def _next(rule=None, *, base=0.3, alpha=1.0, reduction=0.5, stopping_norm=1.0, target=1e-12):
    rule = AdaptiveLinearTolerance() if rule is None else rule
    return rule.next_tolerance(
        base, alpha=alpha, reduction=reduction, stopping_norm=stopping_norm, target=target
    )


def test_a_step_the_line_search_cut_back_is_followed_by_a_tighter_solve() -> None:
    assert _next(alpha=0.5, reduction=0.6) == 0.1
    # ...but never a looser one than the base, when the base is already tighter.
    assert _next(base=0.05, alpha=0.5, reduction=0.6) == 0.05


def test_a_step_whose_residual_rose_is_followed_by_a_tighter_solve() -> None:
    assert _next(alpha=1.0, reduction=1.2) == 0.1


def test_the_setback_rule_switched_off_keeps_the_base() -> None:
    rule = AdaptiveLinearTolerance(after_setback=None)
    assert _next(rule, alpha=0.25, reduction=0.6) == 0.3
    assert _next(rule, alpha=1.0, reduction=1.2) == 0.3


def test_a_full_step_limited_by_its_solve_tightens_the_next_superlinearly() -> None:
    """A reduction of 0.27 at a base of 0.3 is within the gate (0.45), so 0.9 * 0.27**2 = 0.06561."""
    assert _next(reduction=0.27) == pytest.approx(0.9 * 0.27**2, rel=1e-15)
    # Just inside the gate still counts; just past it does not.
    assert _next(reduction=0.44) == pytest.approx(0.9 * 0.44**2, rel=1e-15)
    assert _next(reduction=0.46) == 0.3


def test_a_full_step_limited_by_something_else_keeps_the_base() -> None:
    assert _next(reduction=0.7) == 0.3
    assert _next(AdaptiveLinearTolerance(tail_gate=None), reduction=0.27) == 0.3


def test_a_tail_step_need_not_solve_past_what_the_stopping_target_asks() -> None:
    """At a residual of 1e-4 against a target of 1e-5, the floor is 0.5 * 1e-5 / 1e-4 = 0.05.

    The superlinear choice (0.9 * 0.01**2 = 9e-5) is far tighter, so the floor decides.
    """
    assert _next(reduction=0.01, stopping_norm=1e-4, target=1e-5) == pytest.approx(0.05)
    # ...and a floor above the base never loosens the solve beyond it.
    assert _next(reduction=0.01, stopping_norm=1e-6, target=1e-5) == 0.3


def test_the_tightest_tolerance_is_the_minimum() -> None:
    assert _next(reduction=1e-8, stopping_norm=1.0, target=0.0) == 1e-10
    assert _next(AdaptiveLinearTolerance(minimum=1e-3), reduction=1e-8, target=0.0) == 1e-3


@pytest.mark.parametrize(
    ("alpha", "reduction"), [(math.nan, 0.5), (1.0, math.nan), (1.0, math.inf), (-math.inf, 0.5)]
)
def test_a_step_with_no_finite_measurement_keeps_the_base(alpha, reduction) -> None:
    assert _next(alpha=alpha, reduction=reduction) == 0.3


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"after_setback": 1.0}, r"after_setback must lie in \(0, 1\)"),
        ({"after_setback": 0.0}, r"after_setback must lie in \(0, 1\)"),
        ({"tail_gate": 0.0}, r"tail_gate must be positive"),
        ({"tail_scale": -1.0}, r"tail_scale must be positive"),
        ({"target_margin": 0.0}, r"target_margin must lie in \(0, 1\]"),
        ({"target_margin": 1.5}, r"target_margin must lie in \(0, 1\]"),
        ({"minimum": 1.0}, r"minimum must lie in \(0, 1\)"),
    ],
)
def test_construction_refuses_a_setting_outside_its_range(fields, message) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveLinearTolerance(**fields)


# --- the solver's tolerance --------------------------------------------------------------------------


def test_a_relative_residual_solver_carries_its_tolerance_as_a_64_bit_array() -> None:
    for solver in (residual_stop_gmres(0.3), relative_residual_gmres(0.3)):
        assert relative_tolerance(solver) == pytest.approx(0.3)
        changed = with_relative_tolerance(solver, 0.02)
        assert type(changed) is type(solver)
        assert relative_tolerance(changed) == pytest.approx(0.02)
        assert isinstance(changed.atol, jax.Array) and changed.atol.dtype == jnp.float64


def test_a_solver_with_no_single_relative_tolerance_has_none_to_vary() -> None:
    stock = lx.GMRES(rtol=1e-6, atol=1e-6)
    assert relative_tolerance(stock) is None
    with pytest.raises(TypeError, match=r"only a relative-residual GMRES"):
        with_relative_tolerance(stock, 0.1)


# --- the march ---------------------------------------------------------------------------------------


class _UnitShift(eqx.Module):
    """A unit-diagonal pseudo-transient shift with no preconditioner."""

    def shift_term(self, phi, residual=None):
        return ShiftTerm(diagonal=jnp.ones_like(phi), make_preconditioner=lambda _beta: None)


class _CoupledCubic(eqx.Module):
    """``A phi + phi**3 / 2 - b``: a non-symmetric coupling, so a loose GMRES leaves real error behind."""

    a: jnp.ndarray
    b: jnp.ndarray

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        if isinstance(phi, jax.core.Tracer):
            _TRACES.append(1)
        return self.a @ phi + 0.5 * phi**3 - self.b


class _Arctan(eqx.Module):
    """``arctan(phi - c)``, whose Newton step from far out overshoots and is cut back."""

    c: jnp.ndarray

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        return jnp.arctan(phi - self.c)


def _coupled_cubic(n: int) -> _CoupledCubic:
    noise = jax.random.normal(jax.random.PRNGKey(0), (n, n)) / jnp.sqrt(n)
    a = 4.0 * jnp.eye(n) - 1.3 * jnp.eye(n, k=1) - 0.7 * jnp.eye(n, k=-1) + 0.4 * noise
    return _CoupledCubic(a, jnp.linspace(1.0, 3.0, n))


def _loose_step(beta0: float = 0.5, exponent: float = 2.0) -> PseudoTransientStep:
    return PseudoTransientStep(
        _UnitShift(),
        relaxation_schedule=SwitchedEvolutionRelaxation(beta0=beta0, exponent=exponent),
        line_search=8,
        krylov_solver=residual_stop_gmres(0.3, norm=jnp.linalg.norm, restart=40),
    )


def test_a_march_limited_by_its_linear_solve_converges_in_fewer_steps_with_it() -> None:
    """At a fixed tolerance of 0.3 the tail is linear; tightened in the tail it is superlinear.

    Measured on this system: 12 steps at 0.3 throughout, 5 with the rule, both to the same root.
    """
    residual, phi0 = _coupled_cubic(40), jnp.zeros(40)
    common = dict(max_steps=60, rtol=1e-10, atol=1e-12)

    fixed = newton_march(_loose_step(), residual, phi0, **common)
    adaptive = newton_march(
        _loose_step(), residual, phi0, adaptive_tolerance=AdaptiveLinearTolerance(), **common
    )

    assert fixed.converged and adaptive.converged
    assert len(adaptive.reports) <= len(fixed.reports) - 4
    assert jnp.allclose(adaptive.state, fixed.state, atol=1e-9)
    # Without the setting no report claims a tolerance; with it the first runs at the base and a
    # later one far tighter.
    assert all(math.isnan(report.linear_tolerance) for report in fixed.reports)
    tolerances = [report.linear_tolerance for report in adaptive.reports]
    assert tolerances[0] == pytest.approx(0.3)
    assert min(tolerances) < 1e-3


def test_each_report_carries_the_tolerance_the_rule_chose_from_the_step_before_it() -> None:
    """Replaying the rule over the reports reproduces every tolerance the march ran at."""
    residual, phi0 = _coupled_cubic(40), jnp.zeros(40)
    rule = AdaptiveLinearTolerance()
    marched = newton_march(
        _loose_step(), residual, phi0, max_steps=60, rtol=1e-10, atol=1e-12, adaptive_tolerance=rule
    )
    reports = marched.reports
    initial = float(jnp.linalg.norm(residual(phi0)))
    starts = [initial] + [report.residual_norm for report in reports[:-1]]
    for before, after, start in zip(reports[:-1], reports[1:], starts[:-1], strict=True):
        expected = rule.next_tolerance(
            0.3,
            alpha=before.alpha,
            reduction=before.residual_norm / start,
            stopping_norm=before.stopping_norm,
            target=1e-12 + 1e-10 * initial,
        )
        assert after.linear_tolerance == pytest.approx(expected, rel=1e-12)


def test_the_step_after_a_cut_back_step_is_solved_to_the_setback_tolerance() -> None:
    marched = newton_march(
        _loose_step(beta0=1e-8, exponent=1.0),
        _Arctan(jnp.array([0.0, 0.5])),
        jnp.array([3.0, 3.5]),
        max_steps=40,
        rtol=1e-10,
        atol=1e-12,
        adaptive_tolerance=AdaptiveLinearTolerance(after_setback=0.07),
    )
    first, second = marched.reports[:2]
    assert first.alpha < 1.0
    assert second.linear_tolerance == pytest.approx(0.07)


def test_a_tolerance_that_changes_every_step_does_not_recompile_the_step() -> None:
    # A size no other test uses, so the compiled step is not already cached by one of them.
    residual, phi0 = _coupled_cubic(37), jnp.zeros(37)
    rule = AdaptiveLinearTolerance()
    common = dict(rtol=1e-10, atol=1e-12, adaptive_tolerance=rule)

    _TRACES.clear()
    first = newton_march(_loose_step(), residual, phi0, max_steps=1, **common)
    compiled = len(_TRACES)
    assert compiled > 0

    rest = newton_march(_loose_step(), residual, first.state, max_steps=60, **common)
    tolerances = {round(report.linear_tolerance, 12) for report in rest.reports}
    assert len(tolerances) >= 3  # it really did vary the tolerance
    assert len(_TRACES) == compiled  # ...and no step recompiled for it


def test_a_march_whose_solver_has_no_relative_tolerance_refuses_the_setting() -> None:
    with pytest.raises(TypeError, match=r"adaptive_tolerance varies .* which has none to vary"):
        newton_march(
            DampedNewtonStep(),
            _Arctan(jnp.array([0.0])),
            jnp.array([0.5]),
            max_steps=3,
            rtol=1e-10,
            atol=1e-12,
            adaptive_tolerance=AdaptiveLinearTolerance(),
        )


class _GrowingScale(eqx.Module):
    """A residual measure whose scale the march rebuilds at each step's start, steeply growing with it."""

    scale: jnp.ndarray

    def __call__(self, residual: jnp.ndarray) -> jnp.ndarray:
        return self.scale * jnp.linalg.norm(residual)


def _growing_scale(step, state):
    return _GrowingScale((jnp.linalg.norm(state) / 1.8) ** 50)


def test_a_residual_that_rose_in_the_record_is_a_setback_though_it_fell_in_its_own_scales() -> None:
    """The reduction is read report to report, across the measure rebuilt at each step.

    Newton on ``phi**3 - 8`` from 1.8: the second step cuts the residual a hundredfold in its own
    measure, but that measure was rebuilt at a state where its scale is ~300x larger, so the RECORDED
    residual rises about fourfold. The rule treats that rise as a setback (the next step solves to
    ``after_setback``); re-measuring the step's start in its own scales would read a tail step instead
    and tighten far below it. The first reading is the one measured to keep a Reynolds-continuation
    march converging when the second failed it.
    """
    marched = newton_march(
        _loose_step(beta0=1e-8, exponent=1.0),
        lambda phi: phi**3 - 8.0,
        jnp.array([1.8]),
        max_steps=4,
        rtol=1e-10,
        atol=1e-12,
        norm_builder=_growing_scale,
        adaptive_tolerance=AdaptiveLinearTolerance(after_setback=0.07),
    )
    first, second, third = marched.reports[:3]
    assert first.alpha == second.alpha == 1.0
    assert second.residual_norm > first.residual_norm  # the record rose...
    assert third.linear_tolerance == pytest.approx(0.07)  # ...so the next step is a setback step
