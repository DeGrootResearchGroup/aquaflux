"""Unit: the stopping test as one value -- the measure, its tolerances, and who builds the measure.

A tolerance is meaningless without the measure it is taken in, so :class:`~aquaflux.solve.Convergence`
carries both, and the measure is built against the problem at every outer iteration and handed to the
step. These tests pin each half on stand-ins: which scales a measure takes and when, which problems can
supply which measures, that a root solve applies a measure it is given and keeps the step's own
otherwise, and that every step class stops its linear solve in the measure it is being judged by.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import aquaflux.solve.continuation as continuation_module
import aquaflux.solve.newton as newton_module
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import pytest
from aquaflux.solve import (
    PLAIN_RESIDUAL,
    BlockScaled,
    BlockScaledNorm,
    ConstantRelaxation,
    Convergence,
    DampedNewtonStep,
    DualTimeStep,
    Euclidean,
    PseudoTransientStep,
    ResidualMeasure,
    RootSolver,
    RowScaled,
    ShiftTerm,
    in_progress_measure,
    relative_residual_gmres,
    solve_linear,
)


class _RecordingMeasures:
    """A problem that can supply every measure, recording what each build is asked about."""

    def __init__(self) -> None:
        self.row_scaled_asked: list[tuple[object, jnp.ndarray]] = []
        self.block_scaled_asked: list[jnp.ndarray] = []

    def row_scaled(self, step, state):
        self.row_scaled_asked.append((step, state))
        return BlockScaledNorm((state.shape[0],), (float(jnp.sum(jnp.abs(state))),))

    def block_scaled(self, state):
        self.block_scaled_asked.append(state)
        return BlockScaledNorm((state.shape[0],), (float(jnp.sum(jnp.abs(state))),))


# --- the value -----------------------------------------------------------------------------------


def test_a_convergence_refuses_anything_but_a_measure_value() -> None:
    with pytest.raises(TypeError, match="residual-measure value"):
        Convergence(measure="row_scaled")


def test_the_measure_family_is_abstract() -> None:
    with pytest.raises(TypeError, match="ResidualMeasure is abstract"):
        ResidualMeasure()


def test_an_unset_field_takes_the_base_s() -> None:
    base = Convergence(measure=RowScaled(), rtol=1e-10, atol=1e-12)
    assert Convergence(atol=1e-5).filled_from(base) == Convergence(RowScaled(), 1e-10, 1e-5)


# --- when each measure takes its scales ----------------------------------------------------------


def test_a_row_scaled_measure_is_built_at_every_state_from_the_step_it_is_handed() -> None:
    measures = _RecordingMeasures()
    builder = RowScaled()._builder(measures, jnp.array([1.0, 1.0]))
    step, later = object(), jnp.array([3.0, 4.0])

    norm = builder(step, later)

    assert measures.row_scaled_asked == [(step, later)]
    assert norm.scales == (7.0,)


def test_a_block_scaled_measure_takes_the_initial_state_s_scales_once_and_holds_them() -> None:
    measures = _RecordingMeasures()
    initial = jnp.array([1.0, 1.0])
    builder = BlockScaled()._builder(measures, initial)

    held = {id(builder(object(), jnp.array([30.0, 40.0]))) for _ in range(3)}

    assert len(measures.block_scaled_asked) == 1
    assert jnp.array_equal(measures.block_scaled_asked[0], initial)
    assert len(held) == 1


def test_a_euclidean_measure_asks_the_problem_for_nothing() -> None:
    builder = Euclidean()._builder(PLAIN_RESIDUAL, jnp.array([1.0]))
    assert builder(object(), jnp.array([2.0])) is jnp.linalg.norm


def test_a_plain_residual_refuses_the_scaled_measures_by_name() -> None:
    with pytest.raises(TypeError, match=r"RowScaled\(\).*Euclidean\(\)"):
        RowScaled()._builder(PLAIN_RESIDUAL, jnp.array([1.0]))(object(), jnp.array([1.0]))
    with pytest.raises(TypeError, match=r"BlockScaled\(\).*Euclidean\(\)"):
        BlockScaled()._builder(PLAIN_RESIDUAL, jnp.array([1.0]))


# --- a root solve applies the measure it is given ------------------------------------------------


def _cube_root_residual(phi, theta):
    return phi**3 - theta


class _Amplified(eqx.Module):
    """A step measure a million times the Euclidean norm, so an absolute bar in it is far tighter."""

    def __call__(self, residual):
        return 1e6 * jnp.linalg.norm(residual)


def _residual_at_root(convergence: Convergence) -> float:
    solver = RootSolver(
        convergence=convergence, strategy=DampedNewtonStep(residual_norm=_Amplified())
    )
    root = solver.solve(_cube_root_residual, jnp.array([1.0]), jnp.array(8.0))
    return float(jnp.linalg.norm(_cube_root_residual(root, 8.0)))


def test_a_root_solve_with_no_measure_stops_in_the_step_s_own() -> None:
    # 1e-2 in the amplified measure is 1e-8 of residual, which Newton reaches a step or two later.
    assert _residual_at_root(Convergence(rtol=0.0, atol=1e-2)) < 1e-8


def test_a_root_solve_given_a_measure_stops_in_that_one_instead() -> None:
    # The same bar in the Euclidean norm stops as soon as the residual is under 1e-2.
    assert _residual_at_root(Convergence(measure=Euclidean(), rtol=0.0, atol=1e-2)) > 1e-8


# --- every step stops its linear solve in its own measure ---------------------------------------


class _UnitShift(eqx.Module):
    def shift_term(self, phi, residual=None):
        return ShiftTerm(jnp.ones_like(phi), lambda relaxation: None)


class _Cube(eqx.Module):
    def __call__(self, phi):
        return phi**3 - 8.0


_MEASURE = BlockScaledNorm((1, 1), (2.0, 3.0))


@pytest.mark.parametrize(
    "step",
    [
        PseudoTransientStep(
            _UnitShift(), relaxation_schedule=ConstantRelaxation(beta=1.0), residual_norm=_MEASURE
        ),
        DualTimeStep(
            _UnitShift(),
            relaxation_schedule=ConstantRelaxation(beta=1.0),
            residual_norm=_MEASURE,
            inner_steps=2,
        ),
        DampedNewtonStep(residual_norm=_MEASURE),
    ],
    ids=["pseudo-transient", "dual-time", "damped-newton"],
)
def test_each_step_stops_a_measure_following_linear_solve_in_its_own_measure(
    monkeypatch, step
) -> None:
    """A march rebuilds the measure and swaps it into the step, so the step must read it per solve.

    Otherwise the linear solve keeps stopping in whichever measure was current when it was configured,
    while the line search and the convergence test use the rebuilt one.
    """
    handed: list[lx.AbstractLinearSolver] = []

    def recording(*args, **kwargs):
        handed.append(kwargs["solver"])
        return solve_linear(*args, **kwargs)

    monkeypatch.setattr(continuation_module, "solve_linear", recording)
    monkeypatch.setattr(newton_module, "solve_linear", recording)

    step.stepper()(
        _Cube(), jnp.array([1.0, 1.5]), jnp.array(1.0), relative_residual_gmres(0.1, norm=None)
    )

    assert handed
    assert all(solver.norm == _MEASURE for solver in handed)


def test_a_solver_with_its_own_measure_is_left_alone() -> None:
    explicit = relative_residual_gmres(0.1)
    stock = lx.GMRES(rtol=1e-6, atol=1e-6)
    assert in_progress_measure(explicit, _MEASURE) is explicit
    assert in_progress_measure(stock, _MEASURE) is stock


def test_a_measure_following_solver_refuses_to_run_outside_a_step() -> None:
    with pytest.raises(TypeError, match="outside one"):
        solve_linear(lambda x: 2.0 * x, jnp.ones(2), relative_residual_gmres(0.1, norm=None))
