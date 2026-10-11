"""Unit tests for the declared capabilities of Newton strategies, schedules and step controls.

Each optional capability -- a relaxation schedule to replace, a readable shift, an inner loop that can be
cut short, a shift carried across the march's boundaries -- is a ``runtime_checkable`` protocol the march
asks with ``isinstance``. These pin, in both directions, which of the shipped classes declare which: a
class that stopped declaring a capability it has would silently lose a feature, and one claiming a
capability it lacks would be handed calls it cannot answer.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.solve import (
    AbortsInnerLoop,
    BlockScaledNorm,
    CarriesRelaxationSchedule,
    CflResidualDualTimeControl,
    ConstantRelaxation,
    DampedNewtonStep,
    DualTimeControl,
    DualTimeStep,
    PseudoTransientStep,
    ReadableShift,
    ResidualRatioDualTimeControl,
    RetryPolicy,
    ShiftCarryingControl,
    ShiftTerm,
    SwitchedEvolutionRelaxation,
)
from aquaflux.solve.strategy import (
    shift_of,
)


class _Shift(eqx.Module):
    """A uniform identity shift, enough to construct a shifted step."""

    def shift_term(self, phi, residual=None):
        return ShiftTerm(diagonal=jnp.ones_like(phi), make_preconditioner=lambda beta: None)


def _steps():
    return {
        "damped": DampedNewtonStep(),
        "pseudo-transient": PseudoTransientStep(_Shift()),
        "dual-time": DualTimeStep(_Shift()),
    }


@pytest.mark.parametrize(
    ("name", "schedule", "aborts"),
    [("damped", False, False), ("pseudo-transient", True, False), ("dual-time", True, True)],
)
def test_each_step_declares_exactly_the_capabilities_it_has(name, schedule, aborts):
    """A damped-Newton step has no shift and no inner loop; only the dual-time step has an inner loop."""
    step = _steps()[name]
    assert isinstance(step, CarriesRelaxationSchedule) is schedule
    assert isinstance(step, AbortsInnerLoop) is aborts


def test_only_a_constant_schedule_exposes_a_readable_shift():
    """The default switched-evolution schedule computes beta and holds nothing to read."""
    assert isinstance(ConstantRelaxation(jnp.asarray(0.5)), ReadableShift)
    assert not isinstance(SwitchedEvolutionRelaxation(), ReadableShift)


def test_shift_of_reads_the_shift_a_step_will_run_at_and_none_where_there_is_none():
    """No schedule, an unreadable one, and a readable one: ``None``, ``None`` and the beta itself."""
    steps = _steps()
    assert shift_of(steps["damped"]) is None
    assert shift_of(steps["pseudo-transient"]) is None  # the default schedule hides beta
    controlled = eqx.tree_at(
        lambda s: s.relaxation_schedule,
        steps["dual-time"],
        ConstantRelaxation(jnp.asarray(0.25)),
    )
    assert float(shift_of(controlled)) == 0.25


class _NextStepOnly:
    """A step control that reshapes nothing about a shift, so it carries none."""

    def next_step(self, base_step, previous, state):
        return base_step, state


@pytest.mark.parametrize(
    "control",
    [DualTimeControl(), ResidualRatioDualTimeControl(), CflResidualDualTimeControl()],
    ids=lambda c: type(c).__name__,
)
def test_every_shipped_shift_control_carries_its_shift_across_the_marchs_boundaries(control):
    """All three shift controls answer the four boundary hooks, through their shared base."""
    assert isinstance(control, ShiftCarryingControl)


def test_a_control_that_drives_no_shift_is_not_a_shift_carrying_control():
    """``next_step`` alone is a plain step control, so the march tells it about no shift boundary."""
    assert not isinstance(_NextStepOnly(), ShiftCarryingControl)


@pytest.mark.parametrize("name", ["damped", "pseudo-transient", "dual-time"])
def test_with_norm_swaps_only_the_measure_as_a_data_leaf(name):
    """The swap leaves the step's structure alone, which is what keeps the march step a cache hit.

    Checked with a block measure whose scales are arrays: a swap to the same block structure at new
    scales changes leaf values and nothing in the tree definition.
    """
    step = _steps()[name].with_norm(BlockScaledNorm(sizes=(2, 2), scales=(1.0, 1.0)))
    swapped = step.with_norm(BlockScaledNorm(sizes=(2, 2), scales=(3.0, 5.0)))

    assert type(swapped) is type(step)
    assert jax.tree_util.tree_structure(swapped) == jax.tree_util.tree_structure(step)
    residual = jnp.array([3.0, 4.0, 5.0, 12.0])
    assert float(swapped.norm()(residual)) == pytest.approx(
        ((5.0 / 3.0) ** 2 + (13.0 / 5.0) ** 2) ** 0.5
    )
    assert float(step.norm()(residual)) == pytest.approx((5.0**2 + 13.0**2) ** 0.5)  # untouched


def test_the_dual_time_step_takes_inner_abort_thresholds_and_keeps_any_left_unset():
    """``None`` means keep the step's own threshold, so the policy can set one without clearing the other."""
    step = DualTimeStep(_Shift(), abort_above_inner_cycles=5, abort_below_alpha=0.1)
    only_alpha = step.with_inner_abort(above_cycles=None, below_alpha=0.01)
    assert (only_alpha.abort_above_inner_cycles, only_alpha.abort_below_alpha) == (5, 0.01)
    assert step.with_inner_abort(above_cycles=None, below_alpha=None) is step


class _RecordsInnerAbort:
    """An inner-loop strategy that is not a :class:`DualTimeStep`, recording what it is handed."""

    def __init__(self):
        self.handed = None

    def with_inner_abort(self, *, above_cycles, below_alpha):
        self.handed = (above_cycles, below_alpha)
        return self


def test_the_retry_policy_pushes_its_thresholds_into_any_declared_inner_loop():
    """The thresholds reach a strategy through the protocol, not through two field names.

    A strategy with an inner loop under different field names used to be skipped in silence, because the
    policy probed for ``abort_above_inner_cycles`` and ``abort_below_alpha`` by name.
    """
    policy = RetryPolicy(abort_above_cycles=7, on_alpha=0.01)
    strategy = _RecordsInnerAbort()
    assert policy.with_inner_abort(strategy) is strategy
    assert strategy.handed == (7, 0.01)

    plain = DampedNewtonStep()
    assert policy.with_inner_abort(plain) is plain  # no inner loop to cut short
    untouched = _RecordsInnerAbort()
    assert RetryPolicy().with_inner_abort(untouched) is untouched
    assert untouched.handed is None  # no thresholds set, so nothing is pushed
