"""Unit tests for the injected pseudo-transient damping schedules.

The schedule sets the shift strength β each step from the residual norms. It is a memoryless pure
function (no mesh, no solve), so these tests are direct-value checks. The load-bearing property beyond
the arithmetic is that :class:`ConstantRelaxation` carries β as a *dynamic* leaf, so a jitted consumer
varies it without retracing.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
from aquaflux.solve import ConstantRelaxation, SwitchedEvolutionRelaxation


def test_ser_matches_the_switched_evolution_formula() -> None:
    """SER is ``max(beta_floor, beta0 (||R||/||R0||)^p)`` — the extracted default, unchanged."""
    ser = SwitchedEvolutionRelaxation()  # defaults beta0=2, exponent=1, beta_floor=0
    assert (ser.beta0, ser.exponent, ser.beta_floor) == (2.0, 1.0, 0.0)
    # β at ||R|| = ||R0|| is beta0; the ramp is linear in the residual ratio at exponent 1.
    assert jnp.allclose(ser.relaxation(jnp.asarray(6.0), jnp.asarray(6.0)), 2.0)
    assert jnp.allclose(ser.relaxation(jnp.asarray(3.0), jnp.asarray(6.0)), 1.0)


def test_ser_exponent_and_floor_are_honoured() -> None:
    quadratic = SwitchedEvolutionRelaxation(beta0=2.0, exponent=2.0)
    assert jnp.allclose(quadratic.relaxation(jnp.asarray(3.0), jnp.asarray(6.0)), 2.0 * 0.25)
    # The floor bounds β below, so a small residual ratio cannot ramp β to zero.
    floored = SwitchedEvolutionRelaxation(beta0=2.0, exponent=1.0, beta_floor=0.3)
    assert jnp.allclose(floored.relaxation(jnp.asarray(1e-6), jnp.asarray(1.0)), 0.3)


def test_constant_relaxation_ignores_the_norms() -> None:
    """ConstantRelaxation holds β at whatever it was set to, regardless of the residual."""
    const = ConstantRelaxation(jnp.asarray(7.0))
    assert jnp.allclose(const.relaxation(jnp.asarray(1.0), jnp.asarray(9.0)), 7.0)
    assert jnp.allclose(const.relaxation(jnp.asarray(1e3), jnp.asarray(1e-3)), 7.0)


def test_constant_relaxation_beta_is_a_dynamic_leaf() -> None:
    """β varies without a recompile — the property an external step control relies on.

    A static field would make each new β a new compilation-cache key; a dynamic 0-d array is a cache
    hit. Pinned with a trace counter: a jitted consumer of the schedule traces once across β values.
    """
    traces = []

    @eqx.filter_jit
    def beta_of(schedule):
        traces.append(1)
        return schedule.relaxation(jnp.asarray(1.0), jnp.asarray(1.0))

    first = beta_of(ConstantRelaxation(jnp.asarray(2.0)))
    second = beta_of(ConstantRelaxation(jnp.asarray(5.0)))  # different β value, same structure
    assert jnp.allclose(first, 2.0) and jnp.allclose(second, 5.0)
    assert len(traces) == 1  # varying β did not retrace


def test_a_schedule_stays_on_the_differentiable_path() -> None:
    """A schedule is a pure function of two scalars, so it differentiates cleanly (unlike a control)."""
    ser = SwitchedEvolutionRelaxation(beta0=2.0, exponent=1.0)
    grad = jax.grad(lambda rn: ser.relaxation(rn, jnp.asarray(4.0)))(jnp.asarray(2.0))
    assert jnp.allclose(grad, 2.0 / 4.0)  # d/d(rn) [beta0 * rn/rn0] = beta0/rn0
