"""Unit tests for the BDF transient (accumulation) term."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.discretization import TransientTerm

VOLUME = jnp.array([2.0, 4.0])
PHI = jnp.array([1.0, 2.0])
PHI_OLD = jnp.array([0.5, 1.0])
PHI_OLDER = jnp.array([0.0, 0.5])
DT = 0.1


def test_bdf1_first_step() -> None:
    """First step uses backward Euler: V (phi - phi_old) / dt."""
    r = TransientTerm().residual(PHI, PHI_OLD, PHI_OLDER, DT, first_step=True, volume=VOLUME)
    expected = VOLUME * (PHI - PHI_OLD) / DT
    assert jnp.allclose(r, expected)


def test_bdf2_later_steps() -> None:
    """Later steps use second-order backward Euler: V (3/2 phi - 2 phi_old + 1/2 phi_older)/dt."""
    r = TransientTerm().residual(PHI, PHI_OLD, PHI_OLDER, DT, first_step=False, volume=VOLUME)
    expected = VOLUME * (1.5 * PHI - 2.0 * PHI_OLD + 0.5 * PHI_OLDER) / DT
    assert jnp.allclose(r, expected)


def test_bdf1_ignores_phi_older() -> None:
    """The first-step branch must not depend on phi_older (which is undefined at step one)."""
    r1 = TransientTerm().residual(PHI, PHI_OLD, PHI_OLDER, DT, first_step=True, volume=VOLUME)
    r2 = TransientTerm().residual(
        PHI, PHI_OLD, PHI_OLDER * 99.0, DT, first_step=True, volume=VOLUME
    )
    assert jnp.allclose(r1, r2)


def test_steady_state_vanishes() -> None:
    """A field that has stopped changing gives zero accumulation."""
    steady = jnp.array([3.0, 3.0])
    r = TransientTerm().residual(steady, steady, steady, DT, first_step=False, volume=VOLUME)
    assert jnp.allclose(r, 0.0)


# --- order of accuracy on an analytic field -----------------------------------------------------
#
# The tests above restate the two stencils' coefficients, which catches a typo but cannot catch a
# scheme that is consistent and simply less accurate than it claims: replacing the BDF2 branch with
# the BDF1 one is a legitimate discretization of the same derivative, and only an accuracy
# measurement tells the two apart. These are that measurement -- the operator-level check every
# numerical operator here carries, done on a function whose derivative is known in closed form.


def _stencil_derivative(dt: float, *, first_step: bool) -> float:
    """The stencil's estimate of ``d/dt exp(t)`` at ``t = 0``, where the true value is ``1``.

    ``volume = 1``, so the accumulation term *is* the derivative estimate and the measurement is of
    the time discretization alone, with no cell geometry in it.
    """
    one = jnp.ones(1)
    residual = TransientTerm().residual(
        one,
        jnp.exp(-dt) * one,
        jnp.exp(-2.0 * dt) * one,
        dt,
        first_step=first_step,
        volume=one,
    )
    return float(residual[0])


def _observed_order(error_of) -> float:
    """The slope of log(error) against log(dt) over a halving sequence, as its smallest local rate.

    The smallest rate rather than a fitted average: an operator that is second order over part of the
    range and first order over the rest should fail, and an average would hide that.
    """
    steps = [0.1, 0.05, 0.025, 0.0125]
    errors = [error_of(dt) for dt in steps]
    assert all(e > 0.0 for e in errors), f"an exactly-zero error cannot show an order: {errors}"
    return min(float(jnp.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1))


def test_bdf1_is_first_order_accurate_in_time() -> None:
    """The first-step branch approximates ``dphi/dt`` to O(dt) -- backward Euler, as documented."""
    order = _observed_order(lambda dt: abs(_stencil_derivative(dt, first_step=True) - 1.0))
    assert 0.9 < order < 1.1


def test_bdf2_is_second_order_accurate_in_time() -> None:
    """The later-step branch approximates ``dphi/dt`` to O(dt^2), which is the whole reason it exists.

    This is the assertion that a silently-first-order BDF2 fails and the coefficient test above does
    not: any consistent stencil passes a residual comparison against itself, only the rate separates
    a second-order one from a first-order one.
    """
    order = _observed_order(lambda dt: abs(_stencil_derivative(dt, first_step=False) - 1.0))
    assert order > 1.9


def test_marching_a_decay_problem_converges_at_second_order() -> None:
    """Marched over a fixed interval, the scheme reaches the exact solution at second order.

    The stencil tests above are local; this is the global statement a user actually gets, and it also
    exercises the BDF1-then-BDF2 startup the driver performs -- a first step of the wrong order does
    not spoil the global rate, because it is one step of many.

    ``dphi/dt = -lambda phi`` on one cell, whose residual ``V (BDF stencil) + V lambda phi`` is linear
    in ``phi``, so each step is solved in closed form rather than by a Newton solve: what is under
    test is the time discretization, not the solver.
    """
    rate, horizon, volume = 2.0, 1.0, jnp.ones(1)
    term = TransientTerm()

    def marched(n_steps: int) -> float:
        dt = horizon / n_steps
        phi, older = jnp.ones(1), jnp.ones(1)
        for step in range(n_steps):
            first = step == 0
            # Solve V(a0 phi_new + rest)/dt + V rate phi_new = 0 for phi_new, reading the stencil's
            # own coefficients off it rather than restating them: a0 from the phi-slope, `rest` from
            # the residual at phi_new = 0.
            zero = jnp.zeros(1)
            rest = term.residual(zero, phi, older, dt, first_step=first, volume=volume)
            a0 = term.residual(volume, phi, older, dt, first_step=first, volume=volume) - rest
            phi, older = -rest / (a0 + volume * rate), phi
        return float(phi[0])

    exact = float(jnp.exp(-rate * horizon))
    errors = [abs(marched(n) - exact) for n in (10, 20, 40, 80)]
    orders = [float(jnp.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]
    assert min(orders) > 1.9
