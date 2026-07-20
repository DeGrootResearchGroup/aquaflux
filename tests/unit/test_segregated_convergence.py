"""Unit: the segregated driver's outer control -- convergence stop + adaptive relaxation.

These exercise the residual-agnostic outer loop of :func:`~aquaflux.turbulence.solve_segregated`
without a full coupled CFD solve. The two pieces of new numerical logic are pure functions (the
coupled increment measure and the SER relaxation ramp) and are tested directly; the stop / relax /
warn wiring is then driven with light stand-ins for the flow and scalar solvers whose only job is to
march the fields toward a fixed point at a controlled rate, so the loop's control flow can be
asserted (sweeps taken, non-convergence warning) in a fraction of a second.
"""

from __future__ import annotations

import warnings

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from aquaflux.properties import Constant, PropertyModel
from aquaflux.turbulence.driver import _relative_change, _sweep_relaxation, solve_segregated

# --- the coupled increment measure -------------------------------------------------------------


def test_relative_change_is_the_max_per_field_relative_l2() -> None:
    # A 10 % field change dominates a 1 % one; the measure is ||new - old|| / ||new|| per field.
    slow = (jnp.ones(4), jnp.full(4, 1.01))
    fast = (jnp.ones(4), jnp.full(4, 1.10))
    assert _relative_change(slow, fast) == pytest.approx(0.10 / 1.10)


def test_relative_change_is_scale_free() -> None:
    # The same *relative* move on fields five orders of magnitude apart contributes equally, so a
    # converged velocity cannot mask a still-moving omega.
    small = (jnp.full(4, 1e-3), jnp.full(4, 1e-3 * 1.05))
    big = (jnp.full(4, 1e3), jnp.full(4, 1e3 * 1.05))
    assert _relative_change(small) == pytest.approx(_relative_change(big))


def test_relative_change_is_zero_when_unchanged() -> None:
    x = jnp.array([1.0, 2.0, 3.0])
    assert _relative_change((x, x)) == 0.0


# --- the adaptive relaxation ramp --------------------------------------------------------------


def test_sweep_relaxation_uses_the_floor_before_an_increment_exists() -> None:
    assert _sweep_relaxation(None, None, 0.5, 1.0, 1.0) == 0.5


def test_sweep_relaxation_is_constant_when_the_ceiling_equals_the_floor() -> None:
    # relaxation_max pinned to the floor -> the plain Picard loop, whatever the increment did.
    assert _sweep_relaxation(1e-6, 1.0, 0.5, 0.5, 1.0) == 0.5


def test_sweep_relaxation_opens_up_as_the_increment_falls() -> None:
    # A 2x drop from the reference doubles the floor; a 10x drop saturates at the ceiling.
    assert _sweep_relaxation(0.5, 1.0, 0.4, 1.0, 1.0) == pytest.approx(0.8)
    assert _sweep_relaxation(0.05, 1.0, 0.4, 1.0, 1.0) == pytest.approx(1.0)


def test_sweep_relaxation_never_drops_below_the_floor_while_diverging() -> None:
    # Increment grew above the reference (ratio < 1) -> the clip holds the floor, never undercuts it.
    assert _sweep_relaxation(5.0, 1.0, 0.4, 1.0, 1.0) == 0.4


def test_sweep_relaxation_exponent_sharpens_the_ramp() -> None:
    gentle = _sweep_relaxation(0.5, 1.0, 0.1, 1.0, 1.0)
    steep = _sweep_relaxation(0.5, 1.0, 0.1, 1.0, 3.0)
    assert steep > gentle


# --- the outer loop's stop / warn wiring, on stand-in solvers -----------------------------------


class _StubMomentum(eqx.Module):
    """Just enough of a flow assembler for the driver: the material properties, the eddy-viscosity
    leaf the loop sets each sweep, and the two field queries it makes (their return values are
    ignored by the stub turbulence)."""

    properties: PropertyModel
    eddy_viscosity: jax.Array | None = None

    def with_eddy_viscosity(self, eddy_viscosity: jax.Array) -> _StubMomentum:
        return _StubMomentum(self.properties, eddy_viscosity)

    def velocity_gradient(self, flow: jax.Array) -> jax.Array:
        return flow

    def mass_flux(self, flow: jax.Array) -> jax.Array:
        return flow


class _StubTurbulence(eqx.Module):
    """A closure that contributes no eddy viscosity and whose per-scalar hooks are inert -- the
    stand-in scalar solve supplies the field update the increment is measured from."""

    molecular_viscosity: jax.Array

    def eddy_viscosity(self, velocity_gradient, k, omega):
        return jnp.zeros_like(k)

    def closure_fields(self, velocity_gradient, k, omega):
        return None

    def k_shift_policy(self, mdot, closure, k, method=None):
        return None

    def omega_shift_policy(self, mdot, closure, omega, method=None):
        return None

    def k_residual(self, mdot, closure):
        return None

    def omega_residual(self, mdot, closure):
        return None


def _drive(*, max_sweeps, rtol, contraction, relaxation=1.0, relaxation_max=None):
    """Run the driver with solvers that contract each field toward a fixed target by ``contraction``
    per solve, counting the sweeps actually taken. ``contraction`` in ``[0, 1)`` converges;
    ``rtol=0`` never accepts, forcing the cap."""
    n = 4
    momentum = _StubMomentum(PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}))
    turbulence = _StubTurbulence(jnp.full(n, 1.0))
    flow_target, scalar_target = jnp.ones(n), jnp.full(n, 2.0)
    sweeps = []

    def solve_flow(m, flow):
        sweeps.append(1)
        return flow_target + contraction * (flow - flow_target)

    def solve_scalar(residual, state, policy):
        return scalar_target + contraction * (state - scalar_target)

    flow, k, omega = solve_segregated(
        momentum,
        turbulence,
        solve_flow,
        solve_scalar,
        jnp.zeros(n),
        jnp.full(n, 0.1),
        jnp.full(n, 0.1),
        max_sweeps=max_sweeps,
        rtol=rtol,
        relaxation=relaxation,
        relaxation_max=relaxation_max,
    )
    return len(sweeps), flow, k, omega


def test_loop_stops_on_convergence_before_the_cap() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a non-convergence warning would fail the test
        taken, flow, k, _ = _drive(max_sweeps=50, rtol=1e-3, contraction=0.5)
    assert taken < 50
    assert float(jnp.max(jnp.abs(flow - 1.0))) < 1e-2  # actually parked at the fixed point
    assert float(jnp.max(jnp.abs(k - 2.0))) < 1e-2


def test_loop_warns_and_runs_to_the_cap_when_not_converged() -> None:
    with pytest.warns(UserWarning, match="did not reach"):
        taken, *_ = _drive(max_sweeps=3, rtol=0.0, contraction=0.5)
    assert taken == 3


def test_adaptive_relaxation_is_no_slower_than_a_constant_floor() -> None:
    # Same problem, same floor; opening the relaxation toward 1.0 as it settles cannot need more
    # sweeps than pinning it at the floor.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        constant, *_ = _drive(
            max_sweeps=200, rtol=1e-4, contraction=0.7, relaxation=0.3, relaxation_max=None
        )
        ramped, *_ = _drive(
            max_sweeps=200, rtol=1e-4, contraction=0.7, relaxation=0.3, relaxation_max=1.0
        )
    assert ramped <= constant
