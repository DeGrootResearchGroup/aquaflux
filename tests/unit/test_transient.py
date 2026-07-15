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
