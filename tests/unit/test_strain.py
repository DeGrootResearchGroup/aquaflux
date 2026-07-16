"""Unit tests for the strain-rate magnitude (pure tensor algebra, no model, no mesh)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.turbulence import strain_rate_magnitude


def test_pure_shear_magnitude_equals_the_shear_rate() -> None:
    """For simple shear ``du_x/dy = gamma`` (all else zero), ``S = gamma``."""
    grad = jnp.array([[[0.0, 2.0], [0.0, 0.0]]])  # (1, 2, 2), du_x/dy = 2
    assert jnp.allclose(strain_rate_magnitude(grad), 2.0)


def test_axial_strain_magnitude() -> None:
    """For pure axial strain ``du_x/dx = a``, ``S = sqrt(2) * a``."""
    grad = jnp.array([[[3.0, 0.0], [0.0, 0.0]]])  # du_x/dx = 3
    assert jnp.allclose(strain_rate_magnitude(grad), jnp.sqrt(2.0) * 3.0)


def test_zero_gradient_gives_zero_strain() -> None:
    assert jnp.allclose(strain_rate_magnitude(jnp.zeros((4, 3, 3))), 0.0)


def test_is_invariant_to_the_transpose_convention() -> None:
    """Only the symmetric part enters, so transposing the gradient leaves S unchanged."""
    grad = jnp.array([[[1.0, 2.0], [-0.5, 0.3]]])
    assert jnp.allclose(
        strain_rate_magnitude(grad), strain_rate_magnitude(jnp.swapaxes(grad, -1, -2))
    )
