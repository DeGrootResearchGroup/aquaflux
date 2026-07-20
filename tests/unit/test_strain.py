"""Unit tests for the strain-rate magnitude (pure tensor algebra, no model, no mesh)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
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


def test_derivative_is_finite_at_zero_strain() -> None:
    """The Jacobian at ``S = 0`` is the finite minimum-norm subgradient (``dS = 0``), not NaN.

    A uniform velocity field has zero gradient, so ``S = sqrt(2 S_ij S_ij) = 0``; the plain ``sqrt``
    chain rule differentiates to ``0 / 0 = NaN`` there, which is what poisoned the coupled Jacobian on
    a body-force-driven periodic channel's uniform-plug start. The guarded ``sqrt`` returns ``0``.
    """
    grad = jnp.zeros((4, 3, 3))
    _, ds = jax.jvp(strain_rate_magnitude, (grad,), (jnp.ones_like(grad),))
    assert jnp.all(jnp.isfinite(ds))
    assert jnp.allclose(ds, 0.0)
    # The full Jacobian (all tangent directions) is finite and vanishes at the cone point.
    jacobian = jax.jacfwd(strain_rate_magnitude)(grad)
    assert jnp.all(jnp.isfinite(jacobian))
    assert jnp.allclose(jacobian, 0.0)


def test_derivative_matches_finite_difference_where_strain_is_nonzero() -> None:
    """Where ``S > 0`` the guarded ``sqrt`` gives the ordinary derivative -- unchanged by the fix."""
    key = jax.random.PRNGKey(0)
    grad = jax.random.normal(key, (5, 3, 3))
    tangent = jax.random.normal(jax.random.PRNGKey(1), (5, 3, 3))
    _, ds = jax.jvp(strain_rate_magnitude, (grad,), (tangent,))
    eps = 1e-6
    fd = (
        strain_rate_magnitude(grad + eps * tangent) - strain_rate_magnitude(grad - eps * tangent)
    ) / (2.0 * eps)
    assert np.allclose(np.asarray(ds), np.asarray(fd), rtol=1e-6, atol=1e-9)
