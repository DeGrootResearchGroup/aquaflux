"""Unit tests for the k-omega SST boundary values (pure formulas, no mesh, no solve)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.turbulence import SSTModel, inlet_k, inlet_omega, omega_wall_value

MODEL = SSTModel()


def test_omega_wall_value() -> None:
    nu = jnp.array([1e-5])
    d = jnp.array([1e-3])
    assert jnp.allclose(omega_wall_value(nu, d, MODEL), 60.0 * 1e-5 / (MODEL.beta_1 * 1e-3**2))


def test_inlet_k_from_intensity() -> None:
    assert jnp.allclose(inlet_k(jnp.array([10.0]), 0.05), 1.5 * (10.0 * 0.05) ** 2)


def test_inlet_omega_from_length_scale() -> None:
    k = jnp.array([0.375])
    assert jnp.allclose(inlet_omega(k, 0.1, MODEL), jnp.sqrt(k) / (MODEL.beta_star**0.25 * 0.1))


def test_boundary_values_are_differentiable() -> None:
    """Gradients flow through the viscosity (wall omega) and the velocity (inlet k), no NaNs."""
    grad_nu = jax.grad(lambda nu: jnp.sum(omega_wall_value(nu, jnp.array([1e-3]), MODEL)))(
        jnp.array([1e-5])
    )
    grad_u = jax.grad(lambda u: jnp.sum(inlet_k(u, 0.05)))(jnp.array([10.0]))
    assert not bool(jnp.any(jnp.isnan(grad_nu)))
    assert not bool(jnp.any(jnp.isnan(grad_u)))
