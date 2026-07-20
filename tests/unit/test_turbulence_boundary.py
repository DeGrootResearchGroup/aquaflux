"""Unit tests for the k-omega SST boundary values (pure formulas, no mesh, no solve)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.turbulence import SSTModel, inlet_k, inlet_omega, omega_wall_value

MODEL = SSTModel()


def test_omega_wall_value_solves_the_viscous_sublayer_balance() -> None:
    """The wall omega must *satisfy* the sublayer equation at the distance it is imposed at.

    As ``k -> 0`` the omega equation reduces to viscous diffusion against destruction,
    ``nu d2(omega)/dy2 = beta_1 omega**2``. Checking the residual of that balance -- rather than
    restating the closed-form coefficient -- is what distinguishes the analytical cell-centre value
    from the ten-times-larger wall-face surrogate, which leaves a residual of order the equation
    itself.
    """
    nu, d = 1e-5, jnp.logspace(-5.0, -3.0, 7)
    omega = omega_wall_value(jnp.full_like(d, nu), d, MODEL)
    # omega = A / d**2  =>  d2(omega)/dy2 = 6 A / d**4, evaluated from the returned field itself.
    a = omega * d**2
    residual = nu * 6.0 * a / d**4 - MODEL.beta_1 * omega**2
    assert jnp.max(jnp.abs(residual)) < 1e-8 * jnp.max(MODEL.beta_1 * omega**2)


def test_omega_wall_value_scales_with_viscosity_and_inverse_square_distance() -> None:
    nu, d = jnp.array([1e-5]), jnp.array([1e-3])
    base = omega_wall_value(nu, d, MODEL)
    assert jnp.allclose(omega_wall_value(2.0 * nu, d, MODEL), 2.0 * base)
    assert jnp.allclose(omega_wall_value(nu, 2.0 * d, MODEL), base / 4.0)


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
