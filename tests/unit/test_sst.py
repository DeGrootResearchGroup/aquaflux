"""Unit tests for the k-omega SST constants and the quantities derived from them.

Each closed form (F2, F1, the eddy viscosity, the constant blend) is checked on hand-chosen inputs
where one branch of a min/max is known to win, so the assertions are against the analytic value, not
a re-derivation. Inputs are kept away from the tanh saturation so the formula (not the plateau) is
exercised.
"""

from __future__ import annotations

import math

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
from aquaflux.turbulence import SSTModel


def _cell(*values):
    """A one-cell field from scalar(s)."""
    return jnp.array([float(v) for v in values])


# --- blend -----------------------------------------------------------------------------


def test_blend_selects_inner_when_f1_is_one_and_outer_when_zero() -> None:
    model = SSTModel()
    assert jnp.allclose(model.blend(_cell(1.0), 0.85, 1.0), 0.85)
    assert jnp.allclose(model.blend(_cell(0.0), 0.85, 1.0), 1.0)
    assert jnp.allclose(model.blend(_cell(0.5), 0.85, 1.0), 0.5 * (0.85 + 1.0))


# --- F2 --------------------------------------------------------------------------------


def test_f2_first_branch_dominant() -> None:
    """With nu -> 0 the first term ``2 sqrt(k)/(beta* omega d)`` wins: F2 = tanh(arg**2)."""
    model = SSTModel()
    # 2*sqrt(0.81)/(0.09*100*1) = 0.2; second term (nu=0) = 0.
    f2 = model.f2(_cell(0.81), _cell(100.0), _cell(0.0), _cell(1.0))
    assert jnp.allclose(f2, math.tanh(0.2**2))


def test_f2_second_branch_dominant() -> None:
    """With larger nu the second term ``500 nu/(d^2 omega)`` wins."""
    model = SSTModel()
    # first term 0.2; second = 500*0.06/(1*100) = 0.3.
    f2 = model.f2(_cell(0.81), _cell(100.0), _cell(0.06), _cell(1.0))
    assert jnp.allclose(f2, math.tanh(0.3**2))


# --- F1 --------------------------------------------------------------------------------


def test_f1_floors_cross_diffusion_when_gradients_are_orthogonal() -> None:
    """With grad_k . grad_omega = 0 the cross-diffusion floor makes the third term huge, so arg1 is
    ``max(sqrt(k)/(beta* omega d), 500 nu/(d^2 omega))`` = 0.1 here and F1 = tanh(0.1**4)."""
    model = SSTModel()
    zero = jnp.zeros((1, 2))
    f1 = model.f1(_cell(0.81), _cell(100.0), _cell(0.0), _cell(1.0), zero, zero)
    assert jnp.allclose(f1, math.tanh(0.1**4))


def test_f1_cross_diffusion_branch_lowers_f1() -> None:
    """A large positive grad_k . grad_omega makes the cross-diffusion term bind, so arg1 (hence F1)
    drops below the orthogonal-gradient value."""
    model = SSTModel()
    args = (_cell(0.81), _cell(100.0), _cell(0.0), _cell(1.0))
    grad = jnp.array([[100.0, 0.0]])
    f1_cross = model.f1(*args, grad, grad)
    f1_floor = model.f1(*args, jnp.zeros((1, 2)), jnp.zeros((1, 2)))
    assert float(f1_cross[0]) < float(f1_floor[0])


# --- eddy viscosity --------------------------------------------------------------------


def test_eddy_viscosity_unlimited_branch_is_k_over_omega() -> None:
    """With zero strain the limiter is inactive: nu_t = a1 k /(a1 omega) = k/omega."""
    model = SSTModel()
    nu_t = model.eddy_viscosity(_cell(2.0), _cell(4.0), _cell(0.0), _cell(1e-3), _cell(1.0))
    assert jnp.allclose(nu_t, 0.5)


def test_eddy_viscosity_limiter_caps_at_high_strain() -> None:
    """At high strain the ``S F2`` branch wins, so nu_t = a1 k /(S F2) and is below k/omega."""
    model = SSTModel()
    k, omega, s, nu, d = (_cell(1.0), _cell(1.0), _cell(100.0), _cell(0.0), _cell(1.0))
    nu_t = model.eddy_viscosity(k, omega, s, nu, d)
    expected = model.a1 * k / (s * model.f2(k, omega, nu, d))
    assert jnp.allclose(nu_t, expected)
    assert float(nu_t[0]) < float((k / omega)[0])  # the limiter reduced it


def test_constants_and_state_are_differentiable() -> None:
    """jax.grad flows through a model constant and through the k field, no NaNs."""
    k, omega, s, nu, d = (_cell(1.0), _cell(1.0), _cell(100.0), _cell(0.0), _cell(1.0))
    grad_a1 = jax.grad(lambda a1: jnp.sum(SSTModel(a1=a1).eddy_viscosity(k, omega, s, nu, d)))(0.31)
    grad_k = jax.grad(lambda kk: jnp.sum(SSTModel().eddy_viscosity(kk, omega, s, nu, d)))(k)
    assert not bool(jnp.isnan(grad_a1))
    assert not bool(jnp.any(jnp.isnan(grad_k)))
