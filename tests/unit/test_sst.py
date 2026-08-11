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


# --- transiently negative k (off-solution guards) ---------------------------------------


def test_f1_and_f2_are_finite_and_bounded_for_a_negative_k() -> None:
    """A cell whose ``k`` has been carried below zero must not poison the blend.

    ``k`` is solved directly -- a log transform is singular at a no-slip wall, where ``k = 0`` is the
    physical boundary condition -- so a Newton step can carry it negative, and one such cell is enough:
    a plain ``sqrt`` of it is NaN, which spreads through the whole residual. ``F1``'s cross-diffusion
    branch has the quieter failure of the two: a negative ``k`` makes it negative, so it wins the ``min``
    and ``arg1**4`` selects the wrong blend with no NaN to announce it.
    """
    model = SSTModel()
    grad = jnp.ones((1, 2))
    f1 = model.f1(_cell(-1e-12), _cell(100.0), _cell(1e-5), _cell(1.0), grad, grad)
    f2 = model.f2(_cell(-1e-12), _cell(100.0), _cell(1e-5), _cell(1.0))
    for name, value in (("F1", f1), ("F2", f2)):
        assert jnp.all(jnp.isfinite(value)), name
        assert jnp.all(value >= 0.0) and jnp.all(value <= 1.0), name


def test_the_negative_k_guards_are_inactive_for_a_positive_k() -> None:
    """The guards are off-solution: at any ``k > 0`` both blends are bit-identical to the unguarded form.

    This is the property that lets them sit inside a differentiated residual -- inactive at the fixed
    point means the sensitivity is untouched there.
    """
    model = SSTModel()
    grad = jnp.ones((1, 2))
    k, omega, nu, d = _cell(0.81), _cell(100.0), _cell(0.06), _cell(1.0)
    cross = jnp.maximum(2.0 * model.sigma_omega2 * 2.0 / omega, 1e-10)
    unguarded_f2 = jnp.tanh(
        jnp.maximum(2.0 * jnp.sqrt(k) / (model.beta_star * omega * d), 500.0 * nu / (d**2 * omega))
        ** 2
    )
    unguarded_f1 = jnp.tanh(
        jnp.minimum(
            jnp.maximum(jnp.sqrt(k) / (model.beta_star * omega * d), 500.0 * nu / (d**2 * omega)),
            4.0 * model.sigma_omega2 * k / (cross * d**2),
        )
        ** 4
    )
    assert model.f2(k, omega, nu, d) == unguarded_f2
    assert model.f1(k, omega, nu, d, grad, grad) == unguarded_f1


def test_a_negative_k_gives_a_finite_gradient_through_the_blends() -> None:
    """The guard must be differentiable too -- a NaN derivative poisons the Jacobian as surely as a
    NaN value, and this residual is differentiated by automatic differentiation, not by hand."""
    model = SSTModel()
    grad = jnp.ones((1, 2))

    def blends(k):
        return (
            model.f1(k, _cell(100.0), _cell(1e-5), _cell(1.0), grad, grad)
            + model.f2(k, _cell(100.0), _cell(1e-5), _cell(1.0))
        ).sum()

    assert jnp.all(jnp.isfinite(jax.grad(blends)(_cell(-1e-12))))
    assert jnp.all(jnp.isfinite(jax.grad(blends)(_cell(0.0))))
