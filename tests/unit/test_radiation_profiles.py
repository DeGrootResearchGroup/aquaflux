"""Angular distributions: their normalization, their two views, and how they reduce."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from scipy import integrate

PROFILES = [Isotropic(), Lambertian(), CosinePower(1.0), CosinePower(2.0), CosinePower(40.0)]


def _sphere_integral(profile) -> float:
    """Integrate ``f`` over the whole sphere by quadrature in the polar angle."""
    value, _ = integrate.quad(
        lambda theta: (
            float(profile.intensity_fraction(jnp.cos(theta))) * 2.0 * np.pi * np.sin(theta)
        ),
        0.0,
        np.pi,
        limit=200,
    )
    return value


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: repr(p)[:24])
def test_every_profile_integrates_to_one_over_the_sphere(profile):
    """The normalization the whole design rests on: intensity is power times this.

    Get the constant wrong and every source is uniformly too bright or too dim by that factor,
    which no comparison of one geometry against another can reveal.
    """
    assert _sphere_integral(profile) == pytest.approx(1.0, rel=1e-9)


def test_a_cosine_power_of_one_is_exactly_lambertian():
    """The reduction that pins the ``2 pi`` in the constant.

    ``(n + 1) / (2 pi)`` at ``n = 1`` is ``1 / pi``. Writing ``(n + 1) / pi`` -- the predictable
    slip, since the hemispherical constant for Lambertian is ``1 / pi`` -- doubles every
    cosine-power source, and this is the case where that shows up as an exact disagreement.
    """
    cosines = jnp.linspace(-1.0, 1.0, 21)
    np.testing.assert_allclose(
        CosinePower(1.0).intensity_fraction(cosines), Lambertian().intensity_fraction(cosines)
    )
    np.testing.assert_allclose(
        CosinePower(1.0).radiance_per_exitance(cosines),
        Lambertian().radiance_per_exitance(cosines),
    )


@pytest.mark.parametrize("profile", [Lambertian(), CosinePower(1.0), CosinePower(3.0)])
def test_the_two_views_of_a_distribution_agree(profile):
    """``radiance_per_exitance(c) * c == intensity_fraction(c)`` — the contract between them.

    They are written separately so the Lambertian cosine cancels analytically instead of
    numerically, and separate definitions are exactly what can drift apart.
    """
    cosines = jnp.linspace(0.05, 1.0, 20)
    np.testing.assert_allclose(
        profile.radiance_per_exitance(cosines) * cosines,
        profile.intensity_fraction(cosines),
        rtol=1e-14,
    )


def test_a_lambertian_radiance_is_the_same_constant_everywhere_including_at_grazing():
    """The reason the ratio is supplied rather than divided out.

    A Lambertian emitter looks equally bright from every angle -- that is what the cosine law
    means -- so its radiance per unit exitance is ``1 / pi`` right up to grazing incidence,
    where forming ``f(c) / c`` numerically would be zero over zero.
    """
    cosines = jnp.array([1.0, 0.5, 1e-8, 1e-300])
    np.testing.assert_allclose(
        Lambertian().radiance_per_exitance(cosines), np.full(4, 1.0 / np.pi), rtol=1e-15
    )


@pytest.mark.parametrize("profile", [Lambertian(), CosinePower(2.0)])
def test_nothing_leaves_through_the_back_of_a_surface(profile):
    """The clamp. A facet cannot illuminate what is behind it, and unclamped it would."""
    behind = jnp.array([-1.0, -0.5, -1e-12, 0.0])
    np.testing.assert_allclose(profile.intensity_fraction(behind), np.zeros(4))
    np.testing.assert_allclose(profile.radiance_per_exitance(behind), np.zeros(4))


def test_a_larger_exponent_concentrates_the_beam():
    """Otherwise the exponent is a parameter that normalizes correctly and does nothing."""
    on_axis = [float(CosinePower(n).intensity_fraction(jnp.asarray(1.0))) for n in (1.0, 4.0, 16.0)]
    off_axis = [
        float(CosinePower(n).intensity_fraction(jnp.asarray(0.5))) for n in (1.0, 4.0, 16.0)
    ]
    assert on_axis[0] < on_axis[1] < on_axis[2]
    assert off_axis[0] > off_axis[1] > off_axis[2]


def test_an_isotropic_profile_has_no_radiance_per_exitance():
    """It describes a point source, which has no surface for an exitance to live on.

    Returning some number instead would let an isotropic profile be attached to an emitting
    facet, where it divides by a vanishing cosine at grazing incidence.
    """
    with pytest.raises(NotImplementedError, match="point-source profile"):
        Isotropic().radiance_per_exitance(jnp.asarray(0.5))


def test_an_isotropic_profile_is_the_same_in_every_direction_including_backwards():
    """Unlike a surface distribution, it has no front: a point source radiates into 4 pi."""
    np.testing.assert_allclose(
        Isotropic().intensity_fraction(jnp.linspace(-1.0, 1.0, 9)),
        np.full(9, 1.0 / (4.0 * np.pi)),
    )


@pytest.mark.parametrize("exponent", [0.0, 0.5, -1.0])
def test_an_exponent_below_one_is_refused(exponent):
    """Below one the radiance grows without bound at grazing incidence, which no surface does."""
    with pytest.raises(ValueError, match="at least 1"):
        CosinePower(exponent)


def test_the_exponent_is_a_differentiable_leaf():
    """A beam's width is a design variable, so it has to be one the optimizer can move."""
    gradient = jax.grad(lambda n: CosinePower(n).intensity_fraction(jnp.asarray(0.7)))(
        jnp.asarray(3.0)
    )
    assert jnp.isfinite(gradient)
    assert float(gradient) != 0.0
    assert jax.tree_util.tree_leaves(CosinePower(3.0))
