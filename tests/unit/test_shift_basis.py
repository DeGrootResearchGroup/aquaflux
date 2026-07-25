"""Unit tests for the pseudo-transient shift-basis strategy (pure per-cell combination, no solve)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.solve import LocalCourantBasis


def test_full_weight_is_the_operator_diagonal() -> None:
    """``w = 1`` sums both buckets, i.e. the full operator diagonal (uniform-relaxation default)."""
    convective = jnp.array([1.0, 2.0, 3.0])
    dissipative = jnp.array([0.5, 4.0, 0.25])
    got = LocalCourantBasis().local_diagonal(convective, dissipative)
    assert jnp.array_equal(got, convective + dissipative)


def test_zero_weight_is_the_pure_convective_time_step() -> None:
    """``w = 0`` drops the dissipative bucket -- the pure convective local time step."""
    convective = jnp.array([1.0, 2.0, 3.0])
    dissipative = jnp.array([0.5, 4.0, 0.25])
    got = LocalCourantBasis(dissipative_weight=0.0).local_diagonal(convective, dissipative)
    assert jnp.array_equal(got, convective)


def test_intermediate_weight_down_weights_the_dissipative_bucket() -> None:
    """An intermediate ``w`` keeps a fraction of the dissipative stiffness."""
    convective = jnp.array([1.0, 2.0])
    dissipative = jnp.array([4.0, 8.0])
    got = LocalCourantBasis(dissipative_weight=0.25).local_diagonal(convective, dissipative)
    assert jnp.allclose(got, convective + 0.25 * dissipative)


def test_the_basis_is_non_negative_for_non_negative_buckets() -> None:
    """Both buckets are ``>= 0``, so the base shift diagonal is too (a valid pseudo-time scale)."""
    convective = jnp.array([0.0, 3.0, 1.5])
    dissipative = jnp.array([2.0, 0.0, 0.5])
    for weight in (0.0, 0.5, 1.0):
        got = LocalCourantBasis(dissipative_weight=weight).local_diagonal(convective, dissipative)
        assert bool(jnp.all(got >= 0.0))
