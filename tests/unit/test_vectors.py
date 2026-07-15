"""Unit tests for the vector-field algebra helpers (dot / norm_squared / scale)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.vectors import dot, norm_squared, scale


def test_dot_matches_reference_per_face():
    a = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = jnp.asarray([[7.0, 8.0, 9.0], [1.0, 0.0, -1.0]])
    got = dot(a, b)
    assert got.shape == (2,)
    np.testing.assert_allclose(got, np.sum(np.asarray(a) * np.asarray(b), axis=-1))


def test_dot_single_vector_and_higher_rank_batch():
    # a single (dim,) vector contracts to a scalar
    v = jnp.asarray([3.0, 4.0])
    np.testing.assert_allclose(dot(v, v), 25.0)
    # a (batch, n, dim) field contracts only the last axis
    a = jnp.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    got = dot(a, a)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, np.sum(np.asarray(a) ** 2, axis=-1))


def test_norm_squared_is_dot_with_self():
    a = jnp.asarray([[3.0, 4.0], [5.0, 12.0]])
    np.testing.assert_allclose(norm_squared(a), [25.0, 169.0])
    np.testing.assert_allclose(norm_squared(a), dot(a, a))


def test_scale_broadcasts_scalar_over_components():
    vectors = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scalars = jnp.asarray([10.0, 100.0, 1000.0])
    got = scale(vectors, scalars)
    assert got.shape == vectors.shape
    np.testing.assert_allclose(got, np.asarray(scalars)[:, None] * np.asarray(vectors))


def test_dot_and_scale_are_differentiable():
    a = jnp.asarray([1.0, 2.0, 3.0])
    b = jnp.asarray([4.0, 5.0, 6.0])
    # d/da (a·b) = b
    np.testing.assert_allclose(jax.grad(lambda x: dot(x, b))(a), b)
    # d/ds Σ scale(v, s) sums each vector's components
    v = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    s = jnp.asarray([2.0, 3.0])
    grad_s = jax.grad(lambda sc: jnp.sum(scale(v, sc)))(s)
    np.testing.assert_allclose(grad_s, np.sum(np.asarray(v), axis=1))


@pytest.mark.parametrize("fn", [dot, norm_squared])
def test_zero_vector_contracts_to_zero(fn):
    z = jnp.zeros((4, 3))
    if fn is dot:
        np.testing.assert_allclose(fn(z, z), np.zeros(4))
    else:
        np.testing.assert_allclose(fn(z), np.zeros(4))
