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


def test_dot_broadcasts_a_trailing_axis_of_one_against_dim():
    """A ``(..., 1)`` operand is stretched across the spatial axis of the other.

    ``dot`` multiplies its operands before contracting, so the components it sums are indexed out
    of the already-broadcast product. Contracting component-by-component off the *unbroadcast*
    operands would instead run off the end of the length-one axis.
    """
    field = jnp.arange(12.0).reshape(4, 3)
    per_element = jnp.asarray([[2.0], [3.0], [4.0], [5.0]])
    reference = np.sum(np.asarray(field) * np.asarray(per_element), axis=-1)
    np.testing.assert_allclose(dot(field, per_element), reference)
    np.testing.assert_allclose(dot(per_element, field), reference)


def test_dot_of_an_empty_spatial_axis_is_zero():
    """A contraction over no components sums no terms, so it is zero on every leading index."""
    empty = jnp.zeros((4, 0))
    got = dot(empty, empty)
    assert got.shape == (4,)
    np.testing.assert_array_equal(got, np.zeros(4))


@pytest.mark.parametrize("dim", [1, 2, 3, 9])
def test_dot_agrees_with_the_axis_reduction_to_a_rounding_of_its_terms(dim):
    """``dot`` sums the components explicitly, which is the same contraction as the reduction.

    Summing in a different association lets the compiler contract a different pair of the
    multiply-adds, so the two forms are not required to agree bit for bit — but they must agree to
    within a rounding of the magnitudes being summed. Judging against those magnitudes rather than
    against the result is what makes the tolerance meaningful: a dot product of near-orthogonal
    vectors cancels, so a few ulps of the terms is an unbounded *relative* error on the answer.
    """
    a = jax.random.normal(jax.random.PRNGKey(0), (500, dim), dtype=jnp.float64)
    b = jax.random.normal(jax.random.PRNGKey(1), (500, dim), dtype=jnp.float64)
    magnitude_summed = jnp.sum(jnp.abs(a * b), axis=-1)
    deviation = jnp.abs(dot(a, b) - jnp.sum(a * b, axis=-1))
    assert float(jnp.max(deviation / magnitude_summed)) < 1e-14


def test_dot_does_not_lower_to_an_axis_reduction():
    """The contraction stays elementwise instead of being rooted at a reduction.

    A fusion rooted at a reduction is emitted by the CPU backend as a custom kernel that runs on a
    single thread, while the elementwise fusions around it are split across cores — so spelling the
    sum out is what keeps this operation parallel with its neighbours. This checks the property at
    the level that survives a compiler version: that no reduction primitive is traced. It does
    **not** inspect the emitted kernels, so it cannot confirm the partitioning itself; what it
    catches is ``dot`` being rewritten back into ``jnp.sum(a * b, axis=-1)``.
    """
    jaxpr = jax.make_jaxpr(dot)(jnp.zeros((8, 3)), jnp.zeros((8, 3)))
    primitives = {eqn.primitive.name for eqn in jaxpr.eqns}
    assert not any("reduce" in name for name in primitives), primitives


def test_dot_of_numpy_operands_still_returns_a_jax_array():
    """Contracting two NumPy arrays yields a JAX array, as the reduction it replaces did.

    ``dot`` forms its product before summing, and ``np * np`` is a NumPy array — so spelling that
    product with ``*`` would hand back a ``numpy.ndarray`` and silently drop the JAX surface
    (``.at[]`` indexing above all) for callers that pass concrete arrays.
    """
    a = np.arange(6.0).reshape(2, 3)
    b = np.ones((2, 3))
    got = dot(a, b)
    assert isinstance(got, jax.Array), type(got)
    assert hasattr(got, "at")
    np.testing.assert_allclose(got, np.sum(a * b, axis=-1))
