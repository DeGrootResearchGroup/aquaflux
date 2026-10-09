"""Unit tests for :class:`~aquaflux.schemes.BoundaryClosure`, the boundary closures of one field.

Each method is checked against something other than its own formula. The weights against
closed-form closures whose derivatives are known by hand -- a prescribed value, a zero-gradient
extrapolation along an offset, and a Robin blend between the two -- on a two-cell mesh small enough
to read. The two-pass reconstruction against what it promises: the scheme is fed the closures at a
zero gradient, it is handed the closures at any gradient it asks for, and the returned boundary
values are the closures at the gradient it returned. No scheme argument reconstructs nothing.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.mesh import structured_grid_2d
from aquaflux.schemes import BoundaryClosure, GradientScheme
from aquaflux.vectors import dot


def _closure(mesh):
    """Every boundary face follows its owner along a fixed offset, blended with a datum by ``a``.

    ``value = a * (phi_P + grad phi_P . w) + (1 - a) * 2``, with ``a`` 0 (prescribed), 1 (follows the
    owner) or 0.25 (a Robin blend) by face, and ``w`` a per-face offset. So the value weight is ``a``
    and the gradient weight ``a * w``, known by hand.
    """
    owner = mesh.face_cells.owner
    boundary = ~np.asarray(mesh.face_cells.interior)
    pattern = np.array([0.0, 1.0, 0.25])
    a = np.where(boundary, pattern[np.arange(mesh.n_faces) % 3], 0.0)
    w = np.where(
        boundary[:, None],
        np.stack([np.arange(mesh.n_faces) * 0.1, -0.3 * np.ones(mesh.n_faces)], 1),
        0.0,
    )
    a, w = jnp.asarray(a), jnp.asarray(w)

    def values(field, gradient):
        follows = field[owner] + dot(gradient[owner], w)
        return a * follows + (1.0 - a) * 2.0

    return BoundaryClosure(values), a, w


def test_the_weights_are_the_derivatives_of_the_closures() -> None:
    mesh = structured_grid_2d(3, 2)
    closure, a, w = _closure(mesh)
    field = jnp.linspace(-1.0, 1.0, mesh.n_cells)
    gradient = jnp.ones((mesh.n_cells, 2))

    np.testing.assert_allclose(closure.value_weight(field, gradient), a, atol=1e-15)
    np.testing.assert_allclose(closure.gradient_weight(field, gradient), a[:, None] * w, atol=1e-15)

    at_rest = closure.linearization(mesh.n_cells, mesh.dim)
    np.testing.assert_allclose(at_rest.value_weight, a, atol=1e-15)
    np.testing.assert_allclose(at_rest.gradient_weight, a[:, None] * w, atol=1e-15)


class _Recording(GradientScheme):
    """Returns a fixed gradient and records what it was handed."""

    answer: jnp.ndarray

    def _reconstruct_gradient(self, field, mesh, geometry, boundary_values, **kwargs):
        _Recording.seen = (boundary_values, kwargs)
        return self.answer


def test_the_reconstruction_is_fed_leading_order_values_and_returns_corrected_ones() -> None:
    mesh = structured_grid_2d(3, 2)
    geometry = mesh.geometry()
    closure, _, _ = _closure(mesh)
    field = jnp.linspace(-1.0, 1.0, mesh.n_cells)
    answer = jnp.stack([jnp.arange(mesh.n_cells) * 1.0, -jnp.ones(mesh.n_cells)], axis=1)
    zero = jnp.zeros((mesh.n_cells, 2))

    gradient, boundary_values = closure.reconstruct(_Recording(answer), field, mesh, geometry)

    fed, kwargs = _Recording.seen
    np.testing.assert_array_equal(fed, closure.values(field, zero))
    np.testing.assert_array_equal(
        kwargs["boundary_values_at"](answer), closure.values(field, answer)
    )
    np.testing.assert_array_equal(
        kwargs["boundary_gradient_weight"], closure.gradient_weight(field, zero)
    )
    np.testing.assert_array_equal(gradient, answer)
    np.testing.assert_array_equal(boundary_values, closure.values(field, answer))
    # The corrected values differ from the leading-order ones wherever a face follows its owner.
    assert not np.array_equal(boundary_values, fed)


def test_no_scheme_reconstructs_a_zero_gradient() -> None:
    mesh = structured_grid_2d(3, 2)
    closure, _, _ = _closure(mesh)
    field = jnp.linspace(-1.0, 1.0, mesh.n_cells)

    gradient, boundary_values = closure.reconstruct(None, field, mesh, mesh.geometry())

    np.testing.assert_array_equal(gradient, jnp.zeros((mesh.n_cells, 2)))
    np.testing.assert_array_equal(boundary_values, closure.values(field, gradient))
