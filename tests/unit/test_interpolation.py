"""Unit tests for face interpolation: the owner/neighbour blend and integration-point reconstruction."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.mesh import structured_grid_2d
from aquaflux.schemes.interpolation import (
    interpolate_owner_neighbour,
    interpolate_to_face,
    interpolation_factor,
)

from tests.support.meshes import perturbed_grid_2d

A = jnp.array([1.7, -0.9])  # a linear scalar field's constant gradient


def _linear_scalar(x: jnp.ndarray) -> jnp.ndarray:
    return x @ A + 0.4


def test_interpolate_to_face_is_exact_for_a_linear_scalar_on_a_skewed_grid() -> None:
    """The skewness correction makes the face value exact for a linear field on a skewed mesh."""
    mesh = perturbed_grid_2d(6, 6, perturb=0.25, seed=1)
    geom = mesh.geometry()
    f_cells = _linear_scalar(geom.cell.centroid)
    gradient = jnp.broadcast_to(A, (mesh.n_cells, 2))
    g = interpolation_factor(mesh.face_cells, geom)
    face_value = interpolate_to_face(f_cells, gradient, g, mesh.face_cells, geom)
    np.testing.assert_allclose(
        np.asarray(face_value), np.asarray(_linear_scalar(geom.face.centroid)), atol=1e-10
    )


def test_plain_blend_misses_the_skewness_on_a_skewed_grid() -> None:
    """Contrast: the plain owner/neighbour blend lands at x_g, not the face centroid — so it errs."""
    mesh = perturbed_grid_2d(6, 6, perturb=0.25, seed=1)
    geom = mesh.geometry()
    f_cells = _linear_scalar(geom.cell.centroid)
    g = interpolation_factor(mesh.face_cells, geom)
    blend = interpolate_owner_neighbour(f_cells, g, mesh.face_cells)
    exact = _linear_scalar(geom.face.centroid)
    interior = np.asarray(mesh.face_cells.interior)
    err = np.max(np.abs(np.asarray(blend) - np.asarray(exact))[interior])
    assert err > 1e-4  # the correction is genuinely needed on this perturbed grid


def test_interpolate_to_face_is_exact_for_a_linear_vector_field() -> None:
    """Vector field: the gradient's last axis contracts with the offset (grad[i, j] = d u_i / d x_j)."""
    mesh = perturbed_grid_2d(6, 6, perturb=0.25, seed=1)
    geom = mesh.geometry()
    m = jnp.array([[1.2, -0.3], [0.5, 0.8]])
    c = jnp.array([0.2, -0.1])

    def u(x):
        return x @ m.T + c  # u_i = m[i, j] x_j + c_i

    u_cells = u(geom.cell.centroid)
    gradient = jnp.broadcast_to(m, (mesh.n_cells, 2, 2))
    g = interpolation_factor(mesh.face_cells, geom)
    face_value = interpolate_to_face(u_cells, gradient, g, mesh.face_cells, geom)
    np.testing.assert_allclose(
        np.asarray(face_value), np.asarray(u(geom.face.centroid)), atol=1e-10
    )


def test_reduces_to_the_plain_blend_on_interior_faces_of_an_orthogonal_grid() -> None:
    """On an orthogonal grid x_ip = x_g on interior faces, so the correction vanishes there."""
    mesh = structured_grid_2d(5, 4)
    geom = mesh.geometry()
    f_cells = _linear_scalar(geom.cell.centroid)
    gradient = jnp.broadcast_to(A, (mesh.n_cells, 2))
    g = interpolation_factor(mesh.face_cells, geom)
    to_face = interpolate_to_face(f_cells, gradient, g, mesh.face_cells, geom)
    blend = interpolate_owner_neighbour(f_cells, g, mesh.face_cells)
    interior = np.asarray(mesh.face_cells.interior)
    np.testing.assert_allclose(
        np.asarray(to_face)[interior], np.asarray(blend)[interior], atol=1e-12
    )
