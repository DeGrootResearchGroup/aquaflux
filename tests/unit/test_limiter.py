"""Unit tests for the Venkatakrishnan slope limiter (physics-free)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.mesh import structured_grid_2d
from aquaflux.schemes import CorrectedGreenGauss, VenkatakrishnanLimiter


def _psi(field, boundary_values):
    mesh = structured_grid_2d(24, 24)
    geom = mesh.geometry()
    grad = CorrectedGreenGauss().gradients(
        field(geom.cell.centroid), mesh, geom, field(geom.face.centroid)
    )
    psi = VenkatakrishnanLimiter(k=5.0).limit(
        field(geom.cell.centroid), grad, mesh.face_cells, geom
    )
    boundary_cells = set(
        np.asarray(mesh.face_cells.owner)[np.asarray(mesh.face_cells.neighbour) < 0].tolist()
    )
    interior = np.array([c not in boundary_cells for c in range(mesh.n_cells)])
    return np.asarray(psi), interior, np.asarray(geom.cell.centroid)


def _linear(x):
    return 2.0 * x[..., 0] - 3.0 * x[..., 1] + 1.0


def test_limiter_is_one_on_smooth_field() -> None:
    """A smooth (linear) field is reconstructed without limiting: psi -> 1 in the interior."""
    psi, interior, _ = _psi(_linear, _linear)
    assert psi[interior].min() > 0.999


def test_limiter_activates_near_discontinuity() -> None:
    """Near a step the limiter drops below 1; far from it, it is ~1."""

    def step(x):
        return jnp.where(x[..., 0] < 0.5, 1.0, 0.0)

    psi, _, centroid = _psi(step, step)
    near = np.abs(centroid[:, 0] - 0.5) < 0.1
    far = np.abs(centroid[:, 0] - 0.5) > 0.3
    assert psi[near].min() < 0.9
    assert psi[far].min() > 0.99


def test_limiter_stays_in_unit_interval() -> None:
    def step(x):
        return jnp.where(x[..., 0] < 0.5, 1.0, 0.0)

    psi, _, _ = _psi(step, step)
    assert psi.min() >= 0.0
    assert psi.max() <= 1.0 + 1e-12


def test_limiter_is_differentiable() -> None:
    """jax.grad flows through the limiter (min/max and the smooth ratio) without NaNs."""
    mesh = structured_grid_2d(8, 8)
    geom = mesh.geometry()
    scheme = CorrectedGreenGauss()
    limiter = VenkatakrishnanLimiter(k=5.0)

    def loss(field):
        grad = scheme.gradients(field, mesh, geom, jnp.zeros(mesh.n_faces))
        return jnp.sum(limiter.limit(field, grad, mesh.face_cells, geom) ** 2)

    sens = jax.grad(loss)(jnp.sin(geom.cell.centroid[:, 0] * 3.0))
    assert not bool(jnp.any(jnp.isnan(sens)))
