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


def test_limiter_uses_the_periodic_image_across_a_seam() -> None:
    """A smooth periodic field must not be limited at a periodic seam.

    Before the fix, the neighbour side's unlimited increment was formed against the raw
    (non-periodic-image) neighbour centroid, which across a periodic seam sits a full domain
    length away rather than one cell width -- collapsing `psi` toward 0 there for any field, no
    matter how smooth. `face_cells.neighbour_centroid` is the fix: it gathers the neighbour's
    periodic image instead, matching the displacement `LimitedUpwind.face_value` reconstructs on.
    """
    lx = 1.0
    mesh = structured_grid_2d(8, 4, lx=lx, ly=1.0, periodic=("x",), named_boundaries=True)
    geom = mesh.geometry()

    def field(x):
        return jnp.cos(2.0 * jnp.pi * x[..., 0] / lx)

    grad = CorrectedGreenGauss().gradients(
        field(geom.cell.centroid), mesh, geom, field(geom.face.centroid)
    )
    psi = VenkatakrishnanLimiter(k=5.0).limit(
        field(geom.cell.centroid), grad, mesh.face_cells, geom
    )

    fc = mesh.face_cells
    seam_faces = np.asarray(jnp.any(fc.neighbour_offset != 0.0, axis=-1))
    assert seam_faces.any(), "fixture must actually carry a periodic seam"
    seam_cells = set(np.asarray(fc.owner)[seam_faces].tolist()) | set(
        np.asarray(fc.neighbour)[seam_faces].tolist()
    )
    # 0.95 comfortably separates "correctly unlimited" (~0.99, curvature-limited only, matching
    # the non-seam cells) from the pre-fix bug, which on this fixture collapsed the seam cells'
    # psi to ~0.47 by forming the unlimited increment against a raw neighbour centroid a full
    # domain length away instead of the periodic image.
    assert np.asarray(psi)[list(seam_cells)].min() > 0.95


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
