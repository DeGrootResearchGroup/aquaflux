"""Unit tests for the Rhie--Chow mass flux and momentum diagonal (physics-free)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.flow import interior_mass_flux, momentum_diagonal
from aquaflux.mesh import (
    CellGeometry,
    FaceCellConnectivity,
    FaceGeometry,
    MeshGeometry,
    structured_grid_2d,
)
from aquaflux.schemes.interpolation import interpolation_factor


def _single_face(*, p_owner=0.0, p_neighbour=0.0, grad=(0.0, 0.0), d_coeff=0.5):
    """One interior face between cell 0 (owner, centroid [0,0]) and cell 1 (neighbour, [1,0]).

    Face centroid [0.5, 0] (on the connecting line, so no skewness), n = x, area = 2, d.n = 1;
    owner velocity (3, 1), neighbour velocity (5, 1); uniform velocity gradient set to zero.
    """
    g = jnp.array([list(grad)])
    return dict(
        velocity=jnp.array([[3.0, 1.0], [5.0, 1.0]]),
        grad_velocity=jnp.zeros((2, 2, 2)),
        pressure=jnp.array([p_owner, p_neighbour]),
        grad_pressure=jnp.concatenate([g, g], axis=0),
        d_coeff=jnp.full((2, 2), d_coeff),  # per-component V/a_P (isotropic here)
        face_cells=FaceCellConnectivity(jnp.array([0]), jnp.array([1]), n_cells=2),
        geometry=MeshGeometry(
            face=FaceGeometry(
                area=jnp.array([2.0]),
                centroid=jnp.array([[0.5, 0.0]]),
                normal=jnp.array([[1.0, 0.0]]),
            ),
            cell=CellGeometry(
                volume=jnp.ones(2),
                centroid=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
            ),
        ),
        interp_factor=jnp.array([0.5]),
        normal_distance=jnp.array([1.0]),
        rho=jnp.ones(2),
    )


def test_mass_flux_uniform_pressure_is_interpolated_velocity_flux() -> None:
    """With no pressure variation the Rhie--Chow correction vanishes: mdot = rho (u_f . n) A."""
    mdot = interior_mass_flux(**_single_face())
    u_face = 0.5 * (3.0 + 5.0)  # interpolated u; n = x so only u contributes
    assert abs(float(mdot[0]) - 1.0 * u_face * 2.0) < 1e-12


def test_mass_flux_adverse_pressure_gradient_reduces_flux() -> None:
    """A higher downstream pressure (p_N > p_P) drives the Rhie--Chow term to reduce the flux."""
    mdot = interior_mass_flux(**_single_face(p_owner=0.0, p_neighbour=1.0))
    baseline = interior_mass_flux(**_single_face())
    assert float(mdot[0]) < float(baseline[0])


def test_mass_flux_correction_cancels_when_compact_equals_interpolated() -> None:
    """When the compact and reconstructed pressure jumps agree, the Rhie--Chow correction is zero."""
    # compact jump p_N - p_P = 1 equals the reconstructed jump grad . d = 1 (d = [1, 0]).
    mdot = interior_mass_flux(**_single_face(p_owner=0.0, p_neighbour=1.0, grad=(1.0, 0.0)))
    baseline = interior_mass_flux(**_single_face())
    assert abs(float(mdot[0]) - float(baseline[0])) < 1e-12


def test_momentum_diagonal_positive_and_scales_with_viscosity() -> None:
    mesh = structured_grid_2d(4, 4)
    geom = mesh.geometry()
    owner = mesh.face_cells.owner
    nb = jnp.where(mesh.face_cells.neighbour >= 0, mesh.face_cells.neighbour, owner)
    d = geom.cell.centroid[nb] - geom.cell.centroid[owner]
    dn = jnp.where(
        mesh.face_cells.neighbour >= 0,
        jnp.sum(d * geom.face.normal, axis=1),
        jnp.sum((geom.face.centroid - geom.cell.centroid[owner]) * geom.face.normal, axis=1),
    )
    g = interpolation_factor(mesh.face_cells, geom)
    a1 = momentum_diagonal(mesh.face_cells, geom, jnp.ones(mesh.n_cells), dn, g)
    a2 = momentum_diagonal(mesh.face_cells, geom, 2.0 * jnp.ones(mesh.n_cells), dn, g)
    assert bool(jnp.all(a1 > 0.0))
    assert jnp.allclose(a2, 2.0 * a1)  # viscous diagonal is linear in mu
