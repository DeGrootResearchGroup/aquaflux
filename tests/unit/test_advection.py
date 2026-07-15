"""Unit tests for the advection operator and its upwind schemes."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.discretization import (
    AdvectionFlux,
    FaceContext,
    FirstOrderUpwind,
    LimitedUpwind,
)
from aquaflux.mesh import CellGeometry, FaceCellConnectivity, FaceGeometry, MeshGeometry
from aquaflux.schemes import Limiter

from tests.support.fields import face_mass_flux


class _ZeroLimiter(Limiter):
    """A stub limiter returning ``psi = 0`` (full clipping) — probes the reconstruction only."""

    def limit(self, field, gradient, face_cells, geometry):
        return jnp.zeros(field.shape[0], dtype=field.dtype)


def _single_face(phi_owner, phi_neighbour, *, grad=(0.0, 0.0), boundary_value=0.0, interior=True):
    """A one-face ``(field, FaceContext)``: owner centroid ``(0.5, 0)``, neighbour ``(1.5, 0)``,
    face centroid ``(1, 0)``, ``n = x``, area ``2``."""
    n_cells = 2 if interior else 1
    neighbour = 1 if interior else -1
    field = jnp.array([phi_owner, phi_neighbour][:n_cells])
    gradient = jnp.array([list(grad)] * n_cells)
    geometry = MeshGeometry(
        face=FaceGeometry(
            area=jnp.array([2.0]),
            centroid=jnp.array([[1.0, 0.0]]),
            normal=jnp.array([[1.0, 0.0]]),
        ),
        cell=CellGeometry(
            volume=jnp.ones(n_cells),
            centroid=jnp.array([[0.5, 0.0], [1.5, 0.0]][:n_cells]),
        ),
    )
    context = FaceContext(
        face_cells=FaceCellConnectivity(jnp.array([0]), jnp.array([neighbour]), n_cells=n_cells),
        geometry=geometry,
        boundary_values=jnp.array([boundary_value]),
        gradient=gradient,
        properties={},  # the advection schemes read no property
    )
    return field, context


def test_upwind_interior_takes_owner_when_flow_leaves_owner() -> None:
    field, context = _single_face(3.0, 7.0)
    phi = FirstOrderUpwind().face_value(field, context, jnp.array([1.0]))
    assert float(phi[0]) == 3.0  # mdot > 0: upwind is the owner


def test_upwind_interior_takes_neighbour_when_flow_enters_owner() -> None:
    field, context = _single_face(3.0, 7.0)
    phi = FirstOrderUpwind().face_value(field, context, jnp.array([-1.0]))
    assert float(phi[0]) == 7.0  # mdot < 0: upwind is the neighbour


def test_upwind_boundary_inflow_takes_boundary_value() -> None:
    """Inflow at a boundary uses the prescribed (weak) inlet value, not the owner."""
    field, context = _single_face(3.0, 0.0, boundary_value=9.0, interior=False)
    phi = FirstOrderUpwind().face_value(field, context, jnp.array([-1.0]))
    assert float(phi[0]) == 9.0


def test_upwind_boundary_outflow_takes_owner_value() -> None:
    """Outflow at a boundary is upwinded from the owner (no value need be imposed)."""
    field, context = _single_face(3.0, 0.0, boundary_value=9.0, interior=False)
    phi = FirstOrderUpwind().face_value(field, context, jnp.array([1.0]))
    assert float(phi[0]) == 3.0


def test_limited_upwind_reconstructs_from_upwind_gradient() -> None:
    """phi_f = phi_C + psi grad_C . (x_f - x_C); outflow -> C = owner, x_f - x_owner = (0.5, 0).

    With no limiter injected psi = 1 (unlimited linear upwind)."""
    field, context = _single_face(3.0, 7.0, grad=(2.0, 0.0))
    phi = LimitedUpwind().face_value(field, context, jnp.array([1.0]))
    assert abs(float(phi[0]) - (3.0 + 2.0 * 0.5)) < 1e-12  # 3 + grad_x * dx


def test_limited_upwind_psi_zero_reduces_to_first_order() -> None:
    """psi = 0 removes the gradient term, recovering the first-order upwind value."""
    field, context = _single_face(3.0, 7.0, grad=(2.0, 0.0))
    phi = LimitedUpwind(limiter=_ZeroLimiter()).face_value(field, context, jnp.array([1.0]))
    assert float(phi[0]) == 3.0


def test_advection_flux_is_mdot_times_face_value() -> None:
    field, context = _single_face(3.0, 7.0)
    mdot = jnp.array([2.5])
    flux = AdvectionFlux(mass_flux=mdot, scheme=FirstOrderUpwind()).face_flux(field, context)
    assert float(flux[0]) == 2.5 * 3.0  # mdot > 0 -> owner value


def test_face_mass_flux_projects_velocity_onto_normal() -> None:
    """mdot = (u . n) A for a uniform velocity."""
    fg = FaceGeometry(
        area=jnp.array([2.0, 3.0]),
        centroid=jnp.zeros((2, 2)),
        normal=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    mdot = face_mass_flux(fg, jnp.array([4.0, 5.0]))
    assert jnp.allclose(mdot, jnp.array([4.0 * 2.0, 5.0 * 3.0]))
