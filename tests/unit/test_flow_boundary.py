"""Unit tests for the coupled-flow boundary conditions."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.flow import MovingWall, NoSlipWall, PressureOutlet, VelocityInlet

VEL = jnp.array([[2.0, 1.0]])  # owner velocity
NORMAL = jnp.array([[1.0, 0.0]])
CENTROID = jnp.array([[0.0, 0.5]])
AREA = jnp.array([2.0])
DN = jnp.array([0.5])
GRADP = jnp.array([[3.0, 0.0]])
DCOEFF = jnp.array([[0.4, 0.4]])  # per-component V/a_P (isotropic)
P = jnp.array([1.5])


def test_no_slip_wall() -> None:
    bc = NoSlipWall()
    assert jnp.allclose(bc.velocity_face(VEL, NORMAL, CENTROID), 0.0)
    assert jnp.allclose(bc.pressure_face(P), P)  # zero-gradient
    assert float(bc.mass_flux(VEL, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, 1.0)[0]) == 0.0


def test_moving_wall() -> None:
    """A moving wall imposes its velocity but passes no fluid (mdot = 0)."""
    bc = MovingWall(velocity=(1.0, 0.0))
    assert jnp.allclose(bc.velocity_face(VEL, NORMAL, CENTROID), jnp.array([[1.0, 0.0]]))
    assert jnp.allclose(bc.pressure_face(P), P)
    assert float(bc.mass_flux(VEL, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, 1.0)[0]) == 0.0


def test_velocity_inlet_constant() -> None:
    bc = VelocityInlet(velocity=(4.0, 0.0))
    assert jnp.allclose(bc.velocity_face(VEL, NORMAL, CENTROID), jnp.array([[4.0, 0.0]]))
    # mdot = rho (u_in . n) A = 1 * 4 * 2
    assert (
        abs(float(bc.mass_flux(VEL, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, 1.0)[0]) - 8.0)
        < 1e-12
    )


def test_velocity_inlet_profile() -> None:
    bc = VelocityInlet(velocity=lambda x: jnp.stack([x[:, 1], jnp.zeros(x.shape[0])], axis=1))
    face = bc.velocity_face(VEL, NORMAL, jnp.array([[0.0, 0.25], [0.0, 0.75]]))
    assert jnp.allclose(face, jnp.array([[0.25, 0.0], [0.75, 0.0]]))


def test_pressure_outlet() -> None:
    bc = PressureOutlet(pressure=0.0)
    assert jnp.allclose(bc.pressure_face(P), 0.0)
    assert jnp.allclose(bc.velocity_face(VEL, NORMAL, CENTROID), VEL)  # zero-gradient velocity
    # mdot = rho (u.n - dcoeff((p_b - p)/dn - gradp.n)) A
    expected = 1.0 * (2.0 - 0.4 * ((0.0 - 1.5) / 0.5 - 3.0)) * 2.0
    got = float(bc.mass_flux(VEL, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, 1.0)[0])
    assert abs(got - expected) < 1e-12
