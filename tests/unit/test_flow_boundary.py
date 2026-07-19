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


VISCOUS = jnp.array([7.0])  # a Dirichlet viscous diagonal mu*A/(d.n)
CONVECTIVE = jnp.array([5.0])  # an upwind convective diagonal max(mdot, 0)


def test_no_slip_wall() -> None:
    bc = NoSlipWall()
    assert jnp.allclose(bc.velocity_face(VEL, NORMAL, CENTROID), 0.0)
    assert jnp.allclose(bc.pressure_face(P), P)  # zero-gradient
    assert float(bc.mass_flux(VEL, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, 1.0)[0]) == 0.0


def test_momentum_diagonal_coefficient_per_patch() -> None:
    """Each patch's a_P owner contribution matches the operator it imposes (issue #41).

    A wall passes no fluid, so it contributes only the Dirichlet viscous diagonal, never the
    convective one (4b); a zero-gradient pressure outlet imposes no velocity, so its viscous flux
    vanishes and it contributes only the outflow convective diagonal (4a); a velocity inlet is a
    through-flow Dirichlet patch and contributes both.
    """
    # Walls: viscous only -- the spurious wall convective term is dropped.
    assert float(NoSlipWall().momentum_diagonal_coefficient(VISCOUS, CONVECTIVE)[0]) == 7.0
    assert (
        float(MovingWall(velocity=(1.0, 0.0)).momentum_diagonal_coefficient(VISCOUS, CONVECTIVE)[0])
        == 7.0
    )
    # Pressure outlet: convective only -- the spurious outlet viscous term is dropped.
    assert (
        float(PressureOutlet(pressure=0.0).momentum_diagonal_coefficient(VISCOUS, CONVECTIVE)[0])
        == 5.0
    )
    # Velocity inlet: both (the base through-flow Dirichlet behaviour).
    assert (
        float(
            VelocityInlet(velocity=(1.0, 0.0)).momentum_diagonal_coefficient(VISCOUS, CONVECTIVE)[0]
        )
        == 12.0
    )


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


def test_only_prescribing_patches_declare_a_reference_velocity() -> None:
    """A patch reports the velocity it *imposes* on the flow, not the velocity it happens to see.

    This is the characteristic scale a convection-aware momentum block sizes its frozen convective
    linearization from, so it must come from patches that drive the flow (an inlet, a moving wall)
    and not from one that merely responds to it.
    """
    inlet = VelocityInlet(velocity=(4.0, 0.0)).reference_velocity(NORMAL, CENTROID)
    assert jnp.allclose(inlet, jnp.array([[4.0, 0.0]]))
    lid = MovingWall(velocity=(1.0, 0.0)).reference_velocity(NORMAL, CENTROID)
    assert jnp.allclose(lid, jnp.array([[1.0, 0.0]]))
    # A stationary wall drives nothing, and an outlet prescribes no velocity at all — even though its
    # own face velocity is the (non-zero) owner value it sees, which is a response, not a scale.
    assert jnp.allclose(NoSlipWall().reference_velocity(NORMAL, CENTROID), 0.0)
    assert jnp.allclose(PressureOutlet(pressure=0.0).reference_velocity(NORMAL, CENTROID), 0.0)


def test_reference_velocity_follows_an_inlet_profile() -> None:
    """A spatially-varying inlet reports its profile, evaluated per face centroid."""
    bc = VelocityInlet(velocity=lambda x: jnp.stack([x[:, 1], jnp.zeros(x.shape[0])], axis=1))
    centroid = jnp.array([[0.0, 0.25], [0.0, 0.75]])
    normal = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    assert jnp.allclose(
        bc.reference_velocity(normal, centroid), jnp.array([[0.25, 0.0], [0.75, 0.0]])
    )
