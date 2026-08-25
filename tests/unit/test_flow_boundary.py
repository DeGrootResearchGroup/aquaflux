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

# The face-value closures also take the owner's reconstructed gradient and the owner-centroid →
# face-centroid displacement d, which together carry the tangential non-orthogonal correction
# grad phi_P . (d - (d.n) n). Orthogonal here (d parallel to the normal), so every closure reduces
# to its gradient-free form and these constants leave the values below unchanged; the skewed case
# is exercised separately.
D_ORTHOGONAL = jnp.array([[0.5, 0.0]])
NO_GRADIENT_P = jnp.zeros((1, 2))  # a zero cell pressure gradient
NO_GRADIENT_U = jnp.zeros((1, 2, 2))  # a zero cell velocity-gradient tensor

# A skewed face: the same unit normal, but the owner centroid sits off the face normal, so
# d - (d.n) n = (0, 0.25) is a genuine tangential offset.
D_SKEWED = jnp.array([[0.5, 0.25]])


VISCOUS = jnp.array([7.0])  # a Dirichlet viscous diagonal mu*A/(d.n)
CONVECTIVE = jnp.array([5.0])  # an upwind convective diagonal max(mdot, 0)


def test_no_slip_wall() -> None:
    bc = NoSlipWall()
    assert jnp.allclose(bc.velocity_face(VEL, NO_GRADIENT_U, D_ORTHOGONAL, NORMAL, CENTROID), 0.0)
    assert jnp.allclose(
        bc.pressure_face(P, NO_GRADIENT_P, D_ORTHOGONAL, NORMAL, CENTROID), P
    )  # zero-gradient
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
    assert jnp.allclose(
        bc.velocity_face(VEL, NO_GRADIENT_U, D_ORTHOGONAL, NORMAL, CENTROID),
        jnp.array([[1.0, 0.0]]),
    )
    assert jnp.allclose(bc.pressure_face(P, NO_GRADIENT_P, D_ORTHOGONAL, NORMAL, CENTROID), P)
    assert float(bc.mass_flux(VEL, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, 1.0)[0]) == 0.0


def test_velocity_inlet_constant() -> None:
    bc = VelocityInlet(velocity=(4.0, 0.0))
    assert jnp.allclose(
        bc.velocity_face(VEL, NO_GRADIENT_U, D_ORTHOGONAL, NORMAL, CENTROID),
        jnp.array([[4.0, 0.0]]),
    )
    # mdot = rho (u_in . n) A = 1 * 4 * 2
    assert (
        abs(float(bc.mass_flux(VEL, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, 1.0)[0]) - 8.0)
        < 1e-12
    )


def test_velocity_inlet_profile() -> None:
    bc = VelocityInlet(velocity=lambda x: jnp.stack([x[:, 1], jnp.zeros(x.shape[0])], axis=1))
    two_faces = jnp.array([[0.0, 0.25], [0.0, 0.75]])
    face = bc.velocity_face(
        jnp.repeat(VEL, 2, axis=0),
        jnp.zeros((2, 2, 2)),
        jnp.repeat(D_ORTHOGONAL, 2, axis=0),
        jnp.repeat(NORMAL, 2, axis=0),
        two_faces,
    )
    assert jnp.allclose(face, jnp.array([[0.25, 0.0], [0.75, 0.0]]))


def test_pressure_outlet() -> None:
    bc = PressureOutlet(pressure=0.0)
    assert jnp.allclose(bc.pressure_face(P, NO_GRADIENT_P, D_ORTHOGONAL, NORMAL, CENTROID), 0.0)
    assert jnp.allclose(
        bc.velocity_face(VEL, NO_GRADIENT_U, D_ORTHOGONAL, NORMAL, CENTROID), VEL
    )  # zero-gradient velocity
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


def test_a_gradient_type_patch_carries_the_tangential_correction_on_a_skewed_face() -> None:
    """A zero-gradient closure returns ``phi_P + grad phi_P . (d - (d.n) n)``, not the bare ``phi_P``.

    The whole content of a gradient-type condition is that correction: with the owner centroid off
    the face normal, the face value differs from the owner value by the field's variation along the
    tangential offset, and reporting the owner value instead asserts a rise of zero across a
    non-zero displacement. What consumes the difference is a scheme that *differences* the boundary
    value and divides by the wall-normal distance, which then reads a normal derivative the field
    does not have.
    """
    grad_u = jnp.array([[[0.0, 3.0], [0.0, -2.0]]])  # rows: grad u_0, grad u_1
    outlet = PressureOutlet(pressure=0.0)
    face = outlet.velocity_face(VEL, grad_u, D_SKEWED, NORMAL, CENTROID)
    # component i rises by grad u_i . tangent = 3*0.25 and -2*0.25
    assert jnp.allclose(face, VEL + jnp.array([[0.75, -0.5]]))
    # The same face with an orthogonal displacement carries no correction at all.
    assert jnp.allclose(outlet.velocity_face(VEL, grad_u, D_ORTHOGONAL, NORMAL, CENTROID), VEL)

    grad_p = jnp.array([[5.0, 4.0]])
    wall = NoSlipWall()
    assert jnp.allclose(
        wall.pressure_face(P, grad_p, D_SKEWED, NORMAL, CENTROID), P + jnp.array([1.0])
    )  # grad p . tangent = 4 * 0.25
    assert jnp.allclose(wall.pressure_face(P, grad_p, D_ORTHOGONAL, NORMAL, CENTROID), P)


def test_a_prescribed_patch_is_unmoved_by_the_owner_gradient() -> None:
    """A Dirichlet value is the value: no gradient, no displacement, no correction.

    The counterpart to the gradient-type case above — the two-pass fold must leave a prescribed
    velocity or pressure exactly where it was, on any mesh.
    """
    grad_u = jnp.array([[[7.0, 3.0], [1.0, -2.0]]])
    grad_p = jnp.array([[5.0, 4.0]])
    for bc, expected in (
        (NoSlipWall(), jnp.zeros((1, 2))),
        (MovingWall(velocity=(1.0, 0.0)), jnp.array([[1.0, 0.0]])),
        (VelocityInlet(velocity=(4.0, 0.0)), jnp.array([[4.0, 0.0]])),
    ):
        assert jnp.allclose(bc.velocity_face(VEL, grad_u, D_SKEWED, NORMAL, CENTROID), expected)
    outlet = PressureOutlet(pressure=2.5)
    assert jnp.allclose(outlet.pressure_face(P, grad_p, D_SKEWED, NORMAL, CENTROID), 2.5)


def test_an_inlet_mass_flux_reads_its_own_prescribed_velocity() -> None:
    """``VelocityInlet.mass_flux`` is ``rho (u_in . n) A`` at the patch's own prescribed profile.

    It reads the closure rather than a copy of the profile, and a prescribed value depends on
    neither the owner state nor its gradient — so the flux is unchanged by both, on any mesh.
    """
    bc = VelocityInlet(velocity=lambda x: jnp.stack([x[:, 1], jnp.zeros(x.shape[0])], axis=1))
    # u_in = (y_face, 0) = (0.5, 0); mdot = 1 * 0.5 * 2
    flux = bc.mass_flux(VEL, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, jnp.array([1.0]))
    assert abs(float(flux[0]) - 1.0) < 1e-12
    stirred = bc.mass_flux(
        VEL * 3.0, P, GRADP, DCOEFF, NORMAL, AREA, DN, CENTROID, jnp.array([1.0])
    )
    assert abs(float(stirred[0]) - 1.0) < 1e-12
