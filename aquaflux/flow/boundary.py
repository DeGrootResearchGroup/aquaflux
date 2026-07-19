"""Boundary conditions for the coupled pressure--velocity (flow) system.

A flow boundary condition must close three things on its patch faces: the boundary **velocity**
(for the viscous flux and the velocity gradient reconstruction), the boundary **pressure** (for
the pressure-gradient flux and the pressure gradient reconstruction), and the boundary **mass
flux** ``mdot`` that enters continuity and advection. The three standard closures:

======================  ===================  ===================  ================================
condition               velocity             pressure             mass flux ``mdot``
======================  ===================  ===================  ================================
:class:`NoSlipWall`     zero                 zero-gradient        zero (no through-flow)
:class:`VelocityInlet`  prescribed ``u_in``  zero-gradient        ``rho (u_in . n) A`` (given)
:class:`PressureOutlet` zero-gradient        prescribed ``p_b``   Rhie--Chow from ``p_b`` + owner u
======================  ===================  ===================  ================================

Each is an ``equinox.Module`` acting on per-patch-face arrays; the coupled assembler scatters
their contributions into the global velocity/pressure/mass-flux fields.

A flow condition is a *bundle* — a velocity closure, a pressure closure, and a mass-flux
closure — not a single-field closure, so it is not a subtype of the scalar boundary condition:
it returns three coupled quantities, and the mass flux (a Rhie--Chow expression coupling
pressure) has no single-field analogue. Its velocity and pressure parts, however, are not
re-implemented here: they *are* the scalar :class:`~aquaflux.boundary.Dirichlet` and
:class:`~aquaflux.boundary.ZeroGradient` closures applied to each velocity component and to the
pressure. This module composes those closures rather than re-deriving prescribed-value and
zero-gradient face values, so the two paths share one implementation (including the tangential
non-orthogonal correction). Each is evaluated at a **zero reconstructed gradient** — the flow
assembler reconstructs no boundary-face gradient — which drops that correction, giving a
leading-order boundary value: exact on orthogonal grids, leading-order on skewed ones.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp

from aquaflux.boundary import Dirichlet, DirichletField, ZeroGradient
from aquaflux.vectors import dot


def _leading_order_face_value(
    closure, phi_owner: jnp.ndarray, normal: jnp.ndarray, centroid: jnp.ndarray
) -> jnp.ndarray:
    """A scalar closure's boundary value at zero reconstructed gradient — its leading-order value.

    The coupled-flow assembler reconstructs no boundary-face gradient, so the closure is evaluated
    with a zero owner gradient; its tangential non-orthogonal correction then vanishes, leaving the
    value that is exact on orthogonal grids and leading-order on skewed ones. Only the prescribed
    (:class:`~aquaflux.boundary.Dirichlet`) and zero-gradient (:class:`~aquaflux.boundary.ZeroGradient`)
    closures are used here, and their leading-order value is independent of the owner-to-face
    displacement, so that too is passed as zero.

    Parameters
    ----------
    closure : BoundaryCondition
        The scalar face-value closure to evaluate.
    phi_owner : jnp.ndarray
        Owner cell values, shape ``(n,)``.
    normal : jnp.ndarray
        Owner-outward unit normals, shape ``(n, k)`` (``k`` is the spatial dimension for a velocity
        component; a placeholder axis for the pressure, whose closure ignores it here).
    centroid : jnp.ndarray
        Face centroids, shape ``(n, k)`` (used by a spatially-varying prescribed profile).
    """
    zeros = jnp.zeros((*phi_owner.shape, normal.shape[-1]))
    gamma = jnp.ones(phi_owner.shape)
    return closure.face_value(phi_owner, zeros, zeros, normal, gamma, centroid)


def _velocity_face(
    component_closures, velocity_owner: jnp.ndarray, normal: jnp.ndarray, centroid: jnp.ndarray
) -> jnp.ndarray:
    """Boundary velocity ``(n, dim)`` from one scalar closure per velocity component."""
    columns = [
        _leading_order_face_value(closure, velocity_owner[:, i], normal, centroid)
        for i, closure in enumerate(component_closures)
    ]
    return jnp.stack(columns, axis=1)


def _pressure_face(closure, pressure_owner: jnp.ndarray) -> jnp.ndarray:
    """Boundary pressure from a scalar closure, evaluated at leading order.

    The flow path carries no reconstructed pressure gradient at the boundary, so a placeholder
    normal/centroid suffices — the closure's correction is zero at a zero gradient.
    """
    placeholder = jnp.zeros((*pressure_owner.shape, 1))
    return _leading_order_face_value(closure, pressure_owner, placeholder, placeholder)


def _prescribed_reference_velocity(bc, normal: jnp.ndarray, centroid: jnp.ndarray) -> jnp.ndarray:
    """The prescribed face velocity of a patch that imposes one, shape ``(n, dim)``.

    A prescribed velocity is a Dirichlet value: it does not depend on the owner cell, so evaluating
    the patch's own face closure at a quiescent interior returns it exactly — including a
    spatially-varying profile, which is a function of the face centroids alone. Reusing the closure
    keeps the prescribed value defined in one place per patch.
    """
    return bc.velocity_face(jnp.zeros(centroid.shape), normal, centroid)


def _component(profile, i: int):
    """The ``i``-th scalar component of a vector-valued velocity profile ``x -> (n, dim)``."""
    return lambda face_centroid: profile(face_centroid)[:, i]


def _prescribed_components(velocity, dim: int):
    """Per-component scalar closures reproducing a prescribed velocity.

    A callable ``velocity`` is a profile mapping face centroids to velocity vectors, so each
    component becomes a :class:`~aquaflux.boundary.DirichletField`; a constant ``(dim,)`` vector
    becomes one :class:`~aquaflux.boundary.Dirichlet` per component.
    """
    if callable(velocity):
        return [DirichletField(field_fn=_component(velocity, i)) for i in range(dim)]
    # velocity is a static (dim,) sequence — index it directly, keeping each component a concrete
    # scalar (a jnp array here would not concretize under jit).
    return [Dirichlet(value=velocity[i]) for i in range(dim)]


class FlowBoundary(eqx.Module):
    """Strategy interface: velocity, pressure, and mass-flux closures for a flow patch."""

    @abc.abstractmethod
    def velocity_face(
        self, velocity_owner: jnp.ndarray, normal: jnp.ndarray, centroid: jnp.ndarray
    ) -> jnp.ndarray:
        """Boundary face velocity vectors, shape ``(n, dim)``.

        Parameters
        ----------
        velocity_owner : jnp.ndarray
            Owner-cell velocity per face, shape ``(n, dim)``.
        normal : jnp.ndarray
            Owner-outward unit normals, shape ``(n, dim)``.
        centroid : jnp.ndarray
            Face centroids, shape ``(n, dim)`` (for a spatially-varying inlet profile).
        """

    @abc.abstractmethod
    def pressure_face(self, pressure_owner: jnp.ndarray) -> jnp.ndarray:
        """Boundary face pressure, shape ``(n,)``."""

    @abc.abstractmethod
    def mass_flux(
        self,
        velocity_owner: jnp.ndarray,
        pressure_owner: jnp.ndarray,
        grad_pressure_owner: jnp.ndarray,
        d_coeff_owner: jnp.ndarray,
        normal: jnp.ndarray,
        area: jnp.ndarray,
        normal_distance: jnp.ndarray,
        centroid: jnp.ndarray,
        rho: jnp.ndarray,
    ) -> jnp.ndarray:
        """Owner-outward boundary mass flux ``mdot``, shape ``(n,)``.

        Parameters
        ----------
        velocity_owner : jnp.ndarray
            Owner velocity per face, shape ``(n, dim)``.
        pressure_owner : jnp.ndarray
            Owner pressure per face, shape ``(n,)``.
        grad_pressure_owner : jnp.ndarray
            Owner pressure gradient per face, shape ``(n, dim)``.
        d_coeff_owner : jnp.ndarray
            Owner Rhie--Chow coefficient ``V / a_P`` per face, shape ``(n,)``.
        normal : jnp.ndarray
            Owner-outward unit normals, shape ``(n, dim)``.
        area : jnp.ndarray
            Face areas, shape ``(n,)``.
        normal_distance : jnp.ndarray
            Owner-centroid-to-face normal distance ``d . n``, shape ``(n,)``.
        centroid : jnp.ndarray
            Face centroids, shape ``(n, dim)``.
        rho : jnp.ndarray
            Owner-cell density per face, shape ``(n,)``.
        """

    def pressure_schur_coefficient(
        self,
        d_coeff_owner: jnp.ndarray,
        area: jnp.ndarray,
        normal_distance: jnp.ndarray,
        rho_owner: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-face pressure--pressure coupling this patch adds to its owner cell's Schur diagonal.

        The linearization of the patch mass flux with respect to the owner pressure. It is non-zero
        only where the boundary *fixes* the pressure (a :class:`PressureOutlet`), whose Rhie--Chow
        flux drives the owner cell toward the imposed ``p_b`` and so couples ``mdot`` to ``p_owner``;
        a wall or a velocity inlet sets its mass flux independently of pressure, so the base returns
        zero. This is the boundary analogue of the interior Schur face coefficient, and it is what
        makes the open-domain (inlet/outlet) pressure Schur non-singular — its interior part alone is
        a pure-Neumann Laplacian. Consumed only by the frozen SIMPLE preconditioner; it never enters
        the residual.

        Parameters
        ----------
        d_coeff_owner : jnp.ndarray
            Owner Rhie--Chow coefficient ``V / a_P`` per face, shape ``(n,)``.
        area : jnp.ndarray
            Face areas, shape ``(n,)``.
        normal_distance : jnp.ndarray
            Owner-centroid-to-face normal distance ``d . n``, shape ``(n,)``.
        rho_owner : jnp.ndarray
            Owner-cell density per face, shape ``(n,)``.
        """
        return jnp.zeros_like(area)

    def momentum_diagonal_coefficient(
        self, viscous_owner: jnp.ndarray, convective_owner: jnp.ndarray
    ) -> jnp.ndarray:
        """Per-face owner momentum-diagonal contribution this patch adds, shape ``(n,)``.

        The linearization of the patch's boundary *velocity* flux with respect to the owner velocity
        — the momentum sibling of :meth:`pressure_schur_coefficient` (the same idea for the mass flux
        and pressure). A Dirichlet-velocity patch fixes the face velocity independently of the owner,
        so its viscous flux ``mu (u_b − u_owner)/(d·n) A`` contributes ``+mu A/(d·n)`` to the owner
        diagonal (``viscous_owner``); a zero-gradient (outlet) patch sets ``u_b = u_owner``, so the
        viscous flux vanishes and it contributes nothing there. A no-through-flow patch (a wall)
        carries no convective flux, so the upwind convective diagonal ``max(mdot, 0)``
        (``convective_owner``) is zero for it regardless of the owner-velocity estimate. The base is a
        through-flow Dirichlet patch (a velocity inlet): both terms contribute.

        Parameters
        ----------
        viscous_owner : jnp.ndarray
            The Dirichlet viscous diagonal ``mu A/(d·n)`` per face, shape ``(n,)``.
        convective_owner : jnp.ndarray
            The upwind convective diagonal ``max(mdot, 0)`` per face, shape ``(n,)``.
        """
        return viscous_owner + convective_owner

    def reference_velocity(self, normal: jnp.ndarray, centroid: jnp.ndarray) -> jnp.ndarray:
        """The velocity this patch *prescribes* on the flow, shape ``(n, dim)``.

        The patch's contribution to the characteristic velocity scale: what the boundary drives the
        flow with, independent of whatever the interior develops. It is non-zero only where the
        boundary imposes a velocity — a :class:`VelocityInlet` or a :class:`MovingWall` — so the base
        returns zero: a stationary wall drives nothing, and a :class:`PressureOutlet` prescribes no
        velocity at all (its face velocity is whatever reaches it, however fast, which is a *response*
        to the flow rather than a scale imposed on it).

        This is what a convection-aware momentum block sizes its frozen convective linearization
        from, so that linearization carries the operating cell Peclet without being told the flow
        speed. Consumed only by the frozen preconditioner; it never enters the residual.

        Parameters
        ----------
        normal : jnp.ndarray
            Owner-outward unit normals, shape ``(n, dim)``.
        centroid : jnp.ndarray
            Face centroids, shape ``(n, dim)`` (for a spatially-varying prescribed profile).
        """
        return jnp.zeros(centroid.shape)


class NoSlipWall(FlowBoundary):
    """A stationary solid wall: zero velocity, zero-gradient pressure, no through-flow."""

    def velocity_face(self, velocity_owner, normal, centroid):
        dim = velocity_owner.shape[1]
        return _velocity_face([Dirichlet(value=0.0)] * dim, velocity_owner, normal, centroid)

    def pressure_face(self, pressure_owner):
        return _pressure_face(ZeroGradient(), pressure_owner)

    def mass_flux(
        self,
        velocity_owner,
        pressure_owner,
        grad_pressure_owner,
        d_coeff_owner,
        normal,
        area,
        normal_distance,
        centroid,
        rho,
    ):
        return jnp.zeros(area.shape)

    def momentum_diagonal_coefficient(self, viscous_owner, convective_owner):
        # A wall passes no fluid, so it carries no convective diagonal; only the Dirichlet viscous
        # term contributes.
        return viscous_owner


class MovingWall(FlowBoundary):
    """A wall translating in its own plane: prescribed (tangential) velocity, no through-flow.

    Like :class:`NoSlipWall` but with a non-zero wall velocity (e.g. the driven lid of a cavity).
    The mass flux is zero — a wall, however it moves, passes no fluid — so any small normal
    component of the prescribed velocity is ignored for continuity.

    Attributes
    ----------
    velocity : tuple of float or callable
        The wall velocity: a constant ``(dim,)`` vector, or a callable of face centroids. Static.
    """

    velocity: object = eqx.field(static=True)

    def velocity_face(self, velocity_owner, normal, centroid):
        dim = velocity_owner.shape[1]
        return _velocity_face(
            _prescribed_components(self.velocity, dim), velocity_owner, normal, centroid
        )

    def pressure_face(self, pressure_owner):
        return _pressure_face(ZeroGradient(), pressure_owner)

    def mass_flux(
        self,
        velocity_owner,
        pressure_owner,
        grad_pressure_owner,
        d_coeff_owner,
        normal,
        area,
        normal_distance,
        centroid,
        rho,
    ):
        return jnp.zeros(area.shape)

    def momentum_diagonal_coefficient(self, viscous_owner, convective_owner):
        # A moving wall still passes no fluid, so only the Dirichlet viscous term contributes.
        return viscous_owner

    def reference_velocity(self, normal, centroid):
        return _prescribed_reference_velocity(self, normal, centroid)


class VelocityInlet(FlowBoundary):
    """A prescribed-velocity inlet (optionally a profile), with zero-gradient pressure.

    Attributes
    ----------
    velocity : tuple of float or callable
        The inlet velocity: a constant ``(dim,)`` vector, or a callable mapping face centroids
        ``(n, dim)`` to velocity vectors ``(n, dim)`` for a profile (e.g. a parabola). Static.
    """

    velocity: object = eqx.field(static=True)

    def velocity_face(self, velocity_owner, normal, centroid):
        dim = velocity_owner.shape[1]
        return _velocity_face(
            _prescribed_components(self.velocity, dim), velocity_owner, normal, centroid
        )

    def pressure_face(self, pressure_owner):
        return _pressure_face(ZeroGradient(), pressure_owner)

    def mass_flux(
        self,
        velocity_owner,
        pressure_owner,
        grad_pressure_owner,
        d_coeff_owner,
        normal,
        area,
        normal_distance,
        centroid,
        rho,
    ):
        u_in = self.velocity_face(velocity_owner, normal, centroid)
        return rho * dot(u_in, normal) * area

    def reference_velocity(self, normal, centroid):
        return _prescribed_reference_velocity(self, normal, centroid)


class PressureOutlet(FlowBoundary):
    """A prescribed-pressure outlet: zero-gradient velocity, Rhie--Chow mass flux to ``p_b``.

    Attributes
    ----------
    pressure : float
        The imposed boundary pressure ``p_b``.
    """

    pressure: float

    def velocity_face(self, velocity_owner, normal, centroid):
        dim = velocity_owner.shape[1]
        return _velocity_face([ZeroGradient()] * dim, velocity_owner, normal, centroid)

    def pressure_face(self, pressure_owner):
        return _pressure_face(Dirichlet(value=self.pressure), pressure_owner)

    def mass_flux(
        self,
        velocity_owner,
        pressure_owner,
        grad_pressure_owner,
        d_coeff_owner,
        normal,
        area,
        normal_distance,
        centroid,
        rho,
    ):
        # Owner-velocity flux plus a Rhie--Chow correction driving it toward p_b.
        u_normal = dot(velocity_owner, normal)
        compact = (self.pressure - pressure_owner) / normal_distance
        interpolated = dot(grad_pressure_owner, normal)
        d_hat = dot(normal * normal, d_coeff_owner)  # directional V/a_P projected on the normal
        return rho * (u_normal - d_hat * (compact - interpolated)) * area

    def pressure_schur_coefficient(self, d_coeff_owner, area, normal_distance, rho_owner):
        # d(mdot)/d(p_owner) of the mass flux above: the compact term -rho d_hat (p_b - p_owner)/(d.n)
        # contributes +rho d_hat A / (d.n) to the owner's continuity--pressure coupling. The Schur uses
        # an isotropic V/a_P, for which d_hat = V/a_P, matching the interior face coefficient.
        return rho_owner * d_coeff_owner * area / normal_distance

    def momentum_diagonal_coefficient(self, viscous_owner, convective_owner):
        # Zero-gradient velocity: the viscous flux mu(u_owner - u_owner)/(d.n) vanishes, so only the
        # upwind outflow convective diagonal remains.
        return convective_owner
