"""Weak boundary conditions as face-value closures.

A boundary condition is modelled as a *special face interpolator*: it supplies the value
``phi_ip`` at a boundary face, which the diffusion flux then consumes exactly as it does an
interpolated interior face value. The condition is imposed **weakly**, through that
boundary-face flux, rather than by strongly absorbing it into the matrix — the form that
composes naturally with the residual substrate and with automatic differentiation.

Each closure is written in terms of the owner cell value ``phi_P``, the owner cell gradient
``grad phi_P`` (for the non-orthogonal correction), the displacement ``d = x_ip - x_P`` from
the owner centroid to the face centroid, the owner-outward unit normal ``n``, and the
diffusion coefficient ``Gamma_P``. All share the tangential correction

    corr = grad phi_P . (d - (d . n) n)

which extrapolates the owner value along the face-tangential offset; it vanishes when the
face centroid lies on the cell-centroid normal (an orthogonal grid), so on orthogonal
meshes every closure below reduces to its gradient-free form.

The four closures (``a = d . n`` is the normal distance owner-centroid → face):

======================  ===========================================================
Dirichlet (value)       ``phi_ip = value``
Zero-gradient           ``phi_ip = phi_P + corr``
Neumann (flux ``q``)    ``phi_ip = phi_P + corr - (q / Gamma_P) a``
Convective (h, Tinf)    ``phi_ip = (phi_P + corr + (h/Gamma_P) a Tinf) / (1 + (h/Gamma_P) a)``
======================  ===========================================================

The convective closure enforces ``Gamma_P dphi/dn = h (Tinf - phi_ip)`` at the face — the
Robin balance between diffusive and convective flux — and is the one that carries the Biot
number (``h`` non-dimensionalized), so it is the differentiation target for a sensitivity
with respect to ``Bi``.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, scale


def _tangential_correction(
    grad_owner: jnp.ndarray, d: jnp.ndarray, normal: jnp.ndarray
) -> jnp.ndarray:
    """``grad phi_P . (d - (d.n) n)`` per face — the non-orthogonal face-value correction.

    Parameters
    ----------
    grad_owner : jnp.ndarray
        Owner cell gradient per face, shape ``(n, dim)``.
    d : jnp.ndarray
        Owner-centroid → face-centroid displacement per face, shape ``(n, dim)``.
    normal : jnp.ndarray
        Owner-outward unit normal per face, shape ``(n, dim)``.

    Returns
    -------
    jnp.ndarray
        Correction per face, shape ``(n,)``. Zero when ``d`` is parallel to ``normal``.
    """
    tangential = d - scale(normal, dot(d, normal))
    return dot(grad_owner, tangential)


class BoundaryCondition(eqx.Module):
    """Strategy interface: a weak boundary face-value closure.

    A concrete condition returns the boundary-face value ``phi_ip`` for every face in its
    patch, given per-face owner-cell state and face geometry. It shares the interface of an
    interior face interpolator, so the flux operator consumes boundary and interior faces
    uniformly.

    A closure covers a **single** field. A system that solves several coupled fields (a
    velocity--pressure system, say) is expressed by composing one such closure per field per
    patch, rather than by a separate, parallel boundary hierarchy.
    """

    @abc.abstractmethod
    def face_value(
        self,
        phi_owner: jnp.ndarray,
        grad_owner: jnp.ndarray,
        d: jnp.ndarray,
        normal: jnp.ndarray,
        gamma_owner: jnp.ndarray,
        face_centroid: jnp.ndarray,
    ) -> jnp.ndarray:
        """Boundary-face values ``phi_ip``, shape ``(n,)`` (one per patch face).

        Parameters
        ----------
        phi_owner : jnp.ndarray
            Owner cell values, shape ``(n,)``.
        grad_owner : jnp.ndarray
            Owner cell gradients, shape ``(n, dim)`` (used by the non-orthogonal correction).
        d : jnp.ndarray
            Owner-centroid → face-centroid displacement, shape ``(n, dim)``.
        normal : jnp.ndarray
            Owner-outward unit normals, shape ``(n, dim)``.
        gamma_owner : jnp.ndarray
            Owner diffusion coefficients, shape ``(n,)``.
        face_centroid : jnp.ndarray
            Face centroids, shape ``(n, dim)`` (for spatially-varying closures).
        """


class Dirichlet(BoundaryCondition):
    """Prescribed value: ``phi_ip = value`` (a fixed-temperature / fixed-concentration wall).

    Attributes
    ----------
    value : float
        The imposed face value.
    """

    value: float

    def face_value(self, phi_owner, grad_owner, d, normal, gamma_owner, face_centroid):
        return jnp.full(phi_owner.shape, self.value)


class DirichletField(BoundaryCondition):
    """Prescribed spatially-varying value ``phi_ip = field_fn(x_ip)`` at the face centroid.

    The position-dependent generalization of :class:`Dirichlet` — used for manufactured
    solutions and any wall whose imposed value varies along the patch.

    Attributes
    ----------
    field_fn : callable
        Maps face centroids ``(n, dim)`` to imposed values ``(n,)``. Static (not
        differentiated); a pure function of position.
    """

    field_fn: object = eqx.field(static=True)

    def face_value(self, phi_owner, grad_owner, d, normal, gamma_owner, face_centroid):
        return self.field_fn(face_centroid)


class ZeroGradient(BoundaryCondition):
    """Zero normal gradient: ``phi_ip = phi_P + corr`` (symmetry / adiabatic / outflow).

    The normal derivative at the face is zero, so the face value equals the owner value plus
    only the tangential (non-orthogonal) correction.
    """

    def face_value(self, phi_owner, grad_owner, d, normal, gamma_owner, face_centroid):
        return phi_owner + _tangential_correction(grad_owner, d, normal)


class Neumann(BoundaryCondition):
    """Prescribed diffusive flux ``q``: ``phi_ip = phi_P + corr - (q / Gamma_P)(d.n)``.

    Sign convention: ``q = -Gamma_P dphi/dn`` (the outward diffusive flux density), so a
    positive ``flux`` removes ``phi`` through the boundary.

    Attributes
    ----------
    flux : float
        The imposed outward diffusive flux density ``q``.
    """

    flux: float

    def face_value(self, phi_owner, grad_owner, d, normal, gamma_owner, face_centroid):
        d_normal = dot(d, normal)
        corr = _tangential_correction(grad_owner, d, normal)
        return phi_owner + corr - self.flux / gamma_owner * d_normal


class Convective(BoundaryCondition):
    """Convective (Robin) exchange with an ambient value ``Tinf`` through coefficient ``h``.

    Enforces ``Gamma_P dphi/dn = h (Tinf - phi_ip)`` at the face, giving

        phi_ip = (phi_P + corr + (h/Gamma_P)(d.n) Tinf) / (1 + (h/Gamma_P)(d.n)).

    In the non-dimensional plane-wall problem (``Gamma_P = 1``, unit half-thickness) ``h`` is
    the Biot number, so this closure is the sensitivity-differentiation target.

    Attributes
    ----------
    h : float
        Exchange coefficient (the Biot number, non-dimensionalized).
    t_inf : float
        Ambient value the boundary exchanges with.
    """

    h: float
    t_inf: float

    def face_value(self, phi_owner, grad_owner, d, normal, gamma_owner, face_centroid):
        d_normal = dot(d, normal)
        corr = _tangential_correction(grad_owner, d, normal)
        beta = self.h / gamma_owner * d_normal
        return (phi_owner + corr + beta * self.t_inf) / (1.0 + beta)
