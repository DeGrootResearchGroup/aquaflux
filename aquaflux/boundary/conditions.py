"""Weak boundary conditions as face-value closures.

A boundary condition is modelled as a *special face interpolator*: it supplies the value
``phi_ip`` at a boundary face, which the diffusion flux then consumes exactly as it does an
interpolated interior face value. The condition is imposed **weakly**, through that
boundary-face flux, rather than by strongly absorbing it into the matrix — the form that
composes naturally with the residual substrate and with automatic differentiation.

Each closure is written in terms of the owner cell value ``phi_P``, the owner cell gradient
``grad phi_P`` (for the non-orthogonal correction), the displacement ``d = x_ip - x_P`` from
the owner centroid to the face centroid, the owner-outward unit normal ``n``, and the
diffusion coefficient ``Gamma_P``. The three flux-type closures share the tangential correction

    corr = grad phi_P . (d - (d . n) n)

which extrapolates the owner value along the face-tangential offset; it vanishes when the
face centroid lies on the cell-centroid normal (an orthogonal grid), so on orthogonal
meshes every closure below reduces to its gradient-free form.

The closures (``a = d . n`` is the normal distance owner-centroid → face):

======================  ===========================================================
Dirichlet (value)       ``phi_ip = value``
Dirichlet (field)       ``phi_ip = field_fn(x_ip)``
Zero-gradient           ``phi_ip = phi_P + corr``
Neumann (flux ``q``)    ``phi_ip = phi_P + corr - (q / Gamma_P) a``
Convective (h, Tinf)    ``phi_ip = (phi_P + corr + (h/Gamma_P) a Tinf) / (1 + (h/Gamma_P) a)``
======================  ===========================================================

The convective closure enforces ``Gamma_P dphi/dn = h (Tinf - phi_ip)`` at the face — the
Robin balance between diffusive and convective flux — and is the one that carries the Biot
number (``h`` non-dimensionalized), so it is the differentiation target for a sensitivity
with respect to ``Bi``.

**Every prescribed number is a floating array leaf.** A closure converts its coefficients with
:func:`as_float_leaf` when it is constructed, whatever the caller passes. Transformations that
differentiate a pytree -- ``equinox.filter_grad``, and the implicit-function-theorem adjoint of a
solve that carries the assembler as its parameters -- see only floating array leaves, so a
coefficient kept as a Python float or an integer would receive no cotangent, and its sensitivity
would be missing rather than wrong. A position-dependent value is normalized by
:func:`as_position_function`: an ``equinox.Module`` keeps its array fields as leaves, which is how a
profile's coefficients become differentiable, while a plain function is held static in
:class:`StaticFunction`, since a function cannot be a leaf of an array pytree.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, scale


def as_float_leaf(value) -> jnp.ndarray:
    """A prescribed boundary number as a floating JAX array, the form a gradient can reach.

    Parameters
    ----------
    value : float, int or array_like
        The number (or a traced value) to store; a scalar for the closures here.

    Returns
    -------
    jnp.ndarray
        ``value`` as a floating-point array.
    """
    return jnp.asarray(value, dtype=float)


class StaticFunction(eqx.Module):
    """A plain function held as static structure, so a closure holding it stays a valid pytree.

    A function is not an array, so it cannot ride as a pytree leaf: an adjoint that carries the
    closure as its parameters, or a stack of closures across mesh partitions, would reject it. Held
    static it is part of the tree's structure instead, compared by identity. Whatever it captures is
    therefore invisible to a pytree gradient; to make a profile's coefficients differentiable, pass
    an ``equinox.Module`` whose fields hold them.

    Attributes
    ----------
    function : callable
        The wrapped function (static).
    """

    function: Callable = eqx.field(static=True)

    def __call__(self, *args):
        """Evaluate the wrapped function."""
        return self.function(*args)


def as_position_function(function) -> eqx.Module:
    """A position-dependent boundary value, normalized so a closure holding it is a valid pytree.

    Parameters
    ----------
    function : callable
        A function of face centroids. An ``equinox.Module`` is kept as it is, so its array fields are
        leaves a gradient reaches; a plain function is wrapped in :class:`StaticFunction`.

    Returns
    -------
    equinox.Module
        A callable module evaluating ``function``.

    Raises
    ------
    TypeError
        If ``function`` is not callable.
    """
    if isinstance(function, eqx.Module):
        return function
    if not callable(function):
        raise TypeError(f"expected a callable of face centroids, got {type(function).__name__}")
    return StaticFunction(function)


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


#: What a closure covering exactly one field -- its host equation's -- declares from
#: :meth:`BoundaryCondition.closes`. Empty because the field is the *assembler's*: the closure is
#: used by whichever equation holds it and never names that equation itself.
HOST_EQUATION_FIELD: tuple[str, ...] = ()


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

    def closes(self) -> tuple[str, ...]:
        """Which fields this closure closes: :data:`HOST_EQUATION_FIELD` -- one, its host equation's.

        Declared rather than inferred, so an assembler can say what it needs closed and be refused
        by name when it is handed something else. Without it the mismatch surfaces as an
        ``AttributeError`` for whichever method the wrong family happens to lack, which names an
        internal method rather than the mistake and does not say which patch is wrong.

        Empty for this family on purpose: a single-field closure is used by whichever equation holds
        it and does not know that equation's name. Writing a name here would be a second copy of
        something the *assembler* already owns, free to disagree with it -- the same reasoning as
        :meth:`requires_coefficient`, which is parameterized by a coefficient name the assembler
        holds.

        Returns
        -------
        tuple of str
            :data:`HOST_EQUATION_FIELD`.
        """
        return HOST_EQUATION_FIELD

    def requires_coefficient(self) -> bool:
        """Whether this closure reads the assembler's diffusion coefficient as ``gamma_owner``.

        Default ``False``. A closure that needs it declares so here rather than the assembler
        type-testing for ``Neumann``/``Convective``: the requirement is a property of the closure,
        parameterized by a name the *assembler* owns (its ``coefficient``), not one the closure
        holds itself -- the same shape as
        :meth:`~aquaflux.discretization.face_flux.FaceFluxOperator.requires`.
        :meth:`~aquaflux.discretization.residual.ResidualAssembler.build` checks this against every
        closure in the boundary set and adds the assembler's ``coefficient`` to what it requires the
        properties to supply, so a closure whose coefficient is unset fails at build time rather
        than reading a zero fallback and NaN-ing (a divide by zero) inside the residual.
        """
        return False

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
    value : jnp.ndarray
        The imposed face value (any number given is stored as a floating array).
    """

    value: jnp.ndarray = eqx.field(converter=as_float_leaf)

    def face_value(self, phi_owner, grad_owner, d, normal, gamma_owner, face_centroid):
        return jnp.full(phi_owner.shape, self.value)


class DirichletField(BoundaryCondition):
    """Prescribed spatially-varying value ``phi_ip = field_fn(x_ip)`` at the face centroid.

    The position-dependent generalization of :class:`Dirichlet` — used for manufactured
    solutions and any wall whose imposed value varies along the patch.

    Attributes
    ----------
    field_fn : equinox.Module
        Maps face centroids ``(n, dim)`` to imposed values ``(n,)``. Given as an
        ``equinox.Module``, its array fields are differentiable leaves (a profile's amplitude or
        position, say); a plain function is accepted and held static, and is then not
        differentiable (see :func:`as_position_function`).
    """

    field_fn: eqx.Module = eqx.field(converter=as_position_function)

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
    flux : jnp.ndarray
        The imposed outward diffusive flux density ``q`` (stored as a floating array).
    """

    flux: jnp.ndarray = eqx.field(converter=as_float_leaf)

    def requires_coefficient(self) -> bool:
        return True

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
    h : jnp.ndarray
        Exchange coefficient (the Biot number, non-dimensionalized), stored as a floating array.
    t_inf : jnp.ndarray
        Ambient value the boundary exchanges with, stored as a floating array.
    """

    h: jnp.ndarray = eqx.field(converter=as_float_leaf)
    t_inf: jnp.ndarray = eqx.field(converter=as_float_leaf)

    def requires_coefficient(self) -> bool:
        return True

    def face_value(self, phi_owner, grad_owner, d, normal, gamma_owner, face_centroid):
        d_normal = dot(d, normal)
        corr = _tangential_correction(grad_owner, d, normal)
        beta = self.h / gamma_owner * d_normal
        return (phi_owner + corr + beta * self.t_inf) / (1.0 + beta)
