"""One scalar field's boundary face values, as a function of the field and its reconstructed gradient.

A weak boundary condition gives each boundary face a value from its owner cell: a prescribed value
ignores the owner, a zero-gradient or Neumann value extrapolates from it, and on a non-orthogonal
mesh that extrapolation carries a tangential correction ``grad(phi)_P . d_t`` that reads the owner's
**gradient**. The gradient reconstruction, in turn, reads the boundary values. A
:class:`BoundaryClosure` holds that map -- ``(field, gradient) -> boundary face values`` -- and is
the one place both directions of the dependence are resolved:

- :meth:`BoundaryClosure.reconstruct` breaks the circularity in two passes. The reconstruction is
  fed the **leading-order** boundary values (the closures at a zero gradient), which keeps a residual
  built on it a single pass over the field; the returned boundary values are the closures
  re-evaluated at the reconstructed gradient, which is what a face flux consumes. On an orthogonal
  mesh the correction vanishes and the two passes agree exactly.
- :meth:`BoundaryClosure.linearization` reads how each face value depends on its owner -- the
  :class:`~aquaflux.schemes.BoundaryLinearization` a scheme is bound against -- by differentiating
  the map, so it cannot disagree with the conditions it describes.

Any equation whose boundary values are a per-face function of one scalar field composes it: a scalar
transport equation wraps its single-field closures, and a coupled system wraps each scalar it
reconstructs -- the pressure, and each velocity component read through a per-component view of the
vector closures.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .gradient import BoundaryLinearization

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh, MeshGeometry

    from .gradient import GradientScheme, ImposedGradient


@dataclasses.dataclass(frozen=True)
class BoundaryClosure:
    """One scalar field's boundary face values as a function of the field and its gradient.

    Holds a single callable and the operations every consumer of it needs, so an equation supplies
    only how its own conditions are evaluated. It is a transient helper formed inside an assembler's
    method -- the callable usually closes over that assembler -- and is never stored on a pytree.

    Attributes
    ----------
    values : callable
        ``(field, gradient) -> boundary values``: ``field`` the cell values, shape ``(n_cells,)``;
        ``gradient`` the cell gradients, shape ``(n_cells, dim)``; returns the face value of every
        face, shape ``(n_faces,)``, with interior entries unused. Each face value must read only its
        own owner cell's value and gradient -- which is what lets one directional derivative seeded
        in every cell at once resolve every face in :meth:`value_weight` and
        :meth:`gradient_weight`.

    Examples
    --------
    >>> closure = BoundaryClosure(lambda phi, grad: assembler.boundary_values(phi, grad, props))
    >>> gradient, boundary_values = closure.reconstruct(scheme, phi, mesh, geometry)
    """

    values: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def value_weight(self, field: jnp.ndarray, gradient: jnp.ndarray) -> jnp.ndarray:
        """``d(boundary value)/d(phi_owner)`` per face, shape ``(n_faces,)``.

        Zero where the value is prescribed, one where a normal derivative is, and between the two for
        a Robin condition.

        Parameters
        ----------
        field : jnp.ndarray
            Cell values, shape ``(n_cells,)``.
        gradient : jnp.ndarray
            Cell gradients, shape ``(n_cells, dim)``.
        """
        return jax.jvp(lambda f: self.values(f, gradient), (field,), (jnp.ones_like(field),))[1]

    def gradient_weight(self, field: jnp.ndarray, gradient: jnp.ndarray) -> jnp.ndarray:
        """``d(boundary value)/d(grad phi_owner)`` per face, shape ``(n_faces, dim)``.

        Zero where the value is prescribed, and the tangential offset a zero-gradient, Neumann or
        Robin condition carries its value along otherwise. One directional derivative per gradient
        component.

        Parameters
        ----------
        field : jnp.ndarray
            Cell values, shape ``(n_cells,)``.
        gradient : jnp.ndarray
            Cell gradients, shape ``(n_cells, dim)``.
        """
        return jnp.stack(
            [
                jax.jvp(
                    lambda g: self.values(field, g),
                    (gradient,),
                    (jnp.zeros_like(gradient).at[:, k].set(1.0),),
                )[1]
                for k in range(gradient.shape[1])
            ],
            axis=-1,
        )

    def linearization(self, n_cells: int, dim: int) -> BoundaryLinearization:
        """Both per-face derivatives, evaluated at a zero field and a zero gradient.

        Exact rather than approximate for every condition that is affine in its owner's value and
        gradient -- every shipped condition is -- since the derivatives are then the same at any
        state. Evaluated once, when a scheme is bound, rather than per residual.

        Parameters
        ----------
        n_cells : int
            Number of cells.
        dim : int
            Spatial dimension.
        """
        field = jnp.zeros(n_cells)
        gradient = jnp.zeros((n_cells, dim))
        return BoundaryLinearization(
            value_weight=self.value_weight(field, gradient),
            gradient_weight=self.gradient_weight(field, gradient),
        )

    def reconstruct(
        self,
        scheme: GradientScheme | None,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        *,
        operator_hook: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        imposed: ImposedGradient | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The cell gradient of ``field`` and the boundary values consistent with it.

        The reconstruction is fed the leading-order boundary values (:attr:`values` at a zero
        gradient), and also handed the corrected values at any gradient it asks for
        (``boundary_values_at``) and the gradient weight at zero gradient: a scheme that
        **differentiates** a boundary value needs the corrected one, because a gradient-type
        condition's whole content is its correction, and a scheme that reconstructs against the
        conditions themselves needs the weight. The returned boundary values are :attr:`values`
        re-evaluated at the reconstructed gradient.

        Parameters
        ----------
        scheme : GradientScheme or None
            The reconstruction, already bound for this field's conditions. ``None`` reconstructs
            nothing: the gradient is exactly zero, which is exact on an orthogonal mesh, where the
            non-orthogonal corrections it feeds vanish.
        field : jnp.ndarray
            Cell values, shape ``(n_cells,)``.
        mesh : Mesh
            Provides the connectivity the scheme reconstructs over.
        geometry : MeshGeometry
            Face and cell metrics.
        operator_hook : callable, optional
            A ghost-cell exchange threaded into an iterative reconstruction's own linear solve (see
            :meth:`~aquaflux.schemes.GradientScheme.gradients`); the identity when omitted.
        imposed : ImposedGradient, optional
            Cells whose gradient is a model quantity, used in place of a reconstruction there.

        Returns
        -------
        gradient : jnp.ndarray
            Cell gradients, shape ``(n_cells, dim)``.
        boundary_values : jnp.ndarray
            Boundary face values at that gradient, shape ``(n_faces,)``.
        """
        zero_gradient = jnp.zeros((mesh.n_cells, mesh.dim), dtype=field.dtype)
        if scheme is None:
            return zero_gradient, self.values(field, zero_gradient)
        gradient = scheme.gradients(
            field,
            mesh,
            geometry,
            self.values(field, zero_gradient),
            operator_hook=operator_hook,
            imposed=imposed,
            boundary_values_at=lambda g: self.values(field, g),
            boundary_gradient_weight=self.gradient_weight(field, zero_gradient),
        )
        return gradient, self.values(field, gradient)
