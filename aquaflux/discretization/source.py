"""The volume-source operator contract: a per-cell volumetric source term of the residual.

Not every finite-volume term is a face flux. A *volume source* is generated or consumed within the
cell itself rather than transported across its faces -- a chemical reaction rate, a turbulence
production or dissipation rate, a distributed heat release. Written as a source density ``S(phi)``
per unit volume, such a term contributes its cell integral ``integral over the cell of S dV`` to the
conservation balance.

**Sign convention.** A concrete operator returns the *volume-integrated* source in each cell (shape
``(n_cells,)``), with **production positive** -- a term that adds to ``phi``; a sink is simply a
negative source. The residual engine forms the finite-volume balance

    R = accumulation + sum_faces(owner-outward flux) - sum_sources(cell integral of S dV),

so a source enters the residual with a minus sign, and at steady state a cell's net outward flux
equals what is produced inside it. The volume is baked into the returned value -- the operator owns
its own volume quadrature, exactly as a :class:`~aquaflux.discretization.face_flux.FaceFluxOperator`
bakes the face area into its flux -- so the assembler composes integrated contributions uniformly
and never multiplies by a metric of its own.

A source gathers whatever per-cell state it needs -- the solved field, the reconstructed cell
gradient, material properties, the cell volume -- from the shared
:class:`~aquaflux.discretization.face_flux.FaceContext`, the same context the flux operators
receive; it reads the cell-oriented fields and ignores the face-only ones. Data held fixed for the
evaluation (a lagged coefficient, or a field supplied by a coupled equation) is carried on the
operator as constructor state, the same way
:class:`~aquaflux.discretization.advection.AdvectionFlux` carries its prescribed face mass flux.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp

from .face_flux import FaceContext


class VolumeSource(eqx.Module):
    """Strategy interface: the volume-integrated source of a cell field.

    A concrete source returns one value per cell -- the source integrated over the cell volume,
    production positive (see the module sign convention) -- given the acted-on cell field and the
    shared :class:`~aquaflux.discretization.face_flux.FaceContext` it gathers its inputs from. It is
    an immutable ``equinox.Module``, so any coefficient or frozen field it carries is a
    differentiable leaf and gradients flow through it.
    """

    @abc.abstractmethod
    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        """Volume-integrated source per cell, shape ``(n_cells,)`` (production positive).

        Parameters
        ----------
        field : jnp.ndarray
            The cell field the source acts on, shape ``(n_cells,)``.
        context : FaceContext
            The shared per-evaluation inputs; the source gathers the cell-oriented fields it needs
            (``geometry.cell.volume``, the reconstructed ``gradient``, ``materials``) from it.

        Returns
        -------
        jnp.ndarray
            The source integrated over each cell's volume, shape ``(n_cells,)``.
        """
