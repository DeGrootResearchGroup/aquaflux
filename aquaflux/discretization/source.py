"""The volume-source operator contract: a volumetric source term of the cell residual.

A concrete operator returns the *volume-integrated* source in each cell (shape ``(n_cells,)``),
**production positive**. The residual engine subtracts it, forming

    R = accumulation + sum_faces(owner-outward flux) - sum_sources(cell integral of S dV).

The volume is baked into the returned value: the operator owns its own volume quadrature, as a
:class:`~aquaflux.discretization.face_flux.FaceFluxOperator` bakes in the face area, so the assembler
composes integrated contributions uniformly.

The per-cell state a source needs (the solved field, cell gradient, properties, cell
volume) is gathered from the shared :class:`~aquaflux.context.FieldContext`. A
coefficient or field held fixed for the evaluation is carried as constructor state, as
:class:`~aquaflux.discretization.advection.AdvectionFlux` carries its prescribed mass flux.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from aquaflux.context import FieldContext


class VolumeSource(eqx.Module):
    """Strategy interface: the volume-integrated source of a cell field.

    A concrete source returns one value per cell -- the source integrated over the cell volume,
    production positive (see the module sign convention) -- given the acted-on cell field and the
    shared :class:`~aquaflux.context.FieldContext` it gathers its inputs from. It is
    an immutable ``equinox.Module``, so any coefficient or frozen field it carries is a
    differentiable leaf and gradients flow through it.
    """

    @abc.abstractmethod
    def source(self, field: jnp.ndarray, context: FieldContext) -> jnp.ndarray:
        """Volume-integrated source per cell, shape ``(n_cells,)`` (production positive).

        Parameters
        ----------
        field : jnp.ndarray
            The cell field the source acts on, shape ``(n_cells,)``.
        context : FieldContext
            The shared per-evaluation inputs; the source gathers the cell-oriented fields it needs
            (``mesh.geometry.cell.volume``, the reconstructed ``gradient``, ``mesh.properties``) from
            it.

        Returns
        -------
        jnp.ndarray
            The source integrated over each cell's volume, shape ``(n_cells,)``.
        """

    def requires(self) -> tuple[str, ...]:
        """Names this source reads from ``context.properties`` (default: none).

        Override when the source names a property it reads by key, mirroring
        :meth:`~aquaflux.discretization.face_flux.FaceFluxOperator.requires`; validated by
        :meth:`~aquaflux.discretization.residual.ResidualAssembler.build`.
        """
        return ()
