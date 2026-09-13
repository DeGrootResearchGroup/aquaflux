"""The face-flux operator contract: the strategy interface over a per-field evaluation context.

Every finite-volume transport term is a face-flux operator: given the cell field and the shared
per-evaluation context, it returns the owner-outward flux of the conserved quantity through each
face, which the residual engine scatters back to cells. This module owns
:class:`FaceFluxOperator` (the strategy interface), so the concrete operators (diffusion,
advection, ...) and the assembler that drives them both depend on it, not on each other. The
context itself, :class:`~aquaflux.context.FieldContext`, lives one layer below every strategy
family (see that module) so a scheme or a boundary closure can take it too without this package
importing them.

Each operator **gathers its own owner/neighbour inputs** from the context it is handed —
``field[context.mesh.face_cells.owner]`` and friends — so an operator never pays to gather
another operator's fields (a diffusion-only solve never forms an advection limiter, for
instance). That per-operator gather also makes each operator self-describing about its inputs,
which is what a data-driven (declarative) assembler consumes.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from aquaflux.context import FieldContext


class FaceFluxOperator(eqx.Module):
    """Strategy interface: the owner-outward flux of the conserved quantity through each face.

    **Sign convention.** A concrete operator returns the flux of the conserved quantity in the
    owner-outward direction — the physical flux vector dotted with the outward normal, times area.
    The residual engine forms ``R = accumulation + sum_faces(outward flux)`` (scatter: owner ``+``,
    neighbour ``-``), the standard finite-volume conservation statement. So an advective flux is
    ``+ mdot_f phi_f`` and a diffusive flux is ``- Gamma (grad phi . n) A`` (Fourier's law: flux is
    *down*-gradient).

    **Self-describing inputs.** :meth:`requires` and :meth:`uses_gradient` let an operator declare,
    before any residual is ever evaluated, what it reads from the context beyond the field itself —
    which named properties, and whether it needs a reconstructed (non-zero) gradient.
    :meth:`~aquaflux.discretization.residual.ResidualAssembler.build` validates both, so a mis-named
    property or a scheme paired with no gradient reconstruction fails at build time rather than
    deep inside a jitted residual. Both default to "nothing" so an operator that touches neither
    (:class:`~aquaflux.discretization.advection.AdvectionFlux` with a first-order scheme, say) need
    not override either.
    """

    @abc.abstractmethod
    def face_flux(self, field: jnp.ndarray, context: FieldContext) -> jnp.ndarray:
        """Owner-outward flux of the conserved quantity per face, shape ``(n_faces,)``.

        Parameters
        ----------
        field : jnp.ndarray
            The transported cell field, shape ``(n_cells,)``.
        context : FieldContext
            The shared per-face inputs; the operator gathers its owner/neighbour fields from it.
        """

    def requires(self) -> tuple[str, ...]:
        """Names this operator reads from ``context.properties`` (default: none).

        Override when the operator names a property, e.g.
        :class:`~aquaflux.discretization.diffusion.DiffusionFlux` returns ``(self.coefficient,)``.
        """
        return ()

    def uses_gradient(self) -> bool:
        """Whether this operator reads a non-zero ``context.gradient`` (default: ``False``).

        With no gradient scheme injected, :attr:`~ResidualAssembler.gradient_scheme` is ``None`` and
        ``context.gradient`` is zero everywhere. For most operators that is a graceful degradation
        (:class:`~aquaflux.discretization.diffusion.DiffusionFlux`'s non-orthogonal correction is
        *exactly* zero on an orthogonal grid, so the flux stays correct there). An operator whose
        entire purpose is the gradient term is different: silently reading zero does not fail, it
        silently gives a *worse* answer than the operator claims to compute. Override to ``True`` for
        such an operator: :class:`~aquaflux.discretization.advection.AdvectionFlux` delegates to its
        injected scheme, and :class:`~aquaflux.discretization.advection.LimitedUpwind`'s whole
        2nd-order reconstruction reads exactly this way, so ``build`` can refuse the combination
        outright rather than silently falling back to 1st order.
        """
        return False
