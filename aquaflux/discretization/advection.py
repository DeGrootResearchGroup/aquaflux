"""The advection face-flux operator and its face-value schemes.

The advective flux of a scalar through a face is ``mdot_f phi_f`` — the owner-outward mass
(or volume) flux ``mdot_f = (u . n) A`` carrying the face value ``phi_f`` of the transported
scalar. The mass flux is a *given* per-face field here (a prescribed velocity, or later the
Rhie-Chow face flux of a coupled momentum solve); the transported scalar is the unknown.

What distinguishes advection schemes is the reconstruction of ``phi_f`` from cell values, and
its **upwind bias** — information travels with the flow, so the face value is taken from the
upwind side (the owner when ``mdot_f >= 0``, else the neighbour):

- :class:`FirstOrderUpwind` takes the upwind cell value directly. Unconditionally bounded and
  linear in the field (so the residual stays affine and Newton is one step), at the cost of
  first-order numerical diffusion.
- :class:`LimitedUpwind` adds ``psi_C grad phi_C . (x_f - x_C)`` from the upwind cell ``C``, with
  an optional slope limiter ``psi_C`` for boundedness.

The operator returns the owner-outward flux ``mdot_f phi_f`` (see
:class:`~aquaflux.discretization.face_flux.FaceFluxOperator` for the sign convention); the
residual engine scatters it to the owner (``+``) and neighbour (``-``), so a single face flux
conservatively couples both cells.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot

from .face_flux import FaceContext, FaceFluxOperator

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity
    from aquaflux.schemes import Limiter


def _upwind_value(
    cell_field: jnp.ndarray, outflow: jnp.ndarray, face_cells: FaceCellConnectivity
) -> jnp.ndarray:
    """The upwind side's value per face: the owner where the flow leaves it, else the neighbour.

    Works for a scalar ``(n_cells,)`` or vector ``(n_cells, dim)`` cell field — the per-face
    ``outflow`` mask broadcasts over any trailing component axes, so callers need no ``[:, None]``.

    Parameters
    ----------
    cell_field : jnp.ndarray
        Per-cell values, shape ``(n_cells, ...)``.
    outflow : jnp.ndarray
        Per-face boolean, ``True`` where ``mdot_f >= 0`` (the flow leaves the owner), shape
        ``(n_faces,)``.
    face_cells : FaceCellConnectivity
        The face→cell incidence (``mesh.face_cells``).

    Returns
    -------
    jnp.ndarray
        The upwind-side value per face, shape ``(n_faces, ...)`` matching ``cell_field``'s
        trailing shape.
    """
    mask = outflow.reshape(outflow.shape + (1,) * (cell_field.ndim - 1))  # broadcast over dim
    return jnp.where(mask, cell_field[face_cells.owner], cell_field[face_cells.safe_neighbour])


class AdvectionScheme(eqx.Module):
    """Strategy interface: reconstruct the advected face value ``phi_f`` from cell state.

    A concrete scheme returns one value per face given the transported cell field, the shared
    :class:`~aquaflux.discretization.face_flux.FaceContext`, and the owner-outward face mass flux
    (whose sign is the upwind direction). It gathers whatever owner/neighbour fields it needs from
    the context.
    """

    @abc.abstractmethod
    def face_value(
        self,
        field: jnp.ndarray,
        context: FaceContext,
        mass_flux: jnp.ndarray,
    ) -> jnp.ndarray:
        """Advected face values ``phi_f``, shape ``(n_faces,)``.

        Parameters
        ----------
        field : jnp.ndarray
            The transported cell field, shape ``(n_cells,)``.
        context : FaceContext
            The shared per-face inputs (connectivity, geometry, gradient, boundary values).
        mass_flux : jnp.ndarray
            Owner-outward face mass flux ``mdot_f``, shape ``(n_faces,)``; its sign selects the
            upwind side.
        """


class FirstOrderUpwind(AdvectionScheme):
    """The upwind cell value: ``phi_f = phi_C``, ``C`` the upwind cell.

    On an interior face the upwind side is the owner when the flow leaves it
    (``mdot_f >= 0``) and the neighbour otherwise. On a boundary face outflow takes the owner
    value and inflow takes the weak boundary value (the prescribed inlet value) — the correct
    upwind choice, and the reason an outflow boundary needs no scalar value imposed.
    """

    def face_value(self, field, context, mass_flux):
        fc = context.face_cells
        outflow = mass_flux >= 0.0
        interior_value = _upwind_value(field, outflow, fc)
        # Boundary: outflow carries the owner value; inflow takes the weak boundary value.
        boundary_value = jnp.where(outflow, field[fc.owner], context.boundary_values)
        return fc.combine_face_values(interior_value, boundary_value)


class LimitedUpwind(AdvectionScheme):
    """Second-order limited linear upwind: ``phi_f = phi_C + psi_C grad phi_C . (x_f - x_C)``.

    The upwind cell ``C`` is reconstructed to the face with its gradient (from the context),
    scaled by a per-cell slope limiter ``psi_C in [0, 1]``. The optional :class:`Limiter` is held
    here and evaluated only when this scheme runs, so a diffusion-only or first-order-advection
    solve never forms it. With no limiter (``limiter=None``) ``psi = 1`` and this is unlimited
    linear upwind — second order but not bounded.

    On a boundary face the reconstruction is used for outflow (upwind is the owner) and the weak
    boundary value for inflow.

    Attributes
    ----------
    limiter : Limiter or None
        Slope limiter producing ``psi``; ``None`` gives unlimited (``psi = 1``) linear upwind.
    """

    limiter: Limiter | None = None

    def face_value(self, field, context, mass_flux):
        fc = context.face_cells
        gradient = context.gradient
        x_cell = context.geometry.cell.centroid

        if self.limiter is None:
            psi = jnp.ones(field.shape[0], dtype=field.dtype)
        else:
            psi = self.limiter.limit(field, gradient, fc, context.geometry)

        outflow = mass_flux >= 0.0
        phi_upwind = _upwind_value(field, outflow, fc)
        psi_upwind = _upwind_value(psi, outflow, fc)
        grad_upwind = _upwind_value(gradient, outflow, fc)
        x_upwind = _upwind_value(x_cell, outflow, fc)
        reconstruction = phi_upwind + psi_upwind * dot(
            grad_upwind, context.geometry.face.centroid - x_upwind
        )
        # Interior and boundary-outflow use the upwind reconstruction; boundary-inflow the weak value.
        return jnp.where(fc.interior | outflow, reconstruction, context.boundary_values)


class AdvectionFlux(FaceFluxOperator):
    """Advective face flux ``mdot_f phi_f`` for a prescribed face mass-flux field.

    The mass flux is injected (a constant field for a prescribed velocity; the coupled
    momentum solve supplies it later); the face-value reconstruction is an injected
    :class:`AdvectionScheme`. On a divergence-free mass-flux field the operator is
    conservative and adds no spurious source.

    Attributes
    ----------
    mass_flux : jnp.ndarray
        Owner-outward face mass flux ``mdot_f``, shape ``(n_faces,)`` — the flux the scalar is
        transported by, injected by the caller. For a prescribed divergence-free velocity this is
        ``(u . n) A`` (see :func:`face_mass_flux`); in a coupled momentum solve it is the
        Rhie--Chow face flux, the *same* flux that closes continuity — sharing it is what keeps the
        momentum convection and continuity discretely consistent. The operator only reads
        ``mdot_f``; how it was formed is the caller's concern.
    scheme : AdvectionScheme
        The face-value reconstruction (upwind, limited, ...).
    """

    mass_flux: jnp.ndarray
    scheme: AdvectionScheme

    def face_flux(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        phi_face = self.scheme.face_value(field, context, self.mass_flux)
        return self.mass_flux * phi_face
