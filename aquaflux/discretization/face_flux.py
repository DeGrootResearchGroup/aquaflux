"""The face-flux operator contract: the strategy interface and the shared per-face context.

Every finite-volume transport term is a face-flux operator: given the cell field and the shared
per-evaluation context, it returns the owner-outward flux of the conserved quantity through each
face, which the residual engine scatters back to cells. This module owns the two types that
contract binds — :class:`FaceFluxOperator` (the strategy interface) and :class:`FaceContext` (the
shared inputs) — so the concrete operators (diffusion, advection, ...) and the assembler that
drives them both depend on it, not on each other.

:class:`FaceContext` carries only what is genuinely shared across operators or expensive to form
once: the connectivity and geometry, the weak boundary face values, the reconstructed cell
gradient (a linear solve, so formed once and reused), and the per-cell diffusion coefficient.
Each operator **gathers its own owner/neighbour inputs** from it — ``field[context.face_cells.owner]``
and friends — so an operator never pays to gather another operator's fields (a diffusion-only
solve never forms an advection limiter, for instance). That per-operator gather also makes each
operator self-describing about its inputs, which is what a data-driven (declarative) assembler
consumes.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry


class FaceContext(eqx.Module):
    """Shared per-face inputs the residual assembler provides to every flux operator.

    Assembled once per residual evaluation and passed to each :class:`FaceFluxOperator`, which
    gathers from it only the owner/neighbour fields it needs (via :attr:`face_cells`).

    Attributes
    ----------
    face_cells : FaceCellConnectivity
        The face→cell incidence and its gather/scatter operators — the gather primitive
        (``field[face_cells.owner]`` / ``field[face_cells.safe_neighbour]``).
    geometry : MeshGeometry
        Face metrics (areas, owner-outward normals, centroids) and cell metrics (volumes,
        centroids) — the geometry an operator gathers to faces.
    boundary_values : jnp.ndarray
        Weak boundary face values ``phi_ip``, shape ``(n_faces,)`` (interior entries ignored).
    gradient : jnp.ndarray
        Reconstructed cell gradient shared across operators, shape ``(n_cells, dim)``. Formed once
        per evaluation (a linear solve on skewed grids), so it is a context field rather than
        re-solved per operator; zeros when no gradient scheme is injected.
    properties : mapping of {str: jnp.ndarray}
        The evaluated per-cell properties, ``{name: (n_cells,) array}`` (density,
        viscosity, conductivity, ...), from the assembler's ``PropertyModel``. An operator reads
        the property it names (``context.properties[self.coefficient]``); this is one context field
        regardless of how many properties exist, so adding a property never changes the shape here.
    """

    face_cells: FaceCellConnectivity
    geometry: MeshGeometry
    boundary_values: jnp.ndarray
    gradient: jnp.ndarray
    properties: Mapping[str, jnp.ndarray]


class FaceFluxOperator(eqx.Module):
    """Strategy interface: the owner-outward flux of the conserved quantity through each face.

    **Sign convention.** A concrete operator returns the flux of the conserved quantity in the
    owner-outward direction — the physical flux vector dotted with the outward normal, times area.
    The residual engine forms ``R = accumulation + sum_faces(outward flux)`` (scatter: owner ``+``,
    neighbour ``-``), the standard finite-volume conservation statement. So an advective flux is
    ``+ mdot_f phi_f`` and a diffusive flux is ``- Gamma (grad phi . n) A`` (Fourier's law: flux is
    *down*-gradient).
    """

    @abc.abstractmethod
    def face_flux(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        """Owner-outward flux of the conserved quantity per face, shape ``(n_faces,)``.

        Parameters
        ----------
        field : jnp.ndarray
            The transported cell field, shape ``(n_cells,)``.
        context : FaceContext
            The shared per-face inputs; the operator gathers its owner/neighbour fields from it.
        """
