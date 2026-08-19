"""Slope limiters for bounded second-order reconstruction.

A limited linear reconstruction takes the face value ``phi_f = phi_C + psi_C grad phi_C .
(x_f - x_C)`` from the upwind cell ``C``. The limiter ``psi_C in [0, 1]`` throttles the
gradient term so the reconstructed face values stay within the range of the surrounding cell
values — ``psi = 1`` is full second-order, ``psi = 0`` collapses to first-order upwind. It is a
**per-cell** quantity: the minimum, over the cell's faces, of a per-face limiter function.

:class:`VenkatakrishnanLimiter` is the smooth limiter of Venkatakrishnan (1993). For each face
of cell ``i`` with unlimited increment ``d- = grad phi_i . (x_f - x_i)`` and available headroom
``d+`` (``phi_max - phi_i`` when ``d- > 0``, else ``phi_min - phi_i``, over the cell's stencil):

    psi_face = [ (d+^2 + eps^2) d- + 2 d-^2 d+ ] / [ (d+^2 + 2 d-^2 + d+ d- + eps^2) d- ],

and ``psi_i = min_face psi_face``. The softening parameter ``eps^2 = vol_i K^3`` (``K`` a
constant) switches limiting off in smooth regions, where the classic (non-smooth) min/max
limiters would clip and stall convergence — the whole reason to prefer this one.

Unlike the classic implementation, which **freezes** ``psi`` at the previous iterate and adds
the limited correction as an explicit source, here ``psi(phi, grad phi)`` is written into the
residual. It is smooth in its arguments (the only non-smoothness is the ``min`` over faces and
the stencil ``min``/``max``, continuous with measure-zero kinks), so automatic differentiation
linearizes it and places it in the Jacobian.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry


class Limiter(eqx.Module):
    """Strategy interface: a per-cell slope limiter ``psi in [0, 1]``."""

    @abc.abstractmethod
    def limit(
        self,
        field: jnp.ndarray,
        gradient: jnp.ndarray,
        face_cells: FaceCellConnectivity,
        geometry: MeshGeometry,
    ) -> jnp.ndarray:
        """Per-cell limiter values, shape ``(n_cells,)``.

        Parameters
        ----------
        field : jnp.ndarray
            Cell values, shape ``(n_cells,)``.
        gradient : jnp.ndarray
            Cell gradients, shape ``(n_cells, dim)``.
        face_cells : FaceCellConnectivity
            Owner/neighbour incidence (``mesh.face_cells``).
        geometry : MeshGeometry
            Face and cell metrics (face centroids; cell centroids and volumes).
        """


class VenkatakrishnanLimiter(Limiter):
    """The smooth Venkatakrishnan (1993) limiter.

    Attributes
    ----------
    k : float
        The softening constant ``K`` in ``eps^2 = vol K^3`` (static). Larger ``K`` limits less
        (smoother, less bounded); ``K -> 0`` recovers a strict limiter.
    """

    k: float = eqx.field(static=True, default=5.0)

    def limit(self, field, gradient, face_cells, geometry):
        face_geometry, cell_geometry = geometry.face, geometry.cell
        owner = face_cells.owner
        neighbour = face_cells.safe_neighbour
        phi = field

        # Stencil extrema over each cell and its interior-face neighbours: each cell takes the
        # extremum of the adjacent cells' values (owner ← neighbour's phi, neighbour ← owner's),
        # via the connectivity's max/min scatter, which excludes a boundary face's neighbour side
        # structurally.
        phi_max = jnp.maximum(phi, face_cells.scatter_max(phi[neighbour], phi[owner]))
        phi_min = jnp.minimum(phi, face_cells.scatter_min(phi[neighbour], phi[owner]))

        eps2 = cell_geometry.volume * self.k**3  # eps^2 = vol K^3 (Venkatakrishnan softening)
        x_face = face_geometry.centroid

        def face_limiter(cell):
            """Venkatakrishnan psi for each face as seen from ``cell``."""
            delta_minus = dot(gradient[cell], x_face - cell_geometry.centroid[cell])
            # Regularize away from zero, treating zero as positive (sign of +1 at x == 0) so a
            # vanishing increment (constant field) gives psi -> 1 rather than 0/0.
            sign = jnp.where(delta_minus >= 0.0, 1.0, -1.0)
            delta_minus = sign * (jnp.abs(delta_minus) + 1e-12)
            headroom = jnp.where(
                delta_minus > 0.0, phi_max[cell] - phi[cell], phi_min[cell] - phi[cell]
            )
            e2 = eps2[cell]
            numerator = (headroom**2 + e2) * delta_minus + 2.0 * delta_minus**2 * headroom
            denominator = (
                headroom**2 + 2.0 * delta_minus**2 + headroom * delta_minus + e2
            ) * delta_minus
            return numerator / denominator

        # Per-cell limiter = min over the cell's incident faces. Neither side needs masking: the
        # owner is a real cell on every face, and the connectivity's scatter_min already excludes a
        # boundary face's neighbour side from the reduction, whatever value face_limiter gives it
        # there (a boundary face reads safe_neighbour == owner, so its neighbour-side value is just
        # a harmless second copy of the owner-side one, discarded before it can matter).
        psi = face_cells.scatter_min(face_limiter(owner), face_limiter(neighbour))
        return jnp.clip(psi, 0.0, 1.0)
