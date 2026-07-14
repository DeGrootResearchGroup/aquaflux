"""Cell geometry: volume and centroid via divergence-theorem accumulation.

The divergence-theorem volume/centroid formula is dimension-general (one formula
parameterized by ``dim``), so — unlike the face geometry — there is a single
:class:`CellGeometry` class, not a strategy hierarchy. For a cell bounded by faces with
area ``A``, centroid ``x_ip``, and *outward* unit normal ``n``:

    volume   = (1 / dim) * sum_faces (x_ip . n) A
    centroid = [ sum_faces x_ip (x_ip . n) A ] / [ (dim + 1) * volume ]

Each face's stored normal is owner-outward, so the owner cell sees ``+n`` and the neighbour
``-n``: the volume/centroid accumulation is exactly a *conservative* face→cell scatter, and
the approximate centroid a *symmetric* one (each face centroid contributes to both its cells).
Those scatters — with the boundary convention (neighbour ``< 0`` contributes to its owner
only) — are provided by :class:`~aquaflux.mesh.connectivity.FaceCellConnectivity`, so this
module writes only the divergence-theorem math.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .connectivity import FaceCellConnectivity
from .face import FaceGeometry


class CellGeometry(eqx.Module):
    """Per-cell geometric quantities.

    Attributes
    ----------
    volume : jnp.ndarray
        Cell volumes (areas in 2D), shape ``(n_cells,)``.
    centroid : jnp.ndarray
        Cell centroids, shape ``(n_cells, dim)``.
    """

    volume: jnp.ndarray
    centroid: jnp.ndarray

    @staticmethod
    def approx_centroids(
        face_centroids: jnp.ndarray,
        face_cells: FaceCellConnectivity,
    ) -> jnp.ndarray:
        """Approximate cell centroids: the mean of each cell's face centroids.

        A cheap centroid estimate used to orient face normals outward. It needs no normals
        (avoiding the circular dependency with the exact centroid), so it is computed first
        and used only for orientation. Each face contributes its centroid to both incident
        cells, so the accumulation is a symmetric face→cell scatter.

        Parameters
        ----------
        face_centroids : jnp.ndarray
            Face centroids, shape ``(n_faces, dim)``.
        face_cells : FaceCellConnectivity
            The face→cell scatter operators.
        """
        ones = jnp.ones(face_cells.owner.shape[0], dtype=face_centroids.dtype)
        count = face_cells.scatter_symmetric(ones)
        centroid_sum = face_cells.scatter_symmetric(face_centroids)
        return centroid_sum / count[:, None]

    @classmethod
    def from_faces(
        cls,
        face_geometry: FaceGeometry,
        face_cells: FaceCellConnectivity,
        dim: int,
    ) -> CellGeometry:
        """Compute cell volumes and centroids from oriented face geometry.

        Parameters
        ----------
        face_geometry : FaceGeometry
            Face areas, centroids, and owner-outward normals.
        face_cells : FaceCellConnectivity
            The face→cell scatter operators.
        dim : int
            Spatial dimension.
        """
        area = face_geometry.area
        centroid = face_geometry.centroid
        normal = face_geometry.normal

        # (x_ip . n) A with the owner-outward normal; the neighbour sees the opposite sign, so
        # both accumulations are conservative scatters of this owner-outward face quantity.
        flux = jnp.sum(centroid * normal, axis=1) * area
        volume = face_cells.scatter_conservative(flux) / dim
        centroid_sum = face_cells.scatter_conservative(centroid * flux[:, None])
        cell_centroid = centroid_sum / ((dim + 1) * volume)[:, None]
        return cls(volume=volume, centroid=cell_centroid)
