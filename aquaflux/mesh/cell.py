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

On a **periodic seam** face the neighbour cell sits a full period away, so it "sees" the face at
its periodic image ``x_ip - neighbour_offset`` (the image on the neighbour's own side of the
domain), not at the owner-side centroid ``x_ip``. Feeding that offset-shifted centroid to the
neighbour half of each scatter is what keeps a boundary-column cell from accruing a spurious
``L * A`` of volume; with no offset (``neighbour_offset is None``) the neighbour centroid equals
the owner's and the scatters collapse back to the plain conservative/symmetric forms.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, scale

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
    def _neighbour_face_centroid(
        face_centroids: jnp.ndarray, face_cells: FaceCellConnectivity
    ) -> jnp.ndarray:
        """Face centroids as the *neighbour* cell sees them: shifted to their periodic image.

        Equal to ``face_centroids`` on every non-seam face (and when the mesh carries no periodic
        offset), so a scatter of ``(owner=face_centroids, neighbour=this)`` reduces to the plain
        conservative/symmetric scatter for an ordinary mesh.
        """
        if face_cells.neighbour_offset is None:
            return face_centroids
        return face_centroids - face_cells.neighbour_offset

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
        neighbour_centroids = CellGeometry._neighbour_face_centroid(face_centroids, face_cells)
        centroid_sum = face_cells.scatter(face_centroids, neighbour_centroids)
        return scale(centroid_sum, 1.0 / count)

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

        # (x_ip . n) A with the owner-outward normal; the neighbour sees the opposite sign (-flux),
        # so both accumulations are conservative scatters of this owner-outward face quantity. On a
        # periodic seam the neighbour evaluates the same quantity at the face's periodic image
        # centroid, so its half uses the offset-shifted centroid (a no-op on ordinary meshes, where
        # scatter(f, -f) is exactly scatter_conservative(f)).
        neighbour_centroid = cls._neighbour_face_centroid(centroid, face_cells)
        flux = dot(centroid, normal) * area
        neighbour_flux = dot(neighbour_centroid, normal) * area
        volume = face_cells.scatter(flux, -neighbour_flux) / dim
        centroid_sum = face_cells.scatter(
            scale(centroid, flux), -scale(neighbour_centroid, neighbour_flux)
        )
        cell_centroid = scale(centroid_sum, 1.0 / ((dim + 1) * volume))
        return cls(volume=volume, centroid=cell_centroid)
