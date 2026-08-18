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

**The accumulation is done in coordinates local to each cell, not the mesh's global coordinate
system.** The formula above is exact and origin-independent for the true (continuous) integral,
but evaluated face-by-face in floating point it is not: each term is ``O(|x_ip|)`` in magnitude,
while a cell's own volume and centroid offset are ``O(cell size)`` — for a cell whose size is
small relative to its distance from the coordinate origin, the sum is a near-total cancellation
of much larger terms, and the centroid (which multiplies the flux by ``x_ip`` a second time) can
lose enough precision to land measurably outside the cell it describes. Subtracting a per-cell
reference point before accumulating and adding it back after keeps every term ``O(cell size)``
instead, which is the same reason a pyramid decomposition anchored at an interior point is the
standard way to evaluate this integral rather than one anchored at the origin. The reference
point used is the cheap :meth:`approx_centroids` estimate already computed to orient face
normals — reusing it costs nothing extra and needs no new dependency, and the correction is
exact regardless of how good an approximation it is: any point works as the local origin, so a
mediocre proxy only leaves more of the cancellation to the shift-back step, never a wrong answer.

On a **periodic seam** face the neighbour cell sits a full period away, so it "sees" the face at
its periodic image ``x_ip - neighbour_offset`` (the image on the neighbour's own side of the
domain), not at the owner-side centroid ``x_ip``. Feeding that offset-shifted centroid to the
neighbour half of each scatter is what keeps a boundary-column cell from accruing a spurious
``L * A`` of volume; with no offset (``neighbour_offset is None``) the neighbour centroid equals
the owner's and the scatters collapse back to the plain conservative/symmetric forms. The local
recentring composes with this unchanged: each side is first moved to its own periodic image, then
shifted by that same side's own approximate centroid, so a periodic mesh gets both corrections at
once.
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

        A cheap centroid estimate, needed twice, for two different reasons. First, to orient face
        normals outward: it needs no normals (avoiding the circular dependency with the exact
        centroid), so it is computed before them. Second, as the local reference point
        :meth:`from_faces` recentres around before accumulating the exact centroid — see that
        method for why. Each face contributes its centroid to both incident cells, so the
        accumulation is a symmetric face→cell scatter.

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

    @staticmethod
    def _local_face_position(
        face_centroid: jnp.ndarray, reference: jnp.ndarray, cell_index: jnp.ndarray
    ) -> jnp.ndarray:
        """A face centroid relative to one of its cells' local reference point, gathered per face.

        ``reference`` is indexed per cell and gathered onto faces via ``cell_index`` (the owner or
        safe-neighbour index), so each face's position is expressed relative to *that* cell's own
        reference point — the recentring :meth:`from_faces` needs on both the owner and the
        neighbour side, each against its own reference.
        """
        return face_centroid - reference[cell_index]

    @classmethod
    def from_faces(
        cls,
        face_geometry: FaceGeometry,
        face_cells: FaceCellConnectivity,
        dim: int,
        approx_centroid: jnp.ndarray,
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
        approx_centroid : jnp.ndarray, shape ``(n_cells, dim)``
            Per-cell local reference point the divergence-theorem accumulation is recentred
            around before summing (see the module docstring) — :meth:`approx_centroids` computed
            once from this same face geometry, reused rather than recomputed.
        """
        area = face_geometry.area
        centroid = face_geometry.centroid
        normal = face_geometry.normal

        # (x_ip . n) A with the owner-outward normal; the neighbour sees the opposite sign (-flux),
        # so both accumulations are conservative scatters of this owner-outward face quantity. On a
        # periodic seam the neighbour evaluates the same quantity at the face's periodic image
        # centroid, so its half uses the offset-shifted centroid (a no-op on ordinary meshes, where
        # scatter(f, -f) is exactly scatter_conservative(f)). Each side is additionally recentred on
        # its own cell's approximate centroid before the flux/moment are formed, and the final
        # centroid is shifted back after dividing by volume — the numerically-conditioned form of
        # the identical formula (see the module docstring).
        neighbour_centroid = cls._neighbour_face_centroid(centroid, face_cells)
        owner_local = cls._local_face_position(centroid, approx_centroid, face_cells.owner)
        neighbour_local = cls._local_face_position(
            neighbour_centroid, approx_centroid, face_cells.safe_neighbour
        )
        flux = dot(owner_local, normal) * area
        neighbour_flux = dot(neighbour_local, normal) * area
        volume = face_cells.scatter(flux, -neighbour_flux) / dim
        centroid_sum = face_cells.scatter(
            scale(owner_local, flux), -scale(neighbour_local, neighbour_flux)
        )
        local_centroid = scale(centroid_sum, 1.0 / ((dim + 1) * volume))
        return cls(volume=volume, centroid=local_centroid + approx_centroid)
