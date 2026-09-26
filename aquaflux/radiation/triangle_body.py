"""A solid body, or a sheet, given as triangles: whatever exists only as a mesh can stand in the way.

The analytic bodies answer "does this segment pass through me" by formula. Some geometry has no
formula -- a sculpted baffle, a vessel exported from a drawing no one has described as
primitives, a mesh's own wall -- and exists only as the triangles a file holds.
:class:`TriangleBody` is that geometry as a :class:`~aquaflux.solids.Body`, so it goes in the same
occluder list as the primitives and a scene may mix the two.

**It answers on the host.** A segment is tested only against the triangles a
:class:`~aquaflux.radiation.grid.TriangleGrid` puts along its path, and the walk stops at the
first blocker -- a search whose whole value is the work it skips, which tracing would price at
the same rate as the work done. So the body declares itself not
:attr:`~aquaflux.solids.Body.traceable`, and whoever builds a mask calls it directly.

**What is inside it is decided per connected piece, not for the body as a whole.** A piece of
surface with a free edge is a **sheet**: it has the same medium on both sides and no inside at
all, so nothing is embedded in it. A **closed** piece bounds a region, and which side of it is
solid is the side its triangles' normals point *away* from -- the convention an emitting surface
already follows, since a lamp is wound to face the water and a vessel wall to face the fluid it
holds. A closed piece wound outward (positive signed volume) is a solid lump, a sleeve, say;
one wound inward is a vessel, solid everywhere *outside* it. Both are read from the triangles'
winding number, which is exactly ``+1`` or ``-1`` inside a closed, consistently wound surface
and ``0`` outside it: a point is in the solid where the winding of the closed pieces, plus the
number of pieces wound inward, is one.

⚠️ **Topology cannot see every sheet**, which is why the caller can say. A sheet welded to
another surface all the way round has no free edge and reads as closed; a duct exported without
its end caps reads as a sheet. ``sheet=True`` declares every piece a sheet whatever its topology;
``sheet=False`` insists every piece is closed and refuses the body if one is not.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from aquaflux.radiation.checks import _surface_pieces, enclosure_winding
from aquaflux.radiation.grid import _MAX_VOXELS, TriangleGrid
from aquaflux.solids import Body

__all__ = ["TriangleBody"]

#: How much finer, per axis, the grid a body vouches from is than the grid its segments walk.
#: The two want different voxels: a walk is fastest at about ten triangles a voxel, but a box is
#: vouched for only if it overlaps no occupied voxel at all, and a voxel that size beside a wall
#: is occupied however little of it the wall crosses. Measured on the Sozzi chamber's wall as
#: triangles, a shaft's box over voxels of ~7 mm vouched for 44% of the pairs the exact test
#: could, ~3.6 mm for 59% and ~1.8 mm for 59% -- the first halving is most of it.
_CLEARANCE_REFINEMENT = 4

#: A closed piece whose signed volume is smaller than this share of its bounding box's is taken
#: to enclose nothing: a flat double sheet, or a piece whose winding cancels itself out. Either
#: way no side of it can be called solid, and it is refused rather than guessed at.
_FLAT_PIECE = 1e-9


class TriangleBody(Body):
    """Triangles standing in the way, tested through a uniform grid over them.

    Built by :meth:`build`.

    Attributes
    ----------
    grid : TriangleGrid
        The triangles and the grid that selects which of them a segment is tested against.
    occupancy : TriangleGrid
        The same triangles over finer voxels, read only for which voxels hold any: what
        :meth:`vouches` asks.
    enclosing : np.ndarray, shape ``(n_enclosing, 3, 3)``
        The triangles of the closed pieces, which alone decide what is inside the body.
    inward_pieces : int
        How many closed pieces are wound inward, so that their outside is the solid.
    """

    traceable: ClassVar[bool] = False

    grid: TriangleGrid = eqx.field(static=True)
    occupancy: TriangleGrid = eqx.field(static=True)
    enclosing: np.ndarray
    inward_pieces: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        vertices,
        *,
        sheet: bool | None = None,
        resolution: int | tuple[int, int, int] | None = None,
        clearance_resolution: int | tuple[int, int, int] | None = None,
        tolerance: float | None = None,
    ) -> TriangleBody:
        """A body of these triangles, deciding per piece whether each is a sheet or closed.

        Parameters
        ----------
        vertices : array_like, shape ``(n_triangles, 3, 3)``
            The triangles, wound so that each normal points out of the solid it bounds.
        sheet : bool or None, optional
            ``None`` (the default) reads each connected piece's topology: a piece with a free edge
            is a sheet, a closed one bounds a solid. ``True`` makes every piece a sheet, so the
            body has no inside. ``False`` requires every piece to be closed.
        resolution : int or tuple of int, optional
            Voxels per axis of the grid segments walk; see
            :meth:`~aquaflux.radiation.grid.TriangleGrid.build`.
        clearance_resolution : int or tuple of int, optional
            Voxels per axis of the grid the body vouches from. Unset, four times the walking
            grid's per axis, halved as a whole while it holds more voxels than a default grid
            may.
        tolerance : float, optional
            Distance within which two vertex positions are the same point when pieces are
            found; see :func:`~aquaflux.radiation.checks.open_facets`.

        Returns
        -------
        TriangleBody

        Raises
        ------
        ValueError
            If ``sheet=False`` and a piece is open, or if a closed piece encloses no volume, so
            that neither of its sides can be called solid.
        """
        vertices = np.ascontiguousarray(vertices, dtype=float)
        grid = TriangleGrid.build(vertices, resolution=resolution)
        if clearance_resolution is None:
            clearance_resolution = _CLEARANCE_REFINEMENT * grid.resolution
            while np.prod(clearance_resolution, dtype=float) > _MAX_VOXELS:
                clearance_resolution = np.maximum(clearance_resolution // 2, 1)
        occupancy = TriangleGrid.build(vertices, resolution=tuple(clearance_resolution))
        if sheet:
            return cls(
                grid=grid, occupancy=occupancy, enclosing=np.zeros((0, 3, 3)), inward_pieces=0
            )
        piece, open_piece = _surface_pieces(vertices, tolerance=tolerance)
        if sheet is False and np.any(open_piece):
            msg = (
                f"{int(np.count_nonzero(open_piece))} of {len(open_piece)} pieces of this surface "
                "have a free edge, so they bound nothing and cannot be solid, but the body was "
                "declared closed (sheet=False). Close the surface, or leave `sheet` unset to "
                "treat the open pieces as sheets."
            )
            raise ValueError(msg)
        closed = ~open_piece[piece]
        volume = _signed_volumes(vertices, piece, len(open_piece))
        extent = np.ptp(vertices.reshape(-1, 3), axis=0)
        flat = ~open_piece & (np.abs(volume) <= _FLAT_PIECE * float(np.prod(extent)))
        if np.any(flat):
            msg = (
                f"{int(np.count_nonzero(flat))} closed piece(s) of this surface enclose no volume, "
                "so neither side of them can be called solid -- a flat double sheet, or a piece "
                "whose triangles are wound inconsistently (see check_winding). Pass sheet=True if "
                "the surface is meant to have no inside."
            )
            raise ValueError(msg)
        inward = int(np.count_nonzero(~open_piece & (volume < 0.0)))
        return cls(grid=grid, occupancy=occupancy, enclosing=vertices[closed], inward_pieces=inward)

    def contains(self, position) -> jnp.ndarray:
        """Whether each position is in the solid the closed pieces bound.

        Exact, from the winding number (:func:`~aquaflux.radiation.checks.enclosure_winding`),
        and only paid for where it can matter: a closed surface's winding is zero outside its
        bounding box, so positions there are outside every outward piece and inside every inward
        one without a solid angle computed.
        """
        position = np.asarray(position, dtype=float)
        shape = position.shape[:-1]
        flat = position.reshape(-1, 3)
        winding = np.zeros(len(flat))
        if len(self.enclosing):
            corners = self.enclosing.reshape(-1, 3)
            near = np.all((flat >= corners.min(axis=0)) & (flat <= corners.max(axis=0)), axis=1)
            if np.any(near):
                winding[near] = enclosure_winding(self.enclosing, flat[near])
        return jnp.asarray((winding + self.inward_pieces > 0.5).reshape(shape))

    def blocks(self, origin, target, min_distance) -> jnp.ndarray:
        """See :meth:`aquaflux.solids.Body.blocks`: the grid walk, over the broadcast segments."""
        source, receiver = np.broadcast_arrays(
            np.asarray(origin, dtype=float), np.asarray(target, dtype=float)
        )
        shape = source.shape[:-1]
        near = np.broadcast_to(np.asarray(min_distance, dtype=float), shape)
        blocked = self.grid.blocks(source.reshape(-1, 3), receiver.reshape(-1, 3), near.reshape(-1))
        return jnp.asarray(blocked.reshape(shape))

    def clearance(self, position) -> jnp.ndarray:
        """Each position's coordinates and their negatives, whose maxima are a set's bounding box.

        The largest of each over a set are its upper corner and its lower corner negated, so a
        merged summary is exactly the bounding box of every position merged into it -- which
        holds the convex hull, and with it every segment between the positions.
        """
        position = jnp.asarray(position, dtype=float)
        return jnp.concatenate([position, -position], axis=-1)

    def vouches(self, summary) -> jnp.ndarray:
        """True where the set's bounding box overlaps no occupied voxel of the grid.

        See :meth:`~aquaflux.radiation.grid.TriangleGrid.holds_any`: no triangle can then meet
        the box, and so none meets a segment inside it.
        """
        summary = np.asarray(summary, dtype=float)
        return jnp.asarray(~self.occupancy.holds_any(-summary[..., 3:], summary[..., :3]))


def _signed_volumes(vertices: np.ndarray, piece: np.ndarray, n_pieces: int) -> np.ndarray:
    """Each piece's signed enclosed volume, positive where its normals point outward.

    The divergence theorem over the piece's triangles: a sum of tetrahedra from the origin, so a
    closed piece's total does not depend on where the origin is. Meaningless for an open piece,
    which the caller ignores.
    """
    tetra = np.einsum("ij,ij->i", vertices[:, 0], np.cross(vertices[:, 1], vertices[:, 2])) / 6.0
    return np.bincount(piece, weights=tetra, minlength=n_pieces)
