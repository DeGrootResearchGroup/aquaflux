"""A core of hexahedra from a balanced octree, joined conformally where cube sizes change, its outer
quadrilaterals capped with pyramids for a tetrahedral fill.

**Which cubes.** An :class:`~aquaflux.octree.Octree` of root cubes of the core's ``size``
covers the region's box, is refined towards the region's boundary, and is balanced, so cubes
sharing a face or part of an edge differ by at most one level. A leaf is kept when the region's
boundary passes nowhere through the cube grown by ``clearance`` on every side and the cube's centre
lies inside -- asked of a :class:`~aquaflux.meshing.Region`, exactly of a closed triangle surface
and on a lattice of sample points of an inside test -- which leaves room between the core and the
walls for the layers and the fill.

**Conformal without hanging points.** A cube whose edge has a smaller cube's corner at its midpoint
cannot stay a hexahedron: the point would hang on its edge. Such a cube is a *transition cube*, and
it is cut into pieces that all meet its centre. Each of its six faces is

- one quadrilateral, when none of its edges is halved: the matching face of a hexahedron, or of a
  transition cube of the same size, across it;
- four quadrilaterals through the face's centre, when the cube across is split (balance makes the
  four smaller cubes there leaves whose faces are exactly these);
- otherwise -- some edges halved by a smaller cube that shares only the edge -- a fan of triangles
  from a point at the face's centre through its corners and the halving points, which the cube of
  the same size across the face builds identically.

A quadrilateral and the centre make a pyramid, a triangle and the centre a tetrahedron. Every face
is decided from the whole tree, not from the cubes kept, so the faces of any subset of the leaves
match wherever two of them meet. The pyramids and tetrahedra are interior to the hexahedral core:
the polyhedral dual of the whole primal mesh turns them into polyhedral cells joining the
hexahedra of different sizes.

**A closed manifold boundary.** Cubes touching only along an edge or at a corner would make the
core's boundary pinch there, which neither a tetrahedral fill nor the dual accepts; one cube at each
such point is dropped, and the test repeats until none is left.

**The caps.** Each boundary quadrilateral gets a pyramid built outward, so the boundary left for
the fill is triangles only: the caps' sides and the boundary triangles of transition cubes. Sized by
:class:`~aquaflux.meshing.ClearedCaps`, each is as tall as its room allows -- lower where the core
steps concavely, two caps' sides leaning towards each other there, and within a share of the
clearance. Given as a share of the edge instead, ``cap_height`` must stay below one half, where two
caps over a concave right-angled step would meet.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from aquaflux.octree import Octree

from .caps import ClearedCaps, capped
from .elements import HEXAHEDRON_CORNERS
from .near_surface import NearSurface
from .polygons import Polygons, PyramidCaps
from .regions import ClosedSurface, Region

__all__ = ["HexCore", "OctreeCore"]

#: A face's corners in the plane of its two other axes, counter-clockwise about its axis.
FACE_RING = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

_UNUSED = -1

#: How many of a closed surface's voxels span a wall cube (:meth:`OctreeCore.within`). A cube is
#: kept only clear of every voxel the surface reaches, so a cube may be dropped up to a voxel
#: beyond the clearance; a finer grid costs a register entry per triangle per voxel it spans.
VOXELS_PER_WALL_CUBE = 4


@dataclasses.dataclass(frozen=True)
class HexCore:
    """A hexahedral core, conformal throughout, with pyramids capping its boundary quadrilaterals.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        Every corner of the cells and the caps; the caps' apexes last.
    hexahedra : np.ndarray of int, shape ``(n_hexahedra, 8)``
        The cubes kept whole: a face's four corners, then those of the opposite face, each over
        the one before it.
    pyramids : np.ndarray of int, shape ``(n_pyramids, 5)``
        The transition cubes' pyramids: a quadrilateral's corners in order, then the cube's centre.
    tetrahedra : np.ndarray of int, shape ``(n_tetrahedra, 4)``
        The transition cubes' tetrahedra: the cube's centre and a boundary triangle.
    caps : np.ndarray of int, shape ``(n_caps, 5)``
        The pyramids on the core's boundary quadrilaterals: the quadrilateral, then the apex
        outside the core.
    surface : np.ndarray of int, shape ``(n_triangles, 3)``
        The closed surface left for a fill -- the caps' sides and the core's boundary triangles --
        wound out of the region still to be filled, so into the core.
    octree : Octree
        The balanced tree the core was taken from.
    kept : np.ndarray of int, shape ``(n_cubes,)``
        The leaves of ``octree`` in the core.
    """

    points: np.ndarray
    hexahedra: np.ndarray
    pyramids: np.ndarray
    tetrahedra: np.ndarray
    caps: np.ndarray
    surface: np.ndarray
    octree: Octree
    kept: np.ndarray


@dataclasses.dataclass(frozen=True)
class OctreeCore:
    """Builds a :class:`HexCore` inside a region: cubes of ``size``, made smaller towards the
    region's boundary, and kept ``clearance`` from it.

    The root cubes are halved wherever the boundary passes within ``clearance + wall_size`` of
    them until they are no larger than ``wall_size``, so the cubes facing the gap left between the
    core and the boundary are the largest ``size / 2**k`` not over ``wall_size``, and balance grades
    them up to ``size`` away from it.

    Attributes
    ----------
    size : float
        The largest cubes' edge: the root cubes of the octree.
    wall_size : float or None
        The edge wanted of the cubes facing the boundary, at most ``size``. Unset, the size the
        boundary itself is meshed at (``boundary_size`` of :meth:`mesh`).
    clearance : float or None
        The least distance kept between a cube and the region's boundary. Unset, twice the wall
        size.
    """

    size: float
    wall_size: float | None = None
    clearance: float | None = None

    def __post_init__(self) -> None:
        for name in ("size", "wall_size", "clearance"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"OctreeCore.{name} must be a positive, finite number, got {value!r}."
                )
        if self.wall_size is not None and self.wall_size > self.size:
            raise ValueError(
                f"OctreeCore.wall_size ({self.wall_size!r}) may not exceed its size ({self.size!r})."
            )

    def resolved(self, boundary_size: float) -> tuple[float, float]:
        """The wall size and the clearance, the unset ones from ``boundary_size``.

        Parameters
        ----------
        boundary_size : float
            The size the region's boundary is meshed at.

        Returns
        -------
        wall_size, clearance : float

        Raises
        ------
        ValueError
            If an unset wall size would take a ``boundary_size`` larger than :attr:`size`.
        """
        wall = self.wall_size
        if wall is None:
            if boundary_size > self.size:
                raise ValueError(
                    f"OctreeCore.wall_size is unset, so it is the boundary's size "
                    f"({boundary_size!r}), which exceeds the core's size ({self.size!r})."
                )
            wall = boundary_size
        return wall, (2.0 * wall if self.clearance is None else self.clearance)

    def mesh(
        self,
        region: Region,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        cap_height: float | ClearedCaps,
        boundary_size: float,
    ) -> HexCore:
        """The core of ``region``, within the box from ``lower`` to ``upper``.

        Parameters
        ----------
        region : Region
            The region (see :class:`~aquaflux.meshing.ClosedSurface` and
            :class:`~aquaflux.meshing.SampledRegion`). A cube is kept where the region's boundary
            is taken to pass nowhere through the cube grown by the clearance, and its centre lies
            inside.
        lower, upper : sequence of float, length 3
            A box holding the region.
        cap_height : float or ClearedCaps
            The caps' height over their quadrilaterals' edge, in ``(0, 1/2)`` (see
            :class:`~aquaflux.meshing.PyramidCaps`), or each sized to its room
            (:class:`~aquaflux.meshing.ClearedCaps`, the clearance free in front of every cap).
        boundary_size : float
            The size the region's boundary is meshed at, for an unset :attr:`wall_size`.

        Returns
        -------
        HexCore

        Raises
        ------
        ValueError
            If a cap would reach the clearance's far side (the clearance is no more than the caps'
            height over a wall cube), or no cube fits.
        """
        wall, clearance = self.resolved(boundary_size)
        if not isinstance(cap_height, ClearedCaps):
            PyramidCaps.check_height(cap_height, "the core's cap_height")
        if not isinstance(cap_height, ClearedCaps) and clearance <= cap_height * wall:
            raise ValueError(
                f"a cap {cap_height!r} of a wall cube ({wall!r}) high would cross the clearance "
                f"({clearance!r}) kept from the boundary"
            )
        tree = Octree.lattice(lower, upper, self.size)
        tree = tree.refined_by([NearSurface(region, wall, margin=clearance + wall)]).balanced()
        grown = tree.lower - clearance
        clear = ~region.meets(grown, grown + tree.size[:, None] + 2.0 * clearance)
        candidates = np.flatnonzero(clear)
        candidates = candidates[
            region.inside(tree.lower[candidates] + 0.5 * tree.size[candidates, None])
        ]
        faces = _CubeFaces(tree, candidates)
        kept = _without_pinches(faces)
        if not np.any(kept):
            raise ValueError("no cube fits in the region at this size and clearance")
        return self._assembled(tree, faces, kept, cap_height, clearance)

    def within(
        self,
        points: np.ndarray,
        triangles: np.ndarray,
        *,
        cap_height: float | ClearedCaps,
        boundary_size: float,
    ) -> HexCore:
        """The core inside a closed triangle surface: :meth:`mesh` of its :class:`ClosedSurface`.

        The surface's voxels are a :data:`VOXELS_PER_WALL_CUBE`-th of the wall size across, which
        bounds how far beyond the clearance a cube may be dropped for being near the surface.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        triangles : np.ndarray of int, shape ``(n_triangles, 3)``
            The surface, closed and consistently wound.
        cap_height, boundary_size : float
            As in :meth:`mesh`.

        Returns
        -------
        HexCore
            Its points numbered on their own, not after ``points``.
        """
        wall, _ = self.resolved(boundary_size)
        region = ClosedSurface.of(points, triangles, wall / VOXELS_PER_WALL_CUBE)
        used = np.asarray(points, dtype=float)[np.unique(triangles)]
        return self.mesh(
            region, used.min(axis=0), used.max(axis=0), cap_height=cap_height,
            boundary_size=boundary_size,
        )  # fmt: skip

    @staticmethod
    def _assembled(
        tree: Octree, faces: _CubeFaces, kept: np.ndarray, cap_height, clearance: float = np.inf
    ) -> HexCore:
        """The cells of the kept cubes and the caps on their boundary, numbered compactly."""
        rings, cube = faces.rings[kept[faces.cube]], faces.cube[kept[faces.cube]]
        transition = faces.transition & kept
        whole = ~faces.transition & kept
        centres = faces.centre[transition]
        keys = np.unique(
            np.concatenate([rings[rings != _UNUSED], centres, faces.hexahedra[whole].ravel()])
        )
        points = tree.origin + 0.5 * tree.unit * Octree.coordinates(keys)

        def number(k):
            return np.where(k == _UNUSED, _UNUSED, np.searchsorted(keys, k))

        rings = number(rings)
        centre_of = np.full(faces.transition.size, _UNUSED, dtype=np.int64)
        centre_of[transition] = number(centres)
        inner = transition[cube]
        quads = rings[:, 3] != _UNUSED
        pyramids = np.column_stack([rings[inner & quads], centre_of[cube[inner & quads]]])
        tetrahedra = np.column_stack([centre_of[cube[inner & ~quads]], rings[inner & ~quads, :3]])

        boundary = rings[_once(rings)]
        outward = Polygons(
            boundary[boundary[:, 3] == _UNUSED, :3], boundary[boundary[:, 3] != _UNUSED]
        )
        caps = capped(points, outward, cap_height, clearance)
        triangles = outward.triangles[:, ::-1]
        return HexCore(
            points=np.vstack([points, caps.apexes]),
            hexahedra=number(faces.hexahedra[whole]),
            pyramids=pyramids,
            tetrahedra=tetrahedra,
            caps=caps.pyramids,
            surface=np.vstack([caps.sides, triangles]),
            octree=tree,
            kept=faces.leaves[kept],
        )


class _CubeFaces:
    """The faces of some leaves of a balanced octree, as rings of point keys at half a unit.

    Attributes
    ----------
    leaves : np.ndarray of int, shape ``(n,)``
        The leaves, by index in the tree.
    level : np.ndarray of int, shape ``(n,)``
        Their levels.
    transition : np.ndarray of bool, shape ``(n,)``
        Whether each is cut into pieces about its centre.
    centre : np.ndarray of int, shape ``(n,)``
        Each cube's centre's key.
    hexahedra : np.ndarray of int, shape ``(n, 8)``
        Each cube's corners' keys, as a hexahedron.
    rings : np.ndarray of int, shape ``(n_faces, 4)``
        Every face of every cube, wound out of its cube; a triangle's fourth slot is unused.
    cube : np.ndarray of int, shape ``(n_faces,)``
        Which of ``leaves`` each face bounds.
    """

    def __init__(self, tree: Octree, leaves: np.ndarray) -> None:
        self.leaves = leaves
        self.level = tree.level[leaves]
        lower = 2 * tree.corner[leaves]
        edge = 2 * tree.span[leaves][:, None]
        offsets = 2 * tree.corner[:, None, :] + 2 * tree.span[:, None, None] * HEXAHEDRON_CORNERS
        corners = np.unique(Octree.keys(offsets))  # every leaf's corners, kept or not
        self.centre = Octree.keys(lower + edge // 2)
        self.hexahedra = Octree.keys(lower[:, None, :] + edge[:, :, None] * HEXAHEDRON_CORNERS)

        def present(keys):
            position = np.minimum(np.searchsorted(corners, keys), corners.size - 1)
            return corners[position] == keys

        sides = []
        for axis, high in itertools.product(range(3), (0, 1)):
            ring = FACE_RING if high else FACE_RING[::-1]
            unit = np.zeros((4, 3), dtype=np.int64)
            unit[:, axis] = high
            unit[:, (axis + 1) % 3], unit[:, (axis + 2) % 3] = ring[:, 0], ring[:, 1]
            corner = lower[:, None, :] + edge[:, :, None] * unit  # (n, 4, 3)
            middle = (corner + np.roll(corner, -1, axis=1)) // 2  # middle[i] after corner[i]
            sides.append(
                (
                    Octree.keys(corner),
                    Octree.keys(middle),
                    Octree.keys((corner[:, 0] + corner[:, 2]) // 2),
                )
            )
        halved = np.stack([present(middle) for _, middle, _ in sides], axis=1)
        self.transition = np.any(halved, axis=(1, 2))
        rings, cube = [], []
        for (corner, middle, centre), cut in zip(sides, np.moveaxis(halved, 1, 0), strict=True):
            for ring, of in self._face_pieces(corner, middle, centre, cut, present(centre)):
                rings.append(ring)
                cube.append(of)
        self.rings = np.concatenate(rings)
        self.cube = np.concatenate(cube)

    def _face_pieces(self, corner, middle, centre, halved, split):
        """One face of every cube, as the pieces described in the module docstring."""
        n = corner.shape[0]
        everyone = np.arange(n)
        whole = ~np.any(halved, axis=1)
        yield corner[whole], everyone[whole]
        for i in range(4):
            quarter = np.column_stack(
                [corner[split, i], middle[split, i], centre[split], middle[split, i - 1]]
            )
            yield quarter, everyone[split]
        fan = ~whole & ~split
        pad = np.full(n, _UNUSED, dtype=np.int64)
        for i in range(4):
            after = corner[:, (i + 1) % 4]
            plain = fan & ~halved[:, i]
            yield np.column_stack([centre, corner[:, i], after, pad])[plain], everyone[plain]
            cut = fan & halved[:, i]
            yield np.column_stack([centre, corner[:, i], middle[:, i], pad])[cut], everyone[cut]
            yield np.column_stack([centre, middle[:, i], after, pad])[cut], everyone[cut]


def _once(rings: np.ndarray) -> np.ndarray:
    """Whether each face (a ring, a triangle's fourth slot unused) is the only one with its
    corners."""
    keys = np.sort(np.where(rings == _UNUSED, np.iinfo(np.int64).max, rings), axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse.ravel()] == 1


def _without_pinches(faces: _CubeFaces) -> np.ndarray:
    """Which cubes to keep: all, less enough of those at points where their boundary is not a
    manifold that it is one.

    Each round drops, at every pinched point, at most one of the cubes there: the one with the
    largest share of its faces on the boundary (the one sticking out), the smaller of equals.
    """
    kept = np.ones(faces.leaves.size, dtype=bool)
    while True:
        mine = kept[faces.cube]
        rings, cube = faces.rings[mine], faces.cube[mine]
        once = _once(rings)
        pinched = _pinched_points(rings[once])
        if pinched.size == 0:
            return kept
        exposed = np.bincount(cube[once], minlength=kept.size) / np.maximum(
            np.bincount(cube, minlength=kept.size), 1
        )
        touching = np.isin(rings, pinched)
        pairs = np.unique(
            np.column_stack([np.repeat(cube, 4), rings.ravel()])[touching.ravel()], axis=0
        )
        cubes = np.unique(pairs[:, 0])
        cubes = cubes[np.lexsort((cubes, -faces.level[cubes], -exposed[cubes]))]
        start = np.searchsorted(pairs[:, 0], cubes)
        stop = np.searchsorted(pairs[:, 0], cubes, side="right")
        resolved = set()
        for c, first, last in zip(cubes, start, stop, strict=True):
            points = set(pairs[first:last, 1].tolist())
            if resolved.isdisjoint(points):
                kept[c] = False
                resolved |= points


def _pinched_points(boundary: np.ndarray) -> np.ndarray:
    """The points where a closed surface of faces is not a manifold: on an edge of more than two
    faces, or with more than one fan of faces around them (faces joined where two, and only two,
    share an edge at the point).

    Parameters
    ----------
    boundary : np.ndarray of int, shape ``(n_faces, 4)``
        Rings, a triangle's fourth slot unused.

    Returns
    -------
    np.ndarray of int
        The pinched points, each once.
    """
    size = np.where(boundary[:, 3] == _UNUSED, 3, 4)
    used = np.arange(4)[None, :] < size[:, None]
    face, first = np.nonzero(used)
    second = (first + 1) % size[face]
    a, b = boundary[face, first], boundary[face, second]
    incidence_a, incidence_b = 4 * face + first, 4 * face + second
    swap = a > b
    low, high = np.where(swap, b, a), np.where(swap, a, b)
    at_low = np.where(swap, incidence_b, incidence_a)
    at_high = np.where(swap, incidence_a, incidence_b)
    _, edge, counts = np.unique(
        np.column_stack([low, high]), axis=0, return_inverse=True, return_counts=True
    )
    edge = edge.ravel()
    crowded = counts[edge] > 2
    pinched = [low[crowded], high[crowded]]
    pair = np.flatnonzero(counts[edge] == 2)
    pair = pair[np.argsort(edge[pair], kind="stable")].reshape(-1, 2)
    rows = np.concatenate([at_low[pair[:, 0]], at_high[pair[:, 0]]])
    cols = np.concatenate([at_low[pair[:, 1]], at_high[pair[:, 1]]])
    n = 4 * boundary.shape[0]
    graph = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    fans = np.unique(np.column_stack([boundary[face, first], label[4 * face + first]]), axis=0)
    points, fan_count = np.unique(fans[:, 0], return_counts=True)
    pinched.append(points[fan_count > 1])
    return np.unique(np.concatenate(pinched))
