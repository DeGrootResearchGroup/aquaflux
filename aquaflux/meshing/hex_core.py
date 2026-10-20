"""A core of hexahedra from a balanced octree, joined conformally where cube sizes change, its outer
quadrilaterals capped with pyramids for a tetrahedral fill.

**Which cubes.** An :class:`~aquaflux.meshing.Octree` of root cubes of the core's ``size``
covers the region's box, is refined by the criteria given, and is balanced, so cubes sharing a face
or part of an edge differ by at most one level. A leaf is kept when the cube grown by ``clearance``
on every side lies inside the region -- tested on a lattice of points over the grown cube, so exact
for a convex region and a sampling otherwise -- which leaves room between the core and the walls for
the layers and the fill.

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
core's boundary pinch there, which neither a tetrahedral fill nor the dual accepts; every kept cube
at such a point is dropped, and the test repeats until none is left.

**The caps.** Each boundary quadrilateral gets a pyramid built outward, its apex ``cap_height``
times the quadrilateral's edge above the quadrilateral's centre, so the boundary left for the fill
is triangles only: the caps' sides and the boundary triangles of transition cubes. At a concave
right-angled step of the core two caps' sides lean towards each other, so ``cap_height`` must stay
below one half, where they would meet.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .octree import BITS, Octree, RefinementCriterion, cube_samples, point_keys

__all__ = ["HexCore", "OctreeCore"]

#: A cube's corners in units of its edge, as a hexahedron: its lower face, then its upper one, corner
#: over corner.
HEXAHEDRON_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
)

#: A face's corners in the plane of its two other axes, counter-clockwise about its axis.
FACE_RING = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

_UNUSED = -1


def _decode(keys: np.ndarray) -> np.ndarray:
    """The integer coordinates ``point_keys`` encoded; shape ``(..., 3)``."""
    mask = (np.int64(1) << BITS) - 1
    return np.stack([keys >> (2 * BITS), (keys >> BITS) & mask, keys & mask], axis=-1)


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
    """Builds a :class:`HexCore` inside a region.

    Attributes
    ----------
    size : float
        The largest cubes' edge: the root cubes of the octree.
    clearance : float
        The least distance kept between a cube and the region's boundary.
    refinements : sequence of RefinementCriterion
        Where to make the cubes smaller, for example
        :class:`~aquaflux.meshing.NearSurface`.
    cap_height : float
        A cap's height over its quadrilateral's edge, in ``(0, 1/2)``.
    samples : int
        Sample points along each edge of a grown cube for the inside test, at least two.
    """

    size: float
    clearance: float
    refinements: Sequence[RefinementCriterion] = ()
    cap_height: float = 0.25
    samples: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.cap_height < 0.5:
            raise ValueError(f"cap_height must lie in (0, 1/2), not {self.cap_height}")
        if self.clearance < 0.0 or self.samples < 2:
            raise ValueError("clearance must be non-negative and samples at least two")

    def mesh(
        self,
        inside: Callable[[np.ndarray], np.ndarray],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> HexCore:
        """The core of the region ``inside`` holds, within the box from ``lower`` to ``upper``.

        Parameters
        ----------
        inside : callable
            ``inside(points)`` for ``points`` of shape ``(m, 3)`` returns a boolean array of shape
            ``(m,)``.
        lower, upper : sequence of float, length 3
            A box holding the region.

        Returns
        -------
        HexCore

        Raises
        ------
        ValueError
            If no cube fits.
        """
        tree = Octree.lattice(lower, upper, self.size).refined_by(self.refinements).balanced()
        samples = cube_samples(tree.lower, tree.size, self.samples, self.clearance)
        within = np.asarray(inside(samples.reshape(-1, 3)), dtype=bool).reshape(samples.shape[:2])
        candidates = np.flatnonzero(np.all(within, axis=1))
        faces = _CubeFaces(tree, candidates)
        kept = _without_pinches(faces)
        if not np.any(kept):
            raise ValueError("no cube fits in the region at this size and clearance")
        return self._assembled(tree, faces, kept)

    def _assembled(self, tree: Octree, faces: _CubeFaces, kept: np.ndarray) -> HexCore:
        """The cells of the kept cubes and the caps on their boundary, numbered compactly."""
        rings, cube = faces.rings[kept[faces.cube]], faces.cube[kept[faces.cube]]
        transition = faces.transition & kept
        whole = ~faces.transition & kept
        centres = faces.centre[transition]
        keys = np.unique(
            np.concatenate([rings[rings != _UNUSED], centres, faces.hexahedra[whole].ravel()])
        )
        points = tree.origin + 0.5 * tree.unit * _decode(keys)

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
        capped = boundary[boundary[:, 3] != _UNUSED]
        corner = points[capped]
        normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
        edge = np.linalg.norm(corner[:, 1] - corner[:, 0], axis=1)
        height = (self.cap_height * edge / np.linalg.norm(normal, axis=1))[:, None]
        apex = corner.mean(axis=1) + height * normal
        apex_index = points.shape[0] + np.arange(capped.shape[0])
        caps = np.column_stack([capped, apex_index])
        sides = [
            np.column_stack([capped[:, (i + 1) % 4], capped[:, i], apex_index]) for i in range(4)
        ]
        triangles = boundary[boundary[:, 3] == _UNUSED, :3][:, ::-1]
        return HexCore(
            points=np.vstack([points, apex]),
            hexahedra=number(faces.hexahedra[whole]),
            pyramids=pyramids,
            tetrahedra=tetrahedra,
            caps=caps,
            surface=np.vstack([*sides, triangles]),
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
        corners = np.unique(point_keys(offsets))  # every leaf's corners, kept or not
        self.centre = point_keys(lower + edge // 2)
        self.hexahedra = point_keys(lower[:, None, :] + edge[:, :, None] * HEXAHEDRON_CORNERS)

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
                    point_keys(corner),
                    point_keys(middle),
                    point_keys((corner[:, 0] + corner[:, 2]) // 2),
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
