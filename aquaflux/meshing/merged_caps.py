"""Pyramids on quadrilaterals whose apexes are the tetrahedral fill's own points.

A pyramid cap built before the fill (:class:`PyramidCaps`) adds its apex as a point of the mesh,
and so a cell of its dual, squeezed between the quadrilateral and the fill or, taller, crowding its
neighbours. Here no point is added. Each quadrilateral is cut along a diagonal into two triangles,
which the fill meets; afterwards the fill's tetrahedra on that diagonal are made into one pyramid on
the quadrilateral, its apex a point the fill placed (Owen & Saigal (2001), "Formation of pyramid
elements for hexahedra to tetrahedra transitions"):

- **Two tetrahedra round the diagonal** share a third point, the apex: they are one pyramid.
- **More than two** are reduced to two first, each step a 2-3 swap -- the two tetrahedra either
  side of a point ``p`` round the diagonal become three about the edge between ``p``'s neighbours,
  so ``p`` leaves the ring -- choosing the swap whose new tetrahedra are best shaped, and only one
  that turns none of them inside out.
- **Where no swap will do** (or a single tetrahedron spans both triangles), a point is put on the
  diagonal, every tetrahedron round it split there, and the point lifted off the quadrilateral as
  far as it goes, up to ``lift`` times the first edge, without turning a cell inside out.

The diagonal is the ridge one: the one for which each triangle's far corner lies on the side away
from the fill, so no tetrahedron of the fill can hold all four corners; on a flat quadrilateral the
shorter.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["MergeCounts", "MergedCaps"]


@dataclasses.dataclass(frozen=True)
class MergeCounts:
    """How each quadrilateral's pyramid was made: merged as found, after swaps, or with a point of
    its own."""

    direct: int
    swapped: int
    opened: int
    failed: int


@dataclasses.dataclass(frozen=True)
class MergedCaps:
    """Pyramids on quadrilaterals made from the fill's tetrahedra on each one's diagonal.

    Attributes
    ----------
    lift : float
        Where a pyramid needs a point of its own, the most it is lifted off the quadrilateral, over
        the quadrilateral's first edge.
    """

    lift: float = 0.5

    def __post_init__(self) -> None:
        if not self.lift > 0.0:
            raise ValueError(f"MergedCaps.lift must be positive, got {self.lift!r}")

    @staticmethod
    def diagonals(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
        """Each quadrilateral's cut, as the positions of its two ends (``(0, 2)`` or ``(1, 3)``).

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        quads : np.ndarray of int, shape ``(n_quads, 4)``
            Wound so the right-hand normal points into the fill.

        Returns
        -------
        np.ndarray of bool, shape ``(n_quads,)``
            Whether the cut is ``0-2`` (else ``1-3``).
        """
        corner = np.asarray(points, dtype=np.float64)[np.asarray(quads, dtype=np.int64)]
        # Cut 0-2 is a ridge when corner 3 lies below the plane of 0, 1, 2 (away from the fill).
        across = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
        lean = np.einsum("ij,ij->i", corner[:, 3] - corner[:, 0], across)
        size = np.linalg.norm(corner[:, 1] - corner[:, 0], axis=1)
        flat = np.abs(lean) < 1e-9 * np.linalg.norm(across, axis=1) * size
        shorter = np.linalg.norm(corner[:, 2] - corner[:, 0], axis=1) <= np.linalg.norm(
            corner[:, 3] - corner[:, 1], axis=1
        )
        return np.where(flat, shorter, lean < 0.0)

    def triangles(self, points: np.ndarray, quads: np.ndarray) -> np.ndarray:
        """The two triangles of each quadrilateral's cut, wound out of the fill, shape ``(2 n, 3)``."""
        quads = np.asarray(quads, dtype=np.int64)
        halves = self._halves(quads, self.diagonals(points, quads))
        return np.vstack([halves[:, 0, ::-1], halves[:, 1, ::-1]])

    @staticmethod
    def _halves(quads: np.ndarray, cut02: np.ndarray) -> np.ndarray:
        """Each quadrilateral's two triangles, wound as it is, shape ``(n, 2, 3)``."""
        first = np.where(cut02[:, None], quads[:, [0, 1, 2]], quads[:, [0, 1, 3]])
        second = np.where(cut02[:, None], quads[:, [0, 2, 3]], quads[:, [1, 2, 3]])
        return np.stack([first, second], axis=1)

    def merged(
        self, points: np.ndarray, tetrahedra: np.ndarray, quads: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, MergeCounts]:
        """The fill's tetrahedra on each quadrilateral's cut made into a pyramid on it.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        tetrahedra : np.ndarray of int, shape ``(n_tetrahedra, 4)``
            Positively oriented (the fourth corner on the side the first three's right-hand normal
            points to); the fill, whose surface held :meth:`triangles`.
        quads : np.ndarray of int, shape ``(n_quads, 4)``
            As cut by :meth:`triangles`.

        Returns
        -------
        points : np.ndarray
            ``points``, with any point a pyramid needed after them.
        tetrahedra : np.ndarray of int
            The tetrahedra left.
        pyramids : np.ndarray of int, shape ``(n_quads, 5)``
            Each quadrilateral as given, then its apex.
        counts : MergeCounts
        """
        points = np.asarray(points, dtype=np.float64)
        quads = np.asarray(quads, dtype=np.int64)
        cut02 = self.diagonals(points, quads)
        ends = np.where(cut02[:, None], quads[:, [0, 2]], quads[:, [1, 3]])
        sides = np.where(cut02[:, None], quads[:, [1, 3]], quads[:, [2, 0]])
        mesh = _Tetrahedra(points, tetrahedra, {tuple(sorted(e)) for e in ends.tolist()})
        apexes = np.empty(quads.shape[0], dtype=np.int64)
        apexes[:] = -1
        # A pyramid with a side on three corners of another quadrilateral would cover part of
        # that one: both are left to a point of their own instead.
        halves = {}
        for q, row in enumerate(quads.tolist()):
            for k in range(4):
                halves[tuple(sorted(row[:k] + row[k + 1 :]))] = q
        # In passes, the gentlest first: a swap or a split changes tetrahedra a neighbouring
        # quadrilateral's ring may hold, so each is left until every merge that needs none is done.
        direct = swapped = opened = 0
        for stage in ("direct", "swapped", "opened"):
            for q in np.flatnonzero(apexes < 0):
                a, c = (int(v) for v in ends[q])
                b, d = (int(v) for v in sides[q])
                ring = mesh.ring(a, c, b, d)
                if stage == "opened":
                    if ring is not None and len(ring) == 2:
                        # One flat tetrahedron on all four corners: its top is the other cut's
                        # two triangles, with the fill above them -- open the pyramid there.
                        mesh.remove_round(a, c)
                        a, c = b, d
                    elif ring is None:
                        continue
                    apex = mesh.opened(a, c, quads[q], self.lift)
                    if apex is not None:
                        apexes[q] = apex
                        opened += 1
                    continue
                if ring is None or len(ring) < 3:
                    continue
                if len(ring) > 3:
                    if stage == "direct" or not mesh.reduce(a, c, ring):
                        continue
                    ring = mesh.ring(a, c, b, d)
                apex = ring[1]
                corners = quads[q].tolist()
                taken = [
                    halves.get(tuple(sorted((corners[k], corners[(k + 1) % 4], apex))), q)
                    for k in range(4)
                ]
                if any(other != q for other in taken):
                    continue
                if len(ring) > 3 or stage == "swapped":
                    swapped += 1
                else:
                    direct += 1
                apexes[q] = apex
                mesh.remove_round(a, c)
        # No ring, or one tetrahedron on all four corners: nothing to make a pyramid of.
        failed = int(np.count_nonzero(apexes < 0))
        pyramids = np.column_stack([quads, apexes])
        tetrahedra = mesh.array()  # appends the points the pyramids needed
        return mesh.points, tetrahedra, pyramids, MergeCounts(direct, swapped, opened, failed)


def _volume(p, q, r, s) -> float:
    """Six times the signed volume of the tetrahedron ``p q r s``."""
    return float(np.dot(np.cross(q - p, r - p), s - p))


def _shape(p, q, r, s) -> float:
    """A tetrahedron's volume over a regular one's of its rms edge: 1 regular, 0 flat, negative
    inside out."""
    edges = (q - p, r - p, s - p, r - q, s - q, s - r)
    rms = np.sqrt(np.mean([np.dot(e, e) for e in edges]))
    return _volume(p, q, r, s) / 6.0 / (rms**3 / (6.0 * np.sqrt(2.0)))


class _Tetrahedra:
    """The fill's tetrahedra, changed in place, indexed by the cut edges they hold."""

    def __init__(self, points, tetrahedra, cuts: set) -> None:
        self.points = points
        self.cells: dict[int, tuple[int, int, int, int]] = {
            i: tuple(int(v) for v in t) for i, t in enumerate(np.asarray(tetrahedra).tolist())
        }
        self.next_id = len(self.cells)
        self.cuts = cuts
        self.on_cut: dict[tuple[int, int], set[int]] = {cut: set() for cut in cuts}
        for i, cell in self.cells.items():
            self._index(i, cell, add=True)
        self.new_points: list[np.ndarray] = []

    def _index(self, i, cell, add: bool) -> None:
        for m in range(4):
            for n in range(m + 1, 4):
                edge = (min(cell[m], cell[n]), max(cell[m], cell[n]))
                if edge in self.on_cut:
                    (self.on_cut[edge].add if add else self.on_cut[edge].discard)(i)

    def add(self, cell) -> None:
        self.cells[self.next_id] = cell
        self._index(self.next_id, cell, add=True)
        self.next_id += 1

    def drop(self, i) -> None:
        self._index(i, self.cells.pop(i), add=False)

    def coords(self, v) -> np.ndarray:
        n = self.points.shape[0]
        return self.points[v] if v < n else self.new_points[v - n]

    def ring(self, a, c, b, d):
        """The points round edge ``a c`` from ``b`` to ``d``: ``b, p1, ..., d``, or ``None``."""
        around = [self.cells[i] for i in self.on_cut[(min(a, c), max(a, c))]]
        others = [tuple(v for v in cell if v not in (a, c)) for cell in around]
        if not others:
            return None
        ring, seen = [b], set()
        while ring[-1] != d:
            step = [k for k, pair in enumerate(others) if ring[-1] in pair and k not in seen]
            if len(step) != 1:
                return None
            seen.add(step[0])
            pair = others[step[0]]
            ring.append(pair[1] if pair[0] == ring[-1] else pair[0])
        return ring if len(seen) == len(others) else None

    def _oriented(self, cell):
        """``cell`` with its last two corners swapped if it is inside out."""
        p, q, r, s = (self.coords(v) for v in cell)
        return cell if _volume(p, q, r, s) > 0 else (cell[0], cell[1], cell[3], cell[2])

    def reduce(self, a, c, ring) -> bool:
        """2-3 swaps until two tetrahedra are left round ``a c``; whether it got there."""
        while len(ring) > 3:
            best, best_shape = None, 0.0
            for k in range(1, len(ring) - 1):
                before, at, after = ring[k - 1], ring[k], ring[k + 1]
                new = [(a, c, before, after), (a, before, at, after), (c, before, at, after)]
                shapes = []
                for cell in new:
                    oriented = self._oriented(cell)
                    shapes.append(_shape(*(self.coords(v) for v in oriented)))
                # The three must fill the two they replace: none flat, and the new edge
                # ``before after`` must cross the triangle ``a c at``.
                p_a, p_c, p_at = self.coords(a), self.coords(c), self.coords(at)
                p_b, p_f = self.coords(before), self.coords(after)
                crosses = _segment_crosses_triangle(p_b, p_f, p_a, p_c, p_at)
                if crosses and min(shapes) > best_shape:
                    best, best_shape = k, min(shapes)
            if best is None:
                return False
            before, at, after = ring[best - 1], ring[best], ring[best + 1]
            for i in [
                i
                for i in list(self.on_cut[(min(a, c), max(a, c))])
                if {before, at} <= set(self.cells[i]) or {at, after} <= set(self.cells[i])
            ]:
                self.drop(i)
            for cell in ((a, c, before, after), (a, before, at, after), (c, before, at, after)):
                self.add(self._oriented(cell))
            ring = ring[:best] + ring[best + 1 :]
        return True

    def around(self, a, c) -> list[int]:
        """The tetrahedra holding edge ``a c``: by the index on a cut, else by a search."""
        edge = (min(a, c), max(a, c))
        if edge in self.on_cut:
            return list(self.on_cut[edge])
        return [i for i, cell in self.cells.items() if a in cell and c in cell]

    def remove_round(self, a, c) -> None:
        for i in self.around(a, c):
            self.drop(i)

    def opened(self, a, c, quad, lift: float) -> int | None:
        """A new point on ``a c``, every tetrahedron round it split there, then moved to the best
        of a few places off ``quad`` -- along its normal up to ``lift`` times its first edge, or
        towards the points round it -- by the worst shape of the cells it is in, the pyramid on
        ``quad`` included; its index, or ``None`` (nothing changed) where every place turns one
        inside out."""
        corner = np.array([self.coords(int(v)) for v in quad])
        normal = np.cross(corner[2] - corner[0], corner[3] - corner[1])
        normal /= np.linalg.norm(normal)
        middle = corner.mean(axis=0)
        around = self.around(a, c)
        ring = {v for i in around for v in self.cells[i] if v not in (a, c)}
        neighbours = np.mean([self.coords(v) for v in ring], axis=0)
        height = lift * np.linalg.norm(corner[1] - corner[0])
        places = [middle + t * height * normal for t in (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1)]
        places += [middle + w * (neighbours - middle) for w in (0.25, 0.5, 0.75)]
        index = self.points.shape[0] + len(self.new_points)
        split = [
            cell
            for i in around
            for cell in (
                tuple(index if v == c else v for v in self.cells[i]),
                tuple(index if v == a else v for v in self.cells[i]),
            )
        ]
        # The pyramid's two tetrahedra on the quadrilateral's halves, wound towards its apex.
        base = [(int(quad[0]), int(quad[1]), int(quad[2]), index),
                (int(quad[0]), int(quad[2]), int(quad[3]), index)]  # fmt: skip
        self.new_points.append(middle)
        best, best_shape = None, 0.0
        for place in places:
            self.new_points[-1] = place
            worst = min(_shape(*(self.coords(v) for v in cell)) for cell in split + base)
            if worst > best_shape:
                best, best_shape = place, worst
        if best is None:
            self.new_points.pop()
            return None
        self.new_points[-1] = best
        for i in around:
            self.drop(i)
        for cell in split:
            self.add(cell)
        return index

    def array(self) -> np.ndarray:
        self.points = np.vstack([self.points, *self.new_points]) if self.new_points else self.points
        self.new_points = []
        return np.array(list(self.cells.values()), dtype=np.int64).reshape(-1, 4)


def _segment_crosses_triangle(p, q, a, b, c) -> bool:
    """Whether the segment ``p q`` passes through the triangle ``a b c``."""
    side_p = _volume(a, b, c, p)
    side_q = _volume(a, b, c, q)
    if side_p * side_q >= 0.0:
        return False
    s1 = _volume(p, q, a, b)
    s2 = _volume(p, q, b, c)
    s3 = _volume(p, q, c, a)
    return (s1 > 0 and s2 > 0 and s3 > 0) or (s1 < 0 and s2 < 0 and s3 < 0)
