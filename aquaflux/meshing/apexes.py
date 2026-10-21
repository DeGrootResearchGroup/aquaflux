"""The apexes of the pyramids that cap quadrilaterals for a tetrahedral fill, moved after the fill.

A tetrahedral fill meets triangles only, so each quadrilateral facing it is capped with a pyramid
(:class:`~aquaflux.meshing.PyramidCaps`) whose apex is placed before the fill runs, a fixed share
of the quadrilateral's edge above it. The fill keeps every point it is given where it is, so the
apex stays there however the tetrahedra fall round it -- and each apex is a primal point, so a cell
of the polyhedral dual: a small one where the apex is low and the tetrahedra round it are not.

After the fill an apex is an ordinary interior point, and :class:`ApexRelaxation` moves it towards
the mean of the points it shares a cell with (a Laplacian sweep, all apexes at once), sweep after
sweep until the moves are small. A move is kept only where no cell about the apex becomes worse
shaped than it was before the first sweep, or than a floor: where one would, the apexes in it go
half as far, and in the end not at all.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["ApexRelaxation", "RelaxedApexes"]

#: The four tetrahedra a pyramid is judged by: each corner of its quadrilateral with the corners
#: either side and the apex (corners ``0..3`` in ring order, the apex ``4``).
_PYRAMID_CORNERS = np.array([[3, 0, 1, 4], [0, 1, 2, 4], [1, 2, 3, 4], [2, 3, 0, 4]])

#: The six edges of a tetrahedron, as pairs of its corners.
_TETRAHEDRON_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


@dataclasses.dataclass(frozen=True)
class RelaxedApexes:
    """Where the apexes went.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        Every point, the apexes moved.
    sweeps : int
        The sweeps made.
    converged : bool
        Whether the last sweep moved no apex further than the tolerance.
    """

    points: np.ndarray
    sweeps: int
    converged: bool


@dataclasses.dataclass(frozen=True)
class ApexRelaxation:
    """Moves the caps' apexes towards the middle of the points they share a cell with.

    Attributes
    ----------
    max_sweeps : int
        The most sweeps made.
    tolerance : float
        Sweeps stop once no apex moves further than this share of its mean distance to the points
        it shares a cell with.
    min_quality : float
        The shape below which a move may not take a cell that was above it, as the volume of a
        tetrahedron over that of a regular one with its root-mean-square edge (one for a regular
        tetrahedron); a pyramid is judged by the tetrahedra at its four corners. A cell already
        below it may not get worse.
    max_halvings : int
        How many times a sweep halves the moves of the apexes of a cell that would get worse
        before it leaves them where they are.
    """

    max_sweeps: int = 20
    tolerance: float = 1e-3
    min_quality: float = 0.1
    max_halvings: int = 6

    def __post_init__(self) -> None:
        if self.max_sweeps < 1 or self.max_halvings < 0:
            raise ValueError("ApexRelaxation takes at least one sweep and no negative halvings.")
        if not (0.0 < self.tolerance < 1.0 and 0.0 <= self.min_quality < 1.0):
            raise ValueError(
                "ApexRelaxation.tolerance must lie in (0, 1) and min_quality in [0, 1), got "
                f"{self.tolerance!r} and {self.min_quality!r}."
            )

    def relaxed(
        self,
        points: np.ndarray,
        tetrahedra: np.ndarray,
        pyramids: np.ndarray,
        apexes: np.ndarray,
    ) -> RelaxedApexes:
        """The points with the ``apexes`` moved.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        tetrahedra : np.ndarray of int, shape ``(n_tetrahedra, 4)``
        pyramids : np.ndarray of int, shape ``(n_pyramids, 5)``
            Each pyramid's quadrilateral in ring order, then its apex.
        apexes : np.ndarray of int, shape ``(n_apexes,)``
            The points to move: interior points, each the apex of a pyramid.

        Returns
        -------
        RelaxedApexes
        """
        points = np.array(points, dtype=np.float64)
        apexes = np.unique(np.asarray(apexes, dtype=np.int64))
        if not apexes.size:
            return RelaxedApexes(points, 0, True)
        cells = _Cells.about(points, tetrahedra, pyramids, apexes)
        floor = np.minimum(cells.quality(points), self.min_quality)
        sweeps, converged = 0, False
        while sweeps < self.max_sweeps and not converged:
            sweeps += 1
            step = cells.neighbour_mean(points) - points[apexes]
            share = self._shares(points, apexes, step, cells, floor)
            moved = share[:, None] * step
            points[apexes] += moved
            reach = np.linalg.norm(moved, axis=1) / cells.spread(points)
            converged = bool(reach.max() < self.tolerance)
        return RelaxedApexes(points, sweeps, converged)

    def _shares(self, points, apexes, step, cells: _Cells, floor) -> np.ndarray:
        """How far along its step each apex may go: one, halved for the apexes of any cell the
        moves would take below its floor, and zero once the halvings run out."""
        share = np.ones(apexes.size)
        halvings = 0
        while True:
            trial = points.copy()
            trial[apexes] += share[:, None] * step
            worse = cells.quality(trial) < floor
            if not np.any(worse):
                return share
            blamed = cells.apexes_of(worse)
            if halvings < self.max_halvings:
                share[blamed] *= 0.5
                halvings += 1
            else:
                share[blamed] = 0.0


@dataclasses.dataclass(frozen=True)
class _Cells:
    """The tetrahedra a sweep judges -- those holding an apex, and each pyramid about one cut into
    its four corner tetrahedra -- and the points each apex shares a cell with.

    Attributes
    ----------
    apexes : np.ndarray of int, shape ``(n_apexes,)``
        The points moved, ascending.
    corners : np.ndarray of int, shape ``(n, 4)``
        Each tetrahedron judged.
    sign : np.ndarray, shape ``(n,)``
        ``+1`` or ``-1``: the sign of its volume before any move.
    owner : np.ndarray of int, shape ``(n, 4)``
        For each corner, its position among the apexes, or ``-1``.
    pairs : np.ndarray of int, shape ``(m, 2)``
        Every (apex position, point) pair sharing a tetrahedron, each once.
    """

    apexes: np.ndarray
    corners: np.ndarray
    sign: np.ndarray
    owner: np.ndarray
    pairs: np.ndarray

    @classmethod
    def about(
        cls, points: np.ndarray, tetrahedra: np.ndarray, pyramids: np.ndarray, apexes: np.ndarray
    ) -> _Cells:
        """The cells about ``apexes`` (ascending), oriented as they stand at ``points``."""
        tetrahedra = np.asarray(tetrahedra, dtype=np.int64).reshape(-1, 4)
        pyramids = np.asarray(pyramids, dtype=np.int64).reshape(-1, 5)
        held = tetrahedra[np.any(np.isin(tetrahedra, apexes), axis=1)]
        capped = pyramids[np.any(np.isin(pyramids, apexes), axis=1)]
        corners = np.concatenate([held, *(capped[:, corner] for corner in _PYRAMID_CORNERS)])
        # A cap's sides are faces of the fill, so its quadrilateral's corners share a tetrahedron
        # with its apex: the tetrahedra alone name every point an apex shares a cell with.
        pairs = np.unique(_apex_pairs(held, apexes), axis=0)
        if np.any(np.bincount(pairs[:, 0], minlength=apexes.size) == 0):
            raise ValueError("an apex is in no tetrahedron given")
        cells = cls(apexes, corners, np.ones(corners.shape[0]), _positions(corners, apexes), pairs)
        return dataclasses.replace(cells, sign=np.where(cells.signed(points) < 0.0, -1.0, 1.0))

    def signed(self, points: np.ndarray) -> np.ndarray:
        """Each tetrahedron's shape: its volume over a regular tetrahedron's of the same
        root-mean-square edge, signed."""
        corner = points[self.corners]
        edges = corner[:, 1:] - corner[:, :1]
        volume = np.einsum("ij,ij->i", np.cross(edges[:, 0], edges[:, 1]), edges[:, 2]) / 6.0
        lengths = corner[:, _TETRAHEDRON_EDGES[:, 1]] - corner[:, _TETRAHEDRON_EDGES[:, 0]]
        rms = np.sqrt(np.mean(np.sum(lengths**2, axis=2), axis=1))
        return volume / np.maximum(rms**3 / (6.0 * np.sqrt(2.0)), np.finfo(float).tiny)

    def quality(self, points: np.ndarray) -> np.ndarray:
        """Each tetrahedron's shape (see :meth:`signed`), positive as it was before any move."""
        return self.sign * self.signed(points)

    def apexes_of(self, chosen: np.ndarray) -> np.ndarray:
        """The positions of the apexes among the corners of the ``chosen`` tetrahedra."""
        owner = self.owner[chosen]
        return np.unique(owner[owner >= 0])

    def neighbour_mean(self, points: np.ndarray) -> np.ndarray:
        """The mean of the points each apex shares a cell with; shape ``(n_apexes, 3)``."""
        count = np.bincount(self.pairs[:, 0], minlength=self.apexes.size)
        total = np.zeros((self.apexes.size, 3))
        np.add.at(total, self.pairs[:, 0], points[self.pairs[:, 1]])
        return total / count[:, None]

    def spread(self, points: np.ndarray) -> np.ndarray:
        """Each apex's mean distance to the points it shares a cell with; shape ``(n_apexes,)``."""
        apart = points[self.pairs[:, 1]] - points[self.apexes[self.pairs[:, 0]]]
        count = np.bincount(self.pairs[:, 0], minlength=self.apexes.size)
        total = np.bincount(
            self.pairs[:, 0], weights=np.linalg.norm(apart, axis=1), minlength=self.apexes.size
        )
        return total / count


def _positions(corners: np.ndarray, apexes: np.ndarray) -> np.ndarray:
    """Each corner's position among the (ascending) ``apexes``, or ``-1``."""
    position = np.minimum(np.searchsorted(apexes, corners), apexes.size - 1)
    return np.where(apexes[position] == corners, position, -1)


def _apex_pairs(cells: np.ndarray, apexes: np.ndarray) -> np.ndarray:
    """Every (apex position, other corner) pair within ``cells``; shape ``(m, 2)``."""
    if not cells.size:
        return np.zeros((0, 2), dtype=np.int64)
    position = _positions(cells, apexes)
    width = cells.shape[1]
    pairs = [
        np.stack([position[:, k], cells[:, other]], axis=1)[position[:, k] >= 0]
        for k in range(width)
        for other in range(width)
        if other != k
    ]
    return np.concatenate(pairs)
