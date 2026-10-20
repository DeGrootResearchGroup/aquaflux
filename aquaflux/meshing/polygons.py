"""A surface of triangles and quadrilaterals, and the pyramids that cap its quadrilaterals.

A surface handed between the steps of the mesher -- a wall meshed partly with structured
quadrilaterals, the top of the layers grown on it -- carries both kinds of face. They are kept as two
arrays rather than one padded one, so a face is never read with a missing corner; a quantity per face
runs over the triangles first, then the quadrilaterals.

A tetrahedral fill takes triangles only, so each quadrilateral of the surface it is given is first
capped with a pyramid built on the side being filled: the pyramid's four triangular sides take the
quadrilateral's place in the fill's surface, and the pyramid joins the mesh as a cell.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Polygons", "PyramidCaps"]


@dataclasses.dataclass(frozen=True)
class Polygons:
    """Triangles and quadrilaterals by their corners, each wound the same way round the surface.

    Attributes
    ----------
    triangles : np.ndarray of int, shape ``(n_triangles, 3)``
    quads : np.ndarray of int, shape ``(n_quads, 4)``
        Each quadrilateral's corners in order round it.
    """

    triangles: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 3), np.int64))
    quads: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 4), np.int64))

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "triangles", np.asarray(self.triangles, dtype=np.int64).reshape(-1, 3)
        )
        object.__setattr__(self, "quads", np.asarray(self.quads, dtype=np.int64).reshape(-1, 4))

    @property
    def n_faces(self) -> int:
        """Number of faces, triangles and quadrilaterals."""
        return self.triangles.shape[0] + self.quads.shape[0]

    def kinds(self) -> tuple[np.ndarray, np.ndarray]:
        """The triangles and the quadrilaterals."""
        return self.triangles, self.quads

    def split(self, per_face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A quantity given per face, as its triangles' part and its quadrilaterals' part."""
        per_face = np.asarray(per_face)
        return per_face[: self.triangles.shape[0]], per_face[self.triangles.shape[0] :]

    def where(self, chosen: np.ndarray) -> Polygons:
        """The faces chosen by a boolean per face."""
        triangles, quads = self.split(np.asarray(chosen, dtype=bool))
        return Polygons(self.triangles[triangles], self.quads[quads])

    def renumbered(self, points: np.ndarray) -> Polygons:
        """The faces with each corner ``c`` replaced by ``points[c]``."""
        points = np.asarray(points, dtype=np.int64)
        return Polygons(points[self.triangles], points[self.quads])

    def reversed(self) -> Polygons:
        """The faces wound the other way round."""
        return Polygons(self.triangles[:, ::-1], self.quads[:, ::-1])

    def padded(self) -> np.ndarray:
        """Every face as four corners, a triangle's fourth ``-1``; shape ``(n_faces, 4)``."""
        rings = np.full((self.n_faces, 4), -1, dtype=np.int64)
        rings[: self.triangles.shape[0], :3] = self.triangles
        rings[self.triangles.shape[0] :] = self.quads
        return rings

    def corners(self) -> np.ndarray:
        """Each face's number of corners, shape ``(n_faces,)``."""
        return np.concatenate(
            [np.full(self.triangles.shape[0], 3), np.full(self.quads.shape[0], 4)]
        ).astype(np.int64)

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Every face's edges in winding order, as point pairs, and the face of each.

        Returns
        -------
        pairs : np.ndarray of int, shape ``(n, 2)``
        face : np.ndarray of int, shape ``(n,)``
        """
        pairs, face, base = [], [], 0
        for rings in self.kinds():
            pairs.append(np.stack([rings, np.roll(rings, -1, axis=1)], axis=2).reshape(-1, 2))
            face.append(np.repeat(base + np.arange(rings.shape[0]), rings.shape[1]))
            base += rings.shape[0]
        return np.concatenate(pairs), np.concatenate(face)

    def normals(self, points: np.ndarray) -> np.ndarray:
        """Each face's unit normal by the right-hand rule, shape ``(n_faces, 3)``.

        A quadrilateral's is that of its two diagonals, which is the mean normal of a face that is
        not flat (the Newell normal of four points).
        """
        points = np.asarray(points, dtype=np.float64)
        t = points[self.triangles]
        q = points[self.quads]
        normal = np.concatenate(
            [
                np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]),
            ]
        )
        return normal / np.linalg.norm(normal, axis=1, keepdims=True)

    def centroids(self, points: np.ndarray) -> np.ndarray:
        """Each face's mean corner, shape ``(n_faces, 3)``."""
        points = np.asarray(points, dtype=np.float64)
        return np.concatenate(
            [points[self.triangles].mean(axis=1), points[self.quads].mean(axis=1)]
        )

    def as_triangles(self) -> np.ndarray:
        """Every face as triangles: a triangle itself, a quadrilateral cut along its first diagonal
        (``a, b, c`` and ``a, c, d``); shape ``(n, 3)``. For a test on a surface, not a mesh."""
        q = self.quads
        return np.concatenate([self.triangles, q[:, [0, 1, 2]], q[:, [0, 2, 3]]])

    def triangle_face(self) -> np.ndarray:
        """The face each row of :meth:`as_triangles` comes from, shape ``(n,)``."""
        n_t, n_q = self.triangles.shape[0], self.quads.shape[0]
        quads = n_t + np.arange(n_q)
        return np.concatenate([np.arange(n_t), quads, quads])


@dataclasses.dataclass(frozen=True)
class PyramidCaps:
    """Pyramids on quadrilaterals, and the triangles that take the quadrilaterals' place.

    Attributes
    ----------
    apexes : np.ndarray, shape ``(n_quads, 3)``
        Each pyramid's apex.
    pyramids : np.ndarray of int, shape ``(n_quads, 5)``
        Each pyramid's quadrilateral as given, then its apex, numbered after ``n_points`` points.
    sides : np.ndarray of int, shape ``(4 n_quads, 3)``
        The pyramids' triangular sides, wound into the pyramid: out of the region beyond the caps,
        which is how a fill wants the surface of the region it fills.
    """

    apexes: np.ndarray
    pyramids: np.ndarray
    sides: np.ndarray

    @staticmethod
    def check_height(height: float, owner: str) -> None:
        """Refuse a cap height outside ``(0, 1/2)``, naming the setting ``owner`` holds it as.

        Raises
        ------
        ValueError
        """
        if not 0.0 < height < 0.5:
            raise ValueError(f"{owner} must lie in (0, 1/2), got {height!r}.")

    @classmethod
    def on(cls, points: np.ndarray, quads: np.ndarray, height: float) -> PyramidCaps:
        """Caps on ``quads``, each apex ``height`` times the quadrilateral's first edge above its
        centre, on the side its right-hand normal points to.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        quads : np.ndarray of int, shape ``(n_quads, 4)``
        height : float
            Below one half: at a concave right-angled corner two caps' sides lean towards each
            other, and meet at one half.

        Returns
        -------
        PyramidCaps
            The apexes numbered ``n_points`` onward, in the order of ``quads``.

        Raises
        ------
        ValueError
            If ``height`` is not in ``(0, 1/2)``.
        """
        PyramidCaps.check_height(height, "a cap's height")
        points = np.asarray(points, dtype=np.float64)
        quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
        corner = points[quads]
        normal = np.cross(corner[:, 2] - corner[:, 0], corner[:, 3] - corner[:, 1])
        edge = np.linalg.norm(corner[:, 1] - corner[:, 0], axis=1)
        rise = (height * edge / np.linalg.norm(normal, axis=1))[:, None]
        apexes = corner.mean(axis=1) + rise * normal
        apex = points.shape[0] + np.arange(quads.shape[0])
        sides = np.concatenate(
            [np.column_stack([quads[:, (i + 1) % 4], quads[:, i], apex]) for i in range(4)]
        )
        return cls(apexes, np.column_stack([quads, apex]), sides)
