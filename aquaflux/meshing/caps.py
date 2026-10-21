"""Pyramid caps as tall as there is room for.

A quadrilateral the tetrahedral fill meets is capped with a pyramid (:class:`PyramidCaps`), and the
cap's apex is a point of the mesh, so of its dual: a cap much lower than the spacing of the fill
around it squeezes a small cell between the quadrilateral and the fill. A pyramid's best shape has
its apex about one edge above its base, but two neighbouring caps lean towards each other where the
surface folds concavely, and a cap must stop well short of the surface across from it.
:class:`ClearedCaps` gives each cap a target height and lowers it where either limit binds.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .merged_caps import MergedCaps
from .polygons import Polygons, PyramidCaps
from .regions import first_hits

__all__ = ["ClearedCaps", "capped"]


def capped(
    points: np.ndarray,
    surface: Polygons,
    height: float | ClearedCaps | MergedCaps,
    clearance: float = np.inf,
) -> PyramidCaps:
    """Caps on the quadrilaterals of ``surface``, ``height`` a share of each one's first edge
    (:meth:`PyramidCaps.on`) or sized by :class:`ClearedCaps`.

    Parameters
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
    surface : Polygons
        As :meth:`ClearedCaps.rises`: wound so each face's normal points into the fill.
    height : float or ClearedCaps
    clearance : float
        As :meth:`ClearedCaps.rises`.

    Returns
    -------
    PyramidCaps
        On ``surface.quads``, in their order.
    """
    if isinstance(height, MergedCaps):
        # No apex yet (-1): the fill meets each quadrilateral's two triangles, and its own
        # tetrahedra on them become the pyramid afterwards (MergedCaps.merged).
        quads = surface.quads
        pending = np.column_stack([quads, np.full(quads.shape[0], -1, dtype=np.int64)])
        return PyramidCaps(np.zeros((0, 3)), pending, height.triangles(points, quads))
    if isinstance(height, ClearedCaps):
        return PyramidCaps.at(points, surface.quads, height.rises(points, surface, clearance))
    return PyramidCaps.on(points, surface.quads, height)


@dataclasses.dataclass(frozen=True)
class ClearedCaps:
    """Each cap ``target`` times its quadrilateral's first edge high, lowered where room runs out.

    - **At a fold**: across each edge of the quadrilateral the cap's side leans at most ``fold``
      times half the angle the fill has there (between this face and the one across the edge), so
      two caps leaning towards each other over a concave edge stay apart. Over a right-angled
      concave edge, ``fold`` 0.6 gives a quarter of the distance from the centre to the edge; over a
      flat one about 0.69 of the quadrilateral's width.
    - **Before the surface opposite**: no more than ``gap`` times the distance from the
      quadrilateral's centre, along its normal, to the surface again (and to a ``clearance`` the
      caller knows is free, when given).

    Attributes
    ----------
    target : float
        The height wanted, over the quadrilateral's first edge (see :meth:`PyramidCaps.on` for why
        the first edge).
    fold : float
        The share of half the fill's angle at an edge a cap's side may lean, in ``(0, 1)``.
    gap : float
        The share of the room in front of a cap it may take, in ``(0, 1)``.
    """

    target: float = 0.7
    fold: float = 0.6
    gap: float = 0.4

    def __post_init__(self) -> None:
        if not self.target > 0.0:
            raise ValueError(f"ClearedCaps.target must be positive, got {self.target!r}")
        for name in ("fold", "gap"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"ClearedCaps.{name} must lie in (0, 1), got {value!r}")

    def rises(self, points: np.ndarray, surface: Polygons, clearance: float = np.inf) -> np.ndarray:
        """Each cap's height over its quadrilateral, for the quadrilaterals of ``surface``.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        surface : Polygons
            The closed surface the fill meets on the caps' side, wound so each face's right-hand
            normal points into the fill: its quadrilaterals are the ones capped, its other faces
            what the caps must keep clear of.
        clearance : float
            A distance in front of every quadrilateral known to be free beyond ``surface``.

        Returns
        -------
        np.ndarray, shape ``(n_quads,)``
            Heights, lengths, in the order of ``surface.quads``.
        """
        points = np.asarray(points, dtype=np.float64)
        n_triangles = surface.triangles.shape[0]
        quads = surface.quads
        corner = points[quads]
        centre = corner.mean(axis=1)
        normal = surface.normals(points)
        own = normal[n_triangles:]
        rise = self.target * np.linalg.norm(corner[:, 1] - corner[:, 0], axis=1)
        rise = np.minimum(rise, self._fold_limits(points, surface, centre, own))
        reach = rise / self.gap
        room = first_hits(points, surface, centre, own, reach, quads)
        return np.minimum(rise, self.gap * np.minimum(room, clearance))

    def _fold_limits(self, points, surface: Polygons, centre, own) -> np.ndarray:
        """The height each quadrilateral's cap may reach before a side leans past ``fold`` of half
        the fill's angle at that edge."""
        n_triangles = surface.triangles.shape[0]
        pairs, face = surface.edges()
        key = np.sort(pairs, axis=1)
        order = np.lexsort(key.T[::-1])
        key, face, pairs = key[order], face[order], pairs[order]
        # On a closed surface each edge is two faces' -- consecutive once sorted.
        same = np.all(key[1:] == key[:-1], axis=1)
        first = np.flatnonzero(same)
        across = np.full(face.size, -1)
        across[first], across[first + 1] = face[first + 1], face[first]
        normals = surface.normals(points)
        centroids = surface.centroids(points)
        limit = np.full(surface.quads.shape[0], np.inf)
        mine = (face >= n_triangles) & (across >= 0)
        quad = face[mine] - n_triangles
        other = across[mine]
        a, b = points[pairs[mine, 0]], points[pairs[mine, 1]]
        middle = 0.5 * (a + b)
        along = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
        cosine = np.clip(np.einsum("ij,ij->i", own[quad], normals[other]), -1.0, 1.0)
        concave = np.einsum("ij,ij->i", own[quad], centroids[other] - middle) > 0.0
        angle = np.pi + np.where(concave, -1.0, 1.0) * np.arccos(cosine)  # the fill's, at the edge
        lean = self.fold * 0.5 * angle
        to_edge = centre[quad] - middle
        width = np.linalg.norm(
            to_edge - np.einsum("ij,ij->i", to_edge, along)[:, None] * along, axis=1
        )
        bound = np.where(
            lean < 0.5 * np.pi, width * np.tan(np.minimum(lean, 0.5 * np.pi - 1e-9)), np.inf
        )
        np.minimum.at(limit, quad, bound)
        return limit
