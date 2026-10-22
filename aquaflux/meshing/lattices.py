"""Lattices of hexahedral blocks, laid out by coordinates and shaped by editing arrays.

A lattice is a grid of blocks whose corners are the tensor product of three coordinate lists, in a
frame: Cartesian ``(x, y, z)`` or cylindrical ``(r, theta, z)``. Each interval between successive
coordinates carries its own division, shared by every block in that row, so blocks that meet always
agree on their shared cells. After it is laid out, a lattice is shaped by editing its arrays in
place with numpy indexing:

* ``corners[i, j, k]`` -- a corner's frame coordinates (move a row of corners, twist a group);
* ``divisions[axis][interval]`` -- a cell count, a :class:`~aquaflux.meshing.Graded` progression or
  node fractions;
* ``mask[i, j, k]`` -- leave block ``(i, j, k)`` out (an L-shape, a window);
* ``edges[i, j, k, axis]`` -- the edge from corner ``(i, j, k)`` one step along ``axis``: unset, the
  frame's own edge; :class:`Straight`; or any :class:`~aquaflux.meshing.Curve` from its start to its
  end;
* ``projections[i, j, k, axis]`` -- the face normal to ``axis`` at corner index ``i, j or k`` along it
  and block indices along the other two: unset, the frame's own (a cylindrical lattice projects a
  face of constant radius onto its cylinder, and a face whose radius changes along the axis onto its
  cone); ``False`` for none; or any projection;
* ``patches[i, j, k, axis]`` -- the same faces' boundary patch names.

:meth:`BlockLattice.blocks` then gives the :class:`~aquaflux.meshing.HexBlock` of every block left in.
In a cylindrical frame an edge is the image of the straight segment between its ends' coordinates:
an arc around the axis, a straight radial or axial line, a helix if both change.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np

from .blocks import FACES, HexBlock
from .elements import HEXAHEDRON_CORNERS
from .surfaces import ConeSurface, CylinderSurface

__all__ = ["BlockLattice", "CartesianFrame", "CylindricalFrame", "Straight"]


@dataclasses.dataclass(frozen=True)
class Straight:
    """A lattice edge that is a straight line between its corners, whatever the frame."""


@dataclasses.dataclass(frozen=True)
class CartesianFrame:
    """Coordinates ``(x, y, z)``: edges are straight, and no face is projected by default."""

    def points(self, coordinates: np.ndarray) -> np.ndarray:
        """Cartesian points of frame coordinates, shape kept."""
        return np.asarray(coordinates, dtype=np.float64)

    def edge(self, start: np.ndarray, end: np.ndarray):
        """The frame's edge between two corners' coordinates: ``None``, a straight line."""
        return None

    def projection(self, axis: int, corners: np.ndarray):
        """The frame's projection of a face: ``None``, none."""
        return None


@dataclasses.dataclass(frozen=True)
class CylindricalFrame:
    """Coordinates ``(r, theta, z)`` about the ``z`` axis, ``theta`` in radians from ``x``.

    An edge is the image of the straight segment between its ends' coordinates. A face normal to
    ``r`` whose corners share one radius is projected onto that cylinder; one whose radius is the
    same at each of its two heights but differs between them, onto that cone.
    """

    def points(self, coordinates: np.ndarray) -> np.ndarray:
        """Cartesian points of frame coordinates, shape kept.

        Raises
        ------
        ValueError
            If a radius is not positive (a block cannot close onto the axis).
        """
        coordinates = np.asarray(coordinates, dtype=np.float64)
        r, theta, z = np.moveaxis(coordinates, -1, 0)
        if np.any(r <= 0.0):
            raise ValueError(
                "a cylindrical lattice's radii must be positive: a block cannot close onto the "
                "axis (mesh a full pipe with a square core instead)"
            )
        return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)

    def edge(self, start: np.ndarray, end: np.ndarray):
        """The edge between two corners' coordinates: straight at one angle, else a curve.

        Raises
        ------
        ValueError
            If the edge turns half a revolution or more.
        """
        turn = abs(float(end[1] - start[1]))
        if turn >= math.pi:
            raise ValueError(
                f"a cylindrical lattice edge turns {math.degrees(turn):.1f} degrees; each must turn "
                "less than 180"
            )
        return None if turn == 0.0 else _FrameSegment(self, np.asarray(start), np.asarray(end))

    def projection(self, axis: int, corners: np.ndarray):
        """A face's projection from its corners' coordinates ``(2, 2, 3)``, if on a cylinder or cone."""
        if axis != 0:
            return None
        r, z = corners[..., 0], corners[..., 2]  # indexed (theta, z)
        if np.all(r == r[0, 0]):
            return CylinderSurface(float(r[0, 0]))
        if np.all(r[0] == r[1]) and np.all(z[0] == z[1]) and z[0, 0] != z[0, 1]:
            return ConeSurface(
                np.array([0.0, 0.0, z[0, 0]]),
                np.array([0.0, 0.0, z[0, 1]]),
                float(r[0, 0]),
                float(r[0, 1]),
            )
        return None


@dataclasses.dataclass(frozen=True)
class _FrameSegment:
    """The image in Cartesian space of a straight segment in a frame's coordinates."""

    frame: CylindricalFrame
    start: np.ndarray
    end: np.ndarray

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)[:, None]
        return self.frame.points((1.0 - t) * self.start + t * self.end)


class BlockLattice:
    """A grid of hexahedral blocks laid out by three coordinate lists in a frame.

    Parameters
    ----------
    frame : CartesianFrame or CylindricalFrame
    coordinates : sequence of three array-likes
        Each direction's block boundaries, increasing, at least two.
    divisions : sequence of three
        Each direction's divisions: one division (a count, a
        :class:`~aquaflux.meshing.Graded` or an array of node fractions) for every interval, or a
        list or tuple of one per interval.

    Attributes
    ----------
    frame : CartesianFrame or CylindricalFrame
    corners : np.ndarray, shape ``(n0 + 1, n1 + 1, n2 + 1, 3)``
        Every block corner's frame coordinates; edit in place.
    divisions : list of three lists
        Each direction's division per interval; edit in place.
    mask : np.ndarray of bool, shape ``(n0, n1, n2)``
        Blocks left out.
    edges, projections, patches : np.ndarray of object, shape ``(n0 + 1, n1 + 1, n2 + 1, 3)``
        Edge, face-projection and patch-name overrides, ``None`` where unset (see the module).

    Raises
    ------
    ValueError
        If a coordinate list does not increase or has fewer than two entries, or a list of
        divisions has the wrong length.
    """

    def __init__(self, frame, coordinates, divisions) -> None:
        coordinates = [np.asarray(c, dtype=np.float64) for c in coordinates]
        if len(coordinates) != 3 or len(divisions) != 3:
            raise ValueError("a lattice takes three coordinate lists and three divisions")
        for axis, c in enumerate(coordinates):
            if c.ndim != 1 or c.size < 2 or np.any(np.diff(c) <= 0.0):
                raise ValueError(
                    f"lattice coordinates along axis {axis} must increase, two or more"
                )
        shape = tuple(c.size for c in coordinates)
        self.frame = frame
        self.corners = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
        self.divisions = [
            _per_interval(d, n - 1, axis)
            for axis, (d, n) in enumerate(zip(divisions, shape, strict=True))
        ]
        self.mask = np.zeros(tuple(n - 1 for n in shape), dtype=bool)
        self.edges = np.full((*shape, 3), None, dtype=object)
        self.projections = np.full((*shape, 3), None, dtype=object)
        self.patches = np.full((*shape, 3), None, dtype=object)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The number of blocks along each direction."""
        return self.mask.shape

    def blocks(self) -> tuple[HexBlock, ...]:
        """The block of every lattice cell left in, ``(i, j, k)`` order with ``k`` fastest."""
        return tuple(
            self._block(index)
            for index in itertools.product(*(range(n) for n in self.shape))
            if not self.mask[index]
        )

    def _block(self, index) -> HexBlock:
        base = np.asarray(index)
        nodes = base + HEXAHEDRON_CORNERS  # each block corner's lattice node
        coordinates = self.corners[tuple(nodes.T)]
        edges = {}
        for a, b in _BLOCK_EDGES:
            axis = int(np.flatnonzero(HEXAHEDRON_CORNERS[b] - HEXAHEDRON_CORNERS[a])[0])
            given = self.edges[(*nodes[a], axis)]
            if given is None:
                given = self.frame.edge(coordinates[a], coordinates[b])
            elif isinstance(given, Straight):
                given = None
            if given is not None:
                edges[(a, b)] = given
        projections, patches = {}, {}
        for face, name in enumerate(FACES):
            axis, side = divmod(face, 2)
            node = (*(base + side * np.eye(3, dtype=int)[axis]), axis)
            given = self.projections[node]
            if given is None:
                given = self.frame.projection(axis, _face_corners(coordinates, axis, side))
            if given is not None and given is not False:
                projections[name] = given
            if self.patches[node] is not None:
                patches[name] = self.patches[node]
        return HexBlock(
            self.frame.points(coordinates),
            tuple(self.divisions[axis][index[axis]] for axis in range(3)),
            edges=edges,
            projections=projections,
            patches=patches,
        )


#: A block's twelve edges, each as the corners it runs from and to, along increasing coordinate.
_BLOCK_EDGES = [
    (a, b) if HEXAHEDRON_CORNERS[a].sum() < HEXAHEDRON_CORNERS[b].sum() else (b, a)
    for a, b in itertools.combinations(range(8), 2)
    if np.count_nonzero(HEXAHEDRON_CORNERS[a] != HEXAHEDRON_CORNERS[b]) == 1
]


def _face_corners(coordinates, axis, side) -> np.ndarray:
    """A block face's corner coordinates ``(2, 2, 3)``, indexed by its two directions in order."""
    on = HEXAHEDRON_CORNERS[:, axis] == side
    rows = HEXAHEDRON_CORNERS[on][:, [a for a in range(3) if a != axis]]
    out = np.empty((2, 2, 3))
    out[rows[:, 0], rows[:, 1]] = coordinates[on]
    return out


def _per_interval(division, n: int, axis: int) -> list:
    """One division per interval: a list or tuple as given, else the one repeated."""
    if isinstance(division, list | tuple):
        if len(division) != n:
            raise ValueError(
                f"{len(division)} divisions given along axis {axis}, which has {n} interval(s)"
            )
        return list(division)
    return [division] * n
