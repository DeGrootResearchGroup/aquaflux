"""Hexahedral blocks filled by transfinite interpolation from their edges and faces.

A block is a curved hexahedron cut into ``nu x nv x nw`` hexahedra. Its eight corners and twelve
edges fix its shape: each edge is a straight line unless a curve is given for it, each face is the
surface the transfinite interpolation of its four edges spans unless a surface is given for it, and
the inside is the transfinite interpolation of the six faces. A face may be moved onto a true
surface -- an analytic cylinder or sphere, or a face of a computer-aided design (CAD) model -- by a
projection: any callable taking points and returning the points on the surface.

**Transfinite interpolation** (Gordon and Hall, 1973) fills a structured array of points from its
boundary as the Boolean sum of linear blends along each direction: blend between the two opposite
faces along each direction, subtract the double-counted blends between opposite edges, add back the
blend of the corners. It reproduces the boundary exactly, so two blocks built on one shared face
give that face's points identically, and a block whose edges and faces are flat reproduces the
tensor-product grid of its node fractions.

**Node fractions.** The nodes along each direction sit at fractions of the way from the block's
start to its end -- uniform for a cell count, or any increasing sequence from 0 to 1 (a geometric
progression from :func:`~aquaflux.mesh.graded_nodes` divided by its length, for cells graded toward
a wall). Each edge is evaluated at its direction's fractions, and the blends use them as weights.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from aquaflux.io.cad import Placement

from .divisions import fractions_of, reversed_division, scaled_division
from .elements import HEXAHEDRON_CORNERS

__all__ = ["Arc", "Curve", "HexBlock", "SampledCurve", "Surface"]

#: A block's faces in order: the start and end of each direction. A face is named by the direction
#: it closes and its side; its points are parameterized by the other two directions, in order.
FACES = ("u-", "u+", "v-", "v+", "w-", "w+")


class Curve(Protocol):
    """A curve from one corner of a block to another, at fractions of the way along it."""

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """The curve's points.

        Parameters
        ----------
        t : np.ndarray, shape ``(n,)``
            Fractions of the way along, from 0 (the start) to 1 (the end).

        Returns
        -------
        np.ndarray, shape ``(n, 3)``
        """


class Surface(Protocol):
    """A surface spanning one face of a block, over the face's two directions' fractions."""

    def __call__(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """The surface's points.

        Parameters
        ----------
        s, t : np.ndarray, shape ``(n_s, n_t)``
            Fractions along the face's first and second directions.

        Returns
        -------
        np.ndarray, shape ``(n_s, n_t, 3)``
        """


@dataclasses.dataclass(frozen=True)
class Arc:
    """The shorter circular arc about ``centre`` from ``start`` to ``end``, uniform in angle.

    Attributes
    ----------
    centre, start, end : np.ndarray, shape ``(3,)``
        ``start`` and ``end`` equally far from ``centre``, and not on opposite sides of it (the
        plane of the arc would be undefined).

    Raises
    ------
    ValueError
        If ``start`` and ``end`` are not equally far from ``centre``, or the arc is a half circle
        or more, or has no length.
    """

    centre: np.ndarray
    start: np.ndarray
    end: np.ndarray

    def __post_init__(self) -> None:
        for name in ("centre", "start", "end"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        a, b = self.start - self.centre, self.end - self.centre
        ra, rb = np.linalg.norm(a), np.linalg.norm(b)
        if not np.isclose(ra, rb, rtol=1e-9, atol=0.0):
            raise ValueError(
                f"an arc's ends must be equally far from its centre, got {ra} and {rb}"
            )
        normal = np.cross(a, b)
        if np.linalg.norm(normal) <= 1e-12 * ra * rb:
            raise ValueError(
                "an arc's ends must not be at one point or on opposite sides of its centre"
            )

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """The arc's points, at fractions ``t`` of its angle -- see :class:`Curve`."""
        a, b = self.start - self.centre, self.end - self.centre
        normal = np.cross(a, b)
        angle = np.arctan2(np.linalg.norm(normal), a @ b)
        axis = normal / np.linalg.norm(normal)
        theta = np.asarray(t, dtype=np.float64)[:, None] * angle
        # Rodrigues' rotation of ``a`` about ``axis``; ``a`` is perpendicular to it.
        return self.centre + a * np.cos(theta) + np.cross(axis, a) * np.sin(theta)


@dataclasses.dataclass(frozen=True)
class SampledCurve:
    """The polyline through ``points``, at fractions of its length.

    Attributes
    ----------
    points : np.ndarray, shape ``(n, 3)``
        At least two points, the first and last the curve's ends.
    """

    points: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            raise ValueError(f"a sampled curve takes (n >= 2, 3) points, got shape {points.shape}")
        object.__setattr__(self, "points", points)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """The polyline's points, at fractions ``t`` of its length -- see :class:`Curve`."""
        along = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        )
        at = np.asarray(t, dtype=np.float64) * along[-1]
        return np.stack([np.interp(at, along, self.points[:, k]) for k in range(3)], axis=1)


@dataclasses.dataclass(frozen=True)
class HexBlock:
    """A curved hexahedron cut into hexahedra, filled by transfinite interpolation.

    The block's directions ``u``, ``v``, ``w`` run from corner 0 to corners 1, 3 and 4. Corners
    follow the hexahedron's order (as in :func:`~aquaflux.meshing.element_mesh`): ``(0,0,0),
    (1,0,0), (1,1,0), (0,1,0)``, then the same at ``w = 1``. The corners must be right-handed
    (``u x v`` along ``w``).

    Attributes
    ----------
    corners : np.ndarray, shape ``(8, 3)``
    divisions : tuple of (int, Graded or np.ndarray), one per direction
        Each direction's cells: a count ``n`` (uniform), a :class:`~aquaflux.meshing.Graded`
        progression, or the ``n + 1`` node fractions, increasing from 0 to 1.
    edges : mapping of {(int, int): Curve or np.ndarray}, optional
        Curved edges, keyed by the corners they run from and to (``(1, 2)``, or ``(2, 1)`` for the
        same edge the other way). A value is a :class:`Curve`, evaluated at the direction's node
        fractions, or its nodes themselves, shape ``(n + 1, 3)``, from the first corner to the
        second. An edge given neither here nor by a face is straight.
    faces : mapping of {str: Surface or np.ndarray}, optional
        Faces given as surfaces, keyed by name (:data:`FACES`). A value is a :class:`Surface` over
        the face's two directions, or its nodes themselves, shape ``(n_a + 1, n_b + 1, 3)``. A face
        given here also gives its four edges, which must agree with any given for them.
    projections : mapping of {str: callable}, optional
        Faces moved onto a true surface, keyed by name: after a face is interpolated, its nodes off
        its edges are replaced by the projection's image of them. Give curved edges that lie on the
        surface (an :class:`Arc` on a cylinder or a sphere) so the face's edges are on it too.
    patches : mapping of {str: str}, optional
        Names of the boundary patches the block's faces lie in, keyed by face name.

    Raises
    ------
    ValueError
        If a count is below one, fractions do not increase from 0 to 1, a key names no edge or face,
        a curve does not join its corners, or nodes given have the wrong shape.
    """

    corners: np.ndarray
    divisions: tuple
    edges: Mapping = dataclasses.field(default_factory=dict)
    faces: Mapping = dataclasses.field(default_factory=dict)
    projections: Mapping = dataclasses.field(default_factory=dict)
    patches: Mapping = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners, dtype=np.float64)
        if corners.shape != (8, 3):
            raise ValueError(f"a block takes (8, 3) corners, got shape {corners.shape}")
        object.__setattr__(self, "corners", corners)
        if len(self.divisions) != 3:
            raise ValueError(f"a block takes three divisions, got {len(self.divisions)}")
        for names, kind in (
            (self.faces, "faces"),
            (self.projections, "projections"),
            (self.patches, "patches"),
        ):
            unknown = sorted(set(names) - set(FACES))
            if unknown:
                raise ValueError(
                    f"block {kind} name unknown face(s) {unknown}; the faces are {FACES}"
                )

    @property
    def fractions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each direction's node fractions, from 0 to 1."""
        return tuple(fractions_of(division) for division in self.divisions)

    @property
    def cells(self) -> tuple[int, int, int]:
        """The number of cells along each direction."""
        return tuple(f.size - 1 for f in self.fractions)

    def scaled_counts(self, factor: float) -> HexBlock:
        """The block with ``factor`` times the cells along each direction (at least one).

        A count is rounded, a :class:`~aquaflux.meshing.Graded` progression keeps the ratio of its end
        cells, and explicit fractions take only a whole ``factor``, each cell split evenly.

        Raises
        ------
        ValueError
            If an edge or face is given as nodes (they fix the cell counts), or explicit fractions
            meet a factor that is not whole.
        """
        fixed = [k for k, v in {**self.edges, **self.faces}.items() if not callable(v)]
        if fixed:
            raise ValueError(
                f"edges or faces {fixed} are given as nodes, which fix the cell counts"
            )
        divisions = tuple(scaled_division(d, factor) for d in self.divisions)
        return dataclasses.replace(self, divisions=divisions)

    def placed(self, placement: Placement) -> HexBlock:
        """The block moved by a rigid map: its corners, curves, surfaces and projections with it.

        A reflection turns the block left-handed, so the placed block reads its ``w`` direction the
        other way (corners, faces and divisions relabelled) and stays right-handed.

        Parameters
        ----------
        placement : Placement
            The map ``x -> matrix @ x + offset``, ``matrix`` orthogonal.

        Returns
        -------
        HexBlock
        """
        block = dataclasses.replace(
            self,
            corners=_apply(placement, self.corners),
            edges={key: _placed(placement, curve) for key, curve in self.edges.items()},
            faces={name: _placed(placement, face) for name, face in self.faces.items()},
            projections={
                name: _PlacedProjection(placement, projection)
                for name, projection in self.projections.items()
            },
        )
        return block._w_reversed() if np.linalg.det(placement.matrix) < 0 else block

    def _w_reversed(self) -> HexBlock:
        """The same block with its ``w`` direction read the other way."""
        swap = {"w-": "w+", "w+": "w-"}

        def turned(name, face):
            if name in swap:
                return face
            # A face along w: its second direction is reversed.
            return _WReversedFace(face) if callable(face) else np.asarray(face)[:, ::-1]

        return HexBlock(
            corners=self.corners[_W_REVERSED],
            divisions=(*self.divisions[:2], reversed_division(self.divisions[2])),
            edges={(_W_REVERSED[a], _W_REVERSED[b]): c for (a, b), c in self.edges.items()},
            faces={swap.get(n, n): turned(n, f) for n, f in self.faces.items()},
            projections={swap.get(n, n): p for n, p in self.projections.items()},
            patches={swap.get(n, n): p for n, p in self.patches.items()},
        )

    def points(self) -> np.ndarray:
        """The block's nodes, shape ``(nu + 1, nv + 1, nw + 1, 3)``, indexed by ``(i, j, k)``.

        Raises
        ------
        ValueError
            If the given edges and faces do not agree, or a cell turns inside out (the corners are
            left-handed or the block folds over itself).
        """
        fractions = self.fractions
        nodes = np.full((*(f.size for f in fractions), 3), np.nan)
        scale = float(np.max(np.ptp(self.corners, axis=0)))
        given_edges = self._given_edges(fractions, scale)
        for direction, ends in _EDGES:
            index = _edge_index(direction, ends)
            given = given_edges.get((direction, ends))
            if given is None:
                given = self._straight(direction, ends, fractions)
            nodes[index] = given
        for face in FACES:
            self._fill_face(nodes, face, fractions)
        nodes = transfinite(nodes, fractions)
        _check_right_handed(nodes)
        return nodes

    def _fill_face(self, nodes, face, fractions) -> None:
        """Fill one face's nodes inside its edges: given, or interpolated, then projected."""
        direction, side = _FACE_PLACES[FACES.index(face)]
        index = tuple(-side if axis == direction else slice(None) for axis in range(3))
        plane = [f for axis, f in enumerate(fractions) if axis != direction]
        given = self._given_face(face, plane)  # its edges are already in place, from it
        nodes[index] = transfinite(nodes[index], plane) if given is None else given
        projection = self.projections.get(face)
        if projection is not None:
            inner = nodes[index][1:-1, 1:-1]
            moved = np.asarray(projection(inner.reshape(-1, 3)), dtype=np.float64)
            nodes[index][1:-1, 1:-1] = moved.reshape(inner.shape)

    def _given_edges(self, fractions, scale) -> dict:
        """Each edge given directly or by a given face, its nodes from start to end of direction."""
        found, source = {}, {}

        def place(key, edge_nodes, what):
            if key in found:
                _check_agrees(edge_nodes, found[key], scale, f"{source[key]} and {what}")
            found[key], source[key] = edge_nodes, what

        for (a, b), given in self.edges.items():
            direction, ends, forward = _edge_of(a, b)
            t = fractions[direction] if forward else 1.0 - fractions[direction][::-1]
            edge_nodes = _nodes_of(given, t, (fractions[direction].size, 3), f"edge {(a, b)}")
            _check_agrees(
                edge_nodes[[0, -1]], self.corners[[a, b]], scale, f"edge {(a, b)} and its corners"
            )
            place((direction, ends), edge_nodes if forward else edge_nodes[::-1], f"edge {(a, b)}")
        for name in self.faces:
            direction, side = _FACE_PLACES[FACES.index(name)]
            plane = [f for axis, f in enumerate(fractions) if axis != direction]
            face_nodes = self._given_face(name, plane)
            for edge_direction, ends in _EDGES:
                if edge_direction == direction or ends[_other(direction, edge_direction)] != side:
                    continue
                # The face's own array, indexed by its two directions, holds this edge on its rim.
                axes = [axis for axis in range(3) if axis != direction]
                rim = tuple(
                    slice(None) if axis == edge_direction else -ends[_other(axis, edge_direction)]
                    for axis in axes
                )
                place((edge_direction, ends), face_nodes[rim], f"face {name!r}'s edges")
        return found

    def _given_face(self, name, plane) -> np.ndarray | None:
        given = self.faces.get(name)
        if given is None:
            return None
        if callable(given):
            s, t = np.meshgrid(*plane, indexing="ij")
            return np.asarray(given(s, t), dtype=np.float64).reshape(*s.shape, 3)
        return _nodes_of(given, None, (plane[0].size, plane[1].size, 3), f"face {name!r}")

    def _straight(self, direction, ends, fractions) -> np.ndarray:
        """The straight edge between its corners: the interpolation of its two ends."""
        start = np.insert(np.asarray(ends), direction, 0)
        a, b = (_corner_number(start), _corner_number(start + np.eye(3, dtype=int)[direction]))
        edge = np.full((fractions[direction].size, 3), np.nan)
        edge[[0, -1]] = self.corners[[a, b]]
        return transfinite(edge, [fractions[direction]])


#: The corner order of a block read with ``w`` reversed (its own inverse).
_W_REVERSED = np.array([4, 5, 6, 7, 0, 1, 2, 3])


def _apply(placement: Placement, points) -> np.ndarray:
    """Points moved by a placement, shape kept."""
    points = np.asarray(points, dtype=np.float64)
    return points @ placement.matrix.T + placement.offset


def _placed(placement: Placement, given):
    """A curve or surface, or the nodes given for one, moved by a placement."""
    return _Placed(placement, given) if callable(given) else _apply(placement, given)


@dataclasses.dataclass(frozen=True)
class _Placed:
    """A curve or a surface moved by a placement."""

    placement: Placement
    given: object

    def __call__(self, *parameters) -> np.ndarray:
        return _apply(self.placement, self.given(*parameters))


@dataclasses.dataclass(frozen=True)
class _PlacedProjection:
    """A projection made in the block's own frame, conjugated by a placement."""

    placement: Placement
    projection: object

    def __call__(self, points: np.ndarray) -> np.ndarray:
        local = (
            np.asarray(points, dtype=np.float64) - self.placement.offset
        ) @ self.placement.matrix
        return _apply(self.placement, self.projection(local))


@dataclasses.dataclass(frozen=True)
class _WReversedFace:
    """A surface along ``w`` read with its second direction reversed."""

    given: object

    def __call__(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.given(s, 1.0 - t)


def transfinite(nodes: np.ndarray, fractions) -> np.ndarray:
    """The structured array of points filled inside from its boundary by transfinite interpolation.

    Parameters
    ----------
    nodes : np.ndarray, shape ``(n_0 + 1, ..., n_{d-1} + 1, 3)``
        Points of a ``d``-dimensional structured array, read only on its boundary.
    fractions : sequence of np.ndarray
        Each direction's node fractions, ``n_k + 1`` from 0 to 1, the blends' weights.

    Returns
    -------
    np.ndarray
        ``nodes`` with its interior replaced; its boundary is returned unchanged.
    """
    dims = len(fractions)
    blended = np.zeros(nodes.shape)
    for size in range(1, dims + 1):
        for axes in itertools.combinations(range(dims), size):
            blended += (-1.0) ** (size + 1) * _blend(nodes, fractions, axes)
    filled = nodes.copy()
    interior = (slice(1, -1),) * dims
    filled[interior] = blended[interior]
    return filled


def _blend(nodes, fractions, axes) -> np.ndarray:
    """The linear blend between the array's two ends along each of ``axes`` in turn."""
    blended = nodes
    for axis in axes:
        shape = [1] * (nodes.ndim)
        shape[axis] = -1
        weight = fractions[axis].reshape(shape)
        start = np.take(blended, [0], axis=axis)
        end = np.take(blended, [-1], axis=axis)
        blended = (1.0 - weight) * start + weight * end
    return np.broadcast_to(blended, nodes.shape)


#: The twelve edges, each its direction and its place on the other two directions (in axis order).
_EDGES = [(d, ends) for d in range(3) for ends in itertools.product((0, 1), repeat=2)]
#: Each face's direction and side, in the order of :data:`FACES`.
_FACE_PLACES = [(d, side) for d in range(3) for side in (0, 1)]


def _other(axis: int, edge_direction: int) -> int:
    """Which of an edge's two fixed ends belongs to ``axis``."""
    return [a for a in range(3) if a != edge_direction].index(axis)


def _edge_index(direction, ends) -> tuple:
    """The index into a block's nodes of one edge, from start to end of its direction."""
    fixed = iter(ends)
    return tuple(slice(None) if axis == direction else -next(fixed) for axis in range(3))


def _corner_number(place) -> int:
    return int(np.flatnonzero(np.all(HEXAHEDRON_CORNERS == np.asarray(place), axis=1))[0])


def _edge_of(a: int, b: int):
    """The edge joining corners ``a`` and ``b``: its direction, its ends, and whether ``a`` starts it."""
    if not (0 <= a < 8 and 0 <= b < 8):
        raise ValueError(f"edge {(a, b)} names a corner outside 0..7")
    differ = np.flatnonzero(HEXAHEDRON_CORNERS[a] != HEXAHEDRON_CORNERS[b])
    if differ.size != 1:
        raise ValueError(f"corners {(a, b)} are not joined by an edge of the block")
    direction = int(differ[0])
    ends = tuple(int(HEXAHEDRON_CORNERS[a][axis]) for axis in range(3) if axis != direction)
    return direction, ends, bool(HEXAHEDRON_CORNERS[a][direction] == 0)


def _nodes_of(given, t, shape, what) -> np.ndarray:
    """A curve evaluated at ``t``, or nodes given directly, checked for shape."""
    if callable(given):
        nodes = np.asarray(given(t), dtype=np.float64)
    else:
        nodes = np.asarray(given, dtype=np.float64)
    if nodes.shape != shape:
        raise ValueError(f"{what} gives nodes of shape {nodes.shape}, expected {shape}")
    return nodes


def _check_agrees(given, expected, scale, what) -> None:
    """Refuse nodes that differ from those already fixed by more than rounding."""
    known = ~np.isnan(expected)
    if np.any(np.abs(given - expected)[known] > 1e-9 * scale):
        raise ValueError(f"{what} disagree")


def _check_right_handed(nodes) -> None:
    """Refuse a block with a cell whose corner frame is not right-handed."""
    for corner in itertools.product((0, 1), repeat=3):
        at = tuple(slice(c, nodes.shape[a] - 1 + c) for a, c in enumerate(corner))
        frame = []
        for axis in range(3):
            step = [slice(c, nodes.shape[a] - 1 + c) for a, c in enumerate(corner)]
            step[axis] = slice(1 - corner[axis], nodes.shape[axis] - corner[axis])
            sign = 1.0 if corner[axis] == 0 else -1.0
            frame.append(sign * (nodes[tuple(step)] - nodes[at]))
        jacobian = np.einsum("...i,...i->...", np.cross(frame[0], frame[1]), frame[2])
        if np.any(jacobian <= 0.0):
            raise ValueError(
                f"{int(np.count_nonzero(jacobian <= 0.0))} cell corner(s) of the block are not "
                "right-handed: its corners are left-handed or it folds over itself"
            )
