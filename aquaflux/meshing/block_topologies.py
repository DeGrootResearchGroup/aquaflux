"""Block layouts for the round parts of a water-treatment reactor: pipes, lamp sleeves and their ends.

Each layout is a set of :class:`~aquaflux.meshing.HexBlock` that join conformally, about the ``z``
axis, with its boundary faces in named patches, for :meth:`BlockMesh.assemble`. Every curved edge
is a circular arc and every curved face is projected onto its cylinder or sphere, so the boundary
nodes lie on the true surface. Move a layout elsewhere with :meth:`HexBlock.placed`.

* :class:`PipeOGrid` -- a pipe: a square core and the blocks from its sides to the wall, in one or
  more shells, so no cell is squeezed onto the axis.
* :class:`Annulus` -- the space between coaxial cylinders (a lamp sleeve inside a vessel), in
  blocks around the axis and, optionally, several shells.
* :class:`CubedSphereCap` -- a hemispherical shell (the space about a sleeve's rounded end) in five
  blocks: the top face of a cube and the upper halves of its four sides, projected onto the sphere.
  Its rim is cut at the cube's corners, at 45 degrees to the axes, so an :class:`Annulus` meeting it
  starts at that angle (``azimuth = pi / 4``).

The pipe and the annulus are configurations of a :class:`~aquaflux.meshing.BlockLattice`. Lengths
along the axis are given as ``heights``: the ends and any cuts between them, each segment with its
own axial division. Divisions are external: a count, a :class:`~aquaflux.meshing.Graded` progression
or node fractions, one for every shell or segment or a list of one each.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Sequence

import numpy as np

from aquaflux.io.cad import Placement

from .blocks import Arc, HexBlock
from .elements import HEXAHEDRON_CORNERS
from .lattices import BlockLattice, CartesianFrame, CylindricalFrame, Straight
from .surfaces import SphereSurface

__all__ = ["Annulus", "CubedSphereCap", "PipeOGrid"]


@dataclasses.dataclass(frozen=True)
class PipeOGrid:
    """A pipe of ``radius`` along ``z`` as an O-grid: a square core and shells out to the wall.

    The core is a square of half-width ``core_size * radius``, cut into ``side_blocks`` blocks along
    each side; around it, each shell is one block per core side block, from the core (or the shell
    within) out to the next radius. Its sides can bow outward toward the circle through its corners,
    which evens out the cells at the core's corners.

    Attributes
    ----------
    radius : float
    heights : sequence of float
        The pipe's ends along the axis and any cuts between, increasing.
    tangential : int
        Cells along each core side block, and so along each wall block.
    radial : division, or a list of one per shell
        The divisions from the core outward, through each shell.
    axial : division, or a list of one per segment
        The divisions along the axis.
    shells : sequence of float
        Radii between the core and the wall, increasing, each beyond the core's corners. Unset, none:
        one shell from the core to the wall.
    side_blocks : int
        Core blocks along each side. Unset, 1.
    core_size : float
        The core's half-width over the radius, in ``(0, 1/sqrt(2))``. Unset, ``0.5``.
    core_bow : float
        How far the middle of each core side moves out toward the circle through the core's corners,
        as a fraction of the way, in ``[0, 1)``: 0 keeps the sides straight. Unset, 0.
    azimuth : float
        The angle of a core corner from ``x``, in radians. Unset, ``-pi / 4``.

    Patches: ``wall``, ``inlet`` (at the first height) and ``outlet``.
    """

    radius: float
    heights: Sequence[float]
    tangential: int
    radial: object
    axial: object
    shells: Sequence[float] = ()
    side_blocks: int = 1
    core_size: float = 0.5
    core_bow: float = 0.0
    azimuth: float = -0.25 * math.pi

    def __post_init__(self) -> None:
        if not 0.0 < self.core_size < math.sqrt(0.5):
            raise ValueError(f"PipeOGrid.core_size must be in (0, 1/sqrt(2)), got {self.core_size}")
        if not 0.0 <= self.core_bow < 1.0:
            raise ValueError(f"PipeOGrid.core_bow must be in [0, 1), got {self.core_bow}")
        corner = math.sqrt(2.0) * self.core_size * self.radius
        radii = [corner, *self.shells, self.radius]
        if np.any(np.diff(radii) <= 0.0):
            raise ValueError(
                f"PipeOGrid.shells must increase from beyond the core's corners ({corner:g}) to "
                f"inside the wall ({self.radius:g}), got {list(self.shells)}"
            )
        if int(self.side_blocks) != self.side_blocks or self.side_blocks < 1:
            raise ValueError(
                f"PipeOGrid.side_blocks must be a positive integer, got {self.side_blocks}"
            )

    def blocks(self) -> tuple[HexBlock, ...]:
        """The core's blocks, then the shells' (inner shell first), each lattice in ``(i, j, k)`` order."""
        m, a = self.side_blocks, self.core_size * self.radius
        half = np.linspace(-a, a, m + 1)
        core = BlockLattice(
            CartesianFrame(),
            (half, half, self.heights),
            (self.tangential, self.tangential, self.axial),
        )
        _name_ends(core, "inlet", "outlet")
        ring = _core_ring(m)
        sides = _CoreSides(half, self.core_bow, m)
        positions = sides.positions
        for (i, j), position in zip(ring, positions, strict=True):
            core.corners[i, j, :, :2] = position
        tube = BlockLattice(
            CylindricalFrame(),
            (
                [math.sqrt(2.0) * a, *self.shells, self.radius],
                -0.25 * math.pi + np.linspace(0.0, 2.0 * math.pi, 4 * m + 1),
                self.heights,
            ),
            (self.radial, self.tangential, self.axial),
        )
        _name_ends(tube, "inlet", "outlet")
        tube.patches[-1, :, :, 0] = "wall"
        tube.projections[0, :, :, 0] = False  # the core's sides, not a cylinder
        tube.edges[0, :, :, 0] = Straight()  # from the core's side out to the first shell
        theta = np.unwrap(np.arctan2(positions[:, 1], positions[:, 0]))
        tube.corners[0, :, :, 0] = np.hypot(positions[:, 0], positions[:, 1])[:, None]
        tube.corners[0, :, :, 1] = theta[:, None]
        for t, ((i, j), (p, q)) in enumerate(itertools.pairwise(ring)):
            for k, z in enumerate(self.heights):
                tube.edges[0, t, k, 1] = sides.edge(t, z, forward=True)
                start, axis = min((i, j), (p, q)), 0 if i != p else 1
                core.edges[(*start, k, axis)] = sides.edge(t, z, forward=(i, j) == start)
        blocks = core.blocks() + tube.blocks()
        turn = self.azimuth + 0.25 * math.pi
        if turn == 0.0:
            return blocks
        c, s = math.cos(turn), math.sin(turn)
        about_z = Placement(matrix=[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return tuple(block.placed(about_z) for block in blocks)


@dataclasses.dataclass(frozen=True)
class _CoreSides:
    """The O-grid core's boundary: its nodes around it, and the edges between them, straight or bowed.

    Side ``s`` faces ``s`` quarter turns from ``+x``; its nodes sit at equal steps along it, or at equal
    angles along the circular arc through its two corners whose middle is ``bow`` of the way from the
    side's middle out to the circle through the corners.
    """

    half: np.ndarray
    bow: float
    side_blocks: int

    @property
    def positions(self) -> np.ndarray:
        """The ``(x, y)`` of every node around the core, counter-clockwise from ``(a, -a)``, closed."""
        m = self.side_blocks
        out = [self._on_side(s, n) for s in range(4) for n in range(m)]
        return np.array([*out, out[0]])

    def edge(self, t: int, z: float, forward: bool):
        """The ``t``-th boundary edge at height ``z``: ``Straight`` or an arc, run forward or back."""
        if self.bow == 0.0:
            return Straight()
        side, n = divmod(t, self.side_blocks)
        centre = self._centre(side)
        start = np.array([*self._on_side(side, n), z])
        end = np.array([*self._on_side(side, n + 1), z])
        ends = (start, end) if forward else (end, start)
        return Arc(np.array([*centre, z]), *ends)

    def _on_side(self, side: int, n: int) -> np.ndarray:
        """The ``n``-th node along side ``side`` (``n = side_blocks`` is the next side's first)."""
        a = self.half[-1]
        normal, tangent = _rotated(np.array([1.0, 0.0]), side), _rotated(np.array([0.0, 1.0]), side)
        along = self.half[n]  # from -a to a
        if self.bow == 0.0:
            return a * normal + along * tangent
        centre = self._centre(side)
        offset = float(centre @ normal)
        spread = math.atan2(a, a - offset)
        angle = spread * along / a
        radius = math.hypot(a - offset, a)
        return centre + radius * (math.cos(angle) * normal + math.sin(angle) * tangent)

    def _centre(self, side: int) -> np.ndarray:
        """The centre of the arc through side ``side``'s corners and its bowed middle."""
        a = self.half[-1]
        middle = a + self.bow * (math.sqrt(2.0) - 1.0) * a
        offset = (2.0 * a * a - middle * middle) / (2.0 * (a - middle))
        return offset * _rotated(np.array([1.0, 0.0]), side)


@dataclasses.dataclass(frozen=True)
class Annulus:
    """The space between coaxial cylinders along ``z``, in blocks around the axis and shells.

    Attributes
    ----------
    radii : sequence of float
        The inner cylinder's radius, any between, and the outer's, increasing.
    heights : sequence of float
        The annulus's ends along the axis and any cuts between, increasing.
    tangential : int
        Cells along each block around the axis.
    radial : division, or a list of one per shell
        The divisions from the inner cylinder outward.
    axial : division, or a list of one per segment
        The divisions along the axis.
    around : int
        Blocks around the axis, at least 3. Unset, 4.
    azimuth : float
        The angle of the first block's start, in radians. Unset, 0; ``pi / 4`` with four blocks meets
        a :class:`CubedSphereCap`.

    Patches: ``inner``, ``outer``, ``start`` (at the first height) and ``end``.
    """

    radii: Sequence[float]
    heights: Sequence[float]
    tangential: int
    radial: object
    axial: object
    around: int = 4
    azimuth: float = 0.0

    def __post_init__(self) -> None:
        if len(self.radii) < 2 or self.radii[0] <= 0.0 or np.any(np.diff(self.radii) <= 0.0):
            raise ValueError(
                f"Annulus.radii must be positive and increase, two or more, got {self.radii}"
            )
        if int(self.around) != self.around or self.around < 3:
            raise ValueError(f"Annulus.around must be an integer of at least 3, got {self.around}")

    def blocks(self) -> tuple[HexBlock, ...]:
        """The blocks, shell by shell, each counter-clockwise from ``azimuth``."""
        lattice = BlockLattice(
            CylindricalFrame(),
            (
                self.radii,
                self.azimuth + np.linspace(0.0, 2.0 * math.pi, self.around + 1),
                self.heights,
            ),
            (self.radial, self.tangential, self.axial),
        )
        lattice.patches[0, :, :, 0] = "inner"
        lattice.patches[-1, :, :, 0] = "outer"
        _name_ends(lattice, "start", "end")
        return lattice.blocks()


@dataclasses.dataclass(frozen=True)
class CubedSphereCap:
    """The hemispherical shell between two spheres about a point on ``z``, on the ``+z`` side.

    Five blocks: the top face of the cube ``[-1, 1]^2 x [-1, 1]`` about the centre and the upper
    halves of its four sides, projected onto the spheres. The rim, in the plane ``z = base``, is
    cut at 45, 135, 225 and 315 degrees.

    Attributes
    ----------
    inner_radius, outer_radius : float
    tangential : int
        Cells along each edge of the cube's top face, and so along each quarter of the rim.
    polar : int
        Cells from the rim to the top face's edge.
    radial : int or np.ndarray
        The divisions from the inner sphere to the outer: a count, or node fractions from 0 to 1.
    base : float
        The height of the spheres' centre (and the rim) on the axis. Unset, 0.

    Patches: ``inner``, ``outer`` and ``rim``.
    """

    inner_radius: float
    outer_radius: float
    tangential: int
    polar: int
    radial: int | np.ndarray
    base: float = 0.0

    def __post_init__(self) -> None:
        _check_radii(self.inner_radius, self.outer_radius, "CubedSphereCap")

    def blocks(self) -> tuple[HexBlock, ...]:
        """The top block, then the four side blocks, counter-clockwise from ``+x``."""

        def top(v, w):
            return np.array([-1.0 + 2.0 * v, -1.0 + 2.0 * w, 1.0])

        sides = [
            (lambda v, w, q=quarter: np.array([*_rotated([1.0, -1.0 + 2.0 * v], q), w]))
            for quarter in range(4)
        ]
        layouts = [(top, (self.tangential, self.tangential), {})]
        layouts += [(side, (self.tangential, self.polar), {"w-": "rim"}) for side in sides]
        return tuple(self._block(cube, counts, patches) for cube, counts, patches in layouts)

    def _block(self, cube, counts, patches) -> HexBlock:
        """The block between the spheres over the patch ``cube(v, w)`` of the cube's surface."""
        centre = np.array([0.0, 0.0, self.base])
        radii = (self.inner_radius, self.outer_radius)
        corners = []
        for u, v, w in HEXAHEDRON_CORNERS:
            direction = cube(v, w)
            corners.append(centre + radii[u] * direction / np.linalg.norm(direction))
        corners = np.array(corners)
        # Every edge on a sphere is the image of a straight cube edge: a great-circle arc.
        on_spheres = [(0, 3), (1, 2), (4, 7), (5, 6), (0, 4), (1, 5), (2, 6), (3, 7)]
        return HexBlock(
            corners,
            (self.radial, *counts),
            edges={edge: Arc(centre, corners[edge[0]], corners[edge[1]]) for edge in on_spheres},
            projections={
                "u-": SphereSurface(self.inner_radius, centre),
                "u+": SphereSurface(self.outer_radius, centre),
            },
            patches={"u-": "inner", "u+": "outer", **patches},
        )


def _rotated(point, quarter: int) -> np.ndarray:
    """``point`` in the ``xy`` plane turned by ``quarter`` quarter turns, exactly."""
    x, y = point
    for _ in range(quarter % 4):
        x, y = -y, x
    return np.array([x, y])


def _core_ring(m: int) -> list[tuple[int, int]]:
    """The core lattice's boundary nodes ``(i, j)``, counter-clockwise from ``(m, 0)``, closed."""
    ring = [(m, j) for j in range(m)] + [(m - i, m) for i in range(m)]
    ring += [(0, m - j) for j in range(m)] + [(i, 0) for i in range(m)]
    return [*ring, ring[0]]


def _name_ends(lattice: BlockLattice, first: str, last: str) -> None:
    """Name a lattice's faces at its first and last height."""
    lattice.patches[:, :, 0, 2] = first
    lattice.patches[:, :, -1, 2] = last


def _check_radii(inner: float, outer: float, owner: str) -> None:
    if not 0.0 < inner < outer:
        raise ValueError(f"{owner} needs 0 < inner_radius < outer_radius, got {inner} and {outer}")
