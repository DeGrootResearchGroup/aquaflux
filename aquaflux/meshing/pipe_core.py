"""A structured O-grid filling a pipe's core, inside rings of points round it.

A pipe meshed as a structured band (and its layers, when it has them) leaves a core: a ring of
points round it at each row of the band, one ring after another along the pipe. Rather than leave
that core to tetrahedra, which in a narrow pipe come out poor, it is filled here with hexahedra:
each ring's cross-section is an O-grid -- a square core of ``k x k`` cells for a ring of ``4 k``
points, and a shell of cells from the core's sides out to the ring, one radial line per ring point
-- and each cross-section is joined to the next. The O-grid's outer faces are the rings' own
quadrilaterals, point for point, so it joins what is round it without a seam.

The core is a square of half-width ``core_size`` times the ring's mean radius, a corner on the
line to each fourth ring point. Its sides can bow outward toward the circle through its corners (by
``core_bow`` of the way at their middle, the nodes at equal angles), which evens out the cells at
its corners; the shell's radial lines then start at equal angles too, as the ring's points do.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .block_topologies import _core_ring, _CoreSides
from .blocks import transfinite
from .divisions import Graded, fractions_of

__all__ = ["PipeCore", "PipeCoreFill"]


@dataclasses.dataclass(frozen=True)
class PipeCoreFill:
    """The hexahedra filling a pipe's core, and its two end cross-sections.

    Attributes
    ----------
    points : np.ndarray, shape ``(n_points, 3)``
        The points given, then the O-grid's own.
    hexahedra : np.ndarray of int, shape ``(n_hexahedra, 8)``
        Each hexahedron's quadrilateral in one cross-section, then the same one in the next.
    first, last : np.ndarray of int, shape ``(n_quads, 4)``
        The quadrilaterals of the first and the last cross-section, the core's then the shell's.
    """

    points: np.ndarray
    hexahedra: np.ndarray
    first: np.ndarray
    last: np.ndarray


@dataclasses.dataclass(frozen=True)
class PipeCore:
    """Fills the core inside rings of points with a structured O-grid.

    Attributes
    ----------
    core_size : float
        The square core's half-width over the ring's mean radius, in ``(0, 1/sqrt(2))``.
    core_bow : float
        How far the middle of each core side moves out toward the circle through the core's
        corners, as a fraction of the way, in ``[0, 1)``: 0 keeps the sides straight.
    shell : int, Graded or None
        The cells from the ring in to the core's sides: a count, or a
        :class:`~aquaflux.meshing.Graded` progression (from the ring, ``start``, to the core).
        Unset, cells about as long as the ring's spacing.
    junction_rows : int
        At an end where a pipe opens into the rest of the region, how many rows of its wall's
        grid the O-grid stops short of the opening, leaving them to the tetrahedral fill.
    """

    core_size: float = 0.5
    core_bow: float = 0.0
    shell: int | Graded | None = None
    junction_rows: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.core_size < np.sqrt(0.5):
            raise ValueError(
                f"PipeCore.core_size must lie in (0, 1/sqrt(2)), got {self.core_size!r}"
            )
        if not 0.0 <= self.core_bow < 1.0:
            raise ValueError(f"PipeCore.core_bow must lie in [0, 1), got {self.core_bow!r}")
        if (
            isinstance(self.junction_rows, bool)
            or not isinstance(self.junction_rows, int)
            or (self.junction_rows < 1)
        ):
            raise ValueError(
                f"PipeCore.junction_rows must be a whole number >= 1, got {self.junction_rows!r}"
            )

    def fill(self, points: np.ndarray, rings: np.ndarray) -> PipeCoreFill:
        """The O-grid inside ``rings``.

        Parameters
        ----------
        points : np.ndarray, shape ``(n_points, 3)``
        rings : np.ndarray of int, shape ``(n_rows, 4 k)``
            The points round the core at each cross-section, each ring in the same direction and
            point ``i`` of every ring joined to point ``i`` of the next. At least two rows.

        Returns
        -------
        PipeCoreFill

        Raises
        ------
        ValueError
            If the rings are fewer than two, or their points not a positive multiple of four.
        """
        points = np.asarray(points, dtype=np.float64)
        rings = np.asarray(rings, dtype=np.int64)
        n_rows, n = rings.shape
        if n_rows < 2 or n < 4 or n % 4:
            raise ValueError(
                f"a pipe's core needs at least two rings of a multiple of four points, got "
                f"{n_rows} of {n}"
            )
        k = n // 4
        cores = [self._core(points[ring], k) for ring in rings]
        steps = fractions_of(self._shell(points, rings, cores))  # 0 at the ring
        sections, new = [], [points]
        base = points.shape[0]
        for ring, (lattice, boundary) in zip(rings, cores, strict=True):
            section, added = _section(points[ring], ring, lattice, boundary, steps, k, base)
            sections.append(section)
            new.append(added)
            base += added.shape[0]
        quads = np.stack([_section_quads(section) for section in sections])
        hexahedra = np.concatenate([quads[:-1], quads[1:]], axis=2).reshape(-1, 8)
        return PipeCoreFill(np.vstack(new), hexahedra, quads[0], quads[-1])

    def _core(self, ring_points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """A cross-section's square core inside its ring: the lattice ``(k + 1, k + 1, 3)``, and its
        boundary ``(4 k, 3)`` counter-clockwise from the corner toward the ring's first point."""
        centre = ring_points.mean(axis=0)
        offset = ring_points - centre
        normal = np.cross(offset, np.roll(offset, -1, axis=0)).sum(axis=0)
        normal /= np.linalg.norm(normal)
        radius = float(np.linalg.norm(offset - (offset @ normal)[:, None] * normal, axis=1).mean())
        # The frame in which the core's first corner, at -45 degrees, lies toward the first point.
        first = offset[0] - (offset[0] @ normal) * normal
        first /= np.linalg.norm(first)
        along = np.cross(normal, first)
        x_axis = (first + along) / np.sqrt(2.0)
        y_axis = np.cross(normal, x_axis)
        a = self.core_size * radius
        half = np.linspace(-a, a, k + 1)
        around = _CoreSides(half, self.core_bow, k).positions[:-1]  # (4k, 2), from (a, -a)
        boundary = centre + around[:, :1] * x_axis + around[:, 1:] * y_axis
        lattice = np.zeros((k + 1, k + 1, 3))
        for (i, j), point in zip(_core_ring(k)[:-1], boundary, strict=True):
            lattice[i, j] = point
        fractions = (half - half[0]) / (2.0 * a)
        return transfinite(lattice, [fractions, fractions]), boundary

    def _shell(self, points, rings, cores):
        """The shell's division, the same for every cross-section: :attr:`shell`, else cells about
        as long as the rings' spacing (the layers' own sizing continued would start the shell with
        a thin cell, its caps at a junction low and their apexes' cells small)."""
        if self.shell is not None:
            return self.shell
        length = float(np.mean([
            np.linalg.norm(points[ring] - boundary, axis=1).mean()
            for ring, (_, boundary) in zip(rings, cores, strict=True)
        ]))  # fmt: skip
        ring = points[rings[0]]
        spacing = float(np.mean(np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)))
        return max(1, round(length / spacing))


def _section(ring_points, ring, lattice, boundary, steps, k, base):
    """One cross-section as point indices -- its core lattice, and its shell from the ring (row 0)
    in to the core's boundary -- and its new points: the core's, then the shell's between."""
    n_core = (k + 1) ** 2
    core_index = base + np.arange(n_core).reshape(k + 1, k + 1)
    inward = steps[1:-1]
    shell_points = ring_points[None] + inward[:, None, None] * (boundary - ring_points)[None]
    boundary_index = np.array([core_index[i, j] for i, j in _core_ring(k)[:-1]])
    middle_index = base + n_core + np.arange(inward.size * 4 * k).reshape(inward.size, 4 * k)
    shell = np.vstack([ring[None], middle_index, boundary_index[None]])
    return (core_index, shell), np.vstack([lattice.reshape(-1, 3), shell_points.reshape(-1, 3)])


def _section_quads(section) -> np.ndarray:
    """A cross-section's quadrilaterals: the core's, then the shell's, each wound the same way."""
    core, shell = section
    lower, upper = core[:-1], core[1:]
    core_quads = np.stack([lower[:, :-1], upper[:, :-1], upper[:, 1:], lower[:, 1:]], axis=2)
    closed = np.concatenate([shell, shell[:, :1]], axis=1)
    outer, inner = closed[:-1], closed[1:]
    shell_quads = np.stack([outer[:, :-1], outer[:, 1:], inner[:, 1:], inner[:, :-1]], axis=2)
    return np.concatenate([core_quads.reshape(-1, 4), shell_quads.reshape(-1, 4)])
