"""Structured-grid generators for simple rectangular domains.

These produce ordinary :class:`~aquaflux.mesh.Mesh` objects on axis-aligned boxes, so they
exercise the same geometry/validation path as any other mesh. They are the convenient way to build
a mesh for a rectangular tank or channel without a mesh file.

Both generators assemble connectivity with **vectorized numpy** (no per-face Python loop) and
hand it to :meth:`Mesh.from_csr`, so assembly cost stays low as the grid grows. The two share the
face-family assembly (:class:`_FaceFamilyBuilder`), so the assembly convention lives in one place.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def graded_nodes(n: int, length: float, growth: float, *, both_sides: bool = True) -> np.ndarray:
    """Node coordinates on ``[0, length]`` with geometric cell growth (finest at the wall(s)).

    Returns ``n + 1`` monotonically increasing positions whose cell sizes follow a geometric
    progression: adjacent cells differ by the factor ``growth``, with the *smallest* cells at the
    boundary and coarsening inward. This is the wall-normal grading a wall-resolved boundary layer
    needs — a small ``y`` at the wall (so ``y+ < 1`` for the near-wall cell) without paying a uniform
    fine spacing across the whole channel.

    Parameters
    ----------
    n : int
        Number of cells along the axis.
    length : float
        Axis length; the returned positions span ``[0, length]`` exactly.
    growth : float
        Cell-to-cell size ratio (``> 0``). ``1`` is uniform; ``> 1`` clusters cells toward the
        wall(s). The near-wall cell size is ``length`` times ``growth`` raised to the negative
        half-span, normalised — pick ``growth`` to hit a target first-cell height.
    both_sides : bool
        When ``True`` (a channel between two walls) the mesh is symmetric, fine at *both* ends and
        coarsest at the centre. When ``False`` (a single wall) it is fine at ``0`` and coarsens
        monotonically to ``length``.

    Returns
    -------
    np.ndarray
        Node coordinates, shape ``(n + 1,)``, with ``[0] == 0`` and ``[-1] == length``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if growth <= 0.0:
        raise ValueError(f"growth must be > 0, got {growth}")
    index = np.arange(n)
    # Cells at "wall distance" `exponent` (in cells) get size growth**exponent, so the smallest sit
    # at the wall(s); the double-sided distance min(i, n-1-i) is symmetric for even and odd n alike.
    exponent = np.minimum(index, n - 1 - index) if both_sides else index
    sizes = np.asarray(growth, dtype=float) ** exponent
    positions = np.concatenate([[0.0], np.cumsum(sizes)])
    return length * positions / positions[-1]


class _FaceFamilyBuilder:
    """Accumulate the face families of a structured grid, then emit a :class:`Mesh`.

    A structured grid's faces split into *families* normal to each axis. Every face in a family
    has the same node count and the same owner/neighbour pattern, so a whole family is built with
    one vectorized :meth:`add_family` call. The builder concatenates the families and hands the CSR
    arrays, owner/neighbour, and named-boundary patches to :meth:`Mesh.from_csr` — the single place
    the assembly convention lives, shared by the 2D and 3D generators.
    """

    def __init__(self) -> None:
        self._nodes: list[np.ndarray] = []  # (n_family_faces, k) node-index arrays; k constant
        self._owner: list[np.ndarray] = []
        self._neighbour: list[np.ndarray] = []
        self.sides: dict[str, np.ndarray] = {}
        self._n = 0  # faces accumulated so far (for global side-face indices)

    def add_family(
        self,
        face_nodes: list[np.ndarray],
        lo_cell: np.ndarray,
        hi_cell: np.ndarray,
        lo_valid: np.ndarray,
        hi_valid: np.ndarray,
        lo_name: str,
        hi_name: str,
    ) -> None:
        """Append one face family (a constant-index plane/line of faces) and record its boundaries.

        Parameters
        ----------
        face_nodes : list of np.ndarray
            The face's node indices, one array per node position, each shape ``(n_family_faces,)``,
            winding the face perimeter (length 2 for a 2D edge, 4 for a 3D quad).
        lo_cell, hi_cell : np.ndarray
            Cell index on the low / high side of the family's axis, shape ``(n_family_faces,)``.
        lo_valid, hi_valid : np.ndarray of bool
            Whether the low / high cell exists (``False`` on the corresponding domain boundary).
            The owner is whichever side exists (both, for an interior face); the neighbour is the
            other, or ``-1`` on a boundary face. The ``clip`` on a masked-off cell index is safe:
            the clipped value is never used (it is overwritten by ``np.where``).
        lo_name, hi_name : str
            Patch names for the low / high boundary planes of this family.
        """
        self._owner.append(np.where(lo_valid, lo_cell, hi_cell))
        self._neighbour.append(np.where(lo_valid & hi_valid, hi_cell, -1))
        self._nodes.append(np.stack(face_nodes, axis=1))
        gidx = self._n + np.arange(lo_valid.size)
        self.sides[lo_name] = gidx[~lo_valid]  # low-side boundary faces
        self.sides[hi_name] = gidx[~hi_valid]  # high-side boundary faces
        self._n += lo_valid.size

    def build(self, coords: np.ndarray, n_cells: int, named_boundaries: bool) -> Mesh:
        """Assemble the accumulated families into a validated :class:`Mesh` via ``from_csr``."""
        all_nodes = np.concatenate(self._nodes, axis=0)  # (n_faces, k)
        n_faces, k = all_nodes.shape
        offsets = np.arange(n_faces + 1) * k  # every structured face has the same node count k
        return Mesh.from_csr(
            coords,
            offsets,
            all_nodes.ravel(),
            np.concatenate(self._owner),
            np.concatenate(self._neighbour),
            n_cells=n_cells,
            face_patches=self.sides if named_boundaries else None,
        )


def structured_grid_2d(
    nx: int,
    ny: int,
    lx: float = 1.0,
    ly: float = 1.0,
    named_boundaries: bool = False,
    *,
    x_nodes: np.ndarray | None = None,
    y_nodes: np.ndarray | None = None,
) -> Mesh:
    """A structured quad grid on ``[0, lx] x [0, ly]`` with ``nx * ny`` cells.

    The connectivity is assembled with **vectorized numpy** (no per-face Python loop) and handed
    to :meth:`Mesh.from_csr`, like :func:`structured_grid_3d`. Every face is a 2-node edge, so the
    CSR offsets are a plain stride of 2.

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y.
    lx, ly : float
        Domain size. Ignored on an axis whose node coordinates are given explicitly.
    named_boundaries : bool
        When ``True``, tag the four boundary sides as named face patches ``"left"``
        (x = 0), ``"right"`` (x = lx), ``"bottom"`` (y = 0), ``"top"`` (y = ly), so a
        different boundary condition can be attached to each. The sides are known exactly
        during construction (no geometric detection). Default ``False`` leaves the plain
        ``"interior"`` / ``"boundary"`` split.
    x_nodes, y_nodes : np.ndarray, optional
        Explicit, monotonically increasing node coordinates along each axis, shape ``(nx + 1,)`` /
        ``(ny + 1,)``. Only the coordinates change — the connectivity is identical — so this is how
        a **graded** mesh is built (e.g. :func:`graded_nodes` for a wall-resolved boundary layer).
        Default (``None``) is uniform spacing ``lx / nx`` / ``ly / ny``.

    Returns
    -------
    Mesh
    """

    def _axis(nodes: np.ndarray | None, n: int, length: float) -> np.ndarray:
        if nodes is None:
            return np.linspace(0.0, length, n + 1)
        nodes = np.asarray(nodes, dtype=float)
        if nodes.shape != (n + 1,):
            raise ValueError(f"node coordinates must have shape ({n + 1},), got {nodes.shape}")
        if np.any(np.diff(nodes) <= 0.0):
            raise ValueError("node coordinates must be strictly increasing")
        return nodes

    x = _axis(x_nodes, nx, lx)
    y = _axis(y_nodes, ny, ly)

    def nid(i, j):  # node index; nodes span [0, n*] inclusive on each axis
        return j * (nx + 1) + i

    def cid(i, j):  # cell index; cells span [0, n*) on each axis
        return j * nx + i

    # Node coordinates, raveled in the same C-order as ``nid`` (j slowest, i fastest).
    jj, ii = np.meshgrid(np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    coords = np.stack([x[ii].ravel(), y[jj].ravel()], axis=1)

    builder = _FaceFamilyBuilder()

    # X-normal faces (vertical edges): index i in [0, nx], cells (i-1, j) | (i, j).
    fi, fj = (a.ravel() for a in np.meshgrid(np.arange(nx + 1), np.arange(ny), indexing="ij"))
    builder.add_family(
        [nid(fi, fj), nid(fi, fj + 1)],
        cid(np.clip(fi - 1, 0, nx - 1), fj),
        cid(np.clip(fi, 0, nx - 1), fj),
        fi > 0,
        fi < nx,
        "left",
        "right",
    )

    # Y-normal faces (horizontal edges): index j in [0, ny], cells (i, j-1) | (i, j).
    fi, fj = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny + 1), indexing="ij"))
    builder.add_family(
        [nid(fi, fj), nid(fi + 1, fj)],
        cid(fi, np.clip(fj - 1, 0, ny - 1)),
        cid(fi, np.clip(fj, 0, ny - 1)),
        fj > 0,
        fj < ny,
        "bottom",
        "top",
    )

    return builder.build(coords, nx * ny, named_boundaries)


def structured_grid_3d(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
    named_boundaries: bool = False,
) -> Mesh:
    """A structured hexahedral grid on ``[0, lx] x [0, ly] x [0, lz]`` with ``nx*ny*nz`` cells.

    The 3D analogue of :func:`structured_grid_2d`. Every cell is a hexahedron with six quad
    faces; the connectivity is assembled with **vectorized numpy** (no per-face Python loop) and
    handed to :meth:`Mesh.from_csr`, so assembly cost stays low as the grid grows. Because every
    face has four nodes, the compressed-sparse-row (CSR) offsets are a plain stride.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along x, y, z.
    lx, ly, lz : float
        Domain extent along each axis.
    named_boundaries : bool
        When ``True``, tag the six sides as named face patches ``"left"`` (x=0), ``"right"``
        (x=lx), ``"bottom"`` (y=0), ``"top"`` (y=ly), ``"back"`` (z=0), ``"front"`` (z=lz), so a
        different boundary condition can attach to each (e.g. a moving lid on ``"top"``).

    Returns
    -------
    Mesh
    """
    hx, hy, hz = lx / nx, ly / ny, lz / nz

    def nid(i, j, k):  # node index; nodes span [0, n*] inclusive on each axis
        return (k * (ny + 1) + j) * (nx + 1) + i

    def cid(i, j, k):  # cell index; cells span [0, n*) on each axis
        return (k * ny + j) * nx + i

    # Node coordinates, raveled in the same C-order as ``nid`` (k slowest, i fastest).
    kk, jj, ii = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    coords = np.stack([(ii * hx).ravel(), (jj * hy).ravel(), (kk * hz).ravel()], axis=1)

    builder = _FaceFamilyBuilder()

    # X-normal faces: index i in [0, nx], cells (i-1, j, k) | (i, j, k).
    fi, fj, fk = (
        a.ravel()
        for a in np.meshgrid(np.arange(nx + 1), np.arange(ny), np.arange(nz), indexing="ij")
    )
    builder.add_family(
        [nid(fi, fj, fk), nid(fi, fj + 1, fk), nid(fi, fj + 1, fk + 1), nid(fi, fj, fk + 1)],
        cid(np.clip(fi - 1, 0, nx - 1), fj, fk),
        cid(np.clip(fi, 0, nx - 1), fj, fk),
        fi > 0,
        fi < nx,
        "left",
        "right",
    )

    # Y-normal faces: index j in [0, ny], cells (i, j-1, k) | (i, j, k).
    fi, fj, fk = (
        a.ravel()
        for a in np.meshgrid(np.arange(nx), np.arange(ny + 1), np.arange(nz), indexing="ij")
    )
    builder.add_family(
        [nid(fi, fj, fk), nid(fi + 1, fj, fk), nid(fi + 1, fj, fk + 1), nid(fi, fj, fk + 1)],
        cid(fi, np.clip(fj - 1, 0, ny - 1), fk),
        cid(fi, np.clip(fj, 0, ny - 1), fk),
        fj > 0,
        fj < ny,
        "bottom",
        "top",
    )

    # Z-normal faces: index k in [0, nz], cells (i, j, k-1) | (i, j, k).
    fi, fj, fk = (
        a.ravel()
        for a in np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz + 1), indexing="ij")
    )
    builder.add_family(
        [nid(fi, fj, fk), nid(fi + 1, fj, fk), nid(fi + 1, fj + 1, fk), nid(fi, fj + 1, fk)],
        cid(fi, fj, np.clip(fk - 1, 0, nz - 1)),
        cid(fi, fj, np.clip(fk, 0, nz - 1)),
        fk > 0,
        fk < nz,
        "back",
        "front",
    )

    return builder.build(coords, nx * ny * nz, named_boundaries)
