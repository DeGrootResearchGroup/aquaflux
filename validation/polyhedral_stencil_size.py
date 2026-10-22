"""How much larger is a Jacobian stencil on a polyhedral dual than on hexahedra? (structural, no solve)

The coupled Jacobian a materialized preconditioner stores couples each cell to every cell within its
stencil reach, and the coloured probe that builds it needs one Jacobian-vector product per colour and
field. Both are set by the cell graph alone, so this compares two graphs:

  hex            an ``n`` x ``n`` x ``n`` lattice, six faces a cell
  dual-of-tets   the polyhedral dual of a Delaunay tetrahedralization of the same lattice jittered by
                 +-0.35 of a cell. A dual cell's faces are its point's tet-mesh edges, so the dual's face
                 graph IS the tet mesh's point graph -- what a tetrahedral fill plus the polyhedral dual
                 gives, without running a mesher (about fifteen faces a cell here)

It prints, per reach, the mean number of cells within that many steps of an interior cell (seeds at least
six cells from the boundary) and, for reaches one to three, the colour count of the shipped probe
colouring (:func:`~aquaflux.precondition.sparse_jacobian.block_stencil_colouring`). Numpy and scipy only;
about 1 GB at the default ``n`` = 20.

Usage: ``python validation/polyhedral_stencil_size.py [n]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.sparse as sp
from aquaflux.precondition.sparse_jacobian import block_stencil_colouring
from scipy.spatial import Delaunay

MAX_REACH = 6
MARGIN = 6


def adjacency(owner: np.ndarray, neighbour: np.ndarray, n: int) -> sp.csr_matrix:
    """The cell graph, self loops included."""
    rows = np.r_[owner, neighbour, np.arange(n)]
    cols = np.r_[neighbour, owner, np.arange(n)]
    a = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    return a


def ball_sizes(a: sp.csr_matrix, interior: np.ndarray) -> dict[int, float]:
    """Mean count of cells within each reach of ~40 interior seeds, by breadth-first search."""
    seeds = np.flatnonzero(interior)[:: max(1, int(interior.sum()) // 40)]
    sizes: dict[int, list[int]] = {r: [] for r in range(1, MAX_REACH + 1)}
    for seed in seeds:
        seen = np.zeros(a.shape[0], bool)
        seen[seed] = True
        front = np.array([seed])
        for r in range(1, MAX_REACH + 1):
            reached = np.unique(
                np.concatenate([a.indices[a.indptr[f] : a.indptr[f + 1]] for f in front])
            )
            front = reached[~seen[reached]]
            seen[front] = True
            sizes[r].append(int(seen.sum()))
    return {r: float(np.mean(v)) for r, v in sizes.items()}


def hex_graph(n: int):
    index = np.arange(n**3).reshape(n, n, n)
    edges = []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        edges.append(np.c_[index[tuple(lo)].ravel(), index[tuple(hi)].ravel()])
    edges = np.vstack(edges)
    centres = np.indices((n,) * 3).reshape(3, -1).T.astype(float)
    return edges[:, 0], edges[:, 1], n**3, centres


def dual_of_tets_graph(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    lattice = (
        np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3).astype(float)
    )
    points = lattice + rng.uniform(-0.35, 0.35, lattice.shape)
    tets = Delaunay(points).simplices
    pairs = np.vstack([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1], len(points), points


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for name, (owner, neighbour, cells, centres) in (
        ("hex", hex_graph(n)),
        ("dual-of-tets", dual_of_tets_graph(n)),
    ):
        a = adjacency(owner, neighbour, cells)
        interior = np.all((centres >= MARGIN) & (centres <= n - 1 - MARGIN), axis=1)
        faces = np.diff(a.indptr) - 1
        print(
            f"{name}: {cells} cells, faces a cell (interior) {faces[interior].mean():.2f}",
            flush=True,
        )
        for reach, size in ball_sizes(a, interior).items():
            print(f"  reach {reach}: {size:7.1f} cells", flush=True)
        for reach in (1, 2, 3):
            colouring = block_stencil_colouring(owner, neighbour, cells, reach)
            print(f"  colours at reach {reach}: {colouring.n_colours}", flush=True)


if __name__ == "__main__":
    main()
