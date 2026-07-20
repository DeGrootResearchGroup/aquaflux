"""Graph partitioning: assign every cell to a partition for domain decomposition.

A :class:`Partitioner` maps a mesh to a per-cell partition label (``labels[cell] in [0, n_parts)``),
which :func:`~aquaflux.parallel.partition.partition_mesh` turns into owned + halo local meshes. The
objective is a **balanced min-edge-cut** partition: roughly equal cell counts per partition (load
balance) with few interior faces crossing partition boundaries (small halos / little communication).

Three concrete strategies, in increasing setup cost / partition quality:

- :class:`BlockPartitioner` — dependency-free (SciPy only). It orders cells with a locality-preserving
  reordering (reverse Cuthill--McKee by default) and cuts that ordering into contiguous balanced
  blocks. A space-filling-curve-style partition: always valid and perfectly balanced, with a
  reasonable (not optimal) cut. The portable default, used by the test suite.
- :class:`ScotchCLIPartitioner` — a proper Scotch k-way min-cut via the ``gpart`` **command-line
  tool**, which is available cross-platform (``brew install scotch`` on macOS, the ``scotch`` package
  on Linux). The portable way to get Scotch-quality partitions for local parallel runs on either
  platform; no Python binding needed.
- :class:`ScotchPartitioner` — the same min-cut via the in-process Scotch **Python binding**
  (``pip install scotchpy64``, module ``scotchpy``). Avoids the CLI's temporary files, but the
  prebuilt wheels are Linux x86_64 only, so on macOS prefer the CLI partitioner.

Both Scotch strategies consume the shared :func:`~aquaflux.mesh.cell_adjacency_csr` cell graph, so
the graph construction is shared and tested; only the Scotch invocation differs.

The cell graph itself comes from :mod:`aquaflux.mesh.graph` (``cell_adjacency_csr`` /
``cell_adjacency_coo``), the same graph the reordering code uses.
"""

from __future__ import annotations

import abc
import os
import shutil
import subprocess
import tempfile

import equinox as eqx
import numpy as np

from aquaflux.mesh import Mesh, ReverseCuthillMcKee, cell_adjacency_csr
from aquaflux.mesh.reorder import CellReordering


class Partitioner(eqx.Module):
    """Strategy interface: assign each cell to one of ``n_parts`` partitions."""

    @abc.abstractmethod
    def partition(self, mesh: Mesh, n_parts: int) -> np.ndarray:
        """Return the partition label per cell, shape ``(n_cells,)`` with values in ``[0, n_parts)``.

        Parameters
        ----------
        mesh : Mesh
            The mesh to decompose.
        n_parts : int
            Number of partitions.

        Returns
        -------
        numpy.ndarray of int, shape ``(n_cells,)``
        """


class BlockPartitioner(Partitioner):
    """Balanced contiguous blocks of a locality-preserving cell ordering (SciPy-only default).

    Orders cells with ``reordering`` (reverse Cuthill--McKee by default, which places adjacent cells
    at nearby indices) and assigns the first ``~n_cells / n_parts`` ordered cells to partition 0, the
    next block to partition 1, and so on. Because the ordering is spatially local, the blocks are
    spatially compact, giving a reasonable edge cut; because the blocks are equal-sized, the load is
    balanced by construction. No external dependency.

    Attributes
    ----------
    reordering : CellReordering
        The ordering whose contiguous blocks become partitions (default
        :class:`~aquaflux.mesh.ReverseCuthillMcKee`).
    """

    reordering: CellReordering

    def __init__(self, reordering: CellReordering | None = None):
        self.reordering = reordering if reordering is not None else ReverseCuthillMcKee()

    def partition(self, mesh: Mesh, n_parts: int) -> np.ndarray:
        order = self.reordering.permutation(mesh)  # order[cell] = position in the reordering
        return (order * n_parts) // mesh.n_cells


def _import_scotch():
    """Import the Scotch Python binding, or raise a clear, actionable error.

    The pip package is ``scotchpy64`` but the importable module is ``scotchpy``.
    """
    try:
        import scotchpy
    except ImportError as exc:  # pragma: no cover - exercised only without the binding
        raise ImportError(
            "ScotchPartitioner requires the Scotch Python binding (module 'scotchpy'). "
            "Install it with 'pip install scotchpy64'. Prebuilt wheels are Linux x86_64 only; "
            "on macOS/other platforms build Scotch from source, or use BlockPartitioner (SciPy-only)."
        ) from exc
    return scotchpy


class ScotchPartitioner(Partitioner):
    """K-way min-cut partitioning via the Scotch library (optional ``scotchpy64`` dependency).

    Builds the shared :func:`~aquaflux.mesh.cell_adjacency_csr` cell graph and hands it to Scotch, which computes a
    balanced minimum-edge-cut k-way partition — better cuts (smaller halos) than
    :class:`BlockPartitioner`, at the cost of the external dependency (module ``scotchpy``; prebuilt
    wheels are Linux x86_64 only). This is the production path; the in-repo test suite uses
    :class:`BlockPartitioner` so it runs everywhere, so this partitioner is exercised only where the
    binding is installed — a smoke test skips when it is absent.
    """

    def partition(self, mesh: Mesh, n_parts: int) -> np.ndarray:
        scotchpy = _import_scotch()
        adj_offsets, adj_neighbours = cell_adjacency_csr(mesh)
        graph = scotchpy.Graph()
        # Unweighted symmetric cell graph; baseval is derived from verttab[0] == 0.
        graph.build(verttab=adj_offsets, edgetab=adj_neighbours)
        parttab = np.zeros(mesh.n_cells, dtype=np.int64)  # Scotch fills this in place
        graph.part(n_parts, parttab=parttab)
        return parttab


def _write_scotch_graph(path: str, adj_offsets: np.ndarray, adj_neighbours: np.ndarray) -> None:
    """Write a cell graph to Scotch's text graph (``.grf``) format.

    Format: a version line ``0``; ``vertnbr edgenbr`` (``edgenbr`` = number of arcs =
    ``len(adj_neighbours)``); ``baseval flagval`` (``0 000`` = base 0, no vertex labels/weights, no
    edge weights); then one line per vertex, ``degree n1 n2 ... nk``.

    Parameters
    ----------
    path : str
        Output file path.
    adj_offsets, adj_neighbours : numpy.ndarray
        The CSR cell graph from :func:`~aquaflux.mesh.cell_adjacency_csr`.
    """
    vertnbr = adj_offsets.shape[0] - 1
    degrees = np.diff(adj_offsets)
    with open(path, "w") as f:
        f.write(f"0\n{vertnbr}\t{adj_neighbours.shape[0]}\n0\t000\n")
        for c in range(vertnbr):
            seg = adj_neighbours[adj_offsets[c] : adj_offsets[c + 1]]
            f.write(f"{degrees[c]}\t" + "\t".join(map(str, seg.tolist())) + "\n")


def _read_scotch_mapping(path: str, n_cells: int) -> np.ndarray:
    """Read a Scotch mapping (``.map``) file into a per-cell label array.

    Format: a first line with the vertex count, then one ``vertex_label part_id`` pair per line.
    """
    labels = np.empty(n_cells, dtype=np.int64)
    with open(path) as f:
        count = int(f.readline())
        if count != n_cells:
            raise ValueError(f"Scotch mapping has {count} vertices; expected {n_cells}")
        for line in f:
            vertex, part = line.split()
            labels[int(vertex)] = int(part)
    return labels


class ScotchCLIPartitioner(Partitioner):
    """K-way min-cut partitioning via Scotch's ``gpart`` command-line tool (cross-platform).

    Writes the shared :func:`~aquaflux.mesh.cell_adjacency_csr` cell graph to a temporary Scotch ``.grf`` file, runs
    ``gpart`` to compute a balanced minimum-edge-cut partition, and reads back the mapping. Unlike
    :class:`ScotchPartitioner` (which needs the Linux-only ``scotchpy`` wheel), this works wherever
    the Scotch **command-line tools** are installed — ``brew install scotch`` on macOS, the ``scotch``
    package on Linux — so it is the portable way to run Scotch-quality partitions in local parallel
    mode on either platform. No Python binding required.

    Attributes
    ----------
    command : str
        The ``gpart`` executable name or path (default ``"gpart"``; must be on ``PATH``).
    imbalance : float
        Load-imbalance tolerance passed to ``gpart -b`` (default ``0.05`` = 5%). Higher tolerance
        lets Scotch trade balance for a smaller cut.
    deterministic : bool
        When ``True`` (default), pass ``-Cd`` so the same mesh yields the same partition (reproducible
        science and stable tests).
    """

    command: str = eqx.field(static=True)
    imbalance: float = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)

    def __init__(self, command: str = "gpart", imbalance: float = 0.05, deterministic: bool = True):
        self.command = command
        self.imbalance = imbalance
        self.deterministic = deterministic

    def partition(self, mesh: Mesh, n_parts: int) -> np.ndarray:
        if n_parts == 1:
            return np.zeros(mesh.n_cells, dtype=np.int64)
        if shutil.which(self.command) is None:
            raise FileNotFoundError(
                f"'{self.command}' not found on PATH. Install the Scotch command-line tools "
                "('brew install scotch' on macOS, the 'scotch' package on Linux), or use "
                "BlockPartitioner (SciPy-only)."
            )
        adj_offsets, adj_neighbours = cell_adjacency_csr(mesh)
        with tempfile.TemporaryDirectory() as tmp:
            grf = os.path.join(tmp, "graph.grf")
            mapping = os.path.join(tmp, "graph.map")
            _write_scotch_graph(grf, adj_offsets, adj_neighbours)
            argv = [self.command, str(n_parts), grf, mapping, f"-b{self.imbalance}"]
            if self.deterministic:
                argv.append("-Cd")
            subprocess.run(argv, check=True, capture_output=True, text=True)
            return _read_scotch_mapping(mapping, mesh.n_cells)
