"""Unit tests for graph partitioning: the CSR export and the partitioner strategies.

The partitioner must produce a valid, balanced labelling (every cell in exactly one of ``n_parts``
partitions, near-equal sizes) with a small edge cut. The dependency-free ``BlockPartitioner`` is
tested everywhere; ``ScotchCLIPartitioner`` runs wherever the Scotch ``gpart`` CLI is on PATH (macOS
via Homebrew or Linux) and skips otherwise; the ``ScotchPartitioner`` Python binding is exercised
only where ``scotchpy`` is installed. Each Scotch strategy's missing-dependency error path is checked
unconditionally.
"""

from __future__ import annotations

import shutil

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import pytest
from aquaflux.mesh import cell_adjacency_csr, structured_grid_3d
from aquaflux.parallel import (
    BlockPartitioner,
    ScotchCLIPartitioner,
    ScotchPartitioner,
)


def _edge_cut(mesh, labels) -> int:
    """Number of interior faces whose two cells lie in different partitions."""
    owner = np.asarray(mesh.face_cells.owner)
    nb = np.asarray(mesh.face_cells.neighbour)
    interior = nb >= 0
    return int(np.sum(labels[owner[interior]] != labels[nb[interior]]))


def test_cell_adjacency_csr_matches_interior_faces() -> None:
    mesh = structured_grid_3d(4, 4, 4)
    offsets, neighbours = cell_adjacency_csr(mesh)
    n_interior = int(np.sum(np.asarray(mesh.face_cells.neighbour) >= 0))
    assert offsets.shape == (mesh.n_cells + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == neighbours.shape[0] == 2 * n_interior  # symmetric edges
    assert neighbours.min() >= 0 and neighbours.max() < mesh.n_cells


@pytest.mark.parametrize("n_parts", [2, 3, 4, 8])
def test_block_partitioner_is_valid_and_balanced(n_parts) -> None:
    mesh = structured_grid_3d(8, 8, 8, named_boundaries=True)
    labels = BlockPartitioner().partition(mesh, n_parts)

    assert labels.shape == (mesh.n_cells,)
    assert labels.min() == 0 and labels.max() == n_parts - 1
    counts = np.bincount(labels, minlength=n_parts)
    assert (counts > 0).all()  # no empty partition
    # balanced: every partition within one cell of the exact share
    assert counts.max() - counts.min() <= 1


def test_block_partitioner_has_small_cut_vs_scattered_labelling() -> None:
    """RCM-block partitions are spatially compact: far smaller cut than a scattered (mod) labelling."""
    mesh = structured_grid_3d(10, 10, 10)
    n_parts = 4
    block = BlockPartitioner().partition(mesh, n_parts)
    scattered = np.arange(mesh.n_cells) % n_parts  # maximally non-local
    # RCM-block is compact (space-filling-curve blocks), not optimal slabs — measured ~3x better.
    assert _edge_cut(mesh, block) < _edge_cut(mesh, scattered) / 2


def test_scotch_partitioner_reports_missing_binding() -> None:
    """Without the binding (module ``scotchpy``), ScotchPartitioner raises an actionable error."""
    try:
        import scotchpy  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="scotchpy"):
            ScotchPartitioner().partition(structured_grid_3d(3, 3, 3), 2)
    else:
        pytest.skip("scotchpy is installed; the missing-binding path is not exercised here")


def test_scotch_cli_partitioner_produces_balanced_low_cut_partition() -> None:
    """Scotch's `gpart` CLI (``brew install scotch`` / the ``scotch`` package) yields a valid,
    balanced, deterministic partition with a smaller cut than the block partitioner. Runs wherever
    the CLI is on PATH (macOS or Linux); skips otherwise."""
    if shutil.which("gpart") is None:
        pytest.skip("Scotch 'gpart' CLI not on PATH (install: brew install scotch)")
    mesh = structured_grid_3d(8, 8, 8)
    n_parts = 4
    labels = ScotchCLIPartitioner().partition(mesh, n_parts)

    assert set(np.unique(labels)) == set(range(n_parts))
    counts = np.bincount(labels, minlength=n_parts)
    assert counts.max() - counts.min() <= 0.05 * mesh.n_cells + 1  # ~5% default imbalance
    assert _edge_cut(mesh, labels) < _edge_cut(mesh, BlockPartitioner().partition(mesh, n_parts))
    assert np.array_equal(labels, ScotchCLIPartitioner().partition(mesh, n_parts))  # deterministic


def test_scotch_cli_partitioner_reports_missing_tool() -> None:
    """A clear, actionable error when the `gpart` command is absent."""
    with pytest.raises(FileNotFoundError, match="gpart"):
        ScotchCLIPartitioner(command="gpart-does-not-exist").partition(
            structured_grid_3d(3, 3, 3), 2
        )


def test_scotch_partitioner_produces_valid_balanced_partition() -> None:
    """Where the Scotch binding is installed (e.g. Linux CI), it yields a valid, balanced,
    low-cut partition — no worse than the block partitioner's compact cut. Skips otherwise."""
    pytest.importorskip("scotchpy")
    mesh = structured_grid_3d(8, 8, 8)
    n_parts = 4
    labels = ScotchPartitioner().partition(mesh, n_parts)

    assert labels.shape == (mesh.n_cells,)
    assert set(np.unique(labels)).issubset(set(range(n_parts)))
    counts = np.bincount(labels, minlength=n_parts)
    assert (counts > 0).all()
    assert counts.max() <= 1.5 * counts.min()  # within Scotch's default load imbalance
    block_cut = _edge_cut(mesh, BlockPartitioner().partition(mesh, n_parts))
    assert _edge_cut(mesh, labels) <= block_cut  # min-cut objective beats compact blocks
