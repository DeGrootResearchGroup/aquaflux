"""Unit tests for cell renumbering: the canonical relabelling transform and the strategies.

Reordering changes only cell *indices*, never the geometry or physics — so the invariants are:
the renumbered mesh is still valid and closed, its cell geometry is a permutation of the
original, and reverse Cuthill--McKee actually reduces the cell-graph bandwidth of a scrambled
mesh. The convergence *consequences* of ordering are covered in the multigrid/flow suites.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import numpy as np
import pytest
from aquaflux.mesh import (
    CellZones,
    IdentityReordering,
    RandomReordering,
    ReverseCuthillMcKee,
    cell_adjacency_coo,
    closed_cell_residual,
    permute_cells,
    structured_grid_2d,
)

from tests.support.meshes import perturbed_grid_2d


def _bandwidth(mesh) -> int:
    """Max ``|owner - neighbour|`` over interior faces — the cell-graph bandwidth."""
    owner = np.asarray(mesh.face_cells.owner)
    nb = np.asarray(mesh.face_cells.neighbour)
    interior = nb >= 0
    return int(np.max(np.abs(owner[interior] - nb[interior])))


def test_identity_reordering_is_a_no_op() -> None:
    mesh = structured_grid_2d(6, 5)
    reordered, perm = IdentityReordering().apply(mesh)
    assert np.array_equal(perm, np.arange(mesh.n_cells))
    assert np.array_equal(np.asarray(reordered.face_cells.owner), np.asarray(mesh.face_cells.owner))
    assert np.array_equal(
        np.asarray(reordered.face_cells.neighbour), np.asarray(mesh.face_cells.neighbour)
    )


def test_permute_cells_preserves_geometry_up_to_relabelling() -> None:
    """A renumbered mesh is still closed, and its cell volumes are the original volumes permuted."""
    mesh = perturbed_grid_2d(8, 6, perturb=0.2)
    cg = mesh.geometry().cell
    perm = np.random.default_rng(0).permutation(mesh.n_cells)  # perm[old] = new

    renumbered = permute_cells(mesh, perm)
    cg_new = renumbered.geometry().cell

    # new_volume[perm[old]] = old_volume[old]  =>  new_volume = old_volume[argsort(perm)]
    inverse = np.argsort(perm)
    assert np.allclose(np.asarray(cg_new.volume), np.asarray(cg.volume)[inverse])
    assert float(np.max(np.abs(np.asarray(closed_cell_residual(renumbered))))) < 1e-10


def test_permute_cells_round_trips() -> None:
    """Applying a permutation then its inverse recovers the original connectivity."""
    mesh = perturbed_grid_2d(7, 7, perturb=0.1)
    perm = np.random.default_rng(1).permutation(mesh.n_cells)
    back = permute_cells(permute_cells(mesh, perm), np.argsort(perm))
    assert np.array_equal(np.asarray(back.face_cells.owner), np.asarray(mesh.face_cells.owner))
    assert np.array_equal(
        np.asarray(back.face_cells.neighbour), np.asarray(mesh.face_cells.neighbour)
    )


def test_rcm_reduces_bandwidth_of_a_scrambled_mesh() -> None:
    """RCM re-localizes a randomly-numbered mesh: its cell-graph bandwidth drops sharply."""
    mesh = structured_grid_2d(20, 20)
    scrambled, _ = RandomReordering(seed=0).apply(mesh)
    rcm, _ = ReverseCuthillMcKee().apply(scrambled)

    assert _bandwidth(rcm) < _bandwidth(scrambled)
    # a structured n x n grid has natural bandwidth ~n; RCM should land near that, not O(n^2)
    assert _bandwidth(rcm) < 3 * 20


def test_cell_adjacency_coo_matches_interior_faces() -> None:
    """The adjacency graph has exactly two (symmetric) entries per interior face."""
    mesh = structured_grid_2d(5, 4)
    n_interior = int(np.sum(np.asarray(mesh.face_cells.neighbour) >= 0))
    assert cell_adjacency_coo(mesh).nnz == 2 * n_interior


def test_permute_cells_remaps_zone_labels() -> None:
    """Zone membership follows the cell relabelling — not just owner/neighbour.

    Guards the single line that carries zone labels through a renumbering: without it, a
    reorder would silently scatter zone-dependent coefficients onto the wrong cells.
    """
    mesh = structured_grid_2d(3, 1)  # three cells 0, 1, 2 in a row
    zoned = eqx.tree_at(lambda m: m.cell_zones, mesh, CellZones.from_dict(3, {"a": [0], "c": [2]}))
    renumbered = permute_cells(zoned, np.array([2, 0, 1]))  # old 0->2, 1->0, 2->1

    assert list(np.where(np.asarray(renumbered.cell_zones.mask("a")))[0]) == [2]  # cell 0 -> 2
    assert list(np.where(np.asarray(renumbered.cell_zones.mask("c")))[0]) == [1]  # cell 2 -> 1
    assert renumbered.cell_zones.size("a") == 1 and renumbered.cell_zones.size("c") == 1


def test_permute_cells_rejects_non_bijection() -> None:
    """A non-bijective ``perm`` would silently corrupt the mesh; it is rejected up front."""
    mesh = structured_grid_2d(3, 1)
    with pytest.raises(ValueError, match="permutation"):
        permute_cells(mesh, np.array([0, 0, 1]))  # 0 twice, 2 missing
