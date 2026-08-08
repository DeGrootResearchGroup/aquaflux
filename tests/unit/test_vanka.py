"""The Vanka patch smoother: which unknowns each patch holds, and what a patch solve does with them.

Every test here runs on a hand-sized matrix and needs no PETSc — the smoother is deliberately a pure
``numpy``/``scipy`` object, so the two pieces (choosing patches, solving them) are checked separately
and the exactness properties are checked against a dense inverse.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.vanka import (
    CellStarPatches,
    VankaSmoother,
    _patch_blocks,
    colour_patches,
)

_shell_prefixes = itertools.count()


def chain_matrix(n_cells: int, n_fields: int, *, seed: int = 0) -> sp.csr_matrix:
    """A diagonally dominant block matrix coupling every cell to every other."""
    rng = np.random.default_rng(seed)
    n = n_cells * n_fields
    dense = rng.normal(size=(n, n))
    dense += np.diag(np.abs(dense).sum(axis=1) + 1.0)
    return sp.csr_matrix(dense)


def coupling_matrix(weights: dict[tuple[int, int], float], n_cells: int) -> sp.csr_matrix:
    """A one-field-per-cell matrix with a unit diagonal and the given off-diagonal couplings."""
    dense = np.eye(n_cells)
    for (row, col), value in weights.items():
        dense[row, col] = value
    return sp.csr_matrix(dense)


def test_patch_keeps_the_most_strongly_coupled_neighbours():
    matrix = coupling_matrix({(0, 1): 0.1, (0, 2): 5.0, (0, 3): 1.0}, 4)
    patches = CellStarPatches(n_neighbours=2, centre_field=0).patches(matrix, 1)
    assert sorted(patches.cells[0, 1:]) == [2, 3]
    assert patches.cell_mask[0].all()


def test_a_cell_with_too_few_neighbours_pads_and_masks():
    matrix = coupling_matrix({(0, 1): 2.0, (1, 0): 2.0}, 3)
    patches = CellStarPatches(n_neighbours=2, centre_field=0).patches(matrix, 1)
    assert patches.cells[0, 1] == 1
    assert list(patches.cell_mask[0]) == [True, True, False]
    # Cell 2 couples to nothing, so both neighbour slots are padding.
    assert list(patches.cell_mask[2]) == [True, False, False]


def test_coverage_counts_only_real_slots():
    matrix = coupling_matrix({(0, 1): 2.0, (1, 0): 2.0}, 3)
    patches = CellStarPatches(n_neighbours=2, centre_field=0).patches(matrix, 1)
    # Cell 0's patch holds {0, 1}, cell 1's holds {1, 0}, cell 2's holds {2} alone.
    assert list(patches.coverage()) == [2, 2, 1]


def test_patch_blocks_are_the_submatrices_of_the_operator():
    matrix = chain_matrix(4, 2, seed=1)
    patches = CellStarPatches(n_neighbours=1, centre_field=0).patches(matrix, 2)
    blocks = _patch_blocks(matrix, patches)
    dense = matrix.toarray()
    for patch in range(patches.n_patches):
        dofs = patches.dofs()[patch]
        np.testing.assert_allclose(blocks[patch], dense[np.ix_(dofs, dofs)])


def test_no_neighbours_is_exact_block_jacobi():
    matrix = chain_matrix(4, 3, seed=2)
    patches = CellStarPatches(n_neighbours=0, centre_field=0).patches(matrix, 3)
    smoother = VankaSmoother(matrix, patches)
    residual = np.arange(1.0, 13.0)
    dense = matrix.toarray()
    expected = np.concatenate(
        [
            np.linalg.solve(
                dense[3 * i : 3 * i + 3, 3 * i : 3 * i + 3], residual[3 * i : 3 * i + 3]
            )
            for i in range(4)
        ]
    )
    np.testing.assert_allclose(smoother.apply(residual), expected)


def test_patches_covering_the_whole_operator_give_its_exact_inverse():
    """Every patch solves the whole system, so averaging the identical corrections is ``A^-1``."""
    matrix = chain_matrix(3, 2, seed=3)
    patches = CellStarPatches(n_neighbours=2, centre_field=0).patches(matrix, 2)
    assert patches.width == matrix.shape[0]
    smoother = VankaSmoother(matrix, patches)
    residual = np.array([1.0, -2.0, 3.0, 0.5, -0.25, 4.0])
    np.testing.assert_allclose(
        smoother.apply(residual), np.linalg.solve(matrix.toarray(), residual)
    )


def test_transpose_is_the_adjoint_of_the_apply():
    matrix = chain_matrix(5, 2, seed=4)
    patches = CellStarPatches(n_neighbours=2, centre_field=1).patches(matrix, 2)
    smoother = VankaSmoother(matrix, patches)
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=10), rng.normal(size=10)
    assert smoother.apply(x) @ y == pytest.approx(x @ smoother.apply(y, transpose=True))


def test_the_apply_is_a_fixed_linear_operator():
    """A V-cycle used by a non-flexible Krylov solve, and by an adjoint, needs this to hold."""
    matrix = chain_matrix(5, 2, seed=6)
    patches = CellStarPatches(n_neighbours=2, centre_field=0).patches(matrix, 2)
    smoother = VankaSmoother(matrix, patches)
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=10), rng.normal(size=10)
    np.testing.assert_allclose(
        smoother.apply(3.0 * x - 2.0 * y), 3.0 * smoother.apply(x) - 2.0 * smoother.apply(y)
    )


def test_damping_scales_the_correction():
    matrix = chain_matrix(4, 2, seed=8)
    patches = CellStarPatches(n_neighbours=1, centre_field=0).patches(matrix, 2)
    residual = np.arange(8.0)
    plain = VankaSmoother(matrix, patches).apply(residual)
    damped = VankaSmoother(matrix, patches, damping=0.25).apply(residual)
    np.testing.assert_allclose(damped, 0.25 * plain)


def test_neighbour_fields_restrict_which_unknowns_join_the_patch():
    matrix = chain_matrix(4, 3, seed=9)
    patches = CellStarPatches(n_neighbours=2, centre_field=2, neighbour_fields=(0, 1)).patches(
        matrix, 3
    )
    assert patches.width == 3 + 2 * 2
    # The centre cell contributes every field; each neighbour contributes only fields 0 and 1.
    assert list(patches.slot_field) == [0, 1, 2, 0, 1, 0, 1]
    assert list(patches.slot_cell) == [0, 0, 0, 1, 1, 2, 2]


def test_neighbours_are_ranked_by_the_couplings_the_patch_will_actually_invert():
    """A strong coupling through a field the patch excludes must not win a neighbour slot.

    The continuity row of a collocated discretization carries a pressure-pressure damping term
    alongside its divergence entries; ranking on the whole row would let that term choose the
    patch's cells, when what the patch exists to invert is the divergence coupling.
    """
    dense = np.eye(6)
    dense[1, 3] = 10.0  # cell 0's centre row -> cell 1, through the excluded field 1
    dense[1, 4] = 1.0  # cell 0's centre row -> cell 2, through the included field 0
    patches = CellStarPatches(n_neighbours=1, centre_field=1, neighbour_fields=(0,)).patches(
        sp.csr_matrix(dense), 2
    )
    assert patches.cells[0, 1] == 2


@pytest.mark.parametrize(
    "builder, message",
    [
        (CellStarPatches(centre_field=7), "centre_field"),
        (CellStarPatches(n_neighbours=-1), "n_neighbours"),
        (CellStarPatches(neighbour_fields=(0, 9)), "neighbour_fields"),
    ],
)
def test_a_patch_builder_rejects_an_out_of_range_choice(builder, message):
    with pytest.raises(ValueError, match=message):
        builder.patches(chain_matrix(3, 2), 2)


def test_a_patch_builder_rejects_a_size_that_is_not_a_multiple_of_the_fields():
    with pytest.raises(ValueError, match="not a multiple"):
        CellStarPatches().patches(sp.eye(5, format="csr"), 2)


def test_the_smoother_rejects_patches_that_do_not_match_the_operator():
    patches = CellStarPatches(n_neighbours=1).patches(chain_matrix(3, 2), 2)
    with pytest.raises(ValueError, match="degrees of freedom"):
        VankaSmoother(chain_matrix(4, 2), patches)


def model_saddle(n: int) -> sp.csr_matrix:
    """A field-major ``(u, v, p)`` saddle on a periodic ``n x n`` grid.

    A five-point Laplacian on each velocity component, a centred pressure gradient in the momentum
    rows against the matching divergence in the continuity row, and a small pressure-pressure
    damping — the algebraic shape of a collocated flow Jacobian, small enough to solve directly.
    """
    cells = n * n
    cell = lambda i, j: (i % n) * n + (j % n)  # noqa: E731
    rows, cols, values = [], [], []

    def add(row_cell, row_field, col_cell, col_field, value):
        rows.append(row_field * cells + row_cell)
        cols.append(col_field * cells + col_cell)
        values.append(value)

    for i in range(n):
        for j in range(n):
            here = cell(i, j)
            for field in (0, 1):
                add(here, field, here, field, 4.4)
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    add(here, field, cell(i + di, j + dj), field, -1.0)
            for field, (di, dj) in enumerate(((1, 0), (0, 1))):
                add(here, field, cell(i + di, j + dj), 2, 0.5)
                add(here, field, cell(i - di, j - dj), 2, -0.5)
                add(here, 2, cell(i + di, j + dj), field, -0.5)
                add(here, 2, cell(i - di, j - dj), field, 0.5)
            add(here, 2, here, 2, 1.0)
    matrix = sp.coo_matrix((values, (rows, cols)), shape=(3 * cells, 3 * cells)).tocsr()
    matrix.sum_duplicates()
    return matrix


def vcycle_with_vanka(matrix, **vanka_options):
    """A V-cycle over ``matrix`` whose level smoother is the shell Vanka preconditioner."""
    from aquaflux.solve import build_amg_vcycle

    return build_amg_vcycle(
        matrix,
        3,
        smoother_fill_levels=0,
        smoother_sweeps=2,
        coarse_eq_limit=60,
        extra_options={
            "mg_levels_pc_type": "python",
            "mg_levels_pc_python_type": "aquaflux.solve.vanka.VankaPC",
            "mg_levels_vanka_centre_field": 2,
            **{f"mg_levels_{key}": value for key, value in vanka_options.items()},
        },
    )


def test_the_shell_smoother_installs_on_a_plain_matrix_and_the_cycle_contracts():
    """A plain assembled matrix is enough — no ``DM``, which the patch smoothers PETSc ships require."""
    pytest.importorskip("petsc4py")
    matrix = model_saddle(16)
    vcycle = vcycle_with_vanka(matrix, vanka_neighbours=4)
    rng = np.random.default_rng(0)
    rhs = rng.normal(size=matrix.shape[0])
    residual = matrix @ vcycle.apply(rhs) - rhs
    assert np.linalg.norm(residual) / np.linalg.norm(rhs) < 0.7


def test_the_shell_smoother_keeps_the_cycle_a_fixed_linear_operator():
    """A non-flexible outer Krylov solve, and the adjoint's transpose, both depend on this."""
    pytest.importorskip("petsc4py")
    matrix = model_saddle(12)
    vcycle = vcycle_with_vanka(matrix, vanka_neighbours=4)
    rng = np.random.default_rng(1)
    x, y = (rng.normal(size=matrix.shape[0]) for _ in range(2))
    np.testing.assert_allclose(
        vcycle.apply(2.0 * x - y), 2.0 * vcycle.apply(x) - vcycle.apply(y), atol=1e-12
    )
    assert vcycle.apply(x) @ y == pytest.approx(x @ vcycle.apply(y, transpose=True))


def shell_smoother(matrix, n_fields, **options):
    """The :class:`VankaSmoother` a ``VankaPC`` builds for ``matrix`` under ``options``."""
    from aquaflux.solve.vanka import VankaPC
    from petsc4py import PETSc

    operator = PETSc.Mat().createAIJWithArrays(
        size=matrix.shape,
        csr=(
            matrix.indptr.astype(PETSc.IntType),
            matrix.indices.astype(PETSc.IntType),
            matrix.data.astype(PETSc.ScalarType),
        ),
    )
    operator.setBlockSize(n_fields)
    operator.assemble()
    prefix = f"vk{next(_shell_prefixes)}_"  # a fresh prefix per call, so options never carry over
    for key, value in options.items():
        PETSc.Options()[prefix + key] = value
    pc = PETSc.PC().create()
    pc.setOptionsPrefix(prefix)
    pc.setOperators(operator)
    context = VankaPC()
    context.setUp(pc)
    return context._smoother


def test_the_shell_translates_its_options_into_the_patch_it_advertises():
    """The neighbours really join the patch — a mistranslation here degrades silently to block-Jacobi.

    ``before_centre`` is the classical Vanka choice, and it has to resolve to the fields ahead of the
    pressure rather than to nothing; a patch that quietly shrank to one cell would still converge, just
    as a different smoother than the one under test.
    """
    pytest.importorskip("petsc4py")
    matrix = model_saddle(8)
    velocity = shell_smoother(matrix, 3, vanka_centre_field=2, vanka_neighbours=4)
    assert (
        velocity._patches.width == 3 + 4 * 2
    )  # the two velocity fields of each of four neighbours
    every = shell_smoother(
        matrix, 3, vanka_centre_field=2, vanka_neighbours=4, vanka_neighbour_fields="all"
    )
    assert every._patches.width == 3 + 4 * 3


def test_the_shell_refuses_a_block_too_small_for_the_layout_it_would_assume():
    """Rather than centre the patch on field 0 and take no neighbour fields at all."""
    pytest.importorskip("petsc4py")
    with pytest.raises(ValueError, match="no field ahead"):
        shell_smoother(model_saddle(6), 3, vanka_neighbours=4)


def test_a_gain_cap_drops_the_runaway_patch_and_reweights_the_rest():
    """One near-singular patch is enough to make an additive smoother amplify, so it can be excluded.

    The weights must then be a partition of unity over the patches that *remain* — otherwise dropping
    one silently under-corrects every unknown it used to cover.
    """
    matrix = chain_matrix(4, 2, seed=11).tolil()
    matrix[2, :] = 0.0  # make cell 1's block nearly singular, so its patch inverse blows up
    matrix[2, 2] = 1e-12
    matrix = matrix.tocsr()
    patches = CellStarPatches(n_neighbours=0, centre_field=0).patches(matrix, 2)
    uncapped = VankaSmoother(matrix, patches)
    assert uncapped.worst_patch_gain > 1e9
    assert uncapped.dropped_patches == 0

    capped = VankaSmoother(matrix, patches, max_patch_gain=1e3)
    assert capped.dropped_patches == 1
    residual = np.ones(8)
    correction = capped.apply(residual)
    assert np.all(correction[2:4] == 0.0)  # the dropped patch contributes nothing
    # The surviving patches are disjoint here, so each still reproduces its own exact block solve.
    dense = matrix.toarray()
    np.testing.assert_allclose(correction[0:2], np.linalg.solve(dense[0:2, 0:2], residual[0:2]))


def reference_coloured_sweep(matrix, patches, residual):
    """The coloured sweep written out literally, to check the restricted-update version against."""
    dense, dofs, mask = matrix.toarray(), patches.dofs(), patches.mask()
    colour, n_colours = colour_patches(patches)
    total = np.zeros(patches.n_dofs)
    for group in range(n_colours):
        updated = residual - dense @ total  # the whole residual, refreshed before the group runs
        for patch in np.flatnonzero(colour == group):
            take = dofs[patch][mask[patch]]
            if take.size:
                total[take] += np.linalg.solve(dense[np.ix_(take, take)], updated[take])
    return total


def test_colouring_groups_only_patches_that_share_nothing():
    matrix = chain_matrix(6, 2, seed=20)
    patches = CellStarPatches(n_neighbours=1, centre_field=0).patches(matrix, 2)
    colour, n_colours = colour_patches(patches)
    assert n_colours >= 1
    cells, cell_mask = patches.cells, patches.cell_mask
    for group in range(n_colours):
        seen: set[int] = set()
        for patch in np.flatnonzero(colour == group):
            owned = set(cells[patch][cell_mask[patch]].tolist())
            assert not (owned & seen)  # disjoint, so they can be applied simultaneously
            seen |= owned


def test_the_restricted_update_matches_a_full_residual_refresh():
    """The sweep recomputes each group's residual over only the rows that group touches.

    That is the optimization that makes a multiplicative sweep affordable -- roughly one full
    matrix-vector product per sweep instead of one per colour -- and it has to give exactly what
    refreshing the entire residual before every group would.
    """
    matrix = chain_matrix(6, 2, seed=21)
    patches = CellStarPatches(n_neighbours=1, centre_field=0).patches(matrix, 2)
    residual = np.arange(1.0, 13.0)
    smoother = VankaSmoother(matrix, patches, multiplicative=True)
    np.testing.assert_allclose(
        smoother.apply(residual), reference_coloured_sweep(matrix, patches, residual), atol=1e-10
    )


def test_multiplicative_carries_no_overlap_weighting():
    """Its running residual update already accounts for earlier patches; weighting would double-count."""
    matrix = chain_matrix(4, 2, seed=22)
    patches = CellStarPatches(n_neighbours=0, centre_field=0).patches(matrix, 2)
    residual = np.arange(8.0)
    # With disjoint patches a multiplicative sweep and an averaged additive one both reduce to the
    # exact block inverse, so they must agree here and diverge only once the patches overlap.
    additive = VankaSmoother(matrix, patches).apply(residual)
    multiplicative = VankaSmoother(matrix, patches, multiplicative=True).apply(residual)
    np.testing.assert_allclose(multiplicative, additive)


def test_multiplicative_beats_additive_at_reducing_the_residual():
    """One sweep of each on a saddle: sequencing the patches is the stronger relaxation."""
    matrix = model_saddle(10)
    patches = CellStarPatches(n_neighbours=4, centre_field=2, neighbour_fields=(0, 1)).patches(
        matrix, 3
    )
    rng = np.random.default_rng(23)
    rhs = rng.normal(size=matrix.shape[0])
    additive = VankaSmoother(matrix, patches)
    multiplicative = VankaSmoother(matrix, patches, multiplicative=True)
    left = lambda s: np.linalg.norm(matrix @ s.apply(rhs) - rhs) / np.linalg.norm(rhs)  # noqa: E731
    assert left(multiplicative) < left(additive)


def test_the_multiplicative_transpose_refuses_rather_than_returning_the_wrong_thing():
    matrix = chain_matrix(4, 2, seed=24)
    patches = CellStarPatches(n_neighbours=1, centre_field=0).patches(matrix, 2)
    smoother = VankaSmoother(matrix, patches, multiplicative=True)
    with pytest.raises(NotImplementedError, match="reverse-ordered"):
        smoother.apply(np.ones(8), transpose=True)
