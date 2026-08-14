"""Unit tests for the block-triangular field-split preconditioner.

These run on small dense-backed blocks with *exact* block inverses, so what is under test is the field
split's own algebra -- the partition, the retained triangle, and the transpose -- rather than the quality
of any multigrid cycle. That separation is the point: with exact diagonal blocks the split's behaviour is
completely predictable (a block-triangular operator is inverted exactly by its own triangle), so a
deviation is a defect in this module and nowhere else.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve import NativeHierarchyInverse, NativeSimpleInverse, NodalNativeInverse
from aquaflux.solve.field_split import (
    BlockTriangularFieldSplit,
    FieldGroups,
    _TrailingFirstFieldSplit,
)


class ExactInverse:
    """An exact block inverse, standing in for a multigrid V-cycle in the split's own tests."""

    def __init__(self, block: np.ndarray) -> None:
        self._inverse = np.linalg.inv(np.asarray(block, dtype=np.float64))

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        matrix = self._inverse.T if transpose else self._inverse
        return matrix @ np.asarray(residual, dtype=np.float64)


@pytest.fixture
def groups() -> FieldGroups:
    """Four cells, three leading fields and two trailing ones -- deliberately not square or equal."""
    return FieldGroups(n_cells=4, n_leading_fields=3, n_trailing_fields=2)


@pytest.fixture
def operator(groups: FieldGroups) -> np.ndarray:
    """A reproducible, diagonally dominant coupled operator of the partition's shape."""
    rng = np.random.default_rng(20260808)
    dense = rng.standard_normal((groups.n_dofs, groups.n_dofs))
    return dense + np.eye(groups.n_dofs) * groups.n_dofs


def split_for(operator: np.ndarray, groups: FieldGroups, *, flow_first: bool):
    """The field split over ``operator`` with exact diagonal blocks, leading or trailing first."""
    leading, leading_by_trailing, trailing_by_leading, trailing = groups.blocks(
        sp.csr_matrix(operator)
    )
    if flow_first:
        return BlockTriangularFieldSplit(
            ExactInverse(leading.toarray()),
            ExactInverse(trailing.toarray()),
            trailing_by_leading,
            groups,
        )
    return _TrailingFirstFieldSplit(
        ExactInverse(trailing.toarray()),
        ExactInverse(leading.toarray()),
        leading_by_trailing,
        groups,
    )


def as_matrix(split, n_dofs: int, *, transpose: bool = False) -> np.ndarray:
    """The split's action written out as a dense matrix, by applying it to each unit vector."""
    columns = [split.apply(e, transpose=transpose) for e in np.eye(n_dofs)]
    return np.column_stack(columns)


class TestFieldGroups:
    def test_the_groups_partition_the_degrees_of_freedom_without_gap_or_overlap(self, groups):
        covered = np.zeros(groups.n_dofs, dtype=int)
        covered[groups.leading] += 1
        covered[groups.trailing] += 1
        assert np.all(covered == 1)

    def test_a_group_is_contiguous_whole_fields_of_a_field_major_vector(self, groups):
        # Field-major: degree of freedom (cell i, field f) sits at f * n_cells + i, so the leading group
        # must be exactly the first `n_leading_fields` fields over every cell.
        field_of_dof = np.arange(groups.n_dofs) // groups.n_cells
        assert set(field_of_dof[groups.leading]) == {0, 1, 2}
        assert set(field_of_dof[groups.trailing]) == {3, 4}

    def test_blocks_reassemble_into_the_original_operator(self, groups, operator):
        a_ll, a_lt, a_tl, a_tt = groups.blocks(sp.csr_matrix(operator))
        rebuilt = np.block([[a_ll.toarray(), a_lt.toarray()], [a_tl.toarray(), a_tt.toarray()]])
        np.testing.assert_allclose(rebuilt, operator)

    def test_an_empty_side_is_refused(self):
        with pytest.raises(ValueError, match="not a split"):
            FieldGroups(n_cells=4, n_leading_fields=5, n_trailing_fields=0)

    def test_a_mismatched_matrix_is_refused(self, groups):
        with pytest.raises(ValueError, match="describes"):
            groups.blocks(sp.eye(groups.n_dofs + 1, format="csr"))


class TestBlockTriangularAlgebra:
    @pytest.mark.parametrize("flow_first", [True, False])
    def test_it_inverts_its_own_triangle_exactly(self, groups, operator, flow_first):
        """With exact diagonal blocks the split is the exact inverse of the triangular operator.

        This is the defining property, and it is what distinguishes a block-*triangular* split from a
        block-diagonal one: zeroing the discarded triangle of the operator leaves a matrix the split
        inverts to machine precision, coupling and all.
        """
        triangular = operator.copy()
        discarded = (
            np.s_[groups.leading, groups.trailing]
            if flow_first
            else np.s_[groups.trailing, groups.leading]
        )
        triangular[discarded] = 0.0
        split = split_for(operator, groups, flow_first=flow_first)
        applied = as_matrix(split, groups.n_dofs) @ triangular
        np.testing.assert_allclose(applied, np.eye(groups.n_dofs), atol=1e-10)

    @pytest.mark.parametrize("flow_first", [True, False])
    def test_the_retained_coupling_really_is_retained(self, groups, operator, flow_first):
        """The split must differ from the block-DIAGONAL preconditioner, which drops the coupling.

        Without this the tests would pass for an implementation that silently ignored ``C``, which is
        precisely the weaker object this preconditioner exists not to be.
        """
        split = split_for(operator, groups, flow_first=flow_first)
        diagonal_only = operator.copy()
        diagonal_only[groups.leading, groups.trailing] = 0.0
        diagonal_only[groups.trailing, groups.leading] = 0.0
        block_diagonal = split_for(diagonal_only, groups, flow_first=flow_first)
        assert not np.allclose(
            as_matrix(split, groups.n_dofs), as_matrix(block_diagonal, groups.n_dofs)
        )

    @pytest.mark.parametrize("flow_first", [True, False])
    def test_the_transpose_apply_is_the_transpose_of_the_forward_apply(
        self, groups, operator, flow_first
    ):
        """The adjoint's transpose solve uses ``M^T``, so it must BE the transpose, not merely resemble one.

        Formed by applying both directions to every unit vector, so nothing about the closed-form
        transpose is taken on trust.
        """
        split = split_for(operator, groups, flow_first=flow_first)
        forward = as_matrix(split, groups.n_dofs)
        transposed = as_matrix(split, groups.n_dofs, transpose=True)
        np.testing.assert_allclose(transposed, forward.T, atol=1e-12)

    @pytest.mark.parametrize("flow_first", [True, False])
    def test_it_is_a_linear_operator(self, groups, operator, flow_first):
        """A Krylov method may only use a preconditioner that is a fixed linear operator.

        An inner Krylov solve or any state dependence in a block would break this, and would force the
        outer solve to go flexible -- which the adjoint's transpose solve cannot do.
        """
        split = split_for(operator, groups, flow_first=flow_first)
        rng = np.random.default_rng(11)
        first, second = rng.standard_normal((2, groups.n_dofs))
        combined = split.apply(2.5 * first - 0.75 * second)
        separately = 2.5 * split.apply(first) - 0.75 * split.apply(second)
        np.testing.assert_allclose(combined, separately, atol=1e-12)

    def test_the_two_orderings_retain_opposite_triangles(self, groups, operator):
        """Leading-first and trailing-first are genuinely different preconditioners, not a relabelling."""
        leading_first = as_matrix(split_for(operator, groups, flow_first=True), groups.n_dofs)
        trailing_first = as_matrix(split_for(operator, groups, flow_first=False), groups.n_dofs)
        assert not np.allclose(leading_first, trailing_first)

    def test_a_coupling_of_the_wrong_orientation_is_refused(self, groups, operator):
        """A square-ish coupling passed the wrong way round would apply silently and precondition the
        wrong triangle, so the shape check is load-bearing rather than defensive."""
        _, leading_by_trailing, _, _ = groups.blocks(sp.csr_matrix(operator))
        with pytest.raises(ValueError, match="trailing equations by leading"):
            BlockTriangularFieldSplit(
                ExactInverse(np.eye(groups.n_leading_dofs)),
                ExactInverse(np.eye(groups.n_dofs - groups.n_leading_dofs)),
                leading_by_trailing,
                groups,
            )

    def test_destroy_releases_both_blocks(self, groups, operator):
        """The block inverses may hold host solver resources; both must be released, not just one."""
        released = []

        class Releasable(ExactInverse):
            def __init__(self, block, name):
                super().__init__(block)
                self._name = name

            def destroy(self):
                released.append(self._name)

        leading, _, trailing_by_leading, trailing = groups.blocks(sp.csr_matrix(operator))
        split = BlockTriangularFieldSplit(
            Releasable(leading.toarray(), "leading"),
            Releasable(trailing.toarray(), "trailing"),
            trailing_by_leading,
            groups,
        )
        split.destroy()
        assert sorted(released) == ["leading", "trailing"]


def _nodal_block(n_cells: int = 400, n_fields: int = 2, seed: int = 0) -> sp.csr_matrix:
    """A field-major transported-scalar pair on a chain graph, diagonally dominant."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for cell in range(n_cells):
        for offset in (0, 1, -1, 2, -2):
            other = cell + offset
            if not 0 <= other < n_cells:
                continue
            for row_field in range(n_fields):
                for col_field in range(n_fields):
                    diagonal = offset == 0 and row_field == col_field
                    vals.append(6.0 if diagonal else rng.normal() * 0.2)
                    rows.append(row_field * n_cells + cell)
                    cols.append(col_field * n_cells + other)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_cells * n_fields,) * 2)


def test_the_trailing_inverse_takes_the_hierarchy_knobs_its_sibling_had() -> None:
    """The coarsening knobs are the shared base's, so the trailing block is no longer capped at two.

    Both native inverses wrap a coarsened hierarchy and differ only in their level smoother, but the
    knobs had been added on one side of a duplicated seam: the saddle inverse grew a strength threshold,
    a level cap and a shape ladder, and this one grew none of them — so a trailing coarsening sweep could
    not be run at all. Not a decision, just where the code was written.
    """
    a = _nodal_block()
    plain = NodalNativeInverse(a, 2)
    assert len(plain._hierarchy.levels) == 2, "the two-level default should be unchanged"

    deeper = NodalNativeInverse(a, 2, max_levels=4, max_coarse=40)
    assert len(deeper._hierarchy.levels) == 4, "max_levels did not reach the coarsening"

    # And the value-dependent knobs build, which is what a sweep needs; a single-state probe never
    # refreshes, so the structure-invariance they cost does not arise there.
    for extra in (
        {"strength_threshold": 0.25},
        {"strength_threshold": 0.25, "shape_headroom": 1.3},
    ):
        inverse = NodalNativeInverse(a, 2, max_levels=4, max_coarse=40, **extra)
        x = np.asarray(inverse.apply(np.random.default_rng(1).normal(size=a.shape[0])))
        assert np.all(np.isfinite(x))


def test_the_trailing_refresh_keeps_the_argument_pytree_at_an_isotropic_coarsening() -> None:
    """At threshold zero the coarsening reads only the sparsity pattern, so a refresh cannot move it.

    That invariant is what makes the jitted cycle a compilation-cache hit across a march refresh, and it
    is the same property the saddle inverse's own test pins — which is the point of them now sharing a
    base rather than each carrying their own copy of the refresh.
    """
    a = _nodal_block()
    inverse = NodalNativeInverse(a, 2)
    inverse.apply(np.asarray(np.random.default_rng(2).normal(size=a.shape[0])))

    def signature(inv):
        leaves, structure = jax.tree_util.tree_flatten((inv._hierarchy, inv._extras))
        return structure, [(leaf.shape, leaf.dtype) for leaf in leaves]

    before = signature(inverse)
    inverse.refactor_block((a * 1.9).tocsr())

    assert signature(inverse) == before, "the refresh moved the argument pytree"


def test_both_native_inverses_share_one_refresh_implementation() -> None:
    """The seam itself, asserted: a capability added to the base reaches both smoothers.

    Written as a structural check rather than a behavioural one because the failure it guards against is
    a future divergence — someone re-adding a private `refactor_block` or `apply` to one class, which
    would silently take it off the shared path and let the two drift again.
    """
    for cls in (NodalNativeInverse, NativeSimpleInverse):
        assert issubclass(cls, NativeHierarchyInverse)
        for shared in ("refactor_block", "apply", "destroy", "n_dofs", "_rebuild", "_solve"):
            assert shared not in vars(cls), (
                f"{cls.__name__} overrides {shared!r}, which the shared base owns"
            )
