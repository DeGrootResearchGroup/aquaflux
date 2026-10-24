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
from aquaflux.precondition import (
    AirReduction,
    HierarchyBlockInverse,
    JacobiSmoothedInverse,
    SimpleSmoothedInverse,
)
from aquaflux.precondition.entry_map import EntryMaps
from aquaflux.precondition.field_split import FieldGroups, FieldSplitInverse
from aquaflux.precondition.padded_rows import _PaddedRowOperator
from aquaflux.solve import CellFields, FieldLayout, GlobalDofs, SubLayout


class ExactInverse:
    """An exact block inverse, standing in for a multigrid V-cycle in the split's own tests."""

    def __init__(self, block: np.ndarray) -> None:
        self._inverse = np.linalg.inv(np.asarray(block, dtype=np.float64))

    @property
    def n_dofs(self) -> int:
        return self._inverse.shape[0]

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        matrix = self._inverse.T if transpose else self._inverse
        return matrix @ np.asarray(residual, dtype=np.float64)


@pytest.fixture
def groups() -> FieldGroups:
    """Four cells, three leading fields and two trailing ones -- deliberately not square or equal."""
    return FieldGroups.by_counts(n_cells=4, n_leading_fields=3, n_trailing_fields=2)


@pytest.fixture
def operator(groups: FieldGroups) -> np.ndarray:
    """A reproducible, diagonally dominant coupled operator of the partition's shape."""
    rng = np.random.default_rng(20260808)
    dense = rng.standard_normal((groups.n_dofs, groups.n_dofs))
    return dense + np.eye(groups.n_dofs) * groups.n_dofs


def split_for(operator: np.ndarray, groups: FieldGroups) -> FieldSplitInverse:
    """The field split over ``operator`` with exact diagonal blocks."""
    leading, _, trailing_by_leading, trailing = groups.blocks(sp.csr_matrix(operator))
    return FieldSplitInverse(
        ExactInverse(leading.toarray()),
        ExactInverse(trailing.toarray()),
        trailing_by_leading,
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
        with pytest.raises(ValueError, match="at least one field"):
            FieldGroups.by_counts(n_cells=4, n_leading_fields=5, n_trailing_fields=0)
        with pytest.raises(ValueError, match="not a split"):
            FieldGroups(FieldLayout.cell_fields(4, all_of_them=5), 5)

    def test_a_bordered_state_is_refused_because_its_border_belongs_to_no_field(self):
        """A partition into whole fields cannot describe a constraint multiplier."""
        bordered = FieldLayout.cell_fields(4, flow=3, k=1).appended(GlobalDofs("mass_flow", 1))
        with pytest.raises(ValueError, match="whole per-cell field"):
            FieldGroups(bordered, 3)

    def test_split_before_names_the_partition_against_the_states_own_blocks(self):
        layout = FieldLayout(
            4,
            (
                SubLayout("flow", FieldLayout.cell_fields(4, velocity=3, pressure=1)),
                CellFields("k", 1),
                CellFields("omega", 1),
            ),
        )
        groups = FieldGroups.split_before(layout, "k")
        assert groups.n_leading_fields == 4
        assert groups.n_trailing_fields == 2
        assert groups.leading == slice(0, 16)

    def test_a_mismatched_matrix_is_refused(self, groups):
        with pytest.raises(ValueError, match="describes"):
            groups.blocks(sp.eye(groups.n_dofs + 1, format="csr"))

    def test_active_rows_excludes_exactly_the_triangle_the_split_drops(self, groups):
        active = groups.active_rows()
        assert active.shape == (groups.n_fields, groups.n_fields)
        nl = groups.n_leading_fields
        assert not active[:nl, nl:].any()  # leading rows <- trailing columns: dropped
        assert active[:nl, :nl].all()  # leading diagonal: kept
        assert active[nl:, nl:].all()  # trailing diagonal: kept
        assert active[nl:, :nl].all()  # trailing rows <- leading columns: the RETAINED coupling

    def test_the_dropped_block_never_reaches_the_splits_own_apply(self, groups, operator):
        """The whole point of ``active_rows``: zeroing the block it excludes must not move the split's
        output at all, since :class:`FieldSplitInverse` never reads it."""
        split = split_for(operator, groups)
        baseline = as_matrix(split, groups.n_dofs)

        active = groups.active_rows()
        zeroed = operator.copy()
        for a in range(groups.n_fields):
            for b in range(groups.n_fields):
                if not active[a, b]:
                    a_rows = slice(a * groups.n_cells, (a + 1) * groups.n_cells)
                    b_cols = slice(b * groups.n_cells, (b + 1) * groups.n_cells)
                    zeroed[a_rows, b_cols] = 0.0
        assert not np.array_equal(zeroed, operator), "the fixture must actually populate that block"

        split_on_zeroed = split_for(zeroed, groups)
        np.testing.assert_allclose(as_matrix(split_on_zeroed, groups.n_dofs), baseline)


class TestBlockTriangularAlgebra:
    def test_it_inverts_its_own_triangle_exactly(self, groups, operator):
        """With exact diagonal blocks the split is the exact inverse of the triangular operator.

        This is the defining property, and it is what distinguishes a block-*triangular* split from a
        block-diagonal one: zeroing the discarded triangle of the operator leaves a matrix the split
        inverts to machine precision, coupling and all.
        """
        triangular = operator.copy()
        triangular[groups.leading, groups.trailing] = 0.0
        split = split_for(operator, groups)
        applied = as_matrix(split, groups.n_dofs) @ triangular
        np.testing.assert_allclose(applied, np.eye(groups.n_dofs), atol=1e-10)

    def test_the_retained_coupling_really_is_retained(self, groups, operator):
        """The split must differ from the block-DIAGONAL preconditioner, which drops the coupling.

        Without this the tests would pass for an implementation that silently ignored ``C``, which is
        precisely the weaker object this preconditioner exists not to be.
        """
        split = split_for(operator, groups)
        diagonal_only = operator.copy()
        diagonal_only[groups.leading, groups.trailing] = 0.0
        diagonal_only[groups.trailing, groups.leading] = 0.0
        block_diagonal = split_for(diagonal_only, groups)
        assert not np.allclose(
            as_matrix(split, groups.n_dofs), as_matrix(block_diagonal, groups.n_dofs)
        )

    def test_the_transpose_apply_is_the_transpose_of_the_forward_apply(self, groups, operator):
        """The adjoint's transpose solve uses ``M^T``, so it must BE the transpose, not merely resemble one.

        Formed by applying both directions to every unit vector, so nothing about the closed-form
        transpose is taken on trust.
        """
        split = split_for(operator, groups)
        forward = as_matrix(split, groups.n_dofs)
        transposed = as_matrix(split, groups.n_dofs, transpose=True)
        np.testing.assert_allclose(transposed, forward.T, atol=1e-12)

    def test_it_is_a_linear_operator(self, groups, operator):
        """A Krylov method may only use a preconditioner that is a fixed linear operator.

        An inner Krylov solve or any state dependence in a block would break this, and would force the
        outer solve to go flexible -- which the adjoint's transpose solve cannot do.
        """
        split = split_for(operator, groups)
        rng = np.random.default_rng(11)
        first, second = rng.standard_normal((2, groups.n_dofs))
        combined = split.apply(2.5 * first - 0.75 * second)
        separately = 2.5 * split.apply(first) - 0.75 * split.apply(second)
        np.testing.assert_allclose(combined, separately, atol=1e-12)

    def test_a_coupling_of_the_wrong_orientation_is_refused(self, groups, operator):
        """A square-ish coupling passed the wrong way round would apply silently and precondition the
        wrong triangle, so the shape check is load-bearing rather than defensive."""
        _, leading_by_trailing, _, _ = groups.blocks(sp.csr_matrix(operator))
        with pytest.raises(ValueError, match="trailing equations by leading"):
            FieldSplitInverse(
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
        split = FieldSplitInverse(
            Releasable(leading.toarray(), "leading"),
            Releasable(trailing.toarray(), "trailing"),
            trailing_by_leading,
            groups,
        )
        split.destroy()
        assert sorted(released) == ["leading", "trailing"]

    def test_a_block_inverse_without_the_frozen_inverse_pair_is_refused_at_construction(
        self, groups, operator
    ):
        """A block lacking ``n_dofs`` would otherwise be accepted and fail only when first applied."""

        class NoSize:
            def apply(self, residual, *, transpose=False):
                return residual

        leading, _, trailing_by_leading, _ = groups.blocks(sp.csr_matrix(operator))
        with pytest.raises(
            TypeError, match="trailing block inverse NoSize is not a frozen inverse"
        ):
            FieldSplitInverse(
                ExactInverse(leading.toarray()), NoSize(), trailing_by_leading, groups
            )

    def test_a_refit_reproduces_a_split_built_at_the_new_operator(self, groups, operator):
        """``refactor_block`` re-fits both blocks AND the retained coupling, in place.

        Compared against a split built afresh at the new operator, so a refit that skipped either block
        or kept the stale coupling is caught: each of those leaves a different matrix.
        """

        class Refittable(ExactInverse):
            def refactor_block(self, block):
                self._inverse = np.linalg.inv(np.asarray(block.toarray(), dtype=np.float64))

        leading, _, trailing_by_leading, trailing = groups.blocks(sp.csr_matrix(operator))
        split = FieldSplitInverse(
            Refittable(leading.toarray()),
            Refittable(trailing.toarray()),
            trailing_by_leading,
            groups,
        )
        developed = operator + np.random.default_rng(7).standard_normal(operator.shape)
        split.refactor_block(sp.csr_matrix(developed))
        np.testing.assert_allclose(
            as_matrix(split, groups.n_dofs),
            as_matrix(split_for(developed, groups), groups.n_dofs),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            as_matrix(split, groups.n_dofs, transpose=True),
            as_matrix(split_for(developed, groups), groups.n_dofs, transpose=True),
            atol=1e-12,
        )

    def test_blocks_cut_by_held_maps_are_the_slices_bit_for_bit(self, groups, operator):
        """A refit cuts its operator into blocks by gathers held between refits; at new values on the
        same pattern each must be exactly the slice it replaces (a gather does no arithmetic), and the
        packed coupling exactly the coupling packed afresh."""

        class Refittable(ExactInverse):
            def refactor_block(self, block):
                self._inverse = np.linalg.inv(np.asarray(block.toarray(), dtype=np.float64))

        base = sp.csr_matrix(operator)
        leading, _, trailing_by_leading, trailing = groups.blocks(base)
        split = FieldSplitInverse(
            Refittable(leading.toarray()),
            Refittable(trailing.toarray()),
            trailing_by_leading,
            groups,
        )
        maps = EntryMaps()
        for scale in (1.0, -2.5):
            values = base.data * scale + 0.1 * np.arange(base.nnz)
            matrix = sp.csr_matrix((values, base.indices, base.indptr), shape=base.shape)
            for got, want in zip(groups.blocks(matrix, maps), groups.blocks(matrix), strict=True):
                assert np.array_equal(got.indptr, want.indptr)
                assert np.array_equal(got.indices, want.indices)
                assert np.array_equal(got.data, want.data)
            split.refactor_block(matrix)
            packed = _PaddedRowOperator.from_scipy(groups.blocks(matrix)[2])
            coupling = split.padded_coupling()
            for name in ("columns", "row_lengths", "values", "t_columns", "t_values"):
                assert np.array_equal(
                    np.asarray(getattr(coupling, name)), np.asarray(getattr(packed, name))
                )

    def test_a_refit_with_a_block_that_cannot_refit_changes_nothing(self, groups, operator):
        """The refusal comes before either block is touched, so a split is never left half re-fitted."""
        refitted = []

        class Refittable(ExactInverse):
            def refactor_block(self, block):
                refitted.append(block.shape)

        leading, _, trailing_by_leading, trailing = groups.blocks(sp.csr_matrix(operator))
        split = FieldSplitInverse(
            Refittable(leading.toarray()),
            ExactInverse(trailing.toarray()),
            trailing_by_leading,
            groups,
        )
        before = as_matrix(split, groups.n_dofs)
        with pytest.raises(TypeError, match="ExactInverse offers no refactor_block"):
            split.refactor_block(sp.csr_matrix(operator * 3.0))
        assert refitted == []
        np.testing.assert_array_equal(as_matrix(split, groups.n_dofs), before)


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
    plain = JacobiSmoothedInverse(a, 2)
    assert len(plain._hierarchy.levels) == 2, "the two-level default should be unchanged"

    deeper = JacobiSmoothedInverse(a, 2, max_levels=4, max_coarse=40)
    assert len(deeper._hierarchy.levels) == 4, "max_levels did not reach the coarsening"

    # `strength_threshold` must actually reach the coarsening, not just build something finite: at 0
    # (isotropic) the aggregation reads only the sparsity pattern, and above 0 it reads `|A_ij|`
    # (`_aggregation_edges`) and keeps only the strong connections, which changes which cells aggregate
    # together and therefore the coarse level sizes. On this fixture (400 cells, 2 fields, chain graph
    # with weak `N(0, 0.2)` off-diagonal coupling against a diagonal of 6.0), threshold 0.0 vs 0.6 at
    # `max_levels=4, max_coarse=40` coarsens to (124, 52, 22) vs (154, 68, 32) equations on levels 1-3 —
    # a hardcoded `strength_threshold = 0.0` inside `HierarchyBlockInverse.__init__` would build the
    # isotropic shapes regardless of what is passed here, so comparing the shapes catches it where
    # `isfinite` cannot.
    isotropic = JacobiSmoothedInverse(a, 2, max_levels=4, max_coarse=40, strength_threshold=0.0)
    thresholded = JacobiSmoothedInverse(a, 2, max_levels=4, max_coarse=40, strength_threshold=0.6)
    isotropic_shapes = [level.operator.shape for level in isotropic._hierarchy.levels]
    thresholded_shapes = [level.operator.shape for level in thresholded._hierarchy.levels]
    assert thresholded_shapes != isotropic_shapes, (
        "strength_threshold did not change the coarsening -- it may be discarded before it reaches "
        "the hierarchy builder"
    )

    # And the value-dependent knobs build, which is what a sweep needs; a single-state probe never
    # refreshes, so the structure-invariance they cost does not arise there.
    for extra in (
        {"strength_threshold": 0.6},
        {"strength_threshold": 0.6, "shape_headroom": 1.3},
    ):
        inverse = JacobiSmoothedInverse(a, 2, max_levels=4, max_coarse=40, **extra)
        x = np.asarray(inverse.apply(np.random.default_rng(1).normal(size=a.shape[0])))
        assert np.all(np.isfinite(x))


def test_the_trailing_refresh_keeps_the_argument_pytree_at_an_isotropic_coarsening() -> None:
    """At threshold zero the coarsening reads only the sparsity pattern, so a refresh cannot move it.

    That invariant is what makes the jitted cycle a compilation-cache hit across a march refresh, and it
    is the same property the saddle inverse's own test pins — which is the point of them now sharing a
    base rather than each carrying their own copy of the refresh.
    """
    a = _nodal_block()
    inverse = JacobiSmoothedInverse(a, 2)
    inverse.apply(np.asarray(np.random.default_rng(2).normal(size=a.shape[0])))

    def signature(inv):
        leaves, structure = jax.tree_util.tree_flatten((inv._hierarchy, inv._extras))
        return structure, [(leaf.shape, leaf.dtype) for leaf in leaves]

    before = signature(inverse)
    inverse.refactor_block((a * 1.9).tocsr())

    assert signature(inverse) == before, "the refresh moved the argument pytree"


def test_both_hierarchy_inverses_share_one_refresh_implementation() -> None:
    """The seam itself, asserted: a capability added to the base reaches both smoothers.

    Written as a structural check rather than a behavioural one because the failure it guards against is
    a future divergence — someone re-adding a private `refactor_block` or `apply` to one class, which
    would silently take it off the shared path and let the two drift again.
    """
    for cls in (JacobiSmoothedInverse, SimpleSmoothedInverse):
        assert issubclass(cls, HierarchyBlockInverse)
        for shared in ("refactor_block", "apply", "destroy", "n_dofs", "_rebuild", "_solve"):
            assert shared not in vars(cls), (
                f"{cls.__name__} overrides {shared!r}, which the shared base owns"
            )


def test_field_split_refresh_in_place_no_longer_takes_the_dead_smoother_parameters(groups) -> None:
    """#287: the split used to declare ``smoother_fill_levels``/``smoother_sweeps`` on its refresh and
    immediately ``del`` them, purely because it shared a base with a monolithic multigrid preconditioner
    (since deleted, whose ``build`` genuinely read them) and the two refresh signatures were forced to agree. Now that
    the shared base is :class:`~aquaflux.precondition.MaterializedJacobianPreconditioner`, which knows
    nothing about a smoother, the split's refresh does not take them -- passing either is a ``TypeError``
    rather than a silent no-op. (Its ``build`` no longer takes them either: each block is fitted by an
    injected inverse.)
    """
    from aquaflux.precondition.field_split import FieldSplitPreconditioner

    n = groups.n_dofs
    operator = np.eye(n) * 2.0 + np.eye(n, k=1) * 0.25
    split = split_for(operator, groups)
    preconditioner = FieldSplitPreconditioner(split, groups)
    with pytest.raises(TypeError):
        preconditioner.refresh_in_place(
            lambda v: v, None, np.zeros(n), smoother_fill_levels=0, smoother_sweeps=4
        )


def _two_field_transport(n_cells: int = 40, coupling: float = 30.0) -> sp.csr_matrix:
    """Field-major two-field convection-diffusion block, coupled within each cell."""
    upwind = sp.diags(
        [-np.full(n_cells - 1, 4.0), np.full(n_cells, 6.0), -np.full(n_cells - 1, 1.0)],
        [-1, 0, 1],
        format="csr",
    )
    cross = sp.identity(n_cells, format="csr") * coupling
    return sp.bmat([[upwind, cross], [cross, upwind]], format="csr")


def test_air_block_inverse_applies_transposes_and_refreshes_in_place() -> None:
    """The lAIR trailing inverse against the three things the field split requires of one.

    ``refactor_block`` is the one a single-state probe never reaches and a march depends on:
    ``FieldSplitInverse.refactor`` raises on an inverse without it, because replacing the object
    would recompile the coupled solve that holds it.
    """
    block = _two_field_transport()
    inverse = AirReduction(max_coarse=8)(block, 2)
    assert inverse.n_dofs == block.shape[0]

    rng = np.random.default_rng(0)
    r = rng.standard_normal(block.shape[0])
    x = inverse.apply(r)
    assert np.all(np.isfinite(x))
    # A fixed cycle count makes it a linear map, which the non-flexible outer Krylov requires.
    r2 = rng.standard_normal(block.shape[0])
    assert np.allclose(
        inverse.apply(1.5 * r - 0.5 * r2), 1.5 * x - 0.5 * inverse.apply(r2), atol=1e-9
    )
    # <u, M r> == <M^T u, r>, which is what the adjoint's transpose solve needs (R != P^T here).
    u = rng.standard_normal(block.shape[0])
    assert np.dot(u, inverse.apply(r)) == pytest.approx(
        np.dot(inverse.apply(u, transpose=True), r), rel=1e-9
    )

    # ...and it refreshes in place onto a new operator of the same graph, keeping every shape.
    developed = _two_field_transport(coupling=45.0)
    inverse.refactor_block(developed)
    refreshed = inverse.apply(r)
    assert np.all(np.isfinite(refreshed))
    assert not np.allclose(refreshed, x), "the refresh did not change the preconditioner"


def _stored_pairs(narrow, wide):
    """The two inverses' traced-cycle leaves side by side, with whether each is a prolongation value."""
    narrow_leaves = jax.tree_util.tree_leaves_with_path(narrow.traced_cycle().state)
    wide_leaves = jax.tree.leaves(wide.traced_cycle().state)
    assert len(narrow_leaves) == len(wide_leaves)
    return [
        ("p_val" in jax.tree_util.keystr(path), n, w)
        for (path, n), w in zip(narrow_leaves, wide_leaves, strict=True)
    ]


@pytest.mark.parametrize("family", ["jacobi", "simple"])
def test_an_inverse_held_in_single_precision_holds_its_double_fit_rounded_once(family) -> None:
    """``stored_precision="float32"`` keeps one hierarchy, rounded once, and still re-fits as a double one.

    A split applied in single precision used to hold the double hierarchy and narrow a copy of it for
    the step, so two hierarchies were alive where one is read. Held narrow, every floating entry must be
    the double fit's entry rounded once -- exactly what narrowing it later gave -- except the
    prolongation values, which a frozen refit reads back and so must stay double, or every refit after
    the first would drift. Both are checked after a refit too, and the host apply must stay an exactly
    linear map (a cycle run in single precision is linear only to ~1e-7).
    """
    from tests.unit.test_saddle_multigrid import _saddle

    if family == "jacobi":
        a, n_fields, build = _nodal_block(), 2, JacobiSmoothedInverse
    else:
        a, n_fields, build = _saddle(), 4, SimpleSmoothedInverse
    settings = {"strength_threshold": 0.25, "max_levels": 4, "max_coarse": 40}
    settings["frozen_coarsening"] = True
    wide = build(a, n_fields, **settings)
    narrow = build(a, n_fields, **settings, stored_precision="float32")

    for refit in (False, True):
        if refit:
            developed = (a * 1.9).tocsr()
            developed.setdiag(developed.diagonal() * 1.1)
            wide.refactor_block(developed)
            narrow.refactor_block(developed)
        pairs = _stored_pairs(narrow, wide)
        assert any(is_p for is_p, _, _ in pairs), "the fixture must have a prolongation to keep"
        for is_prolongation, held, double in pairs:
            if not jax.numpy.issubdtype(double.dtype, jax.numpy.floating):
                assert np.array_equal(held, double)
            elif is_prolongation:
                assert held.dtype == double.dtype == np.float64
                assert np.array_equal(held, double)
            else:
                assert held.dtype == np.float32
                assert np.array_equal(held, double.astype(np.float32))

    rng = np.random.default_rng(1)
    first, second = rng.standard_normal(a.shape[0]), rng.standard_normal(a.shape[0])
    together = narrow.apply(first + 2.0 * second)
    apart = narrow.apply(first) + 2.0 * narrow.apply(second)
    assert np.allclose(together, apart, rtol=1e-12, atol=1e-12 * np.abs(apart).max())
    assert together.dtype == np.float64


class _RecordingInverse(ExactInverse):
    """An exact block inverse that also records the block it was last fitted to."""

    def refactor_block(self, block: sp.csr_matrix) -> None:
        self.fitted = block.toarray()
        self._inverse = np.linalg.inv(self.fitted)


@pytest.mark.parametrize("probe", ["whole matrix", "into blocks"])
def test_a_diagonal_correction_reaches_the_split_s_fit_on_either_refit_path(
    probe, groups, operator, monkeypatch
) -> None:
    """A re-fit's ``diagonal_correction`` (a pivot floor, a stand-in operator's diagonal) is fitted.

    A field split probed into its blocks never forms the whole matrix, so it takes a different path
    through the re-fit from the one that adds a correction to the whole matrix's shift. Each path must
    hand the correction the unshifted Jacobian's diagonal and fit Jacobian + shift + correction; a path
    that dropped the correction would fit Jacobian + shift with no error.
    """
    from types import SimpleNamespace

    from aquaflux.precondition.field_split import FieldSplitPreconditioner
    from aquaflux.precondition.probe_layout import BlockLayout, ProbedBlocks

    jacobian = sp.csr_matrix(operator)
    leading, _, trailing_by_leading, trailing = groups.blocks(jacobian)
    split = FieldSplitInverse(
        _RecordingInverse(leading.toarray()),
        _RecordingInverse(trailing.toarray()),
        trailing_by_leading,
        groups,
    )
    preconditioner = FieldSplitPreconditioner(split, groups)
    if probe == "into blocks":
        layout, order = BlockLayout.of(jacobian, groups.n_leading_dofs)
        monkeypatch.setattr(
            FieldSplitPreconditioner,
            "_probed_blocks",
            staticmethod(lambda *_args: ProbedBlocks(jacobian.data[order].copy(), layout)),
        )
        structure = SimpleNamespace(layout=layout)
    else:
        monkeypatch.setattr(
            FieldSplitPreconditioner, "_materialize_jacobian", staticmethod(lambda *_args: jacobian)
        )
        structure = None
    rng = np.random.default_rng(7)
    shift = rng.uniform(1.0, 2.0, groups.n_dofs)
    correction = rng.uniform(3.0, 5.0, groups.n_dofs)
    seen = []

    def diagonal_correction(jacobian_diagonal):
        seen.append(np.array(jacobian_diagonal))
        return correction

    preconditioner.refresh_in_place(
        None, None, shift, structure=structure, diagonal_correction=diagonal_correction
    )

    np.testing.assert_array_equal(seen[0], operator.diagonal())
    want_leading, _, _, want_trailing = groups.blocks(
        sp.csr_matrix(operator + np.diag(shift + correction))
    )
    np.testing.assert_allclose(split._leading.fitted, want_leading.toarray(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(split._trailing.fitted, want_trailing.toarray(), rtol=0, atol=1e-12)


def test_a_refit_probed_into_blocks_stays_on_the_device_and_fits_what_the_host_cut_fits() -> None:
    """Blocks probed on the device reach both hierarchies and the coupling without a host read-back.

    The values, the shift and the coupling stay on the device from the probe to the fit, so nothing in
    the refit may read a block back to the host; and the fit must be the one the host path gives from
    the whole shifted matrix, cut and handed over as host blocks -- the same hierarchies, entry for
    entry, and the same coupling. The coupling's host copy is made only when something reads it.
    """
    import jax.numpy as jnp
    from aquaflux.precondition import field_split, hierarchy_inverse
    from aquaflux.precondition.probe_layout import BlockLayout, ProbedBlocks

    n_cells = 200
    groups = FieldGroups.by_counts(n_cells, n_leading_fields=2, n_trailing_fields=2)
    rng = np.random.default_rng(11)
    coupling = sp.random(2 * n_cells, 2 * n_cells, density=0.01, random_state=4, format="csr")

    def operator(scale: float) -> sp.csr_matrix:
        whole = sp.bmat(
            [
                [_nodal_block(n_cells, seed=0), None],
                [scale * coupling, _nodal_block(n_cells, seed=1)],
            ]
        )
        return sp.csr_matrix(whole * scale)

    def build() -> FieldSplitInverse:
        leading, _, trailing_by_leading, trailing = groups.blocks(operator(1.0))
        return FieldSplitInverse(
            JacobiSmoothedInverse(leading, 2),
            JacobiSmoothedInverse(trailing, 2),
            trailing_by_leading,
            groups,
            stored_precision="float32",
        )

    on_device, on_host = build(), build()
    developed = operator(1.7)
    shift = rng.uniform(1.0, 2.0, groups.n_dofs)
    layout, order = BlockLayout.of(developed, groups.n_leading_dofs)
    probed = ProbedBlocks(jnp.asarray(developed.data[order]), layout)
    probed.add_shift(jnp.asarray(shift))

    def no_host(_block):
        raise AssertionError("a refit probed on the device read a block back to the host")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(field_split, "as_scipy", no_host)
        patch.setattr(hierarchy_inverse, "as_scipy", no_host)
        on_device.refactor_blocks(probed.blocks())
    assert on_device._host_coupling is None
    on_host.refactor_block(developed + sp.diags(shift))

    for side in ("_leading", "_trailing"):
        device_leaves = jax.tree.leaves(getattr(on_device, side).traced_cycle().state)
        host_leaves = jax.tree.leaves(getattr(on_host, side).traced_cycle().state)
        assert len(device_leaves) == len(host_leaves)
        for got, want in zip(device_leaves, host_leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    want = groups.blocks(developed)[2].toarray()
    assert on_device.coupling.dtype == np.float32
    np.testing.assert_array_equal(on_device.coupling.toarray(), on_host.coupling.toarray())
    np.testing.assert_allclose(on_device.coupling.toarray(), want, rtol=1e-7, atol=0)
    np.testing.assert_array_equal(
        np.asarray(on_device.padded_coupling().values), np.asarray(on_host.padded_coupling().values)
    )


def test_narrowing_a_hierarchy_replaces_each_double_level_in_the_list_it_is_given() -> None:
    """``_narrowed`` consumes its list, so the double levels go one at a time, not all at the end.

    A refit forms the whole hierarchy in double precision before rounding it; narrowing it level by
    level, each double level dropped as its rounded copy is made, is what keeps the two whole copies
    from being alive together. Each level must still be the double level rounded once, its
    prolongation values kept double.
    """
    from aquaflux.precondition.hierarchy_inverse import _narrowed

    double = JacobiSmoothedInverse(
        _nodal_block(), 2, strength_threshold=0.25, max_levels=4, max_coarse=40
    )._hierarchy
    levels = list(double.levels)
    narrowed = _narrowed(levels, double.equilibration, "float32")
    assert len(levels) == len(double.levels) > 1
    for held, kept, level in zip(levels, narrowed.levels, double.levels, strict=True):
        assert held is kept
        assert held.operator.values.dtype == jax.numpy.float32
        assert np.array_equal(held.operator.values, np.asarray(level.operator.values, np.float32))
        if level.p_val is not None:
            assert held.p_val is level.p_val
