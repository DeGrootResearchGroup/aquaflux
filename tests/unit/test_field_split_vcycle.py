"""The field split driving REAL multigrid V-cycles, rather than the exact stand-in blocks.

``test_field_split.py`` pins the split's algebra with exact block inverses, which is the right way to test
the algebra but cannot catch a wiring fault against the actual V-cycle builder -- a sub-block handed the
wrong block size, an aggregation that will not coarsen a two-field operator, a permutation applied on the
wrong side of a group boundary. Those only appear once real hierarchies are built over the sub-blocks, and
they are expensive to discover in a case study rather than here. Skipped where ``petsc4py`` is unavailable.

The model operator is a two-group block system on a five-point Laplacian graph -- not the coupled saddle,
whose behaviour is the case study's subject. What is under test is that the pieces fit together and that
the assembled preconditioner is a contraction with a genuine transpose.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("petsc4py")

from aquaflux.solve import FieldGroups, build_block_triangular_field_split


def _laplacian_2d(n: int) -> sp.csr_matrix:
    """The 2D five-point Laplacian on an ``n x n`` grid (symmetric positive definite)."""
    ident = sp.identity(n)
    tri = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    return (sp.kron(ident, tri) + sp.kron(tri, ident)).tocsr()


@pytest.fixture(scope="module")
def groups() -> FieldGroups:
    """Four leading fields and two trailing ones, the shape of the coupled Reynolds-averaged split."""
    return FieldGroups(n_cells=20 * 20, n_leading_fields=4, n_trailing_fields=2)


@pytest.fixture(scope="module")
def operator(groups: FieldGroups) -> sp.csr_matrix:
    """A field-major block operator on a shared mesh graph, with real cross-group coupling.

    Each field gets the Laplacian on the same grid, scaled so the groups differ in magnitude the way the
    momentum and transport rows do, plus a diagonal coupling between every leading and trailing field so
    the retained triangle has something in it.
    """
    laplacian = _laplacian_2d(20)
    n = groups.n_cells
    scales = [1.0, 1.0, 1.0, 1.0, 30.0, 30.0]
    blocks = [[None] * groups.n_fields for _ in range(groups.n_fields)]
    for f in range(groups.n_fields):
        blocks[f][f] = scales[f] * laplacian
    # Cross-group coupling: every trailing field feels every leading one and vice versa, weakly.
    for lead in range(groups.n_leading_fields):
        for trail in range(groups.n_leading_fields, groups.n_fields):
            blocks[trail][lead] = 0.05 * sp.identity(n)
            blocks[lead][trail] = 0.02 * sp.identity(n)
    return sp.bmat(blocks, format="csr")


@pytest.mark.parametrize("flow_first", [True, False])
def test_the_split_contracts_the_residual(groups, operator, flow_first):
    """One application is a genuine approximate inverse of the FULL coupled operator.

    Not of its own triangle -- of the whole thing, coupling included, which is what an outer Krylov
    method is handed.
    """
    split = build_block_triangular_field_split(
        operator, groups, flow_first=flow_first, coarse_eq_limit=200
    )
    rng = np.random.default_rng(0)
    b = rng.standard_normal(groups.n_dofs)
    x = split.apply(b)
    assert np.linalg.norm(operator @ x - b) / np.linalg.norm(b) < 0.7
    split.destroy()


@pytest.mark.parametrize("flow_first", [True, False])
def test_the_transpose_is_the_adjoint_of_the_forward_apply(groups, operator, flow_first):
    """``<y, M x> == <M^T y, x>`` with real V-cycles, so the adjoint's transpose solve is sound.

    The inner-product form rather than a dense build: at this size forming the matrix column by column
    would mean thousands of multigrid applications, and the identity is what the adjoint actually relies
    on.
    """
    split = build_block_triangular_field_split(
        operator, groups, flow_first=flow_first, coarse_eq_limit=200
    )
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((2, groups.n_dofs))
    np.testing.assert_allclose(y @ split.apply(x), split.apply(y, transpose=True) @ x, rtol=1e-10)
    split.destroy()


def test_each_group_is_aggregated_at_its_own_block_size(groups, operator):
    """The point of splitting: the two groups get separate hierarchies, not one shared coarse space.

    Read off the built V-cycles rather than inferred -- a builder that quietly passed the full field
    count to both blocks would still produce a working preconditioner, just not a split one.
    """
    split = build_block_triangular_field_split(operator, groups, coarse_eq_limit=200)
    assert split._leading.n_dofs == groups.n_leading_dofs
    assert split._trailing.n_dofs == groups.n_dofs - groups.n_leading_dofs
    split.destroy()
