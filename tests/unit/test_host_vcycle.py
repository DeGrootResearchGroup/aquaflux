"""The host V-cycle over the traced hierarchy: does it precondition, and is its transpose exact?

The transpose is the reason this file exists. A V-cycle is symmetric only if its smoother is, and an
incomplete factorization is not — so ``M^T`` had to be built rather than borrowed, and a mistake there
is invisible from the forward solve: the march converges perfectly well on ``M`` while every gradient
taken through ``M^T`` is wrong. The adjoint identity is therefore asserted directly, on a deliberately
NONSYMMETRIC operator where a wrong transpose cannot coincide with the right one.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve.ilu_inverse import IluSmoothedInverse, ilu_smoothed_inverse
from aquaflux.solve.ordering import (
    AscendingRowLengthCells,
    CellMajor,
    ReverseCuthillMcKeeCells,
)


def _nonsymmetric_block(
    n_cells: int = 240, n_fields: int = 4, seed: int = 0, diagonal: float = 8.0
) -> sp.csr_matrix:
    """A field-major multi-field operator that is emphatically not symmetric.

    Asymmetry is the point: on a symmetric operator a transposed V-cycle coincides with the forward one,
    so the adjoint identity would pass for an implementation that ignored ``transpose`` entirely.

    ``diagonal`` sets how diagonally dominant it is, and therefore how hard the V-cycle has to work.
    The default is strongly dominant, which suits every test that wants a preconditioner that simply
    works; lower it where a test needs the cycle to leave measurable error behind.
    """
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for cell in range(n_cells):
        for offset in (0, 1, -1, 2):
            other = cell + offset
            if not 0 <= other < n_cells:
                continue
            for row_field in range(n_fields):
                for col_field in range(n_fields):
                    if offset == 0 and row_field == col_field:
                        value = diagonal
                    elif offset == 0:
                        value = 0.9
                    else:
                        # Direction-dependent, so A != A^T by construction.
                        value = rng.normal() * (0.30 if offset > 0 else 0.05)
                    rows.append(row_field * n_cells + cell)
                    cols.append(col_field * n_cells + other)
                    vals.append(value)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_cells * n_fields,) * 2)


_SETTINGS = dict(max_coarse=40, max_levels=3, sweeps=2)


def test_it_actually_preconditions_the_operator() -> None:
    """The point of the object: applied as ``M`` it must reduce the TRUE residual ``||A x - b||``.

    Asserted on the true residual rather than a preconditioned norm — the measure that has produced the
    most retracted verdicts on this operator class.
    """
    a = _nonsymmetric_block()
    inverse = IluSmoothedInverse(a, 4, **_SETTINGS)
    b = np.asarray(np.random.default_rng(1).normal(size=a.shape[0]))

    x = inverse.apply(b)

    assert np.all(np.isfinite(x))
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 0.5


def test_the_cycle_is_a_fixed_linear_operator() -> None:
    """``b -> x`` must be LINEAR, or the non-flexible outer GMRES it preconditions is invalid.

    A fixed cycle count, a fixed sweep count and a stationary smoother are what buy this; a
    Krylov-accelerated smoother would break it silently — the outer solve would still run and would
    simply converge to the wrong thing.
    """
    a = _nonsymmetric_block()
    inverse = IluSmoothedInverse(a, 4, **_SETTINGS)
    rng = np.random.default_rng(2)
    u = rng.normal(size=a.shape[0])
    v = rng.normal(size=a.shape[0])

    combined = inverse.apply(3.0 * u - 2.0 * v)
    separate = 3.0 * inverse.apply(u) - 2.0 * inverse.apply(v)

    np.testing.assert_allclose(combined, separate, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("cycles", [1, 2])
@pytest.mark.parametrize("sweeps", [1, 3])
def test_the_transpose_is_exact(cycles: int, sweeps: int) -> None:
    """``<y, M x> == <M^T y, x>`` — the identity the implicit-function-theorem adjoint rests on.

    Swept over cycle and sweep counts because the transposed recursion has to reverse the whole
    sequence, and an error in the ordering shows up only once there is more than one of something.
    """
    a = _nonsymmetric_block()
    inverse = IluSmoothedInverse(a, 4, max_coarse=40, max_levels=3, sweeps=sweeps, cycles=cycles)
    rng = np.random.default_rng(3)
    x = rng.normal(size=a.shape[0])
    y = rng.normal(size=a.shape[0])

    forward = float(y @ inverse.apply(x))
    transposed = float(inverse.apply(y, transpose=True) @ x)

    assert abs(forward - transposed) <= 1e-9 * max(abs(forward), 1.0)


def test_the_transpose_is_not_the_forward_cycle() -> None:
    """The identity above must not be passing because ``transpose`` is ignored.

    On a nonsymmetric operator ``M`` and ``M^T`` genuinely differ, so this pins that the flag does
    something — without it, an implementation that returned the forward cycle for both would satisfy
    every other test in this file.
    """
    a = _nonsymmetric_block()
    inverse = IluSmoothedInverse(a, 4, **_SETTINGS)
    b = np.asarray(np.random.default_rng(4).normal(size=a.shape[0]))

    forward = inverse.apply(b)
    transposed = inverse.apply(b, transpose=True)

    assert not np.allclose(forward, transposed)


def test_refactor_refits_in_place_onto_a_new_operator() -> None:
    """A mid-march refresh must mutate the object the solve holds, not replace it."""
    a = _nonsymmetric_block()
    inverse = IluSmoothedInverse(a, 4, **_SETTINGS)
    b = np.asarray(np.random.default_rng(5).normal(size=a.shape[0]))
    before = inverse.apply(b)

    inverse.refactor_block((a * 1.7).tocsr())
    after = inverse.apply(b)

    assert np.all(np.isfinite(after))
    assert not np.allclose(before, after), "the refresh did not re-fit to the new operator"


def test_a_mismatched_block_is_rejected() -> None:
    """Refreshing onto a differently-sized block raises rather than building something incoherent."""
    inverse = IluSmoothedInverse(_nonsymmetric_block(n_cells=120), 4, **_SETTINGS)

    with pytest.raises(ValueError, match="cannot refactor"):
        inverse.refactor_block(_nonsymmetric_block(n_cells=80))


def _lattice_block(side: int = 24, n_fields: int = 2, seed: int = 1) -> sp.csr_matrix:
    """A five-point stencil on a ``side x side`` grid — a connectivity ILU(0) genuinely approximates.

    The chain fixture above cannot serve here. It is banded (offsets 0, ±1, +2), and on a narrow band
    a zero-fill factorization drops almost nothing, so it is very nearly a COMPLETE factorization and
    one V-cycle reaches machine precision by itself. On a two-dimensional grid the natural ordering
    puts a whole row's width between a cell and its vertical neighbour, so the elimination generates
    fill far outside the stored pattern, ILU(0) discards it, and the smoother is left genuinely
    inexact — which is the regime the coarse-grid correction exists for, and the only one in which
    "another cycle helps" is a statement about the recursion rather than about round-off.
    """
    rng = np.random.default_rng(seed)
    n_cells = side * side
    rows, cols, vals = [], [], []
    for j in range(side):
        for i in range(side):
            cell = j * side + i
            for di, dj, weight in (
                (0, 0, 4.0),
                (1, 0, -1.0),
                (-1, 0, -1.0),
                (0, 1, -1.0),
                (0, -1, -1.0),
            ):
                x, y = i + di, j + dj
                if not (0 <= x < side and 0 <= y < side):
                    continue
                other = y * side + x
                for row_field in range(n_fields):
                    for col_field in range(n_fields):
                        if row_field != col_field:
                            # Weak inter-field coupling, asymmetric so the block is not symmetric.
                            value = 0.05 * rng.normal() if (di or dj) == 0 else 0.0
                        else:
                            # Direction-dependent off-diagonals: advection on top of the Laplacian.
                            value = (
                                weight * (1.15 if di > 0 or dj > 0 else 0.85)
                                if (di or dj)
                                else weight
                            )
                        if value == 0.0:
                            continue
                        rows.append(row_field * n_cells + cell)
                        cols.append(col_field * n_cells + other)
                        vals.append(value)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_cells * n_fields,) * 2)


def test_more_cycles_reduce_the_residual_further() -> None:
    """Two cycles must beat one, which is the cheapest check that the recursion composes correctly.

    A V-cycle that mis-assembled its coarse-grid correction can still reduce the residual once, by
    smoothing alone; it stops improving when iterated.

    ⚠️ This needs an operator the smoother does NOT already solve. Asserted first on the chain fixture,
    where one cycle lands at ~7e-17 — the floating-point floor — so the comparison was between two
    round-off values and its ordering was platform luck: it passed locally and failed in CI. Both
    guards below exist so that failure mode announces itself as "the fixture is too easy" rather than
    as a broken recursion.
    """
    a = _lattice_block()
    b = np.asarray(np.random.default_rng(6).normal(size=a.shape[0]))

    def true_residual(cycles: int) -> float:
        inverse = IluSmoothedInverse(a, 2, max_coarse=64, max_levels=3, sweeps=1, cycles=cycles)
        return float(np.linalg.norm(a @ inverse.apply(b) - b) / np.linalg.norm(b))

    one, two = true_residual(1), true_residual(2)

    assert one > 1e-6, f"fixture too easy to discriminate: one cycle already reaches {one:.2e}"
    assert two < 0.5 * one, f"two cycles ({two:.2e}) did not clearly beat one ({one:.2e})"


def test_the_factory_builds_what_a_field_split_expects() -> None:
    """``ilu_smoothed_inverse`` returns the ``(block, n_fields) -> inverse`` shape the split calls."""
    a = _nonsymmetric_block()

    inverse = ilu_smoothed_inverse(**_SETTINGS)(a, 4)

    assert inverse.n_dofs == a.shape[0]


def test_the_default_ordering_is_unchanged() -> None:
    """Omitting ``ordering`` must be bit-identical to passing the cell-major default.

    Every recorded measurement of this smoother was taken before the ordering became injectable, so a
    default that moved would silently invalidate all of them while every other test still passed.
    """
    a = _nonsymmetric_block()
    b = np.asarray(np.random.default_rng(7).normal(size=a.shape[0]))

    implicit = IluSmoothedInverse(a, 4, **_SETTINGS).apply(b)
    explicit = IluSmoothedInverse(a, 4, ordering=CellMajor(), **_SETTINGS).apply(b)

    np.testing.assert_array_equal(implicit, explicit)


def test_a_non_default_ordering_changes_the_cycle() -> None:
    """The injected ordering must reach the factorization rather than being accepted and dropped.

    Without this, a smoother that ignored the argument would satisfy every other test here — including
    the adjoint identity below, which would then simply be re-checking the default.
    """
    a = _lattice_block()
    b = np.asarray(np.random.default_rng(8).normal(size=a.shape[0]))

    default = IluSmoothedInverse(a, 2, max_coarse=64, max_levels=3, sweeps=1).apply(b)
    reordered = IluSmoothedInverse(
        a, 2, max_coarse=64, max_levels=3, sweeps=1, ordering=CellMajor(ReverseCuthillMcKeeCells())
    ).apply(b)

    assert not np.allclose(default, reordered)


@pytest.mark.parametrize(
    "cells", [ReverseCuthillMcKeeCells(), AscendingRowLengthCells()], ids=lambda c: type(c).__name__
)
def test_the_transpose_is_exact_under_a_reordered_elimination(cells) -> None:
    """``<y, M x> == <M^T y, x>`` must survive a non-default ordering — the gradient depends on it.

    The transposed cycle undoes the permutation around each smoother application, so an ordering that
    was applied on the way in but not undone on the way out would leave the FORWARD solve perfectly
    healthy and every gradient silently wrong. That is the one failure mode this whole object is
    written to make impossible, and it has to be re-asserted for each order rather than assumed from
    the default.
    """
    a = _nonsymmetric_block()
    inverse = IluSmoothedInverse(a, 4, ordering=CellMajor(cells), **_SETTINGS)
    rng = np.random.default_rng(9)
    x = rng.normal(size=a.shape[0])
    y = rng.normal(size=a.shape[0])

    forward = float(y @ inverse.apply(x))
    transposed = float(inverse.apply(y, transpose=True) @ x)

    assert abs(forward - transposed) <= 1e-9 * max(abs(forward), 1.0)


def test_a_reordered_smoother_still_preconditions_the_operator() -> None:
    """A reordered elimination must remain a working approximate inverse, on the TRUE residual."""
    a = _lattice_block()
    b = np.asarray(np.random.default_rng(10).normal(size=a.shape[0]))

    inverse = IluSmoothedInverse(
        a, 2, max_coarse=64, max_levels=3, sweeps=1, ordering=CellMajor(ReverseCuthillMcKeeCells())
    )
    x = inverse.apply(b)

    assert np.all(np.isfinite(x))
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 0.5
