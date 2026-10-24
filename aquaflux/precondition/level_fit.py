"""The values of an aggregation hierarchy's levels, fitted from the fine operator's values on the device.

An aggregation hierarchy is two kinds of thing. Its **structure** -- the aggregates, the prolongations,
the sparsity pattern of every level operator, the index arrays of its padded rows -- is decided on the
host, once, by a coarsening that walks graphs. Its **values** -- each level operator's entries, its
diagonal, the inverse of every cell's block, the spectral estimate a smoother may damp by, the coarsest
level's dense inverse -- are a fixed function of the fine operator's values once that structure is held.

This module is that function, written once in JAX and applied to a :class:`CoarseningPlan`, the structure
held on the device as fixed index maps. A first build fits its levels through it, and so does every refit
on a frozen coarsening: the fine operator's values go in, every level's values come out, and nothing
moves between host and device but a handful of check counts. The plan's maps are built by
:func:`level_plan` from patterns alone (:class:`~aquaflux.precondition.entry_map.EntryMap`).

The per-level kernels are also the single implementation their host callers use: the coarsening's
cell-block inverse (:func:`invert_cell_blocks`), the coarsest level's dense inverse
(:func:`dense_inverse`) and the power iteration behind a spectral estimate (:func:`spectral_estimate`).
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .entry_map import DeviceEntryMap, EntryMap, SparsePattern, ValueMap
from .kronecker_pattern import KroneckerGalerkin, KroneckerPattern, nodal_cell_transfer
from .padded_rows import _PaddedLayout, _PaddedRowOperator

#: Power-iteration steps behind a level's spectral estimate.
SPECTRAL_ITERATIONS = 20

#: A cell block whose :func:`block_singularity` falls below this is refused as singular.
SINGULAR_BLOCK = 1e-12


def cell_blocks(a: sp.csr_matrix, block_size: int) -> np.ndarray:
    """Each cell's own dense ``block_size × block_size`` block of a field-major operator, ``(n_cells, b, b)``.

    Cell ``i``'s degrees of freedom are ``{f * n_cells + i}``; an entry the operator does not store is
    zero. It only moves entries, so on a fixed pattern it is an
    :class:`~aquaflux.precondition.entry_map.EntryMap` gather (a :class:`_LevelPlan`'s ``cells``).
    """
    n_cells = a.shape[0] // block_size
    rows = np.arange(n_cells)
    blocks = np.empty((n_cells, block_size, block_size))
    for f in range(block_size):
        for g in range(block_size):
            blocks[:, f, g] = np.asarray(a[f * n_cells + rows, g * n_cells + rows]).ravel()
    return blocks


def block_singularity(blocks: jnp.ndarray) -> jnp.ndarray:
    """How near each square block is to singular, independent of its rows' and columns' scales, ``(n,)``.

    ``|det B|`` over the largest ``prod_i |B[i, s(i)]|`` among the permutations ``s``: scaling a row or a
    column multiplies the determinant and every such product by the same factor, so the measure is
    invariant under both, as the variables' units are arbitrary. It is ``1`` for a diagonal or
    triangular block with a nonzero diagonal and ``0`` for a singular one, or for a block with no nonzero
    product along any permutation (structurally singular). ``b!`` passes for ``(n, b, b)`` blocks.
    """
    blocks = jnp.asarray(blocks)
    n, b, _ = blocks.shape
    magnitude, rows = jnp.abs(blocks), np.arange(b)
    largest = jnp.zeros(n, dtype=blocks.dtype)
    for permutation in itertools.permutations(range(b)):
        largest = jnp.maximum(largest, jnp.prod(magnitude[:, rows, list(permutation)], axis=1))
    determinant = jnp.abs(jnp.linalg.det(blocks))
    return jnp.where(largest > 0.0, determinant / jnp.where(largest > 0.0, largest, 1.0), 0.0)


@jax.jit
def invert_cell_blocks(blocks: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The inverse of each cell block, and how many are singular (:data:`SINGULAR_BLOCK`).

    Parameters
    ----------
    blocks : jnp.ndarray
        The cell blocks, shape ``(n_cells, b, b)``.

    Returns
    -------
    inverse : jnp.ndarray
        Each block's inverse, shape ``(n_cells, b, b)``; not finite where a block is singular.
    singular : jnp.ndarray
        The count of singular blocks, a 0-d integer. A caller must refuse a nonzero count before using
        ``inverse``.
    """
    blocks = jnp.asarray(blocks)
    singular = jnp.sum(block_singularity(blocks) < SINGULAR_BLOCK)
    return jnp.linalg.inv(blocks), singular


def apply_block_inverse(
    block_inverse: jnp.ndarray, block_size: int, vector: jnp.ndarray
) -> jnp.ndarray:
    """Apply the per-cell block inverse to a field-major vector: reshape, contract per cell, flatten.

    Takes the two arrays it reads rather than a level, because both level families carry them and a
    function typed for one of them could not serve the other without pretending an lAIR level were an
    aggregation level.
    """
    per_cell = vector.reshape(block_size, -1).T  # (n_cells, block_size)
    return jnp.einsum("cij,cj->ci", block_inverse, per_cell).T.ravel()


@jax.jit
def dense_inverse(dense: jnp.ndarray) -> jnp.ndarray:
    """The coarsest level's dense inverse, by LU where the operator admits one, else its pseudo-inverse.

    The coarse solve must be an actual solve -- an inexact bottom solve leaves the smoothest error in and
    is the dominant cause of mesh-dependent V-cycle degradation -- but it need not be a *pseudo*-inverse.
    A pseudo-inverse is a singular value decomposition, which costs roughly an order of magnitude more
    than a factorization at the same size, and it is charged at every build and every refit. The
    generality is only needed for a **singular** operator, which the coarsening works to avoid (an empty
    aggregate is given a unit diagonal so no coarse row is structurally zero), so the factorization is
    taken and the decomposition kept as the fallback for the case it genuinely covers.

    **Singularity is judged by the reciprocal condition number in the one-norm, not by whether the
    factorization failed**: a factorization of a numerically singular matrix returns finite nonsense
    rather than failing. The threshold, ``n`` times machine epsilon, is the one the pseudo-inverse
    applies when it truncates a small singular value, so the two agree on where the operator stops being
    invertible. A non-finite inverse has no finite condition and so also takes the pseudo-inverse.

    Parameters
    ----------
    dense : jnp.ndarray
        The coarsest level operator, shape ``(n, n)``.

    Returns
    -------
    jnp.ndarray
        Its inverse (or pseudo-inverse), shape ``(n, n)``.
    """
    dense = jnp.asarray(dense)
    n = dense.shape[0]
    factors = jax.scipy.linalg.lu_factor(dense)
    inverse = jax.scipy.linalg.lu_solve(factors, jnp.eye(n, dtype=dense.dtype))
    reciprocal_condition = 1.0 / (jnp.linalg.norm(dense, 1) * jnp.linalg.norm(inverse, 1))
    invertible = reciprocal_condition > n * jnp.finfo(dense.dtype).eps
    return jax.lax.cond(invertible, lambda: inverse, lambda: jnp.linalg.pinv(dense))


def spectral_estimate(
    apply, start: jnp.ndarray, iterations: int = SPECTRAL_ITERATIONS
) -> jnp.ndarray:
    """The largest eigenvalue magnitude of the linear map ``apply``, by power iteration from ``start``.

    ``start`` must have unit norm. If an iterate maps to zero the estimate is ``1`` (the map has no
    spectrum to damp by), and it is held there.

    Parameters
    ----------
    apply : callable
        ``v -> M v``, traced.
    start : jnp.ndarray
        The unit starting vector, shape ``(n,)``.
    iterations : int
        Power-iteration steps.

    Returns
    -------
    jnp.ndarray
        The estimate, a 0-d array.
    """

    def step(_, carry):
        vector, estimate, stopped = carry
        image = apply(vector)
        norm = jnp.linalg.norm(image)
        vanished = norm == 0.0
        estimate = jnp.where(stopped, estimate, jnp.where(vanished, 1.0, norm))
        vector = jnp.where(stopped | vanished, vector, image / jnp.where(vanished, 1.0, norm))
        return vector, estimate, stopped | vanished

    initial = (start, jnp.ones((), dtype=start.dtype), jnp.zeros((), dtype=bool))
    return jax.lax.fori_loop(0, iterations, step, initial)[1]


@eqx.filter_jit
def _operator_spectral_estimate(operator: _PaddedRowOperator, start: jnp.ndarray) -> jnp.ndarray:
    """:func:`spectral_estimate` of a padded-row operator, compiled once per shape."""
    return spectral_estimate(operator.apply, start)


def unit_start(n: int) -> np.ndarray:
    """The power iteration's starting vector: a fixed random draw, normalized, shape ``(n,)``."""
    vector = np.random.default_rng(0).standard_normal(n)
    return vector / np.linalg.norm(vector)


def matrix_spectral_estimate(matrix: sp.spmatrix) -> float:
    """:func:`spectral_estimate` of a host sparse matrix, from :func:`unit_start` (off-jit callers)."""
    operator = _PaddedRowOperator.from_scipy(sp.csr_matrix(matrix))
    return float(_operator_spectral_estimate(operator, jnp.asarray(unit_start(matrix.shape[0]))))


class _LevelPlan(eqx.Module):
    """One level's structure, as the maps that turn its operator's values into everything it holds.

    Attributes
    ----------
    layout : _PaddedLayout
        The level operator's padded rows and their transpose.
    diagonal : DeviceEntryMap
        Each row's diagonal entry (zero where none is stored).
    cells : DeviceEntryMap or None
        Each cell's dense block, flat ``(n_cells, b, b)``; ``None`` on a scalar level.
    dense : DeviceEntryMap or None
        The whole operator as a dense ``(n, n)`` matrix, flat, on the coarsest level only.
    galerkin : ValueMap or None
        The map to the next level's values (``R A P`` for the level's frozen transfers); ``None`` on the
        coarsest level.
    embed : DeviceEntryMap or None
        How this level's values are placed from the previous level's Galerkin product, when the
        coarsening added entries to it (a padded aggregate's unit diagonal); ``None`` when the product
        is this level's operator as it stands.
    constant : jnp.ndarray or None
        The fixed values added after ``embed``, shape ``(nnz,)``.
    start : jnp.ndarray
        The unit starting vector of this level's spectral estimate, shape ``(n,)``.
    prolongation : tuple of jnp.ndarray or None
        ``(fine_row, coarse_column, value)`` of this level's frozen prolongation; ``None`` on the
        coarsest level.
    n, n_coarse, block_size : int
        Degrees of freedom here and on the next level, and per cell. Static.
    """

    layout: _PaddedLayout
    diagonal: DeviceEntryMap
    cells: DeviceEntryMap | None
    dense: DeviceEntryMap | None
    galerkin: ValueMap | None
    embed: DeviceEntryMap | None
    constant: jnp.ndarray | None
    start: jnp.ndarray
    prolongation: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None
    n: int = eqx.field(static=True)
    n_coarse: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)


class CoarseningPlan(eqx.Module):
    """An aggregation hierarchy's structure on the device: one :class:`_LevelPlan` per level, finest first.

    Attributes
    ----------
    levels : tuple of _LevelPlan
        The levels.
    scale_rows, scale_columns : jnp.ndarray or None
        The fine operator's row and column of each stored entry, shape ``(nnz,)``, held only for an
        equilibrated hierarchy, whose fine values are rescaled by ``scale[row] * scale[column]``.
    """

    levels: tuple[_LevelPlan, ...]
    scale_rows: jnp.ndarray | None = None
    scale_columns: jnp.ndarray | None = None

    @property
    def equilibrated(self) -> bool:
        """Whether the fine values are rescaled before they are coarsened."""
        return self.scale_rows is not None


class LevelValues(NamedTuple):
    """What one level holds that depends on its operator's values, as the fit returns it.

    Attributes
    ----------
    slot_values, transposed_slot_values : jnp.ndarray
        The level operator's padded-row values and those of its transpose, on the level's layout
        (:meth:`_PaddedLayout.with_values`). Only the values: an array a compiled function returns is a
        new buffer, so returning the whole operator would copy the layout's index arrays every fit.
    diagonal : jnp.ndarray
        Its diagonal, shape ``(n,)``.
    lam_max : jnp.ndarray
        The spectral estimate of ``D⁻¹ A``, 0-d.
    coarse_inv : jnp.ndarray or None
        The dense inverse, on the coarsest level only.
    block_inverse : jnp.ndarray or None
        Each cell block's inverse, ``(n_cells, b, b)``, on a nodal level only.
    """

    slot_values: jnp.ndarray
    transposed_slot_values: jnp.ndarray
    diagonal: jnp.ndarray
    lam_max: jnp.ndarray
    coarse_inv: jnp.ndarray | None
    block_inverse: jnp.ndarray | None


class LevelChecks(NamedTuple):
    """Per level, what a fit must refuse: counts the host reads back once and raises on.

    Attributes
    ----------
    bad_diagonal : jnp.ndarray
        On a scalar level, the count of non-finite or non-positive diagonal entries; ``0`` otherwise.
    singular_blocks : jnp.ndarray
        On a nodal level, the count of singular cell blocks; ``0`` otherwise.
    n_cells : int
        Cells on the level, for the refusal's message.
    """

    bad_diagonal: jnp.ndarray
    singular_blocks: jnp.ndarray
    n_cells: int


@eqx.filter_jit
def fine_diagonal(plan: CoarseningPlan, values: jnp.ndarray) -> jnp.ndarray:
    """The fine operator's diagonal, shape ``(n,)``, from its stored values."""
    return plan.levels[0].diagonal.apply(values)


@eqx.filter_jit
def fit_levels(
    plan: CoarseningPlan, values: jnp.ndarray, scale: jnp.ndarray | None = None
) -> tuple[tuple[LevelValues, ...], tuple[LevelChecks, ...]]:
    """Every level's values from the fine operator's stored ``values``, shape ``(nnz,)``, traced.

    ``scale`` is the equilibration factor of an equilibrated plan, ``diag(D)`` in ``D A D``; ``None``
    otherwise. The result is only valid once the returned checks are clear.
    """
    if scale is not None:
        values = values * scale[plan.scale_rows] * scale[plan.scale_columns]
    fitted, checks = [], []
    for level in plan.levels:
        if level.embed is not None:
            values = level.embed.apply(values)
        if level.constant is not None:
            values = values + level.constant
        operator = level.layout.packed(values)
        diagonal = level.diagonal.apply(values)
        n_cells = level.n // level.block_size
        if level.block_size == 1:
            bad = ~jnp.isfinite(diagonal) | (diagonal <= 0.0)
            check = LevelChecks(
                bad_diagonal=jnp.sum(bad),
                singular_blocks=jnp.zeros((), dtype=jnp.int32),
                n_cells=n_cells,
            )
            block_inverse = None
            inverse_diagonal = 1.0 / diagonal

            def preconditioned(vector, operator=operator, inverse_diagonal=inverse_diagonal):
                return inverse_diagonal * operator.apply(vector)
        else:
            block = level.block_size
            blocks = level.cells.apply(values).reshape(n_cells, block, block)
            block_inverse, singular = invert_cell_blocks(blocks)
            check = LevelChecks(
                bad_diagonal=jnp.zeros((), dtype=jnp.int32),
                singular_blocks=singular,
                n_cells=n_cells,
            )

            def preconditioned(vector, operator=operator, inverse=block_inverse, block=block):
                return apply_block_inverse(inverse, block, operator.apply(vector))

        coarse_inv = None
        if level.dense is not None:
            coarse_inv = dense_inverse(level.dense.apply(values).reshape(level.n, level.n))
        fitted.append(
            LevelValues(
                slot_values=operator.values,
                transposed_slot_values=operator.t_values,
                diagonal=diagonal,
                lam_max=spectral_estimate(preconditioned, level.start),
                coarse_inv=coarse_inv,
                block_inverse=block_inverse,
            )
        )
        checks.append(check)
        if level.galerkin is not None:
            values = level.galerkin.apply(values)
    return tuple(fitted), tuple(checks)


class _Landing(NamedTuple):
    """A level's operator pattern, and how its values follow from the previous level's product."""

    pattern: SparsePattern
    embed: DeviceEntryMap | None
    constant: jnp.ndarray | None


def level_plan(
    pattern: SparsePattern,
    block_size: int,
    *,
    prolongation: sp.coo_matrix | None,
    width: int | None = None,
    transpose_width: int | None = None,
    embed: DeviceEntryMap | None = None,
    constant: np.ndarray | None = None,
) -> tuple[_LevelPlan, SparsePattern | None]:
    """The plan of one level whose operator stores ``pattern``, and the pattern of the next level's product.

    Parameters
    ----------
    pattern : SparsePattern
        The level operator's pattern.
    block_size : int
        Degrees of freedom per cell.
    prolongation : scipy.sparse.coo_matrix or None
        The frozen prolongation to the next level; ``None`` makes this the coarsest level, solved by a
        dense inverse.
    width, transpose_width : int, optional
        The padded-row budgets (:meth:`_PaddedLayout.of`).
    embed, constant : optional
        How this level's values follow from the previous level's product (:class:`_LevelPlan`).

    Returns
    -------
    plan : _LevelPlan
        The level's plan.
    landing : SparsePattern or None
        The pattern of ``Pᵀ A P``, which the next level's operator starts from; ``None`` on the
        coarsest level.
    """
    n = pattern.shape[0]
    cells = None
    if block_size > 1:
        cells = EntryMap.gathering(pattern, lambda m: cell_blocks(m, block_size)).on_device()
    galerkin = dense = landing = prolongation_arrays = None
    n_coarse = 0
    if prolongation is None:
        dense = EntryMap.gathering(pattern, lambda m: m.toarray()).on_device()
    else:
        transfer = sp.csr_matrix(prolongation)
        galerkin, landing = _galerkin(pattern, block_size, transfer)
        n_coarse = transfer.shape[1]
        prolongation_arrays = (
            jnp.asarray(prolongation.row),
            jnp.asarray(prolongation.col),
            jnp.asarray(prolongation.data),
        )
    plan = _LevelPlan(
        layout=_PaddedLayout.of(pattern, width, transpose_width, fields=(block_size, block_size)),
        diagonal=EntryMap.gathering(pattern, lambda m: m.diagonal()).on_device(),
        cells=cells,
        dense=dense,
        galerkin=galerkin,
        embed=embed,
        constant=None if constant is None else jnp.asarray(constant),
        start=jnp.asarray(unit_start(n)),
        prolongation=prolongation_arrays,
        n=n,
        n_coarse=n_coarse,
        block_size=block_size,
    )
    return plan, landing


def _galerkin(
    pattern: SparsePattern, block_size: int, transfer: sp.csr_matrix
) -> tuple[ValueMap, SparsePattern]:
    """The map ``A -> Pᵀ A P`` onto the product's structural pattern, and that pattern.

    At cell level (:class:`~aquaflux.precondition.kronecker_pattern.KroneckerGalerkin`) when the level
    stores one cell pattern over every field pair and ``P`` carries each field on its own coarse unknown
    through one cell transfer -- a nodal aggregation's tentative prolongation -- else over every term.
    """
    kronecker = KroneckerPattern.of_dof(pattern, block_size, block_size)
    cell_transfer = None if kronecker is None else nodal_cell_transfer(transfer, block_size)
    if cell_transfer is not None:
        galerkin, coarse = KroneckerGalerkin.of(kronecker, cell_transfer)
        return galerkin, coarse.dof_pattern()
    product = EntryMap.product(sp.csr_matrix(transfer.T), pattern, transfer)
    return product.on_device(), product.landing


def landing_with_diagonal(landing: SparsePattern, slots: np.ndarray) -> _Landing:
    """A product's pattern with a unit diagonal added at ``slots``, and how its values are placed there.

    The coarsening gives each structurally empty coarse slot a unit diagonal so no coarse row is zero.
    The added entries are summed into the pattern as a sparse sum does: one already stored keeps its
    place and gains one, a new one is inserted in column order. Returned as the new pattern, the gather
    placing the product's values into it, and the constant ones.
    """
    if slots.size == 0:
        return _Landing(landing, None, None)

    def with_diagonal(values: np.ndarray, added: float) -> sp.csr_matrix:
        entries = landing.csr(values).tocoo()
        return sp.coo_matrix(
            (
                np.concatenate([entries.data, np.full(slots.size, added)]),
                (np.concatenate([entries.row, slots]), np.concatenate([entries.col, slots])),
            ),
            shape=landing.shape,
        ).tocsr()

    numbered = with_diagonal(np.arange(1, landing.nnz + 1, dtype=np.float64), 0.0)
    constant = with_diagonal(np.zeros(landing.nnz), 1.0)
    pattern = SparsePattern.of(numbered)
    embed = EntryMap.placing(landing, numbered.data).on_device()
    return _Landing(pattern, embed, constant.data)
