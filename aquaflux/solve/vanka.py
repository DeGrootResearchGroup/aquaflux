"""A Vanka (patch) smoother for the monolithic coupled saddle-point V-cycle.

The level smoother of an aggregation multigrid is normally a pointwise relaxation — an incomplete
factorization, or Gauss-Seidel — which treats every unknown alike. On a saddle-point system that is
the wrong shape: the continuity row of a cell has **no diagonal of its own** to relax against (the
pressure enters the momentum rows, not its own equation, except through whatever damping the
discretization adds), so a pointwise smoother has nothing to push the pressure error with. The
patch smoother of Vanka (1986) fixes this by relaxing *groups* of unknowns simultaneously: a small
patch is chosen around each pressure unknown, the tiny dense subsystem on that patch is solved
**exactly**, and the corrections are recombined. Each patch solve sees the local velocity–pressure
coupling in full, so the pressure error has somewhere to go.

This module builds that smoother **algebraically**, from the assembled matrix alone: it holds no
mesh and no ``jax``, only ``numpy`` and ``scipy.sparse``, so it is testable on a hand-written
three-cell matrix. Two pieces, separately testable:

* a **patch builder** (:class:`CellStarPatches`) deciding *which* unknowns form each patch, and
* the **smoother** (:class:`VankaSmoother`) that factors and applies them.

Two deliberate choices, both departures from the textbook form, both made for reasons this operator
forces:

**The recombination is additive, weighted by overlap.** Every patch is solved against the same input
residual and the corrections are combined with a partition-of-unity weight — the reciprocal of how
many patches cover each unknown, split symmetrically across the restriction and the prolongation, as
in the algebraic patch smoother of Schöberl and Zulehner (2003). The weight is the load-bearing part:
an unweighted additive sum over-corrects every unknown that several patches share, and on an
indefinite operator that alone makes the smoother amplify rather than smooth. What this gives up is
that the *multiplicative* sweep — solving patches in sequence against a residual updated as it goes —
is the stronger relaxation and is the one Vanka (1986) describes; the additive form is weaker, and a
result measured with it should say so. It is also the only form that runs at a sensible cost here,
and the only one that keeps a single application a bounded, **fixed linear** operator — which is what
a V-cycle used by a non-flexible outer Krylov solve, and by an adjoint's transpose, requires. (An
inner Krylov acceleration of the smoother would break the same property.)

**The patch is truncated by coupling strength.** The classical patch is one pressure plus *every*
velocity its continuity row touches. That row is three cells wide on a collocated second-order
discretization with Rhie-Chow damping — some fifty cells — so the whole patch would be a dense solve
of several hundred unknowns per cell. :class:`CellStarPatches` keeps the strongest few instead.

:class:`VankaPC` exposes the smoother to PETSc's multigrid as a shell preconditioner, so it can be
selected for one level with runtime options and compared against the shipped incomplete-LU smoother
on the same hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

#: Target elements per chunk when the dense patch blocks are gathered. Bounds the transient
#: allocation to a few tens of megabytes rather than the whole ``(n_patches, width, width)`` stack,
#: which on a three-dimensional coupled mesh is a few hundred megabytes on its own.
_GATHER_CHUNK_ELEMENTS = 1 << 23


@dataclass(frozen=True)
class PatchSet:
    """Which degrees of freedom form each Vanka patch, as a padded rectangular index set.

    A patch is a set of **cells** crossed with a set of **fields**, which is what makes it storable
    as a rectangle: slot ``s`` of every patch holds the same ``(cell slot, field)`` pair, so one
    ``(n_patches, n_cell_slots)`` cell array plus two ``(width,)`` slot maps describe every patch.
    Patches whose centre cell has fewer neighbours than the rest — a boundary cell, or a cell in a
    matrix with a ragged pattern — pad their unused cell slots and mark them in ``cell_mask``; the
    smoother zeroes those rows and columns rather than special-casing the patch.

    Degrees of freedom are indexed **cell-major**: ``(cell i, field f)`` sits at ``i * n_fields + f``.

    Attributes
    ----------
    cells : np.ndarray
        The cells in each patch, shape ``(n_patches, n_cell_slots)``. Slot 0 is the centre cell.
    cell_mask : np.ndarray
        Whether each cell slot is real rather than padding, shape ``(n_patches, n_cell_slots)``.
    slot_cell : np.ndarray
        The cell slot each degree-of-freedom slot draws from, shape ``(width,)``.
    slot_field : np.ndarray
        The field each degree-of-freedom slot holds, shape ``(width,)``.
    n_fields : int
        Degrees of freedom per cell.
    n_cells : int
        Cells in the operator the patches were drawn from — which need not equal the patch count, so
        it is carried rather than inferred.
    """

    cells: np.ndarray
    cell_mask: np.ndarray
    slot_cell: np.ndarray
    slot_field: np.ndarray
    n_fields: int
    n_cells: int

    @property
    def n_patches(self) -> int:
        """Number of patches."""
        return int(self.cells.shape[0])

    @property
    def width(self) -> int:
        """Degrees of freedom per patch, including padded slots."""
        return int(self.slot_cell.shape[0])

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom the patches are drawn from."""
        return self.n_cells * self.n_fields

    def dofs(self) -> np.ndarray:
        """The degree-of-freedom index of every slot, shape ``(n_patches, width)``.

        Padded slots carry a valid but arbitrary index (their padded cell's), so a gather never
        reads out of bounds; :meth:`mask` says which ones to believe.
        """
        return self.cells[:, self.slot_cell] * self.n_fields + self.slot_field

    def mask(self) -> np.ndarray:
        """Whether each slot is real rather than padding, shape ``(n_patches, width)``."""
        return self.cell_mask[:, self.slot_cell]

    def coverage(self) -> np.ndarray:
        """How many patches contain each degree of freedom, shape ``(n_dofs,)``.

        The averaging weight of the additive recombination is its reciprocal. A degree of freedom in
        no patch at all is left uncovered and reported as zero, which the smoother turns into a zero
        correction there rather than a division by zero.
        """
        mask = self.mask()
        return np.bincount(
            self.dofs()[mask].ravel(), minlength=self.n_dofs
        )  # padded slots excluded


@dataclass(frozen=True)
class CellStarPatches:
    """A patch per cell: the cell's own unknowns plus selected fields of its strongest neighbours.

    The classical Vanka patch on a staggered grid is one pressure together with the velocities on
    that cell's faces. On a collocated cell-centred grid every unknown lives at the cell centre, so
    the analogue is one cell's unknowns plus the velocities of the cells its continuity equation
    couples to — which is what this builds, choosing those neighbours **by coupling strength** rather
    than from the mesh.

    Strength is read off one row per cell (``centre_field``, the continuity row) of the assembled
    matrix: the coupling to a neighbouring cell is the sum of ``|a_ij|`` over that cell's columns, and
    the ``n_neighbours`` largest are kept. Selecting by strength rather than by mesh adjacency is what
    keeps this module free of the mesh, and it is also the only workable choice here: the coupled
    Jacobian's stencil reaches several cells out (Rhie-Chow damping couples pressure to the
    neighbour-of-neighbour ring), so "every cell in the row" would be a patch of a hundred cells.

    Parameters
    ----------
    n_neighbours : int
        How many neighbouring cells to draw into each patch. ``0`` gives a patch per cell block —
        exact block-Jacobi on the cell, with no cross-cell coupling.
    centre_field : int
        The field whose row measures coupling strength, and which the patch is centred on. For the
        coupled saddle this is the pressure (continuity) row.
    neighbour_fields : tuple of int or None
        Which fields of the *neighbouring* cells enter the patch. ``None`` (default) takes every
        field; the classical choice is the velocity components alone, which keeps the patch small and
        leaves each neighbour's pressure to that neighbour's own patch.
    """

    n_neighbours: int = 6
    centre_field: int = 0
    neighbour_fields: tuple[int, ...] | None = None

    def patches(self, matrix: sp.csr_matrix, n_fields: int) -> PatchSet:
        """Build the patch set for ``matrix``.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The assembled cell-major operator, shape ``(n_cells * n_fields,) * 2``.
        n_fields : int
            Degrees of freedom per cell.

        Returns
        -------
        PatchSet
            One patch per cell.

        Raises
        ------
        ValueError
            If the matrix size is not a multiple of ``n_fields``, if ``centre_field`` or any
            neighbour field is out of range, or if ``n_neighbours`` is negative.
        """
        n_dofs = int(matrix.shape[0])
        if n_dofs % n_fields != 0:
            raise ValueError(
                f"CellStarPatches: {n_dofs} degrees of freedom is not a multiple of "
                f"n_fields={n_fields}."
            )
        if not 0 <= self.centre_field < n_fields:
            raise ValueError(
                f"CellStarPatches: centre_field={self.centre_field} is outside [0, {n_fields})."
            )
        if self.n_neighbours < 0:
            raise ValueError(f"CellStarPatches: n_neighbours={self.n_neighbours} is negative.")
        fields = tuple(range(n_fields)) if self.neighbour_fields is None else self.neighbour_fields
        if any(not 0 <= f < n_fields for f in fields):
            raise ValueError(
                f"CellStarPatches: neighbour_fields={fields} is outside [0, {n_fields})."
            )
        n_cells = n_dofs // n_fields
        centre = np.arange(n_cells, dtype=np.int64)
        neighbours, found = _strongest_neighbours(
            matrix, n_fields, self.centre_field, fields, self.n_neighbours
        )
        cells = np.concatenate([centre[:, None], neighbours], axis=1)
        cell_mask = np.concatenate([np.ones((n_cells, 1), dtype=bool), found], axis=1)
        # The centre cell contributes every field; each neighbour slot contributes `fields`.
        slot_cell = np.concatenate(
            [
                np.zeros(n_fields, dtype=np.int64),
                np.repeat(np.arange(1, self.n_neighbours + 1), len(fields)),
            ]
        )
        slot_field = np.concatenate(
            [np.arange(n_fields), np.tile(np.asarray(fields, dtype=np.int64), self.n_neighbours)]
        ).astype(np.int64)
        return PatchSet(cells, cell_mask, slot_cell, slot_field, n_fields, n_cells)


def _strongest_neighbours(
    matrix: sp.csr_matrix, n_fields: int, centre_field: int, fields: tuple[int, ...], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` most strongly coupled neighbouring cells of every cell, and which slots are real.

    Reads one row per cell (``centre_field``) and collapses its columns onto cells, summing
    ``|a_ij|`` — but only over ``fields``, the fields a neighbour will actually contribute to the
    patch. Ranking a neighbour by couplings the patch is not going to resolve would pick the wrong
    cells: the continuity row of a collocated discretization carries a pressure-pressure damping
    term as well as its divergence entries, and it is the divergence entries the patch exists to
    invert. Restricting the measure to the patch's own fields makes the two agree by construction.

    A cell with fewer than ``k`` such couplings pads the remaining slots with itself — a valid index
    the caller's mask then discards.

    Returns
    -------
    neighbours : np.ndarray
        Cell indices, shape ``(n_cells, k)``.
    found : np.ndarray
        Whether each slot is a real neighbour rather than padding, shape ``(n_cells, k)``.
    """
    n_cells = matrix.shape[0] // n_fields
    rows = matrix[centre_field::n_fields].tocsr()
    keep = np.isin(rows.indices % n_fields, np.asarray(fields))
    strength = sp.coo_matrix(
        (
            np.abs(rows.data[keep]),
            (
                np.repeat(np.arange(n_cells), np.diff(rows.indptr))[keep],
                rows.indices[keep] // n_fields,
            ),
        ),
        shape=(n_cells, n_cells),
    ).tocsr()  # duplicate (cell, cell) entries sum, collapsing the kept fields onto one weight
    strength.setdiag(0.0)
    strength.eliminate_zeros()
    neighbours = np.repeat(np.arange(n_cells, dtype=np.int64)[:, None], max(k, 1), axis=1)[:, :k]
    neighbours = np.ascontiguousarray(neighbours)
    found = np.zeros((n_cells, k), dtype=bool)
    if k == 0:
        return neighbours, found
    indptr, indices, data = strength.indptr, strength.indices, strength.data
    for cell in range(n_cells):
        lo, hi = int(indptr[cell]), int(indptr[cell + 1])
        available = hi - lo
        if available == 0:
            continue
        take = min(k, available)
        candidates = indices[lo:hi]
        if take < available:
            # argpartition puts the `take` largest weights last; an exact fit needs no partition.
            candidates = candidates[np.argpartition(data[lo:hi], available - take)[-take:]]
        neighbours[cell, :take] = candidates
        found[cell, :take] = True
    return neighbours, found


class VankaSmoother:
    """An additive Vanka smoother: an exact solve on every patch, averaged over the overlaps.

    A pure host object (``numpy``/``scipy``, no ``jax``, no PETSc). Construction factors every patch
    once; :meth:`apply` is then a gather, a batched dense solve and a scatter, and is a **fixed linear
    operator** — the same input always gives the same output, so it may precondition a non-flexible
    Krylov solve and its :meth:`apply` transpose serves an adjoint.

    The operator applied is ``M = W^½ (Σ_p R_pᵀ A_pp⁻¹ R_p) W^½``, with ``R_p`` the restriction onto
    patch ``p`` and ``W`` the diagonal reciprocal of how many patches cover each unknown. Splitting
    the averaging symmetrically across the restriction and the prolongation, rather than applying it
    all on one side, is the normalization Schöberl and Zulehner (2003) use, and it keeps ``M``
    symmetric whenever the patch solves are. Either way the weights are a partition of unity: on an
    operator whose patches do not overlap, ``M`` is exactly the block inverse.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The assembled cell-major operator the smoother relaxes, shape ``(n_dofs, n_dofs)``.
    patches : PatchSet
        The patches, from a patch builder such as :class:`CellStarPatches`.
    multiplicative : bool
        Relax the patches in colour groups against a residual updated as the sweep proceeds, instead of
        solving them all against the same input and averaging — Gauss-Seidel rather than Jacobi, which
        on an indefinite operator is often the difference between converging and not. See
        :func:`colour_patches` for how this differs from a strictly sequential Vanka sweep. It carries
        **no overlap weighting**:
        the running residual update already accounts for what earlier patches did, so a partition of
        unity would double-count. Setup additionally colours the patches (:func:`colour_patches`), and
        one application costs a restricted matrix-vector product per colour rather than none at all.
    damping : float
        A scalar factor on the correction. ``1.0`` (default) is the plain averaged smoother; a
        smaller value under-relaxes it.
    max_patch_gain : float or None
        Drop any patch whose inverse has an entry larger than this, recomputing the overlap weights
        over the patches that remain (``None``, the default, keeps every patch).

        A near-singular patch contributes an enormous local correction, and additive recombination has
        no defence against it: one such patch in tens of thousands is enough to make the smoother
        amplify instead of smooth. Dropping it leaves its unknowns to the other patches covering them
        and to the coarse grid, which is a far smaller error than letting it dominate. It is also the
        experiment that turns "the worst patch has a large inverse" from a correlation into a cause —
        if capping the gain converges what otherwise diverges, the near-singular patches *are* the
        mechanism.

    Raises
    ------
    ValueError
        If ``matrix`` is not square, or its size disagrees with ``patches``.
    numpy.linalg.LinAlgError
        If a patch submatrix is singular — which on this operator means the patch was chosen badly
        (too small to see the local pressure coupling), not that the matrix is.
    """

    def __init__(
        self,
        matrix: sp.csr_matrix,
        patches: PatchSet,
        *,
        multiplicative: bool = False,
        damping: float = 1.0,
        max_patch_gain: float | None = None,
    ) -> None:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"VankaSmoother: the operator is not square ({matrix.shape}).")
        if matrix.shape[0] != patches.n_dofs:
            raise ValueError(
                f"VankaSmoother: the operator has {matrix.shape[0]} degrees of freedom but the "
                f"patches cover {patches.n_dofs}."
            )
        self._patches = patches
        self._damping = float(damping)
        self._dofs = patches.dofs()
        self._mask = patches.mask()
        self._inverse = _inverted_in_place(_patch_blocks(matrix, patches))
        self._dropped = 0
        if max_patch_gain is not None:
            over = np.abs(self._inverse).max(axis=(1, 2)) > max_patch_gain
            self._inverse[over] = 0.0
            self._mask = self._mask & ~over[:, None]  # excluded from the overlap count as well
            self._dropped = int(over.sum())
        # Coverage over the patches that actually contribute, so the weights stay a partition of unity.
        # A multiplicative sweep needs none of this -- its running residual update already accounts for
        # the earlier patches -- so its weights are all one.
        self._multiplicative = bool(multiplicative)
        if self._multiplicative:
            self._matrix = matrix.tocsr()
            colour, self._n_colours = colour_patches(patches)
            live = self._mask.any(axis=1)
            self._groups = [
                (
                    np.flatnonzero((colour == c) & live),
                    np.unique(self._dofs[(colour == c) & live][self._mask[(colour == c) & live]]),
                )
                for c in range(self._n_colours)
            ]
            self._half_weight = np.ones(patches.n_dofs)
        else:
            coverage = np.bincount(self._dofs[self._mask].ravel(), minlength=patches.n_dofs)
            self._half_weight = np.where(coverage > 0, 1.0 / np.sqrt(np.maximum(coverage, 1)), 0.0)

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom the smoother acts on."""
        return self._patches.n_dofs

    @property
    def n_colours(self) -> int:
        """Patch groups a multiplicative sweep visits in turn (``0`` for the additive smoother)."""
        return self._n_colours if self._multiplicative else 0

    @property
    def dropped_patches(self) -> int:
        """How many patches ``max_patch_gain`` excluded (``0`` when it was not set)."""
        return self._dropped

    @property
    def patch_width(self) -> int:
        """Degrees of freedom in each patch, including padded slots."""
        return self._patches.width

    @property
    def worst_patch_gain(self) -> float:
        """The largest ``max|A_p⁻¹|`` over the patches — how big a correction the worst patch can make.

        A patch that is nearly singular produces an enormous local correction, and an additive
        smoother then amplifies rather than smooths. That failure looks from the outside exactly like
        "the patch smoother does not work on this operator", when the cause is one badly chosen patch;
        reporting the number keeps the two apart, which for a smoother being evaluated is the whole
        question. Order unity to a few hundred is ordinary on a well-scaled operator.
        """
        return float(np.abs(self._inverse).max())

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply the smoother ``M`` (or its transpose ``M^T``) to a cell-major residual.

        Parameters
        ----------
        residual : np.ndarray
            The right-hand side, shape ``(n_dofs,)``.
        transpose : bool
            Apply ``M^T`` instead of ``M``. The symmetric weighting means only the patch inverses
            transpose; the restriction and prolongation are already each other's transpose.

        Returns
        -------
        np.ndarray
            The correction, shape ``(n_dofs,)``.
        """
        residual = np.asarray(residual, dtype=np.float64)
        if self._multiplicative:
            if transpose:
                raise NotImplementedError(
                    "The transpose of a multiplicative sweep is the reverse-ordered sweep with the "
                    "TRANSPOSED operator in its residual updates, which would mean holding a second "
                    "copy of a matrix this size. Use the additive smoother where an adjoint is needed."
                )
            out = self._multiplicative_sweep(residual)
        else:
            scaled = residual * self._half_weight
            local = scaled[self._dofs] * self._mask
            subscripts = "pji,pj->pi" if transpose else "pij,pj->pi"
            correction = np.einsum(subscripts, self._inverse, local) * self._mask
            out = np.bincount(self._dofs.ravel(), weights=correction.ravel(), minlength=self.n_dofs)
            out *= self._half_weight
        return out * self._damping if self._damping != 1.0 else out

    def _multiplicative_sweep(self, residual: np.ndarray) -> np.ndarray:
        """One Gauss-Seidel-style pass over the patch colours, accumulating the correction.

        Each colour sees the residual left by every colour before it. That is recomputed from the
        accumulated correction rather than maintained incrementally, ``r_c = b_c - (A delta)_c``, so the
        per-colour cost is a matrix-vector product over just the rows that colour touches -- roughly one
        full product per sweep in total, not one per colour.
        """
        total = np.zeros_like(residual)
        for patch_index, rows in self._groups:
            if patch_index.size == 0:
                continue
            local_rows = self._matrix[rows] @ total
            updated = np.zeros(self.n_dofs)
            updated[rows] = residual[rows] - local_rows
            dofs, mask = self._dofs[patch_index], self._mask[patch_index]
            local = updated[dofs] * mask
            correction = np.einsum("pij,pj->pi", self._inverse[patch_index], local) * mask
            # Patches within a colour are disjoint, so scattering cannot collide.
            np.add.at(total, dofs[mask], correction[mask])
        return total


def _patch_blocks(matrix: sp.csr_matrix, patches: PatchSet) -> np.ndarray:
    """Gather every patch's dense submatrix ``A[patch, patch]``, shape ``(n_patches, w, w)``.

    Works on the block form of the operator: a patch entry is a single element of the ``(n_fields,
    n_fields)`` block at ``(cell_a, cell_b)``, so locating one costs a search over the *block*
    sparsity (a factor ``n_fields²`` smaller than the scalar one) and the whole patch is then one
    fancy index. Entries whose block is absent from the pattern are zero.

    Padded slots are zeroed and given a unit diagonal, so the batched inverse of the stack is
    well-defined and reproduces the identity on them — which the mask then discards anyway.
    """
    n_fields = patches.n_fields
    block = matrix.tobsr(blocksize=(n_fields, n_fields))
    block.sort_indices()
    n_cells = matrix.shape[0] // n_fields
    block_rows = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(block.indptr))
    # One ascending key per stored block, so a (cell, cell) lookup is a single searchsorted.
    keys = block_rows * n_cells + block.indices
    del block_rows
    slot_cell, slot_field = patches.slot_cell, patches.slot_field
    width = patches.width
    row_cell, col_cell = slot_cell[:, None], slot_cell[None, :]
    row_field, col_field = slot_field[:, None], slot_field[None, :]
    diagonal = np.arange(width)
    mask = patches.mask()
    out = np.empty((patches.n_patches, width, width), dtype=np.float64)
    chunk = max(1, _GATHER_CHUNK_ELEMENTS // (width * width))
    for lo in range(0, patches.n_patches, chunk):
        hi = min(lo + chunk, patches.n_patches)
        cells = patches.cells[lo:hi].astype(np.int64)
        query = cells[:, :, None] * n_cells + cells[:, None, :]
        position = np.searchsorted(keys, query)
        np.clip(position, 0, keys.size - 1, out=position)
        present = keys[position] == query
        # Expand from (cell slot, cell slot) to (dof slot, dof slot) and pick the field entries.
        dense = np.where(
            present[:, row_cell, col_cell],
            block.data[position[:, row_cell, col_cell], row_field, col_field],
            0.0,
        )
        real = mask[lo:hi]
        dense *= real[:, :, None] & real[:, None, :]
        dense[:, diagonal, diagonal] = np.where(real, dense[:, diagonal, diagonal], 1.0)
        out[lo:hi] = dense
    return out


def colour_patches(patches: PatchSet) -> tuple[np.ndarray, int]:
    """Group patches so that no two in a group share a degree of freedom.

    A multiplicative sweep relaxes patches *in sequence*, each against a residual updated by the ones
    before it. Done literally that is one small dense solve at a time — tens of thousands of them per
    sweep, which no array language will run at a useful speed. Grouping patches that share no unknown
    lets a whole group go at once, with the residual refreshed only between groups: a handful of
    vectorized steps per sweep instead of a loop over patches.

    **This is the multi-coloured variant, and it is NOT bitwise the sequential sweep.** Sharing no
    unknown is weaker than being independent: patch ``q``'s residual depends on patch ``p``'s solution
    whenever the operator couples them, ``A[rows_q, dofs_p] != 0``, which happens between patches that
    share no degree of freedom at all. Exact equivalence would need colours independent in the
    *operator's* graph, and on a stencil that reaches three cells the conflict graph for that is tens of
    millions of pairs — unaffordable to build, let alone colour. So a group here is block-Jacobi within
    itself and Gauss-Seidel across groups, which is the form parallel implementations use, and which
    still carries most of the sequencing benefit over a fully additive smoother.

    Two patches conflict when they share a cell, so the conflict graph is ``P Pᵀ`` for the
    patch-by-cell incidence ``P``, and the grouping is a greedy graph colouring of it.

    Returns
    -------
    colour : np.ndarray
        The group each patch belongs to, shape ``(n_patches,)``.
    n_colours : int
        How many groups there are.
    """
    rows = np.repeat(np.arange(patches.n_patches), patches.cell_mask.sum(axis=1))
    incidence = sp.csr_matrix(
        (np.ones(rows.size), (rows, patches.cells[patches.cell_mask].ravel())),
        shape=(patches.n_patches, patches.n_cells),
    )
    conflict = (incidence @ incidence.T).tocsr()
    colour = np.full(patches.n_patches, -1, dtype=np.int32)
    indptr, indices = conflict.indptr, conflict.indices
    for patch in range(patches.n_patches):
        taken = {int(colour[n]) for n in indices[indptr[patch] : indptr[patch + 1]]}
        candidate = 0
        while candidate in taken:
            candidate += 1
        colour[patch] = candidate
    return colour, int(colour.max()) + 1


def _inverted_in_place(stack: np.ndarray) -> np.ndarray:
    """Invert every matrix in ``stack`` (shape ``(n, w, w)``), overwriting it.

    ``np.linalg.inv(stack)`` would hold the factored and the unfactored stack at once. On a
    three-dimensional mesh with wide patches each is a few hundred megabytes, and this smoother is
    built beside an already-materialized coupled Jacobian, so the second copy is worth avoiding.
    Inverting a chunk at a time bounds the extra allocation to one chunk.
    """
    chunk = max(1, _GATHER_CHUNK_ELEMENTS // max(stack.shape[1] * stack.shape[2], 1))
    for lo in range(0, stack.shape[0], chunk):
        hi = min(lo + chunk, stack.shape[0])
        stack[lo:hi] = np.linalg.inv(stack[lo:hi])
    return stack


class VankaPC:
    """PETSc shell-preconditioner context wrapping a :class:`VankaSmoother`.

    Selected through PETSc's runtime options, so it can replace the level smoother of an existing
    multigrid hierarchy without rebuilding it — for the finest level of a two-level hierarchy, for
    example::

        mg_levels_1_pc_type        python
        mg_levels_1_pc_python_type aquaflux.solve.vanka.VankaPC

    and configured with options carrying the same prefix as the preconditioner it is installed on:

    ``vanka_neighbours``
        Neighbouring cells per patch (default 6, a hexahedral cell's face neighbours).
    ``vanka_centre_field``
        The field whose row measures coupling strength and which the patch centres on — the pressure.
        Defaults to three fields from the end, which is where the coupled state layout
        ``[velocity components, p, k, omega]`` puts it.
    ``vanka_neighbour_fields``
        Which fields of the neighbouring cells join the patch. ``before_centre`` (default) takes the
        fields ahead of the centre one, which in that layout is exactly the velocity components — the
        classical Vanka choice, leaving each neighbour's pressure to its own patch. ``all`` takes
        every field.
    ``vanka_multiplicative``
        Relax the patches in sequence rather than additively -- the classical Vanka sweep (default off).
    ``vanka_damping``
        A scalar factor on the correction (default 1.0).
    ``vanka_max_patch_gain``
        Drop patches whose inverse exceeds this magnitude (default: keep them all).

    The block size of the level operator supplies the fields per cell, so the aggregation must have
    carried it down (it does for a matrix built with a block size set).
    """

    def __init__(self) -> None:
        self._smoother: VankaSmoother | None = None

    def setUp(self, pc) -> None:
        """Read the level operator and its options, and factor the patches."""
        from petsc4py import PETSc

        operator, _ = pc.getOperators()
        n_fields = operator.getBlockSize()
        indptr, indices, data = operator.getValuesCSR()
        matrix = sp.csr_matrix((data, indices, indptr), shape=operator.getSize())
        options = PETSc.Options(pc.getOptionsPrefix() or "")
        which = options.getString("vanka_neighbour_fields", "before_centre")
        if which not in ("before_centre", "all"):
            raise ValueError(
                f"VankaPC: vanka_neighbour_fields must be 'before_centre' or 'all', not {which!r}."
            )
        # The coupled state is [velocity components, p, k, omega], so the pressure sits three fields
        # from the end. A block too small to hold that layout would make the guess silently wrong.
        if (centre := options.getInt("vanka_centre_field", n_fields - 3)) < 1:
            raise ValueError(
                f"VankaPC: a block size of {n_fields} leaves no field ahead of the centre one "
                f"(centre_field={centre}); set vanka_centre_field for this operator's layout."
            )
        self._smoother = VankaSmoother(
            matrix,
            CellStarPatches(
                n_neighbours=options.getInt("vanka_neighbours", 6),
                centre_field=centre,
                neighbour_fields=None if which == "all" else tuple(range(centre)),
            ).patches(matrix, n_fields),
            multiplicative=options.getBool("vanka_multiplicative", False),
            damping=options.getReal("vanka_damping", 1.0),
            max_patch_gain=(
                cap if (cap := options.getReal("vanka_max_patch_gain", 0.0)) > 0.0 else None
            ),
        )
        # Printed rather than kept, so that a diverging arm can be told apart from a near-singular
        # patch without re-running it — the two look identical from the outside.
        print(
            f"    [vanka] {self._smoother.n_dofs} dofs, patch width "
            f"{self._smoother.patch_width}, worst |A_p^-1| "
            f"{self._smoother.worst_patch_gain:.3e}, dropped {self._smoother.dropped_patches}"
            f"{f', {self._smoother.n_colours} colours' if self._smoother.n_colours else ''}",
            flush=True,
        )

    def apply(self, pc, x, y) -> None:
        """``y = M x``."""
        y.setArray(self._smoother.apply(np.array(x.array_r)))

    def applyTranspose(self, pc, x, y) -> None:
        """``y = M^T x``."""
        y.setArray(self._smoother.apply(np.array(x.array_r), transpose=True))
