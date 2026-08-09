"""A block-triangular field-split preconditioner for the coupled flow-plus-transport Newton solve.

The monolithic preconditioners in this package (:mod:`~aquaflux.solve.amg_preconditioner`,
:mod:`~aquaflux.solve.ilut_preconditioner`, :mod:`~aquaflux.solve.lu_preconditioner`) treat the coupled
Jacobian as one undifferentiated block. That is the right default, but it forces every field to share a
single multigrid hierarchy and a single level smoother — and the six fields of a Reynolds-averaged solve
are not one kind of equation. Four of them, ``[u, v, w, p]``, form a pressure-velocity saddle; the other
two, ``k`` and ``omega``, are advection-dominated transported scalars, one of them customarily solved in
a logarithmic variable. A method tuned for the saddle is not thereby tuned for the scalars.

This splits the degrees of freedom into a **leading** and a **trailing** group of whole fields, gives each
its own approximate inverse, and retains **one triangle** of the cross-coupling between them::

    M = [[A_l,  0 ],        M^-1 r  =  y_l = A_l^-1 r_l
         [ C , A_t]]                   y_t = A_t^-1 (r_t - C y_l)

with ``C`` the true off-diagonal block of the operator, taken from the assembled Jacobian rather than
modelled. Two properties make this usable where a general composite preconditioner would not be:

* **It is a fixed linear operator.** One application of each block inverse and one sparse product — no
  inner Krylov iteration, nothing state-dependent. An outer GMRES may therefore use it without going
  flexible.
* **It is transposable in closed form**, which the implicitly-differentiated adjoint requires. The
  transpose of a block-lower-triangular inverse is the block-upper-triangular one built from the
  transposed blocks, so :meth:`BlockTriangularFieldSplit.apply` serves the adjoint's transpose solve by
  reversing the order of the two block solves and using ``C^T``.

Dropping ``C`` entirely — a block-*diagonal* split — is a different and weaker object: it discards the
coupling rather than ordering it, and on this operator the coupling is load-bearing. Retaining a triangle
costs one extra sparse product per application and keeps half the cross-coupling exactly.

The operator being preconditioned stays monolithic throughout, so the automatically-differentiated
Jacobian and the coupled adjoint are untouched; only the preconditioner is split.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from .amg_preconditioner import MonolithicAmgPreconditioner, build_amg_vcycle
from .ilut_preconditioner import equilibrate_cell_major
from .refresh_timing import PhaseTimer

__all__ = [
    "BlockTriangularFieldSplit",
    "FieldGroups",
    "FieldSplitAmgPreconditioner",
    "build_block_triangular_field_split",
]


@dataclasses.dataclass(frozen=True)
class FieldGroups:
    """A field-major degree-of-freedom partition into two contiguous groups of whole fields.

    The coupled state is stored **field-major**: degree of freedom ``(cell i, field f)`` sits at
    ``f * n_cells + i``. A partition that splits on a *field* boundary is therefore a partition into two
    contiguous ranges, which is what makes a field split cheap here — vectors are sliced rather than
    gathered, and the operator's four blocks are contiguous submatrices.

    This object owns that arithmetic so no consumer re-derives ``f * n_cells + i`` inline. It carries no
    matrix and no vector, only the shape of the partition.

    Attributes
    ----------
    n_cells : int
        Cells in the mesh.
    n_leading_fields : int
        Fields in the leading group, taken from the start of the field order.
    n_trailing_fields : int
        Fields in the trailing group, immediately following the leading one.

    Raises
    ------
    ValueError
        If either group is empty or a count is negative — a "split" with an empty side is the monolithic
        preconditioner wearing a disguise, and silently accepting it would report a field-split result
        that was never a field split.
    """

    n_cells: int
    n_leading_fields: int
    n_trailing_fields: int

    def __post_init__(self) -> None:
        if self.n_cells <= 0:
            raise ValueError(f"n_cells must be positive, got {self.n_cells}")
        if self.n_leading_fields <= 0 or self.n_trailing_fields <= 0:
            raise ValueError(
                "both groups must hold at least one field, got "
                f"{self.n_leading_fields} leading and {self.n_trailing_fields} trailing; a split with "
                "an empty side is not a split."
            )

    @property
    def n_fields(self) -> int:
        """Total fields per cell."""
        return self.n_leading_fields + self.n_trailing_fields

    @property
    def n_dofs(self) -> int:
        """Total degrees of freedom."""
        return self.n_fields * self.n_cells

    @property
    def n_leading_dofs(self) -> int:
        """Degrees of freedom in the leading group."""
        return self.n_leading_fields * self.n_cells

    @property
    def leading(self) -> slice:
        """The leading group's degrees of freedom, as a slice into a field-major vector."""
        return slice(0, self.n_leading_dofs)

    @property
    def trailing(self) -> slice:
        """The trailing group's degrees of freedom, as a slice into a field-major vector."""
        return slice(self.n_leading_dofs, self.n_dofs)

    def blocks(
        self, matrix: sp.spmatrix
    ) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """The operator's four blocks under this partition.

        Parameters
        ----------
        matrix : scipy.sparse matrix
            The assembled field-major operator, shape ``(n_dofs, n_dofs)``.

        Returns
        -------
        tuple of scipy.sparse.csr_matrix
            ``(A_ll, A_lt, A_tl, A_tt)`` — leading-leading, leading-trailing (the coupling *into* the
            leading equations), trailing-leading, and trailing-trailing.

        Raises
        ------
        ValueError
            If the matrix does not have this partition's shape.
        """
        matrix = sp.csr_matrix(matrix)
        if matrix.shape != (self.n_dofs, self.n_dofs):
            raise ValueError(
                f"matrix is {matrix.shape}, but this partition describes "
                f"{(self.n_dofs, self.n_dofs)} ({self.n_fields} fields over {self.n_cells} cells)."
            )
        lead, trail = self.leading, self.trailing
        return (
            matrix[lead, :][:, lead],
            matrix[lead, :][:, trail],
            matrix[trail, :][:, lead],
            matrix[trail, :][:, trail],
        )


class BlockTriangularFieldSplit:
    """A block-triangular approximate inverse over a two-group field partition.

    A pure host object (numpy/scipy plus whatever the block inverses are), with the same
    ``apply(residual, transpose=...)`` interface as :class:`~aquaflux.solve.AmgVCycle`, so it is a drop-in
    wherever a frozen approximate inverse of the coupled operator is wanted.

    One application solves the leading group, corrects the trailing group's right-hand side by the
    retained coupling, and solves the trailing group::

        y_l = M_l r_l
        y_t = M_t (r_t - C y_l)

    The transpose reverses both the order and the blocks, which is exactly the transpose of the above and
    is therefore available in closed form rather than by a numerical transpose::

        y_t = M_t^T r_t
        y_l = M_l^T (r_l - C^T y_t)

    Which group leads is the caller's choice and it is a real one: leading with the flow retains
    ``d R_turbulence / d flow`` (the production terms' dependence on the velocity gradient), leading with
    the turbulence retains ``d R_flow / d turbulence`` (the momentum equations' dependence on the eddy
    viscosity). :func:`build_block_triangular_field_split` names both.

    Parameters
    ----------
    leading, trailing : object
        The two block inverses, each exposing ``apply(residual, *, transpose=False) -> np.ndarray`` over
        its own group's degrees of freedom. :class:`~aquaflux.solve.AmgVCycle` satisfies this.
    coupling : scipy.sparse matrix
        The retained off-diagonal block, mapping the **leading** group's degrees of freedom to the
        **trailing** group's equations, shape ``(n_trailing_dofs, n_leading_dofs)``. Taken from the
        assembled operator, not modelled.
    groups : FieldGroups
        The partition the three arguments above are consistent with.

    Raises
    ------
    ValueError
        If ``coupling`` does not have the shape the partition implies.
    """

    def __init__(
        self,
        leading: object,
        trailing: object,
        coupling: sp.spmatrix,
        groups: FieldGroups,
    ) -> None:
        expected = (groups.n_dofs - groups.n_leading_dofs, groups.n_leading_dofs)
        if coupling.shape != expected:
            raise ValueError(
                f"coupling is {coupling.shape}, expected {expected} (trailing equations by leading "
                "unknowns). A block of the transposed orientation would apply silently and precondition "
                "the wrong triangle."
            )
        self._leading = leading
        self._trailing = trailing
        self._coupling = sp.csr_matrix(coupling)
        # The transpose is formed once at build rather than per apply: `A.T` on a CSR matrix yields a CSC
        # view whose product then converts on every call, which for a block of this size is a measurable
        # part of an application that is otherwise two multigrid cycles.
        self._coupling_transpose = sp.csr_matrix(self._coupling.transpose())
        self._groups = groups

    @property
    def groups(self) -> FieldGroups:
        """The field partition this preconditioner was built over."""
        return self._groups

    @property
    def n_dofs(self) -> int:
        """Degrees of freedom the preconditioner acts on."""
        return self._groups.n_dofs

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply the block-triangular inverse ``M`` (or its transpose) to a field-major residual.

        Parameters
        ----------
        residual : np.ndarray
            The field-major right-hand side, shape ``(n_dofs,)``.
        transpose : bool
            Apply ``M^T`` instead of ``M`` — the adjoint's transpose solve.

        Returns
        -------
        np.ndarray
            The preconditioned vector, shape ``(n_dofs,)``.
        """
        residual = np.asarray(residual, dtype=np.float64)
        lead, trail = self._groups.leading, self._groups.trailing
        out = np.empty_like(residual)
        if transpose:
            y_trailing = self._trailing.apply(residual[trail], transpose=True)
            y_leading = self._leading.apply(
                residual[lead] - self._coupling_transpose @ y_trailing, transpose=True
            )
        else:
            y_leading = self._leading.apply(residual[lead])
            y_trailing = self._trailing.apply(residual[trail] - self._coupling @ y_leading)
        out[lead] = y_leading
        out[trail] = y_trailing
        return out

    def _select_coupling(
        self, blocks: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
    ) -> sp.csr_matrix:
        """Which off-diagonal block this ordering retains, given :meth:`FieldGroups.blocks`' four.

        Solving the leading group first means correcting the trailing equations, so the retained block is
        trailing-by-leading. The other ordering overrides this rather than branching in :meth:`refactor`.
        """
        return blocks[2]

    def refactor(self, matrix: sp.spmatrix) -> None:
        """Re-fit both blocks and the retained coupling to a new operator, IN PLACE.

        The counterpart of :meth:`~aquaflux.solve.AmgVCycle.refactor` for a split, and used for the same
        reason: a march's mid-run refresh must re-preconditioner the **same** compiled Krylov solve, which
        means mutating this object rather than replacing it. Each block re-fits through its own
        ``refactor``, so each keeps its own aggregation and re-computes only the coarse operators and the
        smoother's factor values — the economy the monolithic refresh relies on, preserved per block.

        Parameters
        ----------
        matrix : scipy.sparse matrix
            The new assembled field-major operator, already shifted, of this partition's shape.

        Raises
        ------
        AttributeError
            If a block inverse cannot re-fit in place (an injected inverse need not provide ``refactor``).
        """
        blocks = self._groups.blocks(matrix)
        leading_block, trailing_block = blocks[0], blocks[3]
        for inverse, block, n_group_fields in (
            (self._leading, leading_block, self._groups.n_leading_fields),
            (self._trailing, trailing_block, self._groups.n_trailing_fields),
        ):
            if not hasattr(inverse, "refactor"):
                raise AttributeError(
                    f"{type(inverse).__name__} cannot refactor in place, so this split cannot be "
                    "refreshed mid-march; rebuild it instead, or inject an inverse that can."
                )
            inverse.refactor(*equilibrate_cell_major(block, n_group_fields))
        self._coupling = sp.csr_matrix(self._select_coupling(blocks))
        self._coupling_transpose = sp.csr_matrix(self._coupling.transpose())

    def destroy(self) -> None:
        """Release both block inverses' resources, if they hold any."""
        for block in (self._leading, self._trailing):
            release = getattr(block, "destroy", None)
            if release is not None:
                release()


def build_block_triangular_field_split(
    matrix: sp.spmatrix,
    groups: FieldGroups,
    *,
    flow_first: bool = True,
    smoother_fill_levels: int = 0,
    smoother_sweeps: int = 4,
    coarse_eq_limit: int | None = 2000,
    leading_options: dict | None = None,
    trailing_options: dict | None = None,
    leading_inverse: Callable[[sp.csr_matrix, int], object] | None = None,
    trailing_inverse: Callable[[sp.csr_matrix, int], object] | None = None,
) -> BlockTriangularFieldSplit:
    """Build a block-triangular field split with a multigrid V-cycle on each diagonal block.

    Each diagonal block gets its own :class:`~aquaflux.solve.AmgVCycle`, so each is equilibrated and
    reordered cell-major within its own group and aggregated at its own block size — the point of the
    exercise, since a four-field saddle and a two-field transport pair coarsen differently. The retained
    off-diagonal block is taken from ``matrix`` unmodified.

    Either block's inverse can be replaced wholesale by ``leading_inverse`` / ``trailing_inverse``, which
    is the seam for giving a group something other than a multigrid V-cycle over its sub-matrix — a
    reduction-based hierarchy for the transported scalars, say, or an inverse written in a framework that
    can run on an accelerator. Whatever is supplied need only expose the same ``n_dofs`` and
    ``apply(residual, *, transpose=...)`` an :class:`~aquaflux.solve.AmgVCycle` does.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled field-major operator, already shifted for the pseudo-transient step, shape
        ``(n_dofs, n_dofs)``.
    groups : FieldGroups
        The partition. Its leading group is the one listed first in the field order.
    flow_first : bool
        Solve the leading group first and correct the trailing group (retaining the trailing-by-leading
        coupling). ``False`` reverses both, retaining the leading-by-trailing coupling instead. The name
        reflects the usual field order, in which the flow fields lead.
    smoother_fill_levels, smoother_sweeps, coarse_eq_limit
        Passed to both blocks' V-cycles. The defaults are the bundle measured for the monolithic V-cycle
        on this operator: a zero-fill incomplete-LU smoother (fill produces negative pivots as the
        pseudo-transient shift falls), four sweeps of it, and a coarse grid large enough that its direct
        solve captures the global coupling.
    leading_options, trailing_options
        Extra multigrid options for one block only, so the two can be tuned apart. Ignored for a block
        whose inverse is supplied directly.
    leading_inverse, trailing_inverse
        ``(sub_matrix, n_fields_in_group) -> inverse`` replacing that block's V-cycle entirely. The
        returned object must expose ``n_dofs`` and ``apply(residual, *, transpose=...)``.

    Returns
    -------
    BlockTriangularFieldSplit
        The frozen preconditioner.
    """
    leading_block, leading_by_trailing, trailing_by_leading, trailing_block = groups.blocks(matrix)
    common = {
        "smoother_fill_levels": smoother_fill_levels,
        "smoother_sweeps": smoother_sweeps,
        "coarse_eq_limit": coarse_eq_limit,
    }
    leading = (
        leading_inverse(leading_block, groups.n_leading_fields)
        if leading_inverse is not None
        else build_amg_vcycle(
            leading_block, groups.n_leading_fields, extra_options=leading_options, **common
        )
    )
    trailing = (
        trailing_inverse(trailing_block, groups.n_trailing_fields)
        if trailing_inverse is not None
        else build_amg_vcycle(
            trailing_block, groups.n_trailing_fields, extra_options=trailing_options, **common
        )
    )
    if flow_first:
        return BlockTriangularFieldSplit(leading, trailing, trailing_by_leading, groups)
    # Trailing first: the roles swap, and so does the partition the split reports, since the group it
    # solves first is now the trailing one. The coupling is then leading-equations by trailing-unknowns.
    return _TrailingFirstFieldSplit(trailing, leading, leading_by_trailing, groups)


class _TrailingFirstFieldSplit(BlockTriangularFieldSplit):
    """The block-UPPER-triangular sibling: solve the trailing group first, correct the leading one.

    Same algebra with the two groups' roles exchanged. It is a separate class rather than a flag on
    :class:`BlockTriangularFieldSplit` because the alternative is a branch on ordering inside ``apply``,
    on a path that runs once per Krylov iteration.
    """

    def __init__(
        self,
        trailing: object,
        leading: object,
        coupling: sp.spmatrix,
        groups: FieldGroups,
    ) -> None:
        expected = (groups.n_leading_dofs, groups.n_dofs - groups.n_leading_dofs)
        if coupling.shape != expected:
            raise ValueError(
                f"coupling is {coupling.shape}, expected {expected} (leading equations by trailing "
                "unknowns)."
            )
        # Deliberately not calling the base __init__: its shape check describes the other orientation.
        self._leading = leading
        self._trailing = trailing
        self._coupling = sp.csr_matrix(coupling)
        self._coupling_transpose = sp.csr_matrix(self._coupling.transpose())
        self._groups = groups

    def _select_coupling(self, blocks):
        """Trailing-first retains the leading-equations-by-trailing-unknowns block instead."""
        return blocks[1]

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Apply ``M`` (or ``M^T``), solving the trailing group first.

        Parameters
        ----------
        residual : np.ndarray
            The field-major right-hand side, shape ``(n_dofs,)``.
        transpose : bool
            Apply ``M^T`` instead of ``M``.

        Returns
        -------
        np.ndarray
            The preconditioned vector, shape ``(n_dofs,)``.
        """
        residual = np.asarray(residual, dtype=np.float64)
        lead, trail = self._groups.leading, self._groups.trailing
        out = np.empty_like(residual)
        if transpose:
            y_leading = self._leading.apply(residual[lead], transpose=True)
            y_trailing = self._trailing.apply(
                residual[trail] - self._coupling_transpose @ y_leading, transpose=True
            )
        else:
            y_trailing = self._trailing.apply(residual[trail])
            y_leading = self._leading.apply(residual[lead] - self._coupling @ y_trailing)
        out[lead] = y_leading
        out[trail] = y_trailing
        return out


class FieldSplitAmgPreconditioner(MonolithicAmgPreconditioner):
    """The field split as JAX matvecs, with the same lifecycle as the monolithic V-cycle it replaces.

    A subclass rather than a sibling because everything outside the preconditioner's *construction* is
    genuinely shared: the coloured jvp probe that materializes the coupled Jacobian, the
    ``jax.pure_callback`` matvec that reads ``self.factors`` at call time (so an in-place refresh
    re-preconditions the same compiled solve), and the teardown. Only how the frozen inverse is fitted to
    the matrix differs, so only that is overridden — which is what lets a march swap between the two by
    changing one construction line and keep the shift policy, forward solver, step tail and refresh hooks
    common.

    The monolithic path equilibrates and reorders the **whole** matrix to cell-major before handing it to
    one V-cycle; a split does that **per block**, inside each block's own ``build_amg_vcycle``, because the
    two groups have different field counts and different scales. That is why the shift/equilibrate/reorder
    assembler the monolithic refresh precomputes has no counterpart here.

    .. warning::
       ``refresh_in_place`` is forward-march only, for the same reason as its base: the mutation is impure
       and would corrupt an adjoint transpose solve that read the inverse between its own calls.
    """

    def __init__(
        self,
        split: BlockTriangularFieldSplit,
        groups: FieldGroups,
        jacobian_no_shift: sp.csr_matrix | None = None,
        n_fields: int | None = None,
    ) -> None:
        super().__init__(split, jacobian_no_shift=jacobian_no_shift, n_fields=n_fields)
        self._groups = groups

    @property
    def groups(self) -> FieldGroups:
        """The field partition the preconditioner is built over."""
        return self._groups

    @classmethod
    def build(
        cls,
        matvec: Callable,
        colouring,
        n_fields: int,
        shift_diagonal: np.ndarray,
        groups: FieldGroups,
        *,
        smoother_fill_levels: int = 0,
        smoother_sweeps: int = 4,
        coarse_eq_limit: int | None = 2000,
        leading_options: dict | None = None,
        trailing_options: dict | None = None,
        batched_matvec: Callable | None = None,
        probe_batch_size: int | None = None,
        structure: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> FieldSplitAmgPreconditioner:
        """Materialize the coupled Jacobian, shift it, and fit a split to it.

        Parameters
        ----------
        matvec, colouring, n_fields, batched_matvec, probe_batch_size, structure
            The coloured-probe materialization, exactly as the monolithic build takes them.
        shift_diagonal : np.ndarray
            The pseudo-transient shift ``beta d`` added to the diagonal, shape ``(n_dofs,)``.
        groups : FieldGroups
            The partition to split on.
        smoother_fill_levels, smoother_sweeps, coarse_eq_limit, leading_options, trailing_options
            Passed through to each block's V-cycle.

        Returns
        -------
        FieldSplitAmgPreconditioner
            The frozen preconditioner.
        """
        jacobian = cls._materialize_jacobian(
            matvec, colouring, n_fields, batched_matvec, probe_batch_size, structure
        )
        split = build_block_triangular_field_split(
            cls._shifted(jacobian, shift_diagonal),
            groups,
            smoother_fill_levels=smoother_fill_levels,
            smoother_sweeps=smoother_sweeps,
            coarse_eq_limit=coarse_eq_limit,
            leading_options=leading_options,
            trailing_options=trailing_options,
        )
        return cls(split, groups, jacobian_no_shift=jacobian, n_fields=n_fields)

    def refresh_in_place(
        self,
        matvec: Callable,
        colouring,
        n_fields: int,
        shift_diagonal: np.ndarray,
        *,
        smoother_fill_levels: int = 0,
        smoother_sweeps: int = 4,
        batched_matvec: Callable | None = None,
        probe_batch_size: int | None = None,
        structure: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Re-materialize at the developed state and re-fit both blocks IN PLACE.

        Returns the same ``("probe", s), ("assemble", s), ("refactor", s)`` breakdown the monolithic
        refresh reports, so a march log reads identically for either preconditioner. Here "assemble" is
        only the diagonal shift — the per-block equilibration is inside the refactor.
        """
        del smoother_fill_levels, smoother_sweeps  # the smoother config is fixed at build
        timer = PhaseTimer()
        self._jacobian_no_shift = self._materialize_jacobian(
            matvec, colouring, n_fields, batched_matvec, probe_batch_size, structure
        )
        timer.lap("probe")
        self._n_fields = n_fields
        shifted = self._shifted(self._jacobian_no_shift, shift_diagonal)
        timer.lap("assemble")
        self.factors.refactor(shifted)
        timer.lap("refactor")
        return timer.phases()

    def refresh_shift_in_place(self, shift_diagonal: np.ndarray) -> tuple[tuple[str, float], ...]:
        """Re-fit at a new shift REUSING the cached Jacobian — no re-materialization.

        The cheap branch of the refresh, for when only ``beta`` has moved. Raises if no Jacobian was
        cached, rather than silently rebuilding from nothing.
        """
        if self._jacobian_no_shift is None:
            raise RuntimeError(
                "refresh_shift_in_place needs the Jacobian cached by build/refresh_in_place."
            )
        timer = PhaseTimer()
        shifted = self._shifted(self._jacobian_no_shift, shift_diagonal)
        timer.lap("assemble")
        self.factors.refactor(shifted)
        timer.lap("refactor")
        return timer.phases()
