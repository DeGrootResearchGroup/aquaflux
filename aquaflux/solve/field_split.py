"""A block-triangular field-split preconditioner for the coupled flow-plus-transport Newton solve.

The monolithic preconditioners in this package (:mod:`~aquaflux.solve.amg_preconditioner`,
:mod:`~aquaflux.solve.lu_preconditioner`) treat the coupled
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

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from .sparse_jacobian import ProbeGather

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .amg_preconditioner import MonolithicAmgPreconditioner, build_amg_vcycle
from .frozen_operator import equilibrate_cell_major
from .multigrid import SmoothedHierarchy, convection_multigrid_solve
from .native_inverse import NativeHierarchyInverse
from .refresh_timing import PhaseTimer

__all__ = [
    "BlockTriangularFieldSplit",
    "FieldGroups",
    "FieldSplitAmgPreconditioner",
    "NodalNativeInverse",
    "build_block_triangular_field_split",
    "native_nodal_inverse",
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
        self._set_coupling(coupling)
        self._groups = groups
        self._set_order(first="leading")

    def _set_order(self, *, first: str) -> None:
        """Fix which group is solved first, once, so ``apply`` never branches on the ordering.

        The two orderings are the same algebra with the groups exchanged, and which one an instance is
        cannot change after construction -- so resolving it here leaves ``apply`` a single body reading
        a pair of ``(inverse, degrees-of-freedom)`` records. That answers the objection the two-class
        split was originally made to avoid -- a branch on ordering on a path that runs once per Krylov
        iteration -- without paying for the body twice in source.
        """
        lead = (self._leading, self._groups.leading)
        trail = (self._trailing, self._groups.trailing)
        self._order = (lead, trail) if first == "leading" else (trail, lead)

    def _set_coupling(self, coupling: sp.spmatrix) -> None:
        """Store the retained coupling block, discarding any transpose cached for the previous one."""
        self._coupling = sp.csr_matrix(coupling)
        self._coupling_transpose: sp.csr_matrix | None = None

    @property
    def _transposed_coupling(self) -> sp.csr_matrix:
        """The coupling block transposed, formed on first use and then cached.

        It must not be re-derived per application: ``A.T`` on a compressed-sparse-row matrix yields a
        compressed-sparse-column view whose product converts on every call, which for a block of this
        size is a measurable part of an application that is otherwise two multigrid cycles. Nor should
        it be formed before anything asks for it, which is what the caching here buys. Only the
        transpose apply reads it — the adjoint's transpose solve — so a forward march would otherwise
        carry a second full copy of the coupling block, rebuilt at every refresh, and never touch it.
        """
        if self._coupling_transpose is None:
            self._coupling_transpose = sp.csr_matrix(self._coupling.transpose())
        return self._coupling_transpose

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
        # One body for both orderings and both directions. The solve order is fixed at construction
        # (`_set_order`); transposing a block-triangular inverse reverses it and uses the transposed
        # coupling, which is the whole of the difference between the four cases this used to spell out.
        (first, first_dofs), (second, second_dofs) = (
            reversed(self._order) if transpose else self._order
        )
        coupling = self._transposed_coupling if transpose else self._coupling
        out = np.empty_like(residual)
        y_first = first.apply(residual[first_dofs], transpose=transpose)
        y_second = second.apply(residual[second_dofs] - coupling @ y_first, transpose=transpose)
        out[first_dofs] = y_first
        out[second_dofs] = y_second
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

        An inverse may take the new operator in either of **two forms**, and the distinction is real
        rather than two spellings of one thing. A host solver wants it already put into the shape it
        factors — equilibrated and reordered cell-major — so it re-fits without redoing that work, and
        takes ``refactor(cell_major, scale, perm)``. A hierarchy built on the raw field-major block
        cannot use that shape at all: a nodal coarsening recovers each cell as ``index % n_cells``,
        which only holds field-major. Such an inverse takes ``refactor_block(block)`` instead.

        Parameters
        ----------
        matrix : scipy.sparse matrix
            The new assembled field-major operator, already shifted, of this partition's shape.

        Raises
        ------
        AttributeError
            If a block inverse offers neither (an injected inverse need not be refreshable at all).
        """
        blocks = self._groups.blocks(matrix)
        leading_block, trailing_block = blocks[0], blocks[3]
        for inverse, block, n_group_fields in (
            (self._leading, leading_block, self._groups.n_leading_fields),
            (self._trailing, trailing_block, self._groups.n_trailing_fields),
        ):
            if (refit := getattr(inverse, "refactor_block", None)) is not None:
                refit(block)
            elif hasattr(inverse, "refactor"):
                inverse.refactor(*equilibrate_cell_major(block, n_group_fields))
            else:
                raise AttributeError(
                    f"{type(inverse).__name__} cannot refactor in place, so this split cannot be "
                    "refreshed mid-march; rebuild it instead, or inject an inverse that can."
                )
        self._set_coupling(self._select_coupling(blocks))

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
    trailing_smoother_sweeps: int = 1,
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
    smoother_fill_levels, coarse_eq_limit
        Passed to both blocks' V-cycles. The defaults are the bundle measured for the monolithic V-cycle:
        a zero-fill incomplete-LU smoother (fill produces negative pivots as the pseudo-transient shift
        falls) and a coarse grid large enough that its direct solve captures the global coupling.
    smoother_sweeps : int
        Level-smoother sweeps on the **leading** block. Four is the measured default for a
        pressure-velocity saddle, where the sweeps are load-bearing: Jacobi-class smoothers do not
        converge on that block at all, so it is the half that needs the incomplete-LU work. Ignored
        when ``leading_inverse`` supplies that block's inverse directly.
    trailing_smoother_sweeps : int
        Level-smoother sweeps on the **trailing** block, defaulting to **one** rather than four, and
        ignored when ``trailing_inverse`` supplies that block's inverse directly. The two
        halves are not the same kind of equation and do not want the same amount of smoothing: the
        trailing group is a transported-scalar pair with a genuine diagonal, a far easier operator than
        the saddle, and the extra sweeps buy nothing on it. Measured on a three-dimensional
        backward-facing step at ``Re_h = 10000``, four sweeps against one over a whole
        Reynolds-continuation march: **1959 s against 1636 s (−16.5 %)**, with the two marches following
        the same trajectory step for step — same shift, same per-step restart-cycle counts, same
        residuals to four figures, same single line-search escalation — and reaching the same
        reattachment length. So the sweeps were pure cost there rather than a quality/cost trade. Raise
        it if a case shows the trailing block genuinely needing more; the knob is here because that is a
        per-case question, not a universal constant.

        Note the *cycle* count rose slightly (277 → 282) while the wall fell 16.5 %: a restart-cycle
        count is only a cost proxy between candidates that share a per-application price, and changing
        the smoother is exactly what breaks that.
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
        "coarse_eq_limit": coarse_eq_limit,
    }
    leading = (
        leading_inverse(leading_block, groups.n_leading_fields)
        if leading_inverse is not None
        else build_amg_vcycle(
            leading_block,
            groups.n_leading_fields,
            smoother_sweeps=smoother_sweeps,
            extra_options=leading_options,
            **common,
        )
    )
    trailing = (
        trailing_inverse(trailing_block, groups.n_trailing_fields)
        if trailing_inverse is not None
        else build_amg_vcycle(
            trailing_block,
            groups.n_trailing_fields,
            smoother_sweeps=trailing_smoother_sweeps,
            extra_options=trailing_options,
            **common,
        )
    )
    if flow_first:
        return BlockTriangularFieldSplit(leading, trailing, trailing_by_leading, groups)
    # Trailing first: the roles swap, and so does the partition the split reports, since the group it
    # solves first is now the trailing one. The coupling is then leading-equations by trailing-unknowns.
    return _TrailingFirstFieldSplit(trailing, leading, leading_by_trailing, groups)


class _TrailingFirstFieldSplit(BlockTriangularFieldSplit):
    """The block-UPPER-triangular sibling: solve the trailing group first, correct the leading one.

    Same algebra with the two groups' roles exchanged, so it supplies only what genuinely differs --
    which triangle of the operator it retains, and which group it solves first. ``apply`` is the base's,
    reading the order this constructor fixes -- the ordering cannot change after construction, so
    resolving it here keeps a single ``apply`` body rather than two mirrored ones.
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
        self._leading = leading
        self._trailing = trailing
        self._set_coupling(coupling)
        self._groups = groups
        self._set_order(first="trailing")

    def _select_coupling(self, blocks):
        """The leading-by-trailing block: this ordering retains the OTHER triangle."""
        return blocks[1]


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

    @property
    def has_native_solve(self) -> bool:
        """Always ``False``: a block-triangular split offers no native exact solve.

        The base answers this by asking its frozen inverse, which is sound when that inverse is a single
        :class:`~aquaflux.solve.AmgVCycle` and false here -- the split's is a
        :class:`BlockTriangularFieldSplit`, which has no such solve to offer, because a native solve
        inverts the whole shifted operator in one host call and a split deliberately never forms it.
        Without this override the base's attribute lookup **raises**, and the raise is then invisible:
        both callers ask through ``getattr(pc, "is_exact_native", False)`` -- the right spelling for the
        factorization preconditioners, which genuinely lack the attribute -- and a default swallows an
        ``AttributeError`` coming from inside a property body just as readily as a missing name. The
        answer it produced was accidentally the correct ``False``, which is why this went unnoticed.
        """
        return False

    @classmethod
    def build(
        cls,
        matvec: Callable,
        plan,
        shift_diagonal: np.ndarray,
        groups: FieldGroups,
        *,
        smoother_fill_levels: int = 0,
        smoother_sweeps: int = 4,
        trailing_smoother_sweeps: int = 1,
        coarse_eq_limit: int | None = 2000,
        leading_options: dict | None = None,
        trailing_options: dict | None = None,
        leading_inverse: Callable[[sp.csr_matrix, int], object] | None = None,
        trailing_inverse: Callable[[sp.csr_matrix, int], object] | None = None,
        batched_matvec: Callable | None = None,
        probe_batch_size: int | None = None,
        structure: ProbeGather | None = None,
    ) -> FieldSplitAmgPreconditioner:
        """Materialize the coupled Jacobian, shift it, and fit a split to it.

        Parameters
        ----------
        matvec, plan, batched_matvec, probe_batch_size, structure
            The coloured-probe materialization, exactly as the monolithic build takes them.
        shift_diagonal : np.ndarray
            The pseudo-transient shift ``beta d`` added to the diagonal, shape ``(n_dofs,)``.
        groups : FieldGroups
            The partition to split on.
        smoother_fill_levels, smoother_sweeps, trailing_smoother_sweeps, coarse_eq_limit
            Passed through to each block's V-cycle. ``smoother_sweeps`` is the leading (saddle) block's
            and ``trailing_smoother_sweeps`` the trailing (transported-scalar) block's; they differ by
            default because the two halves want different amounts of smoothing.
        leading_options, trailing_options
            Extra multigrid options for one block only, so the two can be tuned apart. Ignored for a
            block whose inverse is supplied directly.
        leading_inverse, trailing_inverse : callable or None
            ``(sub_matrix, n_fields_in_group) -> inverse`` replacing that block's V-cycle entirely — the
            seam for preconditioning a block with something that is not a host solver's V-cycle. When
            set, the corresponding smoother settings above do not apply to it. An injected inverse must
            offer ``refactor_block`` or ``refactor`` to survive a mid-march refresh.

        Returns
        -------
        FieldSplitAmgPreconditioner
            The frozen preconditioner.
        """
        jacobian = cls._materialize_jacobian(
            matvec, plan, batched_matvec, probe_batch_size, structure
        )
        split = build_block_triangular_field_split(
            cls._shifted(jacobian, shift_diagonal),
            groups,
            smoother_fill_levels=smoother_fill_levels,
            smoother_sweeps=smoother_sweeps,
            trailing_smoother_sweeps=trailing_smoother_sweeps,
            coarse_eq_limit=coarse_eq_limit,
            leading_options=leading_options,
            trailing_options=trailing_options,
            leading_inverse=leading_inverse,
            trailing_inverse=trailing_inverse,
        )
        return cls(split, groups, jacobian_no_shift=jacobian, n_fields=plan.n_fields)

    def refresh_in_place(
        self,
        matvec: Callable,
        plan,
        shift_diagonal: np.ndarray,
        *,
        smoother_fill_levels: int = 0,
        smoother_sweeps: int = 4,
        batched_matvec: Callable | None = None,
        probe_batch_size: int | None = None,
        structure: ProbeGather | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Re-materialize at the developed state and re-fit both blocks IN PLACE.

        Returns the same ``("probe", s), ("assemble", s), ("refactor", s)`` breakdown the monolithic
        refresh reports, so a march log reads identically for either preconditioner. Here "assemble" is
        only the diagonal shift — the per-block equilibration is inside the refactor.
        """
        del smoother_fill_levels, smoother_sweeps  # the smoother config is fixed at build
        timer = PhaseTimer()
        self._jacobian_no_shift = self._materialize_jacobian(
            matvec, plan, batched_matvec, probe_batch_size, structure
        )
        timer.lap("probe")
        self._n_fields = plan.n_fields
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


class _NodalSmoother(NamedTuple):
    """The nodal cycle's counts and relaxations -- everything about it that must be concrete.

    A plain tuple of Python numbers, so it is hashable and compares by value: it lands wholly on the
    static side of :func:`_native_nodal_cycle` and two builds at the same settings share one compiled
    cycle.
    """

    cycles: int
    sweeps: int
    omega: float
    spectral_damping: bool


@eqx.filter_jit
def _native_nodal_cycle(
    hierarchy: SmoothedHierarchy,
    extras: None,
    residual: jnp.ndarray,
    smoother: _NodalSmoother,
) -> jnp.ndarray:
    """``smoother.cycles`` V-cycles over ``hierarchy``, relaxed by a damped or block Jacobi sweep.

    Module-level and taking the hierarchy as an ARGUMENT, so a refresh at unchanged shapes swaps its
    values into the SAME compiled cycle. ``extras`` is the smoother-specific record the shared
    :class:`~aquaflux.solve.native_inverse.NativeHierarchyInverse` passes through; this family reads the
    levels alone, so it is always ``None`` and is accepted only to keep the two cycles one shape.
    """
    return convection_multigrid_solve(
        hierarchy,
        residual,
        cycles=smoother.cycles,
        sweeps=smoother.sweeps,
        omega=smoother.omega,
        spectral_damping=smoother.spectral_damping,
    )


class NodalNativeInverse(NativeHierarchyInverse):
    """A block inverse from ONE JAX-native hierarchy over the whole group, coarsening cells.

    Given a block size it coarsens **cells**, so one hierarchy spans the whole group and the cross-field
    coupling sits inside the operator being coarsened rather than being approximated away outside it.
    That is what makes it stronger than a per-field hierarchy composed block-triangularly, and why a
    measurement taken on such a pair does not transfer here.

    Two things have to change together and neither suffices alone — measured, both refused otherwise:
    the aggregation must coarsen cells, and the level smoother must invert each cell's dense block
    rather than the scalar diagonal. On a multi-field operator whose within-cell coupling exceeds its
    diagonal, a point smoother discards the dominant term and the sweep does not contract.

    Host in, host out: the field split is numpy and the hierarchy is JAX, so each
    application crosses the boundary. A production native split would keep the whole thing traced.

    Parameters
    ----------
    block, n_fields, strength_threshold, max_levels, max_coarse, frozen_coarsening, shape_headroom, report
        See :class:`~aquaflux.solve.native_inverse.NativeHierarchyInverse`, which owns the hierarchy,
        the in-place refresh and the host boundary. Only the smoother below belongs to this class.
    cycles : int
        V-cycles per application. Fixed, so ``b -> x`` stays a linear map — required by the
        non-flexible outer Krylov and by the transposed adjoint solve.
    sweeps : int
        Smoother sweeps per level.
    aggressive_levels : int
        Levels coarsened on the **squared** graph, starting from the finest. ``1`` (default) is the
        aggressive first level the defaults note below describes.
    prolongation_smoothing : str
        Which prolongator the hierarchy builds — ``"none"`` (default, the unsmoothed tentative
        prolongation), ``"symmetric-part"`` or ``"standard"``.
    spectral_damping : bool
        Scale the smoother's relaxation by the level's largest eigenvalue estimate. ``False``
        (default) is the undamped sweep the note below explains; it also selects ``omega``'s meaning,
        which is why the two travel together.
    equilibrate : bool
        Coarsen the operator rescaled to a unit-magnitude diagonal; see the note below.

    Notes
    -----
    **The defaults here are the settings measured to reproduce a PETSc GAMG V-cycle on this operator,
    and they are not the multigrid builder's own defaults.** Three of them together took the coupled
    turbulence block from 5 restart cycles to 2, matching an equivalently-configured GAMG on a coarse
    space of the same size: one aggressive (squared-graph) coarsening level, an unsmoothed tentative
    prolongation, and an **undamped** smoother. The last is the one that looks wrong and is not —
    ``D^-1 A`` has a unit diagonal, so scaling the relaxation by ``1 / lambda_max`` can only ever
    under-relax, and it was costing a factor of five in sweeps.

    **``equilibrate`` rescales each cell block to a unit-magnitude diagonal**, which leaves the per-cell
    block triangular with a determinant of exactly one, so the block solve cannot meet a singular block.
    It is **not** what keeps the build safe, though: the singularity test is a row-norm (Hadamard)
    determinant bound, which is invariant under rescaling any row or column, so it reaches the same
    verdict either way.

    **It is not a free choice on a marched solve, and better conditioning is not the deciding property.**
    Rescaling is close to a similarity transform on the Jacobi-preconditioned operator, so the smoother
    and the spectral estimates barely see it; what it does change is the coarse operator built by the
    fixed aggregate indicator, and so the corrections that come out. On a backward-facing-step Reynolds
    continuation, an otherwise-identical pair of marches differing only in this flag came out opposite --
    rescaled, the line search lost its step length on an intermediate rung and the march stalled with the
    residual frozen; unscaled, it converged every rung. Measure it on the case at hand rather than
    assuming the better-conditioned operator marches better.
    """

    def __init__(
        self,
        block: sp.spmatrix,
        n_fields: int,
        *,
        cycles: int = 1,
        sweeps: int = 4,
        max_coarse: int = 16,
        aggressive_levels: int = 1,
        prolongation_smoothing: str = "none",
        spectral_damping: bool = False,
        equilibrate: bool = True,
        **hierarchy_settings,
    ) -> None:
        self._aggressive_levels = aggressive_levels
        self._prolongation_smoothing = prolongation_smoothing
        self._equilibrate = equilibrate
        # The cycle's static half, built once: it never changes over this inverse's life, so a refresh
        # cannot move the compilation key through it.
        self._smoother = _NodalSmoother(
            cycles=cycles,
            sweeps=sweeps,
            # `omega` means different things either side of `spectral_damping`: a damping relative to
            # the level's `lambda_max`, or the absolute relaxation itself. Undamped is the measured
            # default and is why the pair travels together.
            omega=1.0 if not spectral_damping else 0.8,
            spectral_damping=spectral_damping,
        )
        super().__init__(block, n_fields, max_coarse=max_coarse, **hierarchy_settings)

    def build_settings(self) -> dict:
        """This family coarsens by a randomized maximal independent set, on the settings above."""
        return {
            "mis_aggregation": True,
            "aggressive_levels": self._aggressive_levels,
            "prolongation_smoothing": self._prolongation_smoothing,
            "equilibrate": self._equilibrate,
        }

    def smoother(self) -> _NodalSmoother:
        return self._smoother

    def cycle(self):
        return _native_nodal_cycle


def native_nodal_inverse(**settings) -> Callable[[sp.spmatrix, int], object]:
    """A ``leading_inverse``/``trailing_inverse`` factory using :class:`NodalNativeInverse`.

    Every keyword is forwarded, so the defaults — and the reasoning behind them — live on the class
    rather than being restated here. ``max_coarse`` is worth knowing about: it is the coarse-grid size
    the hierarchy stops at and solves directly, and a coarse grid large enough to invert the global
    coupling exactly is measured to be worth a great deal on this operator.
    """

    def build(block: sp.spmatrix, n_group_fields: int) -> object:
        return NodalNativeInverse(block, n_group_fields, **settings)

    return build
