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
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .amg_preconditioner import MaterializedJacobianPreconditioner
from .hierarchy_inverse import HierarchyBlockInverse
from .multigrid import (
    SmoothedHierarchy,
    air_multigrid_solve,
    build_air_hierarchy,
    convection_multigrid_solve,
    refresh_air_hierarchy,
)
from .refresh_timing import PhaseTimer
from .state import FieldLayout

__all__ = [
    "BlockTriangularFieldSplit",
    "FieldGroups",
    "FieldSplitAmgPreconditioner",
    "JacobiSmoothedInverse",
    "build_block_triangular_field_split",
    "jacobi_smoothed_inverse",
]


@dataclasses.dataclass(frozen=True)
class FieldGroups:
    """A field-major degree-of-freedom partition into two contiguous groups of whole fields.

    The coupled state is stored **field-major**: degree of freedom ``(cell i, field f)`` sits at
    ``f * n_cells + i``. A partition that splits on a *field* boundary is therefore a partition into two
    contiguous ranges, which is what makes a field split cheap here — vectors are sliced rather than
    gathered, and the operator's four blocks are contiguous submatrices.

    This is a **view over a** :class:`~aquaflux.solve.FieldLayout`, not a second description of the same
    state: the layout owns the block structure and the ``f * n_cells + i`` arithmetic, and this adds only
    where the two groups meet. So a split can be named against the state's own blocks
    (:meth:`split_before`) rather than counted out by hand, and it moves with the layout when a field is
    added.

    Attributes
    ----------
    layout : FieldLayout
        The state's block layout. Every degree of freedom must belong to a whole per-cell field
        (:attr:`~aquaflux.solve.FieldLayout.is_cell_major`) — a bordered state's trailing multiplier
        belongs to no field, so it belongs to no field group either.
    n_leading_fields : int
        Fields in the leading group, taken from the start of the field order; the rest trail.

    Raises
    ------
    ValueError
        If the layout carries degrees of freedom outside the per-cell fields, or if either group would be
        empty — a "split" with an empty side is the monolithic preconditioner wearing a disguise, and
        silently accepting it would report a field-split result that was never a field split.
    """

    layout: FieldLayout
    n_leading_fields: int

    def __post_init__(self) -> None:
        if not self.layout.is_cell_major:
            raise ValueError(
                f"a field partition needs every dof to belong to a whole per-cell field, but this "
                f"layout is {self.layout.size} dofs over {self.layout.n_fields} fields and "
                f"{self.layout.n_cells} cells (blocks {list(self.layout.names)})."
            )
        if not 0 < self.n_leading_fields < self.layout.n_fields:
            raise ValueError(
                f"both groups must hold at least one of the layout's {self.layout.n_fields} fields, "
                f"got {self.n_leading_fields} leading; a split with an empty side is not a split."
            )

    @classmethod
    def by_counts(cls, n_cells: int, n_leading_fields: int, n_trailing_fields: int) -> FieldGroups:
        """The partition of a state known only by its field counts, with no named blocks.

        For a caller holding a raw field-major operator rather than an assembled system — a probe over a
        materialized Jacobian, or a test fixture. Where the state's own layout is at hand, prefer
        :meth:`split_before`, which cannot fall out of step with it.

        Parameters
        ----------
        n_cells : int
            Cells in the mesh.
        n_leading_fields, n_trailing_fields : int
            Fields in each group.

        Returns
        -------
        FieldGroups
            The partition, over an anonymous two-block layout.

        Raises
        ------
        ValueError
            If either count is not positive — the empty-side refusal, raised by the block it would
            leave empty.
        """
        layout = FieldLayout.cell_fields(
            n_cells, leading=n_leading_fields, trailing=n_trailing_fields
        )
        return cls(layout, n_leading_fields)

    @classmethod
    def split_before(cls, layout: FieldLayout, name: str) -> FieldGroups:
        """The partition that puts every block before ``name`` in the leading group.

        Parameters
        ----------
        layout : FieldLayout
            The state's block layout.
        name : str
            The first block of the trailing group — ``"k"`` for the split that separates a coupled
            Reynolds-averaged state's pressure-velocity saddle from its transported scalars.

        Returns
        -------
        FieldGroups
            The partition.
        """
        return cls(layout, layout.field_offset(name))

    @property
    def n_cells(self) -> int:
        """Cells in the mesh."""
        return self.layout.n_cells

    @property
    def n_trailing_fields(self) -> int:
        """Fields in the trailing group."""
        return self.layout.n_fields - self.n_leading_fields

    @property
    def n_fields(self) -> int:
        """Total fields per cell."""
        return self.layout.n_fields

    @property
    def n_dofs(self) -> int:
        """Total degrees of freedom."""
        return self.layout.size

    @property
    def n_leading_dofs(self) -> int:
        """Degrees of freedom in the leading group."""
        return self.layout.field_dofs(self.n_leading_fields)

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

    def active_rows(self) -> np.ndarray:
        """Which field-pair blocks a block-triangular split over this partition ever applies.

        A block-triangular inverse (:class:`BlockTriangularFieldSplit`) fits one inverse per diagonal
        block plus **one** off-diagonal triangle -- the other is never read, whatever the operator or the
        state. This is that fact as the ``(n_fields, n_fields)`` boolean table
        :class:`~aquaflux.solve.sparse_jacobian.ColumnProbePlan`'s ``active_rows`` wants, so a caller
        materializing a Jacobian specifically to feed a split can drop the unread triangle from the
        pattern before it is ever built rather than slicing it away afterward.

        Returns
        -------
        np.ndarray
            ``[row_field, column_field]``, ``True`` everywhere except the leading-by-trailing block. The
            split solves the leading group first and retains only the trailing-by-leading coupling.
        """
        active = np.ones((self.n_fields, self.n_fields), dtype=bool)
        nl = self.n_leading_fields
        active[:nl, nl:] = False  # leading rows <- trailing columns: never applied
        return active


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

    The leading group is always solved first. On a coupled Reynolds-averaged state, with the flow leading,
    that retains ``d R_turbulence / d flow`` -- the production terms' dependence on the velocity gradient.

    Parameters
    ----------
    leading, trailing : object
        The two block inverses, each exposing ``apply(residual, *, transpose=False) -> np.ndarray`` over
        its own group's degrees of freedom. :class:`~aquaflux.solve.HierarchyBlockInverse` satisfies this.
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
        # One body for both directions: transposing a block-lower-triangular inverse reverses the solve
        # order and uses the transposed coupling, which is the whole of the difference between them.
        order = ((self._leading, self._groups.leading), (self._trailing, self._groups.trailing))
        (first, first_dofs), (second, second_dofs) = reversed(order) if transpose else order
        coupling = self._transposed_coupling if transpose else self._coupling
        out = np.empty_like(residual)
        y_first = first.apply(residual[first_dofs], transpose=transpose)
        y_second = second.apply(residual[second_dofs] - coupling @ y_first, transpose=transpose)
        out[first_dofs] = y_first
        out[second_dofs] = y_second
        return out

    def refactor(self, matrix: sp.spmatrix) -> None:
        """Re-fit both blocks and the retained coupling to a new operator, IN PLACE.

        The counterpart of :meth:`~aquaflux.solve.AmgVCycle.refactor` for a split, and used for the same
        reason: a march's mid-run refresh must re-preconditioner the **same** compiled Krylov solve, which
        means mutating this object rather than replacing it. Each block re-fits through its own
        ``refactor``, so each keeps its own aggregation and re-computes only the coarse operators and the
        smoother's factor values — the economy the monolithic refresh relies on, preserved per block.

        Each inverse takes its new block through ``refactor_block(block)``, in the raw field-major form it
        was built from: a nodal coarsening recovers each cell as ``index % n_cells``, which only holds
        field-major.

        Parameters
        ----------
        matrix : scipy.sparse matrix
            The new assembled field-major operator, already shifted, of this partition's shape.

        Raises
        ------
        AttributeError
            If a block inverse offers no ``refactor_block`` (an injected inverse need not be refreshable
            at all).
        """
        blocks = self._groups.blocks(matrix)
        for inverse, block in ((self._leading, blocks[0]), (self._trailing, blocks[3])):
            if (refit := getattr(inverse, "refactor_block", None)) is not None:
                refit(block)
            else:
                raise AttributeError(
                    f"{type(inverse).__name__} cannot refactor in place, so this split cannot be "
                    "refreshed mid-march; rebuild it instead, or inject an inverse that can."
                )
        # Solving the leading group first corrects the trailing equations: the block retained is
        # trailing-by-leading.
        self._set_coupling(blocks[2])

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
    leading_inverse: Callable[[sp.csr_matrix, int], object],
    trailing_inverse: Callable[[sp.csr_matrix, int], object],
) -> BlockTriangularFieldSplit:
    """Build a block-triangular field split, fitting each diagonal block with its own injected inverse.

    Each factory is handed its own group's diagonal block and field count, so each block is fitted within
    its own group -- the point of the exercise, since a four-field saddle and a two-field transport pair
    want different inverses. The retained off-diagonal block, trailing-by-leading, is taken from
    ``matrix`` unmodified.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The assembled field-major operator, already shifted for the pseudo-transient step, shape
        ``(n_dofs, n_dofs)``.
    groups : FieldGroups
        The partition. Its leading group is the one listed first in the field order, and is solved first.
    leading_inverse, trailing_inverse : callable
        ``(sub_matrix, n_fields_in_group) -> inverse`` for that block -- for example
        :func:`~aquaflux.solve.simple_smoothed_inverse` on a pressure-velocity saddle and
        :func:`~aquaflux.solve.jacobi_smoothed_inverse` on a pair of transported scalars. The returned
        object must expose ``n_dofs`` and ``apply(residual, *, transpose=...)``, be a fixed linear map
        (the outer Krylov solve is not flexible) and transpose exactly (the adjoint's solve uses it).

    Returns
    -------
    BlockTriangularFieldSplit
        The frozen preconditioner.
    """
    leading_block, _, trailing_by_leading, trailing_block = groups.blocks(matrix)
    return BlockTriangularFieldSplit(
        leading_inverse(leading_block, groups.n_leading_fields),
        trailing_inverse(trailing_block, groups.n_trailing_fields),
        trailing_by_leading,
        groups,
    )


class FieldSplitAmgPreconditioner(MaterializedJacobianPreconditioner):
    """The field split as JAX matvecs, sharing the materialized-Jacobian machinery with the monolithic PC.

    A sibling of :class:`~aquaflux.solve.amg_preconditioner.MonolithicAmgPreconditioner` over the shared
    :class:`~aquaflux.solve.amg_preconditioner.MaterializedJacobianPreconditioner` base (#287), rather than
    a subclass of the monolithic class itself: only the coloured jvp probe that materializes the coupled
    Jacobian, the shift-diagonal add, the ``jax.pure_callback`` matvec (which reads ``self.factors`` at
    call time, so an in-place refresh re-preconditions the same compiled solve) and the teardown are
    genuinely shared — those live on the base. Everything the monolithic class builds *from* one
    :class:`~aquaflux.solve.AmgVCycle` (the fixed-pattern cell-major assembler) is monolithic-only, and
    inheriting it forced this class to declare two smoother parameters on its own refresh that a split's
    construction never reads.

    The monolithic path equilibrates and reorders the **whole** matrix to cell-major before handing it to
    one V-cycle; a split leaves any such preparation to each block's own injected inverse, because the two
    groups have different field counts and different scales. That is why the shift/equilibrate/reorder
    assembler the monolithic refresh precomputes has no counterpart here.

    .. warning::
       ``refresh_in_place`` is forward-march only, for the same reason as the monolithic class's: the
       mutation is impure and would corrupt an adjoint transpose solve that read the inverse between its
       own calls.
    """

    def __init__(
        self,
        split: BlockTriangularFieldSplit,
        groups: FieldGroups,
    ) -> None:
        super().__init__(split)
        self._groups = groups

    @property
    def groups(self) -> FieldGroups:
        """The field partition the preconditioner is built over."""
        return self._groups

    @classmethod
    def build(
        cls,
        matvec: Callable,
        plan,
        shift_diagonal: np.ndarray,
        groups: FieldGroups,
        *,
        leading_inverse: Callable[[sp.csr_matrix, int], object],
        trailing_inverse: Callable[[sp.csr_matrix, int], object],
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
        leading_inverse, trailing_inverse : callable
            ``(sub_matrix, n_fields_in_group) -> inverse`` for that block, exactly as
            :func:`build_block_triangular_field_split` takes them. An injected inverse must offer
            ``refactor_block`` or ``refactor`` to survive a mid-march refresh.

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
            leading_inverse=leading_inverse,
            trailing_inverse=trailing_inverse,
        )
        return cls(split, groups)

    def refresh_in_place(
        self,
        matvec: Callable,
        plan,
        shift_diagonal: np.ndarray,
        *,
        batched_matvec: Callable | None = None,
        probe_batch_size: int | None = None,
        structure: ProbeGather | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Re-materialize at the developed state and re-fit both blocks IN PLACE.

        The smoother configuration is fixed at :meth:`build` and cannot be changed by a refresh. Returns
        the same ``("probe", s), ("assemble", s), ("refactor", s)`` breakdown the monolithic refresh
        reports, so a march log reads identically for either preconditioner. Here "assemble" is only the
        diagonal shift — the per-block equilibration is inside the refactor.
        """
        timer = PhaseTimer()
        jacobian = self._materialize_jacobian(
            matvec, plan, batched_matvec, probe_batch_size, structure
        )
        timer.lap("probe")
        shifted = self._shifted(jacobian, shift_diagonal)
        timer.lap("assemble")
        self.factors.refactor(shifted)
        timer.lap("refactor")
        return timer.phases()


class _NodalSmoother(NamedTuple):
    """The nodal cycle's counts and relaxations -- everything about it that must be concrete.

    A plain tuple of Python numbers, so it is hashable and compares by value: it lands wholly on the
    static side of :func:`_jacobi_smoothed_cycle` and two builds at the same settings share one compiled
    cycle.
    """

    cycles: int
    sweeps: int
    omega: float
    spectral_damping: bool


@eqx.filter_jit
def _jacobi_smoothed_cycle(
    hierarchy: SmoothedHierarchy,
    extras: None,
    residual: jnp.ndarray,
    smoother: _NodalSmoother,
) -> jnp.ndarray:
    """``smoother.cycles`` V-cycles over ``hierarchy``, relaxed by a damped or block Jacobi sweep.

    Module-level and taking the hierarchy as an ARGUMENT, so a refresh at unchanged shapes swaps its
    values into the SAME compiled cycle. ``extras`` is the smoother-specific record the shared
    :class:`~aquaflux.solve.hierarchy_inverse.HierarchyBlockInverse` passes through; this family reads the
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


class JacobiSmoothedInverse(HierarchyBlockInverse):
    """A block inverse from ONE traced hierarchy over the whole group, coarsening cells.

    Given a block size it coarsens **cells**, so one hierarchy spans the whole group and the cross-field
    coupling sits inside the operator being coarsened rather than being approximated away outside it.
    That is what makes it stronger than a per-field hierarchy composed block-triangularly, and why a
    measurement taken on such a pair does not transfer here.

    Two things have to change together and neither suffices alone — measured, both refused otherwise:
    the aggregation must coarsen cells, and the level smoother must invert each cell's dense block
    rather than the scalar diagonal. On a multi-field operator whose within-cell coupling exceeds its
    diagonal, a point smoother discards the dominant term and the sweep does not contract.

    Host in, host out: the field split is numpy and the hierarchy is JAX, so each
    application crosses the boundary. A production traced split would keep the whole thing traced.

    Parameters
    ----------
    block, n_fields, strength_threshold, max_levels, max_coarse, frozen_coarsening, shape_headroom, report
        See :class:`~aquaflux.solve.hierarchy_inverse.HierarchyBlockInverse`, which owns the hierarchy,
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
    avoid_singletons : bool
        Attach a maximal-independent-set aggregate with no free neighbour to an adjacent one instead of
        leaving it standing alone. ``False`` (default, byte-identical off) matches the class's original
        behaviour. A nonzero ``strength_threshold`` prunes edges before aggregation and so is more prone
        to stranding vertices this way; see :func:`~aquaflux.solve.multigrid.build_convection_hierarchy`.

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
        avoid_singletons: bool = False,
        **hierarchy_settings,
    ) -> None:
        self._aggressive_levels = aggressive_levels
        self._prolongation_smoothing = prolongation_smoothing
        self._equilibrate = equilibrate
        self._avoid_singletons = avoid_singletons
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
            "avoid_singletons": self._avoid_singletons,
        }

    def smoother(self) -> _NodalSmoother:
        return self._smoother

    def cycle(self):
        return _jacobi_smoothed_cycle


class AirBlockInverse:
    """A block inverse from a **reduction-based** (lAIR) hierarchy over the whole group.

    The alternative to :class:`JacobiSmoothedInverse` for a transported-scalar block. Both coarsen cells
    and both smooth with each cell's own block; they differ in the coarse space. Aggregation groups
    cells and takes ``R = Pᵀ``; lAIR splits them coarse/fine and builds an **independent** restriction
    approximating the ideal ``R = [-A_cf A_ff⁻¹, I]``, which for a convection-dominated operator makes
    eliminating the fine points nearly exact — Peclet-robust and mesh-independent where a deep Galerkin
    recursion is not (Manteuffel, Ruge & Southworth, SISC 2018).

    **It does not subclass** :class:`~aquaflux.solve.hierarchy_inverse.HierarchyBlockInverse`: that base
    owns a :class:`~aquaflux.solve.multigrid.SmoothedHierarchy` and refreshes it by re-fitting the
    aggregation, while this owns an :class:`~aquaflux.solve.multigrid.AirHierarchy` and refreshes by
    re-solving the local restriction systems on a frozen C/F split. Widening one class to hold either
    would make it the union of two coarsening families, needing the settings of each to sit unused on
    the other.

    Parameters
    ----------
    block : scipy.sparse matrix
        The group's diagonal block, field-major, shape ``(n_group_fields * n_cells,) * 2``.
    n_group_fields : int
        Fields per cell in this group.
    cycles : int
        V-cycles per application. Fixed, so ``b -> x`` stays a linear map — required by the non-flexible
        outer Krylov and by the transposed adjoint solve.
    f_iters, c_iters : int
        Fine- and coarse-point sweeps in the FC-Jacobi smoother.
    omega : float
        Smoother damping.
    **settings
        Forwarded to :func:`~aquaflux.solve.multigrid.build_air_hierarchy` (``theta``,
        ``restriction_theta``, ``degree``, ``max_coarse``, ``max_levels``).
    """

    def __init__(
        self,
        block: sp.spmatrix,
        n_group_fields: int,
        *,
        cycles: int = 1,
        f_iters: int = 2,
        c_iters: int = 1,
        omega: float = 1.0,
        **settings,
    ) -> None:
        self._n_dofs = int(block.shape[0])
        self._block_size = int(n_group_fields)
        self._settings = settings
        self._transpose_fn = None
        # The hierarchy rides as an ARGUMENT, never a closure. A `jax.jit` built fresh per refresh
        # starts with an empty compilation cache and recompiles whether or not anything moved --
        # measured at ~4x on a comparable block, with the control being a sibling that already passed
        # its hierarchy in and moved 1.01x.
        self._cycle = jax.jit(
            lambda hierarchy, b: air_multigrid_solve(
                hierarchy, b, cycles=cycles, f_iters=f_iters, c_iters=c_iters, omega=omega
            )
        )
        self._hierarchy = build_air_hierarchy(
            sp.csr_matrix(block), block_size=self._block_size, **settings
        )

    @property
    def n_dofs(self) -> int:
        """Degrees of freedom in this block."""
        return self._n_dofs

    def _solve(self, vector: jnp.ndarray) -> jnp.ndarray:
        return self._cycle(self._hierarchy, vector)

    def refactor_block(self, block: sp.spmatrix) -> None:
        """Re-derive the values on the frozen coarsening, IN PLACE — required to survive a refresh.

        Reuses the C/F split, the prolongation and the restriction's F-neighbourhoods, re-solving only
        the local systems, so every shape is held and the compiled cycle above is a cache hit rather
        than a recompile. :func:`~aquaflux.solve.multigrid.refresh_air_hierarchy` raises if any shape
        moved, which would mean the operator did not come from the same mesh graph.
        """
        self._hierarchy = refresh_air_hierarchy(self._hierarchy, sp.csr_matrix(block))
        self._transpose_fn = None  # the transpose closes over the hierarchy it was built for

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Approximate ``A^-1 r`` (or ``A^-T r``) with a fixed number of lAIR V-cycles.

        ``R != Pᵀ`` on a reduction hierarchy, so the transpose is not the same cycle with the transfers
        swapped by hand — it is the transpose of the whole linear map, which is what
        :func:`jax.linear_transpose` gives exactly for a fixed-cycle, fixed-smoothing operator.
        """
        vector = jnp.asarray(residual, dtype=jnp.float64)
        if not transpose:
            return np.asarray(self._solve(vector), dtype=np.float64)
        if self._transpose_fn is None:
            self._transpose_fn = jax.linear_transpose(
                self._solve, jnp.zeros(self._n_dofs, dtype=jnp.float64)
            )
        return np.asarray(self._transpose_fn(vector)[0], dtype=np.float64)

    def destroy(self) -> None:
        """Nothing to release -- plain arrays, not a host solver's handles."""


def air_inverse(**settings) -> Callable[[sp.spmatrix, int], object]:
    """A ``trailing_inverse`` factory using :class:`AirBlockInverse`.

    Every keyword is forwarded, so the defaults live on the class and on
    :func:`~aquaflux.solve.multigrid.build_air_hierarchy` rather than being restated here.
    ``restriction_theta`` is the one worth knowing about: it trades the restriction's accuracy against
    how dense the coarse operators become, and the density compounds down the hierarchy.
    """

    def build(block: sp.spmatrix, n_group_fields: int) -> object:
        return AirBlockInverse(block, n_group_fields, **settings)

    return build


def jacobi_smoothed_inverse(**settings) -> Callable[[sp.spmatrix, int], object]:
    """A ``leading_inverse``/``trailing_inverse`` factory using :class:`JacobiSmoothedInverse`.

    Every keyword is forwarded, so the defaults — and the reasoning behind them — live on the class
    rather than being restated here. ``max_coarse`` is worth knowing about: it is the coarse-grid size
    the hierarchy stops at and solves directly, and a coarse grid large enough to invert the global
    coupling exactly is measured to be worth a great deal on this operator.
    """

    def build(block: sp.spmatrix, n_group_fields: int) -> object:
        return JacobiSmoothedInverse(block, n_group_fields, **settings)

    return build
