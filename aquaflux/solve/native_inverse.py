"""The shared body of a JAX-native block inverse: one frozen hierarchy, refreshed in place.

Two preconditioners in this package wrap a coarsened hierarchy and hand a Krylov solve an approximate
``A^-1``: :class:`~aquaflux.solve.field_split.NodalNativeInverse` over a transported-scalar group, and
:class:`~aquaflux.solve.saddle_multigrid.NativeSimpleInverse` over the velocity--pressure saddle. They
differ **only** in their level smoother -- a damped/block Jacobi sweep against a SIMPLE relaxation --
and everything around that smoother is the same problem solved twice: build the hierarchy, re-derive it
when the march refreshes, keep the jitted cycle a compilation-cache hit across that refresh, marshal a
host vector in and out, and produce a transpose for the adjoint.

Solving it twice is what let the two drift. The saddle inverse grew a strength threshold, a level cap,
a fixed shape ladder and a refit path; the nodal one grew none of them, so the trailing block was
hard-capped at two levels with an isotropic coarsening and no way to sweep either -- not because that
was decided, but because the knobs were added on the other side of a duplicated seam. This class is the
seam, so a capability added here reaches both.

**What a subclass supplies is exactly the smoother and nothing else:** the build settings its hierarchy
wants, an optional per-level record derived from each rebuilt hierarchy (the SIMPLE pieces; the nodal
smoother needs none), a hashable record of the smoother's own counts and relaxations, and the
module-level jitted cycle that applies them. Everything above is written once, here.

**The jitted cycle is module-level and takes the hierarchy as an ARGUMENT, in both families (binding).**
A cycle built per instance -- or worse, per refresh -- closes over the hierarchy's arrays, which makes
them compile-time constants: a new closure is a new cache key whatever its contents, so every refresh
recompiles the whole unrolled V-cycle, and the first build embeds a hierarchy's worth of arrays in the
compiled program rather than passing them as buffers. Measured on a 92160-degree-of-freedom flow block,
that was 1.07 s per refresh and 1.31 s against 0.16 s on the first apply. The level records' static/traced
split exists precisely to make the argument form possible; taking it is what collects the benefit.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .multigrid import ShapeBudget, SmoothedHierarchy, build_convection_hierarchy


class NativeHierarchyInverse:
    """A block inverse applying a frozen coarsened hierarchy, refreshable without recompiling.

    Not instantiated directly: a subclass supplies its smoother through the four hooks below. Held as a
    plain object rather than an ``equinox.Module`` because a field split holds it by identity and
    mutates it in place on a refresh, which is what lets the compiled Krylov solve keep the same
    preconditioner object across the mutation.

    Parameters
    ----------
    block : scipy.sparse matrix
        The group's diagonal block, **field-major**: ``(cell i, field f)`` at ``f * n_cells + i``. The
        nodal coarsening recovers each cell as ``index % n_cells``, which only holds in that layout.
    n_fields : int
        Fields per cell, the aggregation's block size.
    strength_threshold : float
        Aggregate only along connections at least this fraction of the row's strongest
        (:func:`~aquaflux.solve.multigrid._aggregation_edges`). ``0`` (default) is isotropic aggregation
        on the full graph, which reads no operator values at all.

        **Above zero the coarsening reads ``|A_ij|``, and that forfeits the refresh guarantee below
        unless it is paired with ``frozen_coarsening`` or ``shape_headroom``.** At zero, a hierarchy
        re-derived at a new operator on a fixed mesh has identical aggregates and therefore identical
        shapes, so the jitted cycle is a cache hit; above zero the partition follows the flow and the
        shapes move with it. That is a real trade rather than a defect -- a strength threshold is what
        keeps the V-cycle contracting on an anisotropic or wall-graded operator -- and a single-state
        probe never refreshes, so it costs a sweep nothing.
    max_levels : int
        Stop coarsening at this many levels and solve the last one directly.
    max_coarse : int
        Stop coarsening once a level has at most this many degrees of freedom. Dofs, not cells: the
        limit bounds a dense inverse whose cost is cubic in the dof count.
    frozen_coarsening : bool
        Refresh by refitting values onto the coarsening derived at the first build
        (:meth:`~aquaflux.solve.multigrid.SmoothedHierarchy.refit`) rather than coarsening again. The
        shapes then cannot move at all; what it trades is coarse-space quality, since the partition
        describes the operator at the state it was first built at.
    shape_headroom : float, optional
        Coarsen into a fixed ladder of array sizes, discovered on the first build and padded by this
        factor. Unlike freezing, the partition is still re-derived from the current operator at every
        refresh -- only the sizes it is poured into are held. ``None`` disables it and is
        byte-identical.
    report : callable, optional
        Where the build record goes. ``None`` is silent; a case passes ``print``. A library object must
        not write to stdout on its own.
    """

    def __init__(
        self,
        block: sp.spmatrix,
        n_fields: int,
        *,
        strength_threshold: float = 0.0,
        max_levels: int = 2,
        max_coarse: int = 16,
        frozen_coarsening: bool = False,
        shape_headroom: float | None = None,
        report: Callable[[str], None] | None = None,
    ) -> None:
        matrix = sp.csr_matrix(block)
        self._n_dofs = matrix.shape[0]
        self._report = report if report is not None else lambda _message: None
        self._frozen_coarsening = frozen_coarsening
        self._shape_headroom = shape_headroom
        # Held so a refresh re-derives at exactly the settings this was built at, rather than at
        # whatever the builder's defaults happen to be.
        self._build_settings = {
            "block_size": n_fields,
            "strength_threshold": strength_threshold,
            "max_levels": max_levels,
            "max_coarse": max_coarse,
            **self.build_settings(),
        }
        # Discovered on the first build and held for the life of the inverse, so every later rebuild
        # coarsens into the same ladder. `None` until then, and forever if no headroom was asked for.
        self._budget: ShapeBudget | None = None
        self._transpose_fn = None
        self._rebuild(matrix)

    # --- the four hooks a smoother supplies -------------------------------------------------

    def build_settings(self) -> dict:
        """Extra keyword arguments this family's hierarchy is coarsened with."""
        raise NotImplementedError

    def smoother(self):
        """This smoother's counts and relaxations, as a hashable record.

        It rides on the **static** side of the jitted cycle, so it must compare by value: a plain
        ``NamedTuple`` of Python numbers does, and two builds at the same settings then share one
        compiled cycle.
        """
        raise NotImplementedError

    def cycle(self):
        """The module-level jitted ``(hierarchy, extras, residual, smoother) -> x``."""
        raise NotImplementedError

    def derive_extras(self, hierarchy: SmoothedHierarchy):
        """Per-level records this smoother needs, re-derived from each rebuilt hierarchy.

        Returned as a **pytree of traced leaves**, since it is passed to the cycle as an argument
        beside the hierarchy. ``None`` (the default) is for a smoother that reads the levels alone.
        """
        return None

    # --- the shared body --------------------------------------------------------------------

    @property
    def n_dofs(self) -> int:
        """Degrees of freedom in this block."""
        return self._n_dofs

    def _coarsen(self, matrix: sp.csr_matrix, budget: ShapeBudget | None) -> SmoothedHierarchy:
        """Build the hierarchy at this inverse's settings, optionally into a fixed shape ladder."""
        return build_convection_hierarchy(matrix, shape_budget=budget, **self._build_settings)

    def _rebuild(self, matrix: sp.csr_matrix) -> None:
        """Coarsen from scratch, then derive the smoother over the result.

        With ``shape_headroom`` set, the FIRST build runs twice: once to discover what this operator
        naturally coarsens into, then again into a budget derived from it. Every later rebuild reuses
        that budget, so it re-derives the partition from the current operator -- unlike a frozen
        coarsening -- while keeping the array shapes, and therefore the compiled cycle, fixed.
        """
        self._hierarchy = self._coarsen(matrix, self._budget)
        if self._budget is None and self._shape_headroom is not None:
            self._budget = self._hierarchy.shape_budget(self._shape_headroom)
            self._report(
                f"      shape ladder: cells {self._budget.coarse_cells}, "
                f"nnz {self._budget.operator_nnz} (headroom {self._shape_headroom})",
            )
            self._hierarchy = self._coarsen(matrix, self._budget)
        self._after_coarsening()

    def _after_coarsening(self) -> None:
        """Re-derive whatever the smoother reads off the hierarchy, and drop the stale transpose."""
        self._extras = self.derive_extras(self._hierarchy)
        # LAZY. `jax.linear_transpose` traces eagerly, and a forward march never applies the transpose
        # -- only the adjoint does. Building it at construction cost a measured 0.27 GB and 0.27 s per
        # arm for something most callers never touch. Dropped rather than kept here, because it
        # describes the hierarchy this call has just replaced.
        self._transpose_fn = None

    def _solve(self, residual: jnp.ndarray) -> jnp.ndarray:
        """The cycle over what this inverse currently holds, traced in and not captured."""
        return self.cycle()(self._hierarchy, self._extras, residual, self.smoother())

    def refactor_block(self, block: sp.spmatrix) -> None:
        """Re-fit to a new operator on the same graph, IN PLACE. Required to survive a march refresh.

        The field split refuses to refresh an inverse offering neither this nor ``refactor``, because
        replacing the object would recompile the whole coupled solve -- so without it this
        preconditioner cannot be used in a march at all, and a single-state probe never reaches the
        code path.

        Takes the **raw field-major** block rather than the equilibrated cell-major form a host solver
        would want, because the nodal coarsening recovers each cell as ``index % n_cells``.

        **Whether a rebuild is structure-preserving depends on ``strength_threshold``.** At zero the
        aggregation reads only the sparsity pattern, so every array keeps its shape and the jitted cycle
        stays a compilation-cache hit. Above zero it reads ``|A_ij|`` and the coarsening moves as the
        flow develops; ``frozen_coarsening`` and ``shape_headroom`` are the two answers, and without
        either the level sizes are compared against the previous build so a retrace is reported with a
        reason attached rather than being silent.

        Parameters
        ----------
        block : scipy.sparse matrix
            The group's new diagonal block, field-major, of the shape this was built at.

        Raises
        ------
        ValueError
            If the new block's shape differs from the built one -- silently re-fitting to a different
            operator would give a preconditioner for a system nothing is solving.
        """
        matrix = sp.csr_matrix(block)
        if matrix.shape != (self._n_dofs, self._n_dofs):
            raise ValueError(
                f"cannot refactor a {self._n_dofs}-dof inverse onto a {matrix.shape[0]}-dof block."
            )
        if self._frozen_coarsening:
            self._hierarchy = self._hierarchy.refit(matrix)
            self._after_coarsening()
            return
        before = tuple(level.n for level in self._hierarchy.levels)
        self._rebuild(matrix)
        after = tuple(level.n for level in self._hierarchy.levels)
        if after != before:
            self._report(
                f"      refresh moved the coarsening: levels {before} -> {after} "
                f"(the jitted cycle retraces; the strength threshold reads values)",
            )

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Approximate ``A^-1 r`` (or ``A^-T r``) with one hierarchy over the whole group."""
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
