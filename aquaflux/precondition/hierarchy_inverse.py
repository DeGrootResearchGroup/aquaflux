"""The shared body of a traced block inverse: one frozen hierarchy, refreshed in place.

Two preconditioners in this package wrap a coarsened hierarchy and hand a Krylov solve an approximate
``A^-1``: :class:`~aquaflux.precondition.field_split.JacobiSmoothedInverse` over a transported-scalar
group, and :class:`~aquaflux.precondition.saddle_multigrid.SimpleSmoothedInverse` over the
velocity--pressure saddle. They differ **only** in their level smoother -- a damped/block Jacobi sweep
against a SIMPLE relaxation -- and everything around that smoother is the same problem solved twice:
build the hierarchy, re-derive it when the march refreshes, keep the jitted cycle a compilation-cache hit
across that refresh, marshal a host vector in and out, and produce a transpose for the adjoint.

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

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from ..solve import with_transposes
from .entry_map import DeviceMatrix, as_scipy
from .multigrid import (
    FrozenCoarsening,
    ShapeBudget,
    SmoothedHierarchy,
    build_convection_hierarchy,
)
from .traced_cycle import TracedCycle, in_precision


@dataclasses.dataclass(frozen=True)
class _HierarchyStep:
    """The static half of a hierarchy inverse's :class:`~aquaflux.precondition.traced_cycle.TracedCycle`.

    Holds the module-level jitted cycle and the smoother's hashable settings, and takes the
    ``(hierarchy, extras)`` state as an argument, so it compares by value and closes over no array.
    """

    cycle: Callable
    smoother: object

    def __call__(self, state, residual: jnp.ndarray) -> jnp.ndarray:
        hierarchy, extras = state
        return self.cycle(hierarchy, extras, residual, self.smoother)


def _narrowed(levels: list, equilibration, precision: str) -> SmoothedHierarchy:
    """A hierarchy's levels with their floating-point arrays cast to ``precision``, prolongation values kept.

    The prolongation values stay as they are because they are the very arrays the coarsening holds
    (:class:`~aquaflux.precondition.multigrid.FrozenCoarsening`), so a rounded copy would be held beside
    them rather than in their place. Everything else is
    rounded once, exactly as a traced split applied in ``precision`` rounds it, so a hierarchy held in
    single precision and one narrowed when it is applied hold the same entries.

    ``levels`` is consumed: each double level is replaced in the list by its rounded copy as soon as
    that copy exists, so a caller holding no other reference to the double hierarchy frees it a level
    at a time rather than holding all of it beside all of its rounded copy.
    """
    for index, level in enumerate(levels):
        levels[index] = eqx.tree_at(
            lambda cast: cast.p_val,
            in_precision(level, precision),
            level.p_val,
            is_leaf=lambda leaf: leaf is None,
        )
        del level
    return SmoothedHierarchy(
        levels=tuple(levels), equilibration=in_precision(equilibration, precision)
    )


class HierarchyBlockInverse:
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
        (:func:`~aquaflux.precondition.multigrid._aggregation_edges`). ``0`` (default) is isotropic
        aggregation on the full graph, which reads no operator values at all.

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
        (:meth:`~aquaflux.precondition.multigrid.FrozenCoarsening.fit`) rather than coarsening again.
        The shapes then cannot move at all; what it trades is coarse-space quality, since the partition
        describes the operator at the state it was first built at.
    shape_headroom : float, optional
        Coarsen into a fixed ladder of array sizes, discovered on the first build and padded by this
        factor. Unlike freezing, the partition is still re-derived from the current operator at every
        refresh -- only the sizes it is poured into are held. ``None`` disables it and is
        byte-identical.
    report : callable, optional
        Where the build record goes. ``None`` is silent; a case passes ``print``. A library object must
        not write to stdout on its own.
    stored_precision : {"float64", "float32"}
        The floating-point type the hierarchy is **held** in between refreshes. Each fit and refit is
        computed in double precision and then rounded once to this type (the prolongation values
        excepted, which the held coarsening shares). ``"float32"`` is for an inverse applied in single
        precision anyway (a field split with ``precision="float32"``): its traced cycle then needs no
        second, narrowed copy of the hierarchy beside a double one nothing applies. :meth:`apply` always
        runs in double precision, on the rounded entries.
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
        stored_precision: str = "float64",
    ) -> None:
        if stored_precision not in ("float64", "float32"):
            raise ValueError(
                f"stored_precision must be 'float64' or 'float32', got {stored_precision!r}."
            )
        self._stored_precision = stored_precision
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
        # This inverse's own last aggregate-size histogram (see `_coarsen`) -- empty until the first
        # build, and left untouched by a frozen-coarsening refresh (a fit), which aggregates nothing
        # new. A diagnostic only; a subclass that wants to report it reads this rather than the builder.
        self._aggregate_stats: tuple[dict, ...] = ()
        # The coarsening the last build fitted through, on the device; a frozen refresh fits it again.
        self._coarsening: FrozenCoarsening | None = None
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
        """Build the hierarchy at this inverse's settings, optionally into a fixed shape ladder.

        Captures this build's own per-level aggregate-size histogram into ``self._aggregate_stats`` --
        a plain list handed to the builder and read back here, never a shared accumulator, so nothing
        another hierarchy aggregates can ever land on this one's report (#287).
        """
        stats: list[dict] = []
        coarsening: list[FrozenCoarsening] = []
        hierarchy = build_convection_hierarchy(
            matrix, shape_budget=budget, stats=stats, coarsening=coarsening, **self._build_settings
        )
        self._aggregate_stats = tuple(stats)
        (self._coarsening,) = coarsening
        return hierarchy

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
                f"row widths {self._budget.operator_width} (headroom {self._shape_headroom})",
            )
            self._hierarchy = self._coarsen(matrix, self._budget)
        self._after_coarsening()

    def _after_coarsening(self) -> None:
        """Re-derive whatever the smoother reads off the hierarchy, and drop the stale transpose."""
        # A smoother whose extras are held narrow rounds each level's record as it forms it (the
        # SIMPLE pieces do), so the cast below returns those arrays unchanged.
        self._extras = in_precision(self.derive_extras(self._hierarchy), self._stored_precision)
        if self._stored_precision != "float64":
            # After the extras: they are derived from the double-precision levels, and only then are
            # the levels rounded, so a narrowed hierarchy holds exactly what narrowing it later would.
            # The double hierarchy is let go first, so `_narrowed` holds its only reference.
            double, self._hierarchy = self._hierarchy, None
            levels, equilibration = list(double.levels), double.equilibration
            del double
            self._hierarchy = _narrowed(levels, equilibration, self._stored_precision)
        # LAZY. `jax.linear_transpose` traces eagerly, and a forward march never applies the transpose
        # -- only the adjoint does. Building it at construction cost a measured 0.27 GB and 0.27 s per
        # arm for something most callers never touch -- and it holds the level operators' packed
        # transposes beside them (`with_transposes`). Dropped rather than kept here, because it
        # describes the hierarchy this call has just replaced.
        self._transpose_fn = None

    def traced_cycle(self) -> TracedCycle:
        """The cycle over what this inverse currently holds, its hierarchy as leaves rather than constants.

        A fresh value per call, reading the current hierarchy, so a caller holding one from before a
        refresh holds the old state; take a new one after :meth:`refactor_block`.

        Returns
        -------
        TracedCycle
            ``r -> M r`` with ``(hierarchy, extras)`` as its array state and the jitted cycle plus the
            smoother's settings as its static step.
        """
        return TracedCycle(
            state=(self._hierarchy, self._extras),
            step=_HierarchyStep(self.cycle(), self.smoother()),
        )

    def _solve(self, residual: jnp.ndarray) -> jnp.ndarray:
        """The cycle over what this inverse currently holds, traced in and not captured.

        Applied in double precision whatever the hierarchy is held in: a double residual promotes every
        operation on single-precision entries, so this is an exactly linear map, as a transpose solve run
        tightly needs.
        """
        return self.traced_cycle()(residual)

    def refactor_block(self, block: sp.spmatrix) -> None:
        """Re-fit to a new operator on the same graph, IN PLACE. Required to survive a march refresh.

        A refresh refuses an inverse that does not offer this, because replacing the object would
        recompile the whole coupled solve -- so without it this preconditioner cannot be used in a march
        at all, and a single-state probe never reaches the code path.

        Takes the **raw field-major** block rather than the equilibrated cell-major form a host solver
        would want, because the nodal coarsening recovers each cell as ``index % n_cells``.

        **Whether a rebuild is structure-preserving depends on ``strength_threshold``.** At zero the
        aggregation reads only the sparsity pattern, so every array keeps its shape and the jitted cycle
        stays a compilation-cache hit. Above zero it reads ``|A_ij|`` and the coarsening moves as the
        flow develops; ``frozen_coarsening`` and ``shape_headroom`` are the two answers, and without
        either the level sizes are compared against the previous build so a retrace is reported with a
        reason attached rather than being silent.

        **Where a rebuild would reproduce the coarsening exactly, this refits instead** -- at no strength
        threshold, with an unsmoothed prolongation and no shape ladder (:meth:`_coarsening_reads_values`).
        The aggregates and the prolongation are then functions of the pattern alone, so re-aggregating
        returns the hierarchy already held, and the refit re-derives the same values without the
        aggregation.

        Parameters
        ----------
        block : scipy.sparse matrix or DeviceMatrix
            The group's new diagonal block, field-major, of the shape this was built at. A refit fits a
            :class:`~aquaflux.precondition.entry_map.DeviceMatrix` from its values on the device; only a
            re-coarsening reads it back to the host.

        Raises
        ------
        ValueError
            If the new block's shape differs from the built one -- silently re-fitting to a different
            operator would give a preconditioner for a system nothing is solving.
        """
        if block.shape != (self._n_dofs, self._n_dofs):
            raise ValueError(
                f"cannot refactor a {self._n_dofs}-dof inverse onto a {block.shape[0]}-dof block."
            )
        if self._frozen_coarsening or not self._coarsening_reads_values():
            # The held hierarchy is let go first, so the old levels are not alive beside the new ones.
            self._hierarchy = None
            self._hierarchy = self._coarsening.fit(block, "HierarchyBlockInverse.refactor_block")
            self._after_coarsening()
            return
        before = tuple(level.n for level in self._hierarchy.levels)
        self._rebuild(as_scipy(block))
        after = tuple(level.n for level in self._hierarchy.levels)
        if after != before:
            self._report(
                f"      refresh moved the coarsening: levels {before} -> {after} "
                f"(the jitted cycle retraces; the strength threshold reads values)",
            )

    def _coarsening_reads_values(self) -> bool:
        """Whether rebuilding at a new operator of the same pattern could change the coarsening.

        It cannot when nothing that builds the transfers reads a value: no strength threshold (the
        aggregation then sees the pattern alone, :func:`~aquaflux.precondition.multigrid._aggregation_edges`),
        an unsmoothed prolongation (the aggregates' indicator, not a relaxation of the operator) and no
        shape ladder (asking for one asks for each refresh to re-aggregate into it). Equilibration
        rescales values only, so it does not enter.
        """
        settings = self._build_settings
        return (
            settings["strength_threshold"] > 0.0
            or settings.get("prolongation_smoothing") != "none"
            or self._shape_headroom is not None
        )

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """Approximate ``A^-1 r`` (or ``A^-T r``) with one hierarchy over the whole group."""
        vector = jnp.asarray(residual, dtype=jnp.float64)
        if not transpose:
            return np.asarray(self._solve(vector), dtype=np.float64)
        if self._transpose_fn is None:
            self._transpose_fn = jax.linear_transpose(
                with_transposes(self.traced_cycle()), jnp.zeros(self._n_dofs, dtype=jnp.float64)
            )
        return np.asarray(self._transpose_fn(vector)[0], dtype=np.float64)

    def destroy(self) -> None:
        """Nothing to release -- plain arrays, not a host solver's handles."""


def block_for(inverse: object, block: sp.spmatrix | DeviceMatrix) -> sp.spmatrix | DeviceMatrix:
    """``block`` in the form ``inverse`` re-fits from.

    A :class:`HierarchyBlockInverse` fits on the device and takes a block there as it is; any other
    inverse is handed a host matrix, as :class:`~aquaflux.precondition.RefactorableInverse` promises it.
    """
    return block if isinstance(inverse, HierarchyBlockInverse) else as_scipy(block)
