"""How a coupled Jacobian is materialized: the colouring plan and its fixed de-compression map.

A preconditioner built from the assembled Jacobian (a complete LU, a multigrid V-cycle, a field split)
recovers that matrix by coloured directional-derivative probing. Both the colouring and the map that
de-compresses the probe responses are functions of the cell graph and the stencil / per-column reaches
alone -- never of the state and never of a coefficient -- so **one probe is valid for a whole
continuation** and for the refresh hook running beside it. Nothing about them depends on which residual
is being differentiated, apart from the one optional stand-in a residual may want to differentiate in
place of itself (``narrowing``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

from .sparse_jacobian import (
    ColumnProbePlan,
    ProbeGather,
    block_stencil_colouring,
    block_stencil_gather_map,
    column_probe_plan,
)

__all__ = ["JacobianProbe", "jacobian_probe_plan"]


def jacobian_probe_plan(
    face_cells,
    n_cells: int,
    n_fields: int,
    stencil_reach: int,
    column_reach: Sequence[int] | None = None,
    active_rows: np.ndarray | None = None,
) -> ColumnProbePlan:
    """The probing plan for materializing a coupled Jacobian (a mesh-fixed quantity).

    Shared by every monolithic-factorization builder (the initial factorization) and by each in-place
    refresh, so all of them probe the Jacobian the same way.

    ``column_reach`` gives each column field its own reach while keeping the assembly pattern at
    ``stencil_reach``, so the materialized sparsity is unchanged. It is a property of the *case*: which
    columns close inside a shorter reach follows from the schemes the residual was assembled with, so
    it must be measured for a given case rather than assumed. ``None`` (default) probes every column at
    ``stencil_reach``.

    ``active_rows`` excludes field-pair blocks from the pattern entirely, for a caller that already
    knows some sub-block of the materialized Jacobian will never be read -- see
    :meth:`~aquaflux.solve.FieldGroups.active_rows`. ``None`` (default) wants every block.

    Parameters
    ----------
    face_cells : FaceCellConnectivity
        The mesh's face-to-cell connectivity, read for its interior edges.
    n_cells : int
        Number of cells.
    n_fields : int
        Number of fields in the flat state layout.
    stencil_reach : int
        The cell-graph distance the assembled sparsity covers.
    column_reach : sequence of int, optional
        A shorter reach per column field.
    active_rows : np.ndarray, optional
        Field-pair blocks to exclude from the pattern.

    Returns
    -------
    ColumnProbePlan
        The collision-free colouring and per-column reach.
    """
    owner, nb, _ = face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)
    if column_reach is None:
        return ColumnProbePlan.uniform(
            block_stencil_colouring(owner, nb, n_cells, stencil_reach),
            n_fields,
            active_rows=active_rows,
        )
    return column_probe_plan(
        owner, nb, n_cells, column_reach, stencil_reach, active_rows=active_rows
    )


@dataclasses.dataclass(frozen=True)
class JacobianProbe:
    """How the coupled Jacobian is materialized: the colouring plan and its fixed de-compression map.

    They are one object rather than two arguments threaded in parallel because they are built together
    and consumed together at every call site (the initial build, and every in-place refresh).

    Building them is not free. The colouring is a graph pass over the whole mesh, and on a
    three-dimensional coupled case the gather map is the single largest allocation the case makes -- so
    a driver that builds one engine per continuation rung with a refresh hook beside it would otherwise
    build both twice per rung. A preconditioner session builds exactly one and hands it to every
    consumer.

    Attributes
    ----------
    plan : ColumnProbePlan
        The collision-free colouring and per-column reach the coloured directional-derivative probe
        runs, which is what fixes how many probes a materialize costs.
    structure : ProbeGather
        The fixed compressed-sparse-row (CSR) structure -- a row-pointer array plus a flat column-index
        array -- together with the ordering that scatters the probe responses into it, so a materialize
        de-compresses by one gather rather than a scatter loop and a re-sort.
    narrowing : callable or None
        ``assembler -> assembler``: the stand-in whose Jacobian is materialized in place of the
        assembler itself, or ``None`` to materialize the assembler as it stands. It is how a residual
        keeps the recovered matrix exact for the operator the Krylov solve applies (or collision-free
        for the pattern the colouring was built at) without the probe knowing what the stand-in is. It
        is compared by value, so give it a frozen dataclass rather than a closure.
    """

    plan: ColumnProbePlan
    structure: ProbeGather
    narrowing: Callable | None = None

    @classmethod
    def build(
        cls,
        face_cells,
        n_cells: int,
        n_fields: int,
        stencil_reach: int = 3,
        column_reach: Sequence[int] | None = None,
        *,
        active_rows: np.ndarray | None = None,
        narrowing: Callable | None = None,
    ) -> JacobianProbe:
        """Colour the cell graph at these reaches and precompute the de-compression for it.

        Parameters
        ----------
        face_cells : FaceCellConnectivity
            The mesh's face-to-cell connectivity. Any companion of the same case (a continuation rung
            at a scaled viscosity) has the same one, so gives the same probe.
        n_cells : int
            Number of cells.
        n_fields : int
            Number of fields in the flat state layout.
        stencil_reach : int
            The cell-graph distance the assembled sparsity covers.
        column_reach : sequence of int, optional
            A shorter reach per **column field**, in the flat layout's order, while the assembled
            pattern stays at ``stencil_reach``. Exact only for a column that genuinely carries nothing
            further out; measure it for the case rather than assuming it. ``None`` (default) probes
            every column at ``stencil_reach``.
        active_rows : np.ndarray, optional
            Exclude field-pair blocks from the materialized pattern entirely -- for a probe built
            specifically to feed one consumer that is known never to read some sub-block of the
            Jacobian, such as a :class:`~aquaflux.solve.BlockTriangularFieldSplit`'s dropped triangle
            (:meth:`~aquaflux.solve.FieldGroups.active_rows`). ``None`` (default, and the only sound
            choice for a probe that might be shared with a monolithic consumer) wants every block.
        narrowing : callable, optional
            The assembler stand-in; see the class.

        Returns
        -------
        JacobianProbe
            The shared probe.
        """
        plan = jacobian_probe_plan(
            face_cells, n_cells, n_fields, stencil_reach, column_reach, active_rows
        )
        return cls(plan, block_stencil_gather_map(plan), narrowing)

    def narrow(self, assembler):
        """The assembler this probe differentiates -- ``assembler``, or its stand-in.

        Passed per call rather than held, because a refresh hook is rebound across continuation rungs.
        Every consumer -- the initial build, the refresh hook, and the rebind across a rung -- lands
        here, which is what keeps them all materializing the same operator.

        Parameters
        ----------
        assembler : object
            The assembler whose Jacobian is being materialized.

        Returns
        -------
        object
            ``narrowing(assembler)``, or ``assembler`` itself when the probe has none.
        """
        return assembler if self.narrowing is None else self.narrowing(assembler)
