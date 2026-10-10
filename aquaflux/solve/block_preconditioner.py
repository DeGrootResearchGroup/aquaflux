"""A materialized-Jacobian preconditioner fitted with one block inverse over the whole state.

The field split fits a separate inverse to each of two groups of fields. A problem whose state is a
**single** group -- a laminar flow, whose whole ``(u, p)`` state is the pressure-velocity saddle a
:class:`~aquaflux.solve.SimpleSmoothed` hierarchy is written for -- has nothing to split, and wants that
inverse over the entire materialized Jacobian. :class:`MaterializedBlockPreconditioner` is exactly that:
the coloured-probe materialization, the shift and the in-place refresh the split shares
(:class:`~aquaflux.solve.MaterializedJacobianPreconditioner`), around one injected block inverse.

The inverse is the traced hierarchy family (:mod:`~aquaflux.solve.hierarchy_inverse`), so this needs no
optional dependency, unlike the monolithic V-cycle, which is PETSc's multigrid.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from .amg_preconditioner import MaterializedJacobianPreconditioner
from .host_preconditioner import HostFactors
from .sparse_jacobian import ProbeGather

__all__ = ["MaterializedBlockPreconditioner"]


class MaterializedBlockPreconditioner(MaterializedJacobianPreconditioner):
    """One block inverse fitted to the whole materialized, shifted Jacobian.

    Sibling of :class:`~aquaflux.solve.FieldSplitAmgPreconditioner` over the shared
    :class:`~aquaflux.solve.MaterializedJacobianPreconditioner` base, for a state with a single group of
    fields; the materialization, the shift and the in-place refresh are all the base's. The frozen
    inverse is the injected block inverse itself, which must be a :class:`~aquaflux.solve.HostFactors`,
    be a fixed linear map (the outer Krylov solve is not flexible) and transpose exactly (the adjoint's
    solve uses it).

    .. warning::
       ``refresh_in_place`` is forward-march only, for the same reason as the split's: the mutation is
       impure and would corrupt an adjoint transpose solve that read the inverse between its own calls.
    """

    @classmethod
    def build(
        cls,
        matvec: Callable,
        plan,
        shift_diagonal: np.ndarray,
        *,
        inverse: Callable[[sp.csr_matrix, int], HostFactors],
        n_fields: int,
        batched_matvec: Callable | None = None,
        probe_batch_size: int | None = None,
        structure: ProbeGather | None = None,
    ) -> MaterializedBlockPreconditioner:
        """Materialize the Jacobian, shift it, and fit ``inverse`` to it.

        Parameters
        ----------
        matvec, plan, batched_matvec, probe_batch_size, structure
            The coloured-probe materialization, exactly as the monolithic and split builds take them.
        shift_diagonal : np.ndarray
            The pseudo-transient shift ``beta d`` added to the diagonal, shape ``(n_dofs,)``.
        inverse : callable
            ``(matrix, n_fields) -> inverse``, for example a :class:`~aquaflux.solve.SimpleSmoothed`. An
            injected inverse must be a :class:`~aquaflux.solve.RefactorableFactors` to survive a
            mid-march refresh.
        n_fields : int
            The number of fields in the flat field-major state.

        Returns
        -------
        MaterializedBlockPreconditioner
            The frozen preconditioner.
        """
        jacobian = cls._materialize_jacobian(
            matvec, plan, batched_matvec, probe_batch_size, structure
        )
        return cls(inverse(cls._shifted(jacobian, shift_diagonal), n_fields))
