"""The shared base of every preconditioner fitted to the coloured-probe materialized Jacobian.

A materialized-Jacobian preconditioner probes the coupled Jacobian with graph-coloured
Jacobian-vector products, adds the pseudo-transient shift to its diagonal, and fits a frozen host inverse
to the result: the block-triangular field split
(:class:`~aquaflux.solve.field_split.FieldSplitPreconditioner`) and the single-block inverse of a state
whose fields form one group (:class:`~aquaflux.solve.block_preconditioner.MaterializedBlockPreconditioner`).
:class:`MaterializedJacobianPreconditioner` holds what they share -- the probe, the shift, and the in-place
refresh -- on top of :class:`~aquaflux.solve.HostPreconditioner`'s apply.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .host_preconditioner import HostPreconditioner, require_refactorable
from .refresh_timing import PhaseTimer

if TYPE_CHECKING:
    from .sparse_jacobian import ProbeGather


class MaterializedJacobianPreconditioner(HostPreconditioner):
    """Shared machinery for a preconditioner fitted to the coloured-probe materialized coupled Jacobian.

    What every member needs and nothing more: probing the Jacobian, adding the pseudo-transient shift,
    and the refresh, which re-fits the frozen inverse to the shifted field-major matrix as it stands.
    The inverse itself is the subclass's, built by its own ``build`` and reached through
    :attr:`~aquaflux.solve.HostPreconditioner.inverse`.
    """

    @staticmethod
    def _materialize_jacobian(
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        plan,
        batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        probe_batch_size: int | None = None,
        structure: ProbeGather | None = None,
    ) -> sp.csr_matrix:
        """The coupled Jacobian **without** the shift, from the graph-coloured jvp probe (one jvp per
        (colour, field) -- the expensive part of a refresh; e.g. ~670 probes on a 23k-cell reach-3 bfs3d
        mesh). ``batched_matvec`` (built once, reused) runs the probes as a few batched passes rather than a
        per-probe loop (~1.6x on that mesh); ``probe_batch_size`` chunks the batch for memory. ``structure``
        (a ``ProbeGather`` from ``block_stencil_gather_map``, built once) fills the fixed full-pattern CSR
        one probe-chunk at a time instead of a scatter loop + re-sort (it keeps explicit zeros)."""
        from .sparse_jacobian import materialize_block_jacobian

        return materialize_block_jacobian(
            matvec,
            plan,
            batched_matvec=batched_matvec,
            probe_batch_size=probe_batch_size,
            structure=structure,
        ).tocsr()

    @staticmethod
    def _shifted(jacobian_no_shift: sp.csr_matrix, shift_diagonal: np.ndarray) -> sp.csr_matrix:
        """The Jacobian with the pseudo-transient shift on its diagonal.

        Delegates to :func:`~aquaflux.solve.sparse_jacobian.shifted_jacobian`, which is the one home for
        the shift across the whole host-preconditioner family -- it lives beside the probe that produced
        the pattern, because keeping that pattern intact is the whole point of the spelling it uses.
        Kept as a method because the field split and two tests reach for it here.
        """
        from .sparse_jacobian import shifted_jacobian

        return shifted_jacobian(jacobian_no_shift, shift_diagonal)

    def refresh_in_place(
        self,
        matvec: Callable[[jnp.ndarray], jnp.ndarray],
        plan,
        shift_diagonal: np.ndarray,
        *,
        batched_matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        probe_batch_size: int | None = None,
        structure: ProbeGather | None = None,
    ) -> tuple[tuple[str, float], ...]:
        """Re-materialize at a developed state and re-fit the frozen inverse to it, IN PLACE.

        The arguments are the build's materialization arguments, evaluated at the developed state. The
        inverse's own configuration is fixed when it is built and cannot be changed by a refresh.
        Because this preconditioner is held as a **static field** of the shift policy and
        :meth:`~aquaflux.solve.HostPreconditioner.matvec` reads ``self.inverse`` at call time, re-fitting
        the inverse here re-preconditions the **same compiled** Krylov solve.

        **Forward-march use ONLY -- the mutation is impure and must never touch a differentiated path.**
        The adjoint's transpose solve reads the same inverse and would be corrupted by a change between
        its calls; only the eager, non-differentiated march may refresh. The refresh never moves the
        converged root (the shift vanishes there), so it changes only the forward Krylov path.

        Returns
        -------
        tuple of (str, float)
            ``("probe", s), ("assemble", s), ("refactor", s)`` -- the coloured jvp probe, the shift, and
            the inverse's re-fit -- so a march log can say *where* a refresh spent its time.

        Raises
        ------
        TypeError
            If the inverse is not a :class:`~aquaflux.solve.RefactorableInverse` (an injected inverse need
            not be refreshable).
        """
        inverse = require_refactorable(self.inverse, f"this {type(self).__name__}")
        timer = PhaseTimer()
        jacobian = self._materialize_jacobian(
            matvec, plan, batched_matvec, probe_batch_size, structure
        )
        timer.lap("probe")
        shifted = self._shifted(jacobian, shift_diagonal)
        timer.lap("assemble")
        inverse.refactor_block(shifted)
        timer.lap("refactor")
        return timer.phases()
