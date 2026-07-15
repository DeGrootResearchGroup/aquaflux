"""The transient (accumulation) term of the cell residual.

The time derivative ``dphi/dt`` integrated over a cell contributes ``V_P dphi/dt`` to the
residual. It is discretized with backward differentiation formulae: first-order backward
Euler (BDF1) at the first timestep, where no earlier history exists, and second-order
backward Euler (BDF2) thereafter, for a constant timestep ``dt``:

    BDF1:  V_P (phi^n - phi^{n-1}) / dt
    BDF2:  V_P (3/2 phi^n - 2 phi^{n-1} + 1/2 phi^{n-2}) / dt

This is the accumulation half of ``R = accumulation - transport``; it vanishes for a
steady problem. It carries no physical coefficient (the diffusivity lives on the flux
side); a heat-capacity/porosity coefficient, when needed, is supplied externally as a
per-cell array, never baked in here.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class TransientTerm(eqx.Module):
    """Backward-difference accumulation term ``V_P dphi/dt`` (BDF1 then BDF2).

    ``first_step`` is a static flag chosen by the time-integration driver: the very first
    step has no ``phi^{n-2}`` and must fall back to BDF1. Keeping it static means the
    unused ``phi_older`` branch is never traced, so the driver may pass a dummy
    ``phi_older`` on step one.
    """

    def residual(
        self,
        phi: jnp.ndarray,
        phi_old: jnp.ndarray,
        phi_older: jnp.ndarray,
        dt: float,
        first_step: bool,
        volume: jnp.ndarray,
    ) -> jnp.ndarray:
        """Per-cell accumulation contribution to the residual, shape ``(n_cells,)``.

        Parameters
        ----------
        phi : jnp.ndarray
            Current-step cell field ``phi^n``, shape ``(n_cells,)``.
        phi_old : jnp.ndarray
            Previous step ``phi^{n-1}``, shape ``(n_cells,)``.
        phi_older : jnp.ndarray
            Step before that ``phi^{n-2}``, shape ``(n_cells,)``; ignored when
            ``first_step`` is ``True`` (may be a dummy).
        dt : float
            Timestep size (constant).
        first_step : bool
            ``True`` for the first timestep (use BDF1), ``False`` afterwards (use BDF2).
            Static: selected by a Python branch, not traced.
        volume : jnp.ndarray
            Cell volumes ``V_P``, shape ``(n_cells,)``.
        """
        if first_step:
            return volume * (phi - phi_old) / dt
        return volume * (1.5 * phi - 2.0 * phi_old + 0.5 * phi_older) / dt
