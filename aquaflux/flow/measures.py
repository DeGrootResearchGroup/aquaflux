"""What the coupled flow residual can be measured in: the row-scaled and block-scaled residual norms.

A pressure--velocity residual mixes momentum rows, whose size follows the momentum diagonal, with a
continuity row, which is a mass imbalance and has no diagonal at all. The plain Euclidean norm of it is
therefore dominated by whichever block happens to be largest, and a step that damages another block can
be accepted. The row-equilibrated measure divides each row by its own scale and each block by its field's
magnitude, so it reports a fractional change per equation.

The flow block's part of that measure is defined here, once, because a coupled system that carries the
flow state as a sub-state (the k--omega SST solve) measures its flow rows in exactly the same way and
appends its own.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from aquaflux.solve import BlockScaledNorm, NewtonStrategy, RowScaledNorm, block_reference_scales

from .momentum import MomentumContinuity

__all__ = ["FlowMeasures", "flow_row_scales"]

#: Keeps a row scale strictly positive so it can divide a residual row.
_TINY = 1e-300


def flow_row_scales(
    momentum: MomentumContinuity, flow_diagonal: jnp.ndarray, flow_state: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The flow block's row scales and field scales at ``flow_state``.

    * **Momentum rows** are scaled by the pseudo-transient shift's base diagonal, which is each row's own
      diagonal coefficient and so cannot drift from it. The base diagonal, not the shift: the strength
      ``beta`` is a solver setting, and folding it into the scale would make the measure move with its
      own damping.
    * **The continuity row** has no diagonal (it is a constraint, so the shift leaves it at zero). Its
      residual is a mass imbalance, and the natural scale in the same units is the cell's mass
      throughput ``sum_f max(mdot_f, 0)``. Dividing by it needs no pressure difference, so it stays well
      posed on a periodic or closed domain where a pressure scale degenerates.
    * **Field scales** are the mean velocity magnitude (zero from rest, where a row-scaled measure is undefined) for each velocity component and one for
      continuity, which is already dimensionless once divided by the throughput.

    Parameters
    ----------
    momentum : MomentumContinuity
        The flow assembler, for the layout and the momentum diagonal's buckets.
    flow_diagonal : jnp.ndarray
        The shift's base diagonal over the flow state, shape ``((dim + 1) n_cells,)``; zero on pressure.
    flow_state : jnp.ndarray
        The flow state ``[vel_0..vel_{dim-1}, pressure]``, shape ``((dim + 1) n_cells,)``.

    Returns
    -------
    tuple of jnp.ndarray
        ``(row_scale, field_scale)``: the per-row scale over the flow state, shape
        ``((dim + 1) n_cells,)``, and one field scale per flow field, shape ``(dim + 1,)``.
    """
    velocity_diagonal, _pressure_diagonal = momentum.unpack(flow_diagonal)
    velocity, _pressure = momentum.unpack(flow_state)
    # Continuity's stand-in diagonal: the convective bucket is the per-cell mass throughput, in the same
    # units as the mass imbalance the row measures.
    throughput, _dissipative = momentum.momentum_matrix_diagonal_parts(velocity)
    row_scale = momentum.pack(jnp.abs(velocity_diagonal) + _TINY, jnp.abs(throughput) + _TINY)
    field_scale = jnp.concatenate(
        [jnp.full((momentum.mesh.dim,), jnp.mean(jnp.abs(velocity))), jnp.ones((1,))]
    )
    return row_scale, field_scale


class FlowMeasures(eqx.Module):
    """The measures of the coupled flow residual: row-scaled and block-scaled as well as Euclidean.

    Supplied to a solve's :class:`~aquaflux.solve.Convergence`, which builds the measure it names
    against this (:class:`~aquaflux.solve.ResidualMeasures`).

    Attributes
    ----------
    momentum : MomentumContinuity
        The flow assembler whose residual is measured, with its gradients stopped.
    """

    momentum: MomentumContinuity

    def row_scaled(self, step: NewtonStrategy, state: jnp.ndarray) -> RowScaledNorm:
        """The row-equilibrated measure at ``state``, its row diagonals the step's own shift base."""
        diagonal = jax.lax.stop_gradient(step.shift_policy.shift_term(state).diagonal)
        row_scale, field_scale = flow_row_scales(self.momentum, diagonal, state)
        return RowScaledNorm(
            sizes=(self.momentum.mesh.n_cells,) * self.momentum.layout.n_fields,
            row_scale=jax.lax.stop_gradient(row_scale),
            field_scale=jax.lax.stop_gradient(field_scale),
        )

    def block_scaled(self, state: jnp.ndarray) -> BlockScaledNorm:
        """Each block of the residual divided by its own magnitude at ``state``."""
        layout = self.momentum.layout
        return BlockScaledNorm(
            layout.sizes, block_reference_scales(layout, self.momentum.residual(state))
        )
