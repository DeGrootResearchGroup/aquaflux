"""The coupled Reynolds-averaged state's residual measure, and the names of the equations it measures.

The row-equilibrated measure a coupled march steers and stops by, built at each state the march asks
about, with its blocks named by :func:`coupled_equation_names` -- the one home for those names, so the
march's per-equation residuals, a per-block residual report and a per-field change all label the same
equation alike. The flow's own measure and names are :mod:`aquaflux.flow`'s, which these extend by the
closure's ``k`` and ``omega``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from aquaflux.flow import flow_equation_names, flow_row_scales
from aquaflux.solve import RowScaledNorm

if TYPE_CHECKING:
    from .coupled import CoupledRANS, CoupledShiftPolicy

__all__ = ["coupled_equation_names", "coupled_scaled_norm"]


def coupled_equation_names(dim: int) -> tuple[str, ...]:
    """The coupled state's solved equations, named, **in the order the flat state lays them out**.

    ``(u, v, w, p, k, omega)`` in three dimensions and ``(u, v, p, k, omega)`` in two: one name per
    equal-sized block of the flat layout ``[vel_0..vel_{dim-1}, pressure, k, omega]`` -- the flow's
    own names (:func:`~aquaflux.flow.flow_equation_names`) then the closure's. The row-scaled measure,
    a per-block residual and a per-field change all report the same equation under the same label.

    Parameters
    ----------
    dim : int
        Number of velocity components (the spatial dimension), at most 3.

    Returns
    -------
    tuple of str
        The ``dim + 3`` block names, in block order.

    Raises
    ------
    ValueError
        If ``dim`` exceeds the three named velocity components.

    Examples
    --------
    >>> coupled_equation_names(3)
    ('u', 'v', 'w', 'p', 'k', 'omega')
    >>> coupled_equation_names(2)
    ('u', 'v', 'p', 'k', 'omega')
    """
    return (*flow_equation_names(dim), "k", "omega")


def coupled_scaled_norm(
    coupled: CoupledRANS,
    shift_policy: CoupledShiftPolicy,
    state: jnp.ndarray,
) -> RowScaledNorm:
    """Build the row-equilibrated residual measure for the coupled state at ``state``.

    Assembles the two scales :class:`~aquaflux.solve.RowScaledNorm` needs, per block of the coupled
    layout ``[vel_0..vel_{dim-1}, pressure, k, omega]``:

    * **Row scale** -- each row's own diagonal coefficient, taken from the pseudo-transient shift's
      base diagonal, which is exactly that quantity per block (the momentum ``a_P`` on velocity, the
      transport diagonal on ``k`` and ``omega``) and so cannot drift from it. ⚠️ **The base diagonal,
      not the shift** -- the strength ``beta`` and any per-block multiplier on it
      (:attr:`CoupledShiftPolicy.turbulence_damping`) are solver settings, and folding one into the row
      scale would divide that block's reported residual by it: the march would be steered, stopped and
      compared on a measure that moves with its own damping, so two damping settings could not be
      compared at all. Two rows are not covered by the diagonal and are supplied here:

      - **Continuity carries no diagonal** -- it is a constraint, so the shift leaves it at zero. Its
        residual is a mass imbalance, and the natural scale of the same units is the cell's mass
        throughput ``sum_f max(mdot_f, 0)``. Dividing by it needs no pressure difference, so it stays
        well posed on a periodic or closed domain where a pressure scale degenerates.
      - **The near-wall fixed ``omega`` rows** hold an algebraic constraint rather than a balance, and
        the shift zeroes them. Their derivative comes from the row itself
        (:meth:`~aquaflux.discretization.FixationRow.jacobian_scale`) -- one, for a fixation written in
        the solved variable -- so they pass through unscaled rather than being divided by a
        neighbouring transport row's diagonal, which would misreport them by orders of magnitude.

    * **Field scale** -- ``mean(phi / (dphi/dw))``, which turns the stage-1 quotient (a change in the
      *solved* unknown) into a fractional change in the *physical* field. For a directly-solved field
      this is the familiar ``mean|phi|``; for a log-solved one ``dphi/dw = phi``, so the scale is
      exactly **one** -- a change in ``log phi`` already *is* a fractional change, and dividing by
      ``mean|phi|`` a second time would be wrong. Continuity likewise takes one, being dimensionless
      after stage 1.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled assembler, for the layout and the physical fields.
    shift_policy : ShiftPolicy
        Any policy whose base shift diagonal supplies the per-row diagonals -- the block
        :class:`CoupledShiftPolicy`, or a :class:`MonolithicFactorShiftPolicy` wrapping one.
    state : jnp.ndarray
        The coupled state the scales are measured at, shape ``((dim + 3) n_cells,)``.

    Returns
    -------
    RowScaledNorm
        The measure, with its scales frozen at ``state``. Only the block ``sizes`` are static, so the
        scales ride as ordinary array leaves and **re-deriving the measure at a new state is a
        compilation cache hit** -- the block structure is unchanged, only the numbers move. Rebuild it
        every outer iteration, and hold it fixed across a line search (otherwise a candidate could be
        preferred for shrinking its own denominator rather than its residual).
    """
    layout = coupled.layout
    n, dim = layout.n_cells, coupled.momentum.mesh.dim
    tiny = 1e-300

    diagonal = jax.lax.stop_gradient(shift_policy.shift_term(state).diagonal)
    flow_diag, k_diag, omega_diag = layout.unpack(diagonal)

    flow, k, omega = coupled.physical_fields(state)
    flow_row_scale, flow_field_scale = flow_row_scales(coupled.momentum, flow_diag, flow)

    k_chain = coupled.k_transform.jacobian_scale(k)
    omega_chain = coupled.omega_transform.jacobian_scale(omega)
    # A zeroed shift entry marks a row the shift does not own -- the fixed near-wall omega cells. Ask
    # the fixation row for its own derivative there instead of borrowing a transport row's.
    omega_fixed = coupled.omega_transform.fixation_row().jacobian_scale(omega, omega_chain)
    omega_rows = jnp.where(omega_diag > 0.0, omega_diag, omega_fixed)
    k_rows = jnp.where(
        k_diag > 0.0, k_diag, coupled.k_transform.fixation_row().jacobian_scale(k, k_chain)
    )

    row_scale = layout.pack(flow_row_scale, jnp.abs(k_rows) + tiny, jnp.abs(omega_rows) + tiny)
    field_scale = jnp.concatenate(
        [
            flow_field_scale,
            # phi / (dphi/dw) converts a change in the solved unknown into a fractional change in the
            # physical field: mean|phi| for a directly-solved field, exactly one for a log-solved one.
            jnp.mean(jnp.abs(k) / jnp.maximum(k_chain, tiny))[None],
            jnp.mean(jnp.abs(omega) / jnp.maximum(omega_chain, tiny))[None],
        ]
    )
    return RowScaledNorm(
        sizes=(n,) * (dim + 3),
        row_scale=jax.lax.stop_gradient(row_scale),
        field_scale=jax.lax.stop_gradient(field_scale),
        names=coupled_equation_names(dim),
    )
