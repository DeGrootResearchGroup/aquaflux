"""Freezing the k-production's eddy viscosity in the Krylov operator: everywhere, or per cell.

``nu_t`` is proportional to ``k``, so the k-production is too, and differentiating it subtracts from the k
row's own Jacobian diagonal. Where ``k`` is small -- a stagnant wall corner, where the pseudo-time shift is
small as well -- that term can outweigh everything else on the row, and the Newton correction then points
at negative ``k``. A copy of the coupled residual with ``k`` frozen inside the production's eddy viscosity
has the same value and a Jacobian without that term; handed to a shifted solve as the operator it
differentiates, it changes how the march travels and not where it lands, so the root and the
implicit-function-theorem adjoint are untouched.

:func:`frozen_production_viscosity` freezes every cell. :func:`with_frozen_production_cells` freezes chosen
cells, and :func:`production_freeze_stand_in` chooses them at a materialized-Jacobian re-fit from the
k rows' shifted diagonal (:func:`production_freeze_cells`), returning the stand-in together with the
diagonal its Jacobian differs by, so the inverse can be fitted to the operator actually solved.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.precondition import DiagonalStandIn

if TYPE_CHECKING:
    from .coupled import CoupledRANS

#: Why a per-cell freeze is refused beside a narrowed Jacobian: each re-fit's stand-in becomes the Krylov
#: operator, which would discard the narrowed copy.
FREEZE_DISCARDS_NARROWING = (
    "production_freeze_threshold makes each re-fit's stand-in the Krylov operator, which would discard "
    "jacobian_gradient_sweeps' narrowed copy; set one or the other."
)


def frozen_production_viscosity(coupled: CoupledRANS) -> CoupledRANS:
    """``coupled`` with ``k`` frozen inside the k-production's eddy viscosity.

    The Patankar treatment of the one term that drives the k row's Jacobian diagonal negative:
    ``nu_t`` is proportional to ``k``, so the production ``nu_t S**2`` is too, and subtracting a
    source that grows with its own variable puts a negative term on that row's diagonal. See
    :attr:`~aquaflux.turbulence.SSTTurbulence.explicit_production_viscosity` for what that costs and
    what freezing it buys.

    ⚠️ **The copy is for a Jacobian stand-in only** -- pass its ``residual`` as the operator a shifted
    solve differentiates, so the true residual still decides where the march lands and the converged
    root and its implicit-function-theorem adjoint are untouched. This term is active everywhere, not
    only where some cap bites, so a *residual* carrying it would make the sensitivity wrong
    everywhere.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled coupled system.

    Returns
    -------
    CoupledRANS
        A copy whose turbulence assembler carries the flag. ``coupled`` itself is unchanged.
    """
    # `dataclasses.replace`, not `eqx.tree_at`: the flag is a STATIC field, so it lives in the
    # treedef rather than among the leaves and `tree_at` cannot address it at all.
    return dataclasses.replace(
        coupled,
        turbulence=dataclasses.replace(coupled.turbulence, explicit_production_viscosity=True),
    )


def with_frozen_production_cells(coupled: CoupledRANS, cells: jnp.ndarray) -> CoupledRANS:
    """``coupled`` with ``k`` frozen inside the k-production's eddy viscosity in ``cells`` only.

    The per-cell form of :func:`frozen_production_viscosity`: a cell whose entry in ``cells`` is
    positive drops production's ``k``-derivative from its own k row, every other cell keeps the exact
    derivative, and the residual's value is unchanged everywhere. The production in a cell reads that
    cell's ``k`` alone, so what this changes in the Jacobian is the k rows' diagonal at ``cells`` and
    nothing off it. For a Jacobian stand-in only, as :func:`frozen_production_viscosity` explains.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled coupled system.
    cells : jnp.ndarray
        Shape ``(n_cells,)``; positive where production is frozen. An array leaf, so copies with
        different cells share one compilation.

    Returns
    -------
    CoupledRANS
        The copy. ``coupled`` itself is unchanged.
    """
    return dataclasses.replace(
        coupled,
        turbulence=dataclasses.replace(
            coupled.turbulence, production_freeze_mask=jnp.asarray(cells, dtype=jnp.float64)
        ),
    )


@eqx.filter_jit
def production_diagonal_gain(coupled: CoupledRANS, state: jnp.ndarray) -> jnp.ndarray:
    """What freezing production in each cell adds to its own k row's Jacobian diagonal, at ``state``.

    ``diag(J_frozen) - diag(J)`` on the k rows, shape ``(n_cells,)``, where ``J_frozen`` is the Jacobian
    of :func:`frozen_production_viscosity`'s copy. It is minus production's derivative in the cell's own
    ``k`` through the eddy viscosity -- positive wherever production grows with ``k``. Because each
    cell's production reads that cell's ``k`` alone, the two Jacobians differ only on that diagonal, so
    one Jacobian-vector product of each with every k entry seeded at once gives it exactly.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled coupled system; an argument of the compiled program, so a re-scaled companion is
        a cache hit.
    state : jnp.ndarray
        The coupled state, shape ``(n_dofs,)``.

    Returns
    -------
    jnp.ndarray
        Shape ``(n_cells,)``.
    """
    k = coupled.layout.slice_of("k")
    tangent = jnp.zeros_like(state).at[k.start : k.stop].set(1.0)
    exact = jax.jvp(coupled.residual, (state,), (tangent,))[1]
    frozen = jax.jvp(frozen_production_viscosity(coupled).residual, (state,), (tangent,))[1]
    return (frozen - exact)[k.start : k.stop]


def production_freeze_cells(
    k_diagonal: np.ndarray,
    production_gain: np.ndarray,
    k_shift: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """The cells whose k row production's own derivative has all but cancelled, or turned negative.

    A cell is frozen when its shifted k-row diagonal is below ``threshold`` times what it would be with
    production's derivative removed::

        J_kk + s  <  threshold * (J_kk + gain + s)

    ``nu_t`` is proportional to ``k``, so production is too, and differentiating it subtracts from the
    k row's diagonal. Where ``k`` is small that term can outweigh everything else on the row -- in a
    stagnant wall corner, where the pseudo-time shift is also small -- and the Newton correction then
    points at negative ``k``. Freezing production there keeps the row's diagonal what the destruction
    and the transport make it. Only a cell freezing would raise is chosen (``gain > 0``).

    Parameters
    ----------
    k_diagonal : np.ndarray
        The k rows' Jacobian diagonal without the shift, shape ``(n_cells,)``.
    production_gain : np.ndarray
        :func:`production_diagonal_gain`, shape ``(n_cells,)``.
    k_shift : np.ndarray
        The pseudo-time shift on the k rows, shape ``(n_cells,)``.
    threshold : float
        In ``[0, 1)``; ``0`` freezes only a row production has already driven negative.

    Returns
    -------
    np.ndarray
        Boolean, shape ``(n_cells,)``.
    """
    shifted = np.asarray(k_diagonal) + np.asarray(k_shift)
    gain = np.asarray(production_gain)
    return (gain > 0) & (shifted < threshold * (shifted + gain))


def production_freeze_stand_in(
    coupled: CoupledRANS,
    state: jnp.ndarray,
    jacobian_diagonal: np.ndarray,
    shift_diagonal: np.ndarray,
    threshold: float,
) -> DiagonalStandIn:
    """Production frozen where the k row needs it, chosen from the Jacobian a re-fit materialized.

    The cells are :func:`production_freeze_cells` at ``threshold``, read from the materialized k-row
    diagonal and the shift the inverse is fitted with; the stand-in is :func:`with_frozen_production_cells`
    on them, whose Jacobian is the materialized one plus :func:`production_diagonal_gain` on those rows. It
    is returned even when no cell is frozen, so a step keeps one structure from re-fit to re-fit.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembler the re-fit was taken on.
    state : jnp.ndarray
        The re-fit's state, shape ``(n_dofs,)``.
    jacobian_diagonal, shift_diagonal : np.ndarray
        The materialized Jacobian's diagonal without the shift, and the shift, both shape ``(n_dofs,)``.
    threshold : float
        See :func:`production_freeze_cells`.

    Returns
    -------
    DiagonalStandIn
        The stand-in residual and its diagonal change, shape ``(n_dofs,)``, zero off the frozen k rows.
    """
    k = coupled.layout.slice_of("k")
    gain = np.asarray(production_diagonal_gain(coupled, state))
    cells = production_freeze_cells(
        jacobian_diagonal[k.start : k.stop], gain, shift_diagonal[k.start : k.stop], threshold
    )
    diagonal = np.zeros_like(np.asarray(jacobian_diagonal))
    diagonal[k.start : k.stop] = np.where(cells, gain, 0.0)
    return DiagonalStandIn(
        residual=with_frozen_production_cells(coupled, cells).residual, diagonal=diagonal
    )


def refuse_an_unusable_production_freeze(
    threshold: float | None, *, materialized: bool, freezes_everywhere: bool
) -> None:
    """Refuse a ``production_freeze_threshold`` out of range or beside settings it cannot act with.

    Parameters
    ----------
    threshold : float or None
        The setting; ``None`` is always accepted.
    materialized : bool
        Whether the preconditioner materializes the Jacobian the cells are chosen from.
    freezes_everywhere : bool
        Whether production is already frozen in every cell (``jacobian_production_viscosity``).

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``[0, 1)`` or production is already frozen everywhere.
    TypeError
        If the preconditioner materializes no Jacobian.
    """
    if threshold is None:
        return
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"production_freeze_threshold must be in [0, 1), got {threshold!r}.")
    if freezes_everywhere:
        raise ValueError(
            "production_freeze_threshold freezes production in the cells that need it, and "
            "jacobian_production_viscosity already freezes it in every cell; choose one."
        )
    if not materialized:
        raise TypeError(
            "production_freeze_threshold chooses its cells from the Jacobian a materialized-Jacobian "
            "session probes at each re-fit, and this preconditioner materializes none; use "
            "MaterializedJacobian(...)."
        )
