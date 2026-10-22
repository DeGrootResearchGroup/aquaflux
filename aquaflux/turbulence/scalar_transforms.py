"""The change of variable between a solved turbulence unknown and the physical ``k`` / ``omega``.

The coupled RANS residual is written in the physical fields; the Newton iterate may hold a transformed
unknown instead (the logarithm, so the field stays positive under any step). Each transform also says
how an algebraic value fixation on its field is written as a residual row, since only it knows which
variable is solved for.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp

from aquaflux.discretization import DifferenceRow, FixationRow, LogRatioRow

__all__ = ["DirectScalars", "LogScalars", "ScalarVariableTransform"]


class ScalarVariableTransform(eqx.Module):
    """Strategy: the change of variable between the *solved* turbulence unknown and the physical
    ``k`` / ``omega`` the closure needs.

    The coupled Newton solves for a per-cell scalar unknown ``w``; the closure and transport physics
    are always written in the physical field ``phi = to_physical(w)``. A strategy that maps ``w`` onto
    a strictly positive ``phi`` therefore makes ``k, omega > 0`` hold **by construction under any Newton
    step**, which is what the direct (identity) parametrization cannot guarantee: a full step there can
    drive ``omega`` negative, and ``nu_t = k / omega`` then flips sign without the residual going
    non-finite, so the divergence guard never catches it.

    Because the physics residual is written in ``phi``, its Jacobian with respect to the solved ``w``
    picks up the chain-rule factor ``d(phi)/d(w) = jacobian_scale(phi)``. The frozen scalar
    preconditioner and pseudo-transient shift are assembled for the *physical* operator, so they are
    rescaled by this factor to precondition the reparametrized block (see
    :func:`~aquaflux.turbulence.coupled_step`).
    """

    @abc.abstractmethod
    def to_physical(self, w: jnp.ndarray) -> jnp.ndarray:
        """Map the solved unknown ``w`` to the physical field ``phi`` (shape preserved)."""

    @abc.abstractmethod
    def to_solved(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Map a physical field ``phi`` to the solved unknown ``w`` (the inverse of
        :meth:`to_physical`)."""

    @abc.abstractmethod
    def jacobian_scale(self, phi: jnp.ndarray) -> jnp.ndarray:
        """``d(phi)/d(w)`` evaluated at physical ``phi`` -- the factor the physical operator's rows are
        scaled by to precondition/shift the reparametrized block."""

    @abc.abstractmethod
    def fixation_row(self) -> FixationRow:
        """How an algebraic value fixation on this field should be written as a residual row.

        A value fixation must be expressed in the **solved** unknown, not the physical field, or its
        linearization inherits the transform's nonlinearity: under ``phi = e**w`` the plain difference
        row ``phi - target`` gives a Newton correction ``dw = target/phi - 1``, which overshoots by
        ``e**(r-1)`` against a target ratio ``r``, while the log-ratio row is linear in ``w`` and lands
        on the constraint in one full step. The transform owns this choice because it is the only
        object that knows which variable is actually being solved for.
        """


class DirectScalars(ScalarVariableTransform):
    """The identity parametrization: the solved unknown *is* the physical field (``phi = w``).

    Positivity is not structural here -- it is carried by the pseudo-transient shift, the divergence
    guard, and (for ``k``) the per-cell clip :func:`~aquaflux.turbulence.positive_k_projection` -- so a
    full Newton step can transiently violate ``omega > 0`` on a stiff high-Reynolds case. The
    historical default; use :class:`LogScalars` where that matters.
    """

    def to_physical(self, w: jnp.ndarray) -> jnp.ndarray:
        return w

    def to_solved(self, phi: jnp.ndarray) -> jnp.ndarray:
        return phi

    def jacobian_scale(self, phi: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(phi)

    def fixation_row(self) -> FixationRow:
        """The plain difference -- the solved unknown *is* the physical field here."""
        return DifferenceRow()


class LogScalars(ScalarVariableTransform):
    """The log parametrization ``phi = e^w`` for both ``k`` and ``omega``.

    ``phi = e^w > 0`` for every real ``w``, so ``k`` and ``omega`` stay strictly positive under **any**
    Newton step -- the structural fix for the direct form's transient negativity at high Reynolds
    number. The physical root is unchanged (``e^w`` is a smooth bijection onto the positives, so
    ``R(e^w) = 0`` has the same solution as ``R(phi) = 0``); only the Newton iterate space changes, and
    at the converged state the realizability floor is inactive, so the coupled adjoint is unaffected.
    The chain-rule factor is ``d(e^w)/d(w) = e^w = phi``.
    """

    def to_physical(self, w: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(w)

    def to_solved(self, phi: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(phi)

    def jacobian_scale(self, phi: jnp.ndarray) -> jnp.ndarray:
        return phi

    def fixation_row(self) -> FixationRow:
        """The log ratio ``log(phi/target) = w - log(target)`` -- linear in the solved unknown ``w``.

        The difference row would be exponential in ``w`` here, so a near-wall cell whose ``omega`` is a
        factor ``r`` from its target takes a correction ``dw = r - 1`` and overshoots to ``phi e**(r-1)``
        instead of landing on ``phi r``. It also writes the row on the scale of ``phi`` (which spans
        orders of magnitude near a wall) rather than of ``w``, which lets a handful of fixation cells
        dominate the residual measure the whole march is judged by.
        """
        return LogRatioRow()
