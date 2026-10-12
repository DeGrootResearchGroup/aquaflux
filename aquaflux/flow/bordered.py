"""A residual held at a bulk velocity, in the form any staged march can drive.

A :class:`~aquaflux.flow.MassFlow` drive makes the body force ``beta`` an unknown of the solve: it is
appended to the state, and the residual is bordered with the constraint row,

    R_aug([w, beta]) = [ R(w; beta) ; c . w - target ],

where ``c`` averages the flow-direction velocity over the volume. Everything a march needs to run that
augmented system -- its residual, its shift policy and preconditioner, its residual measures, its seed
and its answer -- is here, written once for every residual that carries a flow block: the flow alone and
the coupled flow-and-turbulence system border themselves through the same :class:`MassFlowBorder`.

**The multiplier enters linearly, and that is what lets the border wrap any residual.** The force acts as
``R = flux - beta V`` on the flow-direction momentum rows, so the residual at one force is the residual at
another plus ``(beta - beta_0) a``, with the column ``a = dR/dbeta = -V``. :class:`BorderedResidual`
therefore needs only a residual evaluated at the force the assembler carries, and adds the difference:
it never writes ``beta`` into the assembler. That is what lets it border a residual handed over as a bare
callable -- the station a viscosity-ramp homotopy enters, say -- as readily as an assembler's own, and it
gives the identical function: the carried force cancels exactly, so neither the values nor the
derivatives depend on it.

**Preconditioning the augmented system.** ``beta`` is a multiplier, not a function of ``w``, so it is
eliminated one layer down, in the preconditioner: :func:`~aquaflux.precondition.bordered_preconditioner`
extends ``M ~ J^{-1}`` for the unbordered block through the 1x1 Schur complement ``c . M a``, with the
drive saying where ``beta`` sits. Building ``a`` and ``c`` by hand here is legitimate: a preconditioner
is an approximate inverse that changes only how fast the Krylov solve converges, never the solution or
its adjoint.

**Measuring the augmented residual.** The constraint row is a velocity, so the row-scaled measure divides
it by the speed the drive holds -- the fraction by which the bulk velocity misses its target -- beside the
fractional changes of every other equation; the block-scaled measure gives it the leading block's scale.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.precondition import bordered_preconditioner
from aquaflux.solve import (
    BlockScaledNorm,
    ContinuationSource,
    FieldLayout,
    NewtonStrategy,
    ResidualMeasures,
    RowScaledNorm,
    ShiftPolicy,
    ShiftTerm,
)

from .drive import MassFlow

if TYPE_CHECKING:
    from aquaflux.solve import ResidualHomotopy, StepReport

    from .momentum import MomentumContinuity

__all__ = [
    "BULK_VELOCITY",
    "BorderedFlowResidual",
    "BorderedMeasures",
    "BorderedResidual",
    "BorderedShiftPolicy",
    "MassFlowBorder",
    "mass_flow_border",
]

_Matvec = Callable[[jnp.ndarray], jnp.ndarray]

#: The name the constraint row is reported under beside a residual's own equations.
BULK_VELOCITY = "bulk_velocity"


class BorderedResidual(eqx.Module):
    """``residual`` bordered with the bulk-velocity constraint, as a residual of the augmented state.

    A module rather than a closure, so it compares by value: the march compiles its step with the
    residual as an argument, and a closure built per solve would be a new compilation every call.

    Attributes
    ----------
    residual : callable
        ``w -> R(w)``, evaluated at the force ``drive`` carries. Pass a bound method of a module (an
        assembler's ``residual``), whose arrays then ride as dynamic leaves.
    drive : MassFlow
        The drive of the assembler ``residual`` belongs to: its target, its layout, and the force
        ``residual`` was evaluated at.
    fields : FieldLayout
        The unbordered layout ``residual`` works in.
    force, average : jnp.ndarray
        The border column ``a = dR/dbeta`` and row ``c``, shape ``(fields.size,)``
        (:meth:`~aquaflux.flow.MassFlow.constraint_vectors`).
    """

    residual: Callable[[jnp.ndarray], jnp.ndarray]
    drive: MassFlow
    fields: FieldLayout
    force: jnp.ndarray
    average: jnp.ndarray

    def __call__(self, augmented: jnp.ndarray) -> jnp.ndarray:
        """The augmented residual ``[R(w; beta) ; c . w - target]``, shape ``(fields.size + 1,)``."""
        state, beta = self.drive.split(self.fields, augmented)
        shifted = self.residual(state) + (beta - self.drive.force) * self.force
        return self.drive.join(
            self.fields, shifted, jnp.dot(self.average, state) - self.drive.target
        )


class BorderedFlowResidual(eqx.Module):
    """The flow residual bordered by its own drive, as a two-argument residual of the augmented state.

    The assembler arrives as the parameter ``theta`` rather than being captured, so the
    implicit-function-theorem adjoint returns its cotangent and a solve over it is reverse-
    differentiable in the assembler (in its viscosity, say). **Everything read here therefore comes from**
    ``theta``, the border vectors included: a value captured from an outer assembler would be a
    closed-over input of the adjoint's ``custom_vjp``, and differentiating it raises. A module rather
    than a closure, so it compares by value and a solver reused across calls compiles its step once.
    """

    def __call__(self, augmented: jnp.ndarray, theta: MomentumContinuity) -> jnp.ndarray:
        return mass_flow_border(theta, theta.layout).residual(theta.residual)(augmented)


class BorderedShiftPolicy(eqx.Module):
    """A shift policy for the unbordered block, bordered for the augmented state.

    The inner preconditioner is wrapped by :func:`~aquaflux.precondition.bordered_preconditioner`, and
    the shift diagonal gains a zero entry for ``beta``: the constraint row is never shifted.

    **An unshifted constraint row, not a damped one.** The constraint is linear in the fields and holds
    no time derivative, so a shifted step asks the damped fields to reach the target bulk velocity in one
    step and puts the whole correction into the force. That is harmless when the march starts at the
    target -- the drive's plug is the held velocity, so the constraint opens satisfied and each step's
    force correction is only what the momentum rows' own progress requires. Shifting the row by its own
    Schur magnitude (``|c . M_0 a|``, damping the force as the fields are damped) was measured and
    rejected: on a laminar periodic channel it crawled where the unshifted row converged, and a warm
    start diverged under it.

    Attributes
    ----------
    inner : ShiftPolicy
        The policy of the unbordered block; a refresh re-derives it and borders the result again.
    drive : MassFlow
        The constraint, whose layout every split and re-attachment goes through.
    fields : FieldLayout
        The unbordered layout ``inner`` works in.
    force, average : jnp.ndarray
        The border column and row, shape ``(fields.size,)``.
    """

    inner: ShiftPolicy
    drive: MassFlow
    fields: FieldLayout
    force: jnp.ndarray
    average: jnp.ndarray

    def shift_term(self, phi: jnp.ndarray, residual: jnp.ndarray | None = None) -> ShiftTerm:
        """The augmented shift and the bordered preconditioner at ``phi``."""
        fields, drive = self.fields, self.drive
        inner_term = self.inner.shift_term(
            drive.fields_state(fields, phi),
            None if residual is None else drive.fields_state(fields, residual),
        )
        diagonal = drive.join(fields, inner_term.diagonal, jnp.asarray(0.0))

        def make_preconditioner(shift_strength: jnp.ndarray) -> _Matvec:
            inner_matvec = inner_term.make_preconditioner(shift_strength)
            return bordered_preconditioner(
                lambda _state: inner_matvec,
                drive,
                fields,
                self.force,
                self.average,
            )(phi)

        # The constraint row carries a zero shift, so its entry in the per-row shift scale only keeps the
        # augmented vector's shape.
        row_shift_scale = (
            None
            if inner_term.row_shift_scale is None
            else lambda shift_strength: drive.join(
                fields, inner_term.row_shift_scale(shift_strength), jnp.asarray(1.0)
            )
        )
        return ShiftTerm(diagonal, make_preconditioner, row_shift_scale)

    def adjoint_factory(self) -> Callable[[jnp.ndarray], _Matvec]:
        """The ``state -> M`` factory for the adjoint transpose solve (the composition at no shift)."""
        return lambda state: self.shift_term(state).make_preconditioner(jnp.asarray(0.0))


def _inner_step(step: NewtonStrategy) -> NewtonStrategy:
    """``step`` with its bordered shift policy replaced by the unbordered one it wraps."""
    return eqx.tree_at(lambda s: s.shift_policy, step, step.shift_policy.inner)


class BorderedMeasures(eqx.Module):
    """What a bordered residual can be measured in: the unbordered residual's measures, plus the border.

    Attributes
    ----------
    inner : ResidualMeasures
        The measures of the unbordered residual.
    drive : MassFlow
        The constraint, whose held speed scales the border row and whose layout splits the state.
    fields : FieldLayout
        The unbordered layout.
    """

    inner: ResidualMeasures
    drive: MassFlow
    fields: FieldLayout

    def row_scaled(self, step: NewtonStrategy, state: jnp.ndarray) -> RowScaledNorm:
        """The unbordered row-scaled measure, with the constraint row as the fractional miss of the target.

        Raises
        ------
        ValueError
            If the drive holds a bulk velocity of zero, which leaves the border row nothing to be a
            fraction of.
        """
        speed = self.drive.held_speed()
        if speed <= 0.0:
            raise ValueError(
                "RowScaled() measures the constraint row as a fraction of the bulk velocity held, and "
                "this drive holds zero. Use Euclidean() or BlockScaled()."
            )
        inner = self.inner.row_scaled(
            _inner_step(step), self.drive.fields_state(self.fields, state)
        )
        return RowScaledNorm(
            sizes=(*inner.sizes, 1),
            row_scale=jnp.append(inner.row_scale, 1.0),
            field_scale=jnp.append(inner.field_scale, speed),
            names=(*inner.names, BULK_VELOCITY) if inner.names else (),
        )

    def block_scaled(self, state: jnp.ndarray) -> BlockScaledNorm:
        """The unbordered block-scaled measure, the constraint row sharing the leading block's scale.

        The leading block is the flow (or its velocity), and the constraint is a bulk velocity, so the
        flow's scale is the one in its units.
        """
        inner = self.inner.block_scaled(self.drive.fields_state(self.fields, state))
        return BlockScaledNorm(
            (*inner.sizes, 1),
            jnp.append(inner.scales, inner.scales[0]),
            names=(*inner.names, BULK_VELOCITY) if inner.names else (),
        )


class _BorderedHomotopy:
    """A homotopy whose every station's residual is bordered by one constraint.

    The stations are related problems on one mesh, sharing the drive and its border vectors, so the
    border wraps whatever residual a station returns. A plain object rather than a module, as the
    homotopies it wraps are: :meth:`enter` may have side effects.
    """

    def __init__(self, inner: ResidualHomotopy, border: MassFlowBorder) -> None:
        self._inner = inner
        self._border = border

    def enter(self, step: int, previous: StepReport | None) -> Callable[[jnp.ndarray], jnp.ndarray]:
        return self._border.residual(self._inner.enter(step, previous))

    def arrived(self, step: int) -> bool:
        return self._inner.arrived(step)

    def shift_factor(self, step: int) -> float:
        return self._inner.shift_factor(step)

    def station(self, step: int) -> int:
        return self._inner.station(step)


class _DetachedSource:
    """A continuation source handed the fields of each augmented state, never the force beside them.

    A march over a bordered state builds and re-freezes its step from that state, while every step
    builder -- a session, a caller's own -- is written against the fields; this is where the two meet,
    so a builder sees one kind of state whatever the drive.
    """

    def __init__(self, inner: ContinuationSource, border: MassFlowBorder) -> None:
        self._inner = inner
        self._border = border
        self.refresh_preconditioner = (
            None if inner.refresh_preconditioner is None else self._refresh_on_fields
        )

    def _refresh_on_fields(
        self, step: NewtonStrategy, state: jnp.ndarray, *, due: bool = False
    ) -> Callable[[NewtonStrategy], NewtonStrategy] | None:
        return self._inner.refresh_preconditioner(step, self._border.detach(state), due=due)

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        return self._inner.build(self._border.detach(state))

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        return self._inner.refresh(self._border.detach(state), previous)


class MassFlowBorder(eqx.Module):
    """The bulk-velocity constraint as it borders one residual: every piece a march needs, in one value.

    Attributes
    ----------
    drive : MassFlow
        The constraint, carried by the assembler whose residual is bordered.
    fields : FieldLayout
        That residual's unbordered layout.
    force, average : jnp.ndarray
        The border column ``a = dR/dbeta`` and row ``c``, shape ``(fields.size,)``.
    """

    drive: MassFlow
    fields: FieldLayout
    force: jnp.ndarray
    average: jnp.ndarray

    def attach(self, state: jnp.ndarray) -> jnp.ndarray:
        """``state`` with the drive's current force attached -- where a bordered march starts."""
        return self.drive.driven_state(self.fields, state)

    def detach(self, augmented: jnp.ndarray) -> jnp.ndarray:
        """The fields of an augmented state, the force dropped."""
        return self.drive.fields_state(self.fields, augmented)

    def force_of(self, augmented: jnp.ndarray) -> jnp.ndarray:
        """The solved force an augmented state carries."""
        return self.drive.split(self.fields, augmented)[1]

    def force_holding(self, residual: jnp.ndarray) -> jnp.ndarray:
        """The force a root of the bordered system holds, read from its fields' residual alone.

        At a root ``R(w; beta_0) + (beta - beta_0) a = 0``, with ``beta_0`` the force the drive carries,
        so the force is the least-squares ``beta = beta_0 - (a . R) / (a . a)`` -- exact to the solve's
        tolerance, and differentiable through the fields like any function of them.

        Parameters
        ----------
        residual : jnp.ndarray
            ``R(w; beta_0)`` at the converged fields, shape ``(fields.size,)``.

        Returns
        -------
        jnp.ndarray
            The scalar force along the drive's flow direction.
        """
        return self.drive.force - jnp.dot(self.force, residual) / jnp.dot(self.force, self.force)

    def residual(self, residual: Callable[[jnp.ndarray], jnp.ndarray]) -> BorderedResidual:
        """``residual``, evaluated at the drive's carried force, bordered by the constraint."""
        return BorderedResidual(residual, self.drive, self.fields, self.force, self.average)

    def policy(self, inner: ShiftPolicy) -> BorderedShiftPolicy:
        """``inner`` bordered for the augmented state."""
        return BorderedShiftPolicy(inner, self.drive, self.fields, self.force, self.average)

    def measures(self, inner: ResidualMeasures) -> BorderedMeasures:
        """The unbordered residual's measures, extended with the constraint row."""
        return BorderedMeasures(inner, self.drive, self.fields)

    def source(self, inner: ContinuationSource) -> _DetachedSource:
        """``inner``, handed the fields of each state the bordered march builds or re-freezes from."""
        return _DetachedSource(inner, self)

    def homotopy(self, inner: ResidualHomotopy | None) -> _BorderedHomotopy | None:
        """``inner`` with every station's residual bordered, or ``None`` for no homotopy."""
        return None if inner is None else _BorderedHomotopy(inner, self)

    def on_fields(
        self, measure: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] | None:
        """A ``start -> (state -> value)`` measure of the fields, read on augmented states instead.

        For a drift measure, which watches a coefficient of the fields and has nothing to say about the
        force.
        """
        if measure is None:
            return None

        def rebased(start: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
            of_fields = measure(self.detach(start))
            return lambda augmented: of_fields(self.detach(augmented))

        return rebased


def mass_flow_border(momentum: MomentumContinuity, layout: FieldLayout) -> MassFlowBorder | None:
    """The border a residual over ``layout`` takes from ``momentum``'s drive, or ``None`` if it has none.

    Parameters
    ----------
    momentum : MomentumContinuity
        The flow assembler, whose drive says whether the force is solved for and whose geometry sets
        the border vectors.
    layout : FieldLayout
        The unbordered layout of the residual being bordered: the flow's own, or an outer one whose
        first block is the flow.

    Returns
    -------
    MassFlowBorder or None
        ``None`` for a drive that solves for no force.
    """
    drive = momentum.drive
    if not isinstance(drive, MassFlow):
        return None
    vectors = (
        drive.constraint_vectors(momentum)
        if layout == momentum.layout
        else drive.constraint_vectors_in(momentum, layout)
    )
    return MassFlowBorder(drive, layout, *vectors)
