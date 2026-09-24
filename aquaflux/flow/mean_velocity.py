"""Bulk-velocity-constrained flow solve: the body force is a solve unknown, not a feedback loop.

A streamwise-periodic channel prescribes velocity nowhere; it is driven to a target bulk (volume-
averaged) velocity ``U_bar`` by a uniform body force ``beta`` along the flow direction. The naive way
to hold ``U_bar`` is an *outer* controller that solves the flow at a fixed ``beta``, measures the bulk
velocity it produced, and nudges ``beta`` for the next solve. That controller can overshoot badly when
the momentum operator changes between solves (a segregated loop's eddy viscosity is stale by a sweep):
the flow can converge to a physically-correct-but-absurd bulk velocity before the controller reacts.

Here ``beta`` is instead a **solve unknown** -- a scalar Lagrange multiplier enforcing the constraint
``<U_dir> - U_bar = 0``. It is appended to the flow state ``w = [u, p]`` and the flow residual is
augmented with the constraint equation:

    R_aug([w, beta]) = [ R_flow(w; beta) ; <U_dir>(w) - U_bar ] = 0.

This is one honest residual: automatic differentiation assembles the whole bordered Jacobian ``J_aug =
[[J, a], [c^T, 0]]`` -- the force column ``a = dR_flow/dbeta = -V`` on the flow-direction momentum rows
(``beta`` enters as ``R = flux - beta V``) and the averaging row ``c^T = d<U>/dw = V/sum(V)`` there --
and the ordinary Newton solver (:class:`~aquaflux.solve.RootSolver`) drives it to a converged
root, with no bespoke solver and no hand-derived linearization. ``<U> = U_bar`` therefore holds at the
converged root **by construction**, so the bulk velocity can never overshoot while the eddy viscosity
is still developing (the failure the old proportional controller had at high Reynolds number / high
aspect ratio: it measured ``U_bulk`` at a fixed ``beta``, which spiked before the feedback could react,
collapsing the near-wall ``k`` onto its floor).

The constraint itself -- what is held, along which axis, where the multiplier sits in the state, and
the border column and row -- is a :class:`~aquaflux.flow.MassFlow` drive, carried by the assembler
being solved (see :mod:`aquaflux.flow.drive`). This module holds only what is specific to bordering
the **flow** block: the constraint preconditioner below, and the residual that reads the velocity out
of a flow state. The coupled RANS solve borders the same constraint over its own state and shares the
rest, so neither can come to enforce a different target from the one its assembler is forced with.

Being the production Newton solve, it is **convergence-gated** (stops on the residual tolerance) and
**reverse-differentiable** through the implicit-function-theorem adjoint at the converged root. The
assembler is threaded as the Newton solve's differentiable parameter (not captured in the residual
closure), so ``jax.grad`` of an objective through the solve returns its cotangent -- e.g. the
sensitivity to viscosity; a captured assembler would instead raise a ``custom_vjp`` closed-over-value
error.

**Preconditioning the augmented system (constraint preconditioning).** ``beta`` is a Lagrange
multiplier, not a function of ``w``, so -- unlike a nested gradient sub-solve -- it cannot be absorbed
inside the residual; the border is eliminated one layer down, in the **preconditioner**. Given a flow-
block preconditioner ``M ~ J^{-1}`` (the block-SIMPLE algebraic multigrid, AMG),
:func:`_bordered_preconditioner` wraps it into a preconditioner for the ``(dim+1) n_cells + 1``
augmented system by Schur-eliminating the scalar ``beta``:

    y   = M r_flow
    dbeta = (c^T y - r_beta) / (c^T M a)      # the 1x1 Schur complement c^T J^{-1} a, approximated
    dw  = y - dbeta (M a)

so one augmented Krylov iteration costs one application of the flow preconditioner plus O(n) dots, and
the flow block is handed to the block preconditioner unchanged. It is exact when ``M = J^{-1}`` (the
augmented solve converges in one step); with the frozen block AMG the augmented system inherits the
flow block's mesh-independence. Hand-building ``a`` and ``c`` here is legitimate: a preconditioner is an
approximate inverse, ``stop_gradient``-ed, that changes only Krylov convergence, never the solution or
its adjoint. A direct or unpreconditioned solve (the small nx=4 channels) passes ``preconditioner=None``
and needs none of this. The **adjoint** transpose solve at the converged root is preconditioned by the
*same* bordered preconditioner **transposed** -- the generic adjoint machinery forms it with
:func:`jax.linear_transpose` (``M_aug`` is linear, so ``M_aug^T ~ J_aug^{-T}`` exactly), so the reverse
pass is mesh-independent too with no extra code; the gradient is identical to the unpreconditioned
adjoint's (a preconditioner never changes the sensitivity). The bordered preconditioner is built once,
in the builder, from a concrete ``reference`` -- **not** inside the solve from the ``momentum`` it is
called with, which under ``jax.grad`` would leave a tracer in the (non-differentiated) preconditioner
and break the gradient.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.solve import (
    DEFAULT_ROOT_SOLVE,
    DampedNewtonStep,
    FieldLayout,
    RootSolveSettings,
)

from .drive import MassFlow, mass_flow_drive

if TYPE_CHECKING:
    from .momentum import MomentumContinuity

_Matvec = Callable[[jnp.ndarray], jnp.ndarray]
_Preconditioner = Callable[[jnp.ndarray], _Matvec]
_ConstrainedSolve = Callable[
    ["MomentumContinuity", jnp.ndarray], tuple["MomentumContinuity", jnp.ndarray]
]


def _bordered_preconditioner(
    flow_preconditioner: _Preconditioner,
    drive: MassFlow,
    fields: FieldLayout,
    force: jnp.ndarray,
    average: jnp.ndarray,
) -> _Preconditioner:
    """Wrap a flow-block preconditioner ``M ~ J^{-1}`` into one for the augmented ``[w, beta]`` system.

    Constraint (Schur) preconditioning: eliminate the scalar ``beta`` via the 1x1 Schur complement
    ``c^T M a`` and apply ``M`` to the flow block (see the module docstring). Exact when ``M = J^{-1}``.

    Parameters
    ----------
    flow_preconditioner : callable
        Factory ``w -> (matvec ~ J^{-1})`` for the un-augmented flow block (e.g.
        :meth:`aquaflux.flow.BlockPreconditioner.factory`).
    drive : MassFlow
        The drive whose layout says where the border entry sits; the same one the residual borders by,
        so the preconditioner and the operator it approximates cannot disagree about the split.
    fields : FieldLayout
        The layout of the un-augmented block ``M`` inverts -- the flow alone here, the whole coupled
        state when a coupled solve borders itself.
    force, average : jnp.ndarray
        The border column ``a`` and row ``c``, shape ``(fields.size,)``.

    Returns
    -------
    callable
        Factory ``augmented -> (matvec ~ J_aug^{-1})`` for the augmented system.
    """

    def factory(augmented: jnp.ndarray) -> _Matvec:
        flow_matvec = flow_preconditioner(drive.fields_state(fields, augmented))
        m_force = flow_matvec(force)  # M a
        schur = jnp.dot(average, m_force)  # c^T M a  (approximates c^T J^{-1} a)

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            block, border = drive.split(fields, residual)
            y = flow_matvec(block)  # M r_flow
            d_beta = (jnp.dot(average, y) - border) / schur
            return drive.join(fields, y - d_beta * m_force, d_beta)

        return apply

    return factory


class _BulkVelocityResidual(eqx.Module):
    """The flow residual bordered with ``<U_dir> - target``, as a two-argument residual.

    The assembler arrives as the Newton solve's differentiable parameter ``theta`` rather than being
    captured, so the implicit-function-theorem adjoint returns its cotangent and the constrained solve
    is reverse-differentiable in it (e.g. in its viscosity). **Everything the residual reads from the
    assembler therefore comes from** ``theta``, including the cell volumes: a value captured from an
    outer assembler would be a closed-over input of the adjoint's ``custom_vjp`` and differentiating it
    raises.

    A **module rather than a closure**, so that the drive below is compared by value: the march
    compiles its step with the residual as an argument, and a closure built per solve is hashed by
    identity, so a solver reused across a sweep would recompile every call.

    Attributes
    ----------
    drive : MassFlow
        The constraint being enforced, and the arithmetic of the border -- the layout the augmented
        vector is split and reassembled by, the bulk-velocity average, and where the multiplier is
        written into the assembler.
    """

    drive: MassFlow

    def __call__(self, augmented: jnp.ndarray, theta: MomentumContinuity) -> jnp.ndarray:
        layout = theta.layout
        flow = self.drive.fields_state(layout, augmented)
        forced = self.drive.forced(theta, self.drive.settled(layout, augmented).force)
        velocity, _ = forced.unpack(flow)
        bulk = self.drive.bulk_velocity(velocity, theta.geometry.cell.volume)
        return self.drive.join(layout, forced.residual(flow), bulk - self.drive.target)


#: ``bulk_velocity_flow_solve``'s own defaults, beneath whatever its caller sets. The augmented system
#: stays near-linear for a fully-developed channel, so a small step cap suffices.
_BULK_VELOCITY_FLOW_SOLVE = RootSolveSettings(max_steps=20)


def bulk_velocity_flow_solve(
    reference: MomentumContinuity,
    *,
    root_solve: RootSolveSettings = DEFAULT_ROOT_SOLVE,
    preconditioner: _Preconditioner | None = None,
) -> _ConstrainedSolve:
    """Build a ``solve(momentum, state) -> (momentum, state)`` that holds ``reference``'s bulk target.

    Solves the flow with the body force ``beta`` treated as a Lagrange multiplier for the constraint
    ``<U_dir> = target`` -- ``beta`` attached to the state and the flow residual augmented with the
    constraint equation, driven by the production :class:`~aquaflux.solve.RootSolver` (see the module
    docstring). What is being held, along which axis, and the ``beta`` the march starts from all come
    from ``reference``'s own :class:`~aquaflux.flow.MassFlow` drive, so the assembler the residual
    writes each iterate into and the constraint this builder enforces cannot name different targets.
    The returned ``momentum`` carries the converged ``beta`` on its drive, so a segregated outer loop
    can thread it forward.

    The assembler is threaded as the Newton solve's differentiable parameter, so the solve is
    **reverse-differentiable in** ``momentum`` (e.g. its viscosity) through the implicit-function-theorem
    adjoint -- one transpose solve at the converged root, preconditioned (when ``preconditioner`` is
    given) by the transpose of the bordered preconditioner, which the adjoint machinery forms
    automatically with :func:`jax.linear_transpose`. Build the solve **outside** ``jax.grad`` (the
    preconditioner and its reference must be concrete) and differentiate a call to it.

    Parameters
    ----------
    reference : MomentumContinuity
        The concrete assembler this solve is built for. Its drive must be a
        :class:`~aquaflux.flow.MassFlow`: that is what makes the body force an unknown rather than a
        prescribed source, and it supplies the target, the flow direction and the initial ``beta``. Its
        geometry sets the border row and column of the constraint preconditioner, built **once here**
        (not inside the jitted, potentially traced solve) so the frozen preconditioner carries no
        tracer and the differentiated solve stays clean.
    root_solve : RootSolveSettings
        How the constrained Newton solve is run -- its step cap, its stopping test and its forward and
        adjoint linear solvers. Its ``linear_solver`` is the solver for the augmented Newton steps
        (e.g. a direct solve for a small coupled system); unset, they take the Newton strategy's
        inexact-GMRES default. Only the fields it sets are applied; left unset, the step cap is this
        builder's own and everything else is the solver's.
    preconditioner : callable or None
        Factory ``w -> (matvec ~ J^{-1})`` for the **un-augmented flow block** (e.g. a frozen
        :meth:`aquaflux.flow.BlockPreconditioner.factory`, built off-jit from a reference). When given,
        it is wrapped by :func:`_bordered_preconditioner` into a constraint preconditioner for the
        augmented Krylov solve -- the mesh-independent path for a large iterative solve. ``None`` solves
        unpreconditioned (a direct or small solve needs nothing).

    Returns
    -------
    callable
        ``solve(momentum, state) -> (momentum, state)``: the flow state meeting the constraint and the
        ``momentum`` carrying the converged body force.

    Raises
    ------
    TypeError
        If ``reference`` is not driven by a :class:`~aquaflux.flow.MassFlow`.
    """
    drive = mass_flow_drive(reference, "bulk_velocity_flow_solve")
    fields = reference.layout
    augmented_preconditioner = None
    if preconditioner is not None:
        force, average = drive.constraint_vectors(reference)
        augmented_preconditioner = _bordered_preconditioner(
            preconditioner, drive, fields, force, average
        )

    augmented_residual = _BulkVelocityResidual(drive)
    settings = root_solve.filled_from(_BULK_VELOCITY_FLOW_SOLVE)

    def solve(
        momentum: MomentumContinuity, state: jnp.ndarray
    ) -> tuple[MomentumContinuity, jnp.ndarray]:
        # The layout comes from the assembler being solved, not from `reference`: only the constraint
        # preconditioner is tied to one mesh, and an unpreconditioned solve is happy to run the same
        # constraint on a refined copy of the channel.
        solved_fields = momentum.layout
        augmented0 = drive.driven_state(solved_fields, state)
        newton = settings.solver(DampedNewtonStep(preconditioner=augmented_preconditioner))
        augmented = newton.solve(augmented_residual, augmented0, momentum)
        settled = drive.settled(solved_fields, augmented)
        return eqx.tree_at(lambda m: m.drive, momentum, settled), drive.fields_state(
            solved_fields, augmented
        )

    return solve
