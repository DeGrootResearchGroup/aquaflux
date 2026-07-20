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
and the ordinary Newton solver (:class:`~aquaflux.solve.ImplicitNewtonSolver`) drives it to a converged
root, with no bespoke solver and no hand-derived linearization. ``<U> = U_bar`` therefore holds at the
converged root **by construction**, so the bulk velocity can never overshoot while the eddy viscosity
is still developing (the failure the old proportional controller had at high Reynolds number / high
aspect ratio: it measured ``U_bulk`` at a fixed ``beta``, which spiked before the feedback could react,
collapsing the near-wall ``k`` onto its floor).

Being the production Newton solve, it is **convergence-gated** (stops on the residual tolerance) and
**reverse-differentiable** through the implicit-function-theorem adjoint at the converged root.

**Preconditioning the augmented system (constraint preconditioning).** ``beta`` is a Lagrange
multiplier, not a function of ``w``, so -- unlike a nested gradient sub-solve -- it cannot be absorbed
inside the residual; the border is eliminated one layer down, in the **preconditioner**. Given a flow-
block preconditioner ``M ~ J^{-1}`` (the block-SIMPLE AMG), :func:`_bordered_preconditioner` wraps it
into a preconditioner for the ``(dim+1) n_cells + 1`` augmented system by Schur-eliminating the scalar
``beta``:

    y   = M r_flow
    dbeta = (c^T y - r_beta) / (c^T M a)      # the 1x1 Schur complement c^T J^{-1} a, approximated
    dw  = y - dbeta (M a)

so one augmented Krylov iteration costs one application of the flow preconditioner plus O(n) dots, and
the flow block is handed to the block preconditioner unchanged. It is exact when ``M = J^{-1}`` (the
augmented solve converges in one step); with the frozen block AMG the augmented system inherits the
flow block's mesh-independence. Hand-building ``a`` and ``c`` here is legitimate: a preconditioner is an
approximate inverse, ``stop_gradient``-ed, that changes only Krylov convergence, never the solution or
its adjoint. A direct or unpreconditioned solve (the small nx=4 channels) passes ``preconditioner=None``
and needs none of this.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import lineax as lx

from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver

if TYPE_CHECKING:
    from .momentum import MomentumContinuity

_Matvec = Callable[[jnp.ndarray], jnp.ndarray]
_Preconditioner = Callable[[jnp.ndarray], _Matvec]
_ConstrainedSolve = Callable[
    ["MomentumContinuity", jnp.ndarray], tuple["MomentumContinuity", jnp.ndarray]
]


def _with_body_force(
    momentum: MomentumContinuity, flow_direction: int, beta: jnp.ndarray
) -> MomentumContinuity:
    """Return ``momentum`` with its body force along ``flow_direction`` set to ``beta``."""
    return eqx.tree_at(
        lambda m: m.body_force, momentum, momentum.body_force.at[flow_direction].set(beta)
    )


def _constraint_vectors(
    momentum: MomentumContinuity, flow_direction: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The constant border column/row ``(a, c)`` of the augmented Jacobian, as flat flow-state vectors.

    ``a = dR_flow/dbeta = -V`` on the flow-direction velocity rows (the body force enters as
    ``R = flux - beta V``); ``c = d<U_dir>/dw = V/sum(V)`` there (so ``c . w = <U_dir>``). Both are
    fixed by the geometry.
    """
    volume = momentum.geometry.cell.volume
    n_cells, dim = momentum.mesh.n_cells, momentum.mesh.dim
    pressure_zero = jnp.zeros(n_cells)
    force = jnp.zeros((n_cells, dim)).at[:, flow_direction].set(-volume)
    average = jnp.zeros((n_cells, dim)).at[:, flow_direction].set(volume / jnp.sum(volume))
    return momentum.pack(force, pressure_zero), momentum.pack(average, pressure_zero)


def _bordered_preconditioner(
    flow_preconditioner: _Preconditioner, force: jnp.ndarray, average: jnp.ndarray
) -> _Preconditioner:
    """Wrap a flow-block preconditioner ``M ~ J^{-1}`` into one for the augmented ``[w, beta]`` system.

    Constraint (Schur) preconditioning: eliminate the scalar ``beta`` via the 1x1 Schur complement
    ``c^T M a`` and apply ``M`` to the flow block (see the module docstring). Exact when ``M = J^{-1}``.

    Parameters
    ----------
    flow_preconditioner : callable
        Factory ``w -> (matvec ~ J^{-1})`` for the un-augmented flow block (e.g.
        :meth:`aquaflux.flow.BlockPreconditioner.factory`).
    force, average : jnp.ndarray
        The border column ``a`` and row ``c`` from :func:`_constraint_vectors`, shape
        ``((dim + 1) n_cells,)``.

    Returns
    -------
    callable
        Factory ``augmented -> (matvec ~ J_aug^{-1})`` for the augmented system.
    """

    def factory(augmented: jnp.ndarray) -> _Matvec:
        flow_matvec = flow_preconditioner(augmented[:-1])
        m_force = flow_matvec(force)  # M a
        schur = jnp.dot(average, m_force)  # c^T M a  (approximates c^T J^{-1} a)

        def apply(residual: jnp.ndarray) -> jnp.ndarray:
            y = flow_matvec(residual[:-1])  # M r_flow
            d_beta = (jnp.dot(average, y) - residual[-1]) / schur
            return jnp.append(y - d_beta * m_force, d_beta)

        return apply

    return factory


def bulk_velocity_flow_solve(
    *,
    target: float,
    flow_direction: int = 0,
    max_steps: int = 20,
    solver: lx.AbstractLinearSolver | None = None,
    preconditioner: _Preconditioner | None = None,
) -> _ConstrainedSolve:
    """Build a ``solve(momentum, state) -> (momentum, state)`` that holds the bulk velocity at ``target``.

    Solves the flow with the body force ``beta`` (along ``flow_direction``) treated as a Lagrange
    multiplier for the constraint ``<U_dir> = target`` -- ``beta`` appended to the state and the flow
    residual augmented with the constraint equation, driven by the production
    :class:`~aquaflux.solve.ImplicitNewtonSolver` (see the module docstring). The initial ``beta`` is
    read from ``momentum.body_force[flow_direction]``, and the returned ``momentum`` carries the
    converged ``beta`` (via :func:`equinox.tree_at`), so a segregated outer loop can thread it forward.

    Parameters
    ----------
    target : float
        The bulk (volume-averaged) velocity component to hold along ``flow_direction``.
    flow_direction : int
        The streamwise axis the bulk velocity is measured and the body force is applied along.
    max_steps : int
        Newton-iteration cap for the constrained solve. The solve stops earlier on the residual
        tolerance; this only bounds the worst case. The augmented system stays near-linear for a
        fully-developed channel, so a small cap suffices.
    solver : lineax.AbstractLinearSolver or None
        Linear solver for the augmented Newton steps (e.g. a direct solve for a small coupled system);
        ``None`` uses the Newton strategy's inexact-GMRES default.
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
    """

    @eqx.filter_jit
    def solve(
        momentum: MomentumContinuity, state: jnp.ndarray
    ) -> tuple[MomentumContinuity, jnp.ndarray]:
        volume = momentum.geometry.cell.volume

        def augmented_residual(augmented: jnp.ndarray, _theta: object) -> jnp.ndarray:
            flow, beta = augmented[:-1], augmented[-1]
            forced = _with_body_force(momentum, flow_direction, beta)
            velocity, _ = forced.unpack(flow)
            bulk = jnp.sum(velocity[:, flow_direction] * volume) / jnp.sum(volume)
            return jnp.append(forced.residual(flow), bulk - target)

        augmented_preconditioner = None
        if preconditioner is not None:
            force, average = _constraint_vectors(momentum, flow_direction)
            augmented_preconditioner = _bordered_preconditioner(preconditioner, force, average)

        augmented0 = jnp.append(state, momentum.body_force[flow_direction])
        newton = ImplicitNewtonSolver(
            max_steps=max_steps,
            solver=solver,
            forward_step=DampedNewtonStep(preconditioner=augmented_preconditioner),
        )
        augmented = newton.solve(augmented_residual, augmented0, None)
        flow, beta = augmented[:-1], augmented[-1]
        return _with_body_force(momentum, flow_direction, beta), flow

    return solve
