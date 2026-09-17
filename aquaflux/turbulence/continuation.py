"""Pseudo-transient continuation for the SST scalar (k, omega) transport solves.

The k and omega equations are stiff convection-diffusion-**reaction** scalars: the production limiter
and the near-wall omega source make a full Newton step overshoot (and drive k or omega negative) from
a cold start, exactly the fragility the coupled flow block solves with pseudo-transient continuation.
This module gives the scalar solves the *same* globalization by supplying a :class:`ScalarShiftPolicy`
to the residual-agnostic :class:`aquaflux.solve.PseudoTransientStep` engine, in place of the
fixed-count Newton loop (no line search, no continuation) the scalar sub-solves used before.

The shift is proportional to the scalar transport operator's own diagonal (the ``a_P`` analogue; see
:func:`~aquaflux.turbulence.preconditioner.scalar_transport_shift_diagonal`), so it is the scalar
counterpart of the momentum ``a_P`` shift and is scale-invariant across a graded, wall-resolved mesh.
The shifted operator is preconditioned by the frozen convection-diffusion AMG the scalar already has
(:func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner`): the shift only *adds*
positive diagonal (more diagonal dominance), so the AMG built for the unshifted operator stays a valid,
effective preconditioner for the shifted one — no per-step rebuild of the off-jit hierarchy is needed.

Because the shift vanishes at the fixed point (``R(phi*) = 0`` exactly), swapping the fixed-count Newton
sub-solve for this **converges the same field** and, unlike an unrolled fixed-count loop, gives it a
clean implicit-function-theorem adjoint — a step toward the fully-coupled ``R(u, p, k, omega)`` adjoint.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from aquaflux.solve import (
    DEFAULT_GLOBALIZATION,
    Convergence,
    Globalization,
    RootSolver,
    ShiftTerm,
)

from .preconditioner import ScalarTransportPreconditioner

_ScalarResidual = Callable[[jnp.ndarray], jnp.ndarray]


class _ParameterFreeResidual(eqx.Module):
    """``(phi, theta) -> residual(phi)``: a bare residual in the two-argument form the solver takes.

    A scalar sweep's residual has no differentiable parameters -- its closure fields are frozen into
    it -- so ``theta`` is ``None`` and this drops it. It is a **module, not a lambda**, because the
    march compiles each step with the residual as an argument: a closure built per sweep is hashed by
    identity and would recompile the whole solve every sweep, which is the cost the frozen
    preconditioner is carried across sweeps to avoid. As a module, ``residual`` (itself a bound method
    of the sweep's assembler) rides as dynamic array leaves over a fixed structure, so a sweep that
    changes only values is a compilation-cache hit.

    Attributes
    ----------
    residual : callable
        The bare ``phi -> R`` for this sweep.
    """

    residual: _ScalarResidual

    def __call__(self, phi: jnp.ndarray, theta: object) -> jnp.ndarray:
        del theta
        return self.residual(phi)


class ScalarShiftPolicy(eqx.Module):
    """The shift policy for a scalar transport equation's pseudo-transient continuation.

    Supplies the two problem-specific choices :class:`~aquaflux.solve.PseudoTransientStep` needs for a
    scalar: the base pseudo-time shift diagonal (the transport operator diagonal, the ``a_P`` analogue)
    and the shifted-operator preconditioner. The policy is a cheap per-sweep carrier the engine scales
    by ``beta`` each step.

    The two have different lifetimes in a segregated outer loop, and the fields are typed to allow
    that: the shift diagonal is rebuilt each sweep from the live closure, so the pseudo-time damping
    always scales with the current operator, while the preconditioner is built **once** and carried
    across sweeps (it only accelerates the Krylov iteration, so freezing it costs at most a few extra
    iterations — see :func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner`).
    Because both fields are pytrees, a policy rebuilt each sweep is still a compilation-cache *hit*
    for a jitted solve that takes it as an argument.

    Attributes
    ----------
    shift_diagonal : jnp.ndarray
        The non-negative per-cell base shift ``d``, shape ``(n_cells,)`` (from
        :func:`~aquaflux.turbulence.preconditioner.scalar_transport_shift_diagonal`). Already the full
        scalar state, so no packing is needed. The engine adds ``beta d`` to the Jacobian diagonal.
    preconditioner : ScalarTransportPreconditioner or None
        The frozen ``phi -> M`` convection-diffusion AMG for the *unshifted* operator, or ``None`` for
        an unpreconditioned solve. Reused unchanged for the shifted operator: the shift only increases
        diagonal dominance, so the unshifted AMG remains a valid preconditioner (and its off-jit
        hierarchy need not be rebuilt per step).
    """

    shift_diagonal: jnp.ndarray
    preconditioner: ScalarTransportPreconditioner | None = None

    def shift_term(self, phi: jnp.ndarray, residual: jnp.ndarray | None = None) -> ShiftTerm:
        """The base shift diagonal and the (beta-independent) frozen preconditioner at ``phi``."""
        precond = self.preconditioner

        def make_preconditioner(
            relaxation: jnp.ndarray,
        ) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
            # The frozen AMG is built for the unshifted operator; the shift only adds positive
            # diagonal, so the same M preconditions the shifted operator (no per-beta rebuild).
            return None if precond is None else precond(phi)

        return ShiftTerm(jax.lax.stop_gradient(self.shift_diagonal), make_preconditioner)


def scalar_pseudo_transient_solve(
    *,
    globalization: Globalization = DEFAULT_GLOBALIZATION,
    max_steps: int = 40,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    solver: lx.AbstractLinearSolver | None = None,
) -> Callable[[_ScalarResidual, jnp.ndarray, ScalarShiftPolicy | None], jnp.ndarray]:
    """Build a ``solve_scalar(residual, state, policy)`` that globalizes the solve by continuation.

    A drop-in for the :func:`~aquaflux.turbulence.solve_segregated` ``solve_scalar`` slot in its
    continuation mode: the driver passes a per-sweep :class:`ScalarShiftPolicy` as the third argument,
    and this drives the scalar residual to convergence with an :class:`~aquaflux.solve.RootSolver`
    whose Newton step is the pseudo-transient march (switched-evolution-relaxation shift + closed-loop
    accept/escalate) -- the same globalization the flow block has, for the stiff reactive k/omega
    equations. A ``None`` policy falls back to an unpreconditioned, unshifted continuation solve.

    This is a **forward** solve: the residual is taken as a bare ``phi -> R`` (its frozen closure
    fields are baked in), matching the forward-only segregated driver, so the returned callable is not
    itself reverse-differentiable (differentiating through it raises a clean ``CustomVJPException``
    rather than silently dropping the coupling gradient). The underlying
    :class:`~aquaflux.solve.PseudoTransientStep` engine *is* adjoint-transparent -- its shift and
    preconditioner are frozen and vanish at the fixed point -- so the exact coupled sensitivity is the
    fully-coupled ``R(u, p, k, omega)`` implicit-function-theorem adjoint (threading the closure fields
    as ``theta``) of :func:`~aquaflux.turbulence.solve_coupled`, not this segregated forward march.

    Parameters
    ----------
    globalization : Globalization
        How hard the march damps and what it does when a step misbehaves -- the schedule, the
        accept/escalate ladder, the divergence guard and the backtracking ladder, shared with the flow
        and coupled marches. ``beta0`` is a starting guess, not a per-case knob: escalation recovers a
        too-small value. Only the fields it sets are applied; left unset, the march takes the full
        shifted step and leaves escalation as the only recourse to an overshoot, and a stiffer scalar
        can be given a ``line_search`` or a ``beta_floor`` here without constructing the step by hand.
    max_steps : int
        Maximum Newton/continuation iterations per scalar solve.
    rtol, atol : float
        Nonlinear stopping tolerances on the residual norm.
    solver : lineax.AbstractLinearSolver or None
        Forward-loop linear solver; ``None`` uses the pseudo-transient march's own inexact-Newton
        default (a loose relative tolerance with a tight absolute floor).

    Returns
    -------
    callable
        ``solve_scalar(residual, state, policy) -> state``. Forward-only: the residual is a bare
        ``phi -> R`` with its closure fields baked in, so there are no differentiable parameters and
        differentiating through it raises rather than silently dropping the coupling gradient (see
        above).
    """

    def solve_scalar(
        residual: _ScalarResidual,
        state: jnp.ndarray,
        policy: ScalarShiftPolicy | None,
    ) -> jnp.ndarray:
        # A `None` policy is the unpreconditioned, unshifted fallback: a zero shift diagonal and no
        # adjoint factory. Same construction either way, so the two cannot be configured differently.
        forward = globalization.step(
            ScalarShiftPolicy(jnp.zeros_like(state)) if policy is None else policy,
            adjoint_preconditioner_factory=None if policy is None else policy.preconditioner,
        )
        newton = RootSolver(
            convergence=Convergence(rtol=rtol, atol=atol),
            max_steps=max_steps,
            linear_solver=solver,
            strategy=forward,
        )
        return newton.solve(_ParameterFreeResidual(residual), state, None)

    return solve_scalar
