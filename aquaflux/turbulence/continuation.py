"""Pseudo-transient continuation for the SST scalar (k, omega) transport solves.

The k and omega equations are stiff convection-diffusion-**reaction** scalars: the production limiter
and the near-wall omega source make a full Newton step overshoot (and drive k or omega negative) from
a cold start, exactly the fragility the coupled flow block solves with pseudo-transient continuation.
This module gives the scalar solves the *same* globalization by supplying a :class:`ScalarShiftPolicy`
to the residual-agnostic :class:`aquaflux.solve.PseudoTransientStep` engine, in place of the fixed-count
``NewtonSolver`` (no line search, no continuation) the scalar sub-solves used before.

The shift is proportional to the scalar transport operator's own diagonal (the ``a_P`` analogue; see
:func:`~aquaflux.turbulence.preconditioner.scalar_transport_shift_diagonal`), so it is the scalar
counterpart of the momentum ``a_P`` shift and is scale-invariant across a graded, wall-resolved mesh.
The shifted operator is preconditioned by the frozen convection-diffusion AMG the scalar already has
(:func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner`): the shift only *adds*
positive diagonal (more diagonal dominance), so the AMG built for the unshifted operator stays a valid,
effective preconditioner for the shifted one — no per-step rebuild of the off-jit hierarchy is needed.

Because the shift vanishes at the fixed point (``R(phi*) = 0`` exactly), swapping the fixed-count Newton
sub-solve for this **converges the same field** and, unlike the unrolled ``NewtonSolver``, gives it a
clean implicit-function-theorem adjoint — a step toward the fully-coupled ``R(u, p, k, omega)`` adjoint.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from aquaflux.solve.continuation import PseudoTransientStep, ShiftTerm
from aquaflux.solve.implicit import ImplicitNewtonSolver

_Factory = Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]
_ScalarResidual = Callable[[jnp.ndarray], jnp.ndarray]


class ScalarShiftPolicy(eqx.Module):
    """The shift policy for a scalar transport equation's pseudo-transient continuation.

    Supplies the two problem-specific choices :class:`~aquaflux.solve.PseudoTransientStep` needs for a
    scalar: the base pseudo-time shift diagonal (the transport operator diagonal, the ``a_P`` analogue)
    and the shifted-operator preconditioner. Both are frozen at build time (from the sweep's closure),
    so the policy is a cheap per-sweep carrier the engine scales by ``beta`` each step.

    Attributes
    ----------
    shift_diagonal : jnp.ndarray
        The non-negative per-cell base shift ``d``, shape ``(n_cells,)`` (from
        :func:`~aquaflux.turbulence.preconditioner.scalar_transport_shift_diagonal`). Already the full
        scalar state, so no packing is needed. The engine adds ``beta d`` to the Jacobian diagonal.
    preconditioner : callable or None
        The frozen ``phi -> M`` convection-diffusion AMG factory for the *unshifted* operator, or
        ``None`` for an unpreconditioned solve. Reused unchanged for the shifted operator: the shift
        only increases diagonal dominance, so the unshifted AMG remains a valid preconditioner (and its
        off-jit hierarchy need not be rebuilt per step).
    """

    shift_diagonal: jnp.ndarray
    preconditioner: _Factory | None = None

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
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
    beta0: float = 2.0,
    exponent: float = 1.0,
    max_escalations: int = 6,
    escalation_factor: float = 2.0,
    divergence_cap: float = 10.0,
    max_steps: int = 40,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    solver: lx.AbstractLinearSolver | None = None,
) -> Callable[[_ScalarResidual, jnp.ndarray, ScalarShiftPolicy | None], jnp.ndarray]:
    """Build a ``solve_scalar(residual, state, policy)`` that globalizes the solve by continuation.

    A drop-in for the :func:`~aquaflux.turbulence.solve_segregated` ``solve_scalar`` slot in its
    continuation mode: the driver passes a per-sweep :class:`ScalarShiftPolicy` as the third argument,
    and this drives the scalar residual to convergence with an :class:`~aquaflux.solve.ImplicitNewtonSolver`
    whose forward step is the pseudo-transient march (switched-evolution-relaxation shift + closed-loop
    accept/escalate) -- the same globalization the flow block has, for the stiff reactive k/omega
    equations. A ``None`` policy falls back to an unpreconditioned, unshifted continuation solve.

    This is a **forward** solve: the residual is taken as a bare ``phi -> R`` (its frozen closure
    fields are baked in), matching the forward-only segregated driver, so the returned callable is not
    itself reverse-differentiable (differentiating through it raises a clean ``CustomVJPException``
    rather than silently dropping the coupling gradient). The underlying
    :class:`~aquaflux.solve.PseudoTransientStep` engine *is* adjoint-transparent -- its shift and
    preconditioner are frozen and vanish at the fixed point -- so the exact coupled sensitivity is the
    fully-coupled ``R(u, p, k, omega)`` implicit-function-theorem adjoint (threading the closure fields
    as ``theta``), the deferred monolithic-coupling step, not this segregated forward march.

    Parameters
    ----------
    beta0, exponent, max_escalations, escalation_factor, divergence_cap
        The pseudo-transient schedule and accept/escalate parameters (see
        :class:`~aquaflux.solve.PseudoTransientStep`). ``beta0`` is a starting guess, not a per-case
        knob — escalation recovers a too-small value.
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
        ``solve_scalar(residual, state, policy) -> state``. Reverse-differentiable through the
        converged scalar solve by the implicit-function-theorem adjoint (the ``jit`` and the shift are
        transparent to it).
    """

    @eqx.filter_jit
    def solve_scalar(
        residual: _ScalarResidual,
        state: jnp.ndarray,
        policy: ScalarShiftPolicy | None,
    ) -> jnp.ndarray:
        if policy is None:
            forward = PseudoTransientStep(
                ScalarShiftPolicy(jnp.zeros_like(state)),
                beta0=beta0,
                exponent=exponent,
                max_escalations=max_escalations,
                escalation_factor=escalation_factor,
                divergence_cap=divergence_cap,
            )
        else:
            forward = PseudoTransientStep(
                policy,
                beta0=beta0,
                exponent=exponent,
                max_escalations=max_escalations,
                escalation_factor=escalation_factor,
                divergence_cap=divergence_cap,
                adjoint_preconditioner_factory=policy.preconditioner,
            )
        newton = ImplicitNewtonSolver(
            rtol=rtol, atol=atol, max_steps=max_steps, solver=solver, forward_step=forward
        )
        return newton.solve(lambda s, _theta: residual(s), state, None)

    return solve_scalar
