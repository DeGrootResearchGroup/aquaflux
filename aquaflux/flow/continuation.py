"""Pseudo-transient continuation for the coupled flow Newton solve at high Reynolds number.

A line-searched Newton step converges the channel at Re ~ 100 but fails once the flow becomes
convection-dominated (a few hundred Reynolds). As convection strengthens, the undamped Newton step
from the uniform cold start **overshoots** — the full step *increases* the residual, more steeply as
the Reynolds number rises — so the basin the step must land in shrinks and the backtracking line
search must retreat to ever-smaller steps, until it can no longer march the convective path
(:mod:`tests.integration.test_channel` documents the Re ~ 100 floor this lifts).

The dominant missing ingredient is *outer* Newton globalization, not a better linear solve. The
block-SIMPLE preconditioner (:mod:`aquaflux.flow.block_preconditioner`) — a velocity block on the
**viscous** momentum operator and a SIMPLE pressure Schur on ``V / a_P`` — stays an effective
approximation of the Jacobian: the preconditioned inner GMRES converges cleanly at the cold start at
every tested Reynolds number (to Re ~ 2000), so the inner solve is not the bottleneck. A stronger
preconditioner extends the line search's reach only modestly (it does not damp the overshoot); what
carries the convective regime is a globalization that keeps each iterate inside the basin the undamped
step overshoots.

The cure is **pseudo-transient continuation**: each outer step solves a *diagonally shifted* Newton
system

    (J(φ) + diag(s) P_u) δ = -R(φ),    s = β a_P    (velocity DOFs only, via P_u),

then takes ``φ ← φ + δ``. The shift ``s`` is proportional to the frozen momentum diagonal ``a_P``,
so it is the coupled-Newton form of SIMPLE velocity under-relaxation (effective diagonal
``a_P (1 + β)`` ⇔ relaxation factor ``1 / (1 + β)``) — equivalently a *local* pseudo-time term
``V / Δt`` with a cell-local ``Δt ∝ V / (β a_P)``. Being proportional to ``a_P`` rather than a global
``V / Δt`` makes the damping **scale-invariant**: a graded, wall-resolved mesh (tiny near-wall cells,
coarse core) is relaxed uniformly in relative terms, which a single global ``Δt`` cannot do. The
shift's essential job is this outer globalization — it keeps each iterate inside the basin the
undamped step would overshoot, so the (already-effective) preconditioner keeps working along the whole
march; the preconditioner inverts the *same* shifted diagonal ``a_P (1 + β)`` the step is damped by, so
it stays consistent with the shifted operator (whose added diagonal only improves its conditioning).

Everything that is *not* flow-specific — the switched-evolution-relaxation schedule
``β = β₀ (‖R‖/‖R₀‖)^p`` (strong damping from a cold start, ``β → 0`` recovering the undamped Newton
step at convergence), the shifted linear solve, and the closed-loop accept/escalate loop that turns
``β₀`` into a starting guess rather than a per-case knob — lives in the residual-agnostic
:class:`aquaflux.solve.PseudoTransientStep`. This module supplies only the flow's choices through a
:class:`MomentumShiftPolicy`: the velocity ``a_P`` shift above and the matching shifted SIMPLE
preconditioner. Because the shift vanishes at the fixed point (``R(φ*) = 0`` exactly), the
implicit-function-theorem adjoint is untouched — continuation only reshapes the forward path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aquaflux.solve.continuation import DivergenceGuard, PseudoTransientStep, ShiftTerm
from aquaflux.solve.implicit import ImplicitNewtonSolver

from .block_preconditioner import BlockPreconditioner

if TYPE_CHECKING:
    from .momentum import MomentumContinuity


class MomentumShiftPolicy(eqx.Module):
    """The coupled-flow shift policy for pseudo-transient continuation (see the module docstring).

    Supplies the two flow-specific choices :class:`~aquaflux.solve.PseudoTransientStep` needs: it
    shifts **only the velocity block**, by the block preconditioner's frozen momentum diagonal
    ``a_P`` — the coupled-Newton form of SIMPLE velocity under-relaxation (effective diagonal
    ``a_P (1 + β)`` ⇔ relaxation ``1 / (1 + β)``) — and preconditions the shifted system with the
    block-SIMPLE preconditioner at that *same* under-relaxed diagonal, restoring its diagonal
    dominance. Being proportional to ``a_P`` (not a global ``V / Δt``) makes the damping
    scale-invariant across a graded, wall-resolved mesh.

    Attributes
    ----------
    preconditioner : BlockPreconditioner
        The block-SIMPLE preconditioner, applied at the shifted diagonal ``a_P (1 + β)`` each step,
        and the source of the frozen ``a_P`` the shift is formed from.
    """

    preconditioner: BlockPreconditioner

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        """The base velocity shift diagonal and the ``β -> M`` shifted preconditioner at ``phi``.

        Parameters
        ----------
        phi : jnp.ndarray
            The flat coupled state ``[vel_0..vel_{dim-1}, pressure]``, shape ``((dim + 1) n_cells,)``.

        Returns
        -------
        ShiftTerm
            ``diagonal`` places the isotropic per-cell ``a_P`` on every velocity component and zero on
            pressure (the full-state base shift ``d``); ``make_preconditioner(β)`` returns the block
            preconditioner at the under-relaxed diagonal ``a_P (1 + β)``.
        """
        block = self.preconditioner
        a_p = block.frozen_momentum_diagonal(phi)  # isotropic per-cell a_P, (n_cells,)
        assembler = block.assembler
        n_cells = assembler.mesh.n_cells
        # a_P on every velocity component, zero on pressure — the full-state base shift d(φ). The
        # engine scales it by β and adds β·d to the Jacobian diagonal (velocity DOFs only).
        diagonal = assembler.pack(
            jnp.broadcast_to(a_p[:, None], (n_cells, assembler.mesh.dim)), jnp.zeros(n_cells)
        )

        def make_preconditioner(
            relaxation: jnp.ndarray,
        ) -> Callable[[jnp.ndarray], jnp.ndarray]:
            # Invert the same under-relaxed diagonal a_P (1 + β) the shift adds to the Jacobian, so
            # the preconditioner matches the shifted operator. Frozen: the coefficient is detached.
            return block.apply_at(phi, jax.lax.stop_gradient(a_p * (1.0 + relaxation)))

        return ShiftTerm(diagonal, make_preconditioner)


def momentum_continuation(
    assembler: MomentumContinuity,
    *,
    beta0: float = 2.0,
    exponent: float = 1.0,
    max_escalations: int = 6,
    escalation_factor: float = 2.0,
    divergence_cap: float = 10.0,
    **preconditioner_kwargs: object,
) -> PseudoTransientStep:
    """The pseudo-transient continuation ``ForwardStep`` for the coupled flow solve.

    Builds the block-SIMPLE preconditioner for ``assembler`` and wires a :class:`MomentumShiftPolicy`
    (the velocity ``a_P`` shift + the matching shifted preconditioner) into the residual-agnostic
    :class:`~aquaflux.solve.PseudoTransientStep` engine, which owns the schedule, the shifted solve,
    and the accept/escalate loop. The result plugs straight into
    :class:`~aquaflux.solve.ImplicitNewtonSolver` as its ``forward_step`` (the forward Newton loop
    uses the diagonally shifted step in place of the default line search; the converged,
    well-conditioned adjoint solve uses the bare block preconditioner, since at ``φ*`` the shift has
    vanished). ``PseudoTransientStep`` is itself the ``ForwardStep``, so no wrapper is needed.

    Parameters
    ----------
    assembler : MomentumContinuity
        The coupled flow residual assembler.
    beta0 : float
        *Initial* under-relaxation strength ``β₀`` — the damping the first attempt of each step tries,
        ``β = β₀ (‖R‖/‖R₀‖)^p``. With the step-acceptance escalation it is a starting guess rather than
        a per-case knob: too small is recovered by escalation, too large only costs a slower march. The
        effective SIMPLE velocity relaxation is ``1/(1+β)``.
    exponent : float
        Switched-evolution-relaxation exponent ``p`` in ``β = β₀ (‖R‖/‖R₀‖)^p``. ``1`` ramps the shift
        linearly with the residual norm.
    max_escalations : int
        Maximum per-step damping escalations. A step whose shifted solve fails to descend is re-damped
        (``β *= escalation_factor``) and retried, up to this many times; a well-behaved step is
        accepted on the first attempt. ``0`` disables escalation. The default makes the solve robust to
        ``β₀`` across Reynolds numbers.
    escalation_factor : float
        Factor ``> 1`` by which ``β`` grows on each rejected attempt.
    divergence_cap : float
        The :class:`~aquaflux.solve.DivergenceGuard` threshold: an attempt is rejected (and the damping
        escalated) if its residual is non-finite or exceeds ``divergence_cap × ‖R₀‖`` — measured
        against the *initial* residual, since the non-monotone march oscillates around and below it.
    **preconditioner_kwargs
        Forwarded to :meth:`BlockPreconditioner.build` (e.g. ``schur_scaling``, ``velocity``).

    Returns
    -------
    PseudoTransientStep
        The configured continuation, ready to pass as ``ImplicitNewtonSolver(forward_step=...)``.
    """
    preconditioner = BlockPreconditioner.build(assembler, **preconditioner_kwargs)
    return PseudoTransientStep(
        MomentumShiftPolicy(preconditioner),
        beta0=beta0,
        exponent=exponent,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        acceptance=DivergenceGuard(divergence_cap=divergence_cap),
        adjoint_preconditioner_factory=preconditioner.factory(),
    )


def reused_flow_solve(
    reference: MomentumContinuity,
    *,
    max_steps: int = 80,
    **build_kwargs: object,
) -> Callable[[MomentumContinuity, jnp.ndarray], jnp.ndarray]:
    """A ``solve_flow(momentum, state)`` that builds its preconditioned continuation **once** and
    reuses it across calls whose effective viscosity differs.

    A segregated outer loop (e.g. the k--omega SST driver) re-solves the momentum system every sweep
    with an updated eddy viscosity ``nu + nu_t``. Building a fresh continuation each sweep rebuilds
    the (off-jit) block-preconditioner AMG hierarchies and, because each is a new object, retraces
    and recompiles the whole solve — a per-sweep cost that grows with mesh size. This helper builds
    the continuation once from ``reference`` and returns a jitted solve, so a sweep changes only the
    viscosity *values* passed as the residual parameters: the compiled solve is reused and nothing is
    rebuilt.

    Freezing the preconditioner at one viscosity stays effective across the sweeps because the
    preconditioner only accelerates the Krylov iteration (it never enters the converged residual or
    the adjoint), and a larger eddy viscosity makes the momentum operator *more* diffusion-dominated
    (lower cell Peclet) — the regime the frozen block preconditioner already handles best. Build
    ``reference`` at a representative operating viscosity (e.g. the initial eddy-viscosity estimate);
    a laminar reference also works but a mid-range one keeps the frozen ``a_P`` closest to the sweeps.

    Parameters
    ----------
    reference : MomentumContinuity
        The flow assembler whose (effective) viscosity sets the frozen preconditioner. Its molecular
        or effective viscosity only calibrates the accelerator; each solve is driven to the residual
        of the ``momentum`` it is called with.
    max_steps : int
        Maximum Newton/continuation iterations per solve.
    **build_kwargs
        Forwarded to :func:`momentum_continuation` (e.g. ``schur_scaling="msimpler"``,
        ``velocity="convection"``, ``beta0``).

    Returns
    -------
    callable
        ``solve_flow(momentum, state) -> state`` solving ``momentum.residual`` from ``state`` with the
        frozen preconditioned continuation. Reverse-differentiable in ``momentum`` (the underlying
        implicit-function-theorem adjoint is unchanged; the ``jit`` wrapper is transparent to it).
    """
    continuation = momentum_continuation(reference, **build_kwargs)
    solver = ImplicitNewtonSolver(max_steps=max_steps, forward_step=continuation)

    @eqx.filter_jit
    def solve_flow(momentum: MomentumContinuity, state: jnp.ndarray) -> jnp.ndarray:
        return solver.solve(lambda s, m: m.residual(s), state, momentum)

    return solve_flow
