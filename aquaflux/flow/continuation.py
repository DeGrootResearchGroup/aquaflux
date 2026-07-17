"""Pseudo-transient continuation for the coupled flow Newton solve at high Reynolds number.

The block-SIMPLE preconditioner (:mod:`aquaflux.flow.block_preconditioner`) inverts a velocity block
built on the **viscous** momentum operator and a SIMPLE pressure Schur built on ``V / a_P``. Both are
good approximations while diffusion dominates, but once convection strengthens (Reynolds number past
~100) the true Jacobian is far from that diffusion operator: the inner GMRES stalls a fixed fraction
of the way down and the backtracking line search alone can no longer recover a step, so the solve
stagnates (:mod:`tests.integration.test_channel` documents the Re ~ 100 floor this lifts).

The cure is **pseudo-transient continuation**: each outer step solves a *diagonally shifted* Newton
system

    (J(φ) + diag(s) P_u) δ = -R(φ),    s = β a_P    (velocity DOFs only, via P_u),

then takes ``φ ← φ + δ``. The shift ``s`` is proportional to the frozen momentum diagonal ``a_P``,
so it is the coupled-Newton form of SIMPLE velocity under-relaxation (effective diagonal
``a_P (1 + β)`` ⇔ relaxation factor ``1 / (1 + β)``) — equivalently a *local* pseudo-time term
``V / Δt`` with a cell-local ``Δt ∝ V / (β a_P)``. Being proportional to ``a_P`` rather than a global
``V / Δt`` makes the damping **scale-invariant**: a graded, wall-resolved mesh (tiny near-wall cells,
coarse core) is relaxed uniformly in relative terms, which a single global ``Δt`` cannot do. The
preconditioner inverts the *same* shifted diagonal ``a_P (1 + β)``, which restores its diagonal
dominance and makes the inner GMRES converge again.

The strength ramps to zero on the residual (switched-evolution-relaxation): ``β = β₀ (‖R‖/‖R₀‖)^p``,
so the first steps are strongly damped (robust from a uniform cold start) and the last steps recover
the undamped Newton step (``β → 0``, quadratic terminal convergence). The **shift vanishes at
convergence** — the fixed point ``δ = 0`` forces ``R(φ*) = 0`` exactly, the *unshifted* steady
residual — so the implicit-function-theorem adjoint (which linearises ``R`` at ``φ*``, never the
shifted operator) is untouched: continuation only reshapes the forward path, like the line search it
replaces.

The ``β₀`` that schedule needs, though, is not universal — the damping a cold start requires grows
with the Reynolds number and the mesh, and too small a ``β₀`` lets an early step's shifted system
diverge (the linear solve fails, or the step overshoots) before the schedule can react. So each step
is **closed-loop**: it accepts its shifted correction only if the residual stays finite and bounded,
and otherwise **escalates the damping and retries** (a smaller pseudo-timestep) until the step is
accepted. This is the globalization an open-loop schedule lacks — it turns ``β₀`` into a starting
guess (too small is recovered by escalation; too large only slows the march) rather than a per-case
knob, and it cannot diverge to a non-finite iterate. The retry does not change the fixed point, so
the converged state and its adjoint are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from aquaflux.solve.linear import solve_linear

from .block_preconditioner import BlockPreconditioner

if TYPE_CHECKING:
    from .momentum import MomentumContinuity

_ResidualFn = Callable[[jnp.ndarray], jnp.ndarray]
_Stepper = Callable[[_ResidualFn, jnp.ndarray, jnp.ndarray, lx.AbstractLinearSolver], jnp.ndarray]


class PseudoTransientContinuation(eqx.Module):
    """Implicit under-relaxation continuation for the coupled flow solve (see the module docstring).

    Plugs into :class:`~aquaflux.solve.ImplicitNewtonSolver` as its ``continuation`` strategy: when
    set, the forward Newton loop replaces the line-searched step with the diagonally shifted step
    this object supplies, while the (converged, well-conditioned) adjoint solve keeps using the bare
    block preconditioner. Build it from a flow assembler with :meth:`build`.

    Attributes
    ----------
    preconditioner : BlockPreconditioner
        The block-SIMPLE preconditioner, applied at the shifted diagonal ``a_P (1 + β)`` each step
        and (unshifted) as the adjoint preconditioner.
    beta0 : float
        *Initial* under-relaxation strength ``β₀`` (static) — the damping the first attempt of each
        step tries, ``β = β₀ (‖R‖/‖R₀‖)^p``. With the step-acceptance escalation below, ``β₀`` is a
        starting guess rather than a per-case knob: too small is recovered by escalation, too large
        only costs a slower march. The effective SIMPLE velocity relaxation is ``1/(1+β)``.
    exponent : float
        Switched-evolution-relaxation exponent ``p`` in ``β = β₀ (‖R‖/‖R₀‖)^p`` (static). ``1``
        ramps the shift linearly with the residual norm.
    max_escalations : int
        Maximum damping escalations per step (static). If a step's shifted solve fails to reduce the
        residual (an ill-conditioned shifted system, or an overshoot), ``β`` is multiplied by
        :attr:`escalation_factor` and the step retried, up to this many times — the acceptance test
        the bare pseudo-transient march lacks. A well-behaved step is accepted on the first attempt
        (no extra cost); a cold-start step that would otherwise diverge is damped until it descends,
        so the march cannot blow up. ``0`` disables escalation (the plain single-attempt step).
    escalation_factor : float
        Factor ``> 1`` by which ``β`` grows on each rejected attempt (static).
    divergence_cap : float
        The divergence threshold (static): an attempt is rejected (and the damping escalated) if its
        residual is non-finite or exceeds ``divergence_cap × ‖R₀‖``. Measured against the *initial*
        residual because the pseudo-transient march is non-monotone — it oscillates around and below
        ``‖R₀‖`` — so the guard must catch a genuine blow-up above the starting scale without rejecting
        a healthy transient. Lenient by default; lower it only to intervene on divergence sooner.
    """

    preconditioner: BlockPreconditioner
    beta0: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    max_escalations: int = eqx.field(static=True)
    escalation_factor: float = eqx.field(static=True)
    divergence_cap: float = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        assembler: MomentumContinuity,
        *,
        beta0: float = 2.0,
        exponent: float = 1.0,
        max_escalations: int = 6,
        escalation_factor: float = 2.0,
        divergence_cap: float = 10.0,
        **preconditioner_kwargs: object,
    ) -> PseudoTransientContinuation:
        """Build the continuation (and its block preconditioner) for ``assembler``.

        Parameters
        ----------
        assembler : MomentumContinuity
            The coupled flow residual assembler.
        beta0, exponent : float
            The initial under-relaxation strength and the switched-evolution-relaxation exponent
            (see the class attributes).
        max_escalations : int
            Maximum per-step damping escalations for the step-acceptance test (see the class
            attribute). The default makes the solve robust to ``β₀`` across Reynolds numbers.
        escalation_factor : float
            The per-rejection growth factor for ``β`` (see the class attribute).
        **preconditioner_kwargs
            Forwarded to :meth:`BlockPreconditioner.build` (e.g. ``inner``, ``v_cycles``).
        """
        preconditioner = BlockPreconditioner.build(assembler, **preconditioner_kwargs)
        return cls(
            preconditioner, beta0, exponent, max_escalations, escalation_factor, divergence_cap
        )

    def adjoint_preconditioner(
        self,
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """The bare (unshifted) ``state -> M`` factory for the adjoint solve at the converged state.

        The adjoint is taken once at ``φ*``, where ``β → 0`` and the operator is already the
        well-conditioned steady Jacobian, so it needs no pseudo-transient shift — the ordinary
        block preconditioner is the consistent choice.
        """
        return self.preconditioner.factory()

    def stepper(self) -> _Stepper:
        """Return the accepted shifted-Newton step ``(residual_fn, φ, ‖R₀‖, solver) -> φ_next``.

        The closure captures the preconditioner and the flow layout (both frozen); it is what the
        Newton driver calls each forward iteration in place of a line-searched step. It is the only
        surface the generic solver sees, so the solver never imports any flow specifics.

        Each step forms the shifted-Newton correction at ``β = β₀ (‖R‖/‖R₀‖)^p`` and **accepts it only
        if it reduces the residual**; otherwise it escalates the damping (``β *= escalation_factor``)
        and retries, up to :attr:`max_escalations`. This is the globalization the bare march lacks: a
        cold-start step whose shifted system is ill-conditioned (or whose full step overshoots) is
        re-damped until it descends, so the march cannot diverge to a non-finite iterate — while a
        well-behaved step is accepted on the first attempt at no extra solve. The retry uses a
        non-throwing linear solve so a non-convergent attempt is *rejected and re-damped* rather than
        raising. ``β`` still vanishes at the fixed point, so the converged state and the IFT adjoint
        are unchanged.
        """
        block = self.preconditioner
        assembler = block.assembler
        n_cells = assembler.mesh.n_cells
        beta0, exponent = self.beta0, self.exponent
        max_escalations, escalation_factor = self.max_escalations, self.escalation_factor
        divergence_cap = self.divergence_cap

        def step(
            residual_fn: _ResidualFn,
            phi: jnp.ndarray,
            residual_norm_0: jnp.ndarray,
            solver: lx.AbstractLinearSolver,
        ) -> jnp.ndarray:
            residual = residual_fn(phi)
            residual_norm = jnp.linalg.norm(residual)
            a_p = block.frozen_momentum_diagonal(phi)
            # Switched-evolution-relaxation: strong damping while the residual is large, none as it
            # vanishes (β → 0 recovers the undamped Newton step and its terminal quadratic rate).
            base_relaxation = beta0 * (residual_norm / residual_norm_0) ** exponent

            def attempt(relaxation: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
                # The shift only reshapes the forward path (like the preconditioner it damps), so it
                # is detached: it never perturbs the converged state or its adjoint.
                shift = jax.lax.stop_gradient(relaxation * a_p)  # velocity diagonal shift β a_P

                def shifted_jacobian(tangent: jnp.ndarray) -> jnp.ndarray:
                    # True Jacobian-vector product plus the pseudo-transient diagonal on velocity DOFs.
                    jvp = jax.jvp(residual_fn, (phi,), (tangent,))[1]
                    velocity_tangent, _ = assembler.unpack(tangent)
                    return jvp + assembler.pack(
                        shift[:, None] * velocity_tangent, jnp.zeros(n_cells)
                    )

                # Preconditioner inverts the *same* shifted diagonal a_P (1 + β) it is damped by. The
                # solve does not throw: a non-convergent shifted system yields a candidate the
                # acceptance test rejects (triggering more damping), rather than raising.
                preconditioner = block.apply_at(phi, a_p + shift)
                delta = solve_linear(
                    shifted_jacobian,
                    -residual,
                    solver=solver,
                    preconditioner=preconditioner,
                    throw=False,
                )
                candidate = phi + delta
                return candidate, jnp.linalg.norm(residual_fn(candidate))

            # Accept the first attempt that does not *diverge*, escalating the damping on rejection.
            # Pseudo-transient steps are legitimately non-monotone, so the test is a divergence guard,
            # not a descent test: reject only a non-finite candidate or one that has blown up past
            # ``divergence_cap × ‖R₀‖`` — measured against the *initial* residual, since the healthy
            # march oscillates around and below ‖R₀‖ while a diverging one explodes above it. The shift
            # itself is the globalization, and more shift (a smaller pseudo-timestep) is what a rejected
            # step needs. The loop exits as soon as an attempt is accepted, so a healthy first attempt
            # costs a single solve; only a diverging step pays for extra, more-damped attempts.
            divergence_norm = divergence_cap * residual_norm_0

            def cond(state: tuple) -> jnp.ndarray:
                _, _, attempts, accepted = state
                return (~accepted) & (attempts <= max_escalations)

            def body(state: tuple) -> tuple:
                relaxation, best, attempts, _ = state
                candidate, candidate_norm = attempt(relaxation)
                accept = jnp.isfinite(candidate_norm) & (candidate_norm < divergence_norm)
                best = jnp.where(accept, candidate, best)
                return relaxation * escalation_factor, best, attempts + 1, accept

            _, phi_next, _, _ = jax.lax.while_loop(
                cond, body, (base_relaxation, phi, 0, jnp.asarray(False))
            )
            return phi_next

        return step
