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
        Initial under-relaxation strength ``β₀`` (static). ``β = β₀`` at the starting residual;
        larger is more robust but slower. The effective SIMPLE velocity relaxation is ``1/(1+β₀)``.
    exponent : float
        Switched-evolution-relaxation exponent ``p`` in ``β = β₀ (‖R‖/‖R₀‖)^p`` (static). ``1``
        ramps the shift linearly with the residual norm.
    """

    preconditioner: BlockPreconditioner
    beta0: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        assembler: MomentumContinuity,
        *,
        beta0: float = 2.0,
        exponent: float = 1.0,
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
        **preconditioner_kwargs
            Forwarded to :meth:`BlockPreconditioner.build` (e.g. ``inner``, ``v_cycles``).
        """
        preconditioner = BlockPreconditioner.build(assembler, **preconditioner_kwargs)
        return cls(preconditioner, beta0, exponent)

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
        """Return the shifted-Newton step ``(residual_fn, φ, ‖R₀‖, solver) -> φ_next``.

        The closure captures the preconditioner and the flow layout (both frozen); it is what the
        Newton driver calls each forward iteration in place of a line-searched step. It is the only
        surface the generic solver sees, so the solver never imports any flow specifics.
        """
        block = self.preconditioner
        assembler = block.assembler
        n_cells = assembler.mesh.n_cells
        beta0, exponent = self.beta0, self.exponent

        def step(
            residual_fn: _ResidualFn,
            phi: jnp.ndarray,
            residual_norm_0: jnp.ndarray,
            solver: lx.AbstractLinearSolver,
        ) -> jnp.ndarray:
            residual = residual_fn(phi)
            # Switched-evolution-relaxation: strong damping while the residual is large, none as it
            # vanishes (β → 0 recovers the undamped Newton step and its terminal quadratic rate).
            relaxation = beta0 * (jnp.linalg.norm(residual) / residual_norm_0) ** exponent
            a_p = block.frozen_momentum_diagonal(phi)
            # The shift only reshapes the forward path (like the preconditioner it damps), so it is
            # detached: it never perturbs the converged state or its adjoint.
            shift = jax.lax.stop_gradient(
                relaxation * a_p
            )  # per-cell velocity diagonal shift β a_P

            def shifted_jacobian(tangent: jnp.ndarray) -> jnp.ndarray:
                # True Jacobian-vector product plus the pseudo-transient diagonal on velocity DOFs.
                jvp = jax.jvp(residual_fn, (phi,), (tangent,))[1]
                velocity_tangent, _ = assembler.unpack(tangent)
                return jvp + assembler.pack(shift[:, None] * velocity_tangent, jnp.zeros(n_cells))

            # Preconditioner inverts the *same* shifted diagonal a_P (1 + β) it is damped by.
            preconditioner = block.apply_at(phi, a_p + shift)
            delta = solve_linear(
                shifted_jacobian, -residual, solver=solver, preconditioner=preconditioner
            )
            return phi + delta

        return step
