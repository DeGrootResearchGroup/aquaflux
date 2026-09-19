"""Pseudo-transient continuation for the coupled flow Newton solve at high Reynolds number.

A line-searched Newton step converges the channel at Re ~ 100 but fails once the flow becomes
convection-dominated (a few hundred Reynolds). As convection strengthens, the undamped Newton step
from the uniform cold start **overshoots** — the full step *increases* the residual, more steeply as
the Reynolds number rises — so the basin the step must land in shrinks and the backtracking line
search must retreat to ever-smaller steps, until it can no longer march the convective path
(it lifts a Reynolds-number floor near ~100 that an undamped Newton solve stalls at).

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
:class:`MomentumShiftPolicy`: the velocity shift and the matching shifted SIMPLE preconditioner.
Because the shift vanishes at the fixed point (``R(φ*) = 0`` exactly), the implicit-function-theorem
adjoint is untouched — continuation only reshapes the forward path.

The ``a_P`` shift above is the **default** :class:`~aquaflux.solve.ShiftBasis`
(:class:`~aquaflux.solve.LocalCourantBasis` at full weight): the velocity shift diagonal is built from
the momentum diagonal's convective and dissipative buckets, and combining them one-to-one gives ``a_P``
(hence the spatially-uniform relaxation described above). A convective-only basis instead gives a
genuine **local convective time step** — the per-cell ``Δt ∝ V / Σ_f max(mdot_f, 0)`` a Courant
condition implies, damping the fast shear layer more than the slow recirculation — while the
preconditioner tracks the actual shifted diagonal ``a_P + β d`` either way.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aquaflux.solve import (
    DEFAULT_GLOBALIZATION,
    Globalization,
    LocalCourantBasis,
    PseudoTransientStep,
    RootSolver,
    ShiftBasis,
    ShiftTerm,
    VelocityShiftParts,
    assembler_residual,
    shifted_step,
)

from .block_preconditioner import BlockPreconditioner

if TYPE_CHECKING:
    from .momentum import MomentumContinuity


class FrozenViscosityVelocityParts(eqx.Module):
    """The velocity shift buckets at the **preconditioner's frozen** effective viscosity (the default).

    Live in velocity, frozen in viscosity: the ``(convective, dissipative)`` buckets come from the block
    preconditioner's own frozen assembler, so ``mu_eff`` is whatever it was at the freeze state. For a
    flow-only solve that is exact, because the viscosity really is constant; for a coupled solve it means
    the velocity time scale ignores the eddy viscosity that develops (use
    :class:`~aquaflux.turbulence.LiveViscosityVelocityParts` there when the shift should track it). This
    is the historical behaviour and the explicit, injectable spelling of the ``None`` default.

    Implements the :class:`~aquaflux.solve.VelocityShiftParts` protocol; it needs only a
    :class:`BlockPreconditioner`, so it lives here in ``flow`` (not with the turbulence-specific live
    variant) and both shift policies can inject it.

    Attributes
    ----------
    block : BlockPreconditioner
        The frozen flow-block preconditioner whose assembler supplies the buckets.
    """

    block: BlockPreconditioner

    def parts(
        self,
        flow: jnp.ndarray,
        k_solved: jnp.ndarray | None = None,
        omega_solved: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del k_solved, omega_solved  # frozen viscosity needs no turbulence context
        return self.block.frozen_momentum_diagonal_parts(flow)


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
        The block-SIMPLE preconditioner, applied at the shifted diagonal ``a_P + β d`` each step, and
        the source of the frozen ``a_P`` (and its convective/dissipative parts) the shift is formed from.
    shift_basis : ShiftBasis
        How the base velocity shift diagonal ``d`` is built from the momentum diagonal's convective and
        dissipative buckets. The default :class:`~aquaflux.solve.LocalCourantBasis` (weight ``1``) is
        ``d = a_P`` -- spatially-uniform under-relaxation, mathematically the historical shift (the
        preconditioner is now fed ``a_P + beta d`` rather than ``a_P (1 + beta)``; equal in exact
        arithmetic, and bitwise equal only for a dyadic ``beta``);
        a convective basis (weight ``0``) gives a genuine local convective time step.
    velocity_shift_parts : VelocityShiftParts or None
        Where the shift's convective/dissipative buckets come from (its own concern, separate from the
        preconditioner's ``a_P``). ``None`` (default) takes them from the frozen preconditioner — live
        in velocity, frozen in viscosity, the historical behaviour and bit-identical. This is the same
        seam the coupled :class:`~aquaflux.turbulence.CoupledShiftPolicy` carries; for a flow-only solve
        the frozen viscosity is exact (constant ``mu``), so it exists for symmetry and for a future
        variable-viscosity flow rather than to change today's behaviour.
    """

    preconditioner: BlockPreconditioner
    shift_basis: ShiftBasis = LocalCourantBasis()
    velocity_shift_parts: VelocityShiftParts | None = None

    def shift_term(self, phi: jnp.ndarray, residual: jnp.ndarray | None = None) -> ShiftTerm:
        """The base velocity shift diagonal and the ``β -> M`` shifted preconditioner at ``phi``.

        Parameters
        ----------
        phi : jnp.ndarray
            The flat coupled state ``[vel_0..vel_{dim-1}, pressure]``, shape ``((dim + 1) n_cells,)``.

        Returns
        -------
        ShiftTerm
            ``diagonal`` places the per-cell base shift ``d`` (from :attr:`shift_basis`) on every
            velocity component and zero on pressure (the full-state base shift); ``make_preconditioner(β)``
            returns the block preconditioner at the shifted diagonal ``a_P + β d``.
        """
        block = self.preconditioner
        # `a_P` is a PRECONDITIONER quantity (what the velocity block inverts), always the frozen
        # diagonal. The SHIFT's buckets are a separate concern with their own lifetime, so their source
        # is injected (:class:`FrozenViscosityVelocityParts` is its explicit default spelling); `None`
        # reuses the preconditioner's frozen parts inline, keeping the default path bit-identical.
        convective, dissipative = block.frozen_momentum_diagonal_parts(phi)
        a_p = convective + dissipative  # the isotropic frozen a_P the velocity block inverts at
        if self.velocity_shift_parts is not None:
            convective, dissipative = self.velocity_shift_parts.parts(phi)
        d = self.shift_basis.local_diagonal(
            convective, dissipative
        )  # base shift diagonal (n_cells,)
        assembler = block.assembler
        n_cells = assembler.mesh.n_cells
        # d on every velocity component, zero on pressure — the full-state base shift. The engine
        # scales it by β and adds β·d to the Jacobian diagonal (velocity DOFs only).
        diagonal = assembler.pack(
            jnp.broadcast_to(d[:, None], (n_cells, assembler.mesh.dim)), jnp.zeros(n_cells)
        )

        def make_preconditioner(
            relaxation: jnp.ndarray,
        ) -> Callable[[jnp.ndarray], jnp.ndarray]:
            # Invert the same shifted diagonal a_P + β·d the shift adds to the Jacobian, so the
            # preconditioner matches the shifted operator. Frozen: the coefficient is detached.
            return block.apply_at(phi, jax.lax.stop_gradient(a_p + relaxation * d))

        return ShiftTerm(diagonal, make_preconditioner)


def momentum_shift_policy(
    assembler: MomentumContinuity,
    shift_basis: ShiftBasis | None = None,
    **preconditioner_kwargs: object,
) -> MomentumShiftPolicy:
    """The flow shift policy over a block-SIMPLE preconditioner built for ``assembler``.

    The one place the flow's preconditioner and its shift policy are assembled, shared by every builder
    of a flow step.

    Parameters
    ----------
    assembler : MomentumContinuity
        The coupled flow residual assembler.
    shift_basis : ShiftBasis, optional
        How the velocity shift diagonal is built from the momentum diagonal's parts. ``None`` keeps
        :class:`MomentumShiftPolicy`'s own default, the full ``a_P``.
    **preconditioner_kwargs
        Forwarded to :meth:`BlockPreconditioner.build`.

    Returns
    -------
    MomentumShiftPolicy
        The policy, holding the built preconditioner.
    """
    preconditioner = BlockPreconditioner.build(assembler, **preconditioner_kwargs)
    return (
        MomentumShiftPolicy(preconditioner)
        if shift_basis is None
        else MomentumShiftPolicy(preconditioner, shift_basis)
    )


def momentum_continuation(
    assembler: MomentumContinuity,
    *,
    globalization: Globalization = DEFAULT_GLOBALIZATION,
    shift_basis: ShiftBasis | None = None,
    **preconditioner_kwargs: object,
) -> PseudoTransientStep:
    """The pseudo-transient continuation ``NewtonStrategy`` for the coupled flow solve.

    Builds the block-SIMPLE preconditioner for ``assembler`` and wires a :class:`MomentumShiftPolicy`
    (the velocity ``a_P`` shift + the matching shifted preconditioner) into the residual-agnostic
    :class:`~aquaflux.solve.PseudoTransientStep` engine, which owns the schedule, the shifted solve,
    and the accept/escalate loop. The result plugs straight into
    :class:`~aquaflux.solve.RootSolver` as its ``strategy`` (the forward Newton loop
    uses the diagonally shifted step in place of the default line search; the converged,
    well-conditioned adjoint solve uses the bare block preconditioner, since at ``φ*`` the shift has
    vanished). ``PseudoTransientStep`` is itself the ``NewtonStrategy``, so no wrapper is needed.

    Parameters
    ----------
    assembler : MomentumContinuity
        The coupled flow residual assembler.
    globalization : Globalization
        How hard the march damps and what it does when a step misbehaves — the schedule
        ``β = β₀(‖R‖/‖R₀‖)^p`` (whose ``β a_P`` shift is SIMPLE velocity under-relaxation at
        ``1/(1+β)``), the escalation ladder, the divergence guard and the backtracking ladder. Only
        the fields it sets are applied. Left unset, this march **takes the full shifted step**: on this
        residual the shift is the globalization, and the escalation ladder alone carries the convective
        regime from a cold start (it is what lifts the Reynolds floor this module exists for). That is
        a default, not a restriction — the coupled RANS residual, whose full step overshoots by orders
        of magnitude, line-searches instead, and this path can be given the same.
    shift_basis : ShiftBasis, optional
        How the velocity shift diagonal is built from the momentum diagonal's convective/dissipative
        parts (see :class:`MomentumShiftPolicy`). Defaults to
        :class:`~aquaflux.solve.LocalCourantBasis` — the full ``a_P`` (uniform under-relaxation),
        unchanged from the historical shift. Pass ``LocalCourantBasis(dissipative_weight=0.0)`` for a
        local convective time step.
    **preconditioner_kwargs
        Forwarded to :meth:`BlockPreconditioner.build` (e.g. ``schur_scaling``, ``composition``,
        ``velocity``).

    Returns
    -------
    PseudoTransientStep
        The configured continuation, ready to pass as ``RootSolver(strategy=...)``.
    """
    policy = momentum_shift_policy(assembler, shift_basis, **preconditioner_kwargs)
    return shifted_step(
        policy,
        globalization=globalization,
        dual_time=None,
        regime=None,
        krylov_solver=None,
        adjoint_preconditioner_factory=policy.preconditioner.factory(),
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
    and recompiles the march step — a per-sweep cost that grows with mesh size. This helper builds
    the continuation once from ``reference``, so a sweep changes only the viscosity *values* passed as
    the residual parameters: the compiled step is reused and nothing is rebuilt.

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
        Forwarded to :func:`momentum_continuation` (e.g. ``schur_scaling="msimple"``,
        ``velocity=ConvectionTwoLevel()``, ``globalization``).

    Returns
    -------
    callable
        ``solve_flow(momentum, state) -> state`` solving ``momentum.residual`` from ``state`` with the
        frozen preconditioned continuation. Reverse-differentiable in ``momentum`` (the
        implicit-function-theorem adjoint is one transpose solve at the root, whatever path the march
        took to reach it). Like every march, it steps in Python and so cannot itself be called from inside
        a traced program -- ``jax.jit``, ``jax.vmap``, or a traced loop such as ``jax.lax.scan``.
    """
    continuation = momentum_continuation(reference, **build_kwargs)
    solver = RootSolver(max_steps=max_steps, strategy=continuation)

    def solve_flow(momentum: MomentumContinuity, state: jnp.ndarray) -> jnp.ndarray:
        # `assembler_residual` rather than a lambda: the march compiles its step with the residual as
        # an argument, so a closure built here would be a fresh cache key on every sweep and recompile
        # the solve this helper exists to reuse. `momentum` rides as the parameter, whose arrays are
        # dynamic leaves, so a sweep that changes only the viscosity is a cache hit.
        return solver.solve(assembler_residual, state, momentum)

    return solve_flow
