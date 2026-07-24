"""The monolithic coupled RANS residual ``R(u, p, k, omega)``.

The segregated driver (:mod:`~aquaflux.turbulence.driver`) is a *forward* convergence device: it
freezes the eddy viscosity for the flow solve and the flow for the turbulence solve, Picard-iterating
to the fixed point. This module assembles the same physics as **one residual over the full unknown**
``[u..., p, k, omega]`` with nothing frozen -- the eddy viscosity ``nu_t(k, omega, grad u)``, the mean
strain ``S(u)``, and the Rhie--Chow mass flux ``mdot(u, p)`` are live functions of the state, so a
single Newton solve sees the exact cross-block coupling.

Why monolithic, when the segregated loop already converges? Two reasons, both in the turbulence
design note (S5): a monolithic Newton reaches **quadratic** coupled convergence the Picard loop
cannot, and -- handed to :class:`~aquaflux.solve.ImplicitNewtonSolver` -- it yields the **exact
coupled adjoint** as a single transpose solve on the unfrozen ``R_coupled`` at the converged state.
The segregated loop is retained as a robust startup pre-smoother / fallback, not the sensitivity
model.

Positivity of ``k, omega`` under a full Newton step. With the default :class:`DirectScalars`
parametrization it is carried by the pseudo-transient continuation
(:mod:`~aquaflux.turbulence.continuation` block policy): the shift damps the step heavily far from the
fixed point, and a step that drives ``k`` or ``omega`` non-positive makes the closure non-finite
through ``sqrt(k)`` -- rejected by the divergence guard, which escalates the damping. That is not
airtight at high Reynolds number: a full step can drive ``omega`` negative while the residual stays
finite (``nu_t = k/omega`` flips sign without a NaN), so the guard never trips. The **log-variable**
parametrization (:class:`LogScalars` on ``omega``) is the structural fix -- ``omega = e^w > 0`` for
every ``w`` -- and is exact for the adjoint because the realizability floor stays **out** of this
residual (a converged RANS field is strictly positive, so the floor is inactive and the coupled
adjoint sees only the smooth interior physics).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from aquaflux.flow import BlockPreconditioner

# The mass-flow-constraint primitives (a body force that is a solve unknown enforcing a bulk velocity)
# are shared with the flow-block solve `aquaflux.flow.bulk_velocity_flow_solve`: the border column/row,
# the Schur (constraint) preconditioner, and the body-force setter. Reused here rather than re-deriving
# the Schur elimination, which one careful place keeps consistent.
from aquaflux.flow.mean_velocity import (
    _bordered_preconditioner,
    _constraint_vectors,
    _with_body_force,
)
from aquaflux.solve import (
    BlockScaledNorm,
    DivergenceGuard,
    ImplicitNewtonSolver,
    PseudoTransientStep,
    RefreshTrigger,
    ShiftTerm,
    StepReport,
    forward_march,
)

from .initialization import hybrid_initialize
from .preconditioner import ScalarTransportPreconditioner, ScaledScalarPreconditioner

if TYPE_CHECKING:
    from aquaflux.flow import MomentumContinuity

    from .transport import SSTTurbulence


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
    :func:`coupled_continuation`).
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


class DirectScalars(ScalarVariableTransform):
    """The identity parametrization: the solved unknown *is* the physical field (``phi = w``).

    Positivity is not structural here -- it is carried by the pseudo-transient shift and the
    realizability floor -- so a full Newton step can transiently violate ``omega > 0`` on a stiff
    high-Reynolds case. The historical default; use :class:`LogScalars` where that matters.
    """

    def to_physical(self, w: jnp.ndarray) -> jnp.ndarray:
        return w

    def to_solved(self, phi: jnp.ndarray) -> jnp.ndarray:
        return phi

    def jacobian_scale(self, phi: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(phi)


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


class CoupledRANSLayout(eqx.Module):
    """Pack/unpack of the flat coupled state ``[flow..., k, omega]``.

    The flow block is the momentum assembler's own ``[vel_0..vel_{dim-1}, pressure]`` layout
    (:class:`~aquaflux.flow.state.BlockStateLayout`) carried verbatim, so the flow sub-vector is
    handed to :class:`~aquaflux.flow.MomentumContinuity` unchanged; ``k`` and ``omega`` follow as two
    ``n_cells``-long blocks. Mesh-free and testable in isolation, mirroring ``BlockStateLayout``.

    Attributes
    ----------
    dim : int
        Number of velocity components (spatial dimension), static.
    n_cells : int
        Number of cells (each scalar block's length), static.
    """

    dim: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)

    @property
    def flow_size(self) -> int:
        """Length of the flow sub-vector, ``(dim + 1) * n_cells``."""
        return (self.dim + 1) * self.n_cells

    @property
    def size(self) -> int:
        """Length of the full coupled state, ``(dim + 3) * n_cells``."""
        return (self.dim + 3) * self.n_cells

    def unpack(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Split the coupled state into the flow sub-vector, ``k``, and ``omega``.

        Parameters
        ----------
        state : jnp.ndarray
            Flat coupled state, shape ``((dim + 3) * n_cells,)``.

        Returns
        -------
        flow, k, omega : jnp.ndarray
            The flat flow state ``((dim + 1) n_cells,)`` and the two fields ``(n_cells,)``.
        """
        n = self.n_cells
        flow_size = self.flow_size
        flow = state[:flow_size]
        k = state[flow_size : flow_size + n]
        omega = state[flow_size + n :]
        return flow, k, omega

    def pack(self, flow: jnp.ndarray, k: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
        """Assemble the flow sub-vector and the two fields into the flat coupled state.

        Parameters
        ----------
        flow : jnp.ndarray
            The flat flow state ``[vel..., pressure]``, shape ``((dim + 1) n_cells,)``.
        k, omega : jnp.ndarray
            The turbulence fields, shape ``(n_cells,)``.
        """
        return jnp.concatenate([flow, k, omega])


class CoupledRANS(eqx.Module):
    """The monolithic ``R(u, p, k, omega)`` assembler.

    Holds the flow and turbulence assemblers and composes their residuals
    with **live** coupling: each :meth:`residual` evaluation recomputes ``nu_t`` and the closure from
    the current ``(k, omega, grad u)``, re-viscosifies the momentum block, and advects ``k`` / ``omega``
    on the current Rhie--Chow flux. The whole module is the differentiable parameter pytree ``theta``
    for the coupled implicit-function-theorem adjoint.

    Attributes
    ----------
    momentum : MomentumContinuity
        The flow assembler; its molecular ``viscosity`` property is overwritten by ``mu_eff`` each
        evaluation (the molecular viscosity comes from ``turbulence``).
    turbulence : SSTTurbulence
        The k-omega SST closure and equation assembler.
    k_transform, omega_transform : ScalarVariableTransform
        The change of variable between each solved turbulence unknown and its physical field (default
        :class:`DirectScalars`, the identity). :class:`LogScalars` makes that field ``> 0`` by
        construction under any Newton step. The two are **independent** on purpose: ``omega`` is the
        field a full Newton step drives negative at high Reynolds number, and ``log(omega)`` is
        well-conditioned (``omega`` is bounded away from zero -- large near walls); ``log(k)`` is not,
        because ``k -> 0`` at a no-slip wall (its Dirichlet value), so ``log(k) -> -inf`` there stalls
        the near-wall cells. The productive high-Reynolds configuration is therefore ``omega`` log, ``k``
        direct -- ``CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())``.
    """

    momentum: MomentumContinuity
    turbulence: SSTTurbulence
    k_transform: ScalarVariableTransform = DirectScalars()
    omega_transform: ScalarVariableTransform = DirectScalars()

    @classmethod
    def build(
        cls,
        momentum: MomentumContinuity,
        turbulence: SSTTurbulence,
        k_transform: ScalarVariableTransform | None = None,
        omega_transform: ScalarVariableTransform | None = None,
    ) -> CoupledRANS:
        """Assemble the coupled system, pre-resolving the turbulence boundaries off the jit path.

        The turbulence residual rebuilds its scalar :class:`~aquaflux.discretization.ResidualAssembler`
        each evaluation, and that build resolves the k/omega boundary patches -- a dynamic-shape
        ``nonzero`` lookup on the mesh labels that cannot run inside the coupled residual's jit. Binding
        those boundaries **once here** (the momentum boundary is already resolved by
        :meth:`~aquaflux.flow.MomentumContinuity.build`) makes the per-evaluation rebuild's ``resolve``
        an idempotent no-op, so the whole coupled residual is jit- and adjoint-safe.

        ``k_transform`` / ``omega_transform`` select each scalar's parametrization (default
        :class:`DirectScalars`); pass ``omega_transform=LogScalars()`` for the productive
        ``omega`` log / ``k`` direct high-Reynolds combination.
        """
        return cls(
            momentum,
            turbulence.resolve_boundaries(),
            k_transform or DirectScalars(),
            omega_transform or DirectScalars(),
        )

    def physical_fields(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Unpack a coupled state into the flow sub-vector and the **physical** ``k``, ``omega``.

        Applies each :meth:`ScalarVariableTransform.to_physical` to its solved scalar block, so the
        result is the physical fields regardless of the parametrization -- what a caller (and the
        closure) always wants. This is the inverse of :meth:`state_from_physical`.
        """
        flow, k_solved, omega_solved = self.layout.unpack(state)
        return (
            flow,
            self.k_transform.to_physical(k_solved),
            self.omega_transform.to_physical(omega_solved),
        )

    def state_from_physical(
        self, flow: jnp.ndarray, k: jnp.ndarray, omega: jnp.ndarray
    ) -> jnp.ndarray:
        """Pack a flow sub-vector and **physical** ``k``, ``omega`` into a coupled state.

        Applies each :meth:`ScalarVariableTransform.to_solved`, so a physical initial condition (e.g.
        from :func:`~aquaflux.turbulence.hybrid_initialize`) is mapped into the solved variable space.
        """
        return self.layout.pack(
            flow, self.k_transform.to_solved(k), self.omega_transform.to_solved(omega)
        )

    @property
    def layout(self) -> CoupledRANSLayout:
        """The coupled state layout ``[flow..., k, omega]`` for this system."""
        return CoupledRANSLayout(self.momentum.mesh.dim, self.momentum.mesh.n_cells)

    def pack_state(self, flow: jnp.ndarray, k: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
        """Assemble a coupled state from a flow state and the two turbulence fields."""
        return self.layout.pack(flow, k, omega)

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        """The coupled residual ``R(u, p, k, omega)`` for the flat state, same shape as ``state``.

        Assembled with nothing frozen: ``nu_t`` and the SST closure are recomputed from the current
        ``(k, omega, grad u)``, the momentum block runs on ``mu_eff = rho (nu + nu_t)``, and both
        scalars advect on the current Rhie--Chow flux. The near-wall ``omega`` rows are the analytical
        fixation carried by :meth:`~aquaflux.turbulence.SSTTurbulence.omega_residual`.

        The scalar blocks of ``state`` hold the *solved* turbulence unknown; the physics below is
        written in the physical ``k`` / ``omega`` recovered by the :attr:`transform` (the identity for
        :class:`DirectScalars`, ``e^w`` for :class:`LogScalars`). The returned scalar residuals are the
        physical transport residuals ``R_k(k, omega)`` / ``R_omega(k, omega)`` -- the same root either
        way -- so the reparametrization changes only the Newton iterate space, and automatic
        differentiation supplies the chain-rule Jacobian.
        """
        flow, k, omega = self.physical_fields(state)

        # The closure carries nu_t and the mean strain, so build it first and take nu_t from it --
        # eddy_viscosity would otherwise recompute the same strain and nu_t the closure already forms.
        closure = self.turbulence.closure_fields(self.momentum.velocity_fields(flow), k, omega)
        momentum = self.momentum.with_eddy_viscosity(
            closure.nu_t, self.turbulence.wall_face_eddy_viscosity(k)
        )

        # One Rhie--Chow assembly at the re-viscosified state feeds both the flow residual and the
        # mass flux the scalars advect on.
        fields = momentum.flow_fields(flow)
        flow_residual = momentum.residual_from_fields(fields)
        k_residual = self.turbulence.k_residual(fields.mdot, closure)(k)
        omega_residual = self.turbulence.omega_residual(fields.mdot, closure)(omega)

        return self.layout.pack(flow_residual, k_residual, omega_residual)


class CoupledShiftPolicy(eqx.Module):
    """The block :class:`~aquaflux.solve.continuation.ShiftPolicy` for the coupled Newton solve.

    Composes the three subsystems' pseudo-transient choices block-diagonally: the momentum block's
    ``a_P`` velocity shift + block-SIMPLE preconditioner (:class:`~aquaflux.flow.MomentumShiftPolicy`),
    and the k and omega transport-operator shift diagonals + convection-diffusion AMGs
    (:class:`~aquaflux.turbulence.continuation.ScalarShiftPolicy`). The full-state shift diagonal is
    ``[a_P on u, 0 on p, d_k on k, d_omega on omega]`` and the preconditioner is the block-diagonal
    matvec gluing the flow preconditioner to the two scalar AMGs.

    The AMG hierarchies and the (numpy-assembled) scalar shift diagonals are **frozen at a reference
    state** (built off-jit by :func:`coupled_continuation`) and carried here as data, exactly as
    :func:`~aquaflux.flow.reused_flow_solve` freezes the flow preconditioner: a pseudo-transient shift
    and its preconditioner are transient devices that vanish at the fixed point, so freezing their
    coefficients at a representative state costs only Krylov iterations, never correctness. The
    velocity ``a_P`` is the one piece recomputed live per iterate (it is a cheap jittable read of the
    momentum diagonal), so the velocity damping still tracks the developing convection.

    Attributes
    ----------
    layout : CoupledRANSLayout
        The coupled state layout, for packing the block-diagonal shift and preconditioner.
    flow_preconditioner : BlockPreconditioner
        The block-SIMPLE preconditioner built at the reference effective viscosity; supplies the
        frozen ``a_P`` and the velocity/Schur solves.
    k_shift_diagonal, omega_shift_diagonal : jnp.ndarray
        The frozen per-cell transport-operator shift diagonals for k and omega, shape ``(n_cells,)``
        (the omega one has its near-wall fixed cells zeroed).
    k_preconditioner, omega_preconditioner : callable or None
        The frozen ``phi -> M`` convection-diffusion AMG factories for the k and omega blocks, or
        ``None`` for an unpreconditioned (identity) scalar block.
    """

    layout: CoupledRANSLayout
    flow_preconditioner: BlockPreconditioner
    k_shift_diagonal: jnp.ndarray
    omega_shift_diagonal: jnp.ndarray
    k_preconditioner: ScalarTransportPreconditioner | None = None
    omega_preconditioner: ScalarTransportPreconditioner | None = None

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        """The block-diagonal full-state shift and the ``beta -> M`` composed preconditioner at ``phi``.

        Parameters
        ----------
        phi : jnp.ndarray
            The flat coupled state ``[flow..., k, omega]``, shape ``((dim + 3) n_cells,)``.
        """
        flow, k, omega = self.layout.unpack(phi)
        block = self.flow_preconditioner
        assembler = block.assembler
        n_cells = self.layout.n_cells
        a_p = block.frozen_momentum_diagonal(flow)  # live per-iterate velocity a_P (jittable)

        # Full-state base shift d: a_P on every velocity component, 0 on pressure, the frozen scalar
        # transport diagonals on k and omega.
        flow_diagonal = assembler.pack(
            jnp.broadcast_to(a_p[:, None], (n_cells, self.layout.dim)), jnp.zeros(n_cells)
        )
        diagonal = self.layout.pack(
            flow_diagonal,
            jax.lax.stop_gradient(self.k_shift_diagonal),
            jax.lax.stop_gradient(self.omega_shift_diagonal),
        )

        def make_preconditioner(relaxation: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
            # Flow block at the under-relaxed a_P (1 + beta) matching the shifted Jacobian; scalar
            # blocks at their frozen AMG (beta-independent -- the shift only adds positive diagonal).
            flow_m = block.apply_at(flow, jax.lax.stop_gradient(a_p * (1.0 + relaxation)))
            k_m = None if self.k_preconditioner is None else self.k_preconditioner(k)
            omega_m = (
                None if self.omega_preconditioner is None else self.omega_preconditioner(omega)
            )

            def precondition(x: jnp.ndarray) -> jnp.ndarray:
                x_flow, x_k, x_omega = self.layout.unpack(x)
                y_k = x_k if k_m is None else k_m(x_k)
                y_omega = x_omega if omega_m is None else omega_m(x_omega)
                return self.layout.pack(flow_m(x_flow), y_k, y_omega)

            return precondition

        return ShiftTerm(diagonal, make_preconditioner)

    def adjoint_factory(self) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """The ``state -> M`` factory for the adjoint transpose solve (the composition at ``beta = 0``).

        At the converged state the pseudo-transient shift vanishes, so the adjoint preconditions the
        unshifted coupled Jacobian with the block-diagonal composition at ``a_P`` -- the same frozen
        flow and scalar preconditioners, transposed by the implicit solver.
        """
        return lambda state: self.shift_term(state).make_preconditioner(jnp.asarray(0.0))


def _reparametrized_preconditioner(
    preconditioner: ScalarTransportPreconditioner | None, jacobian_scale: jnp.ndarray
) -> ScalarTransportPreconditioner | None:
    """Rescale a frozen physical-operator scalar preconditioner for the reparametrized block.

    The reparametrized Jacobian's inverse carries a leading ``diag(1 / jacobian_scale)``, so the
    physical-operator preconditioner is wrapped to apply it (:class:`ScaledScalarPreconditioner`). For
    the identity transform ``jacobian_scale`` is one, so the preconditioner is returned unchanged and
    the direct path stays bit-identical. The scale is materialized off the jit path (the reference
    state is concrete), matching the frozen hierarchy it wraps.
    """
    if preconditioner is None:
        return None
    scale = np.asarray(jacobian_scale)
    if np.allclose(scale, 1.0):
        return preconditioner
    return ScaledScalarPreconditioner(preconditioner, 1.0 / scale)


# The shifted forward solve for the coupled march. Restarted GMRES with a larger Krylov subspace than
# the shared default (restart 40 -> 120): the coupled turbulent saddle system is stiff enough that a
# 40-vector restart discards too much Arnoldi history and converges only after hundreds of restart
# cycles, whereas a 120-vector subspace reaches the same tight solution in far fewer (measured ~1.4x
# faster and to a tighter residual on the ~12k-cell backward-facing step). The tolerances stay tight
# (an inexact/loose linear solve is unsafe here -- an inaccurate step in the log-omega variable is
# exponentiated and diverges), so the accuracy the log-variable closure needs is preserved.
_COUPLED_FORWARD_SOLVER = lx.GMRES(rtol=1e-3, atol=1e-10, restart=120, stagnation_iters=40)

# Backtracking rungs for the shifted step. The full coupled Newton step from the hybrid initial
# condition overshoots violently (the residual blows up many orders of magnitude), so the step length
# is scaled back along {1, 1/2, ..., 1/2**N} until it descends -- recovering a residual-reducing step
# from the one expensive shifted solve, instead of escalating beta (a full re-solve, which changes the
# direction and, measured, does not descend on this case). Ten rungs reach 1/1024, well past the
# ~1/4 the stiff first steps need.
_COUPLED_LINE_SEARCH = 10


def _coupled_block_scales(coupled: CoupledRANS, reference_state: jnp.ndarray) -> tuple[float, ...]:
    """The per-field reference residual magnitudes ``(‖R_flow‖, ‖R_k‖, ‖R_omega‖)`` at
    ``reference_state``, each floored positive so it can divide a block norm."""
    parts = coupled.layout.unpack(coupled.residual(reference_state))
    return tuple(max(float(jnp.linalg.norm(part)), 1e-30) for part in parts)


def _coupled_residual_norm(coupled: CoupledRANS, reference_state: jnp.ndarray) -> BlockScaledNorm:
    """The opt-in block-scaled residual norm over ``[flow, k, omega]`` (``block_scaled_norm=True``).

    Each field's residual is divided by its own initial magnitude before the norm is formed, so the
    switched-evolution-relaxation ramp, the line search, and the outer stopping test all judge every
    field rather than the ``omega`` block that dominates the plain Euclidean norm (``omega`` is
    O(1e5) here, ``k`` O(1e-3)): with the plain norm a step that collapses ``k`` barely moves ‖R‖ and
    is accepted. It weighs every block, but was found to *stall* the pitzDaily march (the per-block
    relative norm plateaus long before the fields converge), so the march uses the Euclidean norm by
    default and this is available only when ``block_scaled_norm=True`` is requested.
    """
    n = coupled.momentum.mesh.n_cells
    sizes = (coupled.layout.flow_size, n, n)
    return BlockScaledNorm(sizes, _coupled_block_scales(coupled, reference_state))


def _mass_flow_residual_norm(coupled: CoupledRANS, reference_state: jnp.ndarray) -> BlockScaledNorm:
    """The :func:`_coupled_residual_norm` measure extended with the mass-flow constraint dof.

    The bordered march carries the augmented residual ``[R_flow, R_k, R_omega, ⟨U⟩ − target]``; the
    trailing scalar constraint is a bulk-velocity (velocity-magnitude) equation, so it shares the
    flow block's reference scale.
    """
    n = coupled.momentum.mesh.n_cells
    s_flow, s_k, s_omega = _coupled_block_scales(coupled, reference_state)
    sizes = (coupled.layout.flow_size, n, n, 1)
    return BlockScaledNorm(sizes, (s_flow, s_k, s_omega, s_flow))


def coupled_continuation(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    *,
    method: str | None = "twolevel",
    beta0: float = 2.0,
    exponent: float = 1.0,
    beta_floor: float = 0.0,
    max_escalations: int = 6,
    escalation_factor: float = 2.0,
    divergence_cap: float = 10.0,
    line_search: int = _COUPLED_LINE_SEARCH,
    forward_solver: lx.AbstractLinearSolver | None = None,
    block_scaled_norm: bool = False,
    reuse: CoupledShiftPolicy | None = None,
    **preconditioner_kwargs: object,
) -> PseudoTransientStep:
    """Build the pseudo-transient continuation step for the coupled Newton solve.

    Freezes the block-diagonal preconditioner (flow block-SIMPLE + the k/omega convection-diffusion
    AMGs) and the scalar shift diagonals at ``reference_state`` -- off the jit path, since their AMG
    hierarchies and numpy-assembled diagonals are data-dependent -- and wraps them in a
    :class:`CoupledShiftPolicy`. The reference should be a representative (e.g. segregated pre-smoothed)
    state so the frozen effective viscosity, mass flux, and closure match the operating point.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler.
    reference_state : jnp.ndarray
        The coupled state the preconditioner and shift diagonals are frozen at.
    method : {"twolevel", "air"} or None
        The AMG method for the k and omega blocks (``None`` leaves those blocks unpreconditioned).
    beta0, exponent, beta_floor, max_escalations, escalation_factor, divergence_cap
        The pseudo-transient schedule and divergence-guard parameters (see
        :class:`~aquaflux.solve.PseudoTransientStep`). ``beta_floor`` (default ``0`` = off) bounds the
        switched-evolution-relaxation ``β`` below to keep the shifted solve out of the ill-conditioned
        low-``β`` regime; it never moves the converged root, only damps the path.
    line_search : int
        Backtracking step-halvings applied to the shifted step before it is judged (default
        :data:`_COUPLED_LINE_SEARCH`); scales an accurate-but-overshooting direction back to a descent
        from the one shifted solve rather than re-solving at larger ``beta``. See
        :class:`~aquaflux.solve.PseudoTransientStep`.
    forward_solver : lineax.AbstractLinearSolver or None
        The shifted-solve Krylov solver; ``None`` uses :data:`_COUPLED_FORWARD_SOLVER` (a
        larger-restart GMRES suited to the stiff coupled system).
    block_scaled_norm : bool
        Which residual measure the march judges progress by (default ``False`` = the plain Euclidean
        norm). When ``True`` the march uses a :class:`~aquaflux.solve.BlockScaledNorm` over
        ``[flow, k, omega]`` (each block divided by its own initial magnitude), so the globalization
        weighs every field rather than the ``omega`` block that dominates the Euclidean norm. The
        block-scaled measure was found to *stall* the pitzDaily march (the per-block relative norm stops
        descending long before the fields converge), so it is available for experimentation but off by
        default; the Euclidean norm is what the solver uses.
    reuse : CoupledShiftPolicy, optional
        An existing policy to **refresh** at ``reference_state`` instead of building one from scratch:
        the k/omega AMGs are re-derived on their reused coarsening while the flow block is carried over
        untouched (see :func:`_coupled_shift_policy`). Use it to re-freeze a stale preconditioner part
        way through a march, once the flow has developed.
    **preconditioner_kwargs
        Forwarded to :meth:`~aquaflux.flow.BlockPreconditioner.build` for the flow block (e.g.
        ``schur_scaling``, ``velocity``). Ignored when ``reuse`` is given, since the flow block is then
        carried over rather than rebuilt.

    Returns
    -------
    PseudoTransientStep
        The forward step to hand :class:`~aquaflux.solve.ImplicitNewtonSolver` as ``forward_step``.
    """
    policy = _coupled_shift_policy(coupled, reference_state, method, reuse, **preconditioner_kwargs)
    residual_norm = (
        _coupled_residual_norm(coupled, reference_state) if block_scaled_norm else jnp.linalg.norm
    )
    return PseudoTransientStep(
        policy,
        beta0=beta0,
        exponent=exponent,
        beta_floor=beta_floor,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        acceptance=DivergenceGuard(divergence_cap=divergence_cap),
        line_search=line_search,
        forward_solver=forward_solver if forward_solver is not None else _COUPLED_FORWARD_SOLVER,
        residual_norm=residual_norm,
        adjoint_preconditioner_factory=policy.adjoint_factory(),
    )


def _coupled_shift_policy(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    method: str | None,
    reuse: CoupledShiftPolicy | None = None,
    **preconditioner_kwargs: object,
) -> CoupledShiftPolicy:
    """Build the block-diagonal :class:`CoupledShiftPolicy` frozen at ``reference_state``.

    The preconditioner-freezing half of :func:`coupled_continuation`, split out so the mass-flow
    constraint (:func:`mass_flow_coupled_continuation`) can border the *same* policy rather than
    re-derive it.

    ``reuse`` **refreshes** an existing policy at a new (more developed) ``reference_state`` rather than
    building one from scratch, and **only the scalar k/omega AMGs are re-derived** on their reused
    coarsening (:func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner`'s ``reuse=``)
    -- on a separated backward-facing-step state that is worth ~2.4x in outer Krylov cycles. Everything
    else is **carried over from ``reuse`` untouched**: the flow block (re-freezing it at the developed
    state was measured no help and slightly harmful, and it is the expensive half), *and the pseudo-time
    shift diagonals*.

    **The shift diagonals must be carried, not rebuilt (binding -- rebuilding them freezes the march).**
    The coupled shift diagonal is the transport-operator diagonal times ``jacobian_scale(field)``, which
    under :class:`LogScalars` is ``omega``; at a developed state both factors grow (stiffer operator,
    larger ``omega``), so a rebuilt diagonal ``d`` is much larger, the pseudo-transient shift ``beta d``
    over-damps, and the Newton step collapses to a standstill -- the relative residual creeps *upward*
    ~1e-5 per step with the recirculation and ``k`` static, no error and no divergence-guard trip.
    (Isolated by a controlled discriminator: from one post-stage-one state, rebuilding the shift and
    carrying the AMG froze the march *byte-identically* to rebuilding both, while carrying the shift and
    refreshing only the AMG descended cleanly -- so the shift rebuild is the freeze, independent of the
    switched-evolution-relaxation ``beta``.) This is safe for the same reason the flow block is carried:
    the shift is a transient device that vanishes at the root, so a slightly-stale ``d`` changes only the
    path, never the converged state or its adjoint. A non-refresh build (``reuse is None``) still builds
    the shift at ``reference_state`` as before.
    """
    # The reference's scalar blocks are the *solved* unknown; the frozen operators (closure, AMG, shift
    # diagonals) are all assembled in the physical fields, so recover them through the transform.
    flow_ref, k_ref, omega_ref = coupled.physical_fields(reference_state)
    closure = coupled.turbulence.closure_fields(
        coupled.momentum.velocity_fields(flow_ref), k_ref, omega_ref
    )
    momentum = coupled.momentum.with_eddy_viscosity(closure.nu_t)
    # The coupled flow block uses the convection-aware velocity AMG + MSIMPLER Schur, not the viscous-
    # smoothed / SIMPLE default: a RANS case is high-Reynolds, and the Peclet-blind smoothed velocity
    # block with the ``a_P`` Schur produces a poor momentum-block direction once the flow separates
    # (the shifted Newton direction was measured only ~40% aligned with the true one on the developed
    # pitzDaily field, stalling the march). The convection block's convective linearization and the
    # MSIMPLER Schur's velocity-independent scaling both stay valid frozen at the cold initial state
    # (the reference), so no per-sweep refresh is needed. Overridable via preconditioner_kwargs.
    block = (
        reuse.flow_preconditioner  # measured: re-freezing the flow block does not help
        if reuse is not None
        else BlockPreconditioner.build(
            momentum,
            **{"velocity": "convection", "schur_scaling": "msimpler", **preconditioner_kwargs},
        )
    )

    mdot = momentum.mass_flux(flow_ref)

    # The reparametrized block's Jacobian is the physical one scaled by d(phi)/d(w): its shift diagonal
    # is scaled by that factor and its (physical-operator) preconditioner by the reciprocal. For the
    # identity transform the factor is one, so the direct path is unchanged.
    k_scale = coupled.k_transform.jacobian_scale(k_ref)
    omega_scale = coupled.omega_transform.jacobian_scale(omega_ref)

    k_amg = omega_amg = None
    if method is not None:
        k_amg = _reparametrized_preconditioner(
            coupled.turbulence.k_preconditioner(
                mdot,
                closure,
                k_ref,
                method=method,
                reuse=None if reuse is None else reuse.k_preconditioner,
            ),
            k_scale,
        )
        omega_amg = _reparametrized_preconditioner(
            coupled.turbulence.omega_preconditioner(
                mdot,
                closure,
                omega_ref,
                method=method,
                reuse=None if reuse is None else reuse.omega_preconditioner,
            ),
            omega_scale,
        )

    if reuse is not None:
        # Carry the shift diagonals from the reused policy -- rebuilding them at the developed state
        # over-damps the step and freezes the march (see the docstring). Only the AMGs are refreshed.
        k_shift = reuse.k_shift_diagonal
        omega_shift = reuse.omega_shift_diagonal
    else:
        k_shift = coupled.turbulence.k_shift_policy(mdot, closure, k_ref).shift_diagonal * k_scale
        omega_shift = (
            coupled.turbulence.omega_shift_policy(mdot, closure, omega_ref).shift_diagonal
            * omega_scale
        )

    return CoupledShiftPolicy(coupled.layout, block, k_shift, omega_shift, k_amg, omega_amg)


def _is_traced(pytree: object) -> bool:
    """Whether any array leaf of ``pytree`` is a JAX tracer (i.e. we are inside a JAX transform).

    ``solve_coupled`` orchestrates the march eagerly (the scalar-block AMG hierarchies are assembled
    off the jit path as ``scipy.sparse`` matrices, so the whole solve cannot be traced), so a tracer
    leaf means the caller has wrapped the solve in ``jax.grad`` / ``jvp`` / ``vmap``. Used to reject the
    forward-only preconditioner refresh under differentiation with a clear error.

    Parameters
    ----------
    pytree : object
        Any pytree (here the ``(coupled, flow, k, omega)`` inputs), possibly containing ``None`` leaves.

    Returns
    -------
    bool
        ``True`` if at least one leaf is a :class:`jax.core.Tracer`.
    """
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(pytree))


def solve_coupled(
    coupled: CoupledRANS,
    flow: jnp.ndarray | None = None,
    k: jnp.ndarray | None = None,
    omega: jnp.ndarray | None = None,
    *,
    continuation: PseudoTransientStep | None = None,
    reference_state: jnp.ndarray | None = None,
    method: str | None = "twolevel",
    max_steps: int = 60,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    refresh_trigger: RefreshTrigger | None = None,
    refresh_limit: int = 1,
    on_step: Callable[[StepReport], None] | None = None,
    **continuation_kwargs: object,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the coupled RANS system ``R(u, p, k, omega) = 0`` by one monolithic Newton solve.

    A single :class:`~aquaflux.solve.ImplicitNewtonSolver` on :meth:`CoupledRANS.residual`, globalized
    by the pseudo-transient :func:`coupled_continuation` step -- the coupled counterpart of the flow
    block's :func:`~aquaflux.flow.reused_flow_solve`. Reverse-differentiable through the converged state
    by the coupled implicit-function-theorem adjoint (a single transpose solve on the unfrozen
    ``R_coupled``), the exact sensitivity the design note (S5) prescribes.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler; **the differentiable parameter pytree** for the adjoint.
    flow, k, omega : jnp.ndarray or None
        The initial flow state ``((dim + 1) n_cells,)`` and turbulence fields ``(n_cells,)``. **Leave
        any of them ``None`` to self-start from a hybrid initial condition**
        (:func:`~aquaflux.turbulence.hybrid_initialize` -- potential-flow velocity + Laplace-smoothed
        turbulence), so ``solve_coupled(coupled)`` converges from nothing; the monolithic Newton stalls
        from a raw cold start otherwise. The initial state also seeds the frozen preconditioner unless
        ``reference_state`` is given. (When differentiating, pass an explicit state built outside
        ``jax.grad``.)
    continuation : PseudoTransientStep or None
        A pre-built continuation step. **Build it once outside ``jax.grad`` and pass it here when
        differentiating** (the block preconditioner must be constructed with concrete parameters, not
        traced -- see the flow preconditioner note); ``None`` builds it internally from the initial
        state, which is the convenient forward-only path.
    reference_state : jnp.ndarray or None
        The coupled state to freeze the internally-built preconditioner at; defaults to the initial
        state. Ignored when ``continuation`` is supplied.
    method : {"twolevel", "air"} or None
        The scalar-block AMG method for the internally-built continuation.
    max_steps : int
        Newton iteration cap for the continuation march.
    rtol, atol : float
        Nonlinear stopping tolerances on the coupled residual norm.
    refresh_trigger : RefreshTrigger, optional
        Re-freeze the preconditioner part way through the march, on the evidence of the march's own
        per-step cost. The solve is run as a sequence of observed segments
        (:func:`~aquaflux.solve.forward_march`): each steps until the trigger fires, at which point
        the k/omega AMGs are re-derived at the state reached and the next segment continues from
        there. ``None`` (default) is the single-stage march.

        Use :class:`~aquaflux.solve.CycleGrowthTrigger`, which watches the linear solve's
        restart-cycle count. **The cycle count, not the residual, is the staleness signal:** a frozen
        preconditioner drifting from the operator shows up as a rising cost on a system that is
        otherwise unchanged, before the residual history shows anything, and unlike wall-clock time
        the count is unaffected by machine load. (That trigger still *gates* on the residual having
        fallen, because the damping schedule also raises the cost as it ramps down -- see its own
        documentation.)
    refresh_limit : int
        The most refreshes one solve may perform (default ``1``). Each costs a preconditioner rebuild
        and a recompilation of the shifted solve, so this bounds that expense independently of how
        eager the trigger is; ``0`` disables refreshing entirely. Note the observed march and the
        finishing solve compile the step separately, so a refreshed solve pays one compilation more
        than an unrefreshed one over and above the per-refresh rebuild -- the price of leaving the
        convergence guard solely with the solve that produces the result.
    on_step : callable, optional
        Called with each :class:`~aquaflux.solve.StepReport` as the march produces it -- the seam for
        logging a long solve's progress and cost. The refresh trigger reads the same reports.

        **Why:** the frozen scalar preconditioners go stale as the flow separates. On a separated
        backward-facing-step state, re-freezing them cut the shifted solve from 30 to 13 outer Krylov
        cycles (~2.4x); the flow block does *not* go stale and is carried over untouched. The refresh
        costs one extra compilation of the shifted solve, which that saving repays within a step or two
        at mesh sizes where this matters. The win appears only once the flow separates -- refreshing at
        a pre-separation state buys nothing and can cost, which is why the trigger gates on the
        residual having fallen as well as on the cost having risen.

        **Forward-only accelerator -- not usable under ``jax.grad`` (raises).** The refresh re-derives
        the preconditioner from the *mid-march* state; when differentiating, that state is a tracer, so
        the refreshed preconditioner would capture it and escape the converged solve's ``custom_vjp`` as
        a leaked tracer (the same reason a preconditioner must be built from concrete parameters outside
        ``jax.grad``). Since a refresh also forbids an explicit ``continuation`` (it must rebuild), there
        is no concrete-preconditioner path through it, so a trigger set under differentiation raises
        rather than leaking. To obtain gradients, drop ``refresh_trigger`` and differentiate the
        single-stage solve with a ``continuation`` built on concrete parameters outside ``jax.grad`` --
        the adjoint is refresh-independent anyway (the preconditioner is ``stop_gradient``-ed and only
        accelerates the Krylov iteration, so both marches reach the same converged state and thus the
        same implicit-function-theorem adjoint).

        **Each segment restarts the damping ramp, and that is load-bearing (binding).** A refresh
        rebuilds the pseudo-transient *shift diagonals* as well as the preconditioner, and under
        :class:`LogScalars` those carry a factor ``d(phi)/d(w) = omega``. Re-derived at a developed
        state, where ``omega`` has grown, the shift diagonal ``d`` grows with it. Each march segment
        therefore measures its **own** reference residual from the state it is handed, so the
        switched-evolution-relaxation ramp restarts at ``beta0`` and the larger ``d`` is paired with a
        correspondingly fresh ``beta``. Carrying the pre-refresh reference across instead -- to keep the
        ramp "continuous", which looks like the more principled choice -- pairs the grown ``d`` with the
        small ``beta`` appropriate to the *pre-refresh* residual. That is over-damping: the step
        collapses and the march **silently stops descending** (no error, no divergence, no guard trip --
        the relative residual simply creeps upward by ~1e-5 per step while the recirculation stays
        frozen). This is why the progress reference and the damping reference are separate quantities.

        ``rtol`` means the same thing with and without a refresh: the finishing solve is given the
        **absolute** target ``atol + rtol * ||R0||`` measured at the initial state, so a refreshed solve
        stops at exactly the residual an unrefreshed one would, for any number of refreshes. (This is
        available precisely because the refresh path is forward-only, so ``||R0||`` is a concrete
        number rather than a traced one.)

        ``max_steps`` applies to **each** segment, so a refreshed solve may take up to
        ``(refresh_limit + 1) * max_steps`` march steps plus the finishing solve's own allowance. The
        budget is deliberately not split: either segment may legitimately need
        the full allowance, and halving it would fail a march that a single-stage solve completes.
    **continuation_kwargs
        Forwarded to :func:`coupled_continuation` when building internally (schedule + preconditioner
        options).

    Returns
    -------
    tuple of jnp.ndarray
        The converged ``(flow, k, omega)``.
    """
    refreshing = refresh_trigger is not None and refresh_limit > 0
    if refreshing and _is_traced((coupled, flow, k, omega)):
        # The refresh re-derives the preconditioner from the mid-march state, which is a tracer when
        # differentiating; the refreshed preconditioner would capture it and escape the converged
        # solve's custom_vjp as a leaked tracer. There is no concrete-preconditioner path through a
        # refresh (it forbids an explicit `continuation`), so this cannot be worked around here --
        # raise with the fix rather than letting the leak surface as an opaque UnexpectedTracerError.
        raise ValueError(
            "refresh_trigger is a forward-only accelerator and cannot be used under jax.grad (or any "
            "JAX transform): the mid-march preconditioner rebuild would capture the differentiation "
            "tracer. Drop refresh_trigger and differentiate the single-stage solve with a `continuation` "
            "built on concrete parameters outside jax.grad -- the adjoint is refresh-independent, so "
            "the gradient is identical."
        )
    if flow is None or k is None or omega is None:
        flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
    # `flow, k, omega` are the physical initial condition; map into the solved-variable space (the
    # identity for DirectScalars, log for LogScalars) so the Newton march iterates on the right unknown.
    state = coupled.state_from_physical(flow, k, omega)
    if continuation is None:
        reference = state if reference_state is None else reference_state
        continuation = coupled_continuation(
            coupled, reference, method=method, **continuation_kwargs
        )
    elif refreshing:
        raise ValueError(
            "refresh_trigger needs solve_coupled to build the continuation (it re-freezes the "
            "preconditioner part way through), but an explicit `continuation` was supplied. Pass the "
            "schedule via **continuation_kwargs instead, or stage the refresh yourself with "
            "coupled_continuation(..., reuse=<the old policy>)."
        )

    stage_rtol, stage_atol = rtol, atol
    if refreshing:
        # Observed pre-march: step until the trigger judges the frozen preconditioner stale, re-freeze,
        # and continue from there. Each segment is an accelerator only -- it may stop short of a root,
        # and carries no convergence guard -- so the finishing solve below still produces the result.
        # `coupled.residual` is passed as a bound method (a pytree), not a lambda, so its arrays ride
        # as dynamic leaves and every step within a segment is a compilation-cache hit.
        reference_norm = float(continuation.norm()(coupled.residual(state)))
        # `refresh_limit` refreshes means `refresh_limit + 1` segments: the segment *after* the last
        # refresh must still be marched here, or the newly-refreshed preconditioner would only ever be
        # used by the finishing solve and its steps would go unobserved.
        for segment in range(refresh_limit + 1):
            result = forward_march(
                continuation,
                coupled.residual,
                state,
                max_steps=max_steps,
                rtol=rtol,
                atol=atol,
                reference_norm=reference_norm,
                trigger=refresh_trigger,
                observer=on_step,
            )
            state = result.state
            if not result.triggered or segment == refresh_limit:
                break
            # Re-freeze at the developed state: the k/omega AMGs are re-derived on their reused
            # coarsening, the flow block and the pseudo-time shift diagonals carried over (measured --
            # see `_coupled_shift_policy`). Rebuilding the shift diagonals instead freezes the march.
            continuation = coupled_continuation(
                coupled,
                jax.lax.stop_gradient(state),
                method=method,
                reuse=continuation.shift_policy,
                **continuation_kwargs,
            )
        # Hand the finishing solve the *absolute* target measured at the initial state, so a refreshed
        # solve stops exactly where an unrefreshed one would. A relative tolerance would be measured
        # against whatever residual the pre-march reached, silently tightening the solve by that factor
        # (and compounding with every extra refresh). This is only possible because the refresh path is
        # forward-only, so `reference_norm` is a concrete number rather than a traced one.
        stage_rtol, stage_atol = 0.0, atol + rtol * reference_norm

    solver = ImplicitNewtonSolver(
        max_steps=max_steps, rtol=stage_rtol, atol=stage_atol, forward_step=continuation
    )
    solved = solver.solve(lambda s, c: c.residual(s), state, coupled)
    return coupled.physical_fields(solved)


class _MassFlowBorderedPolicy(eqx.Module):
    """A coupled shift policy bordered with the mass-flow constraint (``beta`` appended to the state).

    Delegates to the inner :class:`CoupledShiftPolicy` on the coupled sub-state and borders both halves
    of the pseudo-transient step for the augmented ``[flow..., k, omega, beta]`` system: the shift
    diagonal gains a **zero** for ``beta`` (the linear constraint row needs no pseudo-time damping), and
    the block-diagonal preconditioner is wrapped by the constraint (Schur) preconditioner
    (:func:`~aquaflux.flow.mean_velocity._bordered_preconditioner`), which eliminates the scalar ``beta``
    with the border column/row ``(a, c)``. The shift only adds positive diagonal to the coupled block, so
    the border ``(a, c)`` -- the ``beta`` column and the ``<U>`` row, both shift-independent -- is reused
    unchanged.

    Attributes
    ----------
    inner : CoupledShiftPolicy
        The block-diagonal coupled policy for the ``[flow..., k, omega]`` sub-state.
    force, average : jnp.ndarray
        The border column ``a = dR_coupled/dbeta`` and row ``c = d<U_dir>/dstate`` in the coupled
        layout, shape ``((dim + 3) n_cells,)`` (:func:`_coupled_constraint_vectors`).
    """

    inner: CoupledShiftPolicy
    force: jnp.ndarray
    average: jnp.ndarray

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        """The augmented block-diagonal shift and the bordered preconditioner at ``phi``."""
        inner_term = self.inner.shift_term(phi[:-1])
        diagonal = jnp.append(inner_term.diagonal, 0.0)

        def make_preconditioner(relaxation: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
            coupled_m = inner_term.make_preconditioner(relaxation)
            return _bordered_preconditioner(lambda _w: coupled_m, self.force, self.average)(phi)

        return ShiftTerm(diagonal, make_preconditioner)

    def adjoint_factory(self) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """The ``state -> M`` factory for the adjoint transpose solve (the composition at ``beta = 0``)."""
        return lambda state: self.shift_term(state).make_preconditioner(jnp.asarray(0.0))


def _coupled_constraint_vectors(
    coupled: CoupledRANS, flow_direction: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The mass-flow border column/row ``(a, c)`` in the coupled ``[flow..., k, omega]`` layout.

    ``beta`` enters only the momentum block (as the body force), and ``<U>`` reads only the velocity, so
    both vectors are the flow-block border (:func:`~aquaflux.flow.mean_velocity._constraint_vectors`)
    packed with zero ``k`` / ``omega`` blocks.
    """
    force_flow, average_flow = _constraint_vectors(coupled.momentum, flow_direction)
    zero = jnp.zeros(coupled.momentum.mesh.n_cells)
    return (
        coupled.layout.pack(force_flow, zero, zero),
        coupled.layout.pack(average_flow, zero, zero),
    )


def mass_flow_coupled_continuation(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    *,
    flow_direction: int = 0,
    method: str | None = "twolevel",
    beta0: float = 2.0,
    exponent: float = 1.0,
    beta_floor: float = 0.0,
    max_escalations: int = 6,
    escalation_factor: float = 2.0,
    divergence_cap: float = 10.0,
    line_search: int = _COUPLED_LINE_SEARCH,
    forward_solver: lx.AbstractLinearSolver | None = None,
    block_scaled_norm: bool = False,
    **preconditioner_kwargs: object,
) -> PseudoTransientStep:
    """The pseudo-transient continuation step for the **mass-flow-constrained** coupled Newton solve.

    The globalization of :func:`coupled_continuation`, with its :class:`CoupledShiftPolicy` bordered by
    the mass-flow constraint (:class:`_MassFlowBorderedPolicy`), so it drives the augmented
    ``[flow..., k, omega, beta]`` system where ``beta`` is a Lagrange multiplier for ``<U_dir> =
    target``. Parameters are :func:`coupled_continuation`'s (including ``beta_floor`` / ``line_search`` /
    ``forward_solver`` / ``block_scaled_norm``); ``flow_direction`` selects the constrained velocity
    component. ``block_scaled_norm`` here extends the same block-scaled measure with the constraint dof.
    """
    # No `reuse` here: the mass-flow-constrained path has no staged-refresh driver (there is no
    # `refresh_trigger` on `solve_coupled_mass_flow`), so a policy is always built from scratch. Thread
    # `reuse` through if that driver is ever added -- the bordered policy wraps this one unchanged.
    policy = _coupled_shift_policy(coupled, reference_state, method, **preconditioner_kwargs)
    force, average = _coupled_constraint_vectors(coupled, flow_direction)
    bordered = _MassFlowBorderedPolicy(policy, force, average)
    residual_norm = (
        _mass_flow_residual_norm(coupled, reference_state) if block_scaled_norm else jnp.linalg.norm
    )
    return PseudoTransientStep(
        bordered,
        beta0=beta0,
        exponent=exponent,
        beta_floor=beta_floor,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        acceptance=DivergenceGuard(divergence_cap=divergence_cap),
        line_search=line_search,
        forward_solver=forward_solver if forward_solver is not None else _COUPLED_FORWARD_SOLVER,
        residual_norm=residual_norm,
        adjoint_preconditioner_factory=bordered.adjoint_factory(),
    )


def solve_coupled_mass_flow(
    coupled: CoupledRANS,
    target: float,
    *,
    flow_direction: int = 0,
    flow: jnp.ndarray | None = None,
    k: jnp.ndarray | None = None,
    omega: jnp.ndarray | None = None,
    continuation: PseudoTransientStep | None = None,
    reference_state: jnp.ndarray | None = None,
    method: str | None = "twolevel",
    max_steps: int = 60,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    **continuation_kwargs: object,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the coupled RANS system holding the bulk velocity at ``target``, in one monolithic Newton.

    The mass-flow analogue of :func:`solve_coupled`: the body force ``beta`` (along ``flow_direction``)
    is a **coupled unknown** appended to the state, and the coupled residual is bordered with the
    constraint row ``<U_dir> - target`` -- one honest augmented residual

        R_aug([flow, k, omega, beta]) = [ R_coupled(flow, k, omega; beta) ; <U_dir>(flow) - target ],

    driven by a single :class:`~aquaflux.solve.ImplicitNewtonSolver` globalized by
    :func:`mass_flow_coupled_continuation`. ``<U> = target`` therefore holds at the converged root **by
    construction**, and (the point of putting the constraint *in* the coupled residual) the coupled
    implicit-function-theorem adjoint carries it: ``jax.grad`` through the converged constrained solve is
    the exact sensitivity of the whole turbulent flow at fixed bulk velocity. The forward solve is
    monolithic here, but the same bordered residual is what a *segregated* forward loop would need its
    coupled adjoint to transpose (segregated forward, coupled adjoint).

    Parameters mirror :func:`solve_coupled` (``coupled`` is the differentiable parameter pytree; leave
    ``flow``/``k``/``omega`` ``None`` to self-start from the hybrid IC; build ``continuation`` outside
    ``jax.grad`` when differentiating), plus:

    target : float
        The bulk (volume-averaged) velocity component to hold along ``flow_direction``.
    flow_direction : int
        The streamwise axis the bulk velocity is measured and the body force applied along.

    Returns
    -------
    tuple of jnp.ndarray
        The converged ``(flow, k, omega, beta)`` -- the fields and the multiplier that hits ``target``.
    """
    if flow is None or k is None or omega is None:
        flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
    # Map the physical initial condition into the solved-variable space (identity for DirectScalars,
    # log for LogScalars) so the constrained Newton march iterates on the right scalar unknown.
    state = coupled.state_from_physical(flow, k, omega)
    augmented0 = jnp.append(state, coupled.momentum.body_force[flow_direction])

    if continuation is None:
        reference = state if reference_state is None else reference_state
        continuation = mass_flow_coupled_continuation(
            coupled, reference, flow_direction=flow_direction, method=method, **continuation_kwargs
        )
    solver = ImplicitNewtonSolver(
        max_steps=max_steps, rtol=rtol, atol=atol, forward_step=continuation
    )

    def constrained_residual(augmented: jnp.ndarray, theta: CoupledRANS) -> jnp.ndarray:
        # theta is the coupled assembler (the differentiable parameter); beta overrides its body force.
        coupled_state, beta = augmented[:-1], augmented[-1]
        forced_momentum = _with_body_force(theta.momentum, flow_direction, beta)
        forced = eqx.tree_at(lambda c: c.momentum, theta, forced_momentum)
        r_coupled = forced.residual(coupled_state)
        flow_state, _, _ = theta.layout.unpack(coupled_state)
        velocity, _ = theta.momentum.unpack(flow_state)
        volume = theta.momentum.geometry.cell.volume
        bulk = jnp.sum(velocity[:, flow_direction] * volume) / jnp.sum(volume)
        return jnp.append(r_coupled, bulk - target)

    solved = solver.solve(constrained_residual, augmented0, coupled)
    flow_s, k_s, omega_s = coupled.physical_fields(solved[:-1])
    return flow_s, k_s, omega_s, solved[-1]
