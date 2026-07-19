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

Positivity of ``k, omega`` under a full Newton step is carried by the pseudo-transient continuation
(:mod:`~aquaflux.turbulence.continuation` block policy): the shift damps the step heavily far from the
fixed point, and a step that drives ``k`` or ``omega`` non-positive makes the closure non-finite
through ``sqrt(k)`` -- rejected by the divergence guard, which escalates the damping. The realizability
floor stays **out** of this residual (design note S3.3): a converged RANS field is strictly positive,
so the floor is inactive there and the coupled adjoint sees only the smooth interior physics.
Log-variable transport (``k = e^k~``) is the design note's held-in-reserve structural fix if the
direct form proves non-robust at high Reynolds number.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aquaflux.flow import BlockPreconditioner
from aquaflux.properties import FieldProperty, PropertyModel
from aquaflux.solve import ImplicitNewtonSolver
from aquaflux.solve.continuation import DivergenceGuard, PseudoTransientStep, ShiftTerm

if TYPE_CHECKING:
    from aquaflux.flow import MomentumContinuity

    from .transport import SSTTurbulence

# A frozen ``phi -> M`` preconditioner factory (the scalar convection-diffusion AMG carriers).
_ScalarFactory = Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]


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


def _with_viscosity(
    momentum: MomentumContinuity, effective_viscosity: jnp.ndarray
) -> MomentumContinuity:
    """Return ``momentum`` with its ``viscosity`` property replaced by a per-cell field.

    The seam the eddy viscosity enters the momentum block through: ``mu_eff = rho (nu + nu_t)``. The
    same functional swap the segregated driver applies each sweep, here evaluated live inside the
    coupled residual so ``dR_momentum / d(k, omega)`` flows through ``nu_t`` under AD.
    """
    properties = PropertyModel(
        {**momentum.properties.properties, "viscosity": FieldProperty(effective_viscosity)}
    )
    return eqx.tree_at(lambda m: m.properties, momentum, properties)


class CoupledRANS(eqx.Module):
    """The monolithic ``R(u, p, k, omega)`` assembler.

    Holds the flow and turbulence assemblers and the (constant) density, and composes their residuals
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
    density : float
        The constant fluid density forming ``mu_eff = rho (nu + nu_t)``.
    """

    momentum: MomentumContinuity
    turbulence: SSTTurbulence
    density: float

    @classmethod
    def build(
        cls, momentum: MomentumContinuity, turbulence: SSTTurbulence, density: float
    ) -> CoupledRANS:
        """Assemble the coupled system, pre-resolving the turbulence boundaries off the jit path.

        The turbulence residual rebuilds its scalar :class:`~aquaflux.discretization.ResidualAssembler`
        each evaluation, and that build resolves the k/omega boundary patches -- a dynamic-shape
        ``nonzero`` lookup on the mesh labels that cannot run inside the coupled residual's jit. Binding
        those boundaries **once here** (the momentum boundary is already resolved by
        :meth:`~aquaflux.flow.MomentumContinuity.build`) makes the per-evaluation rebuild's ``resolve``
        an idempotent no-op, so the whole coupled residual is jit- and adjoint-safe.
        """
        face_patches = momentum.mesh.face_patches
        turbulence = eqx.tree_at(
            lambda t: (t.k_boundary, t.omega_boundary),
            turbulence,
            (
                turbulence.k_boundary.resolve(face_patches),
                turbulence.omega_boundary.resolve(face_patches),
            ),
        )
        return cls(momentum, turbulence, density)

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
        """
        flow, k, omega = self.layout.unpack(state)

        grad_velocity = self.momentum.velocity_gradient(flow)
        nu_t = self.turbulence.eddy_viscosity(grad_velocity, k, omega)
        momentum = _with_viscosity(
            self.momentum, self.density * (self.turbulence.molecular_viscosity + nu_t)
        )

        flow_residual = momentum.residual(flow)

        mdot = momentum.mass_flux(flow)
        closure = self.turbulence.closure_fields(grad_velocity, k, omega)
        k_residual = self.turbulence.k_residual(mdot, closure)(k)
        omega_residual = self.turbulence.omega_residual(mdot, closure)(omega)

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
    k_preconditioner: _ScalarFactory | None = None
    omega_preconditioner: _ScalarFactory | None = None

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


def coupled_continuation(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    *,
    method: str | None = "twolevel",
    beta0: float = 2.0,
    exponent: float = 1.0,
    max_escalations: int = 6,
    escalation_factor: float = 2.0,
    divergence_cap: float = 10.0,
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
    beta0, exponent, max_escalations, escalation_factor, divergence_cap
        The pseudo-transient schedule and divergence-guard parameters (see
        :class:`~aquaflux.solve.PseudoTransientStep`).
    **preconditioner_kwargs
        Forwarded to :meth:`~aquaflux.flow.BlockPreconditioner.build` for the flow block (e.g.
        ``schur_scaling``, ``velocity``).

    Returns
    -------
    PseudoTransientStep
        The forward step to hand :class:`~aquaflux.solve.ImplicitNewtonSolver` as ``forward_step``.
    """
    flow_ref, k_ref, omega_ref = coupled.layout.unpack(reference_state)
    grad_velocity = coupled.momentum.velocity_gradient(flow_ref)
    nu_t = coupled.turbulence.eddy_viscosity(grad_velocity, k_ref, omega_ref)
    momentum = _with_viscosity(
        coupled.momentum, coupled.density * (coupled.turbulence.molecular_viscosity + nu_t)
    )
    block = BlockPreconditioner.build(momentum, **preconditioner_kwargs)

    mdot = momentum.mass_flux(flow_ref)
    closure = coupled.turbulence.closure_fields(grad_velocity, k_ref, omega_ref)
    k_policy = coupled.turbulence.k_shift_policy(mdot, closure, k_ref, method=method)
    omega_policy = coupled.turbulence.omega_shift_policy(mdot, closure, omega_ref, method=method)

    policy = CoupledShiftPolicy(
        coupled.layout,
        block,
        k_policy.shift_diagonal,
        omega_policy.shift_diagonal,
        k_policy.preconditioner,
        omega_policy.preconditioner,
    )
    return PseudoTransientStep(
        policy,
        beta0=beta0,
        exponent=exponent,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        acceptance=DivergenceGuard(divergence_cap=divergence_cap),
        adjoint_preconditioner_factory=policy.adjoint_factory(),
    )


def solve_coupled(
    coupled: CoupledRANS,
    flow: jnp.ndarray,
    k: jnp.ndarray,
    omega: jnp.ndarray,
    *,
    continuation: PseudoTransientStep | None = None,
    reference_state: jnp.ndarray | None = None,
    method: str | None = "twolevel",
    max_steps: int = 60,
    rtol: float = 1e-10,
    atol: float = 1e-12,
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
    flow, k, omega : jnp.ndarray
        The initial flow state ``((dim + 1) n_cells,)`` and turbulence fields ``(n_cells,)`` -- a
        representative (e.g. segregated pre-smoothed) start, which the frozen preconditioner also uses
        as its reference unless ``reference_state`` is given.
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
    **continuation_kwargs
        Forwarded to :func:`coupled_continuation` when building internally (schedule + preconditioner
        options).

    Returns
    -------
    tuple of jnp.ndarray
        The converged ``(flow, k, omega)``.
    """
    state = coupled.pack_state(flow, k, omega)
    if continuation is None:
        reference = state if reference_state is None else reference_state
        continuation = coupled_continuation(
            coupled, reference, method=method, **continuation_kwargs
        )
    solver = ImplicitNewtonSolver(
        max_steps=max_steps, rtol=rtol, atol=atol, forward_step=continuation
    )
    solved = solver.solve(lambda s, c: c.residual(s), state, coupled)
    return coupled.layout.unpack(solved)
