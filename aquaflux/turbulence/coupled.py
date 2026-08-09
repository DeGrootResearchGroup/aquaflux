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
import dataclasses
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from aquaflux.discretization import DifferenceRow, FixationRow, LogRatioRow
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
    DualTimeControl,
    DualTimeStep,
    FieldGroups,
    FieldSplitAmgPreconditioner,
    ForwardStep,
    ImplicitNewtonSolver,
    LocalCourantBasis,
    MonolithicAmgPreconditioner,
    MonolithicIlutPreconditioner,
    MonolithicLuPreconditioner,
    PseudoTransientStep,
    RefreshTiming,
    RefreshTrigger,
    ResidualNorm,
    RowScaledNorm,
    ShiftBasis,
    ShiftTerm,
    StepControl,
    StepReport,
    SwitchedEvolutionRelaxation,
    TransposedPreconditioner,
    VelocityShiftParts,
    block_stencil_colouring,
    block_stencil_gather_map,
    forward_march,
    positive_block_limit,
    relative_residual_gmres,
)

from .initialization import hybrid_initialize
from .preconditioner import ScalarTransportPreconditioner, ScaledScalarPreconditioner

# The default pseudo-time shift basis (full operator diagonal = uniform under-relaxation), held as a
# module singleton so it is not reconstructed in each function's argument defaults.
_DEFAULT_SHIFT_BASIS = LocalCourantBasis()

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

    @abc.abstractmethod
    def fixation_row(self) -> FixationRow:
        """How an algebraic value fixation on this field should be written as a residual row.

        A value fixation must be expressed in the **solved** unknown, not the physical field, or its
        linearization inherits the transform's nonlinearity: under ``phi = e**w`` the plain difference
        row ``phi - target`` gives a Newton correction ``dw = target/phi - 1``, which overshoots by
        ``e**(r-1)`` against a target ratio ``r``, while the log-ratio row is linear in ``w`` and lands
        on the constraint in one full step. The transform owns this choice because it is the only
        object that knows which variable is actually being solved for.
        """


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

    def fixation_row(self) -> FixationRow:
        """The plain difference -- the solved unknown *is* the physical field here."""
        return DifferenceRow()


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

    def fixation_row(self) -> FixationRow:
        """The log ratio ``log(phi/target) = w - log(target)`` -- linear in the solved unknown ``w``.

        The difference row would be exponential in ``w`` here, so a near-wall cell whose ``omega`` is a
        factor ``r`` from its target takes a correction ``dw = r - 1`` and overshoots to ``phi e**(r-1)``
        instead of landing on ``phi r``. It also writes the row on the scale of ``phi`` (which spans
        orders of magnitude near a wall) rather than of ``w``, which lets a handful of fixation cells
        dominate the residual measure the whole march is judged by.
        """
        return LogRatioRow()


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

    def eddy_viscosity(self, state: jnp.ndarray) -> jnp.ndarray:
        """The eddy viscosity ``nu_t`` at a coupled state, shape ``(n_cells,)``.

        The coefficient the frozen scalar-transport preconditioners are built from, so it is also
        what a staleness measure watches (:func:`eddy_viscosity_drift`).

        Parameters
        ----------
        state : jnp.ndarray
            The flat coupled state, shape ``((dim + 3) n_cells,)``.
        """
        flow, k, omega = self.physical_fields(state)
        return self.turbulence.closure_fields(self.momentum.velocity_fields(flow), k, omega).nu_t

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

    def with_scaled_molecular_viscosity(self, factor: float) -> CoupledRANS:
        """Return a copy whose molecular viscosity is multiplied by ``factor`` in **both** blocks.

        The molecular viscosity lives in two places that must move together: the momentum block's
        dynamic ``mu`` (in its property model) and the turbulence block's kinematic ``nu`` (its
        per-cell field). Both are scaled by the same ``factor`` -- consistent because ``mu = rho nu``
        and the density is unchanged -- so the result is a self-consistent lower-Reynolds-number
        version of the same case (``Re`` scaled by ``1 / factor``). This is the single place that
        knows where the molecular viscosity is stored; a Reynolds-number homotopy builds each
        companion problem through it rather than restating the case.

        Because it rescales the *molecular* viscosity only, the closure's eddy viscosity and every
        other model quantity are unchanged, and the transform / boundaries carry over untouched.

        Parameters
        ----------
        factor : float
            The multiplier applied to the molecular viscosity of both blocks; ``> 1`` lowers the
            Reynolds number. A tracer flows through it under differentiation.

        Returns
        -------
        CoupledRANS
            The same coupled system at the scaled molecular viscosity; ``self`` is unchanged.
        """
        return eqx.tree_at(
            lambda c: (c.momentum, c.turbulence),
            self,
            (
                self.momentum.with_scaled_molecular_viscosity(factor),
                self.turbulence.with_scaled_molecular_viscosity(factor),
            ),
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
        written in the physical ``k`` / ``omega`` recovered by :attr:`k_transform` / :attr:`omega_transform`
        (the identity for
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
        # The near-wall omega fixation is written in the *solved* unknown, so under a log
        # parametrization it is linear in that unknown instead of exponential in it.
        omega_residual = self.turbulence.omega_residual(
            fields.mdot, closure, self.omega_transform.fixation_row()
        )(omega)

        return self.layout.pack(flow_residual, k_residual, omega_residual)


class LiveViscosityVelocityParts(eqx.Module):
    """The momentum diagonal at the **current** effective viscosity — a genuine local time scale.

    Re-forms the closure at the state being stepped, so the velocity buckets carry the eddy viscosity as
    it develops rather than as it was at the freeze state. Costs one closure evaluation per *step* (not
    per residual evaluation), which is a few milliseconds against a shifted solve of tens of seconds.

    Use this when the shift is meant to be a local time step that tracks the flow — in particular with a
    convective :class:`~aquaflux.solve.ShiftBasis`, where a frozen viscosity means the "local Courant
    number" is computed from the wrong operator.

    Attributes
    ----------
    momentum : MomentumContinuity
        The flow assembler at molecular viscosity; the eddy viscosity is applied per call.
    turbulence : SSTTurbulence
        The closure, used to form ``nu_t`` and the wall-face eddy viscosity at the current fields.
    k_transform, omega_transform : ScalarVariableTransform
        The parametrizations of the two scalar blocks, so the solved unknowns handed in can be mapped
        back to the physical fields the closure needs (the identity for a directly-solved scalar).
    """

    momentum: MomentumContinuity
    turbulence: SSTTurbulence
    k_transform: ScalarVariableTransform
    omega_transform: ScalarVariableTransform

    def parts(
        self, flow: jnp.ndarray, k_solved: jnp.ndarray, omega_solved: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        k = self.k_transform.to_physical(k_solved)
        omega = self.omega_transform.to_physical(omega_solved)
        closure = self.turbulence.closure_fields(self.momentum.velocity_fields(flow), k, omega)
        live = self.momentum.with_eddy_viscosity(
            closure.nu_t, self.turbulence.wall_face_eddy_viscosity(k)
        )
        velocity, _pressure = live.unpack(flow)
        return live.momentum_matrix_diagonal_parts(velocity)


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
    k_shift_transport, omega_shift_transport : jnp.ndarray
        The per-cell transport-operator shift diagonals for k and omega, shape ``(n_cells,)`` (the
        omega one has its near-wall fixed cells zeroed). This is the **local time scale** — the
        physics half of the shift — so a refresh *rebuilds* it at the developed state.
    k_jacobian_scale, omega_jacobian_scale : jnp.ndarray
        The per-cell ``d(phi)/d(w)`` coordinate factor for k and omega, shape ``(n_cells,)`` (``omega``
        under :class:`LogScalars`, ``1`` for the identity transform). This is the **coordinate
        transformation** between the physical field and the solved variable — not physics — so a refresh
        *carries* it frozen. Storing the two factors separately (rather than their product) is what lets
        a refresh update the transport time scale while holding the coordinate factor, so the temporal
        ratio ``transport(state)/transport(reference)`` has the field's range cancel and the shift does
        not inherit ``omega``'s growth. :meth:`shift_term` multiplies them.
    k_preconditioner, omega_preconditioner : callable or None
        The frozen ``phi -> M`` convection-diffusion AMG factories for the k and omega blocks, or
        ``None`` for an unpreconditioned (identity) scalar block.
    velocity_shift_parts : VelocityShiftParts or None
        Where the velocity shift's two diagonal buckets come from. ``None`` (default) takes them from
        the frozen flow preconditioner -- live in velocity, frozen in viscosity -- which is the
        historical behaviour. Pass :class:`LiveViscosityVelocityParts` to form them at the current
        effective viscosity instead. **Carried across a refresh**, since it is a configuration choice
        rather than frozen state.
    shift_basis : ShiftBasis
        How the live velocity shift diagonal is built from the momentum diagonal's convective/dissipative
        parts (the scalar shift diagonals are pre-combined at build time). The default
        :class:`~aquaflux.solve.LocalCourantBasis` (weight ``1``) is ``a_P`` -- uniform under-relaxation,
        unchanged from the historical shift; a convective basis gives a local convective time step.
    """

    layout: CoupledRANSLayout
    flow_preconditioner: BlockPreconditioner
    k_shift_transport: jnp.ndarray
    k_jacobian_scale: jnp.ndarray
    omega_shift_transport: jnp.ndarray
    omega_jacobian_scale: jnp.ndarray
    k_preconditioner: ScalarTransportPreconditioner | None = None
    omega_preconditioner: ScalarTransportPreconditioner | None = None
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS
    velocity_shift_parts: VelocityShiftParts | None = None

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
        # `a_p` is a PRECONDITIONER quantity: it is what the velocity block inverts, so it must come
        # from the block itself or the two disagree.
        convective, dissipative = block.frozen_momentum_diagonal_parts(flow)
        a_p = convective + dissipative
        # The SHIFT's buckets are a separate concern with a different lifetime (see
        # `VelocityShiftParts`), so their source is injected; `None` reuses the preconditioner's, which
        # is the historical behaviour and keeps the default path bit-identical.
        if self.velocity_shift_parts is not None:
            convective, dissipative = self.velocity_shift_parts.parts(flow, k, omega)
        d_vel = self.shift_basis.local_diagonal(convective, dissipative)

        # Full-state base shift: d_vel on every velocity component, 0 on pressure, the frozen scalar
        # transport diagonals on k and omega.
        flow_diagonal = assembler.pack(
            jnp.broadcast_to(d_vel[:, None], (n_cells, self.layout.dim)), jnp.zeros(n_cells)
        )
        # The scalar shift diagonal is transport-time-scale * coordinate factor; kept as two fields so a
        # refresh rebuilds the transport half and carries the coordinate half (see `_coupled_shift_policy`).
        diagonal = self.layout.pack(
            flow_diagonal,
            jax.lax.stop_gradient(self.k_shift_transport * self.k_jacobian_scale),
            jax.lax.stop_gradient(self.omega_shift_transport * self.omega_jacobian_scale),
        )

        def make_preconditioner(relaxation: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
            # Flow block at the shifted diagonal a_P + beta*d_vel matching the shifted Jacobian; scalar
            # blocks at their frozen AMG (beta-independent -- the shift only adds positive diagonal).
            flow_m = block.apply_at(flow, jax.lax.stop_gradient(a_p + relaxation * d_vel))
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


def eddy_viscosity_drift(
    coupled: CoupledRANS, reference_state: jnp.ndarray
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """A staleness measure: how far ``nu_t`` has moved from ``reference_state``, relatively.

    ``||nu_t(state) - nu_t(reference)|| / ||nu_t(reference)||``, the drift signal a
    :class:`~aquaflux.solve.CoefficientDriftTrigger` fires on. ``nu_t`` is the right coefficient to
    watch because it is what the frozen k/omega transport operators are assembled from: when it has
    moved, those operators no longer describe the system being solved, which is precisely staleness.

    **Why drift rather than the linear solve's cost.** The restart-cycle count also rises with
    staleness, but it rises with the pseudo-transient damping ``beta`` as well -- by more, on a
    separating flow -- so a cost-based trigger must be gated to suppress that confound. Drift
    responds only to the coefficients, so it measures staleness directly and needs no gate.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled assembler, used to evaluate ``nu_t`` at a state.
    reference_state : jnp.ndarray
        The state the current preconditioner was frozen at, shape ``((dim + 3) n_cells,)``. **Re-base
        this at every refresh**, or the measure keeps reporting movement the refresh already absorbed.

    Returns
    -------
    callable
        ``state -> drift``, a non-negative scalar that is zero at ``reference_state``. Compiled, so
        the per-step cost is one jitted closure evaluation.

    Notes
    -----
    The denominator is floored at a tiny positive value, so a state with no turbulence anywhere
    (``nu_t`` identically zero) yields a finite drift rather than a division by zero. Any real initial
    condition carries some eddy viscosity, so the floor is a guard, not a regime.
    """
    reference = jax.lax.stop_gradient(coupled.eddy_viscosity(reference_state))
    scale = jnp.maximum(jnp.linalg.norm(reference), jnp.finfo(reference.dtype).tiny)
    # The reference rides as an ARGUMENT to a module-level compiled function, not as a captured
    # constant of a locally-defined one. `filter_jit` caches per function object, so a closure built
    # here would be a fresh cache entry every time -- and this measure is deliberately re-based at
    # every materialize, which made each re-base recompile `eddy_viscosity` from scratch. Measured on a
    # three-dimensional coupled march that was ~3.8 s on every full refresh, ~21 % of it, for a value
    # change. Same reason the eager march passes its step and residual to a module-level jitted step.
    return lambda state: _eddy_viscosity_drift(coupled, state, reference, scale)


@eqx.filter_jit
def _jacobian_matvec(coupled: CoupledRANS, state: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """``J(state) @ tangent`` -- the matrix-free coupled Jacobian-vector product, compiled once.

    Everything it needs is an **argument**, including the assembler. A locally-defined ``jax.jit``
    closure over ``coupled`` is a fresh cache entry per closure, so each Reynolds-continuation rung --
    which rebuilds the assembler at its own viscosity -- would recompile the probe from scratch, even
    though a scaled viscosity changes only two leaf *values* and leaves the pytree structure identical.
    As an argument the assembler's arrays are ordinary traced leaves and every rung is a cache hit.
    """
    return jax.jvp(coupled.residual, (state,), (tangent,))[1]


@eqx.filter_jit
def _batched_jacobian_matvec(
    coupled: CoupledRANS, state: jnp.ndarray, tangents: jnp.ndarray
) -> jnp.ndarray:
    """``J(state) @ tangents`` for a stack of tangents -- the batched form the coloured probe uses.

    The same directional derivative as :func:`_jacobian_matvec` applied to each row, so the responses
    are bit-identical to a per-tangent loop; running them as a few fused passes only amortizes dispatch.
    Takes the assembler as an argument for the same reason.
    """
    return jax.vmap(lambda tangent: jax.jvp(coupled.residual, (state,), (tangent,))[1])(tangents)


@eqx.filter_jit
def _eddy_viscosity_drift(
    coupled: CoupledRANS,
    state: jnp.ndarray,
    reference: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """``||nu_t(state) - reference|| / scale`` -- the compiled core of :func:`eddy_viscosity_drift`.

    Module-level and taking everything it needs as arguments, so **one** compilation serves every
    re-based reference: ``coupled`` is an ``equinox.Module`` whose arrays are traced leaves, and
    ``reference``/``scale`` are arrays of fixed shape, so a re-base is a cache hit.
    """
    return jnp.linalg.norm(coupled.eddy_viscosity(state) - reference) / scale


def _row_jacobian_scale(
    transform: ScalarVariableTransform,
    reference: jnp.ndarray,
    fixed_cells: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """``d(row)/d(solved unknown)`` for every row of a scalar block, transport and fixation alike.

    Transport rows are assembled in the physical field, so each carries the chain factor
    ``d(phi)/d(w)``. A **value-fixation** row is not: it is written by the transform's own
    :class:`~aquaflux.discretization.FixationRow`, which for a log-solved field is already expressed
    in ``w`` and so has derivative one, not ``phi``. Applying the chain factor to those rows too
    mis-scales them by ``phi`` -- which near a wall spans orders of magnitude -- so anything that
    rescales the block per row must ask each row for its own derivative.

    Returns the chain factor unchanged when there are no fixed cells, and (for the identity
    transform, where the chain factor is one and the difference row's derivative is one) an array of
    ones, so the directly-solved path is unaffected.
    """
    chain = transform.jacobian_scale(reference)
    if fixed_cells is None:
        return chain
    index = jnp.asarray(fixed_cells)
    return chain.at[index].set(
        transform.fixation_row().jacobian_scale(reference[index], chain[index])
    )


def _reparametrized_preconditioner(
    preconditioner: ScalarTransportPreconditioner | None, jacobian_scale: jnp.ndarray
) -> ScalarTransportPreconditioner | None:
    """Rescale a frozen physical-operator scalar preconditioner for the reparametrized block.

    The reparametrized Jacobian's inverse carries a leading ``diag(1 / jacobian_scale)``, so the
    physical-operator preconditioner is wrapped to apply it (:class:`ScaledScalarPreconditioner`). For
    the identity transform ``jacobian_scale`` is one, so the preconditioner is returned unchanged and
    the direct path stays bit-identical. The scale is materialized off the jit path (the reference
    state is concrete), matching the frozen hierarchy it wraps.

    ``jacobian_scale`` must be the **per-row** derivative from :func:`_row_jacobian_scale`, not the
    transform's chain factor alone: the frozen operator carries an identity row at each fixed cell,
    so scaling those rows by ``d(phi)/d(w)`` when the fixation row's own derivative is one leaves the
    preconditioned operator with a cluster of ``1/phi`` eigenvalues that stalls the Krylov solve.
    """
    if preconditioner is None:
        return None
    scale = np.asarray(jacobian_scale)
    if np.allclose(scale, 1.0):
        return preconditioner
    return ScaledScalarPreconditioner(preconditioner, 1.0 / scale)


# The shifted forward solve for the coupled march. Two measured choices on the ~12k-cell
# backward-facing step:
#  * Restart 120 (vs the shared default 40): the coupled turbulent saddle is stiff enough that a
#    40-vector restart discards too much Arnoldi history and needs hundreds of restart cycles, while a
#    120-vector subspace reaches the same solution in far fewer.
#  * A GLOBAL 2-norm relative-residual stop at ~1% (`relative_residual_gmres`), rather than lineax's
#    stock componentwise `rtol`/`atol` test. Each pseudo-transient step is an inexact Newton step, so
#    the linear solve only has to resolve the correction to the accuracy the globalized march actually
#    uses -- ~1% is ample here, and the converged root and its adjoint are fixed by the nonlinear stop
#    and the vanishing shift, not by the linear tolerance. The stock test does *not* deliver a genuine
#    relative stop on this system: it applies the tolerance per row under a max-norm, and the near-wall
#    omega rows start satisfied -- their right-hand side is ~0 -- so their per-row scale collapses onto
#    the absolute `atol` floor and a handful of them hold the whole solve to ~1e-10, about nine orders
#    past the relative tolerance nominally requested. Stopping on the global 2-norm relative residual is
#    immune to those rows and cuts the solve from ~15 restart cycles to ~3-5 (~3-4x fewer matrix-vector
#    products) with the march trajectory -- the reattachment length reached per step -- unchanged.
_COUPLED_FORWARD_SOLVER = relative_residual_gmres(
    1e-2, restart=120, stagnation_iters=40, max_restarts=15
)

# The monolithic ILUT preconditions the whole coupled saddle with an incomplete factorization that
# forms the true Schur coupling through its fill, so the preconditioned operator's spectrum is tightly
# clustered and the Krylov solve reaches the 1% stop within a handful of vectors -- the large
# 120-vector subspace the block-triangular preconditioner needs is pure waste here. A restarted GMRES
# only tests convergence at each restart boundary, so with `restart = 120` it builds ~120 matrix-vector
# products (each paying the ILUT's triangular back-solve) before it can stop, where ~5--10 already
# solve it. A small restart lets it stop as soon as it has converged: measured on the backward-facing
# step the march trajectory (the row-scaled residual reached at every step) is unchanged from
# `restart = 120`, while the per-step wall drops about ten-fold. `max_restarts` is kept generous so a
# transiently harder (e.g. drifted-reference) solve still completes before the cycle-count refresh
# trigger re-freezes the factorization.
_COUPLED_ILUT_FORWARD_SOLVER = relative_residual_gmres(
    1e-2, restart=10, stagnation_iters=40, max_restarts=40
)


# How many coloured tangents share one vmapped jvp pass when materializing the AMG Jacobian. Smaller uses
# less peak memory (fewer simultaneous forward-AD tapes), larger amortizes dispatch over more probes; the
# coloured probes run in ceil(n_probes / this) fused passes instead of an n_probes-call Python loop.
# Measured on the 3D backward-facing step (~670 reach-3 probes): forward-AD does not vectorize across the
# batch on CPU, so a large batch buys almost nothing in time (16 vs 4: ~33 s vs ~36 s) while costing it in
# peak memory (16 holds ~2.2 GB of simultaneous tapes vs ~0.7 GB at 4). Four keeps essentially all of the
# dispatch amortization at a third of the transient peak -- the right default for a memory-bounded solve.
_PROBE_BATCH_SIZE = 4

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
    is accepted. This is the coarser of the two field-aware measures -- one scale per block -- and is
    the opt-in ``block_scaled_norm=True`` alternative to the default :func:`coupled_scaled_norm`, which
    additionally equilibrates each row by its own diagonal.
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


def positive_k_limit(coupled: CoupledRANS, tau: float = 0.99):
    """The step limiter keeping ``k`` strictly positive, or ``None`` when the transform already does.

    ``k`` is solved DIRECTLY (``log k`` is singular at a no-slip wall, where ``k = 0`` is the physical
    boundary condition), so nothing structurally prevents a Newton step from carrying it negative --
    and the SST closure's ``sqrt(k)`` turns a single negative cell into NaN across the whole residual.
    Measured: a march ran 62 healthy steps and died when **two cells out of 23040** reached
    ``k = -3.3e-4``, with every field still finite and moving by ~1e-4.

    Returns ``None`` when ``k`` is solved in a form that is positive by construction (a log variable),
    since a cap there would only throttle a step for no benefit.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled case, for its block layout.
    tau : float
        Fraction of the distance to the boundary taken (see
        :func:`~aquaflux.solve.positive_block_limit`).

    Returns
    -------
    callable or None
        ``(phi, delta) -> alpha_max`` for a directly-solved ``k``; ``None`` otherwise.
    """
    if not isinstance(coupled.k_transform, DirectScalars):
        return None
    layout = coupled.layout
    n, dim = layout.n_cells, layout.dim
    return positive_block_limit((dim + 1) * n, (dim + 2) * n, tau)


def coupled_scaled_norm(
    coupled: CoupledRANS,
    shift_policy: CoupledShiftPolicy,
    state: jnp.ndarray,
) -> RowScaledNorm:
    """Build the row-equilibrated residual measure for the coupled state at ``state``.

    Assembles the two scales :class:`~aquaflux.solve.RowScaledNorm` needs, per block of the coupled
    layout ``[vel_0..vel_{dim-1}, pressure, k, omega]``:

    * **Row scale** -- each row's own diagonal coefficient, taken from the pseudo-transient shift's
      base diagonal, which is exactly that quantity per block (the momentum ``a_P`` on velocity, the
      transport diagonal on ``k`` and ``omega``) and so cannot drift from it. Two rows are not covered
      by it and are supplied here:
      - **Continuity carries no diagonal** -- it is a constraint, so the shift leaves it at zero. Its
        residual is a mass imbalance, and the natural scale of the same units is the cell's mass
        throughput ``sum_f max(mdot_f, 0)``. Dividing by it needs no pressure difference, so it stays
        well posed on a periodic or closed domain where a pressure scale degenerates.
      - **The near-wall fixed ``omega`` rows** hold an algebraic constraint rather than a balance, and
        the shift zeroes them. Their derivative comes from the row itself
        (:meth:`~aquaflux.discretization.FixationRow.jacobian_scale`) -- one, for a fixation written in
        the solved variable -- so they pass through unscaled rather than being divided by a
        neighbouring transport row's diagonal, which would misreport them by orders of magnitude.

    * **Field scale** -- ``mean(phi / (dphi/dw))``, which turns the stage-1 quotient (a change in the
      *solved* unknown) into a fractional change in the *physical* field. For a directly-solved field
      this is the familiar ``mean|phi|``; for a log-solved one ``dphi/dw = phi``, so the scale is
      exactly **one** -- a change in ``log phi`` already *is* a fractional change, and dividing by
      ``mean|phi|`` a second time would be wrong. Continuity likewise takes one, being dimensionless
      after stage 1.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled assembler, for the layout and the physical fields.
    shift_policy : CoupledShiftPolicy
        The policy whose base shift diagonal supplies the per-row diagonals.
    state : jnp.ndarray
        The coupled state the scales are measured at, shape ``((dim + 3) n_cells,)``.

    Returns
    -------
    RowScaledNorm
        The measure, with its scales frozen at ``state``. Only the block ``sizes`` are static, so the
        scales ride as ordinary array leaves and **re-deriving the measure at a new state is a
        compilation cache hit** -- the block structure is unchanged, only the numbers move. Rebuild it
        every outer iteration, and hold it fixed across a line search (otherwise a candidate could be
        preferred for shrinking its own denominator rather than its residual).
    """
    layout = coupled.layout
    n, dim = layout.n_cells, layout.dim
    tiny = 1e-300

    diagonal = jax.lax.stop_gradient(shift_policy.shift_term(state).diagonal)
    flow_diag, k_diag, omega_diag = layout.unpack(diagonal)
    velocity_diag, _pressure_diag = coupled.momentum.unpack(flow_diag)

    flow, k, omega = coupled.physical_fields(state)
    velocity, _pressure = coupled.momentum.unpack(flow)
    # Continuity's stand-in diagonal: the convective bucket is the per-cell mass throughput, in the
    # same units as the mass imbalance the row measures.
    throughput, _dissipative = coupled.momentum.momentum_matrix_diagonal_parts(velocity)

    k_chain = coupled.k_transform.jacobian_scale(k)
    omega_chain = coupled.omega_transform.jacobian_scale(omega)
    # A zeroed shift entry marks a row the shift does not own -- the fixed near-wall omega cells. Ask
    # the fixation row for its own derivative there instead of borrowing a transport row's.
    omega_fixed = coupled.omega_transform.fixation_row().jacobian_scale(omega, omega_chain)
    omega_rows = jnp.where(omega_diag > 0.0, omega_diag, omega_fixed)
    k_rows = jnp.where(
        k_diag > 0.0, k_diag, coupled.k_transform.fixation_row().jacobian_scale(k, k_chain)
    )

    row_scale = layout.pack(
        coupled.momentum.pack(jnp.abs(velocity_diag) + tiny, jnp.abs(throughput) + tiny),
        jnp.abs(k_rows) + tiny,
        jnp.abs(omega_rows) + tiny,
    )
    velocity_scale = jnp.mean(jnp.abs(velocity))
    field_scale = jnp.concatenate(
        [
            jnp.full((dim,), velocity_scale),
            # Continuity is already dimensionless once divided by the mass throughput.
            jnp.ones((1,)),
            # phi / (dphi/dw) converts a change in the solved unknown into a fractional change in the
            # physical field: mean|phi| for a directly-solved field, exactly one for a log-solved one.
            jnp.mean(jnp.abs(k) / jnp.maximum(k_chain, tiny))[None],
            jnp.mean(jnp.abs(omega) / jnp.maximum(omega_chain, tiny))[None],
        ]
    )
    return RowScaledNorm(
        sizes=(n,) * (dim + 3),
        row_scale=jax.lax.stop_gradient(row_scale),
        field_scale=jax.lax.stop_gradient(field_scale),
    )


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
    inner_steps: int = 1,
    inner_tol: float = 0.05,
    grow: int = 0,
    descent_backoff: int = 0,
    descent_test: bool = False,
    forward_solver: lx.AbstractLinearSolver | None = None,
    block_scaled_norm: bool = False,
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
    velocity_shift_parts: VelocityShiftParts | None = None,
    reuse: CoupledShiftPolicy | None = None,
    residual_norm: ResidualNorm | None = None,
    inner_observer: Callable[..., None] | None = None,
    **preconditioner_kwargs: object,
) -> ForwardStep:
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
    inner_steps : int
        ``> 1`` selects a **dual-time** (backward-Euler) march (:class:`~aquaflux.solve.DualTimeStep`)
        instead of the default single-step pseudo-transient continuation: each outer timestep holds a
        reference and runs up to this many inner Newton iterations on the transient residual
        ``R + beta d (phi - phi_ref)``. That puts the shift in the residual, so the measured steady
        residual is the honest discrete time derivative (not ``beta x travel``) and a larger
        pseudo-timestep (smaller ``beta``, driven by a step control) can be taken stably from a cold
        start. ``1`` (default) is the ordinary single shifted step, unchanged. The inner loop replaces
        the escalation ladder, so ``max_escalations`` / ``escalation_factor`` / ``divergence_cap`` /
        ``grow`` / ``descent_backoff`` / ``descent_test`` do not apply when it is on.
    inner_tol : float
        The dual-time inner loop stops once ``||G||`` has fallen to this fraction of the anchor residual
        (default ``0.05``); ignored unless ``inner_steps > 1``.
    forward_solver : lineax.AbstractLinearSolver or None
        The shifted-solve Krylov solver; ``None`` uses :data:`_COUPLED_FORWARD_SOLVER` (a
        larger-restart GMRES that stops on a global 2-norm relative residual, so each inexact-Newton
        step is solved to ~1% rather than driven to machine precision by the near-zero-right-hand-side
        wall rows).
    block_scaled_norm : bool
        Select the coarser :class:`~aquaflux.solve.BlockScaledNorm` (each of ``[flow, k, omega]``
        divided by its own initial magnitude) instead of the default row-equilibrated
        :class:`~aquaflux.solve.RowScaledNorm`. Both weigh every field rather than the ``omega`` block
        that dominates the plain Euclidean norm; the row-scaled default additionally equilibrates each
        row by its own diagonal, giving a fractional change per equation (see
        :func:`coupled_scaled_norm`). ``False`` (default) uses the row-scaled measure.
    shift_basis : ShiftBasis
        How the pseudo-time shift diagonal is built from each block's convective/dissipative operator
        parts (velocity, k and omega alike; see :class:`CoupledShiftPolicy`). Defaults to
        :class:`~aquaflux.solve.LocalCourantBasis` -- the full operator diagonal (uniform
        under-relaxation), unchanged from the historical shift. Pass
        ``LocalCourantBasis(dissipative_weight=0.0)`` for a local convective time step on the transport
        blocks (pressure keeps its zero shift either way).
    reuse : CoupledShiftPolicy, optional
        An existing policy to **refresh** at ``reference_state`` instead of building one from scratch:
        the k/omega AMGs are re-derived on their reused coarsening while the flow block is carried over
        untouched (see :func:`_coupled_shift_policy`). Use it to re-freeze a stale preconditioner part
        way through a march, once the flow has developed.
    residual_norm : ResidualNorm, optional
        The progress measure to use, overriding ``block_scaled_norm``. ``solve_coupled`` passes the
        march's initial measure here on every refresh, so a self-normalising :class:`RowScaledNorm` /
        :class:`BlockScaledNorm` keeps its per-row/per-block reference scales fixed at the state the
        global progress reference was measured against, rather than re-basing toward one at each
        developed refresh state (seam 4). ``None`` (a fresh, non-refresh build) constructs the default
        row-scaled measure (or the block-scaled one when ``block_scaled_norm``).
    **preconditioner_kwargs
        Forwarded to :meth:`~aquaflux.flow.BlockPreconditioner.build` for the flow block (e.g.
        ``schur_scaling``, ``velocity``). Ignored when ``reuse`` is given, since the flow block is then
        carried over rather than rebuilt.

    Returns
    -------
    ForwardStep
        The forward step to hand :class:`~aquaflux.solve.ImplicitNewtonSolver` as ``forward_step`` -- a
        :class:`~aquaflux.solve.PseudoTransientStep` by default, or a
        :class:`~aquaflux.solve.DualTimeStep` when ``inner_steps > 1``.
    """
    policy = _coupled_shift_policy(
        coupled,
        reference_state,
        method,
        reuse,
        shift_basis,
        velocity_shift_parts,
        **preconditioner_kwargs,
    )
    # An explicit `residual_norm` (passed by `solve_coupled` on every refresh) is used as-is, so the
    # block-scaled measure's per-field reference magnitudes stay fixed at the state the *global*
    # progress reference was measured against. Rebuilding it at each refresh's developed state would
    # re-base a self-normalising `BlockScaledNorm` back toward one, making the convergence test
    # unreachable and mismatching the finishing solve's absolute target (issue #156, seam 4).
    if residual_norm is None:
        # The default progress measure is the row-equilibrated norm. The plain Euclidean norm of the
        # coupled residual is dominated by the omega block (its magnitude dwarfs the flow and k blocks),
        # so it barely moves while the flow develops and mis-ranks a separating flow -- steering and the
        # stopping test then judge omega alone. `RowScaledNorm` (:func:`coupled_scaled_norm`) divides
        # each row by its own diagonal and each block by its field magnitude, reporting a fractional
        # change per equation, so every block contributes comparably. `block_scaled_norm=True` selects
        # the coarser per-block variant; pass `residual_norm=jnp.linalg.norm` for the plain Euclidean one.
        residual_norm = (
            _coupled_residual_norm(coupled, reference_state)
            if block_scaled_norm
            else coupled_scaled_norm(coupled, policy, reference_state)
        )
    schedule = SwitchedEvolutionRelaxation(beta0=beta0, exponent=exponent, beta_floor=beta_floor)
    solver = forward_solver if forward_solver is not None else _COUPLED_FORWARD_SOLVER
    if inner_steps > 1:
        # Dual-time (backward-Euler) march: an inner Newton loop per outer timestep on the transient
        # residual, so the measured steady residual is the honest discrete time derivative rather than
        # beta x travel, and a larger pseudo-timestep (smaller beta, driven by a step control) stays
        # stable. Reuses the same frozen shift policy, schedule, solver and measure. The inner loop
        # replaces the escalation ladder, so the escalation/acceptance parameters do not apply.
        return DualTimeStep(
            policy,
            relaxation_schedule=schedule,
            inner_steps=inner_steps,
            inner_tol=inner_tol,
            line_search=line_search,
            forward_solver=solver,
            residual_norm=residual_norm,
            adjoint_preconditioner_factory=policy.adjoint_factory(),
            inner_observer=inner_observer,
        )
    return PseudoTransientStep(
        policy,
        relaxation_schedule=schedule,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        acceptance=DivergenceGuard(divergence_cap=divergence_cap),
        line_search=line_search,
        grow=grow,
        descent_backoff=descent_backoff,
        descent_test=descent_test,
        forward_solver=solver,
        residual_norm=residual_norm,
        adjoint_preconditioner_factory=policy.adjoint_factory(),
    )


def _coupled_shift_policy(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    method: str | None,
    reuse: CoupledShiftPolicy | None = None,
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
    velocity_shift_parts: VelocityShiftParts | None = None,
    **preconditioner_kwargs: object,
) -> CoupledShiftPolicy:
    """Build the block-diagonal :class:`CoupledShiftPolicy` frozen at ``reference_state``.

    The preconditioner-freezing half of :func:`coupled_continuation`, split out so the mass-flow
    constraint (:func:`mass_flow_coupled_continuation`) can border the *same* policy rather than
    re-derive it.

    ``reuse`` **refreshes** an existing policy at a new (more developed) ``reference_state`` rather than
    building one from scratch. The scalar k/omega AMGs are re-derived on their reused coarsening
    (:func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner`'s ``reuse=`` -- worth
    ~2.4x in outer Krylov cycles on a separated backward-facing-step state), and the shift's **transport
    time scale is rebuilt** at the new state. **Carried over from ``reuse`` untouched**: the flow block
    (re-freezing it at the developed state was measured no help and slightly harmful, and it is the
    expensive half) and the shift's **coordinate factor** ``jacobian_scale``.

    **Why the shift is split into transport time scale x coordinate factor (binding).** Under a
    log-solved scalar the shift diagonal is the transport-operator diagonal times ``jacobian_scale``,
    which **is the field itself** -- the correct linearization of a pseudo-time term on ``omega``, since
    ``V/dt (omega^{n+1} - omega^n)`` becomes ``V/dt * omega * dw``. Its side effect is that the damping
    inherits ``omega``'s dynamic range, and that is what made the **product** unsafe to rebuild: measured
    against the cold-initial-condition diagonal on a developed backward-facing step, the rebuilt ``omega``
    block's ratio has median 0.87 but a p99 of 14 and a maximum of 24, with **15 % of cells above 2x**
    (the velocity and ``k`` blocks have no such tail). Those over-damped cells freeze (``delta_omega ~
    0``) while the rest of the field moves -- the recirculation and ``k`` static while the residual creeps
    upward ~1e-5 per step, no error and no divergence-guard trip. (Isolated by a controlled discriminator:
    rebuilding the product and carrying the AMG froze the march *byte-identically* to rebuilding both,
    while carrying the product and refreshing only the AMG descended -- so the shift rebuild was the
    freeze, independent of the switched-evolution-relaxation ``beta``.) So storing only the product forced
    a choice between a stale time scale and a frozen march.

    Storing the two factors separately dissolves that. The transport diagonal is *physics* -- a local
    time scale that should track the developing flow -- so a refresh rebuilds it; the ``jacobian_scale``
    factor is the *coordinate transformation* between the physical field and the solved log variable, not
    physics, so a refresh carries it frozen. The temporal ratio the shift then presents is
    ``transport(state)/transport(reference)``, in which the frozen ``omega`` weighting cancels, so the
    near-wall weighting is preserved while the field's range no longer leaks in: the ``>2x`` tail drops to
    the ``0.0-0.1 %`` of the velocity/``k`` blocks, and a march upgrading its shift at every refresh holds
    a full unclipped step where rebuilding the old product collapses it. (The preconditioner's copy of the
    factor, ``k_scale``/``omega_scale``, is instead re-derived at the new state, because its AMG is
    refreshed at the new *physical* operator -- so the same quantity legitimately comes from two states in
    one policy.) Carrying the frozen factor is safe for the same reason the flow block is: the shift is a
    transient device that vanishes at the root, so a slightly-stale factor changes only the path, never
    the converged state or its adjoint. A non-refresh build (``reuse is None``) uses the reference-state
    factor, so the shift product is bit-identical to the pre-split form.
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
            **{
                "velocity": "convection",
                "schur_scaling": "msimpler",
                # Aggregate the velocity/Schur AMGs along strong connections. A no-op on a low-aspect-
                # ratio mesh (this pitzDaily case), but the fix that keeps the V-cycle contracting once
                # the near-wall cells are strongly stretched (wall-resolved / skewed meshes), where
                # isotropic aggregation coarsens across the stiff wall-normal direction and stalls. The
                # flow block is frozen at the reference state (never refreshed), so the value-dependent
                # coarsening this turns on carries no refresh cost.
                "strength_threshold": 0.25,
                **preconditioner_kwargs,
            },
        )
    )

    mdot = momentum.mass_flux(flow_ref)

    # The reparametrized block's Jacobian is the physical one scaled by d(phi)/d(w): its shift diagonal
    # is scaled by that factor and its (physical-operator) preconditioner by the reciprocal. For the
    # identity transform the factor is one, so the direct path is unchanged. The omega block's
    # near-wall rows are a value fixation rather than a transport balance, so they take the fixation
    # row's own derivative instead of the chain factor (the shift is zero there either way).
    k_scale = _row_jacobian_scale(coupled.k_transform, k_ref)
    omega_scale = _row_jacobian_scale(
        coupled.omega_transform, omega_ref, coupled.turbulence.wall_cells
    )

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

    # On a refresh keep the basis the reused policy was built with, so the rebuilt transport diagonal
    # combines its convective/dissipative parts the same way the carried coordinate factor expects.
    basis = reuse.shift_basis if reuse is not None else shift_basis

    # The transport time scale is (re)built at THIS reference state -- physics that should track the
    # developing flow. It no longer carries the field's dynamic range: that now lives in the coordinate
    # factor below, which a refresh carries frozen, so the temporal ratio transport(state)/transport(ref)
    # has the range cancel and the shift does not inherit omega's growth (the freeze the old carried
    # product suffered -- see the docstring).
    k_transport = coupled.turbulence.k_shift_policy(
        mdot, closure, k_ref, shift_basis=basis
    ).shift_diagonal
    omega_transport = coupled.turbulence.omega_shift_policy(
        mdot, closure, omega_ref, shift_basis=basis
    ).shift_diagonal

    # The coordinate factor d(phi)/d(w) is the transform between the physical field and the solved
    # variable, not physics: a refresh carries it frozen (the preconditioner's copy, `k_scale`/
    # `omega_scale`, is re-derived at the new state instead, since its AMG is refreshed at the new
    # physical operator -- so the same quantity legitimately comes from two states here). A non-refresh
    # build uses the reference-state factor, which makes the product bit-identical to the old shift.
    k_coord = reuse.k_jacobian_scale if reuse is not None else k_scale
    omega_coord = reuse.omega_jacobian_scale if reuse is not None else omega_scale

    # Carried on a refresh like the basis: the source is a configuration choice, not frozen state, so
    # a refresh must not silently drop the caller's selection back to the preconditioner-derived default.
    parts = reuse.velocity_shift_parts if reuse is not None else velocity_shift_parts
    return CoupledShiftPolicy(
        coupled.layout,
        block,
        k_transport,
        k_coord,
        omega_transport,
        omega_coord,
        k_amg,
        omega_amg,
        shift_basis=basis,
        velocity_shift_parts=parts,
    )


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


class MonolithicFactorShiftPolicy(eqx.Module):
    """A coupled :class:`~aquaflux.solve.ShiftPolicy` that preconditions the whole ``[flow, k, omega]``
    saddle with one monolithic factorization of the assembled coupled Jacobian, in place of the
    block-diagonal composition.

    Reuses :class:`CoupledShiftPolicy`'s pseudo-transient shift diagonal -- the physics, the same
    velocity ``a_P`` and k/omega transport diagonals -- but replaces its block-diagonal preconditioner
    with a single monolithic factorization of the assembled coupled Jacobian, which forms the true
    pressure Schur coupling through its fill rather than approximating it. The factorization is either
    an incomplete threshold-ILU (:class:`~aquaflux.solve.MonolithicIlutPreconditioner`, a handful of
    Krylov cycles) or a complete LU (:class:`~aquaflux.solve.MonolithicLuPreconditioner`, exact, one
    cycle) -- this policy is agnostic to which, needing only the shared callback-matvec interface. On a
    convection-dominated collocated Rhie--Chow RANS saddle either reaches the forward tolerance where the
    block-triangular preconditioner needs hundreds of cycles.

    The factorization is frozen at a reference state and shift (built off the jit path by
    :func:`coupled_ilut_continuation` / :func:`coupled_lu_continuation`). Unlike the block
    preconditioner's live ``a_P`` rescaling it does not track the developing state; being a far stronger
    preconditioner it tolerates that freezing at a cost of a few extra cycles, and the shift vanishes at
    the root so the frozen factorization never changes the converged solution or its adjoint. Because it
    is a host object (``scipy`` / UMFPACK) it rides as a **static** field rather than a traced pytree
    leaf, and is applied inside the jitted Krylov solve through the callback matvec.

    Attributes
    ----------
    base : CoupledShiftPolicy
        The block policy supplying the pseudo-transient shift diagonal.
    preconditioner : MonolithicIlutPreconditioner or MonolithicLuPreconditioner
        The frozen coupled factorization (a static field). Any object exposing the ``matvec`` /
        ``matvec(transpose=True)`` callback interface works.
    """

    base: CoupledShiftPolicy
    preconditioner: (
        MonolithicIlutPreconditioner | MonolithicLuPreconditioner | MonolithicAmgPreconditioner
    ) = eqx.field(static=True)

    def shift_term(self, phi: jnp.ndarray) -> ShiftTerm:
        """The block policy's shift diagonal, glued to the frozen factorization preconditioner.

        For a factorization (ILUT/LU) the preconditioner is a single frozen apply and the step solves the
        shifted system with the JAX-side Krylov. A preconditioner exposing a native full solve (the AMG
        V-cycle) instead returns a **tagged full-solve** the step applies directly on the host -- the
        multigrid V-cycle is only a *moderate* inverse, so the JAX-side Krylov with it as a per-matvec
        callback needs tens of iterations, where PETSc's own GMRES driving the same V-cycle natively
        reaches the 1% stop in ~1 iteration (measured, ~60x faster per step). The forward-only native solve
        does not touch the differentiable path: the adjoint uses the single-V-cycle transpose below.

        Parameters
        ----------
        phi : jnp.ndarray
            The flat coupled state ``[flow..., k, omega]``, shape ``((dim + 3) n_cells,)``.
        """
        diagonal = self.base.shift_term(phi).diagonal
        if getattr(self.preconditioner, "is_exact_native", False):
            # The step applies the native exact-Jacobian full solve directly (see `_shifted_solve`):
            # `preconditioner.exact_solve(phi, -rhs, shift)`. The shift already carries the relaxation.
            return ShiftTerm(diagonal, lambda relaxation: self.preconditioner)
        apply = self.preconditioner.matvec()
        # The factorization is frozen, so the preconditioner does not depend on the shift strength.
        return ShiftTerm(diagonal, lambda relaxation: apply)

    def adjoint_factory(self) -> TransposedPreconditioner:
        """The ``state -> M^T`` factory for the adjoint transpose solve.

        The converged-state adjoint preconditions the (unshifted) transposed coupled Jacobian with the
        frozen factorization's transpose -- the same factors applied with a transposed triangular solve.
        Wrapped in a :class:`~aquaflux.solve.TransposedPreconditioner` because it
        already returns ``M^T``: the generic adjoint machinery derives the transpose with
        :func:`jax.linear_transpose`, which cannot handle the host-callback factorization, so it is
        applied directly instead.
        """
        return TransposedPreconditioner(FrozenTransposeFactory(self.preconditioner))


@dataclasses.dataclass(frozen=True)
class FrozenTransposeFactory:
    """``state -> M^T`` for a frozen monolithic factorization, as a value object rather than a closure.

    The transpose is state-independent -- the factorization is frozen, so the same ``M^T`` serves every
    state -- which is exactly why this can be a value whose equality is the preconditioner's identity.

    That matters because it ends up in a forward step's ``adjoint_preconditioner_factory``, a *static*
    field and hence part of the compiled step's cache key. As a lambda it compared by identity, so a
    Reynolds-continuation rung that rebuilt its engine got a fresh key and recompiled the coupled solve
    even when it was reusing the very same preconditioner. As a value object, two engines sharing one
    preconditioner produce equal factories and the rebuild is a cache hit.

    Attributes
    ----------
    preconditioner : object
        The frozen factorization, supplying ``matvec(transpose=True)``. Compared by identity, which is
        the intended meaning: the same preconditioner object *is* the same operator, and two distinct
        objects generally are not.
    """

    preconditioner: object

    def __call__(self, state: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
        del state  # frozen: the transpose does not depend on where the adjoint is taken
        return self.preconditioner.matvec(transpose=True)


def _coupled_jacobian_colouring(coupled: CoupledRANS, stencil_reach: int):
    """The block-stencil colouring for materializing the coupled Jacobian (a mesh-fixed quantity).

    Shared by :func:`coupled_ilut_continuation` (the initial factorization) and
    :func:`coupled_ilut_refreshing_continuation` (each in-place refactor), so both probe the Jacobian
    with the same colouring.
    """
    n_cells = coupled.momentum.mesh.n_cells
    owner, nb, _ = coupled.momentum.mesh.face_cells.interior_edges()
    return block_stencil_colouring(np.asarray(owner), np.asarray(nb), n_cells, stencil_reach)


def _frozen_shift_diagonal(base: CoupledShiftPolicy, beta: float, state: jnp.ndarray) -> np.ndarray:
    """The frozen pseudo-transient shift diagonal the factorization is built against, at ``state``.

    ``beta`` scales the base policy's shift diagonal; the ``stop_gradient`` keeps the frozen
    factorization off the differentiation path. Shared by the initial build and every in-place refresh,
    for both the ILUT and complete-LU preconditioners.
    """
    return np.asarray(beta * jax.lax.stop_gradient(base.shift_term(state).diagonal))


def _monolithic_factor_step(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    base: CoupledShiftPolicy,
    preconditioner: (
        MonolithicIlutPreconditioner | MonolithicLuPreconditioner | MonolithicAmgPreconditioner
    ),
    *,
    beta0: float,
    exponent: float,
    beta_floor: float,
    max_escalations: int,
    escalation_factor: float,
    divergence_cap: float,
    line_search: int,
    inner_steps: int,
    inner_tol: float,
    forward_solver: lx.AbstractLinearSolver | None,
    block_scaled_norm: bool,
    residual_norm: ResidualNorm | None,
    inner_observer: Callable[..., None] | None = None,
    refresh_on_cycles: int | None = None,
    inner_refresh: Callable[[jnp.ndarray], None] | None = None,
    cycle_budget: int | None = None,
    step_limit: Callable[..., jnp.ndarray] | None = None,
) -> ForwardStep:
    """Assemble the pseudo-transient / dual-time step around a frozen monolithic factorization.

    The shared tail of :func:`coupled_ilut_continuation` and :func:`coupled_lu_continuation`: it glues the
    already-built ``preconditioner`` (ILUT or complete-LU) to the block shift ``base`` via a
    :class:`MonolithicFactorShiftPolicy`, picks the row-equilibrated progress measure, and returns a
    :class:`~aquaflux.solve.DualTimeStep` (``inner_steps > 1``) or :class:`~aquaflux.solve.PseudoTransientStep`.
    The two builders differ only in how they construct ``preconditioner``.
    """
    policy = MonolithicFactorShiftPolicy(base, preconditioner)
    if residual_norm is None:
        # Row-equilibrated by default, as in `coupled_continuation`: the Euclidean coupled residual is
        # dominated by the omega block and mis-ranks a separating flow, so steering and the stopping test
        # would judge omega alone. `block_scaled_norm=True` selects the coarser per-block variant.
        residual_norm = (
            _coupled_residual_norm(coupled, reference_state)
            if block_scaled_norm
            else coupled_scaled_norm(coupled, policy, reference_state)
        )
    schedule = SwitchedEvolutionRelaxation(beta0=beta0, exponent=exponent, beta_floor=beta_floor)
    solver = forward_solver if forward_solver is not None else _COUPLED_ILUT_FORWARD_SOLVER
    if inner_steps > 1:
        # Dual-time (backward-Euler) march: an inner Newton loop per outer pseudo-timestep on the
        # transient residual, so a larger pseudo-timestep (smaller beta) stays stable. The inner loop
        # replaces the escalation ladder, so the escalation/acceptance parameters do not apply.
        return DualTimeStep(
            policy,
            relaxation_schedule=schedule,
            inner_steps=inner_steps,
            inner_tol=inner_tol,
            line_search=line_search,
            forward_solver=solver,
            residual_norm=residual_norm,
            adjoint_preconditioner_factory=policy.adjoint_factory(),
            inner_observer=inner_observer,
            refresh_on_cycles=refresh_on_cycles,
            inner_refresh=inner_refresh,
            cycle_budget=cycle_budget,
            step_limit=step_limit,
        )
    return PseudoTransientStep(
        policy,
        relaxation_schedule=schedule,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        acceptance=DivergenceGuard(divergence_cap=divergence_cap),
        line_search=line_search,
        forward_solver=solver,
        residual_norm=residual_norm,
        adjoint_preconditioner_factory=policy.adjoint_factory(),
    )


def coupled_ilut_continuation(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    *,
    ilut_beta: float = 2.0,
    stencil_reach: int = 3,
    fill_factor: float = 30.0,
    drop_tol: float = 1e-6,
    beta0: float = 2.0,
    exponent: float = 1.0,
    beta_floor: float = 0.0,
    max_escalations: int = 6,
    escalation_factor: float = 2.0,
    divergence_cap: float = 10.0,
    line_search: int = _COUPLED_LINE_SEARCH,
    inner_steps: int = 1,
    inner_tol: float = 0.05,
    forward_solver: lx.AbstractLinearSolver | None = None,
    block_scaled_norm: bool = False,
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
    residual_norm: ResidualNorm | None = None,
    inner_observer: Callable[..., None] | None = None,
) -> ForwardStep:
    """Build a pseudo-transient continuation step preconditioned by a monolithic coupled ILUT.

    The counterpart of :func:`coupled_continuation` that swaps the block-triangular SIMPLE preconditioner
    for one :class:`~aquaflux.solve.MonolithicIlutPreconditioner`. It materializes the coupled Jacobian at
    ``reference_state`` from ``coupled.residual`` (compressed graph-coloured probing -- one source of
    truth, no re-derived assembly), adds the pseudo-transient shift at ``ilut_beta``, and factors the
    result incompletely, all off the jit path. The resulting step is a drop-in for ``solve_coupled``'s
    ``continuation`` argument.

    Frozen preconditioner, like the block path: the factorization is built once at ``reference_state`` and
    does not refresh mid-march. Being a much stronger preconditioner it tolerates the state drift for a
    few extra cycles, and the shift vanishes at the root so freezing changes neither the converged state
    nor its adjoint.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler.
    reference_state : jnp.ndarray
        The coupled state the Jacobian and shift are frozen at, shape ``((dim + 3) n_cells,)``.
    ilut_beta : float
        The pseudo-transient shift strength the factorization is built at (the operator it factors is
        ``J + ilut_beta * d`` for the base shift diagonal ``d``). The march's own ``beta`` varies; the
        strong factorization tolerates the mismatch. Default matches ``beta0``.
    stencil_reach : int
        The cell-graph distance the Jacobian's sparsity is probed to (the coupled RANS Jacobian reaches
        distance ``3`` -- gradient reconstruction, Rhie--Chow, and the k/omega cross-coupling).
    fill_factor, drop_tol : float
        The incomplete-factorization fill controls (see
        :func:`~aquaflux.solve.ilut_preconditioner.factorize_ilut`). ``drop_tol = 1e-6`` keeps the small
        fill that forms the Schur coupling.
    beta0, exponent, beta_floor, max_escalations, escalation_factor, divergence_cap, line_search
        The pseudo-transient schedule and guard parameters, as in :func:`coupled_continuation`.
    inner_steps : int
        ``> 1`` builds a :class:`~aquaflux.solve.DualTimeStep` (an inner Newton loop per outer
        pseudo-timestep on the transient residual) preconditioned by the ILUT, instead of the default
        single-step :class:`~aquaflux.solve.PseudoTransientStep`; ``1`` (default) is the single-step
        march. The ILUT's true-inverse conditioning lets the dual-time pseudo-timestep grow well past the
        low-shift wall where the block-triangular preconditioner's coupled solve breaks down.
    inner_tol : float
        The inner-loop stopping tolerance (fraction of the reference residual), used only when
        ``inner_steps > 1``.
    forward_solver : lineax.AbstractLinearSolver or None
        The shifted-solve Krylov solver; ``None`` uses :data:`_COUPLED_ILUT_FORWARD_SOLVER`, a
        small-restart GMRES matched to the ILUT's fast convergence (the solve tests its stop only at
        each restart boundary, so a large restart would build many more preconditioned matvecs than the
        incomplete factorization needs -- see that constant's note).
    block_scaled_norm : bool
        Select the coarser per-block :class:`~aquaflux.solve.BlockScaledNorm` instead of the default
        row-equilibrated :class:`~aquaflux.solve.RowScaledNorm` (``False``, the default).
    shift_basis : ShiftBasis
        How the shift diagonal is built from the operator's convective/dissipative parts.
    residual_norm : ResidualNorm, optional
        An explicit progress measure (overrides ``block_scaled_norm``); ``solve_coupled`` passes the
        march's initial measure here on every refresh.

    Returns
    -------
    ForwardStep
        The :class:`~aquaflux.solve.PseudoTransientStep` to hand ``solve_coupled`` as ``continuation``.
    """
    # The base block policy supplies the pseudo-transient shift diagonal (the same velocity a_P + k/omega
    # transport diagonals); its scalar AMGs are skipped (`method=None`) since the ILUT preconditions every
    # block. The flow block is still assembled as the a_P source -- a lightweight shift-diagonal-only
    # policy is a follow-up optimization.
    base = _coupled_shift_policy(coupled, reference_state, None, shift_basis=shift_basis)
    n_fields = coupled.layout.dim + 3
    colouring = _coupled_jacobian_colouring(coupled, stencil_reach)
    frozen = jax.lax.stop_gradient(reference_state)

    def matvec(v):
        return _jacobian_matvec(coupled, frozen, v)

    preconditioner = MonolithicIlutPreconditioner.build(
        matvec,
        colouring,
        n_fields,
        _frozen_shift_diagonal(base, ilut_beta, reference_state),
        fill_factor=fill_factor,
        drop_tol=drop_tol,
    )
    return _monolithic_factor_step(
        coupled,
        reference_state,
        base,
        preconditioner,
        beta0=beta0,
        exponent=exponent,
        beta_floor=beta_floor,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        divergence_cap=divergence_cap,
        line_search=line_search,
        inner_steps=inner_steps,
        inner_tol=inner_tol,
        forward_solver=forward_solver,
        block_scaled_norm=block_scaled_norm,
        residual_norm=residual_norm,
        inner_observer=inner_observer,
    )


def _default_dual_time_control(
    step_control: StepControl | None, observing: bool, continuation: ForwardStep
) -> StepControl | None:
    """The step control for an observed march: the caller's, or the default Courant ramp for a dual-time
    march that was given none.

    A **dual-time** march (a :class:`~aquaflux.solve.DualTimeStep`, whose reported ``alpha`` is the
    backward-Euler inner-loop comfort a Courant ramp reads) that is **already observing** (a
    ``refresh_trigger`` or observer set ``observing``) but was handed **no** ``step_control`` defaults to
    :class:`~aquaflux.solve.DualTimeControl`. That ramp grows the pseudo-timestep while the inner loop
    stays comfortable, reaching a developed recirculation in far fewer outer steps than the residual-keyed
    schedule (which pins ``beta`` because the row-scaled steady residual is nearly flat while the flow
    develops). ``step_control`` is returned **unchanged** for a single-step march, a caller-supplied
    control, or a march that is not observing — so the default is injected only where a control actually
    runs, and injecting it never turns observation on (which would wrongly make the differentiable
    single-stage solve raise the forward-only guard).

    Parameters
    ----------
    step_control : StepControl or None
        The caller-supplied control (``None`` if none was given).
    observing : bool
        Whether the march runs the observed eager path (a refresh or observer is active).
    continuation : ForwardStep
        The globalization step the march applies.

    Returns
    -------
    StepControl or None
        ``DualTimeControl()`` when defaulting applies; ``step_control`` otherwise.
    """
    if step_control is None and observing and isinstance(continuation, DualTimeStep):
        return DualTimeControl()
    return step_control


def coupled_ilut_refreshing_continuation(
    coupled: CoupledRANS,
    *,
    ilut_beta: float = 2.0,
    stencil_reach: int = 3,
    fill_factor: float = 30.0,
    drop_tol: float = 1e-6,
    **continuation_kwargs: object,
) -> Callable[[jnp.ndarray], ForwardStep]:
    """A ``refresh_builder`` for :func:`solve_coupled` that keeps the coupled ILUT fresh cheaply.

    As the flow develops, the frozen ILUT goes stale and the shifted solve slows (or, on a low-shift
    dual-time path, fails). The usual fix -- rebuild the continuation at the developed state -- forces a
    full recompile of the jitted march-step, because a fresh continuation is a new pytree. This builder
    instead re-factors the **same** continuation's ILUT in place
    (:meth:`~aquaflux.solve.MonolithicIlutPreconditioner.refresh_in_place`): the preconditioner is a
    static field, so its identity is unchanged and the march-step is a **compilation cache hit** -- a
    refresh then costs only the materialize + factor, not a recompile.

    Returns a callable ``state -> ForwardStep``. The first call builds a
    :func:`coupled_ilut_continuation` at that state; each later call re-factors that continuation's ILUT
    at the given state and returns the **same** object. Pass it to :func:`solve_coupled` as both the
    initial ``continuation`` (via one call) and the ``refresh_builder``::

        rb = coupled_ilut_refreshing_continuation(coupled, inner_steps=10, inner_tol=1e-3, ...)
        solve_coupled(coupled, f0, k0, o0, continuation=rb(state0), refresh_builder=rb,
                      refresh_trigger=CoefficientDriftTrigger(threshold=0.1), ...)

    **Forward-march use ONLY.** The in-place refresh is impure (see ``refresh_in_place``) and must never
    be on a differentiated path; for a differentiated solve use :func:`coupled_ilut_continuation` with no
    refresh, whose converged root (and its adjoint) is refresh-independent anyway.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler.
    ilut_beta, stencil_reach, fill_factor, drop_tol : float / int
        As in :func:`coupled_ilut_continuation`. Used for **both** the initial build and every in-place
        refresh, so the two stay consistent.
    **continuation_kwargs
        Forwarded to :func:`coupled_ilut_continuation` for the initial build (``inner_steps``,
        ``inner_tol``, ``forward_solver``, ``shift_basis``, ``beta0``, ...).

    Returns
    -------
    callable
        ``state -> ForwardStep`` as described.
    """
    colouring = _coupled_jacobian_colouring(coupled, stencil_reach)
    n_fields = coupled.layout.dim + 3

    # `frozen` as a traced argument (not closed over) so this jvp-matvec compiles once and every refresh
    # reuses it, rather than a fresh lambda recompiling each time.
    def matvec_at(frozen, v):
        return _jacobian_matvec(coupled, frozen, v)

    held: dict[str, ForwardStep] = {}

    def builder(state: jnp.ndarray) -> ForwardStep:
        if "step" not in held:
            held["step"] = coupled_ilut_continuation(
                coupled,
                state,
                ilut_beta=ilut_beta,
                stencil_reach=stencil_reach,
                fill_factor=fill_factor,
                drop_tol=drop_tol,
                **continuation_kwargs,
            )
            return held["step"]
        step = held["step"]
        policy = step.shift_policy
        frozen = jax.lax.stop_gradient(state)
        policy.preconditioner.refresh_in_place(
            lambda v: matvec_at(frozen, v),
            colouring,
            n_fields,
            _frozen_shift_diagonal(policy.base, ilut_beta, state),
            fill_factor=fill_factor,
            drop_tol=drop_tol,
        )
        return step

    return builder


def coupled_lu_continuation(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    *,
    lu_beta: float = 2.0,
    stencil_reach: int = 3,
    backend: str = "auto",
    beta0: float = 2.0,
    exponent: float = 1.0,
    beta_floor: float = 0.0,
    max_escalations: int = 6,
    escalation_factor: float = 2.0,
    divergence_cap: float = 10.0,
    line_search: int = _COUPLED_LINE_SEARCH,
    inner_steps: int = 1,
    inner_tol: float = 0.05,
    forward_solver: lx.AbstractLinearSolver | None = None,
    block_scaled_norm: bool = False,
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
    residual_norm: ResidualNorm | None = None,
    inner_observer: Callable[..., None] | None = None,
) -> ForwardStep:
    """Build a pseudo-transient continuation step preconditioned by a monolithic **complete** coupled LU.

    The complete-LU counterpart of :func:`coupled_ilut_continuation`: it materializes the coupled Jacobian
    at ``reference_state`` (compressed graph-coloured probing -- one source of truth, no re-derived
    assembly), adds the pseudo-transient shift at ``lu_beta``, and factors the result **completely** with
    a :class:`~aquaflux.solve.MonolithicLuPreconditioner`, all off the jit path. Because the factorization
    is exact, the preconditioned Krylov solve converges in a single iteration; on a moderate
    two-dimensional mesh a fill-reducing multifrontal LU (UMFPACK) factors the coupled Jacobian roughly an
    order of magnitude faster than the threshold-ILU, so this is the preferred coupled preconditioner where
    the mesh is two-dimensional or moderate. On large three-dimensional meshes the complete factorization's
    fill (memory) becomes the wall and the ILUT / block paths are needed instead (see
    :class:`~aquaflux.solve.MonolithicLuPreconditioner`).

    A drop-in for ``solve_coupled``'s ``continuation`` argument, and (like the ILUT) reverse-differentiable
    through the converged state: the frozen factorization is ``stop_gradient``-ed, so the adjoint's
    transpose solve reuses the same factors and the gradient is the exact coupled implicit-function-theorem
    sensitivity.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler.
    reference_state : jnp.ndarray
        The coupled state the Jacobian and shift are frozen at, shape ``((dim + 3) n_cells,)``.
    lu_beta : float
        The pseudo-transient shift strength the factorization is built at (the operator it factors is
        ``J + lu_beta * d`` for the base shift diagonal ``d``). The march's own ``beta`` varies; the exact
        factorization tolerates the mismatch.
    stencil_reach : int
        The cell-graph distance the Jacobian's sparsity is probed to (coupled RANS reaches distance ``3``).
    backend : {'auto', 'umfpack', 'scipy'}
        The complete-LU backend (see :func:`~aquaflux.solve.lu_preconditioner.factorize_lu`). ``'auto'``
        uses UMFPACK (the fast path, via the optional ``petsc`` dependency) when available, else SciPy
        SuperLU.
    beta0, exponent, beta_floor, max_escalations, escalation_factor, divergence_cap, line_search,
    inner_steps, inner_tol, forward_solver, block_scaled_norm, shift_basis, residual_norm
        The pseudo-transient schedule, dual-time, guard, and measure parameters, exactly as in
        :func:`coupled_ilut_continuation`.

    Returns
    -------
    ForwardStep
        The step to hand ``solve_coupled`` as ``continuation``.
    """
    base = _coupled_shift_policy(coupled, reference_state, None, shift_basis=shift_basis)
    n_fields = coupled.layout.dim + 3
    colouring = _coupled_jacobian_colouring(coupled, stencil_reach)
    frozen = jax.lax.stop_gradient(reference_state)

    def matvec(v):
        return _jacobian_matvec(coupled, frozen, v)

    preconditioner = MonolithicLuPreconditioner.build(
        matvec,
        colouring,
        n_fields,
        _frozen_shift_diagonal(base, lu_beta, reference_state),
        backend=backend,
    )
    return _monolithic_factor_step(
        coupled,
        reference_state,
        base,
        preconditioner,
        beta0=beta0,
        exponent=exponent,
        beta_floor=beta_floor,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        divergence_cap=divergence_cap,
        line_search=line_search,
        inner_steps=inner_steps,
        inner_tol=inner_tol,
        forward_solver=forward_solver,
        block_scaled_norm=block_scaled_norm,
        residual_norm=residual_norm,
        inner_observer=inner_observer,
    )


def coupled_amg_continuation(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    *,
    amg_beta: float = 2.0,
    stencil_reach: int = 3,
    smoother_fill_levels: int = 1,
    smoother_sweeps: int = 2,
    coarse_eq_limit: int | None = None,
    native_forward_solve: bool = False,
    beta0: float = 2.0,
    exponent: float = 1.0,
    beta_floor: float = 0.0,
    max_escalations: int = 6,
    escalation_factor: float = 2.0,
    divergence_cap: float = 10.0,
    line_search: int = _COUPLED_LINE_SEARCH,
    inner_steps: int = 1,
    inner_tol: float = 0.05,
    forward_solver: lx.AbstractLinearSolver | None = None,
    forward_rtol: float = 0.3,
    forward_restart: int = 15,
    block_scaled_norm: bool = False,
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
    residual_norm: ResidualNorm | None = None,
    inner_observer: Callable[..., None] | None = None,
    refresh_on_cycles: int | None = None,
    inner_refresh: Callable[[jnp.ndarray], None] | None = None,
    cycle_budget: int | None = None,
    field_split: bool = False,
) -> ForwardStep:
    """Build a pseudo-transient continuation step preconditioned by a monolithic **algebraic-multigrid** V-cycle.

    The multigrid counterpart of :func:`coupled_ilut_continuation` / :func:`coupled_lu_continuation`: it
    materializes the coupled Jacobian at ``reference_state`` (compressed graph-coloured probing -- one
    source of truth, no re-derived assembly), adds the pseudo-transient shift at ``amg_beta``, and builds a
    single smoothed-aggregation V-cycle for it (:class:`~aquaflux.solve.MonolithicAmgPreconditioner`), all
    off the jit path. Unlike the two factorizations it keeps the heavy fill off the fine grid -- the only
    exact solve is a direct LU on the small coarsest grid -- so its memory stays bounded and its setup is
    seconds where the incomplete factorization's ``spilu`` runs for minutes on a distance-3 three-dimensional
    stencil. It is the coupled preconditioner for **large three-dimensional** meshes, where the complete
    LU's fill is out of memory and the ILUT's factorization is prohibitively slow to build; the LU / ILUT
    remain faster (fewer Krylov iterations) at two-dimensional / moderate size.

    A drop-in for ``solve_coupled``'s ``continuation`` argument, and (like the factorizations)
    reverse-differentiable through the converged state: the frozen V-cycle is ``stop_gradient``-ed, so the
    adjoint's transpose solve reuses the same (transposed) V-cycle -- the multigrid is a fixed *linear*
    operator, transposed through the cycle's own transpose, needing no flexible outer Krylov -- and the
    gradient is the exact coupled implicit-function-theorem sensitivity.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler.
    reference_state : jnp.ndarray
        The coupled state the Jacobian and shift are frozen at, shape ``((dim + 3) n_cells,)``.
    amg_beta : float
        The pseudo-transient shift strength the V-cycle is built at (the operator it preconditions is
        ``J + amg_beta * d`` for the base shift diagonal ``d``). The march's own ``beta`` varies; the strong
        V-cycle tolerates the mismatch.
    stencil_reach : int
        The cell-graph distance the Jacobian's sparsity is probed to (coupled RANS reaches distance ``3``).
    smoother_fill_levels : int
        Incomplete-LU fill levels of the stationary level smoother (``1`` = ILU(1); the indefinite saddle
        stalls at ``0``, and a Krylov-accelerated smoother would make the V-cycle nonlinear).
    smoother_sweeps : int
        Richardson sweeps of the level smoother per V-cycle visit.
    coarse_eq_limit : int or None
        The equation count at which aggregation stops and the coarsest grid is solved directly by LU. ``None``
        (default) keeps PETSc's default (~50); a larger value grows the coarse-level direct solve so it
        inverts more of the saddle's global pressure coupling exactly — a stronger V-cycle (and stronger
        transpose V-cycle, so it helps the adjoint too) at a bounded, sub-linearly-growing coarse-solve cost.
    beta0, exponent, beta_floor, max_escalations, escalation_factor, divergence_cap, line_search,
    inner_steps, inner_tol, forward_solver, block_scaled_norm, shift_basis, residual_norm
        The pseudo-transient schedule, dual-time, guard, and measure parameters, exactly as in
        :func:`coupled_ilut_continuation`. ``forward_solver`` defaults to a restart-40 GMRES matched to the
        V-cycle's convergence (a few tens of vectors), rather than the ILUT's restart-10.
    forward_rtol : float
        The relative tolerance for the default **row-scaled** forward-solve stop (default ``0.3``). With no
        explicit ``forward_solver``, the forward solve stops on the march's own row-scaled progress measure
        (:func:`coupled_scaled_norm`) rather than a plain global 2-norm — the latter is ~100% ``omega``
        (whose residual is orders above the flow), so it halts once ``omega`` is resolved while the
        flow-dominated Newton step is still coarse (measured ~116 % velocity error — effectively blind).
        The row-scaled stop weighs every field comparably; a *loose* ``forward_rtol`` keeps it cheap.
        Calibrated on the developed backward-facing step: at ``0.3`` the velocity correction is resolved to
        ~25 % for ~1.5× the plain-2-norm cycle count; ``0.1`` fully resolves it at ~2.25×. Ignored when an
        explicit ``forward_solver`` is given.
    forward_restart : int
        Arnoldi restart length of that default solver (``15``). Exposed on its own rather than left to a
        caller-built ``forward_solver``, because building one to change the restart also silently drops
        the row-scaled stop described above — a far larger change than intended. Worth varying because a
        restarted GMRES tests convergence only at restart boundaries: against a tolerance as loose as
        ``forward_rtol`` a solve can reach its target early in a cycle and go on building vectors, and a
        restart-*cycle* count cannot see that, since such a solve reports one cycle at any length. Ignored
        when an explicit ``forward_solver`` is given.
    inner_observer : callable or None
        A per-inner-iteration profiling hook forwarded to the built dual-time step (only used when
        ``inner_steps > 1``); see :class:`~aquaflux.solve.DualTimeStep`. ``None`` (default) leaves the step
        byte-identical. Forward-only — do not set it on a differentiated solve.
    cycle_budget : int or None
        A cap on the dual-time inner loop's accumulated linear-solve count, forwarded to the built
        :class:`~aquaflux.solve.DualTimeStep` (only used when ``inner_steps > 1``). It cuts off a primary
        solve grinding on a stiff low-β operator after ~``cycle_budget`` matvecs instead of the full
        ``inner_steps`` into the restart cap; pair it with ``solve_coupled``'s β-escalation
        (``retry_on_cycles < cycle_budget``), which redoes the capped step at a larger β. ``None`` (default)
        is unbounded and byte-identical. Forward-only.
    Returns
    -------
    ForwardStep
        The step to hand ``solve_coupled`` as ``continuation``.
    """
    base = _coupled_shift_policy(coupled, reference_state, None, shift_basis=shift_basis)
    n_fields = coupled.layout.dim + 3
    # The forward-solve stop. A plain global-2-norm relative stop is ~100% the largest-magnitude field
    # (``omega``, whose residual is orders above the flow), so a "1%" solve resolves ``omega`` and halts
    # while the flow-dominated Newton step is still coarse -- leaving the flow correction blind (measured
    # ~116 % velocity error). The forward solve therefore stops on the march's own row-scaled progress
    # measure (``coupled_scaled_norm``, a physically row-scaled -- not per-solve -- measure) at a *loose*
    # ``forward_rtol``: every field block is seen and resolved loosely, so the flow is never left blind
    # while ``omega`` is not over-resolved. A caller that wants a different stop passes ``forward_solver``.
    # Restart 15 is the measured sweet spot for the one-V-cycle preconditioner: enough Arnoldi history for
    # its convergence while checking the stop often enough not to overshoot the loose target deep into the
    # next cycle (a larger restart costs ~2x the expensive host V-cycle applies for the same trajectory);
    # ``max_restarts`` stays generous so a drifted-reference solve still completes. ``forward_restart``
    # exists so that length can be varied on its own -- passing a whole ``forward_solver`` to do it would
    # also drop the loose row-scaled stop above, which is a much larger change than the one intended.
    if forward_solver is None:
        forward_solver = relative_residual_gmres(
            forward_rtol,
            norm=coupled_scaled_norm(coupled, base, reference_state),
            restart=forward_restart,
            stagnation_iters=40,
            max_restarts=60,
        )
    colouring = _coupled_jacobian_colouring(coupled, stencil_reach)
    # The fixed CSR structure + gather map (pattern is mesh-fixed), so each materialize de-compresses by one
    # gather rather than a scatter loop + re-sort. Reused by the β-tracking refresh (built once there too).
    structure = block_stencil_gather_map(colouring, coupled.layout.dim + 3)
    frozen = jax.lax.stop_gradient(reference_state)

    def matvec(v):
        return _jacobian_matvec(coupled, frozen, v)

    # Batched jvp so the coloured materialize probes run as a few fused passes, not a per-probe loop.
    def batched_matvec(seeds):
        return _batched_jacobian_matvec(coupled, frozen, seeds)

    # `native_forward_solve` (EXPERIMENTAL, opt-in) runs the forward Krylov natively in PETSc, its operator
    # a shell over the exact jvp (true Newton, not a frozen Jacobian) -- the native GMRES + GAMG reaches its
    # stop in ~1 iteration where the JAX-side Krylov with the V-cycle as a per-matvec callback needs ~90
    # (the JAX-side GMRES is far slower on a well-preconditioned system; measured). The mechanism is
    # validated (native speed, correct step direction, exact-Newton), but the march currently converges
    # SLOWER than the default path per step: the default's JAX-side solver over-solves each step to
    # ~machine zero, and the pseudo-transient globalization implicitly leans on those near-exact steps,
    # which the native (honest-tolerance) step does not yet match -- a convergence-tuning follow-up. Default
    # off; the default path applies the frozen V-cycle per-matvec through the JAX-side Krylov.
    # `field_split` swaps ONLY which frozen inverse is fitted to the same materialized Jacobian: the flow
    # saddle and the two transported scalars get separate hierarchies, with one triangle of the coupling
    # between them retained exactly. Everything downstream -- the shift policy, the forward solver, the
    # step tail, the refresh hooks -- is shared, which is the point: it makes the two comparable on a
    # march by changing one construction, not by maintaining a second solver.
    shift = _frozen_shift_diagonal(base, amg_beta, reference_state)
    common = {
        "smoother_fill_levels": smoother_fill_levels,
        "smoother_sweeps": smoother_sweeps,
        "coarse_eq_limit": coarse_eq_limit,
        "batched_matvec": batched_matvec,
        "probe_batch_size": _PROBE_BATCH_SIZE,
        "structure": structure,
    }
    if field_split:
        if native_forward_solve:
            raise ValueError(
                "native_forward_solve builds a PETSc KSP around a single monolithic V-cycle and has no "
                "field-split counterpart; use one or the other."
            )
        preconditioner = FieldSplitAmgPreconditioner.build(
            matvec,
            colouring,
            n_fields,
            shift,
            FieldGroups(
                n_cells=coupled.layout.n_cells,
                n_leading_fields=coupled.layout.dim + 1,  # u, v, w, p -- the saddle
                n_trailing_fields=2,  # k, omega -- the transported scalars
            ),
            **common,
        )
    else:
        preconditioner = MonolithicAmgPreconditioner.build(
            matvec,
            colouring,
            n_fields,
            shift,
            native=native_forward_solve,
            residual_fn=coupled.residual if native_forward_solve else None,
            **common,
        )
    return _monolithic_factor_step(
        coupled,
        reference_state,
        base,
        preconditioner,
        beta0=beta0,
        exponent=exponent,
        beta_floor=beta_floor,
        max_escalations=max_escalations,
        escalation_factor=escalation_factor,
        divergence_cap=divergence_cap,
        line_search=line_search,
        inner_steps=inner_steps,
        inner_tol=inner_tol,
        forward_solver=forward_solver,
        block_scaled_norm=block_scaled_norm,
        residual_norm=residual_norm,
        inner_observer=inner_observer,
        refresh_on_cycles=refresh_on_cycles,
        inner_refresh=inner_refresh,
        cycle_budget=cycle_budget,
        # Keep `k` off zero: it is solved directly, and one negative cell reaches the closure's
        # sqrt(k) and NaNs the whole residual. `None` when the transform already guarantees it.
        step_limit=positive_k_limit(coupled),
    )


def coupled_lu_refreshing_continuation(
    coupled: CoupledRANS,
    *,
    lu_beta: float = 2.0,
    stencil_reach: int = 3,
    backend: str = "auto",
    **continuation_kwargs: object,
) -> Callable[[jnp.ndarray], ForwardStep]:
    """A ``refresh_builder`` for :func:`solve_coupled` that keeps the coupled complete-LU fresh cheaply.

    The complete-LU counterpart of :func:`coupled_ilut_refreshing_continuation`. Returns a callable
    ``state -> ForwardStep``: the first call builds a :func:`coupled_lu_continuation` at that state; each
    later call re-factors that continuation's LU **in place**
    (:meth:`~aquaflux.solve.MonolithicLuPreconditioner.refresh_in_place`) at the given state and returns
    the **same** object, so the jitted march-step is a compilation cache hit. With the UMFPACK backend the
    refresh reuses the symbolic factorization, so it costs only a cheap numeric refactorization. Pass it to
    :func:`solve_coupled` as both the initial ``continuation`` (via one call) and the ``refresh_builder``.

    **Forward-march use ONLY** -- the in-place refresh is impure and must never be on a differentiated path
    (use :func:`coupled_lu_continuation` with no refresh for a differentiated solve; the converged root and
    its adjoint are refresh-independent anyway).

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler.
    lu_beta, stencil_reach, backend : float / int / str
        As in :func:`coupled_lu_continuation`. Used for both the initial build and every in-place refresh.
    **continuation_kwargs
        Forwarded to :func:`coupled_lu_continuation` for the initial build.

    Returns
    -------
    callable
        ``state -> ForwardStep`` as described.
    """
    colouring = _coupled_jacobian_colouring(coupled, stencil_reach)
    n_fields = coupled.layout.dim + 3

    def matvec_at(frozen, v):
        return _jacobian_matvec(coupled, frozen, v)

    held: dict[str, ForwardStep] = {}

    def builder(state: jnp.ndarray) -> ForwardStep:
        if "step" not in held:
            held["step"] = coupled_lu_continuation(
                coupled,
                state,
                lu_beta=lu_beta,
                stencil_reach=stencil_reach,
                backend=backend,
                **continuation_kwargs,
            )
            return held["step"]
        step = held["step"]
        policy = step.shift_policy
        frozen = jax.lax.stop_gradient(state)
        policy.preconditioner.refresh_in_place(
            lambda v: matvec_at(frozen, v),
            colouring,
            n_fields,
            _frozen_shift_diagonal(policy.base, lu_beta, state),
        )
        return step

    return builder


def _staleness_beta_gate(*, refresh_every: int, beta_rel_change: float) -> Callable[[float], bool]:
    """A stateful predicate for the β-tracking refresh: *when* is a re-factor worth its cost?

    Returns ``should_refresh(beta) -> bool``. It fires on the first call, whenever ``β`` has moved by
    more than ``beta_rel_change`` (relative to the ``β`` of the last refresh), or after ``refresh_every``
    steps have passed with no refresh -- otherwise it returns ``False`` and the step reuses the standing
    factorization.

    The β-move trigger is what catches a shift-strength spike -- a dual-time overshoot (``β`` driven low,
    the operator stiff) or a rung restart (``β`` jumped back to ``beta_start``). A step-count cap alone
    would miss it for up to ``refresh_every`` steps, and a *drift* trigger (fired by coefficient change)
    would miss it entirely in the worst case: a badly mismatched factor stalls the line search (α→0), so
    the state stops moving, its coefficients stop drifting, and the drift trigger never fires. Keying the
    refresh on ``β`` itself removes that stall mode. The step-count cap is the complementary staleness
    bound for *state* development at a near-constant ``β`` (the flow developing during a cruise).

    Parameters
    ----------
    refresh_every : int
        Force a refresh after this many steps without one (the staleness cap for state development).
    beta_rel_change : float
        Refresh when ``|β - β_last| > beta_rel_change * |β_last|`` (the shift-mismatch trigger).

    Returns
    -------
    callable
        ``should_refresh(beta: float) -> bool``, carrying its own ``(β_last, steps_since)`` state.
    """
    last: dict[str, float | int] = {}

    def should_refresh(beta: float) -> bool:
        if "beta" not in last:
            last["beta"], last["since"] = beta, 0
            return True
        last["since"] += 1
        moved = abs(beta - last["beta"]) > beta_rel_change * max(abs(last["beta"]), 1e-30)
        if moved or last["since"] >= refresh_every:
            last["beta"], last["since"] = beta, 0
            return True
        return False

    return should_refresh


def _materialize_gate(
    drift_factory: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]],
    *,
    materialize_drift: float | None,
    materialize_every: int | None,
) -> Callable[[jnp.ndarray], bool]:
    """A stateful predicate for the β-diagonal split: should this refresh RE-MATERIALIZE the Jacobian
    (full, the coloured jvp probe) or only re-add the shift diagonal to the standing one (cheap)?

    Returns ``should_materialize(state) -> bool``. Re-materializing is the dominant refresh cost, so it is
    reserved for when the frozen Jacobian has actually gone stale -- i.e. when the state it was probed at
    has moved. The staleness signal is a **coefficient drift** since the last materialize, supplied by
    ``drift_factory`` (in the coupled march, :func:`eddy_viscosity_drift`: ``ν_t`` is what the operators are
    assembled from, so its movement is the Jacobian's staleness, and it is cheap -- one jitted evaluation).
    Fires when that drift exceeds ``materialize_drift``, OR after ``materialize_every`` steps without a
    materialize (a state-development cap for a near-constant coefficient) -- the drift-move / step-cap pair
    that mirrors :func:`_staleness_beta_gate`. The reference is re-based at every materialize (so the drift
    measures movement the last materialize did not absorb) and seeded on the first call from the freshly-built
    operator (so the first call needs only a shift, not a redundant materialize).

    Parameters
    ----------
    drift_factory : callable
        ``reference_state -> (state -> drift)`` -- builds a drift measure against a reference (a non-negative
        scalar, zero at the reference). Injected so the gate's decision logic is testable with a synthetic
        drift; the coupled march passes ``lambda ref: eddy_viscosity_drift(coupled, ref)``.
    materialize_drift : float or None
        Re-materialize when the drift since the last materialize exceeds this. ``None`` disables the drift
        trigger (then only the step cap fires).
    materialize_every : int or None
        Force a materialize after this many refreshes without one (the staleness cap). ``None`` disables the
        cap (then only the drift trigger fires).

    Returns
    -------
    callable
        ``should_materialize(state) -> bool``, carrying its own ``(drift reference, steps_since)`` state.
    """
    st: dict[str, object] = {"since": 0, "drift_fn": None}

    def should_materialize(state: jnp.ndarray) -> bool:
        st["since"] = int(st["since"]) + 1  # type: ignore[arg-type]
        if materialize_drift is not None and st["drift_fn"] is None:
            # Seed the drift reference at the freshly-built state; the Jacobian is already current here, so
            # this first refresh needs only a shift (drift is zero against its own reference).
            st["drift_fn"] = drift_factory(jax.lax.stop_gradient(state))
        drift_hit = (
            materialize_drift is not None
            and st["drift_fn"] is not None
            and float(st["drift_fn"](state)) > materialize_drift  # type: ignore[operator]
        )
        cap_hit = materialize_every is not None and int(st["since"]) >= materialize_every
        if drift_hit or cap_hit:
            st["since"] = 0
            if materialize_drift is not None:
                st["drift_fn"] = drift_factory(jax.lax.stop_gradient(state))
            return True
        return False

    return should_materialize


def _refresh_branch(*, stale_state: bool, moved_beta: bool, split: bool) -> str:
    """Which branch a β-tracking refresh should take: ``"full"``, ``"shift"`` or ``"none"``.

    The two staleness signals are **independent questions about different things**, and the whole point
    of this function is that they are combined rather than nested:

    * ``stale_state`` — the frozen Jacobian no longer matches the flow (the eddy viscosity has drifted).
      Only a re-probe fixes that, so it forces a ``"full"``.
    * ``moved_beta`` — the shift the V-cycle was built at no longer matches the one being solved. Only
      the diagonal is wrong, so a ``"shift"`` fixes it where that cheap branch exists.

    ``split`` says whether the cheap branch exists at all (an algebraic-multigrid preconditioner with a
    materialize gate configured). Without it there is one branch, and any trigger means ``"full"``.

    **Why this is a function and not three nested ``if``s at the call site.** It used to be nested — the
    state question asked *only* when the β question had already said yes — and that made state drift
    unable to trigger anything at all below the preconditioner's shift floor, where the clamped β never
    moves so the β question answers "no" forever. That is precisely the low-shift tail where the flow
    develops fastest. Measured on a three-dimensional cold march: below the floor 91 % of steps refreshed
    nothing while the eddy viscosity drifted ~20 % per step, and those steps carried ~47 % of the whole
    march's Krylov cost. The decision is small, total, and worth being able to read and test on its own.

    Parameters
    ----------
    stale_state : bool
        The state-drift gate fired (the Jacobian needs re-probing).
    moved_beta : bool
        The β-mismatch gate fired (the shift needs re-adding).
    split : bool
        Whether the cheap shift-only branch is available.

    Returns
    -------
    str
        ``"full"``, ``"shift"`` or ``"none"``.
    """
    if stale_state:
        return "full"
    if not moved_beta:
        return "none"
    return "shift" if split else "full"


def _beta_tracking_refresh(
    coupled: CoupledRANS,
    stencil_reach: int,
    *,
    gate: Callable[[float], bool] | None = None,
    refresh_kwargs: dict[str, object] | None = None,
    materialize_every: int | None = None,
    materialize_drift: float | None = None,
    beta_floor: float = 0.0,
    observer: Callable[[RefreshTiming], None] | None = None,
) -> Callable[[ForwardStep, jnp.ndarray], None]:
    """Shared skeleton for the β-tracking ``precondition_step`` hooks (complete-LU and ILUT).

    Returns a ``precondition_step(active_step, state)`` that reads ``β`` from the step's
    :class:`~aquaflux.solve.ConstantRelaxation` schedule and, when ``gate(β)`` allows, re-factors the
    step's :class:`MonolithicFactorShiftPolicy` preconditioner in place at ``J(state) + β·d(state)`` via
    ``refresh_in_place(**refresh_kwargs)``. ``gate=None`` refreshes **every** step (the cheap exact-LU
    cadence); a gate returning ``False`` reuses the standing factorization (the ILUT cadence, whose
    re-factor is too expensive to pay every step).

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler (supplies the Jacobian-vector product and the shift diagonal).
    stencil_reach : int
        The cell-graph distance the Jacobian's sparsity is probed to (coupled RANS reaches distance ``3``).
    gate : callable, optional
        ``should_refresh(beta: float) -> bool``. ``None`` means always refresh.
    refresh_kwargs : dict, optional
        Extra keyword arguments forwarded to the preconditioner's ``refresh_in_place`` (e.g. the ILUT's
        ``fill_factor`` / ``drop_tol``). ``None`` forwards none (the complete LU takes no extra options).

    Returns
    -------
    callable
        ``precondition_step(active_step, state) -> None``.
    """
    refresh_kwargs = {} if refresh_kwargs is None else refresh_kwargs
    colouring = _coupled_jacobian_colouring(coupled, stencil_reach)
    n_fields = coupled.layout.dim + 3
    # Fixed CSR structure + gather map for the AMG materialize (mesh-fixed pattern): each materialize
    # de-compresses by one gather rather than a scatter loop + re-sort. Built once; used only on the AMG path.
    structure = block_stencil_gather_map(colouring, n_fields)

    # `frozen` a traced argument (not closed over) so the jvp-matvec compiles once and every refactor
    # reuses it, rather than a fresh lambda recompiling each step.
    def matvec_at(frozen, v):
        return _jacobian_matvec(coupled, frozen, v)

    # Batched form (vmapped over the tangent) so the coloured probes of a full materialize run as a few
    # fused passes rather than a Python loop of separate calls. Built once (state-independent, `frozen` a
    # traced argument) so it compiles a single time and every materialize reuses it. Used only by the AMG
    # preconditioner's `refresh_in_place`.
    def batched_matvec_at(frozen, seeds):
        return _batched_jacobian_matvec(coupled, frozen, seeds)

    # The β-diagonal split's materialize gate (built once): decides per refresh whether to re-materialize
    # the Jacobian or only re-add the shift. `None` when neither trigger is set (then every refresh is a
    # full materialize, the original behaviour).
    materialize_gate = (
        _materialize_gate(
            lambda ref: eddy_viscosity_drift(coupled, ref),
            materialize_drift=materialize_drift,
            materialize_every=materialize_every,
        )
        if (materialize_drift is not None or materialize_every is not None)
        else None
    )

    def _report_refresh(
        kind: str, started: float, phases: tuple[tuple[str, float], ...] | None = None
    ) -> None:
        """Tell an injected observer which branch ran and what each part of it cost.

        The total alone cannot be acted on: a refresh dominated by the coloured jvp probe and one
        dominated by the multigrid setup take the same wall time and call for opposite fixes.
        """
        if observer is not None:
            observer(RefreshTiming(kind, time.perf_counter() - started, tuple(phases or ())))

    # Which step the inner-loop hook is refreshing, kept current by `precondition_step` below.
    bound_step: dict[str, ForwardStep] = {}

    def precondition_step(active_step: ForwardStep, state: jnp.ndarray) -> None:
        # The march calls this immediately before every step and again on every retry, always with the
        # CURRENT step -- so this is also where the inner-loop hook learns which step it is refreshing.
        # Binding once at construction cannot work: the step the builder returns still carries the
        # default schedule, and the march replaces it each iteration with one the control has set β on.
        bound_step["step"] = active_step
        schedule = active_step.relaxation_schedule
        beta = getattr(schedule, "beta", None)
        if beta is None:
            raise ValueError(
                "a β-tracking refresh needs the step's shift strength as a readable constant -- pair it "
                "with a DualTimeControl (which sets a ConstantRelaxation β), not the default "
                f"switched-evolution schedule ({type(schedule).__name__})."
            )
        beta = float(beta)
        started = time.perf_counter()
        # The preconditioner's shift is floored independently of the march's own beta. As beta -> 0 the
        # shift's diagonal dominance vanishes and the frozen V-cycle degrades, but the OPERATOR must keep
        # the small beta to make pseudo-transient progress. Flooring only the preconditioner's copy keeps
        # the V-cycle in a regime it inverts well while the solved system is untouched, so the converged
        # root and its adjoint are unchanged. The resulting mismatch SATURATES at `beta_floor * d` rather
        # than growing without bound the way a stale (never-refreshed) preconditioner's does.
        pc_beta = max(beta, beta_floor)
        policy = active_step.shift_policy
        pc = policy.preconditioner
        # β-diagonal split: β and the per-cell shift ``d`` touch only the diagonal, so between full
        # (re-materialized) refreshes the shift is tracked by re-adding the new ``β d`` diagonal to the
        # frozen Jacobian -- skipping the coloured-probe materialize (the dominant refresh cost). Only the
        # AMG preconditioner exposes that shift-only path; without it (ILUT/LU) every refresh is full.
        is_amg = hasattr(pc, "refresh_shift_in_place")
        split = materialize_gate is not None and is_amg
        # Both gates are stateful, so each must be called EXACTLY ONCE per step -- no short-circuiting.
        stale_state = bool(split and materialize_gate(state))
        moved_beta = gate is None or bool(gate(pc_beta))
        branch = _refresh_branch(stale_state=stale_state, moved_beta=moved_beta, split=split)
        if branch == "none":
            _report_refresh("none", started)
            return
        frozen = jax.lax.stop_gradient(state)
        shift = pc_beta * np.asarray(jax.lax.stop_gradient(policy.base.shift_term(state).diagonal))
        if branch == "shift":
            # The Jacobian still matches the flow, so only the shift needs re-adding. Note the shift is
            # `pc_beta * d(state)` and the per-cell `d` tracks the state even where `pc_beta` is pinned,
            # so this is real work below the floor, not a rebuild of an identical operator.
            _report_refresh("shift", started, pc.refresh_shift_in_place(shift))
            return
        _report_refresh("full", started, _materialize_at(pc, is_amg, frozen, shift))

    def _materialize_at(pc, is_amg, frozen, shift) -> tuple[tuple[str, float], ...]:
        """Re-materialize the preconditioner at ``frozen`` with shift diagonal ``shift``.

        The AMG preconditioner materializes via the coloured probe and takes the batched form; the
        factorization preconditioners (LU/ILUT) do not, so pass it only on the AMG path.
        """
        extra = (
            {
                "batched_matvec": lambda seeds: batched_matvec_at(frozen, seeds),
                "probe_batch_size": _PROBE_BATCH_SIZE,
                "structure": structure,
            }
            if is_amg
            else {}
        )
        return (
            pc.refresh_in_place(
                lambda v: matvec_at(frozen, v),
                colouring,
                n_fields,
                shift,
                **extra,
                **refresh_kwargs,
            )
            or ()
        )

    def refresh_at(iterate) -> None:
        """``inner_refresh`` hook: rebuild the preconditioner at this mid-step iterate.

        *When* to fire is decided by the dual-time loop (``DualTimeStep.refresh_on_cycles``), not here,
        so that the rule which triggers the refresh is the same one that forgives the abort it would
        otherwise be discarded by.

        The march's expensive inner solves are **stale-preconditioner** effects, not hard operators: at
        the hardest solve of a three-dimensional coupled march a preconditioner rebuilt at that very
        iterate converged in **one** cycle where the march's own took fifteen. Refreshing here — between
        inner iterations, after the line search and before the next solve — keeps the step's progress,
        where the alternative reaction (abort the step and escalate β) discards both the work and the
        pseudo-timestep.

        Costs are what make this worth doing as a *replacement* for a scheduled refresh rather than an
        addition to one: on that march the schedule spent 21 % of the wall keeping a preconditioner fresh
        that 83 % of solves did not need, and the right interval is regime-dependent in a way no fixed
        cadence can track (one step of staleness is free at a large shift and triples the cost at a small
        one).
        """
        if "step" not in bound_step:
            return
        started = time.perf_counter()
        pc = bound_step["step"].shift_policy.preconditioner
        beta = max(float(bound_step["step"].relaxation_schedule.beta), beta_floor)
        frozen = jax.lax.stop_gradient(jnp.asarray(iterate))
        shift = beta * np.asarray(
            jax.lax.stop_gradient(bound_step["step"].shift_policy.base.shift_term(frozen).diagonal)
        )
        _report_refresh(
            "inner",
            started,
            _materialize_at(pc, hasattr(pc, "refresh_shift_in_place"), frozen, shift),
        )

    precondition_step.refresh_at = refresh_at
    return precondition_step


def lu_beta_tracking_refresh(
    coupled: CoupledRANS, *, stencil_reach: int = 3
) -> Callable[[ForwardStep, jnp.ndarray], None]:
    """A ``precondition_step`` for :func:`solve_coupled` that re-factors the complete LU at the current β.

    The complete-LU preconditioner is the operator's **exact** inverse only for the operator it factored,
    ``J + β d``. In a dual-time march the shift strength ``β`` ramps (e.g. 0.5 → 0.005), so a factorization
    frozen at one ``β`` mis-preconditions the operator actually solved and the Krylov solve needs many
    cycles (measured: a complete LU frozen at ``β = 0.05`` takes ~470 GMRES iterations on the ``β = 2``
    operator, and can fail outright on a stiff overshot state) -- the opposite of the ILUT, whose
    *approximate* factorization tolerates the mismatch. Because the complete LU is **cheap** to factor
    (~1 s at 2D/moderate size), the right treatment is to re-factor it at the current ``(state, β)`` **every
    step**, so it is exact each step (a single Krylov iteration) and robust through overshoots.

    Returns a ``precondition_step(active_step, state)`` closure: it reads ``β`` from the step's shift
    schedule (a :class:`~aquaflux.solve.ConstantRelaxation` set by a
    :class:`~aquaflux.solve.DualTimeControl`) and re-factors the step's :class:`MonolithicFactorShiftPolicy`
    preconditioner in place at ``J(state) + β·d(state)``. Pass it to ``solve_coupled(precondition_step=…)``
    with a :func:`coupled_lu_continuation` step and a ``DualTimeControl``.

    **Forward-march use ONLY** -- the re-factor is an impure host mutation and must never be on a
    differentiated path (``solve_coupled`` guards this, raising under ``jax.grad``). The finishing solve and
    the adjoint keep the last frozen factorization, which is exact enough at the converged ``β → 0`` root.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler (supplies the Jacobian-vector product and the shift diagonal).
    stencil_reach : int
        The cell-graph distance the Jacobian's sparsity is probed to (coupled RANS reaches distance ``3``).

    Returns
    -------
    callable
        ``precondition_step(active_step, state) -> None``.
    """
    return _beta_tracking_refresh(coupled, stencil_reach)


def ilut_beta_tracking_refresh(
    coupled: CoupledRANS,
    *,
    stencil_reach: int = 3,
    fill_factor: float = 30.0,
    drop_tol: float = 1e-6,
    refresh_every: int = 5,
    beta_rel_change: float = 0.25,
) -> Callable[[ForwardStep, jnp.ndarray], None]:
    """A ``precondition_step`` for :func:`solve_coupled` that re-factors the ILUT at the current β.

    The ILUT counterpart of :func:`lu_beta_tracking_refresh`. As in a dual-time march the shift strength
    ``β`` ramps (e.g. 2.0 → 0.02), an ILUT frozen at one ``β`` mis-preconditions the operator actually
    solved -- mismatched during the low-``β`` cruise, and worst at a rung restart or an overshoot, where a
    stale factor can drive the Krylov solve to hundreds of cycles or stall the march. Re-factoring at the
    current ``(state, β)`` removes the mismatch, exactly as the LU hook does.

    Unlike the complete LU, the ILUT's threshold factorization is **expensive** (its fill is
    value-dependent, so there is no cheap symbolic-reuse refactor) and only **approximate** (a few Krylov
    iterations even when matched). Re-factoring every step is therefore not worth its cost. This hook
    instead **gates** the refactor (see :func:`_staleness_beta_gate`): it re-factors when ``β`` has moved
    by more than ``beta_rel_change`` since the last refresh (the shift-mismatch trigger that catches a
    rung restart or an overshoot before the line search can stall on it), or after ``refresh_every`` steps
    (the staleness cap for flow development at a near-constant ``β``). Between refreshes the approximate
    ILUT tolerates the residual mismatch at a few extra cycles.

    Returns a ``precondition_step(active_step, state)`` closure. Pass it to
    ``solve_coupled(precondition_step=…)`` with a :func:`coupled_ilut_continuation` step and a
    ``DualTimeControl`` (which sets a readable :class:`~aquaflux.solve.ConstantRelaxation` β).

    **Forward-march use ONLY** -- the re-factor is an impure host mutation and must never be on a
    differentiated path (``solve_coupled`` guards this, raising under ``jax.grad``). The finishing solve and
    the adjoint keep the last frozen factorization, which is exact enough at the converged ``β → 0`` root.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler (supplies the Jacobian-vector product and the shift diagonal).
    stencil_reach : int
        The cell-graph distance the Jacobian's sparsity is probed to (coupled RANS reaches distance ``3``).
    fill_factor, drop_tol : float
        The ILUT factorization controls, used for every in-place refresh (as in
        :func:`coupled_ilut_continuation`).
    refresh_every : int
        Force a refresh after this many steps without one (the staleness cap for flow development).
    beta_rel_change : float
        Refresh when ``β`` moves by more than this fraction of the last-refresh ``β`` (the shift-mismatch
        trigger). Smaller tracks ``β`` more tightly at more refresh cost.

    Returns
    -------
    callable
        ``precondition_step(active_step, state) -> None``.
    """
    return _beta_tracking_refresh(
        coupled,
        stencil_reach,
        gate=_staleness_beta_gate(refresh_every=refresh_every, beta_rel_change=beta_rel_change),
        refresh_kwargs={"fill_factor": fill_factor, "drop_tol": drop_tol},
    )


def amg_beta_tracking_refresh(
    coupled: CoupledRANS,
    *,
    stencil_reach: int = 3,
    materialize_every: int | None = None,
    materialize_drift: float | None = None,
    beta_rel_change: float | None = None,
    refresh_every: int = 8,
    beta_floor: float = 0.0,
    observer: Callable[[RefreshTiming], None] | None = None,
) -> Callable[[ForwardStep, jnp.ndarray], None]:
    """A ``precondition_step`` that rebuilds the AMG V-cycle at the current β, every step.

    The algebraic-multigrid counterpart of :func:`lu_beta_tracking_refresh` /
    :func:`ilut_beta_tracking_refresh`, and the preconditioner that makes a **dual-time march tractable in
    three dimensions**, where the complete LU's fill is out of memory and the incomplete-LU's factorization
    is prohibitively slow to build.

    A dual-time march ramps the pseudo-transient shift ``β`` down to develop the recirculation (e.g.
    0.5 → 0.02), and a V-cycle frozen at ``amg_beta`` degrades sharply as ``β`` leaves that value: the
    coarse operators and level smoother approximate ``J + amg_beta·d``, not the ``J + β·d`` actually solved,
    so the outer Krylov count explodes at low ``β`` (measured on the ``bfs3d`` coupled march: ~20 cycles per
    solve at ``β ≈ 0.5`` rising to ~250–285 at ``β ≈ 0.07``, with the per-step wall going from ~60 s to
    ~16–19 min). Rebuilding the V-cycle at the step's ``(state, β)`` restores the matched ~20-cycle solve.
    The rebuild (~tens of seconds: a graph-coloured Jacobian probe plus the smoothed-aggregation setup) is
    far cheaper than the hundreds of extra matvecs a stale V-cycle costs at low ``β``, each of which is a
    full Jacobian-vector product — so it re-factors **every step** (like the cheap complete-LU hook, not the
    gated incomplete-LU one).

    Reads ``β`` from the step's shift schedule (a :class:`~aquaflux.solve.ConstantRelaxation` set by a
    :class:`~aquaflux.solve.DualTimeControl`) and rebuilds the step's :class:`MonolithicFactorShiftPolicy`
    V-cycle in place at ``J(state) + β·d(state)``. Pass it to ``solve_coupled(precondition_step=…)`` (or
    :func:`solve_reynolds_continuation`'s ``point_setup``) with a :func:`coupled_amg_continuation` step and a
    ``DualTimeControl``.

    **Forward-march use ONLY** -- the rebuild is an impure host mutation and must never be on a differentiated
    path (``solve_coupled`` guards this, raising under ``jax.grad``). The finishing solve and the adjoint keep
    the last V-cycle, applied as the differentiable single-cycle transpose, exact at the converged ``β → 0``
    root; for a differentiated solve use the plain :func:`coupled_amg_continuation` with no ``precondition_step``.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler (supplies the Jacobian-vector product and the shift diagonal).
    stencil_reach : int
        The cell-graph distance the Jacobian's sparsity is probed to (coupled RANS reaches distance ``3``).
    materialize_every : int or None
        Enables the **β-diagonal split**. ``None`` (default) re-materializes the Jacobian on every refresh
        (the original behaviour). A value ``K > 1`` re-materializes only every ``K`` steps and, in between,
        tracks the moving shift with a cheap diagonal-only refresh
        (:meth:`~aquaflux.solve.MonolithicAmgPreconditioner.refresh_shift_in_place`) that reuses the frozen
        Jacobian -- since ``β`` and the per-cell shift ``d`` touch only the diagonal, this skips the
        coloured-probe materialization (the dominant refresh cost) while keeping the shift matched. It is the
        step-count arm of the materialize gate (:func:`_materialize_gate`): a full re-materialize is forced
        after ``K`` shift-only refreshes as a staleness cap. Prefer ``materialize_drift`` (a state-staleness
        trigger) as the primary control and keep ``materialize_every`` as a large safety cap.
    materialize_drift : float or None
        The **state-staleness** trigger for the full re-materialize (the drift arm of the materialize gate).
        Re-materializing the Jacobian is the dominant refresh cost, so — rather than a fixed step interval —
        it fires only when the frozen Jacobian has actually gone stale: when the eddy viscosity ``ν_t``
        (:func:`eddy_viscosity_drift`, what the operators are assembled from) has drifted by more than this
        fraction since the last materialize. In between, the shift is tracked by the cheap diagonal-only
        refresh. Reserve the expensive materialize for when it is needed; pair it with a large
        ``materialize_every`` as a backstop. ``None`` (default) disables the drift trigger.
    beta_rel_change : float or None
        The **β-mismatch** refresh trigger. ``None`` (default) refreshes every step. When set, the refresh
        is gated (:func:`_staleness_beta_gate`): it fires only when ``β`` has moved by more than this
        fraction of the ``β`` of the *last refresh* -- keying on the mismatch from the built ``β`` rather
        than the step-to-step change, so an oscillating control (``β`` swinging up and down around one
        value) does not trigger a refresh every step. It is the proactive complement to a reactive
        cycle-count retry: it re-matches the frozen V-cycle *before* a drifted ``β`` inflates the Krylov
        cost, and it composes with ``materialize_every`` (the gate decides *whether* to refresh; the split
        decides shift-only vs full).
    refresh_every : int
        The staleness-cap backstop when ``beta_rel_change`` is set: force a refresh after this many gated
        steps with no β-move (state development at a near-constant ``β``). Ignored when ``beta_rel_change``
        is ``None``.
    observer : callable, optional
        ``(kind, seconds) -> None``, called once per step with what this hook actually did --
        ``"full"`` (re-materialized the Jacobian and re-factored), ``"shift"`` (cheap shift-only
        refresh) or ``"none"`` (the gate declined; the standing factorization was reused) -- and how
        long it took. Forward-only instrumentation for a march being profiled: without it, which branch
        ran is invisible, and a study is left inferring preconditioner behaviour from wall-clock, which
        is exactly how a per-step refresh cost gets mistaken for a fixed overhead. ``None`` (default)
        elides the call.
    beta_floor : float
    beta_floor : float
        A lower bound on the shift strength the **preconditioner** is built at: the V-cycle is refreshed at
        ``max(β, beta_floor)`` while the march keeps solving at its own ``β``. As ``β`` falls the shift's
        diagonal dominance vanishes and the V-cycle degrades sharply -- but the operator needs the small
        ``β`` to make pseudo-transient progress, so the two are decoupled here. Because the preconditioner
        only changes the *path* of a solve and not its solution, this leaves the converged root and its
        adjoint untouched; the operator/preconditioner mismatch it introduces saturates at
        ``beta_floor * d`` instead of growing without bound. ``0.0`` (default) tracks ``β`` exactly.

    Returns
    -------
    callable
        ``precondition_step(active_step, state) -> None``.
    """
    gate = (
        None
        if beta_rel_change is None
        else _staleness_beta_gate(refresh_every=refresh_every, beta_rel_change=beta_rel_change)
    )
    return _beta_tracking_refresh(
        coupled,
        stencil_reach,
        gate=gate,
        materialize_every=materialize_every,
        materialize_drift=materialize_drift,
        beta_floor=beta_floor,
        observer=observer,
    )


def solve_coupled(
    coupled: CoupledRANS,
    flow: jnp.ndarray | None = None,
    k: jnp.ndarray | None = None,
    omega: jnp.ndarray | None = None,
    *,
    continuation: ForwardStep | None = None,
    reference_state: jnp.ndarray | None = None,
    method: str | None = "twolevel",
    max_steps: int = 60,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    refresh_trigger: RefreshTrigger | None = None,
    refresh_limit: int = 1,
    refresh_builder: Callable[[jnp.ndarray], ForwardStep] | None = None,
    step_control: StepControl | None = None,
    scaled_norm: bool = False,
    grow: int = 0,
    descent_backoff: int = 0,
    descent_test: bool = False,
    on_step: Callable[[StepReport], None] | None = None,
    on_checkpoint: Callable[[StepReport, jnp.ndarray], None] | None = None,
    precondition_step: Callable[[ForwardStep, jnp.ndarray], None] | None = None,
    retry_solver: lx.AbstractLinearSolver | None = None,
    retry_divergence_cap: float = float("inf"),
    retry_on_cycles: int | None = None,
    retry_beta_factor: float = 2.0,
    on_retry: Callable[[str, int, float], None] | None = None,
    retry_cycles_limit: int = 2,
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
    continuation : ForwardStep or None
        A pre-built continuation step (a :class:`~aquaflux.solve.PseudoTransientStep` or a
        :class:`~aquaflux.solve.DualTimeStep`). **Build it once outside ``jax.grad`` and pass it here when
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

        Prefer :class:`~aquaflux.solve.CoefficientDriftTrigger`, which fires on how far ``nu_t`` has
        moved since the current preconditioner was frozen. **That movement *is* the staleness:** the
        frozen k/omega transport operators are assembled from ``nu_t``, so once it has changed they no
        longer describe the system being solved. This solve supplies the measure itself
        (:func:`eddy_viscosity_drift`), re-based at every refresh so each segment reports drift from
        its own freeze state.

        :class:`~aquaflux.solve.CycleGrowthTrigger` infers staleness from the linear solve's
        restart-cycle count instead. That works, but the count also rises as the damping schedule
        ramps down -- on a separating flow, by more than staleness does -- so it needs a residual gate
        to separate the two, which the drift signal does not.
    refresh_limit : int
        The most refreshes one solve may perform (default ``1``). Each costs a preconditioner rebuild
        and a recompilation of the shifted solve, so this bounds that expense independently of how
        eager the trigger is; ``0`` disables refreshing entirely. Note the observed march and the
        finishing solve compile the step separately, so a refreshed solve pays one compilation more
        than an unrefreshed one over and above the per-refresh rebuild -- the price of leaving the
        convergence guard solely with the solve that produces the result.
    refresh_builder : callable, optional
        ``state -> ForwardStep``, a builder that (re)constructs the continuation at a given state. When
        supplied it replaces the internal block rebuild on both the initial build (if ``continuation``
        is ``None``) and every refresh, so a preconditioner solve_coupled does not know how to build --
        a :func:`coupled_ilut_continuation` materialized off the jit path -- can still refresh: the loop
        calls ``refresh_builder`` at each developed state and re-injects the initial-state progress
        measure (seam 4). This is what lets the monolithic ILUT re-materialize + re-factor as the flow
        develops (so its pseudo-timestep can grow past the wall where a frozen factorization goes stale),
        rather than being pinned at the reference state. Passing it also lifts the "explicit
        ``continuation`` cannot refresh" restriction, since the builder is how the refresh rebuilds.
    descent_backoff : int
        Lower the shift strength up to this many times, until the shifted correction actually descends
        in the residual measure, before the usual escalation ladder runs. Zero (the default) disables
        it. The shifted correction is not a descent direction by construction -- ``J delta = -R -
        beta D delta``, whose second term has no fixed sign and worsens with ``beta`` -- and past a
        critical shift strength every step length raises the measure, so the line search can only pick
        the least-harmful rung and the march stands still. Escalating there is the wrong direction;
        this backs off instead. Each backoff costs one shifted solve, which is why it is opt-in.
    descent_test : bool
        Reject a correction that does not descend, rather than judging the candidate's norm alone. With
        the backoff off this surfaces a non-descent direction instead of letting it pass as a step that
        quietly went nowhere.
    scaled_norm : bool
        **Rebuild** the default row-equilibrated measure (:class:`~aquaflux.solve.RowScaledNorm`) at the
        start of every outer iteration -- holding it fixed across that iteration's line search -- rather
        than freezing it once at the initial state as the default does. The scales (each row's own
        diagonal, each field's magnitude) then track the developing flow instead of the initial
        condition. Only the observed pre-march is affected; the finishing solve keeps the continuation's
        initial-state measure, so its absolute stopping target is computed there. Off by default: the
        frozen row-scaled measure is already the default steering norm (see ``residual_norm``), and the
        per-iteration rebuild is the finer, more expensive refinement.
    on_step : callable, optional
        Called with each :class:`~aquaflux.solve.StepReport` as the march produces it -- the seam for
        logging a long solve's progress and cost. The refresh trigger reads the same reports.
    on_checkpoint : callable, optional
        Called with ``(report, state)`` after each observed step, for saving intermediate states of a
        long march. Kept separate from ``on_step`` so the report history stays purely numeric and a
        refresh trigger remains replayable offline (see
        :func:`~aquaflux.solve.forward_march`). Note the *state* here is the solved-variable state,
        not the physical fields -- map it with :meth:`CoupledRANS.physical_fields`.

        Both callbacks see only the **observed** segments, not the finishing solve, whose march is
        traced and cannot call back into Python.

        **Supplying either one makes the march observed, which changes how ``max_steps`` is spent.**
        An unobserved solve runs one traced march with the whole ``max_steps`` budget. An observed one
        runs the eager pre-march to ``max_steps`` and, **if it has reached the stopping tolerance in its
        own measure, returns that state directly** -- it is forward-only (never differentiated, since the
        refresh/step control cannot run under a JAX transform), so its converged state needs no adjoint
        and there is nothing to gain from re-marching it through the traced finishing solve. The finishing
        solve runs only when the eager march stops *short* of the tolerance, as a fallback that owns the
        convergence guard, and it is given ``max_steps`` again. So a solve needing many contiguous steps
        can exhaust a tight budget in the pre-march and leave the finishing-solve fallback unable to reach
        a root -- which it reports by raising. Raise ``max_steps`` when instrumenting a solve that was
        already near its limit.

        **Why:** the frozen scalar preconditioners go stale as the flow separates. On a separated
        backward-facing-step state, re-freezing them cut the shifted solve from 30 to 13 outer Krylov
        cycles (~2.4x); the flow block does *not* go stale and is carried over untouched. The refresh
        costs one extra compilation of the shifted solve, which that saving repays within a step or two
        at mesh sizes where this matters. The win appears only once the flow separates -- refreshing at
        a pre-separation state buys nothing and can cost. :class:`~aquaflux.solve.CycleGrowthTrigger`
        therefore gates on the residual having fallen as well as on the cost having risen;
        :class:`~aquaflux.solve.CoefficientDriftTrigger` needs no such gate, because an undeveloped
        flow is one whose coefficients have not moved.

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

        **Each segment restarts the damping ramp, and that is load-bearing (binding).** The
        switched-evolution ramp is defined relative to where a segment began, so a segment handed a new
        state must measure its **own** reference residual; carrying the pre-refresh reference across --
        to keep the ramp "continuous", which looks like the more principled choice -- makes ``beta``
        mean something measured against a state the march has left, and the step is damped by a factor
        chosen for a residual that no longer applies.

        Two corrections to note, because earlier versions of this docstring stated both wrongly. First,
        a refresh **carries** the pseudo-transient shift diagonals rather than rebuilding them
        (rebuilding them at a developed state was measured to freeze the march), so the justification
        is *not* that a grown ``d`` must be paired with a fresh ``beta``. Second, the consequence of
        the segment-local reference is easy to miss and matters more than the rule itself: with
        refreshes every few steps the residual ratio never falls far below one, so **``beta`` stays
        pinned near ``beta0`` for the whole march** instead of ramping down -- if a different damping
        level is wanted it has to come from ``beta0``, not from expecting the ramp to find it.

        ``rtol`` means the same thing with and without a refresh: the finishing solve is given the
        **absolute** target ``atol + rtol * ||R0||`` measured at the initial state, so a refreshed solve
        stops at exactly the residual an unrefreshed one would, for any number of refreshes. (This is
        available precisely because the refresh path is forward-only, so ``||R0||`` is a concrete
        number rather than a traced one.)

        ``max_steps`` applies to **each** segment, so a refreshed solve may take up to
        ``(refresh_limit + 1) * max_steps`` march steps plus the finishing solve's own allowance. The
        budget is deliberately not split: either segment may legitimately need
        the full allowance, and halving it would fail a march that a single-stage solve completes.
    step_control : StepControl, optional
        Reshapes the forward step each observed iteration from the previous step's report (forward-only;
        it raises under ``jax.grad``, and so is consulted only on the observed march, never the
        differentiable single-stage solve). **A dual-time march** (``inner_steps > 1``, a
        :class:`~aquaflux.solve.DualTimeStep`) that is already observing — a ``refresh_trigger`` or an
        observer is set — **defaults to** :class:`~aquaflux.solve.DualTimeControl`, the Courant ramp that
        grows the pseudo-timestep while the inner loop stays comfortable (measured ~4× fewer outer steps
        to a developed recirculation on a cold-start pitzDaily ramp than the residual-keyed schedule).
        Pass an explicit control (e.g. :class:`~aquaflux.solve.ResidualRatioDualTimeControl`) to override,
        or pass one built with different knobs. The single-step march (``inner_steps = 1``, the default)
        gets no default control.
    precondition_step : callable, optional
        ``(active_step, state) -> None``, called before each observed step to refresh the step's frozen
        host preconditioner from the current state and shift strength β (forward-only; it raises under
        ``jax.grad`` like the other observed-march arguments). The use case is
        :func:`lu_beta_tracking_refresh`, which re-factors a complete-LU preconditioner at the current
        ``(state, β)`` each step so it stays the *exact* inverse of the operator being solved as the
        dual-time β ramps -- a frozen LU built at one β mis-preconditions the shifted operator once β
        moves away from it. Pair it with a :class:`~aquaflux.solve.DualTimeControl` (so β is a readable
        constant leaf) and a :func:`coupled_lu_continuation` step; the finishing solve and adjoint keep
        the last frozen factorization (exact enough near the converged β → 0 root).
    retry_solver : lineax.AbstractLinearSolver, optional
        A **tighter** linear solver used to redo an observed step that diverged (forward-only, same guard).
        With an *inexact* preconditioner (a threshold-ILU) the loose default Krylov solve can return a
        non-finite correction on the stiff operator an aggressive Courant overshoot produces, where the
        *exact* complete LU returns a finite one. On a diverged step the march redoes it from the same
        state at ``retry_solver`` -- the (β-tracked) factorization is already fresh, so only the Krylov
        tolerance is tightened -- which recovers the step while keeping the accepted trajectory, and pays
        the tighter solve only on the few trouble steps rather than every step. The exact-LU path never
        diverges, so it needs none; ``None`` (default) never retries. See ``retry_divergence_cap`` for the
        trigger and :func:`aquaflux.solve.forward_march` for the mechanism.
    retry_divergence_cap : float, optional
        A step is retried when its residual norm is non-finite **or**, if this cap is finite, exceeds
        ``retry_divergence_cap * reference``. Defaults to ``inf`` (only non-finiteness triggers), because
        the residual can legitimately rise during development (the ``β × travel`` identity), so a tight
        cap would false-fire on the load-bearing reachability descent.
    retry_on_cycles : int or None
        A **cycle-count** bailout (:func:`aquaflux.solve.forward_march`): a step whose solve exceeds this
        count is redone from the pre-step state with ``β`` escalated by ``retry_beta_factor`` -- more
        damping for the stiff low-``β`` operator, the hard-operator cause of a high count (staleness, the
        other, is pre-empted by a β-mismatch refresh). Needs a ``β``-carrying step control. ``None``
        (default) disables it.
    on_retry : callable, optional
        ``(reason, attempt, beta) -> None``, forwarded to
        :func:`~aquaflux.solve.forward_march`: called before a step is redone, with why. Forward-only
        reporting; a log without it shows a step's work twice and never says what triggered the redo.
    retry_beta_factor, retry_cycles_limit
        The ``β`` escalation factor per retry (default ``2``) and the maximum successive escalations for one
        step (default ``2``); see :func:`aquaflux.solve.forward_march`.
    **continuation_kwargs
        Forwarded to :func:`coupled_continuation` when building internally (schedule + preconditioner
        options). Notably ``inner_steps > 1`` selects the **dual-time** (backward-Euler) march
        (:class:`~aquaflux.solve.DualTimeStep`) — an inner Newton loop per outer timestep whose measured
        steady residual is the honest discrete time derivative rather than ``beta x travel``.
        ``inner_steps = 1`` (default) is the unchanged single-step continuation.

    Returns
    -------
    tuple of jnp.ndarray
        The converged ``(flow, k, omega)``.
    """
    refreshing = refresh_trigger is not None and refresh_limit > 0
    # The observed pre-march also runs when the caller only wants to *watch* the solve. Observability
    # must not require enabling a refresh: the reference march a refresh is calibrated against is by
    # definition unrefreshed, and it is the longest-running one, so it is the one that most needs to
    # report progress rather than sit silent for hours.
    observing = (
        refreshing
        or step_control is not None
        or on_step is not None
        or on_checkpoint is not None
        or precondition_step is not None
        or retry_solver is not None
        or retry_on_cycles is not None
    )
    if observing and _is_traced((coupled, flow, k, omega)):
        # The refresh re-derives the preconditioner from the mid-march state, which is a tracer when
        # differentiating; the refreshed preconditioner would capture it and escape the converged
        # solve's custom_vjp as a leaked tracer. There is no concrete-preconditioner path through a
        # refresh (it forbids an explicit `continuation`), so this cannot be worked around here --
        # raise with the fix rather than letting the leak surface as an opaque UnexpectedTracerError.
        raise ValueError(
            "refresh_trigger/step_control/on_step/on_checkpoint/precondition_step/retry_solver drive a "
            "forward-only eager march and cannot be used "
            "under jax.grad (or any JAX transform): the march steps in Python on concrete residual "
            "norms, and a mid-march preconditioner rebuild would capture the differentiation tracer. "
            "Drop them and differentiate the single-stage solve with a `continuation` "
            "built on concrete parameters outside jax.grad -- the adjoint is refresh-independent, so "
            "the gradient is identical."
        )
    if flow is None or k is None or omega is None:
        flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
    # `flow, k, omega` are the physical initial condition; map into the solved-variable space (the
    # identity for DirectScalars, log for LogScalars) so the Newton march iterates on the right unknown.
    state = coupled.state_from_physical(flow, k, omega)
    if continuation is None:
        if refresh_builder is not None:
            # A caller-supplied builder constructs the continuation from the state (e.g. an ILUT
            # continuation, materialized off the jit path); the refresh loop re-invokes it at each
            # developed state, so an off-jit preconditioner can refresh without solve_coupled knowing
            # how it is built.
            continuation = refresh_builder(state)
        else:
            reference = state if reference_state is None else reference_state
            continuation = coupled_continuation(
                coupled,
                reference,
                method=method,
                grow=grow,
                descent_backoff=descent_backoff,
                descent_test=descent_test,
                **continuation_kwargs,
            )
    elif refreshing and refresh_builder is None:
        raise ValueError(
            "refresh_trigger needs solve_coupled to (re)build the continuation, but an explicit "
            "`continuation` was supplied with no `refresh_builder`. Pass a `refresh_builder` "
            "(state -> ForwardStep) that rebuilds it at each developed state, or drop the explicit "
            "`continuation` so solve_coupled builds it, or stage the refresh yourself."
        )

    # A dual-time observed march with no caller control defaults to the Courant ramp (see the helper): it
    # grows the pseudo-timestep while the inner loop stays comfortable, carried across the refreshes
    # below, reaching a developed recirculation in far fewer outer steps than the residual-keyed schedule.
    # It is injected only where a control runs and never turns observation on, so the differentiable
    # single-stage solve (guarded above) is untouched.
    step_control = _default_dual_time_control(step_control, observing, continuation)

    stage_rtol, stage_atol = rtol, atol
    if observing:
        # Observed pre-march: step until the trigger judges the frozen preconditioner stale, re-freeze,
        # and continue from there. Each segment is an accelerator only -- it may stop short of a root,
        # and carries no convergence guard -- so the finishing solve below still produces the result.
        # `coupled.residual` is passed as a bound method (a pytree), not a lambda, so its arrays ride
        # as dynamic leaves and every step within a segment is a compilation-cache hit.
        # The global progress reference AND the residual measure that produces it must come from the
        # same state, held fixed across every segment: a `BlockScaledNorm` is self-normalising, so a
        # per-refresh rebuild would re-base it and make the convergence test unreachable (seam 4). Hold
        # the initial measure and re-inject it into every refreshed continuation.
        base_norm = continuation.norm()
        # The finishing solve's absolute target is always measured in the continuation's own measure,
        # whatever the march is steered by, because that solve keeps that measure (see below).
        base_reference = float(base_norm(coupled.residual(state)))
        norm_builder = None
        reference_norm = base_reference
        if scaled_norm:
            # Re-derive the row-equilibrated measure at whatever state each outer iteration starts
            # from -- reading `continuation` at call time, so a refreshed segment's diagonals are used.
            def norm_builder(at_state: jnp.ndarray) -> ResidualNorm:
                return coupled_scaled_norm(coupled, continuation.shift_policy, at_state)

            # The march's progress reference must be in the march's own measure, or `residual_ratio`
            # (and the switched-evolution shift that reads it) would divide two different scales.
            reference_norm = float(norm_builder(state)(coupled.residual(state)))
        # `refresh_limit` refreshes means `refresh_limit + 1` segments: the segment *after* the last
        # refresh must still be marched here, or the newly-refreshed preconditioner would only ever be
        # used by the finishing solve and its steps would go unobserved.
        control_state: object = None
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
                step_control=step_control,
                # Threaded across segments so a stateful control (the alpha-targeting shift climb)
                # continues past each refresh rather than restarting -- the same global-lifetime carry
                # as `reference_norm`, unlike the per-segment damping reference and drift measure.
                control_state=control_state,
                observer=on_step,
                checkpoint=on_checkpoint,
                # Re-based every segment, against the state this segment's preconditioner was frozen
                # at -- which is the segment's own starting state, since a refresh re-freezes at the
                # state it stopped on. Carrying one measure across segments would keep reporting the
                # drift a refresh had just absorbed, and re-fire immediately.
                drift_measure=eddy_viscosity_drift(coupled, jax.lax.stop_gradient(state)),
                norm_builder=norm_builder,
                precondition_step=precondition_step,
                retry_solver=retry_solver,
                retry_divergence_cap=retry_divergence_cap,
                retry_on_cycles=retry_on_cycles,
                retry_beta_factor=retry_beta_factor,
                on_retry=on_retry,
                retry_cycles_limit=retry_cycles_limit,
            )
            state = result.state
            control_state = result.control_state
            if not result.triggered or segment == refresh_limit:
                break
            if refresh_builder is not None:
                # Re-freeze at the developed state via the caller's builder (re-materializes an ILUT,
                # say), then re-inject the initial-state measure so the progress reference stays fixed
                # across the refresh (seam 4), just as the block path holds `residual_norm=base_norm`.
                continuation = eqx.tree_at(
                    lambda c: c.residual_norm,
                    refresh_builder(jax.lax.stop_gradient(state)),
                    base_norm,
                )
            else:
                # Re-freeze at the developed state: the k/omega AMGs are re-derived on their reused
                # coarsening and the shift's transport time scale is rebuilt, while the flow block and
                # the shift's coordinate factor are carried over (see `_coupled_shift_policy`).
                continuation = coupled_continuation(
                    coupled,
                    jax.lax.stop_gradient(state),
                    method=method,
                    reuse=continuation.shift_policy,
                    residual_norm=base_norm,  # keep the progress measure fixed (seam 4)
                    grow=grow,
                    descent_backoff=descent_backoff,
                    descent_test=descent_test,
                    **continuation_kwargs,
                )
        # The observed march is never differentiated -- the refresh, step control and per-step norm
        # rebuild cannot run under a JAX transform (guarded above) -- so its converged state needs no
        # adjoint, and there is no reason to re-march it through the traced finishing solve. When the
        # eager march has already reached its stopping tolerance, return that state directly. Judge it in
        # the SAME measure the march steered by (a per-step-rebuilt `RowScaledNorm` under `scaled_norm`),
        # which is what the march actually converged in; the frozen finishing-solve measure disagrees with
        # it at a developed state (the state0 row scales over-report the developed residual), so the traced
        # finishing solve -- which cannot refresh or carry the step control -- would chase an unreachable
        # target off the converged state and diverge on an aggressive low-shift path. The finishing solve
        # below then runs only as the not-converged fallback (and is the plain differentiable path's sole
        # march, unchanged).
        final_measure = norm_builder(state) if norm_builder is not None else base_norm
        if float(final_measure(coupled.residual(state))) <= atol + rtol * reference_norm:
            return coupled.physical_fields(state)
        # Hand the finishing solve the *absolute* target measured at the initial state, so a refreshed
        # solve stops exactly where an unrefreshed one would. A relative tolerance would be measured
        # against whatever residual the pre-march reached, silently tightening the solve by that factor
        # (and compounding with every extra refresh). This is only possible because the refresh path is
        # forward-only, so `reference_norm` is a concrete number rather than a traced one.
        stage_rtol, stage_atol = 0.0, atol + rtol * base_reference

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
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
    velocity_shift_parts: VelocityShiftParts | None = None,
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
    policy = _coupled_shift_policy(
        coupled,
        reference_state,
        method,
        None,
        shift_basis,
        velocity_shift_parts,
        **preconditioner_kwargs,
    )
    force, average = _coupled_constraint_vectors(coupled, flow_direction)
    bordered = _MassFlowBorderedPolicy(policy, force, average)
    residual_norm = (
        _mass_flow_residual_norm(coupled, reference_state) if block_scaled_norm else jnp.linalg.norm
    )
    return PseudoTransientStep(
        bordered,
        relaxation_schedule=SwitchedEvolutionRelaxation(
            beta0=beta0, exponent=exponent, beta_floor=beta_floor
        ),
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
