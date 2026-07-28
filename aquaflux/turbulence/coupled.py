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
    DualTimeStep,
    ForwardStep,
    ImplicitNewtonSolver,
    LocalCourantBasis,
    PseudoTransientStep,
    RefreshTrigger,
    ResidualNorm,
    RowScaledNorm,
    ShiftBasis,
    ShiftTerm,
    StepControl,
    StepReport,
    SwitchedEvolutionRelaxation,
    VelocityShiftParts,
    forward_march,
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

    @eqx.filter_jit
    def drift(state: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(coupled.eddy_viscosity(state) - reference) / scale

    return drift


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
        larger-restart GMRES suited to the stiff coupled system).
    block_scaled_norm : bool
        Which residual measure the march judges progress by (default ``False`` = the plain Euclidean
        norm). When ``True`` the march uses a :class:`~aquaflux.solve.BlockScaledNorm` over
        ``[flow, k, omega]`` (each block divided by its own initial magnitude), so the globalization
        weighs every field rather than the ``omega`` block that dominates the Euclidean norm. The
        block-scaled measure was found to *stall* the pitzDaily march (the per-block relative norm stops
        descending long before the fields converge), so it is available for experimentation but off by
        default; the Euclidean norm is what the solver uses.
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
        march's initial measure here on every refresh, so a self-normalising :class:`BlockScaledNorm`
        keeps its per-block reference magnitudes fixed at the state the global progress reference was
        measured against, rather than re-basing toward one at each developed refresh state (seam 4).
        ``None`` (a fresh, non-refresh build) constructs the measure from ``block_scaled_norm``.
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
        residual_norm = (
            _coupled_residual_norm(coupled, reference_state)
            if block_scaled_norm
            else jnp.linalg.norm
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
            **{"velocity": "convection", "schur_scaling": "msimpler", **preconditioner_kwargs},
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
    step_control: StepControl | None = None,
    scaled_norm: bool = False,
    grow: int = 0,
    descent_backoff: int = 0,
    descent_test: bool = False,
    on_step: Callable[[StepReport], None] | None = None,
    on_checkpoint: Callable[[StepReport, jnp.ndarray], None] | None = None,
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
        Steer the observed march by the **row-equilibrated** measure
        (:class:`~aquaflux.solve.RowScaledNorm`) instead of the continuation's own, rebuilding it at the
        start of every outer iteration and holding it fixed across that iteration's line search. Each
        row is divided by its own diagonal and each block by its field's magnitude, so the measure
        reports a *fractional change* per equation rather than a raw magnitude -- which the plain
        Euclidean norm on this system cannot, being ~100 % the ``omega`` block and so blind to
        flow-block progress. **Experimental and off by default.** Two consequences worth knowing before
        reading a run: the march's stopping test and switched-evolution shift are then in fractional
        units (a target like ``1e-6`` is meaningful; the old absolute magnitude is not comparable), and
        the **finishing solve keeps the continuation's measure regardless** -- it passes its norm
        through a non-differentiated argument slot that requires a hashable object, which a measure
        holding per-row arrays is not, so its absolute target is computed in that measure and not this
        one.
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
        An unobserved solve runs one traced march with the whole ``max_steps`` budget; an observed one
        runs the eager pre-march to ``max_steps`` and then gives the finishing solve ``max_steps``
        again. That is more budget in total, but it is *split*, so a solve needing many contiguous
        steps can exhaust a tight budget in the pre-march and leave the finishing solve unable to
        reach a root -- which it reports by raising, rather than returning a non-root. Raise
        ``max_steps`` when instrumenting a solve that was already near its limit.

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
    **continuation_kwargs
        Forwarded to :func:`coupled_continuation` when building internally (schedule + preconditioner
        options). Notably ``inner_steps > 1`` selects the **dual-time** (backward-Euler) march
        (:class:`~aquaflux.solve.DualTimeStep`) — an inner Newton loop per outer timestep whose measured
        steady residual is the honest discrete time derivative rather than ``beta x travel``; pair it
        with ``step_control=DualTimeControl()`` to ramp the pseudo-timestep by a Courant rule. Both are
        opt-in accelerators; ``inner_steps = 1`` (default) is the unchanged single-step continuation.

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
        refreshing or step_control is not None or on_step is not None or on_checkpoint is not None
    )
    if observing and _is_traced((coupled, flow, k, omega)):
        # The refresh re-derives the preconditioner from the mid-march state, which is a tracer when
        # differentiating; the refreshed preconditioner would capture it and escape the converged
        # solve's custom_vjp as a leaked tracer. There is no concrete-preconditioner path through a
        # refresh (it forbids an explicit `continuation`), so this cannot be worked around here --
        # raise with the fix rather than letting the leak surface as an opaque UnexpectedTracerError.
        raise ValueError(
            "refresh_trigger/step_control/on_step/on_checkpoint drive a forward-only eager march and cannot "
            "be used "
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
    elif refreshing:
        raise ValueError(
            "refresh_trigger needs solve_coupled to build the continuation (it re-freezes the "
            "preconditioner part way through), but an explicit `continuation` was supplied. Pass the "
            "schedule via **continuation_kwargs instead, or stage the refresh yourself with "
            "coupled_continuation(..., reuse=<the old policy>)."
        )

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
            )
            state = result.state
            control_state = result.control_state
            if not result.triggered or segment == refresh_limit:
                break
            # Re-freeze at the developed state: the k/omega AMGs are re-derived on their reused
            # coarsening and the shift's transport time scale is rebuilt, while the flow block and the
            # shift's coordinate factor are carried over (measured -- see `_coupled_shift_policy`).
            continuation = coupled_continuation(
                coupled,
                jax.lax.stop_gradient(state),
                method=method,
                reuse=continuation.shift_policy,
                residual_norm=base_norm,  # keep the progress measure fixed at the initial state (seam 4)
                grow=grow,
                descent_backoff=descent_backoff,
                descent_test=descent_test,
                **continuation_kwargs,
            )
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
