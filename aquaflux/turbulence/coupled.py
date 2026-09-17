"""The monolithic coupled RANS residual ``R(u, p, k, omega)``.

The segregated driver (:mod:`~aquaflux.turbulence.driver`) is a *forward* convergence device: it
freezes the eddy viscosity for the flow solve and the flow for the turbulence solve, Picard-iterating
to the fixed point. This module assembles the same physics as **one residual over the full unknown**
``[u..., p, k, omega]`` with nothing frozen -- the eddy viscosity ``nu_t(k, omega, grad u)``, the mean
strain ``S(u)``, and the Rhie--Chow mass flux ``mdot(u, p)`` are live functions of the state, so a
single Newton solve sees the exact cross-block coupling.

Why monolithic, when the segregated loop already converges? Two reasons: a monolithic Newton reaches
**quadratic** coupled convergence the Picard loop
cannot, and -- handed to :class:`~aquaflux.solve.RootSolver` -- it yields the **exact
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
import inspect
import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from aquaflux.discretization import DifferenceRow, FixationRow, LogRatioRow
from aquaflux.flow import BlockPreconditioner, ConvectionTwoLevel, frozen_momentum_diagonal_parts

# The mass-flow-constraint primitives (a body force that is a solve unknown enforcing a bulk velocity)
# are shared with the flow-block solve `aquaflux.flow.bulk_velocity_flow_solve`: the border column/row,
# the Schur (constraint) preconditioner, and the body-force setter. Reused here rather than re-deriving
# the Schur elimination, which one careful place keeps consistent.
from aquaflux.flow.mean_velocity import (
    _bordered_preconditioner,
    _constraint_vectors,
    _with_body_force,
)
from aquaflux.schemes import narrow_gradient_sweeps
from aquaflux.solve import (
    DEFAULT_GLOBALIZATION,
    NO_REFRESH,
    NO_RETRIES,
    BlockScaledNorm,
    CellFields,
    ColumnProbePlan,
    Convergence,
    DualTimeLoop,
    Euclidean,
    FieldGroups,
    FieldLayout,
    FieldSplitAmgPreconditioner,
    GlobalDofs,
    Globalization,
    LocalCourantBasis,
    MaterializedJacobianPreconditioner,
    MonolithicAmgPreconditioner,
    MonolithicLuPreconditioner,
    NewtonStrategy,
    ProbeGather,
    PseudoTransientStep,
    RefreshPolicy,
    RefreshTiming,
    ResidualHomotopy,
    RetryPolicy,
    RootSolver,
    RowScaled,
    RowScaledNorm,
    ShiftBasis,
    ShiftPolicy,
    ShiftTerm,
    StepControl,
    StepReport,
    SubLayout,
    TransposedPreconditioner,
    VelocityShiftParts,
    assembler_residual,
    block_stencil_colouring,
    block_stencil_gather_map,
    column_probe_plan,
    default_dual_time_control,
    newton_march,
    positive_block_limit,
    positive_block_projection,
    refuse_a_transform_the_march_cannot_run_in,
    relative_residual_gmres,
    root_adjoint,
    stop_array_gradients,
)

from .initialization import hybrid_initialize, wall_consistent_omega
from .march_settings import LinearSolveSettings, ShiftSettings
from .preconditioner import (
    ScalarBlock,
    ScalarTransportPreconditioner,
    ScaledScalarPreconditioner,
    UnpreconditionedScalars,
)
from .preconditioner_spec import (
    BlockDiagonal,
    CompleteLu,
    FieldSplit,
    MaterializedJacobian,
    MonolithicVCycle,
)
from .sources import production_and_limit

# The default pseudo-time shift basis (full operator diagonal = uniform under-relaxation), held as a
# module singleton so it is not reconstructed in each function's argument defaults.
_DEFAULT_SHIFT_BASIS = LocalCourantBasis()

if TYPE_CHECKING:
    from aquaflux.flow import MomentumContinuity

    from .transport import SSTClosureFields, SSTTurbulence


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
    :func:`coupled_step`).
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

    Positivity is not structural here -- it is carried by the pseudo-transient shift, the divergence
    guard, and (for ``k``) the fraction-to-the-boundary step limiter :func:`positive_k_limit` -- so a
    full Newton step can transiently violate ``omega > 0`` on a stiff high-Reynolds case. The
    historical default; use :class:`LogScalars` where that matters.
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


def coupled_rans_layout(flow: FieldLayout) -> FieldLayout:
    """The flat coupled state layout ``[flow..., k, omega]``.

    The flow block is the momentum assembler's own ``[vel_0..vel_{dim-1}, pressure]`` layout
    (:meth:`~aquaflux.flow.MomentumContinuity.layout`) **nested verbatim**, so the flow sub-vector is
    handed to :class:`~aquaflux.flow.MomentumContinuity` unchanged and its widths are never restated
    here; ``k`` and ``omega`` follow as two ``n_cells``-long scalar blocks.

    Parameters
    ----------
    flow : FieldLayout
        The momentum-continuity state's layout.

    Returns
    -------
    FieldLayout
        Blocks ``"flow"`` (the nested layout, read out as the flat flow sub-vector), ``"k"`` and
        ``"omega"``, of total length ``(dim + 3) * n_cells``.
    """
    return FieldLayout(
        flow.n_cells,
        (SubLayout("flow", flow), CellFields("k", 1), CellFields("omega", 1)),
    )


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
        The flow assembler; it owns the molecular viscosity and receives the closure's kinematic
        ``nu_t`` each evaluation, forming ``mu_eff = mu + rho nu_t`` itself (the molecular viscosity in
        its property model is left intact, so re-applying ``nu_t`` never accumulates).
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

        Raises
        ------
        ValueError
            If ``turbulence.density`` does not match ``momentum``'s (its ``PropertyModel``'s
            ``"density"``, evaluated per cell). Both values now originate from a
            :class:`~aquaflux.properties.PropertyModel` -- ``SSTTurbulence.build``'s
            ``properties`` argument and ``MomentumContinuity.build``'s -- but they are still two
            separate models with nothing forcing them to describe the same fluid, so nothing else
            catches a mismatch: the k/omega volume flux (``mdot / density``) would then be wrong
            by that ratio in every SST consumer -- both residuals, both AMGs, both shift diagonals
            -- with no other symptom, since the flow block is unaffected and solves fine.
        """
        flow_density = momentum.density
        if not bool(jnp.all(jnp.isclose(flow_density, turbulence.density))):
            raise ValueError(
                f"SSTTurbulence.density ({turbulence.density}) does not match the flow "
                f"assembler's density (range [{float(jnp.min(flow_density))}, "
                f"{float(jnp.max(flow_density))}]). The two come from separate PropertyModels -- "
                "SSTTurbulence.build(properties=...) and MomentumContinuity.build(properties=...) "
                "-- and nothing else checks that they agree: if they don't, the k/omega volume "
                "flux (mdot / density) is wrong by that ratio in every SST consumer, silently. "
                "Pass a PropertyModel with the same density to both."
            )
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
    def layout(self) -> FieldLayout:
        """The coupled state layout ``[flow..., k, omega]`` for this system."""
        return coupled_rans_layout(self.momentum.layout)

    def pack_state(self, flow: jnp.ndarray, k: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
        """Assemble a coupled state from a flow state and the two turbulence fields."""
        return self.layout.pack(flow, k, omega)

    def effective_momentum(
        self, flow: jnp.ndarray, k: jnp.ndarray, omega: jnp.ndarray
    ) -> tuple[SSTClosureFields, MomentumContinuity]:
        """The SST closure at ``(flow, k, omega)`` and the flow assembler re-viscosified by it.

        The closure carries ``nu_t`` and the mean strain, so it is built first and ``nu_t`` taken from
        it -- :meth:`eddy_viscosity` would otherwise recompute the strain the closure already formed.
        Both halves are returned because the two consumers want different ones: the coupled residual
        needs the closure for its ``k`` / ``omega`` equations as well as the assembler, while a caller
        that only wants the momentum block at the current ``mu_eff = mu + rho nu_t`` drops it.

        Parameters
        ----------
        flow : jnp.ndarray
            The flat flow state ``[vel_0..vel_{dim-1}, pressure]``, shape ``((dim + 1) n_cells,)``.
        k, omega : jnp.ndarray
            The **physical** turbulence fields, shape ``(n_cells,)`` each (not the solved unknowns --
            recover them with :meth:`physical_fields`).

        Returns
        -------
        tuple of (SSTClosureFields, MomentumContinuity)
            The closure at these fields, and the flow assembler carrying its eddy viscosity.
        """
        return _effective_momentum(self.momentum, self.turbulence, flow, k, omega)

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
        closure, momentum = self.effective_momentum(flow, k, omega)

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
        self,
        flow: jnp.ndarray,
        k_solved: jnp.ndarray | None = None,
        omega_solved: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # The defaults are the protocol's, so this satisfies the same call signature its sibling does --
        # but unlike a frozen-viscosity source this one cannot work without the turbulence context, so
        # omitting it is an error rather than something to ignore. Saying so is the whole point of
        # honouring the arity: the alternative is a `TypeError` from deep inside a shift policy.
        if k_solved is None or omega_solved is None:
            raise TypeError(
                "LiveViscosityVelocityParts forms the shift at the CURRENT effective viscosity, so it "
                "needs the k and omega blocks; the caller passed flow alone. A shift policy that has no "
                "turbulence state to give (the flow-only MomentumShiftPolicy) wants "
                "FrozenViscosityVelocityParts instead."
            )
        k = self.k_transform.to_physical(k_solved)
        omega = self.omega_transform.to_physical(omega_solved)
        _closure, live = _effective_momentum(self.momentum, self.turbulence, flow, k, omega)
        velocity, _pressure = live.unpack(flow)
        return live.momentum_matrix_diagonal_parts(velocity)


def turbulence_residual_norm(layout: FieldLayout, residual: jnp.ndarray) -> jnp.ndarray:
    """The Euclidean norm of a coupled residual's ``k`` and ``omega`` rows.

    How far the **closure** is from its own equilibrium, as distinct from how far the flow is. The two
    move on different schedules -- the transported scalars settle long before the recirculation does --
    so a rule that wants to react to the turbulence block specifically must not read the whole-state
    norm, which the mean flow dominates.

    Deliberately the **unscaled** norm of the solved variables. The march's own measure re-equilibrates
    its row scales every outer step, so a ratio of two scaled norms mixes the state's progress with a
    change of measure; the solved coordinates (``log omega`` under
    :class:`~aquaflux.turbulence.LogScalars`) already tame the fields' dynamic range, which is what the
    scaling would otherwise be for.

    Parameters
    ----------
    layout : FieldLayout
        The coupled state's layout, used to slice the scalar rows out of the full residual.
    residual : jnp.ndarray
        A full coupled residual ``R(phi)``, shape ``((dim + 3) n_cells,)``.

    Returns
    -------
    jnp.ndarray
        A scalar: ``||[R_k, R_omega]||_2``.
    """
    _, k, omega = layout.unpack(residual)
    return jnp.sqrt(jnp.sum(k * k) + jnp.sum(omega * omega))


class TurbulenceDamping(eqx.Module):
    """How much harder the ``k``/``omega`` rows are damped than the flow rows, at one iterate.

    The two blocks are hard for different reasons -- the momentum block's convective nonlinearity, which
    a viscosity continuation weakens, against the closure's stiff sign-indefinite sources under a
    positivity constraint, which a continuation barely touches and a diagonal shift addresses squarely.
    A single shift for the whole state is therefore set by whichever block is more fragile.

    This is a strategy rather than a number because the right ratio is **not constant over a march**:
    measured, more damping is monotonically better early and monotonically worse late (see
    :class:`ResidualTaperedDamping`). Implementations differ in what they read to decide.
    """

    def factor(self, relaxation: jnp.ndarray, residual: jnp.ndarray | None) -> jnp.ndarray:
        """The multiplier on the closure's shift strength, ``>= 1``.

        ⚠️ **It multiplies the shift STRENGTH, never the diagonal.** The two differ where it matters:
        ``beta`` reaches the step already clamped at ``beta_min``, so a factor folded into the diagonal
        multiplies that floor and the closure never stops being damped -- and the diagonal is also the
        row scale of the march's own residual measure
        (:func:`coupled_scaled_norm`), so damping it silently divides the ``k``/``omega`` rows of the
        quantity the march is steered and judged by. Returning a factor here keeps both honest.

        Parameters
        ----------
        relaxation : jnp.ndarray
            The shift strength ``beta`` this attempt will actually use, a scalar, already clamped.
        residual : jnp.ndarray or None
            The full coupled residual at this iterate when the caller has it, else ``None`` (the first
            shift a march builds, before any residual has been formed).
        """
        raise NotImplementedError

    def rebased(self, coupled: CoupledRANS, state: jnp.ndarray) -> TurbulenceDamping:
        """This strategy with any reference re-derived at ``state``; the configuration carried.

        ⚠️ **A reference is PHYSICS and belongs to the problem as it currently stands, so a refresh
        rebuilds it** -- the same split this module already makes between ``k_shift_transport`` (rebuilt
        at the developed state) and ``k_jacobian_scale`` (carried). Carrying a reference instead is not a
        harmless conservatism: under a continuation it freezes the taper against the *easiest* problem
        the march ever sees, and the ratio then never falls (see :class:`ResidualTaperedDamping`).
        """
        del coupled, state
        return self


class ConstantDamping(TurbulenceDamping):
    """One ratio for the whole march; ``1.0`` is a single shift for the whole state.

    Attributes
    ----------
    ratio : jnp.ndarray
        The multiplier, ``>= 1``. **Stored as a JAX array, not as the Python float it is usually
        constructed from**, so that swapping it on a live march is a compilation-cache hit rather than
        a full recompile of the coupled solve: ``equinox.filter_jit`` partitions on "is this an array",
        so a Python float would ride on the *static* side and be compared by value. That is what makes a
        ratio the caller varies between continuation stations affordable at all -- the same trap a
        ``float`` molecular viscosity sprang on the Reynolds ramp, where every rung recompiled.
    """

    ratio: jnp.ndarray = eqx.field(converter=jnp.asarray, default=1.0)

    def __check_init__(self) -> None:
        # Concrete at construction -- a damping is configured outside the march, never inside a trace.
        if float(self.ratio) < 1.0:
            raise ValueError(f"ratio must be >= 1 (damping never accelerates), got {self.ratio}")

    def factor(self, relaxation: jnp.ndarray, residual: jnp.ndarray | None) -> jnp.ndarray:
        """``ratio``, whatever the shift or the state."""
        del relaxation, residual
        return self.ratio

    def rebased(self, coupled: CoupledRANS, state: jnp.ndarray) -> TurbulenceDamping:
        """Unchanged -- a constant has no reference to re-derive."""
        del coupled, state
        return self


class ResidualTaperedDamping(TurbulenceDamping):
    """Damp the closure hard while it is far from its own equilibrium, and release as it approaches.

    **Why a taper rather than a constant.** Measured on a backward-facing-step sibling (pitzDaily, 16
    momentum-only viscosity stations, one run per point), the ratio that is best early is the worst
    late. Cycles to reach step 5 fall monotonically with the ratio -- 17 / 12 / 12 / 11 at 1 / 2 / 5 /
    10 -- while the totals are 227 / **191** / 206 / **305**: at 10 the march stalls twice in the late
    phase, its residual moving *backwards* (9.503e-04 -> 9.548e-04) at 15-17 restart cycles a step, and
    finishes worse than no damping at all. Early the closure is far from equilibrium with a flow that
    barely exists and heavy damping lets it settle; once it is near equilibrium the same damping only
    makes it lag the mean flow, and the coupled residual cannot fall.

    ⚠️ **The taper keys on the turbulence block's own residual, NOT on the shift.** Tying it to
    ``beta / beta_0`` fails twice over: ``beta`` is *adaptive*, so a retry ladder raises it on a bad step
    and the damping would spike exactly when the march is already struggling; and ``beta`` **floors** at
    ``beta_min`` early -- on the measured run at step 13 of 41 -- after which the ratio would be frozen
    for the whole remaining march, including every step where the damping is doing the damage.
    :func:`turbulence_residual_norm` has no floor until convergence and is the quantity the taper is
    actually about.

    The factor is ``1 + (initial - 1) * min(1, |R_turb| / reference) ** exponent``, so it starts at
    ``initial`` and reaches ``1`` as the closure converges.

    ⚠️ **Ending at ``1`` is a choice about the PATH, not a correctness requirement** -- the shift term is
    ``beta d (phi - phi_n)``, which is zero at ``phi == phi_n`` whatever multiplies it, so *any* factor
    dissolves at the root and leaves the converged solution and its adjoint untouched. That is what
    licenses :class:`ConstantDamping` at all. A taper is therefore free to end above ``1``.

    A residual that **rises** -- as it does when a continuation station moves the problem -- raises the
    factor again, which is the same response
    :meth:`~aquaflux.solve.ShiftStrengthControl.redamp` makes on entering a station, reached here from a
    different direction.

    Attributes
    ----------
    initial : float
        The ratio at the reference state, ``>= 1``.
    reference : jnp.ndarray
        ``|R_turb|`` at the state the march starts from, the denominator of the taper. Supply it from
        the same seed the march opens on, or the taper is measured against a state it never visits.
    layout : FieldLayout
        The coupled layout, for slicing the scalar rows.
    exponent : float
        Shapes the release: ``1`` is linear in the residual ratio, ``> 1`` holds the damping longer.
    """

    initial: float
    reference: jnp.ndarray
    layout: FieldLayout
    exponent: float = 1.0

    def __check_init__(self) -> None:
        if self.initial < 1.0:
            raise ValueError(
                f"initial must be >= 1 (damping never accelerates), got {self.initial}"
            )

    def factor(self, relaxation: jnp.ndarray, residual: jnp.ndarray | None) -> jnp.ndarray:
        """The tapered multiplier; ``initial`` when no residual is available yet.

        The march's very first shift is built before any residual exists, and that is the step the
        damping is most for -- so ``None`` opens at ``initial`` rather than at ``1``.
        """
        del relaxation
        if residual is None:
            return jnp.asarray(float(self.initial))
        share = jnp.clip(turbulence_residual_norm(self.layout, residual) / self.reference, 0.0, 1.0)
        return 1.0 + (self.initial - 1.0) * share**self.exponent

    def rebased(self, coupled: CoupledRANS, state: jnp.ndarray) -> TurbulenceDamping:
        """The same taper measured against ``state``'s own turbulence residual.

        A continuation march makes the problem **harder** as it walks, so a reference frozen at the
        anchor is the easiest problem the march ever meets and the ratio never falls below one. Measured
        on pitzDaily, walking only the viscosity from the anchor to the target at a fixed state *raises*
        ``|R_turb|`` by 10 % (2.664e+02 -> 2.930e+02) -- so a frozen reference pinned the factor at
        ``initial`` for every step of a 41-step march and the taper was **bit-identical to the constant
        it was meant to replace**, 305 restart cycles either way.
        """
        return ResidualTaperedDamping(
            initial=self.initial,
            reference=turbulence_residual_norm(coupled.layout, coupled.residual(state)),
            layout=coupled.layout,
            exponent=self.exponent,
        )


class BetaTaperedDamping(TurbulenceDamping):
    """Damp the closure hard while the pseudo-timestep is small, and release as the shift relaxes.

    **Why a taper rather than a constant.** Measured on a backward-facing-step sibling (pitzDaily, 16
    momentum-only viscosity stations, one run per point), the ratio that is best early is the worst
    late. Cycles to reach step 5 fall monotonically with the ratio -- 17 / 12 / 12 / 11 at 1 / 2 / 5 /
    10 -- while the totals are 227 / **191** / 206 / **305**: at 10 the march stalls twice in the late
    phase, its residual moving *backwards* (9.503e-04 -> 9.548e-04) at 15-17 restart cycles a step, and
    finishes worse than no damping at all. Early the closure is far from equilibrium with a flow that
    barely exists and heavy damping lets it settle; once it is near equilibrium the same damping only
    makes it lag the mean flow, and the coupled residual cannot fall.

    **Why ``beta`` is the key.** It is the quantity the march already measures its own development by:
    a Courant-ramped control grows the pseudo-timestep (lowers ``beta``) exactly as the flow settles, so
    ``beta`` falling *is* the transition from the phase that wants damping to the phase that does not.
    Two objections that look fatal and are not:

    * ``beta`` **floors at** ``beta_min``, early -- step 13 of 41 on the measured run. What it freezes
      at is ``factor = 1``, i.e. the taper switched off, which is the end state the taper is walking
      toward anyway. A signal that saturates *at the value it is meant to reach* is not inert.
    * ``beta`` is **adaptive**, so the retry ladder raises it on a bad step and the damping rises with
      it. That is the response a bad step wants -- a step that had to be re-damped is one whose closure
      was moving too fast -- so the coupling runs the right way round, not the wrong one.

    The share is taken in ``log beta`` because a Courant control moves ``beta`` geometrically (one
    ``/grow`` per comfortable step), so a share linear in ``log beta`` releases linearly in **outer
    steps** -- an even release over the descent rather than one spent in its first few steps.

    ⚠️ **The endpoints must be the step control's own.** ``beta_start`` and ``beta_min`` say where the
    taper opens and where it reaches exactly ``1``; if they disagree with the control's, the taper still
    runs, but it reaches ``1`` somewhere the march never visits (too low: the closure stays damped for
    the whole march; too high: the damping is spent before it is needed). Nothing detects that -- wire
    all three from the same constants.

    ⚠️ **Ending at ``1`` is a choice about the PATH, not a correctness requirement.** The shift term is
    ``beta d (phi - phi_n)``, zero at ``phi == phi_n`` whatever multiplies it, so any factor dissolves at
    the root -- which is what licenses :class:`ConstantDamping` in the first place. Read ``beta_min`` as
    "where the taper stops releasing", and note that a taper ending above ``1`` is equally legitimate and
    is not expressible here.

    Attributes
    ----------
    initial : float
        The ratio at ``beta_start``, ``>= 1``.
    beta_start : float
        The shift strength the march opens at -- where the factor is ``initial``.
    beta_min : float
        The control's floor -- where the factor reaches ``1``. A ``beta`` at or below it is undamped.
    exponent : float
        Shapes the release: ``1`` is linear in ``log beta`` (and so in outer steps), ``> 1`` holds the
        damping longer.
    """

    initial: float
    beta_start: float
    beta_min: float
    exponent: float = 1.0

    def __check_init__(self) -> None:
        if self.initial < 1.0:
            raise ValueError(
                f"initial must be >= 1 (damping never accelerates), got {self.initial}"
            )
        if not 0.0 < self.beta_min < self.beta_start:
            raise ValueError(
                "the taper needs a span to run over: 0 < beta_min < beta_start, got "
                f"beta_min={self.beta_min}, beta_start={self.beta_start}"
            )

    def factor(self, relaxation: jnp.ndarray, residual: jnp.ndarray | None) -> jnp.ndarray:
        """``1 + (initial - 1) * share ** exponent``, with ``share`` the fraction of the descent left.

        ``share = log(beta / beta_min) / log(beta_start / beta_min)``, clipped to ``[0, 1]`` so a
        ``beta`` escalated above ``beta_start`` damps at ``initial`` rather than running away, and one
        at the floor damps at exactly ``1``.
        """
        del residual
        span = math.log(self.beta_start / self.beta_min)
        share = jnp.clip(jnp.log(relaxation / self.beta_min) / span, 0.0, 1.0)
        return 1.0 + (self.initial - 1.0) * share**self.exponent


def as_damping(damping: TurbulenceDamping | float) -> TurbulenceDamping:
    """A plain number means :class:`ConstantDamping`; a strategy passes through."""
    return damping if isinstance(damping, TurbulenceDamping) else ConstantDamping(float(damping))


class CoupledShiftPolicy(eqx.Module):
    """The block :class:`~aquaflux.solve.continuation.ShiftPolicy` for the coupled Newton solve.

    Composes the three subsystems' pseudo-transient choices block-diagonally: the momentum block's
    ``a_P`` velocity shift + block-SIMPLE preconditioner (:class:`~aquaflux.flow.MomentumShiftPolicy`),
    and the k and omega transport-operator shift diagonals + convection-diffusion algebraic-multigrid
    (AMG) preconditioners
    (:class:`~aquaflux.turbulence.continuation.ScalarShiftPolicy`). The full-state shift diagonal is
    ``[a_P on u, 0 on p, d_k on k, d_omega on omega]`` and the preconditioner is the block-diagonal
    matvec gluing the flow preconditioner to the two scalar AMGs.

    The AMG hierarchies and the (numpy-assembled) scalar shift diagonals are **frozen at a reference
    state** (built off-jit by a :class:`~aquaflux.turbulence.BlockDiagonal` session) and carried here as data, exactly as
    :func:`~aquaflux.flow.reused_flow_solve` freezes the flow preconditioner: a pseudo-transient shift
    and its preconditioner are transient devices that vanish at the fixed point, so freezing their
    coefficients at a representative state costs only Krylov iterations, never correctness. The
    velocity ``a_P`` is the one piece recomputed live per iterate (it is a cheap jittable read of the
    momentum diagonal), so the velocity damping still tracks the developing convection.

    Attributes
    ----------
    layout : FieldLayout
        The coupled state layout, for packing the block-diagonal shift and preconditioner.
    momentum : MomentumContinuity
        The flow assembler at the reference effective viscosity. It is the **shift's** dependency and
        the only one: the velocity damping is built from this assembler's frozen momentum diagonal, and
        the flat flow sub-vector is packed with it.
    flow_preconditioner : BlockPreconditioner or None
        The block-SIMPLE preconditioner built at that same assembler -- the velocity and pressure-Schur
        solves :meth:`shift_term`'s ``make_preconditioner`` composes. ``None`` when this policy is a
        **shift source only**, which is what a monolithically preconditioned step wants: such a step
        supplies its own inverse and never asks for this one, and building it anyway meant two multigrid
        hierarchies per build that were never applied *and* whose value-dependent coarsening made the
        policy's array shapes move with the molecular viscosity -- recompiling the whole coupled solve at
        every Reynolds-continuation rung. Asking a shift-only policy for a preconditioner raises.
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
        the flow **assembler's** frozen momentum diagonal
        (:func:`~aquaflux.flow.frozen_momentum_diagonal_parts`) -- live in velocity, frozen in
        viscosity -- which is the historical behaviour and needs no preconditioner, so a shift-only
        policy works on this path. Pass :class:`LiveViscosityVelocityParts` to form them at the current
        effective viscosity instead. **Carried across a refresh**, since it is a configuration choice
        rather than frozen state.
    shift_basis : ShiftBasis
        How the live velocity shift diagonal is built from the momentum diagonal's convective/dissipative
        parts (the scalar shift diagonals are pre-combined at build time). The default
        :class:`~aquaflux.solve.LocalCourantBasis` (weight ``1``) is ``a_P`` -- uniform under-relaxation,
        unchanged from the historical shift; a convective basis gives a local convective time step.
    turbulence_damping : TurbulenceDamping
        How much harder the ``k``/``omega`` rows are damped than the flow rows: the shift **strength**
        on those rows is multiplied by this, so they run at an effective ``turbulence_damping * beta``
        while the velocity rows keep ``beta``. ``ConstantDamping(1.0)`` (the default) is a single shift
        for the whole state. It rides on the strength rather than on the base diagonal
        (:meth:`TurbulenceDamping.factor` says why), so it reaches the shift and the preconditioner
        fitted to it, and reaches neither the shift's floor nor the march's residual measure.

        **The two blocks are hard for different reasons, and one scalar shift has to satisfy both.** The
        momentum block's difficulty is the convective nonlinearity, which a viscosity continuation
        weakens directly. The closure's is its stiff, sign-indefinite source terms under a positivity
        constraint, which a continuation barely touches and a diagonal shift addresses squarely -- a
        larger shift on those rows is local implicit Euler on the reaction terms. Sharing one ``beta``
        means it is set by whichever block is more fragile.

        ⚠️ **The shift dissolves at the root** (the term is ``beta * d * (phi - phi_n)`` and vanishes when
        ``phi == phi_n``), so this changes the *path* a march takes and neither the converged solution
        nor its adjoint -- the same property that licenses ``beta`` itself.

        ⚠️ **The step control still adapts a single ``beta`` against a GLOBAL line search**, so it cannot
        see which block is asking for the caution -- which is why the ratio is a strategy the caller
        chooses (a constant, or a taper keyed on something the march already measures) rather than
        something the control adapts. A per-block adaptive shift is a different design and needs the
        control to grow a per-block signal first.
    """

    layout: FieldLayout
    momentum: MomentumContinuity
    k_shift_transport: jnp.ndarray
    k_jacobian_scale: jnp.ndarray
    omega_shift_transport: jnp.ndarray
    omega_jacobian_scale: jnp.ndarray
    flow_preconditioner: BlockPreconditioner | None = None
    k_preconditioner: ScalarTransportPreconditioner | None = None
    omega_preconditioner: ScalarTransportPreconditioner | None = None
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS
    velocity_shift_parts: VelocityShiftParts | None = None
    turbulence_damping: TurbulenceDamping = ConstantDamping(1.0)

    def shift_term(self, phi: jnp.ndarray, residual: jnp.ndarray | None = None) -> ShiftTerm:
        """The block-diagonal full-state shift and the ``beta -> M`` composed preconditioner at ``phi``.

        Parameters
        ----------
        phi : jnp.ndarray
            The flat coupled state ``[flow..., k, omega]``, shape ``((dim + 3) n_cells,)``.
        residual : jnp.ndarray or None
            ``R(phi)`` when the caller has it. Only :attr:`turbulence_damping` reads it, and only a
            state-dependent one; ``None`` is always safe.
        """
        flow, k, omega = self.layout.unpack(phi)
        n_cells = self.layout.n_cells
        # The shift's velocity buckets come from the ASSEMBLER's frozen momentum diagonal, which is all
        # they need -- so a shift-only policy carries no block preconditioner at all.
        convective, dissipative = frozen_momentum_diagonal_parts(self.momentum, flow)
        # The SHIFT's buckets are a separate concern with a different lifetime (see
        # `VelocityShiftParts`), so their source is injected; `None` uses the assembler's own, which is
        # the historical behaviour and keeps the default path bit-identical.
        if self.velocity_shift_parts is not None:
            convective, dissipative = self.velocity_shift_parts.parts(flow, k, omega)
        d_vel = self.shift_basis.local_diagonal(convective, dissipative)

        # Full-state base shift: d_vel on every velocity component, 0 on pressure, the frozen scalar
        # transport diagonals on k and omega.
        flow_diagonal = self.momentum.pack(
            jnp.broadcast_to(d_vel[:, None], (n_cells, self.momentum.mesh.dim)), jnp.zeros(n_cells)
        )
        # The scalar shift diagonal is transport-time-scale * coordinate factor; kept as two fields so a
        # refresh rebuilds the transport half and carries the coordinate half (see `_coupled_shift_policy`).
        diagonal = self.layout.pack(
            flow_diagonal,
            jax.lax.stop_gradient(self.k_shift_transport * self.k_jacobian_scale),
            jax.lax.stop_gradient(self.omega_shift_transport * self.omega_jacobian_scale),
        )

        def row_relaxation(relaxation: jnp.ndarray) -> jnp.ndarray:
            """``turbulence_damping`` on the two scalar blocks, ``1`` on the flow rows.

            The closure's rows then run at an effective ``damping * beta`` while the velocity rows keep
            ``beta``. The shift vanishes at the root either way, so this moves the path and not the
            answer. It rides here rather than in ``diagonal`` for the two reasons
            :meth:`TurbulenceDamping.factor` gives: the diagonal multiplies the CLAMPED ``beta``, so a
            factor there survives the floor the march switches damping off at; and the same diagonal is
            the row scale of the march's residual measure, which damping must not touch.
            """
            gamma = self.turbulence_damping.factor(relaxation, residual)
            scalar = jnp.full((n_cells,), gamma)
            return self.layout.pack(jnp.ones_like(flow_diagonal), scalar, scalar)

        def make_preconditioner(relaxation: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
            block = self.flow_preconditioner
            if block is None:
                raise ValueError(
                    "this CoupledShiftPolicy is a shift source only (flow_preconditioner=None), so it "
                    "cannot compose a block preconditioner. A monolithically preconditioned step "
                    "supplies its own inverse and asks only for the shift diagonal; build the policy "
                    "with a flow block if the composed one is wanted."
                )
            # `a_p` is a PRECONDITIONER quantity -- it is what the velocity block inverts, so it comes
            # from the block itself or the two disagree.
            convective_ap, dissipative_ap = block.frozen_momentum_diagonal_parts(flow)
            a_p = convective_ap + dissipative_ap
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

        return ShiftTerm(diagonal, make_preconditioner, row_relaxation)

    def adjoint_factory(self) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """The ``state -> M`` factory for the adjoint transpose solve (the composition at ``beta = 0``).

        At the converged state the pseudo-transient shift vanishes, so the adjoint preconditions the
        unshifted coupled Jacobian with the block-diagonal composition at ``a_P`` -- the same frozen
        flow and scalar preconditioners, transposed by the implicit solver.
        """
        return lambda state: self.shift_term(state).make_preconditioner(jnp.asarray(0.0))


def _effective_momentum(
    momentum: MomentumContinuity,
    turbulence: SSTTurbulence,
    flow: jnp.ndarray,
    k: jnp.ndarray,
    omega: jnp.ndarray,
) -> tuple[SSTClosureFields, MomentumContinuity]:
    """The closure at ``(flow, k, omega)`` and ``momentum`` re-viscosified by it.

    Free of :class:`CoupledRANS` so the consumer that holds the pair of assemblers without holding the
    coupled one -- :class:`LiveViscosityVelocityParts`, which forms the shift at the current effective
    viscosity -- reaches the same three lines as :meth:`CoupledRANS.effective_momentum` rather than
    repeating them. ``k`` and ``omega`` are the **physical** fields.
    """
    closure = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega)
    return closure, momentum.with_eddy_viscosity(
        closure.nu_t, turbulence.wall_face_eddy_viscosity(k)
    )


def wall_consistent_state(coupled: CoupledRANS, state: jnp.ndarray) -> jnp.ndarray:
    """``state`` with its near-wall ``omega`` rows re-imposed at **this** assembler's viscosity.

    The coupled counterpart of :func:`~aquaflux.turbulence.wall_consistent_omega`. Those rows are a
    value fixation whose value the model prescribes from the molecular viscosity, so a state inherited
    from a solve at a different viscosity holds a number this residual says is wrong -- by the viscosity
    ratio exactly. The flow and ``k`` are returned untouched.

    **Forward-only seed device.** The residual fixes these cells at this same value whatever the seed
    held, so the root and its adjoint cannot move; and it is a no-op when the viscosity has not changed.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled assembler at the viscosity the march will run at.
    state : jnp.ndarray
        The flat coupled state ``[vel..., pressure, k, omega]``, shape ``((dim + 3) n_cells,)``.

    Returns
    -------
    jnp.ndarray
        The state with its wall-cell ``omega`` replaced, shape unchanged.
    """
    flow, k, omega = coupled.physical_fields(state)
    return coupled.state_from_physical(flow, k, wall_consistent_omega(coupled.turbulence, k, omega))


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


# The shifted forward solve for the coupled march. Each preconditioner family gets its own restart
# regime, and they all share one stopping *measure* -- the two are separate decisions and only the first
# is a property of the preconditioner.
#
# THE MEASURE (shared: the linear solve stops in whatever progress measure the march hands the step). Each
# pseudo-transient step is an inexact Newton step, so the linear solve only has to resolve the
# correction to the accuracy the globalized march actually uses; the converged root and its adjoint are
# fixed by the nonlinear stop and the vanishing shift, not by the linear tolerance. What that argument
# does not say is *which* norm measures "resolved", and the coupled residual makes the choice
# load-bearing:
#  * lineax's stock componentwise `rtol`/`atol` test is not a relative stop on this system at all. It
#    applies the tolerance per row under a max-norm, and the near-wall omega rows start satisfied --
#    their right-hand side is ~0 -- so their per-row scale collapses onto the absolute `atol` floor and a
#    handful of them hold the whole solve to ~1e-10, about nine orders past the tolerance requested.
#  * A global 2-norm relative stop is immune to those rows, but the coupled residual is ~100% omega
#    (whose residual sits orders above the flow's), so it halts once omega is resolved while the
#    flow-dominated part of the Newton step is still coarse -- measured ~116% velocity error, i.e.
#    effectively blind to the block the march is actually trying to develop.
# So the stop is the march's OWN progress measure -- by default the row-scaled `coupled_scaled_norm`, which
# weighs every field comparably, at a *loose* tolerance: every block is seen and resolved loosely,
# rather than one block resolved and the rest unseen. Calibrated on the developed backward-facing step:
# at `rtol = 0.3` the velocity correction is resolved to ~25% for ~1.5x the plain-2-norm cycle count;
# `0.1` fully resolves it at ~2.25x. This is a property of the coupled residual, so it belongs to every
# family: the calibration was taken on the multigrid one, that being where the measure was first built,
# but nothing in it is about multigrid.
#
# ⚠️ The block and complete-LU families stopped on a plain 2-norm at `1e-2` until this was unified, so
# their `max_restarts` caps below were sized against that arrangement and have NOT been re-measured
# against the row-scaled stop, which costs ~1.5x the cycles where it was measured. They are left at
# their previous values rather than adjusted to an invented number; neither flagship validation case
# runs those builders, so nothing on record is affected, and a cap that binds shows up as a truncated
# solve rather than as a wrong answer.
#
# Steering and judging by one definition is the point: the default solver takes its norm from the step at
# every step (`relative_residual_gmres(norm=None)`), so it stops in the very measure the march reports and
# accepts steps in -- including after the march has rebuilt that measure at a new state.
#
# THE RESTART LENGTH (per family). A restarted GMRES tests its stop only at each restart boundary, so
# the subspace should match how many vectors the preconditioner actually needs.
class _LinearSolveRegime(NamedTuple):
    """One preconditioner family's default Krylov regime for the shifted forward solve.

    Attributes
    ----------
    rtol : float
        Relative tolerance, **in the progress measure the march hands the step** (the solve's
        :class:`~aquaflux.solve.Convergence` measure). The same number is a different tightness under a
        different measure, so each regime's value is set for the measure its path defaults to.
    restart : int
        Arnoldi restart length.
    max_restarts : int
        Restart-cycle cap -- the only bound on a single running solve, since ``cycle_budget`` and the
        march's abort threshold are tested *between* inner iterations.
    """

    rtol: float
    restart: int
    max_restarts: int


#: The block-diagonal preconditioner (flow block-SIMPLE + the two scalar AMGs). The coupled turbulent
#: saddle is stiff enough that a 40-vector restart -- the shared lineax default -- discards too much
#: Arnoldi history and needs hundreds of restart cycles, while a 120-vector subspace reaches the same
#: solution in far fewer.
_BLOCK_LINEAR_SOLVE = _LinearSolveRegime(rtol=0.3, restart=120, max_restarts=15)

#: A monolithic complete-LU factorization. It factors the whole coupled saddle exactly, so the
#: preconditioned operator's spectrum collapses to a single point at the state and shift it was factored
#: at -- the Krylov solve stops within a handful of vectors there, and the large subspace the
#: block-triangular preconditioner needs is pure waste: with `restart = 120` it would build ~120
#: matrix-vector products (each paying the factorization's triangular back-solve) before it could stop.
#: `max_restarts` is kept generous so a transiently harder (e.g. drifted-reference) solve still completes
#: before the next refactor.
_FACTORIZATION_LINEAR_SOLVE = _LinearSolveRegime(rtol=0.3, restart=10, max_restarts=40)

#: A monolithic multigrid V-cycle. Restart 15 is the measured sweet spot for a one-V-cycle
#: preconditioner: enough Arnoldi history for its convergence while checking the stop often enough not to
#: overshoot the loose tolerance deep into the next cycle (a larger restart costs ~2x the expensive host
#: V-cycle applies for the same trajectory).
#:
#: ⚠️ Two constraints bind `max_restarts` against the march's retry threshold, and both bite silently.
#: It is in raw ``lineax`` restarts, which carry a fixed ``+2`` per solve, while
#: ``retry.abort_above_cycles`` is in corrected cycles (:func:`~aquaflux.solve.restart_cycles`), so a
#: corrected cap of ``c`` is ``max_restarts = c + 2``. And the corrected count must stay **strictly
#: above** ``retry.abort_above_cycles``: the march's test is ``max_inner_cycles >
#: retry.abort_above_cycles``, so a cap landing exactly on the threshold does not trip the redo, and the
#: step accepts the truncated, non-converged direction instead of re-running it on a fresh
#: preconditioner.
_VCYCLE_LINEAR_SOLVE = _LinearSolveRegime(rtol=0.3, restart=15, max_restarts=60)

#: The mass-flow-constrained (bordered) system. Its restart regime is the block-diagonal one -- it wraps
#: that preconditioner -- but its **tolerance is a Euclidean one**, because the bordered march has no
#: row-scaled measure to stop in: the row-equilibrated measure has no constraint-aware form yet, and
#: applying it would scale the border row by a diagonal it does not have. So this is a genuinely
#: different path rather than a surface that drifted, and the tolerance differs because the *measure*
#: does. Re-unify it with :data:`_BLOCK_LINEAR_SOLVE` the day the measure gains a constraint-aware form.
_CONSTRAINED_LINEAR_SOLVE = _LinearSolveRegime(rtol=1e-2, restart=120, max_restarts=15)


# How many coloured tangents share one vmapped jvp pass when materializing the AMG Jacobian. Larger
# amortizes dispatch over more probes; the coloured probes run in ceil(n_probes / this) fused passes
# instead of an n_probes-call Python loop.
#
# Measured on the 3D backward-facing step (399 probes; 47.2M structural nonzeros in the fixed sparsity
# pattern, of which ~38.7-39.0M are live at any one state), wall / peak against the batch:
#
#     batch      1      2      4      8     16     32
#     wall    11.7    8.4    6.7    5.9    5.8    6.5   s
#     peak     380    383    388    399    419    460   MB
#
# Two things decide the eight. The curve has an interior optimum and **turns** -- 32 is slower than 16 --
# so this is not "as large as memory allows"; and the memory it costs is ~2.5 MB per unit of batch, which
# is nothing against the matrix being built, so the peak is not what picks the value. Eight is where the
# processor time bottoms; sixteen is a hair faster in wall and slower in processor time, which on a shared
# machine is the less trustworthy of the two.
#
# An earlier default of four came from a measurement -- "16 vs 4: ~2.2 GB against ~0.7 GB" -- taken when
# the seed set and the response array dominated the peak, and NEITHER of those ever scaled with the batch.
# Both are built a chunk at a time now, so that trade no longer exists.
_PROBE_BATCH_SIZE = 8

# Backtracking rungs for the shifted step. The full coupled Newton step from the hybrid initial
# condition overshoots violently (the residual blows up many orders of magnitude), so the step length
# is scaled back along {1, 1/2, ..., 1/2**N} until it descends -- recovering a residual-reducing step
# from the one expensive shifted solve, instead of escalating beta (a full re-solve, which changes the
# direction and, measured, does not descend on this case). Ten rungs reach 1/1024, well past the
# ~1/4 the stiff first steps need. It is every coupled builder's base for an unset
# `Globalization.line_search` (see `_coupled_step`), so a caller who sets one -- including 0 -- keeps it.
# It is the one setting on which the coupled march differs from the flow-only and scalar ones.
_COUPLED_LINE_SEARCH = 10


def _coupled_block_scales(coupled: CoupledRANS, reference_state: jnp.ndarray) -> tuple[float, ...]:
    """The per-field reference residual magnitudes ``(‖R_flow‖, ‖R_k‖, ‖R_omega‖)`` at
    ``reference_state``, each floored positive so it can divide a block norm."""
    parts = coupled.layout.unpack(coupled.residual(reference_state))
    return tuple(max(float(jnp.linalg.norm(part)), 1e-30) for part in parts)


def _coupled_residual_norm(coupled: CoupledRANS, reference_state: jnp.ndarray) -> BlockScaledNorm:
    """The block-scaled residual norm over ``[flow, k, omega]``, the measure :class:`~aquaflux.solve.BlockScaled` names.

    Each field's residual is divided by its own initial magnitude before the norm is formed, so the
    switched-evolution-relaxation ramp, the line search, and the outer stopping test all judge every
    field rather than the ``omega`` block that dominates the plain Euclidean norm (``omega`` is
    O(1e5) here, ``k`` O(1e-3)): with the plain norm a step that collapses ``k`` barely moves ‖R‖ and
    is accepted. This is the coarser of the two field-aware measures -- one scale per block -- and is
    the alternative to the default :func:`coupled_scaled_norm`, which additionally equilibrates each row
    by its own diagonal.
    """
    return BlockScaledNorm(coupled.layout.sizes, _coupled_block_scales(coupled, reference_state))


def _mass_flow_layout(coupled: CoupledRANS) -> FieldLayout:
    """The coupled layout bordered with the mass-flow constraint's scalar multiplier.

    The constrained march carries the augmented state ``[flow..., k, omega, beta]``. ``beta`` is one
    degree of freedom belonging to no cell, so it is an ordinary extra block of the layout rather than
    a length appended by hand wherever the augmented system's shape is wanted.
    """
    return coupled.layout.appended(GlobalDofs("mass_flow", 1))


def _mass_flow_residual_norm(coupled: CoupledRANS, reference_state: jnp.ndarray) -> BlockScaledNorm:
    """The :func:`_coupled_residual_norm` measure extended with the mass-flow constraint dof.

    The bordered march carries the augmented residual ``[R_flow, R_k, R_omega, ⟨U⟩ − target]``; the
    trailing scalar constraint is a bulk-velocity (velocity-magnitude) equation, so it shares the
    flow block's reference scale.
    """
    s_flow, s_k, s_omega = _coupled_block_scales(coupled, reference_state)
    return BlockScaledNorm(_mass_flow_layout(coupled).sizes, (s_flow, s_k, s_omega, s_flow))


class _CoupledMeasures(eqx.Module):
    """What the coupled RANS residual can be measured in: row-scaled and block-scaled as well as Euclidean.

    Attributes
    ----------
    coupled : CoupledRANS
        The assembler whose residual is measured, with its gradients stopped.
    """

    coupled: CoupledRANS

    def row_scaled(self, step: NewtonStrategy, state: jnp.ndarray) -> RowScaledNorm:
        # The row diagonals are the step's own shift base, so a refreshed step's are read.
        return coupled_scaled_norm(self.coupled, step.shift_policy, state)

    def block_scaled(self, state: jnp.ndarray) -> BlockScaledNorm:
        return _coupled_residual_norm(self.coupled, state)


class _MassFlowMeasures(eqx.Module):
    """What the mass-flow-constrained residual can be measured in: block-scaled as well as Euclidean.

    There is no row-scaled measure here: the border row holding the constraint has no diagonal to
    equilibrate by, and borrowing one would misreport it.

    Attributes
    ----------
    coupled : CoupledRANS
        The assembler whose bordered residual is measured, with its gradients stopped.
    """

    coupled: CoupledRANS

    def row_scaled(self, step: NewtonStrategy, state: jnp.ndarray) -> RowScaledNorm:
        del step, state
        raise TypeError(
            "RowScaled() has no form for the mass-flow-constrained solve: the constraint's border row "
            "has no diagonal to equilibrate by. Use Euclidean() (the default) or BlockScaled()."
        )

    def block_scaled(self, state: jnp.ndarray) -> BlockScaledNorm:
        # The bordered state carries the multiplier last; the block scales are the coupled blocks'.
        return _mass_flow_residual_norm(self.coupled, state[:-1])


def positive_k_limit(coupled: CoupledRANS, tau: float = 0.99, floor: float = 0.0):
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
    floor : float
        Absolute room in ``k`` granted to every cell, so a cell whose ``k`` is numerically zero stops
        setting the step length for all of them (see :func:`~aquaflux.solve.positive_block_limit` for
        the collapse this prevents and how to choose it). ``0`` (default) is the plain rule. Give it as
        a fraction of a reference ``k`` for the case -- an absolute constant does not transfer between
        cases with different velocity scales.

    Returns
    -------
    callable or None
        ``(phi, delta) -> alpha_max`` for a directly-solved ``k``; ``None`` otherwise.
    """
    if not isinstance(coupled.k_transform, DirectScalars):
        return None
    k_block = coupled.layout.slice_of("k")
    return positive_block_limit(k_block.start, k_block.stop, tau, floor)


def positive_k_projection(coupled: CoupledRANS, tau: float = 0.99, floor: float = 0.0):
    """The step projection keeping ``k`` positive per cell, or ``None`` when the transform already does.

    The per-cell counterpart of :func:`positive_k_limit`, and the reason to prefer it on this system:
    the cap is a minimum over cells, so a single cell whose ``k`` is numerically zero sets the step
    length for all of them. That is not hypothetical here -- on a 3D backward-facing step the binding
    cell is the stagnant corner where the step face, the floor and a side wall meet, which has no shear
    and therefore no turbulent-energy production, so its ``k`` decays with nothing to arrest it. Its
    correction then throttles the whole march while every other cell is healthy.

    Clipping each cell's own correction leaves that cell to decay alone and lets the rest take a full
    step. See :func:`~aquaflux.solve.positive_block_projection` for the rule, why a floor on the cap
    postpones rather than removes the collapse, and why this stays a single search direction.

    Returns ``None`` when ``k`` is solved in a form that is positive by construction (a log variable),
    since a projection there would clip nothing.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled case, for its block layout.
    tau : float
        Fraction of the distance to the boundary a cell may take.
    floor : float
        Absolute room in ``k`` granted to every cell. ``0`` (default) is the plain rule and is normally
        what this wants -- unlike the cap, a projection has no reason to exempt a dead cell, because a
        dead cell no longer costs anything.

    Returns
    -------
    callable or None
        ``(phi, delta) -> delta'`` for a directly-solved ``k``; ``None`` otherwise.
    """
    if not isinstance(coupled.k_transform, DirectScalars):
        return None
    k_block = coupled.layout.slice_of("k")
    return positive_block_projection(k_block.start, k_block.stop, tau, floor)


def _k_positivity_guards(
    coupled: CoupledRANS, positivity_floor: float, positivity_projection: bool
) -> tuple[Callable[..., jnp.ndarray] | None, Callable[..., jnp.ndarray] | None]:
    """The ``(step_limit, step_projection)`` pair every coupled continuation builder installs.

    One tail for the four builders (#365): each independently built :func:`positive_k_limit` and
    :func:`positive_k_projection` from the same two arguments -- the drifting-sibling shape the
    shared-tail rule warns about, and here it was hiding a real trap rather than just duplication.

    **``positivity_floor`` is provably inert whenever ``positivity_projection`` is true (the
    default).** The Newton step applies ``step_projection`` to ``delta`` first, then hands the
    result to ``step_limit`` (:func:`_coupled_step`). The projection clips every cell to within
    ``tau`` of its own boundary, so after it runs, the limiter's room is `>= 1/tau` for every entry
    and it reports ``alpha_max = 1`` regardless of what floor it was built with. So a caller who
    raises ``positivity_floor`` under the default projection gets neither an error nor the
    protection they asked for -- the limiter that floor feeds never binds. Raised here instead of
    silently doing nothing, the same choice already made for ``preconditioner`` / ``reference_state`` /
    ``**strategy_kwargs`` in :func:`_continuation_source`.

    A non-zero floor is not refused when ``k`` is solved in log form (:func:`positive_k_limit`
    returns ``None`` there): that inertness is a different, already-documented story (the transform
    already keeps ``k`` positive by construction), not the projection masking a real guard.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled case, forwarded to :func:`positive_k_limit` / :func:`positive_k_projection`.
    positivity_floor : float
        Forwarded to the limiter only. See :func:`positive_k_projection`'s own docstring for why
        the projection does not want a floor: a dead cell decays alone under it, so there is nothing
        to exempt it from.
    positivity_projection : bool
        Whether the per-cell projection replaces the global cap as the active guard.

    Returns
    -------
    tuple of (callable or None, callable or None)
        ``(step_limit, step_projection)``, ready for ``_coupled_step`` / ``_monolithic_factor_step``.

    Raises
    ------
    ValueError
        If ``positivity_floor`` is non-zero, ``k`` is solved directly (so the limiter it feeds is a
        real guard, not ``None``), and ``positivity_projection`` is true.
    """
    step_limit = positive_k_limit(coupled, floor=positivity_floor)
    if positivity_floor and positivity_projection and step_limit is not None:
        raise ValueError(
            f"positivity_floor={positivity_floor!r} has no effect here: positivity_projection=True "
            "(the default) clips every k correction to within `tau` of its own boundary before the "
            "limiter runs, so the limiter always reports alpha_max=1 regardless of its floor. Pass "
            "positivity_projection=False to make the floor take effect, or leave positivity_floor=0.0 "
            "to keep the (per-cell) projection."
        )
    step_projection = positive_k_projection(coupled) if positivity_projection else None
    return step_limit, step_projection


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
      transport diagonal on ``k`` and ``omega``) and so cannot drift from it. ⚠️ **The base diagonal,
      not the shift** -- the strength ``beta`` and any per-block multiplier on it
      (:attr:`CoupledShiftPolicy.turbulence_damping`) are solver settings, and folding one into the row
      scale would divide that block's reported residual by it: the march would be steered, stopped and
      compared on a measure that moves with its own damping, so two damping settings could not be
      compared at all. Two rows are not covered by the diagonal and are supplied here:

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
    shift_policy : ShiftPolicy
        Any policy whose base shift diagonal supplies the per-row diagonals -- the block
        :class:`CoupledShiftPolicy`, or a :class:`MonolithicFactorShiftPolicy` wrapping one.
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
    n, dim = layout.n_cells, coupled.momentum.mesh.dim
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


def _coupled_shift_policy(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    scalar: ScalarBlock,
    reuse: CoupledShiftPolicy | None = None,
    shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
    velocity_shift_parts: VelocityShiftParts | None = None,
    turbulence_damping: TurbulenceDamping | float = 1.0,
    build_flow_block: bool = True,
    **flow_block_options: object,
) -> CoupledShiftPolicy:
    """Build the block-diagonal :class:`CoupledShiftPolicy` frozen at ``reference_state``.

    The preconditioner-freezing half of the block-diagonal session's step, split out so the mass-flow
    constraint (:func:`mass_flow_coupled_continuation`) can border the *same* policy rather than
    re-derive it.

    ``reuse`` **refreshes** an existing policy at a new (more developed) ``reference_state`` rather than
    building one from scratch. The scalar k/omega AMGs are re-derived on their reused coarsening
    (:func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner`'s ``reuse=`` -- the
    coarse space stays valid while its operators are re-derived, so the shifted solve stops paying for
    a coarsening fitted to a flow that has since separated), and the shift's **transport time scale is
    rebuilt** at the new state. **Carried over from ``reuse`` untouched**: the flow block
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
    # ⚠️ These three lines are `_effective_momentum` MINUS the wall-face eddy viscosity, and the
    # omission is not yet justified -- it is recorded here rather than silently unified, because routing
    # this through the shared helper would change the frozen operators on a wall-function mesh and that
    # is a measurement, not a refactor. The case for it being inert: everything built from this
    # assembler here is a *frozen* quantity taken at `boundary_corrected=False`, the plain all-faces
    # form, which never consults the per-face boundary viscosity the wall leaf overrides. The case
    # against: nothing checks that, and the leaf also reaches the velocity block's operator. Settle it
    # by measuring, not by pattern-matching the helper.
    momentum = coupled.momentum.with_eddy_viscosity(closure.nu_t)
    # The coupled flow block uses the convection-aware velocity AMG, not the viscous multilevel default:
    # a RANS case is high-Reynolds, and the Peclet-blind viscous block produces a poor
    # momentum-block direction once the flow separates (the shifted Newton direction it returns drifts
    # away from the true one on a developed separated field, and the march stalls). The convection
    # block's convective linearization stays valid frozen at the cold initial state (the reference), so
    # no per-sweep refresh is needed. Overridable through a `BlockDiagonal` spec's fields, which arrive
    # here as `flow_block_options` and are pinned to `BlockPreconditioner.build`'s keywords, so no name
    # it does not take can reach it. `build_flow_block=False`
    # leaves it out entirely: a monolithically preconditioned step reads this policy for its shift
    # diagonal and supplies its own inverse, so the block built here would never be applied.
    #
    # The pressure Schur is left at BlockPreconditioner's own default (a_P-scaled SIMPLE), not the
    # mass-matrix scaling this used to hardcode. That scaling was chosen believing it necessary for a
    # convection-dominated coupled solve; it is not, at the scale this block-diagonal preconditioner is
    # actually used at -- swapping it for the default reaches the identical converged fixed point on
    # every fixture this path is exercised by (residual and fields agree to machine precision). Where a
    # Schur choice genuinely matters (a real, large, separated case), this whole block-diagonal
    # preconditioner is dominated by the field-split / monolithic-AMG preconditioners the flagship
    # validation cases use instead (`MaterializedJacobian`), so tuning the Schur here buys nothing
    # a real case would ever see. It remains available (`BlockDiagonal(schur_scaling="msimple")`, or
    # directly through `BlockPreconditioner`) for the one regime it is not
    # dominated in: a standalone, flow-only, convection-dominated solve, where the plain SIMPLE Schur's
    # inner solve can stall outright.
    block = (
        None
        if not build_flow_block
        else reuse.flow_preconditioner  # measured: re-freezing the flow block does not help
        if reuse is not None
        else BlockPreconditioner.build(
            momentum,
            **{
                "velocity": ConvectionTwoLevel(),
                # Aggregate the velocity/Schur AMGs along strong connections. A no-op on a low-aspect-
                # ratio mesh (this pitzDaily case), but the fix that keeps the V-cycle contracting once
                # the near-wall cells are strongly stretched (wall-resolved / skewed meshes), where
                # isotropic aggregation coarsens across the stiff wall-normal direction and stalls. The
                # flow block is frozen at the reference state (never refreshed), so the value-dependent
                # coarsening this turns on carries no refresh cost.
                "strength_threshold": 0.25,
                **flow_block_options,
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

    k_amg = _reparametrized_preconditioner(
        coupled.turbulence.k_preconditioner(
            mdot,
            closure,
            k_ref,
            scalar=scalar,
            reuse=None if reuse is None else reuse.k_preconditioner,
        ),
        k_scale,
    )
    omega_amg = _reparametrized_preconditioner(
        coupled.turbulence.omega_preconditioner(
            mdot,
            closure,
            omega_ref,
            scalar=scalar,
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
    # Carried on a refresh for the same reason as the basis and the parts: how hard the closure's rows
    # are damped is the caller's configuration, not frozen state, so a refresh must not drop it back to 1.
    # ⚠️ The damping's CONFIGURATION is carried, but any reference it holds is REBUILT at this
    # reference state -- the same split as `k_shift_transport` (rebuilt) against `k_jacobian_scale`
    # (carried). A carried reference freezes a taper against the anchor's problem, which under a
    # continuation is the easiest one the march ever sees.
    damping = (
        reuse.turbulence_damping.rebased(coupled, reference_state)
        if reuse is not None
        else as_damping(turbulence_damping)
    )
    return CoupledShiftPolicy(
        coupled.layout,
        momentum,
        k_transport,
        k_coord,
        omega_transport,
        omega_coord,
        flow_preconditioner=block,
        k_preconditioner=k_amg,
        omega_preconditioner=omega_amg,
        shift_basis=basis,
        velocity_shift_parts=parts,
        turbulence_damping=damping,
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
    saddle with one monolithic inverse of the assembled coupled Jacobian, in place of the
    block-diagonal composition.

    Reuses :class:`CoupledShiftPolicy`'s pseudo-transient shift diagonal -- the physics, the same
    velocity ``a_P`` and k/omega transport diagonals -- but replaces its block-diagonal preconditioner
    with a single monolithic inverse of the assembled coupled Jacobian, which forms the true
    pressure Schur coupling rather than approximating it. That inverse is a complete LU
    (:class:`~aquaflux.solve.MonolithicLuPreconditioner`, exact, one cycle), or a multigrid V-cycle
    (:class:`~aquaflux.solve.MonolithicAmgPreconditioner`, bounded memory on a large three-dimensional
    mesh) -- this policy is agnostic to which, needing only the shared callback-matvec interface. On a
    convection-dominated collocated Rhie--Chow RANS saddle either reaches the forward tolerance
    where the block-triangular preconditioner needs hundreds of cycles.

    The inverse is frozen at a reference state and shift (built off the jit path by a
    :class:`~aquaflux.turbulence.MaterializedJacobian` session). Unlike the block
    preconditioner's live ``a_P`` rescaling it does not track the developing state; being a far stronger
    preconditioner it tolerates that freezing at a cost of a few extra cycles, and the shift vanishes at
    the root so the frozen inverse never changes the converged solution or its adjoint. Because it
    is a host object (``scipy`` / UMFPACK / PETSc) it rides as a **static** field rather than a traced
    pytree leaf, and is applied inside the jitted Krylov solve through the callback matvec.

    Attributes
    ----------
    base : CoupledShiftPolicy
        The block policy supplying the pseudo-transient shift diagonal.
    preconditioner : MonolithicLuPreconditioner or MonolithicAmgPreconditioner
        The frozen coupled inverse (a static field). Any object exposing the ``matvec`` /
        ``matvec(transpose=True)`` callback interface works.
    """

    base: CoupledShiftPolicy
    preconditioner: MonolithicLuPreconditioner | MonolithicAmgPreconditioner = eqx.field(
        static=True
    )

    def shift_term(self, phi: jnp.ndarray, residual: jnp.ndarray | None = None) -> ShiftTerm:
        """The block policy's shift diagonal, glued to the frozen factorization preconditioner.

        The preconditioner is a single frozen apply, and the step solves the shifted system with the
        JAX-side Krylov.

        Parameters
        ----------
        phi : jnp.ndarray
            The flat coupled state ``[flow..., k, omega]``, shape ``((dim + 3) n_cells,)``.
        """
        # ⚠️ Forward BOTH of the base term's β-dependent parts. A wrapper that rebuilds a `ShiftTerm`
        # from only `.diagonal` silently discards whatever else the base put there, and the loss is
        # invisible: the march runs, and the dropped behaviour simply never happens. That is exactly how
        # an earlier per-block damping measured as a no-op on this path.
        base = self.base.shift_term(phi, residual)
        diagonal = base.diagonal
        apply = self.preconditioner.matvec()
        # The factorization is frozen, so the preconditioner does not depend on the shift strength.
        return ShiftTerm(diagonal, lambda relaxation: apply, base.row_relaxation)

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

    That matters because it ends up in a strategy's ``adjoint_preconditioner_factory``, a *static*
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


def _coupled_jacobian_plan(
    coupled: CoupledRANS,
    stencil_reach: int,
    column_reach: Sequence[int] | None = None,
    active_rows: np.ndarray | None = None,
):
    """The probing plan for materializing the coupled Jacobian (a mesh-fixed quantity).

    Shared by every monolithic-factorization builder (the initial factorization) and by each in-place
    refresh, so all of them probe the Jacobian the same way.

    ``column_reach`` gives each column field its own reach while keeping the assembly pattern at
    ``stencil_reach``, so the materialized sparsity is unchanged. It is a property of the *case*: which
    columns close inside a shorter reach follows from the schemes the residual was assembled with, so
    it must be measured for a given case rather than assumed. ``None`` (default) probes every column at
    ``stencil_reach``.

    ``active_rows`` excludes field-pair blocks from the pattern entirely, for a caller that already
    knows some sub-block of the materialized Jacobian will never be read -- see
    :meth:`~aquaflux.solve.FieldGroups.active_rows`. ``None`` (default) wants every block.
    """
    n_cells = coupled.momentum.mesh.n_cells
    owner, nb, _ = coupled.momentum.mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)
    if column_reach is None:
        return ColumnProbePlan.uniform(
            block_stencil_colouring(owner, nb, n_cells, stencil_reach),
            coupled.layout.n_fields,
            active_rows=active_rows,
        )
    return column_probe_plan(
        owner, nb, n_cells, column_reach, stencil_reach, active_rows=active_rows
    )


@dataclasses.dataclass(frozen=True)
class CoupledJacobianProbe:
    """How the coupled Jacobian is materialized: the colouring plan and its fixed de-compression map.

    Both are functions of the cell graph and the stencil / per-column reaches alone -- never of the
    state, and never of the molecular viscosity. So **one is valid for a whole Reynolds continuation**
    and for the refresh hook running beside it, and they are one object rather than two arguments
    threaded in parallel because they are built together and consumed together at every call site (the
    initial build, and every in-place refresh).

    Building them is not free. The colouring is a graph pass over the whole mesh, and on a
    three-dimensional coupled case the gather map is the single largest allocation the case makes -- so
    a driver that builds one engine per continuation rung with a refresh hook beside it would otherwise
    build both twice per rung, for six identical copies over a three-rung ramp. A preconditioner
    session (:func:`open_session`) builds exactly one and hands it to every consumer.

    Attributes
    ----------
    plan : ColumnProbePlan
        The collision-free colouring and per-column reach the coloured directional-derivative probe
        runs, which is what fixes how many probes a materialize costs.
    structure : ProbeGather
        The fixed compressed-sparse-row (CSR) structure -- a row-pointer array plus a flat column-index
        array -- together with the ordering that scatters the probe responses into it, so a materialize
        de-compresses by one gather rather than a scatter loop and a re-sort.
    """

    plan: ColumnProbePlan
    structure: ProbeGather
    gradient_sweeps: int | None = None
    production_viscosity_frozen: bool = eqx.field(static=True, default=False)

    @classmethod
    def build(
        cls,
        coupled: CoupledRANS,
        stencil_reach: int = 3,
        column_reach: Sequence[int] | None = None,
        gradient_sweeps: int | None = None,
        *,
        active_rows: np.ndarray | None = None,
        production_viscosity_frozen: bool = False,
    ) -> CoupledJacobianProbe:
        """Colour the cell graph at these reaches and precompute the de-compression for it.

        Parameters
        ----------
        coupled : CoupledRANS
            The assembled case, read for its mesh graph and block layout only -- so any companion of
            the same case (a Reynolds-continuation rung at a scaled viscosity) gives the same probe.
        stencil_reach : int
            The cell-graph distance the assembled sparsity covers (coupled RANS reaches distance ``3``).
        column_reach : sequence of int, optional
            A shorter reach per **column field**, in the flat layout's order ``[u, ..., p, k, omega]``,
            while the assembled pattern stays at ``stencil_reach``. Exact only for a column that
            genuinely carries nothing further out; measure it for the case rather than assuming it.
            ``None`` (default) probes every column at ``stencil_reach``.
        gradient_sweeps : int, optional
            Probe a copy of the residual whose corrected-gradient solve is capped at this many
            Richardson sweeps, rather than the residual itself. See :meth:`narrow`. ``None`` (default)
            probes the residual as it stands.
        production_viscosity_frozen : bool
            Materialize the Jacobian of the **frozen-production** copy
            (:func:`frozen_production_viscosity`) rather than of ``coupled`` itself. Set it whenever
            the solve runs with ``jacobian_production_viscosity``: the preconditioner must be
            assembled from the operator the Krylov iteration APPLIES, and those two differ by a term
            the size of the k row's own diagonal. ``False`` (default) is byte-identical.

            ⚠️ **This is not the same axis as ``gradient_sweeps`` even though both go through**
            :meth:`narrow`. That one narrows the probe while the operator stays exact — its purpose is
            to make the colouring collision-free. This one follows an operator that has already
            changed. Setting the wrong one leaves a preconditioner built for a matrix nobody solves:
            measured on the pitzDaily target rung at 18--28 restart cycles a step against 4--14.
        active_rows : np.ndarray, optional
            Exclude field-pair blocks from the materialized pattern entirely -- for a probe built
            specifically to feed one consumer that is known never to read some sub-block of the
            Jacobian, such as a :class:`~aquaflux.solve.BlockTriangularFieldSplit`'s dropped triangle
            (:meth:`~aquaflux.solve.FieldGroups.active_rows`). ``None`` (default, and the only sound
            choice for a probe that might be shared with a monolithic consumer) wants every block, and
            is byte-identical to a probe built without this argument.

        Returns
        -------
        CoupledJacobianProbe
            The shared probe.
        """
        plan = _coupled_jacobian_plan(coupled, stencil_reach, column_reach, active_rows)
        return cls(
            plan, block_stencil_gather_map(plan), gradient_sweeps, production_viscosity_frozen
        )

    def narrow(self, coupled: CoupledRANS) -> CoupledRANS:
        """The assembler this probe differentiates -- ``coupled``, or a reduced-sweep copy of it.

        The colouring recovers couplings out to ``stencil_reach`` and no further, and a residual that
        reaches beyond it has its far entries **folded onto near ones** rather than dropped (a
        colouring is collision-free only for the pattern it was built at). A corrected-gradient
        reconstruction is the term most able to reach past a fixed distance: each of its Richardson
        sweeps couples one further ring wherever the mesh is skewed, so ``n`` sweeps put the residual's
        stencil at ``n + 1`` (the reconstruction reads ``n`` cells out, and a face flux gathers the
        gradient of the cells on both sides). Capping the sweeps for the probe alone leaves the matrix exact
        for the residual it was taken from, which is a stated approximation of the operator rather than
        a corrupted matrix.

        Choose the cap from the case: it is the *whole* stencil that has to fit inside
        ``stencil_reach``, and the gradient is only one of the terms feeding it (in the coupled RANS
        residual the eddy viscosity's strain-rate dependence spends a ring of its own). Measure the
        reach rather than deriving it from the sweep count alone.

        The narrowed copy reaches the **preconditioner only** -- the solve's operator stays the exact
        Jacobian--vector product of ``coupled`` -- so neither the converged state nor its adjoint moves.
        On an orthogonal mesh the skewness correction vanishes identically and this is a no-op in value
        as well as in reach.

        Parameters
        ----------
        coupled : CoupledRANS
            The assembler whose Jacobian is being materialized. Passed per call rather than held,
            because a refresh hook is rebound across Reynolds-continuation rungs.

        Returns
        -------
        CoupledRANS
            The narrowed copy, or ``coupled`` itself when no cap was asked for.
        """
        narrowed = _probed_assembler(coupled, self.gradient_sweeps)
        # Both stand-ins land here, which is what keeps every consumer -- the initial build, the
        # refresh hook, and the rebind across a Reynolds rung -- materializing the same operator the
        # Krylov solve applies, without any of them knowing there are two axes.
        return (
            frozen_production_viscosity(narrowed) if self.production_viscosity_frozen else narrowed
        )


def frozen_production_viscosity(coupled: CoupledRANS) -> CoupledRANS:
    """``coupled`` with ``k`` frozen inside the k-production's eddy viscosity.

    The Patankar treatment of the one term that drives the k row's Jacobian diagonal negative:
    ``nu_t`` is proportional to ``k``, so the production ``nu_t S**2`` is too, and subtracting a
    source that grows with its own variable puts a negative term on that row's diagonal. See
    :attr:`~aquaflux.turbulence.SSTTurbulence.explicit_production_viscosity` for what that costs and
    what freezing it buys.

    ⚠️ **The copy is for a Jacobian stand-in only** -- pass its ``residual`` as the operator a shifted
    solve differentiates, so the true residual still decides where the march lands and the converged
    root and its implicit-function-theorem adjoint are untouched. This term is active everywhere, not
    only where some cap bites, so a *residual* carrying it would make the sensitivity wrong
    everywhere.

    Parameters
    ----------
    coupled : CoupledRANS
        The assembled coupled system.

    Returns
    -------
    CoupledRANS
        A copy whose turbulence assembler carries the flag. ``coupled`` itself is unchanged.
    """
    # `dataclasses.replace`, not `eqx.tree_at`: the flag is a STATIC field, so it lives in the
    # treedef rather than among the leaves and `tree_at` cannot address it at all.
    return dataclasses.replace(
        coupled,
        turbulence=dataclasses.replace(coupled.turbulence, explicit_production_viscosity=True),
    )


def _probed_assembler(coupled: CoupledRANS, gradient_sweeps: int | None) -> CoupledRANS:
    """``coupled``, or the reduced-sweep copy a preconditioner's coloured probe differentiates.

    Shared by :meth:`CoupledJacobianProbe.narrow` and by the factorization builders, which materialize
    the same Jacobian without needing the probe's de-compression map. See that method for what the cap
    is for and how to choose it.
    """
    return coupled if gradient_sweeps is None else narrow_gradient_sweeps(coupled, gradient_sweeps)


def _monolithic_shift_source(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    shift_basis: ShiftBasis,
    velocity_shift_parts: VelocityShiftParts | None = None,
    turbulence_damping: TurbulenceDamping | float = 1.0,
) -> CoupledShiftPolicy:
    """The shift policy a **monolithically** preconditioned step reads its diagonal from.

    Shared by all three monolithic builders (threshold-ILU, complete-LU, algebraic multigrid), which
    want exactly the same thing from it and had each written the call out.

    :class:`MonolithicFactorShiftPolicy` takes only ``base.shift_term(phi).diagonal`` -- it supplies its
    own inverse and never calls the block policy's ``make_preconditioner``. So this policy is built
    **without a flow block at all**: the shift's velocity buckets come from the flow assembler's frozen
    momentum diagonal, which is the whole dependency. Building a block preconditioner to supply them
    instead would carry two multigrid hierarchies that are never applied.

    Not building them is not only a saving. Their aggregation reads the operator's *values*, so their
    coarse grids -- and hence the array shapes the policy would carry -- move with the molecular
    viscosity, and the compiled coupled step is keyed on those shapes. Carrying them therefore makes
    every Reynolds-continuation rung a fresh compilation of the whole solve, the largest fixed overhead
    in a three-dimensional march. (Turning the aggregation down to graph-only would fix the shapes and is
    inert in everything measured, but it lets the hierarchy refuse to build on a degenerate coarse row --
    a failure mode for something with no consumer.)

    ``turbulence_damping`` is threaded for the identical reason, and is likewise a property of the shift
    rather than of any preconditioner -- a builder that names a preconditioner strategy is the wrong home
    for it, and omitting it here would leave it absent from the monolithic paths the cases actually run.

    ``velocity_shift_parts`` is threaded through for the same reason it exists on the block path: it is a
    property of the **shift**, not of the preconditioner, and a :class:`LiveViscosityVelocityParts` needs
    only momentum + turbulence + the two variable transforms, so building without a flow block does not
    exclude it. Its motivating configuration -- a dual-time low-shift march whose shift must track the
    developing eddy viscosity -- is a monolithic one, so omitting it here left it absent from exactly the
    path it was built for.
    """
    return _coupled_shift_policy(
        coupled,
        reference_state,
        UnpreconditionedScalars(),
        shift_basis=shift_basis,
        velocity_shift_parts=velocity_shift_parts,
        turbulence_damping=turbulence_damping,
        build_flow_block=False,
    )


def _frozen_shift_diagonal(base: CoupledShiftPolicy, beta: float, state: jnp.ndarray) -> np.ndarray:
    """The frozen pseudo-transient shift diagonal the factorization is built against, at ``state``.

    ``beta`` scales the base policy's shift diagonal; the ``stop_gradient`` keeps the frozen
    factorization off the differentiation path. Shared by the initial build and every in-place refresh,
    for the complete-LU and multigrid preconditioners alike.

    It asks the term for the shift at ``beta`` rather than scaling the diagonal itself, so a policy
    that runs a block at its own pseudo-timestep (:attr:`CoupledShiftPolicy.turbulence_damping`)
    preconditions the operator it actually forms. Open-coding ``beta * diagonal`` here would drop that
    factor silently -- the march would run, and the preconditioner would simply be fitted to a
    different operator than the one being solved.
    """
    return np.asarray(jax.lax.stop_gradient(base.shift_term(state).shift(beta)))


def _coupled_step(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    policy: ShiftPolicy,
    *,
    regime: _LinearSolveRegime,
    globalization: Globalization,
    dual_time: DualTimeLoop | None,
    krylov_solver: lx.AbstractLinearSolver | None,
    inner_observer: Callable[..., None] | None = None,
    inner_refresh: Callable[[jnp.ndarray], None] | None = None,
    step_limit: Callable[..., jnp.ndarray] | None = None,
    step_projection: Callable[..., jnp.ndarray] | None = None,
    jacobian_gradient_sweeps: int | None = None,
    jacobian_production_viscosity: bool = False,
) -> NewtonStrategy:
    """Assemble the pseudo-transient / dual-time step around an already-composed shift policy.

    **The one place the coupled march's globalization is configured**, for every preconditioner. What
    differs between the block-diagonal and monolithic paths is which policy they hand in and which
    Krylov restart regime they name; the schedule, the line search, the escalation ladder, the dual-time
    inner loop and the positivity guard are one implementation.

    That matters more than the duplication it removes. The builders each grew their own copy of this
    tail, and the copies drifted in ways that had nothing to do with preconditioning: the monolithic one
    gained the k-positivity step limit, the cycle budget and the inner refresh, the block-diagonal one
    gained the growth rungs and the descent backoff, and neither gained the other's. A march's
    globalization should not depend on which matrix its preconditioner was built from.

    The step carries **no residual measure of its own choosing**. The measure is part of the solve's
    stopping test (:class:`~aquaflux.solve.Convergence`), and the march that runs the step hands it in at
    every outer iteration. The default linear solve stops in whatever that measure is at the time
    (``relative_residual_gmres(norm=None)``), so the line search, the shift, the linear solve's stop and
    the convergence test are all taken in one definition. A step marched with no measure given is judged
    by the Euclidean norm.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler, for the Jacobian operator the Krylov solve applies.
    reference_state : jnp.ndarray
        The state the preconditioner and the shift were frozen at.
    policy : ShiftPolicy
        The composed shift-and-preconditioner policy — a :class:`CoupledShiftPolicy` for the
        block-diagonal path, a :class:`MonolithicFactorShiftPolicy` for a materialized one.
    regime : _LinearSolveRegime
        The Krylov tolerance and restart regime for the default forward solve, used when
        ``krylov_solver`` is ``None``. Only the *regime* is per-family (a near-exact factorization
        needs a far smaller Arnoldi subspace than a block-diagonal preconditioner); the tolerance is
        measured in the progress measure the march hands the step.
    globalization : Globalization
        How hard the march damps and what it does when a step misbehaves: the pseudo-transient
        schedule, the escalation ladder, the divergence guard and the line search, in one object all
        six of the library's builders take. Nothing in it names a preconditioner, which is why it is
        here rather than on any of them. An unset ``line_search`` takes :data:`_COUPLED_LINE_SEARCH`
        rungs and every other unset field the step class's own default
        (:meth:`~aquaflux.solve.Globalization.with_defaults`).
    dual_time : DualTimeLoop or None
        The dual-time inner loop, or ``None`` for the single shifted step. Unlike the settings above
        this is not shared: the flow-only and scalar marches have no dual-time form, so the loop is a
        coupled choice and stays on the coupled builders.
    krylov_solver, inner_observer, inner_refresh, step_limit, step_projection
        The linear solve and the per-step guards. See the two step classes.
    jacobian_production_viscosity : bool
        Freeze ``k`` inside the k-production's eddy viscosity in the **operator** the shifted solve
        differentiates, leaving the residual it is driving to zero exact (see
        :func:`frozen_production_viscosity`). ``nu_t`` is proportional to ``k``, so the production is
        too, and differentiating that puts a negative term on the k row's Jacobian diagonal which the
        pseudo-time shift then has to cancel. ``False`` (default) is byte-identical.

        ⚠️ **A caller supplying its own ``probe`` must build it from the same stand-in.** A probe built
        here follows the operator automatically, but one passed in does not, and a preconditioner
        assembled from a different matrix than the one being solved is a preconditioner for the wrong
        problem -- measured at 22--26 restart cycles a step against 4--14 when the two were mismatched,
        the stand-in differing from the exact operator by a term the size of the k row's own diagonal.
    jacobian_gradient_sweeps : int, optional
        Cap the gradient reconstruction's sweeps in the copy of the residual the **forward Jacobian**
        is differentiated from, leaving the residual itself untouched. ``None`` (the default)
        differentiates the residual as it stands, which is byte-identical to not passing it.

        The residual and its Jacobian tolerate completely different amounts of approximation, and
        today they are held to one accuracy for no reason but that one function serves both. The
        reconstruction's accuracy inside ``R`` decides **which discrete equations are being solved**,
        so loosening it moves the root; inside ``J`` it decides only how fast the inexact-Newton
        iteration reaches that root, which is the same latitude ``forward_rtol`` already takes. Each
        sweep costs an operator apply in the residual and two in the tangent (a primal and a tangent
        apply), and a march pays a tangent per Krylov iteration and a residual only once per step --
        so the sweeps are a larger share of the differentiated path than of the evaluated one.

        Choose the cap against the reconstruction's own contraction rate rather than by feel: the
        Jacobian's relative error is roughly the gradient's, which falls by that rate per sweep
        (:func:`~aquaflux.schemes.contraction_rate` measures it). Judge it on outer steps and Krylov
        cycles over a whole march, never on the cost of one matrix-vector product -- an operator too
        far from the true Jacobian costs more iterations than the cheaper product saves.

        Distinct from ``probe_gradient_sweeps``, which narrows the copy a coloured probe
        *materializes* to keep the recovered matrix collision-free. Both leave the residual alone;
        they differ in which approximation of the Jacobian they cheapen -- the preconditioner's or
        the Krylov operator's -- and either may be set without the other.

    Returns
    -------
    NewtonStrategy
        A :class:`~aquaflux.solve.DualTimeStep` when ``dual_time`` is given, else a
        :class:`~aquaflux.solve.PseudoTransientStep`.
    """
    # The residual whose Jacobian--vector product is the Krylov operator. `None` leaves the step
    # differentiating the residual it is driving to zero, exactly as before.
    jacobian_operator = _probed_assembler(coupled, jacobian_gradient_sweeps)
    if jacobian_production_viscosity:
        jacobian_operator = frozen_production_viscosity(jacobian_operator)
    jacobian_residual = (
        None
        if jacobian_gradient_sweeps is None and not jacobian_production_viscosity
        else jacobian_operator.residual
    )
    # The linear solve stops in the measure the march is judging the step by at that moment (`norm=None`),
    # so a solve cannot converge in a quantity the march does not read -- including after the march has
    # rebuilt the measure at a new state. That is the shared half of the decision; only the restart
    # regime differs per preconditioner family (see `_LinearSolveRegime`).
    solver = (
        krylov_solver
        if krylov_solver is not None
        else relative_residual_gmres(
            regime.rtol,
            norm=None,
            restart=regime.restart,
            stagnation_iters=40,
            max_restarts=regime.max_restarts,
        )
    )
    # The coupled march line-searches unless its caller said otherwise: an unset `line_search` takes
    # this residual's base, an explicit one -- including 0 -- is kept, and everything else the
    # globalization leaves unset falls through to the step class's own default.
    globalization = globalization.with_defaults(line_search=_COUPLED_LINE_SEARCH)
    if dual_time is None:
        hooks = sorted(
            name
            for name, hook in (("inner_observer", inner_observer), ("inner_refresh", inner_refresh))
            if hook is not None
        )
        if hooks:
            raise TypeError(
                f"{hooks} are hooks of the dual-time inner loop, and this march has none: the single "
                "shifted step runs no inner iterations to observe or refresh. Give "
                "dual_time=DualTimeLoop(...) to march in dual time, or leave them unset."
            )
    elif dual_time.refresh_on_cycles is not None and inner_refresh is None:
        raise TypeError(
            f"DualTimeLoop(refresh_on_cycles={dual_time.refresh_on_cycles}) has nothing to fire: no "
            "inner_refresh was given, and only a materialized-Jacobian preconditioner's session supplies "
            "one of its own -- a frozen step and a block-diagonal session do not. Open a session for a "
            "MaterializedJacobian, pass inner_refresh, or leave refresh_on_cycles unset."
        )
    if dual_time is not None:
        # Dual-time (backward-Euler) march: an inner Newton loop per outer timestep on the transient
        # residual, so the measured steady residual is the honest discrete time derivative rather than
        # beta x travel, and a larger pseudo-timestep (smaller beta, driven by a step control) stays
        # stable. The inner loop replaces the escalation ladder, so `dual_time_step` refuses the
        # escalation/acceptance settings and the line search's growth rung and rule rather than
        # dropping them.
        return globalization.dual_time_step(
            policy,
            **dual_time.settings(),
            krylov_solver=solver,
            adjoint_preconditioner_factory=policy.adjoint_factory(),
            inner_observer=inner_observer,
            inner_refresh=inner_refresh,
            step_limit=step_limit,
            step_projection=step_projection,
            jacobian_residual=jacobian_residual,
        )
    # The positivity guard is passed on BOTH branches, and the single-step one needs it as much: its
    # escalation ladder is no substitute, because the divergence guard fires on a non-finite residual,
    # which is already the poisoned state -- one cell's `k` through zero has by then NaN'd `sqrt(k)` and
    # the whole eddy viscosity with it. Historically only the monolithic path's dual-time branch carried
    # it, which is drift rather than design.
    return globalization.step(
        policy,
        krylov_solver=solver,
        adjoint_preconditioner_factory=policy.adjoint_factory(),
        step_limit=step_limit,
        step_projection=step_projection,
        jacobian_residual=jacobian_residual,
    )


def _monolithic_factor_step(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    base: CoupledShiftPolicy,
    preconditioner: MonolithicLuPreconditioner | MonolithicAmgPreconditioner,
    *,
    globalization: Globalization,
    dual_time: DualTimeLoop | None,
    krylov_solver: lx.AbstractLinearSolver | None,
    regime: _LinearSolveRegime,
    inner_observer: Callable[..., None] | None = None,
    inner_refresh: Callable[[jnp.ndarray], None] | None = None,
    step_limit: Callable[..., jnp.ndarray] | None = None,
    step_projection: Callable[..., jnp.ndarray] | None = None,
    jacobian_gradient_sweeps: int | None = None,
    jacobian_production_viscosity: bool = False,
) -> NewtonStrategy:
    """Compose a monolithic preconditioner with the block shift, then build the step.

    The shared seam of every materialized-Jacobian build: it glues the already-built ``preconditioner``
    (complete LU, multigrid V-cycle or field split) to the block shift ``base`` via a
    :class:`MonolithicFactorShiftPolicy` and hands the result to :func:`_coupled_step`. The inverse
    families differ only in how they construct ``preconditioner``; everything about the march itself is
    :func:`_coupled_step`'s.
    """
    return _coupled_step(
        coupled,
        reference_state,
        MonolithicFactorShiftPolicy(base, preconditioner),
        regime=regime,
        globalization=globalization,
        dual_time=dual_time,
        krylov_solver=krylov_solver,
        inner_observer=inner_observer,
        inner_refresh=inner_refresh,
        step_limit=step_limit,
        step_projection=step_projection,
        jacobian_gradient_sweeps=jacobian_gradient_sweeps,
        jacobian_production_viscosity=jacobian_production_viscosity,
    )


def _beta_tracking_refresh(
    coupled: CoupledRANS,
    stencil_reach: int,
    column_reach: Sequence[int] | None = None,
    probe_gradient_sweeps: int | None = None,
    *,
    every_step: bool,
    beta_floor: float = 0.0,
    observer: Callable[[RefreshTiming], None] | None = None,
    probe: CoupledJacobianProbe | None = None,
) -> Callable[[NewtonStrategy, jnp.ndarray], None]:
    """Shared skeleton for the β-tracking ``refresh_preconditioner`` hooks (complete-LU and algebraic multigrid).

    Returns a ``refresh_preconditioner(active_step, state)`` that reads ``β`` from the step's
    :class:`~aquaflux.solve.ConstantRelaxation` schedule and re-factors the step's
    :class:`MonolithicFactorShiftPolicy` preconditioner in place at ``J(state) + β·d(state)``. With
    ``every_step`` it does so on every step (the cheap exact-LU cadence); without, only on its first call
    and after each ``rebind`` -- a multigrid re-materialize is too expensive to pay every step, so between
    those the rebuild is left to the dual-time loop's cost trigger, through ``refresh_at``.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler (supplies the Jacobian-vector product and the shift diagonal).
    stencil_reach : int
        The cell-graph distance the Jacobian's sparsity is probed to (coupled RANS reaches distance ``3``).
    column_reach : sequence of int, optional
        A stencil reach per **column field**, in the flat layout's order ``[u, ..., p, k, omega]``, while
        the assembled sparsity stays at ``stencil_reach``. The probe costs one directional derivative per
        (colour, column field) and the colour count falls steeply with the reach, so a column whose
        couplings all close inside a shorter reach can be probed far more cheaply and assembled
        unchanged. It is **exact only for a column that genuinely carries nothing further out** -- a
        column with far couplings is corrupted rather than truncated, because the short colouring folds
        them onto near entries. Which columns qualify follows from the schemes the residual was
        assembled with, so measure it for the case rather than assuming it
        (``validation/bfs3d_openfoam/probe_reach_audit.py`` reports it per column). ``None`` (default)
        probes every column at ``stencil_reach``.
    probe_gradient_sweeps : int, optional
        Materialize the preconditioner's Jacobian from a copy of the residual whose corrected-gradient
        solve is capped at this many Richardson sweeps, so its stencil fits inside the reach the
        colouring recovers -- see :meth:`CoupledJacobianProbe.narrow`. The solve's own operator is
        unchanged, so the converged state and its adjoint are too. ``None`` (default) probes the
        residual as it stands.
    every_step : bool
        Re-factor on every step (``True``), or only on the first call and after each ``rebind``
        (``False``).
    beta_floor : float
        A lower bound on the shift strength the **preconditioner** is refreshed at: it is built at
        ``max(beta, beta_floor)`` while the march keeps solving at its own ``beta``. ``0.0`` (default)
        tracks ``beta`` exactly.
    observer : callable, optional
        ``(timing: RefreshTiming) -> None``, called on each refresh with which branch ran, its total
        seconds, and its per-phase costs. ``None`` (default) elides the call.
    probe : CoupledJacobianProbe, optional
        A shared colouring plan and de-compression map, when the caller already has one -- see
        :class:`CoupledJacobianProbe`. ``None`` (default) builds one from
        ``stencil_reach`` / ``column_reach``, which are then ignored if a ``probe`` is given.

    Returns
    -------
    callable
        ``refresh_preconditioner(active_step, state) -> None``, carrying ``refresh_at`` (the inner-loop hook)
        and ``rebind`` (point it at another companion of the same case -- see below).
    """
    if probe is None:
        probe = CoupledJacobianProbe.build(
            coupled, stencil_reach, column_reach, probe_gradient_sweeps
        )
    plan, structure = probe.plan, probe.structure

    # WHICH case this hook currently refreshes for, in a mutable binding rather than closed over, so
    # `rebind` can point it at another companion of the same case. A Reynolds continuation solves a
    # sequence of companions that differ only in their molecular viscosity, and rebinding one hook lets
    # them all share ONE preconditioner -- which is what keeps the compiled coupled step a cache hit
    # across a rung boundary, since the preconditioner rides in a static field compared by identity.
    # Both probes below take the assembler as an argument to a module-level jitted function, so swapping
    # it changes no compilation key of theirs either.
    # `"probed"` is what the coloured probe differentiates, which is the assembler itself unless the probe
    # asks for a reduced-sweep copy (`CoupledJacobianProbe.narrow`). It is stored beside the companion
    # rather than derived per call so a rebind narrows once; the drift measure below deliberately reads
    # `"coupled"`, since it reports the real case's eddy viscosity and not the preconditioner's stand-in.
    bound: dict[str, CoupledRANS] = {"coupled": coupled, "probed": probe.narrow(coupled)}

    # `frozen` a traced argument (not closed over) so the jvp-matvec compiles once and every refactor
    # reuses it, rather than a fresh lambda recompiling each step.
    def matvec_at(frozen, v):
        return _jacobian_matvec(bound["probed"], frozen, v)

    # Batched form (vmapped over the tangent) so the coloured probes of a full materialize run as a few
    # fused passes rather than a Python loop of separate calls. Built once (state-independent, `frozen` a
    # traced argument) so it compiles a single time and every materialize reuses it. Used only by the AMG
    # preconditioner's `refresh_in_place`.
    def batched_matvec_at(frozen, seeds):
        return _batched_jacobian_matvec(bound["probed"], frozen, seeds)

    # Pending on the first call -- the build froze the preconditioner at its own shift, not the march's --
    # and again after `rebind`, since the standing preconditioner then describes the PREVIOUS companion.
    forced_full = {"pending": True}

    def _report_refresh(
        kind: str, started: float, phases: tuple[tuple[str, float], ...] | None = None
    ) -> None:
        """Tell an injected observer which branch ran and what each part of it cost.

        The total alone cannot be acted on: a refresh dominated by the coloured jvp probe and one
        dominated by the multigrid setup take the same wall time and call for opposite fixes.
        """
        if observer is not None:
            observer(RefreshTiming(kind, time.perf_counter() - started, tuple(phases or ())))

    # Which step the inner-loop hook is refreshing, kept current by `refresh_preconditioner` below.
    bound_step: dict[str, NewtonStrategy] = {}

    def refresh_preconditioner(active_step: NewtonStrategy, state: jnp.ndarray) -> None:
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
        if not (every_step or forced_full["pending"]):
            _report_refresh("none", started)
            return
        forced_full["pending"] = False
        frozen = jax.lax.stop_gradient(state)
        shift = np.asarray(jax.lax.stop_gradient(policy.base.shift_term(state).shift(pc_beta)))
        _report_refresh("full", started, _materialize_at(pc, frozen, shift))

    def _materialize_at(pc, frozen, shift) -> tuple[tuple[str, float], ...]:
        """Re-materialize the preconditioner at ``frozen`` with shift diagonal ``shift``.

        The AMG preconditioner materializes via the coloured probe and takes the batched form; the
        complete-LU preconditioner does not, so pass it only on the AMG path.
        """
        extra = (
            {
                "batched_matvec": lambda seeds: batched_matvec_at(frozen, seeds),
                "probe_batch_size": _PROBE_BATCH_SIZE,
                "structure": structure,
            }
            if isinstance(pc, MaterializedJacobianPreconditioner)
            else {}
        )
        return pc.refresh_in_place(lambda v: matvec_at(frozen, v), plan, shift, **extra) or ()

    def refresh_at(iterate) -> None:
        """``inner_refresh`` hook: rebuild the preconditioner at this mid-step iterate.

        *When* to fire is decided by the dual-time loop (``DualTimeStep.refresh_on_cycles``), not here,
        so that the rule which triggers the refresh is the same one that forgives the abort it would
        otherwise be discarded by.

        The march's expensive inner solves are **stale-preconditioner** effects, not hard operators: at
        the hardest solve of a three-dimensional coupled march a preconditioner rebuilt at that very
        iterate converged in an order of magnitude fewer cycles than the march's own. Refreshing here —
        between inner iterations, after the line search and before the next solve — keeps the step's
        progress, where the alternative reaction (abort the step and escalate β) discards both the work
        and the pseudo-timestep.

        Reacting is also what makes this worth doing as a *replacement* for a scheduled refresh rather
        than an addition to one: a fixed cadence pays on every step to protect the minority that needs
        it, and the right interval is regime-dependent in a way no fixed cadence can track (one step of
        staleness is nearly free at a large shift and dominates the solve at a small one).
        """
        if "step" not in bound_step:
            return
        started = time.perf_counter()
        pc = bound_step["step"].shift_policy.preconditioner
        beta = max(float(bound_step["step"].relaxation_schedule.beta), beta_floor)
        frozen = jax.lax.stop_gradient(jnp.asarray(iterate))
        shift = np.asarray(
            jax.lax.stop_gradient(
                bound_step["step"].shift_policy.base.shift_term(frozen).shift(beta)
            )
        )
        _report_refresh(
            "inner",
            started,
            _materialize_at(pc, frozen, shift),
        )

    def rebind(companion: CoupledRANS) -> None:
        """Point this hook at another companion of the same case, and force the next refresh to be full.

        A Reynolds continuation solves a sequence of companions differing only in their molecular
        viscosity. Each is a separate ``solve_coupled`` segment, and rebuilding a preconditioner per
        segment recompiles the whole coupled solve, because the preconditioner rides in a *static* field
        of the Newton step and is compared by identity. Rebinding one hook instead lets every segment
        share a single preconditioner object -- so the compiled step is a cache hit across a rung
        boundary -- while each segment's V-cycle is still fitted to its own problem, at its own state and
        shift, by the refresh the march runs before that segment's first step.

        The standing preconditioner describes the previous companion, so the next refresh is forced to a **full** re-materialize at the new one.

        Forward-only, like everything else on this hook. The companion must be the same case -- same
        mesh, same layout, same schemes -- since the colouring plan and the gather map are not rebuilt.

        Parameters
        ----------
        companion : CoupledRANS
            The assembler the following segment solves.
        """
        bound["coupled"] = companion
        bound["probed"] = probe.narrow(companion)
        forced_full["pending"] = True

    refresh_preconditioner.refresh_at = refresh_at
    refresh_preconditioner.rebind = rebind
    return refresh_preconditioner


#: The shift strength a materialized preconditioner's first build is fitted at when its spec leaves
#: ``build_beta`` unset. A frozen coarse space is chosen at that build and reused by every later refit, so
#: this is not only the first step's operator.
_BUILD_BETA = 2.0

#: The march keywords a session owns rather than receives per build: the preconditioner is what the
#: session was opened with, and the operator stand-in must match the probe the session built for it.
_SESSION_OWNED = frozenset({"preconditioner", "jacobian_production_viscosity"})


class PreconditionerSession(Protocol):
    """One coupled preconditioner, kept current across every step it serves.

    A session is what a march holds on to between its steps: the frozen inverse, the colouring probe it
    was materialized with, and the per-step refresh hook. Those must outlive a single Newton step -- a
    Reynolds continuation builds a step per rung, a refresh builds one per segment -- and must be the
    *same objects* each time, because the inverse and the hooks ride in static fields of the step and a
    new object recompiles the whole coupled solve.

    Open one with :func:`open_session`.

    Attributes
    ----------
    refresh_preconditioner : callable or None
        ``(step, state) -> None``, called by the march before every step to re-fit the inverse at that
        step's shift; ``None`` for a family with nothing to re-fit.
    """

    refresh_preconditioner: Callable[[NewtonStrategy, jnp.ndarray], None] | None

    def build(self, state: jnp.ndarray, **march: object) -> NewtonStrategy:
        """The Newton step at ``state``, configured by the march keywords of :func:`coupled_step`."""
        ...

    def refresh(
        self, state: jnp.ndarray, previous: NewtonStrategy, **march: object
    ) -> NewtonStrategy:
        """Re-freeze at the developed ``state``; ``previous`` is the step it replaces."""
        ...

    def rebind(self, coupled: CoupledRANS) -> None:
        """Point the session at another companion of the same case, such as the next Reynolds rung."""
        ...


def _resolved_regime(
    base: _LinearSolveRegime,
    rtol: float | None,
    restart: int | None,
    max_restarts: int | None,
) -> _LinearSolveRegime:
    """A family's forward-solve regime with any explicitly given setting in place of its default."""
    return _LinearSolveRegime(
        base.rtol if rtol is None else rtol,
        base.restart if restart is None else restart,
        base.max_restarts if max_restarts is None else max_restarts,
    )


def _resolved_shift(
    shift: ShiftSettings | None,
) -> tuple[ShiftBasis, VelocityShiftParts | None, TurbulenceDamping | float]:
    """The shift's basis, velocity parts and damping, with what ``shift`` leaves unset resolved.

    The builders' defaults are written here once: the full-diagonal basis, the frozen velocity parts,
    and a damping ratio of one.

    Parameters
    ----------
    shift : ShiftSettings or None
        The shift settings a builder was given; ``None`` sets nothing.

    Returns
    -------
    tuple
        ``(basis, velocity_parts, turbulence_damping)``.
    """
    shift = ShiftSettings() if shift is None else shift
    return (
        _DEFAULT_SHIFT_BASIS if shift.basis is None else shift.basis,
        shift.velocity_parts,
        1.0 if shift.turbulence_damping is None else shift.turbulence_damping,
    )


def _resolved_linear_solve(
    linear_solve: LinearSolveSettings | lx.AbstractLinearSolver | None, base: _LinearSolveRegime
) -> tuple[_LinearSolveRegime, lx.AbstractLinearSolver | None]:
    """The forward solve's regime and explicit solver, from a builder's ``linear_solve``.

    Parameters
    ----------
    linear_solve : LinearSolveSettings, lineax.AbstractLinearSolver or None
        A regime whose unset fields take ``base``; a whole solver, which replaces the regime; or
        ``None`` for ``base`` itself.
    base : _LinearSolveRegime
        The chosen preconditioner family's own regime.

    Returns
    -------
    tuple
        ``(regime, krylov_solver)``, the solver ``None`` unless one was given.
    """
    if linear_solve is None:
        return base, None
    if isinstance(linear_solve, LinearSolveSettings):
        return (
            _resolved_regime(
                base, linear_solve.rtol, linear_solve.restart, linear_solve.max_restarts
            ),
            None,
        )
    return base, linear_solve


def _march_keywords(march: dict) -> dict:
    """``march`` bound against :func:`coupled_step`'s signature, with its defaults filled in.

    Binding against the signature, rather than restating each default here, keeps those defaults in one
    place -- the public signature -- and makes an unknown keyword a :exc:`TypeError` at the call.
    """
    owned = sorted(_SESSION_OWNED & set(march))
    if owned:
        raise TypeError(
            f"{owned} belong to the session, not to one of its builds: the preconditioner is the one "
            "the session was opened with, and the operator stand-in must match the probe it built. "
            "Pass them to open_session instead."
        )
    signature = inspect.signature(coupled_step)
    unknown = sorted(set(march) - set(signature.parameters))
    if unknown:
        raise TypeError(
            f"{unknown} {'is not a march setting' if len(unknown) == 1 else 'are not march settings'} "
            "of coupled_step. A preconditioner setting belongs on the spec (BlockDiagonal(...), "
            "MaterializedJacobian(...)); a setting of how the march damps, such as beta0 or "
            "line_search, belongs on globalization=Globalization(...); the inner loop, the forward solve "
            "and the shift are dual_time=DualTimeLoop(...), linear_solve=LinearSolveSettings(...) and "
            "shift=ShiftSettings(...)."
        )
    bound = signature.bind(None, None, **march)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    for name in ("coupled", "reference_state", *_SESSION_OWNED):
        arguments.pop(name)
    return arguments


class _BlockSession:
    """The block-diagonal family's session: rebuilt from the transport operators, no Jacobian probe.

    It has no per-step hook, and :meth:`rebind` leaves it on the assembler it was opened with -- the
    behaviour of the block-diagonal continuation before sessions existed, kept deliberately (a ramp that
    re-points it at each station is a separate change, #386).
    """

    refresh_preconditioner = None

    def __init__(
        self,
        spec: BlockDiagonal,
        coupled: CoupledRANS,
        *,
        jacobian_production_viscosity: bool,
        on_build: Callable[[NewtonStrategy], NewtonStrategy] | None,
    ) -> None:
        self._spec = spec
        self._coupled = coupled
        self._production_viscosity = jacobian_production_viscosity
        self._on_build = on_build

    def build(self, state: jnp.ndarray, **march: object) -> NewtonStrategy:
        return self._build(state, march, track=True)

    def refresh(
        self, state: jnp.ndarray, previous: NewtonStrategy, **march: object
    ) -> NewtonStrategy:
        # Re-derives the k/omega hierarchies on their reused coarsening and rebuilds the shift's
        # transport time scale, carrying the flow block and the shift's coordinate factor over.
        return self._finish(self._step(state, _march_keywords(march), reuse=previous.shift_policy))

    def rebind(self, coupled: CoupledRANS) -> None:
        del coupled

    def _build(self, state: jnp.ndarray, march: dict, *, track: bool) -> NewtonStrategy:
        del track  # there is no hook to wire
        return self._finish(self._step(state, _march_keywords(march), reuse=None))

    def _step(
        self, state: jnp.ndarray, keywords: dict, *, reuse: CoupledShiftPolicy | None
    ) -> NewtonStrategy:
        coupled = self._coupled
        step_limit, step_projection = _k_positivity_guards(
            coupled, keywords.pop("positivity_floor"), keywords.pop("positivity_projection")
        )
        policy = _coupled_shift_policy(
            coupled,
            state,
            self._spec.resolved_scalar(),
            reuse,
            *_resolved_shift(keywords.pop("shift")),
            **self._spec.flow_block_options(),
        )
        regime, krylov_solver = _resolved_linear_solve(
            keywords.pop("linear_solve"), _BLOCK_LINEAR_SOLVE
        )
        dual_time = keywords.pop("dual_time")
        return _coupled_step(
            coupled,
            state,
            policy,
            regime=regime,
            krylov_solver=krylov_solver,
            dual_time=dual_time,
            step_limit=step_limit,
            step_projection=step_projection,
            jacobian_production_viscosity=self._production_viscosity,
            **keywords,
        )

    def _finish(self, step: NewtonStrategy) -> NewtonStrategy:
        return step if self._on_build is None else self._on_build(step)


class _MaterializedSession:
    """The materialized-Jacobian family's session: one probe, one inverse and one refresh hook.

    Everything expensive or identity-bearing is created at most once. The probe (a colouring plan and
    its de-compression map, the largest allocation a three-dimensional case makes) and the refresh hook
    are created on first use; the inverse is fitted on the first :meth:`build`, at that build's
    assembler, state and ``build_beta``, and every later build glues that same object in. The two
    callables handed to the march -- :attr:`refresh_preconditioner` and the mid-step refresh -- are created
    when the session is opened, so every step built from it carries the identical objects.
    """

    def __init__(
        self,
        spec: MaterializedJacobian,
        coupled: CoupledRANS,
        *,
        jacobian_production_viscosity: bool,
        observer: Callable[[RefreshTiming], None] | None,
        reports: dict[str, Callable[[str], None]],
        on_build: Callable[[NewtonStrategy], NewtonStrategy] | None,
        precondition_wrapper: Callable[[Callable], Callable] | None,
        inverse_wrapper: Callable[[str, Callable], Callable] | None,
    ) -> None:
        self._spec = spec
        self._coupled = coupled
        self._production_viscosity = jacobian_production_viscosity
        self._observer = observer
        self._reports = reports
        self._on_build = on_build
        self._inverse_wrapper = inverse_wrapper
        self._probe: CoupledJacobianProbe | None = None
        self._hook: Callable | None = None
        self._preconditioner: object | None = None

        def refresh_preconditioner(active_step: NewtonStrategy, state: jnp.ndarray) -> None:
            self._refresh_hook()(active_step, state)

        def refresh_at(iterate: jnp.ndarray) -> None:
            self._refresh_hook().refresh_at(iterate)

        self._refresh_at = refresh_at
        self.refresh_preconditioner = (
            refresh_preconditioner
            if precondition_wrapper is None
            else precondition_wrapper(refresh_preconditioner)
        )

    def build(self, state: jnp.ndarray, **march: object) -> NewtonStrategy:
        return self._build(state, march, track=True)

    def refresh(
        self, state: jnp.ndarray, previous: NewtonStrategy, **march: object
    ) -> NewtonStrategy:
        del previous  # the shared inverse is re-fitted in place, not re-derived from the old step
        step = self._build(state, march, track=True)
        if self._hook is not None:
            # The standing inverse was fitted before the march moved; force the next refresh to be full.
            self._hook.rebind(self._coupled)
        return step

    def rebind(self, coupled: CoupledRANS) -> None:
        current = self._coupled
        if (
            coupled.layout.size != current.layout.size
            or coupled.layout.n_fields != current.layout.n_fields
        ):
            raise ValueError(
                "a session can only be re-pointed at another companion of the SAME case: its colouring "
                f"probe was built for a {current.layout.n_fields}-field state of size "
                f"{current.layout.size}, and this assembler has {coupled.layout.n_fields} fields and "
                f"size {coupled.layout.size}."
            )
        self._coupled = coupled
        if self._hook is not None:
            self._hook.rebind(coupled)

    def _build(self, state: jnp.ndarray, march: dict, *, track: bool) -> NewtonStrategy:
        keywords = _march_keywords(march)
        if _is_traced((self._coupled, state)):
            raise ValueError(
                "a materialized-Jacobian preconditioner is assembled off the jit path from concrete "
                "arrays, so it cannot be built under jax.grad (or any JAX transform). Build the step "
                "with concrete parameters outside the transform and pass it as `strategy`; the "
                "adjoint reuses the same frozen inverse, so the gradient is unchanged."
            )
        coupled = self._coupled
        # Before any inverse is fitted: a misconfigured floor is a caller mistake, and the multigrid
        # families import an optional dependency the check must not wait on.
        step_limit, step_projection = _k_positivity_guards(
            coupled, keywords.pop("positivity_floor"), keywords.pop("positivity_projection")
        )
        base = _monolithic_shift_source(coupled, state, *_resolved_shift(keywords.pop("shift")))
        if self._preconditioner is None:
            self._preconditioner = self._fit(coupled, state, base)
        regime, krylov_solver = _resolved_linear_solve(
            keywords.pop("linear_solve"),
            _FACTORIZATION_LINEAR_SOLVE
            if isinstance(self._spec.inverse, CompleteLu)
            else _VCYCLE_LINEAR_SOLVE,
        )
        dual_time = keywords.pop("dual_time")
        if (
            track
            and dual_time is not None
            and dual_time.refresh_on_cycles is not None
            and keywords["inner_refresh"] is None
        ):
            keywords["inner_refresh"] = self._refresh_at
        step = _monolithic_factor_step(
            coupled,
            state,
            base,
            self._preconditioner,
            regime=regime,
            krylov_solver=krylov_solver,
            dual_time=dual_time,
            step_limit=step_limit,
            step_projection=step_projection,
            jacobian_production_viscosity=self._production_viscosity,
            **keywords,
        )
        return step if self._on_build is None else self._on_build(step)

    def _groups(self) -> FieldGroups:
        # `[u, v, w, p]` (the saddle) leads, `[k, omega]` (the transported scalars) trail.
        return FieldGroups.split_before(self._coupled.layout, "k")

    def _probe_for(self) -> CoupledJacobianProbe:
        if self._probe is None:
            self._probe = CoupledJacobianProbe.build(
                self._coupled,
                **self._spec.probe.settings(),
                # A split never reads the flow-by-[k, omega] triangle, so its probe need not store it.
                active_rows=(
                    self._groups().active_rows()
                    if isinstance(self._spec.inverse, FieldSplit)
                    else None
                ),
                # The preconditioner must be assembled from the operator the Krylov solve applies.
                production_viscosity_frozen=self._production_viscosity,
            )
        return self._probe

    def _refresh_hook(self) -> Callable:
        if self._hook is None:
            beta_floor = self._spec.beta_floor
            self._hook = _beta_tracking_refresh(
                self._coupled,
                # The reach settings are read from the probe passed below; these are not consulted.
                stencil_reach=3,
                every_step=isinstance(self._spec.inverse, CompleteLu),
                observer=self._observer,
                probe=self._probe_for(),
                **({} if beta_floor is None else {"beta_floor": beta_floor}),
            )
        return self._hook

    def _fit(self, coupled: CoupledRANS, state: jnp.ndarray, base: CoupledShiftPolicy) -> object:
        probe = self._probe_for()
        probed = probe.narrow(coupled)
        frozen = jax.lax.stop_gradient(state)

        def matvec(v):
            return _jacobian_matvec(probed, frozen, v)

        build_beta = _BUILD_BETA if self._spec.build_beta is None else self._spec.build_beta
        shift = _frozen_shift_diagonal(base, build_beta, state)
        inverse = self._spec.inverse
        if isinstance(inverse, CompleteLu):
            return MonolithicLuPreconditioner.build(matvec, probe.plan, shift, **inverse.settings())

        def batched_matvec(seeds):
            return _batched_jacobian_matvec(probed, frozen, seeds)

        probing = {
            "batched_matvec": batched_matvec,
            "probe_batch_size": _PROBE_BATCH_SIZE,
            "structure": probe.structure,
        }
        if isinstance(inverse, MonolithicVCycle):
            return MonolithicAmgPreconditioner.build(
                matvec, probe.plan, shift, **inverse.settings(), **probing
            )
        return FieldSplitAmgPreconditioner.build(
            matvec,
            probe.plan,
            shift,
            self._groups(),
            leading_inverse=self._block_inverse("leading"),
            trailing_inverse=self._block_inverse("trailing"),
            **probing,
        )

    def _block_inverse(self, role: str) -> Callable:
        spec = getattr(self._spec.inverse, role)
        factory = spec.bound(report=self._reports[role]) if role in self._reports else spec
        return factory if self._inverse_wrapper is None else self._inverse_wrapper(role, factory)


def open_session(
    preconditioner: BlockDiagonal | MaterializedJacobian | None,
    coupled: CoupledRANS,
    *,
    jacobian_production_viscosity: bool = False,
    observer: Callable[[RefreshTiming], None] | None = None,
    reports: dict[str, Callable[[str], None]] | None = None,
    on_build: Callable[[NewtonStrategy], NewtonStrategy] | None = None,
    precondition_wrapper: Callable[[Callable], Callable] | None = None,
    inverse_wrapper: Callable[[str, Callable], Callable] | None = None,
) -> PreconditionerSession:
    """Open a session for ``preconditioner`` on ``coupled``, doing no array work until it builds.

    Parameters
    ----------
    preconditioner : BlockDiagonal, MaterializedJacobian or None
        What preconditions the march. ``None`` is :class:`BlockDiagonal` with every setting unset.
    coupled : CoupledRANS
        The assembler the session builds on until it is re-pointed with ``rebind``.
    jacobian_production_viscosity : bool
        Whether the steps this session builds differentiate the frozen-production stand-in of the
        residual (see :func:`frozen_production_viscosity`). The session's probe is built from the same
        stand-in, which is why it is fixed here rather than chosen per build: a preconditioner assembled
        from a different matrix than the one the Krylov solve applies preconditions the wrong problem.
    observer : callable, optional
        ``(timing: RefreshTiming) -> None``, told what each refresh of a materialized inverse did and
        what it cost.
    reports : dict, optional
        Where each field-split block inverse sends its build record, keyed ``"leading"`` /
        ``"trailing"``. Only for a :class:`FieldSplit` whose inverse keeps a record.
    on_build : callable, optional
        ``step -> step``, applied to every step the session builds, for instrumenting a driver.
    precondition_wrapper : callable, optional
        ``hook -> hook``, wrapping the per-step :attr:`~PreconditionerSession.refresh_preconditioner` once, so
        a driver can observe each call. The session's ``rebind`` is unaffected by it.
    inverse_wrapper : callable, optional
        ``(role, factory) -> factory``, wrapping a field-split block-inverse factory before it is used,
        for a driver that must see the block an inverse is fitted to.

    Returns
    -------
    PreconditionerSession
        The session.

    Raises
    ------
    TypeError
        If ``preconditioner`` is not a spec, or a setting is given that the chosen family cannot use.
    """
    spec = BlockDiagonal() if preconditioner is None else preconditioner
    if isinstance(spec, BlockDiagonal):
        given = [
            name
            for name, value in (
                ("observer", observer),
                ("reports", reports),
                ("precondition_wrapper", precondition_wrapper),
                ("inverse_wrapper", inverse_wrapper),
            )
            if value is not None
        ]
        if given:
            raise TypeError(
                f"{given} have nothing to act on in the block-diagonal family: it has no per-step "
                "refresh hook and no field-split block inverses."
            )
        return _BlockSession(
            spec,
            coupled,
            jacobian_production_viscosity=jacobian_production_viscosity,
            on_build=on_build,
        )
    if not isinstance(spec, MaterializedJacobian):
        raise TypeError(
            "preconditioner must be BlockDiagonal(...) or MaterializedJacobian(...), got "
            f"{type(spec).__name__}."
        )
    reports = dict(reports or {})
    if (reports or inverse_wrapper is not None) and not isinstance(spec.inverse, FieldSplit):
        raise TypeError(
            "reports and inverse_wrapper address field-split block inverses, and "
            f"{type(spec.inverse).__name__} has none."
        )
    unknown = sorted(set(reports) - {"leading", "trailing"})
    if unknown:
        raise TypeError(f"reports are keyed 'leading' / 'trailing', got {unknown}.")
    for role, sink in reports.items():
        getattr(spec.inverse, role).bound(report=sink)  # refuse a family with no record now
    return _MaterializedSession(
        spec,
        coupled,
        jacobian_production_viscosity=jacobian_production_viscosity,
        observer=observer,
        reports=reports,
        on_build=on_build,
        precondition_wrapper=precondition_wrapper,
        inverse_wrapper=inverse_wrapper,
    )


def coupled_step(
    coupled: CoupledRANS,
    reference_state: jnp.ndarray,
    *,
    preconditioner: BlockDiagonal | MaterializedJacobian | None = None,
    jacobian_production_viscosity: bool = False,
    globalization: Globalization = DEFAULT_GLOBALIZATION,
    dual_time: DualTimeLoop | None = None,
    linear_solve: LinearSolveSettings | lx.AbstractLinearSolver | None = None,
    shift: ShiftSettings | None = None,
    inner_observer: Callable[..., None] | None = None,
    inner_refresh: Callable[[jnp.ndarray], None] | None = None,
    positivity_floor: float = 0.0,
    positivity_projection: bool = True,
    jacobian_gradient_sweeps: int | None = None,
) -> NewtonStrategy:
    """Build the coupled march's Newton step with its preconditioner **frozen** at ``reference_state``.

    The one builder for every preconditioner family: which family, and its settings, is the
    ``preconditioner`` value; everything else configures the march and means the same thing whichever
    preconditioner is chosen. The step is frozen -- nothing re-fits the inverse as the march moves -- which
    is what a differentiated solve and a fixed-point test want. A march that should keep its
    preconditioner current opens a session instead (:func:`open_session`).

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled residual assembler.
    reference_state : jnp.ndarray
        The coupled state the preconditioner and the shift diagonal are frozen at.
    preconditioner : BlockDiagonal or MaterializedJacobian, optional
        What preconditions the shifted solve. ``None`` is :class:`BlockDiagonal` with every setting unset.
    jacobian_production_viscosity : bool
        Differentiate the frozen-production stand-in of the residual in the Krylov operator, and
        materialize the preconditioner from the same stand-in.
    globalization : Globalization
        How hard the march damps and what it does when a step misbehaves: the switched-evolution
        schedule, the escalation ladder, the divergence guard and the backtracking line search. Only the
        fields it sets are applied; an unset ``line_search`` takes this march's ten rungs, because the
        coupled residual's full step overshoots by orders of magnitude from the hybrid start. Beside a
        ``dual_time`` loop the escalation-ladder fields are refused, since a dual-time step has no ladder
        for them to reach.
    dual_time : DualTimeLoop or None
        Given, the march is **dual-time** (backward-Euler, :class:`~aquaflux.solve.DualTimeStep`): each
        outer timestep runs an inner Newton loop on the transient residual ``R + beta d (phi - phi_ref)``,
        so the measured steady residual is the honest discrete time derivative rather than
        ``beta x travel`` and a larger pseudo-timestep can be taken stably. ``None`` (default) is the
        single shifted step, which has no inner loop, so neither ``inner_observer`` nor
        ``inner_refresh`` can be given with it. The loop's ``refresh_on_cycles`` needs a refresh to fire:
        a materialized-Jacobian preconditioner's session supplies one, while a frozen step and a
        block-diagonal session have none, so there it is refused.
    linear_solve : LinearSolveSettings, lineax.AbstractLinearSolver or None
        The shifted solve. A :class:`LinearSolveSettings` moves the regime -- unset, restart ``120`` for the
        block-diagonal family, ``10`` for a complete LU and ``15`` for a multigrid V-cycle or a field
        split, each at a relative tolerance of ``0.3`` -- and keeps the stop in the progress measure the
        march judges the step by (the solve's :class:`~aquaflux.solve.Convergence` measure), not the
        Euclidean norm: the coupled residual's 2-norm is ~100% ``omega``, so a 2-norm stop halts while
        the flow-dominated part of the step is still coarse. ``max_restarts`` is the only bound on a single running solve
        and counts raw ``lineax`` restarts, which carry a fixed ``+2`` per solve; keep its corrected
        count strictly above ``retry.abort_above_cycles``, or a truncated solve is accepted instead of
        redone. A whole solver replaces the regime **and** the stopping measure, which is a larger
        change than it looks.
    shift : ShiftSettings or None
        How the pseudo-time shift diagonal is formed: its basis, where the velocity shift's parts come
        from, and how much harder the closure's rows are damped than the flow's (see
        :class:`ShiftSettings`). Unset, the full operator diagonal, damped uniformly. It changes only
        the path: the shift vanishes at the root.
    inner_observer : callable or None
        A per-inner-iteration hook forwarded to the dual-time step. Forward-only.
    inner_refresh : callable or None
        ``(iterate) -> None``, the mid-step rebuild the loop's ``refresh_on_cycles`` fires. A session
        build wires its own when this is unset. Forward-only.
    positivity_floor : float
        Absolute room in ``k`` the step limiter gives every cell, so a numerically dead cell cannot set
        the step length for all of them. ``0.0`` (default) is the plain fraction-to-the-boundary rule. A
        non-zero floor beside ``positivity_projection=True`` is refused, because the projection runs first
        and the limiter it feeds can then never bind.
    positivity_projection : bool
        Clip each cell's own ``k`` correction instead of shortening the whole step for the worst cell.
        ``True`` (default); ``False`` restores the global cap. Neither moves the converged root or its
        adjoint -- at a root the correction vanishes.
    jacobian_gradient_sweeps : int or None
        Cap the gradient reconstruction's sweeps in the copy of the residual the forward Jacobian is
        differentiated from, leaving the residual itself -- and so the root and the adjoint -- unchanged.
        Distinct from the probe's ``gradient_sweeps``, which narrows the copy the preconditioner
        materializes; either may be set without the other.

    Returns
    -------
    NewtonStrategy
        A :class:`~aquaflux.solve.PseudoTransientStep`, or a :class:`~aquaflux.solve.DualTimeStep` when
        ``dual_time`` is given. It carries no residual measure of its own: :func:`solve_coupled` hands
        it the measure its :class:`~aquaflux.solve.Convergence` names at every outer iteration.
    """
    session = open_session(
        preconditioner, coupled, jacobian_production_viscosity=jacobian_production_viscosity
    )
    return session._build(
        reference_state,
        {
            "globalization": globalization,
            "dual_time": dual_time,
            "linear_solve": linear_solve,
            "shift": shift,
            "inner_observer": inner_observer,
            "inner_refresh": inner_refresh,
            "positivity_floor": positivity_floor,
            "positivity_projection": positivity_projection,
            "jacobian_gradient_sweeps": jacobian_gradient_sweeps,
        },
        track=False,
    )


def production_cap_active(coupled: CoupledRANS, state: jnp.ndarray) -> jnp.ndarray:
    """Per-cell mask: where the k-production cap binds at ``state``.

    The cap is ``min(nu_t S^2, 10 beta* k omega)``, and with
    ``SSTTurbulence.explicit_production_limiter`` set its ``k`` is frozen in the Jacobian. Wherever
    this mask is ``True`` at a **converged** state, that freezing has removed a real term from the
    linearization, so the implicit-function-theorem adjoint no longer differentiates the residual
    that was solved.

    Uses :func:`~aquaflux.turbulence.production_and_limit`, the same expressions the residual forms,
    so this cannot clear a state the residual actually caps.

    Parameters
    ----------
    coupled : CoupledRANS
        The coupled assembler.
    state : jnp.ndarray
        A packed coupled state, shape ``(layout.size,)``.

    Returns
    -------
    jnp.ndarray
        Boolean per cell, shape ``(n_cells,)``.

    Notes
    -----
    A cell whose ``k`` has been driven to ~0 reports ``True`` for an arithmetic reason rather than a
    physical one: the limit carries ``maximum(k, 0)``, so it collapses to ~0 there and any positive
    production exceeds it. That is still a real Jacobian difference, but it is not "the strain is
    high" -- see :func:`~aquaflux.turbulence.production_and_limit`.
    """
    flow, k, omega = coupled.physical_fields(state)
    closure = coupled.turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega)
    production, limit = production_and_limit(
        closure.nu_t, closure.strain_rate, closure.omega, k, coupled.turbulence.model
    )
    return production > limit


def _reject_a_root_the_frozen_cap_invalidates(
    coupled: CoupledRANS, state: jnp.ndarray
) -> jnp.ndarray:
    """Refuse a converged state whose adjoint the frozen production cap has invalidated.

    A no-op unless ``explicit_production_limiter`` is set -- which is opt-in, and off by default. With
    it set, the returned state is guarded by :func:`equinox.error_if` on the cap being active
    anywhere: the forward fields would be perfectly good, and the **gradient through them silently
    wrong**, which is precisely the failure that must not be shipped quietly.

    The same discipline the positivity floors are held to -- a stabilization that alters the
    linearization is free only while it is inactive at the root, and something has to check rather
    than assume. ``error_if`` (not a Python ``if``) because the check must fire on the traced
    ``jax.grad`` path too, which is the only path where the damage is real.
    """
    if not coupled.turbulence.explicit_production_limiter:
        return state
    return eqx.error_if(
        state,
        jnp.any(production_cap_active(coupled, state)),
        "the coupled solve converged to a root at which the k-production cap is ACTIVE while "
        "`explicit_production_limiter=True`, so the cap's `k` is frozen in the Jacobian there. The "
        "fields are fine; any gradient taken through this root is NOT -- the "
        "implicit-function-theorem adjoint would linearize a different residual from the one solved, "
        "and would return a finite, wrong sensitivity. Build the turbulence with "
        "`explicit_production_limiter=False` (the default) for the exact operator, or, if the "
        "stabilization is genuinely needed for this forward solve, take no gradient through the "
        "result. `aquaflux.turbulence.production_cap_active` reports which cells bind.",
    )


class _ContinuationSource(Protocol):
    """Where the coupled march's :class:`~aquaflux.solve.NewtonStrategy` comes from, and how it re-freezes.

    :func:`solve_coupled` needs a continuation twice: once at the start, and again at each refresh, from
    a developed state. Those are one decision — *which* continuation this solve runs — and they were
    written as two independent two-way branches, one at the initial build and one inside the refresh
    loop. That is the shape that drifts: a change to how the continuation is built has to be made in two
    places and, when it is made in one, nothing fails. Behind this interface it is made once.

    Three implementations: the caller supplies the builder (:class:`_CallerBuiltContinuation`); a
    preconditioner session builds it (:class:`_SessionContinuation`, including the default
    block-diagonal one when nothing is named); or the caller finished the step and nothing will rebuild
    it (:class:`_FinishedContinuation`).

    Private because it is a decomposition, not an extension point: a caller who wants a different
    continuation passes a preconditioner, a session, the step, or a builder.

    Attributes
    ----------
    refresh_preconditioner : callable or None
        The per-step refresh hook this source brings with it -- a materialized session's -- or ``None``.
    """

    refresh_preconditioner: Callable[[NewtonStrategy, jnp.ndarray], None] | None

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        """The continuation to start the march with, frozen at ``state``."""
        ...

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        """Re-freeze at the developed ``state``.

        ``previous`` is the step being replaced, for an implementation that can reuse part of it. The
        residual measure is not the step's to keep: the march hands its own to every step it runs.
        """
        ...


@dataclasses.dataclass(frozen=True)
class _CallerBuiltContinuation:
    """A continuation the caller builds from the state, and rebuilds the same way at every refresh.

    The builder constructs it however it likes -- a complete-LU continuation materialized off the jit
    path, say -- so ``solve_coupled`` never learns how it is built, and an off-jit preconditioner can
    re-freeze without that knowledge leaking here. It follows that the builder owns the whole
    configuration: there is no keyword ``solve_coupled`` could forward into a closure it does not
    construct, which is why passing one alongside is refused rather than dropped.
    """

    builder: Callable[[jnp.ndarray], NewtonStrategy]
    #: A caller-built step brings its own refresh hook, if any, on its ``RefreshPolicy``.
    refresh_preconditioner = None

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        return self.builder(state)

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        del previous  # the builder re-derives everything from the state
        return self.builder(state)


@dataclasses.dataclass(frozen=True)
class _FinishedContinuation:
    """The caller handed over a finished step and no builder, so nothing here can re-freeze it.

    :meth:`~aquaflux.solve.RefreshPolicy.require_rebuildable` already refuses that combination when a
    refresh would run, so :meth:`refresh` is unreachable through the driver. It exists so the source is
    never ``None`` -- an optional strategy that three call sites must remember not to dereference is the
    kind of seam that eventually is -- and so that if it ever *is* reached, it says what is missing
    rather than raising ``AttributeError`` on ``None``.
    """

    refresh_preconditioner = None

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        raise TypeError(
            "no strategy to build: `solve_coupled` was given a finished `strategy`. This is a "
            "driver bug -- the supplied step should have been used directly."
        )

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        raise TypeError(
            "a refresh triggered but the explicit `strategy` cannot be rebuilt: pass "
            "`RefreshPolicy(builder=...)` so the solve can re-freeze it at each developed state."
        )


@dataclasses.dataclass(frozen=True)
class _SessionContinuation:
    """A continuation a preconditioner session builds and re-freezes -- the source that has configuration.

    It is the one source ``solve_coupled``'s ``preconditioner`` / ``reference_state`` /
    ``**strategy_kwargs`` describe: the preconditioner chose the session, and the march keywords are
    handed to every build and refresh the session makes.
    """

    session: PreconditionerSession
    reference_state: jnp.ndarray | None
    march: dict

    @property
    def refresh_preconditioner(self) -> Callable[[NewtonStrategy, jnp.ndarray], None] | None:
        return self.session.refresh_preconditioner

    def build(self, state: jnp.ndarray) -> NewtonStrategy:
        reference = state if self.reference_state is None else self.reference_state
        return self.session.build(reference, **self.march)

    def refresh(self, state: jnp.ndarray, previous: NewtonStrategy) -> NewtonStrategy:
        return self.session.refresh(state, previous, **self.march)


def _continuation_source(
    coupled: CoupledRANS,
    strategy: NewtonStrategy | None,
    refresh: RefreshPolicy,
    preconditioner: BlockDiagonal | MaterializedJacobian | PreconditionerSession | None,
    reference_state: jnp.ndarray | None,
    kwargs: dict,
) -> _ContinuationSource:
    """Pick the source, after refusing configuration whichever one is chosen cannot receive.

    **Why this refuses rather than ignores.** ``preconditioner`` / ``reference_state`` /
    ``**strategy_kwargs`` configure the continuation ``solve_coupled`` builds. On the two paths where
    it does not build one -- an explicit ``strategy``, or a ``RefreshPolicy(builder=...)`` -- they
    reached nothing at all, with no error and no log line: a caller asking for a dual-time loop or
    ``positivity_floor=1e-6`` got the library defaults and a march that looked like the one they
    configured. ``**kwargs`` is what made it silent, since it accepts every keyword and checks none, and
    that door is the main entry point's.

    A session owns ``jacobian_production_viscosity``, so it is accepted beside a spec (and passed to the
    session opened for it) and refused beside a session object. A materialized session brings its own
    per-step refresh hook, so a second one on the ``RefreshPolicy`` is refused rather than letting two
    hooks re-fit one inverse.
    """
    given = dict(kwargs)
    if preconditioner is not None:
        given["preconditioner"] = preconditioner
    if reference_state is not None:
        given["reference_state"] = reference_state
    if strategy is not None:
        _refuse(given, "`strategy`", "the step you passed already carries them")
        if refresh.builder is None:
            return _FinishedContinuation()
        return _CallerBuiltContinuation(refresh.builder)
    if refresh.builder is not None:
        _refuse(given, "`RefreshPolicy(builder=...)`", "the builder owns its own configuration")
        return _CallerBuiltContinuation(refresh.builder)
    march = dict(kwargs)
    production = march.pop("jacobian_production_viscosity", None)
    if preconditioner is None or isinstance(preconditioner, BlockDiagonal | MaterializedJacobian):
        session = open_session(
            preconditioner,
            coupled,
            **({} if production is None else {"jacobian_production_viscosity": production}),
        )
    elif production is not None:
        raise TypeError(
            "jacobian_production_viscosity belongs to the preconditioner session, which built its probe "
            "from it: pass it to open_session, not beside the session."
        )
    else:
        session = preconditioner
    if refresh.refresh_preconditioner is not None and session.refresh_preconditioner is not None:
        raise TypeError(
            "RefreshPolicy(refresh_preconditioner=...) was given beside a materialized-Jacobian "
            "preconditioner, whose session already re-fits its inverse before every step. Drop one."
        )
    return _SessionContinuation(session, reference_state, march)


def _refuse(given: dict, owner: str, why: str, solver: str = "solve_coupled") -> None:
    """Raise if any continuation setting was passed to a solve that cannot forward it."""
    if not given:
        return
    raise TypeError(
        f"{sorted(given)} configure the continuation `{solver}` builds, and {owner} was given, so "
        f"{why}. These would have been dropped silently. Pass them where the continuation is built "
        f"instead, or drop {owner}."
    )


#: The stopping test of a coupled solve given no :class:`~aquaflux.solve.Convergence`, and the base an
#: incomplete one is filled from. Row-scaled, because the coupled residual's Euclidean norm is almost
#: entirely the ``omega`` block and so judges nothing else.
_COUPLED_CONVERGENCE = Convergence(measure=RowScaled(), rtol=1e-10, atol=1e-12)


def solve_coupled(
    coupled: CoupledRANS,
    flow: jnp.ndarray | None = None,
    k: jnp.ndarray | None = None,
    omega: jnp.ndarray | None = None,
    *,
    strategy: NewtonStrategy | None = None,
    reference_state: jnp.ndarray | None = None,
    preconditioner: BlockDiagonal | MaterializedJacobian | PreconditionerSession | None = None,
    max_steps: int = 60,
    convergence: Convergence | None = None,
    adjoint_solver: lx.AbstractLinearSolver | None = None,
    refresh: RefreshPolicy = NO_REFRESH,
    step_control: StepControl | None = None,
    on_step: Callable[[StepReport], None] | None = None,
    on_checkpoint: Callable[[StepReport, jnp.ndarray], None] | None = None,
    retry: RetryPolicy = NO_RETRIES,
    on_retry: Callable[[str, int, float], None] | None = None,
    homotopy: ResidualHomotopy | None = None,
    station_step: Callable[[NewtonStrategy, int, bool], NewtonStrategy] | None = None,
    **strategy_kwargs: object,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the coupled RANS system ``R(u, p, k, omega) = 0`` by one monolithic Newton march.

    The march (:func:`~aquaflux.solve.newton_march`) drives :meth:`CoupledRANS.residual` to zero with
    the pseudo-transient step :func:`coupled_step` describes -- the coupled counterpart of the flow
    block's :func:`~aquaflux.flow.reused_flow_solve`. The root it reaches is handed to
    :func:`~aquaflux.solve.root_adjoint`, which makes the result reverse-differentiable by the coupled
    implicit-function-theorem adjoint (a single transpose solve on the unfrozen ``R_coupled``) -- the
    exact coupled sensitivity, rather than a differentiation of the segregated Picard iteration.

    **There is one march, however the solve is configured or observed.** It runs on ``stop_gradient``
    copies of the assembler and the initial state, so everything it does between steps -- a
    preconditioner re-fit or refresh, a step control, a retry, a callback -- is ordinary Python on
    concrete values that never reaches the gradient, and all of it works under ``jax.grad``. Observing a
    solve with ``on_step`` or ``on_checkpoint`` changes nothing about it. For the same reason it cannot
    run inside a traced program (``jax.jit``, ``jax.vmap``, ``jax.lax.scan``), where the values are
    abstract: call it outside those
    transforms.

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
        ``reference_state`` is given. It is never differentiated through: the root does not depend on
        where the march began.
    strategy : NewtonStrategy or None
        A pre-built Newton strategy (a :class:`~aquaflux.solve.PseudoTransientStep` or a
        :class:`~aquaflux.solve.DualTimeStep`, e.g. from :func:`coupled_step`); ``None`` builds one from
        the initial state. A pre-built step keeps the preconditioner it was built with, which under
        ``jax.grad`` means one frozen at those parameter values -- a change to the Krylov iteration,
        never to the gradient.
    reference_state : jnp.ndarray or None
        The coupled state to freeze the internally-built preconditioner at; defaults to the initial
        state. ⚠️ **Rejected**, not ignored, when the strategy is not built here -- see
        ``**strategy_kwargs``.
    preconditioner : BlockDiagonal, MaterializedJacobian, PreconditionerSession or None
        What preconditions the internally-built continuation. A spec opens a session private to this
        solve; ``None`` is :class:`~aquaflux.turbulence.BlockDiagonal` with every setting unset. Pass a
        session (:func:`open_session`) to share one inverse and its refresh hook across several solves
        -- the rungs of a Reynolds continuation, say. A materialized-Jacobian preconditioner re-fits its
        inverse before every step, from concrete values, so it too works under ``jax.grad``. Rejected on
        the same terms as ``reference_state``.
    max_steps : int
        Outer-step cap for **each** march segment (see ``refresh``).
    convergence : Convergence or None
        The stopping test, ``measure(R) <= atol + rtol * measure(R0)``, with ``R0`` the residual at the
        initial state. Unset fields take :class:`~aquaflux.solve.RowScaled`, ``rtol = 1e-10`` and
        ``atol = 1e-12``. The measure is the one the march steers by as well as the one it is judged in:
        every outer iteration builds it at the state it starts from and hands it to the step, whose line
        search, shift and linear solve all read it, and the final test is taken in it at the state
        reached. A :class:`~aquaflux.solve.RowScaled` measure therefore follows the developing flow; a
        :class:`~aquaflux.solve.BlockScaled` one keeps the scales of the initial state for the whole
        solve, across every refresh. It replaces whatever measure a supplied ``strategy`` was built with.
    adjoint_solver : lineax.AbstractLinearSolver, optional
        The linear solver for the **adjoint** (transpose) solve behind every ``jax.grad`` through this
        function -- the one solve the implicit-function-theorem gradient is built from, taken once at the
        converged state on the **unshifted** operator. It is a separate injection point from the forward
        march's solver because the two meet different operators: the Newton steps solve ``J + beta d``,
        which the pseudo-transient shift keeps diagonally dominant, while the transpose solve meets
        ``J`` itself with no shift to soften it, and needs its Krylov settings chosen for that. ``None``
        (default) uses :func:`~aquaflux.solve.default_linear_solver`, a tight restarted GMRES at
        ``lineax``'s own restart length and stagnation budget -- which a coupled 3D saddle can exhaust,
        raising a stagnation error whose suggested remedy (a longer restart, a larger stagnation budget)
        is reachable only through this argument. Build one with
        :func:`~aquaflux.solve.relative_residual_gmres`, which takes both.
    refresh : RefreshPolicy
        How the frozen preconditioner is kept current: the staleness ``trigger``, the refresh
        ``limit``, an optional ``builder`` that reconstructs the Newton step, and the per-step
        ``refresh_preconditioner`` hook. See :class:`~aquaflux.solve.RefreshPolicy` for each setting. The
        default refreshes nothing, which is a single-segment march.

        With a trigger set the march runs as a sequence of **segments**: each steps until the trigger
        fires, the preconditioner is re-derived at the state reached, and the next segment continues
        from there. The last segment, with no refresh left to spend, ignores the trigger and marches to
        convergence or to ``max_steps``.

        This solve supplies the trigger's staleness measure itself (:func:`eddy_viscosity_drift`,
        ``nu_t`` being what the frozen k/omega transport operators are assembled from), **re-based at
        every refresh** so each segment reports drift from its own freeze state; carrying one measure
        across segments would keep reporting drift the refresh had just absorbed and re-fire at once.

        **Each segment restarts the damping ramp, and that is load-bearing.** The ramp is defined
        relative to where a segment began, so a segment handed a new state must measure its **own**
        reference residual; carrying the pre-refresh reference across -- to keep the ramp "continuous",
        which looks like the more principled choice -- makes ``beta`` mean something measured against a
        state the march has left. Two corrections worth keeping: a refresh **rebuilds** the shift's
        transport time scale at the developed state while **carrying** its coordinate factor
        ``d(phi)/d(w)`` frozen (rebuilding the whole product was measured to freeze the march), so the
        justification is *not* that a grown ``d`` needs a fresh ``beta``; and with refreshes every few
        steps the residual ratio never falls far below one, so **``beta`` stays pinned near ``beta0``**
        for the whole march instead of ramping down -- a different damping level has to come from
        ``beta0``, not from expecting the ramp to find it.

        ``convergence`` means the same thing with and without a refresh: the stopping target is measured once,
        at the initial state, and held across every segment, so a refreshed solve stops at exactly the
        residual an unrefreshed one would, for any number of refreshes. ``max_steps`` applies to
        **each** segment, so a refreshed solve may take up to ``refresh.segments * max_steps`` steps --
        the budget is deliberately not split, since either segment may legitimately need the full
        allowance.

        **Why refresh:** the frozen scalar preconditioners go stale as the flow separates. Their coarse
        space was fitted to the pre-separation operator, so re-deriving them at the developed state cuts
        the shifted solve's outer Krylov count; the flow block does *not* go stale and is carried over
        untouched. The refresh costs one extra compilation of the shifted solve, which that saving repays
        within a step or two at mesh sizes where this matters. The win appears only once the flow
        separates -- refreshing at a pre-separation state buys nothing and can cost.
        :class:`~aquaflux.solve.CycleGrowthTrigger` therefore gates on the residual having fallen as
        well as on the cost having risen; :class:`~aquaflux.solve.CoefficientDriftTrigger` needs no such
        gate, because an undeveloped flow is one whose coefficients have not moved.

        A refresh re-derives the preconditioner from a concrete copy of a mid-march state, so it works
        under ``jax.grad``. The gradient is the one an unrefreshed solve gives: the preconditioner only
        accelerates the Krylov iteration, and every march reaches the same root.
    step_control : StepControl, optional
        Reshapes the strategy each iteration from the previous step's report. **A dual-time march**
        (``dual_time`` given, a :class:`~aquaflux.solve.DualTimeStep`) given no control **defaults to**
        :class:`~aquaflux.solve.DualTimeControl`, the Courant ramp that grows the pseudo-timestep while
        the inner loop stays comfortable (measured ~4× fewer outer steps to a developed recirculation on
        a cold-start pitzDaily ramp than the residual-keyed schedule). Pass an explicit control (e.g.
        :class:`~aquaflux.solve.ResidualRatioDualTimeControl`) to override, or pass one built with
        different knobs. The single-step march (``dual_time`` unset, the default) gets no default
        control.
    on_step : callable, optional
        Called with each :class:`~aquaflux.solve.StepReport` as the march produces it -- the seam for
        logging a long solve's progress and cost. The refresh trigger reads the same reports. It only
        observes: a solve given one takes exactly the steps a solve without one takes.
    on_checkpoint : callable, optional
        Called with ``(report, state)`` after each step, for saving intermediate states of a long march.
        Kept separate from ``on_step`` so the report history stays purely numeric and a refresh trigger
        remains replayable offline (see :func:`~aquaflux.solve.newton_march`). Note the *state* here is
        the solved-variable state, not the physical fields -- map it with
        :meth:`CoupledRANS.physical_fields`.
    retry : RetryPolicy
        When and how the march redoes a bad step: the cost and step-length escalation thresholds, the
        ``β`` factor and escalation limit, and the optional tighter linear solver. See
        :class:`~aquaflux.solve.RetryPolicy` for each setting and :func:`~aquaflux.solve.newton_march`
        for the order they fire in. The default policy retries nothing, which is byte-identical to a
        march without retries.

        Both escalation triggers need a ``β``-carrying step control, since escalation works by scaling
        the shift leaf the control sets. The two failures they cover are genuinely different: a high
        cycle count is the stiff low-``β`` operator (the other cause of a high count, a stale
        preconditioner, is pre-empted by a β-mismatch refresh instead), while a collapsed step length is
        a correction that cannot be followed at all and is invisible to the cost trigger because those
        solves are *cheap*.

        ``solver`` covers what more damping cannot: with an inexact preconditioner (a threshold-ILU) the
        loose default Krylov solve can return a non-finite correction on the stiff operator an
        aggressive overshoot produces, where an exact complete-LU returns a finite one. The step is
        redone from the same state at the tighter tolerance -- a β-tracked factorization is already
        fresh, so only the Krylov solve changes -- which recovers it while keeping the accepted
        trajectory, and pays the tighter solve on the few trouble steps rather than on every step. The
        exact-LU path never diverges and needs none.
    on_retry : callable, optional
        ``(reason, attempt, beta) -> None``, forwarded to
        :func:`~aquaflux.solve.newton_march`: called before a step is redone, with why. A log without it
        shows a step's work twice and never says what triggered the redo.
    homotopy : ResidualHomotopy, optional
        Walk a sequence of related problems within this one march, ending at the target (see
        :func:`~aquaflux.solve.newton_march`). The solve converges only once the homotopy has arrived:
        a converged intermediate station is not an answer.
    station_step : callable, optional
        ``(step, station, arrived) -> step``, forwarded to
        :func:`~aquaflux.solve.newton_march`: reshape the Newton step for the continuation station it
        is about to run. The use it exists for is a per-block damping that differs between a homotopy's
        intermediate stations and its target -- measured on a viscosity ramp, the closure's rows want a
        much larger share of the shift while the ramp is walking than once the target problem is
        reached, and no signal a shift policy can read for itself distinguishes those. It reshapes the
        path, not the root, and ``None`` (the default) is byte-identical.
        ⚠️ It must swap **array** leaves over a fixed structure or every station recompiles the solve.
    **strategy_kwargs
        The march settings of :func:`coupled_step` (``dual_time``, ``positivity_floor``, ``linear_solve``,
        ...), handed to every build and refresh of the internally-built continuation; an unknown
        keyword raises. A preconditioner setting belongs on the ``preconditioner`` spec instead.
        Notably ``dual_time`` selects the **dual-time** (backward-Euler) march
        (:class:`~aquaflux.solve.DualTimeStep`) — an inner Newton loop per outer timestep whose measured
        steady residual is the honest discrete time derivative rather than ``beta x travel``.
        Leaving it unset (the default) is the single-step continuation.
        ``jacobian_production_viscosity`` is accepted here only beside a spec, whose session is built
        from it.

        ⚠️ **These, ``preconditioner`` and ``reference_state`` configure the continuation this function builds,
        so they are only accepted when it builds one.** Supply a ``strategy`` or a
        ``RefreshPolicy(builder=...)`` and that object owns its whole configuration; passing any of them
        alongside raises :exc:`TypeError` naming which ones. They used to be **dropped in silence** on
        both of those paths — a solve asked for a dual-time loop and ``positivity_floor=1e-6`` ran the
        library defaults, with no error and no log line — and ``**kwargs`` is what made it quiet, since
        it accepts every keyword and checks none. That door is this function's, which makes it the one
        place in the package where a misplaced setting cannot be caught by reading a signature.

    Returns
    -------
    tuple of jnp.ndarray
        The converged ``(flow, k, omega)``.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the march ends short of its stopping target -- it exhausted ``max_steps`` in its last
        segment, stopped on a stalled positivity cap, went non-finite, or a homotopy never reached its
        target. The implicit-function-theorem adjoint is valid only at a root, so a state that is not
        one is refused rather than returned.
    ValueError
        If called from inside a traced program -- ``jax.jit``, ``jax.vmap``, or a traced loop such as
        ``jax.lax.scan``.
    TypeError
        If a strategy setting is passed where the strategy is not built here.
    """
    refuse_a_transform_the_march_cannot_run_in((coupled, flow, k, omega), caller="solve_coupled")
    # The march runs on a stopped copy of the assembler, so every build, re-fit, refresh and step below
    # sees concrete arrays even under `jax.grad`. The derivative is attached at the root afterwards.
    frozen = stop_array_gradients(coupled)
    # One decision -- which continuation this solve runs -- made once, for the initial build and every
    # refresh alike, and refusing any setting the chosen source cannot receive rather than dropping it.
    # Made before anything else, so a misconfiguration raises before any work is done.
    source = _continuation_source(
        frozen,
        strategy,
        refresh,
        preconditioner,
        stop_array_gradients(reference_state),
        strategy_kwargs,
    )
    # A materialized session re-fits its inverse before every step; a caller-built step brings its own.
    refresh_preconditioner = refresh.refresh_preconditioner or source.refresh_preconditioner
    if flow is None or k is None or omega is None:
        flow, k, omega = hybrid_initialize(frozen.momentum, frozen.turbulence)
    # `flow, k, omega` are the physical initial condition; map into the solved-variable space (the
    # identity for DirectScalars, log for LogScalars) so the march iterates on the right unknown.
    state = frozen.state_from_physical(*stop_array_gradients((flow, k, omega)))
    # A refresh rebuilds the step; a caller-supplied step with no builder leaves it nothing to rebuild
    # WITH, so the refresh would silently never happen. The policy owns that check.
    refresh.require_rebuildable(strategy)
    if strategy is None:
        strategy = source.build(state)

    # A dual-time march with no caller control defaults to the Courant ramp (see the helper): it grows
    # the pseudo-timestep while the inner loop stays comfortable, carried across the refreshes below,
    # reaching a developed recirculation in far fewer outer steps than the residual-keyed schedule.
    step_control = default_dual_time_control(step_control, strategy)

    # The measure is built once per outer iteration by the march, from the one builder made here, and
    # the global progress reference is taken in it at the initial state. A `BlockScaled` builder holds
    # the initial state's scales for the whole solve, so a refresh cannot re-base it (which would put the
    # target, measured once here, out of reach); a `RowScaled` one re-reads the step it is handed, so a
    # refreshed step's diagonals are used. `frozen.residual` is passed as a bound method (a pytree), not
    # a lambda, so its arrays ride as dynamic leaves and every step within a segment is a
    # compilation-cache hit.
    convergence = (
        _COUPLED_CONVERGENCE
        if convergence is None
        else convergence.filled_from(_COUPLED_CONVERGENCE)
    )
    norm_builder = convergence.measure._builder(_CoupledMeasures(frozen), state)
    reference_norm = float(norm_builder(strategy, state)(frozen.residual(state)))
    # `refresh.limit` refreshes means `refresh.segments` segments: the segment *after* the last refresh
    # must still be marched, or the newly-refreshed preconditioner would never be used.
    control_state: object = None
    for segment in range(refresh.segments):
        result = newton_march(
            strategy,
            frozen.residual,
            state,
            max_steps=max_steps,
            rtol=convergence.rtol,
            atol=convergence.atol,
            reference_norm=reference_norm,
            # The last segment has no refresh left to spend, so it marches to convergence or to
            # `max_steps` rather than stopping where the trigger fires -- with no second solve after
            # the march, a segment stopped there would end the solve short of the root.
            trigger=None if refresh.is_last_segment(segment) else refresh.trigger,
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
            drift_measure=eddy_viscosity_drift(frozen, state),
            norm_builder=norm_builder,
            refresh_preconditioner=refresh_preconditioner,
            retry=retry,
            on_retry=on_retry,
            homotopy=homotopy,
            station_step=station_step,
        )
        state = result.state
        control_state = result.control_state
        if not result.triggered or refresh.is_last_segment(segment):
            break
        # Re-freeze at the developed state -- how the re-freeze is done is the source's, and is the same
        # choice made at the initial build above rather than a second one made here.
        strategy = source.refresh(state, strategy)

    # Judge convergence in the measure the march steered by, built at the state reached.
    residual_norm = float(norm_builder(strategy, state)(frozen.residual(state)))
    target = convergence.atol + convergence.rtol * reference_norm
    arrived = homotopy is None or result.converged
    if not (math.isfinite(residual_norm) and residual_norm <= target and arrived):
        raise eqx.EquinoxRuntimeError(
            f"solve_coupled did not converge: the march ended at residual {residual_norm:.3e} against "
            f"a target of {target:.3e} (atol + rtol*||R0||)"
            + ("" if arrived else ", and its homotopy never reached the target problem")
            + ". The implicit-function-theorem adjoint is only valid at a converged root, so the fields "
            "and any gradient built on them would be silently wrong. Raise max_steps, loosen the "
            "tolerances, or strengthen the globalization (a dual-time loop, a retry policy)."
        )
    root = _reject_a_root_the_frozen_cap_invalidates(frozen, state)
    root = root_adjoint(
        assembler_residual,
        root,
        coupled,
        adjoint_solver=adjoint_solver,
        adjoint_preconditioner=strategy.adjoint_preconditioner(),
    )
    return coupled.physical_fields(root)


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

    def shift_term(self, phi: jnp.ndarray, residual: jnp.ndarray | None = None) -> ShiftTerm:
        """The augmented block-diagonal shift and the bordered preconditioner at ``phi``."""
        inner = phi[: self.inner.layout.size]
        inner_term = self.inner.shift_term(
            inner, None if residual is None else residual[: self.inner.layout.size]
        )
        diagonal = jnp.append(inner_term.diagonal, 0.0)

        def make_preconditioner(relaxation: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
            coupled_m = inner_term.make_preconditioner(relaxation)
            return _bordered_preconditioner(lambda _w: coupled_m, self.force, self.average)(phi)

        # The bordered constraint row carries no shift (its diagonal entry is zero above), so it needs
        # no per-row scale; appending 1.0 keeps the multiplier the same shape as the diagonal.
        row_relaxation = (
            None
            if inner_term.row_relaxation is None
            else lambda relaxation: jnp.append(inner_term.row_relaxation(relaxation), 1.0)
        )
        return ShiftTerm(diagonal, make_preconditioner, row_relaxation)

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
    preconditioner: BlockDiagonal | None = None,
    globalization: Globalization = DEFAULT_GLOBALIZATION,
    dual_time: DualTimeLoop | None = None,
    linear_solve: LinearSolveSettings | lx.AbstractLinearSolver | None = None,
    shift: ShiftSettings | None = None,
    inner_observer: Callable[..., None] | None = None,
    inner_refresh: Callable[[jnp.ndarray], None] | None = None,
    positivity_floor: float = 0.0,
    positivity_projection: bool = True,
    jacobian_gradient_sweeps: int | None = None,
    jacobian_production_viscosity: bool = False,
) -> NewtonStrategy:
    """The pseudo-transient continuation step for the **mass-flow-constrained** coupled Newton solve.

    The block-diagonal step :func:`coupled_step` builds, with its :class:`CoupledShiftPolicy` bordered by
    the mass-flow constraint (:class:`_MassFlowBorderedPolicy`), so it drives the augmented
    ``[flow..., k, omega, beta]`` system where ``beta`` is a Lagrange multiplier for ``<U_dir> =
    target``. The march settings are :func:`coupled_step`'s (including ``globalization`` /
    ``linear_solve`` / ``shift``); ``flow_direction`` selects the constrained velocity component. Its
    ``linear_solve`` regime is :func:`coupled_step`'s too, but this path defaults to
    :data:`_CONSTRAINED_LINEAR_SOLVE` rather than :data:`_BLOCK_LINEAR_SOLVE`: the restart regime is the
    same, and the **tolerance differs because the measure does** -- the linear solve stops in the march's
    progress measure, which on this path is by default the plain Euclidean norm (see
    :func:`solve_coupled_mass_flow`), so a Euclidean tolerance is what it takes.

    Routes through :func:`_coupled_step` like its siblings, so the globalization is the same one they
    run. What is genuinely its own is the policy, wrapped in the bordered constraint policy.

    ``preconditioner`` must be a :class:`~aquaflux.turbulence.BlockDiagonal` (``None`` takes
    ``BlockDiagonal()``). The constraint borders a block-diagonal policy, whose composed preconditioner
    the Schur elimination of ``beta`` wraps; a :class:`~aquaflux.turbulence.MaterializedJacobian` inverts
    a Jacobian that has no border row, so it is refused rather than applied to the wrong system.

    Raises
    ------
    TypeError
        If ``preconditioner`` is neither ``None`` nor a ``BlockDiagonal``.
    """
    if preconditioner is None:
        preconditioner = BlockDiagonal()
    if not isinstance(preconditioner, BlockDiagonal):
        raise TypeError(
            f"the mass-flow-constrained step borders a block-diagonal preconditioner, so preconditioner "
            f"must be a BlockDiagonal, not {type(preconditioner).__name__}: a materialized Jacobian "
            "has no constraint row for the bordered solve to eliminate."
        )
    # Checked before any preconditioner is fitted: a misconfigured floor is a caller mistake, not a
    # reason to pay for a build that the raise below would then discard.
    step_limit, step_projection = _k_positivity_guards(
        coupled, positivity_floor, positivity_projection
    )
    # No `reuse` here: the mass-flow-constrained path has no staged-refresh driver (there is no
    # a refresh on `solve_coupled_mass_flow`), so a policy is always built from scratch. Thread
    # `reuse` through if that driver is ever added -- the bordered policy wraps this one unchanged.
    policy = _coupled_shift_policy(
        coupled,
        reference_state,
        preconditioner.resolved_scalar(),
        None,
        *_resolved_shift(shift),
        **preconditioner.flow_block_options(),
    )
    force, average = _coupled_constraint_vectors(coupled, flow_direction)
    bordered = _MassFlowBorderedPolicy(policy, force, average)
    regime, krylov_solver = _resolved_linear_solve(linear_solve, _CONSTRAINED_LINEAR_SOLVE)
    return _coupled_step(
        coupled,
        reference_state,
        bordered,
        regime=regime,
        globalization=globalization,
        dual_time=dual_time,
        krylov_solver=krylov_solver,
        inner_observer=inner_observer,
        inner_refresh=inner_refresh,
        step_limit=step_limit,
        step_projection=step_projection,
        jacobian_gradient_sweeps=jacobian_gradient_sweeps,
        jacobian_production_viscosity=jacobian_production_viscosity,
    )


class _MassFlowConstrainedResidual(eqx.Module):
    """The coupled residual bordered with ``<U_dir> - target``, as a two-argument residual.

    The coupled assembler arrives as the parameter ``theta`` rather than being captured, so the
    implicit-function-theorem adjoint returns its cotangent and the constrained solve is
    reverse-differentiable in it. Everything the residual reads from the assembler therefore comes
    from ``theta``, including the cell volumes.

    A **module rather than a closure**, for the same reason as its flow-block counterpart
    ``_BulkVelocityResidual`` in ``aquaflux/flow/mean_velocity.py``: the march compiles its step with
    the residual as an argument, so a closure built per solve would be hashed by identity and
    recompile the march on every call.

    Attributes
    ----------
    flow_direction : int
        The streamwise axis the bulk velocity is measured and the body force applied along.
    target : float
        The bulk (volume-averaged) velocity component to hold.
    """

    flow_direction: int
    target: float

    def __call__(self, augmented: jnp.ndarray, theta: CoupledRANS) -> jnp.ndarray:
        # beta (the last entry) overrides the assembler's body force.
        coupled_state, beta = augmented[:-1], augmented[-1]
        forced_momentum = _with_body_force(theta.momentum, self.flow_direction, beta)
        forced = eqx.tree_at(lambda c: c.momentum, theta, forced_momentum)
        r_coupled = forced.residual(coupled_state)
        flow_state, _, _ = theta.layout.unpack(coupled_state)
        velocity, _ = theta.momentum.unpack(flow_state)
        volume = theta.momentum.geometry.cell.volume
        bulk = jnp.sum(velocity[:, self.flow_direction] * volume) / jnp.sum(volume)
        return jnp.append(r_coupled, bulk - self.target)


#: The stopping test of a mass-flow-constrained solve given no :class:`~aquaflux.solve.Convergence`. The
#: row-scaled measure the unconstrained solve defaults to has no form for the bordered system.
_MASS_FLOW_CONVERGENCE = Convergence(measure=Euclidean(), rtol=1e-10, atol=1e-12)


def solve_coupled_mass_flow(
    coupled: CoupledRANS,
    target: float,
    *,
    flow_direction: int = 0,
    flow: jnp.ndarray | None = None,
    k: jnp.ndarray | None = None,
    omega: jnp.ndarray | None = None,
    strategy: PseudoTransientStep | None = None,
    reference_state: jnp.ndarray | None = None,
    preconditioner: BlockDiagonal | None = None,
    max_steps: int = 60,
    convergence: Convergence | None = None,
    adjoint_solver: lx.AbstractLinearSolver | None = None,
    **strategy_kwargs: object,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the coupled RANS system holding the bulk velocity at ``target``, in one monolithic Newton.

    The mass-flow analogue of :func:`solve_coupled`: the body force ``beta`` (along ``flow_direction``)
    is a **coupled unknown** appended to the state, and the coupled residual is bordered with the
    constraint row ``<U_dir> - target`` -- one honest augmented residual

        R_aug([flow, k, omega, beta]) = [ R_coupled(flow, k, omega; beta) ; <U_dir>(flow) - target ],

    driven by a single :class:`~aquaflux.solve.RootSolver` globalized by
    :func:`mass_flow_coupled_continuation`. ``<U> = target`` therefore holds at the converged root **by
    construction**, and (the point of putting the constraint *in* the coupled residual) the coupled
    implicit-function-theorem adjoint carries it: ``jax.grad`` through the converged constrained solve is
    the exact sensitivity of the whole turbulent flow at fixed bulk velocity. The forward solve is
    monolithic here, but the same bordered residual is what a *segregated* forward loop would need its
    coupled adjoint to transpose (segregated forward, coupled adjoint).

    Parameters mirror :func:`solve_coupled` (``coupled`` is the differentiable parameter pytree; leave
    ``flow``/``k``/``omega`` ``None`` to self-start from the hybrid IC; build ``strategy`` outside
    ``jax.grad`` when differentiating), plus:

    target : float
        The bulk (volume-averaged) velocity component to hold along ``flow_direction``.
    flow_direction : int
        The streamwise axis the bulk velocity is measured and the body force applied along.
    convergence : Convergence or None
        The stopping test. Unset fields take :class:`~aquaflux.solve.Euclidean`, ``rtol = 1e-10`` and
        ``atol = 1e-12``. The measure may be :class:`~aquaflux.solve.Euclidean` or
        :class:`~aquaflux.solve.BlockScaled` (whose constraint row shares the flow block's scale);
        :class:`~aquaflux.solve.RowScaled` is refused, since the constraint's border row has no diagonal
        to equilibrate by.
    preconditioner : BlockDiagonal or None
        The block-diagonal preconditioner the constrained step is built with; ``None`` takes
        ``BlockDiagonal()``. See :func:`mass_flow_coupled_continuation` for why no other family applies.

    Returns
    -------
    tuple of jnp.ndarray
        The converged ``(flow, k, omega, beta)`` -- the fields and the multiplier that hits ``target``.

    Raises
    ------
    TypeError
        If ``preconditioner``, ``reference_state`` or a march setting is given beside a finished
        ``strategy``, which already carries its configuration -- they would otherwise be dropped
        without a word.
    """
    refuse_a_transform_the_march_cannot_run_in(
        (coupled, flow, k, omega), caller="solve_coupled_mass_flow"
    )
    given = dict(strategy_kwargs)
    if preconditioner is not None:
        given["preconditioner"] = preconditioner
    if reference_state is not None:
        given["reference_state"] = reference_state
    if strategy is not None:
        _refuse(
            given,
            "`strategy`",
            "the step you passed already carries them",
            solver="solve_coupled_mass_flow",
        )
    if flow is None or k is None or omega is None:
        flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
    # Map the physical initial condition into the solved-variable space (identity for DirectScalars,
    # log for LogScalars) so the constrained Newton march iterates on the right scalar unknown.
    state = coupled.state_from_physical(flow, k, omega)
    augmented0 = jnp.append(state, coupled.momentum.body_force[flow_direction])

    if strategy is None:
        reference = state if reference_state is None else reference_state
        strategy = mass_flow_coupled_continuation(
            coupled,
            reference,
            flow_direction=flow_direction,
            preconditioner=preconditioner,
            **strategy_kwargs,
        )
    solver = RootSolver(
        convergence=(
            _MASS_FLOW_CONVERGENCE
            if convergence is None
            else convergence.filled_from(_MASS_FLOW_CONVERGENCE)
        ),
        measures=_MassFlowMeasures(stop_array_gradients(coupled)),
        max_steps=max_steps,
        strategy=strategy,
        # Exposed for the same reason `solve_coupled` exposes it, and this path needs it more: the
        # transpose solve here runs at zero shift against a block-diagonal preconditioner, which is
        # where that preconditioner is weakest, and the default solver's stagnation detector sits close
        # enough to the edge that a perturbation of the warm state in the last few bits decides whether
        # it fires. `None` keeps `RootSolver`'s own default.
        **({} if adjoint_solver is None else {"adjoint_solver": adjoint_solver}),
    )

    solved = solver.solve(_MassFlowConstrainedResidual(flow_direction, target), augmented0, coupled)
    flow_s, k_s, omega_s = coupled.physical_fields(solved[:-1])
    return flow_s, k_s, omega_s, solved[-1]
