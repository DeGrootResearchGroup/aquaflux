"""Block SIMPLE preconditioner for the coupled pressure--velocity Newton solve.

Composes a **velocity-block solve** with a **pressure-Schur inner solve** into the preconditioner
``M ≈ J⁻¹`` that :func:`~aquaflux.solve.newton.newton_step` applies to the coupled saddle-point
system (on the right, its default). Both the Schur inner solve (:class:`InnerSchurSolver`) and the velocity solve
(:class:`VelocityBlockSolver`) are **swappable strategies**, built once off the jit path from the
assembler's frozen geometry and applied per Newton iterate at the current momentum diagonal ``a_P``.
Every coefficient is ``stop_gradient``-ed, so ``M`` only accelerates the Krylov iteration — it never
perturbs the converged solution or its adjoint.

The inner pressure Schur is a **smoothed-aggregation multigrid** (:class:`SmoothedAmgSchur`,
mesh-independent V-cycle contraction ~0.25), paired with a velocity-block algebraic multigrid (AMG)
(:class:`SmoothedAmgVelocity` on the viscous operator, or :class:`SmoothedAmgConvectionVelocity` on
the convection-diffusion operator). How those two solves are then *composed* is a third swappable
strategy (:class:`SaddleComposition`): a lower block-triangular pass, the full SIMPLE block ``LU``, or
SIMPLER's pressure-prediction-first sequence. All three strategy families are abstract interfaces
(:class:`InnerSchurSolver` / :class:`VelocityBlockSolver` / :class:`SaddleComposition`), the seams a new
inner solver, velocity block or composition plugs into.
"""

from __future__ import annotations

import abc
import dataclasses
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum

from aquaflux.discretization import flux_continuous_conductance
from aquaflux.solve import (
    air_multigrid_solve,
    build_air_hierarchy,
    build_convection_hierarchy,
    build_smoothed_hierarchy,
    convection_diffusion_operator,
    convection_multigrid_solve,
    decouple_dof,
    smoothed_multigrid_solve,
)
from aquaflux.solve.settings_value import SettingsValue
from aquaflux.vectors import scale

from .preconditioner import schur_face_coefficient
from .rhie_chow import momentum_diagonal
from .scales import characteristic_velocity

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry
    from aquaflux.solve import AirHierarchy, SmoothedHierarchy

    from .momentum import MomentumContinuity

_PressureSolve = Callable[[jnp.ndarray], jnp.ndarray]
_VelocitySolve = Callable[[jnp.ndarray], jnp.ndarray]
# A solve of one scalar field, shape ``(n_cells,) -> (n_cells,)`` — the shape the composition
# helpers below are generic over (the two aliases above name the *role* the solve plays).
_ScalarSolve = Callable[[jnp.ndarray], jnp.ndarray]


def _symmetric_rescaled(
    inner_solve: _ScalarSolve, diag_ref: jnp.ndarray, diag_cur: jnp.ndarray
) -> _ScalarSolve:
    """Track an operator's current diagonal with a solve frozen at a reference diagonal.

    Every multigrid block here freezes its hierarchy at a reference operator ``A_ref`` and reuses it
    across iterates, where the true operator ``A_cur`` has drifted in scale. Writing that drift as a
    symmetric diagonal congruence ``A_cur ≈ D A_ref D`` with ``D = sqrt(diag_cur/diag_ref)`` gives
    ``A_cur⁻¹ ≈ D⁻¹ A_ref⁻¹ D⁻¹`` — the "sandwich" this returns. It is **exact** for a uniform
    rescale and diagonal-exact whenever ``diag_ref`` is the reference operator's own diagonal;
    otherwise it captures the per-cell scale while leaving the frozen off-diagonal structure alone.

    Symmetric (rather than a one-sided ``diag_cur/diag_ref``) so a symmetric-positive-definite block
    stays symmetric-positive-definite, which the Krylov iteration the preconditioner feeds relies on.

    Parameters
    ----------
    inner_solve : callable
        The frozen solve ``b -> A_ref⁻¹ b``, shape ``(n_cells,) -> (n_cells,)``.
    diag_ref : jnp.ndarray
        Diagonal of the frozen reference operator, shape ``(n_cells,)``.
    diag_cur : jnp.ndarray
        Diagonal of the current operator, shape ``(n_cells,)``.

    Returns
    -------
    callable
        The rescaled solve ``b -> A_cur⁻¹ b``, shape ``(n_cells,) -> (n_cells,)``.
    """
    inv_scale = jnp.sqrt(diag_ref / diag_cur)
    return lambda b: inv_scale * inner_solve(inv_scale * b)


def _per_component(scalar_solve: _ScalarSolve, dim: int) -> _VelocitySolve:
    """Lift a scalar-field solve to a vector field by applying it to each component.

    The momentum block is block-diagonal across velocity components (the components couple only
    through pressure, which the Schur block carries), so inverting it is the same scalar solve run
    per component.

    Parameters
    ----------
    scalar_solve : callable
        The per-component solve, shape ``(n_cells,) -> (n_cells,)``.
    dim : int
        Number of spatial components.

    Returns
    -------
    callable
        The vector solve, shape ``(n_cells, dim) -> (n_cells, dim)``.
    """

    def solve(ru: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([scalar_solve(ru[:, i]) for i in range(dim)], axis=1)

    return solve


class _SchurGeometry(eqx.Module):
    """The geometry the pressure-Schur coefficient needs — bundled so the schur strategies share it.

    Encapsulates the single computation "current SIMPLE Schur face coefficient at momentum diagonal
    ``a_P``" (:meth:`coefficient`), reused by every AMG-based inner solve.
    """

    face_cells: FaceCellConnectivity
    mesh_geometry: MeshGeometry
    boundary: object
    interp_factor: jnp.ndarray
    normal_distance: jnp.ndarray
    rho: jnp.ndarray
    owner_e: jnp.ndarray
    nb_e: jnp.ndarray
    interior_faces: jnp.ndarray
    n_cells: int = eqx.field(static=True)
    pressure_pin: int | None = eqx.field(static=True)

    @classmethod
    def of(cls, assembler: MomentumContinuity) -> _SchurGeometry:
        """Extract the Schur-coefficient geometry from a flow assembler."""
        owner_e, nb_e, interior_faces = assembler.mesh.face_cells.interior_edges()
        return cls(
            assembler.mesh.face_cells,
            assembler.geometry,
            assembler.boundary,
            assembler.interp_factor,
            assembler.normal_distance,
            assembler.density,
            jnp.asarray(owner_e),
            jnp.asarray(nb_e),
            jnp.asarray(interior_faces),
            assembler.mesh.n_cells,
            assembler.pressure_pin,
        )

    def diagonal(self, a_p: jnp.ndarray) -> jnp.ndarray:
        """The current pressure-Schur operator diagonal at momentum diagonal ``a_P``.

        The interior coefficient scattered to both of each face's cells, plus the boundary stiffness
        the reference hierarchy also carries, with the pin row set to one where a closed domain pins
        the pressure. This is the ``diag_cur`` every frozen-hierarchy Schur block rescales against, so
        it lives here — on the object that owns the coefficient — rather than in each strategy.
        """
        coefficient = self.coefficient(a_p)[self.interior_faces]
        diagonal = (
            segment_sum(coefficient, self.owner_e, self.n_cells)
            + segment_sum(coefficient, self.nb_e, self.n_cells)
            + self.boundary_diagonal(a_p)
        )
        if self.pressure_pin is not None:
            diagonal = diagonal.at[self.pressure_pin].set(1.0)
        return diagonal

    def coefficient(self, a_p: jnp.ndarray) -> jnp.ndarray:
        """The (frozen) per-face SIMPLE Schur coefficient at momentum diagonal ``a_P``."""
        return jax.lax.stop_gradient(
            schur_face_coefficient(
                self.face_cells,
                self.mesh_geometry,
                self.interp_factor,
                self.normal_distance,
                a_p,
                self.rho,
            )
        )

    def boundary_diagonal(self, a_p: jnp.ndarray) -> jnp.ndarray:
        """The (frozen) per-cell pressure-Schur boundary stiffness at momentum diagonal ``a_P``.

        Each boundary patch adds its :meth:`~aquaflux.flow.boundary.FlowBoundary.pressure_schur_coefficient`
        (non-zero only for a pressure-fixing outlet) to its owner cell's Schur diagonal — the term that
        de-singularises the open-domain Schur, whose interior part is a pure-Neumann Laplacian. Zero
        everywhere for a closed all-wall domain (regularized instead by the pin).
        """
        face = self.mesh_geometry.face
        d_coeff = self.mesh_geometry.cell.volume / a_p  # isotropic V/a_P per cell
        per_face = self.boundary.apply(
            self.face_cells,
            jnp.zeros(face.area.shape),
            lambda bc, faces, owner: bc.pressure_schur_coefficient(
                d_coeff[owner], face.area[faces], self.normal_distance[faces], self.rho[owner]
            ),
        )
        n_cells = self.mesh_geometry.cell.volume.shape[0]
        return jax.lax.stop_gradient(segment_sum(per_face, self.face_cells.owner, n_cells))


class _VelocityGeometry(eqx.Module):
    """The frozen geometry the velocity-block AMG strategies need — bundled so they build from a
    narrow, testable seam instead of reaching into the full flow assembler (the momentum-block
    counterpart of :class:`_SchurGeometry`).

    Unlike :class:`_SchurGeometry`, this is a **build-time-only** input: the velocity strategies freeze
    their AMG hierarchy at build and do not store the geometry, so this bundle is consumed by
    ``build`` and discarded (it never enters the strategy pytree or the differentiated apply).
    """

    face_cells: FaceCellConnectivity
    mesh_geometry: MeshGeometry
    viscosity: jnp.ndarray
    dim: int = eqx.field(static=True)

    @classmethod
    def of(cls, assembler: MomentumContinuity) -> _VelocityGeometry:
        """Extract the velocity-block geometry from a flow assembler."""
        return cls(
            assembler.mesh.face_cells,
            assembler.geometry,
            assembler.viscosity,
            assembler.mesh.dim,
        )


# --- pressure-Schur inner solvers (strategy family) ------------------------------------


class InnerSchurSolver(eqx.Module):
    """Strategy: solve the compact pressure Schur ``Ŝ x = rp`` for the preconditioner.

    Built once off the jit path; :meth:`apply` returns the solve ``rp -> Ŝ⁻¹ rp`` specialized to the
    current (frozen) momentum diagonal ``a_P``.
    """

    @abc.abstractmethod
    def apply(self, a_p: jnp.ndarray) -> _PressureSolve:
        """Return the pressure solve ``rp -> Ŝ⁻¹ rp`` at momentum diagonal ``a_P``.

        Parameters
        ----------
        a_p : jnp.ndarray
            The frozen isotropic momentum diagonal, shape ``(n_cells,)``.
        """


class SmoothedAmgSchur(InnerSchurSolver):
    """Smoothed-aggregation multigrid, mesh-independent (V-cycle contraction ~0.25).

    The hierarchy is frozen at a reference coefficient; the current operator's scale is tracked by a
    symmetric diagonal rescaling ``Ŝ_cur⁻¹ ≈ D⁻¹ Ŝ_ref⁻¹ D⁻¹``, ``D = sqrt(diag_cur/diag_ref)`` — exact
    for a uniform rescale, and capturing per-cell scale (including convection) otherwise.

    Regime limit (measured, and the reason ``v_cycles`` is not a high-Reynolds lever): with the
    ``"msimple"`` scaling this is a **constant-coefficient** pressure Poisson — a near-Stokes
    approximation of the true Schur complement that degrades as convection strengthens. Once the flow is
    convection-dominated (high Reynolds number, recirculation) that *approximation* — not its inversion —
    sets the outer Krylov cost: inverting it more accurately does not help and can hurt, and neither
    rescaling it nor rebuilding it at a developed state recovers the loss. A stronger Schur approximation
    for the *isolated* flow saddle does not cure a coupled flow--turbulence solve either: under a
    block-diagonal preconditioner and a pseudo-transient shift the coupled iteration is not limited by the
    flow Schur's quality, which is where a monolithic preconditioner of the whole coupled Jacobian is used.
    """

    geometry: _SchurGeometry
    hierarchy: SmoothedHierarchy
    v_cycles: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        geometry: _SchurGeometry,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior: np.ndarray,
        n_cells: int,
        v_cycles: int,
        reference_diagonal: jnp.ndarray | None = None,
        strength_threshold: float = 0.0,
    ) -> SmoothedAmgSchur:
        # Reference diagonal for the frozen hierarchy, fed to the Schur coefficient ``V / d``. SIMPLE
        # uses a unit-viscosity momentum ``a_P`` (the multigrid is scale-invariant, so a concrete
        # reference keeps the scipy build valid even inside a differentiated region), with the
        # per-iterate convective ``a_P`` restored by the symmetric rescaling in :meth:`apply`.
        # The mass-scaled Schur instead supplies the velocity mass-matrix diagonal ``rho V`` — a
        # independent scaling that does not degrade as convection strengthens, so its rescaling is the
        # identity. The isotropic (component-averaged) form is used; the directional per-component
        # ``a_P`` enters only the operator's Rhie--Chow coefficient.
        if reference_diagonal is None:
            reference_diagonal = jnp.mean(
                momentum_diagonal(
                    geometry.face_cells,
                    geometry.mesh_geometry,
                    jnp.ones(n_cells),
                ),
                axis=1,
            )
        reference_coeff = np.asarray(geometry.coefficient(reference_diagonal))[interior]
        # A pressure-fixing outlet adds a boundary diagonal that de-singularises the Schur; freeze it
        # at the reference diagonal (all-zero for a closed all-wall domain, which the pin handles).
        reference_boundary = np.asarray(geometry.boundary_diagonal(reference_diagonal))
        a = convection_diffusion_operator(
            owner_e, nb_e, reference_coeff, n_cells, boundary_diagonal=reference_boundary
        )
        if geometry.pressure_pin is not None:  # closed domain: regularize by decoupling the pin
            a = decouple_dof(a, geometry.pressure_pin)
        hierarchy = build_smoothed_hierarchy(a, strength_threshold=strength_threshold)
        return cls(geometry, hierarchy, v_cycles)

    def apply(self, a_p: jnp.ndarray) -> _PressureSolve:
        # The reference hierarchy carries the boundary (outlet) stiffness in its diagonal, and
        # `geometry.diagonal` includes it, so the symmetric rescaling stays consistent.
        return _symmetric_rescaled(
            lambda rp: smoothed_multigrid_solve(self.hierarchy, rp, cycles=self.v_cycles),
            self.hierarchy.levels[0].diagonal,
            self.geometry.diagonal(a_p),
        )


# --- velocity-block solvers (strategy family) ------------------------------------------


class VelocityBlockSolver(eqx.Module):
    """Strategy: approximately invert the momentum (velocity) block for the preconditioner."""

    @abc.abstractmethod
    def apply(self, a_p: jnp.ndarray) -> _VelocitySolve:
        """Return the velocity solve ``ru -> δu`` at momentum diagonal ``a_P``."""


class _RescaledAmgVelocity(VelocityBlockSolver):
    """A velocity block that inverts a frozen AMG hierarchy, rescaled to the current ``a_P``.

    Both concrete velocity blocks share the same per-iterate structure and differ only in the
    operator their hierarchy is frozen at (viscous or convection-diffusion) and the V-cycle that
    inverts it: hold the coarse-grid structure fixed, track the current momentum diagonal by the
    symmetric rescaling :func:`_symmetric_rescaled`, and apply the result per velocity component
    (:func:`_per_component`). That composition is :meth:`apply`, defined here once; a subclass
    supplies its hierarchy and :meth:`_inner_solve`.
    """

    hierarchy: SmoothedHierarchy | AirHierarchy
    dim: int = eqx.field(static=True)
    v_cycles: int = eqx.field(static=True)

    @abc.abstractmethod
    def _inner_solve(self, b: jnp.ndarray) -> jnp.ndarray:
        """One momentum-component solve against the frozen reference operator."""

    def apply(self, a_p: jnp.ndarray) -> _VelocitySolve:
        # Rescaling against the frozen hierarchy's own diagonal reproduces the current ``a_P`` on the
        # diagonal exactly; the off-diagonal structure (the viscous stencil, or the frozen convection
        # direction) stays as built at the reference.
        return _per_component(
            _symmetric_rescaled(self._inner_solve, self.hierarchy.levels[0].diagonal, a_p), self.dim
        )


class SmoothedAmgVelocity(_RescaledAmgVelocity):
    """Smoothed-aggregation AMG on the viscous momentum operator (mesh-independent), per component.

    The viscous momentum operator is a Dirichlet (no-slip) Laplacian — SPD-nonsingular (boundary faces
    add stiffness to the diagonal, no pin) — so a single AMG hierarchy on a unit-viscosity reference,
    rescaled to the current ``a_P``, replaces the Jacobi-quality diagonal solve.
    """

    @classmethod
    def build(
        cls,
        geometry: _VelocityGeometry,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior: np.ndarray,
        n_cells: int,
        v_cycles: int,
        strength_threshold: float = 0.0,
    ) -> SmoothedAmgVelocity:
        face_cells = geometry.face_cells
        # Unit-viscosity viscous coefficient A/denom — the geometry-only part of the momentum
        # diagonal, rescaled to the current a_P in :meth:`apply`. From the shared conductance with a
        # unit viscosity, so the reference operator matches the momentum diagonal's viscous term.
        over_distance = flux_continuous_conductance(
            jnp.ones(n_cells), geometry.mesh_geometry, face_cells
        )
        # Boundary-face owner stiffness (the boundary part of the momentum diagonal), scattered to
        # cells via the connectivity's own scatter rather than a hand-rolled index add.
        boundary_owner = jnp.where(face_cells.interior, 0.0, over_distance)
        boundary_diagonal = face_cells.scatter(boundary_owner, jnp.zeros_like(over_distance))
        a = convection_diffusion_operator(
            owner_e,
            nb_e,
            np.asarray(over_distance)[interior],
            n_cells,
            boundary_diagonal=np.asarray(boundary_diagonal),
        )
        hierarchy = build_smoothed_hierarchy(a, strength_threshold=strength_threshold)
        return cls(hierarchy, geometry.dim, v_cycles)

    def _inner_solve(self, b: jnp.ndarray) -> jnp.ndarray:
        """One momentum-component inner solve: the smoothed-aggregation V-cycle."""
        return smoothed_multigrid_solve(self.hierarchy, b, cycles=self.v_cycles)


class SmoothedAmgConvectionVelocity(_RescaledAmgVelocity):
    """Convection-aware AMG on the frozen convection-diffusion momentum operator, per component.

    :class:`SmoothedAmgVelocity` builds its hierarchy on the *viscous* (symmetric) momentum operator,
    so it is Peclet-blind: once the cell Peclet number grows, the true momentum block is dominated by
    upwind convective transport the symmetric AMG cannot represent, and rescaling by the convective
    diagonal ``a_P`` does not fix the coarse space. This strategy instead builds a nonsymmetric
    aggregation hierarchy on the full ``viscous + first-order-upwind`` operator, frozen at a reference
    mass flux, so the coarse operators carry the convection direction. The reference operator's
    diagonal is exactly the momentum diagonal ``a_P`` at the reference, so the per-iterate symmetric
    rescaling to the current ``a_P`` is diagonal-exact — the same tracking the viscous block uses.

    The reference mass flux is taken from a representative operating state supplied at build time; a
    cold (zero-flux) reference reduces this to the viscous block. Two coarsening strategies (all
    matrix-free and transposable for the adjoint), selected by ``method``:

    * ``"twolevel"`` — a single aggregation with a direct coarse solve, inverted by a damped-Jacobi
      V-cycle. Stable across cell Peclet, but the direct coarse solve does not scale to large meshes.
    * ``"air"`` — a reduction-based hierarchy (local approximate ideal restriction) that coarsens all
      the way down and stays Peclet-robust and mesh-independent, inverted by an FC-Jacobi V-cycle.
    """

    method: str = eqx.field(static=True)
    sweeps: int = eqx.field(static=True)
    omega: float = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        geometry: _VelocityGeometry,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior: np.ndarray,
        n_cells: int,
        v_cycles: int,
        reference_mdot: jnp.ndarray,
        *,
        method: str = "twolevel",
        sweeps: int = 2,
        omega: float = 0.8,
        strength_threshold: float = 0.0,
    ) -> SmoothedAmgConvectionVelocity:
        # ``reference_mdot`` is the (frozen) Rhie--Chow mass flux of a representative operating state
        # -- the convective linearization the hierarchy is frozen at, so the operator diagonal matches
        # the momentum diagonal ``a_P`` at the reference and the per-iterate rescaling is exact. It is
        # supplied by the caller because it is assembler behaviour (the flux operator), not geometry.
        face_cells = geometry.face_cells
        mu = jax.lax.stop_gradient(geometry.viscosity)
        # Flux-continuous viscous conductance, from the shared definition so the frozen operator's
        # viscous term cannot drift from the momentum diagonal's (both are the diffusion operator's
        # own diagonal contribution, harmonic on graded viscosity).
        viscous = flux_continuous_conductance(mu, geometry.mesh_geometry, face_cells)
        # Boundary-face owner contribution to the momentum diagonal ``a_P`` (the plain all-faces form
        # the frozen diagonal keeps: ``viscous + max(mdot, 0)`` on each boundary face), scattered to
        # cells by the connectivity's own scatter. Built from the same ``viscous`` and reference flux as
        # the interior off-diagonals, so the assembled operator's diagonal is exactly the frozen ``a_P``
        # — no separate reconstruction of the interior upwind stencil.
        boundary_owner = jnp.where(
            face_cells.interior, 0.0, viscous + jnp.maximum(reference_mdot, 0.0)
        )
        boundary_diagonal = face_cells.scatter(boundary_owner, jnp.zeros_like(boundary_owner))
        a = convection_diffusion_operator(
            owner_e,
            nb_e,
            np.asarray(viscous)[interior],
            n_cells,
            flux=np.asarray(reference_mdot)[interior],
            boundary_diagonal=np.asarray(boundary_diagonal),
        )
        hierarchy: SmoothedHierarchy | AirHierarchy
        if method == "air":
            # Reduction-based (lAIR) coarsening: coarsens fully and stays Peclet-robust /
            # mesh-independent, so it scales where the two-level direct coarse solve cannot. Its C/F
            # split is already strength-based, so ``strength_threshold`` (an aggregation knob) does not
            # apply here.
            hierarchy = build_air_hierarchy(a)
        elif method == "twolevel":
            # A single aggregation with a direct coarse solve: the aggregation coarse space stays a
            # stable correction at high cell Peclet, where a deeper Galerkin recursion does not (the
            # builder is two-level for exactly this reason). ``strength_threshold > 0`` aggregates along
            # strong connections only — the fix for a high-aspect-ratio near-wall velocity operator.
            hierarchy = build_convection_hierarchy(a, strength_threshold=strength_threshold)
        else:
            raise ValueError(f"unknown convection method {method!r}; use 'twolevel' or 'air'")
        return cls(hierarchy, geometry.dim, v_cycles, method, sweeps, omega)

    def _inner_solve(self, b: jnp.ndarray) -> jnp.ndarray:
        """One momentum-component inner solve: the reduction-based (lAIR) or two-level V-cycle."""
        if self.method == "air":
            return air_multigrid_solve(self.hierarchy, b, cycles=self.v_cycles)
        return convection_multigrid_solve(
            self.hierarchy, b, cycles=self.v_cycles, sweeps=self.sweeps, omega=self.omega
        )


def _convection_operator(
    geometry: _VelocityGeometry,
    owner_e: np.ndarray,
    nb_e: np.ndarray,
    interior: np.ndarray,
    n_cells: int,
    reference_mdot: jnp.ndarray,
):
    """The frozen ``viscous + first-order-upwind`` momentum operator at ``reference_mdot``, as CSR.

    Shared by both convection-aware velocity blocks, which differ only in the hierarchy that coarsens
    it. The viscous coupling is the flux-continuous conductance at the actual (possibly graded)
    viscosity, and the boundary diagonal is the boundary-face owner coefficient built from the same
    viscous term and reference flux, so the assembled operator's diagonal is exactly the frozen momentum
    diagonal ``a_P`` at the reference -- which is what makes the per-iterate rescaling diagonal-exact.
    """
    face_cells = geometry.face_cells
    viscous = flux_continuous_conductance(
        jax.lax.stop_gradient(geometry.viscosity), geometry.mesh_geometry, face_cells
    )
    boundary_owner = jnp.where(face_cells.interior, 0.0, viscous + jnp.maximum(reference_mdot, 0.0))
    boundary_diagonal = face_cells.scatter(boundary_owner, jnp.zeros_like(boundary_owner))
    return convection_diffusion_operator(
        owner_e,
        nb_e,
        np.asarray(viscous)[interior],
        n_cells,
        flux=np.asarray(reference_mdot)[interior],
        boundary_diagonal=np.asarray(boundary_diagonal),
    )


class TwoLevelConvectionVelocity(_RescaledAmgVelocity):
    """A two-level aggregation hierarchy on the frozen convection-diffusion momentum operator.

    The fine cells are aggregated once, with symmetric-part prolongation smoothing, and the coarse
    operator is solved directly; the fine level is smoothed by damped Jacobi. That coarse space stays a
    stable correction at high cell Peclet, where a deeper Galerkin recursion does not -- but the direct
    coarse solve does not scale to large meshes (see :class:`AirConvectionVelocity` for that).
    ``strength_threshold > 0`` aggregates along strong connections only, which keeps it contracting on a
    high-aspect-ratio near-wall operator.
    """

    sweeps: int = eqx.field(static=True)
    omega: float = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        geometry: _VelocityGeometry,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior: np.ndarray,
        n_cells: int,
        v_cycles: int,
        reference_mdot: jnp.ndarray,
        *,
        sweeps: int = 2,
        omega: float = 0.8,
        strength_threshold: float = 0.0,
    ) -> TwoLevelConvectionVelocity:
        a = _convection_operator(geometry, owner_e, nb_e, interior, n_cells, reference_mdot)
        hierarchy = build_convection_hierarchy(a, strength_threshold=strength_threshold)
        return cls(hierarchy, geometry.dim, v_cycles, sweeps, omega)

    def _inner_solve(self, b: jnp.ndarray) -> jnp.ndarray:
        """One momentum-component inner solve: the two-level V-cycle."""
        return convection_multigrid_solve(
            self.hierarchy, b, cycles=self.v_cycles, sweeps=self.sweeps, omega=self.omega
        )


class AirConvectionVelocity(_RescaledAmgVelocity):
    """A reduction-based (lAIR) hierarchy on the frozen convection-diffusion momentum operator.

    Local approximate ideal restriction coarsens all the way down and stays Peclet-robust and
    mesh-independent, inverted by an FC-Jacobi V-cycle. Its C/F split is already strength-based, so it
    takes no aggregation ``strength_threshold``.
    """

    @classmethod
    def build(
        cls,
        geometry: _VelocityGeometry,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior: np.ndarray,
        n_cells: int,
        v_cycles: int,
        reference_mdot: jnp.ndarray,
    ) -> AirConvectionVelocity:
        a = _convection_operator(geometry, owner_e, nb_e, interior, n_cells, reference_mdot)
        return cls(build_air_hierarchy(a), geometry.dim, v_cycles)

    def _inner_solve(self, b: jnp.ndarray) -> jnp.ndarray:
        """One momentum-component inner solve: the lAIR V-cycle."""
        return air_multigrid_solve(self.hierarchy, b, cycles=self.v_cycles)


@dataclasses.dataclass(frozen=True)
class VelocityBlock(SettingsValue):
    """Which velocity block :meth:`BlockPreconditioner.build` fits, and its settings, as a value.

    The velocity block is fitted to a frozen momentum operator and coarsened by a multigrid hierarchy.
    Those two choices do not vary independently -- the multilevel Chebyshev hierarchy needs a symmetric
    operator, so it cannot coarsen the convection-diffusion one -- so each value names one exercised
    pairing, and carries only the settings that apply to it. Every field defaults to ``None``, meaning
    "not set here": only set fields reach the strategy, whose own defaults stay the only defaults.
    """

    def _build(self, geometry: _VelocityGeometry, inputs: _StrategyInputs) -> VelocityBlockSolver:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ViscousMultilevel(VelocityBlock):
    """A multilevel smoothed-aggregation hierarchy on the viscous momentum operator, Chebyshev-smoothed.

    Fitted at unit viscosity and rescaled to the current momentum diagonal each apply. Mesh-independent
    but blind to convection, so it bounds the reachable Reynolds number; the default for a flow-only
    solve. Builds :class:`SmoothedAmgVelocity`.
    """

    def _build(self, geometry: _VelocityGeometry, inputs: _StrategyInputs) -> VelocityBlockSolver:
        return SmoothedAmgVelocity.build(
            geometry,
            inputs.owner_e,
            inputs.nb_e,
            inputs.interior,
            inputs.n_cells,
            inputs.v_cycles,
            strength_threshold=inputs.strength_threshold,
        )


@dataclasses.dataclass(frozen=True)
class _ConvectionVelocityBlock(VelocityBlock):
    """A velocity block fitted to the convection-diffusion operator frozen at a reference mass flux.

    Owns what every such block shares: computing that flux from the reference state, and saying so when
    it is zero, so a subclass names only its strategy and the settings it forwards.
    """

    def _strategy_class(self) -> type:
        raise NotImplementedError

    def _strategy_settings(self, inputs: _StrategyInputs) -> dict[str, object]:
        return self.settings()

    def _build(self, geometry: _VelocityGeometry, inputs: _StrategyInputs) -> VelocityBlockSolver:
        # The reference mass flux is assembler behaviour (the Rhie--Chow flux operator), so it is
        # computed here and handed to the strategy, keeping the strategy build assembler-free.
        reference_mdot = jax.lax.stop_gradient(inputs.assembler.mass_flux(inputs.reference_state))
        if float(jnp.max(jnp.abs(reference_mdot))) == 0.0:
            # No convective scale to freeze the operator at: it collapses to the viscous one, so the
            # block is not the convection-aware accelerator that was asked for. Warn rather than fail
            # -- the build is still valid.
            warnings.warn(
                f"velocity block {type(self).__name__}() was requested but the reference state "
                "carries no mass flux, so its convective linearization is zero and the block is fitted "
                "to the viscous operator alone. The domain neither prescribes a velocity at any patch "
                "nor carries a body force to size one from. Pass an explicit reference_state (e.g. a "
                "uniform flow at the bulk velocity a mass-flow controller targets) to restore the "
                "convection-aware block.",
                RuntimeWarning,
                # Called from `BlockPreconditioner.build`, so this attributes to *its* caller.
                stacklevel=3,
            )
        return self._strategy_class().build(
            geometry,
            inputs.owner_e,
            inputs.nb_e,
            inputs.interior,
            inputs.n_cells,
            inputs.v_cycles,
            reference_mdot,
            **self._strategy_settings(inputs),
        )


@dataclasses.dataclass(frozen=True)
class ConvectionTwoLevel(_ConvectionVelocityBlock):
    """A two-level aggregation hierarchy on the frozen convection-diffusion operator.

    Stable across cell Peclet, but its direct coarse solve does not scale to large meshes. Builds
    :class:`TwoLevelConvectionVelocity`.

    Attributes
    ----------
    sweeps, omega : int, float or None
        The damped-Jacobi smoother's sweeps per level and damping factor (see
        :func:`~aquaflux.solve.convection_multigrid_solve`).
    """

    sweeps: int | None = None
    omega: float | None = None

    def _strategy_class(self) -> type:
        return TwoLevelConvectionVelocity

    def _strategy_settings(self, inputs: _StrategyInputs) -> dict[str, object]:
        return {**self.settings(), "strength_threshold": inputs.strength_threshold}


@dataclasses.dataclass(frozen=True)
class ConvectionAir(_ConvectionVelocityBlock):
    """A reduction-based (lAIR) hierarchy on the frozen convection-diffusion operator.

    Peclet-robust and mesh-independent, so it scales where :class:`ConvectionTwoLevel` cannot. Builds
    :class:`AirConvectionVelocity`.
    """

    def _strategy_class(self) -> type:
        return AirConvectionVelocity


def _characteristic_reference_state(assembler: MomentumContinuity) -> jnp.ndarray:
    """A uniform flow at the characteristic velocity driving the domain, shape ``((dim+1) n,)``.

    The convection-aware velocity block freezes its convective linearization at the mass flux of a
    representative operating state, so that state has to carry the operating convective scale (cell
    Peclet ``rho U dx / mu``) — a cold zero state carries none, and would silently reduce the block to
    the viscous one it exists to replace. The speed itself comes from
    :func:`~aquaflux.flow.scales.characteristic_velocity` (a prescribed boundary velocity, or the
    body-force balance when the domain prescribes none); this only spreads it over the cells as the
    packed flow state.
    """
    velocity = jnp.broadcast_to(
        characteristic_velocity(assembler), (assembler.mesh.n_cells, assembler.mesh.dim)
    )
    return jax.lax.stop_gradient(assembler.pack(velocity, jnp.zeros(assembler.mesh.n_cells)))


# --- the flow saddle's Jacobian blocks, matrix-free ------------------------------------


class FlowBlocks(eqx.Module):
    """The four Jacobian blocks of the flow saddle point, as matrix-free operators at a frozen state.

    The coupled flow Jacobian has the saddle structure ``[[F, G], [B, Ĉ]]`` over the state
    ``[velocity, pressure]``: ``F`` the momentum block, ``G`` the pressure gradient (velocity rows,
    pressure columns), ``B`` the divergence (pressure rows, velocity columns), and ``Ĉ`` the
    pressure--pressure coupling — which for a collocated Rhie--Chow discretization is the pressure
    damping that suppresses checkerboarding, i.e. this discretization's *stabilization* operator.

    **Sign convention (measured, not assumed).** In this residual's signs ``Ĉ`` is *positive* definite
    and ``B F⁻¹ G`` is *negative* definite, so the pressure Schur complement ``S = Ĉ - B F⁻¹ G`` is
    positive definite — which is the convention every Schur strategy here follows (they return an
    approximate ``S⁻¹`` for that positive ``S``). Note the consequence for anything written in the
    usual textbook saddle form ``[[F, Bᵀ], [B, -C]]``: that form's ``Bᵀ`` is ``-G`` here, so a product
    with an *odd* number of gradient factors picks up a sign flip against the literature formula.

    Every block is one ``jax.jvp`` through the **frozen** residual: inject a tangent in one field and
    read the response in one field. Both the assembler and the state are ``stop_gradient``-ed, so the
    resulting operators are constant — a preconditioner built from them changes only the Krylov
    iteration, never the converged solution or its adjoint.

    The two combined methods are the primitives (each is a *single* ``jvp`` yielding both blocks of a
    column); the four named single-block accessors compose them, so a caller that needs both halves of
    a column pays for one residual linearization rather than two.
    """

    assembler: MomentumContinuity
    state: jnp.ndarray

    @classmethod
    def of(cls, assembler: MomentumContinuity, state: jnp.ndarray) -> FlowBlocks:
        """Freeze the blocks at ``state`` (both the assembler and the state are detached)."""
        return cls(jax.lax.stop_gradient(assembler), jax.lax.stop_gradient(state))

    def _column(self, tangent: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """One linearization: the (velocity, pressure) response to a packed tangent."""
        return self.assembler.unpack(jax.jvp(self.assembler.residual, (self.state,), (tangent,))[1])

    def velocity_column(self, du: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``δu -> (F δu, B δu)`` — the momentum and divergence responses, in one linearization."""
        return self._column(self.assembler.pack(du, jnp.zeros(self.assembler.mesh.n_cells)))

    def pressure_column(self, dp: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``δp -> (G δp, Ĉ δp)`` — the gradient and pressure-coupling responses, in one pass."""
        zeros = jnp.zeros((self.assembler.mesh.n_cells, self.assembler.mesh.dim))
        return self._column(self.assembler.pack(zeros, dp))

    def momentum(self, du: jnp.ndarray) -> jnp.ndarray:
        """``F δu``, shape ``(n_cells, dim) -> (n_cells, dim)``."""
        return self.velocity_column(du)[0]

    def divergence(self, du: jnp.ndarray) -> jnp.ndarray:
        """``B δu``, shape ``(n_cells, dim) -> (n_cells,)``."""
        return self.velocity_column(du)[1]

    def gradient(self, dp: jnp.ndarray) -> jnp.ndarray:
        """``G δp``, shape ``(n_cells,) -> (n_cells, dim)``."""
        return self.pressure_column(dp)[0]

    def pressure_coupling(self, dp: jnp.ndarray) -> jnp.ndarray:
        """The stabilization block ``Ĉ δp``, shape ``(n_cells,) -> (n_cells,)``.

        Positive definite as the residual writes it (see the sign convention above).
        """
        return self.pressure_column(dp)[1]


def _isotropic_momentum_diagonal(assembler: MomentumContinuity, state: jnp.ndarray) -> jnp.ndarray:
    """The frozen, isotropic (component-averaged) momentum diagonal ``a_P`` at ``state``.

    The plain all-faces form (``boundary_corrected=False``): this frozen diagonal is a forward-path
    stabilization scale (the shift and the block it inverts), not the residual's operator-consistent
    coefficient, so it keeps the extra boundary damping that carries the high-Reynolds march. It never
    enters the converged residual or the adjoint.
    """
    velocity, _ = assembler.unpack(jax.lax.stop_gradient(state))
    return jnp.mean(
        jax.lax.stop_gradient(
            assembler.momentum_matrix_diagonal(velocity, boundary_corrected=False)
        ),
        axis=1,
    )


def frozen_momentum_diagonal_parts(
    assembler: MomentumContinuity, state: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The frozen convective/dissipative buckets of the all-faces ``a_P`` at ``state``.

    The two parts a :class:`~aquaflux.solve.ShiftBasis` combines into the pseudo-transient shift; their
    sum is :func:`_isotropic_momentum_diagonal` (to rounding). Frozen and evaluated at a detached state,
    like the total, so the shift stays a constant forward-path scale.

    Public because **a pseudo-transient shift needs only this, not a preconditioner.** A monolithically
    preconditioned coupled step supplies its own inverse and reads the block policy for its shift
    diagonal alone, so making it hold a whole :class:`BlockPreconditioner` for these two arrays built two
    multigrid hierarchies that were never applied -- and, because their aggregation reads the operator's
    values, made the policy's array shapes move with the viscosity, which recompiled the compiled step at
    every Reynolds-continuation rung. The assembler is the whole dependency; take it directly.
    """
    velocity, _ = assembler.unpack(jax.lax.stop_gradient(state))
    convective, dissipative = assembler.momentum_matrix_diagonal_parts(velocity)
    return jax.lax.stop_gradient(convective), jax.lax.stop_gradient(dissipative)


class _StrategyInputs(NamedTuple):
    """The build inputs the Schur and velocity-block strategies share, resolved ONCE in
    :meth:`BlockPreconditioner.build` (issue #272).

    Both `_build_schur` and `_build_velocity_block` need the same mesh connectivity, the same
    multigrid knobs, and the same assembler + reference state a convection-aware strategy freezes
    its linearization at — the build already resolves all eight once before fanning them out, so
    this is that resolved value, not a second description of it. Plain host data (numpy arrays,
    Python scalars, the assembler and an optional state), used only for the duration of one
    `build()` call, so a `NamedTuple` is right rather than an `equinox.Module`: unlike
    `_SchurGeometry`/`_VelocityGeometry`, nothing here is stored on a strategy or crosses `jit`.

    Attributes
    ----------
    owner_e, nb_e, interior : np.ndarray
        The mesh's interior-edge connectivity (`mesh.face_cells.interior_edges()` / `.interior`).
    n_cells : int
        Cells in the mesh.
    v_cycles : int
        Multigrid V-cycles per apply.
    strength_threshold : float
        Strength-of-connection threshold for the AMG aggregation.
    assembler : MomentumContinuity
        The coupled flow residual assembler.
    reference_state : jnp.ndarray or None
        The operating flow state a convection-aware velocity strategy freezes its linearization at.
    """

    owner_e: np.ndarray
    nb_e: np.ndarray
    interior: np.ndarray
    n_cells: int
    v_cycles: int
    strength_threshold: float
    assembler: MomentumContinuity
    reference_state: jnp.ndarray | None


def _build_schur(
    geometry: _SchurGeometry,
    inputs: _StrategyInputs,
    schur_mass_diagonal: jnp.ndarray | None,
) -> InnerSchurSolver:
    """The pressure-Schur strategy :meth:`BlockPreconditioner.build`'s ``schur_scaling`` selects.

    Both scalings assemble the same scaled pressure Laplacian; ``schur_mass_diagonal`` is ``None`` for
    the momentum-diagonal (SIMPLE) scaling and the frozen mass diagonal for the mass-scaled one.
    """
    return SmoothedAmgSchur.build(
        geometry,
        inputs.owner_e,
        inputs.nb_e,
        inputs.interior,
        inputs.n_cells,
        inputs.v_cycles,
        reference_diagonal=schur_mass_diagonal,
        strength_threshold=inputs.strength_threshold,
    )


def _build_velocity_block(
    velocity: str,
    velocity_geometry: _VelocityGeometry,
    inputs: _StrategyInputs,
) -> VelocityBlockSolver:
    """The velocity-block strategy :meth:`BlockPreconditioner.build`'s ``velocity`` selects."""
    if velocity not in ("convection", "convection-air"):
        return SmoothedAmgVelocity.build(
            velocity_geometry,
            inputs.owner_e,
            inputs.nb_e,
            inputs.interior,
            inputs.n_cells,
            inputs.v_cycles,
            strength_threshold=inputs.strength_threshold,
        )
    # The reference mass flux is assembler behaviour (the Rhie--Chow flux operator), so it is
    # computed here and handed to the strategy, keeping the velocity build assembler-free.
    reference_mdot = jax.lax.stop_gradient(inputs.assembler.mass_flux(inputs.reference_state))
    if float(jnp.max(jnp.abs(reference_mdot))) == 0.0:
        # No convective scale to freeze the hierarchy at: the convection-diffusion operator
        # collapses to the viscous one, so this block silently becomes the cheaper `velocity=
        # "smoothed"` it was chosen over. Warn rather than fail — the build is still valid,
        # just not the Peclet-aware accelerator that was asked for.
        warnings.warn(
            "velocity block "
            f"{velocity!r} was requested but the reference state carries no mass flux, so "
            "its convective linearization is zero and the block reduces to the viscous "
            "'smoothed' one. The domain neither prescribes a velocity at any patch nor "
            "carries a body force to size one from. Pass an explicit reference_state (e.g. a "
            "uniform flow at the bulk velocity a mass-flow controller targets) to restore the "
            "convection-aware block.",
            RuntimeWarning,
            # One frame deeper than a warning raised directly in `BlockPreconditioner.build` would
            # be, so this still attributes to *its* caller rather than to `build` itself.
            stacklevel=3,
        )
    return SmoothedAmgConvectionVelocity.build(
        velocity_geometry,
        inputs.owner_e,
        inputs.nb_e,
        inputs.interior,
        inputs.n_cells,
        inputs.v_cycles,
        reference_mdot,
        method="air" if velocity == "convection-air" else "twolevel",
        strength_threshold=inputs.strength_threshold,
    )


# --- saddle compositions (strategy family) ---------------------------------------------


_SaddleSolve = Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


class SaddleComposition(eqx.Module):
    """Strategy: how a velocity solve and a pressure-Schur solve compose into ``M ≈ J⁻¹``.

    The two inner solves — an approximate momentum inverse ``F⁻¹`` and an approximate Schur inverse
    ``Ŝ⁻¹`` — are chosen independently of *how many times* and *in what order* the composed
    preconditioner applies them. That choice is this family. All three members are from Klaij & Vuik
    (2013), whose pressure-correction Schur ``Ŝ ≈ -B F̃⁻¹ G`` (the compact pressure Laplacian, with
    the wide-stencil pressure coupling ``Ĉ`` dropped from the *left*-hand side only) is what
    :class:`SmoothedAmgSchur` assembles.

    ``F̃`` is the **diagonal** stand-in for the momentum block that the Schur was assembled from —
    the momentum diagonal ``a_P`` for the ``a_P``-scaled Schur, the mass diagonal ``Q̂ = ρV/k`` for
    the mass-scaled one. Members receive its inverse as ``inverse_diagonal``; using the same ``F̃``
    the Schur was built from is what makes the composition consistent.
    """

    @abc.abstractmethod
    def apply(
        self,
        blocks: FlowBlocks,
        velocity_solve: _VelocitySolve,
        schur_solve: _PressureSolve,
        inverse_diagonal: jnp.ndarray,
    ) -> _SaddleSolve:
        """Return the residual-to-correction map ``(r_u, r_p) -> (δu, δp)``.

        Parameters
        ----------
        blocks : FlowBlocks
            The saddle's matrix-free Jacobian blocks at the current frozen state, supplying the
            divergence ``B``, gradient ``G`` and pressure-coupling ``Ĉ`` actions.
        velocity_solve : callable
            The approximate momentum solve ``r_u -> F⁻¹ r_u``, shape ``(n_cells, dim)`` both ways.
        schur_solve : callable
            The approximate Schur solve ``r_p -> Ŝ⁻¹ r_p``, shape ``(n_cells,)`` both ways.
        inverse_diagonal : jnp.ndarray
            ``F̃⁻¹`` per cell, shape ``(n_cells,)`` — the reciprocal of the diagonal the Schur was
            assembled from.

        Returns
        -------
        callable
            ``(r_u, r_p) -> (δu, δp)``, a **fixed linear map**: every member applies a fixed number
            of fixed-cycle inner solves, so non-flexible GMRES suffices and
            :func:`jax.linear_transpose` gives the adjoint's transpose exactly.
        """

    @staticmethod
    def _correct_velocity(
        blocks: FlowBlocks,
        inverse_diagonal: jnp.ndarray,
        velocity: jnp.ndarray,
        pressure: jnp.ndarray,
    ) -> jnp.ndarray:
        """``δu - F̃⁻¹ G δp`` — the velocity's response to a pressure the Schur has since moved.

        The closing update of both Klaij & Vuik algorithms, shared here because it is the same
        expression in each.
        """
        return velocity - scale(blocks.gradient(pressure), inverse_diagonal)


class BlockTriangularComposition(SaddleComposition):
    """Lower block-triangular: one velocity solve, then the Schur on the predictor's divergence.

    ``δu = F⁻¹ r_u`` and ``δp = Ŝ⁻¹(r_p - B δu)`` — the classical block-triangular saddle
    preconditioner, which is :class:`SimpleComposition` with its closing velocity update dropped.
    Dropping it is not a truncation of SIMPLE so much as a different preconditioner with its own
    justification: with exact inner solves the preconditioned operator becomes the unipotent
    ``[[I, F⁻¹G], [0, I]]``, the Murphy--Golub--Wathen structure that a Krylov method resolves in two
    iterations. It is one velocity solve, one Schur solve and one residual linearization — the
    cheapest member of the family.
    """

    def apply(
        self,
        blocks: FlowBlocks,
        velocity_solve: _VelocitySolve,
        schur_solve: _PressureSolve,
        inverse_diagonal: jnp.ndarray,
    ) -> _SaddleSolve:
        def solve(
            velocity_residual: jnp.ndarray, pressure_residual: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            velocity = velocity_solve(velocity_residual)
            return velocity, schur_solve(pressure_residual - blocks.divergence(velocity))

        return solve


class SimpleComposition(SaddleComposition):
    """SIMPLE: the block-triangular pass, then the velocity's response to the pressure it found.

    Klaij & Vuik (2013) Algorithm 1 — solve ``F δu' = r_u``, solve ``Ŝ δp = r_p - B δu'``, then
    update ``δu = δu' - F̃⁻¹ G δp``. That closing update is the second (upper-triangular) factor of
    the block ``LU``, so with exact inner solves and ``F̃ = F`` this composition is exactly ``J⁻¹``,
    where :class:`BlockTriangularComposition` still leaves the velocity one Krylov iteration short.
    It costs one extra residual linearization and no extra inner solve.
    """

    def apply(
        self,
        blocks: FlowBlocks,
        velocity_solve: _VelocitySolve,
        schur_solve: _PressureSolve,
        inverse_diagonal: jnp.ndarray,
    ) -> _SaddleSolve:
        def solve(
            velocity_residual: jnp.ndarray, pressure_residual: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            predictor = velocity_solve(velocity_residual)
            pressure = schur_solve(pressure_residual - blocks.divergence(predictor))
            velocity = self._correct_velocity(blocks, inverse_diagonal, predictor, pressure)
            return velocity, pressure

        return solve


class SimplerComposition(SaddleComposition):
    """SIMPLER: predict the pressure first, solve momentum at it, then correct both.

    Klaij & Vuik (2013) Algorithm 2. The distinguishing step is the **pressure prediction** ``δp''``
    that runs *before* the velocity solve, so the momentum block is inverted against an already
    plausible pressure rather than against none:

    1. ``Ŝ δp'' = -B F̃⁻¹ r_u`` — the prediction. It comes from the momentum row alone
       (``G δp'' = r_u`` at zero velocity, multiplied through by ``-B F̃⁻¹``), deliberately *not*
       from the mass row, which is what keeps the wide-stencil ``Ĉ`` out of the operator being
       inverted.
    2. ``F δu' = r_u - G δp''`` — the velocity solve at the predicted pressure.
    3. ``Ŝ δp' = r_p - B δu' - Ĉ δp''`` — the pressure correction. ``Ĉ`` appears here, on the
       right-hand side, where only its *action* is needed.
    4. ``δu = δu' - F̃⁻¹ G δp'`` and ``δp = δp'' + δp'``.

    The paper's Algorithm 2 divides the prediction by the outer relaxation ``ω_p`` in step 4, so that
    a stationary iteration's ``ω_p`` relaxes only the correction. There is no outer relaxation in a
    Newton--Krylov solve, which is the ``ω_p = 1`` limit, so the two pressures simply add.

    It is the dearest member — two Schur solves and four residual linearizations against the
    triangular pass's one and one — and the one the paper measures as needing far fewer linear
    iterations per nonlinear iteration.
    """

    def apply(
        self,
        blocks: FlowBlocks,
        velocity_solve: _VelocitySolve,
        schur_solve: _PressureSolve,
        inverse_diagonal: jnp.ndarray,
    ) -> _SaddleSolve:
        def solve(
            velocity_residual: jnp.ndarray, pressure_residual: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            predicted = schur_solve(-blocks.divergence(scale(velocity_residual, inverse_diagonal)))
            # One linearization yields both halves of the pressure column: the gradient the momentum
            # solve is driven against, and the coupling the correction's right-hand side needs.
            gradient, coupling = blocks.pressure_column(predicted)
            predictor = velocity_solve(velocity_residual - gradient)
            correction = schur_solve(pressure_residual - blocks.divergence(predictor) - coupling)
            velocity = self._correct_velocity(blocks, inverse_diagonal, predictor, correction)
            return velocity, predicted + correction

        return solve


_COMPOSITIONS: dict[str, type[SaddleComposition]] = {
    "triangular": BlockTriangularComposition,
    "simple": SimpleComposition,
    "simpler": SimplerComposition,
}


def _build_composition(composition: str) -> SaddleComposition:
    """The composition strategy :meth:`BlockPreconditioner.build`'s ``composition`` selects."""
    if composition not in _COMPOSITIONS:
        raise ValueError(
            f"unknown composition {composition!r}; use "
            + ", ".join(repr(name) for name in _COMPOSITIONS)
        )
    return _COMPOSITIONS[composition]()


# --- the composed preconditioner -------------------------------------------------------


class BlockPreconditioner(eqx.Module):
    """A block SIMPLE preconditioner composing a velocity solve and a pressure-Schur inner solve.

    Built from a flow assembler by :meth:`build`; :meth:`factory` returns the ``state -> M`` callable
    :func:`~aquaflux.solve.newton.newton_step` expects. It is **block-triangular**: the pressure block
    additionally sees the divergence of the velocity predictor (``δp = Ŝ⁻¹(r_p − D·δu)``), giving the
    Murphy--Golub--Wathen 2-eigenvalue structure; ``D·δu`` is a ``jvp`` through the frozen residual, so
    ``D`` is a constant operator (adjoint-transparent).

    The pressure Schur is scaled either by the momentum diagonal ``a_P`` (SIMPLE, ``Ŝ ~ B diag(V/a_P)
    B^T``) or, when ``schur_mass_diagonal`` is set, by a frozen velocity-independent diagonal
    ``Q̂ = ρ V / k`` (the mass-matrix scaling), giving the constant-coefficient pressure Poisson
    ``Ŝ ~ B diag(k/ρ) B^T``. Because ``Q̂`` does not track the velocity, the mass-scaled Schur — unlike
    ``V/a_P`` — does not degrade as convection strengthens (Klaij & Vuik 2013, for exactly this
    collocated-FV coupled discretization), which carries the coupled solve past the Reynolds number
    where the ``a_P``-Schur stalls. The hierarchy is frozen at the mass matrix ``ρ V`` (``k = 1``); the
    scale ``k`` is applied per iterate in :meth:`apply_at`, auto-calibrated to ``mean(rho V / a_P)`` from
    the real momentum diagonal (see :meth:`_mass_scale`) so its magnitude matches the ``a_P`` Schur
    at the operating convection with no assumption on the characteristic speed. Only the Schur uses
    ``Q̂``; the velocity block always uses the true ``a_P``.
    """

    assembler: MomentumContinuity
    schur: InnerSchurSolver
    velocity: VelocityBlockSolver
    composition: SaddleComposition = BlockTriangularComposition()
    schur_mass_diagonal: jnp.ndarray | None = None
    mass_scale: float | None = eqx.field(static=True, default=None)

    @classmethod
    def build(
        cls,
        assembler: MomentumContinuity,
        *,
        velocity: str = "smoothed",
        reference_state: jnp.ndarray | None = None,
        schur_scaling: str = "simple",
        composition: str = "triangular",
        mass_scale: float | None = None,
        v_cycles: int = 1,
        strength_threshold: float = 0.0,
    ) -> BlockPreconditioner:
        """Build the block-triangular preconditioner for ``assembler``.

        Parameters
        ----------
        assembler : MomentumContinuity
            The coupled flow residual assembler.
        velocity : {"smoothed", "convection", "convection-air"}
            The velocity-block strategy. ``"smoothed"`` builds an AMG on the viscous (symmetric)
            momentum operator — mesh-independent but Peclet-blind, so it bounds the reachable Reynolds
            number. ``"convection"`` and ``"convection-air"`` build a convection-aware hierarchy on the
            frozen ``viscous + first-order-upwind`` operator (see :class:`SmoothedAmgConvectionVelocity`),
            which stays a good momentum-block approximation as convection strengthens: ``"convection"``
            is the stable two-level method, ``"convection-air"`` the reduction-based (lAIR) hierarchy
            that is Peclet-robust *and* mesh-independent (scales to large meshes). Both freeze their
            convective linearization at ``reference_state``, taken from the boundary conditions unless
            given.
        reference_state : jnp.ndarray, optional
            A representative operating flow state whose Rhie--Chow mass flux freezes the convective
            linearization of the convection-aware velocity blocks. ``None`` (default) derives one from
            the boundary conditions — a uniform flow at the fastest velocity any patch prescribes — so
            the frozen linearization carries the operating cell Peclet with no assumption on the flow
            speed. Pass a state only to pin the linearization to a better-known operating point (for
            instance a previously converged flow).
        schur_scaling : {"simple", "msimple"}
            Which pressure-Schur approximation to use. ``"simple"`` uses the momentum diagonal ``a_P``
            (the classical SIMPLE Schur ``V / a_P``, which degrades as convection strengthens);
            ``"msimple"`` uses a **frozen, velocity-independent** diagonal ``Q̂ = ρ V / k`` so the
            Schur is a constant-coefficient pressure Poisson (coefficient ``k · A/(d·n)``) that stays
            Re-robust — the fix that carries a **flow-only** solve past the Reynolds number at which
            the ``a_P`` Schur's inner solve stalls. Both are *scaled Laplacians*, hence near-Stokes
            approximations that eventually stop representing the Schur complement as convection grows,
            at which point inverting them more accurately does not help. Inside a coupled
            flow--turbulence solve the choice between them does not move the converged state, and the
            coupled block-diagonal preconditioner keeps this parameter's default.
        composition : {"triangular", "simple", "simpler"}
            How the velocity and Schur solves compose into ``M`` (see :class:`SaddleComposition`).
            ``"triangular"`` (default) is the lower block-triangular pass — one of each solve;
            ``"simple"`` adds the closing velocity update that makes it the full block ``LU``;
            ``"simpler"`` additionally predicts the pressure *before* the velocity solve, at the cost
            of a second Schur solve. This axis is independent of ``schur_scaling``: the method Klaij &
            Vuik call **MSIMPLER** is ``schur_scaling="msimple", composition="simpler"``, and their
            **SIMPLER** is ``schur_scaling="simple", composition="simpler"``. The prediction is derived
            for a Schur of the form ``-B F̃⁻¹ G``, which is what both scaled Laplacians are.
        mass_scale : float, optional
            The mass-scaled Schur's ``k`` (only for ``schur_scaling="msimple"``). It sets the Schur
            magnitude to the operating convection, or the block preconditioner is unbalanced and
            stalls. ``None`` (default) calibrates it automatically, per iterate, to ``mean(rho V / a_P)``
            from the **real** momentum diagonal at the current flow — which encodes the true velocity
            / density / viscosity scale, so it matches the SIMPLE Schur magnitude with no assumption
            on the characteristic speed. Pass an explicit value only to pin ``k`` (e.g. for a study).
        v_cycles : int
            Multigrid V-cycles per apply. Raising it does **not** rescue the high-Reynolds coupled solve:
            at high cell Peclet the block's accuracy is limited by the *Schur approximation*, not by how
            well that approximation is inverted, so extra velocity cycles leave the preconditioned error
            operator ``I - A M`` unchanged and extra Schur cycles make it worse (inverting the wrong
            operator more accurately). See the regime note on :class:`SmoothedAmgSchur`.
        strength_threshold : float
            Strength-of-connection threshold for the velocity and Schur AMG aggregation (default ``0`` =
            the isotropic aggregation on the full graph). ``> 0`` (e.g. ``0.25``) aggregates only along
            **strong** connections, which is what keeps those V-cycles contracting on a **high-aspect-
            ratio / skewed** mesh — where isotropic aggregation coarsens across the stiff (wall-normal)
            direction and the V-cycle stalls (contraction → 1 as the aspect ratio grows). It is a no-op
            on a low-aspect-ratio mesh and does not apply to the reduction-based ``convection-air``
            block (whose coarsening is already strength-based). It makes the coarsening
            **value-dependent**, so use it where the hierarchy is frozen (as the coupled flow block is)
            rather than refreshed — see :func:`~aquaflux.solve.build_smoothed_hierarchy`.
        """
        if not isinstance(velocity, VelocityBlock) and velocity not in (
            "smoothed",
            "convection",
            "convection-air",
        ):
            raise ValueError(
                f"unknown velocity block {velocity!r}; use 'smoothed', 'convection' or 'convection-air'"
            )
        if schur_scaling not in ("simple", "msimple"):
            raise ValueError(f"unknown schur_scaling {schur_scaling!r}; use 'simple' or 'msimple'")
        composition_strategy = _build_composition(composition)
        geometry = _SchurGeometry.of(assembler)
        n_cells = assembler.mesh.n_cells
        owner_e, nb_e, _ = assembler.mesh.face_cells.interior_edges()
        interior = np.asarray(assembler.mesh.face_cells.interior)

        # The mass scaling replaces the SIMPLE Schur scaling ``a_P`` with a frozen, velocity-independent
        # diagonal ``Q̂ = ρ V / k``; ``None`` keeps the classical SIMPLE (a_P) Schur. The hierarchy is
        # built at the **mass matrix ``ρ V`` (k = 1)** — the constant-coefficient pressure Poisson
        # ``A/(d·n)``; the operating scale ``k`` is applied per iterate in :meth:`apply_at` (the
        # symmetric rescaling absorbs a scalar exactly). ``k`` is auto-calibrated there to
        # ``mean(ρV / a_P)`` from the **real** momentum diagonal (in the same ``ρV`` units as ``Q̂``, so
        # the density is not divided back out of the Schur coefficient), so it tracks the true velocity /
        # density / viscosity scale with no unit-speed assumption; ``mass_scale`` overrides it.
        mass_diagonal = jax.lax.stop_gradient(assembler.density * assembler.geometry.cell.volume)
        # `None` keeps `apply_at` from applying the per-iterate `k` calibration and rescaling, which only
        # the mass scaling wants.
        schur_mass_diagonal = mass_diagonal if schur_scaling == "msimple" else None

        # A convection-aware velocity block freezes its linearization at a representative flow state;
        # derive one from the boundary conditions when none was given.
        freezes_convection = (
            isinstance(velocity, _ConvectionVelocityBlock)
            if isinstance(velocity, VelocityBlock)
            else velocity in ("convection", "convection-air")
        )
        if reference_state is None and freezes_convection:
            reference_state = _characteristic_reference_state(assembler)

        inputs = _StrategyInputs(
            owner_e,
            nb_e,
            interior,
            n_cells,
            v_cycles,
            strength_threshold,
            assembler,
            reference_state,
        )
        schur = _build_schur(geometry, inputs, schur_mass_diagonal)
        velocity_geometry = _VelocityGeometry.of(assembler)
        velocity_block = (
            velocity._build(velocity_geometry, inputs)
            if isinstance(velocity, VelocityBlock)
            else _build_velocity_block(velocity, velocity_geometry, inputs)
        )
        return cls(
            assembler,
            schur,
            velocity_block,
            composition=composition_strategy,
            schur_mass_diagonal=schur_mass_diagonal,
            mass_scale=mass_scale,
        )

    def frozen_momentum_diagonal(self, state: jnp.ndarray) -> jnp.ndarray:
        """The isotropic, frozen momentum diagonal ``a_P`` at ``state``, shape ``(n_cells,)``.

        Isotropic (component-averaged) ``a_P`` for the Schur/velocity blocks; the directional
        per-component form enters only the operator's Rhie--Chow coefficient. The preconditioner
        needs ``a_P`` frozen, so ``stop_gradient`` it here (the residual uses the differentiable
        ``a_P``): the state is already detached, and this keeps ``M`` a constant operator even if
        called on a live state. Exposed so a continuation driver (implicit under-relaxation) can
        form its diagonal shift from the *same* ``a_P`` the preconditioner inverts.
        """
        return _isotropic_momentum_diagonal(self.assembler, state)

    def frozen_momentum_diagonal_parts(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The convective/dissipative buckets of the frozen ``a_P`` at ``state``, each ``(n_cells,)``.

        The split a :class:`~aquaflux.solve.ShiftBasis` combines into a local-time-step pseudo-transient
        shift; their sum is :meth:`frozen_momentum_diagonal` (to rounding). Exposed so a continuation
        driver can build a convective (or weighted) shift while still inverting the total ``a_P``.
        """
        return frozen_momentum_diagonal_parts(self.assembler, state)

    def _mass_scale(self, state: jnp.ndarray) -> jnp.ndarray:
        """The mass-scaled Schur's ``k`` at ``state`` — ``mean(ρV / a_P)`` from the real ``a_P``.

        ``k`` sets the frozen, velocity-independent Schur diagonal ``schur_a_P = Q̂ / k = ρV / k``, an
        ``a_P``-magnitude stand-in for the real momentum diagonal, so the Schur coefficient
        ``ρ_f (V/schur_a_P)_f A/(d·n)`` matches the SIMPLE coefficient ``ρ_f (V/a_P)_f A/(d·n)`` at the
        operating convection. ``k`` is calibrated in the **same ``ρV`` units as ``Q̂``** (reusing the
        frozen :attr:`schur_mass_diagonal`), so the density ``Q̂`` carries is not divided back out — the
        assembled coefficient keeps its ``ρ`` factor for ρ≠1 (air, water), not only at ρ=1. Taken from
        the **actual** momentum diagonal ``a_P`` at the current flow, which encodes the true velocity /
        density / viscosity scale, so a non-unit-speed problem calibrates itself with **no unit-speed
        assumption**. The **un-shifted** diagonal is used (via :meth:`frozen_momentum_diagonal`, not the
        continuation's shifted ``a_P``): an early large pseudo-transient shift would give a spuriously
        large ``a_P``, hence a spuriously weak Schur. ``mass_scale`` overrides the calibration with a
        fixed value. Frozen (``stop_gradient``) like :meth:`frozen_momentum_diagonal`, so the scale never
        leaks a live cell-volume or density gradient into the adjoint.
        """
        if self.mass_scale is not None:
            return jnp.asarray(float(self.mass_scale))
        a_p = self.frozen_momentum_diagonal(state)
        return jax.lax.stop_gradient(jnp.mean(self.schur_mass_diagonal / a_p))

    def apply_at(
        self, state: jnp.ndarray, a_p: jnp.ndarray
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The preconditioner matvec ``M`` at ``state`` for a supplied (frozen) diagonal ``a_P``.

        Splits the ``state -> M`` factory so a caller can pass an *effective* ``a_P`` — e.g. the
        under-relaxed ``a_P (1 + β)`` an implicit-continuation step uses, matching the shifted
        Jacobian it inverts — instead of always the bare diagonal :meth:`frozen_momentum_diagonal`
        returns. ``a_P`` is the isotropic per-cell diagonal, shape ``(n_cells,)``.

        The velocity block always inverts at the supplied ``a_P``; the Schur uses the frozen
        mass-matrix diagonal ``Q̂ = ρ V / k`` instead when set (velocity-independent, so it ignores the
        continuation shift), with ``k`` calibrated per iterate from the real un-shifted ``a_P`` (see
        :meth:`_mass_scale`); else it uses the supplied ``a_P`` (classical SIMPLE).

        How the two inner solves are then composed into ``M`` is :attr:`composition`; the diagonal it
        needs for its velocity corrections is the reciprocal of whichever diagonal the Schur was
        assembled from, so the two stay consistent whatever the scaling.
        """
        if self.schur_mass_diagonal is None:
            schur_a_p = a_p
        else:  # the mass-scaled Schur: Q̂ = ρ V / k with the operating-scale k
            schur_a_p = self.schur_mass_diagonal / self._mass_scale(state)
        blocks = FlowBlocks.of(self.assembler, state)
        solve = self.composition.apply(
            blocks,
            self.velocity.apply(a_p),
            self.schur.apply(schur_a_p),
            1.0 / schur_a_p,
        )

        def apply(v: jnp.ndarray) -> jnp.ndarray:
            return self.assembler.pack(*solve(*self.assembler.unpack(v)))

        return apply

    def factory(self) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """Return the ``state -> M`` factory the Newton step applies (``M`` frozen at that iterate)."""
        return lambda state: self.apply_at(state, self.frozen_momentum_diagonal(state))
