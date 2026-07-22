"""Block SIMPLE preconditioner for the coupled pressure--velocity Newton solve.

Composes a **velocity-block solve** with a **pressure-Schur inner solve** into the left
preconditioner ``M ≈ J⁻¹`` that :func:`~aquaflux.solve.newton.newton_step` applies to the coupled
saddle-point system. Both the Schur inner solve (:class:`InnerSchurSolver`) and the velocity solve
(:class:`VelocityBlockSolver`) are **swappable strategies**, built once off the jit path from the
assembler's frozen geometry and applied per Newton iterate at the current momentum diagonal ``a_P``.
Every coefficient is ``stop_gradient``-ed, so ``M`` only accelerates the Krylov iteration — it never
perturbs the converged solution or its adjoint.

The inner pressure Schur is a **smoothed-aggregation multigrid** (:class:`SmoothedAmgSchur`,
mesh-independent V-cycle contraction ~0.25), paired with a velocity-block AMG
(:class:`SmoothedAmgVelocity` on the viscous operator, or :class:`SmoothedAmgConvectionVelocity` on
the convection-diffusion operator) and the block-triangular ``D·δu`` coupling. Both strategy families
are abstract interfaces (:class:`InnerSchurSolver` / :class:`VelocityBlockSolver`), the seam a new
inner solver or velocity block plugs into.
"""

from __future__ import annotations

import abc
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum

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

from .preconditioner import schur_face_coefficient
from .rhie_chow import momentum_diagonal, viscous_face_coefficient
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
    interp_factor: jnp.ndarray
    normal_distance: jnp.ndarray
    viscosity: jnp.ndarray
    dim: int = eqx.field(static=True)

    @classmethod
    def of(cls, assembler: MomentumContinuity) -> _VelocityGeometry:
        """Extract the velocity-block geometry from a flow assembler."""
        return cls(
            assembler.mesh.face_cells,
            assembler.geometry,
            assembler.interp_factor,
            assembler.normal_distance,
            assembler.viscosity,
            assembler.mesh.dim,
        )


# --- pressure-Schur inner solvers (strategy family) ------------------------------------


class InnerSchurSolver(eqx.Module):
    """Strategy: solve the compact pressure Schur ``Ŝ x = rp`` for the preconditioner.

    Built once off the jit path; :meth:`apply` returns the solve ``rp -> Ŝ⁻¹ rp`` specialized to the
    current (frozen) momentum diagonal ``a_P`` and the frozen saddle blocks at the current iterate.

    Both arguments are offered because the family spans two kinds of approximation: a *scaled-Laplacian*
    Schur (:class:`SmoothedAmgSchur`) is a pure function of ``a_P`` and ignores ``blocks``, whereas a
    *commutator-based* Schur (:class:`StabilizedLscSchur`) needs the momentum, gradient, and divergence
    operators themselves. Taking both keeps the seam one method rather than branching the caller.
    """

    @abc.abstractmethod
    def apply(self, a_p: jnp.ndarray, blocks: FlowBlocks) -> _PressureSolve:
        """Return the pressure solve ``rp -> Ŝ⁻¹ rp`` at momentum diagonal ``a_P``.

        Parameters
        ----------
        a_p : jnp.ndarray
            The frozen isotropic momentum diagonal, shape ``(n_cells,)``.
        blocks : FlowBlocks
            The saddle's matrix-free Jacobian blocks at the current frozen state.
        """


class SmoothedAmgSchur(InnerSchurSolver):
    """Smoothed-aggregation multigrid, mesh-independent (V-cycle contraction ~0.25).

    The hierarchy is frozen at a reference coefficient; the current operator's scale is tracked by a
    symmetric diagonal rescaling ``Ŝ_cur⁻¹ ≈ D⁻¹ Ŝ_ref⁻¹ D⁻¹``, ``D = sqrt(diag_cur/diag_ref)`` — exact
    for a uniform rescale, and capturing per-cell scale (including convection) otherwise.

    Regime limit (measured, and the reason ``v_cycles`` is not a high-Reynolds lever): with the
    ``"msimpler"`` scaling this is a **constant-coefficient** pressure Poisson — a near-Stokes
    approximation of the true Schur complement that degrades as convection strengthens. Once the flow is
    convection-dominated (high Reynolds number, recirculation) that *approximation* — not its inversion —
    sets the outer Krylov cost: inverting it more accurately does not help and can hurt, and neither
    rescaling it nor rebuilding it at a developed state recovers the loss. Escaping that ceiling needs a
    genuinely better Schur approximation, such as the stabilized least-squares-commutator preconditioner
    of Elman, Howle, Shadid, Silvester & Tuminaro (2007) — which reuses this class's assembled pressure
    Poisson. The *stabilized* variant is the relevant one: a Rhie--Chow collocated discretization is
    equal-order stabilized, which the original least-squares-commutator form does not account for.
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
    ) -> SmoothedAmgSchur:
        # Reference diagonal for the frozen hierarchy, fed to the Schur coefficient ``V / d``. SIMPLE
        # uses a unit-viscosity momentum ``a_P`` (the multigrid is scale-invariant, so a concrete
        # reference keeps the scipy build valid even inside a differentiated region), with the
        # per-iterate convective ``a_P`` restored by the symmetric rescaling in :meth:`apply`.
        # MSIMPLER instead supplies the velocity mass-matrix diagonal ``rho V`` — a velocity-
        # independent scaling that does not degrade as convection strengthens, so its rescaling is the
        # identity. The isotropic (component-averaged) form is used; the directional per-component
        # ``a_P`` enters only the operator's Rhie--Chow coefficient.
        if reference_diagonal is None:
            reference_diagonal = jnp.mean(
                momentum_diagonal(
                    geometry.face_cells,
                    geometry.mesh_geometry,
                    jnp.ones(n_cells),
                    geometry.normal_distance,
                    geometry.interp_factor,
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
        hierarchy = build_smoothed_hierarchy(a)
        return cls(geometry, hierarchy, v_cycles)

    def apply(self, a_p: jnp.ndarray, blocks: FlowBlocks) -> _PressureSolve:
        # `blocks` is unused: this Schur is a scaled discrete Laplacian in `a_P` alone. It is part of
        # the interface for the commutator-based strategies, which do need the saddle's operators.
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
    ) -> SmoothedAmgVelocity:
        face_cells = geometry.face_cells
        # Unit-viscosity viscous coefficient A/(d·n) — the geometry-only part of the momentum
        # diagonal, rescaled to the current a_P in :meth:`apply`. From the shared coefficient with a
        # unit viscosity, so the reference operator matches the momentum diagonal's viscous term.
        over_distance = viscous_face_coefficient(
            jnp.ones(n_cells),
            geometry.normal_distance,
            geometry.interp_factor,
            face_cells,
            geometry.mesh_geometry,
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
        hierarchy = build_smoothed_hierarchy(a)
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
    ) -> SmoothedAmgConvectionVelocity:
        # ``reference_mdot`` is the (frozen) Rhie--Chow mass flux of a representative operating state
        # -- the convective linearization the hierarchy is frozen at, so the operator diagonal matches
        # the momentum diagonal ``a_P`` at the reference and the per-iterate rescaling is exact. It is
        # supplied by the caller because it is assembler behaviour (the flux operator), not geometry.
        face_cells = geometry.face_cells
        mu = jax.lax.stop_gradient(geometry.viscosity)
        # Central viscous coefficient, from the shared definition so the frozen operator's viscous term
        # cannot drift from the momentum diagonal's.
        viscous = viscous_face_coefficient(
            mu, geometry.normal_distance, geometry.interp_factor, face_cells, geometry.mesh_geometry
        )
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
            # mesh-independent, so it scales where the two-level direct coarse solve cannot.
            hierarchy = build_air_hierarchy(a)
        elif method == "twolevel":
            # A single aggregation with a direct coarse solve: the aggregation coarse space stays a
            # stable correction at high cell Peclet, where a deeper Galerkin recursion does not (the
            # builder is two-level for exactly this reason).
            hierarchy = build_convection_hierarchy(a)
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
    with an *odd* number of gradient factors — such as the least-squares commutator
    ``B Q̂⁻¹ F Q̂⁻¹ Bᵀ`` — picks up a sign flip against the literature formula.

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

        Positive definite as the residual writes it (see the sign convention above), which is the sign
        the stabilized least-squares-commutator Schur approximation is written in terms of.
        """
        return self.pressure_column(dp)[1]


# The commutator Schur inverts `P_γ` **twice** per apply, so its inversion error compounds — unlike
# the scaled-Laplacian Schurs, which invert theirs once. A single V-cycle is too inexact and the whole
# approximation breaks down: measured on a convection-dominated channel, one cycle fails to converge at
# all, while four or more converge in *fewer* outer iterations than either scaled-Laplacian Schur. This
# is the floor the strategy enforces on the shared V-cycle count.
_COMMUTATOR_MIN_V_CYCLES = 4


def _spectral_radius(matvec: Callable[[np.ndarray], np.ndarray], n: int, iterations: int) -> float:
    """Dominant eigenvalue magnitude of a linear operator, by power iteration (off the jit path).

    Used to size the two scalar parameters of the stabilized least-squares-commutator Schur. A plain
    power iteration is enough: both scalars only set the *balance* between the preconditioner's two
    additive parts, so a few significant figures suffice, and it needs no eigensolver dependency.
    """
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    magnitude = 0.0
    for _ in range(iterations):
        w = matvec(v)
        magnitude = float(np.linalg.norm(w))
        if magnitude == 0.0:
            return 0.0
        v = w / magnitude
    return magnitude


class StabilizedLscSchur(InnerSchurSolver):
    """Stabilized least-squares-commutator (LSC) Schur approximation, for convection-dominated flow.

    :class:`SmoothedAmgSchur` approximates the Schur complement by a *scaled discrete Laplacian*. That
    is a near-Stokes approximation: as convection strengthens it stops representing
    ``S = B F⁻¹ Bᵀ + Ĉ``, and — this is the practical point — no amount of extra accuracy in *inverting*
    it recovers the loss, because the error is in the approximation rather than its inversion. This
    strategy instead builds the Schur approximation from the momentum operator itself, via the
    least-squares commutator of Elman, Howle, Shadid, Shuttleworth & Tuminaro (2006), in the
    **stabilized** form of Elman, Howle, Shadid, Silvester & Tuminaro (2007).

    The stabilized form is the required one here: a collocated Rhie--Chow discretization is *equal-order
    stabilized*, so the saddle's pressure--pressure block ``Ĉ`` (the Rhie--Chow pressure damping) is
    nonzero and the unstabilized commutator is singular on the checkerboard pressure mode. Of the two
    stabilized variants in that work, this implements the **algebraic** one, which needs only assembled
    operators — the element-based variant needs local finite-element assembly information that a
    cell-centred finite-volume solver does not have. Specifically it is the nonuniform-mesh form,

    ``M_S⁻¹ = P_γ⁻¹ (B Q̂⁻¹ F Q̂⁻¹ Bᵀ) P_γ⁻¹ + α D⁻¹``,
    ``P_γ = B Q̂⁻¹ Bᵀ + γ̃ D_r^½ Ĉ D_r^½``,

    with ``Q̂`` the velocity mass diagonal ``ρV``, ``D_r`` the componentwise ratio of ``diag(B Q̂⁻¹ Bᵀ)``
    to ``diag(Ĉ)`` (which makes the added dissipation's spatial variation follow the Laplacian's, the
    adaptation that carries the method to graded and unstructured meshes), and ``D`` the diagonal of
    ``B diag(F)⁻¹ Bᵀ + Ĉ``. The ``α D⁻¹`` term is what keeps the checkerboard mode bounded, and the
    ``γ̃`` term is what keeps the commutator well defined on it.

    **Both scalars are viscosity-free here, by construction.** As published, ``γ = ρ(Q̂⁻¹F)/(3ν)`` and
    ``D`` carry an explicit kinematic viscosity ``ν``. A turbulent flow has no single ``ν`` — the
    effective viscosity ``μ + ρν_t`` varies across the field by orders of magnitude — so that form is
    not directly usable. Writing the expressions in terms of the *assembled* pressure--pressure block
    (``Ĉ``, which already carries the viscosity scaling) rather than a bare stabilization matrix, the
    ``ν`` cancels identically: ``D_r`` scales as ``1/ν``, so ``γ̃ = γ/‖diag(D_r)‖_∞`` is
    ``ρ(Q̂⁻¹F)/(3‖diag(D_r)‖_∞)`` and ``D_r^½ Ĉ D_r^½`` is unchanged. The implementation therefore never
    needs a viscosity value, which is what lets it serve a variable-viscosity turbulent closure.

    Cost, relative to the scaled-Laplacian Schur: two multigrid solves and three residual
    linearizations per apply, against one solve. It buys a Schur approximation that keeps working when
    the cheap one has stopped.
    """

    geometry: _SchurGeometry
    hierarchy: SmoothedHierarchy
    mass_diagonal: jnp.ndarray
    alpha_diagonal: jnp.ndarray
    alpha: float = eqx.field(static=True)
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
        mass_diagonal: jnp.ndarray,
        reference_a_p: jnp.ndarray,
        momentum_radius: float,
        *,
        gamma: float | None = None,
        alpha: float | None = None,
        power_iterations: int = 50,
    ) -> StabilizedLscSchur:
        """Assemble ``P_γ``'s multigrid hierarchy and calibrate the two scalars, off the jit path.

        Parameters
        ----------
        geometry : _SchurGeometry
            The frozen pressure-Schur geometry (owns the face coefficient at a given diagonal).
        owner_e, nb_e, interior : np.ndarray
            Interior-edge owner/neighbour indices and the interior-face mask.
        n_cells : int
            Number of cells.
        v_cycles : int
            Multigrid V-cycles per ``P_γ`` solve, raised to at least ``_COMMUTATOR_MIN_V_CYCLES``
            (this Schur inverts ``P_γ`` twice, so too inexact an inner solve breaks it).
        mass_diagonal : jnp.ndarray
            The velocity mass diagonal ``Q̂ = ρV``, shape ``(n_cells,)``.
        reference_a_p : jnp.ndarray
            The momentum diagonal the stabilization block is frozen at, shape ``(n_cells,)``.
        momentum_radius : float
            The spectral radius of ``Q̂⁻¹F``, used only to size ``γ``. Computed by the builder (which
            owns the assembler) and handed in, so this build stays assembler-free.
        gamma, alpha : float, optional
            Override the calibrated scalars (for a parameter study). ``None`` calibrates them.
        power_iterations : int
            Power-iteration count for the two spectral radii.
        """
        import scipy.sparse as sp

        # The two pressure-space operators, both from the one shared Schur-coefficient definition so
        # they cannot drift: the Laplacian `B Q̂⁻¹ Bᵀ` is that coefficient at the mass diagonal, and the
        # stabilization block `Ĉ` (Rhie--Chow pressure damping) is the same coefficient at `a_P`.
        laplacian = cls._pressure_operator(
            geometry, owner_e, nb_e, interior, n_cells, mass_diagonal
        )
        stabilization = cls._pressure_operator(
            geometry, owner_e, nb_e, interior, n_cells, reference_a_p
        )

        laplacian_diagonal = np.asarray(laplacian.diagonal())
        stabilization_diagonal = np.asarray(stabilization.diagonal())
        safe = np.where(np.abs(stabilization_diagonal) > 0.0, stabilization_diagonal, 1.0)
        ratio = laplacian_diagonal / safe  # D_r

        if gamma is None:
            # γ = ρ(Q̂⁻¹F) / 3, then normalized by ‖diag(D_r)‖_∞ (the viscosity cancels — see the class
            # docstring), giving the scale at which the stabilization enters `P_γ`.
            gamma = momentum_radius / 3.0
        scaled_gamma = gamma / max(float(np.max(np.abs(ratio))), 1e-300)

        root = np.sqrt(np.abs(ratio))
        p_gamma = laplacian + scaled_gamma * (sp.diags(root) @ stabilization @ sp.diags(root))
        if geometry.pressure_pin is not None:  # closed domain: regularize by decoupling the pin
            p_gamma = decouple_dof(p_gamma, geometry.pressure_pin)

        # The additive term α D⁻¹ that bounds the checkerboard mode. `B diag(F)⁻¹ Bᵀ` is the Schur
        # coefficient at the momentum diagonal — the same assembled operator as the stabilization
        # block here, since both are that Laplacian at `a_P`.
        alpha_diagonal_np = stabilization_diagonal + stabilization_diagonal
        if alpha is None:
            inverse_alpha_diagonal = 1.0 / np.where(
                np.abs(alpha_diagonal_np) > 0.0, alpha_diagonal_np, 1.0
            )
            radius = _spectral_radius(
                lambda v: stabilization @ (inverse_alpha_diagonal * v), n_cells, power_iterations
            )
            alpha = 1.0 / radius if radius > 0.0 else 0.0

        return cls(
            geometry,
            build_smoothed_hierarchy(p_gamma),
            jnp.asarray(mass_diagonal),
            jnp.asarray(alpha_diagonal_np),
            float(alpha),
            max(v_cycles, _COMMUTATOR_MIN_V_CYCLES),
        )

    @staticmethod
    def _pressure_operator(
        geometry: _SchurGeometry,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior: np.ndarray,
        n_cells: int,
        diagonal: jnp.ndarray,
    ) -> object:
        """The assembled pressure-space Laplacian ``B diag⁻¹ Bᵀ`` at a given momentum-like diagonal."""
        coefficient = np.asarray(geometry.coefficient(diagonal))[interior]
        boundary = np.asarray(geometry.boundary_diagonal(diagonal))
        return convection_diffusion_operator(
            owner_e, nb_e, coefficient, n_cells, boundary_diagonal=boundary
        )

    def apply(self, a_p: jnp.ndarray, blocks: FlowBlocks) -> _PressureSolve:
        """The stabilized commutator solve ``rp -> M_S⁻¹ rp``.

        ``a_P`` is unused, and deliberately so: unlike the scaled-Laplacian Schurs, whose operator *is*
        a function of the momentum diagonal and so has to be rescaled as it develops, this one is built
        on the velocity mass diagonal ``Q̂ = ρV`` — pure geometry, with no state dependence to track.
        All of this strategy's dependence on the current iterate enters through ``blocks``, i.e. the
        commutator, which is where the convection information actually lives.
        """
        del a_p

        def p_gamma_solve(rp: jnp.ndarray) -> jnp.ndarray:
            return smoothed_multigrid_solve(self.hierarchy, rp, cycles=self.v_cycles)

        inverse_mass = 1.0 / self.mass_diagonal
        alpha_scale = self.alpha / self.alpha_diagonal

        def commutator(pressure: jnp.ndarray) -> jnp.ndarray:
            """``B Q̂⁻¹ F Q̂⁻¹ Bᵀ`` — three linearizations of the frozen residual.

            Negated because this residual's gradient block ``G`` is ``-Bᵀ`` in the textbook saddle
            form the formula is written in (see :class:`FlowBlocks`); with the single gradient factor
            here that is one sign flip, and it is what makes the result positive definite like the
            Schur complement it approximates.
            """
            gradient = blocks.gradient(pressure)
            momentum = blocks.momentum(gradient * inverse_mass[:, None])
            return -blocks.divergence(momentum * inverse_mass[:, None])

        def solve(rp: jnp.ndarray) -> jnp.ndarray:
            return p_gamma_solve(commutator(p_gamma_solve(rp))) + alpha_scale * rp

        return solve


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


def _scaled_momentum_radius(
    assembler: MomentumContinuity,
    state: jnp.ndarray,
    mass_diagonal: jnp.ndarray,
    iterations: int = 30,
) -> float:
    """Spectral radius of ``Q̂⁻¹F`` at ``state``, by power iteration on the frozen momentum block.

    Assembler behaviour (it linearizes the residual), so it is computed here — where the assembler
    lives — and handed to the Schur strategy as a plain number, keeping that strategy assembler-free.
    """
    blocks = FlowBlocks.of(assembler, state)
    inverse_mass = 1.0 / mass_diagonal
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal((assembler.mesh.n_cells, assembler.mesh.dim)))
    v = v / jnp.linalg.norm(v)
    magnitude = 0.0
    for _ in range(iterations):
        w = inverse_mass[:, None] * blocks.momentum(v)
        magnitude = float(jnp.linalg.norm(w))
        if magnitude == 0.0:
            return 0.0
        v = w / magnitude
    return magnitude


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
    ``Q̂ = ρ V / k`` (MSIMPLER, a mass-matrix scaling), giving the constant-coefficient pressure Poisson
    ``Ŝ ~ B diag(k/ρ) B^T``. Because ``Q̂`` does not track the velocity, the MSIMPLER Schur — unlike
    ``V/a_P`` — does not degrade as convection strengthens (Klaij & Vuik 2013, for exactly this
    collocated-FV coupled discretization), which carries the coupled solve past the Reynolds number
    where the ``a_P``-Schur stalls. The hierarchy is frozen at the mass matrix ``ρ V`` (``k = 1``); the
    scale ``k`` is applied per iterate in :meth:`apply_at`, auto-calibrated to ``mean(V / a_P)`` from
    the real momentum diagonal (see :meth:`_msimpler_scale`) so its magnitude matches the SIMPLE Schur
    at the operating convection with no assumption on the characteristic speed. Only the Schur uses
    ``Q̂``; the velocity block always uses the true ``a_P``.
    """

    assembler: MomentumContinuity
    schur: InnerSchurSolver
    velocity: VelocityBlockSolver
    schur_mass_diagonal: jnp.ndarray | None = None
    msimpler_scale: float | None = eqx.field(static=True, default=None)

    @classmethod
    def build(
        cls,
        assembler: MomentumContinuity,
        *,
        velocity: str = "smoothed",
        reference_state: jnp.ndarray | None = None,
        schur_scaling: str = "simple",
        msimpler_scale: float | None = None,
        v_cycles: int = 1,
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
        schur_scaling : {"simple", "msimpler", "lsc"}
            Which pressure-Schur approximation to use. ``"simple"`` uses the momentum diagonal ``a_P``
            (the classical SIMPLE Schur ``V / a_P``, which degrades as convection strengthens);
            ``"msimpler"`` uses a **frozen, velocity-independent** diagonal ``Q̂ = ρ V / k`` so the
            Schur is a constant-coefficient pressure Poisson (coefficient ``k · A/(d·n)``) that stays
            Re-robust — the fix that carries the coupled solve past the ``a_P``-Schur stall. Both are
            *scaled Laplacians*, hence near-Stokes approximations that eventually stop representing the
            Schur complement as convection grows, at which point inverting them more accurately does
            not help. ``"lsc"`` instead builds the approximation from the momentum operator itself
            (:class:`StabilizedLscSchur`, the stabilized least-squares commutator) — markedly dearer per
            apply (two multigrid solves plus three residual linearizations, against one solve) but it
            keeps working in the convection-dominated regime where the scaled Laplacians have stalled.
        msimpler_scale : float, optional
            The MSIMPLER scale ``k`` (only for ``schur_scaling="msimpler"``). It sets the Schur
            magnitude to the operating convection, or the block preconditioner is unbalanced and
            stalls. ``None`` (default) calibrates it automatically, per iterate, to ``mean(V / a_P)``
            from the **real** momentum diagonal at the current flow — which encodes the true velocity
            / density / viscosity scale, so it matches the SIMPLE Schur magnitude with no assumption
            on the characteristic speed. Pass an explicit value only to pin ``k`` (e.g. for a study).
        v_cycles : int
            Multigrid V-cycles per apply. Raising it does **not** rescue the high-Reynolds coupled solve:
            at high cell Peclet the block's accuracy is limited by the *Schur approximation*, not by how
            well that approximation is inverted, so extra velocity cycles leave the preconditioned error
            operator ``I - A M`` unchanged and extra Schur cycles make it worse (inverting the wrong
            operator more accurately). See the regime note on :class:`SmoothedAmgSchur`.
        """
        if velocity not in ("smoothed", "convection", "convection-air"):
            raise ValueError(
                f"unknown velocity block {velocity!r}; use 'smoothed', 'convection' or 'convection-air'"
            )
        if schur_scaling not in ("simple", "msimpler", "lsc"):
            raise ValueError(
                f"unknown schur_scaling {schur_scaling!r}; use 'simple', 'msimpler' or 'lsc'"
            )
        geometry = _SchurGeometry.of(assembler)
        n_cells = assembler.mesh.n_cells
        owner_e, nb_e, _ = assembler.mesh.face_cells.interior_edges()
        interior = np.asarray(assembler.mesh.face_cells.interior)

        # MSIMPLER replaces the SIMPLE Schur scaling ``a_P`` with a frozen, velocity-independent
        # diagonal ``Q̂ = ρ V / k``; ``None`` keeps the classical SIMPLE (a_P) Schur. The hierarchy is
        # built at the **mass matrix ``ρ V`` (k = 1)** — the constant-coefficient pressure Poisson
        # ``A/(d·n)``; the operating scale ``k`` is applied per iterate in :meth:`apply_at` (the
        # symmetric rescaling absorbs a scalar exactly). ``k`` is auto-calibrated there to
        # ``mean(ρV / a_P)`` from the **real** momentum diagonal (in the same ``ρV`` units as ``Q̂``, so
        # the density is not divided back out of the Schur coefficient), so it tracks the true velocity /
        # density / viscosity scale with no unit-speed assumption; ``msimpler_scale`` overrides it.
        mass_diagonal = jax.lax.stop_gradient(assembler.density * assembler.geometry.cell.volume)
        # Only MSIMPLER reinterprets the Schur's diagonal as a mass matrix scaled per iterate by `k`.
        # The commutator Schur uses the mass diagonal directly (it is `Q̂` in the least-squares
        # commutator, not a stand-in for `a_P`), so it wants no `k` calibration and no per-iterate
        # rescaling — leaving this None keeps `apply_at` from applying either.
        schur_mass_diagonal = mass_diagonal if schur_scaling == "msimpler" else None

        schur: InnerSchurSolver
        if schur_scaling == "lsc":
            if reference_state is None:
                reference_state = _characteristic_reference_state(assembler)
            reference_a_p = _isotropic_momentum_diagonal(assembler, reference_state)
            schur = StabilizedLscSchur.build(
                geometry,
                owner_e,
                nb_e,
                interior,
                n_cells,
                v_cycles,
                mass_diagonal,
                reference_a_p,
                _scaled_momentum_radius(assembler, reference_state, mass_diagonal),
            )
        else:
            schur = SmoothedAmgSchur.build(
                geometry,
                owner_e,
                nb_e,
                interior,
                n_cells,
                v_cycles,
                reference_diagonal=schur_mass_diagonal,
            )
        velocity_geometry = _VelocityGeometry.of(assembler)
        velocity_block: VelocityBlockSolver
        if velocity in ("convection", "convection-air"):
            if reference_state is None:
                reference_state = _characteristic_reference_state(assembler)
            # The reference mass flux is assembler behaviour (the Rhie--Chow flux operator), so it is
            # computed here and handed to the strategy, keeping the velocity build assembler-free.
            reference_mdot = jax.lax.stop_gradient(assembler.mass_flux(reference_state))
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
                    stacklevel=2,
                )
            velocity_block = SmoothedAmgConvectionVelocity.build(
                velocity_geometry,
                owner_e,
                nb_e,
                interior,
                n_cells,
                v_cycles,
                reference_mdot,
                method="air" if velocity == "convection-air" else "twolevel",
            )
        else:
            velocity_block = SmoothedAmgVelocity.build(
                velocity_geometry, owner_e, nb_e, interior, n_cells, v_cycles
            )
        return cls(
            assembler,
            schur,
            velocity_block,
            schur_mass_diagonal=schur_mass_diagonal,
            msimpler_scale=msimpler_scale,
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

    def _msimpler_scale(self, state: jnp.ndarray) -> jnp.ndarray:
        """The MSIMPLER scale ``k`` at ``state`` — ``mean(ρV / a_P)`` from the real momentum diagonal.

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
        large ``a_P``, hence a spuriously weak Schur. ``msimpler_scale`` overrides the calibration with a
        fixed value. Frozen (``stop_gradient``) like :meth:`frozen_momentum_diagonal`, so the scale never
        leaks a live cell-volume or density gradient into the adjoint.
        """
        if self.msimpler_scale is not None:
            return jnp.asarray(float(self.msimpler_scale))
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

        The velocity block always inverts at the supplied ``a_P``; the Schur uses the frozen MSIMPLER
        mass-matrix diagonal ``Q̂ = ρ V / k`` instead when set (velocity-independent, so it ignores the
        continuation shift), with ``k`` calibrated per iterate from the real un-shifted ``a_P`` (see
        :meth:`_msimpler_scale`); else it uses the supplied ``a_P`` (classical SIMPLE).
        """
        if self.schur_mass_diagonal is None:
            schur_a_p = a_p
        else:  # MSIMPLER: Q̂ = ρ V / k with the operating-scale k
            schur_a_p = self.schur_mass_diagonal / self._msimpler_scale(state)
        blocks = FlowBlocks.of(self.assembler, state)
        schur_solve = self.schur.apply(schur_a_p, blocks)
        velocity_solve = self.velocity.apply(a_p)
        divergence = blocks.divergence

        def apply(v: jnp.ndarray) -> jnp.ndarray:
            velocity_residual, pressure_residual = self.assembler.unpack(v)
            velocity_correction = velocity_solve(velocity_residual)
            # Block-triangular: the pressure block sees the velocity predictor's divergence D·δu.
            pressure_residual = pressure_residual - divergence(velocity_correction)
            return self.assembler.pack(velocity_correction, schur_solve(pressure_residual))

        return apply

    def factory(self) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """Return the ``state -> M`` factory the Newton step applies (``M`` frozen at that iterate)."""
        return lambda state: self.apply_at(state, self.frozen_momentum_diagonal(state))
