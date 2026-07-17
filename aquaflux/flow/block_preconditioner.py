"""Block SIMPLE preconditioner for the coupled pressure--velocity Newton solve.

Composes a **velocity-block solve** with a **pressure-Schur inner solve** into the left
preconditioner ``M ≈ J⁻¹`` that :func:`~aquaflux.solve.newton.newton_step` applies to the coupled
saddle-point system. Both the Schur inner solve (:class:`InnerSchurSolver`) and the velocity solve
(:class:`VelocityBlockSolver`) are **swappable strategies**, built once off the jit path from the
assembler's frozen geometry and applied per Newton iterate at the current momentum diagonal ``a_P``.
Every coefficient is ``stop_gradient``-ed, so ``M`` only accelerates the Krylov iteration — it never
perturbs the converged solution or its adjoint.

The three inner Schur strategies trade cost for mesh-independence:

* :class:`SmoothedAmgSchur` — smoothed-aggregation multigrid, mesh-independent (V-cycle contraction
  ~0.25); paired with a velocity-block AMG (:class:`SmoothedAmgVelocity`) and the block-triangular
  ``D·δu`` coupling for the strongest preconditioner.
* :class:`AggregationSchur` — unsmoothed aggregation; better than Jacobi but not mesh-independent.
* :class:`DampedJacobiSchur` — a fixed damped-Jacobi sweep on the assembled pressure Laplacian.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum

from aquaflux.solve.multigrid import (
    build_hierarchy,
    build_smoothed_hierarchy,
    level_coefficients,
    multigrid_solve,
    smoothed_multigrid_solve,
)
from aquaflux.vectors import scale

from .preconditioner import damped_jacobi_solve, pressure_schur_laplacian, schur_face_coefficient
from .rhie_chow import momentum_diagonal

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, MeshGeometry
    from aquaflux.solve.multigrid import MultigridHierarchy, SmoothedHierarchy

    from .momentum import MomentumContinuity

_PressureSolve = Callable[[jnp.ndarray], jnp.ndarray]
_VelocitySolve = Callable[[jnp.ndarray], jnp.ndarray]


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
    pressure_pin: int | None = eqx.field(static=True)

    @classmethod
    def of(cls, assembler: MomentumContinuity) -> _SchurGeometry:
        """Extract the Schur-coefficient geometry from a flow assembler."""
        return cls(
            assembler.mesh.face_cells,
            assembler.geometry,
            assembler.boundary,
            assembler.interp_factor,
            assembler.normal_distance,
            assembler.density,
            assembler.pressure_pin,
        )

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
        everywhere for a closed all-wall domain (regularised instead by the pin).
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


# --- pressure-Schur inner solvers (strategy family) ------------------------------------


class InnerSchurSolver(eqx.Module):
    """Strategy: solve the compact pressure Schur ``Ŝ x = rp`` for the preconditioner.

    Built once off the jit path; :meth:`apply` returns the solve ``rp -> Ŝ⁻¹ rp`` specialized to the
    current (frozen) momentum diagonal ``a_P``.
    """

    @abc.abstractmethod
    def apply(self, a_p: jnp.ndarray) -> _PressureSolve:
        """Return the pressure solve ``rp -> Ŝ⁻¹ rp`` at momentum diagonal ``a_P``."""


class DampedJacobiSchur(InnerSchurSolver):
    """Fixed damped-Jacobi sweeps on the assembled pressure Laplacian (simplest; not h-independent)."""

    geometry: _SchurGeometry
    sweeps: int = eqx.field(static=True)
    omega: float = eqx.field(static=True)

    def apply(self, a_p: jnp.ndarray) -> _PressureSolve:
        g = self.geometry
        matvec, diagonal = pressure_schur_laplacian(
            g.face_cells,
            g.mesh_geometry,
            g.interp_factor,
            g.normal_distance,
            a_p,
            g.rho,
            g.pressure_pin,
            boundary_diagonal=g.boundary_diagonal(a_p),
        )
        return lambda rp: damped_jacobi_solve(
            matvec, diagonal, rp, self.sweeps, self.omega, g.pressure_pin
        )


class AggregationSchur(InnerSchurSolver):
    """Unsmoothed-aggregation multigrid V-cycle; coefficients track the iterate, coarse space is weak."""

    geometry: _SchurGeometry
    hierarchy: MultigridHierarchy
    interior_faces: jnp.ndarray
    v_cycles: int = eqx.field(static=True)
    omega: float = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        geometry: _SchurGeometry,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior_faces: jnp.ndarray,
        n_cells: int,
        v_cycles: int,
        omega: float,
    ) -> AggregationSchur:
        hierarchy = build_hierarchy(owner_e, nb_e, n_cells, pin=geometry.pressure_pin)
        return cls(geometry, hierarchy, interior_faces, v_cycles, omega)

    def apply(self, a_p: jnp.ndarray) -> _PressureSolve:
        coeff = self.geometry.coefficient(a_p)[self.interior_faces]
        coeffs, diagonals = level_coefficients(self.hierarchy, coeff)
        # Propagate the boundary (outlet) diagonal up the aggregation — piecewise-constant Galerkin
        # makes a coarse cell's stiffness the sum over its fine members — and fold it into each
        # level's operator, so A = Laplacian + diag(boundary) is non-singular at every level. All-zero
        # (and a no-op) for a closed all-wall domain, which the pin regularises instead.
        levels = self.hierarchy.levels
        extra = self.geometry.boundary_diagonal(a_p)
        extras = []
        for index, level in enumerate(levels):
            extras.append(extra)
            if level.agg is not None:
                extra = segment_sum(extra, level.agg, levels[index + 1].n)
        diagonals = tuple(d + e for d, e in zip(diagonals, extras, strict=True))
        diag_extras = tuple(extras)
        return lambda rp: multigrid_solve(
            self.hierarchy,
            coeffs,
            diagonals,
            rp,
            cycles=self.v_cycles,
            omega=self.omega,
            diag_extras=diag_extras,
        )


class SmoothedAmgSchur(InnerSchurSolver):
    """Smoothed-aggregation multigrid, mesh-independent (V-cycle contraction ~0.25).

    The hierarchy is frozen at a reference coefficient; the current operator's scale is tracked by a
    symmetric diagonal rescaling ``Ŝ_cur⁻¹ ≈ D⁻¹ Ŝ_ref⁻¹ D⁻¹``, ``D = sqrt(diag_cur/diag_ref)`` — exact
    for a uniform rescale, and capturing per-cell scale (including convection) otherwise.
    """

    geometry: _SchurGeometry
    hierarchy: SmoothedHierarchy
    owner: jnp.ndarray
    nb: jnp.ndarray
    interior_faces: jnp.ndarray
    n_cells: int = eqx.field(static=True)
    v_cycles: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        geometry: _SchurGeometry,
        assembler: MomentumContinuity,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior: np.ndarray,
        interior_faces: jnp.ndarray,
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
        hierarchy = build_smoothed_hierarchy(
            owner_e,
            nb_e,
            reference_coeff,
            n_cells,
            pin=geometry.pressure_pin,
            boundary_diagonal=reference_boundary,
        )
        return cls(
            geometry,
            hierarchy,
            jnp.asarray(owner_e),
            jnp.asarray(nb_e),
            interior_faces,
            n_cells,
            v_cycles,
        )

    def apply(self, a_p: jnp.ndarray) -> _PressureSolve:
        current_coeff = self.geometry.coefficient(a_p)[self.interior_faces]
        diag_cur = segment_sum(current_coeff, self.owner, self.n_cells) + segment_sum(
            current_coeff, self.nb, self.n_cells
        )
        # The reference hierarchy carries the boundary (outlet) stiffness in its diagonal, so the
        # current diagonal must include it too for the symmetric rescaling to be consistent.
        diag_cur = diag_cur + self.geometry.boundary_diagonal(a_p)
        if self.geometry.pressure_pin is not None:
            diag_cur = diag_cur.at[self.geometry.pressure_pin].set(1.0)
        inv_scale = jnp.sqrt(self.hierarchy.levels[0].diagonal / diag_cur)
        return lambda rp: (
            inv_scale
            * smoothed_multigrid_solve(self.hierarchy, inv_scale * rp, cycles=self.v_cycles)
        )


# --- velocity-block solvers (strategy family) ------------------------------------------


class VelocityBlockSolver(eqx.Module):
    """Strategy: approximately invert the momentum (velocity) block for the preconditioner."""

    @abc.abstractmethod
    def apply(self, a_p: jnp.ndarray) -> _VelocitySolve:
        """Return the velocity solve ``ru -> δu`` at momentum diagonal ``a_P``."""


class DiagonalVelocity(VelocityBlockSolver):
    """SIMPLE's momentum-diagonal solve ``δu = diag(a_P)⁻¹ ru`` (Jacobi-quality for the viscous block)."""

    def apply(self, a_p: jnp.ndarray) -> _VelocitySolve:
        inv_a_p = 1.0 / a_p
        return lambda ru: scale(ru, inv_a_p)


class SmoothedAmgVelocity(VelocityBlockSolver):
    """Smoothed-aggregation AMG on the viscous momentum operator (mesh-independent), per component.

    The viscous momentum operator is a Dirichlet (no-slip) Laplacian — SPD-nonsingular (boundary faces
    add stiffness to the diagonal, no pin) — so a single AMG hierarchy on a unit-viscosity reference,
    rescaled to the current ``a_P``, replaces the Jacobi-quality diagonal solve.
    """

    hierarchy: SmoothedHierarchy
    dim: int = eqx.field(static=True)
    v_cycles: int = eqx.field(static=True)

    @classmethod
    def build(
        cls,
        assembler: MomentumContinuity,
        owner_e: np.ndarray,
        nb_e: np.ndarray,
        interior: np.ndarray,
        n_cells: int,
        v_cycles: int,
    ) -> SmoothedAmgVelocity:
        area = np.asarray(assembler.geometry.face.area)
        over_distance = area / np.asarray(assembler.normal_distance)
        boundary_owner = np.asarray(assembler.mesh.face_cells.owner)[~interior]
        boundary_diagonal = np.zeros(n_cells)
        np.add.at(boundary_diagonal, boundary_owner, over_distance[~interior])
        hierarchy = build_smoothed_hierarchy(
            owner_e, nb_e, over_distance[interior], n_cells, boundary_diagonal=boundary_diagonal
        )
        return cls(hierarchy, assembler.mesh.dim, v_cycles)

    def apply(self, a_p: jnp.ndarray) -> _VelocitySolve:
        inv_scale = jnp.sqrt(self.hierarchy.levels[0].diagonal / a_p)

        def solve(ru: jnp.ndarray) -> jnp.ndarray:
            columns = [
                inv_scale
                * smoothed_multigrid_solve(
                    self.hierarchy, inv_scale * ru[:, i], cycles=self.v_cycles
                )
                for i in range(self.dim)
            ]
            return jnp.stack(columns, axis=1)

        return solve


# --- the composed preconditioner -------------------------------------------------------


class BlockPreconditioner(eqx.Module):
    """A block SIMPLE preconditioner composing a velocity solve and a pressure-Schur inner solve.

    Built from a flow assembler by :meth:`build`; :meth:`factory` returns the ``state -> M`` callable
    :func:`~aquaflux.solve.newton.newton_step` expects. With ``block_triangular`` set, the pressure
    block additionally sees the divergence of the velocity predictor (``δp = Ŝ⁻¹(r_p − D·δu)``), giving
    the Murphy--Golub--Wathen 2-eigenvalue structure; ``D·δu`` is a ``jvp`` through the frozen residual,
    so ``D`` is a constant operator (adjoint-transparent).

    The pressure Schur is scaled either by the momentum diagonal ``a_P`` (SIMPLE, ``Ŝ ~ B diag(V/a_P)
    B^T``) or, when ``schur_diagonal`` is set, by a frozen velocity-independent diagonal ``Q̂ = ρ V / k``
    (MSIMPLER, a mass-matrix scaling), giving the constant-coefficient pressure Poisson ``Ŝ ~ B
    diag(k/ρ) B^T``. Because ``Q̂`` does not track the velocity, the MSIMPLER Schur — unlike ``V/a_P``
    — does not degrade as convection strengthens (Klaij & Vuik 2013, for exactly this collocated-FV
    coupled discretization), which carries the coupled solve past the Reynolds number where the
    ``a_P``-Schur stalls. The scale ``k`` (see :meth:`build`) matches its magnitude to the SIMPLE Schur
    at the operating convection. Only the Schur uses ``Q̂``; the velocity block always uses the true
    ``a_P``.
    """

    assembler: MomentumContinuity
    schur: InnerSchurSolver
    velocity: VelocityBlockSolver
    block_triangular: bool = eqx.field(static=True)
    schur_diagonal: jnp.ndarray | None = None

    @classmethod
    def build(
        cls,
        assembler: MomentumContinuity,
        *,
        inner: str = "smoothed",
        schur_scaling: str = "simple",
        msimpler_scale: float | None = None,
        v_cycles: int = 1,
        jacobi_sweeps: int = 30,
        omega: float = 0.7,
    ) -> BlockPreconditioner:
        """Build the preconditioner for ``assembler`` with the selected inner Schur solver.

        Parameters
        ----------
        assembler : MomentumContinuity
            The coupled flow residual assembler.
        inner : {"smoothed", "multigrid", "jacobi"}
            The inner pressure-Schur solver strategy.
        schur_scaling : {"simple", "msimpler"}
            How the pressure Schur is scaled. ``"simple"`` uses the momentum diagonal ``a_P`` (the
            classical SIMPLE Schur ``V / a_P``, which degrades as convection strengthens);
            ``"msimpler"`` uses a **frozen, velocity-independent** diagonal ``Q̂ = ρ V / k`` so the
            Schur is a constant-coefficient pressure Poisson (coefficient ``k · A/(d·n)``) that stays
            Re-robust — the fix that carries the coupled solve past the ``a_P``-Schur stall.
        msimpler_scale : float, optional
            The MSIMPLER scale ``k`` (only for ``schur_scaling="msimpler"``). It must set the Schur
            magnitude to the operating convection, or the block preconditioner is unbalanced and
            stalls. ``None`` (default) calibrates it automatically to ``mean(V / a_P)`` at a unit
            streamwise reference speed — matching the SIMPLE Schur magnitude while staying
            velocity-independent. Pass an explicit value for a problem whose characteristic speed is
            far from one.
        v_cycles : int
            Multigrid V-cycles per apply (``"smoothed"`` / ``"multigrid"``).
        jacobi_sweeps : int
            Damped-Jacobi sweeps (``"jacobi"``).
        omega : float
            Jacobi damping factor in ``(0, 1]`` (``"jacobi"`` / ``"multigrid"``).
        """
        if inner not in ("smoothed", "multigrid", "jacobi"):
            raise ValueError(
                f"unknown inner solver {inner!r}; use 'smoothed', 'multigrid' or 'jacobi'"
            )
        if schur_scaling not in ("simple", "msimpler"):
            raise ValueError(f"unknown schur_scaling {schur_scaling!r}; use 'simple' or 'msimpler'")
        geometry = _SchurGeometry.of(assembler)
        n_cells = assembler.mesh.n_cells
        owner_e, nb_e, interior_faces_np = assembler.mesh.face_cells.interior_edges()
        interior = np.asarray(assembler.mesh.face_cells.interior)
        interior_faces = jnp.asarray(interior_faces_np)

        # MSIMPLER's frozen Schur diagonal Q̂ = ρ V / k (per cell), which replaces a_P; ``None`` keeps
        # the classical SIMPLE (a_P) Schur. The scale ``k`` sets the constant Schur coefficient to
        # ``k · A/(d·n)``; auto-calibrated to mean(V/a_P) at unit streamwise speed so it matches the
        # SIMPLE Schur magnitude at the operating convection (a mismatched scale unbalances the block
        # preconditioner and stalls GMRES) while staying velocity-independent.
        schur_diagonal = None
        if schur_scaling == "msimpler":
            scale = msimpler_scale
            if scale is None:
                reference_velocity = jnp.zeros((n_cells, assembler.mesh.dim)).at[:, 0].set(1.0)
                reference_a_p = jnp.mean(
                    assembler.momentum_matrix_diagonal(reference_velocity), axis=1
                )
                scale = jnp.mean(assembler.geometry.cell.volume / reference_a_p)
            schur_diagonal = jax.lax.stop_gradient(
                assembler.density * assembler.geometry.cell.volume / scale
            )

        if inner == "smoothed":
            schur: InnerSchurSolver = SmoothedAmgSchur.build(
                geometry,
                assembler,
                owner_e,
                nb_e,
                interior,
                interior_faces,
                n_cells,
                v_cycles,
                reference_diagonal=schur_diagonal,
            )
            velocity: VelocityBlockSolver = SmoothedAmgVelocity.build(
                assembler, owner_e, nb_e, interior, n_cells, v_cycles
            )
            return cls(
                assembler, schur, velocity, block_triangular=True, schur_diagonal=schur_diagonal
            )
        if inner == "multigrid":
            schur = AggregationSchur.build(
                geometry, owner_e, nb_e, interior_faces, n_cells, v_cycles, omega
            )
        else:
            schur = DampedJacobiSchur(geometry, jacobi_sweeps, omega)
        return cls(
            assembler,
            schur,
            DiagonalVelocity(),
            block_triangular=False,
            schur_diagonal=schur_diagonal,
        )

    def _divergence(self, state: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The velocity predictor's divergence ``δu -> D·δu`` as a jvp through the frozen residual."""
        frozen = jax.lax.stop_gradient(self.assembler)
        frozen_state = jax.lax.stop_gradient(state)

        def divergence(velocity_correction: jnp.ndarray) -> jnp.ndarray:
            tangent = self.assembler.pack(
                velocity_correction, jnp.zeros(self.assembler.mesh.n_cells)
            )
            _, pressure = self.assembler.unpack(
                jax.jvp(frozen.residual, (frozen_state,), (tangent,))[1]
            )
            return pressure

        return divergence

    def frozen_momentum_diagonal(self, state: jnp.ndarray) -> jnp.ndarray:
        """The isotropic, frozen momentum diagonal ``a_P`` at ``state``, shape ``(n_cells,)``.

        Isotropic (component-averaged) ``a_P`` for the Schur/velocity blocks; the directional
        per-component form enters only the operator's Rhie--Chow coefficient. The preconditioner
        needs ``a_P`` frozen, so ``stop_gradient`` it here (the residual uses the differentiable
        ``a_P``): the state is already detached, and this keeps ``M`` a constant operator even if
        called on a live state. Exposed so a continuation driver (implicit under-relaxation) can
        form its diagonal shift from the *same* ``a_P`` the preconditioner inverts.
        """
        velocity, _ = self.assembler.unpack(jax.lax.stop_gradient(state))
        return jnp.mean(
            jax.lax.stop_gradient(self.assembler.momentum_matrix_diagonal(velocity)), axis=1
        )

    def apply_at(
        self, state: jnp.ndarray, a_p: jnp.ndarray
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The preconditioner matvec ``M`` at ``state`` for a supplied (frozen) diagonal ``a_P``.

        Splits the ``state -> M`` factory so a caller can pass an *effective* ``a_P`` — e.g. the
        under-relaxed ``a_P (1 + β)`` an implicit-continuation step uses, matching the shifted
        Jacobian it inverts — instead of always the bare diagonal :meth:`frozen_momentum_diagonal`
        returns. ``a_P`` is the isotropic per-cell diagonal, shape ``(n_cells,)``.

        The velocity block always inverts at the supplied ``a_P``; the Schur uses the frozen
        MSIMPLER diagonal ``Q̂`` instead when set (it is velocity-independent, so it ignores both
        ``a_P`` and any continuation shift), else the same ``a_P`` (classical SIMPLE).
        """
        schur_a_p = a_p if self.schur_diagonal is None else self.schur_diagonal
        schur_solve = self.schur.apply(schur_a_p)
        velocity_solve = self.velocity.apply(a_p)
        divergence = self._divergence(state) if self.block_triangular else None

        def apply(v: jnp.ndarray) -> jnp.ndarray:
            velocity_residual, pressure_residual = self.assembler.unpack(v)
            velocity_correction = velocity_solve(velocity_residual)
            if divergence is not None:  # block-triangular: correct the pressure RHS by D·δu
                pressure_residual = pressure_residual - divergence(velocity_correction)
            return self.assembler.pack(velocity_correction, schur_solve(pressure_residual))

        return apply

    def factory(self) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """Return the ``state -> M`` factory the Newton step applies (``M`` frozen at that iterate)."""
        return lambda state: self.apply_at(state, self.frozen_momentum_diagonal(state))
