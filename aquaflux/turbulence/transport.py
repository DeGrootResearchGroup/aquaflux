"""Assembly of the k-omega SST transport equations on a configured mesh.

The k and omega equations are scalar transport equations that reuse the flux operators of any other
scalar: advection on the flow's mass flux and diffusion of an effective viscosity. What is
turbulence-specific is the coefficients and sources -- the eddy-viscosity-blended diffusivity and
the SST production/destruction/cross-diffusion terms -- and the omega wall treatment, where the
near-wall cells are fixed to the analytical value rather than balanced.

:class:`SSTTurbulence` holds the static configuration (the model, mesh, schemes, molecular
viscosity, wall geometry, and the k / omega boundary closures) and builds the residual of each
equation from the *frozen* closure fields of the current outer sweep (the eddy viscosity, strain
rate, blending function, and gradients) gathered in :class:`SSTClosureFields`. Computing those
fields from the flow and turbulence state, and iterating the sweeps, is the driver's job.

**Constant density.** The equations are written in kinematic form, so advection uses the volume flux
``mdot / rho`` (``mdot`` the Rhie--Chow mass flux, reused so the scalar stays discretely
conservative with continuity) and the diffusivity is kinematic ``nu + sigma nu_t``. This is exact
for constant density; the variable-density (conservative) form is deferred, as it is for the flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax.numpy as jnp

from aquaflux.discretization import AdvectionFlux, DiffusionFlux, FixedValueCells, ResidualAssembler
from aquaflux.mesh import distance_to_patches
from aquaflux.properties import FieldProperty, PropertyModel
from aquaflux.vectors import norm_squared

from .boundary import (
    log_layer_shear_rate,
    nut_wall,
    omega_wall,
    wall_function_weight,
    wall_k_diffusivity,
)
from .continuation import ScalarShiftPolicy
from .preconditioner import (
    ScalarTransportPreconditioner,
    scalar_transport_preconditioner,
    scalar_transport_shift_diagonal,
)
from .sources import (
    KDestruction,
    KProduction,
    NearWallKClosure,
    OmegaCrossDiffusion,
    OmegaDestruction,
    OmegaProduction,
)
from .strain import safe_sqrt, strain_rate_magnitude

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aquaflux.boundary import BoundaryConditions
    from aquaflux.discretization import AdvectionScheme
    from aquaflux.flow import VelocityFields
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.schemes import GradientScheme

    from .sst import SSTModel


class SSTClosureFields(NamedTuple):
    """The frozen SST fields of one outer sweep the transport residuals are built from.

    All are held fixed while the k and omega equations are solved; the driver recomputes them from
    the flow and turbulence state between sweeps.

    Attributes
    ----------
    nu_t : jnp.ndarray
        Eddy viscosity per cell, shape ``(n_cells,)``.
    strain_rate : jnp.ndarray
        Strain-rate magnitude ``S`` per cell, shape ``(n_cells,)``.
    f1 : jnp.ndarray
        The ``F1`` blending function per cell, shape ``(n_cells,)``.
    grad_k, grad_omega : jnp.ndarray
        Cell gradients of ``k`` and ``omega``, shape ``(n_cells, dim)``.
    omega : jnp.ndarray
        The frozen ``omega`` field, shape ``(n_cells,)`` (the k destruction/production read it, and
        the omega cross-diffusion lags it).
    k : jnp.ndarray
        The frozen ``k`` field, shape ``(n_cells,)`` (the ω-production limiter's ``10 β* k ω`` cap
        reads it).
    wall_shear_rate : jnp.ndarray
        Wall-face normal velocity gradient magnitude at the wall-adjacent cells, shape ``(n_wall,)``
        (see :meth:`SSTTurbulence.wall_shear_rate`) -- what the adaptive near-wall k production
        measures the wall stress from.
    """

    nu_t: jnp.ndarray
    strain_rate: jnp.ndarray
    f1: jnp.ndarray
    grad_k: jnp.ndarray
    grad_omega: jnp.ndarray
    omega: jnp.ndarray
    k: jnp.ndarray
    wall_shear_rate: jnp.ndarray


class WallFixedResidual(eqx.Module):
    """A transport residual with a set of cells' rows replaced by a value fixation.

    The omega equation's near-wall cells carry the analytical sublayer value rather than a transport
    balance, so its residual is the assembled balance composed with a
    :class:`~aquaflux.discretization.FixedValueCells` overwrite.

    This is an ``equinox.Module`` rather than a closure so that it can be passed *into* a jitted solve
    without forcing a re-trace. ``equinox.filter_jit`` partitions a plain function onto the static
    side, where it is hashed by object identity — a freshly built closure each outer sweep therefore
    misses the compilation cache and re-compiles the whole solve. As a Module its arrays ride on the
    traced side, so a sweep changes only their *values* and the compiled solve is reused.

    Attributes
    ----------
    assembler : ResidualAssembler
        Assembles the transport balance ``phi -> R(phi)``.
    wall_fix : FixedValueCells
        The rows to replace, and the values to fix them to.
    """

    assembler: ResidualAssembler
    wall_fix: FixedValueCells

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        """The residual at ``phi``, shape ``(n_cells,)``."""
        return self.wall_fix.apply(self.assembler.residual(phi), phi)


class SSTTurbulence(eqx.Module):
    """Assembles the k and omega SST transport residuals for a configured problem.

    Construct with :meth:`build`. :meth:`k_residual` and :meth:`omega_residual` return the residual
    function of each equation, ready for a Newton solve, given the frozen closure fields of the
    current sweep and the flow's mass flux.

    Attributes
    ----------
    model : SSTModel
        The SST constants and blends.
    mesh, geometry : Mesh, MeshGeometry
        Topology and metrics.
    gradient_scheme : GradientScheme
        Reconstruction for the non-orthogonal diffusion correction.
    advection_scheme : AdvectionScheme
        The k / omega convection scheme (e.g. first-order upwind).
    density : float
        The (constant) fluid density, used to form the volume flux ``mdot / rho``.
    molecular_viscosity : jnp.ndarray
        Kinematic molecular viscosity ``nu`` per cell, shape ``(n_cells,)``.
    wall_distance : jnp.ndarray
        Distance to the nearest wall per cell, shape ``(n_cells,)``.
    wall_cells : jnp.ndarray
        Indices of the wall-adjacent cells whose ``omega`` is fixed, shape ``(n_wall,)``.
    wall_faces : jnp.ndarray
        Indices of the wall boundary faces, shape ``(n_wall_faces,)`` — the faces the momentum
        wall-function eddy viscosity is scattered onto (see :meth:`wall_face_eddy_viscosity`).
    k_boundary, omega_boundary : BoundaryConditions
        The scalar boundary closures for each field (Dirichlet inlet / wall, zero-gradient outlet;
        the omega wall is imposed by cell fixation, so its wall closure is a placeholder).
    explicit_production_limiter : bool
        How the k-production limiter is linearized for the forward k-solve (static). ``True``
        (default) freezes the cap's ``k`` (:attr:`KProduction.explicit_limiter`), giving an M-matrix
        the k-solve converges on unpreconditioned -- a robust modified-Newton step. ``False`` keeps
        the exact Jacobian, whose active cap is indefinite: it needs the scalar preconditioner (which
        rescues it) but then converges quadratically. The converged field is the same either way (the
        residual value is identical); only the forward path differs, so the coupled adjoint (built on
        the exact residual) is unaffected.
    """

    model: SSTModel
    mesh: Mesh
    geometry: MeshGeometry
    gradient_scheme: GradientScheme
    advection_scheme: AdvectionScheme
    density: float
    molecular_viscosity: jnp.ndarray
    wall_distance: jnp.ndarray
    wall_cells: jnp.ndarray
    wall_faces: jnp.ndarray
    k_boundary: BoundaryConditions
    omega_boundary: BoundaryConditions
    explicit_production_limiter: bool = eqx.field(static=True, default=True)

    @classmethod
    def build(
        cls,
        model: SSTModel,
        mesh: Mesh,
        geometry: MeshGeometry,
        gradient_scheme: GradientScheme,
        advection_scheme: AdvectionScheme,
        density: float,
        molecular_viscosity: jnp.ndarray,
        wall_patches: Sequence[str],
        k_boundary: BoundaryConditions,
        omega_boundary: BoundaryConditions,
        *,
        explicit_production_limiter: bool = True,
    ) -> SSTTurbulence:
        """Build the assembler, deriving the wall distance and wall-adjacent cell set.

        Parameters
        ----------
        wall_patches : sequence of str
            The boundary patches treated as walls; their wall distance is computed and their
            owner cells become the ``omega`` fixation set.
        explicit_production_limiter : bool
            Linearization of the k-production limiter for the forward solve (see the class
            attribute); ``True`` (default) is the robust unpreconditioned-solvable choice.

        The remaining arguments are stored directly (see the class attributes).
        """
        wall_distance = distance_to_patches(mesh, geometry, wall_patches)
        wall_faces = jnp.concatenate([mesh.face_patches.indices(p) for p in wall_patches])
        wall_cells = jnp.unique(mesh.face_cells.owner[wall_faces])
        return cls(
            model=model,
            mesh=mesh,
            geometry=geometry,
            gradient_scheme=gradient_scheme,
            advection_scheme=advection_scheme,
            density=density,
            molecular_viscosity=molecular_viscosity,
            wall_distance=wall_distance,
            wall_cells=wall_cells,
            wall_faces=wall_faces,
            k_boundary=k_boundary,
            omega_boundary=omega_boundary,
            explicit_production_limiter=explicit_production_limiter,
        )

    def resolve_boundaries(self) -> SSTTurbulence:
        """Return a copy whose k and omega boundaries are bound to the mesh's face patches.

        The k/omega scalar residuals and the closure-gradient reconstruction rebuild a
        :class:`~aquaflux.discretization.ResidualAssembler` each call, and that build resolves the
        boundary patch names to face indices -- a data-dependent ``nonzero`` lookup that cannot run
        under ``jit``. Binding the boundaries **once**, ahead of any jitted use (the coupled residual,
        the jitted segregated sweep prologue), makes each rebuild's ``resolve`` an idempotent no-op.
        Idempotent itself: an already-bound assembler is returned with its boundaries unchanged.
        """
        face_patches = self.mesh.face_patches
        return eqx.tree_at(
            lambda t: (t.k_boundary, t.omega_boundary),
            self,
            (self.k_boundary.resolve(face_patches), self.omega_boundary.resolve(face_patches)),
        )

    def _volume_flux(self, mdot: jnp.ndarray) -> jnp.ndarray:
        """The volume face flux ``mdot / rho`` the kinematic transport advects on."""
        return mdot / self.density

    def _diffusivity(
        self, nu_t: jnp.ndarray, f1: jnp.ndarray, inner: float, outer: float
    ) -> FieldProperty:
        """Effective kinematic diffusivity ``nu + blend(F1, inner, outer) nu_t`` as a property."""
        sigma = self.model.blend(f1, inner, outer)
        return FieldProperty(values=self.molecular_viscosity + sigma * nu_t)

    def strain_rate(self, velocity_gradient: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
        """Mean strain-rate magnitude ``S`` the closure reads, shape ``(n_cells,)``.

        The reconstructed :func:`~aquaflux.turbulence.strain_rate_magnitude` everywhere, except that
        the wall-adjacent cells are blended onto the analytical
        :func:`~aquaflux.turbulence.log_layer_shear_rate` as they leave the viscous sublayer (the
        smooth :func:`~aquaflux.turbulence.wall_function_weight` crossover, so a wall-resolved mesh
        keeps the reconstructed value).

        **Why the substitution matters, and why it belongs here rather than in a source term.** The
        quantity most sensitive to ``S`` is not a production term but the SST shear-stress limiter
        ``nu_t = a1 k / max(a1 omega, F2 S)``. In an equilibrium log layer the two arguments are
        deliberately near-equal (``a1 omega`` edges out ``S`` by a few percent), so the limiter is
        *just* inactive and ``nu_t`` is the mixing-length value ``kappa u_tau y``. A wall-function
        mesh's reconstructed ``S`` overshoots the log-layer shear several-fold, which throws the
        limiter hard the other way and clamps the wall cell's ``nu_t`` to a fraction of its correct
        value -- measured on a ``y+ ~ 30`` channel: ``nu_t`` ~5x low, the velocity profile far too
        steep out of the wall cell (``U+`` jumping 6.6 over the first cell spacing where the log law
        gives 2.7), and the predicted wall shear ~12% low even with the wall stress itself correct.

        Parameters
        ----------
        velocity_gradient : jnp.ndarray
            The velocity-gradient tensor, shape ``(n_cells, dim, dim)``.
        k : jnp.ndarray
            The turbulent kinetic energy per cell, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            The strain-rate magnitude per cell, shape ``(n_cells,)``.
        """
        strain = strain_rate_magnitude(velocity_gradient)
        wall = self.wall_cells
        nu_wall = self.molecular_viscosity[wall]
        d_wall = self.wall_distance[wall]
        k_wall = k[wall]
        weight = wall_function_weight(nu_wall, d_wall, k_wall, self.model)
        return strain.at[wall].set(
            (1.0 - weight) * strain[wall]
            + weight * log_layer_shear_rate(d_wall, k_wall, self.model)
        )

    def eddy_viscosity(
        self, velocity_gradient: jnp.ndarray, k: jnp.ndarray, omega: jnp.ndarray
    ) -> jnp.ndarray:
        """Kinematic eddy viscosity ``nu_t`` for the current state, shape ``(n_cells,)``.

        The quantity the flow solve needs (as ``mu_t = rho nu_t``) to close the momentum viscosity.

        Parameters
        ----------
        velocity_gradient : jnp.ndarray
            The velocity-gradient tensor, shape ``(n_cells, dim, dim)`` (from the flow solve).
        k, omega : jnp.ndarray
            The current turbulence fields, shape ``(n_cells,)``.
        """
        return self.model.eddy_viscosity(
            k,
            omega,
            self.strain_rate(velocity_gradient, k),
            self.molecular_viscosity,
            self.wall_distance,
        )

    def wall_face_eddy_viscosity(self, k: jnp.ndarray) -> jnp.ndarray:
        """Per-face wall-function eddy viscosity ``nu_t,wall``, shape ``(n_faces,)`` (zero off walls).

        The adaptive (``y+``-insensitive) wall-face value :func:`~aquaflux.turbulence.nut_wall`
        scattered onto the wall boundary faces, computed from ``k`` at each wall face's owner cell and
        its wall distance. Handed to the momentum block through
        :meth:`~aquaflux.flow.MomentumContinuity.with_eddy_viscosity` so the wall shear follows the law
        of the wall; it is zero (a resolved wall) where the first cell is in the viscous sublayer, so on
        a wall-resolved mesh the momentum block is unchanged.

        Parameters
        ----------
        k : jnp.ndarray
            The turbulent kinetic energy per cell, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            The wall-face eddy viscosity per face, shape ``(n_faces,)``.
        """
        owner = self.mesh.face_cells.owner[self.wall_faces]
        values = nut_wall(
            self.molecular_viscosity[owner], self.wall_distance[owner], k[owner], self.model
        )
        return jnp.zeros(self.mesh.n_faces).at[self.wall_faces].set(values)

    def _field_gradient(self, field: jnp.ndarray, boundary: BoundaryConditions) -> jnp.ndarray:
        """Reconstruct the cell gradient of a turbulence field with its boundary closures.

        Reuses the residual assembler's leading-order gradient reconstruction (the injected gradient
        scheme evaluated with the field's boundary values), so the boundary handling is not
        re-implemented here.
        """
        assembler = ResidualAssembler.build(
            self.mesh,
            self.geometry,
            PropertyModel({}),
            (),
            boundary,
            gradient_scheme=self.gradient_scheme,
        )
        return assembler.gradient(field)

    def wall_shear_rate(self, velocity: VelocityFields) -> jnp.ndarray:
        """Wall-face normal velocity gradient at the wall-adjacent cells, shape ``(n_wall,)``.

        ``|U_P - U_wall| / d`` per wall face -- the discrete wall-normal derivative of the velocity
        the momentum block's wall flux is built on -- area-averaged over the wall faces of each
        wall-adjacent cell (a corner cell touches more than one). Multiplied by the wall-face
        effective viscosity it is the wall shear stress
        (:func:`~aquaflux.turbulence.wall_shear_stress`), which is why the adaptive near-wall k
        production reads *this* shear rather than the cell strain-rate magnitude: it makes the energy
        fed into the wall cell the work the wall stress actually does.

        The velocity difference is taken against the patch's own boundary velocity, so a moving wall
        contributes only the *relative* shear. Its magnitude uses the guarded square root: the
        Euclidean norm has a cone point at zero, and a stationary fluid (a cold start, a stagnation
        point) sits exactly on it, where a plain ``sqrt`` would return a NaN derivative into every
        Jacobian row that reads a wall closure.

        Parameters
        ----------
        velocity : VelocityFields
            The kinematic flow state (cell velocity, boundary-face velocity, gradient).

        Returns
        -------
        jnp.ndarray
            The wall-normal shear rate per wall-adjacent cell, shape ``(n_wall,)``.
        """
        owner = self.mesh.face_cells.owner[self.wall_faces]
        slip = velocity.velocity[owner] - velocity.boundary_velocity[self.wall_faces]
        rate = safe_sqrt(norm_squared(slip)) / self.wall_distance[owner]

        # Area-average onto the owner cells through the connectivity scatter, keeping the division
        # until after the wall cells are gathered (every non-wall cell has zero weight, and a 0/0
        # there would put a NaN into the reverse-mode tape even though its value is discarded).
        zeros = jnp.zeros(self.mesh.n_faces)
        area = zeros.at[self.wall_faces].set(self.geometry.face.area[self.wall_faces])
        weighted = zeros.at[self.wall_faces].set(rate) * area
        scatter = self.mesh.face_cells.scatter
        return scatter(weighted, zeros)[self.wall_cells] / scatter(area, zeros)[self.wall_cells]

    def closure_fields(
        self, velocity: VelocityFields, k: jnp.ndarray, omega: jnp.ndarray
    ) -> SSTClosureFields:
        """Assemble the frozen SST closure fields for the current state.

        Computes the strain rate from the velocity gradient, reconstructs ``grad k`` and
        ``grad omega`` with their boundary closures, evaluates ``F1`` and the eddy viscosity, and
        measures the near-wall shear rate -- the fields the k and omega equation builders freeze for
        a sweep.

        Parameters
        ----------
        velocity : VelocityFields
            The kinematic flow state: cell velocity, boundary-face velocity, and the cell
            velocity-gradient tensor.
        k, omega : jnp.ndarray
            The current turbulence fields, shape ``(n_cells,)``.
        """
        strain = self.strain_rate(velocity.gradient, k)
        grad_k = self._field_gradient(k, self.k_boundary)
        grad_omega = self._field_gradient(omega, self.omega_boundary)
        f1 = self.model.f1(
            k, omega, self.molecular_viscosity, self.wall_distance, grad_k, grad_omega
        )
        nu_t = self.model.eddy_viscosity(
            k, omega, strain, self.molecular_viscosity, self.wall_distance
        )
        return SSTClosureFields(
            nu_t, strain, f1, grad_k, grad_omega, omega, k, self.wall_shear_rate(velocity)
        )

    def k_residual(
        self, mdot: jnp.ndarray, closure: SSTClosureFields
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The k-equation residual function ``k -> R_k`` for the frozen ``closure`` and ``mdot``.

        Advection on the volume flux, diffusion of ``nu + sigma_k nu_t``, and the limited production
        minus destruction sources.

        Both halves of the adaptive near-wall ``k`` treatment enter here, over the same smooth
        crossover and inactive on a wall-resolved mesh: the wall-adjacent cells' production is blended
        toward the log-layer form (:class:`~aquaflux.turbulence.NearWallKClosure`), and the wall-face
        diffusivity is faded out (:func:`~aquaflux.turbulence.wall_k_diffusivity`) so a modelled
        sublayer carries no turbulent-energy flux into the wall.
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_k1, self.model.sigma_k2
        )
        near_wall = self._near_wall_closure(closure.wall_shear_rate)
        assembler = ResidualAssembler.build(
            self.mesh,
            self.geometry,
            PropertyModel({"diffusivity": diffusivity}),
            (
                AdvectionFlux(self._volume_flux(mdot), self.advection_scheme),
                DiffusionFlux(
                    boundary_coefficient=self._wall_k_diffusivity(diffusivity, closure.k)
                ),
            ),
            self.k_boundary,
            source_operators=(
                KProduction(
                    closure.nu_t,
                    closure.strain_rate,
                    closure.omega,
                    self.model,
                    explicit_limiter=self.explicit_production_limiter,
                    near_wall=near_wall,
                ),
                KDestruction(closure.omega, self.model, near_wall=near_wall),
            ),
            gradient_scheme=self.gradient_scheme,
        )
        return assembler.residual

    def _near_wall_closure(self, wall_shear_rate: jnp.ndarray) -> NearWallKClosure:
        """The adaptive near-wall k-budget collaborator for this sweep's wall shear rate."""
        return NearWallKClosure(
            self.wall_cells,
            self.wall_distance[self.wall_cells],
            self.molecular_viscosity[self.wall_cells],
            wall_shear_rate,
            self.model,
        )

    def _wall_k_diffusivity(self, diffusivity: FieldProperty, k: jnp.ndarray) -> jnp.ndarray:
        """Per-face k diffusion coefficient with the wall faces faded out, shape ``(n_faces,)``.

        The owner-cell coefficient everywhere (which is what the diffusion operator would use anyway),
        overridden on the wall faces by :func:`~aquaflux.turbulence.wall_k_diffusivity`. Only the
        boundary entries are read by :class:`~aquaflux.discretization.DiffusionFlux`.
        """
        owner = self.mesh.face_cells.owner
        face_gamma = diffusivity.values[owner]
        wall_owner = owner[self.wall_faces]
        return face_gamma.at[self.wall_faces].set(
            wall_k_diffusivity(
                face_gamma[self.wall_faces],
                self.molecular_viscosity[wall_owner],
                self.wall_distance[wall_owner],
                k[wall_owner],
                self.model,
            )
        )

    def omega_residual(
        self, mdot: jnp.ndarray, closure: SSTClosureFields
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The omega-equation residual function ``omega -> R_omega`` for the frozen ``closure``.

        Advection, diffusion of ``nu + sigma_omega nu_t``, the production/destruction/cross-diffusion
        sources, and the near-wall cells fixed to the analytical ``omega`` (their balance replaced).
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_omega1, self.model.sigma_omega2
        )
        assembler = ResidualAssembler.build(
            self.mesh,
            self.geometry,
            PropertyModel({"diffusivity": diffusivity}),
            (
                AdvectionFlux(self._volume_flux(mdot), self.advection_scheme),
                DiffusionFlux(),
            ),
            self.omega_boundary,
            source_operators=(
                OmegaProduction(
                    closure.strain_rate,
                    closure.nu_t,
                    closure.k,
                    closure.omega,
                    closure.f1,
                    self.model,
                ),
                OmegaDestruction(closure.f1, self.model),
                OmegaCrossDiffusion(
                    closure.omega, closure.grad_k, closure.grad_omega, closure.f1, self.model
                ),
            ),
            gradient_scheme=self.gradient_scheme,
        )
        wall_fix = FixedValueCells(
            self.wall_cells,
            omega_wall(
                self.molecular_viscosity[self.wall_cells],
                self.wall_distance[self.wall_cells],
                closure.k[self.wall_cells],
                self.model,
            ),
        )
        return WallFixedResidual(assembler, wall_fix)

    def k_preconditioner(
        self,
        mdot: jnp.ndarray,
        closure: SSTClosureFields,
        reference: jnp.ndarray,
        *,
        method: str = "twolevel",
        reuse: ScalarTransportPreconditioner | None = None,
    ) -> ScalarTransportPreconditioner:
        """The convection-diffusion AMG preconditioning the k-equation's shifted solve.

        Split from :meth:`k_shift_policy` because the two have different lifetimes: building the
        hierarchy is scipy graph work whose cost grows with mesh size, and it only accelerates the
        Krylov iteration (it never enters the converged field or its adjoint), so a segregated loop
        builds it **once** and reuses it across sweeps while rebuilding the shift diagonal each sweep.
        Freezing stays effective as the sweeps proceed because a larger eddy viscosity makes the
        transport operator *more* diffusion-dominated — the regime a frozen aggregation hierarchy
        handles best.

        Parameters
        ----------
        mdot : jnp.ndarray
            The flow's Rhie--Chow mass flux, shape ``(n_faces,)``.
        closure : SSTClosureFields
            The closure fields the frozen operator is built from. Use a representative sweep (the
            first is a reasonable choice, and is conservative: its lower eddy viscosity makes the
            frozen operator the *harder* of the two).
        reference : jnp.ndarray
            The field the frozen operator linearizes at, shape ``(n_cells,)``.
        method : {"twolevel", "air"}
            The convection hierarchy: stable two-level aggregation, or the reduction-based (lAIR)
            hierarchy that coarsens fully and stays mesh-independent at large sizes.
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_k1, self.model.sigma_k2
        )
        return scalar_transport_preconditioner(
            self.mesh,
            self.geometry,
            diffusivity.values,
            self._volume_flux(mdot),
            self.k_residual(mdot, closure),
            reference,
            method=method,
            reuse=reuse,
        )

    def k_shift_policy(
        self,
        mdot: jnp.ndarray,
        closure: SSTClosureFields,
        reference: jnp.ndarray,
        *,
        preconditioner: ScalarTransportPreconditioner | None = None,
    ) -> ScalarShiftPolicy:
        """The pseudo-transient continuation policy for the k-equation solve.

        Bundles the transport-operator shift diagonal (the ``a_P`` analogue that damps the reactive
        k-solve from a cold start) with the preconditioner for the shifted operator -- the two
        problem-specific inputs
        :class:`~aquaflux.turbulence.continuation.ScalarShiftPolicy` supplies to the continuation
        engine. The shift diagonal is built for the sweep's ``closure`` and ``mdot`` (the same fields
        ``k_residual`` uses), so it tracks the current operator; the preconditioner is passed in
        because it is built once and reused (see :meth:`k_preconditioner`).

        Parameters
        ----------
        mdot : jnp.ndarray
            The flow's Rhie--Chow mass flux, shape ``(n_faces,)``.
        closure : SSTClosureFields
            The frozen closure fields of the current sweep.
        reference : jnp.ndarray
            The field the shift diagonal linearizes at (the current ``k``), shape ``(n_cells,)``.
        preconditioner : ScalarTransportPreconditioner, optional
            The preconditioner for the shifted solve (from :meth:`k_preconditioner`), or ``None`` for
            a shift-only (unpreconditioned) continuation solve.
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_k1, self.model.sigma_k2
        )
        shift_diagonal = scalar_transport_shift_diagonal(
            self.mesh,
            self.geometry,
            diffusivity.values,
            self._volume_flux(mdot),
            self.k_residual(mdot, closure),
            reference,
        )
        return ScalarShiftPolicy(shift_diagonal, preconditioner)

    def omega_preconditioner(
        self,
        mdot: jnp.ndarray,
        closure: SSTClosureFields,
        reference: jnp.ndarray,
        *,
        method: str = "twolevel",
        reuse: ScalarTransportPreconditioner | None = None,
    ) -> ScalarTransportPreconditioner:
        """The convection-diffusion AMG preconditioning the omega-equation's shifted solve.

        As :meth:`k_preconditioner`, with the omega diffusivity and the near-wall fixed cells
        detached from the coarsening (their rows are the value fixation, not a transport balance).
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_omega1, self.model.sigma_omega2
        )
        return scalar_transport_preconditioner(
            self.mesh,
            self.geometry,
            diffusivity.values,
            self._volume_flux(mdot),
            self.omega_residual(mdot, closure),
            reference,
            method=method,
            fixed_cells=self.wall_cells,
            reuse=reuse,
        )

    def omega_shift_policy(
        self,
        mdot: jnp.ndarray,
        closure: SSTClosureFields,
        reference: jnp.ndarray,
        *,
        preconditioner: ScalarTransportPreconditioner | None = None,
    ) -> ScalarShiftPolicy:
        """The pseudo-transient continuation policy for the omega-equation solve.

        As :meth:`k_shift_policy`, with the omega diffusivity and the near-wall fixed cells: their
        shift is zeroed, since an exact value fixation needs no pseudo-time damping (a full Newton
        step converges it in one) and shifting an identity row only slows it.
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_omega1, self.model.sigma_omega2
        )
        shift_diagonal = scalar_transport_shift_diagonal(
            self.mesh,
            self.geometry,
            diffusivity.values,
            self._volume_flux(mdot),
            self.omega_residual(mdot, closure),
            reference,
            fixed_cells=self.wall_cells,
        )
        return ScalarShiftPolicy(shift_diagonal, preconditioner)
