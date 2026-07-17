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

from .boundary import omega_wall_value
from .preconditioner import scalar_transport_preconditioner
from .sources import (
    KDestruction,
    KProduction,
    OmegaCrossDiffusion,
    OmegaDestruction,
    OmegaProduction,
)
from .strain import strain_rate_magnitude

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aquaflux.boundary import BoundaryConditions
    from aquaflux.discretization import AdvectionScheme
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
    """

    nu_t: jnp.ndarray
    strain_rate: jnp.ndarray
    f1: jnp.ndarray
    grad_k: jnp.ndarray
    grad_omega: jnp.ndarray
    omega: jnp.ndarray


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
            k_boundary=k_boundary,
            omega_boundary=omega_boundary,
            explicit_production_limiter=explicit_production_limiter,
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
        strain = strain_rate_magnitude(velocity_gradient)
        return self.model.eddy_viscosity(
            k, omega, strain, self.molecular_viscosity, self.wall_distance
        )

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

    def closure_fields(
        self, velocity_gradient: jnp.ndarray, k: jnp.ndarray, omega: jnp.ndarray
    ) -> SSTClosureFields:
        """Assemble the frozen SST closure fields for the current state.

        Computes the strain rate from the velocity gradient, reconstructs ``grad k`` and
        ``grad omega`` with their boundary closures, and evaluates ``F1`` and the eddy viscosity --
        the fields the k and omega equation builders freeze for a sweep.

        Parameters
        ----------
        velocity_gradient : jnp.ndarray
            The velocity-gradient tensor, shape ``(n_cells, dim, dim)``.
        k, omega : jnp.ndarray
            The current turbulence fields, shape ``(n_cells,)``.
        """
        strain = strain_rate_magnitude(velocity_gradient)
        grad_k = self._field_gradient(k, self.k_boundary)
        grad_omega = self._field_gradient(omega, self.omega_boundary)
        f1 = self.model.f1(
            k, omega, self.molecular_viscosity, self.wall_distance, grad_k, grad_omega
        )
        nu_t = self.model.eddy_viscosity(
            k, omega, strain, self.molecular_viscosity, self.wall_distance
        )
        return SSTClosureFields(nu_t, strain, f1, grad_k, grad_omega, omega)

    def k_residual(
        self, mdot: jnp.ndarray, closure: SSTClosureFields
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The k-equation residual function ``k -> R_k`` for the frozen ``closure`` and ``mdot``.

        Advection on the volume flux, diffusion of ``nu + sigma_k nu_t``, and the limited production
        minus destruction sources.
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_k1, self.model.sigma_k2
        )
        assembler = ResidualAssembler.build(
            self.mesh,
            self.geometry,
            PropertyModel({"diffusivity": diffusivity}),
            (
                AdvectionFlux(self._volume_flux(mdot), self.advection_scheme),
                DiffusionFlux(),
            ),
            self.k_boundary,
            source_operators=(
                KProduction(
                    closure.nu_t,
                    closure.strain_rate,
                    closure.omega,
                    self.model,
                    explicit_limiter=self.explicit_production_limiter,
                ),
                KDestruction(closure.omega, self.model),
            ),
            gradient_scheme=self.gradient_scheme,
        )
        return assembler.residual

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
                OmegaProduction(closure.strain_rate, closure.f1, self.model),
                OmegaDestruction(closure.f1, self.model),
                OmegaCrossDiffusion(
                    closure.omega, closure.grad_k, closure.grad_omega, closure.f1, self.model
                ),
            ),
            gradient_scheme=self.gradient_scheme,
        )
        wall_fix = FixedValueCells(
            self.wall_cells,
            omega_wall_value(
                self.molecular_viscosity[self.wall_cells],
                self.wall_distance[self.wall_cells],
                self.model,
            ),
        )

        def residual(omega: jnp.ndarray) -> jnp.ndarray:
            return wall_fix.apply(assembler.residual(omega), omega)

        return residual

    def k_preconditioner(
        self,
        mdot: jnp.ndarray,
        closure: SSTClosureFields,
        reference: jnp.ndarray,
        *,
        method: str = "twolevel",
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """A convection-diffusion AMG preconditioner for the k-equation linear solve.

        Frozen for the sweep's ``closure`` and ``mdot`` (the same fields ``k_residual`` uses), it
        makes the k-solve's Krylov iteration mesh-independent at the high cell Peclet where an
        unpreconditioned solve would otherwise cost ``O(N)`` iterations (or stall). See
        :func:`~aquaflux.turbulence.preconditioner.scalar_transport_preconditioner`.
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
        )

    def omega_preconditioner(
        self,
        mdot: jnp.ndarray,
        closure: SSTClosureFields,
        reference: jnp.ndarray,
        *,
        method: str = "twolevel",
    ) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
        """A convection-diffusion AMG preconditioner for the omega-equation linear solve.

        As :meth:`k_preconditioner`, with the omega diffusivity and the near-wall fixed cells detached
        from the coarsening (their rows are the value fixation, not a transport balance).
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
        )
