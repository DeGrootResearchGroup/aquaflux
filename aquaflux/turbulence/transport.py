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
import jax
import jax.numpy as jnp

from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import (
    AdvectionFlux,
    DifferenceRow,
    DiffusionFlux,
    FixationRow,
    FixedValueCells,
    ResidualAssembler,
)
from aquaflux.flow import volume_flux
from aquaflux.mesh import distance_to_patches
from aquaflux.properties import FieldProperty, PropertyModel
from aquaflux.schemes import ImposedGradient
from aquaflux.solve import LocalCourantBasis, ShiftBasis
from aquaflux.vectors import norm_squared

from .boundary import (
    log_layer_shear_rate,
    nut_wall,
    omega_wall,
    omega_wall_gradient,
    wall_function_weight,
    wall_k_diffusivity,
)
from .continuation import ScalarShiftPolicy
from .preconditioner import (
    ScalarTransportPreconditioner,
    scalar_transport_preconditioner,
    scalar_transport_shift_diagonal_parts,
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

# The default pseudo-time shift basis (full operator diagonal = uniform under-relaxation), held as a
# module singleton so it is not reconstructed in each method's argument defaults.
_DEFAULT_SHIFT_BASIS = LocalCourantBasis()
# The default near-wall omega fixation row (the plain difference, right when omega is solved
# directly); a module singleton for the same reason.
_DEFAULT_FIXATION_ROW = DifferenceRow()

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aquaflux.boundary import BoundaryConditions
    from aquaflux.discretization import AdvectionScheme
    from aquaflux.flow import VelocityFields
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.schemes import GradientScheme

    from .sst import SSTModel


def _reconstruct_wall_distance_gradient(
    mesh, geometry, gradient_scheme, wall_distance, wall_patches
):
    """Reconstruct the gradient of the wall-distance field, with its exact boundary closure.

    The distance to the nearest wall is **zero on that wall**, so a Dirichlet zero is the exact
    boundary value there; every other patch takes zero gradient. Unlike ``omega``, this field is
    smooth and of order the geometry, so its reconstruction is well posed -- which is what makes it a
    sound carrier for the near-wall ``omega`` chain rule.
    """
    boundary = BoundaryConditions(
        {
            name: (Dirichlet(0.0) if name in set(wall_patches) else ZeroGradient())
            for name in mesh.face_patches.names
        }
    )
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({}),
        (),
        boundary,
        gradient_scheme=gradient_scheme,
    )
    return assembler.gradient(wall_distance)


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
    imposed_omega_gradient : ImposedGradient
        The wall-adjacent cells' analytical ``omega`` gradient (see
        :meth:`SSTTurbulence._wall_omega_gradient`), carried so that every reconstruction of
        ``omega`` in this sweep -- ``grad_omega`` above, and the omega equation's own -- imposes the
        same one. It is a closure field like the rest: computed once from the frozen ``k``.
    """

    nu_t: jnp.ndarray
    strain_rate: jnp.ndarray
    f1: jnp.ndarray
    grad_k: jnp.ndarray
    grad_omega: jnp.ndarray
    omega: jnp.ndarray
    k: jnp.ndarray
    wall_shear_rate: jnp.ndarray
    imposed_omega_gradient: ImposedGradient


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
        The (constant) fluid density, used to form the volume flux ``mdot / rho``. Independent of
        the flow assembler's own (per-cell) density -- :meth:`~aquaflux.turbulence.CoupledRANS.build`
        checks the two agree, since nothing else does and a mismatch is otherwise silent.
    molecular_viscosity : jnp.ndarray
        Kinematic molecular viscosity ``nu`` per cell, shape ``(n_cells,)``.
    wall_distance : jnp.ndarray
        Distance to the nearest wall per cell, shape ``(n_cells,)``.
    wall_distance_gradient : jnp.ndarray
        Gradient of that distance field, shape ``(n_cells, dim)`` -- pure geometry, so it is
        reconstructed once at build. It carries the chain rule for the *analytical* near-wall
        ``omega`` gradient (see :meth:`closure_fields`).
    wall_cells : jnp.ndarray
        Indices of the wall-adjacent cells whose ``omega`` is fixed, shape ``(n_wall,)``.
    wall_faces : jnp.ndarray
        Indices of the wall boundary faces, shape ``(n_wall_faces,)`` — the faces the momentum
        wall-function eddy viscosity is scattered onto (see :meth:`wall_face_eddy_viscosity`).
    k_boundary, omega_boundary : BoundaryConditions
        The scalar boundary closures for each field (Dirichlet inlet / wall, zero-gradient outlet;
        the omega wall is imposed by cell fixation, so its wall closure is a placeholder).
    explicit_production_limiter : bool
        Freeze the k-production cap's ``k`` in the **linearization** (a Patankar / deferred-correction
        treatment): production keeps its exact value, but the term is dropped from the Jacobian
        wherever the cap is active. ``False`` (default) is the exact operator.
    explicit_production_viscosity : bool
        Freeze ``k`` inside the production's **eddy viscosity** for the linearization; static.
        ``nu_t`` is proportional to ``k``, so the production is too, and differentiating that puts a
        **negative** term on the k row's own Jacobian diagonal -- measured on a separating benchmark
        at ``-1.2e-03`` to ``-1.9e-03`` against a pseudo-time shift of ``+2.3e-03``, i.e. a
        near-cancellation that leaves the effective diagonal at 16--47 % of the shift and amplifies
        anything upstream of it thirtyfold. Freezing it takes that diagonal to ``+4.8e-03``.

        ⚠️ **For a Jacobian stand-in, never for the residual a sensitivity is taken through.** The
        limiter above is safe to freeze in the residual wherever its cap is inactive at the root; this
        term is active *everywhere*, so a residual carrying it would make the adjoint wrong
        everywhere. Build a copy with it set and hand that copy's residual to the solve as its
        **operator**, leaving the true residual to decide where the march lands.

        **Opt in only for a forward solve, and only knowingly.** Where the cap is active at the
        converged state, the implicit-function-theorem adjoint linearizes a residual different from
        the one solved -- the fields are right and the **sensitivity is silently wrong**, with a
        perfectly finite gradient coming back. :func:`~aquaflux.turbulence.solve_coupled` therefore
        refuses to return a root reached with this set whose cap is active; see
        :func:`~aquaflux.turbulence.production_cap_active`.

        On a turbulent channel and on the 3D backward-facing step the two settings produce
        **identical** marches -- same step count, same cycle count, same converged fields -- so on
        both of those the stabilization buys nothing the exact operator does not already give.

        **Who still wants it:** a bare or weakly preconditioned *segregated scalar* solve. The exact
        derivative of the cap is indefinite where the cap is active, and far from the solution the cap
        is active almost everywhere -- on the k-equation unit case it binds in 24 of 24 cells at the
        starting field and in 0 at the closure's own ``k`` -- so an unpreconditioned k-solve stalls on
        the exact operator and converges on the frozen one. A preconditioned solve does not need it
        (the convection-diffusion multigrid rescues the exact operator, which then converges
        quadratically), and the coupled path always carries a preconditioner *and* takes gradients,
        which is why the default belongs on the exact operator.
    """

    model: SSTModel
    mesh: Mesh
    geometry: MeshGeometry
    gradient_scheme: GradientScheme
    advection_scheme: AdvectionScheme
    density: float
    molecular_viscosity: jnp.ndarray
    wall_distance: jnp.ndarray
    wall_distance_gradient: jnp.ndarray
    wall_cells: jnp.ndarray
    wall_faces: jnp.ndarray
    k_boundary: BoundaryConditions
    omega_boundary: BoundaryConditions
    explicit_production_limiter: bool = eqx.field(static=True, default=False)
    explicit_production_viscosity: bool = eqx.field(static=True, default=False)

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
        explicit_production_limiter: bool = False,
        explicit_production_viscosity: bool = False,
    ) -> SSTTurbulence:
        """Build the assembler, deriving the wall distance and wall-adjacent cell set.

        Parameters
        ----------
        wall_patches : sequence of str
            The boundary patches treated as walls; their wall distance is computed and their
            owner cells become the ``omega`` fixation set.
        explicit_production_limiter : bool
            Linearization of the k-production limiter for the forward solve (see the class
            attribute); ``False`` (default) is the exact operator. ``True`` is the robust choice for a
            bare or weakly preconditioned segregated scalar solve, and only for one -- it makes the
            adjoint wrong wherever the cap is active at the root.
        explicit_production_viscosity : bool
            Freeze ``k`` inside the production's **eddy viscosity** for the linearization (see the
            class attribute); ``False`` (default) is the exact operator. ⚠️ Unlike the limiter above
            this term is active *everywhere*, never only where a cap bites, so a copy carrying it is
            for a **Jacobian stand-in** and never for the residual a sensitivity is taken through.

        The remaining arguments are stored directly (see the class attributes).
        """
        wall_distance = distance_to_patches(mesh, geometry, wall_patches)
        wall_faces = jnp.concatenate([mesh.face_patches.indices(p) for p in wall_patches])
        wall_cells = jnp.unique(mesh.face_cells.owner[wall_faces])
        # The wall-distance gradient is pure geometry, so it is reconstructed once here rather
        # than per residual evaluation. Its boundary closure is exact: the distance to a wall is
        # zero *on* that wall, and zero-gradient elsewhere.
        wall_distance_gradient = _reconstruct_wall_distance_gradient(
            mesh, geometry, gradient_scheme, wall_distance, wall_patches
        )
        return cls(
            model=model,
            mesh=mesh,
            geometry=geometry,
            gradient_scheme=gradient_scheme,
            advection_scheme=advection_scheme,
            density=density,
            molecular_viscosity=molecular_viscosity,
            wall_distance=wall_distance,
            wall_distance_gradient=wall_distance_gradient,
            wall_cells=wall_cells,
            wall_faces=wall_faces,
            k_boundary=k_boundary,
            omega_boundary=omega_boundary,
            explicit_production_limiter=explicit_production_limiter,
            explicit_production_viscosity=explicit_production_viscosity,
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

    def with_scaled_molecular_viscosity(self, factor: float) -> SSTTurbulence:
        """Return a copy whose kinematic molecular viscosity is multiplied by ``factor``.

        Scales the per-cell ``molecular_viscosity`` field the k/omega diffusion is built on; the SST
        constants and boundaries are unchanged. The turbulence half of the Reynolds-number rescale a
        homotopy applies to a lower-Re companion problem. ``factor`` is a plain multiplier and a tracer
        flows through it under differentiation.

        Parameters
        ----------
        factor : float
            The multiplier applied to the molecular viscosity.

        Returns
        -------
        SSTTurbulence
            A new closure at the scaled molecular viscosity; ``self`` is unchanged.
        """
        return eqx.tree_at(lambda t: t.molecular_viscosity, self, self.molecular_viscosity * factor)

    def _volume_flux(self, mdot: jnp.ndarray) -> jnp.ndarray:
        """The volumetric face flux the kinematic ``k``/``omega`` transport advects on.

        Shared with every other scalar carried by the flow (see
        :func:`~aquaflux.flow.volume_flux`), so the conversion has one definition. The
        constant-density caveat in this module's docstring is about writing ``k`` and ``omega`` in
        kinematic form at all, not about this conversion.
        """
        return volume_flux(mdot, self.density)

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

    def _assembler(
        self,
        properties: PropertyModel,
        flux_operators: tuple,
        boundary: BoundaryConditions,
        *,
        source_operators: tuple = (),
        imposed_gradient: ImposedGradient | None = None,
    ) -> ResidualAssembler:
        """An assembler on this turbulence model's mesh, geometry and gradient scheme.

        Every equation assembled here -- ``k``, ``omega``, and the bare reconstruction
        :meth:`_field_gradient` uses -- differs only in its properties, operators, boundary closures
        and imposition; the mesh, the geometry and the injected gradient scheme are this object's and
        are the same for all of them. Written once so an assembler-level choice cannot be wired into
        one equation and missed in another.

        Parameters
        ----------
        properties : PropertyModel
            The named per-cell properties this equation reads.
        flux_operators : tuple of FaceFluxOperator
            Its face-flux operators, in the order they are summed.
        boundary : BoundaryConditions
            Its boundary closures.
        source_operators : tuple of VolumeSource, optional
            Its volume sources (default none).
        imposed_gradient : ImposedGradient, optional
            Cells whose gradient of this equation's field is a model quantity rather than something
            to reconstruct (default none).

        Returns
        -------
        ResidualAssembler
            The assembler for that equation.
        """
        return ResidualAssembler.build(
            self.mesh,
            self.geometry,
            properties,
            flux_operators,
            boundary,
            source_operators=source_operators,
            gradient_scheme=self.gradient_scheme,
            imposed_gradient=imposed_gradient,
        )

    def _field_gradient(
        self,
        field: jnp.ndarray,
        boundary: BoundaryConditions,
        *,
        imposed: ImposedGradient | None = None,
    ) -> jnp.ndarray:
        """Reconstruct the cell gradient of a turbulence field with its boundary closures.

        Reuses the residual assembler's leading-order gradient reconstruction (the injected gradient
        scheme evaluated with the field's boundary values), so the boundary handling is not
        re-implemented here. ``imposed`` names cells whose gradient is a model quantity and is passed
        through to the scheme rather than applied to what it returns.
        """
        assembler = self._assembler(PropertyModel({}), (), boundary, imposed_gradient=imposed)
        return assembler.gradient(field)

    def _wall_omega_gradient(self, k: jnp.ndarray, grad_k: jnp.ndarray) -> ImposedGradient:
        """The wall-adjacent cells' ``omega`` gradient, analytical rather than reconstructed.

        Those cells do not solve a transport balance -- their ``omega`` is *imposed*
        (:func:`~aquaflux.turbulence.omega_wall`) -- so their gradient is a model quantity too, and
        reconstructing it from neighbours is both inconsistent and badly inaccurate: ``omega_wall``
        varies like ``1/d**2`` (strongly convex) while the reconstruction is a linear fit, and the
        stencil folds in the wall face, whose ``omega`` a zero-gradient closure sets to the cell value
        even though the true profile diverges there. Measured on a backward-facing step, the
        reconstructed magnitude is ~0.26x the analytical one in these cells.

        The analytical gradient is exact and cheap
        (:func:`~aquaflux.turbulence.omega_wall_gradient`); only the wall-adjacent rows are given,
        every other cell keeps its reconstruction. Replacing them is safe because ``omega`` needs no
        wall-normal flux at these cells -- the row is a value fixation and the wall closure is
        zero-gradient -- so the only consumers are inward: the cross-diffusion and ``F1`` blend that
        read ``grad(k).grad(omega)``, and the diffusion's non-orthogonal correction on faces to
        interior neighbours.

        Returned as an :class:`~aquaflux.schemes.ImposedGradient` and handed **to** the
        reconstruction rather than applied to its result, because a scheme that differentiates its
        own first estimate -- which is how
        :class:`~aquaflux.schemes.MultipleCorrectionGradient` forms a Hessian -- would otherwise
        build on the estimate this replaces and receive the correction only afterwards. It also
        settles the wall faces' *gradient*, which a closure would otherwise have to invent from a
        boundary ``omega`` that is not data.

        Parameters
        ----------
        k : jnp.ndarray
            The turbulent kinetic energy per cell, shape ``(n_cells,)``.
        grad_k : jnp.ndarray
            Its reconstructed cell gradient, shape ``(n_cells, dim)``.

        Returns
        -------
        ImposedGradient
            The wall-adjacent cells and the ``omega`` gradient to impose there.
        """
        wall = self.wall_cells
        return ImposedGradient(
            wall,
            omega_wall_gradient(
                self.molecular_viscosity[wall],
                self.wall_distance[wall],
                k[wall],
                self.wall_distance_gradient[wall],
                grad_k[wall],
                self.model,
            ),
        )

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
        imposed_omega = self._wall_omega_gradient(k, grad_k)
        grad_omega = self._field_gradient(omega, self.omega_boundary, imposed=imposed_omega)
        f1 = self.model.f1(
            k, omega, self.molecular_viscosity, self.wall_distance, grad_k, grad_omega
        )
        nu_t = self.model.eddy_viscosity(
            k, omega, strain, self.molecular_viscosity, self.wall_distance
        )
        return SSTClosureFields(
            nu_t,
            strain,
            f1,
            grad_k,
            grad_omega,
            omega,
            k,
            self.wall_shear_rate(velocity),
            imposed_omega,
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
        assembler = self._assembler(
            PropertyModel({"diffusivity": diffusivity}),
            (
                AdvectionFlux(self._volume_flux(mdot), self.advection_scheme),
                DiffusionFlux(
                    boundary_coefficient=self._wall_k_diffusivity(diffusivity, closure.k)
                ),
            ),
            self.k_boundary,
            source_operators=(
                self._k_production(closure, near_wall),
                KDestruction(closure.omega, self.model, near_wall=near_wall),
            ),
        )
        return assembler.residual

    def _production_viscosity(self, closure: SSTClosureFields) -> jnp.ndarray:
        """The eddy viscosity the k-production reads, shape ``(n_cells,)``.

        :attr:`explicit_production_viscosity` freezes ``k`` **here and only here** -- in the
        ``a_1 k / max(a_1 omega, S F_2)`` that makes production proportional to ``k``. Everything else
        stays live: ``omega``, the strain rate (including its own near-wall ``k`` blend), the k
        diffusivity's ``nu_t``, the destruction, and the momentum closure. So what is removed is the
        single derivative that makes the k row's Jacobian diagonal negative, and nothing else.

        The strain rate is taken from the closure rather than recomputed, so the frozen viscosity is
        the same function of the same inputs as the live one but for that one argument.
        """
        if not self.explicit_production_viscosity:
            return closure.nu_t
        return self.model.eddy_viscosity(
            jax.lax.stop_gradient(closure.k),
            closure.omega,
            closure.strain_rate,
            self.molecular_viscosity,
            self.wall_distance,
        )

    def _k_production(self, closure: SSTClosureFields, near_wall: NearWallKClosure) -> KProduction:
        """The k-production operator for this sweep's ``closure``.

        Built here rather than inline in :meth:`k_residual` because the pseudo-time shift diagonal
        needs the *same* operator to ask it for its k-feedback rate
        (:meth:`~aquaflux.turbulence.KProduction.feedback_rate`); a second construction would drift
        from the residual's the first time a constructor argument changed.
        """
        return KProduction(
            self._production_viscosity(closure),
            closure.strain_rate,
            closure.omega,
            self.model,
            explicit_limiter=self.explicit_production_limiter,
            near_wall=near_wall,
        )

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
        self,
        mdot: jnp.ndarray,
        closure: SSTClosureFields,
        fixation_row: FixationRow = _DEFAULT_FIXATION_ROW,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The omega-equation residual function ``omega -> R_omega`` for the frozen ``closure``.

        Advection, diffusion of ``nu + sigma_omega nu_t``, the production/destruction/cross-diffusion
        sources, and the near-wall cells fixed to the analytical ``omega`` (their balance replaced).

        Parameters
        ----------
        mdot : jnp.ndarray
            The flow's Rhie--Chow mass flux, shape ``(n_faces,)``.
        closure : SSTClosureFields
            The frozen closure fields of the current sweep.
        fixation_row : FixationRow
            How the near-wall fixation is written as a residual row (see
            :class:`~aquaflux.discretization.FixationRow`). Defaults to the plain difference
            ``omega - omega_wall``, which is right when ``omega`` itself is the solved unknown. A
            caller solving in a transformed variable passes that variable's own row form -- the
            coupled solve takes it from its ``omega_transform`` -- so the fixation stays linear in the
            unknown actually being stepped.
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_omega1, self.model.sigma_omega2
        )
        assembler = self._assembler(
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
            # The same imposition the closure fields were built with. This equation reconstructs
            # `omega`'s gradient again for the diffusion's non-orthogonal correction, and a
            # reconstruction that is a model quantity in the wall cells is one wherever it is made --
            # otherwise the residual and the closure disagree about the same derivative.
            imposed_gradient=closure.imposed_omega_gradient,
        )
        wall_fix = FixedValueCells(
            self.wall_cells,
            omega_wall(
                self.molecular_viscosity[self.wall_cells],
                self.wall_distance[self.wall_cells],
                closure.k[self.wall_cells],
                self.model,
            ),
            fixation_row,
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

    def _scalar_shift_diagonal(
        self,
        diffusivity_values: jnp.ndarray,
        mdot: jnp.ndarray,
        residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
        reference: jnp.ndarray,
        shift_basis: ShiftBasis,
        fixed_cells: jnp.ndarray | None = None,
        source_feedback: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Combine the scalar shift diagonal's convective/dissipative parts through ``shift_basis``.

        Shared by :meth:`k_shift_policy` and :meth:`omega_shift_policy`: the default
        :class:`~aquaflux.solve.LocalCourantBasis` (weight ``1``) reproduces the full operator
        diagonal (uniform under-relaxation); a convective basis (weight ``0``) gives a local convective
        pseudo-time step on the scalar.

        ``source_feedback`` is the volume-integrated source derivative ``residual_fn`` does not carry
        (see :func:`~aquaflux.turbulence.preconditioner.scalar_transport_shift_diagonal_parts`).
        """
        convective, dissipative = scalar_transport_shift_diagonal_parts(
            self.mesh,
            self.geometry,
            diffusivity_values,
            self._volume_flux(mdot),
            residual_fn,
            reference,
            fixed_cells=fixed_cells,
            source_feedback=source_feedback,
        )
        return shift_basis.local_diagonal(convective, dissipative)

    def k_shift_policy(
        self,
        mdot: jnp.ndarray,
        closure: SSTClosureFields,
        reference: jnp.ndarray,
        *,
        preconditioner: ScalarTransportPreconditioner | None = None,
        shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
        live_eddy_viscosity: bool = False,
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
        live_eddy_viscosity : bool
            Whether the residual this shift damps differentiates through ``nu_t``. A **segregated**
            k-solve freezes ``nu_t`` for the sweep, so ``False`` (the default) is right there and the
            shift is the frozen-closure operator diagonal exactly. A **coupled** residual recomputes
            ``nu_t = a_1 k / max(a_1 omega, S F_2)`` from the current ``k``, so the production
            ``nu_t S**2`` is proportional to ``k`` and feeds ``k`` back into itself -- a term the
            frozen-closure residual used to build the shift contains no trace of, and which is what
            drives the coupled k row's Jacobian diagonal negative. Set it there, and the production's
            feedback rate (:meth:`~aquaflux.turbulence.KProduction.feedback_rate`) is folded into the
            diagonal so the shift describes the equation actually being solved. Left off, the two can
            differ by more than the whole shift in a wall-adjacent cell, and ``J + beta d`` is then a
            near-cancellation there rather than a damped row -- so a small upstream difference in the
            velocity gradient comes out as a large difference in the correction the step asks for.
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_k1, self.model.sigma_k2
        )
        feedback = None
        if live_eddy_viscosity:
            production = self._k_production(
                closure, self._near_wall_closure(closure.wall_shear_rate)
            )
            feedback = production.feedback_rate(
                reference,
                self.geometry.cell.volume,
                # The identity `dP_k/dk = P_k/k` rests on `nu_t` tracking `k`. If this assembler
                # freezes it, that path contributes no feedback and folding one in would over-damp.
                live_viscosity=not self.explicit_production_viscosity,
            )
        shift_diagonal = self._scalar_shift_diagonal(
            diffusivity.values,
            mdot,
            self.k_residual(mdot, closure),
            reference,
            shift_basis,
            source_feedback=feedback,
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
        shift_basis: ShiftBasis = _DEFAULT_SHIFT_BASIS,
    ) -> ScalarShiftPolicy:
        """The pseudo-transient continuation policy for the omega-equation solve.

        As :meth:`k_shift_policy`, with the omega diffusivity and the near-wall fixed cells: their
        shift is zeroed, since an exact value fixation needs no pseudo-time damping (a full Newton
        step converges it in one) and shifting an identity row only slows it.

        **There is deliberately no ``live_eddy_viscosity`` counterpart here.** That switch exists
        because the k production ``nu_t S**2`` is proportional to ``k`` once ``nu_t`` is live, so it
        sits on the k row's own diagonal. The omega production ``alpha min(S**2, cap)`` carries no
        solved ``omega`` at all -- what it gains from a live closure is a dependence on ``k`` and
        ``nu_t``, which is an off-diagonal coupling to *another* block and not this row's time scale.
        The one omega source that does read the solved field, the cross-diffusion ``~ 1 / omega``,
        contributes with the *stabilizing* sign and is orders below the destruction's ``2 beta omega``
        wherever it is not simply zeroed as a wall fixation.
        """
        diffusivity = self._diffusivity(
            closure.nu_t, closure.f1, self.model.sigma_omega1, self.model.sigma_omega2
        )
        shift_diagonal = self._scalar_shift_diagonal(
            diffusivity.values,
            mdot,
            self.omega_residual(mdot, closure),
            reference,
            shift_basis,
            fixed_cells=self.wall_cells,
        )
        return ScalarShiftPolicy(shift_diagonal, preconditioner)
