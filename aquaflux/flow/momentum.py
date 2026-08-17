"""The coupled momentum--continuity residual for incompressible flow.

Assembles one residual over the whole flow state ``(u, v[, w], p)`` in a system-first
design. The unknowns are stored as a single flat vector with the layout
``[vel_0, vel_1, ..., vel_{dim-1}, pressure]`` (each block ``n_cells`` long), so the coupled
system is solved by the same Newton / implicit-diff machinery as a scalar field.

Per velocity component ``i`` the momentum balance is a scalar transport of ``u_i``:

    R_{u_i} = sum_faces ( mdot_f u_{i,f}  +  p_f n_i A  -  mu (grad u_i . n) A )  -  f_i V

— advection of ``u_i`` by the mass flux, a pressure force, and viscous diffusion, less the volume
integral of any body force / injected momentum source ``f_i``. The first three are
face-flux operators summed by a :class:`~aquaflux.discretization.CellBalance`, the same
composition every scalar transport equation uses: the first and last are
:class:`~aquaflux.discretization.AdvectionFlux` and
:class:`~aquaflux.discretization.DiffusionFlux` verbatim (viscosity as the diffusion
coefficient), and only :class:`PressureForce` is new. Continuity is

    R_p = sum_faces mdot_f ,

with ``mdot_f`` the Rhie--Chow mass flux, which couples pressure implicitly and prevents
checkerboarding. The mass flux and the momentum diagonal ``a_P`` (differentiated here, and lagged
only in the mass-flux estimate its convective term uses) come from :mod:`aquaflux.flow.rhie_chow`;
the Jacobian of the whole coupled residual comes from AD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax.numpy as jnp

from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import (
    AdvectionFlux,
    CellBalance,
    DiffusionFlux,
    FaceContext,
    FaceFluxOperator,
    FixedValueCells,
)
from aquaflux.properties import PropertyModel
from aquaflux.schemes.interpolation import (
    interpolate_to_face,
    interpolation_factor,
)
from aquaflux.vectors import dot

from .rhie_chow import (
    advective_momentum_flux,
    interior_mass_flux,
    momentum_diagonal,
    momentum_diagonal_parts,
)
from .source import MomentumSource
from .state import BlockStateLayout

if TYPE_CHECKING:
    from aquaflux.discretization import AdvectionScheme
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.schemes import GradientScheme


class VelocityFields(NamedTuple):
    """The kinematic velocity state: cell values, boundary-face values, and the cell gradient.

    The part of a flow state that is a pure function of the velocity unknowns -- no pressure, no
    ``a_P``, no Rhie--Chow flux -- and the whole of what a turbulence closure reads from the
    flow. It is produced both by the lightweight :meth:`MomentumContinuity.velocity_fields` (before a
    mass flux is even defined) and, as part of :class:`FlowFields`, by the full assembly, so a
    consumer takes this one bundle rather than three arrays that always travel together.

    Attributes
    ----------
    velocity : jnp.ndarray
        Cell velocity, shape ``(n_cells, dim)``.
    boundary_velocity : jnp.ndarray
        Boundary-face velocity from the flow boundary conditions, shape ``(n_faces, dim)`` (the
        entries on interior faces are unused). The wall value a near-wall shear rate measures the
        cell velocity against.
    gradient : jnp.ndarray
        Cell velocity-gradient tensor, shape ``(n_cells, dim, dim)``, ``[c, i, j] = d u_i / d x_j``.
    """

    velocity: jnp.ndarray  # (n_cells, dim)
    boundary_velocity: jnp.ndarray  # (n_faces, dim)
    gradient: jnp.ndarray  # (n_cells, dim, dim), [c, i, j] = d u_i/d x_j


class FlowFields(NamedTuple):
    """The per-evaluation flow quantities the residual assembles once and shares.

    Returned by :meth:`MomentumContinuity.flow_fields`, so a caller that needs several of these
    quantities at one state (a coupled residual wanting both the residual and the mass flux, a
    segregated sweep wanting both the velocity fields and the mass flux) assembles them **once**
    and reads the fields it needs, rather than re-deriving the boundary fields, gradients, ``a_P``,
    and Rhie--Chow flux per accessor.

    The kinematic half is the nested :class:`VelocityFields`, which is also what the lightweight
    :meth:`MomentumContinuity.velocity_fields` returns on its own.
    """

    velocity_fields: VelocityFields
    pressure: jnp.ndarray  # (n_cells,)
    boundary_pressure: jnp.ndarray  # (n_faces,)
    grad_pressure: jnp.ndarray  # (n_cells, dim)
    mdot: jnp.ndarray  # (n_faces,) Rhie--Chow face mass flux


class PressureForce(FaceFluxOperator):
    """The pressure force on one momentum component as a face flux, ``p_f n_i A``.

    The pressure term of the momentum balance is the surface integral ``∮ p n dA``, so its
    owner-outward contribution to component ``i`` through a face is ``p_f n_i A`` -- a flux of
    momentum through that face, and therefore an ordinary
    :class:`~aquaflux.discretization.FaceFluxOperator` rather than a term the momentum residual has
    to add by hand. Composing it with the viscous and advective fluxes is what makes each momentum
    component a plain scalar transport, assembled by the same
    :class:`~aquaflux.discretization.CellBalance` every other equation uses.

    It does not read the transported field: the pressure is a *different* unknown of the coupled
    system, carried here as the already-reconstructed face value, exactly as
    :class:`~aquaflux.discretization.AdvectionFlux` carries its prescribed mass flux. The
    pressure--velocity coupling is not lost by that -- ``face_pressure`` is a differentiable
    function of the pressure unknowns, so automatic differentiation of the assembled residual
    recovers the full ``dR_momentum / dp`` block.

    Attributes
    ----------
    face_pressure : jnp.ndarray
        Pressure at each face's integration point, shape ``(n_faces,)`` -- interpolated (with its
        skewness correction) on interior faces and taken from the boundary closure on boundary
        faces, so the force stays second order on a non-orthogonal mesh.
    component : int
        Which momentum component this is the force on, indexing the face normal (static).
    """

    face_pressure: jnp.ndarray
    component: int = eqx.field(static=True)

    def face_flux(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        """Owner-outward pressure force per face, shape ``(n_faces,)``.

        Parameters
        ----------
        field : jnp.ndarray
            The transported velocity component, shape ``(n_cells,)``. Unused -- see the class
            docstring.
        context : FaceContext
            The shared per-face inputs; the face normal and area are gathered from its geometry.
        """
        face = context.geometry.face
        return self.face_pressure * face.normal[:, self.component] * face.area


class MomentumContinuity(eqx.Module):
    """Coupled momentum + Rhie--Chow continuity residual for steady incompressible flow.

    Construct with :meth:`build`. The residual acts on the flat state vector (see module
    docstring); :meth:`pack` / :meth:`unpack` convert to and from ``(velocity, pressure)``.

    Attributes
    ----------
    mesh : Mesh
        Topology (owner/neighbour connectivity, patch labels).
    geometry : MeshGeometry
        Face and cell metrics (areas, owner-outward normals, centroids, volumes).
    properties : PropertyModel
        The fluid's **material** properties, supplying its per-cell molecular ``"viscosity"`` and
        ``"density"``. These describe the fluid alone and are never overwritten by a flow state; a
        turbulence closure's contribution rides separately in :attr:`eddy_viscosity`, and
        :attr:`viscosity` combines the two.
    eddy_viscosity : jnp.ndarray or None
        Per-cell **kinematic** eddy viscosity ``nu_t`` from a Reynolds-averaged Navier--Stokes (RANS)
        closure, shape ``(n_cells,)``, or ``None`` for laminar flow. Set with
        :meth:`with_eddy_viscosity`. A differentiable leaf, so a coupled residual that computes
        ``nu_t`` from ``(k, omega)`` differentiates through it.
    wall_eddy_viscosity : jnp.ndarray or None
        Per-face **kinematic** wall-function eddy viscosity ``nu_t,wall``, shape ``(n_faces,)``
        (meaningful on shearing-wall faces), or ``None`` for resolved walls. Also set with
        :meth:`with_eddy_viscosity`; it overrides the momentum wall-face diffusion coefficient with
        ``mu + rho nu_t,wall`` (see :meth:`_wall_boundary_viscosity`).
    gradient_scheme : GradientScheme
        Reconstruction for the velocity and pressure gradients.
    advection_scheme : AdvectionScheme or None
        Momentum convection scheme; ``None`` gives Stokes flow (no convection). A limited scheme
        (``LimitedUpwind``) carries its own slope limiter.
    boundary : BoundaryConditions
        The named per-patch flow closures, resolved to their boundary-face indices.
    interp_factor, normal_distance : jnp.ndarray
        Face interpolation factor ``g`` and normal distance ``d . n`` (precomputed geometry).
    pressure_pin : int or None
        Cell whose continuity equation is replaced by ``p = pressure_pin_value`` (static). Required
        for a closed domain (all-wall, no pressure outlet), where the pressure level is otherwise
        free; ``None`` for a domain with a pressure outlet.
    pressure_pin_value : float
        The pressure imposed at :attr:`pressure_pin`.
    body_force : jnp.ndarray
        Uniform body force per unit volume ``(dim,)``, added to the momentum equation. Drives a
        streamwise-periodic channel: with the pressure split ``p = p̃ + G·x`` into a periodic ``p̃``
        and a mean gradient ``G``, the linear part is a constant force ``f = −G``, so a positive
        ``body_force[0]`` drives the flow in ``+x`` (mean gradient ``G = −body_force``). Default zero.

        This is deliberately **not** one of :attr:`sources`, because it is not only a source term: it
        is the *control variable* of the bulk-velocity-constrained solve
        (:func:`~aquaflux.flow.bulk_velocity_flow_solve`), which treats it as a coupled unknown,
        writes it here every residual evaluation, and forms its border column from the analytic
        ``dR/d(body_force) = −V`` that holds only for a uniform, state-independent force.
    sources : tuple of MomentumSource
        Momentum source terms subtracted from the balance (each returns its cell integral,
        production positive); empty by default. Where buoyancy, porous drag, or a rotating-frame
        term goes.
    """

    mesh: Mesh
    geometry: MeshGeometry
    properties: PropertyModel
    gradient_scheme: GradientScheme
    advection_scheme: AdvectionScheme | None
    boundary: BoundaryConditions
    interp_factor: jnp.ndarray
    normal_distance: jnp.ndarray
    body_force: jnp.ndarray
    pressure_pin: int | None = eqx.field(static=True)
    pressure_pin_value: float
    sources: tuple[MomentumSource, ...] = ()
    eddy_viscosity: jnp.ndarray | None = None
    wall_eddy_viscosity: jnp.ndarray | None = None

    @classmethod
    def build(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        properties: PropertyModel,
        gradient_scheme: GradientScheme,
        boundary: BoundaryConditions,
        *,
        advection_scheme: AdvectionScheme | None = None,
        pressure_pin: int | None = None,
        pressure_pin_value: float = 0.0,
        body_force=None,
        sources: tuple[MomentumSource, ...] = (),
    ) -> MomentumContinuity:
        """Build the coupled assembler, precomputing face interpolation geometry.

        ``boundary`` is a :class:`~aquaflux.boundary.BoundaryConditions` collection of per-patch
        flow closures (``BoundaryConditions({name: FlowBoundary})``), bound to ``mesh.face_patches``
        internally. ``properties`` must supply ``"viscosity"`` and ``"density"``. ``pressure_pin``
        fixes the pressure at one cell (its continuity equation is replaced by
        ``p = pressure_pin_value``) — required for a closed domain (all-wall, no pressure outlet, e.g.
        a streamwise-periodic channel), where pressure is otherwise defined only up to a constant.
        ``body_force`` is a uniform force per unit volume ``(dim,)`` added to the momentum equation
        (see :attr:`body_force`); default (``None``) is no force. It drives a periodic channel and is
        the leaf a mass-flow controller updates via ``eqx.tree_at``. ``sources`` is the tuple of
        :class:`~aquaflux.flow.MomentumSource` terms — buoyancy, porous drag, a rotating-frame term —
        subtracted from the momentum balance; it is separate from ``body_force`` because that one is
        also a solve control variable (see :attr:`body_force`), and the two simply add.
        """
        properties.require("viscosity", "density")
        force = jnp.zeros(mesh.dim) if body_force is None else jnp.asarray(body_force)
        face_geometry, cell_geometry = geometry.face, geometry.cell
        face_cells = mesh.face_cells
        owner = face_cells.owner
        interior = face_cells.interior
        x_p = cell_geometry.centroid[owner]
        x_ip = face_geometry.centroid
        d = (
            face_cells.neighbour_centroid(cell_geometry.centroid) - x_p
        )  # periodic-image across seam
        interp_factor = interpolation_factor(face_cells, geometry)
        normal_distance = jnp.where(
            interior,
            dot(d, face_geometry.normal),
            dot(x_ip - x_p, face_geometry.normal),
        )
        return cls(
            mesh=mesh,
            geometry=geometry,
            properties=properties,
            gradient_scheme=gradient_scheme,
            advection_scheme=advection_scheme,
            boundary=boundary.resolve(mesh.face_patches),
            interp_factor=interp_factor,
            normal_distance=normal_distance,
            body_force=force,
            pressure_pin=pressure_pin,
            pressure_pin_value=pressure_pin_value,
            sources=sources,
        )

    # --- state layout ------------------------------------------------------------------

    @property
    def _layout(self) -> BlockStateLayout:
        """The flat-vector block layout ``[vel_0..vel_{dim-1}, pressure]`` for this system's state."""
        return BlockStateLayout(self.mesh.dim, self.mesh.n_cells)

    def unpack(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Split the flat state into velocity ``(n_cells, dim)`` and pressure ``(n_cells,)``."""
        return self._layout.unpack(state)

    def pack(self, velocity_residual: jnp.ndarray, pressure_residual: jnp.ndarray) -> jnp.ndarray:
        """Assemble component momentum residuals and the continuity residual into a flat vector."""
        return self._layout.pack(velocity_residual, pressure_residual)

    def initial_state(self) -> jnp.ndarray:
        """A zero flat state vector, shape ``((dim + 1) n_cells,)``."""
        return self._layout.zeros()

    def with_eddy_viscosity(
        self, eddy_viscosity: jnp.ndarray, wall_eddy_viscosity: jnp.ndarray | None = None
    ) -> MomentumContinuity:
        """Return a copy carrying the turbulence closure's eddy viscosity ``nu_t``.

        The one seam a RANS closure enters the momentum block through. The closure supplies only its
        own quantity, the **kinematic** eddy viscosity; combining it with the fluid's molecular
        viscosity into the momentum diffusion coefficient ``mu_eff = mu + rho nu_t`` is done once, by
        :attr:`viscosity`, so no caller restates the closure relation.

        ``nu_t`` replaces a dedicated leaf rather than overwriting the material properties, which
        means the molecular viscosity in :attr:`properties` stays intact and authoritative (so this
        is idempotent — applying it twice does not accumulate). A segregated outer loop applies it
        once per sweep with the closure frozen; the coupled residual applies it live, so
        ``dR_momentum / d(k, omega)`` flows through ``nu_t`` under AD.

        ``wall_eddy_viscosity`` is the optional **wall-function** contribution: a per-face kinematic
        eddy viscosity (shape ``(n_faces,)``) that overrides the momentum wall-face diffusion
        coefficient with ``mu + rho nu_t,wall`` on the flow's shearing walls, so the wall shear
        follows the law of the wall on a mesh whose first cell is not in the viscous sublayer (see
        :meth:`_wall_boundary_viscosity`). ``None`` (default) leaves the walls resolved (the plain
        no-slip diffusion), so the momentum block is bit-identical.

        Parameters
        ----------
        eddy_viscosity : jnp.ndarray
            Per-cell **kinematic** eddy viscosity ``nu_t``, shape ``(n_cells,)``.
        wall_eddy_viscosity : jnp.ndarray or None
            Per-face wall-function **kinematic** eddy viscosity ``nu_t,wall``, shape ``(n_faces,)``
            (meaningful on shearing-wall faces, ignored elsewhere), or ``None`` for resolved walls.

        Returns
        -------
        MomentumContinuity
            A new assembler carrying ``nu_t`` (and the wall value); ``self`` is unchanged.
        """
        return eqx.tree_at(
            lambda m: (m.eddy_viscosity, m.wall_eddy_viscosity),
            self,
            (eddy_viscosity, wall_eddy_viscosity),
            is_leaf=lambda x: x is None,
        )

    def with_scaled_molecular_viscosity(self, factor: float) -> MomentumContinuity:
        """Return a copy whose molecular viscosity is multiplied by ``factor``.

        Rescales only the fluid's material ``"viscosity"`` (the dynamic ``mu``) in
        :attr:`properties`; the eddy-viscosity leaves are untouched (they are ``None`` on a freshly
        built assembler and supplied live by the closure). The seam a Reynolds-number homotopy uses to
        raise the viscosity of a lower-Re companion problem without restating the case. ``factor`` is a
        plain multiplier and a tracer flows through it under differentiation.

        Parameters
        ----------
        factor : float
            The multiplier applied to the molecular viscosity.

        Returns
        -------
        MomentumContinuity
            A new assembler at the scaled molecular viscosity; ``self`` is unchanged.
        """
        return eqx.tree_at(
            lambda m: m.properties,
            self,
            self.properties.with_scaled("viscosity", factor),
        )

    # --- properties -----------------------------------------------------------

    @property
    def viscosity(self) -> jnp.ndarray:
        """Per-cell dynamic viscosity — the momentum diffusion coefficient, shape ``(n_cells,)``.

        The molecular viscosity from :attr:`properties`, plus the turbulent contribution
        ``rho nu_t`` when a closure has supplied one via :meth:`with_eddy_viscosity`. This is the
        single place the effective viscosity ``mu_eff = mu + rho nu_t`` is formed.
        """
        molecular = self.properties.evaluate(self.mesh.cell_zones)["viscosity"]
        if self.eddy_viscosity is None:
            return molecular
        return molecular + self.density * self.eddy_viscosity

    @property
    def density(self) -> jnp.ndarray:
        """Per-cell density, shape ``(n_cells,)``."""
        return self.properties.evaluate(self.mesh.cell_zones)["density"]

    def _wall_boundary_viscosity(self) -> jnp.ndarray | None:
        """Per-face momentum diffusion coefficient carrying the wall-function override, or ``None``.

        On the flow's **shearing walls** (:meth:`~aquaflux.flow.boundary.FlowBoundary.shears_flow`) the
        wall-face effective viscosity is ``mu + rho nu_t,wall`` — the wall model's value in place of the
        owner-cell ``k/omega`` closure — so the wall shear follows the law of the wall on a mesh whose
        first cell is not in the viscous sublayer. Every other boundary face keeps the owner-cell
        ``mu_eff`` (a no-op there), and interior faces are ignored by the diffusion operator. Returns
        ``None`` when no wall eddy viscosity is set, so the momentum diffusion stays on the plain
        per-cell coefficient and is bit-identical.

        The molecular ``mu`` and ``rho`` come from this assembler, so the ``mu_eff = mu + rho nu_t``
        relation stays in one place (the closure supplies only the kinematic ``nu_t,wall``).
        """
        if self.wall_eddy_viscosity is None:
            return None
        owner = self.mesh.face_cells.owner
        molecular = self.properties.evaluate(self.mesh.cell_zones)["viscosity"]
        wall_mu = molecular[owner] + self.density[owner] * self.wall_eddy_viscosity
        shears = self.boundary.apply(
            self.mesh.face_cells,
            jnp.zeros(self.mesh.n_faces),
            lambda bc, faces, owner: jnp.full(faces.shape, float(bc.shears_flow())),
        )
        return jnp.where(shears > 0.0, wall_mu, self.viscosity[owner])

    def _boundary_face_viscosity(self) -> jnp.ndarray:
        """The per-face effective viscosity the momentum diffusion uses at each boundary face.

        The wall-function override :meth:`_wall_boundary_viscosity` where a wall eddy viscosity is set
        (``mu + rho nu_t,wall`` on the shearing walls, the owner-cell ``mu_eff`` elsewhere), else the
        owner-cell ``mu_eff`` on every face. This is exactly the coefficient
        :meth:`residual_from_fields`'s ``DiffusionFlux`` carries at the boundary (its
        ``boundary_coefficient`` is :meth:`_wall_boundary_viscosity`, which falls back to the owner
        value on non-wall faces, and its default owner-cell ``gamma`` is that same value) — so building
        the boundary ``a_P`` from it makes the diagonal match the operator at wall faces, where the
        wall model's value and the log-layer owner-cell ``k/omega`` value differ by a large factor.
        """
        wall = self._wall_boundary_viscosity()
        return wall if wall is not None else self.viscosity[self.mesh.face_cells.owner]

    # --- boundary assembly -------------------------------------------------------------

    def _boundary_fields(
        self, velocity: jnp.ndarray, pressure: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Global boundary velocity ``(n_faces, dim)`` and pressure ``(n_faces,)`` from the BCs.

        Each patch's flow closure is evaluated on its own faces and scattered into an
        otherwise-zero per-face array. In each closure ``bc`` is the patch's ``FlowBoundary``,
        ``faces`` its boundary-face indices, and ``owner`` the owner cell behind each of those faces
        (see :meth:`~aquaflux.boundary.BoundaryConditions.apply`).
        """
        face_cells = self.mesh.face_cells
        fg = self.geometry.face
        boundary_velocity = self.boundary.apply(
            face_cells,
            jnp.zeros((self.mesh.n_faces, self.mesh.dim)),
            lambda bc, faces, owner: bc.velocity_face(
                velocity[owner], fg.normal[faces], fg.centroid[faces]
            ),
        )
        boundary_pressure = self.boundary.apply(
            face_cells,
            jnp.zeros(self.mesh.n_faces),
            lambda bc, faces, owner: bc.pressure_face(pressure[owner]),
        )
        return boundary_velocity, boundary_pressure

    def _boundary_mass_flux(
        self,
        velocity: jnp.ndarray,
        pressure: jnp.ndarray,
        grad_pressure: jnp.ndarray,
        d_coeff: jnp.ndarray,
        mdot: jnp.ndarray,
    ) -> jnp.ndarray:
        """Overwrite the boundary-face entries of ``mdot`` with each patch's mass-flux closure."""
        fg = self.geometry.face
        density = self.density
        return self.boundary.apply(
            self.mesh.face_cells,
            mdot,
            lambda bc, faces, owner: bc.mass_flux(
                velocity[owner],
                pressure[owner],
                grad_pressure[owner],
                d_coeff[owner],
                fg.normal[faces],
                fg.area[faces],
                self.normal_distance[faces],
                fg.centroid[faces],
                density[owner],
            ),
        )

    def _face_pressure(
        self, pressure: jnp.ndarray, grad_pressure: jnp.ndarray, boundary_pressure: jnp.ndarray
    ) -> jnp.ndarray:
        """Face pressure for the momentum pressure force, shape ``(n_faces,)``.

        Reconstructed to the integration point on interior faces (:func:`interpolate_to_face`, so it
        carries the ``grad·(x_ip − x_g)`` skewness correction and keeps the pressure force
        second-order on non-orthogonal meshes); the boundary closure's value on boundary faces.
        """
        face_cells = self.mesh.face_cells
        interior_pressure = interpolate_to_face(
            pressure, grad_pressure, self.interp_factor, face_cells, self.geometry
        )
        return jnp.where(face_cells.interior, interior_pressure, boundary_pressure)

    # --- residual ----------------------------------------------------------------------

    def _velocity_gradient(
        self, velocity: jnp.ndarray, boundary_velocity: jnp.ndarray
    ) -> jnp.ndarray:
        """Per-cell velocity gradient tensor, shape ``(n_cells, dim, dim)`` (``[c, i, j] = d u_i/d x_j``).

        Reconstructs each component's cell gradient once — shared by the mass-flux integration-point
        reconstruction and the momentum viscous flux.
        """
        columns = [
            self.gradient_scheme.gradients(
                velocity[:, i], self.mesh, self.geometry, boundary_velocity[:, i]
            )
            for i in range(self.mesh.dim)
        ]
        return jnp.stack(columns, axis=1)

    def _mass_flux(
        self,
        velocity: jnp.ndarray,
        grad_velocity: jnp.ndarray,
        pressure: jnp.ndarray,
        grad_pressure: jnp.ndarray,
        d_coeff: jnp.ndarray,
    ) -> jnp.ndarray:
        """Rhie--Chow face mass flux over all faces (interior formula + boundary closures)."""
        face_cells = self.mesh.face_cells
        interior_flux = interior_mass_flux(
            velocity,
            grad_velocity,
            pressure,
            grad_pressure,
            d_coeff,
            face_cells,
            self.geometry,
            self.interp_factor,
            self.normal_distance,
            self.density,
        )
        mdot = face_cells.combine_face_values(interior_flux, 0.0)
        return self._boundary_mass_flux(velocity, pressure, grad_pressure, d_coeff, mdot)

    def boundary_momentum_diagonal(
        self, boundary_viscosity: jnp.ndarray, mdot: jnp.ndarray | None
    ) -> jnp.ndarray:
        """Per-face boundary owner contribution to ``a_P``, zero on interior faces, shape ``(n_faces,)``.

        Each patch's :meth:`~aquaflux.flow.boundary.FlowBoundary.momentum_diagonal_coefficient` selects
        which of the Dirichlet viscous term ``mu_f A/(d·n)`` and the upwind convective term
        ``max(mdot, 0)`` its faces contribute — a Dirichlet-velocity wall or inlet the viscous, a
        through-flow outlet the convective — so the diagonal matches the operator each patch actually
        imposes (no viscous stiffness where the velocity is zero-gradient, no convective flux through a
        wall). Passed to :func:`~aquaflux.flow.rhie_chow.momentum_diagonal` as ``boundary_owner_coeff``
        for the **residual** ``a_P`` (``momentum_matrix_diagonal``); the frozen preconditioner path uses
        ``boundary_corrected=False`` and does not call this.

        The viscous term uses the **per-face** boundary viscosity the diffusion operator carries at each
        face (:meth:`_boundary_face_viscosity`), not the owner-cell ``mu_eff``: on a wall-function mesh
        the wall model's ``mu + rho nu_t,wall`` and the log-layer owner-cell ``k/omega`` value differ by
        a large factor, so gathering the owner cell's viscosity would over-count the wall stiffness and
        leave ``a_P`` disagreeing with the operator exactly where the wall model is active.

        Parameters
        ----------
        boundary_viscosity : jnp.ndarray
            Per-face effective viscosity the diffusion operator uses at each boundary face, shape
            ``(n_faces,)`` (:meth:`_boundary_face_viscosity`; a unit field recovers the geometry-only
            ``A/(d·n)``).
        mdot : jnp.ndarray or None
            Per-face mass flux for the convective term, shape ``(n_faces,)``; ``None`` for Stokes.
        """
        fg = self.geometry.face

        def coefficient(bc, faces, owner):
            viscous_owner = boundary_viscosity[faces] * fg.area[faces] / self.normal_distance[faces]
            convective_owner = (
                jnp.maximum(mdot[faces], 0.0) if mdot is not None else jnp.zeros(faces.shape)
            )
            return bc.momentum_diagonal_coefficient(viscous_owner, convective_owner)

        return self.boundary.apply(self.mesh.face_cells, jnp.zeros(self.mesh.n_faces), coefficient)

    def momentum_matrix_diagonal(
        self,
        velocity: jnp.ndarray,
        grad_velocity: jnp.ndarray | None = None,
        *,
        boundary_corrected: bool = True,
    ) -> jnp.ndarray:
        """Momentum-matrix diagonal ``a_P`` as a differentiable function of the velocity state.

        The convective part uses a velocity-flux **estimate** for the mass flux (the interpolated
        momentum, no pressure correction) instead of the Rhie--Chow ``mdot`` itself: that estimate
        is what breaks the ``a_P`` <-> ``mdot`` circularity, and it makes ``a_P`` a genuine,
        non-circular function of the velocity. It is deliberately **not** ``stop_gradient``-ed.
        ``a_P`` enters the Rhie--Chow coefficient ``V / a_P``, whose damping term is non-zero for a
        non-linear pressure field, so the converged solution's sensitivity to ``a_P`` is real;
        freezing it would leave the implicit-function-theorem adjoint linearizing a different
        residual than the one being driven to zero (the converged *value* is unchanged, but the
        *sensitivity* is not). ``grad_velocity`` reconstructs the estimated momentum to the
        integration point (consistent with the mass flux); omit it for the cheap leading-order
        estimate.

        ``boundary_corrected`` makes each boundary face contribute only the diagonal its BC actually
        imposes (:meth:`boundary_momentum_diagonal`: a zero-gradient outlet no viscous term, a wall no
        convective term). This is the operator-consistent form the **residual** uses. The frozen
        preconditioner / continuation-shift diagonal passes ``boundary_corrected=False`` (the plain
        all-faces sum): it is a forward-path *stabilization* scale that never enters the converged
        residual or its adjoint, and keeping the extra boundary damping there is what carries the
        high-Reynolds pseudo-transient march (``test_channel_high_reynolds``). The block preconditioner
        needs ``a_P`` as a *frozen* coefficient; it ``stop_gradient``-s the result itself (and evaluates
        it at a ``stop_gradient``-ed state).
        """
        mdot_estimate = self._lagged_mdot_estimate(velocity, grad_velocity)
        boundary_owner_coeff = (
            self.boundary_momentum_diagonal(self._boundary_face_viscosity(), mdot_estimate)
            if boundary_corrected
            else None
        )
        return momentum_diagonal(
            self.mesh.face_cells,
            self.geometry,
            self.viscosity,
            mdot_lagged=mdot_estimate,
            boundary_owner_coeff=boundary_owner_coeff,
        )

    def _lagged_mdot_estimate(
        self, velocity: jnp.ndarray, grad_velocity: jnp.ndarray | None = None
    ) -> jnp.ndarray | None:
        """The lagged face mass-flux estimate ``a_P``'s convective term uses (``None`` for Stokes).

        The shared advective flux (reconstructed ``rho u`` projected on the normal), times area — the
        same face flux the Rhie--Chow mass flux uses, so the convective diagonal stays consistent with
        the ``mdot`` it stands in for, and it breaks the ``a_P`` <-> ``mdot`` circularity.
        """
        if self.advection_scheme is None:
            return None
        return (
            advective_momentum_flux(
                velocity,
                self.density,
                self.interp_factor,
                self.mesh.face_cells,
                self.geometry,
                grad_velocity,
            )
            * self.geometry.face.area
        )

    def momentum_matrix_diagonal_parts(
        self, velocity: jnp.ndarray, grad_velocity: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The convective and dissipative buckets of the all-faces ``a_P``, per cell (isotropic).

        The split :class:`~aquaflux.solve.ShiftBasis` consumes to build a local-time-step
        pseudo-transient shift: ``convective`` is the first-order-upwind outflow sum, ``dissipative``
        the viscous stiffness. Their sum is the ``boundary_corrected=False`` isotropic ``a_P`` (to
        rounding) that :meth:`momentum_matrix_diagonal` returns per component — this is the same frozen
        stabilization scale, only separated into its two parts, so it is likewise not
        operator-consistent at the boundary and never enters the residual or its adjoint.

        Parameters
        ----------
        velocity : jnp.ndarray
            Per-cell velocity, shape ``(n_cells, dim)``.
        grad_velocity : jnp.ndarray, optional
            Per-cell velocity gradient for the 2nd-order flux estimate; omit for the leading-order one.

        Returns
        -------
        tuple of jnp.ndarray
            ``(convective, dissipative)``, each shape ``(n_cells,)`` and ``>= 0``.
        """
        return momentum_diagonal_parts(
            self.mesh.face_cells,
            self.geometry,
            self.viscosity,
            mdot_lagged=self._lagged_mdot_estimate(velocity, grad_velocity),
        )

    def _momentum_residual(
        self,
        kinematic: VelocityFields,
        pressure_face: jnp.ndarray,
        mdot: jnp.ndarray,
    ) -> jnp.ndarray:
        """Momentum cell residual per velocity component, shape ``(n_cells, dim)``.

        Each component ``u_i`` is a scalar transport — viscous diffusion (viscosity as the
        coefficient) + the pressure force ``p_f n_i A`` + advection ``mdot_f u_i`` — so all three
        terms are face-flux operators composed by the same
        :class:`~aquaflux.discretization.CellBalance` that assembles every other transport
        equation; only :class:`PressureForce` is flow-specific. The per-component cell gradient is
        taken from the shared velocity-gradient reconstruction, which is why this forms its own
        context rather than letting a :class:`~aquaflux.discretization.ResidualAssembler` do it:
        the velocity gradient is one tensor reconstruction shared across the components, from flow
        boundary closures that take no gradient.

        Injected :class:`~aquaflux.flow.MomentumSource` terms are then subtracted at the **vector**
        level, once the per-component balances are stacked: a momentum source is coupled across
        components (a rotating-frame term reads the whole velocity), so it is not a per-component
        quantity and cannot ride in a balance's scalar ``source_operators``.
        """
        velocity = kinematic.velocity
        viscosity = self.viscosity  # per-cell mu, the momentum diffusion coefficient
        volume = self.geometry.cell.volume
        # The wall-function effective viscosity, if any, overrides only the shearing-wall boundary
        # faces (None -> plain per-cell coefficient, bit-identical).
        diffusion = DiffusionFlux(
            coefficient="viscosity", boundary_coefficient=self._wall_boundary_viscosity()
        )
        advection = (
            (AdvectionFlux(mass_flux=mdot, scheme=self.advection_scheme),)
            if self.advection_scheme is not None
            else ()
        )
        columns = []
        for i in range(self.mesh.dim):
            component = velocity[:, i]
            context = FaceContext(
                face_cells=self.mesh.face_cells,
                geometry=self.geometry,
                boundary_values=kinematic.boundary_velocity[:, i],
                gradient=kinematic.gradient[:, i],
                properties={"viscosity": viscosity},
            )
            # A balance sums its operators in tuple order, and floating-point addition is not
            # associative, so viscous-pressure-advective is the arithmetic, not just the reading
            # order. Rearranging them perturbs the residual in the last bits.
            balance = CellBalance((diffusion, PressureForce(pressure_face, i), *advection))
            # R = balance - source; the body force is a uniform volume source.
            columns.append(balance.residual(component, context) - self.body_force[i] * volume)
        residual = jnp.stack(columns, axis=1)
        # Each injected source returns its cell integral (production positive), so it leaves the
        # balance as a sink -- the vector counterpart of a CellBalance's scalar source loop.
        if self.sources:
            properties = self.properties.evaluate(self.mesh.cell_zones)
            for source in self.sources:
                residual = residual - source.source(kinematic, self.geometry, properties)
        return residual

    def _continuity_residual(self, mdot: jnp.ndarray, pressure: jnp.ndarray) -> jnp.ndarray:
        """Continuity cell residual: the net Rhie--Chow mass flux ``Σ mdot_f``, shape ``(n_cells,)``.

        In a closed domain (``pressure_pin`` set) the pinned cell's continuity equation is replaced
        by ``p = pressure_pin_value`` to fix the otherwise-free pressure level.
        """
        residual = self.mesh.face_cells.scatter_conservative(mdot)
        if self.pressure_pin is not None:
            fix = FixedValueCells(
                jnp.array([self.pressure_pin]), jnp.array([self.pressure_pin_value])
            )
            residual = fix.apply(residual, pressure)
        return residual

    def flow_fields(self, state: jnp.ndarray) -> FlowFields:
        """Assemble the shared flow quantities for ``state`` (boundary fields, gradients, ``mdot``).

        The one place the velocity/pressure gradients and the Rhie--Chow mass flux are formed. A
        consumer that needs more than one of them at a single state -- the residual and ``mdot`` in a
        coupled solve, the velocity gradient and ``mdot`` in a segregated sweep -- calls this **once**
        and reads the fields it needs (via :meth:`residual_from_fields`, :attr:`FlowFields.mdot`,
        :attr:`FlowFields.velocity_fields`), so the boundary fields, gradients, ``a_P``, and
        Rhie--Chow flux are assembled a single time instead of once per accessor.

        Parameters
        ----------
        state : jnp.ndarray
            The flat flow state ``[vel_0..vel_{dim-1}, pressure]``, shape ``((dim + 1) n_cells,)``.
        """
        velocity, pressure = self.unpack(state)
        boundary_velocity, boundary_pressure = self._boundary_fields(velocity, pressure)
        grad_velocity = self._velocity_gradient(velocity, boundary_velocity)

        # Rhie--Chow coupling: the pressure gradient, the momentum diagonal a_P, and the mass flux
        # mdot that couples pressure implicitly into both continuity and advection.
        grad_pressure = self.gradient_scheme.gradients(
            pressure, self.mesh, self.geometry, boundary_pressure
        )
        a_p = self.momentum_matrix_diagonal(
            velocity, grad_velocity
        )  # (n_cells, dim), per component; differentiable (see the method docstring)
        d_coeff = self.geometry.cell.volume[:, None] / a_p  # Rhie--Chow coefficient V / a_P
        mdot = self._mass_flux(velocity, grad_velocity, pressure, grad_pressure, d_coeff)
        return FlowFields(
            VelocityFields(velocity, boundary_velocity, grad_velocity),
            pressure,
            boundary_pressure,
            grad_pressure,
            mdot,
        )

    def residual_from_fields(self, fields: FlowFields) -> jnp.ndarray:
        """Coupled momentum + continuity residual from a pre-assembled :class:`FlowFields` bundle.

        The residual assembly given the shared quantities, split from :meth:`flow_fields` so a caller
        that also needs ``mdot`` (a coupled RANS residual) assembles the flow fields once and reuses
        them for both. Same shape as the state.

        Parameters
        ----------
        fields : FlowFields
            The bundle from :meth:`flow_fields` at the state whose residual is wanted.
        """
        pressure_face = self._face_pressure(
            fields.pressure, fields.grad_pressure, fields.boundary_pressure
        )
        velocity_residual = self._momentum_residual(
            fields.velocity_fields, pressure_face, fields.mdot
        )
        pressure_residual = self._continuity_residual(fields.mdot, fields.pressure)
        return self.pack(velocity_residual, pressure_residual)

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        """Coupled momentum + continuity residual for the flat state, same shape as ``state``."""
        return self.residual_from_fields(self.flow_fields(state))

    def mass_flux(self, state: jnp.ndarray) -> jnp.ndarray:
        """The Rhie--Chow face mass flux ``mdot`` for ``state``, shape ``(n_faces,)``.

        This is the *same* face flux that closes continuity, so a scalar transported by this flow
        (a turbulence field, a species) must advect on it -- rebuilding ``(u . n) A`` from the cell
        velocities is non-conservative and violates discrete continuity. The coupling seam for
        scalar transport: evaluate on the converged flow ``state``. Needs the full Rhie--Chow
        assembly; a caller that also wants the residual or the velocity gradient at this state should
        call :meth:`flow_fields` once instead.

        Parameters
        ----------
        state : jnp.ndarray
            The flat flow state ``[vel_0..vel_{dim-1}, pressure]``, shape ``((dim + 1) n_cells,)``.
        """
        return self.flow_fields(state).mdot

    def velocity_fields(self, state: jnp.ndarray) -> VelocityFields:
        """The kinematic velocity bundle for ``state`` (cell, boundary-face, and gradient).

        The velocity-gradient tensor's symmetric part is the mean strain rate a turbulence model
        consumes, and the cell/boundary velocity pair is what a near-wall shear rate is measured
        from. Evaluate on the converged flow ``state``.

        Reconstructed directly from the boundary velocity and the shared per-component gradient
        (:meth:`_velocity_gradient`) -- the same formula :meth:`flow_fields` uses -- **without** the
        Rhie--Chow ``a_P`` / ``mdot`` work, since only the kinematic half is wanted here (a segregated
        sweep needs ``nu_t`` before the mass flux is even defined). A caller that also needs ``mdot``
        at this state should call :meth:`flow_fields` once and read
        :attr:`FlowFields.velocity_fields`.

        Parameters
        ----------
        state : jnp.ndarray
            The flat flow state ``[vel_0..vel_{dim-1}, pressure]``, shape ``((dim + 1) n_cells,)``.
        """
        velocity, pressure = self.unpack(state)
        boundary_velocity, _ = self._boundary_fields(velocity, pressure)
        return VelocityFields(
            velocity, boundary_velocity, self._velocity_gradient(velocity, boundary_velocity)
        )
