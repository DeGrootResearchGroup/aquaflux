"""The coupled momentum--continuity residual for incompressible flow.

Assembles one residual over the whole flow state ``(u, v[, w], p)`` in a system-first
design. The unknowns are stored as a single flat vector with the layout
``[vel_0, vel_1, ..., vel_{dim-1}, pressure]`` (each block ``n_cells`` long), so the coupled
system is solved by the same Newton / implicit-diff machinery as a scalar field.

Per velocity component ``i`` the momentum balance is a scalar transport of ``u_i``:

    R_{u_i} = sum_faces ( mdot_f u_{i,f}  +  p_f n_i A  -  mu (grad u_i . n) A )   ( + accumulation )

— advection of ``u_i`` by the mass flux, a pressure-gradient force, and viscous diffusion. The
first and last reuse :class:`~aquaflux.discretization.AdvectionFlux` and
:class:`~aquaflux.discretization.DiffusionFlux` verbatim (viscosity as the diffusion
coefficient); only the pressure term is new. Continuity is

    R_p = sum_faces mdot_f ,

with ``mdot_f`` the Rhie--Chow mass flux, which couples pressure implicitly and prevents
checkerboarding. The mass flux and the lagged momentum diagonal ``a_P`` come from
:mod:`aquaflux.flow.rhie_chow`; the Jacobian of the whole coupled residual comes from AD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import AdvectionFlux, DiffusionFlux, FaceContext
from aquaflux.schemes.interpolation import (
    interpolate_owner_neighbour,
    interpolate_to_face,
    interpolation_factor,
)
from aquaflux.vectors import dot, scale

from .rhie_chow import interior_mass_flux, momentum_diagonal
from .state import BlockStateLayout

if TYPE_CHECKING:
    from aquaflux.discretization import AdvectionScheme
    from aquaflux.materials import MaterialModel
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.schemes import GradientScheme


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
    materials : MaterialModel
        The fluid properties, supplying per-cell ``"viscosity"`` (the momentum diffusion
        coefficient) and ``"density"`` (:attr:`viscosity` / :attr:`density` evaluate them).
    gradient_scheme : GradientScheme
        Reconstruction for the velocity and pressure gradients.
    advection_scheme : AdvectionScheme or None
        Momentum convection scheme; ``None`` gives Stokes flow (no convection). A limited scheme
        (``LimitedUpwind``) carries its own slope limiter.
    boundary : BoundaryConditions
        The named per-patch flow closures, resolved to their boundary-face indices.
    interp_factor, normal_distance : jnp.ndarray
        Face interpolation factor ``g`` and normal distance ``d . n`` (precomputed geometry).
    """

    mesh: Mesh
    geometry: MeshGeometry
    materials: MaterialModel
    gradient_scheme: GradientScheme
    advection_scheme: AdvectionScheme | None
    boundary: BoundaryConditions
    interp_factor: jnp.ndarray
    normal_distance: jnp.ndarray
    pressure_pin: int | None = eqx.field(static=True)
    pressure_pin_value: float

    @classmethod
    def build(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        materials: MaterialModel,
        gradient_scheme: GradientScheme,
        boundary: BoundaryConditions,
        *,
        advection_scheme: AdvectionScheme | None = None,
        pressure_pin: int | None = None,
        pressure_pin_value: float = 0.0,
    ) -> MomentumContinuity:
        """Build the coupled assembler, precomputing face interpolation geometry.

        ``boundary`` is a :class:`~aquaflux.boundary.BoundaryConditions` collection of per-patch
        flow closures (``BoundaryConditions({name: FlowBoundary})``), bound to ``mesh.face_patches``
        internally. ``materials`` must supply ``"viscosity"`` and ``"density"``. ``pressure_pin``
        fixes the pressure at one cell (its continuity equation is replaced by
        ``p = pressure_pin_value``) — required for a closed domain (all-wall, no pressure outlet),
        where pressure is otherwise defined only up to a constant.
        """
        materials.require("viscosity", "density")
        face_geometry, cell_geometry = geometry.face, geometry.cell
        face_cells = mesh.face_cells
        owner = face_cells.owner
        nb = face_cells.safe_neighbour
        interior = face_cells.interior
        x_p = cell_geometry.centroid[owner]
        x_ip = face_geometry.centroid
        d = cell_geometry.centroid[nb] - x_p
        interp_factor = interpolation_factor(face_cells, geometry)
        normal_distance = jnp.where(
            interior,
            dot(d, face_geometry.normal),
            dot(x_ip - x_p, face_geometry.normal),
        )
        return cls(
            mesh=mesh,
            geometry=geometry,
            materials=materials,
            gradient_scheme=gradient_scheme,
            advection_scheme=advection_scheme,
            boundary=boundary.resolve(mesh.face_patches),
            interp_factor=interp_factor,
            normal_distance=normal_distance,
            pressure_pin=pressure_pin,
            pressure_pin_value=pressure_pin_value,
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

    # --- material properties -----------------------------------------------------------

    @property
    def viscosity(self) -> jnp.ndarray:
        """Per-cell dynamic viscosity — the momentum diffusion coefficient, shape ``(n_cells,)``."""
        return self.materials.evaluate(self.mesh.cell_zones)["viscosity"]

    @property
    def density(self) -> jnp.ndarray:
        """Per-cell density, shape ``(n_cells,)``."""
        return self.materials.evaluate(self.mesh.cell_zones)["density"]

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

    def _scatter(self, face_flux: jnp.ndarray) -> jnp.ndarray:
        return self.mesh.face_cells.scatter_conservative(face_flux)

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

    def lagged_momentum_diagonal(
        self, velocity: jnp.ndarray, grad_velocity: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Lagged momentum-matrix diagonal ``a_P`` (``stop_gradient``-ed).

        Its convective part uses a velocity-flux estimate (no pressure correction) to avoid the
        ``a_P`` <-> ``mdot`` circularity. Shared by the residual (Rhie--Chow coefficient) and the
        block preconditioner (velocity block). ``grad_velocity`` reconstructs the lagged momentum to
        the integration point (consistent with the mass flux); omit it for the cheap leading-order
        estimate the preconditioner uses (``a_P`` is a lagged coefficient, so either is exact at
        convergence).
        """
        mdot_estimate = None
        if self.advection_scheme is not None:
            # lagged momentum flux estimate: reconstruct rho*u to the face (density rides with velocity)
            momentum = scale(velocity, self.density)
            if grad_velocity is None:
                momentum_face = interpolate_owner_neighbour(
                    momentum, self.interp_factor, self.mesh.face_cells
                )
            else:
                grad_momentum = (
                    self.density[:, None, None] * grad_velocity
                )  # grad(rho*u) = rho grad(u)
                momentum_face = interpolate_to_face(
                    momentum, grad_momentum, self.interp_factor, self.mesh.face_cells, self.geometry
                )
            mdot_estimate = dot(momentum_face, self.geometry.face.normal) * self.geometry.face.area
        return jax.lax.stop_gradient(
            momentum_diagonal(
                self.mesh.face_cells,
                self.geometry,
                self.viscosity,
                self.normal_distance,
                self.interp_factor,
                mdot_lagged=mdot_estimate,
            )
        )

    def _momentum_residual(
        self,
        velocity: jnp.ndarray,
        grad_velocity: jnp.ndarray,
        boundary_velocity: jnp.ndarray,
        pressure_face: jnp.ndarray,
        mdot: jnp.ndarray,
    ) -> jnp.ndarray:
        """Momentum cell residual per velocity component, shape ``(n_cells, dim)``.

        Each component ``u_i`` is a scalar transport — viscous diffusion (viscosity as the
        coefficient) + the pressure force ``p_f n_i A`` + advection ``mdot_f u_i`` — reusing the
        shared diffusion and advection operators; only the pressure force is flow-specific. The
        per-component cell gradient is taken from the shared ``grad_velocity`` reconstruction.
        """
        viscosity = self.viscosity  # per-cell mu, the momentum diffusion coefficient
        normal, area = self.geometry.face.normal, self.geometry.face.area
        diffusion = DiffusionFlux(coefficient="viscosity")
        advection = (
            AdvectionFlux(mass_flux=mdot, scheme=self.advection_scheme)
            if self.advection_scheme is not None
            else None
        )
        columns = []
        for i in range(self.mesh.dim):
            component = velocity[:, i]
            context = FaceContext(
                face_cells=self.mesh.face_cells,
                geometry=self.geometry,
                boundary_values=boundary_velocity[:, i],
                gradient=grad_velocity[:, i],
                materials={"viscosity": viscosity},
            )
            face_flux = (
                diffusion.face_flux(component, context) + pressure_face * normal[:, i] * area
            )
            if advection is not None:
                face_flux = face_flux + advection.face_flux(component, context)
            columns.append(self._scatter(face_flux))
        return jnp.stack(columns, axis=1)

    def _continuity_residual(self, mdot: jnp.ndarray, pressure: jnp.ndarray) -> jnp.ndarray:
        """Continuity cell residual: the net Rhie--Chow mass flux ``Σ mdot_f``, shape ``(n_cells,)``.

        In a closed domain (``pressure_pin`` set) the pinned cell's continuity equation is replaced
        by ``p = pressure_pin_value`` to fix the otherwise-free pressure level.
        """
        residual = self._scatter(mdot)
        if self.pressure_pin is not None:
            residual = residual.at[self.pressure_pin].set(
                pressure[self.pressure_pin] - self.pressure_pin_value
            )
        return residual

    def residual(self, state: jnp.ndarray) -> jnp.ndarray:
        """Coupled momentum + continuity residual for the flat state, same shape as ``state``."""
        velocity, pressure = self.unpack(state)
        boundary_velocity, boundary_pressure = self._boundary_fields(velocity, pressure)
        grad_velocity = self._velocity_gradient(velocity, boundary_velocity)

        # Rhie--Chow coupling: the pressure gradient, the lagged momentum diagonal a_P, and the mass
        # flux mdot that couples pressure implicitly into both continuity and advection.
        grad_pressure = self.gradient_scheme.gradients(
            pressure, self.mesh, self.geometry, boundary_pressure
        )
        a_p = self.lagged_momentum_diagonal(
            velocity, grad_velocity
        )  # (n_cells, dim), per component
        d_coeff = self.geometry.cell.volume[:, None] / a_p  # Rhie--Chow coefficient V / a_P
        mdot = self._mass_flux(velocity, grad_velocity, pressure, grad_pressure, d_coeff)

        pressure_face = self._face_pressure(pressure, grad_pressure, boundary_pressure)
        velocity_residual = self._momentum_residual(
            velocity, grad_velocity, boundary_velocity, pressure_face, mdot
        )
        pressure_residual = self._continuity_residual(mdot, pressure)
        return self.pack(velocity_residual, pressure_residual)
