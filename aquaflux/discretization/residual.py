"""The residual assembler: gather -> compute -> scatter of the cell residual ``R(phi)``.

Everything the solver needs reduces to one discrete residual per cell — the finite-volume
conservation balance

    R_P = accumulation_P + sum_faces(owner-outward flux) - sum_sources(cell integral of S dV),

which vanishes at the converged solution. Each flux operator returns the owner-outward flux of the
conserved quantity (advection ``+ mdot phi``, diffusion ``- Gamma grad phi . n A``); each volume
source returns its cell integral (production positive), which leaves the balance as a sink.
:class:`ResidualAssembler` builds it by

1. reconstructing cell gradients once (the injected :class:`GradientScheme`, if any) so
   every flux operator shares a single gradient field;
2. evaluating the weak boundary face values from the per-patch
   :class:`~aquaflux.boundary.conditions.BoundaryCondition` closures;
3. gathering owner/neighbour state onto faces, calling each injected
   :class:`~aquaflux.discretization.diffusion.FaceFluxOperator`, and scattering the
   owner-outward face flux back to cells with ``segment_sum`` (owner ``+``, neighbour
   ``-``; boundary faces to the owner only);
4. subtracting each injected volume source (its cell integral, from a
   :class:`~aquaflux.discretization.source.VolumeSource`);
5. adding the injected transient (accumulation) term.

The Jacobian and adjoint are never assembled here — they come from automatic
differentiation of this ``R``. No hand-derived linearization coefficients live in this
module; it only *composes* geometry (from ``mesh``), schemes (from ``schemes``), operators,
and boundary closures.

Boundary patches are resolved to concrete face-index arrays once, ahead of the solve (via
:meth:`aquaflux.boundary.BoundaryConditions.resolve`), because the label-to-index lookup is
data-dependent and cannot run under ``jit``. The resolved index arrays are then constant inputs
to the differentiable residual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.boundary import BoundaryConditions

from .face_flux import FaceContext

if TYPE_CHECKING:
    from aquaflux.materials import MaterialModel
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.schemes import GradientScheme

    from .face_flux import FaceFluxOperator
    from .source import VolumeSource
    from .transient import TransientTerm


class ResidualAssembler(eqx.Module):
    """Assemble the cell residual ``R(phi)`` from injected operators, schemes, and closures.

    Construct with :meth:`build` (it binds the injected
    :class:`~aquaflux.boundary.BoundaryConditions` to the mesh's face patches). The module is
    an ``equinox.Module`` pytree, so differentiating a converged solve with respect
    to a boundary parameter (e.g. the Biot number held inside a
    :class:`~aquaflux.boundary.conditions.Convective` closure) is differentiation with
    respect to a leaf of this tree.

    Attributes
    ----------
    mesh : Mesh
        Topology (owner/neighbour connectivity, patch labels).
    geometry : MeshGeometry
        Face and cell metrics — areas, owner-outward normals, centroids, volumes (computed once,
        shared).
    materials : MaterialModel
        The named per-cell physical properties, evaluated each residual and threaded to the flux
        operators (via the context) and the boundary closures.
    flux_operators : tuple of FaceFluxOperator
        Face-flux operators summed into the transport term.
    source_operators : tuple of VolumeSource
        Volume-source operators subtracted from the balance (each returns its cell integral,
        production positive); empty for a flux-only equation.
    transient : TransientTerm or None
        Accumulation term; ``None`` for a steady residual.
    gradient_scheme : GradientScheme or None
        Cell-gradient reconstruction shared by the flux operators' non-orthogonal
        corrections. ``None`` reconstructs no gradient (exact on orthogonal grids, where the
        correction vanishes identically).
    coefficient : str
        The material property the flux-type boundary closures (Robin/Neumann) use as their
        diffusion coefficient ``Gamma`` (static; matches the ``DiffusionFlux.coefficient`` of the
        equation's diffusion term).
    boundary : BoundaryConditions
        The named per-patch closures, resolved to their boundary-face indices.
    """

    mesh: Mesh
    geometry: MeshGeometry
    materials: MaterialModel
    flux_operators: tuple[FaceFluxOperator, ...]
    source_operators: tuple[VolumeSource, ...]
    transient: TransientTerm | None
    gradient_scheme: GradientScheme | None
    coefficient: str = eqx.field(static=True)
    boundary: BoundaryConditions

    @classmethod
    def build(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        materials: MaterialModel,
        flux_operators: tuple[FaceFluxOperator, ...],
        boundary: BoundaryConditions,
        *,
        coefficient: str = "diffusivity",
        transient: TransientTerm | None = None,
        source_operators: tuple[VolumeSource, ...] = (),
        gradient_scheme: GradientScheme | None = None,
    ) -> ResidualAssembler:
        """Build an assembler from injected operators, schemes, and boundary closures.

        Parameters
        ----------
        mesh : Mesh
            The mesh; its ``face_patches`` name the boundary faces.
        geometry : MeshGeometry
            Geometry from ``mesh.geometry()``.
        materials : MaterialModel
            The named per-cell physical properties; the diffusion term reads its coefficient by
            name, and the flux-type boundary closures read ``coefficient``.
        flux_operators : tuple of FaceFluxOperator
            Face-flux operators (e.g. one :class:`DiffusionFlux`).
        boundary : BoundaryConditions
            The named ``{patch: closure}`` collection (``BoundaryConditions({name: bc})``), bound to
            ``mesh.face_patches`` internally. Every boundary face must lie in a named patch present
            here, or its flux reads an unset (zero) face value.
        coefficient : str
            The property the flux-type boundary closures use as their diffusion coefficient
            (default ``"diffusivity"``; match the equation's ``DiffusionFlux.coefficient``).
        transient : TransientTerm, optional
            Accumulation term; omit for a steady residual.
        source_operators : tuple of VolumeSource, optional
            Volume-source terms subtracted from the balance (default none); each returns its cell
            integral, production positive.
        gradient_scheme : GradientScheme, optional
            Cell-gradient reconstruction for the non-orthogonal corrections; omit on
            orthogonal grids.
        """
        return cls(
            mesh=mesh,
            geometry=geometry,
            materials=materials,
            flux_operators=flux_operators,
            source_operators=source_operators,
            transient=transient,
            gradient_scheme=gradient_scheme,
            coefficient=coefficient,
            boundary=boundary.resolve(mesh.face_patches),
        )

    def boundary_values(
        self, phi: jnp.ndarray, gradient: jnp.ndarray, materials: dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Weak boundary face values ``phi_ip`` for every face, shape ``(n_faces,)``.

        Interior faces keep their zero placeholder (the flux operator ignores them). Each
        named patch's closure is evaluated on its faces and scattered into place.

        Parameters
        ----------
        phi : jnp.ndarray
            Cell field, shape ``(n_cells,)``.
        gradient : jnp.ndarray
            Cell gradients, shape ``(n_cells, dim)`` (used by the closures' corrections).
        materials : dict of {str: jnp.ndarray}
            The evaluated per-cell properties; the flux-type closures read
            ``materials[self.coefficient]`` as ``Gamma``.
        """
        centroid = self.geometry.cell.centroid
        # The diffusion coefficient for the flux-type (Robin/Neumann) closures; zeros when the model
        # has no such property (a pure-advection problem with only value/inflow BCs ignores it).
        gamma = materials.get(self.coefficient, jnp.zeros(self.mesh.n_cells, dtype=phi.dtype))

        def closure(bc, faces, owner):
            face_centroid = self.geometry.face.centroid[faces]
            d = face_centroid - centroid[owner]
            return bc.face_value(
                phi[owner],
                gradient[owner],
                d,
                self.geometry.face.normal[faces],
                gamma[owner],
                face_centroid,
            )

        return self.boundary.apply(
            self.mesh.face_cells,
            jnp.zeros(self.mesh.n_faces, dtype=phi.dtype),
            closure,
        )

    def _context(
        self,
        gradient: jnp.ndarray,
        boundary_values: jnp.ndarray,
        materials: dict[str, jnp.ndarray],
    ) -> FaceContext:
        """The shared per-face inputs each flux operator gathers from."""
        return FaceContext(
            face_cells=self.mesh.face_cells,
            geometry=self.geometry,
            boundary_values=boundary_values,
            gradient=gradient,
            materials=materials,
        )

    def _scatter(self, face_flux: jnp.ndarray) -> jnp.ndarray:
        """Scatter owner-outward face flux to cells (owner ``+``, interior neighbour ``-``)."""
        return self.mesh.face_cells.scatter_conservative(face_flux)

    def _gradient(
        self, phi: jnp.ndarray, materials: dict[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Cell gradients and the boundary values consistent with them.

        With no gradient scheme the gradient is zero (exact on orthogonal grids, where the
        non-orthogonal correction vanishes). Otherwise the reconstruction is fed a
        leading-order boundary value (its own tangential correction dropped, i.e. evaluated
        at zero gradient) to keep ``R`` a single-pass function of ``phi``; the flux then uses
        the full boundary value evaluated at the reconstructed gradient. The two agree
        exactly on orthogonal grids.
        """
        dim = self.mesh.dim
        n_cells = self.mesh.n_cells
        if self.gradient_scheme is None:
            gradient = jnp.zeros((n_cells, dim), dtype=phi.dtype)
            return gradient, self.boundary_values(phi, gradient, materials)
        zero_grad = jnp.zeros((n_cells, dim), dtype=phi.dtype)
        leading_bvals = self.boundary_values(phi, zero_grad, materials)
        gradient = self.gradient_scheme.gradients(phi, self.mesh, self.geometry, leading_bvals)
        return gradient, self.boundary_values(phi, gradient, materials)

    def gradient(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Reconstructed cell gradients of ``phi``, shape ``(n_cells, dim)``.

        The post-processing accessor for the injected gradient scheme — e.g. to form the
        diffusive flux ``-gamma * gradient`` of a converged field. Returns zeros when no
        gradient scheme is injected (orthogonal grids, where the correction vanishes). Its
        accuracy on skewed grids is the scheme's: ``CorrectedGreenGauss`` caps near first
        order, ``HessianCorrectedGradient`` restores second order.

        Parameters
        ----------
        phi : jnp.ndarray
            Cell field, shape ``(n_cells,)``.
        """
        materials = self.materials.evaluate(self.mesh.cell_zones)
        return self._gradient(phi, materials)[0]

    def residual(
        self,
        phi: jnp.ndarray,
        phi_old: jnp.ndarray | None = None,
        phi_older: jnp.ndarray | None = None,
        dt: float | None = None,
        first_step: bool = False,
    ) -> jnp.ndarray:
        """Cell residual ``R(phi)``, shape ``(n_cells,)``.

        Parameters
        ----------
        phi : jnp.ndarray
            Current cell field, shape ``(n_cells,)``.
        phi_old, phi_older : jnp.ndarray, optional
            Previous / second-previous time levels for the transient term (required when a
            :class:`TransientTerm` was injected).
        dt : float, optional
            Timestep (required with a transient term).
        first_step : bool
            ``True`` on the first timestep (BDF1); static.

        Returns
        -------
        jnp.ndarray
            The residual ``accumulation + net outward flux - volume sources``, shape
            ``(n_cells,)``.
        """
        materials = self.materials.evaluate(self.mesh.cell_zones)
        gradient, boundary_values = self._gradient(phi, materials)
        context = self._context(gradient, boundary_values, materials)

        face_flux = jnp.zeros(self.mesh.n_faces, dtype=phi.dtype)
        for operator in self.flux_operators:
            face_flux = face_flux + operator.face_flux(phi, context)
        # Each operator returns the owner-outward flux of the conserved quantity; the residual
        # is the finite-volume balance accumulation + sum of net outward face fluxes.
        residual = self._scatter(face_flux)
        # A volume source is produced inside the cell, so it leaves the balance as a sink: each
        # returns its cell integral (production positive) and is subtracted from the residual.
        for operator in self.source_operators:
            residual = residual - operator.source(phi, context)
        if self.transient is not None:
            residual = residual + self.transient.residual(
                phi, phi_old, phi_older, dt, first_step, self.geometry.cell.volume
            )
        return residual
