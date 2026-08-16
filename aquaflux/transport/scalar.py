"""A scalar transported by a converged flow: concentration, temperature, any passive tracer.

The equation is the finite-volume balance of a per-unit-volume quantity carried by the flow,

    dC/dt + div(u C) = div(Gamma grad C) + S,

which for a species concentration (``kg/m^3``, ``mol/m^3``) is a **mass balance on the species**:
no fluid density appears in it, and it is already in conservative form as written. It is assembled
from exactly the operators every other transport equation uses -- :class:`AdvectionFlux` on the
flow's volumetric face flux, :class:`DiffusionFlux` on an effective diffusivity, any
:class:`VolumeSource` terms, and an optional :class:`TransientTerm`. :class:`ScalarTransport` is
the composition of them, so a caller states the physics rather than the assembly.

**Advect on the flow's own face flux, never on a rebuilt one.** The flux must come from the
Rhie--Chow assembly (:meth:`~aquaflux.flow.MomentumContinuity.mass_flux`, converted by
:func:`~aquaflux.flow.volume_flux`), because that is the flux continuity closes on. Rebuilding
``(u . n) A`` from cell velocities satisfies no discrete continuity, so a uniform tracer would not
stay uniform and the transported scalar would not be conservative with the flow carrying it.

**The flux is a per-state input, not configuration.** :meth:`ScalarTransport.residual` takes it and
returns the residual function for that flow, mirroring how the turbulence transport equations are
built per outer sweep -- so one configured :class:`ScalarTransport` serves a frozen flow, a
sequence of flows, or a coupled solve without being restated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.discretization import (
    AdvectionFlux,
    DiffusionFlux,
    ResidualAssembler,
)
from aquaflux.properties import FieldProperty, PropertyModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from aquaflux.boundary import BoundaryConditions
    from aquaflux.discretization import AdvectionScheme, TransientTerm, VolumeSource
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.schemes import GradientScheme

#: The property name the transported scalar's diffusion coefficient is registered under. It matches
#: :class:`~aquaflux.discretization.DiffusionFlux`'s own default, so the flux-type boundary closures
#: (Robin/Neumann) read the same coefficient the interior flux does.
DIFFUSIVITY = "diffusivity"


def effective_diffusivity(
    molecular: jnp.ndarray,
    eddy_viscosity: jnp.ndarray | None = None,
    turbulent_number: float = 0.7,
) -> FieldProperty:
    """Effective diffusivity ``Gamma = D + nu_t / turbulent_number``, as a per-cell property.

    The turbulent transport of a passive scalar is modelled by dividing the eddy viscosity by a
    **turbulent Schmidt number** (mass) or **turbulent Prandtl number** (heat) -- a single modelling
    constant of order one, on the argument that momentum and the scalar are mixed by the same
    eddies. Both are the same relation and share this one definition.

    Note this is deliberately *not* shared with the ``k``/``omega`` equations' diffusivity, which is
    ``nu + blend(F1, sigma_1, sigma_2) nu_t``: that coefficient is an ``F1``-blended model constant
    of the closure, not a turbulent number, so the two agree only in having the shape
    ``molecular + coefficient * nu_t``. Unifying them would take "the coefficient multiplying
    ``nu_t``" as an argument, which removes no decision from either caller.

    Parameters
    ----------
    molecular : jnp.ndarray
        Molecular diffusivity ``D`` per cell, shape ``(n_cells,)`` (kinematic, ``m^2/s``).
    eddy_viscosity : jnp.ndarray or None
        Kinematic eddy viscosity ``nu_t`` per cell, shape ``(n_cells,)``, from a turbulence closure.
        ``None`` (default) gives the laminar diffusivity unchanged.
    turbulent_number : float
        The turbulent Schmidt or Prandtl number. Default ``0.7``, the usual value for a passive
        scalar in a turbulent shear flow. It is a **modelling choice**: a result sensitive to it
        should say which value it was taken at.

    Returns
    -------
    FieldProperty
        The per-cell effective diffusivity, a differentiable leaf.
    """
    if eddy_viscosity is None:
        return FieldProperty(values=molecular)
    return FieldProperty(values=molecular + eddy_viscosity / turbulent_number)


class ScalarTransport(eqx.Module):
    """A configured scalar transport equation, evaluated on whatever flow flux it is handed.

    Construct with :meth:`build`; call :meth:`residual` with the flow's volumetric face flux to get
    the residual function of that flow. The configuration -- mesh, schemes, boundary closures,
    sources -- is fixed; the flux and the diffusivity are what a developing flow changes.

    Attributes
    ----------
    mesh : Mesh
        Topology (owner/neighbour connectivity, patch labels).
    geometry : MeshGeometry
        Face and cell metrics.
    diffusivity : FieldProperty
        The effective diffusivity ``Gamma`` per cell (see :func:`effective_diffusivity`).
    boundary : BoundaryConditions
        The named per-patch scalar closures. A sub-patch injection is a
        :class:`~aquaflux.boundary.DirichletField` on the inlet patch, whose value is a function of
        the face centroid -- so an injector covering part of a patch needs no separate patch, and
        therefore no change to the mesh.
    advection_scheme : AdvectionScheme
        The face-value reconstruction for the advective flux.
    gradient_scheme : GradientScheme or None
        Cell-gradient reconstruction for the non-orthogonal diffusion correction; ``None`` on
        orthogonal grids, where the correction vanishes.
    sources : tuple of VolumeSource
        Volume-source terms subtracted from the balance -- where a reaction attaches.
    transient : TransientTerm or None
        Accumulation term; ``None`` for a steady scalar.
    """

    mesh: Mesh
    geometry: MeshGeometry
    diffusivity: FieldProperty
    boundary: BoundaryConditions
    advection_scheme: AdvectionScheme
    gradient_scheme: GradientScheme | None
    sources: tuple[VolumeSource, ...]
    transient: TransientTerm | None

    @classmethod
    def build(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        diffusivity: FieldProperty,
        boundary: BoundaryConditions,
        advection_scheme: AdvectionScheme,
        *,
        gradient_scheme: GradientScheme | None = None,
        sources: tuple[VolumeSource, ...] = (),
        transient: TransientTerm | None = None,
    ) -> ScalarTransport:
        """Configure the equation, binding the boundary closures to the mesh's face patches.

        Parameters
        ----------
        mesh, geometry : Mesh, MeshGeometry
            The mesh and its metrics.
        diffusivity : FieldProperty
            The effective diffusivity per cell; :func:`effective_diffusivity` forms the usual one.
        boundary : BoundaryConditions
            The named ``{patch: closure}`` collection, bound to ``mesh.face_patches`` internally.
        advection_scheme : AdvectionScheme
            The face-value reconstruction for advection.
        gradient_scheme : GradientScheme, optional
            Reconstruction for the non-orthogonal correction; omit on orthogonal grids.
        sources : tuple of VolumeSource, optional
            Volume-source terms (default none).
        transient : TransientTerm, optional
            Accumulation term; omit for a steady scalar.
        """
        return cls(
            mesh=mesh,
            geometry=geometry,
            diffusivity=diffusivity,
            boundary=boundary.resolve(mesh.face_patches),
            advection_scheme=advection_scheme,
            gradient_scheme=gradient_scheme,
            sources=sources,
            transient=transient,
        )

    def with_diffusivity(self, diffusivity: FieldProperty) -> ScalarTransport:
        """Return a copy carrying a new effective diffusivity; ``self`` is unchanged.

        The seam a segregated loop refreshes ``Gamma`` through as the eddy viscosity develops,
        without restating the equation.
        """
        return eqx.tree_at(lambda t: t.diffusivity, self, diffusivity)

    def assembler(self, flux: jnp.ndarray) -> ResidualAssembler:
        """The residual assembler for a flow whose volumetric face flux is ``flux``.

        Parameters
        ----------
        flux : jnp.ndarray
            Owner-outward **volumetric** face flux ``Q_f``, shape ``(n_faces,)`` -- the flow's
            Rhie--Chow mass flux through :func:`~aquaflux.flow.volume_flux`, never a flux rebuilt
            from cell velocities (see the module docstring).
        """
        return ResidualAssembler.build(
            self.mesh,
            self.geometry,
            PropertyModel({DIFFUSIVITY: self.diffusivity}),
            (
                AdvectionFlux(mass_flux=flux, scheme=self.advection_scheme),
                DiffusionFlux(coefficient=DIFFUSIVITY),
            ),
            self.boundary,
            coefficient=DIFFUSIVITY,
            source_operators=self.sources,
            gradient_scheme=self.gradient_scheme,
            transient=self.transient,
        )

    def residual(self, flux: jnp.ndarray) -> Callable[..., jnp.ndarray]:
        """The residual function ``C -> R(C)`` for the flow whose volumetric flux is ``flux``.

        A bound :meth:`~aquaflux.discretization.ResidualAssembler.residual`, which ``equinox``
        treats as a pytree -- so handing it to a jitted solve each outer sweep changes only array
        *values* and reuses the compiled solve, where a freshly built closure would land on the
        static side and miss the compilation cache every sweep.

        Parameters
        ----------
        flux : jnp.ndarray
            Owner-outward volumetric face flux, shape ``(n_faces,)`` (see :meth:`assembler`).
        """
        return self.assembler(flux).residual
