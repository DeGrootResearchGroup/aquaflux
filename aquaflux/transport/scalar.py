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

from aquaflux.boundary import (
    HOST_EQUATION_FIELD,
    Dirichlet,
    DirichletField,
    refuse_a_closure_that_closes_other_fields,
)
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


def prescribed_range(boundary: BoundaryConditions, geometry: MeshGeometry) -> float:
    """The magnitude of a scalar, read off the values its boundary conditions prescribe.

    The range ``max - min`` of every value a :class:`~aquaflux.boundary.Dirichlet` or
    :class:`~aquaflux.boundary.DirichletField` patch imposes -- the span a transported scalar lives
    in, by the maximum principle, when nothing else drives it. A single prescribed level has no
    range (an inlet at one concentration, the rest of the boundary open), and then the level itself
    is the magnitude, since the scalar's other bound is the zero a reaction or a dilution tends to.
    A temperature prescribed at a single level has the opposite problem -- its level says nothing
    about its variation -- and wants a scale stated on the limiter.

    Parameters
    ----------
    boundary : BoundaryConditions
        The scalar's closures, bound to a mesh.
    geometry : MeshGeometry
        That mesh's metrics (the face centroids a :class:`DirichletField` is evaluated at).

    Returns
    -------
    float
        The range of the prescribed values, else the largest prescribed magnitude; ``0.0`` when no
        patch prescribes a value.
    """
    values = []
    for name, closure in boundary.conditions.items():
        faces = boundary.faces[name]
        if isinstance(closure, Dirichlet | DirichletField) and faces.shape[0] > 0:
            # Neither closure reads the owner state; a prescribed value is a function of position.
            owner = jnp.zeros(faces.shape[0])
            centroid = geometry.face.centroid[faces]
            values.append(jnp.ravel(closure.face_value(owner, None, None, None, None, centroid)))
    if not values:
        return 0.0
    prescribed = jnp.concatenate(values)
    spread = float(jnp.max(prescribed) - jnp.min(prescribed))
    return spread if spread > 0.0 else float(jnp.max(jnp.abs(prescribed)))


def _scalar_scale(boundary: BoundaryConditions, geometry: MeshGeometry) -> float:
    """The scalar's magnitude for an advection scheme scaled by it, refused when there is none."""
    magnitude = prescribed_range(boundary, geometry)
    if not magnitude > 0.0:
        raise ValueError(
            "ScalarTransport.build: the advection's limiter is softened by a fraction of the "
            "scalar's magnitude, taken from the values its boundary conditions prescribe, and none "
            "prescribes a non-zero value. Give the limiter its scale."
        )
    return magnitude


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
        The face-value reconstruction for the advective flux, set by :meth:`build` for the
        scalar's magnitude (see :func:`prescribed_range`).
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
            The face-value reconstruction for advection. A scheme scaled by the field it advects (a
            softened slope limiter) is set for the scalar's magnitude, the range of the values its
            boundary conditions prescribe (:func:`prescribed_range`), unless it was given a scale of
            its own.
        gradient_scheme : GradientScheme, optional
            Reconstruction for the non-orthogonal correction; omit on orthogonal grids.
        sources : tuple of VolumeSource, optional
            Volume-source terms (default none).
        transient : TransientTerm, optional
            Accumulation term; omit for a steady scalar.

        Raises
        ------
        ValueError
            If ``advection_scheme`` reads a scale and no boundary condition prescribes a value to
            take one from.
        """
        refuse_a_closure_that_closes_other_fields(
            boundary, HOST_EQUATION_FIELD, "ScalarTransport.build"
        )
        resolved = boundary.resolve(mesh.face_patches, mesh.face_cells)
        return cls(
            mesh=mesh,
            geometry=geometry,
            diffusivity=diffusivity,
            boundary=resolved,
            advection_scheme=advection_scheme.with_reference_scale(
                lambda: _scalar_scale(resolved, geometry)
            ),
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
