"""The cell residual ``R(phi)``: the per-face context it is built from, and the balance itself.

Everything the solver needs reduces to one discrete residual per cell — the finite-volume
conservation balance

    R_P = accumulation_P + sum_faces(owner-outward flux) - sum_sources(cell integral of S dV),

which vanishes at the converged solution. Each flux operator returns the owner-outward flux of the
conserved quantity (advection ``+ mdot phi``, diffusion ``- Gamma grad phi . n A``); each volume
source returns its cell integral (production positive), which leaves the balance as a sink.

Forming that residual is two separable jobs, and this module gives each its own object:

- :class:`ResidualAssembler` builds the **context** — it evaluates the per-cell properties,
  reconstructs the cell gradients once (the injected :class:`GradientScheme`, if any) so every
  operator shares a single gradient field, evaluates the weak boundary face values from the
  per-patch :class:`~aquaflux.boundary.conditions.BoundaryCondition` closures, and packs the
  result into a :class:`~aquaflux.context.FieldContext`.
- :class:`CellBalance` assembles the **balance** from that context — it sums the injected
  :class:`~aquaflux.discretization.face_flux.FaceFluxOperator`\\ s, scatters the owner-outward
  face flux back to cells with ``segment_sum`` (owner ``+``, neighbour ``-``; boundary faces to
  the owner only), subtracts each injected
  :class:`~aquaflux.discretization.source.VolumeSource`, and adds the injected transient
  (accumulation) term.

The assembler holds a balance and delegates to it, so a scalar transport equation is still one
object built by :meth:`ResidualAssembler.build` and evaluated by
:meth:`ResidualAssembler.residual`. They are separate because the two halves have genuinely
different consumers: a coupled system that reconstructs its own gradients and evaluates its own
boundary closures — the momentum block reconstructs one velocity-gradient *tensor* shared across
its components, from vector-valued flow boundary conditions — arrives already holding a context,
and needs only the balance. :class:`CellBalance` therefore stores nothing but its operators and
reads the mesh, the geometry, and the boundary values off the context it is handed, exactly as the
flux operators themselves do.

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

import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aquaflux.boundary import (
    HOST_EQUATION_FIELD,
    BoundaryConditions,
    refuse_a_closure_that_closes_other_fields,
)
from aquaflux.context import FieldContext, MeshContext
from aquaflux.schemes import BoundaryLinearization

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.properties import PropertyModel
    from aquaflux.schemes import GradientScheme, ImposedGradient

    from .face_flux import FaceFluxOperator
    from .source import VolumeSource
    from .transient import TransientTerm


class CellBalance(eqx.Module):
    """Assemble the cell balance ``R(phi)`` from injected operators, given a per-face context.

    The half of the residual that is pure operator composition: sum the face fluxes, scatter them
    to cells, subtract the volume sources, add the accumulation. It holds **only** the operators —
    the connectivity, geometry, boundary face values, reconstructed gradient, and properties all
    arrive on the :class:`~aquaflux.context.FieldContext` it is handed, which is
    the same context its operators gather from. So it needs no mesh to construct and none to
    exercise: a stub flux operator and a two-cell context test it on its own.

    :class:`ResidualAssembler` builds that context and delegates here, which is how a scalar
    transport equation uses it. A coupled system that forms its own context drives this directly —
    the momentum block reconstructs one velocity-gradient *tensor* shared across its components,
    from vector-valued flow boundary conditions, so it cannot share the assembler's context step
    but assembles the identical balance.

    Attributes
    ----------
    flux_operators : tuple of FaceFluxOperator
        Face-flux operators summed into the transport term. **Their order is the summation order**,
        so it is part of the arithmetic, not a presentational choice.
    source_operators : tuple of VolumeSource
        Volume-source operators subtracted from the balance (each returns its cell integral,
        production positive); empty for a flux-only equation.
    transient : TransientTerm or None
        Accumulation term; ``None`` for a steady residual.
    """

    flux_operators: tuple[FaceFluxOperator, ...]
    source_operators: tuple[VolumeSource, ...] = ()
    transient: TransientTerm | None = None

    def residual(
        self,
        phi: jnp.ndarray,
        context: FieldContext,
        phi_old: jnp.ndarray | None = None,
        phi_older: jnp.ndarray | None = None,
        dt: float | None = None,
        first_step: bool = False,
    ) -> jnp.ndarray:
        """Cell balance ``R(phi)`` for a context already formed, shape ``(n_cells,)``.

        Parameters
        ----------
        phi : jnp.ndarray
            Current cell field, shape ``(n_cells,)``.
        context : FieldContext
            The per-face inputs the operators gather from, and the source of the connectivity
            (``face_cells``) this scatters over and the cell volumes the transient integrates on.
        phi_old, phi_older : jnp.ndarray, optional
            Previous / second-previous time levels for the transient term (required when a
            :class:`TransientTerm` was injected), shape ``(n_cells,)``.
        dt : float, optional
            Timestep (required with a transient term).
        first_step : bool
            ``True`` on the first timestep (BDF1); static.

        Returns
        -------
        jnp.ndarray
            The balance ``accumulation + net outward flux - volume sources``, shape
            ``(n_cells,)``.
        """
        face_cells = context.mesh.face_cells
        face_flux = jnp.zeros(face_cells.n_faces, dtype=phi.dtype)
        for operator in self.flux_operators:
            face_flux = face_flux + operator.face_flux(phi, context)
        # Each operator returns the owner-outward flux of the conserved quantity; the residual
        # is the finite-volume balance accumulation + sum of net outward face fluxes.
        residual = face_cells.scatter_conservative(face_flux)
        # A volume source is produced inside the cell, so it leaves the balance as a sink: each
        # returns its cell integral (production positive) and is subtracted from the residual.
        for operator in self.source_operators:
            residual = residual - operator.source(phi, context)
        if self.transient is not None:
            residual = residual + self.transient.residual(
                phi, phi_old, phi_older, dt, first_step, context.mesh.geometry.cell.volume
            )
        return residual


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
    properties : PropertyModel
        The named per-cell physical properties, evaluated each residual and threaded to the flux
        operators (via the context) and the boundary closures.
    balance : CellBalance
        The injected operators and the composition of them into the cell balance, which this
        delegates to once the context is formed.
    gradient_scheme : GradientScheme or None
        Cell-gradient reconstruction shared by the flux operators' non-orthogonal
        corrections. ``None`` reconstructs no gradient (exact on orthogonal grids, where the
        correction vanishes identically).
    imposed_gradient : ImposedGradient or None
        Cells whose gradient of *this equation's field* is known analytically and is to be used in
        place of a reconstruction. ``None`` (the usual case) reconstructs everywhere. It is a
        property of the equation rather than of a call, so :meth:`residual` and :meth:`gradient`
        cannot disagree about it -- which is the point, since a field whose gradient must be imposed
        needs it imposed wherever it is reconstructed.
    coefficient : str
        The property the flux-type boundary closures (Robin/Neumann) use as their
        diffusion coefficient ``Gamma`` (static; matches the ``DiffusionFlux.coefficient`` of the
        equation's diffusion term).
    boundary : BoundaryConditions
        The named per-patch closures, resolved to their boundary-face indices.
    """

    mesh: Mesh
    geometry: MeshGeometry
    properties: PropertyModel
    balance: CellBalance
    gradient_scheme: GradientScheme | None
    coefficient: str = eqx.field(static=True)
    boundary: BoundaryConditions
    imposed_gradient: ImposedGradient | None = None

    @classmethod
    def build(
        cls,
        mesh: Mesh,
        geometry: MeshGeometry,
        properties: PropertyModel,
        flux_operators: tuple[FaceFluxOperator, ...],
        boundary: BoundaryConditions,
        *,
        coefficient: str = "diffusivity",
        transient: TransientTerm | None = None,
        source_operators: tuple[VolumeSource, ...] = (),
        gradient_scheme: GradientScheme | None = None,
        boundary_linearization: BoundaryLinearization | None = None,
        bind_gradient_scheme: bool = True,
        imposed_gradient: ImposedGradient | None = None,
    ) -> ResidualAssembler:
        """Build an assembler from injected operators, schemes, and boundary closures.

        Parameters
        ----------
        mesh : Mesh
            The mesh; its ``face_patches`` name the boundary faces.
        geometry : MeshGeometry
            Geometry from ``mesh.geometry()``.
        properties : PropertyModel
            The named per-cell physical properties. Each operator's own
            :meth:`~aquaflux.discretization.face_flux.FaceFluxOperator.requires` names what it reads
            (the diffusion term its coefficient); a flux-type boundary closure in ``boundary``
            declares it reads ``coefficient`` via :meth:`~aquaflux.boundary.conditions.
            BoundaryCondition.requires_coefficient`. Checked against the operators and closures
            actually given -- see ``Raises``.
        flux_operators : tuple of FaceFluxOperator
            Face-flux operators (e.g. one :class:`DiffusionFlux`). They are summed in the order
            given, so the order is part of the arithmetic.
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
        imposed_gradient : ImposedGradient, optional
            Cells whose gradient of this equation's field is a model quantity rather than something
            to reconstruct -- a near-wall ``omega``, whose value is itself imposed, is the standing
            case. Given here rather than per call so that every reconstruction this assembler makes
            honours it.
        boundary_linearization : BoundaryLinearization, optional
            How each boundary value depends on its owner cell, which the gradient scheme is bound
            against. Read off the conditions when omitted; supply it only when a calculated property
            makes that impossible before a field exists.
        bind_gradient_scheme : bool, optional
            Bind ``gradient_scheme`` here (default ``True``). ``False`` takes it as already bound
            against this geometry **and this equation's conditions** -- which is what a caller that
            builds an assembler inside a residual evaluation must do, since binding a scheme whose
            preparation reads the connectivity cannot be traced. The caller then owns the pairing
            that binding here exists to guarantee, so pass a scheme bound for *this* equation.

        Raises
        ------
        ValueError
            If an operator names a property (:meth:`~aquaflux.discretization.face_flux.
            FaceFluxOperator.requires`) that ``properties`` does not supply, if a flux-type boundary
            closure needs ``coefficient`` (:meth:`~aquaflux.boundary.conditions.BoundaryCondition.
            requires_coefficient`) and ``properties`` does not supply it, or if a flux operator needs
            a reconstructed gradient (:meth:`~aquaflux.discretization.face_flux.
            FaceFluxOperator.uses_gradient`) but ``gradient_scheme`` is ``None`` -- all three would
            otherwise surface only inside a jitted residual evaluation, as a bare ``KeyError``, a
            non-finite result (a divide by the coefficient's zero fallback), or a silently degraded
            (1st-order) result respectively.
        """
        needed = {name for op in (*flux_operators, *source_operators) for name in op.requires()}
        if any(bc.requires_coefficient() for bc in boundary.conditions.values()):
            needed.add(coefficient)
        if needed:
            properties.require(*sorted(needed))
        refuse_a_closure_that_closes_other_fields(
            boundary, HOST_EQUATION_FIELD, "ResidualAssembler.build"
        )
        if gradient_scheme is None:
            needing = [op for op in flux_operators if op.uses_gradient()]
            if needing:
                names = ", ".join(sorted({type(op).__name__ for op in needing}))
                raise ValueError(
                    f"flux operator(s) [{names}] need a reconstructed gradient, but no "
                    "gradient_scheme was given -- with none, context.gradient is exactly zero "
                    "everywhere, which silently degrades such an operator rather than failing"
                )
        assembled = cls(
            mesh=mesh,
            geometry=geometry,
            properties=properties,
            balance=CellBalance(
                flux_operators=flux_operators,
                source_operators=source_operators,
                transient=transient,
            ),
            # Prepared for this geometry: geometry-only reconstruction work hoisted out of the
            # per-call path, since the residual is evaluated once per field per Krylov matvec. This
            # assembler owns the geometry and the scheme together, so binding here -- rather than
            # leaving it to a call site -- is what stops the two being paired with a mismatched mesh.
            gradient_scheme=(
                None
                if gradient_scheme is None
                else gradient_scheme.bind(mesh, geometry)
                if bind_gradient_scheme
                else gradient_scheme
            ),
            coefficient=coefficient,
            boundary=boundary.resolve(mesh.face_patches, mesh.face_cells),
            imposed_gradient=imposed_gradient,
        )
        if gradient_scheme is None or not bind_gradient_scheme:
            return assembled
        # Bind the scheme AGAINST the conditions, not merely against the geometry. A scheme that
        # prepares work from its own operator must be prepared from the one it will apply, and the
        # boundary conditions are part of that operator -- otherwise it corrects an operator nobody
        # evaluates and stops reproducing a quadratic.
        linearization = (
            assembled._build_time_boundary_linearization()
            if boundary_linearization is None
            else boundary_linearization
        )
        return dataclasses.replace(
            assembled, gradient_scheme=gradient_scheme.bind(mesh, geometry, linearization)
        )

    def _build_time_boundary_linearization(self) -> BoundaryLinearization:
        """How each boundary value depends on its owner cell, evaluated without a state.

        Both derivatives are properties of the conditions and the geometry, not of the field: every
        condition here is affine in its owner's value and gradient. Measured on a perturbed grid
        carrying zero-gradient, Neumann, Dirichlet and convective patches at once, both are
        **bit-identical** across a zero state, a random state and one offset by 5.0, at zero and at a
        random gradient. So evaluating them here, once, is exact rather than an approximation --
        pinned by ``test_the_boundary_linearization_does_not_depend_on_the_state``.

        They are *not* independent of the properties: a convective condition blends the coefficient
        into its face value, so both of its derivatives carry ``Gamma``. A property that cannot be
        evaluated without a field therefore cannot give them here, and this raises rather than
        quietly using the zero default ``boundary_values`` falls back to -- a silently wrong
        linearization is the very defect binding against the conditions exists to remove.
        """
        try:
            properties = self.properties.evaluate(self.mesh.cell_zones, {})
        except (KeyError, ValueError) as error:
            raise ValueError(
                "ResidualAssembler.build: the gradient scheme is bound against the boundary "
                "conditions, whose linearization needs the properties evaluated, and a calculated "
                f"property cannot be evaluated before a field exists ({error}). Pass it yourself as "
                "`boundary_linearization=` -- a BoundaryLinearization holding "
                "`d(boundary value)/d(phi_owner)` and `d(boundary value)/d(grad phi_owner)` per face."
            ) from error
        zero_field = jnp.zeros(self.mesh.n_cells)
        zero_gradient = jnp.zeros((self.mesh.n_cells, self.mesh.dim))
        return BoundaryLinearization(
            value_weight=self._boundary_value_weight(zero_field, zero_gradient, properties),
            gradient_weight=self._boundary_gradient_weight(zero_field, zero_gradient, properties),
        )

    def boundary_values(
        self, phi: jnp.ndarray, gradient: jnp.ndarray, properties: dict[str, jnp.ndarray]
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
        properties : dict of {str: jnp.ndarray}
            The evaluated per-cell properties; the flux-type closures read
            ``properties[self.coefficient]`` as ``Gamma``.
        """
        centroid = self.geometry.cell.centroid
        # The diffusion coefficient for the flux-type (Robin/Neumann) closures. The zero fallback is
        # only ever read by a closure that does not use it (a pure-advection problem with only
        # value/inflow BCs): `build` requires `coefficient` from `properties` whenever any closure's
        # `requires_coefficient()` is True, so a flux-type closure reaches this with a real Gamma.
        gamma = properties.get(self.coefficient, jnp.zeros(self.mesh.n_cells, dtype=phi.dtype))

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
        properties: dict[str, jnp.ndarray],
    ) -> FieldContext:
        """The shared per-face inputs each flux operator gathers from."""
        mesh_context = MeshContext(
            face_cells=self.mesh.face_cells,
            geometry=self.geometry,
            properties=properties,
        )
        return FieldContext(
            mesh=mesh_context,
            boundary_values=boundary_values,
            gradient=gradient,
        )

    def _gradient(
        self,
        phi: jnp.ndarray,
        properties: dict[str, jnp.ndarray],
        *,
        gradient_hook: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Cell gradients and the boundary values consistent with them.

        With no gradient scheme the gradient is zero (exact on orthogonal grids, where the
        non-orthogonal correction vanishes). Otherwise the reconstruction is fed a
        leading-order boundary value (its own tangential correction dropped, i.e. evaluated
        at zero gradient) to keep ``R`` a single-pass function of ``phi``; the flux then uses
        the full boundary value evaluated at the reconstructed gradient. The two agree
        exactly on orthogonal grids.

        ``gradient_hook`` is the distributed ghost-cell exchange (see :meth:`residual`); it is
        threaded into an iterative reconstruction's own linear solve so a partition-coupled gradient
        scheme refreshes its ghost rows each sweep, and is applied again to the returned gradient in
        :meth:`residual` so the flux reads exchanged ghost gradients.

        :attr:`imposed_gradient`, when this assembler carries one, is handed to the scheme rather
        than applied afterwards, because a scheme that consumes its own reconstructed gradient must
        impose before that consumer reads it.
        """
        dim = self.mesh.dim
        n_cells = self.mesh.n_cells
        if self.gradient_scheme is None:
            gradient = jnp.zeros((n_cells, dim), dtype=phi.dtype)
            return gradient, self.boundary_values(phi, gradient, properties)
        zero_grad = jnp.zeros((n_cells, dim), dtype=phi.dtype)
        leading_bvals = self.boundary_values(phi, zero_grad, properties)
        gradient = self.gradient_scheme.gradients(
            phi,
            self.mesh,
            self.geometry,
            leading_bvals,
            operator_hook=gradient_hook,
            imposed=self.imposed_gradient,
            # A scheme that DIFFERENTIATES a boundary value cannot use the leading-order one: a
            # gradient-type closure's whole content is a correction, which evaluating at zero
            # gradient throws away. This lets such a scheme ask for the corrected values at its own
            # reconstructed gradient; the ones passed above stay leading-order for everything else.
            boundary_values_at=lambda g: self.boundary_values(phi, g, properties),
            # How each boundary face value depends on its owner's gradient, read off the closures
            # themselves rather than declared: zero where the value is prescribed, the tangential
            # offset where a zero-gradient or Neumann condition carries its correction. It lets a
            # scheme reconstruct against the face values the boundary conditions actually define.
            boundary_gradient_weight=self._boundary_gradient_weight(phi, zero_grad, properties),
        )
        return gradient, self.boundary_values(phi, gradient, properties)

    def _boundary_value_weight(
        self,
        phi: jnp.ndarray,
        gradient: jnp.ndarray,
        properties: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """``d(boundary value)/d(phi_owner)`` per face, shape ``(n_faces,)``.

        Differentiated from the closures, as the gradient weight is: zero where the value is
        prescribed, one where a normal derivative is, and between the two for a Robin condition. A
        face value reads only its own owner, so one directional derivative seeded in every cell
        resolves every face.
        """
        return jax.jvp(
            lambda field: self.boundary_values(field, gradient, properties),
            (phi,),
            (jnp.ones_like(phi),),
        )[1]

    def _boundary_gradient_weight(
        self,
        phi: jnp.ndarray,
        gradient: jnp.ndarray,
        properties: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """``d(boundary value)/d(grad phi_owner)`` per face, shape ``(n_faces, dim)``.

        Differentiated from the closures rather than declared, so it cannot disagree with them: a
        prescribed value does not move with the gradient and returns zero, and a zero-gradient,
        Neumann or Robin condition returns the offset its face value carries. A face value reads only
        its own owner's gradient, so one directional derivative per gradient component, seeded in
        every cell at once, resolves every face.
        """
        return jnp.stack(
            [
                jax.jvp(
                    lambda g: self.boundary_values(phi, g, properties),
                    (gradient,),
                    (jnp.zeros_like(gradient).at[:, k].set(1.0),),
                )[1]
                for k in range(gradient.shape[1])
            ],
            axis=-1,
        )

    def gradient(
        self, phi: jnp.ndarray, *, fields: Mapping[str, jnp.ndarray] | None = None
    ) -> jnp.ndarray:
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
        fields : mapping of {str: jnp.ndarray}, optional
            Named per-cell state fields for a state-dependent property (see :meth:`residual`);
            omit when every property is state-independent.
        """
        properties = self.properties.evaluate(self.mesh.cell_zones, fields)
        return self._gradient(phi, properties)[0]

    def residual(
        self,
        phi: jnp.ndarray,
        phi_old: jnp.ndarray | None = None,
        phi_older: jnp.ndarray | None = None,
        dt: float | None = None,
        first_step: bool = False,
        *,
        gradient_hook: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        fields: Mapping[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Cell residual ``R(phi)``, shape ``(n_cells,)``.

        Forms the per-face context — properties, reconstructed gradients, weak boundary face
        values — and hands it to :attr:`balance`, which assembles the balance itself.

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
        gradient_hook : callable, optional
            A transform ``gradient -> gradient`` (shape ``(n_cells, dim)``) that overwrites ghost
            rows with the values their owning partition computed; the identity when omitted. This is
            the seam a distributed residual uses to correct ghost-cell gradients (a ghost's own
            stencil is incomplete locally, so its reconstructed gradient would be wrong). It is used
            at two depths: threaded into an iterative gradient scheme's own linear solve so it
            refreshes ghost rows each sweep (a partition-coupled reconstruction then converges to the
            serial gradient on owned cells), and applied again to the returned gradient before the
            flux consumes it. Boundary values are unaffected either way, because they read only
            owner-cell gradients (every boundary face is owned by an interior cell of its own
            partition).
        fields : mapping of {str: jnp.ndarray}, optional
            Named per-cell state fields a state-dependent property may read (e.g. a
            temperature-dependent viscosity naming ``"temperature"``), forwarded verbatim to
            :meth:`~aquaflux.properties.PropertyModel.evaluate`; omit (or leave empty) when every
            property is state-independent, which every property in this library is today. This
            equation's own ``phi`` is *not* added to it automatically — a caller solving, say, the
            temperature equation itself and wanting a property to see its own field supplies
            ``fields={"temperature": phi}`` explicitly, the same way it would supply another
            equation's converged field; the assembler does not guess a name for the field it solves.

        Returns
        -------
        jnp.ndarray
            The residual ``accumulation + net outward flux - volume sources``, shape
            ``(n_cells,)``.
        """
        properties = self.properties.evaluate(self.mesh.cell_zones, fields)
        # The hook is used at two depths of the same reconstruction: threaded *into* an iterative
        # gradient scheme's solve so it refreshes ghost rows each sweep (owned rows then converge to
        # the serial gradient), and applied *again* to the returned gradient below so the flux reads
        # exchanged ghost gradients. A single-pass scheme ignores the first use; both are the identity
        # when the hook is omitted (the serial path).
        gradient, boundary_values = self._gradient(phi, properties, gradient_hook=gradient_hook)
        if gradient_hook is not None:
            gradient = gradient_hook(gradient)
        context = self._context(gradient, boundary_values, properties)
        return self.balance.residual(phi, context, phi_old, phi_older, dt, first_step)
