"""Momentum source terms: the vector counterpart of the scalar volume source.

A momentum source is a force per unit volume acting on the flow -- buoyancy, porous-media drag,
a Coriolis or rotating-frame term, a prescribed driving force. It is the vector analogue of
:class:`~aquaflux.discretization.VolumeSource` and follows the same sign convention: a concrete
source returns its *volume-integrated* contribution per cell, **production positive**, and the
momentum residual subtracts it, so

    R_{u_i} = sum_faces(owner-outward flux) - sum_sources(cell integral of S_i dV).

**Why this is a separate interface rather than a scalar source evaluated per component.** A
momentum source is generally *coupled across components* -- a rotating-frame term is
``-2 rho Omega x u``, whose component ``i`` cannot be evaluated without the whole velocity vector --
so it acts on the vector state and returns a ``(n_cells, dim)`` contribution. For the same reason it
does not take a :class:`~aquaflux.discretization.FaceContext`: that context carries *one* scalar
component's boundary values and reconstructed gradient, while a momentum source needs the whole
kinematic state (the velocity, and the velocity-gradient **tensor** for anything stress-like). It
therefore receives the :class:`~aquaflux.flow.VelocityFields` bundle those quantities already travel
in, plus the geometry and the evaluated per-cell properties.

**A source answers three questions, and all three are abstract on purpose.** Only the first is the
term itself; the other two exist because a momentum source can silently break machinery that lives
outside the residual, and a default answer would let a new source inherit a wrong one:

- :meth:`MomentumSource.source` -- the term, integrated over the cell volume.
- :meth:`MomentumSource.face_force` -- the face-normal force a *non-uniform* source must expose so
  the mass flux can treat it consistently with the pressure gradient. Skipping this for a spatially
  varying force reintroduces exactly the odd-even pressure-velocity decoupling that the Rhie--Chow
  interpolation exists to suppress, now in the force instead of the pressure. Returning ``None``
  means "this force needs no such treatment", which is true only of a uniform, field-independent
  one. The mass flux does not yet apply this term, so anything other than ``None`` is **refused**
  at build (``reject_unsupported_face_force``) rather than dropped without a word.
- :meth:`MomentumSource.diagonal` -- the contribution to the momentum-matrix diagonal. A
  velocity-dependent source (a linear drag ``-mu/K u``) changes the diagonal of the row it acts on,
  and that diagonal is assembled separately from the residual: it is the Rhie--Chow damping ``V /
  a_P``, and it is what the frozen preconditioner and the pseudo-transient shift are built from. A
  source that reports zero while contributing a real diagonal therefore does more than mis-precondition
  -- the damping is differentiated and solution-affecting, so the converged answer and its adjoint are
  wrong too. Pin any non-trivial implementation against automatic differentiation of the source's own
  :meth:`~MomentumSource.source`, so the two cannot drift.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import scale

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.properties import PropertyModel

    from .momentum import VelocityFields


class MomentumSource(eqx.Module):
    """Strategy interface: a force per unit volume acting on the momentum equation.

    A concrete source returns one vector per cell -- the force integrated over the cell volume,
    production positive (see the module sign convention). It is an immutable ``equinox.Module``, so
    any coefficient or field it carries is a differentiable leaf and gradients flow through it.
    """

    @abc.abstractmethod
    def source(
        self,
        fields: VelocityFields,
        geometry: MeshGeometry,
        properties: Mapping[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Volume-integrated force per cell, shape ``(n_cells, dim)`` (production positive).

        Parameters
        ----------
        fields : VelocityFields
            The kinematic state: cell velocity ``(n_cells, dim)``, boundary-face velocity
            ``(n_faces, dim)``, and the cell velocity-gradient tensor ``(n_cells, dim, dim)``.
        geometry : MeshGeometry
            Face and cell metrics; the source integrates on ``geometry.cell.volume``.
        properties : mapping of {str: jnp.ndarray}
            The evaluated per-cell properties, ``{name: (n_cells,) array}`` -- ``"density"`` for a
            buoyancy term, ``"viscosity"`` for a drag one.

        Returns
        -------
        jnp.ndarray
            The force integrated over each cell's volume, shape ``(n_cells, dim)``.
        """

    @abc.abstractmethod
    def face_force(
        self, geometry: MeshGeometry, properties: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray | None:
        """Owner-outward face-normal force per face, shape ``(n_faces,)``, or ``None``.

        A **spatially varying** force must be reconstructed to faces the same way the pressure
        gradient is, and enter the mass flux alongside it; otherwise the cell-centred force and the
        face-based pressure gradient are inconsistent and the pressure-velocity decoupling the
        Rhie--Chow interpolation suppresses reappears in the force. Returning ``None`` declares that
        no such treatment is needed -- correct for a force that is uniform and independent of the
        solved state, and **not** correct for one that varies in space.

        **Not yet consumed, and a non-``None`` return is REFUSED rather than ignored.** The mass flux
        carries the pressure-gradient term alone; the balanced-force treatment is deferred to arrive
        with buoyancy, which it is entangled with through variable density. Building an assembler
        around a source that declares a face force therefore raises
        (``reject_unsupported_face_force``) instead of silently dropping it. The seam stays
        declared so that an interface built now cannot be shaped in a way that cannot express it.

        Parameters
        ----------
        geometry : MeshGeometry
            Face and cell metrics; the face normal and area define the projection.
        properties : mapping of {str: jnp.ndarray}
            The evaluated per-cell properties.
        """

    @abc.abstractmethod
    def diagonal(
        self,
        velocity: jnp.ndarray,
        geometry: MeshGeometry,
        properties: Mapping[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """This source's contribution to the momentum-matrix diagonal, shape ``(n_cells, dim)``.

        ``d(source_i)/d(u_i)`` integrated over the cell volume and **negated** -- the term as it
        appears on the diagonal of the momentum row, so a dissipative source (a drag opposing the
        velocity) contributes a positive diagonal. Zero for a source that does not read the
        velocity.

        The momentum diagonal is not read from the residual: it is assembled separately
        (:meth:`~aquaflux.flow.MomentumContinuity.momentum_matrix_diagonal`, which adds this term),
        and feeds the Rhie--Chow damping ``V / a_P``, the frozen preconditioner and the
        pseudo-transient shift. The first of those is differentiated and solution-affecting, so a
        source that misreports this leaves the converged answer and its adjoint wrong, not merely
        the preconditioner. Pin a non-trivial implementation against automatic differentiation of
        :meth:`source`.

        Unlike :meth:`source` this takes the **cell velocity alone** rather than the whole
        :class:`~aquaflux.flow.VelocityFields` bundle, because a diagonal is by definition the
        derivative of a cell's own row with respect to that cell's own unknown: the boundary-face
        values and the velocity-gradient tensor are neighbour couplings and belong to off-diagonal
        entries. That is also what lets the diagonal be assembled where only a velocity is in hand
        -- the frozen preconditioner and the shift both build it from a bare velocity, with no
        gradient reconstruction.

        Parameters
        ----------
        velocity : jnp.ndarray
            Per-cell velocity, shape ``(n_cells, dim)``.
        geometry : MeshGeometry
            Face and cell metrics; the contribution is integrated on ``geometry.cell.volume``.
        properties : mapping of {str: jnp.ndarray}
            The evaluated per-cell properties.
        """


class UniformBodyForce(MomentumSource):
    """A spatially uniform force per unit volume, ``f``, acting on every cell.

    The simplest momentum source, and the one that shows what the contract's other two members mean
    when they are trivial: because ``f`` is the same everywhere and independent of the solved state,
    it needs no face treatment and adds no diagonal, so :meth:`face_force` returns ``None`` and
    :meth:`diagonal` returns zero.

    Physically this is gravity in a constant-density fluid, or the mean pressure gradient that drives
    a streamwise-periodic flow -- with the pressure split ``p = p~ + G.x`` into a periodic ``p~`` and
    a mean gradient ``G``, the linear part is a constant force ``f = -G``, so a positive
    ``force[0]`` drives the flow in ``+x``.

    Attributes
    ----------
    force : jnp.ndarray
        Force per unit volume, shape ``(dim,)``. A differentiable leaf.
    """

    force: jnp.ndarray

    def source(
        self,
        fields: VelocityFields,
        geometry: MeshGeometry,
        properties: Mapping[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """The uniform force integrated over each cell, shape ``(n_cells, dim)``."""
        return scale(jnp.broadcast_to(self.force, fields.velocity.shape), geometry.cell.volume)

    def face_force(
        self, geometry: MeshGeometry, properties: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray | None:
        """``None`` -- a uniform force needs no mass-flux treatment.

        The Rhie--Chow face treatment exists to keep a *spatially varying* force consistent with the
        face pressure gradient. A constant force has the same value on both sides of every face, so
        the interpolated and cell-centred forms agree identically and there is nothing to correct.
        """
        return None

    def diagonal(
        self,
        velocity: jnp.ndarray,
        geometry: MeshGeometry,
        properties: Mapping[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Zero -- the force does not read the velocity, so it adds no diagonal."""
        return jnp.zeros_like(velocity)


def reject_unsupported_face_force(
    sources: tuple[MomentumSource, ...],
    geometry: MeshGeometry,
    properties: PropertyModel,
    mesh: Mesh,
) -> None:
    """Refuse a source declaring a face force, because the mass flux cannot yet apply one.

    :meth:`MomentumSource.face_force` is a declared seam with no consumer: the Rhie--Chow mass flux
    (:func:`~aquaflux.flow.rhie_chow.interior_mass_flux` and the per-patch boundary closures) carries
    the pressure-gradient term and nothing else, and the balanced-force treatment a spatially varying
    force needs is deferred to arrive with buoyancy, whose variable density it is entangled with.

    Until then a source returning anything but ``None`` would have its face force **silently dropped**
    -- the mass flux would treat a varying force inconsistently with the face pressure gradient, which
    is the odd-even pressure--velocity decoupling the Rhie--Chow interpolation exists to suppress,
    reappearing in the force. A wrong answer with no indication is the worst of the three possible
    behaviours, so this makes it a refusal instead. The seam stays declared, so an implementation
    written against it is ready when the mass-flux half lands.

    Evaluated once, at build, rather than per residual: whether a source declares a face force is a
    property of the source, not of the state.

    Parameters
    ----------
    sources : tuple of MomentumSource
        The injected sources.
    geometry : MeshGeometry
        Face and cell metrics, as :meth:`MomentumSource.face_force` takes them.
    properties : PropertyModel
        The unevaluated property model; evaluated here only if there is a source to ask.
    mesh : Mesh
        The mesh, for the cell zones the properties are evaluated over.

    Raises
    ------
    NotImplementedError
        If any source returns a face force.
    """
    if not sources:
        return
    evaluated = properties.evaluate(mesh.cell_zones)
    for source in sources:
        if source.face_force(geometry, evaluated) is not None:
            raise NotImplementedError(
                f"{type(source).__name__}.face_force declares a face-normal force, and the "
                "Rhie--Chow mass flux does not yet carry one -- it would be silently dropped, "
                "leaving the force inconsistent with the face pressure gradient. Return None (only "
                "correct for a force that is uniform and independent of the solved state) until the "
                "balanced-force face treatment is built."
            )
