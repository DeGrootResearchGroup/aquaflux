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
  one.
- :meth:`MomentumSource.diagonal` -- the contribution to the momentum-matrix diagonal. A
  velocity-dependent source (a linear drag ``-mu/K u``) changes the diagonal of the row it acts on,
  and the frozen preconditioner and the pseudo-transient shift are both assembled from that
  diagonal rather than from the residual. A source that reports zero while contributing a real
  diagonal leaves them preconditioning an operator the solve is not actually running. Pin any
  non-trivial implementation against automatic differentiation of the source's own
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

    from aquaflux.mesh import MeshGeometry

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
        fields: VelocityFields,
        geometry: MeshGeometry,
        properties: Mapping[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """This source's contribution to the momentum-matrix diagonal, shape ``(n_cells, dim)``.

        ``d(source_i)/d(u_i)`` integrated over the cell volume and **negated** -- the term as it
        appears on the diagonal of the momentum row, so a dissipative source (a drag opposing the
        velocity) contributes a positive diagonal. Zero for a source that does not read the
        velocity.

        The momentum diagonal is not read from the residual: it is assembled separately, and feeds
        the Rhie--Chow damping, the frozen preconditioner and the pseudo-transient shift. A source
        that omits its contribution therefore leaves those built from an operator that is not the
        one being solved. Pin a non-trivial implementation against automatic differentiation of
        :meth:`source`.

        Parameters
        ----------
        fields : VelocityFields
            The kinematic state (see :meth:`source`).
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
        fields: VelocityFields,
        geometry: MeshGeometry,
        properties: Mapping[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Zero -- the force does not read the velocity, so it adds no diagonal."""
        return jnp.zeros_like(fields.velocity)
