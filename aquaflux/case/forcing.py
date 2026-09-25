"""What drives a case's flow beyond its boundary conditions: a held bulk velocity, or a prescribed force.

A streamwise-periodic channel prescribes the velocity nowhere, so something else sets it in motion,
and a case file names it one of two ways -- the same two the flow itself distinguishes:

* :class:`BulkVelocity` -- the **drive**: a uniform streamwise force *solved for* so that the bulk
  velocity is held at a target (:class:`~aquaflux.flow.MassFlow`). The force is an unknown of the
  solve, so the case names only what is held, plus a starting guess for the force.
* :class:`BodyForce` -- a **source**: a uniform force per unit volume that is *prescribed*
  (:class:`~aquaflux.flow.UniformBodyForce`), one entry of the case's ``sources``.

Both are case-file values that build the flow's own objects, because those hold their force as an
array a file cannot write.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Literal

import jax.numpy as jnp

from aquaflux.flow import Drive, MassFlow, MomentumSource, UniformBodyForce

__all__ = ["BodyForce", "BulkVelocity", "DriveSpec", "SourceSpec"]

#: The coordinate axes a direction may name, in order.
_AXES = ("x", "y", "z")


def _refuse_non_finite(owner: str, values: tuple[float, ...]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{owner} must be finite, got {values!r}.")


@dataclasses.dataclass(frozen=True)
class DriveSpec(abc.ABC):
    """What drives a case's flow, when its boundary conditions do not. One implementation: :class:`BulkVelocity`."""

    @abc.abstractmethod
    def drive(self) -> Drive:
        """The flow's drive this describes."""

    def refuse_for_dimension(self, dim: int) -> None:
        """Refuse this drive on a mesh of ``dim`` spatial dimensions if it cannot apply there.

        Raises
        ------
        ValueError
            If it names an axis the mesh does not have.
        """
        del dim


@dataclasses.dataclass(frozen=True)
class BulkVelocity(DriveSpec):
    """Hold the bulk (volume-averaged) velocity along one axis, with the uniform force that does so solved for.

    Attributes
    ----------
    target : float
        The bulk velocity to hold.
    direction : {"x", "y", "z"} or None
        The axis the bulk velocity is measured and the force applied along; unset, ``x``.
    initial_force : float or None
        The force per unit volume the solve starts from -- a guess, not a setting of the problem: the
        solved force does not depend on it, though how quickly the solve reaches it can. Unset, zero.

    Raises
    ------
    ValueError
        If the target or the initial force is not finite.
    """

    target: float
    direction: Literal["x", "y", "z"] | None = None
    initial_force: float | None = None

    def __post_init__(self) -> None:
        _refuse_non_finite("BulkVelocity.target", (self.target,))
        if self.initial_force is not None:
            _refuse_non_finite("BulkVelocity.initial_force", (self.initial_force,))

    def drive(self) -> MassFlow:
        """A :class:`~aquaflux.flow.MassFlow` holding :attr:`target`."""
        options: dict[str, object] = {}
        if self.direction is not None:
            options["flow_direction"] = _AXES.index(self.direction)
        if self.initial_force is not None:
            options["force"] = self.initial_force
        return MassFlow(target=self.target, **options)

    def refuse_for_dimension(self, dim: int) -> None:
        """Refuse a direction the mesh has no axis for."""
        if self.direction is not None and _AXES.index(self.direction) >= dim:
            raise ValueError(
                f"drive: the bulk velocity is held along {self.direction}, but the mesh is "
                f"{dim}-dimensional."
            )


@dataclasses.dataclass(frozen=True)
class SourceSpec(abc.ABC):
    """A term a case adds to its momentum balance. One implementation: :class:`BodyForce`."""

    @abc.abstractmethod
    def momentum_source(self) -> MomentumSource:
        """The flow's momentum source this describes."""

    def refuse_for_dimension(self, dim: int, index: int) -> None:
        """Refuse this source on a mesh of ``dim`` spatial dimensions if it cannot apply there.

        Parameters
        ----------
        dim : int
            The mesh's spatial dimension.
        index : int
            This source's position in the case's ``sources``, for the message.

        Raises
        ------
        ValueError
            If the source's settings do not fit a ``dim``-dimensional mesh.
        """
        del dim, index


@dataclasses.dataclass(frozen=True)
class BodyForce(SourceSpec):
    """A prescribed, uniform force per unit volume on the momentum balance.

    It drives a streamwise-periodic channel at a fixed force rather than a fixed bulk velocity (for
    that, see :class:`BulkVelocity`): the bulk velocity is then whatever the force sustains.

    Attributes
    ----------
    force : tuple of float
        The force per unit volume, one component per spatial dimension.

    Raises
    ------
    ValueError
        If the force does not have two or three finite components.
    """

    force: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.force) not in (2, 3):
            raise ValueError(
                f"a body force has two or three components, got {len(self.force)}: {self.force!r}."
            )
        _refuse_non_finite("BodyForce.force", self.force)

    def momentum_source(self) -> UniformBodyForce:
        """A :class:`~aquaflux.flow.UniformBodyForce` of :attr:`force`."""
        return UniformBodyForce(jnp.asarray(self.force))

    def refuse_for_dimension(self, dim: int, index: int) -> None:
        """Refuse a force whose component count is not the mesh's dimension."""
        if len(self.force) != dim:
            raise ValueError(
                f"sources[{index}]: the body force {self.force!r} has {len(self.force)} components, "
                f"but the mesh is {dim}-dimensional."
            )
