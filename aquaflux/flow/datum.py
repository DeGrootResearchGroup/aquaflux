"""The pressure datum: where a domain whose pressure level is free has that level fixed.

Incompressible flow determines the pressure only up to a constant -- the momentum balance sees its
gradient and continuity does not see it at all -- so the level has to come from somewhere. A
:class:`~aquaflux.flow.PressureOutlet` supplies it: the pressure it prescribes is the level every
other pressure is measured from. A domain with no such patch -- a lid-driven cavity, a
streamwise-periodic channel, a box whose every patch prescribes the velocity -- has a free level,
and its discrete system is singular until one cell's continuity equation is replaced by
``p = value``. :class:`PinnedPoint` names that cell by a location rather than an index, so the choice
survives a renumbering of the cells or a remesh, and can be written in a case file.

Whether a datum is needed is not a choice: it follows from the boundary conditions.
:func:`refuse_an_unsuitable_pressure_datum` is that rule, and it refuses in both directions -- a free
level with no datum solves a singular system, and a datum beside a patch that already fixes the level
over-determines it -- because either would otherwise run and report nothing.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from aquaflux.boundary import BoundaryConditions
    from aquaflux.mesh import MeshGeometry

__all__ = ["PinnedPoint", "PressureDatum", "refuse_an_unsuitable_pressure_datum"]


@dataclasses.dataclass(frozen=True)
class PressureDatum(abc.ABC):
    """Where a domain with a free pressure level has it fixed. One implementation: :class:`PinnedPoint`."""

    @abc.abstractmethod
    def cell(self, geometry: MeshGeometry) -> int:
        """The cell whose continuity equation is replaced by ``p = value``.

        Parameters
        ----------
        geometry : MeshGeometry
            The mesh's geometry, whose cell centroids locate the datum.

        Returns
        -------
        int
            The cell index, in this mesh's numbering.
        """

    @property
    @abc.abstractmethod
    def level(self) -> float:
        """The pressure imposed at that cell."""


@dataclasses.dataclass(frozen=True)
class PinnedPoint(PressureDatum):
    """The pressure level fixed at the cell nearest a point.

    The cell is the one whose centroid is nearest ``point`` (the lowest-numbered of several equally
    near). Which cell carries the datum changes the pressure only by a constant, so "nearest" needs no
    finer definition, and a point -- unlike a cell index -- names the same place after the cells are
    renumbered or the domain is remeshed.

    Attributes
    ----------
    point : tuple of float
        The location, one coordinate per spatial dimension.
    value : float
        The pressure imposed there; every other pressure is measured from it.

    Raises
    ------
    ValueError
        If the point does not have two or three finite coordinates, or the value is not finite.

    Examples
    --------
    >>> PinnedPoint((0.0, 0.0))
    PinnedPoint(point=(0.0, 0.0), value=0.0)
    """

    point: tuple[float, ...]
    value: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", tuple(float(x) for x in self.point))
        if len(self.point) not in (2, 3):
            raise ValueError(
                f"a pressure datum's point has two or three coordinates, got {len(self.point)}: "
                f"{self.point!r}."
            )
        if not all(math.isfinite(x) for x in (*self.point, self.value)):
            raise ValueError(
                f"a pressure datum's point and value must be finite, got {self.point!r} and "
                f"{self.value!r}."
            )

    def cell(self, geometry: MeshGeometry) -> int:
        """The cell whose centroid is nearest :attr:`point` -- see :meth:`PressureDatum.cell`.

        Raises
        ------
        ValueError
            If the point's dimension is not the mesh's.
        """
        centroid = np.asarray(geometry.cell.centroid)
        if centroid.shape[1] != len(self.point):
            raise ValueError(
                f"the pressure datum's point {self.point!r} has {len(self.point)} coordinates, but "
                f"the mesh is {centroid.shape[1]}-dimensional."
            )
        # `argmin` returns the first of equal minima, which is the documented tie-break.
        return int(np.argmin(np.sum((centroid - np.asarray(self.point)) ** 2, axis=1)))

    @property
    def level(self) -> float:
        """:attr:`value`."""
        return self.value


def refuse_an_unsuitable_pressure_datum(
    boundary: BoundaryConditions, datum: PressureDatum | None, caller: str
) -> None:
    """Refuse a datum where the boundary fixes the pressure level, and its absence where none does.

    The level is fixed by any patch whose flow closure prescribes the pressure
    (:meth:`~aquaflux.flow.FlowBoundary.prescribes_pressure`); with none, it is free and needs a datum.

    Parameters
    ----------
    boundary : BoundaryConditions
        The flow closures, by patch.
    datum : PressureDatum or None
        The datum given, if any.
    caller : str
        What is being built, for the message.

    Raises
    ------
    ValueError
        If the level is free and no datum is given -- the system would be singular -- or a datum is
        given and a patch already fixes the level, which would over-determine it.
    """
    fixing = [
        name for name, closure in boundary.conditions.items() if closure.prescribes_pressure()
    ]
    if not fixing and datum is None:
        raise ValueError(
            f"{caller}: no boundary patch prescribes the pressure, so its level is free and the "
            "system is singular without a datum. Fix it at a point, with "
            "pressure_datum=PinnedPoint((x, y[, z])) -- which cell carries it changes the pressure "
            "only by a constant."
        )
    if fixing and datum is not None:
        raise ValueError(
            f"{caller}: {', '.join(map(repr, fixing))} already "
            f"{'fixes' if len(fixing) == 1 else 'fix'} the pressure level, so a pressure datum would "
            "over-determine it. Remove the datum."
        )
