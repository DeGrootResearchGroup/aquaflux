"""Which equations a case solves: laminar flow, or Reynolds-averaged flow with a turbulence closure.

The physics is one of a case's two discriminators (the other is its drive). It decides which settings
a case may carry at all: everything that belongs to a turbulence closure -- its model constants, how
its variables are parametrized, how its fields are advected, the inflow turbulence at an inlet -- lives
inside :class:`RANS` or on a boundary patch, and a :class:`Laminar` case refuses any of it rather than
ignoring it.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping

from aquaflux.discretization import AdvectionScheme
from aquaflux.turbulence import ScalarVariableTransform, SSTModel

from .boundaries import PatchCondition

__all__ = ["RANS", "Laminar", "Physics"]


@dataclasses.dataclass(frozen=True)
class Physics(abc.ABC):
    """The equations a case solves: :class:`Laminar` or :class:`RANS`."""

    @abc.abstractmethod
    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse boundary patches this physics cannot use as written.

        Parameters
        ----------
        boundaries : mapping of {str: PatchCondition}
            The case's patches, by name.

        Raises
        ------
        ValueError
            Naming every offending setting at once, by its path in the case file.
        """


@dataclasses.dataclass(frozen=True)
class Laminar(Physics):
    """Laminar incompressible flow: momentum and continuity, with no turbulence closure."""

    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse any turbulence setting on a patch -- nothing in a laminar case would read it."""
        stray = [
            f"boundaries.{patch}.{setting}"
            for patch, condition in boundaries.items()
            for setting in condition.turbulence_settings()
        ]
        if stray:
            raise ValueError(
                f"{', '.join(stray)}: a laminar case has no turbulence closure, so nothing would read "
                f"{'this setting' if len(stray) == 1 else 'these settings'}. Remove "
                f"{'it' if len(stray) == 1 else 'them'}, or make the physics RANS."
            )


@dataclasses.dataclass(frozen=True)
class RANS(Physics):
    """Reynolds-averaged (RANS) incompressible flow closed by the k-omega shear-stress transport (SST) model.

    The flow and the closure's ``k`` and ``omega`` are solved together, as one coupled system.

    Attributes
    ----------
    advection : AdvectionScheme
        How ``k`` and ``omega`` are advected -- :class:`~aquaflux.discretization.FirstOrderUpwind` or
        :class:`~aquaflux.discretization.LimitedUpwind`. Required: the momentum advection is set
        separately, under the case's numerics.
    model : SSTModel or None
        The model's constants; unset, :class:`~aquaflux.turbulence.SSTModel`'s own.
    k_variable, omega_variable : ScalarVariableTransform or None
        The variable each field is solved in -- itself (:class:`~aquaflux.turbulence.DirectScalars`) or
        its logarithm (:class:`~aquaflux.turbulence.LogScalars`, which keeps it positive under any
        Newton step). Unset, the coupled system's default.
    explicit_production_limiter : bool or None
        Freeze the ``k``-production cap in the linearization; unset, the exact operator. See
        :class:`~aquaflux.turbulence.SSTTurbulence` for when this is safe.

    Raises
    ------
    TypeError
        If a setting is not a value of its family.
    """

    advection: AdvectionScheme
    model: SSTModel | None = None
    k_variable: ScalarVariableTransform | None = None
    omega_variable: ScalarVariableTransform | None = None
    explicit_production_limiter: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.advection, AdvectionScheme):
            raise TypeError(f"RANS.advection must be an AdvectionScheme, got {self.advection!r}.")
        for name, family in (
            ("model", SSTModel),
            ("k_variable", ScalarVariableTransform),
            ("omega_variable", ScalarVariableTransform),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, family):
                raise TypeError(f"RANS.{name} must be a {family.__name__}, got {value!r}.")

    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse a patch missing a setting the closure needs -- the inflow turbulence at an inlet."""
        missing = [
            f"boundaries.{patch}.{setting}"
            for patch, condition in boundaries.items()
            for setting in condition.missing_turbulence_settings()
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)}: a RANS case needs the turbulence every inflow carries in. Give "
                "it, e.g. turbulence: {kind: FixedTurbulence, k: ..., omega: ...}."
            )
