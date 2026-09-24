"""The fluid a case is solved for, stated once.

Every equation of a case reads the same fluid: the momentum balance takes the dynamic viscosity, a
turbulence closure's transport equations the kinematic one, and each needs the density. A case file
states the fluid in one place so that the two viscosities cannot come from two numbers that disagree.
Either viscosity may be given -- whichever the source it was taken from quotes -- and the other follows
from the density.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp

from aquaflux.properties import Constant, PropertyModel

__all__ = ["Fluid"]


@dataclasses.dataclass(frozen=True)
class Fluid:
    """A fluid of constant density and molecular viscosity.

    Exactly one of the two viscosities is given; the other is ``density`` times or divided by it.

    Attributes
    ----------
    density : float
        The density ``rho``.
    kinematic_viscosity : float or None
        The kinematic viscosity ``nu``, or ``None`` when the dynamic one is given.
    dynamic_viscosity : float or None
        The dynamic viscosity ``mu = rho nu``, or ``None`` when the kinematic one is given.

    Raises
    ------
    ValueError
        If neither viscosity or both are given, or if the density or the viscosity given is not a
        positive, finite number.

    Examples
    --------
    >>> Fluid(density=998.0, kinematic_viscosity=1.0e-6)
    Fluid(density=998.0, kinematic_viscosity=1e-06, dynamic_viscosity=None)
    """

    density: float
    kinematic_viscosity: float | None = None
    dynamic_viscosity: float | None = None

    def __post_init__(self) -> None:
        given = {
            name: value
            for name in ("kinematic_viscosity", "dynamic_viscosity")
            if (value := getattr(self, name)) is not None
        }
        if len(given) != 1:
            raise ValueError(
                "a fluid states exactly one of kinematic_viscosity and dynamic_viscosity, got "
                f"{'both' if given else 'neither'}. The other follows from the density, so stating both "
                "would be two numbers free to disagree."
            )
        for name, value in {"density": self.density, **given}.items():
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"the fluid's {name} must be a positive, finite number, got {value!r}."
                )

    def property_model(self) -> PropertyModel:
        """The ``"viscosity"`` (dynamic) and ``"density"`` properties every equation of a case reads.

        One model, handed to the flow and to a turbulence closure alike, so the two cannot describe
        different fluids. The viscosity is held as an array rather than a Python number: a Reynolds
        continuation rescales it at each step, and as an array leaf a rescaled value keeps the solve's
        compiled code, where a changed Python number would recompile it. Nothing rescales the density,
        so it stays a number.

        Returns
        -------
        PropertyModel
            ``{"viscosity": mu, "density": rho}``, with ``mu = rho nu`` when the kinematic viscosity was
            given.
        """
        viscosity = (
            self.density * self.kinematic_viscosity
            if self.dynamic_viscosity is None
            else self.dynamic_viscosity
        )
        return PropertyModel(
            {"viscosity": Constant(jnp.asarray(viscosity)), "density": Constant(self.density)}
        )
