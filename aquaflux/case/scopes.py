"""Which part of a case reads a setting, declared once beside the setting.

A case file holds settings that only some physics read: a wall's ``k`` condition and the turbulence
advection scheme only a Reynolds-averaged case, a wall's reflectance only a radiation case, an inlet's
velocity only a flow. Each such setting names its **scope** in its kind's ``setting_scopes``, and each
physics names the scopes it reads in ``reads_scopes``. A physics refuses a setting it does not read,
and requires the ones its scopes require (``required_in_scope``). The case-file schema publishes the
same declarations, so a form can leave out exactly the settings a case's physics would refuse.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

__all__ = ["FLOW", "RADIATION", "SCOPES", "TURBULENCE", "Scoped"]

#: The settings only a flow reads: an inlet's velocity, an outlet's pressure, a wall's velocity.
FLOW = "flow"

#: The settings only a turbulence closure reads: an inlet's turbulence, a wall's ``k`` condition.
TURBULENCE = "turbulence"

#: The settings only a radiation case reads: a wall's reflectance, a lamp's profile.
RADIATION = "radiation"

#: Every scope, in the order a refusal names them.
SCOPES = (FLOW, TURBULENCE, RADIATION)


class Scoped:
    """A value whose settings each name the scope that reads them.

    Attributes
    ----------
    setting_scopes : mapping of {str: str}
        A class attribute: each scoped setting's name, and its scope. A setting not listed is read by
        every physics that reads the value at all.
    required_in_scope : tuple of str
        A class attribute: the scoped settings a physics reading their scope cannot do without.
    """

    setting_scopes: ClassVar[Mapping[str, str]] = {}
    required_in_scope: ClassVar[tuple[str, ...]] = ()

    def settings_in(self, scope: str) -> tuple[str, ...]:
        """The settings of ``scope`` this value states, in their declared order."""
        return tuple(
            name
            for name, of in self.setting_scopes.items()
            if of == scope and getattr(self, name) is not None
        )

    def missing_in(self, scope: str) -> tuple[str, ...]:
        """The settings ``scope`` requires that this value leaves unset."""
        return tuple(
            name
            for name in self.required_in_scope
            if self.setting_scopes[name] == scope and getattr(self, name) is None
        )
