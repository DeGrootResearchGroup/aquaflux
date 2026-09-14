"""A frozen value of optional settings, where ``None`` means "not set here".

Several configuration objects in the solver describe a family of settings for something another class
builds -- a block inverse, a coupled preconditioner. Each is written the same way: a frozen dataclass
whose every field defaults to ``None``, so that constructing one with a single field changes that
setting and leaves every other one to the default of the class that consumes it. The default is then
written down in exactly one place, beside the reasoning for it, rather than restated by each
configuration object that can reach it.
"""

from __future__ import annotations

import dataclasses

__all__ = ["SettingsValue"]


@dataclasses.dataclass(frozen=True)
class SettingsValue:
    """A frozen value whose ``None`` fields are unset, and whose set fields are its settings.

    Subclasses are frozen dataclasses, so they compare and hash by value and can be stored, compared
    and written in a case description.
    """

    def settings(self) -> dict[str, object]:
        """The fields this value sets, by name -- unset (``None``) fields omitted.

        Returns
        -------
        dict
            The set fields, ready to pass as keyword arguments to the class they configure.
        """
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if (value := getattr(self, field.name)) is not None
        }
