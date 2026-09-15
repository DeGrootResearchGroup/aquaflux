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
from typing import TypeVar

__all__ = ["SettingsValue", "filled_from"]

_Value = TypeVar("_Value")


@dataclasses.dataclass(frozen=True)
class SettingsValue:
    """A frozen value whose ``None`` fields are unset, and whose set fields are its settings.

    Subclasses are frozen dataclasses, so they compare and hash by value and can be stored, compared
    and written in a case description.

    A subclass may be the abstract base of a family of values -- one that also derives from
    :class:`abc.ABC` and leaves an abstract method unimplemented. Constructing such a base raises at
    once, naming the family's public concrete members. Otherwise it would pass an ``isinstance`` check
    wherever the family is required, and fail only later, when something tries to build from it.

    Raises
    ------
    TypeError
        If an abstract subclass is constructed.
    """

    def __new__(cls, *args: object, **kwargs: object) -> SettingsValue:
        if getattr(cls, "__abstractmethods__", None):
            raise TypeError(f"{cls.__name__} is abstract; construct {_concrete_members(cls)}.")
        return super().__new__(cls)

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

    def filled_from(self, base: _Value) -> _Value:
        """This value, with each field it leaves unset taken from ``base`` (see :func:`filled_from`)."""
        return filled_from(self, base)


def filled_from(value: _Value, base: _Value) -> _Value:
    """``value``, with each field it leaves unset (``None``) taken from ``base``.

    How two partial configurations of one kind combine: a builder's own default beneath a caller's
    setting, or a continuation's shared settings beneath one point's own. It is written once for every
    such value -- a :class:`SettingsValue` and the settings objects that are ``equinox`` modules alike,
    since both are dataclasses.

    ⚠️ **``None`` means "take ``base``'s", so it cannot ask for the class default back.** Over a ``base``
    that sets a field, a value leaving that field unset gets ``base``'s setting, not the default of the
    class the value configures. Where that default is a number it can be written out; a field whose
    default is ``None`` itself cannot be restored this way.

    Parameters
    ----------
    value, base : dataclass instance
        Two instances of the same class; ``value`` takes precedence.

    Returns
    -------
    same type as ``value``
        A copy whose set fields are ``value``'s and whose unset fields are ``base``'s.

    Raises
    ------
    TypeError
        If ``base`` is not an instance of exactly ``value``'s class.
    """
    if type(base) is not type(value):
        raise TypeError(
            f"cannot fill a {type(value).__name__} from a {type(base).__name__}: the two must be one class."
        )
    return dataclasses.replace(
        value,
        **{
            field.name: getattr(base, field.name)
            for field in dataclasses.fields(value)
            if field.init and getattr(value, field.name) is None
        },
    )


def _concrete_members(base: type) -> str:
    """The public, concrete classes derived from ``base``, as a list of constructor calls to offer."""
    found: set[str] = set()
    pending = list(base.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if not getattr(cls, "__abstractmethods__", None) and not cls.__name__.startswith("_"):
            found.add(f"{cls.__name__}()")
    names = sorted(found)
    if not names:
        return "a concrete subclass"
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"
