"""Frozen configuration values written as plain mappings, and read back.

A configuration value -- a block inverse, a coupled-preconditioner spec -- is a frozen dataclass, so it
can be described in a case file rather than assembled in code. The file's form is a nested mapping of
the kind a YAML or JSON document parses to: each value is a mapping whose ``kind`` key names its class,
and whose other keys are the fields it sets. A field left at its default is left out, so reading a
mapping back reproduces the value exactly, and a mapping that omits a field leaves that field to the
default of the class that consumes it.

Nothing here parses a file. It works on the mapping a parser produces, so it adds no parsing dependency
and does not care which format the mapping came from.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

__all__ = ["SettingsMapping"]

#: The key that names a value's class in its mapping.
KIND = "kind"


class SettingsMapping:
    """Converts frozen configuration values to and from plain mappings, by kind.

    Each accepted class is named in a mapping by its class name, under the ``kind`` key. A field holding
    another accepted value is written as a nested mapping; a tuple is written as a list and read back as
    a tuple; anything else is written as it is.

    Parameters
    ----------
    kinds : iterable of type
        The frozen dataclasses a mapping may name. Class names must be distinct.

    Raises
    ------
    TypeError
        If a kind is not a dataclass.
    ValueError
        If two kinds share a class name.

    Examples
    --------
    >>> import dataclasses
    >>> @dataclasses.dataclass(frozen=True)
    ... class Smoother:
    ...     sweeps: int | None = None
    >>> mapping = SettingsMapping([Smoother])
    >>> mapping.to_mapping(Smoother(sweeps=2))
    {'kind': 'Smoother', 'sweeps': 2}
    >>> mapping.from_mapping({"kind": "Smoother"})
    Smoother(sweeps=None)
    """

    def __init__(self, kinds: Iterable[type]) -> None:
        by_name: dict[str, type] = {}
        for kind in kinds:
            if not (isinstance(kind, type) and dataclasses.is_dataclass(kind)):
                raise TypeError(f"a settings kind must be a dataclass, got {kind!r}.")
            if kind.__name__ in by_name:
                raise ValueError(f"two settings kinds share the name {kind.__name__!r}.")
            by_name[kind.__name__] = kind
        self._by_name = by_name

    @property
    def kinds(self) -> tuple[type, ...]:
        """The accepted classes, in the order they were given."""
        return tuple(self._by_name.values())

    def to_mapping(self, value: object) -> dict[str, object]:
        """The mapping that describes ``value``, omitting every field left at its default.

        Parameters
        ----------
        value : object
            An instance of one of the accepted kinds.

        Returns
        -------
        dict
            ``{"kind": <class name>, <field>: <value>, ...}``, with nested values as nested mappings and
            tuples as lists.

        Raises
        ------
        TypeError
            If ``value``, or a value nested in it, is not one of the accepted kinds.
        """
        if self._by_name.get(type(value).__name__) is not type(value):
            raise TypeError(
                f"{type(value).__name__} is not a kind this mapping accepts; accepted kinds are "
                f"{sorted(self._by_name)}."
            )
        mapping: dict[str, object] = {KIND: type(value).__name__}
        for field in dataclasses.fields(value):
            if not field.init:
                continue
            setting = getattr(value, field.name)
            if setting != _default(field):
                mapping[field.name] = self._encode(setting)
        return mapping

    def from_mapping(self, mapping: Mapping[str, object]) -> object:
        """The value a mapping describes, with every omitted field at its class default.

        Parameters
        ----------
        mapping : mapping
            ``{"kind": <class name>, <field>: <value>, ...}``. A nested mapping is read as a nested value
            and so needs its own ``kind``; a list is read as a tuple.

        Returns
        -------
        object
            An instance of the kind named. Its constructor runs as usual, so a value it refuses -- a
            nested value of the wrong kind for its field, say -- raises its own error.

        Raises
        ------
        ValueError
            If a mapping names no kind, an unknown kind, or a field its kind does not have. The message
            gives the path to the offending entry.
        """
        return self._decode_value(mapping, path="")

    def _encode(self, setting: object) -> object:
        if dataclasses.is_dataclass(setting) and not isinstance(setting, type):
            return self.to_mapping(setting)
        if isinstance(setting, tuple):
            return [self._encode(item) for item in setting]
        return setting

    def _decode(self, setting: object, path: str) -> object:
        if isinstance(setting, Mapping):
            return self._decode_value(setting, path)
        if isinstance(setting, list | tuple):
            return tuple(self._decode(item, f"{path}[{i}]") for i, item in enumerate(setting))
        return setting

    def _decode_value(self, mapping: object, path: str) -> object:
        where = f" at {path!r}" if path else ""
        if not isinstance(mapping, Mapping):
            raise ValueError(f"expected a mapping with a {KIND!r}{where}, got {mapping!r}.")
        if KIND not in mapping:
            raise ValueError(
                f"the mapping{where} names no {KIND!r}; accepted kinds are {sorted(self._by_name)}."
            )
        name = mapping[KIND]
        kind = self._by_name.get(name) if isinstance(name, str) else None
        if kind is None:
            raise ValueError(
                f"unknown {KIND} {name!r}{where}; accepted kinds are {sorted(self._by_name)}."
            )
        fields = {field.name for field in dataclasses.fields(kind) if field.init}
        unknown = sorted(set(mapping) - fields - {KIND})
        if unknown:
            raise ValueError(
                f"{name}{where} has no field {', '.join(repr(u) for u in unknown)}; its fields are "
                f"{sorted(fields)}."
            )
        settings = {
            key: self._decode(setting, f"{path}.{key}" if path else key)
            for key, setting in mapping.items()
            if key != KIND
        }
        return kind(**settings)


def _default(field: dataclasses.Field) -> object:
    """A field's default, or ``MISSING`` for a required field -- which is then always written."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING
