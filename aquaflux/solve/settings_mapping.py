"""Frozen configuration values written as plain mappings, and read back.

A configuration value -- a block inverse, a coupled-preconditioner spec -- is a frozen dataclass, so it
can be described in a case file rather than assembled in code. The file's form is a nested mapping of
the kind a YAML or JSON document parses to: each value is a mapping whose ``kind`` key names its class,
and whose other keys are the fields it sets. A field left at its default is left out, so reading a
mapping back reproduces the value exactly, and a mapping that omits a field leaves that field to the
default of the class that consumes it. Everything else in it is plain data -- strings, numbers,
booleans, ``None`` and lists of them -- so any YAML or JSON writer can store it.

A field may also be a **table**: a mapping from names the file chooses to entries of one form, such as
a boundary condition per patch. Its keys are names rather than fields, so a table carries no ``kind``
of its own; the field's ``Mapping[str, ...]`` annotation is what says a mapping in that position is
one.

Reading one is checked: a mapping must name a class this accepts, every key must be a field of it, and
every setting must be something that field can hold -- a number where a number belongs, one of a fixed
set of names where the field offers a choice, a nested value of a kind usable in that position. What
each field takes is read from its own annotation, so the rule is stated once, where the field is
declared. A setting that fails is refused with the path to it, rather than loading and failing wherever
it is eventually consumed -- or being ignored, which is indistinguishable from never having written it.

Nothing here parses a file. It works on the mapping a parser produces, so it adds no parsing dependency
and does not care which format the mapping came from.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Iterable, Mapping

__all__ = ["SettingsMapping"]

#: The key that names a value's class in its mapping.
KIND = "kind"


class SettingsMapping:
    """Converts frozen configuration values to and from plain mappings, by kind.

    Each accepted class is named in a mapping by its class name, under the ``kind`` key. A field holding
    another accepted value is written as a nested mapping; a tuple is written as a list and read back as
    a tuple; a table (a field annotated ``Mapping[str, ...]``) is written as a mapping of its entries by
    name and read back as a read-only mapping; a string, number, boolean or ``None`` is written as it
    is. Nothing else is written, so the mapping is always plain data.

    Parameters
    ----------
    kinds : iterable of type
        The frozen dataclasses a mapping may name. Class names must be distinct.

    Raises
    ------
    TypeError
        If a kind is not a dataclass, or if one of its fields is annotated with a form the per-position
        rules cannot express (see :func:`_atoms`) -- which would otherwise load unchecked.
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
        # Read every field's annotation now, so a field this cannot check stops the mapping being
        # built rather than loading unvalidated (see `_atoms`).
        self._accepts = {
            name: {
                field.name: _atoms(hints[field.name], name, field.name)
                for field in dataclasses.fields(kind)
                if field.init
            }
            for name, kind in by_name.items()
            for hints in (typing.get_type_hints(kind),)
        }

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
            tuples as lists: plain data, ready for a YAML or JSON writer.

        Raises
        ------
        TypeError
            If ``value``, or a value nested in it, is not one of the accepted classes -- the class
            itself, not merely one of the same name -- or if a setting is not plain data (a string,
            number, boolean or ``None``, or a list or tuple of those or of accepted values). A numpy
            scalar is refused, for example. The message gives the path to the offending setting.
        """
        return self._to_mapping(value, path="")

    def from_mapping(self, mapping: Mapping[str, object]) -> object:
        """The value a mapping describes, with every omitted field at its class default.

        Parameters
        ----------
        mapping : mapping
            ``{"kind": <class name>, <field>: <value>, ...}``. A nested mapping is read as a nested value
            and so needs its own ``kind`` -- except in a table's position, where it is read as the
            table, each entry by its name; a list is read as a tuple.

        Returns
        -------
        object
            An instance of the kind named. Its constructor runs as usual, so a value it refuses -- a
            nested value of the wrong kind for its field, say -- raises its own error.

        Raises
        ------
        ValueError
            If a mapping names no kind, an unknown kind, a field its kind does not have, or a setting
            that field cannot hold -- a misspelt choice, a boolean where a count belongs, a nested value
            where a string belongs, or a value of a kind that is not usable in that position. The
            message gives the path to the offending entry and what that field takes.
        """
        return self._decode_value(mapping, path="")

    def _names_no_known_kind(self, setting: object) -> bool:
        """Whether ``setting`` holds a nested mapping whose ``kind`` this mapping has never heard of.

        A list is searched too, so an unknown kind inside one is reported as the misspelling it is
        rather than as the whole list being unacceptable.
        """
        if isinstance(setting, Mapping):
            return setting.get(KIND) not in self._by_name
        if isinstance(setting, list | tuple):
            return any(self._names_no_known_kind(item) for item in setting)
        return False

    def _to_mapping(self, value: object, path: str) -> dict[str, object]:
        if self._by_name.get(type(value).__name__) is not type(value):
            raise TypeError(
                f"{type(value).__name__}{_where(path)} is not a kind this mapping accepts; accepted "
                f"kinds are {sorted(self._by_name)}."
            )
        mapping: dict[str, object] = {KIND: type(value).__name__}
        for field in dataclasses.fields(value):
            if not field.init:
                continue
            setting = getattr(value, field.name)
            if setting != _default(field):
                mapping[field.name] = self._encode(setting, _join(path, field.name))
        return mapping

    def _encode(self, setting: object, path: str) -> object:
        if dataclasses.is_dataclass(setting) and not isinstance(setting, type):
            return self._to_mapping(setting, path)
        if isinstance(setting, list | tuple):
            return [self._encode(item, f"{path}[{i}]") for i, item in enumerate(setting)]
        if isinstance(setting, Mapping):
            for entry in setting:
                if not isinstance(entry, str):
                    raise TypeError(
                        f"{entry!r}{_where(path)} is not a name: a table's entries are named by "
                        "strings."
                    )
            return {
                entry: self._encode(item, _join(path, entry)) for entry, item in setting.items()
            }
        if setting is None or isinstance(setting, str | bool | int | float):
            return setting
        raise TypeError(
            f"{type(setting).__name__}{_where(path)} is not plain data: a setting is written as a string, "
            "number, boolean or None, a list of those, a table of them by name, or a nested value of an "
            "accepted kind."
        )

    def _decode(self, setting: object, path: str) -> object:
        if isinstance(setting, Mapping):
            return self._decode_value(setting, path)
        if isinstance(setting, list | tuple):
            return tuple(self._decode(item, f"{path}[{i}]") for i, item in enumerate(setting))
        return setting

    def _decode_value(self, mapping: object, path: str) -> object:
        where = _where(path)
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
        accepts = self._accepts[name]
        # Every field is checked before any is decoded, so of two bad settings the shallower one is
        # reported rather than whichever a nested decode happens to reach first.
        for key, setting in mapping.items():
            if key != KIND:
                self._check(setting, accepts[key], _join(path, key), f"{name}.{key}")
        required = [
            field.name
            for field in dataclasses.fields(kind)
            if field.init and _default(field) is dataclasses.MISSING and field.name not in mapping
        ]
        if required:
            raise ValueError(
                f"{name}{where} needs {', '.join(repr(r) for r in required)}, which "
                f"{'has' if len(required) == 1 else 'have'} no default."
            )
        settings = {
            key: self._read(setting, accepts[key], _join(path, key), f"{name}.{key}")
            for key, setting in mapping.items()
            if key != KIND
        }
        try:
            return kind(**settings)
        except (ValueError, TypeError) as error:
            # A value's own refusal knows the value, not where it sits in the file; for a nested one,
            # say where. Only these two exact types are re-raised, since a subclass may not take a
            # message alone.
            if not path or type(error) not in (ValueError, TypeError):
                raise
            raise type(error)(f"{name}{where}: {error}") from error

    def _check(self, setting: object, atoms: tuple[object, ...], path: str, label: str) -> None:
        """Refuse ``setting`` unless one of ``atoms`` accepts it, naming what ``label`` takes.

        A table's entries are checked when the table is read, entry by entry, so that a bad entry is
        reported by its own name rather than as the whole table being unacceptable.
        """
        if isinstance(setting, Mapping) and _table_of(atoms) is not None:
            return
        if self._names_no_known_kind(setting):
            # A mapping naming a kind nothing knows is misspelt, not misplaced. Decoding it reports
            # exactly that, with the same path, which is more use than a list of what belongs here.
            return
        if not any(atom.accepts(setting, self._by_name) for atom in atoms):
            raise ValueError(
                f"{setting!r}{_where(path)} is not accepted there; {label} takes "
                f"{_describe(atoms, self._by_name)}."
            )

    def _read(self, setting: object, atoms: tuple[object, ...], path: str, label: str) -> object:
        """``setting`` decoded as the position ``atoms`` describes, once :meth:`_check` has passed it.

        The position decides the reading, not the setting's shape alone: a mapping is a table where the
        field is one and a nested value everywhere else, which is what lets a table hold an entry
        whose name happens to be ``kind``.
        """
        table = _table_of(atoms)
        if table is not None and isinstance(setting, Mapping):
            entries = {}
            for entry, item in setting.items():
                where = _join(path, str(entry))
                if not isinstance(entry, str):
                    raise ValueError(
                        f"{entry!r}{_where(path)} is not a name; {label} takes "
                        f"{_describe(atoms, self._by_name)}."
                    )
                self._check(item, table.atoms, where, f"each entry of {label}")
                entries[entry] = self._read(item, table.atoms, where, f"each entry of {label}")
            return types.MappingProxyType(entries)
        if isinstance(setting, list | tuple) and not self._names_no_known_kind(setting):
            sequence = next(
                atom
                for atom in atoms
                if isinstance(atom, _Sequence) and atom.accepts(setting, self._by_name)
            )
            return tuple(
                self._read(item, sequence.atoms, f"{path}[{i}]", label)
                for i, item in enumerate(setting)
            )
        return self._decode(setting, path)


# --- what may appear in one field, read off its annotation ------------------------------------


@dataclasses.dataclass(frozen=True)
class _Null:
    """``None`` is accepted: the field is optional, and an absent key means the same thing."""

    def accepts(self, setting: object, kinds: Mapping[str, type]) -> bool:
        del kinds
        return setting is None

    def describe(self, kinds: Mapping[str, type]) -> str:
        del kinds
        return "null"


@dataclasses.dataclass(frozen=True)
class _Choice:
    """One of a fixed set of values, from a ``Literal`` annotation."""

    values: tuple[object, ...]

    def accepts(self, setting: object, kinds: Mapping[str, type]) -> bool:
        del kinds
        # `==` alone would let `True` match a `1`, since `True == 1` in Python.
        return any(setting == value and type(setting) is type(value) for value in self.values)

    def describe(self, kinds: Mapping[str, type]) -> str:
        del kinds
        return "one of " + ", ".join(repr(value) for value in self.values)


@dataclasses.dataclass(frozen=True)
class _Scalar:
    """A number, string or boolean.

    ⚠️ **``bool`` is a subclass of ``int`` in Python**, so ``True`` would pass an ``isinstance`` test
    for a count — which is the misreading this check exists to catch (``smoother_sweeps: true`` reached
    the multigrid builder as ``1``). A boolean is therefore accepted only where a boolean is asked for.
    An ``int`` position also accepts a whole-numbered ``float``, because a parser may hand back ``3.0``
    where a file says ``3.0``, and the values that take one already round it (a probe's per-column
    reach).
    """

    type: type

    def accepts(self, setting: object, kinds: Mapping[str, type]) -> bool:
        del kinds
        if isinstance(setting, bool):
            return self.type is bool
        if self.type is float:
            return isinstance(setting, int | float)
        if self.type is int:
            return isinstance(setting, int) or (isinstance(setting, float) and setting.is_integer())
        return isinstance(setting, self.type)

    def describe(self, kinds: Mapping[str, type]) -> str:
        del kinds
        return {bool: "a boolean", int: "a whole number", float: "a number", str: "a string"}[
            self.type
        ]


@dataclasses.dataclass(frozen=True)
class _Nested:
    """A nested value: a mapping whose ``kind`` names a class usable in this position.

    ``base`` may be an abstract family base (a block inverse, say), in which case every registered kind
    deriving from it is accepted there and no other.
    """

    base: type

    def accepts(self, setting: object, kinds: Mapping[str, type]) -> bool:
        if not isinstance(setting, Mapping):
            return False
        name = setting.get(KIND)
        kind = kinds.get(name) if isinstance(name, str) else None
        return kind is not None and issubclass(kind, self.base)

    def usable(self, kinds: Mapping[str, type]) -> list[str]:
        """The registered kinds that may appear in this position."""
        return sorted(name for name, kind in kinds.items() if issubclass(kind, self.base))

    def describe(self, kinds: Mapping[str, type]) -> str:
        usable = self.usable(kinds)
        return f"one of {', '.join(repr(name) for name in usable)}" if usable else "no known kind"


@dataclasses.dataclass(frozen=True)
class _Table:
    """A mapping from names to entries each accepted by ``atoms``, from a ``Mapping[str, ...]`` annotation.

    Read back as a read-only mapping (:class:`types.MappingProxyType`), so a frozen value holding one
    cannot be changed through it. Its entries carry no ``kind`` of their own at this level: the names
    are the table's keys, and each entry is whatever ``atoms`` describes -- usually a nested value,
    which has its own ``kind``.
    """

    atoms: tuple[object, ...]

    def accepts(self, setting: object, kinds: Mapping[str, type]) -> bool:
        return isinstance(setting, Mapping) and all(
            isinstance(entry, str) and any(atom.accepts(item, kinds) for atom in self.atoms)
            for entry, item in setting.items()
        )

    def describe(self, kinds: Mapping[str, type]) -> str:
        return f"a table from names to ({_describe(self.atoms, kinds)})"


@dataclasses.dataclass(frozen=True)
class _Sequence:
    """A list (read back as a tuple) whose entries are each accepted by ``atoms``."""

    atoms: tuple[object, ...]

    def accepts(self, setting: object, kinds: Mapping[str, type]) -> bool:
        return isinstance(setting, list | tuple) and all(
            any(atom.accepts(item, kinds) for atom in self.atoms) for item in setting
        )

    def describe(self, kinds: Mapping[str, type]) -> str:
        return f"a list of ({_describe(self.atoms, kinds)})"


def _atoms(annotation: object, owner: str, field: str) -> tuple[object, ...]:
    """What ``annotation`` permits, as the alternatives a setting may satisfy.

    Read once, when a :class:`SettingsMapping` is built, so a field whose annotation this cannot
    express fails **there** rather than loading unchecked. That is deliberate: the mappings are
    module-level constants, so an annotation nothing can validate stops the package importing rather
    than opening a hole nobody sees. If one is ever wanted, widen this function.

    Parameters
    ----------
    annotation : object
        The resolved type hint of one field.
    owner, field : str
        The class and field the annotation belongs to, for the refusal's message.

    Returns
    -------
    tuple
        The alternatives, each with ``accepts`` and ``describe``.

    Raises
    ------
    TypeError
        If the annotation is not one of the forms above.
    """
    origin = typing.get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        atoms = tuple(
            atom for arm in typing.get_args(annotation) for atom in _atoms(arm, owner, field)
        )
        if any(isinstance(atom, _Table) for atom in atoms) and any(
            isinstance(atom, _Nested) for atom in atoms
        ):
            # Both are written as mappings, so the only thing telling them apart would be whether a
            # `kind` key is present -- and a table may well hold an entry of that name.
            raise TypeError(
                f"{owner}.{field} is annotated {annotation!r}: a field may be a table or a nested "
                "value, not either, since both are written as mappings."
            )
        return atoms
    if origin in (Mapping, dict):
        key, value = typing.get_args(annotation) or (None, None)
        if key is not str:
            raise TypeError(
                f"{owner}.{field} is annotated {annotation!r}: a table's entries are named by strings, "
                "so its key type must be str."
            )
        return (_Table(_atoms(value, owner, field)),)
    if annotation is type(None):
        return (_Null(),)
    if origin is typing.Literal:
        return (_Choice(typing.get_args(annotation)),)
    if origin is tuple:
        args = [arg for arg in typing.get_args(annotation) if arg is not Ellipsis]
        return (_Sequence(tuple(atom for arg in args for atom in _atoms(arg, owner, field))),)
    if annotation in (bool, int, float, str):
        return (_Scalar(annotation),)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return (_Nested(annotation),)
    raise TypeError(
        f"{owner}.{field} is annotated {annotation!r}, which a settings mapping cannot check. A field "
        "is a number, string, boolean, Literal, nested settings value, tuple of those, table of those "
        "by name (Mapping[str, ...]), or a union with None."
    )


def _table_of(atoms: Iterable[object]) -> _Table | None:
    """The table among ``atoms``, if the position is one (at most one can be -- see :func:`_atoms`)."""
    return next((atom for atom in atoms if isinstance(atom, _Table)), None)


def _describe(atoms: Iterable[object], kinds: Mapping[str, type]) -> str:
    """What the alternatives accept, for the message naming what a rejected setting should have been.

    The nested-value alternatives are merged into one list of kinds. A field annotated with a union of
    four value families has four of them, and describing each separately reads as four rules rather
    than as the one choice it is.
    """
    atoms = tuple(atoms)
    nested = sorted(
        {name for atom in atoms if isinstance(atom, _Nested) for name in atom.usable(kinds)}
    )
    parts = [atom.describe(kinds) for atom in atoms if not isinstance(atom, _Nested)]
    if nested:
        parts.insert(0, "one of " + ", ".join(repr(name) for name in nested))
    return " or ".join(parts)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _where(path: str) -> str:
    return f" at {path!r}" if path else ""


def _default(field: dataclasses.Field) -> object:
    """A field's default, or ``MISSING`` for a required field -- which is then always written."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING
