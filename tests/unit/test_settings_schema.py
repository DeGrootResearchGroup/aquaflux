"""Tests for :meth:`~aquaflux.solve.SettingsMapping.schema`, the description a form edits mappings by.

The schema is read off the same annotations reading checks against, so what is pinned here is that
each annotation form is described as the alternative reading accepts, that defaults are written as
:meth:`~aquaflux.solve.SettingsMapping.to_mapping` would write them, and that each field's help comes
from its own entry in its class's docstring (or a base class's). Also that a value's unset settings
are described by where they resolve to (``unset_resolves_to``), marked off (``unset_means_off``), and that
its internal state (``not_settings``) is neither described, written nor read.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, Literal

import pytest
from aquaflux.solve import SettingsMapping


@dataclasses.dataclass(frozen=True)
class Family:
    """A family of values usable in one position: a dataclass base, as the registry's are."""


@dataclasses.dataclass(frozen=True)
class First(Family):
    """The first member.

    Attributes
    ----------
    sweeps : int
        How many sweeps, a whole
        number over two lines.

        A second paragraph, detail for the reference documentation that the help leaves out.
    """

    sweeps: int = 2


@dataclasses.dataclass(frozen=True)
class Second(Family):
    """The second member, documenting nothing."""

    weight: float = 0.5


@dataclasses.dataclass(frozen=True)
class Holder:
    """Holds one of everything.

    Attributes
    ----------
    method : str
        A choice.
    low, high : float
        Two fields sharing one entry.
    member : Family or None
        A nested value of the family.

    Raises
    ------
    ValueError
        Never; here so the section above is known to end.
    """

    name: str
    method: Literal["jacobi", "gauss-seidel"] = "jacobi"
    low: float = 0.0
    high: float = 1.0
    flag: bool = False
    member: Family | None = None
    members: tuple[Family, ...] = ()
    table: Mapping[str, Family] = dataclasses.field(default_factory=dict)
    nested: First = dataclasses.field(default_factory=First)


SCHEMA = SettingsMapping([First, Second, Holder]).schema()


def _fields(schema, kind):
    return {field["name"]: field for field in schema["kinds"][kind]["fields"]}


def test_every_annotation_form_is_described_as_the_alternative_reading_accepts():
    fields = _fields(SCHEMA, "Holder")
    assert fields["name"]["accepts"] == [{"type": "string"}]
    assert fields["method"]["accepts"] == [{"type": "choice", "values": ["jacobi", "gauss-seidel"]}]
    assert fields["low"]["accepts"] == [{"type": "number"}]
    assert fields["flag"]["accepts"] == [{"type": "boolean"}]
    assert _fields(SCHEMA, "First")["sweeps"]["accepts"] == [{"type": "integer"}]
    # A family position lists the registered members, in registration order, and nothing else.
    assert fields["member"]["accepts"] == [
        {"type": "nested", "kinds": ["First", "Second"]},
        {"type": "null"},
    ]
    assert fields["members"]["accepts"] == [
        {"type": "list", "of": [{"type": "nested", "kinds": ["First", "Second"]}]}
    ]
    assert fields["table"]["accepts"] == [
        {"type": "table", "of": [{"type": "nested", "kinds": ["First", "Second"]}]}
    ]
    assert fields["nested"]["accepts"] == [{"type": "nested", "kinds": ["First"]}]


def test_a_default_is_written_as_writing_a_mapping_would_write_it():
    fields = _fields(SCHEMA, "Holder")
    assert fields["name"]["required"] is True and "default" not in fields["name"]
    assert (fields["method"]["default"], fields["low"]["default"]) == ("jacobi", 0.0)
    assert fields["members"]["default"] == [] and fields["table"]["default"] == {}
    # A nested default, through its default factory, as the mapping that names it.
    assert fields["nested"]["default"] == {"kind": "First"}
    assert fields["member"]["required"] is False and fields["member"]["default"] is None


def test_each_field_is_documented_by_its_own_entry_in_the_attributes_section():
    holder, first = _fields(SCHEMA, "Holder"), _fields(SCHEMA, "First")
    assert first["sweeps"]["doc"] == "How many sweeps, a whole number over two lines."
    assert holder["method"]["doc"] == "A choice."
    assert holder["low"]["doc"] == holder["high"]["doc"] == "Two fields sharing one entry."
    # The Raises section after Attributes is not read as more fields.
    assert "ValueError" not in holder and holder["flag"]["doc"] == ""
    assert _fields(SCHEMA, "Second")["weight"]["doc"] == ""
    assert SCHEMA["kinds"]["Second"]["summary"] == "The second member, documenting nothing."


def test_a_default_that_is_not_plain_data_is_not_shown():
    @dataclasses.dataclass(frozen=True)
    class WithFunction:
        """Holds a function by default."""

        choose: str = "a"
        callback: str | None = dataclasses.field(default_factory=lambda: print)

    (choose, callback) = SettingsMapping([WithFunction]).schema()["kinds"]["WithFunction"]["fields"]
    assert choose["default"] == "a"
    assert callback["required"] is False and "default" not in callback


class Smoother:
    """What a value's unset settings fall through to: a class whose constructor holds the defaults.

    Parameters
    ----------
    sweeps : int
        Smoothing sweeps per level.
    """

    def __init__(self, sweeps=3, omega=None, **forwarded):
        del sweeps, omega, forwarded


def coarsen(*, levels=4):
    """The function the class forwards its other settings to.

    Parameters
    ----------
    levels : int
        Levels in the hierarchy.
    """
    del levels


@dataclasses.dataclass(frozen=True)
class Defaults:
    """A dataclass whose default is built by a factory."""

    settings: First = dataclasses.field(default_factory=First)


@dataclasses.dataclass(frozen=True)
class Resolving:
    """A value whose every setting is ``None``, "not set here"."""

    sweeps: int | None = None
    levels: int | None = None
    omega: float | None = None
    settings: First | None = None
    budget: int | None = None
    untouched: int | None = None

    unset_resolves_to: ClassVar[tuple] = (Smoother, coarsen, Defaults)
    unset_means_off: ClassVar[tuple[str, ...]] = ("budget",)


def _described(*kinds):
    schema = SettingsMapping([*kinds]).schema()["kinds"]
    return {f["name"]: f for f in schema[kinds[0].__name__]["fields"]}


def test_an_unset_setting_is_described_by_the_default_it_falls_through_to():
    fields = _described(Resolving, First)
    assert fields["sweeps"]["resolved_default"] == 3  # the class's constructor
    assert fields["levels"]["resolved_default"] == 4  # past the class's ``**``, the next target
    assert fields["settings"]["resolved_default"] == {
        "kind": "First"
    }  # a dataclass factory, encoded
    # A parameter whose own default is None has nothing to state; one no target has stays not set.
    assert (
        "resolved_default" not in fields["omega"] and "resolved_default" not in fields["untouched"]
    )
    # The stored default is still the value's own, None: what reading fills in is unchanged.
    assert all(fields[name]["default"] is None for name in fields)


def test_an_unset_settings_help_comes_from_where_it_falls_through_to_when_it_has_none():
    fields = _described(Resolving, First)
    assert fields["sweeps"]["doc"] == "Smoothing sweeps per level."
    assert fields["levels"]["doc"] == "Levels in the hierarchy."


def test_a_setting_whose_unset_meaning_is_off_says_so_rather_than_resolving():
    fields = _described(Resolving, First)
    assert fields["budget"].get("off") is True and "resolved_default" not in fields["budget"]
    assert "off" not in fields["sweeps"]


@dataclasses.dataclass(frozen=True)
class Base:
    """A base class documenting the field it declares.

    Attributes
    ----------
    steps : int
        Steps, documented once on the base.
    """

    steps: int = 5


@dataclasses.dataclass(frozen=True)
class Derived(Base):
    """A derived class documenting only its own field.

    Attributes
    ----------
    rate : float
        Its own rate.
    """

    rate: float = 1.0


def test_a_field_a_base_class_declares_is_documented_from_the_base():
    fields = _described(Derived)
    assert fields["steps"]["doc"] == "Steps, documented once on the base."
    assert fields["rate"]["doc"] == "Its own rate."


@dataclasses.dataclass(frozen=True)
class Cached:
    """A value carrying a cache it fills for itself."""

    sweeps: int = 2
    prepared: object = None

    not_settings: ClassVar[tuple[str, ...]] = ("prepared",)


def test_a_values_internal_state_is_not_described_written_or_read():
    mapping = SettingsMapping([Cached])
    assert list(_described(Cached)) == ["sweeps"]
    assert mapping.to_mapping(Cached(sweeps=3, prepared="filled")) == {
        "kind": "Cached",
        "sweeps": 3,
    }
    with pytest.raises(ValueError, match="has no field 'prepared'"):
        mapping.from_mapping({"kind": "Cached", "prepared": "x"})


def test_every_case_file_kinds_declarations_name_real_fields_and_readable_targets():
    """A misspelt or renamed name would silently describe nothing, so each is checked against the code."""
    import inspect

    from aquaflux.case.spec import _CASE_MAPPING

    declared = 0
    for kind in _CASE_MAPPING.kinds:
        names = {field.name for field in dataclasses.fields(kind)}
        for attribute in ("unset_means_off", "not_settings"):
            listed = set(getattr(kind, attribute, ()))
            assert listed <= names, f"{kind.__name__}.{attribute} names {sorted(listed - names)}"
            declared += len(listed)
        targets = getattr(kind, "unset_resolves_to", ())
        for target in targets if isinstance(targets, tuple) else ():
            inspect.signature(target)  # raises if it describes nothing
            declared += 1
    assert declared > 40  # the declarations are really being read


def test_every_unset_block_inverse_setting_resolves_or_is_off():
    """The block inverses forward every setting, so none of them is left saying "not set"."""
    from aquaflux.case import case_schema

    kinds = case_schema()["kinds"]
    for name in ("SimpleSmoothed", "JacobiSmoothed", "AirReduction"):
        unresolved = [
            field["name"]
            for field in kinds[name]["fields"]
            if "resolved_default" not in field and not field.get("off")
        ]
        # restriction_theta follows theta when unset, which no single value states.
        assert unresolved == (["restriction_theta"] if name == "AirReduction" else []), name
