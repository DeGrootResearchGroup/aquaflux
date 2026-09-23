"""A frozen value written as a plain mapping by kind, read back unchanged, and malformed mappings refused."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal

import numpy as np
import pytest
from aquaflux.solve import SettingsMapping


@dataclasses.dataclass(frozen=True)
class _Smoother:
    sweeps: int | None = None
    omega: float | None = None


@dataclasses.dataclass(frozen=True)
class _Hierarchy:
    smoother: _Smoother
    reach: tuple[int, ...] | None = None
    nested: _Smoother = dataclasses.field(default_factory=_Smoother)


@dataclasses.dataclass(frozen=True)
class _Chain:
    """A value holding a tuple of nested values."""

    smoothers: tuple[_Smoother, ...] = ()


@dataclasses.dataclass(frozen=True)
class _Derived:
    """A value with a field its constructor computes rather than takes."""

    size: int | None = None
    doubled: int = dataclasses.field(init=False, default=0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "doubled", 2 * (self.size or 0))


_MAPPING = SettingsMapping([_Smoother, _Hierarchy, _Chain, _Derived])


def test_a_field_at_its_default_is_omitted_and_a_required_field_is_always_written() -> None:
    assert _MAPPING.to_mapping(_Smoother()) == {"kind": "_Smoother"}
    assert _MAPPING.to_mapping(_Hierarchy(_Smoother())) == {
        "kind": "_Hierarchy",
        "smoother": {"kind": "_Smoother"},
    }


def test_nested_values_become_nested_mappings_and_tuples_become_lists() -> None:
    value = _Hierarchy(_Smoother(sweeps=2), reach=(3, 2), nested=_Smoother(omega=0.5))
    mapping = _MAPPING.to_mapping(value)
    assert mapping == {
        "kind": "_Hierarchy",
        "smoother": {"kind": "_Smoother", "sweeps": 2},
        "reach": [3, 2],
        "nested": {"kind": "_Smoother", "omega": 0.5},
    }
    assert _MAPPING.from_mapping(mapping) == value
    assert isinstance(_MAPPING.from_mapping(mapping).reach, tuple)


def test_nested_values_inside_a_tuple_are_written_and_read_as_values() -> None:
    value = _Chain((_Smoother(sweeps=1), _Smoother()))
    mapping = _MAPPING.to_mapping(value)
    assert mapping == {
        "kind": "_Chain",
        "smoothers": [{"kind": "_Smoother", "sweeps": 1}, {"kind": "_Smoother"}],
    }
    assert _MAPPING.from_mapping(mapping) == value


def test_a_field_the_constructor_computes_is_neither_written_nor_read() -> None:
    """It is not a setting: writing it would make the mapping unreadable, and it cannot be passed in."""
    assert _MAPPING.to_mapping(_Derived(size=2)) == {"kind": "_Derived", "size": 2}
    with pytest.raises(ValueError, match="_Derived has no field 'doubled'"):
        _MAPPING.from_mapping({"kind": "_Derived", "doubled": 4})


def test_an_omitted_field_takes_the_class_default() -> None:
    assert _MAPPING.from_mapping({"kind": "_Hierarchy", "smoother": {"kind": "_Smoother"}}) == (
        _Hierarchy(_Smoother())
    )


def test_the_kinds_are_the_classes_given_in_order() -> None:
    assert _MAPPING.kinds == (_Smoother, _Hierarchy, _Chain, _Derived)


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ({"sweeps": 2}, "names no 'kind'"),
        ({"kind": "Chebyshev"}, "unknown kind 'Chebyshev'"),
        ({"kind": 3}, "unknown kind 3"),
        ({"kind": ["_Smoother"]}, r"unknown kind \['_Smoother'\]"),
        ({"kind": "_Smoother", "sweep": 2}, r"_Smoother has no field 'sweep'; its fields are"),
        (
            {"kind": "_Hierarchy", "smoother": {"kind": "Chebyshev"}},
            "unknown kind 'Chebyshev' at 'smoother'",
        ),
        (
            {"kind": "_Hierarchy", "smoother": {"kind": "_Smoother", "sweep": 2}},
            "_Smoother at 'smoother' has no field 'sweep'",
        ),
        (
            {"kind": "_Chain", "smoothers": [{"kind": "_Smoother"}, {"kind": "Chebyshev"}]},
            r"unknown kind 'Chebyshev' at 'smoothers\[1\]'",
        ),
        ("_Smoother", "expected a mapping"),
    ],
)
def test_a_malformed_mapping_is_refused_naming_the_entry(mapping, match) -> None:
    with pytest.raises(ValueError, match=match):
        _MAPPING.from_mapping(mapping)


def test_a_value_of_a_kind_the_mapping_does_not_accept_is_refused_on_writing() -> None:
    @dataclasses.dataclass(frozen=True)
    class _Other:
        pass

    with pytest.raises(TypeError, match="_Other at 'smoother' is not a kind this mapping accepts"):
        _MAPPING.to_mapping(_Hierarchy(_Other()))  # type: ignore[arg-type]


def test_a_different_class_with_an_accepted_name_is_refused_on_writing() -> None:
    """Accepted by name alone, it would be written as a kind that reads back as another class."""

    @dataclasses.dataclass(frozen=True)
    class _Smoother:  # the same name as the accepted module-level class
        sweeps: int | None = None

    with pytest.raises(TypeError, match="_Smoother is not a kind this mapping accepts"):
        _MAPPING.to_mapping(_Smoother(sweeps=2))


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (_Hierarchy(_Smoother(sweeps=np.int64(2))), "int64 at 'smoother.sweeps' is not plain data"),
        (_Smoother(omega=np.float32(0.5)), "float32 at 'omega' is not plain data"),
        (
            _Hierarchy(_Smoother(), reach=(3, np.int64(2))),
            r"int64 at 'reach\[1\]' is not plain data",
        ),
        (_Smoother(omega=len), "builtin_function_or_method at 'omega' is not plain data"),
    ],
    ids=["numpy-int", "numpy-float", "numpy-in-a-tuple", "callable"],
)
def test_a_setting_that_is_not_plain_data_is_refused_on_writing(value, match) -> None:
    """A mapping a YAML or JSON writer cannot store is refused where it is made, naming the setting."""
    with pytest.raises(TypeError, match=match):
        _MAPPING.to_mapping(value)


def test_a_kind_must_be_a_dataclass_with_a_distinct_name() -> None:
    with pytest.raises(TypeError, match="must be a dataclass"):
        SettingsMapping([int])

    @dataclasses.dataclass(frozen=True)
    class _Smoother:  # the same name as the module-level class
        pass

    with pytest.raises(ValueError, match="share the name '_Smoother'"):
        SettingsMapping([globals()["_Smoother"], _Smoother])


# --- what may appear in each field -------------------------------------------------------------
#
# Until 2026-09-23 a mapping checked the `kind` and the field NAMES and nothing about the values, so a
# misspelt setting loaded and failed much later where the value is consumed -- or never, if the
# consumer ignored it (issue #424). The rules come from the fields' own annotations, so there is no
# second table of them to drift from the dataclass.


@dataclasses.dataclass(frozen=True)
class _Typed:
    """One field of each form the rules are read from."""

    count: int | None = None
    scale: float | None = None
    name: str | None = None
    flag: bool | None = None
    mode: Literal["fast", "slow"] | None = None
    reach: tuple[int, ...] | None = None
    smoother: _Smoother | None = None


_TYPED = SettingsMapping([_Typed, _Smoother])


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ({"count": True}, r"True at 'count' is not accepted there"),
        ({"count": "3"}, r"'3' at 'count' is not accepted there"),
        ({"count": 2.5}, r"2.5 at 'count' is not accepted there"),
        ({"flag": 1}, r"1 at 'flag' is not accepted there"),
        ({"scale": True}, r"True at 'scale' is not accepted there"),
        ({"name": 3}, r"3 at 'name' is not accepted there"),
        ({"mode": "quick"}, r"'quick' at 'mode' is not accepted there"),
        ({"reach": [1, "2"]}, r"at 'reach' is not accepted there"),
        ({"smoother": 2}, r"2 at 'smoother' is not accepted there"),
        ({"count": {"kind": "_Smoother"}}, r"at 'count' is not accepted there"),
    ],
    ids=[
        "bool-for-int",
        "string-for-int",
        "fraction-for-int",
        "int-for-bool",
        "bool-for-float",
        "int-for-string",
        "value-outside-a-literal",
        "string-inside-an-int-list",
        "number-for-a-nested-value",
        "nested-value-for-a-number",
    ],
)
def test_a_setting_of_the_wrong_form_is_refused_where_it_appears(mapping, match) -> None:
    """Named by path, with what that field takes -- the half that was missing.

    ``bool`` is a subclass of ``int`` in Python, so a boolean passes an ``isinstance`` check for a
    count; that is why the two are pinned in both directions here.
    """
    with pytest.raises(ValueError, match=match):
        _TYPED.from_mapping({"kind": "_Typed", **mapping})


@pytest.mark.parametrize(
    "mapping",
    [
        {"count": 3},
        {"count": 3.0},  # a parser may hand back a whole number as a float
        {"scale": 2},  # ... and an integer where a number is wanted
        {"scale": 2.5},
        {"flag": False},
        {"mode": "slow"},
        {"reach": [3, 2]},
        {"smoother": {"kind": "_Smoother", "sweeps": 2}},
        {"count": None, "smoother": None},  # an explicit null is the same as leaving a key out
    ],
    ids=[
        "int",
        "whole-float-for-int",
        "int-for-float",
        "float",
        "bool",
        "literal",
        "list",
        "nested",
        "null",
    ],
)
def test_a_setting_of_the_right_form_still_loads(mapping) -> None:
    """The other half: the check must not refuse what a file legitimately says.

    The whole-float case is load-bearing rather than incidental -- a probe's per-column reach is
    written by this package's own writer and read back through a JSON round trip.
    """
    assert _TYPED.from_mapping({"kind": "_Typed", **mapping}) is not None


def test_the_message_names_what_the_field_takes() -> None:
    """A refusal that says only 'no' leaves the reader to find the choices in the source."""
    with pytest.raises(ValueError, match=r"_Typed.mode takes one of 'fast', 'slow' or null"):
        _TYPED.from_mapping({"kind": "_Typed", "mode": "quick"})
    with pytest.raises(ValueError, match=r"_Typed.smoother takes one of '_Smoother' or null"):
        _TYPED.from_mapping({"kind": "_Typed", "smoother": 2})


def test_an_unknown_kind_is_reported_as_unknown_not_as_misplaced() -> None:
    """A kind nothing knows is misspelt, not in the wrong place, and the message should say so.

    Both are refusals, so it would be easy to let the position check answer first; it would then
    report what belongs in the field and leave the reader looking for a class that does not exist.
    """
    with pytest.raises(ValueError, match=r"unknown kind 'Chebyshev' at 'smoother'"):
        _TYPED.from_mapping({"kind": "_Typed", "smoother": {"kind": "Chebyshev"}})


def test_a_field_whose_annotation_cannot_be_checked_fails_when_the_mapping_is_built() -> None:
    """Not when a file is read -- and deliberately not by skipping it.

    A field the rules cannot express would otherwise load unchecked, which is the silence this whole
    check exists to remove, and it would be invisible: a mapping that validates nothing looks exactly
    like a mapping whose values are all valid. The mappings in this package are module-level
    constants, so this refusal fires at import rather than at the first load.
    """

    @dataclasses.dataclass(frozen=True)
    class _Opaque:
        callback: Callable[[int], int] | None = None

    with pytest.raises(TypeError, match=r"_Opaque.callback is annotated .* cannot check"):
        SettingsMapping([_Opaque])
