"""A frozen value written as a plain mapping by kind, read back unchanged, and malformed mappings refused."""

from __future__ import annotations

import dataclasses

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
