"""A frozen value written as a plain mapping by kind, read back unchanged, and malformed mappings refused."""

from __future__ import annotations

import dataclasses

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


_MAPPING = SettingsMapping([_Smoother, _Hierarchy])


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


def test_an_omitted_field_takes_the_class_default() -> None:
    assert _MAPPING.from_mapping({"kind": "_Hierarchy", "smoother": {"kind": "_Smoother"}}) == (
        _Hierarchy(_Smoother())
    )


def test_the_kinds_are_the_classes_given_in_order() -> None:
    assert _MAPPING.kinds == (_Smoother, _Hierarchy)


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ({"sweeps": 2}, "names no 'kind'"),
        ({"kind": "Chebyshev"}, "unknown kind 'Chebyshev'"),
        ({"kind": "_Smoother", "sweep": 2}, r"_Smoother has no field 'sweep'; its fields are"),
        (
            {"kind": "_Hierarchy", "smoother": {"kind": "Chebyshev"}},
            "unknown kind 'Chebyshev' at 'smoother'",
        ),
        (
            {"kind": "_Hierarchy", "smoother": {"kind": "_Smoother", "sweep": 2}},
            "_Smoother at 'smoother' has no field 'sweep'",
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

    with pytest.raises(TypeError, match="_Other is not a kind this mapping accepts"):
        _MAPPING.to_mapping(_Hierarchy(_Other()))  # type: ignore[arg-type]


def test_a_kind_must_be_a_dataclass_with_a_distinct_name() -> None:
    with pytest.raises(TypeError, match="must be a dataclass"):
        SettingsMapping([int])

    @dataclasses.dataclass(frozen=True)
    class _Smoother:  # the same name as the module-level class
        pass

    with pytest.raises(ValueError, match="share the name '_Smoother'"):
        SettingsMapping([globals()["_Smoother"], _Smoother])
