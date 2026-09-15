"""The block-inverse value objects: field sets pinned to the classes, and only set fields forwarded."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve import (
    AirReduction,
    BlockInverse,
    HierarchyBlockInverse,
    JacobiSmoothed,
    JacobiSmoothedInverse,
    SimpleSmoothed,
    SimpleSmoothedInverse,
    build_air_hierarchy,
)
from aquaflux.solve.field_split import AirBlockInverse


def _keywords(function, *, drop: set[str]) -> set[str]:
    parameters = inspect.signature(function).parameters.values()
    return {
        p.name
        for p in parameters
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.name not in drop
    }


def _fields(spec_class) -> set[str]:
    return {field.name for field in dataclasses.fields(spec_class)}


_HIERARCHY = _keywords(HierarchyBlockInverse.__init__, drop={"report"})


@pytest.mark.parametrize(
    ("spec_class", "expected"),
    [
        (SimpleSmoothed, _keywords(SimpleSmoothedInverse.__init__, drop=set()) | _HIERARCHY),
        (JacobiSmoothed, _keywords(JacobiSmoothedInverse.__init__, drop=set()) | _HIERARCHY),
        (
            AirReduction,
            _keywords(AirBlockInverse.__init__, drop=set())
            | _keywords(build_air_hierarchy, drop={"block_size"}),
        ),
    ],
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_each_value_names_exactly_the_settings_its_class_accepts(spec_class, expected) -> None:
    """A field the class does not accept, or a class keyword with no field, is drift to fix here."""
    assert _fields(spec_class) == expected


def test_only_the_fields_that_are_set_are_forwarded() -> None:
    """An unset field leaves the class's own default in force rather than restating it."""
    assert SimpleSmoothed().settings() == {}
    assert SimpleSmoothed(sweeps=2, frozen_coarsening=False).settings() == {
        "sweeps": 2,
        "frozen_coarsening": False,
    }


def test_values_compare_and_hash_by_their_settings() -> None:
    assert JacobiSmoothed(max_coarse=200) == JacobiSmoothed(max_coarse=200)
    assert hash(JacobiSmoothed(max_coarse=200)) == hash(JacobiSmoothed(max_coarse=200))
    assert JacobiSmoothed(max_coarse=200) != JacobiSmoothed(max_coarse=150)


def _two_field_block(n_cells: int = 40) -> sp.csr_matrix:
    upwind = sp.diags(
        [-np.full(n_cells - 1, 4.0), np.full(n_cells, 6.0), -np.full(n_cells - 1, 1.0)],
        [-1, 0, 1],
        format="csr",
    )
    cross = sp.identity(n_cells, format="csr") * 3.0
    return sp.bmat([[upwind, cross], [cross, upwind]], format="csr")


def test_calling_the_value_builds_the_inverse_with_its_settings() -> None:
    inverse = JacobiSmoothed(max_levels=3, max_coarse=8)(_two_field_block(), 2)
    assert isinstance(inverse, JacobiSmoothedInverse)
    assert len(inverse._hierarchy.levels) == 3


def test_bound_sends_the_build_record_to_the_sink() -> None:
    messages: list[str] = []
    SimpleSmoothed(max_coarse=8).bound(report=messages.append)(_saddle_block(), 3)
    assert messages, "the bound sink received no build record"


def test_a_family_with_no_build_record_refuses_a_sink() -> None:
    with pytest.raises(TypeError, match="no build record"):
        AirReduction().bound(report=print)


def _saddle_block(n_cells: int = 36) -> sp.csr_matrix:
    """A small field-major velocity-pressure block, velocity rows diagonally dominant."""
    rng = np.random.default_rng(0)
    n = 3 * n_cells
    a = sp.random(n, n, density=0.05, random_state=1, format="lil")
    a.setdiag(np.abs(rng.normal(size=n)) + 6.0)
    return sp.csr_matrix(a)


def test_the_abstract_base_is_refused_naming_the_block_inverse_values() -> None:
    """A bare ``BlockInverse()`` passed ``FieldSplit``'s ``isinstance`` refusal with nothing to build."""
    with pytest.raises(
        TypeError,
        match=r"BlockInverse is abstract; construct AirReduction\(\), JacobiSmoothed\(\) or SimpleSmoothed\(\)\.",
    ):
        BlockInverse()
