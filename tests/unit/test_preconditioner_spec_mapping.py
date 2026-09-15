"""A coupled preconditioner spec read from and written to the mapping a case file parses to.

Every value a spec can hold, at any level, round-trips through a plain mapping; the mapping holds only
plain data; unknown kinds and fields are refused by name; and what is read is a spec, never a built
preconditioner.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import pickle
import pkgutil

import aquaflux
import numpy as np
import pytest
from aquaflux.flow import ConvectionAir, ConvectionTwoLevel, VelocityBlock, ViscousMultilevel
from aquaflux.solve import AirReduction, BlockInverse, JacobiSmoothed, SimpleSmoothed
from aquaflux.turbulence import (
    BlockDiagonal,
    CompleteLu,
    FieldSplit,
    JacobianProbeSpec,
    MaterializedJacobian,
    MonolithicVCycle,
    preconditioner_spec_from_mapping,
    preconditioner_spec_to_mapping,
)
from aquaflux.turbulence.preconditioner_spec import _SPEC_MAPPING

#: One spec per shape a case file can take, together holding every value class at least once.
_SPECS = [
    BlockDiagonal(),
    BlockDiagonal(method=None),
    BlockDiagonal(method="air", velocity=ViscousMultilevel()),
    BlockDiagonal(velocity=ConvectionAir(), v_cycles=2),
    BlockDiagonal(
        velocity=ConvectionTwoLevel(sweeps=3, omega=0.7),
        schur_scaling="msimple",
        composition="simpler",
        mass_scale=2.0,
        strength_threshold=0.25,
    ),
    MaterializedJacobian(CompleteLu()),
    MaterializedJacobian(CompleteLu(backend="scipy"), build_beta=0.5),
    MaterializedJacobian(
        MonolithicVCycle(smoother_fill_levels=0, smoother_sweeps=4, coarse_eq_limit=2000),
        beta_floor=0.05,
    ),
    MaterializedJacobian(
        FieldSplit(
            SimpleSmoothed(sweeps=2, strength_threshold=0.25, frozen_coarsening=True),
            JacobiSmoothed(max_coarse=200, prolongation_smoothing="jacobi"),
        ),
        probe=JacobianProbeSpec(column_reach=(3, 3, 3, 3, 2, 2), gradient_sweeps=2),
        beta_floor=0.05,
    ),
    MaterializedJacobian(
        FieldSplit(SimpleSmoothed(), AirReduction(theta=0.5, degree=1)),
        probe=JacobianProbeSpec(stencil_reach=3),
    ),
]


def _held_classes(value: object, into: set[type]) -> set[type]:
    into.add(type(value))
    for setting in vars(value).values():
        if type(setting) in _SPEC_MAPPING.kinds:
            _held_classes(setting, into)
    return into


def _exported_classes() -> set[type]:
    """Every class exported by any subpackage of ``aquaflux``, not only the ones a spec imports today."""
    exported: set[type] = set()
    for info in pkgutil.iter_modules(aquaflux.__path__):
        if info.ispkg:
            module = importlib.import_module(f"aquaflux.{info.name}")
            exported |= {
                member
                for name in getattr(module, "__all__", ())
                if inspect.isclass(member := getattr(module, name))
            }
    return exported


def _public_value_classes() -> set[type]:
    """Every concrete public value a spec can hold.

    The nested families -- velocity blocks and block inverses, which grow as methods are added -- are
    found from every subpackage's exports, so a new member exported anywhere is caught. The spec's own
    classes, which do not form an open family, are named here.
    """
    nested = {
        cls
        for cls in _exported_classes()
        if issubclass(cls, VelocityBlock | BlockInverse)
        and cls not in (VelocityBlock, BlockInverse)
        and not inspect.isabstract(cls)
    }
    spec_classes = {BlockDiagonal, MaterializedJacobian, CompleteLu, MonolithicVCycle, FieldSplit}
    return nested | spec_classes | {JacobianProbeSpec}


def test_the_mapping_accepts_every_public_value_a_spec_can_hold() -> None:
    """A value class added to a family but not to the mapping would be unwritable in a case file."""
    assert set(_SPEC_MAPPING.kinds) == _public_value_classes()


def test_the_census_scans_every_subpackage_not_only_the_ones_a_spec_imports() -> None:
    names = {info.name for info in pkgutil.iter_modules(aquaflux.__path__) if info.ispkg}
    assert {"flow", "solve", "turbulence", "transport"} <= names
    assert _exported_classes() >= {BlockDiagonal, SimpleSmoothed, ConvectionAir}


def test_the_round_trip_specs_hold_every_accepted_class() -> None:
    held: set[type] = set()
    for spec in _SPECS:
        _held_classes(spec, held)
    assert held == set(_SPEC_MAPPING.kinds)


@pytest.mark.parametrize("spec", _SPECS, ids=repr)
def test_every_spec_round_trips_through_plain_json_data(spec) -> None:
    mapping = preconditioner_spec_to_mapping(spec)
    assert preconditioner_spec_from_mapping(json.loads(json.dumps(mapping))) == spec


def test_a_column_reach_read_from_a_file_is_the_same_value_as_one_given_in_code() -> None:
    """A parser hands over a list of whatever numbers the file held; the spec stores integers either way."""
    read = preconditioner_spec_from_mapping(
        {
            "kind": "MaterializedJacobian",
            "inverse": {"kind": "CompleteLu"},
            "probe": {"kind": "JacobianProbeSpec", "column_reach": [3.0, 3.0, 2.0]},
        }
    )
    assert read.probe == JacobianProbeSpec(column_reach=(3, 3, 2))
    assert all(type(r) is int for r in read.probe.column_reach)


@pytest.mark.parametrize(
    "duplicate", [copy.copy, copy.deepcopy, lambda s: pickle.loads(pickle.dumps(s))]
)
def test_an_unset_scalar_method_survives_copying_and_pickling(duplicate) -> None:
    """The sentinel is compared by identity, so a copy that made a new one changed the spec's meaning."""
    duplicated = duplicate(BlockDiagonal())
    assert duplicated == BlockDiagonal()
    assert duplicated.resolved_method() == "twolevel"
    assert preconditioner_spec_to_mapping(duplicated) == {"kind": "BlockDiagonal"}


def test_a_numpy_scalar_setting_is_refused_on_writing_naming_the_setting() -> None:
    with pytest.raises(TypeError, match="float32 at 'build_beta' is not plain data"):
        preconditioner_spec_to_mapping(
            MaterializedJacobian(CompleteLu(), build_beta=np.float32(0.5))
        )


def test_the_file_form_of_a_field_split_spec() -> None:
    """Pins the format itself: a kind per level, defaults omitted, a reach tuple written as a list."""
    spec = MaterializedJacobian(
        FieldSplit(SimpleSmoothed(sweeps=2), JacobiSmoothed(max_coarse=200)),
        probe=JacobianProbeSpec(column_reach=(3, 3, 3, 3, 2, 2)),
        beta_floor=0.05,
    )
    assert preconditioner_spec_to_mapping(spec) == {
        "kind": "MaterializedJacobian",
        "inverse": {
            "kind": "FieldSplit",
            "leading": {"kind": "SimpleSmoothed", "sweeps": 2},
            "trailing": {"kind": "JacobiSmoothed", "max_coarse": 200},
        },
        "probe": {"kind": "JacobianProbeSpec", "column_reach": [3, 3, 3, 3, 2, 2]},
        "beta_floor": 0.05,
    }


def test_the_default_probe_is_omitted_and_an_omitted_probe_reads_back_as_the_default() -> None:
    assert preconditioner_spec_to_mapping(MaterializedJacobian(CompleteLu())) == {
        "kind": "MaterializedJacobian",
        "inverse": {"kind": "CompleteLu"},
    }
    read = preconditioner_spec_from_mapping(
        {"kind": "MaterializedJacobian", "inverse": {"kind": "CompleteLu"}}
    )
    assert read.probe == JacobianProbeSpec()


def test_an_explicit_null_scalar_method_differs_from_an_absent_one() -> None:
    assert preconditioner_spec_to_mapping(BlockDiagonal()) == {"kind": "BlockDiagonal"}
    assert preconditioner_spec_to_mapping(BlockDiagonal(method=None)) == {
        "kind": "BlockDiagonal",
        "method": None,
    }
    assert (
        preconditioner_spec_from_mapping({"kind": "BlockDiagonal"}).resolved_method() == "twolevel"
    )
    assert (
        preconditioner_spec_from_mapping(
            {"kind": "BlockDiagonal", "method": None}
        ).resolved_method()
        is None
    )


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        ({"kind": "Preconditioner"}, "unknown kind 'Preconditioner'"),
        (
            {"kind": "MaterializedJacobian", "inverse": {"kind": "Ilut"}},
            "unknown kind 'Ilut' at 'inverse'",
        ),
        (
            {
                "kind": "MaterializedJacobian",
                "inverse": {
                    "kind": "FieldSplit",
                    "leading": {"kind": "SimpleSmoothed", "smoother_sweeps": 4},
                    "trailing": {"kind": "JacobiSmoothed"},
                },
            },
            "SimpleSmoothed at 'inverse.leading' has no field 'smoother_sweeps'",
        ),
        ({"kind": "BlockDiagonal", "velocity": "convection"}, None),
        ({"inverse": {"kind": "CompleteLu"}}, "names no 'kind'"),
    ],
    ids=[
        "unknown-kind",
        "unknown-nested-kind",
        "unknown-nested-field",
        "velocity-string",
        "no-kind",
    ],
)
def test_a_malformed_spec_file_is_refused_by_name(mapping, match) -> None:
    if match is None:
        # A retired velocity string reaches the value's own refusal, which names the values instead.
        with pytest.raises(TypeError, match="must be a velocity-block value"):
            preconditioner_spec_from_mapping(mapping)
        return
    with pytest.raises(ValueError, match=match):
        preconditioner_spec_from_mapping(mapping)


def test_a_nested_value_of_the_wrong_kind_reaches_that_value_s_own_refusal() -> None:
    with pytest.raises(TypeError, match=r"FieldSplit\.trailing must be a block-inverse value"):
        preconditioner_spec_from_mapping(
            {
                "kind": "MaterializedJacobian",
                "inverse": {
                    "kind": "FieldSplit",
                    "leading": {"kind": "SimpleSmoothed"},
                    "trailing": {"kind": "ConvectionAir"},
                },
            }
        )


@pytest.mark.parametrize(
    "mapping",
    [{"kind": "CompleteLu"}, {"kind": "SimpleSmoothed"}, {"kind": "JacobianProbeSpec"}],
    ids=lambda m: m["kind"],
)
def test_the_outermost_kind_must_be_one_of_the_two_families(mapping) -> None:
    with pytest.raises(TypeError, match="BlockDiagonal or MaterializedJacobian"):
        preconditioner_spec_from_mapping(mapping)


def test_only_a_spec_family_is_written() -> None:
    with pytest.raises(TypeError, match="BlockDiagonal or MaterializedJacobian"):
        preconditioner_spec_to_mapping(CompleteLu())  # type: ignore[arg-type]
