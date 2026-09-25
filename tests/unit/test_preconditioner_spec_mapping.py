"""A coupled preconditioner spec read from and written to the mapping a case file parses to.

Every value a spec can hold, at any level, round-trips through a plain mapping; the mapping holds only
plain data; unknown kinds and fields are refused by name; and what is read is a spec, never a built
preconditioner.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import typing

import aquaflux
import numpy as np
import pytest
from aquaflux.flow import ConvectionAir, ConvectionTwoLevel, VelocityBlock, ViscousMultilevel
from aquaflux.flow.block_preconditioner import _COMPOSITIONS, SCHUR_SCALINGS
from aquaflux.solve import (
    MATERIALIZED_MAPPING,
    AirReduction,
    BlockInverse,
    CompleteLu,
    FieldSplit,
    JacobianProbeSpec,
    JacobiSmoothed,
    MaterializedJacobian,
    MonolithicVCycle,
    SettingsMapping,
    SimpleSmoothed,
)
from aquaflux.solve.lu_preconditioner import LU_BACKENDS
from aquaflux.solve.multigrid import _PROLONGATION_SMOOTHING
from aquaflux.turbulence import (
    PRECONDITIONER_SPEC_MAPPING,
    BlockDiagonal,
    ScalarAir,
    ScalarBlock,
    ScalarTwoLevel,
    UnpreconditionedScalars,
    preconditioner_spec_from_mapping,
    preconditioner_spec_to_mapping,
)

#: One spec per shape a case file can take, together holding every value class at least once.
_SPECS = [
    BlockDiagonal(),
    BlockDiagonal(scalar=UnpreconditionedScalars()),
    BlockDiagonal(scalar=ScalarAir(), velocity=ViscousMultilevel()),
    BlockDiagonal(scalar=ScalarTwoLevel(v_cycles=2)),
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
        refit_beta_floor=0.05,
    ),
    MaterializedJacobian(
        FieldSplit(
            SimpleSmoothed(sweeps=2, strength_threshold=0.25, frozen_coarsening=True),
            JacobiSmoothed(max_coarse=200, prolongation_smoothing="symmetric-part"),
        ),
        probe=JacobianProbeSpec(column_reach=(3, 3, 3, 3, 2, 2), gradient_sweeps=2),
        refit_beta_floor=0.05,
    ),
    MaterializedJacobian(
        FieldSplit(SimpleSmoothed(), AirReduction(theta=0.5, degree=1)),
        probe=JacobianProbeSpec(stencil_reach=3),
    ),
]


def _held_classes(value: object, into: set[type]) -> set[type]:
    into.add(type(value))
    for setting in vars(value).values():
        if type(setting) in PRECONDITIONER_SPEC_MAPPING.kinds:
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

    The nested families -- velocity blocks, scalar blocks and block inverses, which grow as methods are
    added -- are
    found from every subpackage's exports, so a new member exported anywhere is caught. The spec's own
    classes, which do not form an open family, are named here.
    """
    nested = {
        cls
        for cls in _exported_classes()
        if issubclass(cls, VelocityBlock | ScalarBlock | BlockInverse)
        and cls not in (VelocityBlock, ScalarBlock, BlockInverse)
        and not inspect.isabstract(cls)
    }
    spec_classes = {BlockDiagonal, MaterializedJacobian, CompleteLu, MonolithicVCycle, FieldSplit}
    return nested | spec_classes | {JacobianProbeSpec}


def test_the_mapping_accepts_every_public_value_a_spec_can_hold() -> None:
    """A value class added to a family but not to the mapping would be unwritable in a case file."""
    assert set(PRECONDITIONER_SPEC_MAPPING.kinds) == _public_value_classes()


def test_the_census_scans_every_subpackage_not_only_the_ones_a_spec_imports() -> None:
    names = {info.name for info in pkgutil.iter_modules(aquaflux.__path__) if info.ispkg}
    assert {"flow", "solve", "turbulence", "transport"} <= names
    assert _exported_classes() >= {BlockDiagonal, SimpleSmoothed, ConvectionAir}


def test_the_round_trip_specs_hold_every_accepted_class() -> None:
    held: set[type] = set()
    for spec in _SPECS:
        _held_classes(spec, held)
    assert held == set(PRECONDITIONER_SPEC_MAPPING.kinds)


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
        refit_beta_floor=0.05,
    )
    assert preconditioner_spec_to_mapping(spec) == {
        "kind": "MaterializedJacobian",
        "inverse": {
            "kind": "FieldSplit",
            "leading": {"kind": "SimpleSmoothed", "sweeps": 2},
            "trailing": {"kind": "JacobiSmoothed", "max_coarse": 200},
        },
        "probe": {"kind": "JacobianProbeSpec", "column_reach": [3, 3, 3, 3, 2, 2]},
        "refit_beta_floor": 0.05,
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


def test_unpreconditioned_scalar_blocks_are_a_kind_and_a_null_scalar_is_the_default() -> None:
    """``null`` means "not set" for the scalar block as for every field; no preconditioner is a kind."""
    assert preconditioner_spec_to_mapping(BlockDiagonal(scalar=UnpreconditionedScalars())) == {
        "kind": "BlockDiagonal",
        "scalar": {"kind": "UnpreconditionedScalars"},
    }
    null = preconditioner_spec_from_mapping({"kind": "BlockDiagonal", "scalar": None})
    assert null == BlockDiagonal()
    assert null.resolved_scalar() == ScalarTwoLevel()


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
        (
            {"kind": "BlockDiagonal", "velocity": "convection"},
            "'convection' at 'velocity' is not accepted there",
        ),
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
    with pytest.raises(ValueError, match=match):
        preconditioner_spec_from_mapping(mapping)


def test_a_nested_value_of_the_wrong_kind_is_refused_where_it_appears() -> None:
    """A known kind in a position that cannot take it: named by path, with what belongs there.

    It used to load as far as ``FieldSplit``'s own constructor refusal, which says what a field split
    takes but not where in the file the offending entry is -- and a value family whose constructor
    happens not to check would not have been refused at all.
    """
    spec = {
        "kind": "MaterializedJacobian",
        "inverse": {
            "kind": "FieldSplit",
            "leading": {"kind": "SimpleSmoothed"},
            "trailing": {"kind": "ConvectionAir"},
        },
    }
    with pytest.raises(ValueError, match=r"at 'inverse.trailing' is not accepted there"):
        preconditioner_spec_from_mapping(spec)
    # The same value written in code still meets the constructor's own refusal, which is the other
    # route to the same conclusion and is deliberately unchanged (issue #424 is about the file).
    with pytest.raises(TypeError, match=r"FieldSplit\.trailing must be a block-inverse value"):
        FieldSplit(SimpleSmoothed(), ConvectionAir())


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


def test_the_solve_registry_round_trips_a_materialized_spec_without_the_turbulence_package() -> (
    None
):
    """A laminar case file loads its spec from ``aquaflux.solve`` alone, and the schema is unchanged."""
    from aquaflux.solve import (
        FieldSplit,
        JacobiSmoothed,
        MaterializedJacobian,
        SimpleSmoothed,
        materialized_spec_from_mapping,
        materialized_spec_to_mapping,
    )

    spec = MaterializedJacobian(
        FieldSplit(SimpleSmoothed(sweeps=2), JacobiSmoothed()), refit_beta_floor=0.05
    )
    mapping = materialized_spec_to_mapping(spec)
    assert mapping["kind"] == "MaterializedJacobian"
    assert materialized_spec_from_mapping(mapping) == spec
    # ...and the coupled registry, which extends it, reads the very same mapping to the very same value.
    assert preconditioner_spec_from_mapping(mapping) == spec


def test_the_solve_registry_does_not_know_the_turbulence_only_kinds() -> None:
    """``BlockDiagonal`` is a coupled-RANS family: naming it in a laminar spec file is an unknown kind."""
    from aquaflux.solve import materialized_spec_from_mapping

    with pytest.raises(ValueError, match="BlockDiagonal"):
        materialized_spec_from_mapping({"kind": "BlockDiagonal"})


def test_the_coupled_registry_extends_the_solve_one_rather_than_restating_it() -> None:
    assert set(MATERIALIZED_MAPPING.kinds) < set(PRECONDITIONER_SPEC_MAPPING.kinds)


def test_a_bare_block_inverse_spec_round_trips_through_the_solve_registry() -> None:
    from aquaflux.solve import (
        MaterializedJacobian,
        SimpleSmoothed,
        materialized_spec_from_mapping,
        materialized_spec_to_mapping,
    )

    spec = MaterializedJacobian(SimpleSmoothed(sweeps=2, cycles=1))
    mapping = materialized_spec_to_mapping(spec)
    assert mapping["inverse"]["kind"] == "SimpleSmoothed"
    assert materialized_spec_from_mapping(mapping) == spec


# --- the values a spec file may give, per position ---------------------------------------------


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (
            {
                "kind": "MaterializedJacobian",
                "inverse": {"kind": "CompleteLu", "backend": "umfpak"},
            },
            r"'umfpak' at 'inverse.backend' is not accepted there",
        ),
        (
            {
                "kind": "MaterializedJacobian",
                "inverse": {"kind": "CompleteLu", "backend": {"kind": "CompleteLu"}},
            },
            r"at 'inverse.backend' is not accepted there",
        ),
        (
            {
                "kind": "MaterializedJacobian",
                "inverse": {"kind": "MonolithicVCycle", "smoother_sweeps": True},
            },
            r"True at 'inverse.smoother_sweeps' is not accepted there",
        ),
    ],
    ids=["misspelt-choice", "value-where-a-string-belongs", "boolean-where-a-count-belongs"],
)
def test_the_three_spec_files_that_used_to_load_silently_are_refused(spec, match) -> None:
    """Issue #424's own examples.

    Each of these loaded without complaint and failed where the setting is consumed -- or never, since
    an ignored setting is indistinguishable from an absent one. A case file's whole point is that it is
    checked before a solve runs.
    """
    with pytest.raises(ValueError, match=match):
        preconditioner_spec_from_mapping(spec)


def test_every_choice_a_spec_offers_is_one_its_consumer_accepts() -> None:
    """The ``Literal``s are the file's copy of a choice its builder already knows; they must agree.

    Each set is spelled out in the annotation, because a ``Literal`` cannot be computed from a
    variable -- so this is the check that keeps the two from drifting. A value the spec offers and the
    builder rejects is a file that passes validation and fails at build; the reverse is a capability no
    file can reach.
    """
    choices = {
        (CompleteLu, "backend"): set(LU_BACKENDS),
        (BlockDiagonal, "schur_scaling"): set(SCHUR_SCALINGS),
        (BlockDiagonal, "composition"): set(_COMPOSITIONS),
        (SimpleSmoothed, "prolongation_smoothing"): set(_PROLONGATION_SMOOTHING),
        (JacobiSmoothed, "prolongation_smoothing"): set(_PROLONGATION_SMOOTHING),
    }
    for (kind, field), accepted in choices.items():
        annotation = typing.get_type_hints(kind)[field]
        (literal,) = [
            arm for arm in typing.get_args(annotation) if typing.get_origin(arm) is typing.Literal
        ]
        assert set(typing.get_args(literal)) == accepted, f"{kind.__name__}.{field}"


def test_every_field_of_every_spec_is_one_the_mapping_can_check() -> None:
    """A field whose annotation the rules cannot express would load unchecked.

    ``SettingsMapping`` refuses to be built over such a field, and both mappings are module-level
    constants, so this is really a test that importing the package still works -- written out because
    what it pins is a property of every spec class, not of the import.
    """
    for mapping in (PRECONDITIONER_SPEC_MAPPING, MATERIALIZED_MAPPING):
        rebuilt = SettingsMapping(mapping.kinds)
        assert rebuilt.kinds == mapping.kinds
