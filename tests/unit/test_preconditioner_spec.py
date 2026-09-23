"""The coupled-preconditioner specs: field sets pinned to what they configure, and invalid shapes refused."""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from aquaflux.flow import BlockPreconditioner
from aquaflux.solve import (
    CompleteLu,
    FieldSplit,
    JacobianProbeSpec,
    JacobiSmoothed,
    MaterializedJacobian,
    MonolithicAmgPreconditioner,
    MonolithicLuPreconditioner,
    MonolithicVCycle,
    SimpleSmoothed,
)
from aquaflux.turbulence import (
    BlockDiagonal,
    ScalarAir,
    ScalarTwoLevel,
    UnpreconditionedScalars,
    coupled_jacobian_probe,
)


def _fields(value_class) -> set[str]:
    return {field.name for field in dataclasses.fields(value_class)}


def _parameters(function, kind) -> set[str]:
    return {p.name for p in inspect.signature(function).parameters.values() if p.kind is kind}


_KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY


def test_the_block_diagonal_spec_names_the_flow_block_settings_and_the_scalar_block() -> None:
    """``reference_state`` is supplied by the march, so it is the one keyword with no field."""
    expected = _parameters(BlockPreconditioner.build, _KEYWORD_ONLY) - {"reference_state"}
    assert _fields(BlockDiagonal) == expected | {"scalar"}


def test_the_probe_spec_names_the_probe_builders_free_settings() -> None:
    """The skipped blocks and the production-viscosity stand-in follow from other choices."""
    expected = _parameters(coupled_jacobian_probe, inspect.Parameter.POSITIONAL_OR_KEYWORD) - {
        "coupled"
    }
    assert _fields(JacobianProbeSpec) == expected


def test_the_monolithic_vcycle_spec_names_the_v_cycles_own_settings() -> None:
    """The probing callables and the extra options are supplied where the V-cycle is built."""
    supplied = {"batched_matvec", "probe_batch_size", "structure", "extra_options"}
    expected = _parameters(MonolithicAmgPreconditioner.build, _KEYWORD_ONLY) - supplied
    assert _fields(MonolithicVCycle) == expected


def test_the_complete_lu_spec_names_the_factorizations_settings() -> None:
    assert _fields(CompleteLu) == _parameters(MonolithicLuPreconditioner.build, _KEYWORD_ONLY)


def test_an_unset_scalar_block_resolves_to_the_two_level_default() -> None:
    """``None`` is unset, as on every other field; leaving the blocks unpreconditioned is a value."""
    assert BlockDiagonal().scalar is None
    assert BlockDiagonal().resolved_scalar() == ScalarTwoLevel()
    unpreconditioned = BlockDiagonal(scalar=UnpreconditionedScalars())
    assert unpreconditioned.resolved_scalar() == UnpreconditionedScalars()
    assert BlockDiagonal(scalar=ScalarAir(v_cycles=2)).resolved_scalar() == ScalarAir(v_cycles=2)


def test_a_scalar_block_given_as_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match=r"BlockDiagonal\.scalar must be a scalar-block value"):
        BlockDiagonal(scalar="air")


def test_only_set_flow_block_settings_are_forwarded_and_the_scalar_block_never_is() -> None:
    assert BlockDiagonal().flow_block_options() == {}
    assert BlockDiagonal(scalar=ScalarAir(), v_cycles=2).flow_block_options() == {"v_cycles": 2}


def test_a_column_reach_list_is_stored_as_a_tuple_so_the_spec_hashes() -> None:
    from_list = JacobianProbeSpec(column_reach=[3, 3, 3, 3, 2, 2])
    assert from_list.column_reach == (3, 3, 3, 3, 2, 2)
    assert from_list == JacobianProbeSpec(column_reach=(3, 3, 3, 3, 2, 2))
    assert hash(from_list) == hash(JacobianProbeSpec(column_reach=(3, 3, 3, 3, 2, 2)))


def test_nested_specs_compare_and_hash_by_value() -> None:
    def spec(max_coarse: int) -> MaterializedJacobian:
        return MaterializedJacobian(
            FieldSplit(SimpleSmoothed(sweeps=2), JacobiSmoothed(max_coarse=max_coarse)),
            probe=JacobianProbeSpec(column_reach=(3, 3, 3, 3, 2, 2)),
            refit_beta_floor=0.05,
        )

    assert spec(200) == spec(200)
    assert hash(spec(200)) == hash(spec(200))
    assert spec(200) != spec(150)


def test_a_field_split_refuses_an_inverse_that_is_not_a_value() -> None:
    with pytest.raises(TypeError, match=r"FieldSplit\.trailing must be a block-inverse value"):
        FieldSplit(SimpleSmoothed(), lambda block, n_fields: None)


def test_a_bare_block_inverse_is_a_materialized_inverse_for_a_single_group_of_fields() -> None:
    """The spec accepts it; whether the problem can use it is the session's decision.

    A bare block inverse is fitted to the whole state, so it belongs to a problem with one group of
    fields (a laminar flow). The refusal for a two-group problem lives where the groups are known.
    """
    assert MaterializedJacobian(SimpleSmoothed()).inverse == SimpleSmoothed()


def test_the_block_diagonal_family_is_not_a_materialized_inverse() -> None:
    with pytest.raises(TypeError, match=r"MaterializedJacobian\.inverse must be"):
        MaterializedJacobian(BlockDiagonal())


def test_a_probe_must_be_a_probe_spec() -> None:
    with pytest.raises(TypeError, match=r"MaterializedJacobian\.probe must be a JacobianProbeSpec"):
        MaterializedJacobian(CompleteLu(), probe={"stencil_reach": 3})
