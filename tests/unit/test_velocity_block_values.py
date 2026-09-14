"""The velocity-block values: each names one strategy's own settings, and builds what its string built.

``BlockPreconditioner.build``'s ``velocity`` was a string fusing which frozen momentum operator the block
is fitted to with which hierarchy coarsens it. The values replace it one-for-one, so the first obligation
is that they change nothing numerically: every array the preconditioner holds, and what it returns when
applied, must be bitwise equal to the string path's -- on a graded viscosity, where the viscous and
convection operators differ, on a closed domain with no reference flux, and at non-default multigrid
settings. The string path compared against is the original, untouched implementation.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import (
    BlockPreconditioner,
    ConvectionAir,
    ConvectionTwoLevel,
    MomentumContinuity,
    NoSlipWall,
    ViscousMultilevel,
)
from aquaflux.flow.block_preconditioner import (
    AirConvectionVelocity,
    SmoothedAmgConvectionVelocity,
    SmoothedAmgVelocity,
    TwoLevelConvectionVelocity,
    _characteristic_reference_state,
    _VelocityGeometry,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence.coupled import _coupled_shift_policy

from tests.unit.test_coupled_rans import _cavity, _healthy_state
from tests.unit.test_preconditioner import _channel

#: Each retired string beside the value that replaces it.
PAIRS = [
    ("smoothed", ViscousMultilevel()),
    ("convection", ConvectionTwoLevel()),
    ("convection-air", ConvectionAir()),
]
_IDS = [string for string, _ in PAIRS]


def _graded(assembler: MomentumContinuity) -> MomentumContinuity:
    """The same channel with an eddy viscosity rising across the cells, so the viscosity is graded."""
    return assembler.with_eddy_viscosity(jnp.linspace(0.0, 0.2, assembler.mesh.n_cells))


def _closed() -> MomentumContinuity:
    """A cavity with every wall stationary: nothing prescribes a velocity, so no reference flux."""
    mesh = structured_grid_2d(6, 6, lx=1.0, ly=1.0, named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions({side: NoSlipWall() for side in ("top", "bottom", "left", "right")}),
    )


def _assert_bitwise_equal(old: BlockPreconditioner, new: BlockPreconditioner, state) -> None:
    old_leaves, new_leaves = jax.tree.leaves(old), jax.tree.leaves(new)
    assert len(old_leaves) == len(new_leaves)
    for a, b in zip(old_leaves, new_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    a_p = old.frozen_momentum_diagonal(state)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(state.shape))
    np.testing.assert_array_equal(
        np.asarray(old.apply_at(state, a_p)(v)), np.asarray(new.apply_at(state, a_p)(v))
    )


@pytest.mark.parametrize(("string", "value"), PAIRS, ids=_IDS)
@pytest.mark.parametrize(
    "options",
    [{}, {"strength_threshold": 0.25}, {"v_cycles": 2}],
    ids=["defaults", "strength", "two-cycles"],
)
@pytest.mark.parametrize("graded", [False, True], ids=["uniform-viscosity", "graded-viscosity"])
def test_a_value_builds_bitwise_what_its_string_built(string, value, options, graded) -> None:
    assembler = _channel(2.0)
    if graded:
        assembler = _graded(assembler)
    state = _characteristic_reference_state(assembler)
    old = BlockPreconditioner.build(assembler, velocity=string, **options)
    new = BlockPreconditioner.build(assembler, velocity=value, **options)
    _assert_bitwise_equal(old, new, state)


@pytest.mark.parametrize(("string", "value"), PAIRS, ids=_IDS)
def test_a_value_freezes_at_an_explicit_reference_state_as_its_string_did(string, value) -> None:
    assembler = _channel(2.0)
    base = _characteristic_reference_state(assembler)
    reference = base * (1.0 + 0.1 * jnp.sin(jnp.arange(base.size)))
    old = BlockPreconditioner.build(assembler, velocity=string, reference_state=reference)
    new = BlockPreconditioner.build(assembler, velocity=value, reference_state=reference)
    _assert_bitwise_equal(old, new, reference)


@pytest.mark.parametrize(("string", "value"), PAIRS[1:], ids=_IDS[1:])
def test_a_convection_value_warns_on_zero_flux_and_still_builds_what_its_string_did(
    string, value
) -> None:
    assembler = _closed()
    with pytest.warns(RuntimeWarning, match="no mass flux"):
        old = BlockPreconditioner.build(assembler, velocity=string)
    with pytest.warns(RuntimeWarning, match=type(value).__name__):
        new = BlockPreconditioner.build(assembler, velocity=value)
    _assert_bitwise_equal(old, new, _characteristic_reference_state(assembler))


def test_the_viscous_value_says_nothing_on_a_closed_domain() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BlockPreconditioner.build(_closed(), velocity=ViscousMultilevel())


def test_the_coupled_default_flow_block_is_the_two_level_convection_value() -> None:
    """The coupled march's unset velocity block is ``ConvectionTwoLevel()``, bitwise."""
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    default = _coupled_shift_policy(coupled, state, None).flow_preconditioner
    valued = _coupled_shift_policy(
        coupled, state, None, velocity=ConvectionTwoLevel()
    ).flow_preconditioner
    flow, _, _ = coupled.physical_fields(state)
    _assert_bitwise_equal(default, valued, flow)


def test_the_two_level_smoother_settings_are_reachable_and_reach_the_strategy() -> None:
    """``sweeps`` and ``omega`` existed only on the strategy; the value is now the way to set them."""
    assembler = _channel(2.0)
    state = _characteristic_reference_state(assembler)
    new = BlockPreconditioner.build(assembler, velocity=ConvectionTwoLevel(sweeps=3, omega=0.7))
    assert (new.velocity.sweeps, new.velocity.omega) == (3, 0.7)

    owner_e, nb_e, _ = assembler.mesh.face_cells.interior_edges()
    reference_mdot = jax.lax.stop_gradient(assembler.mass_flux(state))
    old = SmoothedAmgConvectionVelocity.build(
        _VelocityGeometry.of(assembler),
        owner_e,
        nb_e,
        np.asarray(assembler.mesh.face_cells.interior),
        assembler.mesh.n_cells,
        1,
        reference_mdot,
        method="twolevel",
        sweeps=3,
        omega=0.7,
    )
    a_p = new.frozen_momentum_diagonal(state)
    ru = jnp.asarray(np.random.default_rng(1).standard_normal((assembler.mesh.n_cells, 2)))
    np.testing.assert_array_equal(
        np.asarray(old.apply(a_p)(ru)), np.asarray(new.velocity.apply(a_p)(ru))
    )


@pytest.mark.parametrize(
    ("value", "strategy"),
    [
        (ViscousMultilevel, SmoothedAmgVelocity),
        (ConvectionTwoLevel, TwoLevelConvectionVelocity),
        (ConvectionAir, AirConvectionVelocity),
    ],
    ids=["viscous", "two-level", "air"],
)
def test_each_value_names_exactly_its_strategy_s_own_settings(value, strategy) -> None:
    """A value's fields are its strategy's keyword-only settings, less what the builder supplies.

    ``strength_threshold`` is shared with the pressure Schur, so it is set once on
    :meth:`BlockPreconditioner.build` rather than on the velocity value.
    """
    keyword_only = {
        name
        for name, parameter in inspect.signature(strategy.build).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert {field.name for field in dataclasses.fields(value)} == keyword_only - {
        "strength_threshold"
    }
