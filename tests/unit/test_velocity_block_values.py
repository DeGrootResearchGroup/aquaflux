"""The velocity-block values: each names one strategy's own settings, and a string is refused.

``BlockPreconditioner.build``'s ``velocity`` was a string fusing which frozen momentum operator the block
is fitted to with which hierarchy coarsens it. The values that replaced it were proven to build bitwise
what each string built before the strings were removed; what is pinned here is what has to stay true
now -- that every value carries exactly its strategy's settings, that the settings the string could not
express reach the strategy, and that a string is refused rather than silently misread.
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
    VelocityBlock,
    ViscousMultilevel,
)
from aquaflux.flow.block_preconditioner import (
    AirConvectionVelocity,
    SmoothedAmgVelocity,
    TwoLevelConvectionVelocity,
    _characteristic_reference_state,
    _ConvectionVelocityBlock,
    _VelocityGeometry,
)
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import BlockDiagonal
from aquaflux.turbulence.coupled import _coupled_shift_policy

from tests.unit.test_coupled_rans import _cavity, _healthy_state
from tests.unit.test_preconditioner import _channel


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
    :meth:`BlockPreconditioner.build` rather than on the velocity value. A setting added to a strategy
    and not to its value -- or the reverse -- fails here rather than becoming unreachable, which is how
    the two-level smoother's ``sweeps`` and ``omega`` sat unreachable behind the string.
    """
    keyword_only = {
        name
        for name, parameter in inspect.signature(strategy.build).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert {field.name for field in dataclasses.fields(value)} == keyword_only - {
        "strength_threshold"
    }


def test_the_two_level_smoother_settings_reach_the_strategy() -> None:
    """``ConvectionTwoLevel(sweeps, omega)`` builds what the strategy builds when handed them directly."""
    assembler = _channel(2.0)
    state = _characteristic_reference_state(assembler)
    built = BlockPreconditioner.build(assembler, velocity=ConvectionTwoLevel(sweeps=3, omega=0.7))
    assert (built.velocity.sweeps, built.velocity.omega) == (3, 0.7)

    owner_e, nb_e, _ = assembler.mesh.face_cells.interior_edges()
    direct = TwoLevelConvectionVelocity.build(
        _VelocityGeometry.of(assembler),
        owner_e,
        nb_e,
        np.asarray(assembler.mesh.face_cells.interior),
        assembler.mesh.n_cells,
        1,
        jax.lax.stop_gradient(assembler.mass_flux(state)),
        sweeps=3,
        omega=0.7,
    )
    a_p = built.frozen_momentum_diagonal(state)
    ru = jnp.asarray(np.random.default_rng(1).standard_normal((assembler.mesh.n_cells, 2)))
    np.testing.assert_array_equal(
        np.asarray(direct.apply(a_p)(ru)), np.asarray(built.velocity.apply(a_p)(ru))
    )


@pytest.mark.parametrize("value", [ConvectionTwoLevel(), ConvectionAir()], ids=["two-level", "air"])
def test_a_convection_value_with_no_reference_flux_says_so_and_names_itself(value) -> None:
    """The warning names the value asked for, and is attributed to the line that called the builder.

    It is raised two frames below that call, inside the value's own build, so a wrong ``stacklevel``
    points a reader at library internals instead of at the code that asked for a convection block.
    """
    with pytest.warns(
        RuntimeWarning, match=rf"{type(value).__name__}\(\) was requested.*no mass flux"
    ) as record:
        BlockPreconditioner.build(_closed(), velocity=value)
    flux_warnings = [w for w in record if "no mass flux" in str(w.message)]
    assert [w.filename for w in flux_warnings] == [__file__], (
        "the zero-flux warning is not attributed to the caller of BlockPreconditioner.build: "
        f"{[w.filename for w in flux_warnings]}"
    )


def test_the_viscous_value_is_the_default_and_says_nothing_on_a_closed_domain() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = BlockPreconditioner.build(_closed())
        explicit = BlockPreconditioner.build(_closed(), velocity=ViscousMultilevel())
    assert isinstance(default.velocity, SmoothedAmgVelocity)
    for a, b in zip(jax.tree.leaves(default), jax.tree.leaves(explicit), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_coupled_default_flow_block_is_the_two_level_convection_value() -> None:
    mesh, coupled = _cavity(4)
    state = _healthy_state(mesh, coupled)
    default = _coupled_shift_policy(coupled, state, None).flow_preconditioner
    explicit = _coupled_shift_policy(
        coupled, state, None, velocity=ConvectionTwoLevel()
    ).flow_preconditioner
    assert isinstance(default.velocity, TwoLevelConvectionVelocity)
    for a, b in zip(jax.tree.leaves(default), jax.tree.leaves(explicit), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("string", ["smoothed", "convection", "convection-air"])
def test_a_velocity_string_is_refused_by_the_builder_and_by_the_spec(string) -> None:
    """Refused, not aliased: a string that still type-checks as "some velocity" would be read by no one."""
    with pytest.raises(TypeError, match="velocity-block value"):
        BlockPreconditioner.build(_channel(2.0), velocity=string)
    with pytest.raises(TypeError, match="velocity-block value"):
        BlockDiagonal(velocity=string)


@pytest.mark.parametrize("base", [VelocityBlock, _ConvectionVelocityBlock])
def test_an_abstract_velocity_block_is_refused_where_it_is_written(base) -> None:
    """An abstract block passes an ``isinstance`` check and has nothing to build.

    Accepted, it reached :meth:`BlockPreconditioner.build`, which built the pressure Schur and then
    failed on an empty ``NotImplementedError``. Construction is where it is refused, naming the values.
    """
    with pytest.raises(
        TypeError, match=rf"{base.__name__} is abstract; construct .*ConvectionTwoLevel\(\)"
    ):
        base()
