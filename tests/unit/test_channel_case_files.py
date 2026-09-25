"""The channel case files build the channels their validation scripts used to assemble by hand.

The two streamwise-periodic channel studies (``validation/turbulent_channel``, against the log law,
and ``validation/turbulent_channel_openfoam``, against OpenFOAM) read one case file per configuration.
Each is compared here against that configuration as the scripts wrote it out by hand before they read
files -- a frozen copy, since the scripts themselves now build from the files under test.

The comparison is one pytree (the same structure, static fields included, and every array leaf
bit-equal), with one deliberate difference: the scripts held the viscosity as a Python number, and a
case file holds it as an array, which a Reynolds continuation can rescale without recompiling. The
reference below uses the array form, and
:func:`test_the_array_viscosity_evaluates_to_the_same_viscosity_as_the_number` shows the number form
evaluates to identical values, so the difference is in the pytree and nowhere else.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.case import read_case
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import MassFlow, MomentumContinuity, NoSlipWall, PinnedPoint
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import CoupledRANS, SSTModel, SSTTurbulence

REPO = Path(__file__).resolve().parents[2]
H, U_BAR = 2.0, 0.1335  # full channel height; OpenFOAM's meanVelocityForce bulk velocity

#: (file, ny, growth, nu, force seed, momentum advection) -- each configuration as its script set it.
CONFIGURATIONS = [
    ("turbulent_channel/cases/re20000.yaml", 96, 1.09, 1.0 * H / 20000, 0.004, "first"),
    ("turbulent_channel/cases/re45000.yaml", 120, 1.075, 1.0 * H / 45000, 0.0035, "first"),
    ("turbulent_channel/cases/re240000.yaml", 224, 1.052, 1.0 * H / 240000, 0.00175, "first"),
    (
        "turbulent_channel_openfoam/cases/low.yaml",
        96,
        1.09,
        H / (U_BAR * H / 2e-5),
        0.004,
        "linear",
    ),
    (
        "turbulent_channel_openfoam/cases/high.yaml",
        224,
        1.052,
        H / (U_BAR * H / 1.57e-6),
        0.004,
        "linear",
    ),
]


def _hand_built(ny, growth, nu, force, advection, *, viscosity_as_array=True) -> CoupledRANS:
    """The channel as its script assembled it by hand (the viscosity form aside -- see the module)."""
    mesh = structured_grid_2d(
        4,
        ny,
        lx=1.0,
        ly=H,
        periodic=("x",),
        named_boundaries=True,
        y_nodes=graded_nodes(ny, H, growth),
    )
    geometry = mesh.geometry()
    viscosity = jnp.asarray(1.0 * nu) if viscosity_as_array else 1.0 * nu
    properties = PropertyModel({"viscosity": Constant(viscosity), "density": Constant(1.0)})
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        properties,
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        gradient_scheme=CompactGreenGauss(),
        advection_scheme=FirstOrderUpwind() if advection == "first" else LimitedUpwind(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
        drive=MassFlow(target=1.0, flow_direction=0, force=force),
    )
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        gradient_scheme=CompactGreenGauss(),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions({"bottom": Dirichlet(0.0), "top": Dirichlet(0.0)}),
        omega_boundary=BoundaryConditions({"bottom": ZeroGradient(), "top": ZeroGradient()}),
    )
    return CoupledRANS.build(momentum, turbulence)


def _same_problem(built: object, reference: object) -> None:
    built_leaves, built_def = jax.tree.flatten(built)
    reference_leaves, reference_def = jax.tree.flatten(reference)
    assert built_def == reference_def
    for a, b in zip(built_leaves, reference_leaves, strict=True):
        assert eqx.is_array(a) == eqx.is_array(b), (type(a), type(b))
        if eqx.is_array(a):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        else:
            assert a == b


@pytest.mark.parametrize(
    ("path", "ny", "growth", "nu", "force", "advection"),
    CONFIGURATIONS,
    ids=[Path(c[0]).parent.parent.name + "/" + Path(c[0]).stem for c in CONFIGURATIONS],
)
def test_each_channel_case_file_builds_the_channel_its_script_assembled(
    path, ny, growth, nu, force, advection
) -> None:
    case = read_case(REPO / "validation" / path)
    # The file states the viscosity the script computed, to the last bit -- it was written from it.
    assert case.spec.fluid.kinematic_viscosity == nu
    _same_problem(case.check().build(), _hand_built(ny, growth, nu, force, advection))


def test_the_array_viscosity_evaluates_to_the_same_viscosity_as_the_number() -> None:
    """The one deliberate difference from the scripts is in the pytree, not in any value a residual reads.

    The viscosity reaches every residual only through the evaluated property model, so equal per-cell
    values there are equal values everywhere downstream.
    """
    _, ny, growth, nu, force, advection = CONFIGURATIONS[0]
    as_array = _hand_built(ny, growth, nu, force, advection)
    as_number = _hand_built(ny, growth, nu, force, advection, viscosity_as_array=False)
    zones = as_array.momentum.mesh.cell_zones
    array_values = as_array.momentum.properties.evaluate(zones)
    number_values = as_number.momentum.properties.evaluate(zones)
    for name in ("viscosity", "density"):
        np.testing.assert_array_equal(
            np.asarray(array_values[name]), np.asarray(number_values[name])
        )
    np.testing.assert_array_equal(
        np.asarray(as_array.turbulence.molecular_viscosity),
        np.asarray(as_number.turbulence.molecular_viscosity),
    )
