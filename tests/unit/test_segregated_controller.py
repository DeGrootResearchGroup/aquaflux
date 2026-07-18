"""Unit test for the mass-flow controller's bulk-velocity measure.

``bulk_velocity`` is the volume-averaged streamwise velocity the periodic-channel controller drives
to a target; the end-to-end controller is exercised by
``tests/integration/test_channel_law_of_the_wall.py``. Here we check the pure measure against a
hand-computed volume average on a graded (non-uniform-volume) mesh, where a plain cell mean would be
wrong.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import MomentumContinuity, NoSlipWall
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import bulk_velocity


def test_bulk_velocity_is_the_volume_weighted_mean() -> None:
    mesh = structured_grid_2d(
        4,
        8,
        lx=1.0,
        ly=2.0,
        periodic=("x",),
        named_boundaries=True,
        y_nodes=graded_nodes(8, 2.0, 1.3),
    )
    geometry = mesh.geometry()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        pressure_pin=0,
    )
    # A non-uniform streamwise field on a graded mesh: the volume weighting matters.
    n = mesh.n_cells
    ux = jnp.asarray(0.5 + 0.3 * np.cos(np.arange(n)))
    state = momentum.pack(jnp.stack([ux, jnp.zeros(n)], axis=1), jnp.zeros(n))

    volume = np.asarray(geometry.cell.volume)
    expected = float(np.sum(np.asarray(ux) * volume) / np.sum(volume))
    assert np.isclose(float(bulk_velocity(momentum, state, 0)), expected)
    # A different direction reads that component; the quiescent y-velocity averages to zero.
    assert np.isclose(float(bulk_velocity(momentum, state, 1)), 0.0)
