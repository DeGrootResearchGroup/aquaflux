"""The flow block's row-scaled measure at and near rest: its scales are floored by the driving speed.

Both scales the measure divides by vanish at rest -- the mean speed and every cell's mass throughput --
so a march that starts there could not be measured at all. They are floored by the characteristic
speed the domain is driven at, and a domain driven by nothing has no speed to floor by, so a state at
rest there is refused with an error that names the measure.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PinnedPoint,
    flow_row_scales,
)
from aquaflux.flow.measures import REST_SPEED_FRACTION, REST_THROUGHPUT_FRACTION
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss

RHO, LID = 2.0, 3.0


def _cavity(lid: float) -> MomentumContinuity:
    """A 5 x 4 box on graded nodes, driven by a lid moving at ``lid`` (at rest when zero)."""
    mesh = structured_grid_2d(
        5,
        4,
        named_boundaries=True,
        x_nodes=np.array([0.0, 0.1, 0.3, 0.6, 0.8, 1.0]),
        y_nodes=np.array([0.0, 0.05, 0.2, 0.4, 0.6]),
    )
    top = MovingWall(velocity=(lid, 0.0)) if lid else NoSlipWall()
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(1e-2), "density": Constant(RHO)}),
        BoundaryConditions(
            {"top": top, "bottom": NoSlipWall(), "left": NoSlipWall(), "right": NoSlipWall()}
        ),
        gradient_scheme=CompactGreenGauss(),
        advection_scheme=FirstOrderUpwind(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
    )


def _cell_face_areas(momentum: MomentumContinuity) -> np.ndarray:
    """Each cell's summed face area, by a plain loop over the faces."""
    owner = np.asarray(momentum.mesh.face_cells.owner)
    neighbour = np.asarray(momentum.mesh.face_cells.neighbour)
    area = np.asarray(momentum.geometry.face.area)
    total = np.zeros(momentum.mesh.n_cells)
    for face in range(momentum.mesh.n_faces):
        total[owner[face]] += area[face]
        if neighbour[face] >= 0:
            total[neighbour[face]] += area[face]
    return total


def _scales(momentum: MomentumContinuity, state: jnp.ndarray):
    diagonal = jnp.ones_like(state)
    row_scale, field_scale = flow_row_scales(momentum, diagonal, state)
    _, continuity = momentum.unpack(row_scale)
    return np.asarray(continuity), np.asarray(field_scale)


def test_at_rest_both_scales_are_the_floors_the_driving_speed_sets() -> None:
    """At rest the scales are the floors themselves, finite and written from their definition.

    The velocity field scale is a fraction of the lid speed; each cell's continuity scale is a fraction
    of ``rho |U| A`` over that cell's own faces, so cells with different face totals (the corner, edge
    and interior cells of this box) get different scales -- a floor using one cell's area for another's
    cannot pass.
    """
    momentum = _cavity(LID)
    continuity, field = _scales(momentum, momentum.initial_state())

    np.testing.assert_allclose(field, [REST_SPEED_FRACTION * LID] * 2 + [1.0], rtol=1e-14)
    expected = REST_THROUGHPUT_FRACTION * RHO * LID * _cell_face_areas(momentum)
    np.testing.assert_allclose(continuity, expected, rtol=1e-14)
    assert len(np.unique(np.round(expected, 12))) > 1, "the fixture must give cells different areas"


def test_a_state_with_flow_in_it_is_scaled_by_its_own_speed_and_throughput() -> None:
    """Where the state has flow of its own, neither floor binds and the scales are the state's own.

    A uniform stream at a third of the lid speed: its mean speed is well above the velocity floor and
    every cell's throughput is well above the throughput floor, so both scales must be exactly what they
    were before the floors existed.
    """
    momentum = _cavity(LID)
    n = momentum.mesh.n_cells
    velocity = jnp.stack([jnp.full(n, LID / 3.0), jnp.zeros(n)], axis=1)
    state = momentum.pack(velocity, jnp.zeros(n))
    continuity, field = _scales(momentum, state)

    throughput, _ = momentum.momentum_matrix_diagonal_parts(velocity)
    mean_speed = float(jnp.mean(jnp.abs(velocity)))
    assert mean_speed > REST_SPEED_FRACTION * LID
    np.testing.assert_array_equal(field, [mean_speed, mean_speed, 1.0])
    np.testing.assert_array_equal(continuity, np.abs(np.asarray(throughput)) + 1e-300)


def test_a_domain_nothing_drives_is_refused_at_rest_naming_the_measure() -> None:
    """With no drive there is no speed to floor by, so a state at rest is refused with the remedy."""
    momentum = _cavity(0.0)
    with pytest.raises(
        ValueError, match=r"row-scaled residual measure is undefined.*Euclidean\(\)"
    ):
        _scales(momentum, momentum.initial_state())


def test_a_domain_nothing_drives_is_still_measured_once_its_state_has_flow() -> None:
    """The refusal is for rest only: an undriven domain's moving state keeps its own scales.

    The field scale is the mean magnitude over every velocity component, so a stream of 0.5 along x
    alone scales at 0.25.
    """
    momentum = _cavity(0.0)
    n = momentum.mesh.n_cells
    velocity = jnp.stack([jnp.full(n, 0.5), jnp.zeros(n)], axis=1)
    _, field = _scales(momentum, momentum.pack(velocity, jnp.zeros(n)))
    np.testing.assert_array_equal(field, [0.25, 0.25, 1.0])
