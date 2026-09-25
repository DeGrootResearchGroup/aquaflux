"""The pressure datum: named by a point, resolved to a cell, and required exactly where the level is free."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import (
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PinnedPoint,
    PressureOutlet,
    VelocityInlet,
    refuse_an_unsuitable_pressure_datum,
)
from aquaflux.mesh import permute_cells, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss

_FLUID = PropertyModel({"viscosity": Constant(0.1), "density": Constant(1.0)})
_CAVITY = {
    "top": MovingWall(velocity=(1.0, 0.0)),
    "bottom": NoSlipWall(),
    "left": NoSlipWall(),
    "right": NoSlipWall(),
}
_DUCT = {
    "left": VelocityInlet(velocity=(1.0, 0.0)),
    "right": PressureOutlet(pressure=0.0),
    "bottom": NoSlipWall(),
    "top": NoSlipWall(),
}


def _build(mesh, boundary, datum):
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        _FLUID,
        BoundaryConditions(boundary),
        gradient_scheme=CompactGreenGauss(),
        pressure_datum=datum,
    )


def _centroids(points):
    """A stand-in geometry holding only cell centroids -- all a point needs to find its cell."""
    return SimpleNamespace(cell=SimpleNamespace(centroid=jnp.asarray(points, dtype=float)))


# --- where the level is fixed: required exactly when no patch fixes it ------------------------


def test_only_a_pressure_outlet_prescribes_the_pressure() -> None:
    assert PressureOutlet(pressure=0.0).prescribes_pressure()
    for closure in (
        NoSlipWall(),
        MovingWall(velocity=(1.0, 0.0)),
        VelocityInlet(velocity=(1.0, 0.0)),
    ):
        assert not closure.prescribes_pressure()


def test_a_closed_domain_without_a_datum_is_refused_rather_than_solved_singular() -> None:
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    with pytest.raises(ValueError, match=r"no boundary patch prescribes the pressure.*PinnedPoint"):
        _build(mesh, _CAVITY, None)


def test_a_datum_beside_an_outlet_is_refused_rather_than_over_determining_the_level() -> None:
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    with pytest.raises(ValueError, match=r"'right' already fixes the pressure level"):
        _build(mesh, _DUCT, PinnedPoint((0.0, 0.0)))


def test_each_domain_builds_with_what_its_boundary_asks_for() -> None:
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    assert _build(mesh, _DUCT, None).pressure_pin is None
    assert _build(mesh, _CAVITY, PinnedPoint((0.0, 0.0))).pressure_pin is not None


def test_the_rule_names_every_patch_that_fixes_the_level() -> None:
    boundary = BoundaryConditions(
        {**_DUCT, "left": PressureOutlet(pressure=1.0), "right": PressureOutlet(pressure=0.0)}
    )
    with pytest.raises(ValueError, match=r"'left', 'right' already fix the pressure level"):
        refuse_an_unsuitable_pressure_datum(boundary, PinnedPoint((0.0, 0.0)), "here")


# --- a point finds its cell --------------------------------------------------------------------


def test_the_nearest_centroid_carries_the_datum() -> None:
    geometry = _centroids([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert PinnedPoint((2.2, 0.4)).cell(geometry) == 2
    assert PinnedPoint((-5.0, 0.0)).cell(geometry) == 0
    assert PinnedPoint((9.0, 1.0)).cell(geometry) == 3


def test_a_point_equally_near_two_cells_takes_the_lower_numbered() -> None:
    geometry = _centroids([[0.0, 0.0], [2.0, 0.0], [1.0, 5.0], [1.0, -5.0]])
    assert PinnedPoint((1.0, 0.0)).cell(geometry) == 0


def test_a_point_of_the_wrong_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match=r"has 3 coordinates, but the mesh is 2-dimensional"):
        PinnedPoint((0.0, 0.0, 0.0)).cell(_centroids([[0.0, 0.0]]))


@pytest.mark.parametrize(
    ("point", "value", "match"),
    [
        ((0.0,), 0.0, "two or three coordinates, got 1"),
        ((0.0, 0.0, 0.0, 0.0), 0.0, "two or three coordinates, got 4"),
        ((float("nan"), 0.0), 0.0, "must be finite"),
        ((0.0, 0.0), float("inf"), "must be finite"),
    ],
)
def test_a_datum_that_names_no_place_is_refused(point, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        PinnedPoint(point, value)


# --- what the datum does to the residual -------------------------------------------------------


def test_the_pinned_row_holds_the_pressure_at_the_datum_value() -> None:
    """At the resolved cell continuity is replaced by ``p - value``; every other row is untouched."""
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    geometry = mesh.geometry()
    target = 5
    point = tuple(np.asarray(geometry.cell.centroid[target]))
    pinned = _build(mesh, _CAVITY, PinnedPoint(point, value=2.5))
    assert pinned.pressure_pin == target

    state = jnp.asarray(np.random.default_rng(0).normal(size=pinned.layout.size))
    _, pressure = pinned.unpack(state)
    _, continuity = pinned.unpack(pinned.residual(state))
    assert continuity[target] == pytest.approx(float(pressure[target]) - 2.5)
    elsewhere = _build(mesh, _CAVITY, PinnedPoint(tuple(np.asarray(geometry.cell.centroid[0]))))
    _, other = elsewhere.unpack(elsewhere.residual(state))
    unpinned_rows = np.setdiff1d(np.arange(mesh.n_cells), [0, target])
    np.testing.assert_array_equal(
        np.asarray(continuity)[unpinned_rows], np.asarray(other)[unpinned_rows]
    )


def test_a_point_pins_the_same_physical_cell_however_the_cells_are_numbered() -> None:
    """The property an index lacks: renumber the cells, and the datum still lands in the same place."""
    mesh = structured_grid_2d(5, 4, named_boundaries=True)
    permutation = np.random.default_rng(1).permutation(mesh.n_cells)
    renumbered = permute_cells(mesh, jnp.asarray(permutation))
    datum = PinnedPoint((0.33, 0.61))

    original = _build(mesh, _CAVITY, datum)
    moved = _build(renumbered, _CAVITY, datum)
    assert moved.pressure_pin != original.pressure_pin  # the numbering really did change
    np.testing.assert_allclose(
        np.asarray(renumbered.geometry().cell.centroid[moved.pressure_pin]),
        np.asarray(mesh.geometry().cell.centroid[original.pressure_pin]),
    )
