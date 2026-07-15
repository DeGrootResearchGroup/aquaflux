"""Unit tests for the property model (physics-free, no mesh geometry)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.mesh import CellZones
from aquaflux.properties import Constant, FieldProperty, PropertyModel, ZoneConstant


def _two_zones(n_cells: int = 6) -> CellZones:
    """A 6-cell partition: cells 0-2 in ``fluid``, 3-5 in ``solid`` (plus an empty ``default``)."""
    return CellZones.from_dict(n_cells, {"fluid": [0, 1, 2], "solid": [3, 4, 5]})


# --- Constant --------------------------------------------------------------------------


def test_constant_broadcasts_to_every_cell() -> None:
    vals = Constant(value=1.2).evaluate(CellZones.default(5), {})
    assert vals.shape == (5,)
    assert jnp.allclose(vals, 1.2)


def test_constant_is_differentiable_in_its_value() -> None:
    zones = CellZones.default(4)
    g = jax.grad(lambda k: jnp.sum(Constant(value=k).evaluate(zones, {})))(2.0)
    assert float(g) == 4.0  # d/dk of sum(k over 4 cells)


# --- ZoneConstant ----------------------------------------------------------------------


def test_zone_constant_maps_each_zone_to_its_value() -> None:
    zones = _two_zones()
    vals = ZoneConstant.from_dict(zones, {"fluid": 1e-3, "solid": 15.0}).evaluate(zones, {})
    np.testing.assert_allclose(np.asarray(vals), [1e-3, 1e-3, 1e-3, 15.0, 15.0, 15.0])


def test_zone_constant_is_differentiable_per_zone() -> None:
    zones = _two_zones()

    def total(k_fluid):
        return jnp.sum(
            ZoneConstant.from_dict(zones, {"fluid": k_fluid, "solid": 15.0}).evaluate(zones, {})
        )

    assert float(jax.grad(total)(1e-3)) == 3.0  # three fluid cells


def test_zone_constant_rejects_unknown_zone() -> None:
    with pytest.raises(ValueError, match="no group named"):
        ZoneConstant.from_dict(_two_zones(), {"plasma": 1.0})


def test_zone_constant_requires_every_populated_zone() -> None:
    with pytest.raises(ValueError, match="no value for populated zone"):
        ZoneConstant.from_dict(_two_zones(), {"fluid": 1e-3})  # 'solid' omitted


def test_zone_constant_allows_empty_zone_omitted() -> None:
    """The empty ``default`` zone need not be given a value (its slot is never gathered)."""
    zones = _two_zones()
    vals = ZoneConstant.from_dict(zones, {"fluid": 2.0, "solid": 3.0}).evaluate(zones, {})
    assert not bool(jnp.any(jnp.isnan(vals)))  # every real cell got a real value


# --- FieldProperty ---------------------------------------------------------------------


def test_field_property_returns_the_supplied_field() -> None:
    values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vals = FieldProperty(values=values).evaluate(CellZones.default(5), {})
    np.testing.assert_allclose(np.asarray(vals), [1.0, 2.0, 3.0, 4.0, 5.0])


def test_field_property_is_differentiable_in_its_field() -> None:
    zones = CellZones.default(3)

    def total(scale):
        return jnp.sum(FieldProperty(values=scale * jnp.array([1.0, 2.0, 3.0])).evaluate(zones, {}))

    assert float(jax.grad(total)(2.0)) == 6.0  # d/dscale of scale * (1 + 2 + 3)


def test_field_property_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="values has length 3 but the partition has 5 cells"):
        FieldProperty(values=jnp.ones(3)).evaluate(CellZones.default(5), {})


# --- PropertyModel ---------------------------------------------------------------------


def test_property_model_evaluates_all_named_properties() -> None:
    zones = _two_zones()
    model = PropertyModel(
        properties={
            "density": Constant(value=1.2),
            "viscosity": ZoneConstant.from_dict(zones, {"fluid": 1e-3, "solid": 1e6}),
        }
    )
    props = model.evaluate(zones)
    assert set(props) == {"density", "viscosity"}
    assert jnp.allclose(props["density"], 1.2)
    np.testing.assert_allclose(np.asarray(props["viscosity"]), [1e-3, 1e-3, 1e-3, 1e6, 1e6, 1e6])


def test_property_model_require_flags_missing_property() -> None:
    model = PropertyModel(properties={"density": Constant(value=1.0)})
    model.require("density")  # present -> no error
    with pytest.raises(ValueError, match="missing required property"):
        model.require("viscosity")


def test_property_model_require_lists_every_missing_property() -> None:
    model = PropertyModel(properties={"density": Constant(value=1.0)})
    with pytest.raises(ValueError, match="missing required properties") as excinfo:
        model.require("viscosity", "conductivity")
    message = str(excinfo.value)
    assert "viscosity" in message and "conductivity" in message


def test_property_model_evaluate_threads_state_fields() -> None:
    """``evaluate`` accepts and forwards a state-field mapping (the state-dependent seam)."""
    zones = _two_zones()
    model = PropertyModel(properties={"density": Constant(value=1.0)})
    props = model.evaluate(zones, {"temperature": jnp.full(zones.label.shape[0], 300.0)})
    assert jnp.allclose(props["density"], 1.0)
