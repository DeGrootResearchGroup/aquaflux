"""Unit tests for the property model (physics-free, no mesh geometry)."""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
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


def test_constant_scaled_multiplies_the_value() -> None:
    scaled = Constant(value=1e-5).scaled(10.0)
    assert isinstance(scaled, Constant)
    assert scaled.value == 1e-4


def test_scaling_an_array_valued_constant_shares_compiled_code_and_a_float_does_not() -> None:
    """A rescaled property shares a jitted consumer's compiled code only if its value is an ARRAY.

    ``scaled`` exists for Reynolds-number continuation, whose every rung rescales the same property. A
    plain Python float is not a JAX array, so it lands on the *static* side of a jitted function and is
    compared by value -- so each rung hands every function taking the owning assembler a fresh cache
    key. On the coupled solve that is a full recompilation per rung. As an array the rungs differ in a
    leaf value and the compilation is shared.

    Both halves are asserted, because the float behaviour is deliberate (the library keeps property
    values plain scalars) and a caller who needs sharing has to opt in rather than discover this.
    """
    traces: list[int] = []
    zones = CellZones.default(4)

    @eqx.filter_jit
    def consume(prop: Constant) -> jnp.ndarray:
        traces.append(1)
        return jnp.sum(prop.evaluate(zones, {}))

    array_valued = Constant(value=jnp.asarray(1e-5))
    consume(array_valued)
    compiled = len(traces)
    assert compiled == 1
    consume(array_valued.scaled(10.0))
    assert len(traces) == compiled  # a rescaled rung reuses the compilation

    traces.clear()
    float_valued = Constant(value=1e-5)
    consume(float_valued)
    consume(float_valued.scaled(10.0))
    assert len(traces) == 2  # ...where a float value recompiles per rung


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


def test_zone_constant_scaled_multiplies_every_zone() -> None:
    zones = _two_zones()
    scaled = ZoneConstant.from_dict(zones, {"fluid": 1e-3, "solid": 15.0}).scaled(2.0)
    np.testing.assert_allclose(np.asarray(scaled.evaluate(zones, {})), [2e-3] * 3 + [30.0] * 3)


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


def test_field_property_scaled_multiplies_the_field() -> None:
    scaled = FieldProperty(values=jnp.array([1.0, 2.0, 3.0])).scaled(4.0)
    np.testing.assert_allclose(np.asarray(scaled.values), [4.0, 8.0, 12.0])


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


def test_property_model_with_scaled_rescales_only_the_named_property() -> None:
    model = PropertyModel(
        properties={"viscosity": Constant(value=1e-5), "density": Constant(value=1.2)}
    )
    scaled = model.with_scaled("viscosity", 10.0)
    assert scaled.properties["viscosity"].value == 1e-4  # rescaled
    assert scaled.properties["density"].value == 1.2  # untouched
    # The original is unchanged (immutable).
    assert model.properties["viscosity"].value == 1e-5


def test_property_model_with_scaled_requires_the_named_property() -> None:
    model = PropertyModel(properties={"density": Constant(value=1.0)})
    with pytest.raises(ValueError, match="missing required property"):
        model.with_scaled("viscosity", 2.0)


def test_property_model_evaluate_threads_state_fields() -> None:
    """``evaluate`` accepts and forwards a state-field mapping (the state-dependent seam)."""
    zones = _two_zones()
    model = PropertyModel(properties={"density": Constant(value=1.0)})
    props = model.evaluate(zones, {"temperature": jnp.full(zones.label.shape[0], 300.0)})
    assert jnp.allclose(props["density"], 1.0)
