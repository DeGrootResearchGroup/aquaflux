"""Unit tests for threading named state fields from the residual assembler to the property model.

:meth:`~aquaflux.discretization.ResidualAssembler.residual` and
:meth:`~aquaflux.discretization.ResidualAssembler.gradient` accept an optional ``fields`` mapping
and forward it verbatim to :meth:`~aquaflux.properties.PropertyModel.evaluate`. No shipped
property reads it yet (a state-dependent property is a later addition), so this is plumbing rather
than physics: a stub property that records what it was handed proves the mapping actually reaches
:class:`~aquaflux.properties.property.Property.evaluate`, independent of any future consumer.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.boundary import BoundaryConditions, ZeroGradient
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, Property, PropertyModel


class _RecordingProperty(Property):
    """A property that ignores the partition and records the ``fields`` mapping it was handed."""

    seen: list[object]

    def evaluate(self, cell_zones, fields):
        self.seen.append(dict(fields))
        return jnp.ones(cell_zones.label.shape[0])

    def scaled(self, factor):
        raise NotImplementedError


def _assembler(properties):
    mesh = structured_grid_2d(2, 1)
    return mesh, ResidualAssembler.build(
        mesh,
        mesh.geometry(),
        properties,
        (DiffusionFlux(coefficient="diffusivity"),),
        BoundaryConditions({"boundary": ZeroGradient()}),
    )


def test_residual_forwards_fields_to_the_property_model() -> None:
    seen: list[object] = []
    mesh, asm = _assembler(PropertyModel({"diffusivity": _RecordingProperty(seen=seen)}))
    state = jnp.array([300.0, 310.0])
    asm.residual(jnp.zeros(mesh.n_cells), fields={"temperature": state})
    assert len(seen) == 1
    assert seen[0].keys() == {"temperature"}
    assert jnp.array_equal(seen[0]["temperature"], state)


def test_gradient_forwards_fields_to_the_property_model() -> None:
    seen: list[object] = []
    mesh, asm = _assembler(PropertyModel({"diffusivity": _RecordingProperty(seen=seen)}))
    state = jnp.array([1.0, 2.0])
    asm.gradient(jnp.zeros(mesh.n_cells), fields={"k": state})
    assert len(seen) == 1
    assert seen[0].keys() == {"k"}
    assert jnp.array_equal(seen[0]["k"], state)


def test_residual_with_no_fields_hands_the_property_model_an_empty_mapping() -> None:
    """Omitting ``fields`` is exactly today's behaviour -- every property sees ``{}``."""
    seen: list[object] = []
    mesh, asm = _assembler(PropertyModel({"diffusivity": _RecordingProperty(seen=seen)}))
    asm.residual(jnp.zeros(mesh.n_cells))
    assert seen == [{}]


def test_residual_is_unaffected_by_fields_when_no_property_reads_them() -> None:
    """A state-independent property model gives the identical residual regardless of ``fields``."""
    _mesh, asm = _assembler(PropertyModel({"diffusivity": Constant(value=2.0)}))
    phi = jnp.array([1.0, 4.0])
    without = asm.residual(phi)
    with_fields = asm.residual(phi, fields={"temperature": jnp.array([300.0, 310.0])})
    assert jnp.array_equal(without, with_fields)
