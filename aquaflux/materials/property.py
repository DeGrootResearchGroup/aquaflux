"""Material properties: per-cell physical property fields (density, viscosity, conductivity, ...).

A :class:`MaterialProperty` is a *single* named property that evaluates to a per-cell array from the
cell partition and (for state-dependent kinds) the current state fields. Three field-independent
kinds are built here, from least to most spatially resolved:

- :class:`Constant` — one uniform value everywhere.
- :class:`ZoneConstant` — a separate constant per named cell zone (fluid vs. solid conductivity, a
  per-zone porosity, ...), broadcast through the :class:`~aquaflux.mesh.CellZones` labels.
- :class:`FieldProperty` — an arbitrary per-cell field of values supplied directly, for a property
  that varies cell-by-cell in a way no zone or constant captures (an externally computed field, or a
  coefficient frozen from a previous solve).

A state-dependent kind (a value computed from a field through a formula, e.g. temperature-dependent
viscosity) is a later addition that consumes the ``fields`` argument.

Property values are **plain scalars**, not wrapped arrays: a property is differentiated by passing
its value as a traced argument (constructing the property inside the differentiated function), the
same pattern the boundary Biot number uses — so a material parameter is a first-class sensitivity /
estimation target with no extra ceremony.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from aquaflux.mesh import CellZones


class MaterialProperty(eqx.Module):
    """A single per-cell physical property, evaluated from the cell partition and state fields."""

    @abc.abstractmethod
    def evaluate(self, cell_zones: CellZones, fields: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        """Per-cell property values, shape ``(n_cells,)``.

        Parameters
        ----------
        cell_zones : CellZones
            The cell partition — supplies the cell count and, for zonal properties, the per-cell
            zone labels.
        fields : mapping of {str: jnp.ndarray}
            Named per-cell state fields a property may depend on; empty (and ignored) for
            state-independent properties.
        """


class Constant(MaterialProperty):
    """A uniform property value everywhere.

    Attributes
    ----------
    value : float
        The property value — a plain scalar; a tracer flows through it when the property is
        differentiated (e.g. a material-parameter sensitivity).
    """

    value: float

    def evaluate(self, cell_zones, fields):
        return jnp.full((cell_zones.label.shape[0],), self.value)


class ZoneConstant(MaterialProperty):
    """A separate constant value per named cell zone.

    Build with :meth:`from_dict`. Each cell's value is looked up by its zone label, so a
    field-independent value is shared across a zone (fluid vs. solid conductivity, a per-zone
    porosity, ...).

    Attributes
    ----------
    values : jnp.ndarray
        Per-zone values, shape ``(n_zones,)``, indexed by zone id — an indexed collection of
        per-zone scalars (each still individually differentiable), not a per-cell field.
    """

    values: jnp.ndarray

    @classmethod
    def from_dict(cls, cell_zones: CellZones, per_zone: Mapping[str, float]) -> ZoneConstant:
        """Build from ``{zone_name: value}``.

        Every zone that actually contains cells must be given a value; a zone with no cells (e.g. an
        empty ``"default"``) may be omitted — its slot is never gathered.

        Parameters
        ----------
        cell_zones : CellZones
            The cell partition whose zones the values are keyed to.
        per_zone : mapping of {str: float}
            The value for each named zone.

        Raises
        ------
        ValueError
            On an unknown zone name, or a populated zone left unspecified.
        """
        values: list[object] = [0.0] * cell_zones.n_groups  # empty zones keep 0.0; never gathered
        for name, value in per_zone.items():
            values[cell_zones.id_of(name)] = value  # id_of raises on an unknown zone name
        populated = {int(z) for z in np.unique(np.asarray(cell_zones.label))}
        specified = {cell_zones.id_of(name) for name in per_zone}
        missing = sorted(populated - specified)
        if missing:
            names = [cell_zones.names[i] for i in missing]
            raise ValueError(f"ZoneConstant.from_dict: no value for populated zone(s) {names}")
        return cls(values=jnp.stack([jnp.asarray(v) for v in values]))

    def evaluate(self, cell_zones, fields):
        return self.values[cell_zones.label]


class FieldProperty(MaterialProperty):
    """An arbitrary per-cell field of property values, supplied directly.

    The point between :class:`Constant` (one value everywhere) and :class:`ZoneConstant` (one value
    per zone): a full per-cell array, for a property that varies cell-by-cell in a way no zone or
    constant captures -- an externally computed field, or a coefficient frozen from a previous solve
    and refreshed between iterations (an effective viscosity ``mu + mu_t`` recomputed each outer
    sweep, say). ``values`` is a differentiable leaf, so gradients flow through the supplied field.

    Attributes
    ----------
    values : jnp.ndarray
        The per-cell property values, shape ``(n_cells,)``.
    """

    values: jnp.ndarray

    def evaluate(self, cell_zones, fields):
        n_cells = cell_zones.label.shape[0]
        if self.values.shape[0] != n_cells:
            raise ValueError(
                f"FieldProperty: values has length {self.values.shape[0]} but the partition has "
                f"{n_cells} cells"
            )
        return self.values
