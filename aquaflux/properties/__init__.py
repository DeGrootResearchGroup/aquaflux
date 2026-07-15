"""Properties: per-cell physical property fields, decoupled from the numerics.

A physics model's properties (density, viscosity, conductivity, ...) are described by a
:class:`PropertyModel` — a named collection of :class:`Property` objects, each a *single*
property that evaluates to a per-cell array. Operators name the property they consume; the model
resolves the name. Kept a separate layer (not part of discretization) so properties are a physical
concern operators compose, not something the numerics own.
"""

from __future__ import annotations

from .model import PropertyModel
from .property import Constant, FieldProperty, Property, ZoneConstant

__all__ = [
    "Constant",
    "FieldProperty",
    "Property",
    "PropertyModel",
    "ZoneConstant",
]
