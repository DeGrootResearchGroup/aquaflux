"""Radiative transfer of ultraviolet light through an absorbing medium.

The package computes the fluence rate — the radiant power arriving at a point from every
direction, the quantity that governs ultraviolet disinfection — on the same unstructured mesh
the flow is solved on, by summing the contribution of every emitting surface element at every
receiver. Contributions are attenuated exponentially through the absorbing water, blocked by
intervening geometry, and closed over diffuse reflection from the surfaces themselves.

Built so far: the geometric kernels every later stage composes — the exact closed-form solid
angle of a triangle at a point, in the two forms the two receiver kinds need — together with the
surface set itself, read from an STL file, checked for the winding defects that would silently
delete part of a source, and refined until each facet is small compared with its distance to the
nearest receiver.
"""

from __future__ import annotations

from aquaflux.radiation.checks import (
    WindingReport,
    check_winding,
    stored_normal_disagreement,
    winding_report,
)
from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle
from aquaflux.radiation.stl import TriangleSoup, read_stl
from aquaflux.radiation.subdivide import Subdivision, refine_for_receivers, subdivide_to_width
from aquaflux.radiation.surfaces import Surfaces

__all__ = [
    "Subdivision",
    "Surfaces",
    "TriangleSoup",
    "WindingReport",
    "check_winding",
    "projected_solid_angle",
    "read_stl",
    "refine_for_receivers",
    "solid_angle",
    "stored_normal_disagreement",
    "subdivide_to_width",
    "winding_report",
]
