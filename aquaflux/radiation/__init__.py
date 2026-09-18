"""Radiative transfer of ultraviolet light through an absorbing medium.

The package computes the fluence rate — the radiant power arriving at a point from every
direction, the quantity that governs ultraviolet disinfection — on the same unstructured mesh
the flow is solved on, by summing the contribution of every emitting surface element at every
receiver. Contributions are attenuated exponentially through the absorbing water, blocked by
intervening geometry, and closed over diffuse reflection from the surfaces themselves.

This module currently provides the geometric kernels that every later stage composes: the exact
closed-form solid angle of a triangle at a point, in the two forms the two receiver kinds need.
"""

from __future__ import annotations

from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle

__all__ = ["projected_solid_angle", "solid_angle"]
