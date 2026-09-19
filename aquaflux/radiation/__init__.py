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
nearest receiver; the angular distributions a source emits with; the gather that sums every source
at every receiver to give the fluence rate and the irradiance; and the absorbing medium between
them, uniform in closed form or graded on a grid and integrated exactly along each path; and the
solid bodies that stand in the way, whose shadows are frozen when the model is built while their
transmittance stays a live, differentiable number.

Still to come: occlusion by the emitting geometry's own triangles, and the reflection system that
closes diffuse interreflection between surfaces.
"""

from __future__ import annotations

from aquaflux.radiation.absorption import Absorption, UniformAbsorption, VoxelAbsorption
from aquaflux.radiation.checks import (
    WindingReport,
    check_profiles,
    check_winding,
    stored_normal_disagreement,
    winding_report,
)
from aquaflux.radiation.gather import fluence_rate, irradiance
from aquaflux.radiation.occluders import Cylinder, HalfSpace, Occluder
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian, Profile
from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle
from aquaflux.radiation.stl import TriangleSoup, read_stl
from aquaflux.radiation.subdivide import Subdivision, refine_for_receivers, subdivide_to_width
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import Visibility, build_visibility

__all__ = [
    "Absorption",
    "CosinePower",
    "Cylinder",
    "HalfSpace",
    "Isotropic",
    "Lambertian",
    "Occluder",
    "Profile",
    "Subdivision",
    "Surfaces",
    "TriangleSoup",
    "UniformAbsorption",
    "Visibility",
    "VoxelAbsorption",
    "WindingReport",
    "build_visibility",
    "check_profiles",
    "check_winding",
    "fluence_rate",
    "irradiance",
    "projected_solid_angle",
    "read_stl",
    "refine_for_receivers",
    "solid_angle",
    "stored_normal_disagreement",
    "subdivide_to_width",
    "winding_report",
]
