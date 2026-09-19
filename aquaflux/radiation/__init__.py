"""Radiative transfer of ultraviolet light through an absorbing medium.

The package computes the fluence rate — the radiant power arriving at a point from every
direction, the quantity that governs ultraviolet disinfection — on the same unstructured mesh
the flow is solved on, by summing the contribution of every emitting surface element at every
receiver. Contributions are attenuated exponentially through the absorbing water, blocked by
intervening geometry, and closed over diffuse reflection from the surfaces themselves.

Start at :func:`~aquaflux.radiation.model.build_radiation_model`, which freezes everything a
scene's shape decides, and then ask the model for what you need::

    surfaces = Surfaces.from_triangles(read_stl("reactor.stl").vertices, ...)
    model = build_radiation_model(cell_centres, surfaces, occluders=[sleeve])

    G, cycles = fluence_rate(model, surfaces, absorption=UniformAbsorption(a))

The build is the expensive step and depends only on geometry, so a design study that sweeps lamp
power, wall reflectance or water quality pays it once and solves many times — with the
derivatives reaching every one of the swept values. Optics are supplied per call through the
surface set, which is why the set is passed again above rather than being held by the model.

Underneath are the pieces the model composes, each usable on its own: the exact closed-form
solid angle of a triangle at a point, in the two forms the two receiver kinds need; the surface
set itself, read from an STL file, checked for the winding defects that would silently delete
part of a source, and refined until each facet is small compared with its distance to the
nearest receiver; the angular distributions a source emits with; the direct gathers that sum
every source at every receiver to give the fluence rate and the irradiance; the absorbing medium
between them, uniform in closed form or graded on a grid and integrated exactly along each path;
the solid bodies that stand in the way — analytic primitives whose transmittance stays live, and
the emitting surface's own triangles, which let a bent duct shadow itself; and the surface
transfer system that closes diffuse interreflection between facets to convergence, so the number
of bounces is not a parameter.
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
from aquaflux.radiation.gather import direct_fluence_rate, direct_irradiance
from aquaflux.radiation.model import (
    RadiationModel,
    RadiationSettings,
    build_radiation_model,
    fluence_rate,
    radiosity,
    surface_irradiance,
)
from aquaflux.radiation.occluders import Cylinder, HalfSpace, Occluder
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian, Profile
from aquaflux.radiation.quadrature import TriangleQuadrature, triangle_quadrature
from aquaflux.radiation.solid_angle import projected_solid_angle, solid_angle
from aquaflux.radiation.stl import TriangleSoup, read_stl
from aquaflux.radiation.subdivide import Subdivision, refine_for_receivers, subdivide_to_width
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import (
    TransferMatrix,
    build_transfer,
    reciprocity_residual,
    row_sum_error,
)
from aquaflux.radiation.triangles import segment_is_cut
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
    "RadiationModel",
    "RadiationSettings",
    "Subdivision",
    "Surfaces",
    "TransferMatrix",
    "TriangleQuadrature",
    "TriangleSoup",
    "UniformAbsorption",
    "Visibility",
    "VoxelAbsorption",
    "WindingReport",
    "build_radiation_model",
    "build_transfer",
    "build_visibility",
    "check_profiles",
    "check_winding",
    "direct_fluence_rate",
    "direct_irradiance",
    "fluence_rate",
    "projected_solid_angle",
    "radiosity",
    "read_stl",
    "reciprocity_residual",
    "refine_for_receivers",
    "row_sum_error",
    "segment_is_cut",
    "solid_angle",
    "stored_normal_disagreement",
    "subdivide_to_width",
    "surface_irradiance",
    "triangle_quadrature",
    "winding_report",
]
