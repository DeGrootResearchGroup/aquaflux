"""Radiative transfer of ultraviolet light through an absorbing medium.

The package computes the fluence rate — the radiant power arriving at a point from every
direction, the quantity that governs ultraviolet disinfection — on the same unstructured mesh
the flow is solved on, by summing the contribution of every emitting surface element at every
receiver. Contributions are attenuated exponentially through the absorbing water, blocked by
intervening geometry, and closed over diffuse reflection from the surfaces themselves.

**What the method is.** A deterministic backward gather: at each receiver, the exact solid angle of
every emitting triangle, weighted by its radiance, attenuated along the path, and gated by whatever
stands in the way. That is the mainstream formulation of ultraviolet reactor modelling, not a new
one — the multiple segment source summation (MSSS) model, a cylindrical diffuse lamp cut into
segments and summed at each point, is this with the lamp restricted to a cylinder (Liu et al.,
2004, find it the best approximation of a lamp among the summation models). The basic summation
models assume an unobstructed path from lamp to point; extensions such as RAD-LSI (a radial
correction to line source integration) and the commercial UVCalc3D add shadowing by the sleeves of
neighbouring lamps in a multi-lamp array (Liu et al., 2005). What this adds is shadowing by
arbitrary triangulated geometry rather than by lamp sleeves alone, diffuse interreflection from
every surface solved to convergence, and exact derivatives. The sum is evaluated at a point
exactly, so it carries neither the statistical error nor the finite-volume scoring bias of a Monte
Carlo estimate.

Start at :func:`~aquaflux.radiation.model.build_radiation_model`, which freezes everything a
scene's shape decides, and then ask the model for what you need. From a lamp rating and a water
quality, the numbers a reactor engineer has::

    soup = read_stl("reactor.stl")        # bodies named "lamp", "wall", ...
    geometry = Surfaces.from_triangles(
        soup.vertices, solid_id=soup.solid_id, solid_names=soup.solid_names
    )
    surfaces = geometry.with_optics(
        emission=lamp_exitance(geometry, {"lamp": 35.0}),          # a 35 W lamp
        reflectance=geometry.per_facet({"wall": 0.3}, default=0.0),
    )
    model = build_radiation_model(cell_centres, surfaces)

    water = UniformAbsorption(absorption_from_uvt(70.0))          # 70% transmittance per cm
    G, cycles = fluence_rate(model, surfaces, absorption=water)   # W/m^2, one per cell centre

The build is the expensive step and depends only on geometry, so a design study that sweeps lamp
power, wall reflectance or water quality pays it once and solves many times — with the
derivatives reaching every one of the swept values. Optics are supplied per call through the
surface set, which is why the set is passed again above; its geometry must be the one the model
was built for, and a moved surface set is refused rather than silently mixed with the frozen
shadows.

**Units and conventions.** Lengths in metres, exitance in W/m², point-source power in W, so the
fluence rate is in W/m². ⚠️ **The fluence rate carries no receiver cosine; the irradiance does**
— they are different integrals of the radiance, over the whole sphere and over a hemisphere
weighted by the cosine, and many papers use "irradiance" or "intensity" for fluence rate. The
absorption coefficient is **napierian, per metre** (``exp(-a r)``): a decadic one is smaller by
``ln 10`` and a per-centimetre one by a hundred, which :func:`absorption_from_uvt` exists to get
right. An angular profile is a distribution normalized to one over the sphere; the power comes
separately, and :func:`lamp_exitance` spreads a rating over the triangulated area so the model
radiates exactly the rated power at any refinement.

**What is differentiable.** Emission, point-source power, reflectance, profile parameters, the
transmittance of analytic bodies, and the absorption coefficient or graded absorption field —
through the interreflection solve by its adjoint, not by replaying it. **Exactly zero, by
construction:** the derivative with respect to where anything stands in the way. Shadows are
decided once, when the model is built, and frozen; a shadow edge moving with a body's radius is
not seen. The receivers are frozen into the model the same way, so a flow solve's derivative with
respect to mesh-node positions gets no contribution from the fluence rate.

**What it does not model, and what that costs.**

- **Refraction and reflection at a quartz sleeve.** Bolton (2000) puts the error of neglecting
  them at a 6.5% reflection correction below 70% transmittance per centimetre, and up to 25%
  above it. So this is a model for lower-transmittance water — wastewater, or the 70% water of
  the Sozzi & Taghipour (2006) reactor benchmark — and carries a systematic error of that size at
  drinking-water transmittances.
- **Specular reflection.** Walls reflect diffusely. At the same reflectivity, fully specular and
  fully diffuse walls have been measured 10–47% apart in log reduction (Hassanpour et al., 2023),
  so a reflectance is only half a description of a wall.
- **Scattering by the water**, and **more than one waveband**: one absorbing, non-scattering
  medium at one wavelength.
- **A source that is not diffuse** has its distribution evaluated along one direction per pair of
  facets, from centroid to centroid; energy balance is exact only for diffuse sources.
- **Zero-thickness sheets** block from both sides only when named: see
  :class:`~aquaflux.radiation.self_occlusion.SilhouetteOcclusion`.

Underneath are the pieces the model composes, each usable on its own: the exact closed-form
solid angle of a triangle at a point, in the two forms the two receiver kinds need; the surface
set itself, read from an STL file, checked for the winding defects that would silently delete
part of a source, and refined until each facet is small compared with its distance to the
nearest receiver; the angular distributions a source emits with; the direct gathers that sum
every source at every receiver to give the fluence rate and the irradiance; the absorbing medium
between them, uniform in closed form or graded on a grid and integrated exactly along each path;
the solid bodies that stand in the way — analytic primitives whose transmittance stays live, and
the emitting surface's own triangles, which let a bent duct shadow itself, tested either by one ray
per pair or by clipping each source against each blocker's silhouette for the exact covered share;
and the surface transfer system that closes diffuse interreflection between facets to
convergence, so the number of bounces is not a parameter.
"""

from __future__ import annotations

from aquaflux.radiation.absorption import Absorption, UniformAbsorption, VoxelAbsorption
from aquaflux.radiation.checks import (
    WindingReport,
    check_points_outside,
    check_profiles,
    check_winding,
    enclosure_winding,
    open_facets,
    stored_normal_disagreement,
    winding_report,
)
from aquaflux.radiation.culling import BodyCulling, ShaftCulling, EveryPair
from aquaflux.radiation.gather import direct_fluence_rate, direct_irradiance
from aquaflux.radiation.model import (
    RadiationModel,
    RadiationSettings,
    build_radiation_model,
    fluence_rate,
    radiosity,
    surface_irradiance,
)
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian, Profile
from aquaflux.radiation.self_occlusion import (
    NoOcclusion,
    OcclusionField,
    RayCastOcclusion,
    SelfOcclusion,
    SilhouetteOcclusion,
)
from aquaflux.radiation.coarsen import Coarsening, coarsen_surfaces, coarsen_to_size
from aquaflux.radiation.quadrature import TriangleQuadrature, triangle_quadrature
from aquaflux.radiation.receiver_shadows import FrozenShadows, ReceiverShadows, StreamedShadows
from aquaflux.radiation.solid_angle import (
    projected_solid_angle,
    signed_solid_angle,
    solid_angle,
)
from aquaflux.radiation.stl import TriangleSoup, read_stl
from aquaflux.radiation.subdivide import Subdivision, refine_for_receivers, subdivide_to_width
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.triangle_body import TriangleBody
from aquaflux.radiation.transfer import (
    TransferMatrix,
    build_transfer,
    reciprocity_residual,
    row_sum_error,
)
from aquaflux.radiation.triangles import segment_is_cut
from aquaflux.radiation.units import absorption_from_uvt, lamp_exitance
from aquaflux.radiation.visibility import Visibility, build_visibility

__all__ = [
    "Absorption",
    "BodyCulling",
    "Coarsening",
    "CosinePower",
    "EveryPair",
    "FrozenShadows",
    "Isotropic",
    "Lambertian",
    "NoOcclusion",
    "OcclusionField",
    "Profile",
    "RadiationModel",
    "RadiationSettings",
    "RayCastOcclusion",
    "ReceiverShadows",
    "SelfOcclusion",
    "ShaftCulling",
    "SilhouetteOcclusion",
    "StreamedShadows",
    "Subdivision",
    "Surfaces",
    "TransferMatrix",
    "TriangleBody",
    "TriangleQuadrature",
    "TriangleSoup",
    "UniformAbsorption",
    "Visibility",
    "VoxelAbsorption",
    "WindingReport",
    "absorption_from_uvt",
    "build_radiation_model",
    "build_transfer",
    "build_visibility",
    "check_points_outside",
    "check_profiles",
    "check_winding",
    "coarsen_surfaces",
    "coarsen_to_size",
    "direct_fluence_rate",
    "direct_irradiance",
    "enclosure_winding",
    "fluence_rate",
    "lamp_exitance",
    "open_facets",
    "projected_solid_angle",
    "radiosity",
    "read_stl",
    "reciprocity_residual",
    "refine_for_receivers",
    "row_sum_error",
    "segment_is_cut",
    "signed_solid_angle",
    "solid_angle",
    "stored_normal_disagreement",
    "subdivide_to_width",
    "surface_irradiance",
    "triangle_quadrature",
    "winding_report",
]
