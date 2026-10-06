# Ultraviolet radiation

An ultraviolet (UV) reactor disinfects a fluid — water, or air in a duct or a room — by exposing it
to light from one or more lamps, and what an organism receives depends on the **fluence rate** at
every point it passes: the radiant power arriving there from every direction, per unit area.
`aquaflux.radiation` computes that field on the same points the flow is solved on — usually the
cell centres — so it can drive a dose or reaction calculation directly, and because it is written
in JAX, every number it returns can be differentiated with respect to lamp power, wall
reflectance, the absorbance of the fluid and the other optical inputs.

This page covers what the model computes and how, how to set up a scene, what is and is not
differentiable, how to run it at the size of a real reactor mesh, and what it has been checked
against. Reading the reactor geometry from a CAD drawing is covered separately, in
[Reactor geometry from CAD](cad_geometry.md).

## What the model computes

At each receiver point the fluence rate is gathered, facet by facet, from a triangulated emitting
surface:

- each **emitting triangle** contributes its radiance times the exact, closed-form solid angle it
  subtends at the receiver — not an inverse-square approximation, which is badly wrong in the near
  field where the fluence rate is highest;
- the contribution is **attenuated** along the straight path by the absorbing medium — the water or
  air being treated (Beer–Lambert);
- it is **blocked**, wholly or in part, by any geometry standing in the way — lamp sleeves, baffles,
  the vessel wall, or the emitting surface itself where it is not convex;
- and every surface that **reflects** re-emits what lands on it diffusely, which lights every other
  surface in turn. That interreflection is solved as a linear system, so the number of bounces is
  not a parameter: the solution is the infinite bounce sum.

This is a deterministic backward gather, and it is the mainstream formulation of UV reactor
modelling rather than a new one: the multiple segment source summation (MSSS) model — a cylindrical
diffuse lamp cut into segments and summed at each point — is this method with the lamp restricted
to a cylinder, and Liu et al. (2004) find it the best approximation of a lamp among the summation
models. Extensions such as RAD-LSI and UVCalc3D add shadowing by the sleeves of neighbouring lamps
(Liu et al., 2005). What `aquaflux` adds is shadowing by arbitrary triangulated geometry, diffuse
interreflection between every pair of surfaces solved to convergence, and exact derivatives.
Because the sum is evaluated at each point exactly, the field carries neither the statistical
noise of a Monte Carlo estimate nor the ray effect of a discrete-ordinates solve (see
[Verification](#verification)). The exact equations are set out in [Theory](#theory).

## Quick start

The calculation has two steps. {func}`~aquaflux.radiation.build_radiation_model` freezes
everything the scene's **shape** decides; {func}`~aquaflux.radiation.fluence_rate` then solves for
the field given the scene's **optics** — what each surface emits and reflects, and what the medium
absorbs.

The smallest complete scene is a closed box whose walls glow and reflect, and its answer can be
checked by hand. Walls that all emit an exitance (power per unit area) `M` and reflect a fraction
`rho` settle to a uniform radiosity — all a wall sends out, emitted plus reflected — of
`B = M / (1 - rho)`. Every wall then has radiance `B / pi` in every direction, so the fluence rate
at *any* point inside is `G = 4 pi (B / pi) = 4B`. The walls of a box mesh make such an enclosure:
{func}`~aquaflux.mesh.patch_triangles` cuts the mesh's boundary patches into triangles facing into
the domain, and the cell centres are the receivers.

```python
from aquaflux.mesh import patch_triangles, structured_grid_3d
from aquaflux.radiation import Surfaces, build_radiation_model, fluence_rate

mesh = structured_grid_3d(6, 6, 6, named_boundaries=True)     # the unit cube, 216 cells
geometry = mesh.geometry()
cells = geometry.cell.centroid                                 # receivers, (216, 3)

walls = patch_triangles(mesh, geometry, ["left", "right", "bottom", "top", "back", "front"])
box = Surfaces.from_triangles(
    walls.vertices, solid_id=walls.patch_id, solid_names=walls.patch_names
)
model = build_radiation_model(cells, box)

G, cycles = fluence_rate(model, box.with_optics(emission=10.0, diffuse_reflectance=0.5))
print(G.min(), G.max(), cycles)    # 80 and 80 to rounding, 3 cycles: G = 4 * 10 / (1 - 0.5)
```

The first call takes some tens of seconds, most of it compiling; later calls on the same model
reuse the compiled program. `G` is a plain array with one entry per receiver, in their order.
`cycles` is how many restart cycles the reflection solve took: a solve that fails to converge
raises, so the count is the cost of the scene rather than part of its answer. Drop it with
`G, _ = fluence_rate(...)` when only the field is wanted.

Two things in this example are general. The model is built from the surfaces' geometry alone, and
the optics arrive only at the call — so the same `model` answers any emission and reflectance on
these walls. And the answer is exact at any mesh: this closed form is one of the checks the
implementation is tested against, at reflectances up to 0.9, where a single bounce would give
less than a fifth of the answer.

### A reactor of your own

The same two steps, for a lamp and a wall read from a file:

```python
from aquaflux.radiation import (
    Surfaces, UniformAbsorption, absorption_from_uvt, build_radiation_model,
    fluence_rate, lamp_exitance, read_stl,
)

soup = read_stl("reactor.stl")                                     # bodies "lamp" and "wall"
geometry = Surfaces.from_triangles(
    soup.vertices, solid_id=soup.solid_id, solid_names=soup.solid_names
)
surfaces = geometry.with_optics(
    emission=lamp_exitance(geometry, {"lamp": 35.0}),              # a 35 W lamp
    diffuse_reflectance=geometry.per_facet({"wall": 0.3}, default=0.0),
)
cell_centres = mesh.geometry().cell.centroid                       # or any (n, 3) array of points
model = build_radiation_model(cell_centres, surfaces)

medium = UniformAbsorption(absorption_from_uvt(70.0))              # water at 70% UVT
G, cycles = fluence_rate(model, surfaces, absorption=medium)       # W/m^2, one per cell centre
```

- ⚠️ **An STL file has no units, and `read_stl` does not scale it.** Everything here is in metres,
  and drawings are very often exported in millimetres. Read in as metres, a millimetre drawing is a
  thousand times too large: every absorption path is a thousand times too long and the lamp's
  exitance a million times too small, with no error. Scale it first — `soup.vertices * 1e-3` for
  millimetres. (The CAD reader always returns metres.)
- **Bodies are how optics are assigned**, so the file must name them. An ASCII STL may hold several
  named solids (`solid lamp` … `endsolid`), and each becomes a body. A binary STL holds one unnamed
  body, named after the file; for several, read one file per body and concatenate their vertices,
  with a `solid_id` saying which body each triangle came from.
- The receivers are any points; with a mesh they are usually its cell centroids. For air, and for
  the other choices the medium involves, see [The medium](#the-medium).

## Units and conventions

- **Lengths in metres, exitance in W/m², point-source power in W**, so the fluence rate is in W/m².
  Much UV practice, air disinfection especially, quotes µW/cm² and mJ/cm²:
  **1 W/m² = 100 µW/cm²**, and a dose of **1 J/m² = 0.1 mJ/cm²**.
- ⚠️ **The fluence rate carries no receiver cosine; the irradiance does.** The fluence rate is the
  radiance integrated over the whole sphere, because the organism it acts on has no orientation.
  The irradiance on a surface is the radiance integrated over a hemisphere and weighted by the
  cosine to that surface's normal. They are different quantities, and much of the literature uses
  "irradiance" or "intensity" for the fluence rate.
- **The absorption coefficient is napierian, per metre** — the `a` in `exp(-a r)`. A *decadic*
  coefficient (paired with `10^(-A r)`) is smaller by `ln 10`, and a per-*centimetre* one by a
  hundred; either, passed where a napierian per-metre one is expected, reads as a clearer medium
  rather than as an error. For water, {func}`~aquaflux.radiation.absorption_from_uvt` converts the
  number a water-quality report carries — the percentage UV transmittance (UVT) through a
  one-centimetre cell — into the right one: 95% UVT is 5.129 per metre. It takes a
  **percentage**, and refuses a fraction such as `0.95`, which would otherwise be read as 0.95% and
  give a coefficient ninety times too large.
- **A lamp is specified by its rated UV output.** {func}`~aquaflux.radiation.lamp_exitance`
  divides each body's rating by the area of **its triangles**, not by the area of the shape they
  approximate, so the model radiates exactly the rated power at any refinement. Dividing by the
  analytic `pi d L` of a cylinder instead makes an inscribed triangulation radiate less than its
  rating — 2.6% less at eight sectors around the circumference — with no warning. Which body
  receives the rating is a modelling decision it cannot make for you: a lamp is rated at its
  envelope, while the geometry in a reactor model is usually the slightly larger quartz sleeve.
  The rating is also taken as given: a low-pressure lamp's output depends on its wall temperature,
  so in a cooled air stream or a cold duct, pass the output at those operating conditions.

## Setting up a scene

A scene has four parts: the **surfaces** that emit and reflect, the **medium** between them, the
**bodies** that stand in the way, and the **receivers** where the field is wanted.

### The surfaces

{class}`~aquaflux.radiation.Surfaces` is a set of triangles with per-facet optical properties.
Build it with {meth}`~aquaflux.radiation.Surfaces.from_triangles`, which derives each facet's
centroid, area and normal from its vertices so they cannot disagree. Triangles come from an STL
file ({func}`~aquaflux.radiation.read_stl`, ASCII or binary, keeping each named solid as a body),
from a mesh's boundary patch ({func}`~aquaflux.mesh.patch_triangles`), or from a CAD drawing
([Reactor geometry from CAD](cad_geometry.md)).

- **The normal comes from the winding**, by the right-hand rule, and a facet emits only on the side
  its normal points to. So a lamp's triangles must face outward, into the fluid, and a vessel's
  walls inward. A facet wound backwards emits nothing, silently —
  `check_winding(surfaces.vertices)` ({func}`~aquaflux.radiation.check_winding`) raises if any
  shared edge is traversed the same way by both of its triangles, which is how an inconsistently
  wound file shows itself, and {func}`~aquaflux.radiation.winding_report` returns the same analysis
  without raising. Both take the vertex array, not the surface set.
- **Optics are set per named body.** {meth}`~aquaflux.radiation.Surfaces.per_facet` expands a
  `{body: value}` mapping into a per-facet array and raises on a body name it does not know, so a
  misspelled `"lmap"` fails rather than emitting nothing.
  {meth}`~aquaflux.radiation.Surfaces.with_optics` returns a copy with new emission, reflectances,
  power or angular profiles and the same geometry, which is how a study varies them.
- **A zero-area facet is a point source.** It carries a radiant `power` in watts rather than an
  exitance, emits isotropically — give it the {class}`~aquaflux.radiation.Isotropic` profile, since
  it has no normal for a directional one to be measured against, and
  {func}`~aquaflux.radiation.check_profiles` refuses any other — and contributes `P / (4 pi r^2)`
  at each receiver. Point sources emit but do not reflect. A 5 W point at the origin:

  ```python
  import numpy as np
  from aquaflux.radiation import Isotropic, Surfaces

  point = Surfaces.from_triangles(np.zeros((1, 3, 3)), power=5.0, profiles=(Isotropic(),))
  ```

  A set that mixes point sources with emitting facets lists both profiles and gives each facet
  its index (`profile_index`).
- **Each facet emits with an angular profile**, a distribution normalized to one over the sphere:
  {class}`~aquaflux.radiation.Lambertian` (the default, and how every reflected ray leaves), or
  {class}`~aquaflux.radiation.CosinePower`, a narrowed beam whose exponent is a differentiable
  parameter and which reduces exactly to Lambertian at an exponent of one;
  {class}`~aquaflux.radiation.Isotropic` is for point sources only. A set may mix them — pass the
  distinct profiles and a per-facet index.
- **A measured luminaire is read from its IES file.** {func}`~aquaflux.radiation.read_ies` parses an
  LM-63 photometric file (Type C, no tilt) into a {class}`~aquaflux.radiation.Photometry`, whose
  {attr}`~aquaflux.radiation.Photometry.flux` is the table's exact integrated flux and whose
  {meth}`~aquaflux.radiation.Photometry.profile` gives a
  {class}`~aquaflux.radiation.PhotometricProfile` for the lamp window's facets. The vertical angle is
  measured from each facet's normal and the horizontal angle round it from a direction `up` you
  choose, so the window's facets must face along the fixture's beam; the table is bilinear between
  its tabulated directions and completed round the circle by its declared symmetry. The profile
  varies around its axis as well as away from it, which is why a profile is asked about a direction
  and a facet normal rather than a single angle ({class}`~aquaflux.radiation.AxisymmetricProfile`
  is the family that depends on the angle alone). The direct gathers take it; the surface transfer,
  which freezes one cosine per pair of facets, refuses it. ⚠️ The file's units are its own — candela
  by default, sometimes milliwatts per steradian stated in a keyword — so convert the flux to watts
  yourself before giving it as the lamp's power.

**Receivers must be far enough from a facet for that facet to be resolved.** Each facet's
contribution is exact, but the facet's radiance and its attenuation are evaluated once per facet,
so a facet much larger than its distance to the nearest receiver smears what it emits.
{func}`~aquaflux.radiation.refine_for_receivers` splits facets until each one's longest edge is at
most `max_ratio` times its distance to the nearest receiver (0.25 by default; 0.15 and 0.05 are
the usual choices for accurate work) and returns a record of the ratio actually reached. A surface
that is already finer than the radiation needs — a mesh patch snapped to a lamp, say — goes the
other way, through {func}`~aquaflux.radiation.coarsen_surfaces`, which keeps each body's emitted
power; see [Reactor geometry from CAD](cad_geometry.md) for both on a real reactor.

### The medium

The fluid between facets and receivers — water or air — is an
{class}`~aquaflux.radiation.Absorption`:

- {class}`~aquaflux.radiation.UniformAbsorption` — one coefficient everywhere, applied in closed
  form;
- {class}`~aquaflux.radiation.VoxelAbsorption` — a coefficient on a regular grid of cells,
  interpolated trilinearly and integrated exactly along each path. The grid need not match the
  flow mesh and usually should not: absorbance varies far more smoothly than velocity, and a
  path's cost grows with the number of grid cells it crosses, so a coarse grid over a fine mesh is
  both cheaper and no less accurate.

Leaving `absorption` out gives the field in a non-absorbing medium. Air absorbs little at the 254 nm
of a germicidal lamp, so an air-disinfection model often does that; pass an absorption where the
paths are long enough, or the air carries enough absorbing gas or aerosol, for it to matter. The
medium is a call argument, not a build argument, and its coefficient is differentiable — cell by
cell, for the voxel grid.

### What stands in the way

Two kinds of geometry block light, and they are handled separately.

**Bodies** are passed to the build as `occluders`: the exact solids of `aquaflux.solids` (a
{class}`~aquaflux.solids.Cylinder` sleeve, a {class}`~aquaflux.solids.Sphere`, their
{class}`~aquaflux.solids.Union` or {class}`~aquaflux.solids.Difference`, and
{class}`~aquaflux.solids.Outside` for a vessel or duct described by the fluid it holds), or a
{class}`~aquaflux.radiation.TriangleBody` for geometry that exists only as triangles. An exact solid
is both cheaper and more accurate than a triangulated one, which is why the CAD reader hands out
solids. A body's **shape** is frozen into the model; what it **lets through** is a call argument:

```python
from aquaflux.solids import Cylinder

# a sensor housing standing between the lamp and some of the fluid
housing = Cylinder(centre=[0.1, 0.0, 0.2], axis=[0.0, 0.0, 1.0], radius=0.01, half_length=0.05)
model = build_radiation_model(cell_centres, surfaces, occluders=[housing])

G, _ = fluence_rate(model, surfaces, absorption=medium, transmittance=[0.5])   # one per body
```

`transmittance` defaults to **opaque**, so a forgotten argument blocks rather than silently passes
light. A receiver or a facet centroid inside a body is a geometry error — the fluence rate there
would mean nothing — and the build refuses it. So a body can never contain an emitting surface,
which settles how a **multi-lamp array** is modelled: every lamp goes into the surface set, and the
lamps shadow one another through the surface set's own shadowing, described next. That shadowing is
opaque — a transmittance is a property of a body, not of a surface — so a partly transmitting sleeve
around a lamp that also emits cannot be represented; a body with a transmittance is for something
that stands in the light without emitting any. For a closed surface, such as a triangulated
lamp sleeve, {func}`~aquaflux.radiation.check_points_outside` checks that no receiver lies inside
it, by the exact winding number of the surface about each point.

**The surface set also shadows itself** — lamps and walls alike, whether they emit, reflect or
neither — wherever it is not convex: a bent duct, a reflector behind a lamp, the lamps of a
multi-lamp array, a louvre in front of a fixture. How that is tested is a
{class}`~aquaflux.radiation.SelfOcclusion` strategy, passed in the build's settings object,
{class}`~aquaflux.radiation.RadiationSettings` (its other fields are listed under
[Build once, solve many](#build-once-solve-many)):

| strategy | what it does | when to use it |
|---|---|---|
| {class}`~aquaflux.radiation.RayCastOcclusion` (the default) | one ray per pair, so a pair is either lit or shadowed | most scenes |
| {class}`~aquaflux.radiation.SilhouetteOcclusion` | clips each source against each blocker's silhouette for the exact covered fraction | where partial shadows matter and the facet count is modest; its cost rises steeply with facets |
| {class}`~aquaflux.radiation.NoOcclusion` | no self-shadowing | a single convex lamp, which cannot shadow itself |

⚠️ **Self-shadowing is switched off with `NoOcclusion()`, never with `None`.** An unset setting
means "use the default", and the default is the ray test.

Both shadow masks — between facets, and from facets to receivers — are built against the same
bodies in the same call, so a body that shadows a wall also shadows the cells behind it.
`receiver_occlusion` in the settings chooses a different strategy for the receivers alone, for
example the ray test for a mesh's cells beside the silhouette clip between facets.

## Reflection and the surface solve

A wall lit by a lamp re-emits, and what it re-emits lights every other wall. With `F_ij` the
fraction of what leaves facet `i` that lands on facet `j`, `M` the emission and `rho` the
reflectance, the irradiance `H` (all that lands on a facet) and the radiosity `B` (all it sends
out, emitted plus reflected) of every facet satisfy

```text
H = F^M M + F (B - M) + H_point + H_external     irradiance landing on each facet
B = M + rho * H                                  radiosity: what each facet sends out
```

where `F^M` carries each facet's emission with its own angular profile and `F` carries reflected
light, which leaves Lambertian (the two are the same matrix when every source is Lambertian), and
`H_point` is what the point sources land on each facet. The model solves the eliminated system
`(I - diag(rho) F) B = ...` with a matrix-free generalized minimal residual (GMRES) method, to a
global relative residual of `1e-10` (see [Theory](#the-interreflection-system) for the full
system). Pass your own `solver` to change that. Three entry points read off the solution:

```python
from aquaflux.radiation import radiosity, surface_irradiance

B, _ = radiosity(model, surfaces, absorption=medium)            # sent out by each facet, (n_facets,)
H, _ = surface_irradiance(model, surfaces, absorption=medium)   # landing on each facet,  (n_facets,)
G, _ = fluence_rate(model, surfaces, absorption=medium)         # in the volume,          (n_receivers,)
```

`surface_irradiance` is the dose rate on a wall — for fouling or a surface reaction — and returns
`NaN` at point sources, which have no surface for light to land on.

**Irradiance at any oriented point, with reflection**, such as an eye-level limit in an occupied
room, is the same two gathers {func}`~aquaflux.radiation.fluence_rate` makes, taken with
{func}`~aquaflux.radiation.direct_irradiance` instead: one of what the sources emit, one of what
the walls reflect, which leaves Lambertian. The shadows are built for the new points with the same
bodies as the model — and with its `self_occlusion`, passed to
{func}`~aquaflux.radiation.build_visibility`, if its settings chose one:

```python
from aquaflux.radiation import Lambertian, build_visibility, direct_irradiance

B, _ = radiosity(model, surfaces, absorption=medium)
shadows = build_visibility(occluders, surfaces, points)          # points: (n, 3)
reflected = surfaces.with_optics(
    emission=B - surfaces.emission, power=0.0, profiles=(Lambertian(),)
)
E = direct_irradiance(surfaces, points, normals, absorption=medium, visibility=shadows) \
    + direct_irradiance(reflected, points, normals, absorption=medium, visibility=shadows)
```

`normals` gives each point's facing direction, `(n, 3)`, as unit vectors: a vertical plane at eye
height facing the fixture, say. In a uniformly glowing box this gives `E = B` at any point and any
orientation, as it should.

### Mirror-like walls

A reflectance of 0.95 does not say whether a wall scatters in every direction or reflects like a
mirror, and the two are not close: simulating a reactor with a cylindrical reflector, Hassanpour et
al. (2023) find a 10–47% spread in log reduction between fully specular and fully diffuse walls at
the same reflectivity. So a surface set
carries the two separately, as `diffuse_reflectance` and `specular_reflectance`, which must each lie
in `[0, 1]` and sum to at most one on each facet.

A body reflects specularly only if it is named as specular when the model is built, because which
walls are mirrors is geometry and is frozen with the rest:

```python
model = build_radiation_model(
    cell_centres,
    surfaces,
    specular=["end plate"],
    occluders=[sleeve],
)
optics = surfaces.with_optics(
    diffuse_reflectance=surfaces.per_facet({"end plate": 0.1, "wall": 0.3}, default=0.0),
    specular_reflectance=surfaces.per_facet({"end plate": 0.6}, default=0.0),
)
G, cycles = fluence_rate(model, optics, transmittance=[0.9])
```

Each specular body is split into the flat planes its facets lie in, and each plane is a mirror: what a
point sees in it is the mirror image of every source, through the mirror's own outline. The light it
reflects reaches the volume and the other walls, and through them the diffuse interreflection, so the
solve and every field include it. Its specular reflectance is one value per body, supplied at each
call and differentiable like the diffuse one.

A reflected path is shadowed like a direct one, on both of its legs: from the source to the mirror,
and from the mirror on. The same occluders and the same self-occlusion setting apply, so a baffle or
a lamp sleeve stands in the way of reflected light exactly as it does of direct light. Each leg is
tested with one ray, through the source's centroid, whichever self-occlusion strategy is chosen: a
reflected path has no single view of its source to clip, so `SilhouetteOcclusion` refines only the
direct paths. A body crossed on both legs filters the light twice, by its transmittance squared, and
the transmittance stays live and differentiable on the reflected paths as on the direct ones.

What a specular body is limited to, and what the model refuses rather than answers wrongly:

- **One bounce.** Light reflecting off two mirrors in turn is not carried. For a single flat mirror,
  or a plane of symmetry, one bounce is all there is.
- **Flat bodies.** A body lying in more than 12 planes is refused (`max_mirror_planes` raises the
  limit): a curved one, such as a lamp sleeve, is one mirror per flat strip of facets, which is slow
  and describes a curved mirror poorly. Each plane costs several direct gathers of its own, so a
  body's planes are the cost to watch: twelve covers a box, or a wall with a few bends.
- **One ray per reflected path.** A source partly hidden on a reflected path counts as wholly hidden
  or wholly clear, as under the ray test, whichever strategy shadows the direct paths. A source
  that straddles a mirror's plane is not shadowed on its reflected paths at all.
- **A uniform medium.** A graded one (`VoxelAbsorption`) is refused with specular bodies.
- **One value per body.** A specular body whose facets carry different specular reflectances is
  refused, and so is a specular reflectance on a body not named as specular.
- **Several planes in one body** share the body's path lengths between each pair of facets as an
  average, weighted by how much each plane carries. That is exact for a body with one plane, so name
  each reflective wall as its own body where the medium absorbs strongly.

## Build once, solve many

{func}`~aquaflux.radiation.build_radiation_model` is the expensive step: it forms the
facet-to-facet transfer, which costs `n_facets^2`, and the shadow mask between facets and
receivers, which costs `n_facets * n_receivers`. It depends only on geometry. Everything a design
study sweeps is supplied per call:

| frozen at the build | supplied per call |
|---|---|
| surface geometry, receiver positions, which bodies are mirrors | emission, point-source power, diffuse and specular reflectance, profile parameters |
| body shapes (which pairs they block) | body transmittance |
| the self-shadowing strategy | the medium (`absorption`) |

So a sweep over lamp power, wall reflectance or the medium's absorbance pays the build once. The
surface set is passed again at each call to supply those optics, and **its geometry must be the one the
model was built from**: the model records a fingerprint of the vertices and refuses a set that
differs, since the field would otherwise be lit from one geometry through the shadows of another.
Change optics with `surfaces.with_optics(...)`; move a lamp with
{meth}`~aquaflux.radiation.Surfaces.with_geometry` and build a new model.

The build's other choices are gathered in {class}`~aquaflux.radiation.RadiationSettings`. Every
field is unset by default, which means the function it reaches uses its own default:

| setting | controls |
|---|---|
| `receiver_quadrature` | points per receiving facet in the transfer build (six by default) — the accuracy of energy conservation between facets |
| `self_occlusion`, `receiver_occlusion` | how the surface shadows itself, between facets and towards the receivers |
| `body_culling` | how the bodies' shadow test is organized — see below |
| `stream_receiver_mask` | whether the receiver shadow mask is held or rebuilt per chunk — see below |
| `transfer_chunk_size`, `gather_pair_limit` | peak memory of the transfer build and of the volume gather |

## Derivatives

Every optical input is differentiable, through the interreflection solve by its adjoint rather
than by replaying the iterations: emission, point-source power, both reflectances, profile parameters,
body transmittance, and the absorption coefficient or voxel field. So a sensitivity is one
reverse-mode pass:

```python
import jax

def mean_fluence_rate(reflectance, coefficient):
    walls = surfaces.per_facet({"wall": 1.0}, default=0.0)
    optics = surfaces.with_optics(diffuse_reflectance=walls * reflectance)
    G, _ = fluence_rate(model, optics, absorption=UniformAbsorption(coefficient))
    return G.mean()

d_reflectance, d_absorption = jax.grad(mean_fluence_rate, argnums=(0, 1))(0.3, medium.coefficient)
```

Derivatives taken this way agree with finite differences.

⚠️ **The derivative with respect to where anything stands in the way is exactly zero, by
construction.** Shadows are decided once, at the build, and frozen: a shadow edge moving with a
sleeve's radius is not seen. The receivers are frozen the same way, so a flow solve's derivative
with respect to the mesh's node positions receives no contribution from the fluence rate.

⚠️ **There is no derivative with respect to a lamp's position through the model.** The
facet-to-facet transfer is frozen along with the shadows, so a lamp moved under a gradient would
change the light it sends straight to each receiver and nothing else: the reflected light would
lose all dependence on where the lamp stands. On a lamp in a duct with walls of reflectance 0.3
such a gradient had the wrong sign. So {func}`~aquaflux.radiation.fluence_rate`,
{func}`~aquaflux.radiation.radiosity` and {func}`~aquaflux.radiation.surface_irradiance` refuse a
surface set whose vertices are traced. A position derivative is exact through
{func}`~aquaflux.radiation.direct_fluence_rate` and {func}`~aquaflux.radiation.direct_irradiance`,
which have no reflection to miss — with black walls, for instance — by moving the set with
{meth}`~aquaflux.radiation.Surfaces.with_geometry` inside the traced function:

```python
import jax.numpy as jnp
from aquaflux.radiation import direct_fluence_rate

def mean_direct_fluence_rate(lift):
    moved = lamp.with_geometry(jnp.asarray(lamp.vertices) + jnp.array([0.0, 0.0, lift]))
    return direct_fluence_rate(moved, cell_centres, absorption=medium).mean()

d_lift = jax.grad(mean_direct_fluence_rate)(0.0)
```

Shadows stay frozen there too. With the ray-cast self-shadowing, a pair is either lit or shadowed,
so a field that is shadowed anywhere is a staircase in position: even a finite difference taken
across rebuilt models jumps as shadow edges cross receivers.

## Running at the size of a reactor mesh

A reactor mesh has millions of cells and a finely resolved lamp several thousand facets, so the
receiver shadow mask — one entry per cell, facet and body — can run to gigabytes per body. Three
settings make that affordable:

- **`RadiationSettings(stream_receiver_mask=True)`** builds the receiver mask one chunk at a time
  at every call and drops it, so memory is one chunk's in the forward pass and under a gradient
  alike. It gives the same field and the same derivatives as holding the mask; the cost is a mask
  build per call. Leave it unset where the mask fits, so every later call reuses it.
- **Shaft culling**, the default for the bodies' shadow test
  ({class}`~aquaflux.radiation.ShaftCulling`), groups receivers and facets into compact blocks and
  clears a whole block pair without testing a segment whenever every body can prove it misses the
  region between them. The mask is identical, bit for bit, to testing every pair
  ({class}`~aquaflux.radiation.EveryPair`); only the work changes.
- **Black walls need no model.** Without reflection there is nothing to solve between facets, and
  {func}`~aquaflux.radiation.direct_fluence_rate` gathers the lamp alone, streaming its shadows
  chunk by chunk when given `occluders=` directly. Here `lamp` is the lamp's surface set on its
  own, and `vessel` the body the fluid is held in — an {class}`~aquaflux.solids.Outside` of the
  vessel's cylinders, for example, or the one the CAD reader returns:

  ```python
  from aquaflux.radiation import NoOcclusion, direct_fluence_rate

  G = direct_fluence_rate(
      lamp, cell_centres, absorption=medium, occluders=[vessel], self_occlusion=NoOcclusion()
  )
  ```

  {func}`~aquaflux.radiation.direct_irradiance` is its counterpart at oriented points, with the
  receiver cosine; it takes a built shadow mask
  ({func}`~aquaflux.radiation.build_visibility`) rather than the bodies.

[Reactor geometry from CAD](cad_geometry.md) walks through both paths on a 1.6-million-cell
reactor, with timings.

## Air disinfection

Nothing in the model is specific to water: an in-duct or upper-room air system is the same
calculation with a different medium and different geometry. What changes in practice:

- **The medium.** Air absorbs little at 254 nm, so an air model often leaves `absorption` out;
  pass one where the paths are long, or the air carries enough absorbing gas or aerosol, for it to
  matter. There is no transmittance convention for air, so give the coefficient directly, napierian
  and per metre.
- **The geometry.** A duct is a closed surface set of its walls, wound inward, or a body: an
  {class}`~aquaflux.solids.Outside` of {class}`~aquaflux.solids.Box` or
  {class}`~aquaflux.solids.Cylinder` pieces describes the air a duct holds. Duct walls are often
  bright — polished aluminium, for one, reflects much of the UV-C that reaches it — which can make
  the reflected light a large part of the answer rather than a correction.
- **Fixtures.** There is no fixture model. A louvred upper-room fixture can be represented by its
  geometry — the lamp as an emitting surface and the louvres as non-emitting, non-reflecting facets
  in the same surface set, which then shadow the lamp through the surface set's own shadowing — or
  by its opening, as a set of facets emitting with the fixture's measured distribution — read from
  its IES file into a {class}`~aquaflux.radiation.PhotometricProfile` — or, lacking one, with a
  {class}`~aquaflux.radiation.CosinePower` profile whose exponent is fitted to it. The first follows
  the geometry; the second needs the distribution, and a single-exponent fit is only as good as the
  fixture's symmetry about its axis.
- **Occupied-zone limits** are irradiances on a plane of a given orientation, at points that are
  not on any facet: see *Irradiance at any oriented point* under
  [Reflection and the surface solve](#reflection-and-the-surface-solve).
- **Units.** See [Units and conventions](#units-and-conventions) for µW/cm² and mJ/cm², and for the
  lamp's output at operating temperature.
- **Other lamps.** The model is monochromatic. A far-UVC krypton-chloride lamp is modelled at
  222 nm with the medium and the reflectances at that wavelength. A lamp emitting over a band —
  a medium-pressure lamp, say — needs one calculation per wavelength band, weighted afterwards: the
  field is linear in the emission, but the absorption and reflectances change with wavelength.

The validation below is of a water reactor; no air case has been compared against measurement or
another code here yet.

## Using the fluence rate in a transport equation

The model takes receiver positions rather than a mesh and returns a bare array in their order,
because it needs nothing else from a mesh: radiation knows about surfaces and points in space, and
nothing about cells, fluxes or residuals. A fluence rate at the cell centres becomes a reaction
source — photolysis, disinfection, an advanced-oxidation radical source — through the volume
sources of a {class}`~aquaflux.transport.ScalarTransport` equation, which is where a reaction
attaches to a scalar carried by the converged flow.

## What the model does not include

- **Refraction and reflection at a quartz sleeve.** In water, Bolton (2000) puts the error of
  neglecting them at a 6.5% reflection correction below 70% UVT, and up to 25% above it. For water
  the model is therefore best suited to lower transmittances — wastewater, or the 70% water of the
  Sozzi & Taghipour (2006) benchmark — and carries a systematic error of that size at
  drinking-water transmittances. A sleeved lamp in air is subject to the same neglect, though its
  size there is not quantified here; a bare lamp has no sleeve to refract through.
- **Specular reflection beyond one flat bounce.** Mirror-like walls are carried for one bounce off
  flat bodies (see [Mirror-like walls](#mirror-like-walls)).
- **Scattering by the medium, and more than one waveband.** The medium absorbs but does not
  scatter — neither particles in water nor aerosols or droplets in air — at one wavelength (see
  [Air disinfection](#air-disinfection) for lamps with more than one).
- **Exact energy balance for a non-diffuse source.** A cosine-power source's distribution is
  evaluated along one direction per pair of facets, centroid to centroid; energy is conserved
  exactly only for diffuse sources, and the error shrinks as the surface is refined.
- **Zero-thickness sheets block from both sides only when declared as such**: see
  {class}`~aquaflux.radiation.SilhouetteOcclusion` and {class}`~aquaflux.radiation.TriangleBody`.

## Theory

This section sets out exactly what is computed: the quantities, the discrete sums and the linear
system, which parts are exact and which are evaluated at one point per facet, and how the system
is solved and differentiated. The notation used throughout:

| symbol | meaning | units |
|---|---|---|
| $\mathbf{x}$, $\mathbf{n}$ | a receiver position, and a receiving surface's unit normal | m, – |
| $j$ | an emitting triangle (facet), with centroid $\mathbf{c}_j$, unit normal $\mathbf{n}_j$ from its winding, and area $A_j$ | – |
| $M_j$ | the exitance a facet emits (its `emission`) | W/m² |
| $P_k$ | the radiant power of point source $k$ | W |
| $\rho_j$ | a facet's diffuse reflectance | – |
| $B_j$, $H_j$ | a facet's radiosity (all it sends out) and irradiance (all that lands on it) | W/m² |
| $a(\mathbf{x})$ | the napierian absorption coefficient of the medium (water or air) | 1/m |
| $t_b$ | the transmittance of occluding body $b$ | – |

### The quantities

With $L(\mathbf{x}, \boldsymbol{\omega})$ the radiance arriving at $\mathbf{x}$ from direction
$\boldsymbol{\omega}$ (a unit vector pointing towards where the light comes from), the **fluence
rate** is its integral over the whole sphere, and the **irradiance** on a surface with normal
$\mathbf{n}$ is its cosine-weighted integral over the hemisphere that surface faces:

$$
G(\mathbf{x}) = \int_{4\pi} L(\mathbf{x}, \boldsymbol{\omega})\, d\omega,
\qquad
E(\mathbf{x}, \mathbf{n}) = \int_{\boldsymbol{\omega}\cdot\mathbf{n} > 0}
  L(\mathbf{x}, \boldsymbol{\omega})\, (\boldsymbol{\omega}\cdot\mathbf{n})\, d\omega .
$$

A uniformly lit enclosure shows the difference: its radiance $L$ is the same everywhere and in
every direction, so $G = 4\pi L$ while $E = \pi L$ on its walls.

### How a source emits

An angular **profile** $f(\hat{\mathbf{d}}, \mathbf{n})$ is a distribution of radiant intensity in
direction $\hat{\mathbf{d}}$, normalized to one over the sphere, for a source whose own normal is
$\mathbf{n}$. A source of power $P$ has intensity $P f$ in W/sr. Most profiles depend on the
emission angle alone, $f(\cos\theta)$ with $\cos\theta = \hat{\mathbf{d}}\cdot\mathbf{n}$
({class}`~aquaflux.radiation.AxisymmetricProfile`); a measured luminaire varies around its axis as
well, and its horizontal angle is measured from a direction fixed to the fixture. A point source has no normal to measure $\theta$ from, so it
is isotropic, with intensity $P/4\pi$. A facet of exitance $M$ radiates $MA$ in total, so its
radiance in direction $\hat{\mathbf{d}}$ is $M\,g$, where $g = f/\cos\theta$ is the **radiance per
unit exitance**:

| profile | $f(c)$ | $g(c) = f(c)/c$ | for |
|---|---|---|---|
| {class}`~aquaflux.radiation.Lambertian` | $\max(c, 0)/\pi$ | $1/\pi$ for $c > 0$, else $0$ | facets (and every reflection) |
| {class}`~aquaflux.radiation.CosinePower` | $(n+1)\max(c,0)^n / 2\pi$ | $(n+1)\,c^{\,n-1}/2\pi$ for $c > 0$, else $0$ | facets, with exponent $n \ge 1$ |
| {class}`~aquaflux.radiation.Isotropic` | $1/4\pi$ | not defined | point sources only |
| {class}`~aquaflux.radiation.PhotometricProfile` | $I(\gamma, h)/\Phi$ from the table | $f/\cos\gamma$ in front, else $0$ | facets, with a measured Type C table |

For the measured profile $\gamma$ is the angle from the facet's normal, $h$ the angle round it from
the fixture's chosen `up` direction, $I$ the table interpolated bilinearly in both, and $\Phi$ the
table's exact integral over the front hemisphere, so $f$ integrates to one.

$g$ is supplied by each profile already reduced, so the Lambertian constant $1/\pi$ involves no
$0/0$ at grazing incidence. A facet sends nothing behind itself: $g$ is zero there.

### The direct gather

The field from the sources is a sum over every facet and every point source:

$$
G_\text{direct}(\mathbf{x}) =
\sum_{j\ \text{areal}} M_j\, g_j(\hat{\mathbf{d}}_j)\; \Omega_j(\mathbf{x})\; T_j(\mathbf{x})\; V_j(\mathbf{x})
\;+\;
\sum_{k\ \text{point}} \frac{P_k}{4\pi r_k^2}\; T_k(\mathbf{x})\; V_k(\mathbf{x})
$$

where

- $\Omega_j(\mathbf{x})$ is the **exact solid angle** the triangle subtends at $\mathbf{x}$ (below);
- $\hat{\mathbf{d}}_j = (\mathbf{x} - \mathbf{c}_j) / |\mathbf{x} - \mathbf{c}_j|$ is the emission
  direction from the facet's centroid, and $\cos\theta_j = \mathbf{n}_j \cdot \hat{\mathbf{d}}_j$ the
  emission angle, and $r_k = |\mathbf{x} - \mathbf{c}_k|$ the distance to a point source;
- $T_j(\mathbf{x}) = \exp\!\left(-\int a\, ds\right)$ is the **transmittance of the medium** along the
  straight segment from $\mathbf{c}_j$ to $\mathbf{x}$;
- $V_j(\mathbf{x}) \in [0, 1]$ is the fraction of the source's view that **gets past the geometry**
  in the way.

The irradiance on a surface point ({func}`~aquaflux.radiation.direct_irradiance`) is the same sum
with the plain solid angle replaced by the **projected** one,
$\Omega^\perp_j(\mathbf{x}, \mathbf{n})$, and each point source's term multiplied by the receiving
cosine $\max(-\mathbf{n}\cdot\hat{\mathbf{r}}_k, 0)$.

**What is exact and what is one-point.** The solid angle is the exact integral over the triangle,
at any distance — including the near field, where the point approximation
$A_j\cos\theta_j / r^2$ is wrong without bound. The other three factors — $g_j$, $T_j$ and $V_j$ —
are evaluated once per facet, along the segment from its centroid. For a Lambertian facet $g_j$ is
constant, so only $T_j$ and $V_j$ carry that approximation; its error shrinks as the facet shrinks
relative to its distance from the receiver, which is what
{func}`~aquaflux.radiation.refine_for_receivers` controls.

### The two solid-angle kernels

For unit vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}$ from the receiver to the three vertices,
the plain solid angle is the van Oosterom & Strackee (1983) form of the spherical excess,

$$
\Omega = 2\,\left|\operatorname{atan2}\!\bigl(\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c}),\;
  1 + \mathbf{a}\cdot\mathbf{b} + \mathbf{a}\cdot\mathbf{c} + \mathbf{b}\cdot\mathbf{c}\bigr)\right| ,
$$

which stays accurate for the small, distant triangles a refined surface is made of, where
summing three interior angles loses everything to cancellation.

The projected solid angle, $\Omega^\perp = \int \cos\theta\, d\omega$ with $\theta$ measured from
the receiving normal $\mathbf{n}$, is Lambert's contour integral over the edges of the triangle's
image on the unit sphere,

$$
\Omega^\perp = \tfrac{1}{2}\left| \sum_{k} \gamma_k\, (\mathbf{u}_k\cdot\mathbf{n}) \right| ,
\qquad
\gamma_k = \operatorname{atan2}\!\bigl(|\mathbf{d}_k\times\mathbf{d}_{k+1}|,\ \mathbf{d}_k\cdot\mathbf{d}_{k+1}\bigr),
\quad
\mathbf{u}_k = \frac{\mathbf{d}_k\times\mathbf{d}_{k+1}}{|\mathbf{d}_k\times\mathbf{d}_{k+1}|} ,
$$

with $\mathbf{d}_k$ the unit directions to the vertices in order. The triangle is first **clipped
to the half-space in front of the receiving surface** (a triangle has at most four vertices after
one cut), because the contour sum is signed and a triangle straddling the plane would otherwise
partly cancel itself. $\Omega^\perp/\pi$ is the fraction of a Lambertian receiver's hemisphere
the triangle covers, so over a closed enclosure these fractions sum to exactly one.

### What gets past the geometry

The surviving fraction multiplies one factor per body by one for the surface set itself:

$$
V_j(\mathbf{x}) = \prod_b \bigl[\,1 - \beta_{bj}(\mathbf{x})\,(1 - t_b)\,\bigr] \cdot \bigl(1 - h_j(\mathbf{x})\bigr) .
$$

- $\beta_{bj}(\mathbf{x}) \in \{0, 1\}$ says whether the segment from $\mathbf{c}_j$ to $\mathbf{x}$
  crosses body $b$. It is decided once, when the model is built, by the body's exact geometry.
- $t_b$ is supplied at each call, with opaque ($t_b = 0$) as the default.
- $h_j(\mathbf{x})$ is the share of the source hidden by the surface set's own triangles — lamps
  and walls alike — which are opaque. {class}`~aquaflux.radiation.RayCastOcclusion` casts the one segment and records $0$ or
  $1$. {class}`~aquaflux.radiation.SilhouetteOcclusion` clips the source's image on the sphere
  against each blocking triangle's silhouette and records the covered fraction of $\Omega_j$ for a
  point in the fluid, or of $\Omega^\perp_j$ for a receiver on a facet. The covered shares of
  separate blockers are added and capped at one, so two blockers covering the same part of a
  source are counted twice. {class}`~aquaflux.radiation.NoOcclusion` sets $h_j = 0$.

Only $t_b$ is live. $\beta$ and $h$ are geometry, fixed when the model is built.

### The medium's absorption

Along a segment of length $r$, the transmittance is $T = \exp(-\tau)$, with optical depth
$\tau = \int_0^r a\, ds$:

- {class}`~aquaflux.radiation.UniformAbsorption`: $\tau = a\,r$, in closed form.
- {class}`~aquaflux.radiation.VoxelAbsorption`: $a$ is sampled at the centres of a regular grid and
  interpolated trilinearly. The segment is cut at every plane through the sample points it
  crosses, and each piece is integrated by Simpson's rule. On a straight line a trilinear field is
  a cubic in the distance along it, and Simpson's rule integrates a cubic exactly, so $\tau$ is the
  exact integral of the interpolated field — no step size is involved.

### Transfer between facets

The **form factor** $F_{ij}$ is the fraction of what leaves facet $i$ diffusely that lands on
facet $j$. The same integral is also, with no appeal to reciprocity, the weight with which facet
$j$'s radiosity contributes to the irradiance on facet $i$ — which is how the solve uses it, and
how it is computed: facet $i$ is the **receiving** facet and facet $j$ the **sending** one. It is a
double area integral. The sending triangle $j$ is integrated exactly by the projected solid angle,
and the receiving triangle $i$ by a symmetric Dunavant (1985) quadrature rule of $Q$ points
$\mathbf{p}_{iq}$ with weights $w_q$ summing to one (six points by default,
`receiver_quadrature`):

$$
F^\text{geo}_{ij} = \frac{1}{\pi} \sum_{q=1}^{Q} w_q\; \Omega^\perp_j(\mathbf{p}_{iq}, \mathbf{n}_i),
\qquad F^\text{geo}_{ii} = 0 .
$$

Because the sending side is exact, in a **closed** enclosure each quadrature point sees
surroundings whose fractions sum to one, so every row of $F^\text{geo}$ sums to one exactly at any
quadrature rule. Where the surface set is open — a duct with its ends left off — the light leaving
through the openings is lost and a row sums to less than one. Either way no row exceeds one, which
is the property that bounds the system below. Reciprocity, $A_i F_{ij} = A_j F_{ji}$, holds only to the
accuracy of the receiving quadrature, and
{func}`~aquaflux.radiation.reciprocity_residual` reports how closely. The diagonal is zero because
a flat triangle sees none of itself. Point sources have no area, so they take no part in $F$.

At each call the frozen geometry is multiplied, entry by entry, by the live factors, evaluated
along the segment between the two centroids:

$$
F_{ij} = F^\text{geo}_{ij}\; T_{ij}\; V_{ij},
\qquad
F^M_{ij} = F_{ij}\;\pi\, g_j(\cos\theta_{ji}) ,
$$

where $F$ carries **reflected** light, which leaves Lambertian, and $F^M$ carries each facet's own
**emission** with its own profile ($\cos\theta_{ji}$ is the emission angle at facet $j$ towards
facet $i$). For a Lambertian source $\pi g_j = 1$, and the two are the same matrix. The transfer
evaluates one emission angle per pair, so it takes only profiles of that angle alone
({class}`~aquaflux.radiation.AxisymmetricProfile`); a measured luminaire enters through the
`external_irradiance` its direct gather gives each facet.

### The interreflection system

Every facet's irradiance is what the others emit and reflect onto it, plus what the point sources
and any external light land on it. Its radiosity is its own emission plus the reflected part of
that irradiance:

$$
H = F^M M + F\,(B - M) + H_\text{point} + H_\text{ext},
\qquad
B = M + \operatorname{diag}(\rho)\, H .
$$

$H_\text{point}$ is the direct irradiance of the point sources at each facet centroid, with that
facet's normal as the receiving normal. $H_\text{ext}$ is the optional `external_irradiance`.
Eliminating $H$ leaves one linear system for the radiosity:

$$
\bigl(I - \operatorname{diag}(\rho)\, F\bigr)\, B
= M + \operatorname{diag}(\rho)\bigl[(F^M - F)\,M + H_\text{point} + H_\text{ext}\bigr] .
$$

Every entry of $F$ is non-negative, and each row sums to at most one, since $T$ and $V$ are at
most one. So the spectral radius of $\operatorname{diag}(\rho) F$ is at most $\max_j \rho_j$, and
for any reflectance below one the solution is the convergent Neumann series
$B = \sum_{m \ge 0} (\operatorname{diag}(\rho) F)^m s$, with $s$ the right-hand side above — every
bounce, not a truncation of them.

**How it is solved.** The system $A B = s$ is solved matrix-free by restarted generalized minimal
residual (GMRES) iterations, 120 per restart. Each iteration costs one product
$B \mapsto B - \rho \odot (F B)$. The solve stops when $\|s - A B\|_2 \le 10^{-10}\,\|s\|_2$. That test is taken over the
whole vector because most facets do not emit, so most entries of the right-hand side are zero, and
a test entry by entry would demand an absolute tolerance there and stall. The system is not
symmetrized for a conjugate-gradient method, because that needs a scaling by $1/(\rho_j A_j)$ and
most facets have $\rho_j = 0$. A solve that does not converge raises; the restart-cycle count is
returned beside each field. `surface_irradiance` returns $H$ from the first equation, evaluated at
the solved $B$.

### The fluence rate with reflection

Once $B$ is known, the part of each facet's radiosity beyond its own emission, $B_j - M_j$, is
reflected light and leaves Lambertian. The volume field is therefore two direct gathers through the
same solid angles, transmittances and shadows:

$$
G(\mathbf{x}) = G_\text{direct}\bigl[M, P, f\bigr](\mathbf{x})
  + G_\text{direct}\bigl[B - M,\ 0,\ \text{Lambertian}\bigr](\mathbf{x}) ,
$$

the first with each source's own profile and the point sources, the second with the reflected
exitance, Lambertian radiance $1/\pi$ and no point sources. Because the two differ only in the
radiance weight, they are evaluated in one pass. The enclosure check in the quick start is
this formula at uniform $B$: every facet then has radiance $B/\pi$ in every direction, and the
solid angles of a closed surface sum to $4\pi$, so $G = 4B$ at every point.

### Differentiating the solve

Everything live enters only through elementwise products with the frozen arrays and through the
linear solve, so reverse-mode differentiation reaches $M$, $P$, $\rho$, the profile parameters,
$t_b$ and $a$ exactly. The solve is differentiated implicitly rather than by replaying its
iterations: for an output $J(B)$, the adjoint is one solve with the transposed operator,

$$
\bigl(I - \operatorname{diag}(\rho)\,F\bigr)^{\!\top} \lambda = \frac{\partial J}{\partial B} ,
$$

after which the gradient with respect to each live input follows from $\lambda$ by the chain rule
through the right-hand side and through $F$. Its cost does not depend on how many iterations the
forward solve took. The frozen quantities — $\Omega$, $\Omega^\perp$, $F^\text{geo}$, $\beta$, $h$,
the separations and the emission cosines — are built once from concrete geometry and carry no
derivative. That is why the derivative with respect to where an occluder stands is exactly zero.

## Verification

**Closed forms.** The implementation is tested against results with no discretization error to
hide behind:

- a uniformly emitting, reflecting enclosure gives `G = 4M / (1 - rho)` everywhere inside it, to
  about `1e-12`, at reflectances up to 0.9;
- an enclosure whose facets carry a cosine-power profile but emit nothing, lit only by an
  `external_irradiance` on its facets, still gives `G = 4B`: every watt in it has been reflected
  once, so it leaves diffusely whatever the facets' own profile says. This is the case that fails
  if reflected light is re-emitted with the source's distribution rather than diffusely;
- a point source in a black enclosure gives `P / (4 pi r^2)` exactly;
- the rows of the facet-to-facet transfer of a closed enclosure sum to one to about `1e-15`.

Two diagnostics report the same things for your own scene:
`row_sum_error(model.transfer)` ({func}`~aquaflux.radiation.row_sum_error`) is the largest departure
of a row from one — near zero for a closed scene, and the share escaping through its openings for
an open one — and `reciprocity_residual(model.transfer, surfaces.area)`
({func}`~aquaflux.radiation.reciprocity_residual`) is how well `A_i F_ij = A_j F_ji` holds, which
the `receiver_quadrature` sets.

**Against discrete ordinates on a real reactor.** On the Sozzi & Taghipour (2006) annular reactor
(1,635,909 cells, black walls, 70% UVT), the fluence rate was compared with a discrete-ordinates
radiation solve on the same mesh (`validation/sozzi_radiation/`):

- the reactor's volume-mean fluence rate agrees to 0.09%;
- the discrete-ordinates field **converges on this one** as its directions are refined: the ratio
  over the central 80% of lit cells narrows from 0.36–1.65 at 64 directions to 0.886–1.076 at 256.
  The remaining spread is the discrete-ordinates ray effect, and it is largest down the inlet and
  outlet pipes, which only a narrow cone of directions reaches;
- the gather's own discretization error near the lamp, for this lamp triangulated at 4 mm and
  measured against the same lamp refined nine times, is a median 0.44% over the cells near it.
  (The 0.16% on the [CAD page](cad_geometry.md) is a different comparison: the drawing's own lamp,
  triangulated at 2.5 mm, measured within 5 mm of it against a 270,336-facet reference.)
- carried through to particle dose with a Lagrangian tracker on the reactor's own flow — the same
  particles, with only the fluence rate changed — the mean dose agrees with the 256-direction solve
  to 0.3%, and the log reduction to within 1.1% at every inactivation rate tested. At 64 directions the low-dose tail — the particles that decide
  disinfection — is badly under-resolved, and the log reduction of a sensitive organism falls
  short by 23%.

## The building blocks

Everything the model composes is public and usable on its own — see the
[API reference](api.md) for the full list. Two pieces deserve a note:

- **The solid-angle kernels.** {func}`~aquaflux.radiation.solid_angle` is the plain solid angle of
  a triangle at a point, which a fluence rate needs; {func}`~aquaflux.radiation.projected_solid_angle`
  is the cosine-weighted one, which an irradiance and every surface-to-surface transfer need. ⚠️
  **They are not interchangeable**, and no constant converts one into the other: the plain one used
  in a transfer over a closed enclosure makes every row sum to exactly two, a clean-looking result
  that is wrong at every refinement. {func}`~aquaflux.radiation.signed_solid_angle` keeps the sign
  of the winding and is meaningful only summed over a closed surface, where it gives the winding
  number ({func}`~aquaflux.radiation.enclosure_winding`).
- **The transfer and visibility builds.** {func}`~aquaflux.radiation.build_transfer` and
  {func}`~aquaflux.radiation.build_visibility` are what the model calls; use them directly to
  inspect a transfer matrix or a shadow mask.
