# Radiation case files

A case file can describe the light of ultraviolet lamps instead of a flow: lamps on some of the
mesh's boundary patches, walls that are black or reflect diffusely, bodies that stand in the way,
the medium in between, and where the light is wanted. `aquaflux check case.yaml` checks it against
its mesh, and `aquaflux run case.yaml` computes the fluence rate in the cells and the irradiance on
the walls and writes them as files any VTK viewer opens. The general rules for case files — how
values are named, how YAML is read, what is checked when — are on {doc}`case_files`; this page
covers what is particular to radiation. How the light is computed is on {doc}`radiation`.

## An example

A 4 x 4 x 3 m room of air lit by a measured far-ultraviolet lamp on its ceiling, with a body
hanging in it and walls that reflect half the light that reaches them:

```yaml
mesh: {kind: OpenFOAMMesh, path: room/constant/polyMesh}

physics:
  kind: Radiation
  occluders:
    - {kind: PatchBody, patches: [bunny], sheet: true}
  lamp_samples: 4
  settings:
    kind: RadiationSettings
    self_occlusion: {kind: NoOcclusion}

boundaries:
  lamp:
    kind: Lamp
    profile: {kind: IesProfile, file: ushio_b1.ies, up: [1.0, 0.0, 0.0]}
  floor:
    kind: Wall
    reflectance: 0.5
    geometry: {kind: MeshPatch, coarsen: {kind: Coarsen, max_edge: 0.2, chord: 1.0e-3}}
  ceiling:
    kind: Wall
    reflectance: 0.5
    geometry: {kind: MeshPatch, coarsen: {kind: Coarsen, max_edge: 0.2, chord: 1.0e-3}}
  walls:
    kind: Wall
    reflectance: 0.5
    geometry: {kind: MeshPatch, coarsen: {kind: Coarsen, max_edge: 0.2, chord: 1.0e-3}}
  bunny: {kind: Wall}

outputs:
  fields: [{kind: Vtk}, {kind: PatchVtk}]
```

A radiation case has **no `fluid` and no `numerics` section**: there is no flow, and the medium and
the radiation's own numerics live inside the physics. Either section, or a `drive`, `sources` or
`pressure_datum`, is refused.

A small case needs no mesh file: a `StructuredGrid` with three cell counts and three lengths is a
uniform box whose six patches are named `left`, `right` (x), `bottom`, `top` (y), `back` and `front`
(z).

## The patches

Each boundary patch is a {class}`~aquaflux.case.Lamp` or a {class}`~aquaflux.case.Wall`.

**A lamp** emits its `power` (in W) from the patch, spread evenly over its triangulated area, with
the angular distribution its `profile` gives:

- {class}`~aquaflux.case.LambertianProfile` — the same radiance in every direction it faces;
- {class}`~aquaflux.case.CosinePowerProfile` — intensity proportional to `cos(theta)^exponent`;
- {class}`~aquaflux.case.IesProfile` — a measured luminaire's IES LM-63 table, with `up` the
  direction of the table's horizontal angle zero in the case's frame.

The lamp's `power` may be left out only for a photometry file that states its intensities in a
radiant unit, through an `[_INTENSITYUNITS]` keyword of `W/sr`, `mW/sr` or `uW/sr`; its own
integrated flux is then the power. A file in candela carries no radiant power and the lamp must
state one.

**A wall** is black unless it has a `reflectance`, in which case it reflects that fraction of the
light arriving on it, diffusely. **A lamp** is the same to the light arriving on it: black unless it
has a `reflectance`, and then reflecting diffusely what lands on it from the other lamps and the
walls. Either way its triangles stand in the way of light the walls reflect past it, so a wall's
light does not reach the water behind a lamp. Interreflection is closed to convergence: the number of
bounces is not a setting.

A lamp is a stationary wall to a flow, but a flow case refuses one, and refuses a reflectance: nothing
in it would read them. A radiation case refuses an `Inlet` or an `Outlet`, and a wall's `velocity` or
`k`.

## Where a surface comes from

A lamp's and a reflecting wall's triangles are, in order of preference:

- {class}`~aquaflux.case.CadSurface` — a solid of the STEP drawing the mesh was made from, with
  every vertex on its true surface, sized by `chord` and `facet_size` and placed by an optional
  `placement` (needs the optional CAD kernel, `pip install aquaflux[cad]`). The solid's whole surface
  is used, so it suits a lamp sleeve or a body standing in the medium — not one wall of a vessel drawn
  as the fluid it holds;
- {class}`~aquaflux.case.StlSurface` — an STL file's triangles, all of them or the named `solids`,
  in metres (an STL file carries no units and is not scaled);
- {class}`~aquaflux.case.MeshPatch` — the mesh's own patch, each face cut into the fan of triangles
  about its centre. This is what is used when `geometry` is left out.

A reflecting surface needs far fewer triangles than a mesh has faces — the exchange between
reflecting triangles is a dense matrix over them — so a reflecting `MeshPatch` (or `StlSurface`) is
normally coarsened: {class}`~aquaflux.case.Coarsen` keeps every vertex on the original surface and
bounds the longest edge (`max_edge`) and how far an original vertex may lie from the coarse
triangle that replaces it (`chord`).

**A surface read from a file is turned to face into the domain**: each triangle is compared with the
nearest face of the patch it stands for, and the whole surface is reversed if it faces out. A
surface whose triangles disagree about their side (one wound the wrong way would emit or reflect
nothing) is refused, as is one that is never square enough to its patch to say. A surface read from a
file is one patch's own, so it cannot be given under the name of a patch group.

## What stands in the way

`occluders` lists the bodies that shadow the light, from lamps and reflecting walls alike:

- {class}`~aquaflux.case.CadSolid` — a solid of a STEP drawing, recognized as an exact body;
- {class}`~aquaflux.case.CadFluid` — the medium held by several solids of a drawing (a vessel and
  its pipes), everything outside them shadowing;
- {class}`~aquaflux.case.StlBody` — an STL file's triangles;
- {class}`~aquaflux.case.PatchBody` — mesh patches. `sheet: true` treats them as a sheet with the
  medium on both sides.

```{note}
The walls of the domain shadow nothing unless they are listed here. In a convex room that is exact;
a domain with a wall that can stand between a lamp and a point — an L-shaped room, a baffle drawn as
part of the boundary — must name it as an occluder.
```

The lamps need not be listed: their own triangles shadow the light of the other lamps and the light the
walls reflect, whenever the surfaces shadow themselves (that is, unless `self_occlusion` is
`NoOcclusion`). Do not list a solid drawn round a lamp as well: the lamp's triangles lie on its
surface, which a solid can count as inside it, and a surface inside a body is refused as a geometry
error.

## The medium, and where the light is gathered

`medium` is a {class}`~aquaflux.case.UniformMedium`: its napierian `absorption` coefficient per
metre, or its ultraviolet `transmittance` in percent over one centimetre. Left out, the medium
absorbs nothing.

`receivers` ({class}`~aquaflux.case.Receivers`) says where the light is gathered: the fluence rate
at every cell centre (`cells`, true by default) and the irradiance at the centres of the faces of the
named `patches` — by default every wall that is not part of a `PatchBody`. A lamp's own faces are
never gathered on.

## Numerics, and the solve

Inside the physics:

- `lamp_refinement` — split each lamp triangle until its longest edge is at most this fraction of
  its distance to the nearest point gathered at. Each lamp triangle casts its own hard-edged shadow,
  so a soft shadow is resolved by many small ones;
- `lamp_samples` — the lamps' light on each reflecting triangle is averaged over `lamp_samples**2`
  points of it, so a shadow falling partly across it is not decided at one point (4 by default);
- `settings` — the library's {class}`~aquaflux.radiation.RadiationSettings`: how the surfaces shadow
  themselves (`self_occlusion`, the ray test by default; `{kind: NoOcclusion}` for a flat lamp in a
  convex room, where the test can find nothing), how bodies' shadows are worked out, and how many
  pairs a pass may form.

The solve is {class}`~aquaflux.case.RadiationSolve`, whose one setting is the relative residual
`rtol` the reflection solve stops at. A radiation case with no `solver` section uses it.

## What a run writes

Into the output directory, beside `case.yaml` and `run.yaml`:

- **`fields.vtu`** (`Vtk`) — the cell fields: `G`, the fluence rate in W/m², and, when anything
  reflects, its parts `G_direct` and `G_reflected`;
- **`patches.vtm`** (`PatchVtk`) — a VTK multiblock index with one block per boundary patch, named by
  the patch, each block a polygonal-data file `patches/<patch>.vtp` holding the patch's faces. The
  face fields are `E`, the irradiance in W/m²; `E_absorbed`, what the wall keeps of it,
  `(1 - reflectance) E`; and, when anything reflects, `E_direct` and `E_reflected`. Every boundary
  patch is written, so the lamps and bodies can be drawn too; the fields appear on the patches they
  were gathered on.

`run.yaml` records, under `results`, where the lamps' power goes: `lamp_power`, the reflection
solve's `radiosity_cycles`, `volume_integral_G`, the power the medium absorbs
(`medium_absorbed_power`), the power the lamps take back when anything reflects
(`lamp_absorbed_power`), each gathered patch's `area`, `incident_power` and `absorbed_power`, and
`unaccounted_power` — the lamps' power less all of those, which is what the occluders and any patch
not gathered on absorb, and the lamps too when nothing reflects.
