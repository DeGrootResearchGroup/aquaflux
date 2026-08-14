# Cell zones and face patches

Solvers rarely treat a whole domain uniformly. Different regions of cells carry
different materials or equations; different sets of boundary faces carry different
boundary conditions. `aquaflux` expresses this with two named groupings on the mesh:

- **Cell zones** ({class}`~aquaflux.mesh.CellZones`) — each cell belongs to one named
  zone (`"fluid"`, `"solid"`, a porous region, …).
- **Face patches** ({class}`~aquaflux.mesh.FacePatches`) — each face belongs to one named
  patch. Boundary patches (`"inlet"`, `"wall"`, …) are the usual case, but a patch can
  also name a set of interior faces (see [Interfaces and baffles](#interfaces-and-baffles)).

Both are **partitions**: every cell is in exactly one zone, every face in exactly one
patch. Each grouping is a single integer label array (one label per element), so
restricting an operator to a region is a cheap boolean selection. The shared behaviour
lives on a common base, {class}`~aquaflux.mesh.LabelledGroups`.

## Working with groups

Given a mesh, its zones and patches are `mesh.cell_zones` and `mesh.face_patches`. Both
offer the same selectors, keyed by name:

```python
zones = mesh.cell_zones

zones.mask("fluid")      # (n_cells,) bool  — True for cells in the "fluid" zone
zones.indices("fluid")   # (n_in_zone,) int — the fluid cell indices
zones.size("fluid")      # number of cells in the zone
zones.n_groups           # number of zones
```

A boolean mask is the usual tool: a per-zone property or a zone-restricted source term is
applied by selecting with `zones.mask(name)`. Face patches work identically over faces:

```python
patches = mesh.face_patches
wall_faces = patches.mask("wall")     # (n_faces,) bool
```

## Defaults and construction

If you build a mesh without specifying groupings, it gets working defaults: a single
`"default"` zone containing every cell, and a `"boundary"` / `"interior"` patch split.
The reserved names `"interior"` and `"boundary"` are assigned automatically and cannot be
reused for a custom patch.

To name regions, pass `cell_zones` and/or `face_patches` to
{meth}`Mesh.from_faces <aquaflux.mesh.Mesh.from_faces>` (or
{meth}`~aquaflux.mesh.Mesh.from_csr`) as a mapping from a name to the indices in that
group:

```python
mesh = Mesh.from_faces(
    coords, face_nodes, owner, neighbour, n_cells=n,
    cell_zones={"solid": solid_cell_indices},           # everything else -> "default"
    face_patches={"inlet": inlet_faces, "wall": wall_faces},
)
```

Any element you do not list is placed in the default group, so the partition always
covers every element exactly once. Overlapping groups — assigning an element to two names
— are rejected at construction with a clear error.

The structured-grid generators offer a shortcut for the common case: with
`named_boundaries=True`, {func}`~aquaflux.mesh.structured_grid_2d` and
{func}`~aquaflux.mesh.structured_grid_3d` name the domain sides as patches (`"left"`,
`"right"`, `"bottom"`, `"top"`, and `"back"`/`"front"` in 3D) automatically.

## Interfaces and baffles

Faces between two different cell zones — a fluid–solid contact, say — are often where the
interesting coupling happens. `aquaflux` handles these two ways, and they coexist.

**Derived interfaces.** A cross-zone interface can be computed straight from the zone
labelling, with no hand-maintained face list. {meth}`CellZones.interface_mask
<aquaflux.mesh.CellZones.interface_mask>` returns the interior faces whose two cells lie
in different zones, and {meth}`~aquaflux.mesh.CellZones.interface_mask_between` narrows to
a specific pair of zones:

```python
zones = mesh.cell_zones
all_interfaces  = zones.interface_mask(mesh.face_cells)
fluid_solid     = zones.interface_mask_between(mesh.face_cells, "fluid", "solid")
```

These take `mesh.face_cells` (the face → cell connectivity) rather than the whole mesh —
the smallest input they need.

**Named interior patches (baffles).** When a set of faces needs bespoke treatment that
the zones do not imply — a **baffle** (an interior face acting as a thin wall), or an
interface with a special model — promote it to a named face patch. Because patches
partition *all* faces, interior ones included, an interior baffle is simply a named patch
of interior faces, and the flux/boundary layer gives it its special treatment.
{meth}`FacePatches.is_boundary_patch <aquaflux.mesh.FacePatches.is_boundary_patch>`
distinguishes a true boundary patch from such an interior one.

Derivation is the convenience; naming is the escape hatch — use a derived interface when
the zones already say everything, and a named patch when a specific set of faces needs a
specific model.

## What this enables

- **Boundary conditions** — iterate the face patches and assign a boundary-condition
  strategy per patch, applied through the patch mask. Baffles are patches too.
- **Materials and multi-region models** — look up a per-zone property through
  `cell_zones.mask(zone)`, and couple regions across `interface_mask_between(...)`.
- **Zone- or patch-restricted operators** — restrict a scheme or a source term to a
  zone's cells or a patch's faces by mask.
