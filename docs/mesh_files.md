# Making a mesh

A case reads its mesh from a file ({doc}`case_files`); the mesh itself is made once, from a **mesh
file**, by

```bash
aquaflux mesh mesh.yaml              # mesh the region, write the mesh where the file says
aquaflux mesh mesh.yaml --overwrite  # replace a mesh made before
```

The mesh is written as aquaflux's own `.aqmesh` file (see [the mesh](mesh.md)), and as an OpenFOAM
polyMesh as well if the file asks; a case reads it with {class}`~aquaflux.case.AquafluxMesh`. The
`.aqmesh` file records how the mesh was made: the mesh file's settings as they ran, the aquaflux
version and commit, when, how long it took, and a summary of its quality
({class}`~aquaflux.mesh.QualitySummary`: the cells, the interior faces' non-orthogonality at the
99th percentile and at most, the neighbour volume ratio likewise, the smallest cell and the concave
cells), so a mesh can be judged without being read. A mesh already where this one would go is
refused before anything is meshed, unless `--overwrite` is given.

`aquaflux mesh` prints each stage of the meshing as it starts, after the seconds since it began --
reading the drawing, triangulating the surface, growing the layers, filling with tetrahedra,
building the polyhedral dual and so on, only those the file asks for -- and then the mesh made and
its quality. The browser interface makes a mesh the same way and shows the same lines
({doc}`user_interface`), and is where a mesh file's patches can be chosen by clicking the drawing's
faces.

A mesh file's top level is its sections; `drawing`, `region`, `sizes` and `output` have one form each
and name no `kind`. Relative paths are taken from the directory the mesh file sits in:

```yaml
drawing: {file: reactor.step}                 # and a placement, if it is drawn in another frame
region: {solids: [reactor_body, inlet_pipe, outlet_pipe], without: [lamp]}
sizes: {max_size: 0.02, feature_angle: 0.785}
patches:
  inlet: {kind: FacesNear, points: [[0.04765, 0.0, 0.8945]]}
output: {path: reactor.aqmesh, openfoam: of_case}   # the polyMesh is optional
```

## What is meshed

A mesh file ({class}`~aquaflux.case.MeshSpec`) meshes a region of a STEP drawing as polyhedra: the union of
some `solids` less those `without`. Netgen fills the region with tetrahedra no larger than
about `max_size`, and the mesh is their polyhedral dual — one cell per tetrahedron point, so
the cells are larger than `max_size` — with every edge of the boundary that folds by more than
`feature_angle` (in radians) kept sharp. Each of its `patches` is named for the faces of the
region a selection chooses: {class}`~aquaflux.meshing.FacesNear`, the face nearest each of some
points (a pipe's end), or {class}`~aquaflux.meshing.FacesOn`, the faces lying on some solids'
surface (a lamp taken out of the water). The faces no selection chooses make the patch
`others`, `walls` unless named. Meshing needs `pip install aquaflux[cad,meshing]`.

```yaml
drawing: {file: reactor.step}
region: {solids: [reactor_body, inlet_pipe, outlet_pipe], without: [lamp]}
sizes: {max_size: 0.02, feature_angle: 0.785}
patches:
    inlet: {kind: FacesNear, points: [[0.04765, 0.0, 0.8945]]}
    outlet: {kind: FacesNear, points: [[1.739, 0.0, 0.0]]}
    lamp: {kind: FacesOn, solids: [lamp]}
```

Every vertex of the mesh's boundary lies on the drawing: on its faces, and on its sharp edges along
them. Between those vertices a boundary face is flat, so a curved wall's patch is a little short of
the drawing's area.

Wall-resolved layers grow from the patches listed in `layered`, sized for the solver's cells: the
cells against the wall are `first_height` tall and each layer above is `growth` times the one below
({class}`~aquaflux.meshing.PrismLayers`; `growth` must exceed 1). Where a layered wall meets a patch
without layers, such as a pipe's end, the layers slide along it:

```yaml
layers: {kind: PrismLayers, first_height: 0.0005, growth: 1.2, count: 5}
layered: [walls, lamp]
```

The dual's faces are not flat: their corners are the centroids of the tetrahedra around an edge.
A `planarizer` flattens them by moving those centroids
({class}`~aquaflux.meshing.FacePlanarizer`), never making a cell less convex than the least convex
cell the mesh started with, a cell's volume more dependent on how its faces are triangulated than
the most dependent one's, a face more non-orthogonal or skewed than its worst face, or a convex
cell concave.
The boundary stays on the drawing and the layers keep their heights from the wall. It is off unless
given; `{kind: FacePlanarizer}` takes its defaults:

```yaml
planarizer: {kind: FacePlanarizer}
```

The faces of the patches listed in `mapped` are meshed as structured grids of quadrilaterals
rather than triangles ({class}`~aquaflux.meshing.MappedQuads`): each must be a band -- a cylinder
or cone round an axis, or a flat ring, bounded by two closed loops -- or four-sided. Layered, such a
wall carries hexahedra, so the solver's cells against it are hexahedra too. The lamp's cylinder
and the pipes' walls are bands; a lamp's rounded tip is not, so it needs a patch of its own:

```yaml
patches:
  lamp: {kind: FacesNear, points: [[0.4, 0.01, 0.0]]}
  lamp_tip: {kind: FacesNear, points: [[0.81, 0.0, 0.0]]}
layered: [walls, lamp, lamp_tip]
mapped: [lamp]
```

Pipes are found by themselves: a cylindrical face the fluid fills, with nothing else inside it,
bounded by two rims. Both rims are divided alike, into a multiple of four, so that the O-grid
can join them at every mesh size: where Netgen's own division would not do, the region is
triangulated again with the rims' division set. Its wall is meshed as a structured grid,
layered if its patch is, and its core -- what the layers leave -- is filled with a structured
O-grid of hexahedra ({class}`~aquaflux.meshing.PipeCore`): a square core, whose sides may bow
outward (`core_bow`) to even out its corners, and a shell out to the layers. Where a pipe opens
into the rest of the region, the O-grid stops a couple of rows short and the tetrahedra take over.
`pipes` names the patches whose faces are pipes instead (`[]` for none), and `pipe_core` sets how
their cores are filled:

```yaml
pipe_core: {kind: PipeCore, core_size: 0.5, core_bow: 0.3, shell: 2}
```

The middle of an open vessel may be meshed as cubes rather than polyhedra: a `core`
({class}`~aquaflux.meshing.OctreeCore`) of cubes `size` across, halved towards the walls until they
are no larger than `wall_size` (unset, `max_size`), and kept `clearance` clear of the walls and of
the layers' top (unset, twice the wall size). The tetrahedra fill the gap, grading from the wall's
size to the cubes', and the solver's cells there are polyhedra joining the cubes to the layers. A
core needs room: a gap narrower than about twice the clearance plus a wall cube, as round a lamp
in a narrow annulus, is left to the tetrahedra, and a region with no such room anywhere is refused.

```yaml
core: {kind: OctreeCore, size: 0.08}
```
