---
paths:
  - "aquaflux/io/**"
---

# Rules — `aquaflux/io/` (mesh import, field export, CAD import)

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors to inform *your*
> understanding; none of that may reach the shipped surface (root `CLAUDE.md`, **Comment Convention**).
> Spell out compressed-sparse-row (CSR) at first use per file. The dated record — the OpenFOAM restart, the
> ParaView verifications, the CAD measurements, test inventories — is in `.claude/notes/io-log.md`.

Reading external formats into a `Mesh`, and writing computed fields back out — into the case they came from
or as a file for a viewer. `io` owns file-format concerns only; `aquaflux/mesh/` owns representation. Mesh
*writing* is the polyMesh writer below.

## Responsibility and dependency direction

- `io` depends on `mesh` (and, for CAD, on `solids`); neither imports `io` back. `solids` is generic
  geometry, not physics (`solids.md`), which is what makes that direction acceptable.
- A reader ends at `Mesh.from_csr(...)`, which owns all topological validation; a reader does not
  re-implement index-range / degenerate-face / orphan-cell checks.
- **The 2D collapse is a mesh transform** (`mesh/collapse.py::collapse_extruded_direction`); `io` only
  detects which patches to collapse (OpenFOAM `empty`) and calls it.

## aquaflux's own mesh file (`.aqmesh`) — built

`io/aqmesh/mesh_file.py`: `AquafluxMeshWriter(path, overwrite=False, provenance=None)` /
`write_aqmesh`, `AquafluxMeshReader(path)` / `read_aqmesh` (`.read()`, `.header()`), `read_aqmesh_header`,
`AquafluxMeshHeader`; the case kind is `case.AquafluxMesh(path)`. Agreed with the project owner,
2026-10-21: **the native format is the mesher's output and the case's default mesh; OpenFOAM stays for
import and export** (polyMesh reorders faces, has no 2D or periodic mesh without extrusion and matched
cyclics, and our reader is ASCII only).
- **The arrays as `Mesh` holds them**, in one uncompressed `.npz` (`allow_pickle=False` on read):
  `node_coords`, `face_node_offsets`/`face_node_indices` (CSR), `owner`, `neighbour`, `patch_label`,
  `zone_label`, `neighbour_offset` (periodic only), and `header` (JSON bytes: `format`, `version`, `dim`,
  sizes, `patches` (named, in label order after `interior`/`boundary`), `patch_types`, `patch_groups`,
  `zones`, `periodic`, `digest`, `provenance`).
- **Read back exactly**: same pytree structure (every static field) and every leaf equal, dtype included
  -- the reader rebuilds patches and zones by label through `Mesh.from_csr`, whose name order matches.
  Pinned by `tests/unit/test_aqmesh.py` on a periodic graded 2D grid, a labelled perturbed hex grid
  (types, a group, a zone), tetrahedra and a hex beside a prism.
- **Refused**: another format or no header; a newer `version` (`AQMESH_VERSION` is 1; a reader reads its
  own and older); a missing array; a mesh whose `mesh_digest` differs from the header's (changed or
  damaged); an existing file without `overwrite`; a path not ending `.aqmesh`; non-JSON provenance; a
  partition's padded mesh. **Written whole or not at all** (a temporary file beside it, then
  `os.replace`).
- **`mesh_digest` lives in `aquaflux/mesh/digest.py`** (moved from `case/restart_file.py`), shared by
  checkpoints and this header.

## OpenFOAM polyMesh reader (ASCII) — built

Three pure seams so most logic tests without a filesystem:
- **`io/reader.py` — `MeshReader`** (`equinox.Module`, abstract `read() -> Mesh`). The axis that varies is
  format, so the ABC sits at the format seam, not around individual file parsers.
- **`io/openfoam/`**:
  - `records.py` — `FoamPatch`, `CellZone`, `PolyMeshData` (`NamedTuple`s, build-time). Faces are stored
    CSR; `neighbour_internal` is the raw interior-only file (padding is the assembler's job).
    `FoamPatch.neighbour_patch` carries a `cyclic` patch's `neighbourPatch`, `in_groups` its `inGroups`
    words (every list form read, count checked); the assembler hands types and groups to `Mesh.from_csr`
    (`patch_types` / `patch_groups`). `patch_face_range(patch)` is the one place a patch's face range
    becomes an index array (shared by the assembler and `cyclic.py`).
  - `foamfile.py` — the file envelope (comments, `FoamFile` header split, `is_binary`).
    **`read_foam_file(path)` is the one place a file on disk becomes a parsed OpenFOAM file**, so the
    ASCII-only gate is enforced once; `read_foam_body` is its body half. The writer needs the header (a
    field inherits its `class`), which is why the split exists — do not add a second read+gate.
  - `grammar.py` — body-grammar free functions sharing one `list_envelope`. Deliberately not a strategy
    hierarchy: the file kind is known statically at every call site.
  - `cyclic.py` — `fuse_cyclic_patches` (below).
  - `assembler.py` — `assemble(PolyMeshData) -> Mesh`: pad `neighbour` with `-1` (relies on OpenFOAM's
    interior-first ordering), `n_cells = max(owner, neighbour) + 1`, fuse cyclic pairs, map patches and
    cellZones, `Mesh.from_csr`.
  - `reader.py` — `OpenFOAMReader(MeshReader)` + `read_openfoam(path)` (a case dir or the polyMesh dir):
    assemble the faithful 3D mesh, then collapse when `empty` patches exist. `cyclic_match_tolerance`
    passes through to `assemble`.
  - `fields.py` — reading fields on an imported mesh (next section).

### Binding decisions

- **A polyMesh is always 3D; a 2D case is one cell thick between two `empty` patches**, collapsed to
  `dim == 2`. No `empty` patches ⇒ the 3D mesh.
- **A reserved patch name** (`interior`, `boundary`, `padding`; `mesh.groups.RESERVED_PATCH_NAMES`) raises
  naming the patch — no silent rename.
- **Unlisted boundary faces are legal in a mesh** (they fall into the automatic `boundary` patch), but every
  assembler's `build` refuses an uncovered boundary face, so solving needs a closure named `"boundary"`
  (#354, `boundary.md`).
- **ASCII only**: `format binary;` → `NotImplementedError`, detected, never misread.
- **Cell indices are OpenFOAM's**: `assemble` derives them from the labels it reads and the collapse keeps
  cells at their indices. **Faces are not**: the collapse renumbers them, and cyclic fusion drops the donor
  side's.

## OpenFOAM polyMesh writer (ASCII) — built

`io/writer.py` — **`MeshWriter`** (`equinox.Module`, abstract `write(mesh) -> Path`), the counterpart of
`MeshReader`. `io/openfoam/`:
- `disassembler.py` — **`disassemble(mesh) -> PolyMeshData`**, the pure inverse of `assemble`: interior faces
  first in **upper-triangular order** (owner < neighbour, sorted by owner then neighbour), then each named
  patch as one contiguous block in the mesh's patch order (a patch keeps its faces' stored order); every ring
  re-wound owner-outward (`reverse = stored_ring_is_outward(mesh) == swap`, through
  `ragged.rows(..., reverse)`); a patch without a declared type is written `patch`
  (`DEFAULT_PATCH_TYPE`); groups become `inGroups`; one `cellZone` per non-default zone.
- `mesh_writer.py` — pure `format_points` / `format_faces` / `format_labels` / `format_boundary` /
  `format_cell_zones` / `format_polymesh`, and `OpenFOAMWriter(MeshWriter)` + `write_openfoam(mesh, case)`,
  the only file I/O, into `<case>/constant/polyMesh`.
- **The header is `foamfile.format_header` + `SEPARATOR`, shared with the field writer** — one banner.

### Binding decisions
- **Cells are never renumbered**, only faces, so a field written with `write_openfoam_time` against the
  written mesh needs no permutation; `read_openfoam` returns the cells in the same order (pinned by
  `test_the_mesh_reads_back_with_its_cells_in_place`).
- **Points are written at `repr` precision** (shortest exact decimal), so they read back bit-identical.
- **Refused before anything is written**: a 2D mesh (no extrusion is built), a periodic mesh or a
  `cyclic`-typed patch (no `cyclic` pairs are written), a boundary face in no named patch (the format has no
  unnamed boundary), a named patch holding interior faces (a baffle is a `faceZone`, not written).
- ⚠️ **`overwrite=True` deletes every file in `MESH_FILES` first**, zones included, so a stale `cellZones`
  can never be read against the new mesh's labels; without it a polyMesh holding any of them is refused.
- Tests: `tests/unit/test_openfoam_mesh_writer.py` checks order and winding as properties of the output on
  meshes restated with shuffled faces, reversed rings and swapped owners (`tests/support/meshes.restated`),
  winding by an independent Newell-normal test (`newell_outward`), not a hand-written array; and a
  **fixed point**: `tests/fixtures/polymesh_3d_blockmesh/` (OpenFOAM 13 `blockMesh` output, its
  `blockMeshDict` beside it) is written back array for array.
- **Verified by OpenFOAM's `checkMesh`** (2026-10-19, `openfoam13` in Docker, `-allGeometry -allTopology`):
  upper-triangular ordering OK on every written test mesh; a writer with the winding fix removed fails 9
  checks (67 incorrectly oriented faces) and one with the sort removed fails "not in upper triangular
  order" (log). ⚠️ "Cells with small determinant" on the tet and hexahedron-beside-a-prism fixtures is
  a property of cells made mostly of boundary faces, not of the writer.

## Reading fields — `io/openfoam/fields.py`

- `parse_scalar_field` (pure), `read_surface_scalar_field(path, mesh)` (faces, e.g. `phi`),
  `read_volume_scalar_field(path, mesh)` (cells, internal block only — a `boundaryField` holds face values, a
  different index space).
- **A face field is placed by index, and the correspondence is checked, not assumed (binding)**: on an
  ordinary mesh aquaflux face `i` is OpenFOAM face `i`, a convention this package cannot enforce, and
  getting it wrong gives a plausible field, not an error. `read_surface_scalar_field` verifies the interior
  faces are the leading block and each named patch a contiguous ascending range. It therefore refuses a
  collapsed 2D case (renumbered) and, by a direct `neighbour_offset is not None` check, any periodic mesh — a
  seam patch declared right after the interior block can pass the positional check while reading the wrong
  entries.
  - ⚠️ `face_patches` carries the automatic `interior` (skip it) and `boundary` (non-empty ⇒ faces with no
    file values ⇒ raise) patches, which no field file writes.
  - A file mixes `uniform X` and `nonuniform List<scalar>` entries (a wall's flux is `uniform 0`).
- **`parse_vector_field` / `read_openfoam_time(case, time, names, mesh, *, extruded_axis=None)`** read a time
  directory by each file's header `class` (`volScalarField` → `(n,)`, `volVectorField` → `(n, 3)`, else
  refused), `uniform` or `nonuniform`. On a 2D mesh a vector's extruded component is **dropped after checking
  it is zero** (`1e-8` of the largest kept component, floored at one) — a refusal when the axis is wrong or
  the flow not planar. The axis is the argument or `infer_extruded_axis(case, mesh)`; a case file passes its
  stated or mesh-path axis (`case.md`). Only the cell count is checked against the mesh. A missing field is
  reported with what the directory holds.

## Cyclic-patch fusion — built

A `cyclic` pair (`neighbourPatch` naming each other) is one periodic seam. `cyclic.fuse_cyclic_patches` turns
it into interior faces carrying `neighbour_offset` before `Mesh.from_csr` — the same representation
`structured_grid_2d(periodic=...)` produces (`mesh.md`).
- **The match is geometric, not declared**: `transform` is not parsed. The translation is estimated from the
  two patches' centroid means and centroids are nearest-neighbour matched (`cKDTree`, order-independent)
  within `DEFAULT_MATCH_TOLERANCE` (a fraction of the kept patch's extent; overridable via
  `cyclic_match_tolerance` on `assemble` / `OpenFOAMReader` / `read_openfoam`). A rotational or mismatched
  pair fails to match and raises naming the pair — **rotational cyclics are not supported**.
- **The patch declared first is kept**; the donor's faces are dropped. Both are removed from `face_patches`.
- **Fusion precedes the 2D collapse**, so the collapse sees ordinary interior faces with `neighbour_offset`
  and needs no cyclic handling.
- Deferred: `cyclicAMI` and rotational `cyclic` (left as ordinary disconnected patches).

## OpenFOAM field writer (ASCII) — built

`io/openfoam/field_writer.py`: `write_openfoam_time` (a time directory) and `write_openfoam_field` (one file).

- ⚠️ **Only the internal values are ours.** `dimensions` and the whole `boundaryField` are copied verbatim
  from a **template** (the case's own field, `template_time="0"` by default). That makes the output
  restartable with the case's own boundary conditions, reproduces `empty` patches the collapse removed, and
  gives units one home. Do not turn this into a boundary-condition translator. A solved time directory as
  template gives the solved per-patch values (closer continuation); `"0"` gives the case's intended
  conditions.
- **The internal block is a straight dump** — cell indices are OpenFOAM's.
- ⚠️ **`value $internalField;` resolves against the template's own internal entry**
  (`FieldTemplate.internal_field`); written literally it would expand every cell onto the patch and OpenFOAM
  fails with an error naming neither. Every other `$name` is left alone.
- **A 2D vector is padded back to three components on the extruded axis**, which the `Mesh` no longer knows.
  `infer_extruded_axis` identifies it from the case's polyMesh: the collapse keeps the surviving axes in
  ascending order with their coordinates, so exactly one candidate's two extents match (not "the thinnest
  axis"); ambiguity raises. Consulted only when a two-component vector arrives — such a mesh is one cell
  thick, so parsing `points` is cheap. `extruded_axis` overrides; `format_volume_field` /
  `write_openfoam_field` default to `-1`.
- **Refuses a non-finite field by default** (`allow_non_finite=True` overrides).
- **Verified by an actual OpenFOAM restart** (2026-09-12, `openfoam13` in Docker, pitzDaily): first-iteration
  residuals match a control restarted from OpenFOAM's own fields with `phi` removed, to every printed digit
  (log). ⚠️ `omega`'s ~0.8 initial residual is `omegaWallFunction`, in both arms.
- ⚠️ **`phi` is not written, and that is a decision — do not "fix" it by writing aquaflux's flux.** `phi`'s
  defining property is discrete divergence-freeness in one particular continuity operator; aquaflux's flux is
  conservative in aquaflux's operator (the same argument that makes a scalar ride the flow's own Rhie–Chow
  flux, `transport.md`), not in OpenFOAM's, and OpenFOAM's incompressible `phi` is volumetric while ours is a
  mass flux. Without it OpenFOAM recomputes the flux from `U` — a re-derived-flux restart, not a bit-identical
  continuation. It would also be hard (a face field; the collapse renumbers faces). Argued, not measured:
  writing our flux into a case and reporting `fvc::div(phi)` is the check to run before reopening this.

## VTK XML `.vtu` writer — built

`io/vtk/{topology,xml,writer}.py`: `write_vtu` (one frame), `write_pvd` (a collection). `topology`
reconstructs connectivity (pure numpy), `xml` serializes, `writer` opens files. Standard library only.

- **Why hand-rolled**: cells are implicit in a face-based mesh, and `VTK_POLYHEDRON` (42) is defined by its
  face-node lists, `VTK_POLYGON` (7) by a ring one edge-walk away. VTKHDF polyhedra were unreadable by the
  installed ParaView, meshio cannot mix polyhedra with other types, `vtk`/`pyvista` are a ~100 MB dependency
  (the `ui` extra uses them in a separate package), XDMF's polyhedra are poor.
- **Hexahedra are written as `VTK_HEXAHEDRON` (12)**, everything else as polyhedra. `_hexahedron_vertices`
  recognizes six quadrilateral faces and eight distinct points, and orders vertices from the outward rings
  (first ring reversed is 0–3; each one's outward edge off that face ends at 4–7). The face stream carries
  only polyhedra (`faceoffsets = -1` for a hexahedron) and is omitted when none remain. Only hexahedra are
  recognized (widening was not agreed). Why: the viewer's cost — at 1M cells, file 70 vs 198 MB, peak memory
  393 vs 1074 MB, `extract_surface` 0.06 vs 1.13 s (2026-10-05, VTK 9.7.1, macOS arm64; log).
- ⚠️ **Winding: a stored ring is not owner-outward.** `Mesh.from_faces` accepts either direction and orients
  the normal separately (roughly half of a generated grid's rings are inward). `aquaflux.mesh.stored_ring_is_outward` (`mesh/winding.py`, shared with the polyMesh writer)
  recovers each ring's direction with the mesh's own face-geometry strategy, and an entry is reversed exactly
  when that disagrees with the side listing it. Pinned by a per-face Newell-normal test and by byte-identical
  output with every ring reversed — not by a hand-written expected array.
- ⚠️ **A 2D ring is a permutation chase, and its check is that the first return to the start is at the
  cell's own edge count** — a cell of two rings returns at the right length after traversing the shorter
  ring twice.
- ⚠️ **A periodic seam face listed under its neighbour uses the neighbour's own nodes**, found at `position −
  neighbour_offset` by one `cKDTree` query after the winding reversal (`topology._onto_own_side`); a missing
  counterpart is refused (`PERIODIC_NODE_MATCH_TOLERANCE`, 1e-2 of the seam face's mean edge). Without it 2D
  raised and 3D silently doubled or quadrupled seam-cell volumes. The seam shift happens before hexahedron
  ordering, so periodic hex cells stay type 12.
- **Binary is the default** (`<AppendedData encoding="raw">`, `UInt64` headers; `binary=False` for
  inspection). The reason is load time (ParaView opened a 216k-cell file 9x faster), not size — a toy mesh's
  raw file is larger. Index widths come from `mesh.connectivity.index_dtype`.
- ⚠️ **A 2D vector is padded on the last axis, not the extruded one** — the opposite of the OpenFOAM writer,
  because a `.vtu` writes the mesh's own two coordinates into `z = 0`.
- **Refuses nothing for being non-finite** — also the opposite: a viewer is how one finds where a solution
  went non-finite.
- **Verified by opening the files** (ParaView 5.12.0 and VTK 9.7.1; log): `IntegrateVariables` / `CellSize`
  match aquaflux's volumes to ~1e-15 on planar-faced meshes. ⚠️ **On non-planar faces (a `polyDualMesh`)
  aquaflux's cell volumes differ from VTK's by up to 4.6 % and that is not a writer defect** — a volume
  depends on the face triangulation (aquaflux uses the centre fan); our `.vtu` and ParaView's own OpenFOAM
  reader agree to 1e-6.

## Boundary patches as VTK `.vtp` + `.vtm` — built

`io/vtk/patches.py::write_patches(mesh, fields_by_patch, path, *, patches=None, binary=True)`: one `PolyData`
file per boundary patch (`<stem>/<patch>.vtp`: compacted nodes, faces as `Polys`, or `Lines` at `z = 0` in
2D; face fields as `CellData`) and a `vtkMultiBlockDataSet` index (`<stem>.vtm`, one block per patch named by
it, relative `file`). Every boundary patch holding faces is written by default (`boundary_patches(mesh)`).
**The layout is a contract with the `aquaflux_ui` viewer** (`fields.vtu` + `patches.vtm` in a run directory;
agreed with the project owner). Reuses `xml.py`'s `_Arrays`, `_narrowed`, `cell_data_arrays`. **Faces are
wound out of the domain** via `stored_ring_is_outward`. A field of the wrong length is refused naming the
patch. Used by `case.PatchVtk`. Checked in ParaView against a run's `incident_power` (log).

`io/vtk/surface.py::write_triangles(corners, fields, path, *, binary=True)`: triangles that are no mesh's
boundary (a mesh file's drawing, for the browser's face picker) as one `.vtp`, shared corners written once
(`np.unique` of the corners), a value per triangle as `CellData` (`face`); built on `vtp_parts`.

`io/vtk/writer.py::write_hexahedra(points, corners, path, *, point_fields=None, cell_fields=None, binary=True)`:
axis-aligned cells that are no mesh's (an octree's leaves, `aquaflux.octree.SampledField.hexahedra()`) as
`VTK_HEXAHEDRON`, values at the points (`PointData`, `vtu_parts(point_fields=)`) and per cell. Corners come
in binary order (corner `4i+2j+k` at `(i,j,k)`, `aquaflux.octree.tree.CORNERS`) and are mapped to VTK's in one
place, `_BINARY_TO_VTK_HEXAHEDRON`. A coarse leaf beside finer ones is written with its own eight corners,
so a viewer draws it from those even where the octree's own interpolant uses the finer nodes on its faces.
Pinned by `tests/unit/test_vtk_hexahedra.py`.

## The two field writers share a helper, not a contract

`write_vtu(mesh, fields, path)` and `write_openfoam_time(case, time, fields, mesh)` are deliberately not one
`FieldWriter` ABC: one writes a file whose payload is the mesh, the other N files into an existing case
needing per-field templates, and every other keyword belongs to one side. Their shared formula is
`io/cell_fields.as_cell_values(name, values, n_cells)`. ⚠️ "Not the same abstraction" says nothing about
whether two functions share an implementation — ask both questions; `tools/sibling_builders.py` cannot see
this shape (neither constructs a class).

## CAD import: STEP → exact bodies and emitting triangles — built (#505)

`aquaflux/io/cad/`: `read_step(path, placement) -> CadModel`, which hands out `solid(name)` (an exact
`solids` body), `fluid(*names)` (an `Outside` — the vessel as the water it holds) and `triangles(name,
chord=, facet_size=)` (an emitting surface).

- **Kernel: OpenCASCADE through OCP** (decided with the user, 2026-09-24): arbitrary CAD needs a full B-rep
  kernel, and a reader for a primitive subset would be dominated. We use the binding (`cadquery-ocp-novtk`),
  not CadQuery. ⚠️ **Pinned exactly at `8.0.1.0.0`** (the `cad` extra) because minor versions break the API
  (collections moved to `OCP.collections`, `TopoDS.Solid_s` → `TopoDS.Solid`, `Bnd_Box.Get()` unusable —
  use `CornerMin`/`CornerMax`). `-novtk` cannot be installed beside CadQuery. No cp310 wheel, hence
  `requires-python >= 3.11`. It runs in CI (the `test` extra carries `aquaflux[cad]`; `test_cad_step.py` is
  in the optional-dependency census).
- **Layout — one module touches the kernel**: `kernel.py` (`OpenCascade`, the only `import OCP`; no kernel
  object leaves `CadModel`), `faces.py` (face records as numbers), `recognize.py` (`RecognitionRule`s),
  `placement.py` (`Placement(matrix, offset)`, orthogonal only — reflections allowed, stretch or scale
  refused; units are the reader's job), `model.py` (`CadModel`, `InexactBody`, `read_step`, which imports the
  kernel lazily so `import aquaflux.io` needs none).
- **Recognition proposes; the proposal is checked.**
  - `CurvedPieces`: the solid is the union of whole-turn convex surfaces it lies inside. ⚠️ Which side the
    solid is on is recorded per face (`solid_inside`) — a pipe wall and a bore are the same cylinder; a
    whole-turn face with the solid outside is refused. ⚠️ Faces of one surface are merged on a canonical
    axis (`_Axial`) before "whole turn" is decided, since exporters split periodic faces. A remaining partial
    face (a pipe end cut to the vessel it joins) marks the proposal `stands_alone=False`: `solid()` refuses
    it, `fluid()` accepts it.
  - `ConvexPolyhedron`: an all-planar solid with every vertex inside every face plane is the `Intersection`
    of its `HalfSpace`s — proven, not checked; in a union check the drawing's own solid stands in for it.
  - A torus face is described (`TorusFace`: centre, axis, major/minor radius, swept angle,
    `solid_inside` = the solid inside the tube, from the normal against the nearest centreline point; the
    drawing view names it a torus; the mesher finds pipes from triangles, not records), but no body
    stands for a piece of one: `CurvedPieces` declines it like an
    `OtherFace` ("surfaces no rule describes (Torus)"). Pinned by `test_cad_step.py` (a bend inside, a
    groove outside).
  - Anything else raises `UnrecognizedSolid` naming every rule's reason. **No fallback to triangles**:
    routing an unrecognized solid to `radiation.TriangleBody` (via `CadModel.triangles`) is not built.
- ⚠️ **The check is boundary distance, not volume.** Points sampled on both boundaries (at
  `sample_spacing`, extent/16 by default) must lie within the drawing's own declared tolerance
  (`ShapeAnalysis_ShapeTolerance`) of the other boundary, in both directions (each direction has its own
  test: a phantom flange, a missed thin fin). A volume test cannot be set: the Sozzi drawing's own volume is
  1.5e-5 off the analytic shape (its B-spline saddle is fitted to 10 µm), so a tight volume tolerance fails a
  correct file and a loose one passes a missing baffle. Nothing finer than the drawing's export precision can
  be detected. Let the spacing, not the chord, set the sample count (a 1e-6 chord ran > 10 min).
- **Kernel details found on the real file**: call `XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 1.0)` before
  transfer, or lengths come out in millimetres (the `xstep.cascade.unit` static does not reach XCAF); take
  `solid_inside` from face orientation, not point classification; never recognize from edge types (seam
  lines report as B-splines). Reading edge topology is allowed for meshing (feature edges, adjacency;
  project owner, 2026-10-19): the ban is on recognition. The mesher's kernel needs are in `meshing.md`.
- **Assemblies**: locations accumulate down the component tree; a part's own name beats its instance's; a
  repeated name gets `[k]`, several solids under one name `.k`.
- **Triangulation bounds facet size by splitting the boundary shells** (`BRepAlgoAPI_Splitter`) with three
  families of planes `facet_size` apart, then meshing — OCCT bounds chord and angle only, leaving
  full-length slivers. ⚠️ Split the shells, never the solid (that meshes the cutting planes' interior faces).
  Longest-edge bisection and `ShapeUpgrade_*Divide*` were tried and dropped (log). Zero-area triangles are
  dropped (a point source to `radiation.Surfaces`). A reflecting placement keeps triangles outward
  (`BRepBuilderAPI_Transform` reverses faces itself).
- **Regions for a mesher — built** (`topology.py` records, numpy only; kernel `region`, `topology`,
  `write_brep`, `project`). `CadRegion(solids, without=())` is a plain value (union of `solids` less the
  union of `without`); `CadModel` builds and caches its shape, so no kernel object leaves the model, and
  answers `topology(region) -> CadTopology`, `face_triangles(region, chord=, facet_size=) -> FaceTriangles`,
  `write_brep(region, path)` and `project(region, points, faces=None, edges=None) -> Projection` (onto
  the boundary, a given face, or a given edge; the face reported for an edge is the first meeting there).
  - **Faces, edges and vertices are numbered in exploration order** (`TopExp.MapShapes_s`, first visit,
    each once) — the order Netgen numbers the faces of the BREP file in (`meshing.md`). Pinned by
    `test_the_file_keeps_the_face_order` (re-read file's explorer order) and, for a single solid, by
    equality with `describe`.
  - **`edge_angle` is the signed fold, positive convex**: `acos(n1·n2)`, negative when the second face's
    outward normal points along the direction into the first (`n1 × t`, `t` the edge as the first face's
    boundary traverses it). ⚠️ **Exploring a face's edges already composes the face's orientation** — a
    second flip for a reversed face made half a box's edges concave. `nan` on an edge not bounding exactly
    two faces; `seam` from `BRep_Tool.IsClosed_s` (a seam lists its one face twice among its ancestors, so
    faces are deduplicated by index). On the Sozzi water (lamp removed) the only concave edges are the two
    pipe junctions: the inlet riser on the vessel's cylinder, the outlet on its end plate.
  - **`triangulate` now returns `(triangles, face)`**: a face cut by `facet_size` keeps its index, traced
    through the splitter's `Modified` history (`_cut` returns the splitter). `CadModel.triangles` keeps
    its return of triangles only.
  - ⚠️ **`project` measures to the boundary shells, never the solid**: a point inside a solid is at
    distance zero from it, and the support returned is the solid, not a face. A point closest to an edge or
    vertex gets the first face meeting there (exploration order). One kernel query per point (a Python
    loop) — for boundary nodes, not a volume mesh.
- The fluid read from the Sozzi drawing shadows exactly as the hand-derived occluder (0 differing pairs,
  2026-09-24; log and `.claude/notes/radiation-log.md`).

## Deferred (additive; no seam changes)

Binary polyMesh; `faceZones`/`pointZones` (reading, and writing baffles as `faceZones`); writing `cyclic`
pairs and 2D extrusions; `.gz` and multi-region cases; other formats as `MeshReader` subclasses under `io/<format>/`; `.pvtu` for
`parallel/PartitionedMesh` (`xml.cell_data_arrays` already yields what a `.pvtu` declares).

## Tests

Parse, assemble, cyclic fusion and collapse are file-free (`tests/support/polymesh.py` fixtures);
`test_openfoam_reader.py` runs end to end on committed fixtures against the structured generators.
⚠️ **A placement test must encode each face's own index in its value**, and must include a fixture whose
patches are declared out of `startFace` order — the committed two-cube fixture declares them in order, so it
alone cannot tell "laid out by startFace" from "declaration order". Interior placement on a real mesh is
measured by `validation/bfs3d_openfoam/phi_placement.py` (log). ⚠️ When checking continuity of an imported
flux, normalize by the domain flow rate, not a cell's own throughput (recirculation cells make the latter
blow up a fixed absolute error).
