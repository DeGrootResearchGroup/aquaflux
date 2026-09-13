---
paths:
  - "aquaflux/io/**"
---

# Rules — `aquaflux/io/` (mesh import/export)

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors to inform *your*
> understanding. Per the root `CLAUDE.md` **Comment Convention**, none of
> that provenance may reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the
> *math/format*, never the reference code, the `.claude/` rules, the design notes, or the author's
> own papers. **Acronyms:** spell out compressed-sparse-row (CSR) at first use per file.

Reading external mesh formats into an aquaflux `Mesh`, and writing computed cell fields back out
— into the case they came from, or as a self-contained file for a viewer (mesh writing itself is
still deferred). This package
owns **file-format concerns only**; `aquaflux/mesh/` owns mesh representation. Governed by the root
`CLAUDE.md` Engineering Principles.

## Responsibility & the one-way dependency
- `io` depends on `mesh` (it builds a `Mesh`); `mesh` must **never** import `io`. A reader ends at
  `Mesh.from_csr(...)`, which owns all topological validation — a reader does **not** re-implement
  index-range / degenerate-face / orphan-cell checks (one source of truth).
- The **2D collapse is not an io concept** — it is a general mesh transform,
  `aquaflux/mesh/collapse.py::collapse_extruded_direction`, reusable and tested file-free. The io
  layer only *detects* which patches to collapse (OpenFOAM `empty` type) and calls it.

## Structure — BUILT (OpenFOAM polyMesh reader, ASCII)
Three pure seams so ~80% of the logic tests with no filesystem (separate I/O from computation):
- **`io/reader.py` — `MeshReader`** (`equinox.Module` + `abc.abstractmethod read() -> Mesh`): the
  format-agnostic strategy interface, mirroring the operator/scheme/BC/solver strategies. The axis
  that genuinely varies is *format* (OpenFOAM now; Gmsh/VTK/CGNS later) — so the ABC lives here, at
  the format-crossing seam, **not** around the individual file parsers.
- **`io/openfoam/` (the first reader):**
  - `records.py` — `FoamPatch`, `CellZone`, `PolyMeshData` value objects (`NamedTuple`, build-time,
    not JAX pytrees). `PolyMeshData` is the cohesive record handed to the assembler (pass the
    record, not a fistful of loose arrays). Faces are stored **CSR already**; `neighbour_internal`
    is the raw interior-only `neighbour` file (padding to full length is a *semantic* step, done in
    the assembler, not the parser).
  - `foamfile.py` — the shared file envelope: strip `/* */` + `//` comments, split the
    `FoamFile { … }` header dict from the body, and `is_binary` (gates ASCII vs binary in **one**
    place).
  - `grammar.py` — the body-grammar **free functions** (`parse_vector_list` / `parse_scalar_list` /
    `parse_face_list` / `parse_boundary` / `parse_cell_zones`), sharing one `list_envelope` for the
    `N ( … )` frame + count-check. **Deliberately not a Strategy hierarchy** — the file kind is
    known statically at every call site, so parser-polymorphism would vary over nothing.
  - `assembler.py` — `assemble(PolyMeshData) -> Mesh` (pure, file-free): pad the interior-only
    neighbour with the `-1` sentinel (relies on OpenFOAM's upper-triangular ordering — interior
    faces first), derive `n_cells = max(owner, neighbour) + 1`, map boundary patches →
    `face_patches` and cellZones → `cell_zones`, then `Mesh.from_csr`.
  - `reader.py` — `OpenFOAMReader(MeshReader)` + `read_openfoam(path)`. `read()` = assemble the
    faithful 3D mesh, then collapse when `empty` patches are present. Accepts a case dir (resolves
    `constant/polyMesh`) or the polyMesh dir directly. `_read_field` handles the *optional*-file
    case and delegates the rest to `foamfile.read_foam_body`.
  - **`foamfile.read_foam_file(path)` is the one place a file on disk becomes a parsed OpenFOAM
    file**, so the ASCII-only limitation is enforced once. It lives beside the `is_binary` predicate
    it uses, and `read_foam_body(path)` is its body half (`read_foam_file(path).body`) for the
    callers that need no header. **`reader.py` is NOT the package's only file I/O** — `fields.py`
    and `field_writer.py` open files too; what is centralized is the *binary gate*.
    ⚠️ The split exists because the **writer needs the header** — a field inherits its `class` from
    its template — and `read_foam_body` throws the header away. Adding a second read+gate in the
    writer would have put the ASCII limitation in two places, which is the one thing this seam is
    for.
  - `fields.py` — **reading a scalar field written on an already-imported mesh** (`phi` above all).
    `parse_scalar_field(body, n_internal, patch_sizes)` is pure and tests on snippets;
    `read_surface_scalar_field(path, mesh)` places the values on the mesh's faces, and
    `read_volume_scalar_field(path, mesh)` reads a `volScalarField`'s internal block onto cells.
    **The two are asymmetric on purpose.** A face field needs the ordering guard below; a cell field
    does not, because `assemble` derives cell indices from the `owner`/`neighbour` labels it reads,
    so a cell's index is OpenFOAM's own by construction rather than by convention. The cell reader
    also returns the internal block *only*: a `volScalarField`'s `boundaryField` holds **face**
    values, a different quantity on a different index space, so concatenating them as the surface
    reader does would produce something no consumer wants.

## Structure — BUILT (OpenFOAM field writer, ASCII) — 2026-09-12

Writing computed cell fields back into the case they were read from, as an ordinary time directory,
so a solved state can be opened by the tools the reference solution is opened with — or used to
restart a run. `io/openfoam/field_writer.py`, exported as `write_openfoam_time` (a whole time
directory) and `write_openfoam_field` (one file).

**⚠️ THE DESIGN DECISION THAT CARRIES THIS: ONLY THE INTERNAL VALUES ARE OURS.** Dimensions and the
whole per-patch `boundaryField` dictionary are copied verbatim from a **template** — the case's own
field of the same name, normally in `0/`. Three things follow, and each is a reason not to "improve"
this into a boundary-condition translator:
- **It is what makes the output restartable** rather than merely viewable. The boundary conditions
  are the case's own by construction, spelled the way its solver expects, so there is no aquaflux-BC
  → OpenFOAM-BC mapping to get wrong or to keep in step with either side.
- **It handles the types this package cannot reconstruct at all.** An `empty` patch is *removed from
  the mesh* by the 2D collapse, so a writer working from the imported mesh could not put it back.
  Copying the dictionary does, and a test pins exactly that.
- **It removes the units table.** A field's `dimensions` has one home — the template — so there is
  no second place for it to be recorded and disagree.

**Why the internal block is a straight dump, with no permutation.** A cell's aquaflux index *is*
OpenFOAM's: `assemble` derives cell indices from the `owner`/`neighbour` labels it reads rather than
renumbering, and `collapse_extruded_direction` keeps cells at their indices (it removes faces, not
cells — `mesh/collapse.py`'s own docstring states this). ⚠️ **Faces are the opposite** — the collapse
does renumber them — which is a second, independent reason the patch dictionaries are copied rather
than rebuilt from the mesh.

**⚠️ A 2D vector field is padded back to three components, and the dropped axis is genuinely NOT
recoverable from the `Mesh`** — `collapse_extruded_direction` infers it, slices with it, and
discards it, so `node_coords` really is `(n_nodes, 2)` and a y-extruded case collapses to a mesh
byte-identical to a z-extruded one. There is no "plane the flat cells lie in" left to inspect.

**It IS recoverable from the case on disk, and `write_openfoam_time` does that by default**
(`infer_extruded_axis`). The collapse keeps surviving axes in **ascending** order and preserves
their coordinate values, so there are only three candidates and the polyMesh's extents on the two
axes a candidate keeps must equal the collapsed mesh's own two extents — exactly one matches. That
is an identification, not the heuristic "take the thinnest axis", which would misread a domain thin
in a resolved direction. Ambiguity (no match, or several) raises and asks for the axis rather than
guessing.

⚠️ **The reason this is cheap is worth keeping, because the obvious objection is wrong.** Parsing
`points` sounds expensive at reactor scale — but **a mesh that needs this is one cell thick by
construction**, so it is never one of the large ones (pitzDaily's `points` is 916 KB); a genuinely 3D
mesh is never collapsed and never asks. The expensive meshes and the ambiguous meshes are disjoint
sets. It is also only consulted when a **vector** field arrives with two components, so a
scalar-only or 3D write opens nothing.

`extruded_axis` survives as an explicit override — for a write with no case to consult, or when the
extents cannot decide. `format_volume_field` and `write_openfoam_field` keep the plain `int` default
of `-1`, since neither is handed a case directory.

**Measured:** pitzDaily infers axis 2, and the Docker restart below is reproduced exactly through the
inference path (no `extruded_axis` passed).

**Refuses a non-finite field by default** (`allow_non_finite=True` to override). The point of
writing is that it can be read back, and a solver handed a NaN restart fails in a way that never
names this file.

**Verified by an actual restart, not only by a round trip (2026-09-12).** Round trip through this
package's own reader is exact (`0.000e+00`, not "close") on the committed pitzDaily case and on a
`structured_grid_2d`, with values encoding their own cell index so a permutation would be visible
rather than plausible. Then, against `openfoam13:latest` in Docker
(`docker run --rm -v "$PWD":/work -w /work/case openfoam13:latest foamRun`): OpenFOAM's own converged
`2000/` fields were read through aquaflux, written back as `2500/` by this writer, and `foamRun`
restarted from `2500` for five iterations. **The velocity was handed to the writer as TWO
components** so the 2D padding path is what the solver reads — a zero on the wrong axis makes the
restart nonsense and the first residual says so.

**It matches a control to every digit printed.** Against `foamRun` continuing from OpenFOAM's own
`2000/` with `phi` removed, first-iteration initial residuals:

| | control (OpenFOAM's own fields) | restarted from ours |
|---|---|---|
| Ux | 0.000657723 | **0.000657723** |
| Uy | 0.00685219 | **0.00685219** |
| p | 0.0127883 | **0.0127883** |
| omega | 0.81593 | **0.81593** |
| k | 0.0269429 | **0.0269429** |

⚠️ **`omega`'s initial residual is ~0.8 in BOTH arms** — that is `omegaWallFunction` overwriting the
near-wall values each iteration, not a defect in the written file. Read it against the control, never
on its own.

**⚠️ `phi` IS NOT WRITTEN, AND THAT IS THE WHOLE OF THE DIFFERENCE FROM A TRUE CONTINUATION.** A time
directory OpenFOAM writes also holds `phi`, the face flux (`surfaceScalarField`), which this cell
field writer does not produce; a solver restarting without it recomputes it from `U`. Measured: with
`phi` present the control's first Ux residual is `0.000616969` against our `0.000657723`, and
**deleting `phi` from the control collapses that gap to zero** — which is how the cause was
established rather than guessed. So a restart from this writer is correct and stable, but is a
*re-derived* flux restart, not a bit-identical continuation.

**⚠️⚠️ AND THAT IS A DECISION, NOT A GAP TO CLOSE LATER — DO NOT "FIX" IT BY WRITING aquaflux's
`phi`.** `phi` is not merely "the flux on a face": its defining property is **discrete
divergence-freeness with respect to one particular continuity operator**. That is exactly why
`.claude/rules/transport.md` makes it binding that a scalar advects on the flow's own Rhie--Chow flux
and never on a rebuilt `(u·n)A` — a rebuilt one "satisfies no discrete continuity, so a uniform
tracer would not stay uniform". **That argument is symmetric.** aquaflux's flux is conservative in
*aquaflux's* operator — its differentiated `a_P`, its flux-continuous harmonic conductance, its
non-orthogonal correction, its coupled p--U block — and there is no reason `fvc::div(phi) == 0` in
OpenFOAM's operator except by coincidence. Writing it would therefore *not* beat omitting it (the
first pressure correction projects it onto OpenFOAM's own manifold either way, which is what
recomputing from `U` already does), could plausibly be **worse** than the `fvc::flux(U)` OpenFOAM
forms with its own interpolation, and would silently assert a conservativeness that does not hold —
in a file that looks right. Note also that OpenFOAM's incompressible `phi` is **volumetric** while
aquaflux's primitive is `mass_flux` (`volume_flux(mdot, rho)` is derived), so the two are not even
the same kind of quantity once density varies.

*Separately* it would also be hard — `phi` is a **face** field and the `empty`-patch collapse
renumbers faces, which is why `read_surface_scalar_field` refuses to run on a 2D case at all. But the
difficulty is the second reason, not the first; if the renumbering were solved tomorrow the answer
would still be no. ⚠️ This is argued from the discretization, not measured: writing an aquaflux
solution's flux into a case and reporting `fvc::div(phi)` would settle it outright, and is the check
to run before anyone reopens this.

**⚠️⚠️ THE TRAP THAT ONLY A REAL SOLVER RUN FOUND: `value $internalField;`.** A `0` directory gives a
patch the interior value by macro, and it is well defined only while that interior value is
*uniform*. This writer always writes a list, so a macro carried through literally expands **every
cell** onto the patch. OpenFOAM rejects it with `compound has already been transferred from token` at
the patch's `value` — an error naming neither the macro nor the cause. The fix is that
`$internalField` resolves against the **template's** own internal entry, which is what the entry
meant where it was written; `FieldTemplate.internal_field` exists only to carry it. Every *other*
`$name` is left alone deliberately — the whole `boundaryField` and `dimensions` are carried across, so
those still resolve, and `internalField` is the one entry this writer replaces.

**`template_time` is a real choice, and both options restart.** `"0"` (the default) gives the case's
*intended* boundary conditions, with any macro resolved against that file's uniform value. A solved
time directory instead gives the solved per-patch `value` lists. The second is the closer
continuation; the first is the more honest statement of the case's boundary conditions and does not
carry another solver's values into your output.
## Structure — BUILT (VTK XML `.vtu` writer) — 2026-09-12

The general-purpose output path: a mesh and its cell-centred fields, written as a VTK XML
unstructured grid a viewer opens directly. `io/vtk/{topology,xml,writer}.py`, exported as
`write_vtu` (one frame) and `write_pvd` (a transient collection). Three seams mirroring the reader's:
`topology` reconstructs the connectivity (pure numpy, file-free), `xml` serializes it (pure), and
`writer` is the only part that opens a file.

**Why hand-rolled, and why VTK's polyhedron in particular.** The mesh is face-based — nodes,
owner/neighbour, ragged CSR face rings — and cells are *implicit*: there is no cell→vertex list.
`VTK_POLYHEDRON` (type 42) is defined *by its face-node lists*, and `VTK_POLYGON` (type 7) by a
vertex ring that is one edge-walk away, so the connectivity is a reconstruction rather than a
translation. Alternatives were measured out before this was built: **VTKHDF**'s polyhedron support is
unreleased and the installed ParaView (5.10.1 / 5.12.0) could not read it anyway; **meshio** cannot
mix polyhedra with other cell types and would still need the topology hand-built; the **`vtk` /
`pyvista`** wheels are a ~100 MB C++ dependency for a lean JAX package; **XDMF**'s polyhedral support
is poor. Nothing but the standard library is imported.

### ⚠️ THE WINDING RULE IS NOT "KEEP THE RING AS STORED" — THE PREMISE THAT SAYS SO IS FALSE
VTK needs each polyhedron face listed outward from the cell listing it, and the obvious rule — keep a
ring as stored under its **owner**, reverse it under its **neighbour** — is only half of it. It
assumes a stored ring is owner-outward, and **a `Mesh` explicitly declines to promise that**:
`Mesh.from_faces` accepts either direction ("the winding *direction* is free") and orients the
**normal**, in a separate array, leaving the ring untouched. Measured on the day this was written:
`structured_grid_3d(2,2,2)` stores **20 of 36** rings owner-outward and `structured_grid_2d(3,2)`
**9 of 17** — so a writer built on the premise emits inward faces on roughly half of every cell.
(pitzDaily read through the OpenFOAM reader is 12448 of 24730, because the `empty`-patch collapse
rebuilds the faces and does not preserve OpenFOAM's own convention.)

So the rule is the **composition of two facts**: `topology.stored_ring_is_outward` recovers the
ring's own direction — by running the mesh's *own* face-geometry strategy
(`unoriented_geometry` → `orient_owner_outward`) rather than re-deriving the orientation test — and
the entry reverses exactly when that disagrees with the side listing it. Pinned two ways:
`test_every_emitted_polyhedron_face_winds_outward_from_its_own_cell` checks each emitted ring's own
Newell normal against its cell centroid, and
`test_the_output_is_invariant_to_the_direction_the_rings_are_stored_in` rebuilds the same mesh with
**every** ring reversed and demands byte-identical output. A test comparing against a hand-written
expected array would have pinned one generator's incidental winding instead, and passed.

### The 2D ring is a permutation chase, and a fixed-length walk is not enough to validate it
Each of a cell's edges is already directed outward-consistently by the rule above, so a cell's edges
form one closed *directed* cycle and chaining them needs no geometric tie-break — `n` vectorized
steps for a mesh whose largest cell has `n` edges, no Python loop over cells. ⚠️ **Checking only that
the walk returns to its start is wrong**: a cell whose edges form two rings returns at every multiple
of the shorter one, so a walk of `max(count)` steps lands back at the start and emits the shorter
ring traversed twice, at exactly the right length. The check is that the *first* return is at the
cell's own edge count. This was a live bug, found by the test written for it
(`test_a_2d_cell_whose_edges_form_two_rings_is_refused`, concentric squares as one cell).

### Binary is the default, and that was decided before shipping rather than after
`<AppendedData encoding="raw">` with `header_type="UInt64"`; `binary=False` gives the same values to
the last bit as decimal text, for reading a small mesh by eye. **Measured 2026-09-12** — macOS arm64,
ParaView 5.12.0, `structured_grid_3d(60,60,60)` = 216 000 cells / 658 800 faces: raw **40.8 MiB
written in 1.9 s and opened by ParaView in 0.2 s**, against ASCII **63.1 MiB / 6.9 s / 1.8 s** — i.e.
**9× the load time**, which is the number that matters, because that cost is paid on every open.
⚠️ **Size is not the argument and does not hold at small sizes**: on `structured_grid_3d(4,4,4)` the
raw file is *larger* (15 125 B vs 10 419 B), because a toy mesh's indices are one or two characters
against a fixed four bytes. A test asserting "binary is smaller" was written, failed, and was
deleted rather than re-scaled — the property is load time, not bytes.

**Scale it was built for** (same configuration, `structured_grid_3d(118,118,118)` = 1 643 032 cells /
4 970 868 faces, i.e. uvreactor scale): the face stream alone is **50.9 M integers**; reconstruction
15.5 s + write 15.2 s at **3.37 GiB peak RSS** (1.83 GiB of which is the mesh), giving a 347 MiB file
ParaView opens in **0.7 s**. The index arrays are formed at a width chosen once per build by
`mesh.connectivity.index_dtype` — `int32` here — which is what keeps that peak off a second gigabyte.

### ⚠️ A 2D vector is padded on the LAST axis, NOT on the collapsed one — the opposite of the OpenFOAM field writer
`infer_extruded_axis` exists and is right for writing a field back into its original 3D case, whose
coordinates still carry the dropped axis. It is **wrong here**, and calling it would be a plausible
bug: a `.vtu` writes the mesh's *own* two coordinates into the plane `z = 0`, so the geometry has
already been re-planarized. Padding a vector at the original axis would put `U_y` in the `z` slot
while the geometry's `y` sits in `y`. The point and the vector must be padded the same way, and for
the points that way is fixed by the file format.

### Refuses nothing for being non-finite — also the opposite of the OpenFOAM field writer
That writer refuses a `NaN` because a solver reading it back fails somewhere that never names the
file. This one is read by a viewer, and looking at where a solution went non-finite is one of the
things it is for; a viewer draws those cells as blanks.

### Verified by opening the files, not by asserting on the XML produced
Every number below is from `pvpython` inside `/Applications/ParaView-5.12.0.app` (5.12.0),
2026-09-12. **The load-bearing check is `IntegrateVariables` and `CellSize`**: VTK computes a
polyhedron's volume from the face stream by the divergence theorem, so a single inward-wound face
makes it wrong or negative — the topology cannot be confirmed by counting cells.

| case | result |
|---|---|
| `structured_grid_3d(3,2,2)`, raw and ASCII | 12 cells, 36 points, all type 42, integrated volume **1.0000000000000002** (exact domain 1.0) |
| `structured_grid_2d(4,3)` | 12 cells, type 7, integrated area **1.0000000000000002**; `U` third component 0 |
| pitzDaily (2D, 12 225 cells) | integrated area **0.01451603999974618** vs aquaflux's own cell-volume sum **0.014516039999746174** — 15 significant figures |
| bfs3d (3D OpenFOAM hex, 23 040 cells) | counts and points identical to **ParaView's own OpenFOAM reader**; per-cell volume vs aquaflux max rel dev **1.9e-15**; **zero** negative volumes; cell centres agree with aquaflux to **5.6e-17** |
| `polyDualMesh` of bfs3d (25 891 genuinely polyhedral cells: 6/8/10 faces per cell, 4/5/6 nodes per face) | counts identical to ParaView's OpenFOAM reader; **zero** negative volumes; per-cell volume **ours vs ParaView's own OpenFOAM reader 1.08e-6** |
| `.pvd` of three frames | opens as one dataset, three timesteps, per-step field ranges correct |

⚠️ **On the dual mesh aquaflux's own cell volumes differ from VTK's by up to 4.6 %, and that is NOT a
writer defect** — do not "fix" it. The two VTK paths into the same case (our `.vtu`, and ParaView's
independent OpenFOAM reader) agree to **1.08e-6** per cell, while aquaflux disagrees with *both* by
4.6e-2. A dual mesh has **non-planar faces**, on which a cell's volume is not defined until a face
triangulation is chosen: aquaflux uses the centre-fan decomposition its `CellGeometry` is built on,
VTK uses its own. The bfs3d hex mesh, whose faces are planar, shows 1.9e-15 — which is how the two
questions were separated. The same applies to the 1.9e-4 centroid gap there: ParaView's `CellCenters`
is the mean of a cell's points, not its volume centroid, and the two coincide only on a hex.

### Signature asymmetry with the OpenFOAM field writer is deliberate
`write_vtu(mesh, fields, path)` against `write_openfoam_time(case, time, fields, mesh)`. The two are
not the same contract dressed differently — see the `FieldWriter` entry under **Deferred**, which
also records the one thing they *do* share (`io/cell_fields.as_cell_values`) and why the two
questions have different answers.

## Binding decisions
- **A polyMesh is always 3D; a 2D case is one cell thick between two `empty` patches.** The reader
  builds the faithful 3D mesh, then `collapse_extruded_direction` reduces it to `dim == 2` (drop the
  through-axis, dedup front/back nodes, reduce each side quad to its 2D edge, carry owner/neighbour +
  zones 1:1, re-index surviving patches). No `empty` patches ⇒ return the 3D mesh.
- **Reserved-name collision fails loud.** An OpenFOAM patch literally named `interior`, `boundary`
  or `padding` (`mesh.groups.RESERVED_PATCH_NAMES`, enforced by `FacePatches.from_dict`) raises a
  reader-level `ValueError` naming the patch — no
  silent rename (it would break the round-trip; the original name stays visible in
  `PolyMeshData.patches`).
- **Unlisted boundary faces are legal in a MESH**, not an error: they fall into aquaflux's automatic
  `"boundary"` patch. (A valid polyMesh tiles all boundary faces with patches, so `"boundary"` is
  normally empty — this is only a leniency, not a reinterpretation.) Overlaps / out-of-range patch
  ranges are still rejected by `FacePatches.from_dict`. ⚠️ **Solving on such a mesh is a different
  matter**: every assembler's `build` refuses a boundary map that leaves a boundary face uncovered,
  so those faces need a closure under the name `"boundary"` (#354, `.claude/rules/boundary.md`).
- **ASCII only (first cut).** `format binary;` → `NotImplementedError` (detected, never misread).
- **A field is placed by INDEX, and the correspondence is CHECKED rather than assumed (binding).**
  OpenFOAM orders faces interior-first, then boundary faces grouped by patch in `boundary`-file
  order, each patch contiguous; `assemble` carries `owner` through unchanged and never renumbers, so
  aquaflux face `i` *is* OpenFOAM face `i`. That is an inherited convention this package cannot
  enforce, and getting it wrong yields a **plausible field rather than an error** — so
  `read_surface_scalar_field` verifies the interior faces really are the leading block and each
  named patch is a contiguous ascending range, and raises naming the mismatch.
  - **A collapsed 2D case cannot be read this way** and is refused by that same guard: the
    `empty`-patch collapse rebuilds the mesh through `from_csr` and renumbers.
  - ⚠️ **`face_patches` carries the automatic `interior` and `boundary` patches, which no field file
    writes.** `interior` holds the interior faces (already the internal block) and must be skipped;
    a non-empty `boundary` means boundary faces no named patch claimed, so there is nowhere to read
    their values from — that raises. Treating every named patch as a boundary patch was a real bug
    here, caught because the placement test encodes each face's own index in its value.
  - **A `uniform X` and a `nonuniform List<scalar>` entry occur in the same file** — a wall's flux
    is written `uniform 0` while an inlet's is a full list — so a reader handling only the list form
    fails on most of the boundary.

## Deferred (additive; no seam changes)
Binary polyMesh; `faceZones`/`pointZones`; `.gz` compression / multi-region cases; **mesh writing**
(a future `MeshWriter` counterpart to `MeshReader` — note the *field* writers are a different thing
and need none, since each writes a mesh it was handed or into a case whose mesh is already on disk);
other formats (Gmsh/VTK/CGNS) as new `MeshReader` subclasses under `io/<format>/`.

**A `FieldWriter` ABC was held open until the second writer existed, and the answer is NO — but the
shared piece was real and IS extracted (`io/cell_fields.py`, 2026-09-12).** Both halves of that
matter, and they are not the same question.

*The contract does not unify.* `write_openfoam_time(case, time, fields, mesh)` writes **N files into
a directory of an existing case**, needs a per-field template on disk for dimensions and boundary
conditions, and uses the mesh only for a length check and the extruded-axis inference;
`write_vtu(mesh, fields, path)` writes **one file** in which the mesh *is* the payload, and needs
nothing but what it is handed. A common `write(mesh, fields, destination)` would make `destination`
mean a filesystem path in one and a time *name inside a case* in the other, and every remaining
keyword (`template_time`, `extruded_axis`, `allow_non_finite` against `binary`) belongs to exactly
one side — the union bundle the Module Review Rubric warns about. Two writers taking a
`{name: array}` mapping is a shared *vocabulary*, not one configuration.

*The formula did.* Both opened with `np.asarray(values, dtype=float)`, a length check against
`n_cells`, and — the tell — a **byte-identical error message**, written independently a week apart.
That is now `io/cell_fields.as_cell_values(name, values, n_cells)`, which both call. ⚠️ **Read this
pair as the general lesson, because it is the shape that hides**: "these two are not the same
abstraction" is a true statement that says *nothing* about whether they share an implementation, and
answering only the abstraction question leaves the duplicate in place looking justified. Ask both.
The duplicate was invisible to `tools/sibling_builders.py` by construction — neither function
constructs a class, so no pair exists for it to report, and its silence here is not evidence.

**`.pvtu`** (parallel pieces, for `parallel/PartitionedMesh`) is deferred with the seam already cut:
`xml.cell_data_arrays` yields the array names and component counts a `.pvtu` declares, with no data
attached, so it reuses that function unchanged.

## Testability seam (satisfied)
- **Parse** — grammar/foamfile on string snippets (`tests/unit/test_foamfile.py`), no files.
- **Assemble** — `assemble` on a hand-built `PolyMeshData` (`tests/support/polymesh.py`
  `two_cube_polymesh_data`; `tests/unit/test_openfoam_assemble.py`), no files.
- **Collapse** — file-free: collapse `structured_grid_3d(nx, ny, 1)` and match
  `structured_grid_2d(nx, ny)` up to renumbering (`tests/unit/test_collapse.py`), plus a hand-built
  periodic slab pinning that the seam's `neighbour_offset` survives the face renumbering (the reader
  emits no offsets, so that path is unreachable from io today).
- **Orchestrate** — end-to-end on committed ASCII fixtures (`tests/fixtures/polymesh_3d_two_cubes`,
  `tests/fixtures/polymesh_2d_slab`), cross-checked against the structured generators
  (`tests/unit/test_openfoam_reader.py`).
- **Fields** — `tests/unit/test_openfoam_fields.py`: the parser on snippets, and placement end to
  end on the two-cube fixture with a field whose **each value encodes its own face index**, so any
  permutation shows up as a mismatch rather than as a plausible field. The ordering guard is tested
  by feeding it a *generated* grid, which genuinely is not interior-first (its `left` patch occupies
  faces 0–2 while the interior starts at 3) — a standing counter-example, not a contrived one.

## The interior placement is MEASURED, not only argued (bfs3d, 2026-08-17)

The unit tests pin the *structure* (interior block leads, patches contiguous, lengths agree) on a
two-cube fixture. Whether values land on the right faces **within** the interior block needs a real
mesh's connectivity, so it is measured by `validation/bfs3d_openfoam/phi_placement.py` — kept in the
repository precisely so the question can be re-asked rather than only cited.

**Measured on** `validation/bfs3d_openfoam/of_case`, the steady `kOmegaSST` run's `2000/phi`; 23040
cells, 71872 faces (66368 internal + 5504 boundary), 3D, uncollapsed. No solver defaults are involved
— this is a mesh build and two scatters, so it does not expire when a solver default moves.

- **Boundary placement.** Inlet net flux is exactly `-4.000000e-03` m³/s = `U_in x A` = `10 x (0.01 x
  0.04)`; all three wall patches are exactly zero; net imbalance `1.15e-06` (`2.9e-04` of throughput)
  is OpenFOAM's own continuity error.
- **Interior placement.** The conservative scatter of `phi` on aquaflux's connectivity gives a max
  per-cell imbalance of `1.96e-08` m³/s = **`4.9e-06` of the domain flow rate** — the reference's own
  convergence level. A seeded permutation of the interior block (the mutation control, and the whole
  point) gives `2.3e-02`, **4.7e+03x worse**. Placement CONFIRMED.
- ⚠️ **Do NOT normalize a continuity error by the cell's own throughput.** It looks like the natural
  measure and it is a trap: in the recirculation and side-wall corners a cell's throughput falls to
  ~2% of median, so the reference's fixed absolute error divides up into a 2.2e-02 "relative" error
  that reads as a failure. Checked, not assumed — the worst-ratio cells carry `3–9e-09` absolute,
  *below* the global max of `1.96e-08`, so it is entirely the denominator. Normalize by the domain
  flow rate; report the local ratio only as a distribution, whose *spread* is what would reveal a
  genuine mis-placement (local, hence a cluster) as opposed to diffuse reference error.
