---
paths:
  - "aquaflux/io/**"
---

# Rules — `aquaflux/io/` (mesh import/export, and CAD import)

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
    the assembler, not the parser). `FoamPatch.neighbour_patch` carries a `cyclic` patch's
    `neighbourPatch` entry (empty for every other patch type). `patch_face_range(patch)` is the one
    place `[start_face, start_face + n_faces)` becomes an index array — shared by the assembler's
    patch-naming step and `cyclic.py`'s fusion, so both read "this patch's faces" the same way.
  - `foamfile.py` — the shared file envelope: strip `/* */` + `//` comments, split the
    `FoamFile { … }` header dict from the body, and `is_binary` (gates ASCII vs binary in **one**
    place).
  - `grammar.py` — the body-grammar **free functions** (`parse_vector_list` / `parse_scalar_list` /
    `parse_face_list` / `parse_boundary` / `parse_cell_zones`), sharing one `list_envelope` for the
    `N ( … )` frame + count-check. **Deliberately not a Strategy hierarchy** — the file kind is
    known statically at every call site, so parser-polymorphism would vary over nothing.
  - `cyclic.py` — `fuse_cyclic_patches(...)` (pure, file-free): turns a matched `cyclic` patch pair
    into interior periodic seam faces before the assembler ever calls `Mesh.from_csr`, so a
    streamwise-periodic OpenFOAM mesh imports with its periodicity intact instead of as two
    disjoint open boundaries. See **Cyclic-patch fusion** below.
  - `assembler.py` — `assemble(PolyMeshData) -> Mesh` (pure, file-free): pad the interior-only
    neighbour with the `-1` sentinel (relies on OpenFOAM's upper-triangular ordering — interior
    faces first), derive `n_cells = max(owner, neighbour) + 1`, fuse `cyclic` patch pairs
    (`cyclic.fuse_cyclic_patches`), map the surviving boundary patches → `face_patches` and
    cellZones → `cell_zones`, then `Mesh.from_csr`.
  - `reader.py` — `OpenFOAMReader(MeshReader)` + `read_openfoam(path)`. `read()` = assemble the
    faithful 3D mesh (cyclic patches already fused), then collapse when `empty` patches are
    present. Accepts a case dir (resolves `constant/polyMesh`) or the polyMesh dir directly.
    `cyclic_match_tolerance` (constructor / function keyword, default
    `cyclic.DEFAULT_MATCH_TOLERANCE`) passes through to `assemble` for a mesh whose cyclic faces
    do not match to the default tolerance. `_read_field` handles the *optional*-file case and
    delegates the rest to `foamfile.read_foam_body`.
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
    **`read_surface_scalar_field` refuses a periodic mesh outright** (`face_cells.neighbour_offset
    is not None`) rather than relying only on the leading-interior-block check below: a fused
    `cyclic` seam face is interior but sat in whichever boundary patch declared it in the original
    file, and — since a patch can be declared immediately after the true interior block — can
    coincidentally still *pass* that positional check while reading the wrong file entries. The
    direct check catches every periodic mesh regardless of where its seam patch happened to sit.

## Structure — BUILT (cyclic-patch fusion) — 2026-09-14

A `cyclic` patch pair (`neighbourPatch` entries naming each other) describes one periodic seam
split across two boundary patches, not two ordinary open boundaries. Left unfused, `assemble`
would give both patches `neighbour = -1` and the mesh would lose its periodicity entirely — this
is what blocked importing an OpenFOAM periodic channel/duct mesh before this was built. `cyclic.py`
fixes it ahead of `Mesh.from_csr`: it turns the matched pair into interior faces carrying the
`neighbour_offset` periodic-image mechanism `structured_grid_2d(periodic=...)` already uses (see
`.claude/rules/mesh.md`), so the two mechanisms converge on one representation regardless of
whether the periodic mesh came from a generator or a file.

**The match is geometric, not declared.** Rather than parse and trust OpenFOAM's `transform`
keyword (`translational` / `rotational` / `noOrdering`), `fuse_cyclic_patches` always attempts a
translational match: estimate the seam's translation from the two patches' centroid means, shift
one side by it, and nearest-neighbour-match centroids (`scipy.spatial.cKDTree`, robust to face
ordering — OpenFOAM does not guarantee the two patches list corresponding faces in the same
order). This one mechanism both performs the fusion and verifies the pair genuinely *is* a pure
translation: a rotational or mismatched pair simply fails to match within tolerance
(`DEFAULT_MATCH_TOLERANCE`, a fraction of the kept patch's bounding extent, overridable via
`cyclic_match_tolerance` on `assemble` / `OpenFOAMReader` / `read_openfoam`), so no separate
`transform`-type check is needed to reach the same guarantee. **Rotational cyclic patches are
therefore not supported** (`neighbour_offset` is a translation only) — they read in as an error
naming the pair, not silently wrong geometry.

**The patch declared earlier in the boundary file is the "kept" side**; its faces stay in place
(now interior) and the other's faces are dropped as duplicates — the offset derivation
(`kept_centroid - donor_centroid`, added to the donor cell's own centroid to give its periodic
image) is symmetric under this choice, so declaration order is only a deterministic tie-break, not
a physical distinction. Both patches — kept and donor — are removed from `face_patches`: a fused
seam is interior, not a named boundary.

**Fusion pre-empts, rather than interacts with, the 2D collapse.** A cyclic pair is fused inside
`assemble` before `Mesh.from_csr` is ever called, so
by the time `reader.py` detects `empty` patches and calls `collapse_extruded_direction`, the
periodic seam is already an ordinary interior face carrying `neighbour_offset` — the collapse's
existing `gather_neighbour_offset` carry-through (`.claude/rules/mesh.md`) needs no cyclic-specific
handling. A cyclic axis and the collapsed (extruded) axis are necessarily different axes — an
extruded axis's caps are the very `empty` patches the collapse removes, and a periodic axis has no
such caps.

**Deferred, additively:** `cyclicAMI` (a different patch type, for a non-matching interface — not
`cyclic`) and rotational `cyclic` patches; both are simply left as ordinary (disconnected) boundary
patches today, since `fuse_cyclic_patches` only ever acts on a `type_ == "cyclic"` pair.

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
  order, each patch contiguous; on an ordinary (non-periodic) mesh `assemble` carries `owner`
  through unchanged and never renumbers, so aquaflux face `i` *is* OpenFOAM face `i`. That is an
  inherited convention this package cannot enforce, and getting it wrong yields a **plausible field
  rather than an error** — so `read_surface_scalar_field` verifies the interior faces really are
  the leading block and each named patch is a contiguous ascending range, and raises naming the
  mismatch.
  - **A collapsed 2D case cannot be read this way** and is refused by that same guard: the
    `empty`-patch collapse rebuilds the mesh through `from_csr` and renumbers.
  - **Neither can a periodic mesh** — a fused `cyclic` patch pair does renumber (the donor side's
    faces are dropped), and its seam face sat in a boundary patch's block in the original file. This
    is refused by a direct `neighbour_offset is not None` check rather than the leading-block guard
    above, because a seam face declared immediately after the true interior block can coincidentally
    still pass that one.
  - ⚠️ **`face_patches` carries the automatic `interior` and `boundary` patches, which no field file
    writes.** `interior` holds the interior faces (already the internal block) and must be skipped;
    a non-empty `boundary` means boundary faces no named patch claimed, so there is nowhere to read
    their values from — that raises. Treating every named patch as a boundary patch was a real bug
    here, caught because the placement test encodes each face's own index in its value.
  - **A `uniform X` and a `nonuniform List<scalar>` entry occur in the same file** — a wall's flux
    is written `uniform 0` while an inlet's is a full list — so a reader handling only the list form
    fails on most of the boundary.

## Structure — BUILT (CAD import: STEP → exact bodies and emitting triangles) — 2026-09-24, #505

`aquaflux/io/cad/`: `read_step(path, placement) -> CadModel`, and the model hands out
`solid(name)` (one exact `aquaflux.solids` body), `fluid(*names)` (an `Outside` — the vessel as the
water it holds) and `triangles(name, chord=, facet_size=)` (an emitting surface). **`io` now depends
on `solids` as well as `mesh`**; `solids` imports nothing back, and it is generic geometry, not a
physics package (`.claude/rules/solids.md`), which is what makes that direction acceptable.

**Why a library and not a reader of our own (decided with the user, 2026-09-24).** The stated
direction is *arbitrary* CAD — B-spline surfaces, trimmed faces, fillets, booleans, tessellation to a
tolerance, point-in-solid — and a reader for the primitive subset would be dominated the day the
library path existed. Open-source full B-rep kernels that read STEP are, in practice, OpenCASCADE and
things built on it (gmsh, FreeCAD, CadQuery, build123d); the Rust kernels are not mature and a bare
STEP entity parser has no geometry. The binding is **OCP**, published by the CadQuery project as
`cadquery-ocp` — **we use the binding, not CadQuery**. `pythonocc-core` is conda-only; `gmsh` embeds
OCCT but does not hand back surface parameters. ⚠️ Pinned exactly
(`cadquery-ocp-novtk==8.0.1.0.0`, the `cad` extra): two short probes of 8.0.1 hit four API breaks —
`TDF_LabelSequence` → `OCP.collections.Sequence_TDF_Label`, `TopTools_ListOfShape` →
`OCP.collections.List_TopoDS_Shape`, `TopoDS.Solid_s` → `TopoDS.Solid`, and `Bnd_Box.Get()` cannot
convert its return type (use `CornerMin`/`CornerMax`). `-novtk` is the same `OCP` module without VTK
(a further large wheel) and cannot be installed beside CadQuery. No cp310 wheel, so
`requires-python` went to `>=3.11` (CI already ran 3.11/3.12 only). **Unlike `petsc4py` it runs in
CI**: the `test` extra carries `aquaflux[cad]`, and `tests/unit/test_cad_step.py` is in the
optional-dependency census as gated on `OCP`.

**Layout — one module touches the kernel; everything else is plain numpy, testable without it.**
- `kernel.py` — `OpenCascade`, the only `import OCP` in the package. Reads STEP into named, placed
  solids; describes faces as records; builds `solids` bodies back into kernel solids; fuses;
  measures boundary distances; triangulates. No kernel object leaves `CadModel`.
- `faces.py` — `PlaneFace`, `CylinderFace`, `ConeFace`, `SphereFace`, `OtherFace`,
  `SolidDescription`: the facts recognition needs, as numbers.
- `recognize.py` — `RecognitionRule` strategies + `recognize(description, rules)`.
- `placement.py` — `Placement(matrix, offset)`, orthogonal only (a reflection is allowed; a stretch
  would make a cylinder elliptic and is refused; so is a uniform scale — units are the reader's job).
- `model.py` — `CadModel`, `InexactBody`, `read_step` (imports the kernel lazily, so
  `import aquaflux.io` needs no CAD kernel).

**Recognition is two rules, and a proposal is a CLAIM that is checked, not a result.**
- `CurvedPieces` — the solid is the union of the whole-turn convex surfaces it lies inside: each
  cylinder / cone / sphere face with the solid on its axis side becomes that primitive over the
  face's own axial extent. ⚠️ **Which side the solid is on is recorded per face** (`solid_inside`),
  because a pipe wall and a sleeve's bore are the same cylinder read from opposite sides and only the
  first is a convex piece; a whole-turn face with the solid outside is refused as a bore.
  ⚠️ **Faces of one surface are merged before "whole turn" is decided** — exporters split a periodic
  face into halves, reported from different origins with opposite axis signs, so faces are compared
  on a canonical axis (`_Axial`: one direction sign, the foot of the line nearest the origin). A
  partial face that remains is skipped and the proposal is marked `stands_alone=False`: that is the
  **pipe whose end is cut to the curve of the vessel it joins** (the Sozzi riser's end is a 0.43-rad
  patch of the *chamber's* cylinder with the pipe outside it). Its cylinder runs on into the vessel;
  the union with the vessel is exact, the pipe alone is not, so `solid()` refuses it and `fluid()`
  accepts it.
- `ConvexPolyhedron` — an all-planar solid whose every vertex is inside every face's plane is the
  `Intersection` of its faces' `HalfSpace`s. A theorem, so `proven=True` and no check; in a union
  check the drawing's own solid stands in for it (a half-space has no finite kernel counterpart to
  build — mutation-checked: rebuilding it fails `test_a_polyhedron_takes_part_in_a_fluid_as_the_solid_it_is`).
- Anything else — a torus, a spline patch, a non-convex planar solid — raises `UnrecognizedSolid`
  naming every rule's reason. **There is no triangle fallback yet**: #510's triangle-backed occluder
  does not exist, so an unrecognized solid is a documented refusal, which is what #505's acceptance
  allows. When #510 lands, the fallback slots in behind the same refusal.

**⚠️⚠️ THE CHECK IS BOUNDARY DISTANCE, NOT VOLUME — AND THE VOLUME VERSION WAS BUILT FIRST AND FAILED
ON A CORRECT FILE.** Sample points on both boundaries (the union of the proposed bodies and the
union of the drawing's solids; mesh nodes after cutting to `sample_spacing`, a sixteenth of the extent
by default) and require every point within the drawing's **own declared tolerance**
(`ShapeAnalysis_ShapeTolerance`, max over edges/vertices/faces) of the other boundary, in both
directions. Why not the obvious symmetric-difference volume, measured on the Sozzi drawing:
- The drawing's `outlet_pipe` is **1.48e-5 below its analytic volume**, and that is geometry, not
  integration — OCCT's adaptive volume integration reports an error estimate of 3.8e-14 at
  `eps` 1e-6/1e-9/1e-12 and the gap does not move. The saddle edge is a B-spline fitted to the
  drawing's **declared 10 µm** tolerance (Onshape's export precision). The recognized primitives,
  meanwhile, match the analytic union to **7e-10**. So the proposal was closer to the true shape than
  the file was, and a 1e-7 relative volume tolerance refused it (measured 1.51e-7 → then 2.2e-7).
- Loosening the volume tolerance to the file's precision (area × 10 µm = 3.6 cm³ here) would pass
  a missing baffle. A volume test cannot be set: either it fails a correct file or it waves a thin
  feature through.
- The distance sees a thin feature regardless of its volume, because every face is sampled at least
  at its own corners. Two configurations, not to be mixed:
  - **Shipped defaults** (`sample_spacing` = extent/16 ≈ 0.056 m, chord = spacing; drawing
    tolerance 1e-5): true fluid **7.5e-7 m**, true lamp **2.1e-6 m** — the numbers
    `CadModel.discrepancy` reports.
  - **The decoy probe** (spacing 0.05 m, chord 1e-4, a scratch harness): true fluid **1.66e-6 m**,
    true lamp **2.33e-6 m**; riser radius −1% **9.55e-5 m** (caught), riser 0.5 mm short **5.0e-4 m**
    (caught), riser radius **−0.1% → 9.6e-6 m, NOT caught** — below what a drawing exported at 10 µm
    can resolve. That limit is the honest one: nothing can be compared to a drawing more finely than
    the drawing was exported. The −1% case is pinned by `test_a_proposal_that_is_wrong_by_a_percent_is_refused`.
- ⚠️ **Both directions are load-bearing, each with its own test** (mutation-checked): a phantom
  flange on the proposal is seen only from the proposal's side
  (`test_a_phantom_flange_on_the_proposal_is_refused`), a thin fin on the drawing that the rule
  misses only from the drawing's side (`test_a_thin_fin_the_rules_do_not_see_is_refused`).
- Cost: 0.5–0.8 s for the Sozzi vessel (BRepExtrema point-to-shape distance ≈ 0.1 ms/point). ⚠️
  Sampling with a 1e-6 chord took > 10 min — millions of points — and was killed; the spacing, not
  the chord, is what should set the sample count.

**Three kernel details that are load-bearing, each found on the real file:**
1. **Units: `XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 1.0)` before transfer.** Otherwise every
   length comes out in **millimetres** whatever the file declares (the Sozzi file is in metres), and
   setting the `xstep.cascade.unit` static does *not* reach the XCAF reader. Pinned by the Sozzi
   dimensions test and by a synthetic file written in millimetres.
2. **`solid_inside` from face orientation, not from classifying a point.** The natural surface
   normal flipped when the face is `REVERSED` points out of a valid solid everywhere on the surface;
   a classifier needs a point certainly on the face, which a face with a hole does not give at its
   parameter midpoint.
3. **Never recognize from edge types.** The adaptor reports straight seam lines as
   `GeomAbs_BSplineCurve` although the Sozzi file holds a single `B_SPLINE_CURVE_WITH_KNOTS`.

**Assemblies**: locations are accumulated down the component tree (`location.Multiplied(...)`), and a
part's own name is preferred to its instance's; a repeated name gets `[k]`, several solids under one
name get `.k`. Pinned by a two-instance synthetic assembly (mutation: dropping the location or the
suffix goes red).

**Triangulation: facet SIZE is bounded by cutting the boundary SHELLS with a grid of planes, and the
first two ways of doing it were wrong.**
- OCCT's mesher bounds chord and angle only (`IMeshTools_Parameters` has `MinSize`, no maximum), so a
  straight cylinder comes out as full-length slivers: the Sozzi lamp's cylinder was 100 triangles
  0.8 m long and ~0.9 mm wide.
- ⚠️ **Longest-edge (Rivara) bisection afterwards is the wrong tool on such input and was deleted.**
  On slivers all three edges are within ~1e-7 of each other, "the longest" is arbitrary, and the
  conformity closure splits three ways per pass: **148,467** triangles at 20 mm against ~18k needed,
  median aspect 318. Marking only each triangle's longest edge (classic Rivara) changed not one count.
- ⚠️ `ShapeUpgrade_ShapeDivideArea` (by area) left patches long and thin (median aspect 25–79, 10 s
  at 2.5 mm); `ShapeUpgrade_FaceDivideArea` by number did nothing (it gates on an area threshold OCP
  exposes only as a reference).
- **What works**: `BRepAlgoAPI_Splitter` of the solid's **shells** (never the solid — splitting the
  volume meshes the cutting planes' interior faces too: area **+175%**) by three families of planes
  `facet_size` apart, then mesh. Every patch fits in a `facet_size` cube, so edges ≤ `√3·facet_size`
  and in practice ≈ `facet_size`: lamp at chord 1e-5 → 11,002 / 22,271 / 40,085 / 86,315 triangles at
  20 / 10 / 5 / 2.5 mm, longest edge 19.8 / 9.9 / 5.7 / 3.1 mm, area −1.8e-4…−1.4e-4, 0.07–6 s.
  Remaining aspect (31 → 4.5) is `facet_size` over the chord-set circumferential spacing, which the
  gather does not care about (its solid angle is exact at any shape).
- **Zero-area triangles are dropped** — the mesher emits one or two at a pole, and a zero-area facet
  is a *point source* to `radiation.Surfaces`, which an emitting surface must not contain by
  accident. Mutation-checked.
- **A reflecting placement keeps triangles outward** — `BRepBuilderAPI_Transform` with a negative
  `gp_Trsf` reverses the faces itself; the Sozzi placement (x↔y) is a reflection and the lamp test
  runs through it (signed volume positive to 2e-3 of the analytic value).

**MEASURED (2026-09-24): the fluid read from the drawing shadows the Sozzi reactor exactly as the
hand-derived occluder does** (`validation/sozzi_radiation/primitive_occlusion.py`, third arm): **0 of
180,384,000 pairs** masked differently and `G` equal to 0.0 relative at median, p99 and max, on
24,000 cells drawn from the meshed case's 1,635,909 (19,437 chamber / 2,214 inlet / 2,433 riser, after
the hand-typed pipes' truncation was fixed — see `.claude/rules/radiation.md`; the first run's population
was 22,250 / 640 / 1,200 and also gave 0), the
case's `lampWall.stl` (7,516 facets), exitance 696.42 W/m², absorption 35.67 /m; jax 0.10.2, CPU, x64,
macOS arm64, 11 cores, `cadquery-ocp-novtk` 8.0.1.0.0, CPython 3.13. The drawing's cylinders differ
from the hand-typed ones in *parameters* (the riser carried into the chamber by recognition rather than
by `REACH_BACK`), and the mask on the mesh's receivers is identical. ⚠️ An earlier version of this
entry said the meshed domain cut the pipes at x = 1.10 / z = 0.40; that was the hand-typed pipes' cut,
not the mesh's, which runs the full 850 mm. Read-and-check took 0.8 s. The three arms' single-pass
timings (8.7 / 12.8 / 11.0 s) are one pass each and are **not** a ratio to quote (#513).

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
- **Parse** — grammar/foamfile on string snippets (`tests/unit/test_foamfile.py`), no files;
  including `parse_boundary` reading a `cyclic` patch's `neighbourPatch` entry.
- **Assemble** — `assemble` on a hand-built `PolyMeshData` (`tests/support/polymesh.py`
  `two_cube_polymesh_data`; `tests/unit/test_openfoam_assemble.py`), no files.
- **Cyclic fusion** — `tests/unit/test_openfoam_cyclic.py`, file-free on hand-built cyclic-patch
  fixtures (`tests/support/polymesh.py` `cyclic_two_cube_polymesh_data` /
  `cyclic_slab_polymesh_data`): the fused seam's owner/neighbour/`neighbour_offset`, every
  pairing error path (missing/unknown/self-referencing/asymmetric `neighbourPatch`, a partner that
  is not itself `cyclic`, a face-count mismatch, a non-translational pair), and a fused-then-
  collapsed slab cross-checked against `structured_grid_2d(periodic=("x",))` — the independent
  oracle for the periodic connectivity a real OpenFOAM cyclic mesh should read in as.
- **Collapse** — file-free: collapse `structured_grid_3d(nx, ny, 1)` and match
  `structured_grid_2d(nx, ny)` up to renumbering (`tests/unit/test_collapse.py`), plus a hand-built
  periodic slab pinning that the seam's `neighbour_offset` survives the face renumbering. (The
  cyclic-fusion + collapse cross-check above is the reader-reachable version of this same property;
  this one stays as the collapse transform's own file-free regression test.)
- **Orchestrate** — end-to-end on committed ASCII fixtures (`tests/fixtures/polymesh_3d_two_cubes`,
  `tests/fixtures/polymesh_2d_slab`), cross-checked against the structured generators
  (`tests/unit/test_openfoam_reader.py`).
- **Fields** — `tests/unit/test_openfoam_fields.py`: the parser on snippets, and placement end to
  end on the two-cube fixture with a field whose **each value encodes its own face index**, so any
  permutation shows up as a mismatch rather than as a plausible field. The ordering guard is tested
  by feeding it a *generated* grid, which genuinely is not interior-first (its `left` patch occupies
  faces 0–2 while the interior starts at 3) — a standing counter-example, not a contrived one. A
  separate test pins that a periodic mesh (`structured_grid_2d(periodic=("x",))`, standing in for a
  fused cyclic mesh) is refused directly, not by that same positional check. The committed two-cube
  fixture's `boundary` file happens to declare its patches in the same order their faces occupy, so
  a further hand-built case (`two_cube_polymesh_data()._replace(patches=...)`, reordering the
  declared patches without moving their face ranges) declares them out of ascending-startFace order
  and checks the same encode-your-own-face-index placement on it — the case that distinguishes
  "laid out by startFace" from "laid out in file/declaration order," which the committed fixture
  alone cannot.

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
