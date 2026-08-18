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

Reading external mesh formats into an aquaflux `Mesh` (and, later, writing them out). This package
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
  - **`foamfile.read_foam_body(path)` is the one place a file on disk becomes a parseable body**,
    so the ASCII-only limitation is enforced once. It lives beside the `is_binary` predicate it
    uses. **`reader.py` is NOT the package's only file I/O** — `fields.py` reads files too; what is
    centralized is the *binary gate*, and its home is `read_foam_body`.
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

## Binding decisions
- **A polyMesh is always 3D; a 2D case is one cell thick between two `empty` patches.** The reader
  builds the faithful 3D mesh, then `collapse_extruded_direction` reduces it to `dim == 2` (drop the
  through-axis, dedup front/back nodes, reduce each side quad to its 2D edge, carry owner/neighbour +
  zones 1:1, re-index surviving patches). No `empty` patches ⇒ return the 3D mesh.
- **Reserved-name collision fails loud.** An OpenFOAM patch literally named `interior`/`boundary`
  (reserved by `FacePatches.from_dict`) raises a reader-level `ValueError` naming the patch — no
  silent rename (it would break the round-trip; the original name stays visible in
  `PolyMeshData.patches`).
- **Unlisted boundary faces are legal**, not an error: they fall into aquaflux's automatic
  `"boundary"` patch. (A valid polyMesh tiles all boundary faces with patches, so `"boundary"` is
  normally empty — this is only a leniency, not a reinterpretation.) Overlaps / out-of-range patch
  ranges are still rejected by `FacePatches.from_dict`.
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
(a future `MeshWriter` counterpart to `MeshReader`); other formats (Gmsh/VTK/CGNS) as new
`MeshReader` subclasses under `io/<format>/`.

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
