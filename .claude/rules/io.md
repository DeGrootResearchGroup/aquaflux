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
    `parse_face_list` / `parse_boundary` / `parse_cell_zones`), sharing one `_list_envelope` for the
    `N ( … )` frame + count-check. **Deliberately not a Strategy hierarchy** — the file kind is
    known statically at every call site, so parser-polymorphism would vary over nothing.
  - `assembler.py` — `assemble(PolyMeshData) -> Mesh` (pure, file-free): pad the interior-only
    neighbour with the `-1` sentinel (relies on OpenFOAM's upper-triangular ordering — interior
    faces first), derive `n_cells = max(owner, neighbour) + 1`, map boundary patches →
    `face_patches` and cellZones → `cell_zones`, then `Mesh.from_csr`.
  - `reader.py` — `OpenFOAMReader(MeshReader)` + `read_openfoam(path)`. The **only** file I/O
    (`_read_field` centralizes missing-file + binary handling). `read()` = assemble the faithful 3D
    mesh, then collapse when `empty` patches are present. Accepts a case dir (resolves
    `constant/polyMesh`) or the polyMesh dir directly.

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

## Deferred (additive; no seam changes)
Binary polyMesh; `faceZones`/`pointZones`; `.gz` compression / multi-region cases; **mesh writing**
(a future `MeshWriter` counterpart to `MeshReader`); other formats (Gmsh/VTK/CGNS) as new
`MeshReader` subclasses under `io/<format>/`.

## Testability seam (satisfied)
- **Parse** — grammar/foamfile on string snippets (`tests/unit/test_foamfile.py`), no files.
- **Assemble** — `assemble` on a hand-built `PolyMeshData` (`tests/support/polymesh.py`
  `two_cube_polymesh_data`; `tests/unit/test_openfoam_assemble.py`), no files.
- **Collapse** — file-free: collapse `structured_grid_3d(nx, ny, 1)` and match
  `structured_grid_2d(nx, ny)` up to renumbering (`tests/unit/test_collapse.py`).
- **Orchestrate** — end-to-end on committed ASCII fixtures (`tests/fixtures/polymesh_3d_two_cubes`,
  `tests/fixtures/polymesh_2d_slab`), cross-checked against the structured generators
  (`tests/unit/test_openfoam_reader.py`).
