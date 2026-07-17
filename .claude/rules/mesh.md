---
paths:
  - "aquaflux/mesh/**"
---

# Rules — `aquaflux/mesh/`

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors to inform
> *your* understanding — that is its job, and why it loads into your
> context. Per the root `CLAUDE.md` **Comment Convention**, none of that provenance may
> reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never the
> reference code, the `.claude/` rules, the design notes, or the author's own papers.
>
> **Also binding here (root `CLAUDE.md` Self-Contained-Docstring Convention).** Mesh code
> leans hard on ragged **compressed-sparse-row (CSR)** storage — spell that acronym out at
> first use in each file, then use "CSR". And never let our working-conversation scaffolding
> reach a docstring: no build-stage labels ("Milestone 0", "Gate B") and no unpinned
> performance numbers ("million-cell meshes", "builds in seconds") — describe the mechanism
> instead (e.g. "avoids the per-face Python loop").

Static mesh: connectivity, node coordinates, and derived face/cell geometry. This is
the substrate the residual assembly scatters over. Governed by the Engineering
Principles in the root `CLAUDE.md`.

## Module layout (mirrors the C++ entity split — do not merge into one file)
All classes are `equinox.Module`s (fully OO, per CLAUDE Principle 1).
- `mesh.py` — the `Mesh` container. Stores `node_coords` + two **connectivity objects**
  (`face_cells`: `FaceCellConnectivity` = owner/neighbour; `face_nodes`: `FaceNodeConnectivity`
  = ragged CSR node lists) + `cell_zones`/`face_patches`. **The raw index arrays live *inside*
  those objects** — callers use `mesh.face_cells.owner` / `mesh.face_nodes.offsets` and the
  gather/scatter operators, never a top-level `mesh.owner` (those delegating properties were
  deliberately **removed**: one home per array, in `connectivity.py`). Only the synthesized size
  queries `dim`/`n_cells`/`n_faces`/`n_nodes` remain `Mesh` properties (mesh-level vocabulary, not
  duplicate array views). `from_faces` / `from_csr` constructors + a `geometry()` method that wires
  the face/cell computation in dependency order and returns a **`MeshGeometry`** bundle (see below).
  **`from_csr` is the vectorized constructor**
  (already-assembled CSR arrays, avoiding the per-face Python loop) — `from_faces` flattens its
  ragged input and delegates to it, so the zone/patch build + `validate()` live in one place, and
  generators that know the connectivity as arrays (`structured_grid_2d`/`_3d`) call it directly.
- `geometry.py` — **`MeshGeometry`**: the bundled `{face: FaceGeometry, cell: CellGeometry}` product
  `Mesh.geometry()` returns (was a bare 2-tuple). It travels together through every consumer
  (`ResidualAssembler`, the gradient/interpolation/limiter schemes, `MomentumContinuity`, the
  preconditioner Schur kernels), so those signatures take **one `geometry` collaborator**, not a
  loose face+cell pair (Principle 3). **Geometry is *derived on demand*, never stored on `Mesh`**:
  it is a pure function of the differentiable `node_coords`, so a stored sibling leaf would make
  `grad` w.r.t. node positions silently skip the node→geometry dependency, and would go stale under
  `permute_cells` / partitioning. Seams that read **only one** side keep a single-geometry arg
  (smallest sufficient collaborator): `FaceFluxOperator.face_flux(state, face_geometry)`,
  `interior_mass_flux(..., face_geometry, ...)`.
  `PartitionedMesh` gathers a local bundle via `LocalPartition.local_geometry(global_geometry)`.
- `reorder.py` — cell **renumbering**. `permute_cells(mesh, perm)` is the single canonical
  "apply a cell relabelling" transform (remaps owner/neighbour values + zone labels →
  `P·A·Pᵀ`); the `CellReordering` strategy hierarchy chooses the permutation
  (`IdentityReordering`, `ReverseCuthillMcKee` bandwidth-reduction via scipy,
  `RandomReordering` for worst-case robustness tests). Build-time preprocessing only — not
  part of the differentiable solve.
- `graph.py` — the **cell↔cell connectivity graph** (sparsity pattern of the cell operator), in
  both sparse forms a consumer needs: `cell_adjacency_coo(mesh)` → scipy **COO** matrix (for SciPy
  `csgraph` RCM), `cell_adjacency_csr(mesh)` → **CSR** arrays `(offsets, neighbours)` (for a graph
  partitioner; `= cell_adjacency_coo(...).tocsr()`). Both build on `face_cells.interior_edges()`, which
  applies the boundary convention once. This is the mesh-level graph *shared by* RCM renumbering
  (`reorder.py`), graph partitioning (`parallel/partitioner.py`), and aggregation multigrid — so it
  lives here, not under `reorder.py` where RCM (its first caller) happens to sit, nor under
  `parallel/` (partitioner imports `cell_adjacency_csr` from `aquaflux.mesh`, which no longer
  re-exports it from `parallel`). **Build-time only** (eager numpy/scipy, returns a SciPy/numpy
  graph, never traced): call during preprocessing, not inside `jit`/`grad`. The object owns the
  *primitive* (`FaceCellConnectivity.interior_edges`); this module only picks the sparse
  representation each consumer's library wants.
- `structured.py` — vectorized **structured-grid generators** for axis-aligned boxes:
  `structured_grid_2d` (quads) / `structured_grid_3d` (hexes), assembled with numpy (no per-face
  Python loop) via the shared `_FaceFamilyBuilder` and handed to `Mesh.from_csr`. These are clean
  **orthogonal** generators (a rectangular tank/channel without a mesh file) — the *user-facing*
  surface. **Grid skewing is deliberately not here.** The `perturb`/`seed` interior-node
  displacement that breaks a smooth grid's error cancellation for order-of-accuracy studies is a
  *verification* concern, so it lives in the test suite (`tests/support/meshes.py`:
  `perturbed_grid_2d`/`_3d`, thin wrappers that displace the clean grid's interior `node_coords`
  after construction — exactly equivalent to perturbing before, since geometry is derived from
  `node_coords` on demand). Keep the library generators orthogonal-only; add skew in the test
  helper, not by re-adding a `perturb` knob here.
- `collapse.py` — `collapse_extruded_direction(mesh, removed_patch_names)`: the general build-time
  transform that turns a one-cell-thick **extruded 3D** mesh (a 2D case stored as a slab capped by
  two planar patches) into a genuine `dim == 2` `Mesh`. Infers the extruded axis from the two named
  caps, dedups coincident front/back nodes, reduces each extruded side quad to its 2D edge, drops the
  caps, and carries owner/neighbour + `n_cells` + `cell_zones` through 1:1 while re-indexing the
  surviving patches (`eqx.tree_at`-free — rebuilds via `Mesh.from_csr`, which validates). It is a
  **mesh** transform, not an io/OpenFOAM one: the reader (`aquaflux/io/`) only *detects* the `empty`
  patches and calls it, so the collapse is reusable and unit-tested file-free (collapse
  `structured_grid_3d(nx, ny, 1)` → matches `structured_grid_2d(nx, ny)`). Build-time only (eager
  numpy), like `reorder.py`/`graph.py`.
- `face.py` — `FaceGeometry` (result) + the **strategy hierarchy** `FaceGeometryScheme`
  → `EdgeFaceGeometry` (2D) / `PolygonFaceGeometry` (3D), mirroring the C++ `Face<T,2>` /
  `Face<T,3>` specialization; selected by `face_geometry_scheme(dim)`.
- `cell.py` — `CellGeometry` (volume, centroid) with `from_faces(...)` and
  `approx_centroids(...)`; a single class (the C++ `Cell<T,N>` is dimension-generic, not
  specialized), computed by the divergence theorem.
- `groups.py` — `LabelledGroups` base → `CellZones` / `FacePatches`. Named **partitions**
  of cells (zones) and faces (patches) — the SoA analogue of the C++ `MeshObjectGroup`
  `name→group` maps. See `docs/mesh_zones_and_patches.md` for the full design.
- `connectivity.py` — **the storage-and-movement layer: the substrate-wide gather/scatter API**
  (Principle 2). It separates *how node/face/cell values are stored and moved* from *geometry
  math* and *physics*, so those read as math, not index plumbing.
  - `FaceCellConnectivity` (obtained as **`mesh.face_cells`**) — the face→cell relation and the
    operators over it. **Gather is direct indexing** with the public accessors `owner` /
    `safe_neighbour` (`cell_field[fc.owner]`, `cell_field[fc.safe_neighbour]` — the boundary-safe
    substitution lives in the `safe_neighbour` property); that idiom is used throughout, so there are
    no `gather_owner`/`gather_neighbour` wrapper methods. The *scatter* side is where the class earns
    its keep: `scatter(owner_contrib, neighbour_contrib)` (auto-masks the neighbour side on boundary
    faces — the *one* place that masking lives), plus conveniences `scatter_conservative(flux)`
    (owner `+`, neighbour `−`; FVM conservation), `scatter_symmetric(contrib)` (both cells `+`; means
    / symmetric coefficients), and `scatter_max`/`scatter_min` (extremum reductions, boundary side
    masked to ∓∞). This is the operator behind **every** `gather → compute → scatter` residual term —
    the mesh geometry (`cell.py`, `quality.py`), the discretization scatter
    (`ResidualAssembler._scatter` delegates here; flux operators gather owner/neighbour by direct
    indexing on `owner`/`safe_neighbour`), the gradient schemes, and the coupled flow all compose it.
  - `FaceNodeConnectivity` (obtained as **`mesh.face_nodes`**) — the ragged face→node relation:
    `gather_node_coords`, `perimeter_next`, `reduce_to_faces`, `vertex_mean`; the face-geometry
    schemes (`face.py`) traverse a polygon through these instead of open-coding CSR arithmetic.
  - `interior_mask(neighbour)` — the boundary convention (`neighbour < 0` marks a boundary face)
    as the one free-function **primitive** the classes are built on (`FaceCellConnectivity.interior`),
    and the form the numpy build-time paths (`mesh.validate`, `reorder.py`, `parallel/`) and
    gather-only sites (`groups.py` interface detection) use directly on raw arrays where no
    connectivity object is in hand. The owner-substitution for a gather/scatter-safe neighbour index
    is **not** a free function — it is `FaceCellConnectivity.safe_neighbour` (owner-substitution is a
    JAX-class concern; the numpy paths that need a placeholder roll their own `where(interior, nb, 0)`
    and re-mask, a different idiom).
  Leaf module (imports only `jax.numpy` / `numpy` / `segment_sum`), so it is safe to import from
  anywhere including inside the mesh package.

Naming note: this is **not** `metrics.py` — "mesh metrics" means quality measures
(skewness, orthogonality, aspect ratio) in CFD, a different thing. "geometry" is
reserved for the CAD/domain sense, which aquaflux does not model (meshes arrive from a
mesh file). Use "face geometry" / "cell geometry".

## Binding decisions (all formulas replicate the C++ framework exactly)
- **Representation is SoA + integer-index connectivity + `segment_sum`**, not the C++
  pointer graph (the agreed XLA-driven deviation). The *math* is identical; the loop
  becomes a scatter. Owner/neighbour arrays are sufficient — cell↔face incidence is
  implicit, so no CSR list is materialized.
- **Store the primitives, compute the derived vectors inline.** `Mesh`/geometry store
  only area, centroid, normal (face) and volume, centroid (cell) — the expensive
  primitives, computed once at build time. The bulky per-face displacement vectors
  (`x_ip − x_owner`, `x_ip − x_neighbour`) are **not stored** (they cost as much memory
  as several fields); they are cheap gathers formed inline in the operator, exactly as
  the C++ `getFlux` does. DRY is satisfied by defining the displacement formula in **one
  place** (a shared method), not by materializing arrays.
- **Owner-outward normal convention, fixed once at build (decided deviation from C++ —
  do not revisit).** Each face's stored normal points out of its owner cell; the neighbour
  uses `−n`. Orientation is done with the approximate cell centroid (mean of face
  centroids) — the same flip the C++ `Cell` performs, hoisted to build time and applied to
  the *stored* normal. **The C++ instead stores the raw node-order normal and trusts the
  mesh reader to have wound faces owner→neighbour** (the flux uses the raw normal). We
  deliberately chose to make "normal points owner→neighbour" a **container invariant**
  rather than an **input precondition** (the deviation audit's "Deviation B", resolved in
  favour of this option): it makes orientation robust to input node ordering, removes a
  whole class of silent flipped-normal bugs, and produces identical values to a correctly
  wound input. Cost: one build-time step the C++ lacks. Keep it.
- **A neighbour index `< 0` (by convention `−1`) marks a boundary face**; it scatters to
  the owner only.
- **2D and 3D both implemented** as `FaceGeometryScheme` strategies (via `face_geometry_scheme(dim)`):
  - 2D face = edge (`face_nodes` width 2): `area=|d|`, `centroid=midpoint`,
    `normal=(d_y,−d_x)/|d|`.
  - 3D face = polygon, **centre fan** (`PolygonFaceGeometry.centre_fan`): triangles
    `(c, v_i, v_{i+1})` around the perimeter, apex `c` = vertex mean. Vector area
    `S = Σ ½(v_i−c)×(v_{i+1}−c)` → `area=|S|`, `normal=S/|S|` (**unit**),
    `centroid=Σ(d_i·n)c_i/|S|` (**signed** projected-area weights — the unsigned `Σ|d_i|c_i/Σ|d_i|`
    is wrong on a **non-convex** face, where a reflex vertex makes one fan triangle wind against
    `n`; `Σ|d_i|` is kept only for the planarity metric). Normalization is zero-safe (a degenerate
    zero-area face yields a zero normal + finite gradient, never a NaN). **Deliberate deviation from the C++ node-0 fan (decided —
    do not revert):** the C++ used `area=Σ|d_i|` + a *sub-unit* normal that is
    vertex-order-dependent on **non-planar/warped faces** (common in real hex/polyhedral
    meshes). The centre fan gives `|S|` + unit normal for any polygon, vertex-order-
    independent, and generalizes the Fortran's two-diagonal averaging (which recovered the
    same but only for quads). Planar faces are unchanged.
  - Face node-lists are stored **ragged (CSR)**: `face_node_offsets` (n_faces+1) +
    `face_node_indices` (flat). Each face holds exactly its own node count — arbitrary
    polygons, **no padding, no wasted memory** (the array analogue of the C++ per-face
    node vector). The 3D path enumerates each face's perimeter edge-triangles across the
    ragged structure (build-time numpy) and scatters into faces with `segment_sum`. Build
    meshes with `Mesh.from_faces(coords, [[...],...], owner, neighbour, n_cells)`.
  - The cell divergence formula is dimension-general (`/dim`, `/(dim+1)`).

- **Meshes are validated at construction; validity is checkable.** Validation is split by
  ownership: `FaceNodeConnectivity.from_csr` checks the **CSR structure** it owns (row pointers
  1-D, start at 0, non-decreasing, `offsets[-1] == len(indices)`) so malformed CSR fails there,
  not deeper in the traversal build; then `Mesh.validate()` (topology-only, `O(n)`, no geometry,
  called by `from_faces`) does the semantic checks: finite coordinates, integer index dtypes,
  index ranges, per-face node counts (**exactly 2** in 2D, `>= 3` in 3D), no repeated node within a
  face (a zero-area/NaN degeneracy), owner/neighbour ranges (neighbour sentinel is **exactly `-1`**),
  self-neighbour, and **every cell referenced** by a face (an orphan cell would be `0/0`). Both raise
  a clear `ValueError` rather than letting a bad mesh silently produce wrong volumes/NaNs
  (`segment_sum` silently *drops* an out-of-range owner). Two defects are deliberately **out of
  scope** (geometric, not topological): a face whose
  distinct nodes are collinear (zero area) and duplicate faces / unclosed cells — both surfaced by
  the *geometry*-level diagnostics `quality.face_planarity` (→ `0`) and `quality.closed_cell_residual`
  (Σ outward face area-vectors per cell ≈ 0 for a closed mesh; also cross-checks orientation).
  `validate()` must **not** move into `__init__` — that runs on the pytree-unflatten path under `jit`.

- **Named groupings are partitions (`groups.py`) — decided.** `CellZones` (cells) and
  `FacePatches` (faces) each assign **one** label per element (each cell in one zone, each
  face in one patch) — a single `int` array, vectorizable, mirroring how the C++ groups
  were used. Defaults: one `"default"` zone; `"interior"`+`"boundary"` patches. **Zone
  interfaces are derived** (`CellZones.interface_mask` / `interface_mask_between` from the
  zone labels + owner/neighbour) **and** may be named as explicit patches — a **baffle** is
  just a named patch on interior faces. Both coexist (derive by default, name for bespoke
  treatment). Node groups omitted until needed; overlapping groups rejected. Shared code
  lives in the `LabelledGroups` base; `CellZones`/`FacePatches` add type-specific helpers.

- **Warped-face robustness is a measured property (`quality.py`).** `face_planarity(mesh)`
  = `|S|/Σ|d_i|` (1 = planar) is a near-free warp screen; `centroid_iteration_shift(mesh)`
  = `|c2−c1|/√area` (a second centre-fan pass using the first centroid as apex) directly
  measures whether centre iteration would refine the centroid. **Decision: one centre-fan
  pass, no iteration** — even a violently warped quad shows a shift of ~3×10⁻⁴ and realistic
  warp ~10⁻¹¹. Re-check on a representative real mesh with `centroid_iteration_shift` before
  adding iteration; it is a once-per-build cost so thoroughness is cheap.

- **Cell ordering affects AMG convergence through the coarse space — reorder large meshes
  (`reorder.py`), measured.** The smoothed-aggregation V-cycle *smoother* (Jacobi/Chebyshev)
  and its direct coarse solve are exactly permutation-invariant, so a symmetric renumbering
  `P·A·Pᵀ` leaves the *spectrum* untouched. But the **greedy aggregation visits cells in index
  order**, so the *coarse space it builds* is ordering-dependent: on a model Poisson a
  spatially-local numbering (natural or RCM) gives compact aggregates and a V-cycle contraction
  factor ~0.25, while a **scrambled numbering nearly doubles it to ~0.45** (measured; RCM of the
  scramble restores ~0.28). Consequence: the aggregation-AMG development was done on the
  already-well-ordered structured grid, so a real mesh arriving in arbitrary order must be
  **RCM-reordered before the preconditioner is built** — reordering protects convergence, not
  just matvec locality. It is *not* required for correctness (a bad ordering only slows the
  inner rate, never breaks it; at small meshes the block-triangular outer count is unchanged),
  which is why it lives here as a build-time preprocessing step, not in the solve. Guarded by
  `tests/unit/test_multigrid.py::test_ordering_affects_coarse_space_and_rcm_restores_it` and the
  flow-level robustness test. (This differs from the classical reason bandwidth reordering
  matters — Gauss–Seidel / ILU factor ordering — which does not apply to our order-invariant
  smoother.)

## Terminology alignment with the papers (D / fip / R are renamed, not gone)
The C++ and DeGroot 2018/2019 use the **same** non-orthogonal diffusion formulation:
- The papers' `D_P,ip` / `D_nb,ip` (cell-centroid → integration-point displacements)
  **are** `x_ip − x_owner` / `x_ip − x_neighbour` — formed inline in the diffusion
  operator, not stored in the mesh.
- The **skewness / non-orthogonal correction** is the tangential projection
  `dxip − (dxip·n) n` (the role the Fortran's `R` plays), computed inline.
- The **inverse-distance interpolation factor `f`** belongs to the *gradient*
  reconstruction (`schemes/`), not the base mesh.
So the mesh provides primitives; the diffusion operator forms `D_P,ip`/`D_nb,ip` and the
tangential correction; the gradient scheme carries `f`.

## Testability seam (satisfied)
Geometry is computed from hand-built connectivity (no reader), so every formula is unit-
tested against analytic values on tiny meshes (`tests/unit/test_mesh.py`): 2D unit
squares, triangle, parallelogram; 3D unit cube, tetrahedron (V=1/6), two adjacent cubes
(interior-face scatter), triangular prism (mixed tri/quad faces = padding path), and a
non-unity box [0,2]x[0,3]x[0,4] with distinct side lengths (V=24) that catches
scale-factor and axis-transposition errors a unit cube would hide. Owner-outward
orientation is proven robust to node ordering.

## Golden reference
C++ `Face_impl.hpp` (2D area/centroid/normal), `Cell_impl.hpp` (divergence-theorem
volume/centroid + approx-centroid orientation), `FaceConnection.hpp` (owner/neighbour,
null-`c1` = boundary). Cite the math in comments, not the reference files (CLAUDE
Comment Convention).
