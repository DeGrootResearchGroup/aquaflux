---
paths:
  - "aquaflux/mesh/**"
---

# Rules — `aquaflux/mesh/`

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors; none of that may
> reach the shipped surface (`.py`, `docs/`) — root `CLAUDE.md` Comment Convention. Also binding here:
> spell out **compressed-sparse-row (CSR)** at first use per file, and no build-stage labels or unpinned
> performance numbers in docstrings (describe the mechanism instead).

Static mesh: connectivity, node coordinates, derived face/cell geometry — the substrate the residual
scatters over. Dated measurements and fix narratives: `.claude/notes/mesh-log.md`.

## Module layout (mirrors the C++ entity split — do not merge into one file)
All classes are `equinox.Module`s.

- `mesh.py` — `Mesh`: `node_coords` + `face_cells` (`FaceCellConnectivity`) + `face_nodes`
  (`FaceNodeConnectivity`, ragged CSR) + `cell_zones`/`face_patches`. **Raw index arrays live inside the
  connectivity objects** (`mesh.face_cells.owner`, never a `mesh.owner` forwarding property); only the size
  queries `dim`/`n_cells`/`n_faces`/`n_nodes` are `Mesh` properties.
  - `from_csr` is the vectorized constructor; `from_faces` flattens and delegates to it, so zone/patch build
    and `validate()` live in one place. `from_csr` narrows `owner`/`neighbour` to `index_dtype(n_cells)`
    **only when already integer-typed**, so a non-integer input still reaches `validate()`'s dtype check.
  - `geometry()` returns a `MeshGeometry` and runs its arithmetic as one `equinox.filter_jit` call
    (`_fused_geometry`) — eager dispatch materialized a full buffer per step (roughly halved peak memory
    at scale). The perimeter traversal is still enumerated eagerly once in `FaceNodeConnectivity.from_csr`
    (data-dependent counts).
- `geometry.py` — `MeshGeometry` (`{face: FaceGeometry, cell: CellGeometry}`), passed as one `geometry`
  collaborator; seams reading one side take only that side (`FaceFluxOperator.face_flux(state,
  face_geometry)`). **Geometry is derived on demand, never stored on `Mesh`**: it is a pure function of the
  differentiable `node_coords`, so a stored leaf would break `grad` w.r.t. node positions and go stale
  under `permute_cells`/partitioning. `LocalPartition.local_geometry(global_geometry)` gathers a local bundle.
- `face.py` — `FaceGeometry` + `FaceGeometryScheme` → `EdgeFaceGeometry` (2D) / `PolygonFaceGeometry`
  (3D), selected by `face_geometry_scheme(dim)`. `PolygonFaceGeometry.fan_triangles` is the one home of
  the fan corners (`centre_fan`, `warp_first_moment`, `surface.py` build on it).
- `cell.py` — `CellGeometry` (volume, centroid) via `from_faces(...)` (divergence theorem,
  dimension-general) and `approx_centroids(...)`.
  - ⚠️ **`from_faces` accumulates relative to each cell's own `approx_centroid`, then shifts back** — a
    conditioning fix: in global coordinates a small cell far from the origin cancels catastrophically and
    its centroid can land outside the cell (a real snappyHexMesh mesh flipped `D_P·n` negative and crashed
    the AMG positivity guard). Owner and neighbour each use their own reference. Pinned by
    `tests/unit/test_mesh.py::test_cell_centroid_accurate_far_from_origin`.
- `connectivity.py` — the storage-and-movement layer (leaf module; imports only `jax.numpy`/`numpy`/
  `segment_sum`):
  - **`FaceCellConnectivity` (`mesh.face_cells`).** Gather is direct indexing with `owner` /
    `safe_neighbour` (boundary-safe); no `gather_*` wrappers. Scatters: `scatter(owner_contrib,
    neighbour_contrib)`, `scatter_to_owner`, `scatter_conservative` (owner `+`, neighbour `−`),
    `scatter_symmetric`, `scatter_max`/`scatter_min`. A boundary face's neighbour-side index is redirected
    to a trash row (`num_segments = n_cells + 1`, then `[:n_cells]`), so no `jnp.where` mask or identity
    element is needed. `interior_edges()` applies the boundary convention once for graph builders.
  - **`interior`, `safe_neighbour` and the trash index are stored fields computed in `__post_init__`**
    (a recomputed property re-runs inside traced `while_loop` bodies). ⚠️ So they go **stale under pytree
    surgery that skips `__init__`** (`eqx.tree_at` on `owner`/`neighbour`; `__check_init__` does not help).
    Change topology only through **`with_topology(owner, neighbour)`** (used by `permute_cells`); never
    patch `owner`/`neighbour` directly. `n_faces` is derived (`owner.shape[0]`); `n_cells` is stored (not
    recoverable). `FaceNodeConnectivity` stores its own static `n_faces`.
  - **Streamwise-periodic seams** carry a per-face `neighbour_offset` (`None` on an ordinary mesh).
    Displacements gather the neighbour centroid via `fc.neighbour_centroid(cc)`
    (`cc[safe_neighbour] + neighbour_offset`); values are periodic and gather unchanged — only positions
    take the offset.
    - ⚠️ **Every face-renumbering transform must carry it via `gather_neighbour_offset(face_index)`**
      (`face_index[new] = old`; negative → zero; returns `None` when non-periodic). Dropping it raises
      nothing and silently un-periodises the seam. All four do, each regression-tested: `permute_cells`,
      `partition_mesh`, `pad_partition`, `collapse_extruded_direction` (projected onto kept axes). The
      polyMesh reader fuses `cyclic` pairs before collapsing (`io/openfoam/cyclic.py`), so the collapse path
      is reachable in production. Fixtures: `tests/unit/test_collapse.py::_periodic_extruded_slab`,
      `tests/support/polymesh.py` (`cyclic_slab_polymesh_data`, `cyclic_two_cube_polymesh_data`).
    - ⚠️ **A seam face's node ring belongs to the owner's side.** Listing it for the neighbour cell needs
      a `−neighbour_offset` translation (`io/vtk/topology._onto_own_side`; `io.md`).
  - **`FaceNodeConnectivity` (`mesh.face_nodes`)**: `gather_node_coords`, `perimeter_next`,
    `reduce_to_faces`, `vertex_mean`, `face_of_incidence` (reused by `validate()`), `select(faces)` (a
    subset's connectivity in the given order, node indices unchanged), `edges()` (each edge once as an
    ascending node pair, rows sorted, and the edge each incidence starts; integer keys, not a row-wise
    `np.unique`; pinned against Euler's formula in `tests/unit/test_mesh_edges.py`).
  - `interior_mask(neighbour)` is the one free-function boundary convention (`neighbour < 0`), for numpy
    build-time paths with no connectivity object. Owner substitution is not a free function — it is
    `FaceCellConnectivity.safe_neighbour`.
- `groups.py` — `LabelledGroups` → `CellZones` / `FacePatches` (design: `docs/mesh_zones_and_patches.md`).
- `reorder.py` — `permute_cells(mesh, perm)`, the one cell-relabelling transform (`P·A·Pᵀ`; faces neither
  reordered nor swapped, so `neighbour_offset` passes through), and the `CellReordering` strategies
  (`IdentityReordering`, `ReverseCuthillMcKee`, `RandomReordering`). Build-time only.
- `graph.py` — the cell↔cell graph: `cell_adjacency_coo(mesh)` (scipy COO, for `csgraph` RCM) and
  `cell_adjacency_csr(mesh)` (`(offsets, neighbours)`, for partitioners), both on `interior_edges()`.
  Shared by reordering, `parallel/partitioner.py` and aggregation multigrid, so it lives here. Build-time,
  never traced.
- `structured.py` — `structured_grid_2d`/`_3d` (vectorized via `_FaceFamilyBuilder` → `Mesh.from_csr`),
  **orthogonal only**. `structured_grid_2d(..., periodic=("x",))` fuses left/right into one seam face per
  row (`nx >= 2`; only `bottom`/`top` named). **Grid skewing stays in the tests**
  (`tests/support/meshes.py` `perturbed_grid_2d`/`_3d` displace interior `node_coords` after construction)
  — do not re-add a `perturb` knob here.
- `collapse.py` — `collapse_extruded_direction(mesh, removed_patch_names)`: one-cell-thick extruded 3D →
  genuine `dim == 2` `Mesh` (infer axis from the two caps, dedup nodes, reduce side quads to edges, drop
  caps, carry owner/neighbour/zones/offsets, rebuild via `from_csr`). A **mesh** transform the OpenFOAM
  reader calls on `empty` patches; unit-tested file-free (`structured_grid_3d(nx, ny, 1)` → matches
  `structured_grid_2d`). Fully vectorized (no per-face Python loop) using `aquaflux/ragged.py::rows`.
- `distance.py` — `distance_to_patches(mesh, geometry, patch_names)`: per-cell distance to the nearest
  face **centroid** of named patches (near-exact at the wall, looser far away), for near-wall closures.
  - **The search dispatches on concrete vs traced inputs**: concrete → `scipy.spatial.cKDTree`
    (`_nearest_squared_distance_concrete`, `workers=-1`, exact); traced (any `jax.grad`, not only `jit`)
    → blocked `jnp` search (`_nearest_squared_distance_traced`) run through one `jax.lax.map`, so the
    program does not grow with block count, with a bounded working set. Pinned in
    `tests/unit/test_distance.py`, including the dispatch itself
    (`test_concrete_inputs_use_the_kdtree_path_and_traced_inputs_do_not`) and the flat program size
    (`test_block_count_does_not_grow_the_compiled_program`).
- `surface.py` — `patch_triangles(mesh, geometry, patch_names)` → `PatchTriangles` (`vertices (n,3,3)`,
  `patch_id`, `patch_names`, `face`): boundary patches cut into the centre-fan triangles, each wound to face
  **into the domain**, decided from the stored normal, never node order
  (`test_mesh_patch_triangles.py::test_every_triangle_faces_into_the_domain_whatever_the_node_order`).
  Refuses 2D meshes, patches holding interior faces (baffles are two-sided), and faces **not star-shaped
  from their vertex mean** (the fan folds). `allow_folded=True` keeps folded faces for surfaces that only
  **block** (the fan still covers the face; `case.PatchBody` uses it); emitters/reflectors keep the
  refusal (overlap double-counts area). Lives here because it is mesh, and `radiation/` imports no mesh.
  Ear clipping is not built; build it when an emitting mesh trips the check.
- `quality.py` — `face_planarity`, `centroid_iteration_shift`, `closed_cell_residual`.
- `winding.py` — `stored_ring_is_outward(mesh)`: which way each stored ring runs relative to its owner,
  recovered from the face-geometry strategy's own flip (a mesh orients the normal, never the ring). The one
  home for every writer that gives ring direction a meaning (VTK polyhedra, the polyMesh writer). Pinned
  against an independent Newell-normal test (`tests/unit/test_mesh_winding.py`).

Naming: not `metrics.py` (CFD "mesh metrics" are quality measures); say "face geometry"/"cell geometry".

## Binding decisions (formulas replicate the C++ framework)
- **SoA + integer-index connectivity + `segment_sum`**, not a pointer graph. Owner/neighbour suffice;
  cell↔face incidence is implicit.
- **Store primitives, form displacements inline.** Stored: face area/centroid/normal, cell
  volume/centroid. `x_ip − x_owner`/`x_ip − x_neighbour` are cheap gathers formed in the operator from one
  shared formula, never materialized.
- **Owner-outward normal is a container invariant, fixed at build** (decided deviation from C++, which
  trusts the reader's winding — do not revisit). Orientation uses the approximate cell centroid; robust to
  node order, identical values for a correctly wound input.
- **A neighbour index `< 0` (exactly `−1`) marks a boundary face**; it scatters to the owner only.
- **2D and 3D geometry:**
  - 2D edge: `area=|d|`, `centroid=midpoint`, `normal=(d_y,−d_x)/|d|`.
  - 3D polygon, **centre fan** (apex = vertex mean): `S = Σ ½(v_i−c)×(v_{i+1}−c)`, `area=|S|`, unit
    `normal=S/|S|`, `centroid=Σ(d_i·n)c_i/|S|` (**signed** weights — the unsigned form is wrong on a
    non-convex face; `Σ|d_i|` is kept only for planarity). Zero-safe via `aquaflux.vectors.norm`.
    **Deliberate deviation from the C++ node-0 fan (do not revert)**: that gave `Σ|d_i|` and a sub-unit,
    order-dependent normal on warped faces.
  - Face node lists are ragged CSR, no padding.
- **Validation at construction.** `FaceNodeConnectivity.from_csr` checks CSR structure;
  `Mesh.validate()` (topology only, `O(n)`, called by `from_faces`/`from_csr`) checks finite coordinates,
  integer dtypes, index ranges, node counts (exactly 2 in 2D, ≥ 3 in 3D), repeated nodes in a face,
  neighbour sentinel exactly `-1`, self-neighbour, and every cell referenced (`segment_sum` silently drops
  out-of-range owners). Collinear faces and unclosed cells are geometric — surfaced by
  `quality.face_planarity` / `quality.closed_cell_residual`. ⚠️ `validate()` must not move into `__init__`
  (that runs on pytree unflatten under `jit`); a padded partition mesh is deliberately not validated
  (`parallel.md`).
- **Named groupings are partitions** (one label per element). Defaults: one `"default"` zone;
  `"interior"` + `"boundary"` patches; `"padding"` reserved for distributed padding. All three are in
  `groups.RESERVED_PATCH_NAMES` (the OpenFOAM reader's guard imports it).
  `FacePatches.uncovered_boundary_faces` backs `BoundaryConditions.resolve`'s coverage refusal
  (`boundary.md`). Zone interfaces are derived (`CellZones.interface_mask`/`interface_mask_between`) and
  may also be named as patches (a baffle). Node groups not built; overlapping groups rejected.
- **`FacePatches` carries what a mesh file declared** — static tuples `patch_types` (`(patch, type)`) and
  `patch_groups` (`(group, members)`), default `()`, **uninterpreted** by the numerics (the case file
  resolves groups via `addressed_by`). A group is not a partition, hence a separate record. `__check_init__`
  refuses unknown members, reserved names, duplicate or empty groups. **A transform removing patches must
  drop them from both records** — `FacePatches.without(removed)` (used by the collapse); partitioning keeps
  the global record (`eqx.tree_at` on `label`), padding copies it.
- **One centre-fan pass, no centroid iteration** — `centroid_iteration_shift` is ~3e-4 even on a violently
  warped quad. Re-check with it on a representative real mesh before adding iteration.
- **Cell ordering does not need fixing upstream for AMG**: the smoother and direct coarse solve are
  permutation-invariant, and the greedy aggregation now visits each level in its own RCM order, so a
  scrambled numbering builds the same compact aggregates
  (`tests/unit/test_multigrid.py::test_aggregation_coarse_space_is_ordering_robust`). `reorder.py` remains
  for matvec locality and partitioning.

## Terminology (D / fip / R are renamed, not gone)
The papers' `D_P,ip`/`D_nb,ip` are `x_ip − x_owner`/`x_ip − x_neighbour`, formed inline in the diffusion
operator; the skewness correction (the Fortran's `R`) is the tangential projection `dxip − (dxip·n)n`,
inline; the interpolation factor `f` belongs to the gradient schemes (`schemes/`). The mesh supplies
primitives only.

## Testability seam
Geometry is unit-tested from hand-built connectivity against analytic values (`tests/unit/test_mesh.py`):
2D squares/triangle/parallelogram; 3D cube, tetrahedron, two cubes, triangular prism (mixed faces), and a
non-unity `[0,2]×[0,3]×[0,4]` box that catches scale and axis-transposition errors. Orientation is proven
robust to node order.

## Golden reference
C++ `Face_impl.hpp`, `Cell_impl.hpp`, `FaceConnection.hpp`. Cite the math in comments, never these files.
