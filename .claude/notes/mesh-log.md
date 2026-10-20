# Investigation log — `aquaflux/mesh/`

> Reference-only, never auto-loads. Dated measurements and fix narratives behind
> `.claude/rules/mesh.md`; the rule holds the current state.

## From `mesh.md` — `geometry()` compiled as one call (`_fused_geometry`)

  - **Measured together, on a synthetic 1,000,000-cell structured hex mesh (4 named wall
    patches), stage-by-stage peak RSS across `structured_grid_3d` + `Mesh.from_csr` +
    `geometry()`, one clean process per arm, macOS/arm64, JAX 0.10.2, x64 enabled** (harness not
    in the repository): mesh construction's own delta **1944.7 MB → 1137.2 MB**, `geometry()`'s
    own delta **2432.8 MB → 1210.9 MB** (close to an isolated eager-vs-jit comparison of
    2204.7 MB → 1109.2 MB taken on `geometry()`'s arithmetic alone), cumulative peak through
    `geometry()` **4521.4 MB → 2492.2 MB**, roughly a 45% cut. ⚠️ The next stage
    (`SSTTurbulence.build`) is excluded from that cumulative figure on purpose: peak RSS is a
    process-wide, monotonically-increasing high-water mark, not a per-call measurement, so a
    "delta after this call" can smear onto whichever call next triggers the allocator to grow —
    an earlier probe attributed a further +2543 MB to this stage and a same-scale repeat of the
    identical call sequence reproduced none of it, so that particular attribution is noise, not a
    real per-stage cost. Only the two figures kept here (mesh construction, `geometry()`) held up
    across repeats. One run per arm even for those; re-measure before citing a tighter number.

## From `mesh.md` — `collapse.py` vectorization (#112, 2026-09-14)

  - **Both internal per-face Python loops (axis inference, side-quad→edge reduction) are vectorized
    (issue #112, fixed 2026-09-14) — no per-face loop remains in this module.** Both used to iterate
    `cap_faces`/`kept_faces` in Python, each doing several numpy calls per face. A shared
    `ragged.rows` helper (`aquaflux/ragged.py`, the numpy-only leaf shared with `FaceNodeConnectivity.select` and the radiation coarsener; it was `collapse.py`'s private `_ragged_subset`) flattens an arbitrary (non-contiguous) subset of CSR rows the same way
    `FaceNodeConnectivity.from_csr` flattens every row, so both loops reduce to one vectorized pass:
    axis inference becomes an unbuffered per-face min/max (`np.minimum.at`/`np.maximum.at`, the same
    idiom `aquaflux/io/openfoam/cyclic.py::_face_centroids` uses for its per-face vertex-mean sum) and
    a distinct-axis check; the side-quad reduction becomes a `(row, value)` lexsort whose block
    boundaries give each face's distinct in-plane node count and values, with no need to preserve
    perimeter order (an edge is undirected, so which of the two endpoints comes first does not
    matter). **Measured on a synthetic one-cell-thick structured slab (`structured_grid_3d(n, n, 1)`),
    same process before/after, macOS/arm64, Python 3.13.4, numpy 2.4.4, one run per size:** end-to-end
    `collapse_extruded_direction` (not just the two loops — node dedup, `Mesh.from_csr`, and
    validation are included) went **0.917 s → 0.123 s** at 160,400 3D faces and **3.470 s → 0.509 s**
    at 640,800 3D faces, a 7–7.5× cut; the acceptance test (`tests/unit/test_collapse.py`) still
    reproduces `structured_grid_2d` exactly.

## From `mesh.md` — `CellGeometry.from_faces` local coordinates (2026-08-17)

  - **⚠️ `from_faces` accumulates in coordinates local to each cell, not the mesh's raw (global)
    ones — a numerical-conditioning fix, not a formula change (measured 2026-08-17).** The
    divergence-theorem sum is exact and origin-independent for the true integral, but each
    per-face term is `O(|x_ip|)` while a cell's own volume/centroid offset is `O(cell size)`; for
    a cell small relative to its distance from the origin the sum is near-total cancellation, and
    the centroid formula (which multiplies the flux by `x_ip` a second time) can lose enough
    precision to land outside the cell it describes. A real snappyHexMesh CAD mesh whose cells run
    ~1e-3 m at ~1.7 m from the origin hit this directly: 0.84% of interior faces had a corrupted
    owner centroid, enough to flip `D_P.n` negative and crash the AMG diagonal-positivity guard,
    even though the cell's own closed-cell residual and face-normal orientation were both exactly
    correct — the corruption was purely in the centroid's floating-point conditioning. `from_faces`
    now takes `approx_centroid` (the same array `approx_centroids` already computes to orient
    normals, reused rather than recomputed) as the per-cell local reference point: each face's
    contribution is formed relative to *its own cell's* reference (owner and neighbour use their
    own, not a shared one), summed, then shifted back by that same reference point. Verified on
    the same mesh: bad-`D_P.n` faces (interior and boundary) go from 40,915/14,232 to zero: full
    scale, not a hand-picked example. A synthetic small-far-from-origin cube pins the mechanism
    (`tests/unit/test_cell_centroid_accurate_far_from_origin`).

## From `mesh.md` — `distance_to_patches` search dispatch (#240, 2026-08-19)

  - **The nearest-target search DISPATCHES on whether the centroids are concrete or traced (issue #240,
    fixed 2026-08-19) — a k-d tree for the ordinary eager build, a blocked `jnp` search only as a
    fallback under differentiation.** The two searches used to be one: a single blocked `jnp` search run
    unconditionally, on the reasoning that a spatial index needs concrete coordinates and geometry may
    carry **tracers** (a differentiated build, gradients w.r.t. node positions). That reasoning is still
    correct for the traced case, but it does not apply to the overwhelmingly common one: an ordinary
    eager case build never differentiates through mesh construction, so its centroids are concrete and a
    `scipy.spatial.cKDTree` applies cleanly. `_nearest_squared_distance` checks
    `isinstance(x, jax.core.Tracer)` on both arrays (`_is_traced`, a local helper — this is a one-line
    type predicate, not a formula, so it is not shared with the differently-scoped `_is_traced` in
    `turbulence/coupled.py`) and routes to `_nearest_squared_distance_concrete` (the tree, queried with
    `workers=-1` for a parallel search) or `_nearest_squared_distance_traced` (the blocked `jnp` search,
    unchanged in spirit). **Measured on a synthetic 1,000,000-cell mesh with 4 wall patches (40,000
    combined wall faces), the exact configuration the issue was filed against**: the concrete path took
    **4.2–4.6 s** end to end through `distance_to_patches` (tree build ~6 ms, parallel query the rest)
    against the blocked search's **61–77 s** on the same machine — a ~15× cut, and what "minutes at scale"
    in the issue's own title actually meant. Exact to floating-point rounding against the brute-force
    form on every case checked, including bit-identical on the standard test mesh (a k-d tree
    nearest-neighbour query finds the true nearest point by a smarter search, not an approximate one).
    Pinned by `test_distance.py::test_concrete_search_matches_the_whole_array_form` and
    `test_concrete_inputs_use_the_kdtree_path_and_traced_inputs_do_not` (the latter pins the *dispatch*,
    not just the numbers — a regression that swapped the two branches would still pass an
    equivalence-only test on any input small enough not to time out, since `scipy` would simply never be
    handed a Tracer to fail loudly on). `jax.grad` alone (no `jit`) is already enough to trigger the
    traced branch — confirmed directly (`isinstance(x, jax.core.Tracer)` is `True` under a bare
    `jax.grad`, not only under `jax.jit`) — which is what
    `test_distance.py::test_distance_is_differentiable_through_node_positions` exercises.
  - **The traced fallback's blocked search is now run through a single compiled `jax.lax.map` loop,
    not a Python loop building one XLA program per block and feeding a large `jnp.concatenate`.** The
    two are numerically identical — each cell's nearest target is independent of every other cell's, so
    a block boundary reorders and drops nothing (`_nearest_squared_distance_traced` still pads to a whole
    number of blocks by repeating the last cell, discarded by the final slice) — but a Python loop's
    per-block trace-and-dispatch, and a concatenate over as many operands as there are blocks, is what
    the issue's own evidence showed compiling for over 5 minutes when embedded in a larger jitted
    computation (`[Compiling module jit_concatenate for CPU] Very slow compile? ... 5m5.656426s`).
    `jax.lax.map` compiles the per-block body once and loops it on-device (built on `scan`), so the
    compiled program's size no longer grows with the block count — pinned by
    `test_distance.py::test_block_count_does_not_grow_the_compiled_program`, which checks the jaxpr's
    own equation count stays flat as the forced block count rises two orders of magnitude. **This does
    NOT reduce the traced path's raw eager wall-clock time — measured slightly WORSE** (the same
    1,000,000-cell/4-patch configuration cost ~97–102 s through `jax.lax.map` against the Python loop's
    61–77 s; block size made no difference across a 64 MB–8 GB sweep, so the traced brute-force search is
    compute-bound, not dispatch-overhead-bound, at that scale). That regression does not matter in
    practice: the traced path is reached only by an actually-differentiated build, which is rare and
    already expensive relative to an ordinary case assembly, and unlike the concrete path it was never
    the one the issue's "impact" section described as paid "repeatedly" per Reynolds-continuation rung.
  - **The blocking mechanics themselves — the memory bound, the padding, the working-set sizing — are
    unchanged from the original fix (measured 2026-08-11) and now apply only to the traced path.**
    `centroid[:, None, :] - target[None, :, :]` is a dense cell-by-target-face structure: on `bfs3d`
    (23040 cells, 4736 wall faces) that is 109M entries, and because `norm_squared` materializes its own
    product under eager JAX ~6.1 GB would be live at once. Blocked at a 64 MB working set it peaks at
    ~850 MB, an 8.1× cut, computing exactly the same numbers in the same order. That residency figure
    (and the `jax.live_arrays()` check that nothing leaks beyond it) was taken when this was the only
    search path there was; it has not been re-measured since the concrete path stopped exercising it, but
    nothing about the traced path's memory behaviour changed.

## From `mesh.md` — `patch_triangles` fold refusal

On the Sozzi snapped mesh (`lampWall` 48,550 faces, `bodyWall` 329,028; quads with a few 5-/6-gons and
218 triangles) **no** face folds — the worst fan triangle carries 12% of its face's mean share — so ear
clipping was not built. The snapped Voronoi bunny of `validation/ray_effects_room` has **291** folded faces
(2026-10-05), which is why `allow_folded=True` exists.

## From `mesh.md` — `FaceCellConnectivity` cached fields (#109 2026-08-18, #268 2026-09-09)

    wrapper methods. **`interior` and `safe_neighbour` are stored fields, computed once in
    `__post_init__`, not properties recomputed on every access (issue #109, fixed 2026-08-18)** — a
    recomputed property folds away outside a loop but not inside a traced `while_loop` body (a
    Krylov matvec), where the compare-plus-select used to run on every iteration instead of once.
    **That caching means they go stale under any pytree surgery that skips `__init__` — `eqx.tree_at`
    on `owner`/`neighbour` of an existing instance included (issue #268, fixed 2026-09-09).** No
    runtime hook catches this: JAX pytree unflattening (what `tree_at`/`jit`/`vmap` rebuild a node
    through) bypasses `__init__`/`__post_init__`, and `eqx.Module.__check_init__` — the hook that
    looks like a fix — is documented to run only inside `__init__`, so it is bypassed the same way.
    The fix is structural, not a check: `with_topology(owner, neighbour)` is the one sanctioned way
    to change an existing relation's topology (it rebuilds through the constructor, carrying
    `n_cells`/`neighbour_offset` across), and `permute_cells` now calls it instead of hand-rolling the
    reconstruction — never patch `owner`/`neighbour` on an existing instance directly.

## From `mesh.md` — cell ordering and AMG (superseded rationale)

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

The test this cited, `test_ordering_affects_coarse_space_and_rcm_restores_it`, no longer exists; the
aggregation now applies reverse Cuthill–McKee internally per level and
`tests/unit/test_multigrid.py::test_aggregation_coarse_space_is_ordering_robust` pins that a scramble no
longer degrades the factor (so no upstream renumbering is needed).

## 2026-10-19 — faces oriented by shared edges, not by the cell centroid

Agreed with the project owner after prism layers on the Sozzi water (first dual cell 0.5 mm, 5 layers)
left 59 wall cells unclosed under the centroid test: cells 0.24–0.37 mm thick on walls of radius 9.5 mm
(a 5 mm tile bows ~0.33 mm), at the riser and outlet junctions and the cap rims. Oriented by shared edges,
every one of them closed to 2.8e-17 of the largest face.

Comparison of the new normals with the centroid test (`ring_signs` time, one run each, macOS arm64):
pitzDaily (2D, 12225 cells) 0 faces differ, 0 undetermined, 0.08 s; bfs3d (23040) 0 / 0, 0.03 s; the
periodic turbulent channel (2D, 960) 0 differ, 240 undetermined (the cells on its cyclic seam, which fall
back to the centroid test); Sozzi snappyHexMesh (1635909 cells, 5249365 faces, refined) 0 / 0, 1.2 s, with
`geometry()` 1.9–2.1 s in all. A first version sorting all 42M joint rows globally took 9.5–11 s; matching
joints cell by cell took it to 1.2 s.

The dual's concave cut, re-measured under the new orientation (Netgen L, maxh 0.01, worst non-convexity):
as built 0.349; vertices between pieces at face centroids 0.559; cut edges' hubs at midpoints 1.106;
both 1.233; uncut 1.203 — all closed now. So the cut is a quality step, not a correctness one; the
Netgen L test asserts convexity (< 0.45) in place of closure.

Mutations, each red: always the centroid; no volume sign; no fallback for undetermined cells; the joint
parity flipped; faces unreached from the first accepted (the hanging-node cube).

## 2026-10-19 — quality metrics against checkMesh

`aquaflux.mesh.quality` (`face_non_orthogonality`, `face_skewness`, `cell_concavity`), macOS arm64, on the
Netgen duals whose checkMesh output (OpenFOAM 13, Docker) is in `meshing-log.md`, rebuilt by
`validation/meshing_dual/dual_scale_convexity.py`'s solids and `PolyhedralDual(45 deg)`, unprojected, one patch;
cell counts identical to the checkMesh runs. Non-orthogonality over interior faces, average as
`acos(mean cos)`:

| dual | cells | non-orth max / avg (ours) | checkMesh | skewness ours / checkMesh | concave ours / checkMesh |
|---|---|---|---|---|---|
| L maxh 0.01 | 3718 | 46.2 / 12.0 | 46.2 / 12.0 | 1.44 / 1.44 | 57 / 57 |
| Sozzi 0.01 | 13297 | 47.8 / 14.2 | 47.8 / 14.2 | 1.96 / 1.96 | 309 / 309 |
| cylinder 0.005 | 14426 | 50.1 / 13.9 | 50.1 / 13.9 | 1.51 / — | 1108 / 1108 |
| sleeve 0.005 | 15995 | 51.7 / 12.2 | 51.7 / 12.2 | 1.24 / — | 258 / 258 |

The mean of the angles reads 10.5 / 12.5 / 12.2 / 10.6: checkMesh averages the cosine. Face flatness min
(`face_planarity`, apex at the vertex mean) 0.80 on the cylinder against checkMesh's 0.79 (apex at the face
centroid). The maximum skewness is on an interior face in all four, so the boundary floor `0.4|d|` is not
checked by these. Not reproduced: the log's "Kuhn tets of the unit cube (n 4, perturb 0.2)" dual (42.6 /
25.1, skewness 1.12) — today's `tetrahedral_grid_3d(4, perturb=0.2)` (seed 0) dual gives 45.9 / 25.2, 1.07;
the log does not record the seed or code state, so which differs is not known. Probe scripts are in this
session's scratchpad (not kept); the harness `validation/meshing_dual/planarization.py` prints the same
quantities.

Mutations of the metrics, each red: boundary line taken as the interior one; periodic image dropped;
skewness boundary line = owner to face centroid; interior floor 0.3; boundary floor 0.2; extent dropped;
warp over area not root area; triangulation change's neighbour sign; gain only (needs faces of 5+ vertices:
a quadrilateral's heights are symmetric); a half not a third; non-convexity's neighbour sign; concavity
pairing a face with itself; a block boundary skipping a cell; `cell_nodes` from owners only.
