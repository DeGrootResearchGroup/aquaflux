---
paths:
  - "aquaflux/precondition/saddle_multigrid.py"
  - "aquaflux/precondition/simple_fit.py"
  - "aquaflux/solve/shift_basis.py"
---

# Rules — `aquaflux/precondition/` the flow block (traced `[u, v, w, p]` saddle preconditioning)

> Split out of `solve.md`; see it for the package-wide contracts. Dated measurements and the full
> investigation live in `.claude/notes/solve-flow-block-log.md` — read it before revisiting the traced
> flow block, since several "wins" there were later qualified or retracted.

## The flow block — traced preconditioning of the `[u, v, w, p]` saddle

**Current status.** The traced SIMPLE-smoothed hierarchy (`SimpleSmoothedInverse`, case-file
`SimpleSmoothed`) is `bfs3d`'s leading-block inverse (`BFS3D_FLOW_INVERSE` accepts only `simplesmooth`).
It was chosen for robustness, not speed: incomplete-LU arms were sensitive to elimination order by orders
of magnitude on this operator, and the traced path is the only one with a GPU route. On a march it was
measured slower than the (since deleted, #371) PETSc ILU(0) incumbent — 1.26x at matched `inner_tol`
1e-2 (log: "THE TRACED FLOW BLOCK IS STILL NOT FASTER ON A MARCH"); the default accepts that.

**The SIMPLE level's pieces are fitted on the device, from a plan made once per coarsening (binding,
2026-10-22; `simple_fit.py`).** `simple_plan(pattern, block_size, width, ...)` holds a level's structure
as a `SimplePlan`: the four block cuts (each row's columns ascending), the structural patterns of `M G`
and the Schur `C − D (M G)` (products over ones, so an entry that comes out exactly zero stays stored and
no shape moves with the values -- do not "simplify" that back), their padded-row layouts, where `C`'s
entries sit among `S`'s, the cell-block gather and the block inverse's layout. `fit_simple_level(plan,
level_slot_values)` (one `eqx.filter_jit`) forms every value from the level operator's padded-row values:
the diagonal inverses (Jacobi / Frobenius / SIMPLEC), each cell's Frobenius block
`M_i = F_ii^T (R_i R_i^T)^-1`, `M G`, `S` and `S`'s diagonal inverse, plus `SimpleChecks` the host reads
once (`refuse_unusable`). `SimpleSmoothedInverse` holds the plans (`_simple_plans`), keyed by the
identity of the level operators' `columns` arrays, which a frozen refit keeps and a rebuild replaces;
`_simple_pieces(a, ...)` makes a plan for one host operator (tests, diagnostics) and returns the double
host Schur beside the pieces. There is no `StructuralPatterns`, `_structured`, `block_approximate_inverse`
or host `_diagonal_approximate_inverse`.
- **No product term list is held (the owner's constraint: memory must not grow).** A product places
  each term by a binary search of its column in its output row (`_search`, `scan_unrolled`, ~2x the
  looped search), `TERMS_PER_CHUNK` terms at a time (2**18, a CPU cache calibration). A cell-level
  position table would remove the search but costs ~7 MB on pitzDaily and ~0.2-0.5 GB on bfs3d.
- **Two shortcuts, each decided per level at plan time and each with the general path beside it**
  (pinned on all three fixture shapes by `tests/unit/test_simple_fit.py`): where a cell's velocity rows
  store the same columns (`aligned_velocity_rows`; true of a probed Jacobian and its nodal Galerkin
  levels) the Gram matrix is a dot product of slot values, no search (0.40 -> 0.004 s on pitzDaily's fine
  level); where `D`'s rows store one run per velocity field and `M G`'s cell rows share columns
  (`divergence_run_width`) the Schur product sums a cell's fields before placing a term, cutting the
  searches by the field count.
- ⚠️ Not bit-identical to the old host pieces: the products sum in another order (≤ ~6e-16 relative on
  the test saddle). Cost on pitzDaily (CPU, 2026-10-22): same march, solve +2-3 % (the first fit; the
  refits are level), peak resident +0.02-0.04 GiB; log, "SIMPLE pieces on the device (2026-10-22)".

## Defaults and why

- **Frobenius-optimal diagonal on both halves** (`frobenius=True`, `schur_frobenius=True`,
  `pressure_smoother_weight=1.0`). Jemcov & Maruszewski (2008) eq. (39): `F~⁻¹_ii = F_ii / ||F_i||²`
  is Jacobi with an automatic per-row under-relaxation (factor `F_ii²/||F_i||² ≤ 1`, median 0.53 on
  `bfs3d`, ~0.9–1.0 on small 2D channels — so small cases cannot screen it). As the velocity predictor
  it turns an amplifying SIMPLE sweep (more sweeps worse) into a contracting one; ~10⁴x at 2 sweeps.
- ⚠️ **The Schur-diagonal choice and `pressure_smoother_weight` are one knob.** An undamped Jacobi
  pressure sweep blows up; an undamped Frobenius one was best; Frobenius plus 0.7 over-damps. Move them
  together.
- ⚠️ **Never rank smoother arms at a single sweep count** — the Schur Frobenius verdict flipped between
  2 and 4 sweeps. (On the β = 0 adjoint, conversely, 4 vs 8 sweeps gave identical counts at 1.6x cost:
  there the count is set by the coarsening, not the smoother.)
- Not transferable from that paper (measured on `bfs3d`): multi-step `N = 10` (1.9x for 10x work) and
  its Algorithm 2 (45x worse). Their optimality rests on a zero `(p, p)` block and `D = Gᵀ`; ours has
  Rhie–Chow damping and a nonsymmetric `D`.

## Adjoint through this block — validated

`jax.grad` through `solve_coupled` on `bfs3d` agrees with a central difference to 1.9e-4 and is
iteration-count independent (two starting checkpoints: forward applications +16 %, adjoint −8 %,
gradients equal to 9 figures). Harness `validation/bfs3d_openfoam/adjoint_probe.py`; numbers in the log.
Traps that transfer:

- **Pass `adjoint_solver=` to `solve_coupled`** with a restart budget sized for the transpose solve
  (~1450–1700 applications at β = 0 on `bfs3d`); the stock lineax budget stagnates.
- ⚠️ **Check the adjoint against a finite difference only at a tight root** (`rtol` ≤ 1e-4). At `rtol`
  2e-2 the gap was 23 % and a larger `eps` made it worse — the IFT adjoint is valid only at `R = 0`, so
  fix the root, not the step.
- ⚠️ **A `−R` linear probe cannot rank adjoint preconditioners.** The adjoint's right-hand side is the
  cotangent, localized in one field block; a traced arm that won 4 vs 11 cycles on the `−R` probe lost
  1.08x on the real gradient. Do not size an adjoint budget from linear-probe cycle counts either.
- **Compare by preconditioner applications, not cycles or wall clock**, when per-application cost
  differs between arms. `root_adjoint`'s backward rule discards the cycle count;
  `adjoint_probe.TransposeApplyCounter` counts transposed applications instead.
- ⚠️ **Compare against the matched incumbent**: an arm replacing only the leading block is measured
  against the field-split arm that differs only there, never the monolithic one (~45 % slower).
- ⚠️ **When a change lands on one arm of an A/B, the other arm's recorded numbers are stale** — re-measure
  the baseline before ranking (the march comparison was misread this way once).

## Dual-time inner tolerance

`bfs3d` ships `inner_tol` 0.01 with `inner_steps` 5 (`case.yaml`): 1e-3 cost a third of the march for
nothing, and 0.05 is where an extra outer step appears (a half-solved large-Δτ step overshoots). The
threshold is case- and Δτ-dependent; only the mechanism transfers. `inner_steps` is inert at these
tolerances — the loop exits on the tolerance first.
