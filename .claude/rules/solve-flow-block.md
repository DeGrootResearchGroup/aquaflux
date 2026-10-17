---
paths:
  - "aquaflux/precondition/saddle_multigrid.py"
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

**The SIMPLE level's derived pieces are stored on their structural pattern (binding, #290).**
`saddle_multigrid.py` forms each level's `D⁻¹G` and Schur `C − D (D⁻¹G)` with `_structured(...)`
(`multigrid.py`'s `_onto_pattern` rule), and the non-block-splitting `D⁻¹` as an explicit CSR rather
than a `diags` product. A plain `scipy` product drops entries that come out exactly zero, so stored
counts moved with the values and recompiled the coupled step on a refit. Do not "simplify" either back.
The patterns are kept between refits by `StructuralPatterns` (`_simple_pieces(patterns=)`); see
`solve-amg-multigrid.md`'s Galerkin-shape entry.

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
