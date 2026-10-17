# Investigation log — `aquaflux/boundary/`

> Reference-only, never auto-loads. Dated measurements and investigation detail behind
> `.claude/rules/boundary.md`; the rule holds the current state.

## From `boundary.md` — "Every prescribed boundary number is a floating array leaf" (#363)

- **Why, measured before the change.** `field_fn` and both `velocity` fields were
  `eqx.field(static=True)`, and the four numeric fields were annotated `float`. The issue said the
  result was a "silently zero" gradient. **That is only one of the three paths, and the others need
  to be known:**
  - An **eager residual** gradient worked regardless, through the value closed over at
    construction.
  - A gradient **through `RootSolver`** (the assembler carried as `theta`) **raised**
    `UnexpectedTracerError`, for both the inlet velocity and a `DirichletField` coefficient.
  - `eqx.filter_grad` with respect to an assembler returned **no cotangent** for the static fields
    *and* for a `float`-valued `PressureOutlet`. That was the genuinely silent one.
- **The old reason for `static=True` was false.** The comment said a `jnp` array "would not
  concretize under jit"; a non-static array velocity worked under `filter_jit` and through the solve,
  with the gradient matching finite differences to 4e-11.
- **Side effect, measured:** a changed inlet value used to recompile a `filter_jit` function taking
  the assembler (the float sat on the static side); as an array leaf it is a cache hit.

## From `boundary.md` — "Every boundary face must be covered" (#354)

The fast-tier census taken before the coverage check found 11 under-covered builds, and 10 were
`BoundaryConditions({})` on a mesh with no named boundaries (the unnamed `"boundary"` group omitted).
`SSTTurbulence`'s wall-distance helper named every patch including `"interior"` (137 builds in that census).

## From `boundary.md` — "One scatter for all patches" (#113)

- **Measured** (`validation/boundary_scatter_cost.py`, 2026-10-09, jax 0.11.2, CPU, Linux x86_64 cloud
  container, 4 cores, persistent compilation cache off, best of 3 after `jax.clear_caches()`, one
  run per arm, nothing else running). Scalar diffusion, 64 x 64 grid, boundary split into P patches:
  lowered scatters in the residual went from 12 / 36 / 132 (P = 4 / 16 / 64) to **6 at every P**, and
  in its Jacobian-vector product (jvp) from 16 / 64 / 256 to **8**. At P = 64 compile time went
  0.435 → 0.353 s (residual) and 0.327 → 0.205 s (jvp). Trace time did not move, because every
  closure is still traced on its own. pitzDaily coupled RANS (4 patches, `case.yaml`, hybrid initial
  condition): lowered scatters 608 → 380 (residual) and 998 → 656 (jvp). Compile went 3.70 → 3.21 s
  and 6.35 → 5.70 s, trace about unchanged (2.0 s and 2.9 → 2.5 s). Residual and jvp were
  **bit-identical** to the per-patch fold's. Execution time was not measured.
