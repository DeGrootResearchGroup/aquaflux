# pitzDaily — gradient reconstruction, head to head

## Why

A second version of the `pitzdaily_openfoam` case whose only question is: **what does the
Hessian-corrected (Betchen) gradient buy, and what does it cost, against the corrected Green–Gauss
gradient the benchmark ships?** Same mesh, same boundary conditions, same model constants, same march
settings, same Reynolds continuation — only the reconstruction differs.

It exists as its own case rather than as a flag on the benchmark for one reason: **it needs a
different preconditioner**, and changing the benchmark's preconditioner would change the
configuration its validation figures were taken under.

## Why a different preconditioner

The benchmark preconditions from a **coloured Jacobian probe**, which recovers the Jacobian out to a
fixed cell-graph distance. Coupling beyond that distance is not dropped — it is *folded onto near
entries*, because a colouring is collision-free only for the pattern it was built at. The benchmark's
reach of 5 is exact for corrected Green–Gauss, whose Jacobian carries exactly zero mass past it
(the sweep count bounds its stencil).

The Hessian-corrected residual has no such cut-off. Its gradient couples to the Hessian, which couples
to the neighbours' Hessians, so it reaches essentially across the mesh at any sweep setting. A probe
sized for the other scheme would hand that arm a corrupted operator, and comparing a sound
preconditioner against a corrupted one measures the probe rather than the reconstruction.

So both arms run the **field-split stack the 3D backward-facing-step case uses**: a SIMPLE-smoothed
multigrid hierarchy on the leading `[u, v, p]` saddle and a nodal hierarchy on the trailing
`[k, omega]` scalars. That family forms an approximate Schur complement and applies V-cycles rather
than eliminating the matrix into pivots, so a perturbed far-field entry degrades a rate instead of
wrecking a factorization — it is the tolerant choice when one arm's operator cannot be probed exactly.
Both arms get the identical preconditioner and the identical reach.

## Reading the result

**Cycle counts before wall clock.** If the two arms' Krylov cycle counts are close, the preconditioner
sees both operators about equally well and the wall-clock difference is the reconstruction. If the
Betchen arm's cycles are far higher, raise `PITZ_AB_REACH` — for **both** arms, since raising it for
one would compare two different preconditioners — and re-run.

**The probe reach is 5, and shortening it breaks this case specifically.** A capped march varying only
the reach gives bit-identical cycles at 2, 3 and 5 while the probe cost scales steeply (100, 165, 380
residual evaluations per refresh) — which looks like a clear saving, and is a trap: that sweep ran the
*standard* gradient, whose Jacobian carries exactly zero mass past its own reach, so it could not have
detected a reach effect on the long-stencil arm this case exists to measure. Run on the Betchen arm
itself, reach 3 costs **834 cycles over 74 steps** against reach 5's **511 over 72**, and the cycle
ratio against the standard arm falls from 1.93× to 1.18× — at reach 3 the probe under-resolves that
arm and the cost gap gets charged to the reconstruction.

The wall clock barely moves either way (the longer probe eats the cycle saving), so this buys a
comparable *measurement*, not a faster march. `PITZ_AB_REACH_SWEEP` re-measures on any configuration —
sweep it with the arm whose stencil is in question, not the one that cannot feel it.

**A small difference is a useful result.** This mesh's worst face is only about 6° off orthogonal and
corrected Green–Gauss is already linear-exact, so the second-order reconstruction has little to
correct here. Measuring that the cheaper scheme is sufficient on meshes of this quality is the point;
it is what says where the expensive scheme is and is not worth reaching for.

## Result (2026-08-21, first run)

Matched marches, both arms at probe reach 5, Betchen at outer/inner swept-5, everything else the
benchmark's own:

| | wall | cycles | steps | max cycles | `x_r/h` |
|---|---|---|---|---|---|
| standard | 711.1 s | 439 | 71 | 15 | 8.069 |
| betchen (swept 5/5) | 1603.5 s | 511 | 72 | 20 | 8.069 |
| ratio | **2.25x** | **1.16x** | — | — | same cell |

Field differences (betchen against standard, relative L2 / max, from the reach-3 pair): `U`
6.7e-03 / 3.1e-02, `p` 1.8e-02 / 1.7e-02, `k` 2.0e-02 / 5.5e-02, `omega` 1.9e-02 / 1.2e-02, `nut`
1.6e-02 / **1.2e-01**.

**The second-order reconstruction does not move what this case is judged by.** Both arms reattach at
`x_r/h` 8.069 against the OpenFOAM reference's 7.741 — and note *how* identical that is: the metric
returns the cell-centre x of the last reversed-flow cell, and cells there are ~0.076 h wide, so the
two arms agreeing exactly means they reattach in the **same cell**, i.e. within ~1%. Report it as
"agree to within one cell", never as "identical". The fields do differ, by 1-2% in L2 and up to 12%
locally in the eddy viscosity, so the schemes are not producing the same solution — that difference
simply does not reach the reattachment length on a mesh whose worst face is ~6 degrees off
orthogonal.

**The cost is 2.25x, and the first number measured was 3.25x — the difference was the probe, not the
scheme.** With the two arms at different reaches, the Betchen arm's under-resolved probe cost it 1.93x
the standard arm's cycles, and that got charged to the reconstruction. Matched at reach 5 the cycle
ratio is 1.16x and the wall ratio is the reconstruction. ⚠️ **This is one run per arm on a machine
whose per-application noise floor is ~15%**; the step counts (71 vs 72) and cycle counts are
contention-immune and are the load-bearing numbers.

## Status

- **Not a validated case.** The physics reference belongs to `pitzdaily_openfoam`; this case reuses
  that reference only to place both arms against it. Nothing here re-validates the benchmark.
- The preconditioner settings are taken from the 3D case's own measured configuration. Whether they
  are well calibrated *for this mesh* is not established — they are a starting point from a measured
  bundle, not a calibration for this case.

## Layout

- `run_ab.py` — the comparison. Imports the case definition (mesh, boundary conditions, model
  constants, and the reattachment-length metric) from `../pitzdaily_openfoam/compare.py` so there is
  no second copy of the benchmark to drift; supplies its own preconditioner and its own march driver.

## Running

```bash
validation/run_case.sh validation/pitzdaily_gradient_ab/run_ab.py
```

Both arms march to the benchmark's own stopping tolerance, so this is two full solves — run it the
same way any long solve is run, and read the per-arm logs (`ab-standard.log`, `ab-betchen.log`) while
it goes.
