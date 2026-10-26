# Option 1 audit: a pressure probe coloured for R2 ∪ S(τ), checked against the real Jacobian

Audit of the proposal's §8 steps 1 (static audit) and 2 (replay) only. The march ensembles (§8 steps 3–5)
and the library implementation (§9) are not part of this work and were not run or built.

## Configuration (applies to every table below)

- **Library**: `8dd92e7f` (branch `claude/bfs3d-preconditioner-reach-hiwoy9`, includes main's #829/#832),
  unchanged. Harness: `validation/pressure_strength_skeleton.py`, `validation/reach_option1_audit.py` and
  the `p@S<tau>` arm of `validation/reach_folding.py`, as committed on `claude/reach-option1-audit`
  (between the first and the final bfs3d step 5 inner 2 audit runs, only the "weakest/strongest face
  coupling" line was added; every other line of that output was identical, checked by `diff`).
- **Machine**: 4-core Linux container, 15 GiB, glibc allocator defaults, jax as installed by
  `pip install -e ".[lint]"`. One job at a time through `validation/run_case.sh`. Only counts and
  residuals are reported; no timing is meant as a result (fit times include compilation).
- **States** (checkpoints captured on this tree, kept outside the repository):
  - bfs3d: `BFS3D_COLUMN_REACH=0 BFS3D_CHECKPOINT_KEEP=60 BFS3D_INNER_DUMP_ABOVE=1`, shipped `case.yaml`
    otherwise (uniform reach 3, float32 field split, `PivotFloor(0.15)`, 12-station viscosity ramp).
    March: **17 steps, 143 restart cycles, x_r/h 8.3611**, the expected trajectory.
  - pitzDaily: `PITZ_CHECKPOINT_KEEP=60 PITZ_INNER_DUMP_ABOVE=1`, shipped `case.yaml` (every column
    probed at reach 2 in a reach-3 pattern). March: **21 steps, 102 cycles, x_r/h 8.0686**; the figure
    step then failed for want of matplotlib (exit 1 after the march, as expected).
- **Static audit Jacobian**: the uniform-reach-3 coloured probe of the **station's** assembler (the one
  the step's preconditioner is fitted from; neither case narrows its probe), float64, at the named state.
- **Replays**: `reach_folding.py`, the step rebuilt as the viscosity ramp builds it, exact operator, the
  shipped float32 split; "march rtol" is the step's own solver (rtol 0.3 row-scaled, residual stop),
  "tight" is rtol 1e-5, which every arm ran to its 16-cycle cap, so "tight" means the row-scaled
  residual after 16 cycles.
- **τ**: 0.01 and 0.03. **Spacing ratio**: `mesh_anisotropy.spacing_ratio` (max/min `|d·n|`), bins
  1–2 / 2–10 / >10 (bfs3d 40 / 13 440 / 9 560 cells, pitzDaily 8 399 / 3 643 / 183).
- Raw outputs: `option1_audit_logs/` beside this file.

## The skeleton reproduces the proposal's colour counts

`pressure_strength_skeleton.py` is the proposal's proxy (`proxy_build.py` + `candidates.py`) as one
module, with `spacing_ratio` in place of the old `conductance_ratio` and a check that the colouring is
collision-free for its pattern. Colours of the p column:

| mesh | reach 2 | R2 ∪ S(0.03) | R2 ∪ S(0.01) | reach 3 | kept distance-3 entries per row, S(0.03) / S(0.01) |
|---|---|---|---|---|---|
| bfs3d (23 040 cells) | 39 | 46 | 48 | 96 | 2.01 / 2.45 |
| pitzDaily (12 225 cells) | 20 | 20 | 21 | 33 | 0.12 / 0.29 |

These are identical to the proposal's numbers. Probes per fit: on bfs3d, uniform reach 3 takes 576 (6 × 96).
The replayed arm (p at S(0.01), the rest at 3) takes 528 (526 at S(0.03)). The proposal's target, p at
S(0.01) with the rest at 2, would take 243 (5 × 39 + 48); that configuration was **not** replayed.

## 1. Static audit: what R2 ∪ S(τ) drops from each (row field, p) block

For each row, the statistic is the largest `|J_ij|` at graph distance 3 that the pattern leaves out,
divided by that row's weakest nonzero face-neighbour `|J_ij|` in the same (row field, p) block. The
tables give the median, p99 and max over rows, and the share of rows above τ and above 3τ. Under
"reach 2" the τ is 0.01. Two context columns follow: the largest dropped entry over the row's
**strongest** entry in the block (max over rows), and the Frobenius norm of everything dropped over the
block's. ω rows with no face coupling to p (the 4 490 fixed wall cells on bfs3d, 472 on pitzDaily) are
left out. The full per-bin tables and the fold measurements are in the logs.

**bfs3d step 5, start (state-00004)**

| row field | arm (τ) | median | p99 | max | rows > τ | rows > 3τ | max dropped/strongest | dropped/block (Frob.) |
|---|---|---|---|---|---|---|---|---|
| u | reach 2 (0.01) | 4.84e-01 | 1.29e+01 | 3.42e+03 | 100.0% | 99.8% | 1.59e-02 | 4.76e-03 |
| u | S(0.01) (0.01) | 4.79e-03 | 2.86e-01 | 1.63e+02 | 26.2% | 10.0% | 3.22e-03 | 5.47e-05 |
| u | S(0.03) (0.03) | 6.85e-03 | 5.51e-01 | 1.63e+02 | 21.1% | 7.2% | 6.07e-03 | 1.35e-04 |
| v | reach 2 (0.01) | 4.97e-01 | 1.59e+01 | 8.89e+03 | 100.0% | 99.9% | 6.26e-03 | 7.71e-04 |
| v | S(0.01) (0.01) | 4.83e-03 | 8.68e-02 | 2.29e+01 | 11.4% | 2.6% | 5.80e-04 | 3.48e-05 |
| v | S(0.03) (0.03) | 8.64e-03 | 2.75e-01 | 1.06e+02 | 10.1% | 2.7% | 2.08e-03 | 9.60e-05 |
| w | reach 2 (0.01) | 4.26e-01 | 2.55e+01 | 4.58e+03 | 100.0% | 99.1% | 9.50e-03 | 9.23e-04 |
| w | S(0.01) (0.01) | 4.15e-03 | 7.43e-02 | 7.30e+01 | 13.8% | 2.2% | 1.20e-03 | 4.77e-05 |
| w | S(0.03) (0.03) | 9.05e-03 | 2.49e-01 | 1.36e+02 | 12.6% | 2.9% | 3.70e-03 | 1.46e-04 |
| p | reach 2 (0.01) | 4.17e-01 | 5.37e+00 | 6.47e+00 | 100.0% | 100.0% | 7.66e-03 | 4.59e-03 |
| p | S(0.01) (0.01) | 4.54e-03 | 9.69e-03 | 1.14e-02 | 0.2% | 0.0% | 1.03e-03 | 4.75e-05 |
| p | S(0.03) (0.03) | 7.08e-03 | 2.96e-02 | 3.72e-02 | 0.7% | 0.0% | 2.43e-03 | 1.26e-04 |
| k | reach 2 (0.01) | 4.45e-01 | 2.55e+01 | 9.73e+01 | 100.0% | 100.0% | 9.99e-03 | 4.77e-03 |
| k | S(0.01) (0.01) | 4.55e-03 | 1.16e-02 | 2.48e-02 | 2.0% | 0.0% | 1.05e-03 | 6.00e-05 |
| k | S(0.03) (0.03) | 7.32e-03 | 4.23e-02 | 9.03e-02 | 3.0% | 0.0% | 2.78e-03 | 1.45e-04 |
| omega | reach 2 (0.01) | 4.23e-01 | 5.50e+00 | 7.15e+00 | 100.0% | 100.0% | 1.19e-02 | 2.58e-03 |
| omega | S(0.01) (0.01) | 4.75e-03 | 1.00e-02 | 3.04e-02 | 1.0% | 0.0% | 8.13e-04 | 2.73e-05 |
| omega | S(0.03) (0.03) | 7.57e-03 | 3.01e-02 | 4.68e-02 | 1.1% | 0.0% | 2.74e-03 | 6.01e-05 |

**bfs3d step 5, inner 2 iterate (inner-00005-01)**

| row field | arm (τ) | median | p99 | max | rows > τ | rows > 3τ | max dropped/strongest | dropped/block (Frob.) |
|---|---|---|---|---|---|---|---|---|
| u | reach 2 (0.01) | 3.80e-01 | 8.41e+00 | 9.22e+03 | 100.0% | 99.5% | 1.85e-02 | 4.72e-03 |
| u | S(0.01) (0.01) | 4.38e-03 | 2.04e-01 | 1.47e+02 | 20.9% | 7.2% | 1.18e-03 | 6.80e-05 |
| u | S(0.03) (0.03) | 5.62e-03 | 2.67e-01 | 3.50e+02 | 13.4% | 3.7% | 6.94e-03 | 1.63e-04 |
| v | reach 2 (0.01) | 4.85e-01 | 1.60e+01 | 2.29e+03 | 100.0% | 99.9% | 5.45e-03 | 7.83e-04 |
| v | S(0.01) (0.01) | 4.77e-03 | 5.39e-02 | 1.54e+01 | 9.4% | 1.7% | 5.05e-04 | 2.84e-05 |
| v | S(0.03) (0.03) | 8.12e-03 | 2.15e-01 | 1.84e+01 | 8.3% | 2.1% | 2.06e-03 | 6.58e-05 |
| w | reach 2 (0.01) | 4.04e-01 | 1.58e+01 | 2.96e+04 | 100.0% | 98.9% | 9.45e-03 | 9.90e-04 |
| w | S(0.01) (0.01) | 3.89e-03 | 4.55e-02 | 2.43e+01 | 12.0% | 1.6% | 1.07e-03 | 3.67e-05 |
| w | S(0.03) (0.03) | 8.29e-03 | 1.70e-01 | 2.43e+01 | 11.1% | 2.0% | 3.13e-03 | 8.48e-05 |
| p | reach 2 (0.01) | 4.13e-01 | 5.33e+00 | 6.55e+00 | 100.0% | 100.0% | 7.61e-03 | 4.49e-03 |
| p | S(0.01) (0.01) | 4.60e-03 | 9.75e-03 | 1.15e-02 | 0.2% | 0.0% | 1.03e-03 | 5.55e-05 |
| p | S(0.03) (0.03) | 7.02e-03 | 2.95e-02 | 3.60e-02 | 0.7% | 0.0% | 2.48e-03 | 1.49e-04 |
| k | reach 2 (0.01) | 4.26e-01 | 3.58e+01 | 1.41e+02 | 100.0% | 99.9% | 1.26e-02 | 4.63e-03 |
| k | S(0.01) (0.01) | 4.61e-03 | 1.09e-02 | 8.95e-02 | 1.7% | 0.0% | 1.48e-03 | 7.68e-05 |
| k | S(0.03) (0.03) | 7.06e-03 | 4.95e-02 | 2.94e-01 | 3.4% | 0.1% | 2.92e-03 | 1.79e-04 |
| omega | reach 2 (0.01) | 4.27e-01 | 5.34e+00 | 7.57e+00 | 100.0% | 100.0% | 1.07e-02 | 2.43e-03 |
| omega | S(0.01) (0.01) | 4.77e-03 | 9.86e-03 | 2.49e-02 | 0.8% | 0.0% | 7.95e-04 | 3.30e-05 |
| omega | S(0.03) (0.03) | 7.56e-03 | 3.05e-02 | 4.88e-02 | 1.3% | 0.0% | 2.82e-03 | 6.66e-05 |

**bfs3d step 10, start (state-00009)**

| row field | arm (τ) | median | p99 | max | rows > τ | rows > 3τ | max dropped/strongest | dropped/block (Frob.) |
|---|---|---|---|---|---|---|---|---|
| u | reach 2 (0.01) | 4.85e-01 | 1.66e+01 | 1.46e+03 | 100.0% | 99.7% | 1.94e-02 | 4.46e-03 |
| u | S(0.01) (0.01) | 4.71e-03 | 2.72e-01 | 1.69e+01 | 27.0% | 9.0% | 1.02e-03 | 6.22e-05 |
| u | S(0.03) (0.03) | 7.22e-03 | 5.44e-01 | 3.48e+01 | 16.9% | 5.4% | 3.80e-03 | 1.47e-04 |
| v | reach 2 (0.01) | 4.62e-01 | 1.19e+01 | 8.24e+02 | 100.0% | 99.9% | 7.17e-03 | 9.24e-04 |
| v | S(0.01) (0.01) | 4.69e-03 | 2.95e-02 | 3.10e+00 | 5.6% | 1.0% | 5.66e-04 | 2.55e-05 |
| v | S(0.03) (0.03) | 7.74e-03 | 1.00e-01 | 2.88e+01 | 6.2% | 1.1% | 2.22e-03 | 5.91e-05 |
| w | reach 2 (0.01) | 4.14e-01 | 8.16e+00 | 1.44e+03 | 100.0% | 99.3% | 1.01e-02 | 1.71e-03 |
| w | S(0.01) (0.01) | 3.75e-03 | 3.93e-02 | 2.04e+01 | 9.4% | 1.4% | 8.77e-04 | 7.41e-05 |
| w | S(0.03) (0.03) | 7.45e-03 | 1.95e-01 | 2.04e+01 | 10.4% | 1.9% | 2.92e-03 | 1.92e-04 |
| p | reach 2 (0.01) | 3.96e-01 | 5.31e+00 | 6.49e+00 | 100.0% | 100.0% | 8.64e-03 | 3.79e-03 |
| p | S(0.01) (0.01) | 4.49e-03 | 9.70e-03 | 1.37e-02 | 0.3% | 0.0% | 1.03e-03 | 4.67e-05 |
| p | S(0.03) (0.03) | 7.13e-03 | 2.91e-02 | 3.52e-02 | 0.6% | 0.0% | 2.47e-03 | 1.13e-04 |
| k | reach 2 (0.01) | 4.20e-01 | 2.91e+01 | 1.14e+02 | 100.0% | 99.9% | 1.07e-02 | 4.10e-03 |
| k | S(0.01) (0.01) | 4.72e-03 | 1.02e-02 | 3.10e-02 | 1.1% | 0.0% | 1.21e-03 | 6.53e-05 |
| k | S(0.03) (0.03) | 7.26e-03 | 3.38e-02 | 1.68e-01 | 1.6% | 0.1% | 2.35e-03 | 1.36e-04 |
| omega | reach 2 (0.01) | 4.47e-01 | 5.24e+00 | 9.81e+00 | 100.0% | 99.9% | 1.21e-02 | 1.33e-03 |
| omega | S(0.01) (0.01) | 4.71e-03 | 9.77e-03 | 3.21e-02 | 0.6% | 0.0% | 7.58e-04 | 1.76e-05 |
| omega | S(0.03) (0.03) | 7.68e-03 | 2.98e-02 | 4.01e-02 | 0.9% | 0.0% | 2.36e-03 | 3.23e-05 |

**pitzDaily step 5, start (state-00004)** (2D: no w)

| row field | arm (τ) | median | p99 | max | rows > τ | rows > 3τ | max dropped/strongest | dropped/block (Frob.) |
|---|---|---|---|---|---|---|---|---|
| u | reach 2 (0.01) | 2.01e-02 | 1.45e+00 | 5.04e+02 | 67.7% | 39.3% | 1.20e-02 | 1.30e-03 |
| u | S(0.01) (0.01) | 1.20e-02 | 7.40e-01 | 5.04e+02 | 53.9% | 26.5% | 5.26e-03 | 3.45e-04 |
| u | S(0.03) (0.03) | 1.61e-02 | 1.34e+00 | 5.04e+02 | 33.8% | 14.6% | 1.20e-02 | 6.12e-04 |
| v | reach 2 (0.01) | 3.26e-03 | 4.95e-01 | 2.87e+01 | 17.6% | 9.7% | 3.01e-03 | 4.04e-04 |
| v | S(0.01) (0.01) | 2.44e-03 | 4.72e-02 | 3.96e+00 | 5.2% | 1.4% | 2.22e-03 | 1.77e-04 |
| v | S(0.03) (0.03) | 2.96e-03 | 8.68e-02 | 2.87e+01 | 3.9% | 1.0% | 2.22e-03 | 1.92e-04 |
| p | reach 2 (0.01) | 3.75e-03 | 3.93e-01 | 1.50e+01 | 21.3% | 8.6% | 6.70e-03 | 1.35e-03 |
| p | S(0.01) (0.01) | 2.36e-03 | 9.93e-03 | 2.86e-02 | 0.9% | 0.0% | 3.56e-03 | 3.25e-04 |
| p | S(0.03) (0.03) | 3.26e-03 | 2.76e-02 | 4.23e-02 | 0.6% | 0.0% | 6.39e-03 | 5.99e-04 |
| k | reach 2 (0.01) | 3.76e-03 | 4.08e-01 | 9.10e+00 | 21.9% | 9.0% | 6.71e-03 | 1.46e-03 |
| k | S(0.01) (0.01) | 2.37e-03 | 1.03e-02 | 1.78e-02 | 1.6% | 0.0% | 3.73e-03 | 2.86e-04 |
| k | S(0.03) (0.03) | 3.27e-03 | 2.95e-02 | 7.43e-02 | 0.9% | 0.0% | 6.39e-03 | 5.63e-04 |
| omega | reach 2 (0.01) | 3.69e-03 | 4.14e-01 | 3.30e+00 | 24.0% | 9.5% | 6.70e-03 | 1.02e-03 |
| omega | S(0.01) (0.01) | 2.54e-03 | 1.29e-02 | 2.56e-02 | 4.7% | 0.0% | 4.65e-03 | 1.20e-04 |
| omega | S(0.03) (0.03) | 3.34e-03 | 3.87e-02 | 1.07e-01 | 2.1% | 0.0% | 6.39e-03 | 2.69e-04 |

A "0.0%" share means fewer than 0.05 % of rows, not none: wherever the max exceeds 3τ, a handful of rows
lie above it.

### Why the momentum rows exceed the bar: the weakest face coupling is the wrong scale there

The ratio of each row's weakest to its strongest face coupling in the same block (median / p1 / min):

| mesh, state | u | v | w | p | k | omega |
|---|---|---|---|---|---|---|
| bfs3d step 5 inner 2 | 1.6e-2 / 8.2e-4 / 8.9e-7 | 1.0e-3 / 8.1e-8 / 1.5e-10 | 2.9e-3 / 1.2e-6 / 8.7e-10 | 1.5e-2 / 8.7e-4 / 6.2e-4 | 1.5e-2 / 1.4e-4 / 3.5e-5 | 1.4e-2 / 7.4e-4 / 4.3e-4 |
| pitzDaily step 5 start | 4.3e-2 / 8.0e-4 / 8.0e-7 | 9.3e-3 / 9.7e-6 / 3.6e-8 | — | 0.35 / 9.5e-3 / 2.7e-4 | 0.35 / 9.2e-3 / 4.4e-4 | 0.34 / 8.8e-3 / 2.8e-4 |

A momentum row reads pressure first through `V ∇p`. The coupling to a face neighbour across a face
parallel to that velocity component is a gradient correction, which is near zero on an orthogonal cell.
So in a momentum row the weakest face coupling has a floor near zero (down to 1e-10 of the strongest,
against 6e-4 for the continuity rows), and any ratio over it can be arbitrarily large. Read against the row's
strongest entry instead, the momentum rows' largest dropped entry under S(0.01) is 5e-4 to 3.2e-3 on
bfs3d. The p rows' is 1.0e-3 and k/ω's is 0.8–1.5e-3, so the momentum drop is within about 3× of them.
Momentum's Frobenius drop is 3–7e-5 of its block, against 4.7e-3 under reach 2. On pitzDaily the u rows
exceed the bar mostly in near-square cells (1–2:1 bin, 67 % above τ under S(0.01)). There S keeps almost
nothing, and the explanation is the same structural zero.

On the same normalization, pitzDaily's momentum rows also exceed the bar under its **shipped** all-reach-2
probe (u: 39 % of rows above 0.03, max 504), and that probe leaves its march unchanged (same 18 steps /
123 cycles / x_r/h 8.0686 as uniform reach 3, measured 2026-10-14, recorded in its `case.yaml`). That is
indirect evidence that the momentum-row ratio does not predict harm. It does not prove it for bfs3d.

This explanation fits v and w well, but u less cleanly. u's median weakest/strongest (1.6e-2 on bfs3d)
is close to the p rows' (1.5e-2), so the tail of u rows above 3τ (7–10 % on bfs3d) is not shown to be an
artifact. u's dropped/strongest is also up to 3× the p rows', and at S(0.03) up to 6.9e-3. Whether those
entries matter is answered only by the replay below and, eventually, by the march ensembles.

### What the real Jacobian says about the proxy (risks 1, 2 and 6)

- **Continuity rows (p)**: the proxy's per-row guarantee carries over to the real Jacobian within about
  1.4× on bfs3d and 2.9× on pitzDaily. Under S(0.01) the largest dropped entry is at most 1.15–1.37e-2 of the weakest face coupling on bfs3d
  and 2.9e-2 on pitzDaily. Under S(0.03) it is at most 3.7e-2 on bfs3d and 4.2e-2 on pitzDaily. This holds at the
  real state's `V / a_P` (risk 2), with the real boundary closures (risk 6), at four states from the
  step's start to step 10, both τ and both meshes.
- **k and ω rows**: they read p through the same mass flux, and S holds their strong entries. p99 is at
  or below about τ under S(0.01) everywhere and below 3τ under S(0.03). The max exceeds 3τ in a few k rows
  on bfs3d (8.95e-2 at τ = 0.01 at step 5 inner 2; 0.29 at τ = 0.03), at most 0.1 % of rows.
- **Momentum rows (risk 1)**: S does not hold every momentum row's strong distance-3 entries, judged by
  the pre-registered scale. See the section above for how much of that is the scale.
- **Stability over the march**: step 5 start, step 5 inner 2 and step 10 give the same numbers to within
  their spread. A static skeleton is consistent with that, but four states are not a proof.

### The fold the pattern probe adds

The pattern probe also sums each dropped coupling onto the kept entry of its row that shares its colour.
On bfs3d step 5 inner 2 under S(0.01), the change to kept entries is 2–5e-5 of each block's norm, against
2–4e-3 under reach 2. Per row in the p, k and ω blocks, the largest change is at most 1.6e-2, 8.9e-2 and
3.1e-2 of the weakest face coupling. Other blocks and states are in the logs.

## 2. Replay: bfs3d inner solves, operator exact, preconditioner fitted under each probe

Every arm floors the same rows (32 `k` rows at step 5, 106 at step 6). Cycles are restart cycles run
(the residual stop's count); residuals are row-scaled, relative.

| state | arm | probes per fit | march rtol: cycles | march rtol: residual | tight (16 cycles): residual |
|---|---|---|---|---|---|
| step 5, inner 2 | `3` (uniform reach 3) | 576 | 6 | 2.95e-1 | 1.35e-4 |
| step 5, inner 2 | `p` (p at 2) | 519 | 16 (cap) | 2.48e+0 | 2.48e+0 (stagnant) |
| step 5, inner 2 | `p@fold` (fold only) | — | 6 | 2.65e-1 | 1.34e-4 |
| step 5, inner 2 | `p@drop` (drop only) | — | 14 | 2.98e-1 | 8.74e-2 |
| step 5, inner 2 | **`p@S0.01`** | 528 | **6** | 2.97e-1 | **2.34e-4** |
| step 5, inner 2 | **`p@S0.03`** | 526 | **6** | 2.95e-1 | **1.17e-4** |
| step 6, inner 1 | `3` | 576 | 5 | 2.85e-1 | 2.61e-4 |
| step 6, inner 1 | `p` | 519 | 6 | 2.83e-1 | 4.77e-3 |
| step 6, inner 1 | **`p@S0.01`** | 528 | **5** | 2.88e-1 | **3.76e-4** |
| step 6, inner 1 | **`p@S0.03`** | 526 | **5** | 2.85e-1 | **5.07e-4** |

`p@fold` / `p@drop` re-fit from a mix of two probes, so they have no probe count of their own. All four
uniform/`p` arms re-establish the record on this tree, but the numbers moved since the proposal's `1c5a187`.
Uniform reach 3 now takes 6 cycles at step 5 inner 2, not 8, and its tight residual is 1.35e-4, not 7.3e-4.
The fold-only arm is now 1.34e-4, not about 2e-3. `p` alone still stagnates there, and the drop alone
still costs it.

Every arm's first line agrees to the third digit, ladder included (`option1_audit_logs/`), so the pattern
probe moves nothing the march would see in these two solves.

## Verdict per pre-registered bar

| bar (from the proposal) | result | verdict |
|---|---|---|
| Static: every row field's dropped ratio at or below about 3τ | p and ω: yes at both τ, every state, both meshes (p max ≤ 1.4τ on bfs3d and 2.9τ on pitzDaily at τ = 0.01; ω max 3.6τ, "about"). k: p99 yes; max 3–10τ in < 0.1 % of rows on bfs3d. **u, v, w: no** — 1–10 % of rows above 3τ on bfs3d, up to 27 % on pitzDaily | **FAIL as registered**, on the momentum rows (and marginally k's max). The failure is in large part an artifact of the normalization for v and w; for u it is not shown to be one |
| Replay: ≤ 9 raw cycles at step 5 inner 2 (uniform took 8; now 6) | 6 at both τ; also equal to uniform at step 6 inner 1 (5) | **PASS** (also within the re-established uniform count) |
| Replay: tight residual ≤ 2e-3 at step 5 inner 2, no worse than the fold-only arm | 2.34e-4 (S(0.01)), 1.17e-4 (S(0.03)); fold-only on this tree 1.34e-4 | **PASS** against 2e-3. Against the re-established fold-only arm, S(0.03) is no worse and S(0.01) is 1.7× above it. That ordering (the smaller drop doing worse) and uniform's own 1.35e-4 say the arms are not separated at this level |
| Colour counts of R2 ∪ S(τ) (to report) | bfs3d 46 / 48, pitzDaily 20 / 21 at τ = 0.03 / 0.01 | as the proposal said |

**Summary.** Option 1 does what it claims for the continuity, k and ω rows, on the real Jacobian. The
replayed bfs3d solves under a p probe coloured for R2 ∪ S(τ) cannot be told apart from uniform reach 3,
at the solve that stagnates under a reach-2 p probe. The pre-registered static bar fails on the momentum
rows. Most of that failure traces to a normalization that is structurally zero in momentum rows, but for
u that is not established. The project owner should decide whether that bar should have been posed
against a different scale before reading this as a pass or a fail of option 1.

## What these measurements do not establish

- **The proposal's target configuration was not replayed.** Every replay probes the non-pressure columns
  at reach 3, as specified. The probe saving the proposal is after (243 probes per fit against 576) needs
  the rest at 2 as well. Each of k, ω and velocity alone at 2 left these solves unchanged in the record,
  but the combination with p at S(τ) is untested.
- **No march.** Two inner solves is not a march: a bfs3d march is one draw from a wide spread, and the
  proposal's own risk 4 (t = 0.25 of the fold cost 8 → 13 cycles) shows a small drop can matter at some
  states. The hardest solves of this march (step 15: 14 cycles) were not replayed.
- **Momentum rows.** Whether the u entries above 3τ (7–10 % of rows, at most 7e-3 of the row's strongest
  entry) matter is not settled statically.
- **pitzDaily's coupled residual reaches distance 4** (its `case.yaml`), so its uniform-reach-3 Jacobian
  already folds distance-4 couplings onto entries within reach 3. The audit sees only distance 3, so its
  pitzDaily numbers are of that folded matrix.
- **One τ per bin is not a calibration.** τ = 0.01 and 0.03 were taken from the proposal and not swept.
  The replay does not separate them.
- **Precision.** The audit is float64. The replays use the shipped float32 split; their tight residuals
  after 16 cycles differ by less than the spread between arms that should agree.
- **Cost.** No timing here is a measurement (the fit times include compilation, one run each).

## What this changes for the proposal's §9 design

- **The skeleton needs no state.** The geometric `V / a_P` stand-in and the zero-value boundary
  closure already give a set that holds the real Jacobian's strong continuity, k and ω entries. A
  skeleton built by probing `interior_mass_flux` would be a refinement, not a necessity, for these rows.
- **The pattern-to-plan step is small and library-shaped.** `pattern_probe_plan` (the reach-3 stored
  pattern, one column read through a membership mask) is the whole plan half of §9's pattern/colouring
  split. The harness reached the library only through its private seams: it installed the probe as the
  session's `_probe`, and used `_CoupledNarrowing` and `_saturation_colouring`. No library change was
  needed for the replay.
- **If a static criterion must cover the momentum rows**, a per-row threshold against the weakest face
  coupling is the wrong form for them. A row scale that is not structurally zero (the strongest entry,
  or the sum of face couplings) would be the thing to pre-register for the next audit. The project owner
  decides that, not this audit.
