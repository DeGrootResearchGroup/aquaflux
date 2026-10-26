# Open directions — making the fluid solve faster

> The forward-looking sibling of
> `solve-refuted-directions.md`. It lives in `.claude/notes/` (tracked, greppable, **never auto-loaded**)
> for the same reason the refutation ledger does: it is read on demand before proposing work, not paid for
> by every session. Each entry below was checked against the refutation ledger, the `solve-*` rules and
> the `-log.md` files on 2026-10-07; none of them is recorded as tried.
>
> **Status discipline.** An entry here is a hypothesis with a pre-registered measurement. When one is
> measured it moves: a win becomes a rule entry (and a default, if it earns one), a loss becomes an entry
> in `solve-refuted-directions.md` with its configuration recorded in full. Nothing stays here after it
> has been run. Open one GitHub issue per entry *when its investigation starts*, linking back to this
> file, so the issue tracker carries the work in flight and this file carries the backlog.

## Where the time is — the reading these entries rest on

The recorded numbers (`solve-march.md`, `solve-march-log.md`, `solve-refuted-directions.md`):

| case | cells | dofs | wall | compile | Krylov matvec |
|---|---|---|---|---|---|
| pitzDaily coupled RANS | 12 225 | 61 125 | 304 s cold before entry 15; back to back 265 s without it, 220 s with it, warm (2026-10-25, below) | ~100 s cold | JVP 27–34 ms, split 78–86 ms, 33 ms with entry 15 (standalone, below) |
| bfs3d coupled RANS | 23 040 | 138 240 | 1635 s warm, 1749 s cold before entry 15; back to back later that day 1853 s without it, 1274 s with it (2026-10-25, below) | ~116 s warm | JVP 142–147 ms, split 525–576 ms, 206 ms with entry 15 (standalone, below) |

The matvec actually run is `jax.jvp` through the full residual, which reconstructs nine gradients per
evaluation on pitzDaily. Measured 2026-10-07 on pitzDaily (4-core Linux, jax 0.11.2): the jvp is 44 ms,
while a SciPy matvec on the materialized `J` (7.4M nnz, ~120 per row) is 6.5 ms, about 7× cheaper.
⚠️ **That 44 ms predates the adjugate first pass (2026-10-16), and it is a standalone call that pays
~11 000 page faults** (re-measured 2026-10-17, `main` `183e9ffa`, same container class). The standalone
jvp is now **16.5 ms** with the glibc default allocator, and **9.8–11.1 ms** with the faults removed,
which is the cost a matvec inside the compiled step program sees (the march's wall does not move when
they are removed). The "7× cheaper" ratio is therefore now nearer 1.5–2.5×. See
`.claude/rules/discretization.md`, "Single-threaded kernels". bfs3d's
`J` is 47.2M structural nnz (~340 per row). The matrix is not used for the matvec because *materializing*
it costs 165–564 batched JVPs (`sparse_jacobian.py`, coloured probing), which is why the preconditioner is
refreshed only every few steps — and the ledger already found that **staleness, not hard operators, is the
dominant linear-solve cost at low β** (the `InnerIterateCheckpointer` result: 15 cycles stale vs 1 cycle
matched at the march's hardest solve).

So the two costs are coupled: the matvec is expensive because the matrix is expensive to build, and the
preconditioner is stale because the matrix is expensive to build. Entries 1–2 tried to break that
coupling and are closed (below): the coloured probe is already near the forward-mode floor. 4–6 attack the
*number* of linear solves rather than their cost; 7–9 are platform and sweep-level levers; 11 attacks the
number of OUTER steps the target station takes, which its shift floor sets (measured and built opt-in;
31 -> 22 steps on pitzDaily).

---

### Re-measured 2026-10-17 by profile, on the shipped march

pitzDaily `case.yaml` at `3c5b58d6` (`main` after #721, #732, #735, #736, #742 and the compiled norm and
field extraction), Linux x86_64 container, 4 cores, 15 GB, no swap, Python 3.13.16, jax/jaxlib 0.11.2;
`PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1`. **21 steps / 50 inner solves / 106 restart cycles /
1258 applications of the preconditioned operator / 21 refits (13 `full`, 8 `cost`), final stopping `R`
2.618e-06, `x_r/h` 8.0686**, identical in every run below. Each column is one run of the whole process
under `py-spy record --idle --threads -r 25` (main thread only), sorted by
`validation/pitzdaily_openfoam/profile_buckets.py`, which assigns a stack to the first rule any of its
frames meets (compilation first, so a refit's first compile counts as compilation):

| bucket | cold s | share | warm s | share | was (`da37958a`, cold, 898 s) |
|---|---|---|---|---|---|
| the compiled step program (GMRES, preconditioner applies, JVPs) | 106 | 32 % | 110 | 38 % | 417 s, 46 % |
| compilation and tracing, all phases | 115 | 35 % | 73 | 25 % | 212 s, 24 % |
| refit: coloured probe (~1.5 s per refit) | 42 | 13 % | 41 | 14 % | 95 s, 11 % |
| refit: hierarchy refactor (host) and other build, the first fit included (predates the entry maps, below) | 48 | 14 % | 49 | 17 % | 56 s, 6 % |
| the step's row-scaled measure rebuilt from each step's state | 2 | 1 % | 2 | 1 % | (not a row) |
| march-log field extraction | 0 | 0 % | 0 | 0 % | 35 s, 4 %, op by op |
| the stopping test's residual | 0 | 0 % | 0 | 0 % | 32 s, 4 %, op by op |
| driver, setup, report | 17 | 5 % | 15 | 5 % | 50 s, 5 % |
| **whole process** | **329** | | **290** | | 898 |

**What the compile row contains (2026-10-17, `9efc5759`, same container and settings, shipped path, two
warm runs logged with `JAX_LOG_COMPILES=1` and read by `compile_timeline.py`'s `breakdown`, one warm run
under the profiler):** all of it is before step 1. **After step 1 the march compiles, reads from the
cache, traces and lowers nothing** -- 0 programs across the 21 refits (13 `full`, 8 `cost`) and the 12
stations, cold and warm, with or without the capture variables; only the 3 tiny post-march reads follow.
Before step 1, warm: 977 small programs compiled (26.2 s; the persistence floor's, owned elsewhere), 6
programs read from the cache (3.9 s, `_march_step` 2.25 s), tracing 15.4 s (`_march_step` 7.5,
`batched_jacobian_matvec` 2.6, three initialization `newton_step`s 2.0, `norm_of_residual` 1.6,
`_eddy_viscosity_drift` 0.7, `_field_values` 0.6) and lowering 7.5 s (`_march_step` 4.0) -- 53 s, every
program traced once. Under the profiler the bucket read 47 s (attached ~10 s after launch); the 17.6 s it
put under `newton_march`'s step call is step 1's own trace, lowering and cache read. So the "~35 s
unexplained" read off the 73 s warm column is not recompiles, and did not reproduce: what remains is the
small programs and tracing the five large ones, neither per refit nor per station. The pin that a refit
and a station hand the march an already-compiled step:
`test_a_traced_split_step_is_one_compiled_program_across_refits_and_stations`
(`tests/unit/test_preconditioner_session.py`). Warm walls in those runs: 224 s (log on), 238 s (profiled),
218 s (log on, capture variables set), all 21 steps to `R` 2.618e-06.

**Since #649 (2026-10-18)** `_march_step` traces in 6.2 s and compiles in 19.0 s from cold (7.7 / 23.1 s at `cf80be32`; HLO 60,970 → 48,377 lines): the step is handed the residual at its start, so the dual-time anchor's `R(φⁿ)` copy is gone. Configuration and the full-march pair: `solve-march-log.md`, "#649".

The march log's solve wall was 320 s cold and 284 s warm under the profiler. Without it, the shipped path
(`compare.py` with no override) took **249, 227 and 236 s** in three warm runs on the same container,
spread over 1.5 hours, the first with the capture variables set. So run-to-run spread on this container
is ~5-10 %, well above the ~1.3 % recorded elsewhere for back-to-back runs, and the profiler adds roughly
another 10-20 %.
The application count comes from `PITZ_KRYLOV_STOP=residual`, which goes through `compare.py`'s
script-only study path (the shipped path reports no count). Its step table was identical to the
shipped run's, so the count is the shipped march's. The refit seconds are the march log's own `pc`
lines (warm, shipped path, 21 refits): 61 s, probe 32 s, refactor 27 s.
**What moved, read from the table:** the compiled step program is now about a quarter of what it was
(417 -> 110 s, while the march grew from 18 to 21 steps), so a cold run is dominated by compilation and
by the refits (62 % of the cold process together). A refit's probe is ~1.5 s, against ~6 s before the
adjugate inverse (#735), which is what a cheaper JVP predicts for ~165 batched JVPs (not separated further).

Inside one Krylov application, timed by `validation/pitzdaily_openfoam/application_cost.py` at the
first inner system of steps 19, 20 and 21 of that capture (the target station; `replay.py` reproduces
the march's refits). Each piece is the best of 15 warm calls of its own compiled program, run with
`MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=2147483647`. Under the default allocator
the JVP's 42.8 MiB temporary buffer and the leading cycle's both page-fault on every standalone call
(11 200 and 9 000 faults), which the step program does not pay; those figures were JVP 22-27 ms and
split apply 53-61 ms:

| piece | ms | was (`da37958a`) |
|---|---|---|
| field-split apply, float32 (as shipped) | **48-50** | 186-190 (float64, CSR) |
| -- the `[u, v, p]` SIMPLE-smoothed V-cycle | 36-39 | ~130 |
| -- the `[k, omega]` Jacobi-smoothed V-cycle | 8-9 | ~55 |
| -- the coupling product | 1.5-1.9 | ~7 |
| JVP `J v` | **14-17** | 77-86 (with the batched LU, #735) |
| the two compiled as one program, `(J + s) M v` | 78-81 (19 000 faults per call under both settings: a ~74 MiB buffer, see below) | -- |

So **the preconditioner is ~75 % of an application (apply / (apply + JVP), 74-78 %), not ~90 %**. The
split's cost fell by ~3.8x (padded-row operators, #721, and float32, #732) while the JVP fell by ~5x
(the adjugate inverse, #735), so the share barely moved from the ~70 % recorded before. It would be
~93 % only if the split had stayed at its CSR float64 cost: arm C below measures exactly that, 195-200 ms
against a 14-18 ms JVP. The composite's extra ~15 ms is its allocator, not its work (next section).

### Inside the step program (2026-10-18): no unaccounted cost; the composite's excess was the allocator

Same case, counts and capture as above, at `main` `9efc5759`, on a DIFFERENT 4-core Linux container
(Xeon @ 2.10 GHz, 15 GB, glibc 2.39, jax 0.11.2), where every piece runs ~1.4x faster than in the table
above, so compare only within this section. Warm cache throughout. Instruments:
`validation/pitzdaily_openfoam/step_program_timing.py` (the march with its compiled step wrapped by a
blocking timer, under `PITZ_KRYLOV_STOP=residual`; compile-free calls only) and `krylov_overhead.py` (the
step's own `solve_linear` over `residual_stop_gmres`, restart 15, row-scaled measure, on the first inner
system of steps 19-21; budgeted arms run exactly 60 iterations; best of 7). jemalloc is Ubuntu's
`libjemalloc.so.2` with `MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1`.

| ms per application | default glibc | jemalloc |
|---|---|---|
| the step program, per application its solves made (20 compile-free calls, 1247 applications) | **60.1, 56.8, 57.3** (3 marches) | 58.0 |
| -- its minor faults | ~41 000 per CALL (558-623 per application) | 40 per application |
| the march's own solve at the march's tolerance (`krylov_overhead` `march`) | 48-53 | 46-53 |
| -- GMRES bookkeeping per iteration (diagonal stand-in operator) | **0.83-1.26** | 0.84-1.32 |
| -- JVP per iteration (identity arm minus stand-in) | 8.4-10.6 | 9.1-11.3 |
| -- preconditioner per iteration (full minus identity; `s M` arm minus stand-in) | 37-45; 31-39 | 34-45; 30-35 |
| standalone `(J + s) M v`, mean of 60 back-to-back calls | 63-65 (19 145 faults per call) | **41-47** (0) |
| standalone `M v`, `J v`, same statistic | 34-44, 12-22 | 32-33, 11-12 |
| standalone, best single call (`application_cost.py`): apply, JVP, composite | 39-44, 15-18, 58-63 | 28-29, 10, 37-46 |

What this settles:
- **A Krylov iteration inside the loop costs what the standalone composite costs once allocation is taken
  out** (46-53 against 41-47 under jemalloc). The "15-25 ms unaccounted" was two artefacts: the standalone
  composite maps a ~74 MiB temporary afresh on every call (19 145 faults), and best-of-15 single calls
  understate a mean (`M v` best 28-29, mean 32-33). The step's loop allocates once per SOLVE, so its
  faults are per call, not per application, and jemalloc removing them changes the in-loop time by nothing
  measurable.
- **Why the thresholds did not remove the composite's faults**: XLA allocates on its worker threads, whose
  glibc arenas are heaps of at most 64 MiB (`HEAP_MAX_SIZE`), so a larger request is `mmap`ed whatever
  `MALLOC_MMAP_THRESHOLD_` says. Reproduced with bare `malloc`/`free` from a Python thread: 43 MiB faults
  every call by default and not with the thresholds raised; 80 MiB faults every call (20 481) with them
  raised, and not under `MALLOC_ARENA_MAX=1` or jemalloc. (On the main thread a 1 GiB threshold DOES
  apply in glibc 2.39; the 32 MiB cap bounds only the adaptive threshold.)
- **GMRES bookkeeping is ~2 % of an application** (0.8-1.3 ms against ~50), so idea 2b (row scaling up
  front, one projection pass, Givens) is bounded at ~1.3 s per march and was not built (ledger:
  "Cheaper GMRES bookkeeping in `_ResidualStopGMRES`").
- **The step program's application (57-60 ms) is ~7-10 ms above its solves' (48-53)**: the line search's
  residuals and measures, the per-solve `x = M y` and the inner loop's own residuals, spread over the
  applications. Not decomposed further; at ~10 s per march it is the only residue left in this bucket.
- **The allocator does not move the march.** Shipped `compare.py`, warm, identical step tables and
  `x_r/h` 8.0686: default 206 s (its step 1 took 68 s against 47-49 in every other run, an outlier) and
  176 s; jemalloc 179 and 169 s. Summed over the 21 refits, the coloured probe took 21.9-24.4 s by default
  and 16.5-18.7 s under jemalloc (its batched-JVP program's temporaries stop faulting), the host refactor
  20.4-21.9 s against 22.7-23.5 s, so the refits gain ~3-4 s net, inside the 5-10 % spread. Not made a
  `run_case.sh` default: it would buy ~2 % and make every timing incomparable with the record.
  `run_case.sh` records `LD_PRELOAD` and `GLIBC_TUNABLES` beside `MALLOC_*` since this change.
  ⚠️ An allocator that keeps freed memory raises the footprint: `MALLOC_ARENA_MAX=1` with both thresholds
  took a bfs3d march to 12.3 GiB resident by step 2 (9.5 GiB at defaults) and its 15 GB container
  restarted, so on bfs3d such a setting is for fixed-state probes, not marches.

**The refactor after the entry maps (2026-10-17, `solve-amg-multigrid-log.md` "Refit by entry maps").**
Not re-profiled by bucket. The march log's `pc` lines, same case and counts, on a different 4-core
container from the table's (base `main` 9efc5759): refactor 22.7 / 21.4 s on `main` (two warm runs)
against 14.8 s with the maps as committed (15.2-17.0 s in two runs of an earlier version), the first
refit paying ~1.5-2.5 s more to build them and every later one ~0.5-0.7 s against ~0.9-1.1 s, for
~0.2 GB more host memory. The refactor row above predates this.

**What each probe tangent computes WAS a lever, and is spent (2026-10-20).** Each column field now has its
own program that differentiates the residual in that field alone (`solve-direct-preconditioners.md`, "One
program per column field"): bfs3d's probe per re-fit 43.1 → 18.5 s, pitzDaily's 1.60 → 0.94 s, at the
hybrid start on a 4-core Xeon (validation-log, "One probe program per column field"). What is left of it is
per-column precision (probe the pressure column in float64, the rest in float32), the owner's next step
and unmeasured; the seam is `FieldwiseMatvec.seed_dtype`.

⚠️ **The probe's PATTERN is not a lever**: the colouring, the CSR structure and its gather map are
built once per session and reused by every refit. A refit's probe (~1.5 s since #735, ~6 s before) is
the ~165 JVPs that compute the new VALUES; keeping old values longer is what `refresh_on_cycles` already trades (2 / 3 / 4 within 7 % on pitzDaily; on bfs3d on 4-core Linux with the pivot floor, where a re-fit costs ~2.5 steps, 4 took 0.59x the solve of 3 and 5 1.18x, one sample each, 2026-10-19).

### bfs3d against pitzDaily on one machine (2026-10-25): what the extra time is

Container: Linux x86_64, 4-core Xeon @ 2.1 GHz, 15 GB, no swap, L3 260 MiB; Python 3.13.17, jax/jaxlib
0.12.0.1, numba 0.68.0, glibc defaults unless an arm says otherwise; `main` `d7f5bf28`; both shipped
`case.yaml`s (float32 split, `PivotFloor(0.15)`, reach 3, bfs3d `refresh_on_cycles` 4); nothing else running.

**The marches** (`validation/run_case.sh <case>/compare.py`; the seconds are the march log's own `t(s)`
column and `pc` lines):

| | pitzDaily (cold) | bfs3d (cold) | bfs3d (warm) | ratio (warm bfs3d / pitzDaily) |
|---|---|---|---|---|
| steps / cycles / re-fits | 21 / 102 / 21 | 17 / 147 / 14 | 17 / 147 / 14, identical table | |
| set-up through step 1 | 137 s | 353 s | 239 s | 1.7× |
| re-fits (probe / refactor) | 34 s (19 / 13) | 411 s (295 / 113) | 401 s (286 / 112) | 12× (18× per re-fit) |
| steps 2–end, outside the re-fits | 112 s over 99 cycles | 960 s over 142 cycles | 970 s | 8.7× (6.0× per cycle) |
| solve | 304 s | 1749 s | 1635 s | |

bfs3d's final `R` 3.672e-06, `x_r/h` 8.36 mid-span. The warm run's JAX work (from `JAX_LOG_COMPILES=1`): 1319
compilations and 14 cache reads summing 50 s, tracing 49 s, lowering 17 s, all before step 1 (the largest
compilation `_march_step` 4.7 s). Under `py-spy --idle --threads --nonblocking` the cold run sorted (by
`profile_buckets.py`) into step program 761 s, coloured probe 301 s, other re-fit build 147 s, compilation and
tracing 180 s; ~250 s fell into `driver` under a stack (`clean_up`/`clear_backends`) the wall clock shows did
not happen (the process exited 6 s after the march ended), so `--nonblocking` mis-reads stacks here: use the
march log's columns, not the capture, for bfs3d's split.

**So the excess over the unknowns' 2.26× is per-unit cost, not counts**: bfs3d takes fewer steps and re-fits
than pitzDaily and 1.44× the cycles. Two per-unit costs carry it, each set by the stencil the reach-3
pattern stores in 3D (63 cells on hexahedra against 25 in 2D; `solve-refuted-directions.md`, "TRUNCATING the Jacobian"):

- **One Krylov application is 6.9× pitzDaily's** (`validation/application_scaling.py`, hybrid start, mean
  of 30 standalone calls, glibc defaults, so each standalone figure carries page faults the step's loop does
  not pay):

  | piece | pitzDaily ms | bfs3d ms | ratio | bfs3d cores |
  |---|---|---|---|---|
  | residual | 11–12 | 72–78 | 6.5× | 2.7 |
  | JVP | 27–34 | 142–147 | 4.8× | 3.0 |
  | split apply | 78–86 | 525–576 | 6.7× | 2.3 |
  | — leading `[u, v, w, p]` cycle | 57–59 | 343–365 | 6.1× | 2.2 |
  | — trailing `[k, ω]` cycle | 20–21 | 143–148 | 7.0× | 2.5 |
  | — coupling product | 5–6 | 34–35 | 6.2× | 2.9 |

  The split is ~78 % of an application on bfs3d. Its cost follows the stored entries: the leading fine
  level holds 21.0M entries (mean row 228, widest 252 = 63 cells × 4 fields) against 2.68M (mean 73) on
  pitzDaily, 7.8×, and the leading cycle costs 16–21 ns per fine-level entry on both cases. The trailing
  hierarchy is 12 levels on bfs3d (5.2M fine entries; its last five levels coarsen 288 → 178 dofs), 2 on
  pitzDaily.
- **One re-fit is 18× pitzDaily's**: the probe 20.4 s against 0.89 s, 576 colours against 100 (5.8×, reach 3
  in 3D) times ~4× per colour; the refactor 8.0 s against 0.64 s.

**Inside the split apply** (`fusion_profile.py`, `FUSION_PROGRAMS=split`, bfs3d hybrid start, best of 15
untraced calls 478 ms on 4 cores, 989 ms on one, parallel efficiency 52 %): 59 % of the one-core time is
gather fusions (`values * x[columns]` written out whole: the leading fine level's is `f32[23224320]`, 177 MiB,
two per apply), 37 % the row-sum custom fusions that read it back (`ynn_fusion`, never partitioned), the rest
negligible. One kernel is in flight for 74 % of the 4-core span: a V-cycle is a chain, and each memory-bound
gather is split into at most `ceil(sqrt(4))` = 2–4 parts. `--xla_cpu_experimental_ynn_fusion_type=-reduce`
on Linux: split 525 → 586 ms (leading 343 → 482, trailing 143 → 107), not a lever, as recorded for macOS.
**Built since: entry 15** (the split apply 576 → 206 ms on bfs3d, the march −31 %).

## 1–2. CLOSED — local Jacobian assembly and the materialized-`J` matvec

Both were measured on 2026-10-07 and moved to `solve-refuted-directions.md` ("Local (staged) AD assembly
of the coupled Jacobian…"). Local assembly gains ~1× to ~1.3× at best, not 15–30×, so the per-iterate
build item 2 relied on is not cheap. The numbering below is kept so cross-references stay valid.

## 3. CLOSED — tangent / secant predictor between continuation stations

Measured on 2026-10-07 and moved to `solve-refuted-directions.md` ("Tangent / secant predictor between
continuation stations"). It was already measured on the old ladder (worse at a decade). On the shipped
one-step-per-station ramp an exact predictor changes the starting residual by −2.6 % to +12 %, for one
extra linear solve per station.

## 4. CLOSED — Eisenstat–Walker forcing terms; replaced by 4b

Measured 2026-10-08 and moved to `solve-refuted-directions.md` ("Eisenstat–Walker adaptive forcing
terms"): tightening the inner tolerance buys no nonlinear iterations on pitzDaily, and the schedule would
tighten most of the time. The gain is the loose end, which needs no schedule.

## 4b. Loosen the fixed inner forcing term from 0.3

**Deferred — tracked in #637** (needs bfs3d and a robustness check before any default moves).

**What.** Raise `LinearSolveSettings.rtol` (the `_VCYCLE_LINEAR_SOLVE` / `VCYCLE_LINEAR_SOLVE` family
value, and the case files' `linear_solve.rtol`) from 0.3 to ~0.6.

⚠️ **Under the residual-only stop of 6b the gain shrinks to 13 % of wall, with 2 extra steps and a
retry** (see 6b). The figures below are under `lineax`'s stop.

**Why.** On pitzDaily, 0.6 took 166 restart cycles against 203 (−18 %), 0.9 took 161, with identical steps,
inner-solve counts and `x_r/h` (table in the ledger entry above). 0.3 was calibrated on bfs3d's
multigrid family and carried to the others.

**Pre-registered measurement.** bfs3d at `BFS3D_FORWARD_RTOL` 0.3 / 0.6 / 0.9 (one case at a time):
steps, cycles, retries, refreshes, wall, mid-span `x_r/h`. A pass is ≥ 10 % fewer cycles with no extra
steps or retries and the same `x_r/h`; then a library-default change needs the project owner. Watch the
retry ladder: a looser direction near the low-β wall could trip `abort_above_cycles` or the line search.

**Measured on bfs3d under the residual stop instead (2026-10-12, #699):** #699 replaced the `lineax`-stop
sweep above with 0.1 / 0.3 / 0.6 under `stop: residual` at `refresh_on_cycles` 4 (tied with 3 in 6b's
bfs3d runs); 0.9 and the `lineax` stop were not run. Configuration as in 6b's bfs3d table:

| `rtol` | outer steps | inner solves | cost re-fits | steps redone | final `\|R\|` | solve wall |
|---|---|---|---|---|---|---|
| 0.1 | 27 | 89 | 4 | 0 | 9.28e-6 | 774 s |
| 0.3 | 30 | 123 | 3 | 0 | 5.47e-6 | 787 s |
| **0.6** | 26 | 126 | 1 | **1** (step 15, `alpha`, beta 0.005 -> 0.01) | 8.02e-6 | **590 s** |

`x_r/h` 8.361 in all. **0.6 is 25 % faster than 0.3 on bfs3d** (13 % on pitzDaily) with one retry, so it
fails the pre-registered "no extra retries" bar by one step on both cases; unlike pitzDaily it took FEWER
outer steps. **0.1 is level with 0.3 here** (pitzDaily: 25 % slower), so tightening is not a loss on this
case, and the Eisenstat–Walker refutation rests on pitzDaily alone. Whether 0.6 moves is the owner's
call; the retry is the thing a forcing safeguard (6b) would also have to cover.

## 5. CLOSED — Anderson acceleration / nonlinear GMRES around the outer step

Measured 2026-10-14 and moved to `solve-refuted-directions.md` ("Anderson acceleration around the outer
step"), #639. Its gain was confined to the linear tail at the shift floor. On today's pitzDaily march
(19 steps, `release_floor` 1e-4) that tail is four steps long, and no Anderson proposal at any depth
reaches the march's own `x_{k+2}`: the bound is zero outer steps. The 2026-10-08 bound (~3–4 steps, ~7 %
of cycles on the 31-step march) is superseded. The release removed the tail Anderson was extrapolating.

## 6. CLOSED — Krylov subspace recycling (GCRO-DR); it found 6b instead

Measured 2026-10-08 and moved to `solve-refuted-directions.md` ("Krylov subspace recycling (GCRO-DR)").
On the replayed sequence of pitzDaily's 42 final-station solves, recycling costs 60–105 % MORE operator
applications than plain GMRES under the same stop, and still more with its recompute charge removed. At
`rtol = 0.3` a solve is ~12 iterations; there are no slow modes worth carrying.

## 6b. The GMRES stopping rule over-solves about 4×

**What.** Stop the inner Krylov solve on its residual alone, tested every iteration, instead of
`lineax`'s rule (tested only at a restart boundary, and only once the solution has also stopped moving
over a whole cycle). In-house restarted GMRES, or a `lineax` subclass; the adjoint's transpose solve is
untouched.

**Why it should win.** On the same 42 replayed systems (`krylov_recycling_probe.py`, exact reproduction
of every recorded cycle count) the march's solver used **2186** applications of `A M`; a residual-only
test at restart boundaries needs **890**, and one checked every iteration **514**. Krylov work is ~half
of pitzDaily's wall (fit over the baseline log: ~4.4 s per restart cycle against a ~2 s per-step
constant, 4-core Linux). Every solve pays at least two full restart cycles today; the median one needs
about one. This plausibly IS 4b's gain: loosening `rtol` mostly makes the solution-change test easier.

**What it does change.** The inner corrections become less over-solved (the march's second cycle drives
the residual far below 0.3), so the inner Newton loop may take more iterations or clip differently. That
is exactly what a single-system replay cannot see, so it needs a march.

**Pre-registered measurement.** pitzDaily and bfs3d marches, shipped stop vs per-iteration residual stop
at 0.3 (and 0.6 on pitzDaily, to see whether 4b survives it): steps, inner solves, cycles AND operator
applications (cycles are no longer comparable across stops), retries, refreshes, wall, `x_r/h`.

**✅ Measured on pitzDaily 2026-10-08 — 1.88× faster wall, same root.** `residual_stop_gmres`
(`solve/linear.py`, opt-in) against the shipped stop, both through the study path
(`PITZ_KRYLOV_STOP=residual` / `shipped`), back to back, shipped `case.yaml`, jax 0.11.2, CPU, 4-core
Linux, one run each (this case's march is deterministic in its counts):

| | shipped stop | residual stop |
|---|---|---|
| outer steps | 31 | 31 |
| inner solves | 99 | 137 (+38 %) |
| applications of `A M` | 4931 | **2147 (−56 %)** |
| mid-step preconditioner refreshes | 12 | 4 |
| wall | 1306 s | **693 s** |
| final `|R|` / `x_r/h` | 7.818e-06 / 8.0686 | 7.700e-06 / 8.0686 |

⚠️ Every refresh count in this entry was taken while a cost re-fit still ran MID-STEP; since #290 a solve
that reaches `refresh_on_cycles` ends its step and the re-fit precedes the next one, so the step counts
above are not comparable with a march on the current tree.

⚠️ **Which solver these applications were counted on.** This table (2026-10-08) and the replay's 514 / 890
were counted with NO recomputation of `b − A x`: the library's solver restarted from the Arnoldi residual and
gave its verdict on it. From 2026-10-09 (`6d52eb1`) to 2026-10-16 every cycle end recomputed it, one more
application per cycle, so every residual-stop march in that window (the `refresh_on_cycles` tables below,
bfs3d's #699 table, the forced-default tiers) paid that charge; since 2026-10-16 only the final cycle does
(see the dated entry at the end of 6b).

The control reproduces the shipped march exactly (31 / 99 / 203). The looser corrections cost about one
inner Newton iteration per step and the cheaper solves pay for it twice over. ⚠️ **Not a pure stop
effect:** the residual stop reports cycles run where `lineax` reports cycles run minus one, so the
`refresh_on_cycles = 3` trigger fired 4 times against 12; part of the wall gain is fewer rebuilds.
**The rebuild threshold re-tuned under the new stop (2026-10-09, same configuration,
`PITZ_FORWARD_STOP=residual PITZ_REFRESH_ON_CYCLES=n`, one run each, all `x_r/h` 8.0686, no retries):**

| `refresh_on_cycles` | mid-step rebuilds | rebuild time | restart cycles | wall |
|---|---|---|---|---|
| 1 | 31 | 217 s | 158 | 695 s |
| **2** | 14 | 143 s | 159 | **637 s** |
| 3 (shipped) | 4 | 94 s | 208 | 693 s |
| 4 | 2 | 87 s | 202 | 694 s |

2 rebuilds about as often as the shipped stop did at 3 (12) and is the fastest, **2.05×** the shipped
configuration's 1306 s. `abort_above_cycles` (10), `cycle_budget` (42) and `max_restarts` (14) bound in no
run at 0.3; `max_restarts` stays above the abort threshold under the new count, so they are left. With the
default flip the case files would carry `stop: residual, refresh_on_cycles: 2` (bfs3d's own value
re-measured there).

**In the library (2026-10-09):** `LinearSolveSettings.stop: residual` selects it from a case file or a
builder (`PITZ_FORWARD_STOP=residual` edits pitzDaily's file); the library default became `residual`
on 2026-10-15 (#645, with the adaptive tolerance).
**pitzDaily opts in (2026-10-09, project owner's decision):** its `case.yaml` carries `stop: residual`
and `refresh_on_cycles: 2`. ⚠️ So "the shipped `case.yaml`" in any pitzDaily entry dated before
2026-10-09 means the `lineax` stop at `refresh_on_cycles` 3; `PITZ_FORWARD_STOP=lineax
PITZ_REFRESH_ON_CYCLES=3` restores it. Making `residual` the library default without breaking the
tight-tolerance tests below is #645.
**Re-tuned again once a cost re-fit ENDS the step (#290, 2026-10-11): pitzDaily now ships
`refresh_on_cycles: 3`.** The 2 above was measured while the re-fit ran mid-step. Ending the step changes
what the threshold costs: under the residual stop nearly every first solve runs 2 cycles, so at 2 almost
every step ended after one inner iteration, and the 1-step-per-station ramp and the floor release (both
count outer steps) moved with it. Shipped `case.yaml` of that day (`grow` 3, 12 stations, `release_floor`
1e-4, `stop: residual`), only `PITZ_REFRESH_ON_CYCLES` overridden, one run each, Linux x86_64, 4 cores,
jax 0.11.2, nothing else running; branch `737d297` against `main` `a2a73d3`:

| | `main` @2 (mid-step) | ends step @2 | @3 | @4 |
|---|---|---|---|---|
| outer steps | 16 | 39 | 20 | 19 |
| inner iterations | 76 | 64 | 77 | 78 |
| restart cycles | 111 | 100 | 140 | 159 |
| cost re-fits | 13 | 23 | 4 | 2 |
| re-fit time | 290 s | 321 s | 158 s | 134 s |
| solve wall | 902 s | 935 s | 874 s | 882 s |
| floor released at step | 15 | 34 | 18 | 17 |

All `x_r/h` 8.0686, no retries. Fewer re-fits trade against more cycles on staler inverses and the two
cancel: every wall is within 7 %, single runs, so no arm is separable from `main` on time. 3 was chosen
for restoring the march's shape (about 20 steps of about four inner iterations), not for speed.

**✅ Measured on bfs3d 2026-10-12 (#699) — 1.48-1.52x faster wall, same root, no new retry.** `validation/bfs3d_openfoam/compare.py` at `46af0749` (after #290's step-ending re-fit), `case.yaml` as shipped plus the `BFS3D_*` edits named; Apple M3 Pro, 11 cores, 18 GB, macOS, Python 3.13.4, jax 0.10.2; one run each, back to back through `run_case.sh` with nothing else running, run 0 warming the compile cache.
Cycle counts are not comparable across stops (see above), so the comparison is wall and steps:

| run | stop, `refresh_on_cycles` | outer steps | inner solves | cost re-fits | re-fit time | steps redone | final `\|R\|` | solve wall |
|---|---|---|---|---|---|---|---|---|
| 0 | `lineax`, 3 (shipped) | 36 | 73 | 17 | 415 s | 0 | 9.94e-6 | 1197 s |
| 1 | `residual`, 3 | 30 | 124 | 5 | 249 s | 0 | 7.98e-6 | **810 s** |
| 2 | `residual`, 2 | 34 | 105 | 16 | 399 s | **1** (step 18, `alpha`) | 9.96e-6 | 814 s |
| 3 | `residual`, 4 | 30 | 123 | 3 | 220 s | 0 | 5.47e-6 | **787 s** |

`x_r/h` 8.361 in all four. bfs3d's `case.yaml` takes the residual stop since 2026-10-13, at `refresh_on_cycles`
3. Runs 1 and 3 pass the pre-registered bar; 2 fails it on the retry and gains
nothing over 3. **3 and 4 were not separable in that table** (23 s apart, inside the machine's spread for one
run each, before #766's pivot floor). With the floor (2026-10-19, one sample each per machine), 4 was fastest
on both machine classes: M3 Pro 531 / 475 / 497 s for 3 / 4 / 5 (21 / 17 / 17 steps), 4-core Linux 3141 /
1860 / 3693 s (32 / 17 / 28): `validation-log.md`, "The re-fit cadence on this machine class";
2 does not win on bfs3d as it did on pitzDaily before #290. No erratic march of the kind below appeared:
runs 1 and 3 end on eight steps at `alpha` 1 with `|R|` halving every step at the floor shift 0.005 and
never rising. ⚠️ That tail's rate is the floor's (no release in this file), so these runs cannot show the
linear-ending failure below either way -- a shifted step converges linearly whatever the stop. ⚠️ bfs3d's march
**changed under #290/#694** (traced split, re-fit ending the step): the shipped file's 28 steps / 174
cycles / one retry of 2026-10-11 (`61a14308`) is now 36 / 166 / none at about the same wall (1191 against
1197 s) -- the two runs part at the first inner solve's fourth digit, and each re-fit's refactor phase
went from ~2.8 to ~5.3 s. Compare bfs3d runs only within one commit. The refactor-phase rise is the
structural-pattern products of 868ab80/f7326c1 (on pitzDaily 0.88 -> 1.06 s per split refit), cached
since 2026-10-13 (`StructuralPatterns`, `solve-amg-multigrid.md`: 0.83 s); the trajectory change is not
explained by it and is not bisected.
**The default moved on 2026-10-15 (#645), with `AdaptiveLinearTolerance` as the forcing safeguard for
the end-of-march failures below.** Still open: the cost thresholds' re-calibration under the new count,
and a bfs3d arm with the adaptive tolerance (its file already ran the residual stop).

**The tiers with `stop: residual` forced as the default (2026-10-09, `residual_stop_gmres` with the
true residual recomputed at every cycle end, commit after `1619fd0`; jax 0.11.2, CPU, 4-core Linux).**
Validation: 18 passed, 8 skipped (case data absent). Slow: everything passes alone except three, which
pass under `lineax`:
- `test_coupled_lu.py`'s two march tests, both failing in the shared **block-diagonal** march
  (`max_steps=40`, ends at 2.807e-11 against 7.093e-12). **Cause: the end of the march converges
  linearly.** A step solved to 0.3 removes about 0.3 of the residual once Newton is otherwise exact, where
  lineax's stop over-solves and gives near-quadratic convergence. Traced (`on_step`, default
  `Convergence`, `rtol` 0.3): block march 27 steps (lineax, `4.0e-06 → 1.1e-15` in two steps) against
  42 (residual, ~0.3 per step from 1e-3 down); complete-LU march 28 against 35. Recomputing the true
  residual changed nothing (bit-identical failure), so it was not Arnoldi drift.
- `test_reynolds_continuation.py::test_adjoint_matches_a_direct_solve_and_is_point_count_independent`
  (`n_points=1`, `max_steps=60`). **Not a budget shortfall: the target march goes erratic** — accepted
  steps raising `|R|` 0.087 → 3.9, 0.030 → 2.7, 0.077 → 3.3, line-search `alpha` down to 1e-3, and 60
  steps end at 4.9e-02 (lineax: 11 steps to 1.2e-13). With the residual stop on the TARGET step only
  (ramp rung on lineax, a slightly different anchor) it converges in 24, linearly. So the inexact step is
  fragile here near the anchor, not only slow at the end.
- **Consequence:** flipping the default as built would trade pitzDaily's 1.88x for slower, linear endings
  on tight-tolerance solves and a lost robustness margin on at least one continuation.
- **The remedy, BUILT opt-in 2026-10-15 (#645): `solve.AdaptiveLinearTolerance`** — two rules, one setting
  (`newton_march(adaptive_tolerance=)`, reachable from every march and a case file's `solver` section):
  tighten in a tail of full-length steps limited by their own solve (`0.9·ρ²`, floored at
  `½ target / |R_k|`), and solve the step after a cut-back or a rising residual to 0.1. The other proposed
  remedy, re-solving the cut-back step tighter, was refuted (ledger: "Re-solving a cut-back step at a
  tighter tolerance"). Measurements, on the prototype and then on the built code, are in `solve-log.md`,
  "Contracts: `adaptive_tolerance.py` — measured on the built setting (#645)", and `validation/adaptive_tolerance_probe.py`. **The slow tier with BOTH forced on
  (`LinearSolveRegime.stop` defaulting to `residual`, and the setting at its defaults on every march whose
  solver is a relative-residual GMRES) passes: 39 of 39** (2026-10-15, `d0871e2` + the two forcings in a
  scratch worktree, Linux x86_64, 4 cores, jax 0.11.2, 1:01:39). The stop alone failed four
  (`test_coupled_field_split.py`'s two channel marches, which errored, plus its adjoint test, and the
  Reynolds-continuation adjoint test). ⚠️ The first forced run, with the
  reduction re-measured in each step's own scales, still failed the Reynolds test (6.5e-3 at 60 target
  steps): see `solve.md` for why the reduction is read report to report. Both became the library
  defaults the same day; bfs3d has no arm with the adaptive tolerance yet.

**Follow-ups.** Two were measured on the replay and are closed (ledger: "Weighted inner product and
longer restarts for the residual-stop GMRES"): neither a measure-weighted inner product nor a restart of
30–60 changes the work. The forcing term was re-measured under the new stop (2026-10-09, same configuration,
`PITZ_KRYLOV_STOP=residual PITZ_FORWARD_RTOL=…`, one run each):

| `rtol` | steps | inner solves | applications | refreshes (mid-step) | retries | wall |
|---|---|---|---|---|---|---|
| 0.1 | 31 | 110 | 2380 | 12 | 0 | 863 s |
| **0.3** | 31 | 137 | 2147 | 4 | 0 | 693 s |
| 0.6 | 33 | 169 | 1747 | 2 | **1** | 604 s |

All reach `x_r/h` 8.0686. **Tightening still loses** (0.1 is 25 % slower), so the Eisenstat–Walker
refutation stands under either stop. **Loosening still gains, but less and at a cost:** 0.6 is 13 % faster
than 0.3 where it was 18–21 % fewer cycles under `lineax`'s stop, and it is the only arm that takes extra
outer steps and a retry (whose tight solve the application count omits). So most of 4b's gain was the
stop; what is left is modest and comes with the first sign of fragility — 4b (#637) stays deferred, now
to be measured on top of 6b rather than instead of it.

**✅ The end-of-cycle recomputation charged only at the verdict (2026-10-16).** `_ResidualStopGMRES` recomputed
`b − A x` at every cycle end (one application of `A M` each, about 0.27 s on pitzDaily). It now hands a cycle
that falls short the Arnoldi residual, `V (β e₁ − H y)`, and recomputes only when the recurrence says the
solve has met its target, so the verdict reads the true residual.
- **The drift, measured first** (`arnoldi_drift_probe.py`): every inner system of a capture of the shipped
  `case.yaml` (`stop: residual`, `rtol` 0.3, restart 15, `refresh_on_cycles` 3, field split `SimpleSmoothed` /
  `JacobiSmoothed`; 18 steps, 68 systems, every station refit), solved by GMRES(15) with classical Gram–Schmidt
  twice, restarting from the recurrence, up to 20 cycles; the true residual formed (uncounted) at each cycle end
  and where each target is first met; row-scaled measure; `44ba609`, Linux x86_64, 4 cores, jax 0.11.2, 5580 s.
  `departure` = measure(true − recurrence) / measure(true); `verdict` = measure(true) / target at the stop:

  | target | systems reaching it | worst departure at a cycle end still above it | at its stop | worst verdict |
  |---|---|---|---|---|
  | 0.3 | 68 | 1.1e-12 | 2.3e-12 | 0.9996 |
  | 1e-3 | 65 | 5.2e-10 | 7.8e-10 | 0.9995 |
  | 1e-6 | 48 | 3.0e-7 | 6.9e-7 | 0.9963 |
  | 1e-10 | 32 | 3.7e-3 | 7.5e-3 | **1.0013** |

  The absolute departure is a round-off of `‖b‖` throughout, so its relative size grows as the target tightens.
  As a restart vector it is harmless at every target; as a verdict it is not at 1e-10 (one solve declared
  converged 0.13 % above its target; the unmet counts are the 20-cycle cap, not drift). Hence the hybrid, not the
  unconditional restart: one application per solve stays, and no new setting.
- **Replay check:** on the target station's 16 systems (`krylov_recycling_probe.py`, arms `residual,gmres-15`)
  the library took exactly the probe's GMRES count plus one, solve for solve (467 against 451), and reproduced
  every recorded cycle count.
- **The march** (`PITZ_KRYLOV_STOP=residual` to count, shipped `case.yaml` otherwise, same machine and commit as
  above, one run each, back to back through `run_case.sh`, nothing else running; this march is deterministic
  in its counts):

  | | every cycle end (`main`) | final cycle only |
  |---|---|---|
  | outer steps / inner solves / restart cycles | 18 / 68 / 123 | 18 / 68 / 123 |
  | applications of `A M` | 1462 | **1407 (−55 = cycles − solves)** |
  | solve wall | 463 s | 470 s |
  | march clock, step 1 → 18 | 380 s | **359 s** |
  | final `\|R\|` / `x_r/h` / peak `nu_t/nu` | 9.574e-06 / 8.0686 / 418.09 | identical |

  Step tables identical in every column but time. ⚠️ Read the march clock, not the solve wall: step 1 carries the
  compilation, and the arm on `main`'s code ran second with its program already cached (64 s to step 1 against
  93 s). The 21 s difference matches 55 applications at about 0.3 s each plus the Gram–Schmidt work, ~4 % of
  the steady march; one run each, so it is not separable from run-to-run spread on its own — the application
  count is the evidence.

## 7. Mixed precision: float64 outside, float32 inside the preconditioner

**What.** Keep the residual, the Krylov true-residual recurrence, the outer stopping test and the
adjoint in float64. Store the preconditioner's `J`, hierarchy, smoothers and V-cycle in float32, with
casts at the apply boundary.

**Why it should win.** Halves memory traffic in the part of the solve that is bandwidth-bound, which
is the whole game on a GPU; on CPU the gain is smaller but non-zero. A float32 preconditioner is still a
*fixed linear map*, so plain GMRES and `jax.linear_transpose` remain valid — the three properties
`preconditioning.md` relies on survive.

**What it does not change.** The preconditioner never enters the state or its gradient; a rounding
change inside it changes Krylov convergence only. `solve-amg-multigrid.md` already notes the one place
a float32 hierarchy *forces* a change (the small-block pivoting, under "The traced multigrid — in progress"): treat that as the checklist.

**Pre-registered measurement.** bfs3d, matched preconditioner at `state-00049`: cycles and apply wall,
float64 vs float32 PC. Pass is cycles within +10 % at lower wall. The real measurement is on a GPU
(next entry).

**Measured on pitzDaily, CPU (2026-10-09) — convergence is unchanged, and the CPU apply is SLOWER.**
⚠️ A substitution for the pre-registered bfs3d measurement, made on the user's instruction, in the belief
that bfs3d's mesh was not in the repository. It was (tracked since 2026-10-17 at
`validation/bfs3d_openfoam/of_case/constant/polyMesh`), so the bfs3d measurement can be taken. `precision_replay_probe.py` replays the march's own 75 target-station systems
(steps 17–31, capture of the shipped `case.yaml`: residual stop, `rtol` 0.3, restart 15, `refresh_on_cycles`
2, field split `SimpleSmoothed` / `JacobiSmoothed`; jax 0.11.2, CPU, 4-core Linux) with both block
inverses' V-cycles in float32 — hierarchy, smoothers and right-hand side; the split's coupling product,
`J + s`, the Krylov recurrences and the measure stay float64. The float64 arm reproduces every recorded
cycle count.
- **Convergence: identical.** 76 cycles / 899 applications in both precisions, every solve equal. Both
  float32 cycles are float32 throughout (0 of 350 and 0 of 94 traced equations produce float64).
- **Apply cost: 2.0× slower in float32** (best-of-10 split apply, mean over 15 steps: 371.5 ms against
  183.7 ms), and it is not a conversion: casting the whole hierarchy is 6 ms and is cached, and the
  directly-jitted flow V-cycle alone is 215.6 against 111.5 ms.
- **The cause is one JAX kernel** (`csr_kernel_precision.py`): the level operators' CSR product
  (`BCSR @ x`, which lowers to the custom call `cpu_csr_sparse_dense_ffi` in both precisions) is
  **2.6× slower in float32** on the finest flow level (17.1 against 6.6 ms; every level the same way),
  while the block solves, the prolongation and the coarse dense solve are sub-millisecond either way.
  On a synthetic matrix of the same shape and density it is 21.1 against 6.4 ms, so it is the kernel,
  not this operator. A gather plus sorted segment sum IS faster in float32 (8.9 against 10.7 ms) but is
  still slower than the float64 CSR kernel, so no CPU kernel choice makes float32 pay here.
- ⚠️ **The kernel this rests on is gone (2026-10-14):** the level operators are padded-row
  (`_PaddedRowOperator`), a fused gather-multiply-sum rather than the CSR kernel call, so "no CPU kernel
  choice makes float32 pay" is unmeasured for the shipped form and the CPU half is open again.
  `csr_kernel_precision.py` now times the padded-row form beside the CSR one, in both precisions.
- **On the padded-row form float32 PAYS, and is shippable as `FieldSplit(precision="float32")` (2026-10-16,
  PR #732, opt-in; `solve-field-split.md` holds the design).** Every number below is one run per arm at a
  fixed configuration; "pitzDaily march" is the shipped `case.yaml` on `float32-field-split` `d8ca69ab`,
  4-core Linux container, jax 0.11.2, the two arms back to back, differing only in `PITZ_SPLIT_PRECISION`.
  - *Kernel, split apply:* pitzDaily at its hybrid start (Linux, best of 15) 52.5 ms against 71-87 ms
    float64; bfs3d at `state-00018` (Apple M3 Pro, jax 0.10.2, best of 10) **82 ms against 140 ms**, which
    also beats `main`'s CSR (105-109 ms) -- float32 is the one change found that recovers the Mac's
    forward loss on wide rows without an XLA flag (see `solve-amg-multigrid.md`'s padded-row entry).
  - *pitzDaily march:* **405 s against 531 s (-24 %)**, 18 steps / 123 cycles both, the step tables
    identical to four figures through step 17, final `R` 9.093e-06 against 9.071e-06, `x_r/h` 8.0686 both.
  - *Adjoint:* a split applied in float32 ARITHMETIC broke the transpose solve (stagnation at rtol 1e-10:
    it is linear only to ~1e-7); the adjoint now widens the rounded split to float64, and the float32
    arm then matches the finite difference to 1e-5 on the coupled channel.
  - *bfs3d march* (shipped `case.yaml`, `7e96745a`, Apple M3 Pro, jax 0.10.2, one run per arm, float64
    first, idle machine): **498 s against 599 s (-17 %)**, 18 steps / 129 cycles both with identical
    per-step inner and cycle counts, final `R` 6.185e-07 and `x_r/h` 8.361 both; steps from 5 on are
    20-35 % faster, step 1 (compilation) level, the refit unchanged (it is float64 work).
  - **Shipped (2026-10-16, the project owner's decision): both case files set `precision: float32`;
    the library default stays float64.** This direction's CPU half is closed as a win.
  - **The two PRs re-measured together, on `main` after the adjugate inverse (2026-10-17).** Both #721
    and #732 were already merged when this was run, so this is an ablation on `main` `3c5b58d6`, not a
    PR-against-base comparison. Arms with #721 removed run in a local worktree of `main` with #721's
    merge reverted (`git revert -m 1 5cc4bf09`: the library code reverts cleanly, the only conflict is
    in this file). The float32 arms are the shipped `case.yaml`, and the float64 arms set
    `PITZ_SPLIT_PRECISION=float64`. Shipped pitzDaily `case.yaml`, warm cache, the Linux container
    and jax 0.11.2 of the table above. Splits are timed by `application_cost.py` at steps 19-21, faults
    removed. **Every arm took the identical march: 21 steps / 50 inner solves / 106 cycles / 1258
    applications / 21 refits, step tables equal in every column but `t(s)`, final `R` 2.618e-06,
    `x_r/h` 8.0686.** So the comparison is wall clock alone:

    | arm | level operators | split | solve wall | split apply | JVP | M share |
    |---|---|---|---|---|---|---|
    | A, both (shipped) | padded-row | float32 | **227-249 s** (3 runs) | 48-50 ms | 14-17 ms | 74-78 % |
    | B, #721 only | padded-row | float64 | 251-283 s (2 runs) | 66-67 ms | 14-18 ms | 78-82 % |
    | D, #732 only | CSR | float32 | 786 s (1 run) | 441-451 ms | 13-18 ms | 96-97 % |
    | C, neither | CSR | float64 | 449 s (1 run) | 195-200 ms | 14-18 ms | 92-93 % |

    - **#721 is the large win: 449 -> 251-283 s (-37 to -44 %) at float64**, split apply 195-200 ->
      66-67 ms (-66 %). That is about twice the ~30 % its PR measured at the hybrid start. Its one
      cost visible here is the refit's host refactor, 20 s (CSR) against 27 s (padded) over the 21
      refits, which is the padded and transposed copies being built.
    - **#732 is worth ~6-12 % on top of #721** (B 251-283 against A 227-249 s; in a B-between-A
      bracket of back-to-back runs, 227 / 251 / 236 s), split apply 66-67 -> 48-50 ms. That is above
      the 3 % tie threshold, but the container's own spread is ~5-10 %, so the size is not pinned
      closer than that. The PR measured -24 % (531 -> 405 s) at `d8ca69ab`, before the adjugate inverse
      made the JVP and the probe cheaper, which raised the step program's split share.
    - **Without #721, float32 LOSES: 449 -> 786 s (+75 %)**, apply 2.2x slower. This is the CSR
      kernel's float32 penalty recorded above (2026-10-09), now measured over a whole march. So #732
      depends on #721, as its PR said.
    - Counts are deterministic here (the same in all thirteen marches of this record), so one run per arm
      holds for counts. For wall clock, arms C and D lie outside any noise and A against B does not by
      much. bfs3d was not re-run here, so its figures remain the PRs' own (M3 Pro, jax 0.10.2); its mesh is
      tracked (`validation/bfs3d_openfoam/of_case/constant/polyMesh`), so it can be.
- **(2026-10-09, under the CSR kernel, since superseded by the two bullets above) the CPU half closed nothing:** precision does not cost convergence on this case,
  and the bandwidth gain cannot show on CPU through this kernel. The GPU measurement (entry 8), where the
  CSR product is a different kernel, remains the deciding one; bfs3d's convergence is unmeasured.

## 8. The GPU path, and what it rules out

Not a new idea — it is the stated plan — but worth stating as a solver direction because it decides
others. bfs3d at 138k dofs should be of order one second per Newton step on a single modern GPU, which
would make the whole march minutes rather than fifty. Every host-callback inverse (PETSc GAMG via
`pure_callback`, scipy LU) is serialized through the host on that path, which is why the traced
hierarchies (`SimpleSmoothed`, `MaterializedBlockPreconditioner`) exist. Consequences for this file:
entry 7 compounds on GPU and is worth more there than its CPU measurement will show;
entry 6 is a JAX solver by necessity; a GPU-friendly smoother (coloured Gauss–Seidel, Chebyshev where
the operator allows, or more ILU-free sweeps) replaces ILU(0), whose quality the ledger credits to a
sequential global sweep that does not parallelize.

**Measurement.** Unavailable until a GPU environment exists (`solve-amg-multigrid.md`, "Faithful smoothed aggregation — matching PETSc GAMG"). Record
the first GPU march's step/cycle counts against the CPU march's — counts should match; clocks are the
result.

## 9. CLOSED ON CPU — batching a parameter sweep with `vmap`; reopen with entry 8

Bounded on pitzDaily (2026-10-10, `batch_bound_probe.py`) at **at most 1.10x on the Krylov work at 8
members**, because the preconditioner is ~80 % of each application and does not batch. Recorded in
`solve-refuted-directions.md` ("Batching a parameter sweep with `vmap`"). It is a GPU question, where
the preconditioner would be traced and a member too small to fill the device; reopen it there.

## 10. Nonlinear elimination of the ω stiffness

**⚠️ ITS MOTIVATION IS REFUTED ON pitzDaily (2026-10-14); bfs3d is the deciding check.** The entry below
reads the census's pure-ω near-null direction as "the cell block is weakly coupled in ω". On pitzDaily `state-00017` (shift 1e-4, 2026-10-14, `cell_block_conditioning.py --case pitzdaily_openfoam`, jax 0.11.2, Linux, 4 cores)
the same signature appears, and it is a large coupling in the ω ROW to the cell's own pressure and
velocity (equilibrated ω–p 1348, ω–u −787 in cell 721, against an ω diagonal of 1), not a weak ω column:

| blocks | `σ_min` < 1e-3 | < 1e-2 |
|---|---|---|
| as assembled | 6 | 273 |
| ω row's couplings to the other fields zeroed | **0** | **0** |
| ω column's couplings zeroed (control) | 6 | 274 |

Over the 200 smallest blocks `σ_min` tracks one over the largest ω-row coupling (log-log correlation
0.99), and the inverse's largest entry is that coupling's size -- non-normal, not singular. A local ω
solve with the flow frozen is already well posed, so the elimination below targets nothing; the shift
not reaching the cells is predicted too (it adds to a diagonal that was never small). **To close or
reopen it:** run `validation/bfs3d_openfoam/cell_block_conditioning.py N <shift> 0.05 0` on the three
states in the table below; the `no ω-row` / `no ω-col` columns decide it. Why ∂R_ω/∂p is so large near
walls is unmeasured (a hypothesis: ω's convective flux through the Rhie--Chow mass flux, where ω is
large, against a small pressure diagonal).

**What.** Before each global Newton step, run a few cheap local nonlinear solves of the ω equation
(flow and `k` frozen) in the cells the cell-block SVD flags (`σ_min < 1e-3`, ~1.5 % of bfs3d), then
take the global step from the eliminated state (Cai & Keyes, nonlinear elimination / ASPIN family).

**Why it might win.** The ledger's finding that *the cell block is weakly coupled in ω everywhere* and
near-singular in a coherent low-`k` region is a *nonlinear* preconditioning argument: the linear PC
cannot see a direction the local block does not contain, but a local nonlinear solve along the
transport direction can remove the stiffness before the Jacobian is formed. It is the one idea the
measured data points at that has not been tried; the notes name it once as a possibility.

**What it does not change.** At the root the elimination is the identity, so state and adjoint are
unchanged — but only if the eliminated residual is the *same* residual; do not hand the IFT a different
operator (the warm-start refutation is the cautionary tale).

**Risk.** Highest in this file; measure on the captured hard iterates before touching the march.

**✅ The census re-taken on today's march (2026-10-12, #699): the pure-ω near-singular cells are still
there, at the states where the march is slowest.** `cell_block_conditioning.py` on bfs3d's shipped march
at `46af0749` (#699 run 0, `BFS3D_CHECKPOINT_KEEP=200`; configuration as in 6b's bfs3d table), at the
checkpoint before each of the three steps with the hardest single inner solve (all three line-search
clipped and positivity-limited; the march redid no step), each at its own shift, 0.05 and 0:

| state (before step) | hardest solve | shift | cells `sigma_min` < 1e-3 (at 0.05 / 0) | worst cond | median `k` there / whole mesh |
|---|---|---|---|---|---|
| `state-00011` (12) | 12 cycles, `alpha` 0.125 | 0.0173 | **644** (614 / 649) | 4.0e7 | 0.115 / 0.544 |
| `state-00027` (28) | 8 cycles, `alpha` 0.022 | 0.005 | 370 (346 / 375) | 2.7e7 | 0.127 / 0.710 |
| `state-00028` (29) | 5 cycles, `alpha` 0.25 | 0.01 | 365 (345 / 373) | 3.6e7 | 0.127 / 0.710 |

In every one of the 60 worst blocks the near-null direction is pure ω to three decimals (a `u` trace up
to 0.003 in two cells), and the worst cells come in spanwise-symmetric pairs (22480/12880,
21840/13520, ...) in low-`k` regions. **The shift does not reach them**: 0 -> 0.05 changes the count by
under 10 % at every state. Against the 2026-08-08 census (353 cells, cond 5e6-9e6, median `k` 0.122) the
count is up to 1.8x higher and the blocks ~5x worse conditioned, but that census was at a different state
under the deleted ILU bundle, so read it as "still present", not "worse". **By #699's rule this direction
keeps its motivation**; the three states above (`checkpoints-baseline/` of that run, gitignored) are the
ones to design and probe it on.

## 11. MEASURED AND BUILT (opt-in) -- releasing the shift floor once the target station has settled

Probed and marched on pitzDaily 2026-10-10: the floor `beta_min` sets the target station's linear rate,
and `release_floor` (`ShiftStrengthControl`, off by default) cut the march from 31 steps / 161 cycles to
22 / 141 at the same `x_r/h`. The probe tables and the march are in `solve-globalization-log.md`
("The shift floor sets the target station's rate"); the built control is in `solve-globalization.md`.
⚠️ An earlier version of this entry quoted the march log's wall-clock column as solver work ("1036 raw
solver steps", "174 solver steps"); those were seconds, and the log entry carries the corrected counts.
**(a) Re-fitting the inverse at the released shift: CLOSED on pitzDaily and bfs3d (2026-10-13), worth one cycle on pitzDaily and none on bfs3d.**
pitzDaily `case.yaml` at `14e3963` (residual stop, `rtol` 0.3, `refresh_on_cycles` 3, `grow` 3 x 12 stations, `release_floor` 1e-4, field split `SimpleSmoothed` (frozen) / `JacobiSmoothed`), only `PITZ_REFIT_BETA_FLOOR` varied; Linux x86_64, 4 cores, jax 0.11.2, one run each, back to back:

| `refit_beta_floor` | steps | cycles | final `\|R\|` | wall (indicative) |
|---|---|---|---|---|
| 0.05 (shipped) | 20 | 154 | 1.7e-6 | 643 s |
| 1e-4 (tracks every shift, released ones too) | 19 | 144 | 3.5e-7 | 534 s |
| 0.005 (tracks to `beta_min`, not the release) | 19 | 143 | 3.4e-7 | 546 s |

All `x_r/h` 8.0686, no retries. The 1e-4 and 0.005 arms are identical through step 17 and differ by one
cycle at step 18: since #290 the release fires at step 17 of 19, so there is nothing left for it to save
(the replay's 1-8 cycles per step were taken on the 31-step march of 2026-10-10). Ledger entry: "Re-fitting
at the released shift". **What the sweep found instead is the floor itself:** the 0.05 and 0.005 arms part
at step 4, the first step below 0.05, and refitting at the step's own shift saves 11 cycles (7 %) and a
step. The 0.05 floor was set for the ILU(0) smoother on bfs3d (git history of `solve-amg-multigrid.md` before 2026-10-17, "Zero-fill is the low-β smoother"); the SIMPLE-smoothed
hierarchy does not evidently need it. **pitzDaily's file drops the floor** (2026-10-13, owner's decision;
none is the 1e-4 arm exactly, since the shift never goes below `release_floor`; confirmed by a march of
the file as committed, whose step table matches the 1e-4 arm's line for line). ⚠️ The 0.05 arm's wall overlapped a few seconds of unit tests; cycles and steps
are deterministic.

**bfs3d, measured 2026-10-13 by the owner (comment on #709):** `validation/bfs3d_openfoam/compare.py` at `7eccaea5`, `case.yaml` as shipped (`lineax` stop, `rtol` 0.3, `refresh_on_cycles` 3, `grow` 1.5 x 12, release as in the table, `SimpleSmoothed` / `JacobiSmoothed`) with a local, unpushed `BFS3D_REFIT_BETA_FLOOR` override; Apple M3 Pro, 11 cores, Python 3.13.4, jax 0.10.2, one run each, back to back, nothing else running; no step redone in any run:

| `refit_beta_floor` | release | steps | cycles | re-fit time | final `\|R\|` | solve wall |
|---|---|---|---|---|---|---|
| 0.05 (shipped) | off | 36 | 166 | 420 s | 9.94e-6 | 1197 s |
| 0.005 | off | 33 | 155 | 336 s | 7.24e-6 | 1079 s |
| none | off | 33 | 155 | 333 s | 7.24e-6 | 1086 s |
| 0.005 | 1e-4 | 29 | 139 | 347 s | 7.00e-6 | 1019 s |
| none | 1e-4 | 29 | 141 | 345 s | 6.47e-7 | 1027 s |

`x_r/h` 8.361 in all; the 0.05 arm reproduces #699's baseline (36 / 166). Dropping the floor saves 3 steps and 11 cycles (7 %, as on pitzDaily) and ~10 % of the wall. 0.005 and none are identical without the release (the shift never goes below `beta_min` there). **(a) on bfs3d is worth nothing:** with the release, the two arms are identical through step 27, the release fires at step 28, both finish at step 29, and no floor costs 2 more cycles for a deeper final `|R|`. Both released steps cost 3 cycles each, not #678's 12 and 15 (measured before #290/#694 changed this march). bfs3d's file dropped the floor and releases since 2026-10-13 (owner's decision, with pitzDaily's other march settings; `solve-globalization-log.md`, "bfs3d takes pitzDaily's march settings").

(b) whether bfs3d's `case.yaml` should
release -- DECIDED 2026-10-13: it does, with `grow` 3 and the residual stop. Measured 2026-10-11 (#678): alone, on its shipped `grow` 1.5 x 12, 28 -> 24 steps and 174 ->
173 cycles at the same root, a pass on steps only; with pitzDaily's `grow` 3 a clear loss under bfs3d's
lineax stop but a win under pitzDaily's `stop: residual` (25 steps / 195 cycles / 792 s, the best arm;
log, "bfs3d: the release, `grow` 3 and the settle-ended ramp"), a choice of the stop as much as of the
release; on the post-#290 march with no refit floor (table above) it saves 4 steps, 16 cycles and 60 s; (c) whether the library default
should release, after the three slow tests #645 names (a loose terminal step is what they measure).
pitzDaily's `case.yaml` releases since 2026-10-11, with `grow` 3 and 12 stations.

## 12. End the viscosity ramp on the settle signal, not on a station count

**BUILT 2026-10-11, opt-in** (`solve.RampSchedule`, case file `end: settled`; the prototype it was
measured with, `settle_ramp_probe.py`, is deleted). **Pre-registered bar not met by a single jump** (table in
`solve-globalization-log.md`, "Ending the ramp on the settle signal"): paces 16 and 24 cost 124 and 118
cycles against the shipped 111, no retry, same `x_r/h`. The jump at the settle is tolerated at any size
measured (up to 21.5x) but undoes the settle, so the target settles a second time. A 3-step finish
(`finish: 3`) at pace 24 met the bar -- 114 against 111 -- though not by keeping the settle: in three of
four arms the target settled at step 14 whatever the ramp did. The rule is far less pace-sensitive than
a fixed count. **bfs3d (2026-10-11, #678):** at `grow` 3 the rule ended the ramp itself after step 9 and
cost 187 accepted / 267 all-attempt cycles, against 214 / 459 for a fixed 12 at `grow` 3 and 174 / 193 for
the shipped `grow` 1.5 x 12, all under the lineax stop -- so it does not replace bfs3d's hand-tuned count
there. Not run under `stop: residual`, where the fixed `grow` 3 x 12 wins.
**Still open, and weaker:** a predictive rate controller keyed on the logged `rho1` trend. On bfs3d it
gave no lead on the settle at either `grow`: at `grow` 3 it stayed above one (to 2.9) through the floor
steps and fell only on a step whose shift had just been raised; at the shipped `grow` 1.5 it was below
one from the first floor step. It did separate the schedule that worked (below one) from the two that
redid steps (above) -- a "shift outran the state" signal, not a settle predictor. The text below is the proposal as written before the measurement.

**What.** Replace `stations` with a rule: ramp the viscosity down at a fixed pace (2x per step cost
nothing in `alpha`), and move to the target on the first full-length step at the floor -- the same gate
`release_floor` uses. Needs `ResidualHomotopy` to see the previous `StepReport`, which it does not today.

**Why it should win.** The `grow` x `stations` sweep (`solve-globalization-log.md`, 2026-10-11): the
step count is set by 6-9 floor steps before the state settles, the station size does not move `alpha`,
and the cheapest fixed schedule (`grow` 3, 12 stations, 111 cycles / 16 steps) is the one that arrives
one step after the settle. A rule keyed on the settle removes the knob and lands there on any case.

**What it is not.** Not an `alpha`-driven station controller (closed by the same sweep: `alpha` responds
to the shift, not to the viscosity), and not a viscosity keyed to `beta` (refuted-directions ledger).

**Pre-registered measurement.** pitzDaily at `grow` 3 with the rule against the 12-station arm its
`case.yaml` ships (111 / 16): a pass is the same cycles within a few at the same `x_r/h`, with the ramp ending by itself. Then
bfs3d (since measured, #678: see the top of this entry).

---

## 13. CLOSED — a Schubert (sparse Broyden) secant update of the materialized Jacobian between probes

Measured 2026-10-17 on the replay and moved to `solve-refuted-directions.md` ("A Schubert secant update
of the materialized Jacobian"). The updated Jacobian is no closer to a fresh probe than the stale one in
any block, and the probe it would replace now costs ~1 s, about one refactor.

## 14. Preconditioner memory for large polyhedral cases — what is left open (2026-10-22)

**Context.** The goal was a k-ω SST case of a few hundred thousand polyhedral cells on an 18 GB laptop. Two
attempts to shrink the split's storage are in `solve-refuted-directions.md`:
- "TRUNCATING the Jacobian to reach 1": truncation refuted, a consistent compact operator untested;
- "A fully matrix-free fine level": exact with the stored Schur diagonal, not closed.

The harness for both is `validation/bfs3d_openfoam/compact_jacobian_screen.py`. The polyhedral stencil sizes
are from `validation/polyhedral_stencil_size.py`. An independent review of the evidence concluded the
preconditioner is probably not the lever for that goal: with the split removed, bfs3d's steady footprint is
still about 4× the per-cell budget, and the compiled step's temporaries alone are about 44 KB a cell. **Take the
measurement in 14a before any of 14b–14d.**

**Re-baselined 2026-10-24 on `7fe27d07`** (`solve-direct-preconditioners-log.md`, "bfs3d memory by owner after
the device fit, probe and seeds"): 2-step peak 8.87 GiB with the tracer, set in re-fit #2's leading block
re-fit; the device holds 2.19 GiB of int32 pattern arrays against 0.66 GiB of float32 values, so the
index storage, not the values, is now the split's memory, and 14d's "narrower column indices" means a
cell-level (block) pattern with the field implicit, since global dof indices already need int32. The
stored transposes are about half of every operator. Freed memory glibc keeps after the step program's
first call (0.75 GiB) is in the baseline the peak sits on, so a `malloc_trim(0)` before each re-fit is
an unmeasured, cheap arm on bfs3d (on pitzDaily it moved the peak 0.03 GiB).

**The non-array residue is named (2026-10-24, same log, "What the resident memory holds beyond live arrays,
on two meshes")**: JAX's compiled-program cache, 2.09 GiB on bfs3d against 1.06 on pitzDaily. About 0.7–0.9
GiB is per-program runtime and compiler state over ~1300 executables; what doubles it between the meshes is
not separated, so no 300k projection stands. Arrays XLA folded into every program holding the residual (14e)
were 0.148 GiB of HLO on bfs3d but only ~0.05 GiB resident, measured on a march (2026-10-25).
**Only the setup-only part of that cache can be given back, and most of it is three programs (2026-10-25,
same log, "How much of the compiled-program cache a mid-run clear gives back" and "Which setup programs hold
the reclaimable cache")**: the march recompiles and holds 1.4–1.5 GiB of the cache whatever is cleared; of
the setup's ~0.38 GiB, `laplace_field`'s three per-call `newton_step` programs held 0.19 and never ran
again; released since 14g.

### 14a. Attribute the compiled programs' temporaries — the deciding measurement

**What.** `jit(...).lower(...).compile().memory_analysis()` on bfs3d's march step and on the probe's batched
JVP, each compiled twice: with the shipped split and with an identity preconditioner. Compile only, no march.
Optionally add one device-memory snapshot at the first re-fit's peak.

**Why.** It splits the step's ~1 GiB of temporaries into the preconditioner's share and the residual
JVP's.
- If the residual's share alone exceeds the per-cell budget once projected by face count, which is about
  2× on a polyhedral dual, the laptop target is closed whatever the preconditioner does.
- If the padded products dominate, 14b–14d are worth doing.

**The probe half, 2026-10-24:** the batched JVP's 829 MiB on bfs3d is the multiple-correction Green–Gauss
face loop at its peak. Face-blocking it and the flux balance reaches ~440–472 MiB, no better than halving
the batch, and is not built (`solve-refuted-directions.md`, "Assembling the residual's face sums in
blocks").

**The step half, 2026-10-25 (bfs3d, first step, `validation/step_temporaries.py`;
`solve-direct-preconditioners-log.md`, "What the march step's temporary holds"):** the step's 1017 MiB of
temporaries is 624 MiB the split's (the identity arm is 393), and 591 of that is int32 decoded padded
columns, every operator's `s32[n_rows, width]` live twice at the peak: one copy XLA hoisted to the entry
computation (the Arnoldi and final `x = M y` applications share it) and one rebuilt inside the true-residual
`lax.cond` of `_ResidualStopGMRES`, where both arms peak. The gathered operand is one `f32[n_rows·width]`
per application, not held; the solve's own vectors are 4 MiB. The residual's share without a split is ~2× a
JVP (393 MiB, 17 KB a cell) because that cond branch recomputes a whole JVP while the Arnoldi loop's hoisted
linearization is still held. So the split is the lever on the step's temporary, but through its decoded
indices, not 14b–14c's values. The true residual moved out of the `lax.cond` into a one-trip loop saves
nothing (+13.8 MiB; the decode is rebuilt in that loop too; ledger). **Built**: the cell-level padded
product gathers once per cell row (`a0fca85b`): the step's temporary 1022.7 → 683.7 MiB, the split's share
629.8 → 290.7; both marches unchanged in every count, bfs3d's 17.5 % faster, peak 5.29 → 5.12 GiB. Left: the decode is still held per call site
(~295 MiB); keeping it in the loop body (`validation/padded_decode_hoisting.py`: four operators 443 → 161
MiB) is untimed and would re-decode every apply. Unchanged on `main` after #825 (lazy transposes cut the step's arguments 1010 → 514 MiB,
not its temporary).

### 14b. A consistent compact operator for the fit

The Jacobian of a narrowed residual, compact by construction:
- first-order upwind;
- a compact Green–Gauss gradient;
- the strain rate stopped in the eddy viscosity;
- the continuity rows kept at Rhie–Chow's reach 2.

It is fitted in place of the truncated one, through `_CoupledNarrowing`, the probe's existing seam. It
differs from the refuted truncation by keeping row sums consistent. **Pass bar:** as in the screen, within
1.3× the exact arm's applications at every state, the root's transpose included.

### 14c. A biased-high Schur row-norm estimate for the matrix-free fine level

The `mffh:` arm with `|S_i|²` estimated, then clipped below at `S_ii²` and inflated, swept jointly with
`pressure_smoother_weight`.

It would also need the coarse levels probed through the aggregates. Otherwise a re-fit still holds the fine
Jacobian and the peak does not fall.

### 14d. Arithmetic-preserving storage cuts

None of these needs a convergence study:
- a cell-level pattern with the field implicit, in place of dof-level index arrays (2.19 GiB of int32 on
  the device at bfs3d's step 1, 2026-10-24; the 36.7M dof entries are ~1.0M cell pairs × 36).

The cell-level pattern was worth up to ~2 GiB on bfs3d; it scales with the entry count, so it is the one of
these that also bears on the 300k target. The stored transposes, a third such cut, are built (below).

**The Schur is no longer stored (2026-10-25, branch `claude/schur-storage`):** the smoother applies
`S = C − D (M G)` from its pieces and the fit forms only `S`'s diagonal and row norms, a chunk of rows at a
time (`solve-flow-block.md`, "`S` is never held"). After the lazy transposes it holds ~28 MiB less on a
bfs3d-sized fine level and its fit's transient ~131 MiB less (lattice). It is not bit-identical (the apply
sums in another order), so it is not strictly one of these cuts; measurements in `solve-flow-block-log.md`,
"The Schur applied from its pieces (2026-10-25)".

**The cell-level pattern is built for the probe, the padded layouts, nodal Galerkin maps and the SIMPLE
cuts (2026-10-24, branch `claude/cell-level-pattern`):** bfs3d device int32 2.19 → 0.40 GiB at step 1's
end, step-1 peak 8.07 → 7.19 GiB, resident 7.13 → 6.11 (`solve-direct-preconditioners-log.md`, "Cell-level
pattern (§14d), stages 1–3"). The probe's host block patterns (140 MiB on bfs3d) followed
(`CellLevelPattern`, branch `claude/host-cell-patterns`). Still dof level: the diagonal and cell-block
gathers (small), a non-nodal or diagonal-padded level, the 1×1 pressure block `C` and `S`'s row columns, and the plan's own COO coordinates
(`ColumnProbePlan.pattern_rows`/`pattern_cols`/`in_reach`, ~18 MiB host on bfs3d).

**The transposes are packed only for an adjoint (2026-10-24, branch `claude/lazy-transposes`):** bfs3d
device at step 1's end 1.20 → 0.69 GiB, step-1 peak 6.49 → 6.13 GiB (census runs), every forward leaf's
digest and step 1's rows unchanged; pitzDaily march identical, peak 3.085 → 2.900 GiB
(`solve-direct-preconditioners-log.md`, "Lazy transposes (§14d)"). Left in §14d: the fine Schur on its
reach-6 pattern (since built, above).

### 14e. BUILT (2026-10-24) — `BoundaryClosure.gradient_weight`'s unit tangents are broadcasts

**What.** In `aquaflux/schemes/boundary_closure.py`, `gradient_weight`'s tangent
`jnp.zeros_like(gradient).at[:, k].set(1.0)` becomes `jnp.broadcast_to(jnp.eye(dim, dtype=gradient.dtype)[k],
gradient.shape)`. Pinned by `tests/unit/test_boundary_closure.py::test_the_gradient_weight_compiles_to_no_per_cell_constant`
(red against the scatter form). `folded_constants.py`'s `RESIDUE_SCATTER_TANGENT=1` restores the old form in
its own process, to re-measure.

**Why.** XLA's constant folding evaluates the scatter at compile time and stores an `(n_cells, dim)` result per
component and field in every program that holds the residual (15.87 MiB in bfs3d's residual alone, cells ×
dim²); a broadcast is not folded. `validation/folded_constants.py` (on `50fe3be0`): bfs3d 15.87 → 0.05 MiB,
pitzDaily 1.87 → 0.01, residual bit-identical on both. **Measured gain on a bfs3d march is small**: all programs'
HLO 0.234 → 0.086 GiB, but the cache the clear frees 1.61 → 1.56 GiB and the 1-step peak 6.25 → 6.43 (inside the
0.4 GiB spread); one run per arm (log entry above). Kept because it costs nothing and removes a per-cell term.

### 14f. What the leading block re-fit holds at the march's peak (proposals, measured 2026-10-25)

**Evidence.** A 2-step bfs3d march peaks in re-fit #2's leading block re-fit: 6.80 GiB with the tracer on
`1bd8d292`, +1.06 GiB of live device arrays over step 1's end (log, "What sets bfs3d's 2-step peak"); after
#825, 5.82 GiB and +0.67 (log, "Fitting the leading block last"; one run each).

**What.** Three library changes, each to be measured with `memory_attribution.py`
(`BFS3D_MEMORY_CENSUS_STAGES="block re-fit #3"`, `BFS3D_MEMORY_SAMPLE_DEVICE=1`, two steps):
- CLOSED (2026-10-25): letting the probe's other float64 blocks go before the leading fit, by fitting it
  last, measured no lower (ledger, "Fitting a refit's leading block last"); the leading block itself is its
  fit's input, and the fit is computed in float64 by design;
- BUILT (2026-10-25): the superseded leading inverse is freed at the start of a re-fit (`refit_keeps()`,
  `solve.md`); live arrays at the re-fit's peak 1.36 -> 1.05 GiB, 2-step peak 5.80 -> 5.58 GiB, now set
  by the step program's first call (log, "Freeing the superseded split at the start of a re-fit");
- `fit_simple_level`'s 993 MiB of temporaries (as large as the step program's): fit per level or per row
  chunk.

**The superseded-inverse design is BUILT** (2026-10-25, measured on `main` `1c5a1875`, before #829 changed
the SIMPLE fit): the design, its safety guard and the pre-registered measurement now live in `solve.md`
(the early-release bullet) and the log, "Freeing the superseded split at the start of a re-fit".

### 14g. BUILT (2026-10-25) — `laplace_field` releases its per-call programs when it returns

`laplace_field` compiles its solve as a function defined per call under `jax.jit`, so the program goes with
it; it had handed per-call closures to a module-level `eqx.filter_jit(newton_step)`, which kept a program
per call, with its closures, for the process. bfs3d: the cache a run holds at its end is 0.17 GiB smaller
(two runs per arm); resident and peak move within the run-to-run spread, because the seed's last bits
also change the first fit's coarsening (`flow-log.md`, "laplace_field's per-call programs"). The Laplace
hierarchy's own `fit_levels` / `compiled_apply` / `_operator_spectral_estimate` entries (~0.01 GiB) stay.
A global `jax.clear_caches()` after setup was measured first and is not the fix (log, "How much of the
compiled-program cache a mid-run clear gives back").

### 14h. Two small allocator-side ideas (proposals, not built, 2026-10-25)

- **Collect the march's cycles when a solve returns.** After `staged_march` returns, its session and splits
  are cyclic garbage until Python's collector reaches them: 1.60 GiB of device arrays on bfs3d on
  `13fa17d4` (before #817--#825), 0.16 GiB on `39786723` (after #817--#819), each read by
  `memory_residue.py`'s `RESIDUE_CLEAR=end` as a `gc.collect()` + `malloc_trim(0)` before the cache clear
  (log, "What the resident memory holds beyond live arrays, on two meshes"). The resident drop at that step
  (1.0--1.3 GiB) includes the trim's and is not separated from the collection's. It costs what runs next in
  the same process (the adjoint, a second solve), never the march's own peak. Candidate: one `gc.collect()`
  at the end of `solve_coupled` / `solve_flow_march`, before `root_adjoint` is attached; measure the
  adjoint's peak with and without.
- **Trim after the step program's first call.** That call leaves 0.68 GiB (bfs3d) / 0.38 (pitzDaily) of freed
  memory in glibc's arenas (`memory_residue.py`, `RESIDUE_TRIM`, `13fa17d4`). The session already trims before
  every re-fit (#818), so this would not lower a march's peak, only what a march holds between its first
  step and its next re-fit. Worth doing only if a case's peak is ever set there, which bfs3d's is not.

## 15. BUILT (2026-10-25) — a fused level product for the traced hierarchies

Every padded-row product on the CPU is one fused, threaded Numba loop called from inside the compiled
program (`precondition/fused_product.py`; rule: `solve-amg-multigrid.md`). Back to back on a 4-core Linux
container: bfs3d 1853 → 1274 s, its step work outside the re-fits 1111 → 570 s; pitzDaily 265 → 220 s with an
identical step table. The kernel bound that motivated it (`validation/spmv_kernel_probe.py`), the spikes and
the measurements: `solve-amg-multigrid-log.md`, "The fused level product (2026-10-25)". Still unmeasured: the
Mac, where Numba's pip wheels usually offer only the `workqueue` layer, which keeps XLA's form.

## 16. A cheaper pressure probe on stretched meshes (bfs3d, 2026-10-26)

bfs3d ships pressure at reach 3, the rest at 2 (291 seeds per fit). A reach-2 pressure probe (234 seeds)
fails because its fit's COARSE levels go stale fast across the inner loop (`solve-refuted-directions.md`,
the 2026-10-26 entries). Raced on three perturbed bfs3d draws per arm (log, "THE REACH-2 RACE"):
- **c. BUILT, off by default, a wash on bfs3d — the strong couplings chosen from the Jacobian**
  (`precondition.StrongCouplings`, `JacobianProbeSpec.strong_couplings`), the library form of the
  raced geometric skeleton: 243 seeds, the same 17 steps on every draw, ~1 % of wall time (log,
  "`StrongCouplings` (library, 2026-10-26)"). bfs3d stays on `[2, 2, 2, 3, 2, 2]`. Unmeasured: the
  zero-shift adjoint under it, pitzDaily, other meshes.
- **a. Re-fit inside the inner loop at reach 2 — the proxy LOST, the real thing is unmeasured.**
  `refresh_on_cycles` 2 or 3 with every column at 2 took 25–51 steps and failed once. A real in-loop
  re-fit (the session re-fitting between inner iterates without ending the step) needs the traced inner
  loop broken on the host; not built.
- **b. Exact coarse operators** through the frozen aggregates, the fine level from a reach-2 probe.
  Unmeasured; the only direction aimed at the component that fails.
Records: `validation/reach_research/option4_diagnostic.md`, `option1_audit.md`.

## Smaller housekeeping

- **Compile on the critical path — CLOSED (2026-10-10), nothing station-specific to overlap.** See the
  ledger ("Compiling the next station ahead of time"). The 396 s quoted here first is gone: a cold
  pitzDaily run compiles before step 1 only, and a station change or a refit compiles nothing
  (2026-10-17, `9efc5759`; the compile row's paragraph under "Re-measured 2026-10-17 by profile").
- **First-order seed for the second-order residual — CLOSED (2026-10-10), even at best.** See the
  ledger ("Seeding the second-order march from a first-order solve").
- **Gradient reconstruction `sweeps` in the tangent — ALREADY ANSWERED; live on bfs3d only.** This entry
  was stale: `jacobian_gradient_sweeps` (narrowing the sweeps only in the copy of the residual the Krylov
  operator is differentiated from) is built, and on pitzDaily under `CorrectedGreenGauss` swept-2 gave the
  identical march 9.7 % faster, the knee at 2 (`schemes-log.md`). pitzDaily now ships
  `MultipleCorrectionGradient`, which has no sweeps, so it no longer applies there. bfs3d moved to
  `MultipleCorrectionGradient` too (2026-10-13), so it no longer applies to either case. `jacobian_gradient_sweeps` stays a script-only study arm and
  is NOT to become a case-file setting (project owner, 2026-10-12): the schemes without sweeps work better
  for this solver, so the direction is to move cases off swept gradients rather than to tune their sweeps.

## Where this file does not go

Not `.claude/rules/` (it would auto-load into every session); not `docs/` (user-facing, and these are
hypotheses); not only an issue (an issue is not greppable from a checkout, and the project's rule is
that findings live in tracked files). The right shape is this note as the backlog, one issue per entry
once it is in progress, and the result filed in a rule or the refutation ledger when it is measured.
