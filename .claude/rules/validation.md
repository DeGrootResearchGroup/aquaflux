---
paths:
  - "validation/**"
---

# Rules — `validation/` (the scientific cases and the study harnesses)

> **Provenance boundary (binding).** What you read here informs your understanding; none of it may reach
> the shipped surface (root `CLAUDE.md`, **Comment Convention**). The dated record behind this file —
> incidents, measurements, mutation passes — is in `.claude/notes/validation-log.md`.

## What lives here, and why it is fragile

- **Cases** (`*/compare.py`) — a full scientific run against a reference solution. Tens of minutes each.
- **Harnesses** (everything else) — single-state probes that measure one question and print a table. They
  are the re-adjudication instruments for most numbers in the `solve*.md` rules; a finding whose harness
  no longer runs can only be cited.

**Nothing in any test tier runs these files**, so the suite is green whether or not any of them works.
The recorded instance (2026-08-16, log): one pitzDaily case broken three ways at once — a settings object
it was not updated for, a tightened guard, no `sys.path` bootstrap — and also fallen far behind its
sibling's march machinery, all with the suite green.

## The obligation (binding)

A change to the library's public surface is not complete until checked against the cases that call it.
When you change a signature, default, settings object, type or seam:

1. **Run the static guard** (fast gate, milliseconds): `pytest tests/unit/test_validation_api.py`. It checks
   that every name a case imports exists; every literal keyword is accepted, including on a method of an
   imported class (`CoupledRANS.build(...)`, `SSTTurbulence.build(...)`, `MomentumContinuity.build(...)`);
   that no case function reads a module global only one branch of an `if`/`try` binds; and that every
   `__main__` script importing `aquaflux` puts the repository root on `sys.path` first.
2. **Use judgement on what it cannot see** (below), and if the change plausibly reaches a case, run it —
   `validation/run_case.sh <case>`.
3. **When you change the march machinery, carry it to the other step case too.** Improvements have
   repeatedly landed on one case and left the sibling behind; carrying it across is part of the change.

The `.githooks/pre-commit` reminder raises this when a commit touches `aquaflux/`; it does not block.

⚠️ Do not edit files under `validation/` while the fast gate runs: `test_validation_api.py` parses every
case and fails on a half-written file.

## ⚠️ What the static guard cannot catch

- **Attribute access on a returned object** — e.g. a method renamed on an object the case holds
  (`momentum.velocity_gradient` → `velocity_fields(flow).gradient` broke the OpenFOAM channel comparison
  after every solve, 2026-09-24).
- **`instance.method(kw=...)`** on an object the case built — its type is not known statically.
- **Semantic breaks** — a parameter that now means something else, a moved default, a changed type.
- **`**kwargs`** — `solve_coupled` forwards `**strategy_kwargs` to its builder. Given an explicit
  `strategy` or `RefreshPolicy(builder=...)` those keywords are refused by name; a wrong keyword reaching
  the default builder still raises only at run time.
- **Behaviour** — a runnable case whose march crawls looks identical to a working one.

Prefer the structural fix to the branch-global guard: have each arm of a case's configuration record its
settings under one shared name (`LEADING_SETTINGS`) beside the object it built, so the reader never
branches.

## The step cases' preconditioner settings

- **pitzDaily's leading inverse is `simplesmooth`**, as bfs3d's, since 2026-08-22: an incomplete
  factorization's behaviour on this saddle depends on elimination order and fill, unpredictably.
  Incomplete-LU preconditioning is no longer a direction the project takes (project owner, 2026-08-22);
  any `petsc`-bundle number recorded for pitzDaily is unreproducible on the current tree (log).
- **The trailing `[k, omega]` inverses differ between the cases**: pitzDaily `{max_coarse: 2000,
  equilibrate: False}`, bfs3d `{max_levels: 20, max_coarse: 200, strength_threshold: 0.25,
  aggressive_levels: 0, frozen_coarsening: True}`. Open question, never measured at the sibling's
  settings — the next thing to try if aligning them.
- **pitzDaily's leading `SimpleSmoothed` sets `frozen_coarsening: true`** (project owner, 2026-10-12,
  #290): at `strength_threshold` 0.25 its coarsening reads values, so unfrozen each refit recompiles the
  on-device split (`solve-field-split.md`).

## ⚠️ Traps when writing a harness (each produced a wrong result here; detail in the log)

- **A log parser keyed on the table's shape returns nothing when the logger gains a column.**
  `bfs3d_openfoam/march_log_compare.py` (also used by `step_policy_replay.py`) accepts the march log with
  and without `rho1` and refuses a log with no step rows. Any march-log reader must do both.
- **A persistent-cache hit also logs a `Finished XLA compilation` line**, right after `Persistent
  compilation cache hit`; `pitzdaily_openfoam/compile_timeline.py` reports hits separately (`timeline`'s
  fifth element), pinned by `tests/unit/test_compile_timeline.py`. Its `breakdown` splits each step at its
  first inner row (`step N set-up` holds the station change, the due refit and the step's dispatch) and
  at each redo, and sums tracing and lowering apart, since a cache hit still pays both; it needs the
  march logged with `"inner"` detail (pitzDaily's is) to see the split.
- **Build a reference state with `state_from_physical`, not `pack_state`** — `pack_state` takes the solved
  variables, so physical ω where `log(ω)` is expected gives a silently NaN residual from a finite-looking
  state, and every factorization then fails in its own misleading idiom. **Assert the starting residual is
  finite before measuring.** Do not copy a wiring idiom from a test without checking the case matches it.
- **A pivot census must read the factor, not the operator handed to it** — symmetric equilibration makes
  the operator's diagonal exactly 1, so `matrix.diagonal()` reports healthy pivots for every arm. Note a
  host factorization may store a pivot where another stores its reciprocal.
- **A harness that spawns a child must forward `kill` to it — use `validation/peak_footprint.py`
  (`run_with_footprint`), never a bare `subprocess.run`.** Otherwise `kill <pid>` leaves the
  `/usr/bin/time` child and its Python grandchild running, invisible to `run_case.sh --status`. `kill -9`
  still orphans; after an interrupted run check `pgrep -fl "usr/bin/time"`.
- **A timer that blocks on its own result also waits for earlier, untimed asynchronous work.** A JAX
  breakdown is sound only where the work before each timed piece is timed too; time a surprisingly large
  piece alone before believing it.
- **On Linux a standalone call whose temporary buffer exceeds 32 MiB page-faults all of it every call**
  (glibc's adaptive mmap threshold); a Krylov iteration inside the step does not pay this (its loop
  allocates once per solve), so per-call timings overstate it; **bfs3d's re-fit probe does pay it, per
  batched JVP**: 15M minor faults and 38.4 of a 58.3 s re-fit at glibc defaults, 36.5 s with the
  thresholds below, 28.8 s (0.15M faults, re-fit 38.6 s) with `MALLOC_ARENA_MAX=1` as well
  (`refit_profile.py`, `state-00029`, 4-core Linux, jax 0.11.2; `.claude/notes/validation-log.md`). Report
  `ru_minflt` per call, and the resident high-water mark beside every allocator arm (`refit_profile.py`
  prints both).
  `MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=2147483647` removes it **only up to 64 MiB**:
  XLA allocates on worker threads, whose arenas are 64 MiB heaps, so a larger buffer (the whole
  `(J + s) M v` program's, ~74 MiB on pitzDaily) is mapped per call regardless. To remove all of it run
  under jemalloc (`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
  MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1`) or `MALLOC_ARENA_MAX=1` **with** both thresholds
  (the single arena alone removes nothing: above the adaptive 32 MiB ceiling the buffer is still mapped
  per call), and say so; `run_case.sh` records all of them. ⚠️ **Both keep freed memory, so both raise
  the footprint: never run a bfs3d march under them on a 15 GB machine** (jemalloc: VmHWM 13.2 GiB after
  one bfs3d re-fit against 10.9 at defaults, 2026-10-18). **`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`
  removes the faults without keeping memory** (glibc asks for transparent huge pages on its mapped
  chunks; needs THP `madvise` or `always`): bfs3d probe faults 15M → 66k, re-fit 48.7 → 39.7 s, VmHWM
  10.98 against 10.86 GiB (`state-00032`); on a whole march (cadence 5, without the pivot floor) solve 2439 → 2180 s, peak 12.63 →
  12.15 GiB, identical step table. Not a `run_case.sh` default (timings would stop being comparable). The retention that rules the others out:
  `MALLOC_ARENA_MAX=1` with both thresholds raised held 12.3 GiB
  resident by step 2 against 9.5 GiB at defaults, and the container restarted (bfs3d session,
  2026-10-18, branch `claude/bfs3d-mesh-home`); safe for fixed-state probes only. Compare a standalone piece with an in-loop one by the MEAN of as many calls as a solve
  makes, not a best call (`krylov_overhead.py`). Through `mallopt` both thresholds are C `int`s: a 4 GiB
  trim threshold wraps to 0. Measurements: `.claude/notes/solve-open-directions.md`, "Inside the step
  program (2026-10-18)".
- **A harness that builds a step must use the case's own shift settings, and form the shift through
  `ShiftTerm.shift(beta)`, never `beta * term.diagonal`** — the latter drops `row_shift_scale`, where
  `turbulence_damping` lives.
- **A harness that applies the march's preconditioner must apply what the step holds, and take it as an
  argument of anything it compiles.** A shipped field split is a traced copy taken at each fit; a refit
  reaches the step only through the replacement `refresh_preconditioner` returns, and a split captured in a
  closure keeps applying the first fit silently. `pitzdaily_openfoam/replay.py`'s `MarchReplay.refit` does
  this; `MarchReplay(host=True)` applies the host object for probes that mutate it in place. Probes that
  need the host inverse read `session.preconditioner.inverse` from `open_session`.
- **Replay a step at the tolerance the march used.** The march solves a step after a setback to 0.1, not
  the regime's 0.3 (`AdaptiveLinearTolerance`, library default since #645); `MarchReplay.tolerance`
  rebuilds it from the checkpointed reports and `march_solver(measure, rtol)` applies it.
  `krylov_recycling_probe.py`, `precision_replay_probe.py` and `batch_bound_probe.py` still build
  fixed-tolerance solvers.
- **A harness that calls `solve_linear` outside a Newton step must bind the measure**
  (`solve.linear.in_progress_measure(engine.linear_solver(), measure)`); the step's GMRES is built with
  `norm=None` and raises elsewhere.
- **Print one line per outer step, flushed; print the operating point (mesh size) before measuring; and
  print only settings actually in force.**
- **Do not gate a loaded checkpoint on its own recorded `residual_norm`** — it is in the row-scaled measure
  the march steered by (default since #370), not a Euclidean one (bfs3d `state-00069`: a factor of 395).
  Gate against the case's own self-start in one norm.
- **A saved `.npz` is not necessarily a checkpoint** — a cached self-start is not a converged root. Read
  what wrote a state before trusting it.
- ⚠️ **bfs3d's leading coarsening moves under rounding** (`strength_threshold` 0.25 reads values), so a march
  A/B of two checkouts differing in rounding anywhere upstream of the first fit — a sparse-product storage
  change, a gradient inverse, a jax upgrade — is not a controlled pair. Compare a preconditioner's cost at
  one fixed checkpoint (`bfs3d_openfoam/split_apply_timing.py` with `BFS3D_SPLIT_TIMING_STATE`; its level
  table must agree between runs) and read a march pair's counts as two samples. pitzDaily happened not to
  move (2026-10-15, PR #721).
- ⚠️ **A single bfs3d march is one draw from a wide spread: judge a change on several perturbed draws per
  arm** (failure count and step spread). A last-bit viscosity change (`1 ± 1e-14`) moves it from ~20 to ~80
  steps or to a failure, so a one-run-per-arm bfs3d A/B -- the 2026-10-14 guard choice, PR #722's reach-2
  cost, #742's "doubling" -- cannot rank arms closer than that. The cause: wall cells whose shifted `k`
  pivot nears zero make the **float32** field split's `A·M` wrong by order one around them, so the
  step's solve stalls at its cap and its direction is decided by rounding. `BFS3D_PIVOT_FLOOR=0.15` /
  `PITZ_PIVOT_FLOOR` (`MaterializedJacobian.pivot_floor`, `solve.md`) removes it: 21 steps both at `1 ± 0`
  and `1 + 1e-14` (2026-10-18, off 38), pitzDaily unchanged at 21; both case files set it since 2026-10-19, so a pre-2026-10-19 bfs3d step count is unfloored.
  `pivot_floor_probe.py` reads it at a checkpoint (float32 vs `BFS3D_SPLIT_PRECISION=float64` vs floored);
  `step_descent_probe.py` and `corner_k_row_probe.py` read the `k` rows. Tables:
  `.claude/notes/solve-log.md`, "Near-zero pivots and the float32 split (2026-10-18)".
- **A fixed-state replay must build its step as the viscosity-ramp march does**: the session re-pointed to
  the anchor companion and the step built at the anchor's seed (the step keeps that shift diagonal and
  that frozen coarsening for the whole march, #701), then re-pointed to the station
  (`step_descent_probe.py`, `pivot_floor_probe.py`). A step built on the target replays a different step.
  At a step decided by rounding even that is not enough: the march hands each step the residual vector the
  previous one returned (#761), and a freshly evaluated one differs at round-off, which flipped one such
  step from descent to a tenfold rise. Reproduce such a step from a capture of its `_march_step` arguments,
  or read it as a spread over perturbed residuals.
- **A per-column probe reach is licensed by the gradient scheme and the split; re-audit it when either
  changes** (`probe_reach_audit.py`, widest pattern 3). bfs3d ships `MultipleCorrectionGradient` at a
  uniform reach of 3 — p, k and ω carry 1e-3–4e-5 of their norm beyond reach 2, and a reach-2 colouring
  cost it 18→21 steps / 137→216 cycles (PR #722; one run per arm, so a single draw -- see the ensemble bullet above). pitzDaily deliberately probes every column at reach 2
  inside its reach-3 pattern: its march is unchanged and the probe 40 % cheaper (2026-10-14).

## Machine contention and run records

- **A test tier and a case must not overlap**: `tools/fastgate.sh` refuses beside a live case (exit 3) and
  beside another tier (exit 4) — root `CLAUDE.md`, "Use the blessed command". ⚠️ The reverse is still open:
  **a case cannot see a running tier** (no tier writes a run-file), so run `validation/run_case.sh` only
  after checking no gate is running.
- **A contended run is good evidence about counts and worthless about seconds.** Steps, cycles,
  escalations, line-search clips and the converged root are deterministic; wall clock moved 2.4–4.6x on an
  identical trajectory, and the cost is mutual (the gate slowed 1.5–2x). Keep its counts, discard its
  timings, as for a run that spanned a sleep (measurements in the log).
- **A timing does not travel between checkouts either**: the JAX compilation cache is keyed on the
  compiled program, so the first run on a new branch partly measures compilation. A cold-cache premium and
  a contention premium are not separated by any single run.
- **A case rerun on one mesh can keep its small programs**: `AQUAFLUX_COMPILATION_CACHE_MIN_COMPILE_SECS=0
  validation/run_case.sh ...` stores the ~980 op-by-op programs the 2.0 s default recompiles every run
  (pitzDaily warm start 89 s -> 63 s, 2026-10-17). Per run only; a mesh-changing study pays and gains nothing.
- ⚠️ **`pgrep -f "tools/fastgate.sh"` matches its own watcher.** Match the real process with watchers
  excluded, or read the gate's log (`$TMPDIR/aquaflux-tests-<worktree>-fast-*.log`, summary on the last
  line).
- **Prefer a run's own record to an account of it** — the gate's log, `run_case.sh`'s run-file,
  `git show --stat <sha>` — over a peer's report, recollection or the process table. A PR's author field
  identifies the account, not the session; authorship is settled by which worktree holds the code.

## `run_case.sh` — how it reports and stops a run (pinned by `tests/unit/test_run_case.py`)

- **`--wait` exits with the case's own status** (before 2026-09-22 it exited 0 for crashed cases; a
  `--wait` status read before then says nothing). The case runs inside a wrapper subshell that writes
  `run-<stamp>.exit` beside the log and a `[run_case] the case exited with status N` footer; a signal death
  is `128+N` (OOM SIGKILL → 137). **A missing `.exit` file is failure.** The status lives in a sibling
  file because `live_pid` deletes the run-file once the pid is dead.
- **`pid=` is the wrapper's pid.** `kill <pid>` stops the run: the wrapper traps TERM/HUP, forwards them,
  and re-`wait`s until the case exits. **`kill -9 <pid>` kills only the wrapper and orphans the case.**
- **Liveness is the pid and its start time** (`pid_started=`, `ps -o lstart=`, compared through
  `canonical_time` — blanks squeezed and trimmed, since macOS pads `lstart`); a reused pid reads as stale.
- `caffeinate` watches the wrapper (`-w`); the wrapper's stdio is detached. `--force` skips taking the
  memory/load readings. `free_gb` reads macOS `vm_stat` or Linux `/proc/meminfo` `MemAvailable`.
  `AQUAFLUX_CASE_POLL_SECONDS` (default 20) sets the poll interval. `run-*.exit` is gitignored.

## Every case with a case file solves from it

The channel harnesses call `checked.solve(coupled)` and hold no solver setting; pitzDaily and bfs3d read
their march from `case.yaml`'s `solver` section, each `PITZ_*` / `BFS3D_*` march variable an edit of it
(`case.md`, "The case files in the repository"). For a harness:
- **A module constant a probe imports (`compare.INNER_STEPS`, `compare.CONTROL`, ...) is read back from the
  file.** Change values in `case.yaml`; bfs3d refuses to import when its reassembled `SOLVER` differs from
  the file's with no override set.
- **A setting cannot be passed to a case's `solve` as a keyword.** Vary one by editing the spec
  (`dataclasses.replace`); a study of something no file can state is a script-only arm calling the library
  (pitzDaily's `_solve_study_arm`).
- pitzDaily's `PITZ_*` march overrides were inert from #437 (2026-09-24) to 2026-10-08 (pinned since by
  `test_a_pitzdaily_march_override_reaches_the_solve_not_only_the_banner`): a pitzDaily arm in that window
  that set one ran the file's solver.
- `turbulent_channel/report.md` was regenerated 2026-09-24; Re_b = 240000 converges on current code.

## bfs3d's mesh — one tracked copy

- **`validation/bfs3d_openfoam/of_case/constant/polyMesh` is the one copy** (23040 hexahedra), where
  `blockMesh` writes it, so the flow case owns its geometry and the species case reaches across to it, as
  #662's dependency edge would. `case.yaml` names it, `compare.MESH` reads it from the file,
  `run_of.sh` / `run_transient.sh` run on it, and `run_of.sh --blockmesh` regenerates it in place (a
  regeneration that differs shows in `git status`; on 2026-10-17 its node lines matched the current
  `blockMeshDict`'s grading to the file's printed precision, 5e-11 m). The OpenFOAM fields are not
  tracked there; without them `compare.py` marches and skips the comparison.
- **A shipped case file may name only a tracked path or one declared in
  `tests/unit/test_shipped_case_paths.py`'s `GENERATED`** (with what writes it), fast tier; the bfs3d
  mesh once sat at an ignored `runs/kwsst/polyMesh` while tracked twice in `bfs3d_species/`.
- ⚠️ `bfs3d_species/of_case/` still tracks a run's scratch output against its own `.gitignore` (`2000/`,
  the frozen flow, its only tracked copy; `2600/s`; `constant/` property copies; `log.foamRun`), and
  `runs/species/{s,phi}`; its `run_of.sh` deletes `constant/` and `2000/` when it runs.

## `bfs3d_species` — dependencies

A passive tracer on `bfs3d_openfoam`'s flow (`transport.md` has the design).
- **It has no mesh or flow of its own.** `of_case/run_of.sh` assembles a scratch case from
  `bfs3d_openfoam/of_case`; `compare.py` loads that case's `build_case` **by path** (`importlib`), because
  both cases have a module named `compare`.
- **Its own-flow arm reads the flow case's rolling checkpoint**, which predates the production limiter
  defaulting off and so is not a root of today's residual. That does not invalidate the arm: it needs a
  flux that closes discretely, which `compare.py` measures directly.
- **`tests/integration/test_bfs3d_species.py` is the one test that reaches into `validation/`**; it skips
  without the case data and pins sub-patch injection, conservation, boundedness and mixing.

## `ray_effects_room` — aquaflux against discrete ordinates in a far-UVC room

A demonstration case: a 4 x 4 x 3 m room, a Care222 module (`ushio_b1.ies`, MIT licence), a lattice bunny,
black or 0.5-reflecting walls; aquaflux, of-optical-radiation's discrete ordinates (DOM) and an independent
brute-force reference (`reference.py`, numpy + Embree, no aquaflux import) on one mesh. Its README carries
the numbers.
- **aquaflux's side is case files** (`cases/*.yaml`) run by `run_cases.py` (`RAY_CASES`, `RAY_OVERWRITE=1`)
  through `run_case.sh` into `work/cases/<name>/`; `figures.py` reads them through `outputs.py` (a numpy VTK
  decoder independent of the writer); `slices_refined.py` is the one harness left. `work` may be a symlink
  (`.gitignore` has `/work`). `run_case.sh` records `RAY_` settings (`test_check_env_prefixes.py`).
- **DOM needs of-optical-radiation's `add-incident-flux-output` branch**; `generate_dom.py` records the
  commit beside every run, names its Docker containers and stops them on SIGTERM/SIGHUP.
- **`snappyHexMesh` writes `polyMesh` before the lamp patch is cut** — each meshing step leaves its own
  completion marker. **The cell count is not `owner.max() + 1`** (top cells may appear only as neighbours);
  `cells.npz`, written by OpenFOAM, is the authority.
- ⚠️ **of-optical-radiation's DOM grid is uniform only when `nPhi == nTheta`** (`n x n` gives `2 n^2`
  directions, matching Fluent's `n/2 x n/2` per octant). Every figure quoted "at 64 / 256 directions"
  (`8 x 4`, `16 x 8`) is on non-uniform grids, here and in `sozzi_radiation/` before 2026-10-02.
- **State the error window with the number** — over the central 2 x 2 m the DOM floor error rose with
  directions, over the whole floor it fell.
- **Pixels are a run parameter (`bunny:16x8/3x3`) and every record carries them.** No effect measured on
  this hex-dominant, axis-aligned mesh; untested on tet or polyhedral meshes.
- `sozzi_radiation/background_grid.py` gathers at Cartesian grid nodes and interpolates; it needs the
  sleeve's own value (G = 2M at a diffuse emitter's surface) as data. Run at most five tracker runs at once
  in Docker's VM.

## `tubular_reactor` — ten lamps on a background grid

Radiation only: 10 Sozzi lamps in a 300 mm vessel, a 100-hole perforated inlet plate, walls reflecting 0.1;
`G` at Cartesian grid nodes (`run.py`, `TUBULAR_*`, outputs `work/runs/<tag>/`); `study.sh` and `compare.py`
run and compare the studies; numbers in its README.
- **The plate is a bespoke exact `Body` (`geometry.PerforatedPlate`)**: a line through the slab is clear only
  if its in-slab part lies in one hole (one hexagonal-lattice lookup). ⚠️ Modelling the holes as 104 convex
  regions in an `Outside` reached a 54 GB footprint at the 4M-pair default.
- **Lamps are emitters only; their shadows are separate exact bodies** shrunk by a fixed `TUBULAR_SHRINK`
  (0.251 mm), so a lamp-tessellation study does not move the shadows.
- **The solve is hand-assembled, not `solve_scene`**, because `solve_scene` puts every lamp facet into the
  dense transfer and 50k lamp facets cannot be held.
- Within 2 mm of a sleeve the grid-interpolation error is 17–36 % at every spacing (#726), and the
  node-averaged mean G is not a converged volume integral. `render.py` draws figures from 0.5 mm pixels
  masked by the exact water test; plate-plane figures are direct light only.

## The aggressive Reynolds continuation does NOT need the complete LU

`pitzdaily_openfoam/compare_reynolds_continuation.py` runs the aggressive `DualTimeControl(beta_start=0.5,
beta_min=0.005)` continuation on `case.yaml`'s field split with `refresh_on_cycles=2`; it converged to the
developed reattachment (2026-10-11, measurement in the log), so the complete LU it once claimed to need was
deleted. ⚠️ The harness puts only its own directory on `sys.path`, so from a second worktree set
`PYTHONPATH` to that worktree or it runs the editable install's code.

## Recovering a converged state (both step cases)

Both `compare.py` files take `checkpoint_dir` and write a rolling per-step state through
`StateCheckpointer` + `combine_observers` (`PITZ_CHECKPOINT_KEEP` / `BFS3D_CHECKPOINT_KEEP`, default 3,
into `<case>/checkpoints/`). Without one, every question about the zero-shift operator — the adjoint's,
which a march never exercises — costs a re-march. `zero_shift_arms.py` / `zero_shift_adjoint.py` consume
them.

## Other live traps

- ⚠️ **Without the pivot floor, bfs3d's march at `refresh_on_cycles` 3 failed on a 4-core Linux container
  (`main` `fe772f59`, 2026-10-18)**: at step 35's re-fit the trailing hierarchy refused one corner cell's
  `[k, ω]` block as singular (`k` ~1e-17; the block is triangular and invertible, and the guard's Hadamard
  ratio is not column-scale invariant). With #766's `PivotFloor(0.15)` it converged (32 steps at 3; 17 at
  the file's 4 since 2026-10-19), by never reaching such a state: the floor does not cure the refusal,
  which the same state still raises on `main` with the floor on. Detail: `.claude/notes/validation-log.md`,
  "The re-fit cadence on this machine class, and a corner cell's `k`".
- **Before attributing a local-only failure to your platform, check CI's skip counts** (root `CLAUDE.md`,
  "CI is not a superset of a local run"); a 2026-08-19 "platform difference" was a skipped test.
- **`_mis_aggregate`'s return annotation is stale**: `tuple[np.ndarray, int]`, but it returns three values
  (labels, roots, count).
- Comparisons between archived runs from before 2026-08-19 may pair a k-guarded march with an unguarded
  one; today every coupled march goes through `_coupled_step`'s per-cell k clip and ω step bound
  (`omega_step_ratio`, overridden by `BFS3D_OMEGA_STEP_RATIO` / `PITZ_OMEGA_STEP_RATIO`, `none` =
  unbounded).
