# Investigation log — `aquaflux/parallel/`

> Reference-only, never auto-loads. Dated measurements and incident history behind
> `.claude/rules/parallel.md`; the rule holds the current state.

## From `parallel.md` — the hand-built per-device body (why the injected-assembler rule exists)

An earlier version of `distributed.py` hand-built `FaceCellConnectivity` + `MeshGeometry` + `FieldContext`
and called `DiffusionFlux` directly — a second implementation of `ResidualAssembler`. It drifted silently
(the `MaterialModel` → `PropertyModel` rename broke it and no test caught it, because there was no shared
code path to break), it hardcoded one operator, and it substituted a pre-baked constant per-face
`boundary_value` array for the boundary closures — which can only express constant Dirichlet, so
`Neumann`, `ZeroGradient`, `Convective`, and wall functions were all silently out of reach.

## From `parallel.md` — closed padding traps (history)

- Periodic seam: both face renumberings used to drop `neighbour_offset`, which raises nothing (geometry is
  gathered), surfacing only as a wrong owner→neighbour displacement in whichever operator next read the seam.
- Padding faces were labelled `"boundary"` until the coverage check (#354) landed.
- The earlier code gave *every* cell volume `1.0`. Diffusion never reads volume, so it was inert — but the
  first transient term or volume source would have been silently wrong, with no error and no `NaN`.

## From `parallel.md` — the simulated-devices spawn helper (2026-08-22)

The device tests simulate 4 CPU devices via `--xla_force_host_platform_device_count=4`, which must
be set **before JAX initializes** — hence the subprocess. Spawning it has exactly one home,
`tests/support/devices.py` (`run_on_simulated_devices`), and that is not tidiness: `XLA_FLAGS` is a
single string of several entries, so the device count must be **appended**, and a child that assigns
it instead drops the test workflow's `--xla_cpu_multi_thread_eigen=false`. The child then starves a
*different* worker, so nothing about the failure names the file that did it. Which is how it came to
be fixed in one of five copies (2026-08-22) and left in the other four for three days: the fix was
made where the symptom appeared, and the copies had no reason to be looked at. Whether a clobbering
child was co-running at either CI crash below is **not** established — the demonstrated cause there is
the timeout — but a job-wide flag that four of five children silently discarded is a defect on its own
terms. `tests/unit/test_simulated_devices.py` pins the append, and fails if the assignment comes back.

## From `parallel.md` — device-test cost and the CI timeout history

⚠️ **These are the most expensive tests in the fast tier, and what they cost is set by the machine,
not by the test.** Each pays a full `shard_map` compile. Measured 2026-08-25 on macOS arm64, 11 cores,
jax/jaxlib 0.10.2, x64, `OMP_NUM_THREADS=1` and Eigen multi-threading off, running the file alone at
system load ~5: **42 s** for `test_distributed_gradient`'s forward check and **17 s** for its control.
The same two tests on the same commit took **901 s** and **521 s** at system load ~65 (another session
was saturating the machine) — a run that is void for cost, but the *ratio* is the point: an order of
magnitude, with no assertion and no log line to say the machine was the cause.

That is what put these tests near the per-test timeout on a hosted runner, where four xdist workers
and their spawned children shared four cores. Six of them land in **one** worker's initial
`--dist load` chunk, so they also ran back to back rather than spread out. At the 300 s timeout the
workflow used until 2026-08-25, one crossed it **twice in three days** — and `--timeout-method=thread`
cannot interrupt a test, so it *kills the worker*, which xdist reports as `node down: Not properly
terminated` and `worker 'gwN' crashed while running <test>`, indistinguishable from an OOM kill and
with no mention of a timeout anywhere.

**⚠️ RAISING THE TIMEOUT TO 900 s DID NOT FIX THIS — it crossed that too.** On 2026-09-09
`test_distributed_swept_gradient_adjoint_matches_and_unsupported_solves_refuse`, which costs ~241 s
when the tier is healthy, went silent for **1074 s** and killed its worker, while its neighbours in
the same run finished at their normal 139–390 s. A timeout is a deadlock guard; it was never the
thing to tune, because the cost was contention and a bigger allowance just moves where it bites.

**These tests therefore carry `pytestmark = pytest.mark.distributed` and run in their own CI job at
`-n 2`** — two workers each forking one child is four processes on four cores, where `-n auto`
alongside the other ~1300 unit tests was eight. Measured on the same eleven tests, isolation is worth
3–4x per test: the one that died drops to 108 s, and 390 s / 171 s / 142 s become 95 s / 79 s / 42 s.
Membership is by marker rather than by path, and `tests/unit/test_distributed_marker.py` fails if a
module importing the spawn helper lacks it, so a new one cannot quietly rejoin the contended job. The
local fast gate still runs them inline — the split is a property of a 4-core runner, not of the tests.

Before adding another device test, account for its cost. The cheapest lever is fewer subprocesses, but
each one's peak memory is why the existing checks were split across them in the first place, and
`-n 2` above is sized to the runner, not to the test count — raising it re-creates the contention.


## From `parallel.md` — partition adjacency measurement (collective choice)

On a 40³ cube a min-cut (Scotch) partition at 32 parts borders between 4 and 14 other partitions,
non-uniformly.
