# Investigation log — `aquaflux/discretization/`

> Reference-only, never auto-loads. Dated measurements, verification runs and probes behind
> `.claude/rules/discretization.md`; the rule holds the current state.

## From `discretization.md` — declared-inputs mutation check (2026-10-09, #362)

Before #362 each term family carried the subset of the declarations it happened to need (sources had no
`uses_gradient`, schemes no `requires`, transient and momentum sources neither), and the assembler checked
`uses_gradient` over fluxes only.
    Mutation-checked 2026-10-09: emptying `DiffusionFlux.requires`, flipping `LimitedUpwind`'s or
    `DiffusionFlux`'s `uses_gradient`, and adding a stale `requires` to `UniformBodyForce` each fail
    it; the assembler-level sweeps (sources, transient, `AdvectionFlux`'s `requires` delegation) are
    pinned in `test_residual_assembler_validation.py`, each mutation-checked the same way.

## From `discretization.md` — `CellBalance` extraction gated on bit-identity (2026-08-15)

  - **Verified, and the extraction was gated on BIT-IDENTITY rather than on tests passing** (2026-08-15,
    at the base `efc10e3`): the coupled `bfs3d` residual at `state-00067`, the momentum residual with
    and without the wall-function eddy viscosity, `mdot` and `a_P` — all bit-unchanged; plus 97 arrays
    over a 2D/3D × compact/corrected × Stokes/upwind/limited × pin-on/off sweep. Stub-operator scatter
    is conservative and correctly signed (`test_cell_balance.py`).
    ⚠️ **Method note for any future bit-identity comparison: include quantities the change should NOT
    touch.** The first run of that harness showed order-one differences in `mdot` and `a_P`, which this
    change cannot reach — the cause was `hash()` on strings being randomized per process, so the two
    runs drew different random states. A residual-only comparison would have read as a real regression.

## From `discretization.md` — the AD-linearized limiter result

- **The AD-linearized-limiter result (`tests/integration/test_limited_advection.py`).** The
  reference **lags** the limiter (freezes `psi`, adds the limited term as an explicit RHS,
  `coeff.F90` line 326) and converges only *linearly*. Writing `psi(phi)` into the residual and
  letting AD linearize it puts the limiter in the Jacobian and recovers **quadratic** Newton
  convergence (measured: ~3 steps vs the lagged ~5 — under the old `eps² = vol K³` softening at
  K = 1, 40 cells; the test now runs `softening` 0.05 at scale 1 and still asserts AD < lagged),
  while staying differentiable (IFT) and giving 2nd-order accuracy — the "after" to the reference's
  "before". Boundedness: the smooth limiter *damps* over/undershoot rather than strictly eliminating
  it — the smoothness is what makes it AD-linearizable. (The "~halves it" once recorded here was at
  K = 0.3 under the old form; the top-hat test now runs `softening` 0.01 and asserts < 0.7×.)

## From `discretization.md` — `DiffusionFlux.boundary_coefficient` (k wall fade)

Fading the boundary face value toward the owner value instead of fading the coefficient was measured to
diverge for the k equation (the state-dependent face value contributes `d(phi_ip)/d(phi_P)` to the
linearization and can drive it past one).

## From `discretization.md` — ⚠️ Single-threaded kernels in the compiled residual and its JVP are NOT a lever (measured 2026-10-17)

The `dot` entry in the root briefing found a reduction-rooted fusion running on one thread at 40 % of a
reconstruction's kernel time. This asks the same question of the whole coupled residual and the JVP the
Krylov matvec and the coloured probe are made of, and the answer is no. Harness:
`validation/fusion_profile.py` (`FUSION_CASE=pitzdaily`). It joins the compiled HLO (fusion kind,
`outer_dimension_partitions`, root opcode, bytes touched) to per-kernel times from
`jax.profiler.trace`. Configuration: `main` `183e9ffa` (adjugate first pass already in), pitzDaily
`case.yaml`, `hybrid_initialize` state, 61125 unknowns, `jax.jit` over `eqx.partition`ed params,
4-core Linux x86_64 container, jax/jaxlib 0.11.2, x64, CPU, Python 3.13.

- **The instrument.** The thunk executor runs independent kernels concurrently: on 4 cores the JVP's
  kernel events sum to ~2× its span, and a lone 0.75 MiB copy reads 550 µs. So per-kernel cost comes
  from a second pass with every thread pinned to one core, timing only *solo* events (no other kernel
  overlapping). The compiled program is identical there (315 partitioned fusions either way, since
  partitioning is decided at compile time). The solo times sum to within ~10 % of the one-core wall.
  The parallel trace contributes **time alone**: how long each kernel was the only one in flight. That
  is the only time an unsplit kernel leaves cores idle. **Verified on a known cost:** with
  `FUSION_ARM=lapack` (batched LU restored in the first pass) the 15 custom calls (5 `getrf` + 10
  `trsm`) are the largest class at 13.2 ms of one-core time, against a 14.7 ms one-core residual
  delta.
- **XLA's rule** (`ParallelTaskAssignment`, `xla/service/cpu/parallel_task_assignment.cc`, openxla main
  read 2026-10-17; the installed jaxlib carries the class and its log strings). A memory-bound kernel
  (flops/byte ≤ 1) gets `min(ceil(sqrt(cores)), bytes_accessed / 256 KiB)` parts, so **at most 2 on 4
  cores**, and only once it touches 512 KiB. A compute-bound one gets
  `min(cores, (flops + 10·bytes) / 1e5)`. Custom fusions, `dot` and custom calls are never split. The
  56 unsplit ≥ 0.75 MiB `copy`s (each a copy of one shared zero broadcast feeding an in-place
  scatter-add) are elementwise and above the threshold. That they carry no partitions is *inferred* to
  be copy insertion running after the pass, not read in the source.
- **Clean JVP** (page faults removed, see below): 11.1 ms on 4 cores, 33.0 ms on one, so parallel
  efficiency is 75 %. One-core solo time 32.3 ms over 952 kernels:

  | class | kernels | ms | share |
  |---|---|---|---|
  | partitioned (2 ways) | 312 | 23.90 | 73.9 % |
  | under 512 KiB touched | 537 | 4.05 | 12.5 % |
  | `dot` (batched 2×2 / 3×1 per cell) | 42 | 2.87 | 8.9 % |
  | other (the post-pass copies) | 56 | 0.83 | 2.6 % |
  | `__ynn_fusion` (reduction-rooted) | 3 | 0.64 | 2.0 % |
  | scatter-rooted | 2 | 0.02 | 0.1 % |

  Unpartitioned: 640 kernels, 8.4 ms (26 %). Half of it sits in the largest 72, and **the largest is
  0.9 %** of the program (a `__ynn_fusion`, 292 µs). On 4 cores an unpartitioned kernel is alone in
  flight for **0.88 ms of a 12.7 ms span (7 %)**, spread over 16+ kernels. The residual is the same
  shape: 28 % unpartitioned over 344 kernels, the largest 1.8 %. **No kernel is worth more than ~1 %,
  so there is nothing for a local rewrite to hit.** The census's "537 single fusions" is mostly
  kernels too small to split, and their time is small too.
- **The precedent's own mechanism was measured anyway, and it lost.** The JVP's three
  `__ynn_fusion`s are the skewness `einsum("f...j,fj->f...")` in `schemes/interpolation.py`
  (`interpolate_to_face`) and `jnp.sum(S * S, axis=(-2, -1))` in `turbulence/strain.py`. Respelled as
  `vectors.dot` / `norm_squared`, the kernel structure moves as predicted: `__ynn_fusion` goes
  3 → 0 (JVP) and 2 → 0 (residual), with partitioned fusions 312 → 315. But wall time does not
  follow. Three interleaved pairs, best of 30, mmap threshold raised: residual 5.22 / 5.40 / 5.25 →
  **6.11 / 5.73 / 5.75 ms (slower)**, JVP 9.83 / 9.86 / 10.23 → 10.08 / 9.92 / 10.39 ms. Results moved
  by a rounding of the summed terms (2e-17 and 8e-18 of max |value|). **Not shipped.** The precedent's
  win came from three kernels being 40 % of their program, and these are 2 %. The 42 `dot`s are 36 in
  `schemes/multiple_correction.py` (`_one_exact`, `reconstruct`) and 6 in `interpolate_to_face`. As
  elementwise multiply-sums (a monkeypatch, since that file has another owner) they go 42 → 2 while
  unpartitioned loop fusions only drop 477 → 461, and the JVP wall stays within noise.
- **⚠️ THE LARGEST THING IN A STANDALONE JVP TIMING ON LINUX IS PAGE FAULTS, AND THE MARCH DOES NOT
  PAY THEM.** The JVP's temporary buffer is 42.8 MiB, the residual's 22.6 MiB. glibc's mmap threshold
  adapts upward only to 32 MiB on 64-bit, so the JVP's buffer is `mmap`ed and `munmap`ed on **every
  call**, about **11 000 minor faults/call** (every page). The residual's buffer is reused (0–354).
  Fixing the threshold (`MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=2147483647`, or the
  harness's `FUSION_MMAP_THRESHOLD_MIB`) takes the JVP from **16.5 → 9.8–11.1 ms** with 10–50
  faults/call and identical results. The residual is unchanged. The faults land in whichever kernel
  touches a page first, which in the default run inflated the copies (5.8 ms) and the small kernels
  (11.9 ms) and made unpartitioned look like 44 % of the JVP. **On the shipped march it buys nothing:**
  `compare.py`, default 183 s (a 212 s first run was a cold compilation cache) against raised
  thresholds 180 s and 186 s, with identical step tables (21 steps, 106 cycles, `x_r/h` 8.0686,
  `run_case.sh` records no `MALLOC_` variable). The matvecs run inside the compiled step program,
  which (inferred) allocates once per step, not once per matvec. So **a per-call JVP time measured
  standalone on Linux overstates the march's matvec by ~40 % at this size**. Quote it with its fault
  count, and do not project a march gain from it. macOS's allocator is a different mechanism and was
  not measured.
