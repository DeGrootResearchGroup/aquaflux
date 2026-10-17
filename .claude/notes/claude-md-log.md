# Dated history behind `CLAUDE.md`'s rules

> **Lives in `.claude/notes/`, outside the auto-loaded `.claude/rules/` tree, so it never auto-loads.**
> It holds, verbatim, the dated narratives, measurements and incident accounts that were condensed out of
> the root `CLAUDE.md` (2026-10-17) so that file carries only the rules and a sentence of why. Each heading
> names the `CLAUDE.md` section the text came from. Read it before re-adjudicating one of those rules;
> statements here are as of their dates, not current fact.

## Preamble — why `paths:` is mandatory on every rules file

⚠️ **`paths:` is MANDATORY on every file under `.claude/rules/`, and an absent one is the opposite of
"scoped to nothing" — it is scoped to EVERYTHING.** Three reference-only files sat there without it,
each opening with the words "this file never auto-loads", and all three loaded into **every** session
in the repository regardless of what was being worked on: ~285 KB, roughly 70k tokens — over a third
of the context window — spent before the first tool call, on top of this file's own ~24k. Nothing
reports this; a session simply starts near its limit and the budget is gone by the first edit. So
**reference-only material goes in `.claude/notes/`** (tracked, greppable, read on demand), never in
`.claude/rules/` with the frontmatter left off. `tools/check_rules.sh` reports any rules file missing
`paths:`, and `tests/unit/test_check_rules.py` runs it over this repository in the fast tier, so an
unscoped file fails the always-on gate rather than quietly costing every future session.

## Engineering Principles → 2. One source of truth — the sibling-builder history (six instances)

- **Two builders for one object are one builder plus an injected difference.** When two functions
  construct the **same class** and their parameter lists overlap heavily, the overlap is not a
  coincidence — it is that object's configuration surface, written twice. Extract the shared tail into
  one private builder taking the differing collaborator as an argument; each public builder then does
  only what is genuinely different (construct its strategy, pick its default). **A keyword that exists
  on one sibling and not the other is a defect the day it is added, not a property of that path.**
  The surfaces drift because nothing compares them, and **no single commit looks wrong**: each adds one
  keyword to one builder, so every after-the-fact check scoped to "your change" — including this file's
  own Post-Change Checklist and Stale-Record Check — is blind to it by construction.
  *This has now happened six times, and the first three were each fixed and written up subsystem-locally
  rather than promoted here, which is why the fourth happened a few hundred lines from the third.*
  `distributed.py` hand-built a second `ResidualAssembler` and a rename broke it with no test failing;
  every march driver wrote its own `on_step` formatter, so "a gap fixed in one persisted in the others";
  three `StepControl` classes each carried their own `next_step`, and `carry_beta` was "byte-identical in
  two controls and *absent* from the third". The fourth cost the most: **four** builders of the coupled
  march's step each grew their own globalization tail, and the k-positivity fraction-to-the-boundary
  limit — the fix for a march that went non-finite from `k < 0` in two cells of 23040 — was wired on one
  of them and reached **none** of the other three.
  **⚠️ EXTRACTING THE SHARED TAIL FIXES THE BODIES AND HIDES THE SURFACES — that is the fifth instance,
  and it is the one to internalize, because it is what a *successful* repair leaves behind.** The fourth
  was fixed exactly as this rule prescribes: the four coupled builders' duplicated tails were pulled into
  one private `_coupled_step`. Their **public surfaces stayed hand-copied**, and drifted twice more —
  the forward solve's stopping measure onto one builder of four, the shift's velocity source onto two —
  while `tools/sibling_builders.py` reported *clean*, because the builders no longer constructed a common
  class directly and so no longer looked like siblings to it. **A green report from a check that cannot
  see the case is worse than no check**: it is read as evidence. Two consequences, both binding.
  *(a)* The tool now follows delegation transitively — through a private tail, across a module boundary,
  and onto a method — and compares public surfaces above it, so this shape is visible again; but treat
  any check's silence as informative only once you have confirmed it can see the thing you are asking
  about (the same reasoning as `tools/check_hooks.sh` and its own unit test). *(b)* **When you extract a shared tail, audit the surfaces above it in the same
  change**, and keep doing so afterwards: consolidating the bodies is only half the repair, and the half
  that remains is now invisible to code review, because the duplication a reader would notice is gone.
  **The discriminator for "does this keyword belong on all of them" is whose property it is.** Both #282
  instances were parameters of the *problem* — the coupled residual's block scaling, the shift's own
  time scale — sitting on a builder that names a *strategy* (a preconditioner). If the reason for a
  parameter can be stated without naming the strategy the builder constructs, it belongs on the shared
  tail, not on that builder. Where a sibling genuinely must differ, say so *at the difference*: the
  mass-flow builder's Euclidean forward tolerance carried the reason (its bordered measure had no
  row-scaled form) at the constant itself, so the next reader saw a decision rather than an omission
  (both went in #541, when the bordered row-scaled measure was built and the builder deleted).
  **⚠️ AND A SHARED TAIL IS NOT THE ONLY REPAIR — for a shared *surface*, the durable one is a shared
  OBJECT, and it is the sixth instance (#372, 2026-09-13).** Six builders configured one shifted-march
  engine, and each listed its share of that engine's settings as its own keywords: eight apiece on the
  four coupled ones, **two** on the flow-only and scalar ones written first, so four capabilities were
  unreachable from those two paths and one was unreachable from all six. Extracting a tail would not
  have helped — the bodies were already one, `_coupled_step`; it was the *keyword lists above* that had
  drifted, which is precisely the half that survives a tail extraction. Making the settings one value
  object (`Globalization`) removes the drift by construction rather than by vigilance: there is nothing
  left to copy, and a setting added to it reaches every builder the day it is added. Reach for this
  whenever several builders' keywords are the same *configuration* rather than the same *namespace*.
  **Give such an object unset (`None`) fields, not defaults of its own:** unset falls through to each
  builder's base and then to the step class's default. The first version carried a full default set and
  two presets, so a one-field override on a coupled builder silently reset its line search to zero, and
  every default was written down a second time. ⚠️ `tools/sibling_builders.py` was silent on this family,
  and the reason first recorded here — a shared-parameter threshold — was **wrong, taken from the issue
  text without being measured**: the flow-only and coupled builders shared six parameters against a
  threshold of five, and a **same-directory** rule discarded the pair before a parameter was compared.
  Measure a check's silence before explaining it.
  This does **not** ban siblings that merely share a vocabulary. Three multigrid solvers taking
  `hierarchy, b, cycles, omega` are three different methods with different smoother families; forcing
  them behind one builder would create the union bundle the Module Review Rubric warns about, needing
  placeholder defaults for each other's parameters. The test is *same constructed class* **and** a shared
  surface that is one configuration, not one namespace.
  And check the other direction before unifying: a parameter present on only one sibling may be
  **dominated rather than missing** (`descent_backoff`/`descent_test` were exactly this — recorded as
  counterproductive and since deleted along with the `while_loop` and seeding dance they alone required.
  `grow`, the line search's rungs above the full step, was kept on the argument that inert is not the same
  as counterproductive -- and was found 2026-10-14 to have reached nothing since 2026-08-15, when a
  whole-step-limit default of `1.0` began capping every rung at the full step; it was deleted. Before
  keeping an "inert" setting, check that it reaches anything at all).
  Principle 0 says delete those, not promote them onto the shared builder.

## Engineering Principles → 3.5 No scope changes without asking — the observed failure

This is not about ceremony; it is about the *evidence* that follows. The observed failure: the user
agreed a residual measure whose scales are rebuilt every outer iteration and held fixed only across a
line search. What got built instead froze the scales once at the initial condition — described as
"Phase 1" rather than flagged as a deviation. The measurement taken under that shortcut then showed the
measure failing, and that failure was reported as if it were a property of the *formula*. It was a
property of the unapproved substitution. The user had to catch it.

## Engineering Principles → 3.6 Layering — the observed failure

The observed failure: the staged solve driver (segments, refresh, convergence check), the preconditioner
sessions, the shifted-step assembly, the residual measures and the Krylov-regime settings were all
written for the k–ω SST solve, **inside `turbulence/coupled.py`**, because turbulence was the first
thing that needed a robust march. None of them mentions turbulence. The consequence was not tidiness: a
**laminar** flow problem had only a bare pseudo-transient step, so it could not be marched by the
dual-time / retry / row-scaled-norm / refresh machinery the turbulent case runs on, and a control
experiment ("does this solver behaviour need turbulence?") had to be run on a weaker driver that did
not isolate the variable (#448). The problem had been filed as #277 weeks earlier; the file was ~4000
lines by the time anything moved out of it, and nothing measured that it was growing.

The size guard first covered only the four physics packages, and the generic ones grew unmeasured the
same way: on 2026-10-17 `schemes/gradient.py` stood at 3637 lines and `precondition/multigrid.py` at
3109 with no budget. The guard was extended that day to every module in `aquaflux/`, nested
subpackages included (the old scan read only each package's top level, so `io/cad/` was invisible to
it even had `io` been listed).

## Engineering Principles → 3.7 Vocabulary — the observed failure (#373)

The observed failure (issue #373): the package grew several names per concept and several concepts
per name, one reasonable-looking change at a time. `relaxation` named an under-relaxation factor in
`(0, 1]`, which damps *less* as it grows, and the shift strength `β`, which damps *more*; `rtol` was a
residual tolerance on one solve and a tolerance on the *state increment* on another, with different
defaults; `omega` was the SST field and a smoother weight, and one case file set both a few lines
apart; `cycles_limit` counted redone steps, not cycles. This is not cosmetic. A settings field is read
straight from a case file, so **its name is a user-facing key**, and a parameter name is how a reader
tells what a recorded setting meant. No single change looked wrong, because nothing compared a new
name with the places the word was already in use.

## Technology Stack — Numba and filelock rows (full text)

| Compiled host loops | **Numba** | a loop whose value is in the work it SKIPS (a ray walked to its first hit) cannot be traced — a traced loop pays its worst case on every item — and written as numpy passes it pays per step for every item still in flight. Its first consumer is the triangle grid's walk (`aquaflux/radiation/grid_walk.py`), 17-37x the numpy walk it replaced, with identical answers; the back-face tile tests (`back_faces.py`) and the transfer build's per-block front test (`transfer._any_in_front`, which exits at the first quadrature point that keeps a facet) are the same shape. The occupancy grid's box certificate (`grid._unions_held`) uses it for the other reason a host loop can win — fusing several whole-array passes into one threaded loop that forms no intermediate — and was ~15x faster in place, identical answers (`.claude/rules/radiation.md`; the measurement is in `.claude/notes/radiation-log.md`, "SHAFT CULLING"). A core dependency since 2026-09-26, by the project owner's decision over keeping it optional. |
| Browser interface (optional) | **PyVista/VTK + trame + plotly** (`aquaflux_ui`, `ui` extra) | a local browser interface (case setup, running and following a run, and a results viewer) that needs no JavaScript toolchain: VTK renders on this machine and trame streams images, so a large mesh never reaches the browser. A separate top-level package, because `import aquaflux` enables x64 and configures JAX — the interface imports neither aquaflux nor JAX, and asks the solver through a separate `aquaflux serve` process (`.claude/rules/ui.md`) |
| Case file | **PyYAML** (`aquaflux/case/`) | a case's mesh, physics, boundaries, (for a flow) fluid and numerics, and solver in one YAML document, read by the YAML 1.2 rules for plain values (PyYAML's own 1.1 rules read `1e-5` as a string and `no` as a boolean) and validated per position by `solve.SettingsMapping` — so no pydantic. The equation DSL (YAML → AST emitting terms) is a different thing and is still the **last** layer, not built |
| Bounded compilation cache | **filelock** (`aquaflux/__init__.py`) | `import aquaflux` points JAX's persistent on-disk compilation cache at `~/.cache/aquaflux/jax` and **bounds it**. JAX's default `jax_compilation_cache_max_size` is `-1`, which its LRU implementation reads as *no eviction* — an unbounded cache never prunes and only grows (one checkout reached 88 GiB across 2128 entries, 95% of it more than a week old, 42 of them ~1 GiB coupled-solve programs). Setting a byte bound turns real eviction on, and JAX takes an inter-process lock through `filelock` to do it. **⚠️ A BOUND WITHOUT `filelock` DISABLES THE CACHE ENTIRELY** — every read and write fails with a `UserWarning` and it stores nothing, which is worse than no bound, so the package checks and degrades to merely unbounded rather than silently dead. Override the size with `AQUAFLUX_COMPILATION_CACHE_MAX_GIB` (negative for no bound), the location with `AQUAFLUX_COMPILATION_CACHE_DIR`, or switch it off with `AQUAFLUX_DISABLE_COMPILATION_CACHE=1`. |

## Architecture — why `dot` is not spelled as a reduction (measured 2026-08-21)

**⚠️ `dot` IS DELIBERATELY NOT SPELLED `jnp.sum(a * b, axis=-1)` — do not "simplify" it back
(measured 2026-08-21).** It multiplies the operands and then sums the components **explicitly**.
The two spell the same contraction, but on the CPU backend a fusion rooted at a *reduction* is
emitted as a `kCustom __ynn_fusion` kernel with an empty `outer_dimension_partitions` — i.e. it
runs on **one thread** — while the `kLoop` fusions around it are split 2–4 ways. In one
corrected-gradient reconstruction 25 kernels were thread-partitioned and exactly 3 were not: the
three `dot`s, at **40 % of kernel time**. Configuration for every number here: jax/jaxlib 0.10.2,
CPU backend, macOS arm64, 11 cores, x64 on, compiled ILU(0) live. On the primitive the compiled
`__ynn_fusion` count goes **1 → 0** and the contraction runs **2.9×** faster at `n = 24730,
dim = 2` and **2.2×** at `n = 400000, dim = 3`. On the pitzDaily coupled RANS residual (12225
cells, evaluated at the time-accurate OpenFOAM field mapped cell for cell onto that mesh) the
residual evaluation is worth **1.6–1.8×** and its `jvp` **1.2–1.5×** — five runs spanning two
bases (before and after the gradient-scheme work of #297), including the independent measurement
that first found this, with `|R|` agreeing to **6.5e-15** relative every time. **Read the residual
figure as the solid one and the `jvp` as merely indicative**: the former lands in its band on every
run, while the latter's spread is wider than this instrument resolves — these are wall-clock on a
shared desktop, where a per-application timing is already on record as carrying ~15 % spread, and
the slowest run had both arms ~30 % up on the others. The contention-immune evidence is the kernel
structure, not the seconds. `norm_squared` delegates to `dot` and inherits it.
- **Multiply first, unroll only the sum.** `sum(a[..., i] * b[..., i] ...)` off the *unbroadcast*
  operands breaks broadcasting: an operand with a trailing axis of 1 against the other's `dim` —
  which the reduction handles silently — runs off the end of that axis. Indexing the already
  broadcast product preserves it. Pinned by a unit test, as is the empty-`dim` case.
- **⚠️ Form the product with `jnp.multiply`, not `*`.** Two NumPy operands multiplied with `*` give
  a NumPy array, so the unrolled sum hands back an `ndarray` where the reduction always returned a
  JAX array — silently dropping `.at[]` for callers that pass concrete arrays. This is the one way
  the rewrite can change behaviour rather than just rounding, and no existing test caught it; there
  is one now.
- **⚠️ IT IS NOT BIT-IDENTICAL to the reduction.** The explicit sum lets the compiler contract a
  different pair of the multiply-adds, so results move by a rounding of the summed terms (~2–6e-16
  of `Σ|aᵢbᵢ|`, which is an unbounded *relative* move on a near-cancelling dot product — judge such
  a difference against the magnitudes summed, never against the answer). Two categories, not to be
  conflated when a test goes red: one comparing two paths that **both** route through `dot` still
  holds exactly, and a break there means the change is wrong; only a value pinned to what the
  *reduction* computed is legitimately re-pinned. In the event the fast, slow and validation tiers
  all passed unchanged — no test anywhere was pinned to the reduction's exact bits.
- **This is what one home buys.** `dot` is imported by 14 modules — geometry, schemes, diffusion,
  advection, Rhie–Chow, momentum, the SST closure and the sources — so a one-line change reached
  every residual evaluation at once. Open-coded at each call site it would have reached none of
  them.

## Testing Architecture → A test that cannot fail — the collapse.py mutation pass (2026-09-14)

**Measured, 2026-09-14** (the vectorized `aquaflux/mesh/collapse.py` rewrite and its test suite, #396):
a 16-mutation pass — introduce one targeted single-line bug, run the suite, confirm it goes red, revert,
repeat — found that **6 of the 16 mutations passed the existing test suite (15 tests at the time)
completely undetected.** Three were genuine coverage gaps, fixed with new tests: disabling the check
that a capping face is planar and normal to a single axis; a bug that could confuse one face's node
count for another's (invisible because every fixture happened to give every face in a subset the *same*
node count, so nothing distinguished a row belonging to face *i* from one belonging to face *j*); and
disabling the low side of a "must reduce to exactly 2 distinct nodes" check (every existing fixture that
reached that branch overshot to *more* than 2, never undershot to fewer). None of the three was a subtle
defect requiring imagination to construct — each was a one-line change that silently produced a wrong
mesh, on a module whose tests were read, before this check, as good coverage. **The other three were
reviewed and confirmed to be non-issues, not gaps** — one mutated an inequality boundary (`<=` vs `<`)
that no realistic input lands on exactly, one hardcoded a value a caller's precondition already makes
the only reachable one, one re-passed a value through explicitly that a sibling function treats as
provably identical to leaving it implicit — and no test was added for any of the three. **Both halves of
this matter equally: chasing every undetected mutation as if it were a bug is the same failure mode as
never checking at all, just spent in the other direction.** Judge each on whether the code path is
actually reachable and whether the two behaviours are actually distinguishable, and record which
mutations were dismissed and why, not only the ones that were fixed.

## Testing Architecture → A test that cannot fail — tautological checks in measurement scripts (2026-09-24)

**⚠️ THE SAME DEFECT IN A MEASUREMENT SCRIPT HAS NO RUNNER TO NOTICE IT, AND THAT MAKES IT WORSE.**
Everything above is about the test suite, where at least a green count is a claim someone might
interrogate. A **printed** check in a harness is read once, believed, and quoted afterwards. Measured
2026-09-24, while separating two variables in the radiation grid's cost: a 2x2 of measurements was
printed with a closure check reading `closure: 5.01 = 5.01 (must agree)`, and it was read as
corroboration that the four corners were consistent. **Both paths through a 2x2 are the same quotient
with the middle corner cancelling** — `(B/A)(D/B)` and `(D/C)(C/A)` are each `D/A` — so it agrees for
*any* four numbers, including four wrong ones. There was no second path to find and the check had never
been able to fail.

- **The phrasing is what did the damage, not the arithmetic.** The line carried a `must agree` label, a
  pair of numbers and a tick. **A tautology dressed as an assertion is worse than the bare quantity**,
  because the phrasing is exactly what stops a reader asking what it could ever have shown.
- **Concrete trigger:** *before writing `must`, `expected`, `should equal` or a tick beside a computed
  pair — in a script, a log line, or a commit message — ask what inputs would make it disagree. If none
  would, it is a derived quantity and must be labelled as one.*
- **The general form, which cost more than the check did:** a number that matches what you already
  believe is the least reliable kind of agreement, and the instinct it produces is to stop. Decompose it
  first. In the same session a gap of exactly 1.37x was read as confirming a known 1.37x machine spread,
  and was in fact **1.12x of receiver composition times 1.22x of everything else** — two effects whose
  product looked like one clean corroboration.

## Testing Architecture → CI is not a superset of a local run

**⚠️ CI IS NOT A SUPERSET OF A LOCAL RUN, AND `importorskip` IS WHY.** CI installs `.[test]`, so any
module guarded by `pytest.importorskip` on a dependency outside that extra is **skipped there and runs
only on a machine that has it**. A green CI tier then says nothing about it, and its failures surface only
locally, which reads as "broken on my machine" when it is the opposite: local is the only place it is
checked at all. (This happened: PETSc-gated modules, since removed with the PETSc dependency on
2026-10-10, failed on `main` for four days with CI green throughout.) This is the same shape as a check that has stopped seeing
anything (`tools/check_hooks.sh`, `tools/sibling_builders.py`) — a skip and a pass are indistinguishable
from the exit status. **When a test fails locally and passes in CI, check the skip counts before assuming
the difference is your machine**: 2026-08-21 the CI slow tier reported `4 skipped` per shard against zero
locally, and three tests had been failing on `main` for four days with CI green throughout.

## Testing Architecture → fastgate.sh, the pipe rule, and fast-tier parallelism

`fastgate.sh` wraps `pytest -q -m <tier>`, writing the run to a file and reporting pytest's **own**
exit status with the summary line found by pattern. Invoke `pytest` directly if you need something the
wrapper does not pass through, but never through a pipe: this suite prints library shutdown chatter
*after* the summary, so `| tail -n` shows the chatter, hides the result, and returns `tail`'s exit
status — which is `0` however the run went. Its own behaviour — that a failing run exits non-zero,
that the summary survives the chatter, that the skip count is reported, that a mistyped tier is
refused, that the fast tier parallelizes while the heavy tiers do not, and that it refuses to start
beside a running validation case (with `FASTGATE_FORCE` past it, no wedge from a stale run-file, and
no guard under `CI`) — is pinned by `tests/unit/test_fastgate.py`, for the same reason
`check_hooks.sh` and `sibling_builders.py` are: a runner that had stopped propagating a failure looks
exactly like a passing suite, and every other check in this project is read through it.

⚠️ **THE PIPE RULE APPLIES TO `fastgate.sh` ITSELF, NOT ONLY TO BARE `pytest` — and knowing the rule is
not what protects you.** `tools/fastgate.sh … | tail -4` has the identical defect the paragraph above
describes: the pipeline's status is `tail`'s `0` whatever pytest did, *and* the `-n` truncates away the
`FAILED (pytest exit N)` block the script prints, so what survives is the final `full log:` line — which
reads exactly like a pass. Observed 2026-09-10, in a session that had quoted this very rule an hour
earlier: a run with **21 failures** was read as green, and the next measurement was launched against
broken code before a log grep caught it. **Redirect (`> file`) and read the script's own exit status;
never pipe it, and never judge it by its last few lines.** The failure is silent in both directions at
once — the status lies and the evidence is cropped — which is why the wrapper exists and why wrapping
the wrapper undoes it.
⚠️ Note the cost of the escape hatch in the sentence above: **invoking `pytest` directly skips the
case guard as well as everything else the wrapper does**, so it is the one route that can still put a
tier on top of a march without saying so.

**The fast tier runs across worker processes; the `slow` and `validation` tiers do not.** That split
is a memory decision, not a preference: the heavy tiers' solves each hold gigabytes of live JAX
buffers (a materialized 3D coupled Jacobian is ~2 GB per copy), so running several at once drives
this machine into swap and suspends every application on it — CI reaches the same conclusion from the
other side, sharding those tiers across jobs at `-n 1` rather than within one. Two details of the
fast tier's parallelism are load-bearing. It distributes by **file** (`--dist loadfile`), so a
module-scoped fixture is built once instead of once per worker and each file keeps its recorded test
order; and it pins **one BLAS/XLA (and Numba) thread per worker**, because otherwise every worker grabs every
core and N workers × M cores thrashes instead of scaling. `FASTGATE_JOBS=<n>` sets the worker count
and `FASTGATE_JOBS=0` (or your own `-n`) opts out — do that when bisecting a failure, since worker
output is interleaved. Measured 2026-08-23 on an 11-core, 19 GB machine with the compiled ILU(0)
kernel live: **26:13 serial → 8:53**, and **6:34** once the handful of heaviest
tests were cut back — see the cost rule immediately below. ⚠️ **Its accompanying memory figure,
"~6.4 GB peak", was RSS and is deleted rather than corrected in place** (Stale-Record Check): RSS
excludes compressed pages, and a footprint re-measurement on the same class of machine
(`top -l 1 -o mem -stats pid,ppid,mem,cmprs,command`, 2026-09-13, 11 xdist workers, jax 0.10.2)
found the workers holding **2.5–5.9 GB each, ~49.5 GB combined**, against a summed RSS at the same
instant of **4.45 GB** — an order of magnitude apart. Judge how many of these a machine can run
from the footprint number, never from RSS. `tools/fastgate.sh` now enforces "one tier at a time"
machine-wide because of exactly this gap; see the tier lock in the Development-workflow section
below.

## Testing Architecture → What a fast-tier test may spend — unit-job swapping and the distributed job

**⚠️ A UNIT JOB THAT SLOWS DOWN OR TIMES OUT IS USUALLY THE RUNNER SWAPPING, NOT YOUR CHANGE — and
the fix is a shard, not a worker.** A fast-tier worker is not cheap: one module alone
(`test_radiation_images.py`) peaks at 4.9 GB, and a worker's footprint grows as it draws module after
module. Run unsharded at `-n 3` on a 16 GB hosted runner, the tier overflowed RAM into swap, every test
slowed 10–35x, and tests crossed the 900 s per-test timeout with no assertion failing anywhere — read
for a week as "runner variance". The unit job's memory step (a `vmstat` line every 30 s, printed after
the tests) is what told the two apart: free RAM at ~200 MB within ten minutes, swap climbing. So the
tier now runs as **three shards of two workers** (`--splits 3 --group N -n 2`, balanced by the unit
durations the weekly refresh records in `.test_durations` -- record them on CI, never locally: local
durations mispredicted CI by up to 2x per test and balanced nothing), measured 2026-10-08 on
a 4-core / 16 GB container without swap, jax 0.10.2: the unsharded `-n 3` run peaked 14.1 GB and was
OOM-killed, `-n 3 --dist loadfile` 14.3 GB with all three workers killed (`loadfile` does not help
here), and each shard 9.2–10.9 GB in 12–17 min with none killed. When it outgrows that, **add a shard;
never add a worker.** Before blaming a slow unit job on your branch, read its memory step, check
whether unrelated branches are slow too, and compare the other interpreter on your own run. One more
measured fact to save the re-derivation: **preserving the JAX persistent compilation cache across CI
runs buys nothing** — the distributed `shard_map` tests (see below) compile 7184 XLA programs
whose largest takes 0.05s, so none clears the 2.0s persistence floor and the cache stays empty. Their
cost is eager per-op tracing and dispatch in mesh/partition setup, which is where a real saving would
have to come from. (`.github/workflows/ci.yml` carries both measurements with their configuration.)

**Those `shard_map` tests now run in their own CI job, selected by the `distributed` marker.** Mixed
into the unit job under `-n auto` they oversubscribed the runner — each xdist worker forks a child
that wants the whole machine — and the contention, not the work, set the tier's wall clock: one test
costing ~241 s when healthy ran past the **900 s per-test timeout**, which `--timeout-method=thread`
cannot interrupt, so it killed the worker and reported `node down: Not properly terminated` with no
mention of a timeout anywhere. **That failure reads as a crash in the test and is not one** — check
what else was running before believing it. Membership is by marker rather than by path, and
`tests/unit/test_distributed_marker.py` fails if a module that spawns simulated devices lacks it, so
a new one cannot quietly rejoin the contended job. The local fast gate still runs them inline; this
split is a property of a 4-core runner, not of the tests.

## Testing Architecture → Conditional coverage — PETSc

Two consequences worth having in mind rather than rediscovering:

- **A dependency kept out of the `test` extra is a cost decision, and the census is where its price
  is written down.** Today that is only `scotchpy` (one partitioner test). PETSc (`petsc4py`, no wheels,
  built from source) was the large case — three modules, two in the fast tier, that had never run in CI
  — until the dependency was removed on 2026-10-10.

## Testing Architecture → Canonical tests — coupled-solve adjoint guard (#370)

  ⚠️ **And such a test has to make the two paths genuinely differ, or it compares a configuration
  against itself.** Raising a `max_steps` **cap** a converged solve never reaches changes nothing:
  the march is bit-identical, so the gradients agree for a reason that has nothing to do with the
  adjoint. Vary something that moves the path — `inner_steps` in
  `test_dual_time_gradient_is_iteration_count_independent`, the shift strength `beta0` in
  `test_the_coupled_adjoint_is_independent_of_the_forward_iteration_count` — and **assert the step
  counts differ**, measured in separate runs with an observer (which changes nothing about the march).
  ⚠️ **A lever that moves the path need not move the step count, and the guard is what notices.** The
  coupled test used `inner_steps` too until #370 made its row-scaled measure rebuild every outer
  iteration: after that, inner steps of 1, 2, 3, 4 and 5 all took **20** outer steps on that channel
  while their inner-iteration counts differed, and the guard refused to run. `beta0` 2.0 against 0.5
  takes 20 against **9** (measured 2026-09-16 on the coupled RANS channel, block-diagonal
  `ScalarTwoLevel` + `ConvectionTwoLevel`, default `Convergence`).

## Development workflow → Use the blessed command — run_case.sh, the case guard, the tier lock, clear_caches

`run_case.sh` puts in one place everything the surrounding rules used to ask you to remember:
unbuffered output **redirected, never piped**, to a timestamped log; the machine held awake; a
free-memory and load pre-flight; a refusal to start a second case; and a run-file recording pid,
worktree, branch, commit and case settings; and, under `--wait`, the case's **own** exit status
(written beside the log by a launch wrapper — before 2026-09-22 both `--wait` forms exited `0` for a
crashed case). Its most valuable output is not the log — it is that
**"is this run mine, and what is it testing?" has a written answer**, a question that has been got
wrong from the process table alone.

**⚠️ That mutual exclusion was over *cases*, and a TEST TIER IS NOT A CASE — so `tools/fastgate.sh` now
refuses to start beside a running case too (exit 3; `FASTGATE_FORCE=1` to override, skipped under `CI`).**
On 2026-09-09 a gate landed on a running march three times in one evening, between three sessions that
all knew the one-heavy-job-at-a-time rule: it was *enforced* for case-vs-case and merely *known* for
tier-vs-case, and knowing it turned out not to be what mattered. **The cost is symmetric, which is the
part worth internalizing — this is not a trade of one job's latency for another's throughput.** In the
measured collision the fast tier alone took **17:25** against its usual 6:34–8:53 while the case it
landed on ran **3.2×** slow over the overlap; run end to end they are about 8 and 9 minutes. Both jobs
lost. The gate asks `validation/run_case.sh --running` rather than reading the run-file itself, so the
file's format and the `kill -0` liveness rule keep one home — a second copy of either is the duplication
that made this class of collision possible to begin with.

**⚠️ That guard was over tier-vs-case, and the same gap existed one level over: TIER-VS-TIER.** On
2026-09-13 two fast tiers started five minutes apart, from two different worktrees, neither touching a
validation case, and the machine (11 cores, 19 GB) had to be hard-reset. A single fast tier's real
memory footprint — its mostly-**compressed** working set, not its RSS, which undercounts it by roughly
an order of magnitude here (see the correction above) — runs to ~50 GB across its 11 workers; two at
once is ~100 GB of that on a machine with 19 GB of RAM and no reserve of swap left to absorb it.
`tools/fastgate.sh` now also holds a machine-wide lock (a lockfile under `~/.cache/aquaflux/`,
released on exit) while any tier runs, refusing a second one from **any** worktree or session with the
same `FASTGATE_FORCE=1` override and the same CI exemption — exit `4`, distinct from the case guard's
exit `3`, so the two refusals are distinguishable from a calling script. `FASTGATE_LOCK_DIR` overrides
the lock's location, the way `FASTGATE_JOBS` overrides the worker count.

**A second, complementary fix cuts a single tier's own footprint, rather than only fencing it off
from a second one.** An xdist worker keeps every compiled XLA executable it has ever built for the
process's lifetime, across however many modules `--dist loadfile` hands it — so the footprint above
only grows over a run and never comes back down. `tests/conftest.py` now calls `jax.clear_caches()`
in a module-scoped autouse fixture, at the end of every module. Measured the same way as the
correction above (`top -l 1 -o mem -stats pid,ppid,mem,cmprs`, same 11-core/19 GB machine, jax
0.10.2, 2026-09-14): peak combined footprint **54.3 GB → 22.7 GB**, wall clock **467 s → 412 s** —
both improve, because the unfixed run was also paying to keep over 30 GB of that footprint
compressed, and the recompiles this costs are cheaper than that. This is most of the reason the
default worker count was **not** also cut to fit a single tier inside physical RAM (a harsher,
slower fix considered and deferred to a follow-up issue): with this fix in place, one tier's peak
footprint sits close to this machine's 19 GB of RAM rather than 2.8× over it.

## Development workflow → Record what a measurement was taken under — the three unfalsifiable findings

This is not hypothetical bookkeeping. Three separate findings in `.claude/rules/solve.md` — "ω is the
unsmoothed field (~700–1300×)", the Vanka "a strong smoother still stalls, so the coarse space is the
wall", and a monolithic-AMG probe's "block-ILU(0) smoother diverges" — record no smoother and no
aggregation. Both of those defaults have since moved (ILU(1) → ILU(0), smoothed → plain aggregation),
and **each move has already inverted a conclusion on that case.** So all three are now unusable: they
cannot be relied on, and they cannot be cheaply re-adjudicated either, because the harnesses were
scratchpad-only and are gone. They are not wrong — they are unfalsifiable, which is worse, because a
wrong finding gets corrected and an unfalsifiable one gets cited.

**Two of the three have since been re-adjudicated (2026-08-08), and how they came out is instructive.**
The **Vanka** one was re-measured under the current bundle and does not survive: the smoother does not
"still stall" against a working coarse space — it stagnates on its own, at a state where the shipped
incomplete-LU converges in two cycles, so the coarse-space inference it carried has no support. The **ω**
one was corroborated *by a different measurement entirely* — the near-null direction of the operator's
worst per-cell blocks is pure ω — so the **field** is now on two independent legs while its **number**
(~700–1300×) remains unverified. Note what made re-adjudication possible in both cases and impossible
before: a harness kept in the repository rather than in a scratch directory. Keeping the probe is the
difference between a finding that can be re-asked and one that can only be cited.

## Development workflow → Documentation — Sphinx details (full text)

User-facing docs live in `docs/` as a **Sphinx site written in MyST Markdown**, mirroring
aquakin: `pydata-sphinx-theme`, autodoc + napoleon (NumPy docstrings), and an API page
(`docs/api.md`) **generated at build time** from each documented subpackage's `__all__`
(`conf.py` `_write_api_page` / `PUBLIC_SUBPACKAGES`) — so it never drifts from the public
surface. Read the Docs builds it (`.readthedocs.yaml`, `fail_on_warning: true`), so every
cross-reference must resolve and every page must sit in a toctree. **Build locally with
`tools/build_docs.sh`** (see the "Use the blessed command" table above) — it runs the same
`sphinx-build -b html -W docs docs/_build/html` Read the Docs and CI run, deleting the gitignored
`docs/generated`/`docs/api.md`/`docs/_build` first so a stale run cannot pass by describing the
previous set of pages. It exists because a plain `pip install -e ".[docs]"` cannot be relied on here:
the docs toolchain is not a runtime dependency, and a PEP-668-managed system Python refuses to
install it. It keeps one cached build environment (`~/.cache/aquaflux/build-venv`, created with
`--system-site-packages`) rather than pip-installing aquaflux itself; where a
normal editable install is available, `pip install -e ".[docs]"` then `cd docs && make html` works
too. **CI builds the docs on every PR** — the `docs` job runs the same `-W` build Read the Docs runs,
and the required `fast gate` depends on it, so a broken cross-reference or a page missing from a
toctree blocks the merge instead of turning the published site red afterwards. Run
`tools/build_docs.sh` as well when you touch a docstring of a documented subpackage, a `docs/` page,
or a cross-reference — the CI job is the backstop, not the first line.

**Publishing a subpackage means publishing its whole `__all__`.** Adding a name to
`PUBLIC_SUBPACKAGES` puts every one of that subpackage's exports on the site, so the export list
*is* the editorial decision — deliberately, because a second hand-maintained "what to document"
list in `conf.py` would drift from `__all__`, which is exactly what generating the page removes.
If a name should not be on the site, take it out of `__all__`. How those exports are *grouped* on
the page is a separate table, `SUBPACKAGE_GROUPS`, keyed on the module each name is **defined** in
rather than on the names themselves — so it survives names being added, and it cannot hide one: an
unlisted module still gets a group headed by its own name. Extend it when a subpackage grows a
module, and note that `-W` cannot tell you when you forgot — the page stays complete, just less
tidy. Three things bite when extending the published list; the first two cost a build each to
rediscover, the third silently reports the wrong answer:

- **A multi-name parameter entry must sit on ONE line.** NumPy style allows
  `beta0, exponent, beta_floor` followed by an indented description, but the *name* line cannot
  wrap: docutils reads the continuation as another field and the description as a block quote,
  giving the `Field list ends without a blank line` / `Unexpected indentation` /
  `Block quote ends without a blank line` triple, all attributed to the docstring rather than to
  the wrap. The same triple comes from a prose paragraph left *after* the last entry inside a
  `Parameters` section — every one of its lines is read as another field name. Such prose belongs
  above `Parameters`, or in its own section.
- **A field whose default is a plain function is documented by that function's docstring.**
  autodoc omits undocumented attributes, which is what keeps the solvers' configuration fields off
  the site — but an `equinox` field like `residual_norm = field(default=jnp.linalg.norm)` leaves
  the *function* as the class attribute, so autodoc finds its docstring and publishes third-party
  prose that need not even be valid reStructuredText. `autodoc_inherit_docstrings` does not reach
  this (autodoc already forces it off for attributes). `conf.py`'s `_skip_borrowed_member_doc`
  (`autodoc-skip-member`) drops any class member that is a routine defined outside `aquaflux`,
  restoring the skip-if-undocumented rule; each such field stays documented by its class's
  `Attributes` section.
- **`docs/generated/`, `docs/api.md`, and `docs/_build/` are gitignored build artifacts — delete
  all three between probe builds.** Stubs written for a previous subpackage set are rebuilt from
  the leftover files, so a run that looks clean can be clean only about the wrong set of pages,
  and the warning count means nothing. `rm -rf docs/generated docs/api.md docs/_build` first,
  every time.

**A Sphinx extension enabled in `conf.py` must be matched by a dependency in the `docs`
extra.** MyST's `linkify` needs `linkify-it-py` (hence `myst-parser[linkify]`), and its
absence does not degrade gracefully: the parser raises on the first inline token of the first
page, so *every* page fails, not just pages containing a bare URL. This shipped once — the
site's first Read the Docs build died on it, because at the time nothing but Read the Docs
ever built the docs. When you enable an extension, install the extra in a **clean**
environment and build; an environment that has accumulated packages will not reproduce what
Read the Docs does.

## Stale-Record Check — incidents

**This is not hypothetical bookkeeping. It is the most expensive recurring defect in this project.**
In one session, three separate wrong facts were lifted out of these files by grep and asserted as
current — a march solver that had been replaced, a tolerance that had moved, and a preconditioning
side that had been deliberately reversed. Two of them were passed into sub-agent briefs, which would
have produced confidently wrong measurements. A later audit found a **binding decision** stating
`NewtonSolver` was deleted and, 34 lines below it, another entry using `NewtonSolver` as live.

## Stale-Record Check — forwarding wrappers and the identifier filter

**⚠️ A FORWARDING WRAPPER HIDES CALL SITES FROM AN AUDIT SCOPED TO THE CALLEE.** When you change what a
function accepts, the callers you must check are not only the ones that name it: a wrapper taking
`**kwargs` and forwarding them passes a caller's keywords through under its *own* name, so a scan for
`the_function(` — grep or AST alike — reports clean on a call that breaks. Scan the wrappers too. (This
has happened: an AST pass over every `solve_coupled(` site cleared a contract change, and the break was
in a `solve_reynolds_continuation(...)` call whose keywords reached `solve_coupled` through
`**solve_kwargs`. It surfaced as a failing adjoint test in a tier that only runs on merge.)

The `_|[a-z][A-Z]` filter keeps only words shaped like code (an underscore or an internal capital).
Without it a prose-heavy diff reports ~70 ordinary English words and the output is too long to read —
which is not hypothetical: a real orphan (`retry_on_cycles`) survived in `.claude/rules/solve.md`
behind exactly that noise. **The cost is a blind spot: a renamed all-lowercase single word.** That is
what the "grep for the CLAIM, not only for the symbol" rule above is for — this identifier grep is
the cheap half of the check, never the whole of it.

## Post-Change Checklist → 2. Sibling-builder check — the tool's blind-spot history

     It is a **report, not a gate** (whether a pair is one builder or two genuinely different methods is
     a judgement) and it always exits `0`, so read it. **It reports pairs on a healthy tree — that is
     correct, and "zero pairs" is not the target.** Each `only here:` list is a capability one sibling
     has and the other does not; every entry must be a genuine property of that path, or it is drift.
     Ask of each one *whose property is this*: if you can justify it without naming the strategy that
     builder constructs, it is the shared tail's and belongs on every sibling.
     It is package-wide rather than diff-scoped — that is why it lives here and not in the Stale-Record
     Check, which reads `git diff` and cannot see a parameter that was only ever added.
     ⚠️ **It follows delegation, so a shared tail no longer hides the surfaces above it** — it
     did not until 2026-08-20, and in that window it reported `no sibling-builder pairs` while two
     parameters drifted across four builders that all route through one private step. Its own coverage
     is pinned by `tests/unit/test_sibling_builders.py`, for the same reason `tools/check_hooks.sh` is:
     a check that has stopped seeing anything looks exactly like a clean tree.
     ⚠️ **Its second blind spot was `@classmethod` factories, closed 2026-08-21.** It recognized only
     `build`/`create`/`make`/`from_*` by name, and credited construction to *capitalized* callees — so
     a factory writing `return cls(...)` looked like it built nothing and never entered the report at
     all. That is worse than the first blind spot: not a quiet pair but **no pair**, indistinguishable
     from a clean tree. It now also knows `calibrated` and credits `cls(...)` to its owning class.
     ⚠️ **The NAME half of that was still a blind spot, and it is closed structurally (2026-08-25, #285):
     a `@classmethod` whose body returns `cls(...)` is a factory whatever it is called.** Teaching the
     list one more name each time leaves the next naming style just as invisible — three new factories
     (`FieldLayout.cell_fields`, `FieldGroups.by_counts`, `FieldGroups.split_before`) would have been
     unseen on the day they were written. The shape test is now primary and `_FACTORY_METHODS` is the
     fallback for factories whose construction the syntax tree cannot follow; pinned by
     `test_it_reaches_a_classmethod_factory_no_naming_convention_covers`, whose fixture the previous
     tool is blind to. **The standing lesson survives the fix: a pair this tool cannot see reports as a
     clean tree, so check that it names both sides of your pair before reading its silence as a clean
     report.**
     ⚠️ **Two more blind spots were closed 2026-09-13 (#372), and both hid one family.** *(i)* Resolution
     stopped at private functions defined in the file being read, so a shared tail extracted into
     **another module**, or onto a **method** of a configuration object (`Globalization.step()`), credited
     its callers with building nothing and they dropped out of the report — not a quiet pair but no pair.
     It now resolves a callee against the module's own functions first and then against every name
     defined **exactly once in the package**, dropping duplicated names rather than unioning them; pinned
     by `test_it_sees_through_a_tail_that_is_a_METHOD_IN_ANOTHER_MODULE`. *(ii)* Pairs were compared only
     within **one directory**, so the flow-only builder and the four coupled builders of one march — six
     shared parameters, threshold five — were thrown away as namesakes. Without the rule a pair needs a shared
     constructed class and a shared surface, and a public builder is not paired with a builder it calls —
     pinned by
     `test_it_pairs_siblings_that_live_in_different_subpackages` and
     `test_it_does_not_pair_a_public_builder_with_the_builder_it_delegates_to`. The same resolution change
     made `solve_reynolds_continuation` / `solve_reynolds_ramp` visible (both `return solve_coupled(...)`);
     they share two parameters and are correctly not paired.
     ⚠️ **Dropping the rule surfaced three assembler pairs, reviewed and left in the report:**
     `ResidualAssembler.build`, `MomentumContinuity.build` and `ScalarTransport.build`, whose only "common
     class" is the `BoundaryConditions` each gets from `boundary.resolve(...)` — an incidental helper, not
     the product. They are three different assemblers. `ScalarTransport.build` in fact *returns*
     `ResidualAssembler.build(...)`, a delegation the wrapper rule cannot see because `build` is defined 24
     times in the package and a call on a receiver whose class the tool cannot recover is never followed
     under an ambiguous name. So "shared constructed class" is weaker evidence than it reads, and a wrapper
     reached through a common method name still pairs.
     ⚠️ **A THIRD blind spot, closed 2026-09-14 (#392), and it hid the coupled family again the day #371
     unified it.** `coupled_step` opens a preconditioner session and returns `session._build(...)`;
     `_build` is defined on four classes, so the call was dropped and `coupled_step` was credited with
     building nothing — the whole coupled family left the report, which read as a clean tree. The tool now
     resolves a method through its **receiver** where the receiver's class is knowable: a local bound by
     `x = f(...)` resolves `x.m` on the classes `f` *returns as its value*, `self.m` on the owning class
     (falling back to the unique bare name for an inherited method), and a local bound to a call and
     returned *as it stands* — the whole return value, an arm of a conditional, an operand of `and`/`or` —
     credits that call. Methods are keyed `Class.method`, so one class's `_build` is never another's.
     **Unioning every definition of the name was rejected** — it would credit `coupled_step` with what
     `MonolithicVCycleInverse._build` constructs — and a receiver whose producer returns no package class **falls back to
     the bare name**, because the first version lost `_coupled_step`'s `globalization.step(...)` (a rebound
     parameter) and silently re-hid the pair it was written to find. The one new pair is `coupled_step` /
     `mass_flow_coupled_continuation`; every other pair is main's, unchanged. Its two carve-outs have
     since both gone — `residual_norm` with the measure's move onto the solve (#370), `flow_direction`
     with the drive's (#375, 2026-09-23) — so the pair now reports **13 shared and `only here: []` on
     both sides**: two identical surfaces over one tail, which is what this rule is for. (#541 then
     deleted the mass-flow builder outright — a held bulk velocity is a property of the problem, which
     `coupled_step` borders itself — so `coupled_step` now pairs with `flow_march_step`.)
     ⚠️ **Its first version INVENTED six pairs, caught by an independent review of PR #397 — two
     over-broad rules, and the lesson is that "more reach" is only a fix if it does not also add noise.**
     *(a)* It credited a returned local wherever the return *mentioned* it, so `return cls(sweeps=
     settings.sweeps_for(rate.rate))` credited `CoupledBlockSweep.calibrated` with the `ContractionRate`
     that `rate` held, and three `calibrated` pairs and three `SmoothedAmg*.build` pairs shared nothing but
     such incidental credits. *(b)* It typed a receiver by every class its producer's return mentioned, so
     `open_session()` returning `Session(Helper())` resolved `session._build` on `Helper` too — the very
     union the fix rejects, arriving through the producer's arguments. Both now use value positions only
     (`_value_positions`, `_returned_values`). **Every resolution rule has its own fixture and was
     mutation-checked** — each deleted in turn fails at least one test: the self-method, returned-local and
     inherited-method routes (`test_each_way_a_session_method_reaches_its_tail_is_followed`, one rule per
     fixture, because one fixture exercising all three let any one be deleted while the others carried the
     builder), receiver typing by value (`test_a_receiver_is_typed_by_what_its_producer_returns_not_by_its_arguments`),
     read-not-returned locals (`test_a_local_the_return_only_reads_is_not_credited`), the qualified-label
     wrapper exclusion (`test_a_wrapper_reached_through_a_typed_receiver_is_not_paired_with_its_callee`), the
     no-union decoy (`test_it_follows_a_method_through_a_receiver_whose_class_is_knowable`), and the
     bare-name fallback (the package test, which asserts `coupled_step` pairs with `flow_march_step`).
     ⚠️ **Still blind, and recorded so its silence is not read as clean:** a builder that returns a
     *closure* it defines rather than a call — `scalar_pseudo_transient_solve` returns `solve_scalar` — is
     credited with building nothing and never enters the report.

## Post-Change Checklist → 3. Tests — `.test_durations` and shard balancing

   **⚠️ The unit, slow and validation shards are balanced by `.test_durations`, and a stale one
   silently unbalances them.** Those tiers are heterogeneous (a 21 s scheme check beside a 242 s adjoint
   continuation), so `pytest-split` partitions them by recorded duration. With **no** durations it
   splits evenly by **count**, and then adding a test *anywhere* shifts the boundaries and can
   migrate the expensive tests onto whichever shard is already heaviest — which is not hypothetical:
   eight unrelated unit tests moved two ~106 s adjoint tests off the lightest slow shard (15:53) onto
   the heaviest (20:58), which then needed ~24.5 min of tests plus ~5 min of setup and was **killed
   at the 30-minute ceiling**. The failure looks exactly like a regression in the change under test
   and is not one. A weekly cron (plus `workflow_dispatch`) re-records the file and opens a
   metadata-only PR; label a PR **`refresh-durations`** to record it on that branch instead, which is
   the only way to populate it before the workflow has merged (`workflow_dispatch` is registered from
   the default branch only). **The weekly merge rewrites the whole file from what the refresh jobs
   record, so a tier that splits by it must also be recorded by one** — which is why the unit tier
   has its own `refresh-unit-durations` job; without it the first refresh would delete every unit
   duration and drop the unit shards back to a count split (13.0 / 9.3 / 15.7 min on 2026-10-08,
   py3.11, a mean of 12.7). The fast integration tier is deliberately **not** duration-balanced —
   those tests are homogeneous, and an even-by-count split balances their memory too.

## Post-Change Checklist → 4. Documentation sync — the left/right-preconditioning drift and the wrong-rule incident

   - **The docstrings and comments in the `.py` files** — both in the code you changed *and in
     every other module that describes the same behaviour*. In-code prose sits closest to the
     change and is the first thing a reader trusts, so a false claim there is the most expensive
     kind; it is also the only item on this list that no other item covers. The trap is the
     second half: one component's behaviour is typically restated in several modules, and a
     change updates only the file it edits. **Grep for the claim, not only for the symbol** —
     this class of drift renames nothing, so the Stale-Record Check's identifier grep is blind
     to it by construction. (Commit `e6bc18e` switched the linear solve from left- to
     right-preconditioning in `solve/linear.py` *alone*. `precondition/multigrid.py`,
     `flow/preconditioner.py`, `solve/continuation.py` and `solve/implicit.py` went on calling
     it a "left preconditioner" for two weeks — correct prose on the day it was written, false
     the day the default moved, and invisible to every check we had because "left" is an
     English word, not an identifier.)
   - **The matching `.claude/rules/*.md`** — the per-subsystem design record: binding decisions,
     interfaces, class/function names, file paths, and `BUILT` / `Not yet built` status. Inline
     the fact rather than pointing at any private note.
     ⚠️ **"Matching" means the rule whose `paths:` GLOB THE FILE YOU EDITED — check the frontmatter,
     do not go by subject.** This item asks which file *describes* your change; it does not ask whether
     that file will be *read*, and those come apart precisely when a rule's `paths:` and its prose have
     drifted. A rule that does not glob the file it governs never auto-loads for the person editing it,
     so binding prose put there is invisible exactly when it is needed. Two sessions hit this on one
     evening (2026-09-09) and **both passed this checklist as written**: one added methods to
     `solve/step_control.py` and recorded them in `solve-march.md` (which globs `march.py`, not
     `step_control.py`); the other *edited a binding class contract* into the same wrong file, making it
     more load-bearing on the way. Neither error is visible from the prose alone, and neither is visible
     from the `paths:` alone — only from checking them against each other. Read the **frontmatter**, not
     the file:

     ```
     for f in .claude/rules/*.md; do sed -n '/^paths:/,/^---/p' "$f" | grep -q "<file you edited>" && echo "$f"; done
     ```

     ⚠️ **A plain `grep -l "<file>" .claude/rules/*.md` is NOT the same question and gives the wrong
     answer here** — it matches prose mentions as well as globs. Asked for `solve/step_control.py` it
     returns three rules where only **one** globs that file, and one of the false positives is
     `solve-march.md`, i.e. precisely the wrong rule this item exists to steer you away from. (That
     mis-check was written into this very paragraph on its first draft and caught by running it, which
     is the general lesson: before trusting a command's output, confirm it answers the question you are
     asking and not a neighbouring one.)

## Engineering Principles → 3. Encapsulation — the `__all__` bullet (full text)

- **A package's `__all__` is its whole surface to other packages (binding, #284).** Another package
  imports from `aquaflux.<package>` (or a nested package such as `aquaflux.io.cad`), never from one of
  its submodules, and only names that package's `__init__` lists in `__all__`.
  `tests/unit/test_layering.py::test_one_package_reaches_another_only_through_its_public_surface`
  enforces it over every module in `aquaflux/`, the top-level ones included. When it fires, the remedy is
  to export the name, or — when the name is private and generic — to move it to the package it belongs
  in (`_bordered_preconditioner` was pure Schur elimination living in `flow/mean_velocity.py`; it is now
  `precondition.bordered_preconditioner`). Tests and `validation/` harnesses are exempt and may reach internals
  from the submodule. Exporting a name from a documented subpackage publishes it, so a name that is
  plumbing rather than something a user configures or implements belongs in the package that uses it.

## Engineering Principles → 3.5 No scope changes — the delivered-less example

- Delivering *less* than agreed and calling it done is the same defect as delivering something
  different. If only part of the agreed work is finished, say which part is missing, in the summary,
  not only in a tracked issue. (Task #29 was closed with half of it — the row-equilibrated measure —
  never built; that gap survived unnoticed until it was needed.)

## Architecture — vector algebra, ragged lists, Morton ordering (full text before condensing)

**Vector algebra lives in one leaf, `aquaflux/vectors.py`.** Per-element operations on fields
of small spatial vectors — the per-face/per-cell dot product `dot(a, b)`, squared magnitude
`norm_squared(a)`, the magnitude `norm(a)` (zero-safe: `0` with a zero gradient at the zero vector,
the root guarded inside its argument — reach for it wherever two points may coincide, never for
`jnp.linalg.norm` or a `sqrt(dot(...))` guarded after the root), scaling a vector field by a per-element scalar `scale(vectors, scalars)`, and
reflecting it in a plane through the origin `reflect(vectors, normal)` (the image-source method's
one formula, shared by `radiation/mirrors.py` and the photometric profile's mirror image)
— are defined once here and imported wherever the geometry, schemes, or flux operators contract
or scale a `(..., dim)` field (it imports nothing from `aquaflux` — only `jax.numpy` and the
standard library — so any subsystem may use it). **Preference (binding): keep vector math
readable — reach for these helpers instead of open-coding `jnp.sum(a * b, axis=-1)` or
`s[..., None] * v`.** The raw axis/broadcast bookkeeping obscures the math and drifts; the named
helper states the intent, and gives one home to change (Principle 2). Rank-3 tensor algebra
(Hessian outer products, `einsum`) stays explicit — the helpers target rank-2 vector fields.


## Architecture — ragged/morton/frozen-operator paragraphs and the build-target line (full text)

**Ragged lists live in one leaf, `aquaflux/ragged.py`** (numpy only, imports nothing from the
package, listed in `tests/unit/test_layering.py`'s neutral leaves): `rows` gathers chosen rows of a
compressed-sparse-row (CSR) list, `group` builds one from integer keys, `pairs_within_groups` forms the
per-group Cartesian product in flat form. The mesh's face-subset code (`FaceNodeConnectivity.select`,
`collapse.py`) and the radiation coarsener's batched checks all use it; before it, the first two
carried their own copy of `rows`. Reach for it before writing `np.repeat(..., counts)` index arithmetic.

**Points are ordered along a space-filling curve in one leaf, `aquaflux/morton.py`** (numpy only, a
neutral leaf like `ragged.py`): `morton_order(points)` sorts points along a Morton (Z-order) curve, so
any run of consecutive points is a local patch. The radiation package cuts every grouping of points
from it — shaft culling's receiver groups and facet clusters, the gather's chunks and lit blocks, the
transfer build's row blocks, `FacetClusters` — which before it carried two copies. ⚠️ **Each axis is
scaled to its own extent, deliberately**: cubic cells make groups on a long thin lamp more compact but
wrap them around it, so their facets stop facing one way, and measured on the whole Sozzi field they
left more pairs undecided and slowed the transfer build (#574, `.claude/notes/radiation-log.md`, "ONE MORTON ORDERING, PER-AXIS CELLS").

**Frozen preconditioner operators are assembled in one place, `aquaflux/precondition/frozen_operator.py`.**
The AMG preconditioners coarsen a *frozen* linearization of a transport equation — a symmetric
diffusive edge coupling, optionally plus first-order-upwind convection at a reference flux —
assembled once, off the jit path, as a `scipy.sparse` matrix. `ConvectionDiffusionStencil` — the
interior-face graph with its per-edge coefficients and per-cell boundary diagonal, as one value that
`assemble()`s itself, reports its `diagonal_parts()` and `detached(cells)` a fixation — is the single
description for all four consumers (with `decouple_dof` for the closed-domain pressure pin): the
pressure Schur, both velocity blocks, and the k/ω scalar transport, whose pseudo-time shift reads the
same stencil's diagonal rather than re-deriving it. It sits **beside**
`precondition/multigrid.py`, not inside it: every multigrid builder takes an assembled operator `a` and
knows nothing about meshes or fluxes. The first-order-upwind stencil is the **preconditioner's**
choice, not the model's — whatever scheme the residual uses for advection, the frozen operator always
upwinds first-order, because that is what makes it an M-matrix an aggregation hierarchy can coarsen —
which is why it is a solver concern and holds no mesh, field, or `jax` import.

```
Mesh (SoA topology) + FaceGeometry/CellGeometry            (classes)
   → operator strategies (DiffusionFlux, ...) consuming injected scheme strategies
   → mesh.face_cells.owner/safe_neighbour → compute flux → mesh.face_cells.scatter_*
   → cell residual R(state, params)
   → AD (jvp / vjp / IFT) → Jacobian / adjoint
   → Newton + implicit-diff linear solve → converged state (and its exact adjoint)
```

A Layer-0 escape hatch still lets an advanced user supply a raw flux/source **closure** —
but aquaflux's own built-in operators, schemes, and BCs are OO strategy classes, not
closures. The DSL that eventually emits terms is the last layer. Current build target:
hardcoded transient diffusion, no DSL.

## Testing Architecture → What a fast-tier test may spend — the speed-ups

- **A dense matrix built one column per dispatch.** A Python loop calling `jax.jvp` on each basis
  vector pays eager dispatch `ndof` times; `jax.jacfwd` / `jax.vmap` push the whole basis through in
  one batched pass and build the identical matrix (worth 3.6x and 2.9x on the two that did this).

## Post-Change Checklist → 4. Documentation sync — rule-of-thumb examples

   Rule of thumb: if a reader of one of these files would now be **misled** by what it says,
   fixing it is part of your change, not a follow-up. (The `MaterialModel`→`PropertyModel`,
   `structured_grid_2d(perturb=)`, and "git/CI not set up" drifts were all exactly this gap.)
   The committed **`.githooks/pre-commit`** reminder surfaces this whenever a commit touches
   `.py` code.

## Engineering Principles → 4. No compatibility shims — the adapter example

- **No thin adapters kept only to preserve an API.** A wrapper class or forwarding function that exists
  solely to re-expose a refactored abstraction under its former shape is dead weight: delete it and
  point callers at the real object. (E.g. a `PseudoTransientStep` *is* the `NewtonStrategy`, so a
  `PseudoTransientContinuation` class that only delegated `stepper`/`linear_solver`/… was removed in
  favour of a `momentum_continuation` factory returning the engine directly.)

## Architecture → `dot`: the compiled-residual census guard (2026-10-17)

- **Guarded since 2026-10-17 by `tests/unit/test_compiled_residual_census.py`** (fast tier, CPU
  backend only), which compiles the coupled RANS residual and its JVP and ratchets the count of
  `kind=kCustom` fusions and of those with an empty `outer_dimension_partitions` (2 in the residual,
  3 in the JVP, jax 0.11.2). Writing `dot` as `jnp.sum(jnp.multiply(a, b), axis=-1)` moves both to 27
  and 46. ⚠️ **The fixture size is load-bearing:** XLA starts partitioning kernels and emitting
  `__ynn_fusion`s somewhere between 768 and 1200 cells, so on a 768-cell grid the reverted `dot` changed no count at all. The module fails
  if its fixture stops seeing partitioned kernels.

## Testing Architecture → Scoped runs — measured cost (2026-10-17)

Measured in a 4-core / 15 GB Linux cloud container, Python 3.13, jax 0.10.2, `FASTGATE_JOBS=2`. Each
diff is a one-line comment appended to one module, run as `tools/fastgate.sh fast --changed --base HEAD`
from a separate worktree, one after another.

| change in | test files selected | tests | wall |
|---|---|---|---|
| `aquaflux/precondition/sparse_jacobian.py` | 53 of 240 | 1033 passed | 20:32 |
| `aquaflux/schemes/multiple_correction.py` | 77 of 240 | 1288 passed | 37:09 ¹ |
| `aquaflux/vectors.py` | (being measured) | | |

¹ A one-worker scoped run of about 50 s overlapped this run's last minute, so the figure may be slightly
high.

The selection is mostly not the module's own tests: `sparse_jacobian` reaches `field_split` and the
block preconditioner, and through them the coupled RANS and march tests, which hold most of the cost.
Before the run, an estimate from `.test_durations` (recorded sum / 2 workers) gave ~13 min for
`sparse_jacobian`; the gap is the fast integration tests, which had no recorded durations then.
