# CLAUDE.md — aquaflux Project Briefing

This is the slim root briefing for the `aquaflux` library: orientation,
conventions, the **Engineering Principles** this project is held to, testing, the
development workflow, and the **Post-Change Checklist**. Read it before writing
any code.

`aquaflux` is a **differentiable, unstructured, cell-centred finite-volume (FVM)
flow solver in JAX**, purpose-built to couple with **aquakin** (the differentiable
reactive-transport package) for water and environmental engineering reactors. It
is *not* a general-purpose CFD code — it is a bespoke tool for the CFD ∩ water
intersection.

Detailed subsystem guidance is **split out of this file** into **path-scoped rules** under
`.claude/rules/*.md`: each carries `paths:` frontmatter and auto-loads when you read or edit a
file matching its glob, so it loads only when relevant rather than every session. Together with
this file, those rules are the project's design record — they carry the *why* behind each
subsystem and must **stand on their own** (a contributor has the `.claude/` tree but none of the
author's private working notes). When a rule would otherwise point at an external note, inline
the fact instead.

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

When you change a subsystem, the matching rule below is the authoritative
guidance for it — keep it updated as part of the change (this file's Post-Change
Checklist still governs).

## Index — where the detail lives

**Path-scoped rules (auto-load on matching files):**

| Rule | Loads for | Covers |
|---|---|---|
| `.claude/rules/mesh.md` | `aquaflux/mesh/**` | static connectivity arrays, face geometry, skewness vectors `D`/`fip`/`R`, boundary patches |
| `.claude/rules/discretization.md` | `aquaflux/discretization/**` | the Layer-0 residual substrate: gather→compute→scatter `segment_sum` assembly, diffusion/transient operators, why hand-linearization is *deleted* |
| `.claude/rules/schemes.md` | `aquaflux/schemes/**` | first-class swappable numerics: face interpolation, gradient reconstruction, non-orthogonal correction |
| `.claude/rules/boundary.md` | `aquaflux/boundary/**` | weak boundary-face-value closures (BC = special face interpolator); the shared per-patch fold |
| `.claude/rules/properties.md` | `aquaflux/properties/**` | physical property model (density/viscosity/conductivity): `Property` (constant / per-zone / calculated) collected in a `PropertyModel`, decoupled from the numerics |
| `.claude/rules/solve.md` | `aquaflux/solve/**` | **the residual-agnostic layer every residual runs on** (Principle 3.6): the staged driver, the shifted-step assembly, measures, Newton on the residual, linear solve with implicit differentiation / `custom_vjp`, the preconditioner risk. Split by subsystem into narrower-scoped siblings (`solve-direct-preconditioners.md`, `solve-amg-multigrid.md`, `solve-flow-block.md`, `solve-field-split.md`, `solve-globalization.md`, `solve-march.md`, plus reference-only `-log.md`/`solve-refuted-directions.md` files in `.claude/notes/`) — see `solve.md`'s own "Index — where the detail lives" |
| `.claude/rules/flow.md` | `aquaflux/flow/**` | coupled p–U block: momentum (reusing advection/diffusion) + Rhie–Chow continuity, differentiated `a_P` (frozen only in the preconditioner), monolithic AD-Jacobian solve |
| `.claude/rules/turbulence.md` | `aquaflux/turbulence/**` | k–ω SST closure + the segregated flow–turbulence loop: segregated forward / coupled adjoint, outer-loop globalization, positivity-floor adjoint honesty |
| `.claude/rules/transport.md` | `aquaflux/transport/**` | scalar transport by a converged flow (species, temperature, tracers): why a concentration rides the *volumetric* flux, the effective-diffusivity convention, sub-patch injection without a mesh change — the aquakin reaction seam |
| `.claude/rules/io.md` | `aquaflux/io/**` | mesh import, **CAD import** (STEP through OpenCASCADE/OCP into exact `solids` bodies checked against the drawing by boundary distance, and size-bounded emitting triangles — the only kernel import is `io/cad/kernel.py`), **and field export**: the `MeshReader` strategy + the OpenFOAM polyMesh reader (ASCII); parse→assemble→collapse seams, empty-patch 2D collapse (a `mesh/` transform), reserved-name guard; the OpenFOAM field writer that inherits dimensions and `boundaryField` from a template so its output restarts; the VTK XML `.vtu`/`.pvd` writer (arbitrary polygon/polyhedron cells, appended-binary) and why its face winding is re-derived rather than taken as stored |
| `.claude/rules/validation.md` | `validation/**` | the scientific cases and study harnesses: why no test tier drives them, the obligation to check an API change against them, what the static guard catches and what it cannot, and the harness traps that have produced wrong results |
| `.claude/rules/parallel.md` | `aquaflux/parallel/**` | distributed memory: graph partitioners, the `PartitionedMesh` owned+halo decomposition, uniform-shape padding, and the `shard_map` residual that runs an *injected* assembler per device (never a re-implementation) |
| `.claude/rules/radiation.md` | `aquaflux/radiation/**` | ultraviolet fluence rate `G` by a deterministic backward gather over surface facets: the two closed-form solid-angle kernels and why they are not interchangeable, the `arctan2`/clip/magnitude/safe-root details that are load-bearing, and the `F_ii = 0` convention callers must honour |
| `.claude/rules/solids.md` | `aquaflux/solids/**` | solid bodies answered by formula rather than search: the `Body` contract (`blocks`/`contains`/`traceable`), analytic primitives, constructive solid geometry over line intervals, `Outside` (a vessel as the fluid it holds), the grazing-robust cylinder discriminant; generic geometry, held to the same import-nothing rule as `solve/` — what a CAD model is read into and what radiation shadows with |
| `.claude/rules/case.md` | `aquaflux/case/**`, `aquaflux/__main__.py`, `validation/*/case.yaml`, `validation/*/cases/*.yaml` | a whole case described in one YAML file: the `CaseSpec` core plus the physics and drive discriminators, one physical kind per boundary patch (closures and wall set derived, never restated), the fluid stated once with exactly one viscosity, why the YAML parse is 1.2 and refuses duplicate keys, how the build derives each equation's closures, the wall set and one property model (bit-identical to the hand-built drivers, and the float-leaf trap that almost hid a recompile), the solver section (three solves, the library's own settings values read directly, a script may observe a solve but never configure it), `aquaflux run` with its outputs section and run records, and what is not built yet |

---

## Task lifecycle — the operational sequence, in order

The rules below are grouped by *topic*, which is the wrong order for doing a task. This is the right
order. Each line links to the section that explains why; the point here is that the sequence is
findable in one place rather than reconstructed from three sections hundreds of lines apart.

**Before the first branch or the first test run**
1. `git fetch origin`, and check `git log --oneline HEAD..origin/main` is empty — branching from a
   stale base is the failure this prevents (*Start from an up-to-date main*).

**While working**
2. Run long solves with `validation/run_case.sh <script.py>`; run tests with `tools/fastgate.sh`
   (*Use the blessed command*). One case at a time — these are memory-bound.
3. Record what every measurement was taken under: defaults in force, state, operating point
   (*Record what a measurement was taken under*). A number without its configuration cannot be
   falsified later, which is worse than being wrong.
4. If you deviate from an agreed approach, say so **at the point of deviation**, and never report a
   measurement taken under a deviation without naming it (*Principle 3.5*).

**Before every commit**
5. **Stale-Record Check** — ask what your change makes FALSE, not just what it leaves incomplete. Run
   the grep in that section; it is the most expensive recurring defect here.
6. **Post-Change Checklist** — principles review, `ruff check` + `ruff format`, the comment-hygiene
   greps, tests (including the `slow`/`validation` tiers if your change could reach them), and the
   documentation sync.

**Never without being asked**: commit or push, change a shipped default, or remove a guard.

---

## Engineering Principles

> **Read this section as binding, not aspirational.** These three principles exist
> because they address specific, observed failure modes. They take precedence over
> shipping something fast. If a principle and a deadline conflict, the principle wins —
> there is no deadline on this project that outranks code quality.

### 0. Maintainability over speed (the overarching preference)

**This project explicitly prefers spending more time to produce maintainable,
well-tested, non-duplicated code over returning something quickly.** Do not
optimize for a fast first answer or the smallest diff. A slower, cleaner,
better-factored solution is the *correct* answer here — not a luxury.

Concretely:
- When two implementations are possible — one quick to write, one better
  structured — **choose the better-structured one and spend the extra time.**
- It is in-scope to refactor code you are touching so the change lands clean.
  Leave each file better than you found it (within the change's blast radius).
- If you find yourself reaching for "this is good enough to ship," stop: that
  instinct is the failure mode this project is guarding against. Take the time.
- A correct-but-ugly prototype is acceptable *only* as an explicitly-labelled
  intermediate step that you then refactor before the task is considered done —
  never as the delivered result.
- **Delete dominated methods while the code is pre-release.** When a method,
  strategy, or code path is *dominated* — another one does its job better across
  the regime this project actually targets, and nothing selects it in production
  or integration tests — prefer **deleting** it over keeping it "just in case."
  Unused alternatives are not free: they carry maintenance cost, bloat the API,
  and force awkward abstractions (a unifying refactor must accommodate the dead
  path too). Before adding or keeping a second method for the same job, ask *is
  the first one dominated, and would we actually miss it?* A method with a
  genuine, exercised niche stays; a dominated one goes. The code is pre-release
  and everything lives in git history, so a method deleted now is cheap to
  restore if a real need for it ever appears — far cheaper than carrying it
  unused indefinitely. (When in doubt about whether a niche is real, surface the
  trade-off rather than silently keeping the cruft.)

### 1. A fully object-oriented, testable design (do NOT write quick-to-ship code)

**This project is fully object-oriented, mirroring the reference C++ architecture
(strategy-per-operator + factories).** Testability is achieved *through* clean OO
design — clear interfaces, dependency injection, small single-responsibility
classes — **not** by avoiding classes. The observed failure mode to guard against
is code that is fast to produce but hard to test: hidden state, hard-wired
dependencies, giant multi-responsibility units, logic entangled with I/O.

Rules, with FVM-specific teeth:

- **Strategy objects with explicit interfaces.** Each operator, scheme, boundary
  condition, and solver is a **class** implementing an abstract interface
  (`Protocol`/ABC) — the strategy-per-operator pattern from the C++
  (`FaceFluxCalculator`, `FaceInterpolator`, `GradientCalculator`,
  `BoundaryValueCalculator`, `TransientCalculator`). Concrete strategies are built
  by **factories** from config. Prefer this to bare module-level functions.
- **Dependency injection through constructors.** A strategy receives its
  collaborators (sub-schemes, coefficients, the mesh/geometry it needs) as
  constructor arguments, so a test constructs it with a stub or trivial
  collaborator. Never hard-wire a concrete dependency inside a class.
- **Single responsibility, small classes.** One class does one thing. If a method
  needs a comment "# step 3", it is probably several classes.
- **All physical and numerical parameters are external**, supplied via the
  constructor (never baked into a method as literals). This mirrors aquakin's
  "rate constants are always external via `params`" — mandatory for AD-based
  sensitivity and parameter estimation, and it makes every strategy testable with
  chosen inputs.
- **Design objects around the smallest sufficient collaborators.** A diffusion-flux
  strategy should be testable with a *stub interpolation and a two-cell mesh* — not
  by constructing a whole solver. If a test needs a large object graph, a seam
  (usually a missing injected interface) is in the wrong place.
- **Immutable, side-effect-free methods — the JAX requirement, satisfied the OO
  way.** Objects are immutable `equinox.Module`s (JAX pytrees); methods return new
  values rather than mutating state. That immutability/referential-transparency is
  what makes the classes `jit`/`grad`-compatible — it is emphatically **not** a
  reason to avoid classes. OO and differentiability coexist via `equinox`.
- **Separate computation from orchestration and I/O.** Mesh *reading* is a distinct
  class from geometry *computation*, which is distinct from residual *assembly*.
- **Every operator/scheme ships an operator-level unit test** — e.g. an
  order-of-accuracy check on an analytic field — not only an end-to-end test. If a
  bug can only be caught end-to-end, the unit was built wrong.
- **Concrete trigger:** *If you cannot unit-test a class without constructing a
  large object graph or monkeypatching state, the design is wrong — fix the seam
  (usually a missing injected interface) before you ship it.*

### 2. One source of truth — no duplicated logic

The observed failure mode: the same logic implemented in more than one place, so
a fix or change has to be made in several spots (and inevitably isn't). The
reference codes re-derive face geometry in multiple subroutines — **do not
replicate that.**

Rules, with FVM-specific teeth:

- **One canonical implementation per concept.** Face geometry (`D`, `fip`,
  skewness `R`), the diffusion flux, the gradient reconstruction, the transient
  term — each is defined *once*, in one module, and imported everywhere it is
  needed. Never re-derive a formula inline "just here."
- **Schemes are first-class objects precisely to avoid duplication** — one scheme
  defined once, consumed by many equations. Never inline
  a scheme choice into an operator.
- **Constants and geometric mappings live in a single module.** No magic numbers
  scattered across kernels.
- **Before writing a formula, search for it.** If similar logic already exists,
  call it. If it exists in two places, that is a defect — extract it into one
  class/method *now*, as part of your change.
- **No copy-paste-modify.** If you are tempted to copy a block and tweak it,
  parameterize the original (or introduce a shared base class / injected
  collaborator) so both call sites share it.
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
  mass-flow builder's Euclidean forward tolerance carries the reason (its bordered measure has no
  row-scaled form) at the constant itself, so the next reader sees a decision rather than an omission.
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
  counterproductive and since deleted along with the `while_loop` and seeding dance they alone required;
  `grow` is inert on a non-default measure and stays, since inert is not the same as counterproductive).
  Principle 0 says delete those, not promote them onto the shared builder.
- **Concrete trigger:** *If a change would require editing the same formula — or the same wiring, or the
  same default — in more than one place, it is in the wrong place: consolidate it first, then make the
  change once. "More than one place" includes two functions in the same file; a duplicated tail is not
  excused by proximity. And before adding a keyword argument to a builder, run `tools/sibling_builders.py`
  (Post-Change Checklist item 2): if a sibling constructs the same class, the argument belongs on a
  shared builder they both call — put it there, in this change.*

> These two principles reinforce each other: small, single-responsibility, injected
> classes (Principle 1) are also the natural unit of reuse (Principle 2). Code that
> is hard to test is usually also duplicated, because it is hard to reuse.

### 3. Encapsulation — pass the object, not its guts

The observed failure mode (a whole review's worth): raw arrays and index plumbing leak across
interfaces instead of living behind the object that owns them, so the same low-level mechanics get
re-open-coded everywhere and drift. The `mesh.face_cells` / `mesh.face_nodes` connectivity refactor
is the reference standard — hold the same bar everywhere. Each rule below is a **real smell that was
found and fixed**; treat them as binding, not aspirational.

- **Pass the cohesive object, not a fistful of its arrays (no primitive obsession).** A function that
  needs owner/neighbour takes the `FaceCellConnectivity` (`mesh.face_cells`), never loose
  `(owner, neighbour)` arrays; one that interpolates a cell field takes the field + `face_cells`, not
  four pre-gathered owner/neighbour pairs. **If a signature carries more than ~5 arrays that travel
  together, they are a missing object** — bundle them (a `FaceState`-style record or a real geometry
  object).
- **Take the smallest sufficient collaborator.** A function that reads only `mesh.face_cells` takes
  `face_cells`, not the whole `Mesh`; one that reads only cell centroids takes the centroid array,
  not a `CellGeometry`. Demanding a big object to touch one field forces every test to build the
  world (the Principle-1 testability seam).
- **No duplicate accessors — one name per value.** Do not add a forwarding property
  (`mesh.owner` → `mesh.face_cells.owner`) or re-derive a value that already has a home
  (`interior_mask(fc.neighbour)` when `fc.interior` exists; a private `_dim` forwarding
  `self.mesh.dim`). Synthesized higher-level *concepts* (`mesh.dim`, `mesh.n_cells`) are fine — they
  are not duplicate views of one stored array; a second spelling of the *same* array is.
- **The object owns the operations over its data, not just the data.** If consumers keep open-coding
  the same `segment_sum` / `neighbour >= 0` / index arithmetic against an object's raw arrays, the
  operation belongs **on** the object (that is why `scatter_max`/`scatter_min`, `interior_edges`,
  `scatter_owned_partitions` exist) — add it there and have callers compose it. **Never duck-type a
  lookalike of a real class** (`_FaceGeom` re-declaring `FaceGeometry`); carry the real object — they
  are `equinox.Module` pytrees and map through `shard_map`/`jax.tree.map` unchanged.
- **One formula, one home (Principle 2, sharpened for arithmetic).** A linear face interpolation
  `(1-g)·a + g·b`, a projection factor `g`, a normal distance `d·n`, an owned→global scatter — each is
  defined once (`schemes/interpolation.py`, a `_normal_distance` helper, …) and imported. Before
  writing `(1-g)*a + g*b` or `sum(d*n)` inline, **search for the existing helper.**
- **No god-methods / god-objects.** A method with several `# step N` blocks, or that branches a
  strategy family inline (`if inner == "smoothed"/"multigrid"/"jacobi"`), is several classes: extract
  a builder plus an **injected strategy hierarchy** (the `InnerSchurSolver` / `VelocityBlockSolver`
  pattern, mirroring the operator/scheme/BC strategies). Inline flat-vector index arithmetic for a
  state layout is a missing value object (`solve/state.py::FieldLayout`).

- **Concrete trigger:** *if you are about to pass an object's raw arrays to a function, thread a loose
  per-cell/per-face array a second time, add a forwarding property, duck-type a lookalike of an
  existing class, or write a one-line formula that already exists — stop and reach for the
  object/helper instead.*

### 3.5 No scope changes without asking (binding)

**Once the user has agreed to an approach, do not change it — not the design, not the cadence, not the
coverage — without saying so and getting agreement first.** Narrowing scope is a change just as much as
widening it. "I'll do the simpler version for now" is a scope change. "Phase 1 / Phase 2" is a scope
change. Substituting a cheaper approximation of an agreed design is a scope change.

This is not about ceremony; it is about the *evidence* that follows. The observed failure: the user
agreed a residual measure whose scales are rebuilt every outer iteration and held fixed only across a
line search. What got built instead froze the scales once at the initial condition — described as
"Phase 1" rather than flagged as a deviation. The measurement taken under that shortcut then showed the
measure failing, and that failure was reported as if it were a property of the *formula*. It was a
property of the unapproved substitution. The user had to catch it.

Concretely:
- **Flag the deviation at the point of deviation**, in plain terms: "you agreed X; I am about to do Y
  instead, because Z — is that acceptable?" Do not bury it in scoping language ("for now", "as a first
  cut", "the pragmatic choice") where it reads as a plan rather than a departure.
- **Never report a measurement taken under a deviation without naming the deviation** in the same
  breath as the result. A number obtained from a configuration the user did not agree to is not
  evidence about the thing they asked about, and presenting it as such actively misleads.
- If an implementation obstacle forces a change (a hashability constraint, a compile cost, a missing
  seam), **that obstacle is the thing to surface** — it is usually the most useful information in the
  task, and the user often knows a way through it that is not visible from the code.
- Delivering *less* than agreed and calling it done is the same defect as delivering something
  different. If only part of the agreed work is finished, say which part is missing, in the summary,
  not only in a tracked issue. (Task #29 was closed with half of it — the row-equilibrated measure —
  never built; that gap survived unnoticed until it was needed.)

### 3.6 Layering — generic machinery goes in the generic layer, not the first physics package that needed it (binding)

The observed failure: the staged solve driver (segments, refresh, convergence check), the preconditioner
sessions, the shifted-step assembly, the residual measures and the Krylov-regime settings were all
written for the k–ω SST solve, **inside `turbulence/coupled.py`**, because turbulence was the first
thing that needed a robust march. None of them mentions turbulence. The consequence was not tidiness: a
**laminar** flow problem had only a bare pseudo-transient step, so it could not be marched by the
dual-time / retry / row-scaled-norm / refresh machinery the turbulent case runs on, and a control
experiment ("does this solver behaviour need turbulence?") had to be run on a weaker driver that did
not isolate the variable (#448). The problem had been filed as #277 weeks earlier; the file was ~4000
lines by the time anything moved out of it, and nothing measured that it was growing.

- **The placement test, applied when you write the code and not later.** Describe the function in one
  sentence. If the sentence needs a physics word — `k`, `ω`, `ν_t`, a closure, a wall function, "coupled
  RANS" — it is physics. If its body would be identical for any other residual (a loop over segments, a
  step assembled from a shift policy, a norm built from a layout, a settings value for a Krylov solve), it
  is **generic and belongs in `solve/`** (or another residual-agnostic module) *the first time*. "Only
  turbulence uses it so far" is exactly how it ended up in the wrong place; the first consumer is not a
  reason.
- **Physics packages compose; they do not own loops.** A physics package supplies a residual, a shift
  policy, a `ResidualMeasures`, a drift measure and a `ContinuationSource` to the generic driver. If a
  physics module is growing a `for segment in ...` loop, a step-assembly tail, a retry rule or a settings
  value that names no physics, that code is in the wrong package.
- **A capability must be reachable from every residual, or its absence is a defect.** Before adding to the
  march (a trigger, a control, a retry, a guard), ask which residuals can reach it: flow-only, scalar,
  coupled. "Exists on one path only" is Principle 2's sibling-builder defect at package scale, and it is
  invisible to a diff-scoped check for the same reason. A laminar solve is the same machinery configured
  differently — never a weaker parallel path; if a control experiment needs a weaker driver, that is the
  finding.
- **Names that carry the first consumer are the smell to grep for.** `coupled_step`, `_coupled_step`,
  `solve_coupled`, `_CoupledMeasures` describe *one* residual. When the machinery underneath them turns out
  to be generic, the generic part takes a neutral name in `solve/` and the specific one becomes a thin
  configuration of it.
- **Two mechanical guards, both in `tests/unit/test_layering.py`, both in the always-on gate.** `solve/`
  imports nothing outside itself (it holds no mesh, field or physics import — that is what lets every
  residual run on it); and a module in `flow/`, `turbulence/`, `transport/` or `radiation/` past 1500
  lines needs a ratchet entry that only ever goes down. **Tripping the size guard is a prompt to sort the
  file's contents into generic and physics — not to split it by topic**, which moves the problem into
  smaller files without moving it to the right package.
- **Concrete trigger:** *if the docstring of what you are about to add to a physics package cannot be
  written without that package's vocabulary, it is physics and stays; if it can, stop and put it in
  `solve/`. And if you are writing a second, simpler path for a residual because the robust one "needs
  turbulence", the robust one has the wrong dependency.*

### 4. No compatibility shims before release (pre-release policy — remove this principle at 1.0)

The project is **pre-release: there are no external API consumers, so breaking API changes are free.**
When a refactor changes a public surface, **change the surface and update every call site — do not
preserve the old one.**

- **No thin adapters kept only to preserve an API.** A wrapper class or forwarding function that exists
  solely to re-expose a refactored abstraction under its former shape is dead weight: delete it and
  point callers at the real object. (E.g. a `PseudoTransientStep` *is* the `NewtonStrategy`, so a
  `PseudoTransientContinuation` class that only delegated `stepper`/`linear_solver`/… was removed in
  favour of a `momentum_continuation` factory returning the engine directly.)
- **No deprecation shims, aliases, or back-compat branches.** No `old_name = new_name` re-exports, no
  `if legacy_arg is not None` compatibility paths, no keeping a parameter alive "so nothing breaks."
  Rename/retype/delete in one change and fix the callers and tests in the same change.
- A **builder function or factory is not a shim** — it constructs and returns the real object (like
  `momentum_continuation` or `reused_flow_solve`). What this bans is the *delegating wrapper* that adds
  a layer over an object already fit to use directly. **A private tail that several public builders
  delegate into is not that wrapper either** — it is Principle 2's one implementation, and nothing in
  this principle licenses a second full builder for an object that already has one.

**This principle is a pre-release convenience and must be deleted at the first stable release**, when
backward compatibility becomes a real constraint and the calculus reverses.

---

## Design Goals

- Fully differentiable end-to-end (flow + coupling) via JAX reverse-mode + implicit
  differentiation — gradients through a single converged linear solve, never by
  unrolling the iteration onto the tape.
- Coupled (block) pressure–velocity with Rhie–Chow — not segregated SIMPLE/PISO.
- Numerics separated from physics: reconstruction/interpolation/gradient schemes are
  named, swappable, independently order-of-accuracy tested.
- The PDE core is a **DAE-residual engine**; aquakin's ODE reactors are the special
  case (all-accumulation, no faces).
- Knob-free robustness for non-expert (water/chem-eng) users.

---

## Technology Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python | Ecosystem, aquakin parity |
| Numerical backend | JAX | AD for free, jit, GPU batching |
| OO modules | **equinox** (`equinox.Module`) | jit/grad-native classes with inheritance — the strategy hierarchies are `Module`s, not bare functions. Already a transitive dep (diffrax/lineax are built on it), so no new footprint. |
| Linear solves | lineax (fallback `jax.scipy.sparse.linalg`) | traced, implicit differentiation of the solve |
| Adjoint of the coupled solve | `custom_vjp` around the linear solve | exact, memory-flat gradient independent of iteration count |
| Transient integration | Diffrax | traced, shared with aquakin |
| Mesh | static connectivity arrays + `segment_sum` scatter | XLA-friendly graph/message-passing layout |
| Case file | **PyYAML** (`aquaflux/case/`) | a case's mesh, fluid, physics, boundaries, numerics and solver in one YAML document, read by the YAML 1.2 rules for plain values (PyYAML's own 1.1 rules read `1e-5` as a string and `no` as a boolean) and validated per position by `solve.SettingsMapping` — so no pydantic. The equation DSL (YAML → AST emitting terms) is a different thing and is still the **last** layer, not built |
| Bounded compilation cache | **filelock** (`aquaflux/__init__.py`) | `import aquaflux` points JAX's persistent on-disk compilation cache at `~/.cache/aquaflux/jax` and **bounds it**. JAX's default `jax_compilation_cache_max_size` is `-1`, which its LRU implementation reads as *no eviction* — an unbounded cache never prunes and only grows (one checkout reached 88 GiB across 2128 entries, 95% of it more than a week old, 42 of them ~1 GiB coupled-solve programs). Setting a byte bound turns real eviction on, and JAX takes an inter-process lock through `filelock` to do it. **⚠️ A BOUND WITHOUT `filelock` DISABLES THE CACHE ENTIRELY** — every read and write fails with a `UserWarning` and it stores nothing, which is worse than no bound, so the package checks and degrades to merely unbounded rather than silently dead. Override the size with `AQUAFLUX_COMPILATION_CACHE_MAX_GIB` (negative for no bound), the location with `AQUAFLUX_COMPILATION_CACHE_DIR`, or switch it off with `AQUAFLUX_DISABLE_COMPILATION_CACHE=1`. |

The coupled-block preconditioner is the **top research risk**. The first build-on candidate
to evaluate is `jaxamg` — do not assume it; verify its adjoint is implicit-diff and that it
preconditions a *block* system. (See `.claude/rules/solve.md` for the chosen block-triangular
SIMPLE-type direction, and `.claude/rules/solve-amg-multigrid.md` for the known traps.)

---

## Architecture

This section is the architectural orientation; the subsystem rules under `.claude/rules/`
carry the per-package detail.

**Object-oriented over a struct-of-arrays substrate.** The mesh and fields are stored
as flat arrays (struct-of-arrays: one array holding data for *all* elements, never one
object per element — the only thing forced on us, because JAX vectorizes arrays, not
object graphs). *Everything above that data layer is object-oriented*: `Mesh`,
`FaceGeometry`/`CellGeometry`, and the strategy hierarchies (operators, schemes, boundary
conditions, solvers) are `equinox.Module` classes mirroring the reference C++
architecture. SoA is a data-layout constraint; it is **not** a licence for procedural
code.

**The Layer-0 residual substrate is the foundation.** Everything reduces to a discrete
cell residual `R(state, params)` assembled by **gather → compute → scatter**. The
gather/scatter *mechanics* are not open-coded per operator: they are the one
**connectivity API** — `mesh.face_cells` (`FaceCellConnectivity`: direct indexing on `owner` /
`safe_neighbour`, `scatter` / `scatter_conservative` / `scatter_symmetric`) and
`mesh.face_nodes` (`FaceNodeConnectivity`) — that owns the SoA `segment_sum` over
face→cell / face→node index arrays and the boundary convention. Operators, schemes, BCs,
and even the mesh geometry *compose* it, so each writes only physics/math (Principle 2;
see `.claude/rules/mesh.md`). The Jacobian and adjoint come from **AD** — there are **no
hand-derived linearization coefficients** anywhere (the central simplification over the
reference codes). Each operator writes the full
physical flux as one honest residual term; AD assembles the matrix.

**Vector algebra lives in one leaf, `aquaflux/vectors.py`.** Per-element operations on fields
of small spatial vectors — the per-face/per-cell dot product `dot(a, b)`, squared magnitude
`norm_squared(a)`, and scaling a vector field by a per-element scalar `scale(vectors, scalars)`
— are defined once here and imported wherever the geometry, schemes, or flux operators contract
or scale a `(..., dim)` field (it imports nothing from `aquaflux` — only `jax.numpy` and the
standard library — so any subsystem may use it). **Preference (binding): keep vector math
readable — reach for these helpers instead of open-coding `jnp.sum(a * b, axis=-1)` or
`s[..., None] * v`.** The raw axis/broadcast bookkeeping obscures the math and drifts; the named
helper states the intent, and gives one home to change (Principle 2). Rank-3 tensor algebra
(Hessian outer products, `einsum`) stays explicit — the helpers target rank-2 vector fields.

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

**Ragged lists live in one leaf, `aquaflux/ragged.py`** (numpy only, imports nothing from the
package, listed in `tests/unit/test_layering.py`'s neutral leaves): `rows` gathers chosen rows of a
compressed-sparse-row (CSR) list, `group` builds one from integer keys, `pairs_within_groups` forms the
per-group Cartesian product in flat form. The mesh's face-subset code (`FaceNodeConnectivity.select`,
`collapse.py`) and the radiation coarsener's batched checks all use it; before it, the first two
carried their own copy of `rows`. Reach for it before writing `np.repeat(..., counts)` index arithmetic.

**Frozen preconditioner operators are assembled in one place, `aquaflux/solve/frozen_operator.py`.**
The AMG preconditioners coarsen a *frozen* linearization of a transport equation — a symmetric
diffusive edge coupling, optionally plus first-order-upwind convection at a reference flux —
assembled once, off the jit path, as a `scipy.sparse` matrix. `convection_diffusion_operator(...)`
(with `decouple_dof` for the closed-domain pressure pin) is the single assembler for all four
consumers: the pressure Schur, both velocity blocks, and the k/ω scalar transport. It sits **beside**
`solve/multigrid.py`, not inside it: every multigrid builder takes an assembled operator `a` and
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

### JAX x64 Mode

FVM (and stiff coupling) require 64-bit floats. `aquaflux` enables x64 mode at import
time. This is **global, process-wide** JAX state, so it is a documented side effect of
`import aquaflux`. Do not remove the x64 enablement. See `tests/unit/test_x64_import.py` —
the effect is process-global, so it is tested in a subprocess.

---

## Testing Architecture

### Three layers

- **Unit** — individual components in isolation; fast; run on every change. Every
  numerical operator has an operator-level test (order-of-accuracy on an analytic
  field). This layer is where Engineering Principle 1 is enforced.
- **Integration** — full assembly → solve pipeline against analytical solutions.
- **Validation** — scientific correctness against analytical/published solutions,
  marked `@pytest.mark.validation`, run separately.

```bash
tools/fastgate.sh              # the always-on gate: not slow, not validation
tools/fastgate.sh validation   # analytical/published-solution suite
tools/fastgate.sh slow         # the slow tier
tools/fastgate.sh all          # everything
```

### A test that cannot fail is not a test (binding)

**Before a new test is considered done, name the specific wrong answer it would catch.** A test that
would pass against a broken implementation is not incomplete coverage — it is a false signal, read by
every future reader (including a reviewing agent, and including *you* on a later task) as evidence the
code works. This applies with equal force to a test you write yourself and one an agent hands back after
being asked for coverage: a subagent asked to "add a test" tends to return something that runs green
against the code as it stands, without ever checking whether it would *also* run green against a
plausible wrong version of that code. That gap is what this section exists to close.

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

**How to check a test can fail, concretely:** comment out, invert, or disable the specific line of
production code the test is meant to pin, rerun *only* that test (or the small file it lives in), confirm
it goes red, then restore the line exactly. This costs seconds per check and is the only reliable way to
know a test is not vacuous — reading a test is not enough, because a plausible wrong implementation is
exactly the thing a reader's own mental model tends not to simulate (that is precisely why the bug was
possible to write in the first place). Do this for a new test before calling coverage complete, and
especially before reporting a coverage number ("N passed", "all tests green") as evidence that a change
is correct — a passing count is evidence the code satisfies the tests, never evidence the tests would
catch a wrong version of the code, unless that has actually been checked.

**Concrete smells — reject each on sight, in your own tests and in an agent's:**
- **The assertion is satisfied by many wrong answers.** `assert result is not None`, `assert result.shape
  == expected_shape`, `assert not np.any(np.isnan(result))` (see this file's own "'without NaNs' is the
  floor, not the test", under Canonical tests below) all pass across a wide range of incorrect outputs.
  Assert the *value* against an independent reference — an analytic solution, a hand-computed number, a
  differently-derived path, a property the *specific* correct answer has and a wrong one plausibly would
  not — wherever one is obtainable.
- **Every fixture exercises the same code path.** If several tests all build the mesh/state/config the
  same way modulo one changed number, a branch that only a genuinely *different shape* of input would
  reach is untested however many of those tests pass. (The collapse.py gap above: every fixture gave a
  whole face-subset a uniform node count, so a bug that mixed up rows across differently-sized faces had
  nothing in the suite that could catch it.)
- **The mock or stub replaces the exact thing under test.** A test that replaces the function whose
  behaviour is in question, then asserts the replacement was called, tests the wiring around the
  function, not the function.
- **`pytest.raises(..., match=...)` only checks that *some* error was raised, not *which* one.** A regex
  broad enough to match several distinct failure branches (or matched against a generic outer message)
  cannot tell you the specific branch that should have fired is the one that did — construct the input so
  only the branch under test can plausibly raise, or match a substring unique to it.
- **A "regression" test pins today's output with no independent way to know it is right.** Useful for
  catching *drift* from this point forward, useless for telling you whether the pinned number was already
  wrong on the day the test was written. Prefer a value checkable against something other than "what the
  code currently returns" whenever the cost of doing so is reasonable.

**When reviewing test coverage — your own or an agent's — ask, per new test: what specific wrong answer
does this catch, and did I verify that it would?** A green suite answers "does the code behave as
written"; it does not answer "can these tests tell a correct implementation from a broken one" unless
that has been checked by actually breaking something and watching the suite react.

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

**⚠️ CI IS NOT A SUPERSET OF A LOCAL RUN, AND `importorskip` IS WHY.** CI installs `.[test]`, which does
**not** include the optional `petsc` extra, so every module guarded by `pytest.importorskip("petsc4py")`
— `tests/integration/test_coupled_amg.py` and `test_coupled_field_split.py` — is **skipped there and runs
only on a machine that has PETSc**. A green CI slow tier therefore says nothing about them, and their
failures surface only locally, which reads as "broken on my machine" when it is the opposite: local is
the only place they are checked at all. This is the same shape as a check that has stopped seeing
anything (`tools/check_hooks.sh`, `tools/sibling_builders.py`) — a skip and a pass are indistinguishable
from the exit status. **When a test fails locally and passes in CI, check the skip counts before assuming
the difference is your machine**: 2026-08-21 the CI slow tier reported `4 skipped` per shard against zero
locally, and three tests had been failing on `main` for four days with CI green throughout.

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
order; and it pins **one BLAS/XLA thread per worker**, because otherwise every worker grabs every
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

### What a fast-tier test may spend (binding)

The fast gate is read on every change, so a test that runs longer than it needs to is a tax on every
future change, paid by everyone. Three patterns produced **most** of the tier's wall clock as of
2026-08-23, and all three are cost with no coverage attached — check for them before adding a solve
to this tier:

- **A fixed iteration count past convergence.** `for _ in range(8): newton_step(...)` on a cavity
  that reaches `|R| ~ 1e-11` on its fourth step spends half its time iterating a converged root, and
  in a differentiability check it puts those extra steps on the reverse-mode tape too. Stop on a
  convergence test, or name the count a constant and say what it is (convergence plus a margin).
- **A dense matrix built one column per dispatch.** A Python loop calling `jax.jvp` on each basis
  vector pays eager dispatch `ndof` times; `jax.jacfwd` / `jax.vmap` push the whole basis through in
  one batched pass and build the identical matrix (worth 3.6x and 2.9x on the two that did this).
- **A deliberately-failing arm running out a large step cap.** A march that is *meant* not to
  converge costs its whole budget. Give both arms of such a comparison the SAME budget, set at a
  small multiple of what the succeeding arm needs — which is also the fairer test, since it stops the
  failing arm from being one that was simply given fewer steps.

**⚠️ A UNIT JOB CANCELLED AT ITS CAP LOOKS EXACTLY LIKE YOUR REGRESSION AND USUALLY IS NOT** — the
same trap the slow shards' duration balancing carries, one tier up. The unit job's wall clock is set
by run-to-run runner variance, not by the change under test or the interpreter: one `main` commit ran
py3.11 in 23:12 against py3.12's 21:55, and a later `main` commit ran py3.11 *faster* than py3.12
(18.7 vs 20.7 min). Before attributing a cancellation to your branch, check whether unrelated branches
are cancelling too, and compare the *other* interpreter on your own run — a branch whose py3.12 leg is
faster than main's did not slow the suite down. Two measured facts to save the re-derivation: the tier
costs 18.7-30 min end to end, and **preserving the JAX persistent compilation cache across CI runs
buys nothing** — the distributed `shard_map` tests (see below) compile 7184 XLA programs
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

None of this licenses weakening a check to make it quick. A finite-difference-validated adjoint costs
three solves and is the point of the project; that is a test spending what it must.

### Conditional coverage is DECLARED, not discovered (binding)

**A skip and a pass are the same exit status.** `tests/unit/test_optional_dependency_skips.py` holds
the census: every module gated behind an optional dependency, and every other file that can skip
anything, with what the skip is conditional on. Adding a gate without adding its entry fails the fast
gate — deliberately, because "this coverage is now conditional" is a decision to make and write down,
not one to arrive at by importing something.

Two consequences worth having in mind rather than rediscovering:

- **`petsc4py` is not installed in CI**, so the three modules gated on it — two of them in the *fast*
  tier, i.e. inside the required check — have never run there. It has no wheels (it builds PETSc from
  source), which is why the `petsc` extra is kept out of `test`; that is a cost decision, and the
  census is where its price is written down.
- **The workflow runs every tier with `-rs`**, so a skipped test is named with its reason in the job
  log rather than collapsing into a count nobody reads.

Marker typos are a related silent failure and are handled in `pyproject.toml`: `strict_markers` makes
an undeclared marker a collection **error**. The tiers are selected with `-m`, so a mistyped marker
does not fail — it moves a test to a tier that does not run it, or drops a heavy solve into the
always-on gate. Note the ini form is load-bearing: putting `--strict-markers` in `addopts` parses
cleanly and does **nothing**, which is the same silent-no-op shape the setting exists to prevent.

### Canonical tests (must always pass once implemented)

- **Primary field (analytical):** plane wall with convection vs the closed-form θ
  (Gate A).
- **Sensitivity (analytical) — the point of the project:** `jax.grad` of the converged
  solver w.r.t. `Bi` vs the closed-form ∂θ/∂Bi (Gate B). Every integration suite must
  include an explicit test that `grad` flows through `solve()` without error and without
  NaNs.
  ⚠️ **"Without NaNs" is the floor, not the test.** `0.0` is finite and is not a NaN, so an
  adjoint severed anywhere along the path — a stray `stop_gradient`, a closure that stopped
  carrying its dependence — passes a finiteness assertion and reports nothing. Demonstrated:
  wrapping `nut_wall`'s return in `stop_gradient` left all 28 tests in
  `tests/unit/test_turbulence_boundary.py` green, including the one named
  `..._is_differentiable_in_k`. **Compare the gradient against a finite difference wherever the
  cost allows it; where it does not, at minimum assert it is non-zero.**
- **AD-exact-linearization (skewed mesh):** on a non-orthogonal mesh, the linear problem
  converges in one Newton step (Gate C) — the concrete improvement over the reference.
- **Coupled-solve adjoint (iteration-count-independent):** for any iterative / coupled / fixed-point
  solver (the segregated flow–turbulence loop, and any future coupled system), `jax.grad` through the
  converged solve must match a reference gradient (finite difference or the closed form) **and be
  independent of the forward iteration count** — the coupling analogue of Gate C. This is what proves
  the adjoint is the implicit-function-theorem solve on the converged coupled residual, not the outer
  loop unrolled onto the tape. An existence/stability smoke test ("stays stable, fields positive")
  does **not** establish this — it must be its own explicit test.
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
- **x64:** `assert jax.config.x64_enabled`.

---

## Comment Convention

Code comments and docstrings are written **for a human reading this repository** and must
stand on their own. They must **never** point at anything a repo reader cannot open, or at
Claude-only / internal working files. Explain the code on its own terms (what the math/logic
is and why). Concretely, do **not** name any of these four things in a comment or docstring —
this is the exact leak that had to be scrubbed once, so treat it as binding:

1. **The precursor codebases** — the C++ framework and the Fortran solver. No "the C++
   `Face`", "mirrors the reference C++", "the original Fortran", "`coeff.F90`", "ported
   from …". They live outside this repository and are acknowledged **once in the README**
   (with links) and nowhere else in the tree.
2. **The Claude-facing files** — this root `CLAUDE.md` and anything under `.claude/` (the
   `.claude/rules/*.md` subsystem rules). These guide Claude Code, not human readers; a
   docstring must never say "See `.claude/rules/mesh.md`" or "per `CLAUDE.md`". They are
   tracked in the repository but remain Claude-facing, so the ban is broader than docstrings —
   see the standalone **Claude-Facing-File Reference Ban** below.
3. **The internal design notes** — the author's private working notes for this project (design
   records, briefings, milestone specs) kept **outside the repository**. A contributor does not
   have them, so a comment that says "see the preconditioner design note §4" points at nothing.
   Never cite one in code — inline the reasoning as prose. (Files under `docs/` **are** shipped
   and may be cross-referenced.)
4. **Self-citations to the author's own papers** — do not name the author's own prior work
   as provenance ("the DeGroot-2019 wall", "the DeGroot–Straatman flux"). State the
   *physics* instead ("the Green–Gauss accuracy ceiling on skewed grids", "a flux-continuous
   non-orthogonal diffusion flux").

**Why:** references to artifacts a reader cannot open (private code, Claude files, unshipped
notes, the author's own PDFs) rot immediately and confuse anyone outside this project.

**What you MAY do — cite real, published science properly.** Cite the *math* by name
("over-relaxed non-orthogonal diffusion correction", "divergence-theorem cell volume"). And
**standard third-party, eponymous citations are welcome** as proper author-year provenance —
"Ghia et al. (1982)", "the Venkatakrishnan (1993) limiter", "Rhie–Chow interpolation",
"Murphy–Golub–Wathen (2000)", "Patankar" — because a reader can look those up. The line is:
*published third-party science, cited properly = fine; pointers to this project's own
private / internal / Claude-only artifacts = never.*

> **Note for Claude specifically.** The `.claude/rules/*.md` files reference the C++/Fortran
> precursors freely — that provenance is exactly their job, and why they load into *your*
> context. What they must **not** do is point at the author's private design notes (a
> contributor does not have them): inline the fact instead. The boundary for shipped code is
> the **shipped surface** (`.py` files and `docs/`): what informs your understanding must not
> leak into a comment or docstring. When a rule file tells you a class "mirrors the C++
> `Face<T,3>`", the code comment must say "polygon centre-fan vector area", **not** "mirrors
> the C++".

## Claude-Facing-File Reference Ban

The **Claude-facing files** — this root `CLAUDE.md` and everything under `.claude/` (the
`.claude/rules/*.md` subsystem rules) — exist to guide Claude Code, not to be read by users of
the library. They are tracked in the repository (so a contributor can see the standards the code
is held to), but they are **agent instructions, not user documentation**. They must therefore
**never be referenced from any public-facing file** — not a `.py` comment or docstring, not the
`README`, and not anything under `docs/`.

Concretely, no public-facing file may say "see `CLAUDE.md`", "per the `.claude` rules",
"`.claude/rules/mesh.md`", "as the project briefing requires", or otherwise point a reader at
these files. State the underlying rule or fact on its own terms instead: not "naming follows
`CLAUDE.md`'s Spelling Convention" but simply *use* Canadian spelling; not "see
`.claude/rules/flow.md` for the coupling" but explain the coupling as prose (or cite a shipped
`docs/` page). This is the same boundary the Comment Convention draws for the precursor codebases
and the internal design notes, stated explicitly for the Claude-facing files themselves.

**Why:** a reader who follows a pointer into a Claude-facing file lands in material written for a
different audience — internal standards, provenance to private precursor code, our working
vocabulary — that neither reads as, nor is maintained as, user-facing prose. Keeping the shipped
surface (`.py`, `README`, `docs/`) free of these pointers is what lets the Claude-facing files
speak frankly to the agent without leaking into what users see.

## Docstring Convention

All public functions, classes, and methods use **NumPy docstring format** (Parameters /
Returns / Raises / Examples), with array shapes stated for every array argument.

## Self-Contained-Docstring Convention

Comments and docstrings must **stand on their own for a repo reader**, describing the code on
its own terms. This is the same spirit as the Comment Convention above (no pointers to things a
reader cannot open), extended to three leaks that came from *our working conversations* rather
than from the code — they read as authoritative but a reader has no way to check or interpret
them, so they rot. All three are **binding**:

1. **Define every acronym / non-obvious term at first use in a file.** Spell it out once, then
   use the short form: "compressed-sparse-row (CSR) form — a row-pointer array plus a flat index
   array", then "CSR" thereafter. Applies per file (a reader may open just one). Standard, widely
   known math names cited author-year are exempt (the Comment Convention already covers these).
2. **No internal-staging or roadmap labels.** Never name our build stages or planning artifacts
   in shipped code — no "Milestone 0", "Gate B", "Phase 2", "the first build target", "not yet
   exercised in <stage>". These are scaffolding for *our* sequencing, invisible and meaningless
   to a repo reader. State the property on its own terms instead: not "not exercised in Milestone
   0" but "node_coords is a differentiable leaf, so gradients w.r.t. node positions flow through
   it". (A genuine, self-explanatory *code* state like "the first timestep uses BDF1" is fine —
   it describes the algorithm, not our roadmap.)
3. **No hard performance numbers invented in conversation.** Do not assert throughput/size
   figures that came from a chat and that no test pins — "million-cell meshes", "builds in
   seconds", "handles N cells". Describe the *mechanism* that gives the property instead: "avoids
   the per-face Python loop, which is the bottleneck for large meshes". A quantitative figure is
   fine only when it is a genuine, reproducible property of the algorithm that a reader could
   re-measure (e.g. "on a model Poisson the V-cycle contraction is ~0.25") — that is checkable
   science, not a remembered benchmark.
4. **No design-principle labels or their names.** The Engineering Principles above are *our*
   working vocabulary for how we build; a repo reader neither has them nor needs them. Never write
   "single source of truth", "one source of truth", "the DRY consolidation point", "one canonical
   implementation", "no duplicated physics", "the unit of reuse", "reuse over reimplementation",
   or "(Principle N)" / "(CLAUDE …)" in shipped code. State the *code fact* that the principle
   produced — which is genuinely useful navigation — without the slogan: not "the diffusion flux,
   one source of truth shared by the serial and distributed paths" but "the diffusion flux, shared
   by the serial and distributed paths"; not "reuses X's terms/operator/rhs (one source of truth)"
   but "reuses X's terms/operator/rhs". Keep "reuses / shared by / delegates to <name>" (it tells
   the reader where the real code lives); drop the editorial tag.

**Why:** a docstring is read by someone who was not in our conversation and cannot see our notes,
stages, or principles. Undefined jargon, roadmap labels, unpinned numbers, and design-principle
slogans all assume context the reader does not have — and the last one adds nothing the code fact
does not already say.

## Spelling Convention

Use **Canadian spelling** by default — in identifiers, comments, docstrings, and docs:
double the final consonant (`labelled`, `modelled`, `travelled`, `cancelled`), `-our`
(`colour`, `neighbour`, `behaviour`), `-re` (`centre`, `metre`, `fibre`). **Keep `-ize`**
(`normalize`, `organize`) and `analyze` — these are standard Canadian, not Americanisms.
So: `LabelledGroups`, `cell-centred`, `neighbour` (already used throughout).

---

## Development workflow

> The workflow mirrors aquakin: branch → PR → green lint gate → merge; never commit on
> `main`; commit/push only when the user asks.

### Use the blessed command; do not hand-roll it (binding)

Four operations have one correct invocation. Use it.

| to do this | run this |
|---|---|
| run a validation case or any long solve | `validation/run_case.sh <script.py> [--wait]` |
| see what a long run is doing | `validation/run_case.sh --status` · `tail -f <its log>` |
| run a test tier | `tools/fastgate.sh [fast \| slow \| validation \| all]` |
| build the Sphinx docs locally, the same strict way CI does | `tools/build_docs.sh` |

Each exists because the hand-rolled version fails *silently* — it produces a plausible answer that is
wrong and says nothing about it. `pytest … | tail -n` reports the exit status of `tail`, which is `0`
whatever pytest did. A case launched with a bare `python … &` dies with its parent, or outlives a
cancellation you believe succeeded. `pgrep -f <script>` matches the watcher's own command line, so a
waiter waits on itself forever. A hand-rolled `sphinx-build` that skips deleting the gitignored
`docs/generated`/`docs/api.md`/`docs/_build` first can report a clean build that is clean about the
*previous* run's set of pages. Every one of those has happened here and cost hours.

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

### What no runner can enforce (binding)

- **Print one line per outer step, `flush=True`.** The runner keeps the log unbuffered; only the script
  can make it worth reading. A log that grows per step lets a bad trajectory be killed at minute three
  instead of at the end.
- **Actually watch it.** A run nobody reads until it finishes is a run that cost its full wall time to
  tell you something it knew in the first minute.
- **A run that spanned a machine sleep is void for cost.** It keeps converging correctly and nothing in
  the log looks wrong, but the wall-clock column has silently absorbed the sleep. `run_case.sh` holds
  the machine awake; if one happened anyway, keep its step and cycle counts and discard its timings.
- **Sweep in one process, not N.** These jobs are memory-bound — a materialized 3D coupled Jacobian is
  ~2 GB per copy — so spare cores are not a reason to parallelize. Loop over β in one process, `del` the
  big arrays between iterations, and load a cached factor from disk rather than re-materializing it.
  Concurrent probes have exhausted this machine and suspended every application on it, including the
  session driving them, which cannot be debugged from inside.
- **Before asserting what a background job is or did, read its own record** — the run-file, the task
  output file, the log. A start time and a parent pid are circumstantial; a run whose gate logged
  `HEALTHY -- launching march` is not.

### Measure the quantity you will be judged by (binding)

**The number a run is steered/watched by and the number it is finally judged by must come from one
definition.** The observed failure: a validation case computed its reattachment length two ways — the
final comparison used a mid-span slab, while the live progress metric used the full span, which in a
3D geometry is set by a *side-wall corner separation* rather than the primary feature. Same name, two
different physical quantities, ~40% apart. An entire run's worth of reasoning was done against the
wrong one before anyone noticed.

So: when a case exposes a scalar for a solver to be watched by, it must be **the same callable** the
final report uses — not a convenient approximation of it. If a diagnostic variant is genuinely wanted,
report it under a **different name**, alongside.

### Record what a measurement was taken under (binding)

**A measured finding written into these files must name the configuration it was measured with — the
defaults that were active, the state, and the operating point. A number without its configuration
expires silently the moment any of them changes, and you cannot tell that it has.**

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

Concretely, when you write a measurement down:

- **Name the defaults in force.** "21 iterations" is worthless; "21 iterations at ILU(1), 2 sweeps,
  smoothed aggregation, 2 levels" survives a default change because a reader can see it no longer applies.
- **Name the state and the operating point** — which case, which step or checkpoint, which shift, and
  whether the state was converged or mid-march. The same arm can measure 6 cycles at one and 22 at another.
- **Say what an inference does NOT distinguish.** The Vanka bullet's reasoning is valid but
  under-determined between two different meanings of "the coarse space", and nothing recorded that, so it
  was read as the expensive one for months.
- **When a default changes, grep the rules for findings measured under the old one** and mark them, in the
  same change. That is part of the Post-Change Checklist's Documentation-sync item, not a follow-up.

### Start from an up-to-date main (do this FIRST — binding)

**Before creating a feature branch, putting any change on it, or running the test suite,
refresh `main` from the remote so you are not working against a stale base.** Branching from
a stale `main` is the observed failure mode this rule guards against: the branch diverges from
what has already merged, tests pass or fail against outdated code, and the eventual PR carries
avoidable conflicts and re-litigates work that is already in.

Concretely, at the **start** of every task — before the first branch or the first test run:

1. `git fetch origin` (or the configured remote) to pull the latest refs.
2. Compare your base against the remote: `git log --oneline HEAD..origin/main` (or
   `git rev-list --count HEAD..origin/main`). If it is non-empty, your base is stale.
3. Bring your working base up to date before branching — update local `main` to
   `origin/main` and branch from it, or rebase an existing feature branch onto the freshened
   `origin/main`. In a worktree, sync the branch you are on against `origin/main` the same way.
4. **Only then** create/switch to the feature branch, make changes, and run tests — so the
   suite runs against current code, not a stale snapshot.

If the base was stale and you have already started, stop and rebase onto the freshened
`origin/main` before continuing. When a sync would pull in changes that could reasonably
affect the task (a touched subsystem moved under you), surface that to the user rather than
silently rebasing over it.

CI runs a ruff gate on every pull request (and on pushes to `main`) via GitHub Actions
(`.github/workflows/ci.yml`): `ruff check` + `ruff format --check` on `aquaflux`, `tests` and
`docs` — the last because `docs/conf.py` generates the API reference and is real code that
nothing else checks — with ruff pinned by the `lint` extra so the gate cannot move under a new
release. (`codespell` stays on `aquaflux tests`; the docs prose is not yet spell-gated.) The same
gate is available locally through the committed pre-push hook (`.githooks/pre-push`) — enable
it once per clone with `git config core.hooksPath .githooks`, and it runs the identical two
commands before every push, so a slip is caught locally instead of as a red check on the PR.

The same `core.hooksPath` setting also enables the committed **pre-commit** hook
(`.githooks/pre-commit`): when a commit touches `.py` code it prints a **non-blocking** reminder
to update the docs that describe that code (the Post-Change Checklist's **Documentation sync**
item) — the guard against the `.claude/rules/`, `CLAUDE.md`, `README`, and `docs/` drifting out
of step with the code. It never blocks a commit (doc-sync is a judgement a script cannot make);
bypass its output with `git commit --no-verify`.

**Both hooks fail silent when `core.hooksPath` does not resolve to the hooks**, which is the state
worth knowing about: git runs no hook and says nothing, so the gate is simply gone while everyone
assumes it is there — and the first sign is a red required check on a PR. Prefer the **relative**
`.githooks`: git resolves a relative `core.hooksPath` from the top level of the working tree, so it
follows every checkout, including every worktree, to its own copy. An **absolute** path instead pins
every worktree to one named directory, whose contents depend on whichever branch *that* checkout is
sitting on — the hooks then appear and disappear according to unrelated state. Because the setting
can also be overridden **per worktree** (`git config --worktree`, active when
`extensions.worktreeConfig` is set), a repository-level value that reads correctly can still be
shadowed; `git config --show-origin --get core.hooksPath` names the file actually in force, and
`git config --worktree --unset core.hooksPath` drops a worktree-level override.

`tools/check_hooks.sh` reports both failures, and `tools/fastgate.sh` runs it before it runs
anything, so the state surfaces on an ordinary test run. It warns when the hooks **will not run**
(nothing configured, or a path holding no executable `pre-push`) and, separately, when they run
only **by coincidence** — an absolute path resolving outside this checkout, which works today and
disappears without a word the moment that other checkout changes branch. It stays silent when the
wiring is sound and under `CI`, where the workflow runs the gate directly, and it always exits `0`,
so a caller's status is its own. That is the only warning you get: the hooks cannot report their
own absence, which is why the checker is covered by `tests/unit/test_check_hooks.py` — a warning
that quietly stops firing looks exactly like a repository whose hooks are fine.

### Documentation

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

The docs are the **shipped surface**, so the Comment Convention and the Claude-Facing-File
Reference Ban apply to them too: no references to the precursor codebases, the Claude-facing
files (`CLAUDE.md` / `.claude/`), the internal design notes, or the author's own papers, and no
roadmap/build-stage labels ("Milestone 0", "BUILT") — write for a user who only has the
published site. Internal repo references that are *not* user docs (the
annotated `package_structure.md` file tree) stay in `docs/` but are listed in `conf.py`
`exclude_patterns` so they never reach the built site.

---

## Module Review Rubric

When asked to review a module (or auditing one before commit), run **two passes**. The second is
the one that gets skipped, and it is the one that catches structural problems a per-file checklist
is blind to — a shared type living in the wrong module, a data bundle that unions every consumer's
needs, a decomposition that won't survive the next feature. A checklist review pattern-matches
*local* smells; structural review requires stepping back from the code-as-written and questioning
the structure, which needs **fresh eyes** (see "How to run it").

### Pass 1 — Local smells (per file)
The Engineering Principles as a checklist, plus correctness and user-facing clarity:
- **Testability (Principle 1):** can each unit be tested in isolation with small inputs, no global
  state? Is there an operator-level test (order-of-accuracy on an analytic field)? And can each
  test actually fail — has it been checked against a broken version of the code, not just read
  (Testing Architecture → "A test that cannot fail is not a test")?
- **Duplication (Principle 2):** logic implemented more than once; copy-paste-modify; a formula
  that should be imported from its one home.
- **Encapsulation (Principle 3):** passing an object's raw arrays instead of the object; taking a
  whole `Mesh` where `face_cells` suffices; duplicate accessors; god-methods; inlined formulas
  that have a home.
- **Correctness & clarity:** errors/edge cases; anything a user would find hard to understand or
  misuse.

### Pass 2 — Decomposition & extensibility (step back; question the structure)
Do not evaluate the code *within* the existing structure — question the structure itself:
- **Placement / cohesion.** Is each type/function in the right module? Does each module have one
  responsibility? *Smell:* a type used mainly by module B but living in module A because that is
  where it was first written (a shared interface/bundle that accreted into the first concrete
  implementation — e.g. a face-flux state/interface living in `diffusion.py`). **Sharpest form: generic
  machinery in a physics package** — a driver, step assembly, measure or settings value whose description
  names no physics, living in `turbulence/` because turbulence was first (Principle 3.6; this left a
  laminar solve unable to reach the robust march).
- **God-objects / union bundles.** Any data structure that is the union of every consumer's needs —
  so unrelated consumers are coupled, it grows with each new operator, and it needs placeholder
  defaults (`psi = ones`) for the fields a given consumer does not use? Any "context/state" object
  carrying fields only one consumer reads?
- **Dependency direction.** Do sibling modules import each other for shared types (they should share
  a base/contract module instead)? Any import that invites a cycle?
- **Extensibility against the stated direction.** As the system grows toward its known future
  (YAML/DSL-driven assembly; N operators / schemes / BCs; a properties model), what will bloat, need
  placeholder defaults, or need re-cutting? Is each strategy **self-describing about its own inputs**
  so a declarative assembler can gather per active term?
- **Self-justification.** For each shared type, state *why* it lives where it does. If the honest
  answer is "that is where it was first written," it is probably misplaced.

### How to run it
Run **both** passes. Pass 2 needs **fresh eyes**: spawn a reviewer (agent) that has *not* been
anchored on the current structure, and prompt it with these Pass-2 dimensions **explicitly** — a
generic "find issues" prompt reverts to local-smell matching and misses them. Scope each review to a
named module (or a small set), evaluate against the future direction (not just the code as-is), and
**verify each finding yourself** before acting (agents surface plausible-but-wrong structural claims
too). A finding is only real if you can name the concrete failure it causes now or the concrete
bloat/re-cut it forces later.

---

## Stale-Record Check (binding — run before EVERY commit, BEFORE the Post-Change Checklist)

**Ask what your change makes FALSE, not only what it leaves incomplete.** The Post-Change Checklist's
documentation-sync item catches the file that *describes* the code you touched. This catches the far
larger class: every entry anywhere in the Claude-facing files that your change silently invalidates —
a symbol you renamed, a default you moved, a measurement taken under the configuration you just
changed. Those entries do not announce themselves, and nothing in the test suite fails when they rot.

**This is not hypothetical bookkeeping. It is the most expensive recurring defect in this project.**
In one session, three separate wrong facts were lifted out of these files by grep and asserted as
current — a march solver that had been replaced, a tolerance that had moved, and a preconditioning
side that had been deliberately reversed. Two of them were passed into sub-agent briefs, which would
have produced confidently wrong measurements. A later audit found a **binding decision** stating
`NewtonSolver` was deleted and, 34 lines below it, another entry using `NewtonSolver` as live.

**Run this before every commit that renames, deletes, moves, or re-values anything:**

```
git diff --cached | grep -E "^-" | grep -oE "\b_?[A-Za-z][A-Za-z0-9_]{3,}\b" \
  | grep -E "_|[a-z][A-Z]" | sort -u > /tmp/touched
grep -rnFf /tmp/touched CLAUDE.md .claude/rules/ README.md docs/
```

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

Then, for every hit, decide: still true / update / **delete**. Concretely:

1. **A renamed or deleted symbol is a defect wherever it still appears.** Grep every Claude-facing
   file for the old name, not just the rule that owns the subsystem — these names spread.
2. **A moved default orphans every measurement taken under it.** Such a number is not merely
   out of date, it is *unfalsifiable*: a reader cannot tell it no longer applies. Mark it with the
   configuration it was taken under, or delete it. (See the Development-workflow rule on recording
   what a measurement was taken under.)
3. **⚠️ SUPERSEDE BY DELETING. Never by striking through, and never by appending a correction
   below the claim.** `~~strikethrough~~` is invisible to grep, and "SUPERSEDED — see below"
   expresses supersession by *adjacency*, which a grep hit does not carry. Both leave the wrong
   statement fully readable and fully greppable. If the dead finding taught a trap worth keeping,
   collapse it to **one line stating the trap** and delete the body.
4. **Prefer a pointer that inoculates.** Where a dead name is likely to be searched for, one line
   saying "there is no such symbol; the real one is X" is worth more than silence — a grep then lands
   on the correction instead of on nothing.
5. **Never cite a path that is not in the repository.** A `scratchpad/` or private-note pointer makes
   its finding permanently un-re-adjudicable; move the harness into `validation/` or drop the pointer.

**Why the rules files specifically:** they auto-load into context and are searched by name, so a stale
entry there is not a passive error — it is actively served to the next reader as current fact.

## Post-Change Checklist

After **every code change**, before considering the task complete, review and act on:

1. **Engineering Principles review (the priority gate for this project).**
   - **Testability (Principle 1):** can every function you added be unit-tested in
     isolation, with small explicit inputs, no global state? If not, fix the seam
     *now*.
   - **Duplication (Principle 2):** did you introduce logic that already exists, or
     copy-paste-modify a block? If so, consolidate to one source of truth *now*.
   - **Encapsulation (Principle 3):** did you pass an object's raw arrays instead of the object,
     take a whole `Mesh` where `face_cells` would do, add a forwarding property or a second
     spelling of one value, duck-type a lookalike of a real class, inline a formula/scatter that
     has a home, or grow a `# step N` god-method? Fix the seam *now* — reach for the object/helper.
   - **Layering (Principle 3.6):** did you put code in a physics package whose one-sentence description
     needs no physics word (a driver, a step assembly, a measure, a settings value)? Move it to `solve/`
     now, and check every residual can reach the capability you added.
   - **Maintainability (Principle 0):** if you took a quick-to-ship shortcut as an
     intermediate step, refactor it before marking the task done.
   - **Scope (Principle 3.5):** did you build what was actually agreed? If you narrowed the design,
     substituted a cheaper approximation, or finished only part of it, say so **in the summary** —
     and never report a measurement taken under that deviation without naming it.
   - **Solver adjoint & globalization (for any new iterative / coupled / fixed-point solver).**
     Before calling such a solver done, confirm all three: (a) **the adjoint is an
     implicit-function-theorem solve on the converged residual, not the iteration unrolled onto
     the tape** — verify it is a single transpose solve, not a taped loop (`.claude/rules/solve.md`);
     (b) **the forward solve stops on a convergence test**, not a hard-coded iteration count (a fixed
     count is allowed *only* as an explicitly-labelled intermediate); (c) **it is globalized** to the
     standard of its neighbours (continuation / line search), with constant under-relaxation + floors
     treated as a stabilizer, not the globalization. If you ship an intermediate that does not yet
     meet (a)–(c), **file the deferred work as a tracked issue in the same change** — an unlabelled,
     untracked prototype delivered as done is the exact Principle-0 failure this gate guards against.

2. **Lint, format & comment hygiene** — from the repo root:
   - `ruff check aquaflux tests docs` — must report no errors.
   - `ruff format aquaflux tests docs` — auto-applies formatting (CI will run `--check`).
   - **Comment-hygiene guard (the Comment Convention + Claude-Facing-File Reference Ban):** the
     shipped surface must not point at the precursor codebases, the Claude-facing files
     (`CLAUDE.md` / `.claude/`), the internal design notes, or the author's own papers. This grep
     must come back **empty** for any `.py` you touched:
     ```
     grep -rniE "c\+\+|fortran|\.claude|claude\.md|reference code|the reference|degroot|\.hpp|\.f90|design.note|briefing\.md|MeshObjectGroup" aquaflux tests --include="*.py"
     ```
     (`design.note` rather than `-design-note`: the hyphenated form missed "the design note (S5)",
     which sat in two shipped docstrings for months. `the reference` is deliberately noisy — most hits
     are the legitimate "the reference state/operator/diagonal" — so read them, don't skim past them;
     that is how "matching the reference's order" survived.)
     (Third-party author-year citations like "Ghia et al. (1982)" or "Venkatakrishnan (1993)"
     are fine and won't trip this pattern.)
   - **README / docs surface:** the same reference ban covers the other public-facing files.
     When you touch `README.md` or anything under `docs/`, this grep must also come back
     **empty** (the `package_structure.md` file tree is excluded — it is a build-excluded
     internal catalogue of the repo layout, not user-facing prose, per the Documentation
     section):
     ```
     grep -rniE "\.claude|claude\.md" README.md docs --exclude=package_structure.md
     ```
   - **Sibling-builder check (Principle 2's "two builders for one object").** Two functions that
     construct the same class from mostly the same parameters are one builder written twice, and the
     copies drift one keyword at a time — which no diff-scoped check can see, because no single commit
     looks wrong. Run this whenever you add a parameter to a builder or add a builder:
     ```
     tools/sibling_builders.py
     ```
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
     `AmgVCycle._build` constructs — and a receiver whose producer returns no package class **falls back to
     the bare name**, because the first version lost `_coupled_step`'s `globalization.step(...)` (a rebound
     parameter) and silently re-hid the pair it was written to find. The one new pair is `coupled_step` /
     `mass_flow_coupled_continuation`; every other pair is main's, unchanged. Its two carve-outs have
     since both gone — `residual_norm` with the measure's move onto the solve (#370), `flow_direction`
     with the drive's (#375, 2026-09-23) — so the pair now reports **13 shared and `only here: []` on
     both sides**: two identical surfaces over one tail, which is what this rule is for.
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
     bare-name fallback (the package test, which asserts `coupled_step` pairs with
     `mass_flow_coupled_continuation`).
     ⚠️ **Still blind, and recorded so its silence is not read as clean:** a builder that returns a
     *closure* it defines rather than a call — `scalar_pseudo_transient_solve` returns `solve_scalar` — is
     credited with building nothing and never enters the report.

   Ruff is pinned via the `lint` extra (`pip install -e ".[lint]"`). Not needed for
   docs/config-only changes touching no `.py` files.

3. **Tests** — are new tests needed?
   - New numerical operator → operator-level unit test (order of accuracy).
   - New public API → integration test against an analytical solution.
   - New analytical/published benchmark → validation test.
   - Bug fix → regression test.
   - **Every new test: verify it can fail before counting it, per Testing Architecture → "A test
     that cannot fail is not a test."** Break the line of code it is meant to pin, confirm the test
     goes red, restore the line. A test you have not done this to is unverified, whatever it reads
     like.

   **Run the tier your change can reach — the fast gate is not the whole suite.** The `slow` and
   `validation` tiers run on **merge to main**, and on a PR only when it carries the `full-ci`
   label; the always-on required check is just the fast gate (`-m "not slow and not validation"`).
   So a change whose *only* coverage lives in those tiers can pass every required check and still
   break on merge. Before calling a change done, ask **whether it could affect a `slow` or
   `validation` test** — you are touching a shared solver / operator / scheme / helper those tests
   call, deleting or renaming a symbol, or changing convergence/behaviour — and if it could, **run
   those tiers locally** (`tools/fastgate.sh validation`, `tools/fastgate.sh slow`) or apply
   `full-ci` to the PR.
   The trap is a migration reached only through a validation-marked test (e.g. a case whose sole
   test is `@pytest.mark.validation`): the fast gate exercises the *mechanism* elsewhere but never
   that call path, so grep for the changed symbol across `-m slow`/`-m validation` tests and run the
   ones that hit it. Don't assume "unit + fast integration green" means safe to merge.

   **⚠️ The slow/validation shards are balanced by `.test_durations`, and a stale one silently
   unbalances them.** Those tiers are heterogeneous (a 21 s scheme check beside a 242 s adjoint
   continuation), so `pytest-split` partitions them by recorded duration. With **no** durations it
   splits evenly by **count**, and then adding a test *anywhere* shifts the boundaries and can
   migrate the expensive tests onto whichever shard is already heaviest — which is not hypothetical:
   eight unrelated unit tests moved two ~106 s adjoint tests off the lightest slow shard (15:53) onto
   the heaviest (20:58), which then needed ~24.5 min of tests plus ~5 min of setup and was **killed
   at the 30-minute ceiling**. The failure looks exactly like a regression in the change under test
   and is not one. A weekly cron (plus `workflow_dispatch`) re-records the file and opens a
   metadata-only PR; label a PR **`refresh-durations`** to record it on that branch instead, which is
   the only way to populate it before the workflow has merged (`workflow_dispatch` is registered from
   the default branch only). The fast integration tier is deliberately **not** duration-balanced —
   those tests are homogeneous, and an even-by-count split balances their memory too.

4. **Documentation sync (binding — this is how the docs stop drifting).** A code change is
   **not complete** until every file that *describes* the changed code is updated in the **same
   change** — docs move with code, never "fix it later." **Run the Stale-Record Check above first:**
   this item covers the file that *describes* what you changed, that one covers everything your
   change silently made FALSE elsewhere, which is the larger and more dangerous set. When you rename
   a symbol, change a signature/default, move a file, add a dependency, or change behaviour, update
   each of these that applies (grep the repo for the old name/path to find every mention):
   - **The docstrings and comments in the `.py` files** — both in the code you changed *and in
     every other module that describes the same behaviour*. In-code prose sits closest to the
     change and is the first thing a reader trusts, so a false claim there is the most expensive
     kind; it is also the only item on this list that no other item covers. The trap is the
     second half: one component's behaviour is typically restated in several modules, and a
     change updates only the file it edits. **Grep for the claim, not only for the symbol** —
     this class of drift renames nothing, so the Stale-Record Check's identifier grep is blind
     to it by construction. (Commit `e6bc18e` switched the linear solve from left- to
     right-preconditioning in `solve/linear.py` *alone*. `solve/multigrid.py`,
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
   - **`CLAUDE.md`** — architecture decision, public-API or package-structure change, new
     dependency, or workflow/tooling change.
   - **`README.md`** — public API, install/dependencies, examples, or the feature list.
   - **`docs/`** — any Sphinx page whose prose or cross-references the change touches.

   Rule of thumb: if a reader of one of these files would now be **misled** by what it says,
   fixing it is part of your change, not a follow-up. (The `MaterialModel`→`PropertyModel`,
   `structured_grid_2d(perturb=)`, and "git/CI not set up" drifts were all exactly this gap.)
   The committed **`.githooks/pre-commit`** reminder surfaces this whenever a commit touches
   `.py` code.

5. **CHANGELOG.md** — once one exists (add at first release-worthy change): user-visible
   API/behaviour changes only, under `[Unreleased]`.

If the answer to any of the above is yes, make those updates as part of the same task
before marking it complete.
