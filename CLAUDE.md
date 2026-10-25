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

⚠️ **`paths:` is mandatory on every file under `.claude/rules/`: an absent one scopes the file to
everything, not to nothing**, and nothing reports what it costs (why: "Preamble: the unscoped-files incident").
**Reference-only material goes in `.claude/notes/`** (tracked, greppable, read on demand). `tools/check_rules.sh` reports any rules file
missing `paths:`, and `tests/unit/test_check_rules.py` runs it in the fast tier.

**A `(why: "<heading>")` pointer in this file names a heading in `.claude/notes/claude-md-log.md`**, which
holds verbatim the incident or measurement behind the rule beside it.

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
| `.claude/rules/solve.md` | `aquaflux/solve/**`, `aquaflux/precondition/**` | **the residual-agnostic layer every residual runs on** (Principle 3.6): the staged driver, shifted-step assembly, measures, Newton, the linear solve with implicit differentiation; and `precondition/`, the preconditioners those solves take (imports `solve/` and nothing else). Split into narrower-scoped `solve-*.md` siblings — see `solve.md`'s own index |
| `.claude/rules/flow.md` | `aquaflux/flow/**` | coupled p–U block: momentum (reusing advection/diffusion) + Rhie–Chow continuity, differentiated `a_P` (frozen only in the preconditioner), monolithic AD-Jacobian solve |
| `.claude/rules/turbulence.md` | `aquaflux/turbulence/**` | k–ω SST closure + the segregated flow–turbulence loop: segregated forward / coupled adjoint, outer-loop globalization, positivity-floor adjoint honesty |
| `.claude/rules/transport.md` | `aquaflux/transport/**` | scalar transport by a converged flow (species, temperature, tracers): why a concentration rides the *volumetric* flux, the effective-diffusivity convention, sub-patch injection without a mesh change — the aquakin reaction seam |
| `.claude/rules/io.md` | `aquaflux/io/**` | mesh import (`MeshReader`, the OpenFOAM polyMesh reader, empty-patch 2D collapse), CAD import (STEP through OCP into exact `solids` bodies; the only kernel import is `io/cad/kernel.py`), and field export (OpenFOAM fields that restart, VTK `.vtu`/`.pvd`, the `patches.vtm` boundary-patch layout the browser viewer depends on) |
| `.claude/rules/validation.md` | `validation/**` | the scientific cases and study harnesses: why no test tier drives them, the obligation to check an API change against them, what the static guard catches and what it cannot, and the harness traps that have produced wrong results |
| `.claude/rules/parallel.md` | `aquaflux/parallel/**` | distributed memory: graph partitioners, the `PartitionedMesh` owned+halo decomposition, uniform-shape padding, and the `shard_map` residual that runs an *injected* assembler per device (never a re-implementation) |
| `.claude/rules/radiation.md` | `aquaflux/radiation/**` | ultraviolet fluence rate `G` by a deterministic backward gather over surface facets: the solid-angle kernels and why they are not interchangeable, the load-bearing numerical details, the `F_ii = 0` convention; the scene (`solve_scene`) and transparent solids (refraction) |
| `.claude/rules/solids.md` | `aquaflux/solids/**` | solid bodies answered by formula rather than search: the `Body` contract, analytic primitives, constructive solid geometry, `Outside`; generic geometry, held to the same import-nothing rule as `solve/` |
| `.claude/rules/octree.md` | `aquaflux/octree/**`, `tests/unit/test_octree.py` | the octree (one lattice of root cubes, leaves as level + lower corner) shared by the mesher's hex core and the adaptive sampler; the sampler's hierarchical interpolant, cut points, ghosts and probes; generic, imports nothing |
| `.claude/rules/meshing.md` | `aquaflux/meshing/**` | mesh generation from STEP (phase 1 built: STEP → Netgen tets → dual, a mesh file → `.aqmesh`): one conformal hybrid primal mesh, then one feature-preserving dual; Netgen only in a child process |
| `.claude/rules/ui.md` | `aquaflux_ui/**`, `tests/ui/**` | the browser interface `aquaflux-ui` (`ui` extra; Mesh / Setup / Run / Results): the binding independence rule (imports neither `aquaflux` nor `jax`, and the solver never imports it — tested), the `ResultSource` seam, the render-free `Pipeline` and its large-mesh policy, the trame page |
| `.claude/rules/case.md` | `aquaflux/case/**`, `aquaflux/__main__.py`, `validation/*/case.yaml`, `validation/*/cases/*.yaml` | a whole case in one YAML file: the `CaseSpec` core, physics and drive, one physical kind per boundary patch, the fluid stated once, the YAML 1.2 parse, the build, the solver section, `aquaflux run` and its run records, the `initial` section (`Checkpoint`, `Fields`) |
| `.claude/rules/docs.md` | `docs/**`, `tools/build_docs.sh`, `.readthedocs.yaml` | the Sphinx site: the API page generated from each documented subpackage's `__all__`, `SUBPACKAGE_GROUPS`, the docstring layouts that break a `-W` build, extension dependencies, the shipped-surface rules for `docs/` |

**Reference-only notes (`.claude/notes/`, tracked, never auto-load — read deliberately before re-investigating a subsystem):**

| Note | Holds the dated record behind |
|---|---|
| `.claude/notes/flow-log.md` | `flow.md` |
| `.claude/notes/schemes-log.md`, `.claude/notes/schemes-435-tetrahedral-gradient-log.md` | `schemes.md` |
| `.claude/notes/turbulence-log.md` | `turbulence.md` |
| `.claude/notes/radiation-log.md` | `radiation.md` |
| `.claude/notes/solve-log.md`, `solve-direct-preconditioners-log.md`, `solve-amg-multigrid-log.md`, `solve-flow-block-log.md`, `solve-globalization-log.md`, `solve-march-log.md` | the matching `solve*.md` rule (see `solve.md`'s index) |
| `.claude/notes/solve-refuted-directions.md` | the cross-cutting ledger of closed/refuted directions — grep it before proposing one |
| `.claude/notes/solve-open-directions.md` | the backlog of unmeasured solver speed-up directions |
| `.claude/notes/claude-md-log.md` | this file's rules (the incidents and measurements behind them) |
| `.claude/notes/module-review-rubric.md` | not a log: the two-pass Module Review Rubric, read when a review is asked for |
| `.claude/notes/{case,validation,io,mesh,meshing,octree,discretization,parallel,ui,solids,boundary,solve-field-split}-log.md` | the matching rule |

### Writing the design records (binding)

Every rules file is paid for in context by each session that touches its paths, so it holds only what a
future agent needs to act correctly. `tools/check_rules.sh` (fast tier) fails any rules file over
**120 KB**, any set of rules that load together over **200 KB**, any rule whose globs match nothing, and
this file over its own ceiling; `tools/rules_budget.py` holds the budgets and the shrink-only lists.

**This file loads into every session, so it may not regrow (binding).** It holds only what a task that
touches none of the subsystems still needs: the binding rules, their concrete triggers, and the commands
and paths the checklists run. Before adding a line, ask whether a session working in an unrelated
package needs it; if it is about one subsystem, it goes in the rule whose `paths:` covers that
subsystem. An incident, measurement or example goes in `.claude/notes/claude-md-log.md`, leaving the
rule and a `(why: "<heading>")` pointer; a new entry is a rule plus at most one sentence of reason.
`ROOT_CEILING` in `tools/rules_budget.py` caps this file a little above its size and must be lowered as
the file shrinks. **When an addition would cross it, move something out in the same change — never
raise the ceiling** without the project owner's agreement.

**Where it goes**
- **Rules file**: current state only — what is built and where, contracts and invariants, the shipped
  default and why, binding decisions, traps that still apply to live code.
- **`.claude/notes/<name>-log.md`**: anything dated — measurements, probe and A/B tables, investigation
  steps, issue narratives, mutation-pass lists. Leave one current-status line in the rule pointing at it.
- **`.claude/notes/solve-refuted-directions.md`**: one paragraph per closed direction (what, on what
  case, why it lost, where the detail is), added in the same change that closes it.
- **Nowhere**: history of deleted code (git has it; keep at most one line per trap that transfers to live
  code), anything the code or its docstring already says, anything only true of a past commit.

**How to write an entry**
- Lead with the fact or the rule, add the one-sentence reason, stop. No account of how it was found.
- A number only when acting on it needs it, with its configuration (case, state, defaults in force, date).
- Name a test only to locate the pin, not as a list of what was checked.
- When a fact changes, edit the entry in place; a superseded entry is deleted, never struck through or
  followed by a correction. State each fact once and point to it elsewhere.
- Route by the `paths:` glob that covers the file you changed, not by subject.
- Before adding to a file near its budget, move its dated content to the log first.

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
   (*Use the blessed command*). One case at a time — these are memory-bound. Before a push from a
   small container, `tools/fastgate.sh --changed` runs the tests the change can reach; CI still runs
   the whole gate (*Scoped runs*).
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

> **Read this section as binding, not aspirational.** Each principle addresses a specific, observed
> failure mode and takes precedence over shipping something fast. If a principle and a deadline
> conflict, the principle wins — no deadline on this project outranks code quality.

### 0. Maintainability over speed (the overarching preference)

**This project prefers spending more time to produce maintainable, well-tested, non-duplicated code
over returning something quickly.** Do not optimize for a fast first answer or the smallest diff.

- When two implementations are possible — one quick to write, one better structured — **choose the
  better-structured one and spend the extra time.**
- Refactoring code you are touching so the change lands clean is in scope. Leave each file better than
  you found it (within the change's blast radius).
- "This is good enough to ship" is the instinct this project guards against; stop and take the time.
- A correct-but-ugly prototype is acceptable *only* as an explicitly-labelled intermediate step that you
  refactor before the task is done — never as the delivered result.
- **Delete dominated methods while the code is pre-release.** When a method, strategy, or code path is
  *dominated* — another does its job better across the regime this project targets, and nothing selects
  it in production or integration tests — **delete** it rather than keep it "just in case": unused
  alternatives carry maintenance cost, bloat the API, and force awkward abstractions. A method with a
  genuine, exercised niche stays. Git history makes a deleted method cheap to restore. (When in doubt
  about whether a niche is real, surface the trade-off rather than silently keeping the cruft.)

### 1. A fully object-oriented, testable design (do NOT write quick-to-ship code)

**This project is fully object-oriented, mirroring the reference C++ architecture
(strategy-per-operator + factories).** Testability comes *through* clean OO design — clear interfaces,
dependency injection, small single-responsibility classes — **not** by avoiding classes. The failure
mode: code fast to produce but hard to test (hidden state, hard-wired dependencies, giant
multi-responsibility units, logic entangled with I/O).

- **Strategy objects with explicit interfaces.** Each operator, scheme, boundary condition, and solver is
  a **class** implementing an abstract interface (`Protocol`/ABC) — the C++ strategy-per-operator pattern
  (`FaceFluxCalculator`, `FaceInterpolator`, `GradientCalculator`, `BoundaryValueCalculator`,
  `TransientCalculator`), built by **factories** from config. Prefer this to bare module-level functions.
- **Dependency injection through constructors.** A strategy receives its collaborators (sub-schemes,
  coefficients, the mesh/geometry it needs) as constructor arguments, so a test can pass a stub. Never
  hard-wire a concrete dependency inside a class.
- **Single responsibility, small classes.** If a method needs a comment "# step 3", it is probably
  several classes.
- **All physical and numerical parameters are external**, supplied via the constructor, never baked into
  a method as literals (like aquakin's "rate constants are always external via `params`") — mandatory for
  AD-based sensitivity and parameter estimation, and it makes every strategy testable with chosen inputs.
- **Design objects around the smallest sufficient collaborators.** A diffusion-flux strategy should be
  testable with a *stub interpolation and a two-cell mesh*, not a whole solver.
- **Immutable, side-effect-free methods — the JAX requirement, satisfied the OO way.** Objects are
  immutable `equinox.Module`s (JAX pytrees); methods return new values. That is what makes the classes
  `jit`/`grad`-compatible — **not** a reason to avoid classes.
- **Separate computation from orchestration and I/O.** Mesh *reading*, geometry *computation*, and
  residual *assembly* are distinct classes.
- **Every operator/scheme ships an operator-level unit test** (e.g. order of accuracy on an analytic
  field), not only an end-to-end test. If a bug can only be caught end-to-end, the unit was built wrong.
- **Concrete trigger:** *If you cannot unit-test a class without constructing a large object graph or
  monkeypatching state, the design is wrong — fix the seam (usually a missing injected interface) before
  you ship it.*

### 2. One source of truth — no duplicated logic

The failure mode: the same logic in more than one place, so a fix has to be made in several spots (and
inevitably isn't). The reference codes re-derive face geometry in multiple subroutines — **do not
replicate that.**

- **One canonical implementation per concept.** Face geometry (`D`, `fip`, skewness `R`), the diffusion
  flux, the gradient reconstruction, the transient term — each defined *once*, in one module, and
  imported everywhere. Never re-derive a formula inline "just here."
- **Schemes are first-class objects precisely to avoid duplication** — defined once, consumed by many
  equations. Never inline a scheme choice into an operator.
- **Constants and geometric mappings live in a single module.** No magic numbers scattered across kernels.
- **Before writing a formula, search for it.** If it exists, call it. If it exists in two places, that is
  a defect — extract it into one class/method *now*, as part of your change.
- **No copy-paste-modify.** Parameterize the original (or introduce a shared base class / injected
  collaborator) so both call sites share it.
- **Two builders for one object are one builder plus an injected difference.** When two functions
  construct the **same class** and their parameter lists overlap heavily, the overlap is that object's
  configuration surface, written twice. Extract the shared tail into one private builder taking the
  differing collaborator as an argument; each public builder then does only what is genuinely different
  (construct its strategy, pick its default). **A keyword that exists on one sibling and not the other is
  a defect the day it is added, not a property of that path.** The surfaces drift because nothing compares
  them and **no single commit looks wrong**, so every check scoped to "your change" — including the
  Post-Change Checklist and Stale-Record Check — is blind to it by construction. (why: "Engineering Principles 2: the sibling-builder count"; the six
  instances are under "Engineering Principles → 2. One source of truth".)
  - **⚠️ Extracting the shared tail fixes the bodies and hides the surfaces.** **When you extract a shared tail, audit the surfaces above
    it in the same change**, and trust a check's silence only once you have confirmed it can see the case
    (why: "Engineering Principles 2: the shared-tail drift").
  - **For a shared *surface*, the durable repair is a shared object**: when several builders' keywords
    are one *configuration* rather than one *namespace*, make them one value object (`Globalization` in
    `solve/continuation.py`), so a new setting reaches every builder by construction. **Give it unset
    (`None`) fields, not defaults of its own** — unset falls through to each builder's base and then the
    step class's default; own defaults silently overrode a builder's on a one-field override.
  - **Whose property is it?** If a parameter's reason can be stated without naming the strategy the
    builder constructs, it belongs on the shared tail. Where a sibling genuinely must differ, say so *at
    the difference*, with the reason.
  - This does **not** ban siblings that merely share a vocabulary (three multigrid solvers taking
    `hierarchy, b, cycles, omega` are different methods; one builder would be a union bundle). The test is
    *same constructed class* **and** a shared surface that is one configuration.
  - A parameter on only one sibling may be **dominated rather than missing** — delete it (Principle 0),
    don't promote it; and check an "inert" setting reaches anything at all before keeping it.
- **Concrete trigger:** *If a change would require editing the same formula — or the same wiring, or the
  same default — in more than one place, consolidate it first, then make the change once. "More than one
  place" includes two functions in the same file. And before adding a keyword argument to a builder, run
  `tools/sibling_builders.py` (Post-Change Checklist item 2): if a sibling constructs the same class, the
  argument belongs on a shared builder they both call — put it there, in this change.*

> Principles 1 and 2 reinforce each other: small, injected classes are also the natural unit of reuse,
> and code that is hard to test is usually also duplicated, because it is hard to reuse.

### 3. Encapsulation — pass the object, not its guts

The failure mode (a whole review's worth): raw arrays and index plumbing leak across interfaces instead
of living behind the object that owns them, so low-level mechanics get re-open-coded everywhere and
drift. The `mesh.face_cells` / `mesh.face_nodes` connectivity refactor is the standard. Each rule below
is a real smell that was found and fixed.

- **Pass the cohesive object, not a fistful of its arrays.** A function needing owner/neighbour takes the
  `FaceCellConnectivity` (`mesh.face_cells`), never loose `(owner, neighbour)` arrays; one that
  interpolates a cell field takes the field + `face_cells`, not pre-gathered pairs. **More than ~5 arrays
  travelling together in a signature are a missing object** — bundle them (a `FaceState`-style record or
  a real geometry object).
- **Take the smallest sufficient collaborator.** A function that reads only `mesh.face_cells` takes
  `face_cells`, not the whole `Mesh`; one that reads only centroids takes the centroid array, not a
  `CellGeometry`. A big object for one field forces every test to build the world.
- **No duplicate accessors — one name per value.** No forwarding property (`mesh.owner` →
  `mesh.face_cells.owner`), no re-derived value that has a home (`interior_mask(fc.neighbour)` when
  `fc.interior` exists; a private `_dim` forwarding `self.mesh.dim`). Synthesized concepts (`mesh.dim`,
  `mesh.n_cells`) are fine; a second spelling of the *same* array is not.
- **The object owns the operations over its data.** If consumers keep open-coding the same `segment_sum`
  / `neighbour >= 0` / index arithmetic against an object's arrays, the operation belongs **on** the
  object (hence `scatter_max`/`scatter_min`, `interior_edges`, `scatter_owned_partitions`). **Never
  duck-type a lookalike of a real class** (`_FaceGeom` re-declaring `FaceGeometry`); carry the real
  object — `equinox.Module` pytrees map through `shard_map`/`jax.tree.map` unchanged.
- **One formula, one home (Principle 2, for arithmetic).** A linear face interpolation `(1-g)·a + g·b`, a
  projection factor `g`, a normal distance `d·n`, an owned→global scatter — each defined once
  (`schemes/interpolation.py`, a `_normal_distance` helper, …). **Search for the helper** before writing
  `(1-g)*a + g*b` or `sum(d*n)` inline.
- **No god-methods / god-objects.** A method with several `# step N` blocks, or that branches a strategy
  family inline (`if inner == "smoothed"/"multigrid"/"jacobi"`), is several classes: extract a builder
  plus an **injected strategy hierarchy** (the `InnerSchurSolver` / `VelocityBlockSolver` pattern).
  Inline flat-vector index arithmetic for a state layout is a missing value object
  (`solve/state.py::FieldLayout`).
- **A package's `__all__` is its whole surface to other packages (binding, #284).** Another package
  imports from `aquaflux.<package>` (or a nested package such as `aquaflux.io.cad`), never from a
  submodule, and only names in that package's `__all__`; enforced over every module by
  `tests/unit/test_layering.py::test_one_package_reaches_another_only_through_its_public_surface`. When it
  fires, export the name, or — when it is private and generic — move it to the package it belongs in.
  Tests and `validation/` harnesses are exempt. Exporting from a documented subpackage publishes the
  name, so plumbing belongs in the package that uses it.
- **Concrete trigger:** *if you are about to pass an object's raw arrays to a function, thread a loose
  per-cell/per-face array a second time, add a forwarding property, duck-type a lookalike of an existing
  class, or write a one-line formula that already exists — stop and reach for the object/helper instead.*

### 3.5 No scope changes without asking (binding)

**Once the user has agreed to an approach, do not change it — not the design, not the cadence, not the
coverage — without saying so and getting agreement first.** Narrowing scope is a change just as much as
widening it: "the simpler version for now", "Phase 1 / Phase 2", and a cheaper approximation of an agreed
design are all scope changes.

It is about the *evidence* that follows: a measurement taken under an unflagged substitution gets read
as a finding about the agreed design (why: "Engineering Principles 3.5: the observed failure").

- **Flag the deviation at the point of deviation**, plainly: "you agreed X; I am about to do Y instead,
  because Z — is that acceptable?" Not buried in scoping language ("for now", "as a first cut").
- **Never report a measurement taken under a deviation without naming the deviation** in the same breath
  as the result.
- If an implementation obstacle forces a change (a hashability constraint, a compile cost, a missing
  seam), **that obstacle is the thing to surface** — usually the most useful information in the task.
- Delivering *less* than agreed and calling it done is the same defect. If only part of the agreed work
  is finished, say which part is missing in the summary, not only in a tracked issue.

### 3.6 Layering — generic machinery goes in the generic layer, not the first physics package that needed it (binding)

Generic machinery written inside the first physics package that needed it leaves every other residual
on a weaker path (why: "Engineering Principles 3.6: the observed failure", #448).

- **The placement test, applied when you write the code.** Describe the function in one sentence. If it
  needs a physics word — `k`, `ω`, `ν_t`, a closure, a wall function, "coupled RANS" — it is physics. If
  its body would be identical for any other residual (a loop over segments, a step assembled from a
  shift policy, a norm built from a layout, a settings value for a Krylov solve), it is **generic and
  belongs in `solve/`** (or another residual-agnostic module) *the first time*. The first consumer is not
  a reason.
- **Physics packages compose; they do not own loops.** A physics package supplies a residual, a shift
  policy, a `ResidualMeasures`, a drift measure and a `ContinuationSource` to the generic driver. A
  `for segment in ...` loop, step-assembly tail, retry rule or physics-free settings value in a physics
  module is in the wrong package.
- **A capability must be reachable from every residual, or its absence is a defect.** Before adding to
  the march (a trigger, a control, a retry, a guard), ask which residuals reach it: flow-only, scalar,
  coupled. A laminar solve is the same machinery configured differently — never a weaker parallel path;
  if a control experiment needs a weaker driver, that is the finding.
- **Names that carry the first consumer are the smell to grep for** (`coupled_step`, `_coupled_step`,
  `solve_coupled`, `_CoupledMeasures`). When the machinery underneath is generic, it takes a neutral name
  in `solve/` and the specific one becomes a thin configuration of it.
- **Two mechanical guards in `tests/unit/test_layering.py`, in the always-on gate.** `solve/` imports
  nothing outside itself (no mesh, field or physics import), and `precondition/` imports `solve/` and
  nothing else; and **any** module in `aquaflux/` (generic packages and subpackages too) past 1500 lines
  needs a ratchet entry that only ever goes down. (A third guard there is Principle 3's `__all__` rule.)
  **Tripping the size guard is a prompt to sort the file, not to split it by topic**: a physics module
  into generic and physics, a generic one into the jobs it has accreted (host-side setup beside traced
  code, one strategy family beside another).
- **Concrete trigger:** *if the docstring of what you add to a physics package needs none of its
  vocabulary, put it in `solve/`; and a second, simpler path written because the robust one "needs
  turbulence" means the robust one has the wrong dependency.*

### 3.7 Vocabulary — one word, one meaning (binding)

Several names per concept, or several concepts per name, make a reader invert a quantity (why: "Engineering Principles 3.7: the observed failure",
#373). A settings field is read straight from a case file, so **its name is a user-facing key**.

- **Before naming a parameter, field or method, grep the package for the word.** If it already means
  something else anywhere, pick another name, even for a local, private use.
- **A name must not invert under the reader.** Quantities that move in opposite directions (a relaxation
  factor and a shift strength; a floor on the operator and a floor on its preconditioner) get names that
  say which. A docstring saying "careful, this one runs the other way" is the defect, not the fix.
- **A count names its unit.** `max_restarts`, `max_retries`, `sweeps`, `cycles` (V-cycles per
  application in `precondition/`; a step's Krylov count is `raw_cycles`, offset-corrected
  `restart_cycles`), never a bare `limit` or a neighbouring count's unit. ⚠️ Still open: the inner-loop
  observer's `cycles` argument and the private `_Attempt.cycles` in `solve/` are Krylov counts.
- **One concept, one name.** Use the name siblings already use (`v_cycles` → `cycles`, `f_iters` →
  `f_sweeps`, both retired).
- **Rename a clash in one change, with no alias** (Principle 4), and add the old name to the guard.

**The guard is `tests/unit/test_vocabulary.py`, in the fast tier.** `VOCABULARY` lists each overloaded
word with its converging meaning and every module defining a parameter or field of that name, with the
meaning there (clashes awaiting a rename name the issue). Defining a listed word in an unlisted module
fails, as does a listed module that no longer defines it; `RETIRED` lists names that may not return.
**Adding a module under a word is the place to ask whether it means the same thing** — if not, use another
name. The scan sees function, method and property names, their parameters (lambdas included) and
class-level fields — not locals, dictionary keys or prose — so it is a floor, not the whole rule. Add a
word with a history of confusion to `VOCABULARY`.
- **Deliberately kept, so do not "finish the job":** `relaxation_schedule` and the `RelaxationSchedule`
  classes (`SwitchedEvolutionRelaxation` is the literature's name); the method returning `β` is
  `shift_strength`, and every parameter carrying `β` is `shift_strength` or `beta`.

### 4. No compatibility shims before release (pre-release policy — remove this principle at 1.0)

The project is **pre-release: there are no external API consumers, so breaking API changes are free.**
When a refactor changes a public surface, **change it and update every call site — do not preserve the
old one.**

- **No thin adapters kept only to preserve an API.** A wrapper that only re-exposes a refactored
  abstraction under its former shape is dead weight: delete it and point callers at the real object.
- **No deprecation shims, aliases, or back-compat branches** — no `old_name = new_name`, no
  `if legacy_arg is not None`, no parameter kept alive "so nothing breaks." Rename/retype/delete in one
  change and fix callers and tests in the same change.
- A **builder function or factory is not a shim** — it constructs and returns the real object (like
  `momentum_continuation` or `reused_flow_solve`). **A private tail several public builders delegate into
  is not that wrapper either** — it is Principle 2's one implementation, and nothing here licenses a
  second full builder for an object that already has one.

**Delete this principle at the first stable release**, when backward compatibility becomes a real constraint.

---

## Design Goals

- Fully differentiable end-to-end (flow + coupling) via JAX reverse-mode + implicit differentiation —
  gradients through a single converged linear solve, never by unrolling the iteration onto the tape.
- Coupled (block) pressure–velocity with Rhie–Chow — not segregated SIMPLE/PISO.
- Numerics separated from physics: reconstruction/interpolation/gradient schemes are named, swappable,
  independently order-of-accuracy tested.
- The PDE core is a **DAE-residual engine**; aquakin's ODE reactors are the special case
  (all-accumulation, no faces).
- Knob-free robustness for non-expert (water/chem-eng) users.

---

## Technology Stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python | Ecosystem, aquakin parity |
| Numerical backend | JAX | AD for free, jit, GPU batching |
| OO modules | **equinox** (`equinox.Module`) | jit/grad-native classes with inheritance — the strategy hierarchies are `Module`s, not bare functions. Already a transitive dep (diffrax/lineax). |
| Linear solves | lineax (fallback `jax.scipy.sparse.linalg`) | traced, implicit differentiation of the solve |
| Adjoint of the coupled solve | `custom_vjp` around the linear solve | exact, memory-flat gradient independent of iteration count |
| Transient integration | Diffrax | traced, shared with aquakin |
| Mesh | static connectivity arrays + `segment_sum` scatter | XLA-friendly graph/message-passing layout |
| Compiled host loops | **Numba** | a loop whose value is in the work it SKIPS (a ray walked to its first hit) cannot be traced — a traced loop pays its worst case on every item — and a host loop can also fuse several whole-array passes into one threaded loop. Used in `aquaflux/radiation/` (`.claude/rules/radiation.md`) and, called from inside XLA programs, for the preconditioner's level products (`.claude/rules/solve-amg-multigrid.md`). A core dependency, by the project owner's decision over keeping it optional. |
| Mesher (optional) | **Netgen** (`meshing` extra, LGPL-2.1, pinned) | CAD-face triangulation and a tet fill that keeps a supplied surface. Its own OCCT aborts a process holding OCP's, so it runs **only in a child process** (`.claude/rules/meshing.md`) |
| Browser interface (optional) | **PyVista/VTK + trame + plotly** (`aquaflux_ui`, `ui` extra) | a local browser interface needing no JavaScript toolchain: VTK renders locally and trame streams images, so a large mesh never reaches the browser. A separate top-level package that imports neither aquaflux nor JAX (`import aquaflux` enables x64), asking the solver through an `aquaflux serve` process (`.claude/rules/ui.md`) |
| Case file | **PyYAML** (`aquaflux/case/`) | a case's mesh, physics, boundaries, (for a flow) fluid and numerics, and solver in one YAML document, read by YAML 1.2 rules for plain values (PyYAML's 1.1 rules read `1e-5` as a string and `no` as a boolean) and validated per position by `solve.SettingsMapping` — so no pydantic. The equation DSL (YAML → AST emitting terms) is different and is still the **last** layer, not built |
| Bounded compilation cache | **filelock** (`aquaflux/__init__.py`) | `import aquaflux` points JAX's persistent compilation cache at `~/.cache/aquaflux/jax` and **bounds it** (JAX's default max size `-1` never evicts). **⚠️ A bound without `filelock` disables the cache entirely**, so the package checks and degrades to merely unbounded. Override with `AQUAFLUX_COMPILATION_CACHE_MAX_GIB` (negative for no bound), `AQUAFLUX_COMPILATION_CACHE_DIR`, or `AQUAFLUX_DISABLE_COMPILATION_CACHE=1`. The 2.0 s persistence floor stays the default; `AQUAFLUX_COMPILATION_CACHE_MIN_COMPILE_SECS=0` opts one run in to persisting the small op-by-op programs, for a case rerun on one mesh (with a bound every write rescans the cache directory). Measurements: `.claude/notes/claude-md-log.md`, "Technology Stack: the Numba and cache rows" and the ledger, "Persisting the small programs". |

The coupled-block preconditioner is the **top research risk**. The first build-on candidate to evaluate
is `jaxamg` — do not assume it; verify its adjoint is implicit-diff and that it preconditions a *block*
system. (See `.claude/rules/solve.md` for the chosen block-triangular SIMPLE-type direction, and
`.claude/rules/solve-amg-multigrid.md` for the known traps.)

---

## Architecture

This is the architectural orientation; the subsystem rules under `.claude/rules/` carry the per-package
detail.

**Object-oriented over a struct-of-arrays substrate.** Mesh and fields are flat arrays (one array per
quantity for *all* elements, never one object per element — forced on us because JAX vectorizes arrays,
not object graphs). *Everything above that layer is object-oriented*: `Mesh`,
`FaceGeometry`/`CellGeometry`, and the strategy hierarchies are `equinox.Module` classes mirroring the
reference C++ architecture. SoA is a data-layout constraint, **not** a licence for procedural code.

**The Layer-0 residual substrate is the foundation.** Everything reduces to a discrete cell residual
`R(state, params)` assembled by **gather → compute → scatter**. The gather/scatter mechanics are the one
**connectivity API** — `mesh.face_cells` (`FaceCellConnectivity`: indexing on `owner` / `safe_neighbour`,
`scatter` / `scatter_conservative` / `scatter_symmetric`) and `mesh.face_nodes` (`FaceNodeConnectivity`)
— which owns the `segment_sum` over face→cell / face→node index arrays and the boundary convention;
operators, schemes, BCs and the mesh geometry compose it (see `.claude/rules/mesh.md`). The Jacobian and
adjoint come from **AD** — there are **no hand-derived linearization coefficients** anywhere (the central
simplification over the reference codes). Each operator writes the full physical flux as one honest
residual term; AD assembles the matrix.

**Vector algebra lives in one leaf, `aquaflux/vectors.py`** (imports only `jax.numpy` and the standard
library, so any subsystem may use it): `dot(a, b)`, `norm_squared(a)`, the zero-safe magnitude `norm(a)`
(`0` with a zero gradient at the zero vector — use it wherever two points may coincide, never
`jnp.linalg.norm` or a `sqrt(dot(...))` guarded after the root), `scale(vectors, scalars)`, and
`reflect(vectors, normal)` (the image-source method's one formula, shared by `radiation/mirrors.py` and
the photometric profile's mirror image). **Preference (binding): reach for these helpers instead of
open-coding `jnp.sum(a * b, axis=-1)` or `s[..., None] * v`** — the named helper states the intent and
gives one home to change. Rank-3 tensor algebra (Hessian outer products, `einsum`) stays explicit.

**⚠️ `dot` is deliberately not spelled `jnp.sum(a * b, axis=-1)` — do not "simplify" it back.** It
multiplies, then sums the components **explicitly**, because on CPU a fusion rooted at a reduction runs
single-threaded (why: "Architecture: the dot speed-up"). `norm_squared` inherits it.
- **Multiply first, unroll only the sum** (indexing unbroadcast operands breaks a trailing axis of 1),
  and **⚠️ form the product with `jnp.multiply`, not `*`** (two NumPy operands would return an `ndarray`
  without `.at[]`). Both pinned by unit tests.
- **⚠️ It is not bit-identical to the reduction** (~2–6e-16 of `Σ|aᵢbᵢ|`; judge such a difference against
  the magnitudes summed, never a near-cancelling answer). A test comparing two paths that **both** route
  through `dot` must still hold exactly; only a value pinned to the reduction's bits may be re-pinned.
- **Guarded by `tests/unit/test_compiled_residual_census.py`** (fast tier, CPU only), which ratchets the
  count of custom and unpartitioned fusions in the compiled coupled RANS residual and its JVP. ⚠️ Its
  fixture size is load-bearing: XLA partitions kernels only above roughly 768–1200 cells, and the module
  fails if its fixture stops seeing partitioned kernels.

**Ragged lists live in one leaf, `aquaflux/ragged.py`** (numpy only; `rows`, `group`,
`pairs_within_groups`). Reach for it before writing `np.repeat(..., counts)` index arithmetic.

**Points are ordered along a Morton (Z-order) curve in one leaf, `aquaflux/morton.py`** (numpy only);
the radiation package cuts every grouping of points from it. ⚠️ **Each axis is scaled to its own extent,
deliberately — do not make the cells cubic** (#574, `.claude/notes/radiation-log.md`, "ONE MORTON
ORDERING, PER-AXIS CELLS").

**Once-per-process warnings live in one leaf, `aquaflux/warn_once.py`** (standard library only, a
neutral leaf): `warn_once(kind, message)` emits the first warning of each kind in the process;
`forget_warnings(*kinds)` resets it for tests. ⚠️ **One record per kind, never one flag for all**: a
shared flag let a routine repair report silence a graver one. Name a kind after its module
(`"multiple_correction.repaired"`); Python's `"once"` filter keys on the message, so it is no substitute.

**Frozen preconditioner operators are assembled in one place, `aquaflux/precondition/frozen_operator.py`**
(`ConvectionDiffusionStencil`, off the jit path). First-order upwinding is the **preconditioner's**
choice, not the model's (it gives an M-matrix aggregation can coarsen). Detail in `.claude/rules/solve.md`.

```
Mesh (SoA topology) + FaceGeometry/CellGeometry            (classes)
   → operator strategies (DiffusionFlux, ...) consuming injected scheme strategies
   → mesh.face_cells.owner/safe_neighbour → compute flux → mesh.face_cells.scatter_*
   → cell residual R(state, params)
   → AD (jvp / vjp / IFT) → Jacobian / adjoint
   → Newton + implicit-diff linear solve → converged state (and its exact adjoint)
```

A Layer-0 escape hatch still lets an advanced user supply a raw flux/source **closure** — but
aquaflux's own operators, schemes, and BCs are OO strategy classes. The DSL that eventually emits terms
is the last layer.

### JAX x64 Mode

FVM (and stiff coupling) require 64-bit floats. `aquaflux` enables x64 mode at import time — **global,
process-wide** JAX state, a documented side effect of `import aquaflux`. Do not remove it. See
`tests/unit/test_x64_import.py` (tested in a subprocess, because the effect is process-global).

---

## Testing Architecture

### Three layers

- **Unit** — components in isolation; fast; run on every change. Every numerical operator has an
  operator-level test (order-of-accuracy on an analytic field). Principle 1 is enforced here.
- **Integration** — full assembly → solve pipeline against analytical solutions.
- **Validation** — scientific correctness against analytical/published solutions, marked
  `@pytest.mark.validation`, run separately.

```bash
tools/fastgate.sh              # the always-on gate: not slow, not validation
tools/fastgate.sh validation   # analytical/published-solution suite
tools/fastgate.sh slow         # the slow tier
tools/fastgate.sh all          # everything
```

### A test that cannot fail is not a test (binding)

**Before a new test is considered done, name the specific wrong answer it would catch.** A test that
would pass against a broken implementation is a false signal, read by every future reader as evidence
the code works. This applies equally to your own tests and to one an agent hands back: a subagent asked
to "add a test" tends to return something green against the code as it stands, without checking it
would go red against a plausible wrong version.

**Chasing every undetected mutation
is the same failure as never checking, spent the other way**: judge each on whether the path is reachable
and the behaviours distinguishable, and record which mutations were dismissed and why (why: "Testing Architecture: the mutation-pass summary").

**How to check a test can fail:** comment out, invert, or disable the line of production code the test
pins, rerun *only* that test (or its file), confirm it goes red, then restore the line exactly. It costs
seconds and is the only reliable way — reading a test is not enough, because a plausible wrong
implementation is exactly what a reader's mental model fails to simulate. Do this before calling coverage
complete, and before reporting "N passed" as evidence a change is correct.

**Concrete smells — reject each on sight, in your own tests and in an agent's:**
- **The assertion is satisfied by many wrong answers** — `is not None`, a shape check, `not any(isnan)`
  (see "'without NaNs' is the floor" under Canonical tests). Assert the *value* against an independent
  reference: an analytic solution, a hand-computed number, a differently-derived path, or a property the
  specific correct answer has and a wrong one plausibly would not.
- **Every fixture exercises the same code path.** Tests differing by one number leave untested any branch
  only a differently *shaped* input reaches (e.g. uniform node counts hid a row mix-up across faces).
- **The mock or stub replaces the exact thing under test** — then the test checks the wiring, not the function.
- **`pytest.raises(..., match=...)` broad enough to match several failure branches** cannot tell you the
  intended one fired — construct the input so only that branch can raise, or match a unique substring.
- **A "regression" test pins today's output with no independent way to know it is right** — it catches
  drift, not a number that was already wrong. Prefer an independently checkable value where the cost is
  reasonable.

**When reviewing coverage — yours or an agent's — ask per new test: what specific wrong answer does this
catch, and did I verify that it would?**

**⚠️ The same defect in a measurement script has no runner to notice it. A tautology dressed as an
assertion is worse than the bare quantity** (why: "Testing Architecture: the closure tautology").
- **Concrete trigger:** *before writing `must`, `expected`, `should equal` or a tick beside a computed
  pair — in a script, a log line, or a commit message — ask what inputs would make it disagree. If none
  would, it is a derived quantity and must be labelled as one.*
- **A number that matches what you already believe is the least reliable kind of agreement.** Decompose it
  before stopping (why: "Testing Architecture: the 1.37x decomposition").

**⚠️ CI is not a superset of a local run, and `importorskip` is why**: a module gated on a dependency
outside the `test` extra is skipped in CI and runs only where it is installed. **When a test fails locally
and passes in CI, check the skip counts before blaming your machine.**

`fastgate.sh` wraps `pytest -q -m <tier>`, writing the run to a file and reporting pytest's **own** exit
status (its behaviour, including the guards below and the scoped `--changed` form, is pinned by
`tests/unit/test_fastgate.py`). Invoke
`pytest` directly only for something the wrapper does not pass through — that also skips the case guard
— and never through a pipe: the suite prints shutdown chatter *after* the summary, so `| tail -n` hides
the result and returns `tail`'s `0`. ⚠️ **The pipe rule applies to `fastgate.sh` itself, and knowing the
rule is not what protects you**: `tools/fastgate.sh … | tail -4` returns `0` *and* crops the
`FAILED (pytest exit N)` block, leaving a `full log:` line that reads like a pass (why: "Testing Architecture: the cropped fastgate run"). **Redirect (`> file`) and read the script's own exit status; never judge it by
its last few lines.**

**The fast tier runs across worker processes; the `slow` and `validation` tiers do not** — a memory
decision (a bfs3d march peaks at ~8.5 GiB on Linux, each copy of its Jacobian 0.41 GiB; CI shards the heavy tiers at `-n 1`). The
fast tier distributes by **file** (`--dist loadfile`: module fixtures built once, file order kept) and
pins **one BLAS/XLA (and Numba) thread per worker**. `FASTGATE_JOBS=<n>` sets the worker count;
`FASTGATE_JOBS=0` (or your own `-n`) opts out — do that when bisecting. ⚠️ **Judge memory from the
footprint (compressed pages included), never RSS**, which undercounted a fast tier's ~50 GB footprint
tenfold (measurements in `.claude/notes/claude-md-log.md`).

### Scoped runs (`--changed`) are a pre-push check, not the gate (binding)

**`tools/fastgate.sh --changed` runs the tests whose imports reach what you changed, plus the declared
reaches. CI's full fast gate stays the required check, and a green `--changed` run says nothing about
tests outside its printed selection. If your change touches a shared leaf, the selection widens to most
of the suite. That is correct, so don't work around it.** It exists because the whole fast tier takes on
the order of two hours on two workers in a 4-core container. It takes any tier, so `slow --changed` /
`validation --changed` is the cheap answer to Post-Change Checklist item 3.

`tools/scope_tests.py` is its one home (`test_layering.py` reads imports through its walk):
- **By import, names followed to their definition**: `from aquaflux.solve import X` depends on the
  submodule defining `X` (re-exports followed through each `__init__`), that `__init__`'s own code and
  every ancestor `__init__` — not every submodule the `__init__` loads. Imports inside functions count,
  and so do those of a child program a test holds in a string; a `conftest.py` reaches the tests below it.
- **By declaration**: `REACHES` lists each test module, or `tests/support` helper (charged to its
  importers), that reaches a file without importing it. `tests/unit/test_scope_reaches.py` fails on a test
  that builds a path from `__file__`, names a repository path, starts a subprocess, touches `sys.path` or
  imports by name without an entry, and on a declared glob matching no file — declared, not discovered.
- **Widening**: a changed path neither rule places selects the WHOLE tier and is named. Only
  `REACHES_NO_TEST` paths and modules nothing imports select nothing, and the run says so in words.

The selection, with what selected each file, is printed and heads the log. Exit 5 (no test of the tier
in the selection) is reported as `0`, for scoped runs only; a change it cannot read exits `2`. **Blind
to**: an import-time error in a submodule that breaks every importer of its package when nothing
importing its own names is selected, and a reach the census does not recognize — CI's gate covers both.
Measured cost: `.claude/notes/claude-md-log.md`, "Scoped runs".

### What a fast-tier test may spend (binding)

The fast gate is read on every change, so a test that runs longer than it needs is a tax paid by
everyone. Three patterns produced most of the tier's wall clock (2026-08-23), all cost with no coverage
attached — check for them before adding a solve to this tier:

- **A fixed iteration count past convergence.** `for _ in range(8): newton_step(...)` on a cavity
  converged by step four spends half its time on a converged root — and in a differentiability check puts
  the extra steps on the tape. Stop on a convergence test, or name the count a constant and say what it
  is (convergence plus a margin).
- **A dense matrix built one column per dispatch.** A Python loop of `jax.jvp` over basis vectors pays
  eager dispatch `ndof` times; `jax.jacfwd` / `jax.vmap` build the identical matrix in one batched pass.
- **A deliberately-failing arm running out a large step cap.** Give both arms of such a comparison the
  SAME budget, a small multiple of what the succeeding arm needs — also the fairer test.

**⚠️ A unit CI job that slows down or times out is usually the runner swapping, not your change — and the
fix is a shard, not a worker.** Workers grow as they draw modules, and unsharded `-n 3` on a 16 GB runner
swapped tests past the 900 s timeout with no assertion failing. The tier runs as **three shards of two
workers** (`--splits 3 --group N -n 2`), balanced by durations recorded **on CI, never locally**. When it
outgrows that, **add a shard; never add a worker.** Before blaming your branch, read the job's memory step
(`vmstat` every 30 s) and check whether unrelated branches are slow too. Caching JAX compilations across CI
runs buys nothing. `.github/workflows/ci.yml` carries these measurements.

**The `shard_map` tests run in their own CI job, selected by the `distributed` marker**, because in the
unit job each xdist worker's forked child oversubscribed the runner. A test past the 900 s timeout under
`--timeout-method=thread` kills its worker and reports `node down: Not properly terminated` — **that reads
as a crash in the test and is not one**; check what else was running. `tests/unit/test_distributed_marker.py`
fails if a module spawning simulated devices lacks the marker. The local fast gate still runs them inline.

None of this licenses weakening a check to make it quick. A finite-difference-validated adjoint costs
three solves and is the point of the project; that is a test spending what it must.

### Conditional coverage is DECLARED, not discovered (binding)

**A skip and a pass are the same exit status.** `tests/unit/test_optional_dependency_skips.py` holds the
census: every module gated behind an optional dependency, and every other file that can skip, with what
the skip is conditional on. Adding a gate without its entry fails the fast gate — "this coverage is now
conditional" is a decision to write down, not one to arrive at by importing something.

- **A dependency kept out of the `test` extra is a cost decision, and the census is where its price is
  written down.** Today that is only `scotchpy` (one partitioner test).
- **The workflow runs every tier with `-rs`**, so a skipped test is named with its reason in the job log.

Marker typos are handled in `pyproject.toml`: `strict_markers` makes an undeclared marker a collection
**error** (tiers are selected with `-m`, so a typo would silently move a test to a tier that does not run
it, or drop a heavy solve into the always-on gate). The ini form is load-bearing: `--strict-markers` in
`addopts` parses cleanly and does **nothing**.

### Canonical tests (must always pass once implemented)

- **Primary field (analytical):** plane wall with convection vs the closed-form θ (Gate A).
- **Sensitivity (analytical) — the point of the project:** `jax.grad` of the converged solver w.r.t. `Bi`
  vs the closed-form ∂θ/∂Bi (Gate B). Every integration suite must include an explicit test that `grad`
  flows through `solve()` without error and without NaNs.
  ⚠️ **"Without NaNs" is the floor, not the test.** `0.0` is finite, so an adjoint severed anywhere — a
  stray `stop_gradient`, a closure that stopped carrying its dependence — passes a finiteness assertion
  (why: "Testing Architecture: the severed nut_wall adjoint"). **Compare the gradient against a finite difference
  wherever the cost allows; where it does not, at minimum assert it is non-zero.**
- **AD-exact-linearization (skewed mesh):** on a non-orthogonal mesh, the linear problem converges in one
  Newton step (Gate C) — the concrete improvement over the reference.
- **Coupled-solve adjoint (iteration-count-independent):** for any iterative / coupled / fixed-point
  solver, `jax.grad` through the converged solve must match a reference gradient (finite difference or
  closed form) **and be independent of the forward iteration count** — the coupling analogue of Gate C,
  proving the adjoint is the implicit-function-theorem solve on the converged residual, not the loop
  unrolled onto the tape. A stability smoke test does **not** establish this; it must be its own test.
  ⚠️ **Such a test must make the two paths genuinely differ** — raising a `max_steps` cap a converged
  solve never reaches leaves the march bit-identical. Vary something that moves the path (`inner_steps`,
  the shift strength `beta0`; see `test_the_coupled_adjoint_is_independent_of_the_forward_iteration_count`)
  and **assert the step counts differ**, measured with an observer: a lever that moves the path need not
  move the step count.
- **x64:** `assert jax.config.x64_enabled`.

---

## Comment Convention

Code comments and docstrings are written **for a human reading this repository** and must stand on their
own: explain the code on its own terms, and **never** point at anything a repo reader cannot open. Do
**not** name any of these four in a comment or docstring (binding — this leak had to be scrubbed once):

1. **The precursor codebases** — the C++ framework and the Fortran solver. No "the C++ `Face`", "mirrors
   the reference C++", "the original Fortran", "`coeff.F90`", "ported from …". They are acknowledged
   **once in the README** (with links) and nowhere else in the tree.
2. **The Claude-facing files** — this `CLAUDE.md` and anything under `.claude/`. A docstring must never
   say "See `.claude/rules/mesh.md`" or "per `CLAUDE.md`" (see the **Claude-Facing-File Reference Ban**).
3. **The internal design notes** — the author's private working notes (design records, briefings,
   milestone specs) kept **outside the repository**. "See the preconditioner design note §4" points at
   nothing; inline the reasoning. (Files under `docs/` **are** shipped and may be cross-referenced.)
4. **Self-citations to the author's own papers** ("the DeGroot-2019 wall", "the DeGroot–Straatman flux").
   State the *physics* instead ("the Green–Gauss accuracy ceiling on skewed grids", "a flux-continuous
   non-orthogonal diffusion flux").

**Why:** references to artifacts a reader cannot open rot immediately and confuse anyone outside this project.

**What you MAY do — cite real, published science properly.** Cite the *math* by name
("over-relaxed non-orthogonal diffusion correction", "divergence-theorem cell volume"), and third-party
eponymous citations are welcome ("Ghia et al. (1982)", "the Venkatakrishnan (1993) limiter",
"Rhie–Chow interpolation", "Murphy–Golub–Wathen (2000)", "Patankar"). *Published third-party science,
cited properly = fine; pointers to this project's private / internal / Claude-only artifacts = never.*

> **Note for Claude specifically.** The `.claude/rules/*.md` files reference the C++/Fortran precursors
> freely — that provenance is their job. What they must **not** do is point at the author's private
> design notes: inline the fact instead. The boundary is the **shipped surface** (`.py` files and
> `docs/`): when a rule says a class "mirrors the C++ `Face<T,3>`", the code comment says "polygon
> centre-fan vector area", **not** "mirrors the C++".

## Claude-Facing-File Reference Ban

The **Claude-facing files** — this `CLAUDE.md` and everything under `.claude/` — are tracked (so a
contributor can see the standards) but are **agent instructions, not user documentation**. They must
**never be referenced from any public-facing file**: not a `.py` comment or docstring, not the `README`,
not anything under `docs/`. No "see `CLAUDE.md`", "per the `.claude` rules", "`.claude/rules/mesh.md`", or
"as the project briefing requires". State the rule or fact on its own terms instead (just *use* Canadian
spelling; explain the coupling as prose or cite a shipped `docs/` page).

**Why:** such a pointer lands a reader in material written for a different audience.

## Docstring Convention

All public functions, classes, and methods use **NumPy docstring format** (Parameters / Returns / Raises
/ Examples), with array shapes stated for every array argument.

## Self-Contained-Docstring Convention

Comments and docstrings must **stand on their own for a repo reader**. Four leaks from *our working
conversations* read as authoritative but cannot be checked by a reader; all four are **binding**:

1. **Define every acronym / non-obvious term at first use in a file** ("compressed-sparse-row (CSR) form
   — a row-pointer array plus a flat index array", then "CSR"). Per file, since a reader may open just
   one. Standard math names cited author-year are exempt.
2. **No internal-staging or roadmap labels** — no "Milestone 0", "Gate B", "Phase 2", "the first build
   target", "not yet exercised in <stage>". State the property instead: not "not exercised in Milestone
   0" but "node_coords is a differentiable leaf, so gradients w.r.t. node positions flow through it". (A
   self-explanatory *code* state like "the first timestep uses BDF1" is fine.)
3. **No hard performance numbers invented in conversation** ("million-cell meshes", "builds in seconds",
   "handles N cells"). Describe the mechanism instead ("avoids the per-face Python loop, which is the
   bottleneck for large meshes"). A figure is fine only when it is a reproducible property of the
   algorithm a reader could re-measure ("on a model Poisson the V-cycle contraction is ~0.25").
4. **No design-principle labels or their names** — never "single source of truth", "one source of truth",
   "the DRY consolidation point", "one canonical implementation", "no duplicated physics", "the unit of
   reuse", "reuse over reimplementation", or "(Principle N)" / "(CLAUDE …)" in shipped code. Keep the code
   fact without the slogan: "the diffusion flux, shared by the serial and distributed paths" ("reuses /
   shared by / delegates to <name>" stays; the editorial tag goes).

**Why:** a docstring is read by someone who cannot see our conversation, notes, stages, or principles.

## Spelling Convention

Use **Canadian spelling** by default — in identifiers, comments, docstrings, and docs: double the final
consonant (`labelled`, `modelled`, `travelled`, `cancelled`), `-our` (`colour`, `neighbour`,
`behaviour`), `-re` (`centre`, `metre`, `fibre`). **Keep `-ize`** (`normalize`, `organize`) and `analyze`
— standard Canadian. So: `LabelledGroups`, `cell-centred`, `neighbour`.

---

## Development workflow

> The workflow mirrors aquakin: branch → PR → green lint gate → merge; never commit on `main`;
> commit/push only when the user asks.

### Use the blessed command; do not hand-roll it (binding)

Four operations have one correct invocation. Use it.

| to do this | run this |
|---|---|
| run a validation case or any long solve | `validation/run_case.sh <script.py> [--wait]` |
| see what a long run is doing | `validation/run_case.sh --status` · `tail -f <its log>` |
| run a test tier | `tools/fastgate.sh [fast \| slow \| validation \| all]` |
| run only the tests a change can reach, before a push | `tools/fastgate.sh [tier] --changed [--base REF]` |
| build the Sphinx docs locally, the same strict way CI does | `tools/build_docs.sh` |

Each exists because the hand-rolled version fails *silently* (how each one fails: "Development workflow: how each hand-rolled command fails").

`run_case.sh` gives unbuffered output **redirected, never piped**, to a timestamped log; the machine held
awake; a free-memory and load pre-flight; a refusal to start a second case; a run-file recording pid,
worktree, branch, commit and case settings; and, under `--wait`, the case's **own** exit status. Its most
valuable output is that **"is this run mine, and what is it testing?" has a written answer**.

**⚠️ `tools/fastgate.sh` refuses to start beside a running case (exit `3`) or beside another tier from
any worktree or session (exit `4`; a machine-wide lockfile under `~/.cache/aquaflux/`, location
overridable with `FASTGATE_LOCK_DIR`).** `FASTGATE_FORCE=1` overrides both; neither applies under `CI`.
Both exist because *knowing* the one-heavy-job rule does not prevent collisions (why: "Development workflow: the collisions behind the fastgate guards"). The case guard asks `validation/run_case.sh --running` rather than reading
the run-file, so its format and `kill -0` liveness rule keep one home. `tests/conftest.py` calls
`jax.clear_caches()` after every module, since a worker otherwise keeps every compiled executable.

### What no runner can enforce (binding)

- **Print one line per outer step, `flush=True`.** A log that grows per step lets a bad trajectory be
  killed at minute three instead of at the end.
- **Actually watch it.** A run nobody reads until it finishes cost its full wall time to tell you
  something it knew in the first minute.
- **A run that spanned a machine sleep is void for cost.** It converges correctly, but the wall-clock
  column silently absorbed the sleep. `run_case.sh` holds the machine awake; if a sleep happened anyway,
  keep its step and cycle counts and discard its timings.
- **Sweep in one process, not N.** These jobs are memory-bound (a bfs3d march peaks at ~8.5 GiB, set by
  its re-fits), so spare cores are not a reason to parallelize. Loop over β in one process, `del` the
  big arrays between iterations, and load a cached factor from disk rather than re-materializing it.
  Concurrent probes have exhausted this machine and suspended every application on it, including the
  driving session.
- **Before asserting what a background job is or did, read its own record** — the run-file, the task
  output file, the log. A start time and a parent pid are circumstantial; a run whose gate logged
  `HEALTHY -- launching march` is not.

### Measure the quantity you will be judged by (binding)

**The number a run is steered/watched by and the number it is finally judged by must come from one
definition.** (why: "Development workflow: the two reattachment lengths".)

So a scalar a case exposes for a solver to be watched by must be **the same callable** the final report
uses. A genuinely wanted diagnostic variant is reported under a **different name**, alongside.

### Record what a measurement was taken under (binding)

**A measured finding written into these files must name the configuration it was measured with — the
defaults that were active, the state, and the operating point. A number without its configuration
expires silently the moment any of them changes, and you cannot tell that it has.**

A finding whose configuration and harness are gone is **unfalsifiable**, worse than wrong (a wrong
finding gets corrected, an unfalsifiable one gets cited; why: "Development workflow: the three unfalsifiable findings"). **Keep the probe in the repository**: it is the
difference between a finding that can be re-asked and one that can only be cited.

- **Name the defaults in force.** Not "21 iterations" but "21 iterations at ILU(1), 2 sweeps, smoothed
  aggregation, 2 levels" — which a reader can see no longer applies after a default change.
- **Name the state and the operating point** — which case, which step or checkpoint, which shift, and
  whether converged or mid-march. The same arm can measure 6 cycles at one and 22 at another.
- **Say what an inference does NOT distinguish.** An under-determined inference left unmarked gets read
  as its most expensive meaning.
- **When a default changes, grep the rules for findings measured under the old one** and mark them, in
  the same change (part of the Post-Change Checklist's Documentation-sync item).

### Start from an up-to-date main (do this FIRST — binding)

**Before creating a feature branch, putting any change on it, or running the test suite, refresh `main`
from the remote.** A branch from a stale `main` diverges from what has merged, tests run against
outdated code, and the PR carries avoidable conflicts.

At the **start** of every task — before the first branch or the first test run:

1. `git fetch origin` (or the configured remote).
2. Compare: `git log --oneline HEAD..origin/main` (or `git rev-list --count HEAD..origin/main`). Non-empty
   means your base is stale.
3. Bring your base up to date before branching — update local `main` to `origin/main` and branch from it,
   or rebase an existing feature branch onto the freshened `origin/main` (in a worktree, sync the branch
   you are on the same way).
4. **Only then** create/switch to the feature branch, make changes, and run tests.

If the base was stale and you have already started, stop and rebase onto the freshened `origin/main`
first. When a sync would pull in changes that could reasonably affect the task (a touched subsystem moved
under you), surface that to the user rather than silently rebasing over it.

CI (`.github/workflows/ci.yml`) runs `ruff check` + `ruff format --check` on `aquaflux`, `aquaflux_ui`,
`tests` and `docs` for every PR and push to `main`, with ruff pinned by the `lint` extra (`codespell`
covers `aquaflux aquaflux_ui tests`). Enable the committed hooks once per clone with
`git config core.hooksPath .githooks`: **pre-push** runs the same ruff commands; **pre-commit** prints a
non-blocking reminder to sync the docs when a commit touches `.py` code (bypass with `--no-verify`).

**Both hooks fail silent when `core.hooksPath` does not resolve to them** — git runs nothing and says
nothing. Use the **relative** `.githooks` (resolved per working tree); an **absolute** path pins every
worktree to one checkout's branch. A per-worktree override (`git config --worktree`) can shadow a correct
value: `git config --show-origin --get core.hooksPath` names the file in force, and
`git config --worktree --unset core.hooksPath` drops it. `tools/check_hooks.sh` (run first by
`tools/fastgate.sh`; silent when sound and under `CI`; always exits `0`; pinned by
`tests/unit/test_check_hooks.py`) warns when the hooks **will not run** and when they run only **by
coincidence** (an absolute path outside this checkout).

### Documentation

User-facing docs are a **Sphinx site in MyST Markdown** under `docs/`, built with `-W` by Read the Docs
and by CI on every PR (the required `fast gate` depends on it). **Build locally with
`tools/build_docs.sh`** when you touch a documented docstring, a `docs/` page, or a cross-reference.
The docs are the **shipped surface**, so the Comment Convention and the Claude-Facing-File Reference Ban
apply. The rest (what publishing a subpackage means, `SUBPACKAGE_GROUPS`, the docstring traps that break
`-W`, extension dependencies, the excluded `package_structure.md`) is `.claude/rules/docs.md`, which
loads for `docs/**`; **read it before editing a documented docstring outside `docs/`**, which it cannot
glob.

---

## Module Review Rubric

**When asked to review a module (or auditing one before commit), read
`.claude/notes/module-review-rubric.md` first and run both of its passes**: Pass 1, the local smells (the
Engineering Principles as a checklist); Pass 2, decomposition and extensibility, which needs **fresh
eyes** — a reviewer agent prompted with the Pass-2 dimensions explicitly — and **verify each finding
yourself** before acting.

---

## Stale-Record Check (binding — run before EVERY commit, BEFORE the Post-Change Checklist)

**Ask what your change makes FALSE, not only what it leaves incomplete.** The Post-Change Checklist's
documentation-sync item catches the file that *describes* the code you touched. This catches the larger
class: every entry anywhere in the Claude-facing files your change silently invalidates — a symbol you
renamed, a default you moved, a measurement taken under the configuration you changed. Nothing in the
test suite fails when they rot.

**It is the most expensive recurring defect in this project**: stale facts lifted from these files by
grep have been asserted as current and passed into sub-agent briefs (why: "Stale-Record Check: the NewtonSolver entry").

**Run this before every commit that renames, deletes, moves, or re-values anything:**

```
git diff --cached | grep -E "^-" | grep -oE "\b_?[A-Za-z][A-Za-z0-9_]{3,}\b" \
  | grep -E "_|[a-z][A-Z]" | sort -u > /tmp/touched
grep -rnFf /tmp/touched CLAUDE.md .claude/rules/ README.md docs/
```

**⚠️ A forwarding wrapper hides call sites from an audit scoped to the callee**: a wrapper taking
`**kwargs` passes keywords through under its own name, so a scan for `the_function(` reports clean on a
call that breaks (`solve_reynolds_continuation` forwards to `solve_coupled` via `**solve_kwargs`). Scan the
wrappers too. The `_|[a-z][A-Z]` filter keeps only code-shaped words, so **its blind spot is a renamed
all-lowercase word** — this grep is the cheap half; also grep for the *claim* (Documentation-sync item).

Then, for every hit, decide: still true / update / **delete**:

1. **A renamed or deleted symbol is a defect wherever it still appears.** Grep every Claude-facing file
   for the old name, not just the rule that owns the subsystem — these names spread.
2. **A moved default orphans every measurement taken under it**, making it *unfalsifiable*. Mark it with
   its configuration, or delete it.
3. **⚠️ Supersede by deleting — never by striking through or appending a correction below the claim**;
   both leave the wrong statement greppable without the correction. If the dead finding taught a trap
   worth keeping, collapse it to **one line stating the trap** and delete the body.
4. **Prefer a pointer that inoculates.** Where a dead name is likely to be searched for, one line saying
   "there is no such symbol; the real one is X" lands a grep on the correction instead of on nothing.
5. **Never cite a path that is not in the repository.** A `scratchpad/` or private-note pointer makes
   its finding permanently un-re-adjudicable; move the harness into `validation/` or drop the pointer.

**Why the rules files specifically:** they auto-load and are searched by name, so a stale entry there is
actively served to the next reader as current fact.

## Post-Change Checklist

After **every code change**, before considering the task complete, review and act on:

1. **Engineering Principles review (the priority gate for this project).**
   - **Testability (Principle 1):** can every function you added be unit-tested in isolation, with small
     explicit inputs, no global state? If not, fix the seam *now*.
   - **Duplication (Principle 2):** did you introduce logic that already exists, or copy-paste-modify a
     block? If so, consolidate to one source of truth *now*.
   - **Encapsulation (Principle 3):** did you pass an object's raw arrays instead of the object, take a
     whole `Mesh` where `face_cells` would do, add a forwarding property or a second spelling of one
     value, duck-type a lookalike of a real class, inline a formula/scatter that has a home, or grow a
     `# step N` god-method? Fix the seam *now* — reach for the object/helper.
   - **Layering (Principle 3.6):** did you put code in a physics package whose one-sentence description
     needs no physics word (a driver, a step assembly, a measure, a settings value)? Move it to `solve/`
     now, and check every residual can reach the capability you added.
   - **Vocabulary (Principle 3.7):** did you name a parameter, field or method with a word the package
     already uses for something else, or name a concept differently from its siblings? Grep the word; if
     `tests/unit/test_vocabulary.py` lists it, your new use must mean the listed thing. A rename adds the
     old name to its `RETIRED` table in the same change.
   - **Maintainability (Principle 0):** if you took a quick-to-ship shortcut as an intermediate step,
     refactor it before marking the task done.
   - **Scope (Principle 3.5):** did you build what was actually agreed? If you narrowed the design,
     substituted a cheaper approximation, or finished only part of it, say so **in the summary** — and
     never report a measurement taken under that deviation without naming it.
   - **Solver adjoint & globalization (for any new iterative / coupled / fixed-point solver).** Confirm
     all three: (a) **the adjoint is an implicit-function-theorem solve on the converged residual, not the
     iteration unrolled onto the tape** — a single transpose solve, not a taped loop
     (`.claude/rules/solve.md`); (b) **the forward solve stops on a convergence test**, not a hard-coded
     iteration count (a fixed count is allowed *only* as an explicitly-labelled intermediate); (c) **it is
     globalized** to the standard of its neighbours (continuation / line search), with constant
     under-relaxation + floors treated as a stabilizer, not the globalization. If you ship an intermediate
     that does not yet meet (a)–(c), **file the deferred work as a tracked issue in the same change**.

2. **Lint, format & comment hygiene** — from the repo root:
   - `ruff check aquaflux aquaflux_ui tests docs` — must report no errors.
   - `ruff format aquaflux aquaflux_ui tests docs` — auto-applies formatting (CI will run `--check`).
   - **Comment-hygiene guard (the Comment Convention + Claude-Facing-File Reference Ban):** this grep
     must come back **empty** for any `.py` you touched:
     ```
     grep -rniE "c\+\+|fortran|\.claude|claude\.md|reference code|the reference|degroot|\.hpp|\.f90|design.note|briefing\.md|MeshObjectGroup" aquaflux aquaflux_ui tests --include="*.py"
     ```
     (`design.note`, not `-design-note`, so "the design note (S5)" is caught too. `the reference` is
     deliberately noisy — most hits are the legitimate "the reference state/operator/diagonal" — so read
     them, don't skim. Third-party author-year citations like "Ghia et al. (1982)" won't trip it.)
   - **README / docs surface:** when you touch `README.md` or anything under `docs/`, this grep must also
     come back **empty** (`package_structure.md` is excluded — a build-excluded internal catalogue, per
     `.claude/rules/docs.md`):
     ```
     grep -rniE "\.claude|claude\.md" README.md docs --exclude=package_structure.md
     ```
   - **Sibling-builder check (Principle 2's "two builders for one object").** Two functions constructing
     the same class from mostly the same parameters are one builder written twice, and the copies drift
     one keyword at a time — invisible to any diff-scoped check. Run this whenever you add a parameter to
     a builder or add a builder:
     ```
     tools/sibling_builders.py
     ```
     A **report, not a gate**; it always exits `0`, so read it. **Pairs on a healthy tree are normal —
     "zero pairs" is not the target.** Every entry in an `only here:` list must be a genuine property of
     that path, or it is drift (ask *whose property is this*). It is package-wide, unlike the diff-scoped
     Stale-Record Check, which cannot see a parameter that was only ever added.
     ⚠️ **A pair this tool cannot see reports as a clean tree — check it names both sides of your pair
     before reading its silence as clean.** It follows delegation (private tails, other modules, methods,
     typed receivers), treats any `@classmethod` returning `cls(...)` as a factory, and pairs across
     subpackages (pinned by `tests/unit/test_sibling_builders.py`). It is **still blind to a builder that
     returns a closure it defines** (`scalar_pseudo_transient_solve` returns `solve_scalar`); three
     assembler `build` pairs sharing only an incidental `BoundaryConditions` are reviewed noise. History of
     its blind spots: `.claude/notes/claude-md-log.md`.

   Ruff is pinned via the `lint` extra (`pip install -e ".[lint]"`). Not needed for docs/config-only
   changes touching no `.py` files.

3. **Tests** — are new tests needed?
   - New numerical operator → operator-level unit test (order of accuracy).
   - New public API → integration test against an analytical solution.
   - New analytical/published benchmark → validation test.
   - Bug fix → regression test.
   - **Every new test: verify it can fail before counting it** ("A test that cannot fail is not a test"):
     break the line it pins, confirm it goes red, restore the line.

   **Run the tier your change can reach — the fast gate is not the whole suite.** The `slow` and
   `validation` tiers run on **merge to main**, and on a PR only with the `full-ci` label; the always-on
   required check is just the fast gate (`-m "not slow and not validation"`). If your change could affect
   a `slow` or `validation` test — a shared solver / operator / scheme / helper those tests call, a
   deleted or renamed symbol, changed convergence/behaviour — **run those tiers locally**
   (`tools/fastgate.sh validation`, `tools/fastgate.sh slow`) or apply `full-ci` to the PR. The trap is a
   call path reached only through a validation-marked test: grep for the changed symbol across
   `-m slow`/`-m validation` tests and run the ones that hit it.
   `tools/fastgate.sh slow --changed` / `validation --changed` answer "could it reach them" mechanically.

   **⚠️ The unit, slow and validation shards are balanced by `.test_durations`; a stale or missing one
   silently unbalances them** (`pytest-split` falls back to a count split, so adding any test can push the
   heaviest shard past the 30-minute CI ceiling — which looks like a regression in your change). A weekly
   cron (plus `workflow_dispatch`, registered from the default branch only) re-records it in a
   metadata-only PR; label a PR **`refresh-durations`** to record it on that branch. The weekly merge
   rewrites the whole file, so **every tier that splits by it needs its own refresh job** (hence
   `refresh-unit-durations`). The fast integration tier is deliberately count-split (homogeneous tests):
   its job points pytest-split at a missing file, while `refresh-fast-durations` still records it, the
   distributed and the `tests/ui` tests, for what a scoped run costs.

4. **Documentation sync (binding — this is how the docs stop drifting).** A code change is **not
   complete** until every file that *describes* the changed code is updated in the **same change**.
   **Run the Stale-Record Check above first:** this item covers the file that *describes* what you
   changed; that one covers everything your change silently made FALSE elsewhere. When you rename a
   symbol, change a signature/default, move a file, add a dependency, or change behaviour, update each of
   these that applies (grep the repo for the old name/path):
   - **The docstrings and comments in the `.py` files** — in the code you changed *and in every other
     module that describes the same behaviour*. In-code prose is the first thing a reader trusts, and only
     this item covers it. **Grep for the claim, not only for the symbol** — this drift renames nothing, so
     the identifier grep is blind to it (why: "Post-Change Checklist 4: the preconditioning-side drift").
   - **The matching `.claude/rules/*.md`** — the per-subsystem design record: binding decisions,
     interfaces, class/function names, file paths, and `BUILT` / `Not yet built` status. Inline the fact
     rather than pointing at any private note.
     ⚠️ **"Matching" means the rule whose `paths:` glob covers the file you edited — check the
     frontmatter, do not go by subject.** A rule that does not glob the file it governs never auto-loads
     for the person editing it, so binding prose put there is invisible exactly when it is needed (why: "Post-Change Checklist 4: the wrongly routed binding prose"). Read the **frontmatter**:

     ```
     for f in .claude/rules/*.md; do sed -n '/^paths:/,/^---/p' "$f" | grep -q "<file you edited>" && echo "$f"; done
     ```

     ⚠️ **A plain `grep -l "<file>" .claude/rules/*.md` is not the same question** — it matches prose
     mentions as well as globs, so it returns rules that do not load for that file. Before trusting a
     command's output, confirm it answers the question you are asking and not a neighbouring one.
   - **`CLAUDE.md`** — architecture decision, public-API or package-structure change, new dependency, or
     workflow/tooling change.
   - **`README.md`** — public API, install/dependencies, examples, or the feature list.
   - **`docs/`** — any Sphinx page whose prose or cross-references the change touches.

   Rule of thumb: if a reader of one of these files would now be **misled** by what it says, fixing it is
   part of your change, not a follow-up. The committed **`.githooks/pre-commit`** reminder surfaces this
   whenever a commit touches `.py` code.

5. **CHANGELOG.md** — once one exists (add at first release-worthy change): user-visible API/behaviour
   changes only, under `[Unreleased]`.

If the answer to any of the above is yes, make those updates as part of the same task before marking it
complete.
