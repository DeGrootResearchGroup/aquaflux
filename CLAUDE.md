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
| `.claude/rules/solve.md` | `aquaflux/solve/**` | Newton on the residual, linear solve with implicit differentiation / `custom_vjp`, the preconditioner risk |
| `.claude/rules/flow.md` | `aquaflux/flow/**` | coupled p–U block: momentum (reusing advection/diffusion) + Rhie–Chow continuity, differentiated `a_P` (frozen only in the preconditioner), monolithic AD-Jacobian solve |
| `.claude/rules/turbulence.md` | `aquaflux/turbulence/**` | k–ω SST closure + the segregated flow–turbulence loop: segregated forward / coupled adjoint, outer-loop globalization, positivity-floor adjoint honesty |
| `.claude/rules/io.md` | `aquaflux/io/**` | mesh import: the `MeshReader` strategy + the OpenFOAM polyMesh reader (ASCII); parse→assemble→collapse seams, empty-patch 2D collapse (a `mesh/` transform), reserved-name guard |
| `.claude/rules/parallel.md` | `aquaflux/parallel/**` | distributed memory: graph partitioners, the `PartitionedMesh` owned+halo decomposition, uniform-shape padding, and the `shard_map` residual that runs an *injected* assembler per device (never a re-implementation) |

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
- **Concrete trigger:** *If a change would require editing the same formula in more
  than one file, the formula is in the wrong place — consolidate it first, then
  make the change once.*

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
  state layout is a missing value object (`BlockStateLayout`).

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

### 4. No compatibility shims before release (pre-release policy — remove this principle at 1.0)

The project is **pre-release: there are no external API consumers, so breaking API changes are free.**
When a refactor changes a public surface, **change the surface and update every call site — do not
preserve the old one.**

- **No thin adapters kept only to preserve an API.** A wrapper class or forwarding function that exists
  solely to re-expose a refactored abstraction under its former shape is dead weight: delete it and
  point callers at the real object. (E.g. a `PseudoTransientStep` *is* the `ForwardStep`, so a
  `PseudoTransientContinuation` class that only delegated `stepper`/`default_solver`/… was removed in
  favour of a `momentum_continuation` factory returning the engine directly.)
- **No deprecation shims, aliases, or back-compat branches.** No `old_name = new_name` re-exports, no
  `if legacy_arg is not None` compatibility paths, no keeping a parameter alive "so nothing breaks."
  Rename/retype/delete in one change and fix the callers and tests in the same change.
- A **builder function or factory is not a shim** — it constructs and returns the real object (like
  `momentum_continuation` or `reused_flow_solve`). What this bans is the *delegating wrapper* that adds
  a layer over an object already fit to use directly.

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
| Linear solves | lineax (fallback `jax.scipy.sparse.linalg`) | JAX-native, implicit differentiation of the solve |
| Adjoint of the coupled solve | `custom_vjp` around the linear solve | exact, memory-flat gradient independent of iteration count |
| Transient integration | Diffrax | JAX-native, shared with aquakin |
| Mesh | static connectivity arrays + `segment_sum` scatter | XLA-friendly graph/message-passing layout |
| Eventual model format | YAML → AST (deferred) | the DSL is the **last** layer; not a dependency yet |

The coupled-block preconditioner is the **top research risk**. The first build-on candidate
to evaluate is `jaxamg` — do not assume it; verify its adjoint is implicit-diff and that it
preconditions a *block* system. (See `.claude/rules/solve.md` for the chosen block-triangular
SIMPLE-type direction and the known traps.)

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
**connectivity API** — `mesh.face_cells` (`FaceCellConnectivity`: `gather_owner` /
`gather_neighbour`, `scatter` / `scatter_conservative` / `scatter_symmetric`) and
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
or scale a `(..., dim)` field (it imports only `jax.numpy`, so any subsystem may use it). **Preference (binding): keep vector math
readable — reach for these helpers instead of open-coding `jnp.sum(a * b, axis=-1)` or
`s[..., None] * v`.** The raw axis/broadcast bookkeeping obscures the math and drifts; the named
helper states the intent, and gives one home to change (Principle 2). Rank-3 tensor algebra
(Hessian outer products, `einsum`) stays explicit — the helpers target rank-2 vector fields.

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
   → mesh.face_cells.gather_* → compute flux → mesh.face_cells.scatter_*
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

`fastgate.sh` wraps `pytest -q -m <tier>`, writing the run to a file and reporting pytest's **own**
exit status with the summary line found by pattern. Invoke `pytest` directly if you need something the
wrapper does not pass through, but never through a pipe: this suite prints library shutdown chatter
*after* the summary, so `| tail -n` shows the chatter, hides the result, and returns `tail`'s exit
status — which is `0` however the run went.

### Canonical tests (must always pass once implemented)

- **Primary field (analytical):** plane wall with convection vs the closed-form θ
  (Gate A).
- **Sensitivity (analytical) — the point of the project:** `jax.grad` of the converged
  solver w.r.t. `Bi` vs the closed-form ∂θ/∂Bi (Gate B). Every integration suite must
  include an explicit test that `grad` flows through `solve()` without error and without
  NaNs.
- **AD-exact-linearization (skewed mesh):** on a non-orthogonal mesh, the linear problem
  converges in one Newton step (Gate C) — the concrete improvement over the reference.
- **Coupled-solve adjoint (iteration-count-independent):** for any iterative / coupled / fixed-point
  solver (the segregated flow–turbulence loop, and any future coupled system), `jax.grad` through the
  converged solve must match a reference gradient (finite difference or the closed form) **and be
  independent of the forward iteration count** — the coupling analogue of Gate C. This is what proves
  the adjoint is the implicit-function-theorem solve on the converged coupled residual, not the outer
  loop unrolled onto the tape. An existence/stability smoke test ("stays stable, fields positive")
  does **not** establish this — it must be its own explicit test.
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

Three operations have one correct invocation. Use it.

| to do this | run this |
|---|---|
| run a validation case or any long solve | `validation/run_case.sh <script.py> [--wait]` |
| see what a long run is doing | `validation/run_case.sh --status` · `tail -f <its log>` |
| run a test tier | `tools/fastgate.sh [fast \| slow \| validation \| all]` |

Each exists because the hand-rolled version fails *silently* — it produces a plausible answer that is
wrong and says nothing about it. `pytest … | tail -n` reports the exit status of `tail`, which is `0`
whatever pytest did. A case launched with a bare `python … &` dies with its parent, or outlives a
cancellation you believe succeeded. `pgrep -f <script>` matches the watcher's own command line, so a
waiter waits on itself forever. Every one of those has happened here and cost hours.

`run_case.sh` puts in one place everything the surrounding rules used to ask you to remember:
unbuffered output **redirected, never piped**, to a timestamped log; the machine held awake; a
free-memory and load pre-flight; a refusal to start a second case; and a run-file recording pid,
worktree, branch, commit and case settings. Its most valuable output is not the log — it is that
**"is this run mine, and what is it testing?" has a written answer**, a question that has been got
wrong from the process table alone.

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
cross-reference must resolve and every page must sit in a toctree. Build locally with the
`docs` extra (`pip install -e ".[docs]"`, then `cd docs && make html`, or
`sphinx-build -b html -W docs docs/_build/html` to match the strict RTD build).
**CI builds the docs on every PR** — the `docs` job runs the same `-W` build Read the Docs
runs, and the required `fast gate` depends on it, so a broken cross-reference or a page
missing from a toctree blocks the merge instead of turning the published site red afterwards.
Build locally with `-W` as well when you touch a docstring of a documented subpackage, a
`docs/` page, or a cross-reference — the CI job is the backstop, not the first line.

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
  state? Is there an operator-level test (order-of-accuracy on an analytic field)?
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
  implementation — e.g. a face-flux state/interface living in `diffusion.py`).
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
git diff --cached | grep -E "^-" | grep -oE "\b_?[A-Za-z][A-Za-z0-9_]{3,}\b" | sort -u > /tmp/touched
grep -rnFf /tmp/touched CLAUDE.md .claude/rules/ README.md docs/
```

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
     grep -rniE "c\+\+|fortran|\.claude|claude\.md|reference code|the reference|degroot|\.hpp|\.f90|-design-note|briefing\.md|MeshObjectGroup" aquaflux tests --include="*.py"
     ```
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
   Ruff is pinned via the `lint` extra (`pip install -e ".[lint]"`). Not needed for
   docs/config-only changes touching no `.py` files.

3. **Tests** — are new tests needed?
   - New numerical operator → operator-level unit test (order of accuracy).
   - New public API → integration test against an analytical solution.
   - New analytical/published benchmark → validation test.
   - Bug fix → regression test.

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

4. **Documentation sync (binding — this is how the docs stop drifting).** A code change is
   **not complete** until every file that *describes* the changed code is updated in the **same
   change** — docs move with code, never "fix it later." **Run the Stale-Record Check above first:**
   this item covers the file that *describes* what you changed, that one covers everything your
   change silently made FALSE elsewhere, which is the larger and more dangerous set. When you rename
   a symbol, change a signature/default, move a file, add a dependency, or change behaviour, update
   each of these that applies (grep the repo for the old name/path to find every mention):
   - **The matching `.claude/rules/*.md`** — the per-subsystem design record: binding decisions,
     interfaces, class/function names, file paths, and `BUILT` / `Not yet built` status. Inline
     the fact rather than pointing at any private note.
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
