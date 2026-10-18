# Module Review Rubric

> Moved verbatim from the root `CLAUDE.md` (2026-10-18), which keeps a pointer: it applies only when a
> module review is asked for, so it is read then rather than loaded into every session.

When asked to review a module (or auditing one before commit), run **two passes**. The second is the one
that gets skipped, and it catches structural problems a per-file checklist is blind to — a shared type in
the wrong module, a bundle that unions every consumer's needs, a decomposition that won't survive the
next feature. It needs **fresh eyes** (see "How to run it").

## Pass 1 — Local smells (per file)
The Engineering Principles as a checklist, plus correctness and user-facing clarity:
- **Testability (Principle 1):** can each unit be tested in isolation with small inputs, no global
  state? Is there an operator-level test (order-of-accuracy on an analytic field)? Can each test actually
  fail — checked against a broken version of the code, not just read ("A test that cannot fail is not a test")?
- **Duplication (Principle 2):** logic implemented more than once; copy-paste-modify; a formula that
  should be imported from its one home.
- **Encapsulation (Principle 3):** raw arrays passed instead of the object; a whole `Mesh` where
  `face_cells` suffices; duplicate accessors; god-methods; inlined formulas that have a home.
- **Vocabulary (Principle 3.7):** a name that already means something else in the package, a quantity
  whose name runs opposite to a neighbour's, a count that does not name its unit, or a concept spelled
  differently from its siblings.
- **Correctness & clarity:** errors/edge cases; anything a user would find hard to understand or misuse.

## Pass 2 — Decomposition & extensibility (step back; question the structure)
Do not evaluate the code *within* the existing structure — question the structure itself:
- **Placement / cohesion.** Is each type/function in the right module, each module one responsibility?
  *Smell:* a type used mainly by module B living in module A because it was first written there (e.g. a
  face-flux state/interface in `diffusion.py`). **Sharpest form: generic machinery in a physics package**
  — a driver, step assembly, measure or settings value naming no physics, living in `turbulence/` because
  turbulence was first (Principle 3.6).
- **God-objects / union bundles.** A structure that unions every consumer's needs — coupling unrelated
  consumers, growing with each operator, needing placeholder defaults (`psi = ones`)? A "context/state"
  object carrying fields only one consumer reads?
- **Dependency direction.** Sibling modules importing each other for shared types (they should share a
  base/contract module)? An import that invites a cycle?
- **Extensibility against the stated direction.** As the system grows (YAML/DSL-driven assembly; N
  operators / schemes / BCs; a properties model), what will bloat, need placeholder defaults, or need
  re-cutting? Is each strategy **self-describing about its own inputs** so a declarative assembler can
  gather per active term?
- **Self-justification.** For each shared type, state *why* it lives where it does. "That is where it was
  first written" means it is probably misplaced.

## How to run it
Run **both** passes. Pass 2 needs **fresh eyes**: spawn a reviewer (agent) not anchored on the current
structure, and prompt it with these Pass-2 dimensions **explicitly** — a generic "find issues" prompt
reverts to local-smell matching. Scope each review to a named module (or a small set), evaluate against
the future direction, and **verify each finding yourself** before acting (agents surface
plausible-but-wrong structural claims too). A finding is real only if you can name the concrete failure
it causes now or the concrete bloat/re-cut it forces later.
