---
paths:
  - "aquaflux/boundary/**"
---

# Rules — `aquaflux/boundary/` (weak boundary-face-value closures)

> **Provenance boundary (binding).** This file cites the C++/Fortran precursors and the
> design notes to inform *your* understanding — that is its job, and why it loads into your
> context. Per the root `CLAUDE.md` **Comment Convention**, none of that provenance may
> reach the shipped surface (`.py` comments/docstrings, `docs/`): cite the *math*, never the
> reference code, the `.claude/` rules, the design notes, or the author's own papers.

Patch-based boundary conditions as **weak face-value closures**. Governed by the root
`CLAUDE.md` Engineering Principles.

## Responsibility
- Per-patch boundary closures that supply a boundary-face value (and its linearization
  via AD): Dirichlet (constant value), Neumann (constant flux), zero-gradient,
  convective/Robin. Milestone-0 targets exactly these four (the plane-wall case needs
  symmetry + convective; the C++ conduction case needs all four).
- **The named boundary-condition collection** (`collection.py` — `BoundaryConditions`, an
  `equinox.Module`): a `{patch: closure}` mapping constructed exactly like a `MaterialModel` —
  `BoundaryConditions({"left": Dirichlet(1.0), "right": ZeroGradient()})` — and handed to an
  assembler's `build` as the single `boundary` argument (the material-model analogue; **not** a bare
  dict, and **not** a `.resolve(...)` call at the site). It has a **two-state lifecycle**: constructed
  *unbound* (it holds patch names + closures, `faces is None`), then **bound to a mesh** by `build`
  via `boundary.resolve(mesh.face_patches)` → a copy whose `faces` dict carries each patch's
  boundary-face indices. The name→index lookup is data-dependent (dynamic `jnp.where` shapes), so it
  **must** run off the jit path — hence resolve-once-and-store rather than the jittable
  `MaterialModel.evaluate(cell_zones)`-per-call pattern. `apply(face_cells, init, closure)` is the one
  iterate-over-patches → gather owner → `.at[faces].set(closure(bc, faces, owner))` fold, run inside
  the residual (raises if still unbound). Owned here so **both** the scalar `ResidualAssembler` and the
  coupled-flow `MomentumContinuity` carry a single `boundary: BoundaryConditions` field — **not** three
  parallel `names`/`conditions`/`faces` tuples — and compose `.apply` instead of re-open-coding the
  loop (the closure differs — a scalar face value vs. a flow velocity/pressure/mass-flux — the
  iteration does not). `resolve` takes `mesh.face_patches`, `apply` takes `mesh.face_cells` (smallest
  sufficient collaborators, not the whole `Mesh`); the held closures are opaque, so the object is
  generic over the closure type (single-field `BoundaryCondition` or the multi-field `FlowBoundary`).
  It is a leaf module (imports only `jax` + the mesh connectivity types), so `discretization` and
  `flow` both depend on `boundary` one-way.

## Status — BUILT (`conditions.py`)
`BoundaryCondition` (interface,
`face_value(phi_owner, grad_owner, d, normal, gamma_owner, face_centroid)`) → `Dirichlet`,
`DirichletField` (position-dependent, `field_fn(x_ip)` — for manufactured solutions and
spatially-varying walls; used by Gate C), `ZeroGradient`, `Neumann`, `Convective`. The four
weak-flux closures share the tangential
correction `corr = grad·(d − (d·n)n)` (zero on orthogonal grids). `Convective` holds `h`
(the Biot number, non-dimensionalized) and `t_inf` as differentiable fields — it is the
Gate-B sensitivity target — and enforces the Robin balance `Gamma dphi/dn = h(Tinf − phi_ip)`.
Verified (`test_boundary.py`): each closure vs its closed form on a single face, the Robin
balance holds, and high-`h` convective → Dirichlet. The closures are consumed by
`ResidualAssembler` both as the flux's boundary value and (leading-order) as the gradient
reconstruction's boundary input.

## Binding decisions
- **A boundary condition is a special face interpolator** (the C++ model,
  `reference-code-findings.md` §A.6): it returns a face value the flux operator
  consumes, imposing the BC **weakly** through the boundary-face flux. **Do NOT** use
  the Fortran strong-absorption-into-the-block approach (§B.8) — the weak closure
  composes naturally with the Layer-0 face-flux substrate and with AD
  (`dsl-design-note.md` §7.2, briefing §3.2).
- **Same interface as an interior face interpolator**, so a boundary patch plugs into
  gradient reconstruction and the exterior-flux path uniformly (CLAUDE Principle 2 —
  one interpolation interface, not two). BCs are strategy classes subtyping the
  face-interpolation `Protocol`.
- Boundary-condition classes are **constructed with their BC parameters** (value, flux,
  `h`/`T_inf`, ...) and act on the boundary-cell state + face geometry (CLAUDE
  Principle 1). Immutable `equinox.Module`s; no global patch registry read inside a
  method.
- **`BoundaryCondition` is a single-field closure — the reusable primitive; multi-field bundles
  COMPOSE it, they do not subtype it.** The coupled-flow `FlowBoundary` (`flow/boundary.py`) is a
  bundle {velocity closure, pressure closure, mass-flux closure}; its velocity/pressure parts are
  `Dirichlet`/`ZeroGradient` applied to those fields, but the whole is not a `BoundaryCondition`
  (three coupled outputs; `mdot` has no single-field analogue). So do **not** add a shared base or
  make `FlowBoundary` inherit here — the intended coupling is composition (see `.claude/rules/flow.md`).
  The one place the two must not drift is the tangential-correction formula `_tangential_correction`:
  keep it single-homed here so any consumer (scalar or flow) reuses it. The flow zero-gradient
  closures currently drop `corr` (leading-order); reconciling that into the flow flux is the
  boundary-gradient two-pass fold-in, tracked in `flow.md`.

## Testability seam
Each BC closure is unit-tested on a single boundary face with a known cell value and
geometry, asserting the returned face value against the closed form (e.g. Robin blend
`1/(1 + h/k·d)`). No mesh, no solve.

## Open design question (from `dsl-design-note.md` §7.2)
How patches are declared in the eventual DSL is deferred; the *runtime* closure model
(weak, face-value) is decided now and is what this module implements.
