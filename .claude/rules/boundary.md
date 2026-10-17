---
paths:
  - "aquaflux/boundary/**"
---

# Rules — `aquaflux/boundary/` (weak boundary-face-value closures)

> **Provenance boundary (binding).** This file may cite the C++/Fortran precursors; none of that may
> reach the shipped surface (`.py`, `docs/`) — root `CLAUDE.md` Comment Convention.

Patch-based boundary conditions as **weak face-value closures**. Dated measurements and investigation
detail: `.claude/notes/boundary-log.md`.

## Status — BUILT

- **`conditions.py`** — `BoundaryCondition` (`eqx.Module` base; `face_value(phi_owner, grad_owner, d,
  normal, gamma_owner, face_centroid)`) → `Dirichlet`, `DirichletField` (`field_fn(x_ip)`, for manufactured
  solutions and varying walls), `ZeroGradient`, `Neumann`, `Convective` (Robin
  `Gamma dphi/dn = h(Tinf − phi_ip)`, `h`/`t_inf` differentiable). All share the tangential correction
  `_tangential_correction`: `corr = grad·(d − (d·n)n)` (zero on orthogonal grids).
- **`collection.py` — `BoundaryConditions`**, a `{patch: closure}` `equinox.Module` built like a
  `PropertyModel` and passed to an assembler's `build` as the single `boundary` argument (not a bare dict).
  Two states: constructed **unbound** (`faces is None`), then bound by `build` via
  `resolve(mesh.face_patches, mesh.face_cells)` to a copy whose `faces` holds each patch's face indices.
  The lookup has data-dependent shapes, so it **must run off the jit path** (resolve once and store;
  `resolve` returns early on a bound collection, which keeps it off the trace). `apply(face_cells, init,
  closure)` is the one patch fold, used by both `ResidualAssembler` and `MomentumContinuity`; the closures
  are opaque, so it is generic over single-field `BoundaryCondition` and multi-field `FlowBoundary`. A leaf
  module (imports `jax` + mesh connectivity), so `discretization` and `flow` depend on it one way.
- Closures feed `ResidualAssembler` both as the flux's boundary value and as the gradient reconstruction's
  boundary input (value at zero gradient + gradient weight, absorbed by `schemes.boundary_gradient_block`).
  ⚠️ **A new closure must stay affine in the owner gradient**, or the absorbed form becomes a linearization.

## Binding decisions

- **A BC is a special face interpolator imposed weakly** through the boundary-face flux (the C++
  face-interpolator model). Do **not** use the Fortran strong absorption into the block; the weak closure
  composes with the face-flux substrate and AD. Closures are immutable modules constructed with their
  parameters; no global patch registry.
- **`BoundaryCondition` is a single-field closure; multi-field bundles compose it, never subtype it.**
  `FlowBoundary` (`flow/boundary.py`) bundles velocity/pressure/`mdot` closures and is not a
  `BoundaryCondition` — no shared base (`.claude/rules/flow.md`). `_tangential_correction` stays
  single-homed here; the flow closures (`velocity_face`/`pressure_face`) take the owner gradient and
  displacement and reach it through the same two-pass fold.
- **`closes() -> tuple[str, ...]`** declares which fields a closure closes. Scalar closures return
  `HOST_EQUATION_FIELD` (empty on purpose: the closure does not know its host equation's name, and a name
  here could disagree with the assembler's); `FlowBoundary.closes()` returns `FLOW_FIELDS = ("velocity",
  "pressure", "mdot")`. The check is `collection.refuse_a_closure_that_closes_other_fields(boundary,
  closes, caller)`, **deliberately not inside `resolve`** (the collection is generic; arity is the
  assembler's knowledge). Called by `MomentumContinuity.build`, `ResidualAssembler.build`,
  `ScalarTransport.build` and `SSTTurbulence.build` (per scalar equation). It names every wrong patch at
  once, and refuses an object declaring nothing rather than skipping it.
- **`requires_coefficient() -> bool`** — `True` on `Neumann` and `Convective` (they read `gamma_owner`).
  `ResidualAssembler.build` then adds its `coefficient` to what the `PropertyModel` must `require`, so a
  mistyped `coefficient=` fails at build instead of reading `Gamma = 0` and dividing to NaN/inf. The zero
  fallback stays legal for closures that do not declare it.
- **`face_value` deliberately does not take a `FieldContext`.** It receives per-face values already
  gathered once by the central fold (`ResidualAssembler.boundary_values`); a context would make every
  closure re-derive that gathering. Unlike `FaceFluxOperator`/`Limiter`, which gather from the whole mesh.
- **Every prescribed boundary number is a floating array leaf, converted at construction.**
  `Dirichlet.value`, `Neumann.flux`, `Convective.h`/`t_inf`, `PressureOutlet.pressure` and constant
  `VelocityInlet`/`MovingWall.velocity` go through `conditions.as_float_leaf`
  (`eqx.field(converter=...)`). Position-dependent values go through `conditions.as_position_function`:
  an `equinox.Module` is kept (its array fields are leaves, so differentiable); a plain function is wrapped
  in the static `StaticFunction`. Static/float values gave no cotangent under `eqx.filter_grad`, raised
  through `RootSolver`, and recompiled per value.
  - ⚠️ **Do not just drop `static=True` on a field that may hold a plain function**: a function leaf makes
    even a forward `RootSolver.solve` raise "not a valid JAX type" (its `custom_vjp` flattens `theta`) and
    breaks the distributed `jax.tree.map(jnp.stack, ...)`. Plain-function profiles are common (Poiseuille,
    skewed Couette, the `bfs3d_species` injector).
  - Pinned by `tests/unit/test_boundary_value_leaves.py` and
    `tests/integration/test_boundary_value_gradients.py`.
- **Every boundary face must be covered — `resolve` refuses a map that leaves one out.** `apply` folds
  into zeros, so an omitted patch would be an unchosen zero Dirichlet value. `resolve` asks
  `FacePatches.uncovered_boundary_faces(conditions, face_cells)` and lists **every** uncovered patch with
  its face count.
  - `"boundary"` (faces no named patch claims) is **not** exempt — it is the most common omission; name it.
  - `"interior"` is never reported and naming it is legal (`combine_face_values` ignores interior rows).
  - Distributed padding faces live in the reserved `"padding"` group and are exempt
    (`.claude/rules/parallel.md`).
- **One scatter for all patches.** `apply` evaluates each closure separately but writes all with one
  `init.at[concat(faces)].set(concat(values))`, each value `broadcast_to` its patch shape first. Do not go
  back to per-patch `.set`: P sequential scatters and a program growing with P. Concatenate inside `apply`,
  not in `resolve` — `parallel/distributed.py` rewrites the per-patch `faces` dict when it pads, and a
  second stored copy would drift.
  - Patches are disjoint, except distributed padding: `_uniform_boundary_faces` fills every padded tail
    with the null face, which repeats. ⚠️ So `unique_indices=True` must **not** be passed.
  - Pinned by `tests/unit/test_boundary_collection.py` (`test_apply_writes_every_patch_in_one_scatter`;
    the interleaved-unequal-patches fixture is the only one that catches a faces/values order mismatch).

## Testability seam
Each closure is unit-tested on a single face with known cell value and geometry against its closed form
(`tests/unit/test_boundary.py`); no mesh, no solve.

## Open design question
How patches are declared in the eventual DSL is deferred; the runtime closure model (weak, face-value) is
decided.
