---
paths:
  - "aquaflux/turbulence/**"
---

# Rules — `aquaflux/turbulence/` (RANS closure: k–ω SST + the segregated coupling)

> **Provenance boundary (binding).** This file cites the internal design record
> (`turbulence-design-note.md`) and the precursor codes to inform *your* understanding — that
> is its job, and why it loads into your context. Per the root `CLAUDE.md` **Comment
> Convention**, none of that provenance may reach the shipped surface (`.py`
> comments/docstrings, `docs/`): cite the *math*, never the reference code, the `.claude/`
> rules, the design notes, or the author's own papers.

The k–ω shear-stress-transport (SST) closure and the loop that couples it to the coupled p–U
flow. The forward coupling is **segregated** (an outer Picard loop, decided with the author) —
but segregation is a *forward-solve strategy only*; the differentiable promise still requires
the adjoint of the **unfrozen coupled residual**. Governed by the root `CLAUDE.md` Engineering
Principles; the flow block it feeds is `.claude/rules/flow.md`, and the Newton / linear-solve
adjoint machinery it must reuse is `.claude/rules/solve.md`.

## Status — BUILT (segregated forward solve **and** monolithic coupled solve + coupled adjoint)
- **`sst.py` — `SSTModel`.** Menter's SST constants and the quantities derived directly from
  them (the F₁/F₂ blend, the eddy-viscosity limiter).
- **`strain.py`** — the strain-rate magnitude `S = sqrt(2 S_ij S_ij)` the production terms read.
- **`sources.py`** — the k and ω production / destruction / cross-diffusion terms as
  `VolumeSourceFn` volume-source operators (the transport equations reuse the shared advection
  and diffusion flux operators; only the sources are turbulence-specific).
- **`transport.py` — `SSTTurbulence`, `SSTClosureFields`.** Assembles the k and ω scalar
  transport residuals on the flow's Rhie–Chow mass flux, with μ_t a **frozen per-cell field**
  recomputed once per outer sweep.
- **`preconditioner.py`** — the convection-diffusion AMG preconditioner for the stiff k/ω scalar
  Krylov solves at high Reynolds number (the scalar analogue of the velocity-block work).
- **`boundary.py`** — inlet/wall closures for k and ω over the generic scalar boundary machinery.
- **`driver.py` — `solve_segregated`.** The outer Picard loop: μ_t → flow solve → k solve → ω
  solve, with under-relaxation and positivity floors as the stabilizers, and injected
  `solve_flow` / `solve_scalar` so the driver is pure orchestration. The loop **stops on the coupled
  Picard increment** (`_relative_change` — the largest per-field relative L2 change over a sweep <
  `rtol`), with `max_sweeps` only a backstop; the outer under-relaxation is the **SER ramp**
  `_sweep_relaxation` (opens from the `relaxation` floor toward `relaxation_max` as that increment
  falls, constant when `relaxation_max is None`). Hitting `max_sweeps` without converging warns.
- **`coupled.py` — `CoupledRANS`, `solve_coupled` (Option 2, the target engine).** The monolithic
  residual `R(u, p, k, ω)` over the flat `[flow…, k, ω]` state (`CoupledRANSLayout`, whose `unpack`
  yields the momentum block's own `[u,p]` sub-vector so `MomentumContinuity` runs on it unchanged),
  with **nothing frozen**: μ_t, the strain `S(u)`, the Rhie–Chow flux, and the closure are live, so
  one Newton solve sees the exact cross-block Jacobian. Globalized by `coupled_continuation`
  (a block `CoupledShiftPolicy` = velocity `a_P` shift ⊕ the k/ω transport-diagonal shifts, and a
  block-diagonal preconditioner gluing `BlockPreconditioner` to the two scalar CD-AMGs; the AMG
  hierarchies + numpy-built scalar shift diagonals **frozen at a reference state** off-jit à la
  `reused_flow_solve`, the velocity `a_P` live). Handed to `ImplicitNewtonSolver`, it gives the
  **exact coupled adjoint** (§5) — a single transpose solve on the unfrozen `R_coupled`. The ω wall
  rows are `FixedValueCells`. `CoupledRANS.build` pre-resolves the k/ω boundaries so the per-eval
  assembler rebuild's `resolve` is an idempotent no-op (else a dynamic-shape `nonzero` on traced mesh
  labels breaks the jit). **Direct k/ω variables** (log-variables held in reserve, below); positivity
  under a full Newton step is carried by the pseudo-transient shift + divergence guard, no in-residual
  floor. FD-verified: coupled ‖R‖→machine-zero, agrees with the segregated fixed point, adjoint
  matches finite differences.
- **`initialization.py` — `hybrid_initialize` (cold-start, the reason `solve_coupled` self-starts).**
  The monolithic Newton is a *local* method: from a raw cold start (`u=0`, uniform k/ω) it **stalls** —
  the near-wall ω fixation alone injects a `~60ν/(β₁d²)` jump, and a uniform interior is far from a
  consistent field the inner solve can precondition. `hybrid_initialize(momentum, turbulence)` builds a
  cheap physical IC (a few linear Laplace solves): **potential-flow velocity** (`flow/initialization.py`
  `potential_flow`), **Laplace-smoothed k** (harmonic interpolant of its BCs), and **ω** =
  boundary-propagated interior with the near-wall cells set to the analytical wall value (a *Laplace*-ω
  over-diffuses that large value and slows the solve — set only the wall cells). From this IC the coupled
  Newton converges from nothing (~10–15 steps, FD-verified). `solve_coupled(coupled)` with no initial
  state calls it automatically; the segregated pre-smooth is no longer required to reach the basin (still
  available as a fallback). **Do not init with an exactly symmetric velocity** — a perfectly symmetric
  `u` (e.g. exact plug, `u_y≡0`) hits a measure-zero degeneracy in the coupled inner solve that stalls;
  the potential flow's discrete-gradient roundoff (`|u_y|~1e-10`) lifts it, and any perturbation ≥1e-10
  converges. The IC is a forward device (the converged-state adjoint is IC-independent); when
  differentiating, pass an explicit state built outside `jax.grad`.

**Issue #69 — CLOSED path (do not re-derive without reading it):** all three planned steps shipped —
scalar continuation (#73), Option 1 hardening (convergence stop + adaptive relaxation), and Option 2
(the monolithic coupled residual + its IFT adjoint, the target engine). The segregated loop is
**retained as a forward pre-smoother / fallback**, not the sensitivity model; for gradients use the
coupled `solve_coupled` (its adjoint is exact) — never differentiate `solve_segregated` (forward-only,
unrolls the Picard sweeps, which §5 forbids). The remaining held-in-reserve item is the **log-variable
fallback** (below), promoted only if a stiff high-Re case shows the direct coupled form non-robust.

## Binding decisions

- **Segregated forward, coupled adjoint (design note §5 — binding) — BUILT via `solve_coupled`.**
  Segregation is a **forward-solve strategy only**. For exact sensitivities the adjoint is the
  implicit-function-theorem solve on the full **unfrozen** coupled residual
  `R_coupled(k, ω, U, p; params) = 0` at the converged state — the `solve/` two-level
  implicit-diff machinery — **not** a differentiation of the Picard iteration. This is now realized:
  `coupled.py`'s `CoupledRANS.residual` **is** that unfrozen `R_coupled`, and `solve_coupled` hands it
  to `ImplicitNewtonSolver`, whose adjoint is a single transpose solve (FD-verified). At the fixed
  point the frozen fields equal the live values, so the coupled residual is satisfied and its
  adjoint is exact; the segregated outer loop is a forward convergence device that is **absent from
  the sensitivity model**. Differentiate **`solve_coupled`, never `solve_segregated`** (the latter is
  forward-only and its docstring says so). When building the coupled continuation for a differentiated
  solve, construct it **outside `jax.grad`** (concrete preconditioner params) and pass it in — see the
  flow preconditioner's same constraint.

- **Never unroll the outer loop onto the differentiation path.** A fixed-count `for` over sweeps
  that is differentiated directly is exactly the failure `solve.md` names ("no loops on the
  differentiation path"). If the coupled solve is not yet wrapped in the coupled-residual IFT
  adjoint, it is **not done** — it is an intermediate step (Principle 0), and the deferred adjoint
  must be filed as a tracked issue at merge time, not left implicit.

- **Convergence-based outer stop, not a fixed sweep count — BUILT.** The loop tests the coupled
  Picard increment and stops on it (`rtol`), with `max_sweeps` only a backstop and a warning when the
  cap is hit unconverged. Do **not** reintroduce a hard-coded `sweeps` count. The increment measure
  is the residual-agnostic per-field relative change, not a raw combined norm (the field scales
  differ by orders of magnitude).

- **Globalize the outer loop and the scalar sub-solves like everything else.** The flow block is
  globalized by pseudo-transient continuation; the k/ω transport sub-solves and the outer coupling
  must reach the same standard (a scalar `ShiftPolicy` continuation on the transport diagonal for the
  sub-solves; adaptive under-relaxation — **the SER ramp is built** — with Aitken/Anderson or a
  monolithic coupled residual as the further steps, for the loop). Constant under-relaxation plus
  positivity floors is the *stabilizer of last resort*, not the globalization.

- **Positivity floors must be inactive at convergence (adjoint honesty, design note §3.3 —
  binding).** `k ← max(k, k_floor)`, `ω ← max(ω, ω_floor)` and the `CD_kω` / F-blend floors have zero
  gradient in the clamped region; they pollute the sensitivity **unless inactive at the fixed point**
  (`k, ω > floor` everywhere, which holds for any properly resolved RANS field). State this precondition
  in code and **check it**: if a case converges with a floor active, the sensitivity through that cell is
  wrong — surface it, do not ship it. (Log-variable `k = e^{k̃}` is the held-in-reserve structural fix.)

- **Frozen coupling data rides as injected pytree leaves** (μ_t, the frozen ∇u, mdot), the same
  blessed mechanism the coupled solver already uses to inject `mdot` — no new freezing mechanism, and
  no re-coupling μ_t ↔ (k, ω) inside a residual via a `Calculated` property in the segregated path.

## Testability seam
- `solve_segregated` takes **injected** `solve_flow` / `solve_scalar` closures, so the loop is
  tested against trivial stub solvers (e.g. identity / one-step) with a known fixed point — no full
  coupled solve needed to test the orchestration.
- Every turbulence operator (sources, strain, transport) ships an operator-level unit test on an
  analytic field (Principle 1), independent of the coupled solve.
- **The coupled solve needs an adjoint-correctness gate**, not only a smoke test: a test that
  `jax.grad` through the converged coupled turbulent solve is **iteration-count-independent** (the
  coupling analogue of Gate C; see the root `CLAUDE.md` Testing Architecture). An existence check
  ("stays stable, fields positive, μ_t active") does not establish the adjoint.

## Post-change
Keep this file's Status and Binding decisions true as the coupling globalization (issue #69) lands —
per the root `CLAUDE.md` Post-Change Checklist's Documentation-sync item.
