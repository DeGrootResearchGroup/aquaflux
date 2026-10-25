# Investigation log — `aquaflux/flow/`

> Split out of `.claude/rules/flow.md` (2026-10-17). **Lives in `.claude/notes/`, outside the
> auto-loaded `.claude/rules/` tree, so it never auto-loads.** It holds the dated investigation behind
> the coupled pressure–velocity block — measurements with their configurations, bug-fix records, and
> qualified findings — moved verbatim. See `flow.md` for the current, load-bearing status. Each heading
> names the `flow.md` section (and entry) the text came from.

## From "Status — BUILT" → `datum.py` (`PinnedPoint`, #500)

  - ⚠️ **The refusal found 15 fast-tier tests (in three files) building a closed domain with NO datum** — assemblers built
    "only for their geometry" or their viscosity, never solved, so the singular system was never met. They
    now carry one. A dry run of the slow + validation tiers (every build stopped just after the check)
    found no others. `pressure_pin_value` had no caller outside the build.

## From "Status — BUILT" → the gradient scheme bound once per solved field (#467)

    ⚠️ **On tetrahedra the first version shipped a singular pressure correction** (94 cells at
    `max|M2⁻¹|` 3.1e16 on the 2462-cell duct) and froze the laminar march; the cause was the probe,
    not the per-field split — see "The boundary-linearization probe" in `.claude/rules/schemes.md`.

## From "Status — BUILT" → `corr` reconciliation, two passes as `schemes.BoundaryClosure.reconstruct` (#58)

      **Gated on bit-identity, not on tests passing** (254 arrays: residual, every `flow_fields` entry,
      `a_P`, `jvp`, jitted residual, scalar assembler, coupled RANS residual and `jvp`, over 2D/3D ×
      compact/corrected/multiple-correction × Stokes/upwind/limited, perturbed meshes): all
      bit-identical **except the flow residual's reverse-mode `vjp`, which moves 1–4 ulp** — the
      per-component closures change the transpose's summation order. Forward values and the `jvp`
      (what a march and its Krylov solves use) do not move, so archived trajectories hold; adjoint
      gradients move at rounding level only.

## From "Status — BUILT" → `mass_flux` reads the patch's own boundary velocity (#319)

    - **⚠️ This changes the shipped default's answer on a non-orthogonal mesh with an outlet.** Before
      the fix, `PressureOutlet`'s through-flow term used the owner velocity, which is a **zeroth-order
      consistency error** at the outlet-owning cells (it does not shrink under mesh refinement: 8×8 →
      32×32 gave `max|div u| = 0.111 → 0.085 → 0.086` at the exact Stokes Couette field, and a
      converged solve stalled at `‖u − u_exact‖∞ ≈ 1.2e-3` regardless of mesh). Any archived
      bit-identity comparison on a skewed mesh with an outlet is void.

## From "Status — BUILT" → bulk-velocity constraint: The periodic seam

  - **The periodic seam is NOT a preconditioner problem — do not go looking there again.** A
    `reused_flow_solve` on a periodic mesh was suspected of a block-SIMPLE defect "across the seam";
    it was measured and the seam is clean. The offset is absorbed one layer down (`interpolation_factor`
    and `normal_distance` both route through `face_cells.neighbour_centroid`; the Schur matvec compares
    *field* values, which are periodic and need no offset), so: preconditioned GMRES converges in 9–18
    iterations on the periodic mesh and returns the same Newton update as unpreconditioned to `5e-8`;
    the velocity-AMG `boundary_diagonal` has no negative entries (min `-4e-16`); every shifted-step
    attempt is accepted with `|δ| ∝ 1/β`. **The real failure was the cold start**: from rest, a
    body-force channel's residual barely moves during the viscous spin-up (measured ratio `0.984` per
    step), so the feedforward SER schedule `β = β₀(‖R‖/‖R₀‖)^p` never relaxes, the pseudo-timestep stays
    tiny, and the march exhausts `max_steps` at ~`0.28‖R₀‖` — a crawl, not a stall. Two independent
    confirmations: `β₀ ≤ 0.1` converges to `1e-16` in 30 steps, and starting from `potential_flow`'s plug
    reaches `7e-13`. **Fix shipped = the velocity scale + plug start** (`flow/scales.py`), not a change
    to the schedule. A cold rest start on a body-force domain still crawls and is left to fail loudly
    (`RootSolver` raises); revisiting that means revisiting cross-step SER, which was measured
    ~24% slower on the inlet channel and rejected.

## From "The flow march" → the tetrahedral duct re-run (2026-09-20)

- **The issue's motivating case (the Re_Dh 50 tetrahedral duct), re-run 2026-09-20 on this march**
  (`validation/tetrahedral_gradient_ab/laminar_duct_march.py`; 2462 tetrahedra, unit density and inlet
  speed, mu 5e-4, `FirstOrderUpwind`, `CorrectedGreenGauss` at its default sweeps, one run per arm, 60-step
  cap). **From a uniform plug every arm converges** (`newton_march` over `momentum_continuation` 12 steps,
  `solve_flow_march` single step 15, dual-time with a complete-LU or `SimpleSmoothed` materialized inverse
  17 and 17). **From rest the single-step arms fail** (`|R|` ~1.6e-3 after 60 steps, bare or staged) **and
  `DualTimeLoop(inner_steps=3)` with `MaterializedJacobian(CompleteLu())` or `(SimpleSmoothed())`
  converges in 13 steps** (Euclidean measure, 5.4e-9 and 5.8e-9). So the failure the issue reported is the
  single-step pseudo-transient march from rest — a configuration, not the mesh or the gradient scheme —
  and this march reaches the configuration that fixes it. **Not established:** the start state and
  tolerance of the issue's original run were never recorded, so rest is a stand-in for it. (These arms
  were run with `Euclidean()` because the default `RowScaled()` was then NaN from rest; its floors now
  let it start there (#459, above), and the duct has not been re-run under it.) `MultipleCorrectionGradient` (with the corner-cell
  fallback) does **not** converge on this duct with the same settings — an open question of #435, not of
  the flow-only path; its record is in `validation/tetrahedral_gradient_ab/README.md`.

## From "A held bulk velocity on every march" → bordered wrappers fix (#386)

- **The bordered wrappers forward everything a march hands them (fixed 2026-10-12, #386).**
  `_BorderedHomotopy.enter(step, previous)` passes `previous` through — it took `enter(step)` alone after
  the protocol gained `previous` (2026-10-11), so **every ramp on a held bulk velocity raised `TypeError`
  before step 1** (found by running `mass_flow_march.py`'s ramp arm; the unit test's stub had the old
  arity too). `_DetachedSource.refresh_preconditioner` detaches the state before calling the inner hook,
  since a session builds from fields; no hook reached it before the block family gained one.

## From "Not yet built" → FlowBlocks linearize (#55, 2026-10-09)

- **`FlowBlocks` re-runs its `jvp` per application ON PURPOSE — `jax.linearize` once per iterate was
  measured and buys nothing (2026-10-09, #55; do not re-attempt without re-running the harness).** Every
  block (`F`, `G`, `B`, `Ĉ`) is a fresh `jax.jvp` through the frozen residual, primal pass included, on
  every Krylov iteration. The issue's defence of that ("`frozen_state` is constant, so XLA constant-folds
  the primal") is **wrong**: `newton_step` builds the preconditioner inside the jitted `_march_step`, so
  the state is a *traced* runtime value and nothing folds. The real reason is that the primal pass is
  **loop-invariant inside the Krylov `while_loop`**, and the compiled loop pays only the tangent pass per
  iteration either way. Harness: `validation/flow_blocks_linearize.py` (builds `M` from a traced state
  inside the compiled solve; the `linearize` arm is a `FlowBlocks` subclass swapped in for that arm only;
  `M(v)` is bit-identical between arms). Configuration: `simple_type_composition.py`'s plane channel,
  `mu` 4e-4, marched to rel 1e-3, `ConvectionTwoLevel` velocity, `msimple` Schur, right-preconditioned
  GMRES restart 30 to TRUE `rtol` 1e-8; jax 0.11.2, CPU, x86_64, 4 cores, one run, median of 7 timings.

  | cells | primal alone | one `jvp` (no loop) | `jvp` / linearized, per apply in a loop |
  |---|---|---|---|
  | 2048 | 1.37 ms | 2.08 ms | 0.76 / 0.79 ms |
  | 8192 | 3.96 ms | 6.38 ms | 3.64 / 3.70 ms |

  | cells | composition | `M` ms/apply, jvp / linearize | cycles (both) | solve s, jvp / linearize | temp MB |
  |---|---|---|---|---|---|
  | 2048 | triangular | 2.12 / 2.07 | 10 | 0.91 / 0.99 | 8.1 / 7.8 |
  | 2048 | simpler | 3.77 / 4.11 | 6 | 0.88 / 0.84 | 10.8 / 9.9 |
  | 8192 | triangular | 7.99 / 8.39 | 13 | 5.67 / 5.77 | 31.9 / 30.7 |
  | 8192 | simpler | 18.05 / 17.48 | 6 | 3.85 / 4.25 | 42.7 / 39.3 |

  So the primal pass is most of an *unlooped* `jvp` (and so would matter for a caller applying a block
  once outside a loop), but in the loop the two arms cost the same, within this instrument's spread in
  both directions; an earlier run of the same arms (same machine, same day) agreed, linearize never
  faster on the solve. **Not measured**: a 256x128 mesh (its march ran past 30 min and the container
  restarted), GPU, the coupled RANS residual, and whether the hoisting is visible in the compiled HLO —
  the mechanism is inferred from the timings, not read off the program.

## From "Not yet built" → Stage 3: the Schur approximation

- **Outer block preconditioner — Stage 3: the remaining limit IS the Schur approximation, and no amount
  of inner accuracy reaches it (measured on a developed Re=1e5 SST channel; binding, do not re-attempt).**
  The `cycles` knob and the MSIMPLE scale are both exhausted: **velocity-AMG V-cycles ×2/×4/×8 leave
  the flow block's error operator `ρ(I − A_flow·M_flow)` at 34.02 / 33.99 / 34.03 / 34.05 (no effect);
  Schur V-cycles ×2/×4/×8 make it *worse* (41.6 / 48.7 / 48.5)**; both-exact never beats the 1-cycle
  baseline; and **rebuilding the whole block at the developed state does not help** (34.0 → 31.6 on the
  channel, 49.9 → 91.9 on pitzDaily). Inverting `Ŝ` more accurately making the preconditioner worse is
  the signature that `Ŝ` is the **wrong operator**. **Rescaling MSIMPLE's `k` is a ρ mirage:** it
  collapses ρ (34.0 → 9.6) while barely moving the one-shot error (24.1 → 22.6) and the ρ-optimal `k`
  sits ~40× above the *maximum* of the per-cell `ρV/a_P` distribution — the degenerate limit that
  switches the pressure correction off — and on the **real march it is slower** (auto-`k` 348 s / 8 steps
  vs `k×4` 447 s, identical trajectory). So the shipped per-apply `mean(ρV/a_P)` calibration is
  near-optimal, and **preconditioner changes must be validated on the real march, not on ρ** (ρ here is
  dominated by isolated outlier eigenvalues GMRES kills anyway). **Root cause:** the MSIMPLE Schur is a
  constant-coefficient (scaled pressure-mass-matrix) Poisson — a near-Stokes/low-Re approximation that
  degrades as convection strengthens. **A better Schur (the stabilized least-squares commutator) was
  built, lost 6–7× on cycles on the coupled solve, and was DELETED 2026-09-13 (#371) — there is no
  `schur_scaling="lsc"` and no `StabilizedLscSchur`.** See `.claude/notes/solve-refuted-directions.md`.
  PCD remains deprioritized
  independently (finite-element boundary recipes that do not transfer to FVM). Full numbers and the
  matching "what a preconditioner can and cannot change" rule are in `.claude/notes/solve-globalization-log.md`.

## From "Not yet built" → SCOPE (2026-08-18): what the MSIMPLE measurements described

- **⚠️ SCOPE (binding, corrected 2026-08-18): "msimple beats SIMPLE" above described
  `coupled_continuation`'s block-diagonal preconditioner, which neither flagship validation case runs and
  which no longer defaults to MSIMPLE — do not cite it as current evidence for either architecture.**
  Neither `validation/bfs3d_openfoam/compare.py` nor `validation/pitzdaily_openfoam/compare.py` reaches
  `coupled_continuation` at all (both build `coupled_amg_continuation(...)` explicitly, whose
  `_monolithic_shift_source` builds the shift policy with `build_flow_block=False` — no
  `BlockPreconditioner`, hence no `schur_scaling`, regardless of `FIELD_SPLIT`/`FLOW_INVERSE`). And where
  MSIMPLE was tested directly against a flagship-scale operator, in the field-split leading-inverse role
  the current architecture actually uses, it was dominated: `validation/bfs3d_openfoam/field_split_probe.py`'s
  `split msimple/ilu0` arm (removed 2026-09-13, #371 — in git history) is part of the "FLAT block preconditioners are CLOSED on this case" family in
  `.claude/notes/solve-flow-block-log.md` (2.554e-06 TRUE relative residual at the 58-restart-cycle cap,
  where the shipped hierarchical leading inverses converge in ~10 cycles), and a tighter,
  single-variable-changed measurement (`simple_type_swap_probe.py`, same file, "MSIMPLE swapped in for the
  SHIPPED leading inverse") that holds the shipped trailing inverse fixed instead of pairing it with an
  unshipped `ilu0` one shows both arms now converge but MSIMPLE costs ~8-9x the cycles (6→53 at the
  adjoint operator, 5→40 at the march's own shift). **`_coupled_shift_policy` no longer hardcodes
  `schur_scaling="msimple"` as of this date** — it was never necessary at the scale that policy is
  actually used at either: dropping it in favour of `BlockPreconditioner`'s own default reaches the
  identical fixed point on every small (hundreds-of-cells) unit/integration fixture that used to pin it,
  up to Re 20000 (see `.claude/notes/turbulence-log.md`, "MSIMPLE no longer hardcoded in `_coupled_shift_policy`", for the full list). What survives
  unqualified: MSIMPLE as the better of `BlockPreconditioner`'s own `schur_scaling` options for a
  *standalone flow-only* solve where plain SIMPLE's inner GMRES genuinely stalls
  (`tests/integration/test_channel_high_reynolds.py`, laminar, 384–3072 cells). It is no longer anyone's
  default.

## From "Not yet built" → MSIMPLE(R) measurements: the naming correction (#274)

- **⚠️⚠️ EVERY MSIMPLE(R) MEASUREMENT IN THESE FILES WAS TAKEN ON A METHOD THE NAME DID NOT DESCRIBE —
  corrected 2026-08-20 (#274). Read this before citing any of them.** `schur_scaling="msimpler"`
  implemented Klaij & Vuik (2013)'s **MSIMPLE**, and not even all of that: the **pressure prediction**
  that defines their `R` variants was absent, and so was Algorithm 1's closing velocity update. What was
  shipped is the lower block-triangular pass. That is a perfectly good preconditioner with its own
  justification (Murphy–Golub–Wathen), but it is not the method whose name it carried — and the paper
  puts the convergence benefit squarely on the `R` axis while saying the `M` axis buys *setup reuse* and
  is worse alone ("MSIMPLE and MSIMPLER are clearly poor solvers"; "the (M)SIMPLER variants require much
  less linear iterations per nonlinear iteration"). So the long-standing puzzle — MSIMPLE measuring worse
  than plain SIMPLE when it "should" be better — was measuring the axis that trades convergence for setup
  cost, with the axis that buys convergence missing. **There is no `"msimpler"` string any more.**

## From "Not yet built" → MSIMPLE(R) measurements: the prediction's worth (channel and bfs3d)

  - **✅ THE PREDICTION IS WORTH ~30 % OF THE OUTER CYCLES, ON BOTH CASES IT WAS MEASURED ON — so the
    paper's claim holds here once the step is actually there (2026-08-20).** Harness:
    `validation/simple_type_composition.py`, a 64×32 plane channel at `mu = 4e-4` (Reynolds number 2500),
    2048 cells, marched to a *loose* rel 1e-3 (`|R|` 1.614e-05) — **not** to a root, because at a
    machine-precision root the right-hand side `-R` is roundoff and every arm ties on noise.
    **Right**-preconditioned GMRES, restart 30, to `rtol` 1e-8 on the **TRUE** residual, cap 200 cycles:

    | schur | composition | cycles | TRUE rel |
    |---|---|---|---|
    | `simple` | `triangular` | 27 | 4.42e-10 |
    | `simple` | `simple` | 39 | 2.23e-10 |
    | `simple` | `simpler` | **129** | **3.99e-01** — **did NOT converge** |
    | `msimple` | `triangular` | 9 | 1.27e-10 |
    | `msimple` | `simple` | 9 | 1.73e-10 |
    | `msimple` | **`simpler` (MSIMPLER)** | **6** | **9.21e-12** |

    MSIMPLER is **33 % fewer cycles**, converging two orders deeper. Reproduced at a second state
    (marched to rel 1e-5): 6 against 9 again, and the whole table reproduced **cycle-for-cycle and
    residual-for-residual** by an independent re-run of the harness.
    ⚠️ **No wall column, deliberately.** The two runs put the MSIMPLER arm at 0.26 s and 0.35 s against
    the triangular arm's 0.27 s and 0.30 s — i.e. marginally faster in one and 17 % slower in the other,
    which is inside this instrument's own ~15 % spread at sub-second scale. **At this problem size the
    second Schur solve's cost and the cycles it saves are indistinguishable in wall clock**; the honest
    claim is the cycle count, and whether the trade pays in seconds has to be measured somewhere the
    solve is not already a third of a second. On `bfs3d` below the per-cycle cost separation is large
    enough to read despite the same caveat — MSIMPLER against the triangular pass is ~3.2 s/cycle
    against ~1.8 s, a 1.8x that matches the mechanism exactly (two Schur solves and four residual
    linearizations against one and one) rather than resting on the timer.
  - **⚠️ BUT PLAIN SIMPLER — the prediction on the `a_P`-scaled Schur — DOES NOT CONVERGE, and that
    asymmetry is the practical rule to take from this.** The prediction inverts `Ŝ` against a
    *velocity*-derived right-hand side, so it is far more exposed to `Ŝ` being a poor match for
    `-B F̃⁻¹ G` than the correction is. The `a_P` Schur is already the worse of the two here (27 cycles
    against 9 for the same composition), and the prediction amplifies that rather than surviving it. So:
    **pair `composition="simpler"` with `schur_scaling="msimple"`, which is exactly the pairing the paper
    promotes.** The mechanism is offered as the reading that fits, and is *not* separately measured — a
    Reynolds sweep to test it directly was started and abandoned (its low-Reynolds march would not
    finish); do not cite it as established.
  - **On `bfs3d`'s real coupled Jacobian the prediction buys the same ~29–45 %, and how that leaves the
    family DEPENDS ON WHICH OPERATOR: ~1.9x behind the shipped inverse at the adjoint's unshifted one,
    within 16 % at the march's own shift.** Harness:
    `validation/bfs3d_openfoam/simple_type_swap_probe.py` (the renamed, extended
    `msimpler_swap_probe.py`) at `state-00059`, the converged root of a full shipped 3-rung cold march;
    field split, column reach (3,3,3,3,2,2), real right-hand side `-R(state)`, one materialization shared
    by every arm, GMRES restart 15 to true `rtol` 1e-8, cap 60 restarts; **only the leading (flow-saddle)
    inverse differs — trailing inverse, wiring and shift held at the shipped bundle throughout**. At the
    **zero-shift (adjoint) operator**:

    | leading inverse | zero shift (the adjoint's operator) | march shift β=0.0050 |
    |---|---|---|
    | **shipped** — `simplesmooth` SIMPLE-smoothed AMG | **17** | **19** |
    | `msimple` / `triangular` — what every earlier measurement tested | 45 | 40 |
    | `msimple` / `simple` — Algorithm 1 in full | **58 (the cap)** | 47 |
    | `msimple` / `simpler` — **MSIMPLER** | **32** | **22** |

    ⚠️ **Other work shared the machine during this run, so cycles are the evidence and the seconds are
    not** (cycle counts on this case are exactly reproducible; wall clock is not). For the record the
    zero-shift solves were 46 / 79 / 124 / 101 s in table order.

    Read four things off it. (1) **The prediction is worth 45 → 32 (−29 %) at zero shift and 40 → 22
    (−45 %) at the march's shift**, bracketing the channel's −33 % — so the missing step was real and
    buys what the paper says it buys, on a flagship-scale operator. (2) **Algorithm 1's closing velocity
    update makes it *worse* here** (45 → the cap; 40 → 47), where on the channel it was neutral; every
    use of `F̃⁻¹` is a liability when `F̃` is a poor stand-in for `F`, and a mass matrix is a poor
    stand-in on a wall-resolved high-Reynolds 3D mesh. So `composition="simple"` is *not* "triangular
    plus a free improvement", and MSIMPLER wins despite that update rather than because of it.
    (3) ⚠️ **The recorded "MSIMPLE is dominated ~8-9x" headline does not survive at the operating point:
    at the march's own shift MSIMPLER is 22 against the shipped 19, within 16 %.** The gap is real at the
    adjoint's unshifted operator (32 against 17) and nearly closed at the march's — so anyone citing this
    family's standing must say **which operator**. (4) It is nonetheless still behind on both, and
    **cycles are not seconds**: MSIMPLER costs ~3.7 s/cycle here against the shipped ~2.6 s (two Schur
    solves and four residual linearizations per apply, against one and one), so at 22 against 19 it
    remains the slower arm. A whole-march A/B would settle it and none has been run. **Nothing here moves
    a default.**
    ⚠️ **The recorded "~8-9x, 6→53" figure was taken when `bfs3d` shipped `hostilu` as its leading
    inverse; it now ships `simplesmooth`, whose baseline is 17, not 6.** Both numbers are right under their own
    bundle — do not compare them across it.

## From "Status — BUILT" → `source.py` `diagonal` wired (#279, 2026-08-20) and the body-force migration (#224)

    - **⚠️ `diagonal` was DECLARED LOAD-BEARING AND NEVER CALLED until 2026-08-20 (#279) — WIRED
      now.** `momentum_matrix_diagonal` adds `Σ_sources source.diagonal(velocity, geometry,
      properties)` (via the private `_source_diagonal`) on **both** the residual
      (`boundary_corrected=True`) and frozen (`False`) paths, and `momentum_matrix_diagonal_parts`
      adds it to the **dissipative** bucket. Nothing was wrong while the only shipped source was
      `UniformBodyForce`, whose diagonal is zero — so the whole change is byte-identical today, and a
      test pins that. It was a trap for the first real one: a porous drag contributes `rate·V` to its
      own momentum row, and `a_P` is **not** preconditioner-only (it is the differentiated Rhie–Chow
      damping `V/a_P`), so a correctly-implemented `diagonal()` sitting unused would have given a
      wrong converged answer *and* a wrong adjoint. Two conventions, both deliberate: the total is
      added **unfloored** (it is the operator coefficient and must stay honest, even if a source
      genuinely weakens the diagonal), while the bucket contribution is **component-averaged and
      floored at zero** (those two are a forward-path stabilization scale documented `>= 0` and
      isotropic, and an anti-dissipative source must not be allowed to cancel the shift's damping).
      Pinned the operator-consistency way, not by transcription: with a drag injected,
      `momentum_matrix_diagonal` must still equal `diag(jacfwd(residual))` — plus a **control** that
      the sourceless `a_P` fails that same comparison by exactly `rate·V`, so the test cannot pass
      vacuously. The velocity AMG needed no change: it rescales its frozen hierarchy to the current
      `a_P` by a symmetric congruence, so a diagonal source term is reproduced there exactly.

  - **⚠️ A PRESCRIBED uniform force IS a `UniformBodyForce` source; only a SOLVED one is not (#224,
    2026-09-23).** Until then `MomentumContinuity` carried a `body_force` leaf beside `sources`, and it
    meant two different things: a fixed input on a periodic channel, and the live iterate of a
    bulk-velocity solve. The two meanings are now two classes — a source and a `MassFlow` drive
    (`flow/drive.py`) — and the leaf is gone. The reason a solved force cannot be a source is specific
    and worth keeping: the bordered solve writes a traced scalar into that leaf on **every** residual
    evaluation (`MassFlow.forced`, `eqx.tree_at` on `momentum.drive.force`) and forms its border column
    from the **analytic** `a = dR_flow/dβ = −V`, which holds only for a uniform, state-independent
    force. Addressing a *source* that way would mean asking it for its own `dR/dβ` instead of assuming
    `−V` — the same hand-derived-coefficient trap as the row-scaler defect.
    ⚠️ **The migration fixed a live defect, which is the general lesson: an inference from a VALUE is
    not the same as the fact it stands for.** `scales.body_force_velocity` decided "this domain is
    body-force-driven" by reading the leaf, so a channel built the source way answered zero and the
    hybrid IC started it at rest. The sum of both routes now has one home,
    `MomentumContinuity.uniform_body_force()`, and `Drive.volumetric_force` returns `None` for "no
    force at all" rather than a zero vector — because a mass-flow solve *is* force-driven at the
    instant its multiplier happens to be zero, which is every solve's first step.

## From "Status — BUILT" → the gradient scheme bound once per solved field (#467): measurement and exactness notes

  - ⚠️ **THE GRADIENT SCHEME IS BOUND ONCE PER SOLVED FIELD, NOT ONCE (#467).** `build` carries
    `velocity_gradient_schemes` (one per component, in the state layout's order) and
    `pressure_gradient_scheme`, each `gradient_scheme.bind(mesh, geometry, linearization)` against **that
    field's** `BoundaryLinearization`; `_velocity_gradient` and `_pressure_gradient`
    apply those, never `gradient_scheme`. A gradient-type condition folds its dependence on the owner
    gradient into the reconstruction's first pass, which then inverts `M1 − B`, so a scheme whose
    corrections were probed on `M1⁻¹` corrects an operator nobody evaluates. **One binding cannot
    serve the flow state**: the patches treat the fields oppositely — a `PressureOutlet` prescribes
    the pressure and extrapolates the velocity, a wall does the reverse — so the two weights are
    nonzero on *disjoint* patches. Measured on an 8×8 perturbed quad grid (perturb 0.3, seed 3,
    `MultipleCorrectionGradient` + `SkewCorrectedGradient` fallback, pressure prescribed on both
    x-normal patches and velocity on both walls): quadratics reproduced to ~4e-15 per field, against
    4.4e-3 of the pressure gradient and 3.1e-3 of the velocity gradient on the single geometry-only
    binding this replaced. Pinned by `test_the_flow_assembler_binds_a_scheme_per_solved_field` and
    `test_the_velocity_and_pressure_linearizations_differ_on_the_same_patch`.
    Each binding now takes a `BoundaryLinearization` (value weight and gradient weight per face),
    built by `_build_time_velocity_linearizations` / `_build_time_pressure_linearization`.
    Both weights are read off the closures by `jax.jvp` at **rest**, which is exact rather than
    approximate: every flow closure is affine in the gradient it is handed (a prescribed value
    ignores it, an extrapolating one adds `grad·d_t`), so the derivative at rest is the derivative
    everywhere — `test_the_flow_boundary_linearizations_do_not_depend_on_the_state`. Unlike the
    scalar twin on `ResidualAssembler`, no property is evaluated on the way, so there is no
    calculated-property failure mode here. `build` is therefore **two-phase**: construct with the
    geometry-only binding in every slot, then `dataclasses.replace` once the closures exist to read
    the weights from.
    ⚠️ **`momentum.gradient_scheme` survives and now means the CONDITION-FREE form** — the
    reconstruction this flow is configured with, prepared for the geometry alone. It is what an
    initializer solving a *different* equation on this mesh reads and re-binds against that
    equation's own conditions (`potential_flow` → `laplace_field` → `ResidualAssembler.build`); it is
    **not** what the residual applies, and reconstructing a flow field with it directly returns the
    inexact answer above.
    Exactness now needs a field to satisfy its conditions only **at the boundary faces**, so the
    ordinary inlet/outlet/wall layout is testable with a pressure varying along the duct
    (`test_the_flow_is_exact_on_an_inlet_outlet_wall_duct`). There is no such thing as the "must
    satisfy a zero-gradient patch identically" trap an earlier version of this note described: that
    was the probe telling every basis field a zero-gradient face had zero normal derivative.

## From "Status — BUILT" → `a_P` does not widen the coupled Jacobian's velocity stencil (2026-08-11)

  - **⚠️ `a_P` does NOT widen the coupled Jacobian's velocity stencil — measured, and the opposite is an
    easy conclusion to reach from this code (2026-08-11).** The argument that looks right: the mass flux
    carries `V/a_P` averaged across the face, so a cell's residual reads its *neighbour's* momentum
    diagonal, and that diagonal is a sum over the neighbour's own faces of a lagged convective flux which
    is itself gradient-reconstructed — a ring on the far side. It is wrong. Removing the velocity gradient
    from `_lagged_mdot_estimate` leaves the velocity columns reaching graph distance **3**, unchanged; so
    does first-order momentum advection, and so does a one-shot compact gradient. What actually spends the
    ring is the **eddy viscosity's dependence on the strain rate** (`.claude/rules/turbulence.md`). Note
    the one asymmetry here that *is* real and is worth keeping: `_lagged_mdot_estimate` is a function of
    velocity alone — it carries no pressure, which is what breaks the `a_P` ↔ `mdot` circularity — so
    pressure never enters `a_P`. That is true, and it is not what sets the reach.
    Harness: `validation/bfs3d_openfoam/column_reach_probe.py`.

## From "Status — BUILT" → the outlet Couette pin (#313/#319)

    - Pinned analytically: `tests/integration/test_skewed_flow.py` opens the Couette domain into a
      `PressureOutlet` (zero-gradient velocity, and the exact field has `grad u·n = 0` there because
      the perturbation leaves boundary nodes in place). Under `MultipleCorrectionGradient` the outlet
      face velocity is the exact field to **0.0** and the *whole coupled residual* (momentum and
      continuity) at the exact field is **≤1e-16** — before this fix continuity held the **1.4e-3**
      through-flow error above instead. The open (outlet) case now reaches the same solver tolerance
      as the fully-Dirichlet control (`test_stokes_couette_is_exact_on_a_skewed_mesh`),
      `‖u − u_exact‖∞ < 1e-6`, rather than stalling at `~1.2e-3`. (Configuration for every number
      here: 8×8 perturbed grid, `perturb=0.2`, `seed=2`,
      `MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None)`, Stokes, x64.)

## From "Status — BUILT" → why `ν_t` is not in the `PropertyModel`

  - **Do not put `ν_t` into the `PropertyModel`.** It is not a material property — water has no eddy
    viscosity, a turbulent flow of water does. The previous design overwrote the `"viscosity"` entry
    with a pre-summed `ρ(ν+ν_t)` field, which **destroyed the molecular value**, forced a second home
    for it (`turbulence.molecular_viscosity` — `solve_segregated`'s docstring openly said the
    assembler's own molecular viscosity "is ignored"), made the swap non-idempotent, duplicated the
    `ρ(ν+ν_t)` formula across three call sites, and changed the property's type `Constant`→
    `FieldProperty` on the first sweep. Keeping the material properties intact fixed all five.
  - The redundant `density=` parameter of `solve_segregated` and the `density` field of `CoupledRANS`
    were **removed** with it: both existed only to form `μ_eff`, which the assembler now does from its
    own `ρ`. (`SSTTurbulence` keeps its `density` and `molecular_viscosity` — the k/ω equations use
    them for their own `ν + σν_t` diffusion, a genuinely different coefficient.)

## From "Status — BUILT" → bulk-velocity constraint: design and preconditioning record (#127, #284)

- **Bulk-velocity constraint — BUILT (`flow/mean_velocity.py`, `bulk_velocity_flow_solve`).** A
  streamwise-periodic channel is driven to a target **bulk velocity** `U_bar` by making the body force
  `β` a **scalar Lagrange multiplier** on the constraint `⟨U_dir⟩ − U_bar = 0`, solved *jointly* with
  the flow — not by an outer proportional controller. `β` is **appended to the flow state** and the
  flow residual is augmented with the constraint equation, `R_aug([w,β]) = [R_flow(w;β); ⟨U_dir⟩(w) −
  U_bar]`; this **one honest residual** is handed to the production `RootSolver`, and **AD
  assembles the whole bordered Jacobian** (the force column `∂R/∂β = −V` on the flow-direction momentum
  rows, since `R = flux − βV`, and the averaging row `∂⟨U⟩/∂w = V/ΣV`) — **no bespoke solver, no
  hand-derived border** (an earlier version hand-rolled a two-solve rank-one elimination purely to keep
  the block preconditioner applicable to the border; that is a preconditioner concern, and the study
  uses a direct solve where it buys nothing — so it was dropped for the augmented residual, which also
  respects the "AD assembles the Jacobian" rule the hand-derived form broke). `⟨U⟩ = U_bar` holds **by
  construction** at the converged root, so the bulk velocity can never overshoot while the eddy
  viscosity is still developing (the failure the old proportional controller had at high Re / high
  aspect ratio: it measured `U_bulk` at a fixed β, which spiked ~17× before the feedback reacted,
  collapsing the near-wall `k` onto its floor). Being the production solve it is **convergence-gated**
  (stops on tolerance, not a fixed count) and **reverse-differentiable** through the IFT adjoint at the
  root — the constraint is part of the differentiable model at no extra cost. Returns `(momentum,
  state)` — the assembler carries the converged β out, so a segregated outer loop threads it forward
  with no controller. The **proportional mass-flow controller in `solve_segregated` was DELETED** (its
  `bulk_velocity_target`/`bulk_velocity_gain`/`flow_direction` args gone); the driver's flow-solve seam
  is now `solve_flow(momentum, state) → (momentum, state)`.
  - **Preconditioning the augmented system — constraint preconditioning (`precondition.bordered_preconditioner`;
    it was the private `mean_velocity._bordered_preconditioner` until #284 moved it to `precondition/bordered.py`,
    since it is Schur elimination over a factory and two vectors and names no flow quantity; since #541 `flow/bordered.py`'s `BorderedShiftPolicy` calls it with the drive as its `ScalarBorder`).**
    β is a multiplier, not a function of `w`, so — unlike a nested gradient sub-solve, which AD absorbs
    into the outer Jacobian — the border cannot be absorbed inside the residual; it is eliminated one
    layer down, **in the preconditioner**. Given the flow block preconditioner `M ≈ J⁻¹` (the block-
    SIMPLE AMG `BlockPreconditioner.factory()`), `bordered_preconditioner(M, drive, fields, a, c)` returns a
    preconditioner for the `(dim+1)·n_cells + 1` augmented system that Schur-eliminates the scalar β:
    `y = M·r_flow`, `dβ = (cᵀy − r_β)/(cᵀMa)`, `dw = y − dβ·(Ma)` — one flow-preconditioner apply plus
    O(n) dots per augmented Krylov iteration, exact when `M = J⁻¹` (converges in one step), inheriting
    the flow block's mesh-independence otherwise. **Hand-building `a`/`c` here is legitimate** — a
    preconditioner is an approximate, `stop_gradient`-ed inverse that changes only Krylov convergence,
    never the solution or adjoint (contrast the *residual*, whose Jacobian is AD-assembled). This is the
    robust production path (large iterative solve); `preconditioner=None` (a direct or small solve) needs
    none of it. **The bordered preconditioner is built once in the builder from a concrete `reference`**
    (required whenever `preconditioner` is given), **not** inside the solve from the `momentum` it is
    called with — a tracer captured into the (non-differentiated) preconditioner breaks `jax.grad`
    ("no constant handler" / closed-over-value). Pinned by `test_mean_velocity.py`: with an exact `M` the
    bordered preconditioner is exactly `J_aug⁻¹` (and AD's border matches the hand-built `a`/`c`), and the
    block-preconditioned GMRES augmented solve lands on the direct solve's answer. The elimination itself
    is pinned mesh-free in `test_bordered_preconditioner.py`, with the border at either end.
  - **The solve is reverse-differentiable in `momentum` (#127).** The assembler is threaded as the Newton
    **`theta`** (not captured in the residual closure), so the IFT adjoint returns its cotangent —
    `jax.grad` of an objective through the solve gives e.g. `d/dμ` (verified vs finite differences and the
    analytic `dβ/dμ = 12U_b/H²` on the laminar channel). A captured assembler instead raises a `custom_vjp`
    closed-over-value error — the reason `theta` threading is load-bearing, not cosmetic. The **adjoint**
    transpose solve is preconditioned by the *same* bordered preconditioner **transposed**, which the
    generic adjoint machinery (`solve/root_adjoint.py::_transposed_preconditioner`) forms with `jax.linear_transpose`
    (`M_aug` is linear ⇒ `M_augᵀ ≈ J_augᵀ⁻¹` exactly) — no hand-written transpose needed. Pinned: the
    preconditioned gradient equals the unpreconditioned (direct) one to ~1e-9 (a preconditioner never
    changes the sensitivity, only the adjoint Krylov iteration count).
  Pinned by `tests/unit/test_mean_velocity.py` (constraint met to machine precision; analytic β
  recovered; initial-force-independent; the preconditioner and gradient tests above).

## From "Status — BUILT" → Bernoulli pressure seed: measurements

- **Bernoulli pressure seed — BUILT (`flow/initialization.py`, `bernoulli_pressure`).** `potential_flow`
  no longer returns `p=0`: it seeds the **dynamic** pressure `p = ½ρ(|u_ref|² − |u|²)` consistent with
  the irrotational velocity (`p + ½ρ|u|² = const`), anchored so the mean pressure over the
  `PressureOutlet` cells is **zero** — consistent with the `p=0` outlet datum (`_pressure_outlet_cells`;
  a domain with no outlet anchors the domain-mean to zero instead, so a uniform plug / quiescent cavity
  stays at 0). **Why:** a coupled Newton solve otherwise starts with the *whole* dynamic-head pressure
  field as a first-step correction — measured `‖δ_flow‖≈2694` with `p=0`, of which `‖δ_pressure‖` is
  ~100% while `‖δ_vel‖≈8`; the seed **halves** it (`δ_flow≈1348`). **Closed-form on purpose (not the
  pressure-Poisson equation):** the PPE source `ρ·tr(∇u·∇u)` blows up at a sharp geometric corner (the
  backward-facing step, whose potential-flow `∇u` is singular) — measured `‖p_ppe‖≈9.4e4` and `δ_flow`
  *35× worse*; Bernoulli reads `|u|`, not `∇u`, so it is immune. This matches how ANSYS/OpenFOAM actually
  behave — ANSYS hybrid init solves a **source-free** pressure Laplace (`∇²p=0`) from the pressure BCs
  and, for a velocity-inlet/pressure-outlet case with no pressure-inlet info, seeds `p≈const` and lets
  the SIMPLE pressure under-relaxation build the field. So the remaining intrinsic `δ_pressure` (the
  potential→viscous change) is a **solver-side** concern (per-block step / pressure under-relaxation in
  the coupled step), not an IC one. Pinned by `test_bernoulli_pressure_*` (outlet-anchored, dynamic-head
  ordering, uniform-flow → 0).

## From "Status — BUILT" → `frozen_momentum_diagonal_parts` made public (2026-08-12)

  - **`frozen_momentum_diagonal_parts(assembler, state)` is a PUBLIC module-level function, not only a
    `BlockPreconditioner` method (2026-08-12).** The convective/dissipative buckets a
    `ShiftBasis` combines need **the assembler and nothing else** — the method has always just
    delegated to it. Exposing it is what lets a pseudo-transient shift be built without a
    preconditioner, and that is not cosmetic: `CoupledShiftPolicy` used to hold a whole
    `BlockPreconditioner` to read these two arrays, so a monolithically-preconditioned coupled step
    (which supplies its own inverse and never applies that one) built two multigrid hierarchies per
    build that were never used — *and*, because their aggregation reads the operator's values, their
    coarse-grid array shapes moved with the molecular viscosity, recompiling the whole coupled solve at
    every Reynolds-continuation rung. See `.claude/rules/turbulence.md`. The method stays as the one
    call site's convenience; the function is the one home.

## From "The flow march" → `flow_row_scales` floors (#459, 2026-09-29)

- **Both of `flow_row_scales`' scales are floored by the characteristic speed `|U|` (#459, 2026-09-29),
  so a march can start from rest**: the velocity field scale at `REST_SPEED_FRACTION = 0.05 |U|`, each
  cell's throughput at `REST_THROUGHPUT_FRACTION = 1e-4 rho |U| A` (`A` its summed face area), and a
  state at rest in a domain nothing drives (`|U| = 0`) is **refused** with an error naming `Euclidean()`.
  **Why floors that stay on, not one applied only at rest** (all measured 2026-09-29 on the 16 x 16 lid
  cavity at Re 100 from rest, `FirstOrderUpwind`, `CompactGreenGauss`, default `solve_flow_march`,
  80-step cap): unfloored the reference measure is NaN; flooring the velocity scale only where the
  velocity is exactly zero removes the NaN, but every step is still rejected, because the continuity
  row divides the trial's mass imbalance by `1e-300`; flooring both only at exact rest gets step 0
  accepted, and then the measure rebuilt at mean speed `0.01 |U|` reads **114x** the start, past
  `DivergenceGuard`'s 10x cap, and all 79 later steps are rejected. With both floors held the march
  converges in 18 steps. **The values:** on converged cavities (16² and 32² at Re 100, 32² at Re 400)
  the mean speed is 0.10-0.11 `|U|`, so the velocity floor does not bind there; the throughput floor
  binds in 0.0-0.7 % of cells (near-stagnant corners) at `1e-4`, against 0.0 % at `1e-5` and 2.3-3.3 % at
  `1e-3`, while step 0 sits at 1.4x, 5.3x and 1.3x of the guard's cap respectively — `1e-4` was chosen
  by the project owner for the margin. Pinned by `tests/unit/test_flow_row_scales.py` (the floors at
  rest from their definition, the state's own scales once it has flow, the refusal and its limit) and
  `test_a_lid_driven_cavity_marches_from_rest_under_the_default_measure`. ⚠️ `flow_row_scales` now reads
  the characteristic speed with a concrete `float`, so it cannot run under a trace — it never did in the
  march, which builds its measures on concrete values.

## From "The flow march" → `MaterializedJacobian` march measurement (2026-09-19, #450)

- **`MaterializedJacobian` march — BUILT 2026-09-19 (#450 stage 2), with a traced inverse (stage 4).** `solve_flow_march(preconditioner=MaterializedJacobian(SimpleSmoothed()))`, or a session from `open_flow_session` (a `CompleteLu()` inverse existed until 2026-10-11; the measurements below that name it were taken while it did). **Use `SimpleSmoothed`**: the traced JAX pressure-velocity hierarchy over the whole `(u, p)` state (`precondition.block_preconditioner.MaterializedBlockPreconditioner`, `precondition/block_preconditioner.py`), the traced kind the hierarchies exist to provide (GPU). (A PETSc `MonolithicVCycle` arm existed beside it until PETSc was removed, 2026-10-10.) `_FlowProblem` (in `flow/march.py`) supplies the assembler, the graph probe (gradient-sweep cap as `_GradientSweepCap`; default stencil reach 3 — over-reach is exact, only costlier; **the laminar reach was not measured**), **no field groups** (so `FieldSplit` is refused and a bare block inverse is allowed — the mirror of the coupled problem, which refuses the bare block), the shift-only `momentum_shift_only_policy` (no block preconditioner ⇒ no value-dependent multigrid hierarchy to recompile per rung) and the flow step. `MomentumShiftPolicy` takes the assembler first and an optional block preconditioner. **Measured 2026-09-19** on the 24 x 16 plane channel, Re 200 (`mu` 5e-3, `FirstOrderUpwind`, `CompactGreenGauss`), dual-time `inner_steps=3`, `RowScaled`, `rtol=0, atol=1e-9`, CPU, default `SimpleSmoothed()` settings, one run each: block-SIMPLE 17 steps / 51 restart cycles / 11.1 s; `CompleteLu(scipy)` 17 / 51 / 5.6 s; `SimpleSmoothed` **17 / 61 / 6.1 s** — the same trajectory, ~20 % more cycles than an exact factorization, on a 384-cell mesh where the wall clock says little. **Not measured**: a larger or 3D laminar case, or a low-shift state. `solve_flow_march` still has no `mass_flow` (bulk-velocity) form. ⚠️ **The on-device traced split (#290, 2026-10-12) does NOT reach this path**: a bare block inverse is not a field split, so `offers_traced_split` is false and the step keeps the host `MonolithicFactorShiftPolicy` (the session glues the policy and hands it to `_FlowProblem.build_step(state, policy, …)`). Giving a single-group problem the on-device apply would need a traced counterpart for `MaterializedBlockPreconditioner`, which is not built.

## From "A held bulk velocity on every march" → the unshifted constraint row (2026-10-11)

- **The constraint row is UNSHIFTED, and the march starts at the target (decided 2026-10-11).**
  `BorderedShiftPolicy` gives `β` a zero shift; `bordered_preconditioner` takes no border shift. A
  shifted step damps the fields' answer to the force while the unshifted row still asks them to reach
  the target in one step, so the force correction grows with the damping. On the first build that froze
  the laminar march (every step rejected, `dβ` 202 on a force of 0.3, escalating the shift made it
  worse) — but the cause was the **start**, not the row: the plug was sized from the seed force, so the
  constraint opened missed. `Drive.held_velocity` now starts the plug at the target, and the row opens
  satisfied. The alternative, shifting the row by its own Schur magnitude `|c · M₀ a|` (damping the force
  as the fields are damped), was built and measured and **rejected**: on `test_mean_velocity._channel`
  (4 x 48 laminar periodic channel, `mu` 0.1, target 1, block-SIMPLE, default `RowScaled`) it moved but
  crawled from the plug, and a warm start diverged under it, where the unshifted row converges the warm
  start in one step and the cold start in 12 at `beta0 = 0.1`. At the default `beta0 = 2` the cold start
  crawls through the viscous spin-up *with or without* the border (the same with a prescribed force) —
  the known cold-start crawl recorded under "The periodic seam", not a border property; the integration
  test runs at `beta0 = 0.1` for that reason.

## From "Not yet built" → Stage 2: mesh-independent AMG inner

- **Outer block preconditioner — Stage 2: MESH-INDEPENDENT AMG inner is done; the residual is the Schur
  approximation.** The inner Schur solve is a **smoothed-aggregation multigrid** V-cycle
  (`aquaflux/precondition/multigrid.py`, `SmoothedAmgSchur`, the one inner — `BlockPreconditioner.build(asm)`).
  Built once off-jit with `scipy.sparse` (no PyAMG needed),
  applied as a frozen matrix-free V-cycle; **~0.25 flat contraction** via three fixes — **direct coarse
  solve** (dense pinv), **Chebyshev smoother**, and **pin decoupling** (pin zeroed out of the operator →
  SPD singleton, null-space-matched to the pinned Jacobian; this fixed the earlier stagnation/wrong-solution
  correctness bugs). A **symmetric diagonal rescaling** `√(diag_cur/diag_ref)` tracks the reference scale
  (essential). **BOTH blocks get an AMG:** a block-by-block diagnostic (make each block's inverse exact in
  turn) showed the Schur-only version's residual ~√N was the **velocity block** (`diag(a_P)⁻¹` is
  Jacobi-quality for the viscous `F⁻¹`), *not* the Schur (making the Schur exact changed nothing) — so
  **PCD/LSC would not have helped.** The velocity momentum block gets its own smoothed AMG (Dirichlet
  no-slip → SPD-nonsingular via a `boundary_diagonal`, no pin; per component). The preconditioner is **block-TRIANGULAR**: the
  pressure block sees the divergence of the velocity predictor, `δp = Ŝ⁻¹(r_p − D·δu)`, with `D·δu`
  applied as a **jvp through a frozen residual** (`stop_gradient(self)`, so `D` is constant and
  adjoint-transparent — verified). **Result: outer GMRES 4→8 over 144→2304 cells (~O(N^0.25)), ≈37 iters
  extrapolated to 1e6 cells** vs unpreconditioned 95→1818; correct/differentiable, all
  `test_flow_preconditioner.py` pass. (Progression: Schur-AMG-only 9→34 ~√N; +velocity-AMG 7→14 ~O(N^0.3);
  +D-coupling 4→8 ~O(N^0.25).) The tiny residual growth is the 1-cycle-AMG block approximation, not the
  structure. **Two cheaper diagonals were measured and rejected** — velocity block-Jacobi and
  inverse-volume-Jacobi on the gradient solve; see `.claude/notes/solve-log.md`, "Binding decisions — where preconditioning must attach (skewed cavity)", do not re-attempt.

## From "Not yet built" → SIMPLE-type compositions (#274)

- **SIMPLE-type compositions — current status (#274).** `schur_scaling` (`"simple"`/`"msimple"`) and
  `composition` (`"triangular"` default / `"simple"` / `"simpler"`) are separate parameters; there is no
  `"msimpler"` string. MSIMPLE is no longer anyone's default. MSIMPLER (`msimple` + `simpler`) saves
  ~30–45 % of outer cycles over the triangular pass but stays behind the shipped `bfs3d` leading inverse;
  plain SIMPLER (the `a_P` Schur) does not converge; nothing here moved a default. Every recorded
  MSIMPLE(R) measurement, and which bundle and operator each was taken at, is in
  `.claude/notes/flow-log.md` § "SCOPE" and § "MSIMPLE(R) measurements".
  - **The two axes are now separate parameters, because they are separate things.**
    `schur_scaling` ∈ `{"simple", "msimple"}` picks the `InnerSchurSolver`; the new
    `composition` ∈ `{"triangular", "simple", "simpler"}` picks a `SaddleComposition`, i.e. how many
    times and in what order the velocity and Schur solves are applied. The paper's **MSIMPLER** is
    `schur_scaling="msimple", composition="simpler"`; its **SIMPLER** is `schur_scaling="simple",
    composition="simpler"`. `composition="triangular"` is the default and is byte-identical to every
    build before this change, so nothing above was invalidated by the *code*, only by the *name*.
    `msimpler_scale` is now `mass_scale` (it is the `M` variant's `k`, and always was).
  - **`SimplerComposition` is Algorithm 2 verbatim**, with `Ŝ` the compact pressure Laplacian the paper
    calls `R`: predict `Ŝ δp'' = −B F̃⁻¹ r_u` (from the momentum row alone — *not* the mass row, which is
    what keeps the wide-stencil `Ĉ` out of the operator being inverted), solve `F δu' = r_u − G δp''`,
    correct `Ŝ δp' = r_p − B δu' − Ĉ δp''` (`Ĉ` appears here, on the right, where only its action is
    needed), then `δu = δu' − F̃⁻¹ G δp'` and `δp = δp'' + δp'`. Their step 4 divides the prediction by
    the outer relaxation `ω_p`; a Newton–Krylov solve has no outer relaxation, which is the `ω_p = 1`
    limit, so the two pressures simply add. `F̃` is **the diagonal the Schur was assembled from** in
    every case (`a_P`, or `ρV/k`), which is what keeps prediction, correction and Schur consistent.
  - **How it is pinned, and why that test is the point.** `tests/unit/test_preconditioner.py` checks each
    composition against the property that *defines* it rather than against a transcription of its own
    code: on the saddle `[[diag(d), G], [B, Ĉ]]` — the real gradient, divergence and coupling blocks,
    with the momentum block replaced by the very diagonal the compositions correct with — `simple` and
    `simpler` are **exact inverses**, because both are complete block factorizations once their two
    approximations are exact. `triangular` comes back exact in pressure and *not* in velocity, which pins
    which half it deliberately omits. A dropped or mis-signed step fails this outright; it is the check
    the missing prediction could never have passed. Plus linearity and `<y,Mx> = <Mᵀy,x>` for all three,
    since a non-flexible outer GMRES and the transposed adjoint both require them.

## From "Field initialization" → laplace_field's per-call programs (2026-10-25)

`laplace_field` compiled its one-step Laplace solve as `eqx.filter_jit(newton_step)(residual, …,
preconditioner=…)`, the residual and the multigrid factory being closures built per call. Measured on
`main` `d7f5bf28` (`validation/memory_residue.py`, `RESIDUE_DROP`; `.claude/notes/solve-direct-preconditioners-log.md`
on branch `claude/reclaimable-cache`, "Which setup programs hold the reclaimable cache"): the three such
programs of a coupled initialization (φ, `k`, ω) held 0.19 GiB on bfs3d and never ran again. Built as a
per-call `jax.jit` (`claude/laplace-field-release`). The seed it gives is not bit-identical to the old one:
on pitzDaily's shipped case the three state blocks differ by at most 4.9e-14, 2.7e-14 and 1.1e-15 of their
largest magnitude (`hybrid_initialize`, same container, jax 0.12.0.1), the last bits of a recompiled
program; a march from it is a new draw. The bfs3d memory re-measure is in progress.
