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
    not the per-field split — see the ⚠️⚠️ note beside the `bind` entry in `.claude/rules/schemes.md`.

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
