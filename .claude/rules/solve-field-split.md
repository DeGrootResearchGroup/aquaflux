---
paths:
  - "aquaflux/solve/field_split.py"
---

# Rules — `aquaflux/solve/field_split.py` (the block-triangular field split)

> Split out of `solve.md` (2026-08-18). See `solve.md` for the package-wide contracts, current
> configuration, and binding decisions this file assumes.
>
> **This file has no `-log.md` sibling yet.** If you are about to push it past ~1,800 lines, split it
> first: peel the dated/historical content into a new `solve-field-split-log.md` (no `paths:`
> frontmatter) and leave a current-status summary here, following the pattern in `solve-flow-block.md` /
> `solve-flow-block-log.md`. See `solve.md`'s "Where new content goes".

## The field split — a saddle plus two transported scalars

- **⚠️ WE ARE NOT SOLVING A SADDLE-POINT PROBLEM — we are solving a saddle point PLUS two
  advection-dominated transported scalars, and that is probably why the saddle-point literature keeps
  not transferring.** Worth stating plainly because a long run of failures is explained by it:
  - **Every published method tried here targets a 2-field `(u,p)` system** — Vanka, Webster's
    stabilization, Metsch's algebraic Vanka, the SIMPLE pre-transform, monolithic saddle-AMG. Our block
    is **six** fields, two of them transported scalars, one solved in a log variable.
  - **The closest published work to this discretization segregates the turbulence.** Uroić–Jasak match
    us on every axis that usually matters (collocated Rhie–Chow finite volume, k–ω SST, backward-facing
    step, monolithic coupled AMG) and still put only `(u,p)` in the coupled block, solving k/ω
    separately with BiCGStab + ILU(0). Their papers contain **zero coverage** of turbulence-in-the-block.
    There is no published precedent for the 6-field monolithic block, so importing a fix from that
    literature is importing it across a regime boundary.
  - **The measured failure is not a saddle pathology.** The near-null direction of the degenerate cell
    blocks is **pure ω** — nothing on `u,v,w,p,k` — and per-field V-cycle smoothing has pressure *well*
    handled with ω the outlier. The saddle part of our operator is not what is hurting.
  **The direction this points at is a FIELD SPLIT that keeps the coupling** — a block-triangular
  preconditioner with the `(u,p)` saddle handled as now and k/ω preconditioned by something suited to
  transport (`scalar_transport_preconditioner` already exists and already serves those blocks in the
  block-preconditioner family). Note this is **not** the refuted arm below: that one was
  block-*diagonal*, i.e. the coupling **dropped**, and it established that the coupling is load-bearing
  — not that ω has to live in the same multigrid hierarchy. Architecturally the split is free: the
  operator stays monolithic, so the AD Jacobian and the coupled adjoint are untouched, and a
  block-triangular preconditioner is a fixed linear operator and therefore transposable.
  **Before building it, get an operator that can show a difference.** At the states currently reachable
  the monolithic ILU(0) V-cycle converges in **2 cycles to 1.3e-14**, so every candidate ties; and the
  march's cost is no longer preconditioner-bound (~65 % Krylov, largely fixed per-step matvec rather
  than cycles; 21 % refresh; the rest globalization). The upside on *this* case is bounded, and the
  test needs the hard inner iterates.
  - **✅ THE SPLIT IS BUILT — `solve/field_split.py` (`FieldGroups`,
    `BlockTriangularFieldSplit`, `build_block_triangular_field_split`).** Three facts worth keeping,
    independent of whether it ever wins:
    - **The partition is free, because the coupled state is FIELD-major.** Degree of freedom
      `(cell i, field f)` sits at `f·n_cells + i`, so a split on a *field* boundary is a split into two
      **contiguous ranges**: `[u,v,w,p]` is `[0, (dim+1)·n)` and `[k,ω]` the rest. Vectors are sliced,
      not gathered, and the four blocks are contiguous submatrices. `FieldGroups` owns that arithmetic
      so no consumer re-derives `f·n_cells + i` inline; `tests/integration/test_coupled_field_split.py`
      pins the partition against `CoupledRANSLayout.unpack`, which is the one thing that would be
      silently wrong rather than loudly wrong — a partition off by one field still preconditions, it
      just preconditions a mislabelled operator.
    - **It needs no JAX wrapper of its own.** `MonolithicAmgPreconditioner.matvec()` reads only
      `factors.n_dofs` and `factors.apply(r, transpose=…)`, both of which the split has, so it rides the
      existing `pure_callback` path unchanged. Each diagonal block is an ordinary `AmgVCycle`
      (`build_amg_vcycle` on the sub-block), which equilibrates and reorders *within its own group* and
      aggregates at its own block size — the whole point, since a four-field saddle and a two-field
      transport pair coarsen differently. `AmgVCycle.apply` returns the inverse in the **original**
      (unequilibrated, field-major) space, so the retained coupling block is applied raw between the two
      block solves, with no scaling bookkeeping.
      **⚠️ But `factors.n_dofs` + `factors.apply` is the WHOLE of what the split satisfies, and the base
      asks for more elsewhere — `has_native_solve` reads `self.factors.has_native_solve`, which only an
      `AmgVCycle` has, so on the split the inherited property RAISED (fixed 2026-08-14 by an explicit
      `has_native_solve = False` override; a split never forms the whole shifted operator, so there is no
      native solve to offer).** It went unseen for the reason worth carrying: **both call sites ask
      through `getattr(pc, "is_exact_native", False)` — the right spelling for the complete LU, which
      genuinely lacks the attribute — and a `getattr` default swallows an `AttributeError` raised *inside*
      a property body exactly as it swallows a missing name.** The value it produced was accidentally the
      correct `False`, so nothing failed. Two consequences: a test of such a property must read it
      **directly**, never through `getattr` with a default (a `getattr` test passes against the defect —
      `test_the_field_split_answers_the_native_solve_question_without_raising` reads it both ways for
      this reason); and the unnamed `factors` contract the family shares is **`n_dofs` + `apply` only**,
      so anything else the base reads off `self.factors` is an inheritance leak, not a contract.
    - **The transpose is closed-form, so the adjoint is served.** The transpose of a
      block-lower-triangular inverse is the block-upper-triangular one over the transposed blocks, so
      `apply(transpose=True)` reverses the two block solves and uses `Cᵀ` — pinned both as an exact dense
      transpose (unit) and as `⟨y, Mx⟩ = ⟨Mᵀy, x⟩` over real V-cycles on the real coupled Jacobian
      (integration).
      **`Cᵀ` is formed on FIRST USE, not at build (2026-08-14).** It was built eagerly in both
      `__init__`s and again in every `refactor`, and it is read **only** by the transpose apply — the
      adjoint's transpose solve. A forward march therefore carried a second full copy of the coupling
      block for its whole life, rebuilt at every refresh, and never touched it. It is now cached behind
      `_transposed_coupling`, with `_set_coupling` the one place that stores a coupling and drops the
      stale transpose with it (previously two sites set both fields in step, which is the pairing a
      third site would eventually get wrong). The reason it must not be re-derived *per apply* is
      unchanged and still recorded on the property: `A.T` on a compressed-sparse-row matrix is a
      compressed-sparse-column view whose product converts on every call. Not measured — the block's
      size is known but its share of a march's peak is not.
    **⚠️ A ONE-APPLICATION CONTRACTION RANKED THE TWO ORDERINGS AND WAS WRONG — invalid shortcut 2, in
    miniature, caught in a test rather than a write-up.** On the small coupled channel one application of
    the turbulence-first split leaves ~3× the input residual where flow-first leaves ~0.3×, which reads as
    a large quality gap. Through **GMRES on the true residual the two are indistinguishable**: both reach
    ~1e-14 inside one restart cycle, as does the monolithic control. Two lessons, and the second is the
    one that keeps costing time: a contraction ratio is not a convergence criterion for a
    Krylov-accelerated preconditioner; and *that state cannot rank the orderings at all*, because an
    operator every candidate solves in one cycle discriminates between none of them.
  - **✅ SHIPPED ON `bfs3d` — 31% FASTER END TO END, and the single-state probe below got the sign
    wrong.** A full 3-rung cold march at the identical configuration (`refresh_on_cycles=3`, ILU(0)×4,
    plain aggregation, `coarse_eq_limit` 2000, reach 3, restart 15), `field_split=True` against the
    shipped monolithic:

    | | monolithic | field split | |
    |---|---|---|---|
    | **wall** | 3140 s | **2161 s** | **−31%** |
    | steps | 58 | 66 | +14% |
    | Krylov cycles | 293 | 324 | **+11%** |
    | refresh | 19 events / 310 s | 23 / 352 s | +42 s |
    | mid-span `x_r/h` | 8.361 | **8.361** | identical |

    **The cycle row is the point.** The split is much faster *while doing more cycles*, because two
    smaller V-cycles plus one sparse coupling product apply far more cheaply than one six-field V-cycle
    — the coupling never enters a factorization or a coarse hierarchy. Per matched step at equal cycle
    counts the split's steps ran ~38–40% faster. Its mean cycles per **inner solve** is *lower* (1.49 vs
    1.68); the higher total is more, cheaper steps. It also crosses the refresh trigger slightly more
    often (9.7% of inner solves vs 9.0%), costing ~42 s of the ~980 s saved — the feedback loop is real
    and small. Refresh cost **per event** is unchanged (~14 s), so an earlier claim that the split
    refreshes more cheaply was wrong: it compared against the *scheduled* run's average.
    **Machine-load control:** the coloured jvp probe is identical work in both runs and took 11.3–14.6 s
    (monolithic) vs 11.7–15.1 s (split), so the faster run was not the quieter machine.
  - **⚠️ THE SINGLE-STATE PROBE BELOW SAID THE OPPOSITE — read it as a lesson, not as a result.** Harness
    `validation/bfs3d_openfoam/field_split_probe.py`. **Configuration, in full:** 3-rung cold march's
    own states; plain aggregation, **ILU(0) ×4** where not overridden, `coarse_eq_limit` 2000, stencil
    reach 3, block sizes 4 and 2; GMRES restart 15 to **rtol 1e-8 on the TRUE residual**; right-hand
    side the steady residual `−R(state)`; one materialization per state shared by every arm.

    | arm (flow / turbulence smoother) | hard iterate, β=0.0293, PC 0.05 | converged, **β=0** |
    |---|---|---|
    | monolithic ILU(0) — control | **3** (1.2e-12) | **16** (1.8e-11) |
    | split flow-first, ILU(0) / ILU(0) | 4 (1.3e-13) | **11** (3.7e-11) |
    | split turbulence-first, ILU(0) / ILU(0) | 4 (5.2e-14) | 13 (1.3e-10) |
    | split flow-first, ILU(0) / **damped Jacobi** | 6 (2.8e-12) | 16 (5.2e-10) |
    | split flow-first, ILU(0) / Chebyshev | 58 cap (3.3e-01) | 58 cap (2.6e-01) |
    | split flow-first, Chebyshev / ILU(0) | 58 cap (3.4e-02) | 58 cap (3.8e-03) |
    | split flow-first, Chebyshev / Chebyshev | 58 cap (9.97e-01) | 58 cap (9.5e-01) |
    | monolithic Chebyshev | 58 cap (9.9e-01) | 58 cap (5.4e-01) |
    | monolithic damped Jacobi | 58 cap (6.0e-01) | 58 cap (2.8e-01) |

    - **Forward: a small loss (4 vs 3), so do not adopt it for the march.** Both orderings tie there.
    - **Adjoint (`β = 0`, converged state): 11 vs 16, a 1.45× reduction** — and flow-first genuinely beats
      turbulence-first (11 vs 13), so the ordering *does* matter once the operator is hard enough to
      discriminate. This is the operator behind every `jax.grad` through a converged coupled solve, and
      it is the same place the monolithic ILUT's value turned out to lie. **Measure a coupled
      preconditioner at `β = 0`, not only on the march.**
    - **⚠️ `β = 0` must be taken at the CONVERGED state.** Stripping the shift off a mid-march iterate
      gives an operator neither the forward march nor the adjoint ever solves.
    - **⚠️ The probe's right-hand side is `−R`, and a dual-time step's is `−G = −(R + βd(φ−φₙ))`.** They
      coincide only at inner 0 (`φ = φₙ`) — which is why a sweep over end-of-step *checkpoints* is right
      to use `−R` — and on the hardest captured **inner** iterate they differ by ~200× (`|G|` 3.8e-03 vs
      `|R|` 8.3e-01). `φₙ` is not recorded by the inner observer, so `G` cannot be reconstructed. The
      self-check therefore gates on the **cycle count** (which reproduced the recorded 1) and reports the
      achieved residual without asserting it. Every arm sees the identical right-hand side, so the
      between-arm comparison is unaffected — but do not compare an absolute residual across the two.
  - **✅ THE INCOMPLETE-LU SWEEP CAN BE CONFINED TO THE SADDLE — the reachable half of the GPU prize.**
    Every smoother `solve/multigrid.py` has is Jacobi-class; the only thing PETSc supplies that it does
    not is the incomplete-LU sweep, which is also the least parallelizable piece (a sequential triangular
    solve). Two claims must be separated, because they have opposite answers:
    - **Remove it everywhere — REFUTED.** A Jacobi-class smoother on the four-field `[u,v,w,p]` block
      fails as badly as on the six-field block (both run to the restart cap). Taking ω out of the
      Chebyshev-smoothed block *helps a great deal* — 9.9e-01 → 3.4e-02 at the hard iterate, 5.4e-01 →
      3.8e-03 at `β=0`, one to two orders — so **ω-locality is real and is part of the obstruction**, as
      the cell-block singular-value decomposition predicted. But it never converges: the `[u,v,w,p]`
      block **is the saddle**, and a field split does not make it definite. Removing ω is necessary and
      not sufficient.
    - **Confine it to the saddle — SUPPORTED.** `[k,ω]` is not a saddle but a two-field
      advection-diffusion-reaction pair with a genuine diagonal, and **damped Jacobi on it converges** at
      both states (6 cycles / 2.8e-12 forward, 16 / 5.2e-10 at `β=0`), for a consistent ~1.5× cycle cost
      against ILU(0) on both blocks. At `β=0` it *ties the shipped monolithic* (16) while taking the
      incomplete factorization off two of six fields. So the k/ω hierarchy could be JAX-native, with
      PETSc confined to the four-field saddle.
    - **The Chebyshev-vs-Jacobi asymmetry on the SAME block is the mechanistic tell, and it is why
      "Jacobi-class" must not be treated as one arm.** Chebyshev **fails** on `[k,ω]` (58 cap, 2.6e-01 at
      `β=0`) where damped Jacobi converges. Chebyshev's polynomial is built for a bounded positive
      **real** spectrum; the transport pair is advection-dominated and strongly nonsymmetric, so its
      spectrum is complex and the real-interval polynomial is the wrong instrument. Damped Jacobi assumes
      only diagonal dominance, which first-order-upwind advection-diffusion-reaction has. So Chebyshev
      fails on both blocks for **two different reasons** — indefiniteness on the saddle, nonsymmetry on
      the scalars — and only the second is cured by dropping to Jacobi.
    - **Untested, and the honest caveat on the refuted half:** PETSc estimates Chebyshev's eigenvalue
      bounds with a few GMRES iterations, which is meaningless on an indefinite operator. A hand-bounded
      polynomial, or one designed for a complex spectrum, is not covered by these arms.
  - **✅ VANKA RE-OPENED ON THE FOUR-FIELD BLOCK, AND RE-CLOSED — but the ω half of the old mechanism is
    now CONFIRMED, quantitatively.** The standing verdict ("cell-centred patch relaxation is the wrong
    shape; do not re-open it with another patch variant") was measured on the **six-field** cell block,
    and its stated mechanism was that the block is weakly coupled in ω *everywhere*. A field split
    removes ω from the patch by construction, leaving the classical velocity-pressure patch the entire
    Vanka literature is built on — so that verdict does not transfer, and this is the new evidence its
    "do not re-litigate without new evidence" clause asks for. Same configuration as the table above,
    `state-00069`, β = 0, `vanka_centre_field = 3` (**mandatory** — the default is three fields from the
    end, which finds `p` in `[u,v,w,p,k,ω]` and would silently centre patches on `v` in a four-field
    block), patch width 22 = centre `[u,v,w,p]` + 6 neighbours × `[u,v,w]`:

    | arm | cycles | TRUE rel | worst patch gain |
    |---|---|---|---|
    | split, ILU(0) / ILU(0) | 11 | 3.7e-11 | — |
    | split, **Vanka** flow / ILU(0) | 58 cap | 2.9e-03 | **12.8** |
    | split, Vanka flow / damped Jacobi | 58 cap | 1.4e-01 | 12.8 |
    | split, **multiplicative** Vanka flow / ILU(0) | 40 | **1.55** (worse than the initial guess), 994 s | 12.8 |

    - **ω WAS the cause of the catastrophic patch conditioning — measured from the opposite direction.**
      The six-field campaign found the worst patch gain **flat at ~3e3 across every patch width**, which
      was the evidence that widening cannot help. On the four-field patch it is **12.8**, ~235× better,
      with **zero** patches dropped. The classical velocity-pressure patch is well conditioned on this
      operator. That is a clean confirmation of the ω-locality finding, obtained by *removing* ω rather
      than by inferring it from a singular-value decomposition.
    - **But patch conditioning was never the binding constraint, and this settles it.** A perfectly
      conditioned classical patch **still fails to converge** (stalls at 2.9e-03). The earlier campaign
      reached the same conclusion by *dropping* the near-singular patches, which was a confounded arm
      (it measured coverage); this reaches it by removing the ill-conditioning at the source, which is
      not confounded. Removing ω does help a great deal in absolute terms — the six-field Vanka stalled
      at 0.24–0.78, a 1.3–4× reduction, against 340× here — so the split genuinely strengthens the
      smoother, just nowhere near ILU(0)'s 11 cycles.
    - **Multiplicative is worse on the four-field block too, and hugely more expensive** (true residual
      1.55, 994 s vs 176 s, 17 colours). The six-field finding that sequencing costs rather than helps
      survives the split. Treat patch relaxation as closed on this operator, now for a *measured*
      reason rather than an inferred one.
    - **Two weak smoothers compound**: Vanka flow + Jacobi k/ω is 1.4e-01, far worse than either
      weakness alone. The two blocks' smoothers are independent *settings* but not independent in effect.
  - **⚠️ THE JAX-NATIVE HIERARCHIES CANNOT BE BUILT ON THE `[k,ω]` JACOBIAN SLICE AT ALL — and the
    reason is structural, not a cost.** Both were tried as the trailing block's inverse and both failed,
    for **one shared cause** rather than two incidental ones:
    - `build_convection_hierarchy` (aggregation) **refuses**: *"operator diagonal must be finite and
      strictly positive, but its minimum is −2.129e+06"*.
      **⚠️ THE OBVIOUS EXPLANATION — that the true `[k,ω]` block carries negative diagonals from its
      live source linearizations — IS WRONG, AND THE ERROR MESSAGE ITSELF SAYS SO (corrected
      2026-08-09).**
      The refusal names **`level 1`**, a *coarse* operator. The fine slice is clean: measured on `bfs3d`
      `state-00057`, **0 of 23040 cells** have a non-positive diagonal, at β = 0, at the march's shift
      and at the preconditioner floor alike (`validation/bfs3d_openfoam/trailing_block_conditioning.py`).
      The negative diagonal is *manufactured by the aggregation*, in the Galerkin `R A P` row — which is
      one of the three causes the guard's own docstring lists.
      **The real cause is that `_aggregate` is FIELD-BLIND.** It takes a bare matrix with no block size,
      so on a multi-field block it can merge a `k` degree of freedom with an `ω` one from a *different*
      cell into a single aggregate, and on a strongly nonsymmetric operator that produces a degenerate
      coarse row. PETSc GAMG does not hit this because it is told `setBlockSize(n_fields)` and coarsens
      whole **cells**. So the obstruction is the coarsening's blindness to fields, not the fine
      operator, not the source linearizations, and not the sign of anything the caller supplies.
      **Consequence: one hierarchy PER FIELD works, on the REAL Jacobian sub-blocks.** An aggregate
      cannot mix fields when there is only one, and `build_convection_hierarchy` accepts `A_kk`
      (diagonal 1.16e-06 … 2.66e-04, 57 nnz/row) and `A_ωω` (3.03e-03 … 1.0, 49 nnz/row) at every
      shift. That keeps the full stencil fill and the true source linearizations, and needs **no**
      reparametrization scaling, because the Jacobian is already in the solved variable — the log-ω
      chain factor and its wall-fixation trap simply do not arise. Built as
      `solve/field_split.PerFieldNativeInverse` / `native_per_field_inverse` (**both DELETED 2026-08-15**
      — see the deletion note in the field-split section; the surviving inverse is `NodalNativeInverse`),
      with the two fields
      composed block-triangularly (k leading).
      **This corrected a real cost: the wrong explanation was taken as an accepted blocker and sent the
      first implementation down a transport-operator detour** — a 13×-sparser, source-clamped, per-field
      stand-in needing the closure, the mass flux and the reparametrization scale — which measured 5
      cycles against the true sub-blocks' 1 on the channel and was deleted. The generalisable lesson:
      the record quoted an error verbatim and stapled an *inference* to it, and only the quotation was
      ever verified. The word `level 1` was in the quotation the whole time.
    - `build_air_hierarchy` (lAIR) ran **~50 minutes without finishing** and was killed. Contributing:
      the slice carries the distance-3 coupled fill at **91 nnz/row** where the frozen transport stencil
      is ~7 (the leading block's slice is 227), and lAIR's local approximate-ideal-restriction solves run
      over a degree-2 F-neighbourhood whose size grows roughly with the square of the row density.
    **Neither is a verdict on the method, and this is MEASURED rather than argued.** Both builders assume
    an M-matrix-like operator, and `scalar_transport_preconditioner` never hands them one that isn't:
    `_scalar_operator_pieces` **clamps its reaction diagonal non-negative** for exactly this reason (an
    anti-diffusive source would make the operator indefinite and its V-cycle diverge). The Jacobian slice
    is the unclamped truth, so it is simply not in these builders' domain. PETSc GAMG with an ILU(0) or
    Jacobi smoother is untroubled because it assumes none of this. Built on the **transport** operator
    instead, at the same state on the same mesh, both are perfectly healthy:

    | hierarchy on the transport operator (`bfs3d`, 23040 cells, `state-00069`) | k | ω |
    |---|---|---|
    | `twolevel` (the shipped scalar default) | 5.0 s | 44.3 s |
    | `air` (lAIR) | 80.0 s | 79.8 s |

    So lAIR builds in ~80 s where it did not finish in 50 minutes on the slice — a ≥40× gap on the same
    mesh — and is ~3.2× the shipped aggregation's combined build (1.8× on ω alone), which matches the
    independent recollection that production lAIR on the 2D case worked and was only somewhat slower.
    **Do not read the slice failures as anything about lAIR's cost in normal use.**
    **Note also that row density was only part of the story and was the weaker part.** The decisive fact
    is the **negative diagonal**, which is a property of the true Jacobian block whatever its sparsity;
    the 91-vs-7 nnz/row gap explains lAIR's *time* but not aggregation's outright refusal.
    **So the correct arm builds the hierarchy on the TRANSPORT operator** —
    `SSTTurbulence.k_preconditioner` / `omega_preconditioner`, which is what the segregated scalar path
    already does — accepting that it then approximates a *different* matrix from `A_tt` (no k↔ω coupling,
    no reach-3 fill, clamped diagonal), the same approximation the block-preconditioner family already
    makes. That arm is **not yet built**: it needs `mdot` and the closure at the state, a `k ⊕ ω`
    block-diagonal composition, and the log-ω chain-rule scaling (`ScaledScalarPreconditioner`) — the
    last of which is a known trap, since a rescale that ignores the wall-fixation rows' own derivative
    cost 27× on the linear residual once already.
  - **⚠️⚠️ SUPERSEDED (2026-08-10) — EVERY COST NUMBER IN THIS BULLET IS VOID, AND ITS CONCLUSION IS
    REVERSED. The native trailing inverse is now AT PARITY on cycles (2 restart cycles against 2 on the
    `[k, ω]` block alone; full configuration below) and at parity on per-apply cost — the pair of timings
    once quoted here carried no sweep count and no state and is deleted.** Both stated blockers are gone.
    The "~2.8× more expensive"
    figure was the **COO `segment_sum`** matvec, not anything intrinsic to a framework-native
    V-cycle — a CSR operator took the apply from 117.8 ms to 13.3 ms (8.8×), and the level operator
    is applied ~10× per cycle so that was essentially all of it. The "learn a block size" half was
    built and is what closed the *quality* gap. Read the 2026-08-10 section at the top of this
    IN-PROGRESS group; what remains open there is the positivity limiter, not the preconditioner.
    **The bullet is kept, unedited below, for two reasons that are worth more than the numbers:** it
    records that the deleted per-field inverse (two per-field hierarchies) was a different and weaker object
    than the single nodal one now used, so its measurements never transferred; and it is the clearest
    instance in this file of a *cost* attributed to a method when it belonged to a data structure.

    **✅ CONFIRMED ON A MARCH (2026-08-11), and the parity above is now a WIN.** Block-level parity does
    not by itself say which arm marches faster, so it was measured end to end as a controlled pair.
    *Configuration, both arms:* `bfs3d`, field split, `zerogradient` k wall BC, `K_POSITIVITY_FLOOR`
    1e-08, `equilibrate=False`, `refresh_on_cycles` 3, β-mismatch gate off, ILU(0) × 4 on the saddle,
    trailing sweeps 1, `coarse_eq_limit` 2000, restart 15, `forward_rtol` 0.3, two continuation rungs.

    | | native (`march-20260811-132658.log`) | petsc (`march-20260811-161642.log`) |
    |---|---|---|
    | wall | **2124 s** | 2893 s (+36 %) |
    | steps | **67** | 72 |
    | Krylov cycles | **329** | 371 |
    | escalations | **4** | 8 |
    | mid-span `x_r/h` | 8.361 | 8.361 |

    **The entire difference is ONE event, and the arms are otherwise bit-identical.** They track for
    **49 steps** — same β, same ‖R‖ to four figures (8.810e-03, 7.274e-03), same α — and part at step
    50. That is the controlled-pair signature the confounded comparison lacked, and it is worth more
    than the totals: it localizes the whole 769 s to the positivity-cap collapse, which each arm meets
    at a *different* step. Native meets it at 50 (cap 1.01e-02 → 1.44e-05); petsc sails through 50–52
    and meets it at 53–55 (α 0.316 / 0.075 / 0.005). Both then pay the cascade, and petsc's is worse —
    β driven to **4.4394** against **0.9364**, hence ten walk-back steps against six by the law below.
    So the arms differ in *when* they hit the cap and *how far* β is driven, **not** in how well either
    inverts its block — and the remaining wall time is in the cascade, not the preconditioner.

    So the native inverse is not merely at parity on a march, it is ahead, and **the case defaults moved
    to it** (`BFS3D_TURBULENCE_INVERSE` now defaults to `native`, with `K_WALL` `zerogradient` and
    `K_POSITIVITY_FLOOR` 1e-08 moved in the same change so the default configuration *is* the arm
    measured above; `compare.py` states all three in its banner).

    **⚠️ THE "PETSC IS FASTER" READING THAT THIS REPLACES WAS A THREE-WAY CONFOUND, AND NO SINGLE LOG
    WAS WRONG.** The belief was 58 steps against 67. The 58-step run (`march-20260811-095526.log`) used
    the **`dirichlet`** k wall BC, had **no positivity floor**, and predated both `ad3e144` and
    `bb46032` — three differences at once, each correctly stated in its own banner. Held like-for-like
    the ranking reverses, as above. **The cheap check that would have caught it on day one:** two arms
    differing only in the preconditioner solve the *same* operator, so their residuals must track for
    the first few steps. These separated at **step 1** (2.051e-01 against 2.046e-01). That test is now
    `validation/bfs3d_openfoam/march_log_compare.py`, which reports the first step at which two logs
    part. Matching aggregate step counts are not evidence of a controlled comparison.

    **The step-count gap is the positivity limiter, not the inverse** — consistent with the `limit`
    warning further down this file. Minimum step cap over a whole march: **2.02e-01** under `dirichlet`,
    against **1.44e-05** (native) and **5.41e-03** (petsc) under `zerogradient`. Under `zerogradient`
    near-wall `k` is free to ratchet down until one numerically dead cell (12800, `k ≈ 3e-20`) sets the
    global cap, which is why the `dirichlet` arm looked fast. It is a different discrete problem, not a
    faster solver.

    **The wall condition's own cost, as a controlled pair** (2026-08-11, `march-bc-dirichlet-native.log`
    against `march-20260811-132658.log`): same code, same native inverse, same 1e-08 floor, only the k
    wall BC differing — `dirichlet` **59 steps / 1911 s / 292 cycles**, `zerogradient` **67 / 2124 /
    329**, both to mid-span `x_r/h` 8.36. So the BC is worth ~11 % of the wall, and the earlier
    cross-version "58 against 67" is superseded by this same-code pair. Note what it does *not* say:
    under `dirichlet` the two inverses are within one step of each other (native 59, petsc 58 — and the
    58 is on older code), so **the native inverse's margin is regime-dependent**, clear under
    `zerogradient` and inside the noise under `dirichlet`. The default rests on `zerogradient` being the
    selected physics, not on the inverse winning everywhere.

    **The reaction to a collapsed cap is quantified, and the RETURN is the waste — not the ladder.** A
    capped step trips `retry.on_alpha` and the ladder escalates β (×2 per retry, to
    `retry.cycles_limit`); β then has to be walked back at `/grow` per step, and that unwind is
    arithmetic: **walk-back steps = log(β_peak / β_resume) / log(grow)** — predicted 2.4 / 6.0 / 10.0
    across the three marches above, observed ~2 / 6 / 10. Replaying `CflResidualDualTimeControl` over
    each logged (α, ‖R‖) sequence reproduces every logged β to four decimals on **all 22 archived
    marches**, so the law is checkable rather than inferred. The cascade regions are **12.4 %** of wall
    (native trio) and **24.7 %** (petsc).

    **⚠️ THREE OBVIOUS FIXES ARE REFUTED BY THE ARCHIVED LOGS — do not re-propose them (2026-08-11).**
    Each was proposed here first and killed by a counterexample:

    - **"Stop a ladder whose attempts move nothing."** `march-20260811-161642.log` step 56: attempts 1
      and 2 are bit-null (`G` 5.670e-03 → 5.669e-03 → 5.668e-03, α 0.000, **0 cycles each**), and
      attempt 3 at β 4.4394 is the one that **cures** the step (α 1.000). Any "stop after N nulls" with
      N ≤ 2 stops one attempt short of the cure, and the accepted null step then triggers the control's
      backoff to the same β anyway — paying a whole outer step (12–30 s) to save two 0-cycle attempts.
    - **Decaying `carry_beta`.** At `132658` step 52 the ladder *tried* β = 0.4682 and it was rejected;
      the cure was 0.9364. Carrying any decayed fraction (half → 0.47, geometric → 0.66) lands the next
      step at or below a β just demonstrated to fail at that state, and re-pays the escalation at once.
    - **"The control should skip its own backoff after an escalation."** They are a **serial search, not
      a redundancy**: across all 24 archived logs there is **not one instance** of the control backing
      off after a *successful* escalation (α ≥ 0.25 ⇒ it grows or holds). The two only co-occur when the
      ladder was *exhausted*, where the control's doubling continues the same search — the ladder gave
      up at 0.2341 / 0.5549 and the cures were 0.9364 / 4.4394. Suppressing it lengthens each cascade.

    **⚠️ "Damping cannot widen a positivity cap" is FALSE as a general statement.** The per-attempt
    `checkpoints/step-limit-*.npz` dumps (which `compare.py` wrongly called unobservable) show
    `d(cap)/dβ` with **no fixed sign** at a fixed anchor: doubling β narrowed the cap ×9.4 then ×1.6 on
    one anchor and widened it ×3.3 then ×5.6 on another, and on a third the binding cell *changed
    between attempts* (3181 → 22400 → 12800) so consecutive caps do not even measure the same thing.
    This does not reopen gating on `binding_limit == 1` (measured and reverted, below).

    **⚠️ AND THE FUTILITY READING IS CONFOUNDED.** The native trio's step-51 attempts all ran with
    `pc none 0.0s` — `REFRESH_ON_BETA` defaults to `inf`, so the `precondition_step` the march calls on
    every escalation does nothing, and every escalated attempt was solved against a V-cycle built for
    β = 0.0293. Step 52 escaped only because its inner solve happened to trip the mid-step rebuild. So
    **"the ladder is futile" and "the ladder was never given a matched preconditioner" are NOT separated
    by this data.** `BFS3D_REFRESH_ON_BETA=0.9` is the existing knob that discriminates them, at roughly
    +35 s of rebuild on one march.

    **What survives as the lever: the asymmetric return.** β is driven up by ×2 (backoff) and up to ×4
    (ladder) but recovers at only ÷1.5 per step — ~5:1 in log space — and every walk-back step is
    massively over-damped (inner counts 2,2,2,2,3,5 and 2,2,2,2,2,2,2,2,2,3, converging the implicit
    step to ~1e-7 of its start). Accelerating the ramp *only while unwinding an escalation* is
    byte-identical on any march that never escalates, and is worth ≈3 steps on the trio and 5–6 on
    `161642` by exact control arithmetic. Second lever: **the ladder is not clamped by the control** —
    it multiplies the β leaf past `beta_max`, giving β = 4.4394 against a ceiling of 4.0 and a stable
    β = 16.0 limit cycle for **74 consecutive steps** in `march-20260810-090711.log`.

    **⚠️ HISTORICAL — measured 2026-08-09, void since:** taking PETSc off the trailing half is not a
    win, and the two blockers are now specific (`bfs3d`, three states, arms `native`/`native2` in
    `turbulence_smoother_sweep.py`). With the per-field hierarchies above wired in as the trailing
    inverse, against the shipped PETSc V-cycle at `ilu0` on both halves, `refresh_on_cycles` 3, plain
    aggregation, `coarse_eq_limit` 2000, reach 3, restart 15:

    | trailing inverse | 00057 | 00058 | hard | blended | apply |
    |---|---|---|---|---|---|
    | PETSc V-cycle (shipped) | 8.66 | 8.83 | 8.88 | **8.76** | 102.0 ms |
    | native, 1 V-cycle/field | 13.09 | 13.22 | 13.23 | 13.16 (**1.50×**) | 144.3 ms |
    | native, 2 V-cycles/field | 16.50 | 17.03 | 12.94 | 16.34 (1.87×) | 204.0 ms |

    Uniformly 1.50× (spread 1.49–1.51) with apply flat to 0.3 % — a low-variance loss, unlike the
    block-Jacobi arm's. The second row shows the two deficits are separable and neither is free: more
    V-cycles close the **quality** gap (tight cycles 8 → 5 against ILU's 4) and double the **cost**.
    **The apply gap decomposes, and a traced-side port would NOT rescue it.** Timed on the same block:
    the two JAX hierarchies jitted, `jnp` in and out, cost **32.6 ms**; through the study adapter's
    numpy↔jnp marshalling, **53.8 ms**; the PETSc V-cycle they replace, **~11.7 ms** (by difference
    against the shipped split's 102.0 ms total, the flow half being identical). So marshalling is only
    ~half the gap and **the JAX V-cycle is ~2.8× more expensive than GAMG on the identical operator**.
    Removing the callback would take 144 → 123 ms, still 20 % above the incumbent and still weaker.
    (If anything 32.6 ms flatters it: XLA constant-folded the frozen hierarchy arrays into the jit.)
    **So the next attempt starts from "the V-cycle must get ~3× faster and learn a block size", not
    from "remove the callback".** The second half is a defined enhancement rather than a research
    problem — teach `_aggregate` a block size so it coarsens *cells* as GAMG does, which would let one
    two-field native hierarchy replace the per-field pair and close the quality gap without doubling
    the cycles. It does nothing about the 2.8×. **`PerFieldNativeInverse` is GONE (deleted 2026-08-15);**
    the paragraph below is why it was kept until then, and it
    is tested, transposable, adjoint-legal, and is what a block-aware aggregation would slot into — it
    is **not** wired into the production builder.

  - **NOT YET BUILT: the production wiring.** There is no `coupled_field_split_continuation` / shift
    policy, deliberately — the forward march is where a continuation builder would be used and the split
    *loses* there. What the measurement argues for is a **`β = 0` adjoint-only** preconditioner seam
    (`ForwardStep.adjoint_preconditioner()` already exists as the natural home), plus the JAX-native k/ω
    hierarchy the damped-Jacobi result unlocks. Both are unbuilt.
  - **⚠️⚠️ `trailing_smoother_sweeps` DOES NOT REACH AN INJECTED TRAILING INVERSE, SO THE NATIVE ARM
    SHIPS AT FOUR SWEEPS, NOT ONE (verified in source, 2026-08-14).** In
    `build_block_triangular_field_split` the injected inverse is called as
    `trailing_inverse(trailing_block, groups.n_trailing_fields)` — two arguments — while
    `smoother_sweeps=trailing_smoother_sweeps` is passed **only on the `else` branch** that builds its own
    `build_amg_vcycle`. `native_nodal_inverse`'s own default is `sweeps=4`, and the case's
    `NATIVE_TRAILING` sets only `max_coarse` and `equilibrate`. **So the entry below, and its measured
    16.5 % wall saving, describe the PETSc trailing V-cycle — which this case no longer uses.** Cutting
    the NATIVE trailing sweeps is therefore an untaken lever, not a shipped setting: measured on a
    same-shape synthetic, its apply runs 10.7 / 6.9 / 5.3 ms at 4 / 2 / 1 sweeps.
    ⚠️ **Two more shipped-vs-record mismatches found in the same pass:** the case runs
    `equilibrate=False` (the class default is `True`), so every statement here about the *equilibrated*
    `[k, ω]` cell block — unit diagonal, determinant exactly 1, subdiagonal 100–340 — describes an
    operator the shipped configuration does not build; and `NATIVE_TRAILING`'s `max_coarse=2000` is
    **inert**, because at `max_levels = _CONVECTION_LEVELS = 2` the level cap fires first regardless
    (`max_coarse=16` builds a bit-identical hierarchy).

  - **✅ THE TWO HALVES ARE NOW SMOOTHED APART, AND THE TRAILING DEFAULT IS ONE SWEEP —
    `trailing_smoother_sweeps=1` (BUILT, SHIPPED, 2026-08-09).** Splitting the hierarchies is only half
    the value; the other half is that they can then be *tuned* apart, which the shipped bundle was not
    doing — both halves inherited the four incomplete-LU sweeps tuned against the **six-field**
    monolithic block. The saddle needs them (Jacobi-class smoothers do not converge on it at all); the
    transported-scalar pair does not. `build_block_triangular_field_split` /
    `FieldSplitAmgPreconditioner.build` / `coupled_amg_continuation` all carry
    `smoother_sweeps` (leading) and `trailing_smoother_sweeps` (trailing, default **1**) as separate
    parameters, plus `leading_options` / `trailing_options` as the raw-PETSc escape hatch. Measured on
    the full 3-point `bfs3d` Reynolds-continuation march (field split, `retry.on_alpha` 0.01,
    `refresh_on_cycles` 3, ILU(0), plain aggregation, `coarse_eq_limit` 2000, reach 3, restart 15):

    | | 4 sweeps | 1 sweep |
    |---|---|---|
    | wall | 1959 s | **1636 s (−16.5 %)** |
    | steps | 58 | 58 |
    | refresh | 21 events / 318 s | 19 / 286 s |
    | Krylov cycles | 277 | **282 (+1.8 %)** |
    | final ‖R‖ | 9.589e-06 | 9.588e-06 |
    | mid-span `x_r/h` | 8.361 | 8.361 |

    **The two marches follow the same trajectory step for step** — identical β, identical per-step
    cycle counts, identical residuals to four figures, and the single α-collapse escalation fires at
    the same step for the same reason to the same β. So this is the *same* path at a lower price per
    matrix-vector product, which is a far stronger single-run result than a 16.5 % margin would
    normally be. (An earlier version justified that with a "~2 %" run-to-run noise figure for this case;
    it was a remembered number with no configuration behind it and is deleted — the strength of the result
    rests on the step-for-step identity, not on a noise floor.) Note again that **cycles rose while wall
    fell**.
  - **⚠️ HOW THAT SMOOTHER WAS CHOSEN, AND THE TWO WAYS THE SCREEN NEARLY GOT IT WRONG
    (`validation/bfs3d_openfoam/turbulence_smoother_sweep.py`).** The screen holds the leading half at
    ILU(0) and varies only the trailing one, ranking on **wall time** at real march states rather than
    on cycles. Two failures are worth carrying, because both produced a wrong answer first:
    - **A HARD state cannot rank candidates — it can only screen them.** The standing caution is about
      *benign* states not discriminating; the complement is equally real and was not written down. On
      the shipped march **139 of 194 inner solves cost one restart cycle and only 7 exceeded three**,
      so the worst iterate is not what a march pays for. Point-block Jacobi buys an extra cycle *only*
      where the operator is hard: it ranks **1.15× (worst of ten)** on the hard iterate and **0.94×**
      over the march's real mix. Rank on step-initial states, screen on the hard one, and weight the
      blend by the march's own solve distribution.
    - **A cost screen is blind to the DIRECTION a preconditioner returns, and a march is not.** The
      forward solve stops at `forward_rtol = 0.3`; a strong preconditioner overshoots that by orders of
      magnitude inside one restart cycle while a weak one lands near it, and **both report one cycle**.
      Since the march takes an inexact-Newton step from whatever comes back, a materially weaker arm
      hands back a worse direction, the line search clips and the step control escalates — none of
      which a timing screen registers. The usable proxy is the screen's **tight** cycle count read as a
      *magnitude*, not as a pass/fail gate: shipped 3/3/4 (two step-initial states, then the hard one),
      `ilu0x1` 3/4/5 — about the same strength, and it reproduced the baseline trajectory exactly —
      against `jacobix2` 5/7/7 and `jacobix1` 8/10/13. **A candidate needs three things: cheap per
      application, convergent at all, and not materially weaker than the incumbent.**
    - **THE `[k, ω]` CELL BLOCK IS TRIANGULAR, AND THAT EXPLAINS THE WHOLE SMOOTHER RANKING ON THIS
      BLOCK.** Equilibrated, each cell's 2×2 is **lower-triangular with unit diagonal and a subdiagonal
      of order 100–340**, every determinant exactly 1.0, and the k↔ω coupling is **~100 % same-cell**
      (‖∂R_ω/∂k‖ splits 2.14e3 same-cell against 23.9 neighbour on the channel;
      `validation/bfs3d_openfoam/field_coupling.py` finds the same asymmetry on `bfs3d` — ∂R_ω/∂k at
      121× the diagonal blocks, ∂R_k/∂ω at 0.19 % of them). ω depends enormously on same-cell k through
      the production limiter and the βkω destruction pair; k barely depends on ω, because wherever the
      SST limiter is active `ν_t = a₁k/max(a₁ω, S F₂)` is **ω-independent**. Consequences:
      - **ILU(0) in cell-major order already captures it exactly** — the forward/backward pair is a
        complete factorization of a 2×2 with one off-diagonal (no fill), so the subdiagonal costs it
        nothing. That is why one sweep is nearly as good as four here (tight cycles 3/4/5 against
        3/3/4) and why `sor`, also a sweep in the favourable order, matches ILU(0)×4 at 3/3/4.
      - **Point Jacobi discards the entire subdiagonal**, which is precisely the term the others get.
      - **Point-block Jacobi recovers it, and measurably does**: split by field, a `pbjacobi`-smoothed
        V-cycle lands **10× closer to an ILU-smoothed one than point Jacobi does in the ω rows**
        (1.48e-4 against 1.49e-3). The end-to-end near-tie is real all the same — the V-cycle output is
        ω-dominated (‖ω‖ 2318 vs ‖k‖ 29.7), all three get the bulk right, and the coarse correction
        plus outer Krylov absorb the rest.
      - **Re-ordering the fields does NOT help.** A symmetric permutation to `(ω, k)` makes the block
        upper-triangular; Jacobi and block-Jacobi are permutation-invariant, ILU holds both triangles
        either way, and a *forward* sweep (SOR/Gauss-Seidel) is made strictly worse — the current order
        lets it solve k first and use the fresh value in ω. The shipped order is already the good one.
      - **For a JAX-native smoother this is the cheap prize:** with the equilibrated unit diagonal the
        block solve is one fused multiply-add per cell (`x_ω -= c·x_k`), fully parallel across cells,
        no factorization and no sequential dependency.
    - **⚠️ BLOCK JACOBI CANNOT REPLACE THE TRAILING HIERARCHY — measured, and the reason is
      ROBUSTNESS rather than reduction.** The `[k, ω]` block is strongly *block*-diagonally dominant
      (neighbour coupling ~12 % of same-cell for the diagonal fields, 1.1 % for `∂R_ω/∂k`), and block
      Jacobi's error operator is exactly that neighbour part — so it looks as though a coarse grid has
      nothing to do here, and on a synthetic operator at that neighbour weight it contracts ~10× per
      sweep. It does not carry over. Measured as the **whole** trailing inverse (a native batched 2×2
      solve, no hierarchy at all) against the shipped V-cycle, on two step-initial states and the hard
      iterate, blended by the march's own solve mix:

      | trailing inverse | 00057 | 00058 | hard | blended |
      |---|---|---|---|---|
      | `ilu0` V-cycle (shipped) | 8.68 | 8.74 | 8.72 | **8.71** |
      | block Jacobi ×4 | **8.63** | 10.94 | 11.17 | 9.94 (1.14×) |
      | block Jacobi ×8 | 9.09 | 11.32 | 9.29 | 10.10 (1.16×) |
      | block Jacobi ×2 | 10.40 | 10.47 | 10.88 | 10.48 (1.20×) |
      | block Jacobi ×1 | 13.72 | 14.09 | 14.29 | 13.95 (1.60×) |

      **What the coarse grid buys is variance, not average.** The V-cycle is 8.68/8.74/8.72 — flat to
      under 1 % across all three states. Block Jacobi ×4 swings 8.63 → 10.94 between *adjacent steps of
      the same rung*, because it sits on the edge of converging inside one restart cycle and small
      state changes flip it to two. And more sweeps cannot buy the edge back: by 8 sweeps its apply
      cost (109 ms) exceeds the V-cycle's (102 ms), so it has spent the whole cost advantage
      recovering what the coarse grid gave for free.
      **This does NOT refute block Jacobi as a SMOOTHER inside a JAX-native hierarchy**, which is the
      actual route for taking PETSc off this block and is untested. The implementation
      (`BlockJacobiInverse` in `validation/bfs3d_openfoam/field_split_probe.py`) is verified exact on a
      block-diagonal operator in one sweep, transposable in closed form (⟨y,Mx⟩ = ⟨Mᵀy,x⟩ to 1e-15, so
      adjoint-legal) and a fixed **linear** operator (1e-16, so the non-flexible outer Krylov is
      legal) — it is the smoother such a hierarchy would need.
      **Methodological note, because it went wrong twice in one session:** the first easy state alone
      showed a tie and was reported as a result. The second easy state, same class and adjacent step,
      was 25 % worse. **Hold a screen's conclusion until every state has landed** — the per-class
      spread is itself the finding here.
    - **⚠️ CHECKPOINT NAMES ARE REUSED ACROSS MARCHES, so a probe can silently run at a state that is
      not the one its table documents.** `StateCheckpointer` keeps only the last few files
      (`BFS3D_CHECKPOINT_KEEP`, default 3) and numbers them from a counter that restarts each run, so a
      later march *replaces* `state-000NN` with a different state under the same name. Observed: a name
      documented as the converged zero-shift adjoint operator came back holding a mid-march iterate at
      shift 0.98 from an abandoned run — a probe would have paired an operator built at the documented
      shift with a state that never had it and reported it as a measurement at that operating point.
      `field_split_probe.load_state` now checks the checkpoint's recorded shift against its `STATES`
      entry and refuses on a mismatch (loosely, at 2 %: the table stores ~4 figures, and what this must
      catch differs by orders of magnitude). Raise the keep count for a study that needs a trajectory.
    - **⚠️ `equilibrate_cell_major` RETURNS UNSORTED COLUMN INDICES, and PETSc's AIJ format requires
      them ascending.** `AmgVCycle._build` and `refactor` both `sort_indices()` before wrapping the
      matrix, and `ShiftedCellMajorOperator` genuinely produces sorted output (its
      `has_sorted_indices = True` is honest, verified), so **every shipped path is correct**. But a
      probe that calls `equilibrate_cell_major` directly and feeds `createAIJWithArrays` gets **NaN in
      most entries from `pbjacobi`**, while `jacobi` and `ilu` survive it — a diagonal scan does not
      care about column order and a block extraction does. That asymmetry reads exactly like "PETSc's
      point-block Jacobi is broken on this operator", which was written up and had to be retracted.
      **Check `has_sorted_indices` before concluding anything about a block method.**
    - **⚠️ TWO MARCH ARMS WERE RUN AT THE WRONG REFRESH TRIGGER AND ARE VOID** — launched without
      `BFS3D_REFRESH_ON_CYCLES=3` when that variable still defaulted to `0` (the *scheduled* cadence,
      measured at 3632 s against 1959 s). Both "results" were the refresh trigger, not the smoother.
      The trap was already written down in bold and that did not prevent it, so the fix was made
      structural: **the case default is now 3**, and every run writes its full configuration into its
      own `march.log` header before any result. A default nobody wants is a trap, not a setting.
    - **Still unsolved, and it bites any future arm comparison:** `refresh_on_cycles` and
      `cycle_budget` are denominated in **cycles**, so a preconditioner that shifts the cycle
      distribution changes the *effective* trigger point — penalizing a weaker arm twice, once in
      cycles and again in the refreshes those cycles provoke. (Same argument as the case's
      `_RESTART_SCALE`.) Report the refresh **count** beside the wall in every arm comparison; it came
      out level for `ilu0x1` (21 vs 19 events), which is what licensed attributing its saving to the
      smoother.

