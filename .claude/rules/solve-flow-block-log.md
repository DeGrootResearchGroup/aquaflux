# Investigation log — `aquaflux/solve/` the flow block

> Split out of `solve.md` / `solve-flow-block.md` (2026-08-18). **No `paths:` frontmatter — this
> file never auto-loads.** It holds the full chronological investigation behind the native
> `[u, v, w, p]` flow-block preconditioner, including several rounds of "found a win" that a later,
> tighter measurement qualified or retracted — kept because a wrong-but-plausible finding gets
> re-derived if the trail that refuted it is deleted rather than archived. See `solve-flow-block.md`
> for the current, load-bearing status (the native path is built and differentiable but is not the
> shipped march default).

### FLAT block preconditioners are CLOSED on this case

Every arm that replaces the flow block's inverse outright — no hierarchy, no coarse-grid correction over
the saddle — fails at `beta = 0` (58-cycle cap):

| arm | TRUE rel |
|---|---|
| block-SIMPLE, MSIMPLER Schur | 2.554e-06 |
| block-SIMPLE, `a_P` Schur | 7.812e-03 |
| algebraic SIMPLE Schur, native sub-block inverses | 1.049e-02 |
| left block transform (exact) + native multigrid on the transformed operator | 1.473e-02 |
| left block transform (Schur-only) | 1.395e-03 |
| multi-step saddle, Frobenius, 1 step | 1.632e-04 |

**The signature is a floor almost insensitive to the shift** — MSIMPLER moves only ~2× between `beta = 0`
and the forward operating point, where ILU(0) goes 11 cycles to 4. That is an *approximation* ceiling, not
a conditioning one: a flat inverse must get every error component right in one application, and a
SIMPLE-type Schur is worst precisely on the smooth global pressure mode a coarse grid exists to handle.

Two sub-results worth keeping:
- **Neither sub-block inverse is the constraint.** A 2×2 over which half is native: PETSc/PETSc 1.707e-02,
  native/PETSc 1.555e-02, PETSc/native 3.904e-03, native/native 1.049e-02 — a 4.4× spread against an
  eight-order gap to the incumbent.
- **The raw `(p, p)` block is NOT usable as the pressure operator in a split.** With host V-cycles it
  *diverges* (39 cycles, true relative residual 1.198e+00). That it is already 0.71× the SIMPLE-Schur
  elliptic operator does not make it a Schur substitute.

⚠️ **The block-SIMPLE arms were built with `reference_state` at the developed field**, where the production
path passes none and falls back to a characteristic uniform flow at the fastest patch velocity. They were
therefore *flattered* relative to the shipped object, and still failed.

### MSIMPLER swapped in for the SHIPPED leading inverse, trailing held fixed — DOMINATED, ~8-9x more cycles (2026-08-18)

**The table above pairs MSIMPLER with an `ilu0` trailing inverse, which `bfs3d` does not ship — its
`TURBULENCE_INVERSE` default is `"native"` (`compare.TRAILING_INVERSE = native_nodal_inverse(
**compare.NATIVE_TRAILING)`), so that comparison changed two things relative to the shipped bundle at
once. This measurement changes exactly one.** Two arms, same materialized Jacobian, same field-split
wiring, same trailing inverse, same column reach, same state — only the leading (flow-saddle) inverse
swapped:

- **shipped** — `compare.LEADING_INVERSE` (`host_ilu_inverse` at the case's own settings: `sweeps=1,
  cycles=1, strength_threshold=0.25, avoid_singletons=True, aggressive_levels=0, max_levels=5,
  max_coarse=500, prolongation_smoothing="none"` — the actual shipped default as of this run,
  `BFS3D_FLOW_INVERSE=hostilu`).
- **msimpler** — `BlockPreconditioner.build(momentum, velocity="convection",
  schur_scaling="msimpler", strength_threshold=0.25)`, built from the real assembler + eddy viscosity
  at the probed state, exactly as `field_split_probe.block_simple_arms` constructs its `msimpler` arm.

Both paired with the SAME trailing inverse, `compare.TRAILING_INVERSE` (`NodalNativeInverse` at
`compare.NATIVE_TRAILING`: `max_coarse=COARSE_EQ_LIMIT, equilibrate=False`).

*Configuration:* `bfs3d`, state `state-00059` from a full shipped 3-rung cold march completed the same
day (`nervous-tereshkova-bf3a80`, converged: step 20 of the target rung, `|R|` 1.830e-06, march shift
0.0050, mid-span `x_r/h` 8.36 against OpenFOAM's 7.24) — the converged root, i.e. the adjoint's operator
at zero shift, plus the march's own operating shift for completeness. Field split, column reach
`(3,3,3,3,2,2)` (the case's shipped default), real right-hand side `-R(state)`, GMRES restart 15 to
rtol 1e-8 on the TRUE residual, cap 60 restarts. Harness:
`validation/bfs3d_openfoam/msimpler_swap_probe.py`.

| operator | arm | cycles | TRUE rel |
|---|---|---|---|
| zero shift (adjoint operator) | **shipped (hostilu + native trailing)** | **6** | 1.986e-11 |
| zero shift (adjoint operator) | msimpler (native trailing, all else shipped) | **53** | 3.250e-09 |
| march shift β=0.0050 | **shipped** | **5** | 1.186e-11 |
| march shift β=0.0050 | msimpler | **40** | 3.384e-09 |

**Both arms converge this time** (unlike the flat-PC table above, which hit the 58-cycle cap under the
mismatched `ilu0` trailing) — matching MSIMPLER with the trailing inverse the case actually ships is
enough for it to reach a real, if far looser, tolerance. But swapped in for the shipped leading inverse
with every other setting held fixed, it costs **8.8x the cycles at the adjoint operator and 8x at the
march's own shift**. This is the controlled, single-variable-changed version of the question
`field_split_probe.py`'s own `block_simple_arms` docstring poses ("is MSIMPLER worth pursuing on the flow
block at all") and it answers it: no — not against the current field-split architecture, at the shipped
trailing inverse, at the case's own hard state. Do not re-propose `schur_scaling="msimpler"` as the
`bfs3d` leading inverse without a new measurement that changes one of these conditions. **Consequence
(same day): `_coupled_shift_policy`'s `BlockPreconditioner.build` no longer hardcodes
`schur_scaling="msimpler"`** — combined with the small-fixture check in `.claude/rules/turbulence.md`
(dropping it reaches the identical fixed point on every unit/integration fixture that used to pin it),
this measurement removed the last reason to keep it as anyone's default; see that file's entry for the
current, corrected `_coupled_shift_policy`.

### Coarsening rate and depth — measured, and the direction that helps is the wrong one

Native nodal hierarchy over the flow block, SIMPLE smoother at the settings above, 4 sweeps, 20-cycle cap:

| coarsening | levels | fine → coarse | ratio | TRUE rel | dense coarse solve |
|---|---|---|---|---|---|
| squared graph (`aggressive_levels = 1`) | 2 | 92160 → 872 | 106× | 4.199e-05 | 6 MB |
| plain maximal-independent-set | 2 | 92160 → 4300 | 21× | **2.522e-05** | 148 MB |
| plain MIS + strength threshold 0.10 | 2 | 92160 → 26244 | 4× | aborted | **5.94 GB** |
| plain MIS, three levels | 3 | 92160 → 644 | 143× | 4.964e-05 | — |

- **`max_levels` is INERT while `max_coarse` binds first.** Asking for three levels at the aggressive rate
  produced a bit-identical two-level hierarchy and a bit-identical result, because the first aggregation
  already lands at 872 < `max_coarse` 2000. A level count alone cannot distinguish that from "depth does
  not help", which is why the coarse size is now printed beside it.
- **A strength threshold coarsens LESS, not more.** It aggregates only along strong connections, so the
  aggregates shrink: 0.10 gives a 4× ratio and a 26244-equation coarse grid whose dense inverse is a
  5.94 GB captured constant (JAX warns and the arm was abandoned). It is the wrong instrument for
  controlling coarse-grid size.
- ⚠️ **The aggressive first level on this block is INHERITED, not measured here.** It was adopted to match
  PETSc GAMG on the `[k, omega]` **trailing** block under a point-block-Jacobi smoother, where it was worth
  5 → 2 cycles. On the flow saddle under a SIMPLE smoother it costs **1.7×**.

**A coarse solve is NOT where to worry about a host solver or a sequential algorithm.** The incomplete-LU
this work exists to remove is **fine-grid** work — 92160 rows, 21M nonzeros, on every sweep of every cycle.
A direct solve on a few hundred coarse equations is negligible beside it in both cost and parallelism. An
earlier reading here, that the dense coarse inverse must be replaced before this direction can
proceed, had that backwards: what has to go is the oversized coarse grid, not the density of its solve.

### Depth, singletons, and the prolongation — where the AMG method itself was wrong

**Depth had never actually been tried, and one arm proved it.** Asking for three levels at the
aggressive rate produced a **bit-identical two-level hierarchy and a bit-identical residual**, because
`max_levels` cannot bind while `max_coarse` (2000) stops the coarsening first. A level count alone
cannot distinguish that from "depth does not help", which is why the coarse size and the aggregate-size
distribution are now printed beside it (`aggregate_size_histogram`, and `_AGGREGATE_STATS` for the most
recent build).

**The defect the histogram found: `_mis_aggregate` manufactures SINGLETON aggregates, and they are an
artifact of arrival order rather than of the graph.** The sweep visits vertices in a random permutation
and any unclaimed vertex becomes a selector; a vertex reached late can find *every* neighbour already
claimed, and then opens an aggregate containing only itself. Measured on the `bfs3d` flow block with
plain aggregation, three levels:

| level | aggregates | size min/median/max | singletons |
|---|---|---|---|
| 0 | 1075 | 1 / 17 / 63 | **107** |
| 1 | 161 | 1 / **3** / 35 | **49 of 161** |

A second level with a median aggregate size of three and 30 % singletons is not a coarse space — it is
a slightly smaller copy of the fine grid in which a third of the unknowns stand for one cell each and
couple to almost nothing.

**Fixed by attaching such a vertex to an adjacent aggregate instead** (`_mis_aggregate(
avoid_singletons=True)`; a vertex with genuinely no neighbours is a true isolate and still gets its
own). Same configuration, 20-cycle cap, native SIMPLE smoother at 4 sweeps:

| arm | level-0 aggregates | level-1 aggregates | coarse dofs | TRUE rel |
|---|---|---|---|---|
| three levels, as built | 1075, 107 singletons | 161, median 3, 49 singletons | 644 | 4.964e-05 |
| **three levels, no singletons** | 968, none | 94, median 8, **none** | **376** | **2.844e-05** |

Worth **1.7×**, and it coarsens *further* while doing it (644 → 376). Off by default
(`avoid_singletons=False`) and byte-identical off.

**The practical consequence is that the coarse-space cost problem largely dissolves.** The best
two-level arm is 2.522e-05 on a **4300**-equation coarse grid; three levels without singletons is
2.844e-05 on **376** — within 13 % at **a eleventh of the coarse space**, which is back inside ordinary
practice and trivial to invert densely. (Both figures are residual-after-18-cycles at a 20-restart cap,
not convergence; uncapped, the same two hierarchies converge in 39 and 44 cycles, which corroborates the
conclusion. Every residual quoted in this subsection is capped — see *Where this stands* for the
converged numbers.) The earlier reading here, that gentler coarsening's 1.7× was
unaffordable and therefore not a lever, was measuring a hierarchy whose second level was degenerate.

**⚠️ ORTHONORMALIZING THE TENTATIVE PROLONGATION IS REFUTED — twice, and the mechanism offered for it
was wrong both times.** The recorded prediction was that our 0/1 columns (against PETSc's QR-orthonormal
ones) are "provably inert at two levels with an exact coarse solve" but "start to matter with inexact or
deeper coarse levels", making them the suspect for depth being unhelpful. Built as
`orthonormal_prolongation=True`:

| arm | TRUE rel |
|---|---|
| two levels, with and without orthonormalization | 4.199e-05 **both, bit-identical** |
| three levels, as built | 4.964e-05 |
| three levels + orthonormalization | 8.411e-04 |
| three levels, no singletons | 2.844e-05 |
| three levels, no singletons + orthonormalization | 8.465e-04 |

The two-level identity confirms the inertness claim exactly. Deeper it is **17–30× worse**, and
eliminating the singletons does not rescue it (8.465e-04 against 8.411e-04) — so the proposed mechanism,
that orthonormalization promotes singletons by scaling a column by `1 / sqrt(|agg|)`, is refuted.

**A better hypothesis, offered as one and NOT measured:** the coarse *space* is identical either way, since
scaling a column does not change its span, so this can only be about the coarse *operator*. With 0/1
columns `A_c = P^T A P` **sums** fine rows over each aggregate, preserving row-sum structure — and for a
finite-volume conservation-law discretization the row sums are the conservation statement. Scaling by
`1 / sqrt(|agg|)` destroys it, which is why agglomeration multigrid for conservation laws uses unscaled
sums. Do not act on this without measuring it.

**The singleton fix helps at TWO levels as well — 1.85×, so it is a property of the aggregation and not
a repair for depth.** Measured 2026-08-13 on `state-00067` (β = 0 on both operator and preconditioner,
the adjoint's operator), native SIMPLE smoother at 4 sweeps with the Frobenius diagonal on both halves,
plain aggregation, restart cap 20 (so every arm below stops at 18 cycles — the cap, not a convergence
test; only the residual separates them):

| two-level arm | level-0 aggregates | coarse dofs | build | TRUE rel |
|---|---|---|---|---|
| as built | 1075, 107 singletons | 4300 | 19 s | 2.522e-05 |
| **no singletons** | 968, none | **3872** | 15 s | **1.362e-05** |

It is better, coarser, and cheaper to build at once. **`avoid_singletons` should therefore become the
default rather than an opt-in** — it is currently `False`, and byte-identical when off.

⚠️ **The same fix paired with the SQUARED-GRAPH first level is a NULL TEST, not a negative result.**
Aggressive coarsening builds aggregates of median size 82 and produces **no singletons at all**, so
`-ns` there returns a bit-identical 4.199e-05 and measures nothing. Only the gentler rate strands
vertices. Read a no-change result from that arm as "nothing to fix here", never as "the fix does not
work".

**DEPTH PAST THREE LEVELS DOES NOT PAY, and singletons were not the reason.** With the aggregation
clean at every level, the same configuration continued to coarsen:

| levels | coarse dofs | ratio | TRUE rel |
|---|---|---|---|
| 2 | 3872 | 24× | 1.362e-05 |
| **3** | 376 | 245× | **2.844e-05** |
| 4 | 56 | 1646× | 4.875e-05 |
| 5 | 12 | 7680× | 3.820e-05 |

Four levels is worse than five, so this is not a monotone degradation that a further fix would unwind —
it is noise around a floor the hierarchy cannot get below. Depth buys coarse-grid *size* (376 at 13 %
worse than 3872, which is the trade worth taking) and nothing else.

⚠️ **`max_levels` cannot bind while `max_coarse` stops the coarsening first.** The three-level arm lands
at 376, already under the 2000-equation limit, so `-L4` and `-L5` rebuild the identical hierarchy and
return the identical residual. The depth rows above were obtained by lowering that limit alongside the
level cap. A depth sweep that moves only the level count is measuring one hierarchy N times.

### Smoothed aggregation on the flow saddle — REFUTED under the SIMPLE smoother

Every native arm interpolates the coarse correction piecewise-constant over each aggregate, which is the
textbook reason a hierarchy works at two levels and gains nothing deeper. Smoothing the prolongator once
with the operator is the standard cure, and it had only ever been measured here under a *Jacobi*
smoother — before the Frobenius diagonal turned the velocity predictor from amplifying to contracting —
so it was a fresh question rather than a settled one. It is now settled. Same state and bundle as above,
three levels, plain aggregation, no singletons:

| prolongator | 4 sweeps | 8 sweeps |
|---|---|---|
| **unsmoothed** (`"none"`) | 2.844e-05 | **8.002e-06** |
| symmetric-part (`"symmetric-part"`) | 2.925e-04 | 1.035e-04 |
| standard σ_max (`"standard"`) | 1.611e-01 | 3.005e-01 |
| standard σ_max, equilibrated | — | 8.810e-01 |

Symmetric-part is **10–13× worse** and the gap *widens* with sweeps. The standard formula does not
converge at all, gets **worse** with more relaxation — the signature of a mode the smoother amplifies and
the coarse grid cannot correct, not of under-smoothing — and equilibrating it costs another 3×. Smoothing
also widens the coarse stencil (level-1 operator 0.4M nnz against 0.1M) and changes what the next
aggregation sees, landing a 108-dof coarse grid instead of 376, so the arms differ in coarse space as
well as in interpolation.

**Sweeps beat every structural change tried in this subsection.** Doubling relaxation on the unchanged
three-level hierarchy was worth **3.6×** (2.844e-05 → 8.002e-06) — more than depth, smoothing, or
orthonormalization returned. Weigh it against cost before treating it as a win: eight sweeps roughly
doubles the work in every application, and at a restart cap all arms report the same cycle count, so a
residual gain there is not a like-for-like comparison with a four-sweep arm.

⚠️ **This was written before the strength threshold was measured, and the threshold beats it — 4× against
3.6×, and on cycles rather than on a capped residual.** Every arm in this subsection ran
`strength_threshold = 0`. Do not read "the lever is sweeps, not the hierarchy" out of it; the aggregation
was the larger lever all along, and these arms simply never varied it. See *Where this stands*.

### Where this stands — the native flow block converges: 1.84x at the adjoint, 2.4x on the march

⚠️ **The tables in THIS subsection predate the two free quality levers** (a per-cell block velocity
splitting and an undamped correction) and the per-iteration split measurement. They are kept because the
closures they record still stand, but for the current standing read *The gap is CONVERGENCE, not cost*
below: **7 cycles / 57 s against 11 / 31 s at β = 0**, and **6 / 29 s against 2 / 12 s at β = 0.1**.

**The four-to-five-order gap never existed: it was the GMRES restart cap read as a convergence floor.**
Every earlier reading here was taken at a 20-restart cap, where `restart_cycles` reports 18 for any arm
whatever its quality. Uncapped, the native hierarchy reaches adjoint grade. Two later corrections then
moved the number again, both in the incumbent's favour, so read the table and not the older prose.

Measured 2026-08-13 on `state-00067` (converged, march shift 0.0064, |R| 3.586e-06) with **β = 0 on both
operator and preconditioner**, real right-hand side `-R(state)`, GMRES restart 15 to rtol 1e-8 on the
**TRUE** residual, 60-restart cap, uniform column reach, ILU(0) on the trailing half. Native arms are the
SIMPLE-smoothed hierarchy with the Frobenius diagonal on both halves, plain aggregation, no singleton
aggregates, 5 levels, `max_coarse` 500.

| arm | strength | outer x inner sweeps | cycles | solve | s/cycle |
|---|---|---|---|---|---|
| **`split flow/ilu0` — the matched incumbent** | — | — | 11 | **29 s** | 2.64 |
| monolithic ILU(0) — *not* the right bar | — | — | 11 | 42 s | 3.82 |
| native | 0.10 | 8 x 4 | 11 | 80 s | 7.27 |
| native | 0.25 | 8 x 4 | 10 | 83 s | 8.30 |
| **native — best** | **0.25** | **8 x 2** | 11 | **68 s** | 6.18 |
| native | 0.25 | 8 x 1 | 14 | 70 s | 5.00 |
| **native — fewest cycles** | 0.25 | **16 x 4** | **6** | 98 s | 16.3 |

**STRENGTH OF CONNECTION IS THE LARGEST LEVER, worth 4x.** At `strength_threshold = 0` — the builder
default, and what every native arm ran until now — `_aggregation_edges` returns the full cell adjacency,
so aggregation ignores the operator's values entirely and coarsens across the stiff direction. Turning it
on took the arm from 44 cycles / 213 s to 10 / 83 s. The optimum is interior and sits on the standard
value: 0 → 44 cycles, 0.10 → 11, 0.25 → 10, 0.50 → 11 (and 0.50 costs the most per cycle). **This is not a
new discovery — `turbulence/coupled.py` already ships `strength_threshold: 0.25` on the frozen
velocity/Schur AMGs for exactly this reason.** The native hierarchy was simply built without a setting the
production path already used.

**The value-dependence is free HERE and only here.** A threshold makes the coarsening read `|A_ij|`, so a
rebuilt hierarchy is no longer structurally invariant. The flow block is frozen at the reference state and
never refreshed, so it costs nothing; the k/ω path refreshes (~48 times in a 61-step march) and a binding
decision keeps it at θ = 0 there.

**The INNER pressure sweep count was never varied before this, and is the best cost lever found.** Four
inner relaxations sit inside every outer SIMPLE sweep; the spec parser had tokens for outer sweeps,
levels, coarse size, threshold, aggregation, prolongator and `pressure_omega`, and none for this one — so
the largest single term in the smoother's cost was the one axis held fixed, while the outer count was
swept 4/8/16. Dropping 4 → 2 is worth **18 % of wall clock** for one extra cycle; 4 → 1 removes the Schur
matvec entirely (the peeled first sweep is a pure diagonal solve) and costs three cycles, which is too
many.

**The (outer, inner) frontier is FLAT — swept, and 8 x 2 survives.** Everything from 68 s to 87 s lies
within 28 %, so the pair is not finely tuned and there is no hidden corner:

| outer x inner | 8x2 | 8x1 | 12x2 | 16x1 | 8x4 | 16x2 | 24x1 | 16x4 |
|---|---|---|---|---|---|---|---|---|
| cycles | 11 | 14 | 8 | 8 | 10 | 7 | **6** | **6** |
| solve | **68 s** | 70 s | 73 s | 79 s | 83 s | 86 s | 87 s | 98 s |

One structural fact falls out and is consistent across the whole table: **six cycles is reachable two
ways — 16x4 at 98 s and 24x1 at 87 s — and the cheaper route is more outer sweeps with fewer inner ones.**
At any fixed cycle count, inner pressure relaxations are the expensive way to buy convergence.

Note for the march rather than for this probe: **12 x 2 buys 8 cycles for 73 s**, 7 % above the optimum.
A march's cost is dominated by staleness and refresh cadence rather than by any single solve, so fewer
cycles per solve may be worth more there than wall clock is here. That cannot be settled from a
single-state probe.

**TEN CYCLES IS NOT A FLOOR — REFUTED by measurement.** Three structurally unrelated leading inverses all
landed at 10–11, which raised the real possibility that the count was set by the trailing block or by the
coupling triangle the split discards, and that the comparison was saturated. It is not: at 16 outer
sweeps the native arm reaches **6 cycles**, below the incumbent's 11. The native hierarchy can be made
*stronger* than the incomplete-LU; what it is not yet is cheaper.

**Two structural levers measured and NOT worth taking.** Aggressive (squared-graph) coarsening *combined*
with a threshold does what it promises mechanically — level 0 coarsens 8.7x instead of 3.4x, the hierarchy
drops to 4 levels, the heavy level-1 Schur falls 1.2M → 0.5M nnz — and nets **zero**: per-cycle cost falls
15 % and the cycle count rises by two (11 → 13 at θ = 0.10; 10 → 11 at θ = 0.25). The fine level is ~60 %
of the smoothing cost, so removing intermediate levels can only ever reach the ~28 % that level 1 holds.
**Depth past three levels is separately closed**, non-monotonically (4 levels worse than 5) with a clean
aggregation at every level.

⚠️ **A kernel-level cost model over-predicted its own first result by 6x — treat its remaining estimates
as unmeasured.** A profiling pass modelled the Schur applications as ~64 % of the V-cycle and predicted
16 % from removing the multiply-by-zero first pressure sweep; measured, that change is worth **2–7 %**.
The probe used synthetic matrices with random column patterns, and real mesh-ordered matrices are far more
cache-friendly. Its other figures (per-level sweep schedule 31 %, asymmetric V-cycle 25 %, Schur
truncation 57 %) carry the same flaw and must be measured before being believed — the last one especially,
being the only proposal that would change what `M` is.

**⚠️ SCOPE, BINDING — THE 2.3x IS A ZERO-SHIFT NUMBER. Across the shift range the march runs, the
native hierarchy is ~4x slower.** Swept 2026-08-13 by overriding the operator shift on the fixed
converged state (`BFS3D_PROBE_BETA`, which is the only way to move β without also changing state — the
other entries in `STATES` are step-initial checkpoints and cost every arm a cycle or two). The
preconditioner takes the march's own floor, so β = 0.01 reproduces the shipped operator/preconditioner
mismatch (0.01 against 0.05):

| β | `split flow/ilu0` | native, 8x4 | native, 8x2 | ratio |
|---|---|---|---|---|
| **0** — the adjoint | 11 cyc, 29 s | 10 cyc, 83 s | 11 cyc, **68 s** | **2.34x** |
| 0.01 | 4 cyc, 15 s | 9 cyc, 76 s | 11 cyc, 69 s | **4.6x** |
| 0.05 | 2 cyc, 11 s | 6 cyc, 56 s | 7 cyc, 49 s | 4.45x |
| 0.10 | 2 cyc, 11 s | 5 cyc, 48 s | 6 cyc, 43 s | 3.9x |

**The asymmetry has a mechanism, and it is not going away.** The shift adds `β·d` to the diagonal, and an
incomplete-LU approaches exactness as diagonal dominance grows — so ILU(0) collapses to its floor (2
cycles, 11 s) the moment any shift is present and cannot go lower. An aggregation multigrid gets no such
windfall: its rate is set by smoother/coarse-space complementarity, which the shift barely improves. The
native arm does improve (68 -> 43 s) but from far behind. **The ratio is therefore bounded below by how
cheap the native V-cycle can be made, not by anything about convergence** — at β = 0.1 it needs 6 cycles,
which is a perfectly healthy preconditioner, and still costs 3.9x.

**So the native path's competitiveness is concentrated at β = 0, which is the ADJOINT and nothing else.**
The forward march floors β at 0.05 and never builds a preconditioner at zero shift; the transpose solve
behind every `jax.grad` meets exactly the unshifted operator, with no floor to soften it, and must
converge tightly. That is a real place to win and it is where a differentiable solver spends its gradient
budget — but it is not the march, and a 2.3x quoted without the shift beside it will be read as the march.

**⚠️ THE W-CYCLE AND PRE-SMOOTH REMOVAL ARE BOTH REFUTED ON THIS SADDLE.** Jasak, Jemcov and
Maruszewski (2007) measure, on a segregated LES pressure equation, a W-cycle at 26 iterations against a
V-cycle's 71 on the same coarsener and smoother, and their fastest arms all drop the pre-sweep entirely
— their stated reason being the residual re-evaluation a nonzero pre-sweep forces, which is exactly the
term that dominates a sweep here. Neither survives on the flow saddle. Measured at β = 0.1, strength
0.25, 4 outer x 2 inner sweeps, 5 levels, no singletons:

| arm | cycles | solve | s/cycle |
|---|---|---|---|
| **V-cycle with pre-smoothing** | **8** | **37 s** | 4.63 |
| W-cycle (`mu = 2`) | 6 | 46 s | 7.67 |
| no pre-smoothing | 14 | 41 s | 2.93 |
| W-cycle + no pre-smoothing | 12 | 46 s | 3.83 |

Each lever does mechanically what it promises and neither pays. The W-cycle buys 25 % of the cycles and
costs 66 % per cycle, because a W-cycle visits level *k* **2^k** times — at five levels that multiplies far
faster than the coarse share can absorb, and our level 1 carries a 1.6M-nonzero Schur where their deeper,
more aggressive hierarchies really are nearly free to revisit. Dropping the pre-sweep does cut per-cycle
cost by 37 %, exactly the mechanism claimed, and costs 75 % more cycles.

**⚠️ A first reading of this called convergence here "smoother-dominated, not coarse-grid-dominated".
That is TOO STRONG and two results of the same day contradict it: the strength threshold is a pure
coarse-SPACE change and was the largest win measured (4x), and the W-cycle DID cut cycles 25 % — it failed
on cost, not on effect. "Did not pay" is not "does not matter". What the evidence supports is narrower:
more smoother (8 -> 16 sweeps: 11 -> 6 cycles) and more coarse grid (W-cycle: 8 -> 6 cycles) each buy
25-45 % fewer cycles and each cost more than they return, so this sits at a MARGINAL-COST OPTIMUM rather
than in a regime where one half is limiting.** The mechanism below still explains why each specific lever
loses: On a symmetric Poisson under an incomplete-LU or
symmetric Gauss–Seidel smoother the coarse correction carries the solve, so revisiting it pays and a
post-smooth alone can clean up what an unsmoothed restriction lets through. On this indefinite saddle
under a SIMPLE smoother the fine level carries it, so extra coarse visits buy little and restricting an
unsmoothed residual costs a lot.

**Separating a DEFECT from an INVESTMENT does organize the results, and that much holds.** The coarse
space had a real defect — isotropic aggregation coarsening across the stiff wall-normal direction — and
repairing it was worth 4x. Investing *further* in the coarse grid once repaired has returned nothing at a
price worth paying: more levels, aggressive coarsening, both smoothed prolongators, orthonormalization,
and the W-cycle. But note the W-cycle case carefully — it improved convergence and lost on cost, so this
is a statement about marginal return, not about the coarse grid being irrelevant. Weigh a new
coarse-grid idea against the *cost* it adds, not against a claim that the coarse space is finished.

**The binding constraint is the smoother's absolute quality, which the balance measurement below puts at
`||S|| ~ 1.45` and `||E|| ~ 1.03` — where useful preconditioning needs both far below 1.** Each sweep
contracts very little, which is why many sweeps are needed and why many are expensive. **Improvement
therefore requires a smoother that is better PER UNIT COST, not more of the same one.**

⚠️ **Scoped to this operator, deliberately.** `mu` and `pre_smooth` are kept on `_VCycleOps` (defaulting
to a V-cycle with pre-smoothing, byte-identical) rather than deleted, because `_frozen_v_cycle` is shared
with the smoothed-aggregation path over the **symmetric pressure Schur** — which is the configuration the
literature result was actually measured on, and where it may well hold. What is refuted is the W-cycle on
the saddle under a SIMPLE smoother, not the W-cycle.

### THE NATIVE FLOW BLOCK IN A REAL MARCH — identical trajectories, but the RATIO GROWS WITH THE RUNG

**COMPLETED. Final: 60 steps / 3044 s against the incumbent's 67 / 1957 s — 1.56x — and the SAME
REATTACHMENT LENGTH, `x_r/h` 8.361 mid-span and 12.53 full-span, to four significant figures.** The native
preconditioner converges to the same solution, in SEVEN FEWER STEPS, and pays 1.56x in wall clock: it
produces better Newton directions and pays for them per application. 26 coarsening retraces are inside
that figure, worth roughly 80 s (~7 % of the gap), so removing them helps without changing the verdict.

⚠️ **READ THE RATIO AS A TREND, NOT A NUMBER, AND DO NOT QUOTE THE EARLY RUNGS.** Measured through the
march: **1.32x** (rung 1) → **1.27x** (rung 2) → **1.16x** (target rung, first step) → **1.57x** (target
rung, step 49, where the native arm is at 2342 s against 1489 s and the incumbent's ENTIRE 67-step march
took 1957 s). The early rungs run at high β where both preconditioners are cheap and they flatter the
native arm; the target rung runs at low β, which is exactly where an incomplete factorization's O(eps^2)
advantage is largest. At matched β inside the target rung the native needs **~1.35x the cycles** (9 against
6 at β = 0.0439; 17 against 13 at β = 0.0293). A first write-up of this section quoted "1.16-1.32x" as the
result, which was the most favourable point of a rising sequence.

Everything above is single-state probing. The preconditioner has now been run in the full `bfs3d`
continuation march, which required three things a probe never reaches, and which are the reason no earlier
arm could have been marched at all.

**The native flow inverse needed a `refactor_block`.** `BlockTriangularFieldSplit.refactor` RAISES on an
inverse offering neither `refactor_block` nor `refactor`, because a mid-march refresh must mutate the same
object the compiled Krylov solve holds — replacing it would recompile. A single-state probe exits before
that path. `leading_inverse` is now threaded through `FieldSplitAmgPreconditioner.build` and the coupled
march builder (the underlying `build_block_triangular_field_split` already supported it), and `compare.py`
selects on `BFS3D_FLOW_INVERSE=native`.

**The result, against an archived march of the incumbent** (`zen-newton-3f1b39`, PETSc ILU(0) on the flow
block, same trailing native inverse, same continuation schedule, same `refresh_on_cycles` 3):

| | steps | wall | step-14 `R` | step-38 `R` |
|---|---|---|---|---|
| incumbent | 67 total, 1957 s | — | 7.821e-06 | 4.155e-06 |
| native | in progress | — | **7.822e-06** | **4.155e-06** |
| ratio | | 1.32x (rung 1), 1.27x (rung 2), **1.16x** (step 39) | | |

**The residuals agree to FOUR SIGNIFICANT FIGURES at every step through both early rungs**, with identical
step counts, identical β schedules, and identical line-search factors. The native preconditioner is not
merely converging — it produces the same Newton steps. It also uses **equal or fewer cycles at most
steps**. They first separate in the target rung at step 41, where the native takes a full step
(α = 1.000, `R` 3.343e-02) against the incumbent's clipped one (α = 0.328, `R` 3.757e-02).

⚠️ **The archived march's column reach differs (3/3/3/2/2/2 against the shipped 3/3/3/3/2/2), and it does
NOT matter.** Measured from the two logs' own refresh breakdowns, the difference lands entirely in the
probe phase at **8.3 s against 7.7 s** — about 8 %, not the 24 % a column count suggests — which over ~10
refreshes is 0.6 % of wall. Measure the phase breakdown rather than reasoning from column counts.

**WHERE THE GAP ACTUALLY IS, from the same breakdown at a matched step 28:**

| | native | incumbent |
|---|---|---|
| refreshes | 8 | 9 |
| probe / refresh | 8.3 s | 7.7 s |
| **refactor / refresh** | **5.7 s** | **2.5 s** |
| cumulative wall | 931 s | 776 s |

Refresh accounts for **20 s of the 155 s gap — 13 %. The other 87 % is the V-cycle's per-application
cost**, which the per-iteration split measured independently at 1.45x. So the frozen-coarsening work below
is worth ~20-26 s of a 155 s gap: real, and NOT the lever. The lever is the apply.

**⚠️ `refresh_on_cycles` IS PER INNER ITERATION, NOT PER STEP, AND RAISING IT TO 8 SWITCHED THE REFRESH
OFF.** The trigger is `corrected >= refresh_on_cycles` against a single inner solve, capped at one refresh
per step. A step reporting `cyc 12` is the SUM over three inners at ~4 each, so nothing ever fired: every
step logged `pc none 0.0s`. The march then ground on a never-refreshed preconditioner and the line search
collapsed at step 9 of the LOWEST rung (α 0.469 → 0.030 → 0.000, β escalating, `R` rising) — a rung that
completes in 14 steps in every other configuration. At the shipped threshold of 3 the same march completes
that rung in **14 steps, 429 s**, full Newton steps throughout. The sizing error underneath it: 6-8 was
taken from the PROBE's per-solve count at rtol 1e-8, but the march solves to `forward_rtol` 0.3 and needs
**2-5 cycles per inner**. A cycle count does not survive a change of tolerance any more than it survives a
change of β.

**⚠️ THE COARSENING MOVES ON EVERY REFRESH, AND THE JITTED CYCLE RETRACES.** At `strength_threshold` 0.25
the aggregation reads `|A_ij|`, so a refresh at a developed state produces a different hierarchy: measured
**14 retraces** in one march, level sizes wandering (coarse dofs 312 → 348 → 340 → 288 → 296 → 268 → …).
The sibling nodal inverse rebuilds freely and calls itself structure-preserving only because its
aggregation reads SPARSITY alone — the property this arm gave up to win 4x. That trade was invisible until
the preconditioner met a march.

**✅ THE MECHANISM IS BUILT — `SmoothedHierarchy.refit(a)` (2026-08-13).** It holds this hierarchy's
prolongations and re-derives only what depends on values: each level's Galerkin operator `Pᵀ A P`, its
diagonal or per-cell block inverse, its spectral estimate, and the coarsest level's dense inverse. Shapes
cannot move, so a refitted hierarchy is a compilation-cache hit **by construction** rather than by the
sparsity-pattern-only argument that the strength threshold invalidated. Reached from the case as
`BFS3D_FLOW_FROZEN_COARSENING=1` (and `-fc` on a probe spec), off by default.

**It is EXACT where the flow arm runs, and NOT exact in general — the boundary is the prolongator.** With
`prolongation_smoothing="none"` — what this arm uses — the tentative prolongation is the aggregates' 0/1
indicator and holds no operator values, so freezing it freezes the partition and nothing else: at
`strength_threshold=0` a refit then reproduces a from-scratch rebuild in every level's values. With either
smoothed prolongator it *also* freezes a relaxation built from the old operator, and the coarse operators
then differ from a rebuild's. Both halves are pinned, so the exact agreement cannot be read as
unconditional.

**⚠️ AND THE "MEASURE MEMBERSHIP, NOT SIZE" CAUTION IS NOW DEMONSTRATED, NOT JUST ARGUED.** On a square
grid, aggregating along either direction gives the **identical count** — so rotating an operator's
anisotropy re-partitions the mesh completely while every level size is unchanged. That is now a unit test
(`test_refit_holds_a_partition_that_a_rebuild_would_move`), which is worth more than the argument: **a
march reporting stable level sizes is not reporting a stable coarsening**, and the 10–20 % size drift
recorded above is a lower bound on how much the partition moved, not an estimate of it.

**❌ AND FREEZING LOSES ON A MARCH — the stale coarse space costs FOUR TIMES what the retraces do
(measured 2026-08-13). Do not ship it; the case default stays on the rebuild.** Two full `bfs3d` marches
differing in **one flag**, `BFS3D_FLOW_FROZEN_COARSENING`; otherwise identical (native flow block at
4 outer x 2 inner sweeps, strength 0.25, no singletons, 5 levels, `max_coarse` 500, block splitting,
`omega` 1.0; native trailing inverse; field split; `refresh_on_cycles` 3; three Reynolds rungs):

| | steps | wall | **Krylov cycles** | final ‖R‖ | mid-span `x_r/h` |
|---|---|---|---|---|---|
| rebuild (the default) | 60 | **3044 s** | **327** | 8.076e-06 | 8.361 |
| frozen | 63 | 3153 s | **381 (+16.5 %)** | 1.689e-06 | 8.361 |

**Read the CYCLE row.** The frozen arm converged deeper (1.7e-06 against 8.1e-06) and took three more
steps, so part of the +109 s is work past the tolerance rather than inefficiency; the cycle count is not
subject to that and says the same thing. The like-for-like window — through **step 35**, the last step at
which the two arms agree to four figures — gives **+18 cycles (+12.1 %)** and **+66 s**, so the whole-march
figure is not an artifact of the trajectories parting. Same reattachment length either way.

**The per-rung split is the mechanistic result, and it is cleaner than the total:**

| | cycles | wall |
|---|---|---|
| rung 1 (steps 1–14, Re/100) | 49 → **49, bit-identical at every step** | 429 → **412 s (−17 s)** |
| rung 2 (steps 15–35) | 100 → **118 (+18 %)** | 776 → **859 s (+83 s)** |

**Rung 1 is a clean null: a coarsening derived at the initial condition is worth EXACTLY as much as one
rebuilt at every refresh, as long as the flow has not moved.** So the degradation is not drift in the
aggregation algorithm — it is the anisotropy PATTERN rotating under a partition chosen by reading `|A_ij|`
at a state the flow has since left. That is the "case against" (the eddy viscosity grows to ~150x molecular
and varies spatially) confirmed, against the "case for" (the anisotropy is geometric — wall-normal grading,
within-row conductance ratios of 64 median and 1700 max, 99 % of cells above 4:1), which does not survive.

**⚠️ THE RETRACE IS NOW PRICED, AND IT IS SMALL: ~3.4 s each, ~88 s over a march, ~3 %.** Rung 1's 17 s came
with *bit-identical* cycle counts over ~5 refreshes, so it is pure retrace removal with nothing else moving
— the one place in this comparison where the two effects are separated. That is what makes the verdict
quantitative rather than directional: the whole prize for eliminating retraces is ~88 s, and freezing spent
+109 s of cycles to collect it.
⚠️ **This prices the retrace AT 23040 CELLS and says nothing about how it scales.** The V-cycle is unrolled
over levels x sweeps x inner sweeps and the level count grows like log(n), so the per-retrace cost grows
with the mesh while this measurement does not. Do not carry the 3 % to a larger case.

**Freezing per RUNG is the surviving variant and is NOT measured.** Rung 1 shows a frozen partition is free
while the flow resembles its build state; rung 2 shows it degrades once the flow develops. Rebuilding at
each Reynolds boundary bounds the staleness to one rung. But note the ceiling before spending a run on it:
the whole retrace prize is ~88 s of a 1087 s gap to the incumbent, so this was never the lever.

### Sweep count, the zero-vector peel, and the forward solve cap — 2026-08-14

**FEWER SWEEPS WIN ON THIS MARCH: 2 against 4 is −16.8 % wall for +34.6 % cycles.** Full `bfs3d` marches,
native flow block, everything else identical (strength 0.25, no singletons, 5 levels, `max_coarse` 500,
block splitting, omega 1.0, 2 inner pressure sweeps, `refresh_on_cycles` 3):

| arm | steps | wall | Krylov cycles | final ‖R‖ | mid-span `x_r/h` |
|---|---|---|---|---|---|
| 4 sweeps | 60 | 3044 s | 327 | 8.076e-06 | 8.361 |
| **2 sweeps** | 63 | **2533 s** | 440 | 1.689e-06 | 8.361 |

The apply cost dominates strongly enough that buying cheapness with convergence pays. ⚠️ These arms part
at **step 10** (a materially weaker preconditioner returns different Newton directions), so this is not
the step-for-step controlled pair the frozen-coarsening comparison was; the direction is consistent
across both early rungs independently.

**✅ THE V-CYCLE MULTIPLIED THE LEVEL OPERATOR BY A VECTOR KNOWN TO BE ZERO, TWICE PER CYCLE (fixed).**
`_fixed_cycle_solve` began each pass with `b − A x` at `x = 0`, and every level's pre-smooth ran its
first sweep's `rhs − A g` at `g = 0`. The second is charged at **every level of every cycle**, and inside
a `fori_loop` the body cannot be specialized away. XLA does not fold it: measured 2.88 ms against 1.33 ms
for the peeled form on a 2.0M-nnz operator held as a jit constant.

`_VCycleOps` gained an optional `smooth_zero(level, b)`; families supplying none fall back to
`smooth(level, b, zeros)` and are byte-identical. **Exact, not approximate** — `smooth_zero(level, b)`
matches `smooth(level, b, zeros)` to **0.0** at every level at 1, 2 and 4 sweeps. Worth, on a 5-level
0.7M-nnz block:

| sweeps | 1 | **2 (shipped)** | 4 |
|---|---|---|---|
| saving | 36.0 % | **12.5 %** | 5.7 % |

The peel removes a fixed **two** operator applications out of `2·sweeps + 2`, so it is worth more at
fewer sweeps — which makes low sweep counts more attractive than the sweep table alone suggests. The
argument is the one the `pre_smooth=False` branch already made in its own comment, and the peel is the
one `_simple_correction` already did for the first pressure sweep one loop further in; it simply had not
been applied to the two outer sites.

**✅ A SINGLE FORWARD SOLVE WAS BOUNDED ONLY AT 60 RESTARTS, LONG PAST THE POINT ITS ATTEMPT WAS DOOMED
(fixed).** `cycle_budget` and `abort_above_inner_cycles` are both tested in `DualTimeStep`'s `cond`,
i.e. **between** inner iterations, so neither can stop a solve already running — and neither bounded any
of the 41–45 cycle solves in the archived marches. Meanwhile `forward_march` discards an attempt whose
worst solve passes `retry.abort_above_cycles` without reaching target, so a solve at 11 corrected cycles has
already determined its work will be thrown away.

Measured over **671 solves across three marches**: no accepted attempt exceeded **9** corrected cycles,
discarded ones ran **39–45**, and the distribution is **empty between**. The archived logs put the
recoverable work at ~404 s of a 2533 s march.

**Two traps decide the constant, and both rule out the obvious choice of 10:**
- **`max_restarts` is in RAW `lineax` restarts** (a fixed +2 per solve); `retry.abort_above_cycles` is in
  **corrected** cycles. A corrected cap of 12 is `max_restarts = 14`.
- **The march's test is `max_inner_cycles > retry.abort_above_cycles`, STRICTLY.** A cap landing exactly
  on the threshold does not trip the redo, so the step **accepts the truncated, non-converged
  direction** instead of re-running it on the refreshed preconditioner — turning a doomed attempt into a
  bad accepted step. The corrected cap must be strictly above the threshold.

Shipped as `coupled_amg_continuation(forward_max_restarts=…)`, library default unchanged at 60; the case
derives it as `retry.abort_above_cycles + 4` from a single scaled constant so the two cannot drift.
⚠️ **The 671-solve distribution describes UNCAPPED solves.** Whether a truncated direction ever needs an
extra escalation rung is not answerable from it, and the first capped march is the first evidence.

**❌ THE FIXED SHAPE LADDER IS DEAD — cycles IDENTICAL to the last step, wall +52.6 %.** Full march at
2 sweeps with `shape_headroom` 1.25 against the 2-sweep control: **431 cycles against 431** through step
62, wall 3808 s against 2495 s. The padding is provably inert (which is the mechanism working exactly as
designed) and unaffordable, because it is charged per **application** (~440 cycles) while the retrace it
avoids is charged per **refresh** (~26). The 25 % headroom hits every level three ways — operator
matvecs, vector-length work in every diagonal scaling, and the dense coarse solve, which goes as the
**square** of its size. Do not revive it at this problem size; the minimum viable headroom is set by how
far the partition genuinely moves (~12 %), which still costs several times the retrace saving.

**⚠️⚠️ AND THE RETRACE WAS NEVER PRICED CORRECTLY — the recompile was UNCONDITIONAL. Read this before
citing any retrace figure.** The flow arm built `jax.jit(lambda …)` on a **fresh lambda every refresh**,
closing over the hierarchy and the smoother pieces. A new `jax.jit` object starts with an empty cache, so
it recompiled **whether or not the coarsening moved**. Consequences that still stand:
- **`refit`'s 17 s rung-1 saving was HOST AGGREGATION, not retrace**: it skips `_mis_aggregate` and
  friends but still rebuilt the jit. So the "~3.4 s per retrace, ~88 s per march, ~3 %" figure recorded
  above is an attribution error; that time is Python, not XLA.
- **Compile was a SEPARATE cost that neither refuted experiment removed**, additional to the 88 s rather
  than part of it.

**✅ INDEPENDENTLY CORROBORATED, by a separate implementation and a separate measurement.** The same
defect was diagnosed and fixed a second time, in parallel and without sight of the above: a different
cycle structure (the hierarchy and per-level state passed to a jit built once in the constructor rather
than to a module-level `filter_jit`), measured on a **different** fixture — a 8000-dof synthetic saddle
at the case's own flow settings, 3 repeats on a quiet machine:

| one refresh, min-max of 3 | refactor | first apply after | total (min) |
|---|---|---|---|
| closure | 0.049-0.050 s | **0.152-0.187 s** | 0.202 s |
| argument | 0.047-0.048 s | **0.005-0.005 s** | **0.051 s** |
| trailing block, either way — the CONTROL | ~6.5 s | 0.002 s | ~6.45 s |

**3.93x**, against the 3.3x measured above at the real block's shape — consistent in direction and
magnitude from two independent implementations. The control is the informative row: the trailing block,
which already passed its hierarchy as an argument, moves **1.01x**, so the effect is the recompile and
not the setup. Note also the *spread*: the closure form's first apply varies 0.152-0.187 s (compile-time
variance) where the argument form is 0.005 s with none.

### The peel and the cap ON A MARCH — the cycles land, the seconds mostly do not (2026-08-14)

Both changes marched at 2 sweeps against the 2-sweep control, native flow block, everything else equal.
The first attempt's wall clock is **void** — pytest tiers, ruff and JAX probes ran on the same machine
throughout it — so it was re-run on a quiet one. Cycle counts and trajectories from both agree, being
contention-immune.

| arm | steps | wall | step cycles | **solve cycles** | max single solve | mid-span `x_r/h` |
|---|---|---|---|---|---|---|
| control, 2 sweeps | 63 | 2533 s | 440 | **622** | 45 | 8.361 |
| peel + cap (contended, wall VOID) | 63 | 3100 s | 440 | 529 | 12 | 8.361 |
| **peel + cap (clean)** | 63 | **2486 s** | 440 | **529** | **12** | 8.361 |

**✅ THE CAP DOES EXACTLY WHAT IT WAS SIZED TO DO, AND COSTS NOTHING IN QUALITY.** Three solves ran
41/43/45 cycles in the control and were truncated to 12; **no solve exceeded the cap**; total solve
cycles fell 622 → 529 (**−93, −15 %**). The accepted trajectory is **identical on all 63 steps** with the
same root and the same reattachment length, so a truncated direction cost **no extra escalation rung** —
the one risk the 671-solve archive could not speak to, since every solve in it ran uncapped.

**⚠️ READ THE STEP TABLE'S CYCLE COLUMN CORRECTLY OR THE CAP LOOKS INERT.** It reports the **accepted**
attempt only, and the cap truncates **discarded** ones — so identical step-table cycles (440 in every
arm) is what a *working* cap produces, not evidence it never fired. The effect is visible only in the
per-inner tables. This was misread once here before the inner-table parse settled it.

**❌ BUT A 15 % CYCLE REDUCTION BOUGHT 1.9 % OF WALL, AND THE PEEL IS INVISIBLE AT MARCH SCALE.** Per
rung: **+1.0 % / +4.9 % / −5.6 %**. The whole saving sits in rung 3, where the cap's three solves live
(~85 s, against ~206 s that a 2.22 s/cycle cost model predicts). The peel measured **12.5 % of the
V-cycle** on a 5-level 0.7M-nnz synthetic and should have shown in every rung; rungs 1 and 2 are
slightly *slower*.

**Two candidate explanations were offered, and the second is now MEASURED to be sufficient on its own:**
- the synthetic does not transfer to the real 21M-nnz block (plausible — a cost model on synthetic
  sparsity already over-predicted itself 6× once here, real mesh-ordered matrices being far more
  cache-friendly); or
- **the instrument's own spread swamps a 12.5 % effect.** ✅ This one is now established, so the peel's
  disappearance needs no further explanation — though it does not *refute* the first, which remains
  untested.

**✅ THE NOISE FLOOR IS MEASURED, AND IT IS LARGE (2026-08-14).** `BFS3D_PROBE_SPLIT=1` at `state-00067`,
β = 0.1, shipped column reach, on a machine with **no competing compute**. The decisive number is an
*internal control*: the two monolithic ILU(0) arms in one run are the **same preconditioner built twice**,
so their apply must cost the same — and they measured **193 ms and 221 ms, 14 % apart, inside a single
quiet process.**

| arm, β = 0.1 | jacobian product | preconditioner | cycles |
|---|---|---|---|
| monolithic ILU(0), march solver | 37 ms | **193 ms** | 1 |
| monolithic ILU(0) | 36 ms | **221 ms** | 2 |
| `split flow/ilu0` | 36 ms | 118 ms | 2 |
| `omega10` — the shipped native bundle | 36 ms | 106 ms | 9 |

**So a per-application timing from this probe cannot resolve anything below roughly 15 %**, and the
12.5 % peel is under that floor. The Jacobian product is the clean counter-example: it is the *same*
computation in every arm and held 36–37 ms across the four (±3 %), which shows the spread lives in the
preconditioner apply rather than in the timer.

⚠️ **AND THE CYCLE COUNTS ARE EXACTLY REPRODUCIBLE**, which is the other half of the result and the
reason to keep quoting them: a second identical invocation returned **1 / 2 / 2 / 9 cycles** and residuals
agreeing to *every digit*, while its timings moved by up to 21 %. Cycles are contention-immune; seconds at
this granularity are not.

⚠️ **A TRAP THIS ALMOST WALKED INTO, worth more than the number.** Run 1 has the native bundle at 106 ms
against the incumbent's 118 ms — the native apply looking **cheaper**, which would overturn the recorded
"~1.45× more expensive per application". The second run has 128 ms against 112 ms, i.e. **the ordering
reverses**. At 2 sweeps the two are indistinguishable per application and **the sign cannot be called**;
what can be said is only that the 1.45× figure was measured at **4 sweeps** and does not describe the
2-sweep bundle the case now ships. A single run would have produced a confident, wrong headline.

⚠️ **Scope: this measures the PER-APPLICATION instrument, not march wall.** Whether a whole march repeats
to better than a few percent is still unmeasured, and the "±1.0 % / +4.9 % / −5.6 %" per-rung figures
above are still uninterpretable for that reason. What is settled is that the cheap discriminator is too
blunt for a 12.5 % effect, so reaching for it again on a sub-15 % question is wasted time.

### Over-relaxing the SIMPLE correction — real, small, and one step from a cliff (2026-08-14)

**The grammar capped `omega` at 1.0, so no arm in this campaign had ever tested over-relaxation.**
`_leading_inverse` read `omega = 1.0 if "-o10" in rest else 0.7` — a binary token — while the class always
took a float. So the recorded "omega 1.0 is worth a cycle" was measuring the **top of the grid**, not an
optimum. `-oNN` (NN/10) now spans it.

*Configuration:* `bfs3d` `state-00067`, **beta = 0 on operator and preconditioner** (the adjoint's
operator), **uniform** column reach, real right-hand side `-R(state)`, GMRES restart 15 to rtol 1e-8 on the
**TRUE** residual, 58-restart cap. Everything but `omega` pinned to the shipped bundle: 2 sweeps x 2 inner,
strength 0.25, no singletons, 5 levels, `max_coarse` 500, per-cell block splitting, ILU(0) trailing.

| omega | 0.7 | **1.0 (shipped)** | **1.2** | 1.4 | 1.6 | 1.8 | 2.0 |
|---|---|---|---|---|---|---|---|
| cycles | 30 | **20** | **19** | 58 cap | 58 cap | 58 cap | 58 cap |
| TRUE rel | 1.99e-10 | 7.24e-10 | 5.62e-10 | 7.1e-05 | 1.2e-02 | 1.8e-02 | 5.0e-02 |
| solve | 84 s | 68 s | **66 s** | 136 s | 139 s | 143 s | 141 s |

**Confirmed in direction, REFUTED in magnitude.** The optimum is above 1.0 and is worth **one cycle (~5 %)**
— not the 1.8x a synthetic saddle predicted. **The synthetic's optimum, 1.4–1.6, is exactly where the real
block stops converging.**

**⚠️ THE DERIVATION IS WRONG, AND PREDICTS ALMOST EXACTLY THE WORST POINT.** The argument was: the Frobenius
velocity predictor is a least-squares fit and so shrinks by construction (median 0.53), SIMPLE's dropped
neighbour terms shrink again, and a correction behaving like `gamma * A^-1` wants a Richardson factor near
`1/gamma ~ 1.9`. Measured, 1.8 is the second-worst arm on the ladder. The premise that fails is
**uniform** shrinkage: the SIMPLE correction's error varies by mode, so one scalar cannot compensate for
it, and pushing the scalar far enough to fix the worst-shrunk mode amplifies the rest. That accounts for
both halves of the result — a small gain from mild over-relaxation, and a hard cliff far below the derived
value.

**DO NOT MOVE THE DEFAULT — 1.0 STAYS.** 1.2 buys 5 % and sits ONE GRID STEP from a cliff that costs
convergence outright, and past the cliff the degradation is monotone (7.1e-05 → 1.2e-02 → 1.8e-02 →
5.0e-02), i.e. genuine amplification rather than a near-miss on the restart budget. The synthetic's one
durable finding was that the optimum **moves with operator hardness**, which makes a fixed 1.2 fragile
across a march that visits a wide range of operators. A per-level, per-refresh derived relaxation would be
a different proposition, and at a 5 % ceiling it is not worth building.

**⚠️ METHOD, worth more than the result: RUN THE WHOLE LADDER, INCLUDING PAST WHERE YOU EXPECT THE
OPTIMUM.** The arms 0.7 / 1.0 / 1.2 alone read as monotone improvement and invite pushing further — and
the next step diverges. A sweep that stops at its best point cannot see a cliff one step beyond it.

**⚠️ AND THE SELF-CHECK EARNED ITS KEEP.** The first attempt ran at the case's shipped column reach and the
control stopped at a TRUE relative residual of **4.779e-03**, so the harness refused to report any arm.
That value is recorded elsewhere in this file as the benign signature of the loose march-solver stop at
`(3,3,3,3,2,2)` — so without the gate there would have been seven plausible omega numbers measured against
a control sitting at 5e-3. Every beta = 0 measurement in this campaign is at **uniform** reach for this
reason; the shipped reach doubles the incumbent's zero-shift cost (22 cycles against 11) and is a separate
axis.

**Unrun:** the same ladder at beta = 0.1. A shift raises diagonal dominance and should pull the optimum
*toward* 1.0, so the march's own regime is where over-relaxation has least to offer; what it would settle
is whether the cliff ever descends BELOW 1.0, which would put the shipped default near an edge.

### The trailing ablation, and the two structural ideas it points at (2026-08-14)

**The question: how far below the coupled cycle count can a better TRAILING inverse alone take it?** The
figure that had stood in for this — "2 restart cycles against a floor of 1" — is a **standalone** solve of
the `[k, ω]` block, which production never performs: one Krylov iteration runs on all six fields with a
block-triangular `M`, so the count is a property of the whole preconditioned operator and of the coupling
the split discards, not of either block alone.

*Configuration:* `bfs3d` `state-00067`, **β = 0**, **uniform** column reach, real right-hand side
`−R(state)`, GMRES restart 15 to rtol 1e-8 on the TRUE residual, 58-restart cap, **leading inverse held at
PETSc ILU(0) in every arm**.

| trailing inverse | cycles |
|---|---|
| PETSc ILU(0) (control, reproduces the recorded 11) | **11** |
| **native nodal, 4 sweeps (SHIPPED)** | **11** |
| native nodal, 2 sweeps | 16 |
| native nodal, 1 sweep | 28 |
| damped Jacobi | 38 |
| near-exact factorization (diagnostic bound) | **58 cap, 2.5e-05 — WORSE than all of them** |

**Three results.** The shipped native trailing inverse **matches PETSc ILU(0)** in the coupled system,
which nothing on record established (the recorded 11 used ILU(0) on *both* halves). **Cutting its sweeps
is measured harmful** — 4 → 2 costs +45 % of coupled cycles, 4 → 1 costs +155 % — which settles the
"4 sweeps is an unexamined default" question in the opposite direction, and is what the record's own
(false) claim that this case already shipped at 1 sweep would have cost. And **more quality does not
help**: nothing in this ablation supports spending on the trailing block's accuracy.

**✅ THE BOUND ARM IS NOW INTERPRETABLE, AND THE BOUND IS REAL — measured per field 2026-08-14.** Its
residual had been reported only as **1.693e-06 globally on a random right-hand side**, and this block's k
and ω rows differ by some eight orders of magnitude, so a global norm is ~100 % ω and would report a
factorization that solves ω beautifully and k not at all as "near-exact". Reported per field, at the
standing configuration of this section (`state-00067`, β = 0, uniform reach, 58-restart cap, leading
inverse held at PETSc ILU(0)):

| trailing factorization, random right-hand side | relative residual |
|---|---|
| global | 1.693e-06 |
| **f0 = k** | **1.936e-06** |
| **f1 = ω** | **1.416e-06** |

**The two fields agree to within 1.4×, so the factorization is as tight in k as in ω** and reading (2) —
"the bound was never tight in k, so it bounds nothing" — is **REFUTED**. Reading (1) stands: an exact
inverse of an ill-conditioned block (recorded cell-block condition ~1e12, ‖B⁻¹‖ ~9.5e8) **amplifies the
coupling the split discards**, where a fixed-cycle V-cycle is bounded and cannot. This has a precedent on
the flow side: inverting the Schur *more* accurately with ×2/×4/×8 V-cycles measured ρ 41.6/48.7/48.5,
strictly worse, and was read as "the operator being inverted is the wrong one".

The same run reproduced every number around it — monolithic ILU(0) 11 cycles / 8.474e-11, `split
flow/ilu0` 11 / 6.393e-11, and the exact arm 58 cap / 2.461e-05 — so the arm is not a one-off.

⚠️ **Quote the CYCLES and the residuals from this run, not its seconds.** Another worktree started a full
test tier partway through it, so the control's timings were taken on a quiet machine and the exact arm's
were not — the two are on opposite sides of that boundary and are **not comparable**. Cycle counts and
residuals are contention-immune and are the load-bearing evidence here. That a near-complete factorization
of this block is far more expensive to build than an incomplete one is not in doubt on other grounds, but
this run does not price it.

**So "the trailing inverse's job is to be WELL-BEHAVED, not accurate" is now a supported claim rather than
a reading.**

⚠️ **BUT THE MECHANISM NAMED IN READING (1) IS WRONG, and the cross-coupling measurement below refutes
it.** Reading (1) says the exact inverse "amplifies **the coupling the split discards**". Measured, the
discarded triangle `∂R_flow/∂turb` is **0.09 %** of the diagonal blocks — there is essentially nothing
there to amplify. What the split *retains*, `∂R_turb/∂flow`, is **83 %**, and that is what feeds the
trailing block's right-hand side. **The surviving mechanism, offered as a hypothesis and NOT measured:** an
exact `B⁻¹` on an ill-conditioned block (cell-block condition ~1e12, ‖B⁻¹‖ ~9.5e8) amplifies whatever error
reaches it, and the error that reaches it is the *approximate flow solve's*, arriving at full strength
through the retained 83 % coupling. A fixed-cycle V-cycle is bounded and cannot do that. Note the flow-side
precedent still stands on its own terms (inverting the Schur more accurately measured ρ 41.6/48.7/48.5,
strictly worse); what changes is which coupling carries the error here.

⚠️ **One scope limit, stated because the arm is a bound.** The residual is measured on a *random* right-hand
side, not on the `r_turb − C·x_flow` the coupled solve actually presents to this block. That is the right
choice for asking "is the factorization tight in k" — an unbiased probe — but it is not evidence about the
per-field scale of the right-hand side the split really sees.

**⚠️ ORDERING (flow-first against turbulence-first) is measured and is nearly a tie**: 11 against 13 at
β = 0, and **4 against 4** at the forward operating point, where the operator discriminates between
neither. Flow-first ships. The ordering decides *which* cross-coupling is discarded — flow-first retains
`∂R_turb/∂flow` and drops `∂R_flow/∂turb`.

**✅ THE TWO CROSS-COUPLING NORMS ARE NOW MEASURED, and the asymmetry is ~900x (2026-08-14).**
`field_coupling.py` at `state-00067`, on the symmetrically equilibrated Jacobian (which is what makes a
block norm comparable across fields whose units differ by orders), 39.2M nonzeros:

| partition (leading \| trailing) | drops | keeps |
|---|---|---|
| **`[u,v,w,p] \| [k,ω]` — the shipped ordering** | **0.09 %** | **83.10 %** |
| `[k,ω] \| [u,v,w,p]` | 83.10 % | 0.09 % |
| `[k] \| [ω]` | 0.19 % | 12016 % |
| `[ω] \| [k]` | 12016 % | 0.19 % |

Both are Frobenius norms of the off-diagonal triangle against the diagonal blocks the two hierarchies
invert. **`∂R_turb/∂flow` dominates `∂R_flow/∂turb` by roughly 900x, so flow-first's win is explained** —
it retains the overwhelmingly larger triangle. The raw grid says why, and it is physically ordinary: the
flow equations depend on turbulence only through the eddy viscosity (`∂R_u/∂k` 26.8 and `∂R_u/∂ω` 0.218
against a `u` diagonal of 244; `∂R_p/∂ω` is 1.6e-03), while the turbulence equations depend on the flow
through production terms in the velocity gradients (`∂R_ω/∂p` 2.12e+04, `∂R_ω/∂v` 1.29e+04 against an `ω`
diagonal of 172). The k↔ω pair repeats the pattern far more extremely: `∂R_ω/∂k` 2.99e+04 against
`∂R_k/∂ω` 0.474, a 63000x asymmetry, which corroborates the recorded k↔ω ordering result independently.

**⚠️ AND READ THE ORDERING TIE AGAINST THIS, because together they say something stronger than either
alone: turbulence-first DISCARDS the 83 % triangle and costs only TWO CYCLES (13 against 11).** So even the
dominant cross-coupling is nearly irrelevant to how fast the preconditioned Krylov solve converges. **The
split's discarded triangle is not what limits it, in either direction** — which is a much sharper statement
than "the ordering is a tie", and it is the fact that kills the alternation idea below.

⚠️ **A Frobenius norm is not an operator norm**: it bounds the block's action rather than equalling it, so
"0.09 %" bounds how much the discarded triangle can matter and does not prove it is inert. The ordering
measurement is the independent leg, and the two agree. The run shared the machine with other test tiers;
block norms are contention-immune, so only its wall clock is affected and nothing here depends on it.

**❌ ALTERNATING THE TWO BLOCKS (block Gauss–Seidel instead of one forward substitution) IS CLOSED BY THE
TWO MEASUREMENTS ABOVE — do not build it.** The idea was that the apply is ONE pass — one flow V-cycle, one
sparse coupling product, one trailing V-cycle (`cycles=1` on both inverses) — so the flow solve never sees
turbulence, and a second pass would pick up the discarded coupling. Its whole value rests on there being
something in `∂R_flow/∂turb` worth capturing, and there is not:

- **The discarded triangle is 0.09 %** of the diagonal blocks. A second pass would buy that, at best.
- **The retained triangle is 83 %, and DISCARDING it costs two cycles** (turbulence-first, 13 against 11).
  So even if the two triangles were swapped in size, the prize would be about two cycles.
- It also needed `∂R_flow/∂turb` **assembled**, which the split does not form — real work, for that prize.

The design was otherwise sound and is worth remembering as a *technique* rather than as a candidate here: a
fixed number of alternations keeps `b → x` linear (which the non-flexible outer GMRES requires) and the
transpose stays closed-form, being the reversed product of transposes that
`BlockTriangularFieldSplit.apply(transpose=True)` already performs for one sweep. What is refuted is
alternation **on this split**, not alternation.

**Where that leaves the trailing block:** its quality buys nothing (the ablation), its coupling buys about
two cycles (the ordering), and the coupling it discards is 0.09 % (the norms). Three independent
measurements now say the same thing, so **stop looking for coupled cycles in the trailing half** — the
remaining reason to touch it is the coarse-grid **scalability** argument, which is about how the dense
coarsest solve grows with the mesh and not about this case's cycle count.

### The zero-vector peel reaches the TRAILING block too (2026-08-14)

`_VCycleOps` has carried an optional `smooth_zero` since the flow block's peel, but the point smoother
the transported scalars run had no such specialization, so the trailing V-cycle kept multiplying its
level operator by a vector known to be zero — once per level per cycle, on the pre-smooth. `_jacobi_smooth_zero`
supplies it and `convection_multigrid_solve` passes it, which reaches the trailing inverse and the frozen
velocity/Schur AMGs alike.

**Exact, and checked as such rather than argued.** `A x` at a zero vector is exactly zero in floating
point, so `b - A x` is `b` bit-for-bit; the peel is the same arithmetic with two operations removed. Pinned
two ways: `_jacobi_smooth_zero` against `_jacobi_smooth(..., zeros)` at 1/2/4 sweeps × both damping
settings (`test_the_zero_guess_peel_matches_the_general_jacobi_sweep_exactly`), and the whole
`fixed_cycle_solve` with and without it across sweeps × damping × cycle counts — **bit-identical
everywhere**.

⚠️ **The size is INHERITED, not measured here.** The flow-side peel measured 36 % / 12.5 % / 5.7 % of a
V-cycle at 1 / 2 / 4 sweeps on a 5-level 0.7M-nnz synthetic; the trailing block ships at **4 sweeps**, so
the low end of that range is the relevant one. And the same synthetic's 12.5 % is separately on record as
having failed to appear at march scale — see the noise-floor entry: a per-application effect of this size
is **below the measurement floor** on this case, so take the peel because it is free and exact, not
because a march will show it.

### ✅ THE DUAL-TIME INNER TOLERANCE WAS COSTING A THIRD OF THE MARCH (2026-08-14)

**`inner_tol` shipped at 1e-3 where `DualTimeStep`'s own docstring says a loose value is enough ("e.g.
0.05 -- the outer march re-solves each timestep anyway"). Fifty times tighter than documented, never
tested, and worth 33 % of the march.** Three full `bfs3d` marches, native flow block at 2 sweeps,
otherwise the case's own settings, on one machine at one commit with only `inner_tol` moved. **Same root,
same reattachment length (`x_r/h` 8.3611 mid-span, 12.53 full-span) in every arm:**

| `inner_tol` | steps | wall | Krylov cycles | inner iterations | final ‖R‖ |
|---|---|---|---|---|---|
| 1e-3 — as shipped | 63 | 2253 s | 445 | 211 | 1.689e-06 |
| **1e-2 — the new default** | **63** | **1510 s** | 338 | 166 | 1.828e-06 |
| 5e-2 — the docstring's value | **64** | 1568 s | **277** | **145** | 1.168e-06 |

**Why loosening is free: `G` and `R` are different quantities.** The inner loop drives
`G = R + β d (φ − φⁿ)`; the march is judged on `R`, and at the next anchor `R = −β d (φ − φⁿ)` — set by
**where φ lands**, not by how far `G` was driven. Two inner iterations place the step as well as three,
and the outer contraction never sees the difference. That is why the step count is identical at 1e-2
while a fifth of the inner Newton iterations disappear.

⚠️ **AND 0.05 IS PAST THE OPTIMUM — the arm with the FEWEST cycles and the FEWEST inner iterations is
the slower one.** It saves 61 cycles and 21 inner iterations against 1e-2 and loses on wall, because it
spends an **extra outer step** (64 against 63), and a step costs a Jacobian assembly and a refresh check.
**Read the failure mode: this knob does not diverge when pushed, it quietly trades inner work for outer
steps.** Nothing looks wrong — ‖R‖ ends *deeper* at 1.168e-06 and the reattachment length is unchanged —
so an arm judged on cycles, or on inner count, or on final residual, picks the slower configuration. Only
wall clock and the step count separate them.

**This is the second time in this file that running the ladder PAST the expected optimum was what
produced the answer** (the first was `omega`, where 0.7/1.0/1.2 read as monotone improvement and 1.4
diverged). There the cliff was loud; here it is silent. Both were invisible from the best measured point.

⚠️ **Each point is ONE march and this case still has no measured march-level noise floor.** The
1e-2/5e-2 wall gap is 4 %, which a single sample cannot resolve on its own — so that comparison rests on
the **step count and cycle count**, which are contention-immune, and not on the seconds. The 1e-3 → 1e-2
result does not need that care: 33 % of wall alongside a 24 % cycle drop at an identical step count.

⚠️ **The optimum is bracketed, not located.** The two points either side of it differ by 5×, and the
region between them is flat to 4 %, so a finer sweep has little to find. `BFS3D_INNER_TOL` and
`BFS3D_INNER_STEPS` are exposed for anyone who wants to try.

**✅ AND IT TRANSFERS TO THE SHIPPED PATH, WHICH THE THREE MARCHES ABOVE DID NOT TEST (2026-08-15).** All
three ran the **native** flow block, which the case does *not* default to — so the default this change
governs (`flow inverse: petsc`) was unmeasured when it was made. It has since been run, on a tree carrying
this change plus ~20 later merges:

| arm | `inner_tol` | steps | wall | cycles | inner iterations | final ‖R‖ |
|---|---|---|---|---|---|---|
| **petsc — the shipped default** | **1e-2** | **59** | **1197 s** | 232 | 155 | 1.861e-06 |
| petsc — archived baseline | 1e-3 | 67 | 1957 s | — | — | — |

Same root, same `x_r/h` 8.3611. ⚠️ **Do not attribute the whole 39 % to `inner_tol`**: that archived
baseline predates the coarse-solve factorization, the trailing zero-guess peel and twenty-odd merged
changes, so the two runs differ in more than this knob. The direction is unambiguous; the decomposition
is not available from this pair.

### ONE AGGREGATION, TWO APPLIES — and why the CPU half cannot use `scipy` (2026-08-15)

**The two multigrids over this operator were never two methods, only two *applies* of one method.** The
coarsening in `solve/multigrid.py` is already a host computation in `scipy`; only the apply is traced. So
a host apply over the *same* hierarchy gives a CPU path relaxed by an incomplete factorization beside the
traced path relaxed by SIMPLE or Jacobi — one aggregation, one refresh path, one coarse space, and only
the smoother differing by machine. Built as `solve/host_vcycle.py` (`HostVCycleInverse`,
`host_ilu_inverse`), satisfying the same `n_dofs` + `apply(residual, transpose=…)` contract as every
other frozen inverse, so it needs no new plumbing and does not touch the traced path.

**The transpose is BUILT, not borrowed.** A V-cycle is symmetric only if its smoother is, and an
incomplete factorization is not: `M^T` is the same recursion with the operator, the coarse solve and the
smoother each transposed and the pre/post smoothing exchanged. Pinned by `<y, M x> == <M^T y, x>` on a
deliberately NONSYMMETRIC operator, swept over cycle and sweep counts, beside a guard that the flag is
not simply ignored — on a symmetric fixture the identity would pass for an implementation that returned
the forward cycle for both.

**⛔ AND `scipy.spilu` CANNOT SUPPLY THE SMOOTHER — this is the finding, and it took four wrong
hypotheses.** On the `bfs3d` flow block the factorization raises `Factor is exactly singular`, or worse
returns and applies to NaN. Measured per level, at a state where the shipped PETSc ILU(0) reaches 11
cycles:

| level | dofs | COLAMD + partial pivoting | NATURAL + diagonal pivoting | ‖A M⁻¹r − r‖/‖r‖ |
|---|---|---|---|---|
| 0 | 92160 | exactly singular | 0.96× nnz, `max|U|` **9.44e+23** | **2.046e+38** |
| 1 | 26828 | exactly singular | 0.96× nnz | 1.020e+01 |
| 2 | 6540 | exactly singular | 0.97× nnz | 1.340e+00 |
| 3 | 1448 | ok | 1.00× nnz | 1.100e+00 |

**Every arm on the fine level is unusable and levels 1–2 leave a residual ABOVE one.** What was
eliminated on the way, each of which looked like the answer first:
- **not the equilibration.** Every level equilibrates to `|diag|` min = median = **1.00** with **zero**
  exact zeros — the operator reaching the factorization is perfectly scaled.
- **not the ordering, and not the pivoting**, though both are real and both are needed: `spilu` applies a
  COLAMD column permutation and partial pivoting by default, either of which discards the cell-major
  interleave. Fixing them converts an exception into a returning-but-garbage factor, which is worse.
- **not a missing pivot shift.** PETSc's `MatILUFactorSymbolic_SeqAIJ` takes a dedicated `ilu0` path at
  identity permutation, performs **no row pivoting** (pivots are used in place), and applies **no shift**
  unless `info->shifttype` is set — which `amg_preconditioner.py` does not set. So the incumbent factors
  this operator **unshifted** and succeeds.
- **it is WHICH ENTRIES ARE KEPT — and `drop_tol = 0` does not fix it.** The first wording here said
  "the dropping", which is imprecise: switching value-based dropping off entirely still fails.

  | level 0 arm (NATURAL, no pivoting) | fill | `max\|U\|` | ‖A M⁻¹r − r‖/‖r‖ |
  |---|---|---|---|
  | `drop_tol` 1e-4 | 0.96× nnz | 9.44e+23 | 2.046e+38 |
  | **`drop_tol` 0 — scipy's closest to ILU(0)** | **0.96× nnz** | 2.46e+04 | **1.028e+08** |
  | `drop_tol` 0 | 1.91× | 8.56e+83 | NaN |
  | `drop_tol` 0 | 3.82× | 3.91e+04 | 2.125e+13 |

  **The second row keeps 0.96x the operator's nonzeros — the right COUNT — and is still useless**, which
  is the whole point: SuperLU chooses *which* entries to keep by magnitude within a memory budget, so it
  drops pattern entries and keeps fill ones. Same size, **different set**. That is the difference between
  ILUT and ILU(0), and no parameter closes it. Note also that MORE fill makes level 0 worse rather than
  better — the signature of a wrong factorization, not an under-resourced one.

  ⚠️ **Read the coarse levels carefully before quoting them.** Levels 1–3 leave residuals of ~1.2–2.9,
  which is *not* by itself disqualifying for a **smoother** — a smoother damps high-frequency error and
  need not approximate `A⁻¹` globally. Level 0's 1e+08 is disqualifying, and level 0 is the fine level.

⚠️ **"0.96× the operator's nonzeros" IS NOT "effectively ILU(0)" — a reading made here and withdrawn.**
Matching the nonzero count says nothing about the values, and those values were 1e+23. A size check is
not a quality check.

⚠️ **AND THE MEASUREMENT LESSON, which is the transferable part: the first survey's success criterion was
"the factorization did not raise".** That is the same family of weak measure this file already warns
about — a one-apply contraction, a preconditioned norm, a cycle count at a benign state — and it passed
an arm whose applied residual is 1e+38. Judge a factorization by what one application does to a real
residual, never by whether it returned.

**What this left.** The native hierarchy was never implicated — PETSc's ILU(0) is a known-good smoother on
these very levels — so the CPU half needed a zero-fill factorization of its own, for which PETSc's source
supplies the whole specification: natural ordering, no pivoting, no dropping, pattern = `A`'s pattern, no
shift. That is now built (`solve/ilu0.py` + the compiled `solve/_ilu0.pyx`), and it is what the section
below measures.

⚠️ **The equilibrate-and-cell-major step is REQUIRED and is now ours.** It was previously implicit in what
the host AMG was handed, so it read as that library's property; it is not, and any smoother put behind
this seam needs it. That survives the change of factorization.

### ✅ THE NATIVE HIERARCHY MARCHES `bfs3d` TO THE SAME ROOT — 10 % FEWER CYCLES, 6 % MORE WALL (2026-08-16)

**The whole-march measurement, which is the only honest one when the preconditioner's SHAPE changes,
and the first time the hand-written hierarchy and factorization have carried a march rather than a
single solve.** Two full 3-rung cold marches differing in **one** environment variable
(`BFS3D_FLOW_INVERSE`), same commit, same machine, back to back, nothing else running:

**⚠️ `hostilu` IS NOW `bfs3d`'s DEFAULT LEADING INVERSE (2026-08-16), so "the incumbent" changed meaning
on that date.** Every measurement in this file that says "the incumbent" of the `bfs3d` **leading** block
without naming an arm was taken against **PETSc ILU(0)** and should be read that way; `BFS3D_FLOW_INVERSE=petsc`
still selects it. The flip was made on the **dependency**, not on the numbers — the table immediately below
is parity, and nothing here claims the host V-cycle is the better preconditioner. It carries the same
coarsening without an optional PETSc build, and the leading block was the last part of this case needing one.
⚠️ **CORRECTED 2026-08-17 — the reason given here was wrong.** This said `hostilu` fails on `pitzDaily`
because "that case needs a fill level `Ilu0` cannot supply". It does not: it needs a different **cell
elimination order**, and `hostilu` marches that case once given one (`PITZ_FLOW_ORDER=rcm`). See
*"Ordering, not fill, is what fails zero-fill on `pitzDaily`"* above. The `pitzDaily` default is still
`petsc` pending the cost comparison, but not for this reason.

⚠️ **THE WALL-CLOCK COLUMN BELOW CANNOT BE ATTRIBUTED, because nothing recorded which `Ilu0` kernel was
live.** `Ilu0` falls back to a pure-Python twin of its compiled kernel when the extension is not built,
and it does so **silently**. The `.so` is gitignored, so it belongs to a checkout rather than a branch
and every fresh worktree starts without one; no worktree on this machine carried it as of 2026-08-17,
and no run log before that date printed `ilu0.COMPILED`. Some `hostilu` runs *were* compiled (the
author confirms at least one), so the point is not that these numbers are wrong — it is that **there is
no way to tell which are which**, which is the unfalsifiable state this file's measurement rule exists
to prevent. **The step and cycle counts are unaffected either way**: the two kernels compute the
identical factorization, pinned by `test_the_compiled_and_reference_paths_agree` (rtol 1e-13) — a test
that is skipped when the extension is absent and had therefore never run on this machine until it was
built. Re-time the wall column before quoting it; the counts stand.

**Fixed structurally rather than noted (2026-08-17):** `tools/build_ext.sh` builds the extension in a
checkout in about a second (it caches one shared build environment under `~/.cache/aquaflux`, so it
does not need Cython in the runtime interpreter — a PEP-668 system Python refuses that);
`validation/run_case.sh` warns at launch when it is missing; and both cases' banners print the live
kernel, so every run from this date carries the answer in its own log.

| | `petsc` (incumbent) | `hostilu` (native AMG + our ILU(0)) |
|---|---|---|
| steps | 59 | 61 |
| **Krylov cycles** | **232** | **208 (−10.3 %)** |
| wall | **1179 s** | 1246 s (+5.7 %) |
| final ‖R‖ | 1.861e-06 | 1.858e-06 |
| preconditioner | 185 s (16 %), probe 131 s | **160 s (13 %), probe 85 s** |
| mid-span / full-span `x_r/h` | 8.361 / 12.53 | **8.361 / 12.53** |

**Same root, identical reattachment to four figures** — so the arm is correct end to end, which is what
this run establishes before anything about cost.

**Read it PER RUNG; the totals hide the structure and invert the conclusion:**

| rung | `petsc` steps / cycles / wall | `hostilu` steps / cycles / wall |
|---|---|---|
| Re/100 (the easy anchor) | 14 / 44 / **246 s** | 14 / **32** / **246 s** |
| Re/10 | 24 / 83 / 406 s | 26 / 87 / **518 s** |
| **target Re (lowest β, hardest)** | 21 / 105 / 527 s | 21 / **89 (−15 %)** / **482 s (−8.5 %)** |

- **Rung 1 is the cleanest controlled comparison available: identical step count AND identical wall
  (246 s both), with 27 % fewer cycles.** So on that operator the native apply is dearer per cycle by
  almost exactly the margin its convergence saves — a wash, measured rather than inferred.
- **On the TARGET rung the native arm wins BOTH axes**, −15 % cycles and −8.5 % wall. That is the low-β
  end where an incomplete factorization's diagonal-dominance windfall is smallest, and it is the rung
  that grows with the mesh — so it is the one that matters for scaling.
- **The entire wall deficit is rung 2**, +112 s, and it is a **trajectory** difference rather than a
  cost one: the native arm took **2 extra outer steps** there. The two marches agree to 3–4 figures in
  β and ‖R‖ through step 10 and part on **α** — `hostilu` takes FULL steps (α = 1.000) where `petsc`
  clips (0.566, 0.803), i.e. it returns the better direction and still ends up needing more steps.
  Nothing here explains that.
- **Fewer cycles buys a cheaper preconditioner too**, which is second-order but real: the refresh
  trigger fires on cycles, so 208 cycles means fewer refreshes, hence 85 s of coloured probe against
  131 s. The probe is identical work per invocation in both arms — the difference is how many times it
  ran.

**⚠️ ONE RUN EACH, AND THIS CASE HAS NO MEASURED MARCH-LEVEL NOISE FLOOR.** Cycle counts are
deterministic and are the load-bearing row; wall clock is not, and +5.7 % on a single pair is not a
number to lean on. The step-count difference (61 against 59) is a trajectory divergence that could go
either way under another bundle.

**Consistency with the two single-state measurements is the reassuring part, and it is not automatic.**
The adjoint put the native arm ~8 % above PETSc in applications while ~14 % cheaper per application (a
wash); β = 0.1 tied outright; and the march now says −10 % cycles / +6 % wall (a wash). Three
independent measurements at three operating points all land on parity. **The `−R` linear probe's 2.75×
win remains the one outlier, and it is the one measurement that is not of the real thing.**

**So: the AMG *can* go native at no cost, and PETSc is no longer load-bearing for the coarsening on
this case.** What it does not do is go native at a *profit*, so nothing here moves `FLOW_INVERSE`; the
case for the direction stays what it was — a GPU where SIMPLE relaxation parallelizes and a sequential
triangular solve does not, which cannot be measured in this CPU-only environment.

### THE NATIVE AGGREGATION BEATS PETSc GAMG ON A LINEAR PROBE AT β = 0 — AND LOSES ON THE ACTUAL ADJOINT (2026-08-16)

**⚠️⚠️ READ THIS FIRST: the win below is a `−R` LINEAR PROBE and it DOES NOT TRANSFER.** The same two arms
at the same state and shift, measured on the real gradient, come out **1696 adjoint applications against
PETSc's 1575 — a 1.08× loss, where the probe predicted a 2.75× win.** The ranking inverts. See *"the count
is a property of the coarsening"* under the `jax.grad` section above for the measurement and the reason
(the adjoint's right-hand side is the cotangent, localized in one field block, not the steady residual).
Everything below is still correct about what it measured; it is simply not an adjoint result, and the
reason it is kept is that it is now this file's cleanest demonstration of the difference.

**This is the measurement the whole host-V-cycle detour existed for, and it had never been taken, because
until now the smoother had never worked.** `solve/ilu0.py` is the zero-fill factorization the section
above specifies — the operator's own pattern, natural order, no pivoting, no dropping, no shift — with a
pure-Python form defining the behaviour and a compiled Cython twin (`_ilu0.pyx`) delivering the speed;
`ilu0.COMPILED` says which is live. Wired behind `_LevelSmoother`, it replaces `spilu` and leaves the
equilibrate-and-cell-major preprocessing in place.

*Configuration:* `bfs3d` `state-00067` (converged, ‖R‖ 3.586e-06), **operator and preconditioner both at
β = 0** — the adjoint's operator — real right-hand side `−R(state)`, **uniform** column reach, GMRES
restart 15 to rtol 1e-8 on the **TRUE** residual, 60-restart cap, field split flow-first with PETSc ILU(0)
on the trailing half in every arm. Native arms: `max_coarse` 500, 5 levels, `strength_threshold` 0.25, no
singleton aggregates, plain aggregation, unsmoothed prolongation. Harness
`validation/bfs3d_openfoam/field_split_probe.py`, arms `split flow/hostilu{1,2,4}`.

| arm | build | cycles | TRUE rel | solve |
|---|---|---|---|---|
| monolithic ILU(0) — *not* the right bar | 3–4 s | 11 | 8.474e-11 | 42 s |
| **`split flow/ilu0` — the matched incumbent** | 3 s | 11 | 6.393e-11 | **29 s** |
| **native hierarchy + our ILU(0) ×1** | 6 s | **4** | **2.449e-13** | **17 s** |
| native hierarchy + our ILU(0) ×2 | 6 s | 6 | 5.939e-11 | 39 s |
| native hierarchy + our ILU(0) ×4 — smoother-matched | 6 s | **4** | 1.498e-12 | 47 s |

- **The aggregation is not what PETSc was contributing.** Smoother-matched at four sweeps the native
  hierarchy takes **4 restart cycles against 11** — 2.75× — on the same operator, the same right-hand
  side and the same trailing inverse. The one thing that differs is the coarsening, and the coarsening is
  ahead.
- **At ONE sweep it also wins on wall clock: 17 s against 29 s, 1.7×**, converging **two to four orders
  deeper** (2.449e-13 against 6.393e-11) while it does so. This is the **first native arm in this campaign
  to beat the incumbent on time rather than only on cycles**, and it does it at the operating point the
  native direction's case has always rested on.
- **Per application the native arm is still the more expensive one** — 4.25 s/cycle against 2.64 — so the
  standing shape of every native-versus-PETSc result here is unchanged. What changed is that the cycle
  advantage finally outruns it.
- **Four sweeps is STRICTLY DOMINATED by one**: identical cycles, 2.8× the solve. Read that with the
  adjoint's own sweep ladder recorded earlier in this file, which found the same flatness on the SIMPLE
  smoother — on this operator, extra smoothing buys nothing the coarse grid has not already bought.

⚠️ **THE LADDER IS NON-MONOTONE AND THAT IS NOT EXPLAINED — 4 / 6 / 4 cycles at 1 / 2 / 4 sweeps.** It is
not run-to-run noise: cycle counts on this case are exactly reproducible (both PETSc controls returned
their recorded values to every digit in **both** runs, 11 / 8.474e-11 and 11 / 6.393e-11), and the ×2 point
comes from the first of those runs at otherwise identical settings. A smoother that is *stronger* landing
on a worse count than one either side of it is the signature of the Krylov space being perturbed rather
than of the smoother being worse, but nothing here establishes that. **Do not quote a single sweep count
from this arm**, and do not assume the ×1 optimum transfers to another operator.

⚠️ **SCOPE — this is β = 0, and β = 0 is the ONLY shift on this state that discriminates.** Re-run at
**β = 0.1**, same state and settings, every arm ties and the ranking carries no information:

| arm at β = 0.1 | build | cycles | TRUE rel | solve |
|---|---|---|---|---|
| monolithic ILU(0) | 3 s | 2 | 2.605e-14 | 14 s |
| **`split flow/ilu0` — incumbent** | 3 s | 2 | 3.103e-15 | **11 s** |
| native + our ILU(0) ×1 | 6 s | 2 | 4.024e-13 | 12 s |
| native + our ILU(0) ×4 | 5 s | **1** | 2.969e-15 | 22 s |

**Two cycles against two, and 12 s against 11 s is inside the ~15 % per-application noise floor** measured
elsewhere in this file (the same preconditioner built twice in one quiet process timed 193 and 221 ms). So
this is the benign-operating-point rule biting on exactly the axis being changed: an easy operator does not
rank preconditioners, and a tie here is *no information*, not evidence of parity. The incumbent reproduced
its recorded 2 cycles / 11 s exactly, so the run is sound — it just cannot answer the question.

**One older claim is refuted in wording and upheld in substance.** "At positive shift no multigrid quality
catches an incomplete factorization" is false as stated — the ×4 native arm takes **1 cycle**, fewer than
any PETSc arm at any shift on this state. What holds is the cost half: it pays 22 s for that cycle against
the incumbent's 11 s for two. Quality catches it; wall clock does not.

**So the march regime is still unmeasured, and a step-initial checkpoint cannot measure it.** This file
already records that all step-initial solves on this case cost ≤ 2 restart cycles while mid-step inner
iterates reach 15 — the hard operators live in the inner iterates and in the *rejected* attempts, which no
checkpoint holds. Ranking these arms for the march needs a captured hard inner iterate
(`BFS3D_INNER_DUMP_ABOVE`) or a whole march, not another β on this state. The recorded 4.6× / 3.9× native
deficits at β = 0.01 / 0.1 were measured against the *SIMPLE*-smoothed traced hierarchy and do not describe
this arm either. **Nothing here licenses moving `FLOW_INVERSE`.**

⚠️ **And a single-state probe is the weaker instrument by this file's own rule**: when the
preconditioner's *shape* changes, only wall clock over a whole march is honest, and this changes shape.
The 17-vs-29 s gap is far outside the ~15 % per-application noise floor; the 4.25-vs-2.64 s/cycle
comparison is much closer to it.

**The refresh cost that motivated writing it is gone, and this is the part that scales.** A zero-fill
factorization's pattern *is* the operator's, so a refresh repeats only the numeric pass. Measured on a
92160-dof, 1.8M-nnz block: first build 0.009 s, **refactor 0.009 s**, one solve 1.7 ms — against
`scipy.spilu`'s **28.1 s** for the same block. The recorded "`spilu` is 88 % of refresh cost and a hard
floor, so the only lever is refreshing less often" was a statement about a *threshold* ILU, whose fill
pattern is value-dependent and therefore cannot be reused; it does not apply to this factorization and
should not be carried to it. ⚠️ The benchmark block is a synthetic at ~20 nnz/row against the real block's
~227, and the elimination's work is superlinear in density, so **do not extrapolate the 0.009 s** — the
real per-level cost is inside the 6 s build above, which is 2× PETSc's 3 s and still trivial beside a
17-second solve.


### ⚠️ THE FORWARD SOLVE'S β = 0 CONTRACTION IS GEOMETRIC, NOT QUADRATIC — still undiagnosed (2026-08-14)

**A root at β = 0 IS reachable — see solve-flow-block.md's *"`jax.grad` RUNS ON THIS CASE"* section, where `rtol` 1e-4 is reached
and the gradient it yields is validated. What has never been explained is the RATE.** An earlier version
of this entry concluded the solve "cannot reach a root at any tolerance a gradient would want"; that is
**false** and is deleted. The rate finding below survives it, and is the part still worth chasing.

*Configuration:* `state-00067` (the converged root), started from its **physical** fields, β = 0
throughout, field split, shipped column reach, `coupled_amg_continuation` built once on concrete
parameters, `forward_rtol` 0.3.

From a converged root the solve decays **geometrically with FULL Newton steps** (`alpha` 1.000 at every
step) where a Newton step at a root should be quadratic. The per-step contraction, measured over 22 steps:

| step | 1 | 5 | 11 | 16 | 19 | 21 | 22 |
|---|---|---|---|---|---|---|---|
| contraction | 0.827 | 0.853 | 0.884 | 0.868 | 0.804 | 0.771 | **0.704** |

**It is roughly flat at 0.83–0.88 for the first ~16 steps and then improves**, reaching 0.70 by step 22
(relative residual 0.019) — mildly superlinear at the tail, **not** the constant rate a first reading took
it for.

**How far it actually gets, measured 2026-08-15 (this supersedes "1e-8 was not demonstrated; how many
steps is unknown"):** the guard in `ImplicitNewtonSolver` passes — i.e. the solve genuinely converged — at

| `rtol` | steps allowed | outcome |
|---|---|---|
| 2e-2 | 30 | reached in **22** steps |
| 1e-3 | 80 | **reached** |
| 1e-4 | 90 | **reached** |

so the reachable tolerance is at least **four orders tighter than the 2e-2 this entry was written at**.
1e-8 is still not demonstrated. ⚠️ The 1e-3 and 1e-4 runs ran with the per-step observer off
(`BFS3D_ADJOINT_SKIP_FORWARD=1`), so the step *counts* at those tolerances are bounded by the budget
rather than measured; the preconditioner-application totals over one gradient evaluation — **1070 → 1702
→ 1944** across the three rows — are the measured proxy.

**Two candidate causes were tested and BOTH are refuted, by controlled arms that share one trajectory:**

| arm | cycles / step | residual trajectory |
|---|---|---|
| `forward_rtol` 0.3 (the march's own), positivity floor 0 | **1** | 8.4766e-06, 7.0091e-06, 5.8031e-06, … |
| `forward_rtol` **1e-6**, positivity floor 0 | **4–8** | **identical to 5 significant figures** |
| `forward_rtol` 1e-6, positivity floor **1e-8** (the case's) | 4–8 | **identical to 5 significant figures** |

- **NOT the linear solve.** Tightening it five orders costs 4–8× the Krylov work per step and moves the
  nonlinear trajectory *not at all*.
- **NOT the k-positivity limiter**, which was the obvious suspect since `coupled_amg_continuation`
  applies `step_limit=positive_k_limit(...)` **unconditionally**. The case's 1e-8 floor reproduces the
  unfloored trajectory exactly.

**So the rate is set by the Jacobian or the residual itself, and that is UNDIAGNOSED.** The residual path
carries no `stop_gradient` (every one in `turbulence/coupled.py` is on the preconditioner or the shift,
which is correct), so a wrong Jacobian is not the cheap explanation. A near-constant rate immune to both
levers is the signature of a fixed-point iteration rather than a Newton one; the next things to look at
are whether the SST closure's blending/limiting makes the residual only piecewise differentiable near
this state, and whether the step actually applied is the one `alpha` reports.

**What it costs, now that the consequence is priced rather than feared.** The rate does not block the
adjoint — it makes a *converged* gradient expensive: reaching `rtol` 1e-4 takes roughly twice the
preconditioner applications of `rtol` 2e-2 (1944 against 1070), and a validated finite-difference check
needs that tight root (at 2e-2 the check reports a 23 % mismatch that is entirely the difference's fault).
So this is a cost and accuracy problem, not a reachability one.

⚠️ **THE ADJOINT-STAGNATION ENTRY THAT USED TO SIT HERE IS DELETED — every claim in it is now false.**
It said `jax.grad` had never succeeded, that the transpose solve stagnated even at a loose root, that both
halves of the adjoint were broken, and that two arms (the Krylov settings, and the native leading inverse)
remained unrun. The stagnation was a **solver-settings artifact reachable only through an API gap that is
now closed** (`solve_coupled(adjoint_solver=…)`), the gradient runs and is validated to 1.9e-04, and both
arms have been run. See solve-flow-block.md's *"`jax.grad` RUNS ON THIS CASE, AND IS VALIDATED"* section for the costs and the
native-versus-PETSc comparison. Three findings from that entry are **not** superseded and are kept here
because they cost real time to learn:

- **❌ The column reach is not a factor in the transpose solve** — measured, not assumed. At **uniform**
  reach (`BFS3D_COLUMN_REACH=0`) the forward objective came back bit-identical to the shipped reach's
  (5.337542799e+04), which is the expected control: the reach is a preconditioner-only approximation and
  cannot move the root.
- **⚠️ `on_step` CANNOT BE USED UNDER `jax.grad`, and it is the observer that makes a march readable.**
  `solve_coupled` raises rather than letting it through: `refresh.trigger` / `step_control` / `on_step` /
  `on_checkpoint` / `precondition_step` / `retry` all drive a forward-only **eager** march that
  steps in Python on concrete residual norms, which a differentiation tracer cannot flow through. The
  adjoint is refresh-independent, so the single-stage solve returns the identical gradient; the cost is
  only that **a differentiated evaluation is silent**. The probe therefore builds the objective twice,
  watched and unwatched, and differentiates the unwatched one — and the transpose solve's own progress is
  reported by counting preconditioner applications instead (see the instrumentation note above).
- **⚠️ Two probe defects, either of which produces a confident wrong number**, and neither announces
  itself:
  - **`layout.unpack` is NOT the physical state.** The checkpoint stores the *solved* variables, whose
    scalar blocks are `log(omega)` under this case's log-variable transport, while `solve_coupled` takes a
    **physical** initial condition and maps it in itself. Feeding it the unpacked blocks applies the log
    twice: the solve starts at |R| ~1.4 instead of ~1e-05, converges to *something*, and returns a finite
    gradient of the wrong problem. Use `coupled.physical_fields(state)`.
  - **A probe that builds its own continuation silently gets LIBRARY defaults**, not the case's — here the
    positivity floor (case 1e-8, library 0). It made no difference, but only because it was checked.

  Both were caught only because the probe prints one line per outer step. It did not, at first.

### The gap is CONVERGENCE, not cost — measured, and it reverses the priorities

**The per-iteration split had never been measured, and both directions of inference about it were
wrong.** `BFS3D_PROBE_SPLIT=1` times the exact matrix-free Jacobian product against the preconditioner
apply, separately, on every arm. At β = 0.1:

| arm | jacobian product | preconditioner | cycles |
|---|---|---|---|
| `split flow/ilu0` (incumbent) | 34 ms | **121 ms** | 2 |
| native, 4 sweeps x 2 inner | 38 ms | **175 ms** | 8 |
| native, block splitting | 49 ms | 212 ms | 7 |
| native, 8 sweeps | 46 ms | 319 ms | 5 |

**The native preconditioner is only ~1.45x more expensive per application and needs 4x the cycles.** So
the wall-clock gap is convergence, and **a lever that buys cycles at no cost is worth about four times an
equally-sized cost reduction.** Two independent audits had inferred the Jacobian product to be roughly
half of an iteration and concluded preconditioner work was capped at half the gap; it is **14 %**. That
inference chained three estimates, one resting on a wrong nonzero count, where the direct measurement
takes seconds — take it before ranking any cost work.

⚠️ **TWO QUALIFICATIONS ON THE 1.45x, added 2026-08-14; the CONCLUSION survives both.**
- **It is a FOUR-SWEEP number, and the case ships TWO.** Re-measured at the shipped bundle (`omega10`:
  2 sweeps x 2 inner, strength 0.25, no singletons, 5 levels, `max_coarse` 500, block splitting,
  `omega` 1.0), the native apply came out 106 ms against the incumbent's 118 ms in one run and 128 ms
  against 112 ms in a second — so at two sweeps **the two are indistinguishable per application and the
  ordering is not callable.** Do not quote 1.45x for the shipped configuration.
- **Every number in the table above carries the ~15 % instrument spread** measured under *The noise floor*
  earlier in this file (the same preconditioner built twice in one quiet process timed 193 and 221 ms), so
  the four rows rank reliably only where they differ by much more than that — which 121 against 319 does
  and 121 against 175 does not.

**The conclusion is unaffected and is if anything stronger**: the cost half of the trade is small or
absent, the cycle half is 4x, so a lever that buys cycles still dominates one that buys arithmetic.

**TWO FREE QUALITY LEVERS, AND THEY COMPOSE.**

`omega` — the relaxation on the whole SIMPLE correction — was 0.7, and **no arm in this campaign had ever
varied it; there was no token.** It stacks on the Frobenius per-row relaxation the velocity predictor
already carries (median 0.53) for an effective ~0.37. This is the **third** time the same over-damping has
been found here, after the pressure relaxation (worth 1.6x) and the transported scalars' smoother
(10 cycles against 2). Undamping is worth one cycle at *both* sweep counts, so it is not a single point.

| configuration (β = 0.1, 4 sweeps x 2 inner) | cycles | solve |
|---|---|---|
| baseline, omega 0.7, scalar diagonal | 8 | 34 s |
| + per-cell block splitting | 7 | 32 s |
| + omega 1.0 | 7 | 31 s |
| **+ both** | **6** | **29 s** |

Each is worth one cycle alone and together they are worth two, so they correct **different** deficiencies
rather than being two routes to the same one. Both are properties of the **velocity splitting** — which is
where the balance measurement pointed (`||S|| = 1.449` against `||E|| = 1.029`) before that reading was
mistakenly talked down.

**AT β = 0 THE NATIVE PRECONDITIONER NOW CONVERGES IN FEWER ITERATIONS THAN THE INCUMBENT.** The
composition holds at zero shift, so it is not a march-only effect:

| arm at β = 0 | cycles | solve |
|---|---|---|
| `split flow/ilu0` | 11 | 31 s |
| native, 8 sweeps, omega 0.7, diagonal | 11 | 76 s |
| **native, 8 sweeps, block + undamped** | **7** | **57 s** |

**1.84x at the adjoint point** (from 2.45x), and **2.4x on the march** (from 3.0x). The asymmetry is the
structural claim with the native side finally measured at its best: at zero shift the native hierarchy is
the *better* preconditioner and loses only on arithmetic; once a shift is present an incomplete-LU is
nearly exact in two triangular passes and no multigrid quality catches it.

⚠️ **SIMPLEC FAILS HERE (true relative residual 1.0 at both relaxations) BUT THE RECORDED MECHANISM WAS
WRONG — and the measurement that would settle it has not been run.** Its velocity coefficient is
`1/(a_P - sum a_nb)`, which is this matrix's **row sum**. The first reading was that a conservative
discretization makes the interior row sum vanish — convection cancels by continuity, diffusion telescopes
by conservation — so the coefficient degenerates and that is why the arm failed. **That reading does not
survive its own numbers.**

The guard fell back to the Frobenius diagonal when a row sum dropped below **0.1** of its own diagonal,
and the arm ran at **beta = 0.1**. For an interior cell the row sum is essentially the shift `beta d` with
`d ~ a_P ~ diagonal`, so `row sum / diagonal ~ beta = 0.1` — sitting exactly on the cutoff. The reported
"46 % of fine rows fell back" therefore measures rows scattered either side of a threshold set, by
coincidence, at the physical value. It is not evidence of a degenerate operator.

On the rows where SIMPLEC did apply the coefficient was about `1/(beta d) = 10/d` against Jacobi's `1/d`.
**That is a legitimate SIMPLEC coefficient for this shift, not a pathology** — a larger, consistent `d` is
the method's entire point, and it is why SIMPLEC needs no pressure under-relaxation as a solver.

**The likelier cause is specific to SMOOTHING rather than solving.** In a segregated solver the larger `d`
is bounded by the outer iteration and the velocity under-relaxation. As a fixed-sweep level smoother
there is no outer control, and a velocity coefficient ten times larger pushes the error operator
`I - omega M A` past unit spectral radius — the same failure shape as an undamped **Jacobi** pressure
sweep (3.435e-01), one block over. Note this is consistent with SIMPLEC converging perfectly well as a
solver without relaxation; the two claims do not conflict.

**To settle it, measure `||I - F~^-1 F||` under the SIMPLEC diagonal** (`_splitting_balance` already
computes this quantity for the other splittings — but ⚠️ it currently has no caller, so wiring it back to a
switch comes first) with the fallback threshold dropped far below `beta` so
SIMPLEC is genuinely in force. Above the Frobenius diagonal's 1.449 means it amplifies and the direction
closes for that reason; comparable or lower means something else killed the arm and it deserves another
look. **Until that is run, treat SIMPLEC as failed-but-unexplained, not as closed.**

**SIMPLER is a separate no, on cost**: it targets the smooth global pressure mode, which is what the
coarse grid already owns, at roughly double the pressure work already measured as not paying.

### Q&A on the remaining levers — three closed, one small win

Four questions were put to measurement together, because each earlier answer had been taken with another
axis pinned. All at β = 0.1 on `state-00067`, strength 0.25, 5 levels, no singletons, 60-restart cap;
matched incumbent `split flow/ilu0` is **2 cycles / 11 s**.

**A STRONGER SPLITTING IS A SMALL, FREE WIN — the only Pareto improvement found.** Replacing the velocity
predictor's scalar diagonal with a per-cell block form saves a cycle at both sweep counts and does not cost
wall clock, despite tripling the nonzeros in `dg` and hence in the Schur:

| sweeps | splitting | cycles | solve |
|---|---|---|---|
| 4 | scalar diagonal | 8 | 34 s |
| 4 | **cell block** | **7** | **33 s** |
| 8 | scalar diagonal | 6 | 43 s |
| 8 | **cell block** | **5** | 40 s |

Five cycles is the fewest any native arm has reached at this shift. Everything else measured here trades
cost against cycles; this is the one change that improved the smoother's *quality* at fixed price.

⚠️ **The block form MUST be the Frobenius-optimal one, and the exact inverse of the cell's own block is
much WORSE than the scalar diagonal it would replace.** Measured `||I - F~^-1 F||` per level: scalar
Frobenius diagonal 1.449 / 1.230 / 1.482 / 1.275, against the cell-block **exact inverse** 12.03 / 10.53 /
8.64 / 5.76 — four to eight times worse. Inverting a cell's own block ignores the row's dominant
off-diagonal mass, which on a convection-dominated operator is most of it. This is the same trap as plain
Jacobi versus Eq. (39) on the diagonal, one block size up.

⚠️ **The Frobenius block is `M_i = F_ii^T (R_i R_i^T)^-1`, and getting the transpose wrong is silent at one
field per cell.** Minimizing `||I - M F||_F` over block-diagonal `M` decouples by cell, with `R_i` the
cell's row block. Computing `F_ii (R R^T)^-1` instead — the same expression without the transpose — is
correct only when the cell's own block is symmetric, which a velocity block is not. Measured on a
nonsymmetric test: correct 1.048, transposed 1.246, plain exact inverse 1.101, so the error makes the
"optimal" form worse than simply inverting the block. In the solver it took the arm from **7 cycles to 58**
(no convergence). It reduces exactly to the scalar `F_ii / ||F_i||^2` at one field per cell, so a scalar
reduction check cannot catch it — check against a brute-force minimizer on a nonsymmetric case instead.

**⚠️ AGGRESSIVE COARSENING x W-CYCLE x DEPTH — CLOSED, and the earlier one-axis results were RIGHT.** The
suspicion was that each had been measured with the other pinned at the value that makes it fail, and that
a W-cycle needs the genuinely cheap deep levels aggressive coarsening produces (the configuration the
literature runs at eight to fifteen levels). Tested jointly, it does not hold:

| arm | cycles | solve |
|---|---|---|
| **plain, V-cycle, 5 levels** | 8 | **34 s** |
| aggressive, V-cycle, deep | 9 | 36 s |
| aggressive, W-cycle, 5 levels | 8 | 47 s |
| aggressive, W-cycle, deep | 8 | 60 s |
| plain, W-cycle, deep | 6 | 54 s |

The W-cycle reliably buys ~25 % of the cycles and never once pays for it, across five configurations — the
2^k visits to level k outrun any cheapness depth and aggressive coarsening can create. And aggressive
coarsening makes the W-cycle **worse** on cycles (8 against 6), the opposite of the hypothesis. Note this
is the one place where suspecting a premature closure was itself the error.

**So the marginal-cost optimum is REAL, not an artifact of exploring one axis at a time.** The coarse-grid
side survived a genuine joint test; the smoother side yielded one small Pareto gain that shifts the
optimum rather than contradicting it. Best native arm is now **7 cycles / 33 s** against the matched
incumbent's **2 / 11 s** — a threefold wall-clock gap that is structural for this smoother family.

### The sparse matvec is at the limit of what JAX's primitives give — block-CSR REFUTED

The level operators are applied through `_CsrOperator.apply`, a `jax.experimental.sparse.BCSR` product at
**scalar** block size. Since the flow block carries four fields per cell it has genuine 4x4 dense block
structure, and a true block form stores one column index per **sixteen** nonzeros rather than one per
nonzero — cutting index traffic from 4 bytes per nonzero to 0.25, about 31 % of total traffic on a kernel
whose fine level is bandwidth-limited. Measured on a synthetic operator at the flow block's exact shape
(92160 rows, 11.8M nonzeros, 128 per row), the saving does not materialize:

| variant | time | Gnnz/s |
|---|---|---|
| scalar CSR (what is shipped) | 5.20 ms | 2.27 |
| block 4x4, state already cell-major | **5.20 ms** | 2.27 |
| block 4x4 including the field-major transposes | 6.58 ms | 1.79 |

**Even granting the layout migration for free, the block form exactly TIES the scalar one**, because the
traffic it saves is spent on intermediates a fused kernel never writes: a per-edge gather buffer and a
per-edge product buffer. The transposes that the field-major layout `(cell i, field f) = f n + i` would
actually require cost a further 27 %. So a block-CSR migration is worth zero at best and -27 % as it would
have to be built, against a state-layout change across the whole solver.

**The corollary is the useful part: JAX's CSR matvec is not index-bound.** It matches a formulation
carrying sixteen times less index traffic, so it is near what the primitive can do and there is no easy
headroom in the kernel. Beating it needs a **fused** custom kernel doing gather → block multiply →
scatter without materializing between the steps, which is a Pallas or custom-call project — and one whose
payoff is on an accelerator, where that access pattern is the bottleneck, not on CPU where it would chase
a fraction of a threefold gap.

⚠️ **Per-call launch overhead is NOT why the coarse levels look expensive** — measured at about **10 us**
per matvec, against roughly 144 matvec calls in a V-cycle of ~400 ms, i.e. under 1 %. Nor are the coarse
levels especially inefficient where it matters: level 1 runs at 1.38 Gnnz/s against the fine level's 1.43
and simply holds 27 % of the nonzeros. Only levels 2-3 degrade (0.45-0.75 Gnnz/s) and they hold ~7 % of
the nonzeros, so their excess costs ~5 % of the cycle. **The fine level is ~70 % of the work and is where
any real saving has to come from.**

⚠️ **A kernel timing taken on RANDOM column indices understates the shipped matvec by ~2.2x** (1.43 against
3.13 Gnnz/s at the fine level's shape) purely through locality — real mesh-ordered matrices are far more
cache-friendly. Any cost model built on synthetic sparsity therefore overweights matvec work relative to
everything else, which is exactly why one such model predicted 16 % for removing a multiply-by-zero and it
delivered 2-7 %.

### The SIMPLE smoother is BALANCED and both halves are poor — measured, not argued

Siefert and de Sturler (2006) analyse precisely this operator class: a generalized saddle point whose
(1,2) block differs from the transposed (2,1) block and whose (2,2) block is nonzero but small in norm —
which is what Rhie–Chow interpolation produces, and which they note has received little attention and no
numerical experiments. They show the preconditioned eigenvalues cluster to within a constant times
`max(||S||, ||E||)`, where `S` is the error of the splitting used for the velocity block and `E` the error
of the approximate Schur inverse, and conclude that the two must be **balanced**: shrinking whichever is
already smaller cannot move the bound, so that effort is wasted.

Both have closed forms for this smoother rather than needing an estimate. The splitting is `F ~ diag`, so
`S = I - F_diag^-1 F`. The Schur inverse is `n` damped-Jacobi sweeps, whose recurrence telescopes to
`M_S S = I - G^n` with `G = I - omega D_S^-1 S`, so `E = -G^n` exactly. `_splitting_balance` in the probe
takes both as largest singular values by sparse iteration.

⚠️ **THE DIAGNOSTIC IS PRESENT BUT UNREACHABLE — it has NO caller, and there is no `BFS3D_PROBE_BALANCE`
switch (checked 2026-08-14; the record claimed one).** So the numbers below cannot currently be re-taken,
and neither can the SIMPLEC question further up that names this function as the way to settle it. Wiring
it back to a switch is a few lines and is the prerequisite for either. It takes the level's Schur in host
sparse form as an explicit argument, because the smoother's own pieces no longer carry a host copy — that
would stop them riding into the compiled cycle as a jit argument.

Measured at β = 0.1 on `state-00067`, strength 0.25, 4 outer x 2 inner sweeps, 5 levels, no singletons:

| level | `||S||` splitting | `||E||` Schur inverse | ratio |
|---|---|---|---|
| 92160 (fine) | 1.449 | 1.029 | 1.41 |
| 26828 | 1.230 | 1.113 | 1.10 |
| 6540 | 1.482 | 1.373 | 1.08 |
| 1448 | 1.275 | 1.238 | 1.03 |

**The DEEP levels are balanced (ratios 1.03–1.10); the FINE level is not (1.41), and it carries ~70 % of
the cost.** Read the two separately — a single "they are balanced" reading of this table is wrong where it
matters most. By the `max(||S||, ||E||)` criterion, bringing the fine level's splitting error down to its
Schur error would cut the bound 1.449 -> 1.113, about **23 %**; beyond that, further gains need both halves
together. So a one-sided improvement of the splitting is worth something, but bounded, and only on the
fine level.

It does explain why dropping the inner pressure sweeps 4 -> 2 was nearly free: the Schur side was already
no worse than the splitting side, so the extra sweeps had nothing to buy.

**Both norms are ~1.0–1.5, which is bad in absolute terms and is the real finding.** Their favourable
regime is `||S||, ||E|| << 1`, where the eigenvalues cluster tightly; at `max ~ 1.4` the bound is vacuous,
which matches this arm needing 8 cycles where the incumbent needs 2. Concretely `||I - F_diag^-1 F|| =
1.449` says the diagonal splitting of the velocity block is a ~100 % error, and `||G^2|| = 1.03` says two
damped-Jacobi sweeps on the Schur barely contract in the worst direction. (A 2-norm above 1 does not mean
divergence for a non-normal iteration — the spectral radius governs asymptotically — but it does mean very
little progress.)

**The consequence is structural, not tunable: a SIMPLE smoother with a DIAGONAL splitting is near its
ceiling here.** A real gain needs both halves improved together, and improving the velocity splitting past
a diagonal means solving with `F`, which is the cost this whole direction exists to avoid.

⚠️ **Cao, Du and Niu (2014) shift-splitting does NOT apply here** — there is no such lever to reach for.
Their construction assumes a **symmetric positive definite** (1,1) block, a **zero** (2,2) block, and
off-diagonal blocks that are exact transposes; the convergence proofs use all three (Lemma 2.2 needs
`||(alpha I + A)^-1 (alpha I - A)|| < 1`, Theorem 3.1 forms `A^{1/2}`). Our velocity block is
convection-dominated and nonsymmetric, our (2,2) block is the Rhie–Chow damping, and Rhie–Chow also breaks
the transpose relation. Their central device — adding `alpha I` to a zero (2,2) block to regularize it —
is something this discretization already provides, and augmenting that block further is separately
recorded as a triple-confirmed no-go that degraded cycles 2.7x. Their experiments are a Stokes problem,
i.e. no convection at all.

**What DID transfer: `8 x 2` is the right sweep pair at every β** (49 s against 56 s at 0.05, 43 s against
48 s at 0.1), so that tuning is not an artifact of zero shift.

⚠️ Still measured under a **uniform** column reach the case does not use; at the shipped reach the
incumbent's cost doubles at β = 0 (22 cycles against 11) and the native arm is unmeasured there.

**Why the gap may still be the right trade.** The incomplete-LU is a sequential triangular solve.
This smoother is sparse matrix-vector products, diagonal scalings and one small dense coarse solve, and
its fine level was measured memory-bandwidth-bound at 37–40 GB/s — the profile of something that
vectorizes. Kernel fusion and JAX dispatch overhead were measured out as levers (a chain of 8 matvecs in
one jit costs exactly 8x one matvec); the only thing that helps on CPU is touching fewer nonzeros.

Cell-local smoothing on the saddle is separately closed (point Jacobi, Chebyshev, cell-block Jacobi and
Vanka all fail), the flat-inverse family is closed, smoothed aggregation is closed, and orthonormalized
prolongation is closed.

⚠️ **lAIR is not the way in here**, and two of the three reasons are already measured on this mesh.
`build_air_hierarchy` is **scalar** — no `block_size` — and its classical C/F splitting picks individual
degrees of freedom, destroying the four-field block structure the SIMPLE smoother needs on every coarse
level. It calls `_require_positive_diagonal` at every level, and the saddle's pressure rows are the
Rhie–Chow damping. And on the true Jacobian slice of the `[k, omega]` block on this same mesh it ran
~50 minutes without finishing at 91 nonzeros per row; the flow block carries 128.

**Transferring the threshold to the `[k, omega]` block will not buy CYCLES**, and the trailing ablation
settled that on the right measure: at `state-00067`, β = 0, with the leading inverse held at ILU(0), the
shipped native trailing inverse and PETSc ILU(0) both give **11** coupled cycles, and a near-exact
factorization of that block is **worse than either** (58 cap). Nothing in that ablation supports spending
on the trailing block's accuracy. ⚠️ **The "2 restart cycles against an absolute floor of 1" figure this
entry used to rest on is a STANDALONE solve of the block, which production never performs** — one Krylov
iteration runs on all six fields with a block-triangular `M`, so the count is a property of the whole
preconditioned operator. Do not cite it.

**The live reason to sweep it anyway is SCALABILITY, not cycles.** At 2 levels the coarse grid grows
**linearly** with the mesh while its dense coarse solve is **cubic** to build and is rebuilt on every
refresh: ~438 dofs at 23040 cells, ~4400 at ten times that, and `_MAX_DENSE_COARSE_DOFS = 8192` is where
it raises outright.

✅ **The old blocker is gone: the trailing path is no longer capped at 2 levels.** It ran on
`build_convection_hierarchy`'s `max_levels` default (`_CONVECTION_LEVELS` = 2) because the class exposed
neither `strength_threshold` nor `max_levels`; the shared `NativeHierarchyInverse` base exposes the whole
coarsening surface (see the shared-base entry above). Two conditions on any sweep that uses it:
- **Sweep `max_levels` and `strength_threshold` TOGETHER.** A threshold coarsens LESS, so it enlarges the
  coarse grid; testing one at 2 levels reproduces the flow block's original failure and would refute the
  transfer for the wrong reason.
- ⚠️ **Normalize `_cell_graph`'s edge weights per row-field first.** It weights each cell edge by the SUM
  of `|A_ij|` over the block, and this block's ω rows sit ~8 orders above its k rows — so a raw threshold
  there is an **ω-only** strength measure. This is the same collapse-over-row-fields failure that let a
  column-reach audit approve a configuration which then diverged the march, and that made the trailing
  bound arm unreadable for a day.

⚠️ A nonzero threshold there still forfeits the refresh cache-hit unless paired with `frozen_coarsening`
or `shape_headroom`; the binding decision keeping the k/ω path at θ = 0 in a *march* is unchanged, and the
knob is safe for a single-state sweep precisely because a sweep never refreshes.

**Defects found here, and what was done about each.**
- **FIXED — `_reattach_to_adjacent_root` could CREATE singletons.** It never steals a root but does
  steal every member, so an aggregate could be cut down to its root alone *after* the aggregation's own
  repair had run and could no longer see it. Two passes are needed and both now run under
  `avoid_singletons`: the sweep refuses to open a singleton, and `_absorb_singleton_aggregates`
  dissolves whatever reattachment strands, relabelling contiguously. Measured on a 24x24 anisotropic
  Poisson at threshold 0.25 with one aggressive level: **15 and 11 singletons on the two levels
  unrepaired, 3 and 0 with the sweep repair alone, 0 and 0 with both** — and it coarsens further while
  doing it (260 -> 208 aggregates). Pinned by
  `test_avoid_singletons_reaches_the_aggressively_coarsened_level`.
- **FIXED (documentation) — `max_coarse` is in DEGREES OF FREEDOM, not cells.** It is compared against
  `a.shape[0]`, while both builder docstrings said *cells* — a `block_size`-fold error on the knob
  bounding a dense inverse. Dofs is the correct unit and the code was right: the limit exists to
  bound a solve whose cost is cubic in the dof count. The docstrings now say so.
- **FIXED (guarded, not defaulted) — the coarse grid grows LINEARLY with the mesh when the level cap
  binds before the size limit.** At `max_levels = 5` and a measured 184x total ratio, 23040 cells give
  500 coarse dofs and 1M cells would give ~21500 (~3.7 GB, dense, cubic to build). The level cap is a
  deliberate shipped default on the turbulence path (`_CONVECTION_LEVELS = 2`), so it was NOT changed;
  instead `_MAX_DENSE_COARSE_DOFS` (8192, ~512 MB) makes the case fail with both ways out named, rather
  than silently allocating. Nothing measured here approaches it — the largest coarse level in any arm
  is 4300 dofs. Pinned by `test_dense_coarse_solve_guard_rejects_an_oversized_coarsest_level`.
- **FIXED — `_AGGREGATE_STATS` accumulated across builds** despite its docstring saying to clear it, so
  a build that stopped early shifted every later arm's window in the consumer's tail slice. Cleared at
  the start of each build; the consumer may now read the whole list.

⚠️ **Stale by this section:** the recorded refutation that *"equilibration does not reach the coarsening"*
(0.03 % of edges) was measured at `strength_threshold = 0`, where only the sparsity **pattern** matters.
With a threshold live the aggregation reads **weights**, and these arms run `equilibrate=False` on a block
whose momentum, gradient, divergence and Rhie–Chow entries sit at raw scales. That refutation does not
transfer to any thresholded arm.

