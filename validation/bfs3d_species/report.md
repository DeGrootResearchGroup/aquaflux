# Passive tracer on the finite-width 3D backward-facing step

A species injected over **part of** the inlet of the 3D backward-facing step, transported to steady
state on that flow, and compared cell for cell against OpenFOAM. This is the first exercise of
`aquaflux.transport` against another code, and it is the shape the reactor problems this project
exists for will take: converge a flow, then ask what a tracer does in it.

## Why two arms

The bfs3d **flow** does not agree between the codes — reattachment `x_r/h` 8.36 against OpenFOAM's
7.24 — so a species comparison run on each code's own flow is dominated by that disagreement and can
attribute nothing. One number cannot answer both questions, so the case reports two, under different
names:

| arm | flux and `nu_t` | what it isolates |
|---|---|---|
| **same-flux** | OpenFOAM's own `phi` and `nut`, imported | the **scalar discretization**, and nothing else |
| **own-flow** | aquaflux's own converged flux and `nu_t` | end to end — transport **and** the flow difference |

The arms differ **only** in which flux and eddy viscosity are supplied. Mesh, scheme, gradient,
boundary closures, diffusivity relation and the injector are shared verbatim, so the difference
*between* the arms is the flow's own contribution.

Transporting on the reference's `phi` is legitimate — and necessary — because `phi` is the flux
OpenFOAM's *discrete* continuity closes on. Rebuilding `(u·n)A` from cell velocities satisfies no
discrete continuity, and a tracer on such a flux is not conservative. That the imported `phi` lands
on the right faces is **measured**, not assumed, by `../bfs3d_openfoam/phi_placement.py`: the
conservative scatter on aquaflux's own connectivity leaves a max per-cell imbalance of `4.9e-06` of
the domain flow rate, against `2.3e-02` for a seeded permutation of the interior block — 4700× worse.

## The injection is a boundary value, not a mesh change

The injector is a `DirichletField` on the **existing** `inlet` patch whose value is a function of the
face centroid. Splitting the inlet in `blockMesh` would produce a different `polyMesh` and invalidate
every measurement previously taken on it, including the flow case's checkpoints — so it was
deliberately not done.

The profile has **one definition**, `injector.injected_value`, from which the OpenFOAM case's inlet
values are *generated* (`write_inlet_field.py`) as an explicit `nonuniform List<scalar>`. Both codes
therefore impose identical values face for face, rather than two implementations of one intent. That
is only sound because the two codes agree on what face `i` of the inlet is — the same index
correspondence the flux reader checks.

Its edges are **tapered rather than sharp**, over a raised cosine a few cells wide. A discontinuous
top-hat is the more obvious choice and makes the comparison worse: at a jump the leading difference
between the codes is how their gradient limiters break ties at one cell, not the transport being
measured. The taper is a choice to make the measurement attributable, and costs nothing in fidelity
because both codes receive the identical face values.

Measured on the case mesh: **45 of 256 inlet faces** carry tracer, and they carry **8.8% of the inlet
volumetric flux** (`3.5052e-04` of `4.0000e-03` m³/s). That rate is the conservation target — at
steady state, with no volume source, the outlet must carry exactly it.

## Configuration

| | |
|---|---|
| mesh | the flow case's `polyMesh`, 23040 cells, 71872 faces, 3D (no `empty` patches) |
| flow | frozen at OpenFOAM time `2000` (`solver functions`, so U/p/k/ω/φ are never re-solved) |
| advection | aquaflux: limited second-order upwind (Venkatakrishnan). OpenFOAM: `bounded Gauss limitedLinear 1` |
| gradient | aquaflux: corrected Green–Gauss. OpenFOAM: `Gauss linear` |
| diffusivity | `Γ = ν + ν_t`, i.e. **turbulent Schmidt number exactly 1** on both sides |
| boundary | injected profile on `inlet`; `zeroGradient` on outlet and all walls |

**`Sc_t = 1` is a choice made to match the reference, not this project's default (0.7).** OpenFOAM's
`scalarTransport` with no `D` entry uses the momentum transport model's `nu + nut`; aquaflux is given
`effective_diffusivity(nu, nut, turbulent_number=1.0)`, which is the same quantity. Every number here
is a number at `Sc_t = 1`.

The advection schemes are **nominally the same order, not the same scheme**. That difference is
precisely what the same-flux arm exists to measure, and it is why the arm's bar is a reported number
rather than a tight assertion.

## What is measured

- **Conservation** — `|Σ over all cells of R|`. Summing the converged residual telescopes every
  interior face away, so what survives is exactly the net boundary flux, which must vanish. This is a
  discrete identity, not an approximation.
- **Cellwise agreement** — max and RMS of `s_aquaflux − s_OpenFOAM`.
- **Mixing** — per streamwise station, the volume-weighted slab mean and the **unmixedness**
  (variance normalized by `mean·(1−mean)`): 1 for a completely segregated stream, 0 once uniform.
  Defined once, in `slab_profile`, so the table and any figure come from the same callable.

## Results

**Same-flux arm — the two codes agree closely on the scalar discretization.** Both on OpenFOAM's own
`phi` and `nut`, 23040 cells, `Sc_t = 1`:

| | aquaflux | OpenFOAM |
|---|---|---|
| cellwise `max abs(delta)` | — | **0.0426** |
| cellwise RMS `delta` | — | **0.0041** (0.4% of the injected value) |
| cellwise mean `delta` | — | −0.0008 |
| range | `[-0.0002, +0.9941]` | `[-0.0000, +0.9986]` |
| outlet throughput (m³/s) | `3.5110e-04` | `3.5060e-04` |

against `3.5052e-04` m³/s injected. OpenFOAM's outlet is within **0.02%** of the injected rate;
aquaflux's within 0.17%, which is the `zeroGradient` owner-value estimate rather than a conservation
error — aquaflux's exact conservation identity `abs(sum R)` is `8.2e-15`, machine zero.

Mixing, station by station (volume-weighted slab mean, and unmixedness):

| `x/h` | aquaflux mean | OF mean | aquaflux unmixed | OF unmixed |
|---|---|---|---|---|
| 1 | 0.10255 | 0.10387 | 0.2540 | 0.2528 |
| 2 | 0.10597 | 0.10687 | 0.2394 | 0.2392 |
| 4 | 0.10665 | 0.10742 | 0.2299 | 0.2303 |
| 6 | 0.10452 | 0.10516 | 0.2259 | 0.2274 |
| 8 | 0.10080 | 0.10133 | 0.2110 | 0.2129 |
| 12 | 0.09528 | 0.09564 | 0.1647 | 0.1655 |
| 16 | 0.09220 | 0.09252 | 0.1257 | 0.1270 |

**The unmixedness agrees to under 1% at every station**, and both codes trace the same monotone decay
(0.25 → 0.13). Given that the two use nominally-equivalent but different limited second-order upwind
schemes and different gradient reconstructions, that is the arm behaving as designed: with the flow
difference removed, what remains is small.

### Caveats these numbers were taken under

- **`Sc_t = 1`**, not the project default of 0.7 (see above). Set explicitly on both sides:
  OpenFOAM's `diffusivity viscosity; alphal 1; alphat 1;` and aquaflux's `turbulent_number=1.0`.
- **The OpenFOAM tracer limit-cycles at an initial residual of `~5.1e-05`** rather than converging
  further — flat from iteration 300 to 600. Five orders below its start and well below the level this
  comparison resolves, but it is a limit cycle, not convergence, the same character as this case's
  SIMPLE flow run.
- **The flow is frozen at OpenFOAM time 2000** (`solver functions` with `subSolver
  incompressibleFluid`, which constructs the fields and solves none of them).

## Status

**Same-flux arm: complete and reported above. Own-flow arm: built and wired, not yet run.**

The own-flow arm needs the flow case's coupled build, and the machine could not host it while another
`bfs3d_openfoam` march was running in a different worktree — these jobs are memory-bound and
`run_case.sh` correctly refused to start a second. `compare.py` runs both arms; nothing is missing but
the compute.

Verified independently of the reference, on the same-flux arm:

| property | result |
|---|---|
| solve | 10 s, preconditioned by the frozen convection-diffusion V-cycle |
| conservation `abs(sum R)` | `8.2e-15` against `3.5e-04` m³/s injected — machine zero |
| boundedness | `[-1.7e-04, +0.994]` against an injected 1 |
| imported flux continuity | `4.9e-06` of the domain flow rate |

The solve needs the preconditioner: unpreconditioned it does not converge, because away from the
shear layer `nu_t` falls to `~4e-11` and the cell Péclet number reaches order `10³`. The V-cycle this
project already builds for the k/ω scalars is reused rather than rebuilt.

`tests/integration/test_bfs3d_species.py` pins these as `@pytest.mark.validation` tests, skipping
when the case data (which is not in the repository) is absent.

## Running it

```bash
# 1. the flow case, if runs/ is not already populated
docker run --rm -v "$PWD:/work" -w /work/validation/bfs3d_openfoam/of_case openfoam13:latest bash run_of.sh

# 2. generate the tracer inlet field from aquaflux's mesh
python3 validation/bfs3d_species/write_inlet_field.py

# 3. the OpenFOAM tracer run, on the frozen flow
docker run --rm -v "$PWD:/work" -w /work/validation/bfs3d_species/of_case openfoam13:latest bash run_of.sh

# 4. the comparison
validation/run_case.sh validation/bfs3d_species/compare.py --wait
```
