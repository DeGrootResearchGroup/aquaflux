"""pitzDaily backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST.

A **same-mesh, cell-for-cell** cross-code validation. The OpenFOAM pitzDailySteady tutorial (its RAS
model switched from the shipped kEpsilon to kOmegaSST) is run in the openfoam13 container by
``of_case/run_of.sh``, which writes the converged fields, the mesh, and the SIMPLE residual history to
``runs/kwsst/``. This script then reads that **same mesh** into aquaflux via ``read_openfoam`` and
solves the coupled RANS system on it, so the two solutions live on identical cells and are compared
directly (no interpolation between independent meshes, unlike ``validation/turbulent_channel_openfoam``,
whose cyclic mesh the reader cannot yet import).

aquaflux setup, as requested for this study:

* the **coupled** turbulent solver (:func:`aquaflux.turbulence.solve_coupled` -- one monolithic Newton
  on ``R(u, p, k, omega)``, globalized by pseudo-transient continuation);
* **hybrid initialization** (potential-flow velocity + Laplace-smoothed turbulence), which
  ``solve_coupled`` invokes automatically to self-start;
* **second-order upwind** momentum advection (:class:`aquaflux.discretization.LimitedUpwind` with the
  :class:`aquaflux.schemes.VenkatakrishnanLimiter` -- the upwind cell reconstructed to the face with
  its gradient, slope-limited so the reconstruction stays bounded). The stiff k/omega scalars use
  bounded first-order upwind: a second-order stencil there lets the coupled Newton step drive omega
  negative (a Newton-update, M-matrix effect the limiter does not prevent -- see ``solve_aquaflux``);
* **multiple-correction** gradients (:class:`aquaflux.schemes.MultipleCorrectionGradient`), a
  quadratic-exact reconstruction in two face passes -- the skewness/non-orthogonality correction this
  mesh needs (the analogue of OpenFOAM's ``corrected`` surface-normal treatment), reached without the
  Richardson sweeps that would push the differentiated residual further across the cell graph;
* **log-variable omega** (:class:`aquaflux.turbulence.LogScalars` on ``omega_transform``): ``omega =
  e^w`` stays strictly positive under any Newton step. Without it a direct-omega step drives omega
  negative once the recirculation forms, poisoning ``nu_t = k/omega`` while the residual stays finite
  (so the divergence guard never trips) -- the failure this case exposes and log-omega structurally
  removes.

The physics caveat this study documents: the pitzDaily mesh is a **wall-function** mesh (first-cell
``y+`` well above the viscous sublayer), whereas aquaflux's SST is **wall-resolving** (it fixes the
analytical sublayer ``omega`` at the wall-adjacent cell). The comparison therefore isolates the *outer*
flow -- the shear-layer development, the recirculation bubble, and the reattachment length -- where the
near-wall treatment matters least, and reports the near-wall fields as the expected point of departure.
The near-wall ``omega`` also differs because the two codes blend the viscous and log branches
differently: aquaflux uses ``sqrt(omega_vis**2 + omega_log**2)`` (the quadrature blend) while OpenFOAM's
default ``omegaWallFunction`` uses ``max(omega_vis, omega_log)`` -- a ~20% difference in the buffer layer
that is a blend-shape choice, not an error in either code.

**Reference caveat (binding -- do not skip):** the OpenFOAM *steady* (SIMPLE / ``ddtSchemes steadyState``)
run does **not** converge this case -- its ``omega`` field limit-cycles and *checkerboards* in the inlet
channel (adjacent cells oscillating between O(0.1) and O(1e8)), which is a non-physical, non-converged
field, not a valid solution. Comparing aquaflux's residual against such a field is meaningless (it will be
huge because the field is garbage, not because aquaflux is wrong). A stable steady solution *does* exist
and is recovered by a time-accurate transient (``pimpleFoam`` / an unsteady ``ddtSchemes``) run to a
statistically steady state; use a **transient-converged** OpenFOAM field as the comparison target, and
compare the outer-flow profiles (velocity, reattachment length) rather than the raw residual.

**Cost note (binding for whoever runs this):** the coupled log-omega solve on the full ~12k-cell mesh
is compute-heavy -- each Newton step is several minutes and the march is long, so a full run is a
matter of hours. omega-log is validated on a smaller channel (``tests/integration/test_coupled_rans``);
efficient large-mesh convergence (the reparametrized-block preconditioner scaling and the
globalization) is a known tuning follow-up. Track the **per-field relative** residuals when running --
the absolute ``||R||`` is dominated by omega's ~1e5 scale and is a misleading convergence metric.

Run (after ``run_of.sh``) from the repo root:
    python3 validation/pitzdaily_openfoam/compare.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

# Running a script puts the SCRIPT's directory on `sys.path`, not the working directory, so
# `python3 validation/pitzdaily_openfoam/compare.py` from the repo root cannot find `aquaflux` unless it
# is separately installed. Add the repo root explicitly so the documented invocation works against a
# plain checkout -- this case had no such bootstrap, so it could not be run through the case launcher
# at all, which is one reason it was left behind while the sibling case was developed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.io import read_openfoam
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import (
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    SweptGradientSolve,
    VenkatakrishnanLimiter,
)
from aquaflux.solve import (
    COMPILED as ILU0_COMPILED,
)
from aquaflux.solve import (
    AscendingRowLengthCells,
    CellMajor,
    CflResidualDualTimeControl,
    MarchLogger,
    NaturalCells,
    RefreshPolicy,
    RetryPolicy,
    ReverseCuthillMcKeeCells,
    StateCheckpointer,
    combine_observers,
    ilu_smoothed_inverse,
    jacobi_smoothed_inverse,
    relative_residual_gmres,
    simple_smoothed_inverse,
)
from aquaflux.turbulence import (
    BetaTaperedDamping,
    ConstantDamping,
    CoupledJacobianProbe,
    CoupledRANS,
    GeometricReynoldsSchedule,
    LogScalars,
    ResidualTaperedDamping,
    SSTModel,
    SSTTurbulence,
    amg_beta_tracking_refresh,
    coupled_amg_continuation,
    coupled_fields,
    scale_both_blocks,
    scale_momentum_only,
    solve_reynolds_continuation,
    solve_reynolds_ramp,
    turbulence_residual_norm,
    wall_consistent_state,
)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "kwsst"
# The comparison target is the TIME-ACCURATE run, not the steady one. The steady case does not
# converge on this geometry: it leaves an odd-even checkerboard in the inlet, with omega spanning
# 0.03 to 1.15e8 across adjacent cells. Ten of those cells alone carry the entire omega residual
# measured on that field, so anything calibrated against it is calibrated against numerical noise.
TRANSIENT = HERE / "of_transient" / "0.14"
FIGS = HERE / "figures"

# The pitzDaily operating point (0/ and constant/): U_in = 10 m/s, nu = 1e-5, k_in = 0.375,
# omega_in = 440.15. rho = 1 (incompressible kinematic).
RHO, NU = 1.0, 1e-5
U_IN, K_IN, OMEGA_IN = 10.0, 0.375, 440.15
WALLS = ["upperWall", "lowerWall"]
STEP_X, STEP_Y = 0.0, 0.0  # the step lip; the lower wall drops to y = -0.0254 for x > 0
# The coupled Newton march budget. This is a stiff, separating, high-Re case on a wall-function mesh
# (aquaflux's SST is wall-resolving), so it converges to an engineering tolerance rather than machine
# zero; the cap is generous so the march exits on the tolerance, not the count.
#: ⚠️ AN ABSOLUTE BAR, MATCHING THE SIBLING CASE: `rtol = 0` so the test is `|R| <= atol` outright.
#: A RELATIVE tolerance means each Reynolds rung targets a fraction of its OWN starting residual, so
#: the cheap anchor rung -- whose only job is to hand the next one a warm start -- is asked for a
#: harder solve than the target rung ever needs. Run briefly with `rtol = 1e-6` here, rung 1's target
#: came out at 8.7e-08 against the 1e-05 the sibling case asks of any rung.
MAX_STEPS = 150  # per continuation rung
RTOL, ATOL = 0.0, 1e-5

# ---------------------------------------------------------------------------------------------
# The march configuration.
#
# ⚠️ THIS CASE RAN FOR A LONG TIME ON A SINGLE-STEP PSEUDO-TRANSIENT MARCH WITH NO DUAL-TIME INNER
# LOOP, NO COURANT CONTROL, NO RETRY LADDER AND NO PER-STEP LOG. All of that was built and calibrated
# on the three-dimensional case and never carried back here, so this case could not benefit from any
# of it -- and, worse, a timing taken from it measured the globalization rather than whatever was
# being studied. Under the old configuration the cold march is a documented reachability crawl,
# needing on the order of eight hundred outer steps to develop the recirculation against a two
# hundred step cap: it could not converge however long it was left.
#
# The values below are the three-dimensional case's, because that is where each was measured. Two are
# load-bearing enough to name:
#
#   * `POSITIVITY_FLOOR` -- without it the step limiter's room is a purely RELATIVE quantity, so a
#     numerically dead cell ratchets the global step cap by a factor of a hundred per step until the
#     march is taking no step at all while every field still reads finite.
#   * `scaled_norm` -- the coupled Euclidean residual is very nearly all omega, so a march judged on
#     it cannot see the flow converge. The row-scaled measure judges every equation comparably.
# ---------------------------------------------------------------------------------------------

#: ⚠️ REYNOLDS CONTINUATION, BECAUSE A COLD SOLVE AT THE TARGET REYNOLDS NUMBER DOES NOT REACH THE
#: ROOT IN ANY REASONABLE NUMBER OF STEPS. Measured on this case without it: the march is perfectly
#: healthy -- full steps, line search never clipping -- and contracts a steady 2.7% per outer step,
#: which needs about 490 steps to reach the stopping tolerance against a 200-step cap. It is not a
#: solver in difficulty; it is a solver correctly integrating a long transient, which is precisely the
#: reachability problem continuation exists to short-circuit. Each rung starts from the previous one's
#: converged field, so the expensive target rung begins near its own root instead of at a cold start.
#:
#: `N_POINTS` is the number of INTERMEDIATE rungs: 2 gives Re/100, Re/10, target, matching the
#: three-dimensional case.
N_POINTS = int(os.environ.get("PITZ_N_POINTS", "2"))

#: The Reynolds-number multiplier per continuation rung. The default decade is a round number in log
#: space rather than a measured one: geometric spacing is justified (the convective nonlinearity scales
#: multiplicatively with Re) but the value is not.
#:
#: Varying it alone changes two things at once, so it is only interpretable when paired with
#: `PITZ_N_POINTS`: the anchor sits at `RATIO ** N_POINTS`, so holding that product fixed varies the
#: ladder's *granularity* at a fixed *span*, which is the comparison worth making. `10 ** 2`,
#: `3.1623 ** 4` and `2.1544 ** 6` all anchor at Re/100.
RATIO = float(os.environ.get("PITZ_RATIO", "10.0"))

#: ⚠️ THE DEFAULT SINCE 2026-09-10: WALK THE VISCOSITY DOWN INSIDE ONE MARCH RATHER THAN SOLVING A
#: LADDER OF RUNGS. `PITZ_RAMP=off` returns to `solve_reynolds_continuation`, which is kept as the
#: comparison arm rather than as a supported path -- this case is where the coupled march's work lands,
#: and it lands on the ramp.
#:
#: Measured on this case, both arms reaching `x_r/h` 8.0686 and `nu_t` peak 417.8, uncontended, warm
#: compilation cache, commit 72c9a96, `simplesmooth` flow inverse:
#:
#:     ladder (PITZ_RAMP=off)                69 outer steps / 417 restart cycles / 555 s
#:     ramp, 4 stations x 3 steps            40 outer steps / 261 restart cycles / 363 s
#:     ramp, 24 stations x 1 step (default)  33 outer steps / 191 restart cycles / 296 s
#:
#: ⚠️ Read the CYCLE column. Outer-step and restart-cycle counts are deterministic and survive machine
#: contention; wall clock does not -- a bit-identical trajectory measured 1325 s against 555 s on this
#: machine depending only on what else was running.
#:
#: The ladder solves each Reynolds rung as its own march, which pays two costs per rung that a single
#: march does not:
#:
#:   * **Each rung is converged, and the next rung's viscosity jump immediately undoes it.** Rung 1
#:     spent 12 of its 28 steps taking `|R|` from 4.5e-04 to 7.2e-06 -- and rung 2 opened at 4.5e-02,
#:     six thousand times worse. Rung 2 then converged to 4.6e-06 and the target opened at 3.0e-03.
#:     Every one of those polishing steps was discarded work: a seed does not need to be a root.
#:   * **Each rung restarts the pseudo-timestep ramp.** `beta` reopens at `BETA_START` on every rung
#:     and walks back to `beta_min` one `1/grow` notch per outer step -- 12 steps at these values,
#:     measured with `a_min = 1.000` on all of them, i.e. the control had no reason for caution and
#:     crawled anyway. Rungs 1, 2 and 3 spent 12, 15 and 16 steps there.
#:
#: One march keeps the state, the shift and the preconditioner across every viscosity change and
#: converges the target and nothing else. The ramp spans the SAME viscosity range as the ladder
#: (`RATIO ** N_POINTS`), so the two arms differ in how the span is walked, not in how far.
#:
#: ⚠️ ONE STEP PER STATION, and the opposite was believed until it was measured. The argument for long
#: stations was that each change re-points the refresh hook and forces a FULL preconditioner rebuild,
#: so per-step viscosity would cost more than it saves. On this case a rebuild is 1.2-1.6 s against a
#: ~9 s outer step -- a sixth of one -- and the fine ramp is the cheapest schedule tried (11 arms).
#: The reason it wins is not the rebuild accounting but the shift: at 24 stations the viscosity moves
#: 1.21x per station instead of 3.16x, so the problem barely moves, no re-damping is needed, and beta
#: descends monotonically through the ramp. The coarse schedule's re-damping very nearly cancelled the
#: descent (net 0.889 per station), handing the target station beta = 0.936 and making it re-descend
#: beta itself -- the very cost this arm exists to remove, recreated inside the ramp.
#: ⚠️ THE BALANCE SHIFTS ON A LARGER CASE: on the 3D sibling a rebuild is 11.5 s against a ~34 s mean
#: step (measured 2026-09-10) -- a third of a step rather than a sixth. An earlier note here said ~36 s,
#: a whole step, taken from the design record rather than a log; it is wrong by 3x.
#: These values are a pitzDaily calibration; measure before carrying them anywhere else.
RAMP = os.environ.get("PITZ_RAMP", "continuous")
#: Which viscosity a ramp station scales (`PITZ_RAMP_SCALE`). `both` (the default) makes each station a
#: genuine lower-Reynolds problem, the same path the rung ladder walks. `flow` scales the momentum
#: block only and leaves the closure at the case's own viscosity, which keeps the near-wall `omega` at
#: its target profile for the whole march instead of starting it a viscosity-ratio high and walking it
#: back down -- at the price of stations that are not a physical Reynolds number, and of `k`/`omega`
#: carrying their target stiffness from the first step. Which of those dominates is a property of the
#: case; see the README.
#: The shift strength the control is allowed to descend to. It is the march's development scale: the
#: control divides `beta` by `grow` per comfortable step, so a rung spends
#: `ceil(ln(beta_start / BETA_MIN) / ln(grow))` steps reaching it -- 12 at these values -- and after
#: that `beta` is pinned and stops carrying any information about the march's phase. `TURB_TAPER` keys
#: on exactly that descent, so the two must be read together.
BETA_MIN = 0.005

#: How much harder the `k`/`omega` rows are damped than the flow rows (`PITZ_TURB_DAMPING`): the shift
#: STRENGTH on those rows is multiplied by this, so they run at an effective `TURB_DAMPING * beta`
#: while the velocity rows keep `beta`. `1.0` (the default) is the single shift the march has always
#: used.
#:
#: The motivation is the split the scaling arms exposed: the momentum block's difficulty is the
#: convective nonlinearity, which the viscosity ramp weakens directly, while the closure's is its stiff
#: source terms, which the ramp barely touches and a diagonal shift addresses squarely. Sharing one
#: `beta` means it is set by whichever block is more fragile. The shift vanishes at the root, so this
#: moves the path and neither the converged solution nor its adjoint.
#:
#: ⚠️ The step control still adapts ONE `beta` against a global line search, so it cannot see which
#: block is asking for the caution -- which is why the ratio is a strategy the caller picks rather than
#: something the control adapts. A per-block adaptive shift would need a per-block signal first.
#: ⚠️⚠️ SWEPT 2026-09-10 on momentum-only scaling at 16 x 1, this file's other defaults, one run each --
#: AND MEASURED UNDER A CONFOUND THE LIBRARY HAS SINCE REMOVED, so re-measure before quoting it.
#: `1` is a control and reproduced the recorded 27 steps / 227 cycles exactly.
#:
#:     gamma    1      2      5     10
#:     steps   27     27     29     41
#:     cycles 227    191    206    305
#:
#: The confound: the damping was folded into the shift's base DIAGONAL, and that diagonal is also the
#: row scale of `coupled_scaled_norm` -- the measure the march is steered by, stopped on (`atol` 1e-5)
#: and compared across arms with. So every arm above divided its own `k`/`omega` residual rows by its
#: own gamma, i.e. each ran to a different physical bar, with the looser bar going to the larger gamma.
#: The bias favours large gamma throughout, which cuts both ways here: gamma=2's 16% win over the
#: control is partly bought by it, while gamma=10's finishing WORSE than no damping at all is if
#: anything understated. The damping now multiplies the shift STRENGTH and leaves the diagonal alone,
#: so the measure no longer moves with the knob and the arms are comparable.
#:
#: What the sweep established that the confound does not touch, because it is read WITHIN an arm: the
#: two march phases want opposite ratios -- early, more damping is monotonically better (cycles to step
#: 5: 17/12/12/11), and late it reverses (gamma 5 stalls once, gamma 10 stalls twice and moves
#: BACKWARDS). And no arm took a retry, gamma=10 included: damping does stabilize the step, and its
#: cost is the closure LAGGING the mean flow, which reads as a stalled residual at healthy alpha rather
#: than as a divergence.
TURB_DAMPING = float(os.environ.get("PITZ_TURB_DAMPING", "1.0"))
#: Taper the ratio from `TURB_DAMPING` down to 1 instead of holding it constant (`PITZ_TURB_TAPER`,
#: the taper's exponent; `0`, the default, keeps the constant). `PITZ_TURB_TAPER_KEY` picks WHAT the
#: release is keyed on -- `residual` (the default) or `beta`.
#:
#: ⚠️ THE PHASE SPLIT IS THE MOTIVATION, AND IT IS MEASURED (2026-09-10, momentum-only at 16 x 1, this
#: file's other defaults, one run per arm, all reaching x_r/h 8.0686, all under the corrected measure):
#:
#:     arm            steps  cycles  esc   ramp(16)  target
#:     gamma = 1         27     227    0         --      --
#:     gamma = 2         29     200    0        129      71
#:     gamma = 10        64     329    0        100     229
#:     taper 2 -> 1      28     234    0        150      84
#:     taper 10 -> 1     39     319    5        135     184
#:
#: `gamma = 10` buys the CHEAPEST ramp of any arm and the worst target station. So the two phases want
#: opposite ratios, and the prize for a taper that gets both is ramp 100 + target 71 = ~171 against the
#: best constant's 200 -- IF they compose, which is untested.
#:
#: ⚠️ `gamma = 10`'s cost is LAG, not instability: 64 steps, zero escalations, zero stalls, alpha =
#: 1.000 throughout, the residual crawling at ~0.74 per step for 34 steps at 3 cycles each. Read a
#: damping failure as a residual that will not fall at healthy alpha, never as a divergence.
#:
#: ⚠️ WHY `beta` IS THE WRONG KEY -- measured, not argued. `beta` reaches `BETA_MIN` at step 12 of the
#: 16-step ramp, so a beta-keyed release is spent ENTIRELY INSIDE the phase that wants the damping, and
#: the target station inherits gamma = 1. Its ramp costs 135 against the constant's 100 and its target
#: is no better than undamped. `beta` is anti-aligned with the phase structure it was meant to track.
#: A second, separate defect: `beta` is adaptive, so the retry ladder RAISES it and the damping rises
#: with it -- in the `10 -> 1` arm gamma climbed back to 6.4 over steps 20-27 while the residual rose
#: monotonically, five escalations, ~90 cycles lost. The damping and the control's own reaction to a
#: bad step form a positive feedback loop.
#:
#: The residual has neither property: it does not floor during the ramp, and it is what the two phases
#: actually differ in. `turbulence_residual_norm` reads the k/omega rows only, because the blocks settle
#: on different schedules and a whole-state norm would release the closure's damping on the FLOW's
#: progress.
#:
#: ⚠️ The recorded reason the residual key was once inert does NOT apply to a momentum-only arm, which
#: is why it is the default here. That diagnosis -- the reference is frozen at the seed while a
#: continuation makes the problem harder as it walks, so the clamp pins the factor at `initial` -- was
#: measured under BOTH-BLOCKS scaling, where each station changes the closure's own viscosity. Under
#: `scale_momentum_only` the closure's assembler is the same object at every station, so the walk
#: cannot move its residual that way. Whether it releases on the schedule the table above wants is the
#: open question, and it is answerable from the run: `PITZ_CHECKPOINT_KEEP` high enough to keep every
#: step lets `turbulence_residual_norm` be replayed per step and the applied gamma reconstructed.
#: The ratio the TARGET station runs at, once the viscosity ramp has arrived (`PITZ_TURB_DAMPING_TARGET`).
#: `PITZ_TURB_DAMPING` then applies to the ramp stations only. Unset means one ratio for the whole
#: march, which is byte-identical to the single-constant behaviour.
#:
#: ⚠️⚠️ MEASURED, AND THE PHASES DO NOT COMPOSE -- so this knob is an instrument, not a win. The split
#: looks large and real: the ramp's cost falls from 129 cycles at gamma=2 to 102 at 5 and 100 at 10
#: (turning back up to 119 at 20), while the target station's rises from 71 at gamma=2 to 110 at 5 and
#: 229 at 10. Best-of-both would be ~171 against the best single constant's 200. It is not available:
#:
#:     arm             ramp  target  total   R handed to the target
#:     gamma = 2        129      71    200   2.814e-03
#:     gamma = 5        102     110    212   5.796e-03
#:     gamma = 10       100     229    329   1.005e-02
#:     station 5 -> 2   102     109    211   5.796e-03
#:
#: The `5 -> 2` arm gets its cheap ramp exactly as asked -- 102 cycles, matching constant gamma=5 to
#: the cycle -- and switching the TARGET to 2 then buys **one** cycle. The target's cost is set by the
#: STATE THE RAMP HANDS IT, not by its own ratio, and the handover residual scales with how hard the
#: ramp damped. So a cheap ramp is not earned, it is BORROWED: damping does not remove the closure's
#: work, it defers it. The two phases are one budget, and that budget is smallest at a constant 2.
#:
#: ⚠️ A TAPER CANNOT EXPRESS THIS, which is why the knob is a pair of constants rather than a shape.
#: Every signal a shift policy can read for itself measures march PROGRESS, and progress saturates
#: long before the last station: keyed on `beta` the release is complete by step 12 of the 16-step
#: ramp, keyed on the closure residual by step 6 (measured -- `damping_taper_trace.py` replays it from
#: the checkpoints). Both therefore spend the whole release inside the phase that wants the damping and
#: hand the target station a ratio of 1. The discriminator is the station index, which only the march
#: knows, and `solve_coupled(station_step=...)` is how it arrives.
TURB_DAMPING_TARGET = float(os.environ.get("PITZ_TURB_DAMPING_TARGET", "0") or 0.0)
TURB_TAPER = float(os.environ.get("PITZ_TURB_TAPER", "0") or 0.0)
TURB_TAPER_KEY = os.environ.get("PITZ_TURB_TAPER_KEY", "residual")
if TURB_TAPER_KEY not in ("residual", "beta"):
    raise SystemExit(f"PITZ_TURB_TAPER_KEY must be 'residual' or 'beta', got {TURB_TAPER_KEY!r}")
#: How the banner says which of the three shapes ran. A damping that reports only its INITIAL ratio
#: reads identically whether it tapered or not, and the taper is the whole variable under test.
_TURB_DAMPING_SHAPE = (
    f" tapered on {TURB_TAPER_KEY}, exponent {TURB_TAPER:g}"
    if TURB_TAPER
    else (
        f" on the ramp, {TURB_DAMPING_TARGET:g} at the target"
        if TURB_DAMPING_TARGET
        else " (constant)"
    )
)
RAMP_SCALE = os.environ.get("PITZ_RAMP_SCALE", "both")
_RAMP_SCALINGS = {"both": scale_both_blocks, "flow": scale_momentum_only}
if RAMP_SCALE not in _RAMP_SCALINGS:
    raise SystemExit(f"PITZ_RAMP_SCALE={RAMP_SCALE!r} is not one of {sorted(_RAMP_SCALINGS)}")
RAMP_COMPANION = _RAMP_SCALINGS[RAMP_SCALE]
RAMP_STATIONS = int(os.environ.get("PITZ_RAMP_STATIONS", "24"))
RAMP_STEPS_PER_STATION = int(os.environ.get("PITZ_RAMP_STEPS", "1"))

#: ⚠️ RE-DAMP ON ENTERING EACH STATION, AND THIS IS LOAD-BEARING RATHER THAN A KNOB. Within a station
#: the control divides the shift by `grow` every step (1.5 ** 3 = 3.375 per station here). Unopposed,
#: that walks beta 0.5 -> 0.148 -> 0.044 -> 0.013 across four stations, and this case has a wall at
#: beta ~ 0.012: measured, the line search collapsed to alpha = 0, |R| went 4.4e-03 -> 4.4e-01, and
#: the retry ladder escalated beta to ~2 to recover -- 50 steps / 301 cycles against 36 / 269 for the
#: same march that never went there. At 2.0 the net is 3.375 / 2 = 1.69 per station, so beta halves
#: per station and bottoms at 0.062.
#: ⚠️ Two different floors here, and they are easy to conflate. `CONTROL.beta_min` (0.005) bounds the
#: shift the OPERATOR is solved with; `PC_BETA_FLOOR` (0.05) floors only the preconditioner's own copy.
#: The preconditioner sat at 0.05 throughout the collapse and never saw 0.012 -- so the wall is the
#: operator/preconditioner MISMATCH opening up, not a shift the V-cycle cannot invert. And the
#: mismatch's size is not the discriminator either: this same march converged at beta = 0.005 against
#: the same 0.05 preconditioner (a 10x mismatch) with alpha = 1.000, where 4.2x was fatal mid-ramp.
#: The wall belongs to `(state, beta)`. Damping while the problem MOVES is the distinction.
RAMP_REDAMPING = (
    float(os.environ["PITZ_RAMP_REDAMPING"]) if "PITZ_RAMP_REDAMPING" in os.environ else None
)

#: The dual-time inner loop. `inner_tol` 1e-2 rather than a tighter value: measured on the
#: three-dimensional case, 1e-3 bought nothing over 1e-2 while costing a third of the march.
INNER_STEPS, INNER_TOL = 5, 1e-2

#: Buys the step limiter out of a numerically dead cell instead of letting one cell ratchet the
#: global step cap toward zero. ⚠️ Reachable only because this case uses `coupled_amg_continuation`:
#: it is a parameter of that builder ALONE, and the default, complete-LU and threshold-ILU builders
#: expose neither it nor the `step_limit` it would be set on.
K_POSITIVITY_FLOOR = 1e-8

#: The inexact-Newton stop per inner linear solve, in the row-scaled measure, and the Krylov restart.
#: `FORWARD_MAX_RESTARTS` bounds a single solve: past the retry threshold the attempt is going to be
#: discarded anyway, so running it to a stagnation is work thrown away. Strictly above the threshold,
#: because the march's test is `>`, and a cap landing exactly on it would accept a truncated direction
#: instead of escalating.
FORWARD_RTOL, FORWARD_RESTART = 0.3, 15
FORWARD_MAX_RESTARTS = 14

#: ⚠️ **THIS WHOLE BUNDLE IS UNREACHABLE AT THE CURRENT DEFAULTS — see `_ILU_SMOOTHER_LIVE` below.**
#: Every measurement in it was taken when the leading `[u, v, p]` block was inverted by the incomplete-LU
#: -smoothed hierarchy these settings configure. `FLOW_INVERSE` now defaults to `simplesmooth`, which
#: supplies that block's inverse directly, and the trailing block's is supplied too, so nothing here is
#: constructed on the default path. The bullets below are kept because they are the record of a real
#: measurement and because the settings revive the moment a block's inverse is `None` -- but they are
#: **not** a description of how a march at the defaults is preconditioned, and a run banner quoting them
#: is not evidence about that march. Read them as history for the ILU path, not as live configuration.
#:
#: ⚠️ THE VALIDATED SMOOTHER BUNDLE, AND NONE OF IT IS OPTIONAL. These are the library defaults'
#: opposites, and each was measured on the sibling case at adjoint-grade tolerance:
#:   * ⚠️ `FILL_LEVELS` **1** HERE, WHERE THE SIBLING CASE USES 0 -- THE TWO RANK THIS OPPOSITELY, AND
#:     copying the sibling's value is what kept this case from taking a single step. At zero fill the
#:     level sweep is not a contraction on this leading block: it AMPLIFIES, and the four sweeps below
#:     compound it (one apply of the split reads 9.88e+05 at one sweep and 1.56e+31 at four). With fill
#:     1 the same operator takes ONE matvec to the march's stop.
#:     The discriminator is a pivot census, and it inverts between the cases: this block's ILU(0) has
#:     NEGATIVE pivots at every shift (27/25/9 of 36675) with min |pivot| some twenty times smaller
#:     than the sibling's, whose ILU(0) has none at any shift. There the fill produces the negative
#:     pivots and dropping it is the fix; here dropping it produces them and the fill is the fix.
#:     ⚠️ **EVERY NUMBER IN THIS BULLET WAS TAKEN UNDER THE MESH'S OWN CELL ORDER, AND THAT IS THE
#:     VARIABLE THAT ACTUALLY DECIDES IT.** Re-measured on this case's `[u, v, p]` block at reach 5,
#:     changing nothing but the order the same matrix is eliminated in: at zero fill the shipped order
#:     amplifies a stationary sweep 5.5x in one application and stalls the Krylov solve, while a
#:     reverse-Cuthill-McKee or ascending-row-length CELL order contracts it and converges in ~113-140
#:     applications. So "zero fill amplifies on this block" is true of this ordering, not of zero fill;
#:     `FILL_LEVELS = 1` remains right for the PETSc path, whose ordering is its own separate option,
#:     but it is no longer evidence that the block needs fill. See `PITZ_FLOW_ORDER`.
#:   * `SWEEPS` 4 -- zero-fill is the weaker smoother, so extra sweeps pay more than they did for
#:     ILU(1) (390 -> 69 iterations at beta 0.01). The library default of 2 was tuned against ILU(1)
#:     and does not carry over.
#:   * `COARSE_EQ_LIMIT` 2000 -- the default coarsens to ~50 equations, whose direct solve captures
#:     only the crudest global mode, and the indefinite saddle's wall is exactly that global pressure
#:     coupling. `None` stalls at every low shift. Not optional.
#:   * `PC_BETA_FLOOR` 0.05 -- the V-cycle is built at `max(beta, floor)` while the march still solves
#:     at its own shift. The OPERATOR is untouched, so the converged root and the adjoint are
#:     unchanged, and the mismatch saturates instead of growing as the shift falls.
FILL_LEVELS, SWEEPS, COARSE_EQ_LIMIT, PC_BETA_FLOOR = 1, 4, 2000, 0.05

#: The field split: the `[u, v, p]` saddle and the `[k, omega]` transported pair get their own
#: hierarchies, because a saddle and an advection-diffusion-reaction pair coarsen differently. Measured
#: 31% faster end to end on the sibling case -- while taking MORE Krylov cycles, because two smaller
#: V-cycles plus one sparse coupling product apply far more cheaply than one six-field V-cycle.
FIELD_SPLIT = os.environ.get("PITZ_FIELD_SPLIT", "1") not in ("", "0")
TRAILING_SWEEPS = 1

#: Clip each cell's own correction rather than scaling the whole step by the worst cell. **ON since
#: 2026-08-25**, because on this case the plain global cap was measured losing a march outright.
#:
#: Every failing step length under the cap was the cap and not a rung of the line search's ladder
#: (`0.003108`, `0.154`, `0.0004484` -- none a power of one half, the last BELOW the shortest rung),
#: and the march then died in the `1 - tau`-per-step collapse `positive_block_projection` derives, its
#: inner residual running `8.462e-06 -> 8.353e-08 -> 8.353e-10` at ratios of exactly 0.01. Raising
#: `K_POSITIVITY_FLOOR` cannot remove that -- `(k + floor)` decays by the same factor whatever the
#: floor is.
#:
#: Measured here, same commit, same everything else: an arm that stalls at the target rung under the
#: cap completes under the projection, and **the arm that already worked got faster** -- 703.7 s /
#: 437 cycles / 73 steps against 664.0 s / 459 cycles / 67 steps, both `x_r/h` 8.069. Two gradient
#: reconstructions whose costs differ sharply under the cap land within 0.5 % of each other under it.
#: `PITZ_K_POSITIVITY_PROJECTION=0` restores the cap.
POSITIVITY_PROJECTION = os.environ.get("PITZ_K_POSITIVITY_PROJECTION", "1") not in ("", "0")

#: ⚠️ THE WALL CONDITION ON `k`, AND IT IS A CHOICE OF PROBLEM RATHER THAN OF SOLVER. Turbulent
#: fluctuations vanish at a no-slip wall, so `k -> 0` and `Dirichlet(0)` is the textbook condition --
#: but it makes a DIFFERENT discrete problem from the zero-gradient one, with its own reattachment
#: length, so the two cannot be compared and a number from one is not a target for the other. The
#: sibling case runs zero-gradient, and this case is the same geometry, so it runs zero-gradient too.
_K_WALL_BCS = {"dirichlet": Dirichlet(0.0), "zerogradient": ZeroGradient()}
K_WALL = os.environ.get("PITZ_K_WALL", "zerogradient")
if K_WALL not in _K_WALL_BCS:
    raise SystemExit(f"PITZ_K_WALL={K_WALL!r} is not one of {sorted(_K_WALL_BCS)}")
K_WALL_BC = _K_WALL_BCS[K_WALL]


#: ⚠️ WHICH INVERSE THE LEADING `[u, v, p]` BLOCK GETS. `simplesmooth` (default since 2026-08-22) is a
#: multigrid hierarchy over the saddle relaxed by SIMPLE sweeps, matching the sibling 3D case's own
#: default. `petsc` is the host GAMG V-cycle smoothed by PETSc's incomplete factorization, at the
#: `FILL_LEVELS` above. `hostilu` is this package's own hierarchy smoothed by its own factorization --
#: which is ZERO-FILL by construction and has no fill parameter at all.
#:
#: ⚠️⚠️ **THE DEFAULT MOVED OFF `petsc` BECAUSE `petsc` STOPPED MARCHING THIS CASE, and the failure is
#: the incomplete factorization's order-and-fill dependence rather than anything about the physics
#: (2026-08-22).** Under `petsc` the march now collapses at the FIRST step of the second Reynolds rung:
#: `alpha` 0, `beta` escalating 0.5 -> 2 -> 16 through the whole ladder, the residual rising
#: 1.674e-01 -> 5.754e-01 -> `inf`. Reproduced three times, including on a tree carrying no local
#: change at all, with the step tables bit-identical -- so it is a property of this bundle and not of
#: whatever else was in flight. The same case under `simplesmooth` marches straight through to the
#: same answer (`x_r/h` 8.0686, `ux` 0.0191, 404 cycles, 711 s) as the last good `petsc` run
#: (8.0686, 0.0191, 743 s), which is what makes this a swap of preconditioner rather than of result.
#:
#: The discretization is NOT what changed, and that was checked rather than assumed: `|R|` at a fixed
#: saved state is identical to twelve digits across every commit merged that day, and reverting the
#: one of them that is recorded as not bit-identical reproduces the collapse unchanged. What sits
#: behind it is the standing property of a zero- or low-fill factorization on this block -- the
#: elimination ORDER decides which couplings it discards, and this same case is on record going from
#: amplifying a residual 5.5x per sweep to contracting it on nothing but a reordering. A SIMPLE-smoothed
#: hierarchy never eliminates the matrix at all; it forms an approximate Schur complement and applies
#: V-cycles, so it has no order or fill to be sensitive to. `petsc` and `hostilu` both stay reachable
#: and both stay measured -- what is no longer defensible is either of them as the *default* here.
#:
#: ⚠️ **`hostilu` was predicted to fail here because zero fill was measured to amplify on this block,
#: and it did -- but the diagnosis was incomplete: the amplification is a property of the ELIMINATION
#: ORDER, not of the fill level alone.** Measured on this case's `[u, v, p]` block at the exact reach,
#: with nothing changed but the order the same matrix is eliminated in, a zero-fill factorization goes
#: from amplifying the residual 5.5x in one stationary sweep (and stalling the Krylov solve it
#: preconditions) to contracting it and converging in ~113 applications. So a zero-fill smoother is not
#: ruled out on this case; the mesh's own cell order is. See `PITZ_FLOW_ORDER`.
FLOW_ORDER = os.environ.get("PITZ_FLOW_ORDER", "natural")
_FLOW_ORDERS = {
    "natural": NaturalCells,
    "rcm": ReverseCuthillMcKeeCells,
    "rowlength": AscendingRowLengthCells,
}
if FLOW_ORDER not in _FLOW_ORDERS:
    raise SystemExit(f"PITZ_FLOW_ORDER={FLOW_ORDER!r} is not one of {sorted(_FLOW_ORDERS)}")
_FLOW_INVERSES = ("simplesmooth", "petsc", "hostilu")
FLOW_INVERSE = os.environ.get("PITZ_FLOW_INVERSE", "simplesmooth")
if FLOW_INVERSE not in _FLOW_INVERSES:
    raise SystemExit(f"PITZ_FLOW_INVERSE={FLOW_INVERSE!r} is not one of {list(_FLOW_INVERSES)}")

#: `simplesmooth` -- a multigrid hierarchy over the `[u, v, p]` saddle relaxed by SIMPLE sweeps rather
#: than by an incomplete factorization, and this case's default since 2026-08-22 (see above). A SIMPLE
#: sweep relaxes through diagonal and Schur approximations, so unlike an incomplete factorization it
#: does not take its pattern from the stored sparsity -- which is why it may answer to the probe's
#: reach quite differently, and why neither the fill nor the elimination order that decide the ILU
#: arms applies to it at all.
#:
#: ⚠️ The settings are the sibling's and are NOT established here. That case ranks a smoother knob
#: oppositely (see `FILL_LEVELS`) and coarsens about three times per level where this one manages
#: seven, so `strength_threshold` and `sweeps` in particular are open questions on this mesh rather
#: than values to trust. `PITZ_FLOW_SWEEPS` is exposed for that reason.
SIMPLE_FLOW = dict(
    sweeps=int(os.environ.get("PITZ_FLOW_SWEEPS", "2")),
    pressure_sweeps=2,
    strength_threshold=0.25,
    avoid_singletons=True,
    aggressive_levels=0,
    max_levels=5,
    max_coarse=500,
    block_splitting=True,
    omega=1.0,
)
#: Settings deliberately NOT ported from the sibling study: this case ranks a smoother knob oppositely,
#: so its sweeps and threshold are its own question rather than a value to copy.
HOST_FLOW = dict(
    sweeps=int(os.environ.get("PITZ_FLOW_SWEEPS", "1")),
    cycles=1,
    # The order the zero-fill smoother eliminates in -- the largest single lever measured on this
    # block, and the reason `hostilu` is worth re-running here at all (see `FLOW_ORDER`).
    ordering=CellMajor(_FLOW_ORDERS[FLOW_ORDER]()),
    strength_threshold=0.0,
    avoid_singletons=True,
    aggressive_levels=0,
    max_levels=10,
    max_coarse=500,
    prolongation_smoothing="none",
)
LEADING_INVERSE = (
    ilu_smoothed_inverse(**HOST_FLOW)
    if FLOW_INVERSE == "hostilu"
    else simple_smoothed_inverse(**SIMPLE_FLOW)
    if FLOW_INVERSE == "simplesmooth"
    else None
)

#: Whether `FILL_LEVELS` / `SWEEPS` / `COARSE_EQ_LIMIT` reach the preconditioner at all.
#:
#: ⚠️ They configure an incomplete-LU-smoothed hierarchy built ONLY for a block whose own inverse was
#: not supplied: the field split takes `leading_inverse(...)` when one is given and falls back to
#: building that hierarchy otherwise, and likewise for the trailing block. Under the field split this
#: file always supplies the trailing (Jacobi-smoothed) inverse, and `FLOW_INVERSE` supplies the leading
#: one unless it names neither strategy -- so at the defaults BOTH are given, neither fallback is taken,
#: and these three settings are dead. Note `hostilu` does not revive them either: it is
#: `ilu_smoothed_inverse(**HOST_FLOW)`, which carries its own fill and sweeps.
#: The banner used to print them regardless, which is how a reader (and a solver study) comes to
#: believe a march was preconditioned by a smoother that was never constructed. A banner is the primary
#: record of what a measurement was taken under, so it must separate a live setting from a carried one.
_ILU_SMOOTHER_LIVE = not FIELD_SPLIT or LEADING_INVERSE is None

#: The trailing `[k, omega]` block's inverse: the differentiable-framework nodal hierarchy, which the
#: sibling case defaults to after a controlled pair measured it ahead of the host V-cycle (67 steps and
#: 2124 s against 72 and 2893, to the same reattachment length).
JACOBI_TRAILING = {"max_coarse": COARSE_EQ_LIMIT, "equilibrate": False}

#: ⚠️ THE PROBED JACOBIAN IS EXACT ONLY AT REACH 5 ON THIS MESH, AND AT REACH 3 ON THE SIBLING'S --
#: WITH IDENTICAL SCHEMES. The cause is the mesh, not the dimension, and it generalizes.
#:
#: `CorrectedGreenGauss` does not compute a gradient in one shot: it solves `A_g G = B phi` by
#: Richardson sweeps (four, by default), and each sweep extends the gradient's stencil by one ring, so
#: the residual reaches `sweeps + 1`. But that coupling is weighted entirely by the skewness offset
#: `D_g,ip = x_f - (x_P + g*d)`. Where it vanishes, `A_g` is diagonal, sweeps two onward add exactly
#: nothing, and the scheme degenerates to compact Green-Gauss at reach 1.
#:
#:      mesh        median skew   max skew   interior faces above 1e-10
#:      pitzDaily      2.2e-09     7.5e-02       20049 of 24170
#:      bfs3d          7.0e-15     1.9e-12           0 of 66368
#:
#: The sibling is a rectilinear blockMesh, skew-free to roundoff, so its sweeps are INERT; this mesh
#: has the slanted lower wall and the contraction, so they are not. Confirmed four ways, the cleanest
#: being that this case with `sweeps=1` floors at reach 3 exactly as the sibling does.
#:
#: ⚠️⚠️ SO `stencil_reach = 3` IS A PROPERTY OF SKEW-FREE MESHES, NOT OF THE DISCRETIZATION. Any case
#: on a genuinely skewed mesh needs `sweeps + 1`, in three dimensions as much as in two. The sibling
#: gets 3 for free and that is luck, not physics.
#:
#: ⚠️ **EVERYTHING ABOVE IS ABOUT THE SWEPT RECONSTRUCTION, AND THE DEFAULT NO LONGER USES ONE.** The
#: shipped `GRADIENT` is the two-face-pass `MultipleCorrectionGradient`, whose residual carries
#: **exactly zero** Jacobian mass beyond distance 3 at every skewness measured -- so a probe at reach 3
#: recovers it exactly, and 3 is the default. Reach is a property of that scheme rather than of this
#: mesh; `validation/gradient_stencil_reach.py` re-measures it in about a minute.
#:
#: ⚠️⚠️ **SET IT BACK TO 5 IF YOU SET `PITZ_GRADIENT=swept`, AND READ THIS BEFORE VARYING EITHER.** The
#: swept scheme's residual reaches `sweeps + 1`, and reach 5 is then NECESSARY AND NOT SUFFICIENT --
#: measured against the smoother fill beside it, on the leading block at beta = 2:
#:
#:      reach 3 + fill 1   fails  (300 matvecs, true residual 3.36)
#:      reach 5 + fill 0   fails  (300 matvecs, true residual 3.50)
#:      reach 5 + fill 1   ONE matvec to the march's stop, ten to 1.7e-09
#:
#: Neither alone is worth anything, which is exactly how a one-variable-at-a-time sweep misleads: reach
#: 5 was measured "step-for-step identical, 35% dearer, buys nothing" and reverted -- a correct
#: measurement of the wrong pair. Under the swept scheme, vary these two together or not at all.
#:
#: The error reach 3 leaves *under the swept scheme* is ~2e-07 concentrated in the PRESSURE column,
#: which enters the residual only through gradients and so inherits the sweep-extended stencil
#: undiluted. Because a colouring is collision-free only for its own pattern, that is corruption of near
#: entries rather than truncation -- which is what the two-pass scheme has nothing of.
STENCIL_REACH = int(os.environ.get("PITZ_STENCIL_REACH", "3"))

#: Cap the gradient's Richardson sweeps FOR THE PROBE ONLY. The sweeps are what carry the stencil out
#: on a skewed mesh, so narrowing them shortens the reach the residual needs -- and only the
#: preconditioner's materialize sees the narrowed copy, the Krylov matvec keeping the exact jvp of the
#: full residual, so the converged state and its adjoint are untouched. `None` is byte-identical.
PROBE_GRADIENT_SWEEPS = (
    int(os.environ["PITZ_PROBE_GRADIENT_SWEEPS"])
    if os.environ.get("PITZ_PROBE_GRADIENT_SWEEPS")
    else None
)

#: Cap the gradient's Richardson sweeps in the copy of the residual the FORWARD JACOBIAN is
#: differentiated from -- the Krylov operator of every shifted solve. Not the residual: the march is
#: still driven to a root of the full-sweep discretization, and the adjoint still differentiates it,
#: so this moves neither the answer nor the gradient. `None` is byte-identical.
#:
#: Distinct from `PROBE_GRADIENT_SWEEPS`, which narrows what the preconditioner MATERIALIZES; this
#: narrows what the Krylov iteration APPLIES. Both approximate the Jacobian and neither touches R.
JACOBIAN_GRADIENT_SWEEPS = (
    int(os.environ["PITZ_JACOBIAN_SWEEPS"]) if os.environ.get("PITZ_JACOBIAN_SWEEPS") else None
)

#: The gradient reconstruction's own sweep count -- the RESIDUAL's, so this one is a change to the
#: discretization and moves the root. 4 is the scheme's shipped default; this mesh's calibrated count
#: at a 1e-4 L2 tolerance is 2 (`rho` = 5.07e-03), so the default is conservative here. Exposed so the
#: converged field's dependence on it can be measured rather than assumed. ⚠️ Raising it lengthens the
#: residual's stencil and needs `PITZ_STENCIL_REACH` raised to match, or the coloured probe folds the
#: far coupling onto near entries instead of capturing it.
GRADIENT_SWEEPS = int(os.environ.get("PITZ_GRADIENT_SWEEPS", "4"))

#: Which gradient reconstruction the case runs. `swept` (default) is the shipped corrected
#: Green-Gauss whose `A_g^-1` apply is a fixed number of Richardson sweeps; `multcorr` is the
#: two-face-pass scheme that reaches the same quadratic-exact contract with no solve.
#:
#: ⚠️ **The choice is coupled to `PITZ_STENCIL_REACH`, and that is the point of having it.** Each
#: Richardson sweep couples a cell one further ring, so the swept residual reaches `sweeps + 1` -- on
#: a randomly perturbed grid, measured reach 5 at the shipped four sweeps, against **3 with exactly
#: zero mass beyond distance 3** for `multcorr`, at every skewness tested. A probe at reach 3
#: therefore recovers the two-pass Jacobian exactly, where against the swept one it would fold far
#: couplings onto near entries. Vary the two together; `validation/gradient_stencil_reach.py`
#: re-measures both halves in about a minute.
GRADIENT = os.environ.get("PITZ_GRADIENT", "multcorr")
_GRADIENTS = {
    "swept": lambda: CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=GRADIENT_SWEEPS)),
    "multcorr": MultipleCorrectionGradient,
}
if GRADIENT not in _GRADIENTS:
    raise SystemExit(f"PITZ_GRADIENT={GRADIENT!r} is not one of {sorted(_GRADIENTS)}")

#: ⚠️ UNIFORM PROBING REACH, deliberately, where the sibling case shortens two columns. Its
#: `(3,3,3,3,2,2)` is a SIX-field layout and was measured on that mesh and those schemes; the analogous
#: five-field value here is unmeasured, and the record is emphatic that shortening the pressure column
#: diverged that case at step one. Uniform reach costs more probes and is always correct, so it is what
#: this case uses until someone measures the shortened one HERE.
COLUMN_REACH = None

#: A cost cap on the inner loop, so a doomed attempt is cut short rather than run to a stagnation.
CYCLE_BUDGET = 42

#: Grow the pseudo-timestep while the inner line search is comfortable; brake on a clipped step or a
#: rising residual. Without a control beta never ramps and the march cannot develop at all.
#:
#: `beta_start` is exposed because the usual intuition about it RUNS BACKWARDS ON THIS CASE. A larger
#: shift adds to the diagonal and normally buys conditioning; here it costs it. Measured on the
#: `[u, v, p]` block at the self-start, a zero-fill factorization under the reverse-Cuthill-McKee cell
#: order takes 140 Krylov applications at beta 0.5, **32** at 0.05 and 55 at 0. The retry ladder
#: escalates beta on a bad step, so a struggling march is driven the wrong way -- which is worth
#: knowing before reading an escalation as evidence about the preconditioner. Override per run so the
#: value lands in the run record; the default is unchanged.
BETA_START = float(os.environ.get("PITZ_BETA_START", "0.5"))

#: The starting shift for a **warm** rung -- one seeded by a converged root a Reynolds step below it,
#: rather than by the cold hybrid initialization the lowest rung starts from.
#:
#: One `beta_start` served both situations, which are not alike: the cold rung's seed is far from any
#: root and genuinely needs heavy damping, while a warm rung is handed a converged field one
#: continuation step away. Both then spend the same `ceil(ln(beta_start / beta_min) / ln(grow))` steps
#: walking the shift back down -- twelve of them at the values here -- before the march can take a full
#: pseudo-timestep, and that descent is a fixed cost per rung no matter how good the seed is.
#:
#: Lowering it is a bet with two ways to pay and one way to lose. It removes descent steps, and the
#: early steps it removes are the expensive ones (the same self-start measurement quoted above: 140
#: Krylov applications at beta 0.5 against 32 at 0.05). Against that, a shift that starts below what the
#: rung can take is caught by the retry ladder rather than by a divergence -- but a caught step costs
#: two to six ordinary ones, so enough of them would give the saving back.
#:
#: Defaults to `BETA_START`, so an unset environment reproduces the single-constant behaviour exactly.
BETA_START_WARM = float(os.environ.get("PITZ_BETA_START_WARM", str(BETA_START)))


#: Re-impose the near-wall `omega` the residual fixes at each WARM Reynolds rung
#: (`PITZ_SEED_REPAIR=wall`). Those rows are not solved -- they are a value fixation at
#: `omega_wall(nu, d, k)`, whose viscous branch is linear in the molecular viscosity -- so a rung
#: handed the previous rung's converged root holds, in those cells, a number the model itself says is
#: wrong, by the viscosity ratio exactly. Measured at this case's rung 2 -> 3 handover: `|R0|`
#: 1.1746e-01 -> 3.0587e-02 (3.84x), every other block unchanged to four significant figures in FIXED
#: scales. Costs one elementwise evaluation.
#:
#: ⚠️ **Default off, and it is a CORRECTNESS fix rather than a performance one.** Marched, the better
#: seed buys nothing -- 14 steps on rung 2 with it or without -- and it is not what carries an
#: aggressive warm shift through the target rung either. Enable it because seeding a known-wrong
#: imposed value is wrong, not because it is expected to be faster.
SEED_REPAIR = os.environ.get("PITZ_SEED_REPAIR", "off")


def _seed_projection(companion, state, point):
    """Repair a WARM rung's seed; leave the cold rung's alone.

    What is corrected is a property of the *handover*: a rung inherits a root converged at the previous
    viscosity, and the wall-`omega` fixation the residual imposes is computed from that viscosity. The
    lowest rung inherits nothing -- it starts from `hybrid_initialize`, which already seeds this closure
    itself -- so there is nothing there to correct, and leaving it alone keeps rung 1 bit-identical
    across arms, which is what makes it a control rather than an arm.
    """
    if point.index == 1 or SEED_REPAIR != "wall":
        return state
    return wall_consistent_state(companion, state)


def dual_time_control(beta_start: float) -> CflResidualDualTimeControl:
    """The case's dual-time control at a chosen starting shift.

    One construction site for the control's settings, so the cold and warm rungs differ in the single
    value that is meant to differ between them and cannot drift apart in the rest.

    Parameters
    ----------
    beta_start : float
        The shift strength the rung's first step runs at, before the control adapts it.
    """
    return CflResidualDualTimeControl(
        beta_start=beta_start,
        beta_min=BETA_MIN,
        grow=1.5,
        backoff=2.0,
        grow_above=0.5,
        backoff_below=0.25,
    )


CONTROL = dual_time_control(BETA_START)

#: ⚠️ REFRESH THE FROZEN PRECONDITIONER, ON SOLVE COST, EXACTLY AS THE THREE-DIMENSIONAL CASE DOES.
#: Frozen at the cold reference state for a whole march, the preconditioner goes stale precisely as the
#: recirculation forms -- which is the one thing a cold march is for. The symptom is unmistakable once
#: known, and was observed here before this was wired: the solve cost climbs while the step length
#: stays healthy, the control lowers the shift, the now-expensive solve trips the retry ladder, the
#: ladder puts the shift back, and the march enters a limit cycle with the residual flat. Five steps at
#: 14-18 cycles a solve moved the residual from 3.404e-03 to 3.441e-03 -- backwards.
#:
#: The trigger is the COST itself: a solve that reaches this many restart cycles rebuilds the
#: preconditioner at the iterate it was handed, and the inner loop carries on rather than the step
#: being discarded. Capped at one refresh per step. Reacting to cost rather than predicting staleness
#: is deliberate: a diagnostic on the sibling case refuted every cheap STATIC predictor of a bad step,
#: so detect-then-react is the honest design.
REFRESH_ON_CYCLES = int(os.environ.get("PITZ_REFRESH_ON_CYCLES", "3"))

#: How many per-step state checkpoints to keep, when `solve_aquaflux` is given a `checkpoint_dir`.
#: A rolling few is enough for the usual purpose -- recovering the CONVERGED state, which no other
#: artifact of a run preserves. Raise it to keep a whole trajectory (~0.5 MB per step here) when a
#: study needs the march's own intermediate states rather than its endpoint.
CHECKPOINT_KEEP = int(os.environ.get("PITZ_CHECKPOINT_KEEP", "3"))

#: Redo a step whose solve was expensive, whose line search collapsed, or that diverged -- escalating
#: the shift first, and falling back to a tighter Krylov solve only for a divergence damping cannot fix.
#:
#: ⚠️ **`abort_above_cycles` IS A COST BUDGET, NOT A DIAGNOSIS — set it ABOVE what the installed
#: preconditioner costs when it is healthy.** Crossing it stops the dual-time inner loop and redoes the
#: step at the SAME shift on the refreshed preconditioner; it no longer escalates (that is `on_alpha`'s
#: job alone). Set too low it truncates convergence a step is in the middle of achieving: with the
#: zero-fill smoother under the reverse-Cuthill-McKee cell order, whose healthy cost at the second
#: Reynolds rung is about 12, a threshold of 10 cut step 29 off after three inner iterations where a
#: fourth would have accepted it -- and, under the previous design where the same number also escalated,
#: took beta 0.5 -> 1.0 -> 1.33 -> 2.67 before the step diverged. It must also stay strictly below
#: `CYCLE_BUDGET`, which truncates a grinding solve and relies on this redo to discard the partial
#: iterate.
#: ⚠️ The default is UNCHANGED at 10, which suits the shipped `petsc` arm. The zero-fill `hostilu`
#: arm wants ~25; it is set per run rather than here, because raising it for every arm would change
#: the incumbent's behaviour on evidence that was never gathered for it.
RETRY_ON_CYCLES = int(os.environ.get("PITZ_RETRY_ON_CYCLES", "10"))
RETRY = RetryPolicy(
    solver=relative_residual_gmres(1e-4, restart=40),
    abort_above_cycles=RETRY_ON_CYCLES,
    on_alpha=0.01,
    beta_factor=2.0,
)


# --- OpenFOAM ascii internalField parsing (nonuniform scalar / vector list) ---
def _of_scalar(path):
    t = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(", t)
    body = t[m.end() :]
    return np.array(re.findall(r"[-+]?\d[\d.eE+-]*", body)[: int(m.group(1))], dtype=float)


def _of_vector(path):
    t = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n\(", t)
    body = t[m.end() :]
    tri = re.findall(r"\(([^)]*)\)", body)[: int(m.group(1))]
    return np.array([[float(v) for v in s.split()] for s in tri], dtype=float)


def read_openfoam_reference():
    """The OpenFOAM comparison fields + cell centres, keyed to their cell centroids.

    The **fields** come from the time-accurate run (:data:`TRANSIENT`), which reaches a statistically
    steady state with a well-defined reattachment (``x_r/h`` 7.74); the steady case's fields are not a
    valid target and are deliberately not read here (see the comment on :data:`TRANSIENT`). The **cell
    centres** still come from the steady run's ``Ccx``/``Ccy``: both cases are built from the same
    ``blockMeshDict``, so the mesh is byte-identical and the centres are geometry rather than a
    solution -- the transient case ships its written fields but no ``Cc*``.

    Returns
    -------
    dict
        ``centroid`` ``(n_cells, 2)``, ``U`` ``(n_cells, 2)``, and ``p``, ``k``, ``omega``, ``nut``
        each ``(n_cells,)``, in the mesh's own cell order.
    """
    ccx, ccy = _of_scalar(RUNS / "Ccx"), _of_scalar(RUNS / "Ccy")
    return dict(
        centroid=np.column_stack([ccx, ccy]),
        U=_of_vector(TRANSIENT / "U")[:, :2],
        p=_of_scalar(TRANSIENT / "p"),
        k=_of_scalar(TRANSIENT / "k"),
        omega=_of_scalar(TRANSIENT / "omega"),
        nut=_of_scalar(TRANSIENT / "nut"),
    )


def build_case(model=None, gradient_scheme=None):
    """Assemble the benchmark: mesh, momentum, turbulence and the coupled residual -- no solve.

    Split out from :func:`solve_aquaflux` so a solver study can re-solve at a saved state (a
    mid-march checkpoint, say) without re-marching to it, and without restating the case. The
    mesh import, boundary conditions, model constants and scheme choices *are* the definition of
    this benchmark; a second copy of them would drift from the one the validation figures use.

    Parameters
    ----------
    model : SSTModel, optional
        The SST constants to use. Defaults to :class:`~aquaflux.turbulence.SSTModel`. Passing a model
        that differs only in the near-wall omega blend (``wall_omega_exponent`` /
        ``wall_omega_viscous_coeff``) is how a wall-treatment study compares blend shapes on the same
        case -- e.g. a large exponent to reproduce the ``max(omega_vis, omega_log)`` blend.
    gradient_scheme : GradientScheme, optional
        The gradient reconstruction, used by momentum and turbulence alike. Defaults to whatever
        ``GRADIENT`` selects -- :class:`~aquaflux.schemes.MultipleCorrectionGradient`, described below.
        Injected so a scheme study compares reconstructions on this exact case rather than on a second
        copy of it. ⚠️ A scheme whose residual reaches further across the cell graph than
        ``stencil_reach`` needs that raised to match: the coloured probe folds coupling beyond its
        reach onto near entries instead of dropping it, so an under-reaching probe corrupts the
        preconditioner rather than approximating it. The default reconstruction carries *exactly zero*
        mass beyond reach 3 and ``CorrectedGreenGauss`` beyond ``sweeps + 1``, which is what makes each
        of those pairings exact;
        :class:`~aquaflux.schemes.HessianCorrectedGradient` does not have that cut-off at any sweep
        setting, so a study that swaps it in needs a preconditioner that is not built by probing --
        which is why that comparison lives in the sibling ``pitzdaily_gradient_ab`` case rather than
        here, and why this benchmark's own configuration is untouched by it.

    Returns
    -------
    dict
        ``coupled``, ``momentum``, ``turbulence`` and ``geom`` for the assembled case.
    """
    if model is None:
        model = SSTModel()
    mesh = read_openfoam(RUNS / "polyMesh")
    geom = mesh.geometry()
    # Corrected (non-orthogonal / skewness) Green-Gauss gradients. Its A_g^-1 apply is the default O(n)
    # matrix-free swept solve (fixed Richardson sweeps), not a nested GMRES: identical discretization,
    # but it avoids a nested Krylov solve (carrying its own implicit-diff tangent) inside every
    # coupled-residual evaluation, which otherwise dominates the monolithic Newton cost on this
    # ~12k-cell mesh (measured ~180x per residual eval here). The default sweep count is used: this mesh
    # is only mildly non-orthogonal (worst face angle ~6 degrees), so the swept solve reaches the
    # converged corrected-gradient to machine precision in the default few sweeps -- and the
    # reconstructed gradient, the coupled residual, and the reattachment length are all unchanged from a
    # much higher sweep count, so paying for more sweeps only enlarges the differentiated residual.
    grad = _GRADIENTS[GRADIENT]() if gradient_scheme is None else gradient_scheme
    # Momentum advection: second-order upwind = Venkatakrishnan-limited linear upwind (the upwind cell
    # reconstructed to the face with its corrected-Green-Gauss gradient, slope-limited so the
    # reconstruction is monotonicity-bounded) -- the analogue of OpenFOAM's `Gauss linearUpwind`.
    momentum_upwind = LimitedUpwind(limiter=VenkatakrishnanLimiter())
    # Turbulence advection: first-order upwind on k and omega. The slope limiter bounds the advective
    # *face value*, but the negative-omega failure of second-order on the stiff omega equation is a
    # Newton-*update* overshoot at the cell centre, not a face-value one: first-order upwind makes the
    # omega transport operator diagonally dominant (an M-matrix) so the pseudo-transient-shifted Newton
    # step preserves positivity, whereas a second-order stencil -- even limited -- weakens that
    # dominance and lets the update drive omega < 0 (then nu_t = k/omega flips sign and poisons the
    # closure while the residual stays finite, so the divergence guard never trips). The structural fix
    # for second-order scalars is log-variable transport (omega = e^w), which is not built here.
    scalar_upwind = FirstOrderUpwind()
    momentum = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(jnp.asarray(RHO * NU)), "density": Constant(RHO)}),
        grad,
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=(U_IN, 0.0)),
                "outlet": PressureOutlet(pressure=0.0),
                "upperWall": NoSlipWall(),
                "lowerWall": NoSlipWall(),
            }
        ),
        advection_scheme=momentum_upwind,
    )
    # ⚠️ MATCHES THE SIBLING CASE'S LINEARIZATION. With the limiter left implicit the Jacobian
    # carries the k-production cap's own derivative, which is destabilizing; freezing it is the
    # Patankar treatment the sibling case runs. The library default is False, so omitting this
    # silently gave the two cases DIFFERENT Newton operators on the same physics.
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geom,
        grad,
        scalar_upwind,
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=WALLS,
        explicit_production_limiter=True,
        k_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(K_IN),
                "outlet": ZeroGradient(),
                "upperWall": K_WALL_BC,
                "lowerWall": K_WALL_BC,
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(OMEGA_IN),
                "outlet": ZeroGradient(),
                "upperWall": ZeroGradient(),
                "lowerWall": ZeroGradient(),
            }
        ),
    )
    # Log-transform omega: omega = e^w stays strictly positive under any Newton step. On this stiff
    # separating case a direct-omega step drives omega negative once the recirculation forms (nu_t =
    # k/omega then flips sign and poisons the closure while the residual stays finite, so the divergence
    # guard never trips). k stays direct -- log(k) is ill-conditioned where k -> 0 at the walls.
    coupled = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    # The monolithic Newton is globalized by the default pseudo-transient continuation: an a_P /
    # transport-diagonal shift that damps each step heavily far from the fixed point and ramps to
    # zero on the residual, recovering the exact steady Newton state at convergence.
    return dict(coupled=coupled, momentum=momentum, turbulence=turbulence, geom=geom)


def _gradient_scheme_label(scheme):
    """A one-line description of the reconstruction actually in force, for the run banner.

    Names the sweep count as well as the class, because the count is what sets both the accuracy of
    the reconstruction and how far the residual reaches across the cell graph -- so a banner naming
    only the class cannot be read against a recorded measurement.
    """
    if scheme is None:
        scheme = _GRADIENTS[GRADIENT]()
    solver = getattr(scheme, "solver", None)
    sweeps = getattr(solver, "sweeps", None)
    return f"{type(scheme).__name__}" + (f" (swept {sweeps})" if sweeps is not None else "")


def solve_aquaflux(
    *,
    log_path=None,
    checkpoint_dir=None,
    gradient_scheme=None,
    stencil_reach=None,
    jacobian_gradient_sweeps=None,
    **solve_kwargs,
):
    """Solve the coupled RANS system on the imported OpenFOAM mesh; return fields + geometry.

    Parameters
    ----------
    log_path : str or Path, optional
        Write the per-step march log here instead of to stdout.
    checkpoint_dir : str or Path, optional
        Write a rolling state checkpoint per step here. Worth setting for any long run: without it a
        converged state exists only inside the process that computed it, so any later study of the
        converged operator -- the adjoint's, in particular, which is the one the march itself never
        exercises -- has to re-run the whole march to ask its question.
    gradient_scheme : GradientScheme, optional
        Overrides :data:`GRADIENT_SWEEPS` for this solve; see :func:`build_case`. ``None`` takes the
        case's own scheme.
    stencil_reach : int, optional
        Overrides :data:`STENCIL_REACH` for this solve. A study that varies the residual's own sweep
        count has to move this with it -- the sweeps are what carry the residual's stencil, and a
        probe shorter than that stencil folds the far coupling onto near entries.
    jacobian_gradient_sweeps : int, optional
        Overrides :data:`JACOBIAN_GRADIENT_SWEEPS` for this solve.
    **solve_kwargs
        Forwarded to :func:`~aquaflux.turbulence.solve_coupled`, overriding the defaults set here.
        This is the seam a solver study uses to instrument or reconfigure the march -- an ``on_step``
        observer, a ``refresh_trigger``, a different ``method``.

    Notes
    -----
    The three overrides above exist so a sweep can run every arm **in one process**, back to back on
    one machine, rather than as N invocations whose wall clocks are not comparable. They default to the
    module constants, so an unparameterized call is the case exactly as it ships.
    """
    if stencil_reach is None:
        stencil_reach = STENCIL_REACH
    if jacobian_gradient_sweeps is None:
        jacobian_gradient_sweeps = JACOBIAN_GRADIENT_SWEEPS
    case = build_case(gradient_scheme=gradient_scheme)
    coupled, momentum, turbulence, geom = (
        case["coupled"],
        case["momentum"],
        case["turbulence"],
        case["geom"],
    )
    # One line per outer step, flushed, to `log_path` or stdout. A march nobody can read until it
    # finishes costs its whole wall time to tell you something it knew in the third minute -- and a
    # crawling march is indistinguishable from a hung one without it.
    log_file = open(log_path, "w") if log_path is not None else sys.stdout
    logger = MarchLogger(
        log_file,
        fields=coupled_fields(coupled),
        detail=("inner", "fields", "pc"),
        rtol=RTOL,
        atol=ATOL,
    )
    # Every run states the configuration it was taken under, in its own log, before any result: a
    # number whose configuration is not written beside it cannot be re-adjudicated later, which is
    # worse than being wrong, because a wrong finding gets corrected and an unanchored one gets cited.
    logger.note("[configuration]")
    for _name, _value in (
        ("dual-time inner steps / tol", f"{INNER_STEPS} / {INNER_TOL}"),
        ("k positivity floor", K_POSITIVITY_FLOOR),
        ("inner forward rtol (row-scaled) / restart", f"{FORWARD_RTOL} / {FORWARD_RESTART}"),
        ("cycle budget", CYCLE_BUDGET),
        ("retry on cycles / alpha", f"{RETRY.abort_above_cycles} / {RETRY.on_alpha}"),
        (
            "step control",
            f"{type(CONTROL).__name__} (beta_start {BETA_START} cold / {BETA_START_WARM} warm)",
        ),
        ("seed repair (warm rungs)", SEED_REPAIR),
        (
            "Reynolds continuation points",
            f"{N_POINTS} (ratio {RATIO:g}, anchor Re/{RATIO**N_POINTS:g})",
        ),
        # ⚠️ HOW THE SPAN ABOVE IS WALKED, because the two arms read identically without it. The ramp
        # arm reports `[point 1/1 ...]` and otherwise logs nothing about its own schedule, so a run's
        # stations, station length and re-damping were not recoverable from its log -- and a schedule
        # comparison whose arms cannot be told apart afterwards is not a measurement.
        (
            "viscosity ramp",
            # ⚠️ `RAMP_REDAMPING` is None unless the environment sets it -- the homotopy derives the
            # value -- so this must not format it as a number. It did, and the case could not start at
            # its OWN DEFAULT: every arm of the schedule sweep set the variable explicitly, so the one
            # configuration nobody passed was the one nobody ran.
            f"{RAMP_STATIONS} stations x {RAMP_STEPS_PER_STATION} steps "
            f"({RAMP_STATIONS * RAMP_STEPS_PER_STATION} ramp steps), redamping "
            f"{'derived' if RAMP_REDAMPING is None else format(RAMP_REDAMPING, 'g')}"
            f", scaling {RAMP_SCALE}"
            f", turbulence damping {TURB_DAMPING:g}"
            f"{_TURB_DAMPING_SHAPE}"
            if RAMP == "continuous"
            else f"off ({RAMP}) -- the span is walked as a rung ladder",
        ),
        ("k wall BC", K_WALL),
        ("preconditioner refresh", f"on {REFRESH_ON_CYCLES} restart cycles (mid-step)"),
        (
            "smoother fill / sweeps / coarse limit",
            f"{FILL_LEVELS} / {SWEEPS} / {COARSE_EQ_LIMIT}"
            + ("" if _ILU_SMOOTHER_LIVE else "  (INERT: both blocks supply their own inverse)"),
        ),
        ("preconditioner beta floor", PC_BETA_FLOOR),
        ("field split / trailing sweeps", f"{FIELD_SPLIT} / {TRAILING_SWEEPS}"),
        (
            "flow inverse",
            FLOW_INVERSE
            if LEADING_INVERSE is None
            # The ordering object prints as a bare repr, which says nothing; name the order instead --
            # a banner line that cannot be read against a recorded measurement is not worth printing.
            else f"{FLOW_INVERSE} {HOST_FLOW | {'ordering': f'cell-major/{FLOW_ORDER}'}}"
            if FLOW_INVERSE == "hostilu"
            else f"{FLOW_INVERSE} {SIMPLE_FLOW}",
        ),
        # ⚠️ WHICH incomplete-LU IMPLEMENTATION IS LIVE, because the two differ by orders of magnitude
        # in speed and nothing recorded which one a run used. `Ilu0` ships a pure-Python reference twin
        # of its compiled kernel and falls back to it silently when the extension is not built; the two
        # compute the identical factorization (pinned by a unit test), so a CONVERGENCE result is the
        # same either way, but a WALL-CLOCK one taken on the fallback is not a preconditioner
        # measurement at all. Printing it is what makes such a number falsifiable later.
        (
            "host ILU kernel",
            "compiled" if ILU0_COMPILED else "PURE PYTHON (fallback -- timings void)",
        ),
        ("trailing inverse", f"jacobi_smoothed {JACOBI_TRAILING}" if FIELD_SPLIT else "n/a"),
        ("probe stencil reach", stencil_reach),
        ("gradient scheme", _gradient_scheme_label(gradient_scheme)),
        ("probe gradient sweeps", PROBE_GRADIENT_SWEEPS or "full (the scheme's own)"),
        (
            "JACOBIAN gradient sweeps",
            jacobian_gradient_sweeps or "full -- the exact Jacobian of the residual",
        ),
        ("probe column reach", COLUMN_REACH or "uniform"),
        ("forward restart / max restarts", f"{FORWARD_RESTART} / {FORWARD_MAX_RESTARTS}"),
        ("k positivity projection", POSITIVITY_PROJECTION),
        ("stop (rtol, atol)", f"{RTOL}, {ATOL}"),
    ):
        logger.note(f"  {_name}: {_value}")

    # The refresh hook, built ONCE and pointed at each rung in turn. Its scheduled cadences are
    # switched OFF so the cycle trigger REPLACES them rather than adding to them: as an addition the
    # trigger was measured break-even on the sibling case, as a replacement it was the largest saving
    # on that march. ⚠️ `beta_rel_change=None` does NOT switch the schedule off -- it removes the gate,
    # and a missing gate means "refresh every step". Off means a gate that exists and never fires.
    # Built once and shared by the engine and the refresh hook: the coloured-probe plan is the single
    # largest allocation this case makes, and building it twice doubles that for nothing.
    probe = CoupledJacobianProbe.build(
        coupled,
        stencil_reach=stencil_reach,
        column_reach=COLUMN_REACH,
        gradient_sweeps=PROBE_GRADIENT_SWEEPS,
    )
    refresh = amg_beta_tracking_refresh(
        coupled,
        probe=probe,
        beta_rel_change=float("inf"),
        refresh_every=10**9,
        materialize_drift=None,
        materialize_every=None,
        beta_floor=PC_BETA_FLOOR,
        observer=logger.on_refresh,
    )
    #: One preconditioner shared across rungs, handed back for the next: only the viscosity changes
    #: between them, so a rung needs the V-cycle FITTED to it, not a fresh object.
    shared_preconditioner: list = []

    def _damping(companion, seed_state, beta_start):
        """Constant, or tapered from `TURB_DAMPING` down to 1 on the chosen key.

        The `beta` key's endpoints are the rung's OWN control endpoints, taken from the same values the
        control is built from a few lines below: a taper whose span disagreed with the control's would
        still run, and would reach 1 somewhere the march never visits, with nothing to detect it. The
        `residual` key's reference is the closure residual at the state this rung OPENS from, so the
        taper measures progress from where the march actually starts rather than from a state it never
        visits.
        """
        if not TURB_TAPER:
            return TURB_DAMPING
        if TURB_TAPER_KEY == "beta":
            return BetaTaperedDamping(
                initial=TURB_DAMPING,
                beta_start=beta_start,
                beta_min=BETA_MIN,
                exponent=TURB_TAPER,
            )
        return ResidualTaperedDamping(
            initial=TURB_DAMPING,
            reference=turbulence_residual_norm(companion.layout, companion.residual(seed_state)),
            layout=companion.layout,
            exponent=TURB_TAPER,
        )

    def point_setup(companion, seed_state, point):
        """Configure each Reynolds rung, re-fitting the one preconditioner to it.

        Only the molecular viscosity changes between rungs, so a rung needs its own residual assembler
        and its own row scales -- both ordinary data -- but not its own V-cycle. It needs that V-cycle
        *fitted to it*, which is what `rebind` arranges, and which is a different thing from rebuilding
        the object (that would only cost a compilation).
        """
        logger.note(f"[{point.label}]")
        refresh.rebind(companion)
        beta_start = BETA_START if point.index == 1 else BETA_START_WARM
        engine = coupled_amg_continuation(
            companion,
            seed_state,
            turbulence_damping=_damping(companion, seed_state, beta_start),
            inner_steps=INNER_STEPS,
            inner_tol=INNER_TOL,
            probe=probe,
            probe_gradient_sweeps=PROBE_GRADIENT_SWEEPS,
            jacobian_gradient_sweeps=jacobian_gradient_sweeps,
            cycle_budget=CYCLE_BUDGET,
            forward_rtol=FORWARD_RTOL,
            forward_restart=FORWARD_RESTART,
            forward_max_restarts=FORWARD_MAX_RESTARTS,
            refresh_on_cycles=REFRESH_ON_CYCLES or None,
            inner_refresh=refresh.refresh_at if REFRESH_ON_CYCLES else None,
            positivity_floor=K_POSITIVITY_FLOOR,
            positivity_projection=POSITIVITY_PROJECTION,
            preconditioner=shared_preconditioner[0] if shared_preconditioner else None,
            smoother_fill_levels=FILL_LEVELS,
            smoother_sweeps=SWEEPS,
            coarse_eq_limit=COARSE_EQ_LIMIT,
            field_split=FIELD_SPLIT,
            trailing_smoother_sweeps=TRAILING_SWEEPS,
            leading_inverse=LEADING_INVERSE if FIELD_SPLIT else None,
            trailing_inverse=jacobi_smoothed_inverse(**JACOBI_TRAILING) if FIELD_SPLIT else None,
            inner_observer=logger.on_inner,
        )
        shared_preconditioner[:] = [engine.shift_policy.preconditioner]
        # The lowest rung (`index == 1`) is the one that self-starts from the hybrid initialization;
        # every rung above it is handed the converged root below it. They get their own starting shift
        # for that reason -- see `BETA_START_WARM`. With the environment unset the two are equal and
        # this is the same control the solve would have used anyway.
        return dict(
            continuation=engine,
            refresh=RefreshPolicy(precondition_step=refresh),
            step_control=dual_time_control(beta_start),
        )

    checkpoints = (
        StateCheckpointer(checkpoint_dir, every=1, keep=CHECKPOINT_KEEP)
        if checkpoint_dir is not None
        else None
    )

    def station_damping(step, station, arrived):
        """Damp the closure's rows harder while the viscosity ramp is walking than at the target.

        The ramp and the target station want opposite ratios, and by a wide margin (see
        `TURB_DAMPING_TARGET`). Nothing the shift policy can read for itself separates them -- every
        such signal measures march progress, which saturates inside the ramp -- so the station index
        arrives from the march, which is the only thing that knows it.

        The swap is `eqx.tree_at` over one ARRAY leaf, so each station is a compilation-cache hit
        rather than a recompile of the whole coupled solve.
        """
        del station
        ratio = TURB_DAMPING_TARGET if arrived else TURB_DAMPING
        return eqx.tree_at(
            lambda s: s.shift_policy.base.turbulence_damping, step, ConstantDamping(ratio)
        )

    solve_options = (
        dict(
            max_steps=MAX_STEPS,
            # `None` unless a target ratio was asked for, which keeps a single-ratio march unchanged.
            station_step=station_damping if TURB_DAMPING_TARGET else None,
            rtol=RTOL,
            atol=ATOL,
            intermediate_rtol=None,  # every rung stops at the same ABSOLUTE bar
            intermediate_atol=ATOL,
            schedule=GeometricReynoldsSchedule(ratio=RATIO),
            step_control=CONTROL,
            retry=RETRY,
            point_setup=point_setup,
            seed_projection=_seed_projection if SEED_REPAIR != "off" else None,
            scaled_norm=True,  # rebuild the row scales each outer step
            on_checkpoint=(
                logger.on_checkpoint
                if checkpoints is None
                else combine_observers(logger.on_checkpoint, checkpoints.on_checkpoint)
            ),
            on_retry=logger.on_retry,
        )
        | solve_kwargs
    )
    try:
        if RAMP == "continuous":
            # The SAME viscosity span the ladder walks (its anchor sits `RATIO ** N_POINTS` below the
            # target) and the SAME `point_setup` and options, so the two arms differ in how the span is
            # walked and in nothing else. That is what makes them comparable.
            flow, k, omega = solve_reynolds_ramp(
                coupled,
                anchor=RATIO**N_POINTS,
                stations=RAMP_STATIONS,
                steps_per_station=RAMP_STEPS_PER_STATION,
                redamping=RAMP_REDAMPING,
                companion=RAMP_COMPANION,
                **solve_options,
            )
        else:
            flow, k, omega = solve_reynolds_continuation(coupled, N_POINTS, **solve_options)
    finally:
        if log_file is not sys.stdout:
            log_file.close()
    velocity, pressure = momentum.unpack(flow)
    nu_t = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
    return dict(
        centroid=np.asarray(geom.cell.centroid),
        U=np.asarray(velocity),
        p=np.asarray(pressure),
        k=np.asarray(k),
        omega=np.asarray(omega),
        nut=np.asarray(nu_t),
    )


def reattachment_length(centroid, u_x):
    """Lower-wall reattachment length x_r/h behind the step (h = step height = 0.0254 m).

    Reads the sign of the wall-adjacent streamwise velocity along the lower wall downstream of the
    step: the recirculation bubble is where it is negative, and reattachment is the last such x.
    """
    h = 0.0254
    x, y = centroid[:, 0], centroid[:, 1]
    # The wall-adjacent row along the lower wall (y just above the -h floor), downstream of the step.
    band = (x > 1e-4) & (y < -h + 0.002) & (y > -h)
    xs = x[band]
    us = u_x[band]
    order = np.argsort(xs)
    xs, us = xs[order], us[order]
    neg = np.where(us < 0)[0]
    if neg.size == 0:
        return 0.0
    return float(xs[neg[-1]] / h)


def main():
    if not (RUNS / "U").exists():
        raise SystemExit(f"OpenFOAM results not found in {RUNS}; run of_case/run_of.sh first.")
    of = read_openfoam_reference()
    print(
        f"OpenFOAM: {of['centroid'].shape[0]} cells, Ux in "
        f"[{of['U'][:, 0].min():.3f}, {of['U'][:, 0].max():.3f}]",
        flush=True,
    )

    t0 = time.time()
    aq = solve_aquaflux(checkpoint_dir=HERE / "checkpoints")
    print(
        f"aquaflux coupled solve: {time.time() - t0:.0f}s, "
        f"Ux in [{aq['U'][:, 0].min():.3f}, {aq['U'][:, 0].max():.3f}]",
        flush=True,
    )

    from scipy.spatial import cKDTree

    tree = cKDTree(of["centroid"])
    dist, idx = tree.query(aq["centroid"])
    assert float(dist.max()) < 1e-6, f"mesh mismatch: max centroid distance {dist.max()}"

    def rel_l2(a, b, scale):
        return float(np.sqrt(np.mean((a - b) ** 2)) / scale)

    metrics = dict(
        ux=rel_l2(aq["U"][:, 0], of["U"][idx, 0], U_IN),
        uy=rel_l2(aq["U"][:, 1], of["U"][idx, 1], U_IN),
        xr_aqua=reattachment_length(aq["centroid"], aq["U"][:, 0]),
        xr_of=reattachment_length(of["centroid"], of["U"][:, 0]),
        nut_peak_aqua=float(aq["nut"].max() / NU),
        nut_peak_of=float(of["nut"].max() / NU),
    )
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}", flush=True)

    _figure(of, aq, idx)
    _report(metrics)


def _figure(of, aq, idx):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    x, y = aq["centroid"][:, 0], aq["centroid"][:, 1]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    vmax = max(abs(of["U"][:, 0]).max(), abs(aq["U"][:, 0]).max())
    for a, data, title in (
        (ax[0], of["U"][idx, 0], "OpenFOAM $U_x$"),
        (ax[1], aq["U"][:, 0], "aquaflux $U_x$"),
    ):
        sc = a.scatter(x, y, c=data, s=3, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(title)
        a.set_aspect("equal")
        fig.colorbar(sc, ax=a, shrink=0.8)
    # Reattachment: lower-wall streamwise velocity sign.
    h = 0.0254
    band = (x > 1e-4) & (y < -h + 0.002) & (y > -h)
    order = np.argsort(x[band])
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].plot(
        x[band][order] / h,
        of["U"][idx][band][order][:, 0],
        "s-",
        ms=3,
        color="C1",
        label="OpenFOAM",
    )
    ax[2].plot(
        x[band][order] / h, aq["U"][band][order][:, 0], ".-", ms=4, color="C0", label="aquaflux"
    )
    ax[2].set_xlabel("$x/h$ behind the step")
    ax[2].set_ylabel("near-wall $U_x$")
    ax[2].set_title("Lower-wall recirculation (sign change = reattachment)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "comparison.png", dpi=130)
    print(f"wrote {FIGS / 'comparison.png'}", flush=True)


def _report(m):
    lines = [
        "# pitzDaily backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST",
        "",
        "The OpenFOAM `pitzDailySteady` tutorial -- its RAS model switched from the shipped `kEpsilon`",
        "to `kOmegaSST` -- run in OpenFOAM, then solved on the **same imported mesh** by aquaflux's",
        "coupled RANS solver (hybrid initialization, second-order upwind momentum advection, corrected",
        "Green-Gauss gradients). U_in = 10 m/s, nu = 1e-5 (Re ~ 25000 on the 25.4 mm inlet).",
        "",
        "## Results",
        "",
        "| quantity | aquaflux | OpenFOAM |",
        "|---|---|---|",
        f"| reattachment length x_r/h (lower wall) | {m['xr_aqua']:.2f} | {m['xr_of']:.2f} |",
        f"| peak nu_t/nu | {m['nut_peak_aqua']:.0f} | {m['nut_peak_of']:.0f} |",
        f"| rel. L2 U_x error (cell-for-cell) | {m['ux']:.3f} | -- |",
        f"| rel. L2 U_y error (cell-for-cell) | {m['uy']:.3f} | -- |",
        "",
        "See `figures/comparison.png`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "# 1. OpenFOAM kOmegaSST reference (needs the openfoam13 image) -> runs/kwsst/",
        "cd validation/pitzdaily_openfoam",
        'docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh',
        "# 2. aquaflux coupled solve + comparison (from the repo root)",
        "cd ../..",
        "python3 validation/pitzdaily_openfoam/compare.py",
        "```",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
