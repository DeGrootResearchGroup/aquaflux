"""Finite-width 3D backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST.

The first **3D** same-mesh, cell-for-cell cross-code validation, the three-dimensional sibling of
``validation/pitzdaily_openfoam``. A backward-facing step of height ``h = 0.01 m`` (expansion ratio 2)
spanning ``4h`` between two **no-slip side walls** -- the finite span with viscous side walls makes the
flow genuinely three-dimensional (corner/secondary flow, spanwise-varying reattachment), not a 2D
extrusion. ``of_case/run_of.sh`` runs the OpenFOAM ``incompressibleFluid`` steady SIMPLE solver with
``kOmegaSST`` and writes the fields, the mesh and the 3D cell centres to ``runs/kwsst/``;
``of_transient/run_transient.sh`` runs a time-accurate reference to a statistically-steady state
(``runs/kwsst_transient/``). This script reads that **same mesh** into aquaflux via ``read_openfoam``
and solves the coupled RANS system on it, so the two solutions live on identical cells and are compared
directly (no interpolation between independent meshes).

aquaflux setup, mirroring the 2D case:

* the **coupled** turbulent solver (:func:`aquaflux.turbulence.solve_coupled` -- one monolithic Newton
  on ``R(u, p, k, omega)``, globalized by pseudo-transient continuation);
* **hybrid initialization** (potential-flow velocity + Laplace-smoothed turbulence), which self-starts
  the monolithic Newton;
* **second-order upwind** momentum advection (:class:`aquaflux.discretization.LimitedUpwind` with the
  :class:`aquaflux.schemes.VenkatakrishnanLimiter`); the stiff k/omega scalars use bounded first-order
  upwind;
* **corrected Green-Gauss** gradients (:class:`aquaflux.schemes.CorrectedGreenGauss`);
* **log-variable omega** (:class:`aquaflux.turbulence.LogScalars` on ``omega_transform``): ``omega =
  e^w`` stays strictly positive under any Newton step, the fix for a direct-omega step driving omega
  negative once recirculation forms.

**Preconditioner (the point of a 3D case):** the coupled Jacobian is preconditioned by an
**algebraic-multigrid V-cycle** (:func:`aquaflux.turbulence.coupled_amg_continuation`), not by a
factorization. The complete LU is exact but its fill is a memory wall in 3D (``O(n^{4/3})``), and even the
threshold-ILU's factorization of the distance-3 3D coupled Jacobian (hundreds of nonzeros per row) is
prohibitively slow to build; the V-cycle keeps the heavy fill on only the small coarsest grid (a direct-LU
coarse solve), so its memory stays bounded and its setup is seconds. This 3D case is the first exercise of
that scaling path.

The V-cycle is **field-split** (``field_split=True``): the ``[u, v, w, p]`` saddle and the ``[k, omega]``
transported scalars get their own hierarchies, with one triangle of the coupling between them retained
exactly from the assembled Jacobian rather than dropped. Measured 31% faster end to end here, to the same
reattachment length -- see :data:`FIELD_SPLIT`, which also records why the *cycle* count moves the other
way.

**Reference caveat (binding -- do not skip):** the OpenFOAM *steady* (SIMPLE) run does **not** fully
converge this case -- it limit-cycles at ~1e-3 residual on the separated 3D flow (though, unlike the 2D
pitzDaily case, the field is physical: no inlet checkerboard). Use a **transient-converged** field
(``runs/kwsst_transient/``) as the comparison target, and compare the outer-flow structure (velocity,
reattachment length) rather than the raw residual.

Run (after both OpenFOAM runs) from the repo root:
    python3 validation/bfs3d_openfoam/compare.py
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import re
import sys
import time
from pathlib import Path

# Running a script puts the SCRIPT's directory on `sys.path`, not the working directory, so
# `python3 validation/bfs3d_openfoam/compare.py` from the repo root cannot find `aquaflux` unless it is
# separately installed. Add the repo root explicitly so the documented invocation works against a plain
# checkout, as the sibling harnesses in this directory already do.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.io import read_openfoam
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, VenkatakrishnanLimiter
from aquaflux.solve import (
    CflResidualDualTimeControl,
    InnerIterateCheckpointer,
    MarchLogger,
    StateCheckpointer,
    combine_observers,
    native_nodal_inverse,
    relative_residual_gmres,
)
from aquaflux.turbulence import (
    CoupledJacobianProbe,
    CoupledRANS,
    LogScalars,
    SSTModel,
    SSTTurbulence,
    amg_beta_tracking_refresh,
    coupled_amg_continuation,
    coupled_fields,
    coupled_residuals,
    solve_reynolds_continuation,
)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "kwsst"  # steady run: the mesh + 3D cell centres (geometry)
TRANSIENT = HERE / "runs" / "kwsst_transient"  # the comparison fields (time-averaged / steady)
FIGS = HERE / "figures"

# The operating point (0.orig/ and constant/): U_in = 10 m/s, nu = 1e-5, h = 0.01 -> Re_h = 10000.
# k_in = 0.375 (5% turbulence intensity), omega_in = 1600 (length scale 0.07 H1). rho = 1.
RHO, NU = 1.0, 1e-5
U_IN, K_IN, OMEGA_IN = 10.0, 0.375, 1600.0
H = 0.01  # step height
WALLS = ["upperWall", "lowerWall", "sideWalls"]
# The coupled Newton march budget. Stiff, separating, 3D, high-Re case on a wall-function mesh; the cap
# is generous so the march exits on the tolerance, not the count.
# --- the solve configuration, each part measured on this case (see the README) -------------------
MAX_STEPS = 150  # per continuation rung
# ABSOLUTE stop on the row-scaled residual. That measure is already a fractional change per equation,
# so dividing it again by |R0| makes the bar a property of the initial guess -- and under continuation
# every rung re-bases |R0|, which let a later rung stop looser than an earlier one had already reached.
RTOL, ATOL = 0.0, 1e-5
# Reynolds continuation: `n` lower-Re points one decade apart, anchored at Re/10**n, then the target.
# A direct solve (0) does NOT converge here, so some continuation is required. Whether the Re/100 anchor
# earns its keep was measured rather than assumed, and the answer is genuinely mixed -- read both halves
# before changing it.
#
#                       2 rungs   1 rung
#     wall               1959 s   2007 s
#     steps                  58       43
#     Krylov cycles         277      264
#     mid-span x_r/h      8.361    8.361
#
# AGAINST the anchor: it is not needed for reachability (the cold start converges at Re/10 to the same
# root, in two independent runs), and reaching a converged Re/10 costs 800 s from cold against 1027 s by
# way of the anchor -- so as a route to Re/10 the extra rung is ~227 s of net cost.
#
# FOR keeping it: the total still comes out ahead, and one rung takes longer despite reaching Re/10
# cheaper, because it pays the difference back with interest in the target rung (932 s against 1207 s).
#
# Neither margin is decisive -- both are ~2%, both are single runs, and the target-rung spread (+-275 s,
# driven by how many beta escalations each ladder needs at the low-shift wall) is larger than the ladder
# effect itself; an earlier pair of runs ordered the totals the other way round. So this stays at 2: the
# measured total favours it, nothing measured argues for changing it, and the anchor is cheap insurance
# at a higher Reynolds number where the cold Re/10 solve may stop converging. `BFS3D_N_POINTS=1 ...`
# runs the one-rung ladder.
N_POINTS = int(os.environ.get("BFS3D_N_POINTS", "2"))
INNER_STEPS, INNER_TOL = 5, 1e-3
# Preconditioner bundle. ILU(1) DIVERGES at the low shifts this march's tail runs at (ground truth: 303
# negative pivots at beta = 0.02, zero for ILU(0)); zero fill converges at every shift tested and builds
# 3-4x faster. ILU(0) is the weaker smoother, so the extra sweeps pay more than they did for ILU(1).
# coarse=None stalls at every low shift. The beta floor is PRECONDITIONER-ONLY: the V-cycle is built at
# max(beta, floor) while the march solves at its own beta, so the root and the adjoint are unchanged.
FILL_LEVELS, SWEEPS, COARSE_EQ_LIMIT, PC_BETA_FLOOR = 0, 4, 2000, 0.05
# How far each COLUMN of the Jacobian is probed. The coloured probe costs one directional derivative per
# (colour, column field) and the colour count climbs steeply with the reach -- 11 colours at reach one,
# 39 at two, 94 at three on this mesh -- so probing a column further than it reaches is pure cost. The
# assembled sparsity stays at reach three either way, so every consumer sees the same matrix.
#
# Measured on this case with `probe_reach_audit.py`, at a cold iterate, a developed step and the march's
# hardest solve alike: the share of each column's norm lying beyond reach two is
#
#     u 1.4e-04   v 1.1e-05   w 3.4e-05   |   p 9.8e-16   k 1.5e-17   omega 1.2e-17
#
# so p, k and omega close inside reach two and the velocities do not. That is a consequence of the
# schemes -- first-order upwind on k/omega with a non-orthogonal diffusion correction reaches two, while
# the velocity columns carry the limited second-order upwind reconstruction out to three -- so it must be
# re-measured for any case that changes them, NOT inherited. Shortening a column that does carry far
# couplings corrupts its near entries rather than truncating them.
#
# Shortening all three would be 564 probes -> 399 (-29%), and the two Jacobians agree to 5.7e-16 relative
# Frobenius, i.e. float64 rounding. The SHIPPED value shortens only k and omega: 564 -> 454 (-16%).
#
# That measurement is sound and it is NOT what licenses the shortening, which is worth stating plainly
# because shortening p on the strength of it has been tried twice and withdrawn twice.
# (3, 3, 3, 2, 2, 2) DIVERGES this case on its first step -- 44 restart cycles instead of 3, the step
# length collapsing to 0.000, the shift at its 16.0 ceiling by step two -- and neither the shell norms
# above nor the Frobenius agreement predicted
# it, because the fault was never in the matrix. Every value in it was exact to the floating-point
# floor. What differed was the SPARSITY: a shortened column writes its out-of-reach entries as exact
# zeros where a uniform probe leaves the true value (tiny, around 1e-26, but nonzero), and the sparse
# arithmetic that assembled the preconditioner stored only entries whose result was nonzero. Six and a
# half million positions -- a sixth of the operator -- were dropped from what the zero-fill incomplete
# LU factorizes, which is a structurally weaker factorization of a numerically identical matrix.
#
# Keeping those positions in the pattern DOES cure that divergence -- with the shift and the
# equilibration applied to the stored values in place, (3, 3, 3, 2, 2, 2) converges to the same root as a
# uniform probe in every reported digit: 67 steps, 320 cycles, residual 3.586e-06, mid-span reattachment
# 8.3611, eddy-viscosity peak 150.1071, and independently of the trailing inverse's `equilibrate`
# setting. But it is not a remedy that is available, because the positions it keeps are stored EXACT
# ZEROS and an incomplete factorization cannot be handed those: it takes its pattern from the entries
# that are stored, so each one is a slot the elimination deposits fill into. Measured at the converged
# state with no pseudo-transient shift -- the operator the adjoint solves, and the one every gradient
# goes through -- carrying them costs 58 restart cycles at a true relative residual of 2.299e-02 against
# 11 cycles to 8.474e-11 without. So they are pruned before the factorization sees them, which puts the
# pressure column back in the configuration that diverges this case.
#
# Hence p stays at reach three. Only k and omega are shortened, and that split follows the SPLIT
# PRECONDITIONER rather than the schemes: with the flow block leading, the field split applies
# `d R_turb / d flow` and never `d R_flow / d turb`, so the turbulence COLUMNS are read only by the
# turbulence ROWS -- whose hierarchy is smoothed by a per-cell block inverse that sees a cell's own 2x2
# block and nothing else. Corruption confined to those columns cannot reach the saddle. The p column has
# no such shelter: it feeds the [u, v, w, p] block, whose smoother is an incomplete LU.
#
# What shortening k and omega costs, measured: nothing at the shift the march runs at (every arm ties at
# 4 restart cycles and 1.435e-13 at a step-initial state, preconditioner floored), and a factor of two at
# zero shift (22 cycles against a uniform probe's 11, to the same 1e-11 floor). It converges either way
# there, so this is a cost to know about rather than a reason to widen the march's probe -- but "proven
# safe" does not cross the shift boundary on its own.
#
# One caution that survives all of it: shortening a column is only sound where its mass beyond the
# shortened reach is negligible AT EVERY STATE THE MARCH VISITS, and a short-probed column with far
# couplings has its NEAR entries corrupted rather than its far entries dropped -- a colouring is
# collision-free only for the pattern it was built from, so two cells sharing a reach-two colour can
# both couple to one row at distance three and the response is charged entirely to the near one.
# Measured structurally, that aliasing touches 53% of the entries of every shortened column here; it
# is harmless only because the folded values are at the floating-point floor (3e-29 for k and omega).
# Re-measure per (row field, column field) pair, never over a whole column -- a column-wide norm on
# this system is set by the omega rows and cannot see a wrong pressure block.
#
# `BFS3D_COLUMN_REACH` takes a comma-separated reach per column ("3,3,3,3,2,2"), or `0` for a uniform
# reach-three probe. Re-measure before shortening any column on a case that changes the schemes or the
# split; none of this is inheritable.
_DEFAULT_COLUMN_REACH = "3,3,3,3,2,2"  # [u, v, w, p, k, omega]
_column_reach = os.environ.get("BFS3D_COLUMN_REACH", _DEFAULT_COLUMN_REACH)
if _column_reach in ("", "0"):
    COLUMN_REACH = None  # uniform, at the widest reach the assembler asks for
else:
    try:
        COLUMN_REACH = tuple(int(r) for r in _column_reach.split(","))
    except ValueError:
        raise SystemExit(
            f"BFS3D_COLUMN_REACH={_column_reach!r} is not a comma-separated list of integers "
            f'(e.g. "{_DEFAULT_COLUMN_REACH}"), nor "0" for a uniform reach.'
        ) from None
    if len(COLUMN_REACH) != 6:
        raise SystemExit(
            f"BFS3D_COLUMN_REACH={_column_reach!r} gives {len(COLUMN_REACH)} reaches; this case has "
            f"six columns [u, v, w, p, k, omega]."
        )
# Split the preconditioner's hierarchy in two -- the [u,v,w,p] saddle and the [k,omega] transported
# scalars each get their own, with one triangle of the coupling between them retained exactly -- rather
# than putting all six fields through one hierarchy with one smoother. ON, because it is measured 31%
# faster end to end on this case at the identical configuration:
#
#                       monolithic     split
#     wall                  3140 s    2161 s   (-31%)
#     steps                     58        66
#     Krylov cycles            293       324   (+11%)
#     refresh          19 / 310 s   23 / 352 s
#     mid-span x_r/h         8.361     8.361   (identical)
#
# Read the cycle row before concluding anything from a cycle count: the split takes MORE cycles and is
# far faster, because two smaller V-cycles plus one sparse coupling product apply much more cheaply than
# one six-field V-cycle. Its mean cycles per inner solve is actually lower (1.49 vs 1.68) -- the higher
# total comes from more, cheaper steps. It also triggers ~4 more cost-driven refreshes (9.7% of inner
# solves cross the threshold against 9.0%), which hands back ~42 s of the ~980 s saved.
# `BFS3D_FIELD_SPLIT=0` restores the monolithic V-cycle for an A/B.
FIELD_SPLIT = os.environ.get("BFS3D_FIELD_SPLIT", "1") not in ("", "0")
# The smoother on the TURBULENCE half of the split, which need not be the one the saddle needs. The
# shipped four sweeps of a zero-fill incomplete LU were tuned against the six-field block; `[k, omega]`
# is not a saddle but a two-field advection-diffusion-reaction pair with a genuine diagonal, and it is
# both easier and cheaper to precondition. Measured across two step-initial states and the march's
# hardest iterate, weighted by how often each class of solve occurs (139 of 194 inner solves take one
# restart cycle), as expected seconds per solve against the shipped smoother:
#
#     ilu0 x1        0.89x     pbjacobi x2    0.90x     jacobi x2      0.91x
#     ilu0 x2        0.92x     jacobi x4      0.93x     pbjacobi x4    0.94x
#     ilu0 x4 (shipped)  --    sor x4         1.01x     jacobi x1      1.07x
#
# The ranking is state-dependent in a way a single probe inverts: point-block Jacobi buys an extra
# restart cycle only where the operator is hard, so it is the WORST arm on the hardest iterate (1.15x)
# and a win over the march's real mix (0.94x). Rank on the states a march actually repeats; screen on
# the hard one.
#
# ⚠️ NO ARM HAS BEEN SETTLED ON A MARCH YET, and the first two attempts do NOT count. Both were launched
# without `BFS3D_REFRESH_ON_CYCLES=3`, which at the time defaulted to the scheduled cadence -- a
# configuration measured at 3632 s against 1959 s for the otherwise identical arm. So both ran a
# different refresh trigger from the archived baseline they were compared against, and the difference
# they showed is not attributable to the smoother. (`pbjacobix2` looked catastrophic and `ilu0x1` looked
# 20% slow; neither reading survives.) The default is now 3 so this cannot recur silently.
#
# A SECOND fairness problem is live even with the trigger set correctly, and it has not been solved:
# `refresh_on_cycles` and `cycle_budget` are denominated in CYCLES, so a preconditioner that shifts the
# cycle distribution changes the EFFECTIVE trigger point rather than leaving it fixed. A weaker smoother
# is then penalized twice -- more cycles, and more refreshes because those cycles cross a threshold
# calibrated for a stronger one. This is the same argument that makes `_RESTART_SCALE` necessary above.
# Comparing arms fairly needs the refresh COUNT reported alongside the wall, and a scaled trigger if the
# counts diverge.
#
# What the screen does support, and what it does not: it ranks per-solve COST honestly, and it measures
# each arm's STRENGTH as the cycles to a tight stop -- shipped 3/3/4 across the two step-initial states
# and the hard iterate, `ilu0x1` 3/4/5, `jacobix4` 4/5/6, `pbjacobix2` 5/6/7, `jacobix1` 8/10/13. What
# it cannot see is the quality of the correction an arm returns at the march's own LOOSE stop
# (`forward_rtol = 0.3`): a strong preconditioner overshoots that target by orders of magnitude inside
# one restart cycle, while a weak one lands near it, and both report "1 cycle". If that gap matters, a
# weaker arm hands back a worse Newton direction, the line search clips and the step control escalates
# -- none of which a timing screen registers. That is a live hypothesis, NOT a measured result; the
# achieved residual at the loose stop is what would test it.
#
# The two knobs are separate on purpose, because they answer separate questions: HOW MANY sweeps of the
# trailing smoother, and WHICH smoother. Sweeps is the one that paid, and it is now a first-class solver
# parameter (`coupled_amg_continuation(trailing_smoother_sweeps=...)`) rather than a raw options string,
# so `BFS3D_TRAILING_SWEEPS` just forwards it. The library default is 1 -- the measurement above.
TRAILING_SWEEPS = int(os.environ.get("BFS3D_TRAILING_SWEEPS", "1"))
# The smoother METHOD on the trailing half. Empty (default) is the zero-fill incomplete LU the saddle
# also uses. The Jacobi-class alternatives are here because they are the ones that could run on an
# accelerator without a host solver: a diagonal scaling (`jacobi`) or a batch of independent per-cell
# dense inverses (`pbjacobi`) is a matrix-vector product with no factorization to store and no
# sequential triangular solve. Neither pins a sweep count -- they inherit `TRAILING_SWEEPS`, so the two
# knobs compose instead of one silently overriding the other.
_TURBULENCE_SMOOTHERS = {
    "": None,
    "jacobi": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "jacobi",
    },
    # `pbjacobi` inverts each cell's own dense 2x2 [k, omega] block instead of just its two diagonal
    # entries, which on this operator is the difference that should matter: the equilibrated cell blocks
    # are lower-triangular with unit diagonal and a subdiagonal of order 100-340 (omega depends
    # enormously on same-cell k through the production limiter and the destruction pair, while k barely
    # depends on omega), and the coupling is ~100 % same-cell. A point method discards all of it; a block
    # solve is a two-line forward substitution, perfectly conditioned and nearly free.
    #
    # Measured, it does capture that: split by field, a pbjacobi-smoothed V-cycle lands 10x closer to an
    # incomplete-LU-smoothed one than a point-Jacobi one does in the omega rows (1.5e-04 against
    # 1.5e-03). The end-to-end gain is nonetheless small -- the coarse correction and the outer Krylov
    # absorb most of it -- which is why it screens as a near-tie with plain Jacobi rather than a rout.
    #
    # ⚠️ It needs SORTED column indices, and `equilibrate_cell_major` does not produce them. The
    # `AmgVCycle` build path sorts before wrapping the matrix for PETSc, so this option is correct here;
    # a probe that skips that path and hands PETSc the raw cell-major output gets NaN in most entries,
    # while `jacobi` and `ilu` survive it (a linear diagonal scan does not care about order). If a
    # point-block arm ever reports NaN, check the index order before concluding anything about the
    # method.
    "pbjacobi": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "pbjacobi",
    },
    # The SAME smoother undamped, which is PETSc's own default Richardson scale and a materially
    # different arm -- the damped one above relaxes by 0.7 of every correction. Kept apart rather than
    # folded together because the recorded screen of the arm above was taken at 0.7, and because the
    # undamped form is the one the framework-native block-Jacobi hierarchy is built to reproduce: an
    # end-to-end comparison against the damped variant would understate the host solver and would not
    # be the like-for-like run it appeared to be.
    "pbjacobi1": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 1.0,
        "mg_levels_pc_type": "pbjacobi",
    },
    "sor": {"mg_levels_ksp_type": "richardson", "mg_levels_pc_type": "sor"},
}
_TURBULENCE_SMOOTHER = os.environ.get("BFS3D_TURBULENCE_SMOOTHER", "")
if _TURBULENCE_SMOOTHER not in _TURBULENCE_SMOOTHERS:
    raise SystemExit(
        f"BFS3D_TURBULENCE_SMOOTHER={_TURBULENCE_SMOOTHER!r} is not one of "
        f"{sorted(k for k in _TURBULENCE_SMOOTHERS if k)}"
    )
TRAILING_OPTIONS = _TURBULENCE_SMOOTHERS[_TURBULENCE_SMOOTHER]

# Which preconditioner the trailing [k, omega] block gets. "native" (default) is the
# differentiable-framework nodal hierarchy; "petsc" is the host GAMG V-cycle the case originally ran.
# Their aggregation and smoother are configured to match -- measured on the block alone they reach the
# same 2 restart cycles on the same-sized coarse space. Beyond cycles, the native inverse is plain array
# work rather than a host callback, so it is the half of the preconditioner that could run on an
# accelerator.
#
# ⚠️ THE DEFAULT WAS "petsc", AND THE MEASUREMENT THAT MOVED IT ALSO RETIRED A LONG-STANDING CLAIM.
# The host arm was believed faster on this case -- 58 steps against 67 -- but the two logs behind that
# reading were not comparable: the 58-step run used the OTHER wall condition on `k` (`dirichlet`), had
# no positivity floor, and predated both. Run as a controlled pair at the settings below, the ranking
# REVERSES:
#
#                     native    petsc
#     wall            2124 s    2893 s   (+36 % for the host arm)
#     steps               67        72
#     Krylov cycles      329       371
#     escalations          4         8
#     mid-span x_r/h    8.36      8.36   (identical -- same root, different path cost)
#
# So the host arm is not the faster one here; the wall condition was carrying that difference, and the
# `dirichlet` number is not a target because it is a different problem (see `K_WALL` below). Leaving
# "petsc" as the default would have shipped the slowest measured arm. `BFS3D_TURBULENCE_INVERSE=petsc`
# still selects the host V-cycle for an A/B of the inverse itself.
_TURBULENCE_INVERSES = ("petsc", "native")
TURBULENCE_INVERSE = os.environ.get("BFS3D_TURBULENCE_INVERSE", "native")
if TURBULENCE_INVERSE not in _TURBULENCE_INVERSES:
    raise SystemExit(
        f"BFS3D_TURBULENCE_INVERSE={TURBULENCE_INVERSE!r} is not one of {list(_TURBULENCE_INVERSES)}"
    )
#: The native inverse's own settings, kept here rather than left to the class defaults so the banner can
#: print them and a later reader can tell two runs apart. `max_coarse` is the one deliberate departure
#: from the class default: the coarse grid is grown to match the host V-cycle's own coarse-equation
#: limit, since a coarse grid big enough to invert the global coupling exactly is worth a great deal on
#: this operator.
NATIVE_TRAILING = {
    "max_coarse": COARSE_EQ_LIMIT,
    # Kept off by default, and kept exposed. With NO positivity floor this flag decides whether the case
    # converges at all: an otherwise-identical pair of marches came out opposite, the rescaled one losing
    # its line-search factor on the middle rung and stalling with the residual frozen. But that turned out
    # to be a trigger rather than a cause -- with `K_POSITIVITY_FLOOR` set, both settings converge to the
    # same root (69 steps rescaled, 67 unscaled), because what actually killed the rescaled arm was one
    # numerically-zero cell ratcheting the global step cap toward zero. Rescaling merely reached that cell
    # a few steps sooner: the two arms differ from step one in the pressure-block residual and amplify
    # until they cross the limiter around step sixteen, where one is clipped and the other is not.
    # Off is free and is what the converging arms were measured with; both arms stay runnable.
    # Default OFF, which is NOT the class default: on this case it is the difference between a march
    # that converges every rung and one that stalls. `BFS3D_NATIVE_EQUILIBRATE=1` selects the rescaled
    # arm for an A/B of the flag itself.
    "equilibrate": os.environ.get("BFS3D_NATIVE_EQUILIBRATE", "0") not in ("", "0"),
}
#: Write every trailing sub-block to disk just BEFORE its inverse is built, keeping only the last.
#: The build refuses a singular cell block, and that refusal fires from a mid-step refresh whose
#: iterate no observer records -- the inner-iterate dump happens after an iteration succeeds, so the
#: one that fails is precisely the one never written. Dumping before the build inverts that: whatever
#: happens, the last file on disk is the operator that failed, with no state to reload and no shift to
#: pair correctly. Off by default; it costs a ~35 MB write per refresh.
DUMP_TRAILING_BLOCK = os.environ.get("BFS3D_DUMP_TRAILING_BLOCK", "") not in ("", "0")


def _dumping(factory):
    """Wrap a trailing inverse so every block it is about to consume is saved first.

    Both entry points have to be covered, and missing one wasted three capture runs: the factory is
    called once when the split is first built, but a mid-march refresh re-fits the **existing** inverse
    through ``refactor_block`` and never goes near the factory. The refusal happens on a refresh, so
    wrapping only the factory dumps every block except the one that matters.
    """
    import scipy.sparse as _sp

    def save(block):
        _sp.save_npz(HERE / "checkpoints" / "trailing-block.npz", _sp.csr_matrix(block))

    class Dumping:
        """Forwards the inverse's interface, saving the operator ahead of any (re)build."""

        def __init__(self, inverse):
            self._inverse = inverse

        @property
        def n_dofs(self):
            return self._inverse.n_dofs

        def apply(self, residual, *, transpose=False):
            return self._inverse.apply(residual, transpose=transpose)

        def refactor_block(self, block):
            save(block)
            return self._inverse.refactor_block(block)

        def destroy(self):
            self._inverse.destroy()

    def build(block, n_group_fields):
        save(block)
        return Dumping(factory(block, n_group_fields))

    return build


TRAILING_INVERSE = (
    native_nodal_inverse(**NATIVE_TRAILING) if TURBULENCE_INVERSE == "native" else None
)


#: `BFS3D_FLOW_INVERSE=native` replaces the LEADING (flow saddle) block's host V-cycle with the JAX-native
#: SIMPLE-smoothed hierarchy, which is the arm the native-preconditioner work has been measuring on single
#: states. Off by default: it is a measurement seam, not a shipped default, and on single states it costs
#: more wall clock than the incumbent even where it converges in fewer cycles.
#:
#: ⚠️ `BFS3D_REFRESH_ON_CYCLES` must be raised alongside it. The refresh fires when a solve REACHES the
#: threshold, and 3 is calibrated to an incomplete-LU that runs two cycles per solve; a preconditioner
#: that healthily takes six or seven would trip it on essentially every step and the march would measure
#: the trigger rather than the preconditioner.
FLOW_INVERSE = os.environ.get("BFS3D_FLOW_INVERSE", "petsc")
if FLOW_INVERSE not in ("petsc", "native"):
    raise SystemExit(f"BFS3D_FLOW_INVERSE={FLOW_INVERSE!r} is not one of ['petsc', 'native']")
LEADING_INVERSE = None
if FLOW_INVERSE == "native":
    #: The arm measured best on single states: strength-of-connection aggregation with no singleton
    #: aggregates, five levels, a per-cell block velocity splitting and an undamped correction.
    #:
    #: Two settings are exposed to the environment because a march is a different operating point from
    #: the state the rest were chosen on. ``BFS3D_FLOW_SWEEPS`` -- the sweep count was calibrated at zero
    #: shift, the adjoint's operator, and every shift the march runs at makes the block easier, so the
    #: march may not need four. ``BFS3D_FLOW_FROZEN_COARSENING`` -- at this strength threshold the
    #: aggregation reads values, so each refresh re-coarsens and retraces the compiled cycle; frozen, the
    #: partition is the one derived at the first build and reused for the whole march.
    _NATIVE_FLOW = dict(
        sweeps=int(os.environ.get("BFS3D_FLOW_SWEEPS", "4")),
        pressure_sweeps=2,
        strength_threshold=0.25,
        avoid_singletons=True,
        aggressive=0,
        levels=5,
        max_coarse=500,
        block_splitting=True,
        omega=1.0,
        frozen_coarsening=os.environ.get("BFS3D_FLOW_FROZEN_COARSENING", "") not in ("", "0"),
        shape_headroom=(
            float(os.environ["BFS3D_FLOW_SHAPE_HEADROOM"])
            if os.environ.get("BFS3D_FLOW_SHAPE_HEADROOM")
            else None
        ),
    )

    def _native_flow_inverse(block, n_fields):
        # Imported HERE, not at module scope: the probe imports this module for its bundle constants, so
        # a top-level import the other way is a cycle. Deferring it to first use breaks the cycle without
        # either module having to know about the other's import order.
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.insert(0, str(_Path(__file__).parent))
        from field_split_probe import NativeSimpleInverse

        return NativeSimpleInverse(block, n_fields, **_NATIVE_FLOW)

    LEADING_INVERSE = _native_flow_inverse


if TRAILING_INVERSE is not None and DUMP_TRAILING_BLOCK:
    TRAILING_INVERSE = _dumping(TRAILING_INVERSE)


def _native_trailing_description() -> str:
    """Every setting the native trailing hierarchy is actually built with, defaults included.

    Reads them off the class signature rather than restating them, so a changed default cannot make the
    banner lie -- which is the one failure a configuration banner must not have.
    """
    from aquaflux.solve import NodalNativeInverse

    settings = {
        name: parameter.default
        for name, parameter in inspect.signature(NodalNativeInverse).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    settings.update(NATIVE_TRAILING)
    return ", ".join(f"{k}={v}" for k, v in settings.items())


#: Suffix marking the PETSc trailing-smoother banner lines as dead. `build_block_triangular_field_split`
#: uses `trailing_options` / `trailing_smoother_sweeps` only on the branch that builds its own V-cycle;
#: a supplied `trailing_inverse` skips that branch entirely.
_TRAILING_SMOOTHER_NOTE = (
    "" if TRAILING_INVERSE is None else "  (unused: the native trailing inverse replaces it)"
)


# Absolute room in `k` the step limiter grants every cell, so a cell whose `k` is numerically zero
# cannot set the step length for all 23040. Calibrated by replaying the recorded clips
# (`positivity_floor_calibration.py`): unfloored, the worst recorded cap is 1.05e-09; the dead cells
# form a graded population rather than a single outlier, so a small floor only promotes the next one
# (1e-12 reaches 1.6e-02, 1e-10 reaches 6.9e-02) and the knee is near 1e-08, where the worst cap
# becomes ~0.35. At 1e-06 the binding cell is a live one (k 4.7e-03) capped at 0.84, which is the
# limiter working -- so 1e-08 keeps two orders of margin below anything physical here, where the inlet
# k is 0.375 and the mesh median ~3e-2.
#
# `0` (the library default, and `BFS3D_K_POSITIVITY_FLOOR=0` here) is the plain
# fraction-to-the-boundary rule. It moves neither the converged root nor the adjoint: at a root the
# correction vanishes and the limiter is inactive for any floor. It is safe here only because every
# consumer of the solved `k` clamps at zero, so a cell that dips slightly negative no longer reaches a
# bare sqrt -- and the destruction term, running on the k-independent viscous omega branch there,
# pushes such a cell back up.
#
# ⚠️ THE CASE DEFAULT IS 1e-08, WHICH IS NOT THE LIBRARY DEFAULT, and it is load-bearing rather than a
# tuning preference: with the `zerogradient` wall condition above, an unfloored march is the one whose
# cap ratchets to 1.05e-09 and stalls. Every archived measurement quoted in this file was taken at
# 1e-08, so leaving the default at the library's `0` would mean the case's own default configuration
# was one that none of its recorded numbers describe -- and, on the arm that was measured both ways,
# one that does not converge.
#
# ⚠️ BUT THE FLOOR POSTPONES THE RATCHET, IT DOES NOT REMOVE IT -- and this is algebra, not a
# measurement, so no configuration caveat applies. When the cap binds, the step takes
# `k_new = k - tau*(k + floor)`, hence
#
#     (k_new + floor) = (1 - tau) * (k + floor)
#
# so the SHIFTED variable `k + floor` decays by exactly `1 - tau` per clipped step whatever the floor
# is. The cell parks at `k -> -tau*floor` while the cap keeps falling 100x per step; the floor buys
# `log10(floor / k_0)` decades of headroom and then reproduces the same collapse. That is why 1e-08
# moved the march 77 -> 67 steps yet the cap still reached 1.44e-05 at step 51 WITH the floor active.
# Raising the floor further buys two more decades and one-off ceiling (worst cap ~0.84 at 1e-06 rather
# than ~0.35 at 1e-08), not a cure -- and 1e-06 is one decade from this case's own live near-wall `k`.
# The structural fix is to stop one cell setting a GLOBAL step length at all; see the discussion of
# clipping the correction per cell rather than capping the step, which leaves the cap at 1.
K_POSITIVITY_FLOOR = float(os.environ.get("BFS3D_K_POSITIVITY_FLOOR", "1e-8") or 0.0)

#: Clip each cell's OWN `k` correction rather than capping the whole step by the worst cell
#: (`BFS3D_K_POSITIVITY_PROJECTION=1`). Off by default and byte-identical off, pending an end-to-end
#: measurement on this case.
#:
#: This is the structural answer to what the floor above can only postpone. The cap is a minimum over
#: cells, so the stagnant corner where the step face, the floor and a side wall meet -- no shear, so no
#: `k` production, so `k` decaying with nothing to arrest it -- sets the step length for all 23040.
#: A floor buys `log10(floor / k)` decades and no more, because when the cap binds the step takes
#: `k_new = k - tau (k + floor)`, hence `(k_new + floor) = (1 - tau)(k + floor)`: the SHIFTED value
#: decays by exactly `1 - tau` per capped step whatever the floor is, so the collapse repeats a fixed
#: number of decades later. That is algebra, not a measurement, and it is why raising the floor is not
#: the lever here. Clipping per cell removes the coupling instead -- the dead cell still decays,
#: because that is what its own equation asks, but alone.
#:
#: It composes with the cap rather than replacing it: applied first, it leaves every decreasing cell
#: with `|dk| <= tau (k + floor)`, so the cap computes exactly 1 and the `limit` aside below keeps
#: reporting (as "nothing bound") instead of going silent.
K_POSITIVITY_PROJECTION = os.environ.get("BFS3D_K_POSITIVITY_PROJECTION", "") not in ("", "0")


# The inexact-Newton stop for each inner linear solve, measured in the ROW-SCALED `coupled_scaled_norm`
# (not the Euclidean one -- the coupled Euclidean residual is ~100% omega, which is why the row-scaled
# measure exists). `0.3` is the builder default: every field block is resolved loosely so the flow is
# never left blind, and the outer Newton iteration recovers the accuracy.
#
# Exposed because a loose stop leaves the accepted correction substantially determined by the
# PRECONDITIONER rather than the operator -- two preconditioners land at different points inside the same
# admissible ball -- and the step length is decided by a MINIMUM over cells, an extreme order statistic
# that a norm-based tolerance does not control. Tightening it trades cycles per step for a correction
# that is closer to the true shifted-Newton direction.
FORWARD_RTOL = float(os.environ.get("BFS3D_FORWARD_RTOL", "0.3"))


CYCLE_BUDGET = 42  # summed per step: a cost cap, so summed is what it should cap
RETRY_ON_CYCLES = (
    10  # PER SOLVE: a summed trigger is ~6x more sensitive for a 5-inner step than a 1-inner one
)
# The step-length bailout, which catches the failure the cycle count cannot see: solves that stay CHEAP
# while the step achieves nothing, because a positivity cap or a non-descending direction leaves almost
# none of the correction followable. On the march this replaced, four consecutive steps ran a full inner
# loop each at 5-12 cycles, moved the residual not at all, and escaped only once the step control's own
# backoff had doubled beta four times -- one whole step per doubling.
#
# Calibrated from that march's own step table rather than chosen: no productive step went below
# a_min 0.191, while all four dead ones reported 0.000 (their inner collapses reaching 0.001 and 0.003).
# 0.01 sits an order of magnitude clear of both. Unlike the cycle thresholds below it is dimensionless,
# so it does NOT scale with the restart length. Measured end to end at the identical configuration:
#
#                       off        on
#     wall            2161 s    1959 s   (-9.3%)
#     steps               66        58
#     Krylov cycles      324       277   (-15%)
#     mid-span x_r/h   8.361     8.361   (identical)
#
# It fires ONCE in the whole march, and the two lower rungs come out identical to the cycle -- the
# trigger is inert wherever the line search is healthy, so the entire saving is the target rung
# (29 steps / 1131 s -> 21 / 932). `BFS3D_RETRY_ON_ALPHA=0 ...` disables it.
RETRY_ON_ALPHA = float(os.environ.get("BFS3D_RETRY_ON_ALPHA", "0.01")) or None
# The forward GMRES restart length, and the reason it is worth varying: a restarted GMRES tests
# convergence only at restart boundaries, so a solve that needs three matrix-vector products still pays
# a full restart's worth. Cycle counts cannot see that -- such a solve reports one cycle either way --
# so shortening the restart reduces seconds per cycle while leaving every cycle-based measurement
# unchanged. `BFS3D_FORWARD_RESTART=4 ...` to try it.
#
# BOTH cost thresholds above are denominated in CYCLES, so they must scale with the restart or the
# experiment changes the march's control behaviour rather than just its cost: at a restart of 5 an
# unscaled `retry_on_cycles = 10` would fire after 50 matrix-vector products where it used to take 150.
# Scaling by the ratio keeps every bailout at the same matvec count, so the only variable is how much
# over-solving happens inside a cycle. Vary the restart through the builder's own `forward_restart`, NOT
# by passing a whole `forward_solver`: the builder's default also carries a loose row-scaled stop that a
# hand-built solver would silently replace, which measures something else entirely.
BASELINE_RESTART = 15  # the coupled AMG builder's own default
FORWARD_RESTART = int(os.environ.get("BFS3D_FORWARD_RESTART", str(BASELINE_RESTART)))
_RESTART_SCALE = BASELINE_RESTART / FORWARD_RESTART
RETRY_BETA_FACTOR = 2.0
# How many per-step states to retain. Three is enough to restart from, which is all a normal run needs.
# A PRECONDITIONER STUDY needs more: an easy operator does not discriminate between preconditioners, so a
# sweep has to run at the march's own HARD states (highest cycle count, clipped a_min, a retry flag) and
# those are mid-march. Set `BFS3D_CHECKPOINT_KEEP` high enough to cover the run (~1.1 MB per step) and the
# whole trajectory is kept: `BFS3D_CHECKPOINT_KEEP=80 python3 validation/bfs3d_openfoam/compare.py`.
CHECKPOINT_KEEP = int(os.environ.get("BFS3D_CHECKPOINT_KEEP", "3"))
# Save the INNER iterates whose linear solve reached this many restart cycles. Off by default (0), and
# the march is byte-identical with it off. It exists because a checkpoint is written at the END of a
# step, so it holds the state the next step begins from -- and this march's step-initial solves all cost
# at most 2 cycles while solves later in the inner loop reach 15. The hard operators are only reachable
# here: they cannot be replayed from a checkpoint afterwards, because a step's preconditioner and shift
# are a product of the refresh history rather than of the state.
INNER_DUMP_ABOVE = int(os.environ.get("BFS3D_INNER_DUMP_ABOVE", "0"))
# Save the (iterate, correction) pair whenever the k-positivity cap comes out below this fraction. Off
# by default (0), and the march is byte-identical with it off.
#
# It exists because the cap is the only quantity in the step table that CANNOT be reconstructed after
# the fact: it is a property of the correction `delta`, and no checkpoint holds one. A step checkpoint
# holds the state a step ended at, and the inner-iterate dump holds the iterate an inner iteration
# reached -- neither carries the direction, so "which cells is the cap binding on" is unanswerable from
# either. Capturing it at the call site is the only honest answer; dividing an iterate difference by the
# reported alpha would recover the direction only up to whatever the line search did to it.
#
# The threshold is `RETRY_ON_ALPHA` by default, i.e. exactly the caps the march reacts to. Dumps STOP
# after `DUMP_STEP_LIMIT_KEEP` of them rather than wrapping, so the FIRST binding events survive -- the
# escalation ladder that follows one drives the cap two orders of magnitude further down, and a rolling
# buffer would keep only that tail and throw away the step that started it.
DUMP_STEP_LIMIT = float(os.environ.get("BFS3D_DUMP_STEP_LIMIT", "0") or 0.0)
DUMP_STEP_LIMIT_KEEP = int(os.environ.get("BFS3D_DUMP_STEP_LIMIT_KEEP", "12"))
# Refresh the preconditioner MID-STEP as soon as one solve reaches this many restart cycles, and switch
# the scheduled refreshes off. `0` selects the scheduled cadence instead.
#
# The point is the swap, not the addition. Measured on the 3501 s march: the scheduled refreshes cost
# 742 s -- 21 % of the wall -- while 193 of 232 solves already took a single cycle, so most of that is
# maintaining a freshness nothing consumes. And no fixed cadence can be right, because the interval that
# matters is regime-dependent: one step of staleness is free at beta 0.333 and triples the cost at 0.029.
# Reacting to the cost itself adapts; a schedule cannot. Refreshing mid-step (rather than at the next
# step boundary) also keeps the inner loop's progress, where the current reaction -- abort and escalate
# beta -- throws away the work and the pseudo-timestep together.
#
# ⚠️ THE DEFAULT IS 3, AND IT USED TO BE 0. Every recorded measurement on this case was taken
# cost-triggered at 3; the scheduled cadence is measured at 3632 s against 1959 s for the otherwise
# identical arm, so `0` selected a configuration that is 1.85x slower and that nothing uses. Leaving it
# as the default meant every A/B launched without the environment variable silently measured the refresh
# TRIGGER instead of the thing under test -- which happened, to two preconditioner arms in one session,
# despite the trap being written down. A default nobody wants is a trap, not a setting: the fix is the
# default, not another warning. `BFS3D_REFRESH_ON_CYCLES=0` still selects the scheduled cadence for an
# A/B of the trigger itself.
REFRESH_ON_CYCLES = int(os.environ.get("BFS3D_REFRESH_ON_CYCLES", "3"))
#: The β-MISMATCH refresh trigger, as a fraction of the β the V-cycle was last built at. Off by default
#: (`inf`, a gate that can never fire), which is byte-identical to the configuration every archived
#: measurement on this case was taken under.
#:
#: What it is for, and why `inf` is not obviously right. With the cost trigger above selected, the
#: scheduled cadences are switched off by setting this gate to `inf` -- so the ONLY thing that rebuilds
#: the V-cycle is a solve that has already cost more than `REFRESH_ON_CYCLES` cycles. That is a purely
#: REACTIVE rule, and it has one blind spot: the β-escalation bailout. When a step is redone at
#: `β *= RETRY_BETA_FACTOR`, the march re-invokes this same refresh hook specifically to re-match the
#: V-cycle to the escalated β -- and with the gate at `inf` that call does nothing, so the escalated
#: attempt is solved against a V-cycle built for a β up to 4x smaller.
#:
#: The escalation then cannot cure the step it was invoked for. Measured across the three converging
#: marches that reached the target rung, the same pair repeats
#: bit-for-bit in all three: an `e2` step at β = 0.2341 whose V-cycle was left stale returns a_min
#: 0.000 -- a step that moves nothing and costs three solves -- and the FOLLOWING step, at β = 0.9364,
#: takes a_min 1.000 once the cost trigger has finally forced a rebuild. Across those same runs every
#: escalated step whose V-cycle WAS rebuilt came back with a_min >= 0.595, and not one was null.
#:
#: The mechanism is the one the smoother-screen note above raises as a live hypothesis: at the march's
#: loose inner stop the accepted correction is substantially determined by the preconditioner, and the
#: step length is a MINIMUM over cells, so a mismatched V-cycle need not cost cycles to hand back a
#: direction whose worst cell collapses the line search. Both null steps above solved in 2 cycles.
#:
#: Sizing, replayed over a completed march's own β sequence: at 0.9 the gate would add 2 step-boundary
#: rebuilds (~35 s) on top of the 23 that run already, plus one per escalation attempt. 0.9 is chosen so
#: a DOUBLING trips it -- which every escalation is -- while the control's own /1.5 growth (a 33 % fall)
#: does not, since re-matching a V-cycle to a β that is drifting slowly is what the cost trigger already
#: covers more cheaply. Lower values get expensive fast: 0.5 would add 13 rebuilds (~225 s).
REFRESH_ON_BETA = float(os.environ.get("BFS3D_REFRESH_ON_BETA", "0") or 0.0) or float("inf")
#: The wall boundary condition on `k`, as an A/B. `Dirichlet(0)` (default) is the resolved-wall
#: condition -- turbulent fluctuations vanish at a no-slip wall, so `k -> 0`. `BFS3D_K_WALL=zerogradient`
#: selects the wall-function condition instead.
#:
#: **Both are defensible, and the established codes SPLIT on it** -- so this is a knob, not a fix:
#: OpenFOAM's `kqRWallFunction` is zero-gradient unconditionally; SU2 pins `k = 0` at the wall node
#: (`solution[0] = 0.0` plus `SetSolution_Old` / `LinSysRes.SetBlock_Zero` / `Jacobian.DeleteValsRowi`)
#: and switches to an algebraic `k = omega nu_t / rho` only when wall functions are enabled.
#:
#: The reason to try zero-gradient here is an INTERNAL INCONSISTENCY rather than either authority. The
#: wall-face `k` diffusivity is already faded to `(1 - f) gamma`, so on a wall-function cell the flux is
#: zero -- but the Dirichlet face value still enters the GRADIENT reconstruction (`grad_k` feeds `F1`'s
#: cross-diffusion, `omega_wall_gradient` and `OmegaCrossDiffusion`), which is not faded. So the model
#: says "no turbulent-energy flux to the wall" while the gradient says `k` falls to zero across half a
#: cell. Zero-gradient removes that disagreement, and unlike the deleted blended face value `f k_P` it
#: carries `d(phi_ip)/d(k_P) = 1` exactly, so it cannot become the `k`-amplifying wall face that failed
#: to converge.
#:
#: In the continuum the two conditions do NOT conflict: `u' ~ y`, `w' ~ y`, `v' ~ y**2` give `k ~ y**2`,
#: so `k -> 0` AND `dk/dy -> 0` at the wall, and the true diffusive wall flux is zero in both regimes.
#: The conflict is purely discrete -- a linear face reconstruction cannot satisfy both on one cell.
#: **The default is `zerogradient`**, on the internal-consistency argument above rather than on cost --
#: this mesh is in the wall-function regime on the walls that matter (the side walls sit at `y* ~ 34`,
#: fully in the log layer), which is where the faded-flux / unfaded-gradient disagreement bites.
#:
#: State the cost honestly, because it runs the other way and a reader comparing archived logs will hit
#: it. Measured as a controlled pair -- same code, same native trailing inverse, same 1e-08 floor, only
#: the wall condition differing:
#:
#:                       dirichlet   zerogradient
#:     wall                 1911 s         2124 s   (+11 %)
#:     steps                    59             67
#:     Krylov cycles           292            329
#:     mid-span x_r/h         8.36           8.36
#:
#: That is not a reason to select it. The two conditions solve DIFFERENT discrete problems, so their
#: step counts are not a like-for-like comparison of solver cost, and the cheaper one is cheaper for a
#: diagnosable
#: reason -- under `zerogradient` near-wall `k` is free to ratchet toward zero, and one numerically dead
#: cell then sets the global positivity cap for the whole march. Measured minimum cap over a whole
#: march: 2.02e-01 under `dirichlet`, against 1.44e-05 (native) and 5.41e-03 (host) under
#: `zerogradient`. The `K_POSITIVITY_FLOOR` below is what keeps that ratchet survivable.
_K_WALL_BCS = {"dirichlet": Dirichlet(0.0), "zerogradient": ZeroGradient()}
K_WALL = os.environ.get("BFS3D_K_WALL", "zerogradient")
if K_WALL not in _K_WALL_BCS:
    raise SystemExit(f"BFS3D_K_WALL={K_WALL!r} is not one of {sorted(_K_WALL_BCS)}")
K_WALL_BC = _K_WALL_BCS[K_WALL]

CONTROL = CflResidualDualTimeControl(
    beta_start=0.5, beta_min=0.005, grow=1.5, backoff=2.0, grow_above=0.5, backoff_below=0.25
)

_STEP_LIMIT_DUMPS = 0
#: The shift and the anchor of the step currently being taken. The limiter is called with only
#: ``(phi, delta)``, so without these a dump cannot be paired with the linear system that produced it --
#: and that is exactly what a reader needs to re-solve it. Captured from ``precondition_step``, which the
#: march calls once per attempt with the step (carrying its beta) and the state the attempt starts from.
_STEP_BETA, _STEP_ANCHOR = float("nan"), None


def _recording_precondition(refresh):
    """Wrap the preconditioner refresh so each attempt's shift and anchor are recorded for the dumps.

    The march calls ``precondition_step(step, state)`` once per attempt, after the control has set beta
    and before the step runs -- so it is the one place where both are in hand together. Delegates
    unchanged; only used when the step-limit dump is switched on.
    """

    def precondition(step, state):
        global _STEP_BETA, _STEP_ANCHOR
        _STEP_BETA = float(getattr(step.relaxation_schedule, "beta", float("nan")))
        _STEP_ANCHOR = np.asarray(state)
        return refresh(step, state)

    return precondition


def _save_step_limit(cap, phi, delta):
    """Write one ``(cap, iterate, correction)`` triple, if the cap is tight and the budget is unspent.

    Records the attempt's ``beta`` and ``anchor`` alongside, so the dump identifies the shifted system
    ``(J + beta*d) delta = -G(phi; anchor)`` rather than just its solution. Without them a reader can
    reproduce the *residual* at the iterate but not the *correction*, which is the half that matters --
    an attempt to identify beta by least squares from the dump alone was inconsistent across blocks
    (beta ~2.7 fitted on the k rows against ~130 on the omega rows, 52% residual).
    """
    global _STEP_LIMIT_DUMPS
    cap = float(cap)
    if cap >= DUMP_STEP_LIMIT or _STEP_LIMIT_DUMPS >= DUMP_STEP_LIMIT_KEEP:
        return
    path = HERE / "checkpoints" / f"step-limit-{_STEP_LIMIT_DUMPS:02d}.npz"
    np.savez(
        path,
        cap=cap,
        state=np.asarray(phi),
        delta=np.asarray(delta),
        beta=_STEP_BETA,
        anchor=_STEP_ANCHOR if _STEP_ANCHOR is not None else np.asarray(phi),
    )
    _STEP_LIMIT_DUMPS += 1
    print(f"[step-limit] cap {cap:.3e} beta {_STEP_BETA:.4g} -> {path.name}", flush=True)


@dataclasses.dataclass(frozen=True)
class _DumpingStepLimit:
    """A step limiter that saves the pair it was called on whenever it comes out tight.

    Wraps the real limiter and returns its cap unchanged, so the march it instruments takes exactly the
    steps it would have taken without it. A **frozen dataclass around the real limiter**, not a closure,
    for the reason the limiter itself is one: it rides in a static field of the compiled step, which is
    compared by ``__eq__``, and a function compares by identity -- so a closure here would make every
    rebuilt step a fresh compilation key and retrace the whole coupled solve.
    """

    inner: object

    def __call__(self, phi, delta):
        cap = self.inner(phi, delta)
        # Fires during execution (this runs inside the inner loop's `while_loop`) and is a no-op under
        # differentiation; the host side decides whether the cap is worth a file.
        jax.debug.callback(_save_step_limit, cap, phi, delta, ordered=True)
        return cap


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
    """The OpenFOAM comparison fields + 3D cell centres, keyed to their cell centroids.

    The **fields** come from the time-accurate run (:data:`TRANSIENT`); the steady case's limit-cycling
    fields are not a valid target and are deliberately not read here (see the module docstring). The
    **cell centres** come from the steady run's ``Ccx``/``Ccy``/``Ccz``: both cases are built from the
    same ``blockMeshDict``, so the mesh is identical and the centres are geometry rather than a solution.

    Returns
    -------
    dict
        ``centroid`` ``(n_cells, 3)``, ``U`` ``(n_cells, 3)``, and ``p``, ``k``, ``omega``, ``nut`` each
        ``(n_cells,)``, in the mesh's own cell order.
    """
    ccx, ccy, ccz = (_of_scalar(RUNS / f) for f in ("Ccx", "Ccy", "Ccz"))
    return dict(
        centroid=np.column_stack([ccx, ccy, ccz]),
        U=_of_vector(TRANSIENT / "U")[:, :3],
        p=_of_scalar(TRANSIENT / "p"),
        k=_of_scalar(TRANSIENT / "k"),
        omega=_of_scalar(TRANSIENT / "omega"),
        nut=_of_scalar(TRANSIENT / "nut"),
    )


def build_case(model=None, momentum_advection=None, gradient=None):
    """Assemble the benchmark: mesh, momentum, turbulence and the coupled residual -- no solve.

    Split out from :func:`solve_aquaflux` so a solver study can re-solve at a saved state without
    re-marching to it, and without restating the case. The mesh import, boundary conditions, model
    constants and scheme choices *are* the definition of this benchmark.

    Parameters
    ----------
    model : SSTModel, optional
        The SST constants to use. Defaults to :class:`~aquaflux.turbulence.SSTModel`.
    gradient : GradientScheme, optional
        The gradient reconstruction, for the same reason as ``momentum_advection``: its stencil is part
        of what the coloured probe has to cover. ``None`` (default) is the benchmark's own corrected
        Green-Gauss.
    momentum_advection : AdvectionScheme, optional
        The momentum face-value reconstruction, so a study can ask what the *scheme* costs -- its
        stencil reach sets how far the coloured Jacobian probe has to see. ``None`` (default) is the
        benchmark's own Venkatakrishnan-limited linear upwind; changing it changes the case, so a result
        measured with it must say so.

    Returns
    -------
    dict
        ``coupled``, ``momentum``, ``turbulence`` and ``geom`` for the assembled case.
    """
    if model is None:
        model = SSTModel()
    mesh = read_openfoam(RUNS / "polyMesh")
    geom = mesh.geometry()
    grad = CorrectedGreenGauss() if gradient is None else gradient
    # Second-order upwind momentum advection (Venkatakrishnan-limited linear upwind); first-order upwind
    # on the stiff k/omega scalars (a second-order stencil there lets the coupled Newton step drive omega
    # negative -- an M-matrix effect the limiter does not prevent).
    momentum_upwind = (
        LimitedUpwind(limiter=VenkatakrishnanLimiter())
        if momentum_advection is None
        else momentum_advection
    )
    scalar_upwind = FirstOrderUpwind()
    momentum = MomentumContinuity.build(
        mesh,
        geom,
        # `jnp.asarray`, not a bare float: the Reynolds continuation RESCALES this value per rung,
        # and a Python float is not a JAX array, so it rides on the static side of every jitted
        # function taking this assembler -- making each rung a fresh compilation key for the whole
        # coupled solve. As an array the rungs differ in a leaf VALUE and share the compilation.
        # Density is left a float: nothing rescales it, so its static value is the same every rung.
        PropertyModel({"viscosity": Constant(jnp.asarray(RHO * NU)), "density": Constant(RHO)}),
        grad,
        BoundaryConditions(
            {
                "inlet": VelocityInlet(velocity=(U_IN, 0.0, 0.0)),
                "outlet": PressureOutlet(pressure=0.0),
                "upperWall": NoSlipWall(),
                "lowerWall": NoSlipWall(),
                "sideWalls": NoSlipWall(),
            }
        ),
        advection_scheme=momentum_upwind,
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geom,
        grad,
        scalar_upwind,
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=WALLS,
        k_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(K_IN),
                "outlet": ZeroGradient(),
                **dict.fromkeys(WALLS, K_WALL_BC),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "inlet": Dirichlet(OMEGA_IN),
                "outlet": ZeroGradient(),
                "upperWall": ZeroGradient(),
                "lowerWall": ZeroGradient(),
                "sideWalls": ZeroGradient(),
            }
        ),
    )
    # Log-transform omega (positive under any Newton step); k stays direct (log(k) is ill-conditioned
    # where k -> 0 at the walls).
    coupled = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    return dict(coupled=coupled, momentum=momentum, turbulence=turbulence, geom=geom)


def solve_aquaflux(*, log_path=None, checkpoint_dir=None, **solve_kwargs):
    """Solve the coupled RANS system on the imported OpenFOAM mesh; return fields + geometry.

    **This is the full continuation solve the case's published result comes from** -- Reynolds
    continuation onto the target Reynolds number, the measured preconditioner bundle, the dual-time
    Courant control, and the beta-tracking refresh. The target Reynolds number is **not reachable
    without the ramp**: a direct cold start diverges on its first step (the residual grows by ~100
    orders of magnitude at a line-search factor already clipped to 0.002), so the ramp buys
    reachability, not merely a better seed.

    Preconditioned by the **field-split algebraic-multigrid V-cycle** -- the 3D coupled path, since the
    complete LU's fill is a memory wall past ~10^4 3D cells and the threshold-ILU's factorization is
    prohibitively slow to build on the distance-3 3D coupled Jacobian. Every non-default setting is a
    measurement rather than a preference; see the constants above and the README.

    Parameters
    ----------
    log_path : str or Path, optional
        Stream the march here, one framed block per step. Pass a path when running interactively -- a
        march of this length must be readable while it runs, not after it ends.
    checkpoint_dir : str or Path, optional
        Write a rolling state checkpoint per step here. Worth setting for any long run: without it a
        failure at the last step discards every step before it.
    **solve_kwargs
        Forwarded to :func:`~aquaflux.turbulence.solve_reynolds_continuation`, overriding the defaults.
    """
    case = build_case()
    coupled, momentum, turbulence, geom = (
        case["coupled"],
        case["momentum"],
        case["turbulence"],
        case["geom"],
    )
    log_file = open(log_path, "w") if log_path is not None else None
    # Each rung rebuilds both the case (its viscosity is rescaled) and its continuation, so the
    # per-equation residual reporter has to be rebuilt with them. The logger holds one callable, which
    # defers to whichever reporter the rung currently being marched installed.
    rung_residuals: list = []

    def residuals(state):
        return rung_residuals[-1](state) if rung_residuals else {}

    logger = MarchLogger(
        log_file,
        metrics=reattachment_metrics(case),
        fields=coupled_fields(coupled),
        residuals=residuals,
        detail=("inner", "fields", "residuals", "pc"),
        rtol=RTOL,
        atol=ATOL,
    )
    # Every run states the configuration it was taken under, in its own log, before any result. A march
    # log is compared against archived ones months later, and a number whose configuration is not written
    # beside it cannot be trusted OR cheaply re-adjudicated -- it is unfalsifiable, which is worse than
    # wrong, because a wrong finding gets corrected and an unfalsifiable one gets cited. This is not
    # hypothetical bookkeeping: two preconditioner arms were compared against an archived baseline whose
    # refresh trigger differed from theirs, and the resulting "the smoother is 1.85x slower" reading was
    # entirely the trigger. Neither log said what it had been run with, so nothing caught it until the
    # per-step refresh counts were dug out by hand.
    logger.note("[configuration]")
    for name, value in (
        ("field split", FIELD_SPLIT),
        # WHICH INVERSE the trailing block gets, before how it is smoothed -- a setting that once cost
        # a finding. Two runs differing only in this wrote banners identical to the character, so
        # afterwards the only way to tell them apart was the launch order and the text of the exception
        # each happened to die on. One of them had quietly reproduced the reference trajectory for four
        # steps and that went unnoticed. A configuration line is worth nothing if it omits the variable
        # under test.
        (
            "flow inverse",
            FLOW_INVERSE if LEADING_INVERSE is None else f"{FLOW_INVERSE} {_NATIVE_FLOW}",
        ),
        ("turbulence inverse", TURBULENCE_INVERSE),
        # ...and, when a `trailing_inverse` is supplied, it REPLACES the PETSc V-cycle wholesale, so the
        # two smoother settings below are never read. Marking them is the same rule as the note above:
        # a banner that prints a setting the run did not use is worse than one that omits it, because a
        # reader diffing two runs attributes a difference to a line that was dead in both.
        (
            "turbulence smoother",
            f"{_TURBULENCE_SMOOTHER or 'ilu0 (shipped)'}{_TRAILING_SMOOTHER_NOTE}",
        ),
        ("turbulence smoother sweeps", f"{TRAILING_SWEEPS}{_TRAILING_SMOOTHER_NOTE}"),
        *(
            [("native trailing settings", _native_trailing_description())]
            if TURBULENCE_INVERSE == "native"
            else []
        ),
        (
            "probe column reach",
            "uniform 3" if COLUMN_REACH is None else "/".join(map(str, COLUMN_REACH)),
        ),
        ("refresh on cycles", REFRESH_ON_CYCLES or "scheduled cadence"),
        # Beside the cost trigger, because the two together are what decides when the V-cycle is
        # rebuilt, and a run that re-matches on a β escalation is a different arm from one that does not.
        ("refresh on beta mismatch", "off" if REFRESH_ON_BETA == float("inf") else REFRESH_ON_BETA),
        ("Reynolds continuation points", N_POINTS),
        ("forward restart", FORWARD_RESTART),
        ("retry on cycles / alpha", f"{RETRY_ON_CYCLES} / {RETRY_ON_ALPHA}"),
        ("cycle budget", CYCLE_BUDGET),
        ("smoother fill / sweeps / coarse limit", f"{FILL_LEVELS} / {SWEEPS} / {COARSE_EQ_LIMIT}"),
        ("preconditioner beta floor", PC_BETA_FLOOR),
        ("stop (rtol, atol)", f"{RTOL}, {ATOL}"),
        ("k wall BC", K_WALL),
        ("k positivity floor", K_POSITIVITY_FLOOR or "0 (plain rule)"),
        # Beside the floor, because the two are alternative answers to the same failure and a run
        # carrying the projection is a different arm from one carrying only a floor.
        (
            "k positivity projection",
            "per-cell clip" if K_POSITIVITY_PROJECTION else "off (cap only)",
        ),
        ("inner forward rtol (row-scaled)", FORWARD_RTOL),
        # Printed because a banner diff is only a CONFIG diff if every knob that installs a wrapper or
        # changes what is retained appears in it. These three were read and never shown, so two runs
        # could differ in them and produce banners identical to the character.
        ("checkpoint keep", CHECKPOINT_KEEP),
        ("inner dump above", INNER_DUMP_ABOVE or "off"),
        ("trailing block dump", DUMP_TRAILING_BLOCK or "off"),
        *(
            [("step-limit dump below / keep", f"{DUMP_STEP_LIMIT} / {DUMP_STEP_LIMIT_KEEP}")]
            if DUMP_STEP_LIMIT
            else []
        ),
    ):
        logger.note(f"  {name}: {value}")

    checkpoints = (
        StateCheckpointer(checkpoint_dir, every=1, keep=CHECKPOINT_KEEP)
        if checkpoint_dir is not None
        else None
    )
    on_checkpoint = (
        logger.on_checkpoint
        if checkpoints is None
        else combine_observers(logger.on_checkpoint, checkpoints.on_checkpoint)
    )
    inner_dump = (
        InnerIterateCheckpointer(checkpoint_dir, above=INNER_DUMP_ABOVE)
        if INNER_DUMP_ABOVE and checkpoint_dir is not None
        else None
    )

    # Everything below is built ONCE and shared by every continuation rung, and each piece is here rather
    # than inside `point_setup` for the same reason: a rung rebuilding it would hand the compiled coupled
    # solve a new object in a STATIC field, which is compared by identity, so the whole solve recompiles.
    #
    # That is the largest fixed overhead a continuation march carries, and it is self-diagnosing in this
    # case's own log: each rung's FIRST step is that rung's most expensive step by a wide margin while
    # its Krylov cycle count is no higher than its cheap steps' -- cost with no linear algebra behind it
    # is compilation. Only the first rung's is unavoidable.
    #
    # The probe (colouring plan + gather map) depends on the mesh alone, so a rung was additionally
    # building the single largest allocation this case makes -- twice, once for the engine and once for
    # the refresh hook beside it.
    probe = CoupledJacobianProbe.build(coupled, column_reach=COLUMN_REACH)
    # With the cycle trigger on, the scheduled cadences are switched OFF so it REPLACES them: as an
    # addition it is break-even, as a replacement it is the largest saving measured on this march.
    # CAREFUL: `beta_rel_change=None` does NOT switch the schedule off -- it removes the gate, and a
    # missing gate means "refresh every step". Switching it off means a gate that exists and never
    # fires again after its first (initialising) call, plus no materialize gates, so the refresh
    # branch resolves to `none`.
    scheduled = not REFRESH_ON_CYCLES
    refresh = amg_beta_tracking_refresh(
        coupled,
        probe=probe,
        beta_rel_change=0.25 if scheduled else REFRESH_ON_BETA,
        refresh_every=8 if scheduled else 10**9,
        materialize_drift=0.05 if scheduled else None,
        materialize_every=4 if scheduled else None,
        beta_floor=PC_BETA_FLOOR,
        observer=logger.on_refresh,
    )
    # A fresh `combine_observers` closure per rung would be its own recompile -- it lands in the step's
    # static `inner_observer` and functions compare by identity.
    inner_observer = combine_observers(
        logger.on_inner,
        *([inner_dump.on_inner] if inner_dump is not None else []),
    )
    #: The one V-cycle every rung shares, held here so `point_setup` can hand it back to the next rung.
    shared_preconditioner: list = []

    def point_setup(companion, seed_state, point):
        """Configure each rung, REUSING one preconditioner and one refresh hook across the whole ramp.

        Only the molecular viscosity changes between rungs, so a rung needs its own residual assembler
        and its own row scales -- both of which ride as ordinary data and cost nothing to rebuild -- but
        it does not need its own V-cycle. It needs that V-cycle *fitted to it*, which is a different
        thing and is what `rebind` arranges: the shared hook is pointed at this rung's companion and
        forced to re-materialize at this rung's own state and shift, which the march does before the
        rung's first step. So each rung still solves with a V-cycle built for its own problem, and the
        compiled coupled solve is a cache hit across the boundary instead of a full recompile.

        The distinction to hold on to is between FITTING the preconditioner per rung, which is real and
        still happens, and rebuilding the *object*, which only costs a compilation.
        """
        logger.note(f"[{point.label}]")
        # Point the shared hook at this rung. Called for EVERY rung including the first, so the V-cycle
        # is always re-materialized at the rung's own state and shift before its first solve -- the first
        # rung's build freezes at `amg_beta`, and the march's own beta_start is a different value.
        refresh.rebind(companion)
        engine = coupled_amg_continuation(
            companion,
            seed_state,
            inner_steps=INNER_STEPS,
            inner_tol=INNER_TOL,
            probe=probe,
            preconditioner=shared_preconditioner[0] if shared_preconditioner else None,
            smoother_fill_levels=FILL_LEVELS,
            smoother_sweeps=SWEEPS,
            coarse_eq_limit=COARSE_EQ_LIMIT,
            cycle_budget=round(CYCLE_BUDGET * _RESTART_SCALE),
            positivity_floor=K_POSITIVITY_FLOOR,
            positivity_projection=K_POSITIVITY_PROJECTION,
            forward_rtol=FORWARD_RTOL,
            forward_restart=FORWARD_RESTART,
            inner_observer=inner_observer,
            refresh_on_cycles=REFRESH_ON_CYCLES or None,
            inner_refresh=refresh.refresh_at if REFRESH_ON_CYCLES else None,
            field_split=FIELD_SPLIT,
            trailing_smoother_sweeps=TRAILING_SWEEPS,
            trailing_options=TRAILING_OPTIONS if FIELD_SPLIT else None,
            leading_inverse=LEADING_INVERSE if FIELD_SPLIT else None,
            trailing_inverse=TRAILING_INVERSE if FIELD_SPLIT else None,
        )
        shared_preconditioner[:] = [engine.shift_policy.preconditioner]
        if DUMP_STEP_LIMIT and engine.step_limit is not None:
            # `dataclasses.replace`, not `eqx.tree_at`: the limiter is a STATIC field, so it lives in the
            # treedef rather than among the leaves and `tree_at` (which addresses leaves) cannot reach it.
            engine = dataclasses.replace(engine, step_limit=_DumpingStepLimit(engine.step_limit))
        # Seeded with this rung's own starting state: the march equilibrates each step at the state it
        # begins from, so without the seed the rung's first step would be scaled at its end state and
        # its per-equation rows would not add up to the residual reported beside them.
        rung_residuals.append(coupled_residuals(companion, engine, seed_state))
        return dict(
            continuation=engine,
            precondition_step=_recording_precondition(refresh) if DUMP_STEP_LIMIT else refresh,
        )

    options = (
        dict(
            max_steps=MAX_STEPS,
            rtol=RTOL,
            atol=ATOL,
            intermediate_rtol=None,  # every rung stops at the same ABSOLUTE bar
            intermediate_atol=ATOL,
            step_control=CONTROL,
            point_setup=point_setup,
            scaled_norm=True,  # rebuild the row scales each outer step
            retry_solver=relative_residual_gmres(1e-4, restart=40),
            on_checkpoint=on_checkpoint,
            on_retry=logger.on_retry,
            retry_on_cycles=round(RETRY_ON_CYCLES * _RESTART_SCALE),
            retry_on_alpha=RETRY_ON_ALPHA,
            retry_beta_factor=RETRY_BETA_FACTOR,
        )
        | solve_kwargs
    )
    try:
        flow, k, omega = solve_reynolds_continuation(coupled, N_POINTS, **options)
    finally:
        if log_file is not None:
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


def mid_span_slab(centroid, half_width=0.06):
    """The thin spanwise slab about ``z = W/2`` that the primary reattachment metric is measured in.

    The **full-span** reattachment length is not the primary bubble: the side walls carry their own
    corner separation that reattaches much further downstream, and because
    :func:`reattachment_length` takes the *last* reversed wall cell, those corner cells set the
    full-span number on their own. On the reference field the two outermost spanwise slabs read
    ``x_r/h = 10.28`` while all six interior slabs read ``7.24`` -- so a full-span number overstates
    the primary bubble by ~40% while being a perfectly good measurement of something else.

    Parameters
    ----------
    centroid : ndarray (n_cells, 3)
        Cell centroids.
    half_width : float
        Half-thickness of the slab as a fraction of the span. Default ``0.06``.

    Returns
    -------
    tuple(float, float)
        The ``(z_lo, z_hi)`` slab, ready for :func:`reattachment_length`'s ``z_slab``.
    """
    z = centroid[:, 2]
    width = z.max() - z.min()
    centre = z.min() + 0.5 * width
    return (centre - half_width * width, centre + half_width * width)


def reattachment_metrics(case):
    """A ``state -> {"xr/h": ..., "xr/h_full": ...}`` metrics callable for ``MarchLogger``.

    The march logger reports everything a solver step knows, but the quantity this case is actually
    steered by -- the reattachment length -- needs the case's own geometry and field layout. This is
    that seam, so a driver logging the march never re-derives the unpacking.

    Reports the **mid-span** length as ``xr/h``, the same measurement the run's final comparison makes,
    so the number a march is watched by and the number it is judged by cannot diverge. The full-span
    length rides alongside as ``xr/h_full`` because the gap between them *is* the corner separation
    (see :func:`mid_span_slab`) -- worth seeing, but not the primary bubble.

    Parameters
    ----------
    case : dict
        The assembled benchmark from :func:`build_case`.

    Returns
    -------
    callable
        ``state -> mapping``, mapping a packed coupled state to reattachment lengths in step heights.
    """
    coupled, momentum, geom = case["coupled"], case["momentum"], case["geom"]
    centroid = np.asarray(geom.cell.centroid)
    slab = mid_span_slab(centroid)

    def metrics(state):
        flow, _k, _omega = coupled.physical_fields(state)
        velocity, _pressure = momentum.unpack(flow)
        u_x = np.asarray(velocity)[:, 0]
        return {
            "xr/h": float(reattachment_length(centroid, u_x, z_slab=slab)),
            "xr/h_full": float(reattachment_length(centroid, u_x)),
        }

    return metrics


def reattachment_length(centroid, u_x, z_slab=None):
    """Lower-wall reattachment length x_r/h behind the step (h = step height).

    Reads the sign of the wall-adjacent streamwise velocity along the lower wall (y just above the -h
    floor) downstream of the step: the recirculation bubble is where it is negative, and reattachment is
    the last such x.

    Parameters
    ----------
    centroid : ndarray (n_cells, 3)
        Cell centroids.
    u_x : ndarray (n_cells,)
        Streamwise velocity.
    z_slab : tuple(float, float), optional
        Restrict to a spanwise slab ``z in [z_lo, z_hi]`` (e.g. a mid-span slice). ``None`` uses the full
        span, which averages the (spanwise-varying) bubble.
    """
    x, y, z = centroid[:, 0], centroid[:, 1], centroid[:, 2]
    band = (x > 1e-4) & (y < -H + 0.002) & (y > -H)
    if z_slab is not None:
        band &= (z >= z_slab[0]) & (z <= z_slab[1])
    xs, us = x[band], u_x[band]
    order = np.argsort(xs)
    xs, us = xs[order], us[order]
    neg = np.where(us < 0)[0]
    if neg.size == 0:
        return 0.0
    return float(xs[neg[-1]] / H)


def spanwise_reattachment(centroid, u_x, n_bins=8):
    """Reattachment length x_r/h in each of ``n_bins`` spanwise slabs -- the 3D structure the 2D case
    cannot show. Returns the per-slab (z_centre, x_r/h) so the writeup can report the mid-span value and
    the spanwise variation (corner vs centre)."""
    z = centroid[:, 2]
    z_lo, z_hi = z.min(), z.max()
    edges = np.linspace(z_lo, z_hi, n_bins + 1)
    out = []
    for i in range(n_bins):
        xr = reattachment_length(centroid, u_x, z_slab=(edges[i], edges[i + 1]))
        out.append((0.5 * (edges[i] + edges[i + 1]), xr))
    return out


def _fresh_log(path: Path) -> Path:
    """Move any existing march log aside, so a new run cannot destroy the one it is compared against.

    The log is the only per-step record a run leaves -- cycle counts, betas, retries, the per-equation
    residual grid -- and every arm comparison this case exists for is a comparison of two of them. It
    used to be written to one fixed path, so starting a run silently deleted the baseline: a
    native-preconditioner arm was measured against aggregates alone for exactly this reason, and the
    step-by-step comparison that would have been most informative could not be made at all.

    The previous log is renamed with the modification time it already carried, not the current time, so
    the archived name says when that run happened rather than when it was displaced.
    """
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(path.stat().st_mtime))
        archived = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
        path.rename(archived)
        print(f"archived the previous march log to {archived.name}", flush=True)
    return path


def main():
    if not (RUNS / "polyMesh").exists():
        raise SystemExit(f"OpenFOAM mesh not found in {RUNS}; run of_case/run_of.sh first.")
    if not (TRANSIENT / "U").exists():
        raise SystemExit(
            f"Transient reference not found in {TRANSIENT}; run of_transient/run_transient.sh first."
        )
    of = read_openfoam_reference()
    print(
        f"OpenFOAM: {of['centroid'].shape[0]} cells, Ux in "
        f"[{of['U'][:, 0].min():.3f}, {of['U'][:, 0].max():.3f}], "
        f"Uz in [{of['U'][:, 2].min():.3f}, {of['U'][:, 2].max():.3f}] (3D)",
        flush=True,
    )

    t0 = time.time()
    aq = solve_aquaflux(
        log_path=_fresh_log(HERE / "march.log"), checkpoint_dir=HERE / "checkpoints"
    )
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

    slab = mid_span_slab(aq["centroid"])

    metrics = dict(
        ux=rel_l2(aq["U"][:, 0], of["U"][idx, 0], U_IN),
        uy=rel_l2(aq["U"][:, 1], of["U"][idx, 1], U_IN),
        uz=rel_l2(aq["U"][:, 2], of["U"][idx, 2], U_IN),
        xr_aqua_mid=reattachment_length(aq["centroid"], aq["U"][:, 0], z_slab=slab),
        xr_of_mid=reattachment_length(of["centroid"], of["U"][:, 0], z_slab=slab),
        nut_peak_aqua=float(aq["nut"].max() / NU),
        nut_peak_of=float(of["nut"].max() / NU),
    )
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}", flush=True)
    print("  spanwise x_r/h (z, aquaflux, OpenFOAM):", flush=True)
    aq_span = spanwise_reattachment(aq["centroid"], aq["U"][:, 0])
    of_span = spanwise_reattachment(of["centroid"], of["U"][:, 0])
    for (zc_i, xa), (_, xo) in zip(aq_span, of_span, strict=True):
        print(f"    z={zc_i:.4f}  aqua {xa:.2f}  OF {xo:.2f}", flush=True)

    _figure(of, aq, idx, slab)
    _report(metrics, aq_span, of_span)


def _figure(of, aq, idx, slab):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    x, y, z = aq["centroid"][:, 0], aq["centroid"][:, 1], aq["centroid"][:, 2]
    # Mid-span slice for the 2D-style Ux comparison.
    mid = (z >= slab[0]) & (z <= slab[1])
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    vmax = max(abs(of["U"][:, 0]).max(), abs(aq["U"][:, 0]).max())
    for a, data, title in (
        (ax[0], of["U"][idx, 0][mid], "OpenFOAM $U_x$ (mid-span)"),
        (ax[1], aq["U"][:, 0][mid], "aquaflux $U_x$ (mid-span)"),
    ):
        sc = a.scatter(x[mid], y[mid], c=data, s=6, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set_title(title)
        a.set_aspect("equal")
        fig.colorbar(sc, ax=a, shrink=0.8)
    band = (x > 1e-4) & (y < -H + 0.002) & (y > -H) & mid
    order = np.argsort(x[band])
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].plot(
        x[band][order] / H,
        of["U"][idx][band][order][:, 0],
        "s-",
        ms=3,
        color="C1",
        label="OpenFOAM",
    )
    ax[2].plot(
        x[band][order] / H, aq["U"][band][order][:, 0], ".-", ms=4, color="C0", label="aquaflux"
    )
    ax[2].set_xlabel("$x/h$ behind the step")
    ax[2].set_ylabel("near-wall $U_x$ (mid-span)")
    ax[2].set_title("Lower-wall recirculation (sign change = reattachment)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "comparison.png", dpi=130)
    print(f"wrote {FIGS / 'comparison.png'}", flush=True)


def _report(m, aq_span, of_span):
    span_rows = "\n".join(
        f"| {za:.4f} | {xa:.2f} | {xo:.2f} |"
        for (za, xa), (_, xo) in zip(aq_span, of_span, strict=True)
    )
    lines = [
        "# 3D backward-facing step: aquaflux coupled k-omega SST vs OpenFOAM k-omega SST",
        "",
        "A finite-width 3D backward-facing step (step height h = 0.01 m, expansion ratio 2, span 4h",
        "between no-slip side walls), solved by OpenFOAM `incompressibleFluid` (kOmegaSST) and, on the",
        "**same imported mesh**, by aquaflux's coupled RANS solver (hybrid initialization, second-order",
        "upwind momentum advection, corrected Green-Gauss gradients, log-omega, field-split",
        "algebraic-multigrid preconditioner). U_in = 10 m/s, nu = 1e-5 -> Re_h = 10000.",
        "",
        "## Results",
        "",
        "| quantity | aquaflux | OpenFOAM |",
        "|---|---|---|",
        f"| reattachment length x_r/h (mid-span) | {m['xr_aqua_mid']:.2f} | {m['xr_of_mid']:.2f} |",
        f"| peak nu_t/nu | {m['nut_peak_aqua']:.0f} | {m['nut_peak_of']:.0f} |",
        f"| rel. L2 U_x error (cell-for-cell) | {m['ux']:.3f} | -- |",
        f"| rel. L2 U_y error (cell-for-cell) | {m['uy']:.3f} | -- |",
        f"| rel. L2 U_z error (cell-for-cell) | {m['uz']:.3f} | -- |",
        "",
        "### Spanwise reattachment (the 3D structure)",
        "",
        "| z (m) | aquaflux x_r/h | OpenFOAM x_r/h |",
        "|---|---|---|",
        span_rows,
        "",
        "See `figures/comparison.png`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "# 1. OpenFOAM kOmegaSST reference (needs the openfoam13 image)",
        "cd validation/bfs3d_openfoam",
        'docker run --rm -v "$PWD":/work -w /work/of_case openfoam13:latest bash run_of.sh',
        'docker run --rm -v "$PWD":/work -w /work/of_transient openfoam13:latest bash run_transient.sh',
        "# 2. aquaflux coupled solve + comparison (from the repo root)",
        "cd ../..",
        "python3 validation/bfs3d_openfoam/compare.py",
        "```",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
