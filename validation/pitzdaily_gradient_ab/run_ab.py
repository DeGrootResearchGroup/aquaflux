"""Corrected Green-Gauss against the Hessian-corrected (Betchen) gradient, head to head.

A second version of the pitzDaily case whose only purpose is to compare the two gradient
reconstructions on identical physics: same mesh, same boundary conditions, same model constants, same
march settings, same Reynolds continuation. Only the reconstruction differs. It reports what the
upgrade **costs** (wall clock, outer steps, Krylov cycles) and what it **changes** (the converged
fields, and the reattachment length the benchmark is judged by).

The case definition is imported from the validated benchmark next door rather than restated here --
that mesh, those boundary conditions and those model constants *are* the case, and a second copy of
them would drift from the one the validation figures were taken under. What this case does supply for
itself is the **preconditioner**, and that is the reason it is a separate case rather than a flag on
the other one.

Why a different preconditioner
------------------------------
The benchmark next door preconditions from a **coloured Jacobian probe**: it recovers the Jacobian out
to a fixed cell-graph distance, and coupling beyond that distance is folded onto near entries rather
than dropped, because a colouring is collision-free only for the pattern it was built at. Its reach of
5 is exact for corrected Green-Gauss, whose Jacobian carries *exactly zero* mass past it. The
Hessian-corrected residual has no such cut-off -- its gradient couples to the Hessian, which couples to
the neighbours' Hessians -- so it reaches essentially across the mesh, and a probe sized for the other
scheme would hand that arm a corrupted operator. Comparing a sound preconditioner against a corrupted
one measures the probe, not the reconstruction.

So both arms run the **field-split stack the 3D backward-facing-step case uses**: a SIMPLE-smoothed
multigrid hierarchy on the leading flow saddle and a nodal hierarchy on the trailing scalars. That
family is far more tolerant of an inexact probe than an incomplete factorization is, because it uses
the blocks to form an approximate Schur complement and apply V-cycles rather than eliminating them
into pivots -- a perturbed far-field entry degrades a V-cycle's rate, where it can wreck a
factorization. Both arms get the identical preconditioner and the identical reach, so whatever remains
between them is the reconstruction.

⚠️ **Read the cycle counts before the wall clock.** If the two arms' Krylov cycle counts are close, the
wall-clock difference is the reconstruction and the comparison means what it says. If the Betchen arm's
cycles are much higher, the preconditioner is not seeing that arm's operator well enough and the reach
is the first thing to raise (``PITZ_AB_REACH``), for **both** arms together -- raising it for one would
compare two different preconditioners.

Usage
-----
    validation/run_case.sh validation/pitzdaily_gradient_ab/run_ab.py

    PITZ_AB_REACH=7 validation/run_case.sh validation/pitzdaily_gradient_ab/run_ab.py
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[0] / "pitzdaily_openfoam"))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare  # noqa: E402  (the validated benchmark: mesh, physics, metric)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.schemes import (  # noqa: E402
    CorrectedGreenGauss,
    GmresGradientSolve,
    HessianCorrectedGradient,
    SweptGradientSolve,
)
from aquaflux.solve import (  # noqa: E402
    MarchLogger,
    RefreshPolicy,
    jacobi_smoothed_inverse,
    simple_smoothed_inverse,
)
from aquaflux.turbulence import (  # noqa: E402
    CoupledJacobianProbe,
    amg_beta_tracking_refresh,
    coupled_amg_continuation,
    coupled_fields,
    solve_reynolds_continuation,
)

#: Both arms probe at the same reach. Raising it for one arm alone would compare two preconditioners.
#:
#: **5, and it must not be shortened for this case — the reason is the whole point of the case.**
#:
#: A short reach looked safe here and is not. A capped march varying only the reach gave BIT-IDENTICAL
#: cycles at 2, 3 and 5 while the probe cost scaled steeply (100, 165, 380 residual evaluations per
#: refresh) -- but that sweep ran the **standard** gradient, whose Jacobian carries *exactly zero* mass
#: past its own reach. It could not have detected a reach effect on a long-stencil reconstruction, so
#: reading it as a property of the preconditioner was wrong. Measured on full marches of the Betchen
#: arm itself:
#:
#:     probe reach 3   834 cycles / 74 steps   11.27 per step   max 36   1715 s
#:     probe reach 5   511 cycles / 72 steps    7.10 per step   max 20   1604 s
#:
#: A third fewer cycles, and the cycle ratio against the standard arm falls from 1.93x to 1.18x -- so
#: at reach 3 the probe was under-resolving that arm's stencil and the cost gap was being charged to
#: the reconstruction. ⚠️ The wall clock barely moves (the longer probe eats the cycle saving), so a
#: longer reach buys a **comparable measurement**, not a faster march; do not read it as a speedup.
#:
#: ``PITZ_AB_REACH_SWEEP`` re-measures on any configuration -- but sweep it with the arm whose stencil
#: is in question, not with the one that cannot feel it.
REACH = int(os.environ.get("PITZ_AB_REACH", "5"))

#: Uniform, deliberately. The sibling 3D case shortens two of its six columns; that value was measured
#: on that mesh, those schemes and a SIX-field layout, and the analogous five-field value here is
#: unmeasured. Uniform costs more probes and is always correct.
COLUMN_REACH = None

#: The leading (flow saddle) inverse, at the 3D case's own measured settings. `sweeps=2` rather than the
#: 4 that single-state probes prefer: on a march the apply cost dominates enough that buying cheapness
#: with convergence pays there. Whether that carries to this mesh is not established -- it is a starting
#: point taken from a measured configuration, not a calibration for this case.
SIMPLE_FLOW = dict(
    sweeps=2,
    pressure_sweeps=2,
    strength_threshold=0.25,
    avoid_singletons=True,
    aggressive_levels=0,
    max_levels=5,
    max_coarse=500,
    block_splitting=True,
    omega=1.0,
    frozen_coarsening=True,
)
JACOBI_TRAILING = dict(max_coarse=500, equilibrate=False)

#: The Betchen arm's two sweep counts. **Fixed sweeps on BOTH systems, not the class's Krylov default**,
#: and the reason is cost on the path a march actually pays. Profiled on this mesh (jitted, warm, min of
#: 7), per reconstruction against corrected Green-Gauss as 1.0x:
#:
#:     outer swept-10 / inner 10   fwd 33.3x   jvp 31.5x   jvp/fwd 1.0x
#:     outer GMRES    / inner 10   fwd 92.9x   jvp 158.6x  jvp/fwd 1.8x   <- the class default
#:
#: A Krylov outer solve is differentiated by the implicit function theorem, so **every Jacobian-vector
#: product solves an entire second Schur system**, each with its own inner solve inside every iteration
#: -- that is the 1.8x, on top of an already dearer forward pass. A fixed sweep differentiates by
#: unrolling and costs 1.0x. Since a march pays a jvp per Krylov iteration, the swept outer is ~5x
#: cheaper there and reaches the same gradient to 1e-12.
#:
#: Cost is linear in the outer count and the inner solve dominates each outer sweep, so 5/5 is roughly a
#: quarter of 10/10. What that trades away is exactness: the inner count sets how faithfully the applied
#: operator matches the true Schur complement, and the outer count how far that system is solved.
#: ``run_ab.py --check-accuracy`` (or ``PITZ_AB_CHECK=1``) measures the pair against an exactly-solved
#: reconstruction on this mesh before the comparison is trusted.
OUTER_SWEEPS = int(os.environ.get("PITZ_AB_OUTER_SWEEPS", "5"))
INNER_SWEEPS = int(os.environ.get("PITZ_AB_INNER_SWEEPS", "5"))

BETCHEN = HessianCorrectedGradient(
    solver=SweptGradientSolve(sweeps=OUTER_SWEEPS, warn_tol=None),
    hessian_solver=SweptGradientSolve(sweeps=INNER_SWEEPS, warn_tol=None),
)

ARMS = (
    ("standard", "CorrectedGreenGauss", CorrectedGreenGauss()),
    (
        "betchen",
        f"HessianCorrectedGradient (outer swept-{OUTER_SWEEPS} / inner swept-{INNER_SWEEPS})",
        BETCHEN,
    ),
)

#: Compared between the arms. `nut` is included because a turbulence case feels a gradient change most
#: in the eddy viscosity, which is built from velocity gradients.
FIELDS = ("U", "p", "k", "omega", "nut")


def solve_arm(gradient_scheme, log_path, *, reach=None, points=None, max_steps=None):
    """March one arm to convergence and return its fields, wall clock and cycle count.

    Mirrors the benchmark's own march settings -- stopping tolerances, step control, retry ladder,
    Reynolds continuation -- and replaces only the preconditioner, so the two arms differ from the
    benchmark in one way and from each other in one way.
    """
    reach = REACH if reach is None else reach
    points = compare.N_POINTS if points is None else points
    max_steps = compare.MAX_STEPS if max_steps is None else max_steps
    case = compare.build_case(gradient_scheme=gradient_scheme)
    coupled, momentum, turbulence, geom = (
        case["coupled"],
        case["momentum"],
        case["turbulence"],
        case["geom"],
    )
    log_file = open(log_path, "w")
    logger = MarchLogger(
        log_file,
        fields=coupled_fields(coupled),
        detail=("inner", "fields", "pc"),
        rtol=compare.RTOL,
        atol=compare.ATOL,
    )
    logger.note("[configuration]")
    for name, value in (
        ("gradient scheme", type(gradient_scheme).__name__),
        ("probe stencil reach", reach),
        ("probe column reach", COLUMN_REACH or "uniform"),
        ("leading (flow) inverse", f"simple_smoothed {SIMPLE_FLOW}"),
        ("trailing inverse", f"jacobi_smoothed {JACOBI_TRAILING}"),
        ("host ILU kernel", "compiled" if compare.ILU0_COMPILED else "PURE PYTHON (timings void)"),
        ("Reynolds continuation points", points),
        ("stop (rtol, atol)", f"{compare.RTOL}, {compare.ATOL}"),
    ):
        logger.note(f"  {name}: {value}")

    probe = CoupledJacobianProbe.build(coupled, stencil_reach=REACH, column_reach=COLUMN_REACH)
    refresh = amg_beta_tracking_refresh(
        coupled,
        probe=probe,
        beta_rel_change=float("inf"),
        refresh_every=10**9,
        materialize_drift=None,
        materialize_every=None,
        beta_floor=compare.PC_BETA_FLOOR,
        observer=logger.on_refresh,
    )
    shared: list = []

    def point_setup(companion, seed_state, point):
        logger.note(f"[{point.label}]")
        refresh.rebind(companion)
        engine = coupled_amg_continuation(
            companion,
            seed_state,
            inner_steps=compare.INNER_STEPS,
            inner_tol=compare.INNER_TOL,
            probe=probe,
            cycle_budget=compare.CYCLE_BUDGET,
            forward_rtol=compare.FORWARD_RTOL,
            forward_restart=compare.FORWARD_RESTART,
            forward_max_restarts=compare.FORWARD_MAX_RESTARTS,
            refresh_on_cycles=compare.REFRESH_ON_CYCLES or None,
            inner_refresh=refresh.refresh_at if compare.REFRESH_ON_CYCLES else None,
            positivity_floor=compare.K_POSITIVITY_FLOOR,
            positivity_projection=compare.POSITIVITY_PROJECTION,
            preconditioner=shared[0] if shared else None,
            coarse_eq_limit=SIMPLE_FLOW["max_coarse"],
            field_split=True,
            leading_inverse=simple_smoothed_inverse(**SIMPLE_FLOW),
            trailing_inverse=jacobi_smoothed_inverse(**JACOBI_TRAILING),
            inner_observer=logger.on_inner,
        )
        shared[:] = [engine.shift_policy.preconditioner]
        return dict(continuation=engine, refresh=RefreshPolicy(precondition_step=refresh))

    started = time.perf_counter()
    try:
        flow, k, omega = solve_reynolds_continuation(
            coupled,
            points,
            max_steps=max_steps,
            rtol=compare.RTOL,
            atol=compare.ATOL,
            intermediate_rtol=None,
            intermediate_atol=compare.ATOL,
            step_control=compare.CONTROL,
            retry=compare.RETRY,
            point_setup=point_setup,
            scaled_norm=True,
            on_checkpoint=logger.on_checkpoint,
            on_retry=logger.on_retry,
        )
    finally:
        log_file.close()
    elapsed = time.perf_counter() - started

    velocity, pressure = momentum.unpack(flow)
    nu_t = turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
    return dict(
        centroid=np.asarray(geom.cell.centroid),
        U=np.asarray(velocity),
        p=np.asarray(pressure),
        k=np.asarray(k),
        omega=np.asarray(omega),
        nut=np.asarray(nu_t),
        wall=elapsed,
    )


def relative_difference(a, b):
    """``(relative L2, relative max)`` of ``a - b``, each normalized by ``b``'s own magnitude.

    Normalized so fields spanning many orders -- a pressure against an omega -- read off one table.
    """
    scale = float(np.linalg.norm(b)) or 1.0
    peak = float(np.abs(b).max()) or 1.0
    return float(np.linalg.norm(a - b)) / scale, float(np.abs(a - b).max()) / peak


#: A row of the march log's outer summary table,
#: ``| step | t(s) | beta | in | cyc | R | a_min | flg |`` -- captured as ``(step, in, cyc)``.
#:
#: ⚠️ Anchored at the line start and matched on the WHOLE row, because the log carries a second,
#: indented per-inner table of the same broad shape. A looser pattern matches both and silently reads
#: the outer table's ``t(s)`` column as a cycle count, which is a plausible-looking number that grows
#: with the march -- checked once against a real log rather than trusted, and it was wrong the first
#: time for exactly that reason.
_STEP_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*[\d.eE+-]+\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
    re.MULTILINE,
)


def cycles_from(log_path):
    """Krylov restart cycles the march logged, as ``(total, per_step)``.

    This is the preconditioner-health signal, and it is the number the march reports for itself
    rather than one re-derived from the inner tables. The distribution matters as much as the total:
    a march at one or two cycles throughout is preconditioned well, and one whose tail climbs is not,
    and those are different situations behind the same sum.
    """
    per_step = [int(cyc) for _, _, cyc in _STEP_ROW.findall(Path(log_path).read_text())]
    return sum(per_step), per_step


def _probe_count(reach):
    """Residual evaluations one refresh costs at ``reach`` -- what a shortened reach actually saves.

    Reported beside the cycle counts so a flat sweep reads as a *saving* rather than as a null, and so
    the sweep demonstrates its own knob has teeth: if this did not move with the reach, flat cycles
    would mean the setting was ignored rather than that it does not matter.
    """
    plan = CoupledJacobianProbe.build(
        compare.build_case()["coupled"], stencil_reach=reach, column_reach=COLUMN_REACH
    ).plan
    n = plan.n_probes
    return int(np.sum(n)) if hasattr(n, "__len__") else int(n)


def reach_sweep() -> None:
    """Does the probe reach matter for THIS preconditioner? Measure it rather than inherit it.

    The benchmark's reach of 5 was calibrated against an **incomplete-LU** smoother, where an
    under-reaching probe is destructive in a specific way: the folded far-field entries become pivots,
    and its own record has reach 3 failing outright where reach 5 converges in one matvec. A
    SIMPLE-smoothed hierarchy does not eliminate the matrix into pivots -- it forms an approximate
    Schur complement and applies V-cycles, so a perturbed far entry degrades a rate instead of
    wrecking a factorization. That predicts a much flatter dependence on reach, and a shorter probe
    would be worth real time: the colouring's cost grows steeply with reach, and the probe is the
    single largest allocation this case makes.

    This runs the standard arm at several reaches on ONE Reynolds rung with a step cap, and reports
    the cycles each buys. It is a preconditioner-cost probe, not a convergence run -- read the cycles
    and the wall clock, not the residual it stops at.
    """
    reaches = tuple(int(r) for r in os.environ.get("PITZ_AB_REACH_SWEEP", "2,3,5").split(","))
    steps = int(os.environ.get("PITZ_AB_SWEEP_STEPS", "6"))
    print(
        f"reach sweep on the standard arm: reaches {reaches}, {steps} steps, 1 Reynolds rung",
        flush=True,
    )
    print(f"  {'reach':>6} {'probes':>8} {'wall*':>9} {'cycles':>8} {'per step':>28}", flush=True)
    for reach in reaches:
        log_path = HERE / f"sweep-reach{reach}.log"
        # ⚠️ The FIRST arm pays this process's JIT compilation, so its wall clock is not comparable
        # to the others'. Cycles are what this sweep is read on; the seconds are a sanity check only.
        started = time.perf_counter()
        try:
            solve_arm(CorrectedGreenGauss(), log_path, reach=reach, points=1, max_steps=steps)
        except Exception as exc:  # a capped march stops on its own cap; that is not a failure here
            print(
                f"  {reach:>6}  stopped: {type(exc).__name__}: {str(exc).splitlines()[0][:40]}",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        total, per_step = cycles_from(log_path)
        print(
            f"  {reach:>6} {_probe_count(reach):>8} {elapsed:>9.1f} {total:>8} {per_step!s:>28}",
            flush=True,
        )
    print(
        "\n* the first arm pays JIT compilation -- read the probes and cycles, not the wall.\n"
        "Flat cycles across reaches means the long probe buys nothing on this preconditioner and the "
        "reach can come down; the probe column is what that saves, per refresh.",
        flush=True,
    )


def check_accuracy() -> None:
    """Is the chosen sweep pair still reconstructing the Betchen gradient, or a cheaper approximation?

    Both counts trade exactness for speed, and a fixed sweep carries no convergence test, so this is
    the check that keeps the A/B honest: a cheaper arm that is quietly a *different* reconstruction
    would show up as a scheme difference that is really a solver truncation. Measured against an
    exactly-solved reconstruction of the same system on this mesh, on a quadratic -- the field the
    scheme is defined to reproduce exactly, so any departure is the solve and not the discretization.
    """
    case = compare.build_case()
    mesh, geom = case["momentum"].mesh, case["geom"]
    x = np.asarray(geom.cell.centroid)
    xf = np.asarray(geom.face.centroid)
    centre, extent = x.mean(axis=0), max(float(np.abs(x - x.mean(axis=0)).max()), 1e-300)

    def quad(pts):
        u = (pts - centre) / extent
        return 0.6 + 1.7 * u[:, 0] - 1.1 * u[:, 1] + 0.4 * u[:, 0] ** 2 + 0.9 * u[:, 0] * u[:, 1]

    phi, bv = jnp.asarray(quad(x)), jnp.asarray(quad(xf))
    exact = np.asarray(
        HessianCorrectedGradient(hessian_solver=GmresGradientSolve()).gradients(phi, mesh, geom, bv)
    )
    scale = max(float(np.abs(exact).max()), 1e-300)
    print("Betchen sweep pair vs an exactly-solved reconstruction (quadratic field):", flush=True)
    for outer, inner in ((OUTER_SWEEPS, INNER_SWEEPS), (10, 10), (20, 10)):
        got = np.asarray(
            HessianCorrectedGradient(
                solver=SweptGradientSolve(sweeps=outer, warn_tol=None),
                hessian_solver=SweptGradientSolve(sweeps=inner, warn_tol=None),
            ).gradients(phi, mesh, geom, bv)
        )
        mark = "  <- this run" if (outer, inner) == (OUTER_SWEEPS, INNER_SWEEPS) else ""
        print(
            f"  outer {outer:>2} / inner {inner:>2}: relative departure "
            f"{np.abs(got - exact).max() / scale:.3e}{mark}",
            flush=True,
        )
    corrected = np.asarray(CorrectedGreenGauss().gradients(phi, mesh, geom, bv))
    print(
        f"  for scale, CorrectedGreenGauss departs by "
        f"{np.abs(corrected - exact).max() / scale:.3e} -- the difference the A/B is measuring, so "
        f"the pair above must be well below it.",
        flush=True,
    )


def main() -> None:
    if os.environ.get("PITZ_AB_REACH_SWEEP"):
        reach_sweep()
        return
    if os.environ.get("PITZ_AB_CHECK"):
        check_accuracy()
        return
    print(
        "gradient-scheme A/B on pitzDaily (field-split SIMPLE-smoothed preconditioner)", flush=True
    )
    print(f"  probe stencil reach (both arms): {REACH}", flush=True)
    if not compare.ILU0_COMPILED:
        print(
            "  ⚠️ compiled ILU kernel missing -- run tools/build_ext.sh; timings are void",
            flush=True,
        )

    #: Run one arm rather than both, to answer a question about that arm without re-paying the other.
    #: The reach question is the one this exists for: if the Betchen arm's cycle count falls toward the
    #: standard arm's when the probe is lengthened, the extra Krylov work was the probe under-resolving
    #: its long stencil; if it does not move, that arm genuinely presents a harder operator and the cost
    #: is the scheme's. Comparing one arm against ITSELF at two reaches is a valid control -- the
    #: standard arm's own reach-independence is already measured (see `reach_sweep`).
    only = os.environ.get("PITZ_AB_ARMS")
    arms = [a for a in ARMS if only is None or a[0] in only.split(",")]
    if not arms:
        raise SystemExit(
            f"PITZ_AB_ARMS={only!r} selected no arm; choose from {[a[0] for a in ARMS]}"
        )

    results = {}
    for name, label, scheme in arms:
        log_path = HERE / f"ab-{name}.log"
        print(f"\n[{name}] {label}  -> {log_path}", flush=True)
        fields = solve_arm(scheme, log_path)
        xr = compare.reattachment_length(fields["centroid"], fields["U"][:, 0])
        cycles, per_step = cycles_from(log_path)
        results[name] = dict(
            fields=fields, xr=xr, cycles=cycles, per_solve=per_step, wall=fields["wall"]
        )
        print(
            f"  {fields['wall']:.1f} s, {cycles} cycles over {len(per_step)} steps "
            f"(max {max(per_step, default=0)}), x_r/h = {xr:.3f}",
            flush=True,
        )

    if len(results) < 2:
        # A single-arm run answers a question about that arm; the cross-arm comparison needs both.
        name, r = next(iter(results.items()))
        print(
            f"\nsingle arm ({name}) at reach {REACH}: {r['wall']:.1f} s, {r['cycles']} cycles over "
            f"{len(r['per_solve'])} solves (max {max(r['per_solve'], default=0)}), x_r/h = {r['xr']:.3f}",
            flush=True,
        )
        return

    standard, betchen = results["standard"], results["betchen"]
    of = compare.read_openfoam_reference()
    xr_of = compare.reattachment_length(of["centroid"], of["U"][:, 0])

    print("\n=== cost ===", flush=True)
    print(f"  {'arm':<10} {'wall (s)':>10} {'cycles':>8} {'wall ratio':>11}", flush=True)
    for name in ("standard", "betchen"):
        r = results[name]
        print(
            f"  {name:<10} {r['wall']:>10.1f} {r['cycles']:>8} "
            f"{r['wall'] / standard['wall']:>10.2f}x",
            flush=True,
        )

    print("\n=== solution difference (betchen vs standard) ===", flush=True)
    print(f"  {'field':<8} {'rel L2':>12} {'rel max':>12}", flush=True)
    for field in FIELDS:
        l2, mx = relative_difference(betchen["fields"][field], standard["fields"][field])
        print(f"  {field:<8} {l2:>12.3e} {mx:>12.3e}", flush=True)

    print("\n=== reattachment length x_r/h (lower wall) ===", flush=True)
    for name in ("standard", "betchen"):
        print(f"  {name:<14} {results[name]['xr']:.3f}", flush=True)
    print(f"  OpenFOAM ref   {xr_of:.3f}", flush=True)
    print(f"  betchen - standard = {betchen['xr'] - standard['xr']:+.3f}", flush=True)

    ratio = betchen["cycles"] / max(standard["cycles"], 1)
    print(
        f"\ncycle ratio betchen/standard = {ratio:.2f}x. Close to 1 means the preconditioner sees "
        f"both arms equally well and the wall-clock difference above is the reconstruction; much "
        f"above 1 means raise PITZ_AB_REACH for both arms and re-run.",
        flush=True,
    )


if __name__ == "__main__":
    main()
