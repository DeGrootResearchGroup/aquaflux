"""Why does a converged shifted-Newton direction fail to descend ||G||? (bfs3d, one step.)

Rebuilds the march's step at a checkpoint exactly as validation/bfs3d_openfoam/step_replay.py does
(full re-fit at the step's state and shift -- faithful when the march re-fitted before the step), checks
the replay against the march log's inner row, then takes the step's FIRST inner direction three ways:

  raw        the shifted solve as the march solves it (its own rtol), before the per-cell projection
  projected  the same after the step projection (per-cell k clip, then the log-omega bound)
  tight      the shifted solve at rtol 1e-8, before and after projection

and reports, for each, the achieved linear residual and ||G(p + a d)||_M over the line-search ladder and at
small a, plus which fields and cells carry the change. Environment: BFS3D_PROBE_STEP (one-based, as the march log
numbers it), BFS3D_PROBE_BETA; plus the march's own BFS3D_* settings so the residual is the one that ran.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys
from pathlib import Path

CASE = Path(__file__).resolve().parent
REPO = CASE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.solve import ConstantRelaxation  # noqa: E402
from aquaflux.solve.continuation import _shifted_solve  # noqa: E402
from aquaflux.solve.linear import in_progress_measure, with_relative_tolerance  # noqa: E402
from aquaflux.solve.march import _march_step  # noqa: E402
from aquaflux.turbulence import coupled_step, open_session, scale_both_blocks, scale_momentum_only  # noqa: E402
from aquaflux.turbulence.coupled import coupled_scaled_norm  # noqa: E402

CHECKPOINTS = CASE / "checkpoints"
NAMES = ["u", "v", "w", "p", "k", "omega"]


def _row(inner, g_in, g_out, cycles, alpha, candidate):
    print(f"    replay inner {int(inner)}: cyc {int(cycles)}  G in {float(g_in):.4e}  G out "
          f"{float(g_out):.4e}  rate {float(g_out) / float(g_in):.3f}  alpha {float(alpha):.3f}", flush=True)


def main() -> None:
    k = int(os.environ["BFS3D_PROBE_STEP"])
    beta = float(os.environ.get("BFS3D_PROBE_BETA", "1e-4"))
    solver_spec = compare.SOLVER
    ramp = solver_spec.continuation
    case = compare.build_case()
    coupled = case["coupled"]
    companion = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}[ramp.scale]
    station = min(k - 1, ramp.stations)

    def assembler_for(s):
        return coupled if s == ramp.stations else companion(coupled, ramp.anchor ** (1.0 - s / ramp.stations))

    anchor = assembler_for(0)
    seed = anchor.state_from_physical(*hybrid_initialize(anchor))
    allowed = inspect.signature(coupled_step).parameters
    settings = solver_spec.settings()
    march = {k_: v for k_, v in settings.items() if k_ in allowed and k_ != "preconditioner"}
    session = open_session(settings["preconditioner"], coupled)
    session.rebind(anchor)
    base = session.build(seed, **march)
    base = dataclasses.replace(base, inner_observer=_row)
    policy = base.shift_policy
    with np.load(CHECKPOINTS / f"state-{k - 1:05d}.npz") as data:
        phi_n = jnp.asarray(data["state"])
    assembler = assembler_for(station)
    reference = float(coupled_scaled_norm(coupled, policy, seed)(anchor.residual(seed)))
    measure = coupled_scaled_norm(coupled, policy, phi_n)
    session.rebind(assembler)
    active = eqx.tree_at(lambda s: s.relaxation_schedule, base, ConstantRelaxation(beta))
    update = session.refresh_preconditioner(active, phi_n, due=True)
    if update is not None:
        active = update(active)
    active = active.with_norm(measure)
    print(f"[probe] step {k}, station {station}, beta {beta:g}, linear {solver_spec.linear_solve}", flush=True)

    # 1. Faithfulness: the march's own step.
    outcome, rnorm = _march_step(active, assembler.residual, phi_n, jnp.asarray(reference), active.linear_solver())
    print(f"[replay] |R| {float(rnorm):.4e}, min alpha {float(outcome.alpha):.3f}, "
          f"{int(outcome.raw_cycles)} raw cycles", flush=True)

    # 2. The first inner direction, three ways.
    residual_fn = assembler.residual
    r0 = residual_fn(phi_n)
    term = active.shift_policy.shift_term(phi_n, r0)
    strength = active.relaxation_schedule.shift_strength(measure(r0), jnp.asarray(reference))
    shift = term.shift(strength)
    preconditioner = term.make_preconditioner(strength)

    def G(p):
        return residual_fn(p) + shift * (p - phi_n)

    g0 = G(phi_n)
    n0 = float(measure(g0))
    base_solver = in_progress_measure(active.linear_solver(), measure)
    tight_solver = in_progress_measure(with_relative_tolerance(active.linear_solver(), 1e-8), measure)
    project = active.step_projection
    directions = {}
    for label, solver in (("raw", base_solver), ("tight", tight_solver)):
        d, cyc = _shifted_solve(residual_fn, phi_n, g0, shift, preconditioner, solver)
        lin = jax.jvp(residual_fn, (phi_n,), (d,))[1] + shift * d + g0
        print(f"[{label}] cycles {int(cyc)}, linear residual |J d + s d + g|_M / |g|_M = "
              f"{float(measure(lin)) / n0:.3e}", flush=True)
        directions[label] = d
        directions[label + "+proj"] = project(phi_n, d) if project is not None else d

    n = phi_n.shape[0] // 6
    alphas = [1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 64, 1 / 256, 1 / 1024, 1e-4, 1e-6]
    print(f"\n|G(phi_n)|_M = {n0:.6e};   ratios |G(p + a d)|_M / |G(p)|_M:", flush=True)
    print("  a        " + "  ".join(f"{lab:>12}" for lab in directions), flush=True)
    for a in alphas:
        vals = [float(measure(G(phi_n + a * d))) / n0 for d in directions.values()]
        print(f"  {a:<8.3g} " + "  ".join(f"{v:12.6f}" for v in vals), flush=True)

    # 3. What the projection changed, and where the change in ||G|| comes from at small a.
    for lab in ("raw", "tight"):
        d, dp = directions[lab], directions[lab + "+proj"]
        diff = np.asarray(dp - d).reshape(6, n)
        changed = np.nonzero(np.abs(diff).sum(axis=0))[0]
        rel = float(jnp.linalg.norm(dp - d) / jnp.linalg.norm(d))
        print(f"\n[{lab}] projection changed {changed.size} cells, |d' - d| / |d| = {rel:.3e}; "
              f"per field |d'-d|: " + ", ".join(f"{NAMES[i]} {np.linalg.norm(diff[i]):.2e}" for i in range(6)),
              flush=True)
        kk = np.asarray(phi_n).reshape(6, n)[4]
        draw = np.asarray(d).reshape(6, n)
        worst = changed[np.argsort(-np.abs(diff[4, changed]))][:6] if changed.size else []
        for c in worst:
            print(f"    cell {c}: k {kk[c]:.3e}, dk raw {draw[4, c]:.3e} -> {np.asarray(dp).reshape(6, n)[4, c]:.3e}, "
                  f"d log omega raw {draw[5, c]:.3e} -> {np.asarray(dp).reshape(6, n)[5, c]:.3e}", flush=True)

    # Directional split of the measure change at a small step: which equations/cells grow?
    a = 1e-4
    for lab in ("raw+proj", "raw"):
        d = directions[lab]
        g1 = G(phi_n + a * d)
        # The measure is a weighted 2-norm; compare per-equation contributions by masking.
        g0b, g1b = np.asarray(g0).reshape(6, n), np.asarray(g1).reshape(6, n)
        parts = []
        for i in range(6):
            mask = np.zeros((6, n)); mask[i] = 1.0
            m0 = float(measure(jnp.asarray((g0b * mask).ravel())))
            m1 = float(measure(jnp.asarray((g1b * mask).ravel())))
            parts.append(f"{NAMES[i]} {m0:.4e}->{m1:.4e}")
        print(f"\n[{lab}] per-equation |G|_M at a={a:g}: " + "; ".join(parts), flush=True)

    # Where the full (fallback) step's growth lives: per-cell weighted |G| (the measure's own terms).
    centres = np.asarray(case["geom"].cell.centroid)
    rs = np.asarray(measure.row_scale).reshape(6, n)
    fs = np.asarray(measure.field_scale)
    state = np.asarray(phi_n).reshape(6, n)
    for lab in ("raw+proj", "tight+proj"):
        d = directions[lab]
        g1 = np.asarray(G(phi_n + d)).reshape(6, n)
        g0b = np.asarray(g0).reshape(6, n)
        w0 = np.abs(g0b) / rs / fs[:, None] / n
        w1 = np.abs(g1) / rs / fs[:, None] / n
        print(f"\n[{lab}] at a=1, per-field block mean term (measure = sqrt of sum of squares): "
              + ", ".join(f"{NAMES[i]} {w0[i].sum():.3e}->{w1[i].sum():.3e}" for i in range(6)), flush=True)
        inc = (w1 - w0)
        flat = np.argsort(-inc.ravel())[:10]
        dd = np.asarray(d).reshape(6, n)
        for idx in flat:
            f, c = divmod(int(idx), n)
            print(f"    {NAMES[f]:5s} cell {c:6d} at {np.round(centres[c], 4)}: term {w0[f, c]:.2e}->{w1[f, c]:.2e}; "
                  f"k {state[4, c]:.3e} dk {dd[4, c]:.2e}; logw {state[5, c]:.3f} dlogw {dd[5, c]:.2e}; "
                  f"|U| {np.linalg.norm(state[:3, c]):.3f}", flush=True)
        # Concentration: fraction of the total increase carried by the top 10 / 100 terms.
        pos = np.sort(inc[inc > 0].ravel())[::-1]
        if pos.size:
            print(f"    increase carried by top 10 terms {pos[:10].sum() / pos.sum():.1%}, top 100 "
                  f"{pos[:100].sum() / pos.sum():.1%} (of {pos.size} rising terms)", flush=True)


if __name__ == "__main__":
    main()
