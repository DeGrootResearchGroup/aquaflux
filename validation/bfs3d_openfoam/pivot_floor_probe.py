"""Is the step at a checkpoint decided by rounding, and does the pivot floor stop that? (bfs3d, one step.)

Rebuilds the march's step at a checkpoint the way the case's viscosity-ramp march builds it -- the session
re-pointed to the ANCHOR companion and the step built at the anchor's own seed, which fixes the step's shift
diagonal and the frozen coarsening, then re-pointed to the step's station and re-fitted at the checkpoint
state and the step's shift -- under the march's own settings (``BFS3D_SPLIT_PRECISION``,
``BFS3D_PIVOT_FLOOR``, ...). Then it reports:

  floored    the rows the pivot floor changed at that re-fit, by field
  linearity  |A M (v + w) - A M v - A M w| / |A M v| for random v, w, with A the exact shifted Jacobian:
             how far the preconditioned operator GMRES applies is from one fixed linear map
  solves     the exact-Jacobian shifted solve the step's first inner iteration makes (the step's own
             solver, stopping in its own measure), on the freshly evaluated residual and on
             ``BFS3D_PROBE_PERTURBATIONS`` copies perturbed by a relative 1e-15: cycles, the linear residual
             in the Euclidean and the row-scaled measure, and the line-search ladder on the projected
             direction

A step decided by rounding gives a different direction for every perturbation; one that is not gives the
same one. ⚠️ The fresh residual differs from the march's carried one at round-off, so at a step decided by
rounding the "unperturbed" row is one more draw, not the march's own step.

Environment: ``BFS3D_PROBE_STEP`` (one-based: the step starts from ``state-<step-1>``),
``BFS3D_PROBE_BETA`` (default: the shift recorded in ``state-<step>``), ``BFS3D_PROBE_CHECKPOINTS`` (default
this case's ``checkpoints/``), ``BFS3D_PROBE_PERTURBATIONS`` (default 4).
"""

from __future__ import annotations

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
from aquaflux.solve.linear import in_progress_measure  # noqa: E402
from aquaflux.turbulence import coupled_step, open_session, scale_both_blocks, scale_momentum_only  # noqa: E402
from aquaflux.turbulence.coupled import coupled_scaled_norm  # noqa: E402

NAMES = ["u", "v", "w", "p", "k", "omega"]
LADDER = [1.0, 0.5, 0.25, 0.125, 1 / 32]


def main() -> None:
    step_number = int(os.environ["BFS3D_PROBE_STEP"])
    checkpoints = Path(os.environ.get("BFS3D_PROBE_CHECKPOINTS", CASE / "checkpoints"))
    perturbations = int(os.environ.get("BFS3D_PROBE_PERTURBATIONS", "4"))
    with np.load(checkpoints / f"state-{step_number - 1:05d}.npz") as data:
        phi = jnp.asarray(data["state"])
    beta_env = os.environ.get("BFS3D_PROBE_BETA")
    if beta_env is not None:
        beta = float(beta_env)
    else:
        with np.load(checkpoints / f"state-{step_number:05d}.npz") as data:
            beta = float(data["shift"])

    solver_spec = compare.SOLVER
    ramp = solver_spec.continuation
    coupled = compare.build_case()["coupled"]
    companion = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}[ramp.scale]

    def assembler_for(station):
        return coupled if station == ramp.stations else companion(coupled, ramp.anchor ** (1.0 - station / ramp.stations))

    anchor = assembler_for(0)
    seed = anchor.state_from_physical(*hybrid_initialize(anchor))
    settings = solver_spec.settings()
    allowed = inspect.signature(coupled_step).parameters
    march = {k: v for k, v in settings.items() if k in allowed and k != "preconditioner"}
    session = open_session(settings["preconditioner"], coupled)
    session.rebind(anchor)
    base = session.build(seed, **march)
    station = min(step_number - 1, ramp.stations)
    assembler = assembler_for(station)
    session.rebind(assembler)
    active = eqx.tree_at(lambda s: s.relaxation_schedule, base, ConstantRelaxation(jnp.asarray(beta)))
    update = session.refresh_preconditioner(active, phi, due=True)
    if update is not None:
        active = update(active)
    active = active.with_norm(coupled_scaled_norm(coupled, active.shift_policy, phi))
    norm = active.residual_norm
    n = coupled.layout.n_cells
    print(f"[probe] step {step_number} (station {station}), beta {beta:g}, split precision "
          f"{compare.SPLIT_PRECISION}, pivot floor {compare.PIVOT_FLOOR}", flush=True)
    rows = session._hook.floored_rows
    print(f"[floored] {rows.size} rows: " + ", ".join(f"{NAMES[f]} {int(np.sum(rows // n == f))}" for f in range(6)),
          flush=True)

    residual_fn = assembler.residual
    g0 = residual_fn(phi)
    term = active.shift_policy.shift_term(phi, g0)
    strength = active.relaxation_schedule.shift_strength(norm(g0), norm(g0))
    shift = jax.lax.stop_gradient(term.shift(strength))
    preconditioner = term.make_preconditioner(strength)
    apply_a = jax.jit(lambda v: jax.jvp(residual_fn, (phi,), (v,))[1] + shift * v)
    apply_m = jax.jit(lambda v: jnp.asarray(preconditioner(v)))
    rng = np.random.default_rng(1234)
    for trial in range(3):
        v, w = (jnp.asarray(rng.standard_normal(phi.shape)) for _ in range(2))
        amv, amw, amvw = apply_a(apply_m(v)), apply_a(apply_m(w)), apply_a(apply_m(v + w))
        print(f"[linearity] pair {trial}: |AM(v+w) - AMv - AMw| / |AMv| = "
              f"{float(jnp.linalg.norm(amvw - amv - amw) / jnp.linalg.norm(amv)):.2e}", flush=True)

    G = jax.jit(lambda p: residual_fn(p) + shift * (p - phi))
    solver = in_progress_measure(active.linear_solver(), norm)
    project = active.step_projection
    for i in range(perturbations + 1):
        noise = np.random.default_rng(i).standard_normal(g0.shape)
        g = g0 if i == 0 else g0 * (1.0 + 1e-15 * jnp.asarray(noise))
        d, cycles = _shifted_solve(residual_fn, phi, g, shift, preconditioner, solver)
        linear = apply_a(d) + g
        v = project(phi, d) if project is not None else d
        ladder = [float(norm(G(phi + a * v))) / float(norm(g)) for a in LADDER]
        print(f"[solve] {'fresh' if i == 0 else f'perturbed {i}':12s} cycles {int(cycles):2d}  "
              f"Euclidean {float(jnp.linalg.norm(linear) / jnp.linalg.norm(g)):.2e}  "
              f"row-scaled {float(norm(linear) / norm(g)):.3f}  ladder "
              + " ".join(f"{r:.3f}" for r in ladder), flush=True)


if __name__ == "__main__":
    main()
