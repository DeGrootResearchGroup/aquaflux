"""The corner cells' k row over a march: J_kk, the shift on it, R_k and the local Newton dk, per checkpoint.

For each checkpoint state-NNNNN (the state step NN+1 starts from) and each watched cell: k, the exact
Jacobian diagonal J_kk (one JVP with a unit tangent on that cell's k), the pseudo-time shift the step
applies there (the held policy's shift term at the step's recorded shift strength, turbulence damping
included), the transient residual R_k, and the local Newton correction -R_k / (J_kk + shift). Run with
the march's own BFS3D_* settings, through validation/run_case.sh (it builds a session, a few GB). BFS3D_PROBE_STATES: comma-separated checkpoint numbers; BFS3D_PROBE_CELLS: cells.
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
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.turbulence import coupled_step, open_session, scale_both_blocks, scale_momentum_only  # noqa: E402
from aquaflux.turbulence.coupled import frozen_production_viscosity  # noqa: E402

CHECKPOINTS = CASE / "checkpoints"


def main() -> None:
    states = [int(s) for s in os.environ["BFS3D_PROBE_STATES"].split(",")]
    cells = [int(c) for c in os.environ.get("BFS3D_PROBE_CELLS", "11467,11387,11507").split(",")]
    spec = compare.SOLVER
    ramp = spec.continuation
    coupled = compare.build_case()["coupled"]
    companion = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}[ramp.scale]

    def assembler_for(s):
        return coupled if s == ramp.stations else companion(coupled, ramp.anchor ** (1.0 - s / ramp.stations))

    anchor = assembler_for(0)
    seed = anchor.state_from_physical(*hybrid_initialize(anchor))
    allowed = inspect.signature(coupled_step).parameters
    settings = spec.settings()
    march = {k_: v for k_, v in settings.items() if k_ in allowed and k_ != "preconditioner"}
    session = open_session(settings["preconditioner"], coupled)
    session.rebind(anchor)
    policy = session.build(seed, **march).shift_policy  # the anchor's shift, held all march (#701)
    n = int(seed.shape[0]) // 6
    k0 = coupled.layout.slice_of("k").start
    print("state step   beta    cell        k          J_kk     shift_kk    J+shift        R_k     newton dk   J_kk(frozen P)", flush=True)
    for s in states:
        with np.load(CHECKPOINTS / f"state-{s:05d}.npz") as d:
            phi = jnp.asarray(d["state"])
        nxt = CHECKPOINTS / f"state-{s + 1:05d}.npz"
        beta = float(np.load(nxt)["shift"]) if nxt.exists() else float("nan")
        residual_fn = assembler_for(min(s, ramp.stations)).residual
        frozen_fn = frozen_production_viscosity(assembler_for(min(s, ramp.stations))).residual
        r = residual_fn(phi)
        term = policy.shift_term(phi, r)
        shift = np.asarray(term.shift(jnp.asarray(beta)))
        for c in cells:
            i = k0 + c
            e = jnp.zeros_like(phi).at[i].set(1.0)
            jkk = float(jax.jvp(residual_fn, (phi,), (e,))[1][i])
            jfz = float(jax.jvp(frozen_fn, (phi,), (e,))[1][i])
            sh = float(shift[i])
            print(f"{s:5d} {s + 1:4d} {beta:8.2e} {c:6d} {float(phi[i]):11.3e} {jkk:11.3e} {sh:11.3e} "
                  f"{jkk + sh:11.3e} {float(r[i]):11.3e} {-float(r[i]) / (jkk + sh):11.3e} {jfz:11.3e}", flush=True)


if __name__ == "__main__":
    main()
