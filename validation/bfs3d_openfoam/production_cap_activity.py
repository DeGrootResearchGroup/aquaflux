"""Does the k-production cap BIND at the converged 3D backward-facing-step root?

The k-production cap is ``P_k = min(nu_t S^2, 10 beta* k omega)``. ``explicit_production_limiter``
wraps the cap's ``k`` in ``stop_gradient`` -- so wherever the cap is **active**, the Jacobian omits
that term. It now defaults to ``False`` (the exact operator); this measures what opting back in would
cost, and ``BFS3D_PRODUCTION_LIMITER=1`` runs that arm. That is a forward-solve device, and it is only
free if the cap is **inactive at the converged state**: where it binds at the root, the
implicit-function-theorem adjoint linearizes a residual different from the one solved, and the
sensitivity is silently wrong while the converged fields are fine.

This case is the one that could differ from an attached channel. The scale-free criterion (derived in
``validation/production_cap_activity.py``) is::

    production / limit = S^2 / (10 beta* omega^2)   on the unlimited branch nu_t = k / omega

so the cap binds at ``S / omega > sqrt(10 beta*) = 0.949``, independent of ``k``, against an
equilibrium boundary-layer value of ``sqrt(beta*) = 0.3``. A separating shear layer is exactly the
kind of strongly non-equilibrium region that can reach it; an attached channel measured 0.333.

Reads the **newest checkpoint** written by a `compare.py` run rather than re-deriving that driver's
option bundle -- several probes in this directory have been invalidated by silently running a
different configuration from the case they claimed to measure.

Run (after `validation/run_case.sh validation/bfs3d_openfoam/compare.py` has finished)::

    python3 validation/bfs3d_openfoam/production_cap_activity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parents[1]))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def newest_checkpoint(directory: Path) -> Path:
    """The most recently written checkpoint, by modification time.

    By name would be wrong: the counter restarts per solve, so a later rung's ``state-00007`` can sit
    beside an earlier rung's ``state-00042``. Modification time is what orders them.
    """
    files = sorted(directory.glob("state-*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(
            f"no checkpoints in {directory} -- run `validation/run_case.sh "
            f"validation/bfs3d_openfoam/compare.py` first"
        )
    return files[-1]


def main() -> None:
    checkpoints = CASE / "checkpoints"
    path = newest_checkpoint(checkpoints)
    payload = np.load(path)
    state = jnp.asarray(payload["state"])
    print(f"checkpoint      : {path.name}")
    print(f"  step {int(payload['step'])}, |R| {float(payload['residual_norm']):.4e}, "
          f"ratio {float(payload['residual_ratio']):.4e}, shift {float(payload['shift']):.4g}")

    case = compare.build_case()  # returns a dict, not a tuple
    coupled = case["coupled"]
    flow, k, omega = coupled.physical_fields(state)
    closure = coupled.turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega)

    beta_star = coupled.turbulence.model.beta_star
    production = closure.nu_t * closure.strain_rate**2
    limit = 10.0 * beta_star * k * omega
    active = production > limit
    n_active = int(jnp.sum(active))
    n = int(k.size)

    s_over_omega = closure.strain_rate / jnp.maximum(omega, 1e-300)
    threshold = float(jnp.sqrt(10.0 * beta_star))

    print(f"\ncells           : {n}")
    print(f"cap ACTIVE in   : {n_active} cells ({100.0 * n_active / n:.3f}%)")
    print(f"S/omega         : max {float(jnp.max(s_over_omega)):.4f}, "
          f"p99 {float(jnp.percentile(s_over_omega, 99)):.4f}, "
          f"median {float(jnp.median(s_over_omega)):.4f}")
    print(f"binds above     : {threshold:.4f}   "
          f"(equilibrium boundary layer sits at {float(jnp.sqrt(beta_star)):.4f})")

    if n_active:
        over = production / jnp.maximum(limit, 1e-300)
        where = jnp.where(active, over, jnp.nan)
        print(f"production/limit: max {float(jnp.max(over)):.4g}, "
              f"median where active {float(jnp.nanmedian(where)):.4g}")
        idx = np.asarray(jnp.argsort(-jnp.where(active, over, -jnp.inf))[: min(5, n_active)])
        centroid = np.asarray(case["geom"].cell.centroid)
        print("worst cells (x, y, z):")
        for i in idx:
            print(f"  cell {int(i):6d}  {centroid[i]}  over={float(over[i]):.4g}")
        print(
            "\nVERDICT: the cap BINDS at the converged root, so `explicit_production_limiter=True`\n"
            "         removes a real Jacobian term there and the coupled adjoint is NOT exact."
        )
    else:
        print(
            "\nVERDICT: the cap binds NOWHERE at the converged root, so the stop_gradient is inert\n"
            "         here and the adjoint is exact despite the limiter being on. It remains a\n"
            "         latent hazard -- nothing checks this, and no other case has been measured."
        )


if __name__ == "__main__":
    main()
