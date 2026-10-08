"""Could Anderson acceleration shorten the target station? Replayed offline from the march's own iterates.

Anderson mixing treats one outer step of the march as a fixed-point map ``x -> G(x)`` and, instead of
taking ``G(x_k)`` as the next iterate, extrapolates from the last ``m`` steps:

    x_AA = G(x_k) - sum_j gamma_j (G(x_{j+1}) - G(x_j)),

with ``gamma`` the least-squares fit that best cancels the newest step ``f_k = G(x_k) - x_k`` by the
differences of the previous ones. It needs nothing but vectors the march already produces, so its
benefit can be bounded without building it: replay the shipped march's checkpointed iterates, form the
point Anderson would have proposed at each step, and evaluate the TRUE residual there. A proposal is
worth something only if its residual beats the march's own next iterate -- and worth a whole saved
outer step only if it reaches what the march reaches one or more steps later.

The map has to be (nearly) the same map from step to step for the extrapolation to mean anything, so
the replay covers the **target station** only: the viscosity ramp has arrived, the residual is the
case's own, and the shift sits at its floor. That is also where most of the march's Krylov work is.

Every candidate at step ``k`` is scored in one measure -- the row-equilibrated measure the march stops
on, built once at the march's own next iterate ``x_{k+1}`` -- so the comparison is not a change of
measure. The ``k`` field is projected to be non-negative before scoring, as the march's own positivity
projection would; ``omega`` is transported as its logarithm and needs nothing. The least-squares fit is
weighted per field by the RMS of the newest step's increment in that field, so a field with large
values does not decide the fit alone.

⚠️ This replays Anderson on the UNACCELERATED sequence. A real implementation feeds its own proposals
back as the next iterates, which can do better (the history improves) or worse (an accepted bad
proposal poisons it). It is a one-step bound, not a march.

Usage
-----
    PITZ_CHECKPOINT_KEEP=500 validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    validation/run_case.sh validation/pitzdaily_openfoam/anderson_replay_probe.py

``PITZ_AA_DEPTHS`` sets the depths (default ``1,2,3,5``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from continuation_seed_error import judged_norm  # noqa: E402

CHECKPOINTS = HERE / "checkpoints"
DEPTHS = tuple(int(d) for d in os.environ.get("PITZ_AA_DEPTHS", "1,2,3,5").split(","))


def load_states():
    states = {}
    for path in sorted(CHECKPOINTS.glob("state-*.npz")):
        with np.load(path) as data:
            states[int(data["step"])] = np.asarray(data["state"])
    return states


def anderson(xs, k, m, field_slices):
    """The Anderson proposal at step ``k`` of depth ``m`` on the iterates ``xs`` (``G(x_j) = x_{j+1}``)."""
    f = [xs[j + 1] - xs[j] for j in range(k - m, k + 1)]
    g = [xs[j + 1] for j in range(k - m, k + 1)]
    weights = np.ones_like(f[-1])
    for sl in field_slices:
        rms = np.sqrt(np.mean(f[-1][sl] ** 2))
        weights[sl] = 1.0 / rms if rms > 0 else 1.0
    df = np.stack([(f[i + 1] - f[i]) * weights for i in range(m)], axis=1)
    dg = np.stack([g[i + 1] - g[i] for i in range(m)], axis=1)
    gamma, *_ = np.linalg.lstsq(df, f[-1] * weights, rcond=None)
    return g[-1] - dg @ gamma, gamma


def main():
    ramp = compare.SOLVER.continuation
    start = ramp.stations * ramp.steps_per_station  # the first state the target map is applied to
    states = load_states()
    last = max(states)
    print(
        f"[configuration] target station from state {start} to {last}; depths {DEPTHS}; "
        f"jax {jax.__version__}, {jax.default_backend()}",
        flush=True,
    )
    coupled = compare.build_case()["coupled"]
    layout = coupled.layout
    n = layout.n_cells
    field_slices = [slice(f * n, (f + 1) * n) for f in range(layout.n_fields)]
    k_slice = layout.slice_of("k")
    xs = {k: v for k, v in states.items() if k >= start}

    def project(x):
        x = np.array(x)
        x[k_slice] = np.maximum(x[k_slice], 0.0)
        return x

    def scored(measure, x):
        r = coupled.residual(jnp.asarray(project(x)))
        value = float(measure(r))
        return value if np.isfinite(value) else float("inf")

    print(
        f"\n{'k':>3} {'m':>2} {'|R(x_k+1)|':>11} {'|R(x_AA)|':>11} {'ratio':>7} "
        f"{'|R(x_k+2)|':>11} {'|R(x_k+3)|':>11} {'beats x_k+j':>11}"
    )
    for k in sorted(xs):
        if k + 1 not in xs:
            continue
        measure = judged_norm(coupled, jnp.asarray(xs[k + 1]))
        r_next = scored(measure, xs[k + 1])
        ahead = [scored(measure, xs[k + j]) if k + j in xs else None for j in (2, 3)]
        for m in DEPTHS:
            if k - m < start:
                continue
            proposal, _ = anderson(xs, k, m, field_slices)
            r_aa = scored(measure, proposal)
            # The furthest of the march's own next iterates the proposal is at least as good as: 0 if it
            # does not beat x_{k+1}, j if it beats x_{k+j}. It saves outer steps only from j = 2 on.
            worth = 0
            for j, r in enumerate([r_next, *ahead], start=1):
                if r is not None and r_aa <= r:
                    worth = j
            fmt = [f"{v:11.3e}" if v is not None else f"{'-':>11}" for v in ahead]
            print(
                f"{k:3d} {m:2d} {r_next:11.3e} {r_aa:11.3e} {r_aa / r_next:7.3f} "
                f"{fmt[0]} {fmt[1]} {worth:>11d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
