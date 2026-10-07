"""How much could a tangent predictor give the viscosity ramp? Measured from the march's own checkpoints.

A tangent (Euler) predictor moves the state along the solution path before a continuation step:
``x + dlam v`` with ``J v = -dR/dlam``. Expanding the residual about ``x`` at the new station,

    R(x + dlam v, lam + dlam) ~= R(x, lam + dlam) - dlam dR/dlam,

so to first order the predictor removes **exactly** the term ``dlam dR/dlam`` and nothing else, at any
state -- converged or not. That makes its benefit boundable from residual evaluations alone, with no
Jacobian, no linear solve and no march.

The case ramps the momentum viscosity down in ``stations`` geometric stations of one outer step each,
inside one march, so the state entering a station is NOT a root of the previous one: it carries
whatever the previous step left unsolved. Per station change this reports, in the row-equilibrated
measure the march stops on (rebuilt at the new station):

``left``   ``|R(x, lam_old)|`` -- what the previous station's step left unsolved.
``E0``     ``|R(x, lam_new)|`` -- what the next step actually faces.
``jump``   ``|dlam dR/dlam|`` -- the part of it the viscosity change introduced.
``E1``     ``|R(x, lam_new) - dlam dR/dlam|`` -- what an exactly solved tangent predictor would leave.

``E1 / E0`` near one means the station jump is a small part of what the step faces, and a predictor --
which costs a linear solve per station -- has little to remove. ``E1`` is a first-order bound: a real
predictor solves loosely and against a frozen preconditioner, so it can only do worse.

The checkpoints come from a run of ``compare.py`` with ``PITZ_CHECKPOINT_KEEP`` large enough to keep
every step. ``state-0000k.npz`` is the state after ``k`` outer steps, which is the state outer step
``k`` (counting from zero) starts from, and that step solves station ``k``.

Usage
-----
    PITZ_CHECKPOINT_KEEP=500 validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    validation/run_case.sh validation/pitzdaily_openfoam/ramp_predictor_probe.py
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
from aquaflux.turbulence import scale_both_blocks, scale_momentum_only  # noqa: E402
from continuation_seed_error import judged_norm  # noqa: E402

CHECKPOINTS = Path(os.environ.get("RAMP_PROBE_CHECKPOINTS", HERE / "checkpoints"))
_COMPANIONS = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}


def main():
    ramp = compare.SOLVER.continuation
    if ramp is None:
        raise SystemExit("the case's solver has no viscosity ramp to probe")
    if ramp.steps_per_station != 1:
        raise SystemExit("this probe assumes one outer step per station")
    companion = _COMPANIONS[ramp.scale]
    stations = ramp.stations
    print(
        f"[configuration] anchor {ramp.anchor}, {stations} stations x {ramp.steps_per_station}, "
        f"scale {ramp.scale!r} ({companion.__name__}); jax {jax.__version__}, {jax.default_backend()}",
        flush=True,
    )
    coupled = compare.build_case()["coupled"]

    def lam(station):
        # Station s runs the viscosity scaled by anchor ** (1 - s / stations); the target is s = stations.
        return (1.0 - min(station, stations) / stations) * np.log(ramp.anchor)

    def at(lam_value):
        if lam_value == 0.0:
            return coupled
        return companion(coupled, float(np.exp(lam_value)))

    print(
        f"\n{'step':>4} {'station':>7} {'dlam':>8} {'left':>10} {'E0':>10} {'jump':>10} "
        f"{'E1':>10} {'E1/E0':>7}"
    )
    for k in range(1, stations + 1):
        path = CHECKPOINTS / f"state-{k:05d}.npz"
        if not path.exists():
            print(f"{k:4d}  missing {path.name}")
            continue
        x = jnp.asarray(np.load(path)["state"])
        lam_old, lam_new = lam(k - 1), lam(k)
        dlam = lam_new - lam_old

        def residual_at(lam_value, x=x):
            return companion(coupled, jnp.exp(lam_value)).residual(x)

        r_new = np.asarray(at(lam_new).residual(x))
        _, slope = jax.jvp(residual_at, (jnp.asarray(lam_new),), (jnp.ones(()),))
        jump = dlam * np.asarray(slope)
        measure = judged_norm(at(lam_new), x)
        left = float(judged_norm(at(lam_old), x)(at(lam_old).residual(x)))
        e0 = float(measure(r_new))
        e1 = float(measure(r_new - jump))
        print(
            f"{k:4d} {k:7d} {dlam:8.4f} {left:10.3e} {e0:10.3e} {float(measure(jump)):10.3e} "
            f"{e1:10.3e} {e1 / e0:7.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
