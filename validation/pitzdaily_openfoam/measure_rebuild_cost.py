"""What does rebuilding the row-scaled residual measure cost, against the work of one outer step?

A coupled solve measured in ``RowScaled()`` rebuilds :func:`~aquaflux.turbulence.coupled_scaled_norm`
at the start of every outer iteration. That measure used to be frozen once at the build state by
default, because the rebuild was assumed expensive; this harness is what showed it is not. It times the
rebuild at a saved pitzDaily state beside the two evaluations every outer step already pays at least
once: a coupled residual, and a Jacobian-vector product of it.

Run with ``validation/run_case.sh validation/pitzdaily_openfoam/measure_rebuild_cost.py``. The state is
read from ``PITZ_STATE`` (default: the newest checkpoint under ``checkpoints/``).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# A script's own directory is on `sys.path`, not the repository root, so put both there before importing.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.turbulence import UnpreconditionedScalars  # noqa: E402
from aquaflux.turbulence.coupled import _coupled_shift_policy, coupled_scaled_norm  # noqa: E402
from compare import build_case  # noqa: E402

REPEATS = 20


def _timed(label, fn):
    """Median wall time of ``fn`` over ``REPEATS`` calls after two warm-up calls."""
    for _ in range(2):
        jax.tree.map(lambda leaf: getattr(leaf, "shape", None), fn())
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn()
        jax.tree.map(lambda leaf: np.asarray(leaf) if hasattr(leaf, "shape") else leaf, out)
        samples.append(time.perf_counter() - start)
    median = float(np.median(samples))
    print(
        f"  {label:<44s} median {median * 1e3:9.2f} ms  (min {min(samples) * 1e3:.2f})", flush=True
    )
    return median


def main():
    checkpoints = sorted((HERE / "checkpoints").glob("state-*.npz"))
    path = Path(os.environ.get("PITZ_STATE", checkpoints[-1]))
    saved = np.load(path)
    print(f"state: {path.name} (step {int(saved['step'])}, R {float(saved['residual_norm']):.3e})")
    case = build_case()
    coupled = case["coupled"]
    state = jax.numpy.asarray(saved["state"])
    print(f"cells: {coupled.layout.n_cells}, unknowns: {state.shape[0]}", flush=True)
    policy = _coupled_shift_policy(coupled, state, UnpreconditionedScalars())

    rebuild = eqx_jit(lambda s: coupled_scaled_norm(coupled, policy, s))
    residual = eqx_jit(coupled.residual)
    tangent = jax.numpy.ones_like(state)
    jvp = eqx_jit(lambda s: jax.jvp(coupled.residual, (s,), (tangent,))[1])
    measure = coupled_scaled_norm(coupled, policy, state)
    apply = eqx_jit(lambda r: measure(r))
    r = residual(state)

    print("[timings, jit-compiled, warm]", flush=True)
    t_rebuild = _timed("rebuild coupled_scaled_norm", lambda: rebuild(state))
    # The march calls its norm builder eagerly, between jit-compiled steps, so time that path too.
    t_eager = _timed(
        "rebuild, eager (as the march calls it)",
        lambda: coupled_scaled_norm(coupled, policy, state),
    )
    t_apply = _timed("apply the measure to a residual", lambda: apply(r))
    t_residual = _timed("coupled residual", lambda: residual(state))
    t_jvp = _timed("coupled residual jvp", lambda: jvp(state))
    print("[ratios]")
    print(f"  rebuild / residual: {t_rebuild / t_residual:.3f}")
    print(f"  rebuild / jvp:      {t_rebuild / t_jvp:.3f}")
    print(f"  eager rebuild / residual: {t_eager / t_residual:.3f}")
    print(f"  apply / residual:   {t_apply / t_residual:.3f}")


def eqx_jit(fn):
    return eqx.filter_jit(fn)


if __name__ == "__main__":
    main()
