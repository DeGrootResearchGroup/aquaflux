"""How long one application of bfs3d's shipped field split takes, forward and transposed, at a fixed state.

The 3D counterpart of ``pitzdaily_openfoam/split_apply_timing.py``. It builds the split ``case.yaml``
ships, applies its traced copy to one random vector, and reports the best of
``BFS3D_SPLIT_TIMING_REPEATS`` timings of the apply and of its transpose (the adjoint's direction), how
long the first call of each took, and every level of both hierarchies (size, stored entries, widest and
mean row), so two checkouts can be checked to have built the same preconditioner before their timings
are compared.

⚠️ **Compare two checkouts at the SAME state, never at their own hybrid starts.** The leading block's
coarsening reads values (``strength_threshold`` 0.25), and on this case a change in rounding alone --
a different but equivalent sparse product inside the potential-flow solve the hybrid start runs, say --
moves the start by ~1e-13 and the leading hierarchy by tens of aggregates (measured 2026-10-15: 26252
against 26248 dofs at level 1, applies 2.5 % apart). Set ``BFS3D_SPLIT_TIMING_STATE`` to one
checkpoint (``checkpoints/state-000NN.npz``) for both runs; the level table they print must then agree.
The same sensitivity makes two marches of such checkouts different marches from step 1.

``BFS3D_SPLIT_TIMING_SAVE`` (a ``.npz`` path) saves both outputs and the level table, so the two runs
can be checked equal.

Usage
-----
    BFS3D_SPLIT_TIMING_STATE=validation/bfs3d_openfoam/checkpoints/state-00018.npz \\
        validation/run_case.sh validation/bfs3d_openfoam/split_apply_timing.py --wait
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402  (enables x64)
import compare  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.precondition.traced_field_split import traced_field_split  # noqa: E402
from aquaflux.turbulence import open_session  # noqa: E402

REPEATS = int(os.environ.get("BFS3D_SPLIT_TIMING_REPEATS", "10"))


def _best(f, *args):
    """Best wall time of ``f(*args)`` in ms after one warm call; the slower runs measure other load."""
    jax.block_until_ready(f(*args))
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(f(*args))
        best = min(best, time.perf_counter() - started)
    return best * 1e3


def _first_call(f, *args):
    started = time.perf_counter()
    jax.block_until_ready(f(*args))
    return time.perf_counter() - started


def _levels(name, inverse):
    """Print and return each level's size, stored entries and row widths."""
    rows = []
    for i, level in enumerate(inverse._hierarchy.levels):
        lengths = np.diff(level.operator.to_scipy().indptr)
        print(
            f"  {name} level {i}: {level.n} dofs, {int(lengths.sum())} entries, "
            f"widest row {int(lengths.max())}, mean row {lengths.mean():.1f}",
            flush=True,
        )
        rows.append((level.n, int(lengths.sum()), int(lengths.max())))
    return np.asarray(rows)


def main():
    print(
        f"[configuration] aquaflux {Path(aquaflux.__file__).parents[1]}, jax {jax.__version__}, "
        f"{jax.default_backend()}, {os.cpu_count()} cpus",
        flush=True,
    )
    coupled = compare.build_case()["coupled"]
    state_file = os.environ.get("BFS3D_SPLIT_TIMING_STATE")
    if state_file:
        with np.load(state_file) as data:
            state = jnp.asarray(data["state"])
        print(f"state: {state_file}", flush=True)
    else:
        state = coupled.state_from_physical(*hybrid_initialize(coupled))
        print("state: the hybrid start (do not compare two checkouts this way)", flush=True)
    session = open_session(compare.SOLVER.settings()["preconditioner"], coupled)
    session.build(state, dual_time=compare.SOLVER.dual_time)
    inverse = session.preconditioner.inverse
    leading = _levels("leading", inverse._leading)
    trailing = _levels("trailing", inverse._trailing)
    split = traced_field_split(inverse)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(split.groups.n_dofs))

    forward = eqx.filter_jit(lambda s, x: s.apply(x))
    transposed = eqx.filter_jit(lambda s, x: jax.linear_transpose(s.apply, x)(x)[0])
    compile_forward = _first_call(forward, split, v)
    compile_transposed = _first_call(transposed, split, v)
    print(
        f"split apply {_best(forward, split, v):.1f} ms, transpose {_best(transposed, split, v):.1f} "
        f"ms (best of {REPEATS}); first call {compile_forward:.1f} s / {compile_transposed:.1f} s",
        flush=True,
    )
    save = os.environ.get("BFS3D_SPLIT_TIMING_SAVE")
    if save:
        np.savez(
            save,
            forward=np.asarray(forward(split, v)),
            transposed=np.asarray(transposed(split, v)),
            leading=leading,
            trailing=trailing,
        )
        print(f"wrote {save}", flush=True)


if __name__ == "__main__":
    main()
