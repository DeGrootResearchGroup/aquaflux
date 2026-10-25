"""How long one application of the shipped field split takes, forward and transposed.

The preconditioner apply is most of a Krylov application on this case, and its cost depends on how the
level operators' sparse products compile on the machine at hand -- which differs between machines by
more than any change to the hierarchy. This builds the split ``case.yaml`` ships at the case's hybrid
start (no march, no capture needed), applies its traced copy to one random vector, and reports the best
of ``PITZ_SPLIT_TIMING_REPEATS`` timings of the apply and of its transpose (the adjoint's direction), and how
long the first call of each took (the compilation).

Run it in two checkouts to compare them; it saves both outputs to ``PITZ_SPLIT_TIMING_SAVE`` (a ``.npy``
path) when that is set, so the two can be checked equal.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/split_apply_timing.py --wait
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
from aquaflux.solve import with_transposes  # noqa: E402
from aquaflux.turbulence import open_session  # noqa: E402

REPEATS = int(os.environ.get("PITZ_SPLIT_TIMING_REPEATS", "15"))


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


def main():
    print(
        f"[configuration] aquaflux {Path(aquaflux.__file__).parents[1]}, jax {jax.__version__}, "
        f"{jax.default_backend()}, {os.cpu_count()} cpus",
        flush=True,
    )
    coupled = compare.build_case()["coupled"]
    seed = coupled.state_from_physical(*hybrid_initialize(coupled))
    session = open_session(compare.SOLVER.settings()["preconditioner"], coupled)
    session.build(seed, dual_time=compare.SOLVER.dual_time)
    split = traced_field_split(session.preconditioner.inverse)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(split.groups.n_dofs))

    forward = eqx.filter_jit(lambda s, x: s.apply(x))
    transposed = eqx.filter_jit(lambda s, x: jax.linear_transpose(s.apply, x)(x)[0])
    compile_forward = _first_call(forward, split, v)
    # A fitted split holds no transposes; an adjoint packs them, and so does this timing, untimed.
    packed = with_transposes(split)
    compile_transposed = _first_call(transposed, packed, v)
    print(
        f"split apply {_best(forward, split, v):.1f} ms, transpose {_best(transposed, packed, v):.1f} "
        f"ms (best of {REPEATS}); first call {compile_forward:.1f} s / {compile_transposed:.1f} s",
        flush=True,
    )
    save = os.environ.get("PITZ_SPLIT_TIMING_SAVE")
    if save:
        np.save(save, np.stack([np.asarray(forward(split, v)), np.asarray(transposed(split, v))]))
        print(f"wrote {save}", flush=True)


if __name__ == "__main__":
    main()
