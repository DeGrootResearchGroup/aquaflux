"""How much could batching several solves with ``vmap`` save on CPU? A bound, from the march's own states.

Batching a parameter sweep runs several marches in lock-step and ``vmap``s each step over the members.
Only the work that is the same JAX program for every member batches: the residual and its
Jacobian-vector product (JVP). The preconditioner does not -- it is a host object applied through a
callback, and every member carries its own fitted hierarchy -- and neither do its refits. So each
Krylov application costs ``pc + jvp`` alone and at best ``pc + jvp_B / B`` per member batched. This
probe measures the three numbers that bound it on pitzDaily, at the target station's first system of
each of three steps:

* ``pc`` -- one application of the shipped field-split preconditioner,
* ``jvp`` -- one JVP of the coupled residual,
* ``jvp_B / B`` -- the per-member JVP cost when ``B`` members (``B`` = 1, 2, 4, 8) at ``B`` different
  march states are pushed through one ``vmap``.

All best of ``PITZ_BATCH_REPEATS`` after a warm call. The states are the march's own (``replay.py``),
standing in for the nearby parameter values a sweep would carry: a member's cost does not depend on the
parameter value, only on the program, which is the same.

Usage
-----
    PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1 \\
        validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    validation/run_case.sh validation/pitzdaily_openfoam/batch_bound_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.solve import jacobian_matvec  # noqa: E402
from replay import MarchReplay  # noqa: E402

REPEATS = int(os.environ.get("PITZ_BATCH_REPEATS", "10"))
BATCHES = (1, 2, 4, 8)


def _best(f, *args):
    """Best wall time of ``f(*args)`` in ms, after a warm call; the slower runs measure other load."""
    jax.block_until_ready(f(*args))
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(f(*args))
        best = min(best, time.perf_counter() - started)
    return best * 1e3


def main():
    replay = MarchReplay()
    print(f"{replay.describe()}; repeats {REPEATS}; batches {BATCHES}", flush=True)
    # The first system of each target-station step, in the march's order (the generator refits the
    # preconditioner as it goes, so the timed apply is the one the march used there).
    firsts, timed = {}, []
    apply_pc = eqx.filter_jit(replay.apply_pc)
    jvp = eqx.filter_jit(jacobian_matvec)
    rng = np.random.default_rng(0)
    timed_steps = set(np.linspace(replay.stations + 1, max(replay.states), 3).round().astype(int))
    for system in replay.systems():
        if system.inner:
            continue
        firsts[system.step] = system
        if system.step in timed_steps:
            v = jnp.asarray(rng.standard_normal(system.b.size))
            pc = _best(apply_pc, v)
            single = _best(jvp, system.assembler, system.p, v)
            timed.append((system.step, pc, single))
            print(f"step {system.step:3d}: pc {pc:8.1f} ms, jvp {single:6.1f} ms", flush=True)
    assembler = replay.coupled
    states = [firsts[k].p for k in sorted(firsts)]
    batched = eqx.filter_jit(jax.vmap(jacobian_matvec, in_axes=(None, 0, 0)))
    per_member = {}
    for b in BATCHES:
        if b > len(states):
            break
        p = jnp.stack(states[:b])
        v = jnp.asarray(rng.standard_normal(p.shape))
        per_member[b] = _best(batched, assembler, p, v) / b
        print(f"vmap over {b} members: jvp {per_member[b]:6.1f} ms per member", flush=True)
    pc = float(np.mean([t[1] for t in timed]))
    single = float(np.mean([t[2] for t in timed]))
    print(f"\nmean over the timed steps: pc {pc:.1f} ms, jvp {single:.1f} ms", flush=True)
    for b, cost in per_member.items():
        print(
            f"B = {b}: one application per member {pc + cost:.1f} ms against {pc + single:.1f} ms "
            f"alone -- at most {(pc + single) / (pc + cost):.2f}x on the Krylov work",
            flush=True,
        )


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[done in {time.perf_counter() - t0:.0f} s]")
