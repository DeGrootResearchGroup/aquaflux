"""What packing the field split's transposes costs an adjoint, against what its transpose solve applies.

A fitted split holds no transposes: ``root_adjoint`` packs them once per transpose solve, through
``aquaflux.solve.with_transposes`` over the step's adjoint factory (each padded-row operator read back to
the host, transposed and packed by the slot rule, uploaded again). This harness builds the shipped bfs3d
step, takes its adjoint factory exactly as the step carries it, and times, at one state:

* the packing (``with_transposes`` of the factory), the cost this design adds to every transpose solve;
* the factory's call (widening a single-precision split to double), which the adjoint paid before too;
* one transposed application of the packed map, compiled as the transpose solve applies it, best and
  mean of ``BFS3D_TRANSPOSE_COST_REPEATS``;

and prints the packing in units of transposed applications, beside the count a bfs3d adjoint was
measured to spend (``adjoint_probe.py``, 1454-1575 at the converged zero-shift state). The packing reads
only the operators' entries and pattern, so its cost does not depend on the state; the default state is
the hybrid start. ``BFS3D_TRANSPOSE_COST_STATE`` takes a checkpoint (``checkpoints/state-000NN.npz``).

Usage::

    validation/run_case.sh validation/bfs3d_openfoam/adjoint_transpose_cost.py --wait
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402  (enables x64)
import compare  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.solve import with_transposes  # noqa: E402
from aquaflux.turbulence import open_session  # noqa: E402

REPEATS = int(os.environ.get("BFS3D_TRANSPOSE_COST_REPEATS", "20"))
#: The transposed applications a bfs3d adjoint spent, from the two starting checkpoints ``adjoint_probe.py``
#: was run from (its module docstring).
ADJOINT_APPLICATIONS = (1454, 1575)


def _seconds(f):
    """``f()``'s result and its wall time, waiting for every array it returns."""
    started = time.perf_counter()
    result = f()
    jax.block_until_ready(jax.tree.leaves(result))
    return result, time.perf_counter() - started


def _resident_gib() -> float:
    """The process's resident high-water mark (``ru_maxrss``, KiB on Linux), in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _state(coupled) -> jnp.ndarray:
    state_file = os.environ.get("BFS3D_TRANSPOSE_COST_STATE")
    if state_file:
        with np.load(state_file) as data:
            print(f"state: {state_file}", flush=True)
            return jnp.asarray(data["state"])
    print("state: the hybrid start", flush=True)
    return coupled.state_from_physical(*hybrid_initialize(coupled))


def main():
    print(
        f"[configuration] aquaflux {Path(aquaflux.__file__).parents[1]}, jax {jax.__version__}, "
        f"{jax.default_backend()}, {os.cpu_count()} cpus, repeats {REPEATS}",
        flush=True,
    )
    coupled = compare.build_case()["coupled"]
    state = _state(coupled)
    session = open_session(compare.SOLVER.settings()["preconditioner"], coupled)
    step = session.build(state, dual_time=compare.SOLVER.dual_time)
    factory = step.adjoint_preconditioner_factory
    print(
        f"adjoint factory {type(factory).__name__}, split precision {factory.form.precision}; "
        f"resident high-water before packing {_resident_gib():.2f} GiB",
        flush=True,
    )

    packed, packing = _seconds(lambda: with_transposes(factory))
    print(
        f"packing the transposes: {packing:.2f} s; resident high-water {_resident_gib():.2f} GiB",
        flush=True,
    )
    m, widening = _seconds(lambda: packed(state))
    print(f"the factory's call (widening to double): {widening:.2f} s", flush=True)

    transposed = jax.linear_transpose(m, state)
    apply = jax.jit(lambda u: transposed(u)[0])
    v = jnp.asarray(np.random.default_rng(0).standard_normal(state.shape[0]))
    _, first = _seconds(lambda: apply(v))
    times = []
    for _ in range(REPEATS):
        times.append(_seconds(lambda: apply(v))[1])
    best, mean = min(times), sum(times) / len(times)
    print(
        f"one transposed application: best {best * 1e3:.1f} ms, mean {mean * 1e3:.1f} ms; "
        f"first call (compile) {first:.1f} s",
        flush=True,
    )
    low, high = ADJOINT_APPLICATIONS
    print(
        f"packing = {packing / mean:.1f} mean transposed applications; against {low}-{high} "
        f"applications, {100 * packing / (packing + high * mean):.2f}-"
        f"{100 * packing / (packing + low * mean):.2f} % of a transpose solve's applications plus "
        f"the packing",
        flush=True,
    )


if __name__ == "__main__":
    main()
