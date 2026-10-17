"""What one Krylov application costs at the march's own target-station systems, piece by piece.

An application of the preconditioned operator is ``v -> (J + s) M v``: one field-split apply ``M`` and
one Jacobian-vector product. Which of the two dominates decides what the next optimization should be, so
this times both on the systems the march actually solved, rebuilt by ``replay.py`` from a capture
(``compare.py`` run with ``PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1``), with the preconditioner
refitted where the march refitted it.

At the first inner system of each chosen step it reports, in ms, the best of ``PITZ_COST_REPEATS`` warm
calls of each compiled piece:

* ``apply``     -- the field split ``M v``, as the step applies it (its precision included);
* ``leading``   -- the ``[u, v, p]`` block's cycle alone, ``trailing`` the ``[k, omega]`` block's, and
  ``coupling`` the retained coupling product, each on a vector already in the split's precision;
* ``jvp``       -- ``J v`` alone;
* ``operator``  -- the whole application ``(J + s) M v`` compiled as one program, which is what a
  Krylov iteration inside the step pays; ``operator - apply`` is the in-program JVP-plus-shift share.

Beside each timing it reports the minor page faults one call takes: on Linux a standalone call of a
program whose temporary buffer exceeds glibc's 32 MiB adaptive mmap threshold faults all of it every
call, which a call inside the compiled step does not pay. Set ``MALLOC_MMAP_THRESHOLD_`` and
``MALLOC_TRIM_THRESHOLD_`` to remove them, and say so beside the result.

Settings: ``PITZ_COST_STEPS`` (comma-separated one-based steps, default the last three captured),
``PITZ_COST_REPEATS`` (default 15). The replay starts at the target station, as ``replay.py`` requires.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/application_cost.py --wait
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
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.precondition.materialized_session import jacobian_matvec  # noqa: E402
from replay import MarchReplay  # noqa: E402

REPEATS = int(os.environ.get("PITZ_COST_REPEATS", "15"))


def _best(f, *args):
    """Best wall time of ``f(*args)`` in ms after one warm call, and the minor faults of that best call."""
    jax.block_until_ready(f(*args))
    best, faults = float("inf"), 0
    for _ in range(REPEATS):
        before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        started = time.perf_counter()
        jax.block_until_ready(f(*args))
        elapsed = time.perf_counter() - started
        if elapsed < best:
            best = elapsed
            faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before
    return best * 1e3, faults


_split_apply = eqx.filter_jit(lambda split, v: split.apply(v))
_leading = eqx.filter_jit(lambda split, v: split.leading_cycle(v))
_trailing = eqx.filter_jit(lambda split, v: split.trailing_cycle(v))
_coupling = eqx.filter_jit(lambda split, v: split.coupling.apply(v))
_jvp = eqx.filter_jit(jacobian_matvec)


@eqx.filter_jit
def _operator(assembler, p, shift, split, v):
    z = split.apply(v)
    return jacobian_matvec(assembler, p, z) + shift * z


def measure(replay: MarchReplay, system, v) -> dict:
    """Every piece's best time (ms) and faults at ``system``, on the direction ``v``."""
    split = replay.policy.split
    narrowed = v.astype(split.precision)
    lead = narrowed[: split.groups.n_leading_dofs]
    trail = narrowed[split.groups.n_leading_dofs :]
    first = split.leading_cycle(lead)
    return {
        "apply": _best(_split_apply, split, v),
        "leading": _best(_leading, split, lead),
        "trailing": _best(_trailing, split, trail),
        "coupling": _best(_coupling, split, first),
        "jvp": _best(_jvp, system.assembler, system.p, v),
        "operator": _best(_operator, system.assembler, system.p, system.shift, split, v),
    }


def main():
    replay = MarchReplay()
    print(replay.describe(), flush=True)
    print(
        f"[configuration] aquaflux {Path(aquaflux.__file__).parents[1]}, {os.cpu_count()} cpus, "
        f"best of {REPEATS}; MALLOC_MMAP_THRESHOLD_={os.environ.get('MALLOC_MMAP_THRESHOLD_', 'unset')}",
        flush=True,
    )
    last = max(replay.states)
    wanted = os.environ.get("PITZ_COST_STEPS")
    steps = {int(s) for s in wanted.split(",")} if wanted else {last - 2, last - 1, last}
    print(f"[configuration] split precision {replay.policy.split.precision}; steps {sorted(steps)}")
    rng = np.random.default_rng(0)
    pieces = ("apply", "leading", "trailing", "coupling", "jvp", "operator")
    print("| step | " + " | ".join(f"{p} ms (faults)" for p in pieces) + " | M share |", flush=True)
    for system in replay.systems(last=max(steps)):
        if system.step not in steps or system.inner != 0:
            continue
        v = jnp.asarray(rng.standard_normal(system.b.shape))
        result = measure(replay, system, v)
        share = result["apply"][0] / result["operator"][0]
        cells = " | ".join(f"{result[p][0]:.1f} ({result[p][1]})" for p in pieces)
        print(f"| {system.step} | {cells} | {share:.0%} |", flush=True)


if __name__ == "__main__":
    main()
