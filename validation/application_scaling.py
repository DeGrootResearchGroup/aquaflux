"""How each piece of one Krylov application scales from pitzDaily to bfs3d, measured at one state per case.

A step's restart cycle is a run of applications of ``v -> (J + s) M v``: one field-split apply ``M``
(its leading ``[velocity, p]`` V-cycle, its trailing ``[k, omega]`` V-cycle and the coupling product
between them) and one Jacobian-vector product. ``pitzdaily_openfoam/application_cost.py`` times these on
pitzDaily's replayed systems; this times the same pieces on either step case (``SCALING_CASE``), at its
hybrid start or a saved state, so that the two cases can be set side by side and the piece whose cost grows
faster than the unknowns can be named.

For each piece it reports the MEAN of ``SCALING_REPEATS`` back-to-back warm calls (a Krylov loop makes
many, so the mean is what it pays; a best call understates it), the best call, the cores the piece kept
busy (processor time over wall time), and the minor page faults
per call (on Linux a standalone call whose temporary exceeds glibc's mmap threshold faults all of it on
every call, which a loop inside the compiled step does not pay). It also prints each hierarchy's levels
(size, stored entries, widest and mean row), the unknowns and the cells, and every time divided by the
unknowns, which is the column to compare between the two cases.

The split is the traced copy the session's first fit hands the step (its precision included), at the
case's own preconditioner settings. Every re-fit keeps its shapes (both cases freeze the coarsening), so
the shift it was fitted at does not change what one application costs.

Usage
-----
    SCALING_CASE=pitzdaily validation/run_case.sh validation/application_scaling.py --wait
    SCALING_CASE=bfs3d validation/run_case.sh validation/application_scaling.py --wait

``SCALING_STATE`` is a ``.npz`` holding ``state`` (default the hybrid start); ``SCALING_REPEATS`` the
timed calls per piece (default 30).
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from step_cases import load_case  # noqa: E402

CASE = os.environ.get("SCALING_CASE", "bfs3d")
STATE = os.environ.get("SCALING_STATE")
REPEATS = int(os.environ.get("SCALING_REPEATS", "30"))


def _timed(f, *args):
    """``(mean ms, best ms, cores, faults per call)`` of ``REPEATS`` back-to-back calls after a warm call.

    ``cores`` is the process's processor time over the wall time of all the calls: how many cores the
    piece kept busy on average.
    """
    import jax

    jax.block_until_ready(f(*args))
    walls = []
    before = resource.getrusage(resource.RUSAGE_SELF)
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(f(*args))
        walls.append(time.perf_counter() - started)
    after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    faults = (after.ru_minflt - before.ru_minflt) / REPEATS
    return 1e3 * sum(walls) / len(walls), 1e3 * min(walls), cpu / sum(walls), faults


def _levels(name, inverse):
    """Print each level of ``inverse``'s hierarchy: size, stored entries, widest and mean row."""
    import numpy as np

    for i, level in enumerate(inverse._hierarchy.levels):
        lengths = np.diff(level.operator.to_scipy().indptr)
        print(
            f"  {name} level {i}: {level.n} dofs, {int(lengths.sum())} entries, "
            f"widest row {int(lengths.max())}, mean row {lengths.mean():.1f}",
            flush=True,
        )


def main():
    case = load_case(CASE)

    import aquaflux  # noqa: F401  (enables x64)
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.initialization import hybrid_initialize
    from aquaflux.precondition.materialized_session import jacobian_matvec
    from aquaflux.turbulence import open_session

    coupled = case.build_case()["coupled"]
    if STATE:
        with np.load(STATE) as data:
            state = jnp.asarray(data["state"])
    else:
        state = coupled.state_from_physical(*hybrid_initialize(coupled))
    if not bool(jnp.all(jnp.isfinite(coupled.residual(state)))):
        raise SystemExit("the starting residual is not finite; the state does not fit this case")
    print(
        f"[configuration] case {CASE}, state {STATE or 'the hybrid start'}, "
        f"jax {jax.__version__}, {os.cpu_count()} cpus, {REPEATS} calls per piece",
        flush=True,
    )

    session = open_session(case.SOLVER.settings()["preconditioner"], coupled)
    session.build(state, dual_time=case.SOLVER.dual_time)
    inverse = session.preconditioner.inverse
    _levels("leading", inverse._leading)
    _levels("trailing", inverse._trailing)
    split = session._form
    n = split.groups.n_dofs
    print(
        f"{n // coupled.layout.n_fields} cells, {n} unknowns, split in {split.precision}",
        flush=True,
    )
    v = jnp.asarray(np.random.default_rng(0).standard_normal(n))
    narrowed = v.astype(split.precision)
    lead = narrowed[: split.groups.n_leading_dofs]
    trail = narrowed[split.groups.n_leading_dofs :]
    first = split.leading_cycle(lead)

    pieces = {
        "residual R(x)": (eqx.filter_jit(lambda a, x: a.residual(x)), (coupled, state)),
        "JVP J v": (eqx.filter_jit(jacobian_matvec), (coupled, state, v)),
        "split apply M v": (eqx.filter_jit(lambda s, x: s.apply(x)), (split, v)),
        "  leading cycle": (eqx.filter_jit(lambda s, x: s.leading_cycle(x)), (split, lead)),
        "  trailing cycle": (eqx.filter_jit(lambda s, x: s.trailing_cycle(x)), (split, trail)),
        "  coupling product": (eqx.filter_jit(lambda s, x: s.coupling.apply(x)), (split, first)),
    }
    print(
        "| piece | mean ms | best ms | cores | faults/call | ns per unknown (mean) |\n|---|---|---|---|---|---|"
    )
    for name, (f, args) in pieces.items():
        mean, best, cores, faults = _timed(f, *args)
        print(
            f"| {name} | {mean:.1f} | {best:.1f} | {cores:.2f} | {faults:.0f} | {1e6 * mean / n:.0f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
