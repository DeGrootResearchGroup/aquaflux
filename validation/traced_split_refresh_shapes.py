"""Does a march refresh keep the traced field split's array shapes, so a jitted solve holding it never retraces?

A traced field split rides into the compiled Krylov solve as an argument. That solve is a compilation-cache
hit after a refresh only if every leaf of the rebuilt split -- both hierarchies, their smoother records, the
coupling block -- has the shape and dtype it had before. A block inverse whose coarsening reads operator
values (a nonzero ``strength_threshold`` without ``frozen_coarsening`` or ``shape_headroom``) can move its
shapes on a refresh, and then every refresh would recompile the whole coupled solve.

This fits the case's own session once, then re-fits it through the march's own refresh hook at each
(state, shift) pair below, rebuilding the traced split after each, and reports:

* whether the leaf signature (tree structure, shapes, dtypes) matches the first build's, per block, and
* how many times a jitted function taking the split as an argument was traced across all the rebuilds
  (one means every refresh was a cache hit).

Settings (environment, all optional)::

    TRACED_CASE     pitzdaily (default) | bfs3d
    TRACED_BETAS    comma-separated shifts each state is re-fitted at (default 2.0,0.5,0.05)

States: the hybrid start, then every ``state-*.npz`` in the case's ``checkpoints/`` directory, oldest
first -- so the refresh crosses from a cold field to a developed one, which is where a value-reading
coarsening moves.

Run with ``validation/run_case.sh validation/traced_split_refresh_shapes.py --wait``.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CASES = {"pitzdaily": HERE / "pitzdaily_openfoam", "bfs3d": HERE / "bfs3d_openfoam"}
CASE = os.environ.get("TRACED_CASE", "pitzdaily")
if CASE not in CASES:
    raise SystemExit(f"TRACED_CASE must be one of {sorted(CASES)}, not {CASE!r}")
sys.path.insert(0, str(CASES[CASE]))

import compare  # noqa: E402  (the case's own module, found on the path set above)
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.precondition.traced_field_split import traced_field_split  # noqa: E402
from aquaflux.solve import ConstantRelaxation  # noqa: E402
from aquaflux.turbulence import coupled_step, open_session  # noqa: E402

BETAS = [float(b) for b in os.environ.get("TRACED_BETAS", "2.0,0.5,0.05").split(",")]


def _states(coupled):
    """``(label, state)`` for the hybrid start, then each checkpoint, oldest first."""
    yield "hybrid start", coupled.state_from_physical(*hybrid_initialize(coupled))
    for path in sorted((CASES[CASE] / "checkpoints").glob("state-*.npz")):
        with np.load(path) as data:
            yield path.name, jnp.asarray(data["state"])


def _signature(tree):
    """The tree structure plus every leaf's shape and dtype: what a jit cache key reads of an argument."""
    leaves, treedef = jax.tree.flatten(eqx.filter(tree, eqx.is_array))
    return treedef, tuple((leaf.shape, str(leaf.dtype)) for leaf in leaves)


def _levels(cycle) -> tuple[int, ...]:
    hierarchy = cycle.state[0]
    return tuple(int(level.n) for level in hierarchy.levels)


def main() -> None:
    coupled = compare.build_case()["coupled"]
    spec = compare.SOLVER.preconditioner
    allowed = inspect.signature(coupled_step).parameters
    march = {
        k: v for k, v in compare.SOLVER.settings().items() if k in allowed and k != "preconditioner"
    }
    states = list(_states(coupled))
    session = open_session(spec, coupled)
    step = session.build(states[0][1], **march)
    split = session.preconditioner.inverse

    traces = 0

    @eqx.filter_jit
    def apply(traced, r):
        nonlocal traces
        traces += 1
        return traced.apply(r)

    first = traced_field_split(split)
    reference = {
        "leading": _signature(first.leading_cycle),
        "trailing": _signature(first.trailing_cycle),
        "coupling": _signature(first.coupling),
    }
    probe = jnp.ones(split.groups.n_dofs)
    jax.block_until_ready(apply(first, probe))
    print(
        f"case {CASE}; {split.groups.n_dofs} dofs; betas {BETAS}; "
        f"leading levels {_levels(first.leading_cycle)}, trailing levels {_levels(first.trailing_cycle)}",
        flush=True,
    )
    print(
        f"{'state':<16} {'beta':>6}  leading  trailing  coupling  levels (leading / trailing)",
        flush=True,
    )
    moved = 0
    for label, state in states:
        for beta in BETAS:
            active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(beta))
            session.refresh_preconditioner(active, state, due=True)
            traced = traced_field_split(split)
            same = {
                "leading": _signature(traced.leading_cycle) == reference["leading"],
                "trailing": _signature(traced.trailing_cycle) == reference["trailing"],
                "coupling": _signature(traced.coupling) == reference["coupling"],
            }
            moved += not all(same.values())
            jax.block_until_ready(apply(traced, probe))
            flags = "  ".join(f"{'same' if same[k] else 'MOVED':>8}" for k in same)
            print(
                f"{label:<16} {beta:>6g}  {flags}  "
                f"{_levels(traced.leading_cycle)} / {_levels(traced.trailing_cycle)}",
                flush=True,
            )
    print(
        f"refreshes with a moved signature: {moved}; traces of the jitted apply across "
        f"{1 + len(states) * len(BETAS)} builds: {traces}",
        flush=True,
    )


if __name__ == "__main__":
    main()
