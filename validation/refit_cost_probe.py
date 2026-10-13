"""Where a preconditioner re-fit's refactor phase spends its time, on a case's own operator.

A march re-fit has three phases (``probe``, ``assemble``, ``refactor``); this isolates the last. It fits
the case's own session once, captures the shifted operator the refresh hands the inverse, and then
calls the inverse's ``refactor_block`` on it ``REFIT_REPEATS`` times, reporting the best wall time for
the whole split and for each block, and a profile of one call's top functions by cumulative time.

Settings (environment, all optional)::

    TRACED_CASE     pitzdaily (default) | bfs3d
    REFIT_STATE     checkpoint file name under the checkpoint directory (default: the latest)
    REFIT_CHECKPOINTS  the checkpoint directory (default: the case's ``checkpoints/``), so a worktree at
                    an older commit can be timed on the same state
    REFIT_BETA      the shift the operator is taken at (default 0.05)
    REFIT_REPEATS   timed repeats per measurement (default 5)

Run with ``validation/run_case.sh validation/refit_cost_probe.py --wait``.
"""

from __future__ import annotations

import cProfile
import inspect
import io
import os
import pstats
import sys
import time
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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.solve import ConstantRelaxation  # noqa: E402
from aquaflux.turbulence import coupled_step, open_session  # noqa: E402

BETA = float(os.environ.get("REFIT_BETA", "0.05"))
REPEATS = int(os.environ.get("REFIT_REPEATS", "5"))


def _state():
    directory = Path(os.environ.get("REFIT_CHECKPOINTS", CASES[CASE] / "checkpoints"))
    name = os.environ.get("REFIT_STATE")
    path = directory / name if name else sorted(directory.glob("state-*.npz"))[-1]
    with np.load(path) as data:
        return path.name, jnp.asarray(data["state"])


def _best(f):
    """Best wall time of ``f()`` in seconds over ``REPEATS`` calls; the slower ones measure other load."""
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    coupled = compare.build_case()["coupled"]
    spec = compare.SOLVER.preconditioner
    allowed = inspect.signature(coupled_step).parameters
    march = {
        k: v for k, v in compare.SOLVER.settings().items() if k in allowed and k != "preconditioner"
    }
    label, state = _state()
    session = open_session(spec, coupled)
    step = session.build(state, **march)
    # ``preconditioner`` became a public property with the traced split (#290); before it, private.
    host = getattr(session, "preconditioner", None) or session._preconditioner
    split = host.inverse

    # Capture the operator the refresh hands the split, by running the march's own refresh hook once.
    captured = {}
    refactor = type(split).refactor_block

    def capture(self, block):
        captured["block"] = block
        return refactor(self, block)

    type(split).refactor_block = capture
    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(BETA))
    if "due" in inspect.signature(session.refresh_preconditioner).parameters:
        session.refresh_preconditioner(active, state, due=True)
    else:  # before a re-fit ended the step (#290), the hook took no ``due``
        session.refresh_preconditioner(active, state)
    type(split).refactor_block = refactor
    block = captured["block"]
    blocks = split.groups.blocks(block)
    print(
        f"case {CASE}; state {label}; beta {BETA}; {block.shape[0]} dofs, {block.nnz} nnz; "
        f"repeats {REPEATS}",
        flush=True,
    )

    whole = _best(lambda: split.refactor_block(block))
    leading = _best(lambda: split._leading.refactor_block(blocks[0]))
    trailing = _best(lambda: split._trailing.refactor_block(blocks[3]))
    print(
        f"refactor_block best of {REPEATS}: split {whole:.3f} s; leading {leading:.3f} s; "
        f"trailing {trailing:.3f} s",
        flush=True,
    )

    profile = cProfile.Profile()
    profile.enable()
    split.refactor_block(block)
    profile.disable()
    out = io.StringIO()
    pstats.Stats(profile, stream=out).sort_stats("cumulative").print_stats(30)
    print(out.getvalue(), flush=True)


if __name__ == "__main__":
    main()
