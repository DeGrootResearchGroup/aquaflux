"""Does a float32 preconditioner cost the march convergence? Replayed on pitzDaily's own linear systems.

The preconditioner never enters the state or its gradient, so its precision can only change how fast
each inner GMRES solve converges. A float32 copy halves the memory traffic of the part of an apply that
is bandwidth-bound, and stays a fixed linear map, so plain GMRES remains valid. This probe asks the
half of the question a CPU can answer: on the systems the march actually solved, how many more
iterations does a float32 preconditioner need, and what does one apply cost in each precision.

**What runs in float32.** Both block inverses of the shipped field split -- the SIMPLE-smoothed flow
hierarchy and the Jacobi-smoothed ``[k, omega]`` hierarchy -- apply their own jitted V-cycle to a
float32 copy of their hierarchy and of the right-hand side. The split's retained coupling product, the
operator ``J + s``, the Krylov recurrences and the stopping measure stay in float64. Each block's
float32 hierarchy is re-cast whenever the march refits it in place, so the refits stay the march's.

**The arms**, on the same systems through :class:`replay.MarchReplay`, both with the march's own solver
(the residual-only stop, stopping in the step's measure):

``float64``  the shipped preconditioner. Its cycle count must equal the one the march recorded for every
             solve; a mismatch is reported, since a replay that does not reproduce the march is
             measuring some other sequence.
``float32``  the same, with the two block inverses applied in float32.

Counted per solve: restart cycles and applications of ``(J + s) M``. Timed once per step, on its first
system: one preconditioner apply in each precision, after warm-up, best of ``PITZ_PRECISION_REPEATS``.

Usage
-----
    PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1 \\
        validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    validation/run_case.sh validation/pitzdaily_openfoam/precision_replay_probe.py

``PITZ_REPLAY_FROM`` / ``PITZ_REPLAY_TO`` bound the replayed steps (default: the target station to the
end of the capture).
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
from aquaflux.solve import HierarchyBlockInverse, residual_stop_gmres, restart_cycles  # noqa: E402
from replay import MarchReplay  # noqa: E402

#: Repeats per timed apply; the best is kept, since the slower ones measure other load on the machine.
REPEATS = int(os.environ.get("PITZ_PRECISION_REPEATS", "10"))
PRECISIONS = (jnp.float64, jnp.float32)


def _narrowed(tree, dtype):
    """``tree`` with every floating array cast to ``dtype``; integer index arrays are left alone."""
    return jax.tree.map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, tree)


class InPrecision:
    """A hierarchy block inverse applied in the precision ``setting["dtype"]`` names.

    Everything but ``apply`` is the wrapped inverse's own, so the field split refits it in place
    exactly as before; the narrowed copy is re-cast whenever the hierarchy it was cast from is replaced.
    """

    def __init__(self, inverse: HierarchyBlockInverse, setting: dict):
        if not isinstance(inverse, HierarchyBlockInverse):
            raise TypeError(f"{type(inverse).__name__} holds no hierarchy to narrow")
        self._inverse = inverse
        self._setting = setting
        self._cast_from = None
        self._cast = None

    def __getattr__(self, name):
        return getattr(self._inverse, name)

    def apply(self, residual, *, transpose=False):
        dtype = self._setting["dtype"]
        if dtype == jnp.float64:
            return self._inverse.apply(residual, transpose=transpose)
        if transpose:
            raise NotImplementedError("the forward march never applies the transpose")
        inverse = self._inverse
        source = (inverse._hierarchy, inverse._extras)
        if self._cast_from is None or any(
            a is not b for a, b in zip(self._cast_from, source, strict=True)
        ):
            self._cast = _narrowed(source, dtype)
            self._cast_from = source
        hierarchy, extras = self._cast
        out = inverse.cycle()(hierarchy, extras, jnp.asarray(residual, dtype), inverse.smoother())
        self._setting["seen"].add(str(out.dtype))
        return np.asarray(out, dtype=np.float64)


def _float64_operations(inverse: HierarchyBlockInverse) -> tuple[int, int]:
    """``(float64 equations, all equations)`` in the cycle traced on float32 inputs.

    A float64 constant inside the cycle would promote everything after it, so a cycle that returns
    float32 can still have done part of its work in float64; this counts how much.
    """
    hierarchy, extras = _narrowed((inverse._hierarchy, inverse._extras), jnp.float32)
    vector = jnp.zeros(inverse.n_dofs, jnp.float32)
    jaxpr = jax.make_jaxpr(lambda h, e, v: inverse.cycle()(h, e, v, inverse.smoother()))(
        hierarchy, extras, vector
    )
    total, wide = 0, 0

    def nested(value):
        if hasattr(value, "eqns"):
            yield value
        elif hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
            yield value.jaxpr
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from nested(item)

    def walk(jaxpr):
        nonlocal total, wide
        for equation in jaxpr.eqns:
            total += 1
            wide += any(getattr(v.aval, "dtype", None) == jnp.float64 for v in equation.outvars)
            for value in equation.params.values():
                for sub in nested(value):
                    walk(sub)

    walk(jaxpr.jaxpr)
    return wide, total


def _best_apply(split, vector) -> float:
    split.apply(vector)  # compile and warm
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        split.apply(vector)
        best = min(best, time.perf_counter() - started)
    return best


def main():
    replay = MarchReplay()
    linear = replay.solver.linear_solve
    print(f"{replay.describe()}; arms float64, float32; repeats {REPEATS}", flush=True)
    split = replay.preconditioner.factors
    setting = {"dtype": jnp.float64, "seen": set()}
    split._leading = InPrecision(split._leading, setting)
    split._trailing = InPrecision(split._trailing, setting)
    for name, block in (("leading", split._leading), ("trailing", split._trailing)):
        wide, total = _float64_operations(block._inverse)
        print(
            f"[{name} {type(block._inverse).__name__}] {block.n_dofs} dofs; float32 cycle: "
            f"{wide} of {total} equations produce float64",
            flush=True,
        )

    applications = {"count": 0}

    def counted(solved, cycles):
        applications["count"] = int(solved)

    totals = {str(np.dtype(d)): {"cycles": 0, "applications": 0} for d in PRECISIONS}
    timings = {str(np.dtype(d)): 0.0 for d in PRECISIONS}
    mismatches = timed_steps = 0
    checked = False
    timed_step = None
    print(
        f"\n{'step':>4} {'in':>2} {'cyc':>3} | {'f64 cyc':>7} {'app':>4} | {'f32 cyc':>7} {'app':>4}"
        f" | {'f64 apply':>9} {'f32 apply':>9}",
        flush=True,
    )
    for system in replay.systems():
        solver = residual_stop_gmres(
            linear.rtol,
            norm=system.measure,
            restart=linear.restart,
            max_restarts=linear.max_restarts,
            on_solve=counted,
        )
        if not checked:
            # The counting solver must be the march's own, but for its observer.
            same = residual_stop_gmres(
                linear.rtol,
                norm=system.measure,
                restart=linear.restart,
                max_restarts=linear.max_restarts,
            )
            if not eqx.tree_equal(same, replay.march_solver(system.measure)):
                raise SystemExit("the march's solver is not the residual stop at these settings")
            checked = True
        row = {}
        for dtype in PRECISIONS:
            key = str(np.dtype(dtype))
            setting["dtype"] = dtype
            raw = int(replay.solve(system, solver)[1])
            jax.effects_barrier()
            row[key] = (restart_cycles(raw), applications["count"])
            totals[key]["cycles"] += row[key][0]
            totals[key]["applications"] += row[key][1]
        mismatches += row["float64"][0] != system.recorded
        apply_times = ""
        if system.step != timed_step:
            timed_step = system.step
            timed_steps += 1
            vector = np.asarray(system.b)
            for dtype in PRECISIONS:
                setting["dtype"] = dtype
                seconds = _best_apply(split, vector)
                timings[str(np.dtype(dtype))] += seconds
                apply_times += f" {seconds * 1e3:8.1f}ms"
        setting["dtype"] = jnp.float64
        print(
            f"{system.step:4d} {system.inner:2d} {system.recorded:3d} | {row['float64'][0]:7d}"
            f"{'!' if row['float64'][0] != system.recorded else ' '}{row['float64'][1]:4d} | "
            f"{row['float32'][0]:7d} {row['float32'][1]:4d} |{apply_times}",
            flush=True,
        )
    print(
        "\ntotals: "
        + ", ".join(
            f"{k} {v['cycles']} cycles / {v['applications']} applications"
            for k, v in totals.items()
        ),
        flush=True,
    )
    if timed_steps:
        print(
            f"mean best apply over {timed_steps} steps: "
            + ", ".join(f"{k} {v / timed_steps * 1e3:.1f} ms" for k, v in timings.items()),
            flush=True,
        )
    print(f"float32 cycle output dtypes seen: {sorted(setting['seen'])}", flush=True)
    if mismatches:
        print(
            f"⚠️ {mismatches} float64 solves did not reproduce the march's recorded cycle count",
            flush=True,
        )


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[done in {time.perf_counter() - t0:.0f} s]")
