"""The host field split against the traced one: per application, and inside a fixed-work GMRES.

The shipped coupled preconditioner is a block-triangular field split whose two blocks are traced multigrid
cycles, composed in ``numpy`` on the host (``FieldSplitInverse``) and reached from the jitted Krylov solve
through a ``jax.pure_callback``. ``TracedFieldSplit`` composes the SAME two cycles and the same coupling
block as one traced map, so nothing crosses the device boundary between the Krylov solve and the
preconditioned vector it gets back. On a CPU the two should cost the same; on an accelerator the host
arm pays a device-host round trip (several, counting the cycles it re-enters) on every application, which
is the cost this harness exists to measure.

Both arms are built from ONE fitted preconditioner -- the case's own session, at a state from the case's
own march, re-fitted at the shift chosen below -- so they apply the same operator, and the harness checks
that before timing anything. Both are timed inside a jitted program against the march's own matrix-free
operator ``J(state) + beta d``:

* one application of the preconditioner, and
* a GMRES of FIXED work (``TRACED_RESTART`` iterations per cycle, ``TRACED_CYCLES`` cycles, no stopping
  test), so each arm does exactly the same number of operator and preconditioner applications and the
  difference is the preconditioner's cost and nothing else.

The operator's own matvec is timed too, as the share the preconditioner competes with.

Settings (environment, all optional)::

    TRACED_CASE      pitzdaily (default) | bfs3d -- the case whose compare.py builds the problem and the
                     preconditioner spec
    TRACED_STATE     a state-*.npz written by that case's march (default: its newest checkpoint, else its
                     hybrid start)
    TRACED_BETA      the shift the inverse is re-fitted at and the operator carries (default: the spec's
                     refit_beta_floor, else 0.05)
    TRACED_REPEATS   timed repetitions per measurement after one warm-up (default 20)
    TRACED_RESTART, TRACED_CYCLES   the fixed GMRES shape (defaults 15 and 4)

Run with ``validation/run_case.sh validation/traced_split_benchmark.py --wait``. Read the ratios, not
the seconds, across machines; the state's difficulty does not enter (both arms apply one fixed operator),
so a converged checkpoint is as good as any.
"""

from __future__ import annotations

import inspect
import os
import statistics
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
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import lineax as lx  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    frozen_shift_diagonal,
    jacobian_matvec,
)
from aquaflux.precondition.traced_field_split import (  # noqa: E402
    contains_host_callback,
    traced_field_split,
)
from aquaflux.solve import ConstantRelaxation, solve_linear  # noqa: E402
from aquaflux.turbulence import coupled_step, open_session  # noqa: E402

REPEATS = int(os.environ.get("TRACED_REPEATS", "20"))
RESTART = int(os.environ.get("TRACED_RESTART", "15"))
CYCLES = int(os.environ.get("TRACED_CYCLES", "4"))


def _state(coupled) -> tuple[jnp.ndarray, str]:
    """The march state to linearize at, and where it came from."""
    named = os.environ.get("TRACED_STATE", "")
    paths = [Path(named)] if named else sorted((CASES[CASE] / "checkpoints").glob("state-*.npz"))
    if paths:
        with np.load(paths[-1]) as data:
            return jnp.asarray(data["state"]), str(paths[-1])
    return coupled.state_from_physical(*hybrid_initialize(coupled)), "hybrid start"


def _timed(fn, *args) -> tuple[float, float]:
    """Median and minimum seconds of ``fn(*args)`` over ``REPEATS`` runs, after one warm-up compile."""
    jax.block_until_ready(fn(*args))
    seconds = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds), min(seconds)


def main() -> None:
    coupled = compare.build_case()["coupled"]
    solver_spec = compare.SOLVER
    spec = solver_spec.preconditioner
    state, source = _state(coupled)
    residual = coupled.residual(state)
    if not bool(jnp.all(jnp.isfinite(residual))):
        raise SystemExit(f"the state from {source} gives a non-finite residual; pick another")
    beta = float(os.environ.get("TRACED_BETA", "") or (spec.refit_beta_floor or 0.05))

    # The case's own session and step, fitted at `state`, then re-fitted at `beta` through the march's own
    # refresh hook -- the path a march takes, so the inverse is the one a march would apply.
    allowed = inspect.signature(coupled_step).parameters
    march = {
        k: v for k, v in solver_spec.settings().items() if k in allowed and k != "preconditioner"
    }
    session = open_session(spec, coupled)
    step = session.build(state, **march)
    active = eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(beta))
    session.refresh_preconditioner(active, state, due=True)
    policy = step.shift_policy
    host_pc = session.preconditioner
    split = host_pc.inverse
    traced = traced_field_split(split)
    shift = jnp.asarray(frozen_shift_diagonal(policy.base, beta, state))
    b = -residual

    host_apply = jax.jit(host_pc.matvec())

    @eqx.filter_jit
    def traced_apply(split, r):
        return split.apply(r)

    @eqx.filter_jit
    def operator(assembler, p, shift, v):
        return jacobian_matvec(assembler, p, v) + shift * v

    gmres = lx.GMRES(
        rtol=0.0, atol=0.0, restart=RESTART, max_steps=CYCLES, stagnation_iters=CYCLES + 1
    )

    @eqx.filter_jit
    def solve_host(assembler, p, shift, b):
        return solve_linear(
            lambda v: jacobian_matvec(assembler, p, v) + shift * v,
            b,
            gmres,
            host_pc.matvec(),
            throw=False,
        )[0]

    @eqx.filter_jit
    def solve_traced(split, assembler, p, shift, b):
        return solve_linear(
            lambda v: jacobian_matvec(assembler, p, v) + shift * v,
            b,
            gmres,
            split.matvec(),
            throw=False,
        )[0]

    probe = jax.random.normal(jax.random.PRNGKey(0), b.shape, dtype=b.dtype)
    agreement = float(
        jnp.max(jnp.abs(host_apply(probe) - traced_apply(traced, probe)))
        / jnp.max(jnp.abs(host_apply(probe)))
    )
    device = jax.devices()[0]
    print(
        f"case {CASE}; state {source}; beta {beta:g}; {split.groups.n_dofs} dofs "
        f"({split.groups.n_leading_dofs} leading, {split.groups.n_dofs - split.groups.n_leading_dofs} "
        f"trailing); coupling nnz {split.coupling.nnz}",
        flush=True,
    )
    print(
        f"jax {jax.__version__}, backend {jax.default_backend()}, device {device.device_kind}; "
        f"{REPEATS} repeats; GMRES restart {RESTART} x {CYCLES} cycles, no stopping test",
        flush=True,
    )
    print(
        f"host split emits a host callback: {contains_host_callback(host_apply, probe)}; "
        f"traced split: {contains_host_callback(lambda r: traced_apply(traced, r), probe)}",
        flush=True,
    )
    print(f"agreement of one application, max relative difference: {agreement:.1e}", flush=True)
    if agreement > 1e-10:
        raise SystemExit("the two arms do not apply the same operator; nothing to compare")

    rows = [
        ("operator J + beta d, one matvec", _timed(operator, coupled, state, shift, probe), None),
        ("preconditioner, host split", _timed(host_apply, probe), "host"),
        ("preconditioner, traced split", _timed(traced_apply, traced, probe), "traced"),
        ("fixed GMRES, host split", _timed(solve_host, coupled, state, shift, b), "host"),
        (
            "fixed GMRES, traced split",
            _timed(solve_traced, traced, coupled, state, shift, b),
            "traced",
        ),
    ]
    print(f"{'measurement':<36} {'median ms':>10} {'min ms':>10}", flush=True)
    for name, (median, best), _ in rows:
        print(f"{name:<36} {1e3 * median:>10.2f} {1e3 * best:>10.2f}", flush=True)
    apply_ratio = rows[1][1][0] / rows[2][1][0]
    gmres_ratio = rows[3][1][0] / rows[4][1][0]
    print(f"host / traced, one application: {apply_ratio:.2f}x; fixed GMRES: {gmres_ratio:.2f}x")
    x_host = solve_host(coupled, state, shift, b)
    x_traced = solve_traced(traced, coupled, state, shift, b)
    for name, x in (("host", x_host), ("traced", x_traced)):
        true = jnp.linalg.norm(b - operator(coupled, state, shift, x)) / jnp.linalg.norm(b)
        print(f"fixed GMRES true relative residual, {name}: {float(true):.3e}", flush=True)


if __name__ == "__main__":
    main()
