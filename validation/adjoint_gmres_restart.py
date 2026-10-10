"""Does the adjoint's default GMRES want a longer restart than lineax's 20?

``aquaflux.solve.default_linear_solver`` is ``lx.GMRES(rtol=1e-10, atol=1e-10)`` at lineax's own
``restart=20, stagnation_iters=20``. It is the solver every implicit-function-theorem adjoint falls back
to, and that one transpose solve sets the gradient's accuracy. This sweeps ``restart`` on the adjoint
alone -- the forward solve is untouched, so every arm differentiates the same converged state -- and
records, per arm:

* operator applications (each is one transpose Jacobian product and one transposed-preconditioner
  application; counted by wrapping the transpose matvec, so the count is exact rather than inferred
  from restart cycles);
* lineax's raw cycle count;
* the TRUE relative residual ``|J^T lam - g| / |g|`` of the returned adjoint, computed after the solve
  with an uncounted product -- lineax's own stop is a componentwise test, which is not this number;
* wall clock of the compiled backward pass (median of ``TIMINGS`` calls after the first, so compile is
  excluded; the first call is reported separately);
* the gradient, against a central finite difference of the same discrete solve and against the
  ``restart=20`` arm.

``stagnation_iters`` is held at lineax's 20 in every arm, so ``restart`` is the only variable. It counts
restart *cycles* without a new residual minimum, so it costs nothing on a solve that converges.

Cases (the gradient tests the question was raised against):

* ``cavity n``: ``tests/integration/test_flow_adjoint.py``'s skewed lid-driven cavity (perturb 0.15,
  Navier--Stokes, first-order upwind, corrected Green--Gauss with 16 swept sweeps), ``mu = 0.02``,
  objective ``mean |u_x|``, block preconditioner transposed for the adjoint. ``n`` = 12 and 16 are the
  432- and 768-unknown systems; 24 and 32 extend it.
* ``rans``: ``tests/integration/test_coupled_rans.py``'s 28x20 graded turbulent channel (k-omega SST,
  Re 2500), objective ``sum k^2`` w.r.t. a molecular-viscosity scale, ``BlockDiagonal(ScalarTwoLevel,
  ConvectionTwoLevel)`` preconditioner, default ``solve_coupled`` adjoint path.

Run: ``validation/run_case.sh validation/adjoint_gmres_restart.py`` (all cases), or
``python3 validation/adjoint_gmres_restart.py [cavity|rans]`` for one.

Recorded 2026-10-09 at ``81a05c5``: jax 0.11.2, lineax 0.1.1, CPU, Linux x86_64, 4 cores. Each cell is
operator applications / compiled backward seconds:

=================  =========  ===========  ============  ===========  ===========
case               unknowns   restart 10   restart 20    restart 40   restart 80
=================  =========  ===========  ============  ===========  ===========
cavity n=12        432        78 / 0.081   85 / 0.080    83 / 0.082   163 / 0.131
cavity n=16        768        100 / 0.118  85 / 0.089    124 / 0.148  163 / 0.188
cavity n=24        1728       122 / 0.259  106 / 0.222   124 / 0.252  163 / 0.328
cavity n=32        3072       155 / 0.630  127 / 0.543   124 / 0.596  163 / 0.718
rans channel       2800       stagnated    736 / 0.998   698 / 0.892  487 / 0.704
=================  =========  ===========  ============  ===========  ===========

Every converged arm reached a true residual of 3e-9 or below and agreed with the ``restart=20`` gradient
to 4e-11 relative. ``restart=80`` costs exactly ``2 (80 + 1) + 1 = 163`` on every cavity: lineax completes
a cycle before it tests convergence and then needs one more to see the solution stop moving, so a solve
that converged inside its first cycle pays for two.
"""

from __future__ import annotations

import importlib
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

RESTARTS = (10, 20, 40, 80)
STAGNATION_ITERS = 20
TOL = 1e-10
TIMINGS = 3
# ``aquaflux.solve.root_adjoint`` is the re-exported function, which shadows the module of that name.
root_adjoint_module = importlib.import_module("aquaflux.solve.root_adjoint")
CAVITY_SIZES = (12, 16, 24, 32)


# --- instrumentation of the transpose solve (harness-only; the shipped function is restored on exit) ---

_RECORD: dict = {}
_shipped_solve_linear = root_adjoint_module.solve_linear


def _count(_):
    _RECORD["matvecs"] = _RECORD.get("matvecs", 0) + 1


def _note(cycles, true_residual):
    _RECORD["cycles"] = int(cycles)
    _RECORD["true_residual"] = float(true_residual)


def _instrumented_solve_linear(matvec, b, solver=None, preconditioner=None, **kwargs):
    def counted(u):
        jax.debug.callback(_count, u[0])
        return matvec(u)

    x, cycles = _shipped_solve_linear(
        counted, b, solver=solver, preconditioner=preconditioner, **kwargs
    )
    true_residual = jnp.linalg.norm(matvec(x) - b) / jnp.linalg.norm(b)
    jax.debug.callback(_note, cycles, true_residual)
    return x, cycles


def _gmres(restart):
    return lx.GMRES(rtol=TOL, atol=TOL, restart=restart, stagnation_iters=STAGNATION_ITERS)


def _sweep(label, make_objective, x0, fd_step):
    """Run every restart arm on one case and print its table."""
    base = make_objective(None)
    t = time.perf_counter()
    fd = float((base(x0 + fd_step) - base(x0 - fd_step)) / (2.0 * fd_step))
    print(
        f"\n[{label}] central FD (h={fd_step:g}) = {fd:.10e}   ({time.perf_counter() - t:.1f}s)",
        flush=True,
    )
    print(
        f"{'restart':>7} {'matvecs':>8} {'cycles':>6} {'true |r|/|b|':>13} {'1st bwd s':>9} "
        f"{'bwd s':>8} {'gradient':>18} {'rel vs FD':>10} {'rel vs r20':>10}",
        flush=True,
    )
    rows = {}
    for restart in RESTARTS:
        objective = make_objective(_gmres(restart))
        _, backward = jax.vjp(objective, jnp.asarray(x0))
        backward = jax.jit(backward)
        try:
            _RECORD.clear()
            t = time.perf_counter()
            (g,) = backward(jnp.asarray(1.0))
            g = float(jax.block_until_ready(g))
            first = time.perf_counter() - t
            record = dict(_RECORD)
            times = []
            for _ in range(TIMINGS):
                _RECORD.clear()
                t = time.perf_counter()
                jax.block_until_ready(backward(jnp.asarray(1.0)))
                times.append(time.perf_counter() - t)
            rows[restart] = g
            ref = rows.get(20)
            print(
                f"{restart:>7} {record['matvecs']:>8} {record['cycles']:>6} {record['true_residual']:>13.3e} "
                f"{first:>9.2f} {statistics.median(times):>8.3f} {g:>18.10e} {abs(g - fd) / abs(fd):>10.2e} "
                f"{'' if ref is None else f'{abs(g - ref) / abs(ref):.2e}':>10}",
                flush=True,
            )
        except Exception as error:  # a stagnated / capped solve raises; record it and move on
            print(
                f"{restart:>7}  FAILED: {type(error).__name__}: {str(error).splitlines()[0]}",
                flush=True,
            )


def cavity():
    from aquaflux.discretization import FirstOrderUpwind
    from aquaflux.flow import BlockPreconditioner
    from aquaflux.solve import DampedNewtonStep, RootSolver
    from tests.integration.test_flow_adjoint import _cavity, _mean_speed, _residual

    mu0 = 0.02
    for n in CAVITY_SIZES:
        precond = BlockPreconditioner.build(_cavity(mu0, n, FirstOrderUpwind())).factory()

        def make_objective(adjoint_solver, n=n, precond=precond):
            solver = RootSolver(
                max_steps=20,
                strategy=DampedNewtonStep(preconditioner=precond),
                adjoint_solver=adjoint_solver,
            )

            def f(mu):
                assembler = _cavity(mu, n, FirstOrderUpwind())
                state = solver.solve(_residual, assembler.initial_state(), assembler)
                return _mean_speed(assembler, state)

            return f

        _sweep(f"cavity n={n} ({3 * n * n} unknowns)", make_objective, mu0, 1e-4)


def rans():
    from aquaflux.turbulence import BlockDiagonal, ScalarTwoLevel, coupled_step, sst_initial_fields
    from aquaflux.turbulence.coupled import CoupledRANS, solve_coupled
    from tests.integration.test_coupled_rans import PRECONDITIONER, _channel

    _, momentum, turbulence, _, _ = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    flow_ws, k_ws, omega_ws = sst_initial_fields(momentum, turbulence)
    continuation = coupled_step(
        coupled,
        coupled.pack_state(flow_ws, k_ws, omega_ws),
        preconditioner=BlockDiagonal(scalar=ScalarTwoLevel(), **PRECONDITIONER),
    )

    def make_objective(adjoint_solver):
        def objective(nu_scale):
            scaled = eqx.tree_at(
                lambda c: c.turbulence.molecular_viscosity,
                coupled,
                coupled.turbulence.molecular_viscosity * nu_scale,
            )
            _, k, _ = solve_coupled(
                scaled,
                flow_ws,
                k_ws,
                omega_ws,
                strategy=continuation,
                max_steps=40,
                **({} if adjoint_solver is None else {"adjoint_solver": adjoint_solver}),
            )
            return jnp.sum(k**2)

        return objective

    n_dof = int(coupled.pack_state(flow_ws, k_ws, omega_ws).size)
    _sweep(f"rans channel 28x20 ({n_dof} unknowns)", make_objective, 1.0, 1e-4)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(
        f"jax {jax.__version__}, lineax {lx.__version__}, numpy {np.__version__}, {platform.platform()}, "
        f"{os.cpu_count()} cores; adjoint GMRES rtol=atol={TOL:g}, stagnation_iters={STAGNATION_ITERS}",
        flush=True,
    )
    root_adjoint_module.solve_linear = _instrumented_solve_linear
    try:
        if which in ("cavity", "all"):
            cavity()
        if which in ("rans", "all"):
            rans()
    finally:
        root_adjoint_module.solve_linear = _shipped_solve_linear
