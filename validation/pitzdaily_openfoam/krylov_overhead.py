"""Where one Krylov iteration's time goes inside the march's own compiled solve, at its own systems.

``application_cost.py`` times the pieces of an application ``(J + s) M v`` as standalone programs. A
Krylov iteration inside the step pays those pieces inside a ``while_loop``, together with the GMRES
bookkeeping around them (two classical Gram--Schmidt passes over the basis, the small least-squares
problem, the measured stop on the Arnoldi residual), and allocates its buffers once per solve rather than
once per piece. This times the solve the step runs -- ``solve_linear`` over ``residual_stop_gmres`` in the
step's row-scaled measure, with the step's restart -- on the systems ``replay.py`` rebuilds from a capture
(``compare.py`` with ``PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1``), in five arms:

* ``march``     -- the march's own solve at the tolerance the march solved the system to, timed per
  operator application (one per Krylov iteration plus one true residual, counted by ``on_solve``);
* ``full``      -- the same compiled solve at a fixed budget of ``PITZ_OVERHEAD_CYCLES`` restart cycles
  (a tolerance no cycle meets, so every arm runs exactly ``restart x cycles`` iterations and none
  recomputes ``b - A x``), with the real operator and preconditioner;
* ``identity``  -- that budget with the preconditioner left out (operator ``J + s``);
* ``stand-in``  -- that budget with the operator replaced by a diagonal scaling of the same shape, which
  costs next to nothing, so what remains is the Arnoldi, least-squares, measured-stop and basis traffic;
* ``apply``     -- that budget with the operator ``s M`` (the preconditioner and a diagonal), so that
  ``apply - stand-in`` is the preconditioner inside the loop without the JVP's share of memory traffic;
* ``alone``     -- ``(J + s) M v``, ``M v`` and ``J v`` as standalone compiled programs (those of
  ``application_cost.py``), each called as many times back to back as a budgeted arm iterates, so their
  per-call figure is a mean over the same number of applications, not a best single call.

Every timing is the best of ``PITZ_COST_REPEATS`` runs of its whole budget. From the budgeted arms, per iteration: bookkeeping = ``stand-in``, JVP = ``identity - stand-in``,
preconditioner = ``full - identity`` (and ``apply - stand-in`` as a second reading of it). Beside each
timing are the minor page faults of that best call. Settings: ``PITZ_COST_STEPS`` (one-based steps,
default the last three captured), ``PITZ_COST_REPEATS`` (default 7), ``PITZ_OVERHEAD_CYCLES`` (default 4).

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/krylov_overhead.py --wait
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
from aquaflux.precondition.materialized_session import jacobian_matvec  # noqa: E402
from aquaflux.solve import residual_stop_gmres, solve_linear  # noqa: E402
from replay import MarchReplay, _matvec  # noqa: E402

REPEATS = int(os.environ.get("PITZ_COST_REPEATS", "7"))
CYCLES = int(os.environ.get("PITZ_OVERHEAD_CYCLES", "4"))
#: A tolerance no restart cycle meets, so a budgeted solve runs exactly ``restart x CYCLES`` iterations.
UNREACHABLE = 1e-30


def _best(f, *args, calls=1):
    """Best wall time of ``calls`` successive ``f(*args)`` in ms after one warm call, and the minor faults
    of that best run."""
    jax.block_until_ready(f(*args))
    best, faults = float("inf"), 0
    for _ in range(REPEATS):
        before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        started = time.perf_counter()
        for _ in range(calls):
            jax.block_until_ready(f(*args))
        elapsed = time.perf_counter() - started
        if elapsed < best:
            best, faults = elapsed, resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before
    return best * 1e3, faults


# Each arm is one compiled program taking every array as an argument, so the systems share it.
@eqx.filter_jit
def _full(assembler, p, shift, b, solver, policy):
    return solve_linear(
        lambda v: jacobian_matvec(assembler, p, v) + shift * v,
        b,
        solver=solver,
        preconditioner=_matvec(policy),
        throw=False,
    )


@eqx.filter_jit
def _identity(assembler, p, shift, b, solver):
    return solve_linear(
        lambda v: jacobian_matvec(assembler, p, v) + shift * v, b, solver=solver, throw=False
    )


@eqx.filter_jit
def _stand_in(diagonal, b, solver):
    return solve_linear(lambda v: diagonal * v, b, solver=solver, throw=False)


@eqx.filter_jit
def _apply(diagonal, b, solver, policy):
    return solve_linear(
        lambda v: diagonal * v, b, solver=solver, preconditioner=_matvec(policy), throw=False
    )


@eqx.filter_jit
def _operator_alone(assembler, p, shift, policy, v):
    z = _matvec(policy)(v)
    return jacobian_matvec(assembler, p, z) + shift * z


_apply_alone = eqx.filter_jit(lambda policy, v: _matvec(policy)(v))


class _Counter:
    """The ``on_solve`` observer: the operator applications of the last solve."""

    applications = 0

    def __call__(self, applications, cycles):
        self.applications = int(applications)


def main():
    replay = MarchReplay()
    print(replay.describe(), flush=True)
    print(
        f"[configuration] aquaflux {Path(aquaflux.__file__).parents[1]}, {os.cpu_count()} cpus, "
        f"best of {REPEATS}, budget {CYCLES} cycles; "
        + ", ".join(
            f"{k}={os.environ.get(k, 'unset')}"
            for k in (
                "MALLOC_MMAP_THRESHOLD_",
                "MALLOC_TRIM_THRESHOLD_",
                "MALLOC_ARENA_MAX",
                "LD_PRELOAD",
                "XLA_FLAGS",
            )
        ),
        flush=True,
    )
    march = replay.step.linear_solver()
    restart, max_restarts = march.restart, march.max_steps
    print(f"[configuration] restart {restart}, max_restarts {max_restarts}", flush=True)
    last = max(replay.states)
    wanted = os.environ.get("PITZ_COST_STEPS")
    steps = {int(s) for s in wanted.split(",")} if wanted else {last - 2, last - 1, last}
    counter = _Counter()
    budget = restart * CYCLES
    print(
        "| step | rtol | march apps | march ms/app (faults/solve) | full | identity | stand-in | apply "
        "| ms/iteration: bookkeeping | JVP | M (full - identity) | M (apply - stand-in) "
        "| ms/call alone (mean of a budget): (J + s) M v | M v | J v |",
        flush=True,
    )
    for system in replay.systems(last=max(steps)):
        if system.step not in steps or system.inner != 0:
            continue
        policy = replay.policy
        args = (system.assembler, system.p, system.shift, system.b)
        solver = residual_stop_gmres(
            system.rtol,
            norm=system.measure,
            restart=restart,
            max_restarts=max_restarts,
            on_solve=counter,
        )
        march_ms, march_faults = _best(_full, *args, solver, policy)
        applications = counter.applications
        # The march's own solver must reproduce the recorded count, or this is some other sequence.
        _, raw = _full(*args, solver, policy)
        cycles_run = int(raw) - 2
        fixed = residual_stop_gmres(
            UNREACHABLE, norm=system.measure, restart=restart, max_restarts=CYCLES, on_solve=counter
        )
        # A diagonal of the operator's own scale, never zero: the pseudo-time shift plus one.
        diagonal = jnp.abs(system.shift) + 1.0
        arms = {
            "full": _best(_full, *args, fixed, policy),
            "identity": _best(_identity, *args, fixed),
            "stand-in": _best(_stand_in, diagonal, system.b, fixed),
            "apply": _best(_apply, diagonal, system.b, fixed, policy),
        }
        alone = {
            "operator": _best(_operator_alone, *args[:3], policy, system.b, calls=budget),
            "apply": _best(_apply_alone, policy, system.b, calls=budget),
            "jvp": _best(jacobian_matvec, *args[:2], system.b, calls=budget),
        }
        if counter.applications != budget:
            raise RuntimeError(f"a budgeted arm ran {counter.applications} != {budget} iterations")
        per = {k: v[0] / budget for k, v in arms.items()}
        cells = " | ".join(f"{v[0]:.0f} ({v[1]})" for v in arms.values())
        print(
            f"| {system.step} | {system.rtol:.3g} | {applications} (cycles {cycles_run}, recorded "
            f"{system.recorded}) | {march_ms / applications:.1f} ({march_faults}) | {cells} | "
            f"{per['stand-in']:.2f} | {per['identity'] - per['stand-in']:.2f} | "
            f"{per['full'] - per['identity']:.2f} | {per['apply'] - per['stand-in']:.2f} | "
            + " | ".join(f"{v[0] / budget:.1f} ({v[1] / budget:.0f})" for v in alone.values())
            + " |",
            flush=True,
        )


if __name__ == "__main__":
    main()
