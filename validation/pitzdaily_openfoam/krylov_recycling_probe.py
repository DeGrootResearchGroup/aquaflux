"""Would Krylov subspace recycling (GCRO-DR) cut the march's linear work? Replayed from its own systems.

Consecutive inner Newton systems of a dual-time march share an operator up to a small change of state,
and at a low shift a few slowly-converging modes are what keep each GMRES solve going. GCRO-DR (Parks,
de Sturler, Mackey, Johnson & Maiti, SISC 28(5), 2006) keeps the harmonic Ritz vectors of those modes
from each solve and deflates them from the next. Whether it would pay here is boundable without
touching the library: rebuild the exact sequence of linear systems the march solved and run each
Krylov variant on the same sequence, counting applications of the preconditioned operator ``A M``.

**The replay.** ``compare.py`` run with ``PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1`` keeps the
state every step starts from and every inner iterate. Inner solve ``i`` of step ``k`` linearizes at
``p_i`` (``p_0`` the step's start ``phi_n``, ``p_{i+1}`` the dumped iterate of inner ``i``) and solves

    (J(p_i) + s) delta = -(R(p_i) + s (p_i - phi_n)),     s = beta_k * row_relaxation * d(phi_n),

with ``beta_k`` the step's recorded shift and ``d`` the step's own shift policy. That policy is built
once, at the hybrid start of the anchor station, exactly as the ramp builds it, and the preconditioner
is refitted where the march refitted it: in full at a station change, and at the iterate a solve reached
``refresh_on_cycles`` restart cycles on, once per step, at ``max(beta, refit_beta_floor)``.

**The arms**, each on the same systems, counted in applications of ``A M``:

``lineax``   the march's own solver (``relative_residual_gmres(0.3, restart=15)`` in the row-scaled
             measure). Its corrected cycle count must equal the one the march recorded for every solve;
             the run refuses to report anything else if it does not, since a replay that does not
             reproduce the march is measuring some other sequence.
``gmres``    restarted GMRES(15) stopping the moment the true residual meets the same relative
             tolerance in the same measure, checked after every iteration.
``gcro-k``   GCRO-DR with ``k`` recycled vectors carried from solve to solve, the same stop. Charged
             for the ``k`` applications that re-derive ``C = A M U`` for each new operator.

``lineax`` against ``gmres`` is NOT recycling: it is the stopping rule (lineax tests only at a restart
boundary and also demands that the solution moved by less than the tolerance over the last cycle). The
recycling gain is ``gmres`` against ``gcro-k``, which share the stop.

Usage
-----
    PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1 \\
        validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    validation/run_case.sh validation/pitzdaily_openfoam/krylov_recycling_probe.py

⚠️ Any later run of ``compare.py`` writes its own rolling checkpoints into the same directory and evicts
the capture's step states (the inner dumps survive, the states do not); re-run the capture first.

``PITZ_RECYCLE_FROM`` / ``PITZ_RECYCLE_TO`` bound the replayed steps (default: the target station to the
end); ``PITZ_REPLAY_ARMS`` the arms (default ``lineax,gmres-15,gcro-5,gcro-10``).

``weighted-R`` asks whether GMRES should minimize what it is judged by. It cannot minimize the measure
itself, an L1 mean per field block, since GMRES minimizes an inner-product norm; it minimizes the
nearest one, ``||W r||`` with ``W`` the measure's own row and field scales and ``1/sqrt(block size)``,
by running on ``W A M`` and ``W b``. The stop is still the measure, on the true residual.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import compare  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    ConstantRelaxation,
    jacobian_matvec,
    relative_residual_gmres,
    restart_cycles,
    solve_linear,
)
from aquaflux.turbulence import (  # noqa: E402
    coupled_step,
    open_session,
    scale_both_blocks,
    scale_momentum_only,
)
from aquaflux.turbulence.coupled import coupled_scaled_norm  # noqa: E402

CHECKPOINTS = HERE / "checkpoints"
#: The arms to run, by name: ``lineax`` (the march's own solver, and the fidelity check), ``gmres-R``
#: (GMRES(R), residual stop every iteration), ``weighted-R`` (the same, minimizing a weighted 2-norm
#: matched to the measure's row and field scales) and ``gcro-D`` (GCRO-DR keeping D vectors).
ARMS = tuple(os.environ.get("PITZ_REPLAY_ARMS", "lineax,gmres-15,gcro-5,gcro-10").split(","))
RESTART = 15
#: A solve that has not converged after this many applications is reported as such, not run on.
MAX_APPLICATIONS = 400
_COMPANIONS = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}


# --------------------------------------------------------------------------------------------------
# GMRES and GCRO-DR, right-preconditioned, in numpy over a jitted operator.
# --------------------------------------------------------------------------------------------------


def _harmonic_ritz(G, WtW, k):
    """A real basis for the ``k`` harmonic Ritz vectors of smallest magnitude, ``G^T G z = t G^T WtW z``."""
    left, right = G.T @ G, G.T @ WtW
    values, vectors = sla.eig(left, right)
    finite = np.isfinite(values)
    order = np.argsort(np.where(finite, np.abs(values), np.inf))
    columns = []
    for index in order:
        if len(columns) >= k or not finite[index]:
            break
        v = vectors[:, index]
        columns.append(v.real)
        if abs(values[index].imag) > 0 and len(columns) < k:
            columns.append(v.imag)
    basis, _ = np.linalg.qr(np.column_stack(columns[:k]))
    return basis


def gcro_dr(apply, b, measure, rtol, restart, depth, recycled):
    """Solve ``(A M) t = b`` to ``measure(r) <= rtol measure(b)``; ``depth = 0`` is plain GMRES.

    Returns ``(t, applications, recycled)``, where ``recycled`` is ``U`` (``A M U`` is re-derived for the
    next operator, which is what the applications charged for it are).
    """
    t = np.zeros_like(b)
    r = b.copy()
    target = rtol * measure(b)
    applications = 0
    U = C = None
    if depth and recycled is not None:
        AU = np.column_stack([apply(u) for u in recycled.T])
        applications += recycled.shape[1]
        Q, R = np.linalg.qr(AU)
        C, U = Q, np.linalg.solve(R.T, recycled.T).T
        t += U @ (C.T @ r)
        r -= C @ (C.T @ r)
        if measure(r) <= target:
            return t, applications, U
    while applications < MAX_APPLICATIONS:
        beta = np.linalg.norm(r)
        V = np.zeros((b.size, restart + 1))
        V[:, 0] = r / beta
        H = np.zeros((restart + 1, restart))
        Bk = None if C is None else np.zeros((C.shape[1], restart))
        converged, steps = False, restart
        for j in range(restart):
            w = apply(V[:, j])
            applications += 1
            if C is not None:
                Bk[:, j] = C.T @ w
                w -= C @ Bk[:, j]
            for _ in range(2):  # classical Gram-Schmidt, twice
                h = V[:, : j + 1].T @ w
                w -= V[:, : j + 1] @ h
                H[: j + 1, j] += h
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 0:
                V[:, j + 1] = w / H[j + 1, j]
            e = np.zeros(j + 2)
            e[0] = beta
            y = np.linalg.lstsq(H[: j + 2, : j + 1], e, rcond=None)[0]
            residual = V[:, : j + 2] @ (e - H[: j + 2, : j + 1] @ y)
            if measure(residual) <= target or H[j + 1, j] == 0 or applications >= MAX_APPLICATIONS:
                converged, steps = measure(residual) <= target, j + 1
                break
        t += V[:, :steps] @ y
        if C is not None:
            t -= U @ (Bk[:, :steps] @ y)
        r = residual
        if depth:
            # The cycle's harmonic Ritz vectors become the recycled space (Parks et al. 2006, Alg. 1).
            if C is None:
                G, W, Wp = H[: steps + 1, :steps], V[:, :steps], V[:, : steps + 1]
            else:
                norms = np.linalg.norm(U, axis=0)
                D = np.diag(1.0 / norms)
                G = np.block(
                    [
                        [D, Bk[:, :steps]],
                        [np.zeros((steps + 1, U.shape[1])), H[: steps + 1, :steps]],
                    ]
                )
                W = np.hstack([U * (1.0 / norms), V[:, :steps]])
                Wp = np.hstack([C, V[:, : steps + 1]])
            if G.shape[1] >= depth:
                P = _harmonic_ritz(G, Wp.T @ W, depth)
                Q, R = np.linalg.qr(G @ P)
                C, U = Wp @ Q, W @ P @ np.linalg.inv(R)
                t += U @ (C.T @ r)
                r -= C @ (C.T @ r)
        if converged:
            break
    return t, applications, U


# --------------------------------------------------------------------------------------------------
# Rebuilding the march's systems.
# --------------------------------------------------------------------------------------------------


def load(directory):
    states = {}
    for path in sorted(directory.glob("state-*.npz")):
        # Keyed by the FILE's index, the number of steps completed: the record's own `step` field
        # counts from zero, so `state-00017` holds step 16 -- the state step 18 (one-based) starts from.
        with np.load(path) as data:
            states[int(path.stem.split("-")[1])] = {k: np.asarray(data[k]) for k in data.files}
    inner = {}
    for path in sorted(directory.glob("inner-*.npz")):
        with np.load(path) as data:
            inner[(int(data["attempt"]), int(data["inner"]))] = {
                k: np.asarray(data[k]) for k in data.files
            }
    return states, inner


def main():
    solver = compare.SOLVER
    ramp = solver.continuation
    stations = ramp.stations * ramp.steps_per_station
    first_step = int(os.environ.get("PITZ_RECYCLE_FROM", stations + 1))
    refresh_on = solver.dual_time.refresh_on_cycles
    states, inner = load(CHECKPOINTS)
    last_step = int(os.environ.get("PITZ_RECYCLE_TO", max(states)))
    print(
        f"[configuration] steps {first_step}..{last_step} of {max(states)}; restart {RESTART}; "
        f"arms {ARMS}; refresh on {refresh_on} cycles; jax {jax.__version__}, "
        f"{jax.default_backend()}",
        flush=True,
    )
    coupled = compare.build_case()["coupled"]
    companion = _COMPANIONS[ramp.scale]

    def assembler_for(step):
        station = min((step - 1) // ramp.steps_per_station, ramp.stations)
        if station == ramp.stations:
            return coupled
        return companion(coupled, ramp.anchor ** (1.0 - station / ramp.stations))

    # The step and its shift policy, built where the ramp builds them: at the anchor's hybrid start.
    anchor = assembler_for(1)
    seed = anchor.state_from_physical(*hybrid_initialize(anchor))
    allowed = inspect.signature(coupled_step).parameters
    march = {k: v for k, v in solver.settings().items() if k in allowed and k != "preconditioner"}
    session = open_session(solver.settings()["preconditioner"], coupled)
    session.rebind(anchor)
    step = session.build(seed, **march)
    policy = step.shift_policy
    apply_pc = policy.preconditioner.matvec()
    rtol = solver.linear_solve.rtol

    @eqx.filter_jit
    def preconditioned(assembler, p, shift, v):
        z = apply_pc(v)
        return jacobian_matvec(assembler, p, z) + shift * z

    @eqx.filter_jit
    def march_solve(assembler, p, shift, b, measure):
        lx_solver = relative_residual_gmres(
            rtol, norm=measure, restart=RESTART, max_restarts=solver.linear_solve.max_restarts
        )
        return solve_linear(
            lambda v: jacobian_matvec(assembler, p, v) + shift * v,
            b,
            solver=lx_solver,
            preconditioner=apply_pc,
            throw=False,
        )[1]

    @eqx.filter_jit
    def measured(measure, r):
        return measure(r)

    def refit(assembler, beta, state):
        session.rebind(assembler)
        session.refresh_preconditioner(
            eqx.tree_at(lambda s: s.relaxation_schedule, step, ConstantRelaxation(beta)), state
        )

    totals = dict.fromkeys(ARMS, 0)
    recycled = {arm: None for arm in ARMS}
    mismatches = 0
    previous = None
    print(
        f"\n{'step':>4} {'in':>2} {'cyc':>3} {'lx':>3} | " + " ".join(f"{a:>11}" for a in ARMS),
        flush=True,
    )
    for k in range(first_step, last_step + 1):
        phi_n = jnp.asarray(states[k - 1]["state"]) if k > 1 else seed
        beta = float(states[k]["shift"])
        assembler = assembler_for(k)
        r_n = assembler.residual(phi_n)
        shift = jax.lax.stop_gradient(policy.shift_term(phi_n, r_n).shift(beta))
        measure = coupled_scaled_norm(coupled, policy, phi_n)
        sizes = np.asarray(measure.sizes)
        weights = 1.0 / (
            np.asarray(measure.row_scale)
            * np.repeat(np.asarray(measure.field_scale) * np.sqrt(sizes), sizes)
        )
        # A station change re-fits in full at the step's start. Starting the replay anywhere but at a
        # station change is NOT faithful: the march's inverse there dates from a mid-step refresh.
        if assembler is not previous:
            refit(assembler, beta, phi_n)
        previous = assembler
        refreshed = False
        i, p = 0, phi_n
        while (k, i) in inner:
            record = inner[(k, i)]
            recorded = int(record["cycles"])
            b = -(assembler.residual(p) + shift * (p - phi_n))

            def apply(v, assembler=assembler, p=p, shift=shift):
                # A copy: an array viewed from JAX is read-only, and the Arnoldi loop works in place.
                return np.array(preconditioned(assembler, p, shift, jnp.asarray(v)))

            def norm(v, measure=measure):
                return float(measured(measure, jnp.asarray(v)))

            bn = np.asarray(b)
            counts, cycles = {}, recorded
            for arm in ARMS:
                kind, _, number = arm.partition("-")
                if kind == "lineax":
                    raw = int(march_solve(assembler, p, shift, b, measure))
                    cycles = restart_cycles(raw)
                    mismatches += cycles != recorded
                    # One application for the start-up residual, then RESTART + 1 per cycle.
                    counts[arm] = 1 + (RESTART + 1) * (raw - 1)
                elif kind == "gmres":
                    counts[arm] = gcro_dr(apply, bn, norm, rtol, int(number), 0, None)[1]
                elif kind == "weighted":
                    counts[arm] = gcro_dr(
                        lambda v, apply=apply, w=weights: w * apply(v),
                        weights * bn,
                        lambda v, norm=norm, w=weights: norm(v / w),
                        rtol,
                        int(number),
                        0,
                        None,
                    )[1]
                elif kind == "gcro":
                    _, counts[arm], recycled[arm] = gcro_dr(
                        apply, bn, norm, rtol, RESTART, int(number), recycled[arm]
                    )
                else:
                    raise SystemExit(f"unknown arm {arm!r}")
                totals[arm] += counts[arm]
            print(
                f"{k:4d} {i:2d} {recorded:3d} {cycles:3d}{'!' if cycles != recorded else ' '}| "
                + " ".join(f"{counts[a]:11d}" for a in ARMS),
                flush=True,
            )
            # The march's mid-step refresh: once per step, at the iterate the expensive solve reached.
            if refresh_on is not None and recorded >= refresh_on and not refreshed:
                refit(assembler, beta, jnp.asarray(record["state"]))
                refreshed = True
            p = jnp.asarray(record["state"])
            i += 1
    print("\ntotals: " + ", ".join(f"{n} {v}" for n, v in totals.items()), flush=True)
    if mismatches:
        print(
            f"⚠️ {mismatches} solves did not reproduce the march's recorded cycle count", flush=True
        )


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[done in {time.perf_counter() - t0:.0f} s]")
