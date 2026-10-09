"""Would Krylov subspace recycling (GCRO-DR) cut the march's linear work? Replayed from its own systems.

Consecutive inner Newton systems of a dual-time march share an operator up to a small change of state,
and at a low shift a few slowly-converging modes are what keep each GMRES solve going. GCRO-DR (Parks,
de Sturler, Mackey, Johnson & Maiti, SISC 28(5), 2006) keeps the harmonic Ritz vectors of those modes
from each solve and deflates them from the next. Whether it would pay here is boundable without
touching the library: rebuild the exact sequence of linear systems the march solved and run each
Krylov variant on the same sequence, counting applications of the preconditioned operator ``A M``.

**The replay** is :class:`replay.MarchReplay`, which rebuilds the march's systems from its own
checkpoints and refits the preconditioner where the march refitted it.

**The arms**, each on the same systems, counted in applications of ``A M``:

``lineax``   ``relative_residual_gmres(0.3, restart=15)`` in the row-scaled measure, the march's own
             solver before pitzDaily moved to the residual-only stop. Its corrected cycle count must equal
             the one the march recorded for every solve, so capture with ``PITZ_FORWARD_STOP=lineax
             PITZ_REFRESH_ON_CYCLES=3`` (the configuration this probe's recorded results were taken in);
             a mismatch is reported, since a replay that does not reproduce the march is measuring some
             other sequence.
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

``PITZ_REPLAY_FROM`` / ``PITZ_REPLAY_TO`` bound the replayed steps (default: the target station to the
end); ``PITZ_REPLAY_ARMS`` the arms (default ``lineax,gmres-15,gcro-5,gcro-10``).

``weighted-R`` asks whether GMRES should minimize what it is judged by. It cannot minimize the measure
itself, an L1 mean per field block, since GMRES minimizes an inner-product norm; it minimizes the
nearest one, ``||W r||`` with ``W`` the measure's own row and field scales and ``1/sqrt(block size)``,
over the same Krylov space as plain GMRES, by running on ``W A M W^-1`` from ``W b`` (left scaling
alone builds a different space). The stop is still the measure, on the true residual.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from aquaflux.solve import relative_residual_gmres, restart_cycles  # noqa: E402
from replay import MarchReplay  # noqa: E402

#: The arms to run, by name: ``lineax`` (the march's own solver, and the fidelity check), ``gmres-R``
#: (GMRES(R), residual stop every iteration), ``weighted-R`` (the same, minimizing a weighted 2-norm
#: matched to the measure's row and field scales) and ``gcro-D`` (GCRO-DR keeping D vectors).
ARMS = tuple(os.environ.get("PITZ_REPLAY_ARMS", "lineax,gmres-15,gcro-5,gcro-10").split(","))
RESTART = 15
#: A solve that has not converged after this many applications is reported as such, not run on.
MAX_APPLICATIONS = 400


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


def main():
    replay = MarchReplay()
    print(f"{replay.describe()}; restart {RESTART}; arms {ARMS}", flush=True)
    rtol = replay.solver.linear_solve.rtol

    @eqx.filter_jit
    def measured(measure, r):
        return measure(r)

    totals = dict.fromkeys(ARMS, 0)
    recycled = {arm: None for arm in ARMS}
    mismatches = 0
    print(
        f"\n{'step':>4} {'in':>2} {'cyc':>3} {'lx':>3} | " + " ".join(f"{a:>11}" for a in ARMS),
        flush=True,
    )
    for system in replay.systems():
        measure = system.measure
        sizes = np.asarray(measure.sizes)
        weights = 1.0 / (
            np.asarray(measure.row_scale)
            * np.repeat(np.asarray(measure.field_scale) * np.sqrt(sizes), sizes)
        )

        def apply(v, system=system):
            # A copy: an array viewed from JAX is read-only, and the Arnoldi loop works in place.
            return np.array(replay.preconditioned(system, jnp.asarray(v)))

        def norm(v, measure=measure):
            return float(measured(measure, jnp.asarray(v)))

        bn = np.asarray(system.b)
        counts, cycles = {}, system.recorded
        for arm in ARMS:
            kind, _, number = arm.partition("-")
            if kind == "lineax":
                lx_solver = relative_residual_gmres(
                    rtol,
                    norm=measure,
                    restart=RESTART,
                    max_restarts=replay.solver.linear_solve.max_restarts,
                )
                raw = int(replay.solve(system, lx_solver)[1])
                cycles = restart_cycles(raw)
                mismatches += cycles != system.recorded
                # One application for the start-up residual, then RESTART + 1 per cycle.
                counts[arm] = 1 + (RESTART + 1) * (raw - 1)
            elif kind == "gmres":
                counts[arm] = gcro_dr(apply, bn, norm, rtol, int(number), 0, None)[1]
            elif kind == "weighted":
                # The same Krylov space as plain GMRES, orthogonalized in the weighted inner product:
                # GMRES on the similar operator W A M W^-1 from W b. Scaling on the left alone
                # (W A M) builds a different space, and on this operator it does not converge.
                counts[arm] = gcro_dr(
                    lambda v, apply=apply, w=weights: w * apply(v / w),
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
            f"{system.step:4d} {system.inner:2d} {system.recorded:3d} "
            f"{cycles:3d}{'!' if cycles != system.recorded else ' '}| "
            + " ".join(f"{counts[a]:11d}" for a in ARMS),
            flush=True,
        )
    print("\ntotals: " + ", ".join(f"{n} {v}" for n, v in totals.items()), flush=True)
    if mismatches:
        print(
            f"⚠️ {mismatches} solves did not reproduce the march's recorded cycle count", flush=True
        )


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[done in {time.perf_counter() - t0:.0f} s]")
