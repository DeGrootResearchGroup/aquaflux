"""Would a Schubert (sparse Broyden) update of the materialized Jacobian replace a fresh probe?

Between refits the march applies a preconditioner fitted to a Jacobian probed some steps back. The march
also holds, for free, each outer step's move ``s = x_{k+1} - x_k`` and the residual change
``y = R(x_{k+1}) - R(x_k)``. Schubert's update (Schubert, Math. Comp. 24, 1970; Dennis & Schnabel,
*Numerical Methods for Unconstrained Optimization and Nonlinear Equations*, 1983, Sec. 11.3) changes each
row ``i`` of ``J`` within a fixed sparsity pattern so that the row satisfies the secant condition:

    J_i <- J_i + ((y_i - J_i s) / (s_i^T D s_i)) (D s_i)^T,

with ``s_i`` the restriction of ``s`` to row ``i``'s columns and ``D`` a diagonal column weight (``D = I``
is classical Schubert). It costs one pass over the stored entries and no Jacobian-vector product, and it
leaves the stored pattern exactly as it was, so the frozen coarsening's refactor still applies. Whether
an updated ``J`` preconditions like a freshly probed one is what this probe measures, on the march's own
systems.

**The replay.** :class:`replay.MarchReplay` supplies the capture (step states and inner iterates), the
case, the session and the step's shift policy. This probe builds its own loop over the TARGET station's
steps, because each arm re-fits the inverse in place and the replay's generator would carry that into the
next system.

**The arms**, for every inner system of target step ``k`` (one-based; step ``k`` starts at state
``x_{k-1}``), each re-fitting the march's own field split at the step's own shift ``beta_k d(x_{k-1})``,
so only the Jacobian differs between arms:

``marched``      the Jacobian the march held at step ``k`` (its last re-fit state); its restart cycles must
                 equal the recorded ones.
``fresh``        probed at ``x_{k-1}``: the ceiling.
``stale-m``      probed at ``x_{k-1-m}``.
``schubert-m``   probed at ``x_{k-1-m}``, then updated with the march's own ``m`` outer moves in order.
``...-full``     the update over every stored entry rather than only those nonzero at the probe.
``...-scaled``   ``D`` the inverse mean square of each field's own step, so every field's columns count
                 alike whatever its units.
``...-resweep``  after the ``m`` updates, the last two re-applied once more, so both secant conditions
                 hold more nearly at once.

Only steps inside the target station are chained, since a station change alters the viscosity and so
the Jacobian in every direction, which a secant along one move cannot carry.

**Reported per arm:** restart cycles and applications of ``A M`` (the march's own residual-stop GMRES in
the step's row-scaled measure, restart 15, at the tolerance the march's adaptive rule solved that step
to -- :meth:`replay.MarchReplay.tolerance` -- counted by its ``on_solve`` hook), the true
relative residual in that measure, and the relative Frobenius distance of the arm's Jacobian from the
fresh one per (row field, column field) block.

Usage
-----
    PITZ_CHECKPOINT_KEEP=500 PITZ_INNER_DUMP_ABOVE=1 \\
        validation/run_case.sh validation/pitzdaily_openfoam/compare.py
    validation/run_case.sh validation/pitzdaily_openfoam/secant_update_probe.py

``PITZ_SECANT_DEPTH`` caps ``m`` (default 4); ``PITZ_SECANT_ARMS`` restricts the arms by name prefix
(comma-separated). ``PITZ_SECANT_TIMING=0`` skips the timing of a probe, a refactor and an update.
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

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    PROBE_BATCH_SIZE,
    batched_jacobian_matvec,
    frozen_shift_diagonal,
    jacobian_matvec,
)
from aquaflux.precondition.monolithic_policy import with_traced_split  # noqa: E402
from aquaflux.precondition.sparse_jacobian import shifted_jacobian  # noqa: E402
from aquaflux.precondition.traced_field_split import traced_field_split  # noqa: E402
from aquaflux.solve import residual_stop_gmres, restart_cycles  # noqa: E402
from aquaflux.turbulence.coupled import coupled_scaled_norm  # noqa: E402
from replay import MarchReplay  # noqa: E402

DEPTH = int(os.environ.get("PITZ_SECANT_DEPTH", "4"))
ARMS = [a for a in os.environ.get("PITZ_SECANT_ARMS", "").split(",") if a]
TIMING = os.environ.get("PITZ_SECANT_TIMING", "1") != "0"
FIELDS = ("u", "v", "p", "k", "w")


def schubert(jacobian, s, y, *, live=None, weight=None):
    """Update ``jacobian`` (CSR) in place so each row meets ``J_i s = y_i`` on its pattern.

    ``live`` (one boolean per stored entry) restricts the update to those entries; ``weight`` (one per
    column) is the diagonal ``D``. Rows whose restricted step is zero are left unchanged.
    """
    rows = np.repeat(np.arange(jacobian.shape[0]), np.diff(jacobian.indptr))
    step = s[jacobian.indices]
    if live is not None:
        step = np.where(live, step, 0.0)
    weighted = step if weight is None else step * weight[jacobian.indices]
    denominator = np.bincount(rows, weights=step * weighted, minlength=jacobian.shape[0])
    defect = y - jacobian @ s
    scale = np.divide(defect, denominator, out=np.zeros_like(defect), where=denominator > 0.0)
    jacobian.data += scale[rows] * weighted


def block_distances(jacobian, reference, n_cells):
    """Relative Frobenius distance of ``jacobian`` from ``reference`` per (row field, column field)."""
    if not (
        np.array_equal(jacobian.indptr, reference.indptr)
        and np.array_equal(jacobian.indices, reference.indices)
    ):
        raise ValueError("the two Jacobians do not share a stored pattern")
    rows = np.repeat(np.arange(reference.shape[0]), np.diff(reference.indptr)) // n_cells
    columns = reference.indices // n_cells
    n_fields = reference.shape[0] // n_cells
    block = rows * n_fields + columns
    size = n_fields * n_fields
    difference = np.bincount(block, (jacobian.data - reference.data) ** 2, minlength=size)
    magnitude = np.bincount(block, reference.data**2, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.sqrt(difference / magnitude)
    return relative.reshape(n_fields, n_fields), np.sqrt(magnitude).reshape(n_fields, n_fields)


class Probe:
    """The march's own probe, inverse and step, re-fitted to any Jacobian the arms hand it."""

    def __init__(self, replay: MarchReplay):
        self.replay = replay
        self.session = replay.session
        self.hook = self.session._refresh_hook()
        self.pc = self.session.preconditioner
        self.precision = self.session._precision
        self.probed = self.hook._probe.narrow(replay.coupled)
        self._residual = eqx.filter_jit(lambda assembler, x: assembler.residual(x))
        self._jacobians = {}

    def jacobian(self, state_index):
        """The probed (unshifted) Jacobian at ``x_{state_index}``, cached."""
        if state_index not in self._jacobians:
            x = jnp.asarray(self.replay.states[state_index]["state"])
            probe, probed = self.hook._probe, self.probed
            self._jacobians[state_index] = self.pc._materialize_jacobian(
                lambda v: jacobian_matvec(probed, x, v),
                probe.plan,
                lambda seeds: batched_jacobian_matvec(probed, x, seeds),
                PROBE_BATCH_SIZE,
                probe.structure,
            )
        return self._jacobians[state_index]

    def residual(self, state_index):
        x = jnp.asarray(self.replay.states[state_index]["state"])
        return np.asarray(self._residual(self.probed, x))

    def state(self, state_index):
        return np.asarray(self.replay.states[state_index]["state"])

    def refit(self, jacobian, shift):
        """Re-fit the split to ``jacobian + shift`` and hand the step the traced copy, as the march does."""
        self.pc.inverse.refactor_block(shifted_jacobian(jacobian, shift))
        split = traced_field_split(self.pc.inverse, self.precision)
        self.replay.step = with_traced_split(self.replay.step, split)


def target_steps(replay):
    """The target station's steps, each with its inner systems' data and the march's re-fit state."""
    first, last = replay.stations + 1, max(replay.states)
    refit_state, due, steps = None, False, []
    for k in range(first, last + 1):
        if k == first or due:
            refit_state = k - 1
        due = False
        records = []
        i = 0
        while (k, i) in replay.inner:
            records.append(replay.inner[(k, i)])
            if (
                replay.refresh_on is not None
                and int(replay.inner[(k, i)]["cycles"]) >= replay.refresh_on
            ):
                due = True
            i += 1
        steps.append((k, refit_state, records))
    return steps


def main():
    replay = MarchReplay()
    print(replay.describe(), flush=True)
    probe = Probe(replay)
    coupled = replay.coupled
    n_cells = coupled.momentum.mesh.n_cells
    policy = replay.policy
    counts = []
    solver_on_solve = lambda applications, cycles: counts.append(int(applications))  # noqa: E731
    linear = replay.solver.linear_solve
    print(
        f"[configuration] target station steps {replay.stations + 1}..{max(replay.states)}; "
        f"refit beta floor {replay.solver.preconditioner.refit_beta_floor}; split precision "
        f"{probe.precision}; depth {DEPTH}; arms {ARMS or 'all'}",
        flush=True,
    )

    if TIMING:
        index = max(replay.states)
        jax.block_until_ready(probe.jacobian(index).data)  # compile
        probe._jacobians.clear()
        started = time.perf_counter()
        jacobian = probe.jacobian(index)
        t_probe = time.perf_counter() - started
        shift = frozen_shift_diagonal(
            policy.base, float(replay.states[index]["shift"]), probe.state(index)
        )
        probe.refit(jacobian, shift)
        started = time.perf_counter()
        probe.refit(jacobian, shift)
        t_refit = time.perf_counter() - started
        r0 = probe.residual(index - 1)
        started = time.perf_counter()
        r1 = probe.residual(index)
        t_residual = time.perf_counter() - started
        copy = jacobian.copy()
        started = time.perf_counter()
        schubert(
            copy, probe.state(index) - probe.state(index - 1), r1 - r0, live=jacobian.data != 0.0
        )
        t_update = time.perf_counter() - started
        print(
            f"[timing] probe {t_probe:.2f} s, refactor+traced split {t_refit:.2f} s, residual "
            f"{t_residual * 1e3:.1f} ms, Schubert update {t_update * 1e3:.0f} ms "
            f"({jacobian.nnz} stored entries, {np.count_nonzero(jacobian.data)} nonzero)",
            flush=True,
        )

    steps = target_steps(replay)
    first = replay.stations + 1
    totals = {}
    for k, refit_state, records in steps:
        phi_n = jnp.asarray(replay.states[k - 1]["state"])
        beta = float(replay.states[k]["shift"])
        assembler = replay.assembler_for(k)
        r_n = assembler.residual(phi_n)
        shift_term = jax.lax.stop_gradient(policy.shift_term(phi_n, r_n).shift(beta))
        measure = coupled_scaled_norm(coupled, policy, phi_n)
        rtol = replay.tolerance(k)
        solver = residual_stop_gmres(
            rtol,
            norm=measure,
            restart=linear.restart,
            max_restarts=linear.max_restarts,
            on_solve=solver_on_solve,
        )
        refit_shift = frozen_shift_diagonal(policy.base, beta, np.asarray(phi_n))
        fresh = probe.jacobian(k - 1)

        arms = {"marched": None, "fresh": fresh}
        for m in range(1, DEPTH + 1):
            origin = k - 1 - m
            if origin < first - 1:
                break
            base = probe.jacobian(origin)
            live = base.data != 0.0
            arms[f"stale-{m}"] = base
            moves = [(j - 1, j) for j in range(origin + 1, k)]
            secants = [
                (probe.state(b) - probe.state(a), probe.residual(b) - probe.residual(a))
                for a, b in moves
            ]
            variants = {"": dict(live=live)}
            if m <= 2:
                variants["-full"] = {}
                variants["-scaled"] = dict(live=live, scaled=True)
            if m >= 2:
                variants["-resweep"] = dict(live=live, resweep=True)
            for suffix, options in variants.items():
                updated = base.copy()
                updated.data = base.data.copy()
                sequence = secants + (secants[-2:] if options.get("resweep") else [])
                for s, y in sequence:
                    weight = None
                    if options.get("scaled"):
                        per_field = np.mean(s.reshape(-1, n_cells) ** 2, axis=1)
                        weight = np.repeat(1.0 / np.where(per_field > 0, per_field, 1.0), n_cells)
                    schubert(updated, s, y, live=options.get("live"), weight=weight)
                arms[f"schubert-{m}{suffix}"] = updated
        if ARMS:
            arms = {
                name: j
                for name, j in arms.items()
                if name == "marched" or any(name.startswith(a) for a in ARMS)
            }

        print(
            f"\n== step {k}: beta {beta:.4g}, rtol {rtol:.3g}, inverse marched from x_{refit_state}, "
            f"{len(records)} inner solves (recorded cycles {[int(r['cycles']) for r in records]})",
            flush=True,
        )
        for name, jacobian in arms.items():
            if name == "marched":
                marched_shift = frozen_shift_diagonal(
                    policy.base,
                    float(replay.states[refit_state + 1]["shift"]),
                    probe.state(refit_state),
                )
                probe.refit(probe.jacobian(refit_state), marched_shift)
                jacobian = probe.jacobian(refit_state)
            else:
                probe.refit(jacobian, refit_shift)
            distance, magnitude = block_distances(jacobian, fresh, n_cells)
            cycles, applications, residuals = [], [], []
            p = phi_n
            for record in records:
                b = -(assembler.residual(p) + shift_term * (p - phi_n))
                counts.clear()
                x, raw = replay._solve(assembler, p, shift_term, b, solver, replay._carrier())
                jax.block_until_ready(x)
                cycles.append(restart_cycles(int(raw)))
                applications.append(counts[-1] if counts else -1)
                true = b - (jacobian_matvec(assembler, p, x) + shift_term * x)
                residuals.append(float(measure(true) / measure(b)))
                p = jnp.asarray(record["state"])
            mark = ""
            if name == "marched":
                recorded = [int(r["cycles"]) for r in records]
                mark = "" if cycles == recorded else f"  ⚠️ recorded {recorded}"
            weight = magnitude / magnitude.max()
            worst = np.nanmax(np.where(weight > 1e-6, distance, np.nan))
            weighted = float(np.sqrt(np.nansum((distance * magnitude) ** 2) / np.sum(magnitude**2)))
            totals.setdefault(name, [0, 0, 0])
            totals[name][0] += sum(cycles)
            totals[name][1] += sum(applications)
            totals[name][2] += 1
            print(
                f"  {name:20s} cycles {sum(cycles):3d} {cycles}  applies {sum(applications):4d}  "
                f"max true rel {max(residuals):.3f}  |dJ|/|J| whole {weighted:.3e} worst block {worst:.3e}"
                f"{mark}",
                flush=True,
            )
            if name in ("stale-1", "schubert-1", "schubert-1-scaled"):
                rows = "  ".join(
                    f"{FIELDS[a]}:" + ",".join(f"{distance[a, b]:.1e}" for b in range(len(FIELDS)))
                    for a in range(len(FIELDS))
                )
                print(f"      per block (row: u,v,p,k,w columns) {rows}", flush=True)

    print("\n== totals over the steps each arm ran (cycles, applications, steps)", flush=True)
    for name, (c, a, n) in totals.items():
        print(f"  {name:20s} {c:4d} {a:5d} {n:3d}", flush=True)


if __name__ == "__main__":
    main()
