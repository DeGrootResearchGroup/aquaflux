"""What a reach-2 probe colouring does to one step of a step case, and where: bfs3d or pitzDaily.

Both step cases assemble their preconditioner from a Jacobian materialized in a reach-3 pattern. Probing
a column at reach 2 inside that pattern does not drop its distance-3 couplings: a reach-2 colouring lets
two cells three apart share a colour, so a row coupled to both receives their sum, and the
de-compression charges it to the one inside reach 2. pitzDaily probes every column that way with no
change to its march; on bfs3d the same colouring costs steps. This measures, at one checkpoint of a
uniform-reach-3 march of either case, the two things that could separate them:

  folding   ``E = J2 - J3``, the reach-2-coloured Jacobian minus the uniform-reach-3 one, both in the
            reach-3 pattern -- exactly the difference the two arms' preconditioners are fitted from.
            Per (row field, column field) block, ``|E_ab| / |J3_ab|`` (Frobenius); and per row field, the
            share of rows whose DIAGONAL moved, with quantiles of ``|E_ii| / |J3_ii + s_i|`` against the
            shifted diagonal the preconditioner sees (a far coupling folds onto the diagonal when the
            row's own cell shares the far cell's colour).
  solves    the step's first inner solve, rebuilt as the viscosity-ramp march builds it (session bound
            to the anchor companion and built at the anchor's seed, then re-bound to the step's station
            and re-fitted at the checkpoint and the step's shift), once per probe arm: the step's own
            solver at the march's tolerance and again at ``REACH_FOLDING_TIGHT_RTOL``, with cycles, the
            Euclidean and row-scaled linear residuals, the rows the pivot floor moved, and the step's
            line-search ladder ``|G(phi + a d)| / |G(phi)|``. The operator is the exact Jacobian in every
            arm; only the preconditioner differs.

Arms (``REACH_FOLDING_ARMS``, comma-separated, default ``3,2,each``): ``3`` probes every column at reach 3,
``2`` every column at reach 2, and ``each`` adds one arm per field with only that column at reach 2 --
which column's folding costs the solve.

Usage, one checkpoint per run (``state-<step-1>`` is the step's start, ``state-<step>`` gives its shift)::

    validation/run_case.sh validation/reach_folding.py   # with REACH_FOLDING_CASE and REACH_FOLDING_STEP

Environment: ``REACH_FOLDING_CASE`` (``bfs3d_openfoam`` or ``pitzdaily_openfoam``), ``REACH_FOLDING_STEP``
(one-based), ``REACH_FOLDING_CHECKPOINTS`` (default the case's ``checkpoints/``), ``REACH_FOLDING_ARMS``,
``REACH_FOLDING_TIGHT_RTOL`` (default 1e-5), ``REACH_FOLDING_SKIP_MATRIX=1`` to run the solves only.
"""

from __future__ import annotations

import dataclasses
import gc
import inspect
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
CASE_NAME = os.environ.get("REACH_FOLDING_CASE", "bfs3d_openfoam")
CASE = HERE / CASE_NAME
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.precondition import materialize_block_jacobian  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    PROBE_BATCH_SIZE,
    batched_jacobian_matvec,
    jacobian_matvec,
)
from aquaflux.precondition.sparse_jacobian import (  # noqa: E402
    ColumnProbePlan,
    block_stencil_colouring,
    column_probe_plan,
)
from aquaflux.solve import ConstantRelaxation  # noqa: E402
from aquaflux.solve.continuation import _shifted_solve  # noqa: E402
from aquaflux.solve.linear import in_progress_measure, with_relative_tolerance  # noqa: E402
from aquaflux.turbulence import (  # noqa: E402
    coupled_step,
    open_session,
    scale_both_blocks,
    scale_momentum_only,
)
from aquaflux.turbulence.coupled import coupled_scaled_norm  # noqa: E402

LADDER = (1.0, 0.5, 0.25, 0.125, 1 / 32)
#: A diagonal counts as moved when the fold shifts it by more than this share of the shifted diagonal.
MOVED = 1e-6


def field_names(n_fields):
    return ("u", "v", "w", "p", "k", "omega") if n_fields == 6 else ("u", "v", "p", "k", "omega")


def arms(n_fields, names):
    """``label -> column_reach`` (``None`` is a uniform reach-3 probe) for ``REACH_FOLDING_ARMS``."""
    out = {}
    for token in os.environ.get("REACH_FOLDING_ARMS", "3,2,each").split(","):
        if token == "3":
            out["uniform 3"] = None
        elif token == "2":
            out["all at 2"] = (2,) * n_fields
        elif token == "each":
            for f in range(n_fields):
                out[f"{names[f]} at 2"] = tuple(2 if g == f else 3 for g in range(n_fields))
        else:
            raise SystemExit(f"REACH_FOLDING_ARMS: unknown arm {token!r} (3, 2 or each)")
    return out


class StepRebuild:
    """The march's step at a checkpoint, rebuilt per preconditioner spec as the viscosity ramp builds it."""

    def __init__(self, coupled, phi, beta, step_number):
        self.coupled, self.phi, self.beta = coupled, phi, beta
        solver_spec = compare.SOLVER
        self.spec = solver_spec
        ramp = solver_spec.continuation
        companion = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}[
            ramp.scale
        ]

        def assembler_for(station):
            if station == ramp.stations:
                return coupled
            return companion(coupled, ramp.anchor ** (1.0 - station / ramp.stations))

        self.anchor = assembler_for(0)
        self.seed = self.anchor.state_from_physical(*hybrid_initialize(self.anchor))
        self.station = min(step_number - 1, ramp.stations)
        self.assembler = assembler_for(self.station)

    def fitted(self, column_reach):
        """The step's state after the re-fit, under a probe of ``column_reach``; and its session."""
        settings = self.spec.settings()
        pc = settings["preconditioner"]
        pc = dataclasses.replace(pc, probe=dataclasses.replace(pc.probe, column_reach=column_reach))
        allowed = inspect.signature(coupled_step).parameters
        march = {k: v for k, v in settings.items() if k in allowed and k != "preconditioner"}
        session = open_session(pc, self.coupled)
        session.rebind(self.anchor)
        base = session.build(self.seed, **march)
        session.rebind(self.assembler)
        active = eqx.tree_at(
            lambda s: s.relaxation_schedule, base, ConstantRelaxation(jnp.asarray(self.beta))
        )
        update = session.refresh_preconditioner(active, self.phi, due=True)
        if update is not None:
            active = update(active)
        active = active.with_norm(coupled_scaled_norm(self.coupled, active.shift_policy, self.phi))
        return active, session

    def shift_and_preconditioner(self, active):
        norm = active.residual_norm
        g0 = self.assembler.residual(self.phi)
        term = active.shift_policy.shift_term(self.phi, g0)
        strength = active.relaxation_schedule.shift_strength(norm(g0), norm(g0))
        shift = jax.lax.stop_gradient(term.shift(strength))
        return g0, shift, term.make_preconditioner(strength)


def solve_arm(rebuild, label, column_reach, names, tight_rtol):
    started = time.time()
    active, session = rebuild.fitted(column_reach)
    n = rebuild.coupled.layout.n_cells
    floored = getattr(getattr(session, "_hook", None), "floored_rows", None)
    floored_text = (
        "n/a"
        if floored is None
        else f"{floored.size} ("
        + ", ".join(f"{names[f]} {int(np.sum(floored // n == f))}" for f in range(len(names)))
        + ")"
    )
    residual_fn = rebuild.assembler.residual
    phi = rebuild.phi
    g, shift, preconditioner = rebuild.shift_and_preconditioner(active)
    norm = active.residual_norm
    apply_a = jax.jit(lambda v: jax.jvp(residual_fn, (phi,), (v,))[1] + shift * v)
    G = jax.jit(lambda p: residual_fn(p) + shift * (p - phi))
    base_solver = active.linear_solver()
    project = active.step_projection
    print(f"[arm] {label}: fitted in {time.time() - started:.0f} s, floored rows {floored_text}", flush=True)
    for tag, solver in (
        ("march rtol", base_solver),
        (f"rtol {tight_rtol:g}", with_relative_tolerance(base_solver, tight_rtol)),
    ):
        solver = in_progress_measure(solver, norm)
        t0 = time.time()
        d, cycles = _shifted_solve(residual_fn, phi, g, shift, preconditioner, solver)
        linear = apply_a(d) + g
        line = (
            f"  {tag:11s} cycles {int(cycles):3d}  Euclidean "
            f"{float(jnp.linalg.norm(linear) / jnp.linalg.norm(g)):.2e}  row-scaled "
            f"{float(norm(linear) / norm(g)):.3e}  ({time.time() - t0:.0f} s)"
        )
        if tag == "march rtol":
            v = project(phi, d) if project is not None else d
            ladder = [float(norm(G(phi + a * v))) / float(norm(g)) for a in LADDER]
            line += "  ladder " + " ".join(f"{r:.3f}" for r in ladder)
        print(line, flush=True)
    shift = np.asarray(shift)
    del active, session, preconditioner, apply_a, G
    gc.collect()
    return shift


def materialize(coupled, state, owner, nb, n, n_fields, column_reach):
    if column_reach is None:
        plan = ColumnProbePlan.uniform(block_stencil_colouring(owner, nb, n, 3), n_fields)
    else:
        plan = column_probe_plan(owner, nb, n, column_reach, pattern_reach=3)
    batched = lambda seeds: batched_jacobian_matvec(coupled, state, seeds)  # noqa: E731
    return materialize_block_jacobian(
        lambda v: jacobian_matvec(coupled, state, v),
        plan,
        batched_matvec=batched,
        probe_batch_size=PROBE_BATCH_SIZE,
    ).tocsr()


def folding_report(coupled, phi, shift, names):
    mesh = coupled.momentum.mesh
    n, n_fields = coupled.layout.n_cells, coupled.layout.n_fields
    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)
    for reach in (2, 3):
        colours = block_stencil_colouring(owner, nb, n, reach).n_colours
        print(f"[graph] reach {reach}: {colours} colours", flush=True)
    t0 = time.time()
    j3 = materialize(coupled, phi, owner, nb, n, n_fields, None)
    j2 = materialize(coupled, phi, owner, nb, n, n_fields, (2,) * n_fields)
    print(f"[matrix] {j3.shape[0]} dofs, {j3.nnz / 1e6:.1f}M nnz, both in {time.time() - t0:.0f} s", flush=True)
    e = (j2 - j3).tocsr()
    print("\n  |E_ab| / |J3_ab| (row field x column field; E = reach-2 colouring - uniform reach 3)", flush=True)
    print("        " + "".join(f"{f:>10s}" for f in names), flush=True)
    for a in range(n_fields):
        cells = []
        for b in range(n_fields):
            sub3 = j3[a * n : (a + 1) * n, b * n : (b + 1) * n]
            sube = e[a * n : (a + 1) * n, b * n : (b + 1) * n]
            total = float(np.linalg.norm(sub3.data))
            cells.append(float(np.linalg.norm(sube.data)) / total if total else 0.0)
        print(f"  {names[a]:>5s} " + "".join(f"{c:10.2e}" for c in cells), flush=True)
    diagonal3 = j3.diagonal()
    shifted = np.abs(diagonal3 + shift)
    moved = np.abs(e.diagonal()) / np.where(shifted > 0, shifted, np.inf)
    print(
        "\n  diagonal fold |E_ii| / |J3_ii + s_i| per row field: share of rows moved by more than "
        f"{MOVED:g}, then quantiles over all rows",
        flush=True,
    )
    print(f"  {'field':>5s} {'moved':>8s} {'median':>10s} {'p99':>10s} {'max':>10s} {'rows > 1e-2':>12s}", flush=True)
    for f in range(n_fields):
        m = moved[f * n : (f + 1) * n]
        print(
            f"  {names[f]:>5s} {np.mean(m > MOVED):8.3f} {np.median(m):10.2e} {np.quantile(m, 0.99):10.2e} "
            f"{m.max():10.2e} {int(np.sum(m > 1e-2)):12d}",
            flush=True,
        )
    # Off-diagonal entries: how far each fold moves an entry relative to the row's shifted diagonal,
    # which is the scale a smoother or a per-cell block inverse weighs it against.
    rows = np.repeat(np.arange(e.shape[0]), np.diff(e.indptr))
    weight = np.abs(e.data) / np.where(shifted[rows] > 0, shifted[rows], np.inf)
    print("\n  off-diagonal fold |E_ij| / |J3_ii + s_i|, largest per row field (over every column)", flush=True)
    for f in range(n_fields):
        in_field = (rows >= f * n) & (rows < (f + 1) * n)
        print(f"  {names[f]:>5s} max {weight[in_field].max() if in_field.any() else 0.0:10.2e}", flush=True)
    del j2, j3, e
    gc.collect()


def main():
    step_number = int(os.environ["REACH_FOLDING_STEP"])
    checkpoints = Path(os.environ.get("REACH_FOLDING_CHECKPOINTS", CASE / "checkpoints"))
    tight_rtol = float(os.environ.get("REACH_FOLDING_TIGHT_RTOL", "1e-5"))
    with np.load(checkpoints / f"state-{step_number - 1:05d}.npz") as data:
        phi = jnp.asarray(data["state"])
        start = {k: data[k] for k in ("step", "residual_norm", "shift") if k in data.files}
    with np.load(checkpoints / f"state-{step_number:05d}.npz") as data:
        beta = float(data["shift"])
    coupled = compare.build_case()["coupled"]
    n, n_fields = coupled.layout.n_cells, coupled.layout.n_fields
    names = field_names(n_fields)
    rebuild = StepRebuild(coupled, phi, beta, step_number)
    print(
        f"[configuration] case {CASE_NAME}, {n} cells, step {step_number} (station {rebuild.station}) from "
        f"state-{step_number - 1:05d} {start}, beta {beta:g}, file probe "
        f"{compare.SOLVER.preconditioner.probe}, jax {jax.__version__}",
        flush=True,
    )
    shift = None
    for label, column_reach in arms(n_fields, names).items():
        arm_shift = solve_arm(rebuild, label, column_reach, names, tight_rtol)
        shift = arm_shift if shift is None else shift
    if os.environ.get("REACH_FOLDING_SKIP_MATRIX") != "1":
        folding_report(coupled, phi, shift, names)


if __name__ == "__main__":
    main()
