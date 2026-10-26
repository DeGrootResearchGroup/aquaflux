"""What a reach-2 probe colouring does to one step of a step case, and where: bfs3d or pitzDaily.

Both step cases assemble their preconditioner from a Jacobian materialized in a reach-3 pattern. Probing
a column at reach 2 inside that pattern does two things to its distance-3 couplings. It drops them: their
positions are written as zeros. And it folds them: a reach-2 colouring lets two cells three apart share a
colour, so a row coupled to both receives their sum, and the de-compression charges it to the one inside
reach 2. pitzDaily probes every column that way with no
change to its march; on bfs3d the same colouring costs steps. This measures, at one checkpoint of a
uniform-reach-3 march of either case, the two things that could separate them:

  folding   ``E = J2 - J3``, the reach-2-coloured Jacobian minus the uniform-reach-3 one, both in the
            reach-3 pattern -- exactly the difference the two arms' preconditioners are fitted from.
            Per (row field, column field) block, ``|E_ab| / |J3_ab|`` (Frobenius); and per row field, the
            share of rows whose DIAGONAL moved, with quantiles of ``|E_ii| / |J3_ii + s_i|`` against the
            shifted diagonal the preconditioner sees. (Expect none: a reach-2 colouring keeps two cells
            of one colour more than four apart, so a distance-3 coupling can land on another
            distance-2 entry of its row but never on the row's own cell.)
  solves    one inner solve of the step (the first, or ``REACH_FOLDING_INNER``), rebuilt as the
            viscosity-ramp march builds it (session bound to the anchor companion and built at the
            anchor's seed, then re-bound to the step's station and re-fitted at the checkpoint and the
            step's shift), once per probe arm: the step's own
            solver at the march's tolerance and again at ``REACH_FOLDING_TIGHT_RTOL``, with cycles, the
            Euclidean and row-scaled linear residuals (the latter also per row field), the rows the pivot
            floor moved, and the step's line-search ladder ``|G(phi + a d)| / |G(phi)|``. The operator
            is the exact Jacobian in every arm; only the preconditioner differs.
  Schur     what the fold does to the pressure Schur complement ``S = C - D M G`` a SIMPLE-smoothed flow
            block relaxes, from the shifted ``[[F, G], [D, C]]`` of each matrix: the norms of ``C``,
            ``D M G`` and ``S`` (how far the two terms cancel) and the relative change of each under the
            fold, overall and on ``S``'s diagonal. ``M`` here is the diagonal ``1 / F_ii`` -- a proxy
            for the per-cell block inverse the shipped smoother builds, so read it for the cancellation
            and the amplification, not as the smoother's own ``S``.

Arms (``REACH_FOLDING_ARMS``, comma-separated, default ``3,2,each``): ``3`` probes every column at reach 3,
``2`` every column at reach 2, ``each`` adds one arm per field with only that column at reach 2 --
which column's folding costs the solve -- and a field name (``p``) adds that one such arm. ``A/B`` builds
the session under probe ``A`` and re-fits it under ``B`` (``3/p``): the leading hierarchy's coarsening is
frozen at the build from the values it read then, so this separates a fold that moves the COARSENING from
one that moves the fitted VALUES. ``X*t`` builds under a uniform reach 3 and re-fits from
``J3 + t (JX - J3)``, the fold of probe ``X`` scaled by ``t`` (``p*0.5``, ``2*4``): how close a case sits to
the fold that breaks its solve. ``X*0`` must reproduce ``3`` and ``X*1`` must reproduce ``X``. A short probe
does two things to a column with couplings past its reach: it DROPS them (their positions are written as
zeros) and it FOLDS them onto near entries it shares a colour with. ``X@drop`` re-fits from ``J3`` with only
the drop, ``X@fold`` from ``JX`` with the dropped entries restored -- which of the two costs the solve.

Usage, one checkpoint per run (``state-<step-1>`` is the step's start, ``state-<step>`` gives its shift)::

    validation/run_case.sh validation/reach_folding.py   # with REACH_FOLDING_CASE and REACH_FOLDING_STEP

⚠️ **Run at most two arms per process on bfs3d.** Each arm opens its own session and compiles its own
programs, and they are not all released: on a 15 GiB machine the fifth arm died in LLVM's allocator
(2026-10-25). Run the ``3`` arm with the matrix report, then each other arm alone with
``REACH_FOLDING_SKIP_MATRIX=1``.

A step's first solve is usually its easy one; the hard solves come later in its inner loop. With
``REACH_FOLDING_INNER=i`` the harness replays inner solve ``i`` instead: it linearizes at the iterate the
march dumped after inner ``i - 1`` (run the case with ``BFS3D_INNER_DUMP_ABOVE=1`` /
``PITZ_INNER_DUMP_ABOVE=1``, which writes ``inner-<attempt>-<inner>.npz`` beside the checkpoints) and
solves ``(J(p_i) + s) d = -(R(p_i) + s (p_i - phi_n))`` on the preconditioner fitted at the step's start
``phi_n``. That is the march's own system wherever the step re-fitted at its start, which every
viscosity-ramp step does (each changes station); ``REACH_FOLDING_ATTEMPT`` names the dump's attempt
(default: the step number, right for a march with no redone step).

Environment: ``REACH_FOLDING_CASE`` (``bfs3d_openfoam`` or ``pitzdaily_openfoam``), ``REACH_FOLDING_STEP``
(one-based), ``REACH_FOLDING_CHECKPOINTS`` (default the case's ``checkpoints/``), ``REACH_FOLDING_ARMS``,
``REACH_FOLDING_INNER`` (default 0), ``REACH_FOLDING_ATTEMPT``, ``REACH_FOLDING_TIGHT_RTOL`` (default 1e-5), ``REACH_FOLDING_SKIP_MATRIX=1`` to run the solves only.
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
import scipy.sparse as sp  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.precondition import FieldSplit, materialize_block_jacobian  # noqa: E402
from aquaflux.precondition import sparse_jacobian as sparse_jacobian_module  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    PROBE_BATCH_SIZE,
    batched_jacobian_matvec,
    jacobian_matvec,
)
from aquaflux.precondition.probe_layout import ProbedBlocks  # noqa: E402
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
from mesh_anisotropy import conductance_ratio  # noqa: E402

LADDER = (1.0, 0.5, 0.25, 0.125, 1 / 32)
#: A diagonal counts as moved when the fold shifts it by more than this share of the shifted diagonal.
MOVED = 1e-6


def field_names(n_fields):
    return ("u", "v", "w", "p", "k", "omega") if n_fields == 6 else ("u", "v", "p", "k", "omega")


def probe_reach(token, n_fields, names):
    """``(label, column_reach)`` for one probe token (``None`` is a uniform reach-3 probe)."""
    if token == "3":
        return "uniform 3", None
    if token == "2":
        return "all at 2", (2,) * n_fields
    if token in names:
        f = names.index(token)
        return f"{token} at 2", tuple(2 if g == f else 3 for g in range(n_fields))
    raise SystemExit(f"REACH_FOLDING_ARMS: unknown probe {token!r} (3, 2, or a field name)")


def arms(n_fields, names):
    """``label -> (build column_reach, refit column_reach, mix)`` for ``REACH_FOLDING_ARMS``.

    The mix is ``None`` except for an ``X*t`` or ``X@part`` arm, whose refit probe is ``X``'s, mixed with
    a uniform reach-3 probe as :class:`ScaledFold` describes.
    """
    out = {}
    for token in os.environ.get("REACH_FOLDING_ARMS", "3,2,each").split(","):
        if "*" in token:
            name, scale = token.split("*")
            label, reach = probe_reach(name, n_fields, names)
            out[f"uniform 3 + {float(scale):g} x fold of {label}"] = (None, reach, float(scale))
        elif "@" in token:
            name, part = token.split("@")
            if part not in ("drop", "fold"):
                raise SystemExit(f"REACH_FOLDING_ARMS: {token!r}: the part after @ is drop or fold")
            label, reach = probe_reach(name, n_fields, names)
            out[
                f"uniform 3 with only the {'dropped' if part == 'drop' else 'folded'} part of {label}"
            ] = (None, reach, part)
        elif token == "each":
            for name in names:
                label, reach = probe_reach(name, n_fields, names)
                out[label] = (reach, reach, None)
        elif "/" in token:
            (built, build), (fitted, refit) = (
                probe_reach(t, n_fields, names) for t in token.split("/")
            )
            out[f"coarsened {built}, re-fitted {fitted}"] = (build, refit, None)
        else:
            label, reach = probe_reach(token, n_fields, names)
            out[label] = (reach, reach, None)
    return out


class ScaledFold:
    """While active, every block materialize returns a mix of a uniform reach-3 probe and probe ``X``'s.

    ``mix`` is a number ``t`` for ``J3 + t (JX - J3)``; ``"drop"`` for ``J3`` with only the entries ``X``
    zeroes (its out-of-reach ring) set to zero; ``"fold"`` for ``JX`` with those entries restored from
    ``J3``, which keeps only what ``X`` adds onto its near entries. The dropped entries are read as
    those ``X`` leaves exactly zero where ``J3`` does not.

    The field split materializes its blocks through ``sparse_jacobian.materialize_blocks``, imported at
    the call, so replacing that module attribute reaches the re-fit without touching the library. Both
    probes run on the re-fit's own ``matvec``; each uses its own colouring and gather structure, and
    the two value arrays share the reach-3 pattern's block layout, which is checked.
    """

    def __init__(self, uniform, folded, mix):
        self.uniform, self.folded, self.mix = uniform, folded, mix
        self.original = sparse_jacobian_module.materialize_blocks

    def materialize(self, matvec, plan, *, structure, batched_matvec=None, probe_batch_size=None):
        def probed(probe):
            return self.original(
                matvec,
                probe.plan,
                structure=probe.structure,
                batched_matvec=batched_matvec,
                probe_batch_size=probe_batch_size,
            )

        uniform, folded = probed(self.uniform), probed(self.folded)
        if uniform.data.shape != folded.data.shape:
            raise RuntimeError("the two probes' block layouts differ; their values cannot be mixed")
        dropped = (folded.data == 0) & (uniform.data != 0)
        if self.mix == "drop":
            data = jnp.where(dropped, 0.0, uniform.data)
        elif self.mix == "fold":
            data = jnp.where(dropped, uniform.data, folded.data)
        else:
            data = uniform.data + self.mix * (folded.data - uniform.data)
        return ProbedBlocks(data, uniform.layout)

    def __enter__(self):
        sparse_jacobian_module.materialize_blocks = self.materialize
        return self

    def __exit__(self, *exc):
        sparse_jacobian_module.materialize_blocks = self.original


class StepRebuild:
    """The march's step at a checkpoint, rebuilt per preconditioner spec as the viscosity ramp builds it."""

    def __init__(self, coupled, phi, beta, step_number, iterate=None):
        self.coupled, self.phi, self.beta = coupled, phi, beta
        #: Where the solve linearizes: the step's start, or a later inner iterate of the same step.
        self.iterate = phi if iterate is None else iterate
        solver_spec = compare.SOLVER
        self.spec = solver_spec
        ramp = solver_spec.continuation
        companion = {
            "flow": scale_momentum_only,
            "both": scale_both_blocks,
            None: scale_both_blocks,
        }[ramp.scale]

        def assembler_for(station):
            if station == ramp.stations:
                return coupled
            return companion(coupled, ramp.anchor ** (1.0 - station / ramp.stations))

        self.anchor = assembler_for(0)
        self.seed = self.anchor.state_from_physical(*hybrid_initialize(self.anchor))
        self.station = min(step_number - 1, ramp.stations)
        self.assembler = assembler_for(self.station)

    def fitted(self, build_reach, refit_reach, mix=None):
        """The step's state after the re-fit, and its session.

        The session is built (which fixes the frozen coarsening) under a probe of ``build_reach`` and
        re-fitted (which supplies every value) under one of ``refit_reach`` -- or, given ``mix``, from
        the :class:`ScaledFold` mix of a uniform reach-3 probe and ``refit_reach``'s. Swapping the probe between the two reaches into the session's
        private spec and probe, which this harness may do and the library never does.
        """
        settings = self.spec.settings()
        file_pc = settings["preconditioner"]

        def with_reach(reach):
            return dataclasses.replace(
                file_pc, probe=dataclasses.replace(file_pc.probe, column_reach=reach)
            )

        allowed = inspect.signature(coupled_step).parameters
        march = {k: v for k, v in settings.items() if k in allowed and k != "preconditioner"}
        session = open_session(with_reach(build_reach), self.coupled)
        session.rebind(self.anchor)
        base = session.build(self.seed, **march)
        if refit_reach != build_reach and mix is None:
            session._spec = with_reach(refit_reach)
            session._probe = None  # rebuilt from the swapped spec when the refresh hook is made
        session.rebind(self.assembler)
        active = eqx.tree_at(
            lambda s: s.relaxation_schedule, base, ConstantRelaxation(jnp.asarray(self.beta))
        )
        if mix is None:
            update = session.refresh_preconditioner(active, self.phi, due=True)
        else:
            problem = session._problem
            groups = problem.groups() if isinstance(file_pc.inverse, FieldSplit) else None
            uniform = problem.probe(with_reach(None).probe.settings(), groups)
            folded = problem.probe(with_reach(refit_reach).probe.settings(), groups)
            with ScaledFold(uniform, folded, mix):
                update = session.refresh_preconditioner(active, self.phi, due=True)
        if update is not None:
            active = update(active)
        active = active.with_norm(coupled_scaled_norm(self.coupled, active.shift_policy, self.phi))
        return active, session

    def shift_and_preconditioner(self, active):
        """The solve's right-hand side ``R(p) + s (p - phi)``, the shift ``s`` and the preconditioner."""
        norm = active.residual_norm
        g0 = self.assembler.residual(self.phi)
        term = active.shift_policy.shift_term(self.phi, g0)
        strength = active.relaxation_schedule.shift_strength(norm(g0), norm(g0))
        shift = jax.lax.stop_gradient(term.shift(strength))
        g = self.assembler.residual(self.iterate) + shift * (self.iterate - self.phi)
        return g, shift, term.make_preconditioner(strength)


#: Cells whose faces' conductances span more than this ratio count as stretched.
STRETCHED = 10.0
#: The bins the pressure fold is reported over, by a cell's conductance ratio.
ANISOTROPY_BINS = ((1.0, 2.0), (2.0, STRETCHED), (STRETCHED, np.inf))


def solve_arm(rebuild, label, reaches, names, tight_rtol, anisotropy):
    started = time.time()
    active, session = rebuild.fitted(*reaches)
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
    phi, p = rebuild.phi, rebuild.iterate
    g, shift, preconditioner = rebuild.shift_and_preconditioner(active)
    norm = active.residual_norm
    apply_a = jax.jit(lambda v: jax.jvp(residual_fn, (p,), (v,))[1] + shift * v)
    G = jax.jit(lambda x: residual_fn(x) + shift * (x - phi))
    base_solver = active.linear_solver()
    project = active.step_projection
    print(
        f"[arm] {label}: fitted in {time.time() - started:.0f} s, floored rows {floored_text}",
        flush=True,
    )
    for tag, solver in (
        ("march rtol", base_solver),
        (f"rtol {tight_rtol:g}", with_relative_tolerance(base_solver, tight_rtol)),
    ):
        solver = in_progress_measure(solver, norm)
        t0 = time.time()
        d, cycles = _shifted_solve(residual_fn, p, g, shift, preconditioner, solver)
        linear = apply_a(d) + g
        line = (
            f"  {tag:11s} cycles {int(cycles):3d}  Euclidean "
            f"{float(jnp.linalg.norm(linear) / jnp.linalg.norm(g)):.2e}  row-scaled "
            f"{float(norm(linear) / norm(g)):.3e}  ({time.time() - t0:.0f} s)"
        )
        # Which row fields the unsolved residual sits in, in the step's measure (each field's rows alone).
        shares = [
            float(norm(jnp.where(jnp.arange(linear.size) // n == f, linear, 0.0)) / norm(g))
            for f in range(len(names))
        ]
        line += "\n              by row field " + " ".join(
            f"{names[f]} {shares[f]:.2e}" for f in range(len(names))
        )
        # The continuity rows' part held by stretched cells, against their share of the cells.
        pressure = len(names) - 3
        stretched_rows = jnp.asarray(
            np.concatenate(
                [np.zeros(pressure * n, bool), anisotropy > STRETCHED, np.zeros(2 * n, bool)]
            )
        )
        in_stretched = float(norm(jnp.where(stretched_rows, linear, 0.0)) / norm(g))
        line += (
            f"\n              p rows in cells above {STRETCHED:g}:1 ({np.mean(anisotropy > STRETCHED):.0%} of "
            f"cells): {in_stretched:.2e} of p's {shares[pressure]:.2e}"
        )
        if tag == "march rtol":
            v = project(p, d) if project is not None else d
            ladder = [float(norm(G(p + a * v))) / float(norm(g)) for a in LADDER]
            line += "\n              ladder " + " ".join(f"{r:.3f}" for r in ladder)
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
    print(
        f"[matrix] {j3.shape[0]} dofs, {j3.nnz / 1e6:.1f}M nnz, both in {time.time() - t0:.0f} s",
        flush=True,
    )
    e = (j2 - j3).tocsr()
    print(
        "\n  |E_ab| / |J3_ab| (row field x column field; E = reach-2 colouring - uniform reach 3)",
        flush=True,
    )
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
    print(
        f"  {'field':>5s} {'moved':>8s} {'median':>10s} {'p99':>10s} {'max':>10s} {'rows > 1e-2':>12s}",
        flush=True,
    )
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
    print(
        "\n  off-diagonal fold |E_ij| / |J3_ii + s_i|, largest per row field (over every column)",
        flush=True,
    )
    for f in range(n_fields):
        in_field = (rows >= f * n) & (rows < (f + 1) * n)
        print(
            f"  {names[f]:>5s} max {weight[in_field].max() if in_field.any() else 0.0:10.2e}",
            flush=True,
        )
    schur_report(j3, j2, shift, n, n_fields)
    pressure_entry_report(j3, e, n, n_fields, conductance_ratio(mesh, mesh.geometry()), (owner, nb))
    del j2, j3, e
    gc.collect()


def schur_parts(jacobian, shift, n, n_fields):
    """``C``, ``D M G`` and ``S = C - D M G`` of the shifted flow block, with ``M = 1 / F_ii``."""
    dim = n_fields - 3  # velocity components, then p, k, omega
    shifted = (jacobian + sp.diags(shift)).tocsr()
    velocity, pressure = slice(0, dim * n), slice(dim * n, (dim + 1) * n)
    f = shifted[velocity, velocity]
    g = shifted[velocity, pressure]
    d = shifted[pressure, velocity]
    c = shifted[pressure, pressure].tocsr()
    dmg = (d @ sp.diags(1.0 / f.diagonal()) @ g).tocsr()
    return c, dmg, (c - dmg).tocsr()


def pressure_entry_report(j3, e, n, n_fields, anisotropy, edges):
    """Entry by entry, how far the fold moves the pressure couplings it touches.

    A block-norm ratio is set by a block's strongest couplings, so it cannot see a fold that is small
    beside those and large beside the weak ones -- which on a stretched cell are the couplings along its
    long direction. This reads each changed entry against its own value, and each pressure row's fold
    against that row's strongest neighbour coupling, binned by the cell's conductance ratio.
    """
    p = n_fields - 3
    velocity = slice(0, p * n)
    pressure = slice(p * n, (p + 1) * n)
    print(
        "\n  pressure couplings, entry by entry: DROPPED (zeroed by the short probe, the distance-3 ring)"
        " vs FOLDED (a near entry moved by r = |E_ij| / |J3_ij|)",
        flush=True,
    )
    for label, rows, cols in (
        ("p rows x p (C)", pressure, pressure),
        ("velocity rows x p (G)", velocity, pressure),
        ("p rows x velocity (D)", pressure, velocity),
    ):
        moved = e[rows, cols].tocoo()
        keep = moved.data != 0
        r_i, c_i, delta = moved.row[keep], moved.col[keep], moved.data[keep]
        exact = np.asarray(j3[rows, cols][r_i, c_i]).ravel()
        dropped = (exact != 0) & (exact + delta == 0)
        folded = ~dropped
        ratio = np.abs(delta[folded]) / np.where(exact[folded] != 0, np.abs(exact[folded]), np.inf)
        near = exact[folded]
        flipped = int(np.sum((near != 0) & (np.sign(near + delta[folded]) != np.sign(near))))
        block = float(np.linalg.norm(j3[rows, cols].data))
        print(
            f"  {label:22s} dropped {int(dropped.sum()):8d} (|.|/|block| "
            f"{np.linalg.norm(exact[dropped]) / block:.2e})  folded {int(folded.sum()):8d} (|.|/|block| "
            f"{np.linalg.norm(delta[folded]) / block:.2e}; r median {np.median(ratio):.2e} p99 "
            f"{np.quantile(ratio, 0.99):.2e}, r > 0.1: {int(np.sum(ratio > 0.1))}, r > 1: "
            f"{int(np.sum(ratio > 1))}, sign flipped {flipped}, onto exact zeros {int(np.sum(near == 0))})",
            flush=True,
        )
    # Each continuity row's largest dropped and largest folded entry, against its largest neighbour
    # coupling (any leading column), by the cell's anisotropy.
    leading = slice(0, (p + 1) * n)
    rows3 = j3[pressure, leading].tocsr()
    kept = (rows3 + e[pressure, leading]) != 0  # what the short probe left nonzero
    strongest_matrix = rows3.copy()
    strongest_matrix.setdiag(0.0, k=p * n)
    strongest = np.abs(strongest_matrix).max(axis=1).toarray().ravel()
    strongest = np.where(strongest > 0, strongest, np.inf)
    dropped = abs(rows3 - rows3.multiply(kept)).max(axis=1).toarray().ravel() / strongest
    folded = abs(e[pressure, leading].multiply(kept)).max(axis=1).toarray().ravel() / strongest
    print(
        "  continuity rows, against each row's strongest neighbour coupling max_(j != i)|J3_ij|, by the "
        "cell's conductance ratio (median / p99):",
        flush=True,
    )
    for low, high in ANISOTROPY_BINS:
        cells = (anisotropy >= low) & (anisotropy < high)
        if cells.any():
            print(
                f"    {low:g}-{high:g}:1  {int(cells.sum()):6d} cells  largest dropped "
                f"{np.median(dropped[cells]):.2e} / {np.quantile(dropped[cells], 0.99):.2e}  largest folded "
                f"{np.median(folded[cells]):.2e} / {np.quantile(folded[cells], 0.99):.2e}",
                flush=True,
            )

    # Graph distance is not physical distance: across a thin cell, a cell three faces away can couple
    # more strongly than the face neighbour along the cell's long side. So read each continuity row's
    # largest DROPPED pressure coupling against the row's WEAKEST face-neighbour coupling in C.
    c3 = j3[pressure, pressure].tocsr()
    owner, nb = edges
    face_values = np.abs(
        np.concatenate([np.asarray(c3[owner, nb]).ravel(), np.asarray(c3[nb, owner]).ravel()])
    )
    face_rows = np.concatenate([owner, nb])
    weakest = np.full(n, np.inf)
    np.minimum.at(weakest, face_rows, np.where(face_values > 0, face_values, np.inf))
    kept_c = (c3 + e[pressure, pressure]) != 0
    dropped_c = abs(c3 - c3.multiply(kept_c)).max(axis=1).toarray().ravel()
    ratio = dropped_c / np.where(weakest > 0, weakest, np.inf)
    print(
        "  C rows: largest dropped |C_ij| / weakest face-neighbour |C_ij|, by the cell's conductance ratio "
        "(median / p90 / p99; share of rows above 0.1):",
        flush=True,
    )
    for low, high in ANISOTROPY_BINS:
        cells = (anisotropy >= low) & (anisotropy < high)
        if cells.any():
            r = ratio[cells]
            print(
                f"    {low:g}-{high:g}:1  {int(cells.sum()):6d} cells  {np.median(r):.2e} / "
                f"{np.quantile(r, 0.9):.2e} / {np.quantile(r, 0.99):.2e}; {np.mean(r > 0.1):.1%}",
                flush=True,
            )


def schur_report(j3, j2, shift, n, n_fields):
    c3, dmg3, s3 = schur_parts(j3, shift, n, n_fields)
    c2, dmg2, s2 = schur_parts(j2, shift, n, n_fields)

    def frobenius(m):
        return float(np.linalg.norm(m.data))

    print(
        "\n  pressure Schur complement S = C - D M G (M = 1/F_ii, a proxy for the smoother's block inverse)",
        flush=True,
    )
    print(
        f"  |C| {frobenius(c3):.3e}  |D M G| {frobenius(dmg3):.3e}  |S| {frobenius(s3):.3e}  "
        f"cancellation |D M G| / |S| {frobenius(dmg3) / frobenius(s3):.2f}",
        flush=True,
    )
    print(
        f"  fold: |dC|/|C| {frobenius(c2 - c3) / frobenius(c3):.2e}  "
        f"|d(D M G)|/|D M G| {frobenius(dmg2 - dmg3) / frobenius(dmg3):.2e}  "
        f"|dS|/|S| {frobenius(s2 - s3) / frobenius(s3):.2e}",
        flush=True,
    )
    diagonal3, diagonal2 = s3.diagonal(), s2.diagonal()
    change = np.abs(diagonal2 - diagonal3) / np.abs(diagonal3)
    print(
        f"  S_ii: relative change median {np.median(change):.2e} p99 {np.quantile(change, 0.99):.2e} "
        f"max {change.max():.2e}; rows above 1e-2 {int(np.sum(change > 1e-2))}, above 1e-1 "
        f"{int(np.sum(change > 1e-1))}, sign flipped {int(np.sum(np.sign(diagonal2) != np.sign(diagonal3)))}",
        flush=True,
    )
    off3 = np.abs(s3 - sp.diags(diagonal3)).sum(axis=1).A1
    print(
        f"  S row dominance |S_ii| / sum_j|S_ij|: median {np.median(np.abs(diagonal3) / off3):.3f}, "
        f"rows below 1: {int(np.sum(np.abs(diagonal3) < off3))} of {n}",
        flush=True,
    )


def main():
    step_number = int(os.environ["REACH_FOLDING_STEP"])
    checkpoints = Path(os.environ.get("REACH_FOLDING_CHECKPOINTS", CASE / "checkpoints"))
    tight_rtol = float(os.environ.get("REACH_FOLDING_TIGHT_RTOL", "1e-5"))
    with np.load(checkpoints / f"state-{step_number - 1:05d}.npz") as data:
        phi = jnp.asarray(data["state"])
        start = {k: data[k] for k in ("step", "residual_norm", "shift") if k in data.files}
    with np.load(checkpoints / f"state-{step_number:05d}.npz") as data:
        beta = float(data["shift"])
    inner = int(os.environ.get("REACH_FOLDING_INNER", "0"))
    iterate, where = None, "the step's start"
    if inner > 0:
        attempt = int(os.environ.get("REACH_FOLDING_ATTEMPT", str(step_number)))
        dump = checkpoints / f"inner-{attempt:05d}-{inner - 1:02d}.npz"
        with np.load(dump) as data:
            iterate = jnp.asarray(data["state"])
            where = (
                f"inner {inner} (after {dump.name}: {int(data['cycles'])} cycles, alpha "
                f"{float(data['alpha']):g})"
            )
    coupled = compare.build_case()["coupled"]
    n, n_fields = coupled.layout.n_cells, coupled.layout.n_fields
    names = field_names(n_fields)
    rebuild = StepRebuild(coupled, phi, beta, step_number, iterate)
    mesh = coupled.momentum.mesh
    anisotropy = conductance_ratio(mesh, mesh.geometry())
    print(
        f"[configuration] case {CASE_NAME}, {n} cells, step {step_number} (station {rebuild.station}) from "
        f"state-{step_number - 1:05d} {start}, solving at {where}, beta {beta:g}, split precision "
        f"{compare.SOLVER.preconditioner.inverse.precision}, file probe "
        f"{compare.SOLVER.preconditioner.probe}, jax {jax.__version__}",
        flush=True,
    )
    shift = None
    for label, reaches in arms(n_fields, names).items():
        arm_shift = solve_arm(rebuild, label, reaches, names, tight_rtol, anisotropy)
        shift = arm_shift if shift is None else shift
    if os.environ.get("REACH_FOLDING_SKIP_MATRIX") != "1":
        folding_report(coupled, rebuild.iterate, shift, names)


if __name__ == "__main__":
    main()
