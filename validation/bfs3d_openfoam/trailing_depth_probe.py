"""Does deepening the native ``[k, omega]`` trailing hierarchy (the flow saddle's own levers) help?

``trailing_hierarchy_sweep.py`` already carries ``max_levels``/``max_coarse`` knobs on its native arms,
but every arm in it is fixed at 2 levels and none of the coarse-size-stopped arms in its ``ARMS`` table
has ever actually been run. Separately, neither ``avoid_singletons`` nor ``strength_threshold`` is wired
into that harness's ``smoothed_cycle`` at all, though both are plain
:func:`~aquaflux.solve.build_convection_hierarchy` parameters and both were the levers that made depth
pay on the flow saddle's own native hierarchy (a coarsening/singleton-aggregate fix worth ~1.7x there,
and a strength-of-connection threshold worth ~4x).

This is a one-shot check of the transfer: does the flow saddle's recipe (avoid singleton aggregates,
deepen past 2 levels via a coarse-size stop rather than a level cap, optionally a strength threshold)
do anything sane on the trailing block, or does it misbehave the way a raw (non row-field-normalized)
threshold might on this specific block -- its k/omega diagonal spans ~8 orders of magnitude, so a
threshold that reads the raw operator's ``|A_ij|`` risks becoming an omega-only measure rather than a
genuinely per-field one?

**Uses a real checkpoint when one validates, a fresh ``hybrid_initialize`` state otherwise.** A clean
worktree has neither, but a checkpoint recovered from another worktree (the rolling buffer is
gitignored, so it isn't carried by `git worktree add`) is preferred and validated the same way
``field_split_probe.py`` validates one -- against its own recorded shift/residual fingerprint -- because
a same-named file from a different run is a silent-failure trap (see ``field_split_probe.load_state``'s
own docstring). ``state-00067`` is the default: the converged root's own **zero-shift** operator, the
one every ``jax.grad`` transpose solve actually meets and the hardest, most-discriminating trailing-block
case this file's sibling harnesses use. Falling back to a cold ``hybrid_initialize`` state (an UNSHIFTED,
undeveloped operator no preconditioner in this package is ever asked to invert in practice) produced
uninformative TRUE-rel-1.0 results across every arm on a first pass at this probe -- keep that in mind
if this ever falls back silently.

Usage::

    python3 -u validation/bfs3d_openfoam/trailing_depth_probe.py [state-000NN]
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    build_convection_hierarchy,
    convection_multigrid_solve,
    relative_residual_gmres,
    restart_cycles,
    solve_linear,
)
from aquaflux.turbulence import hybrid_initialize  # noqa: E402
from aquaflux.turbulence.coupled import _coupled_jacobian_plan  # noqa: E402
from field_split_probe import STATES, load_state, materialize  # noqa: E402
from jax.experimental.sparse import BCOO  # noqa: E402
from trailing_hierarchy_sweep import N_TRAILING, report_scaling, trailing_block  # noqa: E402

RTOL = 1e-8
SOLVER = relative_residual_gmres(RTOL, restart=15, stagnation_iters=40, max_restarts=60)


def smoothed_cycle(
    block: sp.csr_matrix,
    *,
    sweeps: int = 4,
    aggressive: int = 1,
    equilibrate: bool = True,
    undamped: bool = True,
    max_coarse: int = 16,
    max_levels: int = 2,
    avoid_singletons: bool = False,
    strength_threshold: float = 0.0,
):
    """The ``JacobiSmoothedInverse`` class's own bundle, with the two extra levers the flow saddle used.

    Defaults reproduce ``JacobiSmoothedInverse``'s own constructor defaults exactly (``mis_aggregation=True``,
    ``aggressive_levels=1``, ``prolongation_smoothing="none"``, ``spectral_damping=False`` i.e.
    ``undamped=True`` here, ``equilibrate=True``, 4 sweeps) at its 2-level cap. ``avoid_singletons`` and
    ``strength_threshold`` are the two levers this file is measuring the transfer of.
    """
    hierarchy = build_convection_hierarchy(
        block,
        block_size=N_TRAILING,
        mis_aggregation=True,
        prolongation_smoothing="none",
        equilibrate=equilibrate,
        aggressive_levels=aggressive,
        max_coarse=max_coarse,
        max_levels=max_levels,
        avoid_singletons=avoid_singletons,
        strength_threshold=strength_threshold,
    )
    coarse = hierarchy.levels[-1].n

    def apply(residual: jnp.ndarray) -> jnp.ndarray:
        return convection_multigrid_solve(
            hierarchy,
            residual,
            cycles=1,
            sweeps=sweeps,
            omega=1.0 if undamped else 0.8,
            spectral_damping=not undamped,
        )

    return apply, f"{len(hierarchy.levels)} lv, {coarse} coarse eq"


ARMS = (
    ("shipped-2lv", "shipped bundle, 2 levels, aggressive first level (today's default)", dict()),
    (
        "2lv-nosingle",
        "+ avoid_singletons, still 2 levels (isolates the singleton fix alone)",
        dict(avoid_singletons=True),
    ),
    (
        # aggressive=0: the shipped aggressive (squared-graph) first level jumps straight past
        # max_coarse in one step (verified on the first, confounded run of this probe -- "deep"
        # arms at aggressive=1 landed at 2 levels regardless of max_levels). Plain MIS coarsens
        # gradually so max_coarse actually gets a chance to bind after more than one level.
        "deep-plain",
        "deepened, PLAIN (non-aggressive) coarsening, coarse-size stop, singletons NOT fixed",
        dict(max_levels=20, max_coarse=200, aggressive=0),
    ),
    (
        "deep-plain-nosingle",
        "UNIFICATION ARM: deepened (plain coarsening) + avoid_singletons",
        dict(max_levels=20, max_coarse=200, aggressive=0, avoid_singletons=True),
    ),
    (
        "deep-plain-strength-nosinglefix",
        "+ strength_threshold=0.25, singletons NOT fixed -- exactly the live march's configuration "
        "(BFS3D_NATIVE_STRENGTH_THRESHOLD=0.25 with no avoid_singletons support), to test whether "
        "the missing singleton fix is what made that march's refactor step ~49s instead of the "
        "single-digit seconds a comparably deep flow-saddle hierarchy costs elsewhere",
        dict(max_levels=20, max_coarse=200, aggressive=0, strength_threshold=0.25),
    ),
    (
        "deep-plain-nosingle-strength",
        "+ strength_threshold=0.25 AND avoid_singletons -- EXPLORATORY, raw (non row-field-"
        "normalized) threshold on a block whose k/omega diagonal spans ~8 orders; solve.md's "
        "DEFERRED section predicts this may read as an omega-only measure",
        dict(
            max_levels=20,
            max_coarse=200,
            aggressive=0,
            avoid_singletons=True,
            strength_threshold=0.25,
        ),
    ),
)


def run_arm(label: str, build_kwargs: dict, operator, rhs) -> None:
    try:
        started = time.time()
        apply, shape = smoothed_cycle(operator.block, **build_kwargs)
        built = time.time() - started
        solving = time.time()
        solution, raw = solve_linear(
            operator.matvec, rhs, SOLVER, preconditioner=apply, throw=False
        )
        true = float(jnp.linalg.norm(operator.matvec(solution) - rhs) / jnp.linalg.norm(rhs))
        print(
            f"    {label:<24} {shape:<22} build {built:>5.1f}s  cycles {restart_cycles(int(raw)):>4}"
            f"  TRUE rel {true:.3e}  solve {time.time() - solving:>5.1f}s",
            flush=True,
        )
    except Exception as failure:
        print(f"    {label:<24} FAILED  {type(failure).__name__}: {failure}", flush=True)
    finally:
        gc.collect()


class _Operator:
    def __init__(self, block: sp.csr_matrix):
        self.block = block
        sparse = BCOO.from_scipy_sparse(block.tocoo())
        self.matvec = lambda v: sparse @ v


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "state-00067"
    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    groups = FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,
        n_trailing_fields=N_TRAILING,
    )

    checkpoint_dir = CASE / "checkpoints"
    have_checkpoint = name in STATES and (checkpoint_dir / f"{name}.npz").exists()
    if have_checkpoint:
        march_beta = STATES[name].march_beta
        print(
            f"{'=' * 118}\n"
            f"trailing [k, omega] block from a REAL checkpoint, {name} ({STATES[name].description}), "
            f"operator beta {march_beta}, over {groups.n_cells} cells\n"
            f"GMRES to rtol {RTOL:.0e} on the TRUE residual (restart 15)\n{'=' * 118}",
            flush=True,
        )
        state = load_state(name)
    else:
        march_beta = compare.PC_BETA_FLOOR
        print(
            f"{'=' * 118}\n"
            f"NO VALIDATED CHECKPOINT for {name!r} -- falling back to a fresh hybrid_initialize state, "
            f"shifted at the PC beta floor ({march_beta}) as a stand-in for a real operating point. "
            f"This is NOT the converged-state adjoint operator; treat results as a sanity check only.\n"
            f"trailing [k, omega] block over {groups.n_cells} cells\n"
            f"GMRES to rtol {RTOL:.0e} on the TRUE residual (restart 15)\n{'=' * 118}",
            flush=True,
        )
        flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
        state = coupled.state_from_physical(flow, k, omega)

    residual0 = float(jnp.linalg.norm(coupled.residual(state)))
    if not np.isfinite(residual0):
        raise SystemExit(f"the state's residual is not finite ({residual0}) -- refusing to measure")
    print(f"  state residual |R| = {residual0:.4e}\n", flush=True)

    plan = _coupled_jacobian_plan(coupled, 3)
    structure = block_stencil_gather_map(plan)
    jacobian = materialize(coupled, state, plan, structure, n_fields)
    if march_beta > 0:
        # A simple diagonal-proportional stand-in for beta*d. The real physics shift diagonal needs
        # `_coupled_shift_policy`, which builds a "twolevel" AMG on the omega transport operator that
        # independently failed to build at the cold hybrid_initialize state (a real but separate
        # finding, not what this probe tests) -- this sidesteps that detour when no checkpoint exists.
        # At a real checkpoint's own documented `march_beta` this branch does not fire for state-00067
        # (0.0), which is deliberately the operator's OWN unshifted, hardest form.
        shift = march_beta * np.abs(jacobian.diagonal())
        jacobian = MonolithicAmgPreconditioner._shifted(jacobian, shift)
    block = trailing_block(jacobian, groups)
    del jacobian
    gc.collect()
    print(
        f"  block {block.shape[0]} dofs, {block.nnz / 1e6:.2f}M nnz ({block.nnz / block.shape[0]:.0f}/row)",
        flush=True,
    )
    report_scaling(block)

    operator = _Operator(block)
    rhs = jnp.asarray(np.random.default_rng(0).standard_normal(block.shape[0]))

    print("  -- arms\n", flush=True)
    for key, description, kwargs in ARMS:
        print(f"  [{key}] {description}", flush=True)
        run_arm(key, kwargs, operator, rhs)
        print(flush=True)


if __name__ == "__main__":
    main()
