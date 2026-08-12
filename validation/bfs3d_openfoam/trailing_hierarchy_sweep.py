"""Rank multigrid hierarchies on the ``[k, omega]`` block ALONE, against the PETSc V-cycle.

The trailing block of the coupled field split holds ~11 % of the operator's nonzeros, so the leading
saddle carries the solve and an in-situ measurement barely moves when the trailing inverse changes: a
trailing inverse that does not converge *at all* on its own block still produced a plausible blended
march estimate. This measures the block by itself, which is the only way to see the quality of a
preconditioner for it.

Two things are being compared, and until recently only one of them was honest. The PETSc V-cycle
receives an operator rescaled to a unit-magnitude diagonal; the JAX-native builder did not rescale, and
on this block the raw diagonal spans nearly six orders of magnitude. Every step of a multigrid setup
reads the diagonal — the smoother damping, the prolongation smoothing, and the spectral estimates that
scale both — so the two were coarsening different matrices and no comparison between them meant
anything. The ``equilibrate`` arms are the fix; the raw ones are kept beside them because the size of
the difference is the finding.

Reads the same rolling checkpoints as ``field_split_probe.py`` and shares its state table, shift
pairing and refusal-on-mismatch loader, so the operating point is the one that file documents.

Usage::

    python3 -u validation/bfs3d_openfoam/trailing_hierarchy_sweep.py state-00057
    python3 -u validation/bfs3d_openfoam/trailing_hierarchy_sweep.py state-00057 --arms=key,key

Runs one arm at a time and releases each hierarchy before building the next: a materialized 3D coupled
Jacobian is a substantial fraction of a workstation's memory, and holding two is what turns a probe
into a machine that has to be rebooted.
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
    build_amg_vcycle,
    build_convection_hierarchy,
    convection_multigrid_solve,
    relative_residual_gmres,
    solve_linear,
    symmetrically_equilibrate,
)
from aquaflux.solve.linear import restart_cycles  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
)
from field_split_probe import FLOOR, STATES, load_state, materialize  # noqa: E402
from jax.experimental.sparse import BCOO  # noqa: E402

#: Adjoint-grade, far past the march's own 30 % inexact-Newton stop, so arms separate rather than tie.
RTOL = 1e-8
#: A failing arm is identified by its true residual; letting one run to thousands of matrix-vector
#: products costs more than every healthy arm together.
SOLVER = relative_residual_gmres(RTOL, restart=15, stagnation_iters=40, max_restarts=60)
#: Fields in the trailing group.
N_TRAILING = 2
#: Damped-Jacobi sweeps per level for the native arms, where the arm does not say otherwise. The
#: PETSc arms count incomplete-LU sweeps, which are a different and much stronger unit of work — the
#: two smoother counts are NOT comparable, which is why every arm's own count is in its label.
NATIVE_SWEEPS = 4


def trailing_block(shifted: sp.csr_matrix, groups: FieldGroups) -> sp.csr_matrix:
    """The ``[k, omega]`` diagonal sub-block, field-major throughout.

    A field-major layout puts each whole field in a contiguous range, so the group is a contiguous
    slice on both axes rather than a gather — which is also why slicing this matrix cell-major would
    silently yield a different matrix that still looks plausible.
    """
    return sp.csr_matrix(shifted[groups.trailing, :][:, groups.trailing])


def report_scaling(block: sp.csr_matrix) -> None:
    """Print what the equilibration actually changes on this block, before any arm runs.

    The prolongation-smoothing step length is ``1.4 / sigma_max(D^-1 A)``, so these two numbers are
    the whole reason the raw and rescaled hierarchies are different algorithms rather than the same
    one at a different scale.
    """
    diagonal = np.abs(block.diagonal())
    scaled, _ = symmetrically_equilibrate(block)
    print(
        f"  diagonal {diagonal.min():.3e} .. {diagonal.max():.3e}  "
        f"(span {diagonal.max() / diagonal.min():.2e})",
        flush=True,
    )
    for name, matrix in (("raw", block), ("equilibrated", scaled)):
        d_inv = sp.diags(1.0 / matrix.diagonal())
        print(f"  sigma_max(D^-1 A), {name:<13} {_largest_singular_value(d_inv @ matrix):.3e}")
    print(flush=True)


def _largest_singular_value(matrix: sp.spmatrix, iterations: int = 20) -> float:
    """Power iteration on ``M^T M`` — the quantity the standard prolongator smoothing divides by."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(matrix.shape[1])
    v /= np.linalg.norm(v)
    sigma = 1.0
    for _ in range(iterations):
        w = matrix.T @ (matrix @ v)
        norm = np.linalg.norm(w)
        if norm == 0.0:
            return 0.0
        v, sigma = w / norm, np.sqrt(norm)
    return float(sigma)


def petsc_cycle(block: sp.csr_matrix, *, smoother: str, sweeps: int):
    """One PETSc GAMG V-cycle over the block, at the shipped bundle's aggregation and coarse limit."""
    vcycle = build_amg_vcycle(
        block,
        N_TRAILING,
        smoother_fill_levels=compare.FILL_LEVELS,
        smoother_sweeps=sweeps,
        coarse_eq_limit=compare.COARSE_EQ_LIMIT,
        extra_options={"mg_levels_pc_type": smoother} if smoother != "ilu" else None,
    )
    apply = MonolithicAmgPreconditioner(vcycle).matvec()
    return apply, vcycle, f"{vcycle.levels} levels, {vcycle.coarse_size} coarse eq"


def native_cycle(
    block: sp.csr_matrix,
    *,
    mis: bool,
    smoothing: str,
    equilibrate: bool,
    sweeps: int,
    aggressive: int = 0,
    undamped: bool = False,
    max_coarse: int = 16,
    max_levels: int = 2,
):
    """One JAX-native nodal V-cycle over the block.

    ``undamped`` reproduces PETSc's level smoother exactly: it runs ``richardson`` at its default
    scale of **1**, so the sweep is ``x += D^-1 (b - A x)`` with no damping, where ours by default
    relaxes by ``omega / lambda_max``.
    """
    hierarchy = build_convection_hierarchy(
        block,
        block_size=N_TRAILING,
        mis_aggregation=mis,
        prolongation_smoothing=smoothing,
        equilibrate=equilibrate,
        aggressive_levels=aggressive,
        max_coarse=max_coarse,
        max_levels=max_levels,
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

    return apply, None, f"{len(hierarchy.levels)} lv, {coarse} coarse eq"


def matched(**overrides):
    """The arm that reproduces PETSc: aggressive level, plain prolongation, undamped smoother."""
    base = dict(mis=True, smoothing="none", equilibrate=False, aggressive=1, undamped=True)
    return lambda b: native_cycle(b, **(base | overrides))


#: ``key -> (label, build)``.
#:
#: **The arm to read is the matched pair**, ``petsc-pbjacobix4`` against ``mis-aggressive-*``: same
#: smoother class, same sweep count, same plain aggregation, and the same *aggressive* first level.
#: Comparing our damped block Jacobi against PETSc's incomplete-LU measures the smoother instead of
#: the coarsening, and an ILU sweep is both far stronger and the least parallel piece in the cycle.
#:
#: The aggressive level is not a tuning knob here — it is what PETSc does by default and we did not.
#: ``build_amg_vcycle`` never sets ``pc_gamg_aggressive_coarsening``, so GAMG applies its own default
#: of one aggressive level on level 0, coarsening the SQUARED graph. Our builder defaulted to none,
#: which is why the two coarse spaces differed ~5× in size and why calling our hierarchy the same
#: algorithm was wrong. (``use_aggressive_square_graph`` and ``aggressive_mis_k`` are *alternatives*,
#: not a pair: with the squared graph on — the default — the coarsener is plain MIS at distance 1 and
#: the ``mis_k`` setting is never read.)
#:
#: Two verified differences remain even in the matched pair, both under test elsewhere: PETSc's
#: Richardson smoother is UNDAMPED (scale 1) where ours scales by ``omega / lambda_max``, and PETSc
#: follows the squared-graph coarsening with a fix-up pass that re-attaches each distance-2 aggregate
#: member to a root it is genuinely adjacent to in the *unsquared* graph. We do neither.
ARMS = (
    ("petsc-ilu0x1", "PETSc GAMG, ILU(0) x1", lambda b: petsc_cycle(b, smoother="ilu", sweeps=1)),
    ("petsc-ilu0x4", "PETSc GAMG, ILU(0) x4", lambda b: petsc_cycle(b, smoother="ilu", sweeps=4)),
    (
        "petsc-pbjacobix4",
        "PETSc GAMG, point-block Jacobi x4",
        lambda b: petsc_cycle(b, smoother="pbjacobi", sweeps=4),
    ),
    (
        "rcm-raw",
        f"ours RCM / symmetric-part x{NATIVE_SWEEPS}, raw",
        lambda b: native_cycle(
            b, mis=False, smoothing="symmetric-part", equilibrate=False, sweeps=NATIVE_SWEEPS
        ),
    ),
    (
        "rcm-eq",
        f"ours RCM / symmetric-part x{NATIVE_SWEEPS}, EQUILIBRATED",
        lambda b: native_cycle(
            b, mis=False, smoothing="symmetric-part", equilibrate=True, sweeps=NATIVE_SWEEPS
        ),
    ),
    (
        "mis-plain-raw",
        f"ours MIS / no smoothing x{NATIVE_SWEEPS}, raw",
        lambda b: native_cycle(
            b, mis=True, smoothing="none", equilibrate=False, sweeps=NATIVE_SWEEPS
        ),
    ),
    (
        "mis-plain-eq",
        f"ours MIS / no smoothing x{NATIVE_SWEEPS}, EQUILIBRATED",
        lambda b: native_cycle(
            b, mis=True, smoothing="none", equilibrate=True, sweeps=NATIVE_SWEEPS
        ),
    ),
    (
        "mis-standard-raw",
        f"ours MIS / standard x{NATIVE_SWEEPS}, raw",
        lambda b: native_cycle(
            b, mis=True, smoothing="standard", equilibrate=False, sweeps=NATIVE_SWEEPS
        ),
    ),
    (
        "mis-standard-eq",
        f"ours MIS / standard x{NATIVE_SWEEPS}, EQUILIBRATED",
        lambda b: native_cycle(
            b, mis=True, smoothing="standard", equilibrate=True, sweeps=NATIVE_SWEEPS
        ),
    ),
    (
        "mis-standard-eq-x8",
        "ours MIS / standard x8, EQUILIBRATED",
        lambda b: native_cycle(b, mis=True, smoothing="standard", equilibrate=True, sweeps=8),
    ),
    # The matched pair: PETSc's own default coarsening, which is one aggressive (squared-graph) level.
    (
        "mis-aggressive-raw",
        f"ours MIS aggressive / plain x{NATIVE_SWEEPS}, raw",
        lambda b: native_cycle(
            b,
            mis=True,
            smoothing="none",
            equilibrate=False,
            sweeps=NATIVE_SWEEPS,
            aggressive=1,
        ),
    ),
    (
        "mis-aggressive-eq",
        f"ours MIS aggressive / plain x{NATIVE_SWEEPS}, EQUILIBRATED",
        lambda b: native_cycle(
            b, mis=True, smoothing="none", equilibrate=True, sweeps=NATIVE_SWEEPS, aggressive=1
        ),
    ),
    (
        "mis-aggressive-raw-x8",
        "ours MIS aggressive / plain x8, raw",
        lambda b: native_cycle(
            b, mis=True, smoothing="none", equilibrate=False, sweeps=8, aggressive=1
        ),
    ),
    (
        "mis-aggressive-standard-x8",
        "ours MIS aggressive / standard x8, EQUILIBRATED",
        lambda b: native_cycle(
            b, mis=True, smoothing="standard", equilibrate=True, sweeps=8, aggressive=1
        ),
    ),
    # The fully matched arm: PETSc's coarsening AND PETSc's undamped Richardson scale.
    (
        "matched-x1",
        "ours MATCHED (aggressive / plain / undamped) x1",
        lambda b: native_cycle(
            b, mis=True, smoothing="none", equilibrate=False, sweeps=1, aggressive=1, undamped=True
        ),
    ),
    (
        "matched-x2",
        "ours MATCHED (aggressive / plain / undamped) x2",
        lambda b: native_cycle(
            b, mis=True, smoothing="none", equilibrate=False, sweeps=2, aggressive=1, undamped=True
        ),
    ),
    (
        "matched-x4",
        "ours MATCHED (aggressive / plain / undamped) x4",
        matched(sweeps=4),
    ),
    # PETSc stops coarsening on the COARSE SIZE; we stop on the level count, and at 2 levels our
    # `max_coarse` can never fire. This is that rule, not a deeper hierarchy for its own sake.
    (
        "matched-x4-coarsesize",
        "ours MATCHED x4, coarse-size stop (2000)",
        matched(sweeps=4, max_coarse=2000, max_levels=20),
    ),
    (
        "matched-x2-coarsesize",
        "ours MATCHED x2, coarse-size stop (2000)",
        matched(sweeps=2, max_coarse=2000, max_levels=20),
    ),
    (
        "matched-x1-coarsesize",
        "ours MATCHED x1, coarse-size stop (2000)",
        matched(sweeps=1, max_coarse=2000, max_levels=20),
    ),
    # Equilibrated. Measured worse on the DAMPED, non-aggressive arm, and never tried on the matched
    # one -- where it is also the only thing that makes the per-cell block solve SAFE: rescaled, each
    # 2x2 is unit-triangular with determinant exactly 1 and cannot be singular, while a raw one can and
    # on a developed state four of 23040 are.
    ("matched-x4-eq", "ours MATCHED x4, EQUILIBRATED", matched(sweeps=4, equilibrate=True)),
    ("matched-x2-eq", "ours MATCHED x2, EQUILIBRATED", matched(sweeps=2, equilibrate=True)),
)


def run_arm(label, build, operator, rhs):
    """Build one arm, solve the block with it, report cycles and the TRUE residual.

    A raise is a result about that arm — a singular coarse solve, a zero pivot, a refused diagonal —
    so it is reported and the arms queued behind it still run.
    """
    handle = None
    try:
        started = time.time()
        apply, handle, shape = build()
        built = time.time() - started
        solving = time.time()
        solution, raw = solve_linear(operator, rhs, SOLVER, preconditioner=apply, throw=False)
        true = float(jnp.linalg.norm(operator(solution) - rhs) / jnp.linalg.norm(rhs))
        print(
            f"    {label:<44} {shape:<22} build {built:>5.1f}s  cycles {restart_cycles(int(raw)):>4}"
            f"  TRUE rel {true:.3e}  solve {time.time() - solving:>5.1f}s",
            flush=True,
        )
        return restart_cycles(int(raw)), true
    except Exception as failure:
        print(f"    {label:<44} FAILED  {type(failure).__name__}: {failure}", flush=True)
        return None
    finally:
        if handle is not None:
            handle.destroy()
        del handle
        gc.collect()


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--arms=")]
    chosen = [a for a in sys.argv[1:] if a.startswith("--arms=")]
    only = set(chosen[-1].split("=", 1)[1].split(",")) if chosen else None
    if len(argv) != 1 or argv[0] not in STATES:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{' | '.join(STATES)}> [--arms=key,key]")
    if only is not None and (missing := only - {key for key, _, _ in ARMS}):
        raise SystemExit(f"unknown arm(s) {sorted(missing)}; known: {[k for k, _, _ in ARMS]}")

    name = argv[0]
    march_beta, _, description = STATES[name]
    pc_beta = max(march_beta, FLOOR) if march_beta > 0 else 0.0

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    groups = FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,
        n_trailing_fields=N_TRAILING,
    )
    print(
        f"{'=' * 118}\ntrailing [k, omega] block ALONE over {groups.n_cells} cells, "
        f"GMRES to rtol {RTOL:.0e} on the TRUE residual (restart 15)\n"
        f"bundle: plain aggregation, ILU({compare.FILL_LEVELS}) where the arm does not override it, "
        f"coarse_eq_limit {compare.COARSE_EQ_LIMIT}, stencil reach 3\n"
        f"state {name} -- {description}\n"
        f"operator beta {march_beta}, preconditioner beta {pc_beta}\n{'=' * 118}",
        flush=True,
    )

    state = load_state(name)
    plan = _coupled_jacobian_plan(coupled, 3)
    structure = block_stencil_gather_map(plan)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    jacobian = materialize(coupled, state, plan, structure, n_fields)
    shift = _frozen_shift_diagonal(base, pc_beta, state) if pc_beta > 0 else np.zeros(groups.n_dofs)
    block = trailing_block(MonolithicAmgPreconditioner._shifted(jacobian, shift), groups)
    del jacobian
    gc.collect()
    print(
        f"  block {block.shape[0]} dofs, {block.nnz / 1e6:.2f}M nnz ({block.nnz / block.shape[0]:.0f}/row)",
        flush=True,
    )
    report_scaling(block)

    sparse = BCOO.from_scipy_sparse(block.tocoo())

    def operator(v):
        return sparse @ v

    # A RANDOM right-hand side, not the steady residual restricted to these rows. The question is how
    # well each hierarchy inverts this operator, and a physical right-hand side is smooth in exactly the
    # directions a coarse grid handles best -- it flatters every arm and compresses the differences.
    rhs = jnp.asarray(np.random.default_rng(0).standard_normal(block.shape[0]))

    print("  -- arms\n", flush=True)
    for key, label, build in ARMS:
        if only is None or key in only:
            run_arm(label, lambda b=block, f=build: f(b), operator, rhs)


if __name__ == "__main__":
    main()
