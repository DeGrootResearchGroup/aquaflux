"""Can any standard remedy make a ZERO-FILL incomplete-LU smoother work on this case?

🛑 **SCOPE, BINDING: this sweep is MONOLITHIC and the case is FIELD-SPLIT, so no arm here describes the
preconditioner the case actually runs.** Every arm builds one V-cycle over all five fields interleaved
cell-major. The case splits them: the ``[u, v, p]`` saddle goes to the algebraic-multigrid V-cycle whose
level smoother is the incomplete factorization -- the only block a fill level governs -- and the
``[k, omega]`` pair goes to a nodal hierarchy that is not an incomplete factorization at all. So no
``k`` or ``omega`` row is ever eliminated by an ILU in the shipped solver, and every fill, ordering,
shift, reach and conditioning result below is a property of a five-field factorization that nothing
uses. The results are kept because they are real measurements of a real matrix, and because the
questions they answer (is it zero pivots? does a condition estimate rank the arms?) are worth having
answered. They are **not** to be quoted as properties of this case's solver, and the correctly-posed
version of the question is about the ``[u, v, p]`` block alone.

The monolithic algebraic-multigrid V-cycle smooths each level with a stationary incomplete-LU sweep.
Zero fill -- ILU(0) -- is what the three-dimensional backward-facing-step case runs, and it is the
member that survives a small pseudo-transient shift there. On **this** two-dimensional case the ranking
inverts: ILU(0) picks up negative pivots and the preconditioned solve fails, while one level of fill
converges. That inversion is why neither fill level can simply be made the default, and it is what this
script measures against.

The remedies swept are the standard ones for an incomplete factorization breaking down on a matrix that
is not an M-matrix:

* a **diagonal shift on the factorization** (``pc_factor_shift_type`` ``nonzero`` /
  ``positive_definite`` / ``inblocks``, across shift amounts) -- the classical fix, and the one that
  changes only the factorization rather than the operator being preconditioned;
* **more smoother sweeps** -- cheap, and it discriminates "too weak a smoother" from "an amplifying
  one": a weak smoother improves with sweeps, an amplifying one gets worse;
* a different **elimination ordering** (``pc_factor_mat_ordering_type``), since an incomplete
  factorization with no fill is strongly ordering-dependent and the shipped order is the cell-major one
  the equilibration hands it;
* dropping the **symmetric square-root-diagonal equilibration**, to check that the conditioning
  transform is not itself the thing that makes the zero-fill pivots small;
* a shorter **stencil reach** for the materialized preconditioner matrix, which is legal here because
  the Krylov operator is the exact Jacobian-vector product and only the *preconditioner* is built from
  the materialized matrix. A shorter reach is a different (sparser) factorization, and its pivots are
  measurably healthier on this case. **This arm is not one-variable and must not be read as one:** the
  aggregation is built from the same sparser matrix, so the reach arms also carry a different coarse
  space (a visibly different coarse size), and smoother pattern and coarse space move together in them.

Every arm's factorization is additionally scored by an infinity-norm **condition estimate** (see
:func:`smoother_factor_census`), which is what lets the sweep ask a question a cycle count alone cannot:
not merely *which* arm wins, but whether a quantity computable at **build time**, before any Krylov
iteration, could have picked the winner. That matters here because the winner moves with the shift, so
no static choice of fill level or ordering spans a march and only a per-matrix selector could.

**Method (each of these has produced a retracted verdict on this project):**

* judged on the **TRUE** residual through restarted GMRES (``KSP_NORM_UNPRECONDITIONED`` in spirit --
  the outer solve is right-preconditioned, so its Krylov residual is the true one, and the true relative
  residual is recomputed from the returned solution regardless);
* a **real** right-hand side, the case's own ``-R(state)``, never a random vector;
* the operator and the V-cycle are built at the **same** shift -- no preconditioner-only shift floor --
  so an arm's result is a property of the smoother and not of a mismatch;
* swept **down to zero shift**, which is the operator an implicit-function-theorem adjoint solves and
  where a zero-fill factorization has nothing to lean on;
* a cycle count is the verdict; a one-apply contraction ratio and a pivot census are **proxies** and are
  reported only beside it. The pivot census here is taken from the factor the smoother actually built,
  so it reflects whatever shift options the arm set.

The state is the case's own self-start (potential-flow velocity + Laplace-smoothed turbulence), cached
to ``ilu0_remedy_state.npz`` beside this file so repeated runs skip rebuilding it. A saved state from
elsewhere can be named on the command line.

Run from the repo root::

    validation/run_case.sh validation/pitzdaily_openfoam/ilu0_remedy_sweep.py
"""

from __future__ import annotations

import dataclasses
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))  # import aquaflux from the working tree, as compare.py is run
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    AmgVCycle,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    cell_major_permutation,
    relative_residual_gmres,
    restart_cycles,
    solve_linear,
)
from aquaflux.solve.amg_preconditioner import ShiftedCellMajorOperator  # noqa: E402
from aquaflux.turbulence import hybrid_initialize  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _DEFAULT_SHIFT_BASIS,
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_plan,
    _frozen_shift_diagonal,
    _jacobian_matvec,
    _monolithic_shift_source,
)

#: The pseudo-transient shifts to probe, largest first. ``0.0`` is the adjoint's own operator: there is
#: no shift to make the diagonal dominant, so a factorization that only works because of the shift is
#: exposed there and nowhere else.
BETAS = (0.5, 0.05, 0.0)

#: Stencil reach of the materialized Jacobian. This case needs **5** for a materialized matrix that
#: matches the Jacobian-vector product to the float64 floor (reach 3 leaves a 2.0e-07 relative error,
#: carried by the pressure column, because a skewness-corrected gradient's own Richardson sweeps widen
#: the residual's stencil). An arm may override it: a preconditioner is allowed to be built from an
#: inexact matrix, since the Krylov operator is the exact product either way.
REACH = 5

#: Sweeps, coarse-grid size and aggregation of the shipped bundle, held fixed across every arm unless an
#: arm overrides them. Plain aggregation (``pc_gamg_agg_nsmooths = 0``) is the library default.
SWEEPS = 4
COARSE_EQ_LIMIT = 2000

#: Far past any inexact-Newton stop, so arms separate rather than tie, and capped so a failing arm costs
#: a bounded amount: a failure is identified by its true residual, not by letting it run.
STUDY_SOLVER = relative_residual_gmres(1e-6, restart=15, stagnation_iters=40, max_restarts=40)

#: Build each arm's factorization, report its census, and skip the solve (``ILU0_SWEEP_CONDEST_ONLY=1``).
#: The point is to ask whether the condition estimate **ranks** the arms the way their cycle counts
#: already do, so this pass reads the factor and joins against recorded verdicts instead of paying for
#: them twice. Never read a ranking off a run that measured no cycle count of its own.
CONDEST_ONLY = os.environ.get("ILU0_SWEEP_CONDEST_ONLY", "") not in ("", "0")


class BlockIlu0:
    """A level smoother whose zero-fill incomplete-LU eliminates one CELL-BLOCK at a time.

    Reached as a PETSc *shell* preconditioner (``pc_type python``), because the direct route is closed:
    a block-sparse (``BAIJ``) matrix cannot be handed to the aggregation at all — PETSc has no
    ``MatCreateGraph`` for that storage, so the hierarchy refuses to build. Converting the **level
    operator** to block storage inside the smoother keeps the aggregation on the point matrix and makes
    only the factorization block-wise, which is the part the hypothesis is about.

    Why it is worth a separate arm: a scalar zero-fill elimination divides by one diagonal entry at a
    time, and a collocated pressure row's diagonal carries only the Rhie--Chow damping, so it is the
    entry most likely to come out near zero. A block elimination never divides by that entry alone — it
    inverts the whole ``n_fields x n_fields`` cell block, in which the pressure row is coupled to the
    velocity rows that make it invertible. It is also **not** the same factorization as scalar ILU(0):
    block storage pads each block to dense, so the factor keeps entries the point pattern discards.
    """

    def setUp(self, pc) -> None:
        """Convert the level operator to block storage and factorize it with zero block fill."""
        from petsc4py import PETSc

        operator = pc.getOperators()[0]
        # The destination Mat is passed explicitly: without it the conversion is done IN PLACE, which
        # retypes the multigrid's own level operator to block storage and then the aggregation cannot
        # read it at all ("no method creategraph for Mat of type seqbaij"). The failure surfaces at the
        # first apply rather than at the conversion, so it reads as a smoother fault instead of what it
        # is, which is this call quietly mutating something it does not own.
        self._blocked = operator.convert("baij", PETSc.Mat())
        self._sub = PETSc.PC().create()
        self._sub.setType("ilu")
        self._sub.setOperators(self._blocked)
        self._sub.setFactorLevels(0)
        self._sub.setUp()

    def apply(self, pc, x, y) -> None:
        """One block incomplete-LU back-solve."""
        self._sub.apply(x, y)

    def applyTranspose(self, pc, x, y) -> None:
        """The transpose back-solve, which the adjoint's transpose V-cycle needs."""
        self._sub.applyTranspose(x, y)

    def destroy(self, pc) -> None:
        """Release the converted operator and its factor."""
        self._sub.destroy()
        self._blocked.destroy()


@dataclasses.dataclass(frozen=True)
class Arm:
    """One preconditioner variant: what to build it from, and what to set on it.

    Attributes
    ----------
    label : str
        How the arm is reported.
    fill : int
        Incomplete-LU fill levels of the level smoother (``0`` = ILU(0), ``1`` = ILU(1)).
    sweeps : int
        Richardson sweeps of the level smoother per V-cycle visit.
    equilibrate : bool
        Apply the symmetric square-root-diagonal equilibration before the cell-major reorder. ``False``
        reorders only, leaving the row scales as the residual assembled them.
    reach : int
        Stencil reach of the materialized matrix the preconditioner is built from.
    options : dict
        PETSc options passed through the V-cycle's ``extra_options`` seam, without the prefix.
    """

    label: str
    fill: int = 0
    sweeps: int = SWEEPS
    equilibrate: bool = True
    reach: int = REACH
    options: dict = dataclasses.field(default_factory=dict)


#: The sweep. The first two arms are the controls the whole run is read against — the fill level known
#: to converge here, and the zero-fill one that does not — and every candidate differs from the second
#: in exactly one thing.
ARMS = [
    Arm("CONTROL ILU(1) x4", fill=1),
    Arm("CONTROL ILU(0) x4", fill=0),
    # The classical fix for an incomplete factorization breaking down on a non-M-matrix: perturb the
    # pivot rather than the operator. `nonzero` shifts a pivot that would be zero, `positive_definite`
    # shifts the whole diagonal until the factorization succeeds, `inblocks` shifts each diagonal
    # block (the matrix carries a block size of one cell's fields, which is what makes that meaningful).
    Arm("ILU(0) shift nonzero", options={"mg_levels_pc_factor_shift_type": "nonzero"}),
    Arm(
        "ILU(0) shift nonzero 1e-4",
        options={
            "mg_levels_pc_factor_shift_type": "nonzero",
            "mg_levels_pc_factor_shift_amount": 1e-4,
        },
    ),
    Arm(
        "ILU(0) shift nonzero 1e-2",
        options={
            "mg_levels_pc_factor_shift_type": "nonzero",
            "mg_levels_pc_factor_shift_amount": 1e-2,
        },
    ),
    Arm("ILU(0) shift posdef", options={"mg_levels_pc_factor_shift_type": "positive_definite"}),
    Arm("ILU(0) shift inblocks", options={"mg_levels_pc_factor_shift_type": "inblocks"}),
    # Sweeps: a weak smoother improves with more of them, an amplifying one does not.
    Arm("ILU(0) x1", sweeps=1),
    Arm("ILU(0) x8", sweeps=8),
    Arm("ILU(0) x16", sweeps=16),
    # Ordering: a zero-fill elimination keeps only the entries the ordering leaves in place, so the
    # order it eliminates in is part of the factorization rather than a detail of it.
    Arm("ILU(0) order rcm", options={"mg_levels_pc_factor_mat_ordering_type": "rcm"}),
    Arm("ILU(0) order nd", options={"mg_levels_pc_factor_mat_ordering_type": "nd"}),
    Arm("ILU(0) order qmd", options={"mg_levels_pc_factor_mat_ordering_type": "qmd"}),
    Arm("ILU(0) order rowlength", options={"mg_levels_pc_factor_mat_ordering_type": "rowlength"}),
    # Block elimination: never divide by a lone pressure diagonal (see :class:`BlockIlu0`).
    Arm(
        "block ILU(0) x4",
        options={
            "mg_levels_pc_type": "python",
            "mg_levels_pc_python_type": "ilu0_remedy_sweep.BlockIlu0",
        },
    ),
    Arm(
        "block ILU(0) x8",
        sweeps=8,
        options={
            "mg_levels_pc_type": "python",
            "mg_levels_pc_python_type": "ilu0_remedy_sweep.BlockIlu0",
        },
    ),
    # The conditioning transform itself.
    Arm("ILU(0) no equilibration", equilibrate=False),
    # A sparser preconditioner matrix. Legal because only the preconditioner is built from it.
    Arm("ILU(0) reach 3", reach=3),
    Arm("ILU(1) reach 3", fill=1, reach=3),
]


def load_state(coupled, path: Path | None):
    """The state every arm is measured at: a saved one, or the case's own self-start.

    The self-start is cached beside this file, because rebuilding it is the only part of a run that is
    not a measurement. It is packed through ``state_from_physical`` and **not** ``pack_state``: this case
    solves ``log(omega)``, so packing a physical ``omega`` directly would exponentiate it, and the
    residual is then silently non-finite while every field still reads finite -- after which every
    factorization fails in its own idiom and invites a confident wrong story about the method.
    """
    cache = CASE / "ilu0_remedy_state.npz"
    if path is None and cache.exists():
        path = cache
    if path is not None:
        state = jnp.asarray(np.load(path)["state"])
    else:
        flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
        state = coupled.state_from_physical(flow, k, omega)
        np.savez(cache, state=np.asarray(state))
        path = cache
    residual = coupled.residual(state)
    if not bool(jnp.all(jnp.isfinite(residual))):
        raise SystemExit(
            f"the residual at {path} is not finite; nothing measured here would mean anything."
        )
    print(f"state {path.name}: |R| {float(jnp.linalg.norm(residual)):.4e}", flush=True)
    return state


def jacobian(coupled, state, reach, cache):
    """The materialized field-major Jacobian at one stencil reach, plus its fixed pattern.

    Cached because the coloured probe is the dominant cost of a build and every arm at a given reach
    wants the identical matrix -- sharing it also makes it impossible for two arms to differ for any
    reason other than the options under test.

    ⚠️ **The cache is the CALLER's, keyed on reach alone, and that is why it cannot live here.** The
    matrix depends on the state as much as on the reach, so a module-level memo would serve the first
    state's Jacobian to every later one -- silently, since a matrix of the right shape and pattern is
    indistinguishable from the right one. A caller measuring several states holds one cache per state
    and drops it when it moves on, which also keeps a single materialized Jacobian in memory.

    Parameters
    ----------
    coupled : CoupledRANS
        The residual whose Jacobian is probed.
    state : jnp.ndarray
        The state to linearize at, shape ``(n_dofs,)``.
    reach : int
        Stencil reach of the coloured probe.
    cache : dict
        Caller-owned ``{reach: (matrix, structure)}``, valid for ``state`` only.
    """
    if reach not in cache:
        plan = _coupled_jacobian_plan(coupled, reach)
        structure = block_stencil_gather_map(plan)
        matrix = MonolithicAmgPreconditioner._materialize_jacobian(
            lambda v: _jacobian_matvec(coupled, state, v),
            plan,
            lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
            _PROBE_BATCH_SIZE,
            structure,
        )
        cache[reach] = (matrix, structure)
    return cache[reach]


def assemble(coupled, state, arm: Arm, shift: np.ndarray, n_fields: int, cache: dict):
    """The ``(matrix, scale, perm)`` triple the V-cycle is built from, for one arm at one shift.

    With ``arm.equilibrate`` the shipped transform runs: add the shift to the diagonal, symmetrically
    equilibrate by the square-root diagonal, reorder to cell-major. Without it, only the shift and the
    reorder run and the equilibration factor is one -- which is what lets the run say whether the
    zero-fill pivots are small because of the operator or because of the transform applied to it.
    """
    matrix, structure = jacobian(coupled, state, arm.reach, cache)
    if arm.equilibrate:
        return ShiftedCellMajorOperator(structure.indptr, structure.indices, n_fields).assemble(
            matrix.data, shift
        )
    perm = cell_major_permutation(matrix.shape[0] // n_fields, n_fields)
    shifted = matrix + sp.diags(shift)
    reordered = shifted[perm][:, perm].tocsr()
    reordered.sort_indices()
    return reordered, np.ones(matrix.shape[0]), perm


def smoother_factor_census(vcycle: AmgVCycle) -> dict:
    """Health of the factor the FINE-level smoother built: its pivots, and its condition estimate.

    Read off the built hierarchy rather than by factorizing a second time, so it reflects the arm's own
    options -- a shift arm's census is of the shifted factorization, which a standalone one would miss.
    PETSc stores the **reciprocal** pivot in the factor's diagonal (its triangular solve multiplies
    rather than divides), so it is inverted back here; the sign survives that, and the sign is the
    load-bearing part. A negative pivot on an operator equilibrated to a unit-magnitude diagonal is a
    factorization that is not an approximate inverse of anything.

    ``condest`` is the infinity-norm condition estimate of the factorization,
    ``||(LU)^-1 e||_inf`` with ``e`` the vector of all ones (Chapman, Saad and Wigton, 2000; Chow and
    Saad, 1997). It costs one forward-and-back substitution and needs no Krylov solve, which is the
    point of it: an incomplete factorization fails not by producing a small pivot but by producing
    triangular factors whose recurrences **grow**, and that growth is invisible to a pivot census while
    being exactly what this measures.

    Both numbers are **proxies for a cycle count, not substitutes for one**, and each is proxy in a
    known direction. This project records a case where the pivot census came back identical across arms
    whose cycle counts differed five-fold. A low ``condest`` is likewise necessary rather than
    sufficient -- the literature records diagonal perturbation and banded truncation each driving it
    down while making convergence worse. Both are reported beside the cycle count, never instead of it.
    """
    smoother = vcycle._pc.getMGSmoother(vcycle.levels - 1).getPC()
    try:
        factor = smoother.getFactorMatrix()
        reciprocal = factor.getDiagonal().getArray().copy()
    except Exception as failure:  # a refusal is a result about the factorization, not an obstacle
        return {"failed": f"{type(failure).__name__}: {failure}"}
    finite = reciprocal != 0.0
    pivots = np.where(finite, 1.0 / np.where(finite, reciprocal, 1.0), np.inf)
    try:
        ones = factor.createVecRight()
        ones.set(1.0)
        inverse_row_sums = factor.createVecRight()
        factor.solve(ones, inverse_row_sums)
        condest = float(np.abs(inverse_row_sums.getArray()).max())
        ones.destroy()
        inverse_row_sums.destroy()
    except Exception as failure:
        condest = float("nan")
        print(f"      condest unavailable: {type(failure).__name__}: {failure}", flush=True)
    return {
        "negative": int((pivots < 0.0).sum()),
        "min_magnitude": float(np.abs(pivots).min()),
        "median_magnitude": float(np.median(np.abs(pivots))),
        "condest": condest,
    }


def run_arm(coupled, state, arm: Arm, shift: np.ndarray, rhs, n_fields: int, cache: dict) -> None:
    """Build one V-cycle and solve the REAL system with it; report cycles and the TRUE residual."""
    t0 = time.time()
    try:
        cell_major, scale, perm = assemble(coupled, state, arm, shift, n_fields, cache)
        vcycle = AmgVCycle(
            cell_major,
            scale,
            perm,
            n_fields,
            smoother_fill_levels=arm.fill,
            smoother_sweeps=arm.sweeps,
            coarse_eq_limit=COARSE_EQ_LIMIT,
            extra_options=dict(arm.options) or None,
        )
    except Exception as failure:
        print(f"  {arm.label:<28} BUILD FAILED  {type(failure).__name__}: {failure}", flush=True)
        return
    build_s = time.time() - t0
    census = smoother_factor_census(vcycle)
    preconditioner = MonolithicAmgPreconditioner(vcycle)

    def operator(v):
        return _jacobian_matvec(coupled, state, v) + jnp.asarray(shift) * v

    t1 = time.time()
    if CONDEST_ONLY:
        # The cycle counts for these arms are already recorded; this pass adds only the condition
        # estimate, so it reads the factor and skips the solve rather than re-measuring the verdict.
        report = "cycles    -  TRUE rel        -"
    else:
        try:
            solution, raw = solve_linear(
                operator, rhs, STUDY_SOLVER, preconditioner=preconditioner.matvec(), throw=False
            )
            true = float(jnp.linalg.norm(operator(solution) - rhs) / jnp.linalg.norm(rhs))
            cycles = restart_cycles(int(raw))
            report = f"cycles {cycles:>4}  TRUE rel {true:.3e}"
        except Exception as failure:
            report = f"SOLVE FAILED  {type(failure).__name__}: {failure}"
    pivots = (
        census["failed"]
        if "failed" in census
        else (
            f"neg {census['negative']:>5}  min|piv| {census['min_magnitude']:.2e}  "
            f"med {census['median_magnitude']:.2e}  condest {census['condest']:.2e}"
        )
    )
    print(
        f"  {arm.label:<28} lvl {vcycle.levels} coarse {vcycle.coarse_size:>5}  "
        f"build {build_s:>5.1f}s  {report}  solve {time.time() - t1:>5.1f}s  {pivots}",
        flush=True,
    )
    preconditioner.destroy()  # one preconditioner in memory at a time
    del preconditioner, vcycle, cell_major
    gc.collect()


def main():
    """Run every arm at every shift, or the subset named on the command line.

    ``python3 ilu0_remedy_sweep.py`` runs the whole sweep at the cached self-start.
    ``python3 ilu0_remedy_sweep.py CONTROL block`` runs only the arms whose label contains one of those
    words -- keep a control in any subset, or the run has nothing to be read against.
    ``--state PATH`` measures at a saved state instead of the self-start.

    The same words may be given in ``ILU0_SWEEP_ARMS`` instead, because the case runner launches a
    script with no arguments: without this, asking it for a two-arm follow-up silently reruns the
    whole sweep, which is an hour spent re-measuring what is already recorded.
    """
    arguments = sys.argv[1:] or os.environ.get("ILU0_SWEEP_ARMS", "").split()
    state_path = None
    if "--state" in arguments:
        index = arguments.index("--state")
        state_path = Path(arguments[index + 1])
        del arguments[index : index + 2]
    arms = [a for a in ARMS if not arguments or any(word in a.label for word in arguments)]
    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    state = load_state(coupled, state_path)
    cache = {}  # this state's materialized Jacobians, by reach (see `jacobian`)
    base = _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS)
    rhs = -coupled.residual(state)
    for beta in BETAS:
        shift = _frozen_shift_diagonal(base, beta, state)
        print(
            f"\n{'=' * 118}\nbeta {beta} (operator AND V-cycle -- matched, no preconditioner-only "
            f"floor), reach {REACH} unless stated, |rhs| {float(jnp.linalg.norm(rhs)):.4e}",
            flush=True,
        )
        for arm in arms:
            run_arm(coupled, state, arm, shift, rhs, n_fields, cache)


if __name__ == "__main__":
    main()
