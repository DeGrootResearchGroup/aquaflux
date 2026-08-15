"""What the zero-fill ILU's PATTERN contains at zero shift, and the pivots it produces.

At **zero shift** -- the operator the implicit-function-theorem adjoint solves, and therefore the one behind
every ``jax.grad`` through a converged coupled solve -- the shipped monolithic ILU(0) V-cycle converges in
exactly one configuration on this case, and two independent changes each break it, in opposite directions.
Neither is visible from a march: the march's preconditioner is floored at a positive shift, so no march step
ever meets this operator.

The two levers act on the same thing, the set of positions the zero-fill incomplete factorization takes its
pattern from:

* **Shortening a probed column REMOVES positions.** An entry outside its column's own probing reach is
  written as an exact zero rather than gathered, and an assembly written as a sparse product or addition
  stores only entries whose *result* is nonzero -- so those positions are deleted and the factorization is
  handed a structurally weaker pattern.
* **Preserving the pattern ADDS positions that are exactly zero.** Writing the shift and the equilibration
  in place cannot change which entries exist, so every stored zero survives into the factorization. This
  makes the probing reach irrelevant to the assembled operator, which is a real property to want and which
  it genuinely delivers.

**What this harness measured (2026-08-12), and it is not what either of the obvious accounts predicted.**
Configuration as in the run banner: ``state-00067``, both betas zero, real right-hand side, restart 15 to
rtol 1e-8 on the TRUE residual, plain aggregation, ILU(0) x4, ``coarse_eq_limit`` 2000, pattern reach 3.

| column reach | shift / equilibration | nnz | stored zeros | cycles | TRUE rel |
|---|---|---|---|---|---|
| uniform 3 | prune / prune | 39.18M | 0 | 11 | 8.474e-11 |
| uniform 3 | preserve / preserve | 47.21M | 8.03M | 58 | 2.299e-02 |
| uniform 3 | preserve / prune | 39.18M | 0 | 11 | 8.474e-11 |
| 3/3/3/3/2/2 | prune / prune | 36.97M | 0 | 22 | 8.545e-11 |
| 3/3/3/3/2/2 | preserve / preserve | 47.21M | 10.24M | 58 | 2.299e-02 |

* **Only preservation breaks convergence.** Pruning at *either* stage reproduces the good operator
  bit-identically. Shortening merely doubles the cost (22 against 11) and still reaches the 1e-11 floor --
  under a loose march-solver stop its 4.779e-03 reads as a failure, but that is the stop, not divergence.
* **⚠️ AT THE FORWARD MARCH'S OPERATING POINT EVERY ARM TIES, which is why no march can reveal this.** The
  same sweep at ``state-00066`` (step-initial, operator beta 0.0096 with the V-cycle at the 0.05 floor --
  the shipped mismatch) gives **4 cycles at 1.435e-13 for all five arms** (1.444e-13 shortened). Neither the
  stored zeros nor the reach costs a single cycle there. Run that state to confirm a change is confined to
  the adjoint; run ``state-00067`` to see whether it is a change at all.
* **⚠️ IT IS NOT A PIVOT PATHOLOGY, AND A PIVOT CENSUS CANNOT SEE IT.** All four arms give an identical
  fine-level census: min |pivot| 1.546e-01, **zero** negative pivots, median 1.020. Nor is it the coarse
  space's size, identical at 2 levels and 1296 coarse equations throughout. Both were the natural
  diagnoses; both are measured blind here. (The ILU(1) result that fill "acquires negative pivots and
  diverges" as the shift falls does **not** transfer -- retaining fill can only make an incomplete
  factorization a closer approximation to the inverse, and the pivots stay healthy.)
* **What IS demonstrated:** PETSc genuinely keeps the stored zeros (they are counted in the ``Mat``'s and
  the factor's ``nz_used``) and the smoother's *action* differs because of them. **Why a denser incomplete
  factor is a worse smoother on this operator at zero shift is still open** -- the instrument for it is to
  compare the two V-cycles per level, or to degrade the coarse solve to ``jacobi`` in both arms and see
  whether the gap survives.

For each ``(column reach) x (shift spelling, equilibration spelling)`` arm, from one materialization per
reach, it reports:

* the **pattern**: stored entries, how many are exactly zero, and -- per ``(row field, column field)`` block
  -- both the stored count and the zero count. The per-block resolution is the point rather than decoration:
  a position matters in proportion to the entries it sits among, and the ``(p, p)`` block of a collocated
  incompressible saddle is near-zero by construction, so a column-relative measure cannot see damage that an
  entry-relative one can. Comparing the block grids across arms is what shows *which* couplings a short
  reach removed and which ones preservation added.
* the **ILU pivot census** on the equilibrated cell-major operator the level smoother actually factors: the
  smallest pivot magnitude and how many pivots are negative.
* the **hierarchy shape** (levels, coarse equations), which separates "the smoother broke" from "the coarse
  space moved" -- at plain aggregation the coarsening reads the pattern, so a changed pattern can move the
  aggregates as well as the factorization, and without this number the two are indistinguishable.
* the **true relative residual through GMRES** on the real right-hand side ``-R(state)``, which is the only
  measure that decides anything here. A preconditioned norm, a one-application contraction and a spectral
  radius have each produced a retracted verdict on this operator.

Both spellings are implemented **in this file** rather than taken from whichever version of the library is
checked out, so an arm is a property of the spelling and not of the working tree, and the run can be
repeated after either one changes. The arms are fingerprinted over their nonzero entries: within one reach
they are the same matrix presented with different patterns, and an arm comparison whose arms differed in
their *values* as well would measure nothing. Across reaches the fingerprints legitimately differ, because a
shortened column does not merely omit far entries -- a colouring is collision-free only for the pattern it
was built at, so a shortened column's near entries absorb the folded far ones.

The **monolithic** V-cycle is the arm, not the field split, and deliberately: this harness controls the
operator PETSc is handed end to end, whereas a split re-equilibrates each block through the library and
would inherit the working tree's spelling for that half -- making the arm a mixture of the thing under test
and the thing controlled for. Measure the split with ``field_split_probe.py``, which is built for it.

Run it through the blessed runner, which holds the machine awake and refuses a second concurrent case::

    BFS3D_PROBE_STATE=state-00067 \\
      validation/run_case.sh validation/bfs3d_openfoam/zero_pattern_pivots.py

``BFS3D_PROBE_STATE`` selects from this file's own :data:`STATES`; ``state-00067`` is the converged state at
zero shift (the adjoint's operator) and is the discriminating one. A step-initial state such as
``state-00066`` carries its own shift with the preconditioner floored, which is the forward march's own
operating point -- run that to establish whether the damage is confined to the zero-shift adjoint.

The reaches swept are named in :data:`REACHES` rather than read from ``compare.COLUMN_REACH``, because the
case default has moved twice and is currently a value that is only sound with the pattern preserved. An arm
here must say which reach it is, so that a later reader can tell whether it still applies.
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    AmgVCycle,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
)
from aquaflux.solve.frozen_operator import equilibration_scale  # noqa: E402
from aquaflux.solve import cell_major_permutation  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
)
from field_split_probe import FLOOR, RTOL, SOLVER, materialize, run_arm  # noqa: E402


class _State(NamedTuple):
    """One checkpoint this harness runs on.

    ``march_beta`` is the shift the *operator* is measured at (zero for the converged state, which is the
    whole point of it); ``checkpoint_shift`` is the shift the march wrote the file at, which is what
    identifies the file.
    """

    march_beta: float
    checkpoint_shift: float
    description: str


#: The states this harness runs on, with its **own** table rather than the neighbouring probe's, because
#: that one tracks whichever study is live and its entries come and go -- a harness whose states can be
#: rotated out from under it by an unrelated edit cannot be re-run to re-adjudicate its own findings.
#:
#: All three come from ONE march, and which march is part of the measurement: a 67-step Reynolds
#: continuation converging to ``|R|`` 3.586e-06 at mid-span ``x_r/h`` 8.361, under field split, the native
#: trailing inverse, a ``zerogradient`` k wall, positivity floor 1e-08, ILU(0) x4 on the saddle, plain
#: aggregation, ``coarse_eq_limit`` 2000, forward restart 15 and ``refresh_on_cycles`` 3. **A checkpoint set
#: is only usable with the bundle it was written under**; several of those defaults have moved within days
#: of each other, and a state carried over from before them is a different discrete problem.
STATES = {
    "state-00067": _State(
        0.0, 0.0064, "the converged state, |R| 3.586e-06 -- the ADJOINT's operator, at zero shift"
    ),
    "state-00066": _State(
        0.0096,
        0.0096,
        "step 27 of the target rung, step-initial -- the forward march's own operating point",
    ),
    "state-00065": _State(
        0.0144,
        0.0144,
        "step 26 of the target rung, step-initial -- a second cheap solve, higher shift",
    ),
}


def load_state(name: str) -> jnp.ndarray:
    """The captured state, REFUSING one that is not what :data:`STATES` says it is.

    **The shift check is not defensive programming, it is the fix for a real silent failure.** The
    checkpointer keeps only the last few files and numbers them from a counter that restarts with each
    march, so a later run *replaces* ``state-000NN`` with a completely different state under the same name.
    That has happened: a name documented as the converged zero-shift state came back holding a mid-march
    iterate at shift 0.98 from an abandoned run, and nothing complained -- the operator would have been
    built at this table's shift around a state that never had it, and reported as a measurement at the
    documented operating point.

    Compared loosely on purpose: the table records the shift to about four figures, so an exact comparison
    would reject a matching state, while what this must catch differs by orders of magnitude.
    """
    path = CASE / "checkpoints" / f"{name}.npz"
    if not path.exists():
        raise SystemExit(
            f"{name}: no such checkpoint. These are a rolling buffer (`BFS3D_CHECKPOINT_KEEP`, default 3) "
            f"and a later march will have rotated it away -- re-run the case to regenerate it, raising the "
            f"keep count if a study needs the whole trajectory.\n  present: "
            f"{sorted(p.stem for p in path.parent.glob('*.npz'))}"
        )
    data = np.load(path)
    if "shift" not in data:
        raise SystemExit(
            f"{name}: this checkpoint records no shift, so it cannot be identified. This harness needs an "
            "end-of-step checkpoint, not an inner-iterate dump."
        )
    recorded, expected = float(data["shift"]), STATES[name].checkpoint_shift
    if not np.isclose(recorded, expected, rtol=0.02, atol=1e-9):
        raise SystemExit(
            f"{name}: this checkpoint was written at shift {recorded:.6g}, but the table here describes it "
            f"as {expected:.6g}. The file has been overwritten by a later march, so it is NOT the state "
            "this entry documents. Re-run the case to regenerate it, or point the entry at a checkpoint "
            "that still matches."
        )
    print(
        f"{name}: end of step {int(data['step'])}, |R| {float(data['residual_norm']):.4e}, "
        f"march shift {recorded:.4f}",
        flush=True,
    )
    return jnp.asarray(data["state"])


#: Field names in the flat layout's block order, for the per-block census. The coupled state is FIELD-major
#: (dof ``(cell i, field f)`` sits at ``f * n_cells + i``), so a degree of freedom's field is
#: ``index // n_cells`` before the cell-major reorder and ``index % n_fields`` after it. Getting that
#: backwards censuses a transposed grid, which still looks entirely plausible.
FIELDS = ("u", "v", "w", "p", "k", "omega")

#: The probing reaches to sweep, as ``(label, column_reach)``. ``None`` probes every column at the pattern
#: reach and is the only configuration in which this operator's control converges at zero shift; the second
#: shortens the k and omega columns, which is proven on the march and measured here to fail at zero shift.
#: The pattern stays at reach 3 in both -- only which columns are probed to it changes.
REACHES = (
    ("uniform 3", None),
    ("3/3/3/3/2/2", (3, 3, 3, 3, 2, 2)),
)


def _shift_pruning(jacobian: sp.csr_matrix, shift: np.ndarray) -> sp.csr_matrix:
    """``J + diag(shift)`` as a sparse ADDITION -- which stores only entries whose result is nonzero.

    Every explicit zero is therefore deleted, including at zero shift, where the addition changes no value
    at all and its entire effect is the pruning.
    """
    return (jacobian + sp.diags(np.asarray(shift, dtype=np.float64))).tocsr()


def _shift_preserving(jacobian: sp.csr_matrix, shift: np.ndarray) -> sp.csr_matrix:
    """``J + diag(shift)`` as an in-place diagonal assignment -- which cannot change the pattern.

    Only the diagonal is touched, so an off-diagonal position survives whatever its value; a diagonal the
    pattern lacks is created, matching the sparse addition's semantics.
    """
    shifted = jacobian.tocsr().copy()
    shifted.setdiag(shifted.diagonal() + np.asarray(shift, dtype=np.float64))
    return shifted


def _equilibrate_pruning(a: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """``D A D`` as two sparse PRODUCTS -- which likewise store only nonzero results."""
    a = a.tocsr()
    scale = equilibration_scale(a.diagonal())
    d = sp.diags(scale)
    return (d @ a @ d).tocsr(), scale


def _equilibrate_preserving(a: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """``D A D`` by scaling the stored values in place -- pattern preserved entry for entry."""
    scaled = a.tocsr().copy()
    scale = equilibration_scale(scaled.diagonal())
    scaled.data *= np.repeat(scale, np.diff(scaled.indptr))
    scaled.data *= scale[scaled.indices]
    return scaled, scale


def _assemble_library(jacobian: sp.csr_matrix, shift: np.ndarray, n_fields: int):
    """The WORKING TREE's own path, in whatever spelling it currently uses.

    The control on this harness itself. The local spellings above are a reimplementation, and a
    reimplementation can be unfaithful; if this arm disagrees with the local arm that is supposed to match
    it, then what separates the local arms is not the prune/preserve choice but something else in the
    library, and every conclusion drawn from them is about the wrong thing. The fingerprints are what
    settle it, because they compare the arms' **values** rather than only their patterns.
    """
    from aquaflux.solve import MonolithicAmgPreconditioner
    from aquaflux.solve import equilibrate_cell_major

    return equilibrate_cell_major(MonolithicAmgPreconditioner._shifted(jacobian, shift), n_fields)


SHIFTS = {"prune": _shift_pruning, "preserve": _shift_preserving}
EQUILIBRATIONS = {"prune": _equilibrate_pruning, "preserve": _equilibrate_preserving}
#: The spelling that means "ask the library", handled by :func:`_assemble_library`.
LIBRARY = ("library", "library")

#: ``(shift spelling, equilibration spelling)`` per reach. ``prune/prune`` is the pre-existing operator and
#: ``preserve/preserve`` the pattern-preserving one. ``preserve/prune`` is a **candidate fix** -- keep the
#: pattern through the shift and prune exact zeros before the factorization -- and is run at the first reach
#: only, where it also serves as the check that the two stages compose independently rather than interacting.
SPELLINGS = (("prune", "prune"), ("preserve", "preserve"), ("preserve", "prune"))
FIRST_REACH_ONLY = frozenset({("preserve", "prune")})


def assemble(jacobian, shift, n_fields, shift_spelling, equilibration_spelling):
    """The equilibrated cell-major operator PETSc is handed, under one spelling of each stage.

    Mirrors ``equilibrate_cell_major`` -- equilibrate, permute to cell-major, sort the column indices
    (PETSc's AIJ format requires them ascending, and the permutation leaves them out of order) -- with the
    prune/preserve choice at each stage made here rather than by the library version in the working tree.
    """
    if (shift_spelling, equilibration_spelling) == LIBRARY:
        return _assemble_library(jacobian, shift, n_fields)
    shifted = SHIFTS[shift_spelling](jacobian, shift)
    equilibrated, scale = EQUILIBRATIONS[equilibration_spelling](shifted)
    del shifted
    gc.collect()
    perm = cell_major_permutation(equilibrated.shape[0] // n_fields, n_fields)
    cell_major = equilibrated[perm][:, perm].tocsr()
    del equilibrated
    gc.collect()
    cell_major.sort_indices()
    return cell_major, scale, perm


def pattern_census(cell_major: sp.csr_matrix, n_fields: int) -> dict:
    """Stored entries, how many are exactly zero, and both counts per ``(row field, column field)`` block.

    ``cell_major`` is cell-major, so a degree of freedom's field is ``index % n_fields``.
    """
    rows = np.repeat(np.arange(cell_major.shape[0]), np.diff(cell_major.indptr))
    cols = cell_major.indices
    zero = cell_major.data == 0.0
    row_field, col_field = rows % n_fields, cols % n_fields
    stored_blocks = np.zeros((n_fields, n_fields), dtype=np.int64)
    zero_blocks = np.zeros((n_fields, n_fields), dtype=np.int64)
    np.add.at(stored_blocks, (row_field, col_field), 1)
    np.add.at(zero_blocks, (row_field[zero], col_field[zero]), 1)
    # A row with no stored diagonal is a structurally missing pivot rather than a small one, which is a
    # different failure and worth distinguishing.
    stored_diagonal = np.zeros(cell_major.shape[0], dtype=bool)
    stored_diagonal[rows[rows == cols]] = True
    return {
        "nnz": int(cell_major.nnz),
        "zeros": int(zero.sum()),
        "zeros_on_diagonal": int((zero & (rows == cols)).sum()),
        "rows_without_diagonal": int((~stored_diagonal).sum()),
        "stored_blocks": stored_blocks,
        "zero_blocks": zero_blocks,
    }


def fingerprint(cell_major: sp.csr_matrix) -> str:
    """A digest of the NONZERO entries as ``(row, column, value)`` triples.

    Within one reach the arms must be the same matrix in different patterns: if they differ in their values
    as well, whatever separates them is not the pattern and the comparison says nothing. Comparing digests
    rather than matrices is what lets that be asserted without holding two three-dimensional coupled
    operators live at once, which is enough to exhaust this machine.
    """
    rows = np.repeat(np.arange(cell_major.shape[0]), np.diff(cell_major.indptr))
    keep = cell_major.data != 0.0
    digest = hashlib.sha256()
    for part in (rows[keep], cell_major.indices[keep], cell_major.data[keep]):
        digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()[:16]


def ilu_pivots(cell_major: sp.csr_matrix, n_fields: int, fill_levels: int) -> dict:
    """Pivot census of the ILU(``fill_levels``) factorization of ``cell_major``, via PETSc.

    The level smoother of the shipped V-cycle is a stationary incomplete-LU sweep, so this is the
    factorization the smoother applies on the fine grid. PETSc stores the **reciprocal** pivot in the
    factor's diagonal (its triangular solve multiplies rather than divides), so the diagonal is inverted
    back before reporting. The sign survives that inversion, and the sign is the load-bearing part: a
    negative pivot on an operator equilibrated to a unit-magnitude diagonal is a factorization that is not
    an approximate inverse of anything.

    Returns ``{"failed": ...}`` if PETSc refuses -- a zero pivot is a result about the pattern, not an error
    to be worked around.
    """
    from petsc4py import PETSc

    mat = PETSc.Mat().createAIJWithArrays(
        size=cell_major.shape,
        csr=(
            cell_major.indptr.astype(PETSc.IntType),
            cell_major.indices.astype(PETSc.IntType),
            cell_major.data.astype(PETSc.ScalarType),
        ),
    )
    mat.setBlockSize(n_fields)
    mat.assemble()
    pc = PETSc.PC().create()
    try:
        pc.setOperators(mat)
        pc.setType("ilu")
        pc.setFactorLevels(fill_levels)
        pc.setUp()
        factor = pc.getFactorMatrix()
        reciprocal = factor.getDiagonal().getArray().copy()
        # Fill and growth, because the pivots alone do not describe the factor. A retained exactly-zero
        # position is a slot the elimination may deposit fill into, and fill that the pruned pattern would
        # have discarded changes the smoother's action even when every pivot stays healthy -- so how much
        # the factor grew, and how large its largest entry became, is the thing to compare next.
        factor_nnz = int(factor.getInfo()["nz_used"])
        try:
            factor_max = float(factor.norm(PETSc.NormType.INFINITY))
        except Exception:  # MatNorm is not supported for every factored format
            factor_max = float("nan")
    except Exception as failure:  # a zero pivot is the answer, not an obstacle
        return {"failed": f"{type(failure).__name__}: {failure}"}
    finally:
        pc.destroy()
        mat.destroy()
    # A reciprocal of exactly zero would be an infinite pivot, which does not occur here; guard anyway, so
    # the census cannot itself divide by zero and report a NaN as though it were a measurement.
    finite = reciprocal != 0.0
    pivots = np.where(finite, 1.0 / np.where(finite, reciprocal, 1.0), np.inf)
    magnitude = np.abs(pivots)
    return {
        "min_magnitude": float(magnitude.min()),
        "negative": int((pivots < 0.0).sum()),
        "below_1e_6": int((magnitude < 1e-6).sum()),
        "below_1e_3": int((magnitude < 1e-3).sum()),
        "median_magnitude": float(np.median(magnitude)),
        "factor_nnz": factor_nnz,
        "factor_max": factor_max,
    }


def _grid(title: str, blocks: np.ndarray, n_fields: int) -> None:
    print(f"     {title}", flush=True)
    print("       " + "row\\col".ljust(9) + "".join(f"{f:>11}" for f in FIELDS[:n_fields]))
    for i, name in enumerate(FIELDS[:n_fields]):
        print(
            "       " + name.ljust(9) + "".join(f"{blocks[i, j]:>11}" for j in range(n_fields)),
            flush=True,
        )


def _report_pattern(census: dict, n_fields: int) -> None:
    print(
        f"     nnz {census['nnz'] / 1e6:>6.2f}M   exact zeros {census['zeros'] / 1e6:>6.2f}M   "
        f"on the diagonal {census['zeros_on_diagonal']:>7}   rows with no diagonal "
        f"{census['rows_without_diagonal']:>7}",
        flush=True,
    )
    _grid("stored entries by (row field, column field):", census["stored_blocks"], n_fields)
    if census["zeros"]:
        _grid("of which EXACTLY ZERO:", census["zero_blocks"], n_fields)


def _report_pivots(census: dict, fill_levels: int) -> None:
    if "failed" in census:
        print(f"     ILU({fill_levels}) REFUSED  {census['failed']}", flush=True)
        return
    print(
        f"     ILU({fill_levels}) pivots: min |pivot| {census['min_magnitude']:.3e}   negative "
        f"{census['negative']:>7}   below 1e-6 {census['below_1e_6']:>7}   below 1e-3 "
        f"{census['below_1e_3']:>7}   median {census['median_magnitude']:.3e}",
        flush=True,
    )
    # At zero fill the factor's pattern IS the matrix's, so its nnz restates the census above and only
    # becomes informative at ILU(k>0). The row-sum norm is reported when PETSc supplies it -- ``MatNorm`` is
    # not implemented for every factored format, and an unavailable number is said to be unavailable rather
    # than printed as a NaN that reads like a measurement.
    growth = (
        f"   max row sum {census['factor_max']:.3e}"
        if np.isfinite(census["factor_max"])
        else "   max row sum unavailable for this factored format"
    )
    print(
        f"     ILU({fill_levels}) factor: nnz {census['factor_nnz'] / 1e6:>6.2f}M{growth}",
        flush=True,
    )


def _run(label, cell_major, scale, perm, n_fields, coupled, state, rhs, op_shift):
    """Build the monolithic V-cycle on this arm's operator and solve the real system with it."""
    preconditioner = None
    try:
        started = time.time()
        preconditioner = MonolithicAmgPreconditioner(
            AmgVCycle(
                cell_major,
                scale,
                perm,
                n_fields,
                smoother_fill_levels=compare.FILL_LEVELS,
                smoother_sweeps=compare.SWEEPS,
                coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            )
        )
        print(
            f"     hierarchy: {preconditioner.factors.levels} levels, "
            f"{preconditioner.factors.coarse_size} coarse equations",
            flush=True,
        )
        return run_arm(
            label, preconditioner, time.time() - started, coupled, state, rhs, op_shift, SOLVER
        )
    except Exception as failure:
        print(f"    {label:<44} FAILED  {type(failure).__name__}: {failure}", flush=True)
        return None
    finally:
        if preconditioner is not None:
            preconditioner.factors.destroy()
        del preconditioner
        gc.collect()


def main():
    name = os.environ.get("BFS3D_PROBE_STATE") or (
        sys.argv[1] if len(sys.argv) > 1 else "state-00067"
    )
    if name not in STATES:
        raise SystemExit(f"unknown state {name!r}; known: {list(STATES)}")
    # By NAME, not by position: this record has grown a field before, and a positional unpack
    # turns that into a probe that cannot start at all.
    entry = STATES[name]
    march_beta, description = entry.march_beta, entry.description
    # The V-cycle is built at the floor while the operator keeps the march's own beta -- the shipped
    # mismatch. At the converged state's zero shift there is no floor: the adjoint has none, and flooring it
    # here would measure a preconditioner no gradient ever uses.
    pc_beta = max(march_beta, FLOOR) if march_beta > 0 else 0.0

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    n_dofs = n_fields * coupled.layout.n_cells
    print(
        f"{'=' * 100}\nzero-shift ILU({compare.FILL_LEVELS}) pattern census and pivots, monolithic "
        f"V-cycle\nbundle: plain aggregation, ILU({compare.FILL_LEVELS}) x{compare.SWEEPS}, "
        f"coarse_eq_limit {compare.COARSE_EQ_LIMIT}, pattern reach 3, GMRES restart 15 to rtol "
        f"{RTOL:.0e} on the TRUE residual\ncolumn reaches swept: "
        f"{', '.join(label for label, _ in REACHES)}\n"
        f"operator beta {march_beta}, preconditioner beta {pc_beta}\n{'=' * 100}",
        flush=True,
    )
    state = load_state(name)
    print(f"  {description}", flush=True)

    base = _coupled_shift_policy(coupled, state, "twolevel")
    rhs = -coupled.residual(state)
    op_shift = _frozen_shift_diagonal(base, march_beta, state) if march_beta > 0 else 0.0
    shift = _frozen_shift_diagonal(base, pc_beta, state) if pc_beta > 0 else np.zeros(n_dofs)
    print(f"  right-hand side |R| {float(jnp.linalg.norm(rhs)):.4e}", flush=True)

    # One reach's Jacobian and one arm's operator live at a time: the Jacobian is ~0.6 GB, each cell-major
    # operator another ~0.6 GB, PETSc holds its own copy plus factors, and this machine has been exhausted
    # by holding two three-dimensional coupled operators at once.
    results, fingerprints = {}, {}
    for reach_index, (reach_label, column_reach) in enumerate(REACHES):
        print(f"\n{'-' * 100}\n  COLUMN REACH: {reach_label}\n{'-' * 100}", flush=True)
        plan = _coupled_jacobian_plan(coupled, 3, column_reach)
        structure = block_stencil_gather_map(plan)
        jacobian = materialize(coupled, state, plan, structure, n_fields)
        for spelling in SPELLINGS:
            if reach_index and spelling in FIRST_REACH_ONLY:
                continue
            shift_spelling, equilibration_spelling = spelling
            label = f"{reach_label}, {shift_spelling}/{equilibration_spelling}"
            print(
                f"\n  -- shift {shift_spelling}, equilibration {equilibration_spelling}", flush=True
            )
            started = time.time()
            cell_major, scale, perm = assemble(
                jacobian, shift, n_fields, shift_spelling, equilibration_spelling
            )
            print(f"     assembled in {time.time() - started:.0f}s", flush=True)
            _report_pattern(pattern_census(cell_major, n_fields), n_fields)
            fingerprints[label] = fingerprint(cell_major)
            started = time.time()
            _report_pivots(
                ilu_pivots(cell_major, n_fields, compare.FILL_LEVELS), compare.FILL_LEVELS
            )
            print(f"     factored in {time.time() - started:.0f}s", flush=True)
            if os.environ.get("BFS3D_CENSUS_ONLY"):
                # The pattern and the factorization, without the solves. The failing arms run to the
                # restart cap, which is most of this harness's wall clock, so a question answered by the
                # census alone should not pay for them.
                del cell_major, scale, perm
                gc.collect()
                continue
            results[label] = _run(
                f"ILU({compare.FILL_LEVELS}), {label}",
                cell_major,
                scale,
                perm,
                n_fields,
                coupled,
                state,
                rhs,
                op_shift,
            )
            del cell_major, scale, perm
            gc.collect()
        del jacobian, plan, structure
        gc.collect()

    print(
        "\n  -- within one reach the arms must be the SAME matrix in different patterns; across\n"
        "     reaches they legitimately differ, because a shortened column folds its far entries\n"
        "     onto its near ones rather than merely omitting them",
        flush=True,
    )
    for label, digest in fingerprints.items():
        print(f"     {label:<40} {digest}", flush=True)

    print(f"\n  -- summary at {name}, preconditioner beta {pc_beta}", flush=True)
    for label, outcome in results.items():
        if outcome is None:
            print(f"     {label:<40} failed to build", flush=True)
        else:
            cycles, true = outcome
            print(f"     {label:<40} cycles {cycles:>4}   TRUE rel {true:.3e}", flush=True)


if __name__ == "__main__":
    main()
