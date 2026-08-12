"""How much of the coupled Jacobian lives in the outermost stencil ring, block by block.

The coloured probe costs one directional derivative per (colour, field), and the colour count grows
steeply with the stencil reach the probe covers -- on this mesh 11 colours at reach one, 39 at reach
two, 94 at reach three. So the reach is the largest single lever on probe cost, and the question this
answers is what it would cost in *accuracy*: how much of the matrix sits at graph distance three,
which is exactly the part a reach-two probe cannot see.

Two facts make the question worth asking per block rather than once:

* A block-triangular field split preconditions ``[u, v, w, p]`` and ``[k, omega]`` with separate
  hierarchies, so the two halves could be probed at different reaches. The velocity-pressure block's
  reach is forced by the pressure-velocity interpolation, which couples pressure to the
  neighbour-of-neighbour ring; the scalar transport pair is first-order upwind plus a
  non-orthogonal diffusion correction, whose natural stencil is shorter.
* The split applies only one of the two off-diagonal blocks -- with the flow block leading, it uses
  ``d R_turb / d flow`` and never ``d R_flow / d turb``. Accuracy in the unused block is free, so the
  turbulence *columns* need only be accurate in the turbulence *rows*.

**A shorter reach is not simply a truncation, and the distinction decides how to read the numbers.**
A colouring is collision-free only for the pattern it was built from, so two cells sharing a reach-two
colour may still both couple to a common row at distance three. The probe response for that row is
then the *sum* of both couplings, and the de-compression charges the whole sum to the one column
inside the reach-two pattern. So a reach-two probe does not drop the distance-three entries, it folds
them onto distance-two positions. Both the dropped mass and the folded mass are bounded by the shell
norm reported here, which is why this one measurement settles both.

Run it against a state whose provenance you can name, and quote the state with any result -- the shell
magnitude is a property of the flow, not of the mesh alone.

Usage, one state per run (each materializes a Jacobian of some gigabytes)::

    python3 -u validation/bfs3d_openfoam/probe_reach_audit.py state-00045
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    ColumnProbePlan,
    block_stencil_colouring,
    block_stencil_gather_map,
)
from aquaflux.solve.amg_preconditioner import MonolithicAmgPreconditioner  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _jacobian_matvec,
)

#: Field order of the coupled state, and the two groups a block-triangular split cuts it into.
FIELDS = ("u", "v", "w", "p", "k", "omega")
LEADING = (0, 1, 2, 3)
TRAILING = (4, 5)


def column_reach_ladder(jacobian, patterns, widest, n, n_fields):
    """Per column field, the share of that column block's norm lying outside each candidate reach.

    The probe costs one directional derivative per (colour, **column** field), and the colour count
    falls steeply as the reach drops, so the reach that matters is a per-column-field property: a
    column whose couplings all lie inside reach ``r`` can be probed with an ``r``-colouring and still
    assembled into the widest pattern, its outer positions being the explicit zeros they already are.

    A column with **no** mass outside ``r`` is exact under an ``r``-probe, not merely close: the folding
    described in the module docstring adds a distance-three coupling to a distance-two entry, and there
    is nothing there to add.

    Returns
    -------
    dict
        ``column field -> {reach: fraction of the column's norm outside that reach}``.
    """
    out = {}
    for b in range(n_fields):
        column = jacobian[:, b * n : (b + 1) * n]
        total = float(np.linalg.norm(column.data))
        shares = {}
        for reach, pattern in patterns.items():
            if reach >= widest:
                continue
            outside = (patterns[widest] - pattern).tocsr()
            outside.eliminate_zeros()
            outside.data[:] = 1.0
            # The column stacks every row field, so the mask repeats down the field blocks.
            stacked = sp.vstack([outside] * n_fields, format="csr")
            ring = float(np.linalg.norm(column.multiply(stacked).data))
            shares[reach] = ring / total if total else 0.0
        out[b] = shares
    return out


def cell_patterns(owner, nb, n, reaches=(2, 3)):
    """Cell-block sparsity patterns at each reach, as boolean CSR matrices."""
    out = {}
    for reach in reaches:
        colouring = block_stencil_colouring(owner, nb, n, reach)
        out[reach] = sp.csr_matrix(
            (
                np.ones(len(colouring.pattern_rows), dtype=np.float64),
                (colouring.pattern_rows, colouring.pattern_cols),
            ),
            shape=(n, n),
        )
    return out


def shell_mask(patterns, inner=2, outer=3):
    """The positions the ``outer`` pattern has and the ``inner`` one does not -- the outermost ring."""
    shell = (patterns[outer] - patterns[inner]).tocsr()
    shell.eliminate_zeros()
    shell.data[:] = 1.0
    return shell


def block_shell_fractions(jacobian, shell, n, n_fields):
    """For every field pair, the share of the sub-block's norm carried by the outermost ring.

    Returns
    -------
    dict
        ``(row_field, col_field) -> (block_norm, shell_norm, fraction)``.
    """
    out = {}
    for a in range(n_fields):
        rows = jacobian[a * n : (a + 1) * n]
        for b in range(n_fields):
            sub = rows[:, b * n : (b + 1) * n]
            # A sparse matrix's Frobenius norm is the 2-norm of its stored values.
            total = float(np.linalg.norm(sub.data))
            ring = float(np.linalg.norm(sub.multiply(shell).data))
            out[(a, b)] = (total, ring, ring / total if total else 0.0)
    return out


def group_norm(fractions, row_fields, col_fields):
    """Frobenius norms combine in quadrature, so a group's shell share is the root of the sums."""
    total = sum(fractions[(a, b)][0] ** 2 for a in row_fields for b in col_fields)
    ring = sum(fractions[(a, b)][1] ** 2 for a in row_fields for b in col_fields)
    return np.sqrt(total), np.sqrt(ring), (np.sqrt(ring / total) if total else 0.0)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    name = sys.argv[1]

    print(f"[case] building {name}", flush=True)
    case = compare.build_case()
    coupled = case["coupled"]
    mesh = coupled.momentum.mesh
    n = mesh.n_cells
    n_fields = coupled.layout.dim + 3

    path = CASE / "checkpoints" / f"{name}.npz"
    if not path.exists():
        raise SystemExit(
            f"{name}: no such checkpoint (present: "
            f"{sorted(p.stem for p in path.parent.glob('*.npz'))})"
        )
    # Checkpoint names come from a per-run counter over a rolling buffer, so a later march REPLACES a
    # given name with an unrelated state. Print the provenance every run and quote it with the result;
    # a shell fraction is a property of the flow, so a result named only by a file name says nothing.
    with np.load(path) as data:
        state = np.asarray(data["state"])
        if "attempt" in data.files:
            detail = (
                f"inner iterate, attempt {int(data['attempt'])} inner {int(data['inner'])},"
                f" {int(data['cycles'])} cycles at alpha {float(data['alpha']):.2e}"
            )
        else:
            detail = (
                f"end of step {int(data['step'])}, |R| {float(data['residual_norm']):.4e},"
                f" march shift {float(data['shift']):.6g}"
            )
    print(f"[state] {name}: {detail}", flush=True)

    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)

    print("[graph] building reach-1, reach-2 and reach-3 patterns", flush=True)
    patterns = cell_patterns(owner, nb, n, reaches=(1, 2, 3))
    shell = shell_mask(patterns)
    colours = {}
    for reach, pattern in patterns.items():
        colours[reach] = block_stencil_colouring(owner, nb, n, reach).n_colours
        print(
            f"  reach {reach}: {pattern.nnz / 1e6:.2f}M cell-blocks, {colours[reach]} colours"
            f" -> {colours[reach]} probes per column field",
            flush=True,
        )
    print(f"  distance-3 shell: {shell.nnz / 1e6:.2f}M cell-blocks", flush=True)

    colouring = block_stencil_colouring(owner, nb, n, 3)
    structure = block_stencil_gather_map(ColumnProbePlan.uniform(colouring, n_fields))
    started = time.time()
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        ColumnProbePlan.uniform(colouring, n_fields),
        lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
        _PROBE_BATCH_SIZE,
        structure,
    )
    print(
        f"[probe] {jacobian.shape[0]} dofs, {jacobian.nnz / 1e6:.1f}M nnz"
        f" in {time.time() - started:.0f}s",
        flush=True,
    )

    fractions = block_shell_fractions(jacobian, shell, n, n_fields)

    print("\n  shell share of each field-pair block (row field x column field)", flush=True)
    header = "        " + "".join(f"{f:>12s}" for f in FIELDS)
    print(header, flush=True)
    for a in range(n_fields):
        cells = "".join(f"{fractions[(a, b)][2]:12.2e}" for b in range(n_fields))
        print(f"  {FIELDS[a]:>5s} {cells}", flush=True)

    print("\n  the blocks a flow-first block-triangular split actually applies", flush=True)
    for label, rows, cols in (
        ("leading  [uvwp]x[uvwp]", LEADING, LEADING),
        ("trailing [k,om]x[k,om]", TRAILING, TRAILING),
        ("coupling [k,om]x[uvwp]", TRAILING, LEADING),
        ("UNUSED   [uvwp]x[k,om]", LEADING, TRAILING),
    ):
        total, ring, frac = group_norm(fractions, rows, cols)
        print(f"    {label}:  ||A||={total:.4e}  shell={ring:.4e}  share={frac:.3e}", flush=True)

    # The probe is charged per (colour, COLUMN field), so the reach that matters is a per-column-field
    # property. A column with nothing outside reach r can be probed with an r-colouring and assembled
    # into the reach-3 pattern unchanged.
    ladder = column_reach_ladder(jacobian, patterns, 3, n, n_fields)
    print("\n  share of each COLUMN's norm lying outside a candidate reach", flush=True)
    print(f"  {'column':>8s} {'outside r=1':>14s} {'outside r=2':>14s}   minimum reach", flush=True)
    tolerance = 1e-12  # at or below this the column is exactly representable, not merely close
    chosen = {}
    for b in range(n_fields):
        outside = ladder[b]
        chosen[b] = 1 if outside[1] <= tolerance else (2 if outside[2] <= tolerance else 3)
        print(
            f"  {FIELDS[b]:>8s} {outside[1]:14.2e} {outside[2]:14.2e}   {chosen[b]}",
            flush=True,
        )

    shipped = colours[3] * n_fields
    tuned = sum(colours[chosen[b]] for b in range(n_fields))
    print(
        f"\n  probes: {shipped} at a uniform reach 3, {tuned} at the per-column minimum"
        f" ({100 * (1 - tuned / shipped):.0f}% fewer)",
        flush=True,
    )


if __name__ == "__main__":
    main()
