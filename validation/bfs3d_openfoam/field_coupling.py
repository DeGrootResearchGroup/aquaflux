"""How strongly is each field coupled to each other field in the coupled Jacobian?

A block-triangular field split (``solve/field_split.py``) partitions the six fields into a leading and
a trailing group, gives each its own multigrid hierarchy, and keeps **one** triangle of the coupling
between them exactly. The other triangle is dropped. So the question "should these two fields be split
apart?" has two halves, and this answers the first one cheaply:

1. **How much coupling would the split discard**, relative to the diagonal blocks it would preserve?
2. How much *cost* would it save -- which is the nonzero share of the blocks that leave the hierarchy.

Both come from one materialization. The matrix is symmetrically equilibrated by its own square-root
diagonal first, which is what the preconditioner is built on and what makes a block norm comparable
across blocks: on the raw Jacobian the six fields differ in units and scale by many orders, so ``k``'s
rows would look negligible beside the momentum rows for reasons that have nothing to do with coupling.

**This measures the operator, not the preconditioner's performance.** A strong coupling is a reason to
expect a split to hurt and a weak one is a reason to expect it to be free, but neither predicts the
wall-clock outcome -- two smaller V-cycles can apply cheaply enough to win while taking *more* Krylov
cycles, which is exactly what the shipped flow/turbulence split does. Only a whole march settles that.

Usage -- one state per run, since each materializes a Jacobian of some gigabytes::

    python3 -u validation/bfs3d_openfoam/field_coupling.py state-00069
    BFS3D_PROBE_STATE=state-00069 validation/run_case.sh validation/bfs3d_openfoam/field_coupling.py

The environment spelling exists because the case launcher takes only a script path -- a second
positional argument would be read as the script -- so it is the only way to reach this from the
launcher that holds the machine awake and refuses a second concurrent run. It is the same
``BFS3D_PROBE_STATE`` the field-split probe reads, so one export selects the state for both.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parents[1]))

import compare  # noqa: E402
from aquaflux.solve import block_stencil_gather_map  # noqa: E402
from aquaflux.turbulence.coupled import _coupled_jacobian_plan  # noqa: E402
from field_split_probe import STATES, load_state, materialize  # noqa: E402

#: The field order of the flat coupled state, and the order the blocks are reported in.
NAMES_3D = ("u", "v", "w", "p", "k", "omega")


def equilibrate(a: sp.csr_matrix) -> sp.csr_matrix:
    """``D^-1/2 A D^-1/2`` with ``D`` the absolute diagonal -- the operator the V-cycle is built on.

    Unit diagonal, so an off-diagonal block's norm reads directly as a coupling strength relative to the
    equations it couples into. A zero diagonal entry is left unscaled rather than dividing by zero; the
    coupled operator has none, but a silent ``inf`` would poison every norm downstream of it.
    """
    diagonal = np.abs(a.diagonal())
    scale = np.where(diagonal > 0, 1.0 / np.sqrt(np.where(diagonal > 0, diagonal, 1.0)), 1.0)
    scaling = sp.diags(scale)
    return (scaling @ a @ scaling).tocsr()


def block_matrix(a: sp.csr_matrix, n_cells: int, n_fields: int) -> tuple[np.ndarray, np.ndarray]:
    """The per-field-block Frobenius norms and nonzero counts.

    The coupled state is field-major -- degree of freedom ``(cell i, field f)`` sits at
    ``f * n_cells + i`` -- so each field block is a contiguous square submatrix and the partition is a
    slicing rather than a gather.

    Parameters
    ----------
    a : scipy.sparse.csr_matrix
        The equilibrated field-major operator, shape ``(n_fields * n_cells,) * 2``.
    n_cells, n_fields : int
        The partition's shape.

    Returns
    -------
    tuple of numpy.ndarray
        ``(norms, nnz)``, each shape ``(n_fields, n_fields)``, indexed ``[equation, variable]``.
    """
    norms = np.zeros((n_fields, n_fields))
    counts = np.zeros((n_fields, n_fields), dtype=np.int64)
    for row in range(n_fields):
        strip = a[row * n_cells : (row + 1) * n_cells, :].tocsc()
        for col in range(n_fields):
            block = strip[:, col * n_cells : (col + 1) * n_cells]
            norms[row, col] = sp.linalg.norm(block)
            counts[row, col] = block.nnz
    return norms, counts


def report(norms: np.ndarray, counts: np.ndarray, names: tuple[str, ...]) -> None:
    """Print the block grids, then the two split questions each partition raises."""
    width = max(len(n) for n in names) + 2
    for title, grid, fmt in (
        ("Frobenius norm of each block (rows = equation, cols = variable)", norms, "{:>9.3g}"),
        ("nonzeros per block, millions", counts / 1e6, "{:>9.2f}"),
    ):
        print(f"\n{title}")
        print(" " * width + "".join(f"{n:>10}" for n in names))
        for i, name in enumerate(names):
            cells = "".join(fmt.format(grid[i, j]) for j in range(len(names)))
            print(f"{name:>{width}}" + cells)

    print(f"\ntotal {counts.sum() / 1e6:.1f}M nonzeros")
    # Each candidate names the fields on each side of the split and the fields the question is asked
    # WITHIN. The k/omega question is asked inside the trailing group of the shipped split, where the
    # flow fields are already in a hierarchy of their own and so are not part of either side.
    print("\npartition (leading | trailing)     drops     keeps   V-cycle nnz share")
    for label, lead, trail in (
        ("[u,v,w,p] | [k,omega]", (0, 1, 2, 3), (4, 5)),
        ("[k,omega] | [u,v,w,p]", (4, 5), (0, 1, 2, 3)),
        ("[k] | [omega]", (4,), (5,)),
        ("[omega] | [k]", (5,), (4,)),
    ):
        lead, trail = np.array(lead), np.array(trail)
        scope = np.concatenate([lead, trail])
        # A block-lower-triangular split applies `trail <- lead` exactly and discards `lead <- trail`,
        # so which group leads decides which of the two triangles is thrown away. Both are measured
        # against the diagonal blocks, since those are what the two hierarchies do invert -- the
        # question a ratio here answers is whether the discarded coupling is something the preserved
        # blocks could plausibly stand in for.
        diagonal = np.sqrt(
            (norms[np.ix_(lead, lead)] ** 2).sum() + (norms[np.ix_(trail, trail)] ** 2).sum()
        )
        dropped = np.sqrt((norms[np.ix_(lead, trail)] ** 2).sum()) / diagonal
        kept = np.sqrt((norms[np.ix_(trail, lead)] ** 2).sum()) / diagonal
        hierarchy_nnz = counts[np.ix_(lead, lead)].sum() + counts[np.ix_(trail, trail)].sum()
        share = hierarchy_nnz / counts[np.ix_(scope, scope)].sum()
        print(f"{label:<30}{dropped:>10.2%}{kept:>10.2%}{share:>16.1%}")
    print(
        "\n'drops' and 'keeps' are the two off-diagonal triangles, each against the diagonal blocks\n"
        "the hierarchies invert -- so a partition is safe when what it DROPS is small, and which way\n"
        "round it is ordered decides which of the two that is. 'V-cycle nnz share' is what the\n"
        "separate hierarchies still coarsen, out of the block the split is carved from; the rest\n"
        "becomes one sparse product, which is where a split's cost saving comes from."
    )


def main() -> None:
    # Argument first, environment second: an explicit argument should win over an export left in the
    # shell from an earlier run, which is the failure a precedence rule exists to prevent.
    name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BFS3D_PROBE_STATE", "")
    if len(sys.argv) > 2 or name not in STATES:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} <{' | '.join(STATES)}>\n"
            f"       or set BFS3D_PROBE_STATE to one of them"
        )
    # By NAME, not by position. This unpacked three fields positionally and broke silently when the
    # state record grew to five -- a probe that cannot start is the benign version of that failure.
    description = STATES[name].description

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    n_cells = coupled.layout.n_cells
    names = NAMES_3D if coupled.layout.dim == 3 else ("u", "v", "p", "k", "omega")
    print(f"{'=' * 78}\n{name}: {description}\n{n_fields} fields over {n_cells} cells\n{'=' * 78}")
    state = load_state(name)

    plan = _coupled_jacobian_plan(coupled, 3)
    structure = block_stencil_gather_map(plan)
    jacobian = materialize(coupled, state, plan, structure, n_fields)

    started = time.time()
    scaled = equilibrate(jacobian)
    del jacobian
    gc.collect()
    norms, counts = block_matrix(scaled, n_cells, n_fields)
    del scaled
    gc.collect()
    print(f"  blocked in {time.time() - started:.0f}s", flush=True)
    report(norms, counts, names)


if __name__ == "__main__":
    main()
