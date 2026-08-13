"""Is a reach-two probe of the k and omega columns safe, in the block the preconditioner applies?

A shortened column is assembled from a colouring that is collision-free only on its *own* pattern, so
two cells sharing a reach-two colour can both couple to one row of the reach-three assembly pattern.
The probe response for that row is their sum and the de-compression charges all of it to whichever cell
lies in reach -- the near entry is corrupted, not merely the far one dropped.
``column_reach_collisions.py`` shows that happens for over half the entries of every shortened column on
this mesh. What it cannot say is whether the folded values are *large enough to matter*, which is a
question about the state, and about which rows the preconditioner actually reads.

This measures the error exactly rather than bounding it. Two directional derivatives per column:

* **aliased** -- seed every cell of the column's colour, which is literally what the probe does, so the
  response is what de-compression assigns;
* **exact** -- seed the single cell, which is the true column.

Their difference over the rows that read this column *is* the assembled error. No materialization is
involved, so the whole sweep costs megabytes where an audit of the full Jacobian costs gigabytes.

**The reading depends on which rows the error lands in, and that is the point.** A flow-first
block-triangular split applies ``d R_turb / d flow`` and never ``d R_flow / d turb``, so error in the
**flow rows** of a turbulence column sits in a block that is never applied and is free. Only the
**turbulence rows** matter, and that block is preconditioned by a short-range cell-block inverse. The
two are therefore reported separately; a single figure over the whole column would average the one that
matters into the one that does not -- the same collapse that let a corrupted pressure column pass an
audit taken over whole columns.

States are **constructed, not loaded**, so this is reproducible: each continuation rung's initial field
comes from the hybrid initialization at that rung's own viscosity. Those are the coldest states in the
ladder and the furthest from where a converged march was compared, which is exactly where a result that
is "a property of this run" would show itself. Pass checkpoint names to add developed states -- quote
their provenance if you do, since checkpoint names are reused between marches.

Usage::

    python3 -u validation/bfs3d_openfoam/trailing_column_reach_probe.py [state-00042 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.solve import block_stencil_colouring  # noqa: E402
from aquaflux.turbulence import hybrid_initialize  # noqa: E402
from column_reach_probe import graph_distance  # noqa: E402

#: Field order of the coupled state.
FIELDS = ("u", "v", "w", "p", "k", "omega")
#: The columns under test, and the reach in question.
TRAILING = (4, 5)
REACH = 2
#: Probed columns: the two under test plus **pressure as a positive control**. Pressure at reach two is
#: known to diverge this case, so it is what distinguishes "these columns are clean" from "this probe
#: reports zero for everything" -- a measurement that cannot fail is not evidence. Its error lands in
#: the *flow* rows, which for a turbulence column are free but for pressure are the leading saddle block
#: the preconditioner applies, so read each column against the row group its own rows land in.
PROBED = (3, 4, 5)
#: Viscosity scale factors of the continuation ladder, coldest first (two rungs plus the target).
RUNGS = (100.0, 10.0, 1.0)


def sample_cells(mesh, count=4):
    """A few deep-interior cells and a few wall-adjacent ones, as (label, index) pairs.

    Both matter and for opposite reasons: an interior cell pays the full stencil the mesh supports,
    while the near-wall cells are where ``k`` and ``omega`` are stiffest and where the trailing block is
    hardest to precondition.
    """
    centroid = np.asarray(mesh.geometry().cell.centroid)
    span = centroid.max(axis=0) - centroid.min(axis=0)
    middle = centroid.min(axis=0) + 0.5 * span
    deep = np.argsort(np.linalg.norm((centroid - middle) / span, axis=1))[:count]

    wall_faces = np.concatenate(
        [np.asarray(mesh.face_patches.indices(name)) for name in ("lowerWall", "sideWalls")]
    )
    wall_cells = np.unique(np.asarray(mesh.face_cells.owner)[wall_faces])
    # Spread over the wall set rather than taking the first few, which would all be one corner.
    near = wall_cells[np.linspace(0, wall_cells.size - 1, count).astype(int)]
    return [("interior", int(c)) for c in deep] + [("wall", int(c)) for c in near]


def column_error(coupled, state, cell, field, colour, distance, n, n_fields):
    """Exact assembled error for one column under a reach-two probe, split by row field group.

    Returns
    -------
    dict
        Per row group (``turb`` = the k/omega rows the split applies, ``flow`` = the rows it drops):
        the true column's norm over the rows that read it, the norm of the aliasing error there, and
        the norm of the mass at distance three that is written as an explicit zero.
    """
    seeded = np.zeros(n_fields * n)
    seeded[field * n + np.flatnonzero(colour == colour[cell])] = 1.0
    lone = np.zeros(n_fields * n)
    lone[field * n + cell] = 1.0

    def response(seed):
        out = jax.jvp(coupled.residual, (jnp.asarray(state),), (jnp.asarray(seed),))[1]
        return np.asarray(out, dtype=np.float64).reshape(n_fields, n)

    aliased, exact = response(seeded), response(lone)
    # Rows that read this column: the colouring is collision-free on its own pattern, so within reach
    # this cell is the unique representative of its colour and the de-compression charges it the sum.
    reads = distance <= REACH
    dropped = distance == REACH + 1

    out = {}
    for label, rows in (("turb", list(TRAILING)), ("flow", [0, 1, 2, 3])):
        true_here = exact[rows][:, reads]
        out[label] = {
            "true": float(np.linalg.norm(true_here)),
            "error": float(np.linalg.norm((aliased - exact)[rows][:, reads])),
            "dropped": float(np.linalg.norm(exact[rows][:, dropped])),
        }
    return out


def rung_states(case, scales):
    """``(label, assembler, packed state)`` for each rung's own cold initial field."""
    coupled = case["coupled"]
    frozen = jax.lax.stop_gradient(coupled)
    for scale in scales:
        companion = frozen.with_scaled_molecular_viscosity(scale)
        flow, k, omega = hybrid_initialize(companion.momentum, companion.turbulence)
        label = "target Re" if scale == 1.0 else f"Re/{scale:g}"
        yield label, companion, np.asarray(companion.state_from_physical(flow, k, omega))


def main():
    print("[case] building", flush=True)
    case = compare.build_case()
    mesh = case["coupled"].momentum.mesh
    n = mesh.n_cells
    n_fields = case["coupled"].layout.dim + 3
    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)

    colour = block_stencil_colouring(owner, nb, n, REACH).colour
    cells = sample_cells(mesh)
    print(f"[plan] reach {REACH} colouring, {colour.max() + 1} colours; cells {cells}", flush=True)
    distances = graph_distance(owner, nb, n, [c for _, c in cells])

    states = list(rung_states(case, RUNGS))
    for name in sys.argv[1:]:
        path = CASE / "checkpoints" / f"{name}.npz"
        if not path.exists():
            raise SystemExit(f"{name}: no such checkpoint")
        with np.load(path) as data:
            states.append((f"{name} (loaded)", case["coupled"], np.asarray(data["state"])))
            print(
                f"[state] {name}: quote its provenance -- checkpoint names are reused", flush=True
            )

    print(
        "\n  error/true = the ALIASING error actually assembled, relative to the true column,"
        "\n  over the rows that read it. dropped/true = far mass written as an explicit zero."
        "\n  'turb' rows are applied by the split; 'flow' rows of a turbulence column are NOT.\n",
        flush=True,
    )
    header = (
        f"  {'state':>12s} {'cell':>10s} {'col':>6s} "
        f"{'turb err/true':>14s} {'turb drop/true':>15s} {'flow err/true':>14s}"
    )
    worst = 0.0
    for label, assembler, state in states:
        print(f"{header}" if label == states[0][0] else "", end="", flush=True)
        for (kind, cell), distance in zip(cells, distances, strict=True):
            for field in PROBED:
                stats = column_error(assembler, state, cell, field, colour, distance, n, n_fields)
                turb, flow = stats["turb"], stats["flow"]
                ratio = turb["error"] / turb["true"] if turb["true"] else 0.0
                if field in TRAILING:
                    worst = max(worst, ratio)
                print(
                    f"\n  {label:>12s} {kind + ' ' + str(cell):>10s} {FIELDS[field]:>6s} "
                    f"{ratio:14.2e} "
                    f"{(turb['dropped'] / turb['true'] if turb['true'] else 0.0):15.2e} "
                    f"{(flow['error'] / flow['true'] if flow['true'] else 0.0):14.2e}",
                    end="",
                    flush=True,
                )
        print(flush=True)

    print(
        f"\n  worst turbulence-row aliasing error over every column and state: {worst:.2e}",
        flush=True,
    )
    print(
        "  At the float64 floor (~1e-15) reach two is exact for these columns wherever it was tried,\n"
        "  and the licence does not depend on the state. Materially above it, the shortened reach is\n"
        "  carrying a real perturbation into the block the trailing inverse applies.",
        flush=True,
    )


if __name__ == "__main__":
    main()
