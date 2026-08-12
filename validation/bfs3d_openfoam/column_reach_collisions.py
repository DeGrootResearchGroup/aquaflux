"""Can a short-probed column alias at all? A purely structural answer, with no Jacobian.

Shortening a column's probing reach is only sound if the entries it assembles are the ones its own
probe actually measured. The hazard is a **colour collision**: a colouring is collision-free only for
the pattern it was built from, so under a reach-two colouring two cells may share a colour and still
both couple to a common row at distance three. The probe response for that row is then the *sum* of
both couplings, and the de-compression charges the whole sum to whichever of them lies inside reach
two -- so the near entry is corrupted rather than the far one merely dropped.

Whether that can happen is a question about the **mesh graph and the colouring alone**: it needs no
state, no Jacobian and no linear solve. This reports it directly, which is worth doing before any
measurement of magnitude, because the two possible answers point in opposite directions:

* **No collisions** -- aliasing is impossible, and a divergence blamed on it has some other cause.
* **Collisions** -- aliasing is live, and what remains is to bound the mass being folded, which *is* a
  question about the state and must be read per (row field, column field) pair (a ratio taken over a
  whole column stacks every row field, and on this system the omega rows exceed the k rows by orders
  of magnitude, so they set any such ratio on their own).

The plan is built through the shipped :func:`~aquaflux.solve.column_probe_plan`, so this exercises the
real reach bookkeeping rather than a restatement of it.

Three counts per shortened column field, over the assembled pattern:

``corrupted``
    Groups holding exactly one in-reach entry and at least one out-of-reach entry. The in-reach entry
    receives the sum of the out-of-reach ones -- the failure mode above.
``discarded``
    Groups holding only out-of-reach entries. Harmless: those positions are written as the explicit
    zeros they already held, and nothing in reach reads that probe for this row.
``double_in_reach``
    Groups holding two or more in-reach entries. **Must be zero**: the colouring is built to be
    collision-free on its own pattern, so a nonzero count means the reach bookkeeping disagrees with
    the colouring and is a defect rather than a property of the mesh.

Usage (mesh only -- no state argument, and nothing here depends on one)::

    python3 -u validation/bfs3d_openfoam/column_reach_collisions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.solve import column_probe_plan  # noqa: E402

#: Field order of the coupled state, for labelling only.
FIELDS = ("u", "v", "w", "p", "k", "omega")


def collision_census(plan, field):
    """How the pattern entries of one column field group under that field's own colouring.

    Every entry of the assembled pattern is keyed by ``(row, colour of its column)``. Entries sharing
    a key are fed by the *same* probe and are therefore indistinguishable in its response.

    Parameters
    ----------
    plan : ColumnProbePlan
        The probing plan, carrying the assembled pattern, the per-field colourings and the per-field
        in-reach mask.
    field : int
        Which column field to examine.

    Returns
    -------
    dict
        ``corrupted`` / ``discarded`` / ``double_in_reach`` group counts, the number of pattern
        entries those corrupted groups put at risk, and the number of distinct rows involved.
    """
    colour = plan.colour[field]
    in_reach = plan.in_reach[field]
    rows = plan.pattern_rows.astype(np.int64)
    # One key per entry: entries with equal keys are fed by one probe and cannot be told apart.
    key = rows * (int(colour.max()) + 1) + colour[plan.pattern_cols]

    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    # Group boundaries over the sorted keys.
    starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    sizes = np.diff(np.r_[starts, sorted_key.size])

    inside = in_reach[order]
    # Per group: how many of its entries are in reach, and how many are not.
    n_in = np.add.reduceat(inside.astype(np.int64), starts)
    n_out = sizes - n_in

    corrupted = (n_in == 1) & (n_out >= 1)
    return {
        "entries": int(key.size),
        # The entries this column actually assembles from its own probe; the rest are explicit zeros.
        "in_reach": int(in_reach.sum()),
        "groups": int(starts.size),
        "corrupted": int(corrupted.sum()),
        "folded_entries": int(n_out[corrupted].sum()),
        "discarded": int(((n_in == 0) & (n_out >= 1)).sum()),
        "double_in_reach": int((n_in >= 2).sum()),
        "rows_affected": int(np.unique(rows[order][starts[corrupted]]).size),
    }


def main():
    print("[case] building the mesh", flush=True)
    case = compare.build_case()
    mesh = case["coupled"].momentum.mesh
    n = mesh.n_cells
    owner, nb, _ = mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)

    reaches = compare.COLUMN_REACH or (3, 3, 3, 2, 2, 2)
    print(f"[plan] {n} cells, column reach {reaches}, pattern reach {max(reaches)}", flush=True)
    plan = column_probe_plan(owner, nb, n, reaches)
    print(
        f"[plan] {plan.pattern_rows.size} pattern cell-blocks, "
        f"{plan.n_probes} probes ({plan.n_colours} colours per field)",
        flush=True,
    )

    print(
        f"\n  {'column':>8s} {'reach':>6s} {'in reach':>10s} {'corrupted':>10s} {'share':>7s}"
        f" {'folded':>9s} {'discarded':>10s} {'2x in-reach':>12s} {'rows':>8s}",
        flush=True,
    )
    worst = 0
    for field, reach in enumerate(reaches):
        census = collision_census(plan, field)
        worst = max(worst, census["double_in_reach"])
        share = census["corrupted"] / census["in_reach"] if census["in_reach"] else 0.0
        print(
            f"  {FIELDS[field]:>8s} {reach:6d} {census['in_reach']:10d} {census['corrupted']:10d}"
            f" {share:6.1%} {census['folded_entries']:9d} {census['discarded']:10d}"
            f" {census['double_in_reach']:12d} {census['rows_affected']:8d}",
            flush=True,
        )

    print("\n  reading:", flush=True)
    print(
        "    corrupted = 0 for every shortened column -> aliasing is structurally impossible here,\n"
        "      and a divergence attributed to it has another cause.\n"
        "    corrupted > 0 -> aliasing is live; bound the folded mass per (row field, column field)\n"
        "      pair at the state that fails, never over a whole column.",
        flush=True,
    )
    if worst:
        print(
            f"\n  DEFECT: {worst} groups hold two in-reach entries. A colouring is collision-free on\n"
            "  its own pattern, so this is the reach bookkeeping disagreeing with the colouring.",
            flush=True,
        )


if __name__ == "__main__":
    main()
