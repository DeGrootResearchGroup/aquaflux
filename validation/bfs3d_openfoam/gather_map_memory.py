"""Peak memory of building the coloured probe's de-compression map.

``block_stencil_gather_map`` and the :class:`ProbeGather` it returns are built **once**, when a case is
assembled, and they are the largest transient allocation in a run — larger than a materialize, and
larger than the assembled case. That cost is invisible to any measurement taken per materialize, which
is what this harness exists to stop: it reports the peak resident set of the build itself, in bytes per
pattern entry, so the figure can be compared across meshes and re-taken after a change.

Run it on a synthetic cubic lattice rather than on a case file. The quantity depends only on the
pattern's *shape* — entries per row and total entries — so a lattice of the same stencil reach and field
count is representative of a real mesh at a fraction of the memory, and needs no mesh, no state and no
solve.

Two things about the measurement, both learned the hard way and both worth keeping:

* **Peak resident set is a high-water mark, so one arm per process.** Two arms in one process report the
  larger of the two for whichever ran second.
* **Discard the first run.** A cold allocator reads low, and by more than the effects usually being
  looked for here: an arm that then repeats at 65.7 bytes per entry five times running has read 67.7 on
  a session's first run, and a prototype of the same shape read 47 against a steady 71. Run several
  repetitions and take the minimum.

Peak resident set is an allocation high-water mark, so — unlike wall clock — it barely moves under
competition for the machine: the pre-change arm read 100.1 bytes per entry both on an idle machine and
beside a running case. That makes this the measurement to reach for when a probe cannot wait for a quiet
machine, and it is why this harness reports memory rather than time.

Usage::

    python gather_map_memory.py [--side N] [--reach R] [--fields F] [--check]

``--check`` verifies the map against a direct de-compression on a small pattern instead of measuring
anything.

What this reports is the build's peak **as a whole**. Attributing it to a phase within the build needs
the phases run separately, which cannot be done from outside without restating what the build does — and
a copy of that here would drift from it. Do it by instrumenting the function itself, and note the trap
that made it worth doing once: a phase whose peak sits *under* another's is invisible end to end, and
becomes the ceiling the moment the larger one is reduced.
"""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

import numpy as np

# Running a script puts the SCRIPT's directory on `sys.path`, not the working directory, so this cannot
# find `aquaflux` from a plain checkout unless the repo root is added explicitly, as the sibling
# harnesses in this directory do.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aquaflux.solve import (  # noqa: E402
    ColumnProbePlan,
    ProbeGather,
    block_stencil_colouring,
    block_stencil_gather_map,
)


def peak_bytes() -> int:
    """Peak resident set of this process so far, in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes; a peak below a megabyte is not a real process.
    return peak if peak > 1 << 20 else peak * 1024


def lattice_edges(side: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Interior-face cell graph of a ``side x side x side`` lattice of cells.

    Returns
    -------
    owner, neighbour : np.ndarray
        The edge endpoints, shape ``(n_edges,)`` each.
    n_cells : int
        Number of cells.
    """
    n = side**3
    index = np.arange(n).reshape(side, side, side)
    owner, neighbour = [], []
    for axis in range(3):
        owner.append(np.take(index, np.arange(side - 1), axis=axis).ravel())
        neighbour.append(np.take(index, np.arange(1, side), axis=axis).ravel())
    return np.concatenate(owner), np.concatenate(neighbour), n


def build_plan(side: int, reach: int, n_fields: int) -> ColumnProbePlan:
    """A uniform-reach probing plan over the lattice, the input the map is built from."""
    owner, neighbour, n = lattice_edges(side)
    colouring = block_stencil_colouring(owner, neighbour, n, reach)
    return ColumnProbePlan.uniform(colouring, n_fields)


def retained_bytes(gather: ProbeGather) -> int:
    """Bytes the finished map holds for the life of the preconditioner."""
    return sum(
        array.nbytes for array in (gather.indptr, gather.indices, gather._position, gather._source)
    )


def measure(plan: ColumnProbePlan) -> None:
    """Report the build's peak transient and what it leaves behind."""
    entries = plan.pattern_rows.shape[0] * plan.n_fields**2
    before = peak_bytes()
    gather = block_stencil_gather_map(plan)
    peak = peak_bytes()
    print(
        f"  transient {(peak - before) / 1e6:7.0f} MB "
        f"({(peak - before) / entries:5.1f} B/entry)   "
        f"retained {retained_bytes(gather) / 1e6:6.0f} MB"
    )


def check(side: int, reach: int, n_fields: int) -> None:
    """Verify the map against a direct de-compression, on a pattern small enough to do both ways.

    The map says which probe response supplies each Jacobian entry. Filling it from synthetic responses
    whose value encodes ``(probe, row)`` makes every entry's provenance checkable, so a map that is
    self-consistent but wrong is caught rather than merely a map that crashes.
    """
    plan = build_plan(side, reach, n_fields)
    gather = block_stencil_gather_map(plan)
    n, nf = plan.n_cells, plan.n_fields * plan.n_cells
    responses = np.arange(plan.n_probes * nf, dtype=np.float64).reshape(plan.n_probes, nf)

    data = np.zeros(gather.nnz)
    for start in range(plan.n_probes):
        gather.scatter(data, responses[start : start + 1], start, 1)

    # Every entry the map filled must hold the response element its (row, column) implies. Derived for
    # all entries at once rather than by walking rows: at this pattern's size a Python loop over the
    # entries is minutes where the vectorized form is immediate.
    row_of = np.repeat(np.arange(gather.indptr.shape[0] - 1), np.diff(gather.indptr))
    field_of, cell_of = np.divmod(gather.indices.astype(np.int64), n)
    probe_of = np.asarray(plan.probe_base)[field_of] + plan.colour[field_of, cell_of]
    expected = probe_of.astype(np.float64) * nf + row_of

    # An out-of-reach entry is never written and keeps its zero, so only the written ones are compared.
    # An entry whose expected value is genuinely zero is indistinguishable from those and is skipped;
    # that is one entry of the pattern, not a hole in the check.
    written = data != 0.0
    failures = int(np.count_nonzero(data[written] != expected[written]))
    print(f"check: {gather.nnz} entries, {int(written.sum())} written, {failures} mismatched")
    if failures:
        raise SystemExit("gather map does not reproduce a direct de-compression")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=14, help="cells per lattice edge")
    parser.add_argument("--reach", type=int, default=3, help="stencil reach of the pattern")
    parser.add_argument("--fields", type=int, default=6, help="degrees of freedom per cell")
    parser.add_argument("--check", action="store_true", help="verify the map instead of timing it")
    args = parser.parse_args()

    if args.check:
        check(min(args.side, 6), args.reach, args.fields)
        return

    plan = build_plan(args.side, args.reach, args.fields)
    entries = plan.pattern_rows.shape[0] * plan.n_fields**2
    print(
        f"lattice {args.side}^3 = {plan.n_cells} cells, reach {args.reach}, {plan.n_fields} fields\n"
        f"  {plan.pattern_rows.shape[0]} cell blocks, {entries} entries, "
        f"{plan.n_probes} probes, resident before {peak_bytes() / 1e6:.0f} MB"
    )
    measure(plan)
    print(
        "  one arm per process and discard the first run; a cold allocator reads low.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
