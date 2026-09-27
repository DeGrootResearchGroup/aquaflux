"""How many tiles shaft culling asks about at each level of its ladder, and what that costs.

``field_cost_breakdown.py`` charges the certificates -- the bodies vouching for tiles -- as one
number. That number grew from 14.3 s to 53.5 s across two changes that should each have made it
smaller: the facets' side of the tiles formed once per call rather than once per chunk, and tiles
wholly behind their facets dropped before any body is asked. A certificate's cost is set by how
many tiles it is asked about, and that count depends on the receivers' own arrangement -- which is
why a sampled stand-in cannot say what happened on the mesh. This counts it on the mesh.

It streams the field the way ``fluence_rate`` streams a model's -- ``direct_fluence_rate`` with the
water as the occluder, ``NoOcclusion`` for the lamp's own shadowing, every other setting the library
default -- and, per level of the ladder, reports:

- the calls, the tiles asked, the tiles refused (asked again one level down, or tested pair by pair
  at the finest), and the seconds spent vouching;
- where the code drops tiles behind their facets first (it does not before that change), the tiles
  dropped at each level and the seconds that took.

A level is recognized by how many facet clusters the summary it is handed holds, which is fixed by
the facet count and the level's cluster size, so the counting reads nothing the strategy does not
already pass. It wraps functions by name and so runs on the code before and after either change;
``SOZZI_LABEL`` names the arm in the summary and the output file, since the two are compared across
runs of one checkout at two commits. ``SOZZI_RECEIVERS`` takes the first *n* cells in mesh order, for
a quick check of the harness itself; a figure quoted from this file is from the whole mesh.

Run with ``validation/run_case.sh validation/sozzi_radiation/certificate_levels.py`` at each commit
to be compared. Needs ``work/case`` (for the lamp) and ``work/cell_centres.npy``. Writes
``work/compare/certificate_levels[-<label>].json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

WORK = HERE / "work"
LABEL = os.environ.get("SOZZI_LABEL", "")
RECEIVERS = int(os.environ.get("SOZZI_RECEIVERS", 0))
RESULT = WORK / "compare" / f"certificate_levels{'-' + LABEL if LABEL else ''}.json"


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _commit() -> str:
    """The checkout's commit, so a summary says which code it counted."""
    try:
        return subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class LevelCounts:
    """Per level of the ladder: calls, tiles asked and refused or dropped, and seconds.

    Parameters
    ----------
    n_facets : int
        How many facets the tiles' clusters are cut from.
    cluster_sizes : tuple of int
        The ladder's facets per cluster, coarsest first.
    """

    def __init__(self, n_facets: int, cluster_sizes: tuple[int, ...]):
        padded = -(-n_facets // cluster_sizes[0]) * cluster_sizes[0]
        self.level_by_groups = {padded // size: level for level, size in enumerate(cluster_sizes)}
        self.level_by_size = {size: level for level, size in enumerate(cluster_sizes)}
        self.levels: dict[str, dict] = {}

    def level_of(self, n_clusters: int) -> int | str:
        """The level whose cluster size gives ``n_clusters`` groups of the facets."""
        return self.level_by_groups.get(n_clusters, f"unknown ({n_clusters} clusters)")

    def add(self, what: str, level, asked: int, refused: int, seconds: float) -> None:
        entry = self.levels.setdefault(
            f"{what} at level {level}", {"calls": 0, "asked": 0, "refused": 0, "seconds": 0.0}
        )
        entry["calls"] += 1
        entry["asked"] += asked
        entry["refused"] += refused
        entry["seconds"] += seconds


def instrument(counts: LevelCounts) -> None:
    """Count every certificate and every behind-the-facets drop, by level."""
    import numpy as np
    from aquaflux.radiation import culling

    vouched, vouched_pairs = culling._vouched, culling._vouched_pairs

    def counted_vouched(body, receiver_summary, source_summary):
        started = time.perf_counter()
        clear = vouched(body, receiver_summary, source_summary)
        seconds = time.perf_counter() - started
        level = counts.level_of(len(source_summary))
        counts.add("vouch", level, clear.size, int(np.count_nonzero(~clear)), seconds)
        return clear

    def counted_vouched_pairs(body, receiver_summary, source_summary, rows, cols):
        started = time.perf_counter()
        clear = vouched_pairs(body, receiver_summary, source_summary, rows, cols)
        counts.add(
            "vouch",
            counts.level_of(len(source_summary)),
            len(clear),
            int(np.count_nonzero(~clear)),
            time.perf_counter() - started,
        )
        return clear

    culling._vouched, culling._vouched_pairs = counted_vouched, counted_vouched_pairs
    try:
        from aquaflux.radiation.back_faces import BackFaces
    except ImportError:
        return
    tiles_behind = BackFaces.tiles_behind

    def counted_tiles_behind(self, receivers, rows, cols):
        started = time.perf_counter()
        dark = tiles_behind(self, receivers, rows, cols)
        seconds = time.perf_counter() - started
        level = counts.level_by_size.get(cols.shape[-1], f"unknown ({cols.shape[-1]} per cluster)")
        counts.add("drop behind", level, len(dark), int(np.count_nonzero(dark)), seconds)
        return dark

    BackFaces.tiles_behind = counted_tiles_behind


def count(lamp, cells, water, absorption) -> tuple[LevelCounts, float]:
    """Stream the field once with every certificate counted; the counts and the field's seconds."""
    import numpy as np
    from aquaflux.radiation import NoOcclusion, ShaftCulling, direct_fluence_rate

    counts = LevelCounts(int(lamp.n_facets), ShaftCulling().source_clusters)
    instrument(counts)
    started = time.perf_counter()
    field = direct_fluence_rate(
        lamp, cells, occluders=[water], self_occlusion=NoOcclusion(), absorption=absorption
    )
    if not np.isfinite(np.asarray(field)).all():
        raise SystemExit("the field is not finite")
    return counts, time.perf_counter() - started


def main() -> None:
    import aquaflux  # noqa: F401  (enables x64)
    import jax
    import numpy as np
    from aquaflux.radiation import ShaftCulling, UniformAbsorption
    from compare_fluence import ABSORPTION, lamp_surfaces
    from primitive_occlusion import fluid, from_drawing

    cells = np.load(WORK / "cell_centres.npy")
    if RECEIVERS:
        cells = cells[:RECEIVERS]
    lamp = lamp_surfaces()
    water = from_drawing()
    source = "the CAD drawing"
    if water is None:
        water, source = fluid(), "three hand-typed cylinders (no CAD kernel)"
    ladder = ShaftCulling()
    n_facets = int(lamp.n_facets)
    commit = _commit()
    _say(
        f"{len(cells)} cells, {n_facets} lamp facets, water from {source}, default ladder "
        f"{ladder.receiver_blocks} x {ladder.source_clusters}; commit {commit}, "
        f"label {LABEL or '(none)'}; jax {jax.__version__}"
    )
    counts, seconds = count(lamp, cells, water, UniformAbsorption(ABSORPTION))
    _say(f"field computed in {seconds:.1f} s")
    for name, entry in sorted(counts.levels.items()):
        _say(
            f"  {name:<28} calls {entry['calls']:6d}  asked {entry['asked']:>14,}  "
            f"refused/dropped {entry['refused']:>14,}  {entry['seconds']:8.2f} s"
        )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(
        json.dumps(
            {
                "commit": commit,
                "label": LABEL,
                "cells": len(cells),
                "lamp_facets": n_facets,
                "water": source,
                "field_seconds": seconds,
                "levels": counts.levels,
            },
            indent=2,
        )
        + "\n"
    )
    _say(f"wrote {RESULT}")


if __name__ == "__main__":
    main()
