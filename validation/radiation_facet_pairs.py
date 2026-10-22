"""What casting each pair of facets once, and only where it is read, saves the facet-to-facet mask.

The transfer's shadow mask from the surface's own triangles used to cast both rays of every
pair of facets, ``i -> j`` and ``j -> i``. They are one segment with the same two facets excluded,
so they can disagree only where a hit lands within a margin of an end, or by the rounding of the
reversed ray. The mask now casts one ray per pair and writes it to both entries
(``RayCastOcclusion.between_facets``), and only for pairs whose transfer is non-zero one way or the
other, or which join a facet to a point source (``transfer._read_pairs``).

For each scene of a ladder this prints, every arm in one process on the same scene, warmed, two
alternating passes, fastest kept with the spread of the two:

- **both ways**: the old build -- ``build_visibility`` with the facets as receivers, every pair
  cast in both directions;
- **once**: one ray per pair, every pair;
- **once, read**: one ray per pair, only the pairs the transfer reads -- what ``build_transfer``
  now does.

and, from the old mask, how many pairs its two directions disagree on (the only pairs whose answer
can change), and the share of pairs the transfer reads. Each new mask is checked against the old
one at the ray it casts, bit for bit, and on the read pairs where the old mask is symmetric.

The scene is ``radiation_mask_build_cost.py``'s: a closed box with a lamp sleeve down its axis,
every triangle a blocker. ``RADIATION_FACET_PAIRS_GRID=1`` repeats the ladder with the ray test's
grid on; ``RADIATION_FACET_PAIRS_GRID=only`` runs that ladder alone, and
``RADIATION_FACET_PAIRS_RUNGS=3,4`` runs only those rungs of the ladder (0 is the smallest).

Run with ``validation/run_case.sh validation/radiation_facet_pairs.py``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
from aquaflux.radiation.self_occlusion import NoOcclusion, RayCastOcclusion
from aquaflux.radiation.transfer import _read_pairs, build_transfer
from aquaflux.radiation.visibility import build_visibility
from validation.radiation_mask_build_cost import reactor


def both_ways(surfaces, strategy):
    """The old build: every pair of facets cast in both directions."""
    facets = np.arange(surfaces.n_facets)
    mask = build_visibility(
        (), surfaces, surfaces.centroid, receiver_facet=facets, self_occlusion=strategy
    )
    return mask.hidden_by_geometry


def alternating(arms: dict, passes: int = 2) -> dict:
    """Each arm warmed once, then ``passes`` rounds of every arm in turn; fastest kept."""
    answers = {name: np.asarray(build()) for name, build in arms.items()}
    seconds = {name: [] for name in arms}
    for _ in range(passes):
        for name, build in arms.items():
            start = time.perf_counter()
            answers[name] = np.asarray(build())
            seconds[name].append(time.perf_counter() - start)
    return {name: (min(s), max(s) / min(s), answers[name]) for name, s in seconds.items()}


def rung(divisions: int, sectors: int, grid) -> None:
    surfaces = reactor(divisions, sectors)
    n = surfaces.n_facets
    near = 1e-6 * np.sqrt(np.asarray(surfaces.area))
    strategy = RayCastOcclusion(grid=grid).prepared(surfaces)
    geometric = build_transfer(surfaces, self_occlusion=NoOcclusion()).geometric
    read = _read_pairs(geometric, surfaces.is_point_source)

    results = alternating(
        {
            "both ways": lambda: both_ways(surfaces, strategy),
            "once": lambda: strategy.between_facets(surfaces, near).fraction,
            "once, read": lambda: strategy.between_facets(surfaces, near, read).fraction,
        }
    )
    full = results["both ways"][2]
    lower, higher = np.triu_indices(n, 1)
    # No point sources in this scene, so every pair is cast from its lower index.
    cast = full[higher, lower]
    for name in ("once", "once, read"):
        answer = results[name][2]
        kept = np.ones_like(cast) if name == "once" else (read | read.T)[lower, higher]
        assert np.array_equal(answer[lower, higher][kept], cast[kept]), name
        assert np.array_equal(answer[higher, lower][kept], cast[kept]), name
    disagree = int(np.count_nonzero(full[lower, higher] != full[higher, lower]))
    symmetric_read = read & (full == full.T)
    assert np.array_equal(results["once, read"][2][symmetric_read], full[symmetric_read])

    base = results["both ways"][0]
    cells = [f"{n:>7}", f"{np.count_nonzero(full) / full.size:>8.3f}"]
    cells += [f"{disagree:>9}", f"{np.count_nonzero(read | read.T) / read.size:>6.3f}"]
    for name in ("both ways", "once", "once, read"):
        seconds, spread, _ = results[name]
        cells.append(f"{seconds:>9.2f} {spread:>5.2f}x {base / seconds:>5.2f}x")
    print(" ".join(cells), flush=True)


def ladder(cases, grid) -> None:
    print(f"\n### RayCastOcclusion(grid={grid}), facets as receivers\n", flush=True)
    header = [f"{'facets':>7}", f"{'blocked':>8}", f"{'disagree':>9}", f"{'read':>6}"]
    for name in ("both ways", "once", "once, read"):
        header.append(f"{name + ' s':>9} {'spread':>6} {'gain':>6}")
    print(" ".join(header), flush=True)
    for divisions, sectors in cases:
        rung(divisions, sectors, grid)


if __name__ == "__main__":
    print(__doc__.split("Run with")[0].strip(), flush=True)
    cases = [(4, 8), (6, 12), (8, 16), (11, 20), (14, 24)]
    rungs = os.environ.get("RADIATION_FACET_PAIRS_RUNGS")
    if rungs:
        cases = [cases[int(index)] for index in rungs.split(",")]
    grid = os.environ.get("RADIATION_FACET_PAIRS_GRID", "")
    if grid != "only":
        ladder(cases, False)
    if grid:
        ladder(cases, True)
