"""What the facet-to-facet transfer build costs in memory, on the Sozzi lamp.

The transfer is ``n x n`` in each of three stored arrays — the solid-angle weights, the source
cosines, the separations — plus a facet-side shadow mask, so at the case's 7,516 lamp facets it
must keep about 1.9 GB. What it *peaks* at is another matter: on this backend freed memory is not
handed back to the system, so a build's peak is the running total of every array it ever formed,
and the first version, which assembled each quantity whole, peaked at 8.33 GB. This measures the
peak of :func:`~aquaflux.radiation.build_transfer` alone, in a process of its own so that
``/usr/bin/time -l`` can read it, beside what the result actually holds.
``validation/peak_footprint.py`` runs that process, and forwards a ``kill`` of this one to it rather
than leaving it orphaned.

The water is the three hand-typed cylinders of ``primitive_occlusion.py`` and the lamp is convex, so
self-occlusion is off, as in the Sozzi field itself; no CAD kernel is needed.

Run with ``validation/run_case.sh validation/sozzi_radiation/transfer_build_peak.py``. Needs
``work/case`` for the lamp's STL.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def build() -> dict:
    """Build the transfer, in this process, and say what it holds."""
    import aquaflux  # noqa: F401  (enables x64)
    import numpy as np
    from aquaflux.radiation import NoOcclusion, build_transfer
    from compare_fluence import lamp_surfaces
    from primitive_occlusion import fluid

    lamp = lamp_surfaces()
    started = time.perf_counter()
    transfer = build_transfer(lamp, occluders=[fluid()], self_occlusion=NoOcclusion())
    transfer.geometric.block_until_ready()
    seconds = time.perf_counter() - started
    arrays = (
        transfer.geometric,
        transfer.source_cosine,
        transfer.separation,
        transfer.visibility.blocked,
        transfer.visibility.hidden_by_geometry,
        transfer.visibility.overlapping,
    )
    return {
        "facets": int(lamp.n_facets),
        "seconds": round(seconds, 1),
        # A mask stores the surface's own layer at the narrowest type that holds it, and not at
        # all where the surface hides nothing -- so a missing layer is counted as the nothing it is.
        "stored_GB": round(sum(np.asarray(a).nbytes for a in arrays if a is not None) / 1e9, 2),
        "one_n_by_n_float_GB": round(np.asarray(transfer.geometric).nbytes / 1e9, 3),
    }


def main() -> None:
    from peak_footprint import run_with_footprint

    with tempfile.TemporaryDirectory() as scratch:
        result = Path(scratch) / "result.json"
        child = run_with_footprint([sys.executable, "-u", __file__, "--child", str(result)])
        if child.returncode:
            sys.exit(child.returncode)
        summary = json.loads(result.read_text())
    if child.peak_footprint_gb is None:
        sys.exit("/usr/bin/time reported no peak memory footprint")
    summary["peak_footprint_GB"] = round(child.peak_footprint_gb, 2)
    summary["peak_over_one_array"] = round(
        summary["peak_footprint_GB"] / summary["one_n_by_n_float_GB"], 1
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        Path(sys.argv[2]).write_text(json.dumps(build()))
    else:
        main()
