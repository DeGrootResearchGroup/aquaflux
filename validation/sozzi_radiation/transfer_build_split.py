"""Where the lamp's transfer build spends its time, and whether skipping columns changes the answer.

``build_transfer`` fills three ``n x n`` arrays block by block (``transfer._row_blocks``): per block
of receiving facets, a host test picks the sending facets that can contribute
(``transfer._columns_in_front``), one compiled pass evaluates the costly geometric term against
those columns only (``transfer._row_block``), and the block is written into the kept arrays
(``transfer._written``). This harness times each of the three, each waited on until its result is
ready, and counts the columns the host test keeps.

It then builds the same matrix a second time with the host test replaced by "every areal facet",
the reference every column-skipping rule must reproduce, and reports the largest difference in
each of the three arrays. The geometric term may differ by rounding -- a pair with no transfer
comes back from a differently shaped program as dust of order ``1e-17`` and a skipped one as an
exact zero -- but by nothing a transfer factor is made of; the other two arrays are formed against
every facet either way and must agree exactly.

Scene: the lamp alone, as in the Sozzi model (the case's ``lampWall.stl`` when ``work/case``
exists, else the analytic 32 x 128 lamp), at the library's default receiver quadrature and block
size. The lamp is convex, so every entry of its geometric term is zero to rounding; how many
columns the host test still keeps is the waste this measures.

Run with ``validation/run_case.sh validation/sozzi_radiation/transfer_build_split.py``. Writes
``work/compare/transfer_build_split.json``.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import NoOcclusion, build_transfer  # noqa: E402
from aquaflux.radiation import transfer as transfer_module  # noqa: E402
from primitive_occlusion import OUT, lamp  # noqa: E402

ARRAYS = ("geometric", "source_cosine", "separation")


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class _Instrumented:
    """Wraps the build's three per-block steps with timers that wait for their results."""

    def __init__(self):
        self.seconds = {"front test": 0.0, "row block": 0.0, "write": 0.0}
        self.kept = []
        self._real = {
            name: getattr(transfer_module, name)
            for name in ("_columns_in_front", "_row_block", "_written")
        }

    def __enter__(self):
        real = self._real

        def front(*args):
            started = time.perf_counter()
            columns = real["_columns_in_front"](*args)
            self.seconds["front test"] += time.perf_counter() - started
            self.kept.append(len(columns))
            return columns

        def row_block(*args):
            started = time.perf_counter()
            block = jax.block_until_ready(real["_row_block"](*args))
            self.seconds["row block"] += time.perf_counter() - started
            return block

        def written(*args):
            started = time.perf_counter()
            buffers = jax.block_until_ready(real["_written"](*args))
            self.seconds["write"] += time.perf_counter() - started
            return buffers

        transfer_module._columns_in_front = front
        transfer_module._row_block = row_block
        transfer_module._written = written
        return self

    def __exit__(self, *exc):
        for name, function in self._real.items():
            setattr(transfer_module, name, function)


def _build(surfaces):
    started = time.perf_counter()
    built = build_transfer(surfaces, self_occlusion=NoOcclusion())
    arrays = {name: np.asarray(getattr(built, name)) for name in ARRAYS}
    return arrays, time.perf_counter() - started


def main() -> None:
    surfaces, label = lamp()
    n = surfaces.n_facets
    _say(
        f"{n} facets ({label}); jax {jax.__version__}, {platform.system()} {platform.machine()}, "
        f"{os.cpu_count()} cores"
    )

    # A warm-up build compiles every program the timed one runs, so the timed build measures work.
    _build(surfaces)
    with _Instrumented() as probe:
        built, total = _build(surfaces)
    kept = np.asarray(probe.kept)
    _say(
        f"build {total:.1f} s: front test {probe.seconds['front test']:.1f} s, row blocks "
        f"{probe.seconds['row block']:.1f} s, writes {probe.seconds['write']:.1f} s, rest "
        f"{total - sum(probe.seconds.values()):.1f} s"
    )
    _say(
        f"columns kept per block: median {np.median(kept):.0f}, max {kept.max()} of {n}; "
        f"{kept.sum() / (len(kept) * n):.1%} of every block's full width"
    )

    original = transfer_module._columns_in_front
    transfer_module._columns_in_front = lambda vertices, sample, normal, areal, index: (
        np.flatnonzero(areal)
    )
    try:
        reference, reference_seconds = _build(surfaces)
    finally:
        transfer_module._columns_in_front = original
    differences = {name: float(np.abs(built[name] - reference[name]).max()) for name in ARRAYS}
    largest = float(np.abs(reference["geometric"]).max())
    _say(
        f"reference (every column) {reference_seconds:.1f} s; largest |F| in it {largest:.3e}; "
        "largest difference from it: "
        + ", ".join(f"{name} {value:.3e}" for name, value in differences.items())
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transfer_build_split.json").write_text(
        json.dumps(
            {
                "lamp": label,
                "facets": n,
                "build_s": total,
                "seconds": probe.seconds,
                "kept_per_block": kept.tolist(),
                "reference_s": reference_seconds,
                "largest_reference_F": largest,
                "differences": differences,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
