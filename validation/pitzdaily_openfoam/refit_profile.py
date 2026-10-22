"""Where does one refit of the shipped field split spend its host-side refactor?

A refit probes the coupled Jacobian straight into the field split's blocks, shifts them, and re-fits
the split through ``FieldSplitInverse.refactor_blocks``; the first fit goes through
``field_split._field_split_of_blocks``. The march log's ``pc`` lines split a refit into probe and
refactor; this harness splits the refactor itself, on the blocks the shipped march actually refits to.

Two modes, chosen by ``PITZ_REFIT_MODE``:

``capture``
    Runs the shipped march (``compare.py``'s ``solve_aquaflux``) and saves the shifted blocks of the
    first fit and of the first ``PITZ_REFIT_CAPTURE`` (default 3) refits to ``refit_matrices/``, then
    stops the march. Run with ``validation/run_case.sh``.
``profile`` (the default)
    Loads those blocks, fits the case file's field split to the first, then re-fits it to each captured
    refit under ``cProfile`` and prints the cumulative time of the refactor's parts: the products, the
    padded-row packing, the smoother set-up per level, the spectral estimates, the coarse inverse. Then
    times ``PITZ_REFIT_REPEATS`` (default 5) plain refits per capture, best and median.

The split is fitted at the case file's settings (``case.yaml``'s ``FieldSplit``), its blocks held in the
file's precision as the session holds them, so the profile is of the shipped preconditioner.

The capture hooks the two seams the session calls. They once hooked ``field_split_inverse`` and
``refactor_block``, which the session stopped calling when the probe was laid out by the split's
blocks; the capture then ran a whole march and saved nothing, so the run refuses to finish quietly
without a capture.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.precondition import FieldGroups  # noqa: E402
from aquaflux.precondition import field_split as field_split_module  # noqa: E402

STORE = Path(os.environ.get("PITZ_REFIT_STORE", HERE / "refit_matrices"))
MODE = os.environ.get("PITZ_REFIT_MODE", "profile")
CAPTURE = int(os.environ.get("PITZ_REFIT_CAPTURE", "3"))
REPEATS = int(os.environ.get("PITZ_REFIT_REPEATS", "5"))
#: Functions whose cumulative time the profile reports, by name as cProfile records it.
PARTS = (
    "refactor_blocks",
    "refactor_block",
    "fit",
    "fit_levels",
    "fine_diagonal",
    "_rebuild",
    "_galerkin_coarse",
    "_build_aggregation_hierarchy",
    "_mis_aggregate",
    "from_scipy",
    "_padded_rows",
    "_cell_block_inverse",
    "_diagonal_inverse_operator",
    "_dense_inverse",
    "derive_extras",
    "_simple_pieces",
    "block_approximate_inverse",
    "_diagonal_approximate_inverse",
    "to_scipy",
    "_structured",
    "store",
    "symmetrically_equilibrate",
    "_set_coupling",
    "blocks",
    "apply",
)


class _Stop(Exception):
    """Raised by the capture hook once enough refits are saved, to end the march early."""


#: The four blocks a refit hands the split, in ``FieldGroups.blocks`` order; the second is never read.
BLOCK_NAMES = ("ll", "lt", "tl", "tt")


def _save(name: str, blocks: tuple[sp.spmatrix, ...]) -> None:
    STORE.mkdir(exist_ok=True)
    for block_name, block in zip(BLOCK_NAMES, blocks, strict=True):
        sp.save_npz(STORE / f"{name}-{block_name}.npz", sp.csr_matrix(block), compressed=False)
    sizes = ", ".join(
        f"{b}: {block.shape} {block.nnz}" for b, block in zip(BLOCK_NAMES, blocks, strict=True)
    )
    print(f"[refit_profile] saved {name}: {sizes}", flush=True)


def _load(name: str) -> tuple[sp.csr_matrix, ...]:
    return tuple(
        sp.load_npz(STORE / f"{name}-{block_name}.npz").tocsr() for block_name in BLOCK_NAMES
    )


def capture() -> None:
    import compare

    build = field_split_module._field_split_of_blocks
    refactor = field_split_module.FieldSplitInverse.refactor_blocks
    count = {"refits": 0}

    def recording_build(blocks, groups, **inverses):
        _save("fit", blocks)
        np.save(
            STORE / "groups.npy",
            np.array([groups.n_cells, groups.n_leading_fields, groups.n_fields]),
        )
        return build(blocks, groups, **inverses)

    def recording_refactor(self, blocks):
        _save(f"refit-{count['refits']}", blocks)
        count["refits"] += 1
        if count["refits"] >= CAPTURE:
            raise _Stop
        return refactor(self, blocks)

    field_split_module._field_split_of_blocks = recording_build
    field_split_module.FieldSplitInverse.refactor_blocks = recording_refactor
    try:
        compare.solve_aquaflux(checkpoint_dir=None)
    except _Stop:
        print(f"[refit_profile] captured the fit and {CAPTURE} refits", flush=True)
        return
    raise SystemExit(
        f"[refit_profile] the march ended after {count['refits']} refits, short of {CAPTURE}: "
        "nothing reached the hooks the capture needs."
    )


def _split(fit: tuple[sp.csr_matrix, ...]):
    import compare

    spec = compare.SOLVER.preconditioner.inverse
    n_cells, n_leading, n_fields = (int(v) for v in np.load(STORE / "groups.npy"))
    groups = FieldGroups.by_counts(n_cells, n_leading, n_fields - n_leading)

    def bound(block_inverse):
        stored = spec.precision if block_inverse._stores_precision else "float64"
        return block_inverse.bound(stored_precision=stored)

    print(f"[refit_profile] split held in {spec.precision}", flush=True)
    return field_split_module._field_split_of_blocks(
        fit,
        groups,
        leading_inverse=bound(spec.leading),
        trailing_inverse=bound(spec.trailing),
        stored_precision=spec.precision,
    )


def profile() -> None:
    started = time.perf_counter()
    split = _split(_load("fit"))
    print(f"first fit: {time.perf_counter() - started:.2f} s", flush=True)
    names = sorted({path.name.rsplit("-", 1)[0] for path in STORE.glob("refit-*-ll.npz")})
    for name in names:
        blocks = _load(name)
        profiler = cProfile.Profile()
        profiler.enable()
        split.refactor_blocks(blocks)
        profiler.disable()
        stats = pstats.Stats(profiler, stream=io.StringIO())
        print(f"\n[{name}] cumulative seconds by part (nested parts overlap)")
        rows = {}
        for (_, _, function), (_, calls, _, cumulative, _) in stats.stats.items():
            if function in PARTS:
                total_calls, total = rows.get(function, (0, 0.0))
                rows[function] = (total_calls + calls, total + cumulative)
        for name in PARTS:
            if name in rows:
                calls, cumulative = rows[name]
                print(f"  {name:<34s} {cumulative:8.3f} s  ({calls} calls)")
        listing = io.StringIO()
        pstats.Stats(profiler, stream=listing).sort_stats("tottime").print_stats(15)
        print(listing.getvalue())
        samples = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            split.refactor_blocks(blocks)
            samples.append(time.perf_counter() - started)
        print(
            f"[{name}] refactor_blocks: best {min(samples):.3f} s, "
            f"median {float(np.median(samples)):.3f} s over {REPEATS}",
            flush=True,
        )


if __name__ == "__main__":
    {"capture": capture, "profile": profile}[MODE]()
