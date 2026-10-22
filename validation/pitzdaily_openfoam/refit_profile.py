"""Where does one refit of the shipped field split spend its host-side refactor?

A refit probes the coupled Jacobian, shifts it, and re-fits the field split to it through
``FieldSplitInverse.refactor_block``. The march log's ``pc`` lines split a refit into probe and refactor;
this harness splits the refactor itself, on the matrices the shipped march actually refits to.

Two modes, chosen by ``PITZ_REFIT_MODE``:

``capture``
    Runs the shipped march (``compare.py``'s ``solve_aquaflux``) and saves the shifted field-major
    matrix of the first fit and of the first ``PITZ_REFIT_CAPTURE`` (default 3) refits to
    ``refit_matrices/``, then stops the march. Run with ``validation/run_case.sh``.
``profile`` (the default)
    Loads those matrices, fits the case file's field split to the first, then re-fits it to each captured
    refit matrix under ``cProfile`` and prints the cumulative time of the refactor's parts: the products,
    the padded-row packing, the smoother set-up per level, the spectral estimates, the coarse inverse.
    Then times ``PITZ_REFIT_REPEATS`` (default 5) plain refits per matrix, best and median.

The split is fitted at the case file's settings (``case.yaml``'s ``FieldSplit``), so the profile is of
the shipped preconditioner, in float64 (the float32 rounding happens on the traced split, after the
refactor, and is not part of ``refactor_block``).
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
    "refactor_block",
    "refit",
    "_rebuild",
    "_galerkin_coarse",
    "_build_aggregation_hierarchy",
    "_mis_aggregate",
    "_sparse_level",
    "from_scipy",
    "_padded_rows",
    "_cell_block_inverse",
    "_diagonal_inverse_operator",
    "_spectral_radius",
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


def _save(name: str, matrix: sp.spmatrix) -> None:
    STORE.mkdir(exist_ok=True)
    sp.save_npz(STORE / f"{name}.npz", sp.csr_matrix(matrix), compressed=False)
    print(f"[refit_profile] saved {name}: {matrix.shape}, {matrix.nnz} stored", flush=True)


def capture() -> None:
    import compare

    build = field_split_module.field_split_inverse
    refactor = field_split_module.FieldSplitInverse.refactor_block
    count = {"refits": 0}

    def recording_build(matrix, groups, **inverses):
        _save("fit", matrix)
        np.save(
            STORE / "groups.npy",
            np.array([groups.layout.n_cells, groups.n_leading_fields, groups.layout.n_fields]),
        )
        return build(matrix, groups, **inverses)

    def recording_refactor(self, block):
        _save(f"refit-{count['refits']}", block)
        count["refits"] += 1
        if count["refits"] >= CAPTURE:
            raise _Stop
        return refactor(self, block)

    field_split_module.field_split_inverse = recording_build
    field_split_module.FieldSplitInverse.refactor_block = recording_refactor
    import aquaflux.precondition.materialized_session as session_module

    session_module.field_split_inverse = recording_build
    try:
        compare.solve_aquaflux(checkpoint_dir=None)
    except _Stop:
        print(f"[refit_profile] captured the fit and {CAPTURE} refits", flush=True)


def _split(fit: sp.csr_matrix):
    import compare

    spec = compare.SOLVER.preconditioner.inverse
    n_cells, n_leading, n_fields = (int(v) for v in np.load(STORE / "groups.npy"))
    groups = FieldGroups.by_counts(n_cells, n_leading, n_fields - n_leading)
    return field_split_module.field_split_inverse(
        fit, groups, leading_inverse=spec.leading, trailing_inverse=spec.trailing
    )


def profile() -> None:
    fit = sp.load_npz(STORE / "fit.npz").tocsr()
    refits = sorted(STORE.glob("refit-*.npz"))
    started = time.perf_counter()
    split = _split(fit)
    print(f"first fit: {time.perf_counter() - started:.2f} s", flush=True)
    for path in refits:
        matrix = sp.load_npz(path).tocsr()
        profiler = cProfile.Profile()
        profiler.enable()
        split.refactor_block(matrix)
        profiler.disable()
        stats = pstats.Stats(profiler, stream=io.StringIO())
        print(f"\n[{path.stem}] cumulative seconds by part (nested parts overlap)")
        rows = {}
        for (_, _, name), (_, calls, _, cumulative, _) in stats.stats.items():
            if name in PARTS:
                total_calls, total = rows.get(name, (0, 0.0))
                rows[name] = (total_calls + calls, total + cumulative)
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
            split.refactor_block(matrix)
            samples.append(time.perf_counter() - started)
        print(
            f"[{path.stem}] refactor_block: best {min(samples):.3f} s, "
            f"median {float(np.median(samples)):.3f} s over {REPEATS}",
            flush=True,
        )


if __name__ == "__main__":
    {"capture": capture, "profile": profile}[MODE]()
