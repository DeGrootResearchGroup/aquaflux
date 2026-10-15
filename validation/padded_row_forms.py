"""Which form of a sparse product is fastest on this machine, for the operators a field split applies.

The field split's level operators and coupling pieces are padded-row (ELLPACK) operators: every row padded
to one width, applied as a gather, a multiply and a row sum. How XLA compiles that row sum differs between
platforms -- on one it may be split across threads, on another it may run on one -- and the padded form
only beats a compressed-sparse-row (CSR) kernel where it is. This times, on the REAL operators of a split,
each form the product could take, forward and transposed (the adjoint's direction), in float64 and
float32, and prints what XLA compiled for each (its top-level fusions and how each is split across
threads), so a timing difference between two machines can be traced to a compilation difference.

The forms:

``rows``
    The shipped form: ``(n_rows, width)`` arrays, ``sum(values * x[columns], axis=1)``.
``columns_major``
    The same arrays stored ``(width, n_rows)`` and summed over the leading axis -- the layout a GPU
    ELLPACK kernel uses, and a reduction over the outer dimension rather than the inner one.
``chunked``
    The width padded to a multiple of ``CHUNK`` and summed in two stages, chunk then chunks.
``einsum``
    The row sum written as a batched dot product, ``einsum('rw,rw->r', values, x[columns])``.
``halving``
    The row sum folded in half repeatedly, the halves added, with no reduction operation (only
    slices and elementwise adds), so no library reduction kernel can be chosen for it.
``segment``
    A coordinate-form gather and sorted segment sum, the form the intergrid transfers use.
``csr``
    ``jax.experimental.sparse.BCSR``, the levels' form before the padded rows.

Each form's transpose is taken with ``jax.linear_transpose``, as the adjoint's solve takes it. Two more
transposed arms apply a STORED padded-row copy of the transposed matrix as a forward product
(``stored_rows``, ``stored_columns_major``): what a custom transpose rule would run instead of the
scatter-add ``linear_transpose`` produces.

The operators come from a ``.npz`` written by :func:`save_operators`, which both cases'
``split_apply_timing.py`` call when ``PITZ_SPLIT_TIMING_OPERATORS`` / ``BFS3D_SPLIT_TIMING_OPERATORS`` names
a path. It holds every padded-row operator of the split, plus each multigrid level's prolongation and its
transpose (the restriction), so the intergrid transfers are timed too.

Settings (environment)::

    PADDED_ROW_FORMS_OPERATORS   the .npz to read (required)
    PADDED_ROW_FORMS_FORMS       comma-separated subset of the forms above (default: all)
    PADDED_ROW_FORMS_PRECISIONS  float64,float32 (default: both)
    PADDED_ROW_FORMS_MIN_NNZ     skip operators with fewer stored entries (default 10000)
    PADDED_ROW_FORMS_REPEATS     timings per arm; the best is reported (default 20)
    PADDED_ROW_FORMS_CHUNK       the chunked form's chunk width (default 8)

Usage
-----
    PITZ_SPLIT_TIMING_OPERATORS=/tmp/pitz-ops.npz \\
        validation/run_case.sh validation/pitzdaily_openfoam/split_apply_timing.py --wait
    PADDED_ROW_FORMS_OPERATORS=/tmp/pitz-ops.npz \\
        validation/run_case.sh validation/padded_row_forms.py --wait

⚠️ The totals sum each operator's time ONCE. A V-cycle applies its finest operators several times (a
smoother sweep each, a residual) and a coarse one rarely, so the totals rank the forms and do not predict
a split apply; ``split_apply_timing.py`` measures that.
"""

from __future__ import annotations

import os
import re
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import aquaflux  # noqa: E402  (enables x64)
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.precondition.multigrid import _PaddedRowOperator, halving_row_sum  # noqa: E402
from jax.experimental.sparse import BCSR  # noqa: E402

CHUNK = int(os.environ.get("PADDED_ROW_FORMS_CHUNK", "8"))
REPEATS = int(os.environ.get("PADDED_ROW_FORMS_REPEATS", "20"))
MIN_NNZ = int(os.environ.get("PADDED_ROW_FORMS_MIN_NNZ", "10000"))


# ---------------------------------------------------------------------------------------------------
# Collecting a split's operators


def save_operators(split, path: str) -> None:
    """Write every padded-row operator in ``split`` and every level's prolongation and restriction.

    Each matrix is stored as its CSR arrays under ``<name>/data``, ``/indices``, ``/indptr``, ``/shape``;
    a name is the operator's path in the split's pytree.
    """
    arrays = {}

    def store(name, matrix):
        matrix = sp.csr_matrix(matrix)
        arrays[f"{name}/data"] = matrix.data
        arrays[f"{name}/indices"] = matrix.indices
        arrays[f"{name}/indptr"] = matrix.indptr
        arrays[f"{name}/shape"] = np.asarray(matrix.shape)

    def is_leaf(node):
        return isinstance(node, _PaddedRowOperator) or hasattr(node, "prolongation_scipy")

    seen = set()
    for key_path, node in _containers(split, is_leaf):
        name = jax.tree_util.keystr(key_path)
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, _PaddedRowOperator):
            store(name, node.to_scipy())
        elif node.prolongation_scipy() is not None:  # the coarsest level has none
            prolongation = sp.csr_matrix(node.prolongation_scipy())
            store(f"{name}.prolongation", prolongation)
            store(f"{name}.restriction", prolongation.T.tocsr())
    np.savez(path, **arrays)
    print(f"wrote {len(arrays) // 4} operators to {path}", flush=True)


def in_precision(tree, dtype):
    """``tree`` with every floating-point array leaf cast to ``dtype``; integer index arrays unchanged."""
    return jax.tree.map(
        lambda leaf: (
            leaf.astype(dtype)
            if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)
            else leaf
        ),
        tree,
    )


def _containers(tree, is_leaf):
    """Every node ``is_leaf`` accepts, with its path -- including those nested inside an accepted node."""
    found = []

    def visit(path, node):
        if is_leaf(node):
            found.append((path, node))
        children = jax.tree_util.tree_flatten_with_path(
            node, is_leaf=lambda child: child is not node and is_leaf(child)
        )[0]
        for child_path, child in children:
            if child is not node and is_leaf(child):
                visit(path + child_path, child)

    visit((), tree)
    return found


def _load(path):
    with np.load(path) as data:
        names = sorted({key.rsplit("/", 1)[0] for key in data.files})
        return {
            name: sp.csr_matrix(
                (data[f"{name}/data"], data[f"{name}/indices"], data[f"{name}/indptr"]),
                shape=tuple(int(s) for s in data[f"{name}/shape"]),
            )
            for name in names
        }


# ---------------------------------------------------------------------------------------------------
# The forms: each is (prepare: csr -> arrays, apply: (arrays, x) -> A x)


def _padded(a: sp.csr_matrix, dtype):
    operator = _PaddedRowOperator.from_scipy(a)
    return operator.columns, operator.values.astype(dtype)


def _rows_prepare(a, dtype):
    columns, values = _padded(a, dtype)
    return {"columns": columns, "values": values}


def _rows_apply(arrays, x):
    return jnp.sum(arrays["values"] * x[arrays["columns"]], axis=1)


def _columns_major_prepare(a, dtype):
    columns, values = _padded(a, dtype)
    return {"columns": columns.T.copy(), "values": values.T.copy()}


def _columns_major_apply(arrays, x):
    return jnp.sum(arrays["values"] * x[arrays["columns"]], axis=0)


def _chunked_prepare(a, dtype):
    columns, values = _padded(a, dtype)
    extra = (-columns.shape[1]) % CHUNK
    columns = jnp.pad(columns, ((0, 0), (0, extra)))
    values = jnp.pad(values, ((0, 0), (0, extra)))
    shape = (columns.shape[0], columns.shape[1] // CHUNK, CHUNK)
    return {"columns": columns.reshape(shape), "values": values.reshape(shape)}


def _chunked_apply(arrays, x):
    return jnp.sum(jnp.sum(arrays["values"] * x[arrays["columns"]], axis=2), axis=1)


def _einsum_apply(arrays, x):
    return jnp.einsum("rw,rw->r", arrays["values"], x[arrays["columns"]])


def _halving_apply(arrays, x):
    return halving_row_sum(arrays["values"] * x[arrays["columns"]])


def _segment_prepare(a, dtype):
    coo = a.tocoo()
    return {
        "row": jnp.asarray(coo.row.astype(np.int32)),
        "col": jnp.asarray(coo.col.astype(np.int32)),
        "val": jnp.asarray(coo.data.astype(dtype)),
        "n": a.shape[0],
    }


def _segment_apply(arrays, x):
    return jax.ops.segment_sum(
        arrays["val"] * x[arrays["col"]], arrays["row"], arrays["n"], indices_are_sorted=True
    )


def _csr_prepare(a, dtype):
    return {"matrix": BCSR.from_scipy_sparse(sp.csr_matrix(a, dtype=dtype))}


def _csr_apply(arrays, x):
    return arrays["matrix"] @ x


FORMS = {
    "rows": (_rows_prepare, _rows_apply),
    "columns_major": (_columns_major_prepare, _columns_major_apply),
    "chunked": (_chunked_prepare, _chunked_apply),
    "einsum": (_rows_prepare, _einsum_apply),
    "halving": (_rows_prepare, _halving_apply),
    "segment": (_segment_prepare, _segment_apply),
    "csr": (_csr_prepare, _csr_apply),
}
# Transposed arms applying a stored copy of the transposed matrix as a forward product.
STORED_TRANSPOSES = {
    "stored_rows": "rows",
    "stored_columns_major": "columns_major",
    "stored_halving": "halving",
}
STATIC = ("n",)


# ---------------------------------------------------------------------------------------------------
# Timing and compiled structure


def _split_static(arrays):
    return {k: v for k, v in arrays.items() if k not in STATIC}, {
        k: v for k, v in arrays.items() if k in STATIC
    }


def _compiled(apply, arrays, x, transpose):
    dynamic, static = _split_static(arrays)

    def forward(dynamic, x):
        return apply({**dynamic, **static}, x)

    if transpose:

        def run(dynamic, y):
            return jax.linear_transpose(partial(forward, dynamic), x)(y)[0]

    else:
        run = forward
    return jax.jit(run), dynamic


def _best(f, *args):
    jax.block_until_ready(f(*args))
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(f(*args))
        best = min(best, time.perf_counter() - started)
    return best * 1e3


_ENTRY_FUSION = re.compile(r"=\s*\S+\s+(fusion|custom-call|scatter|gather)\((.*)$")


def _structure(f, *args) -> str:
    """The compiled entry computation's heavy instructions: fusion kind and thread partitions."""
    text = f.lower(*args).compile().as_text()
    entry = text[text.rfind("ENTRY") :]
    parts = []
    for line in entry.splitlines():
        match = _ENTRY_FUSION.search(line)
        if not match:
            continue
        op, rest = match.groups()
        if op == "custom-call":
            target = re.search(r'custom_call_target="([^"]+)"', rest)
            parts.append(f"call:{target.group(1) if target else '?'}")
            continue
        kind = re.search(r"kind=(k\w+)", rest)
        label = kind.group(1)[1:] if kind else op
        custom = re.search(r'"kind":"(__\w+)"', rest)
        if custom:
            label = custom.group(1).strip("_")
        partitions = re.search(r'"outer_dimension_partitions":\[([^\]]*)\]', rest)
        if partitions is not None:
            label += f"[{partitions.group(1).replace(chr(34), '') or '-'}]"
        parts.append(label)
    return " + ".join(parts) or "?"


def main():
    print(
        f"[configuration] aquaflux {Path(aquaflux.__file__).parents[1]}, jax {jax.__version__}, "
        f"{jax.default_backend()}, {os.cpu_count()} cpus, best of {REPEATS}, chunk {CHUNK}, "
        f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}",
        flush=True,
    )
    path = os.environ["PADDED_ROW_FORMS_OPERATORS"]
    forms = os.environ.get("PADDED_ROW_FORMS_FORMS", ",".join(FORMS)).split(",")
    precisions = os.environ.get("PADDED_ROW_FORMS_PRECISIONS", "float64,float32").split(",")
    operators = {name: a for name, a in _load(path).items() if a.nnz >= MIN_NNZ}
    print(f"operators: {path}, {len(operators)} with at least {MIN_NNZ} entries", flush=True)
    for name, a in operators.items():
        widths = np.diff(a.indptr)
        print(
            f"  {name}: {a.shape[0]}x{a.shape[1]}, {a.nnz} entries, widest {widths.max()}, "
            f"mean {widths.mean():.1f}, padding {a.shape[0] * widths.max() / max(a.nnz, 1):.2f}x",
            flush=True,
        )

    rng = np.random.default_rng(0)
    totals = {}
    for precision in precisions:
        dtype = jnp.dtype(precision)
        for name, a in operators.items():
            x = jnp.asarray(rng.standard_normal(a.shape[1]), dtype=dtype)
            y = jnp.asarray(rng.standard_normal(a.shape[0]), dtype=dtype)
            reference_forward = a @ np.asarray(x, dtype=np.float64)
            reference_transposed = a.T @ np.asarray(y, dtype=np.float64)
            print(f"\n{precision} {name}", flush=True)
            arms = [(form, form, False) for form in forms if form in FORMS]
            arms += [(form, form, True) for form in forms if form in FORMS]
            arms += [
                (stored, base, "stored")
                for stored, base in STORED_TRANSPOSES.items()
                if base in forms
            ]
            for label, base, transpose in arms:
                prepare, apply = FORMS[base]
                matrix = a.T.tocsr() if transpose == "stored" else a
                arrays = prepare(matrix, dtype)
                f, dynamic = _compiled(apply, arrays, x, transpose is True)
                argument = y if transpose else x
                reference = reference_transposed if transpose else reference_forward
                try:
                    ms = _best(f, dynamic, argument)
                    out = np.asarray(f(dynamic, argument), dtype=np.float64)
                    error = np.max(np.abs(out - reference)) / max(np.max(np.abs(reference)), 1e-300)
                    structure = _structure(f, dynamic, argument)
                except Exception as exc:  # a form a backend cannot compile is a result, not a crash
                    print(
                        f"  {label:22s} {'T' if transpose else 'F'}  FAILED: {exc!r}"[:200],
                        flush=True,
                    )
                    continue
                direction = "transposed" if transpose else "forward"
                totals[(precision, direction, label)] = (
                    totals.get((precision, direction, label), 0) + ms
                )
                print(
                    f"  {label:22s} {'T' if transpose else 'F'} {ms:9.3f} ms  err {error:.1e}  {structure}",
                    flush=True,
                )

    print("\ntotals (each operator once; ranks forms, does not predict a split apply)", flush=True)
    for (precision, direction, label), ms in sorted(totals.items()):
        print(f"  {precision:8s} {direction:10s} {label:22s} {ms:10.2f} ms", flush=True)


if __name__ == "__main__":
    main()
