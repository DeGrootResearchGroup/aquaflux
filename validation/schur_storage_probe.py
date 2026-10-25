"""What holding the SIMPLE smoother's Schur complement costs, and what not holding it costs instead.

The SIMPLE smoother applies ``S = C - D (M G)`` from its pieces and fits only ``S``'s diagonal and row norms
(``simple_fit._schur_rows``), a chunk of rows at a time on ``S``'s row columns. This harness puts numbers on
the alternatives on a saddle shaped like bfs3d's leading block -- a reach-3 hexahedral lattice of
``PROBE_SCHUR_SHAPE`` cells (default ``40x24x24``, bfs3d's 23040), four fields, every field pair storing the
reach-3 cell pattern, random values with a dominant velocity and pressure diagonal -- and prints:

- the entry counts of ``S`` formed and of its pieces, and the bytes each would hold in single precision;
- one product with ``S`` formed (packed from the host product) against the pieces' ``schur_apply``, forward
  and transposed (mean of ``PROBE_SCHUR_CALLS`` calls, three repeats), and how far apart they are; and the
  same products ``LOOPED`` times inside one compiled loop, as the smoother's sweeps run them. ⚠️ The
  isolated call overstates the formed product: its per-row gather (``n x width`` single-precision values,
  33 MiB on the default lattice) is above glibc's 32 MiB mapping threshold, so each standalone call
  faults it in afresh until the threshold adapts, which a loop allocating once does not pay;
- the whole level fit, and the diagonal and squared row norms formed instead by sorting each row chunk's
  terms, with no pattern of ``S`` held -- the alternative that would drop ``S``'s row columns too.

The quantity depends on the pattern's shape, not on a flow, so no mesh, state or solve is needed. Run it
with ``validation/run_case.sh validation/schur_storage_probe.py --wait``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.precondition.entry_map import SparsePattern  # noqa: E402
from aquaflux.precondition.padded_rows import _PaddedRowOperator  # noqa: E402
from aquaflux.precondition.saddle_multigrid import _simple_pieces  # noqa: E402
from aquaflux.precondition.simple_fit import fit_simple_level, simple_plan  # noqa: E402
from aquaflux.precondition.traced_cycle import in_precision  # noqa: E402
from aquaflux.solve import with_transposes  # noqa: E402

SHAPE = tuple(int(v) for v in os.environ.get("PROBE_SCHUR_SHAPE", "40x24x24").split("x"))
CALLS = int(os.environ.get("PROBE_SCHUR_CALLS", "50"))
#: Terms sorted at once by the sort-based row norms.
SORT_TERMS = 1 << 18
#: Products per compiled loop in the looped timing.
LOOPED = 40
SPLITTING = {"frobenius": True, "schur_frobenius": True, "block_splitting": True}


def lattice_saddle(shape: tuple[int, int, int], seed: int = 0) -> sp.csr_matrix:
    """A field-major ``[u, v, w, p]`` saddle storing the reach-3 cell pattern in every field pair."""
    n = int(np.prod(shape))
    index = np.arange(n).reshape(shape)
    owner, neighbour = [], []
    for axis in range(3):
        a = np.take(index, np.arange(shape[axis] - 1), axis=axis).ravel()
        b = np.take(index, np.arange(1, shape[axis]), axis=axis).ravel()
        owner += [a, b]
        neighbour += [b, a]
    adjacency = sp.csr_matrix(
        (np.ones(sum(map(len, owner))), (np.concatenate(owner), np.concatenate(neighbour))), (n, n)
    )
    step = (sp.identity(n) + adjacency).tocsr()
    reach = (step @ step @ step).tocsr()
    reach.data[:] = 1.0
    rng = np.random.default_rng(seed)
    blocks = []
    for f in range(4):
        row = []
        for g in range(4):
            block = reach.copy()
            block.data = 0.1 * rng.standard_normal(block.nnz)
            if f == g:
                block = block + (20.0 if f < 3 else -20.0) * sp.identity(n)
            row.append(block)
        blocks.append(row)
    return sp.bmat(blocks, format="csr").sorted_indices()


def mean_ms(fn, *args) -> float:
    """Mean milliseconds of ``CALLS`` calls after one warm call."""
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    for _ in range(CALLS):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / CALLS * 1e3


def mib(operator: _PaddedRowOperator) -> str:
    """An operator's slot values, in MiB (a transpose, packed only for an adjoint, doubles it)."""
    return f"{operator.values.nbytes / 2**20:.1f} MiB"


def sorted_row_statistics(n, divergence, correction, pressure):
    """``S``'s diagonal and squared row norms with no pattern of ``S``: each row chunk's terms sorted by
    (row, column), equal keys summed, then reduced. Double precision."""
    d_columns, d_values, d_lengths = divergence
    m_columns, m_values, m_lengths = correction
    c_columns, c_values, c_lengths = pressure
    wd, wm, wc = d_columns.shape[1], m_columns.shape[1], c_columns.shape[1]
    rows = max(1, SORT_TERMS // (wd * wm + wc))
    chunks = -(-n // rows)
    extra = chunks * rows - n

    def padded(array):
        return jnp.concatenate([array, jnp.zeros((extra, *array.shape[1:]), array.dtype)])

    d_columns, d_values, d_lengths, c_columns, c_values, c_lengths = map(
        padded, (d_columns, d_values, d_lengths, c_columns, c_values, c_lengths)
    )
    big = jnp.iinfo(jnp.int32).max
    local = jnp.arange(rows)

    def chunk(index):
        def of(array):
            return jax.lax.dynamic_slice_in_dim(array, index * rows, rows)

        dc, dv, dl, cc, cv, cl = map(
            of, (d_columns, d_values, d_lengths, c_columns, c_values, c_lengths)
        )
        d_stored = jnp.arange(wd)[None] < dl[:, None]
        m_stored = (jnp.arange(wm)[None, None] < m_lengths[dc][..., None]) & d_stored[..., None]
        product_keys = jnp.where(m_stored, local[:, None, None] * n + m_columns[dc], big)
        product_values = jnp.where(m_stored, -dv[..., None] * m_values[dc], 0.0)
        c_stored = jnp.arange(wc)[None] < cl[:, None]
        keys = jnp.concatenate(
            [product_keys.reshape(rows, -1), jnp.where(c_stored, local[:, None] * n + cc, big)], 1
        ).ravel()
        values = jnp.concatenate(
            [product_values.reshape(rows, -1), jnp.where(c_stored, cv, 0.0)], 1
        ).ravel()
        keys, values = jax.lax.sort((keys, values), num_keys=1)
        first = jnp.concatenate([jnp.array([True]), keys[1:] != keys[:-1]])
        segment = jnp.cumsum(first) - 1
        sums = jax.ops.segment_sum(values, segment, num_segments=keys.size, indices_are_sorted=True)
        key = jax.ops.segment_max(keys, segment, num_segments=keys.size, indices_are_sorted=True)
        row = jnp.where(key == big, rows, key // n)
        on_diagonal = (key % n) == row + index * rows
        squared = jax.ops.segment_sum(sums * sums, row, num_segments=rows + 1)[:rows]
        diagonal = jax.ops.segment_sum(
            jnp.where(on_diagonal, sums, 0.0), row, num_segments=rows + 1
        )
        return diagonal[:rows], squared

    diagonal, squared = jax.lax.map(chunk, jnp.arange(chunks))
    return diagonal.ravel()[:n], squared.ravel()[:n]


def main() -> None:
    a = lattice_saddle(SHAPE)
    n = int(np.prod(SHAPE))
    nv = 3 * n
    print(f"lattice {SHAPE}, {n} cells, saddle {a.nnz / 1e6:.1f}M entries, jax {jax.__version__}")

    pieces = _simple_pieces(a, 4, **SPLITTING, report=print, precision="float32")
    m = pieces.f_block_inverse.to_scipy()
    schur = (a[nv:, nv:] - a[nv:, :nv] @ (m @ a[:nv, nv:])).tocsr()
    formed = in_precision(_PaddedRowOperator.from_scipy(schur), "float32")
    for name, operator in (
        ("S formed", formed),
        ("C", pieces.pressure),
        ("D", pieces.divergence),
        ("M G", pieces.dg),
    ):
        print(
            f"  {name}: {operator.nnz / 1e6:.2f}M entries, width {operator.width}, "
            f"values {mib(operator)}"
        )

    x = jnp.asarray(np.random.default_rng(1).standard_normal(n), jnp.float32)
    forward_formed = jax.jit(lambda s, v: s.apply(v))
    forward_pieces = jax.jit(lambda p, v: p.schur_apply(v))
    # A transpose is packed only for an adjoint, so each operator is given its own before it is timed.
    formed_t, pieces_t = with_transposes(formed), with_transposes(pieces)
    transposed_formed = jax.jit(lambda s, v: jax.linear_transpose(s.apply, v)(v)[0])
    transposed_pieces = jax.jit(lambda p, v: jax.linear_transpose(p.schur_apply, v)(v)[0])
    one, other = forward_formed(formed, x), forward_pieces(pieces, x)
    print(
        f"  forward products {float(jnp.linalg.norm(one - other) / jnp.linalg.norm(one)):.1e} apart"
    )
    for repeat in range(3):
        print(
            f"  repeat {repeat}: formed {mean_ms(forward_formed, formed, x):.2f} ms, "
            f"pieces {mean_ms(forward_pieces, pieces, x):.2f} ms; transposed formed "
            f"{mean_ms(transposed_formed, formed_t, x):.2f} ms, pieces "
            f"{mean_ms(transposed_pieces, pieces_t, x):.2f} ms",
            flush=True,
        )

    def looped(apply):
        def run(operator, v):
            def body(_, p):
                return p + 1e-3 * apply(operator, p)

            return jax.lax.fori_loop(0, LOOPED, body, v)

        return jax.jit(run)

    looped_formed = looped(lambda s, v: s.apply(v))
    looped_pieces = looped(lambda p, v: p.schur_apply(v))
    for repeat in range(3):
        per_formed = mean_ms(looped_formed, formed, x) / LOOPED
        per_pieces = mean_ms(looped_pieces, pieces, x) / LOOPED
        print(
            f"  looped repeat {repeat}: formed {per_formed:.2f} ms, pieces {per_pieces:.2f} ms a product",
            flush=True,
        )

    plan = simple_plan(SparsePattern.of(a), 4, **SPLITTING)
    slots = _PaddedRowOperator.from_scipy(a).values
    values, _ = fit_simple_level(plan, slots)
    start = time.perf_counter()
    values, _ = jax.block_until_ready(fit_simple_level(plan, slots))
    print(f"  whole level fit {time.perf_counter() - start:.2f} s")
    divergence = plan.divergence_layout.with_values(values.divergence)
    correction = plan.correction_layout.with_values(values.correction)
    pressure = plan.pressure_layout.with_values(values.pressure)
    by_sort = jax.jit(sorted_row_statistics, static_argnums=0)
    arguments = tuple(
        (op.columns, op.values, op.row_lengths) for op in (divergence, correction, pressure)
    )
    diagonal, squared = jax.block_until_ready(by_sort(n, *arguments))
    start = time.perf_counter()
    diagonal, squared = jax.block_until_ready(by_sort(n, *arguments))
    elapsed = time.perf_counter() - start
    reference = schur.diagonal() / np.asarray(schur.multiply(schur).sum(axis=1)).ravel()
    error = np.max(np.abs(np.asarray(diagonal / squared) - reference) / np.abs(reference))
    print(f"  row norms by sort {elapsed:.2f} s, diagonal inverse {error:.1e} from the host's")


if __name__ == "__main__":
    main()
