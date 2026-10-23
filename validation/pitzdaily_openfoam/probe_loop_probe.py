"""The coloured probe run as one compiled loop over its chunks, against one compiled call per chunk.

A refit's probe calls the batched Jacobian-vector product once per chunk of ``PROBE_BATCH_SIZE``
directions. On Linux each call's temporary buffer is mapped afresh and page-faulted (above 64 MiB the
glibc thresholds cannot stop it, since XLA allocates on worker threads), so a loop inside ONE program,
which allocates once, might remove that cost. This times, at the last captured state of a
``replay.py`` capture, on the step's own probe plan and gather:

* ``shipped``            -- the probe as the session runs it (device seeds, one call per chunk, device
  scatter);
* ``looped``             -- one ``fori_loop`` program, seeds in and responses out by host callbacks, so the
  memory bound stays one chunk;
* ``compute-only loop``  -- the same loop over precomputed seeds with no host work, and
* ``calls-only``         -- one call per chunk over the same seeds with no host work,

each the best of ``PITZ_COST_REPEATS`` with its minor faults, and whether ``looped`` reproduces ``shipped``.
Run it under the default allocator and under ``MALLOC_ARENA_MAX=1`` (with the thresholds raised) to
separate the faults' cost from the loop's own.

Usage
-----
    validation/run_case.sh validation/pitzdaily_openfoam/probe_loop_probe.py --wait
"""

import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import aquaflux  # noqa: E402,F401  (enables x64)
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.experimental  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    PROBE_BATCH_SIZE,
    batched_jacobian_matvec,
)
from replay import MarchReplay  # noqa: E402

REPEATS = int(os.environ.get("PITZ_COST_REPEATS", "5"))
replay = MarchReplay()
last = max(replay.states)
state = jnp.asarray(replay.states[last - 1]["state"])
assembler = replay.assembler_for(last)
probe = replay.session._probe_for()
probed = probe.narrow(assembler)
plan, structure = probe.plan, probe.structure
chunk, n_probes, nf = PROBE_BATCH_SIZE, plan.n_probes, plan.n_fields * plan.n_cells
n_chunks = -(-n_probes // chunk)
print(
    f"[configuration] {n_probes} probes, chunk {chunk}, {n_chunks} chunks, nf {nf}, nnz {structure.nnz}; "
    f"MALLOC_ARENA_MAX={os.environ.get('MALLOC_ARENA_MAX', 'unset')} LD_PRELOAD={os.environ.get('LD_PRELOAD', 'unset')}",
    flush=True,
)


CHUNKS = [(0, start, min(chunk, n_probes - start)) for start in range(0, n_probes, chunk)]
WIDTH = structure.width(CHUNKS)
# The looped arm places responses on the host, inside its callback, from host copies of the order.
SOURCE, POSITION = np.asarray(structure._source), np.asarray(structure._position)
STARTS = structure._probe_start


def shipped():
    data = jnp.zeros(structure.nnz)
    for _, start, m in CHUNKS:
        block = plan.seed_block(start, start + m, rows=chunk)
        responses = batched_jacobian_matvec(probed, state, block)
        data = structure.scatter(data, responses, start, m, WIDTH)
    return np.asarray(data)


def host_seeds(start, stop, out):
    """``plan.seed_block``'s indicators written on the host, for the callback (which may not call JAX)."""
    n = plan.n_cells
    out[:] = 0.0
    for row, index in enumerate(range(start, stop)):
        field, colour = plan.probe(index)
        out[row, field * n : (field + 1) * n] = plan.colour[field] == colour
    return out


class _Host:
    """Host side of the looped probe: seeds out, responses in."""

    def __init__(self):
        self.block = np.zeros((chunk, nf))
        self.data = None

    def seeds(self, index):
        start = int(index) * chunk
        return host_seeds(start, min(start + chunk, n_probes), self.block)

    def take(self, index, responses):
        start = int(index) * chunk
        lo, hi = STARTS[start], STARTS[min(start + chunk, n_probes)]
        self.data[POSITION[lo:hi]] = np.asarray(responses).reshape(-1)[
            SOURCE[lo:hi].astype(np.int64) - start * nf
        ]


HOST = _Host()


@eqx.filter_jit
def _looped(assembler, state):
    shape = jax.ShapeDtypeStruct((chunk, nf), state.dtype)

    def body(index, token):
        block = jax.pure_callback(HOST.seeds, shape, index)
        responses = jax.vmap(lambda t: jax.jvp(assembler.residual, (state,), (t,))[1])(block)
        jax.experimental.io_callback(HOST.take, None, index, responses, ordered=True)
        return token

    return jax.lax.fori_loop(0, n_chunks, body, 0)


def looped():
    HOST.data = np.zeros(structure.nnz)
    jax.block_until_ready(_looped(probed, state))
    return HOST.data


def timed(f):
    f()
    best, faults = float("inf"), 0
    for _ in range(REPEATS):
        before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        t = time.perf_counter()
        f()
        dt = time.perf_counter() - t
        if dt < best:
            best, faults = dt, resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before
    return best, faults


ALL = plan.seed_block(0, n_probes, rows=n_chunks * chunk).reshape(n_chunks, chunk, nf)


@eqx.filter_jit
def _compute_only(assembler, state, seeds):
    def body(index, total):
        responses = jax.vmap(lambda t: jax.jvp(assembler.residual, (state,), (t,))[1])(seeds[index])
        return total + jnp.sum(responses)

    return jax.lax.fori_loop(0, n_chunks, body, 0.0)


def compute_only():
    jax.block_until_ready(_compute_only(probed, state, ALL))


def calls_only():
    for index in range(n_chunks):
        jax.block_until_ready(batched_jacobian_matvec(probed, state, ALL[index]))


a = shipped()
b = looped()
print(
    f"bit-identical: {np.array_equal(a, b)}; max abs diff {np.max(np.abs(a - b)):.3e}", flush=True
)
for name, f in (
    ("shipped", shipped),
    ("looped", looped),
    ("compute-only loop", compute_only),
    ("calls-only, no host work", calls_only),
    ("shipped", shipped),
    ("looped", looped),
    ("compute-only loop", compute_only),
    ("calls-only, no host work", calls_only),
):
    s, flt = timed(f)
    print(f"{name}: {s:.3f} s, {flt} minor faults ({flt / n_chunks:.0f} per chunk)", flush=True)
