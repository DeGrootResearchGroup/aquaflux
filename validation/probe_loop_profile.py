"""A step case's coloured probe run as one compiled loop over its chunks, against one compiled call per chunk.

A re-fit's probe calls the batched Jacobian-vector product (JVP) once per chunk of ``PROBE_BATCH_SIZE``
directions, and on Linux each call's temporary buffer is mapped afresh and page-faulted (glibc returns
any allocation XLA makes from a worker thread above 64 MiB to the kernel when it is freed). A loop inside
ONE program allocates its temporary once per re-fit, which removes the faults at the source rather than
through the process allocator. Whether it is faster also depends on how the same vmapped JVP runs inside
a compiled loop. This times, at one fixed state, on the session's own probe plan and narrowed assembler,
over the same precomputed seeds (so the host's seed filling and scattering, identical in every arm, are
left out):

* ``calls``  -- one compiled call per chunk, each response copied to the host, as the session's probe does;
* ``loop``   -- one ``fori_loop`` program over the chunks, each response handed to the host by an ordered
  callback, so the memory bound stays one chunk;
* ``map``    -- one ``lax.map`` program returning every chunk's responses at once (one more probe-sized
  output held, so not a candidate for the session; it says whether a loop's slowdown is the callback);
* ``loop, no output`` -- the loop reduced to a sum, so neither callback nor output.

Each arm reports the mean and best of ``PROBE_LOOP_REPEATS`` passes after one warm pass, with the cores it
kept busy (processor seconds over wall seconds) and its minor faults, and the looped responses are compared with the calls' once. Run it at glibc defaults and under
``GLIBC_TUNABLES=glibc.malloc.hugetlb=1`` to separate the faults' cost from the loop's own.

Usage::

    PROBE_LOOP_STATE=validation/bfs3d_openfoam/checkpoints/state-00032.npz \\
        validation/run_case.sh validation/probe_loop_profile.py --wait

``PROBE_LOOP_CASE`` is ``bfs3d`` (default) or ``pitzdaily``; ``PROBE_LOOP_STATE`` a ``.npz`` holding
``state`` (default the hybrid start); ``PROBE_LOOP_REPEATS`` the timed passes per arm (default 3).
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from step_cases import load_case  # noqa: E402

CASE = os.environ.get("PROBE_LOOP_CASE", "bfs3d")
STATE = os.environ.get("PROBE_LOOP_STATE")
REPEATS = int(os.environ.get("PROBE_LOOP_REPEATS", "3"))


def _usage() -> tuple[float, int]:
    """This process's processor seconds (user + system) and minor faults so far."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime, usage.ru_minflt


def main() -> None:
    case = load_case(CASE)
    import equinox as eqx
    import jax
    import jax.experimental
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.initialization import hybrid_initialize
    from aquaflux.precondition.materialized_session import (
        PROBE_BATCH_SIZE,
        batched_jacobian_matvec,
    )
    from aquaflux.turbulence import open_session

    coupled = case.build_case()["coupled"]
    if STATE:
        with np.load(STATE) as data:
            state = jnp.asarray(data["state"])
    else:
        state = coupled.state_from_physical(*hybrid_initialize(coupled))
    allocator = " ".join(
        f"{name}={os.environ[name]}"
        for name in sorted(os.environ)
        if name.startswith("MALLOC_") or name in ("LD_PRELOAD", "GLIBC_TUNABLES")
    )
    session = open_session(case.SOLVER.settings()["preconditioner"], coupled)
    session.build(state, dual_time=case.SOLVER.dual_time)
    probe = session._probe_for()
    probed = probe.narrow(session.problem.assembler)
    frozen = jax.lax.stop_gradient(state)
    plan, chunk = probe.plan, PROBE_BATCH_SIZE
    n_probes, nf = plan.n_probes, state.shape[0]
    n_chunks = -(-n_probes // chunk)
    print(
        f"[configuration] case {CASE}, state {STATE or 'the hybrid start'}, jax {jax.__version__}, "
        f"{os.cpu_count()} cpus, {n_probes} probes in {n_chunks} chunks of {chunk}, {nf} unknowns, "
        f"allocator settings {allocator or 'none (glibc defaults)'}",
        flush=True,
    )
    seeds = np.zeros((n_chunks * chunk, nf))
    plan.seed_block(0, n_probes, out=seeds[:n_probes])
    seeds = jnp.asarray(seeds.reshape(n_chunks, chunk, nf))

    def calls():
        return [
            np.asarray(batched_jacobian_matvec(probed, frozen, seeds[i])) for i in range(n_chunks)
        ]

    taken: list = []

    def take(responses):
        taken.append(np.array(responses))

    def responses_of(assembler, state, block):
        return jax.vmap(lambda t: jax.jvp(assembler.residual, (state,), (t,))[1])(block)

    @eqx.filter_jit
    def _loop(assembler, state, seeds):
        def body(index, carry):
            jax.experimental.io_callback(
                take, None, responses_of(assembler, state, seeds[index]), ordered=True
            )
            return carry

        return jax.lax.fori_loop(0, n_chunks, body, 0)

    def loop():
        taken.clear()
        jax.block_until_ready(_loop(probed, frozen, seeds))
        return list(taken)

    @eqx.filter_jit
    def _map(assembler, state, seeds):
        return jax.lax.map(lambda block: responses_of(assembler, state, block), seeds)

    def mapped():
        return np.asarray(_map(probed, frozen, seeds))

    @eqx.filter_jit
    def _summed(assembler, state, seeds):
        def body(index, total):
            return total + jnp.sum(responses_of(assembler, state, seeds[index]))

        return jax.lax.fori_loop(0, n_chunks, body, 0.0)

    def summed():
        return jax.block_until_ready(_summed(probed, frozen, seeds))

    reference = np.concatenate(calls())
    for name, arm in (("loop", loop), ("map", mapped)):
        got = (
            np.concatenate(arm()) if name == "loop" else np.asarray(arm()).reshape(reference.shape)
        )
        scale = np.max(np.abs(reference))
        print(
            f"{name} against calls: max abs difference {np.max(np.abs(got - reference)):.3e} "
            f"(largest response {scale:.3e})",
            flush=True,
        )
    print(
        f"\n| arm ({REPEATS} passes after a warm one) | mean s | best s | cores | faults per pass |"
        "\n|---|---|---|---|---|",
        flush=True,
    )
    for _ in range(2):
        for name, arm in (
            ("calls", calls),
            ("loop", loop),
            ("map", mapped),
            ("loop, no output", summed),
        ):
            arm()
            walls, cpu, faults = [], 0.0, 0
            for _ in range(REPEATS):
                (cpu0, faults0), start = _usage(), time.perf_counter()
                arm()
                walls.append(time.perf_counter() - start)
                cpu1, faults1 = _usage()
                cpu, faults = cpu + cpu1 - cpu0, faults + faults1 - faults0
            print(
                f"| {name} | {np.mean(walls):.2f} | {min(walls):.2f} | {cpu / sum(walls):.2f} | "
                f"{faults / REPEATS / 1e3:.0f}k |",
                flush=True,
            )


if __name__ == "__main__":
    main()
