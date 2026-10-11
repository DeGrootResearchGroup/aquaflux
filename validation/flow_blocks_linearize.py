"""Should the flow saddle's frozen Jacobian blocks be linearized once per iterate?

``aquaflux.flow.block_preconditioner.FlowBlocks`` applies each of its blocks (``F``, ``G``, ``B``, ``Ĉ``) as a
``jax.jvp`` through the frozen residual, so every application of the block preconditioner -- once per
Krylov iteration -- re-runs the residual's primal pass alongside its tangent. ``jax.linearize`` would
run the primal pass once per Newton iterate and keep its intermediates, leaving only the tangent pass
per application, at the cost of holding those intermediates in memory for the whole linear solve.

This measures the trade on the operating point the march actually runs. Two arms, which differ ONLY in
how ``FlowBlocks`` forms a column:

* ``jvp`` -- the shipped class, unchanged;
* ``linearize`` -- a subclass whose ``of`` calls ``jax.linearize`` once and whose column applies the
  stored linear map. It is swapped in for the module's ``FlowBlocks`` for that arm alone, so every other
  line of the preconditioner (its inner solves, its composition) is the shipped code.

Two things about the method are load-bearing:

* **The state is TRACED, as in the march.** ``newton_step`` builds the preconditioner from the iterate
  inside the jitted march step, so the frozen state is a runtime value, not a compile-time constant. A
  harness that closed over a concrete state would let XLA constant-fold the very primal pass this
  question is about. Every compiled function here takes the assembler, the preconditioner and the state
  as arguments.
* **Both arms must give the same preconditioner.** ``jax.linearize`` is the same linear map, so the
  applications are compared to roundoff before anything is timed, and the GMRES cycle counts must agree.

First, why the answer comes out as it does: the bare residual decomposed -- the primal pass alone,
one unlooped ``jvp``, and the per-application cost of a ``jvp`` and of a linearized map inside a
compiled loop, where a loop-invariant primal pass can be hoisted. Then three measurements per
(mesh, composition):

1. per-application cost of ``M``: a compiled loop applying ``M`` ``K`` times from one build, timed at
   ``K = 1`` and ``K = 1 + REPEATS``, the difference divided by ``REPEATS`` (so the build is excluded),
   beside the same loop over a bare residual ``jvp`` -- the outer GMRES matvec -- as the yardstick;
2. a whole right-preconditioned GMRES solve of the real Newton system, compiled with the preconditioner
   built inside it, to true ``rtol`` 1e-8: cycles and wall clock;
3. the compiled solve's temporary-buffer footprint, where the backend reports one.

Run: ``validation/run_case.sh validation/flow_blocks_linearize.py`` sweeps ``MESHES`` in one process
(``simple_type_composition.py``'s channel at ``mu = 4e-4``, marched to rel 1e-3, at two sizes -- a
256x128 mesh's march alone ran past half an hour on a 4-core machine);
``python3 validation/flow_blocks_linearize.py nx ny [mu march_rtol]`` runs one mesh.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from aquaflux.flow import BlockPreconditioner, ConvectionTwoLevel
from aquaflux.flow import block_preconditioner as block_module
from aquaflux.flow.block_preconditioner import FlowBlocks
from aquaflux.solve import relative_residual_gmres
from simple_type_composition import channel, developing

RTOL = 1e-8
RESTART = 30
MAX_RESTARTS = 200
#: Applications per timed loop beyond the first, and timed repeats of each compiled call (median kept).
REPEATS = 30
TIMINGS = 7
ARMS = (("msimple", "triangular"), ("msimple", "simpler"))
MESHES = ((64, 32), (128, 64))
MU = 4e-4
MARCH_RTOL = 1e-3


class LinearizedFlowBlocks(FlowBlocks):
    """``FlowBlocks`` with the residual linearized once at ``of`` instead of per column."""

    linear: object

    @classmethod
    def of(cls, assembler, state):
        assembler = jax.lax.stop_gradient(assembler)
        state = jax.lax.stop_gradient(state)
        _, linear = jax.linearize(assembler.residual, state)
        return cls(assembler, state, linear)

    def _column(self, tangent):
        return self.assembler.unpack(self.linear(tangent))


class _Blocks:
    """Swap the class the preconditioner builds its blocks from, for one arm."""

    def __init__(self, cls):
        self.cls = cls

    def __enter__(self):
        block_module.FlowBlocks = self.cls

    def __exit__(self, *exc):
        block_module.FlowBlocks = FlowBlocks


def _preconditioner(blocks, precon, state):
    """``M`` at ``state``, built with ``blocks`` as the block class.

    ``blocks`` is an argument of every compiled function below, so it is part of the compilation key
    (a class is a static leaf to ``equinox.filter_jit``): the two arms can never share a program.
    """
    with _Blocks(blocks):
        return precon.apply_at(state, precon.frozen_momentum_diagonal(state))


@eqx.filter_jit
def _apply_loop(blocks, precon, state, v, count):
    """``count`` applications of ``M`` from ONE build, renormalized so the iterate stays bounded."""
    m = _preconditioner(blocks, precon, state)

    def body(_, x):
        y = m(x)
        return y / jnp.linalg.norm(y)

    return jax.lax.fori_loop(0, count, body, v)


@eqx.filter_jit
def _jvp_loop(assembler, state, v, count):
    """The same loop over the bare residual ``jvp`` -- the outer GMRES matvec, for scale."""

    def body(_, x):
        y = jax.jvp(assembler.residual, (state,), (x,))[1]
        return y / jnp.linalg.norm(y)

    return jax.lax.fori_loop(0, count, body, v)


@eqx.filter_jit
def _linear_loop(assembler, state, v, count):
    """The bare loop again, over a map linearized once before it -- the tangent pass alone."""
    _, linear = jax.linearize(assembler.residual, state)

    def body(_, x):
        y = linear(x)
        return y / jnp.linalg.norm(y)

    return jax.lax.fori_loop(0, count, body, v)


@eqx.filter_jit
def _residual(assembler, state):
    return assembler.residual(state)


@eqx.filter_jit
def _jvp_once(assembler, state, v):
    return jax.jvp(assembler.residual, (state,), (v,))[1]


@eqx.filter_jit
def _apply_once(blocks, precon, state, v):
    return _preconditioner(blocks, precon, state)(v)


@eqx.filter_jit
def _solve(blocks, assembler, precon, state, b):
    """The real Newton system, right-preconditioned, with ``M`` built from the traced state."""
    m = _preconditioner(blocks, precon, state)

    def matvec(y):
        return jax.jvp(assembler.residual, (state,), (m(y),))[1]

    operator = lx.FunctionLinearOperator(matvec, jax.ShapeDtypeStruct(b.shape, b.dtype))
    solver = relative_residual_gmres(
        RTOL, restart=RESTART, stagnation_iters=40, max_restarts=MAX_RESTARTS
    )
    solution = lx.linear_solve(operator, b, solver=solver, throw=False)
    return m(solution.value), solution.stats["num_steps"]


def _time(fn, *args):
    """Median wall clock of a compiled call (compiled on the first, untimed, call)."""
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(TIMINGS):
        started = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _per_application(loop, *args):
    head = _time(loop, *args, 1)
    tail = _time(loop, *args, 1 + REPEATS)
    return (tail - head) / REPEATS


def _temp_bytes(fn, *args):
    try:
        analysis = fn.lower(*args).compile().compiled.memory_analysis()
    except Exception:  # a backend without the analysis reports nothing
        return None
    return None if analysis is None else analysis.temp_size_in_bytes


def main(nx: int, ny: int, mu: float, march_rtol: float) -> None:
    jax.clear_caches()
    print(
        f"jax {jax.__version__}, {jax.default_backend()}, {platform.machine()}, "
        f"{jax.device_count()} device(s); REPEATS {REPEATS}, median of {TIMINGS}",
        flush=True,
    )
    assembler = channel(nx, ny, mu)
    print(
        f"channel {nx}x{ny} ({nx * ny} cells), mu={mu:g}; marching to rel {march_rtol:.0e}",
        flush=True,
    )
    started = time.perf_counter()
    state = developing(assembler, march_rtol)
    residual = assembler.residual(state)
    print(
        f"state |R| = {float(jnp.linalg.norm(residual)):.3e} ({time.perf_counter() - started:.0f} s)"
    )
    v = jax.random.normal(jax.random.PRNGKey(0), residual.shape)
    v = v / jnp.linalg.norm(v)

    jvp_cost = _per_application(_jvp_loop, assembler, state, v)
    linear_cost = _per_application(_linear_loop, assembler, state, v)
    print(f"residual (primal pass) alone:      {1e3 * _time(_residual, assembler, state):8.3f} ms")
    print(
        f"one jvp, not in a loop:            {1e3 * _time(_jvp_once, assembler, state, v):8.3f} ms"
    )
    print(f"jvp per application in a loop:     {1e3 * jvp_cost:8.3f} ms  (the outer matvec)")
    print(f"linearized map, same loop:         {1e3 * linear_cost:8.3f} ms\n", flush=True)
    header = (
        f"{'schur':<9}{'composition':<12}{'arm':<11}{'M ms/app':>10}{'/jvp':>7}"
        f"{'cycles':>8}{'TRUE rel':>11}{'solve s':>9}{'temp MB':>9}"
    )
    print(header)
    for scaling, composition in ARMS:
        precon = BlockPreconditioner.build(
            assembler, schur_scaling=scaling, composition=composition, velocity=ConvectionTwoLevel()
        )
        applied = {}
        for name, cls in (("jvp", FlowBlocks), ("linearize", LinearizedFlowBlocks)):
            applied[name] = _apply_once(cls, precon, state, v)
            cost = _per_application(_apply_loop, cls, precon, state, v)
            (value, cycles) = _solve(cls, assembler, precon, state, -residual)
            wall = _time(_solve, cls, assembler, precon, state, -residual)
            temp = _temp_bytes(_solve, cls, assembler, precon, state, -residual)
            true_rel = float(
                jnp.linalg.norm(jax.jvp(assembler.residual, (state,), (value,))[1] + residual)
                / jnp.linalg.norm(residual)
            )
            print(
                f"{scaling:<9}{composition:<12}{name:<11}{1e3 * cost:>10.3f}{cost / jvp_cost:>7.2f}"
                f"{int(cycles):>8}{true_rel:>11.2e}{wall:>9.3f}"
                f"{'n/a' if temp is None else f'{temp / 2**20:.1f}':>9}",
                flush=True,
            )
        gap = float(
            jnp.linalg.norm(applied["jvp"] - applied["linearize"]) / jnp.linalg.norm(applied["jvp"])
        )
        print(f"  M(v) agreement between arms: rel {gap:.1e}", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    meshes = MESHES if not argv else ((int(argv[0]), int(argv[1])),)
    for nx, ny in meshes:
        main(
            nx,
            ny,
            float(argv[2]) if len(argv) > 2 else MU,
            float(argv[3]) if len(argv) > 3 else MARCH_RTOL,
        )
        print(flush=True)
