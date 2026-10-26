"""Whether XLA holds a cell-level padded operator's decoded columns across a loop, one copy per operator.

A cell-level padded-row operator stores its columns per cell and decodes the dof-level column of every
slot at apply time (``padded_rows._KroneckerColumns``). Inside a loop -- a Krylov iteration, a V-cycle --
the decode depends only on the stored cells, so XLA may hoist it out of the loop and keep the decoded
``int32[n_rows, width]`` array live for the whole loop, for every operator the loop applies.

This compiles a stand-in for ``_padded_product`` (the same counted-field decode, ``x[columns]``, the
row sum) at the shape of bfs3d's leading fine operator (23 040 cells, reach width 63, four fields: an
``(92160, 252)`` operator) and reads ``memory_analysis().temp_size_in_bytes`` in four arms:

* ``plain``, one operator applied ten times in a ``fori_loop``;
* ``promise``, the same with ``mode="promise_in_bounds"`` on the gather (no index normalization);
* ``barrier``, the same with the stored cells passed through ``lax.optimization_barrier`` with the
  iterate in each iteration, so the decode cannot leave the loop body;
* and ``plain`` / ``barrier`` with four such operators applied in turn in each iteration.

Reading: if the four-operator ``plain`` arm holds about four decoded arrays and the ``barrier`` arm one,
the step's temporary holds every operator's decoded columns at once because they are hoisted, and keeping
the decode in the loop bounds it by the largest operator's. Compile only; seconds. Run directly::

    python validation/padded_decode_hoisting.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

N_CELLS, REACH_WIDTH, FIELDS = 23040, 63, 4
WIDTH = REACH_WIDTH * FIELDS
ITERATIONS = 10
MIB = 2**20


def _decode(cells):
    """The dof-level column of every slot, the field counted as ``_KroneckerColumns`` counts it."""
    slot = jnp.arange(WIDTH, dtype=jnp.int32)[None]
    field = sum((slot >= s * REACH_WIDTH).astype(jnp.int32) for s in range(1, FIELDS))
    place = jnp.broadcast_to(slot - field * REACH_WIDTH, (N_CELLS, WIDTH))
    column = field * N_CELLS + jnp.take_along_axis(cells, place, axis=1)
    return jnp.broadcast_to(column[None], (FIELDS, N_CELLS, WIDTH)).reshape(FIELDS * N_CELLS, WIDTH)


def _apply(values, cells, x, promise):
    columns = _decode(cells)
    gathered = x.at[columns].get(mode="promise_in_bounds") if promise else x[columns]
    return jnp.sum(values * gathered, axis=1)


def _loop(values, operators, x, arm):
    def body(_, x):
        for cells in operators:
            if arm == "barrier":
                cells, x = jax.lax.optimization_barrier((cells, x))
            x = x + 1e-3 * _apply(values, cells, x, arm == "promise")
        return x

    return jax.lax.fori_loop(0, ITERATIONS, body, x)


def main():
    rows = FIELDS * N_CELLS
    base = (jnp.arange(N_CELLS)[:, None] * 7 + jnp.arange(REACH_WIDTH)[None]) % N_CELLS
    operators = [((base * (k + 1)) % N_CELLS).astype(jnp.int32) for k in range(4)]
    values = jnp.ones((rows, WIDTH), jnp.float32)
    x = jnp.ones(rows, jnp.float32)
    print(
        f"jax {jax.__version__}; operator ({rows}, {WIDTH}): decoded int32 columns "
        f"{rows * WIDTH * 4 / MIB:.1f} MiB, gathered float32 values {rows * WIDTH * 4 / MIB:.1f} MiB",
        flush=True,
    )
    for count, arms in ((1, ("plain", "promise", "barrier")), (4, ("plain", "barrier"))):
        for arm in arms:
            compiled = (
                jax.jit(_loop, static_argnums=3).lower(values, operators[:count], x, arm).compile()
            )
            temp = compiled.memory_analysis().temp_size_in_bytes
            print(f"{count} operator(s), {arm}: temp {temp / MIB:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
