"""What walking a graded medium costs: the optical-depth scan, and the transfer assemble through it.

Two measurements, each read as XLA's compiled working memory (``memory_analysis()
.temp_size_in_bytes``, exact and unaffected by what else the machine is doing) and as the median
wall clock of warm calls, with a checksum so a before-and-after pair can be seen to compute the
same thing:

* **The walk itself** -- ``VoxelAbsorption.optical_depth`` over ``WALK_PAIRS`` random segments in
  a ``WALK_GRID`` grid of random coefficient, forward and as the gradient with respect to the
  coefficient. Reported per pair, because a gather chunk is a pair count and this is what one
  pair of it costs.
* **The facet-to-facet assemble** -- ``TransferMatrix.assemble`` under a graded
  ``VoxelAbsorption``, which walks every facet pair, in a closed 10 cm box of water triangulated
  at two facet counts -- closed, so every facet sees others and the checksum is not a sum of
  zeros. The frozen build uses no self-occlusion, since the mask is not what is being measured.
  Each size is assembled at the default pair limit and at ``SMALL_PAIR_LIMIT``: 2,028 facets is
  4.1M pairs, one pass at the default, so only the smaller limit shows the passes bounding it.

Run with ``validation/run_case.sh validation/radiation_voxel_walk.py``.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    Surfaces,
    VoxelAbsorption,
    build_transfer,
)
from tests.unit.radiation_references import inward_box  # noqa: E402

#: Cells per axis of the walk's grid, and the segments walked through it.
WALK_GRID = (32, 32, 32)
WALK_PAIRS = 65_536
#: Side of the box, metres, and its divisions per face at each facet count measured.
BOX_SIDE = 0.1
DIVISIONS = (9, 13)
#: The second pair limit the assemble is measured at, below the facet counts' own pair counts.
SMALL_PAIR_LIMIT = 1_000_000
#: Warm calls timed after the first.
CALLS = 5


def graded(shape, extent, rng):
    """A coefficient that varies by 20% about the Sozzi water's 35.67 per metre."""
    field = 35.67 * (1.0 + 0.2 * rng.uniform(-1.0, 1.0, shape))
    return VoxelAbsorption(field, origin=-extent / 2.0, spacing=extent / np.asarray(shape))


def timed(compiled, *arguments):
    """The first result, and the median seconds of ``CALLS`` warm calls."""
    result = jax.block_until_ready(compiled(*arguments))
    seconds = []
    for _ in range(CALLS):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*arguments))
        seconds.append(time.perf_counter() - start)
    return result, statistics.median(seconds)


def walk(rng):
    extent = np.array([0.1, 0.1, 0.1])
    medium = graded(WALK_GRID, extent, rng)
    origin = jnp.asarray(rng.uniform(-0.05, 0.05, (WALK_PAIRS, 3)))
    target = jnp.asarray(rng.uniform(-0.05, 0.05, (WALK_PAIRS, 3)))

    def depth(coefficient):
        field = VoxelAbsorption(coefficient, origin=medium.origin, spacing=medium.spacing)
        return field.optical_depth(origin, target)

    def total(coefficient):
        return jnp.sum(depth(coefficient))

    for name, function in (("forward", depth), ("gradient", jax.grad(total))):
        compiled = jax.jit(function).lower(medium.coefficient).compile()
        working = compiled.memory_analysis().temp_size_in_bytes
        result, seconds = timed(compiled, medium.coefficient)
        print(
            f"walk {name}: grid {WALK_GRID}, {WALK_PAIRS} pairs, "
            f"{working / WALK_PAIRS:.0f} B/pair working, {seconds:.3f} s, "
            f"checksum {float(jnp.sum(result)):.15e}",
            flush=True,
        )


def assemble(rng):
    medium = graded((12, 12, 12), np.full(3, BOX_SIDE), rng)
    for divisions in DIVISIONS:
        box = Surfaces.from_triangles(BOX_SIDE * (inward_box(divisions) - 0.5), emission=1.0)
        transfer = build_transfer(box, self_occlusion=NoOcclusion())
        pairs = box.n_facets**2
        for pair_limit in (None, SMALL_PAIR_LIMIT):
            limit = {} if pair_limit is None else {"pair_limit": pair_limit}

            def reflected(coefficient, box=box, transfer=transfer, limit=limit):
                field = VoxelAbsorption(coefficient, origin=medium.origin, spacing=medium.spacing)
                return transfer.assemble(box, field, **limit)[0]

            try:
                compiled = jax.jit(reflected).lower(medium.coefficient).compile()
            except TypeError:
                # The code before the change takes no pair limit: its assemble is one pass.
                print(f"assemble: {box.n_facets} facets, pair limit {pair_limit}: not accepted")
                continue
            working = compiled.memory_analysis().temp_size_in_bytes
            result, seconds = timed(compiled, medium.coefficient)
            label = "default" if pair_limit is None else pair_limit
            print(
                f"assemble: {box.n_facets} facets, pair limit {label}, "
                f"{working / 1e9:.3f} GB working ({working / pairs:.0f} B/pair), {seconds:.2f} s, "
                f"checksum {float(jnp.sum(result)):.15e}",
                flush=True,
            )


def main():
    print(f"jax {jax.__version__}, {jax.default_backend()}", flush=True)
    rng = np.random.default_rng(0)
    walk(rng)
    assemble(rng)


if __name__ == "__main__":
    main()
