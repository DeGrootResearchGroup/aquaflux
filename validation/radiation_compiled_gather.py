"""What compiling the gather buys: a gradient bounded by the pair limit, and a cheap streamed pass.

Two measurements on one analytic lamp (the Sozzi & Taghipour lamp's radius and length, 32 x 64 =
4,096 facets, exitance 696.42 W/m^2) with receivers drawn uniformly in the chamber annulus and the
water as ``UniformAbsorption(35.67)``:

* **Gradient working memory** -- the compiled working-memory figure of ``jax.grad`` of the summed
  field with respect to the emission, at two receiver counts sixteen times apart and the default
  pair limit. Read from ``memory_analysis().temp_size_in_bytes``, which is exact and unaffected by
  whatever else the machine is doing. With the scan body checkpointed it is flat in the receiver
  count; without, it grows in proportion.
* **The streamed field** -- a sleeve in the water as the only body, passes of 200 receivers, ten
  calls in a row with the wall clock, a checksum and (on macOS) the process's memory footprint
  after each. The first call includes compilation; the later ones are the per-call cost a sweep
  pays. The footprint is what shows whether each call leaves compiled programs behind: run
  eagerly, every pass of every call traced and compiled afresh, and the process grew call by call.

The "before" figures in the rules were taken with this script on the commit before the change.

Run with ``validation/run_case.sh validation/radiation_compiled_gather.py``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
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
    UniformAbsorption,
    direct_fluence_rate,
)
from aquaflux.solids import Cylinder  # noqa: E402
from tests.unit.radiation_references import cylinder_triangles  # noqa: E402

#: Lamp radius and half-length, metres; sectors and slices of its triangulation.
LAMP = (0.0115, 0.2, 32, 64)
#: Receiver counts the gradient's memory is read at.
GRADIENT_RECEIVERS = (2_000, 32_000)
#: Receivers per streamed pass.
PASS_RECEIVERS = 200
#: Streamed calls made in a row.
STREAMED_CALLS = 10


def footprint_gb():
    """This process's physical memory footprint, in GB, or ``None`` off macOS.

    The footprint rather than the resident set, because it counts the compressed pages the resident
    set leaves out. Read from ``proc_pid_rusage``'s ``ri_phys_footprint``.
    """
    if sys.platform != "darwin":
        return None

    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(40)]

    usage = Usage()
    library = ctypes.CDLL(ctypes.util.find_library("proc"))
    library.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage))
    return usage.f7 / 1e9


def receivers(rng, n):
    """Points uniform in angle and height, between the lamp and the chamber wall."""
    radius = rng.uniform(0.013, 0.044, n)
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    height = rng.uniform(-0.25, 0.25, n)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), height], axis=1)


def main():
    radius, half_length, sectors, slices = LAMP
    lamp = Surfaces.from_triangles(
        cylinder_triangles(radius, half_length, sectors, slices), emission=696.42
    )
    water = UniformAbsorption(35.67)
    rng = np.random.default_rng(0)
    print(f"lamp: {lamp.n_facets} facets; jax {jax.__version__}", flush=True)

    for n in GRADIENT_RECEIVERS:
        points = jnp.asarray(receivers(rng, n))

        def total(emission, points=points):
            lit = lamp.with_optics(emission=emission)
            return jnp.sum(direct_fluence_rate(lit, points, absorption=water))

        compiled = jax.jit(jax.grad(total)).lower(jnp.asarray(lamp.emission)).compile()
        working = compiled.memory_analysis().temp_size_in_bytes
        print(
            f"gradient: {n * lamp.n_facets:.3e} pairs, {working / 1e6:.1f} MB working", flush=True
        )

    sleeve = Cylinder(centre=[0.03, 0.0, 0.0], axis=[0, 0, 1], radius=0.004, half_length=0.3)
    points = receivers(rng, 4_000)
    points = points[np.linalg.norm(points[:, :2] - [0.03, 0.0], axis=1) > 0.005]
    for call in range(STREAMED_CALLS):
        start = time.perf_counter()
        field = direct_fluence_rate(
            lamp,
            points,
            occluders=[sleeve],
            self_occlusion=NoOcclusion(),
            absorption=water,
            pair_limit=PASS_RECEIVERS * lamp.n_facets,
        ).block_until_ready()
        elapsed = time.perf_counter() - start
        memory = footprint_gb()
        print(
            f"streamed call {call}: {len(points)} receivers, {elapsed:.2f} s, "
            f"checksum {float(jnp.sum(field)):.12e}"
            + ("" if memory is None else f", footprint {memory:.3f} GB"),
            flush=True,
        )


if __name__ == "__main__":
    main()
