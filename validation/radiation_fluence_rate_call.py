"""What one call of the model's fluence rate costs, held mask and streamed.

The per-call price a design study pays once the model is built: the interreflection solve, the
point-source arrivals, and the volume gather of the emitted and reflected fields. Measured on a
lamp alone -- the Sozzi & Taghipour lamp's radius and length, 32 x 64 = 4,096 facets, exitance
696.42 W/m^2, reflectance 0.3 so the reflected field is not zero -- with 4,000 receivers drawn
uniformly in the chamber annulus, a sleeve beside the lamp as the only body, and the water as
``UniformAbsorption(35.67)`` and then as a graded ``VoxelAbsorption`` of the same mean, where each
path is walked through the grid.

For each of the two receiver-shadow strategies and each medium, three calls in a row are timed
and the field's checksum printed; the first includes compilation. Run with
``validation/run_case.sh validation/radiation_fluence_rate_call.py``.
"""

from __future__ import annotations

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
    RadiationSettings,
    Surfaces,
    UniformAbsorption,
    VoxelAbsorption,
    build_radiation_model,
    fluence_rate,
)
from aquaflux.solids import Cylinder  # noqa: E402
from tests.unit.radiation_references import cylinder_triangles  # noqa: E402

#: Lamp radius and half-length, metres; sectors and slices of its triangulation.
LAMP = (0.0115, 0.2, 32, 64)
N_RECEIVERS = 4_000
CALLS = 3


def receivers(rng, n):
    """Points uniform in angle and height, between the lamp and the chamber wall."""
    radius = rng.uniform(0.013, 0.044, n)
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    height = rng.uniform(-0.25, 0.25, n)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), height], axis=1)


def media():
    """The uniform medium, and a graded one of the same mean over the chamber."""
    shape = (12, 12, 16)
    rng = np.random.default_rng(1)
    field = 35.67 * (1.0 + 0.2 * rng.uniform(-1.0, 1.0, shape))
    extent = np.array([0.1, 0.1, 0.52])
    graded = VoxelAbsorption(field, origin=-extent / 2.0, spacing=extent / np.array(shape))
    return {"uniform": UniformAbsorption(35.67), "graded": graded}


def main():
    radius, half_length, sectors, slices = LAMP
    lamp = Surfaces.from_triangles(
        cylinder_triangles(radius, half_length, sectors, slices), emission=696.42, reflectance=0.3
    )
    rng = np.random.default_rng(0)
    points = receivers(rng, N_RECEIVERS)
    sleeve = Cylinder(centre=[0.03, 0.0, 0.0], axis=[0, 0, 1], radius=0.004, half_length=0.3)
    points = points[np.linalg.norm(points[:, :2] - [0.03, 0.0], axis=1) > 0.005]
    print(
        f"lamp: {lamp.n_facets} facets, {len(points)} receivers; jax {jax.__version__}", flush=True
    )
    for streamed in (False, True):
        settings = RadiationSettings(self_occlusion=NoOcclusion(), stream_receiver_mask=streamed)
        model = build_radiation_model(points, lamp, occluders=[sleeve], settings=settings)
        for name, medium in media().items():
            for call in range(CALLS):
                start = time.perf_counter()
                field, cycles = fluence_rate(model, lamp, absorption=medium)
                field = field.block_until_ready()
                elapsed = time.perf_counter() - start
                print(
                    f"{'streamed' if streamed else 'held'} mask, {name} medium, call {call}: "
                    f"{elapsed:.2f} s, {int(cycles)} cycles, checksum {float(jnp.sum(field)):.12e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
