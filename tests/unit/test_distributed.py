"""Gate 1b-ii — the padded distributed residual under `shard_map` matches serial (value + gradient).

The culmination of Stage 1: partition the mesh, pad partitions to uniform shapes, and evaluate the
scalar-diffusion residual across devices with `shard_map` (an `all_gather` halo). The sharded
residual must equal the serial residual cell-for-cell, and a gradient through it must match the
serial gradient — proving the collective's automatic adjoint carries the real physics correctly.

Runs in a subprocess so multiple CPU devices can be simulated
(``--xla_force_host_platform_device_count``), which must be set before JAX initializes.
"""

from __future__ import annotations

import subprocess
import sys

_SUBPROCESS = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np
import jax, jax.numpy as jnp
import aquaflux  # x64
from aquaflux.mesh import structured_grid_3d
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.boundary import BoundaryConditions, Dirichlet
from aquaflux.parallel import BlockPartitioner, partition_mesh
from aquaflux.properties import Constant, PropertyModel
from aquaflux.parallel.distributed import build_padded_diffusion, distributed_diffusion_residual

SIDES = ("left", "right", "bottom", "top", "back", "front")
BC = {"left": 1.0, "right": 0.0, "bottom": 0.3, "top": -0.2, "back": 0.5, "front": 0.1}
GAMMA = 1.3

mesh = structured_grid_3d(6, 6, 6, named_boundaries=True)
geom = mesh.geometry()
gamma_global = GAMMA * jnp.ones(mesh.n_cells)
boundary = {name: Dirichlet(BC[name]) for name in SIDES}
serial = ResidualAssembler.build(mesh, geom, PropertyModel({"diffusivity": Constant(GAMMA)}), (DiffusionFlux(),), BoundaryConditions(boundary))

# Bake the Dirichlet face values into a per-global-face array (matches the serial closure exactly).
bval_global = np.zeros(mesh.n_faces)
for name in SIDES:
    bval_global[np.asarray(mesh.face_patches.indices(name))] = BC[name]
bval_global = jnp.asarray(bval_global)

labels = BlockPartitioner().partition(mesh, 4)  # real RCM-block min-cut-ish partition
pmesh = partition_mesh(mesh, labels)
padded = build_padded_diffusion(pmesh, geom, gamma_global, bval_global)

assert jax.device_count() == 4
phi = jnp.asarray(np.random.default_rng(0).standard_normal(mesh.n_cells))
weight = jnp.asarray(np.random.default_rng(1).standard_normal(mesh.n_cells))

r_serial = serial.residual(phi)
r_dist = distributed_diffusion_residual(padded, phi)
assert jnp.allclose(r_serial, r_dist, atol=1e-10), float(jnp.max(jnp.abs(r_serial - r_dist)))

g_serial = jax.grad(lambda p: jnp.sum(weight * serial.residual(p)))(phi)
g_dist = jax.grad(lambda p: jnp.sum(weight * distributed_diffusion_residual(padded, p)))(phi)
assert jnp.allclose(g_serial, g_dist, atol=1e-9), float(jnp.max(jnp.abs(g_serial - g_dist)))
print("ok")
"""


def test_distributed_diffusion_residual_matches_serial_value_and_gradient() -> None:
    """The `shard_map` diffusion residual + its gradient match serial on 4 simulated devices."""
    result = subprocess.run([sys.executable, "-c", _SUBPROCESS], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
