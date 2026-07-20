"""The sharded residual matches the serial residual in value and gradient.

Partition the mesh, pad the partitions to uniform shapes, build a real residual assembler per
partition, and evaluate it across devices with ``shard_map`` (an ``all_gather`` halo). The sharded
residual must equal the serial residual cell-for-cell, and a gradient through it must match the
serial gradient — proving the collective's automatically derived adjoint carries the real physics.

Because the per-device body runs the *same* ``ResidualAssembler`` the serial path runs, this also
covers the operators, properties and boundary closures injected into it: the non-constant boundary
conditions below are evaluated by their real closures on each device, not baked into a constant
per-face array.

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
from aquaflux.boundary import BoundaryConditions, Dirichlet, Neumann, ZeroGradient
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.mesh import structured_grid_3d
from aquaflux.parallel import BlockPartitioner, build_distributed_residual, partition_mesh
from aquaflux.properties import Constant, PropertyModel

GAMMA = 1.3
# A mix of closure kinds: the flux-type ones depend on the solution, so they cannot be pre-baked
# into a constant per-face array — they exercise the real closures running on each device.
BOUNDARY = {
    "left": Dirichlet(1.0),
    "right": Dirichlet(0.0),
    "bottom": Neumann(0.25),
    "top": Neumann(-0.4),
    "back": ZeroGradient(),
    "front": Dirichlet(0.1),
}

mesh = structured_grid_3d(6, 6, 6, named_boundaries=True)
geom = mesh.geometry()
properties = PropertyModel({"diffusivity": Constant(GAMMA)})
boundary = BoundaryConditions(BOUNDARY)
operators = (DiffusionFlux(),)


def assemble(local_mesh, local_geometry):
    return ResidualAssembler.build(local_mesh, local_geometry, properties, operators, boundary)


serial = assemble(mesh, geom)
labels = BlockPartitioner().partition(mesh, 4)
pmesh = partition_mesh(mesh, labels)
distributed = build_distributed_residual(pmesh, geom, assemble)

assert jax.device_count() == 4
phi = jnp.asarray(np.random.default_rng(0).standard_normal(mesh.n_cells))
weight = jnp.asarray(np.random.default_rng(1).standard_normal(mesh.n_cells))

r_serial = serial.residual(phi)
r_dist = distributed.residual(phi)
assert jnp.all(jnp.isfinite(r_dist)), "sharded residual is not finite"
assert jnp.allclose(r_serial, r_dist, atol=1e-10), float(jnp.max(jnp.abs(r_serial - r_dist)))

g_serial = jax.grad(lambda p: jnp.sum(weight * serial.residual(p)))(phi)
g_dist = jax.grad(lambda p: jnp.sum(weight * distributed.residual(p)))(phi)
assert jnp.all(jnp.isfinite(g_dist)), "sharded gradient is not finite"
assert jnp.allclose(g_serial, g_dist, atol=1e-9), float(jnp.max(jnp.abs(g_serial - g_dist)))
print("ok")
"""


def test_distributed_residual_matches_serial_value_and_gradient() -> None:
    """The `shard_map` residual and its gradient match serial on 4 simulated devices."""
    result = subprocess.run([sys.executable, "-c", _SUBPROCESS], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
