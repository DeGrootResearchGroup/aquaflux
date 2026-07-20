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


_HALO_VECTOR_SUBPROCESS = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh as DeviceMesh, PartitionSpec as Pspec
import aquaflux  # x64
from aquaflux.mesh import structured_grid_3d
from aquaflux.parallel import (
    AllGatherHaloExchange, BlockPartitioner, PaddedLayout, partition_mesh,
)

DIM = 3
mesh = structured_grid_3d(5, 5, 5, named_boundaries=True)
pmesh = partition_mesh(mesh, BlockPartitioner().partition(mesh, 4))
layout = PaddedLayout.from_partitioned(pmesh)
halo = AllGatherHaloExchange()
plan = halo.plan(layout)

# A per-cell *vector* field -- the shape a reconstructed gradient has.
field = jnp.asarray(np.random.default_rng(0).standard_normal((mesh.n_cells, DIM)))
owned_states = layout.owned_states_from_global(field)  # (n_part, n_owned_max, DIM)

def per_device(owned_shard, plan_shard):
    owned = owned_shard[0]
    shard_plan = jax.tree.map(lambda a: a[0], plan_shard)
    filled = halo.fill(owned, shard_plan, "p")
    return filled.reshape(1, *filled.shape)

device_mesh = DeviceMesh(np.array(jax.devices()[:4]), axis_names=("p",))
filled = jax.shard_map(
    per_device, mesh=device_mesh, in_specs=(Pspec("p"), Pspec("p")), out_specs=Pspec("p"),
)(owned_states, plan)

assert filled.shape == (4, layout.n_owned_max + layout.n_ghost_max, DIM), filled.shape
# Every real owned and ghost row must equal what a direct global gather gives.
for p, part in enumerate(pmesh.partitions):
    expected = part.gather_cells(field)
    got_owned = filled[p, : part.n_owned]
    got_ghost = filled[p, layout.n_owned_max : layout.n_owned_max + part.n_ghost]
    assert jnp.allclose(got_owned, expected[: part.n_owned], atol=1e-14)
    assert jnp.allclose(got_ghost, expected[part.n_owned :], atol=1e-14)
print("ok")
"""


def _run(source: str) -> None:
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_distributed_residual_matches_serial_value_and_gradient() -> None:
    """The `shard_map` residual and its gradient match serial on 4 simulated devices."""
    _run(_SUBPROCESS)


def test_halo_exchange_carries_a_vector_field() -> None:
    """The exchange is generic in the trailing axes, so it carries a per-cell vector field.

    Pins the property the derived-field halo depends on: exchanging a reconstructed gradient
    (shape ``(n_cells, dim)``) rather than only a scalar needs no new code path.
    """
    _run(_HALO_VECTOR_SUBPROCESS)
