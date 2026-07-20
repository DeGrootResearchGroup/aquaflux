"""The neighbour-only ``all_to_all`` halo matches the all-gather reference, in value and gradient.

The all-gather exchange is the simple, obviously-correct reference: every ghost reads its exact
remote owner from a full stack of all owned values. The ``all_to_all`` exchange communicates only
the boundary cells each neighbour actually ghosts, so it scales — but it must fill ghosts to *bit
-for-bit the same values*. These tests pin that equivalence directly (fills agree), then confirm a
full distributed residual built on the ``all_to_all`` halo matches the serial residual in value and
gradient, exactly as the all-gather one does.

Runs in a subprocess so 4 CPU devices can be simulated, which must be configured before JAX starts.
"""

from __future__ import annotations

import subprocess
import sys

_FILLS_AGREE = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh as DeviceMesh, PartitionSpec as Pspec
import aquaflux  # x64
from aquaflux.mesh import structured_grid_3d
from aquaflux.parallel import (
    AllGatherHaloExchange, AllToAllHaloExchange, BlockPartitioner, PaddedLayout, partition_mesh,
)

mesh = structured_grid_3d(6, 6, 6, named_boundaries=True)
pmesh = partition_mesh(mesh, BlockPartitioner().partition(mesh, 4))
layout = PaddedLayout.from_partitioned(pmesh)
device_mesh = DeviceMesh(np.array(jax.devices()[:4]), axis_names=("p",))


def run(exchange, field):
    plan = exchange.plan(layout)
    owned = layout.owned_states_from_global(field)  # (n_part, n_owned_max, ...)

    def body(owned_shard, plan_shard):
        o = owned_shard[0]
        pl = jax.tree.map(lambda a: a[0], plan_shard)
        filled = exchange.fill(o, pl, "p")
        return filled.reshape(1, *filled.shape)

    return jax.shard_map(
        body, mesh=device_mesh, in_specs=(Pspec("p"), Pspec("p")), out_specs=Pspec("p"),
    )(owned, plan)


# Scalar and vector fields: the vector case exercises the trailing-axis genericity.
for shape in [(mesh.n_cells,), (mesh.n_cells, 3)]:
    field = jnp.asarray(np.random.default_rng(0).standard_normal(shape))
    ag = run(AllGatherHaloExchange(), field)
    a2a = run(AllToAllHaloExchange(), field)
    # Compare only the *real* owned+ghost rows of each partition (padding rows are arbitrary).
    for p, part in enumerate(pmesh.partitions):
        no, ng, nom = part.n_owned, part.n_ghost, layout.n_owned_max
        assert jnp.allclose(ag[p, :no], a2a[p, :no], atol=1e-14)
        assert jnp.allclose(ag[p, nom:nom + ng], a2a[p, nom:nom + ng], atol=1e-14), shape
print("ok")
"""

_RESIDUAL_MATCHES = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np
import jax, jax.numpy as jnp
import aquaflux  # x64
from aquaflux.boundary import BoundaryConditions, Dirichlet, Neumann, ZeroGradient
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.mesh import structured_grid_3d
from aquaflux.parallel import (
    AllToAllHaloExchange, BlockPartitioner, build_distributed_residual, partition_mesh,
)
from aquaflux.properties import Constant, PropertyModel

BOUNDARY = {
    "left": Dirichlet(1.0), "right": Dirichlet(0.0), "bottom": Neumann(0.25),
    "top": Neumann(-0.4), "back": ZeroGradient(), "front": Dirichlet(0.1),
}
mesh = structured_grid_3d(6, 6, 6, named_boundaries=True)
geom = mesh.geometry()
properties = PropertyModel({"diffusivity": Constant(1.3)})
boundary = BoundaryConditions(BOUNDARY)


def assemble(m, g):
    return ResidualAssembler.build(m, g, properties, (DiffusionFlux(),), boundary)


serial = assemble(mesh, geom)
pmesh = partition_mesh(mesh, BlockPartitioner().partition(mesh, 4))
distributed = build_distributed_residual(pmesh, geom, assemble, halo=AllToAllHaloExchange())

phi = jnp.asarray(np.random.default_rng(0).standard_normal(mesh.n_cells))
weight = jnp.asarray(np.random.default_rng(1).standard_normal(mesh.n_cells))

r_s, r_d = serial.residual(phi), distributed.residual(phi)
assert jnp.all(jnp.isfinite(r_d))
assert jnp.allclose(r_s, r_d, atol=1e-10), float(jnp.max(jnp.abs(r_s - r_d)))

g_s = jax.grad(lambda p: jnp.sum(weight * serial.residual(p)))(phi)
g_d = jax.grad(lambda p: jnp.sum(weight * distributed.residual(p)))(phi)
assert jnp.all(jnp.isfinite(g_d))
assert jnp.allclose(g_s, g_d, atol=1e-9), float(jnp.max(jnp.abs(g_s - g_d)))
print("ok")
"""


def _run(source: str) -> None:
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_all_to_all_fills_match_all_gather() -> None:
    """The `all_to_all` halo fills every real owned+ghost row to the all-gather values (scalar+vector)."""
    _run(_FILLS_AGREE)


def test_distributed_residual_with_all_to_all_matches_serial() -> None:
    """A distributed residual on the `all_to_all` halo matches serial in value and gradient."""
    _run(_RESIDUAL_MATCHES)
