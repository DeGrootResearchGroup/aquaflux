"""The distributed residual matches serial on a **non-orthogonal** mesh (derived-field halo).

The orthogonal gate leaves the reconstructed gradient identically zero, so it never exercises the
one thing a ghost cell cannot compute locally: its gradient (a ghost's stencil reaches past the
one-cell halo). On a skewed mesh the diffusion flux reads that gradient through the non-orthogonal
correction, so the distributed residual is only correct if each ghost's gradient is *exchanged*
from its owning partition — the assembler's ``gradient_hook`` seam.

Two checks, on a columnwise-skewed 3D grid with a ``CompactGreenGauss`` gradient (one-pass, so it is
correct on owned cells after the ``phi`` exchange):

- with the derived-field halo (the default), the sharded residual and its gradient match serial;
- with the gradient exchange forced off, they do **not** — proving the exchange is load-bearing and
  the first check is not passing for some unrelated reason.

Each check runs in its **own** subprocess: the device count must be set before JAX starts, and a
separate process keeps each check to a single sharded-program compilation, so peak memory stays low
when the unit tier runs many workers in parallel.
"""

from __future__ import annotations

import subprocess
import sys

# Shared setup: a skewed mesh, a diffusion assembler with a one-pass gradient scheme, and its
# distributed counterpart on four simulated devices.
_SETUP = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np
import jax, jax.numpy as jnp
import aquaflux  # x64
from aquaflux.boundary import BoundaryConditions, Dirichlet, Neumann, ZeroGradient
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.parallel import BlockPartitioner, build_distributed_residual, partition_mesh
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from tests.support.meshes import columnwise_perturbed_grid_3d

BOUNDARY = {
    "left": Dirichlet(1.0), "right": Dirichlet(0.0), "bottom": Neumann(0.25),
    "top": Neumann(-0.4), "back": ZeroGradient(), "front": Dirichlet(0.1),
}
# A genuinely non-orthogonal mesh: the diffusion non-orthogonal correction reads the cell gradient,
# so ghost gradients matter on partition-boundary faces.
mesh = columnwise_perturbed_grid_3d(6, 6, 6, perturb=0.25, named_boundaries=True)
geom = mesh.geometry()
properties = PropertyModel({"diffusivity": Constant(1.3)})
boundary = BoundaryConditions(BOUNDARY)


def assemble(m, g):
    return ResidualAssembler.build(
        m, g, properties, (DiffusionFlux(),), boundary, gradient_scheme=CompactGreenGauss()
    )


serial = assemble(mesh, geom)
pmesh = partition_mesh(mesh, BlockPartitioner().partition(mesh, 4))
distributed = build_distributed_residual(pmesh, geom, assemble)
assert distributed.exchange_gradient, "a gradient scheme is injected, so the halo must exchange it"

phi = jnp.asarray(np.random.default_rng(0).standard_normal(mesh.n_cells))
weight = jnp.asarray(np.random.default_rng(1).standard_normal(mesh.n_cells))
"""

_MATCHES = (
    _SETUP
    + r"""
r_s = serial.residual(phi)
r_d = distributed.residual(phi)
assert jnp.all(jnp.isfinite(r_d))
assert jnp.allclose(r_s, r_d, atol=1e-10), ("value", float(jnp.max(jnp.abs(r_s - r_d))))

g_s = jax.grad(lambda p: jnp.sum(weight * serial.residual(p)))(phi)
g_d = jax.grad(lambda p: jnp.sum(weight * distributed.residual(p)))(phi)
assert jnp.all(jnp.isfinite(g_d))
assert jnp.allclose(g_s, g_d, atol=1e-9), ("grad", float(jnp.max(jnp.abs(g_s - g_d))))
print("ok")
"""
)

_CONTROL = (
    _SETUP
    + r"""
import dataclasses
# Without the ghost-gradient exchange, ghosts keep their locally-miscomputed gradient, so the
# partition-boundary non-orthogonal correction is wrong and the residual must diverge from serial.
r_s = serial.residual(phi)
no_exchange = dataclasses.replace(distributed, exchange_gradient=False)
r_wrong = no_exchange.residual(phi)
assert not jnp.allclose(r_s, r_wrong, atol=1e-8), "ghost-gradient exchange is not load-bearing here"
print("ok")
"""
)


def _run(source: str) -> None:
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_distributed_residual_matches_serial_on_skewed_mesh() -> None:
    """The derived-field halo makes the sharded residual serial-exact on a non-orthogonal mesh."""
    _run(_MATCHES)


def test_ghost_gradient_exchange_is_load_bearing() -> None:
    """With the ghost-gradient exchange off, the skewed-mesh residual diverges from serial."""
    _run(_CONTROL)
