"""The distributed mechanism works: `shard_map` + `all_gather` + `jax.grad`.

Before wiring the full padded distributed residual, this guards the environment capability the
whole distributed path depends on: that a `shard_map` program with an `all_gather` collective (the
halo-exchange primitive) runs on multiple devices *and* is reverse-mode differentiable, with the
collective's adjoint derived automatically. It mimics the halo pattern — each shard holds owned
values, all-gathers them, reads a 'ghost' value from another shard by (partition, index), and
computes a local result — and checks both the forward value and the gradient against a serial
reference.

Runs in a subprocess because simulating multiple CPU devices
(``--xla_force_host_platform_device_count``) must be set before JAX initializes, and the pytest
process has already initialized JAX — the same reason `test_x64_import` uses a subprocess.
"""

from __future__ import annotations

import subprocess
import sys

_SUBPROCESS = r"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
jax.config.update("jax_enable_x64", True)

n_part, n_owned = 4, 3
owned = jnp.asarray(np.random.default_rng(0).standard_normal((n_part, n_owned)))
src_part = jnp.asarray(np.array([(p + 1) % n_part for p in range(n_part)]))  # ghost from next shard
src_idx = jnp.zeros(n_part, dtype=jnp.int64)
mesh = Mesh(np.array(jax.devices()), axis_names=("p",))

def local(owned_shard, gsp, gsi):
    all_owned = jax.lax.all_gather(owned_shard, "p", axis=0, tiled=True)  # (n_part, n_owned)
    ghost = all_owned[gsp[0], gsi[0]]
    return (jnp.sum(owned_shard) + ghost).reshape(1)

dist = jax.shard_map(local, mesh=mesh, in_specs=(P("p"), P("p"), P("p")), out_specs=P("p"))

def dist_obj(o):
    return jnp.sum(dist(o, src_part, src_idx) ** 2)

def serial_obj(o):
    per = jnp.array([jnp.sum(o[p]) + o[(p + 1) % n_part, 0] for p in range(n_part)])
    return jnp.sum(per ** 2)

assert jax.device_count() == 4
assert np.isclose(float(dist_obj(owned)), float(serial_obj(owned)))
assert jnp.allclose(jax.grad(dist_obj)(owned), jax.grad(serial_obj)(owned), atol=1e-10)
print("ok")
"""


def test_shard_map_all_gather_is_differentiable_on_multiple_devices() -> None:
    """A `shard_map` + `all_gather` halo pattern matches serial in value and gradient on 4 devices."""
    result = subprocess.run([sys.executable, "-c", _SUBPROCESS], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
