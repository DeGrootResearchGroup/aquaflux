"""A distributed **iterative** gradient scheme matches serial via a per-sweep ghost exchange.

``CompactGreenGauss`` reconstructs each owned cell's gradient in one pass, so exchanging ``phi`` once
makes it serial-exact (covered in ``test_distributed_gradient.py``). ``CorrectedGreenGauss`` instead
solves a partition-coupled linear system for the gradient: its operator reads ghost gradients on
*every* apply, so a single exchange is not enough — the ghost rows of the iterate must be refreshed
before each sweep. The assembler threads the same ``gradient_hook`` into the gradient scheme's own
solve to do exactly that, and the owned gradients then converge to the serial ones.

The solve must form no global inner product for a per-apply exchange to be exact, so this needs
``SweptGradientSolve`` (preconditioned-Richardson sweeps); the checks here therefore use it. Three
subprocess checks on a columnwise-skewed 3D grid:

- **matches:** the sharded ``CorrectedGreenGauss`` + ``SweptGradientSolve`` residual and its gradient
  match serial;
- **control:** with the *per-sweep* exchange dropped (but the final gradient exchange kept), the
  residual diverges from serial — proving the per-sweep exchange, not just the final one, is
  load-bearing for an iterative scheme;
- **rejects:** the reduction-forming GMRES gradient solve and the nested-solve
  ``HessianCorrectedGradient`` raise when asked to run distributed, rather than return a wrong owned
  gradient.

Each check runs in its **own** subprocess: the device count must be set before JAX starts, and a
separate process keeps each check to a single sharded-program compilation, so peak memory stays low
when the unit tier runs many workers in parallel.
"""

from __future__ import annotations

import subprocess
import sys

# Shared setup: a skewed mesh, a diffusion assembler whose gradient is the iterative
# CorrectedGreenGauss with a fixed-sweep solve, and its distributed counterpart on four devices.
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
from aquaflux.schemes import CorrectedGreenGauss, SweptGradientSolve
from tests.support.meshes import columnwise_perturbed_grid_3d

BOUNDARY = {
    "left": Dirichlet(1.0), "right": Dirichlet(0.0), "bottom": Neumann(0.25),
    "top": Neumann(-0.4), "back": ZeroGradient(), "front": Dirichlet(0.1),
}
# A genuinely non-orthogonal mesh: the diffusion non-orthogonal correction reads the cell gradient,
# so ghost gradients matter on partition-boundary faces — and the corrected-gradient solve itself
# couples across those boundaries.
mesh = columnwise_perturbed_grid_3d(6, 6, 6, perturb=0.25, named_boundaries=True)
geom = mesh.geometry()
properties = PropertyModel({"diffusivity": Constant(1.3)})
boundary = BoundaryConditions(BOUNDARY)
SWEEPS = 20  # plenty at 0.25 skew; the swept/serial comparison is exact for any count anyway


def _swept_scheme():
    # warn_tol off so the serial reference emits no host-side under-resolution callback.
    return CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=SWEEPS, warn_tol=None))


def assemble(m, g):
    return ResidualAssembler.build(
        m, g, properties, (DiffusionFlux(),), boundary, gradient_scheme=_swept_scheme()
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
# The distributed swept solve refreshes ghost rows each sweep, so at every sweep its owned rows equal
# the serial iterate's — bit-for-bit up to scatter/collective reordering, for the same sweep count.
r_s = serial.residual(phi)
r_d = distributed.residual(phi)
assert jnp.all(jnp.isfinite(r_d))
assert jnp.allclose(r_s, r_d, atol=1e-9), ("value", float(jnp.max(jnp.abs(r_s - r_d))))

g_s = jax.grad(lambda p: jnp.sum(weight * serial.residual(p)))(phi)
g_d = jax.grad(lambda p: jnp.sum(weight * distributed.residual(p)))(phi)
assert jnp.all(jnp.isfinite(g_d))
assert jnp.allclose(g_s, g_d, atol=1e-8), ("grad", float(jnp.max(jnp.abs(g_s - g_d))))
print("ok")
"""
)

_CONTROL = (
    _SETUP
    + r"""
# Drop *only* the per-sweep exchange (ignore operator_hook), keeping the final gradient exchange the
# distributed residual applies. The corrected-gradient solve then iterates on stale ghost rows, so its
# owned gradients converge to the wrong value and the residual must diverge from serial — isolating
# the per-sweep exchange as load-bearing, distinct from the final one.
class _NoPerSweep(SweptGradientSolve):
    def solve(self, volume, operator, rhs, *, operator_hook=None):
        return super().solve(volume, operator, rhs)


def assemble_ctrl(m, g):
    scheme = CorrectedGreenGauss(solver=_NoPerSweep(sweeps=SWEEPS, warn_tol=None))
    return ResidualAssembler.build(
        m, g, properties, (DiffusionFlux(),), boundary, gradient_scheme=scheme
    )


distributed_ctrl = build_distributed_residual(pmesh, geom, assemble_ctrl)
assert distributed_ctrl.exchange_gradient  # the final gradient exchange is still on
r_s = serial.residual(phi)
r_ctrl = distributed_ctrl.residual(phi)
assert not jnp.allclose(r_s, r_ctrl, atol=1e-8), "per-sweep gradient exchange is not load-bearing"
print("ok")
"""
)

_REJECTS = (
    _SETUP
    + r"""
from aquaflux.schemes import CorrectedGreenGauss as _CGG, HessianCorrectedGradient


def _distributed_with(scheme):
    def _assemble(m, g):
        return ResidualAssembler.build(
            m, g, properties, (DiffusionFlux(),), boundary, gradient_scheme=scheme
        )
    return build_distributed_residual(pmesh, geom, _assemble)


# GMRES forms inner products over the whole local vector, so a per-apply ghost exchange cannot make
# it serial-exact; the distributed solve must refuse rather than silently return a wrong gradient.
gmres = _distributed_with(_CGG())  # default solver is GmresGradientSolve
try:
    gmres.residual(phi)
    raise AssertionError("expected the distributed GMRES gradient solve to be refused")
except NotImplementedError as e:
    assert "SweptGradientSolve" in str(e), str(e)

# The Hessian-corrected gradient couples through nested solves that a single per-apply exchange of the
# outer gradient does not refresh, so it too must refuse.
hessian = _distributed_with(HessianCorrectedGradient())
try:
    hessian.residual(phi)
    raise AssertionError("expected the distributed Hessian gradient solve to be refused")
except NotImplementedError as e:
    assert "SweptGradientSolve" in str(e), str(e)
print("ok")
"""
)


def _run(source: str) -> None:
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"


def test_distributed_swept_gradient_matches_serial_on_skewed_mesh() -> None:
    """The per-sweep ghost exchange makes the sharded corrected-gradient residual serial-exact."""
    _run(_MATCHES)


def test_per_sweep_gradient_exchange_is_load_bearing() -> None:
    """With the per-sweep exchange dropped, the skewed-mesh residual diverges from serial."""
    _run(_CONTROL)


def test_distributed_gradient_solve_refuses_reductions_and_nested_solves() -> None:
    """A distributed GMRES gradient solve and the Hessian scheme raise instead of misconverging."""
    _run(_REJECTS)
