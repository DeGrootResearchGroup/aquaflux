"""A distributed **iterative** gradient scheme matches serial via a per-sweep ghost exchange.

``CompactGreenGauss`` reconstructs each owned cell's gradient in one pass, so exchanging ``phi`` once
makes it serial-exact (covered in ``test_distributed_gradient.py``). ``CorrectedGreenGauss`` instead
solves a partition-coupled linear system for the gradient: its operator reads ghost gradients on
*every* apply, so a single exchange is not enough — the ghost rows of the iterate must be refreshed
before each sweep. The assembler threads the same ``gradient_hook`` into the gradient scheme's own
solve to do exactly that, and the owned gradients then converge to the serial ones.

The solve must form no global inner product for a per-apply exchange to be exact, so this needs
``SweptGradientSolve`` (preconditioned-Richardson sweeps); the checks here therefore use it, on a
columnwise-skewed 3D grid, in two subprocesses:

- **forward:** the sharded ``CorrectedGreenGauss`` + ``SweptGradientSolve`` residual matches serial,
  and — with the *per-sweep* exchange dropped (but the final gradient exchange kept) — it diverges,
  proving the per-sweep exchange, not just the final one, is load-bearing for an iterative scheme;
- **adjoint:** the parameter gradient through the sharded residual matches serial, and the
  reduction-forming GMRES gradient solve and the nested-solve ``HessianCorrectedGradient`` raise when
  asked to run distributed rather than return a wrong owned gradient.

Each subprocess sets the device count before JAX starts (hence the subprocess) and pays a fixed
device-setup cost, so the checks are grouped into as **few** subprocesses as the memory budget allows:
the forward-only checks share one, and the single reverse-mode (gradient) check shares the other with
the trace-only rejection checks — no subprocess combines the reverse-mode compile with another full
compile. The swept/serial match is bit-for-bit for any size or sweep count (both run the same), so a
small mesh and few sweeps keep every compile and the grad tape small.
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
mesh = columnwise_perturbed_grid_3d(5, 5, 5, perturb=0.25, named_boundaries=True)
geom = mesh.geometry()
properties = PropertyModel({"diffusivity": Constant(1.3)})
boundary = BoundaryConditions(BOUNDARY)
# Small mesh, few sweeps: the distributed/serial comparison is exact for any size or count (both run
# the same), so this keeps each subprocess's sharded compile — and the grad's reverse-mode tape —
# small, which matters because these device subprocesses share the fast tier's wall-clock budget.
# warn_tol off so the serial reference stays silent.
SWEEPS = 4


def assemble(m, g):
    scheme = CorrectedGreenGauss(solver=SweptGradientSolve(sweeps=SWEEPS, warn_tol=None))
    return ResidualAssembler.build(
        m, g, properties, (DiffusionFlux(),), boundary, gradient_scheme=scheme
    )


serial = assemble(mesh, geom)
pmesh = partition_mesh(mesh, BlockPartitioner().partition(mesh, 4))
distributed = build_distributed_residual(pmesh, geom, assemble)
assert distributed.exchange_gradient, "a gradient scheme is injected, so the halo must exchange it"

phi = jnp.asarray(np.random.default_rng(0).standard_normal(mesh.n_cells))
weight = jnp.asarray(np.random.default_rng(1).standard_normal(mesh.n_cells))
"""

# Forward-only checks (no reverse-mode compile): the residual matches serial, and dropping just the
# per-sweep exchange breaks it.
_FORWARD = (
    _SETUP
    + r"""
# The distributed swept solve refreshes ghost rows each sweep, so at every sweep its owned rows equal
# the serial iterate's — bit-for-bit up to scatter/collective reordering, for the same sweep count.
r_s = serial.residual(phi)
r_d = distributed.residual(phi)
assert jnp.all(jnp.isfinite(r_d))
assert jnp.allclose(r_s, r_d, atol=1e-9), ("value", float(jnp.max(jnp.abs(r_s - r_d))))


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
r_ctrl = distributed_ctrl.residual(phi)
assert not jnp.allclose(r_s, r_ctrl, atol=1e-8), "per-sweep gradient exchange is not load-bearing"
print("ok")
"""
)

# The one reverse-mode check, grouped with the trace-only rejection checks (which raise before any
# compile) so no subprocess pairs the gradient compile with another full compile.
_ADJOINT = (
    _SETUP
    + r"""
from aquaflux.schemes import CorrectedGreenGauss as _CGG, HessianCorrectedGradient

g_s = jax.grad(lambda p: jnp.sum(weight * serial.residual(p)))(phi)
g_d = jax.grad(lambda p: jnp.sum(weight * distributed.residual(p)))(phi)
assert jnp.all(jnp.isfinite(g_d))
assert jnp.allclose(g_s, g_d, atol=1e-8), ("grad", float(jnp.max(jnp.abs(g_s - g_d))))


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


def test_distributed_swept_gradient_forward_matches_serial_via_per_sweep_exchange() -> None:
    """The sharded corrected-gradient residual is serial-exact, and the per-sweep exchange is why:
    dropping only it makes the skewed-mesh residual diverge from serial."""
    _run(_FORWARD)


def test_distributed_swept_gradient_adjoint_matches_and_unsupported_solves_refuse() -> None:
    """The adjoint through the per-sweep exchange matches the serial parameter gradient, and the
    GMRES gradient solve and Hessian scheme raise when asked to run distributed."""
    _run(_ADJOINT)
