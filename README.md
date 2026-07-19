# aquaflux

[![CI](https://github.com/DeGrootResearchGroup/aquaflux/actions/workflows/ci.yml/badge.svg)](https://github.com/DeGrootResearchGroup/aquaflux/actions/workflows/ci.yml)

**A differentiable, unstructured, cell-centred finite-volume (FVM) flow solver written in [JAX](https://docs.jax.dev/).**

Because the entire solver is written in JAX, the gradient of any output — a drag,
a mean concentration, an outlet flux — with respect to any input — boundary
values, material properties, or the mesh node coordinates themselves — is
available by automatic differentiation. That makes sensitivity analysis,
gradient-based optimization, inverse problems, and parameter calibration
first-class rather than bolt-on, and it lets the solver sit inside a larger
differentiable pipeline.

aquaflux grew out of work on water and environmental engineering reactors
(ozone/UV/chlorine contactors, clarifiers, digesters) and is designed to couple
with [aquakin](https://github.com/DeGrootResearchGroup/aquakin), a differentiable
reactive-transport package. Nothing about the solver is specific to that domain,
though — it is a general incompressible finite-volume code, and the same machinery
applies anywhere you want gradients through a flow solve.

> **Status: early development (pre-alpha).** The core solver works — steady and
> transient scalar transport, coupled pressure–velocity flow, and k–ω SST
> turbulence all run and are tested — but the public API is still settling and
> may change between versions. Not yet published to PyPI.

## What's here

- **Unstructured meshes** — a static cell/face/node `Mesh` with derived geometry
  (face areas and normals, cell volumes and centroids). Build structured grids
  in-process, or import an OpenFOAM `polyMesh`. The node coordinates are a
  differentiable leaf, so gradients flow through the derived geometry — the basis
  for mesh-sensitivity and shape optimization.
- **Scalar transport** — advection (upwind and slope-limited), diffusion (with
  non-orthogonal correction), transient, and volumetric-source operators,
  assembled into a residual `R(state, params)`.
- **Coupled pressure–velocity flow** — the `(u, v[, w], p)` block solved
  monolithically with Rhie–Chow continuity coupling, not segregated SIMPLE/PISO.
- **k–ω SST turbulence** — the RANS closure, coupled to the flow solve through a
  segregated outer loop.
- **Exact adjoints** — steady solves converge with a Newton driver whose gradient
  comes from implicit differentiation (the implicit-function theorem on the
  converged state plus an adjoint through each linear solve). No solver iteration
  is unrolled onto the autodiff tape, so gradient cost and accuracy are
  independent of the iteration count.
- **Solvers** — Krylov linear solves via [lineax](https://github.com/patrick-kidger/lineax),
  a block preconditioner for the flow saddle-point, algebraic multigrid, and
  pseudo-transient continuation for high-Reynolds cases.
- **Swappable numerics** — interpolation, gradient reconstruction, and slope
  limiters are first-class strategy objects with a known order of accuracy,
  injected into operators and tested in isolation, so the numerics can be changed
  without touching the physics.
- **Distributed memory** — domain decomposition (including a SCOTCH partitioner)
  and halo exchange for a sharded residual.

## Installation

aquaflux requires Python ≥ 3.10. Until it is published to PyPI, install from
source:

```bash
git clone https://github.com/DeGrootResearchGroup/aquaflux
cd aquaflux
pip install -e .
```

> **Note:** `import aquaflux` enables JAX 64-bit (x64) mode process-wide.
> Finite-volume transport and stiff coupling require double precision, so this is
> an intentional, documented side effect — other JAX code in the same process
> will use float64 afterward.

## Quick start

A lid-driven cavity: build a mesh, assemble the coupled flow problem, and solve
it to convergence.

```python
import aquaflux  # enables JAX float64
import jax.numpy as jnp
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import BlockPreconditioner, MomentumContinuity, MovingWall, NoSlipWall
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import DampedNewtonStep, ImplicitNewtonSolver


def cavity(viscosity):
    """A lid-driven cavity on a 32x32 unit square, parameterized by viscosity."""
    mesh = structured_grid_2d(32, 32, lx=1.0, ly=1.0, named_boundaries=True)
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(viscosity), "density": Constant(1.0)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),
                "bottom": NoSlipWall(),
                "left": NoSlipWall(),
                "right": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
        pressure_pin=0,
    )


residual = lambda state, problem: problem.residual(state)

# The block preconditioner accelerates the Krylov solve; build it once and reuse it.
# The forward step is a line-searched Newton step carrying that preconditioner.
precond = BlockPreconditioner.build(cavity(1e-2)).factory()
solver = ImplicitNewtonSolver(
    max_steps=30, forward_step=DampedNewtonStep(preconditioner=precond)
)

problem = cavity(1e-2)
state = solver.solve(residual, problem.initial_state(), problem)
velocity, pressure = problem.unpack(state)
```

### Differentiating through the solve

The converged solve is differentiable end to end. Here is the sensitivity of a
flow functional to the viscosity — one exact adjoint, with no unrolled iterations:

```python
import jax


def mean_speed(viscosity):
    problem = cavity(viscosity)
    state = solver.solve(residual, problem.initial_state(), problem)
    velocity, _ = problem.unpack(state)
    return jnp.mean(jnp.abs(velocity[:, 0]))


gradient = jax.grad(mean_speed)(1e-2)
```

The same works for gradients with respect to boundary values, source terms, or
the mesh node coordinates.

## Documentation

Full documentation — the mesh model, zones and patches, and the API reference —
is at [aquaflux.readthedocs.io](https://aquaflux.readthedocs.io).

## Development

```bash
pip install -e ".[lint,test]"

# Optional: enable the local ruff pre-push gate (opt-in, once per clone).
git config core.hooksPath .githooks

ruff check aquaflux tests
ruff format --check aquaflux tests
pytest -m "not validation and not slow"    # fast gate
```

The full test suite includes scientific validation cases (marked `validation`)
and multi-minute solves (marked `slow`); the command above runs the fast gate.
The same ruff checks run in CI on every pull request.

## License

MIT © Christopher T. DeGroot
