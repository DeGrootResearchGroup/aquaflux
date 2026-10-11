"""What does the boundary fold cost a residual's trace and compile, and how does it grow with patches?

``BoundaryConditions.apply`` evaluates each patch's closure and writes the values into one per-face
array. Written as one ``.at[faces].set`` per patch, P patches give P scatters, each depending on the
previous one's result, so they cannot overlap. This harness counts the scatters a residual lowers to and
times its trace (``lower``) and its compile, for:

* **pitzDaily** -- the coupled RANS residual built from ``pitzdaily_openfoam/case.yaml`` (four
  boundary patches), and its Jacobian-vector product, at the hybrid initial condition;
* **a scalar diffusion residual on a 64 x 64 grid** whose boundary faces are split into 4, 16 and 64
  patches (Dirichlet and zero-gradient alternating), to show how the cost grows with the patch count.

Each timing is the minimum of ``REPEATS`` runs, each after ``jax.clear_caches()``, with the persistent
compilation cache switched off, so every run traces and compiles from scratch. The scatter counts are
read from the lowered StableHLO (before XLA's optimizations) and the compiled HLO (after them); the
compiled count also includes the scatters the rest of the residual makes, so compare it across versions,
not against the patch count.

Run from the repository root::

    validation/run_case.sh validation/boundary_scatter_cost.py

It prints one line per measurement.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["AQUAFLUX_DISABLE_COMPILATION_CACHE"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient  # noqa: E402
from aquaflux.case import read_case  # noqa: E402
from aquaflux.discretization import DiffusionFlux, ResidualAssembler  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.mesh import FacePatches, structured_grid_2d  # noqa: E402
from aquaflux.properties import Constant, PropertyModel  # noqa: E402

REPEATS = 3
GRID = 64
PATCH_COUNTS = (4, 16, 64)


def measure(fn, arrays, state) -> dict[str, float]:
    """Scatter counts and best-of-``REPEATS`` trace and compile seconds of ``jax.jit(fn)``.

    ``fn(arrays, state)`` takes the problem's array leaves as an argument rather than closing over
    them, so the mesh enters the program as an input, not as embedded constants.
    """
    trace, compile_ = [], []
    for _ in range(REPEATS):
        jax.clear_caches()
        start = time.perf_counter()
        lowered = jax.jit(fn).lower(arrays, state)
        middle = time.perf_counter()
        compiled = lowered.compile()
        trace.append(middle - start)
        compile_.append(time.perf_counter() - middle)
    return {
        "lowered scatters": lowered.as_text().count("stablehlo.scatter"),
        "compiled scatters": compiled.as_text().count(" scatter("),
        "trace s": min(trace),
        "compile s": min(compile_),
    }


def report(label: str, problem, state) -> None:
    """Print ``problem.residual``'s and its Jacobian-vector product's measurements, one line each."""
    arrays, static = eqx.partition(problem, eqx.is_array)

    def residual(arrays, s):
        return eqx.combine(arrays, static).residual(s)

    def jvp(arrays, s):
        return jax.jvp(lambda x: residual(arrays, x), (s,), (jnp.ones_like(s),))[1]

    for kind, fn in (("residual", residual), ("jvp", jvp)):
        m = measure(fn, arrays, state)
        print(
            f"{label:24s} {kind:8s} lowered scatters {m['lowered scatters']:4d}  "
            f"compiled scatters {m['compiled scatters']:4d}  trace {m['trace s']:7.3f} s  "
            f"compile {m['compile s']:7.3f} s",
            flush=True,
        )


def pitzdaily() -> None:
    """The coupled RANS residual of the pitzDaily case file, at its hybrid initial condition."""
    problem = read_case(HERE / "pitzdaily_openfoam" / "case.yaml").check().build()
    state = problem.state_from_physical(*hybrid_initialize(problem))
    if not bool(jnp.all(jnp.isfinite(problem.residual(state)))):
        raise SystemExit("pitzDaily residual is not finite at the initial condition")
    report("pitzDaily (4 patches)", problem, state)


def split_boundary(n_patches: int) -> tuple:
    """The grid with its boundary faces dealt into ``n_patches`` contiguous patches, and their closures."""
    mesh = structured_grid_2d(GRID, GRID)
    boundary_faces = np.flatnonzero(np.asarray(mesh.face_cells.neighbour) < 0)
    groups = {f"p{i}": chunk for i, chunk in enumerate(np.array_split(boundary_faces, n_patches))}
    patches = FacePatches.from_dict(mesh.face_cells.neighbour, groups)
    mesh = eqx.tree_at(lambda m: m.face_patches, mesh, patches)
    closures = {
        name: Dirichlet(float(i)) if i % 2 == 0 else ZeroGradient() for i, name in enumerate(groups)
    }
    return mesh, BoundaryConditions(closures)


def scalar(n_patches: int) -> None:
    """Steady diffusion on the grid with ``n_patches`` boundary patches."""
    mesh, boundary = split_boundary(n_patches)
    assembler = ResidualAssembler.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        boundary,
    )
    state = jnp.linspace(0.0, 1.0, mesh.n_cells)
    report(f"scalar {GRID}x{GRID} ({n_patches} patches)", assembler, state)


if __name__ == "__main__":
    print(f"jax {jax.__version__}, {jax.default_backend()}, best of {REPEATS}", flush=True)
    for n in PATCH_COUNTS:
        scalar(n)
    pitzdaily()
