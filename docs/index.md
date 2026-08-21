# aquaflux

`aquaflux` is a differentiable, unstructured, cell-centred finite-volume (FVM)
flow solver written in [JAX](https://docs.jax.dev/). It is purpose-built for
water and environmental engineering — a bespoke tool for the intersection of
computational fluid dynamics and reactive transport, designed to couple with
[aquakin](https://aquakin.readthedocs.io) rather than to be a general-purpose
CFD code.

Because the whole solver is written in JAX, gradients of any output with respect
to any input — boundary values, material properties, or the mesh node positions
themselves — are available by automatic differentiation, which is what makes
sensitivity analysis, optimization, and parameter calibration first-class.

```{toctree}
:maxdepth: 2
:caption: Guide

mesh
mesh_zones_and_patches
gradient_reconstruction
steady_state_solving
preconditioning
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Installation

```bash
pip install aquaflux
```

```{note}
`import aquaflux` enables JAX 64-bit (x64) mode process-wide — finite-volume
transport and stiff coupling require double precision. This is global JAX state:
other JAX code in the same process will use float64 afterward.
```

## The mesh

Everything the solver does is assembled over a {class}`~aquaflux.mesh.Mesh`: a
static, unstructured collection of cells, the faces between them, and the derived
geometry (face areas and normals, cell volumes and centroids). A mesh is built
once and then reused across every solve.

```python
import aquaflux
from aquaflux.mesh import structured_grid_2d

mesh = structured_grid_2d(32, 32, lx=1.0, ly=1.0)   # 32 x 32 unit square
geometry = mesh.geometry()                            # areas, normals, volumes, centroids
print(mesh.n_cells, mesh.n_faces)
```

The mesh's node coordinates are a differentiable leaf, so gradients with respect
to node positions flow through the derived geometry — the basis for mesh-sensitivity
diagnostics and shape optimization.

See [The mesh](mesh.md) for a full walkthrough — building meshes, the geometry it
derives, cell renumbering, and quality diagnostics — and
[Cell zones and face patches](mesh_zones_and_patches.md) for naming regions of
cells and faces (the mechanism boundary conditions and multi-region models build on).
