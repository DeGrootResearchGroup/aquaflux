# The mesh

Every quantity `aquaflux` computes is assembled over a mesh: a static, unstructured
collection of **cells** (the finite volumes), the **faces** between them, and the
**nodes** (vertices) that define their shape. The {class}`~aquaflux.mesh.Mesh` object
holds this topology together with the node coordinates; the derived geometry — face
areas and normals, cell volumes and centroids — is computed from it on demand.

A mesh is built once and reused across every solve. This page covers building a mesh,
the geometry it derives, renumbering cells, and checking mesh quality.

## Cells, faces, and the owner/neighbour convention

`aquaflux` stores an unstructured mesh as flat arrays rather than a graph of objects,
which is what lets JAX vectorize over it. The topology is expressed through two
relations, each reachable from the mesh:

- **Face → cell** ({attr}`mesh.face_cells <aquaflux.mesh.Mesh.face_cells>`, a
  {class}`~aquaflux.mesh.FaceCellConnectivity`). Every face has an **owner** cell and,
  if it is interior, a **neighbour** cell. A face on the domain boundary has no
  neighbour; by convention its neighbour index is `-1`.
- **Face → node** ({attr}`mesh.face_nodes <aquaflux.mesh.Mesh.face_nodes>`, a
  {class}`~aquaflux.mesh.FaceNodeConnectivity`). Each face is an ordered ring of nodes,
  stored ragged (a two-node edge in 2D, an arbitrary polygon in 3D) so no memory is
  wasted on padding.

Each face stores a single **outward normal that points out of its owner cell**; the
neighbour cell sees the same face with the opposite normal. This orientation is fixed
once when the mesh is built, so it is robust to how the input node ordering happened to
wind each face.

Size queries are available directly on the mesh:

```python
mesh.dim        # spatial dimension (2 or 3)
mesh.n_cells    # number of cells
mesh.n_faces    # number of faces
mesh.n_nodes    # number of nodes
```

## Building a mesh

### Structured grids

The quickest way to get a mesh for a rectangular domain — a tank or a channel — is a
structured-grid generator. {func}`~aquaflux.mesh.structured_grid_2d` builds a grid of
quadrilaterals; {func}`~aquaflux.mesh.structured_grid_3d` builds hexahedra:

```python
from aquaflux.mesh import structured_grid_2d, structured_grid_3d

mesh2d = structured_grid_2d(32, 32, lx=1.0, ly=1.0)          # 32 x 32 on [0,1] x [0,1]
mesh3d = structured_grid_3d(16, 16, 8, lx=2.0, ly=2.0, lz=1.0)
```

Pass `named_boundaries=True` to tag the domain sides as named face patches
(`"left"`, `"right"`, `"bottom"`, `"top"`, and `"back"`/`"front"` in 3D), so a
different boundary condition can be attached to each side:

```python
mesh = structured_grid_2d(32, 32, named_boundaries=True)
```

See [Cell zones and face patches](mesh_zones_and_patches.md) for what patches are and
how boundary conditions use them.

### From explicit connectivity

For an unstructured mesh — typically read from a mesh file — build the mesh from its
faces with {meth}`Mesh.from_faces <aquaflux.mesh.Mesh.from_faces>`. You supply the node
coordinates, each face's node list, and the owner and neighbour cell of each face
(`-1` for a boundary face):

```python
import numpy as np
from aquaflux.mesh import Mesh

# two adjacent unit squares sharing one interior face
coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                   [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
face_nodes = [[0, 1], [1, 2], [1, 4],          # bottom edges, shared edge
              [0, 3], [3, 4], [4, 5],          # more edges ...
              [2, 5], [3, 4], [4, 5]]
owner     = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1])
neighbour = np.array([-1, -1, 1, -1, -1, -1, -1, -1, -1])
mesh = Mesh.from_faces(coords, face_nodes, owner, neighbour, n_cells=2)
```

If you already hold the connectivity as flat compressed-sparse-row (CSR) arrays — a
row-pointer array plus a flat index array — {meth}`Mesh.from_csr
<aquaflux.mesh.Mesh.from_csr>` takes them directly and avoids the per-face Python loop,
which keeps construction cheap on large meshes.

### Validation

Meshes are validated at construction. The checks are topological and inexpensive:
finite coordinates, index dtypes and ranges, the right number of nodes per face
(exactly two in 2D, at least three in 3D), no cell left unreferenced, and the boundary
sentinel used correctly. An invalid mesh raises `ValueError` with a clear message rather
than silently producing wrong volumes or `NaN`s downstream.

## Geometry

The metrics the solver needs — face areas, face centroids, outward normals, cell
volumes, and cell centroids — are a pure function of the node coordinates and topology.
They are computed on demand by {meth}`mesh.geometry() <aquaflux.mesh.Mesh.geometry>`,
which returns a {class}`~aquaflux.mesh.MeshGeometry` bundling the face and cell metrics:

```python
geometry = mesh.geometry()

geometry.face.area        # (n_faces,)       face areas
geometry.face.centroid    # (n_faces, dim)   face centroids
geometry.face.normal      # (n_faces, dim)   unit normals, owner-outward
geometry.cell.volume      # (n_cells,)       cell volumes
geometry.cell.centroid    # (n_cells, dim)   cell centroids
```

Cell volumes and centroids are computed by the divergence theorem from the bounding
faces; face geometry uses the scheme appropriate to the dimension — a straight edge in
2D, and a centre-fan triangulation of the polygon in 3D that gives a correct unit normal
even for a warped (non-planar) face. The right scheme is selected automatically from the
mesh dimension, so you never choose it by hand.

Geometry is returned fresh rather than stored on the mesh. This is deliberate: because
the node coordinates are a differentiable leaf of the mesh, deriving geometry on demand
keeps the dependency of every metric on the node positions visible to automatic
differentiation. Gradients with respect to node positions therefore flow through the
geometry — the basis for mesh-sensitivity diagnostics and shape optimization — and the
geometry can never go stale relative to the coordinates it comes from.

## Renumbering cells

The integer index each cell carries is arbitrary, but it is not irrelevant on large
meshes: a spatially local numbering (adjacent cells at nearby indices) improves cache
reuse in the residual assembly and helps the algebraic-multigrid (AMG) preconditioner
build a good coarse space. A mesh that arrives in a scrambled order can be renumbered
before it is used.

Renumbering is a build-time preprocessing step — it changes indices, never geometry or
physics — and is expressed as a strategy. {class}`~aquaflux.mesh.ReverseCuthillMcKee`
applies the reverse Cuthill–McKee (RCM) bandwidth-reducing ordering, which places
adjacent cells at nearby indices:

```python
from aquaflux.mesh import ReverseCuthillMcKee

renumbered, perm = ReverseCuthillMcKee().apply(mesh)
```

`apply` returns the renumbered mesh and the permutation used, so any per-cell data you
carry alongside the mesh (a spatially varying coefficient, a pinned-cell index) can be
remapped the same way. {class}`~aquaflux.mesh.IdentityReordering` is the no-op default,
and {class}`~aquaflux.mesh.RandomReordering` produces a deliberately scrambled ordering
for robustness testing. If you already hold a permutation, {func}`~aquaflux.mesh.permute_cells`
applies it directly.

## Mesh quality

Real meshes are imperfect, and a few cheap diagnostics quantify how imperfect. They run
once at build time — not inside the solve — and answer concrete questions about a mesh
before you rely on it:

- {func}`~aquaflux.mesh.face_planarity` — how planar each face is (1 means perfectly
  planar). A screen for badly warped faces.
- {func}`~aquaflux.mesh.closed_cell_residual` — the sum of outward face area-vectors per
  cell, which is zero for a properly closed cell; a useful check that the mesh is closed
  and consistently oriented.
- {func}`~aquaflux.mesh.centroid_iteration_shift` — how much a second centroid pass would
  move each face centroid, i.e. whether the face geometry is accurate on warped faces.

```python
from aquaflux.mesh import face_planarity, closed_cell_residual

planarity = face_planarity(mesh)          # (n_faces,), 1.0 = planar
residual  = closed_cell_residual(mesh)    # (n_cells, dim), ~0 for a closed mesh
```

## The connectivity API

Most users consume the geometry and never touch the raw index arrays. When you do need
them — to write a custom operator over the mesh — go through the connectivity objects
rather than the flat arrays, so the boundary convention is handled for you in one place.
{class}`~aquaflux.mesh.FaceCellConnectivity` gathers owner and neighbour cell values to
faces and scatters face contributions back to cells (masking the neighbour side on
boundary faces automatically); {class}`~aquaflux.mesh.FaceNodeConnectivity` traverses
each face's ring of nodes. These are the building blocks the solver's own operators are
written on.
