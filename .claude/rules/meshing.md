---
paths:
  - "aquaflux/meshing/**"
  - "validation/meshing_netgen/**"
---

# Rules — `aquaflux/meshing/` (mesh generation from CAD)

> **Provenance boundary (binding).** This file may cite the precursor codes and of-optical-radiation's
> `uvmesh` to inform *your* understanding; none of that may reach the shipped surface (root `CLAUDE.md`,
> **Comment Convention**). The dated record — the research survey, the Netgen probes, their numbers — is in
> `.claude/notes/meshing-log.md`.

**Status: the Netgen child process is built** (`netgen_process.py`, `_netgen_child.py`); nothing else of the
mesher is. The design below was agreed with the project owner on 2026-10-19. The probe harnesses are in
`validation/meshing_netgen/`. The package is not on the published API page (`docs/conf.py`
`PUBLIC_SUBPACKAGES`) until it has a mesher a user calls.

A polyhedral mesher for water-treatment reactors: STEP in, an aquaflux `Mesh` (and an OpenFOAM polyMesh)
out, with wall-resolved layers that join the bulk **conformally**. Conformity is a requirement here, not a
refinement: aquaflux has no non-conformal coupling, so the `uvmesh` route (structured pieces joined to a
polyhedral bulk through OpenFOAM 13 non-conformal couples) is not available to us at all.

## Agreed scope (binding — change only with the project owner, Principle 3.5)

- **Input: STEP**, read through the existing `aquaflux.io.cad` kernel. Other CAD formats later.
- **Core: selectable** — polyhedral (dual of tets) or hex core (octree), with **polyhedral transition
  cells, never tets or pyramids**, between the hex core and the layers.
- **Wall layers, two ways**: presets (the UV-lamp sleeve with flat or hemispherical ends, the chamber wall
  layer — `uvmesh`'s set) and, for arbitrary walls, prism layers extruded from a **polygonal surface mesh
  that is the dual of a surface triangle mesh**.
- **Pipes get a structured O-grid automatically**, detected from the CAD with an override, joined to the
  bulk conformally.
- **Baffles are a preset, detected automatically** (with an override) as faces with fluid on both sides.
- **No OpenFOAM tool anywhere in the pipeline** (no blockMesh, snappyHexMesh, polyDualMesh, gmshToFoam).
- **Output: the `Mesh` in memory, and an OpenFOAM polyMesh writer**, as the first iteration.
- **Scale: up to about 2M cells on a laptop.**
- **Dependencies: OCP (already the `cad` extra), Netgen (`netgen-mesher`, LGPL-2.1), numpy, numba.**

## The architecture: one conformal primal mesh, then one dual (agreed)

Build a **conformal hybrid primal mesh** of the whole fluid, then take **one feature-preserving dual** of it.
The dual of a conformal mesh is conformal, so every interface — layer to core, O-grid to bulk, hex core to
transition — is conformal by construction, and conformity is never hand-built at a seam (the `uvmesh`
seams are exactly where its mesh fails, below).

| Primal (built conformally) | Its dual (what the solver sees) |
|---|---|
| triangular prisms extruded from the wall triangulation | polygonal prism layers; the wall faces are the dual polygons of the wall triangulation |
| structured hex blocks (pipe O-grids, lamp annulus, cubed-sphere cap, wall layer) | hexes (the dual of a hex block is a hex block shifted half a cell) |
| octree hex core | hex core |
| pyramids on the hex core's quadrilaterals, then tets filling every gap (Netgen) | polyhedral transition cells — the pyramids and tets do not survive |
| tets throughout, in place of the octree | polyhedral core |

**Consequences to design for, not discover:**
- **A dual wall cell is half a primal cell thick**, so primal spacings are computed from the target dual
  ones (first-cell height, grading, the layer's outer cell).
- **Dual cells of a tet region come out larger than the tets** (`uvmesh` measured about 1.4x the gmsh size
  on Sozzi), which the size field must account for where the transition meets the layers.
- **Dual faces on a curved wall are non-planar** (vertices are face centroids and edge midpoints). aquaflux
  accepts them, but a non-planar face's cell volume depends on its triangulation (`io.md`: up to 4.6 % from
  VTK's on a polyDualMesh), so a planarization pass is in scope.
- **The dual must preserve features**: feature edges and corners stay sharp (dual boundary faces built from
  face centroids and feature-edge midpoints), and a cell at a concave edge is split into convex pieces
  (the published treatment of concave geometrical edges). Reimplement from the published description; the
  OpenFOAM `polyDualMesh` source is GPL and is not copied.

## Netgen: in a child process, reached through files (binding)

- ⚠️ **Netgen must never be imported into a process that has imported OCP.** The `netgen-mesher` wheel
  bundles its own OCCT (7.8.1) and loads it even for `netgen.meshing` alone; OCP's is 8.0.1. With OCP loaded
  first, every such process aborts at exit (SIGABRT, `malloc: pointer being freed was not allocated`, in
  OCP's `TCollection_AsciiString` destructor called from Netgen's `libTKDE`). The reverse order exits
  cleanly, but only because the first-loaded copy serves both, so OCP would run 7.8.1's inline code: not a
  basis to build on. Pinned by `validation/meshing_netgen/coexistence.py` (macOS arm64; **Linux, where CI
  runs, not yet measured**). Hence:
  - **`_netgen_child.py` is the one module that imports Netgen, and it runs only as a script, by path,
    in a fresh interpreter** (`python -P <script> <work dir>`), so it imports neither `aquaflux` nor OCP.
    `NetgenProcess` (`netgen_process.py`) is the parent side and never imports Netgen;
    `netgen_available()` finds it with `importlib.util.find_spec`. **Tests must not import Netgen either**
    (the test process holds OCP), so `tests/unit/test_meshing_netgen.py` gates with a `skipif` on
    `netgen_available()`, not `importorskip`. All pinned in `tests/unit/test_layering.py`: OCP only in
    `io/cad/kernel.py`, Netgen only in the child, nothing (package or tests) imports the child, the child
    imports only the standard library, numpy and Netgen.
  - **Hand-off is by file and arrays**: `request.json` + `input.npz` in, `result.npz` out, zero-based
    indices; a failure exits non-zero and `MeshingError` carries the end of Netgen's output. OCP writes
    the fluid shape as BREP (or STEP); an OCCT 8.0.1 BREP or STEP reads correctly in Netgen's 7.8.1.
  - The `meshing` extra pins `netgen-mesher==6.2.2608` exactly; the `test` extra carries it, so CI (Linux)
    runs the parent-with-OCP, child-with-Netgen arrangement on every PR.
- ⚠️ **Hand Netgen the fluid shape OCP built, never the drawing.** `OCCGeometry` of the Sozzi drawing reads
  only its first solid (the lamp: 1 solid, 3 faces) where `read_step` finds four.
- **What Netgen does for us** (`NetgenProcess`; pinned in `tests/unit/test_meshing_netgen.py`):
  - **`triangulate(shape, max_size=, min_size=, grading=)`**: size-graded triangles on the faces of the one
    solid in a file, **wound outward**, each tagged by its face. ⚠️ **Netgen's face index `k` (one-based) is
    the `k−1`-th face in OCP's `TopExp_Explorer` order of the same file** — that is the contract the kernel's
    face IDs must meet. `max_size` is Netgen's target, not a bound (edges reach ~1.7x).
  - **`fill(points, triangles, max_size=, grading=)`**: tets filling the region a closed, outward-wound
    surface bounds (holes wound into the hole), **keeping the surface exactly**: its points first, at their
    indices and coordinates. This is the conformal gap fill. Tets are returned positively oriented
    (Netgen's own come out all negative).
  - ⚠️ **The child checks the fill, because Netgen can be wrong silently**: given a hole wound out of the
    hole it fills straight through it, keeping every supplied point and boundary element. So the faces used
    by one tet must equal the supplied triangles and the tet volume the enclosed volume, else
    `MeshingError`. **Self-intersection is not detected** (two overlapping closed surfaces are each filled,
    overlapping).
  - ⚠️ **A point no triangle uses makes Netgen renumber every point**, so `check_closed_surface` refuses it
    before the child runs (with: shapes, finite, range, edge used once per direction, closed, positive
    enclosed volume).
  - ⚠️ **Netgen's arrays (`Coordinates()`, `Elements*().NumPy()`) are views of its memory**, freed with the
    mesh: copy before keeping (an `np.asarray` view returned garbage once the mesh was gone).
- **What it does not**: a supplied **quadrilateral** face, or a surface wound into the domain, is refused
  ("too many attempts"). Supply triangles only, wound counter-clockwise seen from outside the domain; the
  hex core's quadrilaterals are capped with pyramids we build, whose triangular sides Netgen sees.
- **Netgen's own prism layers are not used** (tet/prism only, and little documented); the extruder is ours.
- **Cost** (2026-10-19, Sozzi fluid, macOS arm64, netgen-mesher 6.2.2608): about 25k tets / 5.7k points per
  second, and `SetNumThreads(4)` changed nothing. A dual cell is one primal point, so a 2M-cell all-tet
  core would need on the order of 6 minutes of Netgen; the layers and a hex core are not Netgen's.

## Package and layering (agreed)

- **`aquaflux/meshing/`, host-side numpy and numba, not JAX.** It depends on `aquaflux.mesh` (it ends at
  `Mesh.from_csr`, which owns validation) and on `aquaflux.io.cad`; nothing imports it back except a case
  `MeshSource` (`aquaflux/case/mesh_source.py`).
- **CAD access goes through `io/cad/kernel.py`**, which grows what a mesher needs: face and edge topology
  and adjacency, face-tagged triangulation, closest-point projection and normals. Reading edge topology
  for features is allowed (project owner, 2026-10-19); recognizing a solid from edge types still is not
  (`io.md`).
- **The polyMesh writer belongs in `io/openfoam/`** (the deferred `MeshWriter`), not in `meshing/`.

## Components to build

1. **CAD topology** in the kernel, as above.
2. **Feature-preserving dual** (numba), from any conformal primal mesh.
3. **Prism-layer extruder**: smoothed normals, several normals at convex ridges, shrinking at concave
   corners, thickness limited by proximity, layer collapse or backtracking on collision.
4. **Block mesher** in place of blockMesh: transfinite interpolation with projection onto CAD or analytic
   surfaces, carrying the `uvmesh` topologies (four-quadrant annulus with the π/4 offset when a cap joins,
   the five-block cubed-sphere shell, the square-core pipe O-grid, wall-layer window masking) and pipe
   detection from cylindrical faces.
5. **Octree core**, 2:1 balanced, hanging nodes resolved by the tet-and-dual transition.
6. **Quality metrics** (non-orthogonality, skewness, convexity) and planarization.

**Phasing**: (1) CAD topology → surface triangulation → polygonal wall mesh → polyhedral core through the
dual; (2) prism layers; (3) block mesher, presets, pipe O-grids, baffles; (4) octree hex core.

**Not decided**: radial basis function (RBF) morphing *beside* transfinite interpolation — for spreading a
projection's displacement through a block (the pipe-to-chamber saddle), snapping octree nodes, and later
as the differentiable map from CAD parameters to `node_coords`. Raised with the project owner, not agreed.

## Rejected dependencies (why, so they are not re-proposed)

- **OpenFOAM tools**: excluded by scope; also GPL.
- **gmsh**: GPL-2, so at most an optional extra; its boundary-layer extrusion is fragile on curved CAD;
  `io-log.md` records it does not hand back surface parameters.
- **TetGen / MeshPy** (AGPL, and Triangle is non-free); **CGAL / pygalmesh / pymeshlab** (GPL).
- **fTetWild**: does not reproduce its input surface, so unusable at a conformal interface.
- **usd-brep** (NVIDIA, Apache-2.0, 1.0.0 of 2026-10-08): a kernel, so it would replace OCP, not part of the
  mesher; no STEP import, wheels only for CPython 3.12 on Linux and Windows. Revisit if both change.

## Traps carried from `uvmesh` (its documented failures, all at seams)

- A curved seam dualized with a near-tangent lip gives inverted faces.
- At a re-entrant (270°) edge — a pipe meeting the vessel — a 90° feature angle inverts dual cells; 100°
  fixes it at the cost of cutting genuine 90° edges. The concave-edge split above is the real answer.
- Tet slivers become bad dual faces; Netgen's optimization removed them there.
- A size jump between nested structured layers skews cells; balance the counts.
- A layer too thin to contain the near-wall jet left a seam the solve could not settle across
  (the Sozzi lamp layer was thickened from 15 to 22 mm for it).
