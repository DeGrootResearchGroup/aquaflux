---
paths:
  - "aquaflux/meshing/**"
  - "validation/meshing_netgen/**"
  - "validation/meshing_dual/**"
---

# Rules — `aquaflux/meshing/` (mesh generation from CAD)

> **Provenance boundary (binding).** This file may cite the precursor codes and of-optical-radiation's
> `uvmesh` to inform *your* understanding; none of that may reach the shipped surface (root `CLAUDE.md`,
> **Comment Convention**). The dated record — the research survey, the Netgen probes, their numbers — is in
> `.claude/notes/meshing-log.md`.

**Status: phase 1 built** — STEP region → Netgen tets → polyhedral dual → `Mesh` / polyMesh, from a case
file (`CadMesh`) or `CadMesher`: the Netgen child process (`netgen_process.py`, `_netgen_child.py`),
`tetrahedral_mesh` (`tetrahedra.py`), the feature-preserving dual (`dual.py`, cells at sharp concave edges
cut into pieces) and the mesher (`cad_mesher.py`). Prism layers, blocks and the hex core are not built.
The design below was agreed with the project owner on 2026-10-19. The probe harnesses are in
`validation/meshing_netgen/` and `validation/meshing_dual/`. The package is on the published API page.

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

## The feature-preserving dual (`dual.py`) — built

`PolyhedralDual(feature_angle, feature_edges=None).build(primal: Mesh) -> Mesh`. Dual cell `i` is primal
point `i` (its first piece); the further pieces of cut points follow, by point. Vertices: primal cell
centroids, boundary-face centroids, feature-edge midpoints, points on ≥ 2 feature edges. Each primal edge gives the face between its ends' cells (ring of cell centroids; on the
boundary closed through its two boundary faces' centroids, and its midpoint if a feature). Each boundary
point gives its fan of face centroids, cut along feature edges into one face per sector with the point as a
vertex. Features: fold beyond `feature_angle` (no default), every patch border, named edges. Two numba walks
(cells around each edge, boundary faces around each point); faces of edges assembled in numpy.
- **Accepted primal**: 3D, non-periodic, one cell zone, no patch on interior faces, closed manifold
  boundary, every point on a face. Each refused by name.
- **Cells at sharp concave folds are cut into pieces** (`_Pieces`): every point on an edge folding concavely
  by more than `feature_angle` (sign by `boundary_folds`). **Why it is required for correctness, not
  quality**: uncut, such a cell wraps around the fold, and on Netgen tets `Mesh.geometry()`'s centroid-based
  orientation turned exactly one face per cell inward (topology closed, consistently oriented faces summing
  to zero; log). **The rule**: the walls are the feature sectors at the point that touch one of its concave
  edges, each with its area-weighted normal at the point; each primal cell around the point goes to the
  wall its centroid lies deepest behind (`argmax -(c - p)·n`), and each wall's cells are one dual cell
  (face-connected, else `NotImplementedError`). Merging sectors across convex edges is wrong (the L's end
  corner would not be cut); a depth rule over *all* sectors is wrong (it cuts convex corners, and its
  pieces at the L's end are not convex).
  - **Topology** (the barycentric dual, collapsed except where pieces meet): each interior primal face
    between pieces gets a vertex on its dual edge, inserted into **every** dual face along that edge (also
    those of edges not at the point, else a hanging vertex closure cannot see); a dual face whose pair of
    pieces changes around its edge is cut into one face per run, closed through a vertex of the edge; each
    cut face gives a quad `[p, edge vertex, face vertex, edge vertex]` between the two pieces; boundary
    fans are also cut where the faces' cells change piece. The point itself becomes a vertex.
  - ⚠️ **The added vertices must lie on the surface between pieces, or the cut fails the same way.** At
    face and edge centroids the separating surface zig-zags along tet faces and Netgen L cells still
    failed closure (maxh 0.01: 6 cells, 3 points, ~4e-5). So: a face's vertex where its dual edge crosses
    the plane `(n_b − n_a)·(x − p) = 0` (fraction kept to [0.1, 0.9] so a centroid on the plane does not
    give coincident vertices; mean over cutting points); an interior edge's hub the least-squares point
    on every interface it meets and in the dual face's plane (on the triple line where three pieces meet);
    a boundary edge's hub on the segment between its two boundary-face centroids, where it crosses.
  - Measured (log): every Netgen L, cylinder, sleeve and Sozzi dual closes; Sozzi's two junctions give 24
    extra pieces; pieces' non-convexity within the uncut cells' range. A face cut around two points of a
    three-wall corner meets two different planes, so that corner is near-convex (≤ 0.05), not exact.
  - A gently concave wall is not cut: a lamp-sleeve hole folds concavely ≤ 32° per edge and its duals close.
- **Away from features the dual's boundary runs through face centroids**, inside a curved wall by the chord:
  dual volume 2.2 % / 0.57 % under the tets' on a cylinder at maxh 0.02 / 0.01 (second order). The dual
  itself does not know the true surface; `build_traced` says where each boundary vertex came from, and
  `CadMesher` projects them onto the drawing (below).
- ⚠️ **Memory is the open problem at the 2M target**: the primal held as a full `Mesh` costs ~5 KB per
  point and the dual's build ~9 KB per point (111k points; log), so 2M points would need ~28 GB. The likely
  fix is a numpy-only primal (the primal is never solved on) and a leaner assembly; not yet done.
- **Quality, not correctness, on Netgen duals**: `checkMesh` passes structured and Kuhn-tet duals; Netgen
  duals, cut ones included (L, Sozzi), pass topology and closure but report concave cells by face planes
  (1.5–7.7 %), a few low-quality face decompositions and small volume ratios (log). The quality and
  planarization component is for these.
- Pinned in `tests/unit/test_meshing_dual.py` (the cut on hex and Kuhn Ls and a three-wall corner) and
  `test_meshing_netgen.py` (a Netgen L, the case placement is for); scale and convexity, Sozzi included,
  in `validation/meshing_dual/dual_scale_convexity.py` (log).

## The CAD mesher (`cad_mesher.py`) — built

`CadMesher(max_size, feature_angle, min_size=None, grading=None, netgen=NetgenProcess()).mesh(model,
region, patches=None, others="walls") -> Mesh`: `write_brep` → `triangulate` (face-tagged) → `fill` →
`tetrahedral_mesh` (boundary faces in patches by their triangle's CAD face) → `PolyhedralDual.build_traced`
→ boundary vertices projected onto the CAD (below), with
**the triangle edges between two CAD faces whose CAD edge folds beyond `feature_angle` named as features**
(so a CAD edge stays sharp whatever its triangles fold by; pinned by
`test_the_cads_sharp_edges_are_kept_whatever_the_triangles_fold_by`). Netgen is injected, so
`tests/unit/test_meshing_cad_mesher.py` runs the whole pipeline on a stub cube (Kuhn tets) without Netgen
or OCP.
- **Patches are chosen by face selection** (`FaceSelection.faces(RegionFaces) -> face indices`; agreed
  with the project owner, 2026-10-19, who intends a face picker in the browser interface — a click on a
  face is a point on it, which is all `FacesNear` stores): `FacesNear(points)` — the face each point projects onto (a pipe's
  end); `FacesOn(solids, tolerance=None)` — faces whose sample point lies within tolerance (unset 1e-6 of
  the region's extent) of the solids' boundary (a lamp taken out of the water). A face's sample is the
  centroid of one of its triangles projected onto it. Faces nobody selects go to `others`; two
  selections sharing a face, a selection finding none, a reserved name, or `others` naming a selection
  are refused (names before Netgen runs). The CAD records carry no face names, which is why selection is
  geometric.
- **Every boundary vertex is projected onto the drawing** (`_onto_the_drawing`), from the dual's
  `TracedDual` (`vertex_face`: the primal boundary face a vertex is the centroid of; `vertex_edge`: the
  primal boundary edge a midpoint or a cut's hub stands for): a face centroid onto its triangle's CAD
  face; an edge vertex onto the CAD face of its two triangles, or, when they lie on two faces, onto the
  CAD edge between them (`CadModel.project(..., edges=)`), the nearest of several. A primal point is on
  the drawing already. Sozzi after projection (log): patch areas −0.5 to −1.5 % (flat polygons between
  surface points, and the inlet/outlet rims' chords), boundary vertices on the drawing to 4e-17,
  non-convexity unchanged.
  - ⚠️ **A CAD edge can come in pieces** (Sozzi's inlet rim is two edges): project each candidate from the
    vertex's own position — chaining them carried a vertex 7 mm to the pieces' shared end (non-convexity
    1.48). Pinned by `test_a_vertex_on_an_edge_drawn_in_two_pieces_lands_on_the_nearer_piece`.
  - ⚠️ **Netgen's arrays are int32**: a sorted-triple key of 7k points overflows it, and silently put
    lamp triangles on the vessel's face. Keys are built in int64.
- **Cost** (Sozzi water, maxh 0.01, 13297 cells, one run, macOS arm64): 8.7 s and 0.77 GB peak for the
  whole `aquaflux mesh --case --openfoam` before projection; `CadMesher.mesh` 6.8 s with it (3.8 → 5.6 s
  at maxh 0.02: the kernel's one query per vertex, ~15k vertices).
- `tetrahedral_mesh(points, tetrahedra, patches=None)` numbers faces in first-sighting order (the test
  fixture `tetrahedral_grid_3d` is built with it, bit-identical to its former hand assembly).

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
    the `k−1`-th face in OCP's `TopExp_Explorer` order of the same file** — the contract the kernel's face
    IDs meet. Pinned end to end by `test_netgen_numbers_a_written_regions_faces_as_its_topology_does`
    (Sozzi water, `CadModel.write_brep` → Netgen: per-face area, and sampled centroids projected back). `max_size` is Netgen's target, not a bound (edges reach ~1.7x).
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
  `MeshSource` (`CadMesh`, `aquaflux/case/mesh_source.py`) and the test fixtures.
- **CAD access goes through `io/cad/kernel.py`**, via `CadModel` and a `CadRegion` (`io.md`, "Regions for a
  mesher"): topology with signed edge folds and adjacency, face-tagged triangulation, the BREP file Netgen
  reads, closest-point projection with outward normals — all numbered in the exploration order Netgen
  numbers faces in. Reading edge topology for features is allowed (project owner, 2026-10-19);
  recognizing a solid from edge types still is not.
- **The polyMesh writer lives in `io/openfoam/`** (`OpenFOAMWriter`, `write_openfoam`; `io.md`), not in
  `meshing/`; the mesher hands it a `Mesh`.

## Components to build

1. **CAD topology** in the kernel, as above (built).
2. **Feature-preserving dual** (numba), from any conformal primal mesh (built), and the CAD mesher
   around it, its boundary projected onto the drawing (built).
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
  fixes it at the cost of cutting genuine 90° edges. The concave-edge cut above (built) is the answer.
- Tet slivers become bad dual faces; Netgen's optimization removed them there.
- A size jump between nested structured layers skews cells; balance the counts.
- A layer too thin to contain the near-wall jet left a seam the solve could not settle across
  (the Sozzi lamp layer was thickened from 15 to 22 mm for it).
