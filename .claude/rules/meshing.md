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
file (`MeshSpec`, made by `aquaflux mesh mesh.yaml` into a `.aqmesh`; `case.md`) or `CadMesher`: the Netgen child process (`netgen_process.py`, `_netgen_child.py`),
`element_mesh` (`elements.py`: tetrahedra, prisms, hexahedra and pyramids), the feature-preserving dual
(`dual.py`, cells at sharp concave edges cut into pieces), prism layers on triangulated walls and
hexahedral layers on walls meshed with structured quadrilaterals (`layers.py`, `mapped.py`,
`polygons.py`), pipes' cores as structured O-grids (`pipes.py`, `pipe_core.py`), the mesher
(`cad_mesher.py`) and the octree hex core (`octree.py`, `hex_core.py`, `regions.py`), selectable in
`CadMesher` and the mesh file (`MeshSpec.core`). The block mesher is built as a library (below), not yet
wired into `CadMesher` or the mesh file.
Face planarization (`planarize.py`) is built and run by `CadMesher` when given a `planarizer` (off by
default; `planarizer: {kind: FacePlanarizer}` in a case file).
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
  by more than `feature_angle` (sign by `boundary_folds`). **Why, now that the mesh orients faces by
  shared edges** (`mesh.md`): an uncut cell closes but wraps around the fold — worst non-convexity 1.2 on the
  Netgen L at maxh 0.01, against 0.35 cut (log). Under the former centroid-based orientation it was a
  correctness matter: one face per such cell was turned inward. **The rule**: the walls are the feature sectors at the point that touch one of its concave
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
  - **The added vertices lie on the surface between pieces**, or the separating surface zig-zags along tet
    faces: Netgen L at maxh 0.01, worst non-convexity 0.56 with the cut faces' vertices at face centroids,
    1.1 with the cut edges' hubs at midpoints, 1.23 with both (0.35 placed); under the former centroid
    orientation the unplaced cut also failed closure. So: a face's vertex where the segment between its
    cells' centroids crosses the plane `(n_b − n_a)·(x − p) = 0` (fraction kept to [0.1, 0.9] so a
    centroid on the plane does not give coincident vertices; mean over cutting points) -- **unless that
    crossing lies in front of the walls** (depth ≤ 0, i.e. in the solid beyond the corner, as beside a
    long pipe hexahedron at a junction), then where the dual edge through the face's centroid crosses.
    ⚠️ Not the bent edge everywhere: it made plain tet duals worse (Sozzi 0.02 worst 0.33 → 1.12; log). An
    interior edge's hub is the least-squares point on every interface it meets and in the dual face's
    plane (on the triple line where three pieces meet), **dropping singular values under
    `_PLANES_APART` (0.05, about 3°)**: neighbouring points of a curved edge carry nearly parallel planes
    whose exact intersection is far off (131 mm on Sozzi's inlet saddle); a boundary edge's hub on the
    segment between its two boundary-face centroids, where it crosses.
  - **Layers' columns are cut with their base point** (`PolyhedralDual(columns=)`: each row a boundary
    point, then the points grown above it; `CadMesher` passes `LayerExtrusion.columns`). The layer points
    above a sharp concave edge have dual cells that turn the corner too; uncut, their faces along the
    corner fold and the cells do not close (Netgen L with layers on every face: 22 unclosed on `main`
    before this). Each column point is cut by its base point's walls, origin at itself; its pieces follow
    the base's, by column. Pinned by `test_layers_turning_a_concave_edge_are_cut_up_their_columns` (a voxel
    L with hexahedral layers and a lattice core, no Netgen).
  - Measured (log): every Netgen L, cylinder, sleeve and Sozzi dual closes; Sozzi's two junctions give 24
    extra pieces; pieces' non-convexity within the uncut cells' range. A face cut around two points of a
    three-wall corner meets two different planes, so that corner is near-convex (≤ 0.05), not exact.
  - A gently concave wall is not cut: a lamp-sleeve hole folds concavely ≤ 32° per edge and its duals close.
- **Away from features the dual's boundary runs through face centroids**, inside a curved wall by the chord:
  dual volume 2.2 % / 0.57 % under the tets' on a cylinder at maxh 0.02 / 0.01 (second order). The dual
  itself does not know the true surface; `build_traced` says where each boundary vertex came from, and
  `CadMesher` projects them onto the drawing (below).
- ⚠️ **Memory at the 2M target is set by temporaries, not by what is held**: Sozzi, 660k cells, peak
  footprint 3.33 GiB (macOS allocator cache off; ~4.7 KB per cell, so ~9.6 GB at 2M), held after meshing
  0.68 GiB. The peak is now `Mesh.from_csr` (the dual's, then the primal's: `element_mesh`'s sorts and
  `validate`'s `np.unique`); before the blocked centre fan and the dual computing its primal geometry
  once (`build_traced` passes one `MeshGeometry` down) it was the primal's `Mesh.geometry`, four times,
  4.71 GiB. Netgen's child peaks apart (1.44 GiB at 660k). Measured by
  `validation/meshing_dual/memory_by_stage.py` (log, "Memory by stage").
  With mapped, layered walls (since #790's fix): 671030 cells at maxh 0.0025, peak 3.21 GiB; 1001489 at
  0.002, 4.80 GiB (one run each, 2026-10-21; log, "the fill's overlap at maxh 0.0025").
  - ⚠️ **On macOS read footprints with `MallocLargeCache=0`**: libmalloc keeps freed large blocks and
    charges them to the process until a different-sized allocation lands on top, so a default run reads
    up to ~40 % above what is live (266k cells: 2.40 against 1.72 GiB) and "held" stages look large.
- **Quality, not correctness, on Netgen duals**: `checkMesh` passes structured and Kuhn-tet duals; Netgen
  duals, cut ones included (L, Sozzi), pass topology and closure but report concave cells by face planes
  (1.5–7.7 %), a few low-quality face decompositions and small volume ratios (log). `aquaflux.mesh`'s
  quality metrics reproduce these checkMesh numbers; planarization (below) is for them.
- Pinned in `tests/unit/test_meshing_dual.py` (the cut on hex and Kuhn Ls and a three-wall corner) and
  `test_meshing_netgen.py` (a Netgen L, the case placement is for); scale and convexity, Sozzi included,
  in `validation/meshing_dual/dual_scale_convexity.py` (log).

## The CAD mesher (`cad_mesher.py`) — built

`CadMesher(max_size, feature_angle, min_size=None, grading=None, layers=None, cap_height=ClearedCaps(),
pipe_core=PipeCore(), planarizer=None, core=None, netgen=NetgenProcess()).mesh(model, region, patches=None, others="walls",
layered=(), mapped=(), pipes=None, on_stage=None) -> Mesh` (`on_stage`, a `Stage`, is told each stage's name as it starts,
only the stages this mesh runs; `aquaflux mesh` prints them, `case.md`): `write_brep` → `triangulate` (face-tagged; again with the rims divided
alike where a pipe candidate's need it) → pipes found (below) → the `mapped` patches' faces and the pipes' remeshed as structured quadrilaterals (below) →
layers on the `layered` patches (below) → the pipes' cores filled with O-grids, out of the fill (below)
→ every quadrilateral left for the fill
capped with a `PyramidCaps` pyramid on the fill's side (`caps.capped`: sized by `ClearedCaps`, below, or a fixed share of its **first** edge, which on a mapped grid runs round the band; Netgen takes triangles
only) → with a `core`, the octree hex core inside the fill's surface, its capped surface added as a hole
(`_cored`, below) → `fill` → `element_mesh` (tets, prisms, hexahedra, the caps as pyramids; boundary faces in
patches by their CAD face; points no cell uses dropped) → `PolyhedralDual.build_traced` (the layers'
columns passed); with nothing left for tetrahedra (a pipe open at both ends) Netgen's fill is not run
→ boundary vertices projected onto the CAD (below) → if `planarizer` is set, the faces flattened by
moving the centroids of the primal cells outside the layers (`_free_centroids`; "Face planarization"), with
**the triangle edges between two CAD faces whose CAD edge folds beyond `feature_angle` named as features**
(so a CAD edge stays sharp whatever its triangles fold by; pinned by
`test_the_cads_sharp_edges_are_kept_whatever_the_triangles_fold_by`). Netgen is injected, so
`tests/unit/test_meshing_cad_mesher.py` runs the whole pipeline on a stub cube (Kuhn tets) without Netgen
or OCP.
- **Patches are chosen by face selection** (`FaceSelection.faces(RegionFaces) -> face indices`; agreed
  with the project owner, 2026-10-19, who intends a face picker in the browser interface — a click on a
  face is a point on it, which is all `FacesNear` stores): `FacesNear(points)` — the face each point projects onto (a pipe's
  end; `FacesNear.landed(faces)` is each point's face in order, which the browser's picker removes
  points by); `FacesOn(solids, tolerance=None)` — faces whose sample point lies within tolerance (unset
  1e-6 of the region's extent) of the solids' boundary (a lamp taken out of the water). A face's sample
  is the centroid of one of its triangles projected onto it (`RegionFaces.sampled(model, region,
  corners, face)`, any triangulation tagged by face: Netgen's here, the kernel's in the drawing view).
  **`assign_patches(faces, patches)` is the one rule** for each face's patch (the mesher and `aquaflux
  drawing` both call it). Faces nobody selects go to `others`; two
  selections sharing a face, a selection finding none, a reserved name, or `others` naming a selection
  are refused (names before Netgen runs). The CAD records carry no face names, which is why selection is
  geometric.
- **Every boundary vertex is projected onto the drawing** (`_onto_the_drawing`), from the dual's
  `TracedDual` (`vertex_face`: the primal boundary face a vertex is the centroid of; `vertex_edge`: the
  primal boundary edge a midpoint or a cut's hub stands for; `vertex_cell`: the primal cell a vertex is
  the centroid of — the dual's first `n_cells` vertices — which is what a flattening pass may move): a face centroid onto its triangle's CAD
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
  whole `aquaflux mesh` (then a case-file mesh section, with `--openfoam`) before projection; `CadMesher.mesh` 6.8 s with it (3.8 → 5.6 s
  at maxh 0.02: the kernel's one query per vertex, ~15k vertices).
- `element_mesh(points, tetrahedra=None, prisms=None, hexahedra=None, pyramids=None, patches=None,
  patch_types=None)` numbers cells in that order and faces in first-sighting order (the test fixture `tetrahedral_grid_3d` is built with it, bit-identical to its former
  hand assembly); patches take triangles and quadrilaterals by corners. `HEXAHEDRON_CORNERS` (the unit
  cube's corners in hexahedron order) lives here, shared by the blocks and the octree core.

## Prism layers (`layers.py`) — built, on triangles and quadrilaterals

`PrismLayers(first_height, growth, count, gap_share=0.4, max_stretch=2.0, max_halvings=8, min_top_share=0.1).extrude(points,
faces: Polygons, layered) -> LayerExtrusion(points, prisms, hexahedra, inner, walls, sides, side_face,
columns)`: one extruder for both kinds, prisms over triangles and hexahedra over quadrilaterals, layer by
layer in the surface's order; `inner` and `walls` are `Polygons`; `columns` each moved point then its
layer points. A mesh file states it as `MeshSpec.layers` with the patch names in `MeshSpec.layered` (both
or neither). A surface of both kinds travels as a `Polygons` (`polygons.py`: two arrays, never one padded
one, so a face is never read with a missing corner; a per-face quantity runs over the triangles first).
- **Inside-out test per kind**: a prism's three tets of its standard split (as before); a hexahedron's
  eight corner tetrahedra (each corner and its three edge neighbours), a right cell giving all negative
  (lower face wound out of the region). Pinned by
  `test_a_hexahedron_that_would_turn_over_at_a_concave_edge_is_saved_by_thinning`.
- **Sized for the dual, exactly**: the dual's layer boundaries are the prisms' centroids, at `first_height ·
  Σ growthᵏ`; each prism's top is placed so its centroid is there, so the prisms alternate thicker and
  thinner (the first is `2·first_height`; the second only `2·first_height·(growth − 1)`; the alternation
  decays as `growthᵏ`). Hence **`growth > 1` is required** — equal dual layers need zero-height prisms.
  Flat walls: exact (unit cube, maxh 0.1: 0.005, 0.011, 0.0182, 0.02684, 0.03721 as targeted).
- **Directions**: angle-weighted mean of the layered triangles' inward normals; stretched by `1/cos` (≤
  `max_stretch`) so every face at the point is cleared by the full thickness (a cube's shell is exactly the
  cube shrunk by it). **Where a layered wall meets a bare patch the direction is turned into the bare face's
  plane** (two bare planes: along their intersection; three: refused), so the prisms' sides lie on the bare
  face and its triangles are drawn in to the layers' top (`walls`; `sides` are the boundary quads).
  Deferred: several normals at a sharp convex ridge (one averaged normal shears the prisms there).
- **Thickness**: limited to `gap_share` of the distance along the direction to the surface (ray cast within
  2.5 thicknesses), then halved at the points of any inverted prism or turned drawn-in triangle, **or at
  both ends of any edge of the layers' top shorter than `min_top_share` of the wall edge under it**, until none
  is (`max_halvings`), and **tapered**: no point more than 1.5× its thinnest neighbour, iterated to a fixed
  point. ⚠️ A diffusive smoothing (10 sweeps per pass) thinned a whole slab's middle to 30 % of the gap
  limit from its edges' halving; the ratio limit is local.
- ⚠️ **Where walls meet at under 180° (seen from the fluid) the top collapses** (#790): the edge's column
  moves along the bisector, so the top face beside it is narrower than its wall face by about the
  thickness, and a sliver once the thickness nears the wall's spacing (Sozzi at maxh 0.0025: the lamp's
  first quadrilateral at the end wall 2.5 mm → 0.032 mm). A pyramid cap on a sliver stands through the
  other wall's top -- under a fixed `cap_height` share (sized by the long first edge) Netgen's fill stopped,
  "boundary mesh is overlapping"; the default `ClearedCaps`' fold limit instead shrinks it to ~10 µm (it
  meshes, smallest cell 25× smaller than with the floor). Hence
  `min_top_share`: 0.1 sits under the uniform narrowing inside a pipe whose layers take `gap_share`
  (`1 − 2·gap_share` = 0.2), so only collapse is thinned (on Sozzi at 0.0025 one ring of 23 quadrilaterals,
  halved once; 222 of 71732 columns thinner with the taper round it).
  The cause is the extruder's, not the caps': with triangle walls the junction's top faces carry prisms and
  nothing is capped, so that arm meshed. Pinned by `test_a_top_face_collapsed_where_walls_meet_is_thinned_until_its_caps_clear_the_fill`
  and the Netgen cube `test_capped_hexahedral_layers_meeting_at_an_edge_leave_a_surface_netgen_fills`.
  Open: a drawn-in bare face can shrink the same way (not checked); smoothing the columns' directions
  near the edge would spread the shrink instead of thinning (not built). Log: "the fill's overlap at maxh 0.0025".
- **The fill gets only the layers' top plus the bare triangles** (`inner`); `CadMesher._filled` hands Netgen
  just the points that surface uses (Netgen renumbers otherwise) and maps the tets back.
- ⚠️ **Projection must follow the layers** (`_Columns`): a wall face's centroid moves onto the CAD by up to the
  chord's sag (~0.4 mm on the lamp at maxh 0.02), more than a thin first cell; moving it alone inverted cells
  (39 on Sozzi) and would set the first cell's height from the chord. So each prism centroid above a wall
  face moves by `1 − (c/t)²` of its displacement (`c` the centroid's height, `t` the layers' thickness).
  Sozzi at maxh 0.02 (first 0.5 mm, growth 1.2, 5 layers on walls and lamp): first dual layer boundary 0.507
  / 0.508 mm from the CAD on lamp / vessel; layers 2–3 drift with curvature (−2 % lamp, +6 % vessel; log).
- Measured on Sozzi (log): 46124 cells at maxh 0.02 (10446 bare), 59395 at 0.01 (13297 bare), all closed,
  every boundary vertex on the drawing. ⚠️ Inside the 19 mm pipes the core left after the layers is narrow
  and its polyhedra skewed (images in the log): the pipe O-grid of the block mesher is the answer there.
- Pinned in `tests/unit/test_meshing_layers.py` (no Netgen; quadrilateral walls from
  `tests/support/meshes.voxel_surface`) and `test_meshing_netgen.py` (Sozzi, the first height from the true
  wall).

## Structured quadrilateral walls (`mapped.py`) — built

`MappedQuads(corner_tolerance=1e-9).remesh(points, triangles, face, chosen, vertices, onto) ->
MappedSurface(points, faces: Polygons, face)`; `CadMesher.mesh(mapped=...)` / `MeshSpec.mapped` name the
patches (agreed with the project owner, 2026-10-20: quads from mapped meshing of chosen CAD faces, one
extruder for both kinds, pyramid caps to the fill; a block's faces as the source were the alternative
not taken). Layered, a mapped wall carries hexahedra, so the dual's wall cells are hexahedra.
- **Two shapes, read from the face's own triangles** (boundary edges used once, chained into loops in
  their winding): **a band** (two loops: a cylinder or cone, a flat ring) -- rows from the loop with more
  points toward the other, rolled to start across from it, `round(distance / mean spacing)` rows;
  **four-sided** (one loop, four points within `corner_tolerance` × extent of the drawing's vertices) --
  a Coons grid of its sides. Anything else (Sozzi's hemispherical lamp tip: one loop, no four corners;
  the vessel's cylinder: three loops, the riser's hole) is refused by name.
- **Netgen's edge points, kept**: the grid's sides are the triangulation's, which the neighbouring faces
  share (per-edge `maxh` is only a bound; an exact count needs `edge_divisions`, below, used for pipe
  rims only). **Unequal opposite
  sides**: the grid stops a row short and a strip of triangles (`_zipped`, shorter diagonal first) joins it
  to the shorter-counted side; a four-sided face with both pairs unequal is refused.
- New points are interpolated in space and moved onto the face by the injected `onto` (the mesher's
  `CadModel.project(faces=)`); no surface parameters are read. **Wound by construction** (rows run from a
  loop, in its winding, into the face), pinned by the outward-normal assertion; a reorientation step was
  dominated and deleted.
- Measured on Sozzi (lamp cylinder and both pipes mapped; log): 98.6 / 98.8 % of the lamp's / pipes'
  wall faces are quadrilaterals of hexahedral dual cells, 28393 hexahedra; the lamp's first dual layer at
  0.5033 mm (5-95 % 0.5025-0.5040) against 0.5069 (0.5066-0.5087) with triangles. ⚠️ Faces over 80°
  non-orthogonal: 11 against 2, inside the outlet pipe's entry and at the inlet saddle, where five layers
  leave the pipe a core of a few millimetres for caps and tets -- the O-grid's case.
- ⚠️ **Each cap's apex is a dual cell**, and its size is set by the cap's height: low caps (the old 0.25
  × first edge) squeeze ~50 cells under 2 mm above Sozzi's lamp layers (volume ratio p99 37). **Caps are
  sized to their room by default (`ClearedCaps`, `caps.py`)**: `target` 0.7 × the first edge (the width
  round a mapped band: short round a pipe, long round the lamp), lowered so a side leans at most `fold`
  0.6 of half the fill's angle over each edge (0.25 × edge over a concave right angle, ~0.69 on a flat
  wall) and to `gap` 0.4 of the distance to the surface opposite (`regions.first_hits`, shared with the
  layers' gaps) or of the core's clearance. Five draws (log, "taller caps sized to their room"): Sozzi
  small cells 45-54 → 5-8, ratio p99 37 → 16-17, non-orth p99 34 → 32°, faces > 50° 153-170 → 120-129;
  the open vessel's core 1785-1832 → 257-322 small, ratio p99 78-81 → 17.5, **but faces > 50° 872-964 →
  1402-1686**, nearly all between neighbouring core caps' apex cells (across the crevice between them).
  The old tall-cap objection (#785: the pipes' narrow cores) no longer applies -- they are O-grids.
  **Measured and rejected**: a fixed 0.45 (the vessel's faces > 50° to 1899-3254); target 1.0 (Sozzi
  ratio p99 7 but non-orth p99 42°, the apexes then larger than the cells round them); Owen-Saigal
  pyramids merged from the fill's tetrahedra without their cleanup (`MergedCaps`, kept on the pushed
  branch `claude/merged-caps-experiment`, commit 60c3e19f): two of five Sozzi draws unmergeable, and the
  octree core's concave steps send 3316 of 7528 quads back to a point of their own; earlier, a crevice
  limit, the shortest edge and a diagonal split with no caps. Netgen's own quad closing
  (`CloseOpenQuads`) does not close quads given from Python (log). Background: log, "hexahedra to
  tetrahedra: what the literature and other meshers do". Pinned by `tests/unit/test_meshing_caps.py`
  and `test_a_caps_height_is_measured_on_its_quadrilaterals_first_edge`.
- Pinned in `tests/unit/test_meshing_mapped.py` (bands of equal and unequal rims, Coons grids with and
  without a strip, the refusals) and `test_meshing_netgen.py` (the L with all six rectangles mapped: every
  wall face with no vertex on a CAD edge a hexahedron's; Sozzi's lamp and pipes).

## Pipes: structured O-grid cores (`pipes.py`, `pipe_core.py`) — built

A pipe's wall is a mapped band, layered like any wall, and **the core its layers leave is filled with a
structured O-grid** (`PipeCore`) whose outer quads are the layers' top quads, point for point -- no
seam, no caps inside the pipe (agreed with the project owner, 2026-10-20, over replacing whole pipes
with the block mesher's `PipeOGrid`: the extruder keeps the wall cells and their hand-over to the
vessel's layers at the junction; a few layers then a full O-grid inside gives the grading control).
- **Found** (`find_pipes`): a `CylinderFace` with the fluid inside (`solid_inside`), two loops of equal
  counts, a multiple of four, nothing else of the surface inside it between its rims (refuses a vessel
  shell round a lamp); `rim_counts=False` skips the counts, for the drawing view (asked before any
  rim is divided). `CadMesher.mesh(pipes=None)` takes every such face (and maps it); a tuple names
  the patches, each face of which must be one (else `ValueError` naming the reason); `()` none.
  `MeshSpec.pipes` / `MeshSpec.pipe_core` in a mesh file.
- **Rims divided alike** (`rim_divisions`, #789): Netgen's rim counts follow the size (Sozzi: 12/13 at
  maxh 0.005, 17/17 at 0.0035), so a candidate (fluid-filled cylinder, two loops) whose loops differ or
  are not a multiple of four gets both rims `4·ceil(max/4)` segments -- never coarser than Netgen chose
  -- shared among a rim's edges by length (largest remainder, ≥ 1 each; Sozzi's inlet rim is two edges,
  15 + 5); an edge on two candidates' rims takes the larger. `CadMesher.mesh` then triangulates **a
  second time** with `NetgenProcess.triangulate(edge_divisions=)` (only when something is asked) and runs
  `find_pipes` on that; `patch_of_face` comes from the first triangulation (per CAD face, unchanged).
  Both Sozzi pipes cored at every maxh 0.02-0.0035 (log, "Pipe rims divided alike"); at 0.0025 the
  rims already fit (24/24); the fill's failure there was #790 (layers, below). Pinned by
  `test_meshing_pipes.py` (the counts, a two-edge rim, the second triangulation) and
  `test_netgen_divides_the_sozzi_pipes_rims_as_asked_so_both_are_pipes`.
- **`MappedSurface.grids`**: each equal-rimmed band's points as a grid `(rows + 1, n)`, rows round it,
  columns joined. ⚠️ **A band's rows flatten within `RIM_ROWS` (2) of a rim that is not flat**: each row
  is the interpolation of the two rims flattened along the axis, plus each rim's bow fading over two rows.
  Plain interpolation kept the riser's 1 mm saddle in every row near the junction (pipe p99
  non-orthogonality 8.2° against 6.0°, worst 62.7° against 58.6°, two draws; log). A band between flat
  rims (the lamp) is unchanged. Pinned by `test_a_bands_rows_flatten_within_two_rows_of_a_rim_that_is_not_flat`.
- **`PipeCore(core_size=0.5, core_bow=0.0, shell=None, junction_rows=2).fill(points, rings)`**: per
  ring of `4 k` points, a `k × k` core (half-width `core_size` × the ring's mean radius, its first corner
  toward ring point 0; `core_bow` as `PipeOGrid`'s, through `_CoreSides`, nodes at equal angles) by
  `blocks.transfinite`, and a shell of one radial line per ring point; consecutive cross-sections joined.
  The shell's division is the same in every cross-section: `shell` (a count or `Graded`), else cells about
  as long as the ring's spacing. ⚠️ **Not the layers' growth continued** (proposed and agreed, then
  measured worse): the layers' dual sizing alternates thick and thin, so the continued shell starts thin
  and its junction caps' apexes make 0.3 mm cells; one cell gave pipe faces over 50° 36 against 66 (log).
  ⚠️ The dual cells of the core's four corners in each cross-section are **triangular prisms** (three
  primal cells meet there): inherent to an O-grid; the bow evens their neighbours.
- **Ends** (`pipe_ends`, `fill_pipes`): an end on a flat face without layers bounded only by the
  pipe's rim (`_is_cap`: its edges ⊆ the pipe's) is **open** -- the O-grid runs to it and its last
  cross-section replaces that face's own faces inside the layers' ring, same patch (inlet / outlet area
  unchanged, 0.9783 of πr² both ways); else a **junction** -- the O-grid stops `junction_rows` short and
  its cross-section is the interface the fill meets, wound out of the fill and capped (`PyramidCaps`).
  4 rows short was worse than 2 (log). The open end's own interior points are left on no cell and
  dropped (`_compacted`), everything numbered after renumbered.
- Measured on Sozzi (maxh 0.02, layers 0.5 mm × 1.2 × 5, lamp and pipes mapped, five draws; log):
  pipes' faces over 50° 32-37 against 85-93 without cores, p99 6.0° against 46.8°, worst 59-60° against
  87-89°, neighbour volume ratio p99 36-38 against 165-177; ~1k more cells. ⚠️ The junction discs'
  cap apexes leave 2-3 cells of ~3.5e-11 m³ (0.33 mm) per mesh -- #785, now confined to two discs.
- Pinned in `tests/unit/test_meshing_pipes.py` (O-grid exact in a polygon prism, its outer faces the
  rings' quads, corners, bow, radial lines not twisted; `find_pipes`' refusals; the whole mesher on a stub
  cylinder open at both ends, no Netgen, every cell hexahedral but the corner prisms) and
  `test_meshing_netgen.py` (Sozzi's pipes: closure, pipe p99 < 7°, hexahedra along both axes, end areas).

## The block mesher — built as a library (`blocks.py`, `divisions.py`, `surfaces.py`, `lattices.py`, `block_assembly.py`, `block_topologies.py`)

Library code with unit tests (`tests/unit/test_meshing_blocks.py`, `test_meshing_lattices.py`); **not yet
reached from `CadMesher` or a mesh file**, no wall-layer window masking (pipes are found and O-gridded by
`pipes.py` instead, inside their layers; `PipeOGrid`'s core and bow are reused there). Its scope
follows blockmeshbuilder (BMB), the owner's blockMesh front end; the gap list and what was deferred are in
the log (2026-10-20). Dated numbers in the log.
- **`HexBlock(corners, divisions, edges=, faces=, projections=, patches=)`**: corners in hexahedron order
  (`HEXAHEDRON_CORNERS`, right-handed, refused otherwise per cell corner); a division per direction is a count, a
  `Graded(cells, growth, toward)` (on `graded_nodes`; `scaled` keeps the end-cell ratio) or node fractions;
  edges keyed by corner pair (either way round), a `Curve` callable (`Arc`, `SampledCurve` by arc length)
  or `(n+1, 3)` nodes; faces `u-..w+`, a `Surface(s, t)` or nodes, **a given face also gives its four
  edges** (a clash with a given edge is refused, naming both). Order: edges → each face (given, else 2D
  transfinite interpolation (TFI) of its edges) → its projection → 3D TFI of the six faces. **One TFI**
  (`transfinite`, any dimension, weighted by the node fractions) also draws the straight edges.
  `scaled_counts(factor)` (refused for nodes given as arrays) and `placed(Placement)`.
- **`placed` takes `io.cad.Placement`** (orthogonal, may reflect): curves and surfaces composed, projections
  conjugated. ⚠️ **A reflection makes the block left-handed**, so the placed block reads `w` the other way
  (corners `[4..7, 0..3]`, faces/patches `w±` swapped, `w` division reversed, faces along `w` re-parameterized)
  — names follow the geometry (a reflected pipe keeps `inlet` at its first height).
- **A projection moves only a face's nodes off its edges**; the edges must already lie on the surface.
  Edges of two projected faces are not intersected. Projections (`surfaces.py`): plane, cylinder, cone
  (nearest point on the infinite cone through two axial circles), sphere.
- ⚠️ **What a flat or ruled test cannot see.** TFI reproduces any separable field whatever its blend
  weights, so a box cannot pin the weights — the graded annular sector (exact polar grid) does. TFI of a
  cylinder or cone face between arcs and straight generators is already on the surface, so the projection
  is inert there; a twisted ring (helical edges) or a sphere pins it.
- **`BlockLattice(frame, coordinates, divisions)`**: blocks at the tensor product of three coordinate lists,
  one division per interval (a list or tuple per interval, anything else repeated — so `cell_counts`
  returns a list). Shaped by editing arrays in place, BMB's model: `corners` (frame coordinates), `divisions`,
  `mask`, and `edges` / `projections` / `patches` of shape `(n0+1, n1+1, n2+1, 3)` (entry `[i,j,k,a]` = the
  edge from node `(i,j,k)` along `a`, or the face normal to `a` at node index `i|j|k` along it).
  `None` = the frame's default; an edge `Straight()` or a `Curve`; a projection `False` = none.
  `CartesianFrame`: straight edges, no projections. `CylindricalFrame` `(r, θ, z)`: an edge is the image of
  the straight coordinate segment (arc, radial or axial line, helix); a face normal to `r` with one radius
  projects onto that cylinder, one whose radius differs only between its two heights onto that cone;
  edges turning ≥ 180° and radii ≤ 0 (an axis-collapsed block) are refused.
- **`BlockMesh.assemble(blocks, tolerance=None, count_factor=1.0)`** identifies points geometrically
  (`cKDTree`, unset tolerance 1e-6 of the shortest cell edge; two points of one block within it are
  refused). A block face is either met whole by another (every quadrilateral matched) or boundary; met in
  part — some quads matched, or none but an interior point shared — is refused. ⚠️ **Blind spot**: two
  faces overlapping without sharing one point both stay boundary (e.g. an `Annulus` at `azimuth=0` under a
  cap). `count_factor` scales every block's counts (BMB's `density_scale`). `patches(default_patch=None)`;
  `mesh(named, default_patch, patch_types)` → `element_mesh(..., patch_types=)`; `Mesh` refuses a type for
  a patch that does not exist.
- **Layouts** (about `z`, `heights` = ends plus cuts, a division per shell/segment or one for all):
  `PipeOGrid(radius, heights, tangential, radial, axial, shells=(), side_blocks=1, core_size=0.5,
  core_bow=0.0, azimuth=-π/4)` — a Cartesian core lattice plus a cylindrical tube lattice whose inner
  ring is moved onto the core's boundary (its radial edges `Straight`, its inner faces unprojected, its
  inner θ-edges the core's own edges); `core_bow` moves each side's middle that fraction of the way to the
  corners' circle (sides become arcs, nodes at equal angles); `azimuth ≠ −π/4` rotates every block by a
  `Placement`. Patches `wall`, `inlet`, `outlet`. `Annulus(radii, heights, tangential, radial, axial,
  around=4, azimuth=0)` — one cylindrical lattice; `inner`, `outer`, `start`, `end`. `CubedSphereCap`
  (cube top + four upper half-sides projected, equiangular; `inner`, `outer`, `rim`, rim corners at 45°,
  so `Annulus(azimuth=π/4)` joins it conformally). Tangential counts are counts, not fractions:
  neighbouring blocks run opposite ways along the core's sides.
- ⚠️ **The first shell's radial edges must be `Straight`**: with more than two side blocks the core's side
  nodes (even in length) and the shell's (even in angle) are at different angles, and the frame's edge
  between them is a curve. With ≤ 2 side blocks the angles coincide and a test cannot see it.
- **The dual of each layout is closed** and its volume converges at second order (pinned, bowed and
  multi-block included); the primal's error is exactly the inscribed polygon's area deficit.
- **Joining blocks to the rest of the primal is open**: Netgen refuses quadrilaterals, so a block region's
  free quads need pyramids before a tet fill (`element_mesh` takes pyramids).

## The octree hex core (`octree.py`, `hex_core.py`, `regions.py`) — built and wired in

`OctreeCore(size, wall_size=None, clearance=None).mesh(region, lower, upper, *, cap_height,
boundary_size) -> HexCore` (`points`, `hexahedra`, `pyramids`, `tetrahedra`, `caps`, `surface`, `octree`,
`kept`); `within(points, triangles, *, cap_height, boundary_size)` is `mesh` of the `ClosedSurface` of a
closed triangle surface over its box. Agreed with the project owner, 2026-10-20 (the design below):
**one value the case file reads** — `size` (root cubes), `wall_size` (unset: `boundary_size`, which
`CadMesher` passes as `max_size`; refused above `size`), `clearance` (unset: twice the wall size);
`refinements` and its own `cap_height` were removed — **one `cap_height`, `CadMesher`'s, serves the wall
caps and the core's**, passed to `mesh`. `resolved(boundary_size) -> (wall, clearance)` is the one home of
the defaults. Given as a share, a clearance at or under `cap_height × wall` is refused (a cap would cross
it); `ClearedCaps` holds each core cap within `gap` of the clearance instead.
`element_mesh` takes `hexahedra` and `pyramids` beside tetrahedra and prisms (cells numbered tets, prisms,
hexes, pyramids; earlier inputs unchanged).
- **The region is a `Region`** (`regions.py`): `inside(points)` and `meets(lower, upper)` — whether the
  boundary is taken to pass through each box. Two strategies:
  - **`ClosedSurface.of(points, triangles, spacing)`, exact**: the triangles in a
    `aquaflux.triangle_grid.TriangleGrid` of voxels `spacing` across (`within` uses
    `wall / VOXELS_PER_WALL_CUBE`, 4); `meets` is the grid's conservative `holds_any` (a box overlapping no
    occupied voxel meets no triangle; errs towards *met* by at most a voxel), and a box wholly outside the
    grid is not met (the grid's test clips it onto its edge voxels and would say met). `inside` labels the
    face-connected unoccupied voxels once (`scipy.ndimage.label`) and decides each stretch by **ray
    parity** at one voxel centre, three directions in different octants that must agree; a point in an
    occupied voxel gets its own rays. The surface is checked closed and wound outward first
    (`check_closed_surface`). **Why ray parity, not `radiation.enclosure_winding`** (project owner,
    2026-10-20): the winding number sits on radiation's solid-angle kernel tier (`solid_angle.py`,
    `clipping.py`, `work.py`, `checks.py`'s edge machinery), which cannot move with the grid; parity is
    exact on a surface checked closed, and only a handful of points are asked. The ray test is
    `regions.ray_hits` (Möller–Trumbore); `regions.first_hits` (a ray's nearest crossing, a KD-tree of
    face centroids within reach) builds on it for the layers' gaps and `ClearedCaps`' room.
  - **`SampledRegion(contains, samples=3)`**: an inside test (an analytic body, a `solids` body);
    `meets` = the box's sample lattice is mixed, so a boundary between samples is missed. With an odd
    count the cube's centre is a sample, so "kept" is exactly the former every-sample-inside rule.
- **`TriangleGrid` moved out of `radiation/` to the neutral leaves `aquaflux/triangle_grid.py` /
  `triangle_grid_walk.py`** (in `test_layering.NEUTRAL_LEAVES`; `MAX_VOXELS` and
  `near_cubic_resolution` made public), because the mesher may not import a physics package. Its rules
  stay in `radiation.md`, whose `paths:` now globs both files.
- **The tree** (`aquaflux.octree.Octree`, shared with the adaptive sampler and kept in that generic
  package so neither consumer imports the other; `.claude/rules/octree.md`): root cubes **of the target
  `size`** on a lattice from `lower` covering `upper` (refinement only goes finer); leaves in integer units
  of the finest cube, named by their lower corner (leaves never overlap), the unit halved when a refinement
  adds a level. Points are keyed by `Octree.keys` / `Octree.coordinates` (the hex core's half-unit points
  too). `refined_by(criteria)` splits until no `RefinementCriterion.refines(lower, size)` asks;
  `meshing/near_surface.py::NearSurface(region, size, margin=0)` splits cubes whose box grown by `margin`
  the region's boundary `meets`.
  `balanced()` ripples until leaves sharing a face **or part of an edge** are ≤ 1 level apart (vertex
  contact is not balanced): one finest cell across each of the 18 face and edge directions finds any
  leaf ≥ 2 levels coarser, because such a leaf covers the whole own-size region across.
- **Refined towards the boundary**: `NearSurface(region, wall, margin=clearance + wall)` halves every
  leaf whose box grown by the margin the boundary `meets`, down to the largest `size / 2**k` not over
  `wall_size`; balance grades outward. ⚠️ **The `+ wall` is load-bearing**: with a margin of the
  clearance alone, a coarse cube whose finer neighbour across is not kept faces the fill, cut about its
  centre (a transition cube's pyramids under wall-sized caps, so cap edges cannot show it). Pinned by
  `test_a_core_graded_towards_its_boundary_meets_it_with_wall_cubes_alone` (a wall off the lattice); on a
  ball the children of refined cubes always form the boundary and a test cannot see it. So **the core's
  boundary is wall cubes only, capped whole**: the fan-triangle path of `surface` is unreachable from
  `OctreeCore` and kept as the general conformity rule (pinned on a hand-built tree).
- **Kept**: a leaf the boundary `meets` nowhere within the cube grown by `clearance`, whose centre is
  `inside`.
- **Hanging points: resolved by templates, not by Netgen (chosen 2026-10-19).** A cube with a smaller
  cube's corner at any of its edge midpoints is a *transition cube*, cut into pieces meeting a new point
  at its centre; each face is one quad (no halved edge), four quads through the face centre (the cube
  across is split — balance makes its four face-touching children leaves), or a fan of triangles from a
  face-centre point through corners and halving points (edges halved by a cube sharing only the edge;
  the same-size cube across builds the same fan). Quad + centre = pyramid, triangle + centre = tet.
  **Every face is decided from the whole tree, not the kept set**, so any subset of leaves is conformal
  where its cubes meet; the dual turns these pyramids and tets into polyhedra, so the agreed "no tets or
  pyramids in the solver's mesh" holds. **Why not the alternative** (drop every cube with a hanging point
  and let the gap fill tet the holes; reasoned, not tried): a hole one cube thick gets caps from both
  sides whose apexes nearly meet, and every hole is another closed component for the fill; the templates
  are local, exact, and testable without Netgen.
- **Pinches**: cubes touching only along an edge or at a corner give a non-manifold boundary, which
  neither `fill` nor the dual accepts. `_without_pinches` repeats: find points on an edge of > 2
  boundary faces or with > 1 fan of faces (joined across edges shared by exactly two), and drop **at
  most one cube per pinched point per round**, the one with the largest share of its faces exposed,
  the finer of equals. ⚠️ **Dropping every cube at a pinch cascades** on a staircase ball (557 → 34 cubes;
  log), and so does dropping each pinch's minority side (557 → 149: two rules can pick opposite sides);
  the one-per-point rule gives 557 → 485.
- **Caps**: each boundary quad gets a pyramid outward (`caps.capped` on the core's boundary, the
  clearance free in front of it). ⚠️ **A fixed share stays < 1/2**: at a concave right-angled step two
  caps' sides lean together at 45° from their quads at 1/2 and coincide; `ClearedCaps`' fold limit is
  that rule per edge. `surface` (cap sides + transition cubes' boundary triangles) is wound
  **into the core** — out of the region left to fill — so beside an outward wall it is a hole for
  `NetgenProcess.fill`, which wants only the points the triangles use: renumber (as the Netgen test does).
- Pinned in `tests/unit/test_meshing_octree.py` (balance by brute force over leaf pairs, refinement to
  size on the sphere), `test_meshing_regions.py` (box less a hole: `inside` against the closed form on
  both paths, the hole a stretch of its own; `meets` never false where the surface passes, false two
  voxels clear; an open surface refused), `test_meshing_hex_core.py` (box = its uniform grid, sampled and
  as a surface with a clearance over a cube; the clearance guard; wall cubes alone at the boundary; refined
  core: volume, boundary area = the cube union's by voxels, no point at any cell edge's middle; capped
  surface closed, one fan per point, encloses −(cubes + caps); edge-touching boxes parted only on the
  shared edge; hex/pyramid `element_mesh`), `test_meshing_cad_mesher.py` (`_cored` without Netgen: the
  hole closed, the core's cells fill it, `fill_size`), `test_mesh_file.py` (the key, its refusals),
  and `test_meshing_netgen.py` (core + caps + fill in a box: dual closed, volume 1, interior points cubes
  with 6 quad faces; the Netgen L with layers and a core through `CadMesher`: dual closed and exact, > 50
  wall cubes, none nearer the wall than the clearance plus half the layers).
- **In `CadMesher`** (`_cored`, kept apart from the rest of `mesh` so other work there rebases): the
  core is built `within` the fill's own closed surface — the layers' top, the bare triangles and the
  wall caps' sides — so **the clearance is from the layers' top and the wall caps' apexes by
  construction**, not from the drawing. Its points follow the surface's, its capped surface joins the
  fill's as a hole, its tets follow the fill's, its hexahedra the layers' (a column's cells keep their
  numbering), its pyramids and caps the wall caps. **No size field**: Netgen's fill grades its tets from
  the triangles it is given (`validation/meshing_netgen/fill_grading.py`, log), so the fill's
  `max_size` is only raised to the wall cubes' size where larger (`_CoreCells.fill_size`). `MeshSpec.core`
  is the case key (`{kind: OctreeCore, size: ...}`); unset, the polyhedral core.
- **Interaction with #785**: each core cap's apex is a dual cell too; `ClearedCaps` sizes the core's
  caps and the walls' alike (small cells 1785-1832 → 257-322, with more faces over 50°: above). ⚠️
  **Moving the apexes after the fill does not help** (Laplacian with a shape
  guard; measured on five draws, dropped; log, "the caps' apexes moved after the fill"): the apex cell's
  size is set by the fill's own points near the cap, so a fix must change what the fill sees.
- ⚠️ **Measured, the core does not yet pay** (maxh 0.01, log "the octree core wired into `CadMesher`"):
  on an open vessel the agreed defaults give +15 % cells and 31x the faces over a volume ratio of 20
  against the polyhedral core -- 94 % of those faces have a core cap's apex cell as the smaller, and
  the transition cubes' polyhedra carry none (log, "Attribution"); wall cubes of `2 × maxh` save 19 % of the cells, still worse in volume ratio; Sozzi's 34 mm
  annulus keeps no cube at all. The defaults and #785's priority are with the project owner.
- **Not yet**: snapping nodes to walls (the core always stays a clearance from every patch, inlets
  included); user refinement regions; a core in a gap narrower than about `2 × clearance + wall` (left to
  the tets; Sozzi's annulus, log).

## Face planarization (`planarize.py`) — built, run by `CadMesher` when asked

`FacePlanarizer(min_decrease=1e-3, flat=1e-12, damping=1e-3, shortening=100, increase=4, decrease=3,
linear_tolerance=0.1, max_linear_products=200, max_iterations=200, max_attempts=60).planarize(mesh,
movable) -> Planarization(mesh, warp, converged, linear_products)`. Moves only the vertices `movable`
marks, chosen by the caller: **for a dual, the primal cell centroids, `TracedDual.vertex_cell >= 0`**
(boundary vertices, feature points and the cut's vertices between pieces must stay). **`CadMesher`
also holds the layer cells' centroids** (`_free_centroids`): the projection has just set them at their
heights from the true wall, which a move would undo (pinned by
`test_the_planarizer_in_the_mesher_leaves_the_layers_at_their_heights_from_the_wall`). It runs after
the projection, so a curved wall's boundary vertices are on the drawing when the bounds are taken.
**Off by default** (`planarizer=None`): turning it on changes every mesh's interior, and that default
is the project owner's to change. Cost on top of meshing: Sozzi water at maxh 0.03 with layers on the
lamp (17912 cells) 5.9 s without, 58.8 s with (one run, macOS arm64, 2026-10-20).
- **Objective**: total warp `E = |r|²`, `r` each incidence's height above its face's vertex-mean plane
  over √area (the `1/area` weight beat a uniform one on the L under the former solver: non-orthogonality
  max 39.9 vs 46.1 deg).
- **Solver: Levenberg–Marquardt on `r` written in JAX, planes included** (`_Warp`, an `eqx.Module`;
  `_gradient` / `_product` are `filter_jit`ted vjp and linearize-transpose), `(JᵀJ + λI) dx = −Jᵀr` by
  conjugate gradients to `linear_tolerance`, never forming `J`, **preconditioned by each vertex's 3×3
  frozen-plane block** `Σ n nᵀ/area + λI` (−38 % products on the L). λ starts at `damping · mean(1/area)`.
  ⚠️ **This is the one place in `meshing/` that uses JAX's AD and jit** — a deviation from the agreed
  "numpy and numba, not JAX" layering, made because the step needs the true Jacobian (project owner asked
  for the AD Gauss–Newton step, 2026-10-20). The Jacobi sweep it replaced held the planes still and
  converged linearly; it was dominated on every dual and deleted.
- **Stops on a convergence test**: an iteration lowers `E` by less than `min_decrease`, or `E` reaches
  `flat` of its start (where faces can be made exactly flat `E` falls geometrically and the first test
  never trips; pinned by the displaced hex block). `max_iterations` is a bound reported through `converged`.
- **The guard is load-bearing (binding — do not drop a bound).** Unguarded on the Netgen L the former sweep
  cut the warp 98 % but took non-orthogonality to 153 deg and skewness to 11 (vertices dragged to flatten
  sliver faces carried centroids past faces). A step is kept only where, against the **starting** mesh: **no
  cell less convex than the start's least convex cell** (`cell_non_convexity`) **nor more
  volume-ambiguous than its most** (`triangulation_volume_change`), none newly concave by
  checkMesh's test or more so (`cell_concavity`, **per cell**), volume positive, and **no face more
  non-orthogonal or skewed than the starting mesh's worst face**. ⚠️ **Bounds against each element's own
  value trap the pass** (project owner agreed the start's-worst form, 2026-10-20): per-cell convexity
  against the *previous iteration* stalled at 64 % on a Kuhn dual; per-face non-orthogonality against each
  face's own start allowed 0 iterations on the L; per-cell convexity against each cell's own start held
  most of the worst all-free faces (`worst_face_limits.py`), and replacing it with the start's worst gave
  16–22 % (L, Sozzi) to 62 % (cylinder) less warp, fewer concave cells and 1.1–3.2x faster passes on the
  four Netgen duals (log). Concavity stays per cell: a cell turning concave is a failure however mild (off,
  concave cells more than double). **Triangulation-volume dependence is bounded by the start's worst
  too**: without it, under the wider convexity bound, Sozzi's worst rose above its start (0.299 vs 0.231,
  a wall cell); with it 0.228, nothing else changed (log). The mesh is refused
  if a starting cell's volume is not positive or its quality undefined: a NaN would be the bound. The
  volume bound never bites on the fixtures but stays: an inverted cell's convexity is NaN, which passes a
  `>` bound.
- ⚠️ **How a violating step is repaired (binding)**: shorten first (λ ×`increase`, up to `shortening` × its
  start), then **pin the worsened cells' free vertices and re-solve** (a bad face pins both its cells).
  Deleting held vertices from a finished global step instead stalls (kept share 0.64 → 0.24, log); with no
  shortening the face bounds are broken on the fixtures, with no pinning the cell bounds. The bounds, not
  the solver, cap each step's length.
- **Measured (log 2026-10-20)**: on the Netgen L / Sozzi / cylinder / sleeve duals, 0.45–0.9× the former
  sweep's time (90–714 s, 3.7k–16k cells, shared machine), 1.8–10× less total warp, half or fewer of its
  concave cells (L 57→28, Sozzi 309→58, cylinder 1108→66, sleeve 258→30), every p99 lower;
  maxima of warp, non-convexity and triangulation-volume dependence nearly unchanged (within bounds), and
  **boundary faces untouched** (vertices fixed). Those timings predate the cheaper guard (next item).
- **The guard evaluates through one `MeshQuality` per pass** (`mesh.md`): the topology's tables and ring
  signs are found once, and each trial is one `geometry()` plus the four checks. Netgen L 0.01 (shipped
  defaults, macOS arm64, load ~12 from other sessions, 2026-10-20): evaluations 50.3 → 3.1 s of a pass
  (0.229 → 0.014 s each, 220 of them), the pass 132 → 101 s; iterations, products and every quality
  number identical, the warp history equal to 1e-16. **The conjugate-gradient products are now ~87 % of
  a pass** (5503 at ~16 ms): the next lever. Recomputing only the cells a step touches is not worth
  building: a step moves 85–100 % of the cells (Kuhn n = 6 dual, every trial) — log.
- **Where the worst remain (log 2026-10-20)**: most of the worst 1 % interior warp is on faces whose every
  vertex was free (48–68 % on the four Netgen duals, the worst face on Sozzi and the cylinder), so fixed
  vertices are not what holds them: the per-cell convexity bound then in force did (log, 2026-10-20,
  `worst_face_limits.py`: unbounded they flatten, a tighter stop barely helps), which is why the bound is
  now the start's worst; wall cells hold most of the worst triangulation-volume dependence. **Open**:
  sliding boundary vertices along the CAD surface (for the wall cells, on curved walls); **the bounds as penalties in the objective** (a cost rising steeply
  near each limit, so the solver trades quality against flatness rather than stopping at a wall — the
  project owner's preferred longer-term direction, 2026-10-20); cheaper conjugate-gradient products. Pinned in
  `tests/unit/test_meshing_planarize.py` (derivatives against finite differences of the guard's warp,
  the preconditioner against explicit frozen-plane blocks); harness `validation/meshing_dual/planarization.py`.

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
  - **`edge_divisions={edge: segments}`**: those edges divided exactly, at equal arc length, through
    netgen.occ's `Edge.partition` (`_divide_edges`). ⚠️ Three traps, each silent: the partition is the
    edge's **interior** points' parameters **scaled to [0, 1]** over its parameter range (raw parameters
    break the surface mesh); it is read only when an `OCCGeometry` is **made of the shape**, so the child
    makes a second one after setting it; and `shape.edges` lists an edge once per face it bounds, so an
    edge's number is its place among the **distinct** edges in first-met order (dedup by `hash` + `==`),
    which is `CadModel.topology`'s `MapShapes` numbering (all 15 Sozzi edges matched by length and end
    vertices, 2026-10-21). Keep the `shape.edges` list alive while its edges are used (a temporary
    segfaulted). Equal arc length matters on the riser's saddle, whose parameter is not arc length (spacing
    ratio 1.0002 against 1.004 in equal parameter steps).
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
  hex core's and the layers' quadrilaterals are capped with pyramids we build (`PyramidCaps`, shared),
  whose triangular sides Netgen sees.
- **Netgen's own prism layers are not used** (tet/prism only, and little documented); the extruder is ours.
- **Cost** (2026-10-19, Sozzi fluid, macOS arm64, netgen-mesher 6.2.2608): about 25k tets / 5.7k points per
  second, and `SetNumThreads(4)` changed nothing. A dual cell is one primal point, so a 2M-cell all-tet
  core would need on the order of 6 minutes of Netgen; the layers and a hex core are not Netgen's.

## Package and layering (agreed)

- **`aquaflux/meshing/`, host-side numpy and numba, not JAX** (exception: `planarize.py`'s AD step; see
  "Face planarization"). It depends on `aquaflux.mesh` (it ends at
  `Mesh.from_csr`, which owns validation) and on `aquaflux.io.cad`; nothing imports it back except a case
  mesh file (`MeshSpec`, `aquaflux/case/mesh_file.py`) and the test fixtures.
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
3. **Prism-layer extruder** (built, above): smoothed normals, thickness limited by proximity, halving and
   tapering on collision; prisms on triangles, hexahedra on mapped quadrilateral walls. Not yet: several
   normals at convex ridges.
4. **Block mesher** in place of blockMesh: transfinite interpolation with projection onto CAD or analytic
   surfaces, carrying the `uvmesh` topologies (four-quadrant annulus with the π/4 offset when a cap joins,
   the five-block cubed-sphere shell, the square-core pipe O-grid — **these built**, with BMB-style block
   lattices, masking, placement, plane/cone projection and patch types; wall-layer window masking as a
   preset not); pipes found from cylindrical faces and their cores O-gridded inside their layers
   (`pipes.py`, built above).
5. **Octree core**, 2:1 balanced across faces and edges, hanging nodes resolved by pyramid/tet templates
   in the primal that the dual turns polyhedral (built and selectable in `CadMesher` / `MeshSpec.core`).
6. **Quality metrics** (non-orthogonality, skewness, convexity; built in `aquaflux/mesh/quality.py`,
   `mesh.md`) and planarization (built, above; run by `CadMesher` when given a `planarizer`).

**Phasing**: (1) CAD topology → surface triangulation → polygonal wall mesh → polyhedral core through the
dual; (2) prism layers; (3) block mesher, presets, pipe O-grids, baffles; (4) octree hex core.

**Not decided**: radial basis function (RBF) morphing *beside* transfinite interpolation — for spreading a
projection's displacement through a block (the pipe-to-chamber saddle), snapping octree nodes, and later
as the differentiable map from CAD parameters to `node_coords`. **Deferred until the mesher is finished**
(project owner, 2026-10-21): the design-parameter applications come after it; the ranked applications
and costs are in the log, "RBF morphing: where it would fit".

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
