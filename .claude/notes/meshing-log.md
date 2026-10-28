# Meshing log — the dated record behind `.claude/rules/meshing.md`

## 2026-10-19 — research survey and the agreed design

**What exists.** No permissively licensed, pip-installable library builds polyhedral prism layers, a
poly-hexcore transition, a conformal layer–core join, a feature-preserving 3D dual of a tet mesh, or
automatic pipe O-grids. Licences checked on PyPI or the project page: gmsh 4.15.2 GPL-2.0+; netgen-mesher
6.2.2608 LGPL-2.1 (wheels macOS universal2, Linux x86-64, CPython 3.10–3.14); pytetwild MPL-2.0; TetGen
AGPL-3; CGAL Mesh_3 / polygon-mesh processing GPL-3; pymeshlab GPL; libigl core MPL-2.0 (`igl.copyleft`
GPL); geogram BSD-3 (no PyPI wheel found). aquaflux is MIT.

**The precedent, of-optical-radiation `tools/uvMesh` (uvmesh 0.12.0, commit `276d992`, as used for the
Sozzi mesh of the 2026 IUVA talk).** blockMesh (through `blockmeshbuilder`) builds the lamp annulus with a
cubed-sphere cap, the pipe O-grids and the chamber wall layer; gmsh with Netgen optimization builds the tet
bulk, which `polyDualMesh` turns polyhedral; `mergeMeshes` and `createNonConformalCouples` join every
piece to the bulk non-conformally. Sozzi: 1,232,629 cells, max non-orthogonality 52.8°, about 2 minutes.
Its docstrings record the seam failures listed under "Traps" in the rule.

**Commercial methods, as design targets only.** Fluent Mosaic / Poly-Hexcore: octree hexes, poly-prism
layers, general polyhedra joining them conformally (white paper; construction of the polyhedra not
published). STAR-CCM+: dual-of-tet polyhedral core, prism layers extruded from the polygonal surface.
Pointwise T-Rex: advancing anisotropic tets recombined into prisms and hexes.

**Decisions by the project owner (2026-10-19)**: Netgen yes; primal-then-dual yes; pipes and baffles
detected automatically with an override (baffles as faces with fluid on both sides); the kernel may read
edge topology; target 2M cells on a laptop; `Mesh` in memory plus a polyMesh writer first; a new package.

**usd-brep** (NVIDIA-Omniverse/usd-brep, 1.0.0, 2026-10-08), read at that release: Apache-2.0 NURBS kernel
with closest point, surface evaluation, tessellation, booleans and healing; imports OpenUSD, OCCT `.brep`
and PRC; no STEP importer found; PyPI wheels CPython 3.12 for Linux x86-64/aarch64 and Windows only; its
`KNOWN_ISSUES.md` lists healing errors on cylinders, spheres and tori.

## 2026-10-19 — OCP and Netgen in one process

Configuration: macOS arm64 (Darwin 25.3.0), Homebrew CPython 3.13.4, `cadquery-ocp-novtk` 8.0.1.0.0 (OCCT
8.0.1), `netgen-mesher` 6.2.2608 with `netgen-occt` 7.8.1, numpy 2.4.4. Harness:
`validation/meshing_netgen/coexistence.py`, each arm a fresh child three times:

| arm | exit status |
|---|---|
| OCP alone | 0 0 0 |
| Netgen alone | 0 0 0 |
| OCP, then `netgen.occ` | SIGABRT x3 |
| OCP, then `netgen.meshing` | SIGABRT x3 |
| `netgen.occ`, then OCP | 0 0 0 |

The abort comes after all work completes, at exit. Crash report stack: `__cxa_finalize_ranges` →
`libTKDE.7.8.1` → `TCollection_AsciiString::~TCollection_AsciiString()` in **`libTKernel.8.0.1`** →
`___BUG_IN_CLIENT_OF_LIBMALLOC_POINTER_BEING_FREED_WAS_NOT_ALLOCATED`. Reading (not proven): the loader
binds Netgen's references to the inline destructor to the first-loaded OCCT copy, which frees with the
other copy's allocator. With Netgen first, all CAD work also completed, which says only that this probe hit
no mismatch. Linux not measured (Docker was not running).

## 2026-10-19 — what Netgen does when reached through files

Same configuration. `validation/meshing_netgen/write_fluid.py` (OCP only) builds the Sozzi fluid as
reactor body ∪ inlet pipe ∪ outlet pipe − lamp and writes BREP and STEP; `netgen_cad.py` and
`netgen_fill.py` run without OCP.

- Netgen reads both files as 1 solid, 9 faces. `OCCGeometry` of the original drawing reads 1 solid of 3
  faces (the lamp), where `read_step` returns four solids.
- Surface only at `maxh=0.01` m: 18742 triangles, every one naming one of the 9 faces, 1.0 s.
- Volume, one run each, wall clock on an otherwise idle laptop (no gate or case running):

| maxh (m) | tets | points | s |
|---|---|---|---|
| 0.01 | 87,537 | 19,214 | 3.4–3.6 |
| 0.006 | 132,626 | 29,875 | 5.6–5.7 |
| 0.004 | 249,302 | 57,252 | 9.6–10.1 |

  `SetNumThreads(1)` and `(4)` gave 3.4 and 3.5 s at 0.01. Tet count grows slower than `maxh⁻³` because
  Netgen's default grading already refines near curvature.
- Supplied-surface fill (unit box, 4 x 4 per face, `maxh=0.3`): all-triangle, wound outward → 267 tets,
  boundary unchanged, 0 supplied points moved; with a box-shaped hole → 1329 tets, likewise unchanged;
  wound inward → refused; top face as quadrilaterals → refused ("Stop meshing since too many attempts").

## 2026-10-19 — Netgen conventions found while building the child process

netgen-mesher 6.2.2608, macOS arm64, BREP files written by OCP 8.0.1 (box `[0,1]x[0,2]x[0,3]`, cylinder
r 0.5 h 2). Face descriptor index `k` (one-based) is OCP's explorer face `k-1` on both (box faces x=0, x=1,
y=0, y=2, z=0, z=3; cylinder side, top, bottom). Surface triangles all outward (fraction 1.0 per face).
Volume tets all negatively oriented (fraction 0.0 positive; sum −6.0 on the box). `Elements2D().NumPy()` /
`Elements3D().NumPy()` nodes are one-based and equal the per-element loop; `Coordinates()` equals
`Points()`. Bulk `AddPoints` + `AddElements(dim=2, base=0)` fill of a surface-only point set keeps the
leading points and the surface; one extra unused point made the leading points differ (renumbered). A hole
wound out of the hole: 530 tets, volume 1.0 (the hole's 0.027 filled), 192 boundary faces against 240
supplied, 53 tet centroids inside the hole. Two overlapping outward cubes: filled, both checks passed.

## 2026-10-19 — the polyhedral dual: closure, concave folds, scale, checkMesh

`PolyhedralDual(feature_angle=45 degrees)`, macOS arm64, netgen-mesher 6.2.2608 tetrahedra
(`validation/meshing_dual/dual_scale_convexity.py`; primal built from Netgen's tets by the harness, one
patch `walls`).

**Concave folds.** Netgen L (two boxes 0.2x0.1x0.1 + 0.1x0.1x0.1) at maxh 0.02: 3 of 581 dual cells
unclosed (residual 1.3e-4, 1.3e-4, 3.9e-4), all on the concave edge (x = z = 0.1): its two ends and one
interior point. Each is topologically closed (every cell edge on exactly two of its faces) and its faces,
oriented consistently by shared edges, sum to <= 5.4e-20; `Mesh.geometry()` orients exactly one face per
cell inward (1 of 9, 1 of 11, 1 of 18). Non-convexity at the edge (largest vertex excursion beyond a face
plane / cube root of volume) 1.02 at 0.02, 1.01 at 0.01. The structured hex L (2x1x2 less a corner)
closes. Box 0.1x0.1x0.2 with a cylindrical hole r 0.01: sharpest concave fold 32.2 / 29.9 / 18.6 degrees at
maxh 0.01 / 0.005 / 0.003, 0 unclosed cells at each (485 / 572 / 1620 points on the hole); its hole-wall
non-convexity 0.73 / 0.28 at 0.01 / 0.005.

**Volume.** Cylinder r 0.05 h 0.3, dual volume below the tets' by 2.2e-2 / 5.7e-3 / 1.4e-3 / 3.5e-4 /
1.3e-4 at maxh 0.02 / 0.01 / 0.005 / 0.0025 / 0.0015 (the boundary through face centroids; second order).

**Scale** (cylinder, one run each, otherwise idle machine): 2028 / 14426 / 111283 / 542377 points ->
build 0.3 / 0.5 / 2.4 / 13.9 s (first call includes numba compilation, cached after). Peak footprint by
stage at 111283 points (607k tets), one process per stage: Netgen arrays loaded 0.28 GiB, + primal `Mesh`
0.85, + dual 1.88, + dual geometry 1.89. The whole 4-size run peaked at 12.3 GiB (542k points, 3.07M
tets). numpy peaks inside `build` at 111k points (`tracemalloc`): tables 292 MiB, features 5 MiB, assembly
474 MiB (of which the dual `Mesh.from_csr` 110 MiB); the rest of the footprint is JAX arrays. Linear
extrapolation to 2M points: ~10 GB for the primal as a `Mesh`, ~18 GB for the dual's build.

**checkMesh** (OpenFOAM 13 in Docker, `-allGeometry -allTopology`, duals written by `write_openfoam`):
structured hex box 4x3x5 -> Mesh OK, 6 faces per cell, non-orthogonality 1e-6. Kuhn tets of the unit cube
(n 4, perturb 0.2) with one patch -> Mesh OK, non-orthogonality max 42.6 avg 25.1, skewness 1.12. Netgen
cylinder maxh 0.005 (14426 cells) -> failed 3: 7 faces with low-quality decomposition tets, 1108 concave
cells (by face planes), 6 faces with volume ratio < 0.01; openness 2e-16, non-orthogonality max 50.1 avg
13.9, face flatness min 0.79. Netgen sleeve box maxh 0.005 (15995 cells) -> failed 2: 258 concave cells,
1 small volume ratio; non-orthogonality max 51.7 avg 12.2.


## 2026-10-19 — the concave cut: closure, placement, scale, checkMesh

`PolyhedralDual(feature_angle=45 degrees)`, macOS arm64, netgen-mesher 6.2.2608, harness
`validation/meshing_dual/dual_scale_convexity.py` (one patch `walls`); non-convexity as defined there.

**Placement is what closes the Netgen L.** Netgen L at maxh 0.01 (3707 points, 11 on the concave edge):
cut with the added vertices at face centroids and edge midpoints, 6 cells unclosed (3 points, both pieces
each, residual 3.7e-5 to 4.8e-5); topology closed (every cell edge used twice), one separating quad per
point inward by centroid (both piece centroids on one side of a zig-zag quad). With vertices on the
bisecting plane: 0 unclosed at 0.02 / 0.01 / 0.006 (6 / 11 / 18 extra pieces). Same L at 0.02, uncut, was
3 unclosed (entry above).

**Harness after the cut** (closure max, worst non-convexity overall | over extra pieces): L 0.02 / 0.01 /
0.006: 2.7e-19 / 9.8e-20 / 5.8e-20, 0.26 / 0.35 / 0.35 | 0.17 / 0.16; volume error 0 to 2e-16 (flat walls).
Sozzi water (`read_step` of `SozziTaghipour.step` with the test's placement, region reactor_body +
pipes less lamp, `CadModel.write_brep`) at 0.02 / 0.01: 10422 / 13273 points, 24 extra pieces each (the
two junction circles), closure 2.0e-19 / 1.1e-19, non-convexity 0.33 / 0.33 | 0.32 / 0.26, volume 2.5e-2 /
7.8e-3 under the tets' (curved walls through face centroids). Cylinder and sleeve unchanged (no concave
fold beyond 45 degrees).

**Structured fixtures** (`structured_grid_3d` / Kuhn `tetrahedral_grid_3d` less a corner): hex Ls exactly
convex pieces; a three-wall inside corner of hexes 0.097 with each interior hub at the mean of its cut
vertices, 0.026 with the least-squares hub (spacings 0.5 / 0.4 / 0.6); with cubic cells a centroid lies on
the bisecting plane (crossing at 0, clipped to 0.1). Kuhn Ls 0.083, the same as an uncut Kuhn cube's warp.

**Cost** (cylinder maxh 0.0025, 111283 points, no concave fold, so the cut does not run; one run each):
build 3.1 s against 2.7 s before the restructure, peak footprint 1.90 GB against 2.04 GB — not
distinguishable from run-to-run spread.

**checkMesh** (OpenFOAM 13 in Docker, `-allGeometry -allTopology`): Netgen L maxh 0.01 dual (3718 cells)
-> failed 1: 57 concave cells by face planes; openness 1.9e-16, non-orthogonality max 46.2 avg 12.0,
skewness 1.44, 2 faces with concave angles (max 31 degrees). Sozzi 0.01 dual (13297 cells) -> failed 3:
309 concave cells, 2 faces with low-quality decomposition tets, 1 small volume ratio; openness 2.2e-16,
non-orthogonality max 47.8 avg 14.2, skewness 1.96. Topology passes on both.

## 2026-10-19 — phase 1 end to end: STEP → Netgen → dual → polyMesh

macOS arm64, netgen-mesher 6.2.2608, Sozzi water (`SozziTaghipour.step`, placement swapping x and y,
reactor_body + inlet_pipe + outlet_pipe less lamp), feature angle 45 degrees, patches inlet = `FacesNear`
(0.04765, 0, 0.8945), outlet = `FacesNear` (1.739, 0, 0), lamp = `FacesOn(lamp)`, the rest `walls`.

- `CadMesher(max_size=0.02)`: 3.8 s, 10446 cells, closure 1.5e-19; patch areas against the CAD's
  (faces 6 / 8 / 4+7 / the rest): inlet 2.735e-4 vs 2.865e-4 (-4.5 %), outlet the same, lamp 0.04857 vs
  0.05090 (-4.6 %), walls 0.3528 vs 0.3622 (-2.6 %): the dual's boundary through triangle centroids,
  inside the curved walls. Not projected.
- `aquaflux mesh OUT --case case.yaml --openfoam` at maxh 0.01: 13297 cells (inlet 19, outlet 19,
  lamp 1747, walls 7791 faces), 8.65 s wall, 0.77 GB peak footprint, one run. checkMesh of this
  four-patch polyMesh not run (Docker hung); the same mesh with one patch passed topology (entry above).
- `tetrahedral_mesh` reproduces `tetrahedral_grid_3d`'s former hand assembly bit for bit (offsets,
  rings, owner, neighbour; n 2 and 3, perturbed).
- Mutations, each red: owner from the second sighting; a patch triangle off the boundary accepted;
  the cover check off; CAD features not passed to the dual; every between-face triangle edge named;
  `FacesOn` tolerance x1e12; the overlap check off; the name check off; `CadMesh.mesher` dropping
  `grading`; `--openfoam` writing nothing; `--case` taking paths from the working directory (green until
  the test ran from another directory: from the repository root, the relative path's `..` climbed to `/`
  and found the file anyway).

## 2026-10-19 — the dual's boundary projected onto the drawing

Sozzi water as in the entry above (same patches, feature angle 45 degrees), `CadMesher.mesh`, macOS arm64,
one run each. Patch areas against the CAD's, before → after projection:

| maxh | inlet | outlet | lamp | walls | time | boundary vertices off the drawing |
|---|---|---|---|---|---|---|
| 0.02 | −4.5 % → −1.14 % | −4.5 % → −1.15 % | −4.6 % → −1.54 % | −2.6 % → −0.86 % | 3.8 → 5.6 s | 4.4e-17 |
| 0.01 (after only) | −1.14 % | −1.15 % | −1.54 % | −0.47 % | 6.8 s | 4.4e-17 |

Closure ≤ 2.0e-19 and every cell volume positive at both. Mesh volume 5.4935e-3 at 0.02
unprojected; 5.6633e-3 / 5.7329e-3 at 0.02 / 0.01 projected. Non-convexity at 0.02 (harness measure): unprojected p50 0.082 / p99 0.182 / max
0.333; projected 0.082 / 0.176 / 0.326. The flat inlet and outlet caps stay short by their rims' chords
(the rim's points are on the circle, the polygon between them is not). No vertex on Sozzi is an edge
vertex inside one face (89 / 91 edge vertices at maxh 0.03 / 0.02, all between two faces), so that branch
is pinned on a Kuhn L instead.

Two defects found on the way, both now pinned: sorted-triple keys built in Netgen's int32 overflowed at
7394 points and put 3,000 lamp triangles on the vessel's face (areas +150 % to +3300 %, volume 50x); and
chaining the candidate edges of a rim drawn as two edges carried one vertex 7 mm to their shared end (the
one cell at non-convexity 1.48, inlet area −4.1 %). Mutations, each red: no projection; candidates
chained; int32 keys; no edge projection; in-face edge vertices left (red only on the Kuhn L test); the
dual reporting no edge origins; the kernel ignoring `edges`.

## 2026-10-19 — prism layers on triangulated walls

macOS arm64, netgen-mesher 6.2.2608, feature angle 45 degrees. `PrismLayers(first_height=0.0005,
growth=1.2, count=5)` on the Sozzi water's walls and lamp (caps bare), patches as in the phase-1 entry.

- Netgen unit cube, all layered, `first_height 0.005, growth 1.2`: volume exact, closure 7e-18, dual layer
  boundaries at mid-face 0.005 / 0.011 / 0.0182 / 0.02684 / 0.03721 (targets the same). Cylinder with bare
  caps: all 190 prism sides on the cap planes, closed.
- Sozzi maxh 0.02: 46124 cells (10446 bare), 8.4–9.1 s. Under the centroid orientation 59 wall cells
  unclosed (0.24–0.37 mm thick on radius-9.5 mm walls), closed under orientation by shared edges (#777).
  Projection of the wall alone then broke 39 cells (one of negative volume): the face centroid moves up to
  the chord's sag (~0.4 mm on the lamp) past a first dual cell 0.5 mm tall. Moving the prism centroids above
  it by `1 − (c/t)²` of the displacement: 0 unclosed, min cell volume 6.3e-11. Dual layer boundaries from the
  CAD, x in (0.1, 0.7), median [min, max] in mm — lamp: 0.507 [0.506, 0.519], 1.088 [1.085, 1.116], 1.791
  [1.779, 1.846]; vessel: 0.508 [0.505, 0.516], 1.143 [1.133, 1.175], 1.931 [1.904, 2.014]; targets 0.500,
  1.100, 1.820.
- Sozzi maxh 0.01: 59395 cells (13297 bare). Images: inside the 19 mm pipes the core left after the layers
  is narrow and its polyhedra skewed.
- Thinning: a diffusive smoothing (10 sweeps per halving pass) left a 0.05-thick slab's middle at 0.006
  where the gap limit allows 0.02; replaced by a ratio limit (1.5 per neighbour, to a fixed point).
- Mutations, each red: plain geometric tops; no stretch; no sliding (only on a tilted bare top — on a cube
  the inward normals already lie in the top's plane); no gap limit (red once halving could not rescue it);
  no halving; the prism inversion check off (only on an L's concave edge); no tapering; the column move off
  (the Netgen test).

## 2026-10-19 — the octree hex core: pinches, the Netgen fill, mutations

macOS arm64, one run each, `OctreeCore` defaults (`cap_height` 0.25, `samples` 3), netgen-mesher 6.2.2608.

- **Pinches on a staircase ball** (region: ball r 0.9 at (1, 1, 1) in the box [0, 2]^3; `size` s,
  `clearance` 0.05, `NearSurface(same ball, s/2)`), candidate cubes → kept:

  | s | candidates | drop every cube at a pinched point | drop the minority fan's cubes, and all but the first face's cube on a crowded edge | one cube per pinched point per round (shipped) |
  |---|---|---|---|---|
  | 0.2 | 352 | 128 | 176 | 256 |
  | 0.15 | 557 | 34 | 149 | 485 |
  | 0.1 | 1791 | 1756 | 1763 | 1784 |

  The middle rule still cascades because the two rules can pick opposite sides of one pinch, dropping
  both cubes.
  Two unit boxes of 0.25 cubes touching along an edge: 5 of 128 dropped, all on the shared edge; two
  touching at a corner: 1 dropped. Probe: `OctreeCore(s, 0.05, refinements=(NearSurface(ball, s/2),))
  .mesh(ball, [0]*3, [2]*3)`, counting `kept` (the two rejected columns need `_without_pinches` swapped for them).
- **Netgen fill and dual** (the pinned test's case: box [0, 1]^3 walls at 8 x 8 squares per face; core
  `size` 0.1, `clearance` 0.15, `NearSurface(ball r 0.15 at the centre, 0.05)`): 608 cubes (160 of 0.1,
  448 of 0.05), fill 0.45 s / 3713 tets at `max_size` 0.1, primal 7241 cells of volume 1.0, dual 2037
  cells in 0.38 s, closure 7e-18, volume 1.0. A staircase core (region a ball r 0.42 at the centre,
  `clearance` 0.05) in the same box: 472 cubes, 7006 tets, dual closes with volume 1.0 — caps on a
  staircase are accepted by Netgen.
- **Mutations, each red**: balance off; balance across faces only; `NearSurface` never refining; no
  transition cubes; split faces as fans; fans ignoring halved edges; no pinch removal; every cube at a
  pinch dropped; apex inward; `cap_height` ignored; boundary triangles or cap sides wound outward; no
  clearance; a wrong hexahedron or pyramid face; hexahedra and pyramids numbered in the other order; every
  cube a transition cube (red only in the Netgen test: its deep dual cells stop being hexahedra); caps
  wound outward (Netgen test). **Dismissed, equivalent**: dropping `Octree.leaf_at`'s level check —
  levels are visited coarsest first, so a wrong match at a coarser level is always overwritten by the
  true leaf's; kept so the lookup does not depend on that order.

## 2026-10-19 — the block mesher's core: TFI blocks, assembly, round layouts

Built `blocks.py` (`HexBlock`, `transfinite`, `Arc`, `SampledCurve`, `CylinderSurface`, `SphereSurface`),
`block_assembly.py` (`BlockMesh`), `block_topologies.py` (`PipeOGrid`, `Annulus`, `CubedSphereCap`), and
`element_mesh` hexahedra and pyramids. Volume errors relative to the exact solid, `PolyhedralDual` at
feature angle 45°, macOS arm64, one run each (whole assemble + mesh + dual ≤ 0.6 s each):

| layout (counts: tangential n, radial, axial/polar) | n = 4 primal / dual | n = 8 | n = 16 |
|---|---|---|---|
| `PipeOGrid(0.3, 1.2, n, n/2, n/2)` | −2.55e-2 / −5.79e-2 | −6.41e-3 / −1.54e-2 | −1.61e-3 / −3.93e-3 |
| `Annulus(0.2, 0.5, 0.8, n, n/2, 2)` | −2.55e-2 / −5.79e-2 | −6.41e-3 / −1.47e-2 | −1.61e-3 / −3.70e-3 |
| `CubedSphereCap(0.2, 0.5, n, n/2, 2)` | −6.55e-2 / −1.45e-1 | −1.69e-2 / −4.02e-2 | −4.25e-3 / −1.04e-2 |

Second order throughout. The pipe's and annulus's primal errors are exactly the inscribed 4n-gon's
deficit, `1 − sin(2π/4n)/(2π/4n)` (2.550e-2 at the 16-gon), an independent check. The dual sits inside the
primal by the boundary-centroid chord, about 2.3x the primal's error. Every dual closes (≤ 1e-16) with
positive volumes; the cap joined to an `Annulus(azimuth=π/4)` shares the rim's `4n(radial+1)` points.

Mutation pass (each red): blend weights uniform instead of the fractions (only the graded annular sector
sees it — survived the box tests, so that test was added); straight edges at uniform fractions or wrong
corners; an arc turned the other way; a reversed edge not re-parameterized; the polyline by index not
length (survived until the legs were made unequal); projection skipped; the handedness check off; an edge
not checked against its corners; a given face ignored; the face/edge clash check off; a hexahedron or
pyramid face ring wrong; no point identified; the interior-point rule of the partial-join check off; the
unnamed-patch check off; the pipe wall off its radius; pipe ends' patches swapped; an annulus arc dropped;
the cap's outer projection dropped. Two survivors were themselves defects found: the corner placement
loop in `HexBlock.points` was dead (every corner is set by its edges; deleted), and a 1e-30 tolerance still
merged a box's bit-identical points, so the merge mutation is on the grouping itself.

## 2026-10-20 — the block mesher against blockmeshbuilder: lattices, placement, cone, patch types

The owner's blockMesh front end is blockmeshbuilder (BMB, github.com/NauticalMile64/blockmeshbuilder, read
in full: structures, grading, geometry, boundary tags, every example). Gaps against our core, and the
owner's decision (2026-10-20): **built now** — block arrays (`BlockLattice`, Cartesian and cylindrical)
with masking, several shells/segments and grading shared per interval; the bowed and turned O-grid core
(`core_bow`, `azimuth`); placement (`HexBlock.placed` on `io.cad.Placement`); plane and cone projection;
patch types, a default patch, and a global count factor (BMB's `density_scale`) and `cell_counts` (its
`number_of_divisions`). **Deferred** — per-edge grading within a block (BMB's edgeGrading), spline and
B-spline edges (a callable covers them), cell zones per block (the dual takes one zone), periodic patches
(cyclic/AMI; the dual refuses periodic), the axisymmetric wedge with a collapsed axis. **Never** — the
airfoil boundary-layer wrap (prism layers cover arbitrary walls) and `mergePatchPairs` (no non-conformal
coupling). BMB's `thin_baffles` example builds only a graded cavity, so it shows nothing on baffles.

Measured (relative volume error against the exact solid, `PolyhedralDual` at 45°, macOS arm64):
`PipeOGrid(1, [0, 1, 2], n, n/2, n/2)` n = 4/8/16: primal −2.550e-2 / −6.413e-3 / −1.606e-3, dual
−6.02e-2 / −1.57e-2 / −3.97e-3; the bowed pipe (`core_bow` 0.6) gives the same primal error (the core
does not touch the wall polygon). Frustum shell (cylindrical lattice, radii 0.5→0.4 and 1.0→0.7 over
height 1, n/2 × n × n): primal −2.550e-2 / −6.413e-3 / −1.606e-3, dual −6.02e-2 / −1.57e-2 / −3.97e-3.
Every dual closed (≤ 1.7e-16). Refactoring `PipeOGrid` and `Annulus` onto lattices left the existing
block tests' values unchanged.

Mutation pass (`test_meshing_lattices.py`, each red): `Graded` toward the end not mirrored; scaled growth
not re-exponented; explicit fractions split wrongly, or a non-whole factor accepted; `cell_counts` floor 0;
the plane's step halved; the cone's foot without its radial part; placement with the matrix untransposed,
or the projection's offset not removed; a reflection not relabelled; per-interval lists collapsed to their
first; the mask ignored; the default cylinder or cone projection dropped; the cylindrical edge made
straight; the 180° and radius checks loosened; the list-length check off; `Straight` ignored; the shells
misplaced; the bow's middle wrong; the core's bowed edges left straight (a partial join); the azimuth
rotation off by π/4; `around` ignored; `patch_types` dropped by `element_mesh`; `count_factor` ignored;
the pipe tube's inner faces projected; its first-shell radial edges not straight. One survivor was a test
gap and is fixed: with two side blocks the side's middle node is at the uniform angle, so the radial
edge was straight either way — the test now uses three and checks the lines are straight.

## 2026-10-19 — face planarization on Netgen duals

`FacePlanarizer()` defaults (min_decrease 1e-3, damping 0.1, max_sweeps 300 in the harness), harness
`validation/meshing_dual/planarization.py` at base `11323836` plus this change (run before the `flat` stop and
the no-progress exit were added; neither can trip on these duals), macOS arm64, netgen-mesher 6.2.2608,
`PolyhedralDual(45 deg)`, unprojected, moved = the primal cell centroids. One run each, machine shared with
other sessions (timings indicative). Warp = largest vertex height / sqrt(area); tri-volume =
`triangulation_volume_change`; non-orth over interior faces, average acos(mean cos).

| dual (cells) | sweeps, s | total warp | int warp p99 / max | bnd warp p99 | non-convexity p99 / max | concave | tri-volume p99 / max | non-orth max / avg | skew |
|---|---|---|---|---|---|---|---|---|---|
| L 0.01 (3718) | 108, 132 | 255 → 8.9 | .161/.303 → .052/.303 | 0 | .169/.349 → .091/.312 | 57 → 31 | .121/.255 → .058/.255 | 46.2/12.0 → 39.9/7.6 | 1.44 → 1.44 |
| Sozzi 0.01 (13297) | 71, 360 | 836 → 134 | .165/.332 → .097/.327 | .063 → .063 | .176/.331 → .129/.282 | 309 → 119 | .140/.231 → .112/.215 | 47.8/14.2 → 47.2/11.0 | 1.96 → 1.96 |
| cylinder 0.005 (14426) | 112, 990 | 1380 → 130 | .194/.419 → .113/.332 | .024 → .024 | .200/.316 → .136/.248 | 1108 → 321 | .155/.305 → .111/.225 | 50.1/13.9 → 47.7/9.5 | 1.51 → 1.51 |
| sleeve 0.005 (15995) | 118, 1077 | 1200 → 51 | .165/.325 → .068/.290 | .054 → .054 | .168/.293 → .095/.263 | 258 → 59 | .128/.226 → .076/.202 | 51.7/12.2 → 48.3/7.5 | 1.24 → 1.24 |

Closure ≤ 1.1e-19 before and after throughout. Not measured: which faces hold the unchanged maxima (the
likely reading, faces whose vertices are fixed — on the boundary, at features, between cut pieces — is an
inference).

Guard variants on the L 0.01 (same harness quantities): no guard but per-cell convexity → total warp
255 → 6.1 but non-orth max 152.6, skewness 11.2, concave 57 → 65; face bounds per face against its own
start → 0 sweeps; against the starting mesh's worst face, weight 1/area → the row above; weight 1 → total
warp (unweighted) 0.0116 → 0.0004, concave 32, non-orth 46.1/7.7. On a Kuhn dual (n 4, perturb 0.2) the
convexity bound against the previous sweep stalled at total warp 0.79 against 0.033 unguarded (2.17 at
start); against the start it reaches the unguarded result. Over-relaxing the step (1.5, 2, 3) stopped
earlier at a higher warp.

Mutations of the planarizer, each red: no convexity bound; no concavity bound; no non-orthogonality bound;
no skewness bound; a bad face holding its owner only (hung the pass until the no-progress exit was added;
red since); `movable` ignored; no decrease stop; no flat stop; holding everything. Green, dismissed: no
volume bound (never bites on the fixtures; kept because an inverted cell's convexity is NaN and passes a
`>` bound); no halving (no sweep on the fixtures raises the warp; kept as the pass's globalization).

## 2026-10-20 — Gauss–Newton on the true warp against the Jacobi sweeps

Probe `validation/meshing_dual/planarization_gauss_newton.py` (macOS arm64, one run each, machine shared;
counts deterministic, seconds indicative). Total warp as a JAX residual of the free vertices (planes
included); Levenberg–Marquardt `(JᵀJ + λI) dx = −Jᵀr` by unpreconditioned CG to 0.1 relative, `J` by
jvp/vjp; λ₀ = 1e-3·mean(1/area), /3 on success. Judged by `_NoWorse`'s bounds; on a violation λ is raised
×4 until `--shorten` × λ₀, then the worsened cells' free vertices are pinned and the system re-solved;
the planarizer's stopping test.

Kuhn dual (`tetrahedral_grid_3d(8, perturb=0.25)`, 729 cells): Jacobi 282 sweeps, 597 evaluations, total
warp 14.8 → 0.541, 16.8 s. Gauss–Newton: deleting held vertices from a finished step (as the sweeps do)
stalled at 2.67 in 5 iterations — the kept share of each step fell 0.64 → 0.24; pinning and re-solving,
by `--shorten`: 1 → 1.25 (13 it, 3.6 s), 10 → 0.93 (6.5 s), **100 → 0.355 (79 it, 231 evaluations,
1470 CG products, 9.3 s)**, 1000 → 0.54 (155 it, 15.5 s; max warp 0.053 vs Jacobi 0.070). The first
undamped step alone reaches 1.08 with 8 cells' convexity and 5 faces' skewness over their bounds; one
face at the start's worst skewness is worsened by every step tried, so pinning is never avoided.
Handing the result to the sweeps ("gauss-newton+jacobi") added little at 100.

Netgen L maxh 0.01 (3718 cells), `--shorten=100`, against the sweeps (start / Jacobi / Gauss–Newton):

| | start | Jacobi | Gauss–Newton |
|---|---|---|---|
| iterations, evaluations, CG products | | 108, 813, 0 | 50, 233, 10175 |
| time | | 129.8 s | 73.4 s |
| total warp | 255 | 8.94 | 3.77 |
| interior warp p99 / max | .161 / .303 | .052 / .303 | .023 / .249 |
| non-convexity p99 / max | .169 / .349 | .091 / .312 | .069 / .312 |
| concave cells | 57 | 31 | 28 |
| triangulation volume p99 / max | .121 / .255 | .058 / .255 | .034 / .227 |
| non-orth max / avg | 46.2 / 12.0 | 39.9 / 7.6 | 43.2 / 8.0 |
| skewness | 1.44 | 1.44 | 1.44 |

Not measured: where the Gauss–Newton time goes between the 233 guard evaluations and the 10175 CG
products (each a host round trip through a jitted `JᵀJv`); a CG preconditioner (the per-vertex 3×3 blocks
the sweeps already form are the natural one); the other three duals; `--shorten` other than 100 on Netgen.

## 2026-10-20 — Gauss–Newton with a block preconditioner, on all four Netgen duals

Same probe, `--shorten=100`, base `646d1577` plus the preconditioner: CG preconditioned by each free
vertex's 3×3 block `Σ n nᵀ/area + λI` (the frozen-plane blocks the sweeps solve with; identity on a pinned
vertex), stopping on the unpreconditioned residual at 0.1 as before. macOS arm64, one run each, machine
shared (counts deterministic, seconds indicative).

Preconditioner alone, Netgen L 0.01: CG products 10175 → 6332, CG 44.1 → 28.2 s, total 83.0 → 72.0 s,
total warp 3.77 → 3.41, 50 → 59 iterations, other quality within rounding. Kuhn n 8: 1470 → 1382 products
(uniform faces; the 1/area spread is what it corrects).

Preconditioned Gauss–Newton against the sweeps (the sweeps' row from the 2026-10-19 entry above, same
duals and defaults); time split as guard evaluations + CG:

| dual | method | time (eval + CG) | total warp | int warp p99 / max | non-convexity p99 / max | concave | tri-volume p99 / max | non-orth max / avg |
|---|---|---|---|---|---|---|---|---|
| L 0.01 (3718) | start | | 255 | .161 / .303 | .169 / .349 | 57 | .121 / .255 | 46.2 / 12.0 |
| | Jacobi | 132 s | 8.94 | .052 / .303 | .091 / .312 | 31 | .058 / .255 | 39.9 / 7.6 |
| | GN | 72 s (40 + 28) | 3.41 | .021 / .250 | .068 / .312 | 28 | .030 / .229 | 43.4 / 7.9 |
| Sozzi 0.01 (13297) | start | | 836 | .165 / .332 | .176 / .331 | 309 | .140 / .231 | 47.8 / 14.2 |
| | Jacobi | 360 s | 134 | .097 / .327 | .129 / .282 | 119 | .112 / .215 | 47.2 / 11.0 |
| | GN | 323 s (208 + 96) | 74.5 | .055 / .327 | .103 / .281 | 58 | .105 / .179 | 46.1 / 11.0 |
| cylinder 0.005 (14426) | start | | 1380 | .194 / .419 | .200 / .316 | 1108 | .155 / .305 | 50.1 / 13.9 |
| | Jacobi | 990 s | 130 | .113 / .332 | .136 / .248 | 321 | .111 / .225 | 47.7 / 9.5 |
| | GN | 453 s (227 + 203) | 13.4 | .023 / .303 | .077 / .271 | 64 | .038 / .225 | 47.3 / 8.8 |
| sleeve 0.005 (15995) | start | | 1200 | .165 / .325 | .168 / .293 | 258 | .128 / .226 | 51.7 / 12.2 |
| | Jacobi | 1077 s | 50.9 | .068 / .290 | .095 / .263 | 59 | .076 / .202 | 48.3 / 7.5 |
| | GN | 488 s (277 + 183) | 13.4 | .017 / .325 | .072 / .293 | 29 | .046 / .226 | 51.5 / 7.3 |

Iterations / evaluations / CG products: L 59 / 241 / 6332; Sozzi 43 / 415 / 6758; cylinder 58 / 348 /
11384; sleeve 50 / 391 / 9700. Boundary warp unchanged in every run (vertices fixed); skewness max
unchanged but Sozzi 1.96 → 1.95. Closure ≤ 1.2e-19 throughout.

Reading: Gauss–Newton leaves 1.8–10x less total warp and half or fewer of the sweeps' concave cells, with
every p99 lower, in 0.45–0.9x the time. Its maxima are no better than the sweeps' in general (the
cylinder's non-convexity .271 vs .248, the sleeve's warp, non-convexity and triangulation maxima at their
start values); all stay within their bounds. The guard's evaluations are now 45–65 % of the time. Not
measured: which faces hold the maxima; `--shorten` other than 100 on these four.

## 2026-10-20 — the Gauss–Newton step shipped in `FacePlanarizer`

The probe of the two entries above (`validation/meshing_dual/planarization_gauss_newton.py`, deleted with
this change; last at commit `1ec54c02`) became `FacePlanarizer`, its defaults those the probe was run at
(`shortening` 100, the block preconditioner); the Jacobi sweep was deleted. One difference from the probe:
a step still worsening after its cells are pinned raises λ without releasing the pins. Re-measured with
`validation/meshing_dual/planarization.py` (shipped defaults, macOS arm64, one run each, machine shared):

| dual (cells) | iterations, CG products | time | total warp | concave | int warp p99 / max | non-convexity p99 / max | tri-volume p99 / max | non-orth max / avg |
|---|---|---|---|---|---|---|---|---|
| L 0.01 (3718) | 53, 5503 | 90 s | 255 → 3.44 | 57 → 28 | .021 / .250 | .068 / .312 | .030 / .229 | 43.3 / 7.9 |
| Sozzi 0.01 (13297) | 43, 6758 | 349 s | 836 → 74.5 | 309 → 58 | .055 / .327 | .103 / .281 | .105 / .179 | 46.1 / 11.0 |
| cylinder 0.005 (14426) | 58, 11479 | 472 s | 1380 → 13.4 | 1108 → 66 | .023 / .303 | .077 / .271 | .038 / .225 | 47.3 / 8.8 |
| sleeve 0.005 (15995) | 55, 10799 | 714 s | 1200 → 13.3 | 258 → 30 | .017 / .325 | .072 / .293 | .046 / .226 | 51.6 / 7.4 |

Quality matches the probe's within the pin-handling difference; the sleeve's seconds (714 against the
probe's 488) are contention, not the change (its products 10799 vs 9700).

Mutations of the shipped solver (`tests/unit/test_meshing_planarize.py`), each red: no convexity, no
concavity, no non-orthogonality, no skewness bound; a bad face pinning its owner only (the cube's warp
stops at ~13 % of its start, not 7 %); `movable` ignored; no decrease stop; no flat stop; no shortening;
no pinning; planes frozen in the residual (normal and area); the vertex mean frozen; no preconditioner;
an unweighted preconditioner. Green, dismissed: no volume bound (as before: never bites; an inverted
cell's NaN convexity would pass the others); accepting a safe step that raises the warp (no safe step
ever raised it on the fixtures; the history is asserted monotone); no λ decrease after a kept step (path
and cost only: final warp within 10 % either way on three fixtures).

## 2026-10-20 — the planarizer's guard evaluated through one `MeshQuality` per pass

Each trial step used to call the `Mesh`-level quality functions, which between them ran `Mesh.geometry()`
seven times (each re-running `ring_signs`, a topology-only numba pass) and rebuilt the cell checks'
(cell, face) and (cell, vertex) pair tables blockwise. `MeshQuality` (`aquaflux/mesh/quality.py`) now
holds the topology's CSR tables and ring signs, built once; the cell checks are numba kernels looping
each cell's sides against its nodes (no pair table, so no memory beyond the CSR); the face checks and
the vertex heights are `filter_jit`ted. Pair tables cached whole were not built: the same speed would
have cost ~450 pairs per dual cell in int64 (~14 GB at 2M cells).

Results unchanged, measured against the old code on the planarizer's test fixtures (displaced hex
block, Kuhn cube n 3 at two `min_decrease`, Kuhn L n 4, Kuhn n 6): iterations, conjugate-gradient
products, `converged` and the final coordinates bit-identical; warp history within 2.4e-16 of its
start (the heights are now taken in a jitted kernel); the quality functions on the starting mesh within
2.1e-12 deg (non-orthogonality) and 7e-16 otherwise.

Time split of one pass, `validation/meshing_dual/planarization.py`'s Netgen L at maxh 0.01 (3718 cells,
shipped defaults, macOS arm64, one run per arm back to back, load 11–16 from other sessions):

| arm | pass | evaluations (220) | CG products (5503) | block inverse |
|---|---|---|---|---|
| before | 132.2 s | 50.3 s, 0.229 s each | 73.4 s | 3.9 s |
| after | 101.3 s | 3.1 s, 0.014 s each | 87.8 s | 4.8 s |

(The products' rise, 13.3 → 16.0 ms each, is the load climbing from 11 to 16 during the run, not the
change: the product code is untouched.) The harness's own timing, same day: 115.5 → 85.3 s; output
identical in every printed figure. Kuhn duals (`MeshQuality` alone): 0.074 → 0.009 s per evaluation at
1331 cells, 0.219 → 0.0107 s at 4913; parallelizing the cell kernels over cells took the latter from
0.027 to 0.019 s, the jitted heights to 0.011 s.

Direction dismissed: recomputing only the cells and faces whose vertices moved. On the Kuhn n 6 dual
(343 cells, 153 trials) a trial moves a vertex of 85–100 % of the cells (median 100 %, mean 97 %): every
dual cell has free vertices and a Levenberg–Marquardt step moves every unpinned one. With evaluations at
3 % of the pass it could save at most that.

## 2026-10-20 — layers on structured quadrilateral walls, and layers at a sharp concave edge

macOS arm64, netgen-mesher 6.2.2608, Netgen run from a scratch venv through `NetgenProcess(python=)`;
everything else the repository's environment. Layers `PrismLayers(0.0005, 1.2, 5)` on every Sozzi wall
unless said.

**Netgen's edge points, not ours.** Probed whether Netgen meshes faces through edge points we place:
`GenerateMesh(perfstepsend=MESHEDGES)` then a second `GenerateMesh(mesh=..., perfstepsstart=...)` with
`perfstepsstart=MeshingStep.MESHSURFACE` meshes nothing (0 triangles); the integer `3` resumes at the
surface step and keeps the 72 edge segments. Per-edge `maxh = L/n` does not set an edge's count, being
an upper bound only (asking a cylinder's rims for 20, 37, 50 segments gave 31, 37, 49); `Edge.partition`
does (2026-10-21, "Pipe rims divided alike"), used for pipe rims only. So the mapped
grid takes Netgen's own edge points from the face's triangles' boundary, and joins opposite sides of
different counts through one strip of triangles. The kernel's (u, v) evaluation planned for the grid is
not needed: points are interpolated in space and moved onto the face by `CadModel.project`.

**A defect older than this work: layers at a sharp concave edge do not close.** The L (0.2 x 0.1 x 0.1
fused with 0.1 x 0.1 x 0.1, STEP in millimetres), maxh 0.02, layers (0.002, 1.2, 3) on all eight faces,
triangles only, on `main` (d2522fd8): 2 cells of negative volume, 22 unclosed. Sozzi's triangle layers
closed by luck (its junctions are gentler). Three causes, each found on a cell:
- *The layer points above the edge are not cut.* Their dual cells turn the corner as the edge point's
  would uncut: a face of a lifted point's cell, its ring running from one wall's layer cells round to
  the other's, folded (centre-fan area 0.55-0.83 short of its magnitude sum). Giving every interior face
  whose dual edge bends by more than the feature angle a vertex at its centroid (tried) cut the
  unclosed 30 to 14 and was then unneeded: dropped. Cutting the column above each cut point by the same
  walls (`PolyhedralDual.columns`) closes the L, both triangles and quads.
- *The straight centroid-to-centroid segment crosses the corner beside a long cell.* At Sozzi's outlet
  junction a pipe hexahedron (4.9 mm along the pipe) beside an end-plate prism put the cut vertex 0.5 mm
  into the solid (both walls' depths negative); its piece's volume went negative. The crossing on the
  dual edge through the face's centroid is in the fluid. Used everywhere, that rule made plain tet duals
  worse (Sozzi 0.02 worst non-convexity 0.33 → 1.12, L 0.01 0.35 → 0.43), so it is used only where the
  straight crossing is in front of the walls.
- *Nearly parallel planes meet far away.* On the inlet saddle two neighbouring column points carry
  slightly different walls; the least-squares hub on their planes went 131 mm off (`pinv` at rcond
  1e-8). Dropping singular values under 0.05 (`_PLANES_APART`, about 3°) treats them as one plane.
Cutting the top row of each column too is fine once the hub is bounded (an early run with the top row
cut failed on that hub, not on the top row). Each piece red when removed: no columns (fast dual test, the
two Netgen tests); always straight (the same); always through the face centroid (the Netgen L's
convexity bound, tightened 0.45 → 0.4 at 0.349); `_PLANES_APART` 1e-8 (the Sozzi mapped test).

**Plain tet duals after the change** (`dual_scale_convexity.py`, worst non-convexity overall | extra
pieces): L 0.02 / 0.01: 0.26 / 0.349 | 0.160 / 0.188; Sozzi 0.02 / 0.01: 0.333 / 0.331 | 0.303 / 0.282
(before: 0.26 / 0.35 | 0.17 / 0.16; 0.33 / 0.33 | 0.32 / 0.26). Closure and volume unchanged.

**Sozzi, triangles against mapped lamp cylinder and pipes** (`mapped_layers_sozzi.py`; the lamp's tip and
the vessel triangulated; "hexahedral walls" the share of a patch's faces that are quadrilaterals of
hexahedral dual cells; first layer the dual vertices 0.2-0.8 mm from the true lamp wall, 0.1 < x < 0.7):

| maxh, arm | cells | s | hexahedra | hexahedral walls lamp / pipes | first layer on the lamp, median (5-95 %) | non-orthogonality by the lamp max / p99; everywhere max / p99 |
|---|---|---|---|---|---|---|
| 0.02 triangles | 46244 | 13.1 | 86 | 0 / 0 | 0.5069 (0.5066-0.5087) | 58.1 / 15.8; 85.3 / 50.6 |
| 0.02 mapped | 48509 | 13.0 | 28393 | 0.986 / 0.988 | 0.5033 (0.5025-0.5040) | 60.0 / 15.7; 89.8 / 46.2 |
| 0.01 triangles | 59515 | 16.7 | 85 | 0 / 0 | 0.5069 (0.5066-0.5087) | 51.6 / 15.9; 84.4 / 49.1 |
| 0.01 mapped | 60985 | 16.1 | 28389 | 0.986 / 0.988 | 0.5033 (0.5025-0.5040) | 59.7 / 14.8; 88.8 / 45.8 |

The lamp's surface is the same at both sizes (its 10 mm radius sets Netgen's size there), hence equal
lamp numbers. Faces over 80° at 0.02: triangles 2, mapped 11 -- inside the outlet pipe's entry (r ≈ 4 mm,
x ≈ 888 mm) and at the inlet saddle, where the pipes' five layers leave a core of a few millimetres for
pyramid caps and tetrahedra. Quality, not correctness: every cell closes. Timings from one run each,
beside another session's bfs3d case: counts good, seconds indicative.

## 2026-10-20 — the pyramid caps' apex cells: sizing, a diagonal split, a crevice limit

Sozzi, maxh 0.02, layers (0.0005, 1.2, 5) on every wall, lamp cylinder and both pipes mapped, macOS arm64,
netgen-mesher 6.2.2608. Each arm meshed from five drawings rotated 1-5 µrad about x (a single run per arm
is not enough: one change anywhere moves Netgen's whole fill, and lamp counts moved 23 -> 65 between two
runs with the same lamp caps). "Lamp small": dual cells under 2 mm (cube root of volume) 13.5-22 mm from
the lamp axis, 300 < x < 500 mm; "pipes > 50°": faces over 50° non-orthogonal in the pipes; volume ratio
the larger over the smaller of a face's two cells. Scripts were scratch (not kept); the harness they
extended is `validation/meshing_dual/mapped_layers_sozzi.py`.

| arm | lamp small | lamp worst | pipes > 50° | other > 50° | non-orth p99 | volume ratio p99 |
|---|---|---|---|---|---|---|
| triangle layers (no caps) | 0 | 44-52° | 2030-2370 | 72-95 | 50-51° | 14 |
| caps 0.25 × first edge (shipped) | 63-70 | 51-56° | 104-134 | 112-129 | 46.2° | 160-167 |
| sized: 0.7 × mean edge, fold 0.6, gap 0.4 | 23-28 | 53-62° | 4759-4975 | 104-119 | | |
| sized, gap 0.07 | 77-88 | 49-68° | 48-118 | 97-111 | | |
| crevice limit 90° (+ gap 0.4, target 0.7) | 53-60 | 52-59° | 126-158 | 108-123 | 41.3-43.3° | 231-234 |
| crevice limit 105° | 63-70 | 49-53° | 216-348 | 111-123 | 47.1-47.7° | 156-166 |
| diagonal split of the top quad, no caps | 4-39 | 86-90° | 5870-6920 | 300-960 | 64.5-68.3° | 44-63 (max to 19700) |
| caps 0.25 × shortest edge (2 draws) | 97-98 | | 106-107 | 103-113 | 46.2° | |

- **Each apex is a primal point, so a dual cell**, squeezed between the hexahedra's top and the fill:
  hence the small cells and the volume-ratio tail (164 against 14).
- **Tall caps help the open lamp and wreck the pipes' cores.** The layers leave each 9.6 mm pipe a core
  of radius ~4.4 mm with twelve quadrilaterals round it; above ~0.6-0.85 mm the core's dual goes bad
  uniformly along the pipe. The angle left the fill between two neighbouring caps (the crevice,
  `(π − 2π/n) − 2 atan(2h/w)`) is 90° at 0.66 mm there -- but a crevice threshold does not separate
  the cases either (72° along the lamp was fine, 77° in the pipe bad), and limiting it to 90° holds the
  lamp's caps to ~2.5 mm, inside a middle band (2.2-2.5 mm) worse than either low or tall.
- **The diagonal split** (top quad cut by a ridge diagonal, the hexahedron a seven-faced cell, no apex):
  Netgen's tetrahedra on the split triangles are poor (2023 under 0.1 quality, volume over rms edge cubed
  scaled to 1 for a regular one, against 131 for triangle layers), the smallest dual cells in the pipes'
  cores, one draw unclosed (6e-3); the valley diagonal gave a flat tetrahedron on four corners and a
  two-node dual face. Rejected.
- **The first edge is the right constant, not an accident**: a mapped grid's quadrilaterals start along
  its rows, so it is the width round the band -- short round a pipe (2.3 mm), long round the lamp's layers
  (6.8 mm against 5 along). Shortest edge lowered the lamp's caps (97 small); mean edge, 0.9 mm in the
  pipes, was bad there (2969 over 50°, one run).
- Reviewed by three agents (primal join, dual-side merging, critique of the rule). Merging an apex cell
  into a neighbour (toy model, no Netgen): closed but L-shaped cells, non-convexity 0.2 → ~1.0, so the
  planarizer's guard would refuse every merge. Open: #785.

## 2026-10-20 — pipes' cores as structured O-grids

Sozzi, maxh 0.02, layers (0.0005, 1.2, 5) on every wall, lamp cylinder and both pipes mapped, macOS
arm64, netgen-mesher 6.2.2608; ensembles of drawings rotated 1-5 µrad about x (scratch script over
`validation/meshing_dual/mapped_layers_sozzi.py`, which now finds the pipes by default). Pipe faces: face
centroid x > 0.889 m or z > 0.06 m.

| arm (draws) | cells | pipes > 50° | pipes worst | pipes p99 | other > 50° | volume ratio p99 | smallest cells (m³) |
|---|---|---|---|---|---|---|---|
| no cores (#783) (5) | ~48.5k | 85-93 | 87-89° | 46.8° | 109-122 | 165-177 | ~3.6e-10 |
| O-grid, shell continuing the layers' growth (5) | ~53.6k | 66-72 | 62-74° | 7.3° | 150-170 | 34.5-35.2 | 3e-11 (several) |
| O-grid, one-cell shell (5, shipped) | ~49.5k | 32-37 | 58.6-60.1° | 6.0° | 117-137 | 36.3-38.3 | 2-3 at 3.5-3.9e-11, then ~1.4e-10 |
| O-grid, junction_rows 4 (2) | ~53.8k | 120-141 | 64-90° | 12.3-14.7° | 99-117 | 47.5 | ~3e-11 |
| one-cell shell, rows not flattened (2) | ~49.6k | 38-41 | 62.6-62.7° | 8.2° | 123-125 | 39 | ~4e-11 |

- The smallest cells are the junction discs' cap apexes (x ≈ 898.5 mm in the outlet, z ≈ 53 mm in the
  riser), ~0.33 mm across: #785, confined to two discs. The continued shell's first cell is 0.3 of its
  depth (the layers' alternation), so its outer disc quads are thin, their caps low and the apex cells
  smaller and more numerous; it also made other faces over 50° rise (150-170).
- **The saddle rows**: with the continued shell the riser's junction disc, interpolated straight from its
  saddle rim, kept 98 % of the 1 mm saddle and two disc caps crossed (Netgen "boundary mesh is
  overlapping"; one intersecting pair found by a triangle-triangle sweep of the fill's surface). With the
  one-cell shell it no longer fails, but flat rows are still better (table).
- Inlet and outlet areas 0.9783 of πr² in every arm: the open ends' O-grid cross-sections cover what the
  end faces' triangles did. Mutations, each red: open end faces kept (the stub-cylinder test); the core
  joined one ring point round (radial-line test); the fluid-inside check off; the junction disc wound
  the wrong way and rows not flattened (both the Sozzi pipes test; the latter also the mapped test).

## 2026-10-20 — where the worst faces and cells are after planarization

`validation/meshing_dual/planarization.py` (its "where" lines), shipped `FacePlanarizer()` defaults with the
cheaper guard (#784), harness duals (unprojected, one patch, feature angle 45 deg, moved = `vertex_cell >= 0`),
macOS arm64, one run each. Faces: boundary / "by cut" (an interior face with a vertex the cut placed between
pieces) / "by boundary" (a boundary vertex, none between pieces) / "free" (every vertex movable). Cells:
cut piece (a vertex between pieces) / wall (a boundary face) / interior. Shares of the worst 1 %, with each
class's maximum over the mesh:

| dual | interior warp: by cut / by boundary / free | boundary faces' max warp | non-convexity: cut piece / wall / interior | tri-volume: cut piece / wall / interior | pass time |
|---|---|---|---|---|---|
| L 0.01 | 11 % (.184) / 41 % (.250) / 48 % (.117) | 0.000 | 50 % (.312) / 47 % (.250) / 3 % (.069) | 55 % (.124) / 39 % (.229) / 5 % (.038) | 42.5 s |
| Sozzi 0.01 | 3 % (.208) / 30 % (.169) / 67 % (.327) | 0.160 | 38 % (.281) / 47 % (.205) / 16 % (.209) | 9 % (.156) / 87 % (.179) / 4 % (.128) | 131.9 s |
| cylinder 0.005 | – / 32 % (.149) / 68 % (.303) | 0.026 | – / 40 % (.124) / 60 % (.271) | – / 68 % (.116) / 32 % (.225) | 248.1 s |
| sleeve 0.005 | – / 41 % (.325) / 59 % (.233) | 0.125 | – / 79 % (.293) / 21 % (.140) | – / 89 % (.226) / 11 % (.110) | 276.6 s |

Quality after the pass identical to the 2026-10-20 shipped-solver entry (the guard change kept results).
Reading: **most of the worst interior warp is on faces every vertex of which was free** (the worst face of
Sozzi and the cylinder among them), so a fixed vertex is not what keeps them warped; whether the guard's
bounds or the competition between neighbouring faces does is not distinguished by this run. **Wall cells
hold most of the worst triangulation-volume dependence (39–89 %)** and much of the worst non-convexity, so
sliding boundary vertices along the drawing is a lever for those, on curved walls (the L's walls are flat
and its boundary faces exactly so); on the L the cut pieces hold the worst non-convexity.

## 2026-10-20 — what holds the worst all-free faces: the per-cell convexity bound

`validation/meshing_dual/worst_face_limits.py` (base `6532fa0f`), harness duals (unprojected, one patch,
feature angle 45 deg, moved = `vertex_cell >= 0`), macOS arm64, one run per arm. Tracked faces: the
interior faces in the shipped pass's worst 1 % by warp whose every vertex is free (L 0.01: 106, start
median 0.075; Sozzi 0.01: 477, start median 0.083). Bounds loosened by setting their column to infinity
(positive volume always kept). Tracked warp median / max, share halved against the shipped pass, then
total warp, worst non-convexity, concave cells, worst interior non-orthogonality, time:

| arm | L | Sozzi |
|---|---|---|
| shipped | .035 / .117, 0 %; 3.44, .312, 28, 43.3, 42 s | .073 / .327, 0 %; 74.5, .281, 58, 46.1, 131 s |
| no convexity bound | .024 / .117, 27 %; 2.90, .312, 27, 46.1, 20 s | .008 / .321, 85 %; 58.1, .280, 48, 47.8, 30 s |
| no concavity bound | .024 / .102, 34 %; 2.23, .312, 68, 43.2, 14 s | .039 / .169, 51 %; 74.5, .282, 105, 47.7, 17 s |
| no non-orthogonality bound | identical to shipped | identical to shipped |
| no skewness bound | .026 / .117, 27 %; 3.05, .249, 27, 43.3 (skew 1.57), 24 s | ≈ shipped (skew 2.07) |
| no quality bound | .005 / .039, 82 %; 0.63, 9.04, 109, 179.3, 10 s | .004 / .038, 100 %; 52.8, 1.93, 224, 179.4, 21 s |
| shipped, tight stop (1e-6, 1000 it) | .032 / .117, 10 %; 3.16, .312, 29, 46.2, 1171 s (cap) | .065 / .327, 16 %; 69.1, .281, 73, 47.8, 3051 s |
| no quality bound, tight stop | .004 / .024, 95 %; 0.34, 25.6, 155, 178.7, 39 s | .002 / .023, 100 %; 51.9, 2.43, 320, 179.0, 137 s |

Starting worsts: L non-convexity .349, concave 57, non-orth 46.2; Sozzi .331, 309, 47.8. Reading: the
unconstrained optimum flattens the tracked faces (so neighbour competition does not hold them), and a
much tighter stop barely helps (so the stopping test does not); the bounds hold them, chiefly **the
per-cell convexity bound** (no cell less convex than it started), with the concavity bound and on the
L the skewness bound also binding; the non-orthogonality bound never binds on these two. Without the
convexity bound every reported measure stays within its starting worst (non-convexity max .312 / .280,
non-orthogonality 46.1 / 47.8), concave cells fall, total warp falls 16–22 % and the pass is 2–4x faster.
Without the concavity bound concave cells more than double. Not measured: a convexity bound against the
start's worst cell (the analogue of the face bounds) as such; cylinder and sleeve; the effect on
unit-test fixtures (the L4 fixture's convexity bound bites, `test_meshing_planarize.py`).

## 2026-10-20 — the convexity bound against the start's worst cell

`FacePlanarizer` now bounds every cell's non-convexity by the starting mesh's largest (was each cell's own
start); concavity stays per cell. `planarization.py`, shipped defaults otherwise, harness duals as above,
macOS arm64, one run each, shared machine. Per-cell bound (the entry "where the worst faces and cells are"
above) → start's-worst bound:

| dual | total warp | concave (start) | int warp p99 | non-convexity p99 / max (start max) | tri-volume p99 / max (start max) | non-orth max (start) | time |
|---|---|---|---|---|---|---|---|
| L 0.01 | 3.44 → 2.90 | 28 → 27 (57) | .021 → .016 | .068/.312 → .066/.312 (.349) | .030/.229 → .026/.211 (.255) | 43.3 → 46.1 (46.2) | 42 → 37 s |
| Sozzi 0.01 | 74.5 → 58.1 | 58 → 48 (309) | .055 → .015 | .103/.281 → .096/.280 (.331) | .105/.179 → .104/**.299** (.231) | 46.1 → 47.8 (47.8) | 131 → 41 s |
| cylinder 0.005 | 13.4 → 5.23 | 66 → 41 (1108) | .023 → .006 | .077/.271 → .057/.271 (.316) | .038/.225 → .027/.185 (.305) | 47.3 → 50.1 (50.1) | 248 → 187 s |
| sleeve 0.005 | 13.3 → 6.6 | 30 → 8 (258) | .017 → .007 | .072/.293 → .066/.293 (.293) | .046/.226 → .038/.226 (.226) | 48.3 → 51.4 (51.7) | 277 → 128 s |

⚠️ **Sozzi's worst triangulation-volume dependence rose above the start's (0.299 vs 0.231)**: that measure
is not bounded, and the wider convexity bound let one wall cell (93 % of the worst 1 % are wall cells)
get worse. The `worst_face_limits.py` entry's "every reported measure stays within its starting worst"
covered the measures that harness prints, which do not include it. Worst non-orthogonality now rises to
the start's worst where it had stayed below. The mesh is refused if a starting cell's volume is not
positive or its quality undefined (a NaN would otherwise be the bound): a Kuhn n 4 grid perturbed 0.35
(seed 1) has a 3e-19 tetrahedron and two dual cells of volume about −1e19.

Mutations, each red: the bound per cell again (`test_no_cell_ends_less_convex_than_the_least_convex_it_started_with`
asserts some cell does get less convex); the bound removed (the Kuhn L, perturb 0.3 seed 1, ends at
0.744 against 0.730); the concavity bound removed; the refusal removed.

## 2026-10-20 — triangulation-volume dependence bounded by the start's worst

The guard gains a fourth cell bound: no cell's `triangulation_volume_change` above the starting mesh's
largest (`MeshQuality.triangulation_volume_change`; the `Mesh` function is a caller of it). Same harness,
duals and defaults as the entry above, one run each. Sozzi 0.01: worst triangulation-volume change
0.299 → 0.228 (start 0.231); total warp 58.1, concave 48, non-convexity max 0.280, non-orthogonality
47.8, 31 iterations, 41 s — all as without the bound. L, cylinder, sleeve: every printed number
identical to the entry above (the bound never binds there). No small fixture was found on which it
binds (Kuhn cubes and Ls, perturbed; cube-to-ball maps; Netgen Sozzi 0.02 / 0.03, cylinder 0.02, sleeve
0.01 — on each the pass lowers the worst value unbounded), so it is pinned on the guard itself
(`test_the_guard_holds_each_cells_triangulation_volume_to_the_starts_worst`). Mutations, each red: the
bound per cell; the bound removed; the column not measured; the kernel taking the gain only.

## 2026-10-20 — the octree core wired into `CadMesher`: the fill's grading, and two vessels

Configuration: macOS arm64, the scratch Netgen environment (netgen-mesher 6.2.2608, OCP 8.0.1), branch
`claude/octree-core-wiring` on #783's `claude/quad-prism-layers` (`1ed7de5e`), one run each, other
sessions running light jobs (timings indicative only).

**Netgen grades the fill from the surface it is given** (`validation/meshing_netgen/fill_grading.py 40 4`):
a unit box of triangles of edge 0.025 round a hole [0.3, 0.7]^3 of edge 0.1, `max_size` 0.1. Median tet
edge by depth from the outer wall, grading 0.3: 0.036 (at the wall), 0.055, 0.069, 0.075, 0.086 (0.1),
0.098, 0.108, 0.119, 0.123, 0.130 (0.25); grading 0.5 the same shape (0.037 → 0.123). So no size field is
built: the fill's `max_size` is raised to the core's wall cubes where they are larger, nothing else.

**Two vessels at maxh 0.01** (`validation/meshing_dual/octree_core.py`, layers first 0.5 mm, growth 1.2,
five; core cubes 0.04; arms with the clearance unset and, run directly, `0.01 0.04 0.02`):

| | cells | s | cubes | non-orth max / p99 deg | skew max / p99 | volume ratio p99 / max | faces > 20 |
|---|---|---|---|---|---|---|---|
| Sozzi, polyhedral | 60985 | 13 | — | 148.7 / 45.8 | 22.9 / 0.74 | 148 / 598 | 20678 |
| Sozzi, any core arm | refused: no cube fits | | | | | | |
| open vessel, polyhedral | 70993 | 15 | 0 | 59.2 / 28.5 | 1.30 / 0.64 | 10.1 / 128 | 352 |
| core, wall 0.01, clearance 0.02 (the defaults) | 81427 | 17 | 3746 | 62.7 / 39.4 | 1.62 / 0.67 | 78.1 / 612 | 10866 |
| core, wall 0.02, clearance 0.04 (default clearance) | 57256 | 12 | 48 hexahedra in the core | 55.9 / 27.3 | 1.18 / 0.57 | 14.4 / 322 | 971 |
| core, wall 0.02, clearance 0.02 | 57602 | 12 | 583 | 58.3 / 34.7 | 1.62 / 0.60 | 16.9 / 107 | 1605 |

The open vessel: a cylinder 0.3 m across, 0.6 m long, less a sleeve 25 mm across on its axis, layers on
the wall and the sleeve, its ends bare. Sozzi: the mapped lamp and pipes of `mapped_layers_sozzi.py`.
- **Sozzi has no room for a core at this size**: its chamber is an annulus 34 mm across (reactor radius
  44.5 mm, lamp about 10 mm), about 19 mm once the layers are out, under twice a clearance plus a cube
  at either clearance.
- **With the agreed defaults the core is worse on every count**: +15 % cells (likely because wall cubes of `maxh` are
  smaller than the tets' dual cells, which come out larger than the tets) and 31x the faces over a volume
  ratio of 20. Cells under a tenth of `maxh^3` rise by 2581 over the polyhedral arm's 39907 (which are
  nearly all layer cells) -- about one per core cap, so the caps' apex cells (#785's mechanism) are the
  likely cause; **not separated** from the transition cubes' own polyhedra.
- **Wall cubes of twice `maxh` save 19 % of the cells** but stay worse than polyhedral in volume ratio
  (p99 14–17 against 10) and, at clearance 0.02, non-orthogonality p99 (34.7 against 28.5). At the
  default clearance (0.04) almost no core is kept in the 0.12 m annulus: a kept cube needs its box grown
  by the clearance clear of the surface, and an axis-aligned box of 0.1 fits a ring 0.12 wide only near
  the axes (its diagonal is 0.14).
- Not decided; put to the project owner: the defaults (`wall_size`, `clearance`), and whether #785's
  apex cells must be fixed before a hex core is worth selecting. Not tried: a Euclidean clearance in
  place of the grown box; caps of another height on the core.

**Attribution** (`validation/meshing_dual/octree_core_attribution.py`, the open vessel at the defaults:
maxh 0.01, cubes 0.04, wall 0.01, clearance 0.02; one draw): each dual cell classed by the primal point
it stands for.

| class | cells | median volume / wall³ | faces ratio > 20 (the smaller cell) | faces > 45° (touching) |
|---|---|---|---|---|
| core cap apex | 7528 | 0.136 | 10250 | 1365 |
| transition centre | 670 | 2.50 | 0 | 0 |
| cube corner | 18329 | 0.893 | 2 | 334 |
| fill (Netgen's points) | 5726 | 2.33 | 153 | 69 |
| layers and walls | 49174 | 0.062 | 461 | 77 |

Of 10866 faces over a volume ratio of 20, **94 % have a core cap's apex cell as the smaller** (5758
against a cube corner, 4135 against a fill cell); of 1439 faces over 45°, **95 % touch one**. The
transition cubes' polyhedra carry none. The apex cells are also most of the core arm's extra cells: 7528
of them, beside only 5726 points of Netgen's own in the whole gap. So the core's case is #785's.

## 2026-10-20 — the caps' apexes moved after the fill: measured and dropped

`ApexRelaxation` (not merged; commit `5c3dd83d` on the pushed branch `claude/apex-relaxation`, with its
unit tests and the harness `validation/meshing_dual/apex_relaxation.py`): after Netgen's fill, a Jacobi
sweep moving every cap apex (wall caps and the octree core's) to the mean of the points it shares a
tetrahedron with, a move kept only where no tetrahedron about the apex -- the fill's, and each cap's four
corner tetrahedra -- falls below the smaller of its start shape and 0.1 (volume over a regular
tetrahedron's of its rms edge), halved up to six times, at most 20 sweeps to a tolerance of 1e-3 of the
apex's mean neighbour distance. Five draws per case (rotated 1-5 µrad about x), macOS arm64, netgen-mesher
6.2.2608, on `main` `2af33d0e` plus #788; left → relaxed, ranges over the draws:

| | small cells away from walls | faces volume ratio > 20 | ratio p99 | faces > 50° | non-orth p99 |
|---|---|---|---|---|---|
| Sozzi, maxh 0.02, no core (pipes O-gridded, lamp mapped) | 45-54 → unchanged | 2634-2688 → 2720-2770 | 36-38 → 39-41 | 153-170 → 97-115 | 34.1 → 33.9° |
| open vessel, maxh 0.01, octree core 0.04 at its defaults | 1785-1832 → 1853-1915 | 10837-11259 → 11126-11553 | 78-81 → 82-85 | 872-964 → 1598-1800 | 39.8 → 37.8° |

"Small": Sozzi, cells under 2 mm across 13.5-22 mm from the lamp axis at 300 < x < 500 mm; vessel,
cells under 0.4 maxh across more than 12 mm inside its walls (the core caps' apex cells).
- **It does not reach the apex cells**: their count is unchanged on Sozzi and rises 4 % on the vessel,
  and the lopsided faces rise ~3 % on both. The likely reason (inferred, not separately measured): the
  apex cell is small because Netgen places its own points close to the cap, sizing them from the cap's
  small side triangles; moving the apex afterwards reshapes its tetrahedra but cannot enlarge them. A fix
  must change what the fill sees (taller caps on the core, where #785's pipe-core objection does not
  apply, or Netgen closing the quadrilaterals itself), not move the apex after it.
- Faces over 50° fall a third on Sozzi but nearly double on the vessel; not worth an opt-in setting
  (project owner, 2026-10-20: recorded and dropped).
- Seen in both arms on Sozzi at this `main`: the worst face 111-140° non-orthogonal, which usually means an
  inside-out cell; not investigated here.

## 2026-10-20 — Memory by stage, toward 2M cells

Harness `validation/meshing_dual/memory_by_stage.py` (each `CadMesher` stage wrapped: this process's
physical footprint at start, sampled peak and end; Netgen's child's peak resident set beside the stage
that ran it; one process per size). Sozzi water as `mapped_layers_sozzi.py`: `max_size` as listed,
`feature_angle` 0.785, five layers (first 0.5 mm, growth 1.2) on every wall, lamp and pipes mapped, pipe
cores where `find_pipes` accepts them; macOS arm64, netgen-mesher 6.2.2608, jax 0.10.2, one run each.

**The macOS allocator cache.** With libmalloc's defaults a freed large block stays charged to the process:
`element_mesh` alone left the footprint at 1.40 GiB after returning (tracemalloc: 6 MiB live), unchanged by
`malloc_zone_pressure_relief`, reused by a second call, but a new 1.1 GiB array landed on top (2.52).
With `MallocLargeCache=0` the same sequence returns to 0.16. So every footprint below is with
`MallocLargeCache=0`; with defaults the 266k run peaked at 2.40 against 1.72 GiB.

**Before** (main after #786), cache off:

| maxh | cells | peak GiB | held after meshing | Netgen child RSS |
|---|---|---|---|---|
| 0.005 | 133112 | 0.96 | 0.33 | 0.25 |
| 0.0035 | 265279 | 1.72 | 0.40 | 0.52 |
| 0.0025, triangle walls, no pipe cores | 659651 | 4.71 | 0.68 | 1.52 |

(0.0025 with mapped, layered walls does not mesh: #790.) At 266k every peak was `Mesh.geometry`: four
calls on the primal (1.54M faces, 5.19M incidences, 663k cells), each ~1.0 GiB above its start, ~0.5 s
each; the dual's own geometry after meshing +0.92 GiB. The primal's arrays are 119 MiB; the
`element_mesh` inputs 35 MB. XLA's own temp for `_fused_geometry` on that primal: 865 MiB, of which the
centre fan 724.

**After** (blocked centre fan; primal geometry once in `build_traced`), same configuration:

| maxh | cells | peak GiB | dual s | dual geometry above start |
|---|---|---|---|---|
| 0.0035 | 265279 | 1.31 (1.72) | 5.3 (6.4) | 0.23 (0.92) |
| 0.0025, triangle walls, no pipe cores | 659651 | 3.33 (4.71) | 15.2 (21.6) | 0.64 (2.66) |

Now the peak is `Mesh.from_csr`: in the dual's assembly 2.57 → 3.33 GiB at 660k (its inputs built from
Python lists before it, 1.90 → 2.57), and the primal's 2.27 → 3.04 (`element_mesh`'s numpy before it,
0.34 → 2.27). Linear in cells from 266k to 660k within ~15 % (6.0 then 6.8 KB per cell before; 4.1 then
4.7 after), so ~9.6 GB at 2M (cache off), an extrapolation.

Found on the way: #789 (pipes cored only when Netgen's rim counts are equal multiples of four: 1 of 2 at
0.005, 0 at 0.0035, 2 at 0.0025) and #790 (the fill's "boundary mesh is overlapping" at 0.0025 with
hexahedral layers, pipe cores or not; triangle walls with prism layers and mapped walls without layers
both mesh).

## 2026-10-21 — hexahedra to tetrahedra: what the literature and other meshers do

Background for #785/#790 (three research agents, web and source reading, 2026-10-20; [S] a source says
it, else inferred). No source reports our exact pathology, a small dual cell at an inserted apex; mature
codes avoid creating one.
- **Owen & Saigal (2001)**, "Formation of pyramid elements for hexahedra to tetrahedra transitions",
  CMAME 190:4505-4518 (SAND2000-0494J, osti.gov/biblio/751546, full text read) [S]: building pyramids
  before the tet fill was rejected (intersection checks against opposite boundaries). Preferred:
  mesh with each quad split along a diagonal, 2-3 swap until the diagonal has exactly two tets, merge
  them into a pyramid -- the apex is a tet-mesher vertex, not an inserted one; ~10 % fall back to a node
  at the diagonal's midpoint, smoothed. Immediately after, the tets are "unacceptable"; cleanup by local
  Laplacian smoothing within three edges plus T23/T32/T44 flips is part of the method. Ideal apex on the
  base normal at h = 0.98 b (square base b), by the min radius ratio of the pyramid's four tets.
- **Netgen** (`libsrc/meshing/meshfunc.cpp::CloseOpenQuads`, `rules/pyramidrules*.rls`) [S]: closes open
  quads with rule sets (hexes if `try_hexes`, prisms, "connect pyramid to triangle", "connect to
  vis-a-vis point"); rules reusing a front point cost quality 1, new apexes ("Pyramid on quad" h 0.5,
  "small Pyramid on quad" h 0.1) cost 100; `giveuptolopenquads` default 15 (">100 for free pyramids").
  Probed from Python (netgen-mesher 6.2.2608, `quad_probe` scratch, a unit box with one quad or a 4x4
  quad face, both windings, `giveuptolopenquads` to 2000, `giveuptol` 1000, Delaunay on and off): never
  closes them ("mesh has still open quads"); the same box with the quads split meshes. Not pursued.
- **Gmsh** (live-clones mirror, `meshGRegionBoundaryRecovery.cpp`, `meshRelocateVertex.cpp`) [S]: apex
  inserted flat at the quad centroid, relocated after the fill toward centroid + 0.5 sqrt(area) n under
  a quality guard; pyramids off, the quad is split by the diagonal with the better min triangle quality.
- **Production meshers** [S]: Fluent's polyhedral conversion dualises non-hex cells only, keeps hexes and
  splits a hex face that meets a dual cell (Fluent UG §7.11-7.12; prism layers "used directly", with
  stair-stepping at the layer/tet junction); Mosaic/poly-hexcore joins hex core and poly-prism layers with
  polyhedra (algorithm unpublished; Ansys patent US 11,282,274 builds the connection from the last layer's
  outer surface and its dual); STAR-CCM+ meshes the core on a subsurface first and extrudes the layers
  from its boundary faces, so they match by construction; snappyHexMesh and cfMesh transition with split
  polyhedra; NSU3D/DLR form transition pyramids from existing nodes (Mavriplis & Pirzadeh 1999, NASA
  CR-1999-208999). Fluent warns pyramids on high-aspect quads give highly skewed triangles.
- Ranked for us: (1) taller caps, 0.5-1.0 edge, clearance-limited (Owen 0.98 b, Netgen and Gmsh 0.5),
  now that the pipes' cores are O-grids rather than narrow capped cores; (2) Owen-Saigal: the apex a
  Netgen point; (3) a hybrid dual keeping the hexahedral layers primal (Fluent), a redesign of the layers.

## 2026-10-21 — taller caps sized to their room, and pyramids merged from the fill (Owen-Saigal)

`validation/meshing_dual/cap_transitions.py` (five draws rotated 1-5 µrad about x; Sozzi at maxh 0.02
with layers 0.5 mm × 1.2 × 5 on every wall, lamp mapped, pipes found and O-gridded; open vessel at maxh
0.01 about an octree core of 0.04 m cubes, wall size and clearance unset), `main` 33172345 plus this
change, macOS arm64, netgen-mesher 6.2.2608. "Small": Sozzi cells under 2 mm 13.5-22 mm from the lamp
axis at 300 < x < 500 mm; vessel cells under 4 mm more than 12 mm inside its walls. Ranges over the draws.

| Sozzi | small | ratio>20 | ratio p99 | non-orth p99 | >50° (pipes) | >90° | failed |
|---|---|---|---|---|---|---|---|
| caps 0.25 × first edge (shipped) | 45-54 | 2634-2688 | 36-38 | 34.0-34.2° | 153-170 (72-79) | max 111-140° | 0 |
| caps 0.45 | 2-7 | 1126-1221 | 15.6-16.2 | 31.5-31.7° | 122-127 (62-67) | max 151-177° | 0 |
| `ClearedCaps()` (0.7, fold 0.6, gap 0.4) | 5-8 | 1221-1315 | 16.0-17.0 | 31.8-32.1° | 120-129 (56-62) | 3-4 | 0 |
| `ClearedCaps(1.0, fold 0.8)` | 0 | 129-173 | 7.0 | 41.6-41.9° | 200-220 (67-69) | 4-6 | 0 |
| `MergedCaps()` (no cleanup) | 1-31 | 834-1977 | 13.8-25.7 | 32.0-47.9° | 197-1333 (124-144) | 5-41 | 2 of 5 |

| open vessel, octree core | small | ratio>20 | ratio p99 | non-orth p99 | >50° | failed |
|---|---|---|---|---|---|---|
| caps 0.25 (shipped) | 1785-1832 | 10837-11259 | 78-81 | 39.6-39.9° | 872-964 | 0 |
| caps 0.45 | 393-492 | 2823-4131 | 17.5-20.9 | 39.4-47.1° | 1899-3254 | 0 |
| `ClearedCaps()` | 257-322 | 2311-2832 | 17.5-17.8 | 39.3-40.4° | 1402-1686 | 0 |
| `ClearedCaps(1.0, fold 0.8)` | 91-133 | 1471-2947 | 14.9-18.0 | 39.4-43.8° | 1929-2557 | 0 |
| `MergedCaps()` (no cleanup) | 3271-3635 | 14654-16265 | 94-205 | 47.3-47.9° | 2751-3366 | 0 |

Attribution (`validation/meshing_dual/cap_attribution.py`, one draw each, caps 0.25 / `ClearedCaps(1.0)`):
- Faces over 50° touch an apex cell: Sozzi 139 of 155 / 194 of 201; vessel 863 of 875 / 1900 of 2043.
  On the vessel nearly all join two core apex cells (847 / 1741): the face across the crevice between
  neighbouring caps. Apex cells' median size: Sozzi 3.26 / 9.41 mm against 2.56 mm for the rest (tall
  wall caps fill the lamp's annulus: Netgen adds 62 points there, against 1606); vessel 5.14 / 6.97 mm.
- Sozzi's faces over 90° (3-6 a draw, in every cap arm) join two non-apex cells at x ≈ 45-50 mm,
  |y| ≈ 6.5 mm, z ≈ 40 mm; none in the vessel.
- `MergedCaps` after Netgen fills against the quads cut along their ridge diagonal (Sozzi one draw:
  1381 of 1652 quads two tetrahedra round the cut, 247 three, 23 four, 1 one): direct 1341, swapped 245,
  opened 66. On the vessel 1613 of 5329 two-tetrahedra rings have their apex on another core quad's
  corner -- the octree core's concave steps -- so a pyramid there would cover part of that quad, and
  3316 of 7528 quads fall back to a point of their own near the quad, recreating the squeezed cell.
  Owen & Saigal's cleanup (smoothing within three edges, T23/T32/T44 flips) was not built; two Sozzi
  draws still left one quad unmergeable.
- Outcome (project owner, 2026-10-21): `ClearedCaps()` made the default cap rule; `MergedCaps` dropped,
  kept on the pushed branch `claude/merged-caps-experiment` (commit 60c3e19f) with its tests and the
  harness arm, re-runnable.

## 2026-10-21 — RBF morphing: where it would fit (deferred)

Discussed with the project owner and deferred until the mesher is finished. Radial basis function (RBF)
morphing moves every point by a smooth field interpolated from control points' displacements, one
linear solve for the weights, topology unchanged (de Boer, van der Schoot & Bijl 2007; Jakobsson &
Amoignon 2007 for shape optimization; Rendall & Allen 2008-2009 for greedy control-point selection --
cited from memory, not re-read). Ranked for this mesher:
1. **The differentiable map from design parameters to `node_coords`** (lamp position or radius, baffle
   angle, reactor length): boundary vertices moved by re-projection or analytically, the rest by the
   field, linear once the centres are fixed, so `jax.grad` of a solved objective reaches the design
   parameter with no remeshing and no derivative of Netgen.
2. **Design variants from one base mesh**: Netgen's fill moves with any change (the five-draw
   ensembles), so meshing each variant mixes design effect with mesh noise; morphing keeps the
   topology. Same machinery as 1.
3. **The projection's displacement spread into the layers** (`_onto_the_drawing` moves only the
   boundary; the lamp's first dual layer 0.5033 mm for 0.5 wanted): small, local.
4. **The pipe-to-chamber saddle**: an O-grid morphed onto the saddle could reach the junction instead of
   stopping two rows short.
5. **Snapping the octree core to walls**: least, since conformity is topological -- snapped points
   still leave the core's faces unmatched to the layers' tops.
Costs: a dense solve on every boundary vertex is out (Sozzi ~1e5); greedy selection to a few thousand
centres or compactly supported (Wendland) functions with a sparse solve; evaluating 2M points is one
batched product; large motions invert cells, so a quality guard (`MeshQuality`) bounds it. For 1-2 it is
JAX and solver-side, so a generic module (likely `aquaflux/mesh/`), not `meshing/`.

## 2026-10-21 — Pipe rims divided alike (#789)

macOS arm64, netgen-mesher 6.2.2608 from a scratch venv (`NetgenProcess(python=)`), one run per arm,
shared machine (another session meshing), so seconds are rough. Configuration:
`validation/meshing_dual/pipe_rims_sozzi.py` -- `mapped_layers_sozzi.py`'s settings (layers 0.5 mm x 1.2
x 5 on every wall, the lamp mapped, `pipes` unset so every pipe found is cored, no core, no
planarizer). "Cored" read from the mesh: every cell within 2 mm of the pipe's axis, away from
its junction, a hexahedron. Before = `main` at ddfeb518; after = this change on ddfeb518 -- both
under the caps of that commit (`PyramidCaps`, height 0.25 of the first edge), before #792 made the
room-sized caps the default; cell counts and smallest volumes are of that cap rule.

| maxh | Netgen's rims (riser; outlet) | before: cored, cells, closure | after: cored, cells, closure |
|---|---|---|---|
| 0.02 | 12/12; 12/12 | 2 of 2, 49494, 4.2e-16 | 2 of 2, 49494, 4.2e-16 (nothing asked) |
| 0.01 | 12/12; 12/12 | 2 of 2, 61903, 7.1e-16 | 2 of 2, 61903, 7.1e-16 (nothing asked) |
| 0.005 | 12/13; 12/12 | 1 of 2 (riser tets), 131093, 7.2e-16 | 2 of 2 (rims 16), 145676, 7.2e-16 |
| 0.0035 | 17/17; 17/17 | 0 of 2, 266725, 8.8e-16 | 2 of 2 (rims 20), 292145, 6.9e-16 |
| 0.0025 | 24/24; 24/24 | fill fails (#790) | fill fails (#790); nothing asked, so as before |

Closure is the worst cell's residual over the largest face area. At 0.0025 the rims already take an
O-grid (`find_pipes` accepts both), so the cores there wait only on #790. Smallest cell volume 0.0035:
2.41e-11 before, 7.88e-12 after (the junction discs' cap apexes, #785, now present on both pipes); 0.005
3.75e-11 → 1.87e-11. Cost: one more surface triangulation when something is asked (the Netgen test's
two triangulations at 0.0035 plus reading the drawing take ~11 s).

How Netgen is told (probed in this session, `NetgenProcess` child): per-edge `maxh` is a bound, but
netgen.occ's `Edge.partition` sets the division exactly -- the interior points' parameters scaled to
[0, 1] (raw parameters failed five faces' surface meshing), read only when an `OCCGeometry` is made of
the shape (set on the geometry's own shape and meshed, it did nothing). Edge numbering: `shape.edges`
gave 29 entries for Sozzi's 15 edges (once per face); the distinct ones in first-met order matched
`CadModel.topology`'s all 15 by length and end vertices. Asked 16 per rim at 0.005 and 0.0035 and 0.02,
Netgen gave exactly 16/16 on both pipes. Rim spacing at 0.0035 with 20 segments: max/min step 1.0002 on
the riser's saddle placed by arc length, 1.0039 by equal parameter steps (circles 1 + 4e-14 either way).

Rounding rule: `4·ceil(max/4)`, not nearest, so a rim is never coarser than Netgen chose (13 → 16, 17 →
20). Mutations of the unit tests, each red: no rounding, `round` for `ceil`, equal share across a rim's
edges, no second triangulation, asking when the rims already fit; of the Netgen test: no second
`OCCGeometry`, no dedup of the edges, equal parameter steps for arc length.

## 2026-10-21 — the fill's overlap at maxh 0.0025 (#790): a collapsed layer top under a cap

Sozzi water as `memory_by_stage.py` (lamp and pipes mapped, five layers 0.5 mm × 1.2 on every wall,
`feature_angle` 0.785), maxh 0.0025, macOS arm64, netgen-mesher 6.2.2608, main at `ddfeb518`.

**Which triangles.** The fill's surface (layers' top, bare faces, cap sides; dumped just before
`_filled`) has 192222 triangles; a brute edge-through-triangle check finds 52 crossing pairs, all in
one ring: 17 triangles of the vessel's end-wall layer top against the sides of 9 pyramid caps on the
lamp's layer-top quadrilaterals, where the lamp (r = 10 mm) leaves the end wall (x = 0). Netgen reports
only the first pair, which is why `pipes=()` "failed on the same pair".

**What made them.** The water's angle at that junction is 90°, so the corner column moves along the
bisector (stretch √2): its top at x = 2.476 mm, while the lamp's next column (x = 2.508 mm on the wall)
moves only radially. The lamp's first layer-top quadrilateral is 0.032 mm wide axially against its wall
face's 2.508 mm (edge ratio 0.0126). By the shortest top edge over the wall edge under it, 23 layered
quadrilaterals fall under 0.05 (all under 0.1), every one in that ring, and no layered triangle under 0.1. Its cap
is sized by the first edge, which runs round (3.05 mm), so it rises 0.76 mm -- through the end wall's
layer top, which is tapered (0.5 of full thickness at the corner ring, 0.75 one ring out) and so leans
over the lamp's layers by about 0.45 mm per mm. The thickness at the corner (2.26-2.48 mm, already
halved once against inversion) has reached the wall's spacing (2.5 mm): at coarser maxh the spacing is
wider and the top face is not a sliver. With triangle walls the same slivers are prisms and nothing is
capped on them, so that arm meshed.

**Under `ClearedCaps`** (PR #792's default, branch `claude/pyramid-transitions` at its tip, re-extruded
offline from the dumped walls): no crossing, because the fold limit across the sliver's 16 µm
half-width shrinks its cap to **10 µm** -- the small-apex-cell problem (#785) at its worst.

**The fix: thin where the top collapses.** `PrismLayers.min_top_share` (0.1): an edge of the layers'
top shorter than that share of the wall edge under it halves the thickness at its two points, as an
inverted cell does. Offline on the dumped walls (same extrusion, then caps, then the crossing check):

| cap rule | `min_top_share` | crossing pairs | smallest cap |
|---|---|---|---|
| fixed 0.25 | 0 (before) | 52 | 0.33 mm |
| fixed 0.25 | 0.1 / 0.25 / 0.5 | 0 / 0 / 0 | 0.33 mm |
| `ClearedCaps()` | 0 | 0 | 0.010 mm |
| `ClearedCaps()` | 0.1 / 0.25 | 0 / 0 | 0.24 mm |

Columns left thinner than before: 222 of 71732 at 0.1 (the junction's ring halved once, and the taper
round it), 294 at 0.25, 644 at 0.5. The minimum thickness anywhere is unchanged at
0.1-0.25 (1.13 mm, a gap limit elsewhere). 0.1 was chosen under the narrowing a pipe's layers make
uniformly when they take `gap_share` of it (`1 − 2·0.4` = 0.2), so a narrow pipe is never thinned for
it. Reproduced small: a cube of 8 × 8 quadrilaterals per face, one layer 0.1225 thick -- Netgen stops on
"boundary mesh is overlapping" without the floor and fills it (closure 1e-17) with it.

**Whole runs after the fix** (fixed caps unless said; timings beside another session's work, counts good):

| maxh, arm | cells | closure (of the largest face area) | smallest cell, m³ | peak footprint GiB |
|---|---|---|---|---|
| 0.0025, pipe cores | 671030 | 5.0e-16 | 4.70e-12 | 3.21 |
| 0.0025, no pipe cores | 661019 | 5.1e-16 | 3.93e-11 | 3.25 |
| 0.0025, `ClearedCaps()` (#792's tip with the floor) | 666826 | 5.3e-16 | 1.16e-11 | 3.19 |
| 0.0025, `ClearedCaps()` without the floor (#792 alone) | 666876 | 5.2e-16 | 4.54e-13 | 3.18 |
| 0.0025, triangle walls, no pipe cores | 659889 | 6.6e-16 | 1.94e-10 | 3.35 |
| 0.002, pipe cores | 1001489 | 7.4e-16 | 1.49e-11 | 4.80 |

So #792 alone also gets past the fill at 0.0025, but by the 10 µm caps: its smallest cell is 25× smaller than
with the floor. The triangle-wall arm was 659651 cells on main after #786; not a controlled pair (main
has moved since). Netgen's child peaked at 0.86 GiB RSS at 0.0025 and 1.85 at 0.002. Harnesses kept in
the repository: `memory_by_stage.py` (now prints closure and the smallest cell); the surface dump and the
crossing check were scratch, and the cube tests reproduce what they showed.
Rebased onto main after #792 (`ClearedCaps` the default) and #789 (`be11a08c`): maxh 0.0025 with pipe
cores gives 666826 cells, closure 5.3e-16, smallest cell 1.16e-11 m³, peak 3.17 GiB -- the "#792's tip
with the floor" row exactly.

## 2026-10-24 — Rectangular ducts: found, divided along normal cross-sections, cored

Harness `validation/meshing_netgen/ducts.py` (one run each, 4-core Linux container, netgen-mesher
6.2.2608, OCP 8.0.1, `CadMesher` defaults, `feature_angle` 45°; layers `PrismLayers(first, 1.2, 3)`,
first 2 mm, tank 3 mm). Tilt: each core ring's fitted plane against the line between its neighbours'
centroids, degrees.

| arm (maxh) | cells | closure | six-faced | tilt max / mean | non-orth max / p99 | concave |
|---|---|---|---|---|---|---|
| straight 0.2x0.1x1 m (0.025) | 1845 | 7.4e-16 | 1.000 | 0.0 / 0.0 | 1e-6 / 1e-6 | 0 |
| straight, layers | 4797 | 7.2e-16 | 0.932 | 0.0 / 0.0 | 16.4 / 16.4 | 0 |
| bent: leg, quarter bend R 0.4, leg (0.025) | 3240 | 7.2e-16 | 1.000 | 0.7 / 0.1 | 0.50 / 0.41 | 0 |
| bent, layers | 8424 | 7.8e-16 | 0.932 | 0.7 / 0.1 | 17.1 / 16.4 | 0 |
| spline sweep 0.1x0.06 (0.015) | 3640 | 5.6e-16 | 1.000 | 28.2 / 1.7 | 35.0 / 29.6 | 0 |
| sweep, layers | 9646 | 6.0e-16 | 0.925 | 34.2 / 2.3 | 45.8 / 32.5 | 0 |
| tank + inlet duct + lamp, layers (0.05) | 12320 | 5.6e-16 | 0.054 | 0.0 / 0.0 | 178.4 / 31.9 | 862 |
| same, `ducts=()` (the mesher before ducts) | 10916 | **4.6e-1** | 0.004 | -- | 178.7 / 33.1 | 1107 |

- **The rails' points**: at equal arc length on every rail (the first build) the sweep's rows tilted
  max 44 / mean 31.8°, non-orth max 45.4 bare and 65.3 layered; cross-sections normal to the centreline
  gave mean 25.9° because near its oblique ends (the profile is drawn in `z = 0`, the spline leaves the
  origin leaning, so corners fold 73-125°) a normal plane misses a rail and three of four rails fell
  back; turning the planes toward each rim's own plane within a width of it gave mean 1.7; a
  second-order end tangent (`np.gradient(edge_order=2)`) removed a first-order chord's false obliquity
  on the parabola unit test (2.75° → 0.10° at 16 segments; equal arc length 19.3°). The remaining max
  is at the oblique ends, by design.
- **The odd cells with layers** are at the layers' top's four corners: three primal cells meet along
  each, so per row one triangular prism and one seven-faced cell beside it (straight: 164 + 164).
- **The tank**: its worst faces are at the lamp's ends and the tank's three-wall corners (the deferred
  convex-ridge normals), in both arms. Without layers both arms fail in the dual's concave-edge cut
  (`NotImplementedError`: cells deepest behind one wall not connected) where the duct's rectangular
  mouth meets the flat wall: the dual's limit, not the duct's. A box tank with nothing inside is itself
  found as a duct (around the axis whose walls are four-sided), cored with a junction at the wall the
  inlet duct opens through; the lamp arm keeps it out.
- **Dismissed mutations** (`test_meshing_ducts.py`): `DuctCore`'s blend weights (zeroing `u`) stay green
  -- with straight sides spaced alike, which every duct cross-section ring has, transfinite interpolation
  reproduces the lattice whatever its weights. The closed-chain refusal ("closes on itself") and the
  end-face exclusion in `_hollow` have no stub test: the first needs a ring-shaped duct drawing, the
  second a non-cap end face whose points sit on the closing fan (the tank's wall with the duct's mouth).

## 2026-10-24 — Refinement regions: Netgen's local sizes

Configuration: Linux x86_64 container, Python 3.13, netgen-mesher 6.2.2608, branch
`meshing-refinement-regions` on `main` at `4f6ec9f9`; one run each.

**The API, probed in a child-like process (`python -P`, netgen only).** A unit box surface of 4 x 4
squares per face, filled to `maxh` 0.25 with h = 0.03 asked on a lattice in `[0.4, 0.6]^3`: median tet
edge there 0.505 with no request, **0.505 with `MeshingParameters.RestrictH` on the fill's parameters
(ignored)**, 0.036 with `mesh.CalcLocalH(0.3, 1)` then `mesh.RestrictLocalH(Pnt, h)` per point
(elsewhere 0.065, grading out); `CalcLocalH` alone left the mesh as it was (168 points, 604 tets). The
supplied surface was kept in every arm. Surface meshing of an OCC box (`maxh` 0.25, points on z = 0 in
`[0.4, 0.6]^2`): `RestrictH` on the `MeshingParameters` passed as `mp=` gives 0.035 near against 0.255.
`GenerateMesh(mp=MeshingParameters(maxh, minh, grading))` is bit-identical to the keyword form in a
fresh process (box less a cylinder, `maxh` 0.2: 1762 triangles, points, triangles and face indices
equal); the keyword form is not even repeatable within one process (1762 then 1742).

**Through the mesher's own path** (`validation/meshing_netgen/local_sizes.py 0.04`: a
`RefinedSphere`'s local sizes): fill of the unit box at 0.25 with a ball of 0.15 at its centre at 0.04
(437 points) -- median tet edge by distance from the centre [0, 0.15) 0.050, [0.15, 0.25) 0.054,
[0.25, 0.35) 0.075, [0.35, 0.5) 0.139, [0.5, 0.87) 0.250; without, 0.485 / 0.373 / 0.394 / 0.371 / 0.354
(277 tets against 7834). Triangles of a 1 x 2 x 3 box at 0.5 with a ball of 0.2 on its face x = 0 at
0.04: median edge near 0.057, away 0.415 (394 triangles against 180).

**Mutation pass** (each line broken, its tests rerun, restored; all red): the child's `RestrictLocalH`
and `RestrictH`; `refines` always true; the body's reach without the half-diagonal; local sizes
unfiltered; `RefinedNear` without its distance; the hex core dropping its refinements; `CadMesher`'s
core, fill and second triangulation without them; the early patch-name check; the max/min size
refusals; the empty-reach refusal; `MeshSpec.mesher()` not passing them; the unknown-patch refusal;
`LocalSizes`' positivity; and the L end-to-end test with `_located` returning none. None dismissed.

## 2026-10-24 — baffles: sheets standing in the region

**Decided with the project owner (2026-10-24):** a baffle is two coincident boundary faces per place in
an ordinary patch, not an interior-face patch with its own flux (`docs/mesh_zones_and_patches.md` had
promised the latter; nothing implemented it, and every BC, the wall distance, the wall treatment, the
dual and the polyMesh writer act on boundary faces only). Baffles come from the drawing's sheets.

**Probes (Linux x86-64, netgen-mesher 6.2.2608, OCP 8.0.1.0.0; scratch scripts, the findings carried
into `validation/meshing_netgen/baffle_quality.py` and the tests):**
- `netgen.occ` `Glue([box, sheet])`: one solid, the sheet a face with `domin = domout = 1`; the surface
  mesh carries it and the volume mesh conforms (every sheet triangle a face of two tets; none crossing).
  A bare `Mesh` given the outer triangles plus the sheet's under a `domin = domout = 1` descriptor fills
  the same way (unit box, sheet 0.6 x 0.6 standing on a wall, maxh 0.15: volume 1 to 12 digits).
- OCP `BRepAlgoAPI_Splitter(box | plate)`: one solid; the plate a face oriented `INTERNAL` in a shell of
  its own; topology 7 faces, the plate's three free edges bounding it alone, its foot an internal edge
  of the wall face (faces `[wall, plate]`). Netgen's triangulation of `write_brep`'s file numbers all 7
  faces as the explorer does (every triangle's centroid projects onto its own face), and the wall's
  triangles carry the foot as edges.
- A STEP file with a box and a sheet: `read_step` dropped the sheet (solids only) before this change.
- **The opened surface itself (two coincident sides on their own points) does not fill**: the child sat at
  100 % CPU, no result after 120 s, box with plate at maxh 0.1 (782 points); the internal form takes about
  a second. Harness: `baffle_quality.py 0.1 --opened-fill`.

**Quality** (`baffle_quality.py 0.1`, 2026-10-24, defaults in force: `ClearedCaps`, no core, no
planarizer; feature angle 45 degrees; layers `first_height = maxh / 20`, growth 1.3, 3 layers on
`walls`; one run each, Netgen's fill deterministic here):

| mesh | cells | non-convexity max / p99 | non-orth. max | > 50 | > 90 | faces < 1 % median |
|---|---|---|---|---|---|---|
| box | 1304 | 0.312 / 0.170 | 49.5 | 0 | 0 | 0 |
| box and plate | 1334 | 0.598 / 0.203 | 46.3 | 0 | 0 | 3 |
| L | 3771 | 0.956 / 0.280 | 74.9 | 57 | 0 | 114 |
| box + layers | 3324 | 0.432 / 0.355 | 41.5 | 0 | 0 | 0 |
| box and plate + layers | 3339 | 1.130 / 0.365 | 124.9 | 2 | 1 | 4 |
| L + layers | 8219 | 1.887 / 0.349 | 179.7 | 9 | 8 | 5 |

The worst cells with the plate are ordinary interior points about a cell above or below a free edge;
their worst vertex lies beyond a small quadrilateral (about 0.004 across at maxh 0.1) between them and a
free-edge piece, where the piece's interface (the sheet's plane extended) passes close to a cell
centroid. The same cut makes the L's worse faces, so this is the concave-edge cut's trait under layers,
not the baffle's; not pursued here. What it does not distinguish: whether a different piece rule would
help both.

**Two defects found on the way, fixed:** a column point sliding along a free edge was cut twice -- as
a concave-edge point of its own and as a point above its base -- which showed as the dual's edge faces
losing count of their runs (a `zip()` length error, layered plate only); and the free edge read as
convex (coplanar sides: zero rise), so its points were not cut.

## 2026-10-24 — layers fanned round a baffle's free edge

**Decided with the project owner (2026-10-24):** layers on a baffle fan round its free edge (a column
per side and one between, ahead of the edge, the cells between wrapping it), over in-plane columns. A
prototype of in-plane columns (each free-edge point's column turned into the sheet's plane) gave flat
prisms that the extruder could only accept under a relaxed inverted-cell test, and then 161 faces over
50 degrees and 22 over 90 on the plate in a box at maxh 0.1.

**Steps that did not work, each measured on that plate (maxh 0.1, layers 0.005 x 1.3 x 3 on walls and
plate):**
- Landing cells as two pyramids over the two triangles of no area where the fan closes onto the wall:
  merged, the two share two faces (the landing edge runs through both), and the dual made a face of two
  vertices. One six-cornered tent instead.
- The middle column at a landing set from the folded faces alone: it lay along the landing edge, and
  the landing cells came out flat or inverted (6 points still inside out after 8 halvings). Bisecting
  the sides' columns, tipped along the landing edge, fixed it.
- The fan's columns cut by the dual with their base point (as any layered column at a concave edge):
  498 faces over 50 degrees, 55 over 90. Not cut: 13 and 4 -- the 4 at 180 degrees at the sheet's two
  corners, where the corner's one middle column had every cell around it reaching back 0.05 along the
  free edges, so the cut between the corner point's pieces was put in the sheet's own plane, behind the
  corner. Moving that cut's hub along its edge broke closure (a cell inside out); a third piece for the
  fan, by depth or by the cells round the middle column, was worse (76 over 50, closure broken).
- A middle column per edge at a corner joined by one quadrilateral of no area: its pyramid is flat (the
  sides' columns grow straight off the sheet, so the corner point lies between their tops). Two
  triangles split between the middles give two tetrahedra; their middles first leaned off the corner's
  faces, whose centroids lie diagonally behind it, which aimed each ahead of the other edge (inside out).
  Square off its own edge: 9 over 50, 0 over 90.
- Cutting the sides' columns again at the landings only (they stand in the concave corner of sheet and
  wall): 22-30 over 50 and 8-11 over 90 at maxh 0.05-0.13, against 8-9 and 0-1. Reverted.

**Result** (`baffle_quality.py`, 2026-10-24, defaults in force: `ClearedCaps`, no core, no planarizer,
feature angle 45 degrees, layers `first_height = maxh / 20`, growth 1.3, 3; Linux x86-64, netgen-mesher
6.2.2608, one run each):

| maxh | mesh | cells | non-convexity max / p99 | non-orth. max | > 50 | > 90 | faces < 1 % median |
|---|---|---|---|---|---|---|---|
| 0.1 | box and plate + layers on walls | 3339 | 1.130 / 0.365 | 124.9 | 2 | 1 | 4 |
| 0.1 | box, plate + layers on both | 3838 | 1.343 / 0.380 | 74.4 | 9 | 0 | 2 |
| 0.05 | box and plate + layers on walls | 16580 | 0.491 / 0.338 | 43.6 | 0 | 0 | 3 |
| 0.05 | box, plate + layers on both | 18100 | 1.154 / 0.340 | 71.6 | 9 | 0 | 38 |

At 0.07 (probe script, not the harness): 7768 cells, non-orth. max 73.4, 8 over 50, none over 90; at
0.13: 2119 cells, one face of 151 degrees, 9 over 50. Every face over 50 degrees and the worst cells lie
at the two points where the free edge lands on the wall: the tent and the floor's cells round it. Not
pursued: the landing's cells are the open item.
