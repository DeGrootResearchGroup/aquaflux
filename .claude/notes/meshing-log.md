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
surface step and keeps the 72 edge segments. But an edge's count cannot be set: per-edge `maxh = L/n`
is an upper bound only (asking a cylinder's rims for 20, 37, 50 segments gave 31, 37, 49). So the mapped
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
