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
