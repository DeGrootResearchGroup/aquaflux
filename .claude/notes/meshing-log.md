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

