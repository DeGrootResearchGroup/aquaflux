# 90° pipe bend — Sudo, Sumida & Hibara (1998)

A turbulent flow through a circular pipe bent through 90°, against the measurements of
Sudo, K., Sumida, M. & Hibara, H. (1998), *Experimental investigation on turbulent flow in a
circular-sectioned 90-degree bend*, Experiments in Fluids 25, 42–49.

## Why

The step cases (`pitzdaily_openfoam`, `bfs3d_openfoam`) are flows that separate at a sharp edge. This
one does not: the bend turns the flow, and the pressure gradient across it drives slow fluid near the
wall inward round both sides to the inner wall — a pair of counter-rotating Dean vortices — while the
core is thrown toward the outer wall. Streamline curvature, a favourable then adverse pressure
gradient along the inner wall, and a secondary flow that persists for many diameters downstream are
the physics under test. Every term of the momentum equations is three-dimensional and none vanishes,
yet the geometry is simple enough to be meshed natively and structured throughout.

## The case

| | |
|---|---|
| pipe diameter `D` | 104 mm |
| bend radius `R_c` | 2 `D` (208 mm, centreline) |
| straight pipe before the bend | 3 `D` (the experiment's was ~100 `D`: the flow enters fully developed) |
| straight pipe after it | 15 `D` (measuring stations to 10 `D`) |
| fluid | air, `Re = 6.0e4` on the bulk velocity (about 8.7 m/s at ν = 1.5e-5 m²/s) |
| measuring stations | upstream of the bend, through it (0 to 90°), and downstream to 10 `D`; the exact list and the digitized profiles are still to be taken from the paper |

The bend lies in the plane `z = 0`, a plane of symmetry of the mean flow; the fluid enters along
`+x` and leaves along `+y` (`geometry.py`).

## The mesh

`geometry.py` draws the fluid as one solid (`bend.step`: the inlet and outlet discs, two cylinders and
the toroidal bend between them), and `mesh.yaml` meshes it with aquaflux's own mesher:

```
python validation/sudo_bend/geometry.py          # bend.step (tracked)
aquaflux mesh validation/sudo_bend/mesh.yaml     # work/bend.aqmesh
python validation/sudo_bend/render_mesh.py       # figures/mesh_*.png
```

All three walls are found as pipes joined end to end, so a single structured O-grid runs from the
inlet to the outlet: each cross-section is a 9 × 9 square core inside a shell and four wall layers,
36 cells round, and the bend's rows are its cross-sections turned about the bend's axis. No
tetrahedra are used. Along the two straight pipes the rows are graded (`pipe_grading`): as long as the
cells round the pipe are wide where they join the bend (about 9 mm), growing 8 % a row toward the inlet
and the outlet, to at most `D / 5`.

Measured (2026-10-23, `mesh.yaml` as tracked, 4-core Linux container):

| | |
|---|---|
| cells | 42 660, of which 98.6 % hexahedra (the rest the triangular prisms along the core's four corners) |
| rows | the bend's 36 (2.5° each); the straight pipes' graded from 9 mm at the bend to 20.8 mm |
| first cell | 2.45 mm thick (centroid at 1.22 mm, median), `y+ ≈ 35` in the straight pipe |
| non-orthogonality (interior faces) | mean 0.9°, 99th percentile 12.6°, max 13.4° |
| skewness | max 0.21 |
| concave cells | 0 |
| meshing time | 11 s |

The ungraded mesh (rows 9 mm throughout) has 76 788 cells and the same quality. A resolution study
round the pipe is still to be made.

The figures are `figures/mesh_overview.png` (the whole pipe), `mesh_bend_wall.png` (the bend's
wall grid), `mesh_inlet.png` (the inlet face, the O-grid the run is swept from), `mesh_sections.png`
(cross-sections at the stations), `mesh_section_wall.png` (the wall layers and shell),
`mesh_symmetry_plane.png` / `mesh_symmetry_plane_inner_wall.png` (the plane `z = 0` through the bend)
and `mesh_quality.png` (face non-orthogonality and skewness).
