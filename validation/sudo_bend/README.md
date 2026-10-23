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
| straight pipe before the bend | 3 `D` (the experiment's was ~100 `D`: the flow enters fully developed, so the inlet admits a developed flow, below) |
| straight pipe after it | 15 `D` (measuring stations to 10 `D`) |
| fluid | air, `Re = 6.0e4` on the bulk velocity (about 8.7 m/s at ν = 1.5e-5 m²/s) |
| measuring stations | upstream of the bend, through it (0 to 90°), and downstream to 10 `D`; the exact list and the digitized profiles are still to be taken from the paper |

The bend lies in the plane `z = 0`, a plane of symmetry of the mean flow; the fluid enters along
`+x` and leaves along `+y` (`geometry.py`).

## The mesh

`geometry.py` draws the fluid as one solid (`bend.step`: the inlet and outlet discs, two cylinders and
the toroidal bend between them), and `mesh.yaml` meshes it with aquaflux's own mesher:

```
python validation/sudo_bend/geometry.py          # bend.step and precursor.step (tracked)
python validation/sudo_bend/write_grids.py       # grids/round{36,48,64}/*.yaml (tracked)
cd validation/sudo_bend/grids/round36
aquaflux mesh mesh.yaml                          # work/bend.aqmesh
python ../../render_mesh.py                      # figures/mesh_*.png
```

Each grid of the study (below) is a directory of case files under `grids/`, written by
`write_grids.py`; the mesh described here is the coarsest, `grids/round36`.

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

## The inflow

The experiment's inlet pipe is long enough for the flow to reach the bend fully developed, and the
3 `D` drawn here are not. The developed flow is computed instead in a straight length of the same
pipe whose two ends are one periodic seam, so its outflow is its inflow, held at the bulk velocity
by a force solved with the flow:

```
cd validation/sudo_bend/grids/round36
aquaflux mesh precursor_mesh.yaml   # work/precursor.aqmesh
aquaflux run precursor.yaml         # work/precursor/profiles.csv
aquaflux run case.yaml              # work/results
```

`precursor_mesh.yaml` meshes one diameter of the pipe with the bend's own settings, so its
cross-section is the bend inlet's (3 792 cells, 12 rows of 316), and joins its ends. The precursor's
fields, averaged over 40 bands of distance from the axis, are its profiles of axial velocity, `k`
and `omega` against the radius (`profiles.csv`), and the bend's inlet is a `DevelopedInlet` that
gives each face the values at its distance from the inlet's centre, along the inward normal.

Measured (2026-10-23, the tracked files, 4-core Linux container, one run):

| | |
|---|---|
| precursor march | 63 steps, 17.5 min, to 6.0e-10 (1.1e-8 of the start, row-scaled) |
| profiles | 21 radii (40 bands, the empty ones in the core's corners dropped) |
| centreline | 1.160 `U_b` (the one-seventh power law's 1.224) |
| first cell | y+ 40 by Blasius's friction velocity, 0.435 m/s; the friction velocity that puts it on the log law (κ 0.41, B 5.2) is 0.425 m/s, 2.4 % lower |

`plot_inflow.py` draws `figures/inflow_profiles.png`: the velocity against the power law, in wall
units against the log law, and `k` and `omega` against the radius.

## The solve

`case.yaml` marches the bend with bfs3d's settings unchanged (the field split of the materialized
Jacobian, the dual-time loop and the viscosity ramp from 100 times the fluid's viscosity in 12
stations). Measured (2026-10-23, the tracked files, `main` at `7fe27d07` merged, 4-core Linux
container with 15 GB, one run): converged in 15 steps and 42 min (2514 s) to 1.1e-7 in the
row-scaled measure, at least 4.3 GB of memory free throughout. The inlet admits 0.998 of
`U_b` times its area (the faces take the profile at their own radius, the table holds cell means).

`render_flow.py` draws `figures/flow_sections.png` (the axial velocity and the cross-flow at the
stations: the pair of Dean vortices, the core thrown to the outer wall by 90 degrees, slow fluid
gathered at the inner wall, the largest cross-flow 0.45 `U_b` at 60 degrees) and
`figures/flow_symmetry_plane.png`. Nothing is compared with the measurements yet.

## The grid study

The discretization error is estimated by the grid convergence index (GCI) of Celik et al. (2008),
*J. Fluids Eng.* 130, 078001, on three grids each refined by `r = h_coarse / h_fine` above 1.3 in
`h = (V / N)^(1/3)`. `write_grids.py` writes them, one directory each (`grids/round36`, `round48`,
`round64`: the points round the pipe), refined systematically: the rims' points (36, 48, 64; `r =
4/3` round the pipe, a multiple of four for the O-grid's core), and with them the core, the shell
and the rows through the bend; the straight pipes' graded rows (growth `1.08^(1/r^k)`, longest `(D /
5) / r^k`); and the wall layers above a first cell held at 2.5 mm on every grid, so the wall
functions see the same first cell (growth `1.2^(1/r^k)`, one layer more a level: 4, 5, 6). Each
grid's precursor is meshed with its settings, so every bend's inflow is developed on its own
cross-section.

Measured (2026-10-23, `write_grids.py` as tracked, 4-core Linux container):

| grid | cells | `h` ratio to the next coarser | inlet faces | non-orthogonality max | meshing |
|---|---|---|---|---|---|
| `round36` | 42 660 | | 316 | 13.4° | 13 s, 0.80 GiB |
| `round48` | 98 987 | 1.324 | 553 | 17.7° | 19 s, 0.97 GiB |
| `round64` | 221 102 | 1.307 | 929 | 21.5° | 30 s, 1.35 GiB |

The worst faces on every grid are at the corners of the O-grid's square core (r/R 0.43 to 0.50), the
same in the precursors: the layers' stack thickens with the added layer (13.4, 16.8, 20.4 mm), so the
shell between it and the core thins. Keeping four layers instead would give `r = 1.285`, under the
procedure's 1.3.

`gci.py` computes, for each quantity on the three grids, the apparent order, the extrapolated value
and the fine-grid GCI: the pressure-loss coefficient from 1 D before the bend to 10 D after it, the
strength of the secondary flow at 90 degrees, and the axial velocity along the plane of symmetry's
diameter at each station (the local GCI). It first reproduces the paper's own worked example (its
Table 1), and stops if it does not. A value at a point is read by a weighted linear least-squares
fit over the cells round it, taken from the mesh's connectivity (exact for a linear field): the
coarse grid solved twice, on meshes 0.1 µm apart, gives every quantity to 1e-7.

`round36` re-solved on its generated mesh (2026-10-23, one run): 15 steps, 2352 s, to 1.1e-7, its
fields within 2e-6 of the first solve's.

The precursors stop at 1e-6 of their starting residual (row-scaled). Measured from their histories
(2026-10-23, run to 1e-8): 1e-6 is reached at step 42 (610 s) on `round36`, 55 (2554 s) on `round48`
and 72 (3.1 h) on `round64`; the last was stopped at step 82 (2.4e-7) and wrote no profiles.

The bend's march holds the whole materialized Jacobian: the coarse grid used about 10.5 GB of this
container's 15, so the medium (about 20 GB) and the fine (about 45 GB) grids are solved on a larger
machine, from the same files.
