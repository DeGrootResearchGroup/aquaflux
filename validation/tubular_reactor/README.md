# Tubular reactor: ten lamps along the axis, a perforated plate at the inlet

A radiation-only case: the ultraviolet fluence rate `G` in a closed-vessel reactor, gathered at the
nodes of a Cartesian background grid. It has no flow mesh.

## The reactor (`geometry.py`, `Reactor`)

| | |
|---|---|
| vessel | 300 mm inner diameter, 1100 mm long, axis along x |
| inlet | 100 mm pipe on the axis, 250 mm long, upstream end at x = -250 mm |
| perforated plate | x = 100-106 mm (6 mm thick), 100 holes of 21.21 mm on a 26 mm hexagonal lattice: **50.0 % open**, 4.8 mm between holes, 3.6 mm rim to the wall |
| lamps | 10, the Sozzi & Taghipour (2006) lamp: 20 mm sleeve, 800 mm from the end wall at x = 1100 mm to the centre of a hemispherical tip at x = 300 mm, **35.4 W** UV-C each |
| lamp layout | rings of 3 (radius 50 mm) and 7 (radius 110 mm); closest sleeve-to-sleeve gap 41 mm, sleeve to wall 30 mm |
| outlet | 100 mm side pipe at x = 1000 mm, along +y, open end at y = 400 mm |
| water | 70 % UVT over 1 cm, 35.67 /m napierian (the Sozzi case's) |
| walls | vessel, pipes, end walls and the plate (its faces and holes): **diffuse reflectance 0.1** |

The lamps emit diffusely from the sleeve's outer surface. The quartz and the air inside the sleeve are
not modelled, and the lamps absorb whatever lands on them. Light leaving through the pipes' open ends is
lost.

## How it is computed (`run.py`)

- **Shadows by exact bodies, no tessellation.** The water is the union of three cylinders (the chamber
  and both pipes, an `Outside`). The plate is a `PerforatedPlate` body: a sight line crossing the slab
  is clear only if the part of it inside the slab lies in a single hole. That is one hex-lattice lookup
  per line, whatever the number of holes, so there is no 2^100 CSG difference. It agrees with
  brute-force sampling, including a line grazing a hole edge by 7 µm. Each lamp is a cylinder with a
  hemispherical tip, 0.25 mm inside the sleeve so it holds none of the emitting triangles' centroids.
  The lamps' shadows are therefore 0.5 mm narrower than the 20 mm sleeves.
- **Triangles only where something emits.** Each lamp has 24 sectors, with every vertex on the true
  surface and exitance scaled so it radiates exactly 35.4 W. The walls the water touches come from
  gmsh's OpenCASCADE kernel (about 11,700 triangles at a 30 mm edge) and reflect.
- **Steps.**
  1. The lamps' light on the walls is averaged over four points per wall triangle. It uses a lamp
     surface 40 mm along the axis, since no wall is closer than 30 mm to a lamp.
  2. One radiosity solve among the wall triangles. Its shadows come from the same bodies.
  3. `G` at every grid node in the water, at least 0.3 mm from a wall or sleeve. It is the lamps' light
     from a lamp surface 8 mm along the axis (50,640 facets), plus the walls' reflected light.

```bash
validation/run_case.sh validation/tubular_reactor/run.py
```

Settings are environment variables, listed in `run.py`'s docstring:

- `TUBULAR_H`: grid spacing (default 0.006 m);
- `TUBULAR_AXIAL`: lamp triangle length (0.008 m);
- `TUBULAR_SECTORS`: lamp triangles round the sleeve (24);
- `TUBULAR_WALL`: wall triangle edge (0.03 m);
- `TUBULAR_PER_TURN`: sides per plate hole (8);
- `TUBULAR_SHRINK`: how far inside the sleeve the lamp's shadow body sits;
- `TUBULAR_OUT`: output directory;
- `TUBULAR_FIGURES_ONLY` and `TUBULAR_REPLOT`: redraw a finished run's figures from its saved data.

`study.sh` runs the resolution study one case at a time, and `compare.py` compares the runs.

Outputs, in `work/runs/<tag>/` (gitignored; the tag names every setting):

- `fluence_rate.vti`: VTK image data with `G`, `G_direct` and `G_reflected`, NaN outside the water;
- `walls.vtp`: wall triangles with their irradiance `E`, the lamps' share `E_from_lamps`, and `radiosity`;
- `lamps.vtp`;
- `reactor_water.step`: the water as a STEP solid, for meshing a flow later;
- `fluence_rate.npz`;
- `summary.json`: dimensions, settings, counts and the power accounting;
- figures, each drawn up to the true boundary (`render.py`):
  - `slices.png`: along and across the axis;
  - `slices_upstream.png`: behind the plate;
  - `plate_shadow.png` and `plate_beams.png`: fine planes either side of the plate, lamps' direct
    light only, with their data in `plate_shadow.npz`;
  - `lamp_tessellation.png` and `wall_tessellation.png`.

## Result at an 8 mm grid (2026-10-15)

Configuration: `TUBULAR_H=0.008`, with every other setting at its default. 151,747 water nodes, 50,640
lamp facets for the volume, 12,240 for lighting the walls, and 11,656 wall triangles (5,284 vessel and
6,372 plate). Run alone through `run_case.sh` on macOS arm64 (11 cores), jax 0.10.2, at `1937b3bb` plus
these files.

- Volume-mean `G` is **112.9 W/m²**: 127.6 W/m² downstream of the plate and 8e-4 W/m² upstream of it.
  The upstream water is 190 mm or more from the lamp tips, behind the plate, at 35.7 /m.
- `G` percentiles 1 / 10 / 50 / 90 / 99: 1.9e-7, 1.3e-3, 75, 276 and 762 W/m².
- Of the 354 W the lamps emit, 22.4 W reaches the walls, and the walls absorb 20.2 W of it (the
  plate 3e-4 W). The water absorbs about 313 W (the grid's sum of `a G h^3`, a quadrature estimate).
  The remaining ~21 W is the lamps' share, what escapes through the pipe ends, and that estimate's own
  error.
- Reflected light is 0.6 % of the mean `G`.
- The radiosity solve took 3 restart cycles.
- Times: 150 s lighting the walls, 164 s for the radiosity solve, 479 s for direct `G` and 213 s for
  reflected `G`. 1,008 s in all.

## Resolution study (2026-10-15/16, `study.sh`, `compare.py`)

Every run is on the baseline (8 mm grid, 8 mm × 24-sector lamp, 30 mm walls, 8-sided holes) except
for the one setting varied. Each run's data and figures are in `work/runs/<tag>/`, and the
comparison is in `work/study/` (`report.md`, `runs.csv`, `pointwise.csv`, two figures). The
configuration is as in the 8 mm result above. One case ran at a time through `run_case.sh`, with
nothing else heavy running.

How each study is measured:

- **Grid**: G at a node does not depend on the other nodes; shared nodes agree to 7e-13 W/m². So
  the error measured is how well a grid represents the field between nodes: each run interpolated
  (trilinear in log G, dry corners dropped) to every node of the 3 mm run.
- **Tessellation**: runs on one grid have identical nodes, and each is compared with the finest run
  of its kind.
- **Bands**: median |ΔG|/G over lit nodes (G > 1e-3 of the peak), by distance from the nearest sleeve.

| study | run vs reference | all | 0–2 mm | 2–5 mm | 10–30 mm |
|---|---|---|---|---|---|
| grid | 8 mm vs 3 mm | 0.95 % | 36 % | 18 % | 0.81 % |
| grid | 6 mm vs 3 mm | 0.41 % | 26 % | 2.9 % | 0.42 % |
| grid | 4 mm vs 3 mm | 0.23 % | 17 % | 1.6 % | 0.23 % |
| lamp rings | 16 mm vs 2 mm | 0.42 % | 5.4 % | 1.2 % | 0.47 % |
| lamp rings | 8 mm vs 2 mm | 0.10 % | 1.1 % | 0.69 % | 0.11 % |
| lamp rings | 4 mm vs 2 mm | 0.018 % | 0.22 % | 0.11 % | 0.019 % |
| lamp sectors | 24 vs 48 | 0.24 % | 0.68 % | 0.31 % | 0.24 % |
| wall edges | 30 mm vs 20 mm | 0.006 % | — | — | — |
| plate holes | 8 vs 12 sides | 0.003 % | — | — | — |

- **The grid's error is the dominant one, and it lives beside the sleeves.** Away from them it
  falls as h² (0.81 / 0.42 / 0.23 % at 8 / 6 / 4 mm). Within 2 mm of a sleeve it is 17-36 % at every
  spacing tried, which is what issue #726 (adaptive refinement) is for.
- **The node-averaged volume mean G is not a converged volume integral.** It reads 112.9, 114.8,
  115.1 and 114.1 W/m² at 8, 6, 4 and 3 mm. It moves with how many nodes land close to a sleeve,
  not monotonically; quote it as about 114 ± 1 W/m².
- **The baseline lamp is adequate.** It is within 0.1 % in the bulk and 1.1 % within 2 mm of a
  sleeve against 2 mm rings. Mean G is 112.9 against 113.3 W/m² for 2 mm rings, and 113.3 for 48
  sectors.
- **The walls and holes barely matter to G, because reflection is 0.6 % of it.** The reflected part
  itself does change: 60 / 30 / 20 mm walls give a reflected share of 0.579 / 0.605 / 0.614 % of
  mean G. Against the 20 mm walls, the reflected part's median difference is 10 % for 60 mm walls and
  2.1 % for 30 mm.
- **Plate transmission does not depend on any setting.** Mean G 3 mm behind the plate over mean G
  there without it is 0.470 in every run.
- **Times**: grid 8 / 6 / 4 / 3 mm took 17 / 32 / 96 / 216 min (151,747 / 360,015 / 1,222,276 /
  2,894,307 nodes). Lamp rings 16 / 8 / 4 / 2 mm took 13 / 17 / 24 / 41 min (26,640 / 50,640 /
  98,640 / 195,600 facets). Each time includes about 15 min of plate planes, which scale with the
  lamp facets.

## Limits

- **Near a sleeve the field is only as good as the grid.** The nearest node may be up to one spacing
  from a sleeve. The Sozzi background-grid study found that interpolating such a grid to a flow mesh
  needs the sleeve's own value (`G = 2M`) as data within a few mm of it.
- The lamp triangles are 8 mm long. On the Sozzi lamp, triangles of about that length were wrong by
  about 1-2 % median within 5 mm of the lamp, and under 0.5 % elsewhere.
- The walls are triangulated, so a hole in the reflecting plate is an 8-sided polygon. The shadows use
  the exact circles.
