# Ray effects: a far-UVC lamp, a Voronoi bunny, and the floor's irradiance

A demonstration that aquaflux's radiation gather has no ray effect, against the discrete-ordinates
method (DOM) of [of-optical-radiation](https://github.com/DeGrootResearchGroup/of-optical-radiation).

**The scene.** A 4 x 4 x 3 m room of air at 222 nm — transparent and non-scattering, so nothing
masks the angular discretization — with an Ushio Care222 B1 module on the ceiling centre facing
down, and a perforated "Voronoi bunny" (6.5 cm cells, 8 mm struts, 8 mm shell) hanging with its
base at 1.2 m. Every surface is black. DOM carries radiance along a finite set of directions, so
a bright source in a large clear room is its worst case: the light leaves the lamp in discrete
beams and each beam smears sideways as it is transported. aquaflux integrates the lamp's facets
exactly at each receiver, and the reference integrates the lamp window by brute force.

**The lamp** is the module's OSLUV-measured photometry, `ushio_b1.ies` (from
[guv-calcs](https://github.com/jvbelenky/guv-calcs), MIT licence, see `ushio_b1.LICENSE`): 121 mW/sr
on the axis, a 32.5-degree half-intensity angle, up to 8 % of the peak of variation round the axis,
and **118.83 mW** of radiant flux, the exact integral of its table, used as the lamp's power in
every solver. It emits from a patch of the ceiling mesh — 20 x 14 faces of 3.125 mm, 6.25 x
4.375 cm, the long side along x (the fixture's h = 0 direction) — against the module's measured
6 x 4.5 cm opening. The window is that large deliberately (the real module), so the shadow's edges
are soft: a strut's shadow is 13-20 mm wide and the penumbra 3-7 cm.

## Pipeline

| step | script | output (under `work/`) |
|---|---|---|
| mesh both rooms; cell centres and volumes; run DOM | `generate_dom.py` | `bunny/case`, `empty/case`, `<mesh>/cells.npz`, `runs/<mesh>_nphi<P>_ntheta<T>/` (`G`, `qin`, log, `record.json`) |
| extract the patches | `patches.py` | `<mesh>/patches.npz` |
| aquaflux, every case file in `cases/` | `run_cases.py` (`aquaflux run` on each, in turn) | `cases/<case>/` (`patches.vtm`, `fields.vtu`, `case.yaml`, `run.yaml`) |
| aquaflux, the two slices with the lamp split fine | `slices_refined.py` | `cases/<mesh>_volume/slices_refined.npz`, `.json` |
| reference, floor and two slices | `reference.py` | `reference/<mesh>.npz`, `.json` |
| figures and table | `figures.py` | `figures/` |

**Two quantities are compared.** On the floor, the **irradiance** `E` at each floor face: DOM's is
its incident flux `qin` on the floor patch, the new of-optical-radiation output (the exact boundary
term of each ray's transport, so its patch integrals balance the lamp's power); aquaflux's and the
reference's are computed at the face centres. In the air, the **fluence rate** `G` at each cell
centre, which DOM writes natively: aquaflux computes it at every cell, and the reference -- too
costly for 2.5M cells -- on the cells of two planes (`room.SLICES`): a vertical one through the lamp
and the bunny's middle (`y = 0.003 m`), and a horizontal one between the bunny and the floor
(`z = 0.603 m`), each cutting exactly one layer of cells (their squares tile the plane to the
bunny's own cross-section).

**aquaflux runs from case files**, one per configuration, each an ordinary `aquaflux run` case:

| case file | room | what it computes |
|---|---|---|
| `bunny_floor.yaml`, `empty_floor.yaml` | black | `E` on the floor, the lamp split to 17,920 triangles against the floor's faces |
| `bunny_volume.yaml`, `empty_volume.yaml` | black | `G` at every cell and `E` on every patch, the lamp's own 1,120 triangles |
| `bunny_reflecting.yaml` | floor, ceiling, walls diffuse at 0.5 | the same, the reflectors coarsened to triangles of at most 0.2 m |
| `bunny_reflecting_coarse.yaml` | the same | at most 0.4 m: what the reflector size moves |

Each writes the standard outputs -- `patches.vtm` (one `.vtp` per boundary patch, the face fields
under their own names), `fields.vtu` when cells are receivers, and `run.yaml` with the powers --
which `outputs.py` reads back for `figures.py` and `slices_refined.py`, and which open as they are in
ParaView or the aquaflux viewer. `run_cases.py` runs them in one process, one after another
(`RAY_CASES` picks them; `RAY_OVERWRITE=1` replaces a previous run's directory).

`room.py` holds the room's dimensions, the slices, the mesh-quality gate, and the lamp's power and
wall reflectance as DOM and the reference are given them, so those scripts cannot disagree; aquaflux
takes the lamp's photometry, orientation and reflectances from the case files, and its `run.yaml`
records the power it integrated from the IES table (118.826 mW, the same number).

## Reproducing

```bash
# of-optical-radiation with the incident-flux output (branch add-incident-flux-output), and its image
docker build -t oor:local <of-optical-radiation checkout>
export RAY_OOR_SOURCE=<of-optical-radiation checkout>
export RAY_BUNNY_STL=<path to voronoi_bunny_open.stl>
RAY_STAGES=mesh validation/run_case.sh validation/ray_effects_room/generate_dom.py --wait
python3 validation/ray_effects_room/patches.py bunny empty
RAY_STAGES="bunny:6x6 empty:6x6 bunny:12x12 empty:12x12 bunny+reflecting:6x6 bunny:6x6/3x3" \
    validation/run_case.sh validation/ray_effects_room/generate_dom.py --wait
validation/run_case.sh validation/ray_effects_room/run_cases.py --wait
RAY_MESHES="bunny empty" validation/run_case.sh validation/ray_effects_room/slices_refined.py --wait
python3 -m venv work/refvenv && work/refvenv/bin/pip install numpy trimesh embreex
work/refvenv/bin/python validation/ray_effects_room/reference.py
RAY_REFERENCE_BUNNY=stl RAY_MESHES=bunny work/refvenv/bin/python validation/ray_effects_room/reference.py
python3 validation/ray_effects_room/figures.py
```

The bunny STL is not in the repository (13 MB). `voronoi_bunny.py` regenerates it from the Stanford
bunny: `python voronoi_bunny.py --cell 0.065 --strut 0.008 --shell 0.008 --pitch 0.0025 --output
voronoi_bunny_open.stl` (needs `trimesh scikit-image scipy rtree`).

## The meshes (2026-09-28)

OpenFOAM 13 in Docker (image `oor:local`, arm64), `blockMesh` at 5 cm, `snappyHexMesh` with the
bunny at level 4 (3.1 mm), the lamp box at level 4, and a level-2 (1.25 cm) box over the shadow
volume from the floor to the bunny's base and over the floor band `z < 3 cm`, `|x|, |y| < 1.2 m`.

| mesh | cells | floor faces | max non-orthogonality | max skewness | checkMesh failures |
|---|---|---|---|---|---|
| bunny | 2,461,089 | 41,548 | 62.6 deg | 4.79 | 4 faces over skewness 4; 67,136 "concave" cells |
| empty | 1,927,024 | 41,548 | 25.2 deg | 0.33 | 28,024 "concave" cells |

The "concave" count is `-allGeometry`'s test flagging refinement-transition faces — the empty room is
all hexahedra and trips it too — and `room.check_mesh_summary` tolerates exactly those two failures,
recording their counts; any other stops the run. The snapped bunny is one connected piece (every strut
survived), 0.5459 m² against the STL's 0.5524 m². The bunny STL was placed by recentring its bounding
box on x = y = 0 and translating its base to 1.2 m: `translate=(0.000236 0.009318 1.202035)`.

## Results (DOM 2026-10-02/03; the reference 2026-09-28/29; aquaflux 2026-10-05/06)

Configuration for every DOM run below: of-optical-radiation `4f95835` (branch `add-incident-flux-output`:
the `qin` output, the `iesEmitter` exact-power normalization, and no per-ray snapshot without
scattering), OpenFOAM 13 in Docker (`oor:local`, arm64, 11 CPUs, 17.5 GB), 8 MPI ranks (scotch),
`bounded Gauss linearUpwind` ray transport, converged to `convergence 1e-5`, one run at a time;
aquaflux and the reference on the host, CPU only, x64, jax 0.10.2, macOS arm64, 11 cores. aquaflux
is the case files in `cases/` run with `aquaflux run` (through `run_cases.py`) at `943389aa` plus the
uncommitted change that introduced them; its wall times are `run.yaml`'s `seconds`, the mesh read
included. Lamp 118.826 mW. **The angular grids are uniform**: `nPhi = nTheta = 6` (72 directions, every bin 30 degrees
square) and `12` (288, 15 degrees). of-optical-radiation's DOM divides the polar angle over the whole
sphere into `nTheta` bins of `pi/nTheta` and the azimuth into `2 nPhi` bins of `pi/nPhi`, so the
bins are square only when `nPhi == nTheta` (`6 x 6` and `12 x 12` match Fluent's `3 x 3` and `6 x 6`
theta x phi divisions per octant). Errors are on the floor faces within 1 m of the centre,
area-weighted, as shares of the reference's peak.

**Black room, floor irradiance:**

| room | solver | directions | wall time | L2 / peak | worst face / peak |
|---|---|---|---|---|---|
| bunny | reference | 17,920 window samples | 88 s | -- | -- |
| bunny | aquaflux | exact, 17,920 lamp facets | 306 s | 3e-6 | 6e-5 |
| bunny | DOM | 72 (37 sweeps) | 2,654 s | 3.53 | 21.0 |
| bunny | DOM, 3x3 pixels | 72 (37 sweeps) | 2,804 s | 3.53 | 21.0 |
| bunny | DOM | 288 (38 sweeps) | 10,853 s | 1.21 | 4.7 |
| empty | aquaflux | exact | 148 s | 5e-9 | 8e-9 |
| empty | DOM | 72 (37 sweeps) | 1,507 s | 3.52 | 20.9 |
| empty | DOM | 288 (38 sweeps) | 5,867 s | 1.28 | 4.6 |

- **Every DOM run conserves the lamp's power** (its `qin` over all patches sums to 118.8263 mW; the
  3 x 3 run to 118.8256), so its error is entirely in where the light goes. At 72 directions the most
  nearly vertical beams leave the lamp 15 degrees from vertical in 12 azimuths and land as 12 spots
  ~0.8 m from the centre, up to 23 times the true peak over the whole floor; the floor's centre is
  dark and the bunny absorbs -0.20 mW (the scheme's undershoot), against 4.96 mW direct. At 288 the
  beams 7.5 and 22.5 degrees off vertical land as a ring at ~0.4 m and 24 spots at ~1.2 m, up to 6.4
  times the peak, and the bunny absorbs 3.96 mW. **Over the whole floor the error is above the peak at
  both**: L2 / peak 2.58 at 72 and 1.08 at 288; the central 2 x 2 m holds 48 % and 24 % of the
  lamp's power against the true 34 %, and the bounded linearUpwind scheme's undershoots beside each
  beam total -19 % (72) and -12 % (288) of it as negative irradiance.
- **Murthy-Mathur pixelation does not change this.** At 72 directions 3 x 3 pixels give the same
  floor error as 1 x 1 to seven figures and cost 6 % more wall time; the fields differ only at the
  bunny's snapped cells, because elsewhere the mesh's faces are axis-aligned and so are the bins'
  edges, so no bin overhangs a face. The 288-direction run is 1 x 1 only: 3 x 3 at 256 directions
  swapped this machine throughout.
- **The fluence rate agrees the same way.** On the vertical slice aquaflux is 1.3e-4 of the
  reference's 99th percentile (L2) and on the horizontal 1.2e-6; DOM is 0.0051 / 0.0017 (72 / 288)
  vertically and 1.83 / 0.80 on the horizontal slice below the bunny. aquaflux at all 2,461,089
  cells (`bunny_volume.yaml`) took 805 s with the 1,120-facet lamp, and also gives `E` on every
  patch: the bunny absorbs 4.96 mW, the walls 14.38 mW, the floor 99.49 mW. On the slices, which
  `slices_refined.py` gathers again with the refined lamp, the two lamps differ by under 2.4e-5 past
  1 m in the empty room and by 2.9 % (99th percentile) in the bunny's penumbrae.
- **The reference is converged**: four times the window samples moves the floor by 3e-8 of the
  peak (empty) and 7e-4 (bunny); against the STL the mesh was snapped to instead of the snapped
  surface, the floor power moves by 0.025 %.
- **The grid's shape matters as much as its count.** The first runs of this case used `nPhi 8,
  nTheta 4` (64 directions; polar bins 45 degrees, azimuthal 22.5) and `16 x 8` (256), believed
  uniform because the tutorial's comment called `nTheta` the bins per hemisphere. At 64 the nearest
  beams were 22.5 degrees off vertical and landed as 16 spots ~1.7 m out, mostly outside the 1 m
  window, so the window error (0.77) was lower than the uniform 72's while the whole-floor error was
  2.03. Those runs are kept in `work/runs_nonuniform/`; nothing reads them.

**Reflecting room** (floor, ceiling and walls diffuse at 0.5, bunny black; `bunny_reflecting.yaml`):
aquaflux 5,277 s for every patch and every cell, the reflectors the room's own patches coarsened to
9,762 triangles of at most 0.2 m with the lamp's light averaged over 16 points of each (156,192),
the lamp its patch's 1,120 triangles and black to light arriving on it; partly shared with unit
tests, so the time is an upper bound. The room absorbs 113.13 mW and the bunny 5.69 mW (4.96
direct, 0.73 bounced), and the reflected light is 13.44 % of the floor's power. At most 0.4 m
(`bunny_reflecting_coarse.yaml`, 2,442 triangles, 2,009 s, its last minutes beside a test run)
moves the reflected floor by at most 4.5 % of its own peak, under 0.5 % of the total, and the
reflected `G` by 0.24 % (median over the cells). An earlier script, at `8dd4caae` with 2,000 squares
of 0.2 m (4,000 triangles) and the refined lamp, agreed with this to 4e-5 in the room's power, 4e-3 of its peak in the
reflected floor (99th percentile), and 2.6 % of its peak in the reflected `G` (99th percentile,
0.7 % past 10 cm from the walls, where the two sets of reflectors differ most). DOM at 72 directions took
14,597 s (65 sweeps; the machine was shared with other work for its first few minutes), conserves the
power (what the patches absorb sums to the lamp's output), has the bunny absorb 0.85 mW, and is 3.34
of aquaflux's floor peak away in L2. A reflecting sweep cost 3.7 min against 1.2 min in the black
room (3.1x), with 76 % more sweeps. **The 288-direction reflecting run was not made**: scaled from
these it would take about 16 hours, and 72 already shows the ray effect, at a cost far above
aquaflux's.
