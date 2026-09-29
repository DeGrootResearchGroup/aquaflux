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
| aquaflux, floor | `aquaflux_floor.py` | `aquaflux/<mesh>.npz`, `.json` |
| aquaflux, every cell | `aquaflux_volume.py` | `aquaflux/<mesh>_volume.npz`, `.json` |
| aquaflux, reflecting room | `aquaflux_reflecting.py` | `aquaflux/<mesh>_reflecting.npz`, `.json` |
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

`room.py` holds the room's dimensions, the lamp's orientation and power, and the mesh-quality gate,
so the scripts cannot disagree about them.

## Reproducing

```bash
# of-optical-radiation with the incident-flux output (branch add-incident-flux-output), and its image
docker build -t oor:local <of-optical-radiation checkout>
export RAY_OOR_SOURCE=<of-optical-radiation checkout>
export RAY_BUNNY_STL=<path to voronoi_bunny_open.stl>
RAY_STAGES=mesh validation/run_case.sh validation/ray_effects_room/generate_dom.py --wait
python3 validation/ray_effects_room/patches.py bunny empty
RAY_STAGES="bunny:8x4 empty:8x4 bunny:16x8 empty:16x8 bunny+reflecting:8x4" \
    validation/run_case.sh validation/ray_effects_room/generate_dom.py --wait
validation/run_case.sh validation/ray_effects_room/aquaflux_floor.py --wait
validation/run_case.sh validation/ray_effects_room/aquaflux_volume.py --wait
validation/run_case.sh validation/ray_effects_room/aquaflux_reflecting.py --wait
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

## Results (2026-09-28/29)

Configuration for every run below: of-optical-radiation `4f95835` (branch `add-incident-flux-output`:
the `qin` output, the `iesEmitter` exact-power normalization, and no per-ray snapshot without
scattering), OpenFOAM 13 in Docker (`oor:local`, arm64, 11 CPUs, 17.5 GB), 8 MPI ranks (scotch),
`bounded Gauss linearUpwind` ray transport, converged to `convergence 1e-5` (every run ended below
it, between 7.3e-6 and 9.3e-6); aquaflux and the reference on the host, CPU only, x64, jax 0.10.2,
macOS arm64, 11 cores. Lamp 118.826 mW. Errors are on the floor faces within 1 m of the centre,
area-weighted, as shares of the reference's peak.

**Black room, floor irradiance:**

| room | solver | directions | wall time | L2 / peak | worst face / peak |
|---|---|---|---|---|---|
| bunny | reference | 17,920 window samples | 86 s | -- | -- |
| bunny | aquaflux | exact, 17,920 lamp facets | 275 s | 3e-6 | 6e-5 |
| bunny | DOM | 64 (36 sweeps) | 2,251 s | 0.77 | 1.0 |
| bunny | DOM | 256 (38 sweeps) | 9,785 s | 1.76 | 6.0 |
| empty | aquaflux | exact | 352 s | 5e-9 | 8e-9 |
| empty | DOM | 64 (36 sweeps) | 1,285 s | 0.84 | 1.0 |
| empty | DOM | 256 (38 sweeps) | 5,443 s | 1.77 | 5.6 |

- **Every DOM run conserves the lamp's power exactly** (its `qin` over all patches sums to
  118.8263 mW), so its error is entirely in where the light goes. At 64 directions the downward
  beams leave the lamp ~30 degrees from vertical: the floor's centre is dark and the bunny, within
  ~10 degrees of it, is missed (it absorbs -0.004 mW). At 256 the nearest beams are ~11-15 degrees
  out and land as a ring at ~0.8 m up to 6x the true peak, still skirting the bunny (0.64 mW).
  Quadrupling the directions moved the error, not reduced it.
- **The fluence rate agrees the same way.** On the vertical slice aquaflux is 1.3e-4 of the
  reference's 99th percentile (L2) and on the horizontal 1.2e-6; DOM is 0.012 / 0.004 (64 / 256)
  vertically and 1.65 / 0.89 on the horizontal slice below the bunny. aquaflux at all 2,461,089
  cells took 849 s (584 s of it the ray tests against the bunny's 258,008 triangles), with the
  1,120-facet lamp; on the slices, which use the refined lamp, the two lamps differ by under 2.4e-5
  past 1 m in the empty room and by 2.9 % (99th percentile) in the bunny's penumbrae.
- **The reference is converged**: four times the window samples moves the floor by 3e-8 of the
  peak (empty) and 7e-4 (bunny); against the STL the mesh was snapped to instead of the snapped
  surface, the floor power moves by 0.025 %.

**Reflecting room** (floor, ceiling and walls diffuse at 0.5, bunny black): aquaflux 4,115 s for
the floor, both slices and every cell (39 min of it the surface solve at 20 cm squares, with the
lamp's light on 64,000 points of the room); the room absorbs 113.13 mW and the bunny 5.70 mW (4.96
direct, 0.73 bounced); the two routes to the slices agree exactly; halving the squares moves the
reflected floor by at most 5 % of its own peak, under 1 % of the total. DOM at 64 directions took
11,910 s (57 sweeps), conserves the power exactly, has the bunny absorb 0.35 mW, and is 0.73 of
aquaflux's floor peak away in L2. **DOM at 256 directions was stopped after 15 sweeps (4 h)**: a
reflecting sweep cost 16.4 min against 4.3 min in the black room (3.3x at 64 directions, 3.8x at
256), with ~60 % more sweeps to converge, so it needed an estimated 11-15 more hours. **The 576-direction runs were not made**, by decision: 64 and 256 already show the ray
effect, at a cost far above aquaflux's.
