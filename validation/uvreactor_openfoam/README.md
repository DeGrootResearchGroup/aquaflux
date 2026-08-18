# UV reactor — a GPU-scale mesh import case

## Why

The existing validated cases are small (`pitzdaily_openfoam` ~12k cells, `bfs3d_openfoam` ~23–66k
cells) — small enough that a residual or Jacobian-vector-product call dispatches many tiny fused
kernels, each dominated by launch latency rather than compute, which caps how much a GPU actually
helps. This case is a real, CAD-derived `snappyHexMesh` mesh at ~1.6 million cells, meant to give a
GPU enough parallel work per kernel to be worth profiling at scale.

## Source

Geometry, mesh settings, and the flow operating point are taken from
[of-optical-radiation](https://github.com/DeGrootResearchGroup/of-optical-radiation)'s
`uvReactorSozzi2006` tutorial: water at 20 °C, 25 GPM through a 19.1 mm pipe (Re ≈ 1.05e5),
realizable k-epsilon RAS. Only the flow-relevant subset of that tutorial is kept here — mesh
generation, boundary conditions, the operating point. The source case's own radiation-transport
physics and post-processing are not part of this directory; aquaflux does not model radiation
transport. `of_case/` is otherwise the untouched case (the only edit is `system/controlDict`'s
`writeFormat`, `binary` → `ascii`, since aquaflux's OpenFOAM reader is ASCII-only).

## Known status

- The mesh (~1.6M cells, general polyhedra up to 21 faces from `snappyHexMesh`'s cut cells) imports
  cleanly at full scale: OpenFOAM's own `checkMesh` reports "Mesh OK", and `dpn_diagnostic.py`
  confirms every interior and boundary face's diffusion-flux normal-distance denominator is
  strictly positive (checked directly, not inferred from `checkMesh`'s own summary).
- `flow_smoke_test.py`'s pipeline — case assembly, `hybrid_initialize`, a short direct march with a
  finite residual — is confirmed working on this same mesh at a smaller scale (a reduced-refinement
  copy used while diagnosing the mesh-import defect above). At the full ~1.6M-cell scale, case
  assembly plus `hybrid_initialize` alone is memory-heavy enough to have been killed by the OS on a
  shared development machine under concurrent load — one of the reasons this case exists is to move
  that work onto a machine actually sized for it.
- A direct coupled march at this Reynolds number is not expected to converge without Reynolds
  continuation, which this case does not yet attempt.
- **Not a validated case**: unlike `pitzdaily_openfoam`/`bfs3d_openfoam`, no OpenFOAM reference
  solve is run or compared against here. This case exists to exercise mesh import and geometry at a
  scale relevant to GPU profiling, not to check aquaflux's physics against a reference.

## Regenerating the mesh

`of_case/constant/polyMesh` is not committed (regenerable, ~330 MB, several minutes to build).
Rebuild it with the `openfoam13` Docker image and a local `gmsh` install (`make_geometry.py`
triangulates the STEP CAD surface):

```bash
cd validation/uvreactor_openfoam/of_case
python3 make_geometry.py
docker run --rm -v "$PWD":/case -w /case openfoam13:latest bash -lc \
  "blockMesh && surfaceFeatures && snappyHexMesh -overwrite && checkMesh"
```

`snappyHexMesh`'s castellation and snapping passes need several GB of working disk space beyond the
final mesh's own size.

## Layout

- `of_case/` — the OpenFOAM case template: the STEP CAD geometry, `make_geometry.py` (gmsh surface
  triangulation), `system/`, `constant/`, `0/`. The generated mesh, derived surface triangulation, and
  solver logs are git-ignored.
- `flow_smoke_test.py` — imports the mesh, builds a `CoupledRANS` case, runs `hybrid_initialize`,
  then a short direct march as a pipeline sanity check — not a convergence attempt.
- `dpn_diagnostic.py` — a census of the diffusion flux's normal-distance denominator across a
  mesh's faces, by patch and location; the tool that localized the centroid-precision defect fixed
  in `aquaflux/mesh/cell.py`.
