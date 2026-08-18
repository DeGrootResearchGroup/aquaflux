# aquaflux — package structure

The annotated file tree: an orientation map of where each concern lives. Subsystems are
added as physics lands; each module has a single responsibility and each subpackage a
single concern. Empty stub modules are deliberately avoided — they rot. The YAML→AST DSL
and the aquakin reaction coupling are not scaffolded, by design: the DSL is the last
layer, added when that work starts rather than front-loaded.

This page is a repository reference for contributors, not user documentation, so it is
excluded from the built site (`conf.py` `exclude_patterns`).

```
cfd/                                  # repo root
├── README.md                         # public overview, install, examples
├── CLAUDE.md                         # contributor briefing + engineering principles
├── pyproject.toml                    # build, deps, extras (lint/test/docs/petsc), ruff + pytest config
├── .readthedocs.yaml                 # Read the Docs build (fail_on_warning: any warning fails)
├── .github/workflows/ci.yml          # ruff + codespell gate, sharded unit/integration tiers
├── .githooks/pre-push                # local ruff + spelling gate (enable: git config core.hooksPath .githooks)
├── .githooks/pre-commit              # non-blocking documentation-sync reminder
├── .claude/rules/                    # path-scoped subsystem rules (auto-load on edit)
│   ├── mesh.md  discretization.md  schemes.md  boundary.md  properties.md
│   └── solve.md  flow.md  turbulence.md  io.md  parallel.md
│
├── docs/                             # the Sphinx site (MyST Markdown; api.md generated at build)
│   ├── conf.py                       #   Sphinx config + _write_api_page (API page from each __all__)
│   ├── index.md  mesh.md  mesh_zones_and_patches.md
│   └── package_structure.md          #   this file (excluded from the built site)
│
├── tools/
│   ├── fastgate.sh                   # the blessed test-tier runner (redirects, never pipes; reports pytest's own status)
│   ├── check_hooks.sh                # warns when core.hooksPath does not resolve to .githooks (run by fastgate.sh)
│   └── canadian-spelling.dict        # codespell dictionary for the spelling gate
│
├── validation/                       # runnable scientific cases (not the pytest tiers)
│   ├── run_case.sh                   #   the blessed long-run launcher: unbuffered log, memory pre-flight, run-file record
│   ├── bfs3d_openfoam/               #   3D backward-facing step vs OpenFOAM + conditioning/preconditioner probes
│   ├── pitzdaily_openfoam/           #   2D pitzDaily vs OpenFOAM
│   ├── turbulent_channel/            #   turbulent channel vs the analytical log law
│   └── turbulent_channel_openfoam/   #   turbulent channel vs OpenFOAM
│
├── aquaflux/                         # the package
│   ├── __init__.py                   # enables JAX x64 (process-wide); __version__; public API
│   ├── vectors.py                    # leaf: per-element vector algebra — dot / norm_squared / scale (buries axis/[:,None] bookkeeping; imported everywhere)
│   ├── text_table.py                 # fixed-width ASCII tables for solver reports
│   │
│   ├── mesh/                         # Mesh container + face/cell geometry + connectivity (2D and 3D)
│   │   ├── mesh.py                   #   Mesh(eqx.Module): SoA coords + connectivity objects + zones/patches; from_faces(), from_csr() (vectorized, no per-face Python loop), validate(), geometry() → MeshGeometry
│   │   ├── geometry.py               #   MeshGeometry: the {face, cell} bundle geometry() returns (derived on demand from node_coords, never stored — keeps grad-through-node-positions correct, no stale cache)
│   │   ├── connectivity.py           #   FaceCellConnectivity (gather_owner/gather_neighbour/interior + scatter/scatter_conservative/scatter_symmetric/max/min) + FaceNodeConnectivity (ragged CSR) + interior_mask
│   │   ├── face.py                   #   FaceGeometry + FaceGeometryScheme → EdgeFaceGeometry (2D) / PolygonFaceGeometry (3D, centre-fan)
│   │   ├── cell.py                   #   CellGeometry: volume, centroid (divergence theorem, dim-general)
│   │   ├── groups.py                 #   LabelledGroups base → CellZones / FacePatches (named partitions; derived interfaces)
│   │   ├── quality.py                #   diagnostics: face_planarity, centroid_iteration_shift, closed_cell_residual
│   │   ├── distance.py               #   distance_to_patches(): cell→named-patch distance, the geometric field wall models need
│   │   ├── reorder.py                #   cell renumbering: permute_cells() (canonical P·A·Pᵀ) + CellReordering → Identity/ReverseCuthillMcKee/Random (RCM protects the AMG coarse space on large meshes)
│   │   ├── graph.py                  #   cell↔cell adjacency (build-time): cell_adjacency_coo() (SciPy COO, for RCM) / cell_adjacency_csr() (CSR arrays, for partitioners)
│   │   ├── structured.py             #   structured_grid_2d() / structured_grid_3d() / graded_nodes(): clean orthogonal quad/hex grids (interior-node skew for order studies lives in tests/support/meshes.py)
│   │   └── collapse.py               #   collapse_extruded_direction(): one-cell-thick extruded 3D mesh → genuine 2D Mesh (used by the OpenFOAM reader for empty-patch 2D cases)
│   │
│   ├── io/                           # mesh import (file formats → Mesh); depends on mesh, never the reverse
│   │   ├── reader.py                 #   MeshReader(eqx.Module) strategy: read() → Mesh (format-agnostic seam)
│   │   └── openfoam/                 #   OpenFOAM polyMesh reader (ASCII): read_openfoam() / OpenFOAMReader
│   │       ├── records.py            #     FoamPatch / CellZone / PolyMeshData value records
│   │       ├── foamfile.py           #     FoamFile envelope: comment strip, header dict, ASCII/binary detection; read_foam_body() is the one file->body entry
│   │       ├── grammar.py            #     body parsers (points/faces/labels/boundary/cellZones) sharing one list envelope
│   │       ├── fields.py             #     read a scalar field written on an imported mesh (phi, nut): parse_scalar_field (pure) + read_surface_scalar_field (faces, placed by index, ordering checked) + read_volume_scalar_field (cells)
│   │       ├── assembler.py          #     assemble(PolyMeshData) → Mesh: neighbour pad, n_cells, patches/zones, from_csr
│   │       └── reader.py             #     OpenFOAMReader: file I/O; parse → assemble → collapse (empty-patch 2D)
│   │
│   ├── properties/                   # per-cell physical property fields, decoupled from the numerics (operators name what they consume)
│   │   ├── property.py               #   Property → Constant, ZoneConstant, FieldProperty (evaluated from the cell partition and state fields; values are plain differentiable leaves)
│   │   └── model.py                  #   PropertyModel: named {name: Property} collection; evaluate() → {name: (n_cells,)}, require()
│   │
│   ├── discretization/               # the residual substrate: gather → compute → scatter assembly of R(state, params)
│   │   ├── face_flux.py              #   the face-flux contract: FaceFluxOperator (face_flux(field, context)) + FaceContext (the lean shared per-face inputs)
│   │   ├── diffusion.py              #   DiffusionFlux: flux-continuous non-orthogonal diffusion (correction written into the residual; AD linearizes it) + flux_continuous_conductance/_denominator
│   │   ├── advection.py              #   AdvectionScheme → FirstOrderUpwind, LimitedUpwind; AdvectionFlux(mass_flux, scheme) — the mass flux is always injected (Rhie–Chow in flow; a prescribed field in tests)
│   │   ├── source.py                 #   VolumeSource: the volume-integrated source contract (the reaction-coupling seam; turbulence source terms implement it)
│   │   ├── transient.py              #   TransientTerm: BDF1 on the first step, BDF2 thereafter
│   │   ├── fixed_value.py            #   FixedValueCells: replace a set of cells' residual rows with an algebraic constraint (FixationRow → DifferenceRow / LogRatioRow)
│   │   └── residual.py               #   ResidualAssembler (builds the FaceContext: properties, gradients, boundary values) + CellBalance (operators → segment_sum scatter → sources → transient); R = accumulation + Σ outward flux
│   │
│   ├── schemes/                      # first-class swappable numerics (physics-free; one-way discretization → schemes)
│   │   ├── gradient.py               #   GradientScheme → CompactGreenGauss, CorrectedGreenGauss (injected GradientSolve: GmresGradientSolve / SweptGradientSolve fixed-sweep), HessianCorrectedGradient
│   │   ├── interpolation.py          #   face interpolation in one place: interpolation_factor(g), interpolate_owner_neighbour ((1-g)·a + g·b)
│   │   └── limiter.py                #   Limiter → VenkatakrishnanLimiter (per-cell psi for bounded second-order reconstruction; held by LimitedUpwind)
│   │
│   ├── boundary/                     # weak boundary-face-value closures (a BC is a special face interpolator)
│   │   ├── conditions.py             #   BoundaryCondition → Dirichlet, DirichletField, ZeroGradient, Neumann, Convective
│   │   └── collection.py             #   BoundaryConditions: named {patch: closure} collection bound to a mesh's patches, applied by the shared iterate-patches → gather owner → set fold
│   │
│   ├── flow/                         # the coupled pressure–velocity (u, v[, w], p) block
│   │   ├── state.py                  #   BlockStateLayout: the flat [vel_0..vel_{dim-1}, pressure] layout (pack/unpack); FlowFields / VelocityFields
│   │   ├── momentum.py               #   MomentumContinuity: the coupled residual (each momentum component is a CellBalance over Diffusion/PressureForce/Advection; Rhie–Chow continuity; pressure pin; takes a PropertyModel → viscosity/density) + PressureForce
│   │   ├── source.py                 #   MomentumSource (vector volume source: source/face_force/diagonal) + UniformBodyForce; where buoyancy, porous drag, rotating-frame terms attach
│   │   ├── rhie_chow.py              #   interior_mass_flux + momentum_diagonal / frozen_momentum_diagonal_parts (viscous + convective)
│   │   ├── boundary.py               #   FlowBoundary → NoSlipWall, MovingWall, VelocityInlet, PressureOutlet
│   │   ├── preconditioner.py         #   SIMPLE Schur pieces: pressure_schur_laplacian (a_P-based), damped_jacobi_solve
│   │   ├── block_preconditioner.py   #   BlockPreconditioner composing injected InnerSchurSolver / VelocityBlockSolver strategies
│   │   ├── continuation.py           #   momentum_continuation / reused_flow_solve: pseudo-transient continuation for the flow Newton solve at high Reynolds number
│   │   ├── initialization.py         #   cheap initializers: laplace_field, potential_flow, bernoulli_pressure
│   │   ├── mean_velocity.py          #   bulk_velocity_flow_solve: the driving body force is a solve unknown, not a feedback loop
│   │   └── scales.py                 #   characteristic_velocity: the flow's velocity scale, derived from what drives it
│   │
│   ├── transport/                    # scalar transport by a converged flow (species, temperature, tracers); the aquakin reaction seam
│   │   └── scalar.py                 #   ScalarTransport (composes Advection/Diffusion/sources/transient) + effective_diffusivity (D + nu_t/Sc_t)
│   │
│   ├── turbulence/                   # k–ω SST closure and the flow–turbulence coupling
│   │   ├── sst.py                    #   SSTModel: the closure constants and the quantities derived directly from them
│   │   ├── transport.py              #   assembly of the k and ω transport equations on a configured mesh (ScalarVariableTransform → DirectScalars / LogScalars)
│   │   ├── sources.py                #   the SST source terms as VolumeSource operators: KProduction/KDestruction, OmegaProduction/OmegaDestruction/OmegaCrossDiffusion
│   │   ├── strain.py                 #   strain_rate_magnitude: the scalar invariant the closure consumes
│   │   ├── boundary.py               #   wall and inlet values: omega_wall, nut_wall, inlet_k/inlet_omega, wall_y_star, wall_shear_stress (y+-insensitive wall treatment)
│   │   ├── coupled.py                #   CoupledRANS: the monolithic residual R(u, p, k, ω) + CoupledRANSLayout
│   │   ├── driver.py                 #   solve_segregated: the segregated outer loop coupling the flow solve to the closure
│   │   ├── continuation.py           #   pseudo-transient continuation for the (k, ω) scalar transport solves
│   │   ├── reynolds.py               #   solve_reynolds_continuation: reach a high-Reynolds root through easier lower-Re ones
│   │   ├── initialization.py         #   hybrid_initialize: potential velocity + smoothed turbulence initial condition
│   │   ├── preconditioner.py         #   convection-diffusion AMG for the scalar transport solves; the coupled *_continuation builders (AMG / ILUT / LU)
│   │   └── diagnostics.py            #   named physical fields of a coupled state, for march-log reporting
│   │
│   ├── solve/                        # Newton on the residual, the differentiated linear solve, and the AMG that preconditions it
│   │   ├── newton.py                 #   newton_step: the Newton correction on the cell residual
│   │   ├── implicit.py               #   ImplicitNewtonSolver: Newton to convergence + the implicit-function-theorem adjoint (one transpose solve, not the iteration on the tape)
│   │   ├── linear.py                 #   solve_linear: differentiable matrix-free linear solve, optional left/right preconditioning; relative_residual_gmres
│   │   ├── norm.py                   #   ResidualNorm → BlockScaledNorm / RowScaledNorm: the convergence and globalization measures
│   │   ├── march.py                  #   forward_march: the observed forward-only Newton march + the staleness trigger watching it
│   │   ├── march_log.py              #   MarchLogger: the streaming per-step log (the reporting half of the on_step seam)
│   │   ├── checkpoint.py             #   StateCheckpointer: periodic state persistence (the on_checkpoint seam)
│   │   ├── step_control.py           #   feedback step controls for the eager march (DualTimeControl and friends)
│   │   ├── continuation.py           #   PseudoTransientStep / ForwardStep: continuation as a residual-agnostic forward step
│   │   ├── relaxation.py             #   RelaxationSchedule → SwitchedEvolutionRelaxation, ConstantRelaxation: how the shift strength beta is set each step
│   │   ├── shift_basis.py            #   ShiftBasis: how the pseudo-transient shift's spatial distribution is built from a cell's operator parts
│   │   ├── line_search_growth.py     #   LineSearchGrowth: how much the residual may grow and still be accepted
│   │   ├── multigrid.py              #   matrix-free algebraic multigrid for the inner solves (smoothed/plain aggregation, AIR; scipy RAP off the jit path)
│   │   ├── frozen_operator.py        #   convection_diffusion_operator / decouple_dof: the one assembler of the frozen operator every AMG consumer coarsens
│   │   ├── amg_preconditioner.py     #   MonolithicAmgPreconditioner for the coupled saddle-point solve
│   │   ├── ilut_preconditioner.py    #   MonolithicIlutPreconditioner (threshold incomplete LU)
│   │   ├── lu_preconditioner.py      #   MonolithicLuPreconditioner (complete sparse LU)
│   │   ├── field_split.py            #   BlockTriangularFieldSplit: block-triangular field-split preconditioning for flow-plus-transport
│   │   ├── native_inverse.py         #   NativeHierarchyInverse: the shared body of a JAX-native block inverse (hierarchy, in-place refresh, transpose)
│   │   ├── saddle_multigrid.py       #   NativeSimpleInverse: a JAX-native multigrid over the flow saddle, smoothed by SIMPLE relaxation
│   │   ├── host_vcycle.py            #   HostVCycleInverse: the same hierarchy applied on the host, smoothed by an incomplete factorization
│   │   ├── ilu0.py                   #   Ilu0: zero-fill incomplete LU on the operator's own pattern, refreshable without repeating the symbolic phase
│   │   ├── ordering.py               #   CellMajor + cell-order strategies: which order an incomplete factorization eliminates in, which at zero fill decides what it discards
│   │   ├── _ilu0.pyx                 #   the compiled elimination and triangular solves behind Ilu0 (sequential by nature, so not array operations)
│   │   ├── sparse_jacobian.py        #   materialize_block_jacobian: the sparse Jacobian by compressed graph-coloured probing
│   │   └── refresh_timing.py         #   RefreshTiming: what a preconditioner refresh did, and what each part cost
│   │
│   └── parallel/                     # distributed memory: decomposition and halo exchange (the concern kept out of Mesh)
│       ├── partitioner.py            #   Partitioner → BlockPartitioner (RCM-block, dependency-free default) / ScotchCLIPartitioner / ScotchPartitioner; consumes cell_adjacency_csr
│       ├── partition.py              #   PartitionedMesh + LocalPartition; partition_mesh(mesh, labels) → owned+halo local meshes, gathered geometry, halo plan
│       ├── padding.py                #   pad_partition / PaddedLayout: uniform per-partition shapes so shard_map can map over the decomposition
│       ├── halo.py                   #   HaloExchange → AllGatherHaloExchange / AllToAllHaloExchange (refresh ghost cells from the partitions that own them)
│       └── distributed.py            #   build_distributed_residual: pad → shard_map an *injected* per-partition assembler (never a re-implementation of the physics)
│
└── tests/
    ├── conftest.py                   # shared fixtures
    ├── unit/                         # component tests in isolation (operator order-of-accuracy, schemes, solvers, readers, properties, vectors, …)
    ├── integration/                  # full assembly→solve against analytical solutions; the validation-marked benchmarks live here too
    ├── support/                      # test-only helpers: meshes.py (clean + perturbed grids), polymesh.py (polyMesh fixtures), fields.py (prescribed divergence-free mass flux)
    └── fixtures/                     # on-disk polyMesh fixture directories for the reader tests
```

## Intended modules (added as physics proceeds)

Not yet created — listed so their home is decided in advance, which avoids duplication
when the work starts:

| Planned module | Holds |
|---|---|
| `properties/property.py::Calculated` | a state-dependent property (temperature-dependent viscosity, …): a differentiable `formula(params, *fields)` evaluated inside the residual |
| `dsl/` | the YAML→AST model format that emits residual terms — the last layer, deliberately unscaffolded |

Keep this tree in sync when a module is created or a home changes.
