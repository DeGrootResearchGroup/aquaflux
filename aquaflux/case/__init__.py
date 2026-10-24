"""A whole case -- mesh, physics, boundaries, and for a flow its fluid and numerics -- described in one file.

A case file is a YAML document naming each part of a case as plain settings::

    mesh: {kind: OpenFOAMMesh, path: constant/polyMesh}
    fluid: {density: 1.0, kinematic_viscosity: 1.0e-5}
    physics:
      kind: RANS
      omega_variable: {kind: LogScalars}
    boundaries:
      inlet:
        kind: Inlet
        velocity: [10.0, 0.0]
        turbulence: {kind: FixedTurbulence, k: 0.375, omega: 440.15}
      outlet: {kind: Outlet, pressure: 0.0}
      upperWall: {kind: Wall}
      lowerWall: {kind: Wall}
    numerics:
      momentum_advection: {kind: LimitedUpwind, limiter: {kind: VenkatakrishnanLimiter}}
      turbulence_advection: {kind: FirstOrderUpwind}

:func:`read_case` reads one into a :class:`CaseSpec`, refusing any setting that is unknown, misspelt,
of the wrong form or inconsistent with the rest of the case, with the path to it.
:meth:`CaseFile.check` then reads the mesh and checks the case against it -- every boundary face has a
patch, and every patch fits the mesh -- without computing any geometry, so a case can be checked
cheaply before anything expensive is built.

The file's ``solver`` section says how the case is solved, its optional ``initial`` section what it
starts from (a :class:`Checkpoint` of an earlier run, to resume one that stopped short, or the
:class:`Fields` of an OpenFOAM time directory) and its
``outputs`` section what a run writes; :func:`prepare_run` reads and checks a file for a run, and :meth:`PreparedRun.run` builds,
solves and writes it -- what ``aquaflux run case.yaml`` does.

Each boundary patch is described once for every field: an :class:`Inlet`, an :class:`Outlet`, a
:class:`Wall` or a :class:`Lamp`. The closures each equation needs, and the set of walls a turbulence
closure measures its wall distance from, all follow from that one statement.

A :class:`Radiation` case is the light of its lamps rather than a flow: its patches are lamps and walls
(black, or reflecting by a ``reflectance``), its physics holds the medium, what stands in the way and
where the light is gathered, and it has no fluid and no numerics section::

    mesh: {kind: OpenFOAMMesh, path: constant/polyMesh}
    physics:
      kind: Radiation
      medium: {kind: UniformMedium, transmittance: 70.0}
    boundaries:
      lamp:
        kind: Lamp
        power: 35.0
        profile: {kind: LambertianProfile}
      walls: {kind: Wall, reflectance: 0.3, geometry: {kind: StlSurface, file: walls.stl}}
    outputs:
      fields: [{kind: Vtk}, {kind: PatchVtk}]

It writes the fluence rate ``G`` at the cell centres, the irradiance ``E`` on the walls' faces, and in
``run.yaml`` where the lamps' power goes.
"""

from __future__ import annotations

from .boundaries import (
    FixedTurbulence,
    Inlet,
    InletTurbulence,
    IntensityLength,
    Lamp,
    Outlet,
    PatchCondition,
    Wall,
)
from .case_file import (
    CaseFile,
    CheckedCase,
    MeshFile,
    read_case,
    read_case_document,
    read_mesh_file,
    write_case,
    write_mesh_file,
)
from .fluid import Fluid
from .forcing import BodyForce, BulkVelocity, DriveSpec, SourceSpec
from .initial import Checkpoint, Fields, InitialState, StartingFields
from .drawing_view import DRAWING_FILE, DrawingView, view_drawing
from .mesh_file import Drawing, DrawingRegion, DrawingSections, MeshOutput, MeshSizes, MeshSpec
from .mesh_run import MeshRecord, make_mesh
from .mesh_source import AquafluxMesh, MeshSource, OpenFOAMMesh
from .outputs import Checkpoints, FieldWriter, OpenFOAMTime, Outputs, PatchVtk, RunFields, Vtk
from .paths import relocated
from .drawing import CadPlacement
from .physics import RANS, Laminar, Physics, Radiation
from .radiation import (
    CadFluid,
    CadSolid,
    CadSurface,
    Coarsen,
    CosinePowerProfile,
    IesProfile,
    LambertianProfile,
    LampProfile,
    MeshPatch,
    OccluderSpec,
    PatchBody,
    Receivers,
    StlBody,
    StlSurface,
    SurfaceSource,
    UniformMedium,
)
from .run import PreparedRun, RunPlan, RunRecord, plan_run, prepare_run
from .solver import (
    CoupledMarch,
    FlowMarch,
    NotConverged,
    RadiationSolve,
    RootSolve,
    Segregated,
    SolverSpec,
    ViscosityRamp,
    solver_for,
)
from .spec import (
    CaseSpec,
    Numerics,
    case_schema,
    case_spec_from_mapping,
    case_spec_to_mapping,
    drawing_sections_from_mapping,
    mesh_schema,
    mesh_source_from_mapping,
    mesh_spec_from_mapping,
    mesh_spec_to_mapping,
)

__all__ = [
    "DRAWING_FILE",
    "RANS",
    "AquafluxMesh",
    "BodyForce",
    "BulkVelocity",
    "CadFluid",
    "CadPlacement",
    "CadSolid",
    "CadSurface",
    "CaseFile",
    "CaseSpec",
    "CheckedCase",
    "Checkpoint",
    "Checkpoints",
    "Coarsen",
    "CosinePowerProfile",
    "CoupledMarch",
    "Drawing",
    "DrawingRegion",
    "DrawingSections",
    "DrawingView",
    "DriveSpec",
    "FieldWriter",
    "Fields",
    "FixedTurbulence",
    "FlowMarch",
    "Fluid",
    "IesProfile",
    "InitialState",
    "Inlet",
    "InletTurbulence",
    "IntensityLength",
    "LambertianProfile",
    "Laminar",
    "Lamp",
    "LampProfile",
    "MeshFile",
    "MeshOutput",
    "MeshPatch",
    "MeshRecord",
    "MeshSizes",
    "MeshSource",
    "MeshSpec",
    "NotConverged",
    "Numerics",
    "OccluderSpec",
    "OpenFOAMMesh",
    "OpenFOAMTime",
    "Outlet",
    "Outputs",
    "PatchBody",
    "PatchCondition",
    "PatchVtk",
    "Physics",
    "PreparedRun",
    "Radiation",
    "RadiationSolve",
    "Receivers",
    "RootSolve",
    "RunFields",
    "RunPlan",
    "RunRecord",
    "Segregated",
    "SolverSpec",
    "SourceSpec",
    "StartingFields",
    "StlBody",
    "StlSurface",
    "SurfaceSource",
    "UniformMedium",
    "ViscosityRamp",
    "Vtk",
    "Wall",
    "case_schema",
    "case_spec_from_mapping",
    "case_spec_to_mapping",
    "drawing_sections_from_mapping",
    "make_mesh",
    "mesh_schema",
    "mesh_source_from_mapping",
    "mesh_spec_from_mapping",
    "mesh_spec_to_mapping",
    "plan_run",
    "prepare_run",
    "read_case",
    "read_case_document",
    "read_mesh_file",
    "relocated",
    "solver_for",
    "view_drawing",
    "write_case",
    "write_mesh_file",
]
