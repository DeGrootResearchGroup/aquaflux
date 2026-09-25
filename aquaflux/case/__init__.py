"""A whole case -- mesh, fluid, physics, boundaries, numerics -- described in one file.

A case file is a YAML document naming each part of a case as plain settings::

    mesh: {kind: OpenFOAMMesh, path: constant/polyMesh}
    fluid: {density: 1.0, kinematic_viscosity: 1.0e-5}
    physics:
      kind: RANS
      advection: {kind: FirstOrderUpwind}
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

:func:`read_case` reads one into a :class:`CaseSpec`, refusing any setting that is unknown, misspelt,
of the wrong form or inconsistent with the rest of the case, with the path to it.
:meth:`CaseFile.check` then reads the mesh and checks the case against it -- every boundary face has a
patch, and every patch fits the mesh -- without computing any geometry, so a case can be checked
cheaply before anything expensive is built.

The file's ``solver`` section says how the case is solved and its ``outputs`` section what a run
writes; :func:`prepare_run` reads and checks a file for a run, and :meth:`PreparedRun.run` builds,
solves and writes it -- what ``aquaflux run case.yaml`` does.

Each boundary patch is described once for every field: an :class:`Inlet`, an :class:`Outlet` or a
:class:`Wall`. The closures each equation needs, and the set of walls a turbulence closure measures its
wall distance from, all follow from that one statement.
"""

from __future__ import annotations

from .boundaries import FixedTurbulence, Inlet, InletTurbulence, Outlet, PatchCondition, Wall
from .case_file import CaseFile, CheckedCase, read_case, write_case
from .fluid import Fluid
from .forcing import BodyForce, BulkVelocity, DriveSpec, SourceSpec
from .mesh_source import AxisGrading, GeometricGrading, MeshSource, OpenFOAMMesh, StructuredGrid
from .outputs import Checkpoints, FieldWriter, OpenFOAMTime, Outputs, Vtk
from .physics import RANS, Laminar, Physics
from .run import PreparedRun, RunRecord, prepare_run
from .solver import (
    CoupledMarch,
    FlowMarch,
    NotConverged,
    RootSolve,
    Segregated,
    SolverSpec,
    ViscosityRamp,
    solver_for,
)
from .spec import CaseSpec, Numerics, case_spec_from_mapping, case_spec_to_mapping

__all__ = [
    "RANS",
    "AxisGrading",
    "BodyForce",
    "BulkVelocity",
    "CaseFile",
    "CaseSpec",
    "CheckedCase",
    "Checkpoints",
    "CoupledMarch",
    "DriveSpec",
    "FieldWriter",
    "FixedTurbulence",
    "FlowMarch",
    "Fluid",
    "GeometricGrading",
    "Inlet",
    "InletTurbulence",
    "Laminar",
    "MeshSource",
    "NotConverged",
    "Numerics",
    "OpenFOAMMesh",
    "OpenFOAMTime",
    "Outlet",
    "Outputs",
    "PatchCondition",
    "Physics",
    "PreparedRun",
    "RootSolve",
    "RunRecord",
    "Segregated",
    "SolverSpec",
    "SourceSpec",
    "StructuredGrid",
    "ViscosityRamp",
    "Vtk",
    "Wall",
    "case_spec_from_mapping",
    "case_spec_to_mapping",
    "prepare_run",
    "read_case",
    "solver_for",
    "write_case",
]
