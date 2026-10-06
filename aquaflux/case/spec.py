"""A whole case as one frozen value: the mesh, the fluid, the physics, the boundaries, the numerics.

:class:`CaseSpec` is what a case file describes, and nothing more: plain settings, read and written by
:func:`case_spec_from_mapping` and :func:`case_spec_to_mapping` from the nested mapping a YAML document
parses to. It holds no mesh and builds no equation -- that is what makes checking a case cheap, and
what leaves the solver free to be rebuilt from whatever state a march has reached.

It is not a flat record of every setting any case might need. A small core -- mesh, boundaries,
outputs -- is common to every case; what differs between cases is decided by two discriminators, the
**physics** (:class:`~aquaflux.case.Laminar`, :class:`~aquaflux.case.RANS` or
:class:`~aquaflux.case.Radiation`) and the **drive** (what sets the flow in motion). A setting that
belongs to one physics lives inside it, so a case cannot carry one its physics would ignore, and the
physics decides which of the other sections there are: a flow states its fluid and its numerics, a
radiation case neither.

A spec that is constructed is already consistent on its own terms: its physics has accepted its
boundaries and its sections, and a flow's pressure level is fixed exactly once -- by an outlet, or, in
a closed domain, by its ``pressure_datum``. What it cannot know until its mesh is read -- that every
boundary face has a patch, and that each patch, the datum and the physics' own settings fit the mesh
-- is checked by :meth:`CaseSpec.check_against`.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from collections.abc import Mapping

import numpy as np

from aquaflux.discretization import AdvectionScheme, FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import PinnedPoint, PressureDatum
from aquaflux.mesh import Mesh
from aquaflux.radiation import (
    EveryPair,
    NoOcclusion,
    RadiationSettings,
    RayCastOcclusion,
    ShaftCulling,
    SilhouetteOcclusion,
)
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    GmresGradientSolve,
    GradientScheme,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
    SweptGradientSolve,
    VenkatakrishnanLimiter,
)
from aquaflux.solve import (
    BlockScaled,
    CflResidualDualTimeControl,
    Convergence,
    DirectSolve,
    DualTimeControl,
    DualTimeLoop,
    Euclidean,
    GmresSolve,
    LinearSolveSettings,
    ResidualRatioDualTimeControl,
    RetryPolicy,
    RowScaled,
    SettingsMapping,
)
from aquaflux.turbulence import PRECONDITIONER_SPEC_MAPPING, DirectScalars, LogScalars, SSTModel

from .boundaries import (
    FixedTurbulence,
    Inlet,
    IntensityLength,
    Lamp,
    Outlet,
    PatchCondition,
    Wall,
)
from .fluid import Fluid
from .forcing import BodyForce, BulkVelocity, DriveSpec, SourceSpec
from .mesh_source import GeometricGrading, MeshSource, OpenFOAMMesh, StructuredGrid
from .outputs import Checkpoints, OpenFOAMTime, Outputs, PatchVtk, Vtk
from .physics import RANS, Laminar, Physics, Radiation
from .radiation import (
    CadFluid,
    CadPlacement,
    CadSolid,
    CadSurface,
    Coarsen,
    CosinePowerProfile,
    IesProfile,
    LambertianProfile,
    MeshPatch,
    PatchBody,
    Receivers,
    StlBody,
    StlSurface,
    UniformMedium,
)
from .solver import (
    CoupledMarch,
    FlowMarch,
    RadiationSolve,
    RootSolve,
    Segregated,
    SolverSpec,
    ViscosityRamp,
)

__all__ = ["CaseSpec", "Numerics", "case_spec_from_mapping", "case_spec_to_mapping"]


@dataclasses.dataclass(frozen=True)
class Numerics:
    """The discretization choices every case makes, whatever its physics.

    Attributes
    ----------
    momentum_advection : AdvectionScheme
        How the momentum is advected. Required: a flow with no advection scheme is Stokes flow, which is
        a different problem rather than a default.
    gradient : GradientScheme or None
        How cell gradients are reconstructed, for every field of the case; unset,
        :data:`~aquaflux.schemes.DEFAULT_GRADIENT_SCHEME`.

    Raises
    ------
    TypeError
        If a setting is not a value of its family.
    """

    momentum_advection: AdvectionScheme
    gradient: GradientScheme | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.momentum_advection, AdvectionScheme):
            raise TypeError(
                f"Numerics.momentum_advection must be an AdvectionScheme, got {self.momentum_advection!r}."
            )
        if self.gradient is not None and not isinstance(self.gradient, GradientScheme):
            raise TypeError(f"Numerics.gradient must be a GradientScheme, got {self.gradient!r}.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class CaseSpec:
    """One case: its mesh, physics, boundary patches, and -- for a flow -- its fluid, numerics and drive.

    Attributes
    ----------
    mesh : MeshSource
        Where the mesh is read from.
    physics : Physics
        :class:`~aquaflux.case.Laminar`, :class:`~aquaflux.case.RANS` or
        :class:`~aquaflux.case.Radiation`.
    boundaries : mapping of {str: PatchCondition}
        What each boundary patch is, by the patch's name in the mesh or by the name of a patch group
        (every wall at once, say), whose condition then applies to each patch in it. Stored read-only,
        in the order given; :meth:`patch_conditions` gives the per-patch form.
    fluid : Fluid or None
        The fluid, stated once for every equation of a flow; a radiation case has none.
    numerics : Numerics or None
        The discretization choices common to every flow physics; a radiation case keeps its own inside
        its physics.
    drive : DriveSpec or None
        What drives the flow when its boundary conditions do not -- :class:`~aquaflux.case.BulkVelocity`,
        a bulk velocity held by a solved force. Unset, the boundary conditions and the sources do.
    sources : tuple of SourceSpec
        Terms added to the momentum balance -- :class:`~aquaflux.case.BodyForce`, a prescribed uniform
        force. Empty by default.
    pressure_datum : PressureDatum or None
        Where the pressure level is fixed in a domain no patch fixes it in -- a closed domain, such as
        a lid-driven cavity: a :class:`~aquaflux.flow.PinnedPoint`. Required exactly when no patch is
        an :class:`~aquaflux.case.Outlet`, and refused otherwise.
    solver : SolverSpec or None
        How the case is solved -- :class:`~aquaflux.case.CoupledMarch`, :class:`~aquaflux.case.FlowMarch`,
        :class:`~aquaflux.case.Segregated` or :class:`~aquaflux.case.RadiationSolve`. Unset, its
        physics' own with the library's settings (see :meth:`~aquaflux.case.CheckedCase.solve`).
    outputs : Outputs
        What a run writes and where (see :func:`~aquaflux.case.run_case`); unset, the fields as VTK and
        the log, in ``results/`` beside the case file.

    Raises
    ------
    TypeError
        If a section is not a value of its family.
    ValueError
        If there are no boundary patches, if the physics refuses one (a turbulence setting in a laminar
        case, an inlet with no inflow turbulence in a Reynolds-averaged one, a reflectance in either)
        or refuses a section (a flow with no fluid, a radiation case with one), or if a flow's pressure
        level is not fixed exactly once: a closed domain with no ``pressure_datum``, or a datum beside
        an outlet; if the solver cannot solve this physics or hold this drive; or if an output writes an
        OpenFOAM time directory for a mesh that is not an OpenFOAM one.
    """

    mesh: MeshSource
    physics: Physics
    boundaries: Mapping[str, PatchCondition]
    fluid: Fluid | None = None
    numerics: Numerics | None = None
    drive: DriveSpec | None = None
    pressure_datum: PressureDatum | None = None
    sources: tuple[SourceSpec, ...] = ()
    solver: SolverSpec | None = None
    outputs: Outputs = dataclasses.field(default_factory=Outputs)

    def __post_init__(self) -> None:
        for name, family in (
            ("mesh", MeshSource),
            ("physics", Physics),
            ("outputs", Outputs),
        ):
            if not isinstance(getattr(self, name), family):
                raise TypeError(
                    f"CaseSpec.{name} must be a {family.__name__}, got {getattr(self, name)!r}."
                )
        for name, family in (("fluid", Fluid), ("numerics", Numerics)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, family):
                raise TypeError(f"CaseSpec.{name} must be a {family.__name__}, got {value!r}.")
        if self.drive is not None and not isinstance(self.drive, DriveSpec):
            raise TypeError(
                f"CaseSpec.drive must be a drive such as BulkVelocity(target), got {self.drive!r}."
            )
        for source in self.sources:
            if not isinstance(source, SourceSpec):
                raise TypeError(
                    f"CaseSpec.sources holds momentum sources such as BodyForce(force), got {source!r}."
                )
        if self.pressure_datum is not None and not isinstance(self.pressure_datum, PressureDatum):
            raise TypeError(
                f"CaseSpec.pressure_datum must be a PressureDatum, got {self.pressure_datum!r}."
            )
        if self.solver is not None and not isinstance(self.solver, SolverSpec):
            raise TypeError(
                f"CaseSpec.solver must be a solver such as CoupledMarch(), got {self.solver!r}."
            )
        if not self.boundaries:
            raise ValueError("a case names at least one boundary patch.")
        for patch, condition in self.boundaries.items():
            if not isinstance(patch, str) or not isinstance(condition, PatchCondition):
                raise TypeError(
                    "CaseSpec.boundaries maps each patch name to an Inlet, Outlet, Wall or Lamp, got "
                    f"{patch!r}: {condition!r}."
                )
        # A copy, read-only: the spec is frozen, and a dict handed in would otherwise stay mutable
        # through the caller's reference.
        object.__setattr__(self, "boundaries", types.MappingProxyType(dict(self.boundaries)))
        self.physics.refuse_boundaries(self.boundaries)
        self.physics.refuse_sections(self)
        if self.solver is not None:
            self.solver.refuse_for(self.physics, self.drive)
        if not isinstance(self.mesh, OpenFOAMMesh) and any(
            isinstance(writer, OpenFOAMTime) for writer in self.outputs.fields
        ):
            raise ValueError(
                "outputs.fields: an OpenFOAMTime writes into an OpenFOAM case, whose mesh must be the "
                f"case's own, but the mesh is a {type(self.mesh).__name__}. Write the fields as Vtk."
            )

    def check_against(self, mesh: Mesh) -> None:
        """Refuse this case on ``mesh`` unless its patches fit it exactly.

        Every key must name a boundary patch of the mesh, or a patch group of them, and no patch may be
        reached by two keys; every boundary face must lie in a patch given a condition (a face nobody
        gave a condition would keep a zero face value, which is a boundary condition nobody chose); each
        patch's, the drive's and each source's settings must fit the mesh's dimension; a pressure
        datum's point must have one coordinate per dimension and lie within the mesh's bounding box; and
        the physics' own settings must fit the mesh (a radiation case's occluders and receivers name
        its walls). Needs the mesh's topology and node coordinates only, not its geometry.

        Parameters
        ----------
        mesh : Mesh
            The case's mesh, as read from :attr:`mesh`.

        Raises
        ------
        ValueError
            Listing every problem found, not only the first.
        """
        problems = []
        conditions, patch_problems = _patch_conditions(self.boundaries, mesh)
        problems.extend(patch_problems)
        uncovered = mesh.face_patches.uncovered_boundary_faces(conditions, mesh.face_cells)
        if uncovered:
            listed = ", ".join(
                f"{name!r} ({count} face{'' if count == 1 else 's'})"
                for name, count in uncovered.items()
            )
            problems.append(f"no condition is given for the boundary faces of {listed}")
        # Everything with a dimension of its own, each checked against the mesh's.
        refusals = [
            *(
                functools.partial(condition.refuse_for_dimension, mesh.dim, patch)
                for patch, condition in self.boundaries.items()
            ),
            *(
                []
                if self.drive is None
                else [functools.partial(self.drive.refuse_for_dimension, mesh.dim)]
            ),
            *(
                functools.partial(source.refuse_for_dimension, mesh.dim, index)
                for index, source in enumerate(self.sources)
            ),
        ]
        for refuse in refusals:
            try:
                refuse()
            except ValueError as error:
                problems.append(str(error))
        if isinstance(self.pressure_datum, PinnedPoint):
            problems.extend(_datum_misfits(self.pressure_datum, mesh))
        problems.extend(self.physics.mesh_misfits(self, mesh))
        if problems:
            raise ValueError("the case does not fit its mesh: " + "; ".join(problems) + ".")

    def patch_conditions(self, mesh: Mesh) -> dict[str, PatchCondition]:
        """Each boundary patch of ``mesh`` with its condition, a group's condition given to every member.

        A key of :attr:`boundaries` names a patch or a patch group of the mesh (an OpenFOAM ``boundary``
        file's ``inGroups``); a group's condition applies to each patch in it. This is the per-patch
        form every closure is built from.

        Parameters
        ----------
        mesh : Mesh
            The case's mesh.

        Returns
        -------
        dict of {str: PatchCondition}
            Patch name to condition, in the order the keys name them.

        Raises
        ------
        ValueError
            If a key names no patch or group, names one that holds a face not on the boundary, or is
            both a patch and a group of other patches, or if two keys reach the same patch -- the problems
            :meth:`check_against` reports.
        """
        conditions, problems = _patch_conditions(self.boundaries, mesh)
        if problems:
            raise ValueError(
                "the case's boundaries do not fit its mesh: " + "; ".join(problems) + "."
            )
        return conditions


def _patch_conditions(
    boundaries: Mapping[str, PatchCondition], mesh: Mesh
) -> tuple[dict[str, PatchCondition], list[str]]:
    """Resolve each key of ``boundaries`` to the patches it names, and say what does not resolve.

    Returns the conditions of the patches reached, by patch, and the problems found: a key that names
    nothing (or two different sets of faces), a patch reached that is not a boundary patch, and a patch
    reached by two keys -- every one of them, not only the first.
    """
    patches = mesh.face_patches
    conditions: dict[str, PatchCondition] = {}
    reached_by: dict[str, str] = {}
    unknown, ambiguous, not_boundary, twice = [], [], [], []
    for key, condition in boundaries.items():
        if key not in patches.names and key not in patches.group_names:
            unknown.append(key)
            continue
        try:
            members = patches.addressed_by(key)
        except ValueError as error:
            ambiguous.append(str(error))
            continue
        for patch in members:
            if patch in reached_by:
                twice.append(f"{patch!r} (by {reached_by[patch]!r} and {key!r})")
                continue
            reached_by[patch] = key
            conditions[patch] = condition
            if not patches.is_boundary_patch(patch, mesh.face_cells):
                not_boundary.append(repr(patch) if patch == key else f"{patch!r} (in {key!r})")
    problems = [*ambiguous]
    if unknown or not_boundary:
        if unknown:
            problems.append(f"the mesh has no patch {', '.join(map(repr, unknown))}")
        if not_boundary:
            problems.append(
                f"{', '.join(not_boundary)} {'is not a boundary patch' if len(not_boundary) == 1 else 'are not boundary patches'}"
            )
        available = sorted(
            name for name in patches.names if patches.is_boundary_patch(name, mesh.face_cells)
        )
        groups = (
            f", and its patch groups are {list(patches.group_names)}" if patches.group_names else ""
        )
        problems[-1] += f" (its boundary patches are {available}{groups})"
    if twice:
        problems.append(f"a patch is given a condition twice: {', '.join(twice)}")
    return conditions, problems


def _datum_misfits(datum: PinnedPoint, mesh: Mesh) -> list[str]:
    """How a pinned point fails to fit ``mesh``: a wrong dimension, or a place outside its bounding box.

    The bounding box is the cheap test available before any geometry: a point outside it is certainly
    outside the domain, and its nearest cell would be an arbitrary cell on the boundary. A point inside
    the box but outside a non-convex domain is not caught, and is harmless -- the datum only sets a
    level, so any cell may carry it.
    """
    point = datum.point
    if len(point) != mesh.dim:
        return [
            f"pressure_datum: the point {point!r} has {len(point)} coordinates, but the mesh is "
            f"{mesh.dim}-dimensional"
        ]
    nodes = np.asarray(mesh.node_coords)
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    if np.any(np.asarray(point) < low) or np.any(np.asarray(point) > high):
        return [
            f"pressure_datum: the point {point!r} lies outside the mesh, whose bounding box is "
            f"{tuple(low.tolist())} to {tuple(high.tolist())}"
        ]
    return []


#: Every value a case file may name, at any level. The schemes, the march settings and the
#: preconditioners are the library's own classes, read and written as they are -- the preconditioners
#: taken from the coupled preconditioner's own registry, so a kind added there reaches a case file the day
#: it is added; the boundary kinds and the solvers are case-file values that describe a patch for every
#: field and a solve for every setting.
_CASE_MAPPING = SettingsMapping(
    [
        CaseSpec,
        OpenFOAMMesh,
        StructuredGrid,
        GeometricGrading,
        Fluid,
        Laminar,
        RANS,
        Radiation,
        SSTModel,
        DirectScalars,
        LogScalars,
        Inlet,
        Outlet,
        Wall,
        Lamp,
        FixedTurbulence,
        IntensityLength,
        Numerics,
        FirstOrderUpwind,
        LimitedUpwind,
        VenkatakrishnanLimiter,
        CompactGreenGauss,
        CorrectedGreenGauss,
        SweptGradientSolve,
        GmresGradientSolve,
        MultipleCorrectionGradient,
        OwnerGradient,
        SkewCorrectedGradient,
        BulkVelocity,
        BodyForce,
        PinnedPoint,
        CoupledMarch,
        FlowMarch,
        Segregated,
        ViscosityRamp,
        RootSolve,
        Convergence,
        Euclidean,
        RowScaled,
        BlockScaled,
        DualTimeLoop,
        LinearSolveSettings,
        DualTimeControl,
        ResidualRatioDualTimeControl,
        CflResidualDualTimeControl,
        RetryPolicy,
        GmresSolve,
        DirectSolve,
        *PRECONDITIONER_SPEC_MAPPING.kinds,
        RadiationSolve,
        LambertianProfile,
        CosinePowerProfile,
        IesProfile,
        UniformMedium,
        Coarsen,
        CadPlacement,
        MeshPatch,
        StlSurface,
        CadSurface,
        PatchBody,
        StlBody,
        CadSolid,
        CadFluid,
        Receivers,
        RadiationSettings,
        NoOcclusion,
        RayCastOcclusion,
        SilhouetteOcclusion,
        ShaftCulling,
        EveryPair,
        Outputs,
        Vtk,
        PatchVtk,
        OpenFOAMTime,
        Checkpoints,
    ]
)

#: The sections with one form each, whose ``kind`` a file may leave out.
_ONE_FORM_SECTIONS = (("fluid", Fluid), ("numerics", Numerics), ("outputs", Outputs))

#: The case's own kind name. A case file does not write it: the whole document is the case.
_CASE_KIND = CaseSpec.__name__


def case_spec_from_mapping(mapping: Mapping[str, object]) -> CaseSpec:
    """Read a case from the nested mapping a case file parses to.

    The top level holds the sections -- ``mesh``, ``physics``, ``boundaries``, for a flow ``fluid`` and
    ``numerics``, and optionally ``drive``, ``sources``, ``pressure_datum``, ``solver`` and ``outputs``
    -- and names no ``kind``, since the whole document is the case. Below it,
    each value is a mapping whose ``kind`` names its class, except ``boundaries``, which maps each patch
    name to that patch's condition::

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

    The ``fluid``, ``numerics`` and ``outputs`` sections have one form each, so their ``kind`` may be
    left out.

    Parameters
    ----------
    mapping : mapping
        The case, as parsed from a case file.

    Returns
    -------
    CaseSpec
        The case, checked on its own terms but not yet against its mesh.

    Raises
    ------
    ValueError
        If a section or setting is unknown, missing, or of a form its position cannot hold -- the
        message gives the path to it -- or if the case is inconsistent (see :class:`CaseSpec`).
    TypeError
        If a value is of the wrong family for its position.
    """
    if not isinstance(mapping, Mapping):
        raise ValueError(f"a case is a mapping of its sections, got {mapping!r}.")
    if "kind" in mapping and mapping["kind"] != _CASE_KIND:
        raise ValueError(
            f"a case file's top level is the case itself and names no kind, got kind {mapping['kind']!r}."
        )
    sections = dict(mapping)
    for section, kind in _ONE_FORM_SECTIONS:
        if isinstance(sections.get(section), Mapping) and "kind" not in sections[section]:
            sections[section] = {"kind": kind.__name__, **sections[section]}
    return _CASE_MAPPING.from_mapping({**sections, "kind": _CASE_KIND})


def case_spec_to_mapping(spec: CaseSpec) -> dict[str, object]:
    """Write a case as the nested mapping a case file stores.

    The inverse of :func:`case_spec_from_mapping`: every setting left at its default is omitted, and
    reading the mapping back gives an equal case.

    Parameters
    ----------
    spec : CaseSpec
        The case to write.

    Returns
    -------
    dict
        Plain data -- mappings, lists, strings, numbers, booleans and ``None`` -- ready for a YAML
        writer.

    Raises
    ------
    TypeError
        If ``spec`` is not a :class:`CaseSpec`, or holds a value no case file can name (a scheme outside
        the ones listed for case files, say).
    """
    if not isinstance(spec, CaseSpec):
        raise TypeError(f"only a CaseSpec is written as a case, got {type(spec).__name__}.")
    mapping = _CASE_MAPPING.to_mapping(spec)
    del mapping["kind"]
    for section, _ in _ONE_FORM_SECTIONS:
        if section in mapping:
            del mapping[section]["kind"]
    return mapping
