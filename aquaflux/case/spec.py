"""A whole case as one frozen value: the mesh, the fluid, the physics, the boundaries, the numerics.

:class:`CaseSpec` is what a case file describes, and nothing more: plain settings, read and written by
:func:`case_spec_from_mapping` and :func:`case_spec_to_mapping` from the nested mapping a YAML document
parses to. It holds no mesh and builds no equation -- that is what makes checking a case cheap, and
what leaves the solver free to be rebuilt from whatever state a march has reached.

It is not a flat record of every setting any case might need. A small core -- mesh, fluid, boundaries,
numerics -- is common to every case; what differs between cases is decided by two discriminators, the
**physics** (:class:`~aquaflux.case.Laminar` or :class:`~aquaflux.case.RANS`) and the **drive** (what
sets the flow in motion). A setting that belongs to one physics lives inside it, so a case cannot carry
one its physics would ignore.

A spec that is constructed is already consistent on its own terms: its physics has accepted its
boundaries, and some patch fixes the pressure level. What it cannot know until its mesh is read -- that
every boundary face has a patch, and that each patch fits the mesh's dimension -- is checked by
:meth:`CaseSpec.check_against`.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping

from aquaflux.discretization import AdvectionScheme, FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import BoundaryDriven, Drive
from aquaflux.mesh import Mesh
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
from aquaflux.solve import SettingsMapping
from aquaflux.turbulence import DirectScalars, LogScalars, SSTModel

from .boundaries import FixedTurbulence, Inlet, Outlet, PatchCondition, Wall
from .fluid import Fluid
from .mesh_source import MeshSource, OpenFOAMMesh
from .physics import RANS, Laminar, Physics

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


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    """One case: its mesh, fluid, physics, boundary patches, numerics and drive.

    Attributes
    ----------
    mesh : MeshSource
        Where the mesh is read from.
    fluid : Fluid
        The fluid, stated once for every equation.
    physics : Physics
        :class:`~aquaflux.case.Laminar` or :class:`~aquaflux.case.RANS`.
    boundaries : mapping of {str: PatchCondition}
        What each boundary patch is, by the patch's name in the mesh. Stored read-only, in the order
        given.
    numerics : Numerics
        The discretization choices common to every physics.
    drive : Drive or None
        What sets the flow in motion; unset, :class:`~aquaflux.flow.BoundaryDriven` -- the boundary
        conditions do. A case file can name no other drive yet.

    Raises
    ------
    TypeError
        If a section is not a value of its family.
    ValueError
        If there are no boundary patches, if the physics refuses one (a turbulence setting in a laminar
        case, an inlet with no inflow turbulence in a Reynolds-averaged one), or if no patch fixes the
        pressure level. A domain whose pressure level is free needs a datum, which a case file cannot
        yet state; such a case is built in code, with ``MomentumContinuity.build(pressure_pin=...)``.
    """

    mesh: MeshSource
    fluid: Fluid
    physics: Physics
    boundaries: Mapping[str, PatchCondition]
    numerics: Numerics
    drive: Drive | None = None

    def __post_init__(self) -> None:
        for name, family in (
            ("mesh", MeshSource),
            ("fluid", Fluid),
            ("physics", Physics),
            ("numerics", Numerics),
        ):
            if not isinstance(getattr(self, name), family):
                raise TypeError(
                    f"CaseSpec.{name} must be a {family.__name__}, got {getattr(self, name)!r}."
                )
        if self.drive is not None and not isinstance(self.drive, Drive):
            raise TypeError(f"CaseSpec.drive must be a Drive, got {self.drive!r}.")
        if not self.boundaries:
            raise ValueError("a case names at least one boundary patch.")
        for patch, condition in self.boundaries.items():
            if not isinstance(patch, str) or not isinstance(condition, PatchCondition):
                raise TypeError(
                    "CaseSpec.boundaries maps each patch name to an Inlet, Outlet or Wall, got "
                    f"{patch!r}: {condition!r}."
                )
        # A copy, read-only: the spec is frozen, and a dict handed in would otherwise stay mutable
        # through the caller's reference.
        object.__setattr__(self, "boundaries", types.MappingProxyType(dict(self.boundaries)))
        self.physics.refuse_boundaries(self.boundaries)
        if not any(condition.prescribes_pressure() for condition in self.boundaries.values()):
            raise ValueError(
                "no boundary patch fixes the pressure (none is an Outlet), so the pressure level is free "
                "and the case needs a datum -- which a case file cannot state yet. Build a closed domain "
                "in code, with MomentumContinuity.build(pressure_pin=...)."
            )

    def check_against(self, mesh: Mesh) -> None:
        """Refuse this case on ``mesh`` unless its patches fit it exactly.

        Every patch named must be a boundary patch of the mesh; every boundary face must lie in a named
        patch (a face nobody gave a condition would keep a zero face value, which is a boundary condition
        nobody chose); and each patch's settings must fit the mesh's dimension. Needs the mesh's topology
        only, not its geometry.

        Parameters
        ----------
        mesh : Mesh
            The case's mesh, as read from :attr:`mesh`.

        Raises
        ------
        ValueError
            Listing every problem found, not only the first.
        """
        patches = mesh.face_patches
        problems = []
        unknown = [name for name in self.boundaries if name not in patches.names]
        not_boundary = [
            name
            for name in self.boundaries
            if name in patches.names and not patches.is_boundary_patch(name, mesh.face_cells)
        ]
        known = [name for name in self.boundaries if name in patches.names]
        uncovered = patches.uncovered_boundary_faces(known, mesh.face_cells)
        if unknown or not_boundary:
            available = sorted(
                name for name in patches.names if patches.is_boundary_patch(name, mesh.face_cells)
            )
            if unknown:
                problems.append(f"the mesh has no patch {', '.join(map(repr, unknown))}")
            if not_boundary:
                problems.append(
                    f"{', '.join(map(repr, not_boundary))} {'is not a boundary patch' if len(not_boundary) == 1 else 'are not boundary patches'}"
                )
            problems[-1] += f" (its boundary patches are {available})"
        if uncovered:
            listed = ", ".join(
                f"{name!r} ({count} face{'' if count == 1 else 's'})"
                for name, count in uncovered.items()
            )
            problems.append(f"no condition is given for the boundary faces of {listed}")
        for patch, condition in self.boundaries.items():
            try:
                condition.refuse_for_dimension(mesh.dim, patch)
            except ValueError as error:
                problems.append(str(error))
        if problems:
            raise ValueError("the case does not fit its mesh: " + "; ".join(problems) + ".")


#: Every value a case file may name, at any level. The schemes are the library's own classes, read and
#: written as they are; the boundary kinds are case-file values that describe a patch for every field.
_CASE_MAPPING = SettingsMapping(
    [
        CaseSpec,
        OpenFOAMMesh,
        Fluid,
        Laminar,
        RANS,
        SSTModel,
        DirectScalars,
        LogScalars,
        Inlet,
        Outlet,
        Wall,
        FixedTurbulence,
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
        BoundaryDriven,
    ]
)

#: The case's own kind name. A case file does not write it: the whole document is the case.
_CASE_KIND = CaseSpec.__name__


def case_spec_from_mapping(mapping: Mapping[str, object]) -> CaseSpec:
    """Read a case from the nested mapping a case file parses to.

    The top level holds the sections -- ``mesh``, ``fluid``, ``physics``, ``boundaries``, ``numerics``
    and optionally ``drive`` -- and names no ``kind``, since the whole document is the case. Below it,
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

    The ``fluid`` and ``numerics`` sections have one form each, so their ``kind`` may be left out.

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
    for section, kind in (("fluid", Fluid), ("numerics", Numerics)):
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
    for section in ("fluid", "numerics"):
        del mapping[section]["kind"]
    return mapping
