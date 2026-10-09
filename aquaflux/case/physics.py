"""Which equations a case solves: laminar flow, Reynolds-averaged flow with a turbulence closure, or
the light of lamps.

The physics is one of a case's two discriminators (the other is its drive). It decides which settings
a case may carry at all: everything that belongs to a turbulence closure -- its model constants, how
its variables are parametrized, how its fields are advected, the inflow turbulence at an inlet -- lives
inside :class:`RANS` or on a boundary patch, and a :class:`Laminar` case refuses any of it rather than
ignoring it. It decides which of the case's sections there are too: a flow states its fluid and its
numerics, while a :class:`Radiation` case has neither, and keeps its medium and its numerics inside
itself.

Each physics also builds its own problem from a checked case (:meth:`Physics.build`): a laminar case
is the flow assembler, a Reynolds-averaged one the coupled system holding the flow and the closure, a
radiation case the scene its lamps light. What comes back is the problem the initializers and the
solves already take, so nothing downstream needs a case-specific type.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
import numpy as np

from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import MomentumContinuity, refuse_an_unsuitable_pressure_datum, sheared_patches
from aquaflux.mesh import patch_triangles
from aquaflux.radiation import (
    RadiationSettings,
    Scene,
    SceneSolution,
    SurfaceReceivers,
    Surfaces,
    VolumeReceivers,
    lamp_exitance,
    refine_for_receivers,
)
from aquaflux.turbulence import (
    CoupledRANS,
    ScalarVariableTransform,
    SSTModel,
    SSTTurbulence,
    coupled_fields,
)

from .boundaries import Lamp, PatchCondition, Wall
from .radiation import MeshPatch, OccluderSpec, PatchSurface, Receivers, UniformMedium, _Drawings
from .scopes import FLOW, RADIATION, SCOPES, TURBULENCE

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh, MeshGeometry

    from .spec import CaseSpec

__all__ = ["RANS", "Laminar", "Physics", "Radiation"]


@dataclasses.dataclass(frozen=True)
class Physics(abc.ABC):
    """The equations a case solves: :class:`Laminar`, :class:`RANS` or :class:`Radiation`.

    Attributes
    ----------
    reads_scopes : tuple of str
        A class attribute: the scopes of setting this physics reads (see :mod:`.scopes`). A setting
        of any other scope is refused wherever a case states it.
    state_fields : tuple of str
        A class attribute: the physical fields a solve of this physics starts from -- what
        :meth:`initial_fields` needs and :meth:`restart_fields` gives. Empty for a physics that
        marches nothing.
    """

    reads_scopes: ClassVar[tuple[str, ...]] = ()
    state_fields: ClassVar[tuple[str, ...]] = ()

    @abc.abstractmethod
    def refuse_sections(self, spec: CaseSpec) -> None:
        """Refuse a case whose sections this physics cannot use as written.

        Which of the case's sections there must be, and which there must not, is the physics'
        knowledge: a flow states its fluid, a radiation case has none to state.

        Parameters
        ----------
        spec : CaseSpec
            The case, with its boundaries already accepted by :meth:`refuse_boundaries`.

        Raises
        ------
        ValueError
            Naming the sections that are missing or that nothing would read.
        """

    def mesh_misfits(self, spec: CaseSpec, mesh: Mesh) -> list[str]:
        """How this physics' own settings fail to fit ``mesh``; nothing, by default.

        Parameters
        ----------
        spec : CaseSpec
            The case.
        mesh : Mesh
            Its mesh, topology only.

        Returns
        -------
        list of str
            One entry per problem, for :meth:`~aquaflux.case.CaseSpec.check_against` to report with
            the others.
        """
        del spec, mesh
        return []

    @abc.abstractmethod
    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse boundary patches this physics cannot use as written.

        Parameters
        ----------
        boundaries : mapping of {str: PatchCondition}
            The case's patches, by name.

        Raises
        ------
        ValueError
            Naming every offending setting at once, by its path in the case file.
        """

    @abc.abstractmethod
    def build(self, spec: CaseSpec, mesh: Mesh, geometry: MeshGeometry, directory: Path) -> object:
        """The problem ``spec`` describes, assembled on ``mesh``.

        Parameters
        ----------
        spec : CaseSpec
            The case, already checked against ``mesh``.
        mesh : Mesh
            Its mesh.
        geometry : MeshGeometry
            The mesh's geometry, computed once and shared by every equation.
        directory : pathlib.Path
            The directory the case file sits in, which the files it names are relative to.

        Returns
        -------
        object
            The problem's assembler, in the form its initializer and its solve take.
        """

    @abc.abstractmethod
    def output_fields(self, problem: object, solution: object) -> dict[str, np.ndarray]:
        """The converged fields a run writes, by name.

        Parameters
        ----------
        problem : object
            What :meth:`build` returned.
        solution : object
            What the case's solve returned for it.

        Returns
        -------
        dict of {str: np.ndarray}
            The physical fields -- a vector ``(n_cells, dim)``, a scalar ``(n_cells,)`` -- under the
            names a file's field writers select by. The pressure is the solved one, not re-based.
        """

    def output_patch_fields(
        self, problem: object, solution: object
    ) -> dict[str, dict[str, np.ndarray]]:
        """The fields a run writes on boundary patches, by patch and then by name; none by default.

        Parameters
        ----------
        problem : object
            What :meth:`build` returned.
        solution : object
            What the case's solve returned for it.

        Returns
        -------
        dict of {str: dict of {str: np.ndarray}}
            Per patch, its face fields -- ``(n_faces,)`` for a scalar -- in the patch's own face order.
        """
        del problem, solution
        return {}

    def results(self, problem: object, solution: object) -> dict[str, object]:
        """The scalar results a run records beside its fields; none by default.

        Parameters
        ----------
        problem : object
            What :meth:`build` returned.
        solution : object
            What the case's solve returned for it.

        Returns
        -------
        dict
            Plain data -- numbers, strings and mappings of them -- ready for a YAML writer.
        """
        del problem, solution
        return {}

    def restart_fields(self, problem: object, state: object) -> dict[str, np.ndarray]:
        """The physical fields of a march state, by name: what a checkpoint holds.

        A physics with no march to checkpoint refuses; the flow physics override this.

        Parameters
        ----------
        problem : object
            What :meth:`build` returned.
        state : object
            A state of the march, as its ``on_checkpoint`` observer is handed it -- the march's
            solved variables, which need not be the physical ones.

        Returns
        -------
        dict of {str: np.ndarray}
            The fields a solve of this physics can start from -- ``U`` and ``p``, and under RANS ``k``
            and ``omega`` -- in the form :meth:`initial_fields` takes back. The pressure is the solved
            one.

        Raises
        ------
        ValueError
            If this physics marches nothing.
        """
        raise ValueError(f"a {type(self).__name__} case has no march state to save.")

    def initial_fields(self, problem: object, fields: Mapping[str, np.ndarray]) -> object:
        """The starting state a solve takes, from physical fields: the inverse of :meth:`restart_fields`.

        A physics with nothing to start from refuses; the flow physics override this.

        Parameters
        ----------
        problem : object
            What :meth:`build` returned.
        fields : mapping of {str: np.ndarray}
            Physical fields by name -- ``U`` ``(n_cells, dim)``, ``p`` ``(n_cells,)``, and under RANS
            ``k`` and ``omega`` ``(n_cells,)`` -- on the problem's mesh.

        Returns
        -------
        object
            What the case's solve takes as its starting state: the flow state for a laminar case, the
            tuple ``(flow, k, omega)`` for a Reynolds-averaged one.

        Raises
        ------
        ValueError
            If a field this physics needs is missing, naming it and the fields given; or if this
            physics has no state to start from.
        """
        raise ValueError(f"a {type(self).__name__} case has no state to start from.")

    @abc.abstractmethod
    def progress_fields(self, problem: object) -> Callable[[object], Mapping[str, object]] | None:
        """What a march's log reports the change of at each step, or ``None`` for nothing.

        Parameters
        ----------
        problem : object
            What :meth:`build` returned.

        Returns
        -------
        callable or None
            ``state -> {name: field}``, over the state the march iterates on.
        """


@dataclasses.dataclass(frozen=True)
class _Flow(Physics):
    """What every flow physics shares: it states its fluid and its numerics, and fixes its pressure level."""

    def refuse_sections(self, spec: CaseSpec) -> None:
        """Require the fluid and the numerics, and the pressure level fixed exactly once.

        Which closure fixes the pressure level is the flow's knowledge, asked of the closures the patches
        build, so it is not restated for the case: by an outlet, or in a closed domain by the case's
        ``pressure_datum``.
        """
        missing = [name for name in ("fluid", "numerics") if getattr(spec, name) is None]
        if missing:
            raise ValueError(
                f"{' and '.join(missing)}: a {type(self).__name__} case states its fluid and its "
                "numerics; give "
                f"{'it' if len(missing) == 1 else 'them'}."
            )
        numerics = spec.numerics
        stray = [
            f"numerics.{name}"
            for scope in SCOPES
            if scope not in self.reads_scopes
            for name in numerics.settings_in(scope)
        ]
        if stray:
            raise ValueError(
                f"{', '.join(stray)}: a {type(self).__name__} case does not read "
                f"{'this setting' if len(stray) == 1 else 'these settings'}; remove "
                f"{'it' if len(stray) == 1 else 'them'}."
            )
        needed = [
            f"numerics.{name}" for scope in self.reads_scopes for name in numerics.missing_in(scope)
        ]
        if needed:
            raise ValueError(
                f"{', '.join(needed)}: a {type(self).__name__} case needs "
                f"{'this setting' if len(needed) == 1 else 'these settings'}; give "
                f"{'it' if len(needed) == 1 else 'them'}."
            )
        refuse_an_unsuitable_pressure_datum(
            BoundaryConditions(
                {name: condition.flow_closure() for name, condition in spec.boundaries.items()}
            ),
            spec.pressure_datum,
            "the case",
        )


def _refuse_stray(
    boundaries: Mapping[str, PatchCondition], scope: str, case: str, remedy: str
) -> None:
    """Refuse every patch setting of one scope at once."""
    stray = [
        f"boundaries.{patch}.{setting}"
        for patch, condition in boundaries.items()
        for setting in condition.settings_in(scope)
    ]
    if stray:
        one = len(stray) == 1
        raise ValueError(
            f"{', '.join(stray)}: {case}, so nothing would read "
            f"{'this setting' if one else 'these settings'}. "
            + remedy.format(it="it" if one else "them")
        )


def _refuse_light(boundaries: Mapping[str, PatchCondition]) -> None:
    """Refuse a reflectance or a lamp in a flow case -- nothing in it would read either."""
    _refuse_stray(
        boundaries,
        RADIATION,
        "a flow case gathers no light",
        "Remove {it}, or make the physics Radiation.",
    )


@dataclasses.dataclass(frozen=True)
class Laminar(_Flow):
    """Laminar incompressible flow: momentum and continuity, with no turbulence closure."""

    #: The settings this physics reads, by scope (see :mod:`.scopes`).
    reads_scopes: ClassVar[tuple[str, ...]] = (FLOW,)
    state_fields: ClassVar[tuple[str, ...]] = ("U", "p")

    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse any turbulence setting on a patch -- nothing in a laminar case would read it -- and any
        light."""
        _refuse_light(boundaries)
        _refuse_stray(
            boundaries,
            TURBULENCE,
            "a laminar case has no turbulence closure",
            "Remove {it}, or make the physics RANS.",
        )

    def build(
        self, spec: CaseSpec, mesh: Mesh, geometry: MeshGeometry, directory: Path
    ) -> MomentumContinuity:
        """The flow assembler -- see :meth:`Physics.build`."""
        del directory
        return _momentum(spec, mesh, geometry)

    def output_fields(self, problem: MomentumContinuity, solution: object) -> dict[str, np.ndarray]:
        """``U`` and ``p`` -- see :meth:`Physics.output_fields`."""
        velocity, pressure = problem.unpack(solution)
        return {"U": np.asarray(velocity), "p": np.asarray(pressure)}

    def restart_fields(self, problem: MomentumContinuity, state: object) -> dict[str, np.ndarray]:
        """``U`` and ``p`` -- see :meth:`Physics.restart_fields`. A laminar march's state is the flow's own."""
        return self.output_fields(problem, state)

    def initial_fields(
        self, problem: MomentumContinuity, fields: Mapping[str, np.ndarray]
    ) -> jnp.ndarray:
        """The flow state of ``U`` and ``p`` -- see :meth:`Physics.initial_fields`."""
        velocity, pressure = _required(fields, self.state_fields)
        return problem.pack(jnp.asarray(velocity), jnp.asarray(pressure))

    def progress_fields(self, problem: MomentumContinuity) -> None:
        """None: the log reports the residual alone -- see :meth:`Physics.progress_fields`."""
        del problem


@dataclasses.dataclass(frozen=True)
class RANS(_Flow):
    """Reynolds-averaged (RANS) incompressible flow closed by the k-omega shear-stress transport (SST) model.

    The flow and the closure's ``k`` and ``omega`` are solved together, as one coupled system. How
    ``k`` and ``omega`` are advected is the case's ``numerics.turbulence_advection``.

    Attributes
    ----------
    model : SSTModel
        The model's constants. Unset, the standard ones (:class:`~aquaflux.turbulence.SSTModel` with
        its own defaults).
    k_variable, omega_variable : ScalarVariableTransform or None
        The variable each field is solved in. :class:`~aquaflux.turbulence.DirectScalars` solves for
        the field itself and :class:`~aquaflux.turbulence.LogScalars` for its logarithm, which keeps
        it positive under any Newton step. Unset, the coupled system's default.
    explicit_production_limiter : bool or None
        Freeze the ``k``-production cap in the linearization; unset, the exact operator. See
        :class:`~aquaflux.turbulence.SSTTurbulence` for when this is safe.

    Raises
    ------
    TypeError
        If a setting is not a value of its family.
    """

    model: SSTModel = dataclasses.field(default_factory=SSTModel)
    k_variable: ScalarVariableTransform | None = None
    omega_variable: ScalarVariableTransform | None = None
    explicit_production_limiter: bool | None = None

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (SSTTurbulence.build,)
    #: The settings this physics reads, by scope (see :mod:`.scopes`).
    reads_scopes: ClassVar[tuple[str, ...]] = (FLOW, TURBULENCE)
    state_fields: ClassVar[tuple[str, ...]] = ("U", "p", "k", "omega")

    def __post_init__(self) -> None:
        for name, family in (
            ("model", SSTModel),
            ("k_variable", ScalarVariableTransform),
            ("omega_variable", ScalarVariableTransform),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, family):
                raise TypeError(f"RANS.{name} must be a {family.__name__}, got {value!r}.")

    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse a patch missing a setting the closure needs -- the inflow turbulence at an inlet -- and
        any light."""
        _refuse_light(boundaries)
        missing = [
            f"boundaries.{patch}.{setting}"
            for patch, condition in boundaries.items()
            for setting in condition.missing_in(TURBULENCE)
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)}: a RANS case needs the turbulence every inflow carries in. Give "
                "it, e.g. turbulence: {kind: FixedTurbulence, k: ..., omega: ...}."
            )

    def build(
        self, spec: CaseSpec, mesh: Mesh, geometry: MeshGeometry, directory: Path
    ) -> CoupledRANS:
        """The coupled flow and closure -- see :meth:`Physics.build`.

        The closure's walls are the patches whose flow closure is a wall, and its ``k`` and ``omega``
        closures are each patch's own, so neither is stated a second time. Both equations read the one
        property model the fluid gives, and the same gradient reconstruction.
        """
        del directory
        momentum = _momentum(spec, mesh, geometry)
        model = self.model
        # The model is the one the closure is built with, so an inlet's length scale reads its constants.
        closures = {
            name: condition.turbulence_closures(model)
            for name, condition in spec.patch_conditions(mesh).items()
        }
        options = _set(
            gradient_scheme=spec.numerics.gradient,
            explicit_production_limiter=self.explicit_production_limiter,
        )
        turbulence = SSTTurbulence.build(
            model,
            mesh,
            geometry,
            spec.numerics.turbulence_advection,
            momentum.properties,
            wall_patches=list(sheared_patches(momentum.boundary)),
            k_boundary=BoundaryConditions({name: k for name, (k, _) in closures.items()}),
            omega_boundary=BoundaryConditions(
                {name: omega for name, (_, omega) in closures.items()}
            ),
            **options,
        )
        return CoupledRANS.build(
            momentum, turbulence, k_transform=self.k_variable, omega_transform=self.omega_variable
        )

    def output_fields(self, problem: CoupledRANS, solution: object) -> dict[str, np.ndarray]:
        """``U``, ``p``, ``k``, ``omega`` and the eddy viscosity ``nut`` -- see :meth:`Physics.output_fields`."""
        flow, k, omega = solution
        momentum = problem.momentum
        nut = problem.turbulence.closure_fields(momentum.velocity_fields(flow), k, omega).nu_t
        return {**self._solved_fields(problem, solution), "nut": np.asarray(nut)}

    def restart_fields(self, problem: CoupledRANS, state: object) -> dict[str, np.ndarray]:
        """``U``, ``p``, ``k`` and ``omega`` -- see :meth:`Physics.restart_fields`.

        The march's state holds the solved variables, so ``omega`` is mapped back out of its logarithm
        when the case solves for that.
        """
        return self._solved_fields(problem, problem.physical_fields(state))

    def initial_fields(
        self, problem: CoupledRANS, fields: Mapping[str, np.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """``(flow, k, omega)`` from the physical fields -- see :meth:`Physics.initial_fields`."""
        velocity, pressure, k, omega = _required(fields, self.state_fields)
        flow = problem.momentum.pack(jnp.asarray(velocity), jnp.asarray(pressure))
        return flow, jnp.asarray(k), jnp.asarray(omega)

    @staticmethod
    def _solved_fields(problem: CoupledRANS, solution: object) -> dict[str, np.ndarray]:
        """The fields the closure is solved for, from ``(flow, k, omega)``: everything but ``nut``."""
        flow, k, omega = solution
        velocity, pressure = problem.momentum.unpack(flow)
        return {
            "U": np.asarray(velocity),
            "p": np.asarray(pressure),
            "k": np.asarray(k),
            "omega": np.asarray(omega),
        }

    def progress_fields(self, problem: CoupledRANS) -> Callable[[object], Mapping[str, object]]:
        """The physical velocity components, pressure, ``k``, ``omega`` and ``nu_t`` -- see :meth:`Physics.progress_fields`."""
        return coupled_fields(problem)


@dataclasses.dataclass(frozen=True)
class Radiation(Physics):
    """The light of lamps, gathered in the medium and on the walls: no flow.

    The patches are :class:`~aquaflux.case.Lamp` and :class:`~aquaflux.case.Wall` -- a lamp emits, a
    wall is black unless it is given a reflectance -- and the case states neither a fluid nor numerics:
    the medium and the radiation's own numerics are here. What is solved is a
    :class:`~aquaflux.radiation.Scene` (see :func:`~aquaflux.radiation.solve_scene`): the lamps'
    direct light with their own profiles, and what the reflecting walls send back, closed over every
    bounce.

    Attributes
    ----------
    medium : UniformMedium or None
        The medium the light crosses; unset, it absorbs nothing (air at most ultraviolet wavelengths
        over a room's distances is close to this).
    occluders : tuple of OccluderSpec
        What shadows the light: :class:`~aquaflux.case.CadSolid`, :class:`~aquaflux.case.CadFluid`,
        :class:`~aquaflux.case.StlBody` or :class:`~aquaflux.case.PatchBody`. The walls of a domain that
        is not convex shadow it too, and must be named here to do so.
    receivers : Receivers
        Where the light is gathered; unset, at every cell centre and on every wall that is not part
        of a :class:`~aquaflux.case.PatchBody`.
    lamp_refinement : float or None
        Split each lamp facet until its longest edge is at most this fraction of its distance to the
        nearest point the light is gathered at (:func:`~aquaflux.radiation.refine_for_receivers`); unset,
        the lamps are used as their geometry gives them.
    lamp_samples : int or None
        Points along each edge of a reflecting facet at which the lamps' light on it is averaged; unset,
        :data:`~aquaflux.radiation.DEFAULT_LAMP_SAMPLES`.
    settings : RadiationSettings or None
        Build-time choices that are not physics: how the surfaces shadow themselves and the points they
        light, how the bodies' shadows are worked out, how many pairs a pass may form. Unset, the
        library's own for every one.

    Raises
    ------
    TypeError
        If a setting is not a value of its family.
    ValueError
        If ``lamp_refinement`` is not positive, or ``lamp_samples`` is less than one.
    """

    medium: UniformMedium | None = None
    occluders: tuple[OccluderSpec, ...] = ()
    receivers: Receivers = dataclasses.field(default_factory=Receivers)
    lamp_refinement: float | None = None
    lamp_samples: int | None = None
    settings: RadiationSettings | None = None

    #: Where an unset setting takes its default from (read by the case-file schema).
    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (Scene,)
    #: The settings for which unset means the feature is off (read by the case-file schema).
    unset_means_off: ClassVar[tuple[str, ...]] = (
        "medium",
        "lamp_refinement",
    )

    #: The settings this physics reads, by scope (see :mod:`.scopes`).
    reads_scopes: ClassVar[tuple[str, ...]] = (RADIATION,)

    def __post_init__(self) -> None:
        for name, family in (("medium", UniformMedium), ("settings", RadiationSettings)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, family):
                raise TypeError(f"Radiation.{name} must be a {family.__name__}, got {value!r}.")
        if not isinstance(self.receivers, Receivers):
            raise TypeError(f"Radiation.receivers must be Receivers, got {self.receivers!r}.")
        for occluder in self.occluders:
            if not isinstance(occluder, OccluderSpec):
                raise TypeError(
                    f"Radiation.occluders holds bodies such as PatchBody(patches), got {occluder!r}."
                )
        if self.lamp_refinement is not None and not self.lamp_refinement > 0:
            raise ValueError(
                f"Radiation.lamp_refinement must be positive, got {self.lamp_refinement!r}."
            )
        if self.lamp_samples is not None and self.lamp_samples < 1:
            raise ValueError(f"Radiation.lamp_samples must be >= 1, got {self.lamp_samples!r}.")

    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse what only a flow reads -- an inlet, an outlet, a wall's velocity or ``k`` -- and a case
        with no lamp."""
        for settings, case in (
            (FLOW, "a radiation case has no flow"),
            (TURBULENCE, "a radiation case has no turbulence closure"),
        ):
            _refuse_stray(
                boundaries,
                settings,
                case,
                "Remove {it}; a surface that only absorbs light is a Wall with no reflectance.",
            )
        if not any(isinstance(condition, Lamp) for condition in boundaries.values()):
            raise ValueError(
                "boundaries: a radiation case is lit by its lamps, but no patch is a Lamp."
            )

    def refuse_sections(self, spec: CaseSpec) -> None:
        """Refuse a fluid, numerics, a drive, sources, a pressure datum or a starting state: nothing here reads them."""
        stray = [
            name
            for name in ("fluid", "numerics", "drive", "pressure_datum", "initial")
            if getattr(spec, name) is not None
        ] + (["sources"] if spec.sources else [])
        if stray:
            raise ValueError(
                f"{', '.join(stray)}: a radiation case has no flow, so nothing would read "
                f"{'this section' if len(stray) == 1 else 'these sections'}. Its medium and numerics "
                "go in the physics section."
            )
        if self.lamp_refinement is not None and not (
            self.receivers.cells or self.receivers.patches != ()
        ):
            raise ValueError(
                "physics.lamp_refinement refines the lamps against the points the light is gathered "
                "at, and the receivers name none."
            )

    def mesh_misfits(self, spec: CaseSpec, mesh: Mesh) -> list[str]:
        """Patches named by the occluders or the receivers that the mesh does not have as walls, and a
        surface read from a file given to a patch group.

        An occluder's patches must be boundary patches of the mesh; a receiving patch must be one the
        case calls a :class:`~aquaflux.case.Wall` -- a lamp's own faces lie on the emitting surface --
        and not part of a :class:`~aquaflux.case.PatchBody`, whose faces lie on the body. A surface
        read from a file is one patch's, so it cannot be given under the name of a group of several.
        """
        patches = mesh.face_patches
        problems = []
        # A surface read from a file is one surface, the patch's own; under a patch group's name it
        # would be handed to every member, each then emitting or reflecting all of it.
        for key, condition in spec.boundaries.items():
            source = getattr(condition, "geometry", None)
            if source is None or isinstance(source, MeshPatch) or key in patches.names:
                continue
            if key in patches.group_names and len(patches.addressed_by(key)) > 1:
                problems.append(
                    f"boundaries.{key}.geometry: {key!r} is a group of "
                    f"{len(patches.addressed_by(key))} patches, and a {type(source).__name__} is one "
                    "surface, which each of them would take whole; name its patches one by one"
                )
        for index, occluder in enumerate(self.occluders):
            unknown = [
                name
                for name in occluder.mesh_patches()
                if name not in patches.names or not patches.is_boundary_patch(name, mesh.face_cells)
            ]
            if unknown:
                problems.append(
                    f"physics.occluders[{index}]: {', '.join(map(repr, unknown))} "
                    f"{'is not a boundary patch' if len(unknown) == 1 else 'are not boundary patches'} "
                    "of the mesh"
                )
        if self.receivers.patches is not None:
            try:
                conditions = spec.patch_conditions(mesh)
            except ValueError:
                return problems  # the case's own check reports why its patches do not resolve
            in_bodies = {name for occluder in self.occluders for name in occluder.mesh_patches()}
            for name in self.receivers.patches:
                if not isinstance(conditions.get(name), Wall):
                    problems.append(
                        f"physics.receivers.patches: {name!r} is not a wall of the case, and light is "
                        "gathered on walls"
                    )
                elif name in in_bodies:
                    problems.append(
                        f"physics.receivers.patches: {name!r} is part of an occluding body, whose faces "
                        "lie on the body itself"
                    )
        return problems

    def receiving_patches(self, conditions: Mapping[str, PatchCondition]) -> tuple[str, ...]:
        """The patches the irradiance is gathered on: :attr:`Receivers.patches`, or every wall not in a body.

        Parameters
        ----------
        conditions : mapping of {str: PatchCondition}
            The case's conditions, per patch.
        """
        if self.receivers.patches is not None:
            return self.receivers.patches
        in_bodies = {name for occluder in self.occluders for name in occluder.mesh_patches()}
        return tuple(
            name
            for name, condition in conditions.items()
            if isinstance(condition, Wall) and name not in in_bodies
        )

    def build(self, spec: CaseSpec, mesh: Mesh, geometry: MeshGeometry, directory: Path) -> Scene:
        """The scene the lamps light -- see :meth:`Physics.build`.

        Every lamp is one body of the lamps' surface set, named by its patch, emitting its power evenly
        over its own triangulated area with its own profile; every reflecting wall one body of the
        reflectors', named likewise. The points the light is gathered at are the cell centres and the
        centres of the receiving patches' faces, each facing into the domain.
        """
        conditions = spec.patch_conditions(mesh)
        drawings = _Drawings(directory)
        face = geometry.face
        centroid, normal = np.asarray(face.centroid), np.asarray(face.normal)

        def surface(name: str) -> PatchSurface:
            faces = np.asarray(mesh.face_patches.indices(name))
            return PatchSurface(
                name=name,
                triangles=patch_triangles(mesh, geometry, [name]).vertices,
                centres=centroid[faces],
                # A boundary face's stored normal points out of the domain, owner-outward.
                inward=-normal[faces],
            )

        def triangles_of(name: str, source) -> np.ndarray:
            return (MeshPatch() if source is None else source).triangles(
                surface(name), directory, drawings
            )

        lamps = _lamps(
            {name: c for name, c in conditions.items() if isinstance(c, Lamp)},
            triangles_of,
            directory,
        )
        volume = (
            VolumeReceivers(
                points=np.asarray(geometry.cell.centroid), volumes=np.asarray(geometry.cell.volume)
            )
            if self.receivers.cells
            else None
        )
        surfaces = {}
        for name in self.receiving_patches(conditions):
            faces = np.asarray(mesh.face_patches.indices(name))
            wall = conditions[name]
            surfaces[name] = SurfaceReceivers(
                points=centroid[faces],
                normals=-normal[faces],
                areas=np.asarray(face.area)[faces],
                reflectance=wall.reflectance or 0.0,
                reflector=name if wall.reflects else None,
            )
        if self.lamp_refinement is not None:
            points = [r.points for r in surfaces.values()]
            if volume is not None:
                points.append(volume.points)
            lamps, _ = refine_for_receivers(
                lamps, np.concatenate(points), max_ratio=self.lamp_refinement
            )
        reflecting = {
            name: c for name, c in conditions.items() if isinstance(c, Wall) and c.reflects
        }
        reflectors = None
        if reflecting:
            pieces = [triangles_of(name, wall.geometry) for name, wall in reflecting.items()]
            solid_id = np.repeat(np.arange(len(pieces)), [len(piece) for piece in pieces])
            reflectors = Surfaces.from_triangles(
                np.concatenate(pieces),
                solid_id=solid_id,
                solid_names=tuple(reflecting),
                diffuse_reflectance=np.asarray([w.reflectance for w in reflecting.values()])[
                    solid_id
                ],
            )
        bodies = tuple(
            occluder.body(
                directory,
                drawings,
                lambda names, **options: patch_triangles(mesh, geometry, names, **options).vertices,
            )
            for occluder in self.occluders
        )
        return Scene(
            lamps=lamps,
            reflectors=reflectors,
            occluders=bodies,
            absorption=None if self.medium is None else self.medium.absorption_model(),
            volume=volume,
            surfaces=surfaces,
            **_set(lamp_samples=self.lamp_samples, settings=self.settings),
        )

    def output_fields(self, problem: Scene, solution: SceneSolution) -> dict[str, np.ndarray]:
        """The fluence rate ``G`` at the cell centres, and its parts ``G_direct`` and ``G_reflected``
        when something reflects -- see :meth:`Physics.output_fields`."""
        del problem
        if solution.fluence_rate_direct is None:
            return {}
        fields = {"G": solution.fluence_rate}
        if solution.fluence_rate_reflected is not None:
            fields["G_direct"] = solution.fluence_rate_direct
            fields["G_reflected"] = solution.fluence_rate_reflected
        return fields

    def output_patch_fields(
        self, problem: Scene, solution: SceneSolution
    ) -> dict[str, dict[str, np.ndarray]]:
        """Per receiving patch: the irradiance ``E``, what the wall absorbs of it, ``E_absorbed``, and
        the parts ``E_direct`` and ``E_reflected`` when something reflects."""
        out = {}
        for name, receivers in problem.surfaces.items():
            total = solution.irradiance(name)
            fields = {"E": total, "E_absorbed": (1.0 - receivers.reflectance) * total}
            if solution.irradiance_reflected is not None:
                fields["E_direct"] = solution.irradiance_direct[name]
                fields["E_reflected"] = solution.irradiance_reflected[name]
            out[name] = fields
        return out

    def results(self, problem: Scene, solution: SceneSolution) -> dict[str, object]:
        """Where the lamps' power goes.

        ``lamp_power`` and ``lamp_facets``; ``reflector_facets`` and ``radiosity_cycles`` (when something
        reflects), and the power the lamps take back, ``lamp_absorbed_power`` (likewise); per receiving
        patch its ``area``, the power arriving on it, ``incident_power``, and the part it keeps,
        ``absorbed_power``; and, when the cells were gathered at, ``volume_integral_G``. The medium's
        share, ``medium_absorbed_power``, is zero when it absorbs nothing and otherwise needs the cells.
        ``unaccounted_power`` is the lamps' power less the medium's, the lamps' own and every receiving
        patch's share: what the occluders and any patch not gathered on absorb, and the lamps too when
        nothing reflects -- or, where it is small and nothing else can absorb, the error of the gathers.
        """
        patches = {}
        for name, receivers in problem.surfaces.items():
            total = solution.irradiance(name)
            patches[name] = {
                "area": float(np.sum(receivers.areas)),
                "incident_power": float(np.sum(total * receivers.areas)),
                "absorbed_power": float(
                    np.sum((1.0 - receivers.reflectance) * total * receivers.areas)
                ),
            }
        medium = 0.0 if problem.absorption is None else solution.medium_absorbed_power
        out: dict[str, object] = {
            "lamp_power": solution.lamp_power,
            "lamp_facets": problem.lamps.n_facets,
        }
        if problem.reflectors is not None:
            out["reflector_facets"] = problem.reflectors.n_facets
        if solution.cycles is not None:
            out["radiosity_cycles"] = solution.cycles
        if solution.lamp_absorbed_power is not None:
            out["lamp_absorbed_power"] = solution.lamp_absorbed_power
        if problem.volume is not None:
            out["volume_integral_G"] = float(np.sum(solution.fluence_rate * problem.volume.volumes))
        out["medium_absorbed_power"] = medium
        out["patches"] = patches
        if medium is not None:
            out["unaccounted_power"] = (
                solution.lamp_power
                - medium
                - (solution.lamp_absorbed_power or 0.0)
                - sum(entry["absorbed_power"] for entry in patches.values())
            )
        return out

    def progress_fields(self, problem: Scene) -> None:
        """None: nothing is marched -- see :meth:`Physics.progress_fields`."""
        del problem


def _lamps(lamps: Mapping[str, Lamp], triangles_of, directory: Path) -> Surfaces:
    """Every lamp as one body of a surface set, emitting its power over its own area with its profile,
    and reflecting its reflectance of the light arriving on it."""
    pieces, powers, profiles = [], {}, []
    for name, lamp in lamps.items():
        pieces.append(triangles_of(name, lamp.geometry))
        power = lamp.power if lamp.power is not None else lamp.profile.radiant_power(directory)
        if power is None:
            raise ValueError(
                f"boundaries.{name}: the lamp states no power, and its photometry file states its "
                "intensities in no radiant unit (an [_INTENSITYUNITS] keyword of W/sr, mW/sr or "
                "uW/sr), so it carries none either. Give the lamp's power in W."
            )
        powers[name] = power
        profiles.append(lamp.profile.profile(directory))
    solid_id = np.repeat(np.arange(len(pieces)), [len(piece) for piece in pieces])
    surfaces = Surfaces.from_triangles(
        np.concatenate(pieces),
        solid_id=solid_id,
        solid_names=tuple(lamps),
        diffuse_reflectance=np.asarray([lamp.reflectance or 0.0 for lamp in lamps.values()])[
            solid_id
        ],
        profiles=tuple(profiles),
        profile_index=solid_id,
    )
    return surfaces.with_optics(emission=lamp_exitance(surfaces, powers))


def _required(fields: Mapping[str, np.ndarray], names: tuple[str, ...]) -> list[np.ndarray]:
    """The named fields, in order, refusing a mapping that lacks one."""
    missing = [name for name in names if name not in fields]
    if missing:
        raise ValueError(
            f"a starting state needs the fields {list(names)}, but {missing} "
            f"{'is' if len(missing) == 1 else 'are'} missing from {sorted(fields)}."
        )
    return [fields[name] for name in names]


def _set(**settings: object) -> dict[str, object]:
    """The settings that are set: a case leaves a builder's default in force by leaving one unset."""
    return {name: value for name, value in settings.items() if value is not None}


def _momentum(spec: CaseSpec, mesh: Mesh, geometry: MeshGeometry) -> MomentumContinuity:
    """The flow assembler every physics starts from: the fluid, each patch's flow closure, the numerics."""
    return MomentumContinuity.build(
        mesh,
        geometry,
        spec.fluid.property_model(),
        BoundaryConditions(
            {
                name: condition.flow_closure()
                for name, condition in spec.patch_conditions(mesh).items()
            }
        ),
        advection_scheme=spec.numerics.momentum_advection,
        sources=tuple(source.momentum_source() for source in spec.sources),
        **_set(
            gradient_scheme=spec.numerics.gradient,
            drive=None if spec.drive is None else spec.drive.drive(),
            pressure_datum=spec.pressure_datum,
        ),
    )
