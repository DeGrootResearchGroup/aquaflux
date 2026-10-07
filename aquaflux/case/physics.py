"""Which equations a case solves: laminar flow, or Reynolds-averaged flow with a turbulence closure.

The physics is one of a case's two discriminators (the other is its drive). It decides which settings
a case may carry at all: everything that belongs to a turbulence closure -- its model constants, how
its variables are parametrized, how its fields are advected, the inflow turbulence at an inlet -- lives
inside :class:`RANS` or on a boundary patch, and a :class:`Laminar` case refuses any of it rather than
ignoring it.

Each physics also builds its own problem from a checked case (:meth:`Physics.build`): a laminar case
is the flow assembler, a Reynolds-averaged one the coupled system holding the flow and the closure.
What comes back is the problem the initializers and the solves already take, so nothing downstream
needs a case-specific type.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import AdvectionScheme
from aquaflux.flow import MomentumContinuity, sheared_patches
from aquaflux.turbulence import (
    CoupledRANS,
    ScalarVariableTransform,
    SSTModel,
    SSTTurbulence,
    coupled_fields,
)

from .boundaries import PatchCondition

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh, MeshGeometry

    from .spec import CaseSpec

__all__ = ["RANS", "Laminar", "Physics"]


@dataclasses.dataclass(frozen=True)
class Physics(abc.ABC):
    """The equations a case solves: :class:`Laminar` or :class:`RANS`."""

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
    def build(self, spec: CaseSpec, mesh: Mesh, geometry: MeshGeometry) -> object:
        """The problem ``spec`` describes, assembled on ``mesh``.

        Parameters
        ----------
        spec : CaseSpec
            The case, already checked against ``mesh``.
        mesh : Mesh
            Its mesh.
        geometry : MeshGeometry
            The mesh's geometry, computed once and shared by every equation.

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

    @abc.abstractmethod
    def restart_fields(self, problem: object, state: object) -> dict[str, np.ndarray]:
        """The physical fields of a march state, by name: what a checkpoint holds.

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
        """

    @abc.abstractmethod
    def initial_fields(self, problem: object, fields: Mapping[str, np.ndarray]) -> object:
        """The starting state a solve takes, from physical fields: the inverse of :meth:`restart_fields`.

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
            If a field this physics needs is missing, naming it and the fields given.
        """

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
class Laminar(Physics):
    """Laminar incompressible flow: momentum and continuity, with no turbulence closure."""

    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse any turbulence setting on a patch -- nothing in a laminar case would read it."""
        stray = [
            f"boundaries.{patch}.{setting}"
            for patch, condition in boundaries.items()
            for setting in condition.turbulence_settings()
        ]
        if stray:
            raise ValueError(
                f"{', '.join(stray)}: a laminar case has no turbulence closure, so nothing would read "
                f"{'this setting' if len(stray) == 1 else 'these settings'}. Remove "
                f"{'it' if len(stray) == 1 else 'them'}, or make the physics RANS."
            )

    def build(self, spec: CaseSpec, mesh: Mesh, geometry: MeshGeometry) -> MomentumContinuity:
        """The flow assembler -- see :meth:`Physics.build`."""
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
        velocity, pressure = _required(fields, ("U", "p"))
        return problem.pack(jnp.asarray(velocity), jnp.asarray(pressure))

    def progress_fields(self, problem: MomentumContinuity) -> None:
        """None: the log reports the residual alone -- see :meth:`Physics.progress_fields`."""
        del problem


@dataclasses.dataclass(frozen=True)
class RANS(Physics):
    """Reynolds-averaged (RANS) incompressible flow closed by the k-omega shear-stress transport (SST) model.

    The flow and the closure's ``k`` and ``omega`` are solved together, as one coupled system.

    Attributes
    ----------
    advection : AdvectionScheme
        How ``k`` and ``omega`` are advected -- :class:`~aquaflux.discretization.FirstOrderUpwind` or
        :class:`~aquaflux.discretization.LimitedUpwind`. Required: the momentum advection is set
        separately, under the case's numerics.
    model : SSTModel or None
        The model's constants; unset, :class:`~aquaflux.turbulence.SSTModel`'s own.
    k_variable, omega_variable : ScalarVariableTransform or None
        The variable each field is solved in -- itself (:class:`~aquaflux.turbulence.DirectScalars`) or
        its logarithm (:class:`~aquaflux.turbulence.LogScalars`, which keeps it positive under any
        Newton step). Unset, the coupled system's default.
    explicit_production_limiter : bool or None
        Freeze the ``k``-production cap in the linearization; unset, the exact operator. See
        :class:`~aquaflux.turbulence.SSTTurbulence` for when this is safe.

    Raises
    ------
    TypeError
        If a setting is not a value of its family.
    """

    advection: AdvectionScheme
    model: SSTModel | None = None
    k_variable: ScalarVariableTransform | None = None
    omega_variable: ScalarVariableTransform | None = None
    explicit_production_limiter: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.advection, AdvectionScheme):
            raise TypeError(f"RANS.advection must be an AdvectionScheme, got {self.advection!r}.")
        for name, family in (
            ("model", SSTModel),
            ("k_variable", ScalarVariableTransform),
            ("omega_variable", ScalarVariableTransform),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, family):
                raise TypeError(f"RANS.{name} must be a {family.__name__}, got {value!r}.")

    def refuse_boundaries(self, boundaries: Mapping[str, PatchCondition]) -> None:
        """Refuse a patch missing a setting the closure needs -- the inflow turbulence at an inlet."""
        missing = [
            f"boundaries.{patch}.{setting}"
            for patch, condition in boundaries.items()
            for setting in condition.missing_turbulence_settings()
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)}: a RANS case needs the turbulence every inflow carries in. Give "
                "it, e.g. turbulence: {kind: FixedTurbulence, k: ..., omega: ...}."
            )

    def build(self, spec: CaseSpec, mesh: Mesh, geometry: MeshGeometry) -> CoupledRANS:
        """The coupled flow and closure -- see :meth:`Physics.build`.

        The closure's walls are the patches whose flow closure is a wall, and its ``k`` and ``omega``
        closures are each patch's own, so neither is stated a second time. Both equations read the one
        property model the fluid gives, and the same gradient reconstruction.
        """
        momentum = _momentum(spec, mesh, geometry)
        model = SSTModel() if self.model is None else self.model
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
            self.advection,
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
        velocity, pressure, k, omega = _required(fields, ("U", "p", "k", "omega"))
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
