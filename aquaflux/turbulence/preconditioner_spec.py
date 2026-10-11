"""What preconditions a coupled RANS march, described as a value.

The coupled march can be preconditioned by one of two families, and they differ in what they are built
from:

* :class:`BlockDiagonal` -- a block-SIMPLE preconditioner for the flow block beside a scalar multigrid
  for each of ``k`` and ``omega``, assembled from the transport operators without materializing a
  Jacobian;
* :class:`MaterializedJacobian` -- the coupled Jacobian materialized by coloured probing, and inverted
  by a block-triangular field split with a separate inverse for the velocity-pressure saddle and for the
  transported scalars (:class:`FieldSplit`).

The materialized family keeps the inverse as a nested choice, beside the settings every inverse shares:
how the Jacobian is probed, the shift the first build is fitted at, and the floor its refresh is held
above.

Every field defaults to ``None``, meaning "not set here" (see :class:`~aquaflux.solve.SettingsValue`),
so a spec changes the settings it names and leaves every other one at the default of the class that
consumes it.

A spec can also be written in a case file: :func:`preconditioner_spec_from_mapping` reads one from the
nested mapping a YAML or JSON document parses to, and :func:`preconditioner_spec_to_mapping` writes one
back. What they produce is the spec, not a preconditioner -- a preconditioner is fitted to a state, and
the march re-fits it from states a case file never sees, so the spec is handed to the solve (or to
:func:`~aquaflux.turbulence.open_session`) exactly as one written in code would be.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Literal

from aquaflux.flow import ConvectionAir, ConvectionTwoLevel, VelocityBlock, ViscousMultilevel
from aquaflux.solve import (
    MATERIALIZED_MAPPING,
    MaterializedJacobian,
    SettingsMapping,
    SettingsValue,
)

from .preconditioner import (
    _DEFAULT_SCALAR_BLOCK,
    ScalarAir,
    ScalarBlock,
    ScalarTwoLevel,
    UnpreconditionedScalars,
)

__all__ = [
    "PRECONDITIONER_SPEC_MAPPING",
    "BlockDiagonal",
    "preconditioner_spec_from_mapping",
    "preconditioner_spec_to_mapping",
]


@dataclasses.dataclass(frozen=True)
class BlockDiagonal(SettingsValue):
    """The block-diagonal family: a block-SIMPLE flow preconditioner and a scalar multigrid for k and omega.

    Assembled from the transport operators at a reference state, with no Jacobian materialized. Every
    field but ``scalar`` is the keyword of the same name on
    :meth:`~aquaflux.flow.BlockPreconditioner.build`; unset, the coupled march takes
    :class:`~aquaflux.flow.ConvectionTwoLevel` as the velocity block and a strength threshold of
    ``0.25``, and every other setting takes that builder's own default. That builder's
    ``reference_state`` is not a setting: the coupled march supplies it.

    Attributes
    ----------
    scalar : ScalarBlock or None
        How the ``k`` and ``omega`` blocks are preconditioned: :class:`~aquaflux.turbulence.ScalarTwoLevel`,
        :class:`~aquaflux.turbulence.ScalarAir`, or :class:`~aquaflux.turbulence.UnpreconditionedScalars`
        to leave them unpreconditioned. Unset, the two-level hierarchy.
    velocity : VelocityBlock or None
        Which velocity block is fitted, and how: :class:`~aquaflux.flow.ViscousMultilevel`,
        :class:`~aquaflux.flow.ConvectionTwoLevel` or :class:`~aquaflux.flow.ConvectionAir`. Unset,
        the coupled march takes :class:`~aquaflux.flow.ConvectionTwoLevel`.

        :class:`~aquaflux.flow.ViscousMultilevel` is a multilevel algebraic multigrid (AMG) on the
        viscous momentum operator, mesh-independent but blind to the Peclet number, so it bounds the
        reachable Reynolds number. :class:`~aquaflux.flow.ConvectionTwoLevel` is a two-level
        hierarchy on the frozen viscous plus first-order-upwind operator, which stays a good
        momentum-block approximation as convection strengthens but whose direct coarse solve does
        not scale to large meshes. :class:`~aquaflux.flow.ConvectionAir` puts the same operator
        under a local approximate ideal restriction (lAIR) hierarchy that is robust in the Peclet
        number *and* mesh-independent. See :meth:`~aquaflux.flow.BlockPreconditioner.build`.
    schur_scaling : {"simple", "msimple"} or None
        Which pressure-Schur approximation the flow block uses. ``"simple"`` scales by the momentum
        diagonal ``a_P``; ``"msimple"`` by a frozen, velocity-independent mass diagonal. Unset,
        ``"simple"``.

        The classical SIMPLE Schur ``V / a_P`` degrades as convection strengthens. The mass-scaled
        ``Q̂ = ρ V / k`` makes the Schur a constant-coefficient pressure Poisson that stays robust
        in the Reynolds number, which carries a **flow-only** solve past the point where the ``a_P``
        Schur's inner solve stalls. Both are scaled Laplacians, hence near-Stokes approximations.
        Inside a coupled flow and turbulence solve the choice does not move the converged state. See
        :meth:`~aquaflux.flow.BlockPreconditioner.build`.
    composition : {"triangular", "simple", "simpler"} or None
        How the flow block's velocity and Schur solves are combined. ``"triangular"`` is one of
        each, ``"simple"`` adds a closing velocity update, and ``"simpler"`` also predicts the
        pressure first, at the cost of a second Schur solve. Unset, ``"triangular"``.

        ``"simple"`` makes the pass the full block ``LU``. This axis is independent of
        ``schur_scaling``. The method Klaij & Vuik call **MSIMPLER** is
        ``schur_scaling="msimple", composition="simpler"``, and their **SIMPLER** is
        ``schur_scaling="simple", composition="simpler"``. See :meth:`~aquaflux.flow.BlockPreconditioner.build`.
    mass_scale : float or None
        The mass-scaled Schur's ``k``, used only with ``schur_scaling="msimple"``. Unset, it is
        calibrated automatically at every iterate from the real momentum diagonal.

        It sets the Schur magnitude to the operating convection, or the block preconditioner is
        unbalanced and stalls. The automatic value is ``mean(rho V / a_P)``, which encodes the true
        velocity, density and viscosity scale. Pass an explicit value only to pin ``k``, for a study
        say. See :meth:`~aquaflux.flow.BlockPreconditioner.build`.
    v_cycles : int or None
        Multigrid V-cycles per application of the flow block. This count is the flow block's alone;
        the scalar blocks' is set on their own value. Unset, one.

        Raising it does **not** rescue the high-Reynolds coupled solve. At high cell Peclet number
        the block's accuracy is limited by the Schur approximation, not by how well that
        approximation is inverted, so extra velocity cycles leave the preconditioned error operator
        ``I - A M`` unchanged and extra Schur cycles make it worse. See
        :meth:`~aquaflux.flow.BlockPreconditioner.build`.
    strength_threshold : float or None
        Strength-of-connection threshold for the flow block's velocity and Schur multigrid
        aggregation. ``0`` aggregates on the full graph, and a positive value (such as ``0.25``)
        only along strong connections. Unset, the coupled march takes ``0.25``.

        Aggregating along strong connections is what keeps those V-cycles contracting on a
        high-aspect-ratio or skewed mesh, where isotropic aggregation coarsens across the stiff
        wall-normal direction and the V-cycle stalls. It is a no-op on a low-aspect-ratio mesh and
        does not apply to the :class:`~aquaflux.flow.ConvectionAir` block, whose coarsening is
        already strength-based. It makes the coarsening value-dependent. See
        :meth:`~aquaflux.flow.BlockPreconditioner.build`.

    Raises
    ------
    TypeError
        If ``scalar`` or ``velocity`` is set to something other than a value of its family.
    """

    scalar: ScalarBlock | None = None
    velocity: VelocityBlock | None = None
    schur_scaling: Literal["simple", "msimple"] | None = None
    composition: Literal["triangular", "simple", "simpler"] | None = None
    mass_scale: float | None = None
    v_cycles: int | None = None
    strength_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.scalar is not None and not isinstance(self.scalar, ScalarBlock):
            raise TypeError(
                "BlockDiagonal.scalar must be a scalar-block value such as ScalarTwoLevel(), "
                f"ScalarAir() or UnpreconditionedScalars(), got {self.scalar!r}."
            )
        if self.velocity is not None and not isinstance(self.velocity, VelocityBlock):
            raise TypeError(
                "BlockDiagonal.velocity must be a velocity-block value such as ConvectionTwoLevel(), "
                f"ViscousMultilevel() or ConvectionAir(), got {self.velocity!r}."
            )

    def resolved_scalar(self) -> ScalarBlock:
        """The scalar block this spec selects, with an unset ``scalar`` resolved to the default.

        Returns
        -------
        ScalarBlock
            The set value, or :class:`~aquaflux.turbulence.ScalarTwoLevel` when unset.
        """
        return _DEFAULT_SCALAR_BLOCK if self.scalar is None else self.scalar

    def flow_block_options(self) -> dict[str, object]:
        """The flow block's settings this spec sets, as keywords for the block-SIMPLE builder.

        Returns
        -------
        dict
            The set flow-block fields; ``scalar`` is never among them.
        """
        return {name: value for name, value in self.settings().items() if name != "scalar"}


#: The two families a coupled march can be preconditioned by -- the values a spec file describes.
_SPEC_FAMILIES = (BlockDiagonal, MaterializedJacobian)

#: Every value a spec file may name, at any level -- read by a whole case file's registry too: the two families, the materialized inverses and the
#: probe, and the block inverses, velocity blocks and scalar blocks nested inside them. The materialized
#: side is **derived from** :data:`~aquaflux.solve.MATERIALIZED_MAPPING` rather than listed again, so a
#: kind added there reaches this registry -- and the laminar one -- the day it is added.
PRECONDITIONER_SPEC_MAPPING = SettingsMapping(
    [
        BlockDiagonal,
        *MATERIALIZED_MAPPING.kinds,
        ViscousMultilevel,
        ConvectionTwoLevel,
        ConvectionAir,
        ScalarTwoLevel,
        ScalarAir,
        UnpreconditionedScalars,
    ]
)


def _refuse_non_spec(value: object, verb: str) -> None:
    if not isinstance(value, _SPEC_FAMILIES):
        raise TypeError(
            f"{verb} a coupled preconditioner spec, which is BlockDiagonal or MaterializedJacobian, got "
            f"{type(value).__name__}. An inverse on its own is the `inverse` of a MaterializedJacobian."
        )


def preconditioner_spec_from_mapping(
    mapping: Mapping[str, object],
) -> BlockDiagonal | MaterializedJacobian:
    """Read a coupled preconditioner spec from the nested mapping a case file parses to.

    Each level is a mapping whose ``kind`` names the value's class and whose other keys are the fields it
    sets; a key left out leaves that field at its default. A list is read as a tuple. For example, a
    field split over a probe with a per-column reach::

        kind: MaterializedJacobian
        inverse:
          kind: FieldSplit
          leading: {kind: SimpleSmoothed, sweeps: 2}
          trailing: {kind: JacobiSmoothed, max_coarse: 200}
        probe: {kind: JacobianProbeSpec, column_reach: [3, 3, 3, 3, 2, 2]}
        refit_beta_floor: 0.05

    As everywhere in a spec, ``null`` and an absent key both mean "not set". Leaving a
    ``BlockDiagonal``'s scalar blocks unpreconditioned is a value of its own,
    ``scalar: {kind: UnpreconditionedScalars}``.

    Parameters
    ----------
    mapping : mapping
        The spec, as parsed from a case file.

    Returns
    -------
    BlockDiagonal or MaterializedJacobian
        The spec, ready to pass as a solve's ``preconditioner`` or to
        :func:`~aquaflux.turbulence.open_session`.

    Raises
    ------
    ValueError
        If a level names no kind, an unknown kind, or a field its kind does not have; the message gives
        the path to the entry.
    TypeError
        If the outermost kind is not one of the two families, or a nested value is of the wrong kind for
        its field.
    """
    spec = PRECONDITIONER_SPEC_MAPPING.from_mapping(mapping)
    _refuse_non_spec(spec, "a spec file describes")
    return spec


def preconditioner_spec_to_mapping(spec: BlockDiagonal | MaterializedJacobian) -> dict[str, object]:
    """Write a coupled preconditioner spec as the nested mapping a case file stores.

    The inverse of :func:`preconditioner_spec_from_mapping`: every field left at its default is omitted,
    and reading the mapping back gives an equal spec.

    Parameters
    ----------
    spec : BlockDiagonal or MaterializedJacobian
        The spec to write.

    Returns
    -------
    dict
        Plain data -- mappings, lists, strings, numbers, booleans and ``None`` -- ready for a YAML or
        JSON writer.

    Raises
    ------
    TypeError
        If ``spec`` is not one of the two families.
    """
    _refuse_non_spec(spec, "only")
    return PRECONDITIONER_SPEC_MAPPING.to_mapping(spec)
