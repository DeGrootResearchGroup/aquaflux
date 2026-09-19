"""What preconditions a coupled RANS march, described as a value.

The coupled march can be preconditioned by one of two families, and they differ in what they are built
from:

* :class:`BlockDiagonal` -- a block-SIMPLE preconditioner for the flow block beside a scalar multigrid
  for each of ``k`` and ``omega``, assembled from the transport operators without materializing a
  Jacobian;
* :class:`MaterializedJacobian` -- the coupled Jacobian materialized by coloured probing, and inverted
  by one of a complete LU (:class:`CompleteLu`), a single multigrid V-cycle over all fields
  (:class:`MonolithicVCycle`), or a block-triangular field split with a separate inverse for the
  velocity-pressure saddle and for the transported scalars (:class:`FieldSplit`).

The three materialized inverses share everything but the inverse: how the Jacobian is probed, the shift
the first build is fitted at, and the floor its refresh is held above. So they are one family with a
nested choice rather than three families that each restate those settings, which is also what keeps a
setting from being accepted by one inverse and silently ignored by another -- a monolithic smoother
setting cannot be written beside a field split, because there is nowhere to write it.

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

from aquaflux.flow import ConvectionAir, ConvectionTwoLevel, VelocityBlock, ViscousMultilevel
from aquaflux.solve import (
    AirReduction,
    BlockInverse,
    JacobiSmoothed,
    SettingsMapping,
    SettingsValue,
    SimpleSmoothed,
)

from .preconditioner import (
    _DEFAULT_SCALAR_BLOCK,
    ScalarAir,
    ScalarBlock,
    ScalarTwoLevel,
    UnpreconditionedScalars,
)

__all__ = [
    "BlockDiagonal",
    "CompleteLu",
    "FieldSplit",
    "JacobianProbeSpec",
    "MaterializedJacobian",
    "MonolithicVCycle",
    "preconditioner_spec_from_mapping",
    "preconditioner_spec_to_mapping",
]


@dataclasses.dataclass(frozen=True)
class JacobianProbeSpec(SettingsValue):
    """How the coupled Jacobian is materialized by coloured directional-derivative probing.

    Each field is the keyword of the same name on :func:`~aquaflux.turbulence.coupled_jacobian_probe`
    (see it for the meaning and default of each). Two settings of that builder are deliberately not
    here: which field-pair blocks to skip follows from the inverse (a field split never reads one
    triangle), and whether the production viscosity is frozen follows from the operator the march
    differentiates -- neither is a free choice about the probe.

    Attributes
    ----------
    stencil_reach : int or None
        The cell-graph distance the assembled sparsity covers.
    column_reach : tuple of int or None
        A shorter reach per column field, in the flat layout's order ``[u, ..., p, k, omega]``. Any
        sequence is stored as a tuple of integers, so the spec stays hashable and a reach read from a
        case file is the same value as one given in code.
    gradient_sweeps : int or None
        Probe a copy of the residual whose corrected-gradient solve is capped at this many sweeps.
    """

    stencil_reach: int | None = None
    column_reach: tuple[int, ...] | None = None
    gradient_sweeps: int | None = None

    def __post_init__(self) -> None:
        if self.column_reach is not None:
            object.__setattr__(self, "column_reach", tuple(int(r) for r in self.column_reach))


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
    velocity, schur_scaling, composition, mass_scale, v_cycles, strength_threshold
        The flow block's settings -- see :meth:`~aquaflux.flow.BlockPreconditioner.build`.
        ``v_cycles`` is the flow block's alone; the scalar blocks' is set on their own value.

    Raises
    ------
    TypeError
        If ``scalar`` or ``velocity`` is set to something other than a value of its family.
    """

    scalar: ScalarBlock | None = None
    velocity: VelocityBlock | None = None
    schur_scaling: str | None = None
    composition: str | None = None
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


@dataclasses.dataclass(frozen=True)
class CompleteLu(SettingsValue):
    """A complete LU factorization of the materialized Jacobian.

    Exact, so each shifted solve converges in one Krylov iteration, and refactored at the march's own
    shift on every step. Its fill is the limit: it suits two-dimensional or moderate meshes.

    Attributes
    ----------
    backend : {"auto", "umfpack", "scipy"} or None
        The factorization backend -- see :meth:`~aquaflux.solve.MonolithicLuPreconditioner.build`.
    """

    backend: str | None = None


@dataclasses.dataclass(frozen=True)
class MonolithicVCycle(SettingsValue):
    """One algebraic-multigrid V-cycle over every field of the materialized Jacobian.

    Each field is the keyword of the same name on
    :meth:`~aquaflux.solve.MonolithicAmgPreconditioner.build`.

    Attributes
    ----------
    smoother_fill_levels, smoother_sweeps, coarse_eq_limit : int or None
        The level smoother's fill and sweeps, and the equation count of the directly solved coarse grid.
    """

    smoother_fill_levels: int | None = None
    smoother_sweeps: int | None = None
    coarse_eq_limit: int | None = None


@dataclasses.dataclass(frozen=True)
class FieldSplit:
    """A block-triangular field split: the velocity-pressure saddle first, then ``[k, omega]``.

    Each block gets its own inverse, and the coupling of the transported scalars to the flow is retained
    exactly between the two block solves. Both inverses are required, and each is a
    :class:`~aquaflux.solve.BlockInverse` value, so a split is a value too.

    Attributes
    ----------
    leading : BlockInverse
        The inverse fitted to the ``[u, v, w, p]`` saddle -- :class:`~aquaflux.solve.SimpleSmoothed`, for
        example.
    trailing : BlockInverse
        The inverse fitted to the ``[k, omega]`` block -- :class:`~aquaflux.solve.JacobiSmoothed`, for
        example.

    Raises
    ------
    TypeError
        If either inverse is not a :class:`~aquaflux.solve.BlockInverse`.
    """

    leading: BlockInverse
    trailing: BlockInverse

    def __post_init__(self) -> None:
        for role in ("leading", "trailing"):
            inverse = getattr(self, role)
            if not isinstance(inverse, BlockInverse):
                raise TypeError(
                    f"FieldSplit.{role} must be a block-inverse value such as SimpleSmoothed(), "
                    f"JacobiSmoothed() or AirReduction(), got {type(inverse).__name__}. A factory "
                    "closure cannot be compared or written down; a build-record sink is attached "
                    "where the preconditioner is opened, not bound into the inverse."
                )


@dataclasses.dataclass(frozen=True)
class MaterializedJacobian:
    """The family built on the materialized coupled Jacobian, inverted by ``inverse``.

    Attributes
    ----------
    inverse : CompleteLu, MonolithicVCycle or FieldSplit
        How the materialized, shifted Jacobian is inverted. Its type also decides how often the inverse
        is refitted during a march and which Krylov restart regime the forward solve defaults to.
    probe : JacobianProbeSpec
        How the Jacobian is probed. The default probes every column at the builder's own reach.
    build_beta : float or None
        The shift strength the first build is fitted at. It matters beyond the first step when the
        inverse freezes its coarse space at that build, since every later refit reuses it.
    beta_floor : float or None
        A lower bound on the shift the inverse is refitted at, while the march keeps solving at its own
        shift.

    Raises
    ------
    TypeError
        If ``inverse`` or ``probe`` is not one of the accepted values.
    """

    inverse: CompleteLu | MonolithicVCycle | FieldSplit
    probe: JacobianProbeSpec = JacobianProbeSpec()
    build_beta: float | None = None
    beta_floor: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inverse, CompleteLu | MonolithicVCycle | FieldSplit):
            hint = (
                " A single block inverse inverts one block of a field split; wrap two of them in "
                "FieldSplit(leading=..., trailing=...)."
                if isinstance(self.inverse, BlockInverse)
                else ""
            )
            raise TypeError(
                "MaterializedJacobian.inverse must be CompleteLu(), MonolithicVCycle() or "
                f"FieldSplit(...), got {type(self.inverse).__name__}.{hint}"
            )
        if not isinstance(self.probe, JacobianProbeSpec):
            raise TypeError(
                f"MaterializedJacobian.probe must be a JacobianProbeSpec, got {type(self.probe).__name__}."
            )


#: The two families a coupled march can be preconditioned by -- the values a spec file describes.
_SPEC_FAMILIES = (BlockDiagonal, MaterializedJacobian)

#: Every value a spec file may name, at any level: the two families, the materialized inverses and the
#: probe, and the block inverses, velocity blocks and scalar blocks nested inside them.
_SPEC_MAPPING = SettingsMapping(
    [
        *_SPEC_FAMILIES,
        CompleteLu,
        MonolithicVCycle,
        FieldSplit,
        JacobianProbeSpec,
        SimpleSmoothed,
        JacobiSmoothed,
        AirReduction,
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
        beta_floor: 0.05

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
    spec = _SPEC_MAPPING.from_mapping(mapping)
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
    return _SPEC_MAPPING.to_mapping(spec)
