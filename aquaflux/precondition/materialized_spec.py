"""The materialized-Jacobian preconditioner as a value.

A coupled march can be preconditioned by a Jacobian materialized by coloured probing and inverted by
either a block-triangular field split with a separate inverse for the leading and the trailing fields
(:class:`FieldSplit`) or a single block inverse over every field (a
:class:`~aquaflux.precondition.BlockInverse`). These are one family with a nested choice
(:class:`MaterializedJacobian`) rather than two, because they share everything but the inverse: how the
Jacobian is probed, the shift the first build is fitted at, and the floor its refresh is held above.

Nothing here names a residual. Every field defaults to ``None``, meaning "not set here" (see
:class:`~aquaflux.solve.SettingsValue`), so a spec changes the settings it names and leaves every other one
at the default of the class that consumes it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import ClassVar, Literal

from ..solve import SettingsMapping, SettingsValue
from .block_inverse import AirReduction, BlockInverse, JacobiSmoothed, SimpleSmoothed

__all__ = [
    "MATERIALIZED_MAPPING",
    "FieldSplit",
    "JacobianProbeSpec",
    "MaterializedJacobian",
    "materialized_spec_from_mapping",
    "materialized_spec_to_mapping",
]


@dataclasses.dataclass(frozen=True)
class JacobianProbeSpec(SettingsValue):
    """How a coupled Jacobian is materialized by coloured directional-derivative probing.

    Each field is the keyword of the same name on :meth:`~aquaflux.precondition.JacobianProbe.build`
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

    #: The settings for which unset means the feature is off (read by the case-file schema).
    unset_means_off: ClassVar[tuple[str, ...]] = (
        "column_reach",
        "gradient_sweeps",
    )

    def __post_init__(self) -> None:
        if self.column_reach is not None:
            object.__setattr__(self, "column_reach", tuple(int(r) for r in self.column_reach))


#: The precisions a field split can be applied in (:attr:`FieldSplit.precision`).
SPLIT_PRECISIONS = ("float64", "float32")


@dataclasses.dataclass(frozen=True)
class FieldSplit:
    """A block-triangular field split: the problem's leading group of fields first, then the rest.

    For a flow coupled to transported scalars the leading group is the velocity-pressure saddle and the
    trailing one the scalars. Each block gets its own inverse, and the coupling of the trailing fields to
    the leading ones is retained exactly between the two block solves. A problem with a single group of
    fields (a laminar flow) has nothing to split and refuses this inverse. Both inverses are required, and each is a
    :class:`~aquaflux.precondition.BlockInverse` value, so a split is a value too.

    Attributes
    ----------
    leading : BlockInverse
        The inverse fitted to the leading fields (``[u, v, w, p]``), for example
        :class:`~aquaflux.precondition.SimpleSmoothed`.
    trailing : BlockInverse
        The inverse fitted to the trailing fields (``[k, omega]``), for example
        :class:`~aquaflux.precondition.JacobiSmoothed`.
    precision : {"float64", "float32"}
        The floating-point precision the split is applied in. Everything is fitted in double
        precision -- the hierarchies, the inverted diagonal blocks, the coupling -- and under
        ``"float32"`` the fitted arrays are then stored in single precision and each application runs
        in it, its input rounded and its output widened back. The residual, the Krylov recurrences and
        the stopping test stay in double precision, so the precision changes how fast a solve
        converges, never what it converges to. A single-precision application moves fewer bytes per
        stored entry. The adjoint's transpose solve applies the same rounded split in double-precision
        arithmetic, since one applied in single precision is linear only to about ``1e-7`` and a solve
        run tighter than that stagnates. Only a split applied on the device (both blocks offering a
        traced cycle) can take it.

    Raises
    ------
    TypeError
        If either inverse is not a :class:`~aquaflux.precondition.BlockInverse`.
    ValueError
        If ``precision`` is neither ``"float64"`` nor ``"float32"``.
    """

    leading: BlockInverse
    trailing: BlockInverse
    precision: Literal["float64", "float32"] = "float64"

    def __post_init__(self) -> None:
        if self.precision not in SPLIT_PRECISIONS:
            raise ValueError(
                f"FieldSplit.precision must be one of {SPLIT_PRECISIONS}, got {self.precision!r}."
            )
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
    """The family built on a materialized coupled Jacobian, inverted by ``inverse``.

    Attributes
    ----------
    inverse : FieldSplit or BlockInverse
        How the materialized, shifted Jacobian is inverted. A bare
        :class:`~aquaflux.precondition.BlockInverse` such as
        :class:`~aquaflux.precondition.SimpleSmoothed` inverts the **whole** state, so it is for a
        problem whose fields form a single group, such as a laminar flow. A problem with two groups, such
        as a turbulent flow, takes a :class:`FieldSplit`.
    probe : JacobianProbeSpec
        How the Jacobian is probed. The default probes every column at the builder's own reach.
    build_beta : float or None
        The shift strength the first build is fitted at. It matters beyond the first step when the
        inverse freezes its coarse space at that build, since every later refit reuses it.
    refit_beta_floor : float or None
        A lower bound on the shift the inverse is refitted at, while the march keeps solving at its own
        shift. It changes how well the preconditioner tracks the operator, never what is solved.

        This is not the march's own ``beta_floor`` (a field of
        :class:`~aquaflux.solve.Globalization`), which bounds the shift the *solve* runs at. This
        one bounds only the shift the *inverse is fitted to*.

    Raises
    ------
    TypeError
        If ``inverse`` or ``probe`` is not one of the accepted values.
    """

    inverse: FieldSplit | BlockInverse
    probe: JacobianProbeSpec = JacobianProbeSpec()
    build_beta: float | None = None
    refit_beta_floor: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inverse, FieldSplit | BlockInverse):
            raise TypeError(
                "MaterializedJacobian.inverse must be FieldSplit(...) "
                f"or a block inverse such as SimpleSmoothed(), got {type(self.inverse).__name__}."
            )
        if not isinstance(self.probe, JacobianProbeSpec):
            raise TypeError(
                f"MaterializedJacobian.probe must be a JacobianProbeSpec, got {type(self.probe).__name__}."
            )


#: Every value a materialized-Jacobian spec file may name, at any level: the family, its inverses and
#: probe, and the block inverses nested inside a field split. A solve with more kinds of preconditioner
#: (the coupled RANS one adds a block-diagonal family) extends this list rather than restating it, so the
#: kinds and their names are written down once.
MATERIALIZED_MAPPING = SettingsMapping(
    [
        MaterializedJacobian,
        FieldSplit,
        JacobianProbeSpec,
        SimpleSmoothed,
        JacobiSmoothed,
        AirReduction,
    ]
)


def materialized_spec_from_mapping(mapping: Mapping[str, object]) -> MaterializedJacobian:
    """Read a materialized-Jacobian spec from the nested mapping a case file parses to.

    Each level is a mapping whose ``kind`` names the value's class and whose other keys are the fields it
    sets; a key left out leaves that field at its default. A list is read as a tuple. For example::

        kind: MaterializedJacobian
        inverse: {kind: FieldSplit, leading: {kind: SimpleSmoothed}, trailing: {kind: JacobiSmoothed}}
        probe: {kind: JacobianProbeSpec, stencil_reach: 2}
        refit_beta_floor: 0.05

    Parameters
    ----------
    mapping : mapping
        The spec, as parsed from a case file.

    Returns
    -------
    MaterializedJacobian
        The spec.

    Raises
    ------
    ValueError
        If a level names no kind, an unknown kind, or a field its kind does not have; the message gives
        the path to the entry.
    TypeError
        If the outermost kind is not :class:`MaterializedJacobian`, or a nested value is of the wrong
        kind for its field.
    """
    spec = MATERIALIZED_MAPPING.from_mapping(mapping)
    if not isinstance(spec, MaterializedJacobian):
        raise TypeError(
            "a spec file describes a MaterializedJacobian, got "
            f"{type(spec).__name__}. An inverse on its own is the `inverse` of a MaterializedJacobian."
        )
    return spec


def materialized_spec_to_mapping(spec: MaterializedJacobian) -> dict[str, object]:
    """Write a materialized-Jacobian spec as the nested mapping a case file stores.

    The inverse of :func:`materialized_spec_from_mapping`: every field left at its default is omitted,
    and reading the mapping back gives an equal spec.

    Parameters
    ----------
    spec : MaterializedJacobian
        The spec to write.

    Returns
    -------
    dict
        Plain data ready for a YAML or JSON writer.

    Raises
    ------
    TypeError
        If ``spec`` is not a :class:`MaterializedJacobian`.
    """
    if not isinstance(spec, MaterializedJacobian):
        raise TypeError(f"only a MaterializedJacobian is written here, got {type(spec).__name__}.")
    return MATERIALIZED_MAPPING.to_mapping(spec)
