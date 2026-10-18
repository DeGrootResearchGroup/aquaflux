"""Block inverses for a field split, described as values.

A block-triangular field split (:func:`~aquaflux.precondition.field_split_inverse`) fits one
inverse to each of its diagonal blocks through a factory ``(block, n_fields) -> inverse``. The classes
here are those factories written as **frozen value objects**: each names the settings of one inverse
family, compares by value, and builds the inverse when called with a block. That makes a preconditioner
configuration something that can be stored, compared and written in a case description, rather than a
closure over keyword arguments.

**Every field defaults to** ``None``, **meaning "not set here".** Only the fields that are set are passed
to the inverse class, so each class's own defaults — and the reasoning recorded beside them — stay the
only defaults. ``SimpleSmoothed(sweeps=2)`` changes one setting and nothing else.

The field set of each value object is exactly its class's keyword arguments plus the coarsening settings
the class forwards to its hierarchy, so a setting the class does not accept cannot be written down.
The build record sink (``report``) is not a setting but a destination; it is bound with :meth:`bound`.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable
from typing import ClassVar, Literal

import scipy.sparse as sp

from ..solve import SettingsValue
from .field_split import AirReductionInverse, JacobiSmoothedInverse
from .hierarchy_inverse import HierarchyBlockInverse
from .host_preconditioner import RefactorableInverse
from .multigrid import build_air_hierarchy
from .saddle_multigrid import SimpleSmoothedInverse

__all__ = ["AirReduction", "BlockInverse", "JacobiSmoothed", "SimpleSmoothed"]


@dataclasses.dataclass(frozen=True)
class BlockInverse(SettingsValue, abc.ABC):
    """The settings of one field-split block inverse, and the factory that builds it.

    The base of :class:`SimpleSmoothed`, :class:`JacobiSmoothed` and :class:`AirReduction`. Its role is
    to be the type a configuration can require: a split described by values holds one of these for each
    of its blocks, not an arbitrary callable. It is abstract, so constructing it raises rather than
    passing that requirement with no inverse to build.
    """

    #: Whether the inverse class accepts a ``report`` sink for its build record.
    _reports = True
    #: Whether the inverse class can hold its hierarchy in single precision (``stored_precision``).
    _stores_precision = True

    @property
    @abc.abstractmethod
    def unset_resolves_to(self) -> tuple[Callable, ...]:
        """The inverse class this value builds, then what that class forwards its other settings to.

        A setting left unset takes its default from the first of these that has a parameter of its
        name, which is what lets the case-file schema report each default without restating it.
        """

    def _inverse_class(self) -> type:
        """The inverse class this value builds."""
        return self.unset_resolves_to[0]

    def _build(
        self,
        block: sp.spmatrix,
        n_fields: int,
        report: Callable[[str], None] | None,
        stored_precision: str = "float64",
    ) -> RefactorableInverse:
        extra = {} if report is None else {"report": report}
        if stored_precision != "float64":
            extra["stored_precision"] = stored_precision
        return self._inverse_class()(block, n_fields, **self.settings(), **extra)

    def __call__(self, block: sp.spmatrix, n_fields: int) -> RefactorableInverse:
        """Build the inverse for ``block``, the ``(block, n_fields) -> inverse`` a field split calls.

        Parameters
        ----------
        block : scipy.sparse matrix
            The group's diagonal block, field-major, shape ``(n_fields * n_cells,) * 2``.
        n_fields : int
            Fields per cell in the group.

        Returns
        -------
        RefactorableInverse
            The inverse, re-fittable in place so that a march can refresh it.
        """
        return self._build(block, n_fields, None)

    def bound(
        self,
        *,
        report: Callable[[str], None] | None = None,
        stored_precision: str = "float64",
    ) -> Callable[[sp.spmatrix, int], RefactorableInverse]:
        """The same factory with the inverse's build record sent to ``report``, held in ``stored_precision``.

        Parameters
        ----------
        report : callable or None
            Where the build record goes; ``None`` keeps the inverse silent.
        stored_precision : {"float64", "float32"}
            The precision the inverse holds its hierarchy in between refreshes (see
            :class:`~aquaflux.precondition.HierarchyBlockInverse`).

        Returns
        -------
        callable
            ``(block, n_fields) -> inverse``.

        Raises
        ------
        TypeError
            If ``report`` is given for an inverse family that keeps no build record, or a single
            ``stored_precision`` for one that holds its hierarchy only in double precision.
        """
        if report is not None and not self._reports:
            raise TypeError(
                f"{type(self).__name__} builds an inverse that keeps no build record, so there is "
                "nothing to send to `report`."
            )
        if stored_precision != "float64" and not self._stores_precision:
            raise TypeError(
                f"{type(self).__name__} builds an inverse that holds its hierarchy in double precision "
                f"only, so it cannot be stored in {stored_precision!r}."
            )

        def build(block: sp.spmatrix, n_fields: int) -> RefactorableInverse:
            return self._build(block, n_fields, report, stored_precision)

        return build


@dataclasses.dataclass(frozen=True)
class SimpleSmoothed(BlockInverse):
    """A :class:`~aquaflux.precondition.SimpleSmoothedInverse`: multigrid over a pressure-velocity saddle.

    Each field is the class keyword of the same name (see the class for its meaning and default); the
    last five are the coarsening settings of :class:`~aquaflux.precondition.HierarchyBlockInverse`.
    """

    cycles: int | None = None
    sweeps: int | None = None
    pressure_sweeps: int | None = None
    pressure_smoother_weight: float | None = None
    smoother_weight: float | None = None
    frobenius: bool | None = None
    schur_frobenius: bool | None = None
    aggressive_levels: int | None = None
    orthonormal: bool | None = None
    avoid_singletons: bool | None = None
    block_splitting: bool | None = None
    simplec: bool | None = None
    cycle_index: int | None = None
    pre_smooth: bool | None = None
    prolongation_smoothing: Literal["none", "standard", "symmetric-part"] | None = None
    equilibrate: bool | None = None
    strength_threshold: float | None = None
    max_levels: int | None = None
    max_coarse: int | None = None
    frozen_coarsening: bool | None = None
    shape_headroom: float | None = None

    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (
        SimpleSmoothedInverse,
        HierarchyBlockInverse,
    )

    #: The settings for which unset means the feature is off (read by the case-file schema).
    unset_means_off: ClassVar[tuple[str, ...]] = ("shape_headroom",)


@dataclasses.dataclass(frozen=True)
class JacobiSmoothed(BlockInverse):
    """A :class:`~aquaflux.precondition.JacobiSmoothedInverse`: one cell-coarsening hierarchy for a group.

    Each field is the class keyword of the same name (see the class for its meaning and default); the
    last four are the coarsening settings of :class:`~aquaflux.precondition.HierarchyBlockInverse`.
    """

    cycles: int | None = None
    sweeps: int | None = None
    max_coarse: int | None = None
    aggressive_levels: int | None = None
    prolongation_smoothing: Literal["none", "standard", "symmetric-part"] | None = None
    spectral_damping: bool | None = None
    equilibrate: bool | None = None
    avoid_singletons: bool | None = None
    strength_threshold: float | None = None
    max_levels: int | None = None
    frozen_coarsening: bool | None = None
    shape_headroom: float | None = None

    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (
        JacobiSmoothedInverse,
        HierarchyBlockInverse,
    )

    #: The settings for which unset means the feature is off (read by the case-file schema).
    unset_means_off: ClassVar[tuple[str, ...]] = ("shape_headroom",)


@dataclasses.dataclass(frozen=True)
class AirReduction(BlockInverse):
    """An :class:`~aquaflux.precondition.field_split.AirReductionInverse`: a reduction (lAIR) hierarchy.

    ``cycles``, ``f_sweeps``, ``c_sweeps`` and ``smoother_weight`` are the class's own keywords; the rest are the
    settings it forwards to :func:`~aquaflux.precondition.build_air_hierarchy`. The block size is not a
    setting: it is the group's field count, supplied when the inverse is built. This family keeps no
    build record, so :meth:`bound` refuses a ``report``.
    """

    _reports = False
    _stores_precision = False

    cycles: int | None = None
    f_sweeps: int | None = None
    c_sweeps: int | None = None
    smoother_weight: float | None = None
    theta: float | None = None
    restriction_theta: float | None = None
    degree: int | None = None
    max_coarse: int | None = None
    max_levels: int | None = None

    unset_resolves_to: ClassVar[tuple[Callable, ...]] = (AirReductionInverse, build_air_hierarchy)
