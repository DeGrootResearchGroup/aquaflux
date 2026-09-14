"""Block inverses for a field split, described as values.

A block-triangular field split (:func:`~aquaflux.solve.build_block_triangular_field_split`) fits one
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

import dataclasses
from collections.abc import Callable

import scipy.sparse as sp

from .field_split import AirBlockInverse, JacobiSmoothedInverse
from .saddle_multigrid import SimpleSmoothedInverse

__all__ = ["AirReduction", "JacobiSmoothed", "SimpleSmoothed"]


@dataclasses.dataclass(frozen=True)
class _BlockInverseSpec:
    """The shared half of every block-inverse value object: forward the set fields, bind a sink."""

    #: Whether the inverse class accepts a ``report`` sink for its build record.
    _reports = True

    def _inverse_class(self) -> type:
        raise NotImplementedError

    def settings(self) -> dict[str, object]:
        """The settings this value sets, as keyword arguments — unset (``None``) fields omitted."""
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if (value := getattr(self, field.name)) is not None
        }

    def _build(
        self, block: sp.spmatrix, n_fields: int, report: Callable[[str], None] | None
    ) -> object:
        extra = {} if report is None else {"report": report}
        return self._inverse_class()(block, n_fields, **self.settings(), **extra)

    def __call__(self, block: sp.spmatrix, n_fields: int) -> object:
        """Build the inverse for ``block``, the ``(block, n_fields) -> inverse`` a field split calls.

        Parameters
        ----------
        block : scipy.sparse matrix
            The group's diagonal block, field-major, shape ``(n_fields * n_cells,) * 2``.
        n_fields : int
            Fields per cell in the group.

        Returns
        -------
        object
            The inverse, exposing ``n_dofs``, ``apply(residual, *, transpose=...)`` and
            ``refactor_block``.
        """
        return self._build(block, n_fields, None)

    def bound(
        self, *, report: Callable[[str], None] | None
    ) -> Callable[[sp.spmatrix, int], object]:
        """The same factory with the inverse's build record sent to ``report``.

        Parameters
        ----------
        report : callable or None
            Where the build record goes; ``None`` keeps the inverse silent.

        Returns
        -------
        callable
            ``(block, n_fields) -> inverse``.

        Raises
        ------
        TypeError
            If ``report`` is given for an inverse family that keeps no build record.
        """
        if report is not None and not self._reports:
            raise TypeError(
                f"{type(self).__name__} builds an inverse that keeps no build record, so there is "
                "nothing to send to `report`."
            )

        def build(block: sp.spmatrix, n_fields: int) -> object:
            return self._build(block, n_fields, report)

        return build


@dataclasses.dataclass(frozen=True)
class SimpleSmoothed(_BlockInverseSpec):
    """A :class:`~aquaflux.solve.SimpleSmoothedInverse`: a hierarchy over a pressure-velocity saddle.

    Each field is the class keyword of the same name (see the class for its meaning and default); the
    last five are the coarsening settings of :class:`~aquaflux.solve.HierarchyBlockInverse`.
    """

    cycles: int | None = None
    sweeps: int | None = None
    pressure_sweeps: int | None = None
    pressure_omega: float | None = None
    omega: float | None = None
    frobenius: bool | None = None
    schur_frobenius: bool | None = None
    aggressive_levels: int | None = None
    orthonormal: bool | None = None
    avoid_singletons: bool | None = None
    block_splitting: bool | None = None
    simplec: bool | None = None
    mu: int | None = None
    pre_smooth: bool | None = None
    prolongation_smoothing: str | None = None
    equilibrate: bool | None = None
    strength_threshold: float | None = None
    max_levels: int | None = None
    max_coarse: int | None = None
    frozen_coarsening: bool | None = None
    shape_headroom: float | None = None

    def _inverse_class(self) -> type:
        return SimpleSmoothedInverse


@dataclasses.dataclass(frozen=True)
class JacobiSmoothed(_BlockInverseSpec):
    """A :class:`~aquaflux.solve.JacobiSmoothedInverse`: one cell-coarsening hierarchy over the group.

    Each field is the class keyword of the same name (see the class for its meaning and default); the
    last four are the coarsening settings of :class:`~aquaflux.solve.HierarchyBlockInverse`.
    """

    cycles: int | None = None
    sweeps: int | None = None
    max_coarse: int | None = None
    aggressive_levels: int | None = None
    prolongation_smoothing: str | None = None
    spectral_damping: bool | None = None
    equilibrate: bool | None = None
    avoid_singletons: bool | None = None
    strength_threshold: float | None = None
    max_levels: int | None = None
    frozen_coarsening: bool | None = None
    shape_headroom: float | None = None

    def _inverse_class(self) -> type:
        return JacobiSmoothedInverse


@dataclasses.dataclass(frozen=True)
class AirReduction(_BlockInverseSpec):
    """An :class:`~aquaflux.solve.field_split.AirBlockInverse`: a reduction-based (lAIR) hierarchy.

    ``cycles``, ``f_iters``, ``c_iters`` and ``omega`` are the class's own keywords; the rest are the
    settings it forwards to :func:`~aquaflux.solve.build_air_hierarchy`. The block size is not a setting:
    it is the group's field count, supplied when the inverse is built. This family keeps no build
    record, so :meth:`bound` refuses a ``report``.
    """

    _reports = False

    cycles: int | None = None
    f_iters: int | None = None
    c_iters: int | None = None
    omega: float | None = None
    theta: float | None = None
    restriction_theta: float | None = None
    degree: int | None = None
    max_coarse: int | None = None
    max_levels: int | None = None

    def _inverse_class(self) -> type:
        return AirBlockInverse
