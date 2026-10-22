"""Where a computer-aided design (CAD) drawing sits in a case's frame."""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["CadPlacement"]


@dataclasses.dataclass(frozen=True)
class CadPlacement:
    """The rigid map from a drawing's frame to the case's: ``x -> matrix @ x + offset``, in metres.

    Attributes
    ----------
    matrix : tuple of tuple of float or None
        Three rows of three; a rotation, or a rotation and a reflection. Unset, the identity.
    offset : tuple of float or None
        Three components; unset, none.
    """

    matrix: tuple[tuple[float, ...], ...] | None = None
    offset: tuple[float, ...] | None = None

    def placement(self):
        """The CAD reader's :class:`~aquaflux.io.cad.Placement`, which checks the matrix."""
        from aquaflux.io.cad import Placement

        options = {}
        if self.matrix is not None:
            options["matrix"] = np.asarray(self.matrix, dtype=float)
        if self.offset is not None:
            options["offset"] = np.asarray(self.offset, dtype=float)
        return Placement(**options)
