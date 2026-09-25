"""Where a computer-aided design (CAD) model sits in the coordinates of the case it describes.

A CAD model is drawn in its own frame, and that frame is rarely the mesh's: a vessel drawn with its
axis along ``y`` is meshed with it along ``x``, or the drawing's origin is somewhere else. A
:class:`Placement` is the rigid map from one to the other, ``x_case = matrix @ x_cad + offset``,
applied once as the model is read so that every body, every face and every triangle it yields is in
the case's frame.

**The matrix must be orthogonal, and may be a reflection.** A rotation or a reflection maps a
circle to a circle, so a cylinder stays a cylinder; anything else — a stretch along one axis —
turns it into an elliptic cylinder, which no body here describes, so it is refused rather than
approximated. Reflections are allowed because swapping two axes, the most common placement of all,
is one. Units are not a placement's business: the reader always delivers metres.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Placement"]

#: How far ``matrix.T @ matrix`` may stray from the identity, entry by entry. Loose enough for a
#: matrix typed as decimals, tight enough that a genuine stretch is refused.
_ORTHOGONALITY_TOLERANCE = 1e-9


@dataclasses.dataclass(frozen=True)
class Placement:
    """A rigid map ``x -> matrix @ x + offset`` from a CAD model's frame to a case's.

    Attributes
    ----------
    matrix : np.ndarray, shape ``(3, 3)``
        Orthogonal: a rotation, or a rotation composed with a reflection.
    offset : np.ndarray, shape ``(3,)``
        Translation, in metres, applied after ``matrix``.

    Examples
    --------
    A vessel drawn with its axis along ``y``, meshed with it along ``x``::

        Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    """

    matrix: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    offset: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float)
        if matrix.shape != (3, 3) or offset.shape != (3,):
            msg = (
                f"a Placement needs a (3, 3) matrix and a (3,) offset, got {matrix.shape} and "
                f"{offset.shape}"
            )
            raise ValueError(msg)
        departure = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
        if departure > _ORTHOGONALITY_TOLERANCE:
            msg = (
                "a Placement's matrix must be orthogonal -- a rotation, possibly with a "
                f"reflection -- and this one departs from it by {departure:.3g}. A stretch turns a "
                "cylinder into an elliptic one, which no body describes; scale the drawing's units "
                "in the CAD tool instead."
            )
            raise ValueError(msg)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)
