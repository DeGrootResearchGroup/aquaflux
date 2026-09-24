"""What a computer-aided design (CAD) solid's faces are, stated without the kernel that read them.

A boundary-representation (B-rep) solid is a closed set of faces, each a bounded piece of an
underlying surface. Recognizing a solid as one of :mod:`aquaflux.solids`' bodies needs only a few
facts about each face — which kind of surface it lies on, that surface's parameters, how much of
the surface the face covers, and **which side of the surface the solid is on**. These records carry
exactly that, as plain numbers, so the recognition rules can be written and tested with no CAD
kernel installed, and the kernel stays behind one module.

The side is recorded rather than inferred from a normal because it is the fact recognition turns
on, and the one that is easy to get backwards. A pipe's wall seen from the water is a cylinder
whose solid lies *inside* it; the same cylinder seen as the bore of a sleeve has the solid
*outside* it, and only the first is a convex piece that a :class:`~aquaflux.solids.Cylinder` can
stand for.
"""

from __future__ import annotations

import dataclasses

import numpy as np

__all__ = [
    "ConeFace",
    "CylinderFace",
    "OtherFace",
    "PlaneFace",
    "SolidDescription",
    "SphereFace",
]


@dataclasses.dataclass(frozen=True)
class PlaneFace:
    """A face lying on a plane.

    Attributes
    ----------
    point : np.ndarray, shape ``(3,)``
        A point on the plane.
    outward_normal : np.ndarray, shape ``(3,)``
        Unit normal pointing out of the solid.
    """

    point: np.ndarray
    outward_normal: np.ndarray


@dataclasses.dataclass(frozen=True)
class CylinderFace:
    """A face lying on a circular cylinder.

    Attributes
    ----------
    origin : np.ndarray, shape ``(3,)``
        The point on the axis where the axial parameter is zero.
    axis : np.ndarray, shape ``(3,)``
        Unit vector along the axis.
    radius : float
    axial_range : tuple of float
        The face's extent along the axis, as distances from ``origin``, low then high.
    angle : float
        How far round the axis the face goes, in radians — ``2 pi`` for a whole turn. Exporters
        often split one cylinder into two half-turn faces, so this is summed over the faces of one
        surface before anything is decided from it. A face that stays short of a turn is a trim —
        typically where the solid meets a neighbour — and cannot stand for a whole cylinder.
    solid_inside : bool
        Whether the solid lies on the axis side of the surface, i.e. whether this face is convex
        seen from outside the solid.
    """

    origin: np.ndarray
    axis: np.ndarray
    radius: float
    axial_range: tuple[float, float]
    angle: float
    solid_inside: bool


@dataclasses.dataclass(frozen=True)
class ConeFace:
    """A face lying on a circular cone.

    Attributes
    ----------
    origin : np.ndarray, shape ``(3,)``
        The point on the axis where the axial parameter is zero.
    axis : np.ndarray, shape ``(3,)``
        Unit vector along the axis.
    axial_range : tuple of float
        The face's extent along the axis, as distances from ``origin``, low then high.
    radii : tuple of float
        The cone's radius at each end of ``axial_range``, in the same order.
    angle : float
        As for :class:`CylinderFace`.
    solid_inside : bool
        As for :class:`CylinderFace`.
    """

    origin: np.ndarray
    axis: np.ndarray
    axial_range: tuple[float, float]
    radii: tuple[float, float]
    angle: float
    solid_inside: bool


@dataclasses.dataclass(frozen=True)
class SphereFace:
    """A face lying on a sphere.

    Attributes
    ----------
    centre : np.ndarray, shape ``(3,)``
    radius : float
    solid_inside : bool
        Whether the solid lies inside the sphere.
    """

    centre: np.ndarray
    radius: float
    solid_inside: bool


@dataclasses.dataclass(frozen=True)
class OtherFace:
    """A face on a surface no recognition rule knows — a torus, a free-form spline patch.

    Attributes
    ----------
    kind : str
        The kernel's name for the surface type, so a refusal can say what it met.
    """

    kind: str


@dataclasses.dataclass(frozen=True)
class SolidDescription:
    """One named solid, as the faces bounding it.

    Attributes
    ----------
    name : str
        The solid's name in the CAD file.
    faces : tuple
        One record per face: :class:`PlaneFace`, :class:`CylinderFace`, :class:`ConeFace`,
        :class:`SphereFace` or :class:`OtherFace`.
    vertices : np.ndarray, shape ``(n_vertices, 3)``
        The solid's topological vertices — the corners where edges meet. Enough to decide whether
        a solid bounded only by planes is convex.
    extent : float
        The length of the solid's bounding-box diagonal, the scale every tolerance is relative to.
    """

    name: str
    faces: tuple
    vertices: np.ndarray
    extent: float

    def of_kind(self, kind: type) -> tuple:
        """The faces that are instances of ``kind``, in order."""
        return tuple(face for face in self.faces if isinstance(face, kind))
