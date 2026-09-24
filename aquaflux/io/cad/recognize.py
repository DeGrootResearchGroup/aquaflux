"""Recognizing a computer-aided design (CAD) solid as exact analytic bodies.

A rule reads a :class:`~aquaflux.io.cad.faces.SolidDescription` and either proposes a body from
:mod:`aquaflux.solids` or declines, saying why. **A proposal is a claim, not a result**: every rule
here states what it assumed, and the model that reads the file checks each claim against the solid
itself before handing a body out (see :class:`~aquaflux.io.cad.model.CadModel`). That split is
what makes a small set of rules safe to apply to an arbitrary file — a rule that guesses wrong is
refused by the check, rather than silently shadowing the wrong geometry.

Two rules, because two facts about a solid make it exactly describable:

* :class:`CurvedPieces` — **the solid is the union of the convex surfaces it is wrapped in.** Each
  cylinder, cone or sphere that the solid lies *inside* and that goes all the way round becomes
  that primitive over the face's own extent, and the solid is their union. That is a flat-ended
  pipe, a lamp with a hemispherical tip, a reducer — and a pipe whose end is cut to the curve of
  the vessel it joins, where the cut face is a partial piece of the *vessel's* cylinder and is
  skipped. A solid with a skipped face is only claimed to be covered, with the excess lying inside
  a neighbour, so it can be checked only together with that neighbour.
* :class:`ConvexPolyhedron` — **a convex solid bounded only by planes is the intersection of its
  faces' half-spaces.** That is a theorem rather than a claim, so it needs no further check: a box,
  a rectangular channel, a wedge, a plate.

Anything else is declined: a face on a surface neither rule knows (a torus, a free-form spline), a
bore (a full-turn face with the solid outside it), a planar solid that is not convex.
"""

from __future__ import annotations

import abc
import dataclasses
import math

import numpy as np

from aquaflux.io.cad.faces import (
    ConeFace,
    CylinderFace,
    OtherFace,
    PlaneFace,
    SolidDescription,
    SphereFace,
)
from aquaflux.solids import Cone, Cylinder, HalfSpace, Intersection, Solid, Sphere, Union

__all__ = [
    "ConvexPolyhedron",
    "CurvedPieces",
    "Recognition",
    "RecognitionRule",
    "UnrecognizedSolid",
    "recognize",
]

#: How close to a whole turn the angles of one surface's faces must sum, in radians. The sum of a
#: few floating-point parameter ranges; a genuine gap in a turn is far larger than this.
_TURN_TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True)
class Recognition:
    """A rule's proposal: the body it claims the solid is, and what that claim rests on.

    Attributes
    ----------
    body : Solid
        The proposed body, in the same frame as the description.
    stands_alone : bool
        ``True`` if the body is claimed to **be** the solid. ``False`` if a face was skipped —
        a trim against a neighbour — so the body is claimed only to cover the solid, with whatever
        it covers beyond the solid lying inside the solids it is combined with.
    proven : bool
        ``True`` if the claim holds by construction and needs no check, ``False`` if it must be
        checked against the solid before it is used.
    rule : str
        Which rule proposed it, for reporting.
    """

    body: Solid
    stands_alone: bool
    proven: bool
    rule: str


class UnrecognizedSolid(ValueError):
    """No rule could describe a solid exactly. The message says what each rule found."""


class RecognitionRule(abc.ABC):
    """Strategy interface: propose an exact body for a solid, or say why not."""

    @abc.abstractmethod
    def propose(self, description: SolidDescription) -> Recognition | str:
        """A :class:`Recognition`, or the reason this rule does not apply, as a sentence."""


# ---------------------------------------------------------------------------------------------
# Canonical axes, so that faces of one surface can be found and merged
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Axial:
    """A curved face's axis in a canonical form: one direction sign, one reference point.

    ``foot`` is the point of the axis line nearest the frame's origin and ``low``/``high`` are
    positions along ``direction`` measured from it, so two faces of one surface — whatever
    origin and orientation the kernel happened to give each — compare equal field by field.
    """

    direction: np.ndarray
    foot: np.ndarray
    low: float
    high: float

    @classmethod
    def of(cls, origin, axis, axial_range) -> _Axial:
        """The canonical form of the axis ``origin + v axis`` over ``v`` in ``axial_range``."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        # One sign per line: the first clearly non-zero component is made positive.
        lead = axis[np.argmax(np.abs(axis) > 0.5 / math.sqrt(3.0))]
        sign = 1.0 if lead > 0.0 else -1.0
        direction = sign * axis
        origin = np.asarray(origin, dtype=float)
        along = float(origin @ direction)
        foot = origin - along * direction
        ends = sorted(along + sign * v for v in axial_range)
        return cls(direction=direction, foot=foot, low=ends[0], high=ends[1])

    def centre(self) -> np.ndarray:
        """The midpoint of the covered stretch of axis."""
        return self.foot + 0.5 * (self.low + self.high) * self.direction

    def matches(self, other: _Axial, length: float) -> bool:
        """Whether two stretches of axis are the same, to ``length``."""
        return (
            np.allclose(self.direction, other.direction, atol=length, rtol=0.0)
            and np.allclose(self.foot, other.foot, atol=length, rtol=0.0)
            and abs(self.low - other.low) <= length
            and abs(self.high - other.high) <= length
        )


@dataclasses.dataclass
class _Surface:
    """One underlying surface and the faces found on it so far."""

    kind: str
    axial: _Axial | None
    radii: tuple[float, ...]
    centre: np.ndarray | None
    solid_inside: bool
    angle: float

    def absorbs(self, other: _Surface, length: float) -> bool:
        """Merge ``other`` into this surface if it lies on the same one; report whether it did."""
        if other.kind != self.kind or other.solid_inside != self.solid_inside:
            return False
        if not np.allclose(self.radii, other.radii, atol=length, rtol=0.0):
            return False
        if self.axial is not None and not self.axial.matches(other.axial, length):
            return False
        if self.centre is not None and not np.allclose(
            self.centre, other.centre, atol=length, rtol=0.0
        ):
            return False
        self.angle += other.angle
        return True


def _surface_of(face) -> _Surface:
    """The surface a curved face lies on, canonicalized."""
    if isinstance(face, CylinderFace):
        axial = _Axial.of(face.origin, face.axis, face.axial_range)
        return _Surface("cylinder", axial, (face.radius,), None, face.solid_inside, face.angle)
    if isinstance(face, ConeFace):
        axial = _Axial.of(face.origin, face.axis, face.axial_range)
        # Radii follow the canonical direction: low end first.
        forward = float(np.asarray(face.axis, dtype=float) @ axial.direction) > 0.0
        radii = tuple(face.radii) if forward else tuple(reversed(face.radii))
        return _Surface("cone", axial, radii, None, face.solid_inside, face.angle)
    return _Surface(
        "sphere",
        None,
        (face.radius,),
        np.asarray(face.centre, float),
        face.solid_inside,
        2 * math.pi,
    )


def _body_of(surface: _Surface) -> Solid:
    """The convex primitive a whole-turn surface wraps."""
    if surface.kind == "sphere":
        return Sphere(centre=surface.centre, radius=surface.radii[0])
    axial = surface.axial
    half_length = 0.5 * (axial.high - axial.low)
    if surface.kind == "cylinder":
        return Cylinder(
            centre=axial.centre(),
            axis=axial.direction,
            radius=surface.radii[0],
            half_length=half_length,
        )
    return Cone(
        centre=axial.centre(),
        axis=axial.direction,
        half_length=half_length,
        base_radius=surface.radii[0],
        tip_radius=surface.radii[1],
    )


# ---------------------------------------------------------------------------------------------
# The two rules
# ---------------------------------------------------------------------------------------------


class CurvedPieces(RecognitionRule):
    """A solid as the union of the whole-turn convex surfaces it lies inside.

    Parameters
    ----------
    tolerance : float
        How far apart two faces may be, as a fraction of the solid's extent, and still be taken to
        lie on one surface.
    """

    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance

    def propose(self, description: SolidDescription) -> Recognition | str:
        """Union the whole-turn convex pieces; decline on an unknown surface or a bore."""
        unknown = description.of_kind(OtherFace)
        if unknown:
            kinds = ", ".join(sorted({face.kind for face in unknown}))
            return f"it has faces on surfaces no rule describes ({kinds})"
        curved = [
            f for f in description.faces if isinstance(f, (CylinderFace, ConeFace, SphereFace))
        ]
        if not curved:
            return "it has no curved faces"

        length = self.tolerance * description.extent
        surfaces: list[_Surface] = []
        for surface in map(_surface_of, curved):
            if not any(known.absorbs(surface, length) for known in surfaces):
                surfaces.append(surface)

        pieces, skipped = [], 0
        for surface in surfaces:
            whole = surface.angle >= 2.0 * math.pi - _TURN_TOLERANCE
            if whole and not surface.solid_inside:
                return (
                    f"it lies outside a whole-turn {surface.kind} -- a bore or a hole, which is "
                    "material taken away rather than a convex piece of it"
                )
            if whole:
                pieces.append(_body_of(surface))
            else:
                skipped += 1
        if not pieces:
            return "none of its curved faces goes all the way round"
        body = pieces[0] if len(pieces) == 1 else Union(*pieces)
        return Recognition(body=body, stands_alone=skipped == 0, proven=False, rule="curved pieces")


class ConvexPolyhedron(RecognitionRule):
    """A convex solid bounded only by planes, as the intersection of its faces' half-spaces.

    Parameters
    ----------
    tolerance : float
        How far outside a face's plane a vertex may lie, as a fraction of the solid's extent, before
        the solid is called non-convex.
    """

    def __init__(self, tolerance: float = 1e-9):
        self.tolerance = tolerance

    def propose(self, description: SolidDescription) -> Recognition | str:
        """The half-space intersection, if every face is planar and every vertex is inside all."""
        planes = description.of_kind(PlaneFace)
        if len(planes) != len(description.faces):
            return "not all of its faces are planar"
        length = self.tolerance * description.extent
        distinct: list[PlaneFace] = []
        for plane in planes:
            normal = np.asarray(plane.outward_normal, dtype=float)
            offset = float(np.asarray(plane.point, dtype=float) @ normal)
            if not any(
                np.allclose(normal, np.asarray(d.outward_normal, float), atol=self.tolerance)
                and abs(
                    offset - float(np.asarray(d.point, float) @ np.asarray(d.outward_normal, float))
                )
                <= length
                for d in distinct
            ):
                distinct.append(plane)
        if len(distinct) < 4:
            return f"its {len(distinct)} distinct planes cannot enclose a volume"
        vertices = np.asarray(description.vertices, dtype=float)
        for plane in distinct:
            height = (vertices - np.asarray(plane.point, float)) @ np.asarray(
                plane.outward_normal, float
            )
            if np.max(height) > length:
                return "it is bounded by planes but is not convex"
        body = Intersection(*(HalfSpace(p.point, p.outward_normal) for p in distinct))
        return Recognition(body=body, stands_alone=True, proven=True, rule="convex polyhedron")


#: The rules tried, in order, when none are given.
DEFAULT_RULES = (CurvedPieces(), ConvexPolyhedron())


def recognize(description: SolidDescription, rules=DEFAULT_RULES) -> Recognition:
    """The first rule's proposal for a solid.

    Parameters
    ----------
    description : SolidDescription
    rules : sequence of RecognitionRule, optional
        Tried in order; the first to propose wins.

    Returns
    -------
    Recognition

    Raises
    ------
    UnrecognizedSolid
        If every rule declines. The message gives each rule's reason, so a user can see whether
        the solid needs a different description or a rule this package does not have yet.
    """
    reasons = []
    for rule in rules:
        outcome = rule.propose(description)
        if isinstance(outcome, Recognition):
            return outcome
        reasons.append(f"{type(rule).__name__}: {outcome}")
    msg = (
        f"the solid {description.name!r} cannot be described exactly by any rule here, and it is "
        "refused rather than approximated:\n  " + "\n  ".join(reasons)
    )
    raise UnrecognizedSolid(msg)
