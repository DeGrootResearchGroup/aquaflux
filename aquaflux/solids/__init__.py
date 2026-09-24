"""Solid bodies described exactly: analytic primitives, their set combinations, and fluid regions.

A body here answers two questions — does a straight segment pass through it, and is a point
inside it — by a formula rather than a search, so a whole array of segments is one expression that
fuses and compiles. The primitives (:class:`HalfSpace`, :class:`Sphere`, :class:`Cylinder`,
:class:`Cone`, :class:`Box`) combine by constructive solid geometry (:class:`Union`,
:class:`Intersection`, :class:`Difference`), and :class:`Outside` describes a vessel by the fluid it
holds, which is usually a handful of convex regions where its wall is awkward to write down.

These are the shapes a computer-aided design (CAD) model is read into, and what the radiation
model shadows its sight lines with. The package holds no mesh, field or physics, so any consumer
can use them::

    chamber = Cylinder(centre=[0.4, 0.0, 0.0], axis=[1.0, 0.0, 0.0], radius=0.05, half_length=0.4)
    riser = Cylinder(centre=[0.05, 0.0, 0.2], axis=[0.0, 0.0, 1.0], radius=0.01, half_length=0.25)
    wall = Outside(chamber, riser)       # everything that is not water
    wall.blocks(origin, target, min_distance)
"""

from __future__ import annotations

from aquaflux.solids.bodies import (
    Body,
    Box,
    Cone,
    ConvexSolid,
    Cylinder,
    Difference,
    HalfSpace,
    Intersection,
    Outside,
    Solid,
    Sphere,
    Union,
)

__all__ = [
    "Body",
    "Box",
    "Cone",
    "ConvexSolid",
    "Cylinder",
    "Difference",
    "HalfSpace",
    "Intersection",
    "Outside",
    "Solid",
    "Sphere",
    "Union",
]
