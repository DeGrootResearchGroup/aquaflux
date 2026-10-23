"""Computer-aided design (CAD) import: STEP files read into exact bodies and emitting triangles.

A reactor is drawn in CAD, and its shape is needed twice by the radiation model: as the geometry
that shadows the light, and as the lamp that emits it. :func:`read_step` reads a STEP file (the
ISO 10303 exchange format every CAD tool writes) and the resulting :class:`CadModel` gives both:

* **exact bodies** from :mod:`aquaflux.solids` — a vessel as the cylinders it is, not as a
  faceted triangle soup — recognized from the solids' faces and **checked against them by
  volume** before being handed out, so a shape the rules misread is refused rather than used;
* **triangles** of a solid's surface, wound outward, every vertex on the true surface, with the
  facet size bounded as well as the chord error;
* for a mesher, a **region** (:class:`CadRegion`: some solids' union, less others) as its topology,
  its face-tagged triangles, a file for the mesher to read, and points projected onto it.

Reading needs the optional CAD kernel, ``pip install aquaflux[cad]``; everything else here —
the face records and the recognition rules — is plain numpy, so it imports and is
tested without it.
"""

from __future__ import annotations

from aquaflux.io.cad.faces import (
    ConeFace,
    CylinderFace,
    OtherFace,
    PlaneFace,
    SolidDescription,
    SphereFace,
    TorusFace,
)
from aquaflux.io.cad.model import CadModel, InexactBody, read_step
from aquaflux.io.cad.placement import Placement
from aquaflux.io.cad.topology import CadRegion, CadTopology, FaceTriangles, Projection
from aquaflux.io.cad.recognize import (
    ConvexPolyhedron,
    CurvedPieces,
    Recognition,
    RecognitionRule,
    UnrecognizedSolid,
    recognize,
)

__all__ = [
    "CadModel",
    "CadRegion",
    "CadTopology",
    "ConeFace",
    "ConvexPolyhedron",
    "CurvedPieces",
    "CylinderFace",
    "FaceTriangles",
    "InexactBody",
    "OtherFace",
    "Placement",
    "PlaneFace",
    "Projection",
    "Recognition",
    "RecognitionRule",
    "SolidDescription",
    "SphereFace",
    "TorusFace",
    "UnrecognizedSolid",
    "read_step",
    "recognize",
]
