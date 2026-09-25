"""Mesh import/export: external mesh formats in, computed fields out.

Separates file-format concerns from mesh representation: :mod:`aquaflux.mesh` owns the mesh
structure, and this package owns the readers that build one from an external source and the writers
that serialize what was computed on it. Every reader implements the format-agnostic
:class:`MeshReader` contract (``read() -> Mesh``); the first is the OpenFOAM polyMesh reader, exposed
as :class:`OpenFOAMReader` and the :func:`read_openfoam` convenience.

The export half has two destinations, for two different purposes. :func:`write_openfoam_time` writes
cell fields back into the case they were read from, as an ordinary time directory -- so a solved
state can be viewed beside the solution it is validated against, post-processed with the same tools,
or used to restart a run. :func:`write_vtu` instead writes the mesh *and* its fields as one
self-contained VTK XML file, which needs no case to write into and so is the path for a mesh that
came from anywhere; :func:`write_pvd` indexes a series of those as one transient dataset.

Geometry comes in from computer-aided design (CAD) as well as from meshes: :mod:`aquaflux.io.cad`
reads a STEP drawing into exact :mod:`aquaflux.solids` bodies — each checked against the drawing
before it is handed out — and into emitting triangles. It needs the optional CAD kernel and keeps
its own namespace, so importing this package never requires it.
"""

from __future__ import annotations

from .openfoam import (
    FieldTemplate,
    OpenFOAMReader,
    infer_extruded_axis,
    read_openfoam,
    read_surface_scalar_field,
    read_field_template,
    read_volume_scalar_field,
    write_openfoam_field,
    write_openfoam_time,
)
from .reader import MeshReader
from .vtk import write_pvd, write_vtu

__all__ = [
    "FieldTemplate",
    "MeshReader",
    "OpenFOAMReader",
    "infer_extruded_axis",
    "read_field_template",
    "read_openfoam",
    "read_surface_scalar_field",
    "read_volume_scalar_field",
    "write_openfoam_field",
    "write_openfoam_time",
    "write_pvd",
    "write_vtu",
]
