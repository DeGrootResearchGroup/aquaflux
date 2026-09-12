"""Mesh import/export: external mesh formats in, computed fields out.

Separates file-format concerns from mesh representation: :mod:`aquaflux.mesh` owns the mesh
structure, and this package owns the readers that build one from an external source. Every reader
implements the format-agnostic :class:`MeshReader` contract (``read() -> Mesh``); the first is the
OpenFOAM polyMesh reader, exposed as :class:`OpenFOAMReader` and the :func:`read_openfoam`
convenience.

The export half writes computed cell fields back into the case they were read from, as an
ordinary time directory (:func:`write_openfoam_time`) -- so a solved state can be viewed beside
the solution it is validated against, post-processed with the same tools, or used to restart
a run.
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
]
