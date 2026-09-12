"""OpenFOAM polyMesh reader and field writer.

Reads an ASCII OpenFOAM ``constant/polyMesh`` directory into an aquaflux
:class:`~aquaflux.mesh.Mesh`, collapsing a one-cell-thick ``empty``-capped case to a genuine 2D
mesh. The public entry points are :class:`OpenFOAMReader` and the :func:`read_openfoam`
convenience. :func:`write_openfoam_time` is the return path: computed cell fields written back
as a time directory of the same case, inheriting each field's dimensions and boundary
conditions from the case's own copy so the result is a valid restart state.
"""

from __future__ import annotations

from .field_writer import (
    FieldTemplate,
    format_volume_field,
    infer_extruded_axis,
    parse_field_template,
    read_field_template,
    write_openfoam_field,
    write_openfoam_time,
)
from .fields import parse_scalar_field, read_surface_scalar_field, read_volume_scalar_field
from .reader import OpenFOAMReader, read_openfoam

__all__ = [
    "FieldTemplate",
    "OpenFOAMReader",
    "format_volume_field",
    "infer_extruded_axis",
    "parse_field_template",
    "parse_scalar_field",
    "read_field_template",
    "read_openfoam",
    "read_surface_scalar_field",
    "read_volume_scalar_field",
    "write_openfoam_field",
    "write_openfoam_time",
]
