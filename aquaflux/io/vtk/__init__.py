"""VTK XML output: write a mesh and its cell-centred fields to files a viewer opens directly.

The face-based mesh storage -- nodes, owner/neighbour face->cell incidence, and ragged face->node
rings -- is almost exactly what VTK's arbitrary-polygon and arbitrary-polyhedron cell types are
defined by, so the connectivity is reconstructed rather than translated, and the files are built
here from the standard library alone. The one standard type recognized is the hexahedron, which a
viewer processes far more cheaply than the equivalent polyhedron.

Three seams, mirroring the readers' split: :mod:`.topology` reconstructs the cell connectivity,
:mod:`.xml` serializes it, and :mod:`.writer` is the only part that opens a file. The boundary
patches, and fields that live on their faces, are written by :mod:`.patches`: one polygonal-data
file per patch, indexed by one multiblock file.
"""

from __future__ import annotations

from .patches import boundary_patches, vtm_document, vtp_parts, write_patches
from .topology import (
    VTK_HEXAHEDRON,
    VTK_POLYGON,
    VTK_POLYHEDRON,
    VtkCells,
    build_vtk_cells,
    stored_ring_is_outward,
)
from .writer import write_pvd, write_vtu
from .xml import CellField, cell_data_arrays, pvd_document, vtu_parts

__all__ = [
    "VTK_HEXAHEDRON",
    "VTK_POLYGON",
    "VTK_POLYHEDRON",
    "CellField",
    "VtkCells",
    "boundary_patches",
    "build_vtk_cells",
    "cell_data_arrays",
    "pvd_document",
    "stored_ring_is_outward",
    "vtm_document",
    "vtp_parts",
    "vtu_parts",
    "write_patches",
    "write_pvd",
    "write_vtu",
]
