"""VTK XML output: write a mesh and its cell-centred fields to files a viewer opens directly.

The face-based mesh storage -- nodes, owner/neighbour face->cell incidence, and ragged face->node
rings -- is almost exactly what VTK's arbitrary-polygon and arbitrary-polyhedron cell types are
defined by, so the connectivity is reconstructed rather than translated, and the files are built
here from the standard library alone.

Three seams, mirroring the readers' split: :mod:`.topology` reconstructs the cell connectivity,
:mod:`.xml` serializes it, and :mod:`.writer` is the only part that opens a file.
"""

from __future__ import annotations

from .topology import VTK_POLYGON, VTK_POLYHEDRON, VtkCells, build_vtk_cells, stored_ring_is_outward
from .writer import write_pvd, write_vtu
from .xml import CellField, cell_data_arrays, pvd_document, vtu_parts

__all__ = [
    "VTK_POLYGON",
    "VTK_POLYHEDRON",
    "CellField",
    "VtkCells",
    "build_vtk_cells",
    "cell_data_arrays",
    "pvd_document",
    "stored_ring_is_outward",
    "vtu_parts",
    "write_pvd",
    "write_vtu",
]
