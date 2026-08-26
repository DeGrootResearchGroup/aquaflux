"""Flat-vector layout for the coupled block flow state.

The monolithic ``(velocity, pressure)`` unknown is stored as one flat vector laid out
``[vel_0, vel_1, ..., vel_{dim-1}, pressure]`` (each block ``n_cells`` long). The slice arithmetic
that packs/unpacks that layout is a distinct responsibility from residual assembly, and it is the
same arithmetic every other coupled state in the package needs, so it is not written here: this
module only *names* the flow system's blocks over the shared
:class:`~aquaflux.solve.FieldLayout`. A coupled system that carries the flow state as a sub-state
(the Reynolds-averaged ``[flow, k, omega]`` unknown) nests the layout built here rather than
restating its widths.
"""

from __future__ import annotations

from aquaflux.solve import FieldLayout

__all__ = ["flow_state_layout"]


def flow_state_layout(dim: int, n_cells: int) -> FieldLayout:
    """The flat ``[vel_0..vel_{dim-1}, pressure]`` layout of a momentum-continuity state.

    Parameters
    ----------
    dim : int
        Number of velocity components (spatial dimension).
    n_cells : int
        Number of cells; each block is that long per field.

    Returns
    -------
    FieldLayout
        Blocks ``"velocity"`` (``dim`` fields, read out as ``(n_cells, dim)``) then ``"pressure"``
        (one field, ``(n_cells,)``), of total length ``(dim + 1) * n_cells``.
    """
    return FieldLayout.cell_fields(n_cells, velocity=dim, pressure=1)
