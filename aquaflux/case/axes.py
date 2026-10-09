"""The coordinate axes a case file names, by letter, and the one setting that names one for a mesh.

A bulk velocity is held along an axis, and an OpenFOAM field read from or written for a two-dimensional
mesh needs the axis the mesh was extruded along; both name it by letter, ``x``, ``y`` or ``z``.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["AXES", "AxisName", "refuse_an_extruded_axis_of_a_three_dimensional_mesh"]

#: The coordinate axes a file may name, in the order of their component indices.
AXES = ("x", "y", "z")

AxisName = Literal["x", "y", "z"]


def refuse_an_extruded_axis_of_a_three_dimensional_mesh(
    where: str, axis: AxisName | None, dim: int
) -> None:
    """Refuse an extruded axis stated for a mesh that was not extruded.

    Parameters
    ----------
    where : str
        The setting's place in the file, for the message.
    axis : {"x", "y", "z"} or None
        The axis the file states, or ``None`` when it states none.
    dim : int
        The mesh's dimension.

    Raises
    ------
    ValueError
        If ``axis`` is stated and the mesh is three-dimensional, which has no extruded axis.
    """
    if axis is not None and dim == 3:
        raise ValueError(
            f"{where}: extruded_axis names the axis a two-dimensional mesh was extruded along, and "
            "this mesh is three-dimensional."
        )
