"""A mesh file's drawing, as someone choosing its patches sees it: the region's faces and their patches.

Before a region is meshed its patches are chosen from its faces -- a click on a face is a point on
it, which is all :class:`~aquaflux.meshing.FacesNear` stores. :func:`view_drawing` reads the drawing
and region a mesh file names, triangulates the region's faces finely enough to look at, writes them
with each triangle's face, and says what each face is, which patch the file's selections give it (by
the mesher's own rule, :func:`~aquaflux.meshing.assign_patches`), the face each point of a
``FacesNear`` lands on, and which faces are pipes. It reads the file as it stands, unfinished or not:
only the drawing need be set, and until the region is it says only which solids the drawing holds.

The drawing is read once and kept while its file is unchanged, so a page asking again after each
click pays only for the selections.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from aquaflux.io import write_triangles
from aquaflux.io.cad import CadModel, CadRegion, FaceTriangles, read_step
from aquaflux.meshing import FaceSelection, FacesNear, RegionFaces, assign_patches, find_pipes

from .drawing import CadPlacement
from .mesh_file import Drawing, DrawingRegion

__all__ = ["DRAWING_FILE", "DrawingView", "view_drawing"]

#: The file :func:`view_drawing` writes the region's triangles into.
DRAWING_FILE = "drawing.vtp"

#: The largest distance a viewed triangle lies from its face, as a share of the region's extent.
CHORD_SHARE = 1e-3

#: The largest angle, in radians, between the surface normals at a viewed triangle's corners.
VIEW_ANGLE = 0.3

#: The others patch's name when a mesh file leaves it unset: the mesher's own default.
DEFAULT_OTHERS = "walls"


@dataclasses.dataclass(frozen=True)
class DrawingView:
    """A region's faces and the patches a mesh file's selections give them.

    Attributes
    ----------
    surface : pathlib.Path or None
        The triangles written, each carrying the cell value ``face``: the face it lies on; ``None``
        until a region is chosen.
    solids : tuple of str
        The drawing's solids, by name.
    faces : tuple of dict
        One per face, in the mesher's numbering: ``{"kind"}`` (``"plane"``, ``"cylinder"``,
        ``"cone"``, ``"sphere"`` or ``"other"``), ``{"area"}`` and ``{"point"}``, a point on it.
    patches : tuple of str or None
        Each face's patch, the others patch for a face no selection chooses; ``None`` when the
        selections are refused (``selections_error`` says why).
    landed : mapping of {str: tuple of int}
        For each ``FacesNear`` patch, the face each of its points lands on, in the points' order.
    pipes : tuple of int
        The faces that are pipes -- each a cylinder the fluid fills, two rims, nothing inside --
        whatever their rims' division, which the mesher sets for them.
    selections_error : str or None
        Why the selections are refused, if they are.
    """

    surface: Path | None
    solids: tuple[str, ...]
    faces: tuple[dict, ...]
    patches: tuple[str, ...] | None
    landed: Mapping[str, tuple[int, ...]]
    pipes: tuple[int, ...]
    selections_error: str | None

    def to_mapping(self) -> dict:
        """The view as plain data, for a JSON reply."""
        return {
            "surface": None if self.surface is None else str(self.surface),
            "solids": list(self.solids),
            "faces": list(self.faces),
            "patches": None if self.patches is None else list(self.patches),
            "landed": {name: list(faces) for name, faces in self.landed.items()},
            "pipes": list(self.pipes),
            "selections_error": self.selections_error,
        }


def view_drawing(
    drawing: Drawing,
    region: DrawingRegion | None,
    patches: Mapping[str, FaceSelection],
    others: str | None,
    directory: str | Path,
    out: str | Path,
) -> DrawingView:
    """Look at a mesh file's drawing: write the region's faces into ``out`` and say what each is.

    Parameters
    ----------
    drawing : Drawing
    region : DrawingRegion or None
        The region; unset, only the drawing's solids are reported.
    patches : mapping of {str: FaceSelection}
        The mesh file's patches, as they stand.
    others : str or None
        The patch for the faces no selection chooses; unset, the mesher's default.
    directory : path-like
        The directory the mesh file's relative paths are taken from.
    out : path-like
        Where the triangles are written, as :data:`DRAWING_FILE`; created if absent.

    Returns
    -------
    DrawingView

    Raises
    ------
    FileNotFoundError
        If the drawing's file is missing.
    ValueError
        If the region names a solid the drawing does not hold.
    """
    path = (Path(directory) / drawing.file).resolve()
    model = _model(path, path.stat().st_mtime_ns, drawing.placement)
    if region is None:
        return DrawingView(None, tuple(model.names), (), None, {}, (), None)
    cad_region = region.cad_region()
    unknown = sorted(set(cad_region.solids + cad_region.without) - set(model.names))
    if unknown:
        raise ValueError(
            f"the region names {unknown}, which the drawing does not hold; it holds "
            f"{list(model.names)}."
        )
    topology = model.topology(cad_region)
    surface = _triangles(model, cad_region)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = write_triangles(surface.triangles, {"face": surface.face}, out / DRAWING_FILE)
    faces = RegionFaces.sampled(model, cad_region, surface.triangles, surface.face)
    landed = {
        name: tuple(selection.landed(faces).tolist())
        for name, selection in patches.items()
        if isinstance(selection, FacesNear)
    }
    names = [*patches, others if others is not None else DEFAULT_OTHERS]
    try:
        assigned = tuple(names[index] for index in assign_patches(faces, patches).tolist())
        refused = None
    except ValueError as refusal:
        assigned, refused = None, str(refusal)
    points, inverse = np.unique(surface.triangles.reshape(-1, 3), axis=0, return_inverse=True)
    pipes, _ = find_pipes(
        topology, points, inverse.reshape(-1, 3), surface.face, np.arange(topology.n_faces),
        rim_counts=False,
    )  # fmt: skip
    return DrawingView(
        surface=written,
        solids=tuple(model.names),
        faces=tuple(
            {"kind": _kind(record), "area": float(area), "point": sample.tolist()}
            for record, area, sample in zip(
                topology.faces, topology.face_area, faces.samples, strict=True
            )
        ),
        patches=assigned,
        landed=landed,
        pipes=tuple(pipes),
        selections_error=refused,
    )


def _kind(record: object) -> str:
    """A face record's surface, in a word: ``CylinderFace`` is ``"cylinder"``."""
    return type(record).__name__.removesuffix("Face").lower()


@functools.lru_cache(maxsize=4)
def _model(path: Path, modified: int, placement: CadPlacement | None) -> CadModel:
    """The drawing at ``path``, read once per version of the file (``modified``) and placement."""
    return read_step(path, None if placement is None else placement.placement())


@functools.lru_cache(maxsize=8)
def _triangles(model: CadModel, region: CadRegion) -> FaceTriangles:
    """The region's faces triangulated to be looked at: a chord of :data:`CHORD_SHARE` of its extent."""
    vertices = model.topology(region).vertices
    extent = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return model.face_triangles(region, chord=CHORD_SHARE * extent, angle=VIEW_ANGLE)
