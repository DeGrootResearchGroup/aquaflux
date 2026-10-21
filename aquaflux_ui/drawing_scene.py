"""A mesh file's drawing, drawn: its region's faces, each in its patch's colour, and the face under a
click. Faces can be hidden, to reach the ones behind them: a lamp inside a vessel.

The solver writes the region's faces as triangles (``aquaflux drawing``), each carrying the face of
the drawing it lies on. Nothing here reads the drawing itself: it draws what the solver wrote, and
finds which face a point of the view shows, by the view's own picker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker

from .theme import LIGHT, THEMES, RenderStyle

__all__ = ["FACE", "DrawingScene"]

#: The cell array naming each triangle's face of the drawing.
FACE = "face"

#: The plotter's name for the drawing.
_DRAWING = "drawing"

#: How far from a triangle, as a share of the view's diagonal, a click still picks it.
PICK_TOLERANCE = 0.0005

#: How far the camera stands back from a tight fit, so the drawing does not touch the view's edges.
FRAME_ZOOM = 0.85


class DrawingScene:
    """Draws a drawing's faces into a plotter, each in a colour given per face, and picks them.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Where it is drawn.
    style : RenderStyle, optional
        The view's colours; unset, the light theme's. Assign :attr:`style` to change theme.
    """

    def __init__(self, plotter: pv.Plotter, style: RenderStyle | None = None) -> None:
        self.plotter = plotter
        self.style = style if style is not None else THEMES[LIGHT]
        self.surface: pv.PolyData | None = None
        self.face: np.ndarray = np.zeros(0, dtype=np.int64)
        self._colours: np.ndarray | None = None
        self._hidden: frozenset[int] = frozenset()
        self._shown: pv.DataSet | None = None
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(PICK_TOLERANCE)
        self.pressed: tuple | None = None
        # The view's mouse reaches this interactor, so the camera is read where a press begins.
        interactor = plotter.render_window.GetInteractor()
        if interactor is not None:
            interactor.AddObserver("LeftButtonPressEvent", self._on_press, 1.0)

    def _on_press(self, *_) -> None:
        self.pressed = self.camera_position()

    def load(self, path: Path) -> None:
        """Read the triangles the solver wrote at ``path``."""
        self.surface = pv.read(path)
        self.face = np.asarray(self.surface.cell_data[FACE]).reshape(-1).astype(np.int64)

    def clear(self) -> None:
        """Draw nothing: no region is chosen."""
        self.surface, self.face, self._colours = None, np.zeros(0, dtype=np.int64), None
        self._hidden = frozenset()
        self.show()

    def show(self, colours: np.ndarray | None = None, hidden=None) -> None:
        """Draw the faces, each in its row of ``colours`` (bytes, ``(n_faces, 3)``), but the
        ``hidden`` ones; either unset, as last drawn."""
        if colours is not None:
            self._colours = np.asarray(colours, dtype=np.uint8)
        if hidden is not None:
            self._hidden = frozenset(int(face) for face in hidden)
        self.plotter.remove_actor(_DRAWING, render=False)
        self.plotter.set_background(self.style.background_bottom, top=self.style.background_top)
        self._shown = None
        if self.surface is None or self._colours is None:
            return
        self.surface.cell_data["colour"] = self._colours[self.face]
        kept = ~np.isin(self.face, list(self._hidden))
        if not kept.any():
            return
        self._shown = self.surface if kept.all() else self.surface.extract_cells(kept)
        self.plotter.add_mesh(
            self._shown, name=_DRAWING, scalars="colour", rgb=True, smooth_shading=True,
            render=False,
        )  # fmt: skip

    def frame(self) -> None:
        """Look at the whole drawing, with a margin round it."""
        self.plotter.view_isometric(render=False)
        self.plotter.reset_camera(render=False)
        self.plotter.camera.zoom(FRAME_ZOOM)

    def pick(self, x: float, y: float) -> int | None:
        """The face of the drawing shown at ``(x, y)``, in the view's pixels from its lower left.

        Returns
        -------
        int or None
            The face, or ``None`` when no face is there.
        """
        if self._shown is None:
            return None
        renderer = self.plotter.renderer
        if not self._picker.Pick(x, y, 0.0, renderer):
            return None
        cell = self._picker.GetCellId()
        actor = self.plotter.actors.get(_DRAWING)
        if cell < 0 or actor is None or self._picker.GetActor() is not actor:
            return None
        return int(np.asarray(self._shown.cell_data[FACE]).reshape(-1)[cell])

    def dragged(self) -> bool:
        """Whether the camera moved since the mouse button last went down: a drag, not a click."""
        return self.pressed is not None and self.pressed != self.camera_position()

    def camera_position(self) -> tuple:
        """Where the camera is and looks, to tell a click from the end of a drag that moved it.

        The view-up is left out: rendering makes it square to the view direction, so it changes
        when nothing moved.
        """
        position, focus, _ = self.plotter.camera_position
        return tuple(position), tuple(focus)
