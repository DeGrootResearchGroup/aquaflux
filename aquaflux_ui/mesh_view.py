"""A case's mesh, drawn: its cells' outline, and the boundary patches, with chosen ones highlighted.

The solver writes the mesh it reads or generates as VTK files (``aquaflux mesh``): ``mesh.vtu``, and
``patches.vtm`` when it writes the boundary too -- one block per patch, named by the patch. Nothing
here reads a mesh format or generates a grid of its own; it draws what the solver wrote.

:func:`mesh_layers` decides what is drawn and is pure; :class:`MeshView` draws it into a plotter.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pyvista as pv

from .sources import multiblock_leaves
from .theme import LIGHT, THEMES, RenderStyle

__all__ = ["MeshLayer", "MeshView", "mesh_layers", "patches_addressed"]

#: The plotter's name for the mesh layer; patch layers are named ``patch:<name>``.
MESH = "mesh"

#: How far the camera stands back from a tight fit, so the mesh does not touch the view's edges.
FRAME_ZOOM = 0.85

#: Above this many faces drawn, the cell edges are left out: they would fill the view solid.
EDGE_LIMIT = 200_000


@dataclasses.dataclass(frozen=True)
class MeshLayer:
    """One mesh to draw, and how.

    Attributes
    ----------
    name : str
        Unique in the view.
    mesh : pyvista.DataSet
        What is drawn.
    highlighted : bool
        Drawn in the highlight colour, over everything else.
    """

    name: str
    mesh: pv.DataSet
    highlighted: bool = False


def patches_addressed(name: str, groups: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """The patches a boundary key names: a group's members, or the patch of that name.

    Parameters
    ----------
    name : str
        A key of a case's ``boundaries`` section.
    groups : mapping of {str: sequence of str}
        The mesh's patch groups.

    Returns
    -------
    tuple of str
    """
    return tuple(groups[name]) if name in groups else (name,)


def mesh_layers(
    mesh: pv.DataSet, patches: Mapping[str, pv.DataSet], selected: Iterable[str] = ()
) -> list[MeshLayer]:
    """What to draw: the mesh, then every patch, the selected ones highlighted and drawn last.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The mesh's cells.
    patches : mapping of {str: pyvista.DataSet}
        Each boundary patch's faces, by name; empty when the solver wrote none.
    selected : iterable of str
        The patches to highlight; a name that is not a patch is ignored.

    Returns
    -------
    list of MeshLayer
        The mesh -- its outer surface, for a volume -- then the patches, unselected before selected,
        each group in the order given.
    """
    chosen = [name for name in selected if name in patches]
    surface = (
        mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface(algorithm="dataset_surface")
    )
    layers = [MeshLayer(MESH, surface)]
    layers += [
        MeshLayer(f"patch:{name}", block) for name, block in patches.items() if name not in chosen
    ]
    layers += [MeshLayer(f"patch:{name}", patches[name], highlighted=True) for name in chosen]
    return layers


def read_patches(path: Path) -> dict[str, pv.DataSet]:
    """Each patch in a ``patches.vtm`` by name; empty if there is no such file."""
    if not path.is_file():
        return {}
    blocks = pv.read(path)
    return {
        name: block
        for name, block in multiblock_leaves(blocks)
        if block is not None and block.n_cells
    }


class MeshView:
    """Draws a mesh and its patches into a plotter, highlighting the ones asked for.

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
        self.mesh: pv.DataSet | None = None
        self.patches: dict[str, pv.DataSet] = {}
        self.dim = 3
        self._drawn: tuple[str, ...] = ()

    def load(self, directory: Path, dim: int) -> None:
        """Read what the solver wrote into ``directory``, and frame it."""
        self.mesh = pv.read(Path(directory) / "mesh.vtu")
        self.patches = read_patches(Path(directory) / "patches.vtm")
        self.dim = dim
        self.show(())
        self.frame()

    def frame(self) -> None:
        """Look at the whole mesh -- face on for a two-dimensional one -- with a margin round it."""
        if self.dim == 2:
            self.plotter.view_xy(render=False)
        else:
            self.plotter.view_isometric(render=False)
        self.plotter.reset_camera(render=False)
        self.plotter.camera.zoom(FRAME_ZOOM)

    def show(self, selected: Iterable[str]) -> None:
        """Draw the mesh, highlighting the ``selected`` patches."""
        for name in self._drawn:
            self.plotter.remove_actor(name, render=False)
        self.plotter.set_background(self.style.background_bottom, top=self.style.background_top)
        if self.mesh is None:
            self._drawn = ()
            return
        layers = mesh_layers(self.mesh, self.patches, selected)
        for layer in layers:
            if layer.name == MESH:
                self.plotter.add_mesh(
                    layer.mesh,
                    name=layer.name,
                    color=self.style.context,
                    show_edges=layer.mesh.n_cells <= EDGE_LIMIT,
                    edge_color=self.style.text,
                    edge_opacity=0.25,
                    opacity=0.35 if self.dim == 3 else 1.0,
                    render=False,
                )
            else:
                self.plotter.add_mesh(
                    layer.mesh,
                    name=layer.name,
                    color=self.style.highlight if layer.highlighted else self.style.text,
                    line_width=6 if layer.highlighted else 2,
                    opacity=1.0 if layer.highlighted else 0.25,
                    render=False,
                )
        self._drawn = tuple(layer.name for layer in layers)
