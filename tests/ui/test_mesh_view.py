"""Tests for the mesh view: what it reads from the solver's files, and what it highlights.

The patches file here is built the way the solver's patch writer lays it out -- a multiblock with one
block per patch, named by the patch -- so these pin the layout the view relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from aquaflux.io import write_vtu  # noqa: E402
from aquaflux.mesh import structured_grid_2d  # noqa: E402
from aquaflux_ui.mesh_view import (  # noqa: E402
    MeshView,
    mesh_layers,
    patches_addressed,
    read_patches,
)


@pytest.fixture
def written(tmp_path):
    write_vtu(structured_grid_2d(4, 3), None, tmp_path / "mesh.vtu")
    lines = {
        "left": pv.Line((0, 0, 0), (0, 1, 0)),
        "right": pv.Line((1, 0, 0), (1, 1, 0)),
        "top": pv.Line((0, 1, 0), (1, 1, 0)),
    }
    pv.MultiBlock(lines).save(tmp_path / "patches.vtm")
    return tmp_path


def test_a_group_names_its_members_and_a_patch_names_itself():
    groups = {"walls": ["top", "bottom"]}
    assert patches_addressed("walls", groups) == ("top", "bottom")
    assert patches_addressed("inlet", groups) == ("inlet",)


def test_the_patches_are_read_by_name(written):
    patches = read_patches(written / "patches.vtm")
    assert sorted(patches) == ["left", "right", "top"]
    assert read_patches(written / "absent.vtm") == {}


def test_the_selected_patches_are_highlighted_and_drawn_last(written):
    mesh = pv.read(written / "mesh.vtu")
    patches = read_patches(written / "patches.vtm")
    layers = mesh_layers(mesh, patches, ["top", "left", "no-such-patch"])
    assert [layer.name for layer in layers] == ["mesh", "patch:right", "patch:top", "patch:left"]
    assert [layer.highlighted for layer in layers] == [False, False, True, True]
    assert layers[0].mesh.n_cells == 12


def test_a_volume_is_drawn_by_its_outer_surface():
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()  # 2 x 2 x 2 cells
    (layer,) = mesh_layers(grid, {})
    assert isinstance(layer.mesh, pv.PolyData) and layer.mesh.n_cells == 24


def test_the_view_draws_the_mesh_and_redraws_the_highlight(written):
    plotter = pv.Plotter(off_screen=True)
    view = MeshView(plotter)
    view.load(written, dim=2)
    assert {"mesh", "patch:left", "patch:right", "patch:top"} <= set(plotter.actors)
    view.show(["top"])
    top = plotter.actors["patch:top"].prop
    assert top.color.hex_rgb.lower() == view.style.highlight.lower()
    view.show([])
    assert plotter.actors["patch:top"].prop.color.hex_rgb.lower() != view.style.highlight.lower()
    plotter.close()


def test_a_mesh_without_patches_is_still_drawn(tmp_path):
    write_vtu(structured_grid_2d(2, 2), {"p": np.zeros(4)}, tmp_path / "mesh.vtu")
    plotter = pv.Plotter(off_screen=True)
    view = MeshView(plotter)
    view.load(tmp_path, dim=2)
    assert view.patches == {} and set(plotter.actors) >= {"mesh"}
    plotter.close()
