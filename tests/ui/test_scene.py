"""Tests for the viewer's pipeline: what is drawn of a dataset for a view. Nothing here renders.

The dataset is a structured grid written and read back through the solver's VTK writer, so each
count below follows from the grid's shape -- a slice normal to ``z`` cuts one layer of ``nx * ny``
cells, the outer surface is the grid's boundary faces -- rather than from what the code returned.
"""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from aquaflux.io import write_vtu  # noqa: E402
from aquaflux.mesh import structured_grid_3d  # noqa: E402
from aquaflux_ui import Pipeline, Scene, View  # noqa: E402
from aquaflux_ui.scene import (  # noqa: E402  # noqa: E402
    Slice,
    Threshold,
    automatic_range,
    field_components,
    field_values,
    focus_values,
    slice_origin,
)

NX, NY, NZ = 4, 3, 5


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    mesh = structured_grid_3d(NX, NY, NZ, lx=2.0, ly=1.5, lz=2.5)
    centre = np.asarray(mesh.geometry().cell.centroid)
    path = tmp_path_factory.mktemp("grid") / "fields.vtu"
    write_vtu(mesh, {"z": centre[:, 2], "U": centre}, path)
    return pv.read(path)


def test_fields_are_offered_with_their_component_counts(grid):
    assert field_components(grid) == {"z": 1, "U": 3}


def test_a_vector_is_shown_by_its_magnitude_or_one_component(grid):
    centre = np.asarray(grid.cell_data["U"])
    np.testing.assert_allclose(field_values(grid, "U"), np.linalg.norm(centre, axis=1))
    np.testing.assert_array_equal(field_values(grid, "U", 1), centre[:, 1])
    with pytest.raises(IndexError, match="3 components"):
        field_values(grid, "U", 3)
    with pytest.raises(KeyError, match="no field 'T'"):
        field_values(grid, "T")


def test_the_automatic_range_is_the_finite_extent():
    assert automatic_range(np.array([3.0, -1.0, np.nan, np.inf, 2.0])) == (-1.0, 3.0)
    assert automatic_range(np.array([np.nan])) is None


def test_a_log_range_ignores_zeros_and_is_floored_six_decades_below_the_peak():
    # A fluence rate in full shadow is zero: the scale starts at the smallest positive value...
    assert automatic_range(np.array([0.0, 1e-2, 1.0]), log_scale=True) == (1e-2, 1.0)
    # ...but never more than six decades below the largest.
    assert automatic_range(np.array([0.0, 1e-12, 1.0]), log_scale=True) == (1e-6, 1.0)
    assert automatic_range(np.array([0.0, -1.0]), log_scale=True) is None


def test_a_constant_field_still_gets_a_scale_of_some_width():
    low, high = automatic_range(np.array([5.0, 5.0]))
    assert low < 5.0 < high
    low, high = automatic_range(np.array([2.0]), log_scale=True)
    assert low < 2.0 < high


def test_a_slice_origin_moves_along_its_axis_only():
    bounds = (0.0, 2.0, 10.0, 11.0, -1.0, 1.0)
    assert slice_origin(bounds, Slice("z", -0.5)) == (1.0, 10.5, -0.5)
    assert slice_origin(bounds, Slice("x", 2.0)) == (2.0, 10.5, 0.0)


def test_the_outer_surface_is_the_grids_boundary_faces(grid):
    (surface,) = Pipeline().layers(grid, View("fields", "z"))
    assert surface.name == "surface"
    assert surface.mesh.n_cells == 2 * (NX * NY + NY * NZ + NX * NZ)
    # Coloured by the boundary cells' own values, carried onto their faces.
    assert surface.values.shape == (surface.mesh.n_cells,)
    assert set(np.round(surface.values, 12)) <= set(np.round(grid.cell_data["z"], 12))


# Coordinates inside a layer of cells rather than on a face between two, where a plane cuts both.
@pytest.mark.parametrize(
    ("plane", "cells"),
    [(Slice("z", 0.8), NX * NY), (Slice("x", 0.6), NY * NZ), (Slice("y", 0.7), NX * NZ)],
)
def test_a_slice_cuts_one_layer_of_cells(grid, plane, cells):
    (cut,) = Pipeline().layers(grid, View("fields", "z", surface=False, slices=(plane,)))
    assert cut.name == "slice 1" and cut.mesh.n_cells == cells


def test_a_z_slice_shows_the_values_of_the_layer_it_cuts(grid):
    # Layers are 0.5 thick; z = 0.8 lies in the second, whose cells' centres are at 0.75.
    view = View("fields", "z", surface=False, slices=(Slice("z", 0.8),))
    (cut,) = Pipeline().layers(grid, view)
    np.testing.assert_allclose(cut.values, 0.75)


def test_several_slices_are_drawn_each_through_its_own_layer(grid):
    view = View(
        "fields", "z", surface=False, slices=(Slice("z", 0.3), Slice("z", 2.2), Slice("x", 0.6))
    )
    layers = Pipeline().layers(grid, view)
    assert [layer.name for layer in layers] == ["slice 1", "slice 2", "slice 3"]
    np.testing.assert_allclose(layers[0].values, 0.25)
    np.testing.assert_allclose(layers[1].values, 2.25)
    assert layers[2].mesh.n_cells == NY * NZ


def test_a_slice_outside_the_dataset_draws_nothing(grid):
    assert (
        Pipeline().layers(grid, View("fields", "z", surface=False, slices=(Slice("z", 9.0),))) == []
    )


def test_a_threshold_keeps_exactly_the_cells_in_range(grid):
    # z runs over five layers at 0.25, 0.75, ... 2.25: [1.0, 2.0] keeps the middle two.
    view = View("fields", "z", surface=False, thresholds=(Threshold("z", 1.0, 2.0),))
    (kept,) = Pipeline().layers(grid, view)
    assert kept.name == "threshold 1"
    assert set(np.round(kept.values, 12)) == {1.25, 1.75}
    # Its outer surface: a 4 x 3 x 2 block of cells.
    assert kept.mesh.n_cells == 2 * (NX * NY + NY * 2 + NX * 2)


def test_a_threshold_tests_its_own_field_and_is_coloured_by_the_views(grid):
    # Keep the cells whose x-velocity component lies in the first column, colour them by z.
    region = Threshold("U", 0.0, 0.5, component=0)
    (kept,) = Pipeline().layers(grid, View("fields", "z", surface=False, thresholds=(region,)))
    np.testing.assert_allclose(kept.mesh.cell_data["U"][:, 0], 0.25)
    assert set(np.round(kept.values, 12)) == {0.25, 0.75, 1.25, 1.75, 2.25}


def test_several_thresholds_are_drawn(grid):
    view = View(
        "fields", "z", surface=False,
        thresholds=(Threshold("z", 0.0, 0.5), Threshold("z", 2.0, 2.5)),
    )  # fmt: skip
    first, second = Pipeline().layers(grid, view)
    assert (first.name, second.name) == ("threshold 1", "threshold 2")
    assert set(np.round(first.values, 12)) == {0.25}
    assert set(np.round(second.values, 12)) == {2.25}


def test_a_threshold_that_keeps_nothing_draws_nothing(grid):
    view = View("fields", "z", surface=False, thresholds=(Threshold("z", 10.0, 11.0),))
    assert Pipeline().layers(grid, view) == []


def test_beside_a_slice_or_a_threshold_the_surface_is_drawn_plain(grid):
    pipeline = Pipeline()
    for view in (
        View("fields", "z", slices=(Slice("z", 0.8),)),
        View("fields", "z", thresholds=(Threshold("z", 1.0, 2.0),)),
    ):
        surface = pipeline.layers(grid, view)[0]
        assert surface.name == "surface" and surface.values is None
    # Alone, it is coloured.
    assert pipeline.layers(grid, View("fields", "z"))[0].values is not None


def test_derived_geometry_is_reused_until_what_it_depends_on_changes(grid):
    pipeline = Pipeline()
    surface, cut = pipeline.layers(grid, View("fields", "z", slices=(Slice("z", 0.8),)))
    # A colour change recomputes nothing.
    again_surface, again_cut = pipeline.layers(
        grid, View("fields", "z", colormap="gray", slices=(Slice("z", 0.8),))
    )
    assert again_surface.mesh is surface.mesh and again_cut.mesh is cut.mesh
    # Adding a second slice leaves the first as it was.
    _, kept_cut, new_cut = pipeline.layers(
        grid, View("fields", "z", slices=(Slice("z", 0.8), Slice("x", 0.6)))
    )
    assert kept_cut.mesh is cut.mesh and new_cut.mesh is not cut.mesh
    # Moving the first slice recomputes it and nothing else.
    moved_surface, moved_cut, same_new = pipeline.layers(
        grid, View("fields", "z", slices=(Slice("z", 1.8), Slice("x", 0.6)))
    )
    assert moved_surface.mesh is surface.mesh and moved_cut.mesh is not cut.mesh
    assert same_new.mesh is new_cut.mesh


def test_a_removed_slice_is_not_kept(grid):
    pipeline = Pipeline()
    pipeline.layers(grid, View("fields", "z", slices=(Slice("z", 0.8), Slice("x", 0.6))))
    pipeline.layers(grid, View("fields", "z"))
    assert set(pipeline._derived) == {"surface"}


def test_slices_set_the_automatic_range_and_the_surface_beside_them_does_not(grid):
    # The grid's z runs from 0.25 to 2.25 cell to cell; two slices hold the values 0.75 and 1.25.
    view = View("fields", "z", slices=(Slice("z", 0.8), Slice("z", 1.3)))
    layers = Pipeline().layers(grid, view)
    assert automatic_range(focus_values(layers)) == pytest.approx((0.75, 1.25))


def test_with_nothing_but_the_surface_the_surface_sets_the_range(grid):
    (surface,) = Pipeline().layers(grid, View("fields", "z"))
    np.testing.assert_array_equal(focus_values([surface]), surface.values)


def test_uncoloured_layers_carry_no_values(grid):
    (surface,) = Pipeline().layers(grid, View("fields", None))
    assert surface.values is None


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: Slice("w", 0.0), "Slice.axis is one of"),
        (lambda: Threshold("z", 2.0, 1.0), "low must not exceed high"),
        (lambda: View("fields", surface_opacity=-0.1), "surface_opacity is in"),
    ],
    ids=["slice-axis", "threshold-inverted", "opacity"],
)
def test_a_value_outside_its_ranges_is_refused(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_a_scene_draws_in_its_style(grid):
    from aquaflux_ui.theme import DARK, THEMES

    plotter = pv.Plotter(off_screen=True)
    scene = Scene(plotter, THEMES[DARK])
    scene.show({"fields": grid}, View("fields", "z", slices=(Slice("z", 0.8),)))
    context = plotter.actors["surface"].prop.color
    assert context.hex_rgb.lower() == THEMES[DARK].context.lower()
    plotter.close()


def test_a_scene_replaces_what_it_drew(grid):
    # Adds actors to an off-screen plotter without rendering a frame, so it needs no display.
    plotter = pv.Plotter(off_screen=True)
    scene = Scene(plotter)
    scene.show({"fields": grid}, View("fields", "z", slices=(Slice("z", 0.8), Slice("x", 0.6))))
    assert set(plotter.actors) >= {"surface", "slice 1", "slice 2"}
    scene.show({"fields": grid}, View("fields", "z", slices=(Slice("z", 0.8),)))
    assert "slice 2" not in plotter.actors and {"surface", "slice 1"} <= set(plotter.actors)
    plotter.close()
