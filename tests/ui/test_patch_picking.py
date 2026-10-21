"""What a click on a drawing's face does to a mesh file, and how the faces are coloured.

The solver's view is written out by hand here: four faces, each with a point on it, the patch each
is in and the face each of a patch's points lands on -- what ``aquaflux drawing`` prints. So every
assertion is against the document a person would have written by hand for the same choice.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyvista")

from aquaflux_ui.patch_picking import face_colours, patch_rows, picked
from aquaflux_ui.solver_commands import DrawingView

POINTS = ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0])
FACES = [{"kind": "plane", "area": 1.0, "point": point} for point in POINTS]
DOCUMENT = {
    "drawing": {"file": "tank.step"},
    "patches": {
        "inlet": {"kind": "FacesNear", "points": [[0.0, 0.1, 1.0], [0.9, 0.0, 0.0]]},
        "lamp": {"kind": "FacesOn", "solids": ["lamp"]},
    },
}
#: Faces 0 and 1 are the inlet's (its points land on them), face 2 the lamp's, face 3 the rest's.
VIEW = DrawingView(
    None, None, ["tank", "lamp"], FACES, ["inlet", "inlet", "lamp", "walls"], {"inlet": [0, 1]}
)


def test_a_face_in_no_patch_joins_the_chosen_one_by_its_point():
    pick = picked(DOCUMENT, VIEW, 3, "inlet", "walls")
    assert pick.document["patches"]["inlet"]["points"] == [
        [0.0, 0.1, 1.0],
        [0.9, 0.0, 0.0],
        POINTS[3],
    ]
    assert pick.message == "" and pick.document["patches"]["lamp"] == DOCUMENT["patches"]["lamp"]


def test_a_new_patch_is_written_as_the_face_nearest_its_first_point():
    pick = picked(DOCUMENT, VIEW, 3, "outlet", "walls")
    assert pick.document["patches"]["outlet"] == {"kind": "FacesNear", "points": [POINTS[3]]}


def test_a_face_already_in_the_chosen_patch_leaves_it_with_only_its_own_points():
    pick = picked(DOCUMENT, VIEW, 1, "inlet", "walls")
    assert pick.document["patches"]["inlet"]["points"] == [[0.0, 0.1, 1.0]]


def test_a_patch_whose_last_face_leaves_leaves_the_file():
    only = {"patches": {"inlet": {"kind": "FacesNear", "points": [[0.0, 0.1, 1.0]]}}}
    view = DrawingView(None, None, [], FACES, ["inlet", "walls", "walls", "walls"], {"inlet": [0]})
    assert picked(only, view, 0, "inlet", "walls").document == {}


def test_a_face_in_another_patch_by_its_points_moves_to_the_chosen_one():
    pick = picked(DOCUMENT, VIEW, 1, "outlet", "walls")
    assert pick.document["patches"]["inlet"]["points"] == [[0.0, 0.1, 1.0]]
    assert pick.document["patches"]["outlet"]["points"] == [POINTS[1]]


def test_with_the_rest_chosen_a_face_leaves_its_patch_and_a_free_face_is_left_alone():
    assert picked(DOCUMENT, VIEW, 0, "walls", "walls").document["patches"]["inlet"]["points"] == [
        [0.9, 0.0, 0.0]
    ]
    assert picked(DOCUMENT, VIEW, 3, "walls", "walls").document == DOCUMENT


def test_a_face_chosen_by_solids_is_not_moved_and_a_solids_patch_takes_no_point():
    held = picked(DOCUMENT, VIEW, 2, "inlet", "walls")
    assert held.document == DOCUMENT and "by its FacesOn selection" in held.message
    refused = picked(DOCUMENT, VIEW, 3, "lamp", "walls")
    assert (
        refused.document == DOCUMENT and "lamp' chooses its faces by its FacesOn" in refused.message
    )


def test_while_the_selections_are_refused_a_face_is_found_by_the_points_landing_on_it():
    refused = DrawingView(None, None, [], FACES, None, {"inlet": [0, 1]}, [], "patch 'x' overlaps")
    pick = picked(DOCUMENT, refused, 0, "inlet", "walls")
    assert pick.document["patches"]["inlet"]["points"] == [[0.9, 0.0, 0.0]]


def test_the_patch_list_and_the_faces_colours_follow_the_files_order():
    palette = ["#FF0000", "#00FF00", "#0000FF"]
    rows = patch_rows(DOCUMENT, VIEW, palette, "#808080", "walls", ["outlet", "spare"])
    assert [(r.name, r.faces, r.chosen_by, r.colour) for r in rows] == [
        ("inlet", 2, "points", "#FF0000"),
        ("lamp", 1, "solids", "#00FF00"),
        ("outlet", 0, "new", "#0000FF"),  # a new name takes the next colour after the file's
        ("spare", 0, "new", "#FF0000"),  # past the last colour, round again
        ("walls", 1, "others", "#808080"),
    ]
    assert face_colours(rows, VIEW).tolist() == [
        [255, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [128, 128, 128],
    ]
    refused = DrawingView(None, None, [], FACES, None, {}, [], "overlaps")
    assert face_colours(rows, refused).tolist() == [[128, 128, 128]] * 4
