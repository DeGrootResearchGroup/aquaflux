"""Choosing a mesh file's patches by clicking the drawing's faces: what a click changes, and how the
faces are coloured.

A mesh file names each patch for the faces a selection chooses: ``FacesNear`` -- the face nearest each
of some points -- or ``FacesOn``, the faces on some solids' surface. A click on a face is a point on
it, which is all ``FacesNear`` stores, so picking a face for a patch adds that face's point to the
patch's points, and picking it again takes the points landing on it away. Which face each point lands
on, and which patch each face ends up in, are the solver's answers (the drawing view it prints), so
nothing here decides a selection itself: it edits the document, and the solver is asked again.

Everything here is pure -- a document and the solver's view in, a new document out -- so it is
tested without a page.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

from .case_form import KIND
from .solver_commands import DrawingView

__all__ = ["FACES_NEAR", "FACES_ON", "PatchRow", "Pick", "face_colours", "patch_rows", "picked"]

#: The selection a click edits: the face nearest each of some points.
FACES_NEAR = "FacesNear"

#: The selection of the faces on some solids' surface, which a click cannot edit.
FACES_ON = "FacesOn"

#: The mesh file's table of patches.
PATCHES = "patches"


@dataclasses.dataclass(frozen=True)
class PatchRow:
    """One patch as the patch list shows it.

    Attributes
    ----------
    name : str
    colour : str
        Its faces' colour in the drawing.
    faces : int
        How many of the drawing's faces it holds, by the solver's assignment.
    chosen_by : str
        ``points`` (a ``FacesNear``), ``solids`` (a ``FacesOn``), another kind's name, ``others``
        (the faces no selection chooses), or ``new`` for a name with no face yet.
    """

    name: str
    colour: str
    faces: int
    chosen_by: str

    def to_state(self) -> dict:
        """The row as the page holds it."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Pick:
    """What a click on a face did.

    Attributes
    ----------
    document : dict
        The mesh file after it; the same as before when nothing changed.
    message : str
        What to tell the person who clicked, when it is not what they would expect; else empty.
    """

    document: dict
    message: str = ""


def patch_rows(
    document: Mapping,
    view: DrawingView,
    palette: Sequence[str],
    context: str,
    others: str,
    pending: Sequence[str] = (),
) -> list[PatchRow]:
    """The patch list: each patch of the file, then each name not yet given a face, then ``others``.

    Parameters
    ----------
    document : mapping
        The mesh file.
    view : DrawingView
        The solver's view of its drawing.
    palette : sequence of str
        The patches' colours, in order, repeating past the last.
    context : str
        The colour of the faces left to ``others``.
    others : str
        The name of the patch for the faces no selection chooses.
    pending : sequence of str
        Names given a patch that has no face yet, so is not in the file.
    """
    patches = document.get(PATCHES) or {}
    assigned = view.patches or []
    rows = []
    for index, (name, selection) in enumerate(patches.items()):
        kind = selection.get(KIND) if isinstance(selection, Mapping) else None
        chosen_by = {FACES_NEAR: "points", FACES_ON: "solids"}.get(kind, str(kind))
        colour = palette[index % len(palette)]
        rows.append(PatchRow(name, colour, assigned.count(name), chosen_by))
    for offset, name in enumerate(name for name in pending if name not in patches):
        colour = palette[(len(patches) + offset) % len(palette)]
        rows.append(PatchRow(name, colour, 0, "new"))
    rows.append(PatchRow(others, context, assigned.count(others), "others"))
    return rows


def face_colours(rows: Sequence[PatchRow], view: DrawingView) -> np.ndarray:
    """Each face's colour, as red, green and blue bytes, shape ``(n_faces, 3)``.

    A face takes its patch's colour; every face takes the others patch's while the selections are
    refused, since no face's patch is known then.
    """
    colour_of = {row.name: _rgb(row.colour) for row in rows}
    others = next(row.colour for row in rows if row.chosen_by == "others")
    if view.patches is None:
        return np.tile(_rgb(others), (len(view.faces), 1)).astype(np.uint8)
    return np.array(
        [colour_of.get(name, _rgb(others)) for name in view.patches], dtype=np.uint8
    ).reshape(-1, 3)


def picked(document: Mapping, view: DrawingView, face: int, patch: str, others: str) -> Pick:
    """The mesh file after ``face`` is clicked while ``patch`` is chosen.

    - A face already in ``patch`` by its points leaves it: the points landing on it are taken away,
      and a patch left with none leaves the file.
    - A face in another patch by its points moves to ``patch``.
    - Any other face joins ``patch``: its point is added, making ``patch`` a ``FacesNear`` if it is
      not in the file yet.
    - With ``others`` chosen, a face in a patch by its points leaves that patch for ``others``.

    A face chosen by a ``FacesOn`` (or any selection but points) is not changed by a click: its
    selection names solids, not faces, so the message says to edit the selection instead.

    Parameters
    ----------
    document : mapping
        The mesh file.
    view : DrawingView
        The solver's view of its drawing, with the file's selections as they are.
    face : int
        The face clicked, in the drawing's numbering.
    patch : str
        The patch chosen in the list.
    others : str
        The name of the patch for the faces no selection chooses.
    """
    edited = copy.deepcopy(dict(document))
    patches = dict(edited.get(PATCHES) or {})
    holder = _holder(patches, view, face)
    if holder is not None and _kind(patches[holder]) != FACES_NEAR:
        return Pick(
            dict(document),
            f"That face is in {holder!r} by its {_kind(patches[holder])} selection; change the "
            "selection to move it.",
        )
    if holder is not None:
        _take(patches, view, holder, face)
    if patch != others and holder != patch:
        if patch in patches and _kind(patches[patch]) != FACES_NEAR:
            return Pick(
                dict(document),
                f"{patch!r} chooses its faces by its {_kind(patches[patch])} selection, not by "
                "points; change the selection to add a face.",
            )
        point = list(view.faces[face]["point"])
        entry = patches.get(patch) or {KIND: FACES_NEAR, "points": []}
        patches[patch] = {**entry, "points": [*entry["points"], point]}
    if holder is None and patch == others:
        return Pick(dict(document))  # already the others patch's
    if patches:
        edited[PATCHES] = patches
    else:
        edited.pop(PATCHES, None)
    return Pick(edited)


def _holder(patches: Mapping, view: DrawingView, face: int) -> str | None:
    """The patch holding ``face``: by the solver's assignment, or by a point landing on it while the
    selections are refused."""
    if view.patches is not None:
        name = view.patches[face]
        return name if name in patches else None
    for name, faces in view.landed.items():
        if face in faces and name in patches:
            return name
    return None


def _take(patches: dict, view: DrawingView, name: str, face: int) -> None:
    """Take from patch ``name`` the points landing on ``face``; a patch left with none goes."""
    landed = view.landed.get(name, [])
    kept = [
        point for point, lands in zip(patches[name]["points"], landed, strict=True) if lands != face
    ]
    if kept:
        patches[name] = {**patches[name], "points": kept}
    else:
        del patches[name]


def _kind(selection: object) -> str | None:
    return selection.get(KIND) if isinstance(selection, Mapping) else None


def _rgb(colour: str) -> tuple[int, int, int]:
    """``#RRGGBB`` as three bytes."""
    text = colour.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
