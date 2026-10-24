"""A mesh file's drawing as someone choosing its patches sees it, on the Sozzi reactor's own drawing.

The faces' patches come from the mesher's own selections, so what is pinned here is the view's part:
every face written and described, the patches of each by name, the face each point lands on, the
pipes found whatever their rims, the refusals, and the drawing read once while its file is unchanged.
The references are the drawing's own face records, read apart from the view.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("OCP")

from aquaflux.case import DRAWING_FILE, Drawing, DrawingRegion, DrawingSections, view_drawing
from aquaflux.case import drawing_view as module
from aquaflux.io.cad import CadRegion, CylinderFace, SphereFace, read_step
from aquaflux.meshing import FacesNear, FacesOn

SOZZI = (
    Path(__file__).resolve().parents[2]
    / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
)
REGION = DrawingRegion(("reactor_body", "inlet_pipe", "outlet_pipe"), ("lamp",))
R_PIPE, R_LAMP = 0.00955, 0.010


@pytest.fixture(scope="module")
def faces():
    """The region's face records, read apart from the view."""
    model = read_step(SOZZI)
    return model.topology(CadRegion(REGION.solids, REGION.without)).faces


def _view(tmp_path, patches, others=None, region=REGION, drawing=None, feature_angle=None):
    sections = DrawingSections(
        drawing or Drawing(SOZZI.name), region, patches, others, feature_angle
    )
    return view_drawing(sections, SOZZI.parent, tmp_path / "out")


def test_every_face_is_written_and_described_and_given_its_patch(tmp_path, faces):
    view = _view(tmp_path, {"lamp": FacesOn(("lamp",))}, others="vessel")
    assert view.surface == tmp_path / "out" / DRAWING_FILE and view.selections_error is None
    assert set(view.solids) == {"lamp", "reactor_body", "inlet_pipe", "outlet_pipe"}
    # The triangles are written, each carrying the face it lies on.
    header = view.surface.read_bytes().split(b"<AppendedData")[0].decode()
    assert int(re.search(r'NumberOfPolys="(\d+)"', header).group(1)) > 100
    assert 'Name="face"' in header
    kinds = {"CylinderFace": "cylinder", "SphereFace": "sphere", "PlaneFace": "plane"}
    assert [face["kind"] for face in view.faces] == [kinds[type(f).__name__] for f in faces]
    # The lamp's faces are its sleeve and its tip, the fluid outside both: no other face is.
    lamp = {
        index
        for index, record in enumerate(faces)
        if isinstance(record, (CylinderFace, SphereFace))
        and not record.solid_inside
        and abs(record.radius - R_LAMP) < 1e-9
    }
    assert lamp and {i for i, name in enumerate(view.patches) if name == "lamp"} == lamp
    assert set(view.patches) == {"lamp", "vessel"}


def test_each_point_lands_on_its_face_and_the_patch_takes_it(tmp_path):
    first = _view(tmp_path, {})
    assert set(first.patches) == {"walls"}  # the mesher's own name for the rest, unset
    ends = [i for i, face in enumerate(first.faces) if face["kind"] == "plane"]
    smallest = min(ends, key=lambda i: first.faces[i]["area"])
    point = first.faces[smallest]["point"]
    view = _view(tmp_path, {"inlet": FacesNear((tuple(point),))})
    assert view.landed == {"inlet": (smallest,)}
    assert [i for i, name in enumerate(view.patches) if name == "inlet"] == [smallest]


def test_the_pipes_are_found_whatever_their_rims(tmp_path, faces):
    view = _view(tmp_path, {})
    pipes = [
        i
        for i, record in enumerate(faces)
        if isinstance(record, CylinderFace) and abs(record.radius - R_PIPE) < 1e-9
    ]
    assert len(pipes) == 2 and list(view.pipes) == pipes


def test_ducts_are_told_once_the_feature_angle_is_and_sozzi_has_none(tmp_path):
    assert _view(tmp_path, {}).ducts is None
    assert _view(tmp_path, {}, feature_angle=0.785).ducts == ()


def test_a_bent_ducts_sides_are_its_ducts_faces_and_its_ends_are_not(tmp_path):
    # A leg, a bend and a leg: twelve sides round one duct, and its two end faces, neither a side.
    from tests.support.drawings import write_bent_duct

    drawn = write_bent_duct(tmp_path / "duct.step")
    model = read_step(drawn)
    region = DrawingRegion((model.names[0],))
    sections = DrawingSections(Drawing(drawn.name), region, {}, None, 0.785)
    view = view_drawing(sections, tmp_path, tmp_path / "out")
    assert len(view.faces) == 14 and len(view.ducts) == 12
    ends = sorted(set(range(14)) - set(view.ducts))
    # Each end is a 0.2 x 0.1 m rectangle; every side is larger.
    assert [view.faces[f]["area"] for f in ends] == pytest.approx([0.02, 0.02])
    assert min(view.faces[f]["area"] for f in view.ducts) > 0.02
    # A feature angle above the duct's right-angled corners makes no edge a corner: no duct.
    flat = DrawingSections(Drawing(drawn.name), region, {}, None, 1.7)
    assert view_drawing(flat, tmp_path, tmp_path / "flat").ducts == ()


def test_selections_that_share_a_face_are_refused_but_the_drawing_is_still_shown(tmp_path):
    lamp = FacesOn(("lamp",))
    view = _view(tmp_path, {"sleeve": lamp, "again": lamp})
    assert (
        view.patches is None and "selects a face another patch has already" in view.selections_error
    )
    assert view.surface.is_file() and len(view.faces) == 9


def test_until_a_region_is_chosen_only_the_drawings_solids_are_said(tmp_path):
    view = _view(tmp_path, {}, region=None)
    assert view.surface is None and view.faces == () and "lamp" in view.solids
    with pytest.raises(ValueError, match="which the drawing does not hold"):
        _view(tmp_path, {}, region=DrawingRegion(("tank",)))


def test_the_drawing_is_read_once_while_its_file_is_unchanged(tmp_path, monkeypatch):
    copy = tmp_path / "drawing.step"
    shutil.copy(SOZZI, copy)
    reads = []
    real = module.read_step
    monkeypatch.setattr(module, "read_step", lambda *a: reads.append(a) or real(*a))
    module._model.cache_clear()
    drawing = Drawing(copy.name)
    sections = DrawingSections(drawing, REGION, {})
    for _ in range(2):
        view_drawing(sections, tmp_path, tmp_path / "out")
    assert len(reads) == 1
    stat = copy.stat()
    os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))  # the file changed
    view_drawing(sections, tmp_path, tmp_path / "out")
    assert len(reads) == 2


def test_a_baffles_face_is_shown_in_the_baffles_patch_as_the_mesher_puts_it(tmp_path):
    from aquaflux.meshing import BAFFLE_PATCH
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pln, gp_Pnt

    from tests.support.step import write_step

    plane = gp_Pln(gp_Ax3(gp_Pnt(0.05, 0.0, 0.02), gp_Dir(1, 0, 0), gp_Dir(0, 1, 0)))
    write_step(
        tmp_path / "tank.step",
        {
            "tank": BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.1, 0.1, 0.1).Shape(),
            "plate": BRepBuilderAPI_MakeFace(plane, 0.0, 0.06, 0.0, 0.06).Face(),
        },
    )
    drawing = Drawing("tank.step")
    view = view_drawing(
        DrawingSections(drawing, DrawingRegion(("tank",)), {}), tmp_path, tmp_path / "out"
    )
    assert view.solids == ("tank",)
    assert sorted(view.patches) == sorted(["walls"] * 6 + [BAFFLE_PATCH])
    plate = view.patches.index(BAFFLE_PATCH)
    assert view.faces[plate]["area"] == pytest.approx(0.0036, rel=1e-9)
    with pytest.raises(ValueError, match=r"baffles \['lid'\], which are not sheets"):
        sections = DrawingSections(drawing, DrawingRegion(("tank",), baffles=("lid",)), {})
        view_drawing(sections, tmp_path, tmp_path / "o")
