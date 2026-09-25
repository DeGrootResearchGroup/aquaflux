"""Reading STEP files into exact bodies and triangles, through the CAD kernel.

Two kinds of file. The Sozzi & Taghipour reactor's own drawing (an Onshape export in metres, body
axis along ``y``) is checked against the tutorial's dimensions, which are typed independently of
anything read here. Small synthetic files are written through the kernel in this module — in
millimetres, and as an assembly of placed instances — so the unit conversion and the assembly
locations are exercised on files that differ from that one in exactly those respects.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")

from aquaflux.io.cad import (
    CurvedPieces,
    InexactBody,
    Placement,
    Recognition,
    RecognitionRule,
    UnrecognizedSolid,
    read_step,
)
from aquaflux.solids import Cone, Cylinder, Intersection, Outside, Union
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeTorus,
)
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool

SOZZI = (
    Path(__file__).resolve().parents[2]
    / "validation/uvreactor_openfoam/of_case/SozziTaghipour.step"
)
#: The drawing has the body axis along y; the case, and the tutorial's meshes, along x.
SWAP_XY = Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
VESSEL = ("reactor_body", "inlet_pipe", "outlet_pipe")

# The tutorial's own dimensions, in the case frame, typed from its geometry script.
R_BODY, X_BODY_END, R_PIPE, X_RISER, R_LAMP, X_LAMP_TIP = (
    0.0445,
    0.889,
    0.00955,
    0.04765,
    0.010,
    0.80,
)


@pytest.fixture(scope="module")
def sozzi():
    return read_step(SOZZI, SWAP_XY)


def write_step(path, parts, instances=None):
    """Write named solids to STEP in millimetres; ``instances`` places a part several times.

    ``parts`` maps a name to a kernel solid given in metres. ``instances``, if given, maps a part's
    name to translations (in metres); the part is then written once and placed at each, as an
    assembly, which is how a CAD tool writes a repeated component.
    """
    document = TDocStd_Document(TCollection_ExtendedString("test"))
    XCAFDoc_DocumentTool.SetLengthUnit_s(document, 1.0)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    assembly = tool.NewShape() if instances else None
    for name, shape in parts.items():
        label = tool.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
        for offset in (instances or {}).get(name, ()):
            move = gp_Trsf()
            move.SetTranslation(gp_Vec(*offset))
            tool.AddComponent(assembly, label, TopLoc_Location(move))
    if assembly is not None:
        tool.UpdateAssemblies()
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = STEPCAFControl_Writer()
    writer.Transfer(document, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def cylinder(base, axis, radius, height):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*base), gp_Dir(*axis)), radius, height).Shape()


# ---------------------------------------------------------------------------------------------
# The Sozzi reactor's own drawing
# ---------------------------------------------------------------------------------------------


def test_the_sozzi_drawing_reads_as_the_tutorials_dimensions(sozzi):
    assert sozzi.names == ("lamp", "outlet_pipe", "inlet_pipe", "reactor_body")

    body = sozzi.solid("reactor_body")
    assert isinstance(body, Cylinder)
    np.testing.assert_allclose(np.asarray(body.centre), [X_BODY_END / 2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(np.abs(np.asarray(body.axis)), [1, 0, 0], atol=1e-12)
    assert float(body.radius) == pytest.approx(R_BODY, abs=1e-12)
    assert float(body.half_length) == pytest.approx(X_BODY_END / 2, abs=1e-12)

    riser = sozzi.recognition("outlet_pipe").body
    np.testing.assert_allclose(np.asarray(riser.centre)[:2], [X_RISER, 0], atol=1e-12)
    np.testing.assert_allclose(np.abs(np.asarray(riser.axis)), [0, 0, 1], atol=1e-12)
    assert float(riser.radius) == pytest.approx(R_PIPE, abs=1e-12)

    lamp = sozzi.solid("lamp")
    assert isinstance(lamp, Union)
    tip = np.array([[X_LAMP_TIP + R_LAMP - 1e-6, 0, 0]])
    past_tip = np.array([[X_LAMP_TIP + R_LAMP + 1e-6, 0, 0]])
    assert bool(lamp.contains(tip)[0]) and not bool(lamp.contains(past_tip)[0])


def test_the_vessel_reads_as_a_fluid_checked_to_the_files_own_tolerance(sozzi):
    water = sozzi.fluid(*VESSEL)
    assert isinstance(water, Outside) and len(water.regions) == 3
    distance, tolerance = sozzi.discrepancy(VESSEL)
    assert tolerance == pytest.approx(1e-5), "the drawing declares 10 micrometres"
    assert distance < 0.2 * tolerance


def test_a_pipe_cut_to_fit_the_vessel_is_refused_on_its_own(sozzi):
    with pytest.raises(InexactBody, match="neighbour"):
        sozzi.solid("outlet_pipe")


class _Shrunk(RecognitionRule):
    """The curved-pieces proposal with every cylinder one percent too thin: a plausible misread."""

    def propose(self, description):
        found = CurvedPieces().propose(description)
        if isinstance(found, Recognition) and isinstance(found.body, Cylinder):
            b = found.body
            thin = Cylinder(b.centre, b.axis, 0.99 * float(b.radius), b.half_length)
            return Recognition(thin, found.stands_alone, found.proven, "shrunk")
        return found


def test_a_proposal_that_is_wrong_by_a_percent_is_refused():
    wrong = read_step(SOZZI, SWAP_XY, rules=(_Shrunk(),))
    with pytest.raises(InexactBody, match="refused rather than used"):
        wrong.fluid(*VESSEL)


class _Flanged(RecognitionRule):
    """The chamber proposed with a thin phantom flange standing off its wall.

    A defect only one direction of the comparison can see: every point of the drawing's surface is
    within the flange's thickness of the proposal's, and only the flange's own rim, sampled from the
    proposal's side, is far from anything in the drawing.
    """

    def propose(self, description):
        found = CurvedPieces().propose(description)
        if description.name != "reactor_body":
            return found
        flange = Cylinder([0.4, 0.0, 0.0], [1.0, 0.0, 0.0], 0.06, 1e-6)
        return Recognition(Union(found.body, flange), True, False, "flanged")


def test_a_phantom_flange_on_the_proposal_is_refused():
    wrong = read_step(SOZZI, SWAP_XY, rules=(_Flanged(),))
    with pytest.raises(InexactBody, match="refused rather than used"):
        wrong.fluid(*VESSEL)


# ---------------------------------------------------------------------------------------------
# Synthetic files: units, assemblies, and the other shapes
# ---------------------------------------------------------------------------------------------


def test_lengths_are_metres_whatever_unit_the_file_is_written_in(tmp_path):
    path = write_step(tmp_path / "pipe.step", {"pipe": cylinder((0, 0, 0), (0, 0, 1), 0.01, 0.2)})
    assert "MILLI" in path.read_text(), "the fixture must really be written in millimetres"
    pipe = read_step(path).solid("pipe")
    assert float(pipe.radius) == pytest.approx(0.01, rel=1e-12)
    assert float(pipe.half_length) == pytest.approx(0.1, rel=1e-12)


def test_every_instance_of_an_assembly_is_read_where_it_is_placed(tmp_path):
    path = write_step(
        tmp_path / "pair.step",
        {"pipe": cylinder((0, 0, 0), (0, 0, 1), 0.01, 0.2)},
        instances={"pipe": [(0.1, 0, 0), (0, 0.3, 0)]},
    )
    model = read_step(path)
    assert model.names == ("pipe[0]", "pipe[1]")
    centres = sorted(tuple(np.round(np.asarray(model.solid(n).centre), 12)) for n in model.names)
    assert centres == [(0.0, 0.3, 0.1), (0.1, 0.0, 0.1)]


def test_a_box_is_a_proven_polyhedron_and_a_cone_is_a_cone(tmp_path):
    box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.1, 0.2, 0.3).Shape()
    cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(1, 0, 0), gp_Dir(0, 0, 1)), 0.05, 0.02, 0.1).Shape()
    model = read_step(write_step(tmp_path / "parts.step", {"box": box, "reducer": cone}))

    assert model.recognition("box").proven
    plate = model.solid("box")
    assert isinstance(plate, Intersection)
    assert bool(plate.contains(np.array([[0.05, 0.1, 0.15]]))[0])
    assert not bool(plate.contains(np.array([[0.05, 0.21, 0.15]]))[0])

    reducer = model.solid("reducer")
    assert isinstance(reducer, Cone)
    assert sorted([float(reducer.base_radius), float(reducer.tip_radius)]) == pytest.approx(
        [0.02, 0.05], abs=1e-12
    )


def test_a_thin_fin_the_rules_do_not_see_is_refused(tmp_path):
    """A plate fin on a pipe: the curved-pieces rule proposes the bare cylinder and misses it.

    The fin holds almost no volume, which is why the check is a distance and not a volume. And only
    one direction of the comparison can see it: every point of the proposal is within half the fin's
    thickness of the drawing, and only the fin's own edge, sampled from the drawing's side, is far
    from the proposal.
    """
    pipe = cylinder((0, 0, 0), (0, 0, 1), 0.01, 0.2)
    fin = BRepPrimAPI_MakeBox(gp_Pnt(0.005, -2e-4, 0.05), 0.02, 4e-4, 0.1).Shape()
    finned = BRepAlgoAPI_Fuse(pipe, fin).Shape()
    model = read_step(write_step(tmp_path / "finned.step", {"finned": finned}))
    assert isinstance(model.recognition("finned").body, Cylinder), "the rule must miss the fin"
    with pytest.raises(InexactBody, match="refused rather than used"):
        model.solid("finned")


def test_a_polyhedron_takes_part_in_a_fluid_as_the_solid_it_is(tmp_path):
    """A rectangular channel with a pipe off it: the proven polyhedron is compared as its solid.

    A half-space has no finite counterpart to build in the kernel, so rebuilding the channel from
    its half-spaces cannot work; the drawing's own solid stands in for it.
    """
    channel = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.4, 0.05, 0.05).Shape()
    riser = cylinder((0.2, 0.025, 0.04), (0, 0, 1), 0.01, 0.2)
    model = read_step(write_step(tmp_path / "channel.step", {"channel": channel, "riser": riser}))
    water = model.fluid("channel", "riser")
    assert len(water.regions) == 2
    distance, tolerance = model.discrepancy(("channel", "riser"))
    assert distance <= tolerance


def test_a_torus_is_refused_by_name(tmp_path):
    ring = BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.1, 0.02).Shape()
    model = read_step(write_step(tmp_path / "ring.step", {"elbow": ring}))
    with pytest.raises(UnrecognizedSolid, match="Torus"):
        model.solid("elbow")


# ---------------------------------------------------------------------------------------------
# Triangles
# ---------------------------------------------------------------------------------------------


def test_the_lamp_triangulates_outward_onto_its_true_surface_and_to_size(sozzi):
    """Through the reflecting placement, which reverses every triangle unless it is handled."""
    size = 0.01
    triangles = sozzi.triangles("lamp", chord=2e-5, facet_size=size)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.cross(b - a, c - a)
    assert np.all(np.linalg.norm(cross, axis=1) > 0.0), "no zero-area facet may reach an emitter"

    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    exact_volume = math.pi * R_LAMP**2 * X_LAMP_TIP + 2.0 / 3.0 * math.pi * R_LAMP**3
    assert volume == pytest.approx(exact_volume, rel=2e-3), "wound outward: the volume is positive"

    # Area, and not only volume: meshing a solid cut into pieces adds the cuts' internal faces in
    # oppositely wound pairs, which cancel in the volume and double up in the area.
    exact_area = 2.0 * math.pi * R_LAMP * X_LAMP_TIP + 3.0 * math.pi * R_LAMP**2
    assert 0.5 * np.linalg.norm(cross, axis=1).sum() == pytest.approx(exact_area, rel=2e-3)

    edges = np.linalg.norm(np.roll(triangles, -1, axis=1) - triangles, axis=2)
    assert edges.max() <= math.sqrt(3.0) * size

    # The vertices lie on the drawing's surface, which is the ideal lamp only to the precision the
    # drawing was exported at -- so that, and not rounding, is what they are held to.
    lamp = sozzi.solid("lamp")
    _, tolerance = sozzi.discrepancy(["lamp"])
    assert np.abs(np.asarray(lamp.signed_distance(triangles.reshape(-1, 3)))).max() < tolerance
