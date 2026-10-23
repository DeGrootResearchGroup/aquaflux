"""A mesh file, and making the mesh it describes: how the file reads, writes and refuses, and what
``aquaflux mesh`` writes.

The meshing itself is pinned with Netgen in ``test_meshing_netgen.py``; here the region's meshing is
replaced by a structured box where a mesh must be made, so neither Netgen nor the CAD kernel runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from aquaflux.__main__ import main
from aquaflux.case import (
    AquafluxMesh,
    CadPlacement,
    Drawing,
    DrawingRegion,
    MeshOutput,
    MeshSizes,
    MeshSpec,
    OpenFOAMMesh,
    drawing_sections_from_mapping,
    make_mesh,
    mesh_schema,
    mesh_spec_from_mapping,
    mesh_spec_to_mapping,
    read_mesh_file,
    relocated,
    write_mesh_file,
)
from aquaflux.io import read_aqmesh, read_aqmesh_header, read_openfoam
from aquaflux.mesh import QualitySummary, structured_grid_3d
from aquaflux.meshing import (
    FacePlanarizer,
    FacesNear,
    FacesOn,
    Graded,
    OctreeCore,
    PipeCore,
    PipeGrading,
)

DOCUMENT = {
    "drawing": {"file": "drawing.step", "placement": {"kind": "CadPlacement", "offset": [0.0, 0.0, 1.0]}},
    "region": {"solids": ["vessel", "inlet"], "without": ["lamp"]},
    "sizes": {"max_size": 0.02, "feature_angle": 0.785, "grading": 0.2},
    "patches": {
        "inlet": {"kind": "FacesNear", "points": [[0.0, 0.0, 2.0]]},
        "lamp": {"kind": "FacesOn", "solids": ["lamp"]},
    },
    "others": "vessel_walls",
    "mapped": ["lamp"],
    "pipes": ["inlet"],
    "pipe_core": {
        "kind": "PipeCore",
        "core_bow": 0.3,
        "shell": {"kind": "Graded", "cells": 3, "growth": 1.2},
    },
    "pipe_grading": {"kind": "PipeGrading", "growth": 1.1, "max_length": 0.05},
    "core": {"kind": "OctreeCore", "size": 0.08, "clearance": 0.03},
    "output": {"path": "reactor.aqmesh"},
}  # fmt: skip


def test_the_file_reads_into_its_sections_and_writes_back_as_given():
    spec = mesh_spec_from_mapping(DOCUMENT)
    assert spec == MeshSpec(
        drawing=Drawing("drawing.step", CadPlacement(offset=(0.0, 0.0, 1.0))),
        region=DrawingRegion(("vessel", "inlet"), ("lamp",)),
        sizes=MeshSizes(max_size=0.02, feature_angle=0.785, grading=0.2),
        patches={"inlet": FacesNear(((0.0, 0.0, 2.0),)), "lamp": FacesOn(("lamp",))},
        others="vessel_walls",
        mapped=("lamp",),
        pipes=("inlet",),
        pipe_core=PipeCore(core_bow=0.3, shell=Graded(3, 1.2)),
        pipe_grading=PipeGrading(growth=1.1, max_length=0.05),
        core=OctreeCore(size=0.08, clearance=0.03),
        output=MeshOutput(path="reactor.aqmesh"),
    )
    assert mesh_spec_to_mapping(spec) == DOCUMENT


def test_a_file_on_disk_reads_back_equal_and_relative_to_its_own_directory(tmp_path):
    spec = mesh_spec_from_mapping(DOCUMENT)
    write_mesh_file(spec, tmp_path / "mesh.yaml")
    assert yaml.safe_load((tmp_path / "mesh.yaml").read_text()) == DOCUMENT
    read = read_mesh_file(tmp_path / "mesh.yaml")
    assert (read.spec, read.directory) == (spec, tmp_path)


def test_the_mesher_takes_the_set_settings_and_leaves_the_rest_to_its_own_defaults():
    mesher = mesh_spec_from_mapping(DOCUMENT).mesher()
    assert mesher.pipe_core == PipeCore(core_bow=0.3, shell=Graded(3, 1.2))
    assert mesher.pipe_grading == PipeGrading(growth=1.1, max_length=0.05)
    assert (mesher.max_size, mesher.feature_angle, mesher.grading, mesher.min_size) == (
        0.02, 0.785, 0.2, None,
    )  # fmt: skip
    # The core's unset wall size is the mesher's size; its clearance is as set.
    assert mesher.core.resolved(mesher.max_size) == (0.02, 0.03)


def test_an_unset_setting_is_described_by_what_it_falls_through_to():
    kinds = mesh_schema()["kinds"]
    spec = {field["name"]: field for field in kinds["MeshSpec"]["fields"]}
    sizes = {field["name"]: field for field in kinds["MeshSizes"]["fields"]}
    assert spec["others"]["resolved_default"] == "walls"
    assert sizes["min_size"]["resolved_default"] == 0.0
    assert sizes["grading"]["resolved_default"] == 0.3
    assert sizes["max_size"]["required"] and sizes["feature_angle"]["required"]
    assert mesh_schema()["root"] == "MeshSpec"
    assert mesh_schema()["one_form_sections"] == ["drawing", "region", "sizes", "output"]


def test_the_drawing_and_the_outputs_are_re_based_with_the_file():
    spec = mesh_spec_from_mapping(
        {**DOCUMENT, "output": {"path": "reactor.aqmesh", "openfoam": "of_case"}}
    )
    moved = relocated(spec, Path("meshes/a"), Path("meshes"))
    assert moved.drawing.file == "a/drawing.step"
    assert (moved.output.path, moved.output.openfoam) == ("a/reactor.aqmesh", "a/of_case")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"region": {"solids": []}}, "names the solids whose union is the region"),
        ({"sizes": {"max_size": 0.0, "feature_angle": 0.785}}, "CadMesher.max_size must be a pos"),
        ({"sizes": {"max_size": 0.02, "feature_angle": -0.1}}, "CadMesher.feature_angle must be"),
        (
            {"sizes": {"max_size": 0.02, "feature_angle": 0.785, "grading": math.inf}},
            "CadMesher.grading must be a positive, finite number",
        ),
        ({"output": {"path": "reactor.msh"}}, r"written to a \.aqmesh file"),
        ({"layered": ["walls"]}, "takes both layers and layered, or neither"),
        (
            {"patches": {"inlet": {"kind": "FacesOn", "solids": []}}},
            "FacesOn names the solids whose surface the faces lie on",
        ),
        (
            {"core": {"kind": "OctreeCore", "size": 0.01}},
            r"is the boundary's size \(0.02\), which exceeds the core's size",
        ),
        ({"kind": "CaseSpec"}, "names no kind"),
    ],
)
def test_a_file_out_of_range_is_refused_by_name(change, message):
    with pytest.raises(ValueError, match=message):
        mesh_spec_from_mapping({**DOCUMENT, **change})


def test_a_patch_that_is_not_a_face_selection_is_refused():
    with pytest.raises(TypeError, match=r"MeshSpec.patches\['inlet'\] must be a face selection"):
        MeshSpec(
            drawing=Drawing("drawing.step"),
            region=DrawingRegion(("vessel",)),
            sizes=MeshSizes(max_size=0.02, feature_angle=0.785),
            patches={"inlet": "the pipe's end"},
        )


def test_a_planarizer_reads_from_its_kind_and_reaches_the_mesher():
    document = {**DOCUMENT, "planarizer": {"kind": "FacePlanarizer", "shortening": 10.0}}
    spec = mesh_spec_from_mapping(document)
    assert spec.mesher().planarizer == FacePlanarizer(shortening=10.0)
    assert mesh_spec_to_mapping(spec) == document
    assert mesh_spec_from_mapping(DOCUMENT).mesher().planarizer is None


# --- making the mesh -------------------------------------------------------------------------

BOX = structured_grid_3d(2, 3, 2, 1.0, 1.5, 1.0, named_boundaries=True)


def _mesh_file(directory: Path, **output) -> Path:
    path = directory / "mesh.yaml"
    write_mesh_file(mesh_spec_from_mapping({**DOCUMENT, "output": output or {}}), path)
    return path


@pytest.fixture
def boxed(monkeypatch):
    """The region's meshing replaced by a box, counting the times it is asked for."""
    calls = []

    def mesh(self, directory, on_stage=None):
        calls.append(directory)
        if on_stage is not None:
            on_stage("meshing the box")
        return BOX

    monkeypatch.setattr(MeshSpec, "mesh", mesh)
    return calls


def test_making_the_mesh_writes_it_with_how_it_was_made(tmp_path, boxed):
    path = _mesh_file(tmp_path, path="out/reactor.aqmesh", openfoam="of_case")
    record = make_mesh(path)
    assert boxed == [tmp_path]
    assert record.aqmesh == tmp_path / "out" / "reactor.aqmesh"
    assert read_aqmesh(record.aqmesh).n_cells == BOX.n_cells
    assert read_openfoam(tmp_path / "of_case").n_cells == BOX.n_cells
    provenance = read_aqmesh_header(record.aqmesh).provenance
    assert provenance["settings"] == mesh_spec_to_mapping(read_mesh_file(path).spec)
    assert provenance["mesh_file"] == str(path.resolve())
    assert {"version", "commit", "modified"} <= set(provenance["aquaflux"])
    assert provenance["seconds"] >= 0.0 and provenance["started"]
    # The quality is the mesh's own, so it can be judged without being read.
    assert (
        provenance["quality"] == QualitySummary.of(BOX).to_mapping() == record.quality.to_mapping()
    )
    # A case reading the mesh sees the same record; a polyMesh records none.
    assert AquafluxMesh(str(record.aqmesh)).provenance(tmp_path) == provenance
    assert OpenFOAMMesh("of_case").provenance(tmp_path) is None


def test_each_stage_is_told_as_it_starts_then_measuring_and_writing(tmp_path, boxed):
    told = []
    make_mesh(_mesh_file(tmp_path), on_stage=told.append)
    assert told == ["meshing the box", "measuring the mesh's quality", "writing the mesh"]


def test_a_mesh_already_there_is_refused_before_anything_is_meshed(tmp_path, boxed):
    path = _mesh_file(tmp_path, path="reactor.aqmesh", openfoam="of_case")
    make_mesh(path)
    for target in (tmp_path / "reactor.aqmesh", tmp_path / "of_case" / "constant" / "polyMesh"):
        with pytest.raises(FileExistsError, match="already there"):
            make_mesh(path)
        assert len(boxed) == 1  # refused without meshing
        if target.suffix == ".aqmesh":
            target.unlink()
    make_mesh(path, overwrite=True)
    assert len(boxed) == 2


def test_the_command_makes_the_mesh_and_refuses_a_bad_file(tmp_path, boxed, capsys):
    path = _mesh_file(tmp_path)
    assert main(["mesh", str(path)]) == 0
    out = capsys.readouterr().out
    assert f"{BOX.n_cells} cells (3D)" in out and str(tmp_path / "mesh.aqmesh") in out
    lines = out.splitlines()
    # Each stage on a line of its own as it starts, after the seconds since meshing began.
    assert [line.split("s  ", 1)[1] for line in lines[:3]] == [
        "meshing the box", "measuring the mesh's quality", "writing the mesh",
    ]  # fmt: skip
    assert "quality: non-orthogonality 0.0 deg at the 99th percentile" in out
    assert main(["mesh", str(path)]) == 2  # already there
    assert "already there" in capsys.readouterr().err
    bad = tmp_path / "bad.yaml"
    bad.write_text("region: {solids: []}\n")
    assert main(["mesh", str(bad)]) == 2


def test_the_drawing_and_its_patches_are_read_while_the_rest_is_unfinished():
    unfinished = {key: DOCUMENT[key] for key in ("drawing", "region", "patches", "others")}
    drawing, region, patches, others = drawing_sections_from_mapping(unfinished)
    spec = mesh_spec_from_mapping(DOCUMENT)
    assert (drawing, region, patches, others) == (
        spec.drawing, spec.region, dict(spec.patches), spec.others,
    )  # fmt: skip
    # A region not yet given, or naming no solid yet, is not a refusal: the drawing is still shown.
    for region_section in ({}, {"region": {"solids": []}}):
        read = drawing_sections_from_mapping({"drawing": DOCUMENT["drawing"], **region_section})
        assert read[1:] == (None, {}, None)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"region": DOCUMENT["region"]}, "drawing is needed"),
        (
            {**DOCUMENT, "patches": {"inlet": {"kind": "FacesNear", "points": [[1.0]]}}},
            "at 'patches.inlet'",
        ),
        ({**DOCUMENT, "patches": {"inlet": {"kind": "MeshOutput"}}}, "a patch is a face selection"),
        ({**DOCUMENT, "others": 3}, "others names a patch"),
    ],
)
def test_the_drawings_sections_are_refused_by_name(document, message):
    with pytest.raises(ValueError, match=message):
        drawing_sections_from_mapping(document)
