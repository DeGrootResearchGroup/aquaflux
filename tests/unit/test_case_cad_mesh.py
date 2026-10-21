"""A case file's mesh made from a STEP drawing: how the section reads, writes and describes itself.

The meshing is pinned with Netgen in ``test_meshing_netgen.py`` (a case file's mesh written as a
polyMesh); here, the section itself, which needs neither Netgen nor the CAD kernel.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from aquaflux.case import CadMesh, CadPlacement, case_schema, mesh_source_from_mapping, relocated
from aquaflux.case.spec import _CASE_MAPPING
from aquaflux.meshing import ApexRelaxation, FacesNear, FacesOn, Graded, OctreeCore, PipeCore

SECTION = {
    "kind": "CadMesh",
    "file": "drawing.step",
    "solids": ["vessel", "inlet"],
    "without": ["lamp"],
    "max_size": 0.02,
    "feature_angle": 0.785,
    "placement": {"kind": "CadPlacement", "offset": [0.0, 0.0, 1.0]},
    "patches": {
        "inlet": {"kind": "FacesNear", "points": [[0.0, 0.0, 2.0]]},
        "lamp": {"kind": "FacesOn", "solids": ["lamp"]},
    },
    "others": "vessel_walls",
    "grading": 0.2,
    "mapped": ["lamp"],
    "pipes": ["inlet"],
    "pipe_core": {
        "kind": "PipeCore",
        "core_bow": 0.3,
        "shell": {"kind": "Graded", "cells": 3, "growth": 1.2},
    },
    "core": {"kind": "OctreeCore", "size": 0.08, "clearance": 0.03},
    "apex_relaxation": {"kind": "ApexRelaxation", "max_sweeps": 10},
}


def test_the_section_reads_into_its_selections_and_writes_back_as_given():
    source = mesh_source_from_mapping(SECTION)
    assert source == CadMesh(
        file="drawing.step",
        solids=("vessel", "inlet"),
        without=("lamp",),
        max_size=0.02,
        feature_angle=0.785,
        placement=CadPlacement(offset=(0.0, 0.0, 1.0)),
        patches={"inlet": FacesNear(((0.0, 0.0, 2.0),)), "lamp": FacesOn(("lamp",))},
        others="vessel_walls",
        grading=0.2,
        mapped=("lamp",),
        pipes=("inlet",),
        pipe_core=PipeCore(core_bow=0.3, shell=Graded(3, 1.2)),
        core=OctreeCore(size=0.08, clearance=0.03),
        apex_relaxation=ApexRelaxation(max_sweeps=10),
    )
    assert _CASE_MAPPING.to_mapping(source) == SECTION


def test_the_mesher_takes_the_set_settings_and_leaves_the_rest_to_its_own_defaults():
    mesher = mesh_source_from_mapping(SECTION).mesher()
    assert mesher.pipe_core == PipeCore(core_bow=0.3, shell=Graded(3, 1.2))
    assert (mesher.max_size, mesher.feature_angle, mesher.grading, mesher.min_size) == (
        0.02,
        0.785,
        0.2,
        None,
    )
    # The core's unset wall size is the mesher's size; its clearance is as set.
    assert mesher.core.resolved(mesher.max_size) == (0.02, 0.03)
    assert mesher.apex_relaxation == ApexRelaxation(max_sweeps=10)


def test_an_unset_setting_is_described_by_what_it_falls_through_to():
    fields = {field["name"]: field for field in case_schema()["kinds"]["CadMesh"]["fields"]}
    assert fields["others"]["resolved_default"] == "walls"
    assert fields["min_size"]["resolved_default"] == 0.0
    assert fields["grading"]["resolved_default"] == 0.3
    assert fields["max_size"]["required"] and fields["feature_angle"]["required"]


def test_the_drawing_is_re_based_with_the_case_file():
    moved = relocated(mesh_source_from_mapping(SECTION), Path("cases/a"), Path("cases"))
    assert moved.file == "a/drawing.step"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"solids": []}, "names the solids whose union is the region"),
        ({"max_size": 0.0}, "CadMesher.max_size must be a positive, finite number"),
        ({"feature_angle": -0.1}, "CadMesher.feature_angle must be a positive, finite number"),
        ({"grading": math.inf}, "CadMesher.grading must be a positive, finite number"),
        (
            {"patches": {"inlet": {"kind": "FacesOn", "solids": []}}},
            "FacesOn names the solids whose surface the faces lie on",
        ),
        (
            {"patches": {"inlet": {"kind": "FacesOn", "solids": ["a"], "tolerance": 0}}},
            "FacesOn.tolerance must be a positive, finite number",
        ),
        (
            {"core": {"kind": "OctreeCore", "size": 0.08, "wall_size": 0.1}},
            r"OctreeCore.wall_size \(0.1\) may not exceed its size",
        ),
        (
            {"core": {"kind": "OctreeCore", "size": 0.01}},
            r"is the boundary's size \(0.02\), which exceeds the core's size",
        ),
    ],
)
def test_a_section_out_of_range_is_refused_by_name(change, message):
    with pytest.raises(ValueError, match=message):
        mesh_source_from_mapping({**SECTION, **change})


def test_a_patch_that_is_not_a_face_selection_is_refused():
    with pytest.raises(TypeError, match=r"CadMesh.patches\['inlet'\] must be a face selection"):
        CadMesh(
            file="drawing.step",
            solids=("vessel",),
            max_size=0.02,
            feature_angle=0.785,
            patches={"inlet": "the pipe's end"},
        )
