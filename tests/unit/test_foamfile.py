"""Unit tests for the OpenFOAM FoamFile envelope and body-grammar parsers, on text snippets."""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.io.openfoam.foamfile import is_binary, parse_foamfile
from aquaflux.io.openfoam.grammar import (
    parse_boundary,
    parse_cell_zones,
    parse_face_list,
    parse_scalar_list,
    parse_vector_list,
)

_HEADER = (
    "/*---------------------------------------------------------------------------*\\\n"
    "| a banner comment                                                          |\n"
    "\\*---------------------------------------------------------------------------*/\n"
    "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class vectorField;\n"
    '    location "constant/polyMesh";\n    object points;\n}\n'
    "// * * * * * * * * * * * * * //\n"
)


def test_parse_foamfile_strips_comments_and_reads_header():
    foam = parse_foamfile(_HEADER + "0\n(\n)\n")
    assert foam.header["format"] == "ascii"
    assert foam.header["object"] == "points"
    assert foam.header["location"] == "constant/polyMesh"  # quotes stripped
    assert not is_binary(foam)


def test_is_binary_detects_binary_format():
    foam = parse_foamfile("FoamFile { format binary; class vectorField; }\n0\n(\n)\n")
    assert is_binary(foam)


def test_parse_foamfile_without_header_raises():
    with pytest.raises(ValueError, match="no 'FoamFile"):
        parse_foamfile("3\n(\n(0 0 0)\n)\n")


def test_parse_vector_list():
    points = parse_vector_list("3\n(\n(0 0 0)\n(1 0 0)\n(0.5 2 -1)\n)")
    assert points.shape == (3, 3)
    np.testing.assert_allclose(points[2], [0.5, 2.0, -1.0])


def test_parse_vector_list_empty():
    assert parse_vector_list("0\n(\n)").shape == (0, 3)


def test_parse_vector_list_count_mismatch_raises():
    with pytest.raises(ValueError, match=r"declares 3 .* lists 2"):
        parse_vector_list("3\n(\n(0 0 0)\n(1 0 0)\n)")


def test_parse_scalar_list():
    np.testing.assert_array_equal(parse_scalar_list("3\n(\n0\n0\n1\n)"), [0, 0, 1])


def test_parse_face_list_ragged():
    offsets, indices = parse_face_list("2\n(\n4(0 1 2 3)\n3(4 5 6)\n)")
    np.testing.assert_array_equal(offsets, [0, 4, 7])
    np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4, 5, 6])


def test_parse_face_list_node_count_mismatch_raises():
    with pytest.raises(ValueError, match="declares 4 nodes but lists 3"):
        parse_face_list("1\n(\n4(0 1 2)\n)")


def test_parse_boundary():
    patches = parse_boundary(
        "2\n(\n"
        "  inlet\n  {\n    type patch;\n    nFaces 1;\n    startFace 10;\n  }\n"
        "  walls\n  {\n    type wall;\n    inGroups 1(walls);\n    nFaces 8;\n    startFace 11;\n  }\n"
        ")\n"
    )
    assert [p.name for p in patches] == ["inlet", "walls"]
    assert patches[1].type_ == "wall"
    assert patches[1].start_face == 11
    assert patches[1].n_faces == 8


def test_parse_boundary_missing_entry_raises():
    with pytest.raises(ValueError, match="missing"):
        parse_boundary("1\n(\n  inlet { type patch; nFaces 1; }\n)\n")


def test_parse_cell_zones():
    zones = parse_cell_zones(
        "1\n(\n  fluid\n  {\n    type cellZone;\n    cellLabels List<label> 3 ( 0 1 2 );\n  }\n)\n"
    )
    assert zones[0].name == "fluid"
    np.testing.assert_array_equal(zones[0].cell_labels, [0, 1, 2])
