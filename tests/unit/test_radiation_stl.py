"""Reading STL surfaces, in both formats and in the shapes real exporters produce."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from aquaflux.radiation.stl import read_stl

TRIANGLE = "\n".join(
    [
        " facet normal {nx} {ny} {nz}",
        "  outer loop",
        "   vertex 0 0 0",
        "   vertex 1 0 0",
        "   vertex 0 1 0",
        "  endloop",
        " endfacet",
    ]
)


def _ascii(tmp_path, body, name="surface.stl"):
    path = tmp_path / name
    path.write_text(body)
    return path


def _binary(tmp_path, triangles, header=b"binary stl", name="surface.stl"):
    """Write a binary STL by hand, so the reader is tested against the format and not itself."""
    payload = header.ljust(80, b"\0") + struct.pack("<I", len(triangles))
    for triangle in triangles:
        payload += np.zeros(3, "<f4").tobytes()
        payload += np.asarray(triangle, "<f4").tobytes()
        payload += struct.pack("<H", 0)
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_an_ascii_file_yields_its_triangles_in_file_order(tmp_path):
    path = _ascii(tmp_path, f"solid wall\n{TRIANGLE.format(nx=0, ny=0, nz=1)}\nendsolid wall\n")
    soup = read_stl(path)
    assert soup.n_facets == 1
    assert soup.vertices.shape == (1, 3, 3)
    np.testing.assert_allclose(soup.vertices[0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(soup.stored_normal[0], [0, 0, 1])


def test_each_triangle_is_attributed_to_the_body_it_was_declared_under(tmp_path):
    """Body names are how optical properties get assigned, so a misattribution silently
    reassigns emission from a lamp to a wall."""
    body = (
        f"solid wall\n{TRIANGLE.format(nx=0, ny=0, nz=1)}\nendsolid wall\n"
        f"solid lamp sleeve\n{TRIANGLE.format(nx=0, ny=0, nz=-1)}\n"
        f"{TRIANGLE.format(nx=0, ny=0, nz=-1)}\nendsolid lamp sleeve\n"
    )
    soup = read_stl(_ascii(tmp_path, body))
    assert soup.solid_names == ("wall", "lamp sleeve")
    np.testing.assert_array_equal(soup.solid_id, [0, 1, 1])


def test_a_body_name_may_contain_spaces(tmp_path):
    """A tokenizing parser silently truncates these, and the name is the lookup key."""
    body = f"solid outer reactor wall\n{TRIANGLE.format(nx=0, ny=0, nz=1)}\nendsolid\n"
    assert read_stl(_ascii(tmp_path, body)).solid_names == ("outer reactor wall",)


def test_scientific_notation_and_negative_coordinates_survive(tmp_path):
    body = (
        "solid s\n facet normal 0 0 1\n  outer loop\n"
        "   vertex -1.5e-3 0 0\n   vertex 1E2 0 0\n   vertex 0 +2.5 0\n"
        "  endloop\n endfacet\nendsolid\n"
    )
    np.testing.assert_allclose(
        read_stl(_ascii(tmp_path, body)).vertices[0],
        [[-1.5e-3, 0, 0], [100.0, 0, 0], [0, 2.5, 0]],
    )


def test_a_binary_file_yields_its_triangles(tmp_path):
    soup = read_stl(_binary(tmp_path, [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]))
    assert soup.n_facets == 1
    np.testing.assert_allclose(soup.vertices[0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert soup.solid_names == ("surface",)


def test_a_binary_file_whose_header_begins_with_solid_is_still_read_as_binary(tmp_path):
    """The trap the format detection exists for.

    An 80-byte binary header is arbitrary text and exporters have shipped ones beginning with
    the word ``solid``. A reader that sniffs the leading keyword tries an ASCII parse on binary
    data and fails on unparsable numbers -- or worse, finds a stray ``vertex`` byte sequence and
    returns garbage. Detection is by the file's own length arithmetic instead.
    """
    path = _binary(tmp_path, [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], header=b"solid exported by CAD")
    np.testing.assert_allclose(read_stl(path).vertices[0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_a_large_binary_file_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    triangles = rng.uniform(size=(500, 3, 3)).astype("<f4")
    soup = read_stl(_binary(tmp_path, triangles))
    assert soup.n_facets == 500
    np.testing.assert_allclose(soup.vertices, triangles, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "empty file"),
        ("solid s\nendsolid s\n", "no triangles"),
        (
            "solid s\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n"
            "   vertex 1 0 0\n  endloop\n endfacet\nendsolid\n",
            "not a whole number of triangles",
        ),
        (
            "solid s\n facet normal 0 0 1\n  outer loop\n   vertex x y z\n"
            "   vertex 1 0 0\n   vertex 0 1 0\n  endloop\n endfacet\nendsolid\n",
            "unparsable vertex coordinate",
        ),
        ("vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n", "no `solid` declaration"),
    ],
)
def test_a_malformed_file_is_refused_with_a_message_naming_the_problem(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        read_stl(_ascii(tmp_path, body))


def test_a_binary_file_declaring_no_triangles_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no triangles"):
        read_stl(_binary(tmp_path, []))


def test_a_file_without_per_facet_normals_still_reads(tmp_path):
    """Writers that omit the advisory normal must not lose their triangles with it."""
    body = (
        "solid s\n facet\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n"
        "   vertex 0 1 0\n  endloop\n endfacet\nendsolid\n"
    )
    soup = read_stl(_ascii(tmp_path, body))
    assert soup.n_facets == 1
    np.testing.assert_allclose(soup.stored_normal, np.zeros((1, 3)))
