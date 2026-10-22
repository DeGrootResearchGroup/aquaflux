"""A surface of triangles written as one ``.vtp``, read back by the standard library alone.

Written as decimal text, so the file's points, polygons and cell values can be read straight back
and compared with the triangles given.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from aquaflux.io import write_triangles

#: Two triangles sharing an edge, and a third apart; corners shared by two are written once.
CORNERS = np.array(
    [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]],
    ]
)


def _array(piece, name):
    node = next(a for a in piece.iter("DataArray") if a.get("Name") == name)
    return np.array(node.text.split(), dtype=float)


def test_the_triangles_and_their_values_read_back_with_shared_corners_written_once(tmp_path):
    path = write_triangles(
        CORNERS, {"face": np.array([4, 4, 7])}, tmp_path / "drawing.vtp", binary=False
    )
    piece = ET.parse(path).getroot().find("PolyData/Piece")
    assert (piece.get("NumberOfPoints"), piece.get("NumberOfPolys")) == ("7", "3")
    points = _array(piece, "Points").reshape(-1, 3)
    connectivity = _array(piece, "connectivity").astype(int).reshape(-1, 3)
    np.testing.assert_array_equal(points[connectivity], CORNERS)  # each triangle as given, in order
    np.testing.assert_array_equal(_array(piece, "offsets"), [3, 6, 9])
    np.testing.assert_array_equal(_array(piece, "face"), [4, 4, 7])


def test_corners_not_three_by_three_and_values_of_another_length_are_refused(tmp_path):
    with pytest.raises(ValueError, match=r"shape \(n, 3, 3\)"):
        write_triangles(CORNERS[:, :2], None, tmp_path / "a.vtp")
    with pytest.raises(ValueError, match="face"):
        write_triangles(CORNERS, {"face": np.array([1, 2])}, tmp_path / "b.vtp")
