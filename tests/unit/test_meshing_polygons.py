"""A surface of triangles and quadrilaterals, and the pyramids capping its quadrilaterals.

Checked on a cube's surface (its faces quadrilaterals, or its top cut into triangles): the
pyramids on its faces enclose, with the cube, the cube and the caps' volume; the caps' sides with
the surface's triangles bound that region and nothing else.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from aquaflux.meshing import Polygons, PyramidCaps, element_mesh

from tests.support.meshes import voxel_surface


def test_a_quadrilateral_that_is_not_flat_has_the_newell_normal():
    # Newell's normal (the vector area of the closed polygon) by its own formula, independently.
    points = np.array([[0, 0, 0], [1.2, 0, 0.3], [1, 0.9, -0.1], [-0.1, 1, 0.4]], float)
    nxt = np.roll(points, -1, axis=0)
    newell = np.array([
        np.sum((points[:, 1] - nxt[:, 1]) * (points[:, 2] + nxt[:, 2])),
        np.sum((points[:, 2] - nxt[:, 2]) * (points[:, 0] + nxt[:, 0])),
        np.sum((points[:, 0] - nxt[:, 0]) * (points[:, 1] + nxt[:, 1])),
    ])  # fmt: skip
    normal = Polygons(quads=[[0, 1, 2, 3]]).normals(points)[0]
    np.testing.assert_allclose(normal, newell / np.linalg.norm(newell), rtol=1e-12)


def test_a_quantity_per_face_runs_over_the_triangles_then_the_quadrilaterals():
    faces = Polygons([[0, 1, 2], [2, 1, 3]], [[0, 1, 4, 5]])
    triangles, quads = faces.split(np.array([7, 8, 9]))
    np.testing.assert_array_equal(triangles, [7, 8])
    np.testing.assert_array_equal(quads, [9])
    chosen = faces.where(np.array([False, True, True]))
    np.testing.assert_array_equal(chosen.triangles, [[2, 1, 3]])
    np.testing.assert_array_equal(chosen.quads, [[0, 1, 4, 5]])
    np.testing.assert_array_equal(faces.padded()[-1], [0, 1, 4, 5])
    np.testing.assert_array_equal(faces.padded()[0], [0, 1, 2, -1])
    np.testing.assert_array_equal(faces.triangle_face(), [0, 1, 2, 2])


@pytest.mark.parametrize("height", [0.1, 0.25, 0.4])
def test_caps_on_a_cubes_faces_close_it_and_add_their_volume(height):
    points, quads = voxel_surface([(0, 0, 0)], 2)
    caps = PyramidCaps.on(points, quads, height)
    every = np.vstack([points, caps.apexes])
    # Each apex height times its edge (1/2) out of the cube, over its quadrilateral's centre.
    centre = points[quads].mean(axis=1)
    outward = Polygons(quads=quads).normals(points)
    np.testing.assert_allclose(caps.apexes, centre + 0.5 * height * outward, atol=1e-15)
    # The caps' sides bound the cube with its caps, wound into it: edges used once each way.
    used = Counter(map(tuple, Polygons(caps.sides).edges()[0].tolist()))
    assert all(count == 1 and used.get((b, a)) == 1 for (a, b), count in used.items())
    corner = every[caps.sides]
    enclosed = -np.einsum("ij,ij->i", corner[:, 0], np.cross(corner[:, 1], corner[:, 2])).sum() / 6
    pyramid = len(quads) * 0.25 * (0.5 * height) / 3.0
    assert enclosed == pytest.approx(1.0 + pyramid, rel=1e-12)
    # Each pyramid a cell right side out, as element_mesh numbers its corners.
    volume = np.asarray(element_mesh(every, pyramids=caps.pyramids).geometry().cell.volume)
    np.testing.assert_allclose(volume, 0.25 * (0.5 * height) / 3.0, rtol=1e-12)
