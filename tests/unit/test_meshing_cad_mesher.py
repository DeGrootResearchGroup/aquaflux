"""The CAD mesher's own work, with the CAD model and Netgen replaced by a unit cube's.

The model is a unit cube of six faces (numbered ``2 axis + side``: ``x-, x+, y-, y+, z-, z+``)
whose twelve edges fold by an angle the test chooses; Netgen hands back Kuhn tetrahedra of the cube,
their boundary triangles tagged by the face each lies on. So what is checked is the mesher's part:
patches chosen by face, the CAD's sharp edges reaching the dual, and the refusals -- each against
the cube's areas and volume.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from aquaflux.io.cad import CadRegion, CadTopology, Projection
from aquaflux.mesh import closed_cell_residual
from aquaflux.meshing import (
    CadMesher,
    FacesNear,
    FacesOn,
    SurfaceTriangulation,
    TetrahedralMesh,
)

from tests.support.meshes import tetrahedral_grid_3d

REGION = CadRegion(("cube",))


def _wall(points: np.ndarray) -> np.ndarray:
    """The face of the cube nearest each point."""
    distance = np.concatenate([points, 1.0 - points], axis=1)  # x-, y-, z-, x+, y+, z+
    nearest = np.argmin(distance, axis=1)
    return 2 * (nearest % 3) + nearest // 3


class _Cube:
    """The unit cube as a CAD model: its topology, a projection, and a named solid ``floor``
    standing in for whatever the face ``z-`` is shared with."""

    def __init__(self, fold: float) -> None:
        corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        edges, faces = [], []
        for a in range(8):
            for b in range(a + 1, 8):
                differ = np.flatnonzero(corners[a] != corners[b])
                if differ.size == 1:
                    fixed = [axis for axis in range(3) if axis != differ[0]]
                    edges.append((a, b))
                    faces.append([2 * axis + int(corners[a][axis]) for axis in fixed])
        self._topology = CadTopology(
            faces=tuple(f"face {i}" for i in range(6)),
            face_area=np.ones(6),
            vertices=corners,
            edge_vertices=np.array(edges),
            edge_length=np.ones(len(edges)),
            edge_face_offsets=np.arange(len(edges) + 1) * 2,
            edge_face_indices=np.array(faces).reshape(-1),
            edge_angle=np.full(len(edges), fold),
            seam=np.zeros(len(edges), dtype=bool),
        )

    def topology(self, region):
        assert region == REGION
        return self._topology

    def write_brep(self, region, path):
        return path

    def project(self, region, points, faces=None):
        points = np.asarray(points, dtype=float)
        if region == CadRegion(("floor",)):
            distance = np.abs(points[:, 2])
            return Projection(points, np.zeros(len(points), int), np.zeros_like(points), distance)
        face = _wall(points) if faces is None else np.asarray(faces)
        return Projection(points, face, np.zeros_like(points), np.zeros(len(points)))


class _Netgen:
    """Kuhn tetrahedra of the cube, as a triangulation and a fill; counts its calls."""

    def __init__(self) -> None:
        mesh = tetrahedral_grid_3d(2)
        self.points = np.asarray(mesh.node_coords)
        offsets = np.asarray(mesh.face_nodes.offsets)
        nodes = np.asarray(mesh.face_nodes.face_node_indices).reshape(-1, 3)
        owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
        assert np.all(np.diff(offsets) == 3)
        corners = [set() for _ in range(mesh.n_cells)]
        for face, ring in enumerate(nodes):
            for cell in (owner[face], neighbour[face]):
                if cell >= 0:
                    corners[cell].update(ring.tolist())
        self.tetrahedra = np.array([sorted(c) for c in corners])
        self.triangles = nodes[neighbour < 0]
        self.calls = 0

    def triangulate(self, shape, **options):
        self.calls += 1
        face = _wall(self.points[self.triangles].mean(axis=1))
        return SurfaceTriangulation(self.points, self.triangles, face)

    def fill(self, points, triangles, **options):
        self.calls += 1
        return TetrahedralMesh(self.points, self.tetrahedra)


def _areas(mesh) -> dict[str, float]:
    area = np.asarray(mesh.geometry().face.area)
    return {
        name: float(area[np.asarray(mesh.face_patches.indices(name))].sum())
        for name in mesh.face_patches.names
        if name not in ("interior", "boundary")
    }


def test_the_faces_each_selection_chooses_make_its_patch_and_the_rest_make_others():
    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=_Netgen())
    mesh = mesher.mesh(
        _Cube(fold=math.pi / 2),
        REGION,
        {"lid": FacesNear(((0.5, 0.5, 1.0),)), "floor": FacesOn(("floor",))},
        others="sides",
    )
    assert _areas(mesh) == pytest.approx({"lid": 1.0, "floor": 1.0, "sides": 4.0}, rel=1e-12)
    assert float(np.asarray(mesh.geometry().cell.volume).sum()) == pytest.approx(1.0, rel=1e-12)
    assert float(np.max(np.asarray(closed_cell_residual(mesh)))) < 1e-14
    # One cell per point of the tetrahedra: their polyhedral dual, not the tetrahedra themselves.
    assert mesh.n_cells == 27


def test_the_cads_sharp_edges_are_kept_whatever_the_triangles_fold_by():
    # One patch over the whole cube, and a feature angle above the triangles' 90 degree folds:
    # only the CAD's say-so keeps the cube's edges, and with them its corners and its volume.
    def volume(fold):
        mesher = CadMesher(max_size=0.5, feature_angle=math.radians(100), netgen=_Netgen())
        mesh = mesher.mesh(_Cube(fold=fold), REGION)
        return float(np.asarray(mesh.geometry().cell.volume).sum())

    assert volume(math.radians(120)) == pytest.approx(1.0, rel=1e-12)
    assert volume(math.radians(90)) < 1.0 - 1e-3


def test_two_selections_choosing_one_face_are_refused():
    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=_Netgen())
    with pytest.raises(ValueError, match="'top' selects a face another patch has already"):
        mesher.mesh(
            _Cube(fold=math.pi / 2),
            REGION,
            {"lid": FacesNear(((0.5, 0.5, 1.0),)), "top": FacesNear(((0.2, 0.3, 1.0),))},
        )


def test_a_solid_no_face_lies_on_is_refused():
    # The floor solid sits at z = 0; moved off the cube, no face of the cube lies on it.
    class _Away(_Cube):
        def project(self, region, points, faces=None):
            landed = super().project(region, points, faces)
            if region == CadRegion(("floor",)):
                return Projection(landed.points, landed.face, landed.normal, landed.distance + 0.1)
            return landed

    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=_Netgen())
    with pytest.raises(
        ValueError, match=r"no face of the region lies on the surface of \['floor'\]"
    ):
        mesher.mesh(_Away(fold=math.pi / 2), REGION, {"floor": FacesOn(("floor",))})


@pytest.mark.parametrize(
    ("patches", "others", "message"),
    [
        ({"boundary": FacesNear(((0.5, 0.5, 1.0),))}, "walls", r"\['boundary'\] are reserved"),
        ({}, "interior", r"\['interior'\] are reserved"),
        ({"walls": FacesNear(((0.5, 0.5, 1.0),))}, "walls", "'walls' names both a selection"),
    ],
)
def test_a_patch_name_that_cannot_be_used_is_refused_before_meshing(patches, others, message):
    netgen = _Netgen()
    mesher = CadMesher(max_size=0.5, feature_angle=math.radians(45), netgen=netgen)
    with pytest.raises(ValueError, match=message):
        mesher.mesh(_Cube(fold=math.pi / 2), REGION, patches, others=others)
    assert netgen.calls == 0


@pytest.mark.parametrize("name", ["max_size", "feature_angle", "min_size", "grading"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_a_size_angle_or_grading_out_of_range_is_refused(name, value):
    settings = {"max_size": 0.5, "feature_angle": 0.5, name: value}
    with pytest.raises(ValueError, match=f"CadMesher.{name} must be a positive, finite number"):
        CadMesher(**settings)


def test_facesnear_takes_points_of_three_coordinates():
    with pytest.raises(ValueError, match="points of three coordinates"):
        FacesNear(((0.5, 0.5),))
