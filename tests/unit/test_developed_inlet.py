"""An inlet admitting a fully developed pipe flow (``DevelopedInlet``), and the table it reads, written by
another case's ``RadialProfiles`` output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aquaflux.case import (
    CaseFile,
    DevelopedInlet,
    PatchSite,
    RadialProfiles,
    RunFields,
    case_spec_from_mapping,
)
from aquaflux.io import read_aqmesh, read_profile_table, write_profile_table
from aquaflux.turbulence import SSTModel

from tests.support.case_meshes import grid_mesh, grid_mesh_path

#: A 3D box, 1 x 2 x 2: its left side, x = 0, is the inlet, centred on (0, 1, 1).
CELLS, LENGTHS = (4, 6, 6), (1.0, 2.0, 2.0)
CENTRE = np.array([0.0, 1.0, 1.0])
RADII = np.array([0.0, 0.5, 1.0, 1.5])


def _table(directory: Path, **columns) -> str:
    columns = columns or {
        "U": np.array([3.0, 2.5, 1.5, 0.0]),
        "k": np.array([0.2, 0.3, 0.5, 0.9]),
        "omega": np.array([10.0, 20.0, 40.0, 80.0]),
    }
    write_profile_table(directory / "developed.csv", RADII, columns)
    return "developed.csv"


def _case(directory: Path, physics: str = "Laminar", inlet: str = "left"):
    """The box, its inlet a ``DevelopedInlet``, its opposite side an outlet, the rest walls, built."""
    boundaries = {
        side: {"kind": "Wall"} for side in ("left", "right", "bottom", "top", "back", "front")
    }
    boundaries[inlet] = {"kind": "DevelopedInlet", "profile_table": "developed.csv"}
    boundaries["right" if inlet == "left" else "left"] = {"kind": "Outlet", "pressure": 0.0}
    numerics = {"momentum_advection": {"kind": "FirstOrderUpwind"}}
    if physics == "RANS":
        numerics["turbulence_advection"] = {"kind": "FirstOrderUpwind"}
    spec = case_spec_from_mapping(
        {
            "mesh": grid_mesh(CELLS, LENGTHS),
            "fluid": {"density": 1.0, "kinematic_viscosity": 1.0e-3},
            "physics": {"kind": physics},
            "boundaries": boundaries,
            "numerics": {**numerics, "gradient": {"kind": "CompactGreenGauss"}},
        }
    )
    return CaseFile(spec, directory).check().build()


def _left_faces():
    mesh = read_aqmesh(grid_mesh_path(CELLS, LENGTHS))
    faces = np.asarray(mesh.face_patches.indices("left"))
    return np.asarray(mesh.geometry().face.centroid)[faces]


def _expected(column: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.interp(np.linalg.norm(centroids - CENTRE, axis=1), RADII, column)


def test_the_inlets_faces_take_the_developed_velocity_by_distance_from_its_centre(tmp_path):
    """Along the inward normal (+x on the left side), as fast as the table says at each face's radius.

    A profile laid out from a corner, or along the outward normal, or read at the wrong radius, gives
    other face velocities: the faces of the left side are at radii 0.24 to 1.18 from its centre.
    """
    _table(tmp_path)
    momentum = _case(tmp_path)
    centroids = _left_faces()
    velocity = np.asarray(momentum.boundary.conditions["left"].velocity(centroids))
    expected = np.zeros_like(centroids)
    expected[:, 0] = _expected(np.array([3.0, 2.5, 1.5, 0.0]), centroids)
    np.testing.assert_allclose(velocity, expected, atol=1e-12)


def test_a_rans_inlet_admits_the_developed_k_and_omega(tmp_path):
    _table(tmp_path)
    coupled = _case(tmp_path, physics="RANS")
    centroids = _left_faces()
    turbulence = coupled.turbulence
    for name, column in (("k", [0.2, 0.3, 0.5, 0.9]), ("omega", [10.0, 20.0, 40.0, 80.0])):
        closure = getattr(turbulence, f"{name}_boundary").conditions["left"]
        np.testing.assert_allclose(
            np.asarray(closure.field_fn(centroids)), _expected(np.array(column), centroids)
        )


def test_an_inlet_on_the_far_side_admits_the_flow_the_other_way(tmp_path):
    """On the right side (x = 1) the inward normal is -x, so the velocity is too."""
    _table(tmp_path)
    momentum = _case(tmp_path, inlet="right")
    mesh = read_aqmesh(grid_mesh_path(CELLS, LENGTHS))
    faces = np.asarray(mesh.face_patches.indices("right"))
    centroids = np.asarray(mesh.geometry().face.centroid)[faces]
    velocity = np.asarray(momentum.boundary.conditions["right"].velocity(centroids))
    radius = np.linalg.norm(centroids - np.array([1.0, 1.0, 1.0]), axis=1)
    np.testing.assert_allclose(velocity[:, 0], -np.interp(radius, RADII, [3.0, 2.5, 1.5, 0.0]))
    np.testing.assert_allclose(velocity[:, 1:], 0.0, atol=1e-14)


def test_a_table_without_a_velocity_is_refused_naming_the_patch(tmp_path):
    _table(tmp_path, k=np.ones(4))
    with pytest.raises(
        ValueError, match=r"boundaries.left.profile_table: developed.csv has no U column"
    ):
        _case(tmp_path)


def test_a_rans_case_refuses_a_table_without_the_turbulence(tmp_path):
    _table(tmp_path, U=np.ones(4), k=np.ones(4))
    with pytest.raises(ValueError, match=r"the table has no omega column"):
        _case(tmp_path, physics="RANS")


def test_a_patch_that_is_not_flat_is_refused():
    normals = np.array([[-1.0, 0.0, 0.0], [-np.cos(0.1), np.sin(0.1), 0.0]])
    site = PatchSite("inlet", np.zeros((2, 3)), np.ones(2), normals, Path("."))
    with pytest.raises(ValueError, match=r"boundaries.inlet: a face leans .* not flat"):
        DevelopedInlet("developed.csv").placed(site)


def test_a_developed_inlet_is_not_built_before_it_is_placed():
    with pytest.raises(ValueError, match="place it on the patch first"):
        DevelopedInlet("developed.csv").flow_closure()
    with pytest.raises(ValueError, match="place it on the patch first"):
        DevelopedInlet("developed.csv").turbulence_closures(SSTModel())
    assert not DevelopedInlet("developed.csv").prescribes_pressure()


def test_a_case_names_the_table_as_a_file_it_reads(tmp_path):
    """A missing table is refused when the case is checked, before its mesh is read."""
    with pytest.raises(FileNotFoundError, match=r"developed\.csv"):
        _case(tmp_path)


def test_the_profiles_a_periodic_case_writes_are_the_table_an_inlet_reads(tmp_path):
    """A field that is a function of the radius alone comes back as that function, a vector along the
    axis as its axial component."""
    mesh = read_aqmesh(grid_mesh_path((4, 12, 12), (1.0, 2.0, 2.0)))
    cells = mesh.geometry().cell
    centroid = np.asarray(cells.centroid)
    radius = np.linalg.norm(centroid[:, 1:] - 1.0, axis=1)
    velocity = np.zeros_like(centroid)
    velocity[:, 0] = 2.0 * radius + 1.0
    velocity[:, 1] = 5.0  # across the axis, so not written
    writer = RadialProfiles(centre=(0.0, 1.0, 1.0), axis=(-3.0, 0.0, 0.0), bands=6)
    path = writer.write(tmp_path, tmp_path, mesh, RunFields({"U": velocity, "k": radius**2}))
    table = read_profile_table(path)
    assert list(table.columns) == ["U", "k"]
    # Along -x, the axial velocity is minus the x component; linear in r, its band mean is exact.
    np.testing.assert_allclose(table.columns["U"], -(2.0 * table.radii + 1.0), atol=1e-12)
    assert np.all(table.columns["k"] >= table.radii**2 - 1e-12)  # the mean of r^2 over a band


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"centre": (0.0, 0.0), "axis": (1.0, 0.0, 0.0)}, "of two or three components each"),
        ({"centre": (0.0, 0.0, 0.0), "axis": (0.0, 0.0, 0.0)}, "axis must not be zero"),
        ({"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "bands": 0}, "bands must be >= 1"),
        ({"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "file": "a/b.csv"}, "a file name"),
    ],
)
def test_a_profile_writer_out_of_range_is_refused(settings, match):
    with pytest.raises(ValueError, match=match):
        RadialProfiles(**settings)
