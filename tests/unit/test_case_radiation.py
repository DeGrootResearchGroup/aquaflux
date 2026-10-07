"""A radiation case described in one file: what it reads, what it refuses, and what it builds.

The build is compared, array for array, with the scene assembled by hand from the library's own calls
on the same mesh -- the lamps' triangles and exitance, the reflecting surfaces, the bodies in the way,
the points the light is gathered at -- so a setting the case drops, or a patch built from the wrong
faces, shows as a difference rather than as a plausible field.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path

import jax
import numpy as np
import pytest
from aquaflux.case import (
    CaseFile,
    IesProfile,
    LambertianProfile,
    Lamp,
    Radiation,
    RadiationSolve,
    Wall,
    case_spec_from_mapping,
    case_spec_to_mapping,
    prepare_run,
    read_case,
    solver_for,
    write_case,
)
from aquaflux.case.paths import named_paths, with_paths
from aquaflux.case.radiation import PatchSurface, _facing
from aquaflux.mesh import Mesh, patch_triangles, structured_grid_3d
from aquaflux.radiation import (
    Lambertian,
    NoOcclusion,
    RadiationSettings,
    Surfaces,
    TriangleBody,
    UniformAbsorption,
    absorption_from_uvt,
    coarsen_to_size,
    lamp_exitance,
    read_ies,
)

REPO = Path(__file__).resolve().parents[2]
IES = REPO / "validation" / "ray_effects_room" / "ushio_b1.ies"
CELLS, LENGTHS = (3, 4, 5), (1.0, 1.5, 2.0)


def _sections(**changes) -> dict:
    """A radiation case on a generated box: a lamp on top, two reflecting walls, a body in front."""
    sections = {
        "mesh": {"kind": "StructuredGrid", "cells": list(CELLS), "lengths": list(LENGTHS)},
        "physics": {
            "kind": "Radiation",
            "medium": {"kind": "UniformMedium", "transmittance": 80.0},
            "occluders": [{"kind": "PatchBody", "patches": ["front"], "sheet": True}],
            "lamp_samples": 2,
            "settings": {"kind": "RadiationSettings", "self_occlusion": {"kind": "NoOcclusion"}},
        },
        "boundaries": {
            "top": {
                "kind": "Lamp",
                "power": 2.0,
                "profile": {"kind": "LambertianProfile"},
                "reflectance": 0.25,
            },
            "bottom": {
                "kind": "Wall",
                "reflectance": 0.3,
                "geometry": {
                    "kind": "MeshPatch",
                    "coarsen": {"kind": "Coarsen", "max_edge": 0.8, "chord": 1e-3},
                },
            },
            "left": {"kind": "Wall", "reflectance": 0.6},
            "right": {"kind": "Wall"},
            "back": {"kind": "Wall"},
            "front": {"kind": "Wall"},
        },
    }
    for key, value in changes.items():
        sections[key] = value
    return sections


def _mesh():
    return structured_grid_3d(*CELLS, *LENGTHS, named_boundaries=True)


def _assert_same_tree(built, expected) -> None:
    """One pytree, static fields included, every array leaf equal to the last bit."""
    built_leaves, built_tree = jax.tree.flatten(built)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    assert built_tree == expected_tree
    for a, b in zip(built_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_case_builds_the_scene_the_library_calls_build_by_hand(tmp_path) -> None:
    checked = CaseFile(case_spec_from_mapping(_sections()), tmp_path).check()
    scene = checked.build()

    mesh = _mesh()
    geometry = mesh.geometry()

    def triangles(name, **options):
        return patch_triangles(mesh, geometry, [name], **options).vertices

    lamps = Surfaces.from_triangles(
        triangles("top"),
        solid_id=np.zeros(len(triangles("top")), dtype=int),
        solid_names=("top",),
        diffuse_reflectance=np.full(len(triangles("top")), 0.25),
        profiles=(Lambertian(),),
        profile_index=np.zeros(len(triangles("top")), dtype=int),
    )
    lamps = lamps.with_optics(emission=lamp_exitance(lamps, {"top": 2.0}))
    _assert_same_tree(scene.lamps, lamps)

    bottom = coarsen_to_size(triangles("bottom"), max_edge=0.8, chord=1e-3).vertices
    left = triangles("left")
    reflectors = Surfaces.from_triangles(
        np.concatenate([bottom, left]),
        solid_id=np.repeat([0, 1], [len(bottom), len(left)]),
        solid_names=("bottom", "left"),
        diffuse_reflectance=np.repeat([0.3, 0.6], [len(bottom), len(left)]),
    )
    _assert_same_tree(scene.reflectors, reflectors)
    # The coarsening did coarsen, so the setting is not ignored by a case that happens to match.
    assert len(bottom) < len(triangles("bottom"))

    (body,) = scene.occluders
    expected_body = TriangleBody.build(triangles("front", allow_folded=True), sheet=True)
    np.testing.assert_array_equal(body.grid.vertices, expected_body.grid.vertices)
    assert body.inward_pieces == expected_body.inward_pieces == 0

    np.testing.assert_array_equal(scene.volume.points, np.asarray(geometry.cell.centroid))
    np.testing.assert_array_equal(scene.volume.volumes, np.asarray(geometry.cell.volume))
    # Every wall but the body's is gathered on, each facing into the domain.
    assert list(scene.surfaces) == ["bottom", "left", "right", "back"]
    for name, receivers in scene.surfaces.items():
        faces = np.asarray(mesh.face_patches.indices(name))
        np.testing.assert_array_equal(receivers.points, np.asarray(geometry.face.centroid)[faces])
        np.testing.assert_array_equal(receivers.normals, -np.asarray(geometry.face.normal)[faces])
        np.testing.assert_array_equal(receivers.areas, np.asarray(geometry.face.area)[faces])
    assert scene.surfaces["bottom"].reflector == "bottom"
    assert scene.surfaces["left"].reflectance == 0.6
    assert scene.surfaces["right"].reflector is None and scene.surfaces["right"].reflectance == 0.0
    assert float(scene.absorption.coefficient) == absorption_from_uvt(80.0)
    assert isinstance(scene.absorption, UniformAbsorption)
    assert scene.lamp_samples == 2
    assert scene.settings == RadiationSettings(self_occlusion=NoOcclusion())


def test_unset_numerics_reach_the_librarys_own_defaults(tmp_path) -> None:
    physics = {"kind": "Radiation"}
    sections = _sections(physics=physics)
    scene = CaseFile(case_spec_from_mapping(sections), tmp_path).check().build()
    assert scene.absorption is None and scene.occluders == ()
    assert scene.lamp_samples == type(scene).__dataclass_fields__["lamp_samples"].default
    assert scene.settings == RadiationSettings()
    # With no body, every wall is gathered on.
    assert list(scene.surfaces) == ["bottom", "left", "right", "back", "front"]


def test_a_lamp_with_a_measured_table_emits_the_tables_own_power(tmp_path) -> None:
    (tmp_path / "lamp.ies").write_bytes(IES.read_bytes())
    sections = _sections()
    sections["boundaries"]["top"] = {
        "kind": "Lamp",
        "profile": {"kind": "IesProfile", "file": "lamp.ies", "up": [1.0, 0.0, 0.0]},
    }
    scene = CaseFile(case_spec_from_mapping(sections), tmp_path).check().build()
    photometry = read_ies(IES)
    # The file states mW/sr, so its integrated flux is milliwatts.
    assert photometry.keywords["_INTENSITYUNITS"] == "mW/sr"
    power = float(np.sum(np.asarray(scene.lamps.emission) * np.asarray(scene.lamps.area)))
    assert power == pytest.approx(photometry.flux * 1e-3, rel=1e-13)
    profile = scene.lamps.profiles[0]
    expected = photometry.profile(up=(1.0, 0.0, 0.0))
    np.testing.assert_array_equal(np.asarray(profile.table), np.asarray(expected.table))
    np.testing.assert_array_equal(np.asarray(profile.up), np.asarray(expected.up))


def test_a_photometry_file_in_candela_carries_no_power_so_the_lamp_must_state_one(tmp_path) -> None:
    text = IES.read_text().replace("[_INTENSITYUNITS] mW/sr", "")
    (tmp_path / "candela.ies").write_text(text)
    sections = _sections()
    sections["boundaries"]["top"] = {
        "kind": "Lamp",
        "profile": {"kind": "IesProfile", "file": "candela.ies", "up": [1.0, 0.0, 0.0]},
    }
    checked = CaseFile(case_spec_from_mapping(sections), tmp_path).check()
    with pytest.raises(ValueError, match=r"boundaries.top: the lamp states no power"):
        checked.build()


def _write_stl(path: Path, solids: dict[str, np.ndarray]) -> None:
    lines = []
    for name, triangles in solids.items():
        lines.append(f"solid {name}")
        for tri in triangles:
            lines += ["  facet normal 0 0 0", "    outer loop"]
            lines += [f"      vertex {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in tri]
            lines += ["    endloop", "  endfacet"]
        lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n")


def _plate(y: float, flip: bool) -> np.ndarray:
    """The bottom wall (y = 0) as two triangles, wound up (into the box) unless ``flip``."""
    corners = np.array([[0, y, 0], [0, y, 2.0], [1.0, y, 2.0], [1.0, y, 0]], float)
    pair = np.array([corners[[0, 1, 2]], corners[[0, 2, 3]]])
    return pair[:, ::-1] if flip else pair


@pytest.mark.parametrize("flip", [False, True])
def test_a_surface_read_from_a_file_is_turned_to_face_into_the_domain(tmp_path, flip) -> None:
    _write_stl(tmp_path / "room.stl", {"floor": _plate(0.0, flip), "other": _plate(1.5, False)})
    sections = _sections()
    sections["boundaries"]["bottom"]["geometry"] = {
        "kind": "StlSurface",
        "file": "room.stl",
        "solids": ["floor"],
    }
    scene = CaseFile(case_spec_from_mapping(sections), tmp_path).check().build()
    bottom = np.asarray(scene.reflectors.vertices)[np.asarray(scene.reflectors.solid_id) == 0]
    assert len(bottom) == 2
    np.testing.assert_allclose(np.asarray(scene.reflectors.normal)[:2], [[0, 1, 0]] * 2)


def test_a_surface_whose_triangles_disagree_about_their_side_is_refused() -> None:
    mesh = _mesh()
    geometry = mesh.geometry()
    faces = np.asarray(mesh.face_patches.indices("bottom"))
    patch = PatchSurface(
        name="bottom",
        triangles=patch_triangles(mesh, geometry, ["bottom"]).vertices,
        centres=np.asarray(geometry.face.centroid)[faces],
        inward=-np.asarray(geometry.face.normal)[faces],
    )
    mixed = np.concatenate([_plate(0.0, False)[:1], _plate(0.0, True)[1:]])
    with pytest.raises(ValueError, match=r"1 of its triangles face into the domain and 1 face out"):
        _facing(mixed, patch, "StlSurface")
    # A surface standing square to the patch everywhere says nothing about which way it faces.
    edge_on = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0]]], float)
    with pytest.raises(ValueError, match="cannot be told which way it faces"):
        _facing(edge_on, patch, "StlSurface")
    np.testing.assert_array_equal(_facing(_plate(0.0, True), patch, "S"), _plate(0.0, False))


def test_the_files_a_case_names_are_found_checked_and_rebased(tmp_path) -> None:
    (tmp_path / "lamp.ies").write_bytes(IES.read_bytes())
    sections = _sections()
    sections["boundaries"]["top"] = {
        "kind": "Lamp",
        "profile": {"kind": "IesProfile", "file": "lamp.ies", "up": [1.0, 0.0, 0.0]},
    }
    sections["physics"]["occluders"].append({"kind": "StlBody", "file": "missing.stl"})
    spec = case_spec_from_mapping(sections)
    assert dict(named_paths(spec)) == {
        "physics.occluders[1].file": "missing.stl",
        "boundaries.top.profile.file": "lamp.ies",
    }
    with pytest.raises(FileNotFoundError, match=r"physics.occluders\[1\].file: missing.stl"):
        CaseFile(spec, tmp_path).check()
    moved = with_paths(spec, lambda path: f"../{path}")
    assert moved.boundaries["top"].profile.file == "../lamp.ies"
    assert moved.physics.occluders[1].file == "../missing.stl"
    # What names no file comes back as the very object it was.
    assert moved.boundaries["left"] is spec.boundaries["left"]


def test_a_radiation_case_file_round_trips(tmp_path) -> None:
    spec = case_spec_from_mapping(_sections())
    write_case(spec, tmp_path / "case.yaml")
    assert read_case(tmp_path / "case.yaml").spec == spec
    assert "fluid" not in case_spec_to_mapping(spec)
    assert solver_for(spec) == RadiationSolve()


def _with(sections: dict, path: str, value) -> dict:
    """``sections`` with the entry at a dotted ``path`` replaced (or removed, for ``None``)."""
    out = copy.deepcopy(sections)
    *head, last = path.split(".")
    target = out
    for key in head:
        target = target[key]
    if value is None:
        del target[last]
    else:
        target[last] = value
    return out


FLUID = {"density": 1.2, "kinematic_viscosity": 1.5e-5}


@pytest.mark.parametrize(
    ("sections", "match"),
    [
        (
            _with(_sections(), "boundaries.right", {"kind": "Inlet", "velocity": [0.0, 1.0, 0.0]}),
            r"boundaries.right.velocity: a radiation case has no flow",
        ),
        (
            _with(_sections(), "boundaries.right", {"kind": "Wall", "velocity": [1.0, 0.0, 0.0]}),
            r"boundaries.right.velocity: a radiation case has no flow",
        ),
        (
            _with(_sections(), "boundaries.right", {"kind": "Wall", "k": "zero"}),
            r"boundaries.right.k: a radiation case has no turbulence closure",
        ),
        (
            _with(_sections(), "boundaries.top", {"kind": "Wall"}),
            "no patch is a Lamp",
        ),
        (_sections(fluid=FLUID), r"^fluid: a radiation case has no flow"),
        (
            _sections(numerics={"momentum_advection": {"kind": "FirstOrderUpwind"}}),
            r"^numerics: a radiation case has no flow",
        ),
        (
            _with(_sections(), "boundaries.top.power", None),
            r"Lamp.power is unset, and a LambertianProfile states no power",
        ),
        (
            _with(
                _sections(), "boundaries.right", {"kind": "Wall", "geometry": {"kind": "MeshPatch"}}
            ),
            "this wall reflects nothing, so nothing would read it",
        ),
        (
            _with(
                _sections(),
                "physics",
                {
                    **_sections()["physics"],
                    "lamp_refinement": 0.25,
                    "receivers": {"kind": "Receivers", "cells": False, "patches": []},
                },
            ),
            "lamp_refinement refines the lamps against the points the light is gathered at",
        ),
        (
            _with(_sections(), "solver", {"kind": "FlowMarch"}),
            r"FlowMarch solves a Laminar case, but the physics is Radiation; use RadiationSolve",
        ),
    ],
    ids=[
        "an-inlet",
        "a-moving-wall",
        "a-wall-k",
        "no-lamp",
        "a-fluid",
        "numerics",
        "a-lambertian-lamp-with-no-power",
        "geometry-on-a-black-wall",
        "refinement-against-nothing",
        "a-flow-solver",
    ],
)
def test_a_radiation_case_refuses_what_it_would_not_read(sections, match) -> None:
    with pytest.raises(ValueError, match=match):
        case_spec_from_mapping(sections)


def _laminar(**boundary) -> dict:
    return {
        "mesh": {"kind": "StructuredGrid", "cells": [2, 2], "lengths": [1.0, 1.0]},
        "fluid": FLUID,
        "physics": {"kind": "Laminar"},
        "boundaries": {
            "left": {"kind": "Inlet", "velocity": [1.0, 0.0]},
            "right": {"kind": "Outlet", "pressure": 0.0},
            "bottom": {"kind": "Wall"},
            "top": {"kind": "Wall"},
            **boundary,
        },
        "numerics": {"momentum_advection": {"kind": "FirstOrderUpwind"}},
    }


@pytest.mark.parametrize(
    ("sections", "match"),
    [
        (
            _laminar(top={"kind": "Wall", "reflectance": 0.5}),
            r"boundaries.top.reflectance: a flow case gathers no light",
        ),
        (
            _laminar(top={"kind": "Lamp", "power": 1.0, "profile": {"kind": "LambertianProfile"}}),
            r"boundaries.top.profile, boundaries.top.power: a flow case gathers no light",
        ),
        (
            {**_laminar(), "solver": {"kind": "RadiationSolve"}},
            r"RadiationSolve solves a Radiation case, but the physics is Laminar",
        ),
    ],
    ids=["a-reflectance", "a-lamp", "the-radiation-solve"],
)
def test_a_flow_case_refuses_light(sections, match) -> None:
    with pytest.raises(ValueError, match=match):
        case_spec_from_mapping(sections)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            ("physics.occluders", [{"kind": "PatchBody", "patches": ["roof"]}]),
            r"physics.occluders\[0\]: 'roof' is not a boundary patch",
        ),
        (
            ("physics.receivers", {"kind": "Receivers", "patches": ["top"]}),
            r"'top' is not a wall of the case",
        ),
        (
            ("physics.receivers", {"kind": "Receivers", "patches": ["front"]}),
            r"'front' is part of an occluding body",
        ),
    ],
    ids=["an-unknown-occluder-patch", "gathering-on-a-lamp", "gathering-on-a-body"],
)
def test_a_radiation_case_that_does_not_fit_its_mesh_is_refused(tmp_path, change, match) -> None:
    path, value = change
    spec = case_spec_from_mapping(_with(_sections(), path, value))
    with pytest.raises(ValueError, match=match):
        CaseFile(spec, tmp_path).check()


def test_a_lamp_on_a_two_dimensional_mesh_is_refused(tmp_path) -> None:
    sections = {
        "mesh": {"kind": "StructuredGrid", "cells": [2, 2], "lengths": [1.0, 1.0]},
        "physics": {"kind": "Radiation"},
        "boundaries": {
            "top": {"kind": "Lamp", "power": 1.0, "profile": {"kind": "LambertianProfile"}},
            "bottom": {"kind": "Wall"},
            "left": {"kind": "Wall"},
            "right": {"kind": "Wall"},
        },
    }
    with pytest.raises(ValueError, match="a lamp lights a three-dimensional domain"):
        CaseFile(case_spec_from_mapping(sections), tmp_path).check()


def test_a_lamp_is_a_wall_to_the_flow() -> None:
    lamp = Lamp(profile=LambertianProfile(), power=1.0)
    assert lamp.flow_closure() == Wall().flow_closure()
    assert lamp.settings_in("radiation") == ("profile", "power")
    assert Wall(reflectance=0.2).settings_in("radiation") == ("reflectance",)
    assert Lamp(profile=LambertianProfile(), power=1.0, reflectance=0.1).settings_in(
        "radiation"
    ) == ("profile", "power", "reflectance")
    with pytest.raises(ValueError, match=r"Lamp.reflectance must lie in \[0, 1\], got 1.5"):
        Lamp(profile=LambertianProfile(), power=1.0, reflectance=1.5)
    assert IesProfile(file="x.ies", up=(0.0, 0.0, 1.0)).can_state_power
    assert isinstance(Radiation().receivers.cells, bool)


def test_a_run_writes_the_fields_the_patches_and_where_the_power_went(tmp_path) -> None:
    sections = _sections()
    sections["outputs"] = {"fields": [{"kind": "Vtk"}, {"kind": "PatchVtk"}]}
    write_case(case_spec_from_mapping(sections), tmp_path / "case.yaml")
    record = prepare_run(tmp_path / "case.yaml").run(terminal=io.StringIO())
    results = tmp_path / "results"
    assert record.converged
    assert {p.name for p in record.written} >= {
        "fields.vtu",
        "patches.vtm",
        "run.yaml",
        "case.yaml",
    }
    assert sorted(p.name for p in (results / "patches").iterdir()) == [
        f"{name}.vtp" for name in sorted(("bottom", "top", "left", "right", "back", "front"))
    ]
    out = record.results
    assert out["lamp_power"] == pytest.approx(2.0, rel=1e-12)
    assert list(out["patches"]) == ["bottom", "left", "right", "back"]
    absorbed = sum(entry["absorbed_power"] for entry in out["patches"].values())
    # The lamp reflects a quarter of what lands on it and keeps the rest, which is on the books too.
    assert out["lamp_absorbed_power"] > 0.0
    assert out["unaccounted_power"] == pytest.approx(
        out["lamp_power"] - out["medium_absorbed_power"] - out["lamp_absorbed_power"] - absorbed,
        rel=1e-12,
    )
    # Every wall absorbs what arrives on it, less what it reflects.
    assert out["patches"]["left"]["absorbed_power"] == pytest.approx(
        0.4 * out["patches"]["left"]["incident_power"], rel=1e-12
    )
    assert out["medium_absorbed_power"] > 0.0 and out["radiosity_cycles"] >= 1
    assert "results" in (results / "run.yaml").read_text()


def test_a_surface_read_from_a_file_cannot_be_given_to_a_group_of_patches() -> None:
    mesh = _mesh()
    names = [n for n in mesh.face_patches.names if n not in ("interior", "boundary")]
    grouped = Mesh.from_csr(
        mesh.node_coords,
        np.asarray(mesh.face_nodes.offsets),
        np.asarray(mesh.face_nodes.face_node_indices),
        mesh.face_cells.owner,
        mesh.face_cells.neighbour,
        mesh.face_cells.n_cells,
        face_patches={n: np.asarray(mesh.face_patches.indices(n)) for n in names},
        patch_groups={"sides": ("left", "right")},
    )
    sections = _sections()
    del sections["boundaries"]["left"], sections["boundaries"]["right"]
    surface = {"kind": "StlSurface", "file": "sides.stl"}
    sections["boundaries"]["sides"] = {"kind": "Wall", "reflectance": 0.4, "geometry": surface}
    spec = case_spec_from_mapping(sections)
    (problem,) = spec.physics.mesh_misfits(spec, grouped)
    assert problem.startswith("boundaries.sides.geometry: 'sides' is a group of 2 patches")
    # The mesh's own patches are each their own surface, so a group may share that source.
    sections["boundaries"]["sides"]["geometry"] = {"kind": "MeshPatch"}
    spec = case_spec_from_mapping(sections)
    assert spec.physics.mesh_misfits(spec, grouped) == []
