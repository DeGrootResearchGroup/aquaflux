"""Zero-thickness sheets that reflect and absorb on both faces.

A baffle or a shelf given as one layer of triangles has the medium on both sides. Declared on its
surface set, it is solved as two faces per facet -- the facet and its back face -- and the checks
here say that this is the same as a solid of vanishing thickness, whose top and bottom are those
two faces.
"""

from __future__ import annotations

import numpy as np
import pytest
from aquaflux.radiation.model import (
    RadiationSettings,
    build_radiation_model,
    fluence_rate,
    radiosity,
)
from aquaflux.radiation.profiles import Isotropic, Lambertian
from aquaflux.radiation.scene import Scene, VolumeReceivers, solve_scene
from aquaflux.radiation.self_occlusion import NoOcclusion, RayCastOcclusion, SilhouetteOcclusion
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import build_transfer

from tests.unit.radiation_references import inward_box, rectangle_triangles

REFLECTANCE = 0.6
#: The sheet's plane and the half-width of each of its four squares, off the box's grid: on it a
#: segment between two wall centroids can run exactly along the sheet's rim, where a sheet and a
#: slab of any thickness answer a ray differently.
HEIGHT, HALF = 0.5137, 0.1213
#: The slab the sheet is checked against: this thick, it differs from the sheet by about as much.
THICKNESS = 1e-5


def _plate(z, *, facing_up=True) -> np.ndarray:
    """Four squares of two triangles at height ``z``, facing up, or down with the same triangles."""
    squares = np.concatenate(
        [
            rectangle_triangles([x, y, z], [HALF, 0.0, 0.0], [0.0, HALF, 0.0])
            for x in (0.3871, 0.3871 + 2 * HALF)
            for y in (0.4013, 0.4013 + 2 * HALF)
        ]
    )
    return squares if facing_up else squares[:, [0, 2, 1]]


def _rim(z_low, z_high) -> np.ndarray:
    """The slab's four thin sides, facing out."""
    low, high = np.array([0.3871 - HALF, 0.4013 - HALF]), np.array([0.3871, 0.4013]) + 3 * HALF
    corners = [low, [high[0], low[1]], high, [low[0], high[1]]]
    sides = []
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        quad = [[*a, z_low], [*b, z_low], [*b, z_high], [*a, z_high]]
        sides += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
    return np.asarray(sides, dtype=float)


WALLS = inward_box(4)
N_WALLS = len(WALLS)
N_SHEET = len(_plate(HEIGHT))


def _sheet(two_sided=("sheet",), reflectance=REFLECTANCE) -> Surfaces:
    """The walls of a closed box, emitting, round one sheet that reflects and does not emit."""
    return Surfaces.from_triangles(
        np.concatenate([WALLS, _plate(HEIGHT)]),
        solid_id=[0] * N_WALLS + [1] * N_SHEET,
        solid_names=("walls", "sheet"),
        emission=np.r_[np.ones(N_WALLS), np.zeros(N_SHEET)],
        diffuse_reflectance=np.r_[np.full(N_WALLS, REFLECTANCE), np.full(N_SHEET, reflectance)],
        two_sided=two_sided,
    )


def _slab() -> Surfaces:
    """The same box round a closed slab :data:`THICKNESS` thick: its top, its bottom, its sides."""
    top = _plate(HEIGHT + THICKNESS / 2)
    bottom = _plate(HEIGHT - THICKNESS / 2, facing_up=False)
    rim = _rim(HEIGHT - THICKNESS / 2, HEIGHT + THICKNESS / 2)
    n_slab = 2 * N_SHEET + len(rim)
    return Surfaces.from_triangles(
        np.concatenate([WALLS, top, bottom, rim]),
        solid_id=[0] * N_WALLS + [1] * n_slab,
        solid_names=("walls", "slab"),
        emission=np.r_[np.ones(N_WALLS), np.zeros(n_slab)],
        diffuse_reflectance=REFLECTANCE,
    )


def _points() -> np.ndarray:
    rng = np.random.default_rng(0)
    points = rng.uniform(0.1, 0.9, (30, 3))
    points[:, 2] = np.where(np.abs(points[:, 2] - HEIGHT) < 0.05, 0.3, points[:, 2])
    return points


# -- the surface set ------------------------------------------------------------------------------


def test_a_sheet_s_back_faces_follow_the_set_each_its_front_s_triangle_turned_over():
    sheet = _sheet()
    faces = sheet.with_back_faces()
    assert faces.n_facets == N_WALLS + 2 * N_SHEET
    fronts = np.arange(N_WALLS, N_WALLS + N_SHEET)
    backs = fronts + N_SHEET
    assert faces.back_of == tuple(fronts)
    np.testing.assert_array_equal(faces.twin[fronts], backs)
    np.testing.assert_array_equal(faces.twin[backs], fronts)
    assert np.all(faces.twin[:N_WALLS] == -1)
    # The same triangle, exactly, the other way up.
    np.testing.assert_array_equal(faces.centroid[backs], faces.centroid[fronts])
    np.testing.assert_array_equal(faces.area[backs], faces.area[fronts])
    np.testing.assert_array_equal(faces.normal[backs], -faces.normal[fronts])
    np.testing.assert_array_equal(
        faces.diffuse_reflectance[backs], faces.diffuse_reflectance[fronts]
    )
    # Once is enough, and a set with no sheet has none.
    assert faces.with_back_faces() is faces
    plain = _sheet(two_sided=())
    assert plain.with_back_faces() is plain


def test_a_sheet_moved_keeps_its_back_faces_on_its_triangles():
    faces = _sheet().with_back_faces()
    moved = faces.with_geometry(np.asarray(faces.vertices) + np.array([0.0, 0.0, 0.01]))
    assert moved.back_of == faces.back_of
    np.testing.assert_allclose(
        moved.centroid, np.asarray(faces.centroid) + np.array([0.0, 0.0, 0.01])
    )
    np.testing.assert_array_equal(moved.normal, faces.normal)


def test_a_point_source_in_a_sheet_s_body_has_no_back_face():
    sheet = Surfaces.from_triangles(
        np.concatenate([_plate(HEIGHT), np.zeros((1, 3, 3))]),
        profiles=(Lambertian(), Isotropic()),
        profile_index=[0] * N_SHEET + [1],
        two_sided=("surface",),
    )
    assert sheet.with_back_faces().back_of == tuple(range(N_SHEET))


@pytest.mark.parametrize("field", ["emission", "power"])
def test_a_sheet_that_emits_is_refused(field):
    sheet = _sheet()
    with pytest.raises(ValueError, match=f"two-sided sheet may not carry {field}"):
        sheet.with_optics(**{field: 1.0}).with_back_faces()


def test_a_misspelt_sheet_is_refused_rather_than_left_one_sided():
    with pytest.raises(ValueError, match=r"two_sided names no body.*\['shete'\]"):
        _sheet(two_sided=("shete",))


# -- the transfer ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy", [RayCastOcclusion(), SilhouetteOcclusion()], ids=["ray", "clip"]
)
def test_a_face_and_its_back_exchange_nothing_and_each_sees_a_whole_hemisphere(strategy):
    """In a closed box every face's row sums to one, its back face's included.

    A quadrature point a rounding off its triangle reads the whole of the coincident back face as
    a hemisphere, which made the row of a face turned down sum to 4/3 and its twin's to one; the
    sign of that rounding decided which. The rows are of the geometry, before the shadows.
    """
    faces = _sheet().with_back_faces()
    transfer = build_transfer(faces, self_occlusion=strategy)
    geometric = np.asarray(transfer.geometric)
    fronts = np.arange(N_WALLS, N_WALLS + N_SHEET)
    np.testing.assert_array_equal(geometric[fronts, fronts + N_SHEET], 0.0)
    np.testing.assert_array_equal(geometric[fronts + N_SHEET, fronts], 0.0)
    np.testing.assert_allclose(geometric[N_WALLS:].sum(axis=1), 1.0, atol=1e-12)


def test_a_ray_ending_on_a_face_is_not_stopped_by_its_back():
    """Every wall below the sheet sees its faces turned down; a ray aimed at one ends on its back
    face's triangle too, and must ignore both, as it ignores the facet it is aimed at."""
    faces = _sheet().with_back_faces()
    hidden = np.asarray(
        build_transfer(faces, self_occlusion=RayCastOcclusion()).visibility.hidden_by_geometry
    )
    below = np.asarray(faces.centroid)[:N_WALLS, 2] < HEIGHT - 0.1
    backs = np.arange(N_WALLS + N_SHEET, N_WALLS + 2 * N_SHEET)
    assert not np.any(hidden[np.ix_(backs, np.flatnonzero(below))])


# -- the solve, against a slab of vanishing thickness --------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "tolerance"),
    [(SilhouetteOcclusion(), 2e-5), (RayCastOcclusion(), 3e-4)],
    ids=["clip", "ray"],
)
def test_a_sheet_is_lit_and_lights_as_a_slab_of_vanishing_thickness(strategy, tolerance):
    """Radiosity of the walls and of each face, and the field, against the slab's.

    The slab's top and bottom are the sheet's two faces, so they agree to the slab's thickness
    under the exact clip. One ray per pair reads the slab's top as blocking a wall that straddles
    the sheet's plane from its bottom, which only the slab's thickness puts there, so the back
    faces are held to the walls' tolerance and not to their own.
    """
    sheet, slab = _sheet(), _slab()
    settings = RadiationSettings(self_occlusion=strategy)
    with_sheet = build_radiation_model(_points(), sheet, settings=settings)
    with_slab = build_radiation_model(_points(), slab, settings=settings)
    on_sheet, _ = radiosity(with_sheet, sheet)
    on_slab, _ = radiosity(with_slab, slab)
    on_sheet, on_slab = np.asarray(on_sheet), np.asarray(on_slab)
    assert on_sheet.shape == (N_WALLS + 2 * N_SHEET,)
    walls, fronts = slice(0, N_WALLS), slice(N_WALLS, N_WALLS + N_SHEET)
    backs, bottoms = slice(N_WALLS + N_SHEET, None), slice(N_WALLS + N_SHEET, N_WALLS + 2 * N_SHEET)
    np.testing.assert_allclose(on_sheet[walls], on_slab[walls], rtol=tolerance)
    np.testing.assert_allclose(on_sheet[fronts], on_slab[fronts], rtol=tolerance)
    np.testing.assert_allclose(on_sheet[backs], on_slab[bottoms], rtol=max(tolerance, 2e-3))
    field_sheet, _ = fluence_rate(with_sheet, sheet)
    field_slab, _ = fluence_rate(with_slab, slab)
    np.testing.assert_allclose(field_sheet, field_slab, rtol=max(tolerance, 3e-4))


def test_an_undeclared_sheet_is_not_the_slab():
    """The defect the declaration exists for. One-sided, the sheet has no back face: the light that
    lands on its back leaves the solve, and seen from below, the clip lets the walls above shine
    straight through it. The field below it is then wrong by percents, where declared it is the
    slab's to a few parts in ten thousand."""
    settings = RadiationSettings(self_occlusion=SilhouetteOcclusion())
    one_sided = _sheet(two_sided=())
    with pytest.warns(UserWarning, match="free edge"):
        model = build_radiation_model(_points(), one_sided, settings=settings)
    field, _ = fluence_rate(model, one_sided)
    slab = _slab()
    reference, _ = fluence_rate(build_radiation_model(_points(), slab, settings=settings), slab)
    below = _points()[:, 2] < HEIGHT
    error = np.abs(np.asarray(field) / np.asarray(reference) - 1.0)[below]
    assert np.all(error > 0.02)


# -- the scene ------------------------------------------------------------------------------------


def _shelf_scene(two_sided, occlusion=None) -> Scene:
    """A lamp on the floor facing up, a reflecting shelf over it facing up, and a point beside the
    lamp below the shelf: the only reflected light that point gets is the shelf's back face's."""
    lamp = rectangle_triangles([0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0])
    shelf = rectangle_triangles([0.0, 0.0, 0.5], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0])
    return Scene(
        lamps=Surfaces.from_triangles(lamp, solid_names=("lamp",), emission=7.0),
        reflectors=Surfaces.from_triangles(
            shelf, solid_names=("shelf",), diffuse_reflectance=REFLECTANCE, two_sided=two_sided
        ),
        volume=VolumeReceivers(np.array([[0.3, 0.0, 0.2]])),
        settings=RadiationSettings(
            receiver_quadrature=1,
            self_occlusion=NoOcclusion() if occlusion is None else occlusion,
        ),
    )


def test_a_lamp_under_a_shelf_lights_the_room_by_the_shelf_s_back_face():
    one_sided = solve_scene(_shelf_scene(()))
    two_sided = solve_scene(_shelf_scene(("shelf",)))
    # The shelf faces up, away from the lamp: one-sided, the light it receives is lost.
    assert float(one_sided.fluence_rate_reflected[0]) == 0.0
    assert float(two_sided.fluence_rate_reflected[0]) > 0.0
    np.testing.assert_array_equal(two_sided.reflector_back_of, [0, 1])
    assert two_sided.radiosity.shape == (4,)
    np.testing.assert_array_equal(two_sided.radiosity[:2], 0.0)
    assert np.all(two_sided.radiosity[2:] > 0.0)
    # Every watt landing on the shelf's back is reflected or absorbed there.
    landed = two_sided.reflector_irradiance[2:]
    area = np.asarray(_shelf_scene(()).reflectors.area)
    assert two_sided.reflector_absorbed_power == pytest.approx(
        (1.0 - REFLECTANCE) * float(landed @ area), rel=1e-12
    )


def test_a_back_face_keeps_its_front_s_body_as_an_integer_label():
    """Labels are gathered, not computed with: a body index that came back a float would no longer
    index the body names."""
    faces = _sheet().with_back_faces()
    assert np.asarray(faces.solid_id).dtype.kind == "i"
    np.testing.assert_array_equal(np.asarray(faces.solid_id)[N_WALLS + N_SHEET :], 1)


def test_the_lamps_light_on_a_back_face_is_not_shadowed_by_its_own_triangle():
    """Nothing stands between the lamp, the shelf and the point, so testing the scene's own
    triangles for shadows changes nothing -- provided a back face's sample points are named as lying
    on their front's triangle, which a ray from the lamp otherwise ends in, and counts as blocked."""
    clear = solve_scene(_shelf_scene(("shelf",)))
    traced = solve_scene(_shelf_scene(("shelf",), RayCastOcclusion()))
    assert float(clear.reflector_irradiance[2]) > 0.0
    np.testing.assert_allclose(traced.reflector_irradiance, clear.reflector_irradiance, rtol=1e-12)
    np.testing.assert_allclose(
        traced.fluence_rate_reflected, clear.fluence_rate_reflected, rtol=1e-12
    )
