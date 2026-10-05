"""Specular reflection through the assembled model: the transfer, the solve and the volume field."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.absorption import UniformAbsorption, VoxelAbsorption
from aquaflux.radiation.model import (
    RadiationSettings,
    build_radiation_model,
    fluence_rate,
    radiosity,
    surface_irradiance,
)
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.self_occlusion import NoOcclusion, RayCastOcclusion
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import build_transfer, reciprocity_residual
from aquaflux.solids import Sphere

from tests.unit.radiation_references import cylinder_triangles, inward_box, rectangle_triangles

UNSHADOWED = RadiationSettings(self_occlusion=NoOcclusion())


def _halves():
    """The half ``x <= 0.5`` of a closed unit box, and its mirror image across ``x = 0.5``.

    Triangle for triangle, so the whole box made of the two is mirror-symmetric and not merely
    two translates.
    """
    walls = inward_box(2)
    left = walls[np.all(walls[:, :, 0] <= 0.5, axis=1)]
    right = (left * np.array([-1.0, 1.0, 1.0]) + np.array([1.0, 0.0, 0.0]))[:, ::-1]
    return left, right


def _symmetric_scene(emission, diffuse, *, profiles=None, profile_index=0, lamp=None):
    """A whole box, and its left half closed by a perfect mirror across the plane of symmetry.

    The whole box carries the left half's optics on both halves; the half box carries them on
    its walls and a mirror of specular reflectance one, which emits and diffuses nothing. With
    ``lamp``, a point source sits at that position in the half box and at it and its reflection
    in the whole.
    """
    left, right = _halves()
    n = len(left)
    mirror = rectangle_triangles([0.5, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0])
    profiles = (Lambertian(),) if profiles is None else profiles
    index = np.broadcast_to(profile_index, (n,))
    lamps = [] if lamp is None else [np.asarray(lamp, dtype=float)]
    reflected = [] if lamp is None else [lamps[0] * np.array([-1.0, 1.0, 1.0]) + [1.0, 0.0, 0.0]]
    point = [np.full((1, 3, 3), position) for position in lamps]
    point_whole = [np.full((1, 3, 3), position) for position in lamps + reflected]
    catalogue = (*profiles, Isotropic())
    point_kind = len(profiles)

    half = Surfaces.from_triangles(
        np.concatenate([left, mirror, *point]),
        solid_id=[0] * n + [1, 1] + [0] * len(point),
        solid_names=("walls", "mirror"),
        emission=np.concatenate([emission, [0.0, 0.0], np.zeros(len(point))]),
        power=np.concatenate([np.zeros(n + 2), np.full(len(point), 8.0)]),
        diffuse_reflectance=np.concatenate([diffuse, [0.0, 0.0], np.zeros(len(point))]),
        specular_reflectance=np.concatenate([np.zeros(n), [1.0, 1.0], np.zeros(len(point))]),
        profiles=catalogue,
        profile_index=np.concatenate([index, [0, 0], np.full(len(point), point_kind)]),
    )
    whole = Surfaces.from_triangles(
        np.concatenate([left, right, *point_whole]),
        emission=np.concatenate([emission, emission, np.zeros(len(point_whole))]),
        power=np.concatenate([np.zeros(2 * n), np.full(len(point_whole), 8.0)]),
        diffuse_reflectance=np.concatenate([diffuse, diffuse, np.zeros(len(point_whole))]),
        profiles=catalogue,
        profile_index=np.concatenate([index, index, np.full(len(point_whole), point_kind)]),
    )
    return half, whole, n


def _receivers(count=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([0.05, 0.05, 0.05], [0.45, 0.95, 0.95], size=(count, 3))


# ---------------------------------------------------------------------------------------
# The symmetry plane: the release gate
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["lambertian", "absorbing", "cosine power", "point lamp"],
)
def test_a_perfect_mirror_on_a_plane_of_symmetry_reproduces_the_whole_scene(case):
    """Cut a symmetric scene in half and close the cut with a mirror of reflectance one: every
    facet of the half sees, in the mirror, exactly what the whole scene's other half shows it,
    and one bounce is all a single plane can give. So the radiosity, the irradiance on the facets
    and the field in the volume all equal the whole scene's, interreflection included -- the
    specular transfer, the point sources' arrivals through the mirror and the mirrored volume
    gather each have to be right for that to hold."""
    rng = np.random.default_rng(1)
    n_left = len(_halves()[0])
    emission = rng.uniform(0.0, 3.0, n_left)
    diffuse = rng.uniform(0.2, 0.8, n_left)
    options, absorption = {}, None
    if case == "absorbing":
        absorption = UniformAbsorption(1.3)
    if case == "cosine power":
        options = {
            "profiles": (Lambertian(), CosinePower(3.0)),
            "profile_index": rng.integers(0, 2, n_left),
        }
    if case == "point lamp":
        emission = np.zeros(n_left)
        options = {"lamp": [0.3, 0.4, 0.6]}
    half, whole, n = _symmetric_scene(emission, diffuse, **options)
    points = _receivers()

    half_model = build_radiation_model(points, half, specular=["mirror"], settings=UNSHADOWED)
    whole_model = build_radiation_model(points, whole, settings=UNSHADOWED)
    calls = {"absorption": absorption}
    half_out, _ = radiosity(half_model, half, **calls)
    whole_out, _ = radiosity(whole_model, whole, **calls)
    np.testing.assert_allclose(half_out[:n], whole_out[:n], rtol=1e-9)
    half_in, _ = surface_irradiance(half_model, half, **calls)
    whole_in, _ = surface_irradiance(whole_model, whole, **calls)
    np.testing.assert_allclose(half_in[:n], whole_in[:n], rtol=1e-9)
    half_field, _ = fluence_rate(half_model, half, **calls)
    whole_field, _ = fluence_rate(whole_model, whole, **calls)
    assert float(jnp.min(whole_field)) > 0.0
    np.testing.assert_allclose(half_field, whole_field, rtol=1e-9)


# ---------------------------------------------------------------------------------------
# A uniform enclosure, one bounce
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("diffuse", "specular"), [(0.0, 0.6), (0.3, 0.5), (0.6, 0.2)])
def test_a_uniform_box_of_part_mirror_walls_reaches_its_one_bounce_closed_form(diffuse, specular):
    """Every wall emitting ``M``, reflecting ``rho_d`` diffusely and ``rho_s`` as a mirror: each
    direction from a facet meets a wall and, once reflected, another, so one specular bounce
    carries ``(1 + rho_s)`` of what leaves Lambertian. The radiosity solves to
    ``B = M / (1 - rho_d (1 + rho_s))`` and a point inside sees ``G = 4 B (1 + rho_s)``. Dropping the
    specular transfer, or the mirrored gather, gives a different number."""
    exitance = 2.0
    surfaces = Surfaces.from_triangles(
        inward_box(1),
        solid_names=("box",),
        emission=exitance,
        diffuse_reflectance=diffuse,
        specular_reflectance=specular,
    )
    points = np.array([[0.5, 0.5, 0.5], [0.2, 0.7, 0.4], [0.8, 0.15, 0.9]])
    model = build_radiation_model(points, surfaces, specular=["box"], settings=UNSHADOWED)
    assert len(model.transfer.mirrors) == 6
    outgoing, _ = radiosity(model, surfaces)
    expected = exitance / (1.0 - diffuse * (1.0 + specular))
    np.testing.assert_allclose(outgoing, expected, rtol=1e-12)
    field, _ = fluence_rate(model, surfaces)
    np.testing.assert_allclose(field, 4.0 * expected * (1.0 + specular), rtol=1e-12)


def test_specular_exchange_is_reciprocal_to_the_receiving_quadrature():
    """``A_i E_ij = A_j E_ji``, as for the direct transfer, and like it held only to the
    quadrature over the receiving facet: the residual falls with the rule's points."""
    surfaces = Surfaces.from_triangles(
        inward_box(1) * np.array([1.0, 1.0, 3.0]), solid_names=("box",)
    )
    residuals = []
    for points in (1, 6):
        transfer = build_transfer(
            surfaces,
            receiver_quadrature=points,
            self_occlusion=NoOcclusion(),
            specular=["box"],
        )
        residuals.append(
            reciprocity_residual(
                transfer.__class__(
                    geometric=transfer.specular_geometric[0],
                    source_cosine=transfer.source_cosine,
                    separation=transfer.separation,
                    visibility=transfer.visibility,
                ),
                surfaces.area,
            )
        )
    assert residuals[1] < 0.2 * residuals[0]
    assert residuals[1] < 0.05


# ---------------------------------------------------------------------------------------
# What is live
# ---------------------------------------------------------------------------------------


def _mirrored_room():
    """A box whose floor is a part mirror, lit by a lamp patch on the ceiling."""
    walls = inward_box(2)
    floor = np.all(walls[:, :, 2] == 0.0, axis=1)
    ceiling = np.all(walls[:, :, 2] == 1.0, axis=1)
    solid = np.where(floor, 1, 0)
    surfaces = Surfaces.from_triangles(
        walls,
        solid_id=solid,
        solid_names=("walls", "floor"),
        emission=np.where(ceiling, 5.0, 0.0),
        diffuse_reflectance=np.where(floor, 0.2, 0.5),
        specular_reflectance=np.where(floor, 0.6, 0.0),
    )
    points = np.array([[0.3, 0.4, 0.5], [0.7, 0.2, 0.8]])
    return surfaces, build_radiation_model(
        points, surfaces, specular=["floor"], settings=UNSHADOWED
    )


def test_the_specular_reflectance_and_the_medium_are_differentiable_through_the_solve():
    """Both against central differences over the whole call: the solve's adjoint, the specular
    transfer and the mirrored gather all carry them."""
    surfaces, model = _mirrored_room()
    floor = np.asarray(surfaces.solid_id) == 1

    def total(specular, coefficient):
        optics = surfaces.with_optics(specular_reflectance=jnp.where(floor, specular, 0.0))
        field, _ = fluence_rate(model, optics, absorption=UniformAbsorption(coefficient))
        return jnp.sum(field)

    at = (0.6, 0.8)
    gradient = jax.grad(total, argnums=(0, 1))(*at)
    step = 1e-6
    for k, derivative in enumerate(gradient):
        up, down = list(at), list(at)
        up[k] += step
        down[k] -= step
        difference = (total(*up) - total(*down)) / (2.0 * step)
        assert abs(float(derivative)) > 0.0
        np.testing.assert_allclose(derivative, difference, rtol=1e-6)


# ---------------------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------------------


def test_a_specular_body_is_refused_where_something_could_shadow_a_reflected_path():
    surfaces, _ = _mirrored_room()
    points = np.array([[0.3, 0.4, 0.5]])
    with pytest.raises(NotImplementedError, match="nothing yet standing in the way"):
        build_radiation_model(points, surfaces, specular=["floor"])
    with pytest.raises(NotImplementedError, match="nothing yet standing in the way"):
        build_radiation_model(
            points,
            surfaces,
            specular=["floor"],
            occluders=[Sphere([0.5, 0.5, 0.5], 0.05)],
            settings=UNSHADOWED,
        )
    with pytest.raises(NotImplementedError, match="receivers' self-occlusion must be off"):
        build_radiation_model(
            points,
            surfaces,
            specular=["floor"],
            settings=RadiationSettings(
                self_occlusion=NoOcclusion(), receiver_occlusion=RayCastOcclusion()
            ),
        )


def test_a_specular_reflectance_must_be_one_value_on_a_declared_body_and_none_elsewhere():
    surfaces, model = _mirrored_room()
    floor = np.asarray(surfaces.solid_id) == 1
    varying = np.where(floor, np.linspace(0.1, 0.6, surfaces.n_facets), 0.0)
    with pytest.raises(ValueError, match="a specular body reflects with one value"):
        radiosity(model, surfaces.with_optics(specular_reflectance=varying))
    stray = np.where(floor, 0.6, 0.0) + np.where(np.asarray(surfaces.emission) > 0, 0.1, 0.0)
    with pytest.raises(ValueError, match=r"bodies \['walls'\] have a specular reflectance"):
        radiosity(model, surfaces.with_optics(specular_reflectance=stray))


def test_a_graded_medium_is_refused_with_specular_bodies():
    surfaces, model = _mirrored_room()
    medium = VoxelAbsorption(
        np.full((2, 2, 2), 1.0), origin=[0.0, 0.0, 0.0], spacing=[0.5, 0.5, 0.5]
    )
    with pytest.raises(NotImplementedError, match="a graded one cannot"):
        radiosity(model, surfaces, absorption=medium)


def test_a_curved_specular_body_is_refused_past_the_plane_limit():
    tube = cylinder_triangles(0.1, 0.5, sectors=12, slices=2)
    surfaces = Surfaces.from_triangles(tube, solid_names=("sleeve",))
    with pytest.raises(ValueError, match="lies in 12 planes, past the limit of 8"):
        build_transfer(
            surfaces, self_occlusion=NoOcclusion(), specular=["sleeve"], max_mirror_planes=8
        )
