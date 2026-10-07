"""Specular reflection through the assembled model: the transfer, the solve and the volume field."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.absorption import UniformAbsorption, VoxelAbsorption
from aquaflux.radiation.images import plane_exchange
from aquaflux.radiation.model import (
    RadiationSettings,
    build_radiation_model,
    fluence_rate,
    radiosity,
    surface_irradiance,
)
from aquaflux.radiation.profiles import CosinePower, Isotropic, Lambertian
from aquaflux.radiation.quadrature import triangle_quadrature
from aquaflux.radiation.self_occlusion import (
    NoOcclusion,
    RayCastOcclusion,
    SilhouetteOcclusion,
)
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.transfer import build_transfer, reciprocity_residual
from aquaflux.solids import Box, Sphere

from tests.unit.radiation_references import cylinder_triangles, inward_box, rectangle_triangles

UNSHADOWED = RadiationSettings(self_occlusion=NoOcclusion())


#: A plate standing in the left half of the box, facing up: the surface's own triangles across
#: some of the paths, direct and reflected. Its edges sit off the box's thirds and sixths on
#: purpose: at x = 0.35 a reflected path between two wall centroids grazed the edge exactly, and
#: an edge hit is decided by rounding, differently along the half's leg and the whole's segment.
BAFFLE = rectangle_triangles([0.24, 0.43, 0.47], [0.093, 0.0, 0.0], [0.0, 0.137, 0.0])

#: A partly transmitting ball in the left half, away from the baffle.
BALL = Sphere([0.3, 0.75, 0.3], 0.08)


def _reflected(points):
    """``points`` reflected in the plane of symmetry ``x = 0.5``."""
    return np.asarray(points) * np.array([-1.0, 1.0, 1.0]) + np.array([1.0, 0.0, 0.0])


def _halves(baffle=False):
    """The half ``x <= 0.5`` of a closed unit box, and its mirror image across ``x = 0.5``.

    Triangle for triangle, so the whole box made of the two is mirror-symmetric and not merely
    two translates. With ``baffle``, :data:`BAFFLE` stands in the left half and its image in the
    right.
    """
    walls = inward_box(2)
    left = walls[np.all(walls[:, :, 0] <= 0.5, axis=1)]
    if baffle:
        left = np.concatenate([left, BAFFLE])
    right = _reflected(left)[:, ::-1]
    return left, right


def _symmetric_scene(emission, diffuse, *, profiles=None, profile_index=0, lamp=None, baffle=False):
    """A whole box, and its left half closed by a perfect mirror across the plane of symmetry.

    The whole box carries the left half's optics on both halves; the half box carries them on
    its walls and a mirror of specular reflectance one, which emits and diffuses nothing. With
    ``lamp``, a point source sits at that position in the half box and at it and its reflection
    in the whole; with ``baffle``, :data:`BAFFLE` is one more part of the walls.
    """
    left, right = _halves(baffle)
    n = len(left)
    mirror = rectangle_triangles([0.5, 0.5, 0.5], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0])
    profiles = (Lambertian(),) if profiles is None else profiles
    index = np.broadcast_to(profile_index, (n,))
    lamps = [] if lamp is None else [np.asarray(lamp, dtype=float)]
    reflected = [] if lamp is None else [_reflected(lamps[0])]
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
    """Points in the left half, none inside :data:`BALL`."""
    rng = np.random.default_rng(seed)
    points = rng.uniform([0.05, 0.05, 0.05], [0.45, 0.95, 0.95], size=(3 * count, 3))
    return points[~np.asarray(BALL.contains(points))][:count]


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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", ["lambertian", "point lamp"])
def test_a_mirror_on_a_plane_of_symmetry_reproduces_the_whole_scene_s_shadows(case, stream):
    """The symmetry plane again, with a baffle of the walls' own triangles and a partly
    transmitting ball in the half, and both again in the whole. A path reflected in the mirror
    is the whole scene's straight path folded at the plane, so the half's two legs cross the
    ball and the baffle exactly where the whole's one segment crosses them and their images --
    and the half's crossings of its one ball, filtered by its transmittance once per leg, must be
    the whole's crossings of two balls of that transmittance. Held to the same 1e-9, which only a
    mask that tests both legs, counts a body per leg and drops a path its own triangles hide can
    reach; streaming the receivers' masks must not change it."""
    rng = np.random.default_rng(2)
    n_left = len(_halves(baffle=True)[0])
    emission = rng.uniform(0.0, 3.0, n_left)
    diffuse = rng.uniform(0.2, 0.8, n_left)
    options = {}
    if case == "point lamp":
        emission = np.zeros(n_left)
        options = {"lamp": [0.3, 0.3, 0.75]}
    half, whole, n = _symmetric_scene(emission, diffuse, baffle=True, **options)
    points = _receivers()
    image_of_ball = Sphere(_reflected(BALL.centre), BALL.radius)

    settings = RadiationSettings(stream_receiver_mask=stream)
    half_model = build_radiation_model(
        points, half, specular=["mirror"], occluders=[BALL], settings=settings
    )
    whole_model = build_radiation_model(points, whole, occluders=[BALL, image_of_ball])
    absorption = UniformAbsorption(0.7)
    half_calls = {"absorption": absorption, "transmittance": [0.4]}
    whole_calls = {"absorption": absorption, "transmittance": [0.4, 0.4]}
    half_out, _ = radiosity(half_model, half, **half_calls)
    whole_out, _ = radiosity(whole_model, whole, **whole_calls)
    np.testing.assert_allclose(half_out[:n], whole_out[:n], rtol=1e-9)
    half_field, _ = fluence_rate(half_model, half, **half_calls)
    whole_field, _ = fluence_rate(whole_model, whole, **whole_calls)
    np.testing.assert_allclose(half_field, whole_field, rtol=1e-9)

    # The shadows matter here: without them the half scene is brighter.
    bare = build_radiation_model(points, half, specular=["mirror"], settings=UNSHADOWED)
    bare_field, _ = fluence_rate(bare, half, absorption=absorption)
    assert float(jnp.max(bare_field / whole_field)) > 1.01


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


def _plates_over_a_mirror():
    """Two lamp patches facing down at height one over a floor mirror, and a slab between.

    The patches face the same way, so neither sees the other straight; every path between them
    is reflected in the floor, and crosses the slab once going down and once coming up.
    """
    floor = rectangle_triangles([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0])
    left = rectangle_triangles([-0.6, 0.0, 1.0], [0.0, 0.2, 0.0], [0.2, 0.0, 0.0])
    right = rectangle_triangles([0.6, 0.0, 1.0], [0.0, 0.2, 0.0], [0.2, 0.0, 0.0])
    surfaces = Surfaces.from_triangles(
        np.concatenate([floor, left, right]),
        solid_id=[0, 0, 1, 1, 1, 1],
        solid_names=("floor", "lamps"),
        emission=[0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
        diffuse_reflectance=[0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        specular_reflectance=[0.8, 0.8, 0.0, 0.0, 0.0, 0.0],
    )
    assert np.all(np.asarray(surfaces.normal)[2:, 2] < 0.0)
    slab = Box([0.0, 0.0, 0.5], [5.0, 5.0, 0.05])
    return surfaces, slab


def test_a_body_crossed_on_both_legs_of_a_reflected_path_filters_it_twice():
    """Every path between the patches crosses the slab twice and a ball beside the left patch
    once, so the transfer between them is the unshadowed one times the slab's transmittance
    squared times the ball's -- one term, of pattern (2, 1) -- and its derivative with respect to
    the slab's transmittance is twice that transmittance times the rest."""
    surfaces, slab = _plates_over_a_mirror()
    ball = Sphere([-0.3, 0.0, 0.5], 0.15)
    shadowed = build_transfer(surfaces, specular=["floor"], occluders=[slab, ball])
    bare = build_transfer(surfaces, specular=["floor"], self_occlusion=NoOcclusion())
    patches = np.ix_([4, 5], [2, 3])
    assert (0, (2, 1)) in shadowed.specular_terms

    def between(slab_transmittance):
        reflected, _ = shadowed.assemble(
            surfaces, transmittance=jnp.asarray([slab_transmittance, 0.6])
        )
        return reflected[patches]

    clear, _ = bare.assemble(surfaces)
    assert float(jnp.min(clear[patches])) > 0.0
    np.testing.assert_allclose(between(0.3), 0.3**2 * 0.6 * clear[patches], rtol=1e-12)
    slope = jax.jacfwd(between)(0.3)
    np.testing.assert_allclose(slope, 2 * 0.3 * 0.6 * clear[patches], rtol=1e-12)


def test_reflected_shadows_follow_the_transmittance_to_a_finite_difference():
    """The field's derivative with respect to a body's transmittance, through the transfer's
    pattern terms, the solve and the mirrored gather -- held and streamed alike."""
    rng = np.random.default_rng(3)
    n_left = len(_halves(baffle=True)[0])
    half, _, _ = _symmetric_scene(
        rng.uniform(0.0, 3.0, n_left), rng.uniform(0.2, 0.8, n_left), baffle=True
    )
    points = _receivers(6)
    slopes = []
    for stream in (False, True):
        model = build_radiation_model(
            points,
            half,
            specular=["mirror"],
            occluders=[BALL],
            settings=RadiationSettings(stream_receiver_mask=stream),
        )

        def total(transmittance, model=model):
            field, _ = fluence_rate(model, half, transmittance=jnp.asarray([transmittance]))
            return jnp.sum(field)

        slope = jax.grad(total)(0.4)
        step = 1e-6
        difference = (total(0.4 + step) - total(0.4 - step)) / (2.0 * step)
        assert abs(float(slope)) > 0.0
        np.testing.assert_allclose(slope, difference, rtol=1e-6)
        slopes.append(slope)
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=1e-12)


# The half box is open where the mirror closes it, which the silhouette clip warns about.
@pytest.mark.filterwarnings("ignore:SilhouetteOcclusion")
def test_reflected_paths_are_rays_whichever_self_occlusion_strategy_is_chosen():
    """The silhouette clip gives the direct pairs an exact fraction, but a reflected path has no
    single source view to clip, so its legs are one ray each, as under the ray test."""
    rng = np.random.default_rng(4)
    n_left = len(_halves(baffle=True)[0])
    half, _, _ = _symmetric_scene(
        rng.uniform(0.0, 3.0, n_left), rng.uniform(0.2, 0.8, n_left), baffle=True
    )
    points = _receivers(6)
    by_ray, by_clip = (
        build_radiation_model(
            points,
            half,
            specular=["mirror"],
            settings=RadiationSettings(self_occlusion=strategy),
        )
        for strategy in (RayCastOcclusion(), SilhouetteOcclusion())
    )
    assert by_ray.transfer.specular_terms == by_clip.transfer.specular_terms
    np.testing.assert_array_equal(
        by_ray.transfer.specular_geometric, by_clip.transfer.specular_geometric
    )
    for ray, clip in zip(
        by_ray.receiver_shadows.mirror_visibility,
        by_clip.receiver_shadows.mirror_visibility,
        strict=True,
    ):
        assert bool(jnp.any(ray.hidden))
        np.testing.assert_array_equal(ray.hidden, clip.hidden)


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


def test_by_default_a_box_is_flat_enough_and_a_tessellated_tube_is_not():
    """The default limit admits the six planes of a box and refuses a tube cut into sixteen
    strips."""
    box = Surfaces.from_triangles(inward_box(1), solid_names=("box",))
    built = build_transfer(box, self_occlusion=NoOcclusion(), specular=["box"])
    assert len(built.mirrors) == 6
    tube = Surfaces.from_triangles(
        cylinder_triangles(0.1, 0.5, sectors=16, slices=2), solid_names=("sleeve",)
    )
    with pytest.raises(ValueError, match="lies in 16 planes, past the limit of 12"):
        build_transfer(tube, self_occlusion=NoOcclusion(), specular=["sleeve"])


def test_a_body_s_planes_share_their_path_lengths_weighted_by_what_each_carries():
    """A body of several planes keeps one path length per pair of facets: the mean over its planes,
    weighted by each plane's share of the exchange. That is what makes the attenuation exact to
    first order in the absorption coefficient -- its derivative at zero is minus the sum over the
    planes of each one's exchange times its own path length, which an unweighted mean, or any
    one plane's length, does not give."""
    surfaces = Surfaces.from_triangles(
        inward_box(1) * np.array([1.0, 2.0, 3.0]), solid_names=("box",), specular_reflectance=1.0
    )
    transfer = build_transfer(
        surfaces, receiver_quadrature=1, self_occlusion=NoOcclusion(), specular=["box"]
    )
    assert len(transfer.mirrors) == 6
    rule = triangle_quadrature(1)
    # The direct transfer is attenuated over its own centroid separations, alongside.
    expected = -transfer.geometric * transfer.separation - sum(
        exchange.geometric * exchange.separation
        for exchange in (
            plane_exchange(mirror, surfaces, rule.points(surfaces.vertices), rule.weight)
            for mirror in transfer.mirrors
        )
    )

    def reflected(coefficient):
        matrix, _ = transfer.assemble(surfaces, UniformAbsorption(coefficient))
        return matrix

    slope = jax.jacfwd(reflected)(0.0)
    np.testing.assert_allclose(slope, expected, atol=1e-14)
    assert float(jnp.max(jnp.abs(slope))) > 0.0
