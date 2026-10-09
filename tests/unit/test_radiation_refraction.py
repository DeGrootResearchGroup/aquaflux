"""Transparent solids: Fresnel transmittance, the media a point is in, and the refracted path.

Each check is against an answer worked out another way -- the amplitude form of the Fresnel
coefficients, the optical length minimized by a general-purpose optimizer, a finite difference --
so a slip in the path solve, the crossing order or the transmittance moves a number rather than
only a shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import Surfaces, UniformAbsorption
from aquaflux.radiation.refraction import (
    Chain,
    Media,
    Transparent,
    fresnel_transmittance,
    solve_paths,
    straight_reach,
)
from aquaflux.solids import Cylinder, HalfSpace, Sphere, Union
from scipy import optimize

WATER, QUARTZ, AIR = 1.376, 1.5048, 1.0003


# ---------------------------------------------------------------------------------------------
# Fresnel
# ---------------------------------------------------------------------------------------------


def test_at_normal_incidence_the_reflected_share_is_the_square_of_the_index_contrast():
    """``R = ((n1 - n2) / (n1 + n2))^2`` either way across: 0.2 % water-quartz, 4 % quartz-air."""
    for n1, n2 in ((WATER, QUARTZ), (QUARTZ, AIR), (AIR, QUARTZ)):
        expected = 1.0 - ((n1 - n2) / (n1 + n2)) ** 2
        assert float(fresnel_transmittance(1.0, n1, n2)) == pytest.approx(expected, rel=1e-15)


def test_the_transmittance_is_the_amplitude_coefficients_power_share_at_every_angle():
    """Against the transmitted amplitudes ``t_s``, ``t_p`` and the beam's change of width.

    The power carried across is ``(t_s^2 + t_p^2) / 2`` times ``n2 cos(theta_t) / (n1
    cos(theta_i))`` -- written from the transmission side, where the function under test is
    written from the reflection side, so the two meet only if both are right.
    """
    cos_i = np.linspace(0.02, 1.0, 60)
    for n1, n2 in ((WATER, QUARTZ), (QUARTZ, WATER), (AIR, QUARTZ)):
        sin_t2 = (n1 / n2) ** 2 * (1.0 - cos_i**2)
        ok = sin_t2 < 1.0
        cos_t = np.sqrt(1.0 - sin_t2[ok])
        t_s = 2 * n1 * cos_i[ok] / (n1 * cos_i[ok] + n2 * cos_t)
        t_p = 2 * n1 * cos_i[ok] / (n2 * cos_i[ok] + n1 * cos_t)
        expected = 0.5 * (t_s**2 + t_p**2) * (n2 * cos_t) / (n1 * cos_i[ok])
        got = np.asarray(fresnel_transmittance(cos_i[ok], n1, n2))
        np.testing.assert_allclose(got, expected, rtol=1e-13)


def test_beyond_the_critical_angle_nothing_crosses_and_the_crossing_is_the_same_both_ways():
    critical = np.sqrt(1.0 - (AIR / QUARTZ) ** 2)
    assert float(fresnel_transmittance(critical * (1 - 1e-9), QUARTZ, AIR)) == 0.0
    assert float(fresnel_transmittance(critical * (1 + 1e-6), QUARTZ, AIR)) > 0.0
    # Reciprocity: across one interface at one pair of angles, the same share either way.
    cos_1 = np.linspace(0.1, 1.0, 20)
    cos_2 = np.sqrt(1.0 - (WATER / QUARTZ) ** 2 * (1.0 - cos_1**2))
    np.testing.assert_allclose(
        np.asarray(fresnel_transmittance(cos_1, WATER, QUARTZ)),
        np.asarray(fresnel_transmittance(cos_2, QUARTZ, WATER)),
        rtol=1e-13,
    )


# ---------------------------------------------------------------------------------------------
# Media and chains
# ---------------------------------------------------------------------------------------------


def _sleeve(length: float = 2.0) -> Media:
    """A quartz sleeve about the z axis with its air gap inside it, in water."""
    axis = [0.0, 0.0, 1.0]
    gap = Transparent(Cylinder([0, 0, 0], axis, 0.01025, length / 2), AIR)
    quartz = Transparent(Cylinder([0, 0, 0], axis, 0.0115, length / 2), QUARTZ, inside=(gap,))
    return Media(WATER, (quartz,))


def test_a_point_is_in_the_innermost_region_holding_it():
    media = _sleeve()
    points = np.array([[0.005, 0, 0], [0.011, 0, 0], [0.02, 0, 0]])
    np.testing.assert_array_equal(media.region_of(points), [1, 0, -1])
    np.testing.assert_array_equal(media.parents, [-1, 0])


def test_a_point_on_a_surface_or_in_overlapping_or_escaping_regions_is_refused():
    media = _sleeve()
    with pytest.raises(ValueError, match="lie on the surface of transparent region 1"):
        media.region_of(np.array([[0.01025, 0.0, 0.0]]))
    overlapping = Media(
        1.0, (Transparent(Sphere([0, 0, 0], 1.0), 1.5), Transparent(Sphere([1.5, 0, 0], 1.0), 1.5))
    )
    with pytest.raises(ValueError, match="neither of which holds the other"):
        overlapping.region_of(np.array([[0.8, 0.0, 0.0]]))
    escaping = Media(
        1.0,
        (
            Transparent(
                Sphere([0, 0, 0], 1.0), 1.5, inside=(Transparent(Sphere([1.0, 0, 0], 0.5), 1.2),)
            ),
        ),
    )
    with pytest.raises(ValueError, match="but not inside region 0"):
        escaping.region_of(np.array([[1.3, 0.0, 0.0]]))


def test_a_region_must_be_convex():
    with pytest.raises(TypeError, match="must be a convex solid"):
        Transparent(Union(Sphere([0, 0, 0], 1.0), Sphere([1, 0, 0], 1.0)), 1.5)


def test_a_chain_leaves_every_region_round_the_source_and_enters_every_one_round_the_receiver():
    """Out of the air gap and then the quartz; between two siblings, out of one and into the other."""
    sleeve = _sleeve()
    out = Chain.between(sleeve.parents, 1, -1)
    assert out.crossings == ((1, True), (0, True))
    assert out.legs == (1, 0, -1)
    assert Chain.between(sleeve.parents, -1, 1).crossings == ((0, False), (1, False))
    # Two cells inside one tank: the tank holds both, so it is not crossed.
    parents = np.array([-1, 0, 0])
    across = Chain.between(parents, 1, 2)
    assert across.crossings == ((1, True), (2, False))
    assert across.legs == (1, 0, 2)
    # The leg through the tank may pass straight through nothing it crossed, nor the tank.
    assert across.beside == ((), (), ())


def test_a_route_through_a_region_enters_it_from_a_leg_s_medium_and_leaves_it_the_same_way():
    """Two sleeves side by side: from one's air gap to the water, and from the water to the water.

    Regions 0 and 2 are the quartz of each, 1 and 3 their gaps. Out of the first gap the path may
    pass through nothing, through the second sleeve's quartz, or through its quartz and gap; each
    pass sits on the water leg, entered outermost first and left in the reverse. A region holding an
    end is never passed, and between two points in one medium there is no route through nothing --
    that is the straight line.
    """
    parents = np.array([-1, 0, -1, 2])
    routes = Chain.routes(parents, 1, -1)
    assert [chain.passing for chain in routes] == [(), (2,), (2, 3)]
    assert routes[0].crossings == ((1, True), (0, True))
    assert routes[1].crossings == ((1, True), (0, True), (2, False), (2, True))
    assert routes[1].legs == (1, 0, -1, 2, -1)
    assert routes[2].crossings == (
        (1, True),
        (0, True),
        (2, False),
        (3, False),
        (3, True),
        (2, True),
    )
    assert routes[2].legs == (1, 0, -1, 2, 3, 2, -1)
    assert routes[1].pass_at == routes[2].pass_at == 2
    # The route through nothing must miss the second sleeve; the one through its quartz alone must
    # miss its gap, in the quartz too; the one across the gap misses nothing.
    assert routes[0].beside[-1] == (2, 3)
    assert routes[1].beside[3] == routes[1].beside[-1] == (3,)
    assert routes[2].beside[-1] == ()
    assert [chain.n_starts for chain in routes] == [1, 4, 4]
    in_water = Chain.routes(parents, -1, -1)
    assert [chain.passing for chain in in_water] == [(0,), (0, 1), (2,), (2, 3)]
    assert in_water[0].crossings == ((0, False), (0, True))
    # From the first gap to the first gap nothing can be passed without leaving it.
    assert Chain.routes(parents, 1, 1) == ()
    with pytest.raises(ValueError, match="holds an end of the path"):
        Chain.between(parents, 1, -1, through=0)
    with pytest.raises(ValueError, match="no leg of the path runs in a medium holding region 3"):
        Chain.between(parents, 1, 1, through=3)


# ---------------------------------------------------------------------------------------------
# The path
# ---------------------------------------------------------------------------------------------


def _optical_length_minimum(media, indices, surfaces, source, receiver, guess):
    """The path minimizing ``sum n |leg|`` with each crossing on its surface, by a general optimizer.

    ``surfaces`` maps an unconstrained vector to the crossing points: an independent
    parameterization of the same surfaces, so nothing of the solver under test is reused.
    """

    def length(parameters):
        points = [np.asarray(source), *surfaces(parameters), np.asarray(receiver)]
        return sum(
            n * np.linalg.norm(b - a)
            for n, a, b in zip(indices, points[:-1], points[1:], strict=True)
        )

    result = optimize.minimize(
        length,
        guess,
        method="Nelder-Mead",
        options={"xatol": 1e-13, "fatol": 1e-16, "maxiter": 40000},
    )
    return np.asarray(surfaces(result.x))


def test_through_a_flat_interface_the_path_is_the_shortest_optical_one_and_obeys_snell():
    """Including a source seen at grazing incidence through an interface close to the receiver.

    That one is where Newton's method started from the straight line runs away -- the crossing is
    a fifth of a millimetre from where the straight line meets the plane, over a separation of
    three -- and only the descent on the optical length reaches it.
    """
    media = Media(WATER, (Transparent(HalfSpace([0, 0, 0], [0, 0, 1]), AIR),))
    chain = Chain.between(media.parents, 0, -1)
    for source, receiver in (
        ([0.02, 0.01, -0.03], [0.0, 0.0, 0.05]),
        ([0.0135, 0, -0.003], [0, 0, 0.0002]),
    ):
        paths = solve_paths(media, chain, jnp.asarray(source), jnp.asarray(receiver))
        assert bool(paths.valid)
        found = np.asarray(paths.points)[0]
        best = _optical_length_minimum(
            media,
            (AIR, WATER),
            lambda p: [np.array([p[0], p[1], 0.0])],
            source,
            receiver,
            [0.0, 0.0],
        )[0]
        np.testing.assert_allclose(
            found, best, atol=1e-7 * np.linalg.norm(np.subtract(receiver, source))
        )
        before, after = np.asarray(paths.departure), -np.asarray(paths.arrival)
        assert AIR * np.hypot(*before[:2]) == pytest.approx(WATER * np.hypot(*after[:2]), rel=1e-8)


def test_out_of_a_sleeve_the_path_is_the_shortest_optical_one_off_the_cross_section_too():
    """Two coaxial cylinders, a source on the arc and a receiver off its plane: a skew path."""
    media = _sleeve()
    chain = Chain.between(media.parents, 1, -1)
    source = np.array([0.0075 * np.cos(0.3), 0.0075 * np.sin(0.3), 0.004])
    receiver = np.array([0.02, -0.006, -0.003])
    paths = solve_paths(media, chain, jnp.asarray(source), jnp.asarray(receiver))
    assert bool(paths.valid)

    def on_cylinders(p):
        return [
            np.array([0.01025 * np.cos(p[0]), 0.01025 * np.sin(p[0]), p[1]]),
            np.array([0.0115 * np.cos(p[2]), 0.0115 * np.sin(p[2]), p[3]]),
        ]

    straight = np.asarray(paths.points)
    guess = [
        np.arctan2(straight[0, 1], straight[0, 0]) + 0.05,
        0.0,
        np.arctan2(straight[1, 1], straight[1, 0]) - 0.05,
        0.0,
    ]
    best = _optical_length_minimum(
        media, (AIR, QUARTZ, WATER), on_cylinders, source, receiver, guess
    )
    np.testing.assert_allclose(
        np.asarray(paths.points), best, atol=1e-7 * np.linalg.norm(receiver - source)
    )


def test_a_crossing_off_the_end_of_its_face_is_no_crossing_and_carries_nothing():
    """A short air cylinder in glass, left by the side on the straight line and above it on the path.

    The path is held to the face the straight line leaves by. Here its stationary point on that
    face is beyond the flat end -- on the tube's extension, not on the body -- as the same chain on
    a cylinder long enough to have no end there shows; counting it would count light through glass
    that is not there.
    """
    # The straight line leaves the side at z = 0.0983, under the end at 0.1; leaving air for glass
    # the path is steeper inside, and crosses the side's line at 0.1007.
    source, receiver = jnp.array([0.0, 0.0, 0.09]), jnp.array([0.3, 0.0, 0.115])

    def glass(half_length):
        return Media(1.5, (Transparent(Cylinder([0, 0, 0], [0, 0, 1], 0.1, half_length), 1.0),))

    chain = Chain.between(glass(0.1).parents, 0, -1)
    long = solve_paths(glass(10.0), chain, source, receiver)
    assert bool(long.valid)
    assert float(long.points[0, 2]) > 0.1  # beyond where the short cylinder ends
    short = solve_paths(glass(0.1), chain, source, receiver)
    assert not bool(short.valid)
    assert float(short.transmittance) == 0.0


def test_the_path_s_derivative_with_respect_to_an_index_is_the_finite_difference():
    """The implicit derivative through the converged path, not the iterations on the tape."""
    source = jnp.array([0.0075 * np.cos(0.4), 0.0075 * np.sin(0.4), 0.002])
    receiver = jnp.array([0.03, 0.005, 0.0])
    chain = Chain.between(_sleeve().parents, 1, -1)

    def arrival(quartz):
        gap = _sleeve().regions[0].inside
        media = Media(
            WATER, (Transparent(Cylinder([0, 0, 0], [0, 0, 1], 0.0115, 1.0), quartz, inside=gap),)
        )
        paths = solve_paths(media, chain, source, receiver)
        return paths.arrival[1] + paths.transmittance

    derivative = float(jax.grad(arrival)(QUARTZ))
    step = 1e-6
    finite = (float(arrival(QUARTZ + step)) - float(arrival(QUARTZ - step))) / (2 * step)
    assert derivative != 0.0
    assert derivative == pytest.approx(finite, rel=1e-6)


def test_a_region_holding_neither_end_is_passed_through_bent_and_a_path_meeting_it_otherwise_is_none():
    """A sphere centred on the line between two points, both outside it, is a route of its own.

    On that route the axial path crosses it at normal incidence both ways, so what gets past is
    ``(1 - R)^2`` times the sphere's own absorption along the diameter, less what the water would have
    absorbed there; its several starting points find that one path once. On the route that goes
    round the sphere the only stationary path is the same straight line, which meets the sphere and
    so belongs to the other route: there it is no path at all.
    """
    absorbing, water = 7.0, 2.0
    lens = Transparent(Sphere([0.0, 0.0, 0.05], 0.01), QUARTZ, UniformAbsorption(absorbing))
    box = Transparent(Sphere([0, 0, 0], 0.001), WATER)
    source, receiver = jnp.array([0.0, 0.0, 0.0002]), jnp.array([0.0, 0.0, 0.1])
    with_lens = Media(WATER, (box, lens), UniformAbsorption(water))
    without = Media(WATER, (box,), UniformAbsorption(water))
    routes = Chain.routes(with_lens.parents, 0, -1)
    assert [chain.passing for chain in routes] == [(), (1,)]
    assert routes[0].beside[-1] == (1,)
    around = solve_paths(with_lens, routes[0], source, receiver)
    assert not bool(around.valid)
    assert float(around.transmittance) == 0.0
    through = solve_paths(with_lens, routes[1], source, receiver)
    assert through.valid.shape == (routes[1].n_starts,)
    assert int(np.sum(np.asarray(through.valid))) == 1
    plain = solve_paths(without, Chain.between(without.parents, 0, -1), source, receiver)
    ratio = float(jnp.sum(through.transmittance)) / float(plain.transmittance)
    normal = 1.0 - ((WATER - QUARTZ) / (WATER + QUARTZ)) ** 2
    assert ratio == pytest.approx(normal**2 * np.exp(-(absorbing - water) * 0.02), rel=1e-12)


def _tiny_facets(centroids) -> Surfaces:
    """One small triangle about each centroid, in the plane normal to x."""
    corners = np.array([[0.0, -1.0, -1.0], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]]) * 1e-4 / 3
    return Surfaces.from_triangles(np.asarray(centroids)[:, None, :] + corners[None])


def test_a_straight_segment_carries_light_only_in_one_medium_and_clear_of_every_region():
    """Which centroid-to-point segments the straight gather takes, across a sleeve and beside it.

    A line through the sleeve's axis passes the quartz and the air gap, so its light goes by a route
    through them, bent, and not straight; one that misses the sleeve goes straight; a pair whose two
    ends lie in different media is the refracted gather's; and two ends in the air gap with nothing
    inside it are joined straight.
    """
    axis = [0.0, 0.0, 1.0]
    gap = Transparent(Cylinder([0, 0, 0], axis, 0.01025, 0.5), AIR)
    sleeve = Transparent(Cylinder([0, 0, 0], axis, 0.0115, 0.5), QUARTZ, inside=(gap,))
    media = Media(WATER, (sleeve,))
    facets = _tiny_facets([[-0.05, 0.0, 0.0], [0.0, 0.0, 0.002], [-0.05, 0.08, 0.0]])
    points = np.array([[0.05, 0.0, 0.0], [0.05, 0.08, 0.0], [0.0, 0.005, 0.0]])
    reach = straight_reach(media, facets, points)
    assert reach.dtype == bool
    expected = np.array(
        [
            [False, False, True],  # Through the sleeve's axis; another medium; misses it.
            [True, False, True],  # Both lines from the water miss the sleeve.
            [False, True, False],  # From the gap to the water is refracted; in the gap, straight.
        ]
    )
    np.testing.assert_array_equal(reach, expected)
    # However the pairs are cut into passes.
    np.testing.assert_array_equal(straight_reach(media, facets, points, pair_limit=2), reach)
