"""The direct gather through transparent solids, against answers worked out another way.

With every index equal the refracted gather is the direct gather, term for term. Through one flat
interface the fluence rate from a large flat Lambertian emitter is an integral over the receiver's
directions -- radiance times the square of the index ratio, the Fresnel transmittance and the
absorption along both legs -- that a quadrature evaluates independently of any path solve; that is
the case that pins the solid angle seen through the interface, the radiance ratio and the leg
absorption together.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import (
    NoOcclusion,
    Surfaces,
    UniformAbsorption,
    direct_fluence_rate,
    direct_irradiance,
    refracted,
)
from aquaflux.radiation.refracted import (
    build_refracted_visibility,
    refracted_fluence_rate,
    refracted_irradiance,
)
from aquaflux.radiation.refraction import Media, Paths, Transparent, fresnel_transmittance
from aquaflux.radiation.visibility import build_visibility
from aquaflux.solids import Box, Cylinder, HalfSpace, Sphere
from scipy import integrate, optimize

WATER, QUARTZ, AIR = 1.376, 1.5048, 1.0003


def _drum(radius: float, height: float, sectors: int, slices: int) -> np.ndarray:
    """An open drum about the z axis, wound outward, ``(n, 3, 3)``."""
    angle = np.linspace(0.0, 2.0 * np.pi, sectors + 1)[:-1]
    z = np.linspace(-height / 2, height / 2, slices + 1)
    triangles = []
    for i in range(sectors):
        a, b = angle[i], angle[(i + 1) % sectors]

        def at(t, h):
            return [radius * np.cos(t), radius * np.sin(t), h]

        for k in range(slices):
            triangles.append([at(a, z[k]), at(b, z[k]), at(b, z[k + 1])])
            triangles.append([at(a, z[k]), at(b, z[k + 1]), at(a, z[k + 1])])
    return np.asarray(triangles)


def _square(half_width: float, cells: int, height: float) -> np.ndarray:
    """A square in the plane ``z = height``, facing up, ``(2 cells^2, 3, 3)``."""
    xs = np.linspace(-half_width, half_width, cells + 1)
    triangles = []
    for i in range(cells):
        for j in range(cells):
            a, b = [xs[i], xs[j], height], [xs[i + 1], xs[j], height]
            c, d = [xs[i + 1], xs[j + 1], height], [xs[i], xs[j + 1], height]
            triangles.extend([[a, b, c], [a, c, d]])
    return np.asarray(triangles)


def _arc() -> Surfaces:
    surfaces = Surfaces.from_triangles(_drum(0.0075, 0.2, 24, 8))
    return surfaces.with_optics(emission=jnp.full(surfaces.n_facets, 100.0))


def _sleeve(water=WATER, quartz=QUARTZ, air=AIR, absorption=None) -> Media:
    axis = [0.0, 0.0, 1.0]
    gap = Transparent(Cylinder([0, 0, 0], axis, 0.01025, 0.5), air)
    return Media(
        water,
        (Transparent(Cylinder([0, 0, 0], axis, 0.0115, 0.5), quartz, inside=(gap,)),),
        absorption,
    )


_POINTS = np.array([[0.02, 0.0, 0.0], [0.015, 0.01, 0.03], [0.05, -0.02, 0.12], [0.0, 0.03, -0.05]])
_NORMALS = np.array([[-1.0, 0.0, 0.0], [-0.6, -0.8, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def test_with_every_index_equal_it_is_the_direct_gather():
    """The arrival directions are then the straight ones and every crossing transmits all of it."""
    surfaces, media = _arc(), _sleeve(water=1.3, quartz=1.3, air=1.3)
    np.testing.assert_allclose(
        np.asarray(refracted_fluence_rate(surfaces, media, _POINTS)),
        np.asarray(direct_fluence_rate(surfaces, _POINTS)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(refracted_irradiance(surfaces, media, _POINTS, _NORMALS)),
        np.asarray(direct_irradiance(surfaces, _POINTS, _NORMALS)),
        rtol=1e-12,
    )


def _over_directions(reach, carried, widest, half_width, cosine=False):
    """Integrate what arrives over the receiver's directions, from a square emitter centred below it.

    ``reach(theta)`` is how far sideways a ray arriving at polar angle ``theta`` meets the emitter's
    plane, and ``carried(theta)`` what radiance it brings per unit of the emitter's; ``widest`` is the
    largest polar angle any ray arrives at. Eight symmetric sectors; for each azimuth the polar limit
    is where the reach meets the square's edge. With ``cosine``, each direction is weighted by its
    cosine to the vertical, as on a surface facing down at the emitter.
    """

    def radiance(theta):
        weight = np.cos(theta) if cosine else 1.0
        return carried(theta) * np.sin(theta) * weight

    def sector(phi):
        top = widest * (1 - 1e-12)
        if reach(top) > half_width / np.cos(phi):
            top = optimize.brentq(
                lambda t: reach(t) - half_width / np.cos(phi), 0.0, top, xtol=1e-14
            )
        return integrate.quad(radiance, 0.0, top, epsabs=0, epsrel=1e-11, limit=200)[0]

    return 8.0 * integrate.quad(sector, 0.0, np.pi / 4, epsabs=0, epsrel=1e-10)[0]


def _through_a_plane(
    n_source, n_receiver, a_source, a_receiver, depth, height, half_width, exitance, cosine=False
):
    """Fluence rate on the axis above a square Lambertian emitter seen through the plane ``z = 0``.

    With ``cosine``, the irradiance on a surface there facing down at the emitter instead. Each
    direction arrives with the source's radiance, times ``(n_receiver / n_source)^2``, the Fresnel
    transmittance and both legs' absorption, wherever the refracted ray lands on the square.
    """

    def source_angle(theta):
        return np.arcsin(np.clip(n_receiver * np.sin(theta) / n_source, -1.0, 1.0))

    def reach(theta):
        return height * np.tan(theta) + depth * np.tan(source_angle(theta))

    def carried(theta):
        inside = source_angle(theta)
        crossing = float(fresnel_transmittance(np.cos(inside), n_source, n_receiver))
        legs = a_source * depth / np.cos(inside) + a_receiver * height / np.cos(theta)
        return (n_receiver / n_source) ** 2 * crossing * exitance / np.pi * np.exp(-legs)

    widest = np.pi / 2 if n_source >= n_receiver else np.arcsin(n_source / n_receiver)
    return _over_directions(reach, carried, widest, half_width, cosine)


@pytest.mark.parametrize(
    ("n_source", "n_receiver", "cosine"),
    [(AIR, WATER, False), (1.5, 1.0, False), (AIR, WATER, True)],
)
def test_through_a_flat_interface_it_is_the_quadrature_over_the_receiver_s_directions(
    n_source, n_receiver, cosine
):
    """Into a denser medium (a cone of light) and out of one (beyond the critical angle, nothing).

    The fluence rate both ways, and the irradiance on a surface facing the emitter, whose cosine
    weight is taken from the arrival directions the same way. Within 0.25 % at 32 cells a side; it
    converges at second order in the cell size.
    """
    depth, height, half_width, exitance = 0.05, 0.05, 0.2, 10.0
    a_source, a_receiver = 3.0, 5.0
    expected = _through_a_plane(
        n_source, n_receiver, a_source, a_receiver, depth, height, half_width, exitance, cosine
    )
    media = Media(
        n_receiver,
        (Transparent(HalfSpace([0, 0, 0], [0, 0, 1]), n_source, UniformAbsorption(a_source)),),
        UniformAbsorption(a_receiver),
    )
    triangles = _square(half_width, 32, -depth)
    surfaces = Surfaces.from_triangles(triangles).with_optics(
        emission=jnp.full(len(triangles), exitance)
    )
    receiver = np.array([[0.0, 0.0, height]])
    if cosine:
        got = float(
            refracted_irradiance(surfaces, media, receiver, np.array([[0.0, 0.0, -1.0]]))[0]
        )
    else:
        got = float(refracted_fluence_rate(surfaces, media, receiver)[0])
    assert got == pytest.approx(expected, rel=2.5e-3)


def test_through_a_flat_interface_the_error_falls_at_second_order():
    """Halving the cells quarters the error, which is what averaging over the corners buys.

    The radiance, Fresnel share and absorption are taken as the mean over a triangle's three corners;
    any one corner alone is right to first order only, and inside the tolerance above at 32 cells, so
    the order is what tells them apart.
    """
    depth, height, half_width, exitance = 0.05, 0.05, 0.2, 10.0
    expected = _through_a_plane(AIR, WATER, 3.0, 5.0, depth, height, half_width, exitance)
    media = Media(
        WATER,
        (Transparent(HalfSpace([0, 0, 0], [0, 0, 1]), AIR, UniformAbsorption(3.0)),),
        UniformAbsorption(5.0),
    )
    errors = []
    for cells in (16, 32):
        triangles = _square(half_width, cells, -depth)
        surfaces = Surfaces.from_triangles(triangles).with_optics(
            emission=jnp.full(len(triangles), exitance)
        )
        got = float(refracted_fluence_rate(surfaces, media, np.array([[0.0, 0.0, height]]))[0])
        errors.append(abs(got / expected - 1.0))
    assert errors[0] / errors[1] > 3.0


def test_a_slab_between_an_emitter_and_a_point_in_one_medium_is_the_quadrature_over_directions():
    """A thick absorbing quartz slab across all the light from a square emitter, all in water.

    The light reaches the point only by routes through the slab, at every angle up to grazing on
    the wide emitter: each direction enters and leaves the slab at the same angle, so it arrives
    parallel to how it left, shifted sideways, with the source's radiance times both crossings'
    Fresnel transmittance and the absorption along both water legs and the slab. Each of the four
    starting points of a route finds the one path there is, and it is counted once.
    """
    n_water, n_slab, a_water, a_slab = WATER, QUARTZ, 5.0, 12.0
    below, thickness, above, half_width, exitance = 0.015, 0.03, 0.055, 0.2, 10.0
    height = below + thickness + above

    def slab_angle(theta):
        return np.arcsin(n_water * np.sin(theta) / n_slab)

    def reach(theta):
        return (height - thickness) * np.tan(theta) + thickness * np.tan(slab_angle(theta))

    def carried(theta):
        inside = slab_angle(theta)
        crossings = float(fresnel_transmittance(np.cos(theta), n_water, n_slab)) * float(
            fresnel_transmittance(np.cos(inside), n_slab, n_water)
        )
        legs = a_water * (height - thickness) / np.cos(theta) + a_slab * thickness / np.cos(inside)
        return crossings * exitance / np.pi * np.exp(-legs)

    expected = _over_directions(reach, carried, np.pi / 2, half_width)
    slab = Box([0.0, 0.0, -above - thickness / 2], [1.0, 1.0, thickness / 2])
    media = Media(
        n_water,
        (Transparent(slab, n_slab, UniformAbsorption(a_slab)),),
        UniformAbsorption(a_water),
    )
    triangles = _square(half_width, 32, -height)
    surfaces = Surfaces.from_triangles(triangles).with_optics(
        emission=jnp.full(len(triangles), exitance)
    )
    got = float(refracted_fluence_rate(surfaces, media, np.zeros((1, 3)))[0])
    assert got == pytest.approx(expected, rel=2.5e-3)


def test_a_triangle_with_a_corner_that_has_no_path_carries_nothing():
    """Under the end of a short air cylinder in glass, a corner high enough has no path out.

    The path from it would leave by the side's extension beyond the flat end, so that corner is
    absent; the triangle holding it is dropped whole rather than drawn from two real arrival
    directions and the straight line the third started from. A triangle wholly below that height
    is seen, so the zero is not a receiver that sees nothing at all.
    """
    media = Media(1.5, (Transparent(Cylinder([0, 0, 0], [0, 0, 1], 0.1, 0.1), 1.0),))
    receiver = np.array([[0.3, 0.0, 0.115]])

    def triangle(top):
        corners = np.array([[[0.0, -0.005, 0.07], [0.0, 0.005, 0.07], [0.0, 0.0, top]]])
        return Surfaces.from_triangles(corners).with_optics(emission=jnp.array([10.0]))

    assert float(refracted_fluence_rate(triangle(0.085), media, receiver)[0]) > 0.0
    assert float(refracted_fluence_rate(triangle(0.095), media, receiver)[0]) == 0.0


def test_a_body_across_the_refracted_path_shadows_it_and_one_across_the_straight_line_does_not():
    """The mask follows the light, not the line from the receiver to the source.

    A small square deep under a flat interface, seen from a receiver above and to the side: the
    refracted path leaves the water steeply and the straight line shallowly, so a ball on the one
    misses the other. Each ball is checked against the straight mask too, so the test cannot pass
    by the ball missing both.
    """
    media = Media(WATER, (Transparent(HalfSpace([0, 0, 0], [0, 0, 1]), AIR),))
    triangles = _square(0.002, 2, -0.05)
    surfaces = Surfaces.from_triangles(triangles).with_optics(
        emission=jnp.full(len(triangles), 10.0)
    )
    receiver = np.array([[0.2, 0.0, 0.05]])
    clear = float(refracted_fluence_rate(surfaces, media, receiver)[0])
    assert clear > 0.0
    # Where each path crosses the plane, and a point a quarter of the way along its last leg.
    from aquaflux.radiation.refraction import Chain, solve_paths

    path = solve_paths(
        media,
        Chain.between(media.parents, 0, -1),
        jnp.zeros(3) + jnp.array([0, 0, -0.05]),
        receiver[0],
    )
    crossing = np.asarray(path.points)[0]
    on_path = crossing + 0.25 * (receiver[0] - crossing)
    straight_crossing = np.array([0.2 * 0.05 / 0.1, 0.0, 0.0])
    on_line = straight_crossing + 0.25 * (receiver[0] - straight_crossing)
    assert np.linalg.norm(on_path - on_line) > 0.01
    for centre, shadows_path in ((on_path, True), (on_line, False)):
        ball = Sphere(centre, 0.003)
        mask = build_refracted_visibility(
            [ball], surfaces, media, receiver, self_occlusion=NoOcclusion()
        )
        through = float(refracted_fluence_rate(surfaces, media, receiver, visibility=mask)[0])
        straight = build_visibility([ball], surfaces, receiver, self_occlusion=NoOcclusion())
        if shadows_path:
            assert through == 0.0
            assert not np.any(np.asarray(straight.blocked))
        else:
            assert through == pytest.approx(clear, rel=1e-14)
            assert np.all(np.asarray(straight.blocked))
        half = float(
            refracted_fluence_rate(surfaces, media, receiver, visibility=mask, transmittance=[0.5])[
                0
            ]
        )
        assert half == pytest.approx(0.5 * clear if shadows_path else clear, rel=1e-14)


def test_the_fluence_rate_s_derivative_with_respect_to_an_index_is_the_finite_difference():
    surfaces = _arc()
    points = _POINTS[:2]

    def total(quartz):
        return jnp.sum(
            refracted_fluence_rate(
                surfaces, _sleeve(quartz=quartz, absorption=UniformAbsorption(5.0)), points
            )
        )

    derivative = float(jax.grad(total)(QUARTZ))
    step = 1e-5
    finite = (float(total(QUARTZ + step)) - float(total(QUARTZ - step))) / (2 * step)
    assert derivative != 0.0
    assert derivative == pytest.approx(finite, rel=1e-5)


def test_every_route_through_a_neighbour_is_kept_its_wall_alone_by_shooting():
    """With a second sleeve the arc's light reaches the water by three routes, all solved."""
    axis = [0.0, 0.0, 1.0]
    second = Transparent(
        Cylinder([0.05, 0.0, 0.0], axis, 0.0115, 0.5),
        QUARTZ,
        inside=(Transparent(Cylinder([0.05, 0.0, 0.0], axis, 0.01025, 0.5), AIR),),
    )
    media = Media(WATER, (*_sleeve().regions, second))
    planned = refracted._plan(_arc(), media, _POINTS, pair_limit=1_000_000)
    assert [group.chain.passing for group in planned] == [(), (2,), (2, 3)]


def test_what_it_cannot_gather_is_refused():
    media = _sleeve()
    lamp = Surfaces.from_triangles(_drum(0.0075, 0.2, 6, 1), point_sources=[0])
    with pytest.raises(
        ValueError, match="a point source's light would cross the surface of a transparent region"
    ):
        refracted_fluence_rate(lamp, media, _POINTS[:1])
    straddling = Surfaces.from_triangles(
        np.array([[[0.01, 0.0, 0.0], [0.012, 0.0, 0.0], [0.011, 0.001, 0.0]]])
    )
    with pytest.raises(ValueError, match="cross the surface of a transparent region"):
        refracted_fluence_rate(straddling, media, _POINTS[:1])
    surfaces = _arc()
    straight = build_visibility([], surfaces, _POINTS, self_occlusion=NoOcclusion())
    with pytest.raises(TypeError, match="visibility must be a RefractedVisibility"):
        refracted_fluence_rate(surfaces, media, _POINTS, visibility=straight)
    with pytest.raises(ValueError, match="transmittance was given without a visibility mask"):
        refracted_fluence_rate(surfaces, media, _POINTS, transmittance=[0.5])


def test_with_every_index_equal_the_corners_absorption_converges_on_the_centroid_s():
    """In an absorbing medium the corner mean stands in for the direct gather's centroid.

    Each is a second-order estimate of the absorption across a facet, with different constants, so
    at a coarse arc they differ by a percent and the difference falls as the facets shrink; any one
    corner alone is off by a quarter at the far receiver here. Taken at that receiver, the one whose
    paths cross the most absorbing water.
    """
    absorbing = UniformAbsorption(40.0)
    gap = Transparent(Cylinder([0, 0, 0], [0, 0, 1], 0.01025, 0.5), 1.3, absorbing)
    media = Media(
        1.3,
        (Transparent(Cylinder([0, 0, 0], [0, 0, 1], 0.0115, 0.5), 1.3, absorbing, inside=(gap,)),),
        absorbing,
    )
    point = _POINTS[2:3]
    for (sectors, slices), tolerance in (((24, 8), 0.02), ((48, 16), 0.005)):
        surfaces = Surfaces.from_triangles(_drum(0.0075, 0.2, sectors, slices))
        surfaces = surfaces.with_optics(emission=jnp.full(surfaces.n_facets, 100.0))
        got = float(refracted_fluence_rate(surfaces, media, point)[0])
        direct = float(direct_fluence_rate(surfaces, point, absorption=absorbing)[0])
        assert got == pytest.approx(direct, rel=tolerance)


def _rim_scene(sectors: int, slices: int):
    """An arc in one sleeve, a second sleeve 50 mm off, and a point behind it: the rim's light."""
    axis = [0.0, 0.0, 1.0]

    def lamp(x, y):
        gap = Transparent(Cylinder([x, y, 0.0], axis, 0.01025, 1.0), AIR)
        return Transparent(Cylinder([x, y, 0.0], axis, 0.0115, 1.0), QUARTZ, inside=(gap,))

    media = Media(WATER, (lamp(0.025, -0.025), lamp(0.025, 0.025)))
    arc = _drum(0.0074, 0.02, sectors, slices) + np.array([0.025, -0.025, 0.0])
    return media, Surfaces.from_triangles(jnp.asarray(arc), emission=1.0)


def _rim_light(media, arc, points, masks=None, *, whole_facets_only=False, monkeypatch=None):
    """The light of the routes round the second sleeve's rim alone, at each point."""
    plan = refracted._plan(arc, media, points, 1_000_000)
    masks = masks or [None] * len(plan)
    if whole_facets_only:
        monkeypatch.setattr(refracted, "_shot", lambda media, chain: False)
    total = np.zeros(len(points))
    for group, mask in zip(plan, masks, strict=True):
        if group.chain.passing == (2,):
            total[group.rows] += np.asarray(
                refracted._group_field(
                    arc, media, group, jnp.asarray(points), None, mask, 1_000_000
                )
            )
    if whole_facets_only:
        monkeypatch.undo()
    return total


def test_round_a_rim_a_triangle_is_counted_for_the_part_of_it_that_has_a_path(monkeypatch):
    """Cut where the paths stop, a coarse arc gives the fine arc's light; dropped, a fraction of it.

    The light round a sleeve's rim comes from a patch of the arc a few facets across, so most of the
    patch's triangles straddle its edge. Counting only the triangles with a path at every corner
    converges on the same limit from far below; counting the part of each that has one is within a
    few percent at the coarsest arc.
    """
    point = np.array([[0.025, 0.06, 0.0]])
    fine = _rim_light(*_rim_scene(16, 8), point)[0]
    media, coarse = _rim_scene(8, 4)
    assert abs(_rim_light(media, coarse, point)[0] / fine - 1.0) < 0.05
    dropped = _rim_light(media, coarse, point, whole_facets_only=True, monkeypatch=monkeypatch)[0]
    assert dropped / fine < 0.5


def test_a_cut_triangle_takes_the_shadow_of_its_own_facet_round_its_own_side(monkeypatch):
    """A box in the middle of one side's patch takes the same light from the cut count as the whole.

    The point is off the line of the two axes, so both sides of the second sleeve carry light,
    unequally, and the box stands across the upper half of the paths past one side, clear of the
    other's and of the patch's edge: every facet it shades has a path at all three corners. So
    counting the part of each straddling triangle that has a path adds only light the box does not
    touch, and the light it takes is the whole-facet rule's to a rounding. A cut triangle given
    another side's, receiver's or facet's shadow takes a different share.
    """
    media, arc = _rim_scene(16, 8)
    point = np.array([[0.027, 0.06, 0.0]])
    box = Box([0.0204, 0.0035, 0.25], [0.003, 0.003, 0.25])
    plan = refracted._plan(arc, media, point, 1_000_000)
    visibility = build_refracted_visibility([box], arc, media, point, self_occlusion=NoOcclusion())
    masks = refracted._route_masks(media, visibility, None, point, plan)

    def taken(**whole):
        return (
            _rim_light(media, arc, point, **whole)[0]
            - _rim_light(media, arc, point, masks, **whole)[0]
        )

    by_cut = taken()
    by_whole = taken(whole_facets_only=True, monkeypatch=monkeypatch)
    assert 0.1 < by_cut / _rim_light(media, arc, point)[0] < 0.9
    assert by_cut == pytest.approx(by_whole, rel=1e-6)


def test_a_straddling_facet_is_shadowed_along_a_path_it_has_not_along_its_centroid_s():
    """Two boxes, each across every path past one side of the sleeve, take all of the light.

    A straddling facet's centroid often has no path round the rim, and the path the solve fell back
    to is a straight line that need not meet either box; the cut part of the facet would then be
    counted unshaded behind a box that stands across every path it has. Taken along a corner's path,
    each box shades its own side's light completely, so the two together take the whole.
    """
    media, arc = _rim_scene(16, 8)
    point = np.array([[0.027, 0.06, 0.0]])
    plan = refracted._plan(arc, media, point, 1_000_000)
    unshaded = _rim_light(media, arc, point)[0]
    left = []
    for box in (
        Box([0.019, 0.0035, 0.0], [0.004, 0.003, 0.5]),
        Box([0.031, 0.003, 0.0], [0.003, 0.003, 0.5]),
    ):
        visibility = build_refracted_visibility(
            [box], arc, media, point, self_occlusion=NoOcclusion()
        )
        masks = refracted._route_masks(media, visibility, None, point, plan)
        left.append(_rim_light(media, arc, point, masks)[0])
    assert all(0.0 < share < unshaded for share in left)
    assert sum(left) == pytest.approx(unshaded, rel=1e-9)


def _area_below(triangle, cut):
    """The area of the part of a flat ``(3, 3)`` triangle with ``x < cut``, clipped independently."""
    polygon = []
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
        if a[0] < cut:
            polygon.append(a)
        if (a[0] < cut) != (b[0] < cut):
            polygon.append(a + (cut - a[0]) / (b[0] - a[0]) * (b - a))
    if len(polygon) < 3:
        return 0.0
    polygon = np.asarray(polygon)
    fan = [
        np.cross(polygon[k] - polygon[0], polygon[k + 1] - polygon[0])
        for k in range(1, len(polygon) - 1)
    ]
    return 0.5 * np.linalg.norm(np.sum(fan, axis=0))


def test_a_straddling_triangle_is_cut_into_exactly_the_part_that_has_a_path(monkeypatch):
    """With paths where ``x < 0.3`` on a flat sheet, the triangles kept tile exactly that part of it.

    Path existence is stood in for, so this pins the cutting alone: each edge bisected towards the
    line, a triangle with one corner on the near side kept as one triangle, with two as two that
    together are the quadrilateral, and none kept with none. Each side of the shell is cut on its own.
    """
    cut = 0.3  # Off every edge of the sheet's lattice.
    sheet = _square(1.0, 4, 0.0)
    surfaces = Surfaces.from_triangles(jnp.asarray(sheet), emission=1.0)
    receivers = np.zeros((2, 3))

    def host_paths(media, chain, sources, points, pair_limit):
        near = np.asarray(sources)[:, 0] < cut
        shape = (2, len(points), len(sources))
        # Side 0 has paths where the near side does; side 1 nowhere.
        valid = np.zeros(shape, bool)
        valid[0] = near[None]
        return None, valid

    def shot_paths(media, chain, sources, points, sides):
        valid = (np.asarray(sources)[:, 0] < cut) & (np.asarray(sides) > 0)
        return Paths(None, None, None, valid, None)

    monkeypatch.setattr(refracted, "_host_paths", host_paths)
    monkeypatch.setattr(refracted, "_shot_paths", shot_paths)
    facets = np.arange(surfaces.n_facets)
    side, row, facet, vertices = refracted._reached_triangles(
        surfaces, None, None, facets, receivers, 1_000_000
    )
    assert np.all(side == 0)
    area = 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1
    )
    for r in range(2):
        for f in facets:
            kept = area[(row == r) & (facet == f)].sum()
            assert kept == pytest.approx(_area_below(sheet[f], cut), abs=2e-4)
