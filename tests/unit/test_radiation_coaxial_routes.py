"""The cull of routes through coaxial cylinders: it never leaves out a receiver a ray reaches."""

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import coaxial_routes
from aquaflux.radiation.coaxial_routes import CoaxialStack, kept_receivers, reachable
from aquaflux.radiation.refraction import Chain, Media, Transparent
from aquaflux.solids import Box, Cylinder

WATER, QUARTZ, AIR = 1.376, 1.5048, 1.0003
OUTER, INNER = 0.0115, 0.01025


def _refract(d, normal, n_from, n_to):
    cosine = -np.dot(d, normal)
    ratio = n_from / n_to
    k = 1.0 - ratio * ratio * (1.0 - cosine * cosine)
    return None if k < 0 else ratio * d + (ratio * cosine - np.sqrt(k)) * normal


def _meets(position, d, centre, axis, radius, inside):
    q = position - centre
    q = q - np.dot(q, axis) * axis
    e = d - np.dot(d, axis) * axis
    a, b, c = e @ e, q @ e, q @ q - radius * radius
    disc = b * b - a * c
    if disc < 0 or a == 0:
        return None
    t = (-b + np.sqrt(disc)) / a if inside else (-b - np.sqrt(disc)) / a
    return t if t > 1e-12 else None


def _outward(position, centre, axis, radius):
    q = position - centre
    return (q - np.dot(q, axis) * axis) / radius


def _through(position, d, centre, axis, radii, indices):
    """A ray traced by vector Snell into every cylinder of a stack and out again, in three
    dimensions and with no conserved quantity; ``None`` unless it enters every one and comes out."""
    here = indices[0]
    for depth, radius in enumerate(radii):
        t = _meets(position, d, centre, axis, radius, inside=False)
        if t is None:
            return None
        position = position + t * d
        d = _refract(d, _outward(position, centre, axis, radius), here, indices[depth + 1])
        if d is None:
            return None
        here = indices[depth + 1]
    if len(radii) == 1 and _meets(position, d, centre, axis, INNER, inside=False) is not None:
        return None  # it would meet the core, which a route through the wall alone misses
    for depth in reversed(range(len(radii))):
        position = position + _meets(position, d, centre, axis, radii[depth], inside=True) * d
        normal = -_outward(position, centre, axis, radii[depth])
        d = _refract(d, normal, here, indices[depth])
        if d is None:
            return None
        here = indices[depth]
    return position, d


def _stack(axis, centre, gap, shell_index=QUARTZ):
    radii = [OUTER, INNER] if gap else [OUTER]
    indices = [WATER, shell_index, AIR] if gap else [WATER, shell_index]
    return CoaxialStack(
        centre=centre,
        axis=axis,
        frame=coaxial_routes._across(axis),
        radii=np.asarray(radii),
        missed=None if gap else INNER,
        indices=np.asarray(indices),
        span=(-10.0, 10.0),
    )


def _traced_rays(count, seed):
    """Rays from a disc through a tilted stack, and receivers along each ray that leaves it."""
    rng = np.random.default_rng(seed)
    found = []
    while len(found) < count:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        gap = bool(rng.random() < 0.5)
        stack = _stack(axis, rng.normal(size=3) * 0.1, gap, 1.2 if rng.random() < 0.3 else QUARTZ)
        radius = rng.uniform(0.002, 0.012)
        # Half the discs all but touch the stack, where whether a line meets the disc before the
        # stack is decided closest.
        near = OUTER + radius + rng.uniform(0.0, 1e-3)
        distance = near if rng.random() < 0.5 else rng.uniform(0.02, 0.08)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        disc = distance * np.array([np.cos(angle), np.sin(angle)])
        turn = rng.uniform(0.0, 2.0 * np.pi)
        start = disc + radius * np.sqrt(rng.random()) * np.array([np.cos(turn), np.sin(turn)])
        across = rng.uniform(-OUTER, OUTER, size=2) - start
        across /= np.linalg.norm(across)
        axial = rng.uniform(-1.0, 1.0) * 0.999 * min(stack.indices) / WATER
        d = np.sqrt(1.0 - axial**2) * (across @ stack.frame) + axial * stack.axis
        origin = stack.centre + start @ stack.frame + rng.uniform(-0.1, 0.1) * stack.axis
        out = _through(origin, d, stack.centre, stack.axis, list(stack.radii), stack.indices)
        if out is not None:
            receivers = out[0] + rng.uniform(1e-3, 0.1, size=(4, 1)) * out[1]
            found.append((stack, disc, radius, receivers))
    return found


@pytest.mark.parametrize("splits", [coaxial_routes._SPLITS, 0])
def test_every_receiver_a_ray_through_the_stack_reaches_is_kept(monkeypatch, splits):
    """Rays traced forward in three dimensions, the stack tilted, through the wall alone and across
    the gap, through a denser shell and a less dense one: the cull keeps every receiver on them --
    with its cells split, and on its first, wide cells, where the bounds are loosest."""
    monkeypatch.setattr(coaxial_routes, "_SPLITS", splits)
    rays = _traced_rays(300, seed=1)
    assert {len(stack.radii) for stack, *_ in rays} == {1, 2}
    for stack, disc, radius, receivers in rays:
        assert reachable(stack, disc, radius, (-10.0, 10.0), receivers).all()


def test_the_bounds_on_a_cell_s_turn_hold_the_turn_of_every_ray_in_it():
    """The closed form's interval bounds, against the turn of rays traced at points of the cell."""
    rng = np.random.default_rng(2)
    for gap in (False, True):
        stack = _stack(np.array([0.0, 0.0, 1.0]), np.zeros(3), gap)
        cells = coaxial_routes._first_cells(stack)[:, 1:]
        possible, low, high = stack.turns(cells)
        inside = 0
        for cell, can, least, most in zip(cells, possible, low, high, strict=True):
            for _ in range(3):
                impact = rng.uniform(cell[0], cell[1])
                axial = rng.uniform(cell[2], cell[3])
                uz = axial / WATER
                d = np.array([np.sqrt(1.0 - uz * uz), 0.0, uz])
                out = _through(
                    np.array([-1.0, impact, 0.0]),
                    d,
                    np.zeros(3),
                    stack.axis,
                    list(stack.radii),
                    stack.indices,
                )
                if out is None:
                    continue
                inside += 1
                turn = -np.arctan2(out[1][1], out[1][0])
                assert can
                assert least - 1e-12 <= turn <= most + 1e-12
        assert inside > 100


def test_most_receivers_round_a_neighbour_are_left_out():
    """Not every receiver is kept: from a disc to one side, the ring round the stack is lit only
    where rays turned round its rim land (found here by tracing), and the cull keeps an arc a little
    wider than that and nothing on the side facing the disc."""
    stack = _stack(np.array([0.0, 0.0, 1.0]), np.zeros(3), gap=False)
    disc, _ = stack.plane(np.array([-0.05, 0.0, 0.0]))
    angles = np.radians(np.arange(-180.0, 180.0))
    ring = np.stack([0.03 * np.cos(angles), 0.03 * np.sin(angles), np.zeros(360)], axis=1)
    keep = reachable(stack, disc[0], OUTER, (-10.0, 10.0), ring)

    rng = np.random.default_rng(3)
    landed = []
    for _ in range(40000):
        turn = rng.uniform(0.0, 2.0 * np.pi)
        start = np.array([-0.05, 0.0]) + OUTER * np.sqrt(rng.random()) * np.array(
            [np.cos(turn), np.sin(turn)]
        )
        d = np.array([0.0, rng.uniform(-OUTER, OUTER)]) - start
        d /= np.linalg.norm(d)
        out = _through(
            np.array([*start, 0.0]),
            np.array([*d, 0.0]),
            np.zeros(3),
            stack.axis,
            [OUTER],
            stack.indices,
        )
        if out is not None:
            position, direction = out[0][:2], out[1][:2]
            along = position @ direction
            ahead = -along + np.sqrt(along**2 - position @ position + 0.03**2)
            hit = position + ahead * direction
            landed.append(int(np.round(np.degrees(np.arctan2(hit[1], hit[0])))))
    assert len(landed) > 100
    lit = np.isin(np.arange(-180, 180), landed)
    assert keep[lit].all()
    assert keep.sum() < 2 * lit.sum() + 30
    assert not keep[np.cos(angles) < 0.0].any()


def _two_sleeves(second=None):
    first = Transparent(
        Cylinder([0.0, 0.0, 0.0], [0, 0, 1], OUTER, 0.5),
        QUARTZ,
        inside=(Transparent(Cylinder([0.0, 0.0, 0.0], [0, 0, 1], INNER, 0.5), AIR),),
    )
    if second is None:
        second = Transparent(
            Cylinder([0.05, 0.0, 0.0], [0, 0, 1], OUTER, 0.5),
            QUARTZ,
            inside=(Transparent(Cylinder([0.05, 0.0, 0.0], [0, 0, 1], INNER, 0.5), AIR),),
        )
    return Media(WATER, (first, second))


_RING = np.stack(
    [
        0.05 + 0.03 * np.cos(np.linspace(0, 6, 40)),
        0.03 * np.sin(np.linspace(0, 6, 40)),
        np.zeros(40),
    ],
    axis=1,
)
_ARC = np.array([[[0.005, 0.0, -0.1], [0.0, 0.005, 0.0], [-0.005, 0.0, 0.1]]])


def test_a_route_from_the_gap_of_a_sleeve_leaves_out_receivers_and_its_fallbacks_keep_all():
    """Through the second sleeve from inside the first: some receivers left out. Every fallback
    keeps every receiver -- a route through nothing, the second region not a cylinder, a source
    side not a parallel cylinder, a receiver in another region after the stack, heights beyond the
    stack's ends, a traced index."""
    media = _two_sleeves()
    through = Chain.between(media.parents, 1, -1, through=2)
    assert not kept_receivers(media, through, _ARC, _RING).all()
    assert kept_receivers(media, Chain.between(media.parents, 1, -1), _ARC, _RING).all()

    boxed = _two_sleeves(Transparent(Box([0.05, 0.0, 0.0], [0.01, 0.01, 0.5]), QUARTZ))
    assert kept_receivers(boxed, Chain.between(boxed.parents, 1, -1, through=2), _ARC, _RING).all()

    leaning = Media(
        WATER,
        (
            Transparent(Cylinder([0.0, 0.0, 0.0], [0.1, 0.0, 1.0], OUTER, 0.5), QUARTZ),
            media.regions[1],
        ),
    )
    chain = Chain.between(leaning.parents, 0, -1, through=1)
    assert kept_receivers(leaning, chain, _ARC, _RING).all()

    third = Transparent(
        Cylinder([-0.05, 0.0, 0.0], [0, 0, 1], OUTER, 0.5),
        QUARTZ,
        inside=(Transparent(Cylinder([-0.05, 0.0, 0.0], [0, 0, 1], INNER, 0.5), AIR),),
    )
    three = Media(WATER, (*media.regions, third))
    into_third = Chain.between(three.parents, 1, 5, through=2)
    behind_source = np.array([[-0.05, 0.005, 0.0]])
    assert not kept_receivers(
        three,
        Chain.between(three.parents, 1, -1, through=2),
        _ARC,
        behind_source + np.array([0.0, 0.02, 0.0]),
    ).any()
    assert kept_receivers(three, into_third, _ARC, behind_source).all()

    tilted = _two_sleeves(
        Transparent(
            Cylinder([0.05, 0.0, 0.0], [0, 0, 1], OUTER, 0.5),
            QUARTZ,
            inside=(Transparent(Cylinder([0.05, 0.0, 0.0], [0.02, 0.0, 1.0], INNER, 0.5), AIR),),
        )
    )
    assert kept_receivers(tilted, through, _ARC, _RING).all()

    in_water = Chain.between(media.parents, -1, -1, through=2)
    near = np.array([[[-0.01, -0.004, 0.0], [-0.01, 0.004, 0.0], [-0.012, 0.0, 0.01]]])
    assert not kept_receivers(media, in_water, near, _RING).all()
    assert kept_receivers(media, in_water, near + np.array([0.0, 0.0, 0.6]), _RING).all()

    high = _RING + np.array([0.0, 0.0, 0.6])
    assert kept_receivers(media, through, _ARC, high).all()

    def left_out(index):
        traced = _two_sleeves(
            Transparent(
                Cylinder([0.05, 0.0, 0.0], [0, 0, 1], OUTER, 0.5),
                index,
                inside=(Transparent(Cylinder([0.05, 0.0, 0.0], [0, 0, 1], INNER, 0.5), AIR),),
            )
        )
        return jnp.sum(~kept_receivers(traced, through, _ARC, _RING))

    assert int(jax.jit(left_out)(QUARTZ)) == 0
    assert int(left_out(QUARTZ)) > 0


def test_a_source_in_the_stack_s_own_medium_is_bounded_by_its_facets_disc():
    """Sources in the water: the disc is the facets' own, so moving them moves what is reached."""
    media = _two_sleeves()
    chain = Chain.between(media.parents, -1, -1, through=2)
    near = np.array([[[-0.01, -0.004, 0.0], [-0.01, 0.004, 0.0], [-0.012, 0.0, 0.01]]])
    left = kept_receivers(media, chain, near, _RING)
    right = kept_receivers(media, chain, near + np.array([0.12, 0.0, 0.0]), _RING)
    assert not left.all() and not right.all()
    assert not np.array_equal(left, right)
    assert kept_receivers(media, chain, None, _RING).all()


@pytest.mark.parametrize("gap", [False, True])
def test_a_receiver_inside_the_stack_s_disc_is_kept(gap):
    """Above or below the stack's end a receiver can lie within its disc; that is not tested."""
    stack = _stack(np.array([0.0, 0.0, 1.0]), np.zeros(3), gap)
    point = np.array([[0.001, 0.0, 0.0]])
    assert reachable(stack, np.array([-0.05, 0.0]), OUTER, (-10.0, 10.0), point).all()


def test_the_least_and_greatest_cosine_over_an_interval_of_angles():
    """Against dense sampling, over intervals that hold a peak, a trough, both, or neither."""
    rng = np.random.default_rng(4)
    low = rng.uniform(-10.0, 10.0, 2000)
    high = low + rng.uniform(0.0, 7.0, 2000)
    least, greatest = coaxial_routes._cosine_bounds(low, high)
    sampled = np.cos(low[:, None] + (high - low)[:, None] * np.linspace(0.0, 1.0, 4001))
    np.testing.assert_allclose(least, sampled.min(axis=1), atol=1e-5)
    np.testing.assert_allclose(greatest, sampled.max(axis=1), atol=1e-5)


def test_a_ray_from_any_part_of_a_wide_source_in_the_water_is_kept():
    """A source in the stack's own medium is bounded by a disc round all its vertices: rays traced
    from each corner of a wide facet, round the second sleeve's rim, land where the cull keeps."""
    media = _two_sleeves()
    chain = Chain.between(media.parents, -1, -1, through=2)
    stack = CoaxialStack.of(media, chain)
    facet = np.array([[[-0.02, -0.06, 0.0], [-0.02, 0.06, 0.0], [-0.03, 0.0, 0.0]]])
    landed = 0
    for corner in facet[0]:
        to_axis = stack.centre - corner
        reach = np.linalg.norm(to_axis)
        towards = np.arctan2(to_axis[1], to_axis[0])
        for impact in np.linspace(-0.9999, 0.9999, 2001) * OUTER:
            angle = towards + np.arcsin(impact / reach)
            d = np.array([np.cos(angle), np.sin(angle), 0.0])
            out = _through(corner, d, stack.centre, stack.axis, list(stack.radii), stack.indices)
            if out is None:
                continue
            landed += 1
            receivers = out[0] + np.array([[0.01], [0.1], [0.3]]) * out[1]
            assert kept_receivers(media, chain, facet, receivers).all()
    assert landed > 100
