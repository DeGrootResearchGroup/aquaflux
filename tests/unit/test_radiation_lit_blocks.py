"""The gather's block layout: receivers in compact blocks, each against the facets that can light it.

The claim is that leaving a facet off a block's list changes nothing, because what is left off is
exactly zero. So every field here is checked against a reference that shares none of the layout:
the sum over every facet at every receiver, written out pair by pair in numpy. And each test also
checks that the layout *did* leave something out, since agreement reached by listing every facet
anyway would pass while saving nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import (
    BackFaces,
    CosinePower,
    Isotropic,
    Lambertian,
    NoOcclusion,
    Surfaces,
    UniformAbsorption,
    build_visibility,
    direct_fluence_rate,
)
from aquaflux.radiation import gather as gather_module
from aquaflux.radiation.lit_blocks import lit_segments, rounded_width
from aquaflux.radiation.solid_angle import solid_angle
from aquaflux.solids import Cylinder

from tests.unit.radiation_references import cylinder_triangles

A = 3.0


def _lamp(areal_profiles=None, pattern=None) -> Surfaces:
    """A tube lamp along z, wound outward, with an isotropic point source beside it.

    ``areal_profiles`` are the tube's profiles and ``pattern`` which one each tube facet takes;
    unset, the tube is Lambertian throughout.
    """
    areal_profiles = (Lambertian(),) if areal_profiles is None else tuple(areal_profiles)
    tube = cylinder_triangles(0.05, 0.3, sectors=16, slices=6)
    vertices = np.concatenate([tube, np.full((1, 3, 3), [0.0, 0.3, 0.0])])
    emission = np.concatenate([np.linspace(1.0, 2.0, len(tube)), [0.0]])
    power = np.concatenate([np.zeros(len(tube)), [0.7]])
    kinds = np.zeros(len(tube), dtype=int) if pattern is None else pattern(np.arange(len(tube)))
    return Surfaces.from_triangles(
        vertices,
        emission=emission,
        power=power,
        profiles=(*areal_profiles, Isotropic()),
        profile_index=np.concatenate([kinds, [len(areal_profiles)]]),
    )


def _receivers(count: int = 300, seed: int = 1) -> np.ndarray:
    """Points around the lamp, outside it and outside the sleeve beside it."""
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.08, 0.6, count)
    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    points = np.stack(
        [radius * np.cos(angle), radius * np.sin(angle), rng.uniform(-0.5, 0.5, count)], 1
    )
    return points[np.hypot(points[:, 0] - 0.2, points[:, 1]) > 0.04]


SLEEVE = Cylinder(centre=[0.2, 0.0, 0.0], axis=[0.0, 0.0, 1.0], radius=0.03, half_length=0.2)


def _pair_by_pair(surfaces: Surfaces, points: np.ndarray, surviving=None) -> np.ndarray:
    """The field summed over every facet at every receiver, one receiver at a time, in numpy."""
    centroid = np.asarray(surfaces.centroid)
    normal = np.asarray(surfaces.normal)
    emission, power = np.asarray(surfaces.emission), np.asarray(surfaces.power)
    index = np.asarray(surfaces.profile_index)
    point = surfaces.is_point_source
    field = np.zeros(len(points))
    for r, receiver in enumerate(points):
        offset = receiver - centroid
        distance = np.linalg.norm(offset, axis=1)
        cosine = np.einsum("fk,fk->f", offset, normal) / distance
        through = np.exp(-A * distance) * (1.0 if surviving is None else surviving[r])
        omega = np.asarray(solid_angle(jnp.asarray(receiver)[None], surfaces.vertices))
        for f in range(surfaces.n_facets):
            profile = surfaces.profiles[int(index[f])]
            if point[f]:
                fraction = float(profile.intensity_fraction_at(jnp.asarray(cosine[f])))
                field[r] += power[f] * fraction * through[f] / distance[f] ** 2
            else:
                radiance = float(profile.radiance_per_exitance_at(jnp.asarray(cosine[f])))
                field[r] += emission[f] * radiance * omega[f] * through[f]
    return field


def _listed_pairs(layout) -> int:
    return sum(
        int(np.asarray(segment.valid).sum()) * (1 if not segment.shared else len(segment.rows))
        * segment.block
        for segments in layout
        for segment in segments
    )  # fmt: skip


# --- The layout ------------------------------------------------------------------------------


def test_every_point_is_in_one_block_and_each_list_is_what_the_box_test_leaves():
    """Rows cover the points exactly once, padding is the one-past-the-end index, and a block's
    valid entries are exactly the facets whose one-facet tile the box test does not prove behind.
    """
    surfaces = _lamp()
    points = _receivers(203)
    facing = BackFaces.of(surfaces)
    facets = np.flatnonzero(~surfaces.is_point_source)
    segments = lit_segments(points, facets, facing, block=8)
    rows = np.concatenate([np.asarray(s.rows).ravel() for s in segments])
    real = rows[rows < len(points)]
    assert sorted(real.tolist()) == list(range(len(points)))
    assert set(rows[rows >= len(points)].tolist()) <= {len(points)}
    left_out = 0
    for segment in segments:
        for members, listed, valid in zip(segment.rows, segment.facets, segment.valid, strict=True):
            members = members[members < len(points)]
            if len(members) == 0:
                assert not valid.any()
                continue
            dark = facing.tiles_behind(
                points, np.repeat(members[None], len(facets), axis=0), facets[:, None]
            )
            np.testing.assert_array_equal(np.sort(listed[valid]), facets[~dark])
            left_out += int(dark.sum())
        assert segment.width == rounded_width(
            int(np.asarray(segment.valid).sum(1).max()), len(facets)
        )
    assert left_out > 0.2 * len(points) / 8 * len(facets), "the layout left out almost nothing"


def test_without_the_planes_every_block_is_listed_against_every_facet_once():
    """A full listing is one shared row, not a copy per block."""
    facets = np.arange(40)
    (segment,) = lit_segments(np.zeros((50, 3)), facets, None, block=8)
    assert segment.shared and segment.facets.shape == (1, 40) and segment.valid.all()
    assert segment.rows.shape == (7, 8)


def test_equal_segments_have_one_block_count_whatever_the_lists():
    surfaces = _lamp()
    points = _receivers(203)
    facets = np.flatnonzero(~surfaces.is_point_source)
    segments = lit_segments(points, facets, BackFaces.of(surfaces), block=8, segments=4)
    assert len({s.rows.shape[0] for s in segments}) == 1
    widths = [s.width for s in segments]
    assert widths == sorted(widths), "blocks are sorted by the length of their lists"


def test_the_width_ladder_rounds_up_by_at_most_a_step_and_stays_short():
    counts = np.arange(1, 5000)
    widths = np.array([rounded_width(k, 5000) for k in counts])
    assert np.all(widths >= counts) and np.all(widths <= 5000)
    assert np.all(widths <= np.ceil(counts * 1.25) + 1)
    assert len(np.unique(widths)) < 60


# --- The field ------------------------------------------------------------------------------


@pytest.mark.parametrize("pair_limit", [4_000_000, 97])
def test_the_field_is_the_sum_over_every_pair_with_a_body_a_medium_and_a_point_source(pair_limit):
    """Two profiles on the lamp make two groups; the sleeve blocks some pairs; the point source
    goes the dense way. The layout must still have left pairs out."""
    surfaces = _lamp((Lambertian(), CosinePower(3.0)), pattern=lambda f: (f < 48).astype(int))
    points = _receivers()
    mask = build_visibility([SLEEVE], surfaces, points, self_occlusion=NoOcclusion())
    field = direct_fluence_rate(
        surfaces, points, absorption=UniformAbsorption(A), visibility=mask,
        transmittance=[0.3], pair_limit=pair_limit,
    )  # fmt: skip
    surviving = np.asarray(mask.surviving(jnp.asarray([0.3])))
    np.testing.assert_allclose(field, _pair_by_pair(surfaces, points, surviving), rtol=1e-12)
    groups = gather_module._areal_groups((surfaces,))
    layout = gather_module.areal_layout(jnp.asarray(points), surfaces, groups)
    assert len(groups) == 2
    everything = len(points) * int((~surfaces.is_point_source).sum())
    assert _listed_pairs(layout) < 0.8 * everything


def test_a_group_whose_profile_lights_behind_itself_is_listed_in_full():
    """A profile not dark behind sends light backwards, so none of its facets may be left out."""

    class Glowing(Lambertian):
        dark_behind = False

    surfaces = _lamp((Lambertian(), Glowing()), pattern=lambda f: f % 2)
    points = jnp.asarray(_receivers(120))
    groups = gather_module._areal_groups((surfaces,))
    layout = gather_module.areal_layout(points, surfaces, groups)
    for group, segments in zip(groups, layout, strict=True):
        if group.dark_behind:
            assert not any(s.shared for s in segments)
        else:
            (segment,) = segments
            assert segment.shared and segment.width == len(group.facets)


def test_traced_receivers_are_gathered_in_full_and_give_the_same_field():
    surfaces = _lamp()
    points = jnp.asarray(_receivers(150))
    medium = UniformAbsorption(A)
    eager = direct_fluence_rate(surfaces, points, absorption=medium)
    traced = jax.jit(lambda p: direct_fluence_rate(surfaces, p, absorption=medium))(points)
    np.testing.assert_allclose(traced, eager, rtol=1e-13)


def test_a_streamed_field_is_the_held_one_and_so_are_its_gradients():
    """Streaming lays each chunk out on its own, from the curve order, in equal segments -- a
    different layout from the held gather's, so a different order of the same terms."""
    surfaces = _lamp()
    points = _receivers(260)
    mask = build_visibility([SLEEVE], surfaces, points, self_occlusion=NoOcclusion())

    def held(emission, coefficient):
        lit = surfaces.with_optics(emission=emission)
        return direct_fluence_rate(
            lit, points, absorption=UniformAbsorption(coefficient), visibility=mask,
            transmittance=[0.2],
        )  # fmt: skip

    def streamed(emission, coefficient):
        lit = surfaces.with_optics(emission=emission)
        return direct_fluence_rate(
            lit, points, absorption=UniformAbsorption(coefficient), occluders=[SLEEVE],
            self_occlusion=NoOcclusion(), transmittance=[0.2], pair_limit=40 * surfaces.n_facets,
        )  # fmt: skip

    emission, coefficient = jnp.asarray(surfaces.emission), jnp.asarray(A)
    np.testing.assert_allclose(
        streamed(emission, coefficient), held(emission, coefficient), rtol=1e-12
    )
    weights = jnp.linspace(0.5, 1.5, len(points))
    for argnum in (0, 1):
        np.testing.assert_allclose(
            jax.grad(lambda *a: jnp.sum(weights * streamed(*a)), argnums=argnum)(
                emission, coefficient
            ),
            jax.grad(lambda *a: jnp.sum(weights * held(*a)), argnums=argnum)(emission, coefficient),
            rtol=1e-11,
        )


def test_a_stream_compiles_a_few_segment_programs_not_one_per_chunk(monkeypatch):
    """A segment's shape comes from the width ladder, so many chunks share few programs."""
    traced = []
    real = gather_module._segment_fluence

    def watched(*args, **kwargs):
        traced.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gather_module, "_segment_fluence", watched)
    surfaces = _lamp()
    points = _receivers(600)
    pair_limit = 24 * surfaces.n_facets
    direct_fluence_rate(
        surfaces, points, occluders=[SLEEVE], self_occlusion=NoOcclusion(), pair_limit=pair_limit
    )
    chunks = -(-len(points) // 24)
    assert chunks * 4 > 3 * len(traced), (chunks, len(traced))


def test_points_closed_over_by_a_compiled_function_are_still_laid_out_by_their_planes(monkeypatch):
    """Inside a trace a concrete array becomes a tracer the moment it passes through ``jnp``; a
    layout formed after that conversion reads the points as traced and lists every facet, which
    gives the right field for the full price. Measured once: 2.2 times the gather's cost."""
    layouts = []
    real = gather_module.areal_layout

    def watched(*args, **kwargs):
        layouts.append(real(*args, **kwargs))
        return layouts[-1]

    monkeypatch.setattr(gather_module, "areal_layout", watched)
    surfaces = _lamp()
    points = _receivers(150)
    jax.jit(lambda: direct_fluence_rate(surfaces, points, absorption=UniformAbsorption(A)))()
    (layout,) = layouts
    assert not any(segment.shared for segments in layout for segment in segments)
    assert _listed_pairs(layout) < 0.8 * len(points) * int((~surfaces.is_point_source).sum())
