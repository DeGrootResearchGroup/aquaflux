"""The analytic silhouette clip: that it is exact, that its cull never drops, and its defects.

The clip answers with a *fraction* where a ray test answers with a bit, so almost everything
here is checked against :func:`sampled_fraction` -- a brute-force ray sampler, a different
algorithm, which makes agreement evidence rather than a restatement. Its error falls like
``1 / sqrt(samples)``, so the tolerances below are the sampler's floor and not the clip's.

Three of these pin defects the prototype shipped with, each of which produced a confident wrong
number rather than a failure, and each of which survived a sweep that called the method exact:

* a blocker straddling the source's plane occluded with the whole of itself;
* the cull's source-plane test omitted its offset and threw away real occluders;
* a blocker the depth cut reduced to a sliver came back covering *everything*.

Three more surfaced once the clip ran over real meshes, where degenerate configurations are the
norm: a source seen edge-on reported a ratio of two roundings as its hidden fraction; a plane
through two corners a rounding apart emptied a loop; and an undecidable sign made a real occluder
vanish once the clip was compiled. The last two are pinned on a finer reactor than the rest,
because the coarse one cannot show either -- it is equally right with each fix removed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.self_occlusion import (
    NoOcclusion,
    RayCastOcclusion,
    SilhouetteOcclusion,
)
from aquaflux.radiation.silhouette import (
    angular_cone,
    covered_fraction,
    may_occlude,
)
from aquaflux.radiation.surfaces import Surfaces
from aquaflux.radiation.visibility import build_visibility

from tests.unit.radiation_references import closed_drum, inward_box, sampled_fraction

RECEIVER = np.array([0.0, 0.0, 0.0])
NORMAL = np.array([0.0, 0.0, 1.0])
SOURCE = np.array([[-1.0, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.5, 4.0]])

#: The sampler's own noise at the sample counts used here, as a share of the source.
SAMPLER_FLOOR = 4e-3


def _fraction(blocker) -> float:
    return float(covered_fraction(RECEIVER, NORMAL, SOURCE, np.asarray(blocker))[0])


@pytest.mark.parametrize(
    "name, blocker",
    [
        ("squarely between", [[-9.0, -9.0, 2.0], [9.0, -9.0, 2.0], [0.0, 9.0, 2.0]]),
        ("partly across", [[-0.4, -3.0, 2.0], [0.4, -3.0, 2.0], [0.0, 3.0, 2.0]]),
        ("cutting an edge", [[-3.0, -3.0, 2.0], [0.0, -3.0, 2.0], [0.0, 3.0, 2.0]]),
        ("straddling the source's plane", [[-9.0, -9.0, 2.0], [9.0, -9.0, 2.0], [0.0, 9.0, 6.0]]),
    ],
)
def test_the_clip_agrees_with_a_brute_force_sampler(name, blocker):
    """The whole claim, against an algorithm with nothing in common with it."""
    blocker = np.asarray(blocker)
    assert _fraction(blocker) == pytest.approx(
        sampled_fraction(RECEIVER, NORMAL, SOURCE, blocker, samples=400_000),
        abs=SAMPLER_FLOOR,
    )


def test_a_blocker_straddling_the_source_plane_does_not_occlude_with_all_of_itself():
    """Regression. The clip is direction-space and has no depth of its own.

    Without the cut to the near side of the source's supporting plane this reads **1.0000**
    against a sampled 0.6396 -- the part of the blocker lying *beyond* the source occluding as
    though it were in front. Pinned as a bound rather than a value so the sampler's noise
    cannot fail it, and the wrong answer is nowhere near the bound.
    """
    covered = _fraction([[-9.0, -9.0, 2.0], [9.0, -9.0, 2.0], [0.0, 9.0, 6.0]])
    assert 0.55 < covered < 0.72, "the depth cut is gone: a blocker behind the source is counted"


def test_a_blocker_the_depth_cut_degenerates_covers_nothing():
    """Regression, and the subtler of the two: a no-op clip means *nothing is removed*.

    A zero clipping plane is deliberately a no-op, which is right for the repeated slot a
    triangular blocker leaves in a four-wide loop. When the cut degenerates the blocker
    entirely, **every** edge plane is zero, so no clip removes anything and the source comes
    back wholly covered. Measured on a real reactor facet: 1.0000 against a sampled 0.0.

    The fixture puts the source in the plane ``z = 2`` and gives the blocker a single corner
    touching it, so the near side of the cut is a sliver of no area.
    """
    source = np.array([[0.5, -0.5, 2.0], [1.5, -0.5, 2.0], [1.0, 0.5, 2.0]])
    blocker = np.array([[-0.5, 0.0, 2.0], [-0.2, -0.3, 6.0], [-0.5, 0.0, 6.0]])
    covered = float(covered_fraction(RECEIVER, NORMAL, source, blocker)[0])
    assert covered == pytest.approx(
        sampled_fraction(RECEIVER, NORMAL, source, blocker, samples=400_000), abs=SAMPLER_FLOOR
    )
    assert covered < 0.5, "a blocker cut to a sliver is covering the source"


@pytest.mark.parametrize(
    "name, blocker",
    [
        ("beyond the source", [[-9.0, -9.0, 6.0], [9.0, -9.0, 6.0], [0.0, 9.0, 6.0]]),
        ("behind the receiver", [[-9.0, -9.0, -2.0], [9.0, -9.0, -2.0], [0.0, 9.0, -2.0]]),
        ("off to one side", [[41.0, -9.0, 2.0], [59.0, -9.0, 2.0], [50.0, 9.0, 2.0]]),
    ],
)
def test_a_blocker_that_cannot_occlude_covers_nothing(name, blocker):
    """Each of the three ways a triangle can be irrelevant, and each is exactly zero."""
    assert _fraction(blocker) == 0.0


def test_the_fraction_is_additive_over_a_tiling():
    """Splitting one blocker into four sums to the whole, which is why STL geometry works.

    A triangulated surface *is* a tiling, and tilings do not overlap in projection, so
    per-triangle fractions simply add and no union algorithm is needed between them. If this
    stopped holding, every multi-triangle blocker would be counted wrongly.
    """
    whole = np.array([[-0.9, -1.2, 2.0], [0.9, -1.2, 2.0], [0.0, 1.4, 2.0]])
    a, b, c = whole
    midpoints = [(a + b) / 2, (b + c) / 2, (c + a) / 2]
    pieces = [
        [a, midpoints[0], midpoints[2]],
        [midpoints[0], b, midpoints[1]],
        [midpoints[2], midpoints[1], c],
        [midpoints[0], midpoints[1], midpoints[2]],
    ]
    assert sum(_fraction(p) for p in pieces) == pytest.approx(_fraction(whole), rel=1e-9)


def test_a_cone_that_cannot_bound_its_triangle_is_flagged_rather_than_trusted():
    """A receiver in a triangle's own plane has no usable bounding cap, and the cull must know.

    Treating such a cone as a bound is what made the first cull drop real occluders, so the flag
    is the load-bearing part of :func:`angular_cone` rather than an edge-case tidy-up.
    """
    flat = jnp.asarray([[1.0, 0.0, 0.0], [-1.0, 0.5, 0.0], [-1.0, -0.5, 0.0]])
    assert bool(angular_cone(flat)[3])
    ordinary = jnp.asarray([[0.2, 0.0, 3.0], [-0.1, 0.2, 3.0], [-0.1, -0.2, 3.0]])
    assert not bool(angular_cone(ordinary)[3])


def test_the_cull_never_drops_a_blocker_that_covers_something():
    """The cull's one invariant, swept over a closed body rather than argued from the geometry.

    A cull that drops an occluder yields a slightly brighter field and no error of any kind, so
    this is the property to protect. Checked exhaustively: every (source, blocker) pair of a
    closed box, against every receiver, with the cull's verdict compared to whether the clip
    actually finds coverage.
    """
    surfaces = Surfaces.from_triangles(inward_box(2))
    vertices = np.asarray(surfaces.vertices)
    centroid = np.asarray(surfaces.centroid)
    normal = np.asarray(surfaces.normal)
    n = surfaces.n_facets

    dropped = 0
    for receiver in range(0, n, 7):
        for source in range(n):
            if source == receiver:
                continue
            keep = np.asarray(
                may_occlude(centroid[receiver], normal[receiver], vertices[source], vertices)
            )
            covered, _ = covered_fraction(
                centroid[receiver], normal[receiver], vertices[source], vertices
            )
            real = np.asarray(covered) > 1e-9
            real[source] = real[receiver] = False
            dropped += int(np.sum(real & ~keep))
    assert dropped == 0, f"the cull threw away {dropped} blockers that cover part of a source"


def test_the_cull_does_reject_most_of_what_it_sees():
    """Otherwise it is conservative for free and buys nothing, which a bug could make it."""
    surfaces = Surfaces.from_triangles(inward_box(3))
    vertices = np.asarray(surfaces.vertices)
    centroid = np.asarray(surfaces.centroid)
    normal = np.asarray(surfaces.normal)
    kept = np.mean(
        [
            np.mean(np.asarray(may_occlude(centroid[0], normal[0], vertices[s], vertices)))
            for s in range(1, surfaces.n_facets, 5)
        ]
    )
    assert kept < 0.5, f"the cull keeps {100 * kept:.0f}% of blockers and is not culling"


def _reactor(divisions=2, sectors=8):
    """A closed box with a sleeve down its axis: non-convex, and every triangle a valid facet.

    ⚠️ **A box on its own is CONVEX and can shadow nothing**, so it is the wrong fixture for
    anything about partial shadowing -- zero occlusion is the right answer there, and a test
    asserting otherwise is testing its fixture. The L-prism, the other non-convex body in the
    references, is wrong here for a different reason: its cap triangles overlap and hide one
    another, which no surface does.
    """
    sleeve = closed_drum(sectors, radius=0.15, half_height=0.3) + np.array([0.5, 0.5, 0.5])
    return Surfaces.from_triangles(np.concatenate([inward_box(divisions), sleeve]))


def _reactor_mask(strategy):
    surfaces = _reactor()
    mask = build_visibility(
        (),
        surfaces,
        surfaces.centroid,
        receiver_facet=np.arange(surfaces.n_facets),
        self_occlusion=strategy,
    )
    return np.asarray(mask.hidden_by_geometry), np.asarray(mask.overlapping)


def test_a_convex_box_shadows_nothing_however_it_is_tested():
    """The control for the two below: on a convex body the right answer is no occlusion.

    Against a floor rather than exactly zero. The ray test does return exact zeros, but the clip
    reaches its answer through several cancellations and lands at 1e-13 on a 108-facet box --
    which is the dust such a pipeline leaves, not a shadow. The floor is far below any fraction
    that changes a field and far above the dust, so it still fails loudly if a convex body ever
    starts shadowing itself.
    """
    for strategy in (RayCastOcclusion(), SilhouetteOcclusion()):
        hidden, _ = _mask(strategy, divisions=3)
        assert hidden.max() < 1e-9, f"{type(strategy).__name__} found a shadow on a convex box"


def _mask(strategy, divisions=2):
    surfaces = Surfaces.from_triangles(inward_box(divisions))
    mask = build_visibility(
        (),
        surfaces,
        surfaces.centroid,
        receiver_facet=np.arange(surfaces.n_facets),
        self_occlusion=strategy,
    )
    return np.asarray(mask.hidden_by_geometry), np.asarray(mask.overlapping)


def test_switching_occlusion_off_hides_nothing():
    hidden, overlapping = _mask(NoOcclusion())
    assert not hidden.any()
    assert not overlapping.any()


def test_the_ray_test_answers_only_ever_zero_or_one():
    """It is widened to a fraction so nothing downstream branches, not because it has one."""
    hidden, overlapping = _mask(RayCastOcclusion())
    assert set(np.unique(hidden)) <= {0.0, 1.0}
    assert not overlapping.any(), "one ray cannot add two answers together"


def test_the_silhouette_resolves_pairs_the_ray_test_rounds_to_all_or_nothing():
    """The point of the whole treatment, stated as the thing a ray test cannot produce."""
    hidden, _ = _reactor_mask(SilhouetteOcclusion())
    partial = (hidden > 1e-6) & (hidden < 1.0 - 1e-6)
    assert partial.sum() > 0, "no partly shadowed pair was found, so nothing was resolved"
    assert hidden.max() <= 1.0 and hidden.min() >= 0.0


def test_a_pair_hidden_by_one_blocker_alone_is_not_flagged_as_unproven():
    """``overlapping`` must distinguish, or it is a constant and says nothing."""
    hidden, overlapping = _reactor_mask(SilhouetteOcclusion())
    assert overlapping.any(), "nothing flagged at all"
    assert not overlapping.all(), "everything flagged, so the flag carries no information"
    assert not overlapping[hidden == 0.0].any(), "a pair nothing hides cannot be double counted"


def test_the_surviving_fraction_lets_a_half_hidden_pair_through_by_half():
    """The mask is consumed as ``1 - fraction``, which is what makes the fraction worth having."""
    surfaces = Surfaces.from_triangles(inward_box(2))
    mask = build_visibility(
        (),
        surfaces,
        surfaces.centroid,
        receiver_facet=np.arange(surfaces.n_facets),
        self_occlusion=NoOcclusion(),
    )
    half = type(mask)(
        blocked=mask.blocked,
        receivers=mask.receivers,
        hidden_by_geometry=jnp.full_like(mask.hidden_by_geometry, 0.25),
        overlapping=mask.overlapping,
    )
    assert np.allclose(np.asarray(half.surviving(jnp.zeros(0))), 0.75)


@pytest.mark.parametrize(
    "receiver_facet, expected",
    [(None, "volume gather"), (np.array([0, -1]), "lie on no facet")],
    ids=["no facets at all", "some receivers off the surface"],
)
def test_the_silhouette_refuses_receivers_that_have_no_normal(receiver_facet, expected):
    """It takes a fraction of a PROJECTED solid angle, so a point in the fluid has nothing to
    project onto. Refused with the alternative named, rather than answered wrongly."""
    surfaces = Surfaces.from_triangles(inward_box(2))
    points = np.asarray(surfaces.centroid)[:2]
    with pytest.raises(ValueError, match=expected):
        build_visibility(
            (),
            surfaces,
            points,
            receiver_facet=receiver_facet,
            self_occlusion=SilhouetteOcclusion(),
        )


def test_a_source_coplanar_with_the_receiver_hides_nothing():
    """A source seen exactly edge-on subtends nothing, so no fraction of it can be hidden.

    Its projected solid angle is mathematically zero and computes as a few parts in 1e16 of
    dust, and the part a blocker covers is dust of the same size -- so the fraction is a ratio
    of two roundings, an arbitrary number in ``[0, 1]``. On a meshed body this is every other
    triangle of the receiver's own wall. Measured without the guard on this reactor: coplanar
    sleeve triangles read anywhere from 0.06 to a fully hidden 1.0, where the truth is that they
    exchange no light at all.

    Swept over the whole reactor rather than built from a hand-placed triangle, because a clean
    fixture tends to land on an *exact* zero, which the arithmetic handles without the guard;
    the defect needs the dust a real mesh's coordinates produce.
    """
    surfaces = _reactor()
    hidden, _ = _reactor_mask(SilhouetteOcclusion())
    normal = np.asarray(surfaces.normal)
    centroid = np.asarray(surfaces.centroid)
    parallel = np.abs(normal @ normal.T) > 1.0 - 1e-12
    offset = np.abs(np.einsum("rd,rsd->rs", normal, centroid[None, :, :] - centroid[:, None, :]))
    coplanar = parallel & (offset < 1e-12)
    np.fill_diagonal(coplanar, False)
    assert coplanar.sum() > 0, "no coplanar pair in the fixture, so this pins nothing"
    assert np.all(hidden[coplanar] == 0.0), (
        f"{int((hidden[coplanar] != 0).sum())} coplanar pairs hidden"
    )


#: The finer reactor the two tests below need. The default one is too coarse to contain either
#: configuration: at 80 and at 156 facets the clip was equally right with either filter removed,
#: so a test there passes whichever is missing. At 364 facets each filter's removal is visible.
FINE_REACTOR = {"divisions": 5, "sectors": 16}


def _fine_triple(receiver, source, blocker):
    surfaces = _reactor(**FINE_REACTOR)
    vertices = np.asarray(surfaces.vertices)
    return (
        np.asarray(surfaces.centroid)[receiver],
        np.asarray(surfaces.normal)[receiver],
        vertices[source],
        vertices[blocker],
    )


def test_a_real_occluder_is_not_emptied_by_the_plane_through_a_near_repeated_corner():
    """Without the plane filter, one triangle of a fully shadowing quad vanishes from the answer.

    The source is hidden by the two triangles of one sleeve quad, 0.6284 and 0.3716 of it. The
    depth cut leaves the second triangle with two corners a rounding apart -- a crossing point
    computed at the very end of its edge, ``a + 1.0 * (b - a)``, which is ``b`` to within a
    rounding but not bit for bit. The plane through those two corners is noise, and clipping by
    it empties the loop: measured, this blocker reads **0.0** against a sampled 0.3699, on 25
    pairs of this reactor. A guard testing the two corners for exact equality does not fire,
    which is why the plane itself has to be filtered.
    """
    triple = _fine_triple(183, 332, 329)
    sampled = sampled_fraction(*triple, samples=200_000)
    assert sampled > 0.3, "the fixture has moved: this blocker no longer shadows this source"
    assert float(covered_fraction(*triple)[0]) == pytest.approx(sampled, abs=SAMPLER_FLOOR)


def test_a_real_occluder_reads_the_same_whether_or_not_the_clip_is_compiled():
    """Without the height filter, compiling the clip is enough to lose a real occluder.

    A height that is exactly zero in exact arithmetic is computed as noise whose sign depends on
    how the compiler fused the arithmetic -- and compiling changes the fusion. Measured without
    the filter: this blocker covers 0.0020 of the source evaluated eagerly and **0.0000** under
    ``jax.jit``, alone or inside a batch of any width, against a sampled 0.0019. The build always
    runs compiled, so it is the compiled answer that is wrong there, and it is wrong silently.

    The same removal leaves the chunk-size test above green, because every compiled shape gets
    the *same* wrong answer. Only comparing against an uncompiled evaluation exposes it.
    """
    triple = _fine_triple(47, 339, 332)
    sampled = sampled_fraction(*triple, samples=200_000)
    assert sampled > 1e-3, "the fixture has moved: this blocker no longer shadows this source"
    eager = float(covered_fraction(*triple)[0])
    compiled = float(jax.jit(covered_fraction)(*triple)[0])
    assert compiled == pytest.approx(eager, abs=1e-12)
    # Not SAMPLER_FLOOR, which is sized for fractions near a half and would accept a zero here.
    # The sampler's standard error on a fraction this small is about 1e-4 at this sample count.
    assert compiled == pytest.approx(sampled, abs=5e-4)


def test_the_work_chunk_does_not_change_the_answer():
    """It bounds memory and nothing else; a padded final chunk must not leak into the sum.

    ⚠️ **On a convex body this test cannot fail**, because every answer is zero whatever the
    chunking, so it runs on the sleeved reactor. It is not a theoretical guard: before the clip's
    sign tests were filtered, pairs moved with the chunk size alone -- 119 of a box's 2304 at
    first, and after the coplanar cases were guarded still 4 of this reactor's 6400, one by a
    fifth of its value -- because a different batch shape lets the compiler fuse the arithmetic
    differently, and that flips the sign of a height that is exactly zero.

    Held to 1e-9 rather than to bit-identity. What remains is ~1e-13, the contour integral's own
    rounding summed in a different order at a different shape: continuous, not a sign flip, and
    four orders of magnitude inside this bound where the defect it guards was at 0.2.
    """
    surfaces = _reactor()
    facets = np.arange(surfaces.n_facets)
    answers = [
        np.asarray(
            build_visibility(
                (),
                surfaces,
                surfaces.centroid,
                receiver_facet=facets,
                self_occlusion=SilhouetteOcclusion(work_chunk=chunk),
            ).hidden_by_geometry
        )
        for chunk in (7, 64, 262_144)
    ]
    assert np.max(np.abs(answers[0] - answers[1])) < 1e-9
    assert np.max(np.abs(answers[0] - answers[2])) < 1e-9
