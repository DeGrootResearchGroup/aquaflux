"""The two solid-angle kernels, against references from outside their own derivation.

Each test here names the wrong answer it exists to catch. Three of them are aimed at
mistakes that are easy to make and hard to see, because the wrong result is *clean* rather
than noisy: using the unweighted solid angle where the obliquity-weighted one belongs, which
overstates every surface-to-surface transfer by exactly a factor of two; forming the contour
angle with an inverse cosine, which is wrong by about 1e-8 at every refinement; and omitting
the half-space clip, which makes a closed enclosure sum to nothing at all.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation import projected_solid_angle, signed_solid_angle, solid_angle

from tests.unit.radiation_references import (
    PYVIEWFACTOR_PARALLEL_SQUARES,
    PYVIEWFACTOR_PERPENDICULAR_SQUARES,
    axial_rectangle_solid_angle,
    box_enclosure,
    parallel_squares_view_factor,
    rectangle_triangles,
)

PI = np.pi


def _sum_over(kernel, point, vertices, normal=None):
    """Sum a kernel over a stack of triangles at one receiver."""
    point = jnp.broadcast_to(jnp.asarray(point, dtype=float), (len(vertices), 3))
    vertices = jnp.asarray(vertices, dtype=float)
    if normal is None:
        return float(jnp.sum(kernel(point, vertices)))
    normal = jnp.broadcast_to(jnp.asarray(normal, dtype=float), (len(vertices), 3))
    return float(jnp.sum(kernel(point, normal, vertices)))


def _refined_receiver(kernel, emitter, divisions, normal=None, side=1.0):
    """Area-average a kernel over a ``divisions x divisions`` receiver in the z = 0 plane.

    Every sub-receiver goes through in one batched call rather than one call each: the whole
    grid is a few thousand points, and dispatching them individually costs far more than the
    arithmetic does.
    """
    edges = np.linspace(-side / 2, side / 2, divisions + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    grid = np.stack(np.meshgrid(centres, centres, indexing="ij"), axis=-1).reshape(-1, 2)
    points = jnp.asarray(np.column_stack([grid, np.zeros(len(grid))]))[:, None, :]
    triangles = jnp.asarray(emitter, dtype=float)[None, ...]
    if normal is None:
        contributions = kernel(points, triangles)
    else:
        normals = jnp.broadcast_to(jnp.asarray(normal, dtype=float), points.shape)
        contributions = kernel(points, normals, triangles)
    return float(jnp.sum(contributions)) / len(grid)


# --------------------------------------------------------------------------------------
# The plain solid angle
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0])
def test_the_solid_angle_matches_the_closed_form_for_an_axial_rectangle(ratio):
    """Catches any constant factor, and any loss of accuracy in the near field.

    The distance sweeps two and a half decades either side of the emitter's own width. The
    near end is the point of the test: it is where the elementary inverse-square
    approximation this kernel replaces is wrong by hundreds of percent, and where an angle
    formed by summing a spherical triangle's interior angles starts to lose digits.
    """
    half = 0.5
    distance = ratio * half
    emitter = rectangle_triangles([0.0, 0.0, distance], [half, 0.0, 0.0], [0.0, half, 0.0])
    measured = _sum_over(solid_angle, [0.0, 0.0, 0.0], emitter)
    expected = axial_rectangle_solid_angle(half, half, distance)
    assert measured == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("divisions", [1, 2, 4, 8])
@pytest.mark.parametrize("point", [[0.5, 0.5, 0.5], [0.97, 0.5, 0.5], [0.999, 0.999, 0.5]])
def test_the_solid_angles_around_an_interior_point_close_the_sphere(divisions, point):
    """Catches a kernel that is right on axis and wrong off it.

    A point anywhere inside a closed surface sees exactly ``4 pi`` of it, however the surface
    is triangulated and however far off-centre the point sits. The off-centre points are the
    ones that bite: they put facets at grazing incidence and at a distance comparable to their
    own width, where an approximate kernel's errors no longer cancel by symmetry.
    """
    vertices, _, _ = box_enclosure(divisions)
    assert _sum_over(solid_angle, point, vertices) == pytest.approx(4 * PI, rel=1e-13)


# --------------------------------------------------------------------------------------
# The projected solid angle
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("divisions", [1, 2, 4, 8])
def test_the_projected_solid_angles_on_a_wall_close_the_hemisphere(divisions):
    """The single most load-bearing test in this module. It catches three separate errors.

    A point on the wall of a closed box sees the rest of the box across one hemisphere, and
    the projected solid angles must therefore sum to ``pi`` — the identity that makes every
    row of a surface-to-surface transfer matrix sum to one, which is in turn what bounds the
    spectral radius of a reflection system and makes it converge.

    Substituting the unweighted solid angle doubles this sum, exactly, at every refinement.
    Forming the contour angle as ``arccos`` of a dot product leaves it wrong by around 1e-8,
    also at every refinement — the flatness is the tell that it is a defect of the formula
    rather than an accumulation of rounding. Letting the contour sum keep its sign collapses
    it to approximately zero, because that sign records each triangle's stored vertex order
    and this fixture's faces are wound inconsistently on purpose.

    It is **not** a test of the half-space clip, which the two clip tests below cover: a
    receiver on the wall of a closed box has nothing behind it, so the clip never fires here.

    The receiver's own facet is excluded, as any caller assembling a transfer matrix must
    exclude it: a facet contains its own centroid, so the contour winds fully around the
    receiver and returns a whole extra hemisphere.
    """
    vertices, normals, centroids = box_enclosure(divisions)
    receiver = 0
    others = np.arange(len(vertices)) != receiver
    total = _sum_over(
        projected_solid_angle, centroids[receiver], vertices[others], normals[receiver]
    )
    assert total == pytest.approx(PI, rel=1e-13)


@pytest.mark.parametrize("separation", sorted(PYVIEWFACTOR_PARALLEL_SQUARES))
def test_the_projected_solid_angle_reproduces_an_independent_view_factor(separation):
    """Agreement with an implementation that shares no code and no derivation with ours.

    ``Omega_proj / pi`` at a receiver point is the differential-to-finite transfer factor, so
    averaging it over a refined receiver must converge on the area-to-area view factor. The
    reference is pyviewfactor's contour-integral result, recorded rather than imported.
    """
    emitter = rectangle_triangles([0.0, 0.0, separation], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    measured = _refined_receiver(projected_solid_angle, emitter, 48, [0.0, 0.0, 1.0]) / PI
    assert measured == pytest.approx(PYVIEWFACTOR_PARALLEL_SQUARES[separation], rel=2e-3)


def test_the_projected_solid_angle_reproduces_the_oblique_independent_view_factor():
    """The same check where the obliquity sweeps its whole range across the emitter.

    Two unit squares meeting at a right angle along a shared edge. Every on-axis test in this
    file is blind to a missing receiver cosine, because on axis the cosine is one; here it runs
    from one to zero, so an unweighted kernel cannot pass by accident.
    """
    emitter = rectangle_triangles([0.0, -0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5])
    measured = _refined_receiver(projected_solid_angle, emitter, 64, [0.0, 0.0, 1.0]) / PI
    assert measured == pytest.approx(PYVIEWFACTOR_PERPENDICULAR_SQUARES, rel=5e-3)


def test_the_unweighted_solid_angle_does_not_converge_to_a_view_factor():
    """What separates a wrong kernel from a coarse one: refinement does not help it.

    This is the discriminator that the earlier drafts of this module lacked. A quadrature
    error shrinks as the receiver is refined; the wrong integral does not. Refining the
    receiver sixteen-fold leaves the unweighted kernel stuck around 40% high, while the
    projected kernel has already converged to four figures.
    """
    emitter = rectangle_triangles([0.0, 0.0, 0.25], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    truth = parallel_squares_view_factor(1.0, 0.25)
    coarse = _refined_receiver(solid_angle, emitter, 4) / PI
    fine = _refined_receiver(solid_angle, emitter, 64) / PI
    assert abs(coarse - truth) / truth > 0.3
    assert abs(fine - truth) / truth > 0.3
    assert _refined_receiver(projected_solid_angle, emitter, 64, [0.0, 0.0, 1.0]) / PI == (
        pytest.approx(truth, rel=1e-3)
    )


# --------------------------------------------------------------------------------------
# The half-space clip
# --------------------------------------------------------------------------------------


def test_a_triangle_behind_the_receiver_contributes_nothing():
    """Catches a missing clip in the one case where its absence is a sign error, not a gap."""
    behind = rectangle_triangles([0.0, 0.0, -1.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    assert _sum_over(projected_solid_angle, [0.0, 0.0, 0.0], behind, [0.0, 0.0, 1.0]) == (
        pytest.approx(0.0, abs=1e-15)
    )


def test_a_straddling_triangle_keeps_exactly_its_visible_part():
    """Catches a clip that fires but cuts in the wrong place.

    A triangle with one vertex behind the receiver's plane is compared against the same
    visible region assembled by hand from two triangles that do not cross the plane, so the
    clip is checked against geometry rather than against itself.
    """
    apex = np.array([0.5, -0.8, -0.6])
    left = np.array([0.5, -0.8, 0.9])
    right = np.array([0.5, 0.8, 0.9])
    straddling = np.array([[apex, left, right]])

    # Where the two edges that cross the receiver's plane z = 0 meet it.
    cut_left = apex + (left - apex) * (-apex[2] / (left[2] - apex[2]))
    cut_right = apex + (right - apex) * (-apex[2] / (right[2] - apex[2]))
    by_hand = np.array([[cut_left, left, right], [cut_left, right, cut_right]])

    point, normal = [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    clipped = _sum_over(projected_solid_angle, point, straddling, normal)
    assembled = _sum_over(projected_solid_angle, point, by_hand, normal)
    assert clipped == pytest.approx(assembled, rel=1e-13)
    assert clipped > 0.0


# --------------------------------------------------------------------------------------
# Conventions the callers depend on
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "extra"), [(solid_angle, ()), (projected_solid_angle, ([0.0, 0.0, 1.0],))]
)
def test_neither_kernel_depends_on_the_order_the_vertices_are_stored_in(kernel, extra):
    """An imported surface file's winding is whatever the file says; orientation is the normal's job."""
    forward = rectangle_triangles([0.0, 0.0, 1.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    reversed_winding = forward[:, ::-1, :]
    args = (jnp.zeros(3), *(jnp.asarray(e) for e in extra))
    assert float(kernel(*args, jnp.asarray(forward[0]))) == pytest.approx(
        float(kernel(*args, jnp.asarray(reversed_winding[0]))), rel=1e-15
    )


def test_a_receiver_in_the_plane_of_a_triangle_but_outside_it_sees_nothing():
    """Coplanar facets on the same wall must not contribute, or every row sum is wrong."""
    coplanar = np.array([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]]])
    assert _sum_over(solid_angle, [0.0, 0.0, 0.0], coplanar) == pytest.approx(0.0, abs=1e-14)
    assert _sum_over(projected_solid_angle, [0.0, 0.0, 0.0], coplanar, [0.0, 0.0, 1.0]) == (
        pytest.approx(0.0, abs=1e-14)
    )


def test_a_receiver_inside_a_triangle_returns_the_oriented_area():
    """Pins the convention a caller must mask, rather than leaving it to be discovered.

    A facet's own centroid lies inside it, so the spherical image of its boundary is a great
    circle: a whole hemisphere by area. That is the documented behaviour, and it is why an
    assembled transfer matrix must zero its own diagonal — left in place it adds exactly one
    to every row sum.
    """
    triangle = jnp.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    inside = jnp.asarray([0.0, -0.25, 0.0])
    assert float(solid_angle(inside, triangle)) == pytest.approx(2 * PI, rel=1e-13)
    assert float(projected_solid_angle(inside, jnp.asarray([0.0, 0.0, 1.0]), triangle)) == (
        pytest.approx(PI, rel=1e-13)
    )


def test_a_receiver_sitting_on_a_vertex_returns_zero_rather_than_a_nan():
    """One degenerate pair must not turn a whole gather into NaN."""
    triangle = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert float(solid_angle(jnp.zeros(3), triangle)) == 0.0


def test_the_vertex_guard_is_there_for_the_GRADIENT_the_value_needs_no_help():
    """The half of that guard the value cannot show, and the half that actually matters.

    With the receiver on a vertex the numerator is zero and the denominator ``1 + b·c``, which
    for unit vectors cannot be negative — so the forward value is zero whether or not the
    degenerate entry is selected away, and a test that only reads the value passes either way.
    The derivative does not: unguarded it comes back ``(0, 0, -2)``, a finite, plausible, wrong
    sensitivity to moving the receiver, which is worse than a NaN because nothing flags it.

    This is the case the project's rule about NaNs being the floor is written for — the gather
    is differentiated with respect to vertex positions whenever a lamp moves, so a degenerate
    pair anywhere in a scene would contribute a fictitious term to that whole derivative.
    """
    triangle = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for kernel in (solid_angle, signed_solid_angle):
        gradient = jax.grad(lambda point, k=kernel: jnp.sum(k(point, triangle)))(jnp.zeros(3))
        np.testing.assert_array_equal(np.asarray(gradient), 0.0)


# --------------------------------------------------------------------------------------
# Shape and tracing contract
# --------------------------------------------------------------------------------------


def test_the_kernels_broadcast_receivers_against_triangles():
    """The gather evaluates every (receiver, emitter) pair, so the pairing must come for free."""
    points = jnp.asarray(np.random.default_rng(0).uniform(size=(5, 3)))
    vertices = jnp.asarray(box_enclosure(1)[0])
    normals = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (5, 1, 3))
    paired = solid_angle(points[:, None, :], vertices[None, ...])
    assert paired.shape == (5, len(vertices))
    assert projected_solid_angle(points[:, None, :], normals, vertices[None, ...]).shape == (
        5,
        len(vertices),
    )
    assert float(paired[2, 3]) == pytest.approx(
        float(solid_angle(points[2], vertices[3])), rel=1e-15
    )


def test_both_kernels_survive_jit_and_reverse_mode_differentiation():
    """Every consumer of these kernels is traced, and the gradient is the point of the project."""
    vertices = jnp.asarray(box_enclosure(1)[0][0])
    normal = jnp.asarray([0.0, 0.0, 1.0])

    def brightness(point):
        return solid_angle(point, vertices) + projected_solid_angle(point, normal, vertices)

    point = jnp.asarray([0.5, 0.5, 0.5])
    assert float(jax.jit(brightness)(point)) == pytest.approx(float(brightness(point)), rel=1e-15)
    gradient = jax.grad(brightness)(point)
    assert jnp.all(jnp.isfinite(gradient))
    assert float(jnp.linalg.norm(gradient)) > 0.0
