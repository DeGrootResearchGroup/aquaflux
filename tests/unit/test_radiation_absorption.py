"""Optical depth: the closed-form case, and the graded field walked cell by cell."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.radiation.absorption import UniformAbsorption, VoxelAbsorption, grid_covers

DIMS = (6, 5, 4)
SPACING = np.array([0.3, 0.4, 0.25])
ORIGIN = np.array([-0.5, -0.2, 0.1])

#: Segments chosen to exercise every branch of the traversal: wholly inside the sample hull,
#: straddling it, crossing the whole grid diagonally from far outside, axis-aligned, running
#: backwards, and lying inside a single cell.
SEGMENTS = [
    ((0.0, 0.1, 0.2), (1.1, 1.5, 0.9)),
    ((-0.3, 0.0, 0.12), (1.3, 1.8, 1.1)),
    ((-2.0, -2.0, -2.0), (3.0, 3.0, 3.0)),
    ((-5.0, 0.5, 0.5), (5.0, 0.5, 0.5)),
    ((0.4, 0.4, 0.4), (0.4, 0.4, 0.9)),
    ((1.2, 1.7, 1.0), (0.05, 0.05, 0.15)),
]


def graded_field(seed: int = 0) -> VoxelAbsorption:
    rng = np.random.default_rng(seed)
    return VoxelAbsorption(rng.uniform(0.5, 4.0, DIMS), ORIGIN, SPACING)


def quadrature_reference(field: VoxelAbsorption, start, finish, samples: int = 400001) -> float:
    """Integrate the field's own interpolant along the segment, finely.

    The reference is the *interpolant*, not the underlying function: what the traversal claims
    is that it integrates the interpolated field exactly, and that claim is what this checks.
    """
    start, finish = np.asarray(start, dtype=float), np.asarray(finish, dtype=float)
    parameter = np.linspace(0.0, 1.0, samples)
    positions = start + (finish - start) * parameter[:, None]
    values = np.asarray(field.sample(jnp.asarray(positions)))
    return float(np.trapezoid(values, parameter) * np.linalg.norm(finish - start))


# ---------------------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("coefficient", [0.0, 0.7, 35.67])
def test_a_uniform_medium_gives_the_coefficient_times_the_distance(coefficient):
    start, finish = jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([4.0, 6.0, 3.0])
    measured = float(UniformAbsorption(coefficient).optical_depth(start, finish))
    assert measured == pytest.approx(coefficient * 5.0, rel=1e-15)


def test_a_uniform_medium_is_differentiable_in_its_coefficient():
    """A water-quality sensitivity is a derivative with respect to exactly this number."""
    gradient = jax.grad(
        lambda a: UniformAbsorption(a).optical_depth(jnp.zeros(3), jnp.asarray([3.0, 4.0, 0.0]))
    )(jnp.asarray(0.5))
    assert float(gradient) == pytest.approx(5.0, rel=1e-14)


# ---------------------------------------------------------------------------------------
# The graded field
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("start", "finish"), SEGMENTS)
def test_a_constant_graded_field_reproduces_the_closed_form_exactly(start, finish):
    """Two strategies, one answer. The cheap one is the reference for the expensive one.

    A graded field whose values happen to be equal is the closed-form case, so any bookkeeping
    error in the traversal — a dropped piece, a double-counted one, a segment that leaves the
    grid and loses its tail — shows up as a discrepancy against a formula.
    """
    coefficient = 2.0
    field = VoxelAbsorption(np.full(DIMS, coefficient), ORIGIN, SPACING)
    closed_form = UniformAbsorption(coefficient)
    start, finish = jnp.asarray(start), jnp.asarray(finish)
    assert float(field.optical_depth(start, finish)) == pytest.approx(
        float(closed_form.optical_depth(start, finish)), rel=1e-13
    )


@pytest.mark.parametrize(("start", "finish"), SEGMENTS)
def test_the_traversal_integrates_the_interpolated_field_exactly(start, finish):
    """Not approximately: exactly, to the reference quadrature's own accuracy.

    A trilinear field restricted to a straight line is a cubic in the path parameter, and
    Simpson's rule integrates cubics exactly — provided each piece lies within one patch of the
    interpolant, which is what cutting the segment on the sample planes achieves.
    """
    field = graded_field()
    measured = float(field.optical_depth(jnp.asarray(start), jnp.asarray(finish)))
    assert measured == pytest.approx(quadrature_reference(field, start, finish), rel=1e-9)


def test_a_segment_far_outside_the_grid_keeps_its_whole_path():
    """The step budget is fixed, so a long segment is where it can quietly run out.

    Cuts are generated only at planes that carry samples. Generating them on the infinite
    lattice instead would spend the budget outside the grid and drop the segment's tail —
    which reads as a weakly absorbing medium rather than as a bug.
    """
    field = VoxelAbsorption(np.full(DIMS, 2.0), ORIGIN, SPACING)
    start, finish = jnp.asarray([-9.0, -9.0, -9.0]), jnp.asarray([9.0, 9.0, 9.0])
    expected = 2.0 * float(jnp.linalg.norm(finish - start))
    assert float(field.optical_depth(start, finish)) == pytest.approx(expected, rel=1e-13)


def test_the_field_is_interpolated_and_not_sampled_at_the_nearest_cell():
    """A nearest-cell lookup is piecewise constant, so it has no useful derivative in position
    and puts a staircase in a path the module promises gradients through."""
    field = graded_field()
    offsets = np.linspace(0.0, float(SPACING[0]), 9)
    probes = jnp.asarray([[0.4 + d, 0.4, 0.4] for d in offsets])
    values = np.asarray(field.sample(probes))
    assert len(np.unique(np.round(values, 12))) == len(offsets)


def test_a_linear_field_is_interpolated_without_error_inside_the_samples():
    """Trilinear interpolation reproduces a linear function exactly, which is the property the
    exactness of the traversal rests on."""
    indices = np.meshgrid(*[np.arange(n) for n in DIMS], indexing="ij")
    centres = ORIGIN + SPACING * (np.stack(indices, axis=-1) + 0.5)
    linear = 1.0 + 2.0 * centres[..., 0] - 0.5 * centres[..., 1] + 0.75 * centres[..., 2]
    field = VoxelAbsorption(linear, ORIGIN, SPACING)
    inside = jnp.asarray([[0.2, 0.3, 0.4], [0.9, 1.1, 0.8]])
    expected = 1.0 + 2.0 * inside[:, 0] - 0.5 * inside[:, 1] + 0.75 * inside[:, 2]
    np.testing.assert_allclose(np.asarray(field.sample(inside)), np.asarray(expected), rtol=1e-13)


def test_outside_the_samples_the_field_takes_its_boundary_value():
    """Clamped rather than zero: a segment straying past the edge attenuates like the water at
    the edge, not like vacuum."""
    field = graded_field()
    edge = ORIGIN + SPACING * 0.5
    np.testing.assert_allclose(
        np.asarray(field.sample(jnp.asarray(edge - np.array([5.0, 5.0, 5.0])))),
        np.asarray(field.coefficient[0, 0, 0]),
        rtol=1e-14,
    )


def test_the_step_budget_is_computed_from_the_shape_alone():
    field = graded_field()
    assert field.max_crossings == sum(DIMS) + 1
    assert field.dims == DIMS


def test_a_zero_length_segment_has_no_optical_depth():
    field = graded_field()
    point = jnp.asarray([0.4, 0.4, 0.4])
    assert float(field.optical_depth(point, point)) == pytest.approx(0.0, abs=1e-15)


def test_the_optical_depth_is_differentiable_in_the_absorbance_field():
    """The gradient that makes a water-quality inverse problem possible, cell by cell."""
    field = graded_field()
    start, finish = jnp.asarray([0.0, 0.1, 0.2]), jnp.asarray([1.1, 1.5, 0.9])

    def depth(coefficient):
        return VoxelAbsorption(coefficient, ORIGIN, SPACING).optical_depth(start, finish)

    jacobian = jax.grad(depth)(field.coefficient)
    assert jacobian.shape == DIMS
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert float(jnp.sum(jacobian)) > 0.0

    step, cell = 1e-6, (2, 2, 1)
    bumped = field.coefficient.at[cell].add(step)
    lowered = field.coefficient.at[cell].add(-step)
    finite_difference = (float(depth(bumped)) - float(depth(lowered))) / (2.0 * step)
    assert float(jacobian[cell]) == pytest.approx(finite_difference, rel=1e-6)


def _walk_working_memory(n: int, pairs: int = 512):
    """Compiled working bytes per segment of the walk on an ``n``-cube grid, forward and reverse."""
    rng = np.random.default_rng(4)
    origin = jnp.asarray(rng.uniform(0.0, 1.0, (pairs, 3)))
    target = jnp.asarray(rng.uniform(0.0, 1.0, (pairs, 3)))
    coefficient = jnp.asarray(rng.uniform(0.5, 4.0, (n, n, n)))

    def depth(values):
        return VoxelAbsorption(values, 0.0, 1.0 / n).optical_depth(origin, target)

    def per_pair(function):
        compiled = jax.jit(function).lower(coefficient).compile()
        return compiled.memory_analysis().temp_size_in_bytes / pairs

    return per_pair(depth), per_pair(jax.grad(lambda values: jnp.sum(depth(values))))


def test_the_walk_s_working_memory_does_not_grow_with_the_number_of_cells_it_crosses():
    """Forward, a segment's working set is its carry, whatever the grid; reverse, a carry a step.

    The walk runs a fixed ``nx + ny + nz + 1`` steps. Collected as a stacked array of pieces and
    summed afterwards, its forward memory grew by eight bytes a segment per step -- about 1 kB a
    segment on a 32-cube grid, 4 GB in one of the gather's 4M-pair chunks; carried as a running
    total it is flat. Under a gradient every step must keep something for the way back, so that
    grows with the steps regardless; kept as the carry alone it is tens of bytes a step, against
    the ~500 each step cost when every lookup's indices and weights were kept instead.
    """
    coarse_forward, _ = _walk_working_memory(4)
    fine_forward, fine_reverse = _walk_working_memory(32)
    assert fine_forward <= 1.1 * coarse_forward, (coarse_forward, fine_forward)
    steps = 3 * 32 + 1
    assert fine_reverse / steps < 100.0, fine_reverse / steps


def test_the_optical_depth_is_differentiable_in_the_segment_endpoints():
    """Moving a source changes how much water its light crosses, smoothly."""
    field = graded_field()
    finish = jnp.asarray([1.1, 1.5, 0.9])

    def depth(shift):
        return field.optical_depth(jnp.asarray([0.0, 0.1, 0.2]) + shift, finish)

    direction = jnp.asarray([0.0, 0.0, 1.0])
    step = 1e-7
    finite_difference = (float(depth(direction * step)) - float(depth(-direction * step))) / (
        2.0 * step
    )
    gradient = jax.grad(lambda t: depth(direction * t))(jnp.asarray(0.0))
    assert float(gradient) == pytest.approx(finite_difference, rel=1e-6)


def test_a_huge_optical_depth_gives_zero_transmittance_rather_than_a_nan():
    """No clamp: in double precision the exponential reaches zero where zero is correct, and a
    clamp would only flatten the sensitivity to absorbance across a whole region."""
    depth = UniformAbsorption(1000.0).optical_depth(jnp.zeros(3), jnp.asarray([0.0, 0.0, 1.0]))
    assert float(jnp.exp(-depth)) == 0.0
    assert bool(jnp.isfinite(depth))


def test_the_traversal_compiles_and_batches_over_many_segments():
    """Closed over rather than passed: a field is geometry, and its grid shape sets the step
    count, so it belongs on the built side of the same boundary the gather draws."""
    field = graded_field()
    rng = np.random.default_rng(1)
    starts = jnp.asarray(rng.uniform(-1.0, 2.0, (40, 7, 3)))
    finishes = jnp.asarray(rng.uniform(-1.0, 2.0, (40, 7, 3)))
    batched = jax.jit(lambda a, b: field.optical_depth(a, b))(starts, finishes)
    assert batched.shape == (40, 7)
    assert float(field.optical_depth(starts[3, 5], finishes[3, 5])) == pytest.approx(
        float(batched[3, 5]), rel=1e-13
    )


def test_a_grid_that_does_not_contain_the_geometry_can_be_detected():
    field = graded_field()
    assert grid_covers(field, [[0.0, 0.0, 0.5], [1.0, 1.0, 1.0]])
    assert not grid_covers(field, [[0.0, 0.0, 0.5], [9.0, 0.0, 0.5]])


def test_a_coefficient_that_is_not_a_grid_is_refused():
    with pytest.raises(ValueError, match="three-dimensional grid"):
        VoxelAbsorption(np.ones((4, 4)), ORIGIN, SPACING)
