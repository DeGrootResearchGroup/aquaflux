"""The block-scaled residual norm and its effect on the line search.

``BlockScaledNorm`` scales each contiguous block of a residual by its own reference magnitude before
combining, so a heterogeneous block system (a coupled RANS state whose ``omega`` residual is O(1e5)
and ``k`` residual O(1e-3)) is judged on every block rather than the one with the largest magnitude.
These tests pin the norm's arithmetic and, behaviourally, that it makes the shared backtracking line
search accept a step the plain Euclidean norm would reject — the globalization fix for the coupled
march, exercised here with no mesh or flow.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import BlockScaledNorm, RowScaledNorm
from aquaflux.solve.implicit import backtracking_line_search


def test_block_scaled_norm_is_the_l2_of_per_block_relative_norms():
    # Block 0 (size 2) has norm 5 and scale 1 -> relative 5; block 1 (size 3) has norm 30 and scale
    # 10 -> relative 3. The combined norm is sqrt(5**2 + 3**2).
    norm = BlockScaledNorm(sizes=(2, 3), scales=(1.0, 10.0))
    residual = jnp.array([3.0, 4.0, 10.0, 20.0, 20.0])
    assert float(norm(residual)) == pytest.approx(np.sqrt(34.0))


def test_block_scaled_norm_reduces_to_the_relative_residual_for_one_block():
    # A single block scaled by its own reference magnitude is the plain relative residual.
    r0 = jnp.array([3.0, 4.0])  # norm 5
    norm = BlockScaledNorm(sizes=(2,), scales=(5.0,))
    assert float(norm(r0)) == pytest.approx(1.0)
    assert float(norm(0.1 * r0)) == pytest.approx(0.1)


def test_block_scaling_stops_the_large_block_from_dominating():
    # Two blocks of equal Euclidean norm but disparate reference scale: the plain norm weights them
    # equally, while the scaled norm makes the small-scale block dominate (its residual is far above
    # its reference, the large-scale block's far below).
    residual = jnp.array([10.0, 10.0])
    norm = BlockScaledNorm(sizes=(1, 1), scales=(1.0, 100.0))
    # block 0 relative 10/1 = 10 dominates block 1 relative 10/100 = 0.1
    assert float(norm(residual)) == pytest.approx(np.sqrt(10.0**2 + 0.1**2))


def test_block_norm_lets_the_line_search_see_a_small_scale_block():
    # R = phi, so a step is judged purely by where it moves phi. From [10, 10] the step [-9, +5]
    # reduces block 0 (10 -> 1) while raising block 1 (10 -> 15).
    def residual_fn(phi):
        return phi

    phi = jnp.array([10.0, 10.0])
    delta = jnp.array([-9.0, 5.0])

    # The plain Euclidean norm is dominated by block 1's rise (||[1, 15]|| = 15.03 > ||[10, 10]|| =
    # 14.14), so the search backtracks away from the full step instead of taking block 0's descent.
    plain = backtracking_line_search(residual_fn, phi, delta, jnp.linalg.norm(phi), steps=4).phi
    assert not jnp.allclose(plain, phi + delta)
    assert float(plain[0]) > 1.5  # block 0 not fully reduced

    # Scaling block 1 by a large reference makes its rise negligible and block 0's descent visible,
    # so the full step is accepted.
    norm = BlockScaledNorm(sizes=(1, 1), scales=(1.0, 100.0))
    got = backtracking_line_search(
        residual_fn, phi, delta, norm(residual_fn(phi)), steps=4, norm=norm
    ).phi
    assert jnp.allclose(got, phi + delta)


def test_row_scaled_norm_equilibrates_then_normalizes_by_the_field_scale():
    """Stage 1 divides each row by its own diagonal; stage 2 divides each block by its field scale."""
    norm = RowScaledNorm(
        sizes=(2, 2), row_scale=jnp.array([2.0, 4.0, 1.0, 1.0]), field_scale=jnp.array([1.0, 10.0])
    )
    residual = jnp.array([2.0, 8.0, 3.0, 7.0])
    # block 0: mean(|2|/2, |8|/4) / 1 = mean(1, 2) = 1.5 ; block 1: mean(3, 7) / 10 = 0.5
    assert np.allclose(np.asarray(norm.per_block(residual)), [1.5, 0.5])
    assert float(norm(residual)) == pytest.approx(float(np.hypot(1.5, 0.5)))


def test_a_row_written_in_the_solved_variable_passes_through_unscaled():
    """A fixation row's derivative is one, so it must not be rescaled by a transport diagonal.

    Writing a value fixation in log form makes its row exactly linear in the solved unknown with unit
    derivative. Scaling such a row by a neighbouring transport row's diagonal would misreport it by
    orders of magnitude on a field spanning several -- the concrete defect that motivated asking each
    row for its own derivative rather than assuming one for the whole block.
    """
    row_scale = jnp.array([1000.0, 1000.0, 1.0])  # two transport rows, one fixation row
    norm = RowScaledNorm(sizes=(3,), row_scale=row_scale, field_scale=jnp.ones(1))
    # The fixation row's residual is already a fractional quantity and survives at full weight.
    assert float(norm(jnp.array([0.0, 0.0, 3.0]))) == pytest.approx(1.0)
    # The same raw magnitude in a transport row is a thousand times smaller once equilibrated.
    assert float(norm(jnp.array([3.0, 0.0, 0.0]))) == pytest.approx(1e-3)


def test_the_l1_mean_is_far_less_concentrated_than_a_root_mean_square():
    """The reason for the L1 mean: a squared measure is dominated by its largest entries.

    On a converged turbulent field the residual concentrates into a few cells with the sharpest
    near-wall gradients. A root-mean-square then keeps reporting those cells while the field as a whole
    converges; the L1 mean weighs them in proportion. Here one cell in a thousand is a hundred times
    the rest: it carries a few per cent of the L1 mean but the overwhelming majority of the square.
    """
    n = 1000
    values = np.full(n, 1.0)
    values[0] = 100.0
    l1_share = values[0] / values.sum()
    l2_share = values[0] ** 2 / (values**2).sum()
    assert l1_share < 0.10
    assert l2_share > 0.90

    norm = RowScaledNorm(sizes=(n,), row_scale=jnp.ones(n), field_scale=jnp.ones(1))
    # The measure the solver sees is the mean, so the outlier moves it by ~10%, not by ~10x.
    assert float(norm(jnp.asarray(values))) == pytest.approx(values.mean())


def test_rebuilding_the_measure_at_a_new_state_is_a_compilation_cache_hit():
    """The scales are rebuilt every outer iteration, so a rebuild must not trigger a recompile.

    Only the block sizes are static; the two scale arrays are ordinary leaves. Re-deriving the measure
    at a new state therefore changes values over an unchanged structure, which is a cache hit. Baking
    the scales in as compile-time constants instead would cost a full recompile per iteration -- far
    more than the measure is worth -- so this property is what makes the per-iteration cadence viable.
    """
    traces = 0

    @eqx.filter_jit
    def measure(norm, residual):
        nonlocal traces
        traces += 1
        return norm(residual)

    residual = jnp.array([1.0, 3.0, 2.0, 6.0])
    # Built through np.asarray so both carry the same (non-weak) dtype. A weakly-typed scale -- what
    # jnp.full with a Python float produces -- has a different abstract type and would retrace, so the
    # builder must emit ordinary arrays for the per-iteration rebuild to stay a cache hit.
    first = RowScaledNorm(
        sizes=(2, 2),
        row_scale=jnp.asarray(np.ones(4)),
        field_scale=jnp.asarray(np.ones(2)),
    )
    rebuilt = RowScaledNorm(
        sizes=(2, 2),
        row_scale=jnp.asarray(np.full(4, 2.0)),
        field_scale=jnp.asarray(np.full(2, 4.0)),
    )
    a = measure(first, residual)
    after_first = traces
    b = measure(rebuilt, residual)

    assert traces == after_first, "rebuilding the scales retraced instead of hitting the cache"
    # ...and the rebuild is a real change, not a silently ignored one.
    assert not jnp.allclose(a, b)


def test_the_row_scaled_measure_is_the_euclidean_combination_of_its_own_per_block_view():
    """The reporting view and the number the solver steers on must be the same arithmetic, or a log
    would explain a convergence history the march never had."""
    norm = RowScaledNorm(
        sizes=(2, 2),
        row_scale=jnp.array([2.0, 2.0, 1.0, 1.0]),
        field_scale=jnp.array([1.0, 4.0]),
    )
    residual = jnp.array([1.0, 3.0, 2.0, 6.0])

    assert float(norm(residual)) == pytest.approx(float(jnp.linalg.norm(norm.per_block(residual))))
