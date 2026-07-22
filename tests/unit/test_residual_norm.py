"""The block-scaled residual norm and its effect on the line search.

``BlockScaledNorm`` scales each contiguous block of a residual by its own reference magnitude before
combining, so a heterogeneous block system (a coupled RANS state whose ``omega`` residual is O(1e5)
and ``k`` residual O(1e-3)) is judged on every block rather than the one with the largest magnitude.
These tests pin the norm's arithmetic and, behaviourally, that it makes the shared backtracking line
search accept a step the plain Euclidean norm would reject — the globalization fix for the coupled
march, exercised here with no mesh or flow.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.solve import BlockScaledNorm
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
    plain = backtracking_line_search(residual_fn, phi, delta, jnp.linalg.norm(phi), steps=4)
    assert not jnp.allclose(plain, phi + delta)
    assert float(plain[0]) > 1.5  # block 0 not fully reduced

    # Scaling block 1 by a large reference makes its rise negligible and block 0's descent visible,
    # so the full step is accepted.
    norm = BlockScaledNorm(sizes=(1, 1), scales=(1.0, 100.0))
    got = backtracking_line_search(
        residual_fn, phi, delta, norm(residual_fn(phi)), steps=4, norm=norm
    )
    assert jnp.allclose(got, phi + delta)
