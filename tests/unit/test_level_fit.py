"""The device fit of a frozen coarsening's level values (``precondition/level_fit.py``).

A frozen coarsening holds a hierarchy's structure and fits every value a level carries -- operator,
diagonal, cell-block inverse, spectral estimate, coarse inverse -- from the fine operator's stored values
in one compiled function, for the first build and every refit alike. Each test checks those values
against the same quantities computed independently on the host with scipy and numpy.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.precondition.frozen_operator import ConvectionDiffusionStencil, equilibration_scale
from aquaflux.precondition.level_fit import (
    dense_inverse,
    invert_cell_blocks,
    spectral_estimate,
    unit_start,
)
from aquaflux.precondition.multigrid import build_convection_hierarchy


def _chain(n: int, flux_scale: float, diffusivity: np.ndarray) -> sp.csr_matrix:
    owner, nb = np.arange(n - 1), np.arange(1, n)
    return ConvectionDiffusionStencil(
        owner,
        nb,
        diffusivity,
        n,
        flux=flux_scale * np.ones(n - 1),
        boundary_diagonal=np.ones(n),
    ).assemble()


def _two_field(n: int, coupling: float, seed: int) -> sp.csr_matrix:
    """A field-major two-field operator: two chains, every cell's fields coupled both ways."""
    rng = np.random.default_rng(seed)
    first = _chain(n, 1.0 + rng.random(), 1.0 + rng.random(n - 1))
    second = _chain(n, 0.5 + rng.random(), 2.0 + rng.random(n - 1))
    cross = sp.diags(coupling * (1.0 + rng.random(n)))
    return sp.bmat([[first, cross], [-0.5 * cross, second]]).tocsr()


def _host_power_estimate(matrix: np.ndarray, start: np.ndarray, iterations: int = 20) -> float:
    vector, estimate = start, 1.0
    for _ in range(iterations):
        image = matrix @ vector
        estimate = np.linalg.norm(image)
        vector = image / estimate
    return float(estimate)


@pytest.mark.parametrize("block_size", [1, 2])
def test_a_refit_holds_every_level_value_a_host_computation_gives(block_size: int) -> None:
    """Refitted to a new operator, each level holds what scipy and numpy compute from that operator.

    The level operator is the Galerkin product of the one above through the held prolongation; the
    diagonal is its diagonal; a nodal level's block inverse inverts each cell's block; ``lam_max`` is
    20 power steps of ``D⁻¹A`` from the fixed start; the coarsest level holds the dense inverse.
    """
    n = 90
    build = _two_field if block_size == 2 else (lambda n, c, s: _chain(n, 1.0 + s, np.ones(n - 1)))
    held = []
    build_convection_hierarchy(
        build(n, 0.3, 1),
        block_size=block_size,
        prolongation_smoothing="none",
        mis_aggregation=True,
        max_coarse=12,
        max_levels=4,
        coarsening=held,
    )
    operator = build(n, 0.7, 2)
    fitted = held[0].fit(operator)
    assert len(fitted.levels) >= 3

    host = operator.toarray()
    for depth, level in enumerate(fitted.levels):
        np.testing.assert_allclose(
            level.operator.to_scipy().toarray(), host, rtol=1e-12, atol=1e-12 * np.abs(host).max()
        )
        np.testing.assert_allclose(np.asarray(level.diagonal), np.diag(host), rtol=1e-12)
        if block_size == 1:
            preconditioned = host / np.diag(host)[:, None]
        else:
            cells = level.n // block_size
            blocks = np.stack(
                [host[c::cells, c::cells] for c in range(cells)]
            )  # field-major: cell c's fields at c, c + cells
            want = np.linalg.inv(blocks)
            np.testing.assert_allclose(np.asarray(level.block_inverse), want, rtol=1e-10)
            inverse = np.zeros_like(host)
            for c in range(cells):
                inverse[c::cells, c::cells] = want[c]
            preconditioned = inverse @ host
        np.testing.assert_allclose(
            float(level.lam_max),
            _host_power_estimate(preconditioned, unit_start(level.n)),
            rtol=1e-10,
        )
        if depth == len(fitted.levels) - 1:
            np.testing.assert_allclose(np.asarray(level.coarse_inv), np.linalg.inv(host), rtol=1e-9)
            break
        prolongation = level.prolongation_scipy().tocsr()
        host = (prolongation.T @ sp.csr_matrix(host) @ prolongation).toarray()


def test_an_equilibrated_fit_rescales_by_the_operator_it_is_fitted_to() -> None:
    """The scale is re-derived from each operator fitted, by the host rule, not kept from the build.

    Fitting ``4 A`` must carry half ``A``'s scale and the very levels ``A`` gives, since ``D A D`` is
    unchanged when ``A`` is scaled and ``D`` scaled inversely by its square root.
    """
    a = _two_field(60, 0.4, 3)
    held = []
    build_convection_hierarchy(
        a, block_size=2, equilibrate=True, prolongation_smoothing="none", coarsening=held
    )
    at_a, at_4a = held[0].fit(a), held[0].fit(4.0 * a)
    np.testing.assert_allclose(np.asarray(at_a.equilibration), equilibration_scale(a.diagonal()))
    np.testing.assert_allclose(
        np.asarray(at_4a.equilibration), 0.5 * np.asarray(at_a.equilibration), rtol=1e-15
    )
    for one, four in zip(at_a.levels, at_4a.levels, strict=True):
        np.testing.assert_allclose(
            np.asarray(one.operator.values), np.asarray(four.operator.values), rtol=1e-13
        )


def test_a_refit_refuses_a_level_whose_diagonal_stops_being_positive() -> None:
    """A scalar level's diagonal is checked on the device and refused on the host, naming the level."""
    held = []
    build_convection_hierarchy(
        _chain(120, 1.0, np.ones(119)), prolongation_smoothing="none", coarsening=held
    )
    flipped = _chain(120, 1.0, np.ones(119)).tolil()
    flipped[5, 5] = -1.0
    with pytest.raises(
        ValueError,
        match=r"where \(level 0\): operator diagonal must be finite and strictly positive",
    ):
        held[0].fit(flipped.tocsr(), "where")


def test_a_refit_refuses_a_singular_cell_block_and_counts_it() -> None:
    """A nodal level counts its singular cell blocks on the device and the host refuses them."""
    held = []
    a = _two_field(40, 0.3, 4)
    build_convection_hierarchy(a, block_size=2, prolongation_smoothing="none", coarsening=held)
    singular = a.tolil()
    # Cell 3's block made rank one: both rows equal, both columns equal.
    singular[3, 3], singular[3, 43], singular[43, 3], singular[43, 43] = 1.0, 1.0, 1.0, 1.0
    with pytest.raises(ValueError, match="1 of 40 are singular"):
        held[0].fit(singular.tocsr())


def test_the_device_pieces_match_numpy_on_their_own() -> None:
    """The three traced kernels against numpy: block inverses and their singularity count, the dense
    inverse (and its pseudo-inverse fallback), and the power estimate (held at one on a vanishing map)."""
    rng = np.random.default_rng(5)
    blocks = rng.standard_normal((6, 3, 3)) + 3.0 * np.eye(3)
    blocks[2] = np.outer([1.0, 2.0, 3.0], [1.0, -1.0, 2.0])
    inverse, singular = invert_cell_blocks(jnp.asarray(blocks))
    assert int(singular) == 1
    keep = [0, 1, 3, 4, 5]
    np.testing.assert_allclose(np.asarray(inverse)[keep], np.linalg.inv(blocks[keep]), rtol=1e-12)

    dense = rng.standard_normal((30, 30)) + 30.0 * np.eye(30)
    np.testing.assert_allclose(np.asarray(dense_inverse(jnp.asarray(dense))), np.linalg.inv(dense))
    almost = np.eye(30)
    almost[4, 4] = 1e-18
    np.testing.assert_allclose(
        np.asarray(dense_inverse(jnp.asarray(almost))), np.linalg.pinv(almost), atol=1e-12
    )

    matrix = rng.standard_normal((25, 25))
    start = unit_start(25)
    got = spectral_estimate(lambda v: jnp.asarray(matrix) @ v, jnp.asarray(start))
    np.testing.assert_allclose(float(got), _host_power_estimate(matrix, start), rtol=1e-12)
    assert float(spectral_estimate(lambda v: 0.0 * v, jnp.asarray(start))) == 1.0
