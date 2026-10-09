"""The frozen convection-diffusion stencil: assembly, its diagonal split, and detaching cells.

The hand-built three-cell stencil below is small enough to write its operator out entry by entry, so
the assembly is checked against an independent reference rather than against itself. Its two edges
carry fluxes of opposite sign and run in opposite directions relative to the middle cell, so a stencil
that scattered an upwind term to the wrong endpoint, or the wrong sign of the flux, cannot reproduce
it.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import numpy as np
import pytest
from aquaflux.solve import ConvectionDiffusionStencil, decouple_dof

# Edges (owner -> neighbour): 0 -> 1 with an outflow, 2 -> 1 with an inflow.
OWNER = np.array([0, 2])
NB = np.array([1, 1])
COEFFICIENT = np.array([2.0, 1.0])
FLUX = np.array([3.0, -0.5])
BOUNDARY = np.array([0.1, 0.0, 0.4])

# A[P, N] = -(c + max(-f, 0)), A[N, P] = -(c + max(f, 0)); diagonal P += c + max(f, 0),
# N += c + max(-f, 0), plus the boundary diagonal.
EXPECTED = np.array(
    [
        [0.1 + 2.0 + 3.0, -2.0, 0.0],
        [-(2.0 + 3.0), 2.0 + 1.0 + 0.5, -1.0],
        [0.0, -(1.0 + 0.5), 0.4 + 1.0],
    ]
)


def _stencil(**overrides) -> ConvectionDiffusionStencil:
    settings = dict(flux=FLUX, boundary_diagonal=BOUNDARY) | overrides
    return ConvectionDiffusionStencil(OWNER, NB, COEFFICIENT, 3, **settings)


def test_it_assembles_the_hand_written_operator() -> None:
    np.testing.assert_allclose(_stencil().assemble().toarray(), EXPECTED, rtol=0, atol=1e-15)


def test_without_a_flux_it_assembles_the_symmetric_graph_laplacian() -> None:
    a = _stencil(flux=None, boundary_diagonal=None).assemble().toarray()
    laplacian = np.array([[2.0, -2.0, 0.0], [-2.0, 3.0, -1.0], [0.0, -1.0, 1.0]])
    np.testing.assert_allclose(a, laplacian, rtol=0, atol=1e-15)


def test_the_diagonal_parts_are_the_upwind_outflow_and_the_rest() -> None:
    convective, dissipative = _stencil().diagonal_parts()
    np.testing.assert_allclose(convective, [3.0, 0.5, 0.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(dissipative, [2.1, 3.0, 1.4], rtol=0, atol=1e-15)


@pytest.mark.parametrize("flux", [None, "mixed"])
def test_the_diagonal_parts_sum_to_the_assembled_diagonal_on_an_irregular_graph(flux) -> None:
    """Without assembling, the two parts reproduce the assembled operator's diagonal.

    This is the property the pseudo-time shift relies on to equal the operator the preconditioner
    coarsens. The graph is random, so cells have different numbers of edges and appear as owner on
    some and neighbour on others, and the fluxes take both signs.
    """
    rng = np.random.default_rng(3)
    n, n_edges = 40, 120
    owner = rng.integers(0, n, n_edges)
    nb = (owner + rng.integers(1, n, n_edges)) % n  # never a self-loop
    stencil = ConvectionDiffusionStencil(
        owner,
        nb,
        rng.uniform(0.1, 2.0, n_edges),
        n,
        flux=None if flux is None else rng.normal(0.0, 3.0, n_edges),
        boundary_diagonal=rng.uniform(0.0, 1.0, n),
    )
    convective, dissipative = stencil.diagonal_parts()
    if flux is None:
        assert not convective.any()
    np.testing.assert_allclose(
        convective + dissipative, stencil.assemble().diagonal(), rtol=1e-14, atol=0
    )


def test_detaching_a_cell_drops_its_edges_from_both_ends_and_leaves_an_identity_row() -> None:
    stencil = _stencil()
    detached = stencil.detached(np.array([2]))
    expected = np.array([[5.1, -2.0, 0.0], [-5.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(detached.assemble().toarray(), expected, rtol=0, atol=1e-15)
    # The stencil it was detached from is unchanged.
    np.testing.assert_allclose(stencil.assemble().toarray(), EXPECTED, rtol=0, atol=1e-15)


def test_detaching_a_cell_drops_the_edges_it_is_the_NEIGHBOUR_of() -> None:
    """Cell 1 owns no edge, so only a rule that reads both endpoints detaches it."""
    detached = _stencil().detached(np.array([1])).assemble().toarray()
    np.testing.assert_allclose(detached, np.diag([0.1, 1.0, 0.4]), rtol=0, atol=1e-15)


def test_detaching_is_not_decoupling_the_assembled_operator() -> None:
    """The two regularizations differ at the neighbour: detaching removes the edge's diagonal there,
    decoupling an assembled operator keeps it."""
    detached = _stencil().detached(np.array([2])).assemble().toarray()
    decoupled = decouple_dof(_stencil().assemble(), 2).toarray()
    assert detached[1, 1] == pytest.approx(2.0)
    assert decoupled[1, 1] == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(coefficient=np.ones(3)), "coefficient must have shape"),
        (dict(flux=np.ones(3)), "flux must have shape"),
        (dict(boundary_diagonal=np.ones(2)), "boundary_diagonal must have shape"),
    ],
)
def test_an_array_that_does_not_match_the_graph_is_refused(overrides, message) -> None:
    settings = dict(coefficient=COEFFICIENT, flux=FLUX, boundary_diagonal=BOUNDARY) | overrides
    coefficient = settings.pop("coefficient")
    with pytest.raises(ValueError, match=message):
        ConvectionDiffusionStencil(OWNER, NB, coefficient, 3, **settings)
