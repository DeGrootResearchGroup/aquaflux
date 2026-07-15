"""Unit tests for the aggregation-multigrid building blocks (structure, Galerkin, V-cycle).

These cover the properties that hold regardless of the interpolation order: a healthy aggregation,
an exact piecewise-constant Galerkin coarse operator, a convergent and linear V-cycle. (The current
unsmoothed prolongation is a correct-but-weak coarse space; smoothed aggregation is the accuracy
upgrade.)
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.mesh import structured_grid_2d
from aquaflux.solve.multigrid import (
    build_hierarchy,
    build_smoothed_hierarchy,
    level_coefficients,
    multigrid_solve,
    smoothed_multigrid_solve,
)


def _poisson(n):
    """Interior-face edges of an n x n grid (a model graph Laplacian) and the cell count."""
    mesh = structured_grid_2d(n, n)
    interior = np.asarray(mesh.face_cells.neighbour) >= 0
    return (
        np.asarray(mesh.face_cells.owner)[interior],
        np.asarray(mesh.face_cells.neighbour)[interior],
        mesh.n_cells,
    )


def _dense_laplacian(owner, nb, n, coeff=None):
    coeff = np.ones(len(owner)) if coeff is None else np.asarray(coeff)
    a = np.zeros((n, n))
    for o, m, c in zip(owner, nb, coeff, strict=True):
        a[o, o] += c
        a[m, m] += c
        a[o, m] -= c
        a[m, o] -= c
    return a


def _pinned_v_cycle_factor(owner, nb, ncell, pin, *, seed=0):
    """Geometric-mean contraction of the fixed smoothed V-cycle on a pin-decoupled Poisson.

    The pin row/column is zeroed out of both the hierarchy (via ``build_smoothed_hierarchy``'s
    ``pin=``) and the dense operator, so the two share a null space and the V-cycle is a valid
    inner solve. Returns the cube-root of the residual-norm ratio over the last three cycles —
    the asymptotic per-cycle contraction factor. Shared by the mesh-independence and
    ordering-invariance checks.
    """
    hierarchy = build_smoothed_hierarchy(owner, nb, np.ones(len(owner)), ncell, pin=pin)
    a = _dense_laplacian(owner, nb, ncell)
    a[pin, :] = 0.0
    a[:, pin] = 0.0
    a[pin, pin] = 1.0
    a = jnp.asarray(a)
    b = jnp.asarray(np.random.default_rng(seed).standard_normal(ncell))
    x = jnp.zeros(ncell)
    norms = [float(jnp.linalg.norm(a @ x - b))]
    for _ in range(8):
        x = x + smoothed_multigrid_solve(hierarchy, b - a @ x, cycles=1)
        norms.append(float(jnp.linalg.norm(a @ x - b)))
    return (norms[-1] / norms[-4]) ** (1.0 / 3.0)


def test_aggregation_is_healthy() -> None:
    """The two-pass aggregation covers every cell, has no singleton-dominated coarsening."""
    owner, nb, ncell = _poisson(16)
    hierarchy = build_hierarchy(owner, nb, ncell)
    agg = np.asarray(hierarchy.levels[0].agg)
    assert (agg >= 0).all()  # every cell aggregated
    assert ncell / hierarchy.levels[1].n > 3.0  # healthy coarsening ratio (~4-5x)


def test_galerkin_coarse_operator_is_correct() -> None:
    """The coarse operator built by ``level_coefficients`` equals the exact ``P^T A P``."""
    owner, nb, ncell = _poisson(8)
    hierarchy = build_hierarchy(owner, nb, ncell, max_levels=2)
    coeffs, _ = level_coefficients(hierarchy, jnp.ones(owner.shape[0]))
    a = _dense_laplacian(owner, nb, ncell)
    agg = np.asarray(hierarchy.levels[0].agg)
    n1 = hierarchy.levels[1].n
    p = np.zeros((ncell, n1))
    p[np.arange(ncell), agg] = 1.0
    a_coarse = _dense_laplacian(
        np.asarray(hierarchy.levels[1].owner), np.asarray(hierarchy.levels[1].nb), n1, coeffs[1]
    )
    assert np.max(np.abs(a_coarse - p.T @ a @ p)) < 1e-10


def test_coarse_coefficients_conserve_the_crossing_sum() -> None:
    """Each coarse edge coefficient is the sum of the fine coefficients crossing the aggregates."""
    owner, nb, ncell = _poisson(8)
    hierarchy = build_hierarchy(owner, nb, ncell, max_levels=2)
    fine = jnp.asarray(np.random.default_rng(0).uniform(0.5, 2.0, owner.shape[0]))
    coeffs, _ = level_coefficients(hierarchy, fine)
    agg = np.asarray(hierarchy.levels[0].agg)
    crossing = agg[owner] != agg[nb]
    assert abs(float(jnp.sum(coeffs[1])) - float(jnp.sum(fine[crossing]))) < 1e-9


def test_v_cycle_reduces_residual() -> None:
    """Fixed V-cycles drive the residual of a pinned model Poisson down (a valid inner solve)."""
    owner, nb, ncell = _poisson(16)
    hierarchy = build_hierarchy(owner, nb, ncell, pin=0)
    coeffs, diagonals = level_coefficients(hierarchy, jnp.ones(owner.shape[0]))
    o, m = hierarchy.levels[0].owner, hierarchy.levels[0].nb

    def matvec(p):
        flux = p[o] - p[m]
        return (
            (jax.ops.segment_sum(flux, o, ncell) - jax.ops.segment_sum(flux, m, ncell))
            .at[0]
            .set(p[0])
        )

    b = jnp.asarray(np.random.default_rng(0).standard_normal(ncell)).at[0].set(0.0)
    x = multigrid_solve(hierarchy, coeffs, diagonals, b, cycles=12)
    assert float(jnp.linalg.norm(matvec(x) - b)) < 0.3 * float(jnp.linalg.norm(b))


def test_multigrid_solve_is_linear_in_rhs() -> None:
    """A fixed cycle count makes rhs -> x linear (needed for a plain-GMRES left preconditioner)."""
    owner, nb, ncell = _poisson(8)
    hierarchy = build_hierarchy(owner, nb, ncell, pin=0)
    coeffs, diagonals = level_coefficients(hierarchy, jnp.ones(owner.shape[0]))
    rng = np.random.default_rng(1)
    r1 = jnp.asarray(rng.standard_normal(ncell))
    r2 = jnp.asarray(rng.standard_normal(ncell))

    def solve(r):
        return multigrid_solve(hierarchy, coeffs, diagonals, r, cycles=2)

    assert jnp.allclose(solve(2.0 * r1 - 3.0 * r2), 2.0 * solve(r1) - 3.0 * solve(r2), atol=1e-10)


# --- smoothed aggregation --------------------------------------------------------------


def _smoothed_residual_factor(n):
    """Geometric-mean V-cycle residual factor of the smoothed hierarchy on a singular Poisson."""
    owner, nb, ncell = _poisson(n)
    hierarchy = build_smoothed_hierarchy(owner, nb, np.ones(owner.shape[0]), ncell, pin=None)
    o, m = jnp.asarray(owner), jnp.asarray(nb)

    def matvec(p):
        flux = p[o] - p[m]
        return jax.ops.segment_sum(flux, o, ncell) - jax.ops.segment_sum(flux, m, ncell)

    def proj(v):
        return v - jnp.mean(v)

    b = proj(jnp.asarray(np.random.default_rng(0).standard_normal(ncell)))
    x = jnp.zeros(ncell)
    norms = [float(jnp.linalg.norm(matvec(x) - b))]
    for _ in range(10):
        x = proj(x + proj(smoothed_multigrid_solve(hierarchy, b - matvec(x), cycles=1)))
        norms.append(float(jnp.linalg.norm(matvec(x) - b)))
    return (norms[-1] / norms[-4]) ** (1 / 3)


def test_smoothed_aggregation_beats_unsmoothed() -> None:
    """The smoothed coarse space contracts the V-cycle far faster than piecewise-constant."""
    owner, nb, ncell = _poisson(24)
    smoothed = build_smoothed_hierarchy(owner, nb, np.ones(owner.shape[0]), ncell, pin=0)
    unsmoothed = build_hierarchy(owner, nb, ncell, pin=0)
    coeffs, diagonals = level_coefficients(unsmoothed, jnp.ones(owner.shape[0]))
    o, m = jnp.asarray(owner), jnp.asarray(nb)

    def matvec(p):
        flux = p[o] - p[m]
        return (
            (jax.ops.segment_sum(flux, o, ncell) - jax.ops.segment_sum(flux, m, ncell))
            .at[0]
            .set(p[0])
        )

    b = jnp.asarray(np.random.default_rng(0).standard_normal(ncell)).at[0].set(0.0)
    r_smoothed = float(jnp.linalg.norm(matvec(smoothed_multigrid_solve(smoothed, b, cycles=3)) - b))
    r_unsmoothed = float(
        jnp.linalg.norm(matvec(multigrid_solve(unsmoothed, coeffs, diagonals, b, cycles=3)) - b)
    )
    assert r_smoothed < r_unsmoothed  # smoothed converges faster at equal cycles


def test_smoothed_aggregation_v_cycle_is_mesh_independent() -> None:
    """The whole point: the fixed smoothed V-cycle's contraction factor stays bounded (does not
    degrade toward 1) as the mesh refines — what makes it a scalable inner solve."""

    def factor(n):
        owner, nb, ncell = _poisson(n)
        return _pinned_v_cycle_factor(owner, nb, ncell, 0)

    coarse, fine = factor(16), factor(48)  # 256 vs 2304 cells
    assert coarse < 0.5 and fine < 0.5  # strong contraction at both sizes (~0.25 measured)
    assert fine < 1.8 * coarse  # bounded — not the ~0.97-degrading unsmoothed behaviour


def test_ordering_affects_coarse_space_and_rcm_restores_it() -> None:
    """Cell ordering affects the V-cycle contraction *through the aggregation coarse space*, and
    reverse Cuthill--McKee restores the good rate a scramble destroys.

    The smoother (damped Jacobi / Chebyshev) is a polynomial in the operator and the coarse solve
    is direct, so both are exactly permutation-invariant. But the *greedy* aggregation visits
    cells in index order: a spatially-local numbering (the natural structured one) yields compact
    aggregates and a good factor (~0.25); a scrambled numbering yields irregular aggregates and a
    markedly worse factor (~0.45). RCM re-localizes the numbering and recovers the good factor.
    This is why the large-mesh pipeline reorders (RCM) before building the preconditioner — it is
    a coarse-space effect, not a smoother effect."""
    owner, nb, ncell = _poisson(24)
    natural = _pinned_v_cycle_factor(owner, nb, ncell, 0)

    perm = np.random.default_rng(1).permutation(ncell)  # perm[old] = new
    o_s, m_s = perm[owner], perm[nb]
    scrambled = _pinned_v_cycle_factor(o_s, m_s, ncell, int(perm[0]))

    # RCM the scrambled graph (via a tiny stand-in Mesh carrying just the connectivity we need).
    rcm_of_scramble = _rcm_reordered_factor(o_s, m_s, ncell)

    assert natural < 0.35  # good ordering: strong contraction
    assert scrambled > natural + 0.1  # scrambling measurably degrades the coarse space
    assert rcm_of_scramble < natural + 0.1  # RCM restores the near-natural rate


def _rcm_reordered_factor(owner, nb, ncell):
    """Contraction factor after RCM-reordering a graph given by (owner, nb) edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    rows = np.concatenate([owner, nb])
    cols = np.concatenate([nb, owner])
    graph = coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(ncell, ncell)).tocsr()
    old_order = reverse_cuthill_mckee(graph, symmetric_mode=True)  # old_order[new] = old
    relabel = np.argsort(old_order)  # relabel[old] = new
    return _pinned_v_cycle_factor(relabel[owner], relabel[nb], ncell, int(relabel[owner[0]]))


def test_smoothed_multigrid_is_linear_in_rhs() -> None:
    """A fixed cycle count makes the smoothed V-cycle a constant linear operator."""
    owner, nb, ncell = _poisson(8)
    hierarchy = build_smoothed_hierarchy(owner, nb, np.ones(owner.shape[0]), ncell, pin=0)
    rng = np.random.default_rng(1)
    r1 = jnp.asarray(rng.standard_normal(ncell))
    r2 = jnp.asarray(rng.standard_normal(ncell))

    def solve(r):
        return smoothed_multigrid_solve(hierarchy, r, cycles=2)

    assert jnp.allclose(solve(1.5 * r1 - 0.5 * r2), 1.5 * solve(r1) - 0.5 * solve(r2), atol=1e-10)
