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
    _convection_diffusion_csr,
    air_multigrid_solve,
    build_convection_air_hierarchy,
    build_convection_hierarchy,
    build_hierarchy,
    build_smoothed_hierarchy,
    convection_multigrid_solve,
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


# --- convection-diffusion (nonsymmetric) aggregation ------------------------------------


def _convective_grid(n, mu, speed):
    """Interior edges of an ``n x n`` grid with a uniform streamwise mass flux (strong convection).

    Returns ``(owner, nb, viscous, mdot, n_cells, boundary_diagonal)`` for a convection-diffusion
    operator whose x-faces carry the flux ``speed / n`` (cell Peclet ``speed / (n mu)``) and whose
    boundary diagonal makes it a nonsingular M-matrix (Dirichlet on all sides, outflow at the right).
    """
    mesh = structured_grid_2d(n, n)
    owner = np.asarray(mesh.face_cells.owner)
    nb = np.asarray(mesh.face_cells.neighbour)
    interior = nb >= 0
    o, m = owner[interior], nb[interior]
    h = 1.0 / n
    ncell = mesh.n_cells
    coords = np.asarray(mesh.geometry().cell.centroid)
    # Streamwise (x) faces connect cells that differ in the x-centroid; they carry the mass flux.
    x_face = np.abs(coords[o, 0] - coords[m, 0]) > np.abs(coords[o, 1] - coords[m, 1])
    viscous = np.full(o.shape, mu)  # mu * A/(d.n) with A = d.n = h on a unit grid
    mdot = np.where(x_face, speed * h, 0.0)
    # Boundary diagonal: a Dirichlet stiffness on every border cell plus outflow convection at the
    # right column, enough to make the operator diagonally dominant and nonsingular.
    bd = np.zeros(ncell)
    on_border = (
        (coords[:, 0] < h) | (coords[:, 0] > 1 - h) | (coords[:, 1] < h) | (coords[:, 1] > 1 - h)
    )
    bd[on_border] += mu
    bd[coords[:, 0] > 1 - h] += speed * h  # outflow leaves the owner at the right boundary
    return o, m, viscous, mdot, ncell, bd


def _dense(owner, nb, visc, mdot, n, bd):
    a = np.asarray(_convection_diffusion_csr(owner, nb, visc, mdot, n).toarray())
    a[np.diag_indices(n)] += bd
    return jnp.asarray(a)


def test_convection_operator_is_a_nonsymmetric_m_matrix() -> None:
    """The convection-diffusion operator has a positive diagonal, non-positive off-diagonals, and is
    nonsymmetric exactly when there is a mass flux (symmetric viscous limit at zero flux)."""
    owner, nb, visc, mdot, n, bd = _convective_grid(6, mu=1e-2, speed=1.0)
    a = np.asarray(_dense(owner, nb, visc, mdot, n, bd))
    off = a - np.diag(np.diag(a))
    assert np.all(np.diag(a) > 0.0)
    assert np.all(off <= 1e-12)  # M-matrix: off-diagonals non-positive
    assert not np.allclose(a, a.T)  # convection makes it nonsymmetric
    a0 = np.asarray(_dense(owner, nb, visc, np.zeros_like(mdot), n, bd))
    assert np.allclose(a0, a0.T)  # zero flux -> symmetric viscous operator


def test_convection_v_cycle_preconditions_gmres_at_high_peclet() -> None:
    """The frozen convection-diffusion V-cycle accelerates GMRES on the strongly-convective operator —
    its actual role as a left preconditioner. On a fixed, small Krylov budget the preconditioned solve
    reaches a residual orders of magnitude below the unpreconditioned one, at a cell Peclet number
    where the operator is convection-dominated."""
    owner, nb, visc, mdot, n, bd = _convective_grid(24, mu=1e-3, speed=1.0)  # cell Peclet ~40
    hierarchy = build_convection_hierarchy(owner, nb, visc, mdot, n, boundary_diagonal=bd)
    a = _dense(owner, nb, visc, mdot, n, bd)
    b = jnp.asarray(np.random.default_rng(0).standard_normal(n))

    def matvec(x):
        return a @ x

    def preconditioner(r):
        return convection_multigrid_solve(hierarchy, r, cycles=1)

    budget = dict(tol=0.0, atol=0.0, maxiter=1, restart=20)  # a fixed 20-vector Krylov budget
    plain, _ = jax.scipy.sparse.linalg.gmres(matvec, b, **budget)
    pre, _ = jax.scipy.sparse.linalg.gmres(matvec, b, M=preconditioner, **budget)
    rb = float(jnp.linalg.norm(b))
    plain_residual = float(jnp.linalg.norm(matvec(plain) - b)) / rb
    pre_residual = float(jnp.linalg.norm(matvec(pre) - b)) / rb
    assert pre_residual < 1e-3  # the preconditioned solve makes real progress on the same budget
    assert pre_residual < 0.05 * plain_residual  # far below the unpreconditioned residual


def test_convection_multigrid_is_linear_in_rhs() -> None:
    """A fixed cycle/sweep count makes the convection V-cycle a constant linear operator (so it is a
    valid frozen left preconditioner that transposes cleanly for the adjoint)."""
    owner, nb, visc, mdot, n, bd = _convective_grid(8, mu=1e-2, speed=1.0)
    hierarchy = build_convection_hierarchy(owner, nb, visc, mdot, n, boundary_diagonal=bd)
    rng = np.random.default_rng(1)
    r1 = jnp.asarray(rng.standard_normal(n))
    r2 = jnp.asarray(rng.standard_normal(n))

    def solve(r):
        return convection_multigrid_solve(hierarchy, r, cycles=2)

    assert jnp.allclose(solve(1.5 * r1 - 0.5 * r2), 1.5 * solve(r1) - 0.5 * solve(r2), atol=1e-10)


# --- local approximate ideal restriction (lAIR) -----------------------------------------


def test_air_v_cycle_is_mesh_independent_at_high_peclet() -> None:
    """The reduction-based (lAIR) V-cycle converges the strongly-convective operator with a **flat**
    per-cycle contraction as the mesh refines — the Peclet-robust, mesh-independent property the
    two-level aggregation method (and deep Galerkin recursion) lack. For a convection-dominated
    operator the approximate ideal restriction is nearly exact, so a few cycles behave like a direct
    solve regardless of the mesh size."""
    contractions = []
    for n in (24, 48):  # cell Peclet ~40; a 4x change in cell count
        owner, nb, visc, mdot, ncell, bd = _convective_grid(n, mu=1e-3, speed=1.0)
        hierarchy = build_convection_air_hierarchy(
            owner, nb, visc, mdot, ncell, boundary_diagonal=bd
        )
        a = _dense(owner, nb, visc, mdot, ncell, bd)
        b = jnp.asarray(np.random.default_rng(0).standard_normal(ncell))
        x = jnp.zeros(ncell)
        norms = [1.0]
        for _ in range(4):
            x = x + air_multigrid_solve(hierarchy, b - a @ x, cycles=1)
            norms.append(float(jnp.linalg.norm(a @ x - b)) / float(jnp.linalg.norm(b)))
        # geometric-mean per-cycle contraction over the cycles above the machine floor
        ratios = [norms[k + 1] / norms[k] for k in range(len(norms) - 1) if norms[k] > 1e-11]
        contractions.append(float(np.exp(np.mean(np.log(ratios)))))
    assert max(contractions) < 0.1  # strong contraction at high Peclet
    assert abs(contractions[1] - contractions[0]) < 0.05  # ~mesh-independent across the refinement


def test_air_multigrid_is_linear_and_transposable() -> None:
    """A fixed cycle/sweep count makes the lAIR V-cycle a constant linear operator that transposes
    cleanly (``R != Pᵀ``), so it is a valid frozen left preconditioner for the forward solve and its
    ``M^T`` adjoint."""
    owner, nb, visc, mdot, n, bd = _convective_grid(12, mu=1e-2, speed=1.0)
    hierarchy = build_convection_air_hierarchy(owner, nb, visc, mdot, n, boundary_diagonal=bd)
    rng = np.random.default_rng(1)
    r1 = jnp.asarray(rng.standard_normal(n))
    r2 = jnp.asarray(rng.standard_normal(n))

    def solve(r):
        return air_multigrid_solve(hierarchy, r, cycles=2)

    assert jnp.allclose(solve(1.5 * r1 - 0.5 * r2), 1.5 * solve(r1) - 0.5 * solve(r2), atol=1e-10)

    # transpose consistency: <u, M r1> == <M^T u, r1>
    transpose = jax.linear_transpose(solve, r1)
    u = jnp.asarray(rng.standard_normal(n))
    assert jnp.allclose(jnp.dot(u, solve(r1)), jnp.dot(transpose(u)[0], r1), rtol=1e-9)
