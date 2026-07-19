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
import pytest
from aquaflux.mesh import structured_grid_2d
from aquaflux.solve.multigrid import (
    _chebyshev_smooth,
    _convection_diffusion_csr,
    _jacobi_smooth,
    _SparseLevel,
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


def test_aggregation_coarse_space_is_ordering_robust() -> None:
    """The V-cycle contraction does not depend on the incoming cell numbering: the aggregation
    visits cells in a locality-preserving (reverse Cuthill--McKee) order taken from each level's own
    graph, so a scrambled numbering yields the same compact aggregates a spatially-local one does.

    The smoother (damped Jacobi / Chebyshev) is a polynomial in the operator and the coarse solve is
    direct, so both are exactly permutation-invariant. The *greedy* aggregation is the only
    ordering-sensitive piece — it seeds aggregates in visit order — so ordering that visit internally
    is what makes the whole coarse space ordering-robust. Without it a scramble degrades the factor
    markedly (~0.25 -> ~0.45); with it the scramble stays near the natural rate. This is why no mesh
    renumbering is needed upstream: the aggregation re-localizes each level itself."""
    owner, nb, ncell = _poisson(24)
    natural = _pinned_v_cycle_factor(owner, nb, ncell, 0)

    perm = np.random.default_rng(1).permutation(ncell)  # perm[old] = new
    scrambled = _pinned_v_cycle_factor(perm[owner], perm[nb], ncell, int(perm[0]))

    assert natural < 0.35  # good contraction on the natural numbering
    assert scrambled < natural + 0.1  # a scramble no longer degrades it — RCM is applied internally


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


def _chebyshev_propagation_polynomial(eigenvalues, lam_max, degree, lo_frac):
    """Sample the smoother's error-propagation polynomial ``P(mu)`` at the given eigenvalues.

    Builds a diagonal level whose preconditioned operator is ``D^-1 A = diag(eigenvalues)`` (unit
    diagonal, ``A`` diagonal), then smooths ``x0 = 1`` against ``b = 0`` (so the exact solution is
    ``0`` and ``x`` after smoothing equals ``P(mu)`` mode by mode). Returns ``P`` at each eigenvalue.
    """
    n = len(eigenvalues)
    index = jnp.arange(n)
    level = _SparseLevel(
        n=n,
        row=index,
        col=index,
        val=jnp.asarray(eigenvalues),
        diagonal=jnp.ones(n),
        lam_max=float(lam_max),
        coarse_inv=None,
        p_frow=None,
        p_ccol=None,
        p_val=None,
        n_coarse=0,
    )
    x0 = jnp.ones(n)
    return np.asarray(_chebyshev_smooth(level, jnp.zeros_like(x0), x0, degree, lo_frac))


def _scaled_chebyshev(eigenvalues, lo, hi, degree):
    """Analytic scaled Chebyshev polynomial ``T_k((theta - z)/delta) / T_k(theta/delta)`` on ``[lo, hi]``."""
    theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
    coeffs = [0] * degree + [1]
    argument = (theta - np.asarray(eigenvalues)) / delta
    return np.polynomial.chebyshev.chebval(argument, coeffs) / np.polynomial.chebyshev.chebval(
        theta / delta, coeffs
    )


def test_chebyshev_smoother_matches_the_scaled_chebyshev_polynomial() -> None:
    """The smoother realizes the min-max scaled Chebyshev polynomial, and damps the whole band.

    Regression for the first-step coefficient: an earlier ``2/theta`` first step (twice the correct
    scaled-Richardson ``1/theta``) made ``|P| > 1`` at low degree — the smoother *amplified* the
    highest-frequency modes instead of damping them. The correct three-term recurrence keeps
    ``max|P| < 1`` across ``[lo, hi]`` and matches the analytic min-max value at every degree.
    """
    lam_max, lo_frac = 1.0, 0.25
    lo, hi = lo_frac * lam_max, 1.05 * lam_max
    band = np.linspace(lo, hi, 400)
    for degree in (1, 2, 3, 4):
        realized = _chebyshev_propagation_polynomial(band, lam_max, degree, lo_frac)
        analytic = _scaled_chebyshev(band, lo, hi, degree)
        max_abs = float(np.max(np.abs(realized)))
        assert max_abs < 1.0  # every mode in the band is damped, not amplified
        assert np.allclose(realized, analytic, atol=1e-10)  # exactly the min-max polynomial


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


def test_convection_hierarchy_is_two_level_with_a_contractive_fine_smoother() -> None:
    """The convection hierarchy is two-level (a smoothed fine level + a single direct-solve coarse
    level), and the fine-level damped-Jacobi smoother genuinely contracts at high cell Peclet.

    A *deeper* Galerkin recursion would produce a coarse operator whose near-imaginary-axis
    eigenvalues no single-factor damped-Jacobi smoother can damp — a non-contractive (amplifying)
    coarse smoother. Keeping the hierarchy two-level removes that failure by construction: the only
    smoothed level is the diagonally dominant M-matrix fine level, and the coarse level is solved
    directly. (Deep, mesh-independent convection coarsening is the reduction-based lAIR hierarchy.)
    """
    owner, nb, visc, mdot, n, bd = _convective_grid(24, mu=1e-3, speed=1.0)  # cell Peclet ~40
    hierarchy = build_convection_hierarchy(owner, nb, visc, mdot, n, boundary_diagonal=bd)
    assert (
        len(hierarchy.levels) == 2
    )  # fine + one direct-solve coarse level; no smoothed coarse level

    # b = 0 has exact solution 0, so smoothing a random error must shrink it: a contraction, not the
    # amplification a deep Galerkin coarse smoother would apply at this cell Peclet.
    fine = hierarchy.levels[0]
    error = jnp.asarray(np.random.default_rng(0).standard_normal(n))
    smoothed = _jacobi_smooth(fine, jnp.zeros(n), error, sweeps=5, omega=0.8)
    assert float(jnp.linalg.norm(smoothed)) < float(jnp.linalg.norm(error))


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


# --- degenerate-mesh guards (fail loudly at build, not a silent inf in the frozen preconditioner) ---


def _triangle_with_isolated_cell():
    """Four cells, edges forming a triangle on cells 0-1-2; cell 3 has no incident edge.

    Cell 3's graph-Laplacian diagonal is therefore zero — the isolated-cell / disconnected-component
    degeneracy that would invert to ``inf`` in the frozen preconditioner.
    """
    owner = np.array([0, 1, 2])
    nb = np.array([1, 2, 0])
    return owner, nb, 4


def test_build_hierarchy_rejects_empty_mesh() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        build_hierarchy(np.array([], dtype=int), np.array([], dtype=int), 0)


def test_build_hierarchy_rejects_out_of_range_edge() -> None:
    with pytest.raises(ValueError, match="out of range"):
        build_hierarchy(np.array([0, 5]), np.array([1, 1]), 4)  # endpoint 5 >= n = 4


def test_isolated_cell_zero_diagonal_is_rejected_at_build() -> None:
    """A zero-diagonal (isolated) cell makes the smoothed-aggregation build fail loudly rather than
    bake ``1/0 = inf`` into the frozen operator and silently stall the runtime V-cycle."""
    owner, nb, n = _triangle_with_isolated_cell()
    with pytest.raises(ValueError, match="strictly positive"):
        build_smoothed_hierarchy(owner, nb, np.ones(len(owner)), n, pin=None)


def test_air_build_rejects_isolated_cell() -> None:
    """The reduction (lAIR) build guards its coarse operators' diagonals on the same footing."""
    owner, nb, n = _triangle_with_isolated_cell()
    with pytest.raises(ValueError, match="strictly positive"):
        build_convection_air_hierarchy(owner, nb, np.ones(len(owner)), np.zeros(len(owner)), n)


def test_boundary_stiffened_cell_is_allowed() -> None:
    """The diagonal is checked *after* boundary stiffness is folded in, so a cell that is closed off
    from the interior but carries a boundary coefficient is a valid operator — not a false positive."""
    owner, nb, n = _triangle_with_isolated_cell()
    boundary_diagonal = np.array([0.0, 0.0, 0.0, 1.0])  # cell 3 gets a boundary stiffness
    hierarchy = build_smoothed_hierarchy(
        owner, nb, np.ones(len(owner)), n, boundary_diagonal=boundary_diagonal
    )
    assert len(hierarchy.levels) >= 1  # builds without raising
