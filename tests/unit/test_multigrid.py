"""Unit tests for the algebraic-multigrid building blocks (smoothed aggregation, convection, lAIR).

These cover the frozen-operator V-cycle families: a mesh-independent smoothed-aggregation V-cycle for
the symmetric pressure Schur, its convection-diffusion variant, and the reduction-based (lAIR) cycle
for a strongly convection-dominated operator — each a fixed linear operator (a valid frozen left
preconditioner), plus the degenerate-mesh build guards.
"""

from __future__ import annotations

import itertools

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.mesh import structured_grid_2d
from aquaflux.solve.frozen_operator import convection_diffusion_operator, decouple_dof
from aquaflux.solve.multigrid import (
    _AGGREGATE_STATS,
    _cell_graph,
    _chebyshev_smooth,
    _CsrOperator,
    _dense_inverse,
    _fc_jacobi,
    _frozen_neighbourhoods,
    _galerkin_coarse,
    _jacobi_smooth,
    _jacobi_smooth_zero,
    _lair_restriction,
    _mis_aggregate,
    _one_point_interpolation,
    _reattach_to_adjacent_root,
    _restriction_neighbourhoods,
    _rs_split,
    _SparseLevel,
    _strength_classical,
    air_multigrid_solve,
    build_air_hierarchy,
    build_convection_hierarchy,
    build_smoothed_hierarchy,
    convection_multigrid_solve,
    refresh_air_hierarchy,
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

    The pin row/column is zeroed out of both the assembled operator (via ``decouple_dof``) and the
    dense operator, so the two share a null space and the V-cycle is a valid inner solve. Returns the cube-root of the residual-norm ratio over the last three cycles —
    the asymptotic per-cycle contraction factor. Shared by the mesh-independence and
    ordering-invariance checks.
    """
    a_pinned = decouple_dof(
        convection_diffusion_operator(owner, nb, np.ones(len(owner)), ncell), pin
    )
    hierarchy = build_smoothed_hierarchy(a_pinned)
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


# --- smoothed aggregation --------------------------------------------------------------


def _smoothed_residual_factor(n):
    """Geometric-mean V-cycle residual factor of the smoothed hierarchy on a singular Poisson."""
    owner, nb, ncell = _poisson(n)
    hierarchy = build_smoothed_hierarchy(
        convection_diffusion_operator(owner, nb, np.ones(owner.shape[0]), ncell)
    )
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
    hierarchy = build_smoothed_hierarchy(
        decouple_dof(convection_diffusion_operator(owner, nb, np.ones(owner.shape[0]), ncell), 0)
    )
    rng = np.random.default_rng(1)
    r1 = jnp.asarray(rng.standard_normal(ncell))
    r2 = jnp.asarray(rng.standard_normal(ncell))

    def solve(r):
        return smoothed_multigrid_solve(hierarchy, r, cycles=2)

    assert jnp.allclose(solve(1.5 * r1 - 0.5 * r2), 1.5 * solve(r1) - 0.5 * solve(r2), atol=1e-10)


def _anisotropic_poisson(nx: int, ny: int, aspect_ratio: float) -> sp.csr_matrix:
    """A uniformly-anisotropic Dirichlet Poisson: x-face coefficient ``dy/dx`` and y-face ``dx/dy``, so
    the strong (y) / weak (x) coefficient ratio is ``(dx/dy)**2 = aspect_ratio``. The canonical operator
    that defeats isotropic aggregation — which coarsens across the stiff (strongly-coupled) direction."""
    dx = 1.0 / nx
    dy = dx / np.sqrt(aspect_ratio)
    cx, cy = dy / dx, dx / dy
    owner, nb, coeff = [], [], []
    bd = np.zeros(nx * ny)
    for i in range(nx):
        for j in range(ny):
            c = i * ny + j
            if i + 1 < nx:
                owner.append(c), nb.append(c + ny), coeff.append(cx)
            if j + 1 < ny:
                owner.append(c), nb.append(c + 1), coeff.append(cy)
            if i in (0, nx - 1):
                bd[c] += 2 * cx
            if j in (0, ny - 1):
                bd[c] += 2 * cy
    return convection_diffusion_operator(
        np.asarray(owner), np.asarray(nb), np.asarray(coeff, float), nx * ny, boundary_diagonal=bd
    )


def test_strength_of_connection_aggregation_fixes_an_anisotropic_operator() -> None:
    """On a high-aspect-ratio operator the isotropic aggregation (the default) stalls, while the
    strength-of-connection aggregation reaches a 1% residual in a few V-cycles — the fix for a
    wall-resolved / skewed mesh. ``strength_threshold=0`` is exactly the isotropic build."""
    a = _anisotropic_poisson(48, 48, aspect_ratio=100.0)
    b = np.asarray(np.random.default_rng(0).standard_normal(a.shape[0]))
    b_norm = np.linalg.norm(b)

    def relative_after(hierarchy, cycles):
        x = np.asarray(smoothed_multigrid_solve(hierarchy, jnp.asarray(b), cycles=cycles))
        return float(np.linalg.norm(a @ x - b) / b_norm)

    plain = build_smoothed_hierarchy(a)  # isotropic aggregation on the full graph
    soc = build_smoothed_hierarchy(a, strength_threshold=0.25)  # aggregate along strong connections
    # The default is the isotropic build, bit-for-bit (same coarsening, same result).
    assert relative_after(plain, 5) == relative_after(
        build_smoothed_hierarchy(a, strength_threshold=0.0), 5
    )
    # Isotropic aggregation has not reached 1% even after 8 V-cycles; SoC reaches it in 3.
    assert relative_after(plain, 8) > 1e-2
    assert relative_after(soc, 3) < 1e-2


def test_avoid_singletons_reaches_the_aggressively_coarsened_level() -> None:
    """``avoid_singletons`` must apply to the squared-graph level too, not only the plain ones.

    A vertex reached late in the aggregation sweep can find every neighbour already claimed and then
    opens an aggregate holding only itself — a coarse unknown standing for one cell and coupling to
    almost nothing. The repair attaches such a vertex to an adjacent aggregate instead. It went to the
    plain branch only, which was invisible while the squared graph ran unfiltered (its aggregates are
    large enough that the case never arises); a strength threshold thins the graph enough to bring it
    back, so the two settings have to work together.

    Two passes are needed, which is why both levels are checked. Refusing to *open* a singleton
    handles the aggregation sweep; the reattachment pass that repairs the squared graph's reach runs
    afterwards, moves members between aggregates, and can strand a root the sweep had no way to
    foresee — so those are dissolved separately once the final assignment is known.
    """
    a = _anisotropic_poisson(24, 24, aspect_ratio=100.0)

    def singletons_per_level(avoid_singletons):
        _AGGREGATE_STATS.clear()
        build_convection_hierarchy(
            a,
            max_coarse=20,
            max_levels=3,
            mis_aggregation=True,
            aggressive_levels=1,
            strength_threshold=0.25,
            avoid_singletons=avoid_singletons,
        )
        return [level["singletons"] for level in _AGGREGATE_STATS]

    without = singletons_per_level(False)
    with_repair = singletons_per_level(True)

    assert without[0] > 0 and without[1] > 0  # both levels strand vertices when unrepaired
    assert with_repair == [0] * len(with_repair)  # and neither does once both passes run


def test_the_coarse_solve_is_a_factorization_where_the_operator_admits_one() -> None:
    """A pseudo-inverse is a singular value decomposition, and the coarse operator rarely needs one.

    The decomposition costs roughly an order of magnitude more than a factorization at the same size
    and the gap widens with it — measured at a 2000-equation coarse level, 1.03 s against 0.12 s, and at
    the 8192 this module permits, 102 s against 9.7 s. That is charged at every build and every
    mid-march refresh, and it was 59 % of a hierarchy setup.

    So the factorization is taken where it is valid and the decomposition kept for the case it covers.
    Both halves are pinned, because a fallback that never fires and one that always does look identical
    from the outside.
    """
    rng = np.random.default_rng(0)
    n = 200
    nonsingular = sp.csr_matrix(rng.normal(size=(n, n)) + n * np.eye(n))
    assert np.allclose(
        _dense_inverse(nonsingular), np.linalg.pinv(nonsingular.toarray()), atol=1e-12
    )

    # Exactly singular: the factorization cannot be used at all.
    singular = np.eye(n)
    singular[7, 7] = 0.0
    assert np.allclose(_dense_inverse(sp.csr_matrix(singular)), np.linalg.pinv(singular))

    # Numerically singular, which is the case a raised/not-raised test misses: the factorization does
    # NOT fail here, it returns finite nonsense, so the fallback has to be chosen on conditioning.
    almost = np.eye(n)
    almost[7, 7] = 1e-18
    assert np.allclose(_dense_inverse(sp.csr_matrix(almost)), np.linalg.pinv(almost))


def test_dense_coarse_solve_guard_rejects_an_oversized_coarsest_level() -> None:
    """The coarsest level is inverted densely, so its size has to stay bounded as the mesh grows.

    Coarsening stops on whichever of the two limits is reached first. When the level cap wins, nothing
    bounds the coarse grid at all — it then scales with the mesh, and the dense inverse is quadratic to
    store and cubic to build, so a large enough case turns into a multi-gigabyte allocation with no
    indication of why. Fail with the two ways out instead.
    """
    a = _anisotropic_poisson(128, 128, aspect_ratio=1.0)  # 16384 dofs, above the dense limit
    with pytest.raises(ValueError, match="inverted densely"):
        build_convection_hierarchy(a, max_coarse=20, max_levels=1)


def _chebyshev_propagation_polynomial(eigenvalues, lam_max, degree, lo_frac):
    """Sample the smoother's error-propagation polynomial ``P(mu)`` at the given eigenvalues.

    Builds a diagonal level whose preconditioned operator is ``D^-1 A = diag(eigenvalues)`` (unit
    diagonal, ``A`` diagonal), then smooths ``x0 = 1`` against ``b = 0`` (so the exact solution is
    ``0`` and ``x`` after smoothing equals ``P(mu)`` mode by mode). Returns ``P`` at each eigenvalue.
    """
    n = len(eigenvalues)
    level = _SparseLevel(
        n=n,
        operator=_CsrOperator.from_scipy(sp.diags(np.asarray(eigenvalues)).tocsr()),
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


# --- precomputed inverse diagonals (issue #110) -----------------------------------------


def test_sparse_level_inv_diagonal_matches_the_reciprocal_diagonal() -> None:
    """``inv_diagonal`` is exactly ``1.0 / diagonal``, derived once at construction."""
    owner, nb, ncell = _poisson(8)
    hierarchy = build_smoothed_hierarchy(
        convection_diffusion_operator(owner, nb, np.ones(owner.shape[0]), ncell)
    )
    for level in hierarchy.levels:
        np.testing.assert_array_equal(
            np.asarray(level.inv_diagonal), np.asarray(1.0 / level.diagonal)
        )


def test_air_level_inv_diagonal_matches_the_reciprocal_diagonal() -> None:
    """The lAIR level carries the same derived reciprocal, for the FC-Jacobi smoother."""
    owner, nb, visc, mdot, n, bd = _convective_grid(8, mu=1e-2, speed=1.0)
    hierarchy = build_air_hierarchy(_operator(owner, nb, visc, mdot, n, bd))
    for level in hierarchy.levels:
        np.testing.assert_array_equal(
            np.asarray(level.inv_diagonal), np.asarray(1.0 / level.diagonal)
        )


def test_smoother_bodies_never_close_over_the_raw_diagonal() -> None:
    """Chebyshev, damped-Jacobi and FC-Jacobi read the precomputed reciprocal, never re-divide it.

    Regression for the frozen-diagonal recomputation: each smoother used to compute
    ``1.0 / level.diagonal`` (or, for FC-Jacobi, ``omega * mask * (1.0 / level.diagonal)``) inside its
    own body on every call, even though the diagonal is a build-time constant. A smoother traced in
    isolation closes over ``level``'s arrays as jaxpr constants, so checking which of ``diagonal`` /
    ``inv_diagonal`` actually appears among them is a direct check of the acceptance criterion ("no
    n-element divide in the smoother bodies") that does not confuse this with the other, unrelated
    divides these smoothers legitimately still perform (e.g. Chebyshev's own scalar-derived step
    scaling) — a blanket "no ``div`` primitive at all" check would flag those too.
    """
    owner, nb, ncell = _poisson(6)
    smoothed = build_smoothed_hierarchy(
        convection_diffusion_operator(owner, nb, np.ones(owner.shape[0]), ncell)
    )
    fine = smoothed.levels[0]
    b, x = jnp.zeros(fine.n), jnp.ones(fine.n)

    chebyshev_jaxpr = jax.make_jaxpr(lambda b, x: _chebyshev_smooth(fine, b, x, 3, 0.25))(b, x)
    jacobi_jaxpr = jax.make_jaxpr(lambda b, x: _jacobi_smooth(fine, b, x, 2, 0.8))(b, x)
    jacobi_zero_jaxpr = jax.make_jaxpr(lambda b: _jacobi_smooth_zero(fine, b, 2, 0.8))(b)
    for name, jaxpr in (
        ("chebyshev", chebyshev_jaxpr),
        ("jacobi", jacobi_jaxpr),
        ("jacobi_zero", jacobi_zero_jaxpr),
    ):
        consts = jaxpr.consts
        assert not any(c is fine.diagonal for c in consts), f"{name} closes over the raw diagonal"
        assert any(c is fine.inv_diagonal for c in consts), f"{name} does not read inv_diagonal"

    o, nb2, visc, mdot, n, bd = _convective_grid(6, mu=1e-2, speed=1.0)
    air = build_air_hierarchy(_operator(o, nb2, visc, mdot, n, bd))
    air_fine = air.levels[0]
    b2, x2 = jnp.zeros(air_fine.n), jnp.ones(air_fine.n)
    fc_jacobi_jaxpr = jax.make_jaxpr(lambda b, x: _fc_jacobi(air_fine, b, x, 2, 2, 0.8))(b2, x2)
    consts = fc_jacobi_jaxpr.consts
    assert not any(c is air_fine.diagonal for c in consts), "fc_jacobi closes over the raw diagonal"
    assert any(c is air_fine.inv_diagonal for c in consts), "fc_jacobi does not read inv_diagonal"


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


def _operator(owner, nb, visc, mdot, n, bd):
    """The assembled frozen convection-diffusion operator (flow-side builder), as a scipy CSR matrix."""
    return convection_diffusion_operator(owner, nb, visc, n, flux=mdot, boundary_diagonal=bd)


def _dense(owner, nb, visc, mdot, n, bd):
    return jnp.asarray(np.asarray(_operator(owner, nb, visc, mdot, n, bd).toarray()))


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
    hierarchy = build_convection_hierarchy(_operator(owner, nb, visc, mdot, n, bd))
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
    hierarchy = build_convection_hierarchy(_operator(owner, nb, visc, mdot, n, bd))
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
    hierarchy = build_convection_hierarchy(_operator(owner, nb, visc, mdot, n, bd))
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


def _air_contractions(restriction_theta=None) -> list[float]:
    """Per-cycle lAIR contraction on the strongly-convective operator, at two mesh sizes."""
    contractions = []
    for n in (24, 48):  # cell Peclet ~40; a 4x change in cell count
        owner, nb, visc, mdot, ncell, bd = _convective_grid(n, mu=1e-3, speed=1.0)
        hierarchy = build_air_hierarchy(
            _operator(owner, nb, visc, mdot, ncell, bd), restriction_theta=restriction_theta
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
    return contractions


def test_air_v_cycle_contracts_strongly_at_high_peclet() -> None:
    """The reduction-based (lAIR) V-cycle converges the strongly-convective operator in a handful of
    cycles, and degrades only mildly as the mesh refines — the Peclet-robust behaviour the two-level
    aggregation method (and deep Galerkin recursion) lack. For a convection-dominated operator the
    approximate ideal restriction is nearly exact, so a few cycles behave close to a direct solve.

    **The bars here are calibrated against the restriction neighbourhood the method actually uses, and
    that is worth stating because they used to be calibrated against something unaffordable.** An
    earlier form of this test required a contraction below ``0.1`` and a spread below ``0.05`` across
    the refinement. Those held only while the neighbourhood walk ran over the operator's whole sparsity
    pattern, which makes each coarse operator denser than the one above it: on a three-dimensional
    block that recursion never finished building. Walking strong connections instead — what
    ``restriction_theta`` selects, and what the method is specified as — costs real accuracy, and no
    affordable threshold reaches the old bars. The companion test below pins that trade directly.
    """
    contractions = _air_contractions()
    assert max(contractions) < 0.25  # measured 0.054 and 0.166
    assert (
        abs(contractions[1] - contractions[0]) < 0.15
    )  # degrades with refinement, but not sharply


def test_a_wider_restriction_neighbourhood_buys_accuracy_and_costs_density() -> None:
    """The accuracy-against-cost trade ``restriction_theta`` controls, pinned in both directions.

    Admitting every stored connection (``0.0``, which walks the full sparsity pattern) gives a markedly
    better V-cycle than the default's strong-connection walk — and pays for it in coarse-operator
    density, which is why it is not the default despite being the more accurate method. The point of
    pinning both halves is that either one alone reads as a reason to move the default.
    """
    default = _air_contractions()
    widest = _air_contractions(restriction_theta=0.0)
    assert max(widest) < 0.1 * max(default), "the wider neighbourhood is no longer more accurate"

    owner, nb, visc, mdot, ncell, bd = _convective_grid(48, mu=1e-3, speed=1.0)
    operator = _operator(owner, nb, visc, mdot, ncell, bd)

    def peak_density(**kwargs):
        levels = build_air_hierarchy(operator, **kwargs).levels
        return max(level.operator.data.shape[0] / level.n for level in levels)

    assert peak_density() < 0.5 * peak_density(restriction_theta=0.0), (
        "the wider neighbourhood is no longer denser, so the default is paying accuracy for nothing"
    )


def test_air_multigrid_is_linear_and_transposable() -> None:
    """A fixed cycle/sweep count makes the lAIR V-cycle a constant linear operator that transposes
    cleanly (``R != Pᵀ``), so it is a valid frozen left preconditioner for the forward solve and its
    ``M^T`` adjoint."""
    owner, nb, visc, mdot, n, bd = _convective_grid(12, mu=1e-2, speed=1.0)
    hierarchy = build_air_hierarchy(_operator(owner, nb, visc, mdot, n, bd))
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


# --- lAIR setup internals (strength graph, C/F split, interpolation, restriction) --------
#
# The setup runs once off the jit path in scipy/numpy, so it is exercised here directly on small
# explicit matrices — including the degenerate shapes a real hierarchy only hits occasionally (a row
# with nothing strong to depend on, a split that cannot coarsen, an F-point with no C-neighbour, an
# empty or singular local solve), which a mesh-driven end-to-end test cannot reach on demand.


def _upwind_chain(n, *, diffusion=1.0, flux=4.0):
    """A nonsymmetric first-order-upwind convection-diffusion chain on ``n`` cells, as a CSR matrix.

    Row ``i`` couples strongly to its upwind neighbour ``i-1`` (weight ``diffusion + flux``) and
    weakly to the downwind ``i+1`` (weight ``diffusion``), over a diagonally dominant positive
    diagonal — the flow-aligned M-matrix shape the reduction setup targets, small enough to compare
    against dense linear algebra.
    """
    a = np.zeros((n, n))
    for i in range(n):
        a[i, i] = 2.0 * diffusion + flux
        if i > 0:
            a[i, i - 1] = -(diffusion + flux)
        if i < n - 1:
            a[i, i + 1] = -diffusion
    return sp.csr_matrix(a)


def test_strength_graph_marks_the_flow_aligned_couplings() -> None:
    """The classical strength graph keeps a row's couplings within ``theta`` of its largest one — for
    an upwind operator, the upwind (flow-aligned) neighbour, not the much weaker downwind one."""
    n = 5
    strength = np.asarray(_strength_classical(_upwind_chain(n), 0.25).toarray())
    assert np.all(np.diag(strength) == 0.0)  # a point never depends on itself
    for i in range(1, n):
        assert strength[i, i - 1] == 1.0  # upwind coupling (weight 5) is the row maximum
    for i in range(1, n - 1):
        assert strength[i, i + 1] == 0.0  # downwind (weight 1) is below 0.25 * 5
    # Row 0 has no upwind neighbour, so its lone downwind coupling *is* its maximum and is strong.
    assert strength[0, 1] == 1.0


def test_strength_graph_skips_rows_with_no_usable_off_diagonal() -> None:
    """Two rows carry no strength at all: one with no off-diagonal entry, and one whose stored
    off-diagonals are exactly zero. Explicit stored zeros are not hypothetical — the Galerkin ``R A P``
    product that builds each coarser level can produce them."""
    a = sp.csr_matrix(
        ([1.0, 1.0, 0.0, 1.0], ([0, 1, 1, 2], [0, 1, 2, 2])), shape=(3, 3)
    )  # row 0: diagonal only; row 1: a stored *zero* off-diagonal
    assert a.nnz == 4  # the explicit zero survived assembly
    assert _strength_classical(a, 0.25).nnz == 0


def test_rs_split_leaves_every_fine_point_strongly_dependent_on_a_coarse_point() -> None:
    """The splitting decides every point and produces a real coarsening whose C-points cover the
    strong connections: each F-point depends strongly on at least one C-point, which is what the
    one-point interpolation and the local restriction solves both rely on."""
    n = 12
    strength = _strength_classical(_upwind_chain(n), 0.25)
    split = _rs_split(strength)
    assert set(np.unique(split)) <= {0, 1}  # nothing left undecided
    assert 0 in split and 1 in split  # a real coarsening, neither all-C nor all-F
    s = np.asarray(strength.toarray())
    for i in np.where(split == 0)[0]:
        assert np.any(s[i] * (split == 1))


def test_air_build_stops_when_the_split_cannot_coarsen() -> None:
    """An operator with no off-diagonal couplings has an empty strength graph, so the splitting makes
    every point coarse. Rather than recurse on a hierarchy that would never shrink, the build stops
    and solves that level directly."""
    n = 30  # above max_coarse, so the build would otherwise try to coarsen
    diagonal = np.linspace(1.0, 2.0, n)
    hierarchy = build_air_hierarchy(sp.csr_matrix(np.diag(diagonal)))
    assert len(hierarchy.levels) == 1
    level = hierarchy.levels[0]
    assert level.r_row is None and level.p_row is None  # coarsest: a direct solve, no transfers
    b = np.random.default_rng(0).standard_normal(n)
    x = air_multigrid_solve(hierarchy, jnp.asarray(b), cycles=1)
    assert np.allclose(np.asarray(x), b / diagonal)


def test_one_point_interpolation_leaves_a_fine_point_without_a_coarse_neighbour_empty() -> None:
    """Each F-point takes its strongest C-neighbour and each C-point is injected; an F-point with no
    C-neighbour at all interpolates nothing, giving a zero row rather than an invalid entry."""
    n = 5
    split = np.array([1, 0, 0, 0, 1])  # cell 2 sits between two F-points -> no C-neighbour
    p = np.asarray(_one_point_interpolation(_upwind_chain(n), split).toarray())
    assert p.shape == (n, 2)
    assert np.array_equal(p[0], [1.0, 0.0])  # C-point injection
    assert np.array_equal(p[4], [0.0, 1.0])
    assert np.array_equal(p[1], [1.0, 0.0])  # F-point 1 -> its only C-neighbour, cell 0
    assert np.array_equal(p[3], [0.0, 1.0])  # F-point 3 -> cell 4
    assert np.array_equal(p[2], [0.0, 0.0])  # no C-neighbour -> zero row


def test_lair_restriction_handles_an_empty_and_a_singular_local_solve() -> None:
    """Two degenerate local solves, on a matrix built to force both.

    Coarse point 0's F-neighbourhood ``{1, 2}`` gives an exactly singular ``A_ff``, so the local solve
    falls back to the minimum-norm least-squares solution instead of failing; coarse point 3 has no
    F-neighbour at all, so its restriction row is the identity entry alone.
    """
    a = sp.csr_matrix(
        np.array(
            [
                [4.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],  # rows 1 and 2 agree over columns {1, 2}:
                [1.0, 1.0, 1.0, 0.0],  # A_ff = [[1, 1], [1, 1]] is exactly singular
                [-1.0, 0.0, 0.0, 4.0],  # cell 3's only off-diagonal reaches a C-point
            ]
        )
    )
    split = np.array([1, 0, 0, 1])
    strength = _strength_classical(a, 0.25)
    r = np.asarray(_lair_restriction(a, split, degree=1, strength=strength).toarray())
    assert r.shape == (2, 4)
    assert np.all(np.isfinite(r))
    # Minimum-norm least-squares solution of [[1, 1], [1, 1]] z = [-1, -1].
    assert np.allclose(r[0], [1.0, -0.5, -0.5, 0.0])
    assert np.allclose(r[1], [0.0, 0.0, 0.0, 1.0])  # empty F-neighbourhood -> identity entry only


def test_lair_restriction_reproduces_the_exact_schur_complement() -> None:
    """With an F-neighbourhood wide enough to reach every F-point it couples to, the local
    approximate-ideal solve *is* the ideal restriction ``R = [-A_cf A_ff⁻¹, I]``: it annihilates the
    F-columns of ``R A``, so the Galerkin coarse operator ``R A P`` is exactly the Schur complement
    ``A_cc - A_cf A_ff⁻¹ A_fc`` — the coarse action of the fine operator, reproduced exactly.

    "Wide enough" means **every** coupling, not every strong one, which is why the walk here runs at a
    zero threshold rather than at the one that chose the C/F split. Row ``g``'s weak downwind coupling
    is still a term of ``A[g, F]`` that the ideal restriction has to cancel, so a neighbourhood that
    admits only strong connections leaves it in ``R A`` however far the walk is allowed to reach. That
    is exactly the accuracy the default trades away for a coarse operator that stays sparse; this test
    fixes the upper end of the trade, and
    :func:`test_a_wider_restriction_neighbourhood_buys_accuracy_and_costs_density` measures the middle.
    """
    n = 9
    a = _upwind_chain(n)
    split = _rs_split(_strength_classical(a, 0.25))
    strength = _strength_classical(
        a, 0.0
    )  # every coupling -- the ideal restriction's neighbourhood
    coarse, fine = np.where(split == 1)[0], np.where(split == 0)[0]
    assert len(fine) > 0 and len(coarse) > 0
    p = _one_point_interpolation(a, split)
    r = _lair_restriction(a, split, degree=n, strength=strength)
    assert p.shape == (n, len(coarse))
    assert r.shape == (len(coarse), n)

    dense = a.toarray()
    assert np.allclose(np.asarray((r @ a).toarray())[:, fine], 0.0, atol=1e-12)
    schur = dense[np.ix_(coarse, coarse)] - dense[np.ix_(coarse, fine)] @ np.linalg.solve(
        dense[np.ix_(fine, fine)], dense[np.ix_(fine, coarse)]
    )
    assert np.allclose(np.asarray((r @ a @ p).toarray()), schur, atol=1e-10)


# --- degenerate-mesh guards (fail loudly at build, not a silent inf in the frozen preconditioner) ---


def _triangle_with_isolated_cell():
    """Four cells, edges forming a triangle on cells 0-1-2; cell 3 has no incident edge.

    Cell 3's graph-Laplacian diagonal is therefore zero — the isolated-cell / disconnected-component
    degeneracy that would invert to ``inf`` in the frozen preconditioner.
    """
    owner = np.array([0, 1, 2])
    nb = np.array([1, 2, 0])
    return owner, nb, 4


def test_assembly_rejects_empty_mesh() -> None:
    """The graph is validated where it is consumed — at assembly, before any coarsening."""
    with pytest.raises(ValueError, match="at least one cell"):
        convection_diffusion_operator(
            np.array([], dtype=int), np.array([], dtype=int), np.array([]), 0
        )


def test_assembly_rejects_out_of_range_edge() -> None:
    with pytest.raises(ValueError, match="out of range"):
        # endpoint 5 >= n = 4
        convection_diffusion_operator(np.array([0, 5]), np.array([1, 1]), np.ones(2), 4)


def test_assembly_rejects_mismatched_edge_arrays() -> None:
    with pytest.raises(ValueError, match="same shape"):
        convection_diffusion_operator(np.array([0, 1]), np.array([1]), np.ones(2), 4)


def test_isolated_cell_zero_diagonal_is_rejected_at_build() -> None:
    """A zero-diagonal (isolated) cell makes the smoothed-aggregation build fail loudly rather than
    bake ``1/0 = inf`` into the frozen operator and silently stall the runtime V-cycle."""
    owner, nb, n = _triangle_with_isolated_cell()
    with pytest.raises(ValueError, match="strictly positive"):
        build_smoothed_hierarchy(convection_diffusion_operator(owner, nb, np.ones(len(owner)), n))


def _distance_two_operator(n_side: int) -> sp.csr_matrix:
    """A convection-dominated operator on a three-dimensional grid, with a **distance-2** pattern.

    The shape a coupled block Jacobian actually has: a second-order discretization couples a cell to
    its neighbours' neighbours, so the operator stores tens of entries per row rather than the seven a
    nearest-neighbour stencil gives. That density is what a reduction hierarchy's coarsening has to
    survive, and a seven-point fixture cannot exercise it.
    """
    index = np.arange(n_side**3).reshape(n_side, n_side, n_side)
    owner, neighbour = [], []
    for axis in range(3):
        rolled = np.moveaxis(index, axis, 0)
        owner.append(rolled[:-1].ravel())
        neighbour.append(rolled[1:].ravel())
    owner, neighbour = np.concatenate(owner), np.concatenate(neighbour)
    n = n_side**3
    flux = np.full(owner.size, 4.0)  # convection dominated, so the strength graph is directional
    nearest = sp.csr_matrix(
        (-(1.0 + np.maximum(flux, 0.0)), (neighbour, owner)), shape=(n, n)
    ) + sp.csr_matrix((-(1.0 + np.maximum(-flux, 0.0)), (owner, neighbour)), shape=(n, n))
    diagonal = np.zeros(n)
    np.add.at(diagonal, owner, 1.0 + np.maximum(flux, 0.0))
    np.add.at(diagonal, neighbour, 1.0 + np.maximum(-flux, 0.0))
    reach_two = abs(nearest) @ abs(nearest)
    reach_two.data[:] = 1e-3  # a weak but stored distance-2 coupling
    return (nearest + reach_two + sp.diags(diagonal + 1.0)).tocsr()


def test_air_coarse_operators_stay_sparse_on_a_distance_two_operator() -> None:
    """The Galerkin recursion must not densify as it coarsens — the defect this method is one step from.

    A reduction hierarchy's coarse operator ``R A P`` inherits the product of the transfer patterns, so
    an over-wide restriction neighbourhood does not cost once: it makes the next level denser, which
    widens the next neighbourhood, and so on. The failure has no symptom other than a build that stops
    finishing, because the growth lands in dense local solves rather than in anything that raises —
    which is why it is pinned here on an operator dense enough to show it, rather than left to be
    rediscovered on a real mesh.
    """
    hierarchy = build_air_hierarchy(_distance_two_operator(10))
    density = [level.operator.data.shape[0] / level.n for level in hierarchy.levels]
    assert len(density) > 2, "too few levels to say anything about how the recursion behaves"
    growth = [after / before for before, after in itertools.pairwise(density)]
    assert max(growth) < 2.0, f"the coarse operators are densifying: {density}"


def test_air_build_rejects_a_densifying_hierarchy_rather_than_grinding() -> None:
    """A neighbourhood too wide for the operator fails loudly and names both ways out.

    ``restriction_theta=0.0`` admits every stored connection, which on a distance-2 operator is exactly
    the recursion the test above forbids. The guard exists because the alternative is not a wrong
    answer but an absent one: the build simply runs until someone stops it.
    """
    with pytest.raises(ValueError, match="densifies as it coarsens"):
        build_air_hierarchy(_distance_two_operator(10), restriction_theta=0.0)


def test_air_build_rejects_isolated_cell() -> None:
    """The reduction (lAIR) build guards its coarse operators' diagonals on the same footing."""
    owner, nb, n = _triangle_with_isolated_cell()
    with pytest.raises(ValueError, match="strictly positive"):
        build_air_hierarchy(
            _operator(owner, nb, np.ones(len(owner)), np.zeros(len(owner)), n, np.zeros(n))
        )


def test_boundary_stiffened_cell_is_allowed() -> None:
    """The diagonal is checked *after* boundary stiffness is folded in, so a cell that is closed off
    from the interior but carries a boundary coefficient is a valid operator — not a false positive."""
    owner, nb, n = _triangle_with_isolated_cell()
    boundary_diagonal = np.array([0.0, 0.0, 0.0, 1.0])  # cell 3 gets a boundary stiffness
    hierarchy = build_smoothed_hierarchy(
        convection_diffusion_operator(
            owner, nb, np.ones(len(owner)), n, boundary_diagonal=boundary_diagonal
        )
    )
    assert len(hierarchy.levels) >= 1  # builds without raising


def _chain_operator(n, flux_scale, diffusivity):
    """A 1-D chain convection-diffusion operator: same graph, caller-chosen coefficients."""
    owner, nb = np.arange(n - 1), np.arange(1, n)
    return convection_diffusion_operator(
        owner, nb, diffusivity, n, flux=flux_scale * np.ones(n - 1)
    )


def test_aggregation_hierarchy_structure_is_value_independent() -> None:
    """Re-deriving the hierarchy at a different operator on the same graph gives the same structure.

    The aggregation reads only the sparsity pattern (``_aggregate`` takes ``owner``/``nb``/``n``, never
    the coefficients), so on a fixed mesh the aggregates, the coarse sizes and every array shape are
    invariant under a change of viscosity or mass flux — only the *values* differ. This is what makes a
    frozen hierarchy refreshable at a developed state instead of rebuildable, and it is asserted here
    because the no-recompile refresh below depends on it.
    """
    n = 600
    cold = build_convection_hierarchy(_chain_operator(n, 0.01, np.ones(n - 1)))
    developed = build_convection_hierarchy(
        _chain_operator(n, 50.0, np.linspace(1.0, 1000.0, n - 1))
    )

    assert len(cold.levels) == len(developed.levels)
    for lo, hi in zip(cold.levels, developed.levels, strict=True):
        assert (lo.n, lo.n_coarse) == (hi.n, hi.n_coarse)  # static metadata
        assert lo.operator.data.shape == hi.operator.data.shape  # operator sparsity
        assert lo.diagonal.shape == hi.diagonal.shape
        if lo.p_val is not None:
            assert lo.p_val.shape == hi.p_val.shape  # prolongation sparsity
    # ...and the values really do differ, so the invariance above is not a trivial no-op.
    assert not np.allclose(
        np.asarray(cold.levels[0].operator.data), np.asarray(developed.levels[0].operator.data)
    )


def test_refreshing_a_hierarchy_is_a_compilation_cache_hit() -> None:
    """Swapping in a hierarchy rebuilt at another operator must not retrace the jitted V-cycle.

    Only ``n``/``n_coarse`` are static (they size the sparse matvec); the operator values, diagonal,
    ``lam_max``, prolongation values and coarse inverse are all traced leaves. So a hierarchy passed as
    a jit *argument* keeps one compiled V-cycle across a refresh — which is what lets the frozen
    preconditioner track a developing flow without paying a recompile per refresh.
    """
    n = 600
    cold = build_convection_hierarchy(_chain_operator(n, 0.01, np.ones(n - 1)))
    developed = build_convection_hierarchy(
        _chain_operator(n, 50.0, np.linspace(1.0, 1000.0, n - 1))
    )
    traces = []

    @jax.jit
    def apply(hierarchy, b):
        traces.append(1)  # appended once per trace, not per call
        return convection_multigrid_solve(hierarchy, b, cycles=1)

    b = jnp.asarray(np.random.default_rng(0).normal(size=n))
    x_cold = apply(cold, b)
    x_cold.block_until_ready()
    assert len(traces) == 1

    apply(cold, b).block_until_ready()  # same hierarchy: no retrace
    x_developed = apply(developed, b)  # refreshed values: still no retrace
    x_developed.block_until_ready()
    assert len(traces) == 1, "refreshing the hierarchy values retraced the jitted V-cycle"

    # The refreshed values genuinely change the preconditioner (else the cache hit is meaningless).
    assert not np.allclose(np.asarray(x_cold), np.asarray(x_developed))


def test_lair_structure_is_value_dependent_unlike_aggregation() -> None:
    """A reduction (lAIR) hierarchy is NOT refreshable on a fixed structure — its split reads values.

    The aggregation coarsening reads only the graph, so re-deriving it at a new operator is shape-stable
    (above). lAIR instead picks its coarse points from a strength graph thresholded on ``|A_ij|``
    (:func:`_strength_classical`), so changing the coefficients changes the C/F split and, from some
    level down, every shape. That is why a cheap lAIR refresh has to *reuse* the reference's frozen split
    rather than re-derive it: a from-scratch rebuild is a different jit signature, so it would recompile
    even though the values are all that changed. Pinned so the asymmetry is not mistaken for a bug.
    """
    n = 600
    cold = build_air_hierarchy(_chain_operator(n, 0.01, np.ones(n - 1)))
    developed = build_air_hierarchy(_chain_operator(n, 50.0, np.linspace(1.0, 1000.0, n - 1)))

    shapes = [(lv.n, lv.n_coarse, lv.operator.data.shape) for lv in cold.levels]
    developed_shapes = [(lv.n, lv.n_coarse, lv.operator.data.shape) for lv in developed.levels]
    assert shapes != developed_shapes, (
        "lAIR coarsening happened to be shape-stable here; the refresh-by-reusing-the-split "
        "requirement is justified by its value dependence, so re-check _strength_classical"
    )


def test_refresh_air_hierarchy_keeps_the_structure_and_is_a_cache_hit() -> None:
    """Refreshing lAIR on its frozen coarsening changes only values — so it is a jit cache hit.

    A plain rebuild at a new operator changes lAIR's C/F split and shapes (above), which would force a
    recompile of the solve the preconditioner accelerates. Reusing the frozen split, prolongation and
    restriction neighbourhoods — and re-solving only the restriction's weights — keeps every shape, so
    the compiled V-cycle is reused.
    """
    n = 600
    cold_operator = _chain_operator(n, 0.01, np.ones(n - 1))
    developed_operator = _chain_operator(n, 50.0, np.linspace(1.0, 1000.0, n - 1))
    cold = build_air_hierarchy(cold_operator)
    refreshed = refresh_air_hierarchy(cold, developed_operator)

    # Structure preserved exactly (a from-scratch rebuild does NOT preserve it -- see the test above).
    assert len(refreshed.levels) == len(cold.levels)
    for old, new in zip(cold.levels, refreshed.levels, strict=True):
        assert (old.n, old.n_coarse) == (new.n, new.n_coarse)
        assert old.operator.data.shape == new.operator.data.shape
    # ...but the values genuinely moved to the new operator.
    assert not np.allclose(
        np.asarray(cold.levels[0].operator.data), np.asarray(refreshed.levels[0].operator.data)
    )

    traces = []

    @jax.jit
    def apply(hierarchy, b):
        traces.append(1)
        return air_multigrid_solve(hierarchy, b, cycles=1)

    b = jnp.asarray(np.random.default_rng(0).normal(size=n))
    apply(cold, b).block_until_ready()
    assert len(traces) == 1
    apply(refreshed, b).block_until_ready()
    assert len(traces) == 1, "refreshing the lAIR hierarchy retraced the jitted V-cycle"


def test_refreshed_air_hierarchy_preconditions_the_new_operator() -> None:
    """The refreshed hierarchy must actually precondition the *new* operator, not just keep its shape.

    Reusing the build state's C/F split, prolongation and restriction neighbourhoods is a deliberate
    trade (any valid choice of those gives a valid preconditioner), so the test is that one refreshed
    V-cycle reduces the new operator's residual substantially better than the stale hierarchy does —
    i.e. the re-solved restriction weights really track the new coefficients.
    """
    n = 600
    cold_operator = _chain_operator(n, 0.01, np.ones(n - 1))
    developed_operator = _chain_operator(n, 50.0, np.linspace(1.0, 1000.0, n - 1))
    cold = build_air_hierarchy(cold_operator)
    refreshed = refresh_air_hierarchy(cold, developed_operator)

    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.normal(size=n))
    a_dev = jnp.asarray(developed_operator.toarray())

    def residual(hierarchy):
        x = air_multigrid_solve(hierarchy, b, cycles=1)
        return float(jnp.linalg.norm(a_dev @ x - b) / jnp.linalg.norm(b))

    stale_residual, fresh_residual = residual(cold), residual(refreshed)
    assert fresh_residual < stale_residual, (
        f"refreshed V-cycle ({fresh_residual:.3e}) did not beat the stale one ({stale_residual:.3e})"
    )


def test_galerkin_coarse_pattern_does_not_move_when_a_weight_becomes_zero() -> None:
    """The coarse operator's stored shape must not depend on the transfer operators' *values*.

    ``scipy`` drops an exactly-zero product entry, and lAIR generates those structurally: a degree-2
    neighbourhood can hold an F-point that the C-point's row does not couple to, whose restriction
    weight is then exactly zero. Which entries that hits moves with the operator's values, so without
    this the same mesh at a developed state gives a coarse operator of a slightly different size — and
    :func:`refresh_air_hierarchy`, whose entire purpose is to hold the jit signature, fails its own
    structure check on a perfectly legitimate operator. Measured on a 46080-row operator, 14 such
    weights moved the level-1 entry count by 160.
    """
    # Minimal triple where one zeroed weight really does cost the plain product an entry: with an
    # identity operator and prolongation, each product entry has exactly one contribution.
    identity = sp.identity(3, format="csr")
    restriction = sp.csr_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    zeroed = restriction.tocsr(copy=True)
    zeroed.data[4] = 0.0  # as a local solve over an uncoupled F-point would produce

    assert (zeroed @ identity @ identity).tocsr().nnz == 8, (
        "premise: the plain product loses an entry"
    )
    stable = _galerkin_coarse(zeroed, identity, identity)
    assert stable.nnz == 9, "the coarse shape moved with the restriction's values"
    assert np.allclose(stable.toarray(), zeroed.toarray())

    # On a real triple the pattern is the structural one, and the values are the product's own.
    n = 40
    a = _chain_operator(n, 0.01, np.ones(n - 1))
    strength = _strength_classical(a, 0.25)
    split = _rs_split(strength)
    r = _lair_restriction(a, split, 2, strength)
    prolongation = _one_point_interpolation(a, split)

    def ones_like(matrix):
        skeleton = matrix.tocsr(copy=True)
        skeleton.data = np.ones_like(skeleton.data)
        return skeleton

    coarse = _galerkin_coarse(r, a, prolongation)
    structural = (ones_like(r) @ ones_like(a) @ ones_like(prolongation)).tocsr()
    assert coarse.nnz == structural.nnz
    assert np.array_equal(coarse.indptr, structural.indptr)
    assert np.array_equal(coarse.indices, structural.indices)
    assert np.allclose(coarse.toarray(), (r @ a @ prolongation).toarray())


def test_frozen_neighbourhoods_recover_the_walk_that_built_the_restriction() -> None:
    """A refresh reuses the F-neighbourhoods rather than re-walking, so they must round-trip exactly.

    This is what makes :func:`refresh_air_hierarchy` structure-preserving **by construction** rather
    than by the check it also runs afterwards, and it is a quiet invariant: recovering the wrong
    neighbourhoods would still produce a plausible restriction, just of a different operator's shape,
    and the failure would surface later as a mismatched-structure error with no obvious cause.
    """
    n = 600
    operator = _chain_operator(n, 0.01, np.ones(n - 1))
    strength = _strength_classical(operator, 0.25)
    split = _rs_split(strength)
    walked = _restriction_neighbourhoods(strength, split, degree=2)

    level = build_air_hierarchy(operator).levels[0]
    recovered = _frozen_neighbourhoods(level)

    assert np.array_equal(recovered[0], walked[0]), "neighbourhood offsets did not round-trip"
    assert np.array_equal(recovered[1], walked[1]), "neighbourhood members did not round-trip"
    # Not vacuous: the level must actually hold neighbourhoods for this to have checked anything.
    assert walked[1].size > 0


def test_refresh_air_hierarchy_rejects_a_mismatched_operator() -> None:
    """A refresh that would change any shape raises instead of silently returning a recompiling one."""
    cold = build_air_hierarchy(_chain_operator(600, 0.01, np.ones(599)))
    with pytest.raises(ValueError, match="refresh_air_hierarchy"):
        refresh_air_hierarchy(cold, _chain_operator(400, 0.01, np.ones(399)))


def _badly_scaled_chain(n: int) -> sp.csr_matrix:
    """A chain convection-diffusion operator whose diagonal spans several orders of magnitude.

    The scale comes from a symmetric row/column weighting, so the *conditioning of the underlying
    problem* is untouched and only the scale it is presented in changes — which is exactly the
    situation equilibration exists for, and is what the coupled turbulence block looks like.
    """
    boundary = np.zeros(n)
    boundary[0] = boundary[-1] = 1.0  # both ends pinned: a chain with no boundary term is singular
    # Moderate cell Peclet, so the Galerkin coarse operator stays positive-diagonal and these tests
    # measure the rescaling rather than the separate two-level convection limit.
    base = convection_diffusion_operator(
        np.arange(n - 1),
        np.arange(1, n),
        2.0,
        n,
        flux=np.linspace(1.0, 5.0, n - 1),
        boundary_diagonal=boundary,
    )
    weight = sp.diags(np.exp(np.linspace(-4.0, 4.0, n)))
    return (weight @ base @ weight).tocsr()


def test_equilibration_is_off_by_default_and_carries_no_factor() -> None:
    """The default build is the unscaled one, and says so rather than carrying an identity factor.

    A hierarchy that always carried a factor would make ``equilibration is None`` untestable and would
    silently add two vector multiplies to every shipped V-cycle apply.
    """
    hierarchy = build_convection_hierarchy(_badly_scaled_chain(200))
    assert hierarchy.equilibration is None


def test_equilibration_rescales_the_coarsened_operator_to_a_unit_diagonal() -> None:
    """The levels really are built on ``D A D``: the fine diagonal comes out unit-magnitude.

    This is the whole point of the option — every spectral estimate, the smoother damping and the
    prolongation smoothing read that diagonal — so it is asserted on the level rather than inferred
    from the solve.
    """
    a = _badly_scaled_chain(200)
    span = np.abs(a.diagonal()).max() / np.abs(a.diagonal()).min()
    assert span > 1e5, f"the fixture must be badly scaled to test anything, span {span:.2e}"

    hierarchy = build_convection_hierarchy(a, equilibrate=True)
    assert hierarchy.equilibration is not None
    assert np.allclose(np.abs(np.asarray(hierarchy.levels[0].diagonal)), 1.0)


def test_equilibration_solves_the_original_operator() -> None:
    """``x = D M(D b)`` inverts ``A`` itself, at either scale.

    The rescaling is a similarity transform rather than an approximation, so where the hierarchy is
    exact — ``max_coarse`` above the problem size makes level 0 a direct solve — both paths must
    reproduce ``A x = b``. That is what licenses treating the factor as bookkeeping the hierarchy owns
    rather than a change of preconditioner.

    **The rescaling is not what makes the coarse solve accurate, and a version of this test that asserted
    so was measuring the old pseudo-inverse.** A pseudo-inverse truncates small singular values, and a
    badly scaled operator defeats that first: on this fixture (condition ~1.6e8) it left the unscaled
    solve at 1.2e-09 against the rescaled 5.3e-13, which read as a second, coarsening-independent reason
    to rescale. A factorization is not defeated by the scaling — the unscaled solve is 2.8e-13, four
    thousand times better and at the same floating-point floor as the rescaled one — so that reason has
    gone with the pseudo-inverse. The remaining case for rescaling is per-cell block conditioning, which
    is a different measurement.
    """
    n = 60
    a = _badly_scaled_chain(n)
    b = jnp.asarray(np.random.default_rng(0).normal(size=n))

    def residual(equilibrate):
        hierarchy = build_convection_hierarchy(a, equilibrate=equilibrate, max_coarse=n)
        assert len(hierarchy.levels) == 1  # a single direct level, so the cycle is an exact solve
        x = np.asarray(convection_multigrid_solve(hierarchy, b, cycles=1))
        return np.linalg.norm(a @ x - np.asarray(b)) / np.linalg.norm(np.asarray(b))

    raw, scaled = residual(False), residual(True)
    assert raw < 1e-10, f"the unscaled direct solve should invert A, got {raw:.3e}"
    assert scaled < 1e-10, f"the scaled direct solve should invert A, got {scaled:.3e}"


def test_an_equilibrated_cycle_is_a_fixed_linear_operator_and_transposes() -> None:
    """The scaled apply stays linear and transposable — the adjoint and the outer Krylov both need it.

    Scaling composes two fixed diagonal maps with the V-cycle, so neither property should be at risk;
    both are asserted because losing either would invalidate every gradient taken through a solve this
    preconditions, silently.
    """
    n = 120
    hierarchy = build_convection_hierarchy(_badly_scaled_chain(n), equilibrate=True)

    def cycle(b):
        return convection_multigrid_solve(hierarchy, b, cycles=2)

    rng = np.random.default_rng(0)
    u, v = (jnp.asarray(rng.normal(size=n)) for _ in range(2))
    assert np.allclose(
        np.asarray(cycle(2.0 * u + 3.0 * v)), np.asarray(2.0 * cycle(u) + 3.0 * cycle(v))
    )

    transpose = jax.linear_transpose(cycle, jnp.zeros(n, dtype=jnp.float64))
    assert np.isclose(float(v @ cycle(u)), float(transpose(v)[0] @ u))


def test_plain_prolongation_keeps_the_tentative_injection() -> None:
    """``prolongation_smoothing="none"`` freezes the piecewise-constant prolongation unsmoothed.

    Plain aggregation is a real choice, not a degenerate one — it is what the shipped PETSc bundle runs
    on this saddle — so the unsmoothed branch is pinned by the property that identifies it: every
    prolongation entry is exactly one, where a smoothed operator spreads them.
    """
    a = _chain_operator(200, 0.5, np.ones(199))
    plain = build_convection_hierarchy(a, mis_aggregation=True, prolongation_smoothing="none")
    smoothed = build_convection_hierarchy(
        a, mis_aggregation=True, prolongation_smoothing="standard"
    )

    assert np.allclose(np.asarray(plain.levels[0].p_val), 1.0)
    assert not np.allclose(np.asarray(smoothed.levels[0].p_val), 1.0)


def test_an_unknown_prolongation_smoothing_is_refused() -> None:
    """A misspelled strategy must raise, not silently fall through to a default nobody chose."""
    with pytest.raises(ValueError, match="prolongation_smoothing"):
        build_convection_hierarchy(
            _chain_operator(50, 0.5, np.ones(49)), prolongation_smoothing="smoothed"
        )


def test_undamped_smoothing_relaxes_further_than_the_spectral_default() -> None:
    """``spectral_damping=False`` makes ``omega`` the absolute relaxation, and that is the stronger one.

    ``D^-1 A`` has a unit diagonal, so its eigenvalues average one and ``lambda_max >= 1`` always —
    the spectral default therefore never relaxes by more than ``omega`` and usually by far less. The
    undamped sweep is what an equivalently-configured PETSc GAMG runs (``richardson`` at its default
    scale of 1), and matching it is what closed the measured gap on the coupled turbulence block, so
    the distinction is pinned rather than left to the two docstrings to agree about.
    """
    n = 400
    hierarchy = build_convection_hierarchy(_badly_scaled_chain(n))
    assert float(hierarchy.levels[0].lam_max) >= 1.0

    b = jnp.asarray(np.random.default_rng(0).normal(size=n))
    damped = convection_multigrid_solve(hierarchy, b, cycles=1, sweeps=4)
    undamped = convection_multigrid_solve(
        hierarchy, b, cycles=1, sweeps=4, omega=1.0, spectral_damping=False
    )
    assert not np.allclose(np.asarray(damped), np.asarray(undamped))

    # The default is recovered exactly by asking for the same factor the other way round, which is
    # what makes `spectral_damping` a change of units rather than a change of smoother.
    restated = convection_multigrid_solve(
        hierarchy,
        b,
        cycles=1,
        sweeps=4,
        omega=0.8 / float(hierarchy.levels[0].lam_max),
        spectral_damping=False,
    )
    assert np.allclose(np.asarray(damped), np.asarray(restated))


def test_the_coarsening_graph_takes_magnitudes_before_symmetrizing() -> None:
    """An antisymmetric coupling must not cancel itself out of the aggregation graph.

    Building the graph from the symmetric part ``(A + A^T)/2`` and taking the magnitude afterwards
    loses any edge with ``A_ij ~ -A_ji`` entirely, so two strongly coupled cells end up with nothing
    to aggregate across. Taking the magnitude first cannot do that. The check is on `_cell_graph`,
    which is where the block collapse and the magnitude both happen.
    """
    # Two cells, one field, coupled antisymmetrically: the symmetric part is exactly zero off-diagonal.
    a = sp.csr_matrix(np.array([[4.0, 3.0], [-3.0, 4.0]]))
    symmetric_part = (0.5 * (a + a.T)).tocsr()

    assert _cell_graph(symmetric_part, 1)[0, 1] == 0.0  # the edge vanishes — the old behaviour
    assert _cell_graph(a, 1)[0, 1] == 3.0  # ...and survives on the true operator


def test_magnitude_order_changes_weights_but_not_the_pattern_on_an_m_matrix() -> None:
    """On a frozen upwind transport operator the two orders give the same GRAPH, at different weights.

    Its off-diagonals all share a sign, so nothing can cancel and the sparsity is identical — which is
    what makes the change free wherever only the pattern is read, i.e. at ``strength_threshold = 0``,
    the default and what every scalar hierarchy uses. The *weights* do differ (``|A_ij|`` against
    ``|A_ij + A_ji| / 2``), so a hierarchy built with a strength threshold — the coupled flow block
    runs at 0.25 — genuinely sees a different strong-connection set. Both halves are asserted, because
    reporting only the first would make the change look inert where it is not.
    """
    n = 200
    a = _chain_operator(n, 2.0, np.linspace(1.0, 5.0, n - 1))
    from_true = _cell_graph(a, 1)
    from_symmetric_part = _cell_graph((0.5 * (a + a.T)).tocsr(), 1)

    assert (from_true != 0).nnz == (from_symmetric_part != 0).nnz
    assert np.array_equal(from_true.indices, from_symmetric_part.indices)
    assert not np.allclose(from_true.data, from_symmetric_part.data)


def test_reattaching_gives_every_member_a_root_it_touches() -> None:
    """After a squared-graph aggregation, the repair leaves no member two hops from its own root.

    Aggregating the squared graph is what buys a shallow hierarchy, and the cost is that an aggregate
    can reach a cell its root does not couple to — a poor support for a piecewise-constant coarse
    basis function.

    **The guarantee is conditional, and stating it as absolute would over-claim.** A member whose
    every neighbour is itself a member has no adjacent root to move to and keeps its distant one; only
    members that *have* an adjacent root are repaired. That is the reference's behaviour too, so the
    test asserts exactly it, plus the two invariants the repair must not break: no aggregate is
    emptied, and no root is ever stolen.
    """
    n = 60
    graph = sp.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], format="csr")  # a path
    squared = (graph @ graph).tocsr()
    squared.setdiag(0)
    squared.eliminate_zeros()

    aggregate, roots, count = _mis_aggregate(squared, seed=0)
    repaired = _reattach_to_adjacent_root(aggregate, roots, graph)

    assert len(np.unique(repaired)) == count  # no aggregate emptied
    assert np.all(repaired[roots] == aggregate[roots])  # roots keep their own aggregates

    adjacency = graph.toarray() != 0
    members = [v for v in range(n) if v not in set(roots.tolist())]
    repairable = [v for v in members if any(adjacency[v, r] for r in roots)]
    for vertex in repairable:
        assert adjacency[vertex, roots[repaired[vertex]]], (
            f"cell {vertex} has an adjacent root but was left on a distant one"
        )

    # ...and the repair was not vacuous: the squared aggregation really did leave members stranded.
    stranded = sum(not adjacency[v, roots[aggregate[v]]] for v in repairable)
    assert stranded > 0


def test_a_degenerate_cell_block_is_refused_by_name() -> None:
    """A singular cell block raises, naming how many — it does not silently return something usable.

    This is a real state, not a defensive check: on the coupled turbulence pair a few cells out of tens
    of thousands go degenerate as the flow develops, so the refusal is what a march actually meets, and
    it must say so rather than bake an ``inf`` into a frozen preconditioner. A pseudo-inverse that
    truncates the null direction is a plausible alternative and is deliberately NOT the behaviour here;
    if it is ever adopted it needs its own opt-in and its own evidence.
    """
    from aquaflux.solve.multigrid import _cell_block_inverse

    # Two cells, two fields, field-major. Cell 0 is ordinary; cell 1's block [[1,1],[1,1]] is singular.
    dense = np.zeros((4, 4))
    dense[0, 0], dense[2, 2] = 2.0, 4.0
    dense[1, 1], dense[1, 3], dense[3, 1], dense[3, 3] = 1.0, 1.0, 1.0, 1.0

    with pytest.raises(ValueError, match="1 of 2 are singular"):
        _cell_block_inverse(sp.csr_matrix(dense), 2)

    # ...and it says WHICH operator, because the caller coarsens and runs this on every level. A bare
    # count is not diagnosable: it sent three capture attempts after a fine-grid state that was never
    # degenerate, when the refusal was coming from a Galerkin coarse operator.
    with pytest.raises(ValueError, match=r"level 1 \(coarse\): block smoothing"):
        _cell_block_inverse(sp.csr_matrix(dense), 2, "level 1 (coarse)")

    # A small but well-conditioned block is NOT degenerate: the test is scale-free.
    tiny = sp.csr_matrix(np.diag([1e-9, 1e-9, 2e-9, 2e-9]))
    assert np.all(np.isfinite(_cell_block_inverse(tiny, 2)))

    # A structurally empty row IS degenerate, and must be named rather than slipping through: it makes
    # the Hadamard bound zero, against which no determinant compares as smaller.
    empty = np.zeros((4, 4))
    empty[0, 0], empty[2, 2], empty[3, 3] = 1.0, 1.0, 1.0  # cell 1's k row is entirely absent
    with pytest.raises(ValueError, match="1 of 2 are singular"):
        _cell_block_inverse(sp.csr_matrix(empty), 2)


def test_a_badly_scaled_block_is_not_mistaken_for_a_singular_one() -> None:
    """A block whose rows differ by orders of magnitude must be judged on the right scale.

    These are the real numbers from a coupled turbulence cell that a Frobenius-normalized test refused
    mid-march: rows differing by 1.5e8, a determinant of 1.35e-06 against a Frobenius-squared norm of
    1.6e+06, so the bar landed just above the determinant. On the row-norm (Hadamard) bound the same
    block scores 1.2e-04 -- eight orders clear of degenerate -- and it is genuinely invertible.

    The block is still nastily conditioned, around 1e12, and the assertion on the inverse says so:
    this guard's job is invertibility, not conditioning, and conflating the two is what produced a
    false refusal.
    """
    from aquaflux.solve.multigrid import _cell_block_inverse

    dense = np.zeros((4, 4))
    dense[0, 0], dense[2, 2] = 1.0, 1.0  # cell 0: an ordinary, well-scaled block
    dense[1, 1], dense[1, 3] = 8.816352e-06, 1.694848e-12  # cell 1: the captured pathological one
    dense[3, 1], dense[3, 3] = -1.284523e03, 1.526684e-01

    inverses = _cell_block_inverse(sp.csr_matrix(dense), 2)  # must not raise

    block = np.array([[8.816352e-06, 1.694848e-12], [-1.284523e03, 1.526684e-01]])
    assert np.linalg.cond(block) > 1e11  # the fixture really is the hard case, not a benign one
    assert np.allclose(
        inverses[1] @ block, np.eye(2), atol=1e-6
    )  # ...and inverting it is meaningful


def test_refit_reproduces_a_rebuild_where_the_coarsening_is_value_independent() -> None:
    """With an unsmoothed prolongator at ``strength_threshold=0``, a rebuild and a refit must agree.

    Both halves of that condition are load-bearing, and together they are what makes a refit *exact*
    rather than approximate. The coarsening then reads only the sparsity pattern, so a rebuild at a new
    operator lands on the aggregates a refit reuses; and the tentative prolongation is their 0/1
    indicator, which holds no operator values, so freezing it freezes nothing but the partition.
    Everything downstream is then a deterministic function of the new operator and that partition, so
    the two paths must land in the same place.
    """
    plain = dict(prolongation_smoothing="none")
    n = 600
    cold = build_convection_hierarchy(_chain_operator(n, 0.01, np.ones(n - 1)), **plain)
    developed_operator = _chain_operator(n, 50.0, np.linspace(1.0, 1000.0, n - 1))

    refitted = cold.refit(developed_operator)
    rebuilt = build_convection_hierarchy(developed_operator, **plain)

    assert len(refitted.levels) == len(rebuilt.levels)
    for got, want in zip(refitted.levels, rebuilt.levels, strict=True):
        assert (got.n, got.n_coarse) == (want.n, want.n_coarse)
        assert np.allclose(np.asarray(got.operator.data), np.asarray(want.operator.data))
        assert np.allclose(np.asarray(got.diagonal), np.asarray(want.diagonal))
        assert np.allclose(float(got.lam_max), float(want.lam_max))
    # ...and the refit really moved the values, so the agreement above is not two copies of `cold`.
    assert not np.allclose(
        np.asarray(refitted.levels[0].operator.data), np.asarray(cold.levels[0].operator.data)
    )


def test_refit_keeps_a_smoothed_prolongator_a_rebuild_would_re_derive() -> None:
    """A smoothed prolongator reads operator values, so a refit freezes MORE than the partition.

    ``P <- P_tent - w D^-1 A P_tent`` carries a relaxation built from the operator it was derived at, so
    holding it fixed holds that relaxation too and the coarse operator then differs from a rebuild's.
    That is a real limitation of the frozen path rather than a defect, and it is pinned here so nobody
    reads the exact agreement above as holding for every configuration.
    """
    n = 600
    smoothed = dict(prolongation_smoothing="symmetric-part")
    cold = build_convection_hierarchy(_chain_operator(n, 0.01, np.ones(n - 1)), **smoothed)
    developed_operator = _chain_operator(n, 50.0, np.linspace(1.0, 1000.0, n - 1))

    refitted = cold.refit(developed_operator)
    rebuilt = build_convection_hierarchy(developed_operator, **smoothed)

    coarsest = -1
    assert not np.allclose(
        np.asarray(refitted.levels[coarsest].operator.data),
        np.asarray(rebuilt.levels[coarsest].operator.data),
    )


def test_refit_holds_a_partition_that_a_rebuild_would_move() -> None:
    """With a strength threshold live, a rebuild re-partitions and a refit does not — the whole point.

    The aggregation then reads ``|A_ij|``, so an operator whose anisotropy has rotated (strong direction
    x rather than y, on the identical graph) aggregates differently. A rebuild follows it; a refit keeps
    the partition and moves only the values.

    **The assertion is on aggregate MEMBERSHIP, not on level sizes, because sizes do not discriminate
    here:** on a square grid, aggregating along either direction gives the identical *count*. Two
    hierarchies can agree in every level size and still partition the mesh completely differently, which
    is exactly why a march reporting stable level sizes is not reporting a stable coarsening.
    """
    threshold, coarse = 0.25, 16
    plain = dict(prolongation_smoothing="none", max_coarse=coarse, strength_threshold=threshold)
    strong_y = _anisotropic_poisson(24, 24, aspect_ratio=100.0)
    strong_x = _anisotropic_poisson(24, 24, aspect_ratio=0.01)
    built = build_convection_hierarchy(strong_y, **plain)

    rebuilt = build_convection_hierarchy(strong_x, **plain)
    refitted = built.refit(strong_x)

    def membership(hierarchy):
        """Which coarse unknown each fine degree of freedom feeds, on the finest level."""
        level = hierarchy.levels[0]
        order = np.argsort(np.asarray(level.p_frow))
        return np.asarray(level.p_ccol)[order]

    assert np.array_equal(membership(refitted), membership(built))
    assert not np.array_equal(membership(rebuilt), membership(built)), (
        "the rotated operator did not re-partition, so this fixture cannot show the difference"
    )
    # The refit is over the NEW operator, not a copy of the old hierarchy.
    assert np.allclose(np.asarray(refitted.levels[0].operator.data), strong_x.tocsr().data, atol=0)


def test_refit_rejects_an_operator_of_the_wrong_size() -> None:
    """A mismatched operator raises rather than returning a hierarchy that silently coarsens nothing."""
    hierarchy = build_convection_hierarchy(_chain_operator(200, 0.01, np.ones(199)))
    with pytest.raises(ValueError, match="cannot refit"):
        hierarchy.refit(_chain_operator(100, 0.01, np.ones(99)))


def _rotated_anisotropy():
    """Two operators on one graph whose aggregations genuinely differ in SIZE, not just membership.

    A square grid is the wrong fixture for this: aggregating along either direction gives the identical
    count, so the shapes coincide and a budget appears to do nothing. A rectangular one separates them.
    """
    return (
        _anisotropic_poisson(24, 16, aspect_ratio=100.0),
        _anisotropic_poisson(24, 16, aspect_ratio=0.01),
    )


def test_a_repartitioning_rebuild_changes_shape_without_a_budget() -> None:
    """The premise of the budget: at a live strength threshold, a rebuild moves the array shapes.

    Asserted separately so the budget tests below cannot pass vacuously on a fixture where the two
    hierarchies would have coincided anyway — which is exactly what a square grid does.
    """
    plain = dict(prolongation_smoothing="none", max_coarse=16, strength_threshold=0.25)
    strong_y, strong_x = _rotated_anisotropy()

    sizes = [
        [(level.n, level.n_coarse) for level in build_convection_hierarchy(a, **plain).levels]
        for a in (strong_y, strong_x)
    ]
    assert sizes[0] != sizes[1], "fixture does not repartition; the budget tests would be vacuous"


def test_shape_budget_makes_a_repartitioning_rebuild_a_compilation_cache_hit() -> None:
    """Coarsening into a fixed ladder keeps one compiled V-cycle across a rebuild that repartitions.

    This is the difference between a budget and freezing the coarsening: the partition is still
    re-derived from the current operator — only the *sizes* it is poured into are fixed — so the coarse
    space tracks the flow while the compiled cycle does not have to be rebuilt.
    """
    plain = dict(prolongation_smoothing="none", max_coarse=16, strength_threshold=0.25)
    strong_y, strong_x = _rotated_anisotropy()
    budget = build_convection_hierarchy(strong_y, **plain).shape_budget(headroom=1.3)

    built = [
        build_convection_hierarchy(a, **plain, shape_budget=budget) for a in (strong_y, strong_x)
    ]

    def signature(hierarchy):
        return [
            (
                level.n,
                level.n_coarse,
                int(level.operator.data.shape[0]),
                None if level.p_val is None else int(level.p_val.shape[0]),
            )
            for level in hierarchy.levels
        ]

    assert signature(built[0]) == signature(built[1])

    traces = []

    @jax.jit
    def apply(hierarchy, b):
        traces.append(1)  # appended once per trace, not per call
        return convection_multigrid_solve(hierarchy, b, cycles=1)

    b = jnp.asarray(np.random.default_rng(0).normal(size=strong_y.shape[0]))
    apply(built[0], b).block_until_ready()
    assert len(traces) == 1
    apply(built[1], b).block_until_ready()
    assert len(traces) == 1, "a budgeted rebuild retraced the jitted V-cycle"


def test_budget_padding_leaves_the_preconditioner_unchanged() -> None:
    """Padding must be INERT: the budgeted hierarchy is the same operator as the one it padded.

    Empty aggregates carry a unit diagonal and no coupling, so nothing restricts into them and their
    correction prolongates as zero; the operator's padded entries are zeros in the last row. If either
    were wrong the coarse correction would differ, and the budget would be buying its cache hit by
    quietly changing the preconditioner.
    """
    plain = dict(prolongation_smoothing="none", max_coarse=16, strength_threshold=0.25)
    strong_y, strong_x = _rotated_anisotropy()
    budget = build_convection_hierarchy(strong_y, **plain).shape_budget(headroom=1.3)

    unpadded = build_convection_hierarchy(strong_x, **plain)
    padded = build_convection_hierarchy(strong_x, **plain, shape_budget=budget)
    assert padded.levels[-1].n > unpadded.levels[-1].n  # padding really happened

    b = jnp.asarray(np.random.default_rng(1).normal(size=strong_x.shape[0]))
    reference = convection_multigrid_solve(unpadded, b, cycles=2)
    assert np.allclose(
        np.asarray(reference),
        np.asarray(convection_multigrid_solve(padded, b, cycles=2)),
        rtol=1e-10,
    )


def test_budget_overflow_raises_rather_than_dropping_entries() -> None:
    """A budget too small to hold the rebuild must RAISE. Dropping to fit would corrupt the operator.

    Silently truncating the Galerkin product is the failure this guards: it would leave a coarse
    operator missing arbitrary couplings, with no bound on the error and nothing in the output to say
    so. Raising turns that into a re-budget.
    """
    plain = dict(prolongation_smoothing="none", max_coarse=16, strength_threshold=0.25)
    strong_y, strong_x = _rotated_anisotropy()
    # Budget from the SMALLER hierarchy at no headroom, then rebuild at the operator that aggregates
    # into more cells than it — the direction a budget cannot absorb.
    tight = build_convection_hierarchy(strong_x, **plain).shape_budget(headroom=1.0)

    with pytest.raises(ValueError, match="budget"):
        build_convection_hierarchy(strong_y, **plain, shape_budget=tight)


def test_budget_padding_does_not_compound_down_the_hierarchy() -> None:
    """A padded cell must not become its own aggregate on the next level, or the padding multiplies.

    The regression this guards was found only by running a budget through several levels. A padded
    cell is decoupled — unit diagonal, no off-diagonal coupling — so in the next level's aggregation
    graph it is an ISOLATED vertex, and an isolated vertex gets an aggregate to itself. Each level's
    padding therefore inflated the next level's aggregate count by the padding it had just added, which
    compounded until the budget overflowed: on a four-level synthetic, level 1 asked for 180 slots
    against a budget of 85 that its own natural coarsening fitted inside comfortably.

    The fix is to aggregate the real cells only and send every padded cell to one reserved slot, so the
    count entering each level is the number of REAL aggregates and the padding cannot accumulate.
    """
    # `max_levels` must be raised explicitly: the convection builder defaults to two levels, which is
    # one coarsening — too shallow for padding to compound, so the test would pass vacuously.
    plain = dict(prolongation_smoothing="none", max_coarse=8, strength_threshold=0.25, max_levels=5)
    strong_y, strong_x = _rotated_anisotropy()
    budget = build_convection_hierarchy(strong_y, **plain).shape_budget(headroom=1.4)
    assert len(budget.coarse_cells) >= 3, "fixture must be deep enough for padding to compound"

    hierarchy = build_convection_hierarchy(strong_x, **plain, shape_budget=budget)

    # Every level lands exactly on its budget — no level absorbed the one above it's padding.
    assert [level.n // level.block_size for level in hierarchy.levels[1:]] == list(
        budget.coarse_cells
    )
    # And the coarse operators stay sparse: a level of isolated padded cells would show up as a
    # diagonal-heavy operator with an aggregate count near its cell count.
    b = jnp.asarray(np.random.default_rng(0).normal(size=strong_x.shape[0]))
    assert np.all(np.isfinite(np.asarray(convection_multigrid_solve(hierarchy, b, cycles=2))))


def test_the_zero_guess_peel_matches_the_general_jacobi_sweep_exactly() -> None:
    """The peeled first sweep is EXACT, not an approximation, at every sweep count and damping.

    At a zero iterate the residual ``b - A x`` is exactly ``b``, so the peel removes an application of
    the level operator -- the densest thing in the cycle, charged at every level of every V-cycle --
    without changing the arithmetic. Asserted as exact equality rather than a tolerance, because a
    tolerance would hide a genuine reordering, and the whole claim is that there is none.

    Covers the per-cell BLOCK branch as well as the scalar diagonal one: the two peel differently (a
    batched contraction against an elementwise multiply) and only the block branch is what the coupled
    transported scalars actually run.
    """
    rng = np.random.default_rng(0)
    n_cells, n_fields = 60, 2
    rows, cols, vals = [], [], []
    for cell in range(n_cells):
        for offset in (0, 1, -1):
            other = cell + offset
            if not 0 <= other < n_cells:
                continue
            for row_field in range(n_fields):
                for col_field in range(n_fields):
                    same = offset == 0 and row_field == col_field
                    rows.append(row_field * n_cells + cell)
                    cols.append(col_field * n_cells + other)
                    vals.append(4.0 if same else rng.normal() * 0.2)
    a = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells * n_fields,) * 2)
    hierarchy = build_convection_hierarchy(
        a, block_size=n_fields, max_coarse=16, mis_aggregation=True, prolongation_smoothing="none"
    )
    level = hierarchy.levels[0]
    b = jnp.asarray(rng.normal(size=level.n))

    for sweeps in (1, 2, 4):
        for damping in (False, True):
            peeled = _jacobi_smooth_zero(level, b, sweeps, 1.0, damping)
            general = _jacobi_smooth(level, b, jnp.zeros_like(b), sweeps, 1.0, damping)
            assert np.array_equal(np.asarray(peeled), np.asarray(general)), (
                f"the peel moved the answer at {sweeps} sweeps, spectral_damping={damping}"
            )


def _two_field_operator(n_cells: int = 60, coupling: float = 40.0) -> sp.csr_matrix:
    """A field-major two-field transport operator whose WITHIN-CELL coupling exceeds its diagonal.

    The shape the coupled turbulence pair has, and the reason a multi-field level needs a block
    smoother: with ``coupling`` above the diagonal a point sweep discards the dominant term entirely.
    Degree of freedom ``(cell i, field f)`` sits at ``f * n_cells + i``.
    """
    upwind = sp.diags(
        [-np.full(n_cells - 1, 4.0), np.full(n_cells, 6.0), -np.full(n_cells - 1, 1.0)],
        [-1, 0, 1],
        format="csr",
    )
    # [[A, c I], [c I, A]] — the off-diagonal blocks are the within-cell coupling.
    cross = sp.identity(n_cells, format="csr") * coupling
    return sp.bmat([[upwind, cross], [cross, upwind]], format="csr")


def test_block_air_hierarchy_splits_whole_cells_not_degrees_of_freedom() -> None:
    """A cell must be entirely coarse or entirely fine.

    Splitting per degree of freedom is field-blind: it can put one field of a cell on the coarse grid
    and another on the fine one, which produces the degenerate Galerkin row that makes a build fail for
    a reason that looks like the *fine* operator being at fault when it is clean.
    """
    a = _two_field_operator()
    hierarchy = build_air_hierarchy(a, block_size=2, max_coarse=8)
    for level in hierarchy.levels:
        mask = np.asarray(level.c_mask)
        cells = level.n // level.block_size
        for field in range(1, level.block_size):
            assert np.array_equal(mask[:cells], mask[field * cells : (field + 1) * cells]), (
                "the C/F split differs between a cell's fields, so it was decided per degree of freedom"
            )


def test_block_air_builds_where_the_scalar_diagonal_is_negative() -> None:
    """The block path must stand on an operator a point smoother's guard refuses.

    A true Jacobian slice can carry a negative scalar diagonal while every cell block is perfectly well
    conditioned. ``_require_positive_diagonal`` is a *point*-smoother requirement, so a block hierarchy
    must not be held to it — otherwise it is refused on exactly the operator it exists to precondition.
    """
    a = _two_field_operator().tolil()
    # Flip one cell's first-field diagonal negative. Its 2x2 block stays invertible (the off-diagonal
    # coupling is 40), so this is a legal operator for a block smoother and not for a point one.
    a[3, 3] = -6.0
    a = a.tocsr()
    assert a.diagonal().min() < 0

    with pytest.raises(ValueError, match="positive"):
        build_air_hierarchy(a, max_coarse=8)  # scalar path: correctly refuses

    hierarchy = build_air_hierarchy(a, block_size=2, max_coarse=8)  # block path: builds
    assert len(hierarchy.levels) > 1
    assert hierarchy.levels[0].block_inverse is not None


def test_block_air_v_cycle_contracts_where_a_point_smoother_cannot() -> None:
    """The block smoother is what makes a multi-field level work, measured against the point one.

    With the within-cell coupling above the diagonal, inverting each cell's own block is not a
    refinement — it is the difference between a V-cycle that contracts and one that does not.
    """
    a = _two_field_operator()
    hierarchy = build_air_hierarchy(a, block_size=2, max_coarse=8)
    dense = jnp.asarray(a.toarray())
    b = jnp.asarray(np.random.default_rng(0).standard_normal(a.shape[0]))

    x = jnp.zeros(a.shape[0])
    for _ in range(4):
        x = x + air_multigrid_solve(hierarchy, b - dense @ x, cycles=1)
    residual = float(jnp.linalg.norm(dense @ x - b)) / float(jnp.linalg.norm(b))
    assert residual < 1e-3, f"the block V-cycle did not contract: {residual:.2e}"

    # Not vacuous: the same hierarchy with its block inverse removed — i.e. a point smoother — does not.
    pointwise = eqx.tree_at(
        lambda h: [lv.block_inverse for lv in h.levels],
        hierarchy,
        replace=[None] * len(hierarchy.levels),
        is_leaf=lambda x: x is None,
    )
    y = jnp.zeros(a.shape[0])
    for _ in range(4):
        y = y + air_multigrid_solve(pointwise, b - dense @ y, cycles=1)
    point_residual = float(jnp.linalg.norm(dense @ y - b)) / float(jnp.linalg.norm(b))
    assert point_residual > 10 * residual, (
        f"the point smoother did as well ({point_residual:.2e} vs {residual:.2e}), so this fixture "
        "does not exercise the within-cell coupling the block inverse exists for"
    )


def test_block_air_refresh_preserves_the_structure() -> None:
    """A block hierarchy refreshes on its frozen coarsening like a scalar one, keeping every shape."""
    cold = _two_field_operator(coupling=40.0)
    developed = _two_field_operator(coupling=55.0)
    built = build_air_hierarchy(cold, block_size=2, max_coarse=8)
    refreshed = refresh_air_hierarchy(built, developed)

    assert len(refreshed.levels) == len(built.levels)
    for old, new in zip(built.levels, refreshed.levels, strict=True):
        assert (old.n, old.n_coarse, old.block_size) == (new.n, new.n_coarse, new.block_size)
        assert old.operator.data.shape == new.operator.data.shape
    # ...and the refresh is not a no-op: the values moved with the operator.
    assert not np.allclose(
        np.asarray(built.levels[0].operator.data), np.asarray(refreshed.levels[0].operator.data)
    )
