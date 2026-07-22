"""Unit tests for the algebraic-multigrid building blocks (smoothed aggregation, convection, lAIR).

These cover the frozen-operator V-cycle families: a mesh-independent smoothed-aggregation V-cycle for
the symmetric pressure Schur, its convection-diffusion variant, and the reduction-based (lAIR) cycle
for a strongly convection-dominated operator — each a fixed linear operator (a valid frozen left
preconditioner), plus the degenerate-mesh build guards.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.mesh import structured_grid_2d
from aquaflux.solve.frozen_operator import convection_diffusion_operator, decouple_dof
from aquaflux.solve.multigrid import (
    _chebyshev_smooth,
    _jacobi_smooth,
    _lair_restriction,
    _one_point_interpolation,
    _rs_split,
    _SparseLevel,
    _strength_classical,
    air_multigrid_solve,
    build_air_hierarchy,
    build_convection_hierarchy,
    build_smoothed_hierarchy,
    convection_multigrid_solve,
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


def test_air_v_cycle_is_mesh_independent_at_high_peclet() -> None:
    """The reduction-based (lAIR) V-cycle converges the strongly-convective operator with a **flat**
    per-cycle contraction as the mesh refines — the Peclet-robust, mesh-independent property the
    two-level aggregation method (and deep Galerkin recursion) lack. For a convection-dominated
    operator the approximate ideal restriction is nearly exact, so a few cycles behave like a direct
    solve regardless of the mesh size."""
    contractions = []
    for n in (24, 48):  # cell Peclet ~40; a 4x change in cell count
        owner, nb, visc, mdot, ncell, bd = _convective_grid(n, mu=1e-3, speed=1.0)
        hierarchy = build_air_hierarchy(_operator(owner, nb, visc, mdot, ncell, bd))
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
    r = np.asarray(_lair_restriction(a, split, degree=1).toarray())
    assert r.shape == (2, 4)
    assert np.all(np.isfinite(r))
    # Minimum-norm least-squares solution of [[1, 1], [1, 1]] z = [-1, -1].
    assert np.allclose(r[0], [1.0, -0.5, -0.5, 0.0])
    assert np.allclose(r[1], [0.0, 0.0, 0.0, 1.0])  # empty F-neighbourhood -> identity entry only


def test_lair_restriction_reproduces_the_exact_schur_complement() -> None:
    """With an F-neighbourhood wide enough to reach every F-point it couples to, the local
    approximate-ideal solve *is* the ideal restriction ``R = [-A_cf A_ff⁻¹, I]``: it annihilates the
    F-columns of ``R A``, so the Galerkin coarse operator ``R A P`` is exactly the Schur complement
    ``A_cc - A_cf A_ff⁻¹ A_fc`` — the coarse action of the fine operator, reproduced exactly."""
    n = 9
    a = _upwind_chain(n)
    split = _rs_split(_strength_classical(a, 0.25))
    coarse, fine = np.where(split == 1)[0], np.where(split == 0)[0]
    assert len(fine) > 0 and len(coarse) > 0
    p = _one_point_interpolation(a, split)
    r = _lair_restriction(a, split, degree=n)
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
        assert lo.val.shape == hi.val.shape  # operator sparsity
        assert lo.diagonal.shape == hi.diagonal.shape
        if lo.p_val is not None:
            assert lo.p_val.shape == hi.p_val.shape  # prolongation sparsity
    # ...and the values really do differ, so the invariance above is not a trivial no-op.
    assert not np.allclose(np.asarray(cold.levels[0].val), np.asarray(developed.levels[0].val))


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
