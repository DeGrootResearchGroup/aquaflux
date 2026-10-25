"""The SIMPLE-smoothed saddle hierarchy: exactness, adjoint legality, and the refresh contract.

Every test here builds a small generalized saddle point directly — a velocity block, a gradient, a
divergence that is NOT the transposed gradient, and a nonzero pressure--pressure block — because that is
what a collocated Rhie--Chow discretization produces and what the classical saddle-point results do not
cover. Nothing here needs a mesh, a flow, or a case.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.precondition import SimpleSmoothed, SimpleSmoothedInverse, saddle_multigrid
from aquaflux.precondition.saddle_multigrid import (
    _simple_correction,
    _simple_pieces,
    _simple_smoothed_cycle,
)
from aquaflux.solve import with_transposes


def _saddle(
    n_cells: int = 240,
    dim: int = 3,
    seed: int = 0,
    *,
    velocity_pressure_coupling: float = 0.12,
) -> sp.csr_matrix:
    """A field-major generalized saddle block on a chain graph, diagonally dominant in velocity.

    ``D != G^T`` and the (2,2) block is nonzero, so this is the operator class the smoother targets
    rather than a classical Stokes saddle.

    ``velocity_pressure_coupling`` scales only the velocity-pressure cross terms (the entries feeding
    ``D`` and ``G``); the intra-velocity off-diagonal stays at the same ``0.12`` scale the default also
    uses, so the velocity block itself stays strongly diagonally dominant however this is set. At the
    default it reproduces the original single-scale fixture exactly (same value, same draw order from
    ``rng``, so the matrix is bit-identical). Raised well above the default, the velocity-block Jacobi
    predictor alone is no longer enough to reduce the residual — see
    ``test_the_schur_correction_sign_is_load_bearing``, which needs exactly that.
    """
    rng = np.random.default_rng(seed)
    n_fields = dim + 1
    rows, cols, vals = [], [], []
    for cell in range(n_cells):
        for offset in (0, 1, -1, 2, -2):
            other = cell + offset
            if not 0 <= other < n_cells:
                continue
            for row_field in range(n_fields):
                for col_field in range(n_fields):
                    diagonal = offset == 0 and row_field == col_field
                    if diagonal and row_field < dim:
                        value = 9.0  # velocity: strongly diagonally dominant
                    elif diagonal:
                        value = 0.35  # pressure: small but nonzero, the Rhie--Chow damping
                    elif row_field < dim and col_field < dim:
                        value = rng.normal() * 0.12  # intra-velocity: kept weak
                    else:
                        value = rng.normal() * velocity_pressure_coupling
                    rows.append(row_field * n_cells + cell)
                    cols.append(col_field * n_cells + other)
                    vals.append(value)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_cells * n_fields, n_cells * n_fields))


_SETTINGS = dict(
    max_levels=4, max_coarse=40, strength_threshold=0.25, avoid_singletons=True, aggressive_levels=0
)


def test_the_inverse_reduces_the_true_residual() -> None:
    """The point of the object: applied as a preconditioner it must actually approximate ``A^-1``.

    Asserted on the TRUE residual ``||A x - b|| / ||b||``, never a preconditioned norm — the measure
    that has produced the most retracted verdicts on this operator.
    """
    a = _saddle()
    inverse = SimpleSmoothedInverse(a, 4, **_SETTINGS)
    b = np.asarray(np.random.default_rng(1).normal(size=a.shape[0]))

    x = np.asarray(inverse.apply(b))

    assert np.all(np.isfinite(x))
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 0.75


def test_the_schur_correction_sign_is_load_bearing() -> None:
    """Isolates ``_simple_correction`` itself, where the full-hierarchy test above cannot.

    On the default fixture (velocity-pressure coupling 0.12) the velocity block is so diagonally
    dominant that the predictor alone -- ignoring the pressure correction entirely -- already clears
    the full-hierarchy test's bound, so a sign error in the Schur right-hand side
    (``rhs = pressure_residual - pieces.divergence.apply(predictor)`` becoming ``+``, #409) moves the
    residual-reduction ratio from 0.031 to 0.034: real, but far too small a gap for any bound on the
    full multigrid V-cycle to separate from ordinary seed-to-seed noise.

    Two changes make the sign load-bearing here, and both are needed. First, this calls
    ``_simple_correction`` directly -- ONE relaxation sweep on the fine level, no hierarchy, no coarse
    grid -- so nothing outside the function under test can compensate for its sign. Second, the fixture
    raises ``velocity_pressure_coupling`` to 2.0 (vs. the default 0.12), which is far stronger than this
    discretization's own Rhie--Chow coupling but is exactly what the issue's acceptance criteria call
    for: coupling strong enough that the Schur correction, not the velocity predictor alone, decides
    whether the sweep contracts the residual at all. (Raising the coupling this much in the OTHER
    tests' fixture breaks the multigrid hierarchy itself -- coarsening and smoothing this saddle no
    longer converges in one V-cycle regardless of the Schur sign -- which is why this is a narrowly
    scoped companion test rather than a change to the shared fixture's default.)

    Measured under: ``n_cells=240, dim=3`` (the module default), ``pressure_sweeps=4,
    pressure_smoother_weight=1.0`` (``SimpleSmoothedInverse``'s defaults), ``frobenius=True,
    schur_frobenius=True`` (also its defaults), fixture seed 0, right-hand side
    ``np.random.default_rng(1)``. At those settings one correct sweep measures a true-residual ratio of
    0.873; the sign-flipped RHS measures 1.020 -- the correction stops even contracting the residual.
    The 0.95 bound sits with real margin on both sides of that gap.
    """
    a = _saddle(velocity_pressure_coupling=2.0)
    pieces = _simple_pieces(a, block_size=4, frobenius=True, schur_frobenius=True)
    b = np.asarray(np.random.default_rng(1).normal(size=a.shape[0]))

    x = np.asarray(
        _simple_correction(pieces, jnp.asarray(b), pressure_sweeps=4, pressure_smoother_weight=1.0)
    )

    assert np.all(np.isfinite(x))
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 0.95


@pytest.mark.parametrize("block_splitting", [False, True])
def test_the_schur_is_applied_from_its_pieces_both_ways(block_splitting) -> None:
    """``S p`` is ``(C - D M G) p`` formed on the host, and its transpose ``Sᵀ y``, from the pieces alone.

    The pieces hold no Schur, so the pressure sweep's product and the adjoint's transpose of it are only
    as right as the three products they chain; a sign, or a piece transposed in the wrong place, shows
    here on a nonsymmetric saddle.
    """
    a = _saddle()
    pieces = _simple_pieces(a, 4, frobenius=True, block_splitting=block_splitting)
    nv = pieces.n_velocity
    f, g, d, c = a[:nv, :nv], a[:nv, nv:], a[nv:, :nv], a[nv:, nv:]
    m = (
        pieces.f_block_inverse.to_scipy()
        if block_splitting
        else sp.diags(np.asarray(pieces.f_diagonal_inverse))
    )
    schur = (c - d @ (m @ g)).toarray()
    assert not np.allclose(schur, schur.T)
    rng = np.random.default_rng(4)
    p, y = rng.standard_normal(c.shape[0]), rng.standard_normal(c.shape[0])
    np.testing.assert_allclose(
        np.asarray(pieces.schur_apply(jnp.asarray(p))), schur @ p, rtol=1e-12, atol=1e-12
    )
    held = with_transposes(pieces)
    (transposed,) = jax.linear_transpose(held.schur_apply, jnp.asarray(p))(jnp.asarray(y))
    np.testing.assert_allclose(np.asarray(transposed), schur.T @ y, rtol=1e-12, atol=1e-12)
    assert f.shape == (nv, nv)


def test_the_cycle_is_a_fixed_linear_operator() -> None:
    """``b -> x`` must be LINEAR, or the non-flexible outer GMRES it preconditions is invalid.

    A fixed cycle count, a fixed sweep count and no inner Krylov method are what buy this; an adaptive
    smoother or an inner solve to a tolerance would break it silently — the outer solve would still run
    and would simply converge to the wrong thing.
    """
    a = _saddle()
    inverse = SimpleSmoothedInverse(a, 4, **_SETTINGS)
    rng = np.random.default_rng(2)
    u = np.asarray(rng.normal(size=a.shape[0]))
    v = np.asarray(rng.normal(size=a.shape[0]))

    combined = np.asarray(inverse.apply(3.0 * u - 2.0 * v))
    separate = 3.0 * np.asarray(inverse.apply(u)) - 2.0 * np.asarray(inverse.apply(v))

    assert np.allclose(combined, separate, rtol=1e-10, atol=1e-12)


def test_the_transpose_is_the_adjoint_of_the_forward_cycle() -> None:
    """``<y, M x> == <M^T y, x>``, which the implicit-function-theorem adjoint's transpose solve needs.

    The transpose is built lazily by :func:`jax.linear_transpose`, so this also pins that the cycle
    stays traceable — a host callback or a data-dependent branch inside it would have no transpose rule.
    """
    a = _saddle()
    inverse = SimpleSmoothedInverse(a, 4, **_SETTINGS)
    rng = np.random.default_rng(3)
    x = np.asarray(rng.normal(size=a.shape[0]))
    y = np.asarray(rng.normal(size=a.shape[0]))

    forward = float(np.dot(y, np.asarray(inverse.apply(x))))
    transposed = float(np.dot(np.asarray(inverse.apply(y, transpose=True)), x))

    assert abs(forward - transposed) <= 1e-11 * abs(forward)


def test_refactor_refits_in_place_onto_a_new_operator() -> None:
    """A mid-march refresh must mutate the object the compiled solve holds, not replace it.

    The field split RAISES on an inverse offering neither ``refactor_block`` nor ``refactor``, because
    replacing the object would recompile the whole coupled solve. A single-state probe never reaches
    this path, which is why it went missing once already.
    """
    a = _saddle()
    inverse = SimpleSmoothedInverse(a, 4, **_SETTINGS)
    b = np.asarray(np.random.default_rng(4).normal(size=a.shape[0]))
    before = np.asarray(inverse.apply(b))

    inverse.refactor_block((a * 1.7).tocsr())
    after = np.asarray(inverse.apply(b))

    assert np.all(np.isfinite(after))
    assert not np.allclose(before, after), "the refresh did not re-fit to the new operator"


def test_a_refresh_at_unchanged_shapes_reuses_the_compiled_cycle() -> None:
    """The whole point of the level records' static/traced split — and it is easy to lose.

    The cycle is a module-level jitted function taking the hierarchy and the smoother pieces as
    ARGUMENTS. Building one per refresh instead — a fresh ``jax.jit`` closing over them — recompiles the
    entire unrolled V-cycle every time, because a new closure is a new cache key whatever its contents.
    Measured on a 92160-degree-of-freedom flow block that cost 1.07 s per refresh, which is most of what
    a refresh costs; it also embeds the hierarchy's arrays in the compiled program as constants, which is
    why the first build took 1.31 s rather than 0.15 s.

    **Asserted on what DETERMINES the cache key — the argument pytree's structure and its leaves' shapes
    and dtypes — plus a trace count over a function this test owns.** A first version read
    ``jax.jit._cache_size()`` off the module's own jitted cycle, which passes alone and fails in a long
    run: those entries are not retained for the life of a process, so the probe reports zero for reasons
    that have nothing to do with the code under test. A cache-size probe measures JAX's retention policy;
    this measures the invariant.
    """
    a = _saddle()
    inverse = SimpleSmoothedInverse(a, 4, **_SETTINGS)
    b = np.asarray(np.random.default_rng(11).normal(size=a.shape[0]))
    inverse.apply(b)

    def signature(inv):
        arguments = (inv._hierarchy, inv._extras)
        leaves, structure = jax.tree_util.tree_flatten(arguments)
        return structure, [(leaf.shape, leaf.dtype) for leaf in leaves]

    before = signature(inverse)
    inverse.refactor_block((a * 1.7).tocsr())
    after = signature(inverse)

    assert after == before, (
        "the refresh moved the argument pytree, so the compiled cycle cannot be reused; a shape or a "
        "static field is tracking the operator's values"
    )

    # And they are genuinely arguments rather than captures: a function over them traces ONCE across
    # both hierarchies. Owned here, so the count is this test's and not JAX's to evict.
    traces = []

    @eqx.filter_jit
    def cycle(hierarchy, pieces, residual):
        traces.append(1)  # appended once per trace, not per call
        return _simple_smoothed_cycle(hierarchy, pieces, residual, inverse._smoother)

    rhs = jnp.asarray(b)
    cycle(inverse._hierarchy, inverse._extras, rhs).block_until_ready()
    assert len(traces) == 1
    inverse.refactor_block((a * 2.3).tocsr())
    cycle(inverse._hierarchy, inverse._extras, rhs).block_until_ready()
    assert len(traces) == 1, "a refreshed hierarchy retraced the cycle"


def test_the_pieces_carry_no_host_matrix_so_they_can_be_traced() -> None:
    """A host sparse matrix on the record is neither a traced leaf nor a hashable static field.

    Carrying one would make the record unusable as a jit argument — silently, by falling back to a
    closure.
    """
    a = _saddle()
    inverse = SimpleSmoothedInverse(a, 4, **_SETTINGS)

    leaves = jax.tree_util.tree_leaves(inverse._extras)
    assert leaves, "the pieces have no traced leaves at all"
    assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves), (
        "a non-array leaf would be traced as one and fail"
    )


def test_a_mismatched_block_is_rejected() -> None:
    """Refreshing onto a differently-sized block raises instead of building something incoherent."""
    inverse = SimpleSmoothedInverse(_saddle(n_cells=120), 4, **_SETTINGS)

    with pytest.raises(ValueError, match="cannot refactor"):
        inverse.refactor_block(_saddle(n_cells=80))


def test_the_object_is_silent_unless_a_report_sink_is_supplied() -> None:
    """A library preconditioner must not write to stdout on its own; the caller supplies the sink.

    The build record is genuinely worth having in a run log — a spec-token collision once made an arm
    run forty sweeps while every other line of output looked correct — so it is kept, and injected.
    """
    a = _saddle()
    captured: list[str] = []

    SimpleSmoothedInverse(a, 4, **_SETTINGS)  # no sink: nothing is emitted
    SimpleSmoothedInverse(a, 4, **_SETTINGS, report=captured.append)

    assert captured, "the report sink received nothing"
    assert any("SIMPLE-smoothed hierarchy" in line for line in captured)


def test_a_frozen_coarsening_refresh_reports_its_own_last_aggregation_not_a_strangers() -> None:
    """#287: the reported aggregate stats must be THIS inverse's own, never another block's.

    A field split builds a leading and a trailing hierarchy from one `_build_aggregation_hierarchy`
    call each; whichever one aggregates LAST used to leave every hierarchy's diagnostic pointed at a
    single module-level accumulator. A `frozen_coarsening` refresh calls `refit`, which reuses the
    stored prolongations and aggregates nothing new -- so on that path the accumulator held whatever
    some *other* hierarchy's most recent build had left there, and the report attributed it to this one.

    Reproduced here without any field split: build two inverses on two DIFFERENTLY-SIZED operators (so
    their aggregate-size histograms provably differ -- a same-sized but differently-valued block turns
    out to aggregate identically here, since the strength graph at this threshold is set by the fixture's
    deterministic diagonal-dominance pattern, not by its small random off-diagonals), in an order that
    makes the second the "most recent" aggregation, then refresh the FIRST one under `frozen_coarsening`.
    Its reported stats must be unchanged from its own initial build -- not the second inverse's.
    """
    leading = SimpleSmoothedInverse(_saddle(n_cells=240), 4, **_SETTINGS, frozen_coarsening=True)
    leading_stats = leading._aggregate_stats
    assert leading_stats, "the fixture must actually aggregate at least one level"

    # A different-shaped operator, built afterwards, so it is the most recently populated aggregation of
    # any shared accumulator -- exactly the corrupting build the frozen-coarsening report used to leak.
    trailing = SimpleSmoothedInverse(_saddle(n_cells=90), 4, **_SETTINGS)
    assert trailing._aggregate_stats != leading_stats, (
        "the fixture must aggregate differently, or this test cannot discriminate the two"
    )

    # Refit onto a rescaled copy of the SAME operator: same shape, same sparsity pattern, so a frozen
    # coarsening is legal, and no new aggregation occurs.
    leading.refactor_block((_saddle(n_cells=240) * 1.7).tocsr())

    assert leading._aggregate_stats == leading_stats, (
        "a frozen-coarsening refresh aggregates nothing new, so the stats must be unchanged"
    )
    assert leading._aggregate_stats != trailing._aggregate_stats


def test_a_frozen_refresh_reuses_its_plans_and_returns_what_fresh_plans_would(monkeypatch) -> None:
    """A frozen-coarsening refit on an unchanged pattern makes no SIMPLE plan after the first.

    A level's plan -- its block cuts, the structural patterns of ``D^-1 G`` and the Schur complement,
    their layouts -- is its structure, and on a frozen coarsening none of it can move, so a refresh must
    make none; and the refreshed hierarchy and smoother pieces must be exactly the ones an inverse
    making every plan afresh arrives at (the fit is one compiled function of the values).
    """
    cached = SimpleSmoothedInverse(_saddle(), 4, **_SETTINGS, frozen_coarsening=True)
    fresh = SimpleSmoothedInverse(_saddle(), 4, **_SETTINGS, frozen_coarsening=True)
    for inverse in (cached, fresh):
        inverse.refactor_block((_saddle() * 1.7).tocsr())

    made = []
    plan = saddle_multigrid.simple_plan

    def counted(*args, **kwargs):
        made.append(1)
        return plan(*args, **kwargs)

    monkeypatch.setattr(saddle_multigrid, "simple_plan", counted)
    refreshed = (_saddle(seed=1) * 0.6).tocsr()
    # Stored exact zeros off the diagonal: a plain product would drop what the patterns keep.
    rows = np.repeat(np.arange(refreshed.shape[0]), np.diff(refreshed.indptr))
    off_diagonal = np.flatnonzero(rows != refreshed.indices)
    refreshed.data[off_diagonal[::7]] = 0.0
    cached.refactor_block(refreshed)
    assert not made
    fresh._plans_for = ()  # every plan made afresh
    fresh.refactor_block(refreshed)
    assert made
    monkeypatch.undo()

    got = jax.tree.leaves(cached.traced_cycle().state)
    want = jax.tree.leaves(fresh.traced_cycle().state)
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_factory_builds_what_the_field_split_expects() -> None:
    """``SimpleSmoothed`` returns the ``(block, n_fields) -> inverse`` shape the split calls."""
    a = _saddle()

    inverse = SimpleSmoothed(**_SETTINGS)(a, 4)

    assert inverse.n_dofs == a.shape[0]


def test_the_frobenius_block_inverse_beats_inverting_the_cell_block_alone() -> None:
    """The Frobenius fit must use ``F_ii^T``, and dropping that transpose is silent at one field/cell.

    ``M_i = F_ii^T (R_i R_i^T)^-1`` minimizes ``||I - M F||_F`` over block-diagonal ``M``. Writing
    ``F_ii (R R^T)^-1`` instead is correct only for a symmetric cell block, and reduces to the identical
    scalar formula at one field per cell — so a scalar reduction check cannot catch the error, and in
    the solver it took an arm from 7 restart cycles to no convergence. Pinned against a brute-force
    minimizer on a deliberately NONSYMMETRIC block.
    """
    a = _saddle(n_cells=40)
    n_cells, n_fields = 40, 4

    def block_inverse(frobenius):
        pieces = _simple_pieces(a, n_fields, frobenius=frobenius, block_splitting=True)
        return pieces.f_block_inverse.to_scipy()

    fitted, exact = block_inverse(True), block_inverse(False)
    velocity = a[: n_cells * 3, : n_cells * 3]
    identity = sp.eye(velocity.shape[0], format="csr")

    def splitting_error(m):
        return sp.linalg.norm(identity - m @ velocity)

    assert splitting_error(fitted) < splitting_error(exact)
    assert n_fields == 4  # the saddle's field count, stated so the slice above is readable


@pytest.mark.parametrize("block_splitting", [False, True])
def test_pieces_held_in_single_precision_are_the_double_pieces_rounded_once(
    block_splitting,
) -> None:
    """``precision="float32"`` rounds each piece as it is formed, to what rounding them afterwards gave.

    A level's pieces are rounded one at a time so that they are never all held in double beside their
    rounded copies. The rounding must be the one a single-precision split applies, entry for entry, and
    every floating piece must come out narrow -- a piece left double would be rounded later anyway,
    silently keeping the double copy alive.
    """
    from aquaflux.precondition.traced_cycle import in_precision

    a = _saddle()
    wide = _simple_pieces(a, 4, block_splitting=block_splitting)
    narrow = _simple_pieces(a, 4, block_splitting=block_splitting, precision="float32")
    assert (narrow.f_block_inverse is None) == (not block_splitting)
    rounded = jax.tree.leaves(in_precision(wide, "float32"))
    held = jax.tree.leaves(narrow)
    assert len(held) == len(rounded)
    for leaf, reference in zip(held, rounded, strict=True):
        if jnp.issubdtype(reference.dtype, jnp.floating):
            assert leaf.dtype == jnp.float32
        assert leaf.dtype == reference.dtype
        assert np.array_equal(np.asarray(leaf), np.asarray(reference))


def test_an_inverse_held_in_single_precision_forms_its_pieces_narrow() -> None:
    """The inverse's own precision reaches the pieces it derives, not only the cast applied after them."""
    a = _saddle()
    settings = {"strength_threshold": 0.25, "max_levels": 3, "max_coarse": 40}
    double = SimpleSmoothedInverse(a, 4, **settings)
    narrow = SimpleSmoothedInverse(a, 4, **settings, stored_precision="float32")
    pieces = narrow.derive_extras(double._hierarchy)
    floating = [
        leaf for leaf in jax.tree.leaves(pieces) if jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    assert floating and all(leaf.dtype == jnp.float32 for leaf in floating)
