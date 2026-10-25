"""The traced field split must be the host split's answer, without the host round trip.

The whole point of the traced composition is a property that leaves no trace in a result: whether the
preconditioned vector crossed the device boundary on its way back. So these tests pin both halves —
that it computes the same thing, and that it computes it without a callback.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.precondition import (
    AirReduction,
    FieldGroups,
    JacobiSmoothed,
    SimpleSmoothed,
    field_split_inverse,
)
from aquaflux.precondition.monolithic_policy import _TracedFormAdjoint
from aquaflux.precondition.padded_rows import _PaddedRowOperator
from aquaflux.precondition.traced_field_split import (
    TracedFieldSplit,
    contains_host_callback,
    traced_field_split,
)
from aquaflux.solve import with_transposes
from aquaflux.solve.root_adjoint import _transposed_preconditioner

N_CELLS, N_LEADING, N_TRAILING = 600, 4, 2


def _operator(seed: int = 0) -> sp.csr_matrix:
    """A diagonally-dominant operator with the field-major layout a split assumes."""
    n = (N_LEADING + N_TRAILING) * N_CELLS
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=0.01, random_state=seed, format="lil")
    a.setdiag(np.abs(rng.normal(size=n)) + 12.0)
    return sp.csr_matrix(a)


def _grid_operator(seed: int = 0) -> sp.csr_matrix:
    """A field-major operator over a 30 x 20 cell grid: a nonsymmetric five-point stencil per field pair.

    Local coupling only, which a reduction (lAIR) hierarchy needs -- the random operator above couples
    cells across the whole mesh, and an lAIR coarse grid built on it densifies and is refused.
    """
    rng = np.random.default_rng(seed)

    def chain(m: int) -> sp.spmatrix:
        return sp.diags(
            [-1.3 * np.ones(m - 1), 2.6 * np.ones(m), -0.7 * np.ones(m - 1)], [-1, 0, 1]
        )

    cells = sp.kronsum(chain(30), chain(20))
    n_fields = N_LEADING + N_TRAILING
    fields = np.eye(n_fields) + 0.1 * rng.normal(size=(n_fields, n_fields))
    a = sp.kron(fields, cells).tocsr()
    a.data = a.data * rng.uniform(0.8, 1.2, a.nnz)
    return a


def _groups() -> FieldGroups:
    return FieldGroups.by_counts(
        n_cells=N_CELLS, n_leading_fields=N_LEADING, n_trailing_fields=N_TRAILING
    )


def _pair():
    """A host split and a traced split over the SAME two inverse objects."""
    matrix, groups = _operator(), _groups()
    host = field_split_inverse(
        matrix,
        groups,
        leading_inverse=SimpleSmoothed(strength_threshold=0.25, max_levels=4, max_coarse=200),
        trailing_inverse=JacobiSmoothed(max_coarse=150),
    )
    traced = traced_field_split(host)
    return host, traced, groups


def test_the_traced_split_reproduces_the_host_split() -> None:
    """Same algebra, same inverses, same answer — to machine precision, not merely closely.

    Sharing the inverse *objects* is what makes this a test of the composition alone: any difference
    is the numpy-versus-traced arrangement of the slices, the coupling product and the concatenation,
    since both sides run the identical multigrid cycles underneath.
    """
    host, traced, groups = _pair()
    r = np.random.default_rng(7).normal(size=groups.n_dofs)

    expected = np.asarray(host.apply(r))
    got = np.asarray(traced.apply(jnp.asarray(r)))

    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-13


def test_the_transpose_comes_from_jax_and_matches_the_host_arrangement() -> None:
    """``M^T`` is derived, not written — and it agrees with the transpose the host arranges by hand.

    The host split has to arrange its transpose itself: reverse the solve order, transpose the
    coupling, and transpose each block inverse. This map is traced and linear, so
    :func:`jax.linear_transpose` gives ``M^T`` from the forward code. That is one implementation
    instead of two, and this asserts the two agree rather than assuming they must.
    """
    host, traced, groups = _pair()
    r = np.random.default_rng(11).normal(size=groups.n_dofs)

    expected = np.asarray(host.apply(r, transpose=True))
    (got,) = jax.linear_transpose(with_transposes(traced).matvec(), jnp.zeros(groups.n_dofs))(
        jnp.asarray(r)
    )

    assert np.linalg.norm(np.asarray(got) - expected) / np.linalg.norm(expected) < 1e-13


def _padded_operators(tree) -> list[_PaddedRowOperator]:
    return [
        leaf
        for leaf in jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, _PaddedRowOperator))
        if isinstance(leaf, _PaddedRowOperator)
    ]


@pytest.mark.parametrize("operator", [_operator, _grid_operator], ids=["held", "cell-level"])
def test_a_fitted_split_holds_no_transposes_and_an_adjoint_packs_them_for_its_solve_only(
    operator,
) -> None:
    """A forward solve never applies a transpose, so the split a step carries holds none.

    Every padded-row operator in the split -- each level of both hierarchies, the SIMPLE pieces and the
    coupling -- holds its rows alone. The adjoint's transpose solve packs the transposes into its own
    copy (``root_adjoint`` asks the factory's operators through ``with_transposes``), which applies
    ``Mᵀ`` exactly as the transpose the host arranges by hand, and leaves the step's split as it was.
    Both the held form and the cell-level form of the operators' columns are fitted.
    """
    groups = _groups()
    host = field_split_inverse(
        operator(),
        groups,
        leading_inverse=SimpleSmoothed(strength_threshold=0.25, max_levels=4, max_coarse=200),
        trailing_inverse=JacobiSmoothed(max_coarse=150),
    )
    traced = traced_field_split(host)
    operators = _padded_operators(traced)
    assert len(operators) > 6  # the coupling, the SIMPLE pieces, and every level's operator
    assert all(op.transposed is None and op.t_values is None for op in operators)

    r = jnp.asarray(np.random.default_rng(13).normal(size=groups.n_dofs))
    factory = _TracedFormAdjoint(traced)
    transposed = _transposed_preconditioner(factory, r, r)
    expected = np.asarray(host.apply(np.asarray(r), transpose=True))
    got = np.asarray(transposed(r))
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-13
    # The copy belongs to the transpose solve; the factory the step carries still holds no transpose.
    assert all(op.transposed is None for op in _padded_operators(factory))

    packed = with_transposes(traced)
    assert all(op.t_values is not None for op in _padded_operators(packed))
    # Only the transposes were added: each operator's own arrays are the split's, not copies.
    for before, after in zip(operators, _padded_operators(packed), strict=True):
        assert after.values is before.values
    (again,) = jax.linear_transpose(packed.matvec(), jnp.zeros(groups.n_dofs))(r)
    np.testing.assert_array_equal(np.asarray(again), got)


def test_transposing_a_fitted_split_without_its_transposes_raises_by_name() -> None:
    """Never a silent fallback to the scatter-add transpose of every gather: the error names the fix."""
    _, traced, groups = _pair()
    with pytest.raises(RuntimeError, match="with_transposes"):
        jax.linear_transpose(traced.matvec(), jnp.zeros(groups.n_dofs))(jnp.ones(groups.n_dofs))


def test_the_traced_split_emits_no_host_callback() -> None:
    """The property the module exists for, and the one a result cannot show.

    A round trip to the host is invisible in the answer and shows up only as time — and as a
    synchronization that drains the device pipeline. So it is asserted on the jaxpr.
    """
    _, traced, groups = _pair()
    assert not contains_host_callback(traced.matvec(), jnp.zeros(groups.n_dofs))


def test_the_host_split_does_emit_one() -> None:
    """The control for the test above: without it, a detector that never fires would pass anything."""
    host, _, groups = _pair()

    def through_the_host_split(residual):
        return jax.pure_callback(
            lambda r: host.apply(r),
            jax.ShapeDtypeStruct((groups.n_dofs,), jnp.float64),
            residual,
        )

    assert contains_host_callback(through_the_host_split, jnp.zeros(groups.n_dofs))


def test_the_traced_split_is_a_fixed_linear_map() -> None:
    """Required by the non-flexible outer GMRES and by the transposed adjoint alike."""
    _, traced, groups = _pair()
    m = jax.jit(traced.matvec())
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=groups.n_dofs))
    y = jnp.asarray(rng.normal(size=groups.n_dofs))

    combined = m(2.5 * x - 0.75 * y)
    separate = 2.5 * m(x) - 0.75 * m(y)

    assert np.allclose(np.asarray(combined), np.asarray(separate), rtol=1e-10, atol=1e-12)


class _HostOnlyInverse:
    """A block inverse with the host interface and no traced cycle -- a factorization, say."""

    def __init__(self, block: sp.spmatrix, n_fields: int) -> None:
        self.n_dofs = block.shape[0]

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        return np.asarray(residual)


def test_a_host_only_inverse_is_refused_rather_than_silently_composed_on_the_host() -> None:
    """A host factorization has no traced cycle, and falling back would hide the round trip.

    A sequential triangular solve genuinely belongs on a CPU, so the refusal is the honest outcome —
    but it has to be loud, because a silent fallback to the host split would leave a caller believing
    a bundle was on device when the thing this module exists to remove was still there.
    """
    matrix, groups = _operator(), _groups()
    host = field_split_inverse(
        matrix,
        groups,
        leading_inverse=_HostOnlyInverse,
        trailing_inverse=JacobiSmoothed(max_coarse=150),
    )

    with pytest.raises(TypeError, match="leading inverse _HostOnlyInverse offers no traced cycle"):
        traced_field_split(host)


def _perturbed(matrix: sp.csr_matrix, seed: int) -> sp.csr_matrix:
    """The same sparsity pattern with different values -- a march refresh on a fixed mesh."""
    moved = matrix.copy()
    moved.data = moved.data * np.random.default_rng(seed).uniform(0.5, 1.5, moved.nnz)
    return moved


# Structure-preserving refreshes on both families: an isotropic aggregation reads only the pattern, and an
# lAIR refresh re-solves values on its frozen split. So a refresh moves values and no shapes.
_TRAILING_INVERSES = {
    "jacobi": JacobiSmoothed(max_coarse=150),
    "air": AirReduction(max_coarse=150),
}


@pytest.mark.parametrize("trailing", sorted(_TRAILING_INVERSES))
def test_a_split_rebuilt_after_a_refresh_reuses_the_compiled_solve_and_applies_the_refresh(
    trailing: str,
) -> None:
    """A refresh at unchanged shapes must be BOTH a compilation-cache hit AND a changed answer.

    Either alone is easy and wrong. Retracing every refresh is correct and pays a recompile per refresh;
    hitting the cache with the hierarchy baked in as a constant is fast and applies the *old*
    preconditioner with no error -- which a split reaching its inverses through a bound method did, since
    tracing the method read the hierarchy and closed over it. Both answers are compared with the
    refreshed split's own eager answer and the host split's; the stale build was off by order one.
    """
    matrix, groups = _grid_operator(), _groups()
    host = field_split_inverse(
        matrix,
        groups,
        leading_inverse=SimpleSmoothed(max_levels=4, max_coarse=200),
        trailing_inverse=_TRAILING_INVERSES[trailing],
    )
    traces = []

    @eqx.filter_jit
    def apply(split: TracedFieldSplit, residual: jnp.ndarray) -> jnp.ndarray:
        traces.append(1)
        return split.apply(residual)

    r = jnp.asarray(np.random.default_rng(5).normal(size=groups.n_dofs))
    before = apply(traced_field_split(host), r)

    refreshed = _perturbed(matrix, seed=9)
    host.refactor_block(refreshed)
    split = traced_field_split(host)
    after = apply(split, r)
    expected = split.apply(r)

    assert len(traces) == 1, "the refresh moved no shape, so it must not retrace"
    assert np.linalg.norm(np.asarray(after - before)) > 1e-3 * np.linalg.norm(np.asarray(before))
    np.testing.assert_allclose(np.asarray(after), np.asarray(expected), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(after), host.apply(np.asarray(r)), rtol=1e-12, atol=1e-14)


def test_both_hierarchies_ride_as_leaves_not_as_constants() -> None:
    """The hierarchies are arguments of the traced apply, so nothing array-sized is closed over.

    This is the structural statement of the property the refresh test pins behaviourally: a jaxpr that
    closes over an array has baked it into the compiled program, where a refresh cannot reach it.
    """
    _, traced, groups = _pair()
    leaves, _ = jax.tree.flatten(eqx.filter(traced, eqx.is_array))
    closed = jax.make_jaxpr(lambda split, r: split.apply(r))(traced, jnp.zeros(groups.n_dofs))

    hierarchy_leaves = jax.tree.leaves(eqx.filter(traced.leading_cycle, eqx.is_array))
    assert len(hierarchy_leaves) > 0
    assert len(leaves) > len(jax.tree.leaves(traced.coupling))
    assert not [c for c in closed.consts if np.ndim(c) > 0 and np.size(c) > 1]


def test_the_split_rides_into_a_jit_as_an_argument_without_retracing() -> None:
    """Every array it holds is a leaf, so new values of the same shapes reuse the compiled cycle.

    That is the same property :class:`~aquaflux.precondition.padded_rows._PaddedRowOperator` has and for the same
    reason: a mid-march refresh must not recompile the solve it is refreshing.
    """
    _, traced, groups = _pair()
    traces = []

    @jax.jit
    def apply(split: TracedFieldSplit, residual: jnp.ndarray) -> jnp.ndarray:
        traces.append(1)
        return split.apply(residual)

    r = jnp.asarray(np.random.default_rng(5).normal(size=groups.n_dofs))
    apply(traced, r).block_until_ready()
    # Every leaf scaled -- coupling and both hierarchies: values move, shapes do not.
    moved = jax.tree.map(lambda a: 1.5 * a if jnp.issubdtype(a.dtype, jnp.floating) else a, traced)
    apply(moved, r).block_until_ready()

    assert len(traces) == 1


def _equations(jaxpr):
    """Every equation of ``jaxpr``, those of the sub-programs a loop or a call holds included."""
    for equation in jaxpr.eqns:
        yield equation
        for parameter in equation.params.values():
            for sub in parameter if isinstance(parameter, tuple | list) else (parameter,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _equations(inner)


def test_a_single_precision_split_runs_in_single_precision_and_returns_double() -> None:
    """Rounded once after the fit, applied in single precision, widened back on the way out.

    It must agree with the double-precision split to single-precision rounding and NOT to double
    precision -- a split that ignored the setting would agree exactly -- and every arithmetic equation
    of the apply must produce single precision, so the bytes moved are the ones the setting is for.
    """
    host, double, groups = _pair()
    single = traced_field_split(host, "float32")
    floating = [
        leaf
        for leaf in jax.tree.leaves(eqx.filter(single, eqx.is_array))
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    assert floating and all(leaf.dtype == jnp.float32 for leaf in floating)
    r = jnp.asarray(np.random.default_rng(3).normal(size=groups.n_dofs))
    out = single.apply(r)
    assert out.dtype == jnp.float64
    reference = double.apply(r)
    difference = float(jnp.max(jnp.abs(out - reference)) / jnp.max(jnp.abs(reference)))
    assert 1e-12 < difference < 1e-5
    jaxpr = jax.make_jaxpr(single.apply)(r).jaxpr
    widened = [
        equation.primitive.name
        for equation in _equations(jaxpr)
        if equation.primitive.name != "convert_element_type"
        and any(getattr(v.aval, "dtype", None) == jnp.float64 for v in equation.outvars)
    ]
    assert widened == []


def test_a_single_precision_split_transposes_to_its_adjoint() -> None:
    """``jax.linear_transpose`` still yields the adjoint, to single-precision rounding."""
    host, _, groups = _pair()
    single = with_transposes(traced_field_split(host, "float32"))
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(size=groups.n_dofs))
    y = jnp.asarray(rng.normal(size=groups.n_dofs))
    (transposed,) = jax.linear_transpose(single.apply, x)(y)
    assert transposed.dtype == jnp.float64
    forward, backward = float(y @ single.apply(x)), float(transposed @ x)
    assert abs(forward - backward) / abs(forward) < 1e-5


def test_the_adjoint_applies_a_single_precision_split_in_double_and_so_exactly_linearly() -> None:
    """A transpose solve run to ``1e-10`` stagnates on a map that is linear only to ``1e-7``.

    Applied in single precision the split is not exactly linear; the adjoint's factory widens its
    rounded entries back to double precision, which is. The single-precision defect is asserted too, so
    the test cannot pass on a split that was never narrowed.
    """
    from aquaflux.precondition.monolithic_policy import _TracedFormAdjoint

    host, double, groups = _pair()
    single = traced_field_split(host, "float32")
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=groups.n_dofs))
    y = jnp.asarray(rng.normal(size=groups.n_dofs))

    def linearity_defect(apply) -> float:
        combined = apply(0.3 * x + 1.7 * y)
        return float(
            jnp.max(jnp.abs(combined - 0.3 * apply(x) - 1.7 * apply(y)))
            / jnp.max(jnp.abs(combined))
        )

    assert linearity_defect(single.apply) > 1e-10
    adjoint = _TracedFormAdjoint(single)(None)
    assert linearity_defect(adjoint) < 1e-13
    assert (
        1e-12
        < float(jnp.max(jnp.abs(adjoint(x) - double.apply(x))) / jnp.max(jnp.abs(double.apply(x))))
        < 1e-5
    )


def test_a_split_holding_its_coupling_in_single_precision_composes_the_same_single_split() -> None:
    """``stored_precision="float32"`` holds the coupling rounded once, and changes nothing applied.

    A split applied in single precision rounds its coupling when it is composed, so a host split holding
    the coupling in double kept a copy nothing applies. Held narrow, the coupling must be the double
    block rounded once, packed into single-precision slots (not a double copy of them), and the traced
    split composed from it must equal, leaf for leaf and bit for bit, the one composed from a split
    holding the coupling in double -- at the fit and after a refit to a new operator.

    One coupling entry sits below the single-precision normal range, where numpy's cast keeps a
    subnormal and the traced split's conversion flushes it to zero: the held coupling must round as the
    traced split does, or the two would differ in that entry.
    """
    groups = _groups()
    inverses = {
        "leading_inverse": SimpleSmoothed(strength_threshold=0.25, max_levels=4, max_coarse=200),
        "trailing_inverse": JacobiSmoothed(max_coarse=150),
    }
    tiny = (groups.n_leading_dofs + 5, 7)  # a trailing row, a leading column: the coupling

    def operator(scale: float = 1.0) -> sp.csr_matrix:
        a = (_operator() * scale).tolil()
        a[tiny] = 1e-40
        return sp.csr_matrix(a)

    wide = field_split_inverse(operator(), groups, **inverses)
    narrow = field_split_inverse(operator(), groups, **inverses, stored_precision="float32")
    for refit in (False, True):
        if refit:
            developed = operator(1.7)
            developed.setdiag(developed.diagonal() + 3.0)
            wide.refactor_block(developed)
            narrow.refactor_block(developed)
        assert wide.coupling.dtype == np.float64
        assert np.count_nonzero(wide.coupling.data == 1e-40) == 1
        assert narrow.coupling.dtype == np.float32
        assert np.array_equal(narrow.coupling.indices, wide.coupling.indices)
        assert narrow.coupling[tiny[0] - groups.n_leading_dofs, tiny[1]] == 0.0
        packed = narrow.padded_coupling()
        assert packed.values.dtype == packed.with_transpose().t_values.dtype == jnp.float32
        from_wide, from_narrow = (traced_field_split(split, "float32") for split in (wide, narrow))
        wide_leaves, narrow_leaves = jax.tree.leaves(from_wide), jax.tree.leaves(from_narrow)
        assert len(wide_leaves) == len(narrow_leaves)
        for held, reference in zip(narrow_leaves, wide_leaves, strict=True):
            assert held.dtype == reference.dtype
            assert np.array_equal(np.asarray(held), np.asarray(reference))


def test_a_coupling_precision_outside_the_two_is_refused() -> None:
    with pytest.raises(ValueError, match="stored_precision must be"):
        field_split_inverse(
            _operator(),
            _groups(),
            leading_inverse=SimpleSmoothed(max_coarse=200),
            trailing_inverse=JacobiSmoothed(max_coarse=150),
            stored_precision="bfloat16",
        )
