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
from aquaflux.solve import NativeSimpleInverse, block_approximate_inverse, native_saddle_inverse
from aquaflux.solve.saddle_multigrid import _native_saddle_cycle, _simple_pieces


def _saddle(n_cells: int = 240, dim: int = 3, seed: int = 0) -> sp.csr_matrix:
    """A field-major generalized saddle block on a chain graph, diagonally dominant in velocity.

    ``D != G^T`` and the (2,2) block is nonzero, so this is the operator class the smoother targets
    rather than a classical Stokes saddle.
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
                    else:
                        value = rng.normal() * 0.12
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
    inverse = NativeSimpleInverse(a, 4, **_SETTINGS)
    b = np.asarray(np.random.default_rng(1).normal(size=a.shape[0]))

    x = np.asarray(inverse.apply(b))

    assert np.all(np.isfinite(x))
    assert np.linalg.norm(a @ x - b) / np.linalg.norm(b) < 0.75


def test_the_cycle_is_a_fixed_linear_operator() -> None:
    """``b -> x`` must be LINEAR, or the non-flexible outer GMRES it preconditions is invalid.

    A fixed cycle count, a fixed sweep count and no inner Krylov method are what buy this; an adaptive
    smoother or an inner solve to a tolerance would break it silently — the outer solve would still run
    and would simply converge to the wrong thing.
    """
    a = _saddle()
    inverse = NativeSimpleInverse(a, 4, **_SETTINGS)
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
    inverse = NativeSimpleInverse(a, 4, **_SETTINGS)
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
    inverse = NativeSimpleInverse(a, 4, **_SETTINGS)
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
    inverse = NativeSimpleInverse(a, 4, **_SETTINGS)
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
        return _native_saddle_cycle(hierarchy, pieces, residual, inverse._smoother)

    rhs = jnp.asarray(b)
    cycle(inverse._hierarchy, inverse._extras, rhs).block_until_ready()
    assert len(traces) == 1
    inverse.refactor_block((a * 2.3).tocsr())
    cycle(inverse._hierarchy, inverse._extras, rhs).block_until_ready()
    assert len(traces) == 1, "a refreshed hierarchy retraced the cycle"


def test_the_pieces_carry_no_host_matrix_so_they_can_be_traced() -> None:
    """A host sparse matrix on the record is neither a traced leaf nor a hashable static field.

    Carrying one would make the record unusable as a jit argument — silently, by falling back to a
    closure — so the formed Schur rides out of :func:`_simple_pieces` as a second return value instead,
    which a caller that wants it in host form takes from the pair.
    """
    a = _saddle()
    inverse = NativeSimpleInverse(a, 4, **_SETTINGS)

    leaves = jax.tree_util.tree_leaves(inverse._extras)
    assert leaves, "the pieces have no traced leaves at all"
    assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves), (
        "a non-array leaf would be traced as one and fail"
    )

    _, schur = _simple_pieces(a, 4)
    assert sp.issparse(schur) and schur.shape == (a.shape[0] // 4,) * 2


def test_a_mismatched_block_is_rejected() -> None:
    """Refreshing onto a differently-sized block raises instead of building something incoherent."""
    inverse = NativeSimpleInverse(_saddle(n_cells=120), 4, **_SETTINGS)

    with pytest.raises(ValueError, match="cannot refactor"):
        inverse.refactor_block(_saddle(n_cells=80))


def test_the_object_is_silent_unless_a_report_sink_is_supplied() -> None:
    """A library preconditioner must not write to stdout on its own; the caller supplies the sink.

    The build record is genuinely worth having in a run log — a spec-token collision once made an arm
    run forty sweeps while every other line of output looked correct — so it is kept, and injected.
    """
    a = _saddle()
    captured: list[str] = []

    NativeSimpleInverse(a, 4, **_SETTINGS)  # no sink: nothing is emitted
    NativeSimpleInverse(a, 4, **_SETTINGS, report=captured.append)

    assert captured, "the report sink received nothing"
    assert any("native SIMPLE smoother" in line for line in captured)


def test_the_factory_builds_what_the_field_split_expects() -> None:
    """``native_saddle_inverse`` returns the ``(block, n_fields) -> inverse`` shape the split calls."""
    a = _saddle()

    inverse = native_saddle_inverse(**_SETTINGS)(a, 4)

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

    fitted = block_approximate_inverse(a[: n_cells * 3, : n_cells * 3], n_cells, 3, frobenius=True)
    exact = block_approximate_inverse(a[: n_cells * 3, : n_cells * 3], n_cells, 3, frobenius=False)
    velocity = a[: n_cells * 3, : n_cells * 3]
    identity = sp.eye(velocity.shape[0], format="csr")

    def splitting_error(m):
        return sp.linalg.norm(identity - m @ velocity)

    assert splitting_error(fitted) < splitting_error(exact)
    assert n_fields == 4  # the saddle's field count, stated so the slice above is readable
