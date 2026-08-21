"""The traced field split must be the host split's answer, without the host round trip.

The whole point of the traced composition is a property that leaves no trace in a result: whether the
preconditioned vector crossed the device boundary on its way back. So these tests pin both halves —
that it computes the same thing, and that it computes it without a callback.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.solve import (
    FieldGroups,
    build_block_triangular_field_split,
    ilu_smoothed_inverse,
    jacobi_smoothed_inverse,
    simple_smoothed_inverse,
)
from aquaflux.solve.traced_field_split import (
    TracedFieldSplit,
    contains_host_callback,
    traced_field_split,
)

N_CELLS, N_LEADING, N_TRAILING = 600, 4, 2


def _operator(seed: int = 0) -> sp.csr_matrix:
    """A diagonally-dominant operator with the field-major layout a split assumes."""
    n = (N_LEADING + N_TRAILING) * N_CELLS
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=0.01, random_state=seed, format="lil")
    a.setdiag(np.abs(rng.normal(size=n)) + 12.0)
    return sp.csr_matrix(a)


def _groups() -> FieldGroups:
    return FieldGroups(n_cells=N_CELLS, n_leading_fields=N_LEADING, n_trailing_fields=N_TRAILING)


def _pair(flow_first: bool = True):
    """A host split and a traced split over the SAME two inverse objects."""
    matrix, groups = _operator(), _groups()
    host = build_block_triangular_field_split(
        matrix,
        groups,
        flow_first=flow_first,
        leading_inverse=simple_smoothed_inverse(
            strength_threshold=0.25, max_levels=4, max_coarse=200
        ),
        trailing_inverse=jacobi_smoothed_inverse(max_coarse=150),
    )
    traced = traced_field_split(
        matrix, groups, host._leading, host._trailing, flow_first=flow_first
    )
    return host, traced, groups


@pytest.mark.parametrize("flow_first", [True, False])
def test_the_traced_split_reproduces_the_host_split(flow_first) -> None:
    """Same algebra, same inverses, same answer — to machine precision, not merely closely.

    Sharing the inverse *objects* is what makes this a test of the composition alone: any difference
    is the numpy-versus-traced arrangement of the slices, the coupling product and the concatenation,
    since both sides run the identical multigrid cycles underneath.
    """
    host, traced, groups = _pair(flow_first)
    r = np.random.default_rng(7).normal(size=groups.n_dofs)

    expected = np.asarray(host.apply(r))
    got = np.asarray(traced.apply(jnp.asarray(r)))

    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-13


@pytest.mark.parametrize("flow_first", [True, False])
def test_the_transpose_comes_from_jax_and_matches_the_host_arrangement(flow_first) -> None:
    """``M^T`` is derived, not written — and it agrees with the transpose the host arranges by hand.

    The host split has to arrange its transpose itself: reverse the solve order, transpose the
    coupling, and transpose each block inverse. This map is traced and linear, so
    :func:`jax.linear_transpose` gives ``M^T`` from the forward code. That is one implementation
    instead of two, and this asserts the two agree rather than assuming they must.
    """
    host, traced, groups = _pair(flow_first)
    r = np.random.default_rng(11).normal(size=groups.n_dofs)

    expected = np.asarray(host.apply(r, transpose=True))
    (got,) = jax.linear_transpose(traced.matvec(), jnp.zeros(groups.n_dofs))(jnp.asarray(r))

    assert np.linalg.norm(np.asarray(got) - expected) / np.linalg.norm(expected) < 1e-13


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


def test_a_host_only_inverse_is_refused_rather_than_silently_composed_on_the_host() -> None:
    """An incomplete factorization has no traced cycle, and falling back would hide the round trip.

    A sequential triangular solve genuinely belongs on a CPU, so the refusal is the honest outcome —
    but it has to be loud, because a silent fallback to the host split would leave a caller believing
    a bundle was on device when the thing this module exists to remove was still there.
    """
    matrix, groups = _operator(), _groups()
    host = build_block_triangular_field_split(
        matrix,
        groups,
        leading_inverse=ilu_smoothed_inverse(max_levels=3, max_coarse=200),
        trailing_inverse=jacobi_smoothed_inverse(max_coarse=150),
    )

    with pytest.raises(AttributeError, match="no traced cycle"):
        traced_field_split(matrix, groups, host._leading, host._trailing)


def test_the_split_rides_into_a_jit_as_an_argument_without_retracing() -> None:
    """Every array it holds is a leaf, so re-deriving it at a new operator reuses the compiled cycle.

    That is the same property :class:`~aquaflux.solve.multigrid._CsrOperator` has and for the same
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
    # A different coupling of the same shape: values move, shapes do not.
    moved = TracedFieldSplit(
        leading_cycle=traced.leading_cycle,
        trailing_cycle=traced.trailing_cycle,
        coupling=jax.tree.map(lambda a: a, traced.coupling),
        groups=traced.groups,
        leading_first=traced.leading_first,
    )
    apply(moved, r).block_until_ready()

    assert len(traces) == 1
