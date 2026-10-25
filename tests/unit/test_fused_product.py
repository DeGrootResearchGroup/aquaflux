"""Unit tests for the fused padded-row products (``precondition/fused_product.py``).

The fused loop is called from inside compiled programs as a foreign function, so these check it the
way the preconditioner reaches it: through each form of ``PaddedColumns`` packed from a real pattern,
against a serial product computed here in numpy, in the operator's own precision and slot order.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.precondition import fused_product
from aquaflux.precondition.entry_map import SparsePattern
from aquaflux.precondition.kronecker_pattern import KroneckerPattern
from aquaflux.precondition.padded_rows import (
    _HeldColumns,
    _KroneckerColumns,
    _PaddedLayout,
    _PaddedRowOperator,
)


def _cell_pattern(shape, seed):
    """A random cell pattern with rows of uneven length, one of them empty, columns ascending."""
    matrix = sp.random(*shape, density=0.35, random_state=seed, format="lil")
    matrix[2, :] = 0.0
    matrix = sp.csr_matrix(matrix)
    matrix.sort_indices()
    return SparsePattern(matrix)


def _operator(fields, cell_shape, dtype, seed=3, width=None):
    """An operator storing a cell pattern over every pair of ``fields``, held at cell level."""
    pattern = KroneckerPattern(_cell_pattern(cell_shape, seed), *fields).dof_pattern()
    values = np.random.default_rng(seed).standard_normal(pattern.nnz).astype(dtype)
    layout = _PaddedLayout.of(pattern, width, fields=fields)
    return layout.operator(pattern.csr(values)), pattern.csr(values)


def _serial(op: _PaddedRowOperator, x: np.ndarray) -> np.ndarray:
    """``A x`` summed row by row in slot order, in the operator's own precision."""
    values = np.asarray(op.values)
    columns = np.asarray(op.columns)
    lengths = np.asarray(op.row_lengths)
    out = np.zeros(values.shape[0], dtype=values.dtype)
    for row in range(values.shape[0]):
        total = values.dtype.type(0.0)
        for slot in range(lengths[row]):
            total = values.dtype.type(total + values[row, slot] * x[columns[row, slot]])
        out[row] = total
    return out


_apply = jax.jit(lambda op, x: op.apply(x))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("fields", "cell_shape"), [((4, 4), (9, 9)), ((2, 3), (7, 5))])
def test_the_cell_target_decodes_every_slot_and_sums_it_in_slot_order(dtype, fields, cell_shape):
    """Bit for bit the serial product: a wrong field, column or row decode would move a row's sum.

    The rectangular case has more column fields than row fields and fewer column cells than row cells,
    so a decode mixing the two counts reads the wrong entry of ``x``.
    """
    op, _ = _operator(fields, cell_shape, dtype)
    assert isinstance(op.rows, _KroneckerColumns)
    x = np.random.default_rng(1).standard_normal(op.shape[1]).astype(dtype)
    got = np.asarray(
        fused_product.cell_product(
            op.rows.cells,
            op.rows.lengths,
            op.values,
            jnp.asarray(x),
            op.rows.other_fields,
            op.rows.n_other,
        )
    )
    assert got.dtype == dtype
    np.testing.assert_array_equal(got, _serial(op, x))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_held_target_sums_each_row_in_slot_order(dtype):
    """Bit for bit the serial product on an operator whose rows are padded beyond their entries.

    A slot read from the wrong row or column moves a row's sum; a padding slot holds zero, so whether
    the loop reads it is not something this can tell.
    """
    a = sp.random(40, 30, density=0.2, random_state=5, format="csr").astype(dtype)
    op = _PaddedRowOperator.from_scipy(a, width=int(np.diff(a.indptr).max()) + 3)
    assert isinstance(op.rows, _HeldColumns)
    x = np.random.default_rng(2).standard_normal(30).astype(dtype)
    got = np.asarray(
        fused_product.held_product(op.rows.held_columns, op.rows.lengths, op.values, x)
    )
    np.testing.assert_array_equal(got, _serial(op, x))


def test_an_operators_product_is_the_matrix_product_whichever_form_computes_it(monkeypatch):
    """Both forms of ``PaddedColumns.product`` give ``A x`` to rounding, forward and transposed."""
    op, matrix = _operator((3, 3), (8, 8), np.float64)
    x = np.random.default_rng(4).standard_normal(op.shape[1])
    y = np.random.default_rng(6).standard_normal(op.shape[0])
    packed = op.with_transpose()
    transpose = jax.jit(lambda o, v: jax.linear_transpose(o.apply, jnp.zeros(o.shape[1]))(v)[0])
    results = {}
    for fused in (True, False):
        monkeypatch.setattr(fused_product, "fused_products", lambda fused=fused: fused)
        jax.clear_caches()  # the choice is made at trace time, and a traced product is cached
        results[fused] = (np.asarray(_apply(op, x)), np.asarray(transpose(packed, y)))
    for forward, backward in results.values():
        np.testing.assert_allclose(forward, matrix @ x, rtol=0, atol=1e-13)
        np.testing.assert_allclose(backward, matrix.T @ y, rtol=0, atol=1e-13)


def test_an_operator_on_the_cpu_compiles_its_product_to_the_fused_target_when_it_is_chosen(
    monkeypatch,
):
    """The routing: chosen, the compiled product is the foreign call and no gather; refused, the gather."""
    op, _ = _operator((2, 2), (6, 6), np.float32)
    x = jnp.ones(op.shape[1], dtype=jnp.float32)
    for chosen in (True, False):
        monkeypatch.setattr(fused_product, "fused_products", lambda chosen=chosen: chosen)
        jax.clear_caches()
        compiled = jax.jit(lambda o, v: o.apply(v)).lower(op, x).compile().as_text()
        assert (fused_product.CELL_TARGET in compiled) is chosen
        assert ("gather" in compiled) is not chosen


def test_a_mixed_precision_product_keeps_the_gathered_form(monkeypatch):
    """The loop takes one floating type for both operands; a float64 vector on float32 values promotes."""
    monkeypatch.setattr(fused_product, "fused_products", lambda: True)
    jax.clear_caches()
    op, _ = _operator((2, 2), (6, 6), np.float32)
    x = np.random.default_rng(7).standard_normal(op.shape[1])
    compiled = jax.jit(lambda o, v: o.apply(v)).lower(op, jnp.asarray(x)).compile().as_text()
    assert fused_product.CELL_TARGET not in compiled
    assert _apply(op, jnp.asarray(x)).dtype == jnp.float64


def test_the_choice_needs_a_threading_layer_that_several_threads_may_enter(monkeypatch):
    """The ``workqueue`` layer aborts on concurrent entry, so with it the gathered form is kept."""
    monkeypatch.delenv("AQUAFLUX_DISABLE_FUSED_PRODUCT", raising=False)
    monkeypatch.setattr(fused_product.numba, "threading_layer", lambda: "workqueue")
    assert fused_product.fused_products.__wrapped__() is False
    monkeypatch.setattr(fused_product.numba, "threading_layer", lambda: "omp")
    assert fused_product.fused_products.__wrapped__() is True


def test_the_environment_turns_the_fused_product_off(monkeypatch):
    monkeypatch.setenv("AQUAFLUX_DISABLE_FUSED_PRODUCT", "1")
    assert fused_product.fused_products.__wrapped__() is False


def test_registration_checks_both_targets_against_numpy():
    """The registration-time self-check passes here, so a disagreement it reports would be real."""
    assert fused_product.registered()
    assert fused_product._self_check() is None
