"""A column probed for a chosen cell pattern, and the strong couplings that choose it."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from aquaflux.precondition import (
    JacobianProbe,
    JacobianProbeSpec,
    MaterializedJacobian,
    MaterializedProblem,
    MaterializedSession,
    SimpleSmoothed,
    StrongCouplings,
)
from aquaflux.precondition.sparse_jacobian import (
    block_stencil_colouring,
    block_stencil_gather_map,
    column_probe_plan,
    materialize_block_jacobian,
)
from aquaflux.solve import FieldLayout

_N = 18
_OWNER, _NB = np.arange(_N - 1), np.arange(1, _N)


def _chain_matrix(far: dict[tuple[int, int], float], rng: np.random.Generator) -> sp.csr_matrix:
    """A two-field chain-graph matrix: every face coupling of magnitude at least one, plus ``far``.

    Column field 0 couples each cell to itself and its face neighbours in both row fields, and to the
    pairs in ``far`` (cell ``i`` to cell ``j`` beyond a face, value ``far[i, j]``) in row field 0 only.
    Column field 1 couples within one face only.
    """
    dense = np.zeros((2 * _N, 2 * _N))
    for i in range(_N):
        for j in range(max(0, i - 1), min(_N, i + 2)):
            for row_field in range(2):
                for column_field in range(2):
                    dense[row_field * _N + i, column_field * _N + j] = (
                        1.0 + rng.random()
                    ) * rng.choice((-1.0, 1.0))
    for (i, j), value in far.items():
        dense[i, j] = value
    return sp.csr_matrix(dense)


def _matvec_of(matrix):
    def matvec(v):
        return jnp.asarray(matrix @ np.asarray(v))

    return matvec


def _products_of(matrix, column):
    """Seeds on column field ``column`` alone, ``(batch, n)``, to responses ``(batch, 2 n)``."""

    def products(seeds):
        whole = np.zeros((seeds.shape[0], 2 * _N))
        whole[:, column * _N : (column + 1) * _N] = seeds
        return (matrix @ whole.T).T

    return products


#: Pairs beyond a face that column 0 couples in row field 0: two strong, two weak.
_STRONG = {(4, 6): 0.6, (9, 12): -0.3}
_WEAK = {(5, 7): 2.0e-4, (12, 15): -1.0e-4}


def test_a_column_probed_for_a_pattern_is_recovered_exactly_where_its_couplings_lie_in_it() -> None:
    """The far pairs in the pattern are read; the column takes few more colours than its short reach.

    A column probed at reach one alone loses the far pairs entirely, so the comparison shows what the
    pattern buys, and the plan's other column and stored pattern are untouched.
    """
    matrix = _chain_matrix(_STRONG, np.random.default_rng(3))
    short = column_probe_plan(_OWNER, _NB, _N, (1, 1), pattern_reach=3)
    near = sp.coo_matrix(
        (np.ones(short.pattern_rows.size), (short.pattern_rows, short.pattern_cols))
    )
    within_one = np.abs(near.row - near.col) <= 1
    rows = np.r_[near.row[within_one], [i for i, _ in _STRONG]]
    cols = np.r_[near.col[within_one], [j for _, j in _STRONG]]

    planned = short.with_column_pattern(0, rows, cols)
    recovered = materialize_block_jacobian(_matvec_of(matrix), planned)
    without = materialize_block_jacobian(_matvec_of(matrix), short)

    assert abs(recovered - matrix).max() < 1e-13
    assert abs(without - matrix).max() >= 0.3 - 1e-12  # the short probe drops the far pairs
    assert planned.n_colours[1] == short.n_colours[1]
    assert (
        short.n_colours[0]
        < planned.n_colours[0]
        < block_stencil_colouring(_OWNER, _NB, _N, 3).n_colours
    )
    np.testing.assert_array_equal(planned.pattern_rows, short.pattern_rows)
    assert planned.pattern_reach == short.pattern_reach == 3


def test_a_pattern_pair_outside_the_stored_pattern_is_refused() -> None:
    short = column_probe_plan(_OWNER, _NB, _N, (1, 1), pattern_reach=2)
    with pytest.raises(ValueError, match="couples cell 2 to cell 5"):
        short.with_column_pattern(0, np.array([2]), np.array([5]))


def _selection(couplings, matrix, reach=1):
    return couplings.pattern(
        _products_of(matrix, couplings.column),
        {0: 0, 1: _N},
        _OWNER,
        _NB,
        _N,
        reach,
        block_stencil_colouring(_OWNER, _NB, _N, 3),
        batch=3,
    )


def test_the_far_pairs_kept_are_exactly_those_above_the_fraction_of_their_rows_weakest_face() -> (
    None
):
    """Every face coupling has magnitude at least one, so a fraction of 0.01 sits between the strong
    pairs (0.3, 0.6) and the weak ones (1e-4, 2e-4)."""
    matrix = _chain_matrix(_STRONG | _WEAK, np.random.default_rng(5))
    rows, cols, kept = _selection(StrongCouplings(column=0, fraction=0.01), matrix)
    far = {(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if abs(i - j) > 1}
    assert far == set(_STRONG)
    assert kept == len(_STRONG)
    within = {(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if abs(i - j) <= 1}
    assert len(within) == _N + 2 * (_N - 1)  # every pair within the column's own reach


def test_the_fraction_is_measured_against_each_rows_own_weakest_face_coupling() -> None:
    """Row 9's face couplings are scaled down to 0.01, so its far pair of 3e-4 passes there while an
    equal far pair in an ordinary row does not."""
    matrix = _chain_matrix({(9, 12): 3.0e-4, (3, 6): 3.0e-4}, np.random.default_rng(7)).tolil()
    for j in (8, 10):
        matrix[9, j] = 0.01
    rows, cols, _ = _selection(StrongCouplings(column=0, fraction=0.01), matrix.tocsr())
    far = {(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if abs(i - j) > 1}
    assert far == {(9, 12)}


def test_a_face_coupling_that_vanishes_is_not_the_rows_yardstick() -> None:
    """Row 9's coupling to cell 8 is exactly zero, as a structurally empty entry is; its weakest face
    coupling is then the other one, and its weak far pair is still left out rather than every pair."""
    matrix = _chain_matrix({(9, 12): 1.0e-4}, np.random.default_rng(9)).tolil()
    matrix[9, 8] = 0.0
    rows, cols, _ = _selection(StrongCouplings(column=0, fraction=0.01), matrix.tocsr())
    far = {(int(i), int(j)) for i, j in zip(rows, cols, strict=True) if abs(i - j) > 1}
    assert far == set()


def test_only_the_named_row_fields_are_judged() -> None:
    """The far pairs sit in row field 0; judging row field 1 alone finds none of them."""
    matrix = _chain_matrix(_STRONG, np.random.default_rng(5))
    _, _, kept_own = _selection(StrongCouplings(column=0, fraction=0.01), matrix)
    _, _, kept_other = _selection(StrongCouplings(column=0, fraction=0.01, rows=(1,)), matrix)
    assert (kept_own, kept_other) == (len(_STRONG), 0)


def test_a_selection_needs_a_reach_shorter_than_the_pattern() -> None:
    matrix = _chain_matrix({}, np.random.default_rng(1))
    with pytest.raises(ValueError, match="shorter than the pattern's 3"):
        _selection(StrongCouplings(column=0, fraction=0.01), matrix, reach=3)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"column": -1, "fraction": 0.01}, "field index"),
        ({"column": 3, "fraction": 0.0}, "positive"),
        ({"column": 3, "fraction": 0.01, "rows": ()}, "at least one"),
    ],
)
def test_a_strong_couplings_value_refuses_what_it_cannot_mean(arguments, message) -> None:
    with pytest.raises(ValueError, match=message):
        StrongCouplings(**arguments)


def test_a_probe_spec_refuses_strong_couplings_with_every_column_at_the_full_reach() -> None:
    with pytest.raises(ValueError, match="column_reach unset"):
        JacobianProbeSpec(strong_couplings=StrongCouplings(column=3, fraction=0.01))


# --- the session ------------------------------------------------------------------------------------


_LAYOUT = FieldLayout.cell_fields(_N, a=1, b=1)


class _Linear:
    """A residual ``J x`` with a fixed matrix, so its Jacobian is known everywhere."""

    def __init__(self, matrix) -> None:
        self.matrix = jnp.asarray(matrix.toarray())

    def residual(self, state):
        return self.matrix @ state


class _ChainProblem(MaterializedProblem):
    def __init__(self, matrix) -> None:
        self._assembler = _Linear(matrix)

    assembler = property(lambda self: self._assembler)
    layout = property(lambda self: _LAYOUT)

    def with_assembler(self, assembler):
        return self

    def probe(self, settings, groups):
        plan = column_probe_plan(_OWNER, _NB, _N, settings["column_reach"], pattern_reach=3)
        return JacobianProbe(plan, block_stencil_gather_map(plan))

    def groups(self):
        return None

    def bind_march(self, march):
        return march

    def shift_source(self, state, march):
        raise NotImplementedError

    def build_step(self, state, policy, march, base_regime):
        raise NotImplementedError


def test_a_session_re_plans_the_column_from_its_jacobian_and_reports_what_it_chose() -> None:
    """The re-planned probe recovers the strong far pairs and leaves the weak ones out."""
    matrix = _chain_matrix(_STRONG | _WEAK, np.random.default_rng(11))
    spec = MaterializedJacobian(
        SimpleSmoothed(),
        probe=JacobianProbeSpec(
            column_reach=(1, 1), strong_couplings=StrongCouplings(column=0, fraction=0.01)
        ),
    )
    problem = _ChainProblem(matrix)
    lines = []
    session = MaterializedSession(spec, problem, reports={"probe": lines.append})
    probe = session._probe_for()
    replanned = session._with_strong_couplings(probe, problem.assembler, jnp.zeros(_LAYOUT.size))

    recovered = materialize_block_jacobian(_matvec_of(matrix), replanned.plan).toarray()
    for (i, j), value in _STRONG.items():
        assert recovered[i, j] == pytest.approx(value, abs=5e-4)  # weak pairs fold in at most
    for i, j in _WEAK:
        assert recovered[i, j] == 0.0
    selection = session.coupling_selection
    assert (selection.column, selection.kept) == (0, len(_STRONG))
    assert selection.reach_colours == probe.plan.n_colours[0] < selection.colours
    assert selection.probes == replanned.plan.n_probes
    assert lines == [
        f"strong couplings: column 0 keeps 2 pairs beyond reach 1, {selection.reach_colours} -> "
        f"{selection.colours} colours, {selection.probes} probes per fit"
    ]


def test_a_probe_whose_plan_carries_no_cell_graph_refuses_a_selection() -> None:
    class _Uniform(_ChainProblem):
        def probe(self, settings, groups):
            plan = column_probe_plan(_OWNER, _NB, _N, settings["column_reach"], pattern_reach=3)
            plan.edges = None  # as a plan built from a colouring alone has none
            return JacobianProbe(plan, block_stencil_gather_map(plan))

    matrix = _chain_matrix({}, np.random.default_rng(2))
    spec = MaterializedJacobian(
        SimpleSmoothed(),
        probe=JacobianProbeSpec(
            column_reach=(1, 1), strong_couplings=StrongCouplings(column=0, fraction=0.01)
        ),
    )
    problem = _Uniform(matrix)
    session = MaterializedSession(spec, problem)
    with pytest.raises(ValueError, match="does not carry its cell graph"):
        session._with_strong_couplings(
            session._probe_for(), problem.assembler, jnp.zeros(_LAYOUT.size)
        )
