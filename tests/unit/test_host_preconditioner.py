"""Unit tests for the frozen host preconditioner contract and its shared JAX wrapper.

Structural rather than numerical: what each concrete preconditioner *computes* is pinned by its own
test module, and what is pinned here is that the three share one application path and one declared
contract. The failure these guard against is a future divergence -- someone re-adding a private
``matvec``, or a base reaching for a capability only some factorizations have, which is exactly how a
``has_native_solve`` lookup came to raise on the field split while a ``getattr`` default hid it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from aquaflux.solve import (
    HostFactors,
    HostPreconditioner,
    MonolithicAmgPreconditioner,
    MonolithicIlutPreconditioner,
    MonolithicLuPreconditioner,
)

FAMILY = (
    MonolithicIlutPreconditioner,
    MonolithicLuPreconditioner,
    MonolithicAmgPreconditioner,
)


class _ExactInverse:
    """An exact block inverse, standing in for a multigrid V-cycle so no PETSc build is needed."""

    def __init__(self, block: np.ndarray) -> None:
        self._inverse = np.linalg.inv(np.asarray(block, dtype=np.float64))

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        matrix = self._inverse.T if transpose else self._inverse
        return matrix @ np.asarray(residual, dtype=np.float64)


class _Doubling:
    """A trivial `HostFactors`: ``M`` doubles, ``M^T`` triples, so the two are distinguishable."""

    n_dofs = 4

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        return np.asarray(residual) * (3.0 if transpose else 2.0)


def test_every_host_preconditioner_shares_one_application_path() -> None:
    """`matvec` was written out three times, byte for byte. It is now written once."""
    for cls in FAMILY:
        assert issubclass(cls, HostPreconditioner)
        assert "matvec" not in vars(cls), (
            f"{cls.__name__} defines its own matvec, which the shared base owns"
        )


def test_the_wrapper_applies_the_inverse_and_its_transpose() -> None:
    """The transpose is what the adjoint's transpose solve calls, so it is not an optional extra."""
    pc = HostPreconditioner(_Doubling())
    residual = jnp.asarray([1.0, 2.0, 3.0, 4.0])

    assert np.allclose(np.asarray(pc.matvec()(residual)), np.asarray(residual) * 2.0)
    assert np.allclose(np.asarray(pc.matvec(transpose=True)(residual)), np.asarray(residual) * 3.0)


def test_the_callback_reads_the_factors_at_call_time_not_at_build_time() -> None:
    """The property a mid-march refresh rests on, asserted rather than assumed.

    ``refresh_in_place`` rebinds ``factors`` on an object whose *identity* is part of the compiled
    solve's pytree structure. If the callback captured the factorization instead of reading it through
    ``self``, a refresh would silently keep preconditioning with the stale one -- and nothing about the
    solve would look wrong.
    """
    pc = HostPreconditioner(_Doubling())
    apply = pc.matvec()  # built ONCE, before the refresh
    residual = jnp.asarray([1.0, 1.0, 1.0, 1.0])
    assert np.allclose(np.asarray(apply(residual)), 2.0)

    class _Tenfold(_Doubling):
        def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
            return np.asarray(residual) * 10.0

    pc.factors = _Tenfold()
    assert np.allclose(np.asarray(apply(residual)), 10.0), (
        "the already-built matvec did not pick up the refreshed factors"
    )


def test_the_real_factor_types_satisfy_the_declared_contract() -> None:
    """The contract claims to be what every frozen inverse in the package already provides.

    Checked against the real types rather than the docstring, since a `Protocol` that nothing is
    verified against is a comment.

    The split is assembled from stub block inverses rather than through
    ``build_block_triangular_field_split``, which would build real V-cycles and so need ``petsc4py`` --
    an *optional* dependency the unit tier does not install. What is under test is the split's own
    contract, and that does not depend on what inverts its blocks.
    """
    from aquaflux.solve.field_split import BlockTriangularFieldSplit, FieldGroups
    from aquaflux.solve.ilut_preconditioner import factorize_ilut
    from aquaflux.solve.lu_preconditioner import factorize_lu

    n = 12
    operator = (sp.random(n, n, density=0.4, random_state=0, format="csr") + sp.eye(n) * 5).tocsr()
    groups = FieldGroups(n_cells=6, n_leading_fields=1, n_trailing_fields=1)
    leading, _, trailing_by_leading, trailing = groups.blocks(operator)
    split = BlockTriangularFieldSplit(
        _ExactInverse(leading.toarray()),
        _ExactInverse(trailing.toarray()),
        trailing_by_leading,
        groups,
    )

    for factors in (
        factorize_ilut(operator, 2),
        factorize_lu(operator, backend="scipy"),
        split,
    ):
        assert isinstance(factors, HostFactors), f"{type(factors).__name__} is not HostFactors"
        assert factors.n_dofs == n


def test_the_base_asks_its_factors_for_nothing_beyond_the_declared_contract() -> None:
    """Nothing beyond ``n_dofs`` and ``apply`` may be reached for through ``self.factors``.

    The base is what every family member inherits, so a capability it reaches for becomes a requirement
    on *all* of them -- including the block-triangular splits and the patch smoother, which are not
    factorizations and cannot answer factorization questions. That is not hypothetical: the AMG's
    ``has_native_solve`` read ``self.factors.has_native_solve``, which only its own V-cycle has, so the
    property raised on the field split and a ``getattr`` default at the call site turned the exception
    into a plausible ``False``.

    Read off the source rather than exercised, because the failure is a *lookup that is never taken* on
    the paths a test would naturally drive -- which is precisely why the original went unnoticed.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(HostPreconditioner))
    reached = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "factors"
    }
    assert reached <= {"n_dofs", "apply"}, (
        f"the base reaches for {sorted(reached - {'n_dofs', 'apply'})} on its factors, which is not "
        "part of HostFactors -- either add it to the contract or answer it on the subclass"
    )

    # And the pair really is sufficient: a stub offering exactly it builds and applies.
    assert np.allclose(np.asarray(HostPreconditioner(_Doubling()).matvec()(jnp.ones(4))), 2.0)
