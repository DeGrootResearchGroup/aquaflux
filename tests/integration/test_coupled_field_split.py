"""Integration: the block-triangular field split over the REAL coupled Reynolds-averaged Jacobian.

The unit tests pin the split's algebra on synthetic blocks. What they cannot check is the thing most
easily got silently wrong -- whether the partition this preconditioner slices actually corresponds to the
flow and turbulence blocks of the coupled state. A partition off by one field would still produce a
working, contracting preconditioner; it would simply be preconditioning a mislabelled operator, and every
downstream measurement taken with it would be describing something other than a field split.

So the first test here compares the partition against the coupled layout's own ``unpack``, and the rest
drive the split against the assembled coupled Jacobian on a small turbulent channel: preconditioned GMRES
converges it on the true residual, its transpose satisfies the adjoint identity that the
implicitly-differentiated gradient depends on, and it reaches a solve through the existing callback
wrapper without one of its own. The V-cycles need PETSc, so the module is skipped where ``petsc4py`` is
unavailable.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("petsc4py")

from aquaflux.solve import (
    FieldGroups,
    MonolithicAmgPreconditioner,
    build_amg_vcycle,
    build_block_triangular_field_split,
    relative_residual_gmres,
    restart_cycles,
    solve_linear,
)
from aquaflux.turbulence import CoupledRANS, hybrid_initialize
from aquaflux.turbulence.coupled import (
    _coupled_jacobian_colouring,
    _jacobian_matvec,
)

from tests.integration.test_coupled_ilut import _channel


@pytest.fixture(scope="module")
def case():
    """A small turbulent channel, its cold state, and the assembled coupled Jacobian there."""
    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    flow, k, omega = hybrid_initialize(momentum, turbulence)
    state = coupled.pack_state(flow, k, omega)
    n_fields = coupled.layout.dim + 3
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        _coupled_jacobian_colouring(coupled, 3),
        n_fields,
    )
    groups = FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,
        n_trailing_fields=2,
    )
    # A shift keeps the cold operator away from the singular limit, as the march's own step does.
    shifted = MonolithicAmgPreconditioner._shifted(jacobian, np.full(groups.n_dofs, 0.5))
    return {
        "coupled": coupled,
        "state": state,
        "groups": groups,
        "shifted": shifted,
        "n_fields": n_fields,
    }


def test_the_partition_matches_the_coupled_layout(case):
    """The leading group must be exactly the flow sub-vector the coupled layout unpacks.

    This is the assumption the whole preconditioner rests on -- that a field-major coupled state puts
    ``[u, v, (w,) p]`` in one contiguous range and ``[k, omega]`` in the next. If it ever stopped holding,
    every arm of every field-split study would be measuring a mislabelled partition rather than failing.
    """
    coupled, state, groups = case["coupled"], case["state"], case["groups"]
    flow, k, omega = coupled.layout.unpack(state)
    np.testing.assert_array_equal(np.asarray(state[groups.leading]), np.asarray(flow))
    np.testing.assert_array_equal(
        np.asarray(state[groups.trailing]), np.asarray(jnp.concatenate([k, omega]))
    )


def _gmres_matvecs(shifted, preconditioner, b, *, rtol=1e-8):
    """Restart cycles and the TRUE relative residual of a preconditioned GMRES on the real operator.

    Judged this way rather than by a one-application contraction or a stationary Richardson sweep. Both of
    those are cheaper to write and both are invalid on an indefinite saddle: a contraction ratio is not a
    convergence criterion for a Krylov-accelerated preconditioner (a preconditioner with a one-apply ratio
    well above one can still converge in tens of matrix-vector products), and Richardson on an indefinite
    operator diverges on its own account, so it would report the iteration's failure as the
    preconditioner's.
    """
    operator = jnp.asarray(shifted.toarray()) if shifted.shape[0] <= 2000 else None
    assert operator is not None, "this helper densifies; keep the integration mesh small"
    solution, raw = solve_linear(
        lambda v: operator @ v,
        jnp.asarray(b),
        relative_residual_gmres(rtol, restart=30, stagnation_iters=40, max_restarts=40),
        preconditioner=MonolithicAmgPreconditioner(preconditioner).matvec(),
        throw=False,
    )
    true = float(jnp.linalg.norm(operator @ solution - jnp.asarray(b)) / jnp.linalg.norm(b))
    return restart_cycles(int(raw)), true


@pytest.mark.parametrize("flow_first", [True, False])
def test_the_split_preconditions_the_real_coupled_saddle(case, flow_first):
    """Both orderings converge the assembled coupled system through GMRES, on the true residual.

    Worth recording why this is *not* asserted as a difference between the two orderings. One application
    of the turbulence-first split leaves a residual some three times the input where flow-first leaves a
    third of it, which reads as one ordering being far weaker -- and through GMRES on this operator the two
    are indistinguishable, both reaching machine precision inside a single restart cycle. A one-application
    contraction is not a convergence criterion for a Krylov-accelerated preconditioner, and this is that
    trap in miniature.

    Which also means this state cannot rank the orderings at all: an operator every candidate solves in one
    cycle discriminates between none of them. That comparison needs a state where the operator is hard, and
    belongs to the case study rather than to a fast test.
    """
    groups, shifted = case["groups"], case["shifted"]
    split = build_block_triangular_field_split(
        shifted, groups, flow_first=flow_first, coarse_eq_limit=200
    )
    rng = np.random.default_rng(1)
    b = rng.standard_normal(groups.n_dofs)
    cycles, true = _gmres_matvecs(shifted, split, b)
    assert true < 1e-7, f"left a true relative residual of {true:.3e} after {cycles} cycles"
    split.destroy()


def test_the_transpose_serves_the_adjoint_on_the_real_operator(case):
    """``<y, M x> == <M^T y, x>`` with real V-cycles over the coupled Jacobian.

    The implicitly-differentiated adjoint solves the transposed system with ``M^T``, so this identity is
    what makes the split legal on a differentiated solve at all.
    """
    groups, shifted = case["groups"], case["shifted"]
    split = build_block_triangular_field_split(shifted, groups, coarse_eq_limit=200)
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((2, groups.n_dofs))
    np.testing.assert_allclose(y @ split.apply(x), split.apply(y, transpose=True) @ x, rtol=1e-10)
    split.destroy()


def test_it_drops_into_the_jax_callback_wrapper_unchanged(case):
    """The split satisfies the same frozen-inverse interface the monolithic V-cycle does.

    The JAX-side wrapper reads only ``n_dofs`` and ``apply(residual, transpose=...)``, so a field split
    needs no wrapper of its own -- which is what lets it reach a solve through the existing callback path.
    """
    groups, shifted, n_fields = case["groups"], case["shifted"], case["n_fields"]
    split = build_block_triangular_field_split(shifted, groups, coarse_eq_limit=200)
    monolithic = build_amg_vcycle(shifted, n_fields, coarse_eq_limit=200)
    rng = np.random.default_rng(3)
    b = jnp.asarray(rng.standard_normal(groups.n_dofs))
    for inverse in (split, monolithic):
        applied = MonolithicAmgPreconditioner(inverse).matvec()(b)
        assert applied.shape == b.shape
        assert bool(jnp.all(jnp.isfinite(applied)))
    split.destroy()
    monolithic.destroy()
