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
wrapper without one of its own. The split's blocks are fitted by the traced inverses the flagship cases
ship; the monolithic V-cycle it is compared against needs PETSc, so the module is skipped where
``petsc4py`` is unavailable.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("petsc4py")

from aquaflux.solve import (
    FieldGroups,
    JacobiSmoothed,
    MonolithicAmgPreconditioner,
    SimpleSmoothed,
    build_amg_vcycle,
    build_block_triangular_field_split,
    relative_residual_gmres,
    restart_cycles,
    solve_linear,
)
from aquaflux.turbulence import (
    CoupledRANS,
    FieldSplit,
    MaterializedJacobian,
    MonolithicVCycle,
    coupled_step,
    hybrid_initialize,
)
from aquaflux.turbulence.coupled import (
    _coupled_jacobian_plan,
    _jacobian_matvec,
)

from tests.integration.test_coupled_lu import _channel


@pytest.fixture(scope="module")
def case():
    """A small turbulent channel, its cold state, and the assembled coupled Jacobian there."""
    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    flow, k, omega = hybrid_initialize(momentum, turbulence)
    state = coupled.pack_state(flow, k, omega)
    n_fields = coupled.layout.n_fields
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        _coupled_jacobian_plan(coupled, 3),
    )
    groups = FieldGroups.split_before(coupled.layout, "k")
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


def _split(shifted, groups):
    """The split with the traced inverses both flagship cases ship."""
    return build_block_triangular_field_split(
        shifted,
        groups,
        leading_inverse=SimpleSmoothed(),
        trailing_inverse=JacobiSmoothed(),
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


def test_the_split_preconditions_the_real_coupled_saddle(case):
    """The split converges the assembled coupled system through GMRES, on the true residual.

    A convergence check, not a ranking: an operator every candidate solves in a cycle or two
    discriminates between none of them. Comparing preconditioners needs a state where the operator is
    hard, and belongs to the case study rather than to a fast test.
    """
    groups, shifted = case["groups"], case["shifted"]
    split = _split(shifted, groups)
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
    split = _split(shifted, groups)
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
    split = _split(shifted, groups)
    monolithic = build_amg_vcycle(shifted, n_fields, coarse_eq_limit=200)
    rng = np.random.default_rng(3)
    b = jnp.asarray(rng.standard_normal(groups.n_dofs))
    for inverse in (split, monolithic):
        applied = MonolithicAmgPreconditioner(inverse).matvec()(b)
        assert applied.shape == b.shape
        assert bool(jnp.all(jnp.isfinite(applied)))
    split.destroy()
    monolithic.destroy()


@pytest.mark.slow
def test_the_split_continuation_converges_to_the_monolithic_fixed_point():
    """`field_split=True` is a drop-in: same solver, same root, only the frozen inverse differs.

    The point of routing it through `coupled_step` rather than a parallel builder is that the
    shift policy, forward solver, step tail and refresh hooks stay shared -- so this asserts the thing that
    would break if they had quietly diverged: both reach the same converged state.
    """
    from aquaflux.turbulence import solve_coupled

    from tests.integration.test_coupled_amg import SMOOTHER_FILL
    from tests.integration.test_coupled_lu import _channel

    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    flow, k, omega = hybrid_initialize(momentum, turbulence)
    reference = coupled.pack_state(flow, k, omega)

    # The monolithic arm takes the fixture's extra level of smoother fill, for the reason recorded at
    # `SMOOTHER_FILL`: at this initial condition the operator's degenerate couplings are exactly zero,
    # so the pruned ILU(1) pattern loses the fill the V-cycle depends on. The split's blocks are fitted
    # by their own injected inverses, which read no smoother fill.
    split = coupled_step(
        coupled,
        reference,
        preconditioner=MaterializedJacobian(FieldSplit(SimpleSmoothed(), JacobiSmoothed())),
    )
    flow_s, k_s, omega_s = solve_coupled(coupled, flow, k, omega, continuation=split, max_steps=40)
    assert float(jnp.linalg.norm(coupled.residual(coupled.pack_state(flow_s, k_s, omega_s)))) < 1e-8

    mono = coupled_step(
        coupled,
        reference,
        preconditioner=MaterializedJacobian(MonolithicVCycle(smoother_fill_levels=SMOOTHER_FILL)),
    )
    flow_m, k_m, omega_m = solve_coupled(coupled, flow, k, omega, continuation=mono, max_steps=40)
    assert float(jnp.linalg.norm(flow_s - flow_m) / jnp.linalg.norm(flow_m)) < 1e-4
    assert float(jnp.linalg.norm(k_s - k_m) / jnp.linalg.norm(k_m)) < 1e-3
    assert float(jnp.linalg.norm(omega_s - omega_m) / jnp.linalg.norm(omega_m)) < 1e-4


def test_the_split_refreshes_in_place_onto_the_same_object(case):
    """A mid-march refresh must MUTATE the preconditioner, not replace it.

    The march holds it as a static field and the callback reads `factors` at call time, so a refresh that
    returned a new object would silently keep preconditioning with the stale one -- and would still
    converge, just slower, which is exactly the kind of bug a march hides.
    """
    from aquaflux.solve import FieldSplitAmgPreconditioner

    groups = case["groups"]
    coupled, state = case["coupled"], case["state"]
    plan = _coupled_jacobian_plan(coupled, 3)

    def matvec(v):
        return _jacobian_matvec(coupled, state, v)

    shift = np.full(groups.n_dofs, 0.5)
    pc = FieldSplitAmgPreconditioner.build(
        matvec,
        plan,
        shift,
        groups,
        leading_inverse=SimpleSmoothed(),
        trailing_inverse=JacobiSmoothed(),
    )
    split_before = pc.factors
    rng = np.random.default_rng(4)
    b = rng.standard_normal(groups.n_dofs)
    before = pc.factors.apply(b).copy()

    phases = pc.refresh_in_place(matvec, plan, shift * 4.0)

    assert pc.factors is split_before, "the refresh replaced the object instead of mutating it"
    assert [name for name, _ in phases] == ["probe", "assemble", "refactor"]
    assert not np.allclose(before, pc.factors.apply(b)), (
        "a 4x shift change left the inverse unchanged"
    )
    pc.destroy()


def test_the_jacobi_smoothed_inverse_is_a_fixed_linear_map_that_transposes(case):
    """The two properties the coupled solve requires of any block inverse, on the real trailing block.

    Both are structural preconditions rather than quality measures, and both fail silently: a
    preconditioner that is not a fixed *linear* map makes the non-flexible outer GMRES invalid, and one
    whose transpose is not exact corrupts every gradient taken through a converged solve while the
    forward march looks perfectly healthy. This runs on the real ``[k, omega]`` block because that is
    what it will precondition, and its aggregation and smoother settings are chosen to reproduce a host
    GAMG V-cycle rather than to be conservative.
    """
    import numpy as np
    import scipy.sparse as sp
    from aquaflux.solve import JacobiSmoothedInverse

    groups, shifted = case["groups"], case["shifted"]
    block = sp.csr_matrix(shifted[groups.trailing, :][:, groups.trailing])
    inverse = JacobiSmoothedInverse(block, groups.n_trailing_fields)
    try:
        rng = np.random.default_rng(0)
        u, v = (rng.standard_normal(inverse.n_dofs) for _ in range(2))

        combined = inverse.apply(2.0 * u - 3.0 * v)
        separately = 2.0 * inverse.apply(u) - 3.0 * inverse.apply(v)
        assert np.allclose(combined, separately, rtol=1e-10, atol=1e-12)

        assert np.isclose(
            v @ inverse.apply(u), inverse.apply(v, transpose=True) @ u, rtol=1e-10, atol=1e-12
        )
    finally:
        inverse.destroy()
