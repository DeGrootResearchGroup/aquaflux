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


@pytest.mark.slow
def test_the_split_continuation_converges_to_the_monolithic_fixed_point():
    """`field_split=True` is a drop-in: same solver, same root, only the frozen inverse differs.

    The point of routing it through `coupled_amg_continuation` rather than a parallel builder is that the
    shift policy, forward solver, step tail and refresh hooks stay shared -- so this asserts the thing that
    would break if they had quietly diverged: both reach the same converged state.
    """
    from aquaflux.turbulence import coupled_amg_continuation, solve_coupled

    from tests.integration.test_coupled_ilut import _channel

    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    flow, k, omega = hybrid_initialize(momentum, turbulence)
    reference = coupled.pack_state(flow, k, omega)

    split = coupled_amg_continuation(coupled, reference, field_split=True)
    flow_s, k_s, omega_s = solve_coupled(coupled, flow, k, omega, continuation=split, max_steps=40)
    assert float(jnp.linalg.norm(coupled.residual(coupled.pack_state(flow_s, k_s, omega_s)))) < 1e-8

    mono = coupled_amg_continuation(coupled, reference)
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

    groups, n_fields = case["groups"], case["n_fields"]
    coupled, state = case["coupled"], case["state"]
    colouring = _coupled_jacobian_colouring(coupled, 3)

    def matvec(v):
        return _jacobian_matvec(coupled, state, v)

    shift = np.full(groups.n_dofs, 0.5)
    pc = FieldSplitAmgPreconditioner.build(
        matvec, colouring, n_fields, shift, groups, coarse_eq_limit=200
    )
    split_before = pc.factors
    rng = np.random.default_rng(4)
    b = rng.standard_normal(groups.n_dofs)
    before = pc.factors.apply(b).copy()

    phases = pc.refresh_in_place(matvec, colouring, n_fields, shift * 4.0)

    assert pc.factors is split_before, "the refresh replaced the object instead of mutating it"
    assert [name for name, _ in phases] == ["probe", "assemble", "refactor"]
    assert not np.allclose(before, pc.factors.apply(b)), (
        "a 4x shift change left the inverse unchanged"
    )
    pc.destroy()


def test_the_split_shift_refresh_reuses_the_cached_jacobian(case):
    """The cheap branch: re-fit at a new shift without re-running the coloured probe."""
    from aquaflux.solve import FieldSplitAmgPreconditioner

    groups, n_fields = case["groups"], case["n_fields"]
    coupled, state = case["coupled"], case["state"]
    colouring = _coupled_jacobian_colouring(coupled, 3)

    def matvec(v):
        return _jacobian_matvec(coupled, state, v)

    pc = FieldSplitAmgPreconditioner.build(
        matvec, colouring, n_fields, np.full(groups.n_dofs, 0.5), groups, coarse_eq_limit=200
    )
    phases = pc.refresh_shift_in_place(np.full(groups.n_dofs, 2.0))
    # No probe phase at all -- that absence IS the saving this branch exists for.
    assert [name for name, _ in phases] == ["assemble", "refactor"]
    pc.destroy()


def test_per_block_smoother_options_reach_the_trailing_hierarchy(case):
    """The two halves must be tunable APART, and the setting must survive a refresh.

    The saddle needs its incomplete-LU sweep; the transported scalars are a much easier operator and are
    served by cheaper relaxations, so a split that could not smooth them differently would be giving up
    most of what splitting is for. The refresh half matters just as much: the march re-fits the
    preconditioner repeatedly, and a per-block option honoured only at build would silently revert to
    the default part-way through a run -- a bug that shows up as an unexplained slowdown, not a failure.
    """
    groups, shifted = case["groups"], case["shifted"]
    # A single Jacobi sweep against the shipped four sweeps of incomplete LU: different enough that a
    # hierarchy built with it cannot coincidentally match one built without.
    cheap = {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_max_it": 1,
    }
    shipped = build_block_triangular_field_split(shifted, groups)
    tuned = build_block_triangular_field_split(shifted, groups, trailing_options=cheap)
    rng = np.random.default_rng(11)
    b = rng.standard_normal(groups.n_dofs)
    baseline, altered = shipped.apply(b), tuned.apply(b)
    assert np.all(np.isfinite(altered))
    # The LEADING half is untouched, so its part of the answer must be identical; only the trailing
    # half may move. Checking both halves is what distinguishes "the option was applied to the right
    # block" from "the option was applied somewhere".
    assert np.allclose(baseline[groups.leading], altered[groups.leading], rtol=0, atol=0)
    assert not np.allclose(baseline[groups.trailing], altered[groups.trailing])

    # ... and it must still be the cheap smoother after a refresh re-fits the same objects in place.
    refitted = tuned.apply(b)
    tuned.refactor(shifted)
    assert np.allclose(refitted, tuned.apply(b), rtol=1e-10, atol=0)
    shipped.destroy()
    tuned.destroy()


def test_per_block_options_are_rejected_without_a_field_split(case):
    """Passing them to the monolithic path would silently do nothing, so it raises instead.

    There is one hierarchy without a split, so there is no "trailing half" to tune. Failing loudly is
    the difference between a typo that costs a run and a typo that costs a run AND is reported as a
    measurement of the smoother it never applied. It raises before the coloured probe, so the cost of
    finding out is nothing.
    """
    from aquaflux.turbulence import coupled_amg_continuation

    with pytest.raises(ValueError, match="field_split=True"):
        coupled_amg_continuation(
            case["coupled"], case["state"], trailing_options={"mg_levels_ksp_max_it": 1}
        )


def test_the_trailing_block_defaults_to_fewer_sweeps_and_the_count_is_tunable(case):
    """The two halves default to DIFFERENT smoothing, and the trailing count is a real parameter.

    The saddle needs its four incomplete-LU sweeps -- Jacobi-class smoothers do not converge on it at
    all -- while the transported-scalar pair does not, and on a three-dimensional march three of those
    four sweeps were pure cost (1959 s -> 1636 s on a step-for-step identical trajectory). That makes
    the asymmetry a default worth pinning: a refactor that quietly re-unified the two counts would give
    back the saving with nothing failing.

    The count is asserted through BEHAVIOUR rather than by reading the options dict back, because what
    matters is that the number reaches the hierarchy PETSc actually builds.
    """
    groups, shifted = case["groups"], case["shifted"]
    rng = np.random.default_rng(17)
    b = rng.standard_normal(groups.n_dofs)

    default = build_block_triangular_field_split(shifted, groups)
    matched = build_block_triangular_field_split(shifted, groups, trailing_smoother_sweeps=4)
    # Same sweeps on both halves is a different preconditioner from the shipped asymmetric default...
    assert not np.allclose(default.apply(b)[groups.trailing], matched.apply(b)[groups.trailing])
    # ...and the LEADING half is untouched by the trailing count, which is what makes it a per-block
    # knob rather than a global one.
    assert np.allclose(
        default.apply(b)[groups.leading], matched.apply(b)[groups.leading], rtol=0, atol=0
    )
    # Asking for the default explicitly must reproduce it exactly.
    explicit = build_block_triangular_field_split(shifted, groups, trailing_smoother_sweeps=1)
    assert np.allclose(default.apply(b), explicit.apply(b), rtol=0, atol=0)
    for pc in (default, matched, explicit):
        pc.destroy()
