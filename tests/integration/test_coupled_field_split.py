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
ship.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    ConvectionTwoLevel,
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import (
    DualTimeLoop,
    FieldGroups,
    FieldSplit,
    HostPreconditioner,
    JacobiSmoothed,
    MaterializedJacobian,
    MaterializedJacobianPreconditioner,
    SimpleSmoothed,
    field_split_inverse,
    jacobian_matvec,
    relative_residual_gmres,
    restart_cycles,
    solve_linear,
)
from aquaflux.turbulence import (
    BlockDiagonal,
    CoupledRANS,
    ScalarTwoLevel,
    SSTModel,
    SSTTurbulence,
    UnpreconditionedScalars,
    coupled_step,
    inlet_k,
    inlet_omega,
    open_session,
    solve_coupled,
    sst_initial_fields,
)
from aquaflux.turbulence.coupled import _coupled_jacobian_plan

RHO, U_IN, H, L = 1.0, 1.0, 1.0, 4.0
NU = 4e-4  # Re = U H / nu = 2500
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * H
PRECONDITIONER = {"velocity": ConvectionTwoLevel()}


def _channel(nx=20, ny=14, growth=1.2):
    y_nodes = graded_nodes(ny, H, growth)
    mesh = structured_grid_2d(nx, ny, lx=L, ly=H, named_boundaries=True, y_nodes=y_nodes)
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, model))
    properties = PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)})
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        properties,
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(U_IN, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        gradient_scheme=CompactGreenGauss(),
        advection_scheme=FirstOrderUpwind(),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        gradient_scheme=CompactGreenGauss(),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(k_in),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(omega_in),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return momentum, turbulence


@pytest.fixture(scope="module")
def case():
    """A small turbulent channel, its cold state, and the assembled coupled Jacobian there."""
    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    flow, k, omega = sst_initial_fields(momentum, turbulence)
    state = coupled.pack_state(flow, k, omega)
    n_fields = coupled.layout.n_fields
    jacobian = MaterializedJacobianPreconditioner._materialize_jacobian(
        lambda v: jacobian_matvec(coupled, state, v),
        _coupled_jacobian_plan(coupled, 3),
    )
    groups = FieldGroups.split_before(coupled.layout, "k")
    # A shift keeps the cold operator away from the singular limit, as the march's own step does.
    shifted = MaterializedJacobianPreconditioner._shifted(jacobian, np.full(groups.n_dofs, 0.5))
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
    return field_split_inverse(
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
        preconditioner=HostPreconditioner(preconditioner).matvec(),
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
    """The split satisfies the frozen-inverse interface the JAX callback wrapper reads.

    The wrapper reads only ``n_dofs`` and ``apply(residual, transpose=...)``, so a field split needs no
    wrapper of its own -- which is what lets it reach a solve through the existing callback path. The
    wrapped apply must be the split's own, forward and transposed.
    """
    groups, shifted = case["groups"], case["shifted"]
    split = _split(shifted, groups)
    rng = np.random.default_rng(3)
    b = rng.standard_normal(groups.n_dofs)
    wrapper = HostPreconditioner(split)
    for transpose in (False, True):
        applied = wrapper.matvec(transpose=transpose)(jnp.asarray(b))
        np.testing.assert_allclose(
            np.asarray(applied), split.apply(b, transpose=transpose), rtol=1e-12, atol=1e-14
        )
    split.destroy()


def _shipped_split() -> MaterializedJacobian:
    """The coupled preconditioner both flagship cases ship: a field split of traced inverses."""
    return MaterializedJacobian(FieldSplit(SimpleSmoothed(), JacobiSmoothed()))


def _with_scaled_viscosity(coupled, nu_scale):
    """``coupled`` with its molecular viscosity scaled, the parameter the adjoint tests differentiate."""
    return eqx.tree_at(
        lambda c: c.turbulence.molecular_viscosity,
        coupled,
        coupled.turbulence.molecular_viscosity * nu_scale,
    )


@pytest.fixture(scope="module")
def channel():
    """The channel's coupled assembler and its cold start, for the marches below."""
    momentum, turbulence = _channel()
    coupled = CoupledRANS.build(momentum, turbulence)
    return {"coupled": coupled, "start": sst_initial_fields(momentum, turbulence)}


@pytest.fixture(scope="module")
def block_root(channel):
    """The root the block-triangular SIMPLE preconditioner reaches: an independent reference.

    The fixed point is a property of the residual, not of the preconditioner, so a split that has
    quietly diverged from the shared step tail would land somewhere else.
    """
    flow, k, omega = channel["start"]
    return solve_coupled(
        channel["coupled"],
        flow,
        k,
        omega,
        max_steps=40,
        preconditioner=BlockDiagonal(scalar=ScalarTwoLevel(), **PRECONDITIONER),
    )


def _assert_same_root(reached, reference) -> None:
    flow, k, omega = reached
    flow_r, k_r, omega_r = reference
    assert float(jnp.linalg.norm(flow - flow_r) / jnp.linalg.norm(flow_r)) < 1e-4
    assert float(jnp.linalg.norm(k - k_r) / jnp.linalg.norm(k_r)) < 1e-3
    assert float(jnp.linalg.norm(omega - omega_r) / jnp.linalg.norm(omega_r)) < 1e-4


@pytest.mark.slow
def test_a_split_builds_the_right_step_types(channel) -> None:
    """A dual-time loop builds a split-preconditioned dual-time step; none, a single step."""
    from aquaflux.solve import DualTimeStep, PseudoTransientStep

    coupled = channel["coupled"]
    reference = coupled.pack_state(*channel["start"])
    single = coupled_step(coupled, reference, preconditioner=_shipped_split())
    assert isinstance(single, PseudoTransientStep)
    dual = coupled_step(
        coupled,
        reference,
        preconditioner=_shipped_split(),
        dual_time=DualTimeLoop(inner_steps=5, inner_tol=1e-3),
    )
    assert isinstance(dual, DualTimeStep)
    assert dual.inner_steps == 5


@pytest.mark.slow
def test_the_split_continuation_reaches_the_block_preconditioned_root(channel, block_root) -> None:
    """A field split is a drop-in: same solver, same root, only the frozen inverse differs.

    The point of routing it through `coupled_step` rather than a parallel builder is that the
    shift policy, forward solver, step tail and refresh hooks stay shared -- so this asserts the thing that
    would break if they had quietly diverged: the split reaches the root the block-triangular SIMPLE
    preconditioner reaches, and a genuinely turbulent one.
    """
    coupled = channel["coupled"]
    flow, k, omega = channel["start"]
    split = coupled_step(
        coupled, coupled.pack_state(flow, k, omega), preconditioner=_shipped_split()
    )
    reached = solve_coupled(coupled, flow, k, omega, strategy=split, max_steps=40)
    _, k_s, omega_s = reached
    assert float(jnp.linalg.norm(coupled.residual(coupled.pack_state(*reached)))) < 1e-8
    assert float(jnp.min(k_s)) >= 0.0
    assert float(jnp.min(omega_s)) > 0.0
    assert float(jnp.max(k_s)) > 10.0 * float(jnp.min(jnp.abs(k_s)) + 1e-30)  # genuinely turbulent
    _assert_same_root(reached, block_root)


@pytest.mark.slow
def test_the_split_adjoint_matches_finite_difference(channel) -> None:
    """The coupled implicit-function-theorem adjoint is exact through the split-preconditioned solve.

    The split only accelerates the Krylov solves and is ``stop_gradient``-ed, so the gradient is the one
    transpose solve on the unfrozen coupled residual -- preconditioned by the split's transpose -- and
    must match a finite difference. Built once outside ``jax.grad`` on concrete parameters.
    """
    coupled = channel["coupled"]
    flow, k, omega = channel["start"]
    step = coupled_step(
        coupled, coupled.pack_state(flow, k, omega), preconditioner=_shipped_split()
    )

    def objective(nu_scale):
        scaled = _with_scaled_viscosity(coupled, nu_scale)
        _, k_out, _ = solve_coupled(scaled, flow, k, omega, strategy=step, max_steps=40)
        return jnp.sum(k_out**2)

    analytic = float(jax.grad(objective)(1.0))
    eps = 1e-4
    finite_difference = float((objective(1.0 + eps) - objective(1.0 - eps)) / (2 * eps))
    assert analytic != 0.0
    assert abs(analytic - finite_difference) / abs(finite_difference) < 1e-5


@pytest.mark.slow
def test_a_split_session_re_fits_at_the_current_beta(channel) -> None:
    """A session's refresh hook re-fits the split at the step's CURRENT shift, not the one it was built at.

    The control sets the step's shift strength; the hook must read it. A split is not exact, so this
    compares the refreshed inverse with one fitted from scratch to the operator at that shift: they must
    apply identically, and differ from the inverse fitted at the build shift.
    """
    from aquaflux.solve import DualTimeControl
    from aquaflux.turbulence.coupled import _coupled_shift_policy

    coupled = channel["coupled"]
    state = coupled.pack_state(*channel["start"])
    session = open_session(
        MaterializedJacobian(FieldSplit(SimpleSmoothed(), JacobiSmoothed()), build_beta=0.05),
        coupled,
    )
    dual = session.build(state, dual_time=DualTimeLoop(inner_steps=5))
    preconditioner = dual.shift_policy.preconditioner
    rng = np.random.default_rng(5)
    b = rng.standard_normal(preconditioner.inverse.n_dofs)
    at_build = preconditioner.inverse.apply(b).copy()

    # the control sets a ConstantRelaxation(beta) on the step, at a beta DIFFERENT from the build beta
    active, _ = DualTimeControl(beta_start=0.7).next_step(dual, None, None)
    session.refresh_preconditioner(active, state)
    refreshed = active.shift_policy.preconditioner
    assert refreshed is preconditioner, (
        "the refresh replaced the preconditioner instead of re-fitting it"
    )

    # the same probe and the same shift diagonal, fitted from scratch at beta = 0.7
    jacobian = MaterializedJacobianPreconditioner._materialize_jacobian(
        lambda v: jacobian_matvec(coupled, state, v), _coupled_jacobian_plan(coupled, 3)
    )
    d = np.asarray(
        _coupled_shift_policy(coupled, state, UnpreconditionedScalars()).shift_term(state).diagonal
    )
    fresh = _split(
        MaterializedJacobianPreconditioner._shifted(jacobian, 0.7 * d),
        FieldGroups.split_before(coupled.layout, "k"),
    )
    try:
        np.testing.assert_allclose(
            refreshed.inverse.apply(b), fresh.apply(b), rtol=1e-10, atol=1e-12
        )
        assert not np.allclose(refreshed.inverse.apply(b), at_build)
    finally:
        fresh.destroy()


@pytest.mark.slow
def test_a_beta_tracking_split_march_reaches_the_block_preconditioned_root(
    channel, block_root
) -> None:
    """``solve_coupled`` with a split and a ``DualTimeControl`` reaches the block preconditioner's root.

    The spec opens a session whose per-step hook re-fits the split at each step's own shift.
    """
    from aquaflux.solve import DualTimeControl

    reached = solve_coupled(
        channel["coupled"],
        *channel["start"],
        preconditioner=_shipped_split(),
        dual_time=DualTimeLoop(inner_steps=5, inner_tol=1e-3),
        step_control=DualTimeControl(beta_start=0.5, beta_min=0.02),
        max_steps=60,
    )
    _assert_same_root(reached, block_root)


@pytest.mark.slow
def test_a_solve_that_re_fits_its_split_every_step_is_differentiable(channel) -> None:
    """A materialized preconditioner re-fits before every step, and ``jax.grad`` still runs through it.

    It used to raise: the re-fit ran on the path being differentiated and would have captured the
    tracer. The march now runs on stopped copies and the adjoint is attached at the root it reaches, so
    no re-fit ever sees a tracer, and the gradient must match finite differences like the frozen step's
    does above. The re-fit reads the step's shift strength, so the march carries a ``DualTimeControl``.
    """
    from aquaflux.solve import DualTimeControl

    coupled = channel["coupled"]
    flow, k, omega = channel["start"]

    def objective(nu_scale):
        _, k_out, _ = solve_coupled(
            _with_scaled_viscosity(coupled, nu_scale),
            flow,
            k,
            omega,
            preconditioner=_shipped_split(),
            dual_time=DualTimeLoop(inner_steps=5, inner_tol=1e-3),
            step_control=DualTimeControl(beta_start=0.5, beta_min=0.02),
            max_steps=60,
        )
        return jnp.sum(k_out**2)

    analytic = float(jax.grad(objective)(1.0))
    eps = 1e-4
    finite_difference = float((objective(1.0 + eps) - objective(1.0 - eps)) / (2 * eps))
    assert analytic != 0.0
    assert abs(analytic - finite_difference) / abs(finite_difference) < 1e-5


def test_the_split_refreshes_in_place_onto_the_same_object(case):
    """A mid-march refresh must MUTATE the preconditioner, not replace it.

    The march holds it as a static field and the callback reads `factors` at call time, so a refresh that
    returned a new object would silently keep preconditioning with the stale one -- and would still
    converge, just slower, which is exactly the kind of bug a march hides.
    """
    from aquaflux.solve import FieldSplitPreconditioner

    groups = case["groups"]
    coupled, state = case["coupled"], case["state"]
    plan = _coupled_jacobian_plan(coupled, 3)

    def matvec(v):
        return jacobian_matvec(coupled, state, v)

    shift = np.full(groups.n_dofs, 0.5)
    pc = FieldSplitPreconditioner.build(
        matvec,
        plan,
        shift,
        groups,
        leading_inverse=SimpleSmoothed(),
        trailing_inverse=JacobiSmoothed(),
    )
    split_before = pc.inverse
    rng = np.random.default_rng(4)
    b = rng.standard_normal(groups.n_dofs)
    before = pc.inverse.apply(b).copy()

    phases = pc.refresh_in_place(matvec, plan, shift * 4.0)

    assert pc.inverse is split_before, "the refresh replaced the object instead of mutating it"
    assert [name for name, _ in phases] == ["probe", "assemble", "refactor"]
    assert not np.allclose(before, pc.inverse.apply(b)), (
        "a 4x shift change left the inverse unchanged"
    )
    pc.destroy()


def test_the_jacobi_smoothed_inverse_is_a_fixed_linear_map_that_transposes(case):
    """The two properties the coupled solve requires of any block inverse, on the real trailing block.

    Both are structural preconditions rather than quality measures, and both fail silently: a
    preconditioner that is not a fixed *linear* map makes the non-flexible outer GMRES invalid, and one
    whose transpose is not exact corrupts every gradient taken through a converged solve while the
    forward march looks perfectly healthy. This runs on the real ``[k, omega]`` block because that is
    what it will precondition, with the inverse's own default aggregation and smoother settings.
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
