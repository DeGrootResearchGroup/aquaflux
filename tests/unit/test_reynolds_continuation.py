"""Unit tests for Reynolds-number continuation (schedule + the viscosity rescale, no solve).

The end-to-end "reaches the same root" and "gradient matches a direct solve" gates live in
``tests/integration/test_reynolds_continuation.py`` -- these cover the pure pieces: the geometric
schedule (physics-free) and the molecular-viscosity rescale on a small built coupled system.
"""

from __future__ import annotations

import math

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    AdaptiveReynoldsSchedule,
    CoupledRANS,
    GeometricReynoldsSchedule,
    SSTModel,
    SSTTurbulence,
    solve_reynolds_continuation,
)

RHO, NU, U_IN = 1.0, 1e-3, 1.0


# --- the schedules (pure, no mesh) ------------------------------------------------------


def _walk(schedule, n_points: int) -> list[float]:
    """The ladder a schedule produces when every rung converges -- the whole-ramp view.

    A schedule is asked for one scale at a time so it can react to a failure, so a test that wants the
    ladder has to walk it. This is that walk, and nothing else in the library does it.
    """
    scale = schedule.anchor(n_points)
    ladder = [scale]
    while scale > 1.0:
        scale = schedule.next_scale(tuple(ladder), None)
        ladder.append(scale)
    return ladder


def test_schedule_zero_points_is_the_target_alone() -> None:
    assert _walk(GeometricReynoldsSchedule(), 0) == [1.0]


def test_schedule_default_decade_per_step() -> None:
    assert _walk(GeometricReynoldsSchedule(), 1) == [10.0, 1.0]
    assert _walk(GeometricReynoldsSchedule(), 2) == [100.0, 10.0, 1.0]
    assert _walk(GeometricReynoldsSchedule(), 3) == [1000.0, 100.0, 10.0, 1.0]


def test_schedule_descends_to_one_and_has_n_plus_one_points() -> None:
    ladder = _walk(GeometricReynoldsSchedule(), 4)
    assert len(ladder) == 5 == GeometricReynoldsSchedule().planned_total(4)
    assert ladder[-1] == 1.0  # dissolves at the target
    assert ladder == sorted(ladder, reverse=True)  # strictly descending anchor -> target


def test_schedule_ratio_is_configurable() -> None:
    assert _walk(GeometricReynoldsSchedule(ratio=4.0), 2) == [16.0, 4.0, 1.0]


def test_a_ratio_that_does_not_divide_the_anchor_still_lands_exactly_on_the_target() -> None:
    # Repeated division drifts off 1.0; without the snap the ramp would end on a spurious extra rung
    # at 1.0000000000000002 -- a companion microscopically different from the target.
    ladder = _walk(GeometricReynoldsSchedule(ratio=3.1623), 4)
    assert len(ladder) == 5
    assert ladder[-1] == 1.0


def test_the_geometric_schedule_gives_up_when_a_rung_fails() -> None:
    # Its defining property: the ladder is fixed in advance, so there is nothing gentler to fall back
    # to and the caller is told to re-run with a deeper anchor.
    assert GeometricReynoldsSchedule().next_scale((100.0,), 10.0) is None


# --- AdaptiveReynoldsSchedule -----------------------------------------------------------


def test_the_adaptive_schedule_matches_the_geometric_one_when_nothing_fails() -> None:
    assert _walk(AdaptiveReynoldsSchedule(), 2) == _walk(GeometricReynoldsSchedule(), 2)


def test_a_failed_rung_retreats_to_the_geometric_mean() -> None:
    # Halving the step in the log of the viscosity scale -- the parameterization the ramp is
    # geometric in, so a bisection there bisects the step.
    retreat = AdaptiveReynoldsSchedule().next_scale((100.0,), 10.0)
    assert retreat == pytest.approx(math.sqrt(100.0 * 10.0))
    assert 10.0 < retreat < 100.0  # strictly between the root in hand and the scale that failed


def test_repeated_failures_bisect_again_and_eventually_give_up() -> None:
    schedule = AdaptiveReynoldsSchedule()
    root, attempt, retreats = 100.0, 10.0, 0
    while (nxt := schedule.next_scale((root,), attempt)) is not None:
        # A retreat moves back TOWARD the root, and the ramp descends, so a gentler step is a LARGER
        # viscosity scale -- nearer the rung already converged, not nearer the target.
        assert attempt < nxt < root
        attempt, retreats = nxt, retreats + 1
        assert retreats < 50, "retreating did not terminate"
    assert retreats > 1  # it does retreat more than once before giving up
    assert root / attempt < AdaptiveReynoldsSchedule().min_ratio + 0.05


def test_the_step_recovers_gradually_after_a_retreat() -> None:
    # Without recovery the ramp would carry one hard rung's caution to the target; with it unbounded
    # the rung after a retreat would jump straight back to the step that had just failed.
    schedule = AdaptiveReynoldsSchedule()
    after_retreat = schedule.next_scale((100.0, math.sqrt(1000.0)), None)
    achieved = 100.0 / math.sqrt(1000.0)
    assert math.sqrt(1000.0) / after_retreat == pytest.approx(achieved * schedule.recovery)
    assert (
        achieved < achieved * schedule.recovery < schedule.ratio
    )  # between the two, not at either


def test_the_step_never_grows_past_the_base_ratio() -> None:
    schedule = AdaptiveReynoldsSchedule(ratio=10.0, recovery=100.0)
    assert 100.0 / schedule.next_scale((1000.0, 100.0), None) == pytest.approx(10.0)


def test_a_failed_anchor_gives_up_because_there_is_nothing_to_retreat_toward() -> None:
    # No converged root exists yet, and a gentler step is not the remedy -- the ramp has to start
    # lower, which is `n_points`.
    assert AdaptiveReynoldsSchedule().next_scale((), 100.0) is None


def test_negative_points_raise() -> None:
    coupled = _tiny_coupled()
    with pytest.raises(ValueError, match="n_points must be >= 0"):
        solve_reynolds_continuation(coupled, -1)


# --- the molecular-viscosity rescale (small built coupled, no solve) --------------------


def _tiny_coupled(nx: int = 4, ny: int = 3) -> CoupledRANS:
    mesh = structured_grid_2d(nx, ny, lx=2.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    model = SSTModel()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(U_IN, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(0.1),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(100.0),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return CoupledRANS.build(momentum, turbulence)


def _dynamic_viscosity(coupled: CoupledRANS) -> float:
    return coupled.momentum.properties.properties["viscosity"].value


def test_coupled_rescale_scales_both_viscosity_leaves() -> None:
    coupled = _tiny_coupled()
    scaled = coupled.with_scaled_molecular_viscosity(10.0)
    # Momentum dynamic viscosity mu = rho * nu: scaled by the factor.
    assert _dynamic_viscosity(scaled) == 10.0 * (RHO * NU)
    # Turbulence kinematic viscosity nu: the whole per-cell field scaled by the factor.
    np.testing.assert_allclose(
        np.asarray(scaled.turbulence.molecular_viscosity),
        np.asarray(coupled.turbulence.molecular_viscosity) * 10.0,
    )


def test_coupled_rescale_leaves_density_untouched() -> None:
    coupled = _tiny_coupled()
    scaled = coupled.with_scaled_molecular_viscosity(10.0)
    assert scaled.momentum.properties.properties["density"].value == RHO
    assert float(scaled.turbulence.density) == RHO


def test_coupled_rescale_is_immutable() -> None:
    coupled = _tiny_coupled()
    coupled.with_scaled_molecular_viscosity(10.0)
    assert _dynamic_viscosity(coupled) == RHO * NU  # original unchanged


def test_coupled_rescale_preserves_the_scalar_transforms() -> None:
    """The rescale carries the omega log-transform (and everything else) through unchanged."""
    from aquaflux.turbulence import LogScalars

    momentum = _tiny_coupled().momentum
    turbulence = _tiny_coupled().turbulence
    coupled = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    scaled = coupled.with_scaled_molecular_viscosity(5.0)
    assert isinstance(scaled.omega_transform, LogScalars)


# --- the ramp structure: seeds, viscosity scales, and the intermediate tolerance ---------
#
# These stub out solve_coupled (via the name the wrapper calls) to record each per-Re solve's inputs,
# so the loop's structure is verified without any actual Newton solve.


def _record_solves(monkeypatch):
    """Patch the wrapper's ``solve_coupled`` to record ``(scale, seed_is_none, rtol)`` per call."""
    import aquaflux.turbulence.reynolds as reynolds

    calls = []
    result = (jnp.zeros(3), jnp.ones(1), jnp.ones(1))  # a stand-in converged (flow, k, omega)

    def fake_solve_coupled(coupled, flow=None, k=None, omega=None, **kwargs):
        # The momentum dynamic viscosity encodes the scale (mu = factor * RHO * NU).
        scale = float(coupled.momentum.properties.properties["viscosity"].value / (RHO * NU))
        calls.append(
            {
                "scale": scale,
                "seed_is_none": flow is None,
                "rtol": kwargs.get("rtol"),
                "kwargs": kwargs,
            }
        )
        return result

    monkeypatch.setattr(reynolds, "solve_coupled", fake_solve_coupled)
    return calls


def test_ramp_visits_every_scale_and_threads_seeds(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10)
    # Three solves at the schedule's scales, descending to the target (1.0).
    assert [round(c["scale"], 6) for c in calls] == [100.0, 10.0, 1.0]
    # The first point self-starts (no seed); every later point is warm-started.
    assert [c["seed_is_none"] for c in calls] == [True, False, False]


def test_intermediate_points_use_the_loose_tolerance_and_target_uses_rtol(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10, intermediate_rtol=1e-2)
    # Lower-Re points converge loosely; the target keeps the caller's tight rtol.
    assert [c["rtol"] for c in calls] == [1e-2, 1e-2, 1e-10]


def test_intermediate_rtol_none_converges_every_point_to_rtol(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10, intermediate_rtol=None)
    assert [c["rtol"] for c in calls] == [1e-10, 1e-10, 1e-10]


def test_zero_points_calls_solve_once_at_the_target(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=0, rtol=1e-8)
    assert len(calls) == 1
    assert calls[0]["scale"] == 1.0 and calls[0]["seed_is_none"] and calls[0]["rtol"] == 1e-8


def test_the_ramp_and_the_target_get_opposite_halves_of_the_continuation_settings(
    monkeypatch,
) -> None:
    """A pre-built continuation goes to the target; the settings that build one go to the ramp.

    Passing both at once is the ordinary case rather than a mistake: the pre-built step is frozen at the
    *target* viscosity, so each lower-Re point has to build its own, from ``method`` and whatever else
    the caller passes for ``coupled_continuation``. The split therefore runs both ways, and only the
    ramp half existed — so the target solve was handed a continuation *and* the settings for one, which
    ``solve_coupled`` used to ignore in silence and now rejects outright.
    """
    calls = _record_solves(monkeypatch)
    step = object()  # `solve_coupled` is patched out, so its type does not matter here
    solve_reynolds_continuation(
        _tiny_coupled(),
        n_points=1,
        rtol=1e-10,
        continuation=step,
        method="twolevel",
        schur_scaling="msimple",
    )
    ramp, target = calls
    # The ramp builds its own at its own viscosity, so it takes the settings and not the frozen step.
    assert "continuation" not in ramp["kwargs"]
    assert ramp["kwargs"]["method"] == "twolevel"
    assert ramp["kwargs"]["schur_scaling"] == "msimple"
    # The target takes the frozen step and none of the settings, which describe a build it will not do.
    assert target["kwargs"]["continuation"] is step
    assert "method" not in target["kwargs"]
    assert "schur_scaling" not in target["kwargs"]
    # ...but the keywords that drive the *solve* rather than a build still reach it.
    assert target["kwargs"]["rtol"] == 1e-10


def test_without_a_continuation_the_target_keeps_every_setting(monkeypatch) -> None:
    """The strip is conditional: with nothing pre-built, the target builds its own and needs them all.

    Stripping unconditionally would silently drop the target's configuration, which is the same defect
    one layer down and the reason this is pinned rather than assumed.
    """
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(
        _tiny_coupled(), n_points=1, rtol=1e-10, method="twolevel", schur_scaling="msimple"
    )
    target = calls[-1]
    assert target["kwargs"]["method"] == "twolevel"
    assert target["kwargs"]["schur_scaling"] == "msimple"


def test_point_setup_builds_per_point_kwargs_and_materializes_the_first_seed(monkeypatch) -> None:
    """``point_setup`` is called for every Reynolds point with that point's companion; its returned
    kwargs are merged into that point's solve; and the lowest point's seed is materialized (so a
    per-point continuation can freeze at the same state the solve starts from).
    """
    import aquaflux.turbulence.reynolds as reynolds

    coupled = _tiny_coupled()
    n = coupled.momentum.mesh.n_cells
    dim = coupled.momentum.mesh.dim
    # A correctly-shaped stand-in converged state, so each point's seed packs into the next cleanly.
    fields = (jnp.zeros((dim + 1) * n), jnp.full(n, 0.5), jnp.full(n, 100.0))

    calls = []

    def fake_solve_coupled(c, flow=None, k=None, omega=None, **kwargs):
        scale = float(c.momentum.properties.properties["viscosity"].value / (RHO * NU))
        calls.append({"scale": scale, "seed_is_none": flow is None, "tag": kwargs.get("tag")})
        return fields

    monkeypatch.setattr(reynolds, "solve_coupled", fake_solve_coupled)
    # Stub the hybrid start so the test stays structural (no real Laplace solve).
    monkeypatch.setattr(reynolds, "hybrid_initialize", lambda momentum, turbulence: fields)

    setups = []

    def point_setup(companion, state, point):
        scale = float(companion.momentum.properties.properties["viscosity"].value / (RHO * NU))
        setups.append(scale)
        return {"tag": scale}  # a marker kwarg proving the merge reaches solve_coupled

    solve_reynolds_continuation(coupled, n_points=2, rtol=1e-10, point_setup=point_setup)

    # Called once per point (lower-Re and target), at each companion's viscosity scale...
    assert setups == [100.0, 10.0, 1.0]
    # ...its kwargs are merged into every point's solve...
    assert [c["tag"] for c in calls] == [100.0, 10.0, 1.0]
    # ...and the lowest point is now warm-started from the materialized seed too (not solve_coupled's
    # internal hybrid start), so the built continuation and the solve agree on the starting state.
    assert [c["seed_is_none"] for c in calls] == [False, False, False]


def test_point_setup_none_is_byte_identical_to_the_plain_ramp(monkeypatch) -> None:
    """Default (``point_setup=None``): the lowest point self-starts inside solve_coupled and no
    per-point kwargs are added -- the ramp is exactly the pre-existing one."""
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10)
    assert [c["seed_is_none"] for c in calls] == [True, False, False]  # first point self-starts


def test_point_setup_receives_the_points_position_in_the_ramp(monkeypatch) -> None:
    """``point_setup`` is told which point it is configuring, rather than having to count its own calls.

    The index, the total and the viscosity scaling are the continuation's own bookkeeping; a caller
    tracking its invocations would duplicate the loop counter, and could not know the total or the
    scaling at all.
    """
    import aquaflux.turbulence.reynolds as reynolds

    coupled = _tiny_coupled()
    n = coupled.momentum.mesh.n_cells
    dim = coupled.momentum.mesh.dim
    fields = (jnp.zeros((dim + 1) * n), jnp.full(n, 0.5), jnp.full(n, 100.0))

    monkeypatch.setattr(reynolds, "solve_coupled", lambda c, *a, **k: fields)
    monkeypatch.setattr(reynolds, "hybrid_initialize", lambda momentum, turbulence: fields)

    seen = []

    def point_setup(companion, state, point):
        seen.append(point)
        return {}

    solve_reynolds_continuation(coupled, n_points=2, rtol=1e-10, point_setup=point_setup)

    assert [p.index for p in seen] == [1, 2, 3]  # 1-based, anchor first
    assert [p.total for p in seen] == [3, 3, 3]  # n_points + 1, including the target
    assert [p.viscosity_scale for p in seen] == [100.0, 10.0, 1.0]
    assert [p.is_target for p in seen] == [False, False, True]  # only the true-viscosity point
    assert seen[1].label == "point 2/3 (Re/10)"


def test_seed_projection_replaces_the_state_the_point_actually_solves_from(monkeypatch) -> None:
    """``seed_projection`` corrects the seed itself, and ``point_setup`` then sees the corrected one.

    The order is the point of the test, not a detail: a per-point continuation built by ``point_setup``
    freezes at the state it is handed, so if the projection ran afterwards the preconditioner would be
    fitted to a state the solve never visits. It is also why the projection cannot be expressed through
    ``point_setup``, which returns keyword arguments while the seed travels positionally.
    """
    import aquaflux.turbulence.reynolds as reynolds

    coupled = _tiny_coupled()
    n = coupled.momentum.mesh.n_cells
    dim = coupled.momentum.mesh.dim
    fields = (jnp.zeros((dim + 1) * n), jnp.full(n, 0.5), jnp.full(n, 100.0))

    solved, setups = [], []

    def fake_solve_coupled(c, flow=None, k=None, omega=None, **kwargs):
        solved.append(None if flow is None else float(flow[0]))
        return fields

    monkeypatch.setattr(reynolds, "solve_coupled", fake_solve_coupled)
    monkeypatch.setattr(reynolds, "hybrid_initialize", lambda momentum, turbulence: fields)

    def seed_projection(companion, state, point):
        return state.at[0].set(float(point.index))  # a marker only a projection could put there

    def point_setup(companion, state, point):
        setups.append(float(state[0]))
        return {}

    solve_reynolds_continuation(
        coupled,
        n_points=2,
        rtol=1e-10,
        seed_projection=seed_projection,
        point_setup=point_setup,
    )

    # Every point solves from the projected seed, including the lowest -- whose seed is materialized
    # from the hybrid start for exactly this reason.
    assert solved == [1.0, 2.0, 3.0]
    # And `point_setup` is handed the projected state, not the one the projection was given.
    assert setups == [1.0, 2.0, 3.0]


def test_seed_projection_none_leaves_the_ramp_untouched(monkeypatch) -> None:
    """The default must not materialize the lowest point's seed, which is the one observable
    difference the hook's plumbing could otherwise leak into the ungated path."""
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10)
    assert [c["seed_is_none"] for c in calls] == [True, False, False]


def test_a_projection_that_declines_a_point_leaves_that_point_untouched(monkeypatch) -> None:
    """Returning the state unchanged is a real no-op, not a round-trip through the transforms.

    The usual shape of a projection is that it acts on the *handovers* and declines the anchor, which
    inherits nothing. Unpacking the state to physical fields and back inverts the scalar transforms, so
    a declined point would otherwise be re-seeded with a state differing in its last bits under a
    log-solved field -- and this project compares march trajectories bit-for-bit across arms, so a
    declined rung has to stay identical to the arm it is being compared against.
    """
    import aquaflux.turbulence.reynolds as reynolds

    coupled = _tiny_coupled()
    n = coupled.momentum.mesh.n_cells
    dim = coupled.momentum.mesh.dim
    fields = (jnp.zeros((dim + 1) * n), jnp.full(n, 0.5), jnp.full(n, 100.0))

    seeded = []

    def fake_solve_coupled(c, flow=None, k=None, omega=None, **kwargs):
        seeded.append((flow, k, omega))
        return fields

    monkeypatch.setattr(reynolds, "solve_coupled", fake_solve_coupled)
    monkeypatch.setattr(reynolds, "hybrid_initialize", lambda momentum, turbulence: fields)

    solve_reynolds_continuation(
        coupled,
        n_points=1,
        rtol=1e-10,
        seed_projection=lambda companion, state, point: state,  # declines every point
    )

    # The very objects the ramp already held, not equal-valued rebuilds of them.
    for flow, k, omega in seeded:
        assert flow is fields[0]
        assert k is fields[1]
        assert omega is fields[2]


# --- the retreat, end to end through the wrapper ----------------------------------------


def _fail_at(monkeypatch, doomed):
    """Patch ``solve_coupled`` to fail at the scales in ``doomed``, recording every attempt.

    ``doomed`` is consumed as a set of scales that raise the convergence guard the *first* time they
    are attempted, so a retreat that later re-attempts a gentler scale succeeds.
    """
    import aquaflux.turbulence.reynolds as reynolds

    attempts, remaining = [], set(doomed)
    result = (jnp.zeros(3), jnp.ones(1), jnp.ones(1))

    def fake_solve_coupled(coupled, flow=None, k=None, omega=None, **kwargs):
        scale = float(coupled.momentum.properties.properties["viscosity"].value / (RHO * NU))
        attempts.append(round(scale, 6))
        if round(scale, 6) in remaining:
            remaining.discard(round(scale, 6))
            raise eqx.EquinoxRuntimeError("stand-in for the convergence guard")
        return result

    monkeypatch.setattr(reynolds, "solve_coupled", fake_solve_coupled)
    return attempts


def test_a_failed_rung_retreats_and_the_ramp_continues(monkeypatch) -> None:
    """The point of the adaptive schedule: a rung that fails costs a gentler retry, not the run.

    The rungs already converged are kept -- the retry is seeded by the last converged root rather than
    restarting the ramp -- which is what makes this cheaper than the caller re-running at a larger
    ``n_points``.
    """
    attempts = _fail_at(monkeypatch, doomed={10.0})
    solve_reynolds_continuation(
        _tiny_coupled(), n_points=2, rtol=1e-10, schedule=AdaptiveReynoldsSchedule()
    )
    assert attempts[0] == 100.0  # the anchor
    assert attempts[1] == 10.0  # the decade step, which fails
    assert attempts[2] == pytest.approx(math.sqrt(1000.0), rel=1e-6)  # the retreat, gentler
    assert attempts[-1] == 1.0  # and the ramp still reaches the target
    assert len(attempts) > 4  # it did not simply skip the rungs it could not take


def test_the_same_failure_ENDS_the_run_under_the_fixed_ladder(monkeypatch) -> None:
    """The control for the test above: without an adaptive schedule this is still a hard failure.

    Keeping both pins that the retreat is the schedule's doing and not something the loop now does for
    every schedule -- the default must stay exactly as unforgiving as it was.
    """
    _fail_at(monkeypatch, doomed={10.0})
    with pytest.raises(RuntimeError, match="offered no gentler step"):
        solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10)


def test_a_failed_TARGET_rung_also_retreats(monkeypatch) -> None:
    """The target is the hardest point on the ramp, so it is the one most worth inserting a rung before.

    A loop that special-cased the final solve could not do this; the rung and the target take the same
    path precisely so that they share the retreat.
    """
    attempts = _fail_at(monkeypatch, doomed={1.0})
    solve_reynolds_continuation(
        _tiny_coupled(), n_points=1, rtol=1e-10, schedule=AdaptiveReynoldsSchedule()
    )
    assert attempts[:2] == [10.0, 1.0]  # anchor, then the target, which fails
    assert 1.0 < attempts[2] < 10.0  # a rung inserted between the last root and the target
    assert attempts[-1] == 1.0  # and the target is then reached
