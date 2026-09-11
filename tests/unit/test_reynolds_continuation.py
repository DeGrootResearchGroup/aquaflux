"""Unit tests for Reynolds-number continuation (schedule + the viscosity rescale, no solve).

The end-to-end "reaches the same root" and "gradient matches a direct solve" gates live in
``tests/integration/test_reynolds_continuation.py`` -- these cover the pure pieces: the geometric
schedule (physics-free) and the molecular-viscosity rescale on a small built coupled system.
"""

from __future__ import annotations

import itertools
import math

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax
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
    ViscosityRampHomotopy,
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


# --- the within-march viscosity ramp ----------------------------------------------------


def test_the_ramp_walks_the_viscosity_geometrically_down_to_exactly_the_target() -> None:
    """Equal ratios between stations, and the last one is the case's own viscosity exactly.

    Geometric rather than linear because the convective nonlinearity scales multiplicatively with the
    Reynolds number, so equal *ratios* are equal difficulty; and exactly ``1.0`` at the end because a
    ramp that stopped near the target would leave the march solving a slightly wrong problem.
    """
    ramp = ViscosityRampHomotopy(_tiny_coupled(), anchor=100.0, stations=4, steps_per_station=3)
    scales = [ramp.scale(s) for s in range(5)]

    assert scales[0] == 100.0
    assert scales[-1] == 1.0
    ratios = [a / b for a, b in itertools.pairwise(scales)]
    assert all(math.isclose(r, ratios[0]) for r in ratios)
    assert math.isclose(ratios[0], 100.0 ** (1 / 4))


def test_the_ramp_holds_each_station_for_its_step_budget_and_then_arrives() -> None:
    """The step-to-station map, and that arrival is exactly where the ramp's budget ends."""
    ramp = ViscosityRampHomotopy(_tiny_coupled(), anchor=100.0, stations=3, steps_per_station=2)

    assert [ramp.station(i) for i in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [ramp.arrived(i) for i in range(8)] == [False] * 6 + [True, True]
    assert ramp.ramp_steps == 6


def test_the_ramp_rebinds_once_per_station_change_not_once_per_step() -> None:
    """The affordability property: a station change refits a preconditioner, so it must be rare.

    Moving the viscosity every step would force a full preconditioner rebuild every step -- the most
    expensive operation in the march -- which would cost far more than the steps the ramp saves.
    """
    seen: list[float] = []
    coupled = _tiny_coupled()
    ramp = ViscosityRampHomotopy(
        coupled,
        anchor=100.0,
        stations=3,
        steps_per_station=4,
        rebind=lambda assembler: seen.append(
            float(jnp.asarray(assembler.turbulence.molecular_viscosity).ravel()[0])
        ),
    )

    for step in range(12):
        ramp.enter(step)

    assert len(seen) == 3  # twelve steps, three stations
    assert math.isclose(seen[0], NU * 100.0, rel_tol=1e-12)


def test_the_ramp_hands_back_the_CASES_OWN_assembler_at_the_target() -> None:
    """Identity, not a rescale by one: the root and its adjoint must belong to the case.

    ``with_scaled_molecular_viscosity(1.0)`` returns a different object holding a multiplied array. It
    is numerically the same problem, but the assembler is the differentiable parameter pytree the
    adjoint is taken with respect to, so handing back a copy is not the same thing as handing back the
    case.
    """
    coupled = _tiny_coupled()
    ramp = ViscosityRampHomotopy(coupled, anchor=100.0, stations=2, steps_per_station=1)

    assert ramp.enter(0).__self__ is not coupled  # a ramp station is a rescaled companion
    assert ramp.enter(2).__self__ is coupled  # the target station is the case itself


def test_each_station_is_the_assembler_at_that_stations_own_scale() -> None:
    """The wiring: the station handed to the march is the case rescaled by that station's own factor.

    That the rescale itself moves both viscosity leaves consistently is
    ``test_coupled_rescale_scales_both_viscosity_leaves``; what is pinned here is that the ramp feeds
    it :meth:`scale` and hands the result on, rather than that the rescale works.
    """
    coupled = _tiny_coupled()
    ramp = ViscosityRampHomotopy(coupled, anchor=100.0, stations=2, steps_per_station=1)

    for step in (0, 1):
        station = ramp.enter(step).__self__
        assert _dynamic_viscosity(station) == pytest.approx(ramp.scale(step) * RHO * NU)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(anchor=0.5, stations=2, steps_per_station=1), "anchor must be >= 1"),
        (dict(anchor=10.0, stations=0, steps_per_station=1), "stations must be >= 1"),
        (dict(anchor=10.0, stations=2, steps_per_station=0), "steps_per_station must be >= 1"),
    ],
)
def test_the_ramp_refuses_a_configuration_that_cannot_walk_down_to_the_target(kwargs, message):
    """An anchor below one ramps the wrong way, and a zero budget is a ramp that never runs."""
    with pytest.raises(ValueError, match=message):
        ViscosityRampHomotopy(_tiny_coupled(), **kwargs)


def test_the_ramp_derives_its_re_damping_from_the_stations_own_ratio() -> None:
    """A station that barely moves the problem barely damps -- which a constant cannot express.

    Pinned at both ends of the range rather than at one point: the coarse setting must reproduce the
    value that was calibrated by measurement, and the fine one must come out near unity, because the
    same constant serving both is exactly what the derivation replaces.
    """
    coarse = ViscosityRampHomotopy(_tiny_coupled(), anchor=100.0, stations=4, steps_per_station=3)
    fine = ViscosityRampHomotopy(_tiny_coupled(), anchor=100.0, stations=24, steps_per_station=3)

    assert coarse.redamping == pytest.approx(2.0, rel=1e-2)  # the measured-good four-station value
    assert fine.redamping == pytest.approx(1.12, rel=1e-2)
    assert coarse.redamping == pytest.approx(coarse.ratio**0.6)


def test_one_step_per_station_forces_re_damping_off() -> None:
    """At one step per station every step ENTERS one, so the control is held on every step.

    It then never adapts at all and the shift is whatever ``redamping ** n`` makes it -- a runaway, not
    a damping. The default must therefore be exactly 1.0 there whatever the ratio rule would give, and
    an explicit value must be refused rather than silently corrected.
    """
    ramp = ViscosityRampHomotopy(_tiny_coupled(), anchor=100.0, stations=24, steps_per_station=1)
    assert ramp.redamping == 1.0

    with pytest.raises(ValueError, match=r"must be exactly 1\.0 when steps_per_station == 1"):
        ViscosityRampHomotopy(
            _tiny_coupled(), anchor=100.0, stations=24, steps_per_station=1, redamping=2.0
        )


def test_an_explicit_re_damping_still_overrides_the_derived_one() -> None:
    """The derivation is a default, not a policy -- a caller who measured a value keeps it."""
    ramp = ViscosityRampHomotopy(
        _tiny_coupled(), anchor=100.0, stations=4, steps_per_station=3, redamping=3.0
    )
    assert ramp.redamping == 3.0


# --- damping the closure's rows harder than the flow's ------------------------------------------


class _StubHostPreconditioner:
    """The narrowest stand-in ``MonolithicFactorShiftPolicy`` accepts: it only asks for a matvec."""

    solves_exactly_on_host = False

    def matvec(self):
        return lambda x: x


def _shift(coupled, state, damping, beta=0.5):
    """The full-state shift ``beta scale(beta) d`` a monolithically preconditioned step forms.

    The production quantity: damping multiplies the shift STRENGTH, not the base diagonal, so a test
    that read ``.diagonal`` would see no damping at all and pass for the wrong reason.
    """
    from aquaflux.turbulence.coupled import _DEFAULT_SHIFT_BASIS, _monolithic_shift_source

    source = _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS, None, damping)
    return source.shift_term(state).shift(jnp.asarray(beta))


def _seeded_state(coupled):
    from aquaflux.turbulence import hybrid_initialize

    return coupled.state_from_physical(*hybrid_initialize(coupled.momentum, coupled.turbulence))


def test_turbulence_damping_scales_the_closures_rows_and_leaves_the_flow_rows_alone() -> None:
    """The whole point: one shift for the flow, a harder one for the stiff source terms.

    The two blocks are hard for different reasons -- the momentum block's convective nonlinearity,
    which a viscosity continuation weakens, against the closure's stiff sources, which it barely
    touches -- so the shift they share must be able to differ between them.
    """
    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    n, dim = coupled.momentum.mesh.n_cells, coupled.momentum.mesh.dim
    flow = (dim + 1) * n

    plain = _shift(coupled, state, 1.0)
    damped = _shift(coupled, state, 10.0)

    assert jnp.allclose(plain[:flow], damped[:flow])
    assert jnp.allclose(damped[flow:], 10.0 * plain[flow:])


def test_swapping_the_damping_RATIO_is_a_compilation_cache_hit_not_a_recompile() -> None:
    """What makes a per-station damping affordable at all, and it is a property of one field's TYPE.

    ``equinox.filter_jit`` partitions on "is this an array": array leaves are traced, everything else
    rides on the **static** side and is compared by value, so a change there is a new cache key. A
    ratio stored as the Python float it is constructed from would therefore recompile the entire
    coupled solve once per station -- turning the cheapest setting in the march into its dominant cost,
    which is exactly what a ``float`` molecular viscosity did to the Reynolds ramp.

    The check is the one ``filter_jit`` itself makes: the two policies' static halves must be equal,
    and the ratio must land in the dynamic half.
    """
    from aquaflux.turbulence import ConstantDamping
    from aquaflux.turbulence.coupled import _DEFAULT_SHIFT_BASIS, _monolithic_shift_source

    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    five, two = (
        _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS, None, gamma)
        for gamma in (5.0, 2.0)
    )

    assert eqx.is_array(ConstantDamping(5.0).ratio), "the ratio must be traced, not static"
    dynamic_five, static_five = eqx.partition(five, eqx.is_array)
    dynamic_two, static_two = eqx.partition(two, eqx.is_array)
    assert eqx.tree_equal(static_five, static_two) is True
    assert jax.tree.structure(dynamic_five) == jax.tree.structure(dynamic_two)

    # And the swap a station hook performs keeps that property, rather than only a fresh build doing so.
    swapped = eqx.tree_at(lambda p: p.turbulence_damping, five, ConstantDamping(2.0))
    assert float(swapped.turbulence_damping.factor(jnp.asarray(0.5), None)) == 2.0
    assert eqx.tree_equal(eqx.partition(swapped, eqx.is_array)[1], static_two) is True


def test_damping_leaves_the_BASE_DIAGONAL_alone_so_the_marchs_measure_cannot_move_with_it() -> None:
    """The row scale of ``coupled_scaled_norm`` IS this diagonal, so damping must not reach it.

    A factor folded into the diagonal divides the ``k``/``omega`` rows of the residual the march is
    steered by, stopped on, and compared across arms with -- so two damping settings would be judged
    on two different measures and could not be compared at all. It would also multiply the shift's
    FLOOR, since ``beta`` arrives already clamped at ``beta_min``.
    """
    from aquaflux.turbulence.coupled import (
        _DEFAULT_SHIFT_BASIS,
        _monolithic_shift_source,
        coupled_scaled_norm,
    )

    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    plain, damped = (
        _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS, None, gamma)
        for gamma in (1.0, 10.0)
    )
    assert jnp.array_equal(plain.shift_term(state).diagonal, damped.shift_term(state).diagonal)

    residual = coupled.residual(state)
    assert coupled_scaled_norm(coupled, plain, state)(residual) == pytest.approx(
        float(coupled_scaled_norm(coupled, damped, state)(residual))
    )


def test_a_damping_of_one_is_bit_identical_to_no_damping_at_all() -> None:
    """The default must not move any incumbent march by a rounding."""
    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    from aquaflux.turbulence.coupled import _DEFAULT_SHIFT_BASIS, _monolithic_shift_source

    default = _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS)
    assert default.turbulence_damping.factor(jnp.asarray(0.5), None) == 1.0

    # The multiplier rides through as an array of ones, and `beta * 1.0 == beta` exactly in IEEE
    # arithmetic -- so an undamped march is bit-identical, not merely close.
    beta = jnp.asarray(0.5)
    term = default.shift_term(state)
    assert jnp.array_equal(term.shift(beta), beta * term.diagonal)
    assert jnp.array_equal(_shift(coupled, state, 1.0, 0.5), beta * term.diagonal)


def test_the_shift_still_vanishes_at_the_root_under_damping() -> None:
    """Why this cannot move the answer: the shift TERM is ``beta * d * (phi - phi_n)``.

    Whatever the per-block multiplier, a state that solves the steady residual has ``phi == phi_n``
    and the whole term is zero -- so damping changes the path a march takes and neither the converged
    root nor its adjoint. That is the same property that licenses ``beta`` itself, and it is the reason
    this is a globalization knob rather than a model change.
    """
    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    for damping in (1.0, 3.0, 25.0):
        term = _shift(coupled, state, damping)
        assert jnp.all(jnp.isfinite(term))
        # `d * (phi - phi_n)` at `phi == phi_n`, which is what the march adds to the residual.
        assert jnp.allclose(term * (state - state), 0.0)


def test_a_refresh_CARRIES_the_damping_rather_than_dropping_it_to_one() -> None:
    """A refresh rebuilds the transport diagonal; the caller's damping is configuration, not state.

    ``shift_basis`` and ``velocity_shift_parts`` are carried across a refresh for exactly this reason,
    and a damping silently reset to ``1.0`` mid-march would be invisible -- the march would simply get
    harder at the step the preconditioner was refreshed.
    """
    from aquaflux.turbulence.coupled import (
        _DEFAULT_SHIFT_BASIS,
        _coupled_shift_policy,
        _monolithic_shift_source,
    )

    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    built = _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS, None, 7.0)
    assert built.turbulence_damping.factor(jnp.asarray(0.5), None) == 7.0

    refreshed = _coupled_shift_policy(
        coupled, state, None, reuse=built, turbulence_damping=1.0, build_flow_block=False
    )
    assert refreshed.turbulence_damping.factor(jnp.asarray(0.5), None) == 7.0


# --- tapering the damping on the closure's own residual ------------------------------------------


def test_the_taper_starts_at_its_initial_ratio_and_reaches_exactly_one_at_convergence() -> None:
    """The two ends that matter: damped where the closure is far from equilibrium, released at the root.

    Ending at ``1`` is a choice about the path rather than a correctness requirement -- the shift term
    is zero at the root whatever multiplies it, which is what licenses a CONSTANT ratio at all -- so what
    this pins is the taper's declared shape, not an invariant.
    """
    from aquaflux.turbulence import ResidualTaperedDamping

    coupled = _tiny_coupled()
    reference = jnp.asarray(4.0)
    taper = ResidualTaperedDamping(initial=10.0, reference=reference, layout=coupled.layout)

    n = coupled.momentum.mesh.n_cells
    at_reference = coupled.layout.pack(
        jnp.zeros((coupled.momentum.mesh.dim + 1) * n),
        jnp.full(n, 4.0 / jnp.sqrt(2 * n)),
        jnp.full(n, 4.0 / jnp.sqrt(2 * n)),
    )
    assert taper.factor(jnp.asarray(0.5), at_reference) == pytest.approx(10.0, rel=1e-6)

    converged = jnp.zeros_like(at_reference)
    assert taper.factor(jnp.asarray(0.5), converged) == pytest.approx(1.0)


def test_the_taper_opens_at_its_initial_ratio_when_no_residual_exists_yet() -> None:
    """The march's FIRST shift is built before any residual is formed -- and that is the step the
    damping is most for, so ``None`` must not read as "converged"."""
    from aquaflux.turbulence import ResidualTaperedDamping

    coupled = _tiny_coupled()
    taper = ResidualTaperedDamping(initial=8.0, reference=jnp.asarray(1.0), layout=coupled.layout)
    assert taper.factor(jnp.asarray(0.5), None) == pytest.approx(8.0)


def test_the_taper_reads_the_CLOSURE_residual_and_ignores_the_flow_rows() -> None:
    """The point of keying on the turbulence block: the mean flow's residual must not move the ratio.

    The two blocks settle on different schedules, so a rule that read the whole-state norm would be
    dominated by the flow and would release the damping on the flow's progress rather than the
    closure's.
    """
    from aquaflux.turbulence import ResidualTaperedDamping

    coupled = _tiny_coupled()
    n, dim = coupled.momentum.mesh.n_cells, coupled.momentum.mesh.dim
    taper = ResidualTaperedDamping(initial=5.0, reference=jnp.asarray(1.0), layout=coupled.layout)

    quiet_flow = coupled.layout.pack(jnp.zeros((dim + 1) * n), jnp.full(n, 0.1), jnp.full(n, 0.1))
    loud_flow = coupled.layout.pack(
        jnp.full((dim + 1) * n, 1e3), jnp.full(n, 0.1), jnp.full(n, 0.1)
    )
    assert taper.factor(jnp.asarray(0.5), quiet_flow) == pytest.approx(
        float(taper.factor(jnp.asarray(0.5), loud_flow))
    )


def test_a_residual_that_RISES_re_damps_rather_than_releasing() -> None:
    """A continuation station change moves the problem, so the closure is far from equilibrium again.

    The taper's response there is to damp harder, which is the same thing
    ``ShiftStrengthControl.redamp`` does on entering a station -- reached from a different direction.
    """
    from aquaflux.turbulence import ResidualTaperedDamping, turbulence_residual_norm

    coupled = _tiny_coupled()
    n, dim = coupled.momentum.mesh.n_cells, coupled.momentum.mesh.dim
    flow = jnp.zeros((dim + 1) * n)
    settled = coupled.layout.pack(flow, jnp.full(n, 0.01), jnp.full(n, 0.01))
    disturbed = coupled.layout.pack(flow, jnp.full(n, 0.5), jnp.full(n, 0.5))
    reference = turbulence_residual_norm(coupled.layout, disturbed)

    taper = ResidualTaperedDamping(initial=6.0, reference=reference, layout=coupled.layout)
    assert float(taper.factor(jnp.asarray(0.5), disturbed)) > float(
        taper.factor(jnp.asarray(0.5), settled)
    )


def test_the_taper_is_clamped_so_a_residual_above_its_reference_never_exceeds_the_initial() -> None:
    """A march that gets worse than it started must not run away to an unbounded shift."""
    from aquaflux.turbulence import ResidualTaperedDamping

    coupled = _tiny_coupled()
    n, dim = coupled.momentum.mesh.n_cells, coupled.momentum.mesh.dim
    taper = ResidualTaperedDamping(initial=3.0, reference=jnp.asarray(1.0), layout=coupled.layout)
    blown_up = coupled.layout.pack(jnp.zeros((dim + 1) * n), jnp.full(n, 1e6), jnp.full(n, 1e6))
    assert taper.factor(jnp.asarray(0.5), blown_up) == pytest.approx(3.0)


def test_damping_below_one_is_refused_because_damping_never_accelerates() -> None:
    """Both strategies: a factor under 1 would *reduce* the closure's shift below the flow's."""
    from aquaflux.turbulence import ConstantDamping, ResidualTaperedDamping

    coupled = _tiny_coupled()
    with pytest.raises(ValueError, match="never accelerates"):
        ConstantDamping(0.5)
    with pytest.raises(ValueError, match="never accelerates"):
        ResidualTaperedDamping(initial=0.5, reference=jnp.asarray(1.0), layout=coupled.layout)


# --- tapering the damping on the shift strength --------------------------------------------------


def test_the_residual_taper_survives_the_MONOLITHIC_WRAPPER_the_cases_actually_run() -> None:
    """The plumbing that decided whether the residual taper was ever live, end to end on the real path.

    Both flagship cases precondition monolithically, so their shift policy is a
    ``MonolithicFactorShiftPolicy`` wrapping the block one. A wrapper that rebuilt its ``ShiftTerm``
    from ``base.shift_term(phi).diagonal`` alone dropped BOTH the residual on the way in and the
    per-row multiplier on the way out -- in silence, so the march ran and the taper simply never
    happened. Asserting on the base policy alone cannot see that; this drives the wrapper.
    """
    from aquaflux.turbulence import ResidualTaperedDamping, turbulence_residual_norm
    from aquaflux.turbulence.coupled import (
        _DEFAULT_SHIFT_BASIS,
        MonolithicFactorShiftPolicy,
        _monolithic_shift_source,
    )

    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    n, dim = coupled.momentum.mesh.n_cells, coupled.momentum.mesh.dim
    flow = (dim + 1) * n

    residual = coupled.residual(state)
    taper = ResidualTaperedDamping(
        initial=10.0,
        reference=turbulence_residual_norm(coupled.layout, residual),
        layout=coupled.layout,
    )
    base = _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS, None, taper)
    wrapped = MonolithicFactorShiftPolicy(base, _StubHostPreconditioner())

    beta = jnp.asarray(0.5)
    at_reference = wrapped.shift_term(state, residual).shift(beta)
    converged = wrapped.shift_term(state, jnp.zeros_like(residual)).shift(beta)
    plain = _shift(coupled, state, 1.0, 0.5)

    assert jnp.allclose(at_reference[flow:], 10.0 * plain[flow:])
    assert jnp.allclose(converged[flow:], plain[flow:])
    assert jnp.allclose(at_reference[:flow], plain[:flow])


def test_the_beta_taper_opens_at_its_initial_ratio_and_reaches_exactly_one_at_the_floor() -> None:
    """The two ends that matter: damped while the pseudo-timestep is small, released once it is not.

    Ending at ``1`` is the taper's declared shape, not a correctness requirement (the shift is zero at
    the root whatever multiplies it, which is what licenses a constant ratio) -- and it is why ``beta``
    flooring early is not the objection it looks like: what the taper freezes at is the value it was
    walking toward.
    """
    from aquaflux.turbulence import BetaTaperedDamping

    taper = BetaTaperedDamping(initial=10.0, beta_start=0.5, beta_min=0.005)
    assert taper.factor(jnp.asarray(0.5), None) == pytest.approx(10.0)
    assert taper.factor(jnp.asarray(0.005), None) == pytest.approx(1.0)


def test_the_beta_taper_is_clamped_at_BOTH_ends_so_an_escalation_cannot_run_away() -> None:
    """The retry ladder RAISES ``beta`` on a bad step, and a sub-floor ``beta`` is still a floor.

    Above ``beta_start`` the factor holds at ``initial`` rather than growing without bound -- the right
    direction (a step that had to be re-damped is one whose closure was moving too fast) but a bounded
    amount of it.
    """
    from aquaflux.turbulence import BetaTaperedDamping

    taper = BetaTaperedDamping(initial=4.0, beta_start=0.5, beta_min=0.005)
    assert taper.factor(jnp.asarray(16.0), None) == pytest.approx(4.0)
    assert taper.factor(jnp.asarray(1e-8), None) == pytest.approx(1.0)


def test_the_beta_taper_releases_linearly_in_LOG_beta_so_it_is_linear_in_outer_steps() -> None:
    """A Courant control divides ``beta`` by a constant per comfortable step, so ``log beta`` is what
    moves linearly with the step index. Keyed on ``beta`` itself the release would be spent in the
    first few steps of a descent that spans a dozen."""
    from aquaflux.turbulence import BetaTaperedDamping

    taper = BetaTaperedDamping(initial=11.0, beta_start=1.0, beta_min=1e-4)
    # Geometric in beta -> arithmetic in the factor.
    factors = [float(taper.factor(jnp.asarray(1e-1 * 10.0**-i), None)) for i in range(3)]
    assert factors == pytest.approx([8.5, 6.0, 3.5])


def test_the_beta_taper_refuses_a_span_it_cannot_run_over() -> None:
    """``beta_min >= beta_start`` is not a taper at all -- and it would divide by zero or invert."""
    from aquaflux.turbulence import BetaTaperedDamping

    with pytest.raises(ValueError, match="never accelerates"):
        BetaTaperedDamping(initial=0.5, beta_start=0.5, beta_min=0.005)
    with pytest.raises(ValueError, match="needs a span"):
        BetaTaperedDamping(initial=4.0, beta_start=0.005, beta_min=0.5)
    with pytest.raises(ValueError, match="needs a span"):
        BetaTaperedDamping(initial=4.0, beta_start=0.5, beta_min=0.0)


def test_a_beta_tapered_shift_is_damped_at_the_start_and_undamped_at_the_floor() -> None:
    """End to end on the real policy: the same object gives two different shifts at two betas.

    This is what a factor folded into the base diagonal could not do -- ``beta`` reaches the step
    already clamped, so a diagonal factor multiplies the floor and the closure is still damped at the
    step the march has switched damping off for every other block.
    """
    from aquaflux.turbulence import BetaTaperedDamping

    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    n, dim = coupled.momentum.mesh.n_cells, coupled.momentum.mesh.dim
    flow = (dim + 1) * n
    taper = BetaTaperedDamping(initial=10.0, beta_start=0.5, beta_min=0.005)

    opened = _shift(coupled, state, taper, beta=0.5)
    floored = _shift(coupled, state, taper, beta=0.005)
    plain_open = _shift(coupled, state, 1.0, beta=0.5)
    plain_floor = _shift(coupled, state, 1.0, beta=0.005)

    assert jnp.allclose(opened[flow:], 10.0 * plain_open[flow:])
    assert jnp.allclose(floored[flow:], plain_floor[flow:])
    assert jnp.allclose(opened[:flow], plain_open[:flow])


def test_a_refresh_REBUILDS_the_tapers_reference_rather_than_carrying_it() -> None:
    """The defect that made the first taper inert, pinned.

    A reference is physics -- how hard the closure's problem is *now* -- so a refresh must re-derive it
    at the developed state, exactly as ``k_shift_transport`` is re-derived while ``k_jacobian_scale`` is
    carried. Carried instead, a continuation march measures the taper against the anchor's problem,
    which is the easiest it ever sees, and the factor never leaves ``initial``.
    """
    from aquaflux.turbulence import ResidualTaperedDamping, turbulence_residual_norm
    from aquaflux.turbulence.coupled import (
        _DEFAULT_SHIFT_BASIS,
        _coupled_shift_policy,
        _monolithic_shift_source,
    )

    coupled = _tiny_coupled()
    state = _seeded_state(coupled)
    stale = ResidualTaperedDamping(initial=9.0, reference=jnp.asarray(1e9), layout=coupled.layout)
    built = _monolithic_shift_source(coupled, state, _DEFAULT_SHIFT_BASIS, None, stale)
    assert float(built.turbulence_damping.reference) == pytest.approx(1e9)

    refreshed = _coupled_shift_policy(coupled, state, None, reuse=built, build_flow_block=False)
    # The configuration survives; the reference is the one this state actually implies.
    assert refreshed.turbulence_damping.initial == 9.0
    assert float(refreshed.turbulence_damping.reference) == pytest.approx(
        float(turbulence_residual_norm(coupled.layout, coupled.residual(state))), rel=1e-9
    )


def test_a_constant_damping_is_unchanged_by_a_rebase() -> None:
    """Nothing to re-derive, so the rebase must be an identity rather than a silent reset."""
    from aquaflux.turbulence import ConstantDamping

    coupled = _tiny_coupled()
    constant = ConstantDamping(4.0)
    assert constant.rebased(coupled, _seeded_state(coupled)) is constant


# --- which viscosity a station scales -------------------------------------------------------------


def test_scaling_both_blocks_moves_the_momentum_and_the_closure_together() -> None:
    """The default: a station is a genuine lower-Reynolds problem, so both viscosities move."""
    from aquaflux.turbulence import scale_both_blocks

    coupled = _tiny_coupled()
    scaled = scale_both_blocks(coupled, 100.0)

    assert float(scaled.momentum.properties.properties["viscosity"].value) == pytest.approx(
        float(coupled.momentum.properties.properties["viscosity"].value) * 100.0
    )
    assert jnp.allclose(
        scaled.turbulence.molecular_viscosity, coupled.turbulence.molecular_viscosity * 100.0
    )


def test_scaling_momentum_only_leaves_the_closures_viscosity_untouched() -> None:
    """The alternative: only the convective nonlinearity is weakened, the closure is not touched."""
    from aquaflux.turbulence import scale_momentum_only

    coupled = _tiny_coupled()
    scaled = scale_momentum_only(coupled, 100.0)

    assert float(scaled.momentum.properties.properties["viscosity"].value) == pytest.approx(
        float(coupled.momentum.properties.properties["viscosity"].value) * 100.0
    )
    assert jnp.allclose(
        scaled.turbulence.molecular_viscosity, coupled.turbulence.molecular_viscosity
    )


def test_scaling_momentum_only_makes_the_anchor_and_target_hybrid_starts_IDENTICAL() -> None:
    """Why the alternative is interesting: it removes the seed choice rather than answering it.

    ``hybrid_initialize`` reads the closure's molecular viscosity to place the near-wall ``omega``, so
    under :func:`scale_both_blocks` an anchor-built seed and a target-built one differ there by the
    viscosity ratio -- the whole reason the ramp arm has to be explicit about which one it opens from.
    Leaving the closure alone makes the two seeds the same fields, so the question cannot be got wrong.
    """
    from aquaflux.turbulence import hybrid_initialize, scale_both_blocks, scale_momentum_only

    coupled = _tiny_coupled()
    target = hybrid_initialize(coupled.momentum, coupled.turbulence)

    momentum_only = scale_momentum_only(coupled, 100.0)
    for mine, theirs in zip(
        hybrid_initialize(momentum_only.momentum, momentum_only.turbulence), target, strict=True
    ):
        assert jnp.allclose(mine, theirs)

    # The contrast that makes the point: scaling both blocks moves the seed.
    both = scale_both_blocks(coupled, 100.0)
    assert not all(
        jnp.allclose(mine, theirs)
        for mine, theirs in zip(
            hybrid_initialize(both.momentum, both.turbulence), target, strict=True
        )
    )


def test_the_ramp_still_ends_on_the_callers_own_assembler_under_either_scaling() -> None:
    """The target station is never built by the strategy -- the root and adjoint belong to the case."""
    from aquaflux.turbulence import scale_both_blocks, scale_momentum_only

    for companion in (scale_both_blocks, scale_momentum_only):
        ramp = ViscosityRampHomotopy(
            _tiny_coupled(), anchor=100.0, stations=4, steps_per_station=1, companion=companion
        )
        ramp.enter(0)
        assert ramp._assembler is not ramp.coupled
        ramp.enter(4)
        assert ramp._assembler is ramp.coupled


# --- the ramp ARM: one march over the ladder's span, configured by the ladder's own hooks ---------


def _ramp_arm_fixtures(monkeypatch):
    """Stub out the solve and the hybrid start, and record what the ramp arm hands ``solve_coupled``.

    Returns ``(coupled, calls, fields)``: the target assembler, the recorded call dicts, and the
    correctly-shaped stand-in converged fields the stubs pass around.
    """
    import aquaflux.turbulence.reynolds as reynolds

    coupled = _tiny_coupled()
    n = coupled.momentum.mesh.n_cells
    dim = coupled.momentum.mesh.dim
    fields = (jnp.zeros((dim + 1) * n), jnp.full(n, 0.5), jnp.full(n, 100.0))
    calls: list[dict] = []

    def fake_solve_coupled(assembler, flow=None, k=None, omega=None, **kwargs):
        scale = float(assembler.momentum.properties.properties["viscosity"].value / (RHO * NU))
        calls.append({"scale": scale, "seed_is_none": flow is None, "kwargs": kwargs})
        return fields

    monkeypatch.setattr(reynolds, "solve_coupled", fake_solve_coupled)
    monkeypatch.setattr(reynolds, "hybrid_initialize", lambda momentum, turbulence: fields)
    return coupled, calls, fields


def test_the_ramp_arm_is_one_warm_started_solve_on_the_target_carrying_the_homotopy(
    monkeypatch,
) -> None:
    """The whole point of the arm: ONE march, on the case's own assembler, walking the span inside it.

    The ladder's `n` rungs are `n + 1` separate converged solves; this must be a single call, and it
    must be on the target -- the root and its adjoint belong to the case, not to a rescaled companion.
    """
    from aquaflux.turbulence import solve_reynolds_ramp

    coupled, calls, _ = _ramp_arm_fixtures(monkeypatch)

    solve_reynolds_ramp(
        coupled,
        anchor=100.0,
        stations=24,
        steps_per_station=1,
        point_setup=lambda companion, state, point: {},
        rtol=1e-10,
    )

    assert len(calls) == 1
    assert calls[0]["scale"] == pytest.approx(1.0)
    # Warm-started from the materialized seed, so the anchor station's preconditioner is frozen at the
    # state the march actually begins from rather than at one solve_coupled invents afterwards.
    assert calls[0]["seed_is_none"] is False
    homotopy = calls[0]["kwargs"]["homotopy"]
    assert isinstance(homotopy, ViscosityRampHomotopy)
    assert (homotopy.anchor, homotopy.stations, homotopy.steps_per_station) == (100.0, 24, 1)
    assert calls[0]["kwargs"]["rtol"] == 1e-10


def test_the_ramp_arm_seeds_the_hybrid_start_from_the_anchor_not_from_the_target(
    monkeypatch,
) -> None:
    """The seed is built at the viscosity the march BEGINS at, which is the anchor's, not the target's.

    ``hybrid_initialize`` reads the assembler's own molecular viscosity to seed the near-wall ``omega``
    at the profile that assembler's residual fixes there. Seeded from the target while the first station
    solves the anchor, every wall-adjacent cell starts off its own boundary condition by the viscosity
    ratio -- exactly the residual the seeding exists to remove, and by construction invisible to any
    check that stubs ``hybrid_initialize`` without looking at what it was handed. Measured on a
    three-dimensional backward-facing step at a hundredfold anchor, that mismatch moved ``omega`` by up
    to 90x and cost the anchor's first step two discarded attempts and a shift escalation to 4.0.
    """
    import aquaflux.turbulence.reynolds as reynolds
    from aquaflux.turbulence import solve_reynolds_ramp

    coupled, _, fields = _ramp_arm_fixtures(monkeypatch)
    seeded_with: list[float] = []

    def recording_hybrid_initialize(momentum, turbulence):
        seeded_with.append(float(momentum.properties.properties["viscosity"].value / (RHO * NU)))
        return fields

    monkeypatch.setattr(reynolds, "hybrid_initialize", recording_hybrid_initialize)

    solve_reynolds_ramp(
        coupled,
        anchor=100.0,
        stations=24,
        steps_per_station=1,
        point_setup=lambda companion, state, point: {},
    )

    assert seeded_with == [pytest.approx(100.0)]


def test_the_ramp_arm_configures_its_anchor_station_through_the_ladders_own_point_setup(
    monkeypatch,
) -> None:
    """Called ONCE, with the anchor's companion -- and its keywords merged, exactly as the ladder does.

    Reusing the hook is what makes the two arms comparable: they are then preconditioned, logged and
    step-controlled identically and differ only in how the viscosity span is walked.
    """
    from aquaflux.turbulence import solve_reynolds_ramp

    coupled, calls, _ = _ramp_arm_fixtures(monkeypatch)
    seen = []

    def point_setup(companion, state, point):
        scale = float(companion.momentum.properties.properties["viscosity"].value / (RHO * NU))
        seen.append((scale, point.index, point.total, point.viscosity_scale))
        return {"tag": scale}

    solve_reynolds_ramp(
        coupled, anchor=100.0, stations=4, steps_per_station=3, point_setup=point_setup
    )

    assert seen == [(100.0, 1, 1, 100.0)]
    assert calls[0]["kwargs"]["tag"] == 100.0


def test_the_ramp_arm_drops_exactly_the_keywords_the_ladder_owns(monkeypatch) -> None:
    """One options dict must drive either arm, so the ladder-only keywords are stripped, not forwarded.

    Forwarding one is a ``TypeError`` at ``solve_coupled``. The set is derived from the two signatures
    rather than listed, so a keyword added to the ladder cannot quietly break this arm later.
    """
    from aquaflux.turbulence import solve_reynolds_ramp
    from aquaflux.turbulence.reynolds import _LADDER_ONLY

    coupled, calls, _ = _ramp_arm_fixtures(monkeypatch)

    # The ladder's OWN options dict, handed over whole -- `point_setup` included, because that is how
    # a case calls this and passing it separately beside the dict is the same keyword twice.
    ladder_options = dict(
        schedule=GeometricReynoldsSchedule(ratio=10.0),
        intermediate_rtol=1e-2,
        intermediate_atol=1e-5,
        seed_projection=lambda assembler, state, point: state,
        point_setup=lambda companion, state, point: {},
        rtol=1e-10,
        atol=1e-5,
        max_steps=7,
    )
    solve_reynolds_ramp(coupled, anchor=100.0, stations=4, steps_per_station=3, **ladder_options)

    passed = calls[0]["kwargs"]
    assert not (_LADDER_ONLY & set(passed))
    assert (passed["rtol"], passed["atol"], passed["max_steps"]) == (1e-10, 1e-5, 7)


def test_the_ladder_only_keywords_are_derived_from_the_two_signatures() -> None:
    """Pinned against the signatures themselves, because a hand-copied list goes stale in silence.

    The failure it guards is one-directional and invisible: a keyword added to the ladder and not to
    this set is forwarded to ``solve_coupled``, which raises -- but only in whichever case selects the
    ramp arm, which no test tier runs.
    """
    import inspect

    from aquaflux.turbulence import solve_coupled, solve_reynolds_continuation
    from aquaflux.turbulence.reynolds import _LADDER_ONLY

    ladder = set(inspect.signature(solve_reynolds_continuation).parameters)
    solve = set(inspect.signature(solve_coupled).parameters)
    assert _LADDER_ONLY == ladder - solve - {"solve_kwargs"}
    assert {"schedule", "intermediate_rtol", "intermediate_atol", "seed_projection"} <= _LADDER_ONLY
    # `point_setup` is the ramp arm's own parameter and never travels in the forwarded options.
    assert "point_setup" in _LADDER_ONLY


def test_the_ramp_arm_hands_the_homotopy_the_refreshs_own_rebind(monkeypatch) -> None:
    """A station change must re-point the frozen preconditioner at the station's own assembler.

    Without it every station after the first solves against a V-cycle built for the anchor's
    viscosity -- a march that still converges, only slowly, which is why it is wired rather than left
    to a case to remember.
    """
    from aquaflux.solve import RefreshPolicy
    from aquaflux.turbulence import solve_reynolds_ramp

    coupled, calls, _ = _ramp_arm_fixtures(monkeypatch)
    rebound: list[float] = []

    def precondition_step(active_step, state):  # pragma: no cover - never called here
        raise AssertionError("the stub march does not step")

    precondition_step.rebind = lambda companion: rebound.append(
        float(companion.momentum.properties.properties["viscosity"].value / (RHO * NU))
    )

    solve_reynolds_ramp(
        coupled,
        anchor=100.0,
        stations=2,
        steps_per_station=1,
        point_setup=lambda companion, state, point: {
            "refresh": RefreshPolicy(precondition_step=precondition_step)
        },
    )

    homotopy = calls[0]["kwargs"]["homotopy"]
    # Driving the homotopy's own stations is what proves the wiring: each entry re-points the hook at
    # that station's viscosity, ending at the target's own 1.0.
    for step in range(3):
        homotopy.enter(step)
    assert rebound == pytest.approx([100.0, 10.0, 1.0])


def test_a_refresh_that_cannot_be_repointed_is_refused_rather_than_ignored(monkeypatch) -> None:
    """Its only symptom would be a slower march, so it must raise instead of quietly passing ``None``."""
    from aquaflux.solve import RefreshPolicy
    from aquaflux.turbulence import solve_reynolds_ramp

    coupled, _, _ = _ramp_arm_fixtures(monkeypatch)

    with pytest.raises(ValueError, match="cannot be re-pointed"):
        solve_reynolds_ramp(
            coupled,
            anchor=100.0,
            stations=2,
            steps_per_station=1,
            point_setup=lambda companion, state, point: {
                "refresh": RefreshPolicy(precondition_step=lambda step, state: None)
            },
        )


def test_no_refresh_at_all_is_not_an_error(monkeypatch) -> None:
    """A case with no frozen preconditioner has nothing to re-point, and ``None`` is the honest answer.

    Distinguished from the case above deliberately: answering both with ``None`` would turn a real
    misconfiguration into a silent slowdown.
    """
    from aquaflux.turbulence import solve_reynolds_ramp

    coupled, calls, _ = _ramp_arm_fixtures(monkeypatch)

    solve_reynolds_ramp(
        coupled,
        anchor=100.0,
        stations=2,
        steps_per_station=1,
        point_setup=lambda companion, state, point: {},
    )

    assert calls[0]["kwargs"]["homotopy"].rebind is None
