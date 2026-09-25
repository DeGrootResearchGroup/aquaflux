"""A case file's solver section: read, refused where it is wrong, and handed to the library solve intact.

The library solves are replaced here by recorders, so each test sees exactly the keywords a case hands
its solve -- the settings the file states, nothing it leaves unset, and the observers a script passes.
What the solves do with those keywords is their own tests' business; that a case's solve really runs is
``tests/integration/test_case_solve.py``'s.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path

import aquaflux.case.solver as solver_module
import lineax as lx
import pytest
from aquaflux.case import (
    RANS,
    BulkVelocity,
    CaseSpec,
    CoupledMarch,
    FlowMarch,
    Fluid,
    Laminar,
    Numerics,
    RootSolve,
    Segregated,
    StructuredGrid,
    ViscosityRamp,
    Wall,
    case_spec_from_mapping,
    case_spec_to_mapping,
    read_case,
)
from aquaflux.case.solver import solver_for
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MassFlow, PinnedPoint
from aquaflux.solve import (
    CflResidualDualTimeControl,
    Convergence,
    DirectSolve,
    DualTimeLoop,
    FieldSplit,
    GmresSolve,
    JacobianProbeSpec,
    JacobiSmoothed,
    LinearSolveSettings,
    MaterializedJacobian,
    RetryPolicy,
    RootSolveSettings,
    RowScaled,
    SimpleSmoothed,
    relative_residual_gmres,
)
from aquaflux.turbulence import (
    BlockDiagonal,
    CoupledShiftSettings,
    ScalarAir,
    scale_both_blocks,
    scale_momentum_only,
)

REPO = Path(__file__).resolve().parents[2]


def _sections(physics: str = "RANS", **overrides: object) -> dict[str, object]:
    """A 2D channel between walls, as a file would state it, with ``overrides`` added as sections."""
    sections = {
        "mesh": {"kind": "StructuredGrid", "cells": [4, 4], "lengths": [1.0, 1.0]},
        "fluid": {"density": 1.0, "kinematic_viscosity": 1.0e-3},
        "physics": {
            "kind": physics,
            **({"advection": {"kind": "FirstOrderUpwind"}} if physics == "RANS" else {}),
        },
        "boundaries": {
            "left": {"kind": "Inlet", "velocity": [1.0, 0.0]}
            | (
                {"turbulence": {"kind": "FixedTurbulence", "k": 1e-3, "omega": 1.0}}
                if physics == "RANS"
                else {}
            ),
            "right": {"kind": "Outlet", "pressure": 0.0},
            "bottom": {"kind": "Wall"},
            "top": {"kind": "Wall"},
        },
        "numerics": {"momentum_advection": {"kind": "FirstOrderUpwind"}},
    }
    return {**sections, **overrides}


def _periodic_rans(**overrides: object) -> CaseSpec:
    """A streamwise-periodic RANS channel holding its bulk velocity."""
    return CaseSpec(
        mesh=StructuredGrid(cells=(4, 8), lengths=(1.0, 2.0), periodic=("x",)),
        fluid=Fluid(density=1.0, kinematic_viscosity=1e-4),
        physics=RANS(advection=FirstOrderUpwind()),
        boundaries={"bottom": Wall(k="zero"), "top": Wall(k="zero")},
        numerics=Numerics(momentum_advection=FirstOrderUpwind()),
        drive=BulkVelocity(target=1.0, direction="x"),
        pressure_datum=PinnedPoint((0.0, 0.0)),
        **overrides,
    )


#: A coupled march with every setting a file can state set, as a file states it.
_FULL_COUPLED_MARCH = {
    "kind": "CoupledMarch",
    "max_steps": 150,
    "convergence": {
        "kind": "Convergence",
        "measure": {"kind": "RowScaled"},
        "rtol": 0.0,
        "atol": 1e-5,
    },
    "preconditioner": {
        "kind": "MaterializedJacobian",
        "inverse": {
            "kind": "FieldSplit",
            "leading": {"kind": "SimpleSmoothed", "sweeps": 2},
            "trailing": {"kind": "JacobiSmoothed", "max_coarse": 2000},
        },
        "probe": {"kind": "JacobianProbeSpec", "column_reach": [3, 3, 3, 2, 2]},
        "refit_beta_floor": 0.05,
    },
    "dual_time": {"kind": "DualTimeLoop", "inner_steps": 5, "inner_tol": 0.01},
    "linear_solve": {"kind": "LinearSolveSettings", "rtol": 0.3, "restart": 15},
    "step_control": {"kind": "CflResidualDualTimeControl", "beta_start": 0.5, "beta_min": 0.005},
    "retry": {
        "kind": "RetryPolicy",
        "solver": {"kind": "GmresSolve", "rtol": 1e-4, "restart": 40},
        "abort_above_cycles": 10,
        "on_alpha": 0.01,
    },
    "turbulence_damping": 3.0,
    "positivity_floor": 1e-8,
    "positivity_projection": False,
    "continuation": {
        "kind": "ViscosityRamp",
        "anchor": 100.0,
        "stations": 16,
        "steps_per_station": 1,
        "scale": "flow",
        "redamping": 2.0,
    },
}


# --- reading and writing -----------------------------------------------------------------------


def test_a_coupled_march_reads_into_the_librarys_own_settings_values() -> None:
    spec = case_spec_from_mapping(_sections(solver=_FULL_COUPLED_MARCH))
    assert spec.solver == CoupledMarch(
        max_steps=150,
        convergence=Convergence(measure=RowScaled(), rtol=0.0, atol=1e-5),
        preconditioner=MaterializedJacobian(
            FieldSplit(SimpleSmoothed(sweeps=2), JacobiSmoothed(max_coarse=2000)),
            probe=JacobianProbeSpec(column_reach=(3, 3, 3, 2, 2)),
            refit_beta_floor=0.05,
        ),
        dual_time=DualTimeLoop(inner_steps=5, inner_tol=0.01),
        linear_solve=LinearSolveSettings(rtol=0.3, restart=15),
        step_control=CflResidualDualTimeControl(beta_start=0.5, beta_min=0.005),
        retry=RetryPolicy(
            solver=GmresSolve(1e-4, restart=40), abort_above_cycles=10, on_alpha=0.01
        ),
        turbulence_damping=3.0,
        positivity_floor=1e-8,
        positivity_projection=False,
        continuation=ViscosityRamp(
            anchor=100.0, stations=16, steps_per_station=1, scale="flow", redamping=2.0
        ),
    )


def test_a_solver_section_is_written_as_plain_data_and_read_back_equal() -> None:
    for solver in (
        _FULL_COUPLED_MARCH,
        {
            "kind": "Segregated",
            "sweeps": 100,
            "relaxation": 0.9,
            "flow_solve": {"kind": "RootSolve", "linear_solver": {"kind": "DirectSolve"}},
            "scalar_solve": {
                "kind": "RootSolve",
                "max_steps": 400,
                "convergence": {"kind": "Convergence", "rtol": 1e-8, "atol": 1e-10},
            },
            "scalar_preconditioner": {"kind": "ScalarAir"},
        },
    ):
        spec = case_spec_from_mapping(_sections(solver=solver))
        assert case_spec_to_mapping(spec)["solver"] == solver
        assert case_spec_from_mapping(case_spec_to_mapping(spec)) == spec


# --- refused -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sections", "error", "match"),
    [
        (
            _sections("Laminar", solver={"kind": "CoupledMarch"}),
            ValueError,
            r"CoupledMarch solves a RANS case, but the physics is Laminar; use FlowMarch",
        ),
        (
            _sections("Laminar", solver={"kind": "Segregated", "sweeps": 10}),
            ValueError,
            r"Segregated solves a RANS case, but the physics is Laminar",
        ),
        (
            _sections(solver={"kind": "FlowMarch"}),
            ValueError,
            r"FlowMarch solves a Laminar case, but the physics is RANS",
        ),
        (
            _sections(
                "Laminar",
                solver={"kind": "FlowMarch", "preconditioner": {"kind": "BlockDiagonal"}},
            ),
            TypeError,
            r"BlockDiagonal preconditions a Reynolds-averaged march",
        ),
        (
            _sections(
                solver={
                    "kind": "CoupledMarch",
                    "continuation": {
                        "kind": "ViscosityRamp",
                        "anchor": 1.0,
                        "stations": 4,
                        "steps_per_station": 1,
                    },
                }
            ),
            ValueError,
            r"ViscosityRamp.anchor must be above 1, got 1.0",
        ),
        (
            _sections(
                solver={
                    "kind": "CoupledMarch",
                    "continuation": {
                        "kind": "ViscosityRamp",
                        "anchor": 10.0,
                        "stations": 0,
                        "steps_per_station": 1,
                    },
                }
            ),
            ValueError,
            r"ViscosityRamp.stations must be >= 1, got 0",
        ),
        (
            _sections(
                solver={
                    "kind": "CoupledMarch",
                    "continuation": {
                        "kind": "ViscosityRamp",
                        "anchor": 10.0,
                        "stations": 4,
                        "steps_per_station": 0,
                    },
                }
            ),
            ValueError,
            r"ViscosityRamp.steps_per_station must be >= 1, got 0",
        ),
        (
            _sections(
                solver={
                    "kind": "CoupledMarch",
                    "continuation": {
                        "kind": "ViscosityRamp",
                        "anchor": 10.0,
                        "stations": 4,
                        "steps_per_station": 1,
                        "scale": "momentum",
                    },
                }
            ),
            ValueError,
            r"'momentum' at 'solver.continuation.scale' is not accepted there",
        ),
        (
            _sections(solver={"kind": "CoupledMarch", "max_steps": 0}),
            ValueError,
            r"CoupledMarch.max_steps must be >= 1, got 0",
        ),
        (
            _sections(solver={"kind": "Segregated", "sweeps": 0}),
            ValueError,
            r"Segregated.sweeps must be >= 1, got 0",
        ),
        (
            _sections(solver={"kind": "Segregated"}),
            ValueError,
            r"Segregated at 'solver' needs 'sweeps'",
        ),
        (
            _sections(
                solver={
                    "kind": "CoupledMarch",
                    "retry": {"kind": "RetryPolicy", "solver": {"kind": "GmresSolve", "rtol": 0.0}},
                }
            ),
            ValueError,
            r"GmresSolve.rtol must be a positive, finite number, got 0.0",
        ),
        (
            _sections(
                solver={
                    "kind": "CoupledMarch",
                    "retry": {
                        "kind": "RetryPolicy",
                        "solver": {"kind": "GmresSolve", "rtol": 1e-4, "restart": 0},
                    },
                }
            ),
            ValueError,
            r"GmresSolve.restart must be >= 1, got 0",
        ),
        (
            _sections(solver={"kind": "CoupledMarch", "step_control": {"kind": "CflControl"}}),
            ValueError,
            r"unknown kind 'CflControl' at 'solver.step_control'",
        ),
    ],
    ids=[
        "coupled-march-for-a-laminar-case",
        "segregated-for-a-laminar-case",
        "flow-march-for-a-rans-case",
        "flow-march-with-a-block-diagonal-preconditioner",
        "ramp-anchor-not-above-one",
        "ramp-with-no-stations",
        "ramp-with-no-steps-per-station",
        "ramp-scale-misspelt",
        "no-steps",
        "no-sweeps",
        "segregated-without-sweeps",
        "retry-solver-with-no-tolerance",
        "retry-solver-with-no-restart",
        "step-control-misspelt",
    ],
)
def test_a_solver_that_cannot_solve_the_case_is_refused_when_the_file_is_read(
    sections, error, match
) -> None:
    with pytest.raises(error, match=match):
        case_spec_from_mapping(sections)


@pytest.mark.parametrize(
    ("physics", "march"),
    [(RANS(advection=FirstOrderUpwind()), CoupledMarch()), (Laminar(), FlowMarch())],
    ids=["coupled", "flow"],
)
def test_a_march_is_refused_a_bulk_velocity_it_would_leave_at_its_starting_force(
    physics, march
) -> None:
    with pytest.raises(
        ValueError, match=r"cannot hold the bulk velocity the case's BulkVelocity drive"
    ):
        dataclasses.replace(
            _periodic_rans(),
            physics=physics,
            boundaries={"bottom": Wall(), "top": Wall()},
            solver=march,
        )


def test_a_coupled_march_refused_a_bulk_velocity_names_the_solve_that_holds_one() -> None:
    with pytest.raises(ValueError, match=r"Solve it with a Segregated solver, which holds it"):
        _periodic_rans(solver=CoupledMarch())


# --- the default solver --------------------------------------------------------------------------


def test_a_ramp_given_in_code_refuses_a_scale_no_file_could_name() -> None:
    with pytest.raises(
        ValueError, match=r"ViscosityRamp.scale is one of \['both', 'flow'\] or unset"
    ):
        ViscosityRamp(anchor=10.0, stations=2, steps_per_station=1, scale="momentum")


def test_a_case_that_states_no_solver_runs_its_physics_march_with_the_librarys_settings() -> None:
    assert solver_for(case_spec_from_mapping(_sections())) == CoupledMarch()
    assert solver_for(case_spec_from_mapping(_sections("Laminar"))) == FlowMarch()
    stated = Segregated(sweeps=3)
    assert (
        solver_for(case_spec_from_mapping(_sections(solver={"kind": "Segregated", "sweeps": 3})))
        == stated
    )


def test_a_case_holding_a_bulk_velocity_must_name_its_solver() -> None:
    with pytest.raises(
        ValueError, match=r"the case states no solver, and its default cannot solve it"
    ):
        solver_for(_periodic_rans())


# --- what a solve is handed ----------------------------------------------------------------------


class _Recorder:
    """Stands in for a library solve: records what it was called with and returns a marker."""

    def __init__(self, result: object = "solved") -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def recorded(monkeypatch):
    """Every library solve and session opener the case layer calls, replaced by a recorder."""
    names = (
        "solve_coupled",
        "solve_reynolds_ramp",
        "solve_flow_march",
        "solve_segregated",
        "open_session",
        "bulk_velocity_flow_solve",
        "reused_flow_solve",
        "scalar_pseudo_transient_solve",
        "sst_initial_fields",
    )
    recorders = {name: _Recorder(result=f"<{name}>") for name in names}
    recorders["sst_initial_fields"].result = ("flow0", "k0", "omega0")
    for name, recorder in recorders.items():
        monkeypatch.setattr(solver_module, name, recorder)
    return types.SimpleNamespace(**recorders)


def test_a_coupled_march_hands_solve_coupled_its_settings_and_nothing_it_leaves_unset(
    recorded,
) -> None:
    march = CoupledMarch(
        max_steps=7,
        dual_time=DualTimeLoop(inner_steps=3),
        turbulence_damping=2.0,
        positivity_projection=False,
    )
    assert march.solve("problem", on_retry="logger") == "<solve_coupled>"
    ((args, kwargs),) = recorded.solve_coupled.calls
    assert args == ("problem",)
    assert kwargs == {
        "max_steps": 7,
        "dual_time": DualTimeLoop(inner_steps=3),
        "shift": CoupledShiftSettings(turbulence_damping=2.0),
        "positivity_projection": False,
        "on_retry": "logger",
    }
    assert recorded.solve_reynolds_ramp.calls == []


def test_a_continuation_runs_the_ramp_with_its_own_settings_beside_the_marchs(recorded) -> None:
    ramp = ViscosityRamp(anchor=50.0, stations=6, steps_per_station=2, scale="flow", redamping=1.5)
    CoupledMarch(max_steps=9, continuation=ramp).solve("problem")
    ((args, kwargs),) = recorded.solve_reynolds_ramp.calls
    assert args == ("problem",)
    point_setup = kwargs.pop("point_setup")
    assert point_setup("companion", "state", "point") == {}
    assert kwargs == {
        "anchor": 50.0,
        "stations": 6,
        "steps_per_station": 2,
        "redamping": 1.5,
        "companion": scale_momentum_only,
        "max_steps": 9,
    }
    assert recorded.solve_coupled.calls == []


@pytest.mark.parametrize(
    ("scale", "companion"),
    [("flow", scale_momentum_only), ("both", scale_both_blocks), (None, None)],
)
def test_a_ramp_scales_the_viscosity_its_file_names(recorded, scale, companion) -> None:
    CoupledMarch(
        continuation=ViscosityRamp(anchor=10.0, stations=2, steps_per_station=1, scale=scale)
    ).solve("problem")
    ((_, kwargs),) = recorded.solve_reynolds_ramp.calls
    assert kwargs.get("companion") is companion
    assert ("companion" in kwargs) == (scale is not None)


def test_a_materialized_preconditioner_is_opened_as_one_session_from_the_files_spec(
    recorded,
) -> None:
    spec = MaterializedJacobian(FieldSplit(SimpleSmoothed(), JacobiSmoothed()))
    CoupledMarch(preconditioner=spec).solve("problem", session_options={"observer": "log"})
    ((args, kwargs),) = recorded.open_session.calls
    assert args == (spec, "problem") and kwargs == {"observer": "log"}
    ((_, solve_kwargs),) = recorded.solve_coupled.calls
    assert solve_kwargs["preconditioner"] == "<open_session>"


def test_a_block_diagonal_preconditioner_is_left_to_the_solve(recorded) -> None:
    CoupledMarch(preconditioner=BlockDiagonal()).solve("problem")
    assert recorded.open_session.calls == []
    ((_, kwargs),) = recorded.solve_coupled.calls
    assert kwargs["preconditioner"] == BlockDiagonal()
    with pytest.raises(TypeError, match=r"this march's preconditioner is not one"):
        CoupledMarch(preconditioner=BlockDiagonal()).solve(
            "problem", session_options={"observer": "log"}
        )


@pytest.mark.parametrize(
    "keyword",
    ["dual_time", "max_steps", "retry", "shift", "positivity_floor", "homotopy", "preconditioner"],
)
def test_a_script_cannot_pass_a_setting_even_one_the_file_leaves_unset(recorded, keyword) -> None:
    with pytest.raises(TypeError, match=rf"CoupledMarch: {keyword} is a setting of the case"):
        CoupledMarch().solve("problem", **{keyword: object()})
    assert recorded.solve_coupled.calls == []


def test_a_flow_march_refuses_a_setting_passed_as_an_observer(recorded) -> None:
    with pytest.raises(TypeError, match=r"FlowMarch: step_control is a setting of the case"):
        FlowMarch().solve("problem", step_control=object())
    FlowMarch(max_steps=4).solve("problem", on_step="observer")
    ((args, kwargs),) = recorded.solve_flow_march.calls
    assert args == ("problem",) and kwargs == {"max_steps": 4, "on_step": "observer"}


def test_a_point_setup_may_observe_a_ramps_anchor_but_not_configure_it(recorded) -> None:
    seen = []
    march = CoupledMarch(continuation=ViscosityRamp(anchor=10.0, stations=2, steps_per_station=1))
    march.solve("problem", point_setup=lambda *arguments: seen.append(arguments))
    ((_, kwargs),) = recorded.solve_reynolds_ramp.calls
    assert kwargs["point_setup"]("companion", "state", "point") == {}
    assert seen == [("companion", "state", "point")]

    march.solve("problem", point_setup=lambda *_: {"step_control": object()})
    _, (_, kwargs) = recorded.solve_reynolds_ramp.calls
    with pytest.raises(
        TypeError, match=r"may only observe, but it returned settings \['step_control'\]"
    ):
        kwargs["point_setup"]("companion", "state", "point")

    with pytest.raises(TypeError, match=r"this march has none"):
        CoupledMarch().solve("problem", point_setup=lambda *_: {})


@pytest.mark.parametrize("held", [True, False], ids=["bulk-velocity", "boundary-driven"])
def test_a_segregated_solve_picks_its_flow_solve_by_the_drive_and_hands_on_its_settings(
    recorded, held
) -> None:
    momentum = types.SimpleNamespace(drive=MassFlow(target=1.0, flow_direction=0) if held else None)
    problem = types.SimpleNamespace(momentum=momentum, turbulence="turbulence")
    segregated = Segregated(
        sweeps=12,
        relaxation=0.9,
        flow_solve=RootSolve(linear_solver=DirectSolve()),
        scalar_solve=RootSolve(max_steps=400, convergence=Convergence(rtol=1e-8, atol=1e-10)),
        scalar_preconditioner=ScalarAir(),
    )
    assert segregated.solve(problem) == "<solve_segregated>"
    chosen, other = (
        (recorded.bulk_velocity_flow_solve, recorded.reused_flow_solve)
        if held
        else (recorded.reused_flow_solve, recorded.bulk_velocity_flow_solve)
    )
    assert other.calls == []
    assert chosen.calls == [
        (
            (momentum,),
            {"root_solve": RootSolveSettings(linear_solver=lx.AutoLinearSolver(well_posed=True))},
        )
    ]
    assert recorded.scalar_pseudo_transient_solve.calls == [
        (
            (),
            {
                "root_solve": RootSolveSettings(
                    max_steps=400, convergence=Convergence(rtol=1e-8, atol=1e-10)
                )
            },
        )
    ]
    assert recorded.sst_initial_fields.calls == [((momentum, "turbulence"), {})]
    ((args, kwargs),) = recorded.solve_segregated.calls
    assert args == (
        momentum,
        "turbulence",
        chosen.result,
        "<scalar_pseudo_transient_solve>",
        "flow0",
        "k0",
        "omega0",
    )
    assert kwargs == {"max_sweeps": 12, "relaxation": 0.9, "scalar_preconditioner": ScalarAir()}


def test_a_segregated_solve_left_at_its_defaults_passes_none_of_them(recorded) -> None:
    momentum = types.SimpleNamespace(drive=None)
    Segregated(sweeps=5).solve(types.SimpleNamespace(momentum=momentum, turbulence="t"))
    assert recorded.reused_flow_solve.calls == [((momentum,), {})]
    assert recorded.scalar_pseudo_transient_solve.calls == [((), {})]
    ((_, kwargs),) = recorded.solve_segregated.calls
    assert kwargs == {"max_sweeps": 5}
    with pytest.raises(TypeError, match=r"Segregated takes no observers"):
        Segregated(sweeps=5).solve(
            types.SimpleNamespace(momentum=momentum, turbulence="t"), on_step=1
        )


def test_a_root_solve_states_every_setting_of_the_librarys_root_solve() -> None:
    """``RootSolve`` is the file's form of ``RootSolveSettings``; a field added to one must reach the other."""
    names = [field.name for field in dataclasses.fields(RootSolve)]
    assert sorted(names) == sorted(field.name for field in dataclasses.fields(RootSolveSettings))
    settings = RootSolve(
        max_steps=3,
        convergence=Convergence(atol=1e-9),
        linear_solver=DirectSolve(),
        adjoint_solver=GmresSolve(1e-6, restart=30),
    ).root_solve_settings()
    assert settings == RootSolveSettings(
        max_steps=3,
        convergence=Convergence(atol=1e-9),
        linear_solver=lx.AutoLinearSolver(well_posed=True),
        adjoint_solver=relative_residual_gmres(1e-6, restart=30),
    )


def test_a_linear_solver_spec_builds_the_solver_it_names() -> None:
    assert GmresSolve(1e-4, restart=40).build() == relative_residual_gmres(1e-4, restart=40)
    assert GmresSolve(1e-4, stagnation_iters=7, max_restarts=3).build() == relative_residual_gmres(
        1e-4, stagnation_iters=7, max_restarts=3
    )
    assert GmresSolve(1e-4).build() != GmresSolve(1e-3).build()
    assert DirectSolve().build() == lx.AutoLinearSolver(well_posed=True)


# --- the validation cases' files state what their scripts used to pass -------------------------


def _pitzdaily_march_as_its_script_passed_it() -> tuple[dict, dict]:
    """``validation/pitzdaily_openfoam/compare.py``'s march at its defaults, before it read its file.

    A frozen copy: the script now reads these from the file, so it cannot be the reference. Returned as
    the solve's settings and the ramp's own.
    """
    settings = {
        "max_steps": 150,
        "convergence": Convergence(rtol=0.0, atol=1e-5),
        "preconditioner": MaterializedJacobian(
            FieldSplit(
                SimpleSmoothed(
                    sweeps=2,
                    pressure_sweeps=2,
                    strength_threshold=0.25,
                    avoid_singletons=True,
                    aggressive_levels=0,
                    max_levels=5,
                    max_coarse=500,
                    block_splitting=True,
                    omega=1.0,
                ),
                JacobiSmoothed(max_coarse=2000, equilibrate=False),
            ),
            probe=JacobianProbeSpec(stencil_reach=3, column_reach=None, gradient_sweeps=None),
            refit_beta_floor=0.05,
        ),
        "dual_time": DualTimeLoop(
            inner_steps=5, inner_tol=1e-2, cycle_budget=42, refresh_on_cycles=3
        ),
        "linear_solve": LinearSolveSettings(rtol=0.3, restart=15, max_restarts=14),
        "positivity_floor": 0.0,
        "positivity_projection": True,
        "step_control": CflResidualDualTimeControl(
            beta_start=0.5,
            beta_min=0.005,
            grow=1.5,
            backoff=2.0,
            grow_above=0.5,
            backoff_below=0.25,
        ),
        "retry": RetryPolicy(
            solver=GmresSolve(1e-4, restart=40),
            abort_above_cycles=10,
            on_alpha=0.01,
            beta_factor=2.0,
        ),
        "shift": CoupledShiftSettings(turbulence_damping=3.0),
    }
    ramp = {
        "anchor": 10.0**2,
        "stations": 16,
        "steps_per_station": 1,
        "companion": scale_momentum_only,
    }
    return settings, ramp


def _bfs3d_march_as_its_script_passed_it() -> tuple[dict, dict]:
    """``validation/bfs3d_openfoam/compare.py``'s march at its defaults -- see the pitzDaily twin."""
    settings = {
        "max_steps": 150,
        "convergence": Convergence(rtol=0.0, atol=1e-5),
        "preconditioner": MaterializedJacobian(
            FieldSplit(
                SimpleSmoothed(
                    sweeps=2,
                    pressure_sweeps=2,
                    strength_threshold=0.25,
                    avoid_singletons=True,
                    aggressive_levels=0,
                    max_levels=5,
                    max_coarse=500,
                    block_splitting=True,
                    omega=1.0,
                    frozen_coarsening=True,
                ),
                JacobiSmoothed(
                    max_levels=20,
                    max_coarse=200,
                    strength_threshold=0.25,
                    aggressive_levels=0,
                    frozen_coarsening=True,
                    equilibrate=False,
                    avoid_singletons=False,
                ),
            ),
            probe=JacobianProbeSpec(column_reach=(3, 3, 3, 3, 2, 2)),
            refit_beta_floor=0.05,
        ),
        "dual_time": DualTimeLoop(
            inner_steps=5, inner_tol=1e-2, cycle_budget=42, refresh_on_cycles=3
        ),
        "linear_solve": LinearSolveSettings(rtol=0.3, restart=15, max_restarts=14),
        "positivity_floor": 1e-8,
        "positivity_projection": False,
        "step_control": CflResidualDualTimeControl(
            beta_start=0.5,
            beta_min=0.005,
            grow=1.5,
            backoff=2.0,
            grow_above=0.5,
            backoff_below=0.25,
        ),
        "retry": RetryPolicy(
            solver=GmresSolve(1e-4, restart=40),
            abort_above_cycles=10,
            on_alpha=0.01,
            beta_factor=2.0,
        ),
        "shift": CoupledShiftSettings(turbulence_damping=5.0),
    }
    ramp = {
        "anchor": 10.0**2,
        "stations": 12,
        "steps_per_station": 1,
        "companion": scale_momentum_only,
    }
    return settings, ramp


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pitzdaily_openfoam/case.yaml", _pitzdaily_march_as_its_script_passed_it),
        ("bfs3d_openfoam/case.yaml", _bfs3d_march_as_its_script_passed_it),
    ],
    ids=["pitzdaily", "bfs3d"],
)
def test_each_step_case_file_hands_its_ramp_the_march_its_script_used_to_pass(
    recorded, path, expected
) -> None:
    solver = read_case(REPO / "validation" / path).spec.solver
    settings, ramp = expected()
    assert solver.settings() == settings
    # The preconditioner a script opened as a session is opened from the file's spec here.
    solver.solve("problem")
    ((args, _),) = recorded.open_session.calls
    assert args == (settings["preconditioner"], "problem")
    ((_, ramp_kwargs),) = recorded.solve_reynolds_ramp.calls
    ramp_kwargs.pop("point_setup")
    assert ramp_kwargs == ramp | settings | {"preconditioner": "<open_session>"}


#: Each channel file's solve as its script passed it before it read the file: the sweep budget, the
#: flow solve's direct factorization, the scalar solve's budget and stop, and the scalar preconditioner.
_CHANNEL_SOLVES = [
    ("turbulent_channel/cases/re20000.yaml", 100, 400, ScalarAir()),
    ("turbulent_channel/cases/re45000.yaml", 110, 400, ScalarAir()),
    ("turbulent_channel/cases/re240000.yaml", 150, 400, ScalarAir()),
    ("turbulent_channel_openfoam/cases/low.yaml", 90, 200, None),
    ("turbulent_channel_openfoam/cases/high.yaml", 140, 200, None),
]


@pytest.mark.parametrize(
    ("path", "sweeps", "scalar_steps", "scalar_preconditioner"),
    _CHANNEL_SOLVES,
    ids=[Path(c[0]).parent.parent.name + "/" + Path(c[0]).stem for c in _CHANNEL_SOLVES],
)
def test_each_channel_case_file_hands_its_segregated_solve_what_its_script_used_to_pass(
    recorded, path, sweeps, scalar_steps, scalar_preconditioner
) -> None:
    solver = read_case(REPO / "validation" / path).spec.solver
    momentum = types.SimpleNamespace(drive=MassFlow(target=1.0, flow_direction=0))
    solver.solve(types.SimpleNamespace(momentum=momentum, turbulence="turbulence"))
    assert recorded.bulk_velocity_flow_solve.calls == [
        (
            (momentum,),
            {"root_solve": RootSolveSettings(linear_solver=lx.AutoLinearSolver(well_posed=True))},
        )
    ]
    assert recorded.scalar_pseudo_transient_solve.calls == [
        (
            (),
            {
                "root_solve": RootSolveSettings(
                    max_steps=scalar_steps, convergence=Convergence(rtol=1e-8, atol=1e-10)
                )
            },
        )
    ]
    ((_, kwargs),) = recorded.solve_segregated.calls
    expected = {"max_sweeps": sweeps, "relaxation": 0.9}
    if scalar_preconditioner is not None:
        expected["scalar_preconditioner"] = scalar_preconditioner
    assert kwargs == expected


@pytest.mark.parametrize(
    ("case", "check"),
    [
        ("pitzdaily_openfoam", "compare.SOLVER == compare.CASE.spec.solver"),
        ("bfs3d_openfoam", "compare.SOLVER == compare.FILE_SOLVER"),
    ],
)
def test_each_step_case_script_runs_its_files_solver_when_no_override_is_set(case, check) -> None:
    """With no study variable set, the solve a script assembles is its case file's, exactly.

    Each script reads its march settings back from the file into the constants its probes import, and
    the bfs3d one reassembles them into the solver it runs. A constant that stopped reading the file --
    a literal restored, say -- would leave the scripts running a march the file does not describe, so
    each is imported in an environment without the variables that override it.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PITZ_", "BFS3D_"))
    }
    result = subprocess.run(
        [sys.executable, "-c", f"import compare; assert {check}"],
        cwd=REPO / "validation" / case,
        env=environment | {"PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
