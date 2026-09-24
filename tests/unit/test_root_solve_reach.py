"""One value, three builders: every root solve can be given every setting of the solve around its step.

:class:`~aquaflux.solve.RootSolver` is the one Newton driver in this package, and three public builders
construct one -- the reused flow solve, the bulk-velocity-constrained flow solve, and the scalar
transport's pseudo-transient solve. Until 2026-09-22 each spelled its own share of the driver's surface
out as keywords, and the shares had drifted apart: all three took ``max_steps``, two took a forward
linear solver, one took ``rtol``/``atol``, and **none** could reach ``adjoint_solver`` -- the setting
whose absence makes a transpose solve on a hard case raise an error naming a remedy the entry point
cannot reach (issue #428). Nothing about those settings is a property of a flow block or of a transported
scalar; they describe how a Newton solve is run, so the gap was drift, not design, and it was invisible
to review because no single builder looked wrong.

**What this does NOT claim.** ``strategy`` is the step each builder exists to construct, and ``measures``
is what a problem's residual can be measured in -- neither is a setting of the solve, and neither is on
the value. The scalar builder is forward-only, so its ``adjoint_solver`` reaches the driver and is never
used; it is reachable rather than useful there.

These tests pin the repair from four sides. Every builder **takes** the object. A non-default object
**arrives** on the built solver field for field -- the half a signature check cannot see. The shipped
step caps are pinned as **literal numbers**, so a moved default fails here instead of silently moving
every case that runs one of these solves. And the refusals the value makes are pinned where it would
otherwise drop a setting without a word.

The builders are driven with the recorder below rather than by running their solves: what is under test
is the solver object each builder constructs, and constructing it is the whole of what these settings
decide.
"""

from __future__ import annotations

import dataclasses
import inspect

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import lineax as lx
import pytest
from aquaflux.flow import bulk_velocity_flow_solve, reused_flow_solve
from aquaflux.solve import (
    DEFAULT_ROOT_SOLVE,
    PLAIN_RESIDUAL,
    Convergence,
    DampedNewtonStep,
    Euclidean,
    RootSolver,
    RootSolveSettings,
)
from aquaflux.turbulence import ScalarShiftPolicy, scalar_pseudo_transient_solve

from tests.unit.test_coupled_rans import _cavity, _mass_flow_cavity

#: Every field set, each to a value no builder defaults to, so a setting that fails to arrive shows up as
#: its default rather than coinciding with what was asked for.
ASKED = RootSolveSettings(
    convergence=Convergence(measure=Euclidean(), rtol=1e-3, atol=1e-7),
    max_steps=123,
    linear_solver=lx.GMRES(rtol=0.5, atol=0.5),
    adjoint_solver=lx.GMRES(rtol=0.25, atol=0.25),
)

#: The shipped step caps, written as LITERALS rather than read back from any object -- the only way a
#: test can notice that one moved. The tolerances are deliberately absent: no builder sets them, which is
#: itself pinned below.
SHIPPED_MAX_STEPS = {
    reused_flow_solve: 80,
    bulk_velocity_flow_solve: 20,
    scalar_pseudo_transient_solve: 40,
}

BUILDERS = tuple(SHIPPED_MAX_STEPS)


class _Stop(Exception):
    """Raised by the recorder once a builder has constructed its solver, so no solve is run."""


@dataclasses.dataclass(frozen=True)
class _Measures:
    """A residual-measure source that is not the default, so its arrival is visible."""

    def row_scaled(self, step, state):
        raise AssertionError("not reached")

    def block_scaled(self, state):
        raise AssertionError("not reached")


@pytest.fixture(scope="module")
def momentum():
    """A small flow assembler, built once for the whole module."""
    _, coupled = _cavity(4)
    return coupled.momentum


@pytest.fixture(scope="module")
def mass_flow_momentum():
    """The same assembler, driven by a constrained body force -- what the bordered builder needs."""
    _, coupled = _mass_flow_cavity(4)
    return coupled.momentum


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[RootSolver]:
    """The solvers the builders construct, recorded on the way past.

    The real :meth:`RootSolveSettings.solver` runs first, so what is recorded is the object a solve would
    have used, not a reconstruction of it.
    """
    recorded: list[RootSolver] = []
    original = RootSolveSettings.solver

    def recording(self, strategy, **fields):
        recorded.append(original(self, strategy, **fields))
        raise _Stop

    monkeypatch.setattr(RootSolveSettings, "solver", recording)
    return recorded


def _drive(builder, momentum, mass_flow_momentum, settings: RootSolveSettings) -> None:
    """Take ``builder`` as far as constructing its solver, whichever side of the closure that happens on."""
    with pytest.raises(_Stop):
        if builder is reused_flow_solve:
            reused_flow_solve(momentum, root_solve=settings)
        elif builder is bulk_velocity_flow_solve:
            solve = bulk_velocity_flow_solve(mass_flow_momentum, root_solve=settings)
            solve(mass_flow_momentum, mass_flow_momentum.initial_state())
        else:
            solve_scalar = scalar_pseudo_transient_solve(root_solve=settings)
            state = jnp.full((3,), 2.0)
            solve_scalar(lambda phi: phi - 1.0, state, ScalarShiftPolicy(jnp.zeros(3)))


def test_every_builder_of_a_root_solve_takes_the_settings() -> None:
    """A signature check on its own, because the failure it catches is a keyword added to one builder
    and not its siblings, which is how this went wrong before."""
    for builder in BUILDERS:
        parameters = inspect.signature(builder).parameters
        assert "root_solve" in parameters, f"{builder.__name__} cannot be given root-solve settings"


def test_every_builder_defaults_to_nothing_overridden() -> None:
    """The default is the empty override, so no builder carries a second copy of any default.

    A value holding a full default set restates every default beside the solver that uses it, and a
    one-field override of it then silently resets every other field to that copy.
    """
    assert all(
        getattr(DEFAULT_ROOT_SOLVE, field.name) is None
        for field in dataclasses.fields(RootSolveSettings)
    )
    for builder in BUILDERS:
        got = inspect.signature(builder).parameters["root_solve"].default
        assert got is DEFAULT_ROOT_SOLVE, f"{builder.__name__} defaults to {got!r}"


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_every_builder_forwards_every_field(builder, momentum, mass_flow_momentum, built) -> None:
    """Field for field on the solver each builder builds -- including ``adjoint_solver``, which is the
    setting none of the three could reach."""
    _drive(builder, momentum, mass_flow_momentum, ASKED)

    (solver,) = built
    assert solver.max_steps == ASKED.max_steps
    assert solver.convergence == ASKED.convergence
    assert solver.linear_solver is ASKED.linear_solver
    assert solver.adjoint_solver is ASKED.adjoint_solver


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_the_shipped_defaults_are_the_ones_every_case_runs_under(
    builder, momentum, mass_flow_momentum, built
) -> None:
    """Pinned as literal numbers, because nothing else in the suite would notice one moving.

    Every case in the repository solves under these values and passes none of them explicitly, so a
    default that moves moves every case at once -- and a test comparing a solver against the value that
    configured it is blind to that by construction.
    """
    _drive(builder, momentum, mass_flow_momentum, DEFAULT_ROOT_SOLVE)

    (solver,) = built
    assert solver.max_steps == SHIPPED_MAX_STEPS[builder]
    # No builder sets a tolerance or a linear solver: the stopping test is the solver's own and the
    # forward solve is the strategy's inexact-Newton default, each declared in one place.
    assert solver.convergence == Convergence()
    assert solver.linear_solver is None
    assert solver.adjoint_solver is None
    assert solver.measures is PLAIN_RESIDUAL


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_one_override_changes_one_setting_and_each_builder_keeps_its_own_cap(
    builder, momentum, mass_flow_momentum, built
) -> None:
    """``RootSolveSettings(adjoint_solver=...)`` means the same thing on every builder: that setting,
    and nothing else. The step cap is the one default that differs between them."""
    adjoint = lx.GMRES(rtol=0.125, atol=0.125)
    _drive(builder, momentum, mass_flow_momentum, RootSolveSettings(adjoint_solver=adjoint))

    (solver,) = built
    assert solver.adjoint_solver is adjoint
    assert solver.max_steps == SHIPPED_MAX_STEPS[builder]
    assert solver.convergence == Convergence()


def test_the_settings_reach_the_solver_and_a_problem_field_rides_alongside() -> None:
    """:meth:`RootSolveSettings.solver` is the one place a value becomes a solver.

    ``measures`` is not a setting of the solve -- it says what the *problem's* residual can be measured
    in -- so it arrives as a field beside the value rather than on it.
    """
    step = DampedNewtonStep()
    measures = _Measures()
    solver = ASKED.solver(step, measures=measures)
    assert solver.strategy is step
    assert solver.measures is measures
    assert solver.max_steps == ASKED.max_steps
    # Left out, a field keeps the solver's own default rather than becoming `None`.
    assert RootSolveSettings().solver(step).measures is PLAIN_RESIDUAL


def test_a_field_the_solver_does_not_declare_is_refused_even_as_none() -> None:
    """Names are checked before unset values are dropped, so a misplaced setting cannot vanish exactly
    when it is ``None``."""
    step = DampedNewtonStep()
    with pytest.raises(TypeError, match="no field named nonsense"):
        RootSolveSettings().solver(step, nonsense=None)


def test_a_setting_given_both_ways_is_refused() -> None:
    """Neither side silently wins -- which one would is an accident of how a dict merge was written."""
    step = DampedNewtonStep()
    with pytest.raises(TypeError, match="max_steps"):
        RootSolveSettings(max_steps=3).solver(step, max_steps=4)
