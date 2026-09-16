"""One globalization, six builders: every march can be given every setting that describes how it damps.

:class:`~aquaflux.solve.PseudoTransientStep` is the one shifted-march engine in the package, and six
public builders configure it -- the flow block's, the scalar transport's, and the four coupled ones.
Until 2026-09-13 each spelled its share of the engine's surface out for itself, and the shares had
drifted apart: the coupled four carried eight keywords apiece, the two written first carried two, so
the shift floor, the line search, the growth rungs and the growth rule could not be reached from the
flow-only or scalar paths at all (issue #372). Nothing about those settings is a property of
turbulence or of coupling -- they describe how a march damps -- so the gap was drift, not design, and
it was invisible to review because no single builder looked wrong.

**What this does NOT claim.** The problem-specific step fields -- ``step_limit``, ``step_projection``,
``krylov_solver``, ``residual_norm``, ``jacobian_residual`` -- are not settings of the globalization,
and the flow-only and scalar builders still do not expose them. Nor can any builder be given a
non-default acceptance rule or relaxation schedule: those are flattened to ``divergence_cap`` and
``beta0``/``exponent``/``beta_floor``.

These tests pin the repair from four sides. Every builder **takes** the object. A non-default object
**arrives** on the built step field for field -- the half a signature check cannot see. The shipped
defaults are pinned as **literal numbers**, so a moved default fails here instead of silently moving
every case. And one **override changes one setting** on every builder, with each refusal the object
makes where it would otherwise drop a setting without a word.

``coupled_step`` is built here with the block-diagonal and complete-LU preconditioners. A multigrid
V-cycle needs ``petsc4py``, which CI does not install, and it shares ``_monolithic_factor_step`` with the
complete LU, so its forwarding is the same code path.
"""

from __future__ import annotations

import dataclasses
import inspect

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import pytest
from aquaflux.flow import momentum_continuation, reused_flow_solve
from aquaflux.flow.block_preconditioner import BlockPreconditioner
from aquaflux.solve import (
    DEFAULT_GLOBALIZATION,
    DivergenceGuard,
    DualTimeLoop,
    DualTimeStep,
    Globalization,
    MonotoneLineSearch,
    PseudoTransientStep,
    RelaxedFarFromRoot,
    SwitchedEvolutionRelaxation,
)
from aquaflux.turbulence import (
    BlockDiagonal,
    CompleteLu,
    MaterializedJacobian,
    ScalarShiftPolicy,
    coupled_step,
    open_session,
    scalar_pseudo_transient_solve,
)
from aquaflux.turbulence.coupled import mass_flow_coupled_continuation

from tests.unit.test_coupled_rans import _cavity, _healthy_state

#: Every field set, each to a value no builder defaults to, so a setting that fails to arrive shows up
#: as its default rather than coinciding with what was asked for.
ASKED = Globalization(
    beta0=1.25,
    exponent=0.75,
    beta_floor=0.03,
    max_escalations=3,
    escalation_factor=3.5,
    divergence_cap=25.0,
    line_search=7,
    grow=2,
    line_search_growth=RelaxedFarFromRoot(),
)

#: The same, cut to the fields a dual-time step has: its inner loop has no escalation ladder.
DUAL_TIME_ASKED = Globalization(beta0=1.25, exponent=0.75, beta_floor=0.03, line_search=7)

#: The shipped defaults, written as LITERALS rather than read back from any object -- the only way a test
#: can notice that a default moved. ``line_search`` is the flow-only and scalar value; the coupled
#: builders' base is ``COUPLED_LINE_SEARCH``.
SHIPPED = Globalization(
    beta0=2.0,
    exponent=1.0,
    beta_floor=0.0,
    max_escalations=6,
    escalation_factor=2.0,
    divergence_cap=10.0,
    line_search=0,
    grow=0,
    line_search_growth=MonotoneLineSearch(),
)
COUPLED_LINE_SEARCH = 10

BUILDERS = (
    momentum_continuation,
    reused_flow_solve,
    scalar_pseudo_transient_solve,
    coupled_step,
    mass_flow_coupled_continuation,
)


@pytest.fixture(scope="module")
def case():
    """A small coupled case and a well-positive state, built once for the whole module."""
    mesh, coupled = _cavity(4)
    return coupled, _healthy_state(mesh, coupled)


@dataclasses.dataclass(frozen=True)
class _IdentityPreconditioner:
    """A preconditioner that does nothing, so a scalar policy's two branches can be told apart."""

    def __call__(self, phi):
        return lambda residual: residual


def _assert_carries(step, expected: Globalization, *, dual_time: bool = False) -> None:
    """The step was configured with ``expected``'s values -- field for field, not merely accepted."""
    schedule = step.relaxation_schedule
    assert isinstance(schedule, SwitchedEvolutionRelaxation)
    assert (schedule.beta0, schedule.exponent, schedule.beta_floor) == (
        expected.beta0,
        expected.exponent,
        expected.beta_floor,
    )
    assert step.line_search == expected.line_search
    if dual_time:
        # The inner Newton loop replaces the escalation ladder, so the ladder's settings have no field to
        # land on; `Globalization.dual_time_step` refuses them, which a test below pins.
        return
    assert step.max_escalations == expected.max_escalations
    assert step.escalation_factor == expected.escalation_factor
    assert isinstance(step.acceptance, DivergenceGuard)
    assert step.acceptance.divergence_cap == expected.divergence_cap
    assert step.grow == expected.grow
    assert step.line_search_growth == expected.line_search_growth


def test_every_builder_of_the_shifted_march_takes_the_globalization() -> None:
    """All six -- and ``reused_flow_solve``, which forwards to one of them.

    A signature check is worth having on its own because it covers the builder whose construction needs
    an optional dependency, and because the failure it catches is a keyword added to one builder and not
    its siblings, which is how this went wrong before.
    """
    for builder in BUILDERS:
        parameters = inspect.signature(builder).parameters
        if builder is reused_flow_solve:
            # It builds a `momentum_continuation` behind `**build_kwargs`, so the keyword rides rather
            # than being declared. Pinned so the forwarding cannot quietly narrow.
            assert "build_kwargs" in parameters
            continue
        assert "globalization" in parameters, f"{builder.__name__} cannot be given a globalization"


def test_every_builder_defaults_to_nothing_overridden() -> None:
    """The default is the empty override, so no builder carries a second copy of any default.

    An object holding a full default set restates every default beside the step class that uses it, and
    a builder whose default differs then needs a second preset -- which is what the first version had,
    and a one-field override of the wrong preset silently reset every other field.
    """
    assert all(
        getattr(DEFAULT_GLOBALIZATION, field.name) is None
        for field in dataclasses.fields(Globalization)
    )
    for builder in BUILDERS:
        if builder is reused_flow_solve:
            continue
        got = inspect.signature(builder).parameters["globalization"].default
        assert got is DEFAULT_GLOBALIZATION, f"{builder.__name__} defaults to {got!r}"


def test_the_shipped_defaults_are_the_ones_every_case_was_measured_under(case) -> None:
    """Pinned as literal numbers, on both step shapes, because nothing else in the suite would notice.

    Every case in the repository marches under these values and passes none of them explicitly, so a
    default that moves moves every case at once -- and a test comparing a step against the object that
    configured it is blind to that by construction.
    """
    coupled, state = case
    _assert_carries(momentum_continuation(coupled.momentum), SHIPPED)
    coupled_shipped = dataclasses.replace(SHIPPED, line_search=COUPLED_LINE_SEARCH)
    _assert_carries(
        coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None)), coupled_shipped
    )
    _assert_carries(
        coupled_step(
            coupled,
            state,
            preconditioner=BlockDiagonal(method=None),
            dual_time=DualTimeLoop(inner_steps=3),
        ),
        coupled_shipped,
        dual_time=True,
    )


def test_one_override_changes_one_setting_and_each_builder_keeps_its_own_base(case) -> None:
    """``Globalization(beta0=1.5)`` means the same thing on every builder: that setting, and nothing else.

    The coupled march's line search is the one default that differs between builders. With a full
    default set on the object, overriding ``beta0`` alone reset it to zero on a coupled builder -- the
    difference, on that residual, between a march that descends and one that stalls at its initial
    residual. And an explicit ``line_search=0`` has to survive the base, or a caller cannot opt out.
    """
    coupled, state = case
    one = Globalization(beta0=1.5)
    _assert_carries(
        momentum_continuation(coupled.momentum, globalization=one),
        dataclasses.replace(SHIPPED, beta0=1.5),
    )
    _assert_carries(
        coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None), globalization=one),
        dataclasses.replace(SHIPPED, beta0=1.5, line_search=COUPLED_LINE_SEARCH),
    )
    opted_out = coupled_step(
        coupled,
        state,
        preconditioner=BlockDiagonal(method=None),
        globalization=Globalization(line_search=0),
    )
    assert opted_out.line_search == 0


def test_the_flow_block_builder_forwards_every_field(case) -> None:
    """``momentum_continuation`` reached ``escalation_factor`` and ``max_escalations`` and nothing else.

    Its narrow surface was not a judgement that a flow-only march cannot use a line search -- the
    argument for backtracking a shifted correction never mentions the residual being solved. It was the
    surface the builder happened to be written with.
    """
    coupled, _ = case
    step = momentum_continuation(coupled.momentum, globalization=ASKED)
    assert isinstance(step, PseudoTransientStep)
    _assert_carries(step, ASKED)
    # The preconditioner it builds still reaches the adjoint solve, which is the builder's own job.
    assert step.adjoint_preconditioner_factory is not None


def test_the_scalar_builder_forwards_every_field_on_both_of_its_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Including the ``policy is None`` fallback, which used to be a second construction site.

    The two branches differ only in the shift policy and whether there is an adjoint factory; written
    out twice they were free to be configured differently, and a fallback that silently ran a different
    globalization from the preconditioned path would show up as a march that behaves differently when
    the multigrid is switched off -- which reads as a preconditioner result.
    """
    seen: list[PseudoTransientStep] = []
    solve_scalar = scalar_pseudo_transient_solve(globalization=ASKED, max_steps=20)

    def residual(phi):
        return phi - 1.0

    # The step is built inside the jitted solve, so it is captured on the way past rather than
    # returned: what is under test is the object the solve actually runs, not a reconstruction of it.
    original = PseudoTransientStep.stepper

    def recording(self):
        seen.append(self)
        return original(self)

    monkeypatch.setattr(PseudoTransientStep, "stepper", recording)
    for policy in (ScalarShiftPolicy(jnp.zeros(3), _IdentityPreconditioner()), None):
        solved = solve_scalar(residual, jnp.full((3,), 2.0), policy)
        assert jnp.allclose(solved, 1.0, atol=1e-8)

    assert len(seen) == 2, "both branches should have built a step"
    for step in seen:
        _assert_carries(step, ASKED)
    # Which step came from which branch: only the preconditioned one carries an adjoint factory.
    assert sorted(step.adjoint_preconditioner_factory is None for step in seen) == [False, True]


@pytest.mark.parametrize("dual_time", [False, True])
def test_the_coupled_builders_forward_every_field(case, dual_time: bool) -> None:
    """Both step shapes, since the dual-time branch is a separate construction.

    A setting wired onto the single-step branch and missed on the dual-time one would be inert on exactly
    the marches that matter -- both flagship cases run ``inner_steps > 1``.
    """
    coupled, state = case
    asked = DUAL_TIME_ASKED if dual_time else ASKED
    extra = {"dual_time": DualTimeLoop(inner_steps=3, inner_tol=1e-3)} if dual_time else {}
    built = {
        "block": coupled_step(
            coupled, state, preconditioner=BlockDiagonal(method=None), globalization=asked, **extra
        ),
        "lu": coupled_step(
            coupled,
            state,
            preconditioner=MaterializedJacobian(CompleteLu(backend="scipy")),
            globalization=asked,
            **extra,
        ),
        "mass flow": mass_flow_coupled_continuation(
            coupled, state, preconditioner=BlockDiagonal(method=None), globalization=asked, **extra
        ),
    }
    for name, step in built.items():
        assert isinstance(step, DualTimeStep if dual_time else PseudoTransientStep), name
        _assert_carries(step, asked, dual_time=dual_time)


@pytest.mark.parametrize(
    "field",
    ["max_escalations", "escalation_factor", "divergence_cap", "grow", "line_search_growth"],
)
def test_a_dual_time_step_refuses_a_setting_it_has_no_field_for(case, field: str) -> None:
    """Refused rather than dropped: a setting that reaches nothing looks exactly like no setting."""
    coupled, state = case
    one = Globalization(**{field: getattr(ASKED, field)})
    with pytest.raises(ValueError, match=field):
        coupled_step(
            coupled,
            state,
            preconditioner=BlockDiagonal(method=None),
            globalization=one,
            dual_time=DualTimeLoop(inner_steps=3),
        )
    # ...and the same object is accepted by the single-step shape, which has the ladder.
    coupled_step(coupled, state, preconditioner=BlockDiagonal(method=None), globalization=one)


def test_a_step_field_the_step_does_not_declare_is_refused_even_as_none() -> None:
    """Names are checked before unset values are dropped.

    Filtering first made a misplaced setting vanish exactly when it was ``None``: a field only the
    single-step class declares, handed to the dual-time one, disappeared without a word.
    """
    policy = ScalarShiftPolicy(jnp.zeros(3))
    with pytest.raises(TypeError, match="acceptance"):
        Globalization().dual_time_step(policy, acceptance=None)
    with pytest.raises(TypeError, match="inner_steps"):
        Globalization().step(policy, inner_steps=None)


def test_a_setting_given_both_ways_is_refused_and_an_unset_one_is_the_caller_s() -> None:
    """Neither side silently wins -- which one would is an accident of how a dict merge was written."""
    policy = ScalarShiftPolicy(jnp.zeros(3))
    with pytest.raises(TypeError, match="relaxation_schedule"):
        Globalization(beta0=1.5).step(policy, relaxation_schedule=SwitchedEvolutionRelaxation())
    # Left unset on the object, the step field is the caller's to supply.
    step = Globalization().step(policy, relaxation_schedule=SwitchedEvolutionRelaxation(beta0=3.0))
    assert step.relaxation_schedule.beta0 == 3.0


def test_a_refresh_refuses_a_keyword_the_flow_block_does_not_take(case) -> None:
    """The regression: moving the march's settings onto ``Globalization`` made a stale ``beta0=`` silent.

    A refresh carries the flow block rather than rebuilding it, so when leftover keywords went to the
    flow-block builder a ``beta0=1.5`` handed to a refresh built a march at ``beta0`` 2.0 without a word,
    while the same call on a first build raised. A preconditioner session binds its march keywords
    against one signature on every build and every refresh, so both raise, and say where it belongs.
    """
    coupled, state = case
    session = open_session(BlockDiagonal(method=None), coupled)
    base = session.build(state)
    with pytest.raises(TypeError, match=r"beta0.*Globalization"):
        session.build(state, beta0=1.5)
    with pytest.raises(TypeError, match=r"beta0.*Globalization"):
        session.refresh(state, base, base.norm(), beta0=1.5)
    # A flow-block setting cannot be misplaced at all: the spec's fields are that builder's keywords,
    # which is exact only while `build` declares every option it accepts.
    kinds = {
        parameter.kind
        for parameter in inspect.signature(BlockPreconditioner.build).parameters.values()
    }
    assert inspect.Parameter.VAR_KEYWORD not in kinds
