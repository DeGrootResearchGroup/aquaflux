"""Pre-screen candidate step-control policies against the archived marches, without solving anything.

A march of this case costs 35-50 minutes, so a step-control change that has to be judged by running one
is, in practice, never judged: candidates accumulate faster than they can be measured. But the *rule*
that sets the shift strength β is pure arithmetic over three recorded numbers per step -- the accepted
step length α, the steady residual, and how many times the escalation ladder redid the step -- and all
three are in the logs. Replaying the rule over an archived log therefore reconstructs its whole β
trajectory in milliseconds, and swapping the rule for a candidate reconstructs what that candidate would
have produced from the *same* recorded inputs.

**The mechanism being modelled.** Each outer step of the march does three things in order:

1. The control turns the previous step's ``(α, residual)`` into this step's β. The shipped control grows
   the pseudo-timestep (divides β by ``grow``) only when the step length was comfortable *and* the
   residual is not rising, brakes (multiplies by ``backoff``) when either wall is hit, and holds in
   between.
2. If the step comes back with a collapsed step length or a runaway solve cost and it missed its own
   stopping criterion, an escalation ladder redoes it at ``β × retry.beta_factor``, up to a fixed number
   of rungs. The ladder multiplies the shift leaf directly, so it is **not** subject to the control's
   ``beta_max`` clamp -- an archived march reaches β = 4.44 against a ceiling of 4.0 this way.
3. The escalated β is then seeded back into the control, so the ramp continues from the
   discovered-safe level instead of re-paying the escalation on the next step.

The interaction those three produce, and the reason this harness exists: a step whose length collapsed is
escalated by the ladder (say ×4), and *then* the control -- reading that same collapsed α -- applies its
own backoff (×2) on top. β lands eight times above where the march was working, and the control can only
walk it back down one ``grow`` factor per step. The walk-back is pure overhead: those steps exist only to
undo an overshoot.

**What a replay can and cannot tell you (read this before quoting any number below).** Changing the policy
changes β, which changes the linear system, which changes α and the residual -- none of which a replay
knows, because it has only the α and residual the *shipped* policy produced. So a candidate's replayed
trajectory is a **counterfactual under frozen recorded inputs**, never a prediction of a march. It cannot
say "policy X saves N steps"; a policy that lowers β might well need more steps, or diverge. Every column
this harness prints from a non-baseline policy is labelled ``cf:`` for exactly that reason. What the
replay is good for is *elimination*: a candidate that does not even change the trajectory it was designed
to change, or that drives β somewhere obviously unsafe, is not worth 40 minutes of machine time.

Three quantities here are **not** counterfactual, and are the ones worth acting on:

``self-test``
    The baseline policy's replayed β must equal the logged β on every step of every archived march. It is
    the whole harness's licence to exist: if the model of the control were wrong, every counterfactual
    below it would be void. It is checked first and fails loudly.
``walk-back length``
    After an escalation cascade, β descends by exactly one factor of ``grow`` per comfortable step, so a
    descent's length is ``log(β_top / β_bottom) / log(grow)``. Two readings of it are printed. The
    *descent identity*, over the growth run that actually followed, is a closed form of a counted run --
    they agree by construction, and printing both checks that the run really was uninterrupted growth.
    The *implied walk-back*, from the carried β down to the level the march was working at when the
    ladder fired, is the debt the escalation incurs at the moment it fires and needs nothing that happens
    afterwards; it is the number the candidate policies are aimed at.
``provably-null attempts``
    Steps whose residual is unchanged, to every digit the log records, from the step before. These are
    steps that cost a solve and moved nothing; counting them needs no model at all.

Usage
-----
Self-test and summarize every archived march::

    python3 -u validation/bfs3d_openfoam/step_policy_replay.py

Detail one march (per-step β trajectory for each policy, and the cascades found in it)::

    python3 -u validation/bfs3d_openfoam/step_policy_replay.py march-20260811-161642.log

Restrict the detail to one continuation rung -- the interesting cascades are on the third::

    python3 -u validation/bfs3d_openfoam/step_policy_replay.py march-20260811-161642.log --rung 3
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from march_log_compare import Run, Step, parse

CASE = Path(__file__).resolve().parent

#: How close a replayed β must be to the logged one to count as reproduced. The summary grid writes β to
#: four decimals, so a half-unit in the last place is the tightest bar the record can support; anything
#: tighter would fail on rounding rather than on a modelling error.
BETA_TOLERANCE = 5e-5

#: How close ``β_next`` must be to ``β / grow`` to count as a pure-growth step. Twice
#: :data:`BETA_TOLERANCE`, because here **both** values are read off the log and each carries its own
#: half-unit of rounding; at the small β a long descent ends in, the single-value bar rejects genuine
#: growth steps and truncates the descent by one.
GROWTH_TOLERANCE = 2 * BETA_TOLERANCE


@dataclass(frozen=True)
class ControlSettings:
    """The numbers the marched control and escalation ladder were configured with.

    Defaults are the values this case runs: a Courant-plus-residual dual-time control started at
    β = 0.5, and a ladder that doubles β for at most two rungs. They are not recorded in the log banner,
    so they are stated here and validated by the self-test rather than parsed -- a wrong value shows up
    immediately as a reproduction failure.

    Attributes
    ----------
    beta_start : float
        β for the first step of a continuation rung. Each rung is a separate solve, so the control's
        carried state starts empty there and β resets to this.
    grow, backoff : float
        Factors ``> 1`` the pseudo-timestep is grown by (β divided by) and shrunk by (β multiplied by).
    grow_above, backoff_below : float
        The α thresholds bounding the grow / hold / back-off bands.
    hold_ratio, rise_ratio : float
        The residual-ratio thresholds: growth needs ``ratio <= hold_ratio``, braking fires above
        ``rise_ratio``, and the band between them holds β.
    beta_min, beta_max : float
        Clamps applied by the control (**not** by the ladder -- see the module docstring).
    retry_beta_factor : float
        The factor each ladder rung multiplies β by.
    retry_cycles_limit : int
        The most rungs one step may climb.
    """

    beta_start: float = 0.5
    grow: float = 1.5
    backoff: float = 2.0
    grow_above: float = 0.5
    backoff_below: float = 0.25
    hold_ratio: float = 1.05
    rise_ratio: float = 1.10
    beta_min: float = 0.005
    beta_max: float = 4.0
    retry_beta_factor: float = 2.0
    retry_cycles_limit: int = 2

    def clamp(self, beta: float) -> float:
        """``beta`` brought inside the control's own clamps."""
        return min(max(beta, self.beta_min), self.beta_max)


@dataclass(frozen=True)
class ControlState:
    """What the control carries from one step to the next.

    Attributes
    ----------
    beta : float
        The shift strength the control last settled on -- the value the next update is applied to. After
        an escalation this is whatever the carry rule decided, which is the seam the policies differ at.
    previous_residual : float or None
        The residual of the step *before* the one being fed in, so a ratio can be formed. ``None`` until
        two steps have been seen, and the ratio then defaults to 1 so α alone drives that step.
    """

    beta: float
    previous_residual: float | None


# --------------------------------------------------------------------------------------------------
# The three seams a policy can differ at. Each is its own small strategy so a candidate changes exactly
# one of them -- a policy that had to be spelled as a branch inside one update function could not be
# composed with another, and the interesting candidates are compositions.
# --------------------------------------------------------------------------------------------------


class ControlRule(Protocol):
    """How the control turns the previous step into the next β."""

    def __call__(self, state: ControlState, previous: Step, settings: ControlSettings) -> float:
        """The β for the step about to run.

        Parameters
        ----------
        state : ControlState
            The carried state, whose ``beta`` is the value being updated.
        previous : Step
            The step just taken, read for its α, its residual and its escalation count.
        settings : ControlSettings
            The configured thresholds and factors.
        """


class LadderRule(Protocol):
    """How many rungs the escalation ladder climbs on a step."""

    def __call__(self, step: Step, previous: Step | None, settings: ControlSettings) -> int:
        """The number of escalations to apply to ``step``.

        Parameters
        ----------
        step : Step
            The recorded step, whose ``escalations`` is what the shipped ladder did.
        previous : Step or None
            The step before it (``None`` at the start of a rung), needed to see whether the residual
            moved at all.
        settings : ControlSettings
            The configured factors, read for the rung limit.
        """


class CarryRule(Protocol):
    """What the control carries forward after the ladder has escalated a step."""

    def __call__(
        self,
        control_beta: float,
        escalated_beta: float,
        escalations: int,
        settings: ControlSettings,
    ) -> float:
        """The β the control should resume from.

        Parameters
        ----------
        control_beta : float
            What the control itself chose for this step, before the ladder touched it.
        escalated_beta : float
            What the step actually ran at, after the ladder.
        escalations : int
            How many rungs were climbed; ``0`` means the ladder did not fire.
        settings : ControlSettings
            The configured factors.
        """


@dataclass(frozen=True)
class CflResidualRule:
    """The shipped control: grow on the Courant signal, brake on a rising residual.

    Growth needs *both* signals comfortable -- a step long enough that the inner loop was not clipping,
    and a residual that is flat or falling. Braking fires when *either* wall is hit. The band between the
    two residual thresholds holds β, so the mildly-noisy plateau a developing flow sits on does not make
    the ramp chatter between grow and brake.
    """

    def __call__(self, state: ControlState, previous: Step, settings: ControlSettings) -> float:
        ratio = (
            previous.residual / state.previous_residual
            if state.previous_residual is not None and state.previous_residual > 0.0
            else 1.0
        )
        beta = state.beta
        if previous.alpha < settings.backoff_below or ratio > settings.rise_ratio:
            beta = beta * settings.backoff
        elif previous.alpha >= settings.grow_above and ratio <= settings.hold_ratio:
            beta = beta / settings.grow
        return settings.clamp(beta)


@dataclass(frozen=True)
class SkipBackoffAfterEscalation:
    """Suppress the control's own brake on a step the ladder has already braked.

    The ladder and the control read the *same* collapsed step length and both respond by raising β, so a
    step that escalates twice and is then backed off lands eight times above the level the march was
    working at, and the control can only walk that back one ``grow`` factor per step. This rule holds β
    at its carried value whenever the previous step escalated and the inner rule would have raised it,
    leaving the ladder's raise as the whole response. A raise the inner rule makes for some *other*
    reason is left alone, and so is any lowering.

    Attributes
    ----------
    inner : ControlRule
        The rule whose brake is being suppressed; injected rather than assumed so the suppression can be
        layered over any control.
    """

    inner: ControlRule

    def __call__(self, state: ControlState, previous: Step, settings: ControlSettings) -> float:
        beta = self.inner(state, previous, settings)
        if previous.escalations and beta > state.beta:
            return settings.clamp(state.beta)
        return beta


@dataclass(frozen=True)
class RecordedLadder:
    """Climb exactly the rungs the archived march climbed.

    The baseline. It makes the frozen-input assumption explicit: the number of escalations is a recorded
    input, not something the replay re-derives, because whether a redone step would have met its target
    depends on a solve the replay cannot perform.
    """

    def __call__(self, step: Step, previous: Step | None, settings: ControlSettings) -> int:
        return min(step.escalations, settings.retry_cycles_limit)


@dataclass(frozen=True)
class AbandonOnNullAttempt:
    """Stop climbing once an attempt has produced no residual change at all.

    A step whose residual is identical, to every digit the log records, to the step before it moved
    nothing -- the state is pinned by something more damping cannot lift (a positivity cap already on its
    boundary, most often), and each further rung buys another solve and another walk-back for no
    progress. On the archived marches these come in long runs: one shows 96 consecutive null steps.

    **The one inference this rule makes beyond the record**, which is why it is a counterfactual and not
    a measurement: the log stores only the *accepted* attempt's residual, so a null accepted attempt is
    taken as evidence that the rungs leading to it were null too, and the count is capped at the first
    rung. A ladder that was genuinely making progress on an intermediate rung would be mis-scored here.

    Attributes
    ----------
    inner : LadderRule
        The rule being capped; injected so the abandonment can wrap any ladder.
    """

    inner: LadderRule

    def __call__(self, step: Step, previous: Step | None, settings: ControlSettings) -> int:
        climbed = self.inner(step, previous, settings)
        if previous is not None and step.residual == previous.residual:
            return min(climbed, 1)
        return climbed


@dataclass(frozen=True)
class CarryEscalated:
    """The shipped carry: resume from whatever β the escalation settled at.

    The reasoning it encodes is sound in isolation -- the ladder is the feedback for "how low is safe
    here", so a persistently hard region should not re-pay the escalation every step. What it does not
    account for is the control's own brake landing on top of the same signal.
    """

    def __call__(
        self,
        control_beta: float,
        escalated_beta: float,
        escalations: int,
        settings: ControlSettings,
    ) -> float:
        return escalated_beta if escalations else control_beta


@dataclass(frozen=True)
class NoCarry:
    """Discard the escalation entirely; the control resumes from its own β.

    The other extreme, and the natural control experiment: it removes the walk-back completely, at the
    cost of re-paying the escalation on every step of a genuinely hard region. Whether that trade is
    worth taking is exactly what a replay cannot answer -- the re-paid escalations would change α, and α
    is frozen here.
    """

    def __call__(
        self,
        control_beta: float,
        escalated_beta: float,
        escalations: int,
        settings: ControlSettings,
    ) -> float:
        return control_beta


@dataclass(frozen=True)
class DecayedCarry:
    """Carry part of the escalation, interpolated in the multiplicative sense the ladder works in.

    The ladder is a product of factors, so the halfway point between the control's β and the escalated β
    is the *geometric* mean, not the arithmetic one: ``control × (escalated / control) ** fraction``.
    With ``fraction = 0.5`` a two-rung ×4 escalation is carried as ×2, which costs half as many
    ``grow`` steps to walk back while still leaving the next step better damped than the control alone
    would have made it.

    Attributes
    ----------
    fraction : float
        How much of the escalation to keep, in log-β. ``0`` reproduces :class:`NoCarry` and ``1``
        reproduces :class:`CarryEscalated`, so the two extremes are the endpoints of this family.
    """

    fraction: float = 0.5

    def __call__(
        self,
        control_beta: float,
        escalated_beta: float,
        escalations: int,
        settings: ControlSettings,
    ) -> float:
        if not escalations or control_beta <= 0.0:
            return control_beta
        return control_beta * (escalated_beta / control_beta) ** self.fraction


@dataclass(frozen=True)
class StepPolicy:
    """One candidate, as the composition of a control rule, a ladder rule and a carry rule.

    Attributes
    ----------
    name : str
        Short label used as the column head.
    description : str
        One line saying what the policy changes and why anyone would try it.
    control : ControlRule
        How β is updated between steps.
    ladder : LadderRule
        How many rungs an escalation climbs.
    carry : CarryRule
        What the control resumes from afterwards.
    """

    name: str
    description: str
    control: ControlRule
    ladder: LadderRule
    carry: CarryRule


BASELINE = StepPolicy(
    name="shipped",
    description="the marched policy: Courant-plus-residual control, recorded ladder, carry the escalation",
    control=CflResidualRule(),
    ladder=RecordedLadder(),
    carry=CarryEscalated(),
)

#: The candidates, baseline first. Each differs from the baseline at exactly one seam, so a difference in
#: the replayed trajectory is attributable; a candidate that combined two changes would leave you unable
#: to say which one moved it.
POLICIES: tuple[StepPolicy, ...] = (
    BASELINE,
    replace(
        BASELINE,
        name="no-carry",
        description="drop the escalation the moment the step is accepted; no walk-back, but the next step re-pays it",
        carry=NoCarry(),
    ),
    replace(
        BASELINE,
        name="carry-half",
        description="carry the geometric half of the escalation, halving the walk-back it implies",
        carry=DecayedCarry(0.5),
    ),
    replace(
        BASELINE,
        name="no-double-brake",
        description="the control holds instead of braking on a step the ladder already escalated",
        control=SkipBackoffAfterEscalation(CflResidualRule()),
    ),
    replace(
        BASELINE,
        name="abandon-null",
        description="stop climbing rungs once an attempt has moved the residual not at all",
        ladder=AbandonOnNullAttempt(RecordedLadder()),
    ),
)


@dataclass(frozen=True)
class ReplayStep:
    """One step as a policy would have run it, beside what was recorded.

    Attributes
    ----------
    recorded : Step
        The archived step, kept whole so every downstream reader can reach its rung, flags and cost
        without those being copied out into loose fields.
    control_beta : float
        What the policy's control chose, before the ladder.
    escalations : int
        How many rungs the policy's ladder climbed.
    beta : float
        What the step would have run at: ``control_beta * retry_beta_factor ** escalations``. For the
        baseline this equals ``recorded.beta``, which is the self-test.
    carried : float
        What the policy carries into the next step.
    """

    recorded: Step
    control_beta: float
    escalations: int
    beta: float
    carried: float

    @property
    def matches_record(self) -> bool:
        """Whether this step's β reproduces the logged one to the log's precision."""
        return abs(self.beta - self.recorded.beta) <= BETA_TOLERANCE


def replay(run: Run, policy: StepPolicy, settings: ControlSettings) -> list[ReplayStep]:
    """Walk a policy over an archived march's recorded ``(α, residual, escalations)`` sequence.

    The control's carried state is reset at each continuation rung, because a rung is a separate solve
    and starts with no history -- so β returns to ``beta_start`` there, which the archived logs confirm.
    Within a rung the state threads unbroken.

    Parameters
    ----------
    run : Run
        The parsed march. Its ``steps`` are read in order.
    policy : StepPolicy
        The candidate to walk.
    settings : ControlSettings
        The configured thresholds and factors.

    Returns
    -------
    list of ReplayStep
        One entry per recorded step, in order; length equals ``len(run.steps)``.
    """
    replayed: list[ReplayStep] = []
    state: ControlState | None = None
    previous: Step | None = None
    rung: int | None = None

    for step in run.steps:
        if step.rung != rung:  # a new continuation rung is a fresh solve: no carried state
            state, previous, rung = None, None, step.rung
        if state is None:
            control_beta, previous_residual = settings.beta_start, None
        else:
            control_beta = policy.control(state, previous, settings)
            previous_residual = previous.residual
        escalations = policy.ladder(step, previous, settings)
        # The ladder scales the shift leaf directly, so it is deliberately NOT re-clamped here: an
        # archived march runs at 4.44 against a beta_max of 4.0 for exactly this reason, and clamping
        # would silently break the reproduction.
        beta = control_beta * settings.retry_beta_factor**escalations
        carried = policy.carry(control_beta, beta, escalations, settings)
        replayed.append(
            ReplayStep(
                recorded=step,
                control_beta=control_beta,
                escalations=escalations,
                beta=beta,
                carried=carried,
            )
        )
        state = ControlState(beta=carried, previous_residual=previous_residual)
        previous = step
    return replayed


# --------------------------------------------------------------------------------------------------
# Bounded diagnostics. Nothing below depends on a policy or on an unknown future -- they are properties
# of the recorded trajectory, and are the numbers that may be quoted as measurements.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaPoint:
    """One point of a shift trajectory -- the least a cascade search needs to read.

    A recorded march and a replayed policy both produce a β per step, and the cascade arithmetic is the
    same over either, so both are reduced to this and searched by the one function. Without it the
    counterfactual walk-back would be a second implementation of the recorded one, and the two would
    drift.

    Attributes
    ----------
    step : int
        The step number, as the log numbers it.
    rung : int
        The continuation rung, so a descent is not counted across the reset at a rung boundary.
    beta : float
        The shift the step ran at, after any escalation.
    carried : float
        The shift the control resumes from. It equals ``beta`` under the shipped carry, and is what
        differs under a carry rule -- which is why the walk-back is measured from this and not from
        ``beta``: the step is charged for the damping it ran at, but the *descent* is owed only on what
        the control keeps.
    escalations : int
        How many ladder rungs this step climbed.
    """

    step: int
    rung: int
    beta: float
    carried: float
    escalations: int


def recorded_track(run: Run) -> list[BetaPoint]:
    """The β trajectory the march actually ran, as searchable points.

    The march's own carry is to keep the escalated β, so ``carried`` is the β itself here.
    """
    return [
        BetaPoint(step=s.step, rung=s.rung, beta=s.beta, carried=s.beta, escalations=s.escalations)
        for s in run.steps
    ]


def replayed_track(entries: list[ReplayStep]) -> list[BetaPoint]:
    """The β trajectory a policy would have run, as searchable points."""
    return [
        BetaPoint(
            step=e.recorded.step,
            rung=e.recorded.rung,
            beta=e.beta,
            carried=e.carried,
            escalations=e.escalations,
        )
        for e in entries
    ]


@dataclass(frozen=True)
class Cascade:
    """One run of consecutive escalated steps, and the descent that followed it.

    Attributes
    ----------
    first, last : int
        The step numbers the cascade spans, inclusive.
    beta_before : float
        The β of the step immediately before the cascade -- the level the march was working at when the
        ladder fired.
    beta_peak : float
        The largest β any step inside the cascade ran at.
    beta_owed : float
        The β the control resumes from at the end of the cascade -- what the descent is actually owed on.
        Under the shipped carry this equals :attr:`beta_peak`; a carry rule that keeps less of the
        escalation shows up here and nowhere else.
    beta_resume : float
        The β at the end of the pure-growth run that follows, i.e. the first step at which the trajectory
        does something other than divide β by ``grow`` (or the track's last β, if it grows to the end).
    descent_steps : int
        How many consecutive pure-growth steps that run lasted.
    grow : float
        The growth factor the descent is expressed in, kept here so a cascade can state its own lengths
        without a caller having to re-supply the configuration it was found under.
    """

    first: int
    last: int
    beta_before: float
    beta_peak: float
    beta_owed: float
    beta_resume: float
    descent_steps: int
    grow: float

    @property
    def descent_identity(self) -> float:
        """``log(β_owed / β_resume) / log(grow)`` -- must equal :attr:`descent_steps`.

        Pure growth divides β by exactly ``grow`` per step, so this closed form and the counted run are
        two spellings of one number. Printing both is a check that the run really was uninterrupted
        growth; a disagreement means the descent was not what it looked like.
        """
        return _descent_length(self.beta_owed, self.beta_resume, self.grow)

    @property
    def implied_walk_back(self) -> float:
        """How many growth steps it takes to get from the carried β back to the pre-cascade level.

        This is the cost the escalation commits the march to at the moment it fires, and unlike
        :attr:`descent_steps` it needs nothing that happens afterwards -- the march may be interrupted
        part-way down (and on the archived logs it often is), but the debt was incurred regardless.
        It is the quantity the candidate policies are aimed at.
        """
        return _descent_length(self.beta_owed, self.beta_before, self.grow)


def _descent_length(top: float, target: float, grow: float) -> float:
    """Growth steps from ``top`` down to ``target``, or 0 if it is already at or below."""
    if target <= 0.0 or top <= target:
        return 0.0
    return math.log(top / target) / math.log(grow)


def cascades(track: list[BetaPoint], settings: ControlSettings) -> list[Cascade]:
    """Find every escalation cascade in a β trajectory, with the descent that followed it.

    A cascade is a maximal run of consecutive steps the ladder fired on; consecutive escalations are one
    event rather than several because their β compounds -- the peak is what the march then has to walk
    back from, not each individual raise. The search reads a trajectory rather than a march so the
    recorded and the replayed ones are measured by identical arithmetic.

    Parameters
    ----------
    track : list of BetaPoint
        The trajectory, in step order.
    settings : ControlSettings
        Read for ``grow`` (what counts as a pure-growth step) and ``beta_start`` (the level a cascade on
        the very first step of a rung is measured against).

    Returns
    -------
    list of Cascade
        In step order; empty for a trajectory the ladder never fired on.
    """
    found: list[Cascade] = []
    index = 0
    while index < len(track):
        if not track[index].escalations:
            index += 1
            continue
        start = index
        while index + 1 < len(track) and track[index + 1].escalations:
            index += 1
        peak = max(point.beta for point in track[start : index + 1])
        before = track[start - 1].carried if start else settings.beta_start
        # Walk the descent: a pure-growth step divides the previous β by exactly `grow`, so anything else
        # -- a hold, a brake, another escalation, a rung boundary -- ends the run.
        descent, cursor = 0, index
        while cursor + 1 < len(track):
            here, nxt = track[cursor], track[cursor + 1]
            if (
                nxt.rung != here.rung
                or abs(nxt.beta - here.carried / settings.grow) > GROWTH_TOLERANCE
            ):
                break
            descent, cursor = descent + 1, cursor + 1
        found.append(
            Cascade(
                first=track[start].step,
                last=track[index].step,
                beta_before=before,
                beta_peak=peak,
                beta_owed=track[index].carried,
                beta_resume=track[cursor].beta,
                descent_steps=descent,
                grow=settings.grow,
            )
        )
        index += 1
    return found


def null_attempts(run: Run) -> list[Step]:
    """The steps whose residual is unchanged, to the log's precision, from the step before.

    Such a step ran a solve and moved nothing. It is the signature of a state pinned by a constraint that
    damping cannot lift, and it needs no model to count -- which makes it the one cost figure here that
    can be quoted without a caveat.
    """
    return [b for a, b in zip(run.steps, run.steps[1:], strict=False) if b.residual == a.residual]


def cascade_wall_fraction(run: Run, settings: ControlSettings) -> float:
    """The fraction of a march's wall clock spent inside cascades and their descents.

    Recorded, not counterfactual: it sums the elapsed-time deltas of the steps that are part of an
    escalation cascade or of the pure-growth descent after one. It is emphatically **not** the time a
    better policy would save -- those steps do real work on the state, and a policy that avoided them
    would reach a different state. It is an upper bound on what is even at stake.
    """
    if not run.steps:
        return 0.0
    involved: set[int] = set()
    for cascade in cascades(recorded_track(run), settings):
        involved.update(range(cascade.first, cascade.last + 1 + cascade.descent_steps))
    seconds, previous = 0.0, 0.0
    for step in run.steps:
        if step.step in involved:
            seconds += step.elapsed - previous
        previous = step.elapsed
    return seconds / max(run.wall, 1.0)


# --------------------------------------------------------------------------------------------------
# Self-test and reporting.
# --------------------------------------------------------------------------------------------------


def self_test(runs: list[Run], settings: ControlSettings) -> bool:
    """Assert that the baseline policy reproduces every archived march's logged β.

    This is the harness's licence to exist. Nothing else here is worth reading if this fails: a control
    model that cannot reconstruct what actually happened cannot say anything about what would have.

    Parameters
    ----------
    runs : list of Run
        Every archived march to check. A run with no steps is reported and skipped.
    settings : ControlSettings
        The configuration to check against; a wrong value shows up here as a reproduction failure rather
        than as a quietly wrong counterfactual.

    Returns
    -------
    bool
        True only if every step of every run reproduced.
    """
    print("=" * 104)
    print(
        "SELF-TEST -- the shipped policy must reproduce every logged beta from the recorded inputs"
    )
    print("=" * 104)
    ok = True
    for run in runs:
        if not run.steps:
            print(f"  {run.path.name:<40} no steps parsed -- skipped")
            continue
        bad = [entry for entry in replay(run, BASELINE, settings) if not entry.matches_record]
        status = "ok" if not bad else f"FAILED on {len(bad)} of {len(run.steps)}"
        print(f"  {run.path.name:<40} {len(run.steps):>4} steps   {status}")
        for entry in bad[:5]:
            print(
                f"      step {entry.recorded.step}: replay {entry.beta:.4f} vs logged "
                f"{entry.recorded.beta:.4f}  (alpha {entry.recorded.alpha}, esc {entry.escalations})"
            )
        ok = ok and not bad
    if not ok:
        print(
            "\n  ⚠️  THE CONTROL MODEL IS WRONG. Every counterfactual this harness prints is void until\n"
            "      the replay reproduces the archive. Check ControlSettings against the march that failed."
        )
    return ok


def summarize(run: Run, settings: ControlSettings) -> None:
    """Print one march's recorded diagnostics, then each policy's counterfactual β envelope."""
    events = cascades(recorded_track(run), settings)
    nulls = null_attempts(run)
    print(
        f"\n{'=' * 104}\n{run.path.name}  --  {len(run.steps)} steps, {run.wall:.0f} s\n{'=' * 104}"
    )
    print("  RECORDED (no model, quotable as measurement)")
    print(
        f"    escalation cascades {len(events):<3}   provably-null attempts {len(nulls):<4}"
        f"   cascade+descent share of wall {100 * cascade_wall_fraction(run, settings):.0f}%"
    )
    for cascade in events:
        print(
            f"    steps {cascade.first}-{cascade.last}: beta {cascade.beta_before:.4f} -> peak "
            f"{cascade.beta_peak:.4f}; descent {cascade.descent_steps} steps to {cascade.beta_resume:.4f} "
            f"(identity {cascade.descent_identity:.2f}), walk-back to the pre-cascade level "
            f"{cascade.implied_walk_back:.2f} steps"
        )
    if not events:
        print("    (the ladder never fired on this march)")

    print(
        "\n  cf: COUNTERFACTUAL -- the beta each policy would have set from the SAME recorded alpha and"
    )
    print(
        "      residual sequence. The real march would have produced different ones. Not a saving."
    )
    print(
        f"    {'policy':<18}{'beta peak':>11}{'beta min':>10}{'escalated':>11}{'cascades':>10}"
        f"{'walk-back (steps)':>19}"
    )
    for policy in POLICIES:
        entries = replay(run, policy, settings)
        if not entries:
            continue
        betas = [entry.beta for entry in entries]
        # Summed over cascades by the SAME finder the recorded line above uses, so the baseline row is
        # literally the recorded number and the candidates are read off the same scale.
        events_here = cascades(replayed_track(entries), settings)
        walk_back = sum(cascade.implied_walk_back for cascade in events_here)
        label = policy.name if policy is BASELINE else f"cf: {policy.name}"
        print(
            f"    {label:<18}{max(betas):>11.4f}{min(betas):>10.4f}"
            f"{sum(1 for e in entries if e.escalations):>11}{len(events_here):>10}{walk_back:>19.2f}"
        )
    for policy in POLICIES:
        print(f"      {policy.name:<18} {policy.description}")


def detail(run: Run, settings: ControlSettings, rung: int | None) -> None:
    """Print the per-step β trajectory of every policy side by side, for one march.

    Parameters
    ----------
    run : Run
        The parsed march.
    settings : ControlSettings
        The configuration to replay under.
    rung : int or None
        Restrict to one continuation rung; ``None`` prints all of them.
    """
    tracks = {policy.name: replay(run, policy, settings) for policy in POLICIES}
    print(f"\n{'=' * 104}\nper-step beta -- {run.path.name}" + (f", rung {rung}" if rung else ""))
    print(
        "every column but 'logged'/'shipped' is a COUNTERFACTUAL under the frozen recorded alpha and "
        f"residual\n{'=' * 104}"
    )
    heads = "".join(f"{('cf:' + p.name if p is not BASELINE else p.name):>20}" for p in POLICIES)
    print(f"  {'step':>5}{'alpha':>8}{'esc':>5}{'logged':>10}{heads}")
    for index, step in enumerate(run.steps):
        if rung is not None and step.rung != rung:
            continue
        row = "".join(f"{tracks[p.name][index].beta:>20.4f}" for p in POLICIES)
        print(f"  {step.step:>5}{step.alpha:>8.3f}{step.escalations:>5}{step.beta:>10.4f}{row}")


def main() -> None:
    rung, arguments, pending = None, [], iter(sys.argv[1:])
    for argument in pending:
        if argument == "--rung":
            rung = int(next(pending))
        elif argument.startswith("--"):
            raise SystemExit(__doc__)
        else:
            arguments.append(argument)

    settings = ControlSettings()
    archive = [parse(path) for path in sorted(CASE.glob("march-*.log"))]
    if not archive:
        raise SystemExit(f"no march-*.log found under {CASE}")
    if not self_test(archive, settings):
        raise SystemExit(1)

    if arguments:
        for name in arguments:
            path = Path(name) if Path(name).exists() else CASE / name
            run = parse(path)
            summarize(run, settings)
            detail(run, settings, rung)
    else:
        for run in archive:
            if run.steps:
                summarize(run, settings)


if __name__ == "__main__":
    main()
