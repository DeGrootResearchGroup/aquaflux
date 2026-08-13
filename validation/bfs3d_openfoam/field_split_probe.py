"""Does splitting the turbulence out of the coupled preconditioner's hierarchy help?

The coupled preconditioner puts all six fields -- the ``[u, v, w, p]`` saddle and the two transported
scalars ``k`` and ``omega`` -- through one multigrid hierarchy with one level smoother. This asks whether
giving the two groups their own hierarchies, while retaining one triangle of the coupling between them, is
better. It is **not** the block-*diagonal* arrangement that was tried and refuted: that one drops the
coupling, and the coupling is load-bearing. A triangle keeps half of it exactly.

**Where the headroom is, and is not.** At the states the march visits, a preconditioner rebuilt at the
iterate solves the forward system in one restart cycle *at the march's own loose stop*, so that pairing
cannot separate two candidates -- an easy operator is not a test, and a tie there is no information. Two
kinds of state restore the discrimination, and both are configurations something really solves:

* the **converged state at zero shift** -- the operator every gradient's transpose solve meets. Removing
  the pseudo-transient shift is what makes this operator hard, and the adjoint has no preconditioner floor
  to soften it, so this is where a better preconditioner has something real to win. Note it must be the
  *converged* state: stripping the shift off a mid-march iterate would measure an operator that nothing,
  forward or adjoint, ever solves. This is the discriminating state in the shipped ``STATES`` set.
* a **captured hard inner iterate**, at the shift the march solved it under, driven far past the march's
  own stop so the arms separate instead of all stopping at one cycle. The march's expensive solves are
  measured to be staleness rather than hard operators, so this is a comparison of quality at a matched
  preconditioner, not a reproduction of the march's cost. These are written only when a march is run with
  ``BFS3D_INNER_DUMP_ABOVE`` set, so a set captured without it has none, and adding one means re-running
  the march rather than reusing an older capture under a bundle whose defaults have since moved.

**The second question, which the same arms answer.** Every smoother the JAX-native multigrid in this
package has is Jacobi-class; the one thing PETSc supplies that it does not is the incomplete-LU sweep the
shipped bundle depends on -- which is also the least parallelizable component, being a sequential
triangular solve. So: is the ``[u, v, w, p]`` block alone smoothable by a Jacobi-class method even though
the six-field block is not? If it is, a field split is the route to a multigrid with no incomplete-LU
anywhere in it.

Two distinct obstructions could defeat a Jacobi-class smoother here, and the arms are chosen to tell them
apart, because they point at opposite conclusions:

* the ``omega`` **column is nearly empty** in the diagonal cell blocks -- perturbing ``omega`` in a cell
  barely changes that cell's own equations, since its influence travels by neighbour transport. A local
  smoother cannot see the field. If this is the obstruction, removing ``omega`` from the block fixes it.
* the ``[u, v, w, p]`` block is **indefinite** -- it is the saddle. Chebyshev acceleration assumes a
  bounded positive spectrum, which a saddle does not have. If *this* is the obstruction it survives the
  split untouched, because the split does not make the saddle definite.

A Chebyshev arm that fails on the four-field block as badly as on the six-field one indicts the
indefiniteness and closes the route; one that succeeds on four where it failed on six indicts ``omega``
and opens it.

**A third obstruction sits between those two and was missed by both**, which is why the cell-block arms
exist. Chebyshev and damped Jacobi are *point* methods -- they read a cell's scalar diagonal and discard
the local pressure-velocity coupling outright. On a saddle that coupling is the difficulty, not a
detail, so their failure indicts point smoothing and says nothing about whether a **cell-block** solve
would serve. Point-block Jacobi inverts each cell's dense ``[u, v, w, p]`` block and is still a batch of
independent small dense solves with no sequential dependency, so it has the property the incomplete-LU
sweep lacks. It is the smoother a JAX-native hierarchy would use, and it had never been paired with this
block. The native arms then ask the same question of a hierarchy written here rather than in PETSc: if
the matched PETSc row converges and the native one caps, the deficit is in our coarsening and is a
defined thing to fix; if both cap, the next candidate has to be globally coupled rather than cell-local.

**And the converse arm, which is the one a split is really for.** Removing the incomplete-LU sweep
*everywhere* is the ambitious prize; **confining** it to the saddle is the reachable one. The
``[k, omega]`` block is not a saddle -- it is a two-field advection-diffusion-reaction pair with a genuine
diagonal -- so a Jacobi-class smoother ought to serve it, and if it does, the sequential triangular solve
is needed on four fields instead of six and the scalars could run on a multigrid this package can write
itself. Hence the asymmetric arms: keep ILU(0) where it is known to be needed, relax it only where the
operator should not need it.

**Method** -- each of these has produced a verdict on this case that had to be retracted:

* the TRUE residual through GMRES, never a preconditioned norm, a one-apply contraction, or a spectral
  radius;
* a REAL right-hand side, the steady residual ``-R(state)``, never a random vector;
* the REAL shift diagonal ``beta * d``, not a uniform stand-in;
* one materialization per state, shared by every arm, so two arms can never differ for any reason but the
  options under test -- and so only one copy of a multi-gigabyte Jacobian is ever live;
* a **faithfulness gate**: where a restart-cycle count is on record for the shipped monolithic arm at a
  state, that arm must reproduce it or the run refuses to report. A state with nothing on record is
  gated only on the control converging at all;
* **states from ONE march, whose bundle is written down beside them** (see ``STATES``). The checkpoint
  names come from a per-run counter over a rolling buffer, so they carry no date and no configuration:
  a file from a march run before a default moved looks exactly like a current one and describes a
  different discrete problem.

**The one way this departs from the march, stated because it cannot be removed.** A dual-time step solves
for its own residual ``G = R + beta d (phi - phi_n)``, not for ``R``. At inner iteration 0 the two are
identical (``phi = phi_n``), which is why a sweep over end-of-step checkpoints is right to use ``R``; at a
captured inner iterate they are not, and on the hardest one they differ by a factor of some 200 (``|G|``
3.8e-03 against ``|R|`` 8.3e-01). Recovering ``G`` needs ``phi_n``, which the observer does not record. So
the operator, the state and the shift here are the march's and the right-hand side is not: the cycle count
is comparable to the record and is gated on, the achieved residual is not and is only reported. Since every
arm sees the identical right-hand side, the comparison *between* arms -- the point of the probe -- is
unaffected.

**Usage** -- one state per run, since each materializes a Jacobian of some gigabytes::

    python3 -u validation/bfs3d_openfoam/field_split_probe.py state-00067 > field_split.log 2>&1

A second argument builds the preconditioner at a **different** state from the operator, which is the third
pairing worth measuring: the march freezes its preconditioner for a whole inner loop, and its expensive
solves are measured to be that staleness rather than hard operators. Two consecutive iterates reproduce
exactly one iteration of it, and whether a field split decays more gracefully under staleness matters more
on this case than whether it is better when matched::

    python3 -u validation/bfs3d_openfoam/field_split_probe.py state-00066 state-00065

``--arms=key,key`` restricts the ladder, which is how to run the cell-block and native arms alone::

    python3 -u validation/bfs3d_openfoam/field_split_probe.py state-00067 \
        --arms=split flow/ilu0,split pbjacu/ilu0,split native4/ilu0

Every smoother named here is a **fixed linear operator**, which the adjoint's transpose solve requires: a
Chebyshev smoother is a fixed polynomial once its eigenvalue bounds are estimated during setup, unlike a
GMRES-accelerated smoother, whose polynomial depends on the right-hand side.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))  # import aquaflux from the working tree, as compare.py is run
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.flow.block_preconditioner import BlockPreconditioner  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
    NodalNativeInverse,
    air_multigrid_solve,
    block_stencil_gather_map,
    build_air_hierarchy,
    build_amg_vcycle,
    build_block_triangular_field_split,
    build_convection_hierarchy,
    convection_multigrid_solve,
    relative_residual_gmres,
    solve_linear,
)
from aquaflux.solve.linear import restart_cycles  # noqa: E402
from aquaflux.solve.multigrid import (  # noqa: E402
    _CsrOperator,
    _operator_matvec,
    _smoothed_ops,
)
from aquaflux.turbulence.coupled import (  # noqa: E402
    _PROBE_BATCH_SIZE,
    _batched_jacobian_matvec,
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
    coupled_scaled_norm,
)

#: Adjoint-grade, far past the march's own 30 % inexact-Newton stop, so arms separate rather than tie.
RTOL = 1e-8
#: A failing arm is identified by its true residual; letting one run to thousands of matrix-vector
#: products costs more than every healthy arm together.
#: ``BFS3D_PROBE_MAX_RESTARTS`` caps the restart cycles. **A failing arm costs 3-4x a converging one**
#: precisely because it runs the cap out -- measured, the Krylov solves are ~80 % of a run's wall and
#: the fixed setup is a constant ~95 s -- so an exploratory sweep is far cheaper at a low cap. What it
#: costs is comparability: the reported residual is wherever the arm had reached when the budget ran
#: out, so numbers from two different caps are NOT comparable and every run must carry its own
#: controls. A candidate that will beat the incumbent's 11 cycles shows it well inside 20.
MAX_RESTARTS = int(os.environ.get("BFS3D_PROBE_MAX_RESTARTS", "60"))
SOLVER = relative_residual_gmres(RTOL, restart=15, stagnation_iters=40, max_restarts=MAX_RESTARTS)


#: The states this runs on. All three come from ONE march, and which march is part of the measurement:
#: a 67-step Reynolds continuation converging to ``|R|`` 3.586e-06 at mid-span ``x_r/h`` 8.361 for 319
#: restart cycles, run under the shipped bundle -- field split, native trailing inverse,
#: ``zerogradient`` k wall, positivity floor 1e-08, ILU(0) x4 on the saddle, plain aggregation,
#: ``coarse_eq_limit`` 2000, column reach 3/3/3/3/2/2, forward restart 15, ``refresh_on_cycles`` 3.
#:
#: **A checkpoint set is only usable with the bundle it was written under**, which is why that list is
#: here rather than in a commit message. Several of those defaults moved within the last few days --
#: the trailing inverse, the wall closure and the positivity floor all changed -- and a state carried
#: over from before them is a different discrete problem, not an older measurement of this one. The
#: names are a per-run counter over a rolling buffer, so they carry no date and nothing complains.
#:
#: Nothing is on record for any of them from this probe, so the self-check reduces to the control
#: converging at all; the monolithic arm is in ``ARMS`` for that reason.
class _State(NamedTuple):
    """One probed operating point.

    ``march_beta`` and ``checkpoint_shift`` are **different quantities that happen to coincide** for
    every mid-march entry, which is why they were one field until the converged state needed them apart.
    ``march_beta`` is the shift to build the OPERATOR at -- the operating point under test.
    ``checkpoint_shift`` is the shift the checkpoint was WRITTEN at, and is only an identity check: the
    names come from a per-run counter over a rolling buffer, so a later march silently replaces a file
    under a name this table documents.

    They part at ``state-00069``. That entry is measured at ``march_beta = 0`` **on purpose** -- the
    adjoint's transpose solve meets the unshifted operator and has no preconditioner floor to soften it
    -- while the checkpoint itself was written mid-march at the shift that step ran under. Conflating
    the two made the identity check demand a shift of zero from a file that could never carry one, so
    the converged entry could not be loaded at all.
    """

    march_beta: float
    checkpoint_shift: float | None  # None for an inner iterate, which records no shift
    checkpoint_residual: float | None  # likewise; the stronger half of the identity fingerprint
    recorded: tuple[int, float] | None
    description: str


STATES = {
    # The ADJOINT's operating point, and the one discriminating state this set has. Every other entry
    # here is an end-of-step checkpoint, which is by construction the CHEAP solve of a step -- a settled
    # state met with a freshly refreshed preconditioner -- and on the shipped march those all cost one
    # or two restart cycles, so no arm can separate from another on them. Stripping the shift is what
    # makes this one hard, and it is not an artificial hardness: the transpose solve behind every
    # `jax.grad` meets exactly this operator, with no preconditioner floor to soften it.
    "state-00067": _State(
        0.0,
        0.0064,
        3.5860e-06,
        None,
        "the converged state, |R| 3.586e-06 -- the ADJOINT's operator, at zero shift",
    ),
    # STEP-INITIAL states, at their own shift. They cannot rank candidates, but they are what a march
    # actually pays for, so they say what an arm COSTS where the operator is easy -- the complement to
    # the hard state, which only says whether it survives at all.
    "state-00066": _State(
        0.0096,
        0.0096,
        2.6025e-05,
        None,
        "step 27 of the target rung, step-initial -- the cheap solve that is most of the march",
    ),
    "state-00065": _State(
        0.0144,
        0.0144,
        1.3046e-04,
        None,
        "step 26 of the target rung, step-initial -- a second cheap solve, higher shift",
    ),
}

#: Level-smoother recipes, as PETSc options layered over the shipped bundle. ``ilu0`` is the shipped
#: default (an empty override). The other two are the Jacobi-class candidates a JAX-native multigrid could
#: actually implement, since neither needs a sequential triangular solve.
SMOOTHERS = {
    "ilu0": {},
    "chebyshev": {"mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi"},
    "jacobi": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "jacobi",
    },
    # Sweep-count variants of the shipped smoother. The four sweeps in the shipped bundle were tuned
    # against the SIX-field block; the transported scalars are a much easier operator than the saddle
    # they were tuned on, so the same count may simply be more work than that block needs. `ksp_max_it`
    # is where the sweep count lands (the builder sets it from `smoother_sweeps`), and caller overrides
    # are applied last, so naming it here replaces it for this block alone.
    "ilu0x2": {"mg_levels_ksp_max_it": 2},
    "ilu0x1": {"mg_levels_ksp_max_it": 1},
    # Point-block Jacobi: invert each cell's own dense 2x2 [k, omega] block, nothing else. The Mat
    # carries a block size of the group's field count, so PETSc reads the blocks straight off it. This is
    # the natural smoother for the measured shape of this block -- the k/omega coupling is almost
    # entirely a same-cell algebraic term (the destruction pair and the production limiter), which a
    # per-cell block inverse captures exactly while a point method sees only the diagonal. It is also a
    # batch of independent 2x2 solves, so unlike the incomplete-LU sweep it carries no sequential
    # dependency and maps onto an accelerator directly.
    "pbjacobi": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "pbjacobi",
    },
    # The same smoother UNDAMPED, which is PETSc's own default Richardson scale and is what the
    # JAX-native hierarchy runs. Carrying both is not thoroughness: on the turbulence block the damping
    # factor alone was worth 10 restart cycles against 2, so an arm quoted at one damping is not a
    # result about point-block Jacobi. This is the row to compare a native arm against, since a damped
    # PETSc row against an undamped native one measures the damping and gets attributed to the
    # hierarchy.
    "pbjacobi-undamped": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 1.0,
        "mg_levels_pc_type": "pbjacobi",
    },
    # Successive over-relaxation: still a sequential sweep, but a much cheaper one than an incomplete
    # factorization -- no factor to form or store, and nothing to re-form when the operator changes, so
    # it would also cut the setup half of a refresh. Included because a transported-scalar pair with a
    # genuine diagonal is the operator SOR is actually designed for; it diverged on the six-field block,
    # which is a statement about the saddle rather than about these two equations.
    "sor": {"mg_levels_ksp_type": "richardson", "mg_levels_pc_type": "sor"},
    # The damped-Jacobi sweep ladder. This family matters out of proportion to its cycle count: a
    # diagonal scaling is a sparse matrix-vector product and nothing else, so unlike every incomplete-LU
    # or Gauss-Seidel variant it carries no sequential dependency and needs no factorization to store or
    # re-form. If it is adequate on this block, the block can leave a host solver entirely.
    "jacobix8": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_max_it": 8,
    },
    "jacobix2": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_max_it": 2,
    },
    "jacobix1": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "jacobi",
        "mg_levels_ksp_max_it": 1,
    },
    # Point-block Jacobi at two sweeps. At four it bought an extra restart cycle, which cost more than
    # its cheaper application saved; halving the sweeps asks whether that was the sweep count or the
    # method.
    "pbjacobix2": {
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_richardson_scale": 0.7,
        "mg_levels_pc_type": "pbjacobi",
        "mg_levels_ksp_max_it": 2,
    },
    # A Vanka patch relaxation on the FOUR-field saddle block. Two things make this a different
    # experiment from the one that condemned cell-centred patch relaxation on the six-field block:
    # that verdict's mechanism was that the cell block is weakly coupled in omega *everywhere*, which a
    # split removes by construction; and what remains -- the indefiniteness of the saddle -- is the one
    # thing a patch method is actually designed for, where Chebyshev is not.
    #
    # `vanka_centre_field` MUST be given here. Its default is three fields from the end, which finds the
    # pressure in the six-field layout [u,v,w,p,k,omega] and would silently pick `v` in a four-field
    # block -- an arm that centres its patches on a velocity component measures nothing about Vanka.
    # With the centre at 3, `before_centre` takes the neighbours' [u,v,w], which is the classical patch.
    "vanka": {
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "aquaflux.solve.vanka.VankaPC",
        "mg_levels_vanka_centre_field": 3,
    },
    "vanka-mult": {
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "aquaflux.solve.vanka.VankaPC",
        "mg_levels_vanka_centre_field": 3,
        "mg_levels_vanka_multiplicative": True,
    },
}

FLOOR = (
    compare.PC_BETA_FLOOR
)  # 0.05 -- the forward V-cycle is built here, the operator keeps its own beta


def load_state(name: str) -> jnp.ndarray:
    """The captured state, reporting what it was and REFUSING one that is not what :data:`STATES` says.

    The two checkpoint kinds carry different metadata -- an inner iterate knows its attempt and how its
    own solve went, an end-of-step checkpoint knows the step and its residual -- so each is reported on
    its own terms rather than through a lowest common denominator that would name neither.

    **The shift check is not defensive programming, it is the fix for a real silent failure.** The
    checkpointer keeps only the last few files and numbers them from a counter that restarts with each
    march, so a later run REPLACES ``state-000NN`` with a completely different state under the same name.
    That happened: a name documented here as the converged zero-shift state came back holding a mid-march
    iterate at shift 0.98 from an abandoned run. Nothing would have complained -- the probe would have
    paired an operator built at this table's shift with a state that never had it, and reported the
    result as a measurement at the documented operating point. Every step checkpoint records the shift it
    was written at, so the mismatch is free to detect; refuse rather than measure.
    """
    path = CASE / "checkpoints" / f"{name}.npz"
    if not path.exists():
        raise SystemExit(
            f"{name}: no such checkpoint. These are a rolling buffer (`BFS3D_CHECKPOINT_KEEP`, default "
            f"3) and a later march will have rotated it away -- re-run the case to regenerate, raising "
            f"the keep count if a study needs the whole trajectory.\n  present: "
            f"{sorted(p.stem for p in path.parent.glob('*.npz'))}"
        )
    data = np.load(path)
    if "attempt" in data:
        detail = (
            f"attempt {int(data['attempt'])} inner {int(data['inner'])}, the march took "
            f"{int(data['cycles'])} cycles at alpha {float(data['alpha']):.2e}, "
            f"|G| {float(data['g_before']):.4e} -> {float(data['g_after']):.4e}"
        )
    else:
        entry = STATES.get(name)
        # Fingerprint on the SHIFT and the RESIDUAL together, and require both to be documented.
        #
        # The shift alone is a weak fingerprint precisely where this has to work: every end-of-step
        # checkpoint in a converged tail carries essentially the same shift, so a file replaced by a
        # DIFFERENT march's tail state passes a shift-only test while being a different state. The
        # residual separates them for free -- it is in the file already, and it moves by orders of
        # magnitude along a march where the shift moves by a few percent.
        #
        # `checkpoint_shift` is what the march WROTE the file at, which is not `march_beta`, the shift
        # the probe goes on to operate it at. They coincide everywhere except the converged state,
        # whose whole point is to be measured unshifted.
        for label, recorded, expected, tolerance in (
            ("shift", float(data["shift"]), entry.checkpoint_shift if entry else None, 0.02),
            (
                "residual",
                float(data["residual_norm"]),
                entry.checkpoint_residual if entry else None,
                0.05,
            ),
        ):
            if expected is None:
                # A documented step checkpoint with nothing to check against is not a state this probe
                # can stand behind: the guard exists because these files are silently replaced, and a
                # missing expectation is how it was previously switched off by accident.
                raise SystemExit(
                    f"{name}: the STATES entry documents no expected {label}, so this checkpoint's "
                    "identity cannot be verified. Fill it in from the march log that wrote the file."
                )
            # Loose on purpose: the table records these to about four figures, so an exact comparison
            # rejects a matching state. What this must catch is a REPLACED state, which differs by
            # orders of magnitude (a shift of 0.98 where 0.0064 was documented), not by rounding.
            if not np.isclose(recorded, expected, rtol=tolerance, atol=1e-12):
                raise SystemExit(
                    f"{name}: this checkpoint carries {label} {recorded:.6g}, but the STATES table "
                    f"describes it as {expected:.6g}. The file has been overwritten by a later march "
                    "(the names come from a per-run counter over a rolling buffer), so it is NOT the "
                    "state this entry documents. Re-run the case to regenerate it, or point the entry "
                    "at a checkpoint that still matches."
                )
        recorded = float(data["shift"])
        detail = (
            f"end of step {int(data['step'])}, |R| {float(data['residual_norm']):.4e}, "
            f"march shift {recorded:.4f}"
        )
    print(f"{name}: {detail}", flush=True)
    return jnp.asarray(data["state"])


def march_solver(coupled, policy, state):
    """The forward solver the coupled multigrid march actually runs, for the self-check arm.

    Not the coupled incomplete-LU path's solver, which is a different object -- 1 % in a plain 2-norm at
    restart 10 against this one's 30 % in a row-scaled norm at restart 15. Reaching for the wrong one is
    easy and it does not announce itself: at a state where both converge in a single cycle the check still
    passes and reports a validation it never performed.
    """
    return relative_residual_gmres(
        0.3,
        norm=coupled_scaled_norm(coupled, policy, state),
        restart=15,
        stagnation_iters=40,
        max_restarts=60,
    )


def materialize(coupled, state, plan, structure, n_fields) -> sp.csr_matrix:
    """The **unshifted** field-major Jacobian at this iterate -- the one expensive step, done once.

    Unshifted because the operating points below differ only in the diagonal they add, and re-running a
    several-hundred-probe coloured jvp for each of them would dominate the run.
    """
    started = time.time()
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v),
        plan,
        lambda seeds: _batched_jacobian_matvec(coupled, state, seeds),
        _PROBE_BATCH_SIZE,
        structure,
    )
    print(
        f"  materialized {jacobian.shape[0]} dofs, {jacobian.nnz / 1e6:.1f}M nnz "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )
    return jacobian


def monolithic(shifted, groups, n_fields, smoother):
    """The shipped arrangement: one V-cycle over all six fields. The control."""
    return MonolithicAmgPreconditioner(
        build_amg_vcycle(
            shifted,
            n_fields,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            extra_options=SMOOTHERS[smoother] or None,
        )
    )


class JaxNativeBlockInverse:
    """A JAX-native fixed-cycle multigrid, wearing the host ``apply(residual, transpose=...)`` interface.

    The hierarchies in :mod:`aquaflux.solve.multigrid` are written in JAX and are the ones this package
    could run on an accelerator without PETSc. They are also a fixed number of cycles with fixed smoothing
    and a direct coarse solve, so ``b -> x`` is a constant linear map -- which is what lets the transpose
    come from :func:`jax.linear_transpose` rather than from a hand-written transposed cycle, and what
    makes it legal for the adjoint at all.

    The conversion at each boundary is real work, so this is a study adapter: a production JAX-native
    split would keep the whole application on the traced side instead of crossing back to numpy per apply.
    """

    def __init__(self, cycle, n_dofs: int) -> None:
        self._cycle = cycle
        self._n_dofs = n_dofs
        self._transpose = jax.linear_transpose(cycle, jnp.zeros(n_dofs, dtype=jnp.float64))

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        vector = jnp.asarray(residual, dtype=jnp.float64)
        out = self._transpose(vector)[0] if transpose else self._cycle(vector)
        return np.asarray(out, dtype=np.float64)

    def destroy(self) -> None:
        """Nothing to release -- the hierarchy is plain arrays, not a host solver's handles."""


class BlockJacobiInverse:
    """A fixed number of block-Jacobi sweeps as a WHOLE block inverse -- no multigrid, no host solver.

    Not a smoother inside a V-cycle: this replaces the hierarchy. That is worth trying on the ``[k, ω]``
    block specifically because the block is strongly **block**-diagonally dominant -- the neighbour
    coupling is ~12 % of the same-cell coupling for the diagonal fields and ~1 % for ``∂R_ω/∂k`` -- and
    the error operator of block Jacobi is exactly that neighbour part. A hierarchy exists to move
    information globally; where the operator is this local there may be nothing for it to do.

    Each sweep is ``x += D_blk⁻¹ (r − A x)`` with ``D_blk`` the per-cell dense block. Two properties make
    it usable where a general iterative inverse would not be: a **fixed** sweep count keeps ``b -> x`` a
    constant linear map, which the non-flexible outer Krylov and the adjoint both require; and the
    transpose is the same iteration over ``Aᵀ`` with the transposed blocks, so it is available in closed
    form rather than numerically.

    It is also the shape an accelerator wants: the per-cell solves are independent (a batched tiny solve,
    not a sequential triangular sweep), and the residual is one sparse matrix-vector product. On this
    block the cell matrices are lower-triangular with unit diagonal after equilibration, so the block
    solve degenerates to a single fused multiply-add per cell -- but this keeps the general dense solve,
    since the point here is to measure the method rather than to hand-fuse one operator's structure.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The block to invert, **field-major** within the group: degree of freedom ``(cell i, field f)``
        sits at ``f * n_cells + i``.
    n_fields : int
        Fields in the group, so ``n_cells = matrix.shape[0] // n_fields``.
    sweeps : int
        Iterations. Fixed, not a tolerance -- see above.
    """

    def __init__(self, matrix: sp.spmatrix, n_fields: int, sweeps: int) -> None:
        self._a = sp.csr_matrix(matrix)
        self._at = sp.csr_matrix(self._a.transpose())
        self._sweeps = sweeps
        self._n_dofs = self._a.shape[0]
        n_cells = self._n_dofs // n_fields
        self._shape = (n_fields, n_cells)
        # Gather the per-cell dense blocks. Field-major means a cell's degrees of freedom are strided by
        # n_cells, so the blocks are NOT contiguous and have to be picked out by index.
        rows = np.arange(n_cells)
        blocks = np.empty((n_cells, n_fields, n_fields))
        for f in range(n_fields):
            for gcol in range(n_fields):
                blocks[:, f, gcol] = np.asarray(
                    self._a[f * n_cells + rows, gcol * n_cells + rows]
                ).ravel()
        self._inverse = np.linalg.inv(blocks)
        self._inverse_t = np.transpose(self._inverse, (0, 2, 1))

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    def _precondition(self, vector: np.ndarray, inverse: np.ndarray) -> np.ndarray:
        """Apply the block-diagonal inverse: reshape to (field, cell), contract per cell, flatten."""
        per_cell = vector.reshape(self._shape).T  # (n_cells, n_fields)
        return np.einsum("cij,cj->ci", inverse, per_cell).T.ravel()

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        operator = self._at if transpose else self._a
        inverse = self._inverse_t if transpose else self._inverse
        x = self._precondition(residual, inverse)
        for _ in range(self._sweeps - 1):
            x = x + self._precondition(residual - operator @ x, inverse)
        return x

    def destroy(self) -> None:
        """Nothing to release -- plain numpy arrays, no host solver handles."""


class AlgebraicSimpleInverse:
    """A velocity/pressure SIMPLE split of the flow block, built from the assembled matrix alone.

    Every arm that treats ``[u, v, w, p]`` as one block asks a single hierarchy to coarsen a saddle and
    a single smoother to relax one. This does the classical thing instead: eliminate pressure through a
    Schur complement, leaving two operators that are **not** saddles -- a convection-diffusion velocity
    block and a scalar elliptic pressure operator -- each of which an ordinary aggregation multigrid,
    including this package's own, handles without an incomplete-LU sweep.

    The field-major layout makes the partition free, exactly as it does one level up: within the flow
    block ``[u, v, w]`` occupies ``[0, dim*n)`` and ``p`` the rest, so ``F``, ``G``, ``D`` and ``C`` are
    contiguous submatrices rather than gathers.

    The Schur complement is the algebraic SIMPLE one, ``S = C - D diag(F)^-1 G`` -- one sparse triple
    product on the assembled block, needing no assembler, no mass flux and no closure. That matters for
    where it can be used: it is a function of the matrix, so it fits the same ``(block, n_fields) ->
    inverse`` seam every other arm here uses, and it would fit a distributed or accelerator path the
    same way. Note ``C`` is not zero for this discretization -- Rhie--Chow damping gives the continuity
    row a genuine elliptic diagonal -- so ``S`` is a correction to an already-elliptic operator rather
    than a construction of one from nothing.

    One application is the standard SIMPLE sequence, which is block-triangular plus a velocity
    correction::

        du0 = F^-1 ru
        dp  = S^-1 (rp - D du0)
        du  = du0 - diag(F)^-1 G dp

    Dropping the last line would leave a plain block-triangular split; keeping it is what makes this
    SIMPLE rather than a reordering, and it costs one sparse matrix-vector product.

    **The transpose is closed form**, which the adjoint's transpose solve requires: writing the sequence
    as a matrix and transposing it reverses the order and transposes each block, so

        t   = S^-T (yp - (diag(F)^-1 G)^T yu)
        du' = F^-T (yu - D^T t)

    returns ``(du', t)``. With fixed-cycle inner inverses the whole thing is a fixed **linear** operator,
    so the non-flexible outer Krylov is legal too.

    Parameters
    ----------
    block : scipy.sparse matrix
        The flow block, field-major, shape ``((dim + 1) * n_cells,)`` square.
    n_fields : int
        Fields in the block, ``dim + 1``.
    inner_velocity, inner_pressure : callable
        ``(sub_matrix, n_fields_in_sub) -> inverse`` for ``F`` and for ``S``, **separately**. Keeping
        them apart is what lets one half be varied while the other is held, which is the only way to
        attribute a failure to one of them: with a single factory for both, an arm that fails says the
        decomposition failed and nothing about where. Holding the pressure half on a known-good host
        V-cycle while the velocity half goes native isolates the velocity question, and the converse
        isolates the pressure one.
    """

    def __init__(self, block: sp.spmatrix, n_fields: int, inner_velocity, inner_pressure) -> None:
        matrix = sp.csr_matrix(block)
        self._n_dofs = matrix.shape[0]
        n_cells = self._n_dofs // n_fields
        self._split = nv = (n_fields - 1) * n_cells
        f_block = matrix[:nv, :nv].tocsr()
        self._g = matrix[:nv, nv:].tocsr()
        self._d = matrix[nv:, :nv].tocsr()
        c_block = matrix[nv:, nv:].tocsr()
        # diag(F)^-1 G, formed once: it appears in both the velocity correction and the Schur.
        f_diagonal = f_block.diagonal()
        if not np.all(np.isfinite(f_diagonal)) or np.any(f_diagonal == 0.0):
            raise ValueError("the velocity block has a zero or non-finite diagonal; cannot form S.")
        self._dg = (sp.diags(1.0 / f_diagonal) @ self._g).tocsr()
        schur = (c_block - self._d @ self._dg).tocsr()
        self._dt, self._dgt = self._d.T.tocsr(), self._dg.T.tocsr()
        self._f_inv = inner_velocity(f_block, n_fields - 1)
        self._s_inv = inner_pressure(schur, 1)
        print(
            f"      SIMPLE split: F {f_block.shape[0]} dofs / {f_block.nnz / 1e6:.1f}M nnz, "
            f"S {schur.shape[0]} dofs / {schur.nnz / 1e6:.1f}M nnz "
            f"(C {c_block.nnz / 1e6:.1f}M)",
            flush=True,
        )

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        nv = self._split
        top, bottom = residual[:nv], residual[nv:]
        if transpose:
            pressure = self._s_inv.apply(bottom - self._dgt @ top, transpose=True)
            velocity = self._f_inv.apply(top - self._dt @ pressure, transpose=True)
        else:
            predictor = self._f_inv.apply(top)
            pressure = self._s_inv.apply(bottom - self._d @ predictor)
            velocity = predictor - self._dg @ pressure
        return np.concatenate([velocity, pressure])

    def destroy(self) -> None:
        for inverse in (self._f_inv, self._s_inv):
            if hasattr(inverse, "destroy"):
                inverse.destroy()


class _SimplePieces(NamedTuple):
    """One level's SIMPLE relaxation pieces, all frozen and all traced.

    ``n_velocity`` is the split point in the level's field-major vector; everything else is either an
    elementwise reciprocal or a sparse operator, so a sweep is diagonal scalings and matrix-vector
    products and nothing else -- no factorization, no triangular solve, no sequential dependency.
    """

    n_velocity: int
    f_diagonal_inverse: jnp.ndarray  # (n_velocity,) 1 / diag(F)
    dg: _CsrOperator  # diag(F)^-1 G, the velocity correction operator
    divergence: _CsrOperator  # D
    schur: _CsrOperator  # S = C - D diag(F)^-1 G
    schur_diagonal_inverse: jnp.ndarray  # (n_pressure,) 1 / diag(S)


def _simple_pieces(
    a: sp.csr_matrix, block_size: int, frobenius: bool = True, schur_frobenius: bool = False
) -> _SimplePieces:
    """Form one level's SIMPLE pieces from that level's assembled operator.

    Works on **any** level, including a Galerkin coarse operator, because it reads nothing but the
    matrix -- which is the property that decides whether a SIMPLE relaxation can be a smoother at all.
    The assembler-built Schur cannot: it needs the mesh, the Rhie--Chow coefficients and the boundary
    closures, none of which a coarse level has.
    """
    n_cells = a.shape[0] // block_size
    nv = (block_size - 1) * n_cells
    # The approximate velocity inverse the whole relaxation is built on. Jacobi (`1 / F_ii`) is what
    # this smoother used when it amplified; the Frobenius-optimal diagonal is the same object with a
    # derived per-row relaxation, which is measured to be worth 25x in a flat setting on this operator
    # -- and an under-relaxed sweep is exactly what an amplifying one is missing.
    f_inverse = _diagonal_approximate_inverse(a[:nv, :nv].tocsr(), frobenius)
    g_block = a[:nv, nv:].tocsr()
    d_block = a[nv:, :nv].tocsr()
    dg = (sp.diags(f_inverse) @ g_block).tocsr()
    schur = (a[nv:, nv:] - d_block @ dg).tocsr()
    schur_diagonal = schur.diagonal()
    # A zero Schur diagonal would make the pressure relaxation undefined. It does not arise on this
    # discretization -- Rhie-Chow damping gives the continuity row a genuine diagonal, and the
    # correction only strengthens it -- so this is a guard against a coarse level degenerating, not a
    # routine case to smooth over.
    if np.any(schur_diagonal == 0.0):
        raise ValueError("the Schur complement has a zero diagonal on some level.")
    schur_inverse = _diagonal_approximate_inverse(schur, schur_frobenius)
    if schur_frobenius:
        ratio = (
            schur_inverse * schur_diagonal
        )  # vs Jacobi's 1/S_ii, so this is the relaxation factor
        print(
            f"      Schur relaxation (Eq. 39): min {ratio.min():.3e} median {np.median(ratio):.3e} "
            f"max {ratio.max():.3e}  ({schur.data.shape[0] / schur.shape[0]:.0f} nnz/row)",
            flush=True,
        )
    return _SimplePieces(
        n_velocity=nv,
        f_diagonal_inverse=jnp.asarray(f_inverse),
        dg=_CsrOperator.from_scipy(dg),
        divergence=_CsrOperator.from_scipy(d_block),
        schur=_CsrOperator.from_scipy(schur),
        schur_diagonal_inverse=jnp.asarray(schur_inverse),
    )


def _simple_correction(
    pieces: _SimplePieces, residual, pressure_sweeps: int, pressure_omega: float
):
    """One SIMPLE correction for a level residual: the classical predictor / Schur / correct sequence.

    ``du* = diag(F)^-1 r_u``; ``dp`` from a few damped-Jacobi sweeps on ``S`` against
    ``r_p - D du*``; ``du = du* - diag(F)^-1 G dp``. Every operation is a diagonal scaling or a sparse
    matrix-vector product, which is the whole point: this is what an incomplete-LU sweep is *not*, and
    it is why it can run on an accelerator.

    The pressure solve is deliberately a fixed handful of sweeps rather than anything converged. As a
    smoother it only has to damp high-frequency error -- the coarse grid carries the smooth pressure
    mode, which is exactly the mode a SIMPLE Schur approximates worst. A fixed count also keeps the
    whole correction a constant **linear** map, which the non-flexible outer Krylov and the transposed
    adjoint both require.
    """
    nv = pieces.n_velocity
    velocity_residual, pressure_residual = residual[:nv], residual[nv:]
    predictor = pieces.f_diagonal_inverse * velocity_residual
    rhs = pressure_residual - pieces.divergence.apply(predictor)
    # The first sweep is peeled because it would otherwise multiply the Schur complement by a zero
    # vector: starting from p = 0, the update collapses to omega * S_diag^-1 * rhs. That is one of every
    # `pressure_sweeps` applications of the densest operator in the smoother, and it is not folded away
    # -- the sparse product runs at full cost against the zeros. Peeling it is bit-identical.
    if pressure_sweeps <= 0:
        pressure = jnp.zeros_like(rhs)
    else:
        pressure = pressure_omega * pieces.schur_diagonal_inverse * rhs
    for _ in range(pressure_sweeps - 1):
        pressure = pressure + pressure_omega * pieces.schur_diagonal_inverse * (
            rhs - pieces.schur.apply(pressure)
        )
    return jnp.concatenate([predictor - pieces.dg.apply(pressure), pressure])


class NativeSimpleInverse:
    """A JAX-native multigrid over the flow saddle whose LEVEL SMOOTHER is a SIMPLE relaxation.

    This is the arm every earlier one was not. Those replaced the flow block's preconditioner outright
    -- a flat block inverse with no hierarchy, no levels and no coarse-grid correction -- so their
    accuracy was capped by how well a single application approximates ``A^-1``, and for a SIMPLE-type
    method that cap is set by the Schur approximation's worst mode, the smooth global pressure mode.
    That is visible in the measurements as a residual floor almost insensitive to the pseudo-transient
    shift, which is the signature of an approximation ceiling rather than a conditioning one.

    Here the hierarchy is kept and SIMPLE replaces the incomplete-LU **smoother**. The division of
    labour is the point: the smoother only has to damp high-frequency error, where a local algebraic
    Schur is accurate, and the coarse solve carries the global mode, where it is not.

    Nothing in it is a host solver and nothing in it is sequential -- the setup is sparse products in
    ``scipy`` and the apply is diagonal scalings and sparse matrix-vector products in JAX.

    Parameters
    ----------
    block : scipy.sparse matrix
        The flow block, field-major, ``[u, v, w]`` then ``p``.
    n_fields : int
        Fields per cell, the aggregation's block size.
    cycles, sweeps : int
        V-cycles per application, and SIMPLE sweeps per level. Both fixed, so ``b -> x`` stays linear.
    pressure_sweeps, pressure_omega : int, float
        The damped-Jacobi sweeps used for the Schur solve inside one SIMPLE sweep, and their damping.
    omega : float
        Relaxation applied to the whole SIMPLE correction.
    """

    def __init__(
        self,
        block: sp.spmatrix,
        n_fields: int,
        *,
        cycles: int = 1,
        sweeps: int = 2,
        pressure_sweeps: int = 4,
        # Undamped, and the Frobenius diagonals below are why. Measured as a 2x2 at four sweeps: with
        # the Jacobi diagonal an undamped pressure sweep blows up (3.4e-01 against 6.8e-05), while with
        # the Frobenius one removing the damping HELPS (6.8e-05 -> 4.2e-05). The hand-set 0.7 and the
        # derived per-row relaxation were doing the same job, so stacking them over-damped -- but the
        # constant cannot simply be dropped, it can only be replaced.
        pressure_omega: float = 1.0,
        omega: float = 0.7,
        max_coarse: int = 2000,
        frobenius: bool = True,
        schur_frobenius: bool = True,
        levels: int = 2,
        aggressive: int = 1,
        strength_threshold: float = 0.0,
        orthonormal: bool = False,
        avoid_singletons: bool = False,
        # Every arm here has run UNSMOOTHED aggregation, which interpolates a coarse correction by
        # injecting it piecewise-constant over each aggregate. That is the standard explanation for a
        # hierarchy that works at two levels and gains nothing deeper: the interpolation error does not
        # fall as the grids coarsen. Smoothing the prolongator once with the operator is what makes
        # aggregation multigrid depth-independent.
        prolongation_smoothing: str = "none",
        equilibrate: bool = False,
    ) -> None:
        matrix = sp.csr_matrix(block)
        self._n_dofs = matrix.shape[0]
        self._hierarchy = build_convection_hierarchy(
            matrix,
            block_size=n_fields,
            max_coarse=max_coarse,
            mis_aggregation=True,
            # Depth and coarsening RATE are two halves of one choice. One aggressive level gives a
            # roughly hundredfold jump in a single step, so one prolongation from a ~860-equation
            # coarse space carries the whole error; more levels at a gentler rate ask less of each
            # interpolation. The recorded caution against depth here was derived for a damped-Jacobi
            # smoother on a convection-dominated operator, whose coarse-of-coarse operators it could
            # not damp -- that is not the smoother running now, so it is a reason to measure.
            aggressive_levels=aggressive,
            max_levels=levels,
            # The coarsening RATE, which is the knob the depth sweep pointed at. Aggregating only along
            # strong connections makes the aggregates smaller, so it lands a coarse grid between the
            # squared graph's ~106x and plain aggregation's ~21x -- and the coarse grid size is the
            # binding cost here, since the coarsest level is inverted DENSELY (O(n^3) to build,
            # O(n^2) to store), which is affordable at ~1000 equations and is not at 4300 and above.
            strength_threshold=strength_threshold,
            orthonormal_prolongation=orthonormal,
            avoid_singletons=avoid_singletons,
            prolongation_smoothing=prolongation_smoothing,
            equilibrate=equilibrate,
        )
        # Keyed by level size: the V-cycle recursion is unrolled at trace time, so the smoother is
        # handed the concrete level object and can look its pieces up by a static attribute. The
        # coarsest level solves directly and needs none.
        pieces = {}
        for level in self._hierarchy.levels:
            if level.coarse_inv is not None:
                continue
            operator = level.operator
            level_matrix = sp.csr_matrix(
                (
                    np.asarray(operator.data),
                    np.asarray(operator.indices),
                    np.asarray(operator.indptr),
                ),
                shape=operator.shape,
            )
            pieces[level.n] = _simple_pieces(
                level_matrix, level.block_size, frobenius, schur_frobenius
            )
        sizes = ", ".join(
            f"level {n}: S {p.schur.shape[0]} dofs / {p.schur.data.shape[0] / 1e6:.1f}M nnz"
            for n, p in pieces.items()
        )
        # Report the COARSE GRID SIZE and the coarsening ratio beside the level count. A level count
        # alone cannot distinguish a hierarchy that stopped because it hit its cap from one that
        # stopped because it reached its coarse limit, and it says nothing about whether a single
        # aggregation had to represent the error across a hundredfold jump.
        from aquaflux.solve.multigrid import _AGGREGATE_STATS

        for depth, stat in enumerate(_AGGREGATE_STATS[-(len(self._hierarchy.levels) - 1) :]):
            print(
                f"      aggregates level {depth}: {stat['aggregates']} of size "
                f"{stat['min']}/{stat['median']:.0f}/{stat['max']} (min/med/max), "
                f"spread {stat['spread']:.0f}x, {stat['singletons']} singletons",
                flush=True,
            )
        coarse = self._hierarchy.levels[-1]
        print(
            f"      native SIMPLE smoother: {len(self._hierarchy.levels)} levels, "
            f"fine {self._n_dofs} -> coarse {coarse.n} dofs "
            f"({self._n_dofs / max(coarse.n, 1):.0f}x, direct solve), {sizes}",
            flush=True,
        )

        def smooth(level, rhs, guess):
            piece = pieces[level.n]
            for _ in range(sweeps):
                correction = _simple_correction(
                    piece, rhs - _operator_matvec(level, guess), pressure_sweeps, pressure_omega
                )
                guess = guess + omega * correction
            return guess

        ops = _smoothed_ops(smooth)
        cycle = jax.jit(lambda r: self._hierarchy.fixed_cycle_solve(r, cycles, ops))
        self._solve = cycle
        self._transpose = jax.linear_transpose(cycle, jnp.zeros(self._n_dofs, dtype=jnp.float64))

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        vector = jnp.asarray(residual, dtype=jnp.float64)
        out = self._transpose(vector)[0] if transpose else self._solve(vector)
        return np.asarray(out, dtype=np.float64)

    def destroy(self) -> None:
        """Nothing to release -- plain arrays, no host solver handles."""


class BlockTransformedInverse:
    """A native hierarchy over the flow saddle LEFT-TRANSFORMED so its pressure block is a real Schur.

    Every other arm here attacks the smoother. This attacks what the coarse grid is asked to coarsen,
    which is a different layer of the same problem and the one a smoother cannot reach: a multigrid
    applied to the raw saddle asks its coarse space to represent a ``(p, p)`` block that is only the
    Rhie--Chow damping, and no relaxation repairs a coarse correction built from the wrong operator.

    Write the block as ``A = [[F, G], [D, C]]`` over the field-major split ``[u, v, w] | [p]`` and apply
    the unit block-lower-triangular ``P = [[I, 0], [-D diag(F)^-1, I]]``::

        B = P A = [[F,                      G                    ],
                   [D - D diag(F)^-1 F,     C - D diag(F)^-1 G   ]]

    The trailing diagonal block is now the SIMPLE Schur complement -- an elliptic pressure operator --
    where the untransformed one was the damping term. ``P`` is unit triangular, hence invertible for
    any ``F`` with a nonzero diagonal, so the transform is **exact**: preconditioning ``A`` by
    ``M = M_B . P`` is a genuine preconditioner, and with ``M_B = B^-1`` it is exactly ``A^-1``.

    Cost is the open question rather than correctness. Both corrections are sparse triple products; the
    ``(p, p)`` one was measured at 308 nonzeros per row on this case, denser than the flow block itself,
    and the ``(p, u)`` one is unmeasured. ``exact`` selects between the two forms: ``True`` builds ``B``
    as above; ``False`` leaves the ``(p, u)`` block as ``D``, which is no longer an exact transform but
    is still a fixed linear operator and therefore still a legal preconditioner -- it simply
    approximates a different matrix, and the arms measure whether that costs anything. Both print the
    nonzero counts, because if the transform is not affordable that is the result.

    **Adjoint-legal.** ``M = M_B . P`` composes two fixed linear maps, so ``M^T = P^T . M_B^T`` in
    closed form: apply the transposed hierarchy, then ``y_u <- y_u - (D diag(F)^-1)^T y_p``.
    """

    def __init__(
        self,
        block: sp.spmatrix,
        n_fields: int,
        *,
        exact: bool = True,
        cycles: int = 1,
        sweeps: int = 4,
        max_coarse: int = 2000,
        damped: bool = True,
        equilibrate: bool = True,
    ) -> None:
        matrix = sp.csr_matrix(block)
        self._n_dofs = matrix.shape[0]
        n_cells = self._n_dofs // n_fields
        self._split = nv = (n_fields - 1) * n_cells
        f_block = matrix[:nv, :nv].tocsr()
        g_block = matrix[:nv, nv:].tocsr()
        d_block = matrix[nv:, :nv].tocsr()
        c_block = matrix[nv:, nv:].tocsr()
        f_diagonal = f_block.diagonal()
        if not np.all(np.isfinite(f_diagonal)) or np.any(f_diagonal == 0.0):
            raise ValueError(
                "the velocity block has a zero or non-finite diagonal; P is undefined."
            )
        # D diag(F)^-1, the one factor both corrections and the transform's own apply share.
        self._dfd = (d_block @ sp.diags(1.0 / f_diagonal)).tocsr()
        self._dfd_t = self._dfd.T.tocsr()
        schur = (c_block - self._dfd @ g_block).tocsr()
        lower = (d_block - self._dfd @ f_block).tocsr() if exact else d_block
        transformed = sp.bmat([[f_block, g_block], [lower, schur]], format="csr")
        print(
            f"      block transform ({'exact' if exact else 'Schur-only'}): "
            f"(p,p) {c_block.nnz / 1e6:.1f}M -> {schur.nnz / 1e6:.1f}M nnz, "
            f"(p,u) {d_block.nnz / 1e6:.1f}M -> {lower.nnz / 1e6:.1f}M nnz, "
            f"total {matrix.nnz / 1e6:.1f}M -> {transformed.nnz / 1e6:.1f}M",
            flush=True,
        )
        self._hierarchy = build_convection_hierarchy(
            transformed,
            block_size=n_fields,
            max_coarse=max_coarse,
            mis_aggregation=True,
            aggressive_levels=1,
            prolongation_smoothing="none",
            # Equilibrate by DEFAULT here, unlike every other native arm, and the reason is the
            # transform itself: the pressure rows of `B` are sparse triple products, so their
            # magnitudes bear no relation to the velocity rows they now sit beside. The nodal smoother
            # inverts a cell block assembled from both, and the coarse operator is built from the same
            # rows, so an unscaled `B` is not the same kind of object an unscaled `A` was.
            equilibrate=equilibrate,
        )
        # Damped by default, which is the OPPOSITE of the turbulence block's measured preference and
        # the same as the saddle's: an undamped Richardson assumes a spectrum on the positive real
        # axis, which this operator does not have.
        cycle = jax.jit(
            lambda h, r: convection_multigrid_solve(
                h,
                r,
                cycles=cycles,
                sweeps=sweeps,
                omega=0.8 if damped else 1.0,
                spectral_damping=damped,
            )
        )
        self._solve = lambda r: cycle(self._hierarchy, r)
        self._transpose_solve = lambda r: jax.linear_transpose(
            lambda v: cycle(self._hierarchy, v), jnp.zeros(self._n_dofs, dtype=jnp.float64)
        )(r)[0]

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        nv = self._split
        if transpose:
            # M^T = P^T . M_B^T: invert first, then apply the transposed transform.
            out = np.asarray(self._transpose_solve(jnp.asarray(residual, dtype=jnp.float64)))
            corrected = out.copy()
            corrected[:nv] = out[:nv] - self._dfd_t @ out[nv:]
            return corrected
        # M = M_B . P: transform the residual, then invert the transformed operator.
        transformed = np.asarray(residual, dtype=np.float64).copy()
        transformed[nv:] = transformed[nv:] - self._dfd @ transformed[:nv]
        return np.asarray(self._solve(jnp.asarray(transformed, dtype=jnp.float64)))

    def destroy(self) -> None:
        """Nothing to release -- plain arrays, no host solver handles."""


def _diagonal_approximate_inverse(f_block: sp.csr_matrix, frobenius: bool) -> np.ndarray:
    """The diagonal approximate inverse of a block, Jacobi or Frobenius-optimal.

    Written for the velocity block but specific to nothing about it -- the derivation reads only the
    rows of whatever matrix it is handed, so it applies equally to the Schur complement, whose own
    relaxation faces the same choice and whose diagonal is a **worse** approximation to its inverse
    (the Schur here carries some 300 nonzeros per row against the flow block's 227).

    Jacobi takes ``1 / F_ii``. The Frobenius-optimal diagonal instead minimizes ``||I - F~^-1 F||_F``,
    which for a diagonal unknown decouples row by row and has the closed form ``F_ii / ||F_i||^2``.

    **Written as a ratio the two differ by ``F_ii^2 / ||F_i||^2``, which is at most one and is the
    fraction of row i's energy sitting on its diagonal — so the optimal choice is Jacobi with an
    automatic, per-row under-relaxation.** That is worth stating plainly because the hand-tuned version
    of the same quantity appears throughout this solver: a velocity-row under-relaxation, a
    preconditioner-only shift floor on the velocity rows, and the relative velocity-row relaxation the
    closest published work on this discretization never manages to drop. Here it is derived rather than
    chosen, which is the paper's own argument for why its formulation needs no under-relaxation at all.
    """
    diagonal = f_block.diagonal()
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0.0):
        raise ValueError("the velocity block has a zero or non-finite diagonal.")
    if not frobenius:
        return 1.0 / diagonal
    # Row 2-norms squared, straight off the CSR values.
    squared = np.asarray(f_block.multiply(f_block).sum(axis=1)).ravel()
    if np.any(squared == 0.0):
        raise ValueError("the velocity block has an empty row; the optimal inverse is undefined.")
    return diagonal / squared


class MultiStepSaddleInverse:
    """The multi-step block-LU saddle preconditioner (Jemcov & Maruszewski), on the flow block.

    Every SIMPLE-shaped arm here so far has been *inconsistent* in the sense that paper identifies: it
    forms the Schur complement from an approximate ``F~^-1`` but then solves for the velocity with the
    real ``F`` (an incomplete-LU sweep or a multigrid cycle). The paper's argument is that this
    mismatch is precisely what obliges SIMPLE to under-relax — the two halves of the step are built
    from different operators — and that using the same ``F~^-1`` in both places removes the need.
    Here the velocity update is ``F~^-1`` applied as a diagonal, so the formulation is consistent by
    construction and carries no relaxation parameter.

    One step, from ``u = 0``, is::

        v = u + F~^-1 (f - F u)
        solve  S~ p = g - D v          with   S~ = C - D F~^-1 G
        u = v - F~^-1 G p

    and ``steps`` of it form a stationary iteration on the regular splitting ``A = A~ - N``, so the
    error from approximating ``F`` by ``F~`` is swept out across steps rather than baked into the
    operator. ``algorithm=2`` seeds ``u`` with a real solve of ``F u = f`` at ``p = 0`` first, which the
    paper reports as much the better start: from ``u = 0`` many eigenvalues sit near zero and converge
    slowly, whereas the seeded start leaves them all above one.

    **Two structural differences from the paper, and they must be stated with any result.** Its system
    has a ``(p, p)`` block that is exactly zero and ``D = G^T`` (inf-sup-stable unequal-order finite
    elements); ours has a nonzero Rhie--Chow damping block and a nonsymmetric ``D``. So the Schur here
    is ``C - D F~^-1 G`` rather than ``-D F~^-1 G``, and — more importantly — the paper's optimality
    result (two distinct eigenvalues, hence a Krylov method converging in two iterations) rests on that
    zero block and does **not** carry over. What carries over is the algorithm, not its spectrum.

    Everything is a diagonal scaling, a sparse matrix-vector product, or a multigrid cycle on a scalar
    elliptic operator, so there is no triangular solve anywhere and a fixed ``steps`` keeps the whole
    map linear and transposable.
    """

    def __init__(
        self,
        block: sp.spmatrix,
        n_fields: int,
        *,
        steps: int = 1,
        frobenius: bool = True,
        algorithm: int = 1,
        cycles: int = 1,
        sweeps: int = 4,
        max_coarse: int = 2000,
    ) -> None:
        matrix = sp.csr_matrix(block)
        self._n_dofs = matrix.shape[0]
        n_cells = self._n_dofs // n_fields
        self._split = nv = (n_fields - 1) * n_cells
        f_block = matrix[:nv, :nv].tocsr()
        g_block = matrix[:nv, nv:].tocsr()
        d_block = matrix[nv:, :nv].tocsr()
        c_block = matrix[nv:, nv:].tocsr()
        f_inverse = _diagonal_approximate_inverse(f_block, frobenius)
        schur = (c_block - d_block @ sp.diags(f_inverse) @ g_block).tocsr()
        jacobi = 1.0 / f_block.diagonal()
        relaxation = f_inverse / jacobi  # the per-row factor Eq. (39) applies on top of Jacobi
        print(
            f"      multi-step ({'Frobenius' if frobenius else 'Jacobi'} inverse, {steps} step(s), "
            f"algorithm {algorithm}): S {schur.shape[0]} dofs / {schur.data.shape[0] / 1e6:.1f}M nnz, "
            f"per-row relaxation min {relaxation.min():.3e} median {np.median(relaxation):.3e} "
            f"max {relaxation.max():.3e}",
            flush=True,
        )
        self._steps, self._algorithm = steps, algorithm
        self._f_inverse = jnp.asarray(f_inverse)
        self._f = _CsrOperator.from_scipy(f_block)
        self._g = _CsrOperator.from_scipy(g_block)
        self._d = _CsrOperator.from_scipy(d_block)
        schur_hierarchy = build_convection_hierarchy(
            schur,
            block_size=1,
            max_coarse=max_coarse,
            mis_aggregation=True,
            aggressive_levels=1,
            prolongation_smoothing="none",
            equilibrate=True,
        )
        velocity_hierarchy = (
            build_convection_hierarchy(
                f_block,
                block_size=n_fields - 1,
                max_coarse=max_coarse,
                mis_aggregation=True,
                aggressive_levels=1,
                prolongation_smoothing="none",
                equilibrate=True,
            )
            if algorithm == 2
            else None
        )

        def cycle(hierarchy, rhs):
            return convection_multigrid_solve(
                hierarchy, rhs, cycles=cycles, sweeps=sweeps, omega=0.8, spectral_damping=True
            )

        def apply(residual):
            velocity_rhs, pressure_rhs = residual[:nv], residual[nv:]
            velocity = (
                cycle(velocity_hierarchy, velocity_rhs)
                if velocity_hierarchy is not None
                else jnp.zeros_like(velocity_rhs)
            )
            pressure = jnp.zeros_like(pressure_rhs)
            for _ in range(self._steps):
                predictor = velocity + self._f_inverse * (velocity_rhs - self._f.apply(velocity))
                pressure = cycle(schur_hierarchy, pressure_rhs - self._d.apply(predictor))
                velocity = predictor - self._f_inverse * self._g.apply(pressure)
            return jnp.concatenate([velocity, pressure])

        jitted = jax.jit(apply)
        self._solve = jitted
        self._transpose = jax.linear_transpose(jitted, jnp.zeros(self._n_dofs, dtype=jnp.float64))

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    def apply(self, residual: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        vector = jnp.asarray(residual, dtype=jnp.float64)
        out = self._transpose(vector)[0] if transpose else self._solve(vector)
        return np.asarray(out, dtype=np.float64)

    def destroy(self) -> None:
        """Nothing to release -- plain arrays, no host solver handles."""


def _sub_inverse(kind):
    """``(sub_matrix, n_fields) -> inverse`` for one half of a velocity/pressure decomposition.

    ``petsc`` is the structure control -- it keeps the incomplete-LU sweep, so an arm using it measures
    what the *decomposition* is worth while changing nothing else. ``native`` is the candidate, damped
    because the saddle's damping preference is the opposite of the transported scalars' and these
    sub-blocks inherit that class's default otherwise.
    """
    if kind == "petsc":

        def build(sub, n_sub_fields):
            return build_amg_vcycle(
                sub,
                n_sub_fields,
                smoother_sweeps=compare.SWEEPS,
                smoother_fill_levels=compare.FILL_LEVELS,
                coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            )

        return build
    # An unrecognized half must RAISE, not fall through to a default. This used to return the native
    # inverse for any string it did not recognize, so `simple-pestc-native` -- or any third kind added
    # later -- would have run a native block under an arm labelled PETSc. An arm that measures something
    # other than its label is the worst failure mode a study like this has, because every check
    # downstream of it still passes.
    if kind != "native":
        raise ValueError(f"unknown sub-block inverse {kind!r}; use 'petsc' or 'native'.")

    def build(sub, n_sub_fields):
        return NodalNativeInverse(
            sub,
            n_sub_fields,
            cycles=1,
            sweeps=4,
            max_coarse=compare.COARSE_EQ_LIMIT,
            equilibrate=False,
            spectral_damping=True,
        )

    return build


def _trailing_inverse(spec):
    """The trailing block's inverse factory: a PETSc V-cycle by smoother name, or a JAX-native cycle."""
    if spec in SMOOTHERS:
        return None  # the builder's own V-cycle, configured through `trailing_options`
    if spec.startswith("blockjacobi"):
        # `blockjacobi` or `blockjacobiN` for N sweeps.
        sweeps = int(spec.removeprefix("blockjacobi") or 1)

        def build(block, n_group_fields):
            return BlockJacobiInverse(block, n_group_fields, sweeps)

        return build
    if spec == "air":

        def build(block, n_group_fields):
            hierarchy = build_air_hierarchy(block.tocsr())
            return JaxNativeBlockInverse(
                lambda b: air_multigrid_solve(hierarchy, b), block.shape[0]
            )

        return build
    if spec == "twolevel":

        def build(block, n_group_fields):
            hierarchy = build_convection_hierarchy(block.tocsr())
            return JaxNativeBlockInverse(
                lambda b: convection_multigrid_solve(hierarchy, b), block.shape[0]
            )

        return build
    raise ValueError(f"unknown trailing inverse {spec!r}")


def _leading_inverse(spec):
    """The leading (flow saddle) block's inverse factory: a PETSc V-cycle by smoother name, or a native one.

    The sibling of :func:`_trailing_inverse`, and the reason it did not exist until now is worth stating.
    Every arm ever run on the flow block has been a PETSc GAMG hierarchy with a smoother swapped in
    through ``leading_options``; the JAX-native hierarchy has only ever been given ``[k, omega]``. So the
    question the whole native-preconditioner effort turns on -- whether a hierarchy this package can
    write itself is viable on the *saddle* -- has never been asked, and it is one dispatcher away.

    ``nativeN`` selects N smoother sweeps (default 4), and a trailing ``d`` damps the smoother
    (``native4d``). The other settings are the ones measured to reproduce a PETSc GAMG V-cycle on the
    turbulence block -- one aggressive squared-graph coarsening level and an unsmoothed tentative
    prolongation -- since the point is to compare hierarchies rather than to re-discover those.
    ``equilibrate`` is left off, matching the case default: on a marched solve it is not a conditioning
    question but a change to the coarse operator, and an A/B differing only in that flag came out
    opposite.

    **The damping is a knob here and not a default because its sign REVERSES between the two blocks.**
    On the transported-scalar pair an undamped Richardson was the fix, worth 10 restart cycles against
    2, on the argument that ``D^-1 A`` has a unit diagonal so a spectral factor can only under-relax.
    That argument assumes the spectrum sits on the positive real axis. The saddle's does not, and
    measured on this block the ranking inverts: damped point-block Jacobi leaves a true residual of
    9.7e-02 where the undamped one leaves 2.4e-01. So a native arm inherits the turbulence block's
    undamped default into a regime that penalizes it, and both spellings have to be run.
    """
    if spec in SMOOTHERS:
        return None  # the builder's own V-cycle, configured through `leading_options`
    if spec.startswith("multistep"):
        # Jemcov & Maruszewski: the consistent block-LU saddle preconditioner. `-jacobi` selects the
        # plain Jacobi inverse instead of the Frobenius-optimal one, which is the control that isolates
        # what Eq. (39) is worth; `-alg2` seeds the velocity with a real solve of F u = f.
        rest = spec.removeprefix("multistep")
        frobenius = "-jacobi" not in rest
        algorithm = 2 if "-alg2" in rest else 1
        steps = int(rest.replace("-jacobi", "").replace("-alg2", "") or 1)

        def build(block, n_group_fields):
            return MultiStepSaddleInverse(
                block,
                n_group_fields,
                steps=steps,
                frobenius=frobenius,
                algorithm=algorithm,
                max_coarse=compare.COARSE_EQ_LIMIT,
            )

        return build
    if spec.startswith("transform"):
        # Left block transform, then a native hierarchy on the TRANSFORMED operator.
        rest = spec.removeprefix("transform")
        exact = "-schur" not in rest
        undamped = "-undamped" in rest
        raw = rest.replace("-schur", "").replace("-undamped", "").replace("-raw", "")
        sweeps = int(raw or 4)

        def build(block, n_group_fields):
            return BlockTransformedInverse(
                block,
                n_group_fields,
                exact=exact,
                sweeps=sweeps,
                max_coarse=compare.COARSE_EQ_LIMIT,
                damped=not undamped,
                equilibrate="-raw" not in rest,
            )

        return build
    if spec.startswith("simplesmooth"):
        # The SIMPLE relaxation as a level SMOOTHER inside a native hierarchy, not as a flat inverse.
        rest = spec.removeprefix("simplesmooth")
        frobenius = "-jacobi" not in rest
        schur_frobenius = "-sjacobi" not in rest
        levels = next((int(t[2:]) for t in ("-L3", "-L4", "-L5") if t in rest), 2)
        # `-cNNN` lowers the size at which coarsening stops. Depth alone cannot go deeper than the
        # first level that falls under this limit, so raising `-L` without lowering it is a no-op:
        # the loop breaks on `size <= max_coarse` before it ever reaches the level cap.
        coarse_token = next((t for t in ("-c1000", "-c500", "-c200", "-c50") if t in rest), None)
        # `-psN` sets the inner pressure relaxations per SIMPLE sweep. Never varied before this: every
        # arm ran at four, while the OUTER sweep count was swept 4/8/16 -- so the single largest term in
        # the smoother's cost is the one axis that was held fixed. Four inner sweeps are two thirds of an
        # outer sweep, so they buy relaxation far cheaper than an outer sweep does, and the question is
        # where the (outer, inner) pair sits rather than whether either alone is too high.
        pressure_sweeps = next(
            (int(t[3:]) for t in ("-ps1", "-ps2", "-ps3", "-ps6") if t in rest), 4
        )
        max_coarse = int(coarse_token[2:]) if coarse_token else compare.COARSE_EQ_LIMIT
        aggressive = 0 if "-a0" in rest else 1
        threshold = next((float(t[2:]) / 100 for t in ("-t10", "-t25", "-t50") if t in rest), 0.0)
        orthonormal = "-qr" in rest
        avoid_singletons = "-ns" in rest
        # `-sm` is the historical formula, `-smstd` the textbook sigma_max one. Both must be read
        # BEFORE the tokens are stripped, and `-smstd` before `-sm`, since one contains the other.
        if "-smstd" in rest:
            prolongation_smoothing = "standard"
        elif "-sm" in rest:
            prolongation_smoothing = "symmetric-part"
        else:
            prolongation_smoothing = "none"
        equilibrate = "-eq" in rest
        # `-p1` removes the explicit pressure relaxation. It exists to test whether Eq. (39) on the
        # Schur was harmful in itself or only because it stacked on top of an existing damping: the
        # velocity predictor carries no relaxation of its own, which is why the same substitution was
        # worth four orders there and negative here.
        pressure_omega = 0.7 if "-p07" in rest else 1.0
        for token in (
            "-smstd",
            "-sm",
            "-eq",
            "-qr",
            "-ns",
            "-jacobi",
            "-sjacobi",
            "-L3",
            "-L4",
            "-L5",
            "-ps1",
            "-ps2",
            "-ps3",
            "-ps6",
            "-c1000",
            "-c500",
            "-c200",
            "-c50",
            "-a0",
            "-p07",
            "-t10",
            "-t25",
            "-t50",
        ):
            rest = rest.replace(token, "")
        sweeps = int(rest or 2)

        def build(block, n_group_fields):
            return NativeSimpleInverse(
                block,
                n_group_fields,
                sweeps=sweeps,
                pressure_sweeps=pressure_sweeps,
                max_coarse=max_coarse,
                frobenius=frobenius,
                schur_frobenius=schur_frobenius,
                levels=levels,
                aggressive=aggressive,
                strength_threshold=threshold,
                orthonormal=orthonormal,
                avoid_singletons=avoid_singletons,
                pressure_omega=pressure_omega,
                prolongation_smoothing=prolongation_smoothing,
                equilibrate=equilibrate,
            )

        return build
    if spec.startswith("simple-"):
        # The velocity/pressure SIMPLE decomposition, with a Schur complement. The spec names the two
        # halves independently -- `simple-native-petsc` is a native velocity block against a host
        # pressure V-cycle -- so a failure can be attributed to one half instead of to the pair.
        halves = spec.removeprefix("simple-").split("-")
        velocity_kind = halves[0]
        pressure_kind = halves[1] if len(halves) > 1 else halves[0]
        inner_velocity = _sub_inverse(velocity_kind)
        inner_pressure = _sub_inverse(pressure_kind)

        def build(block, n_group_fields):
            return AlgebraicSimpleInverse(block, n_group_fields, inner_velocity, inner_pressure)

        return build
    if spec.startswith("nested-"):
        # The same partition WITHOUT a Schur complement: the trailing pressure operator is the raw
        # (p, p) block. It is the control that separates "stop asking one hierarchy to coarsen a
        # saddle" from "build the Schur complement", and it is not a straw man on this discretization
        # -- Rhie-Chow damping already makes the (p, p) block elliptic.
        halves = spec.removeprefix("nested-").split("-")
        inner_velocity = _sub_inverse(halves[0])
        inner_pressure = _sub_inverse(halves[1] if len(halves) > 1 else halves[0])

        def build(block, n_group_fields):
            n_cells = sp.csr_matrix(block).shape[0] // n_group_fields
            return build_block_triangular_field_split(
                block,
                FieldGroups(
                    n_cells=n_cells,
                    n_leading_fields=n_group_fields - 1,  # u, v, w
                    n_trailing_fields=1,  # p
                ),
                leading_inverse=inner_velocity,
                trailing_inverse=inner_pressure,
            )

        return build
    if spec.startswith("native"):
        damped = spec.endswith("d")
        sweeps = int(spec.removeprefix("native").removesuffix("d") or 4)

        def build(block, n_group_fields):
            return NodalNativeInverse(
                block,
                n_group_fields,
                cycles=1,
                sweeps=sweeps,
                max_coarse=compare.COARSE_EQ_LIMIT,
                equilibrate=False,
                spectral_damping=damped,
            )

        return build
    raise ValueError(f"unknown leading inverse {spec!r}")


def field_split(
    shifted,
    groups,
    n_fields,
    flow_smoother,
    turbulence_smoother,
    *,
    flow_first,
    leading_inverse=None,
    trailing_inverse=None,
):
    """A hierarchy per field group, retaining one triangle of the coupling between them.

    ``trailing_inverse`` overrides the trailing half wholesale. It is threaded rather than resolved from
    the arm name because the interesting override -- the JAX-native scalar hierarchies -- has to be built
    from the coupled system and its state, which the name alone cannot supply. The leading half has no
    such need, so its override is resolved from the name by :func:`_leading_inverse`.
    """
    return MonolithicAmgPreconditioner(
        build_block_triangular_field_split(
            shifted,
            groups,
            flow_first=flow_first,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            leading_options=SMOOTHERS.get(flow_smoother) or None,
            trailing_options=SMOOTHERS.get(turbulence_smoother) or None,
            leading_inverse=(
                leading_inverse if leading_inverse is not None else _leading_inverse(flow_smoother)
            ),
            trailing_inverse=(
                trailing_inverse
                if trailing_inverse is not None
                else _trailing_inverse(turbulence_smoother)
            ),
        )
    )


def block_simple_arms(coupled, pc_state, pc_beta):
    """Arms whose flow-block inverse is the shipped block-SIMPLE preconditioner, not a matrix slice.

    These cannot live in :data:`ARMS` because they are not a function of the assembled block: the
    velocity/pressure Schur is built from the *assembler* -- the Rhie--Chow coefficients, the mass flux
    and the boundary closures -- so they need the coupled system and the state, which an arm key alone
    cannot supply. Same reason ``trailing_inverse`` is threaded rather than named.

    Worth running despite the recorded verdict against block-SIMPLE, and the reason is what changed. That
    verdict -- the Schur approximation is the wall at high Reynolds number, and inverting it *more*
    accurately makes the preconditioner worse -- was measured with ``k`` and ``omega`` still inside the
    preconditioned block, as a block-*diagonal* coupled preconditioner. Here the transported scalars are
    split off and carry their own hierarchy, so what block-SIMPLE is being asked to precondition is the
    thing it was designed for and nothing else. MSIMPLER specifically replaces the ``V/a_P`` Schur scaling
    with a velocity-independent mass-matrix diagonal, which is what stops it degrading as convection
    strengthens, so it is the variant with a reason to survive the regime that killed the others.

    **The effective diagonal assumes the shipped shift basis.** The march inverts ``a_P + beta * d``, and
    with the default basis ``d = a_P``, so the shifted diagonal is ``a_P (1 + beta)``. That identity is
    what is used here; it is exact for the shipped configuration and wrong for any other ``ShiftBasis``.
    """
    # The assembler has to carry the EDDY VISCOSITY, and building it from `coupled.momentum` directly
    # does not. The closure rides on its own leaf, applied inside the coupled residual, so the bare
    # assembler is the MOLECULAR-viscosity operator -- and on a developed Reynolds-averaged field the two
    # differ by the eddy-viscosity ratio, which peaks near 150 here. A velocity block built from it
    # inverts a laminar operator against a turbulent Jacobian, and the arm measures that mismatch rather
    # than the Schur it is supposed to be testing. This is the same sequence the shipped coupled shift
    # policy runs, and it must stay the same sequence.
    flow_ref, k_ref, omega_ref = coupled.physical_fields(pc_state)
    closure = coupled.turbulence.closure_fields(
        coupled.momentum.velocity_fields(flow_ref), k_ref, omega_ref
    )
    momentum = coupled.momentum.with_eddy_viscosity(closure.nu_t)
    flow = flow_ref
    n_flow = int(flow.shape[0])

    def arm(scaling, frobenius=False, **overrides):
        def build(shifted, groups, n_fields):
            block = BlockPreconditioner.build(
                momentum,
                **{
                    "velocity": "convection",
                    "reference_state": flow,
                    "schur_scaling": scaling,
                    # Aggregate along strong connections, as the shipped coupled path does. Left at the
                    # default here once, which is a different preconditioner from the one that ships.
                    "strength_threshold": 0.25,
                    **overrides,
                },
            )
            a_p = jax.lax.stop_gradient(block.frozen_momentum_diagonal(flow) * (1.0 + pc_beta))
            if frobenius:
                # Replace the momentum diagonal by the Frobenius-optimal EFFECTIVE diagonal. The block
                # preconditioner's approximate velocity inverse is `1 / a_P`, so the analogue of the
                # optimal `F_ii / ||F_i||^2` is an effective diagonal `||F_i||^2 / F_ii`; since
                # `||F_i||^2 >= F_ii^2` that is always the larger of the two, which is the automatic
                # per-row under-relaxation stated as a diagonal rather than as a relaxation factor.
                #
                # Both halves are taken from the ASSEMBLED velocity block so the ratio is
                # self-consistent. That does mean this arm replaces the assembler's lagged `a_P` with a
                # Jacobian-derived one, which is a second change riding along with the first -- the
                # `-diag` control arm below isolates it by passing `F_ii` alone.
                lead, _, _, _ = groups.blocks(shifted)
                nv = (groups.n_leading_fields - 1) * groups.n_cells
                velocity = sp.csr_matrix(lead)[:nv, :nv]
                per_dof = np.asarray(velocity.multiply(velocity).sum(axis=1)).ravel()
                diagonal = velocity.diagonal()
                effective = diagonal if frobenius == "diag" else per_dof / diagonal
                # `a_P` is the isotropic per-cell diagonal, so reduce the per-component rows the same way.
                per_cell = effective.reshape(groups.n_leading_fields - 1, groups.n_cells).mean(
                    axis=0
                )
                ratio = np.asarray(a_p) / per_cell
                print(
                    f"      Frobenius a_P: min {per_cell.min():.3e} median {np.median(per_cell):.3e}; "
                    f"relaxation vs a_P min {ratio.min():.3e} median {np.median(ratio):.3e} "
                    f"max {ratio.max():.3e}",
                    flush=True,
                )
                a_p = jax.lax.stop_gradient(jnp.asarray(per_cell) * (1.0 + pc_beta))
            matvec = jax.jit(block.apply_at(flow, a_p))
            return field_split(
                shifted,
                groups,
                n_fields,
                "ilu0",  # unused: the leading inverse below replaces the V-cycle wholesale
                "ilu0",
                flow_first=True,
                leading_inverse=lambda sub, n_sub: JaxNativeBlockInverse(matvec, n_flow),
            )

        return build

    return (
        (
            "split simple-block/ilu0",
            "split flow-first, block-SIMPLE (a_P Schur) on flow",
            arm("simple"),
        ),
        (
            "split msimpler/ilu0",
            "split flow-first, block-SIMPLE (MSIMPLER Schur) on flow",
            arm("msimpler"),
        ),
        # Eq. (39) in the shipped block-SIMPLE: replace the momentum diagonal by the Frobenius-optimal
        # effective one. `-frobdiag` is the control that passes the assembled block's own `F_ii`
        # instead, isolating "the optimal formula" from "a Jacobian-derived diagonal rather than the
        # assembler's lagged one" -- two changes that would otherwise ride together.
        (
            "split msimpler-frob/ilu0",
            "split flow-first, block-SIMPLE (MSIMPLER) with the Frobenius-optimal a_P",
            arm("msimpler", frobenius="optimal"),
        ),
        (
            "split msimpler-frobdiag/ilu0",
            "split flow-first, block-SIMPLE (MSIMPLER) with the assembled F_ii as a_P",
            arm("msimpler", frobenius="diag"),
        ),
        # The CONSISTENT vehicle. With the `simple` scaling the Schur uses the same `a_p` the velocity
        # block inverts at, so a change to that diagonal reaches both halves -- which is the condition
        # the optimal inverse is derived under. Under `msimpler` the Schur uses a frozen mass diagonal
        # instead, so changing `a_p` there only over-damps the velocity solve and cannot test it.
        (
            "split simple-frob/ilu0",
            "split flow-first, block-SIMPLE (a_P Schur) with the Frobenius-optimal a_P",
            arm("simple", frobenius="optimal"),
        ),
        (
            "split simple-frobdiag/ilu0",
            "split flow-first, block-SIMPLE (a_P Schur) with the assembled F_ii as a_P",
            arm("simple", frobenius="diag"),
        ),
        # The V-cycle ladder, which is a DIAGNOSTIC and not a tuning sweep: it separates two causes of a
        # stall that need opposite fixes. If the arm improves with more inner cycles, the sub-solves are
        # the limit and the fix is to strengthen them -- cheap, and something a native hierarchy can do.
        # If it plateaus, or worsens, the limit is the Schur APPROXIMATION itself and no amount of inner
        # accuracy reaches it, because inverting the wrong operator more exactly is not progress.
        #
        # That second outcome is what was measured for this preconditioner as the whole COUPLED inverse
        # (velocity cycles inert, Schur cycles strictly worse). Whether it still holds with the
        # transported scalars split off is the open question, and it is the one that decides whether
        # MSIMPLER is worth pursuing on the flow block at all.
        (
            "split msimpler2/ilu0",
            "split flow-first, block-SIMPLE (MSIMPLER Schur) on flow, 2 V-cycles",
            arm("msimpler", v_cycles=2),
        ),
        (
            "split msimpler4/ilu0",
            "split flow-first, block-SIMPLE (MSIMPLER Schur) on flow, 4 V-cycles",
            arm("msimpler", v_cycles=4),
        ),
    )


#: ``(key, label, builder)``. The builder takes the shifted matrix rather than closing over it, so nothing
#: holds a reference to a multi-gigabyte operator the run wants to free between operating points.
ARMS = (
    # The control, and the two Jacobi-class smoothers applied to the SIX-field block -- the comparison
    # that says whether the split changes the answer for them.
    ("mono/ilu0", "monolithic, ILU(0)", lambda m, g, n: monolithic(m, g, n, "ilu0")),
    ("mono/cheb", "monolithic, Chebyshev", lambda m, g, n: monolithic(m, g, n, "chebyshev")),
    ("mono/jac", "monolithic, damped Jacobi", lambda m, g, n: monolithic(m, g, n, "jacobi")),
    # The split itself, both triangles, on the shipped smoother -- does ordering the coupling help at all?
    (
        "split flow/ilu0",
        "split flow-first, ILU(0) both",
        lambda m, g, n: field_split(m, g, n, "ilu0", "ilu0", flow_first=True),
    ),
    (
        "split turb/ilu0",
        "split turbulence-first, ILU(0) both",
        lambda m, g, n: field_split(m, g, n, "ilu0", "ilu0", flow_first=False),
    ),
    # The asymmetric arms, and the ones a split is really FOR: keep the incomplete-LU sweep on the saddle,
    # where it is known to be needed, and relax the smoother only on the transported scalars. That block
    # is not a saddle -- it is a two-field advection-diffusion-reaction pair with a genuine diagonal -- so
    # a Jacobi-class smoother ought to serve it, and if it does the sequential triangular solve is confined
    # to four fields instead of six. This is the configuration that decides whether the incomplete-LU
    # requirement can be LOCALIZED, which is a weaker but far more reachable prize than removing it.
    (
        "split ilu0/cheb",
        "split flow-first, Chebyshev on k-omega",
        lambda m, g, n: field_split(m, g, n, "ilu0", "chebyshev", flow_first=True),
    ),
    (
        "split ilu0/jac",
        "split flow-first, damped Jacobi on k-omega",
        lambda m, g, n: field_split(m, g, n, "ilu0", "jacobi", flow_first=True),
    ),
    # The turbulence block on a JAX-NATIVE hierarchy instead of a PETSc V-cycle -- an independent knob
    # from the flow smoother, and the one that would actually remove PETSc from this half of the
    # preconditioner. lAIR is reduction-based coarsening built for nonsymmetric advection-dominated
    # transport, which is what [k, omega] is; "twolevel" is the cheaper aggregation sibling. Both are the
    # hierarchies `scalar_transport_preconditioner` already uses for the segregated k and omega solves.
    (
        "split ilu0/air",
        "split flow-first, lAIR on k-omega",
        lambda m, g, n: field_split(m, g, n, "ilu0", "air", flow_first=True),
    ),
    (
        "split ilu0/2lvl",
        "split flow-first, twolevel AMG on k-omega",
        lambda m, g, n: field_split(m, g, n, "ilu0", "twolevel", flow_first=True),
    ),
    # Vanka on the four-field saddle. Chebyshev fails there because a saddle has no bounded positive real
    # spectrum; a patch method has no such assumption, and the omega weakness that condemned patches on
    # the six-field block is exactly what the split takes away. Additive Vanka is also a batch of small
    # independent dense inverses -- no sequential triangular solve -- so if it works it is the
    # parallelizable smoother the saddle block otherwise lacks.
    (
        "split vanka/ilu0",
        "split flow-first, Vanka on flow",
        lambda m, g, n: field_split(m, g, n, "vanka", "ilu0", flow_first=True),
    ),
    (
        "split vanka/jac",
        "split flow-first, Vanka flow + Jacobi k-omega",
        lambda m, g, n: field_split(m, g, n, "vanka", "jacobi", flow_first=True),
    ),
    (
        "split vankamult/ilu0",
        "split flow-first, multiplicative Vanka on flow",
        lambda m, g, n: field_split(m, g, n, "vanka-mult", "ilu0", flow_first=True),
    ),
    # The corner of the grid with no PETSc-only component left: a patch smoother on the saddle (a batch
    # of independent small dense inverses) and a JAX-native hierarchy on the scalars.
    (
        "split vanka/air",
        "split flow-first, Vanka flow + lAIR k-omega",
        lambda m, g, n: field_split(m, g, n, "vanka", "air", flow_first=True),
    ),
    # The GPU question: a Jacobi-class smoother on the FOUR-field saddle, with the scalars kept on
    # incomplete-LU, then on both. If the four-field block is smoothable where six is not, these work.
    (
        "split flow/cheb",
        "split flow-first, Chebyshev on flow",
        lambda m, g, n: field_split(m, g, n, "chebyshev", "ilu0", flow_first=True),
    ),
    (
        "split flow/cheb both",
        "split flow-first, Chebyshev both",
        lambda m, g, n: field_split(m, g, n, "chebyshev", "chebyshev", flow_first=True),
    ),
    # The CELL-BLOCK smoother on the saddle, which is the gap every arm above leaves open. Chebyshev and
    # damped Jacobi are *point* methods: they see a cell's scalar diagonal and discard the local
    # pressure-velocity coupling entirely, which on a saddle is not a detail but the whole difficulty.
    # Point-block Jacobi inverts each cell's dense [u,v,w,p] block instead, so it captures that coupling
    # exactly -- and it is still a batch of independent small dense solves with no sequential
    # dependency, which is the property the incomplete-LU sweep lacks and an accelerator needs.
    #
    # It is defined in SMOOTHERS above and has never been paired with the flow block in any arm; every
    # measurement of it on this case is on [k, omega]. Both damping factors run, because the damped and
    # undamped Richardson differed by a factor of five in sweeps on the trailing block.
    (
        "split pbjac/ilu0",
        "split flow-first, point-block Jacobi on flow (damped 0.7)",
        lambda m, g, n: field_split(m, g, n, "pbjacobi", "ilu0", flow_first=True),
    ),
    (
        "split pbjacu/ilu0",
        "split flow-first, point-block Jacobi on flow (undamped)",
        lambda m, g, n: field_split(m, g, n, "pbjacobi-undamped", "ilu0", flow_first=True),
    ),
    # And the arm the effort is actually for: the flow block on a hierarchy written in this package
    # rather than in PETSc, with the same cell-block smoother. If this converges, the whole
    # preconditioner can leave the host solver; if it caps where the matched PETSc row above does not,
    # the deficit is in our coarsening and is a defined thing to fix; if both cap, the cell-block
    # smoother is not enough on the saddle and the next candidate has to be globally coupled.
    (
        "split native4/ilu0",
        "split flow-first, native nodal hierarchy on flow, 4 sweeps",
        lambda m, g, n: field_split(m, g, n, "native4", "ilu0", flow_first=True),
    ),
    (
        "split native8/ilu0",
        "split flow-first, native nodal hierarchy on flow, 8 sweeps",
        lambda m, g, n: field_split(m, g, n, "native8", "ilu0", flow_first=True),
    ),
    (
        "split native4d/ilu0",
        "split flow-first, native nodal hierarchy on flow, 4 sweeps, damped",
        lambda m, g, n: field_split(m, g, n, "native4d", "ilu0", flow_first=True),
    ),
    (
        "split native8d/ilu0",
        "split flow-first, native nodal hierarchy on flow, 8 sweeps, damped",
        lambda m, g, n: field_split(m, g, n, "native8d", "ilu0", flow_first=True),
    ),
    # DECOMPOSING the saddle instead of smoothing it. Every arm above hands all four fields to one
    # hierarchy; these eliminate pressure first, leaving a convection-diffusion velocity block and a
    # scalar elliptic pressure operator -- neither a saddle, and both inside the domain an ordinary
    # aggregation multigrid was built for. The `petsc` rows keep the incomplete-LU sweep so they
    # measure what the DECOMPOSITION is worth on its own; the `native` rows are the candidate.
    (
        "split nested-petsc/ilu0",
        "split flow-first, nested u/p on flow (raw p-block), PETSc V-cycles",
        lambda m, g, n: field_split(m, g, n, "nested-petsc", "ilu0", flow_first=True),
    ),
    (
        "split simple-petsc/ilu0",
        "split flow-first, SIMPLE Schur on flow, PETSc V-cycles",
        lambda m, g, n: field_split(m, g, n, "simple-petsc", "ilu0", flow_first=True),
    ),
    (
        "split nested-native/ilu0",
        "split flow-first, nested u/p on flow (raw p-block), native V-cycles",
        lambda m, g, n: field_split(m, g, n, "nested-native", "ilu0", flow_first=True),
    ),
    (
        "split simple-native/ilu0",
        "split flow-first, SIMPLE Schur on flow, native V-cycles",
        lambda m, g, n: field_split(m, g, n, "simple-native", "ilu0", flow_first=True),
    ),
    # ONE HALF AT A TIME. `simple-native-petsc` is a native (cell-block-smoothed) velocity block with
    # the pressure half held on the known-good host V-cycle, so it asks whether the velocity block can
    # go native without the pressure question confounding the answer; the converse arm asks the other.
    # Read them against `simple-petsc` (both host) and `simple-native` (both native): if holding
    # pressure recovers the host arm, velocity is solved and pressure is the whole remaining problem.
    (
        "split simple-native-petsc/ilu0",
        "split flow-first, SIMPLE on flow: native velocity, PETSc pressure",
        lambda m, g, n: field_split(m, g, n, "simple-native-petsc", "ilu0", flow_first=True),
    ),
    (
        "split simple-petsc-native/ilu0",
        "split flow-first, SIMPLE on flow: PETSc velocity, native pressure",
        lambda m, g, n: field_split(m, g, n, "simple-petsc-native", "ilu0", flow_first=True),
    ),
    # SIMPLE as a level SMOOTHER inside a native hierarchy -- the arm the flat inverses above are not.
    (
        "split simplesmooth2/ilu0",
        "split flow-first, native MG + SIMPLE smoother on flow, 2 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth2", "ilu0", flow_first=True),
    ),
    # The MULTI-STEP block-LU saddle preconditioner, and the control that isolates the Frobenius-
    # optimal diagonal from the rest of it. One step from a zero start is the minimal consistent form.
    (
        "split multistep1/ilu0",
        "split flow-first, multi-step saddle (Frobenius, 1 step) on flow",
        lambda m, g, n: field_split(m, g, n, "multistep1", "ilu0", flow_first=True),
    ),
    (
        "split multistep1-jacobi/ilu0",
        "split flow-first, multi-step saddle (Jacobi, 1 step) on flow",
        lambda m, g, n: field_split(m, g, n, "multistep1-jacobi", "ilu0", flow_first=True),
    ),
    # The multi-step axis, which is the one the paper says carries the method: N steps sweep the
    # splitting error A - A~ out on the right-hand side instead of leaving it in the operator.
    # `-alg2` seeds the velocity with a real solve of F u = f at p = 0, reported as much the better
    # start (from u = 0 many eigenvalues sit near zero; seeded, all are above one).
    (
        "split multistep1-alg2/ilu0",
        "split flow-first, multi-step saddle (Frobenius, 1 step, alg 2) on flow",
        lambda m, g, n: field_split(m, g, n, "multistep1-alg2", "ilu0", flow_first=True),
    ),
    (
        "split multistep3-alg2/ilu0",
        "split flow-first, multi-step saddle (Frobenius, 3 steps, alg 2) on flow",
        lambda m, g, n: field_split(m, g, n, "multistep3-alg2", "ilu0", flow_first=True),
    ),
    (
        "split multistep10-alg2/ilu0",
        "split flow-first, multi-step saddle (Frobenius, 10 steps, alg 2) on flow",
        lambda m, g, n: field_split(m, g, n, "multistep10-alg2", "ilu0", flow_first=True),
    ),
    # The LEFT BLOCK TRANSFORM: fix what the coarse grid coarsens, rather than the smoother. The
    # `-schur` variant drops the (p,u) correction, which makes it an inexact transform but a far
    # cheaper one -- if the exact form's triple products are unaffordable, that is itself the result.
    (
        "split transform4/ilu0",
        "split flow-first, left block transform (exact) + native MG on flow",
        lambda m, g, n: field_split(m, g, n, "transform4", "ilu0", flow_first=True),
    ),
    (
        "split transform4-schur/ilu0",
        "split flow-first, left block transform (Schur-only) + native MG on flow",
        lambda m, g, n: field_split(m, g, n, "transform4-schur", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4/ilu0",
        "split flow-first, native MG + SIMPLE smoother on flow, 4 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4", "ilu0", flow_first=True),
    ),
    # Eq. (39) applied to the SCHUR relaxation as well, tested on its own axis. The velocity side is
    # held at the Frobenius inverse in both arms, so the only thing that moves is the pressure
    # relaxation's diagonal -- which is the crudest component of the sweep and acts on the densest
    # operator in it.
    # The pressure relaxation's own damping, which is the control that decides whether Eq. (39) on the
    # Schur is a bad idea or was merely applied on top of an existing one.
    # SINGLETONS. A vertex reached late in the random MIS sweep can find every neighbour claimed and
    # then opens an aggregate containing only itself: measured 49 of 161 aggregates on the second level,
    # median aggregate size 3. `-ns` attaches such a vertex to an adjacent aggregate instead. `-qr` is
    # re-tested on top, because orthonormalization scales a column by 1/sqrt(|agg|) and therefore
    # PROMOTES singletons -- so the two changes interact and the earlier QR result was measured against
    # a coarse space full of them.
    (
        "split simplesmooth4-a0-L4-ns/ilu0",
        "split flow-first, native MG + SIMPLE smoother, no singletons, deep",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-L4-ns", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4-a0-L4-ns-qr/ilu0",
        "split flow-first, native MG + SIMPLE smoother, no singletons + orthonormal, deep",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-L4-ns-qr", "ilu0", flow_first=True),
    ),
    # The singleton fix at TWO levels. It was only ever measured deep, where a degenerate coarse-of-
    # coarse operator is the obvious place for it to matter; whether it also improves the two-level
    # hierarchies decides if it is a general property of the aggregation or a depth-only repair.
    (
        "split simplesmooth4-ns/ilu0",
        "split flow-first, native MG + SIMPLE smoother, no singletons, 2 levels",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-ns", "ilu0", flow_first=True),
    ),
    # DEPTH, now that the aggregation no longer manufactures degenerate coarse unknowns. Raising the
    # level cap on its own does nothing -- coarsening stops at the first level under `max_coarse`, and
    # the three-level arm already lands there -- so each of these lowers that limit as well. The
    # coarsest level is inverted densely, so shrinking it is the cost that matters.
    (
        "split simplesmooth4-a0-L4-ns-c200/ilu0",
        "split flow-first, native MG + SIMPLE smoother, no singletons, coarse under 200",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth4-a0-L4-ns-c200", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth4-a0-L5-ns-c50/ilu0",
        "split flow-first, native MG + SIMPLE smoother, no singletons, coarse under 50",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-L5-ns-c50", "ilu0", flow_first=True),
    ),
    # QR-orthonormalized tentative prolongation. Provably inert at two levels with an exact coarse
    # solve, so the two-level arm is a REGRESSION CHECK rather than a candidate; the deep arm is the
    # test, since that is where the 0/1 columns' scaling stops cancelling.
    (
        "split simplesmooth4-qr/ilu0",
        "split flow-first, native MG + SIMPLE smoother, orthonormal prolongation, 2 levels",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-qr", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4-a0-L4-qr/ilu0",
        "split flow-first, native MG + SIMPLE smoother, orthonormal prolongation, deep",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-L4-qr", "ilu0", flow_first=True),
    ),
    # The coarsening RATE at a fixed two levels, which is what the depth sweep actually pointed at:
    # gentle (21x, coarse 4300) beat aggressive (106x, coarse 872) by 1.7x, but a 4300-equation DENSE
    # coarse solve is not affordable and does not scale. A strength threshold aggregates only along
    # strong connections, landing between the two.
    (
        "split simplesmooth4-a0-t10/ilu0",
        "split flow-first, native MG + SIMPLE smoother, gentle coarsening, threshold 0.10",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-t10", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4-a0-t25/ilu0",
        "split flow-first, native MG + SIMPLE smoother, gentle coarsening, threshold 0.25",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-t25", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4-a0-t50/ilu0",
        "split flow-first, native MG + SIMPLE smoother, gentle coarsening, threshold 0.50",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-t50", "ilu0", flow_first=True),
    ),
    # DEPTH and coarsening RATE. `-L3`/`-L4` raise the level cap; `-a0` drops the aggressive first
    # level so each step coarsens gently instead of ~100x at once. Two levels with a direct coarse
    # solve is what every native arm has run at so far, and it has never been varied.
    (
        "split simplesmooth4-L3/ilu0",
        "split flow-first, native MG + SIMPLE smoother, 3 levels, 4 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-L3", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4-L4-a0/ilu0",
        "split flow-first, native MG + SIMPLE smoother, 4 levels, gentle coarsening, 4 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-L4-a0", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4-a0/ilu0",
        "split flow-first, native MG + SIMPLE smoother, 2 levels, gentle coarsening, 4 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0", "ilu0", flow_first=True),
    ),
    # The two changes that have actually paid, combined -- the singleton fix and the sweep count. Each
    # was measured against a hierarchy carrying the other at its old value, so their product is an
    # assumption until it is run. `simplesmooth16` asks the separate question of whether relaxation
    # saturates: doubling sweeps doubles the cost of every application, so a gain that keeps halving
    # the residual is worth taking and one that flattens is not.
    (
        "split simplesmooth8-a0-ns/ilu0",
        "split flow-first, native MG + SIMPLE smoother, 2 levels, no singletons, 8 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth8-a0-ns", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth16-a0-L4-ns/ilu0",
        "split flow-first, native MG + SIMPLE smoother, no singletons, deep, 16 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth16-a0-L4-ns", "ilu0", flow_first=True),
    ),
    # DOES AGGREGATION NEED TO READ THE OPERATOR AT ALL? At a zero strength threshold the edge set is
    # the full cell adjacency, so aggregates are chosen from connectivity alone and every coupling
    # magnitude -- and with it every trace of the flow direction -- is discarded before coarsening
    # begins. A threshold makes the choice value-dependent, which is the cheapest available test of
    # whether that blindness costs anything.
    #
    # The earlier threshold sweep is not usable as that test: filtering edges REMOVES aggregation
    # candidates, so the coarse grid grew to 26244 dofs against the unfiltered arm's few hundred, and a
    # knob that moves coarse size by seventy-fold cannot be compared against one that does not. These
    # three therefore share one coarsening limit and one level cap, so each is free to take as many
    # levels as its own rate needs to reach a comparable coarse grid.
    (
        "split simplesmooth8-a0-ns-L5-c500/ilu0",
        "split flow-first, SIMPLE smoother, isotropic aggregation, coarse under 500",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-a0-ns-L5-c500", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth8-a0-t10-ns-L5-c500/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.10 aggregation, coarse under 500",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-a0-t10-ns-L5-c500", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth8-a0-t25-ns-L5-c500/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25 aggregation, coarse under 500",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-a0-t25-ns-L5-c500", "ilu0", flow_first=True
        ),
    ),
    # Where the threshold turns over. A stronger filter keeps aggregating along the stiff directions but
    # discards more of the graph, so past some point there is too little left to aggregate across and the
    # coarsening rate collapses -- 0.25 already needs five levels where the unfiltered graph needed three.
    # The coarsening RATE recovered without giving up strong-connection SELECTION. The strength filter
    # is applied BEFORE the graph is squared, so squaring acts on the strong graph rather than the full
    # one: aggregates get large again, but they are still grown along the couplings the threshold kept.
    # This is the arm the per-cycle cost points at -- at a threshold the hierarchy already matches the
    # incomplete-LU on iteration count, and everything left is the price of the intermediate levels that
    # a three-times-per-level coarsening rate leaves behind.
    (
        "split simplesmooth8-t10-ns-L5-c500/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.10 + aggressive coarsening",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-t10-ns-L5-c500", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth8-t25-ns-L5-c500/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25 + aggressive coarsening",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-t25-ns-L5-c500", "ilu0", flow_first=True
        ),
    ),
    # The INNER pressure relaxation count, on its own axis. Four of these sit inside every outer sweep
    # and are two thirds of its cost, so they are the largest single term in the smoother -- and the only
    # one never varied.
    (
        "split simplesmooth8-a0-t25-ns-L5-c500-ps2/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25, 2 inner pressure sweeps",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-a0-t25-ns-L5-c500-ps2", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth8-a0-t25-ns-L5-c500-ps1/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25, 1 inner pressure sweep",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-a0-t25-ns-L5-c500-ps1", "ilu0", flow_first=True
        ),
    ),
    # IS TEN CYCLES A FLOOR? Three structurally unrelated leading inverses all land at 10-11 here, and
    # nothing has ever beaten 10 at this state. If doubling the smoothing does not move it, the leading
    # block is saturated and the count is being set somewhere else -- the trailing block, or the coupling
    # triangle the split discards -- in which case this comparison says nothing about the leading block.
    (
        "split simplesmooth16-a0-t25-ns-L5-c500/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25, 16 sweeps (saturation probe)",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth16-a0-t25-ns-L5-c500", "ilu0", flow_first=True
        ),
    ),
    # The (outer sweeps, inner pressure sweeps) FRONTIER. The two trade against different things --
    # outer sweeps buy cycles (8 -> 16 halves them), inner sweeps buy per-cycle cost (4 -> 2 cuts it by a
    # quarter) -- and every arm measured so far sits at one corner of that grid, with the other axis at a
    # default nobody chose. Sixteen outer sweeps reach six cycles, below the incomplete-LU's eleven, so a
    # cheaper-per-cycle variant of that corner is where a win would be.
    (
        "split simplesmooth16-a0-t25-ns-L5-c500-ps2/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25, 16 sweeps, 2 inner",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth16-a0-t25-ns-L5-c500-ps2", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth16-a0-t25-ns-L5-c500-ps1/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25, 16 sweeps, 1 inner",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth16-a0-t25-ns-L5-c500-ps1", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth12-a0-t25-ns-L5-c500-ps2/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25, 12 sweeps, 2 inner",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth12-a0-t25-ns-L5-c500-ps2", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth24-a0-t25-ns-L5-c500-ps1/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.25, 24 sweeps, 1 inner",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth24-a0-t25-ns-L5-c500-ps1", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth8-a0-t50-ns-L5-c500/ilu0",
        "split flow-first, SIMPLE smoother, strength 0.50 aggregation, coarse under 500",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-a0-t50-ns-L5-c500", "ilu0", flow_first=True
        ),
    ),
    # The singleton fix against the two-level arm that can actually contain singletons. The squared
    # graph builds aggregates of median size 82 and produces none, so pairing the fix with it tests
    # nothing; only the gentler maximal-independent-set rate leaves vertices that arrive to find every
    # neighbour already claimed.
    (
        "split simplesmooth4-a0-ns/ilu0",
        "split flow-first, native MG + SIMPLE smoother, 2 levels, gentle coarsening, no singletons",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-ns", "ilu0", flow_first=True),
    ),
    # SMOOTHED AGGREGATION under the SIMPLE smoother. Every native arm to date has interpolated the
    # coarse correction piecewise-constant over each aggregate, which is the usual reason a hierarchy
    # stops improving past two levels. Smoothing was measured before, but under a Jacobi smoother and a
    # different SIMPLE relaxation, so it is a fresh question here rather than a settled one.
    #
    # Sweeps are a CONFOUND, not a detail: the smoothed prolongator is on record as failing outright at
    # four sweeps and being the best native arm at eight, so a smoothing result quoted at one sweep
    # count says nothing. Each formula therefore runs at both, against an eight-sweep UNSMOOTHED
    # control -- without which a win at eight cannot be attributed to the prolongator rather than to
    # the extra relaxation.
    (
        "split simplesmooth8-a0-L4-ns/ilu0",
        "split flow-first, native MG + SIMPLE smoother, no singletons, deep, 8 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth8-a0-L4-ns", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4-a0-L4-ns-sm/ilu0",
        "split flow-first, native MG + SIMPLE smoother, smoothed prolongator, deep, 4 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth4-a0-L4-ns-sm", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth8-a0-L4-ns-sm/ilu0",
        "split flow-first, native MG + SIMPLE smoother, smoothed prolongator, deep, 8 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth8-a0-L4-ns-sm", "ilu0", flow_first=True),
    ),
    (
        "split simplesmooth4-a0-L4-ns-smstd/ilu0",
        "split flow-first, native MG + SIMPLE smoother, standard prolongator, deep, 4 sweeps",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth4-a0-L4-ns-smstd", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth8-a0-L4-ns-smstd/ilu0",
        "split flow-first, native MG + SIMPLE smoother, standard prolongator, deep, 8 sweeps",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-a0-L4-ns-smstd", "ilu0", flow_first=True
        ),
    ),
    # The standard prolongator's best recorded configuration, carried over intact: eight sweeps with
    # the operator equilibrated before coarsening.
    (
        "split simplesmooth8-a0-L4-ns-smstd-eq/ilu0",
        "split flow-first, native MG + SIMPLE smoother, standard prolongator equilibrated, deep",
        lambda m, g, n: field_split(
            m, g, n, "simplesmooth8-a0-L4-ns-smstd-eq", "ilu0", flow_first=True
        ),
    ),
    (
        "split simplesmooth2-jacobi/ilu0",
        "split flow-first, native MG + SIMPLE smoother (Jacobi inverse), 2 sweeps",
        lambda m, g, n: field_split(m, g, n, "simplesmooth2-jacobi", "ilu0", flow_first=True),
    ),
)


def run_arm(label, preconditioner, built, coupled, state, rhs, op_shift, solver):
    """Solve the REAL system with one already-built preconditioner; report cycles and the TRUE residual."""

    def operator(v):
        return _jacobian_matvec(coupled, state, v) + op_shift * v

    solving = time.time()
    solution, raw = solve_linear(
        operator, rhs, solver, preconditioner=preconditioner.matvec(), throw=False
    )
    true = float(jnp.linalg.norm(operator(solution) - rhs) / jnp.linalg.norm(rhs))
    cycles = restart_cycles(int(raw))
    print(
        f"    {label:<36} build {built:>5.0f}s  cycles {cycles:>4}  TRUE rel {true:.3e}  "
        f"solve {time.time() - solving:>4.0f}s",
        flush=True,
    )
    return cycles, true


def one_arm(label, build, shifted, groups, n_fields, coupled, state, rhs, op_shift, solver):
    """Build and run a single arm, surviving a failure so the arms queued behind it still run.

    A raise here -- a singular coarse solve, a zero pivot, a failed Chebyshev eigenvalue estimate -- is a
    result about that arm, and by the time it happens the remaining arms represent most of the run.
    """
    preconditioner = None
    try:
        started = time.time()
        preconditioner = build(shifted, groups, n_fields)
        return run_arm(
            label, preconditioner, time.time() - started, coupled, state, rhs, op_shift, solver
        )
    except Exception as failure:
        print(f"    {label:<36} FAILED  {type(failure).__name__}: {failure}", flush=True)
        return None
    finally:
        if preconditioner is not None:
            preconditioner.factors.destroy()
        del preconditioner
        gc.collect()


def self_check(name, recorded, shifted, groups, n_fields, coupled, state, rhs, op_shift, solver):
    """Reproduce the recorded control measurement, and refuse to go on if the CYCLE COUNT disagrees.

    Run at the **march's own** solver, because that is the configuration the recorded numbers were taken
    under: judging them against this study's far tighter stop would fail a faithful harness, since a solve
    that reaches 1e-07 in one cycle keeps going when asked for 1e-08.

    **Only the cycle count is gated, and the reason is a real difference this probe cannot remove.** The
    recorded numbers come from driving an actual dual-time step, whose right-hand side is the step's own
    residual ``G = R + beta d (phi - phi_n)``; this probe uses the steady residual ``R``. At inner
    iteration 0 the two coincide exactly (``phi = phi_n``), which is why a checkpoint-based sweep is right
    to use ``R`` -- but a captured inner iterate is precisely where they part, and on the hardest one they
    differ by a factor of some 200 (``|G|`` 3.8e-03 against ``|R|`` 8.3e-01). Reconstructing ``G`` would
    need ``phi_n``, the state the outer step began from, which the observer does not record.

    So the operator, state and shift are the march's; the right-hand side is not. The cycle count is
    comparable and is gated; the achieved residual is not comparable and is reported for the record
    rather than asserted against. Every arm sees the identical right-hand side, so the comparison
    *between* arms -- which is what this probe exists for -- is unaffected.
    """
    print("\n  -- self-check: the shipped preconditioner at the march's own solver", flush=True)
    measured = one_arm(
        "monolithic, ILU(0), march solver",
        ARMS[0][2],
        shifted,
        groups,
        n_fields,
        coupled,
        state,
        rhs,
        op_shift,
        solver,
    )
    if measured is None:
        raise SystemExit(f"SELF-CHECK FAILED for {name}: the control arm did not run at all.")
    cycles, true = measured
    if recorded is None:
        if not np.isfinite(true) or true > 1e-3:
            raise SystemExit(
                f"SELF-CHECK FAILED for {name}: the control did not converge (true relative residual "
                f"{true:.3e}), so nothing else measured here can be trusted."
            )
        print("    [no recorded value for this state; control converges, continuing]", flush=True)
        return
    expected_cycles, expected_true = recorded
    if cycles != expected_cycles:
        raise SystemExit(
            f"SELF-CHECK FAILED for {name}: the shipped preconditioner took {cycles} restart cycles "
            f"where {expected_cycles} is on record for this state and pairing. The harness is not "
            "solving the operator the record describes; fix that before reading any other row."
        )
    print(
        f"    [self-check passed on CYCLES: {cycles}, as recorded. The true residual reads {true:.3e} "
        f"against the recorded {expected_true:.1e}; these are not comparable, because the recorded run's "
        "right-hand side was the step's dual-time residual G and this one's is the steady residual R.]",
        flush=True,
    )


def study(
    coupled, state, rhs, shifted, op_shift, groups, n_fields, only=None, pc_state=None, pc_beta=0.0
):
    """Every arm at this state's pairing, at the study's own tight stop so the arms separate.

    ``only`` restricts to a subset of arm keys. Re-running the whole ladder to add one arm costs several
    minutes of arms whose answer is already on the log -- and the arms that FAIL are the expensive ones,
    since running to the restart cap is what failing means here. The control is always kept, because a
    subset without it cannot be compared against anything.

    The block-SIMPLE arms are appended here rather than declared in :data:`ARMS` because they are built
    from the assembler at ``pc_state``, not from the assembled block (see :func:`block_simple_arms`).
    """
    arms = ARMS + block_simple_arms(coupled, state if pc_state is None else pc_state, pc_beta)
    missing = set(only or ()) - {key for key, _, _ in arms}
    if missing:
        raise SystemExit(f"unknown arm(s) {sorted(missing)}; known: {[key for key, _, _ in arms]}")
    selected = [a for a in arms if only is None or a[0] == arms[0][0] or a[0] in only]
    print(f"\n  -- study arms, GMRES to rtol {RTOL:.0e} on the TRUE residual", flush=True)
    return {
        key: one_arm(label, build, shifted, groups, n_fields, coupled, state, rhs, op_shift, SOLVER)
        for key, label, build in selected
    }


def _invocation() -> list[str]:
    """The command line, falling back to the environment when there is none.

    A probe of this size is a long solve on a shared machine, which means it belongs behind
    ``validation/run_case.sh`` -- the runner that refuses to start a second one, holds the machine
    awake, and writes a run-file saying what is running and under what settings. That runner takes a
    script and forwards the **environment**, not script arguments, so a probe configured only through
    ``sys.argv`` cannot be launched through it and has to be run bare, where two sessions can collide on
    a machine with room for one 2 GB Jacobian.

    So the state and the arm list are readable from ``BFS3D_PROBE_STATE`` / ``BFS3D_PROBE_PC_STATE`` /
    ``BFS3D_PROBE_ARMS`` as well, which is the same convention the case itself uses. Arguments win where
    both are given, so every existing invocation is unchanged.
    """
    if len(sys.argv) > 1:
        return sys.argv[1:]
    state = os.environ.get("BFS3D_PROBE_STATE")
    if not state:
        return []
    argv = [state]
    pc_state = os.environ.get("BFS3D_PROBE_PC_STATE")
    if pc_state:
        argv.append(pc_state)
    arms = os.environ.get("BFS3D_PROBE_ARMS")
    if arms:
        argv.append(f"--arms={arms}")
    return argv


def main():
    supplied = _invocation()
    argv = [a for a in supplied if not a.startswith("--arms=")]
    chosen = [a for a in supplied if a.startswith("--arms=")]
    only = tuple(chosen[-1].split("=", 1)[1].split(",")) if chosen else None
    if not 1 <= len(argv) <= 2 or argv[0] not in STATES:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} <{' | '.join(STATES)}> [preconditioner state] "
            "[--arms=key,key]"
        )
    sys.argv = [sys.argv[0], *argv]
    name = sys.argv[1]
    entry = STATES[name]
    march_beta, recorded, description = entry.march_beta, entry.recorded, entry.description
    # `BFS3D_PROBE_BETA` builds the OPERATOR at a chosen shift on whichever state is loaded, which is the
    # only way to vary beta as an axis: every entry in `STATES` carries one fixed shift, and the two
    # shifted ones are step-initial checkpoints that cost a cycle or two for every arm and so cannot rank
    # anything. Holding the state fixed and moving the shift separates the shift from the state, where
    # switching entries confounds them. `checkpoint_shift` is a separate field and still checks identity
    # against the file, so this does not weaken the faithfulness gate.
    override = os.environ.get("BFS3D_PROBE_BETA")
    if override is not None:
        march_beta = float(override)
        recorded = None  # nothing is on record at a synthesized shift
        description = f"{description} -- OPERATOR SHIFT OVERRIDDEN to beta={march_beta}"
    # An optional SECOND state builds the preconditioner, while the operator and right-hand side stay at
    # the first. That is what the march actually does -- it freezes the preconditioner for a whole inner
    # loop -- and its expensive solves are measured to be this staleness rather than hard operators (15
    # cycles against 1 when matched), so it is the pairing with real headroom on this case. Passing two
    # consecutive inner iterates of one attempt reproduces exactly one inner iteration of staleness.
    pc_state_name = sys.argv[2] if len(sys.argv) == 3 else name
    if pc_state_name not in STATES:
        raise SystemExit(f"unknown preconditioner state {pc_state_name!r}")
    stale = pc_state_name != name
    if stale:
        recorded = None  # nothing is on record for a deliberately mismatched pairing
    # The V-cycle is built at the floor while the operator keeps the march's own beta -- the shipped
    # mismatch. At the converged state's zero shift there is no floor: the adjoint has none, and flooring
    # it here would measure a preconditioner the gradient path never uses.
    pc_beta = max(march_beta, FLOOR) if march_beta > 0 else 0.0

    coupled = compare.build_case()["coupled"]
    n_fields = coupled.layout.dim + 3
    groups = FieldGroups(
        n_cells=coupled.layout.n_cells,
        n_leading_fields=coupled.layout.dim + 1,  # u, v, w, p -- the saddle
        n_trailing_fields=2,  # k, omega -- the transported scalars
    )
    print(
        f"{'=' * 100}\nfield split: {groups.n_leading_fields} leading + {groups.n_trailing_fields} "
        f"trailing fields over {groups.n_cells} cells\nbundle: plain aggregation, "
        f"ILU({compare.FILL_LEVELS}) x{compare.SWEEPS} where not overridden, coarse_eq_limit "
        f"{compare.COARSE_EQ_LIMIT}, stencil reach 3, column reach "
        f"{'uniform' if compare.COLUMN_REACH is None else '/'.join(map(str, compare.COLUMN_REACH))}"
        f", GMRES restart 15, max restarts {MAX_RESTARTS}\n"
        f"operator beta {march_beta}, preconditioner beta {pc_beta}\n{'=' * 100}",
        flush=True,
    )
    state = load_state(name)
    print(f"  {description}", flush=True)

    # Probe each column at the reach the CASE uses, read from `compare` rather than restated here, so a
    # probe cannot measure a preconditioner built from a sparsity the march does not use. That is not a
    # hypothetical: this default has already moved twice, and both moves turned on the SPARSITY rather
    # than on any value. A shortened column writes its out-of-reach entries as exact zeros where a
    # uniform probe leaves the true value -- tiny, but nonzero -- and an assembly written as a sparse
    # product stores only entries whose result is nonzero, so it deletes those explicit zeros and hands
    # a zero-fill incomplete factorization a structurally weaker pattern for a numerically identical
    # matrix. Reading the case's value is what keeps this probe on the right side of that.
    plan = _coupled_jacobian_plan(coupled, 3, compare.COLUMN_REACH)
    structure = block_stencil_gather_map(plan)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    rhs = -coupled.residual(state)
    op_shift = _frozen_shift_diagonal(base, march_beta, state) if march_beta > 0 else 0.0
    # Report both norms, so the run records the one way it departs from the march: the march solved for
    # the step's dual-time residual G, this solves for the steady residual R, and on an inner iterate
    # those are different right-hand sides over the same operator.
    print(f"  right-hand side |R| {float(jnp.linalg.norm(rhs)):.4e}", flush=True)

    # The preconditioner is assembled at its own state, which is the same one unless a stale pairing was
    # asked for. Only one Jacobian is ever live: the operator side is applied matrix-free, by the exact
    # jvp at `state`, so the materialization is needed only for the preconditioner.
    if stale:
        print("  preconditioner built at a DIFFERENT state:", flush=True)
        pc_state = load_state(pc_state_name)
        pc_base = _coupled_shift_policy(coupled, pc_state, "twolevel")
    else:
        pc_state, pc_base = state, base
    jacobian = materialize(coupled, pc_state, plan, structure, n_fields)
    pc_shift = (
        _frozen_shift_diagonal(pc_base, pc_beta, pc_state)
        if pc_beta > 0
        else np.zeros(groups.n_dofs)
    )
    shifted = MonolithicAmgPreconditioner._shifted(jacobian, pc_shift)
    del jacobian
    gc.collect()

    self_check(
        name,
        recorded,
        shifted,
        groups,
        n_fields,
        coupled,
        state,
        rhs,
        op_shift,
        march_solver(coupled, base, state),
    )
    study(
        coupled,
        state,
        rhs,
        shifted,
        op_shift,
        groups,
        n_fields,
        only=only,
        pc_state=pc_state,
        pc_beta=pc_beta,
    )


if __name__ == "__main__":
    main()
