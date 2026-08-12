"""Does splitting the turbulence out of the coupled preconditioner's hierarchy help?

The coupled preconditioner puts all six fields -- the ``[u, v, w, p]`` saddle and the two transported
scalars ``k`` and ``omega`` -- through one multigrid hierarchy with one level smoother. This asks whether
giving the two groups their own hierarchies, while retaining one triangle of the coupling between them, is
better. It is **not** the block-*diagonal* arrangement that was tried and refuted: that one drops the
coupling, and the coupling is load-bearing. A triangle keeps half of it exactly.

**Where the headroom is, and is not.** At the states the march visits, a preconditioner rebuilt at the
iterate solves the forward system in one restart cycle *at the march's own loose stop*, so that pairing
cannot separate two candidates -- an easy operator is not a test, and a tie there is no information. Two
states restore the discrimination, and both are configurations something really solves:

* a **captured hard inner iterate**, at the shift the march solved it under, driven far past the march's
  own stop so the arms separate instead of all stopping at one cycle. The march's expensive solves are
  measured to be staleness rather than hard operators, so this is a comparison of quality at a matched
  preconditioner, not a reproduction of the march's cost.
* the **converged state at zero shift** -- the operator every gradient's transpose solve meets. Removing
  the pseudo-transient shift is what makes this operator hard, and the adjoint has no preconditioner floor
  to soften it, so this is where a better preconditioner has something real to win. Note it must be the
  *converged* state: stripping the shift off a mid-march iterate would measure an operator that nothing,
  forward or adjoint, ever solves.

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
* a **faithfulness gate**: the shipped monolithic arm must reproduce the restart-cycle count already
  recorded for it at this state, or the run refuses to report. The captured iterates predate the current
  march log, so this replaces the usual join against it.

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

    python3 -u validation/bfs3d_openfoam/field_split_probe.py inner-00050-03 > field_split.log 2>&1

A second argument builds the preconditioner at a **different** state from the operator, which is the third
pairing worth measuring: the march freezes its preconditioner for a whole inner loop, and its expensive
solves are measured to be that staleness rather than hard operators. Two consecutive inner iterates of one
attempt reproduce exactly one iteration of it, and whether a field split decays more gracefully under
staleness matters more on this case than whether it is better when matched::

    python3 -u validation/bfs3d_openfoam/field_split_probe.py inner-00050-03 inner-00050-02

Every smoother named here is a **fixed linear operator**, which the adjoint's transpose solve requires: a
Chebyshev smoother is a fixed polynomial once its eigenvalue bounds are estimated during setup, unlike a
GMRES-accelerated smoother, whose polynomial depends on the right-hand side.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))  # import aquaflux from the working tree, as compare.py is run
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
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
SOLVER = relative_residual_gmres(RTOL, restart=15, stagnation_iters=40, max_restarts=60)

#: The states this runs on, as ``name -> (operator beta, recorded self-check, description)``.
#:
#: The two ``inner-`` entries are captured mid-inner-loop iterates: their shift is not in the file (the
#: observer that writes them is not told it) and the march log that recorded it has since been overwritten,
#: so these are the two whose pairing is on record. Their ``recorded`` entry is what the shipped monolithic
#: preconditioner is already documented as achieving there, and it gates the run.
#:
#: ``state-00069`` is the converged end of that same march (``|R|`` 2.64e-06). It carries the **zero-shift**
#: operator, which is the one every gradient's transpose solve meets -- and the reason to measure there
#: rather than to strip the shift off a mid-march iterate, which would be a configuration nothing solves.
#: Nothing is on record for it, so it is self-checked only for the control converging at all.
STATES = {
    "inner-00050-03": (
        0.0293,
        (1, 6.6e-06),
        "the march's hardest solve: 15 cycles, line search collapsed to alpha 0",
    ),
    "inner-00040-03": (0.3333, (1, 1.7e-10), "8 cycles at a healthy alpha = 1"),
    "inner-00050-02": (
        0.0293,
        None,
        "the iteration before the hardest one -- the stale side of a one-iteration pairing",
    ),
    "state-00069": (0.0, None, "the converged state -- the ADJOINT's operator, at zero shift"),
    # STEP-INITIAL states, and they are here because the hard iterates above are NOT what a march mostly
    # pays for. A checkpoint is written at the end of a step, so it holds the state the next step begins
    # from -- a settled state met with a freshly refreshed preconditioner, which is the cheap first solve
    # of a step. On the shipped march 139 of 194 inner solves cost one restart cycle and only 7 exceeded
    # three, so this class is the bulk of the cost and the hard iterates are the tail. A candidate has to
    # be measured on both: the tail says whether it survives, the bulk says what it costs.
    "state-00057": (
        0.0103,
        None,
        "step 19 of the middle rung, step-initial -- the CHEAP solve that is most of the march",
    ),
    "state-00058": (
        0.0069,
        None,
        "step 20 of the middle rung, step-initial -- a second cheap solve, lower shift",
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
        recorded = float(data["shift"])
        expected = STATES[name][0] if name in STATES else recorded
        # An inner iterate carries no shift of its own (the observer is not told it), so only the
        # step checkpoints can be checked -- which is exactly the kind that gets rotated and reused.
        # Loose on purpose: the table records the shift to about four figures, so an exact comparison
        # rejects a matching state. What this has to catch is a REPLACED state, which differs by orders
        # of magnitude (0.98 where 0.0064 was documented), not by rounding.
        if not np.isclose(recorded, expected, rtol=0.02, atol=1e-9):
            raise SystemExit(
                f"{name}: this checkpoint was written at shift {recorded:.6g}, but the STATES table "
                f"describes it as {expected:.6g}. The file has been overwritten by a later march "
                "(the names come from a per-run counter over a rolling buffer), so it is NOT the state "
                "this entry documents. Re-run the case to regenerate it, or point the entry at a "
                "checkpoint that still matches."
            )
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


def field_split(
    shifted,
    groups,
    n_fields,
    flow_smoother,
    turbulence_smoother,
    *,
    flow_first,
    trailing_inverse=None,
):
    """A hierarchy per field group, retaining one triangle of the coupling between them.

    ``trailing_inverse`` overrides the trailing half wholesale. It is threaded rather than resolved from
    the arm name because the interesting override -- the JAX-native scalar hierarchies -- has to be built
    from the coupled system and its state, which the name alone cannot supply.
    """
    return MonolithicAmgPreconditioner(
        build_block_triangular_field_split(
            shifted,
            groups,
            flow_first=flow_first,
            smoother_fill_levels=compare.FILL_LEVELS,
            smoother_sweeps=compare.SWEEPS,
            coarse_eq_limit=compare.COARSE_EQ_LIMIT,
            leading_options=SMOOTHERS[flow_smoother] or None,
            trailing_options=SMOOTHERS.get(turbulence_smoother) or None,
            trailing_inverse=(
                trailing_inverse
                if trailing_inverse is not None
                else _trailing_inverse(turbulence_smoother)
            ),
        )
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


def study(coupled, state, rhs, shifted, op_shift, groups, n_fields, only=None):
    """Every arm at this state's pairing, at the study's own tight stop so the arms separate.

    ``only`` restricts to a subset of arm keys. Re-running the whole ladder to add one arm costs several
    minutes of arms whose answer is already on the log -- and the arms that FAIL are the expensive ones,
    since running to the restart cap is what failing means here. The control is always kept, because a
    subset without it cannot be compared against anything.
    """
    missing = set(only or ()) - {key for key, _, _ in ARMS}
    if missing:
        raise SystemExit(f"unknown arm(s) {sorted(missing)}; known: {[key for key, _, _ in ARMS]}")
    selected = [a for a in ARMS if only is None or a[0] == ARMS[0][0] or a[0] in only]
    print(f"\n  -- study arms, GMRES to rtol {RTOL:.0e} on the TRUE residual", flush=True)
    return {
        key: one_arm(label, build, shifted, groups, n_fields, coupled, state, rhs, op_shift, SOLVER)
        for key, label, build in selected
    }


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--arms=")]
    chosen = [a for a in sys.argv[1:] if a.startswith("--arms=")]
    only = tuple(chosen[-1].split("=", 1)[1].split(",")) if chosen else None
    if not 1 <= len(argv) <= 2 or argv[0] not in STATES:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} <{' | '.join(STATES)}> [preconditioner state] "
            "[--arms=key,key]"
        )
    sys.argv = [sys.argv[0], *argv]
    name = sys.argv[1]
    march_beta, recorded, description = STATES[name]
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
        f"{compare.COARSE_EQ_LIMIT}, stencil reach 3, GMRES restart 15\n"
        f"operator beta {march_beta}, preconditioner beta {pc_beta}\n{'=' * 100}",
        flush=True,
    )
    state = load_state(name)
    print(f"  {description}", flush=True)

    plan = _coupled_jacobian_plan(coupled, 3)
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
    study(coupled, state, rhs, shifted, op_shift, groups, n_fields, only=only)


if __name__ == "__main__":
    main()
