"""Is a Reynolds-continuation rung seed consistent with the rung it is handed to?

A rung inherits the converged root of the rung below, and **not everything in that state is a solved
quantity**. The near-wall ``omega`` rows are a *value fixation* -- the residual imposes
``omega_wall(nu, d, k)`` there rather than solving them -- and that value's viscous branch is
``C 6 nu / (beta_1 d^2)``, **linear in the molecular viscosity**. So the number carried across a rung
boundary is one the new rung's own model says is wrong, by the viscosity ratio exactly.

Two arms, selected by environment variable rather than command-line flag because
``validation/run_case.sh`` -- the only supported launcher for a long run here -- takes a script path and
forwards no arguments to it.

``consistency`` (the default arm)
    The coupled residual at one state under each viscosity in :data:`SCALES`, split per equation block
    ``[vel_0..vel_{dim-1}, continuity, k, omega]``, with the ``omega`` block split again by its shift:
    a row the pseudo-transient shift leaves at exactly zero has no time derivative and is a
    *constraint*, which is what the 472 wall fixations are.

``wallomega``
    What re-imposing those rows at the new viscosity does to the seed, per block.

Two readings per block throughout. ``scaled`` is the march's own row-equilibrated measure
(:func:`coupled_scaled_norm`) -- the number the solver steers and stops on, and the one to quote.
``euclidean`` is the raw block 2-norm, reported alongside because a verdict that flips between them is
a more interesting finding than either; ⚠️ it is nearly all ``omega``, so do not rank anything by it.

Each state is **also** reported in one fixed set of scales. That is not decoration: a repair moves
near-wall ``omega``, hence ``nu_t``, hence every block's row scales, so a fall in the self-scaled column
could be a fall in a numerator or a rise in a denominator, and only the fixed-scale column separates
them.

From the repository root, with no march running::

    validation/run_case.sh validation/pitzdaily_openfoam/seed_repair_probe.py
    CONSISTENCY_ARMS=consistency,wallomega \
        validation/run_case.sh validation/pitzdaily_openfoam/seed_repair_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))  # import aquaflux from the working tree, as compare.py is run
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
from aquaflux.turbulence.coupled import (  # noqa: E402
    _DEFAULT_SHIFT_BASIS,
    _frozen_shift_diagonal,
    _monolithic_shift_source,
    coupled_scaled_norm,
    wall_consistent_state,
)

#: The state to examine: the converged root of the middle Reynolds rung, which is verbatim the seed the
#: target rung begins from. That handover is the event this probe exists to measure.
#:
#: ⚠️ **It is a COPY, under a name the checkpointer never writes, and that is deliberate.** The rolling
#: `state-000NN.npz` files are evicted by the next march that runs (`PITZ_CHECKPOINT_KEEP`, default 3) --
#: which is exactly what happened to the state the first round of measurements here was taken at, killed
#: by the very march those measurements motivated. A finding is only re-askable if the harness *and* the
#: state it was taken at both survive. Make a copy under a stable name before measuring anything you
#: intend to write down; retention only ever deletes files the checkpointer itself wrote.
DEFAULT_STATE = CASE / "checkpoints" / "rung2-root.npz"

#: Which arms to run, comma-separated. Selected by environment rather than by command-line flag
#: because `validation/run_case.sh` -- the only supported way to launch a long run here -- takes a
#: script path and forwards no arguments to it.
ARMS = tuple(a for a in os.environ.get("CONSISTENCY_ARMS", "consistency").split(",") if a)

#: States to examine, comma-separated paths (relative ones resolve against this case directory).
STATES = tuple(s for s in os.environ.get("CONSISTENCY_STATES", str(DEFAULT_STATE)).split(",") if s)

#: Molecular-viscosity scale factors to evaluate the residual under, in order. The default pair is the
#: rung the seed was converged at and the rung it is handed to: a geometric Reynolds schedule at the
#: default decade ratio visits ``(100, 10, 1)``, so the last handover is ``10 -> 1``.
SCALES = tuple(float(s) for s in os.environ.get("CONSISTENCY_SCALES", "10,1").split(","))


def block_names(dim: int) -> tuple[str, ...]:
    """Per-block labels for the coupled layout ``[vel_0..vel_{dim-1}, continuity, k, omega]``."""
    return (*("u", "v", "w")[:dim], "continuity", "k", "omega")


def measure_at(coupled, state, scale):
    """``(companion, residual, measure)`` for ``state`` at molecular viscosity ``scale``.

    The measure's scales are built at ``state`` under the companion's own viscosity, which is exactly
    what the march does at the top of each outer step.
    """
    companion = coupled.with_scaled_molecular_viscosity(scale)
    residual = jax.lax.stop_gradient(companion.residual(state))
    base = _monolithic_shift_source(companion, state, _DEFAULT_SHIFT_BASIS)
    diagonal = np.asarray(_frozen_shift_diagonal(base, 1.0, state), dtype=np.float64)
    return companion, residual, coupled_scaled_norm(companion, base, state), diagonal


def equilibrated_block(measure, residual, index, n_cells):
    """One block of ``|R| / row_scale`` -- the per-cell fractional change the march's measure averages.

    Read out of the measure rather than re-derived: the continuity row's divisor is that cell's mass
    throughput, and forming it here from a second mass-flux evaluation would be a different number
    (the coupled residual's flux carries the eddy viscosity, an uncoupled momentum assembler's does
    not) while looking like the same one.
    """
    rows = slice(index * n_cells, (index + 1) * n_cells)
    return np.abs(np.asarray(residual)[rows]) / np.asarray(measure.row_scale)[rows]


def print_blocks(names, scaled, fixed, euclidean, fixed_label):
    """One per-block table: the march's measure, the same in fixed scales, and the raw block norm."""
    print(f"  {'block':<12} {'scaled':>12} {fixed_label:>18} {'euclidean':>12}", flush=True)
    for index, name in enumerate(names):
        print(
            f"  {name:<12} {scaled[index]:12.4e} {fixed[index]:18.4e} {euclidean[index]:12.4e}",
            flush=True,
        )


def report_consistency(coupled, state, label):
    """Assumption 2.1 part 1: the per-block residual of ``state`` under every viscosity in :data:`SCALES`."""
    dim, n_cells = coupled.momentum.mesh.dim, coupled.layout.n_cells
    names = block_names(dim)
    print(f"\n=== consistency: {label} ===", flush=True)

    reference_measure = None
    for scale in SCALES:
        _companion, residual, measure, diagonal = measure_at(coupled, state, scale)
        # A state packed in the wrong variables (physical omega where a log is transported, say) reads
        # finite and gives a silently non-finite residual, and every number below would then be noise
        # dressed as a measurement. Gate on it rather than discovering it three tables later.
        if not bool(jnp.all(jnp.isfinite(residual))):
            raise SystemExit(
                f"the residual of {label} at viscosity x{scale:g} is not finite -- check what wrote "
                f"the state, and that the case's variable transform matches it"
            )
        if reference_measure is None:
            reference_measure = measure
        scaled = np.asarray(measure.per_block(residual))
        fixed = np.asarray(reference_measure.per_block(residual))
        euclidean = np.linalg.norm(np.asarray(residual).reshape(len(names), -1), axis=1)

        print(
            f"\nviscosity x{scale:g}   |R| scaled {float(np.linalg.norm(scaled)):.4e}"
            f"   (scales built at this viscosity)",
            flush=True,
        )
        print_blocks(names, scaled, fixed, euclidean, f"in x{SCALES[0]:g} scales")

        imbalance = equilibrated_block(measure, residual, dim, n_cells)
        over = int(np.count_nonzero(imbalance > 1e-2))
        print(
            f"  continuity imbalance per cell (|net mass flux| / mass throughput): "
            f"mean {imbalance.mean():.4e}, median {np.median(imbalance):.4e}, "
            f"max {imbalance.max():.4e}, {over} cells above 1e-2",
            flush=True,
        )

        # ⚠️ CONTINUITY IS ONLY HALF THE ALGEBRAIC BLOCK. A row the pseudo-transient shift leaves at
        # exactly zero is a row with no time derivative -- a constraint -- and the near-wall omega
        # fixations are such rows just as continuity is. Assumption 2.1 part 1 asks for *every* one of
        # them to be satisfied at the initial iterate, so reporting only continuity would answer half
        # the question and could put the lever on the smaller half.
        omega = equilibrated_block(measure, residual, len(names) - 1, n_cells)
        fixed_rows = np.asarray(diagonal)[(len(names) - 1) * n_cells :] == 0.0
        transport = omega[~fixed_rows]
        print(
            f"  omega rows split by shift: {int(fixed_rows.sum())} algebraic (fixation, d = 0) mean "
            f"{omega[fixed_rows].mean():.4e} max {omega[fixed_rows].max():.4e}; "
            f"{int((~fixed_rows).sum())} differential (transport) mean {transport.mean():.4e} "
            f"max {transport.max():.4e}",
            flush=True,
        )


def report_wall_omega(coupled, state, label):
    """The ``wallomega`` arm: what re-imposing the wall-cell ``omega`` does to the seed, per block.

    The other half of the algebraic block, and the cheap half. These rows are a value *fixation* whose
    value the model computes from the molecular viscosity, so a state carried across a Reynolds rung
    holds a number the new residual says is wrong -- and unlike the pressure, the right number is
    looked up rather than solved for.

    Reported in **both** the post-repair scales and the pre-repair ones. That is not decoration: the
    repair moves near-wall ``omega``, which moves ``nu_t``, which moves the row scales of every block --
    so a fall in the self-scaled column could be a fall in a numerator or a rise in a denominator, and
    only the fixed-scale column separates them. (Dropping exactly this control is how the
    pressure-and-velocity variant was first read as helping momentum when in fixed scales it hurt it.)
    """
    dim = coupled.momentum.mesh.dim
    names = block_names(dim)
    scale = SCALES[-1]
    print(f"\n=== wall-omega re-imposition: {label}, viscosity x{scale:g} ===", flush=True)

    companion, residual, measure, diagonal = measure_at(coupled, state, scale)
    before = np.asarray(measure.per_block(residual))
    print(f"\nbefore   |R| scaled {float(np.linalg.norm(before)):.4e}", flush=True)
    print_blocks(
        names,
        before,
        before,
        np.linalg.norm(np.asarray(residual).reshape(len(names), -1), axis=1),
        "(same)",
    )

    repaired = wall_consistent_state(companion, state)
    _companion, after_residual, after_measure, _d = measure_at(coupled, repaired, scale)
    after = np.asarray(after_measure.per_block(after_residual))
    after_fixed = np.asarray(measure.per_block(after_residual))
    euclidean = np.linalg.norm(np.asarray(after_residual).reshape(len(names), -1), axis=1)
    print(f"\nafter    |R| scaled {float(np.linalg.norm(after)):.4e}", flush=True)
    print_blocks(names, after, after_fixed, euclidean, "in pre-repair scales")

    n_cells = coupled.layout.n_cells
    fixed_rows = np.asarray(diagonal)[(len(names) - 1) * n_cells :] == 0.0
    for tag, meas, res in (("before", measure, residual), ("after", after_measure, after_residual)):
        rows = equilibrated_block(meas, res, len(names) - 1, n_cells)
        print(
            f"  {tag}: {int(fixed_rows.sum())} algebraic omega rows mean {rows[fixed_rows].mean():.4e} "
            f"max {rows[fixed_rows].max():.4e}",
            flush=True,
        )
    physical_before = np.asarray(coupled.physical_fields(state)[2])
    physical_after = np.asarray(coupled.physical_fields(repaired)[2])
    cells = np.asarray(companion.turbulence.wall_cells)
    ratio = physical_after[cells] / physical_before[cells]
    print(
        f"  omega at the {cells.size} wall cells moved by a factor "
        f"{ratio.min():.4f} to {ratio.max():.4f} (median {np.median(ratio):.4f}); "
        f"the viscosity ratio is {SCALES[-1] / SCALES[0]:.4f}",
        flush=True,
    )


def main() -> None:
    unknown = set(ARMS) - {"consistency", "wallomega"}
    if unknown:
        raise SystemExit(
            f"unknown arm(s) {sorted(unknown)}; CONSISTENCY_ARMS takes consistency and/or wallomega"
        )
    coupled = compare.build_case()["coupled"]
    print(
        f"pitzDaily: {coupled.layout.n_cells} cells, dim {coupled.momentum.mesh.dim}, "
        f"viscosity scales {SCALES}",
        flush=True,
    )
    for raw in STATES:
        path = Path(raw)
        resolved = path if path.is_absolute() else CASE / path
        if not resolved.exists():
            raise SystemExit(f"no such checkpoint: {resolved}")
        state = jnp.asarray(np.load(resolved)["state"])
        if "consistency" in ARMS:
            report_consistency(coupled, state, resolved.name)
        if "wallomega" in ARMS:
            report_wall_omega(coupled, state, resolved.name)


if __name__ == "__main__":
    main()
