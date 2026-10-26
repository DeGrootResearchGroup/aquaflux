"""Does a pressure probe coloured for ``R2 + S(tau)`` drop anything the real Jacobian holds large?

``pressure_strength_skeleton.py`` selects, from geometry alone, the distance-3 pressure couplings
``S(tau)`` a probe must keep, judged on a proxy of the continuity rows. Every row field reads the
pressure column, though: momentum through ``V grad p`` and Rhie--Chow's flux in its convection, ``k``
and ``omega`` through the convective flux. This materializes the uniform-reach-3 coupled Jacobian at one
checkpoint of a uniform-reach-3 march (the step's start, or an inner iterate), exactly as the
preconditioner of that step is fitted (the station's assembler, float64), and reads each (row field, p)
block row by row:

  dropped   the largest ``|J_ij|`` at graph distance 3 that the pattern leaves out, over the row's
            weakest face-neighbour ``|J_ij|`` in the same block -- per row field, for reach 2 alone
            and for each ``R2 + S(tau)``, as median / p99 / max and the share of rows above ``tau`` and
            ``3 tau``, overall and by the cell's spacing ratio (``mesh_anisotropy.spacing_ratio``).
            Rows with no nonzero face-neighbour entry in the block are counted and left out.
  folded    the pattern probe itself, materialized: a dropped coupling is not only zeroed, it is summed
            onto a kept entry of its row sharing its colour. The largest such change per row, over the
            same weakest face coupling, and the Frobenius share of the block the probe drops and moves.
  strongest the same dropped entry over the row's strongest entry in the block, a second scale for rows
            whose weakest face coupling is near zero; and each row's weakest over strongest face
            coupling, which says where the first scale is structurally small.

Pre-registered pass (option 1 proposal): every row field's dropped ratio at or below about ``3 tau``.

Environment: ``REACH_FOLDING_CASE``, ``REACH_FOLDING_STEP``, ``REACH_FOLDING_INNER``,
``REACH_FOLDING_CHECKPOINTS`` and ``REACH_FOLDING_ATTEMPT`` as in ``reach_folding.py`` (the state is the
iterate that harness would solve at), and ``REACH_AUDIT_TAUS`` (default ``0.01,0.03``). One state per
process::

    REACH_FOLDING_STEP=5 REACH_FOLDING_INNER=2 validation/run_case.sh validation/reach_option1_audit.py --wait
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reach_folding  # noqa: E402  (puts the case on the path and imports its compare)
from aquaflux.precondition import materialize_block_jacobian  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    PROBE_BATCH_SIZE,
    batched_jacobian_matvec,
    jacobian_matvec,
)
from aquaflux.precondition.sparse_jacobian import (  # noqa: E402
    ColumnProbePlan,
    block_stencil_colouring,
)
from pressure_strength_skeleton import (  # noqa: E402
    PressureSkeleton,
    pattern_colouring,
    pattern_probe_plan,
    restricted,
    row_max,
    row_min,
)

compare = reach_folding.compare
BINS = reach_folding.ANISOTROPY_BINS


def materialized(assembler, state, plan):
    """The Jacobian of ``assembler`` at ``state`` under ``plan``, as a field-major CSR matrix."""
    return materialize_block_jacobian(
        lambda v: jacobian_matvec(assembler, state, v),
        plan,
        batched_matvec=lambda seeds: batched_jacobian_matvec(assembler, state, seeds),
        probe_batch_size=PROBE_BATCH_SIZE,
    ).tocsr()


def pressure_blocks(jacobian, n, n_fields, pressure):
    """The (row field, p) blocks, each ``(n, n)`` CSR."""
    column = jacobian[:, pressure * n : (pressure + 1) * n].tocsr()
    return [column[a * n : (a + 1) * n].tocsr() for a in range(n_fields)]


def quantiles(values, tau):
    return (
        f"{np.median(values):9.2e} {np.quantile(values, 0.99):9.2e} {values.max():9.2e} "
        f"{np.mean(values > tau):7.1%} {np.mean(values > 3 * tau):7.1%}"
    )


def report_block(name, block, skeleton, labels, folded_blocks, taus):
    n = skeleton.n
    weakest = row_min(restricted(block, skeleton.graph.faces), n)
    strongest = row_max(restricted(block, skeleton.graph.within[3]), n)
    usable = np.isfinite(weakest)
    total = float(np.linalg.norm(block.data))
    distance3 = skeleton.graph.at_distance(3)
    print(
        f"\n  row field {name}: |block| {total:.3e}, rows with no nonzero face coupling "
        f"{int((~usable).sum())} of {n}",
        flush=True,
    )
    # How far a row's face couplings spread: where the weakest face coupling is structurally small
    # beside the strongest (a gradient component across a face parallel to it), it is a poor scale.
    spread = weakest[usable] / row_max(restricted(block, skeleton.graph.faces), n)[usable]
    print(
        f"    weakest/strongest face coupling: median {np.median(spread):.2e} p1 "
        f"{np.quantile(spread, 0.01):.2e} min {spread.min():.2e}",
        flush=True,
    )
    for label, (pattern, folded) in zip(labels, folded_blocks, strict=True):
        dropped_mask = (distance3 - distance3.multiply(pattern)).tocsr()
        dropped_mask.eliminate_zeros()
        dropped_entries = restricted(block, dropped_mask)
        dropped = row_max(dropped_entries, n)
        ratio = dropped[usable] / weakest[usable]
        tau = taus.get(label)
        shown = [tau] if tau is not None else sorted(set(taus.values()))
        for t in shown:
            print(
                f"    {label:9s} dropped/weakest (tau {t:g}) median p99 max >tau >3tau: "
                f"{quantiles(ratio, t)}  |dropped|/|block| {np.linalg.norm(dropped_entries.data) / total:.2e}",
                flush=True,
            )
            for low, high in BINS:
                cells = (skeleton.spacing >= low) & (skeleton.spacing < high) & usable
                if cells.any():
                    r = dropped[cells] / weakest[cells]
                    print(
                        f"      {low:g}-{high:g}:1 {int(cells.sum()):6d} cells  {quantiles(r, t)}",
                        flush=True,
                    )
        relative = dropped[usable] / np.where(strongest[usable] > 0, strongest[usable], np.inf)
        print(
            f"      dropped/strongest median {np.median(relative):.2e} p99 {np.quantile(relative, 0.99):.2e} max {relative.max():.2e}",
            flush=True,
        )
        if folded is not None:
            kept = (skeleton.graph.within[3] - dropped_mask).tocsr()
            change = restricted(folded - block, kept)
            moved = row_max(change, n)[usable] / weakest[usable]
            print(
                f"      folded onto kept entries: |change|/|block| {np.linalg.norm(change.data) / total:.2e}; per row "
                f"max|change|/weakest median {np.median(moved):.2e} p99 {np.quantile(moved, 0.99):.2e} max {moved.max():.2e}",
                flush=True,
            )


def main():
    step_number = int(os.environ["REACH_FOLDING_STEP"])
    checkpoints = Path(
        os.environ.get("REACH_FOLDING_CHECKPOINTS", reach_folding.CASE / "checkpoints")
    )
    taus = [float(t) for t in os.environ.get("REACH_AUDIT_TAUS", "0.01,0.03").split(",")]
    with np.load(checkpoints / f"state-{step_number - 1:05d}.npz") as data:
        phi = jnp.asarray(data["state"])
    inner = int(os.environ.get("REACH_FOLDING_INNER", "0"))
    iterate, where = None, f"the step's start (state-{step_number - 1:05d})"
    if inner > 0:
        attempt = int(os.environ.get("REACH_FOLDING_ATTEMPT", str(step_number)))
        dump = checkpoints / f"inner-{attempt:05d}-{inner - 1:02d}.npz"
        with np.load(dump) as data:
            iterate = jnp.asarray(data["state"])
        where = f"inner {inner} ({dump.name})"
    spec = compare.SOLVER.preconditioner
    if spec.probe.gradient_sweeps is not None:
        raise SystemExit(
            "the case's probe narrows the gradient sweeps; this audit materializes the assembler itself"
        )
    coupled = compare.build_case()["coupled"]
    n, n_fields = coupled.layout.n_cells, coupled.layout.n_fields
    names = reach_folding.field_names(n_fields)
    pressure = names.index("p")
    rebuild = reach_folding.StepRebuild(coupled, phi, 1.0, step_number, iterate)
    commit = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    print(
        f"[configuration] case {reach_folding.CASE_NAME}, {n} cells, step {step_number} (station {rebuild.station}), "
        f"Jacobian of the station's assembler at {where}, uniform reach-3 probe, float64, commit {commit}, "
        f"taus {taus}",
        flush=True,
    )
    t0 = time.time()
    mesh = coupled.momentum.mesh
    skeleton = PressureSkeleton.of(mesh)
    owner, nb = skeleton.owner, skeleton.nb
    labels, patterns, plans = [], [], []
    for tau in [None, *taus]:
        pattern = skeleton.pattern(tau)
        colour, count = pattern_colouring(pattern)
        label = "reach 2" if tau is None else f"S({tau:g})"
        kept = 0.0 if tau is None else skeleton.strong_distance_three(tau).nnz / n
        print(
            f"[colours] p column {label}: {count} colours, kept distance-3 entries per row {kept:.2f}",
            flush=True,
        )
        labels.append(label)
        patterns.append(pattern)
        plans.append(pattern_probe_plan(owner, nb, n, n_fields, pressure, pattern, colour, count))
    print(
        f"[colours] reach 3: {block_stencil_colouring(owner, nb, n, 3).n_colours}; skeleton in {time.time() - t0:.0f} s",
        flush=True,
    )
    state = rebuild.iterate
    t0 = time.time()
    exact = pressure_blocks(
        materialized(
            rebuild.assembler,
            state,
            ColumnProbePlan.uniform(block_stencil_colouring(owner, nb, n, 3), n_fields),
        ),
        n,
        n_fields,
        pressure,
    )
    gc.collect()
    folded = []
    for plan in plans:
        folded.append(
            pressure_blocks(materialized(rebuild.assembler, state, plan), n, n_fields, pressure)
        )
        gc.collect()
    print(f"[matrix] {1 + len(plans)} materializes in {time.time() - t0:.0f} s", flush=True)
    by_label = dict(zip(labels[1:], taus, strict=True))
    for a, name in enumerate(names):
        report_block(
            name,
            exact[a],
            skeleton,
            labels,
            [(p, f[a]) for p, f in zip(patterns, folded, strict=True)],
            by_label,
        )


if __name__ == "__main__":
    main()
