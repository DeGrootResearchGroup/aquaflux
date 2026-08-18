"""Would probing ``[k, omega]`` on its own, instead of slicing it out of the full coupled Jacobian,
save anything?

``FieldSplitAmgPreconditioner`` materializes the WHOLE six-field coupled Jacobian with one coloured jvp
probe and only then slices out the leading ``[u,v,w,p]`` block, the trailing ``[k,omega]`` block, and the
one retained coupling triangle between them (the field-major layout makes a field-boundary split a cheap
contiguous-range slice, not a gather). The colouring itself already gives ``k`` and ``omega`` their own
cheap reach-two probe (``BFS3D_COLUMN_REACH``'s trailing ``2,2``) -- that part is not on the table here.
What the combined materialize still pays for, on every one of those reach-two probes, is a response over
**all six** row-field blocks, including the ``[u,v,w,p]`` rows a flow-first split never reads back out of
a turbulence column (see ``field_coupling.py``: that triangle is a small fraction of the operator's
Frobenius norm, but the point here is its share of stored entries, not its magnitude).

This measures what a probe scoped to *just* ``[k, omega]`` -- its own two-field colouring, its own
reach-two pattern, a residual wrapped to return only the k/omega rows with the flow state held fixed --
would cost against what the trailing block already costs as a slice of the combined materialize. Two
questions, and they have different answers:

* Does scoping change the number of colours/probes for k and omega? No -- ``column_probe_plan`` already
  colours a reach-two column at reach two regardless of how many other fields share the materialize, so
  the probe count for k and omega alone is the same whether they are probed standalone or as two columns
  of the six-field plan.
* Does scoping change how much is STORED and carried through de-compression for those probes? Yes, in
  principle -- a two-field pattern has no room for the ``[u,v,w,p]``-row entries the six-field pattern
  reserves for every column, k/omega included. This measures the size of what that removes.

Run each arm as its own process (peak resident set is a high-water mark, so two arms sharing a process
just report the larger one's peak for both) and read back the saved trailing block from an earlier
``--arm full`` run to check the two constructions agree, since they are two different ways of computing
the same partial derivative and either could sample it wrongly. The state is constructed, not loaded, so
this is reproducible: the hybrid initial field at the target Reynolds number, before any march step --
the cheapest state available and one every continuation rung visits.

Usage::

    python3 -u validation/bfs3d_openfoam/trailing_scoped_probe.py --arm full
    python3 -u validation/bfs3d_openfoam/trailing_scoped_probe.py --arm scoped
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parents[1]))

import compare  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    ColumnProbePlan,
    block_stencil_colouring,
    block_stencil_gather_map,
    materialize_block_jacobian,
)
from aquaflux.turbulence import CoupledJacobianProbe, hybrid_initialize  # noqa: E402

OUT = CASE / "checkpoints"
TRAILING_NPZ = OUT / "trailing_scoped_probe_full.npz"
#: Batched-probe chunk size, matching the production materialize (`coupled.py`'s `_PROBE_BATCH_SIZE`).
PROBE_BATCH_SIZE = 8


def peak_bytes() -> int:
    """Peak resident set of this process so far, in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if peak > 1 << 20 else peak * 1024  # Linux reports KiB, macOS bytes.


def cold_state(coupled):
    """The hybrid initial field at this case's own viscosity -- no march, seconds not minutes."""
    flow, k, omega = hybrid_initialize(coupled.momentum, coupled.turbulence)
    return np.asarray(coupled.state_from_physical(flow, k, omega), dtype=np.float64)


def batched_from(matvec):
    """A batched matvec over a leading axis of stacked seeds, from a single-seed matvec."""
    return jax.jit(jax.vmap(matvec))


def full_arm(coupled, state, n_cells, n_fields):
    print("[full] building the shipped six-field probe (BFS3D_COLUMN_REACH)", flush=True)
    probe = CoupledJacobianProbe.build(coupled, column_reach=compare.COLUMN_REACH)
    plan = probe.plan
    print(
        f"[full] {plan.n_probes} probes total; colours per column field "
        f"{dict(zip(('u', 'v', 'w', 'p', 'k', 'omega'), plan.n_colours, strict=True))}",
        flush=True,
    )

    state_j = jnp.asarray(state)

    def matvec(v):
        return jax.jvp(coupled.residual, (state_j,), (v,))[1]

    started = time.time()
    jacobian = materialize_block_jacobian(
        matvec,
        plan,
        batched_matvec=batched_from(matvec),
        probe_batch_size=PROBE_BATCH_SIZE,
        structure=probe.structure,
    ).tocsr()
    elapsed = time.time() - started
    peak = peak_bytes()
    print(f"[full] materialized in {elapsed:.1f}s, peak RSS so far {peak / 1e9:.2f} GB", flush=True)
    print(f"[full] {jacobian.nnz / 1e6:.2f}M nonzeros total", flush=True)

    lead_dofs = 4 * n_cells  # [u, v, w, p]
    blocks = {
        "leading diag  [u,v,w,p] <- [u,v,w,p]": jacobian[:lead_dofs, :lead_dofs],
        "trailing diag [k,omega] <- [k,omega]": jacobian[lead_dofs:, lead_dofs:],
        "retained coupling (kept) [k,omega] <- [u,v,w,p]": jacobian[lead_dofs:, :lead_dofs],
        "dropped coupling (discarded) [u,v,w,p] <- [k,omega]": jacobian[:lead_dofs, lead_dofs:],
    }
    print("\n  block                                                  nnz        share", flush=True)
    for name, block in blocks.items():
        print(
            f"  {name:<52s} {block.nnz / 1e6:8.3f}M   {block.nnz / jacobian.nnz:6.1%}", flush=True
        )

    trailing = blocks["trailing diag [k,omega] <- [k,omega]"].tocsr()
    OUT.mkdir(exist_ok=True)
    sp.save_npz(TRAILING_NPZ, trailing)
    print(
        f"\n[full] saved the trailing diagonal block to {TRAILING_NPZ.name} for the scoped arm",
        flush=True,
    )
    return jacobian


def scoped_arm(coupled, state, n_cells):
    print("[scoped] building a two-field [k, omega]-only reach-2 probe", flush=True)
    owner, nb, _ = coupled.momentum.mesh.face_cells.interior_edges()
    owner, nb = np.asarray(owner), np.asarray(nb)
    colouring = block_stencil_colouring(owner, nb, n_cells, 2)
    plan = ColumnProbePlan.uniform(colouring, 2)
    print(
        f"[scoped] {plan.n_probes} probes, {colouring.n_colours} colours (k and omega share them)",
        flush=True,
    )
    structure = block_stencil_gather_map(plan)

    lead_dofs = 4 * n_cells
    flow_frozen = jnp.asarray(state[:lead_dofs])
    turb_state = jnp.asarray(state[lead_dofs:])

    def turb_residual(turb):
        full = jnp.concatenate([flow_frozen, turb])
        return coupled.residual(full)[lead_dofs:]

    def matvec(v):
        return jax.jvp(turb_residual, (turb_state,), (v,))[1]

    started = time.time()
    jacobian = materialize_block_jacobian(
        matvec,
        plan,
        batched_matvec=batched_from(matvec),
        probe_batch_size=PROBE_BATCH_SIZE,
        structure=structure,
    ).tocsr()
    elapsed = time.time() - started
    peak = peak_bytes()
    print(
        f"[scoped] materialized in {elapsed:.1f}s, peak RSS so far {peak / 1e9:.2f} GB", flush=True
    )
    print(
        f"[scoped] {jacobian.nnz / 1e6:.2f}M nonzeros (trailing block alone, no flow rows stored)",
        flush=True,
    )

    if TRAILING_NPZ.exists():
        reference = sp.load_npz(TRAILING_NPZ).tocsr()
        diff = (jacobian - reference).tocsr()
        ref_norm = sp.linalg.norm(reference)
        rel = sp.linalg.norm(diff) / ref_norm if ref_norm else float("nan")
        print(
            f"\n[verify] against the '--arm full' trailing block: "
            f"relative Frobenius difference {rel:.2e} "
            f"({reference.nnz / 1e6:.2f}M stored entries there against {jacobian.nnz / 1e6:.2f}M here)",
            flush=True,
        )
    else:
        print(
            f"\n[verify] run '--arm full' first to write {TRAILING_NPZ.name}; skipping the cross-check",
            flush=True,
        )
    return jacobian


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("full", "scoped"), required=True)
    args = parser.parse_args()

    print("[case] building", flush=True)
    case = compare.build_case()
    coupled = case["coupled"]
    n_cells = coupled.layout.n_cells
    n_fields = coupled.layout.dim + 3
    print(
        f"[case] {n_cells} cells, {n_fields} fields, peak RSS so far {peak_bytes() / 1e9:.2f} GB",
        flush=True,
    )

    state = cold_state(coupled)
    print(
        f"[state] hybrid initial field, |R| = {float(np.linalg.norm(coupled.residual(state))):.3e}",
        flush=True,
    )

    if args.arm == "full":
        full_arm(coupled, state, n_cells, n_fields)
    else:
        scoped_arm(coupled, state, n_cells)

    print(f"\n[done] peak RSS for this process: {peak_bytes() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
