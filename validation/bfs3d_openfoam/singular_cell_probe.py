"""Which cells have a singular ``[k, omega]`` block, and are they the cells bounded by positivity?

The framework-native block smoother inverts each cell's own 2x2, so it refuses to build when any of
them is singular -- and on a developed state of this case a handful are, which stops a march at a
mid-march refresh. The question this answers is whether those cells are *special* in a way that
explains them, and specifically whether they coincide with the cells where the positivity limiter is
capping the Newton step.

That link is worth testing rather than assuming, and it is plausible in both directions: a cell whose
``k`` is being driven toward zero is a cell whose ``k`` row loses what makes it invertible, so the
degenerate blocks and the step-length cap could be one phenomenon observed twice. If they coincide,
anything that discards the local correction in those cells -- truncating the block inverse, say --
removes it exactly where the solve is already struggling, and that has a bearing on how the build's
refusal should be handled.

Takes a checkpoint path directly rather than a name from a curated table, because the states that
exhibit this are whatever the march happened to reach when it failed.

Usage::

    python3 -u validation/bfs3d_openfoam/singular_cell_probe.py checkpoints/state-00019.npz
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
)
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_jacobian_colouring,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
    _jacobian_matvec,
)
from cell_block_scaling import cell_blocks  # noqa: E402

N_TRAILING = 2


def where(centroids: np.ndarray, cells: np.ndarray) -> str:
    """A readable position summary for a handful of cells."""
    return "; ".join(
        f"({centroids[c, 0]:.4f}, {centroids[c, 1]:.4f}, {centroids[c, 2]:.4f})" for c in cells[:8]
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <checkpoint.npz>")
    path = Path(sys.argv[1])
    stored = np.load(path)
    state = jnp.asarray(stored["state"])
    # An INNER iterate carries no shift -- the observer that writes one is not told it -- which is
    # exactly why the sweep below exists rather than a single pairing. A step checkpoint does carry one,
    # and it is reported for reference, but it is never the only shift probed.
    shift_recorded = float(stored["shift"]) if "shift" in stored else float("nan")
    described = (
        f"step {int(stored['step'])}, |R| {float(stored['residual_norm']):.4e}, "
        f"march shift {shift_recorded:.4g}"
        if "step" in stored
        else (
            f"attempt {int(stored['attempt'])} inner {int(stored['inner'])}, "
            f"{int(stored['cycles'])} cycles, alpha {float(stored['alpha']):.3f}, "
            f"|G| {float(stored['g_before']):.4e} -> {float(stored['g_after']):.4e}"
        )
    )
    print(f"{'=' * 92}\n{path.name}: {described}\n{'=' * 92}", flush=True)

    case = compare.build_case()
    coupled = case["coupled"]
    n_fields = coupled.layout.dim + 3
    n_cells = coupled.layout.n_cells
    groups = FieldGroups(
        n_cells=n_cells, n_leading_fields=coupled.layout.dim + 1, n_trailing_fields=N_TRAILING
    )
    pc_beta = (
        compare.PC_BETA_FLOOR
        if np.isnan(shift_recorded)
        else max(shift_recorded, compare.PC_BETA_FLOOR)
    )

    # The block the smoother is handed, at the pairing the march uses: the operator at its own shift,
    # the preconditioner built at the floor.
    colouring = _coupled_jacobian_colouring(coupled, 3)
    structure = block_stencil_gather_map(colouring, n_fields)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    jacobian = MonolithicAmgPreconditioner._materialize_jacobian(
        lambda v: _jacobian_matvec(coupled, state, v), colouring, n_fields, None, None, structure
    )
    # SWEEP the shift rather than probing one value. The block is `J + beta d`, so beta props up the
    # very diagonal that makes each cell block invertible -- and the refresh that failed happens at the
    # NEXT step's beta, not the one the checkpoint records. Pairing the operator with the wrong beta is
    # how this probe first reported no singular blocks at a state that had just produced four.
    print(f"  singular cell blocks against the shift (checkpoint recorded {shift_recorded:.4g}):")
    betas = tuple(
        b
        for b in (shift_recorded, 0.4, 0.2963, 0.1975, 0.1317, compare.PC_BETA_FLOOR, 0.02, 0.0)
        if not np.isnan(b)
    )
    counts = {}
    for beta in betas:
        probe = sp.csr_matrix(
            MonolithicAmgPreconditioner._shifted(
                jacobian,
                _frozen_shift_diagonal(base, beta, state) if beta > 0 else np.zeros(groups.n_dofs),
            )[groups.trailing, :][:, groups.trailing]
        )
        b = cell_blocks(probe, N_TRAILING)
        det = np.abs(np.linalg.det(b))
        sing = det < 1e-12 * np.maximum(
            np.linalg.norm(b, axis=(1, 2)) ** N_TRAILING, np.finfo(float).tiny
        )
        counts[beta] = (int(sing.sum()), np.flatnonzero(sing))
        print(
            f"    beta {beta:<8.4g} singular {int(sing.sum()):>4}   min |det|/scale "
            f"{np.min(det / np.maximum(np.linalg.norm(b, axis=(1, 2)) ** N_TRAILING, 1e-300)):.3e}",
            flush=True,
        )
        del probe, b
        gc.collect()

    # Study the smallest shift that actually shows the failure, which is the configuration the march met.
    beta_used = next((bt for bt in betas if counts[bt][0] > 0), pc_beta)
    print(f"\n  studying beta {beta_used:.4g}\n")
    shifted = MonolithicAmgPreconditioner._shifted(
        jacobian,
        _frozen_shift_diagonal(base, beta_used, state)
        if beta_used > 0
        else np.zeros(groups.n_dofs),
    )
    block = sp.csr_matrix(shifted[groups.trailing, :][:, groups.trailing])
    del jacobian, shifted
    gc.collect()

    blocks = cell_blocks(block, N_TRAILING)
    scale = np.linalg.norm(blocks, axis=(1, 2)) ** N_TRAILING
    determinant = np.abs(np.linalg.det(blocks))
    singular = determinant < 1e-12 * np.maximum(scale, np.finfo(float).tiny)
    cells = np.flatnonzero(singular)
    print(f"\nsingular cell blocks: {cells.size} of {n_cells}  -> cells {cells.tolist()}\n")
    if cells.size == 0:
        return

    # The solved fields, in the layout the state is packed in.
    _flow, k_solved, omega_solved = coupled.layout.unpack(state)
    k = np.asarray(coupled.k_transform.to_physical(k_solved))
    omega = np.asarray(coupled.omega_transform.to_physical(omega_solved))
    centroids = np.asarray(case["mesh"].cell_geometry.centroid)

    print("  the singular cells:")
    for c in cells:
        rank = int((k < k[c]).sum())
        print(
            f"    cell {c:>6}  k {k[c]:.4e} (rank {rank}/{n_cells}, {100 * rank / n_cells:.2f}th pct)"
            f"  omega {omega[c]:.4e}  det {determinant[c]:.3e}"
            f"  at ({centroids[c, 0]:.4f}, {centroids[c, 1]:.4f}, {centroids[c, 2]:.4f})"
        )

    print(
        f"\n  k over the mesh: min {k.min():.4e}  p1 {np.percentile(k, 1):.4e}  "
        f"p50 {np.median(k):.4e}  max {k.max():.4e}"
    )
    print(f"  k at the singular cells: {np.array2string(k[cells], precision=4)}")

    # THE question: are these the cells the positivity limiter binds on? The limiter caps the step by
    # min over decreasing entries of k / -dk, so the cells at risk are those with the smallest k
    # relative to how fast it is falling. Without the step direction the honest proxy is k itself --
    # so report where these cells sit in the k distribution, and how many cells are below them.
    below = int((k < k[cells].max()).sum())
    print(
        f"\n  cells with k below the largest singular cell's k: {below} "
        f"({100 * below / n_cells:.2f}%) -- if this is small, the singular cells ARE the low-k tail"
    )
    lowest = np.argsort(k)[: max(cells.size * 5, 20)]
    print(
        f"  are the singular cells among the {lowest.size} lowest-k cells? "
        f"{sorted(set(cells.tolist()) & set(lowest.tolist()))}"
    )
    print(f"\n  lowest-k cells overall: {lowest[:8].tolist()}")
    print(f"    their k: {np.array2string(k[lowest[:8]], precision=4)}")
    print(f"    their det: {np.array2string(determinant[lowest[:8]], precision=3)}")
    print(f"    positions: {where(centroids, lowest)}", flush=True)


if __name__ == "__main__":
    main()
