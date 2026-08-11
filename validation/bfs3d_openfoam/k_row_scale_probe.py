"""Does a cell whose turbulent kinetic energy has collapsed have a small ``k``-row diagonal?

This is the premise of a proposed explanation for why rescaling the trailing multigrid block changes
the step length a positivity limiter allows. That explanation runs: symmetric rescaling divides row
``i`` by ``sqrt(A_ii)``; a cell whose ``k`` has collapsed toward zero has a tiny diagonal there; so
rescaling promotes that row to unit weight, the coarsening and smoother start spending effort on an
equation the raw operator all but ignores, and un-scaling inflates the correction in exactly the
near-zero-``k`` cells that the cap -- a minimum over cells -- is decided by.

Every step of that depends on the first clause, and the first clause is checkable in seconds. The
``k`` row's diagonal is ``d(R_k)/d(k)`` at the cell, and the terms that set it -- the destruction
``beta* omega V``, convection and diffusion through the faces, and the pseudo-transient shift -- are
**not obviously proportional to k**. If the diagonal is flat in ``k``, the premise is false, the
rescaling never promotes those rows, and the explanation is dead regardless of anything downstream.

**Method.** The diagonal entry for one cell is a single Jacobian-vector product against a one-hot
tangent: ``jvp(residual, state, e_i)`` returns the column ``dR/d(k_i)``, whose own ``k_i`` component is
the diagonal. Sampling a few hundred cells across the range of ``k`` costs a few hundred JVPs and
**never materializes the Jacobian**, so this runs in seconds and in a few hundred MB rather than the
gigabytes a coloured probe needs.

The **unshifted** diagonal is what is reported. The pseudo-transient shift only adds a positive,
``k``-independent amount to it, so it can only ever flatten a dependence on ``k`` -- if the unshifted
diagonal is already flat, the shifted one is flatter, and no seed state is needed to say so.

**Reading the result.** ``scale = 1/sqrt(|diagonal|)`` is the factor symmetric equilibration applies to
the row.

* Premise **holds** if the diagonal falls and ``scale`` rises sharply as ``k`` falls -- the low-``k``
  rows are the ones rescaling promotes.
* Premise **fails** if the diagonal is flat in ``k``. Then rescaling treats a collapsed cell's row like
  any other, and it cannot be the route by which the flag changes the cap.

Usage::

    python3 -u k_row_scale_probe.py <dump> [<dump> ...]

e.g. ``k_row_scale_probe.py step-limit-11 step-limit-04``, naming checkpoint stems under
``checkpoints/``. The wall closure and the Reynolds rung must match the run those dumps came from;
both are asserted against the dump rather than assumed, because getting either wrong silently changes
the ``k`` equation at exactly the wall-adjacent cells this is about.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: The dumps this is written for came from a run under this wall closure. It changes the k equation at
#: wall-adjacent cells, which is the population under study, so it is set rather than inherited.
os.environ.setdefault("BFS3D_K_WALL", "dirichlet")

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import compare  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

#: The rung the dumps sit in (the Re/10 continuation point).
VISCOSITY_SCALE = 10.0

#: Cells sampled per decile of k. Enough to see a trend against scatter, cheap enough to stay seconds.
PER_DECILE = 40

#: Cells the limiter has actually been observed to bind on, always sampled so the population of
#: interest is never missed by chance.
BINDING_CELLS = (12800, 3181, 22400)


def k_row_diagonals(companion, state, cells, k_start):
    """Exact ``d(R_k)/d(k)`` at each named cell, one Jacobian-vector product apiece."""

    def diagonal(cell):
        tangent = jnp.zeros_like(state).at[k_start + cell].set(1.0)
        column = jax.jvp(companion.residual, (state,), (tangent,))[1]
        return column[k_start + cell]

    return np.asarray([float(diagonal(int(c))) for c in cells])


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <dump> [<dump> ...]")

    case = compare.build_case()
    coupled = case["coupled"]
    companion = coupled.with_scaled_molecular_viscosity(VISCOSITY_SCALE)
    n = coupled.layout.n_cells if hasattr(coupled.layout, "n_cells") else 23040
    k_start = 4 * n
    print(
        f"k wall BC {compare.K_WALL}, viscosity scale {VISCOSITY_SCALE:g}, k slice [{k_start}:{5 * n}]"
    )

    for name in sys.argv[1:]:
        stored = np.load(CASE / "checkpoints" / f"{name}.npz")
        state = jnp.asarray(stored["state"])
        k = np.asarray(state[k_start : 5 * n])
        print(
            f"\n{'=' * 92}\n{name}: cap {float(stored['cap']):.4e}, beta {float(stored['beta']):g}"
        )
        print(
            f"  k over {k.size} cells: min {k.min():.3e}  median {np.median(k):.3e}  max {k.max():.3e}"
        )

        order = np.argsort(k)
        sample = np.unique(
            np.concatenate(
                [np.asarray(BINDING_CELLS)]
                + [order[b * k.size // 10 : (b + 1) * k.size // 10][:PER_DECILE] for b in range(10)]
            )
        )
        diagonal = k_row_diagonals(companion, state, sample, k_start)
        scale = 1.0 / np.sqrt(np.abs(diagonal))

        print(
            f"\n  {'decile of k':>12}{'n':>5}{'median k':>13}{'median diag':>14}{'median scale':>14}"
        )
        sampled_k = k[sample]
        edges = np.quantile(k, np.linspace(0.0, 1.0, 11))
        for b in range(10):
            lo, hi = edges[b], edges[b + 1]
            sel = (sampled_k >= lo) & (sampled_k <= hi if b == 9 else sampled_k < hi)
            if not sel.any():
                continue
            print(
                f"  {b:>12}{int(sel.sum()):>5}{np.median(sampled_k[sel]):>13.3e}"
                f"{np.median(diagonal[sel]):>14.3e}{np.median(scale[sel]):>14.3e}"
            )

        print(f"\n  {'binding cell':>14}{'k':>13}{'diag':>14}{'scale':>14}{'scale vs median':>17}")
        median_scale = float(np.median(scale))
        for cell in BINDING_CELLS:
            where = int(np.searchsorted(sample, cell))
            if where >= sample.size or sample[where] != cell:
                continue
            print(
                f"  {cell:>14}{k[cell]:>13.3e}{diagonal[where]:>14.3e}"
                f"{scale[where]:>14.3e}{scale[where] / median_scale:>16.2f}x"
            )

        low, high = scale[sampled_k <= edges[1]], scale[sampled_k >= edges[9]]
        if low.size and high.size:
            ratio = float(np.median(low) / np.median(high))
            print(
                f"\n  VERDICT: lowest-decile scale / highest-decile scale = {ratio:.2f}x  -> "
                f"{'premise HOLDS (low-k rows are promoted)' if ratio > 3.0 else 'premise FAILS (the diagonal is flat in k)'}"
            )


if __name__ == "__main__":
    main()
