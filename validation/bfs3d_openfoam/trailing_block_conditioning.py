"""Can a BLOCK smoother accept the true ``[k, omega]`` Jacobian slice, where a point smoother cannot?

The JAX-native multigrid builders refuse the coupled Jacobian's trailing slice outright: they form
``D^-1`` twice -- once for the Jacobi-class smoother, once for the prolongation-smoothing damping -- so
``aquaflux.solve.multigrid`` rejects any operator with a non-positive diagonal rather than bake ``1/0``
into a frozen preconditioner or silently invert the sign of a correction. The slice violates that (a
recorded minimum diagonal of −2.1e+06 from the live source-term linearizations), which is why the native
path is built on each field's **transport operator** instead -- a 13x-sparser, source-clamped, per-field
stand-in that is a materially worse approximation of what is being solved.

**But that precondition is on the SCALAR diagonal, and a block smoother does not need it.** Point Jacobi
needs each ``a_ii`` invertible; block Jacobi needs each per-cell ``2x2`` invertible, which is strictly
weaker -- a block can be perfectly well conditioned with a negative entry on its diagonal. So this asks
the question that decides whether the approximation is necessary at all:

    at the cells where the scalar diagonal is non-positive, are the 2x2 cell blocks SINGULAR,
    or merely indefinite?

Singular, and a block smoother buys nothing -- the operator is genuinely degenerate there and the
transport-operator detour stands. Merely indefinite, and a block-smoothed native hierarchy could
precondition the **real** operator, dropping both approximations at once.

**What this does NOT settle.** Only the fine level. A hierarchy's coarse Galerkin operators ``R A P``
must be invertible too, and aggregation on an indefinite operator producing a poor coarse space is a
separate, recorded difficulty that this cannot speak to. A positive answer here licenses building the
thing and measuring it; it does not predict that it works.

Usage::

    python3 -u validation/bfs3d_openfoam/trailing_block_conditioning.py state-00057
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parents[1]))

import compare  # noqa: E402
from aquaflux.solve import (  # noqa: E402
    FieldGroups,
    MonolithicAmgPreconditioner,
    block_stencil_gather_map,
    build_convection_hierarchy,
)
from aquaflux.turbulence.coupled import (  # noqa: E402
    _coupled_jacobian_plan,
    _coupled_shift_policy,
    _frozen_shift_diagonal,
)
from field_split_probe import FLOOR, STATES, load_state, materialize  # noqa: E402

#: Below this the 2x2 is called singular. The blocks are compared on their own scale (the determinant
#: against the product of the row norms), so this is a genuine conditioning threshold rather than a
#: magnitude one -- an operator row scaled by 1e-6 is not thereby degenerate.
SINGULAR = 1e-10


def cell_blocks(block: sp.csr_matrix, n_cells: int) -> np.ndarray:
    """The per-cell dense ``2x2`` blocks, shape ``(n_cells, 2, 2)``.

    Field-major within the group -- degree of freedom ``(cell i, field f)`` sits at ``f * n_cells + i``
    -- so a cell's two unknowns are strided by ``n_cells`` and the blocks have to be gathered by index
    rather than sliced.
    """
    rows = np.arange(n_cells)
    out = np.empty((n_cells, 2, 2))
    for f in range(2):
        for g in range(2):
            out[:, f, g] = np.asarray(block[f * n_cells + rows, g * n_cells + rows]).ravel()
    return out


def report(blocks: np.ndarray, label: str) -> None:
    """Whether the non-positive-diagonal cells are singular or merely indefinite."""
    n = blocks.shape[0]
    diagonal = np.stack([blocks[:, 0, 0], blocks[:, 1, 1]], axis=1)
    bad = np.nonzero((diagonal <= 0).any(axis=1))[0]
    print(f"\n{label}")
    print(f"  cells                                   {n}")
    print(f"  with a non-positive scalar diagonal     {len(bad)}  ({100 * len(bad) / n:.2f} %)")
    if not len(bad):
        print("  -> the point-Jacobi precondition is not violated at this state at all.")
        return
    print(f"    of which k-row negative               {int((blocks[bad, 0, 0] <= 0).sum())}")
    print(f"    of which omega-row negative           {int((blocks[bad, 1, 1] <= 0).sum())}")

    det = np.linalg.det(blocks)
    # Scale-free: a determinant is only small RELATIVE to the entries it is built from.
    scale = np.linalg.norm(blocks, axis=(1, 2)) ** 2
    relative = np.abs(det) / np.maximum(scale, np.finfo(float).tiny)
    singular = relative < SINGULAR
    print(
        f"  scaled |det| over the bad cells         "
        f"min {relative[bad].min():.3e}  median {np.median(relative[bad]):.3e}"
    )
    print(f"  SINGULAR (scaled |det| < {SINGULAR:.0e}) among them   {int(singular[bad].sum())}")
    print(f"  singular anywhere in the block          {int(singular.sum())}")
    worst = bad[np.argmin(relative[bad])]
    with np.printoptions(precision=4, suppress=False):
        print(f"  worst bad cell {worst}:\n{blocks[worst]}")
    verdict = (
        "SINGULAR -- a block smoother buys nothing here; the transport-operator detour stands."
        if singular[bad].any()
        else "merely INDEFINITE -- every bad cell is block-invertible, so a block-smoothed native "
        "hierarchy could precondition the REAL slice. Fine level only; the coarse operators are "
        "a separate question."
    )
    print(f"  VERDICT: {verdict}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STATES:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{' | '.join(STATES)}>")
    name = sys.argv[1]
    march_beta, _, description = STATES[name]
    coupled = compare.build_case()["coupled"]
    n_fields, n_cells = coupled.layout.dim + 3, coupled.layout.n_cells
    groups = FieldGroups(
        n_cells=n_cells, n_leading_fields=coupled.layout.dim + 1, n_trailing_fields=2
    )
    print(f"{'=' * 88}\n{name}: {description}\nmarch shift {march_beta}\n{'=' * 88}", flush=True)
    state = load_state(name)
    base = _coupled_shift_policy(coupled, state, "twolevel")
    plan = _coupled_jacobian_plan(coupled, 3)
    jacobian = materialize(
        coupled,
        state,
        plan,
        block_stencil_gather_map(plan),
        n_fields,
    )
    # THREE pairings, and the unshifted one is the point. The pseudo-transient shift adds `beta * d`
    # to the diagonal, so a large enough beta can rescue a diagonal the bare Jacobian does not have --
    # which means "the slice has negative diagonal entries" is a claim about a SHIFT, not about the
    # operator. beta = 0 is not hypothetical: it is exactly the operator the implicitly-differentiated
    # adjoint's transpose solve meets, where there is no shift and no preconditioner floor to soften it.
    for label, beta in (
        ("UNSHIFTED -- the adjoint's operator", 0.0),
        ("march shift", march_beta),
        ("preconditioner floor -- the shipped build", max(march_beta, FLOOR)),
    ):
        shift = _frozen_shift_diagonal(base, beta, state) if beta > 0 else np.zeros(groups.n_dofs)
        shifted = MonolithicAmgPreconditioner._shifted(jacobian, shift)
        trailing = sp.csr_matrix(shifted[groups.trailing, :][:, groups.trailing])
        report(cell_blocks(trailing, n_cells), f"[k, omega] slice, {label} (beta {beta:g})")
        # The decisive question is not what the diagonal looks like but whether the builder ACCEPTS it:
        # if the shifted slice is admissible, the transport-operator detour is unnecessary on the
        # forward path and the native hierarchy could precondition the real operator directly.
        try:
            hierarchy = build_convection_hierarchy(trailing)
            print(
                f"  build_convection_hierarchy: ACCEPTED ({len(hierarchy.levels)} level(s))"
                if hasattr(hierarchy, "levels")
                else "  build_convection_hierarchy: ACCEPTED"
            )
        except Exception as failure:
            print(f"  build_convection_hierarchy: REFUSED -- {type(failure).__name__}: {failure}")
        del shifted, trailing


if __name__ == "__main__":
    main()
