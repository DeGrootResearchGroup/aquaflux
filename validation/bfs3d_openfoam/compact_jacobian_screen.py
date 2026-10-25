"""Can the field split be fitted to a COMPACT Jacobian and still precondition the exact operator? (bfs3d)

The shipped split stores the coupled Jacobian at its full reach of three cells, which on a polyhedral mesh
(about fifteen faces a cell) is some 237 cells a row against 63 on these hexahedra -- the memory wall for
a case of a few hundred thousand cells. This screen asks whether a split fitted to a Jacobian with its
far couplings removed still preconditions the exact operator, before anything is built to probe such a
Jacobian cheaply.

**Arms.** Each arm zeroes the stored entries of the probed Jacobian lying further than a given number of
cells from their row's cell, per (row field, column field) pair, immediately before EVERY fit and re-fit
(the anchor fit that freezes the coarsening included), so the split is built, coarsened and fitted
exactly as a march would build it from such a matrix. The kept entries keep their exact values, so an arm
measures what the dropped couplings were worth to the preconditioner and nothing else. The zeroed
positions stay stored: the padded operators keep their widths, so every arm costs the same per
application, and the comparison is of convergence alone (memory is the reason for the arms, not what
they measure).

  exact             nothing dropped (the control; reproduces the shipped split)
  reach2            every pair to distance two
  compact           continuity rows to distance two (Rhie--Chow's ring), every other row to one
  compact+coupling  ``compact``, but the [k, omega] <- [u, v, w, p] coupling block kept whole
  pressure2         every pair with a pressure row or column to distance two, the rest to one (added
                    after ``compact`` failed, to test whether the momentum <- pressure gradient's
                    distance-two mass is what it lost; not part of the registered bar)
  reach1            every pair to distance one, continuity rows too (expected to fail: it changes the
                    pressure rows the way a pressure-Laplacian augmentation does)

**Modes.** An arm is ``[mode:]truncation``. Without a mode (``host``) the tight solve applies the host
split, as the first screen did. ``traced:`` applies the traced split the step holds, widened to double
precision for the tight solves (the adjoint's form) and in the split's own precision for the march's
solve. ``mf:`` is ``traced:`` with the FINE level of both hierarchies and the coupling block applied
matrix-free -- each product an exact Jacobian-vector product of the residual (plus the shift the fit
added) instead of the stored operator -- while the smoothers' pieces and the coarse levels stay those
fitted to the truncated Jacobian. That is the storage a matrix-free fine level would keep: a split fitted
to a compact Jacobian, with the exact operator wherever the cycle multiplies by the fine one.
``mff:`` goes further: the leading fine level's SIMPLE smoother also forms its divergence, gradient and
Schur products as Jacobian-vector products, keeping only its per-cell velocity inverse and its Schur
diagonal -- the FULLY matrix-free fine level, which stores nothing at the fine level but per-cell blocks
and diagonals, the prolongation and the coarse levels (all of which a matrix-free build can obtain exactly
from a streamed probe without holding the Jacobian), so with the ``exact`` truncation it is the shipped
split's arithmetic and measures what that costs per application. ``mffh:`` replaces the one thing such a
build cannot obtain cheaply, the Frobenius-optimal Schur diagonal's row norms ``|S_i|^2`` (``S`` is reach
six), by a Hutchinson estimate from ``BFS3D_SCREEN_SCHUR_SAMPLES`` (default 32) random sign vectors, each
two Jacobian-vector products; ``S_ii`` itself stays exact.

**Measurements, per state.** The OPERATOR is always the exact one: the shifted Jacobian-vector product of
the true residual, never a materialized copy.

  march    the step's own solve (its residual-stop GMRES at the regime's tolerance, in the step's
           row-scaled measure) with the split the step would apply: restart cycles, applications of the
           preconditioner, and the linear residual reached in the Euclidean measure
  tight    right-preconditioned GMRES(15) to 1e-8 on the TRUE Euclidean residual with the host split:
           applications, and the true relative residual reached

At the converged root (``BFS3D_SCREEN_ROOT``) the operator is unshifted -- the adjoint's -- and ``tight``
runs forward on ``-R`` and transposed on a cotangent localized in the ``u`` block (the mean streamwise
velocity's), since the transpose is what the adjoint calls and a ``-R`` probe cannot rank it.

**Pass bar (registered before the run):** ``compact`` within 1.3 times ``exact``'s applications at every
state, the root's transpose included. For the matrix-free modes the same bar, ``mf:compact`` against
``traced:exact``.

Environment: ``BFS3D_SCREEN_STEPS`` (one-based steps, comma-separated: the step starts from
``state-<step-1>`` at the shift ``state-<step>`` records, rebuilt the way ``pivot_floor_probe.py`` rebuilds
it); ``BFS3D_SCREEN_ROOT`` (a converged checkpoint's step number, optional); ``BFS3D_SCREEN_ARMS`` (default
all, in the order above); ``BFS3D_PROBE_CHECKPOINTS`` (default this case's ``checkpoints/``).
"""

from __future__ import annotations

import gc
import inspect
import os
import sys
import time
from pathlib import Path

CASE = Path(__file__).resolve().parent
REPO = CASE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CASE))

import aquaflux.precondition.multigrid as multigrid  # noqa: E402
import aquaflux.precondition.saddle_multigrid as saddle_multigrid  # noqa: E402
import compare  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.precondition import SmoothedHierarchy  # noqa: E402
from aquaflux.precondition.materialized_session import jacobian_matvec  # noqa: E402
from aquaflux.precondition.probe_layout import ProbedBlocks  # noqa: E402
from aquaflux.solve import ConstantRelaxation  # noqa: E402
from aquaflux.solve.continuation import _shifted_solve  # noqa: E402
from aquaflux.solve.linear import in_progress_measure  # noqa: E402
from aquaflux.turbulence import (  # noqa: E402
    coupled_step,
    open_session,
    scale_both_blocks,
    scale_momentum_only,
)
from aquaflux.turbulence.coupled import coupled_scaled_norm  # noqa: E402

NAMES = ["u", "v", "w", "p", "k", "omega"]
U, P, K = 0, 3, 4
TIGHT_RTOL, RESTART, MAX_RESTARTS = 1e-8, 15, 40


def _reach(arm: str):
    """``(row field, column field) -> the farthest kept distance``, or ``None`` for the exact arm."""
    if arm == "exact":
        return None
    if arm == "reach2":
        return lambda r, c: 2
    if arm == "reach1":
        return lambda r, c: 1
    if arm == "compact":
        return lambda r, c: 2 if r == P else 1
    if arm == "pressure2":
        return lambda r, c: 2 if P in (r, c) else 1
    if arm == "compact+coupling":
        return lambda r, c: 3 if (r >= K and c < K) else (2 if r == P else 1)
    raise SystemExit(f"unknown arm {arm!r}")


ARMS = ["exact", "reach2", "compact", "compact+coupling", "pressure2", "reach1"]
MODES = ("host", "traced", "mf", "mff", "mffh")
MATRIX_FREE = ("mf", "mff", "mffh")
SCHUR_SAMPLES = int(os.environ.get("BFS3D_SCREEN_SCHUR_SAMPLES", "32"))


class FineSimple:
    """Routes the leading FINE level's SIMPLE correction to matrix-free products, by its velocity size.

    ``saddle_multigrid._simple_correction`` is looked up by name when the cycle is traced, so replacing it
    reaches every correction. A level whose velocity size has no entry in ``by_velocity`` keeps the
    stored correction; one that has gets ``(divergence, gradient, schur, schur_diagonal_inverse)``,
    the last ``None`` to keep the stored diagonal. ``calls`` counts traced replacements.
    """

    def __init__(self) -> None:
        self._original = saddle_multigrid._simple_correction
        self.by_velocity: dict[int, tuple] = {}
        self.calls = 0

    def install(self) -> None:
        def correction(pieces, residual, pressure_sweeps, weight):
            products = self.by_velocity.get(pieces.n_velocity)
            if products is None:
                return self._original(pieces, residual, pressure_sweeps, weight)
            self.calls += 1
            return self._matrix_free(products, pieces, residual, pressure_sweeps, weight)

        saddle_multigrid._simple_correction = correction

    @staticmethod
    def _matrix_free(products, pieces, residual, pressure_sweeps, weight):
        """The same predictor / Schur sweeps / correction as the stored one, every block product exact."""
        divergence, gradient, schur, diagonal = products
        dtype = residual.dtype
        nv = pieces.n_velocity
        velocity_residual, pressure_residual = residual[:nv], residual[nv:]

        def f_inverse(v):
            if pieces.f_block_inverse is not None:
                return pieces.f_block_inverse.apply(v.astype(dtype))
            return pieces.f_diagonal_inverse * v.astype(dtype)

        def dg(p):
            return f_inverse(gradient(p))

        s_inverse = pieces.schur_diagonal_inverse if diagonal is None else diagonal.astype(dtype)
        predictor = f_inverse(velocity_residual)
        rhs = pressure_residual - divergence(predictor).astype(dtype)
        if pressure_sweeps <= 0:
            pressure = jnp.zeros_like(rhs)
        else:
            pressure = weight * s_inverse * rhs

        def sweep(_, p):
            return p + weight * s_inverse * (rhs - schur(p, dg(p)).astype(dtype))

        pressure = jax.lax.fori_loop(0, max(pressure_sweeps - 1, 0), sweep, pressure)
        return jnp.concatenate([predictor - dg(pressure), pressure])


def split_arm(arm: str) -> tuple[str, str]:
    """``"[mode:]truncation"`` -> ``(mode, truncation)``, refusing an unknown mode or truncation."""
    mode, _, truncation = arm.rpartition(":")
    mode = mode or "host"
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r} in arm {arm!r}")
    _reach(truncation)
    return mode, truncation


class FineLevels:
    """Routes a hierarchy's FINE-level operator products to a supplied callable, by level size.

    Every level operator product in the traced cycles goes through ``multigrid._operator_matvec``
    (``saddle_multigrid`` holds its own reference to it), so replacing that one function reaches the
    V-cycle's residuals and both smoothers. A level is recognized by its static size: the leading fine
    level has ``4 n`` unknowns and the trailing one ``2 n``, and :meth:`check` refuses a coarse level of
    either size. ``calls`` counts how many products were traced through a supplied callable, so a run in
    which the replacement never took effect is visible rather than silently identical to the stored arm.
    """

    def __init__(self) -> None:
        self._original = multigrid._operator_matvec
        self.by_size: dict[int, object] = {}
        self.calls = 0

    def install(self) -> None:
        def dispatch(level, x):
            replacement = self.by_size.get(level.n)
            if replacement is None:
                return self._original(level, x)
            self.calls += 1
            return replacement(x)

        multigrid._operator_matvec = dispatch
        saddle_multigrid._operator_matvec = dispatch

    @staticmethod
    def hierarchy(cycle) -> SmoothedHierarchy:
        found = [
            leaf
            for leaf in jax.tree_util.tree_leaves(
                cycle.state, is_leaf=lambda x: isinstance(x, SmoothedHierarchy)
            )
            if isinstance(leaf, SmoothedHierarchy)
        ]
        if len(found) != 1:
            raise SystemExit(f"expected one hierarchy in a traced cycle, found {len(found)}")
        if found[0].equilibration is not None:
            raise SystemExit("an equilibrated hierarchy's fine level is D A D; not handled here")
        return found[0]

    def check(self, split, n: int) -> None:
        for name, cycle, fine in (
            ("leading", split.leading_cycle, 4 * n),
            ("trailing", split.trailing_cycle, 2 * n),
        ):
            sizes = [level.n for level in self.hierarchy(cycle).levels]
            if sizes[0] != fine or fine in sizes[1:] or {4 * n, 2 * n} & set(sizes[1:]):
                raise SystemExit(f"{name} level sizes {sizes} do not single out the fine level")


class CellDistance:
    """Graph distance (0, 1, 2 or 3) between two cells over interior faces, for pairs within three."""

    def __init__(self, face_cells, n: int) -> None:
        interior = np.asarray(face_cells.interior)
        owner = np.asarray(face_cells.owner)[interior]
        neighbour = np.asarray(face_cells.neighbour)[interior]
        one = sp.coo_matrix(
            (np.ones(2 * owner.size), (np.r_[owner, neighbour], np.r_[neighbour, owner])),
            shape=(n, n),
        ).tocsr()
        one.data[:] = 1.0
        two = (one + sp.identity(n, format="csr")) @ one
        two.data[:] = 1.0
        self._n = n
        self._keys = [self._sorted_keys(one), self._sorted_keys(two)]

    def _sorted_keys(self, matrix: sp.csr_matrix) -> np.ndarray:
        coo = matrix.tocoo()
        return np.sort(coo.row.astype(np.int64) * self._n + coo.col)

    def __call__(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        keys = rows.astype(np.int64) * self._n + cols
        distance = np.full(rows.size, 3, dtype=np.int8)
        for d, table in ((2, self._keys[1]), (1, self._keys[0])):
            at = np.minimum(np.searchsorted(table, keys), table.size - 1)
            distance[table[at] == keys] = d
        distance[rows == cols] = 0
        return distance


class Truncation:
    """Zeroes a probe's out-of-distance entries in place, before every fit and re-fit of the split.

    Installed by replacing :meth:`ProbedBlocks.add_shift`, which both fit paths call on the freshly
    probed values just before the blocks reach the inverse (and after the pivot floor has read the
    diagonal, which no arm touches).
    """

    def __init__(self, distance: CellDistance, n_cells: int, n_fields: int) -> None:
        self._distance = distance
        self._n = n_cells
        self._nf = n_fields
        self._masks: dict[tuple[int, str], np.ndarray] = {}
        self._original = ProbedBlocks.add_shift
        self.arm = "exact"
        #: The diagonal the last fit added (the shift plus the pivot floor's correction), shape (n_dofs,).
        self.fitted_shift: np.ndarray | None = None

    def install(self) -> None:
        truncation = self

        def add_shift(probed, shift_diagonal):
            truncation.apply(probed)
            truncation.fitted_shift = np.asarray(shift_diagonal, dtype=np.float64)
            truncation._original(probed, shift_diagonal)

        ProbedBlocks.add_shift = add_shift

    def apply(self, probed: ProbedBlocks) -> None:
        reach = _reach(self.arm)
        if reach is None:
            return
        key = (id(probed.layout), self.arm)
        if key not in self._masks:
            self._masks = {key: self._mask(probed, reach)}
            self._report(probed, self._masks[key])
        probed.data = jnp.where(self._masks[key], 0.0, probed.data)

    def _entries(self, probed: ProbedBlocks):
        """Global (row dof, column dof) of every stored value, in the block-major order of ``data``."""
        layout = probed.layout
        offsets = (0, layout.leading_dofs)
        rows, cols = np.empty(probed.data.size, np.int64), np.empty(probed.data.size, np.int64)
        for block, (start, stop, pattern) in enumerate(layout._blocks):
            indptr, indices = pattern.indptr, pattern.indices
            rows[start:stop] = (
                np.repeat(np.arange(indptr.size - 1), np.diff(indptr)) + offsets[block // 2]
            )
            cols[start:stop] = indices.astype(np.int64) + offsets[block % 2]
        return rows, cols

    def _mask(self, probed: ProbedBlocks, reach) -> np.ndarray:
        rows, cols = self._entries(probed)
        distance = self._distance(rows % self._n, cols % self._n)
        limit = np.empty((self._nf, self._nf), dtype=np.int8)
        for r in range(self._nf):
            for c in range(self._nf):
                limit[r, c] = reach(r, c)
        return distance > limit[rows // self._n, cols // self._n]

    def _report(self, probed: ProbedBlocks, mask: np.ndarray) -> None:
        rows, cols = self._entries(probed)
        data = np.asarray(probed.data)
        pair = (rows // self._n) * self._nf + cols // self._n
        total = np.bincount(pair, weights=data**2, minlength=self._nf**2)
        dropped = np.bincount(pair[mask], weights=data[mask] ** 2, minlength=self._nf**2)
        kept = np.bincount(pair[~mask], minlength=self._nf**2)
        print(
            f"[{self.arm}] keeps {int((~mask).sum()) / 1e6:.2f}M of {mask.size / 1e6:.2f}M stored entries "
            f"({int((~mask).sum()) / self._n:.0f} a cell); Frobenius share dropped per (row, column) pair, "
            f"at the fit it was first applied to:",
            flush=True,
        )
        print("        " + " ".join(f"{c:>8s}" for c in NAMES), flush=True)
        for r in range(self._nf):
            cells = []
            for c in range(self._nf):
                i = r * self._nf + c
                cells.append(
                    "       -"
                    if kept[i] == 0 and total[i] == 0
                    else f"{np.sqrt(dropped[i] / total[i]) if total[i] else 0.0:8.1e}"
                )
            print(f"  {NAMES[r]:>5s} " + " ".join(cells), flush=True)


def right_gmres(apply_a, apply_m, b, rtol):
    """Right-preconditioned restarted GMRES judged on the TRUE residual: ``(x, applications, true |r|/|b|)``.

    The Arnoldi recurrence of a right-preconditioned solve tracks ``b - A x`` itself, so the per-iteration
    stop is on the true residual (to rounding); the verdict recomputes it.
    """
    n = b.size
    x = np.zeros(n)
    bnorm = np.linalg.norm(b)
    applications = 0
    for _ in range(MAX_RESTARTS):
        r = b - apply_a(x)
        beta = np.linalg.norm(r)
        if beta <= rtol * bnorm:
            break
        V = np.zeros((RESTART + 1, n))
        Z = np.zeros((RESTART, n))
        H = np.zeros((RESTART + 1, RESTART))
        V[0] = r / beta
        e1 = np.zeros(RESTART + 1)
        e1[0] = beta
        for j in range(RESTART):
            Z[j] = apply_m(V[j])
            applications += 1
            w = apply_a(Z[j])
            for _twice in range(2):
                h = V[: j + 1] @ w
                H[: j + 1, j] += h
                w = w - h @ V[: j + 1]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 0:
                V[j + 1] = w / H[j + 1, j]
            y, *_ = np.linalg.lstsq(H[: j + 2, : j + 1], e1[: j + 2], rcond=None)
            estimate = np.linalg.norm(H[: j + 2, : j + 1] @ y - e1[: j + 2])
            if estimate <= rtol * bnorm or H[j + 1, j] == 0:
                break
        x = x + y @ Z[: y.size]
    true = float(np.linalg.norm(b - apply_a(x)) / bnorm)
    return x, applications, true


def traced_apply(
    split, mode, fine, assembler, phi, fitted_shift, n, simple=None, schur_diagonal=None
):
    """The split's map ``r -> M r`` as the step would apply it, or with its fine levels matrix-free.

    ``split`` is the step's :class:`~aquaflux.precondition.TracedFieldSplit` in whichever precision it
    is to be applied in. In ``mf`` mode the composition is spelled out here -- leading cycle, coupling,
    trailing cycle, as ``TracedFieldSplit.apply`` does -- with the coupling block, and each hierarchy's
    fine-level operator (through ``fine``), replaced by exact Jacobian-vector products plus the diagonal
    the fit added. Each product is formed in double precision and returned in its argument's.
    """
    if mode not in MATRIX_FREE:
        fine.by_size = {}
        if simple is not None:
            simple.by_velocity = {}
        return split.matvec()
    lead_dofs = split.groups.n_leading_dofs
    shift = jnp.asarray(fitted_shift)

    # Through the library's compiled product, the assembler and state its ARGUMENTS: closing over them
    # instead bakes each product's residual intermediates into the program as constants, eleven times
    # per application and once per program (a first run held 22 GB by its second arm and was stopped).
    def jacobian(tangent):
        return jacobian_matvec(assembler, phi, tangent)

    def leading(x):
        x64 = x.astype(jnp.float64)
        full = jnp.concatenate([x64, jnp.zeros(2 * n)])
        return (jacobian(full)[:lead_dofs] + shift[:lead_dofs] * x64).astype(x.dtype)

    def trailing(x):
        x64 = x.astype(jnp.float64)
        full = jnp.concatenate([jnp.zeros(lead_dofs), x64])
        return (jacobian(full)[lead_dofs:] + shift[lead_dofs:] * x64).astype(x.dtype)

    def coupling(x):
        full = jnp.concatenate([x.astype(jnp.float64), jnp.zeros(2 * n)])
        return jacobian(full)[lead_dofs:].astype(x.dtype)

    fine.check(split, n)
    fine.by_size = {lead_dofs: leading, 2 * n: trailing}
    if mode in ("mff", "mffh"):
        nv = 3 * n
        p_shift = shift[nv:lead_dofs]

        def divergence(u):
            full = jnp.concatenate([u.astype(jnp.float64), jnp.zeros(3 * n)])
            return jacobian(full)[nv:lead_dofs]

        def gradient(p):
            full = jnp.concatenate([jnp.zeros(nv), p.astype(jnp.float64), jnp.zeros(2 * n)])
            return jacobian(full)[:nv]

        def schur(p, dg_p):
            p64 = p.astype(jnp.float64)
            full = jnp.concatenate([-dg_p.astype(jnp.float64), p64, jnp.zeros(2 * n)])
            return jacobian(full)[nv:lead_dofs] + p_shift * p64

        diagonal = None if mode == "mff" else jnp.asarray(schur_diagonal)
        simple.by_velocity = {nv: (divergence, gradient, schur, diagonal)}
    else:
        simple.by_velocity = {}

    def apply(residual):
        narrowed = residual.astype(split.precision)
        lead, trail = narrowed[:lead_dofs], narrowed[lead_dofs:]
        first = split.leading_cycle(lead)
        rest = split.trailing_cycle(trail - coupling(first))
        return jnp.concatenate([first, rest]).astype(residual.dtype)

    return apply


def wiring(split, fine, n, apply_mf) -> None:
    """How far the matrix-free fine operators are from the stored ones, and that they were traced in.

    For the exact truncation the two differ by the stored copy's rounding (the split's precision) and
    the probe's own; for a truncated one, by what was dropped. A difference of exactly zero, or no traced
    call, would mean the replacement never took effect.
    """
    rng = np.random.default_rng(7)
    for name, cycle, size in (
        ("leading", split.leading_cycle, 4 * n),
        ("trailing", split.trailing_cycle, 2 * n),
    ):
        level = fine.hierarchy(cycle).levels[0]
        x = jnp.asarray(rng.standard_normal(size), dtype=level.operator.values.dtype)
        stored = np.asarray(fine._original(level, x), dtype=np.float64)
        exact = np.asarray(fine.by_size[size](x), dtype=np.float64)
        print(
            f"    [wiring] {name} fine level: |stored - exact| / |exact| = "
            f"{np.linalg.norm(stored - exact) / np.linalg.norm(exact):.2e}",
            flush=True,
        )
    before = fine.calls
    jax.jit(apply_mf).lower(jnp.zeros(6 * n))
    print(
        f"    [wiring] fine-level products traced through the replacement: {fine.calls - before}",
        flush=True,
    )


def leading_fine(split):
    """The leading hierarchy's fine level and its SIMPLE pieces, from the traced split's state."""
    hierarchy, extras = split.leading_cycle.state
    level = hierarchy.levels[0]
    return level, extras[level.n]


def _bytes(tree) -> int:
    return sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf))


def storage(split, n) -> None:
    """What the split holds, per hierarchy level, and what a fully matrix-free fine level would drop.

    Dropped: both fine levels' operators (values, columns and stored transposes), the leading fine
    level's divergence, velocity-correction and Schur operators, and the coupling block. Kept: per-cell
    blocks and diagonals, the prolongations, every coarse level.
    """
    total = _bytes(split)
    dropped = _bytes(split.coupling)
    print(
        f"    [storage] split {total / 2**20:.0f} MiB (coupling {_bytes(split.coupling) / 2**20:.0f})",
        flush=True,
    )
    for name, cycle in (("leading", split.leading_cycle), ("trailing", split.trailing_cycle)):
        hierarchy, extras = cycle.state
        for i, level in enumerate(hierarchy.levels):
            pieces = extras.get(level.n) if isinstance(extras, dict) else None
            line = f"    [storage] {name} level {i} ({level.n} dofs): operator {_bytes(level.operator) / 2**20:.0f} MiB, level {_bytes(level) / 2**20:.0f} MiB"
            if pieces is not None:
                heavy = {k: _bytes(getattr(pieces, k)) for k in ("dg", "divergence", "schur")}
                line += ", pieces " + ", ".join(f"{k} {v / 2**20:.0f}" for k, v in heavy.items())
                line += f" (all {_bytes(pieces) / 2**20:.0f})"
                if i == 0:
                    dropped += sum(heavy.values())
            if i == 0:
                dropped += _bytes(level.operator)
            print(line, flush=True)
    print(
        f"    [storage] a fully matrix-free fine level drops {dropped / 2**20:.0f} of {total / 2**20:.0f} MiB "
        f"({dropped / total:.0%})",
        flush=True,
    )


def schur_diagonal_estimate(split, assembler, phi, fitted_shift, n, samples):
    """The Frobenius-optimal Schur diagonal ``S_ii / |S_i|^2`` with ``|S_i|^2`` estimated matrix-free.

    ``S p = [J(-F~^-1 G p, p)]_p + shift_p p`` costs two Jacobian-vector products; with Rademacher ``z``,
    ``E[(S z)_i^2] = |S_i|^2``. ``S_ii`` is taken exact from the stored pieces (a streamed probe gives it
    exactly too). Prints the estimate's error against the stored diagonal's implied ``|S_i|^2``.
    """
    level, pieces = leading_fine(split)
    nv, lead = 3 * n, 4 * n
    shift = jnp.asarray(fitted_shift)

    @jax.jit
    def apply_s(z):
        g = jacobian_matvec(assembler, phi, jnp.concatenate([jnp.zeros(nv), z, jnp.zeros(2 * n)]))[
            :nv
        ]
        dg = pieces.f_block_inverse.apply(g.astype(pieces.f_diagonal_inverse.dtype)).astype(
            jnp.float64
        )
        full = jnp.concatenate([-dg, z, jnp.zeros(2 * n)])
        return jacobian_matvec(assembler, phi, full)[nv:lead] + shift[nv:lead] * z

    rng = np.random.default_rng(11)
    squares = np.zeros(n)
    for _ in range(samples):
        z = jnp.asarray(rng.choice([-1.0, 1.0], size=n))
        squares += np.asarray(apply_s(z)) ** 2
    squares /= samples
    matrix = level.operator.to_scipy().tocsr()
    c_diagonal = matrix.diagonal().astype(np.float64)[nv:lead]
    d = pieces.divergence.to_scipy().tocsr().astype(np.float64)
    dg = pieces.dg.to_scipy().tocsr().astype(np.float64)
    s_diagonal = c_diagonal - np.asarray(d.multiply(dg.T).sum(axis=1)).ravel()
    stored = np.asarray(pieces.schur_diagonal_inverse, dtype=np.float64)
    implied = s_diagonal / stored
    error = np.abs(squares / implied - 1.0)
    print(
        f"    [schur] |S_i|^2 from {samples} samples against the stored diagonal's: relative error "
        f"median {np.median(error):.2f}, 90th {np.percentile(error, 90):.2f}, max {error.max():.2f}",
        flush=True,
    )
    return s_diagonal / squares


def wiring_simple(split, simple, n) -> None:
    """The matrix-free divergence, velocity correction and Schur against the pieces' products."""
    _, pieces = leading_fine(split)
    divergence, gradient, schur, _ = simple.by_velocity[3 * n]
    rng = np.random.default_rng(3)
    u = jnp.asarray(rng.standard_normal(3 * n))
    p = jnp.asarray(rng.standard_normal(n))
    dtype = pieces.f_diagonal_inverse.dtype
    dg = pieces.f_block_inverse.apply(gradient(p).astype(dtype))
    pairs = (
        ("divergence", divergence(u), pieces.divergence.apply(u.astype(dtype))),
        ("velocity correction", dg, pieces.dg.apply(p.astype(dtype))),
        ("Schur", schur(p, dg), pieces.schur_apply(p.astype(dtype))),
    )
    for name, exact, stored in pairs:
        exact, stored = np.asarray(exact, np.float64), np.asarray(stored, np.float64)
        print(
            f"    [wiring] fine {name}: |stored - exact| / |exact| = "
            f"{np.linalg.norm(stored - exact) / np.linalg.norm(exact):.2e}",
            flush=True,
        )


def measure_state(
    arm, label, session, base, coupled, assembler, phi, beta, unshifted, truncation, fine, simple
) -> None:
    """Re-fit the arm's split at one state and print its ``march`` and ``tight`` solves."""
    mode, _ = split_arm(arm)
    n = coupled.layout.n_cells
    session.rebind(assembler)
    active = eqx.tree_at(
        lambda s: s.relaxation_schedule, base, ConstantRelaxation(jnp.asarray(beta))
    )
    update = session.refresh_preconditioner(active, phi, due=True)
    if update is not None:
        active = update(active)
    active = active.with_norm(coupled_scaled_norm(coupled, active.shift_policy, phi))
    norm = active.residual_norm
    residual_fn = assembler.residual
    g0 = residual_fn(phi)
    term = active.shift_policy.shift_term(phi, g0)
    strength = active.relaxation_schedule.shift_strength(norm(g0), norm(g0))
    shift = jnp.zeros_like(phi) if unshifted else jax.lax.stop_gradient(term.shift(strength))
    jvp = jax.jit(lambda v: jax.jvp(residual_fn, (phi,), (v,))[1] + shift * v)
    _, vjp_fn = jax.vjp(residual_fn, phi)
    vjp = jax.jit(lambda v: vjp_fn(v)[0] + shift * v)

    def a(v):
        return np.asarray(jvp(jnp.asarray(v)), dtype=np.float64)

    def at(v):
        return np.asarray(vjp(jnp.asarray(v)), dtype=np.float64)

    if mode == "host":
        inverse = session.preconditioner.inverse
        march_m = term.make_preconditioner(strength)

        def m(v):
            return np.asarray(
                inverse.apply(np.asarray(v, np.float64), transpose=False), dtype=np.float64
            )

        def mt(v):
            return np.asarray(
                inverse.apply(np.asarray(v, np.float64), transpose=True), dtype=np.float64
            )

    else:
        # A fresh trace per state: the matrix-free products capture this state, and the cycles' own
        # compiled programs would otherwise be reused with the previous state's.
        jax.clear_caches()
        split = session._split
        widened = split.in_precision("float64")
        diagonal = None
        if mode in MATRIX_FREE:
            storage(split, n)
        if mode == "mffh":
            diagonal = schur_diagonal_estimate(
                widened, assembler, phi, truncation.fitted_shift, n, SCHUR_SAMPLES
            )
        options = dict(simple=simple, schur_diagonal=diagonal)
        march_m = traced_apply(
            split, mode, fine, assembler, phi, truncation.fitted_shift, n, **options
        )
        wide = traced_apply(
            widened, mode, fine, assembler, phi, truncation.fitted_shift, n, **options
        )
        if mode in MATRIX_FREE:
            before = simple.calls
            wiring(widened, fine, n, wide)
            if mode != "mf":
                wiring_simple(widened, simple, n)
                print(
                    f"    [wiring] fine SIMPLE corrections traced through the replacement: "
                    f"{simple.calls - before}",
                    flush=True,
                )
        wide_jit = jax.jit(wide)
        wide_t = jax.jit(jax.linear_transpose(wide, jnp.zeros(6 * n)))

        def m(v):
            return np.asarray(wide_jit(jnp.asarray(v)), dtype=np.float64)

        def mt(v):
            return np.asarray(wide_t(jnp.asarray(v))[0], dtype=np.float64)

    shown = 0.0 if unshifted else beta
    if not unshifted:
        solver = in_progress_measure(active.linear_solver(), norm)
        applications = [0]

        def count():
            applications[0] += 1

        def counted(v):
            jax.debug.callback(count)
            return march_m(v)

        t0 = time.perf_counter()
        d, cycles = _shifted_solve(residual_fn, phi, g0, shift, counted, solver)
        linear = float(jnp.linalg.norm(jvp(d) + g0) / jnp.linalg.norm(g0))
        print(
            f"[{arm}] {label} beta {shown:.3g} march: cycles {int(cycles):2d} applications "
            f"{applications[0]:4d}  Euclidean {linear:.2e}  {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    t0 = time.perf_counter()
    _, apps, true = right_gmres(a, m, -np.asarray(g0, dtype=np.float64), TIGHT_RTOL)
    print(
        f"[{arm}] {label} beta {shown:.3g} tight forward: applications {apps:4d} true {true:.2e}  "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    if unshifted:
        cotangent = np.zeros(phi.shape[0])
        cotangent[U * n : (U + 1) * n] = 1.0 / n
        t0 = time.perf_counter()
        _, apps, true = right_gmres(at, mt, cotangent, TIGHT_RTOL)
        print(
            f"[{arm}] {label} beta 0 tight TRANSPOSE: applications {apps:4d} true {true:.2e}  "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    fine.by_size = {}
    simple.by_velocity = {}


def main() -> None:
    steps = [int(s) for s in os.environ.get("BFS3D_SCREEN_STEPS", "").split(",") if s]
    root = os.environ.get("BFS3D_SCREEN_ROOT")
    arms = [a for a in os.environ.get("BFS3D_SCREEN_ARMS", ",".join(ARMS)).split(",") if a]
    for arm in arms:
        split_arm(arm)
    checkpoints = Path(os.environ.get("BFS3D_PROBE_CHECKPOINTS", CASE / "checkpoints"))
    if not steps and root is None:
        raise SystemExit("set BFS3D_SCREEN_STEPS and/or BFS3D_SCREEN_ROOT")

    solver_spec = compare.SOLVER
    ramp = solver_spec.continuation
    coupled = compare.build_case()["coupled"]
    n = coupled.layout.n_cells
    companion = {"flow": scale_momentum_only, "both": scale_both_blocks, None: scale_both_blocks}[
        ramp.scale
    ]

    def assembler_for(station):
        return (
            coupled
            if station == ramp.stations
            else companion(coupled, ramp.anchor ** (1.0 - station / ramp.stations))
        )

    def load(step):
        with np.load(checkpoints / f"state-{step:05d}.npz") as data:
            return jnp.asarray(data["state"]), float(data["shift"])

    # (label, state, station, beta, unshifted operator?)
    states = []
    for step in steps:
        phi, _ = load(step - 1)
        _, beta = load(step)
        states.append((f"step {step}", phi, min(step - 1, ramp.stations), beta, False))
    if root is not None:
        phi, beta = load(int(root))
        states.append((f"root {root}", phi, ramp.stations, beta, True))

    anchor = assembler_for(0)
    seed = anchor.state_from_physical(*hybrid_initialize(anchor))
    settings = solver_spec.settings()
    allowed = inspect.signature(coupled_step).parameters
    march = {k: v for k, v in settings.items() if k in allowed and k != "preconditioner"}
    truncation = Truncation(
        CellDistance(coupled.momentum.mesh.face_cells, n), n, coupled.layout.n_fields
    )
    truncation.install()
    fine = FineLevels()
    fine.install()
    simple = FineSimple()
    simple.install()
    print(
        f"[screen] {n} cells; states {[s[0] for s in states]}; arms {arms}; split precision "
        f"{compare.SPLIT_PRECISION}, pivot floor {compare.PIVOT_FLOOR}; tight GMRES({RESTART}) to "
        f"{TIGHT_RTOL:g}, {MAX_RESTARTS} restarts",
        flush=True,
    )

    for arm in arms:
        truncation.arm = split_arm(arm)[1]
        began = time.perf_counter()
        session = open_session(settings["preconditioner"], coupled)
        session.rebind(anchor)
        base = session.build(seed, **march)
        print(f"[{arm}] anchor fit {time.perf_counter() - began:.1f}s", flush=True)
        for label, phi, station, beta, unshifted in states:
            measure_state(
                arm,
                label,
                session,
                base,
                coupled,
                assembler_for(station),
                phi,
                beta,
                unshifted,
                truncation,
                fine,
                simple,
            )
        del session, base
        gc.collect()
        jax.clear_caches()


if __name__ == "__main__":
    main()
