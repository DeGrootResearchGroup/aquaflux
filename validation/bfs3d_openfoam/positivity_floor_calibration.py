"""What step length would a floored positivity limiter have allowed, at every recorded clip?

The fraction-to-the-boundary limiter caps the whole step at ``tau * min_i(phi_i / -d(phi)_i)`` over the
cells the step decreases. Because that is a **minimum over every cell**, one cell whose turbulent
kinetic energy is numerically zero -- and whose equation is therefore asking it to be zero -- throttles
the entire march: the recorded clips run down to ``1e-09`` while the binding cell's ``k`` falls by a
factor ``1 - tau`` per clipped step, its correction unchanged. Protecting such a cell's positivity to
*relative* precision buys nothing, and the failure it was introduced to prevent (a negative ``k``
reaching a bare ``sqrt`` and poisoning the residual) is now closed at every consumer of the solved
``k``, each of which clamps at zero.

Two ways to stop a numerically-dead cell from setting the cap, both keeping the step a **scalar**
multiple of the correction, so the line search still tests points on its own search ray:

``exempt``
    Take the minimum only over cells with ``phi_i`` above a floor. One comparison; a cell crossing the
    floor changes the cap discontinuously.
``softened``
    ``room_i = (phi_i + floor) / -d(phi)_i``. A near-zero cell gets a floor's worth of *absolute* room
    rather than meaningless relative room, and the cap moves continuously as ``phi_i`` crosses the
    floor -- which matters because the cap feeds controls that react to it.

Both leave the limiter inactive at a root (there ``d(phi) = 0``, so the room is infinite whatever the
floor), which is what keeps it out of the converged state and therefore out of the adjoint.

This replays both against the recorded clips rather than re-running anything: each step-limit dump
stores the ``(state, delta)`` pair the limiter was called with, so the cap under any candidate floor is
arithmetic on data already on disk. The question a floor has to answer is not only "does it free the
stuck cell" but "**what binds instead**" -- a floor is only safe if the cell that takes over has a
physically meaningful value, so the constrained population is reported alongside every cap.

Usage::

    python3 -u positivity_floor_calibration.py [<dump> ...]

With no arguments every ``step-limit-*.npz`` under ``checkpoints/`` is replayed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent

#: The limiter's fraction-to-the-boundary constant, as shipped.
TAU = 0.99

#: The constrained block is the k field: the flat state is [u, v, w, p, k, omega] over the cells.
N_CELLS = 23040
K_START, K_STOP = 4 * N_CELLS, 5 * N_CELLS

#: Floors to sweep. The case's inlet k is 0.375 and the mesh median is ~3e-2, so everything here is far
#: below any physically live value; the question is only how far below.
FLOORS = (0.0, 1e-20, 1e-16, 1e-14, 1e-13, 1e-12, 1e-10, 1e-8, 1e-6)


def cap_exempt(k, dk, floor):
    """Cap with cells at or below ``floor`` excluded from the minimum."""
    decreasing = (dk < 0.0) & (k > floor)
    room = np.where(decreasing, k / np.where(decreasing, -dk, 1.0), np.inf)
    return min(1.0, TAU * float(room.min())), int(np.argmin(room)), room


def cap_softened(k, dk, floor):
    """Cap with each cell given ``floor`` of absolute room on top of its own value."""
    decreasing = dk < 0.0
    room = np.where(decreasing, (k + floor) / np.where(decreasing, -dk, 1.0), np.inf)
    return min(1.0, TAU * float(room.min())), int(np.argmin(room)), room


def replay(path):
    stored = np.load(path)
    state, delta = stored["state"], stored["delta"]
    k, dk = state[K_START:K_STOP], delta[K_START:K_STOP]
    recorded = float(stored["cap"])

    shipped, cell, _ = cap_exempt(k, dk, -np.inf)
    status = "ok" if abs(shipped - recorded) <= 1e-9 * max(recorded, 1e-30) else "MISMATCH"
    print(f"\n{'=' * 104}")
    print(
        f"{path.stem}: recorded cap {recorded:.6e}, beta {float(stored['beta']):g}  "
        f"[replay of the shipped rule: {shipped:.6e} -- {status}]"
    )
    if status != "ok":
        print(
            "  ⚠️ the replay does not reproduce the recorded cap; every row below is untrustworthy."
        )
    decreasing = dk < 0.0
    print(
        f"  {int(decreasing.sum())} cells decreasing; binding cell {cell} has k {k[cell]:.3e}, "
        f"dk {dk[cell]:.3e}  (mesh median k {np.median(k):.3e})"
    )
    print(
        f"\n  {'floor':>10}{'cap (exempt)':>15}{'binds':>8}{'its k':>12}"
        f"{'cap (softened)':>17}{'binds':>8}{'its k':>12}{'n below floor':>15}"
    )
    for floor in FLOORS:
        cap_a, cell_a, _ = cap_exempt(k, dk, floor)
        cap_b, cell_b, _ = cap_softened(k, dk, floor)
        below = int((k <= floor).sum())
        print(
            f"  {floor:>10.0e}{cap_a:>15.4e}{cell_a:>8}{k[cell_a]:>12.2e}"
            f"{cap_b:>17.4e}{cell_b:>8}{k[cell_b]:>12.2e}{below:>15}"
        )
    return k, dk, recorded


def main() -> None:
    names = sys.argv[1:]
    paths = (
        [CASE / "checkpoints" / f"{n}.npz" for n in names]
        if names
        else sorted((CASE / "checkpoints").glob("step-limit-*.npz"))
    )
    if not paths:
        raise SystemExit("no step-limit dumps found under checkpoints/")

    replayed = [replay(p) for p in paths]

    print(f"\n{'=' * 104}\nthe WORST cap each floor leaves, over every recorded clip\n{'=' * 104}")
    # Counting "how many are still capped" is the wrong summary: a cap of 0.87 is the limiter working,
    # not failing. What decides whether the march grinds is the SMALLEST cap any step is given, and
    # which cell sets it -- a floor is doing its job once the worst cap is a workable step length set by
    # a cell with a physically meaningful value.
    print(
        f"  {'floor':>10}{'exempt: worst cap':>20}{'set by':>9}{'its k':>11}"
        f"{'softened: worst cap':>22}{'set by':>9}{'its k':>11}"
    )
    for floor in FLOORS:
        arms = []
        for rule in (cap_exempt, cap_softened):
            per_dump = [(*rule(k, dk, floor)[:2], k) for k, dk, _ in replayed]
            cap, cell, k = min(per_dump, key=lambda row: row[0])
            arms.append((cap, cell, float(k[cell])))
        (ca, la, ka), (cb, lb, kb) = arms
        print(f"  {floor:>10.0e}{ca:>20.4e}{la:>9}{ka:>11.2e}{cb:>22.4e}{lb:>9}{kb:>11.2e}")
    print(
        "\n  A floor is only safe if what binds INSTEAD is a physically live cell -- read the 'its k'\n"
        "  columns above, not just the caps. These dumps are all from deep in a lock-up (they were\n"
        "  written only when the cap fell below 0.05), so they show that a floor frees the stuck cell;\n"
        "  they cannot show that a healthy clip survives, because no healthy clip was ever dumped."
    )


if __name__ == "__main__":
    main()
