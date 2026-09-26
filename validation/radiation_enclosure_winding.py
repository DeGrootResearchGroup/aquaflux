"""What does the cell-in-the-metal check cost, and what do its compiled kernel and its skip buy?

``checks.enclosure_winding`` sums every facet's signed solid angle at every point: ``n_points x
n_facets`` pairs, the same product the receiver mask pays, which is why the model does not call it
by default. Two changes made it cheaper, and this harness measures each on its own, all arms in one
process on the same points, alternating so no ratio spans a run boundary:

1. **The per-pass kernel is compiled.** Eagerly each operation of the solid-angle kernel writes a
   whole pass of pairs out to memory for the next one to read back; compiled, they fuse. Arm
   ``eager`` is the per-pass loop as it was, arm ``compiled`` is the same loop through the shipped
   kernel with no skipping, so the pair between them is the compile alone.
2. **A closed piece is not summed outside its bounding box**, where its winding number is exactly
   zero. Arm ``shipped`` is the public function, so ``compiled`` against ``shipped`` is the skip
   alone, and what it is worth is the share of points inside the box -- printed beside it.

A third section re-asks a claim the subsystem record made under eager evaluation: that on a unit
box the winding number is ``1`` "to the last bit" a thousandth of a box-width from a wall. It
prints the distance from ``1`` in units in the last place, eager and compiled.

Every arm's answers are compared: inside/outside must agree exactly, and the largest difference in
the winding numbers is printed. Neither is a closure identity -- different summation orders can
disagree, and a wrong skip would disagree by the whole of a piece's contribution.

Run with ``validation/run_case.sh validation/radiation_enclosure_winding.py``. ``RADIATION_WINDING_SECTORS``
(default 2000, i.e. 8,000 facets) and ``RADIATION_WINDING_POINTS`` (default 4,000) set the size.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
from aquaflux.radiation.checks import _signed_total, enclosure_winding
from aquaflux.radiation.solid_angle import signed_solid_angle
from aquaflux.radiation.work import DEFAULT_PAIR_LIMIT, receivers_per_pass
from tests.unit.radiation_references import closed_drum, inward_box

SECTORS = int(os.environ.get("RADIATION_WINDING_SECTORS", "2000"))
POINTS = int(os.environ.get("RADIATION_WINDING_POINTS", "4000"))
PASSES = 2


def eager_winding(vertices: np.ndarray, points: np.ndarray, pair_limit: int) -> np.ndarray:
    """The per-pass loop as it stood before compiling: every operation dispatched on its own."""
    facets = jnp.asarray(vertices)
    chunk = receivers_per_pass(pair_limit, len(vertices))
    totals = [
        np.asarray(
            jnp.sum(
                signed_solid_angle(
                    jnp.asarray(points[first : first + chunk])[:, None, :], facets[None]
                ),
                axis=-1,
            )
        )
        for first in range(0, len(points), chunk)
    ]
    return np.concatenate(totals) / (4.0 * np.pi)


def compiled_winding(vertices: np.ndarray, points: np.ndarray, pair_limit: int) -> np.ndarray:
    """The shipped compiled kernel over every facet at every point: no pieces, no skipping."""
    return _signed_total(points, vertices, pair_limit) / (4.0 * np.pi)


def shipped_winding(vertices: np.ndarray, points: np.ndarray, pair_limit: int) -> np.ndarray:
    """The public function: compiled, and closed pieces skipped outside their boxes."""
    return enclosure_winding(vertices, points, pair_limit=pair_limit)


ARMS = {"eager": eager_winding, "compiled": compiled_winding, "shipped": shipped_winding}


def cost() -> None:
    """Time the three arms on one closed drum and one set of points."""
    vertices = closed_drum(SECTORS)
    rng = np.random.default_rng(0)
    # A cube three drum-widths across, so most points are clear of the drum's own box -- the
    # regime a lamp inside a reactor is in.
    points = rng.uniform(-3.0, 3.0, (POINTS, 3))
    corners = vertices.reshape(-1, 3)
    in_box = np.all((points >= corners.min(0)) & (points <= corners.max(0)), axis=1)
    pairs = len(points) * len(vertices)
    print(
        f"### cost: closed drum, {len(vertices):,} facets x {len(points):,} points "
        f"= {pairs / 1e6:.1f} Mpair, pair_limit {DEFAULT_PAIR_LIMIT:,}; "
        f"{in_box.mean():.1%} of points inside the drum's box\n",
        flush=True,
    )
    answers = {}
    for name, arm in ARMS.items():  # warm-up: compiles, and records the answers
        answers[name] = arm(vertices, points, DEFAULT_PAIR_LIMIT)
    best = {name: np.inf for name in ARMS}
    runs = {name: [] for name in ARMS}
    for _ in range(PASSES):
        for name, arm in ARMS.items():
            start = time.perf_counter()
            arm(vertices, points, DEFAULT_PAIR_LIMIT)
            seconds = time.perf_counter() - start
            runs[name].append(seconds)
            best[name] = min(best[name], seconds)
            print(f"  {name:>9}: {seconds:8.3f} s", flush=True)
    print(f"\n{'arm':>9} {'fastest s':>10} {'Mpair/s':>9} {'spread':>7} {'vs eager':>9}")
    for name in ARMS:
        spread = max(runs[name]) / min(runs[name])
        print(
            f"{name:>9} {best[name]:10.3f} {pairs / best[name] / 1e6:9.1f} {spread:6.2f}x "
            f"{best['eager'] / best[name]:8.2f}x"
        )
    print(
        f"\n  the skip alone (compiled / shipped): {best['compiled'] / best['shipped']:.2f}x, "
        f"against {1.0 / max(in_box.mean(), 1e-12):.1f}x if only the in-box points cost anything"
    )
    reference = answers["eager"]
    for name in ("compiled", "shipped"):
        same = np.array_equal(np.abs(answers[name]) > 0.5, np.abs(reference) > 0.5)
        moved = float(np.max(np.abs(answers[name] - reference)))
        print(f"  {name} against eager: inside/outside identical {same}, max |diff| {moved:.2e}")
    outside = ~in_box
    print(
        f"  shipped at the {outside.sum():,} points outside the box: "
        f"all exactly 0.0 {bool(np.all(answers['shipped'][outside] == 0.0))}; "
        f"eager's largest there {float(np.max(np.abs(reference[outside]))):.2e}"
    )


def last_bit() -> None:
    """How far from 1 the unit box's winding number is, a thousandth of a width from a wall."""
    print(
        "\n### last bit: unit box, three interior points, |winding| - 1 in units in the last place\n"
    )
    point = np.array([[0.999, 0.5, 0.5], [0.5, 0.5, 0.5], [0.001, 0.3, 0.7]])
    ulp = np.spacing(1.0)

    def off(winding):
        return ", ".join(f"{(abs(w) - 1.0) / ulp:+.0f}" for w in winding)

    print(f"{'divisions':>9} {'facets':>7} {'eager ulp':>20} {'compiled ulp':>20}")
    for divisions in (1, 2, 4):
        vertices = inward_box(divisions)
        eager = eager_winding(vertices, point, DEFAULT_PAIR_LIMIT)
        compiled = compiled_winding(vertices, point, DEFAULT_PAIR_LIMIT)
        print(f"{divisions:9d} {len(vertices):7d} {off(eager):>20} {off(compiled):>20}")


if __name__ == "__main__":
    last_bit()
    cost()
