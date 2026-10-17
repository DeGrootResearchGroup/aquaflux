"""How a torus decides a line that grazes its tube, against exact arithmetic (#504).

A line nearly tangent to a body's surface is where a closed-form intersection test throws its
digits away, and for a ray-traced shadow the wrong answer is the dangerous one: a line that does
enter the body reported as a miss lets light through. The cylinder's version of this is recorded
with the cylinder; this is the torus's, whose surface is a quartic.

Three readings of the same lines:

- **exact** -- the quartic's coefficients from the stored floats, as rationals, and the count of
  its real roots by a Sturm sequence (``tests/unit/torus_reference.py``): the true answer for the
  inputs as given, with no rounding anywhere;
- **the torus** -- :class:`aquaflux.solids.Torus`, which decides from the line's distance to the
  tube's centre circle at the quartic's turning points;
- **a general root finder** -- ``numpy.roots`` on the quartic's (rounded) coefficients, a real root
  being one whose imaginary part is within ``1e-12`` of its size: the obvious way to write it.

Two families of lines: the fixed grazing set the unit test asserts on (seven tangencies, offsets of
1e-9 to 1e-12 of the tube radius either side, sources one to a hundred ring diameters away), and a
random one -- touch points and tangent directions drawn uniformly, offsets from 1e-12 to 1e-2. A
misclassification is counted, and split by direction: a *leak* is a line exact arithmetic says
enters the tube, reported as a miss.

Run with ``validation/run_case.sh validation/torus_tangency.py``. ``TORUS_LINES`` sets the size of
the random family.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.solids import Torus  # noqa: E402

from tests.unit.test_solids import _grazing_lines  # noqa: E402
from tests.unit.torus_reference import has_real_root, quartic  # noqa: E402

LINES = int(os.environ.get("TORUS_LINES", 3000))
OUT = HERE / "torus_tangency"


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def random_grazing(rng, ring, tube, count):
    """Lines touching the tube at random points, along random tangent directions."""
    around, turn = rng.uniform(0, 2 * np.pi, count), rng.uniform(0, 2 * np.pi, count)
    reach = ring + tube * np.cos(turn)
    point = np.stack([reach * np.cos(around), reach * np.sin(around), tube * np.sin(turn)], 1)
    outward = np.stack(
        [np.cos(turn) * np.cos(around), np.cos(turn) * np.sin(around), np.sin(turn)], 1
    )
    first = np.stack([-np.sin(around), np.cos(around), 0 * around], 1)
    second = np.cross(outward, first)
    angle = rng.uniform(0, 2 * np.pi, count)
    along = np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second
    offset = 10.0 ** rng.uniform(-12, -2, count) * tube * rng.choice([-1.0, 1.0], count)
    distance = 10.0 ** rng.uniform(0, 2, count) * 2 * ring
    origin = point - offset[:, None] * outward - distance[:, None] * along
    return list(zip(origin, 2 * distance[:, None] * along, strict=True))


def by_numpy_roots(origin, direction, ring, tube):
    coefficients = [float(c) for c in quartic(origin, direction, [0, 0, 0], [0, 0, 1], ring, tube)]
    roots = np.roots(coefficients[::-1])
    return bool(np.any(np.abs(roots.imag) <= 1e-12 * np.maximum(1.0, np.abs(roots.real))))


def score(name, lines, ring, tube):
    body = Torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], ring, tube)
    exact = np.array(
        [has_real_root(quartic(o, d, [0, 0, 0], [0, 0, 1], ring, tube)) for o, d in lines]
    )
    origin = jnp.asarray([o for o, _ in lines])
    direction = jnp.asarray([d for _, d in lines])
    enter, exit_ = map(np.asarray, body.intervals(origin, direction))
    torus = (exit_ >= enter).any(axis=-1)
    general = np.array([by_numpy_roots(o, d, ring, tube) for o, d in lines])
    result = {"lines": len(lines), "meeting the tube": int(exact.sum())}
    for label, told in (("torus", torus), ("numpy.roots", general)):
        wrong = told != exact
        result[label] = {"misclassified": int(wrong.sum()), "leaks": int((wrong & exact).sum())}
    _say(f"{name}: {json.dumps(result)}")
    return result


def main() -> None:
    ring, tube, fixed, _ = _grazing_lines()
    summary = {
        "fixed grazing set": score("fixed grazing set", fixed, ring, tube),
        "random grazing set": score(
            "random grazing set",
            random_grazing(np.random.default_rng(5), 1.0, 0.25, LINES),
            1.0,
            0.25,
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
    sys.exit(0)
