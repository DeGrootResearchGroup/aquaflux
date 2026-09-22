"""Perturb the nodes of an OpenFOAM polyMesh of the duct, keeping the geometry exactly the same.

Random displacements, uniform in ``[-AMOUNT, +AMOUNT]`` times the cell size, are applied to every node
with the components that would move it off the duct's boundary removed: an interior node moves in three
dimensions, a node on a wall/inlet/outlet face slides within that face, a node on an edge slides along it
and a corner node stays. The boundary is therefore unchanged and only the cell shapes (non-orthogonality,
skew, size variation) change -- a controlled way to add skew to the orthogonal hexahedral duct.

Usage::

    python3 perturb_mesh.py SRC_POLYMESH DST_POLYMESH AMOUNT [SEED]

``AMOUNT`` is a fraction of the cell size (0.5 would let neighbouring nodes meet); the topology files are
copied unchanged and only ``points`` is rewritten.
"""

import re
import shutil
import sys
from pathlib import Path

import numpy as np

LX, LY, LZ = 0.25, 0.025, 0.025
BOX = np.array([LX, LY, LZ])
TOL = 1e-9


def read_points(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "(")
    count = int(lines[start - 1])
    body = lines[start + 1 : start + 1 + count]
    points = np.array([[float(v) for v in re.findall(r"[-+0-9.eE]+", row)] for row in body])
    return lines[: start + 1], points, lines[start + 1 + count :]


def main(src: Path, dst: Path, amount: float, seed: int) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    head, points, tail = read_points(src / "points")
    cell = BOX / np.array([60.0, 6.0, 6.0])
    on_low = np.abs(points) < TOL
    on_high = np.abs(points - BOX) < TOL
    fixed = on_low | on_high  # components that must not move
    rng = np.random.default_rng(seed)
    step = rng.uniform(-1.0, 1.0, points.shape) * amount * cell
    step[fixed] = 0.0
    moved = points + step
    body = [f"({x:.10g} {y:.10g} {z:.10g})" for x, y, z in moved]
    (dst / "points").write_text("\n".join(head + body + tail) + "\n")
    print(f"  wrote {dst / 'points'}: {len(points)} nodes, amount {amount} of a cell, seed {seed}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        float(sys.argv[3]),
        int(sys.argv[4]) if len(sys.argv) > 4 else 0,
    )
