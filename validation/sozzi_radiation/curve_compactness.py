"""What the groups cut from the Morton curve look like, with per-axis cells and with cubic cells.

Every grouping of points in the radiation package -- shaft culling's receiver groups and lamp-facet
clusters, the gather's chunks and lit blocks, the transfer build's row blocks, ``FacetClusters`` --
is a run of consecutive points along :func:`aquaflux.morton.morton_order`, which scales each axis of
the bounding box to its own extent. #574 proposed cubic cells instead; ``cubic_order`` below is that
alternative, kept here so the comparison can be re-run. It was measured on the whole Sozzi field and
rejected (it left more pairs undecided and slowed the transfer build), and this harness shows why.

Reported per point set and group size, under each ordering:

- the median and largest group **radius** (half the diagonal of the group's bounding box) -- what a
  bounding-volume certificate wants small, and where cubic cells win;
- for the lamp, the median and 90th percentile of each group's **normal spread** (the largest angle
  between a member facet's normal and the group's mean normal) -- what a "wholly behind these
  points" test wants small, and where per-axis cells win by a wide margin.

Deterministic -- geometry only, no timing -- so a figure from here reproduces exactly anywhere.
Point sets: the lamp's facets (the case STL when present, else the analytic 32 x 128 lamp) and the
receivers of ``primitive_occlusion.py`` (the meshed case's cell centres when present, else points
sampled inside the three cylinders); the summary says which.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import numpy as np  # noqa: E402
from aquaflux.morton import _BITS, _spread, morton_order  # noqa: E402
from primitive_occlusion import OUT, fluid, lamp, receivers  # noqa: E402

SIZES = (2, 8, 32, 64)


def cubic_order(points: np.ndarray) -> np.ndarray:
    """The rejected alternative: cubic cells on the bounding box's longest side."""
    low = points.min(axis=0)
    side = float((points.max(axis=0) - low).max())
    levels = (1 << _BITS) - 1
    scaled = (points - low) / side if side > 0.0 else np.zeros_like(points)
    cell = np.minimum((scaled * levels).astype(np.int64), levels)
    key = _spread(cell[:, 0]) | (_spread(cell[:, 1]) << 1) | (_spread(cell[:, 2]) << 2)
    return np.argsort(key, kind="stable")


def _groups(values: np.ndarray, order: np.ndarray, size: int) -> np.ndarray:
    """``values`` of the whole groups of ``size`` cut from ``order``, shape ``(n_groups, size, 3)``."""
    whole = len(order) // size * size
    return values[order[:whole]].reshape(-1, size, 3)


def radii(points: np.ndarray, order: np.ndarray, size: int) -> tuple[float, float]:
    """Median and largest group radius."""
    groups = _groups(points, order, size)
    radius = 0.5 * np.linalg.norm(groups.max(axis=1) - groups.min(axis=1), axis=1)
    return float(np.median(radius)), float(radius.max())


def normal_spread(normals: np.ndarray, order: np.ndarray, size: int) -> tuple[float, float]:
    """Median and 90th percentile, over groups, of the largest angle from the group's mean normal."""
    groups = _groups(normals, order, size)
    mean = groups.sum(axis=1)
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    cosine = np.clip(np.einsum("gkd,gd->gk", groups, mean), -1.0, 1.0)
    spread = np.degrees(np.arccos(cosine)).max(axis=1)
    return float(np.median(spread)), float(np.quantile(spread, 0.9))


def main() -> None:
    surfaces, lamp_label = lamp()
    points, receiver_label = receivers(np.random.default_rng(0), fluid())
    normals = np.array(surfaces.normal, dtype=float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    sets = {"lamp facets": np.asarray(surfaces.centroid, dtype=float), "receivers": points}
    orderings = {"per-axis": morton_order, "cubic": cubic_order}
    rows = []
    for name, cloud in sets.items():
        extent = cloud.max(axis=0) - cloud.min(axis=0)
        print(f"{name}: {len(cloud)} points, bounding box {np.round(extent, 4).tolist()} m")
        for size in SIZES:
            row = {"set": name, "size": size}
            for label, ordering in orderings.items():
                order = ordering(cloud)
                row[label] = {"radius": radii(cloud, order, size)}
                if name == "lamp facets":
                    row[label]["normal_spread_deg"] = normal_spread(normals, order, size)
            rows.append(row)
            line = f"  groups of {size:>2}:"
            for label in orderings:
                median, largest = row[label]["radius"]
                line += f"  {label} radius {median:.4f} / {largest:.4f} m"
                if "normal_spread_deg" in row[label]:
                    median, p90 = row[label]["normal_spread_deg"]
                    line += f", normals {median:.1f} / {p90:.1f} deg"
            print(line)
    print("(radius: median / max; normals: median / 90th percentile of each group's spread)")
    summary = {"lamp": lamp_label, "receivers": receiver_label, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "curve_compactness.json").write_text(json.dumps(summary, indent=2))
    print(f"lamp: {lamp_label}; receivers: {receiver_label}")


if __name__ == "__main__":
    main()
