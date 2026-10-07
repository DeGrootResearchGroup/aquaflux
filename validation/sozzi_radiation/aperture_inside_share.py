"""How often a source image is seen wholly through a mirror: the share an "inside" shortcut skips.

After the cone cull, a mirrored gather is limited by the clip itself: each image is clipped against
the few aperture triangles it overlaps (``aperture_cull_share.py``). An image whose projection onto
the mirror lies **entirely inside the mirror's outline** needs no clip at all -- the mirror shows all
of it, so its share is its whole solid angle, a closed form. This measures how many images that is,
on ``specular_cost.py``'s scene (the Sozzi reactor's flat end plates), to bound what merging a plane's
triangles into its outline and testing "inside" first could save.

Per (receiver, image) pair, with the receiver in front of the mirror and the image's source in front
too, the image's corners are projected onto the mirror's plane along the lines from the receiver.
The projected triangle is then, against the union of the mirror's triangles:

- **inside**: every corner inside the union, no edge crossing its boundary, and no boundary vertex
  strictly inside the triangle -- which together are exact for a polygon with holes (the last catches
  a hole lying wholly within the triangle);
- **outside**: no corner inside, no crossing, no boundary vertex inside -- the mirror shows none of it;
- **partial**: anything else, which still needs the clip.

Configuration as ``specular_cost.py`` reads it (``SOZZI_SPECULAR_*``), plus ``SOZZI_INSIDE_RECEIVERS``
(default 300) volume points drawn uniformly from the drawing's fluid.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

RESULT = HERE / "work" / "aperture_inside_share.json"
RECEIVERS = int(os.environ.get("SOZZI_INSIDE_RECEIVERS", 300))


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def plane_frame(normal):
    """Two unit vectors spanning the plane with this normal."""
    import numpy as np

    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, helper)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def boundary_edges(triangles2d, decimals: int = 9):
    """The edges of a triangle set that belong to one triangle only: its outline and holes."""
    import numpy as np

    edges = np.concatenate([triangles2d[:, [k, (k + 1) % 3]] for k in range(3)])
    key = np.round(edges, decimals)
    ordered = np.where(
        (key[:, 0, 0] < key[:, 1, 0])
        | ((key[:, 0, 0] == key[:, 1, 0]) & (key[:, 0, 1] <= key[:, 1, 1])),
        0,
        1,
    )
    canonical = np.where(ordered[:, None, None] == 0, key, key[:, ::-1])
    _, inverse, counts = np.unique(
        canonical.reshape(len(edges), 4), axis=0, return_inverse=True, return_counts=True
    )
    return edges[counts[inverse.ravel()] == 1]


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def inside_triangles(points, triangles2d):
    """Whether each point lies in any of the triangles (closed), ``points (..., 2)``."""
    import numpy as np

    inside = np.zeros(points.shape[:-1], dtype=bool)
    for a, b, c in triangles2d:
        d1 = _cross(b - a, points - a)
        d2 = _cross(c - b, points - b)
        d3 = _cross(a - c, points - c)
        negative = (d1 < 0) | (d2 < 0) | (d3 < 0)
        positive = (d1 > 0) | (d2 > 0) | (d3 > 0)
        inside |= ~(negative & positive)
    return inside


def segments_cross(p, q, edges):
    """Whether segment ``p -> q`` (``(n, 2)`` each) properly crosses any of ``edges (m, 2, 2)``."""
    import numpy as np

    a, b = edges[None, :, 0], edges[None, :, 1]
    p, q = p[:, None], q[:, None]
    d1, d2 = _cross(b - a, p - a), _cross(b - a, q - a)
    d3, d4 = _cross(q - p, a - p), _cross(q - p, b - p)
    return np.any((d1 * d2 < 0) & (d3 * d4 < 0), axis=1)


def classify(projected, triangles2d, edges):
    """``inside``, ``outside`` or ``partial`` per projected triangle ``(n, 3, 2)``, as int 0/1/2."""
    import numpy as np

    corners_in = inside_triangles(projected, triangles2d)
    crossing = np.zeros(len(projected), dtype=bool)
    for k in range(3):
        crossing |= segments_cross(projected[:, k], projected[:, (k + 1) % 3], edges)
    vertices = np.unique(edges.reshape(-1, 2), axis=0)
    a, b, c = (projected[:, k][:, None] for k in range(3))
    v = vertices[None]
    d1, d2, d3 = _cross(b - a, v - a), _cross(c - b, v - b), _cross(a - c, v - c)
    strictly = ((d1 > 0) & (d2 > 0) & (d3 > 0)) | ((d1 < 0) & (d2 < 0) & (d3 < 0))
    holds_vertex = np.any(strictly, axis=1)
    inside = corners_in.all(axis=1) & ~crossing & ~holds_vertex
    outside = ~corners_in.any(axis=1) & ~crossing & ~holds_vertex
    return np.where(inside, 0, np.where(outside, 1, 2))


def run() -> dict:
    import aquaflux  # noqa: F401  (enables x64)
    import numpy as np
    from aquaflux.radiation import planar_mirrors
    from specular_cost import CHORD, LAMP, PLATES, WALL, receivers, scene

    rows = []
    for plate_size in PLATES:
        surfaces, water, counts = scene(plate_size)
        points = receivers(water, RECEIVERS)
        areal = np.flatnonzero(~surfaces.is_point_source)
        for mirror in planar_mirrors(surfaces, ["end plates"]):
            point, normal = np.asarray(mirror.point), np.asarray(mirror.normal)
            first, second = plane_frame(normal)
            aperture = np.asarray(surfaces.vertices)[np.asarray(mirror.facets)] - point
            triangles2d = np.stack([aperture @ first, aperture @ second], axis=-1)
            edges = boundary_edges(triangles2d)
            sources = np.intersect1d(mirror.sources_in_front(surfaces), areal)
            image = np.asarray(mirror.image(surfaces).vertices)[sources] - point
            image_centroid = image.mean(axis=1)
            image_normal = np.asarray(mirror.image(surfaces).normal)[sources]
            front = points[mirror.in_front(points)] - point
            started = time.perf_counter()
            tally = np.zeros(3, dtype=np.int64)
            lit_pairs = 0
            for receiver in front:
                # Only pairs the gather keeps: the receiver in front of the image's own plane.
                lit = np.einsum("sk,sk->s", receiver - image_centroid, image_normal) > 0.0
                corners = image[lit]
                if not len(corners):
                    continue
                height_r = receiver @ normal
                height_c = corners @ normal
                share = height_r / (height_r - height_c)
                hit = receiver + share[..., None] * (corners - receiver)
                projected = np.stack([hit @ first, hit @ second], axis=-1)
                tally += np.bincount(classify(projected, triangles2d, edges), minlength=3)
                lit_pairs += len(corners)
            seen = tally[0] + tally[2]
            row = {
                "plate_facet_size_m": plate_size,
                "facets": counts,
                "mirror_point": point.round(4).tolist(),
                "aperture_triangles": len(triangles2d),
                "boundary_edges": len(edges),
                "lit_pairs": int(lit_pairs),
                "inside": int(tally[0]),
                "outside": int(tally[1]),
                "partial": int(tally[2]),
                "inside_share_of_seen": float(tally[0] / seen) if seen else 0.0,
                "seconds": round(time.perf_counter() - started, 1),
            }
            _say(json.dumps(row))
            rows.append(row)
    return {
        "configuration": {
            "chord_m": CHORD,
            "lamp_facet_size_m": LAMP,
            "wall_facet_size_m": WALL,
            "plate_facet_sizes_m": PLATES,
            "receivers": f"{RECEIVERS} drawn uniformly from the drawing's fluid",
            "pairs": "receiver in front of the mirror and of the image's plane",
        },
        "rows": rows,
    }


def main() -> None:
    result = run()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    _say(f"written {RESULT}")


if __name__ == "__main__":
    main()
