"""How many of a mirror's triangles each image needs clipping against: the cone cull's survival share.

The mirrored gather and the specular exchange clip every source image against every triangle of the
mirror's aperture, so their cost is the direct gather's times the aperture's triangle count. A cone
cull -- skip an aperture triangle whose angular cone, seen from the receiver, cannot overlap the
image's -- would clip only the survivors. This measures how many survive, before anything is built,
with the cone test the silhouette clip already uses (``silhouette.angular_cone`` and
``cones_may_overlap``, unusable cones overlapping everything).

The scene is ``specular_cost.py``'s: the Sozzi reactor from its CAD drawing, its two flat end plates
as mirrors, at a ladder of plate facet sizes. Two kinds of receiver, because the gather and the
exchange see different images:

- **volume**: points drawn uniformly from the drawing's fluid (no meshed case is needed for a share,
  but a snapped mesh's cells crowd the walls and the lamp, so read the figure as a uniform-sample one);
- **facets**: the centroids of the facets in front of the mirror, standing in for the exchange's
  receiving quadrature points.

Per (receiver, image) pair in front of the mirror, it counts the aperture triangles kept. Reported:
the mean kept against the aperture size (the cost fraction a cull would leave), percentiles, the share
of pairs with none kept (the image is not seen through the mirror at all), and the share of
(receiver, triangle) cones flagged unusable.

Configuration by environment, as ``specular_cost.py`` reads it (``SOZZI_SPECULAR_*``), plus
``SOZZI_CULL_RECEIVERS`` volume points (default 2000).
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

RESULT = HERE / "work" / "aperture_cull_share.json"
VOLUME = int(os.environ.get("SOZZI_CULL_RECEIVERS", 2000))


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def kept_counts(receivers, image_vertices, aperture, chunk: int = 64):
    """Aperture triangles kept by the cone cull, per (receiver, image), and unusable-cone share.

    Parameters
    ----------
    receivers : numpy.ndarray, shape ``(r, 3)``
    image_vertices : numpy.ndarray, shape ``(s, 3, 3)``
    aperture : numpy.ndarray, shape ``(a, 3, 3)``

    Returns
    -------
    tuple of (numpy.ndarray of int, shape ``(r, s)``, float)
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.radiation.silhouette import angular_cone, cones_may_overlap

    @jax.jit
    def block(points):
        image = angular_cone(image_vertices[None] - points[:, None, None, :])
        mirror = angular_cone(aperture[None] - points[:, None, None, :])
        spread = tuple(part[:, :, None] for part in image)
        across = tuple(part[:, None, :] for part in mirror)
        kept = cones_may_overlap(spread, across)
        return jnp.sum(kept, axis=-1), jnp.mean(mirror[3])

    image_vertices = jnp.asarray(image_vertices)
    aperture = jnp.asarray(aperture)
    counts, unusable = [], []
    for start in range(0, len(receivers), chunk):
        kept, bad = block(jnp.asarray(receivers[start : start + chunk]))
        counts.append(np.asarray(kept))
        unusable.append(float(bad) * min(chunk, len(receivers) - start))
    return np.concatenate(counts), sum(unusable) / len(receivers)


def summary(counts, aperture: int, unusable: float) -> dict:
    import numpy as np

    flat = counts.ravel()
    return {
        "pairs": int(flat.size),
        "aperture_triangles": aperture,
        "mean_kept": float(flat.mean()),
        "cost_fraction": float(flat.mean() / aperture),
        "p50_p90_p99_max_kept": [float(np.percentile(flat, q)) for q in (50, 90, 99)]
        + [int(flat.max())],
        "share_with_none_kept": float(np.mean(flat == 0)),
        "unusable_mirror_cone_share": unusable,
    }


def run() -> dict:
    import aquaflux  # noqa: F401  (enables x64)
    import jax
    import numpy as np
    from aquaflux.radiation import planar_mirrors
    from specular_cost import CHORD, LAMP, PLATES, WALL, receivers, scene

    rows = []
    for plate_size in PLATES:
        surfaces, water, counts = scene(plate_size)
        points = receivers(water, VOLUME)
        areal = np.flatnonzero(~surfaces.is_point_source)
        for mirror in planar_mirrors(surfaces, ["end plates"]):
            aperture = np.asarray(surfaces.vertices)[np.asarray(mirror.facets)]
            sources = np.intersect1d(mirror.sources_in_front(surfaces), areal)
            image = np.asarray(mirror.image(surfaces).vertices)[sources]
            volume = points[mirror.in_front(points)]
            facets = np.asarray(surfaces.centroid)[sources]
            facets = facets[mirror.heights(facets) > 0.0]
            started = time.perf_counter()
            volume_counts, volume_bad = kept_counts(volume, image, aperture)
            facet_counts, facet_bad = kept_counts(facets, image, aperture)
            row = {
                "plate_facet_size_m": plate_size,
                "facets": counts,
                "mirror_point": np.asarray(mirror.point).round(4).tolist(),
                "volume": summary(volume_counts, len(aperture), volume_bad),
                "facets_as_receivers": summary(facet_counts, len(aperture), facet_bad),
                "seconds": round(time.perf_counter() - started, 1),
            }
            _say(json.dumps(row))
            rows.append(row)
        jax.clear_caches()
    return {
        "configuration": {
            "chord_m": CHORD,
            "lamp_facet_size_m": LAMP,
            "wall_facet_size_m": WALL,
            "plate_facet_sizes_m": PLATES,
            "volume_receivers": f"{VOLUME} drawn uniformly from the drawing's fluid",
            "cone_test": "silhouette.angular_cone + cones_may_overlap, unusable overlaps all",
            "jax": jax.__version__,
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
