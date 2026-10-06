"""What specular end plates cost on the Sozzi reactor: the transfer, the masks and the field.

The reactor read from its own CAD drawing (``SozziTaghipour.step``): the lamp, the chamber's side
wall (diffuse), and its two flat end plates named as specular. Each piece of the specular work is
timed against the direct work it sits beside, at a ladder of end-plate facet sizes, because the
reflected gather clips every image against every facet of the mirror's aperture -- so the aperture's
facet count, not only the number of planes, sets its cost.

Per plate facet size, it reports:

- the transfer's specular exchange per plane (``images.plane_exchange``) and the facet-receiver mask
  per plane, against the direct transfer build;
- the receivers' mirror masks against their direct mask, and the mirrored volume gather against the
  direct gather (first call, which compiles, and a second);
- the throughput of each as work items per second -- receiving point x source x aperture facet for
  the clipped exchanges, counted over every aperture facet whether or not the cone cull lets it
  through (so it is an unculled-equivalent rate), receiver x source for the masks;
- the share of (receiver, source) pairs through each mirror whose source image faces away from the
  receiver, which a back-face cull of images could skip.

**Receivers are drawn uniformly from the drawing's fluid**, not from the meshed case's cells: the mesh
needs the OpenFOAM tutorial run (``generate_dom_reference.py``), which a machine without OpenFOAM
cannot make. The summary says so. A uniform sample under-weights the wall and lamp regions a snapped
mesh refines towards, which matters to culling shares and not to per-item throughput.

Configuration by environment, every value recorded in the summary:

- ``SOZZI_SPECULAR_PLATES``: end-plate facet sizes in metres, comma separated (default
  ``0.05,0.03,0.01``: 56, 74 and 434 plate facets at the default chord);
- ``SOZZI_SPECULAR_LAMP``: lamp facet size (default 0.04: 1,294 facets);
- ``SOZZI_SPECULAR_WALL``: side-wall facet size (default 0.05: 1,008 facets);
- ``SOZZI_SPECULAR_CHORD``: the chord every piece is triangulated to (default 1e-3 m). The scene is
  deliberately coarse: what is measured is work per item, which a coarse scene measures as well as a
  fine one, at a cost a four-core machine can pay (the 1e-4 m, 20 mm scene's direct transfer alone
  took 495 s there at 9,916 facets);
- ``SOZZI_SPECULAR_RECEIVERS``: receivers drawn (default 4000).

Run with ``validation/run_case.sh validation/sozzi_radiation/specular_cost.py --wait`` in an
environment with the CAD kernel (``pip install "aquaflux[cad]"``).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

DRAWING = HERE.parent / "uvreactor_openfoam" / "of_case" / "SozziTaghipour.step"
RESULT = HERE / "work" / "specular_cost.json"
VESSEL = ("reactor_body", "inlet_pipe", "outlet_pipe")

PLATES = tuple(
    float(s) for s in os.environ.get("SOZZI_SPECULAR_PLATES", "0.05,0.03,0.01").split(",")
)
LAMP = float(os.environ.get("SOZZI_SPECULAR_LAMP", 0.04))
WALL = float(os.environ.get("SOZZI_SPECULAR_WALL", 0.05))
RECEIVERS = int(os.environ.get("SOZZI_SPECULAR_RECEIVERS", 4000))
CHORD = float(os.environ.get("SOZZI_SPECULAR_CHORD", 1e-3))

#: The case's lamp exitance (W/m²) and the water's absorption coefficient (1/m).
EXITANCE = 696.42
ABSORPTION = 35.67
#: Chamber length and radius, inlet pipe radius, in the case frame (axis along x).
LENGTH, RADIUS, INLET = 0.889, 0.0445, 0.00955
LAMP_RADIUS = 0.01


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _timed(function, *args, **kwargs):
    """``function``'s result, blocked until ready, and the seconds it took."""
    import jax

    started = time.perf_counter()
    result = jax.block_until_ready(function(*args, **kwargs))
    return result, time.perf_counter() - started


def scene(plate_size: float):
    """The surface set, the water, and its pieces' facet counts."""
    import numpy as np
    from aquaflux.io.cad import Placement, read_step
    from aquaflux.radiation import Lambertian, Surfaces

    cad = read_step(DRAWING, Placement(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    lamp = cad.triangles("lamp", chord=CHORD, facet_size=LAMP)
    # The lamp's base disc lies on the end plate it stands on, and radiates nothing into the water.
    lamp = lamp[~np.all(np.abs(lamp[:, :, 0]) < 1e-9, axis=1)]

    def body(size):
        # Wound outward from the solid; the walls face the water inside it.
        triangles = cad.triangles("reactor_body", chord=CHORD, facet_size=size)[:, ::-1]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return triangles, np.abs(normal[:, 0]) > 0.99 * np.linalg.norm(normal, axis=1)

    side, on_plate = body(WALL)
    side = side[~on_plate]
    plates, on_plate = body(plate_size)
    plates = plates[on_plate]
    centroid = plates.mean(axis=1)
    radial = np.hypot(centroid[:, 1], centroid[:, 2])
    # The inlet pipe opens through the far plate and the lamp stands on the near one: neither
    # footprint is a mirror.
    opening = np.where(centroid[:, 0] > LENGTH / 2, radial < INLET, radial < LAMP_RADIUS)
    plates = plates[~opening]

    triangles = np.concatenate([lamp, side, plates])
    kind = np.concatenate([np.zeros(len(lamp)), np.ones(len(side)), np.full(len(plates), 2)])
    surfaces = Surfaces.from_triangles(
        triangles,
        solid_id=kind.astype(int),
        solid_names=("lamp", "wall", "end plates"),
        emission=np.where(kind == 0, EXITANCE, 0.0),
        diffuse_reflectance=np.choose(kind.astype(int), [0.0, 0.3, 0.1]),
        specular_reflectance=np.where(kind == 2, 0.6, 0.0),
        profiles=(Lambertian(),),
    )
    counts = {"lamp": len(lamp), "wall": len(side), "end_plates": len(plates)}
    return surfaces, cad.fluid(*VESSEL), counts


def receivers(water, count: int, seed: int = 0):
    """``count`` points drawn uniformly from the water, away from the lamp."""
    import numpy as np

    rng = np.random.default_rng(seed)
    low, high = np.array([0.0, -RADIUS, -RADIUS]), np.array([1.75, RADIUS, 0.9])
    kept = []
    while sum(len(k) for k in kept) < count:
        points = rng.uniform(low, high, (20 * count, 3))
        inside = ~np.asarray(water.contains(points))
        clear_of_lamp = (np.hypot(points[:, 1], points[:, 2]) > LAMP_RADIUS * 1.05) | (
            points[:, 0] > 0.82
        )
        kept.append(points[inside & clear_of_lamp])
    return np.concatenate(kept)[:count]


def facing_away_share(mirror, surfaces, points) -> float:
    """Of the (receiver in front, source in front) pairs through ``mirror``, the share whose source's
    image faces away from the receiver -- what a back-face cull of images could skip."""
    import numpy as np

    image = mirror.image(surfaces)
    rows = mirror.in_front(points)
    cols = np.intersect1d(
        mirror.sources_in_front(surfaces), np.flatnonzero(~surfaces.is_point_source)
    )
    if not (len(rows) and len(cols)):
        return 0.0
    centroid = np.asarray(image.centroid)[cols]
    normal = np.asarray(image.normal)[cols]
    height = np.einsum("rsk,sk->rs", points[rows][:, None, :] - centroid[None], normal)
    return float(np.mean(height <= 0.0))


def run() -> dict:
    import aquaflux  # noqa: F401  (enables x64)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.radiation import (
        RayCastOcclusion,
        UniformAbsorption,
        build_mirror_visibility,
        build_transfer,
        build_visibility,
        planar_mirrors,
        plane_exchange,
    )
    from aquaflux.radiation.gather import summed_fluence_rate
    from aquaflux.radiation.images import summed_mirrored_fluence_rate
    from aquaflux.radiation.quadrature import triangle_quadrature

    strategy = RayCastOcclusion(grid=True)
    absorption = UniformAbsorption(ABSORPTION)
    rule = triangle_quadrature(6)
    rows = []
    for plate_size in PLATES:
        surfaces, water, counts = scene(plate_size)
        points = receivers(water, RECEIVERS)
        n = surfaces.n_facets
        _say(f"plates at {plate_size} m: {counts}, {n} facets, {len(points)} receivers")
        prepared = strategy.prepared(surfaces)

        _, direct_transfer = _timed(
            build_transfer, surfaces, occluders=[water], self_occlusion=prepared
        )
        _say(f"  direct transfer build {direct_transfer:.1f} s")

        sample = rule.points(jnp.asarray(surfaces.vertices))
        weight = jnp.asarray(rule.weight)
        planes = []
        for mirror in planar_mirrors(surfaces, ["end plates"]):
            aperture = len(mirror.facets)
            front = mirror.sources_in_front(surfaces)
            areal_front = int(np.sum(~surfaces.is_point_source[front]))
            exchange_items = areal_front * areal_front * aperture * len(rule.weight)
            # Timed once: at minutes per plane its compilation is noise, and a second call would
            # double the run.
            _, exchange = _timed(plane_exchange, mirror, surfaces, sample, weight)
            _, facet_mask = _timed(
                build_mirror_visibility,
                mirror,
                [water],
                surfaces,
                surfaces.centroid,
                receiver_facet=np.arange(n),
                self_occlusion=prepared,
            )
            in_front = mirror.in_front(points)
            _, receiver_mask = _timed(
                build_mirror_visibility, mirror, [water], surfaces, points, self_occlusion=prepared
            )
            planes.append(
                {
                    "aperture_facets": aperture,
                    "sources_in_front": len(front),
                    "receivers_in_front": len(in_front),
                    "exchange_seconds": round(exchange, 2),
                    "exchange_triples_per_second_unculled_equivalent": exchange_items / exchange,
                    "facet_mask_seconds": round(facet_mask, 2),
                    "facet_mask_paths_per_second": n * len(front) / facet_mask,
                    "receiver_mask_seconds": round(receiver_mask, 2),
                    "receiver_mask_paths_per_second": len(in_front) * len(front) / receiver_mask,
                    "facing_away_share": facing_away_share(mirror, surfaces, points),
                }
            )
            _say(f"  plane: {planes[-1]}")

        mirrors = planar_mirrors(surfaces, ["end plates"])
        _, direct_mask = _timed(
            build_visibility, [water], surfaces, points, self_occlusion=prepared
        )
        visibility = build_visibility([water], surfaces, points, self_occlusion=prepared)
        masks = [
            build_mirror_visibility(m, [water], surfaces, points, self_occlusion=prepared)
            for m in mirrors
        ]
        sets = (surfaces,)
        _, direct_first = _timed(
            summed_fluence_rate, sets, points, absorption=absorption, visibility=visibility
        )
        _, direct_second = _timed(
            summed_fluence_rate, sets, points, absorption=absorption, visibility=visibility
        )
        mirrored = dict(absorption=absorption, shadows=masks)
        _, mirrored_first = _timed(summed_mirrored_fluence_rate, sets, mirrors, points, **mirrored)
        _, mirrored_second = _timed(summed_mirrored_fluence_rate, sets, mirrors, points, **mirrored)
        gather_items = sum(
            p["receivers_in_front"] * p["sources_in_front"] * p["aperture_facets"] for p in planes
        )
        row = {
            "plate_facet_size_m": plate_size,
            "facets": counts,
            "direct_transfer_seconds": round(direct_transfer, 1),
            "planes": planes,
            "direct_receiver_mask_seconds": round(direct_mask, 2),
            "direct_gather_seconds": [round(direct_first, 2), round(direct_second, 2)],
            "mirrored_gather_seconds": [round(mirrored_first, 2), round(mirrored_second, 2)],
            "mirrored_gather_triples_per_second_unculled_equivalent": gather_items
            / mirrored_second,
        }
        _say(
            f"  gathers: direct {row['direct_gather_seconds']} s, mirrored "
            f"{row['mirrored_gather_seconds']} s; direct mask {direct_mask:.2f} s"
        )
        rows.append(row)
        jax.clear_caches()

    return {
        "configuration": {
            "drawing": DRAWING.name,
            "chord_m": CHORD,
            "lamp_facet_size_m": LAMP,
            "wall_facet_size_m": WALL,
            "plate_facet_sizes_m": PLATES,
            "receivers": f"{RECEIVERS} drawn uniformly from the drawing's fluid (no meshed case)",
            "self_occlusion": "RayCastOcclusion(grid=True)",
            "occluders": "the drawing's fluid (Outside of three cylinders)",
            "receiver_quadrature_points": 6,
            "absorption_per_m": ABSORPTION,
            "jax": jax.__version__,
            "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores",
        },
        "rows": rows,
    }


def main() -> None:
    summary = run()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(summary, indent=2))
    _say(f"written {RESULT}")


if __name__ == "__main__":
    main()
