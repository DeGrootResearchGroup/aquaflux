"""The fluence rate in the tubular reactor of ``geometry.py``, on a Cartesian background grid.

No flow mesh is needed for the light: ``G`` is gathered at the nodes of a uniform grid of spacing
``h`` that lie in the water, and written as VTK image data any viewer opens (and from which a
dose calculation can interpolate onto a flow mesh later).

What is modelled
    Ten lamps, each emitting 35.4 W diffusely from its sleeve's outer surface (the quartz and the
    air inside it are not modelled); the lamps absorb whatever lands on them. Every wall the water
    touches -- the vessel, both pipes, the end walls and the perforated plate with its holes --
    reflects 10 % of what lands on it, diffusely. Water absorbs 35.67 /m (70 % UVT over 1 cm, as
    the Sozzi & Taghipour reactor). Light leaving through the open ends of the pipes is lost.

How
    Every shadow is decided by exact bodies: the water as the union of convex pieces (the chamber
    on either side of the plate, the two pipes, the hundred holes), and each lamp as a cylinder with
    a hemispherical tip. Triangles are used only where something emits:

    1. the lamps' light on the walls, gathered at four points of each wall triangle from a lamp
       surface coarse along the axis (the walls are at least 30 mm from any lamp);
    2. one radiosity solve among the wall triangles for the light they reflect back and forth;
    3. ``G`` at the grid nodes: the lamps' light from a lamp surface fine along the axis, plus the
       walls' reflected light.

Settings, as environment variables (``validation/run_case.sh`` records them):

``TUBULAR_H``        grid spacing in metres (default 0.006)
``TUBULAR_WALL``     largest wall triangle edge in metres (default 0.03)
``TUBULAR_AXIAL``    largest lamp triangle extent along the axis for the volume gather (0.008)
``TUBULAR_SECTORS``  lamp triangle columns round the sleeve (24)
``TUBULAR_PER_TURN`` wall triangle edges per turn of a curve, so sides of each plate hole (8)
``TUBULAR_SHRINK``   how far inside the sleeve each lamp's shadow body lies, metres (0.000251)
``TUBULAR_OUT``      output directory (default ``work/runs/<tag>``, the tag naming every setting)
``TUBULAR_FIGURES_ONLY``  set to redraw an existing run's figures without recomputing its field

Each run writes its fields, its summary, and every figure: ``slices.png``, ``lamp_tessellation.png``,
``wall_tessellation.png`` and the plate's shadow planes (``plate_shadow.png``, ``plate_shadow.npz``).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.radiation import (  # noqa: E402
    NoOcclusion,
    RadiationSettings,
    Surfaces,
    UniformAbsorption,
    absorption_from_uvt,
    build_radiation_model,
    build_visibility,
    direct_fluence_rate,
    direct_irradiance,
    lamp_exitance,
    subtriangle_centroids,
    surface_irradiance,
)
from geometry import Reactor, wall_triangles  # noqa: E402

H = float(os.environ.get("TUBULAR_H", "0.006"))
WALL_SIZE = float(os.environ.get("TUBULAR_WALL", "0.03"))
AXIAL = float(os.environ.get("TUBULAR_AXIAL", "0.008"))
SECTORS = int(os.environ.get("TUBULAR_SECTORS", "24"))
PER_TURN = int(os.environ.get("TUBULAR_PER_TURN", "8"))
#: How far inside the sleeve each lamp's shadowing body lies. Fixed rather than derived from the
#: lamp's triangles, so that a study of the lamp tessellation does not move the shadows too; a
#: tessellation whose centroids lie deeper than this is refused.
SHRINK = float(os.environ.get("TUBULAR_SHRINK", "0.000251"))
TAG = f"h{H * 1e3:g}_ax{AXIAL * 1e3:g}_s{SECTORS}_w{WALL_SIZE * 1e3:g}_t{PER_TURN}"
OUT = Path(os.environ.get("TUBULAR_OUT", HERE / "work" / "runs" / TAG))

UVT = 70.0
REFLECTANCE = 0.1
#: The lamps lighting the walls, which are at least 30 mm away: coarse along the axis.
WALL_LIGHTING_AXIAL = 0.04
#: Sub-triangle points per edge at which the lamps' light on a wall triangle is averaged.
WALL_SAMPLES = 2
#: Grid nodes closer than this to a wall or a lamp are not gathered at.
NODE_MARGIN = 3e-4
#: How far outside the water a wall triangle's centroid may round before it reads as in the wall.
WATER_TOLERANCE = 1e-6
#: Points per pass when the lamps' light on the walls is gathered through a held mask.
WALL_PASS = 4096


def say(line: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)


def lamps(reactor: Reactor, axial: float) -> Surfaces:
    """The ten lamps' surfaces, each radiating its rating whatever its triangulation."""
    triangles, which = reactor.lamp_triangles(SECTORS, axial)
    surfaces = Surfaces.from_triangles(
        jnp.asarray(triangles), solid_id=which, solid_names=reactor.lamp_names
    )
    rating = dict.fromkeys(reactor.lamp_names, reactor.lamp_power)
    return surfaces.with_optics(emission=lamp_exitance(surfaces, rating))


def shadow_shrink(reactor: Reactor, *surface_sets: Surfaces) -> float:
    """``SHRINK``, refused if a lamp facet's centroid lies within a twentieth of a millimetre of
    the shadowing body or inside it."""
    deepest = max(-reactor.lamp_distance(np.asarray(s.centroid)).min() for s in surface_sets)
    if deepest + 5e-5 > SHRINK:
        raise ValueError(
            f"lamp centroids lie {deepest * 1e3:.3f} mm inside the sleeve; the shadow body at "
            f"{SHRINK * 1e3:.3f} mm would hold them"
        )
    return SHRINK


def grid_nodes(reactor: Reactor, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid's origin and shape, and which of its nodes are in the water."""
    low, high = reactor.bounds
    shape = np.floor((high - low) / h).astype(int) + 1
    axes = [low[a] + h * np.arange(shape[a]) for a in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return low, shape, reactor.in_water(points, NODE_MARGIN)


def lamp_light_on_walls(walls, lamp_set, occluders, absorption) -> np.ndarray:
    """The lamps' irradiance on each wall triangle, averaged over its sub-triangle points."""
    samples = subtriangle_centroids(np.asarray(walls.vertices), WALL_SAMPLES)
    per_facet = samples.shape[1]
    points = samples.reshape(-1, 3)
    normals = np.repeat(np.asarray(walls.normal), per_facet, axis=0)
    arriving = np.zeros(len(points))
    for start in range(0, len(points), WALL_PASS):
        part = slice(start, start + WALL_PASS)
        mask = build_visibility(
            occluders,
            lamp_set,
            jnp.asarray(points[part]),
            receiver_normal=jnp.asarray(normals[part]),
            self_occlusion=NoOcclusion(),
        )
        arriving[part] = np.asarray(
            direct_irradiance(
                lamp_set,
                jnp.asarray(points[part]),
                jnp.asarray(normals[part]),
                absorption=absorption,
                visibility=mask,
            )
        )
    return arriving.reshape(-1, per_facet).mean(axis=1)


def _array_xml(name: str, values: np.ndarray, components: int = 1) -> str:
    raw = np.ascontiguousarray(values, dtype="<f8").tobytes()
    blob = base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode()
    return (
        f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{components}" '
        f'format="binary">{blob}</DataArray>'
    )


def write_image(path: Path, origin, h, shape, fields: dict[str, np.ndarray]) -> None:
    """VTK image data with point fields; ``fields`` are flat in (x, y, z) C order, NaN outside the
    water."""
    extent = " ".join(f"0 {n - 1}" for n in shape)
    # VTK orders points x fastest, so the C-ordered (x, y, z) arrays are transposed.
    arrays = "".join(
        _array_xml(name, values.reshape(shape).transpose(2, 1, 0).ravel())
        for name, values in fields.items()
    )
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        f'header_type="UInt64"><ImageData WholeExtent="{extent}" '
        f'Origin="{" ".join(map(str, origin))}" Spacing="{h} {h} {h}">'
        f'<Piece Extent="{extent}"><PointData Scalars="G">{arrays}</PointData></Piece>'
        "</ImageData></VTKFile>\n"
    )


def write_surface(path: Path, triangles: np.ndarray, fields: dict[str, np.ndarray]) -> None:
    """Triangles as VTK polygonal data with cell fields."""
    n = len(triangles)
    points = triangles.reshape(-1, 3)
    connectivity = np.arange(3 * n, dtype="<i8")
    offsets = 3 * np.arange(1, n + 1, dtype="<i8")

    def ints(name, values):
        raw = values.tobytes()
        blob = base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode()
        return f'<DataArray type="Int64" Name="{name}" format="binary">{blob}</DataArray>'

    cells = "".join(_array_xml(name, values) for name, values in fields.items())
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
        f'header_type="UInt64"><PolyData><Piece NumberOfPoints="{3 * n}" NumberOfPolys="{n}">'
        f"<Points>{_array_xml('Points', points, 3)}</Points>"
        f"<Polys>{ints('connectivity', connectivity)}{ints('offsets', offsets)}</Polys>"
        f"<CellData>{cells}</CellData></Piece></PolyData></VTKFile>\n"
    )


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    reactor = Reactor()
    if os.environ.get("TUBULAR_FIGURES_ONLY"):
        fine = lamps(reactor, AXIAL)
        occluders = (
            reactor.water(WATER_TOLERANCE),
            reactor.plate(),
            *reactor.lamp_bodies(shadow_shrink(reactor, fine)),
        )
        triangles, part = wall_triangles(reactor, WALL_SIZE, per_turn=PER_TURN)
        figures(reactor, fine, occluders, triangles, part)
        return
    absorption = UniformAbsorption(absorption_from_uvt(UVT))
    say(
        f"grid {H * 1e3:g} mm, walls {WALL_SIZE * 1e3:g} mm, lamp axial {AXIAL * 1e3:g} mm -> {OUT}"
    )

    fine = lamps(reactor, AXIAL)
    coarse = lamps(reactor, WALL_LIGHTING_AXIAL)
    shrink = shadow_shrink(reactor, fine, coarse)
    water = reactor.water(WATER_TOLERANCE)
    occluders = (water, reactor.plate(), *reactor.lamp_bodies(shrink))
    say(
        f"lamps: {fine.n_facets} facets (volume), {coarse.n_facets} (walls), "
        f"{float(np.sum(np.asarray(fine.emission) * np.asarray(fine.area))):.3f} W; "
        f"shadows {shrink * 1e3:.3f} mm inside each sleeve"
    )

    triangles, part = wall_triangles(
        reactor, WALL_SIZE, step_file=OUT / "reactor_water.step", per_turn=PER_TURN
    )
    names = ("vessel", "plate")
    walls = Surfaces.from_triangles(
        jnp.asarray(triangles),
        solid_id=part,
        solid_names=names,
        diffuse_reflectance=REFLECTANCE,
    )
    area = np.asarray(walls.area)
    say(f"walls: {walls.n_facets} triangles ({np.bincount(part).tolist()} vessel/plate)")

    tic = time.perf_counter()
    lit = lamp_light_on_walls(walls, coarse, occluders, absorption)
    say(
        f"lamps' light on the walls: {time.perf_counter() - tic:.0f} s, {np.sum(lit * area):.3f} W lands"
    )

    tic = time.perf_counter()
    model = build_radiation_model(
        np.zeros((0, 3)),
        walls,
        occluders=occluders,
        settings=RadiationSettings(self_occlusion=NoOcclusion()),
    )
    landing, cycles = surface_irradiance(
        model, walls, absorption=absorption, external_irradiance=jnp.asarray(lit)
    )
    landing = np.asarray(landing)
    radiosity = REFLECTANCE * landing
    del model
    say(
        f"radiosity among the walls: {time.perf_counter() - tic:.0f} s, {int(cycles)} restart cycles"
    )

    origin, shape, wet = grid_nodes(reactor, H)
    low = origin
    index = np.flatnonzero(wet)
    ijk = np.column_stack(np.unravel_index(index, shape))
    nodes = low + H * ijk
    say(f"grid {shape.tolist()}: {len(nodes)} nodes in the water of {int(np.prod(shape))}")

    tic = time.perf_counter()
    g_direct = np.asarray(
        direct_fluence_rate(
            fine,
            jnp.asarray(nodes),
            absorption=absorption,
            occluders=occluders,
            self_occlusion=NoOcclusion(),
        )
    )
    say(f"direct G: {time.perf_counter() - tic:.0f} s")
    tic = time.perf_counter()
    glowing = walls.with_optics(emission=jnp.asarray(radiosity))
    g_reflected = np.asarray(
        direct_fluence_rate(
            glowing,
            jnp.asarray(nodes),
            absorption=absorption,
            occluders=occluders,
            self_occlusion=NoOcclusion(),
        )
    )
    say(f"reflected G: {time.perf_counter() - tic:.0f} s")
    g = g_direct + g_reflected

    fields = {}
    for name, values in (("G", g), ("G_direct", g_direct), ("G_reflected", g_reflected)):
        full = np.full(int(np.prod(shape)), np.nan)
        full[index] = values
        fields[name] = full
    write_image(OUT / "fluence_rate.vti", origin, H, shape, fields)
    write_surface(
        OUT / "walls.vtp",
        triangles,
        {"E": landing, "E_from_lamps": lit, "radiosity": radiosity, "part": part.astype(float)},
    )
    write_surface(
        OUT / "lamps.vtp", np.asarray(fine.vertices), {"exitance": np.asarray(fine.emission)}
    )
    np.savez_compressed(
        OUT / "fluence_rate.npz",
        origin=origin,
        h=H,
        shape=shape,
        index=index,
        G=g,
        G_direct=g_direct,
        G_reflected=g_reflected,
    )

    coefficient = float(absorption.coefficient)
    volume = len(nodes) * H**3
    lamp_power = reactor.n_lamps * reactor.lamp_power
    wall_absorbed = {
        names[k]: float(np.sum(((1 - REFLECTANCE) * landing * area)[part == k])) for k in (0, 1)
    }
    downstream = nodes[:, 0] > reactor.plate_far_x
    summary = {
        "reactor": reactor.summary(),
        "settings": {
            "grid_spacing_m": H,
            "wall_triangle_edge_m": WALL_SIZE,
            "lamp_axial_facet_m": AXIAL,
            "lamp_sectors": SECTORS,
            "wall_lighting_axial_facet_m": WALL_LIGHTING_AXIAL,
            "wall_samples": WALL_SAMPLES**2,
            "uvt_percent": UVT,
            "absorption_per_m": coefficient,
            "reflectance": REFLECTANCE,
            "lamp_shadow_shrink_m": shrink,
        },
        "counts": {
            "lamp_facets_volume": fine.n_facets,
            "lamp_facets_walls": coarse.n_facets,
            "wall_triangles": walls.n_facets,
            "grid_shape": shape.tolist(),
            "water_nodes": len(nodes),
            "radiosity_restart_cycles": int(cycles),
        },
        "results": {
            "lamp_power_W": lamp_power,
            "water_volume_on_grid_L": 1e3 * volume,
            "mean_G_W_per_m2": float(g.mean()),
            "mean_G_downstream_of_plate_W_per_m2": float(g[downstream].mean()),
            "mean_G_upstream_of_plate_W_per_m2": float(g[~downstream].mean()),
            "reflected_share_of_mean_G": float(g_reflected.mean() / g.mean()),
            "G_percentiles_1_10_50_90_99": np.percentile(g, [1, 10, 50, 90, 99]).tolist(),
            "water_absorbed_W_grid_estimate": coefficient * float(g.sum()) * H**3,
            "lamps_light_landing_on_walls_W": float(np.sum(lit * area)),
            "walls_absorbed_W": wall_absorbed,
        },
        "seconds": time.perf_counter() - started,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    say(json.dumps(summary["results"], indent=2))
    say(f"field done in {summary['seconds']:.0f} s")
    figures(reactor, fine, occluders, triangles, part)
    say(f"done in {time.perf_counter() - started:.0f} s")


def figures(reactor, fine, occluders, triangles, part) -> None:
    """Every figure for this run, into its directory."""
    import plate_shadow
    import plot
    import tessellation

    plot.main(OUT)
    tessellation.lamp_figure(reactor, OUT, AXIAL, SECTORS, WALL_LIGHTING_AXIAL, SHRINK)
    tessellation.wall_figure(reactor, triangles, part, OUT, WALL_SIZE, PER_TURN)
    tessellation.open_fraction(reactor, triangles, part, OUT)
    plate_shadow.run(reactor, fine, occluders, UniformAbsorption(absorption_from_uvt(UVT)), OUT)


if __name__ == "__main__":
    main()
