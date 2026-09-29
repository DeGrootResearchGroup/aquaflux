"""Slide figures and the error table: the reference, the discrete-ordinates runs and aquaflux.

Everything is drawn on the floor patch's own faces (each face one colour, from the fan triangles
``patches.py`` stored), so no field is resampled onto a grid it was not computed on. Every map of
one quantity shares one colour scale, set by the reference.

Outputs in ``work/figures/``, each 16:9:

- ``floor_linear.png`` / ``floor_log.png`` -- reference | DOM runs | aquaflux, bunny case;
- ``floor_error.png`` -- each solver minus the reference, as a share of the reference's peak;
- ``profile.png`` -- a line across the shadow at ``y = PROFILE_Y``, all solvers overlaid;
- ``empty_room.png`` -- the empty room, the ray effect on its own;
- ``summary.json`` and ``summary.md`` -- L2 and maximum error, wall time, directions, cells.

The error norms are over the floor faces within ``MAP_HALF_WIDTH`` of the centre, area-weighted:
``L2 = sqrt(sum A (E - E_ref)^2 / sum A) / max E_ref`` and ``max = max |E - E_ref| / max E_ref``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import patches  # noqa: E402
import room  # noqa: E402

WORK = HERE / "work"
OUT = WORK / "figures"
SLIDE = (16, 9)


def load(mesh: str) -> dict:
    """Every floor field available for one mesh: the reference, aquaflux, and each DOM run."""
    geometry = patches.load(mesh)
    floor = geometry["floor"]
    n_cells, n_faces = geometry["_mesh"]["n_cells"], len(floor["area"])
    fields: dict[str, dict] = {}
    reference = WORK / "reference" / f"{mesh}.npz"
    if reference.exists():
        record = json.loads(reference.with_suffix(".json").read_text())
        fields["reference"] = {"E": np.load(reference)["E"], "label": "Reference", "record": record}
    ours = WORK / "aquaflux" / f"{mesh}.npz"
    if ours.exists():
        record = json.loads(ours.with_suffix(".json").read_text())
        # The refined lamp's field: the one whose lamp discretization has been shown not to matter.
        fields["aquaflux"] = {
            "E": np.load(ours)["E_refined"],
            "label": "aquaflux",
            "record": record,
        }
    runs = sorted(
        (WORK / "runs").glob(f"{mesh}_nphi*"),
        key=_dom_order,
    )
    for run in runs:
        record = json.loads((run / "record.json").read_text())
        values = patches.patch_values(run / "qin", room.FLOOR_PATCH, n_cells, n_faces)
        fields[run.name] = {
            "E": values,
            "label": _dom_label(record),
            "record": record,
        }
    return {"floor": floor, "fields": fields, "n_cells": n_cells}


def _dom_label(record: dict) -> str:
    """A DOM run's panel title: its directions and its pixels (every record states both)."""
    return (
        f"DOM, {record['directions']} directions, "
        f"{record['n_pixel_phi']}x{record['n_pixel_theta']} pixels"
    )


def _dom_order(run: Path) -> tuple[int, int]:
    """DOM runs by direction count, then by pixels per bin."""
    record = json.loads((run / "record.json").read_text())
    return record["directions"], record["n_pixel_phi"] * record["n_pixel_theta"]


def _window(floor) -> np.ndarray:
    return np.all(np.abs(floor["centre"][:, :2]) <= room.MAP_HALF_WIDTH, axis=1)


def _baseline(case: dict) -> str:
    """The field the others are measured against: the brute-force reference where there is one,
    and aquaflux in the reflecting room, which the reference cannot do."""
    return case.get("baseline", "reference")


def errors(case: dict) -> dict:
    """L2 and maximum difference of every field from the baseline, as shares of its peak."""
    floor, fields = case["floor"], case["fields"]
    base = _baseline(case)
    inside = _window(floor)
    exact = fields[base]["E"][inside]
    area = floor["area"][inside]
    peak = exact.max()
    out = {}
    for name, field in fields.items():
        if name == base:
            continue
        difference = field["E"][inside] - exact
        out[name] = {
            "L2_of_peak": float(np.sqrt(np.sum(area * difference**2) / area.sum()) / peak),
            "max_of_peak": float(np.abs(difference).max() / peak),
            "floor_power_W": float(np.sum(field["E"] * floor["area"])),
        }
    return out


def _draw(ax, floor, values, **style):
    triangles = floor["triangles"]
    face = floor["triangle_face"]
    inside = _window(floor)[face]
    xy = triangles[inside][:, :, :2].reshape(-1, 2)
    shown = ax.tripcolor(
        xy[:, 0],
        xy[:, 1],
        np.arange(len(xy)).reshape(-1, 3),
        facecolors=values[face[inside]],
        edgecolors="none",
        **style,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-room.MAP_HALF_WIDTH, room.MAP_HALF_WIDTH)
    ax.set_ylim(-room.MAP_HALF_WIDTH, room.MAP_HALF_WIDTH)
    ax.set_xlabel("x (m)")
    return shown


def maps(case: dict, names: list[str], path: Path, *, log: bool, title: str) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    fields = case["fields"]
    peak = fields[_baseline(case)]["E"][_window(case["floor"])].max()
    scale = 1e6 / 1e4  # W/m^2 to uW/cm^2
    norm = LogNorm(peak * scale * 1e-3, peak * scale) if log else Normalize(0.0, peak * scale)
    fig, axes = plt.subplots(1, len(names), figsize=SLIDE, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        values = np.maximum(fields[name]["E"] * scale, peak * scale * 1e-3 if log else 0.0)
        shown = _draw(ax, case["floor"], values, cmap="inferno", norm=norm)
        ax.set_title(fields[name]["label"], fontsize=14)
    axes[0].set_ylabel("y (m)")
    fig.colorbar(shown, ax=axes, shrink=0.6, label="floor irradiance (µW/cm²)")
    fig.suptitle(title, fontsize=16)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def error_maps(case: dict, names: list[str], path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fields = case["fields"]
    base = _baseline(case)
    exact = fields[base]["E"]
    peak = exact[_window(case["floor"])].max()
    fig, axes = plt.subplots(1, len(names), figsize=SLIDE, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names, strict=True):
        share = 100.0 * (fields[name]["E"] - exact) / peak
        shown = _draw(ax, case["floor"], share, cmap="RdBu_r", norm=Normalize(-30.0, 30.0))
        ax.set_title(f"{fields[name]['label']} minus {fields[base]['label']}", fontsize=14)
    axes[0].set_ylabel("y (m)")
    fig.colorbar(
        shown, ax=axes, shrink=0.6, label=f"difference (% of {fields[base]['label']}'s peak)"
    )
    fig.suptitle(f"Floor irradiance: difference from {fields[base]['label']}", fontsize=16)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def profile(case: dict, names: list[str], path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    floor, fields = case["floor"], case["fields"]
    centre = floor["centre"]
    # The row of floor faces the line crosses: those whose centre is within half a face of it.
    half = 0.5 * np.sqrt(np.median(floor["area"][_window(floor)]))
    row = np.flatnonzero(
        (np.abs(centre[:, 1] - room.PROFILE_Y) < half)
        & (centre[:, 0] >= room.PROFILE_X[0])
        & (centre[:, 0] <= room.PROFILE_X[1])
    )
    row = row[np.argsort(centre[row, 0])]
    scale = 1e6 / 1e4
    fig, ax = plt.subplots(figsize=SLIDE, layout="constrained")
    for name in names:
        style = {"color": "k", "lw": 2.5} if name == "reference" else {"lw": 1.5}
        if name == "aquaflux" and _baseline(case) == "aquaflux":
            style = {"color": "k", "lw": 2.5}
        elif name == "aquaflux":
            style = {"color": "C3", "lw": 1.5, "ls": "--"}
        ax.plot(
            centre[row, 0], fields[name]["E"][row] * scale, label=fields[name]["label"], **style
        )
    ax.set_xlabel("x (m)", fontsize=14)
    ax.set_ylabel("floor irradiance (µW/cm²)", fontsize=14)
    ax.set_title(f"Across the shadow at y = {room.PROFILE_Y} m", fontsize=16)
    ax.legend(fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summary_table(results: dict) -> str:
    lines = [
        "| mesh | solver | directions | cells | wall time (s) | hardware | L2 / peak | max / peak |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mesh, case in results.items():
        for name, field in case["fields"].items():
            record = field["record"]
            error = case["errors"].get(name, {})
            if "_nphi" in name:
                directions, seconds = record["directions"], record["wall_seconds"]
                hardware = f"CPU, {record['mpi_ranks']} MPI ranks"
            elif name == "aquaflux" and "timing_s" in record:  # the reflecting room
                directions = f"exact ({record['room_triangles']} room triangles, reflectance {record['wall_reflectance']})"
                seconds = record["timing_s"]["total"]
                hardware = f"CPU ({record['backend']}), {record['cpu_count']} cores"
            elif name == "aquaflux":
                directions = f"exact ({record['refined_lamp_facets']} lamp facets)"
                seconds = round(sum(record["timing"]["refined"].values()), 1)
                hardware = f"CPU ({record['backend']}), {record['cpu_count']} cores"
            else:
                directions = f"{record['window_samples']} window samples"
                seconds = record["seconds"]
                hardware = "CPU, Embree"
            lines.append(
                f"| {mesh} | {field['label']} | {directions} | {case['n_cells']} | {seconds} | "
                f"{hardware} | {error.get('L2_of_peak', 0):.4f} | {error.get('max_of_peak', 0):.4f} |"
            )
    return "\n".join(lines) + "\n"


def load_volume(mesh: str) -> dict | None:
    """``G`` on each slice from every source, and over the whole volume from aquaflux and DOM."""
    cells = np.load(WORK / mesh / "cells.npz")
    reference = WORK / "reference" / f"{mesh}.npz"
    ours = WORK / "aquaflux" / f"{mesh}_volume.npz"
    if not reference.exists() or not ours.exists():
        return None
    stored_reference, stored_ours = np.load(reference), np.load(ours)
    from aquaflux.io.openfoam.fields import parse_scalar_field
    from aquaflux.io.openfoam.foamfile import read_foam_body

    volume = {"aquaflux": {"G": stored_ours["G"], "label": "aquaflux"}}
    for run in sorted(
        (WORK / "runs").glob(f"{mesh}_nphi*"),
        key=_dom_order,
    ):
        record = json.loads((run / "record.json").read_text())
        volume[run.name] = {
            "G": parse_scalar_field(read_foam_body(run / "G"), len(cells["volume"]), {}),
            "label": _dom_label(record),
        }
    slices = {}
    for name in room.SLICES:
        chosen = stored_reference[f"cells_{name}"]
        fields = {"reference": {"G": stored_reference[f"G_{name}"], "label": "Reference"}}
        for key, field in volume.items():
            fields[key] = {"G": field["G"][chosen], "label": field["label"]}
        # On the slices aquaflux's field is the refined lamp's (1,120 facets leave banding in the
        # bunny's penumbrae, 2.9 % at the 99th percentile); the whole volume has only the coarse.
        where = np.searchsorted(stored_ours["slice_cells"], chosen)
        if not np.array_equal(stored_ours["slice_cells"][where], chosen):
            msg = f"{mesh}: the reference's {name} slice cells are not among aquaflux's"
            raise ValueError(msg)
        fields["aquaflux"]["G"] = stored_ours["G_refined"][where]
        slices[name] = {"cells": chosen, "fields": fields}
    return {
        "centre": cells["centre"],
        "volume": cells["volume"],
        "fields": volume,
        "slices": slices,
    }


def load_reflecting(mesh: str) -> tuple[dict, dict] | None:
    """The reflecting room's floor case and volume case, measured against aquaflux.

    There is no brute-force reference here -- it cannot follow light that has bounced -- so the
    DOM runs are compared with aquaflux, whose own checks (energy, facet size, two routes to the
    slices) are in its record.
    """
    ours = WORK / "aquaflux" / f"{mesh}_reflecting.npz"
    if not ours.exists():
        return None
    stored = np.load(ours)
    record = json.loads(ours.with_suffix(".json").read_text())
    geometry = patches.load(mesh)
    floor = geometry["floor"]
    cells = np.load(WORK / mesh / "cells.npz")
    n_cells = len(cells["volume"])
    from aquaflux.io.openfoam.fields import parse_scalar_field
    from aquaflux.io.openfoam.foamfile import read_foam_body

    label = "aquaflux"
    floor_fields = {"aquaflux": {"E": stored["E"], "label": label, "record": record}}
    volume_fields = {"aquaflux": {"G": stored["G"], "label": label}}
    for run in sorted(
        (WORK / "runs").glob(f"{mesh}_reflecting_nphi*"),
        key=_dom_order,
    ):
        run_record = json.loads((run / "record.json").read_text())
        text = _dom_label(run_record)
        floor_fields[run.name] = {
            "E": patches.patch_values(run / "qin", room.FLOOR_PATCH, n_cells, len(floor["area"])),
            "label": text,
            "record": run_record,
        }
        volume_fields[run.name] = {
            "G": parse_scalar_field(read_foam_body(run / "G"), n_cells, {}),
            "label": text,
        }
    slices = {}
    for name in room.SLICES:
        chosen = room.slice_cells(cells["centre"], cells["volume"], name)
        where = np.searchsorted(stored["slice_cells"], chosen)
        fields = {k: {"G": f["G"][chosen], "label": f["label"]} for k, f in volume_fields.items()}
        fields["aquaflux"]["G"] = stored["G_slices"][where]  # the refined lamp's, as elsewhere
        slices[name] = {"cells": chosen, "fields": fields}
    floor_case = {
        "floor": floor,
        "fields": floor_fields,
        "n_cells": n_cells,
        "baseline": "aquaflux",
    }
    volume_case = {
        "centre": cells["centre"],
        "volume": cells["volume"],
        "fields": volume_fields,
        "slices": slices,
        "baseline": "aquaflux",
    }
    return floor_case, volume_case


def volume_errors(case: dict) -> dict:
    """Slice errors against the reference, and whole-volume differences from aquaflux.

    On a slice: area-weighted over its cells, as shares of the reference's 99th percentile there
    (its maximum sits in the cells touching the lamp, and would say nothing about the room). Over
    the volume: each DOM field against aquaflux, volume-weighted, as shares of aquaflux's
    volume-weighted mean -- there is no reference over the whole volume, and aquaflux's field there
    is the 1,120-facet lamp's (within 2.9 % of the refined lamp's at the 99th percentile on the
    slices, far below the differences this measures).
    """
    out = {"slices": {}, "volume": {}}
    for name, piece in case["slices"].items():
        side = np.cbrt(case["volume"][piece["cells"]]) ** 2
        base = _baseline(case)
        exact = piece["fields"][base]["G"]
        scale = np.percentile(exact, 99)
        out["slices"][name] = {
            key: {
                "L2_of_p99": float(
                    np.sqrt(np.sum(side * (f["G"] - exact) ** 2) / side.sum()) / scale
                ),
                "median_abs_rel": float(
                    np.median(np.abs(f["G"] - exact) / np.maximum(exact, 1e-30))
                ),
            }
            for key, f in piece["fields"].items()
            if key != base
        }
    ours, weight = case["fields"]["aquaflux"]["G"], case["volume"]
    mean = np.sum(weight * ours) / weight.sum()
    for key, field in case["fields"].items():
        if key == "aquaflux":
            continue
        difference = field["G"] - ours
        out["volume"][key] = {
            "L2_of_mean": float(np.sqrt(np.sum(weight * difference**2) / weight.sum()) / mean),
            "volume_integral_ratio": float(np.sum(weight * field["G"]) / np.sum(weight * ours)),
        }
    return out


def slice_maps(case: dict, name: str, path: Path, *, error: bool) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm, Normalize

    piece = case["slices"][name]
    axis, _ = room.SLICES[name]
    across = [i for i in range(3) if i != axis]
    centre = case["centre"][piece["cells"]][:, across]
    half = 0.5 * np.cbrt(case["volume"][piece["cells"]])[:, None]
    corners = np.stack(
        [centre + half * np.array(sign) for sign in ((-1, -1), (1, -1), (1, 1), (-1, 1))], axis=1
    )
    base = _baseline(case)
    exact = piece["fields"][base]["G"]
    order = [base, *[k for k in piece["fields"] if "_nphi" in k]]
    if base != "aquaflux":
        order.append("aquaflux")
    if error:
        order = order[1:]
    scale = 1e6 / 1e4  # W/m^2 to uW/cm^2
    top = np.percentile(exact, 99.5) * scale
    fig, axes = plt.subplots(1, len(order), figsize=SLIDE, layout="constrained")
    for ax, key in zip(np.atleast_1d(axes), order, strict=True):
        values = piece["fields"][key]["G"]
        if error:
            colours = 100.0 * (values - exact) / (top / scale)
            base_label = piece["fields"][base]["label"]
            norm, cmap, label = (
                Normalize(-30, 30),
                "RdBu_r",
                f"difference (% of {base_label}'s p99.5)",
            )
            title = f"{piece['fields'][key]['label']} minus {base_label}"
        else:
            colours = np.maximum(values * scale, top * 1e-5)
            norm, cmap, label = LogNorm(top * 1e-5, top), "inferno", "fluence rate (µW/cm²)"
            title = piece["fields"][key]["label"]
        shown = ax.add_collection(
            PolyCollection(corners, array=colours, cmap=cmap, norm=norm, edgecolors="none")
        )
        ax.set_xlim(room.ROOM_LOW[across[0]], room.ROOM_HIGH[across[0]])
        ax.set_ylim(room.ROOM_LOW[across[1]], room.ROOM_HIGH[across[1]])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("xyz"[across[0]] + " (m)")
    np.atleast_1d(axes)[0].set_ylabel("xyz"[across[1]] + " (m)")
    fig.colorbar(shown, ax=axes, shrink=0.6, label=label)
    where = f"{'yz'[axis - 1] if axis else 'x'} = {room.SLICES[name][1]} m"
    fig.suptitle(f"Fluence rate on the {name} slice, {where}", fontsize=16)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def black_versus_reflecting(mesh: str) -> None:
    """aquaflux in the black room beside the reflecting room, and the reflected light on its own.

    Floor irradiance on one linear scale, and the fluence rate on both slices on one log scale, so
    the three panels of each row compare directly.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm, Normalize

    reflecting = WORK / "aquaflux" / f"{mesh}_reflecting.npz"
    if not reflecting.exists():
        return
    lit = np.load(reflecting)
    black_floor = np.load(WORK / "aquaflux" / f"{mesh}.npz")["E_refined"]
    black_volume = np.load(WORK / "aquaflux" / f"{mesh}_volume.npz")
    floor = patches.load(mesh)["floor"]
    cells = np.load(WORK / mesh / "cells.npz")
    scale = 1e6 / 1e4  # W/m^2 to uW/cm^2
    titles = ("black room", f"reflecting room ({room.WALL_REFLECTANCE})", "reflected light alone")

    fig, axes = plt.subplots(1, 3, figsize=SLIDE, layout="constrained")
    top = max(black_floor.max(), lit["E"].max()) * scale
    for ax, values in zip(axes[:2], (black_floor, lit["E"]), strict=True):
        shown = _draw(ax, floor, values * scale, cmap="inferno", norm=Normalize(0.0, top))
    fig.colorbar(shown, ax=axes[:2], shrink=0.6, label="floor irradiance (µW/cm²)")
    # The reflected light alone is about a tenth of the direct and nearly uniform, so it gets a
    # scale of its own, from its minimum to its maximum.
    inside = _window(floor)
    bounced = lit["E_reflected"] * scale
    own = _draw(
        axes[2],
        floor,
        bounced,
        cmap="inferno",
        norm=Normalize(bounced[inside].min(), bounced[inside].max()),
    )
    fig.colorbar(own, ax=axes[2], shrink=0.6, label="reflected irradiance (µW/cm²), own scale")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=14)
    axes[0].set_ylabel("y (m)")
    fig.suptitle("aquaflux: floor irradiance, black and reflecting room", fontsize=16)
    fig.savefig(OUT / "aquaflux_black_vs_reflecting_floor.png", dpi=200)
    plt.close(fig)

    for name in room.SLICES:
        chosen = room.slice_cells(cells["centre"], cells["volume"], name)
        black = black_volume["G_refined"][np.searchsorted(black_volume["slice_cells"], chosen)]
        where = np.searchsorted(lit["slice_cells"], chosen)
        total, bounced = lit["G_slices"][where], lit["G_slices_reflected"][where]
        axis, value = room.SLICES[name]
        across = [i for i in range(3) if i != axis]
        centre = cells["centre"][chosen][:, across]
        half = 0.5 * np.cbrt(cells["volume"][chosen])[:, None]
        corners = np.stack(
            [centre + half * np.array(sign) for sign in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
            axis=1,
        )
        peak = np.percentile(total, 99.5) * scale
        norm = LogNorm(peak * 1e-5, peak)
        fig, axes = plt.subplots(1, 3, figsize=SLIDE, layout="constrained")
        for ax, values, title in zip(axes, (black, total, bounced), titles, strict=True):
            shown = ax.add_collection(
                PolyCollection(
                    corners,
                    array=np.maximum(values * scale, peak * 1e-5),
                    cmap="inferno",
                    norm=norm,
                    edgecolors="none",
                )
            )
            ax.set_xlim(room.ROOM_LOW[across[0]], room.ROOM_HIGH[across[0]])
            ax.set_ylim(room.ROOM_LOW[across[1]], room.ROOM_HIGH[across[1]])
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("xyz"[across[0]] + " (m)")
        axes[0].set_ylabel("xyz"[across[1]] + " (m)")
        fig.colorbar(shown, ax=axes, shrink=0.6, label="fluence rate (µW/cm²)")
        fig.suptitle(
            f"aquaflux: fluence rate on the {name} slice ({'xyz'[axis]} = {value} m)", fontsize=16
        )
        fig.savefig(OUT / f"aquaflux_black_vs_reflecting_G_{name}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for mesh in ("bunny", "empty"):
        if not patches.npz_path(mesh).exists():
            continue
        case = load(mesh)
        if "reference" not in case["fields"]:
            print(f"{mesh}: no reference yet, skipped", flush=True)
            continue
        case["errors"] = errors(case)
        results[mesh] = case
        dom = [n for n in case["fields"] if n.startswith(mesh + "_nphi")]
        order = (
            ["reference", *dom, "aquaflux"] if "aquaflux" in case["fields"] else ["reference", *dom]
        )
        prefix = "floor" if mesh == "bunny" else "empty_room"
        title = "Floor irradiance under the Voronoi bunny" if mesh == "bunny" else "Empty room"
        maps(case, order, OUT / f"{prefix}_linear.png", log=False, title=title)
        maps(case, order, OUT / f"{prefix}_log.png", log=True, title=title + " (log scale)")
        error_maps(case, order[1:], OUT / f"{prefix}_error.png")
        profile(case, order, OUT / f"{prefix}_profile.png")
        print(f"{mesh}: {json.dumps(case['errors'], indent=1)}", flush=True)
        volume = load_volume(mesh)
        if volume is not None:
            case["volume_errors"] = volume_errors(volume)
            for name in room.SLICES:
                slice_maps(volume, name, OUT / f"{prefix}_G_{name}.png", error=False)
                slice_maps(volume, name, OUT / f"{prefix}_G_{name}_error.png", error=True)
            print(f"{mesh} volume: {json.dumps(case['volume_errors'], indent=1)}", flush=True)
    reflecting = load_reflecting("bunny")
    if reflecting is not None:
        case, volume = reflecting
        case["errors"] = errors(case)
        case["volume_errors"] = volume_errors(volume)
        results["bunny, reflecting"] = case
        order = ["aquaflux", *[n for n in case["fields"] if "_nphi" in n]]
        title = f"Reflecting room (walls, floor, ceiling at {room.WALL_REFLECTANCE})"
        maps(case, order, OUT / "reflecting_linear.png", log=False, title=title)
        maps(case, order, OUT / "reflecting_log.png", log=True, title=title + " (log scale)")
        if len(order) > 1:
            error_maps(case, order[1:], OUT / "reflecting_error.png")
        profile(case, order, OUT / "reflecting_profile.png")
        for name in room.SLICES:
            slice_maps(volume, name, OUT / f"reflecting_G_{name}.png", error=False)
            if len(order) > 1:
                slice_maps(volume, name, OUT / f"reflecting_G_{name}_error.png", error=True)
        print(f"reflecting: {json.dumps(case['errors'], indent=1)}", flush=True)
    black_versus_reflecting("bunny")
    table = summary_table(results)
    (OUT / "summary.md").write_text(table)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                mesh: {"floor": case["errors"], "volume": case.get("volume_errors")}
                for mesh, case in results.items()
            },
            indent=2,
        )
        + "\n"
    )
    print(table, flush=True)


if __name__ == "__main__":
    main()
