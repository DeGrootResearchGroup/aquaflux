"""Pictures of the bend's mesh: its wall, cross-sections at the measuring stations, the plane of
symmetry through the bend, and the quality of its faces.

Reads a grid's mesh (``aquaflux mesh mesh.yaml`` in ``validation/sudo_bend/grids/<grid>``; by default
the coarsest grid's), and writes PNG files into
``validation/sudo_bend/figures/``. Needs the ``ui`` extra (PyVista) and an off-screen OpenGL
(OSMesa or EGL, or a virtual X server on Linux).

Usage::

    python validation/sudo_bend/render_mesh.py [<mesh.aqmesh>]
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from aquaflux.io import read_aqmesh, write_patches, write_vtu
from aquaflux.mesh import face_non_orthogonality, face_skewness
from geometry import Bend

HERE = Path(__file__).parent
MESH = HERE / "grids" / "round36" / "work" / "bend.aqmesh"
FIGURES = HERE / "figures"
SIZE = (1600, 1200)
HEXAHEDRON, OTHER = "#9ec5e8", "#f0a35e"


def cell_kinds(mesh) -> np.ndarray:
    """1 for a cell bounded by six quadrilaterals (a hexahedron), else 0, shape ``(n_cells,)``."""
    fc = mesh.face_cells
    owner, neighbour = np.asarray(fc.owner), np.asarray(fc.neighbour)
    offsets = np.asarray(mesh.face_nodes.offsets)
    quad = np.diff(offsets) == 4
    faces = np.bincount(owner, minlength=mesh.n_cells)
    faces += np.bincount(neighbour[neighbour >= 0], minlength=mesh.n_cells)
    quads = np.bincount(owner, weights=quad, minlength=mesh.n_cells)
    quads += np.bincount(
        neighbour[neighbour >= 0], weights=quad[neighbour >= 0], minlength=mesh.n_cells
    )
    return ((faces == 6) & (quads == 6)).astype(np.int64)


def worst_face(mesh, per_face: np.ndarray) -> np.ndarray:
    """Each cell's largest value over its faces, shape ``(n_cells,)``."""
    fc = mesh.face_cells
    owner, neighbour = np.asarray(fc.owner), np.asarray(fc.neighbour)
    worst = np.zeros(mesh.n_cells)
    np.maximum.at(worst, owner, per_face)
    inside = neighbour >= 0
    np.maximum.at(worst, neighbour[inside], per_face[inside])
    return worst


def section(grid, origin, normal):
    """The cells cut by a plane, as the cut's polygons."""
    return grid.slice(normal=normal, origin=origin, generate_triangles=False)


def plane_camera(plotter, origin, normal, up, distance, zoom):
    """Look straight at a plane, in parallel projection."""
    origin, normal = np.asarray(origin, float), np.asarray(normal, float)
    plotter.camera_position = [tuple(origin + distance * normal), tuple(origin), tuple(up)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = zoom


def kinds_mesh(plotter, mesh, **kwargs):
    plotter.add_mesh(
        mesh, scalars="hexahedron", cmap=[OTHER, HEXAHEDRON], clim=[0, 1], show_edges=True,
        edge_color="#1a1a1a", line_width=0.6, show_scalar_bar=False, lighting=False, **kwargs,
    )  # fmt: skip


def main(path: Path) -> None:
    FIGURES.mkdir(exist_ok=True)
    bend = Bend()
    r, rc, d = bend.radius, bend.bend_radius, bend.diameter
    mesh = read_aqmesh(path)
    kinds = cell_kinds(mesh)
    non_orth = np.asarray(face_non_orthogonality(mesh))
    skew = np.asarray(face_skewness(mesh))
    interior = np.asarray(mesh.face_cells.interior)
    print(
        f"{mesh.n_cells} cells, {kinds.sum()} hexahedra ({100 * kinds.mean():.1f} %); "
        f"non-orthogonality (interior) mean {non_orth[interior].mean():.2f}, "
        f"p99 {np.percentile(non_orth[interior], 99):.2f}, max {non_orth[interior].max():.2f} deg; "
        f"skewness max {skew.max():.3f}",
        flush=True,
    )
    with tempfile.TemporaryDirectory() as work:
        cells = write_vtu(
            mesh,
            {"hexahedron": kinds, "non_orthogonality": worst_face(mesh, non_orth)},
            Path(work) / "cells.vtu",
        )
        patches = write_patches(mesh, None, Path(work) / "patches.vtm")
        grid = pv.read(cells)
        blocks = pv.read(patches)
        boundary = {name: blocks[name] for name in blocks.keys()}

        # The whole pipe and a close look at the bend's wall.
        for name, focus, zoom_box in (
            ("mesh_overview.png", None, None),
            ("mesh_bend_wall.png", (0.5 * rc, 0.5 * rc, 0.0), 0.33),
        ):
            p = pv.Plotter(off_screen=True, window_size=SIZE)
            p.set_background("white")
            p.add_mesh(boundary["walls"], color="#d9d9d9", show_edges=True,
                       edge_color="#303030", line_width=0.5)  # fmt: skip
            p.add_mesh(boundary["inlet"], color="#7fc97f", show_edges=True, line_width=0.5)
            p.add_mesh(boundary["outlet"], color="#ef8a8a", show_edges=True, line_width=0.5)
            if focus is None:
                p.camera_position = "iso"
                p.camera.azimuth = -60.0
                p.camera.elevation = 10.0
                p.reset_camera()
                p.camera.zoom(1.3)
            else:
                p.camera_position = [(0.55, -0.45, 0.55), focus, (0.0, 0.0, 1.0)]
                p.enable_parallel_projection()
                p.camera.parallel_scale = zoom_box
            p.screenshot(FIGURES / name)
            p.close()

        # Cross-sections at the measuring stations, each seen along the flow.
        stations = [("1 D upstream", (-d, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
        for degrees in (0, 30, 60, 90):
            a = math.radians(degrees)
            centre = (rc * math.sin(a), rc * (1.0 - math.cos(a)), 0.0)
            # An exact station can fall on a cross-section of faces; a hair past it cuts cells.
            a += 1e-4
            centre = (rc * math.sin(a), rc * (1.0 - math.cos(a)), 0.0)
            stations.append(
                (f"bend {degrees} deg", centre, (math.cos(a), math.sin(a), 0.0), (0, 0, 1))
            )
        stations.append(("5 D downstream", (rc, rc + 5 * d, 0.0), (0.0, 1.0, 0.0), (0, 0, 1)))
        images = []
        for label, origin, normal, up in stations:
            cut = section(grid, origin, normal)
            p = pv.Plotter(off_screen=True, window_size=(900, 900))
            p.set_background("white")
            kinds_mesh(p, cut)
            plane_camera(p, origin, normal, up, 1.0, 1.08 * r)
            image = p.screenshot(None, return_img=True)
            p.close()
            images.append((label, image))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10.4))
        for axis, (label, image) in zip(axes.flat, images, strict=True):
            axis.imshow(image)
            axis.set_title(label, fontsize=14)
            axis.axis("off")
        fig.suptitle(
            "Cross-sections, seen along the flow (+z up; the bend's inside on the left in the bend)"
            "\nblue: hexahedra, orange: other polyhedra",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(FIGURES / "mesh_sections.png", dpi=110)
        plt.close(fig)

        # One cross-section close to the wall: the layers and the shell.
        cut = section(grid, stations[3][1], stations[3][2])
        p = pv.Plotter(off_screen=True, window_size=SIZE)
        p.set_background("white")
        kinds_mesh(p, cut)
        a = math.radians(30.0)
        outer = np.array(stations[3][1]) + r * np.array(
            [math.cos(a - 0.5 * math.pi), math.sin(a - 0.5 * math.pi), 0.0]
        )
        plane_camera(
            p,
            outer * 0.75 + np.array(stations[3][1]) * 0.25,
            stations[3][2],
            (0, 0, 1),
            1.0,
            0.35 * r,
        )
        p.screenshot(FIGURES / "mesh_section_wall.png")
        p.close()

        # The plane of symmetry through the bend, seen from above. Faces of the O-grid's core lie
        # in it, so the cut is taken a millimetre above.
        cut = section(grid, (0.0, 0.0, 1e-3), (0.0, 0.0, 1.0))
        a = math.radians(45.0)
        inner = (0.0, rc, 0.0) + (rc - 0.6 * r) * np.array([math.sin(a), -math.cos(a), 0.0])
        for name, focus, zoom in (
            ("mesh_symmetry_plane.png", (0.45 * rc, 0.55 * rc, 0.0), 0.27),
            ("mesh_symmetry_plane_inner_wall.png", inner, 0.045),
        ):
            p = pv.Plotter(off_screen=True, window_size=SIZE)
            p.set_background("white")
            kinds_mesh(p, cut)
            plane_camera(p, focus, (0, 0, 1), (0, 1, 0), 1.0, zoom)
            p.screenshot(FIGURES / name)
            p.close()

        # The inlet face: the O-grid the whole run is swept from.
        p = pv.Plotter(off_screen=True, window_size=(1200, 1200))
        p.set_background("white")
        p.add_mesh(boundary["inlet"], color=HEXAHEDRON, show_edges=True, edge_color="#1a1a1a",
                   line_width=0.8, lighting=False)  # fmt: skip
        plane_camera(p, bend.inlet_centre, (-1.0, 0.0, 0.0), (0, 0, 1), 1.0, 1.05 * r)
        p.screenshot(FIGURES / "mesh_inlet.png")
        p.close()

    # The faces' quality.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(non_orth[interior], bins=60, color="#4c72b0")
    axes[0].set_xlabel("non-orthogonality of interior faces (deg)")
    axes[1].hist(skew, bins=60, color="#dd8452")
    axes[1].set_xlabel("skewness of faces")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylabel("faces")
    fig.suptitle(f"{mesh.n_cells} cells, {100 * kinds.mean():.1f} % hexahedra")
    fig.tight_layout()
    fig.savefig(FIGURES / "mesh_quality.png", dpi=110)
    plt.close(fig)
    print(f"wrote the figures into {FIGURES}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else MESH)
