"""Write the turbulent channel cases' meshes, which their case files read (``meshes/*.aqmesh``).

Each is the streamwise-periodic channel of full height 2 and length 1, four cells along it and its
wall-normal cells growing geometrically from each wall toward the centre: 96 cells by 1.09, 120 by
1.075 and 224 by 1.052. The files are tracked; run this to write them again (``--overwrite``
replaces them), and a mesh that differs shows in ``git status``::

    python validation/turbulent_channel/write_meshes.py [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from aquaflux.io import write_aqmesh
from aquaflux.mesh import graded_nodes, structured_grid_2d

#: Each mesh's wall-normal cell count and growth, by its file's name.
MESHES = {"channel_96": (96, 1.09), "channel_120": (120, 1.075), "channel_224": (224, 1.052)}
NX, LENGTH, HEIGHT = 4, 1.0, 2.0


def main(overwrite: bool) -> None:
    directory = pathlib.Path(__file__).resolve().parent / "meshes"
    for name, (ny, growth) in MESHES.items():
        mesh = structured_grid_2d(
            NX, ny, LENGTH, HEIGHT, named_boundaries=True, periodic=("x",),
            y_nodes=graded_nodes(ny, HEIGHT, growth),
        )  # fmt: skip
        provenance = {
            "written_by": "validation/turbulent_channel/write_meshes.py",
            "grid": {"cells": [NX, ny], "lengths": [LENGTH, HEIGHT], "periodic": ["x"],
                     "y_growth": growth},
        }  # fmt: skip
        path = write_aqmesh(
            mesh, directory / f"{name}.aqmesh", overwrite=overwrite, provenance=provenance
        )
        print(f"{path}: {mesh.n_cells} cells", flush=True)


if __name__ == "__main__":
    main("--overwrite" in sys.argv[1:])
