"""Can the CAD kernel binding (OCP) and the Netgen mesher share one Python process?

Each arm runs in a fresh child process, three times, and the table reports the child's exit status
(0 clean; a negative status is the signal that killed it, -6 being SIGABRT). The children only import modules, then exit, so a non-zero status is a
crash in the libraries' load or teardown, not in any meshing.

Usage (an environment holding both ``cadquery-ocp-novtk`` and ``netgen-mesher``)::

    python validation/meshing_netgen/coexistence.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from peak_footprint import run_forwarded

ARMS = {
    "OCP alone": "import OCP.gp",
    "Netgen alone": "import netgen.occ",
    "OCP, then netgen.occ": "import OCP.gp, netgen.occ",
    "OCP, then netgen.meshing (no netgen CAD)": "import OCP.gp, netgen.meshing",
    "netgen.occ, then OCP": "import netgen.occ, OCP.gp",
}
REPEATS = 3


def main() -> None:
    print(f"{'arm':45s} exit status x{REPEATS}", flush=True)
    for name, code in ARMS.items():
        statuses = [
            run_forwarded([sys.executable, "-c", code], capture_output=True).returncode
            for _ in range(REPEATS)
        ]
        print(f"{name:45s} {statuses}", flush=True)


if __name__ == "__main__":
    main()
