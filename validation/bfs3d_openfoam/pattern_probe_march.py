"""bfs3d's march with its pressure column probed for ``R2 + S(tau)`` rather than at a fixed reach.

The probe colours the pressure column for the reach-2 pattern plus the distance-3 pairs a static,
geometry-only strength skeleton calls strong at ``tau`` (``validation/pressure_strength_skeleton.py``),
and every other column at ``BFS3D_PATTERN_OTHERS`` (2 by default), all inside the reach-3 stored
pattern. No case file can state such a probe, so this script installs it where the march's
materialized-Jacobian session would build its own from the case's settings, and otherwise runs
``compare.py`` unchanged: same case file, same ``BFS3D_*`` overrides, same log and checkpoints.

Usage::

    BFS3D_PATTERN_TAU=0.01 BFS3D_NU_PERTURBATION=0 \\
        validation/run_case.sh validation/bfs3d_openfoam/pattern_probe_march.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import compare  # noqa: E402
from aquaflux.precondition import FieldSplit  # noqa: E402
from aquaflux.precondition.materialized_session import MaterializedSession  # noqa: E402
from pressure_strength_skeleton import pattern_jacobian_probe  # noqa: E402

#: The skeleton's strength threshold for the pressure column's distance-3 pairs.
TAU = float(os.environ.get("BFS3D_PATTERN_TAU", "0.01"))
#: The reach every other column is probed at.
OTHERS_REACH = int(os.environ.get("BFS3D_PATTERN_OTHERS", "2"))
#: The pressure column's index in bfs3d's ``[u, v, w, p, k, omega]`` state.
PRESSURE = 3

_library_probe_for = MaterializedSession._probe_for


def _pattern_probe_for(session):
    """The session's probe: the pattern probe for the coupled problem, the library's otherwise."""
    problem = session._problem
    if session._probe is None and hasattr(problem, "coupled"):
        spec = session._spec
        session._probe = pattern_jacobian_probe(
            problem.coupled,
            problem.groups() if isinstance(spec.inverse, FieldSplit) else None,
            PRESSURE,
            TAU,
            spec.probe.gradient_sweeps,
            problem.production_viscosity,
            others_reach=OTHERS_REACH,
        )
    return _library_probe_for(session)


if __name__ == "__main__":
    print(
        f"[pattern probe] p coloured for R2 + S({TAU:g}), the rest at {OTHERS_REACH}, inside the "
        "reach-3 pattern; the case file's column_reach is not used",
        flush=True,
    )
    MaterializedSession._probe_for = _pattern_probe_for
    compare.main()
