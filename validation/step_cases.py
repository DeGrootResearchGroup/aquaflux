"""The two step cases' own ``compare.py`` modules, loaded by path, for the harnesses that study either.

Both cases have a module named ``compare``, so a plain ``import compare`` resolves by whatever order the
paths sit in. Loading each file explicitly, under a name of its own, makes the case a harness builds
unambiguous; the case's assembly (``build_case``) and its solver (``SOLVER``) are then the case's own,
never restated by the harness.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
#: Each step case's name, as a harness's environment spells it, and its directory under ``validation/``.
CASES = {"bfs3d": "bfs3d_openfoam", "pitzdaily": "pitzdaily_openfoam"}


def load_case(name: str):
    """The ``compare.py`` module of step case ``name`` (``bfs3d`` or ``pitzdaily``).

    Puts the repository root and the case's directory on ``sys.path`` first, as running the file would.
    """
    if name not in CASES:
        raise SystemExit(f"the case must be one of {sorted(CASES)}, not {name!r}")
    directory = HERE / CASES[name]
    for path in (ROOT, directory):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(f"{name}_case", directory / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
