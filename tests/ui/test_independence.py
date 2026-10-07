"""The viewer and the solver do not import each other.

The viewer reads files, so it must run where JAX is absent or unusable; the solver must never pay for
VTK and trame. Each direction is checked the way it could break: the viewer by importing all of it in
a fresh interpreter and looking at what came in, the solver by parsing every module for an import.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("trame")

ROOT = Path(__file__).resolve().parents[2]


def test_importing_the_whole_viewer_imports_neither_the_solver_nor_jax():
    probe = (
        "import sys\n"
        "import aquaflux_ui, aquaflux_ui.app, aquaflux_ui.__main__\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'aquaflux', 'jax', 'jaxlib'}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert result.stdout.strip() == "[]"


def test_no_solver_module_imports_the_viewer():
    offenders = []
    for path in sorted((ROOT / "aquaflux").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            offenders += [
                f"{path.relative_to(ROOT)}: {name}"
                for name in names
                if name.split(".")[0] == "aquaflux_ui"
            ]
    assert offenders == []
