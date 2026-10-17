"""Load a repository script from ``tools/`` as a module, for a test that calls its functions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_tool(name: str) -> ModuleType:
    """The script ``tools/<name>.py``, imported once and cached under ``sys.modules``.

    The scripts are not a package (they run as ``tools/<name>.py``), so a test reaches one by path.
    """
    key = f"_aquaflux_tool_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, TOOLS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]
