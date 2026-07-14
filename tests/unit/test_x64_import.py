"""x64 enablement is a documented side effect of ``import aquaflux``.

Finite-volume assembly requires 64-bit floats; ``aquaflux/__init__.py`` enables JAX
x64 mode process-wide at import. Because that state is process-global, the
fresh-import behaviour is checked in a subprocess so this test is order-independent
of whatever else has already configured JAX in the current process.
"""

from __future__ import annotations

import subprocess
import sys


def test_x64_enabled_after_import() -> None:
    """Importing aquaflux leaves JAX in x64 mode in the current process."""
    import aquaflux  # noqa: F401  (import side effect is the thing under test)
    import jax

    assert jax.config.x64_enabled


def test_fresh_import_enables_x64() -> None:
    """A clean subprocess that imports aquaflux ends up in x64 mode."""
    code = "import aquaflux; import jax; assert jax.config.x64_enabled; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
