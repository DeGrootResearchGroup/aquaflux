"""aquaflux — differentiable unstructured finite-volume flow solver in JAX.

A bespoke, fully differentiable, cell-centred FVM flow solver built to couple with
``aquakin`` for water and environmental engineering reactors.

Importing ``aquaflux`` enables JAX 64-bit mode process-wide, and turns on JAX's
persistent on-disk compilation cache (see below); both are documented, intentional
side effects of ``import aquaflux``.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

# --- JAX x64 enablement (must run before any submodule builds JAX state) --------
# Finite-volume assembly and stiff reactive coupling require 64-bit floats. This is
# global, process-wide JAX configuration, so it is a documented side effect of
# ``import aquaflux``. It is set here, at the top of the package init, before any
# submodule import that would construct JAX arrays/tracers. Do not remove it.
import jax as _jax

_jax.config.update("jax_enable_x64", True)


# --- JAX persistent compilation cache -------------------------------------------
# The coupled solve compiles a large graph (a restarted-GMRES linear solve over the
# whole saddle system, with the AMG V-cycles inlined), which can take minutes and is
# recompiled every fresh process and whenever a preconditioner refresh changes an
# operator shape. XLA can persist those compilations to disk and reuse them across
# runs, turning a repeated multi-minute cost into a one-time one -- with no effect on
# results (a cache hit returns the identical executable). Only compilations slower
# than the threshold below are stored, so the cheap jits that dominate the tests do
# not fill the cache.
#
# This is process-wide JAX state, so it is a documented side effect of the import.
# Override the location with ``AQUAFLUX_COMPILATION_CACHE_DIR`` (respecting
# ``XDG_CACHE_HOME``), or disable it entirely with ``AQUAFLUX_DISABLE_COMPILATION_CACHE=1``
# (e.g. on a read-only or ephemeral filesystem). A failure to configure the cache
# (an unwritable path) is non-fatal: the import proceeds, only without caching.
def _enable_compilation_cache() -> None:
    if _os.environ.get("AQUAFLUX_DISABLE_COMPILATION_CACHE"):
        return
    cache_dir = _os.environ.get("AQUAFLUX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        base = _os.environ.get("XDG_CACHE_HOME") or (_Path.home() / ".cache")
        cache_dir = str(_Path(base) / "aquaflux" / "jax")
    try:
        _Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", cache_dir)
        # Cache only genuinely expensive compilations (the solve steps), not the many
        # sub-second jits, so the cache stays small and useful.
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
    except OSError:
        # An unwritable cache location must not break `import aquaflux`.
        pass


_enable_compilation_cache()

__version__ = "0.0.0"

__all__ = ["__version__"]

# NOTE: the public API surface is intentionally empty at this pre-code scaffold
# stage. Export symbols here as subsystems land (mesh, discretization, solve),
# keeping this file's import order compatible with the x64 enablement above.
