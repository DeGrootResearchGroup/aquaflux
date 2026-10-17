"""aquaflux — differentiable unstructured finite-volume flow solver in JAX.

A bespoke, fully differentiable, cell-centred FVM flow solver built to couple with
``aquakin`` for water and environmental engineering reactors.

Importing ``aquaflux`` enables JAX 64-bit mode process-wide, and turns on JAX's
persistent on-disk compilation cache (see below); both are documented, intentional
side effects of ``import aquaflux``.
"""

from __future__ import annotations

import importlib.util as _importlib_util
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
# than a floor are stored (``_DEFAULT_CACHE_FLOOR_SECS`` below), so the many small
# programs every run compiles do not fill the cache unless a caller asks for them.
#
# This is process-wide JAX state, so it is a documented side effect of the import.
# Override the location with ``AQUAFLUX_COMPILATION_CACHE_DIR`` (respecting
# ``XDG_CACHE_HOME``), or disable it entirely with ``AQUAFLUX_DISABLE_COMPILATION_CACHE=1``
# (e.g. on a read-only or ephemeral filesystem); set its size bound in gibibytes with
# ``AQUAFLUX_COMPILATION_CACHE_MAX_GIB`` (negative for no bound), and the shortest
# compilation it stores in seconds with ``AQUAFLUX_COMPILATION_CACHE_MIN_COMPILE_SECS``.
# A failure to configure the cache (an unwritable path) is non-fatal: the import
# proceeds, only without caching.

# Bound the cache so it cannot grow without limit. JAX's default
# ``jax_compilation_cache_max_size`` is ``-1``, which its own least-recently-used (LRU)
# implementation reads as "no eviction" -- so an unbounded cache never prunes anything and
# only ever grows. A long-lived checkout accumulated tens of gigabytes of month-old entries
# this way, dominated by a handful of ~1 GiB coupled-solve programs. Setting a byte bound
# turns on real LRU eviction, which is what keeps the cache a cache.
_DEFAULT_CACHE_MAX_BYTES = 10 * 1024**3
_CACHE_UNBOUNDED = -1


def _cache_max_bytes(raw: str | None) -> int:
    """Return the cache size bound in bytes for a ``...MAX_GIB`` environment value.

    Parameters
    ----------
    raw : str or None
        The raw environment value, in gibibytes. ``None`` or empty selects the default
        bound; a NEGATIVE value selects no bound (JAX's ``-1``); an unparseable value
        falls back to the default, because a typo in an environment variable must not
        decide how much disk this cache is allowed to use. ``0`` is taken literally, as a
        bound of zero bytes -- it is deliberately not folded into "no bound", since a
        reader who writes ``0`` means less caching, and the opposite reading would hand
        them the unbounded growth this bound exists to prevent.

    Returns
    -------
    int
        The bound in bytes, or ``-1`` for unbounded.
    """
    if raw is None or not raw.strip():
        return _DEFAULT_CACHE_MAX_BYTES
    try:
        gib = float(raw)
    except ValueError:
        return _DEFAULT_CACHE_MAX_BYTES
    return _CACHE_UNBOUNDED if gib < 0 else int(gib * 1024**3)


# Store only compilations of at least two seconds: the solve programs, not the several hundred
# small programs a solve compiles op by op before its first step (one per primitive, shape and
# dtype, 10-40 ms each), which are therefore recompiled on every run, warm or cold.
#
# A case that is run repeatedly on one mesh can opt in to keeping those too, by setting
# ``AQUAFLUX_COMPILATION_CACHE_MIN_COMPILE_SECS=0`` for that run. Measured 2026-10-17 (jax 0.11.2,
# CPU, 4-core Linux, one run per arm, process wall clock), a warm start at floor 2.0 against 0:
# pitzDaily two steps 89 s -> 63 s (977 compilations, 26 s, against none: 983 cache hits), the
# re = 20000 turbulent channel 87 s -> 65 s; cold starts 132 s -> 141 s and 101 s -> 104 s.
#
# It is not the default because of what it costs on writes: with a size bound set, JAX rescans
# the whole cache directory on every write to choose what to evict, ~30 microseconds per entry.
# A small program's key does not depend on the code that called it, so the entries grow with
# each new mesh or case (pitzDaily leaves 949, the channel 590, sharing 48), and nothing but the
# byte bound limits their count. A warm run writes nothing, so a repeated case pays once; a run
# on a new mesh pays a write per program at the current count, and gains nothing. Set it per
# run rather than in a shell profile, or the test suite stores its small programs too.
_DEFAULT_CACHE_FLOOR_SECS = 2.0


def _cache_floor_secs(raw: str | None) -> float:
    """Return the persistence floor in seconds for a ``...MIN_COMPILE_SECS`` environment value.

    Parameters
    ----------
    raw : str or None
        The raw environment value, in seconds. ``None``, empty or unparseable selects the
        default; a negative value is read as zero (store everything).

    Returns
    -------
    float
        The shortest compile time, in seconds, that JAX writes to the persistent cache.
    """
    if raw is None or not raw.strip():
        return _DEFAULT_CACHE_FLOOR_SECS
    try:
        secs = float(raw)
    except ValueError:
        return _DEFAULT_CACHE_FLOOR_SECS
    return max(secs, 0.0)


def _cache_eviction_is_supported() -> bool:
    """Report whether JAX can enforce a cache size bound in this environment.

    JAX's bounded cache takes an inter-process lock through ``filelock``. That package is a
    declared dependency, so this is normally true; it is checked rather than assumed because
    of how the failure presents (see :func:`_enable_compilation_cache`).

    Returns
    -------
    bool
        True when ``filelock`` can be imported.
    """
    return _importlib_util.find_spec("filelock") is not None


def _enable_compilation_cache() -> None:
    """Point JAX's persistent compilation cache at a bounded on-disk location."""
    if _os.environ.get("AQUAFLUX_DISABLE_COMPILATION_CACHE"):
        return
    cache_dir = _os.environ.get("AQUAFLUX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        base = _os.environ.get("XDG_CACHE_HOME") or (_Path.home() / ".cache")
        cache_dir = str(_Path(base) / "aquaflux" / "jax")
    try:
        _Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", cache_dir)
        # The floor: see ``_DEFAULT_CACHE_FLOOR_SECS``.
        _jax.config.update(
            "jax_persistent_cache_min_compile_time_secs",
            _cache_floor_secs(_os.environ.get("AQUAFLUX_COMPILATION_CACHE_MIN_COMPILE_SECS")),
        )
        # ⚠️ SETTING THE BOUND WITHOUT ``filelock`` DISABLES THE CACHE ENTIRELY, and does it
        # quietly: every read and write then fails with a ``UserWarning`` naming the missing
        # package, the cache stores nothing at all, and the only visible symptom is warning
        # noise during compilation. That is strictly worse than an unbounded cache, so an
        # environment that cannot evict is left merely unbounded rather than silently dead.
        if _cache_eviction_is_supported():
            bound = _cache_max_bytes(_os.environ.get("AQUAFLUX_COMPILATION_CACHE_MAX_GIB"))
            _jax.config.update("jax_compilation_cache_max_size", bound)
    except Exception:
        # Caching is an optimization, so no failure configuring it may break the import --
        # an unwritable path, a read-only filesystem, or a JAX release that renames one of
        # these settings. Deliberately broad for that reason.
        pass


_enable_compilation_cache()

__version__ = "0.0.0"

__all__ = ["__version__"]

# NOTE: the root package deliberately re-exports nothing but ``__version__``; import
# from the subpackages instead (``aquaflux.mesh``, ``aquaflux.discretization``,
# ``aquaflux.solve``, ...). Any name exported here must keep this file's import order
# compatible with the x64 enablement above.
