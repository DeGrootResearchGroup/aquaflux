"""Shared pytest fixtures and configuration for the aquaflux test suite."""

from __future__ import annotations

import os
from pathlib import Path

import aquaflux
import jax
import pytest

#: Where the test suite keeps its persistent compilation cache, and the shortest compile it stores.
TEST_CACHE_DIR = aquaflux._cache_root() / "jax-tests"
TEST_CACHE_FLOOR_SECS = 2.0


def _point_the_compilation_cache_at_the_tests_own_directory() -> None:
    """Give the test suite its own persistent compilation cache, apart from the solver's.

    ``import aquaflux`` points JAX's on-disk cache at the solver's directory and persists every
    compilation, however short, so that a solver's warm start skips the several hundred small
    programs it otherwise recompiles op by op. The test suite must not write there: it compiles
    thousands of small programs of its own per run, and a bounded cache rescans its whole
    directory on every write, so those entries would make every later solver write dearer and
    compete with the solver's programs for the size bound. The tests keep the older behaviour
    in a directory of their own: only compilations of at least ``TEST_CACHE_FLOOR_SECS`` are
    stored, so the expensive solve programs of the heavier tiers are still reused across runs.

    Measured on six fast-tier files (274 tests, serial, jax 0.11.2, 4-core Linux, 2026-10-17, one
    run each): cache off 156 s; this directory at 2.0, 155 s cold and 166 s warm (nothing in those
    files compiles for 2 s, so it stores nothing); at floor 0, 326 s cold and 59 s warm, with 3099
    entries. The cold run doubles because each write rescans the growing directory, and a library
    change re-keys the tests' own compiled programs, so most runs would be partly cold. Off would
    cost the slow tiers their reuse of multi-second solve programs, which this keeps.

    It runs here, at import of this file, which is before any test compiles anything. It is set
    in the environment as well as applied to this process, so that the interpreters some tests
    spawn configure the same cache when they import aquaflux. An explicit setting of either
    variable by the caller is kept.
    """
    os.environ.setdefault("AQUAFLUX_COMPILATION_CACHE_DIR", str(TEST_CACHE_DIR))
    os.environ.setdefault("AQUAFLUX_COMPILATION_CACHE_MIN_COMPILE_SECS", str(TEST_CACHE_FLOOR_SECS))
    aquaflux._enable_compilation_cache()


_point_the_compilation_cache_at_the_tests_own_directory()

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to the ``tests/fixtures`` directory."""
    return FIXTURES


@pytest.fixture(autouse=True, scope="module")
def _release_jax_compiled_programs_between_modules():
    """Drop every live-compiled XLA executable at the end of each test module.

    An xdist worker otherwise keeps every compiled program it has ever built for the lifetime of
    the process, across however many modules it draws (``--dist loadfile`` -- see
    ``tools/fastgate.sh``) -- so its real memory footprint only grows over a run and never comes
    back down. Measured on the fast tier (11 xdist workers, 11-core/19 GB machine, jax 0.10.2,
    2026-09-14, footprint read the same way as the #379 postmortem:
    ``top -l 1 -o mem -stats pid,ppid,mem,cmprs``, summed over the worker pool every 8 s): this
    fixture took the tier's peak combined footprint from 54.3 GB to 22.7 GB and its wall clock from
    467 s to 412 s -- both improve, because the peak run was also paying to keep 30+ GB of that
    footprint compressed. The corresponding recompiles cost less than that saves.
    """
    yield
    jax.clear_caches()
