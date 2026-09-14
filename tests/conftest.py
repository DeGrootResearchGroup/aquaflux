"""Shared pytest fixtures and configuration for the aquaflux test suite."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

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
