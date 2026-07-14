"""Shared pytest fixtures and configuration for the aquaflux test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to the ``tests/fixtures`` directory."""
    return FIXTURES
