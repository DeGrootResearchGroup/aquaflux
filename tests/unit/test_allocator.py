"""``GlibcTrim``: calls the C library's ``malloc_trim(0)`` where there is one, and nothing where not."""

from __future__ import annotations

import platform
from types import SimpleNamespace

import pytest
from aquaflux.allocator import GlibcTrim


def test_the_trim_asks_the_library_to_release_everything_and_reports_its_answer() -> None:
    """``malloc_trim``'s argument is the padding kept at the heap's top: ``0`` keeps none."""
    asked: list[int] = []
    loads: list[int] = []

    def malloc_trim(pad: int) -> int:
        asked.append(pad)
        return 1

    def load():
        loads.append(1)
        return SimpleNamespace(malloc_trim=malloc_trim)

    trim = GlibcTrim(load_library=load)
    assert trim.released is None
    trim()
    trim()
    assert asked == [0, 0]
    assert trim.released is True
    assert loads == [1]  # looked up once, on the first call


@pytest.mark.parametrize("library", [None, SimpleNamespace()], ids=["no library", "no malloc_trim"])
def test_without_malloc_trim_the_trim_does_nothing(library) -> None:
    """macOS and musl have no ``malloc_trim``: the trim must do nothing there, not raise."""
    trim = GlibcTrim(load_library=lambda: library)
    trim()
    assert trim.released is None


def test_the_default_lookup_finds_the_trim_exactly_where_glibc_is_the_c_library() -> None:
    """A missed symbol makes the trim a silent no-op where it matters; a found one elsewhere would call
    something else by that name."""
    trim = GlibcTrim()
    trim()
    assert (trim.released is not None) == (platform.libc_ver()[0] == "glibc")
