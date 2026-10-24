"""Returning memory the C allocator has kept to the operating system.

When a process frees memory, glibc's ``malloc`` keeps much of it in its arenas for reuse rather than
giving it back to the kernel, so the resident set stays high after the arrays that needed it are gone.
XLA allocates on worker threads, each with its own arena, so a program run once can leave hundreds of
megabytes of freed memory resident. ``malloc_trim(0)`` releases what every arena can give back; it
changes no value a computation holds, only how much freed memory stays resident.

Imports nothing from the package, so any layer may use it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Protocol

__all__ = ["GlibcTrim", "MemoryTrim"]


class MemoryTrim(Protocol):
    """Gives the allocator's kept, freed memory back to the operating system when called."""

    def __call__(self) -> None:
        """Release what can be released; never changes a value the process holds."""
        ...


class GlibcTrim:
    """A :class:`MemoryTrim` calling glibc's ``malloc_trim(0)``, and doing nothing without glibc.

    The C library is looked up on the first call, not at construction, so building one costs nothing
    where it is never called. On a platform whose C library has no ``malloc_trim`` (macOS, musl) every
    call does nothing.

    Parameters
    ----------
    load_library : callable, optional
        ``() -> library or None``, the C library to call ``malloc_trim`` on. ``None`` (default) loads
        the process's C library through :mod:`ctypes`.

    Attributes
    ----------
    released : bool or None
        Whether the last call gave memory back (``malloc_trim``'s own report); ``None`` before the
        first call or where there is nothing to call.
    """

    def __init__(self, load_library=None) -> None:
        self._load_library = _c_library if load_library is None else load_library
        self._trim = None
        self._looked_up = False
        self.released: bool | None = None

    def __call__(self) -> None:
        if not self._looked_up:
            self._looked_up = True
            library = self._load_library()
            self._trim = getattr(library, "malloc_trim", None) if library is not None else None
        if self._trim is not None:
            self.released = bool(self._trim(0))


def _c_library():
    """The process's C library, or ``None`` where :mod:`ctypes` cannot find one."""
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    try:
        return ctypes.CDLL(name)
    except OSError:
        return None
