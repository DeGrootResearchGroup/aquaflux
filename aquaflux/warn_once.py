"""Warnings emitted once per process for each kind, rather than once per call.

A diagnostic about a fixed property of a mesh and its conditions -- an under-resolved sweep count, an
ill-conditioned stencil, a repaired closure -- is raised wherever a scheme is bound or a gradient is
reconstructed, which can be once per field and many times per Newton step. Repeated, it buries itself
and anything printed beside it, so it is said once.

The record is kept **per kind**, not as one flag for everything: a first, routine report must not
silence a later, graver one. Python's own ``"once"`` warning filter is no substitute, since it keys on
the message text and these messages carry the numbers that differ from one binding to the next.

Imports nothing from the package, so any layer may use it.
"""

from __future__ import annotations

import warnings

__all__ = ["forget_warnings", "warn_once"]

#: The kinds already warned about in this process.
_WARNED: set[str] = set()


def warn_once(
    kind: str,
    message: str,
    *,
    category: type[Warning] = UserWarning,
    stacklevel: int = 1,
) -> None:
    """Emit ``message`` the first time a warning of this ``kind`` is raised in this process.

    Parameters
    ----------
    kind : str
        What the warning is about. Every caller in the process shares one record, so name the kind
        after its module (``"multiple_correction.repaired"``) to keep two modules' kinds apart.
    message : str
        The warning text.
    category : type of Warning, optional
        The warning's class (default :class:`UserWarning`).
    stacklevel : int, optional
        As for :func:`warnings.warn`, counted from the code that calls this function: ``1`` (the
        default) attributes the warning to that call, ``2`` to its caller.

    Examples
    --------
    >>> import warnings
    >>> from aquaflux.warn_once import forget_warnings, warn_once
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warn_once("example.kind", "said once")
    ...     warn_once("example.kind", "said once")
    >>> len(caught)
    1
    >>> forget_warnings("example.kind")
    """
    if kind in _WARNED:
        return
    _WARNED.add(kind)
    warnings.warn(message, category, stacklevel=stacklevel + 1)


def forget_warnings(*kinds: str) -> None:
    """Let the given kinds be warned about again, or every kind when none is given.

    For tests, which need a warning's first emission to observe it, and for a long-lived process that
    starts on a different mesh and wants its diagnostics heard afresh.

    Parameters
    ----------
    *kinds : str
        The kinds to forget; none forgets them all.
    """
    if kinds:
        _WARNED.difference_update(kinds)
    else:
        _WARNED.clear()
