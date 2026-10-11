"""The files a case names, found wherever they sit in it, so they are checked and moved together.

A value that names a file says which of its fields are paths, in a ``path_fields`` class attribute;
every such path is relative to the case file unless absolute. :func:`named_paths` finds them all in a
case, at any depth -- a mesh, a lamp's photometry, an STL surface on one wall -- and
:func:`with_paths` rewrites them; :func:`relocated` uses it to re-base them all, which is how a copy of
a case written elsewhere keeps pointing at the same files.
"""

from __future__ import annotations

import dataclasses
import os
import types
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

__all__ = ["named_paths", "relocated", "with_paths"]


def named_paths(value: object, where: str = "") -> Iterator[tuple[str, str]]:
    """Every path ``value`` names, with where in it the path sits.

    Parameters
    ----------
    value : object
        A case, or any part of one.
    where : str, optional
        The location of ``value`` itself, prefixed to every location found.

    Yields
    ------
    (str, str)
        The location -- ``boundaries.lamp.profile.file``, say -- and the path as written.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        paths = getattr(type(value), "path_fields", ())
        for field in dataclasses.fields(value):
            child = getattr(value, field.name)
            location = f"{where}.{field.name}" if where else field.name
            if field.name in paths and isinstance(child, str):
                yield location, child
            else:
                yield from named_paths(child, location)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from named_paths(child, f"{where}.{key}" if where else str(key))
    elif isinstance(value, tuple):
        for index, child in enumerate(value):
            yield from named_paths(child, f"{where}[{index}]")


def with_paths(value: object, rewrite: Callable[[str], str]) -> object:
    """``value`` with every path it names rewritten by ``rewrite``, and nothing else changed.

    A part holding no path is returned as it is, not rebuilt, so values with constructors of their
    own pass through untouched.

    Parameters
    ----------
    value : object
        A case, or any part of one.
    rewrite : callable
        ``path -> path``.

    Returns
    -------
    object
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        paths = getattr(type(value), "path_fields", ())
        changes = {}
        for field in dataclasses.fields(value):
            child = getattr(value, field.name)
            if field.name in paths and isinstance(child, str):
                changes[field.name] = rewrite(child)
            else:
                rewritten = with_paths(child, rewrite)
                if rewritten is not child:
                    changes[field.name] = rewritten
        return dataclasses.replace(value, **changes) if changes else value
    if isinstance(value, Mapping):
        rewritten = {key: with_paths(child, rewrite) for key, child in value.items()}
        if all(rewritten[key] is child for key, child in value.items()):
            return value
        return types.MappingProxyType(rewritten)
    if isinstance(value, tuple):
        rewritten = tuple(with_paths(child, rewrite) for child in value)
        if all(new is old for new, old in zip(rewritten, value, strict=True)):
            return value
        return rewritten
    return value


def relocated(value: object, source: Path, target: Path) -> object:
    """``value`` for a case file moved from directory ``source`` to ``target``, so it means the same.

    Every relative path it names is re-based onto ``target``; an absolute one is left as it is. The
    output directory names no file and is not re-based: it is where *this* case file's runs write,
    beside it.

    Parameters
    ----------
    value : object
        A case, or any part of one.
    source, target : pathlib.Path
        The directory the case file is in, and the one it is moving to.

    Returns
    -------
    object
        ``value`` with each relative path it names re-based onto ``target``.
    """

    def rebased(path: str) -> str:
        return path if os.path.isabs(path) else os.path.relpath(Path(source) / path, Path(target))

    return with_paths(value, rebased)
