"""The OpenFOAM file envelope: comment stripping and the ``FoamFile`` header.

Every file in a polyMesh directory shares the same wrapper — an optional banner comment, a
``FoamFile { … }`` header dictionary, then the payload (a points list, a face list, a boundary
dictionary, …). This module owns that wrapper so each payload parser (:mod:`.grammar`) sees only
its own list body, and so the header's ``format`` entry can gate ASCII vs binary in one place.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# The FoamFile header dictionary: ``FoamFile { key value; ... }`` (no nested braces).
_HEADER_RE = re.compile(r"FoamFile\s*\{(?P<entries>[^{}]*)\}", re.DOTALL)
# A ``key value;`` entry inside a flat dictionary (value runs to the semicolon).
_ENTRY_RE = re.compile(r"(\w+)\s+([^;{}]+);")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


class FoamFile(NamedTuple):
    """A comment-stripped OpenFOAM file split into its header dictionary and payload body.

    Attributes
    ----------
    header : dict of {str: str}
        The ``FoamFile`` header entries (``version``, ``format``, ``class``, ``object``, …).
    body : str
        Everything after the header — the payload the :mod:`.grammar` parsers consume.
    """

    header: dict[str, str]
    body: str


def strip_comments(text: str) -> str:
    """Remove ``/* … */`` block comments and ``// …`` line comments, replacing each with a space."""
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    return _LINE_COMMENT_RE.sub(" ", text)


def parse_dict_entries(block: str) -> dict[str, str]:
    """Parse a flat ``key value;`` dictionary body into a string-valued mapping.

    Parameters
    ----------
    block : str
        The text between a dictionary's braces (no nested dictionaries).

    Returns
    -------
    dict of {str: str}
        One entry per ``key value;`` pair; surrounding double quotes are stripped from the value.
    """
    return {key: value.strip().strip('"') for key, value in _ENTRY_RE.findall(block)}


def parse_foamfile(text: str) -> FoamFile:
    """Strip comments and split an OpenFOAM file into its header dictionary and payload body.

    Parameters
    ----------
    text : str
        The full text of one polyMesh file.

    Returns
    -------
    FoamFile
        The parsed header and the remaining body.

    Raises
    ------
    ValueError
        If the text has no ``FoamFile { … }`` header.
    """
    stripped = strip_comments(text)
    match = _HEADER_RE.search(stripped)
    if match is None:
        raise ValueError("not an OpenFOAM file: no 'FoamFile { ... }' header found")
    header = parse_dict_entries(match.group("entries"))
    return FoamFile(header=header, body=stripped[match.end() :])


def is_binary(foam: FoamFile) -> bool:
    """Whether the file declares ``format binary;`` (ASCII is the default when unset)."""
    return foam.header.get("format", "ascii").strip().lower() == "binary"
