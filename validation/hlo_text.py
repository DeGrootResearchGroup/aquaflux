"""Reading XLA's after-optimization HLO text and its buffer dumps, for the harnesses that read compiled programs.

Imports nothing but the standard library, so a harness can import it before JAX initializes (it must set
``XLA_FLAGS`` first to get the dumps at all).

* :func:`arrays` / :func:`shape_bytes` -- the arrays an HLO shape string names, and their bytes;
* :func:`instructions` -- every instruction of every computation, with its shape, opcode, the
  computation it is in, and the ``op_name`` / ``stack_frame_id`` metadata JAX attaches;
* :func:`stack_frames` -- the module header's ``FileNames`` / ``FunctionNames`` / ``FileLocations`` /
  ``StackFrames`` tables, so a ``stack_frame_id`` resolves to the Python call chain that emitted it;
* :func:`allocations` -- the ``*-buffer-assignment.txt`` dump: each allocation's size and kind
  (``parameter``, ``output``, ``constant``, ``thread-local``, ``preallocated-temp``) and the values placed
  in it with their offsets;
* :func:`live_ranges` -- the ``*-live-range.txt`` dump: each value's interval in the flattened schedule
  (a ``while`` body's instructions are numbered inside the loop's own range).
"""

from __future__ import annotations

import re

BYTES = {"f64": 8, "s64": 8, "u64": 8, "f32": 4, "s32": 4, "u32": 4, "pred": 1, "s8": 1, "u8": 1}
ARRAY = re.compile(r"\b(f64|f32|s64|s32|u64|u32|pred|s8|u8)\[([0-9,]*)\]")
INSTRUCTION = re.compile(r"^\s*(?:ROOT )?%(\S+) = (.*?) ([a-z][a-z0-9\-]*)\((.*)$")
COMPUTATION = re.compile(r"^%(\S+) .*\{$")
_OP_NAME = re.compile(r'op_name="([^"]*)"')
_FRAME_ID = re.compile(r"stack_frame_id=(\d+)")
_TABLE_ROW = re.compile(r"^(\d+) (.*)$")
_FIELD = re.compile(r"(\w+)=(\d+)")
_ALLOCATION = re.compile(r"^allocation (\d+): size (\d+), (.*)$")
_VALUE = re.compile(
    r"^ value: <\d+ (\S+?)(\{[\d,]*\})?(?: \(phi\))? @\d+> \(size=(\d+),offset=(\d+)\): (.+)$"
)
_RANGE = re.compile(r"^    (\S+?)(\{[\d,]*\}):(\d+)-(\d+)$")


def arrays(text):
    """``(dtype, n_elements)`` for each array shape written in ``text``."""
    out = []
    for dtype, dims in ARRAY.findall(text):
        n = 1
        for d in filter(None, dims.split(",")):
            n *= int(d)
        out.append((dtype, n))
    return out


def shape_bytes(text):
    """The bytes of every array shape written in ``text``."""
    return sum(BYTES[dtype] * n for dtype, n in arrays(text))


def instructions(hlo):
    """``{name: dict(computation, shape, opcode, rest, op_name, frame)}`` over every computation."""
    out, current = {}, None
    for line in hlo.splitlines():
        if line.startswith("ENTRY"):
            current = "ENTRY"
            continue
        heading = COMPUTATION.match(line)
        if heading:
            current = heading.group(1)
            continue
        m = INSTRUCTION.match(line)
        if not m or current is None:
            continue
        name, shape, opcode, rest = m.groups()
        op_name = _OP_NAME.search(rest)
        frame = _FRAME_ID.search(rest)
        out[name] = {
            "computation": current,
            "shape": shape.split("{")[0] if not shape.startswith("(") else shape,
            "opcode": opcode,
            "rest": rest,
            "op_name": op_name.group(1) if op_name else "",
            "frame": int(frame.group(1)) if frame else 0,
        }
    return out


def stack_frames(hlo):
    """``frame id -> [(file, function, line), ...]``, innermost first, from the module header's tables.

    A frame's ``parent_frame_id`` is printed one above the parent's own id, with ``0`` (printed ``1`` on a
    root frame, which then names itself) meaning none; the chain is followed to its root.
    """
    tables, table = {}, None
    for line in hlo.splitlines():
        if line in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            table = tables.setdefault(line, {})
            continue
        if table is None:
            continue
        if line.startswith(("%", "ENTRY")):
            break
        row = _TABLE_ROW.match(line)
        if row:
            table[int(row.group(1))] = row.group(2)
    files = {k: v.strip('"') for k, v in tables.get("FileNames", {}).items()}
    functions = {k: v.strip('"') for k, v in tables.get("FunctionNames", {}).items()}
    locations = {}
    for k, v in tables.get("FileLocations", {}).items():
        f = dict((a, int(b)) for a, b in _FIELD.findall(v))
        locations[k] = (
            files.get(f.get("file_name_id"), "?"),
            functions.get(f.get("function_name_id"), "?"),
            f.get("line", 0),
        )
    raw = {}
    for k, v in tables.get("StackFrames", {}).items():
        f = dict((a, int(b)) for a, b in _FIELD.findall(v))
        raw[k] = (f.get("file_location_id"), f.get("parent_frame_id", 0) - 1)
    chains = {}
    for k in raw:
        chain, current, seen = [], k, set()
        while current in raw and current not in seen:
            seen.add(current)
            location, parent = raw[current]
            chain.append(locations.get(location, ("?", "?", 0)))
            current = parent
        chains[k] = chain
    return chains


def allocations(text):
    """The buffer assignment: ``[dict(index, size, kind, values=[(name, size, offset, shape)])]``.

    A value's ``name`` carries its tuple index as the live-range dump writes it (``{}`` for an array).
    """
    out = []
    for line in text.splitlines():
        a = _ALLOCATION.match(line)
        if a:
            out.append(
                {
                    "index": int(a.group(1)),
                    "size": int(a.group(2)),
                    "kind": a.group(3),
                    "values": [],
                }
            )
            continue
        v = _VALUE.match(line)
        if v and out:
            name, index, size, offset, shape = v.groups()
            out[-1]["values"].append((name + (index or "{}"), int(size), int(offset), shape))
    return out


def live_ranges(text):
    """``{value name with its tuple index: (start, end)}`` from the live-range dump."""
    out = {}
    for line in text.splitlines():
        m = _RANGE.match(line)
        if m:
            out[m.group(1) + m.group(2)] = (int(m.group(3)), int(m.group(4)))
    return out


_CALLED = re.compile(r"(condition|body|branch_computations|to_apply|calls)=(\{[^}]*\}|%[^,\s]+)")


def loop_depths(instruction_map):
    """``{computation: number of while loops enclosing it}``, ``ENTRY`` at 0.

    Followed through ``while`` (its body and condition one loop deeper), ``conditional`` branches and
    ``call`` (at the caller's depth). A fused computation is not listed: its instructions own no buffer.
    """
    children = {}
    for info in instruction_map.values():
        if info["opcode"] not in ("while", "conditional", "call"):
            continue
        for kind, called in _CALLED.findall(info["rest"]):
            for name in re.findall(r"%([^,\s}]+)", called):
                children.setdefault(info["computation"], []).append(
                    (name, kind in ("body", "condition"))
                )
    depths, stack = {"ENTRY": 0}, ["ENTRY"]
    while stack:
        parent = stack.pop()
        for child, deeper in children.get(parent, []):
            if child not in depths:
                depths[child] = depths[parent] + int(deeper)
                stack.append(child)
    return depths
