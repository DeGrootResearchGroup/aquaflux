"""What a step case's resident memory holds beyond its live arrays, stage by stage, on either step case.

``bfs3d_openfoam/memory_attribution.py`` names every live array and glibc's free-but-retained bytes, and
leaves part of the resident set unnamed. This runs that harness's march and stages unchanged (on
``bfs3d`` or ``pitzdaily``, so the unnamed part can be read on two meshes and given a per-cell slope)
and adds, at each stage's end, readings that split the remainder by what could hold it:

* **readings** (``RESIDUE_READ=1``, no effect on the run beyond the time they take):

  - ``/proc/self/smaps`` mapping by mapping, each classed by size (``<1M``, ``1-64M``, ``arena heap``: a
    glibc thread arena's heap, 64 MiB-aligned and at most 64 MiB, ``>64M``; ``[heap]`` and stacks by name) and by whether a
    live array's buffer lies in it -- the numpy arrays reachable from Python and JAX's live arrays,
    located by their data pointers -- with the resident bytes of the mappings holding none (memory no
    Python object or array names: XLA's runtime, the compiler's structures, glibc's own retention);
  - glibc's arenas one by one (``malloc_info``): the main arena against the thread arenas XLA's
    worker threads allocate from, each arena's system bytes and its free bytes;
  - the Python objects ``gc.get_objects()`` returns, summed by type with ``sys.getsizeof`` less any
    array data (objects the collector does not track -- ints, strings -- are not seen; the traced
    total of ``memory_attribution.py`` includes them);
  - JAX's loaded executables (``client.live_executables()``): their count and generated code;
  - the process's thread count.

* **``RESIDUE_TRIM=1``**, an arm: ``malloc_trim(0)`` at every stage's end, reading the resident set
  and the anonymous resident bytes before and after -- what glibc retains, separated from what it holds.
  It changes the run's later peaks (memory given back is not re-faulted from a retained arena).

* **always**, a reading: the loaded executable count, and the compilations since the run started,
  through JAX's monitoring events -- programs traced (count, seconds), lowered (seconds), handed to the
  backend (count, seconds; a persistent-cache hit is one too, its retrieval inside the seconds) and
  persistent-cache hits and misses. Read with ``RESIDUE_CLEAR`` at an earlier stage, the counts at the
  later stages say what the clear made the run recompile, and the seconds what that cost.

* **``RESIDUE_HLO=<stage label>;...``**, a reading: the live executables ranked by their serialized
  HLO module's size (a host array a jitted function closes over is compiled in as a constant, so a
  program carrying data is about the data's size), with the resident, live array and traced bytes.
  ``RESIDUE_CLEAR`` takes the same ranking before it clears.

* **``RESIDUE_CLEAR=<stage label>;...``**, an arm: ``jax.clear_caches()`` and a ``gc.collect()`` at the
  end of each named stage (``end`` names the reading taken once the measured steps are done), then
  ``malloc_trim(0)``, reading the resident set and the executable count after each. Dropping the
  executables mid-run makes the next call recompile; ``end`` changes nothing measured after it but the
  programs' memory analyses, which compile again from the persistent cache in either case.

* **``RESIDUE_FUNCTIONS=<stage label>;...``**, a reading: every jitted function alive (each
  ``jax.jit`` / ``eqx.filter_jit`` wrapper the collector can reach, including ``jax.numpy``'s own: an
  operation run outside a jit on concrete arrays is a call of a jitted library function) with the number
  of programs its cache holds, summed by the module that defined the function and grouped into ``jax``
  (the library's own functions, the op-by-op programs), ``aquaflux`` and ``other`` (equinox, lineax,
  the case scripts). Executables held by no jitted function's cache are counted apart.

* **``RESIDUE_DROP=<stage label>=<group>[,<group>];...``**, an arm: at the end of the named stage, the
  caches of every jitted function of the named groups (as ``RESIDUE_FUNCTIONS`` groups them) are cleared
  -- that function's programs only, nothing else of JAX's -- then a ``gc.collect()`` and a
  ``malloc_trim(0)``, reading the resident set and the executables before and after. Read with
  ``RESIDUE_FUNCTIONS`` at later stages, it says which of the dropped functions' programs the run compiles
  again.

Each row's readings are written into ``memory_attribution.py``'s record (``RESIDUE_RECORD`` names it,
default ``memory_residue.<case>.json`` beside this file) under ``residue``, and printed under the
row. ``BFS3D_MEMORY_STEPS`` (default 3), ``BFS3D_MEMORY_CENSUS`` and ``BFS3D_MEMORY_COLLECT`` keep
their meaning for both cases. Run it through the case runner, alone, one step::

    RESIDUE_CASE=pitzdaily RESIDUE_READ=1 RESIDUE_TRIM=1 RESIDUE_CLEAR=end BFS3D_MEMORY_STEPS=1 \\
        BFS3D_MEMORY_CENSUS=0 validation/run_case.sh validation/memory_residue.py --wait

Linux only (``smaps``, glibc).
"""

from __future__ import annotations

import bisect
import ctypes
import gc
import importlib.util
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from step_cases import CASES, load_case  # noqa: E402

CASE = os.environ.get("RESIDUE_CASE", "bfs3d")
READ = os.environ.get("RESIDUE_READ", "0") == "1"
TRIM = os.environ.get("RESIDUE_TRIM", "0") == "1"
#: Stage labels (as the table prints them, without indentation; separated by `;`, since a label may
#: hold a comma) after which the caches are cleared.
CLEAR = {label.strip() for label in os.environ.get("RESIDUE_CLEAR", "").split(";") if label.strip()}
#: Stage labels after which the live executables are ranked by HLO size, without clearing anything.
HLO = {label.strip() for label in os.environ.get("RESIDUE_HLO", "").split(";") if label.strip()}
#: Stage labels after which the live jitted functions are counted by module.
FUNCTIONS = {
    label.strip() for label in os.environ.get("RESIDUE_FUNCTIONS", "").split(";") if label.strip()
}
#: Stage label -> the function groups whose caches are dropped at its end.
DROP = {
    label.strip(): {group.strip() for group in groups.split(",") if group.strip()}
    for label, _, groups in (
        entry.rpartition("=") for entry in os.environ.get("RESIDUE_DROP", "").split(";")
    )
    if label.strip()
}
RECORD = Path(os.environ.get("RESIDUE_RECORD", HERE / f"memory_residue.{CASE}.json"))
GIB = 2**30
MIB = 2**20
#: The size of a glibc thread arena's heap (``HEAP_MAX_SIZE`` on 64-bit Linux).
ARENA_HEAP = 64 * MIB

_LIBC = ctypes.CDLL("libc.so.6")
_LIBC.fopen.restype = ctypes.c_void_p
_LIBC.fopen.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
_LIBC.fclose.argtypes = (ctypes.c_void_p,)
_LIBC.malloc_info.argtypes = (ctypes.c_int, ctypes.c_void_p)


def _attribution():
    """``bfs3d_openfoam/memory_attribution.py``, loaded by path (it is not a package module)."""
    path = HERE / CASES["bfs3d"] / "memory_attribution.py"
    spec = importlib.util.spec_from_file_location("memory_attribution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["memory_attribution"] = module
    spec.loader.exec_module(module)
    return module


# --- readings -----------------------------------------------------------------------------------


def _resident() -> dict[str, int]:
    """The resident set and its anonymous part (``RssAnon``), bytes."""
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmRSS", "RssAnon", "Threads"):
            fields[name] = int(value.split()[0]) * (1 if name == "Threads" else 1024)
    return fields


def _mappings() -> list[dict]:
    """Every mapping of ``/proc/self/smaps``: start, end, name, permissions, resident bytes."""
    rows: list[dict] = []
    for line in Path("/proc/self/smaps").read_text().splitlines():
        head = line.split(None, 5)
        if len(head) >= 5 and "-" in head[0] and len(head[1]) == 4:
            start, end = (int(x, 16) for x in head[0].split("-"))
            name = head[5] if len(head) == 6 else ""
            rows.append({"start": start, "end": end, "perms": head[1], "name": name, "rss": 0})
        elif line.startswith("Rss:") and rows:
            rows[-1]["rss"] = int(line.split()[1]) * 1024
    return rows


def _buffers() -> list[tuple[int, int]]:
    """(address, bytes) of every live array buffer: numpy arrays reachable from Python, JAX arrays."""
    import jax
    import numpy as np

    seen: dict[int, int] = {}

    def add(array) -> None:
        root = array
        while isinstance(root.base, np.ndarray):
            root = root.base
        if root.nbytes:
            address = root.__array_interface__["data"][0]
            seen[address] = max(seen.get(address, 0), root.nbytes)

    for obj in gc.get_objects():
        if isinstance(obj, np.ndarray):
            add(obj)
        for child in gc.get_referents(obj):
            if type(child) is np.ndarray:
                add(child)
    for array in jax.live_arrays():
        try:
            for shard in array.addressable_shards:
                seen[shard.data.unsafe_buffer_pointer()] = shard.data.nbytes
        except Exception:  # a deleted or donated buffer: nothing resident to place
            continue
    return sorted(seen.items())


def _size_class(mapping: dict) -> str:
    name = mapping["name"]
    if name == "[heap]":
        return "[heap]"
    if name.startswith("[stack"):
        return "stack"
    if name and not name.startswith("["):
        return "file"
    size = mapping["end"] - mapping["start"]
    # A thread arena's heap is mapped 64 MiB-aligned and grown in place, so its resident part is a
    # mapping of any size up to 64 MiB starting on that alignment.
    if mapping["start"] % ARENA_HEAP == 0 and size <= ARENA_HEAP:
        return "arena heap"
    if size < MIB:
        return "<1M"
    if size <= ARENA_HEAP:
        return "1-64M"
    return ">64M"


def _smaps_by_owner() -> dict:
    """Resident bytes by mapping size class, split into mappings holding a live array buffer or not."""
    buffers = _buffers()
    starts = [address for address, _ in buffers]
    classes: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "rss": 0, "rss_without_arrays": 0, "array_bytes": 0}
    )
    for mapping in _mappings():
        if not mapping["rss"]:
            continue
        entry = classes[_size_class(mapping)]
        entry["count"] += 1
        entry["rss"] += mapping["rss"]
        lo = bisect.bisect_left(starts, mapping["start"])
        hi = bisect.bisect_left(starts, mapping["end"])
        inside = sum(
            min(address + size, mapping["end"]) - address for address, size in buffers[lo:hi]
        )
        # A buffer starting below the mapping and running into it.
        if lo and buffers[lo - 1][0] + buffers[lo - 1][1] > mapping["start"]:
            inside += (
                min(buffers[lo - 1][0] + buffers[lo - 1][1], mapping["end"]) - mapping["start"]
            )
        entry["array_bytes"] += inside
        if not inside:
            entry["rss_without_arrays"] += mapping["rss"]
    return {"classes": dict(classes), "array_buffers": sum(size for _, size in buffers)}


def _arenas() -> dict:
    """glibc's arenas from ``malloc_info``: the main arena and the thread arenas, system and free."""
    with tempfile.NamedTemporaryFile(suffix=".xml") as handle:
        stream = _LIBC.fopen(handle.name.encode(), b"w")
        _LIBC.malloc_info(0, stream)
        _LIBC.fclose(stream)
        text = Path(handle.name).read_text()
    arenas = []
    for body in re.findall(r'<heap nr="\d+">(.*?)</heap>', text, re.S):
        free = sum(
            int(size)
            for kind, size in re.findall(
                r'<total type="(fast|rest)" count="\d+" size="(\d+)"', body
            )
        )
        system = int(re.search(r'<system type="current" size="(\d+)"', body).group(1))
        arenas.append({"system": system, "free": free})
    mmapped = re.search(r'<total type="mmap" count="(\d+)" size="(\d+)"', text)
    main, threads = (arenas[0], arenas[1:]) if arenas else ({"system": 0, "free": 0}, [])
    return {
        "main": main,
        "thread_arenas": len(threads),
        "thread_system": sum(a["system"] for a in threads),
        "thread_free": sum(a["free"] for a in threads),
        "mmap_count": int(mmapped.group(1)) if mmapped else 0,
        "mmap_bytes": int(mmapped.group(2)) if mmapped else 0,
    }


def _objects_by_type(top: int = 12) -> dict:
    """The collector's objects summed by type (shallow size, array data left out), largest first."""
    import numpy as np

    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for obj in gc.get_objects():
        size = sys.getsizeof(obj)
        if isinstance(obj, np.ndarray) and obj.base is None:
            size -= obj.nbytes
        entry = totals[type(obj).__name__]
        entry[0] += 1
        entry[1] += size
    ranked = sorted(totals.items(), key=lambda kv: -kv[1][1])
    return {
        "objects": sum(c for c, _ in totals.values()),
        "bytes": sum(b for _, b in totals.values()),
        "top": [(name, count, size) for name, (count, size) in ranked[:top]],
    }


def _executables() -> dict:
    import jax

    live = jax.devices()[0].client.live_executables()
    code = 0
    for executable in live:
        try:
            code += executable.size_of_generated_code_in_bytes()
        except Exception:
            continue
    return {"count": len(live), "code": code}


def _executables_by_hlo(top: int = 15) -> dict:
    """The live executables ranked by their serialized HLO module's size.

    A jitted function that closes over a host array compiles it into its program as a constant, so its
    HLO is about the array's size; a program whose data all arrive as arguments is a few kilobytes.
    """
    import jax

    sizes = []
    for executable in jax.devices()[0].client.live_executables():
        try:
            for module in executable.hlo_modules():
                sizes.append((module.name, len(module.as_serialized_hlo_module_proto())))
        except Exception:
            continue
    sizes.sort(key=lambda named: -named[1])
    by_name: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, size in sizes:
        by_name[name][0] += 1
        by_name[name][1] += size
    return {
        "modules": len(sizes),
        "bytes": sum(size for _, size in sizes),
        "over_1_mib": sum(1 for _, size in sizes if size >= MIB),
        "bytes_over_1_mib": sum(size for _, size in sizes if size >= MIB),
        "top": sizes[:top],
        "top_names": sorted(
            ((name, count, size) for name, (count, size) in by_name.items()), key=lambda r: -r[2]
        )[:top],
    }


#: JAX's monitoring events counted by :func:`_count_compilations`, by the name the record uses.
_COMPILE_EVENTS = {
    "/jax/core/compile/jaxpr_trace_duration": "trace",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "lower",
    "/jax/core/compile/backend_compile_duration": "backend",
}
_CACHE_EVENTS = {
    "/jax/compilation_cache/cache_hits": "cache_hits",
    "/jax/compilation_cache/cache_misses": "cache_misses",
}
_compilations: dict[str, float] = defaultdict(float)


def _count_compilations() -> None:
    """Count every trace, lowering and backend compilation (and its seconds) from here on."""
    import jax

    def on_duration(event: str, seconds: float, **_) -> None:
        name = _COMPILE_EVENTS.get(event)
        if name is not None:
            _compilations[name] += 1
            _compilations[f"{name}_seconds"] += seconds

    def on_event(event: str, **_) -> None:
        name = _CACHE_EVENTS.get(event)
        if name is not None:
            _compilations[name] += 1

    jax.monitoring.register_event_duration_secs_listener(on_duration)
    jax.monitoring.register_event_listener(on_event)


def _modules_by_qualname() -> dict[str, str]:
    """The defining module of every function and method of the loaded aquaflux modules, by qualname."""
    import types

    index: dict[str, str] = {}

    def add(obj, module: str) -> None:
        function = getattr(obj, "__func__", obj)
        function = getattr(function, "fget", function)
        if isinstance(function, types.FunctionType) and function.__module__ == module:
            index.setdefault(function.__qualname__, module)

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("aquaflux") or module is None:
            continue
        for value in list(vars(module).values()):
            if isinstance(value, type) and value.__module__ == module_name:
                for member in vars(value).values():
                    add(member, module_name)
            else:
                add(value, module_name)
    return index


def _defining_module(qualname: str, index: dict[str, str]) -> str | None:
    """The module defining ``qualname``, or the function it is a closure of."""
    while qualname:
        if qualname in index:
            return index[qualname]
        qualname = qualname.rpartition(".<locals>")[0]
    return None


def _jitted_functions() -> list[tuple[str, str, object]]:
    """(group, defining module, the jit wrapper) for every jitted function the collector reaches.

    An ``eqx.filter_jit`` wrapper's inner ``jax.jit`` wraps a function equinox defines, named after the
    user's function but not carrying its module: that is read from a live ``filter_jit`` wrapper holding
    it, or else looked up by qualname among the loaded aquaflux modules (closures by their enclosing
    function); a function found in neither is grouped ``other`` under equinox.
    """
    index = _modules_by_qualname()
    owners: dict[int, str] = {}
    wrappers = []
    for obj in gc.get_objects():
        name = type(obj).__name__
        if name == "PjitFunction":
            wrappers.append(obj)
        elif name == "_JitWrapper":
            cached = getattr(obj, "_cached", None)
            module = getattr(getattr(obj, "__wrapped__", None), "__module__", None)
            if cached is not None and module:
                owners[id(cached)] = module
    rows = []
    for wrapper in wrappers:
        module = owners.get(id(wrapper)) or getattr(
            getattr(wrapper, "__wrapped__", None), "__module__", None
        )
        if module == "equinox._jit":
            module = _defining_module(getattr(wrapper, "__qualname__", ""), index) or module
        module = module or getattr(wrapper, "__module__", None) or "?"
        if module.startswith(("jax", "jaxlib")):
            group = "jax"
        elif module.startswith("aquaflux"):
            group = "aquaflux"
        else:
            group = "other"
        rows.append((group, module, wrapper))
    return rows


def _functions(top: int = 25) -> dict:
    """The live jitted functions' cached programs, by group and by defining module and name."""
    import jax

    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_name: dict[tuple[str, str], int] = defaultdict(int)
    held = 0
    for group, module, wrapper in _jitted_functions():
        size = wrapper._cache_size()
        if not size:
            continue
        held += size
        by_group[group][0] += 1
        by_group[group][1] += size
        name = getattr(wrapper, "__qualname__", None) or getattr(wrapper, "__name__", "?")
        by_name[(f"{group}:{module}", name)] += size
    executables = len(jax.devices()[0].client.live_executables())
    return {
        "executables": executables,
        "held_by_functions": held,
        "by_group": {g: {"functions": f, "programs": n} for g, (f, n) in by_group.items()},
        "by_module": _summed_by_module(by_name),
        "top": sorted(((m, n, s) for (m, n), s in by_name.items()), key=lambda r: -r[2])[:top],
    }


def _summed_by_module(by_name: dict[tuple[str, str], int]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for (module, _), size in by_name.items():
        totals[module] += size
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def _drop(groups: set[str]) -> dict:
    """Clear the caches of the jitted functions in ``groups``, then collect and trim; read before and after."""
    import jax

    def count() -> int:
        return len(jax.devices()[0].client.live_executables())

    gc.collect()
    _LIBC.malloc_trim(0)
    before = {**_live(), "executables": count()}
    dropped = 0
    for group, _, wrapper in _jitted_functions():
        if group in groups:
            dropped += wrapper._cache_size()
            wrapper.clear_cache()
    gc.collect()
    _LIBC.malloc_trim(0)
    after = {**_live(), "executables": count()}
    return {"groups": sorted(groups), "programs_dropped": dropped, "before": before, "after": after}


def _trim() -> dict:
    before = _resident()
    _LIBC.malloc_trim(0)
    after = _resident()
    return {
        "rss_before": before["VmRSS"],
        "rss_after": after["VmRSS"],
        "anon_after": after["RssAnon"],
    }


def _live() -> dict[str, int]:
    """Resident, live JAX array and traced host bytes now."""
    import tracemalloc

    import jax

    return {
        "rss": _resident()["VmRSS"],
        "device": sum(a.nbytes for a in jax.live_arrays()),
        "traced": tracemalloc.get_traced_memory()[0],
    }


def _clear() -> dict:
    """A collection and a trim, then the caches cleared, collected and trimmed, read after each.

    The first reading separates what the collector frees from what only the caches held; the live
    array bytes after the clear say how much of what it freed was arrays held by a cache (a jitted
    function's captured constants) rather than compiled programs.
    """
    import jax

    before = _live()
    executables = _executables()["count"]
    hlo = _executables_by_hlo()
    gc.collect()
    _LIBC.malloc_trim(0)
    collected = _live()
    jax.clear_caches()
    gc.collect()
    cleared = _live()
    _LIBC.malloc_trim(0)
    trimmed = _live()
    after = {"executables": _executables()["count"]}
    if READ:
        after["smaps"] = _smaps_by_owner()
        after["arenas"] = _arenas()
        after["python"] = _objects_by_type()
    return {
        "executables_before": executables,
        "hlo": hlo,
        "before": before,
        "after_collect_and_trim": collected,
        "after_clear": cleared,
        "after_clear_and_trim": trimmed,
        "after": after,
    }


def reading(row: dict) -> None:
    """The residue readings at one stage's end, added to its row and printed under it."""
    label = row["label"]
    t0 = time.perf_counter()
    import jax

    residue: dict = {
        "resident": _resident(),
        "executable_count": len(jax.devices()[0].client.live_executables()),
        "compilations": dict(_compilations),
    }
    if READ:
        residue["smaps"] = _smaps_by_owner()
        residue["arenas"] = _arenas()
        residue["python"] = _objects_by_type()
        residue["executables"] = _executables()
    if TRIM:
        residue["trim"] = _trim()
    if label in HLO:
        residue["hlo"] = {"live": _live(), **_executables_by_hlo()}
    if label in FUNCTIONS:
        residue["functions"] = _functions()
    if label in DROP:
        residue["drop"] = _drop(DROP[label])
        if label in FUNCTIONS:
            residue["functions_after_drop"] = _functions()
    if label in CLEAR:
        residue["clear"] = _clear()
    residue["seconds"] = time.perf_counter() - t0
    row["residue"] = residue
    _print(label, residue)


def _print_reading(title: str, reading: dict) -> None:
    """The smaps classes, the arenas and the Python objects of one reading."""
    if title:
        print(f"      {title}:", flush=True)
    smaps = reading["smaps"]
    for name, entry in sorted(smaps["classes"].items(), key=lambda kv: -kv[1]["rss"]):
        print(
            f"      smaps {name:9s} x{entry['count']:<5d} rss {entry['rss'] / GIB:6.3f} "
            f"(no array {entry['rss_without_arrays'] / GIB:6.3f}) arrays "
            f"{entry['array_bytes'] / GIB:6.3f} GiB",
            flush=True,
        )
    arenas = reading["arenas"]
    print(
        f"      arenas: main system {arenas['main']['system'] / GIB:.3f} free "
        f"{arenas['main']['free'] / GIB:.3f}; {arenas['thread_arenas']} thread arenas system "
        f"{arenas['thread_system'] / GIB:.3f} free {arenas['thread_free'] / GIB:.3f}; mmapped "
        f"x{arenas['mmap_count']} {arenas['mmap_bytes'] / GIB:.3f} GiB; array buffers "
        f"{smaps['array_buffers'] / GIB:.3f}",
        flush=True,
    )
    python = reading["python"]
    top = ", ".join(f"{n} {s / MIB:.0f}M/{c}" for n, c, s in python["top"][:6])
    print(
        f"      python: {python['objects']} objects {python['bytes'] / GIB:.3f} GiB ({top})",
        flush=True,
    )


def _print_hlo(hlo: dict) -> None:
    print(
        f"        HLO: {hlo['modules']} modules {hlo['bytes'] / GIB:.3f} GiB; "
        f"{hlo['over_1_mib']} over 1 MiB hold {hlo['bytes_over_1_mib'] / GIB:.3f} GiB",
        flush=True,
    )
    for name, count, size in hlo["top_names"][:10]:
        print(f"          {size / MIB:9.1f} MiB  x{count:<4d} {name}", flush=True)


def _print_functions(title: str, functions: dict) -> None:
    groups = ", ".join(
        f"{g} {e['programs']} in {e['functions']} functions"
        for g, e in sorted(functions["by_group"].items())
    )
    print(
        f"      {title}: {functions['executables']} executables, "
        f"{functions['held_by_functions']} in jitted functions' caches ({groups})",
        flush=True,
    )
    for module, size in list(functions["by_module"].items())[:12]:
        print(f"        {size:5d}  {module}", flush=True)
    for module, name, size in functions["top"][:12]:
        print(f"        {size:5d}  {module}.{name}", flush=True)


def _print(label: str, residue: dict) -> None:
    resident = residue["resident"]
    print(
        f"    residue {label}: VmRSS {resident['VmRSS'] / GIB:.2f}, anon {resident['RssAnon'] / GIB:.2f}"
        f" GiB, {resident['Threads']} threads, reading {residue['seconds']:.1f} s",
        flush=True,
    )
    compiled = residue["compilations"]
    print(
        f"      executables {residue['executable_count']}; since the start: traced "
        f"{compiled.get('trace', 0):.0f} ({compiled.get('trace_seconds', 0):.1f} s), lowered "
        f"{compiled.get('lower_seconds', 0):.1f} s, backend {compiled.get('backend', 0):.0f} "
        f"({compiled.get('backend_seconds', 0):.1f} s), persistent cache hits "
        f"{compiled.get('cache_hits', 0):.0f} misses {compiled.get('cache_misses', 0):.0f}",
        flush=True,
    )
    if "smaps" in residue:
        _print_reading("", residue)
        executables = residue["executables"]
        print(
            f"      executables: {executables['count']}, code {executables['code'] / MIB:.1f} MiB",
            flush=True,
        )
    if "trim" in residue:
        trim = residue["trim"]
        print(
            f"      malloc_trim: VmRSS {trim['rss_before'] / GIB:.2f} -> {trim['rss_after'] / GIB:.2f}"
            f" GiB (gave back {(trim['rss_before'] - trim['rss_after']) / GIB:.2f})",
            flush=True,
        )
    if "hlo" in residue:
        live = residue["hlo"]["live"]
        print(
            f"      live: VmRSS {live['rss'] / GIB:.2f} device {live['device'] / GIB:.2f} traced "
            f"{live['traced'] / GIB:.2f} GiB",
            flush=True,
        )
        _print_hlo(residue["hlo"])
    for key in ("functions", "functions_after_drop"):
        if key in residue:
            _print_functions(key.replace("_", " "), residue[key])
    if "drop" in residue:
        drop = residue["drop"]
        before, after = drop["before"], drop["after"]
        print(
            f"      drop {','.join(drop['groups'])}: {drop['programs_dropped']} programs; executables "
            f"{before['executables']} -> {after['executables']}; VmRSS {before['rss'] / GIB:.2f} -> "
            f"{after['rss'] / GIB:.2f}, traced {before['traced'] / GIB:.2f} -> "
            f"{after['traced'] / GIB:.2f} GiB (freed {(before['rss'] - after['rss']) / GIB:.2f})",
            flush=True,
        )
    if "clear" in residue:
        clear = residue["clear"]
        print(
            f"      clear_caches: executables {clear['executables_before']} -> "
            f"{clear['after']['executables']}",
            flush=True,
        )
        _print_hlo(clear["hlo"])
        for name in ("before", "after_collect_and_trim", "after_clear", "after_clear_and_trim"):
            live = clear[name]
            print(
                f"        {name:22s} VmRSS {live['rss'] / GIB:.2f} device {live['device'] / GIB:.2f}"
                f" traced {live['traced'] / GIB:.2f} GiB",
                flush=True,
            )
        if "smaps" in clear["after"]:
            _print_reading("after the clear", clear["after"])


def _after_march(recorder) -> None:
    """One more reading once the measured steps are done: the stage ``end``."""
    with recorder.stage("end"):
        pass


def main() -> None:
    attribution = _attribution()
    _count_compilations()
    recorder = attribution.Recorder(record=RECORD)
    recorder.readings.append(reading)
    recorder.save_every_stage = True
    recorder.extra["residue_settings"] = {
        "case": CASE,
        "read": READ,
        "trim": TRIM,
        "clear": sorted(CLEAR),
        "hlo": sorted(HLO),
        "functions": sorted(FUNCTIONS),
        "drop": {label: sorted(groups) for label, groups in DROP.items()},
    }
    print(f"residue settings: {recorder.extra['residue_settings']}", flush=True)
    attribution.main(
        load_case=lambda: load_case(CASE),
        recorder=recorder,
        march_log=HERE / f"memory_residue.{CASE}.march.log",
        after_march=_after_march,
    )


if __name__ == "__main__":
    main()
