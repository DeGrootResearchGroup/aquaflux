"""Who owns bfs3d's memory: every stage of a short march, named by what it allocated and what it kept.

A resident-set reading per stage says *when* the memory went and not *to whom*. This harness runs the
case's own march (``compare.solve_aquaflux``, the shipped solver section of ``case.yaml``) for
``BFS3D_MEMORY_STEPS`` steps (default 3) in one process, with the library's own functions wrapped as
nested stages -- nothing they compute is changed -- and for each stage records:

* ``VmRSS`` after it and its change, ``VmHWM``, and the stage's own resident peak (a 20 ms sampler);
* host allocations traced by ``tracemalloc`` (numpy registers every array buffer with it, so scipy's
  sparse matrices are included): the current total and the stage's own peak -- a transient that never
  shows in a before/after reading;
* JAX's live device buffers (``jax.live_arrays()``; on the CPU backend these are host memory too);
* ``/proc/self/smaps`` split into anonymous executable pages (the compiled XLA programs), other
  anonymous pages and file-backed pages, and glibc's own free-but-retained bytes (``mallinfo2``);
* for the stages marked with a census, every array of at least 1 MiB alive at its end -- numpy buffers
  (deduplicated through their ``base``), ``scipy.sparse`` arrays and JAX arrays -- named by the chain of
  objects (or the running function's local variable) that holds it, so a duplicated matrix shows as two
  owners of one shape. ``BFS3D_MEMORY_CENSUS_STAGES`` names the stages that take one instead, so a
  longer run can afford a census at the stage that sets its peak;
* with ``BFS3D_MEMORY_SAMPLE_DEVICE=1``, the resident, live device and traced host bytes at the instant
  of each stage's resident peak (from the sampler), so a peak is split into what was live then and what
  was not -- the running programs' temporary buffers, the compiled-program cache, the allocator's
  retention;
* with ``BFS3D_MEMORY_PROGRAMS=1``, once the measured steps are done, the loaded programs ranked by
  their temporary buffers.

The march's own stages are the session's fit and re-fits, each split into its probe, its shift, the
field split's block cut, each block's re-fit and the traced split, and the march step's first call (its
compilation). After the last step the compiled step, probe and residual programs are compiled again from
the persistent cache and their ``memory_analysis`` (code and temporary-buffer sizes) is printed.

Run it through the case runner, alone::

    validation/run_case.sh validation/bfs3d_openfoam/memory_attribution.py --wait

It writes a Markdown table to the log and the full record, every census included, to
``memory_attribution.json`` beside this file (``BFS3D_MEMORY_RECORD`` overrides the path).

``tracemalloc`` costs time and some memory of its own (reported per stage as ``tracer``), so wall times
here are not march timings and the resident figures sit slightly above an untraced run's.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: The steps marched before the measurement stops.
STEPS = int(os.environ.get("BFS3D_MEMORY_STEPS", "3"))
#: The smallest array a census lists.
CENSUS_BYTES = 1 << 20
#: Classes an owner chain looks through to the object holding them (a matrix, a hierarchy level).
TRANSPARENT = (
    "TracedCycle",
    "SmoothedHierarchy",
    "_SparseLevel",
    "_PaddedRowOperator",
    "_SimplePieces",
    "TracedFieldSplit",
)
#: Collect cyclic garbage at the end of every stage, recording the device and resident bytes it frees.
#: An arm, not a reading: it changes what the rest of the run holds.
COLLECT = os.environ.get("BFS3D_MEMORY_COLLECT", "0") == "1"
#: ``0`` takes no census at all (a census walks every Python object, minutes per call late in a run).
CENSUS = os.environ.get("BFS3D_MEMORY_CENSUS", "1") == "1"
#: A class name whose live instances are traced back to their holders at each step's end (for a copy
#: that should be gone, say), as referrer chains printed to the log; unset traces nothing.
HOLDERS = os.environ.get("BFS3D_MEMORY_HOLDERS", "")
#: Stage labels (as the table prints them, without indentation; separated by ``;``, since a label may
#: hold a comma) whose end alone takes a census, in place of the stages the run marks; unset, the run's
#: own. A census late in a run walks every Python object, so a 2-step run can afford one or two.
CENSUS_STAGES = {
    label.strip()
    for label in os.environ.get("BFS3D_MEMORY_CENSUS_STAGES", "").split(";")
    if label.strip()
}
#: ``1`` also samples the live device bytes every 10th resident sample (every 0.2 s), so each stage
#: reports what the device held at its resident peak; the host bytes ``tracemalloc`` traces are sampled
#: with every resident sample regardless.
SAMPLE_DEVICE = os.environ.get("BFS3D_MEMORY_SAMPLE_DEVICE", "0") == "1"
#: ``1`` prints the loaded programs ranked by their temporary buffers once the measured steps are done.
PROGRAMS = os.environ.get("BFS3D_MEMORY_PROGRAMS", "0") == "1"
#: ``1`` also times each entry map's build as a stage (a row per map: dozens per re-fit).
MAPS = os.environ.get("BFS3D_MEMORY_MAPS", "0") == "1"
RECORD = Path(os.environ.get("BFS3D_MEMORY_RECORD", HERE / "memory_attribution.json"))
#: ``1`` prints a checksum of every leaf of each traced split the session composes (at the fit and at
#: every re-fit), so two checkouts' runs show whether anything the solve applies differs, bit for bit.
CHECKSUMS = os.environ.get("BFS3D_MEMORY_CHECKSUMS", "0") == "1"
#: ``1`` (default, the session's own default) gives the allocator's kept memory back before every re-fit
#: (``malloc_trim(0)``), each trim printing the resident size and glibc's retained bytes on either side
#: of it; ``0`` opens every session with no trim. An arm, not a reading: it changes what the run holds.
TRIM = os.environ.get("BFS3D_MEMORY_TRIM", "1") == "1"
GIB = 2**30


# --- process readings ---------------------------------------------------------------------------


#: On macOS the process is read by its physical footprint, which counts compressed pages as the
#: resident set does not: a process the system has compressed reads far smaller by its resident set.
DARWIN = sys.platform == "darwin"


class _RusageInfoV4(ctypes.Structure):
    """``rusage_info_v4`` of ``<sys/resource.h>``: a UUID, then 64-bit counters in this order."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
            "proc_exit_abstime", "child_user_time", "child_system_time", "child_pkg_idle_wkups",
            "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
            "diskio_byteswritten", "cpu_time_qos_default", "cpu_time_qos_maintenance",
            "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy",
            "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive", "billed_system_time",
            "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint", "instructions",
            "cycles", "billed_energy", "serviced_energy", "interval_max_phys_footprint",
            "runnable_time",
        )
    ]  # fmt: skip


def _rusage() -> _RusageInfoV4:
    info = _RusageInfoV4()
    if _LIBPROC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:  # RUSAGE_INFO_V4
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return info


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True) if DARWIN else None


def _peak_bytes() -> int:
    """The process's peak: ``VmHWM`` on Linux, the lifetime peak physical footprint on macOS."""
    if DARWIN:
        return _rusage().lifetime_max_phys_footprint
    return _status_bytes("VmHWM")


def _status_bytes(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f"/proc/self/status has no {field}")


def _rss_bytes() -> int:
    """The process's memory now: the resident set on Linux, the physical footprint on macOS."""
    if DARWIN:
        return _rusage().phys_footprint
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _smaps() -> dict[str, int]:
    """Resident bytes by kind of mapping: anonymous executable, anonymous other, file-backed (Linux
    only; empty elsewhere)."""
    if DARWIN:
        return {}
    kinds = dict.fromkeys(("anon_exec", "anon", "file"), 0)
    kind = "anon"
    for line in Path("/proc/self/smaps").read_text().splitlines():
        head = line.split(None, 5)
        if len(head) >= 5 and "-" in head[0] and len(head[1]) == 4:
            perms, inode = head[1], head[4]
            path = head[5] if len(head) == 6 else ""
            if inode != "0" and not path.startswith("["):
                kind = "file"
            else:
                kind = "anon_exec" if "x" in perms else "anon"
        elif line.startswith("Rss:"):
            kinds[kind] += int(line.split()[1]) * 1024
    return kinds


class _MallInfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.mallinfo2.restype = _MallInfo2
except (OSError, AttributeError):  # not glibc, or glibc older than 2.33
    _LIBC = None


def _malloc() -> dict[str, int]:
    """glibc's view: bytes in its arenas, in use there, free there (retained), and mmapped directly."""
    if _LIBC is None:
        return {}
    info = _LIBC.mallinfo2()
    return {
        "arena": info.arena,
        "in_use": info.uordblks,
        "free_retained": info.fordblks,
        "mmapped": info.hblkhd,
    }


def _machine_memory() -> dict[str, int]:
    """The machine's memory and swap, in bytes, as ``MemTotal`` and ``SwapTotal``."""
    if DARWIN:
        sysctl = functools.partial(subprocess.run, capture_output=True, text=True, check=True)
        total = int(sysctl(["sysctl", "-n", "hw.memsize"]).stdout)
        swap = sysctl(["sysctl", "-n", "vm.swapusage"]).stdout.split()  # total = 1024.00M ...
        return {"MemTotal": total, "SwapTotal": int(float(swap[2].rstrip("M")) * 2**20)}
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }


class _Sampler(threading.Thread):
    """The resident set every 20 ms, kept as a series so any nested stage can read its own peak.

    Each sample also keeps the host bytes ``tracemalloc`` traces then, and with ``SAMPLE_DEVICE`` every
    10th keeps the live device bytes, so a stage can say what was live at the instant of its peak.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.times: list[float] = []
        self.rss: list[int] = []
        self.traced: list[int] = []
        self.device_times: list[float] = []
        self.device: list[int] = []

    def run(self) -> None:
        count = 0
        while True:
            self.times.append(time.perf_counter())
            self.rss.append(_rss_bytes())
            self.traced.append(tracemalloc.get_traced_memory()[0])
            if SAMPLE_DEVICE and count % 10 == 0:
                import jax

                with contextlib.suppress(Exception):  # an array deleted while it is read
                    device = sum(a.nbytes for a in jax.live_arrays())
                    self.device_times.append(time.perf_counter())
                    self.device.append(device)
            count += 1
            time.sleep(0.02)

    def peak(self, start: float, stop: float) -> int:
        import bisect

        lo = bisect.bisect_left(self.times, start)
        hi = bisect.bisect_right(self.times, stop)
        return max(self.rss[lo:hi], default=0)

    def at_peak(self, start: float, stop: float) -> dict[str, int]:
        """The resident, traced and (if sampled) device bytes at the stage's highest resident sample."""
        import bisect

        lo = bisect.bisect_left(self.times, start)
        # The sampler appends a sample's time before its readings, so the newest time may not have them.
        hi = min(bisect.bisect_right(self.times, stop), len(self.rss), len(self.traced))
        if hi <= lo:
            return {}
        index = max(range(lo, hi), key=self.rss.__getitem__)
        reading = {"rss": self.rss[index], "traced": self.traced[index]}
        nearest = bisect.bisect_right(self.device_times, self.times[index]) - 1
        if nearest >= 0 and self.device_times[nearest] >= start:
            reading["device"] = self.device[nearest]
            reading["device_age"] = self.times[index] - self.device_times[nearest]
        return reading


# --- the census ---------------------------------------------------------------------------------


def _census() -> dict[int, dict]:
    """Every array of at least ``CENSUS_BYTES`` alive now, keyed by buffer, with what holds it.

    Numpy arrays are reduced to the array owning their buffer (a view and its base are one buffer).
    A buffer that numpy does not own (``np.asarray`` of a JAX array on the CPU backend aliases the
    device buffer) is recorded as such and left out of the host total. Owners are read by walking
    referrers up to three non-container hops, and the running stack's local variables are roots too.
    """
    import jax
    import numpy as np
    import scipy.sparse as sp

    objects = gc.get_objects()
    frame_roots: list[tuple[str, object]] = []
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_name
        if frame.f_code.co_filename != __file__:
            for var, value in frame.f_locals.items():
                frame_roots.append((f"local {name}:{var}", value))
        frame = frame.f_back

    found: dict[int, dict] = {}

    def consider(obj) -> None:
        if isinstance(obj, np.ndarray):
            root = obj
            while isinstance(root.base, np.ndarray):
                root = root.base
            if root.nbytes >= CENSUS_BYTES and id(root) not in found:
                foreign = root.base is not None
                found[id(root)] = {
                    "kind": "host-alias" if foreign else "host",
                    "shape": list(root.shape),
                    "dtype": str(root.dtype),
                    "bytes": int(root.nbytes),
                    "obj": root,
                }
        elif isinstance(obj, jax.Array) and not isinstance(obj, jax.core.Tracer):
            with contextlib.suppress(Exception):
                if obj.nbytes >= CENSUS_BYTES and id(obj) not in found:
                    found[id(obj)] = {
                        "kind": "device",
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "bytes": int(obj.nbytes),
                        "obj": obj,
                    }

    for obj in objects:
        for child in gc.get_referents(obj):
            if type(child) is np.ndarray:
                consider(child)
    for _, value in frame_roots:
        consider(value)
    for array in jax.live_arrays():
        consider(array)

    # Owners: up to three passes, each finding the parents of the objects still to be named.
    containers = (list, tuple, dict, set, frozenset)
    module_dicts = {id(m.__dict__): m.__name__ for m in list(sys.modules.values()) if m}
    local_name = {}
    for label, value in frame_roots:
        local_name.setdefault(id(value), label)
    parents: dict[int, list] = defaultdict(list)
    wanted = {key for key in found}
    # A view of a numpy root is a referent of its holder, not the root itself: map views to roots.
    view_root: dict[int, int] = {}
    for obj in objects:
        for child in gc.get_referents(obj):
            if isinstance(child, np.ndarray) and isinstance(child.base, np.ndarray):
                root = child
                while isinstance(root.base, np.ndarray):
                    root = root.base
                if id(root) in found:
                    view_root[id(child)] = id(root)
    for _ in range(4):
        next_wanted = set()
        for obj in objects:
            for child in gc.get_referents(obj):
                key = view_root.get(id(child), id(child))
                if key in wanted:
                    parents[key].append(obj)
                    if (
                        isinstance(obj, containers)
                        or isinstance(obj, sp.sparray | sp.spmatrix)
                        or type(obj).__name__ in TRANSPARENT
                    ):
                        next_wanted.add(id(obj))
        wanted = next_wanted - set(parents)
        if not wanted:
            break

    def attribute(holder, child) -> str:
        names = []
        with contextlib.suppress(Exception):
            names = list(vars(holder))
        if not names and dataclasses.is_dataclass(holder):
            names = [field.name for field in dataclasses.fields(holder)]
        for name in names:
            value = getattr(holder, name, None)
            if value is child or view_root.get(id(value)) == id(child):
                return name
        return "?"

    def owner_of(key: int, child, hops: int = 0) -> str:
        if key in local_name:
            return local_name[key]
        if hops >= 6:
            return "…"
        for holder in parents.get(key, ()):
            if holder is objects or holder is frame_roots:
                continue
            if id(holder) in module_dicts:
                for name, value in holder.items():
                    if value is child:
                        return f"module {module_dicts[id(holder)]}.{name}"
                return f"module {module_dicts[id(holder)]}"
            if isinstance(holder, containers):
                return owner_of(id(holder), holder, hops + 1)
            name = f"{type(holder).__name__}.{attribute(holder, child)}"
            if isinstance(holder, sp.sparray | sp.spmatrix) or type(holder).__name__ in TRANSPARENT:
                return f"{owner_of(id(holder), holder, hops + 1)} > {name}"
            return name
        return "(unreferenced from Python)"

    for key, entry in found.items():
        entry["owner"] = owner_of(key, entry["obj"])
        del entry["obj"]
    return found


def _device_by_dtype() -> dict[str, int]:
    import jax

    totals: dict[str, int] = defaultdict(int)
    for array in jax.live_arrays():
        totals[str(array.dtype)] += array.nbytes
    return dict(totals)


def _summarize(census: dict[int, dict]) -> dict[str, int]:
    totals = defaultdict(int)
    for entry in census.values():
        totals[entry["kind"]] += entry["bytes"]
    return dict(totals)


def _by_owner(entries) -> list[tuple[str, int, int]]:
    """(owner, bytes, count) grouped, largest first."""
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for entry in entries:
        key = f"{entry['kind']}: {entry['owner']}"
        totals[key][0] += entry["bytes"]
        totals[key][1] += 1
    return sorted(((k, v[0], v[1]) for k, v in totals.items()), key=lambda r: -r[1])


def _by_shape(census: dict[int, dict]) -> list[tuple[str, int, int]]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for entry in census.values():
        key = f"{entry['kind']} {entry['dtype']}{tuple(entry['shape'])}"
        totals[key][0] += entry["bytes"]
        totals[key][1] += 1
    return sorted(((k, v[0], v[1]) for k, v in totals.items()), key=lambda r: -r[1])


# --- stages -------------------------------------------------------------------------------------


class Recorder:
    """Nested stages: each reports its own resident peak and traced peak, and optionally a census."""

    def __init__(self, record: Path = RECORD) -> None:
        self.sampler = _Sampler()
        self.rows: list[dict] = []
        self._stack: list[dict] = []
        self._last_census: dict = {}
        self.extra: dict = {}
        self.record = record
        #: Further readings taken at every stage's end, each called with the stage's row to add to
        #: (``validation/memory_residue.py`` adds its own); run after ``BFS3D_MEMORY_COLLECT``'s.
        self.readings: list = []
        #: Write the record at every stage's end, not only after a census (a run that may be stopped
        #: early, by a container restart say, keeps every stage measured before it).
        self.save_every_stage = False

    def save(self) -> None:
        """Write the record so far: a run stopped early still leaves everything measured before it."""
        self.record.write_text(json.dumps({"rows": self.rows, **self.extra}, indent=1, default=str))

    def start(self) -> None:
        tracemalloc.start(1)
        self.sampler.start()

    @contextlib.contextmanager
    def stage(self, label: str, *, census: bool = False):
        import jax

        census = CENSUS and (label in CENSUS_STAGES if CENSUS_STAGES else census)

        if self._stack:  # hand the parent the peak so far before this stage resets it
            parent = self._stack[-1]
            parent["traced_peak"] = max(parent["traced_peak"], tracemalloc.get_traced_memory()[1])
        tracemalloc.reset_peak()
        frame = {
            "label": label,
            "depth": len(self._stack),
            "t0": time.perf_counter(),
            "rss0": _rss_bytes(),
            "traced0": tracemalloc.get_traced_memory()[0],
            "traced_peak": 0,
        }
        self._stack.append(frame)
        try:
            yield
        finally:
            self._stack.pop()
            t1 = time.perf_counter()
            traced, traced_peak = tracemalloc.get_traced_memory()
            traced_peak = max(traced_peak, frame["traced_peak"])
            if self._stack:
                parent = self._stack[-1]
                parent["traced_peak"] = max(parent["traced_peak"], traced_peak)
            rss = _rss_bytes()
            row = {
                "label": label,
                "depth": frame["depth"],
                "seconds": t1 - frame["t0"],
                "rss": rss,
                "rss_delta": rss - frame["rss0"],
                "stage_peak": max(self.sampler.peak(frame["t0"], t1), rss),
                "at_peak": self.sampler.at_peak(frame["t0"], t1),
                "hwm": _peak_bytes(),
                "traced": traced,
                "traced_delta": traced - frame["traced0"],
                "traced_peak": traced_peak,
                "tracer": tracemalloc.get_tracemalloc_memory(),
                "device": sum(a.nbytes for a in jax.live_arrays()),
                "device_by_dtype": _device_by_dtype(),
                "smaps": _smaps(),
                "malloc": _malloc(),
            }
            if COLLECT:
                unreachable = gc.collect()
                row["collected"] = {
                    "objects": unreachable,
                    "device_after": sum(a.nbytes for a in jax.live_arrays()),
                    "rss_after": _rss_bytes(),
                }
            for reading in self.readings:
                reading(row)
            if census:
                after = _census()
                # Against the previous census anywhere, so one census per stage boundary suffices.
                before = self._last_census
                self._last_census = after
                kept = [e for k, e in after.items() if k not in before]
                freed = [e for k, e in before.items() if k not in after]
                row["census"] = {
                    "totals": _summarize(after),
                    "kept": _by_owner(kept)[:12],
                    "freed": _by_owner(freed)[:8],
                    "all_by_owner": _by_owner(after.values())[:40],
                    "all_by_shape": _by_shape(after)[:25],
                }
            self.rows.append(row)
            if census or self.save_every_stage:
                self.save()
            self._print(row)
            tracemalloc.reset_peak()

    def _print(self, row: dict) -> None:
        indent = "  " * row["depth"]
        smaps = row["smaps"]
        line = (
            f"| {indent}{row['label']} | {row['seconds']:.1f} | {row['rss'] / GIB:.2f} | "
            f"{row['rss_delta'] / GIB:+.2f} | {row['stage_peak'] / GIB:.2f} | {row['hwm'] / GIB:.2f} | "
            f"{row['traced'] / GIB:.2f} | {row['traced_peak'] / GIB:.2f} | {row['device'] / GIB:.2f} | "
            f"{smaps.get('anon_exec', 0) / GIB:.2f} | "
            f"{row['malloc'].get('free_retained', 0) / GIB:.2f} | {row['tracer'] / GIB:.2f} |"
        )
        print(line, flush=True)
        at_peak = row.get("at_peak", {})
        if SAMPLE_DEVICE and "device" in at_peak:
            print(
                f"    at the stage peak: resident {at_peak['rss'] / GIB:.2f}, device "
                f"{at_peak['device'] / GIB:.2f} (read {at_peak['device_age']:.2f} s before), host "
                f"traced {at_peak['traced'] / GIB:.2f} GiB",
                flush=True,
            )
        collected = row.get("collected")
        if collected:
            print(
                f"    gc.collect: {collected['objects']} objects, device "
                f"{(row['device'] - collected['device_after']) / GIB:.2f} GiB and resident "
                f"{(row['rss'] - collected['rss_after']) / GIB:.2f} GiB freed",
                flush=True,
            )
        census = row.get("census")
        if census:
            totals = ", ".join(f"{k} {v / GIB:.2f}" for k, v in sorted(census["totals"].items()))
            print(f"    census at end (GiB): {totals}", flush=True)
            for title, rows in (("kept", census["kept"][:6]), ("freed", census["freed"][:4])):
                for owner, size, count in rows:
                    print(f"    {title} {size / 2**20:9.1f} MiB  x{count}  {owner}", flush=True)

    def wrap(self, owner, attribute: str, label: str, *, census: int = 0, first: int = 0):
        """Replace ``owner.attribute`` by itself run as a stage (only its first ``first`` calls if set).

        ``census`` is how many of its calls end in a census; ``-1`` is every call.
        """
        raw = (
            owner.__dict__.get(attribute, getattr(owner, attribute))
            if isinstance(owner, type)
            else getattr(owner, attribute)
        )
        kind = type(raw) if isinstance(raw, classmethod | staticmethod) else None
        function = raw.__func__ if kind is not None else raw
        calls = [0]

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            calls[0] += 1
            if first and calls[0] > first:
                return function(*args, **kwargs)
            name = label if first == 1 else f"{label} #{calls[0]}"
            with self.stage(name, census=census < 0 or calls[0] <= census):
                return function(*args, **kwargs)

        setattr(owner, attribute, kind(wrapped) if kind is not None else wrapped)
        return wrapped


# --- the run ------------------------------------------------------------------------------------


def _header() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    meminfo = _machine_memory()
    cache = Path(
        os.environ.get("AQUAFLUX_COMPILATION_CACHE_DIR", Path.home() / ".cache/aquaflux/jax")
    )
    header = {
        "commit": commit + (" (uncommitted changes)" if dirty else ""),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "mem_total_gib": meminfo["MemTotal"] / GIB,
        "swap_total_gib": meminfo["SwapTotal"] / GIB,
        "compilation_cache_entries": len(list(cache.iterdir())) if cache.is_dir() else 0,
        "malloc_env": {k: v for k, v in os.environ.items() if k.startswith(("MALLOC_", "Malloc"))},
        "steps": STEPS,
        "memory_trim": TRIM,
    }
    print(json.dumps(header), flush=True)
    return header


def _instrument(recorder: Recorder, compare) -> dict:
    """Wrap the library's own stages; returns the captured arguments of the programs analysed later."""
    from aquaflux.flow import MomentumContinuity
    from aquaflux.mesh import Mesh
    from aquaflux.precondition import (
        field_split,
        hierarchy_inverse,
        jacobian_probe,
        materialized_session,
        probe_layout,
        sparse_jacobian,
    )
    from aquaflux.solve import march
    from aquaflux.turbulence import CoupledRANS, SSTTurbulence, reynolds, transport

    wrap = recorder.wrap
    if hasattr(compare, "read_openfoam"):  # bfs3d reads its mesh itself; pitzDaily's case file does
        wrap(compare, "read_openfoam", "mesh read", first=1)
    wrap(Mesh, "geometry", "geometry", first=1)
    wrap(compare, "build_case", "case build", census=True, first=1)
    wrap(MomentumContinuity, "build", "momentum + continuity build", first=1)
    wrap(SSTTurbulence, "build", "SST build (boundaries, wall distance)", first=1)
    wrap(transport, "distance_to_patches", "wall-distance search", first=1)
    wrap(CoupledRANS, "build", "coupled build", first=1)
    wrap(reynolds, "hybrid_initialize", "hybrid_initialize", census=True, first=1)

    session = materialized_session.MaterializedSession
    wrap(session, "_probe_for", "probe plan (colouring, gather map)", census=True, first=1)
    wrap(jacobian_probe, "block_stencil_colouring", "pattern, conflict graph, colouring", first=1)
    wrap(sparse_jacobian, "_saturation_colouring", "saturation colouring", first=1)
    wrap(jacobian_probe, "block_stencil_gather_map", "gather map", first=1)
    wrap(probe_layout._CellEntries, "__init__", "probe entry order (cell level)", first=1)
    wrap(probe_layout._DofEntries, "__init__", "probe entry order (entry by entry)", first=1)
    wrap(session, "_fit", "fit (first build)", census=True, first=1)
    wrap(
        materialized_session,
        "batched_jacobian_matvec",
        "batched JVP, first call (compile)",
        first=1,
    )
    wrap(
        materialized_session.BetaTrackingRefresh,
        "_materialize_at",
        "re-fit",
        census=-1,
    )
    wrap(sparse_jacobian, "materialize_block_jacobian", "probe (materialize)")
    wrap(sparse_jacobian, "materialize_blocks", "probe (into the split's blocks)")
    wrap(sparse_jacobian, "shifted_jacobian", "shift (copy + setdiag)")
    wrap(field_split.FieldGroups, "blocks", "block cut (four slices)", census=1)
    wrap(hierarchy_inverse.HierarchyBlockInverse, "__init__", "block inverse build")
    wrap(hierarchy_inverse.HierarchyBlockInverse, "refactor_block", "block re-fit")
    wrap(materialized_session, "traced_field_split", "traced split (device leaves)")
    if CHECKSUMS:
        _print_split_checksums(materialized_session)
    _hand_the_session_a_trim(materialized_session, TRIM)
    if MAPS:
        from aquaflux.precondition import entry_map

        wrap(entry_map.EntryMap, "product", "entry map: product")
        wrap(entry_map.EntryMap, "gathering", "entry map: gathering")
        wrap(entry_map.EntryMap, "placing", "entry map: placing")

    captured: dict = {}
    step_program = march._march_step
    calls = [0]

    def march_step(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            with recorder.stage("march step, first call (compile + run)", census=True):
                out = step_program(*args, **kwargs)
        else:
            out = step_program(*args, **kwargs)
        if captured.get("want_step"):
            captured["step"] = (args, kwargs)
            captured["want_step"] = False
        return out

    march._march_step = march_step
    captured["step_program"] = step_program
    return captured


def _hand_the_session_a_trim(materialized_session, trimmed: bool) -> None:
    """Open every session with a reporting :class:`~aquaflux.allocator.GlibcTrim`, or with none."""
    from aquaflux.allocator import GlibcTrim

    trim = GlibcTrim()
    count = [0]

    def reported() -> None:
        before_rss, before_free = _rss_bytes(), _malloc().get("free_retained", 0)
        trim()
        count[0] += 1
        print(
            f"    [trim {count[0]}] released {trim.released}: resident {before_rss / GIB:.3f} -> "
            f"{_rss_bytes() / GIB:.3f} GiB, malloc free {before_free / GIB:.3f} -> "
            f"{_malloc().get('free_retained', 0) / GIB:.3f} GiB",
            flush=True,
        )

    session = materialized_session.MaterializedSession
    opened = session.__init__

    @functools.wraps(opened)
    def with_trim(self, *args, **kwargs):
        opened(self, *args, memory_trim=reported if trimmed else None, **kwargs)

    session.__init__ = with_trim


def _print_split_checksums(materialized_session) -> None:
    """Print a digest of every leaf of each traced split the session composes, after the stage wrap."""
    import hashlib

    import jax
    import numpy as np

    composed = materialized_session.traced_field_split
    splits = [0]

    def checksummed(*args, **kwargs):
        split = composed(*args, **kwargs)
        splits[0] += 1
        whole = hashlib.sha256()
        leaves = jax.tree_util.tree_leaves_with_path(split)
        for path, leaf in leaves:
            data = np.asarray(leaf)
            digest = hashlib.sha256(data.dtype.str.encode() + data.tobytes()).hexdigest()
            whole.update(digest.encode())
            print(
                f"    split {splits[0]} leaf {jax.tree_util.keystr(path)} {data.dtype} "
                f"{data.shape} {digest[:16]}",
                flush=True,
            )
        print(
            f"    split {splits[0]}: {len(leaves)} leaves, digest {whole.hexdigest()}", flush=True
        )
        return split

    materialized_session.traced_field_split = checksummed


def _programs_by_temporaries(top: int = 15) -> list[tuple[str, int, int, int]]:
    """The loaded programs grouped by name: (name, count, largest temporary, its arguments), largest first.

    A program's temporary buffers are allocated for each call and freed after it, so the largest one a
    stage runs sits on that stage's resident baseline at its peak.
    """
    import jax

    grouped: dict[str, list[int]] = {}
    for executable in jax.devices()[0].client.live_executables():
        try:
            name = executable.hlo_modules()[0].name
            stats = executable.get_compiled_memory_stats()
        except Exception:  # a reading, not a gate
            continue
        entry = grouped.setdefault(name, [0, 0, 0])
        entry[0] += 1
        if stats.temp_size_in_bytes > entry[1]:
            entry[1], entry[2] = stats.temp_size_in_bytes, stats.argument_size_in_bytes
    ranked = sorted(grouped.items(), key=lambda item: -item[1][1])[:top]
    for name, (count, temp, argument) in ranked:
        print(
            f"program {name} x{count}: temporaries {temp / 2**20:.1f} MiB, arguments "
            f"{argument / 2**20:.1f} MiB",
            flush=True,
        )
    return [(name, count, temp, argument) for name, (count, temp, argument) in ranked]


def _program_sizes(captured: dict, case) -> dict[str, dict]:
    """Compile the step, the batched probe and the residual again and read their memory analyses."""
    import jax
    import jax.numpy as jnp
    from aquaflux.precondition import materialized_session

    sizes = {}

    def analyse(name, program, *args, **kwargs):
        try:
            analysis = program.lower(*args, **kwargs).compile().compiled.memory_analysis()
        except Exception as error:  # a reading, not a gate
            sizes[name] = {"error": repr(error)[:300]}
            return
        sizes[name] = {
            "code": analysis.generated_code_size_in_bytes,
            "temp": analysis.temp_size_in_bytes,
            "argument": analysis.argument_size_in_bytes,
            "output": analysis.output_size_in_bytes,
            "alias": analysis.alias_size_in_bytes,
        }

    if "step" in captured:
        args, kwargs = captured.pop("step")
        analyse("march step", captured["step_program"], *args, **kwargs)
    coupled = case["coupled"]
    state = jnp.zeros(coupled.layout.size)
    analyse(
        "batched JVP (8 tangents)",
        # Unwrapped: the stage wrapper above replaced the module's name with a plain function.
        getattr(
            materialized_session.batched_jacobian_matvec,
            "__wrapped__",
            materialized_session.batched_jacobian_matvec,
        ),
        coupled,
        state,
        jnp.zeros((materialized_session.PROBE_BATCH_SIZE, state.size)),
    )
    analyse("single JVP", materialized_session.jacobian_matvec, coupled, state, state)
    import equinox as eqx

    analyse("residual", eqx.filter_jit(lambda c, s: c.residual(s)), coupled, state)
    for name, size in sizes.items():
        print(
            f"program {name}: "
            + ", ".join(
                f"{k} {v / 2**20:.1f} MiB" if isinstance(v, int) else f"{k} {v}"
                for k, v in size.items()
            ),
            flush=True,
        )
    del jax
    return sizes


def _holders(class_name: str, depth: int = 7) -> list[str]:
    """Each live instance of ``class_name``, as the chain of objects holding it (first referrer each hop).

    A running function holding it is named (``frames ...``) and the chain goes on through its other
    holders; a module dictionary ends a chain, and the lists this function builds are left out.
    """
    import types

    objects = gc.get_objects()
    found = [o for o in objects if type(o).__name__ == class_name]
    module_dicts = {id(m.__dict__): m.__name__ for m in list(sys.modules.values()) if m}
    ignore = {id(objects), id(found)}
    chains = []
    for target in found:
        chain, current, seen = [], target, {id(target)}
        for _ in range(depth):
            referrers = gc.get_referrers(current)
            ignore.add(id(referrers))
            holders = [h for h in referrers if id(h) not in ignore and id(h) not in seen]
            ignore.add(id(holders))
            if not holders:
                chain.append("(no Python holder)")
                break
            frames = [h for h in holders if isinstance(h, types.FrameType)]
            if frames:
                chain.append("frames " + ", ".join(f.f_code.co_name for f in frames))
                holders = [h for h in holders if not isinstance(h, types.FrameType)]
                if not holders:
                    break
            holder = holders[0]
            seen.add(id(holder))
            if id(holder) in module_dicts:
                chain.append(f"module {module_dicts[id(holder)]}")
                break
            label = type(holder).__qualname__
            if isinstance(holder, types.FunctionType | types.CellType):
                label += f"[{getattr(holder, '__qualname__', '')}]"
            others = sorted({type(h).__qualname__ for h in holders[1:]})
            chain.append(f"{label}" + (f" (also {', '.join(others)})" if others else ""))
            current = holder
        chains.append(" <- ".join(chain))
    return chains


def _stack_steps() -> list[str]:
    """Every running function's local that is a step carrying a traced split, with that split's identity.

    A running function's locals are invisible to ``gc.get_referrers``; this names them, so a step a
    driver keeps across re-fits shows beside the one the march is applying.
    """
    lines = []
    frame = sys._getframe(1)
    while frame is not None:
        for name, value in frame.f_locals.items():
            split = getattr(getattr(value, "shift_policy", None), "split", None)
            if split is not None:
                import jax

                leaf = jax.tree.leaves(split)[0]
                lines.append(f"frame {frame.f_code.co_name}:{name} holds a split at {id(leaf):#x}")
        frame = frame.f_back
    return lines


class _Stop(Exception):
    """Raised by the step observer once the measured steps are done."""


def _load_bfs3d():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    import compare  # imports aquaflux (x64) and reads case.yaml

    return compare


def main(
    load_case=_load_bfs3d,
    recorder: Recorder | None = None,
    march_log: Path = HERE / "memory_attribution.march.log",
    after_march=None,
) -> None:
    """Run the measured march.

    ``load_case`` returns a step case's ``compare`` module (imported inside the ``imports`` stage);
    ``after_march(recorder)``, if given, runs once the measured steps are done, before the programs'
    memory analyses.
    """
    recorder = recorder or Recorder()
    recorder.start()
    header = _header()
    recorder.extra["header"] = header
    print(
        "| stage | s | RSS | ΔRSS | stage peak | HWM | traced | traced peak | device | exec | "
        "malloc free | tracer |\n|---|---|---|---|---|---|---|---|---|---|---|---|",
        flush=True,
    )
    with recorder.stage("imports"):
        compare = load_case()
        import jax

    header["mesh"] = str(getattr(compare, "MESH", "the case file's"))
    header["jax"] = jax.__version__
    print(f"jax {jax.__version__}, mesh {header['mesh']}", flush=True)

    captured = _instrument(recorder, compare)
    steps: list[int] = []

    def on_step(report):
        steps.append(1)
        with recorder.stage(f"step {len(steps)} ends (census)", census=True):
            pass
        if HOLDERS:
            for chain in _holders(HOLDERS):
                print(f"    {HOLDERS} held by: {chain}", flush=True)
            for line in _stack_steps():
                print(f"    {line}", flush=True)
        if len(steps) == STEPS - 1:
            captured["want_step"] = True
        if len(steps) >= STEPS:
            raise _Stop

    # A one-step run captures its only step's arguments; a longer one captures the last step's, armed
    # by `on_step` after the step before it.
    captured["want_step"] = STEPS == 1
    with contextlib.suppress(_Stop), recorder.stage("march (to the last measured step)"):
        compare.solve_aquaflux(log_path=march_log, on_step=on_step)
    peak = _peak_bytes()
    print(
        f"peak (VmHWM, or the peak footprint on macOS) {peak / GIB:.2f} GiB after {len(steps)} steps",
        flush=True,
    )

    recorder.extra["peak"] = peak
    if PROGRAMS:
        recorder.extra["programs_by_temporaries"] = _programs_by_temporaries()
    recorder.save()
    if after_march is not None:
        after_march(recorder)
        recorder.save()
    recorder.extra["programs"] = _program_sizes(captured, compare.build_case())
    recorder.save()
    print(f"record written to {recorder.record}", flush=True)


if __name__ == "__main__":
    main()
