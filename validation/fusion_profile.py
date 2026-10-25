"""Where a step case's compiled residual, its Jacobian-vector product and the probe's batched product spend their time, kernel by kernel.

The Krylov matvec of the coupled march is ``jax.jvp`` of the coupled residual, and the coloured probe
that refits the preconditioner is a batch of the same products (``batched_jacobian_matvec`` over the
probe's own, possibly narrowed, residual), so those compiled programs are the ones whose kernels matter.
This builds a step case (``FUSION_CASE``: ``pitzdaily`` or ``bfs3d``, by its own ``build_case``) at a state,
compiles each program with ``jax.jit``, and joins two things per kernel:

* **the compiled HLO** -- each entry-computation instruction's name, fusion kind, whether the CPU
  backend split it across threads (an ``outer_dimension_partitions`` backend config), its output
  shape and element count, the opcode at the root of its fused computation, and an estimate of the
  bytes it touches (operand plus output buffers, the quantity the backend's partitioning cost model
  divides by its per-thread minimum);
* **a profiler trace** (``jax.profiler.trace``) over several warm calls, in which each kernel the CPU
  thunk executor runs is one event carrying its ``hlo_op``.

The executor runs independent kernels concurrently, so on several cores an event's duration includes
waiting for a core, and it is not the kernel's cost. Each kernel's cost is therefore taken from a second
trace with every thread of the process pinned to one core (the compiled program is the same, since
partitioning is decided at compile time), counting only *solo* events that no other kernel overlapped.
The trace on all cores supplies, per kernel, its *time alone*: how long it was the only kernel in
flight. That is the serialized path, and the only time a kernel left on one thread leaves the other
cores idle, so it bounds what splitting that kernel could buy.

It reports the time in partitioned against unpartitioned kernels, the unpartitioned ones classified by
why the backend left them on one thread, the time by the opcode at each kernel's root (so two cases'
programs can be compared by kind of work), how concentrated that time is, and the top kernels by
one-core time and by time alone.

The backend's rule (``ParallelTaskAssignment`` in the CPU compiler): a memory-bound kernel (flops per
byte at most 1) is split into ``min(ceil(sqrt(cores)), bytes / 256 KiB)`` parts, a compute-bound one
into ``min(cores, (flops + 10 bytes) / 100000)``; custom fusions are never split. A kernel touching
under 512 KiB is therefore never split, whatever its root.

``FUSION_ARM=lapack`` restores the batched LU inverse of the multiple-correction first pass that the
closed-form adjugate replaced, as a check that the instrument attributes a known cost: that arm must
show its custom calls (``not a loop kernel (custom-call)``) carrying about the one-core difference
between the two arms.

Usage
-----
    FUSION_CASE=pitzdaily validation/run_case.sh validation/fusion_profile.py --wait
    FUSION_CASE=bfs3d FUSION_PROGRAMS=probe validation/run_case.sh validation/fusion_profile.py --wait

Environment: ``FUSION_CASE`` (default ``pitzdaily``); ``FUSION_PROGRAMS``, a comma list of ``residual``,
``jvp``, ``probe`` (default those three) and ``probe-fields``, the probe's one program per column field
(``batched_field_jacobian_matvec``, each seeded with its own field's first colours); ``FUSION_BATCH``, the probe's batch (default the library's
``PROBE_BATCH_SIZE``), seeded with the probe's own first colours; ``FUSION_STATE``, a ``.npz`` holding
``state`` (default the hybrid start); ``FUSION_CALLS`` traced calls (default 10), ``FUSION_REPEATS``
untraced timing calls (default 15), ``FUSION_TOP`` rows in each table (default 25), and
``FUSION_MMAP_THRESHOLD_MIB`` to fix glibc's mmap threshold (see ``_raise_mmap_threshold``): every timing
here is of one program called on its own, and on Linux a program whose temporary buffer exceeds 32 MiB
is page-faulted afresh on every such call -- the JVPs are, the residual is not. The per-call page-fault
count is printed so that cost is visible rather than folded into the kernels.
"""

from __future__ import annotations

import collections
import ctypes
import glob
import gzip
import itertools
import json
import os
import re
import resource
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    PROBE_BATCH_SIZE,
    batched_field_jacobian_matvec,
    batched_jacobian_matvec,
)
from aquaflux.turbulence import open_session  # noqa: E402
from hlo_text import COMPUTATION, INSTRUCTION, shape_bytes  # noqa: E402
from hlo_text import arrays as _hlo_arrays  # noqa: E402
from step_cases import load_case  # noqa: E402

CASE = os.environ.get("FUSION_CASE", "pitzdaily")
PROGRAMS = os.environ.get("FUSION_PROGRAMS", "residual,jvp,probe").split(",")
BATCH = int(os.environ.get("FUSION_BATCH", PROBE_BATCH_SIZE))
STATE = os.environ.get("FUSION_STATE")
ARM = os.environ.get("FUSION_ARM", "main")
CALLS = int(os.environ.get("FUSION_CALLS", "10"))
REPEATS = int(os.environ.get("FUSION_REPEATS", "15"))
TOP = int(os.environ.get("FUSION_TOP", "25"))
MMAP_THRESHOLD_MIB = os.environ.get("FUSION_MMAP_THRESHOLD_MIB")

#: ``mallopt`` parameters (glibc ``malloc.h``).
_M_TRIM_THRESHOLD, _M_MMAP_THRESHOLD = -1, -3

#: The CPU backend's per-thread minimum for a memory-bound kernel, in bytes touched.
L2_BYTES_PER_THREAD = 256 << 10


def _raise_mmap_threshold(mib):
    """Serve allocations below ``mib`` MiB from the heap, and keep freed heap memory mapped.

    glibc maps an allocation above its mmap threshold afresh and unmaps it on free. The threshold
    adapts upward as such blocks are freed, but not past 32 MiB on a 64-bit build, so a compiled
    program whose temporary buffer is larger maps and unmaps it on every call and takes a page fault
    on the first touch of every page. Fixing the threshold here (with the trim threshold above it, or
    freed memory at the top of the heap is returned to the kernel instead) is how that cost is
    removed for the comparison; it changes no arithmetic.
    """
    # Both settings are C ``int``s: a value past 2**31 - 1 wraps (a trim threshold of 4 GiB becomes 0,
    # which returns every freed page to the kernel and multiplies the faults instead).
    threshold = int(mib) << 20
    if not 0 < threshold < 2**31:
        raise SystemExit("FUSION_MMAP_THRESHOLD_MIB must be between 1 and 2047")
    libc = ctypes.CDLL("libc.so.6")
    if not (
        libc.mallopt(_M_MMAP_THRESHOLD, threshold) and libc.mallopt(_M_TRIM_THRESHOLD, 2**31 - 1)
    ):
        raise SystemExit("mallopt refused the thresholds")


def _restore_lapack_first_pass():
    """Put the batched LU inverse back into the multiple-correction first pass (the check arm)."""
    import aquaflux.schemes.multiple_correction as mc

    def first_pass(m1, weight, face_cells, geometry):
        return jnp.linalg.inv(m1 - mc.boundary_gradient_block(weight, face_cells, geometry))

    mc._boundary_condition_first_pass = first_pass


class Kernel:
    """One entry-computation instruction of the compiled program, as the HLO text describes it."""

    def __init__(self, name, shape, opcode, rest, shapes, roots):
        self.name = name
        self.opcode = opcode
        self.shape = shape.split("{")[0] if not shape.startswith("(") else shape
        arrays = _hlo_arrays(shape)
        self.elements = max((n for _, n in arrays), default=0)
        kind = re.search(r"kind=(k[A-Za-z]+)", rest)
        self.kind = kind.group(1) if kind else opcode
        self.partitions = re.search(r'outer_dimension_partitions":\["?(\d+)', rest)
        self.partitions = int(self.partitions.group(1)) if self.partitions else 1
        called = re.search(r"calls=%(\S+?)[,\s]", rest)
        self.root = roots.get(called.group(1), opcode) if called else opcode
        operands = re.findall(r"%([\w.\-]+)", rest.split(")", 1)[0])
        self.bytes = shape_bytes(shape) + sum(shapes.get(o, 0) for o in operands)
        self.time = 0.0
        self.alone = 0.0

    @property
    def why_single(self):
        """Why the backend left this kernel on one thread, as far as the HLO says."""
        if self.partitions > 1:
            return "partitioned"
        if self.kind == "kCustom":
            return "custom fusion"
        if self.opcode not in ("fusion", "copy", "transpose", "broadcast", "concatenate"):
            return f"not a loop kernel ({self.opcode})"
        if self.bytes < 2 * L2_BYTES_PER_THREAD:
            return "under 512 KiB touched"
        if self.root in ("scatter", "dynamic-update-slice"):
            return "scatter-rooted"
        if self.root == "reduce":
            return "reduction-rooted"
        return "other"


def _parse(hlo):
    """The entry computation's kernels, each with its fused computation's root opcode."""
    roots, current, entry, shapes, local = {}, None, [], {}, {}
    for line in hlo.splitlines():
        if line.startswith("ENTRY"):
            current, local = "ENTRY", {}
            continue
        heading = COMPUTATION.match(line)
        if heading:
            current, local = heading.group(1), {}
            continue
        m = INSTRUCTION.match(line)
        if not m:
            continue
        name, shape, opcode, rest = m.groups()
        if line.lstrip().startswith("ROOT") and current != "ENTRY":
            if opcode == "tuple":
                operands = re.findall(r"%([\w.\-]+)", rest)
                roots[current] = "tuple(" + ",".join(sorted({local[o] for o in operands})) + ")"
            else:
                roots[current] = opcode
        if current == "ENTRY":
            entry.append((name, shape, opcode, rest))
            shapes[name] = shape_bytes(shape)
        else:
            local[name] = opcode
    return [Kernel(n, s, o, r, shapes, roots) for n, s, o, r in entry if o not in _FREE]


#: Instructions that run no kernel.
_FREE = {"parameter", "constant", "get-tuple-element", "tuple", "bitcast"}


def _pin_to_one_core():
    """Restrict every thread of this process (the compiled program's pool included) to one core."""
    core = min(os.sched_getaffinity(0))
    for task in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(task), {core})
        except OSError:
            pass  # a thread that exited between the listing and the call


def _unpin(cores):
    for task in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(task), cores)
        except OSError:
            pass


def _trace(fn, args, module):
    """Kernel intervals of ``CALLS`` traced calls: one list of ``(op, start, end)`` per call, in us.

    Calls run back to back and each blocks until its result is ready, so call ``i``'s kernels are the
    ones that end after call ``i - 1``'s executor finished waiting and no later than call ``i``'s did.
    """
    jax.block_until_ready(fn(*args))
    with tempfile.TemporaryDirectory() as directory:
        with jax.profiler.trace(directory):
            for _ in range(CALLS):
                jax.block_until_ready(fn(*args))
        path = glob.glob(f"{directory}/**/*.trace.json.gz", recursive=True)[0]
        events = json.load(gzip.open(path))["traceEvents"]
    kernels = sorted(
        (e["args"]["hlo_op"], e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("ph") == "X"
        and e.get("args", {}).get("hlo_module") == module
        and "hlo_op" in e["args"]
    )
    ends = sorted(
        e["ts"] + e["dur"]
        for e in events
        if e.get("name") == "ThunkExecutor::Execute (wait for completion)"
    )
    calls, previous = [], -float("inf")
    for end in ends:
        calls.append([k for k in kernels if previous < k[2] <= end + 1.0])
        previous = end + 1.0
    return [c for c in calls if c]


def _occupancy(call):
    """``(span, summed event time, time with N kernels in flight, solo durations, time alone)`` of a call.

    A kernel's event is *solo* when no other kernel's event overlaps it. On one core the executor's
    threads are time-sliced, so an overlapped event includes the slices its neighbours ran in, at the
    scheduler's granularity rather than in proportion to the work; only a solo event is the kernel's
    own time. *Time alone* is, per op, how long it was the only kernel in flight: on several cores
    that is the program's serialized path, the only time an unsplit kernel leaves cores idle.
    """
    points = sorted({t for _, s, f in call for t in (s, f)})
    histogram, alone = collections.Counter(), collections.Counter()
    starts = sorted(call, key=lambda k: k[1])
    active, j = [], 0
    for left, right in itertools.pairwise(points):
        while j < len(starts) and starts[j][1] <= left:
            active.append(starts[j])
            j += 1
        active = [k for k in active if k[2] > left]
        histogram[len(active)] += right - left
        if len(active) == 1:
            alone[active[0][0]] += right - left
    solo, latest = {}, -float("inf")
    for i, (op, begin, end) in enumerate(starts):
        later = starts[i + 1][1] if i + 1 < len(starts) else float("inf")
        if begin >= latest and end <= later:
            solo[op] = end - begin
        latest = max(latest, end)
    span = points[-1] - points[0]
    return span, sum(f - s for _, s, f in call), histogram, solo, alone


def _best(fn, args):
    """Best wall time of ``REPEATS`` warm calls in ms, and the minor page faults per call."""
    jax.block_until_ready(fn(*args))
    best = float("inf")
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    for _ in range(REPEATS):
        started = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - started)
    faults = (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults) / REPEATS
    return best * 1e3, faults


def _concentration(times, fraction):
    """How many of ``times`` (largest first) it takes to reach ``fraction`` of their sum."""
    total, running = sum(times), 0.0
    for n, t in enumerate(sorted(times, reverse=True), 1):
        running += t
        if running >= fraction * total:
            return n
    return len(times)


def _report(label, fn, args, module, tangents=1):
    compiled = fn.lower(*args).compile()
    kernels = _parse(compiled.as_text())
    temp = compiled.memory_analysis().temp_size_in_bytes / 2**20
    cores = os.sched_getaffinity(0)
    wall, faults = _best(fn, args)
    parallel = [_occupancy(c) for c in _trace(fn, args, module)]
    _pin_to_one_core()
    try:
        serial_wall, _ = _best(fn, args)
        serial = [_occupancy(c) for c in _trace(fn, args, module)]
    finally:
        _unpin(cores)
    samples = collections.defaultdict(list)
    for *_, solo, _ in serial:
        for name, t in solo.items():
            samples[name].append(t)
    unmeasured = []
    for k in kernels:
        if samples.get(k.name):
            k.time = float(np.median(samples[k.name]))
        elif k.opcode not in _FREE:
            unmeasured.append(k)
    print(f"\n=== {label} ({ARM} arm, {CASE}) ===", flush=True)
    print(f"temporary buffer {temp:.1f} MiB; {faults:.0f} minor page faults per call")
    print(
        f"bytes touched per call (operands + outputs, summed over kernels): "
        f"{sum(k.bytes for k in kernels) / 1e9:.2f} GB; {wall / tangents:.2f} ms per tangent"
    )
    print(
        f"untraced wall, best of {REPEATS}: {wall:.2f} ms on {len(cores)} cores, {serial_wall:.2f} ms on one "
        f"core; parallel efficiency {serial_wall / (len(cores) * wall):.0%}"
    )
    for name, runs in (("one core", serial), (f"{len(cores)} cores", parallel)):
        span = np.median([r[0] for r in runs]) / 1e3
        busy = np.median([r[1] for r in runs]) / 1e3
        histogram = collections.Counter()
        for r in runs:
            histogram.update({n: v / len(runs) / 1e3 for n, v in r[2].items()})
        print(
            f"traced, {name}: kernel span {span:.2f} ms/call, kernel events summed {busy:.2f} ms/call; "
            "ms/call with N kernels in flight: "
            + ", ".join(f"{n}: {histogram[n]:.2f}" for n in sorted(histogram))
        )
    total = sum(k.time for k in kernels)
    custom = [k for k in kernels if k.opcode == "custom-call"]
    print(
        f"one-core solo kernel time {total / 1e3:.2f} ms/call over {len(kernels)} kernels "
        f"({len(unmeasured)} never ran solo in {CALLS} calls, counted as 0); custom calls "
        f"{len(custom)}, {sum(k.time for k in custom) / 1e3:.2f} ms/call"
    )
    groups = collections.defaultdict(list)
    for k in kernels:
        groups[k.why_single].append(k)
    print(
        f"{'class':34s} {'kernels':>7s} {'ms/call':>8s} {'share':>6s} {'half in':>7s} {'out elems':>10s}"
    )
    for cls, members in sorted(groups.items(), key=lambda kv: -sum(k.time for k in kv[1])):
        t = sum(k.time for k in members)
        print(
            f"{cls:34s} {len(members):7d} {t / 1e3:8.2f} {t / total:6.1%} "
            f"{_concentration([k.time for k in members], 0.5):7d} "
            f"{sum(k.elements for k in members) / 1e6:9.2f}M"
        )
    if tangents > 1:
        print(
            f"per tangent ({tangents} per call): {wall / tangents:.2f} ms on {len(cores)} cores, "
            f"{serial_wall / tangents:.2f} ms on one core, {total / 1e3 / tangents:.2f} ms solo kernel time"
        )
    roots = collections.defaultdict(list)
    for k in kernels:
        roots[k.root.split("(")[0] if k.opcode == "fusion" else k.opcode].append(k)
    print(
        f"\n{'root opcode':24s} {'kernels':>7s} {'ms/call':>8s} {'share':>6s} {'split':>6s} {'MiB touched':>11s}"
    )
    for root, members in sorted(roots.items(), key=lambda kv: -sum(k.time for k in kv[1])):
        t = sum(k.time for k in members)
        split = sum(k.time for k in members if k.partitions > 1)
        print(
            f"{root[:24]:24s} {len(members):7d} {t / 1e3:8.2f} {t / total:6.1%} "
            f"{split / max(t, 1e-9):6.0%} {sum(k.bytes for k in members) / 2**20:11.1f}"
        )
    for k in kernels:
        k.alone = sum(r[4].get(k.name, 0.0) for r in parallel) / len(parallel)
    span = np.median([r[0] for r in parallel])
    alone_single = sum(k.alone for k in kernels if k.partitions == 1)
    alone_split = sum(k.alone for k in kernels if k.partitions > 1)
    print(
        f"{len(cores)}-core trace, one kernel in flight: unpartitioned {alone_single / 1e3:.2f} ms/call "
        f"({alone_single / span:.0%} of the span), partitioned {alone_split / 1e3:.2f} ms/call; the "
        f"unpartitioned time alone is half in the largest "
        f"{_concentration([k.alone for k in kernels if k.partitions == 1], 0.5)} kernels"
    )
    single = [k for k in kernels if k.partitions == 1]
    print(
        f"unpartitioned: {len(single)} kernels, {sum(k.time for k in single) / 1e3:.2f} ms/call "
        f"({sum(k.time for k in single) / total:.0%}); half of it in the largest "
        f"{_concentration([k.time for k in single], 0.5)}, the largest one "
        f"{max(k.time for k in single) / total:.1%} of the program"
    )
    tables = (
        ("by one-core time, all kernels", kernels, "time"),
        ("by one-core time, unpartitioned", single, "time"),
        (f"by time alone on {len(cores)} cores, unpartitioned", single, "alone"),
    )
    for title, rows, key in tables:
        print(f"\ntop {TOP} {title}:")
        print(
            f"{'us/call':>8s} {'share':>6s} {'alone':>7s} {'parts':>5s} {'MiB':>6s}  "
            f"{'root':24s} {'shape':22s} name / why"
        )
        for k in sorted(rows, key=lambda k: -getattr(k, key))[:TOP]:
            print(
                f"{k.time:8.1f} {k.time / total:6.1%} {k.alone:7.1f} {k.partitions:5d} "
                f"{k.bytes / 2**20:6.2f}  {k.root[:24]:24s} {k.shape[:22]:22s} {k.name} / {k.why_single}"
            )
    return kernels


def _state(coupled):
    """The state to profile at: ``FUSION_STATE``'s, or the hybrid start; refused unless its residual is finite."""
    if STATE:
        with np.load(STATE) as data:
            state = jnp.asarray(data["state"])
    else:
        state = coupled.state_from_physical(*hybrid_initialize(coupled))
    if not bool(jnp.all(jnp.isfinite(coupled.residual(state)))):
        raise SystemExit(
            "the residual at this state is not finite; nothing here would mean anything"
        )
    return state


def main():
    if MMAP_THRESHOLD_MIB is not None:
        _raise_mmap_threshold(MMAP_THRESHOLD_MIB)
    if ARM == "lapack":
        _restore_lapack_first_pass()
    elif ARM != "main":
        raise SystemExit(f"FUSION_ARM must be 'main' or 'lapack', not {ARM!r}")
    unknown = set(PROGRAMS) - {"residual", "jvp", "probe", "probe-fields"}
    if unknown:
        raise SystemExit(f"FUSION_PROGRAMS names no such program: {sorted(unknown)}")
    print(
        f"[configuration] case {CASE}, state {STATE or 'the hybrid start'}, programs "
        f"{','.join(PROGRAMS)}, probe batch {BATCH}; jax {jax.__version__}, backend "
        f"{jax.default_backend()}, {os.cpu_count()} cores ({len(os.sched_getaffinity(0))} usable), "
        f"mmap threshold {MMAP_THRESHOLD_MIB or 'glibc default'} MiB, allocator settings "
        + (
            " ".join(
                f"{name}={os.environ[name]}"
                for name in sorted(os.environ)
                if name.startswith("MALLOC_") or name in ("LD_PRELOAD", "XLA_FLAGS")
            )
            or "none"
        ),
        flush=True,
    )
    case = load_case(CASE)
    coupled = case.build_case()["coupled"]
    state = _state(coupled)
    print(f"{state.size} unknowns", flush=True)
    params, static = eqx.partition(coupled, eqx.is_array)

    def residual(p, s):
        return eqx.combine(p, static).residual(s)

    def product(p, s, t):
        return jax.jvp(lambda x: eqx.combine(p, static).residual(x), (s,), (t,))[1]

    if "residual" in PROGRAMS:
        _report("residual", jax.jit(residual), (params, state), "jit_residual")
    if "jvp" in PROGRAMS:
        tangent = jnp.asarray(np.random.default_rng(0).standard_normal(state.size))
        _report("JVP", jax.jit(product), (params, state, tangent), "jit_product")
    if "probe" in PROGRAMS:
        # The probe's own program: the session's probe residual (narrowed as the case's probe spec says),
        # the library's batched product, and seeds that are the plan's first colours.
        session = open_session(case.SOLVER.settings()["preconditioner"], coupled)
        probe = session._probe_for()
        plan = probe.plan
        print(f"{plan.n_cells} cells, {plan.n_fields} fields, {plan.n_probes} probes", flush=True)
        probed_params, probed_static = eqx.partition(
            probe.narrow(session.problem.assembler), eqx.is_array
        )
        seeds = plan.seed_block(0, min(BATCH, plan.n_probes), rows=BATCH)

        def probe_batch(p, s, block):
            return batched_jacobian_matvec(eqx.combine(p, probed_static), s, block)

        _report(
            f"probe's batched JVP, batch {BATCH}",
            jax.jit(probe_batch),
            (probed_params, jax.lax.stop_gradient(state), seeds),
            "jit_probe_batch",
            tangents=BATCH,
        )
    if "probe-fields" in PROGRAMS:
        # The probe's per-column-field programs, as the session runs them: each field's own first colours,
        # its residual differentiated in that field alone.
        session = open_session(case.SOLVER.settings()["preconditioner"], coupled)
        probe = session._probe_for()
        plan = probe.plan
        probed_params, probed_static = eqx.partition(
            probe.narrow(session.problem.assembler), eqx.is_array
        )
        layout = session.problem.layout
        for field in range(plan.n_fields):
            start = plan.probe_base[field]
            stop = start + min(BATCH, plan.n_colours[field])
            seeds = plan.field_seeds(start, stop, rows=BATCH)

            def field_batch(p, s, block, field=field):
                return batched_field_jacobian_matvec(
                    eqx.combine(p, probed_static), layout, s, field, block
                )

            _report(
                f"probe's column field {field} program, batch {BATCH}",
                jax.jit(field_batch),
                (probed_params, jax.lax.stop_gradient(state), seeds),
                "jit_field_batch",
                tangents=BATCH,
            )


if __name__ == "__main__":
    main()
