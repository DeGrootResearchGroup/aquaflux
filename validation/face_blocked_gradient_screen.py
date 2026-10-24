"""Would reconstructing the gradient in blocks of faces bound the coloured probe's temporary, and at what cost per tangent?

The coloured Jacobian probe runs a batch of Jacobian-vector products of the coupled residual, compiled
as one program whose temporary buffer scales with the face count: on bfs3d (23 040 cells, 71 872 faces)
its batch of 8 tangents holds ~829 MiB, and the buffer is faulted in afresh on every call. This screen
asks whether forming the face sums in blocks -- a ``lax.scan`` over face ranges, each range scattered
into a carried per-cell accumulator -- bounds that temporary, and what it costs per tangent.

**Where the blocks go, and why there.** Before choosing, the probe program's buffers live at its peak
were attributed to the source lines that create them (:func:`peak_owners`, from XLA's buffer
assignment and live ranges). On bfs3d at the hybrid start the peak sits inside the default gradient
scheme's Green--Gauss face loop (``schemes/multiple_correction.py::_green_gauss`` and its conservative
scatter, ~505 of 829 MiB, the velocity gradient's second pass with its tangents), with ~164 MiB of
compiler-inserted copies beside it and ~15 MiB in the flux operators. Blocking the flux assembly alone
could therefore save at most that ~15 MiB at the peak, so the blocked arm blocks the gradient's face
loop and leaves the flux assembly whole-mesh. Every arm re-runs the attribution, so a peak that
moves into the flux operators once the gradient is blocked is visible.

**The blocked arm (screen-only, never shipped).** :func:`blocked_green_gauss` replaces the module's
``_green_gauss`` -- the one function both reconstruction passes run their face loop through -- with a
scan over ``B`` equal face ranges, each calling the ORIGINAL ``_green_gauss`` on that range's own
``FaceCellConnectivity`` (built through its constructor) and slices of the face arrays. No formula is
copied; only the chunking is new. It is installed after the case is built, so the correction matrices
bound at build are byte-identical between arms. This wrapper is deleted with the screen's verdict or
replaced by a library design; it is not a library path.

**Arms.** ``whole/8``, ``whole/4`` (the whole-mesh assembly at batch 8 and 4), and ``blocked-B/8``
for each ``B`` in ``SCREEN_BLOCKS``. Per arm: the compiled residual's and batched JVP's
``memory_analysis().temp_size_in_bytes``; the owners of the JVP's peak; time per tangent as the MEAN
over the number of batched calls a whole-state re-fit makes (``ceil(n_probes / batch)``), on the
re-fit's own seed sequence, after one warm call; minor faults per call; ``VmRSS``/``VmHWM`` (process-wide
and monotone, so they show growth only); and the residual's and the first seed block's JVP agreement
with ``whole/8`` as ``max |delta| / max |reference|`` per field. Agreement to rounding, not bit
identity, is expected: a blocked sum adds in a different order.

**Pre-registered pass bar (written before the first run).** A face-blocked assembly is worth building
only if EITHER (a) at a per-tangent time no worse than ``whole/8`` (within 5 %) its batch-8 JVP
temporary is below ``whole/4``'s, measured in the same run; OR (b) it is faster per tangent than
``whole/4`` at a smaller temporary than ``whole/4``'s. Halving the batch is the cheap way to halve the
temporary; a blocked assembly must beat it.

Usage::

    validation/run_case.sh validation/face_blocked_gradient_screen.py --wait

Environment: ``SCREEN_CASE`` (``bfs3d``, default, or ``pitzdaily``); ``SCREEN_STATE``, a ``.npz``
holding ``state`` (default the hybrid start); ``SCREEN_ARMS`` (default
``whole/8,whole/4,blocked-2/8,blocked-4/8,blocked-8/8,blocked-16/8``); ``SCREEN_CALLS`` overrides the
number of timed calls per arm. Allocator settings are reported as found (glibc defaults unless set).
"""

from __future__ import annotations

import collections
import glob
import math
import os
import re
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# The buffer attribution reads XLA's dumps, so the dump directory must be set before JAX starts.
DUMP = tempfile.mkdtemp(prefix="face_blocked_screen_")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + f" --xla_dump_to={DUMP} --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*screen_.*"
).strip()

import aquaflux  # noqa: E402,F401  (enables x64)
import aquaflux.schemes.multiple_correction as multiple_correction  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.mesh import FaceCellConnectivity  # noqa: E402
from aquaflux.precondition.materialized_session import batched_jacobian_matvec  # noqa: E402
from aquaflux.turbulence import open_session  # noqa: E402
from step_cases import load_case  # noqa: E402

CASE = os.environ.get("SCREEN_CASE", "bfs3d")
STATE = os.environ.get("SCREEN_STATE")
ARMS = os.environ.get(
    "SCREEN_ARMS", "whole/8,whole/4,blocked-2/8,blocked-4/8,blocked-8/8,blocked-16/8"
).split(",")
CALLS = os.environ.get("SCREEN_CALLS")

WHOLE_GREEN_GAUSS = multiple_correction._green_gauss


def blocked_green_gauss(blocks: int):
    """The module's Green--Gauss sum, formed over ``blocks`` equal face ranges in a ``lax.scan``.

    Each range calls the original function on its own connectivity and face slices; the per-range
    cell sums are added into a carried accumulator. The last range is padded with zero-area
    boundary faces.
    """

    def green_gauss(cell_values, face_values, factor, face_cells, area, geometry):
        n_faces = face_cells.n_faces
        size = -(-n_faces // blocks)
        pad = size * blocks - n_faces

        def ranges(array, fill):
            # A padding face is a zero-area boundary face of cell 0: it adds nothing to any sum.
            array = jnp.broadcast_to(array, (n_faces, *array.shape[1:]))
            array = jnp.pad(array, [(0, pad)] + [(0, 0)] * (array.ndim - 1), constant_values=fill)
            return array.reshape(blocks, size, *array.shape[1:])

        def body(total, block):
            owner, neighbour, values, g, a = block
            part = FaceCellConnectivity(
                owner=owner, neighbour=neighbour, n_cells=face_cells.n_cells
            )
            return total + WHOLE_GREEN_GAUSS(cell_values, values, g, part, a, geometry), None

        first = jax.eval_shape(
            WHOLE_GREEN_GAUSS, cell_values, face_values, factor, face_cells, area, geometry
        )
        total, _ = jax.lax.scan(
            body,
            jnp.zeros(first.shape, first.dtype),
            (
                ranges(face_cells.owner, 0),
                ranges(face_cells.neighbour, -1),
                ranges(face_values, 0),
                ranges(factor, 0),
                ranges(area, 0),
            ),
        )
        return total

    return green_gauss


# --- the peak attribution: which source lines own the buffers live at the temporary's peak ---------


def _numbered(text: str, header: str) -> dict[int, str]:
    rows = {}
    for line in text.split(f"\n{header}\n", 1)[1].split("\n"):
        match = re.match(r"(\d+) (.*)$", line)
        if not match:
            break
        rows[int(match.group(1))] = match.group(2)
    return rows


def peak_owners(module: str, top: int = 8) -> tuple[float, list[tuple[float, str]]]:
    """The temporary's peak (MiB) and the largest owners of the buffers live there.

    Reads the module's optimized HLO (instruction -> stack frame), buffer assignment (each value's
    size and offset in the preallocated temporary) and live ranges, finds the instruction at which
    the union of live byte ranges is largest, and sums the live values by their innermost
    ``aquaflux`` frame. Values sharing an offset at the same time are each counted, so the owners can
    sum to slightly more than the peak.
    """
    paths = {
        kind: glob.glob(f"{DUMP}/*.{module}.cpu_after_optimizations{kind}.txt")
        for kind in ("", "-buffer-assignment", "-live-range")
    }
    if not all(paths.values()):
        return float("nan"), [(0.0, f"no dump for {module}")]
    hlo, assignment, live_range = (open(sorted(p)[-1]).read() for p in paths.values())
    files = {k: v.strip('"') for k, v in _numbered(hlo, "FileNames").items()}
    functions = {k: v.strip('"') for k, v in _numbered(hlo, "FunctionNames").items()}
    locations = {}
    for k, v in _numbered(hlo, "FileLocations").items():
        m = re.search(r"file_name_id=(\d+) function_name_id=(\d+) line=(\d+)", v)
        locations[k] = (files[int(m.group(1))], functions[int(m.group(2))], m.group(3))
    frames = {}
    for k, v in _numbered(hlo, "StackFrames").items():
        m = re.search(r"file_location_id=(\d+) parent_frame_id=(\d+)", v)
        frames[k] = (int(m.group(1)), int(m.group(2)) - 1)
    frame_of = {}
    for m in re.finditer(r"^\s*(?:ROOT )?%(\S+) = .*?stack_frame_id=(\d+)", hlo, re.M):
        frame_of.setdefault(m.group(1), int(m.group(2)))

    def owner(name):
        frame, seen = frame_of.get(name, -1), 0
        while frame in frames and seen < 300:
            location, parent = frames[frame]
            path, function, line = locations[location]
            if "/aquaflux/aquaflux/" in path or path.endswith(Path(__file__).name):
                return f"{path.rsplit('/aquaflux/', 1)[-1]}:{function}:{line}"
            if parent <= 0 or parent == frame:
                break
            frame, seen = parent, seen + 1
        return "(no source frame: a compiler-inserted copy or layout change)"

    temp = re.search(
        r"^allocation \d+: size \d+, page 0, preallocated-temp:\n((?: value:.*\n)+)",
        assignment,
        re.M,
    )
    ranges = {
        (m.group(1), m.group(2)): (int(m.group(3)), int(m.group(4)))
        for m in re.finditer(r"^\s+(\S+?)\{([^}]*)\}:(\d+)-(\d+)$", live_range, re.M)
    }
    live = []
    for m in re.finditer(r"value: <\d+ (\S+) @(\S+)> \(size=(\d+),offset=(\d+)\)", temp.group(1)):
        name, index = m.group(1), m.group(2)
        span = ranges.get((name, "" if index == "0" else index.strip("{}"))) or ranges.get(
            (name, "")
        )
        if span:
            live.append((name, int(m.group(3)), int(m.group(4)), span))
    best, at = 0, 0
    for t in range(max(span[1] for *_, span in live) + 1):
        total, end = 0, -1
        for lo, hi in sorted((o, o + s) for _, s, o, span in live if span[0] <= t <= span[1]):
            if hi > end:
                total, end = total + hi - max(lo, end), hi
        if total > best:
            best, at = total, t
    owners = collections.Counter()
    for name, size, _, span in live:
        if span[0] <= at <= span[1]:
            owners[owner(name)] += size
    return best / 2**20, [(v / 2**20, k) for k, v in owners.most_common(top)]


# --- measurement --------------------------------------------------------------------------------


def _status() -> str:
    try:
        with open("/proc/self/status") as handle:
            lines = dict(line.split(":", 1) for line in handle if ":" in line)
        return (
            f"VmRSS {int(lines['VmRSS'].split()[0]) / 2**20:.2f} GiB, "
            f"VmHWM {int(lines['VmHWM'].split()[0]) / 2**20:.2f} GiB"
        )
    except OSError:
        return "unavailable"


def _relative(a, b, layout_fields: int) -> float:
    """``max |a - b| / max |b|`` per field block of the flat state, worst field."""
    a = np.asarray(a).reshape(*a.shape[:-1], layout_fields, -1)
    b = np.asarray(b).reshape(*b.shape[:-1], layout_fields, -1)
    axes = tuple(i for i in range(a.ndim) if i != a.ndim - 2)
    return float(np.max(np.max(np.abs(a - b), axis=axes) / np.max(np.abs(b), axis=axes)))


def main() -> None:
    print(
        f"[configuration] case {CASE}, state {STATE or 'the hybrid start'}, arms {','.join(ARMS)}; "
        f"jax {jax.__version__}, {os.cpu_count()} cores; allocator settings "
        + (
            " ".join(
                f"{n}={os.environ[n]}"
                for n in sorted(os.environ)
                if n.startswith("MALLOC_") or n in ("LD_PRELOAD", "GLIBC_TUNABLES")
            )
            or "none (glibc defaults)"
        ),
        flush=True,
    )
    case = load_case(CASE)
    coupled = case.build_case()["coupled"]
    if STATE:
        with np.load(STATE) as data:
            state = jnp.asarray(data["state"])
    else:
        state = coupled.state_from_physical(*hybrid_initialize(coupled))
    state = jax.lax.stop_gradient(state)
    session = open_session(case.SOLVER.settings()["preconditioner"], coupled)
    probe = session._probe_for()
    plan = probe.plan
    n_fields = plan.n_fields
    params, static = eqx.partition(probe.narrow(session.problem.assembler), eqx.is_array)
    print(
        f"{plan.n_cells} cells, {coupled.momentum.mesh.n_faces} faces, {n_fields} fields, "
        f"{plan.n_probes} probes; after the case build and probe plan: {_status()}",
        flush=True,
    )

    reference = {}
    rows = []
    for arm in ARMS:
        kind, batch = arm.split("/")
        batch = int(batch)
        blocks = int(kind.split("-")[1]) if kind.startswith("blocked") else 1
        multiple_correction._green_gauss = (
            WHOLE_GREEN_GAUSS if blocks == 1 else blocked_green_gauss(blocks)
        )
        tag = f"b{blocks}_n{batch}"

        def residual(p, s):
            return eqx.combine(p, static).residual(s)

        def product(p, s, block):
            return batched_jacobian_matvec(eqx.combine(p, static), s, block)

        # A fresh name per arm: a fresh jit cache entry (the patched function is read at trace time)
        # and its own dump to attribute.
        residual.__name__ = f"screen_residual_{tag}"
        product.__name__ = f"screen_probe_{tag}"
        residual_program = jax.jit(residual).lower(params, state).compile()
        seeds = plan.seed_block(0, min(batch, plan.n_probes), rows=batch)
        start = time.perf_counter()
        product_program = jax.jit(product).lower(params, state, seeds).compile()
        compile_s = time.perf_counter() - start
        residual_temp = residual_program.memory_analysis().temp_size_in_bytes / 2**20
        product_temp = product_program.memory_analysis().temp_size_in_bytes / 2**20

        r = residual_program(params, state)
        jv = product_program(params, state, plan.seed_block(0, min(8, plan.n_probes), rows=batch))
        if arm == ARMS[0]:
            reference["r"] = r
        if batch == 8 and "jv" not in reference:
            reference["jv"] = jv
        agreement_r = _relative(r, reference["r"], n_fields)
        agreement_jv = _relative(jv, reference["jv"], n_fields) if batch == 8 else float("nan")

        calls = int(CALLS) if CALLS else math.ceil(plan.n_probes / batch)
        blocks_of_seeds = [
            plan.seed_block(
                c * batch % plan.n_probes,
                min(c * batch % plan.n_probes + batch, plan.n_probes),
                rows=batch,
            )
            for c in range(calls)
        ]
        jax.block_until_ready(product_program(params, state, blocks_of_seeds[0]))  # warm
        wall, faults = 0.0, 0
        for block in blocks_of_seeds:
            jax.block_until_ready(block)
            f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
            t0 = time.perf_counter()
            jax.block_until_ready(product_program(params, state, block))
            wall += time.perf_counter() - t0
            faults += resource.getrusage(resource.RUSAGE_SELF).ru_minflt - f0
        per_tangent = wall / calls / batch * 1e3
        peak, owners = peak_owners(f"jit_screen_probe_{tag}")
        row = (
            arm,
            residual_temp,
            product_temp,
            peak,
            per_tangent,
            faults / calls,
            calls,
            agreement_r,
            agreement_jv,
            compile_s,
        )
        rows.append(row)
        print(
            f"\n[{arm}] residual temp {residual_temp:.1f} MiB, batched JVP temp {product_temp:.1f} MiB "
            f"(attributed peak {peak:.1f}), {per_tangent:.2f} ms per tangent over {calls} calls, "
            f"{faults / calls / 1e3:.1f}k faults per call, compile {compile_s:.0f} s; agreement with "
            f"{ARMS[0]}: residual {agreement_r:.1e}, JVP {agreement_jv:.1e}; {_status()}",
            flush=True,
        )
        for size, where in owners:
            print(f"    {size:7.1f} MiB  {where}", flush=True)
        del residual_program, product_program, blocks_of_seeds
        jax.clear_caches()

    multiple_correction._green_gauss = WHOLE_GREEN_GAUSS
    print(
        "\n| arm | residual temp MiB | batched JVP temp MiB | ms per tangent | faults per call | "
        "calls | residual agreement | JVP agreement |\n|---|---|---|---|---|---|---|---|"
    )
    for arm, rt, pt, _, ms, fl, calls, ar, aj, _ in rows:
        print(
            f"| {arm} | {rt:.1f} | {pt:.1f} | {ms:.2f} | {fl / 1e3:.1f}k | {calls} | {ar:.1e} | "
            f"{aj:.1e} |"
        )
    shutil.rmtree(DUMP, ignore_errors=True)


if __name__ == "__main__":
    main()
