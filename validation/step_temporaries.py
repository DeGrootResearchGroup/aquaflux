"""Which buffers are the march step program's temporaries: the shipped field split against no preconditioner.

``memory_analysis()`` of the compiled march step (``aquaflux.solve.march._march_step``) gives one
number for its temporary buffer. This names what fills it. On a step case (``STEP_TEMPS_CASE``:
``bfs3d`` or ``pitzdaily``, by its own ``compare.solve_aquaflux``) it:

1. runs the case's own march with ``_march_step`` replaced by a stub that records the arguments the
   march hands its first step and stops before running it. The session's fit and its forced first
   re-fit run as in the march; no step runs. The state is the march's first: the ramp's anchor station
   at the hybrid start;
2. compiles the step from those arguments in each arm (``STEP_TEMPS_ARMS``, default both):

   * ``split`` -- the step as the march built it, the case file's field split applied in each Krylov
     iteration;
   * ``identity`` -- the same step with its shift policy wrapped so its ``ShiftTerm`` carries the same
     diagonal and per-row shift scale and a ``make_preconditioner`` returning ``None``: the documented
     way a step solves unpreconditioned (``PseudoTransientStep`` / ``DualTimeStep`` then hand
     ``preconditioner=None`` to ``solve_linear``). Shift, line search, dual-time loop and Krylov
     settings are the split arm's; the split's arrays are still arguments (the adjoint factory holds
     them) but nothing reads them;

   * ``split-loop`` / ``identity-loop`` -- either arm with the true-residual ``lax.cond`` of the
     residual-stop GMRES (``solve/linear.py``, its only ``lax.cond``) run as a ``while_loop`` of at most
     one trip: while it is lowered, that module's ``jax`` is a proxy whose ``lax.cond`` is
     ``_cond_as_loop`` and whose every other name is JAX's own. Same arithmetic; the question is whether
     XLA, which does not move code out of a conditional branch, then hoists the branch's decode and
     linearization and shares them with the loop's. ``_check_cond_as_loop`` first solves a small system
     both ways and refuses to measure unless the operator applications (counted by a host callback) and
     the solution are identical;

   with the persistent compilation cache bypassed for those compiles (``_bypass_compilation_cache``): a
   cache hit compiles nothing and so dumps nothing, the dump options are not in the cache key, and JAX
   decides whether the cache is in use once per process, so the switch is the config and a reset of
   that decision (``jax._src.compilation_cache.reset_cache``, a private hook);
3. for each arm reads ``memory_analysis()`` (temp, argument, output, alias, code) and XLA's own dumps
   of the program (``--xla_dump_to``, set here before JAX starts, limited to the step's module): the
   buffer assignment (which values share the ``preallocated-temp`` allocation, at which offsets) and the
   live ranges (each value's interval in the flattened schedule). Values at one offset and size of the
   temp allocation are one slot; a slot is live while any of its values is. The **peak** is the
   schedule position with the most temp bytes live, and the table lists that peak's slots, largest
   first, until ``STEP_TEMPS_COVER`` (default 0.9) of the temp total is named, each with its HLO
   instruction, shape, opcode, ``op_name`` and the innermost ``aquaflux`` frames of its Python origin
   (the HLO metadata's stack frames);
4. compiles a single Jacobian-vector product (``jacobian_matvec``) and the residual of the step's own
   assembler at the same state and prints their temp sizes, as the record's 202 / 106 MiB (bfs3d) were
   taken, to check the method.

Two limits of the reading, found on the first runs: the "hoisted" flag also fires on an instruction from
a one-trip loop XLA inlined (it compares ``while/body`` counts, not operands), and a slot's site is the
metadata of whichever instruction survived XLA's merging (one entry-computation decode serves two call
sites but names one); a per-call-site claim needs the ``while`` carries traced.

**Owner of a slot (registered before the first run).** By the innermost ``aquaflux`` frame of its
instruction: ``aquaflux/precondition/`` -- the split's apply, and an ``s32``/``s64`` slot there is the
decoded or index arrays (``decode``); ``aquaflux/solve/linear.py`` -- the Krylov solve's basis and
vectors; ``line_search`` in a frame's function -- the line search; a frame in the residual's packages
(``turbulence``, ``flow``, ``schemes``, ``discretization``, ``boundary``, ``mesh``, ``properties``,
``vectors``) with ``jvp`` in the ``op_name`` -- the residual's JVP, without it -- a residual evaluation;
anything else -- the step's own bookkeeping, named by its frame.

**Reading rule (registered before the first run).** The split's share of the temporary is the split arm's
temp minus the identity arm's, read beside the sum of the peak slots owned by the split. The identity
arm is trusted as a measure of what the step needs without a split only if its temp is within ~20 % of
one JVP's temp plus the Krylov basis (``(restart + 1) * n_dof * 8`` bytes) plus a few residual-sized
vectors; if it is not, the reason is found before anything else is read. A split-owned slot of shape
``(n_rows, width)`` in a float dtype at the peak is a materialized padded product; an ``s32`` one is a
decoded column array.

Run it through the case runner, alone::

    STEP_TEMPS_CASE=pitzdaily validation/run_case.sh validation/step_temporaries.py --wait

The record (every arm's analysis, peak slots and owner sums) is written to
``step_temporaries.<case>.json`` beside this file (git-ignored); the dumps stay under
``STEP_TEMPS_DUMP`` (default a fresh directory under the system's temporary directory). Linux or macOS;
the CPU backend.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CASE = os.environ.get("STEP_TEMPS_CASE", "bfs3d")
ARMS = [a.strip() for a in os.environ.get("STEP_TEMPS_ARMS", "split,identity").split(",")]
COVER = float(os.environ.get("STEP_TEMPS_COVER", "0.9"))
DUMP = Path(
    os.environ.get("STEP_TEMPS_DUMP")
    or tempfile.mkdtemp(prefix=f"aquaflux-step-temporaries-{CASE}-")
)
RECORD = HERE / f"step_temporaries.{CASE}.json"
#: ``1`` re-reads the record's dumps (no case run, no compile) and rewrites its tables.
REREAD = os.environ.get("STEP_TEMPS_REREAD", "0") == "1"
MIB = 2**20

# Before JAX starts: the dumps are configured from the environment when the backend initializes.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + f" --xla_dump_to={DUMP} --xla_dump_hlo_module_re=.*march_step.*"
).strip()

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import hlo_text  # noqa: E402
from step_cases import load_case  # noqa: E402

#: The packages whose code is the residual (its evaluation, or its JVP inside the Krylov operator).
RESIDUAL_PACKAGES = (
    "turbulence",
    "flow",
    "schemes",
    "discretization",
    "boundary",
    "mesh",
    "properties",
    "vectors.py",
)


class _Captured(BaseException):
    """Raised by the stub in place of the march's first step once its arguments are recorded."""


def _capture(compare):
    """The arguments the march hands its first step, and the compiled-step function itself."""
    from aquaflux.solve import march

    program = march._march_step
    captured = {}

    def stub(*args, **kwargs):
        captured["args"], captured["kwargs"] = args, kwargs
        raise _Captured

    march._march_step = stub
    try:
        with contextlib.suppress(_Captured):
            compare.solve_aquaflux(log_path=HERE / f"step_temporaries.{CASE}.march.log")
    finally:
        march._march_step = program
    if "args" not in captured:
        raise SystemExit("the march returned without taking a step")
    return program, captured["args"], captured["kwargs"]


def _no_preconditioner(_shift_strength):
    return None


def _cond_as_loop(pred, true_fn, false_fn, *operands):
    """``lax.cond`` as a ``while_loop`` running ``true_fn`` at most once (bfs3d: XLA kept the decode in it)."""
    import jax
    import jax.numpy as jnp

    def once(_):
        return jnp.asarray(False), true_fn(*operands)

    start = (jnp.asarray(pred), false_fn(*operands))
    return jax.lax.while_loop(lambda carry: carry[0], once, start)[1]


@contextlib.contextmanager
def _true_residual_in_a_loop():
    """While open, ``solve/linear.py`` sees a ``jax`` whose ``lax.cond`` is :func:`_cond_as_loop`."""
    import types

    import jax
    from aquaflux.solve import linear

    class Namespace(types.SimpleNamespace):
        def __init__(self, real, **replaced):
            super().__init__(**replaced)
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

    real = linear.jax
    linear.jax = Namespace(jax, lax=Namespace(jax.lax, cond=_cond_as_loop))
    try:
        yield
    finally:
        linear.jax = real


def _check_cond_as_loop():
    """Refuse to measure unless the looped arm solves exactly as the cond does, application for application."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from aquaflux.solve import linear

    rng = np.random.default_rng(0)
    n = 60
    a = jnp.asarray(np.eye(n) * 4 + rng.standard_normal((n, n)) * 0.3)
    b = jnp.asarray(rng.standard_normal(n))
    inverse_diagonal = 1.0 / jnp.diag(a)
    solver = linear.in_progress_measure(
        linear.residual_stop_gmres(1e-10, restart=7, max_restarts=40), jnp.linalg.norm
    )
    count = [0]

    def matvec(x):
        jax.debug.callback(lambda: count.__setitem__(0, count[0] + 1))
        return a @ x

    def solve():
        count[0] = 0
        jax.clear_caches()
        x, cycles = jax.jit(
            lambda b: linear.solve_linear(
                matvec, b, solver, preconditioner=lambda v: inverse_diagonal * v
            )
        )(b)
        jax.effects_barrier()
        return np.asarray(x), int(cycles), count[0]

    as_cond = solve()
    with _true_residual_in_a_loop():
        as_loop = solve()
    same = as_cond[1:] == as_loop[1:] and np.array_equal(as_cond[0], as_loop[0])
    print(
        f"cond as a loop: applications {as_cond[2]} / {as_loop[2]}, cycles {as_cond[1]} / "
        f"{as_loop[1]}, solutions {'identical' if same else 'DIFFER'}",
        flush=True,
    )
    if not same:
        raise SystemExit("the looped true residual does not reproduce the cond; nothing measured")


def _identity_arm(step):
    """``step`` with its shift policy's preconditioner removed and everything else kept."""
    import equinox as eqx
    from aquaflux.solve import ShiftTerm

    class Unpreconditioned(eqx.Module):
        """A shift policy giving ``base``'s shift term with no preconditioner."""

        base: object

        def shift_term(self, phi, residual=None):
            term = self.base.shift_term(phi, residual)
            return ShiftTerm(term.diagonal, _no_preconditioner, term.row_shift_scale)

    return eqx.tree_at(lambda s: s.shift_policy, step, Unpreconditioned(step.shift_policy.base))


def _owner(info, frames):
    """The owner of an instruction by the registered rule (module docstring), in its order."""
    ours = [f for f in frames if "/aquaflux/" in f[0] and "/validation/" not in f[0]]
    dtype = info["shape"].split("[")[0]
    if not ours:
        return "unattributed"
    innermost = ours[0][0]
    if "/aquaflux/precondition/" in innermost:
        return "split: decode/index" if dtype in ("s32", "s64", "u32") else "split: apply"
    if innermost.endswith("/aquaflux/solve/linear.py"):
        return "Krylov basis/vectors"
    if any("line_search" in f[1] for f in ours):
        return "line search"
    if any(f"/aquaflux/{p}" in innermost for p in RESIDUAL_PACKAGES):
        return "residual JVP" if "jvp" in info["op_name"] else "residual"
    return "step: " + innermost.split("/aquaflux/")[1]


def _hoisted(info, depths):
    """Whether XLA moved an instruction out of a loop its ``op_name`` says it was traced in."""
    traced_in = info["op_name"].count("while/body")
    return traced_in > depths.get(info.get("computation"), traced_in)


def _module_file(before):
    """The after-optimization HLO dump of the step module compiled since ``before`` was listed."""
    new = sorted(set(DUMP.glob("*march_step*cpu_after_optimizations-live-range.txt")) - before)
    if len(new) != 1:
        raise RuntimeError(f"expected one new step module in {DUMP}, found {[p.name for p in new]}")
    return Path(str(new[0])[: -len("-live-range.txt")] + ".txt")


def _peak(allocation_list, ranges):
    """The temp allocation's slots, the peak position and the slots live there."""
    temp = [a for a in allocation_list if "preallocated-temp" in a["kind"]]
    slots = collections.defaultdict(list)
    for a in temp:
        for name, size, offset, shape in a["values"]:
            slots[(a["index"], offset, size)].append((name, shape))
    events = collections.defaultdict(int)
    intervals = {}
    for key, values in slots.items():
        spans = sorted(ranges[v] for v, _ in values if v in ranges)
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        intervals[key] = merged
        for start, end in merged:
            events[start] += key[2]
            events[end + 1] -= key[2]
    live, best, at = 0, -1, 0
    for position in sorted(events):
        live += events[position]
        if live > best:
            best, at = live, position
    at_peak = [key for key, spans in intervals.items() if any(s <= at <= e for s, e in spans)]
    return sum(a["size"] for a in temp), best, at, at_peak, slots


def _name_of(values, ranges, at, instructions):
    """The value that names a slot at position ``at``: the one live there, preferring a computed value.

    A slot holds many values over the program (XLA reuses a temp offset once a value dies), so only a
    value whose own live range contains ``at`` describes what the slot holds at the peak.
    """
    plumbing = ("tuple", "get-tuple-element", "parameter", "bitcast", "while", "copy")
    live = [
        v.split("{")[0] for v, _ in values if v in ranges and ranges[v][0] <= at <= ranges[v][1]
    ]
    for name in live:
        info = instructions.get(name)
        if info and info["opcode"] not in plumbing:
            return name
    return next((n for n in live if n in instructions), live[0] if live else values[0][0])


def _bypass_compilation_cache():
    """Compile from here on without reading or writing the persistent compilation cache."""
    import jax
    from jax._src import compilation_cache

    jax.config.update("jax_enable_compilation_cache", False)
    compilation_cache.reset_cache()


def _analyse(program, args, kwargs, label):
    before = set(DUMP.glob("*march_step*cpu_after_optimizations-live-range.txt"))
    started = time.perf_counter()
    compiled = program.lower(*args, **kwargs).compile().compiled
    seconds = time.perf_counter() - started
    analysis = compiled.memory_analysis()
    hlo_path = _module_file(before)
    result = {
        "compile_seconds": round(seconds, 1),
        "temp": analysis.temp_size_in_bytes,
        "argument": analysis.argument_size_in_bytes,
        "output": analysis.output_size_in_bytes,
        "alias": analysis.alias_size_in_bytes,
        "code": analysis.generated_code_size_in_bytes,
    }
    result.update(_read_dump(hlo_path))
    _print_arm(label, result)
    return result


def _read_dump(hlo_path):
    """The peak and its named slots, from one compiled step's after-optimization dumps."""
    stem = str(hlo_path)[: -len(".txt")]
    hlo = hlo_path.read_text()
    instructions = hlo_text.instructions(hlo)
    frames = hlo_text.stack_frames(hlo)
    depths = hlo_text.loop_depths(instructions)
    ranges = hlo_text.live_ranges(Path(stem + "-live-range.txt").read_text())
    total, peak, at, at_peak, slots = _peak(
        hlo_text.allocations(Path(stem + "-buffer-assignment.txt").read_text()), ranges
    )
    rows = []
    for key in sorted(at_peak, key=lambda k: -k[2]):
        name = _name_of(slots[key], ranges, at, instructions)
        info = instructions.get(name, {"shape": "?", "opcode": "?", "op_name": "", "frame": 0})
        chain = frames.get(info["frame"], [])
        ours = [
            f"{f[0].split('/aquaflux/')[-1]}:{f[2]} {f[1]}" for f in chain if "/aquaflux/" in f[0]
        ]
        rows.append(
            {
                "bytes": key[2],
                "offset": key[1],
                "name": name,
                "shape": info["shape"],
                "opcode": info["opcode"],
                "computation": info.get("computation", "?"),
                "op_name": info["op_name"],
                "frames": ours[:4],
                "owner": _owner(info, chain),
                "hoisted": _hoisted(info, depths),
                "values": len(slots[key]),
            }
        )
    by_owner = collections.Counter()
    for row in rows:
        by_owner[row["owner"] + (" (hoisted)" if row["hoisted"] else "")] += row["bytes"]
    return {
        "temp_allocation": total,
        "peak_live_temp": peak,
        "peak_position": at,
        "slots_at_peak": rows,
        "by_owner_at_peak": dict(by_owner.most_common()),
        "dump": str(hlo_path),
    }


def _print_arm(label, r):
    print(
        f"\n## arm {label}: temp {r['temp'] / MIB:.1f} MiB (allocation {r['temp_allocation'] / MIB:.1f}, "
        f"peak live {r['peak_live_temp'] / MIB:.1f} at position {r['peak_position']}), argument "
        f"{r['argument'] / MIB:.1f}, output {r['output'] / MIB:.1f}, alias {r['alias'] / MIB:.1f}, "
        f"code {r['code'] / MIB:.1f} MiB; compiled in {r['compile_seconds']} s",
        flush=True,
    )
    print(
        "| MiB | cum % | owner | hoisted | instruction | shape | opcode | origin | op_name |",
        flush=True,
    )
    print("|---|---|---|---|---|---|---|---|---|", flush=True)
    cumulative = 0
    for row in r["slots_at_peak"]:
        cumulative += row["bytes"]
        print(
            f"| {row['bytes'] / MIB:.1f} | {100 * cumulative / max(r['temp'], 1):.0f} | {row['owner']} | "
            f"{'yes' if row['hoisted'] else ''} | "
            f"{row['name']} | {row['shape']} | {row['opcode']} | {'; '.join(row['frames'][:2])} | "
            f"{row['op_name'][-90:]} |",
            flush=True,
        )
        if cumulative >= COVER * r["temp"]:
            break
    print(
        "by owner at the peak (MiB): "
        + ", ".join(f"{k} {v / MIB:.1f}" for k, v in r["by_owner_at_peak"].items()),
        flush=True,
    )


def _sanity(args):
    """Temp sizes of one JVP and one residual of the step's own assembler, at the step's state."""
    import equinox as eqx
    from aquaflux.precondition.materialized_session import jacobian_matvec

    residual_fn, state = args[1], args[2]
    assembler = getattr(residual_fn, "__self__", None)
    out = {}
    if assembler is None:
        print(
            f"the step's residual is a {type(residual_fn).__name__}, not a bound method", flush=True
        )
        return out
    for name, program, extra in (
        ("single JVP", jacobian_matvec, (state, state)),
        ("residual", eqx.filter_jit(lambda c, s: c.residual(s)), (state,)),
    ):
        analysis = program.lower(assembler, *extra).compile().compiled.memory_analysis()
        out[name] = {
            "temp": analysis.temp_size_in_bytes,
            "argument": analysis.argument_size_in_bytes,
        }
        print(
            f"program {name}: temp {analysis.temp_size_in_bytes / MIB:.1f} MiB, argument "
            f"{analysis.argument_size_in_bytes / MIB:.1f} MiB",
            flush=True,
        )
    return out


def _header(compare, args):
    import jax

    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    preconditioner = getattr(compare, "PRECONDITIONER", None)
    if preconditioner is None:
        preconditioner = getattr(getattr(compare, "SOLVER", None), "preconditioner", None)
    step = args[0]
    header = {
        "case": CASE,
        "commit": commit,
        "jax": jax.__version__,
        "machine": f"{platform.machine()} {os.cpu_count()} cores, {platform.platform()}",
        "state": "the march's first step (anchor station, hybrid start), after the fit and first re-fit",
        "n_dof": int(args[2].size),
        "step": type(step).__name__,
        "solver": repr(args[5])[:300],
        "preconditioner": repr(preconditioner),
        "dump": str(DUMP),
    }
    for key, value in header.items():
        print(f"{key}: {value}", flush=True)
    return header


def _reread():
    """Rebuild every arm's tables from its dumps and the record's analyses, without running the case."""
    record = json.loads(RECORD.read_text())
    for label, arm in record["arms"].items():
        arm.update(_read_dump(Path(arm["dump"])))
        _print_arm(label, arm)
    _print_difference(record)
    RECORD.write_text(json.dumps(record, indent=1))
    print(f"record rewritten to {RECORD}", flush=True)


def _print_difference(record):
    arms = record["arms"]
    for name in ("split", "identity"):
        if {name, f"{name}-loop"} <= arms.keys():
            print(
                f"\n{name}-loop minus {name}: temp "
                f"{(arms[name + '-loop']['temp'] - arms[name]['temp']) / MIB:+.1f} MiB",
                flush=True,
            )
    if not {"split", "identity"} <= arms.keys():
        return
    split, identity = arms["split"], arms["identity"]
    owned = sum(row["bytes"] for row in split["slots_at_peak"] if row["owner"].startswith("split"))
    print(
        f"\nsplit arm minus identity arm: temp {(split['temp'] - identity['temp']) / MIB:.1f} MiB, "
        f"argument {(split['argument'] - identity['argument']) / MIB:.1f} MiB; split-owned slots "
        f"at the split arm's peak {owned / MIB:.1f} MiB",
        flush=True,
    )


def main():
    if REREAD:
        _reread()
        return

    import jax

    compare = load_case(CASE)
    started = time.perf_counter()
    program, args, kwargs = _capture(compare)
    print(f"step arguments captured in {time.perf_counter() - started:.0f} s", flush=True)
    record = {"header": _header(compare, args), "arms": {}}
    _bypass_compilation_cache()
    if any(arm.endswith("-loop") for arm in ARMS):
        _check_cond_as_loop()
    for arm in ARMS:
        base, _, variant = arm.partition("-")
        if base == "split":
            arm_args = args
        elif base == "identity":
            arm_args = (_identity_arm(args[0]), *args[1:])
        if base not in ("split", "identity") or variant not in ("", "loop"):
            raise SystemExit(f"unknown arm {arm!r}: split, identity, split-loop or identity-loop")
        # A trace is cached per function, so an arm lowered after another could reuse its trace.
        jax.clear_caches()
        with _true_residual_in_a_loop() if variant else contextlib.nullcontext():
            record["arms"][arm] = _analyse(program, arm_args, kwargs, arm)
        RECORD.write_text(json.dumps(record, indent=1))
    _print_difference(record)
    record["sanity"] = _sanity(args)
    RECORD.write_text(json.dumps(record, indent=1))
    print(f"record written to {RECORD}", flush=True)


if __name__ == "__main__":
    main()
