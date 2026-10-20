"""Whether the probe's one-program-per-column-field Jacobian equals the whole-state one, entry for entry.

The coloured probe can seed the whole state (``batched_jacobian_matvec``, a direction that is zero outside
one column field) or differentiate the residual in that one field alone (``batched_field_jacobian_matvec``,
the other fields constants of the derivative, so their tangents are never formed). The second only removes
additions of zero and products with zero, so it should recover the same matrix; this materializes both at
one state over the case's own probe plan (colouring, reach and narrowed residual are the session's) and
compares the CSR arrays.

It reports, per (row field, column field) block, the entries that differ, their largest absolute
difference and that difference over the block's largest entry; the time and calls of each materialize;
and whether any entry of either matrix is not finite (a pruned path through an ``inf`` or ``NaN`` is the
one way removing a zero could change a value).

Rounding is the expected source of any difference, not a defect: a changed program is compiled
differently, and on a processor with fused multiply-add the compiler contracts a multiply and an add where
it finds them, which moves once an add of zero is gone. Run once more with
``XLA_FLAGS=--xla_cpu_max_isa=AVX`` (no fused multiply-add instructions) to tell that apart from a
genuine difference: there both matrices must be identical.

``PROBE_IDENTITY_SAVE`` writes the whole-state matrix to a ``.npz`` (arrays ``data``, ``indices``,
``indptr``); ``PROBE_IDENTITY_REFERENCE`` compares the per-field one against such a file instead -- say,
one written by another checkout's whole-state probe at the same state.

Usage::

    PROBE_IDENTITY_CASE=pitzdaily validation/run_case.sh validation/fieldwise_probe_identity.py --wait
    PROBE_IDENTITY_CASE=bfs3d PROBE_IDENTITY_STATE=<checkpoint>.npz validation/run_case.sh ...

``PROBE_IDENTITY_CASE`` is ``pitzdaily`` (default) or ``bfs3d``; ``PROBE_IDENTITY_STATE`` a ``.npz``
holding ``state`` (default the hybrid start).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.initialization import hybrid_initialize  # noqa: E402
from aquaflux.precondition.materialized_session import (  # noqa: E402
    PROBE_BATCH_SIZE,
    _probe_products,
)
from aquaflux.precondition.sparse_jacobian import (  # noqa: E402
    block_stencil_gather_map,
    materialize_block_jacobian,
)
from aquaflux.turbulence import open_session  # noqa: E402
from step_cases import load_case  # noqa: E402

CASE = os.environ.get("PROBE_IDENTITY_CASE", "pitzdaily")
STATE = os.environ.get("PROBE_IDENTITY_STATE")
SAVE = os.environ.get("PROBE_IDENTITY_SAVE")
REFERENCE = os.environ.get("PROBE_IDENTITY_REFERENCE")


def build(case_name: str = CASE, state_path: str | None = STATE):
    """The case's coupled assembler, the state, its session's probe and the narrowed residual it probes."""
    case = load_case(case_name)
    coupled = case.build_case()["coupled"]
    if state_path:
        with np.load(state_path) as data:
            state = jnp.asarray(data["state"])
    else:
        state = coupled.state_from_physical(*hybrid_initialize(coupled))
    if not bool(jnp.all(jnp.isfinite(coupled.residual(state)))):
        raise SystemExit(
            "the residual at this state is not finite; nothing here would mean anything"
        )
    session = open_session(case.SOLVER.settings()["preconditioner"], coupled)
    probe = session._probe_for()
    return coupled, jax.lax.stop_gradient(state), probe, probe.narrow(coupled)


def materialize(probed, state, plan, structure, by_field):
    """The CSR Jacobian through one form of the product, its seconds and its batched calls."""
    matvec, batched = _probe_products(probed, state, by_field)
    calls = [0]

    def counted(*args):
        calls[0] += 1
        return batched(*args)

    counting = type(batched)(counted) if by_field is not None else counted
    materialize_block_jacobian(  # warm: every program compiled before the timed call
        matvec,
        plan,
        batched_matvec=counting,
        probe_batch_size=PROBE_BATCH_SIZE,
        structure=structure,
    )
    calls[0] = 0
    started = time.perf_counter()
    jacobian = materialize_block_jacobian(
        matvec,
        plan,
        batched_matvec=counting,
        probe_batch_size=PROBE_BATCH_SIZE,
        structure=structure,
    )
    return jacobian, time.perf_counter() - started, calls[0]


def compare(name, whole, fieldwise, n_cells, n_fields) -> bool:
    """Print the comparison of two CSR matrices of one pattern; return whether they are identical."""
    same_pattern = np.array_equal(whole.indices, fieldwise.indices) and np.array_equal(
        whole.indptr, fieldwise.indptr
    )
    identical = same_pattern and np.array_equal(whole.data, fieldwise.data)
    print(f"\n{name}: pattern identical {same_pattern}, data identical {identical}")
    print(
        f"non-finite entries: whole state {int((~np.isfinite(whole.data)).sum())}, "
        f"per field {int((~np.isfinite(fieldwise.data)).sum())}"
    )
    if not same_pattern or identical:
        return identical
    rows = np.repeat(np.arange(whole.shape[0]), np.diff(whole.indptr)) // n_cells
    cols = whole.indices // n_cells
    difference = np.abs(whole.data - fieldwise.data)
    print(
        "| row field | column field | entries | differing | max abs difference | over block max |"
    )
    print("|---|---|---|---|---|---|")
    for a in range(n_fields):
        for b in range(n_fields):
            block = (rows == a) & (cols == b)
            if not block.any():
                continue
            scale = np.abs(whole.data[block]).max()
            worst = difference[block].max()
            print(
                f"| {a} | {b} | {int(block.sum())} | {int((difference[block] > 0).sum())} | "
                f"{worst:.3e} | {worst / scale if scale > 0 else 0.0:.3e} |"
            )
    return identical


def main() -> None:
    print(
        f"[configuration] case {CASE}, state {STATE or 'the hybrid start'}, probe batch "
        f"{PROBE_BATCH_SIZE}; jax {jax.__version__}, XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}",
        flush=True,
    )
    coupled, state, probe, probed = build()
    plan = probe.plan
    structure = block_stencil_gather_map(plan)
    print(f"{plan.n_cells} cells, {plan.n_fields} fields, {plan.n_probes} probes {plan.n_colours}")
    fieldwise, seconds, calls = materialize(probed, state, plan, structure, coupled.layout)
    print(f"per field: {seconds:.2f} s, {calls} batched calls", flush=True)
    if REFERENCE:
        with np.load(REFERENCE) as data:
            whole = type(fieldwise)(
                (data["data"], data["indices"], data["indptr"]), shape=fieldwise.shape
            )
        label = f"per field against {REFERENCE}"
    else:
        whole, seconds, calls = materialize(probed, state, plan, structure, None)
        print(f"whole state: {seconds:.2f} s, {calls} batched calls", flush=True)
        label = "per field against whole state"
        if SAVE:
            np.savez(SAVE, data=whole.data, indices=whole.indices, indptr=whole.indptr)
    compare(label, whole, fieldwise, plan.n_cells, plan.n_fields)


if __name__ == "__main__":
    main()
