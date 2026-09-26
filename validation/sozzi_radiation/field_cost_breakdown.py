"""Where the Sozzi reactor's whole-field call spends its time, component by component.

``model_at_mesh_scale.py`` times the model's build and its one ``fluence_rate`` call as two
numbers. This runs the same model -- the same 1,635,909 cell centres, the case's ``lampWall.stl``,
the water as ``Outside(chamber, inlet, riser)``, ``NoOcclusion``, the receiver mask streamed, and
every other setting left at the library default -- and charges the time to the pieces the call is
made of:

- the build: the facet-to-facet transfer (its shadow mask separately) and the rest;
- the surface solve: assembling the transfer and the interreflection solve;
- the streamed field, per chunk: building the chunk's shadow mask -- and, inside it, each stage of
  :class:`~aquaflux.radiation.ShaftCulling`: ordering points along the curve, the bodies'
  clearance features, the tile certificates, and the pair-by-pair test of the tiles left over --
  then the compiled gather.

Each piece is wrapped in a timer that waits for its result (``jax.block_until_ready``) before
stopping the clock, so compiled work is charged to the piece that launched it rather than to
whatever next waits on it. ⚠️ That only holds where the work before a piece is timed too: a timer
that waits on its own result also waits for anything dispatched earlier and still running, which is
why the transfer's row blocks have a timer of their own -- without one, their ~27 s was charged to
the facet mask, which itself takes a few hundredths of a second. That also removes any overlap between host work and compiled work the
unwrapped call might get, so the instrumented total is compared with the plain one and the
difference reported, not assumed away. Timers nest: each is inclusive of the ones inside it, and
the summary gives each stage's share of its parent and what is left of the parent outside its
named children.

``SOZZI_CULLING`` chooses the bodies' layer as in ``model_at_mesh_scale.py`` (unset: the library
default). ``SOZZI_RECEIVERS`` takes the first *n* cells in mesh order instead of all of them, for a
quick check of the harness itself; a figure quoted from this file is from the whole mesh.

Run with ``validation/run_case.sh validation/sozzi_radiation/field_cost_breakdown.py``. Needs
``work/case`` (for the lamp) and ``work/cell_centres.npy``. Writes
``work/compare/field_cost_breakdown[-<arm>].json``.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

WORK = HERE / "work"
CULLING = os.environ.get("SOZZI_CULLING", "")
RECEIVERS = int(os.environ.get("SOZZI_RECEIVERS", 0))
RESULT = WORK / "compare" / f"field_cost_breakdown{'-' + CULLING if CULLING else ''}.json"


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Timers:
    """Inclusive wall time per named piece: calls, total, the first call, and the slowest."""

    def __init__(self):
        self.entries: dict[str, dict] = {}

    def wrap(self, name: str, function):
        """``function``, timed under ``name`` until its result is ready."""
        import jax

        @functools.wraps(function)
        def timed(*args, **kwargs):
            started = time.perf_counter()
            result = function(*args, **kwargs)
            jax.block_until_ready(result)
            self.add(name, time.perf_counter() - started)
            return result

        return timed

    def add(self, name: str, seconds: float) -> None:
        entry = self.entries.setdefault(name, {"calls": 0, "seconds": 0.0, "first": seconds})
        entry["calls"] += 1
        entry["seconds"] += seconds
        entry["slowest"] = max(entry.get("slowest", 0.0), seconds)

    def seconds(self, name: str) -> float:
        return self.entries.get(name, {}).get("seconds", 0.0)


#: Each timed piece and the piece it sits inside, in the order the summary prints them.
TREE = {
    "build": None,
    "build / transfer": "build",
    "build / transfer / row blocks": "build / transfer",
    "build / transfer / facet mask": "build / transfer",
    "call": None,
    "call / radiosity": "call",
    "call / radiosity / assemble": "call / radiosity",
    "call / radiosity / solve": "call / radiosity",
    "call / field": "call",
    "call / field / refuse points inside": "call / field",
    "call / field / chunk": "call / field",
    "call / field / chunk / mask": "call / field / chunk",
    "call / field / chunk / mask / bodies": "call / field / chunk / mask",
    "call / field / chunk / mask / bodies / curve order": "call / field / chunk / mask / bodies",
    "call / field / chunk / mask / bodies / undecided tiles": "call / field / chunk / mask / bodies",
    "call / field / chunk / mask / bodies / undecided tiles / clearance": (
        "call / field / chunk / mask / bodies / undecided tiles"
    ),
    "call / field / chunk / mask / bodies / undecided tiles / clearance / lamp facets": (
        "call / field / chunk / mask / bodies / undecided tiles / clearance"
    ),
    "call / field / chunk / mask / bodies / undecided tiles / clearance / receivers": (
        "call / field / chunk / mask / bodies / undecided tiles / clearance"
    ),
    "call / field / chunk / mask / bodies / undecided tiles / certificates": (
        "call / field / chunk / mask / bodies / undecided tiles"
    ),
    "call / field / chunk / mask / bodies / test tiles": "call / field / chunk / mask / bodies",
    "call / field / chunk / mask / bodies / test tiles / compiled test": (
        "call / field / chunk / mask / bodies / test tiles"
    ),
    "call / field / chunk / gather": "call / field / chunk",
}


def instrument_build(timers: Timers) -> None:
    """Wrap the build's pieces where the library calls them."""
    from aquaflux.radiation import model, transfer

    model.build_transfer = timers.wrap("build / transfer", model.build_transfer)
    # Timed on its own because its work is dispatched asynchronously: untimed, it would still be
    # running when the facet mask's timer starts, and be charged to the mask.
    transfer._row_blocks = timers.wrap("build / transfer / row blocks", transfer._row_blocks)
    transfer.build_visibility = timers.wrap(
        "build / transfer / facet mask", transfer.build_visibility
    )


def instrument_call(timers: Timers, n_facets: int, pairs: dict) -> None:
    """Wrap the call's pieces where the library calls them.

    Applied only once the model is built, because the transfer's own mask goes through the same
    culling code, and would otherwise be charged to the field's chunks.
    """
    from aquaflux.radiation import culling, gather, model, receiver_shadows
    from aquaflux.radiation.triangles import padded_length
    from aquaflux.solids import Outside

    model.radiosity = timers.wrap("call / radiosity", model.radiosity)
    model.TransferMatrix.assemble = timers.wrap(
        "call / radiosity / assemble", model.TransferMatrix.assemble
    )
    model._interreflection = timers.wrap("call / radiosity / solve", model._interreflection)
    receiver_shadows.streamed_fluence_rate = timers.wrap(
        "call / field", receiver_shadows.streamed_fluence_rate
    )
    gather.refuse_points_inside = timers.wrap(
        "call / field / refuse points inside", gather.refuse_points_inside
    )
    gather._chunk_total = timers.wrap("call / field / chunk", gather._chunk_total)
    gather._unchecked_visibility = timers.wrap(
        "call / field / chunk / mask", gather._unchecked_visibility
    )
    compiled = gather._compiled_gather
    gather._compiled_gather = lambda *args: timers.wrap(
        "call / field / chunk / gather", compiled(*args)
    )
    for strategy in (culling.EveryPair, culling.ShaftCulling):
        strategy.blocked = timers.wrap("call / field / chunk / mask / bodies", strategy.blocked)
    culling._Curve.of = classmethod(
        timers.wrap(
            "call / field / chunk / mask / bodies / curve order", culling._Curve.of.__func__
        )
    )
    culling.ShaftCulling._undecided = timers.wrap(
        "call / field / chunk / mask / bodies / undecided tiles", culling.ShaftCulling._undecided
    )
    clearance = "call / field / chunk / mask / bodies / undecided tiles / clearance"
    of_facets = timers.wrap(f"{clearance} / lamp facets", Outside.clearance)
    of_receivers = timers.wrap(f"{clearance} / receivers", Outside.clearance)
    Outside.clearance = timers.wrap(
        clearance,
        lambda body, points: (of_facets if len(points) == n_facets else of_receivers)(body, points),
    )
    for name in ("_vouched", "_vouched_pairs"):
        setattr(
            culling,
            name,
            timers.wrap(
                "call / field / chunk / mask / bodies / undecided tiles / certificates",
                getattr(culling, name),
            ),
        )
    test_tiles = timers.wrap(
        "call / field / chunk / mask / bodies / test tiles", culling.ShaftCulling._test_tiles
    )

    def counted_test_tiles(self, body, sources, near, receivers, rows, cols, pair_limit, out):
        # The same batching arithmetic as the strategy's own, to count what it tests.
        per_tile = rows.shape[1] * cols.shape[1]
        per_batch = max(1, pair_limit // per_tile)
        batches = [min(per_batch, len(rows) - start) for start in range(0, len(rows), per_batch)]
        pairs["in undecided tiles"] += len(rows) * per_tile
        pairs["tested, padding included"] += sum(padded_length(n) for n in batches) * per_tile
        return test_tiles(self, body, sources, near, receivers, rows, cols, pair_limit, out)

    culling.ShaftCulling._test_tiles = counted_test_tiles
    blocked = culling.ShaftCulling.blocked

    def counted_blocked(self, bodies, sources, near, receivers, *args, **kwargs):
        pairs["all"] += len(bodies) * len(receivers) * len(sources)
        return blocked(self, bodies, sources, near, receivers, *args, **kwargs)

    culling.ShaftCulling.blocked = counted_blocked
    culling._body_blocks = timers.wrap(
        "call / field / chunk / mask / bodies / test tiles / compiled test", culling._body_blocks
    )


def summary(timers: Timers) -> list[dict]:
    """Each piece's time, its share of its parent, and what its named children leave over."""
    rows = []
    for name, parent in TREE.items():
        entry = timers.entries.get(name)
        if entry is None:
            continue
        children = sum(timers.seconds(child) for child, up in TREE.items() if up == name)
        own = entry["seconds"]
        rows.append(
            {
                "piece": name,
                "calls": entry["calls"],
                "seconds": round(own, 2),
                "share_of_parent": None
                if parent is None
                else round(own / timers.seconds(parent), 4),
                "outside_named_children": round(own - children, 2) if children else None,
                "first_call_seconds": round(entry["first"], 3),
                "slowest_call_seconds": round(entry["slowest"], 3),
            }
        )
    return rows


def main() -> None:
    import aquaflux  # noqa: F401  (enables x64)
    import numpy as np
    from aquaflux.radiation import (
        NoOcclusion,
        RadiationSettings,
        UniformAbsorption,
        build_radiation_model,
        fluence_rate,
    )
    from compare_fluence import ABSORPTION, lamp_surfaces
    from model_at_mesh_scale import culling_settings
    from primitive_occlusion import fluid, from_drawing

    cells = np.load(WORK / "cell_centres.npy")
    if RECEIVERS:
        cells = cells[:RECEIVERS]
    lamp = lamp_surfaces()
    water = from_drawing()
    source = "the CAD drawing"
    if water is None:
        water, source = fluid(), "three hand-typed cylinders (no CAD kernel)"
    chosen = culling_settings()
    described = repr(chosen.get("body_culling", "library default"))
    _say(f"{len(cells)} cells, {lamp.n_facets} lamp facets, water from {source}, {described}")

    timers = Timers()
    instrument_build(timers)

    started = time.perf_counter()
    model = build_radiation_model(
        cells,
        lamp,
        occluders=[water],
        settings=RadiationSettings(
            self_occlusion=NoOcclusion(), stream_receiver_mask=True, **chosen
        ),
    )
    timers.add("build", time.perf_counter() - started)
    _say(f"model built in {timers.seconds('build'):.1f} s")
    pairs = {"all": 0, "in undecided tiles": 0, "tested, padding included": 0}
    instrument_call(timers, int(lamp.n_facets), pairs)

    started = time.perf_counter()
    field, cycles = fluence_rate(model, lamp, absorption=UniformAbsorption(ABSORPTION))
    field = np.asarray(field)
    timers.add("call", time.perf_counter() - started)
    _say(f"field computed in {timers.seconds('call'):.1f} s ({int(cycles)} solver cycles)")

    rows = summary(timers)
    for row in rows:
        depth = row["piece"].count(" / ")
        share = "" if row["share_of_parent"] is None else f"{100 * row['share_of_parent']:5.1f}%"
        left = (
            ""
            if row["outside_named_children"] is None
            else f"  (outside named children {row['outside_named_children']:.1f} s)"
        )
        print(
            f"{'  ' * depth}{row['piece'].split(' / ')[-1]:<24} {row['seconds']:9.1f} s "
            f"{share:>7} x{row['calls']:<6}{left}",
            flush=True,
        )
    for name, count in pairs.items():
        print(f"pairs {name}: {count:,} ({count / max(pairs['all'], 1):.1%} of all)", flush=True)
    RESULT.write_text(
        json.dumps(
            {
                "cells": len(cells),
                "water": source,
                "body_culling": described,
                "lamp_facets": int(lamp.n_facets),
                "solver_cycles": int(cycles),
                "finite": bool(np.isfinite(field).all()),
                "pairs": pairs,
                "pieces": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
