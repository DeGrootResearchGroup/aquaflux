"""The per-cell arrays XLA folds into a step case's compiled residual as constants, and what removes them.

XLA's constant folding evaluates any computation whose inputs are all compile-time constants and stores
the result in the program. A tangent built as ``jnp.zeros_like(gradient).at[:, k].set(1.0)`` is such a
computation -- a scatter of a constant into a constant -- so each one becomes an ``(n_cells, ...)``
literal in every program that differentiates the residual, one copy per program. A broadcast of a
small constant is not folded, so the same tangent written as ``jnp.broadcast_to(unit_k, shape)`` costs
nothing.

This compiles the case's coupled residual (``RESIDUE_CASE``, ``bfs3d`` or ``pitzdaily``) and lists its
constants of at least 4 KiB, with ``boundary_closure.BoundaryClosure.gradient_weight`` as shipped, or,
with ``RESIDUE_BROADCAST_TANGENT=1``, replaced in this process by the broadcast form (the library is not
changed). It saves the residual at a fixed state to ``folded_constants.<case>.<arm>.npy`` beside this
file, so the two arms can be compared bit for bit. Run each arm through the case runner::

    RESIDUE_CASE=bfs3d validation/run_case.sh validation/folded_constants.py --wait
    RESIDUE_CASE=bfs3d RESIDUE_BROADCAST_TANGENT=1 validation/run_case.sh validation/folded_constants.py --wait
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.schemes import boundary_closure  # noqa: E402
from step_cases import load_case  # noqa: E402

CASE = os.environ.get("RESIDUE_CASE", "bfs3d")
BROADCAST = os.environ.get("RESIDUE_BROADCAST_TANGENT", "0") == "1"
#: The smallest constant listed.
LISTED_BYTES = 4096
WIDTH = {"pred": 1, "s8": 1, "u8": 1, "s32": 4, "u32": 4, "f32": 4, "s64": 8, "f64": 8}


def _broadcast_gradient_weight(self, field, gradient):
    """``gradient_weight`` with each unit tangent a broadcast, which XLA does not fold."""
    dim = gradient.shape[1]
    units = jnp.eye(dim, dtype=gradient.dtype)
    return jnp.stack(
        [
            jax.jvp(
                lambda g: self.values(field, g),
                (gradient,),
                (jnp.broadcast_to(units[k], gradient.shape),),
            )[1]
            for k in range(dim)
        ],
        axis=-1,
    )


def constants(hlo_text: str) -> list[tuple[int, str, str]]:
    """(bytes, element type, shape) of every constant instruction in an optimized HLO module's text."""
    found = []
    for line in hlo_text.splitlines():
        match = re.search(r"= (\w+)\[([0-9,]*)\][^ ]* constant\(", line)
        if match and match.group(1) in WIDTH:
            size = WIDTH[match.group(1)]
            for extent in filter(None, match.group(2).split(",")):
                size *= int(extent)
            found.append((size, match.group(1), match.group(2)))
    return found


def main() -> None:
    if BROADCAST:
        boundary_closure.BoundaryClosure.gradient_weight = _broadcast_gradient_weight
    coupled = load_case(CASE).build_case()["coupled"]
    print(
        f"{CASE}: {coupled.momentum.mesh.n_cells} cells, broadcast tangent {BROADCAST}", flush=True
    )
    state = jnp.zeros(coupled.layout.size)
    residual = eqx.filter_jit(lambda c, s: c.residual(s))
    found = constants(residual.lower(coupled, state).compile().compiled.as_text())
    listed = sorted((c for c in found if c[0] >= LISTED_BYTES), reverse=True)
    print(
        f"constants: {len(found)}, {sum(c[0] for c in found) / 2**20:.2f} MiB; "
        f"{len(listed)} of at least {LISTED_BYTES} bytes:",
        flush=True,
    )
    for size, element, shape in listed:
        print(f"  {size / 2**10:9.1f} KiB  {element}[{shape}]", flush=True)
    arm = "broadcast" if BROADCAST else "shipped"
    path = HERE / f"folded_constants.{CASE}.{arm}.npy"
    np.save(path, np.asarray(residual(coupled, state + 0.01)))
    other = HERE / f"folded_constants.{CASE}.{'shipped' if BROADCAST else 'broadcast'}.npy"
    if other.exists():
        difference = np.max(np.abs(np.load(path) - np.load(other)))
        print(f"max |residual difference| against the other arm: {difference:.3e}", flush=True)


if __name__ == "__main__":
    main()
