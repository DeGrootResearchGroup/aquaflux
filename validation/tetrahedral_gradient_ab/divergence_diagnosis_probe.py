"""Issue #435: option 1 clears every seed-state symptom, but the march still diverges -- why?

``wall_velocity_gradient_probe.py`` showed that imposing a wall-model velocity gradient at wall cells
plus excluding the Hessian correction on the first ring of cells around them (option 1) fully clears
the local symptoms measured AT THE SMOOTH RE/10 SEED: the strain-rate/omega ratio drops from 18.8 to
0.88, the omega-production cap stops binding, and every omega-diagonal Jacobian entry is positive. But
marched forward from that same seed the coupled residual still grows without bound.

This probe checkpoints every step of that march (``on_checkpoint``) and, at a handful of the states the
march actually visits -- not only the seed -- re-runs the seed-state diagnostics: the residual broken
down by field block (velocity, pressure, k, omega) and by graph distance from the nearest wall cell, the
strain-rate/omega ratio, where the omega-production cap binds, and omega-diagonal negativity. The field
block breakdown is cheap and printed every step; the more expensive per-state diagnostics run only at a
few checkpoints spread across the march (including the step nearest the residual's minimum, since that
is the state closest to the march genuinely stalling rather than merely still developing).

**Every field-block/ring number is reported twice: raw and scaled.** The raw figure is a plain
Euclidean norm of the physical residual, which conflates fields of different units and magnitudes --
``omega`` alone can span several orders of magnitude near a wall, so a raw comparison across fields, or
across cells within the ``omega`` block, is dominated by units and scale rather than by which equation
is actually furthest from being satisfied. The scaled figure is the march's own
:class:`~aquaflux.solve.RowScaledNorm` -- row-equilibrated by each row's own diagonal, then normalized
by each field's own magnitude -- captured directly from the solve by wrapping ``coupled_scaled_norm``
(the shift-policy machinery that builds it is private and state-dependent, so this reuses the exact
object the march judged each step by rather than reconstructing an approximation of it). Read the
scaled numbers as the answer; the raw numbers are kept alongside only to show how much a raw comparison
would have missed.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/divergence_diagnosis_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aquaflux.turbulence.coupled as coupled_module
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.solve import CompleteLu, DualTimeLoop, MaterializedJacobian
from aquaflux.turbulence import (
    SSTModel,
    inlet_k,
    inlet_omega,
    solve_coupled,
)
from compare import (
    BACKEND,
    INNER_STEPS,
    INNER_TOL,
    INTENSITY,
    LENGTH_SCALE,
    POLYMESH,
    RETRY,
    U_IN,
    build_case,
)
from seed_state_probe import CAP, rings_from_wall
from wall_velocity_gradient_probe import with_wall_model
from warm_start_probe import RATIO, SEED_STEPS, march

STEPS = int(os.environ.get("TET_STEPS", "20"))
#: How many of the checkpointed states to run the expensive per-state diagnostics on.
DETAIL_STEPS = int(os.environ.get("TET_DETAIL_STEPS", "5"))
REFRESH_ON_CYCLES = int(os.environ.get("TET_REFRESH_ON_CYCLES", "3")) or None


def field_block_norms(coupled, state: jnp.ndarray) -> dict[str, float]:
    """The residual 2-norm of each named block: velocity, pressure, k, omega."""
    flow_residual, k_residual, omega_residual = coupled.layout.unpack(coupled.residual(state))
    velocity_residual, pressure_residual = coupled.momentum.layout.unpack(flow_residual)
    return {
        "u": float(jnp.linalg.norm(velocity_residual)),
        "p": float(jnp.linalg.norm(pressure_residual)),
        "k": float(jnp.linalg.norm(k_residual)),
        "omega": float(jnp.linalg.norm(omega_residual)),
    }


def ring_norms(per_cell: np.ndarray, rings: np.ndarray, max_ring: int = 3) -> list[float]:
    """``per_cell``'s 2-norm restricted to each wall-cell ring ``0..max_ring`` (last bucket: ``>=``)."""
    out = [
        float(np.linalg.norm(per_cell[rings == r])) if (rings == r).any() else 0.0
        for r in range(max_ring)
    ]
    mask = rings >= max_ring
    out.append(float(np.linalg.norm(per_cell[mask])) if mask.any() else 0.0)
    return out


def scaled_ring_means(
    per_cell_equilibrated: np.ndarray, field_scale: float, rings: np.ndarray, max_ring: int = 3
) -> list[float]:
    """The row-scaled measure's own recipe (mean, then divide by field scale) applied per ring.

    Matches :meth:`~aquaflux.solve.RowScaledNorm.per_block`, restricted to one ring at a time instead
    of to the whole block, so a ring can be compared on the same fractional-change footing the march
    itself is judged by.
    """
    out = [
        float(np.mean(per_cell_equilibrated[rings == r])) / field_scale
        if (rings == r).any()
        else 0.0
        for r in range(max_ring)
    ]
    mask = rings >= max_ring
    out.append(float(np.mean(per_cell_equilibrated[mask])) / field_scale if mask.any() else 0.0)
    return out


def capture_scaled_norms():
    """Monkeypatch ``coupled_scaled_norm`` to record every ``RowScaledNorm`` the march builds.

    The shift policy it needs is private, state-carrying continuation machinery -- reconstructing an
    equivalent one outside the march would risk approximating the very thing under test. Capturing the
    actual object each outer iteration builds (see ``aquaflux/solve/march.py``'s ``norm_builder``: once
    before the loop for the reference, then once per outer iteration, held fixed across that
    iteration's line search) means every later analysis uses the measure the march was really judged
    by. Returns ``(captured, restore)``; ``captured`` is a list of ``RowScaledNorm``, oldest first,
    growing live as the march runs.
    """
    captured: list = []
    original = coupled_module.coupled_scaled_norm

    def wrapped(coupled, shift_policy, state):
        norm = original(coupled, shift_policy, state)
        captured.append(norm)
        return norm

    coupled_module.coupled_scaled_norm = wrapped

    def restore() -> None:
        coupled_module.coupled_scaled_norm = original

    return captured, restore


def velocity_component_slices(coupled) -> list[slice]:
    """Global-state slices for each velocity component, in the coupled layout's field-major order."""
    n, dim = coupled.momentum.mesh.n_cells, coupled.momentum.mesh.dim
    velocity_slice = coupled.momentum.layout.slice_of("velocity")
    start = velocity_slice.start
    return [slice(start + c * n, start + (c + 1) * n) for c in range(dim)]


def scaled_block_ring_report(label, name, per_cell_residual, row_scale, field_scale, rings) -> None:
    """Print a named scalar block's SCALED fractional change by ring, given its own row/field scale."""
    equilibrated = np.abs(np.asarray(per_cell_residual)) / np.asarray(row_scale)
    scaled_rings = scaled_ring_means(equilibrated, field_scale, rings)
    print(
        f"    [{label}] SCALED {name} fractional change by ring (0,1,2,>=3): "
        f"{[f'{v:.3e}' for v in scaled_rings]}",
        flush=True,
    )


def scaled_velocity_ring_report(label, coupled, residual, norm, rings) -> None:
    """The velocity block's SCALED fractional change by ring, combining components by Euclidean norm.

    Mirrors how :meth:`~aquaflux.solve.RowScaledNorm.__call__` combines already-meaned per-block
    fractional numbers -- applied here per ring instead of over the whole block.
    """
    row_scale = np.asarray(norm.row_scale)
    field_scale = float(np.asarray(norm.field_scale)[0])  # shared across every velocity component
    per_component = [
        scaled_ring_means(np.abs(np.asarray(residual)[s]) / row_scale[s], field_scale, rings)
        for s in velocity_component_slices(coupled)
    ]
    combined = np.linalg.norm(np.stack(per_component), axis=0)
    print(
        f"    [{label}] SCALED velocity fractional change by ring (0,1,2,>=3): "
        f"{[f'{v:.3e}' for v in combined.tolist()]}",
        flush=True,
    )


def diagonal_sign_report(label, name, coupled, state, block_slice, rings, considered) -> None:
    """Print how many of a scalar block's Jacobian diagonal entries are non-positive, and where.

    ``considered`` names which cells a non-positive diagonal is meaningful for: ``omega``'s
    wall-fixation cells (ring 0) hold an algebraic constraint by construction and must be excluded,
    while ``k`` has no such cells -- excluding ring 0 there would hide exactly the cells nearest the
    wall-model velocity gradient and the near-wall production blend, which is where a defect is most
    likely to show up.
    """

    def r_block(phi):
        return coupled.residual(state.at[block_slice].set(phi))[block_slice]

    diag = np.diag(np.asarray(jax.jacfwd(r_block)(state[block_slice])))
    bad = considered & (diag <= 0)
    clipped = np.minimum(rings, 3)
    print(
        f"    [{label}] {name} diag<=0 {bad.sum()}"
        + (
            f" at rings (0,1,2,>=3) {np.bincount(clipped[bad], minlength=4).tolist()}"
            if bad.any()
            else ""
        ),
        flush=True,
    )


def strain_ratio_by_ring(ratio: np.ndarray, rings: np.ndarray, max_ring: int = 3) -> list[str]:
    """``median (max)`` of ``S/omega`` per wall-cell ring, as display strings."""
    out = []
    for r in range(max_ring):
        mask = rings == r
        out.append(
            f"{np.median(ratio[mask]):.2f} ({ratio[mask].max():.2f})" if mask.any() else "--"
        )
    mask = rings >= max_ring
    out.append(f"{np.median(ratio[mask]):.2f} ({ratio[mask].max():.2f})" if mask.any() else "--")
    return out


def detailed_report(label, coupled, state, rings, norm) -> None:
    flow, k, omega = coupled.physical_fields(state)
    closure, _ = coupled.effective_momentum(flow, k, omega)
    ratio = np.asarray(closure.strain_rate / omega)
    free = rings > 0

    residual = coupled.residual(state)
    flow_residual, k_residual, omega_residual = coupled.layout.unpack(residual)
    velocity_residual, _ = coupled.momentum.layout.unpack(flow_residual)
    u_rings = ring_norms(np.asarray(jnp.linalg.norm(velocity_residual, axis=-1)), rings)
    omega_rings = ring_norms(np.abs(np.asarray(omega_residual)), rings)
    k_rings = ring_norms(np.abs(np.asarray(k_residual)), rings)

    omega_slice = coupled.layout.slice_of("omega")
    k_slice = coupled.layout.slice_of("k")

    print(
        f"    [{label}] S/omega by ring (0,1,2,>=3), median (max): "
        f"{strain_ratio_by_ring(ratio, rings)}; cap binds {(free & (ratio > CAP)).sum()}",
        flush=True,
    )
    diagonal_sign_report(label, "omega", coupled, state, omega_slice, rings, free)
    diagonal_sign_report(label, "k", coupled, state, k_slice, rings, np.ones_like(free))
    print(
        f"    [{label}] RAW |R_u| by ring (0,1,2,>=3): {[f'{v:.3e}' for v in u_rings]}", flush=True
    )
    print(
        f"    [{label}] RAW |R_k| by ring (0,1,2,>=3): {[f'{v:.3e}' for v in k_rings]}", flush=True
    )
    print(
        f"    [{label}] RAW |R_omega| by ring (0,1,2,>=3): {[f'{v:.3e}' for v in omega_rings]}",
        flush=True,
    )

    if norm is None:
        print(
            f"    [{label}] no captured scaled measure -- skipping the scaled breakdown", flush=True
        )
        return

    names = [f"u{c}" for c in range(coupled.momentum.mesh.dim)] + ["p", "k", "omega"]
    per_block = np.asarray(norm.per_block(residual))
    print(
        f"    [{label}] SCALED per-block fractional change: "
        + " ".join(f"{n}={v:.3e}" for n, v in zip(names, per_block, strict=True)),
        flush=True,
    )
    row_scale = np.asarray(norm.row_scale)
    field_scale = np.asarray(norm.field_scale)
    scaled_velocity_ring_report(label, coupled, residual, norm, rings)
    scaled_block_ring_report(
        label, "k", k_residual, row_scale[k_slice], float(field_scale[-2]), rings
    )
    scaled_block_ring_report(
        label, "omega", omega_residual, row_scale[omega_slice], float(field_scale[-1]), rings
    )


def run_and_checkpoint(coupled, flow, k, omega, rtol, max_steps):
    started = time.perf_counter()
    checkpoints: list[tuple[int, jnp.ndarray]] = []
    captured, restore = capture_scaled_norms()

    def on_step(report):
        print(
            f"  [option1] step {report.step:3d} |R| {report.residual_norm:.4e} "
            f"alpha {report.alpha:.4g} cyc {report.cycles} esc {report.escalations} "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )

    def on_checkpoint(report, state):
        checkpoints.append((report.step, state))

    try:
        solve_coupled(
            coupled,
            flow,
            k,
            omega,
            preconditioner=MaterializedJacobian(CompleteLu(backend=BACKEND)),
            dual_time=DualTimeLoop(
                inner_steps=INNER_STEPS, inner_tol=INNER_TOL, refresh_on_cycles=REFRESH_ON_CYCLES
            ),
            max_steps=max_steps,
            rtol=rtol,
            atol=0.0,
            positivity_projection=True,
            retry=RETRY,
            on_step=on_step,
            on_checkpoint=on_checkpoint,
        )
        print("  [option1] CONVERGED", flush=True)
    except Exception as exc:  # the march's own guard raises on non-convergence -- report it
        print(
            f"  [option1] ended: {type(exc).__name__}: {str(exc).splitlines()[0][:140]}", flush=True
        )
    finally:
        restore()
    return checkpoints, captured


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    seed_case = build_case(CorrectedGreenGauss()).with_scaled_molecular_viscosity(RATIO)
    n = mesh.n_cells
    flow0 = seed_case.momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
    k0 = jnp.full(n, float(inlet_k(jnp.array(U_IN), INTENSITY)))
    omega0 = jnp.full(n, float(inlet_omega(jnp.array(k0[0]), LENGTH_SCALE, SSTModel())))
    seed = march("seed", seed_case, flow0, k0, omega0, 1e-4, SEED_STEPS)
    if seed is None:
        return

    rings = rings_from_wall(mesh, seed_case.turbulence.wall_cells)
    multcorr = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ).bind(mesh, geometry)
    option1 = with_wall_model(
        build_case(multcorr).with_scaled_molecular_viscosity(RATIO), first_ring=True
    )

    print(f"=== option 1, from the seed, checkpointed ({STEPS} steps) ===", flush=True)
    checkpoints, captured = run_and_checkpoint(option1, *seed, 1e-3, STEPS)
    if not checkpoints:
        print("no checkpoints were recorded -- the first step itself failed", flush=True)
        return

    # `captured[0]` is the reference-state build before the loop; `captured[i + 1]` is the measure the
    # march judged step `i` by (see `capture_scaled_norms`'s docstring). If a step were fully rejected
    # with no report, the two lists would drift -- fall back to the last captured measure rather than
    # silently misattributing one step's norm to another.
    print(
        f"captured {len(captured)} scaled-norm builds for {len(checkpoints)} checkpointed steps "
        + (
            "(aligned 1:1)"
            if len(captured) == len(checkpoints) + 1
            else "(COUNTS DIFFER -- see below)"
        ),
        flush=True,
    )

    def norm_for(i: int):
        if not captured:
            return None
        return captured[i + 1] if i + 1 < len(captured) else captured[-1]

    norms = [field_block_norms(option1, state) for _, state in checkpoints]
    total = [sum(v * v for v in b.values()) ** 0.5 for b in norms]
    names = [f"u{c}" for c in range(option1.momentum.mesh.dim)] + ["p", "k", "omega"]
    print("=== residual by field block, per checkpointed step (RAW, then SCALED) ===", flush=True)
    for i, ((step, _), blocks, t) in enumerate(zip(checkpoints, norms, total, strict=True)):
        by_block = " ".join(f"{name}={v:.3e}" for name, v in blocks.items())
        print(f"  step {step:3d} RAW    |R| {t:.4e}  {by_block}", flush=True)
        norm = norm_for(i)
        if norm is not None:
            per_block = np.asarray(norm.per_block(option1.residual(checkpoints[i][1])))
            scaled_total = float(np.linalg.norm(per_block))
            by_block_scaled = " ".join(
                f"{n}={v:.3e}" for n, v in zip(names, per_block, strict=True)
            )
            print(f"  step {step:3d} SCALED |R| {scaled_total:.4e}  {by_block_scaled}", flush=True)
    best = int(np.argmin(total))
    picks = sorted(
        {
            0,
            best,
            len(checkpoints) - 1,
            *(round(f * (len(checkpoints) - 1)) for f in (0.25, 0.5, 0.75)),
        }
    )
    picks = picks[-DETAIL_STEPS:]
    print(
        f"=== detailed diagnostics at steps {[checkpoints[i][0] for i in picks]} "
        f"(residual minimum at step {checkpoints[best][0]}) ===",
        flush=True,
    )
    for i in picks:
        step, state = checkpoints[i]
        detailed_report(f"step {step}", option1, state, rings, norm_for(i))


if __name__ == "__main__":
    main()
