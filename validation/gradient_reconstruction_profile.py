"""Where the time goes inside a Hessian-corrected gradient reconstruction.

The Hessian-corrected (Betchen) reconstruction is several times the cost of corrected Green--Gauss,
and a march-level A/B says only *that* it is -- not *which part* of it is. This decomposes one
reconstruction into the pieces a targeted optimization could act on, so that a proposal to cache or
merge something is aimed at a measured cost rather than a guessed one.

The split that matters is **prologue against sweep**. The prologue is geometry-only: the per-cell
blocks the two preconditioners invert, built by probing the face kernel a fixed number of times
(``dim`` for the unreduced Hessian block, ``n_sym`` for the reduced one, ``dim`` for the gradient
block, and ``dim + n_sym`` more if the local Schur block is on). It does not depend on the field, so it looks like the
obvious thing to cache -- but read its share carefully before sizing that prize: a residual
reconstructs several fields on one geometry inside one compiled region, and the compiler already
collapses those identical prologues to one. What a cache could collect is therefore only the
repetition *across* residual evaluations, and at reactor scale the arrays involved are hundreds of
megabytes made permanently resident to buy it. The sweep is the field-dependent part, and its cost is per-sweep, so it is what a sweep
calibration or a merged kernel evaluation acts on.

Every arm is jitted, warmed, and reported as the **minimum** of several runs, because this machine is
shared and the per-application spread on it has been measured at roughly 15 %. Differences smaller
than that are not resolved here and are not reported as findings.

Usage
-----
    validation/run_case.sh validation/gradient_reconstruction_profile.py

    PROFILE_SWEEPS=20 PROFILE_REPEATS=5 validation/run_case.sh validation/gradient_reconstruction_profile.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "pitzdaily_openfoam"))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.schemes.gradient import (  # noqa: E402
    AveragedNeighbourHessian,
    CoupledBlockSweep,
    HessianCorrectedGradient,
    OwnerHessian,
    SweptGradientSolve,
    contraction_rate,
    symmetric_components,
)

SWEEPS = int(os.environ.get("PROFILE_SWEEPS", "20"))
REPEATS = int(os.environ.get("PROFILE_REPEATS", "5"))
CHAIN = int(os.environ.get("PROFILE_CHAIN", "8"))

#: Which boundary closure to profile under. It is not a detail: the closure changes the operator, so
#: it changes the sweep's contraction rate and hence the whole sweeps-versus-accuracy ladder below.
#: On pitzDaily the shipped ``OwnerHessian`` contracts at 0.2263 and ``AveragedNeighbourHessian`` at
#: 0.3152 -- sixteen sweeps against twenty to reach ``1e-10``. Default to what the marches run.
CLOSURES = {"owner": OwnerHessian, "averaged": AveragedNeighbourHessian}
CLOSURE = os.environ.get("PROFILE_CLOSURE", "owner")


def timed(fn, *args, repeats=REPEATS):
    """Minimum wall time of a jitted callable over `repeats` runs, after one warm-up."""
    compiled = jax.jit(fn)
    jax.block_until_ready(compiled(*args))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        best = min(best, time.perf_counter() - start)
    return best


def main():
    mesh = read_openfoam(HERE / "pitzdaily_openfoam" / "runs" / "kwsst" / "polyMesh")
    geom = mesh.geometry()
    dim = mesh.dim
    n_cells = mesh.n_cells
    n_sym = symmetric_components(dim)

    closure = CLOSURES[CLOSURE]()
    scheme = HessianCorrectedGradient(
        hessian_solve=CoupledBlockSweep(sweeps=SWEEPS), boundary_closure=closure
    )

    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (n_cells,))
    bvals = jnp.zeros((mesh.n_faces,))
    g0 = jax.random.normal(key, (n_cells, dim))
    u0 = jax.random.normal(key, (n_cells, n_sym))

    print(f"mesh: {n_cells} cells, {mesh.n_faces} faces, dim {dim}, n_sym {n_sym}")
    print(f"sweeps {SWEEPS}, repeats {REPEATS}, chain {CHAIN}, closure {CLOSURE}")
    print()

    def systems():
        return scheme._systems(mesh, geom, closure)

    # ---- the whole reconstruction, as a consumer pays for it.
    total = timed(lambda f, b: scheme.gradients(f, mesh, geom, b), field, bvals)

    # ---- the geometry-only prologue: everything a per-geometry cache could hold.
    def prologue():
        s = systems()
        inner = s.inner()
        return inner.preconditioner.inverse, s.outer_preconditioner(inner, False).inverse

    def prologue_inner_only():
        return systems().inner().preconditioner.inverse

    def prologue_with_schur():
        s = systems()
        inner = s.inner()
        return inner.preconditioner.inverse, s.outer_preconditioner(inner, True).inverse

    t_prologue = timed(prologue)
    t_inner = timed(prologue_inner_only)
    t_schur = timed(prologue_with_schur)

    # ---- the sweep, by differencing two sweep counts so the prologue cancels exactly.
    def sweep_at(n):
        def run(f, b):
            s = systems()
            inner = s.inner()
            return s.block_sweep(
                CoupledBlockSweep(sweeps=n),
                s.outer_preconditioner(inner, False),
                inner.preconditioner,
                s.gradient_rhs(f, b),
            )

        return run

    t_sweep_n = timed(sweep_at(SWEEPS), field, bvals)
    t_sweep_2n = timed(sweep_at(2 * SWEEPS), field, bvals)
    per_sweep = (t_sweep_2n - t_sweep_n) / SWEEPS

    # ---- the individual kernel evaluations inside one sweep, by chaining to defeat elimination.
    def chained(pick, x0, k):
        def run(x):
            s = systems()
            inner = s.inner()
            step = pick(s, inner)
            for _ in range(k):
                x = step(x)
            return x

        return run

    def per_eval(pick, x0):
        one = timed(chained(pick, x0, 1), x0)
        many = timed(chained(pick, x0, CHAIN), x0)
        return (many - one) / (CHAIN - 1)

    e_defect = per_eval(lambda s, i: lambda u: s.hessian_row_defect(g0, u), u0)
    e_ahh = per_eval(lambda s, i: i.operator, u0)
    e_grad = per_eval(lambda s, i: lambda g: s.gradient_row_defect(g, u0), g0)

    def coupled_step(s, i):
        def step(packed):
            return s.coupled.operator(packed)

        return step

    packed0 = jnp.concatenate([g0, u0], axis=1)
    e_coupled = per_eval(coupled_step, packed0)

    print(f"{'piece':<34} {'time (s)':>10} {'% of total':>11}")
    print("-" * 58)
    for name, t in [
        ("TOTAL reconstruction", total),
        ("  prologue (geometry only)", t_prologue),
        ("    of which inner() block", t_inner),
        ("  prologue + local_schur_block", t_schur),
        (f"  sweep x{SWEEPS}", t_sweep_n),
        ("    per sweep", per_sweep),
        ("      hessian_row_defect (1 HFT)", e_defect),
        ("      a_hh (1 HFT, no gradient)", e_ahh),
        ("      gradient_row_defect (1 GFT)", e_grad),
        ("      coupled (1 GFT + 1 HFT)", e_coupled),
    ]:
        print(f"{name:<34} {t:>10.4f} {100 * t / total:>10.1f}%")

    print()
    print(f"one gradient pass measured directly:        {e_grad:.4f} s")
    print(f"one gradient pass inferred (coupled-defect): {e_coupled - e_defect:.4f} s")
    print(f"one gradient + one Hessian pass:            {e_grad + e_defect:.4f} s")
    print(f"measured per sweep:                         {per_sweep:.4f} s")
    print()
    print(f"prologue share of one reconstruction: {100 * t_prologue / total:.1f}%")
    print(f"local_schur_block adds:               {t_schur - t_prologue:.4f} s")

    # ---- the sweep COUNT, which is the lever the per-sweep breakdown above says it is: the sweep
    # is most of the reconstruction, and its cost is the count rather than the work inside one.
    # ⚠️ Measure the rate exactly as `CoupledBlockSweep.calibrated` does -- same relaxation, same
    # preconditioner pair, same closure -- or this harness recommends a count the library would not
    # choose, and the disagreement looks like a finding rather than a harness bug.
    built = systems()
    built_inner = built.inner()
    built_outer = built.outer(SweptGradientSolve(warn_tol=None), built_inner)
    rate = contraction_rate(
        built.coupled_error(1.0, built_outer.preconditioner, built_inner.preconditioner)
    )
    print()
    print(f"coupled-sweep contraction rate on this mesh: {rate.rate:.4f}")
    for target in (1e-4, 1e-6, 1e-10):
        if rate.rate < 1:
            print(
                f"  sweeps to reach {target:.0e}: {math.ceil(math.log(target) / math.log(rate.rate))}"
            )
    print()
    print(f"{'sweeps':>7} {'time (s)':>10} {'vs default':>11} {'rel diff':>11}")
    reference = None
    for count in (SWEEPS, 12, 8, 6, 4):
        scheme_k = HessianCorrectedGradient(
            hessian_solve=CoupledBlockSweep(sweeps=count), boundary_closure=closure
        )
        run = jax.jit(lambda f, b, sc=scheme_k: sc.gradients(f, mesh, geom, b))
        got = run(field, bvals)
        if reference is None:
            reference = got
        t = timed(lambda f, b, sc=scheme_k: sc.gradients(f, mesh, geom, b), field, bvals)
        diff = float(jnp.linalg.norm(got - reference) / jnp.linalg.norm(reference))
        print(f"{count:>7} {t:>10.4f} {t / total:>10.2f}x {diff:>11.2e}")


if __name__ == "__main__":
    main()
