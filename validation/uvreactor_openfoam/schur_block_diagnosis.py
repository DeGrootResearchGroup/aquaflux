"""Why does the Schur-block preconditioner diverge on a handful of cells of a real mesh?

``HessianCorrectedGradient(local_schur_block=True)`` builds its outer preconditioner from the Schur
complement's own per-cell block rather than from ``A_gg``'s. On every synthetic mesh tested that is
better everywhere; on a 1.6M-cell snappyHexMesh reactor it is better on the median cell and
catastrophically wrong on a few -- the reconstruction of a quadratic reaches ``4.4e+18`` at its worst
cell, where the ``A_gg`` block reaches ``6.2e-02`` and an exact Krylov solve reaches ``2.4e-02``.

The operator is not at fault: an exact solve of the same system is fine. What fails is the fixed-count
Richardson sweep under this preconditioner, which is what a preconditioner that is catastrophically
wrong on one row does to a stationary iteration -- a Krylov method would merely spend iterations.

This script finds those cells, describes their geometry, and compares three blocks on them:

* the **true** Schur block, obtained by lighting one cell's degree of freedom and reading back that
  cell's own row. No neighbour can contribute to it, so it is the true block whatever preconditioner
  is in force -- which makes it the one measurement here that cannot be circular;
* the cell-local **approximation** the preconditioner actually builds;
* ``A_gg``'s block, which does not diverge.

That comparison separates the candidate mechanisms. If the approximation is near-singular where the
true block is not, the dropped neighbour paths are the cause and a per-cell conditioning fallback is
the fix. If the true block is itself near-singular, then ``A_gg``'s block survives only by being
wrong in a harmless direction, and the answer is a Krylov outer solve on such meshes rather than a
patched preconditioner.

Usage
-----
    UV_MESH=<path-to-polyMesh> validation/run_case.sh validation/uvreactor_openfoam/schur_block_diagnosis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.io import read_openfoam  # noqa: E402
from aquaflux.mesh.quality import closed_cell_residual, face_planarity  # noqa: E402
from aquaflux.schemes import HessianCorrectedGradient, SweptGradientSolve  # noqa: E402
from aquaflux.schemes.gradient import cell_diagonal_block  # noqa: E402,F401

INNER = int(os.environ.get("UV_INNER", "12"))
WORST = int(os.environ.get("UV_WORST", "12"))
OUTER = int(os.environ.get("UV_OUTER", "20"))
RATE_ITERS = int(os.environ.get("UV_RATE_ITERS", "20"))
OMEGA = os.environ.get("UV_OMEGA", "1.0,0.8,0.5,0.25,0.1")
# The reconstruction arms alone, for a question that only needs them — the spectral sections cost
# several times what they do, and re-running everything to re-ask one thing wastes the machine.
QUICK = os.environ.get("UV_QUICK", "") not in ("", "0")


def quadratic(points, centre, extent):
    u = (points - centre) / extent
    return 0.6 + 1.7 * u[:, 0] - 1.1 * u[:, 1] + 0.8 * u[:, 2] + 0.4 * u[:, 0] ** 2


def quadratic_gradient(points, centre, extent):
    u = (points - centre) / extent
    return (
        np.stack(
            [1.7 + 0.8 * u[:, 0], -1.1 * np.ones_like(u[:, 0]), 0.8 * np.ones_like(u[:, 0])],
            axis=-1,
        )
        / extent
    )


def power_iteration(apply_fn, shape, iters, seed=0):
    """Dominant eigenvalue and eigenvector of a linear map, by normalized power iteration.

    Parameters
    ----------
    apply_fn : callable
        ``(n_cells, dim) -> (n_cells, dim)``, the linear map to iterate.
    shape : tuple of int
        Shape of the vector the map acts on.
    iters : int
        Iterations to run. The estimate at half the budget is returned beside the final one, because
        a power iteration that has not settled reports a number that looks just as definite as one
        that has -- the two agreeing is the only evidence the answer means anything.
    seed : int
        Seed for the starting vector.

    Returns
    -------
    rate : float
        Growth factor of the last iteration -- the dominant eigenvalue's modulus.
    half : float
        The same estimate at half the budget.
    rayleigh : float
        ``v.(A v)`` at the final iterate, which carries the dominant eigenvalue's SIGN. Negative,
        where ``A`` is the amplification ``P^-1 S``, means the preconditioner is indefinite relative
        to the operator in that direction -- and then no positive relaxation can stabilize it.
    vector : ndarray
        The final iterate, normalized.
    """
    rng = np.random.default_rng(seed)
    v = jnp.asarray(rng.standard_normal(shape))
    v = v / jnp.linalg.norm(v)
    rate = half = rayleigh = float("nan")
    for i in range(iters):
        w = apply_fn(v)
        nrm = float(jnp.linalg.norm(w))
        if not np.isfinite(nrm) or nrm == 0.0:
            return nrm, half, rayleigh, np.asarray(v)
        rayleigh = float(jnp.sum(v * w))
        rate = nrm
        if i + 1 == max(iters // 2, 1):
            half = nrm
        v = w / nrm
    return rate, half, rayleigh, np.asarray(v)


def localization(vector, n_cells):
    """How many cells carry a mode -- the discriminator between a local and a global failure.

    A spectral radius is a whole-mesh number, and is perfectly compatible with a cause confined to a
    handful of cells, so ``rho > 1`` on its own cannot tell the two apart. The participation ratio
    can: it is the effective number of cells the eigenvector occupies, near ``1`` for a mode pinned
    to one cell and near ``n_cells`` for one spread over the mesh.

    Parameters
    ----------
    vector : ndarray
        A normalized eigenvector, shape ``(n_cells, dim)``.
    n_cells : int
        Cell count of the mesh.

    Returns
    -------
    dict or None
        ``cells`` (participation ratio), ``top1`` / ``top10`` / ``top1000`` (share of the mode's
        energy on that many cells), and ``order`` (cells by descending share). ``None`` if the
        vector is not finite.
    """
    mass = np.linalg.norm(np.asarray(vector).reshape(n_cells, -1), axis=-1) ** 2
    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    p = mass / total
    order = np.argsort(p)[::-1]
    return {
        "cells": 1.0 / float(np.sum(p**2)),
        "top1": float(p[order[0]]),
        "top10": float(p[order[:10]].sum()),
        "top1000": float(p[order[:1000]].sum()),
        "order": order,
    }


def dilate(seed_mask, owner_int, nb_int, rounds):
    """Grow a cell mask outward by ``rounds`` face-neighbour hops.

    A fixed sweep count propagates a bad row outward one cell per sweep, so a blow-up confined to a
    few cells contaminates its whole ``sweeps``-hop neighbourhood and no further. This is what tests
    that reading: if the diverging cells are the dilation of a much smaller set, that smaller set is
    the cause and the population is its shadow.

    Parameters
    ----------
    seed_mask : ndarray of bool
        Per-cell mask to grow, shape ``(n_cells,)``.
    owner_int, nb_int : ndarray of int
        Owner and neighbour cell of each interior face.
    rounds : int
        Hops to grow.

    Returns
    -------
    ndarray of bool
        The grown mask.
    """
    m = seed_mask.copy()
    for _ in range(rounds):
        nxt = m.copy()
        nxt[owner_int] |= m[nb_int]
        nxt[nb_int] |= m[owner_int]
        m = nxt
    return m


def main() -> None:
    mesh_path = os.environ.get("UV_MESH")
    if mesh_path is None:
        raise SystemExit("set UV_MESH to a polyMesh directory")
    mesh = read_openfoam(Path(mesh_path))
    geom = mesh.geometry()
    print(f"mesh: {mesh.n_cells} cells, {mesh.n_faces} faces", flush=True)

    x = np.asarray(geom.cell.centroid)
    centre, extent = x.mean(axis=0), max(float(np.abs(x - x.mean(axis=0)).max()), 1e-300)
    field = jnp.asarray(quadratic(x, centre, extent))
    bvals = jnp.asarray(quadratic(np.asarray(geom.face.centroid), centre, extent))
    exact = quadratic_gradient(x, centre, extent)

    inner = SweptGradientSolve(sweeps=INNER, warn_tol=None)
    errors = {}
    for label, flag in (("schur", True), ("a_gg", False)):
        grad = np.asarray(
            HessianCorrectedGradient(
                hessian_solver=inner, local_schur_block=flag, coupled_sweep=None
            ).gradients(field, mesh, geom, bvals)
        )
        errors[label] = np.linalg.norm(grad - exact, axis=-1) / np.linalg.norm(exact, axis=-1)
        print(
            f"  {label:5s} median {np.median(errors[label]):.3e}  "
            f"p99 {np.percentile(errors[label], 99):.3e}  max {errors[label].max():.3e}",
            flush=True,
        )

    # HOW MANY cells are bad, not just how bad the worst is -- one pathological cell is a guard, a
    # population is a design fault, and the max alone cannot tell those apart.
    bad = errors["schur"] > 1.0
    n_bad = int(bad.sum())
    # Counted for BOTH arms, because which one diverges is not a fixed property of the scheme: under
    # an unsymmetrized Hessian it was the Schur block, and under the symmetric one it is `A_gg`'s.
    # A count reported for one arm only cannot see that swap, and reads as "fixed" when it is not.
    print(
        f"\n  cells above 100% error — Schur block: {n_bad} | A_gg block: "
        f"{int((errors['a_gg'] > 1.0).sum())} | of {mesh.n_cells}"
    )
    # None is the outcome this harness exists to reach, so it must not be the one that crashes it.
    worst_a_gg = f"{errors['a_gg'][bad].max():.3e}" if n_bad else "n/a — none diverge"
    print(f"  the Schur block's diverging cells under the A_gg block: max {worst_a_gg}", flush=True)

    if QUICK:
        print("\n  UV_QUICK set — stopping after the reconstruction arms.", flush=True)
        return

    worst = np.argsort(errors["schur"])[::-1][:WORST]
    volume = np.asarray(geom.cell.volume)
    planarity = np.asarray(face_planarity(mesh))
    closure = np.asarray(closed_cell_residual(mesh))
    owner, neighbour = np.asarray(mesh.face_cells.owner), np.asarray(mesh.face_cells.neighbour)
    faces_per_cell = np.bincount(owner, minlength=mesh.n_cells) + np.bincount(
        neighbour[neighbour >= 0], minlength=mesh.n_cells
    )
    # worst planarity among each cell's own faces -- a cell-level view of a face-level metric
    worst_planarity = np.ones(mesh.n_cells)
    np.minimum.at(worst_planarity, owner, planarity)
    np.minimum.at(worst_planarity, neighbour[neighbour >= 0], planarity[neighbour >= 0])

    print("\n  the worst cells, and how they differ from the mesh as a whole")
    print(
        f"    {'cell':>9} {'err schur':>10} {'err a_gg':>10} {'volume':>10} "
        f"{'vol/med':>9} {'faces':>6} {'planarity':>10} {'closure':>10}"
    )
    for c in worst:
        print(
            f"    {c:9d} {errors['schur'][c]:10.2e} {errors['a_gg'][c]:10.2e} {volume[c]:10.2e} "
            f"{volume[c] / np.median(volume):9.2e} {faces_per_cell[c]:6d} "
            f"{worst_planarity[c]:10.4f} {closure[c]:10.2e}",
            flush=True,
        )
    # `C` -- the Hessian equation's reduced per-cell block -- is the only new ingredient the Schur
    # correction brings: `A_gH C^-1 A_Hg`. `A_gg`'s block never forms `C^-1`, which is why it survives
    # where this does not, so its conditioning is the first thing to rule in or out.
    #
    # ⚠️ The hypothesis this was written to test -- that a cell with too few faces cannot determine
    # every Hessian component, leaving `C` near-singular there -- is REFUTED and the numbers below are
    # what refuted it: `cond C` is ~1.2 mesh-wide and at worst 3.37 over every four-faced cell, the
    # best-conditioned group on the mesh. Since the Hessian is now solved as its six independent
    # symmetric components, `C` is also symmetric positive definite by construction, so it is reported
    # to confirm that rather than to test it.
    systems = HessianCorrectedGradient._systems(mesh, geom)
    inner_system = systems.inner()
    hessian_block = np.linalg.inv(np.asarray(inner_system.preconditioner.inverse))
    plain_block = np.linalg.inv(
        np.asarray(systems.outer(inner, inner_system, False).preconditioner.inverse)
    )
    schur_block = np.linalg.inv(
        np.asarray(systems.outer(inner, inner_system, True).preconditioner.inverse)
    )
    cond_c = np.linalg.cond(hessian_block)
    cond_plain = np.linalg.cond(plain_block)
    cond_schur = np.linalg.cond(schur_block)
    print("\n  conditioning: is `C` the culprit?")
    print(f"    {'cell':>9} {'faces':>6} {'cond C':>11} {'cond A_gg':>11} {'cond Schur':>11}")
    for c in worst[:8]:
        print(
            f"    {c:9d} {faces_per_cell[c]:6d} {cond_c[c]:11.3e} "
            f"{cond_plain[c]:11.3e} {cond_schur[c]:11.3e}",
            flush=True,
        )
    print(
        f"    mesh-wide medians: cond C {np.median(cond_c):.3e} | "
        f"cond A_gg {np.median(cond_plain):.3e} | cond Schur {np.median(cond_schur):.3e}"
    )
    for n_faces in (4, 5, 6):
        pick = faces_per_cell == n_faces
        if pick.sum():
            print(
                f"    cells with {n_faces} faces (n={int(pick.sum())}): "
                f"cond C median {np.median(cond_c[pick]):.3e} max {cond_c[pick].max():.3e}",
                flush=True,
            )
    # ⚠️ THE PRECONDITIONER-INDEPENDENT CHECK, and the one that discriminates. Conditioning does not
    # explain the failures: the three worst cells carry a badly conditioned Schur block, but others
    # fail just as hard with a perfectly conditioned one. A block can be well conditioned and simply
    # WRONG -- a poor approximation of the true local Schur complement makes `I - P^-1 S` expansive on
    # that row, and twenty sweeps of a modest amplification is an enormous number.
    #
    # Lighting ONE cell's degree of freedom and reading back THAT cell's own row gives the true block:
    # no neighbour can contribute to it, so it is the truth whatever preconditioner is in force. Done
    # for a handful of named cells it costs `dim` operator applies each, rather than the `n * dim` a
    # whole-mesh extraction would.
    outer_system = systems.outer(inner, inner_system, True)
    print(
        "\n  is the block WRONG rather than ill-conditioned? (relative error vs the true S block)"
    )
    print(f"    {'cell':>9} {'faces':>6} {'A_gg block':>12} {'Schur block':>12} {'cond Schur':>11}")
    basis = np.eye(dim := mesh.dim)
    for c in worst[:8]:
        columns = []
        for k in range(dim):
            probe = jnp.zeros((mesh.n_cells, dim)).at[c, k].set(1.0)
            columns.append(np.asarray(outer_system.operator(probe))[c])
        true_block = np.stack(columns, axis=-1)
        scale = np.linalg.norm(true_block)
        err_plain = np.linalg.norm(plain_block[c] - true_block) / scale
        err_schur = np.linalg.norm(schur_block[c] - true_block) / scale
        print(
            f"    {c:9d} {faces_per_cell[c]:6d} {err_plain:12.3e} {err_schur:12.3e} "
            f"{cond_schur[c]:11.3e}",
            flush=True,
        )
    _ = basis

    # ---- DOES THE CORRECTION COLLAPSE THE BLOCK? The Schur block is `A_gg`'s block MINUS a
    # correction, so on a cell where the two nearly cancel its smallest singular value collapses
    # while `A_gg`'s does not -- and the preconditioner is that block's INVERSE, so a collapse
    # there amplifies every neighbour coupling on that row. The true global Schur complement need
    # not be near-singular in that direction: the neighbour paths this per-cell block drops are what
    # restore it. That is how a block can be closer to the truth and worse to invert.
    smin_plain = np.linalg.svd(plain_block, compute_uv=False)[:, -1]
    smin_schur = np.linalg.svd(schur_block, compute_uv=False)[:, -1]
    shrink = smin_schur / np.maximum(smin_plain, 1e-300)
    print("\n  does the correction COLLAPSE the block? (sigma_min, and what P^-1 grows by)")
    print(f"    {'cell':>9} {'faces':>6} {'smin A_gg':>11} {'smin Schur':>11} {'shrink':>10}")
    for c in worst[:6]:
        print(
            f"    {c:9d} {faces_per_cell[c]:6d} {smin_plain[c]:11.3e} {smin_schur[c]:11.3e} "
            f"{shrink[c]:10.3e}",
            flush=True,
        )
    print(
        f"    mesh-wide: shrink median {np.median(shrink):.3e} min {shrink.min():.3e} | "
        f"cells below 1e-1: {int((shrink < 1e-1).sum())} | below 1e-2: {int((shrink < 1e-2).sum())}"
    )

    # ---- IS THE DIVERGING SET THE NEIGHBOURHOOD OF A MUCH SMALLER ONE? The outer solve runs a
    # fixed count of sweeps, and one sweep moves information across one face. So a row that
    # amplifies contaminates its `sweeps`-hop neighbourhood and no further, and a diverging
    # population far larger than the collapsed one is what a few bad rows look like after 20 sweeps.
    interior = neighbour >= 0
    owner_int, nb_int = owner[interior], neighbour[interior]
    for tau in (1e-1, 1e-2):
        seed = shrink < tau
        grown = dilate(seed, owner_int, nb_int, OUTER)
        caught = int((bad & grown).sum())
        print(
            f"\n  cells with shrink < {tau:.0e}: {int(seed.sum())} -> dilated {OUTER} hops: "
            f"{int(grown.sum())} cells, covering {caught} of the {n_bad} diverging "
            f"({100.0 * caught / max(n_bad, 1):.1f}%)",
            flush=True,
        )

    # ---- IS THE FAILURE A BOUNDARY EFFECT? Betchen & Straatman close the Hessian at a boundary by
    # taking the owner's, and note that this can leave the system under-determined -- their example
    # being a mesh one cell thick in some direction, where the second derivative across it is
    # arbitrary. Their remedy is an inverse-distance average of the Hessian over interior neighbours
    # not themselves adjacent to a boundary; this scheme implements the simpler closure and not that
    # remedy, so if the closure is implicated the diverging cells should sit against the boundary.
    #
    # Reported as an ENRICHMENT rather than a share: on a refined mesh most cells are interior, so
    # "80% of the diverging cells are near a boundary" means nothing without the base rate to divide
    # by. An enrichment near 1 is no association whatever the share looks like.
    boundary_cell = np.zeros(mesh.n_cells, dtype=bool)
    boundary_cell[owner[neighbour < 0]] = True
    base_rate = n_bad / mesh.n_cells
    print(
        f"\n  boundary proximity: {int(boundary_cell.sum())} cells own a boundary face "
        f"({100.0 * boundary_cell.mean():.1f}% of the mesh); diverging base rate "
        f"{100.0 * base_rate:.4f}%"
    )
    print(f"    {'within':>8} {'cells':>10} {'diverging':>10} {'rate':>9} {'enrichment':>11}")
    near = boundary_cell
    for hops in range(4):
        if hops:
            near = dilate(near, owner_int, nb_int, 1)
        hit = int((bad & near).sum())
        rate = hit / max(int(near.sum()), 1)
        print(
            f"    {hops:8d} {int(near.sum()):10d} {hit:10d} {100.0 * rate:8.4f}% "
            f"{rate / max(base_rate, 1e-300):11.2f}x",
            flush=True,
        )
    # The worst cells individually, since an enrichment over thousands of cells can hide the handful
    # that actually diverge by 1e+18.
    hop_of = np.full(mesh.n_cells, -1)
    reach = boundary_cell.copy()
    hop_of[reach] = 0
    for hops in range(1, 8):
        grown = dilate(reach, owner_int, nb_int, 1)
        hop_of[grown & ~reach] = hops
        reach = grown
    print(f"    the {WORST} worst cells sit at boundary hop: {sorted(hop_of[worst].tolist())}")

    # ---- THE GLOBAL CONTRACTION RATE, and the two things it cannot tell you on its own.
    # `rho(I - P^-1 S)` decides whether the sweep converges, but it is a whole-mesh number: it
    # cannot say whether the responsible mode sits on ten cells or a million, and it cannot say
    # whether under-relaxation would fix it. The eigenvector's participation ratio answers the
    # first; the SIGN of the dominant eigenvalue of the amplification `P^-1 S` answers the second.
    plain_outer = systems.outer(inner, inner_system, False)
    schur_outer = outer_system
    shape = (mesh.n_cells, dim)

    print("\n  the contraction rate -- rho(I - P^-1 S), which decides whether a sweep converges")
    modes, rho = {}, {}
    for label, sysm in (("A_gg", plain_outer), ("Schur", schur_outer)):
        step = jax.jit(lambda v, s=sysm: v - s.preconditioner.apply(s.operator(v)))
        rate, half, rq, vec = power_iteration(step, shape, RATE_ITERS)
        verdict = "CONVERGES" if rate < 1.0 else "DIVERGES"
        # The GOVERNING eigenvalue of `P^-1 S`, with its sign, read off THIS iteration rather than a
        # separate one over `P^-1 S`. The dominant mode here is the one furthest from 1 -- which is
        # what the sweep's rate is -- and `I - P^-1 S` acts on it as `1 - lambda`, so the Rayleigh
        # quotient gives `lambda` directly. A power iteration over `P^-1 S` answers a DIFFERENT
        # question: it returns the largest-MODULUS eigenvalue, which on this operator is the cluster
        # near +1 where the preconditioner is nearly exact, and that one says nothing about the rate.
        lam = 1.0 - rq
        relax = (
            "relaxation could stabilize it"
            if lam > 0
            else "NO positive relaxation stabilizes it (1 - w*lambda > 1 for every w > 0)"
        )
        print(
            f"    {label:6s} block: rho = {rate:.4f}  ({verdict})  half-budget {half:.4f}  "
            f"| governing lambda = {lam:+.4f} -- {relax}",
            flush=True,
        )
        modes[label] = localization(vec, mesh.n_cells)
        rho[label] = rate

    print("\n  WHERE does that mode live? (participation ratio = effective cells carrying it)")
    for label, loc in modes.items():
        if loc is None:
            print(f"    {label:6s}: eigenvector not finite")
            continue
        # Each arm's mode is checked against ITS OWN diverging set. Against a single shared mask the
        # statistic reads zero for whichever arm the mask does not describe, which looks like "the
        # mode misses the failures" when it means "the mask is the other arm's".
        own = errors["a_gg" if label == "A_gg" else "schur"] > 1.0
        overlap = int(own[loc["order"][:1000]].sum())
        print(
            f"    {label:6s}: {loc['cells']:12.1f} cells of {mesh.n_cells} | "
            f"top1 {loc['top1']:.3f} top10 {loc['top10']:.3f} top1000 {loc['top1000']:.3f} | "
            f"of its top 1000 cells, {overlap} diverge under that same block",
            flush=True,
        )

    # ---- CAN UNDER-RELAXATION FIX IT? Betchen and Straatman solve this reconstruction by
    # under-relaxed block-Jacobi and state that a relaxation strictly inside (0, 1] is needed for
    # convergence on an arbitrary grid; the solver here runs undamped. Relaxation scales every
    # eigenvalue of the amplification, so `rho(I - w P^-1 S) = max|1 - w lambda|`: it can stabilize
    # a mode whose `lambda` is positive and too large, and can do NOTHING for one whose `lambda` is
    # negative, since `1 - w lambda > 1` for every positive `w`. So the sign is measured first and
    # the ladder second -- the sign says whether the ladder can succeed before it is spent.
    # ---- CROSS-CHECK the governing eigenvalue against an independent bound. A power iteration over
    # `P^-1 S` returns its largest-modulus eigenvalue, which is NOT the governing one -- but it is an
    # upper bound on every modulus, so it rules out one of the two values consistent with `rho`:
    # `rho = |1 - lambda|` admits `lambda = 1 - rho` and `lambda = 1 + rho`, and whichever exceeds
    # this bound is impossible. When the two routes disagree on the survivor, neither is trustworthy.
    print("\n  cross-check: |lambda|_max over P^-1 S, which bounds the candidates rho admits")
    for label, sysm in (("A_gg", plain_outer), ("Schur", schur_outer)):
        amp = jax.jit(lambda v, s=sysm: s.preconditioner.apply(s.operator(v)))
        rate, half, _, _ = power_iteration(amp, shape, RATE_ITERS)
        measured = rho.get(label, float("nan"))
        feasible = [c for c in (1.0 - measured, 1.0 + measured) if abs(c) <= rate * 1.02]
        survivor = f"{feasible[0]:+.4f}" if len(feasible) == 1 else f"{len(feasible)} candidates"
        print(
            f"    {label:6s}: |lambda|_max = {rate:.4f} (half {half:.4f})  "
            f"| rho admits {1.0 - measured:+.4f} / {1.0 + measured:+.4f} -> {survivor}",
            flush=True,
        )

    print("\n  rho(I - w P^-1 S) under the Schur block, over Betchen's relaxation range")
    # ⚠️ ONE jitted function for the whole ladder, with the relaxation as a traced ARGUMENT. Building
    # `jax.jit(lambda v: ... w ...)` per rung closes `w` in as a constant, and a fresh `jax.jit` object
    # starts with an empty cache -- so every rung recompiles a program that captures 2.3 GB of
    # constants at this mesh size, which costs far more than the twenty applies it then runs.
    relaxed = jax.jit(
        lambda v, w: v - w * schur_outer.preconditioner.apply(schur_outer.operator(v))
    )
    for omega in [float(w) for w in OMEGA.split(",")]:
        rate, half, _, _ = power_iteration(lambda v, w=omega: relaxed(v, w), shape, RATE_ITERS)
        verdict = "CONVERGES" if rate < 1.0 else "DIVERGES"
        print(
            f"    w = {omega:4.2f}: rho = {rate:9.4f}  ({verdict})  half-budget {half:9.4f}",
            flush=True,
        )

    print(
        f"\n  mesh-wide: volume median {np.median(volume):.2e} min {volume.min():.2e} | "
        f"faces/cell median {int(np.median(faces_per_cell))} max {faces_per_cell.max()} | "
        f"planarity min {planarity.min():.4f} | closure max {closure.max():.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
