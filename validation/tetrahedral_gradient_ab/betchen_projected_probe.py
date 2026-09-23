"""A compact reconstruction exact for quadratics, built as the nearest thing to Betchen's operator.

Every linear, geometry-only gradient reconstruction is a fixed set of per-cell weights,
``g_P = sum_j w_Pj phi_j + sum_f v_Pf phi_f`` over cell values and boundary-face values. The
multiple-correction scheme is one such set on the cells within two face hops; it is exact for
quadratics and, on this tetrahedral mesh, anti-damps the Rhie–Chow pressure coupling. Betchen and
Straatman's (2010) converged reconstruction damps, but its weights reach the whole mesh (about 80 % of
their magnitude within two hops, 92 % within three).

This probe builds, per cell, the weights on the ``r``-hop stencil (cells within ``r`` face hops, and the
boundary faces those cells own) that are **exact for every polynomial up to quadratic** and **closest to
Betchen's own row** restricted to that stencil: ``min |w - w_B|`` subject to ``C w = t``, ``C`` the ten
monomials at the stencil points (scaled about the cell by its mean neighbour distance) and ``t`` their
derivatives at the cell. Betchen's operator comes from ``betchen_variants_probe.BetchenPrototype`` with
the library's reduction, solved directly.

Reported per scheme, on the damping operator of ``rhie_chow_sign_probe.py`` (unit weights,
Dirichlet-zero boundary values): flipped diagonals, the largest eigenvalue, mean nonzeros per row, the
worst quadratic error (with the quadratic's own boundary values) and the largest weight relative to
Betchen's. Schemes: the multiple-correction gradient (``OwnerGradient`` closure, ``SkewCorrectedGradient``
fallback, bound on geometry alone), converged Betchen, Betchen truncated to ``r`` hops without the
projection, the projection onto ``r`` hops, and -- to show whether aiming at Betchen matters -- the
minimum-norm exact weights on the same stencil (``w_B = 0``).

A second table reports accuracy beyond quadratics, which every scheme here but the truncation reproduces
exactly: the worst and volume-weighted mean gradient error, relative to the field's largest gradient,
on three analytic fields given their own boundary values -- ``smooth`` (a product of a sine, a cosine
and an exponential over the duct), ``singular`` (``1/|x - x0|`` with ``x0`` just outside the middle of a
wall, curvature varying sharply near it) and ``layer`` (a ``tanh`` layer a tenth of the duct deep against
a wall). One mesh, so these are error sizes at one resolution, not orders of accuracy.

Settings: ``BP_WEIGHT`` (default 0.5) Betchen's neighbour-averaged boundary Hessian weight; ``BP_REACH``
(comma-separated, default ``2,3``); ``BP_TARGETS`` (comma-separated, default ``converged``) the Betchen
rows projected -- ``converged``, or an integer ``k`` for ``k`` per-cell 9x9 block-Jacobi sweeps of the same
equations from zero, a local target whose rows reach ``k`` hops and cost no global solve (not the
library's coupled-sweep ordering, so not the same weights as its ``k``-sweep scheme);
``BP_BLEND`` (comma-separated, default ``0,1``) the fractions of the target the projection starts
from -- ``0`` is the minimum-norm exact weights (an unweighted quadratic least-squares fit on the
stencil), ``1`` the nearest exact weights to the target itself, and a value between trades one for
the other; ``BP_REFERENCES`` = ``1`` (default) or ``0`` to skip the multiple-correction and corrected Green-Gauss
rows; ``BP_BC`` = ``dirichlet`` (default, every boundary face carries a prescribed value) or
``pressure`` (the duct pressure's own conditions: zero gradient on the inlet and the walls, a
prescribed value at the outlet). Under ``pressure`` a wall face's datum is its normal derivative, so
the stencil's constraint there is the monomial's normal derivative at that face rather than its
value, and the damping operator's zero boundary data is the wall's true zero-gradient condition.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/betchen_projected_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from betchen_variants_probe import VARIANTS, BetchenPrototype, Damping, quadratic_case
from compare import POLYMESH
from scipy.sparse.linalg import splu

WEIGHT = float(os.environ.get("BP_WEIGHT", "0.5"))
REACHES = [int(r) for r in os.environ.get("BP_REACH", "2,3").split(",")]
TARGETS = os.environ.get("BP_TARGETS", "converged").split(",")
BLENDS = [float(a) for a in os.environ.get("BP_BLEND", "0,1").split(",")]
REFERENCES = os.environ.get("BP_REFERENCES", "1") == "1"
BC = os.environ.get("BP_BC", "dirichlet")


def monomials(points: np.ndarray) -> np.ndarray:
    """The ten monomials up to quadratic at scaled offsets ``points`` (m, 3) -> (10, m)."""
    x, y, z = points.T
    return np.stack([np.ones_like(x), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z])


def monomial_gradients(points: np.ndarray) -> np.ndarray:
    """Each monomial's gradient at scaled offsets ``points`` (m, 3) -> (10, m, 3)."""
    x, y, z = points.T
    zero, one = np.zeros_like(x), np.ones_like(x)
    return np.stack(
        [
            np.stack([zero, zero, zero], axis=-1),
            np.stack([one, zero, zero], axis=-1),
            np.stack([zero, one, zero], axis=-1),
            np.stack([zero, zero, one], axis=-1),
            np.stack([2 * x, zero, zero], axis=-1),
            np.stack([zero, 2 * y, zero], axis=-1),
            np.stack([zero, zero, 2 * z], axis=-1),
            np.stack([y, x, zero], axis=-1),
            np.stack([z, zero, x], axis=-1),
            np.stack([zero, z, y], axis=-1),
        ]
    )


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    neumann = np.zeros(mesh.n_faces, dtype=bool)
    if BC == "pressure":
        label = np.asarray(mesh.face_patches.label)
        for patch in ("inlet", "walls"):
            neumann |= label == mesh.face_patches.id_of(patch)
    elif BC != "dirichlet":
        raise ValueError(f"BP_BC must be 'dirichlet' or 'pressure', not {BC!r}")
    proto = BetchenPrototype(
        mesh,
        geometry,
        **VARIANTS["baseline"],
        weight=WEIGHT,
        reduction="weighted",
        neumann=neumann,
    )
    n = proto.n
    boundary_faces = np.flatnonzero(~proto.interior_np)
    face_owner = proto.owner_np[boundary_faces]
    xc, xf = np.asarray(proto.x), np.asarray(proto.x_ip)
    cells_q, face_data, exact = quadratic_case(proto)
    exact_g = exact[:, :3]
    damping = Damping(proto)
    neumann_face = neumann[boundary_faces]
    print(
        f"{n} cells, {boundary_faces.size} boundary faces ({int(neumann_face.sum())} carrying a "
        f"normal derivative under BP_BC={BC}); Betchen weight {WEIGHT}",
        flush=True,
    )

    a, b, b_b = proto.assemble()
    right = np.concatenate([b.toarray(), b_b[:, boundary_faces].toarray()], axis=1)
    targets = {}
    if "converged" in TARGETS:
        solved = splu(a.tocsc()).solve(right).reshape(n, 9, -1)[:, :3, :]
        targets["converged"] = (solved[:, :, :n], solved[:, :, n:])
        del solved
    sweeps = sorted(int(t) for t in TARGETS if t != "converged")
    if sweeps:
        blocks = np.stack([a[9 * q : 9 * q + 9, 9 * q : 9 * q + 9].toarray() for q in range(n)])
        preconditioner = sp.block_diag(np.linalg.inv(blocks), format="csr")
        state = np.zeros_like(right)
        for k in range(1, sweeps[-1] + 1):
            state = state + preconditioner @ (right - a @ state)
            if k in sweeps:
                gradient_rows = state.reshape(n, 9, -1)[:, :3, :].copy()
                targets[f"{k} sweeps"] = (gradient_rows[:, :, :n], gradient_rows[:, :, n:])
        del state
    del right
    first = next(iter(targets.values()))
    scale_w = np.abs(first[0]).max()

    lower, upper = xc.min(axis=0), xc.max(axis=0)
    extent = upper - lower
    depth = 0.1 * extent[1]
    source = np.array([0.5 * (lower[0] + upper[0]), lower[1] - depth, 0.5 * (lower[2] + upper[2])])

    def smooth(point):
        unit = (point - lower) / extent
        return jnp.sin(jnp.pi * unit[0]) * jnp.cos(jnp.pi * unit[1]) * jnp.exp(unit[2])

    def singular(point):
        return 1.0 / jnp.linalg.norm(point - source)

    def layer(point):
        return jnp.tanh((point[1] - lower[1]) / depth)

    volume = np.asarray(proto.volume)
    clear = ~proto.at_wall
    fields = {}
    for label, function in (("smooth", smooth), ("singular", singular), ("layer", layer)):
        values = np.asarray(jax.vmap(function)(jnp.asarray(xc)))
        face_values = np.asarray(jax.vmap(function)(jnp.asarray(xf[boundary_faces])))
        face_gradient = np.asarray(jax.vmap(jax.grad(function))(jnp.asarray(xf[boundary_faces])))
        face_normal = np.asarray(proto.normal)[boundary_faces]
        face_data_here = np.where(
            neumann_face, np.sum(face_gradient * face_normal, axis=-1), face_values
        )
        gradient = np.asarray(jax.vmap(jax.grad(function))(jnp.asarray(xc)))
        fields[label] = (values, face_data_here, gradient)
    accuracy = []

    rows = []

    def report(name, cell_weights, face_weights):
        flipped, largest, reach = damping(cell_weights)
        g = np.einsum("pin,n->pi", cell_weights, cells_q) + np.einsum(
            "pif,f->pi", face_weights, face_data[boundary_faces]
        )
        error = np.linalg.norm(g - exact_g, axis=-1).max() / np.abs(exact_g).max()
        biggest = max(np.abs(cell_weights).max(), np.abs(face_weights).max()) / scale_w
        rows.append((name, flipped, largest, reach, error, biggest))
        line = []
        for values, face_values, gradient in fields.values():
            computed = np.einsum("pin,n->pi", cell_weights, values) + np.einsum(
                "pif,f->pi", face_weights, face_values
            )
            local = np.linalg.norm(computed - gradient, axis=-1) / np.abs(gradient).max()
            line.append((local.max(), local[clear].max(), (local * volume).sum() / volume.sum()))
        accuracy.append((name, line))
        print(
            f"{name:38s} {flipped:6d} {largest:+11.3e} {reach:8.1f} {error:11.2e} {biggest:9.2f}",
            flush=True,
        )

    print(
        f"\n{'scheme':38s} {'flip':>6s} {'max eig':>11s} {'nnz/row':>8s} {'quad err':>11s} "
        f"{'max|w|':>9s}"
    )
    if REFERENCES:
        references(report, mesh, geometry, proto, boundary_faces)
    for label, (cells_w, faces_w) in targets.items():
        report(f"Betchen, {label}", cells_w, faces_w)

    # hop distance from every cell, by breadth-first search on the face graph
    for reach in REACHES:
        stencil_cells, stencil_faces = [], []
        for p in range(n):
            seen, front = {p}, {p}
            for _ in range(reach):
                front = {q for r in front for q in proto.adjacent[r]} - seen
                seen |= front
            cells = np.array(sorted(seen))
            stencil_cells.append(cells)
            stencil_faces.append(np.flatnonzero(np.isin(face_owner, cells)))

        systems, rank_short, worst_condition = [], 0, 0.0
        for p in range(n):
            cells, faces = stencil_cells[p], stencil_faces[p]
            h = np.mean(np.linalg.norm(xc[list(proto.adjacent[p])] - xc[p], axis=-1))
            face_points = xf[boundary_faces[faces]]
            scaled_cells = (xc[cells] - xc[p]) / h
            scaled_faces = (face_points - xc[p]) / h
            face_constraints = monomials(scaled_faces)
            if neumann_face[faces].any():
                normals = np.asarray(proto.normal)[boundary_faces[faces]]
                derivative = np.einsum("mfi,fi->mf", monomial_gradients(scaled_faces), normals) / h
                face_constraints = np.where(neumann_face[faces], derivative, face_constraints)
            constraints = np.concatenate([monomials(scaled_cells), face_constraints], axis=1)
            singular = np.linalg.svd(constraints, compute_uv=False)
            rank_short += int(singular[-1] < 1e-10 * singular[0])
            worst_condition = max(worst_condition, singular[0] / max(singular[-1], 1e-300))
            systems.append((constraints, np.linalg.pinv(constraints @ constraints.T), h))
        print(
            f"  ({reach} hops: median stencil {int(np.median([c.size for c in stencil_cells]))} "
            f"cells; {rank_short} cells whose monomials are rank-short; worst stencil "
            f"condition {worst_condition:.2e})",
            flush=True,
        )

        def project(
            aim_cells,
            aim_faces,
            *,
            blend,
            stencil_cells=stencil_cells,
            stencil_faces=stencil_faces,
            systems=systems,
        ):
            """Exact weights nearest to ``blend`` times the aim (``blend=0`` is the minimum-norm
            ones, ``1`` the aim itself), or the bare truncation when ``blend`` is ``None``."""
            out, out_b = np.zeros((n, 3, n)), np.zeros((n, 3, boundary_faces.size))
            for p in range(n):
                cells, faces = stencil_cells[p], stencil_faces[p]
                constraints, pseudo, h = systems[p]
                for i in range(3):
                    target = np.zeros(10)
                    target[1 + i] = 1.0 / h
                    aim = np.concatenate([aim_cells[p, i, cells], aim_faces[p, i, faces]])
                    if blend is None:
                        weights = aim
                    else:
                        start = blend * aim
                        weights = start + constraints.T @ (pseudo @ (target - constraints @ start))
                    out[p, i, cells] = weights[: cells.size]
                    out_b[p, i, faces] = weights[cells.size :]
            return out, out_b

        for label, (cells_w, faces_w) in targets.items():
            report(f"{label}: truncated to {reach} hops", *project(cells_w, faces_w, blend=None))
            for alpha in BLENDS:
                name = (
                    f"minimum-norm exact on {reach} hops"
                    if alpha == 0.0
                    else f"{label}: {alpha:g} x target, {reach} hops"
                )
                report(name, *project(cells_w, faces_w, blend=alpha))

    print(
        "\naccuracy beyond quadratics: |g - g_exact| / max|g_exact| -- "
        "worst all / worst interior / volume mean"
    )
    print(f"{'scheme':38s} " + " ".join(f"{label:>28s}" for label in fields))
    for name, line in accuracy:
        print(
            f"{name:38s} " + " ".join(f"{a:8.2e} {b:8.2e} {c:8.2e} " for a, b, c in line),
            flush=True,
        )


def references(report, mesh, geometry, proto, boundary_faces):
    """The multiple-correction and corrected Green-Gauss rows, from their operators."""
    n = proto.n
    mcg = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ).bind(mesh, geometry)
    mcg_cells = np.asarray(
        jax.jacfwd(
            lambda phi: mcg.reconstruct(phi, mesh, geometry, jnp.zeros(proto.fc.n_faces))[0]
        )(jnp.zeros(n))
    )
    mcg_faces = np.asarray(
        jax.jacfwd(lambda bv: mcg.reconstruct(jnp.zeros(n), mesh, geometry, bv)[0])(
            jnp.zeros(proto.fc.n_faces)
        )
    )[:, :, boundary_faces]
    report("multiple correction", mcg_cells, mcg_faces)
    del mcg_cells, mcg_faces
    cgg = CorrectedGreenGauss().bind(mesh, geometry)
    zero_faces = jnp.zeros(proto.fc.n_faces)
    cgg_cells = np.asarray(
        jax.jacfwd(lambda phi: cgg.gradients(phi, mesh, geometry, zero_faces))(jnp.zeros(n))
    )
    cgg_faces = np.asarray(
        jax.jacfwd(lambda bv: cgg.gradients(jnp.zeros(n), mesh, geometry, bv))(zero_faces)
    )[:, :, boundary_faces]
    report("corrected Green-Gauss", cgg_cells, cgg_faces)


if __name__ == "__main__":
    main()
