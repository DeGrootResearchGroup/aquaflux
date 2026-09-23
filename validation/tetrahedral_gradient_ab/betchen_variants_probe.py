"""Two changes to Betchen's coupled reconstruction, measured against it on this tetrahedral mesh.

Betchen and Straatman (2010) reconstruct each cell's gradient ``g`` and Hessian ``H`` from two coupled
equations: a Green–Gauss sum of face values (the gradient equation) and a Green–Gauss sum of face
gradients (the Hessian equation). Their sweep converges slowly on these tetrahedra, and this probe asks
whether either of two reformulations helps while staying exact for quadratic fields:

* **condition closure** -- at a boundary face the Hessian equation needs the face gradient, which the
  paper extrapolates from the owner with some Hessian (``g_P + H_bnd d``). Here its NORMAL component
  comes from the boundary data instead: ``(phi_f - phi_P - g_P . d_t - 1/2 d H_P d) / (d . n) + n H_P d``,
  the one-sided difference against the prescribed value, curvature-corrected so it stays exact for a
  quadratic; the tangential component keeps the owner extrapolation. No neighbour average is used.
  This is the Dirichlet form (every boundary face carries a prescribed value here).
* **local gradient equation** -- the gradient equation's face value carries the gradient across the face
  centroid's offset from the owner--neighbour line, blending BOTH cells' gradients and Hessians. Here
  each cell's equation instead Taylor-expands its own: ``g_P + H_P (x_line - x_P)``, and ``H_P`` in the
  curvature term. Exact for a quadratic, and the gradient equation then couples to no neighbour's
  unknowns at all; the Hessian equation, which must read neighbours' gradients, is unchanged.

:class:`BetchenPrototype` holds the equations and assembles them; ``betchen_projected_probe.py`` reuses
it. The linear system ``A u = B phi + B_b phi_b`` on ``u = [g, H]`` (3 + 6 unknowns per cell, ``phi_b``
the boundary-face values) is assembled sparse by colouring and solved directly; the sweep compared
here is a per-cell 9x9 block-Jacobi iteration whose asymptotic rate is the spectral radius of
``I - P^-1 A``. The Hessian equation's nine rows are reduced to six either as the library does --
weighted by the cell's own block, ``(A_P E)^T R / V`` with ``E`` the symmetric expansion -- or by their
symmetric part. Every face here is a planar triangle, so the warp terms of a general polyhedral face
vanish and are omitted.

Per variant: the equations' residual on an exact quadratic (must be round-off), the converged
reconstruction's quadratic error, the sweep's rate and where its slowest mode lives, the error after
``k`` sweeps on a smooth and a rough field, and at ``k`` = 1, 2, 3, 5 and converged the Rhie–Chow
damping operator's flipped diagonals and largest eigenvalue (unit weights, Dirichlet-zero boundary
values, as in ``rhie_chow_sign_probe.py``) with the gradient operator's mean nonzeros per row. The
baseline's converged gradient is compared with the library's ``HessianCorrectedGradient``.

Settings: ``BV_WEIGHT`` (default 0.5) the neighbour-averaged closure's weight; ``BV_REDUCTION`` =
``weighted`` (default, the library's) or ``symmetric``; ``BV_VARIANTS`` (comma-separated names from
``VARIANTS``, default all).

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/betchen_variants_probe.py
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
from aquaflux.schemes import AveragedNeighbourHessian, CoupledBlockSweep, HessianCorrectedGradient
from aquaflux.schemes.interpolation import interpolation_factor
from scipy.sparse.linalg import LinearOperator, eigs, splu

SYM = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
VARIANTS = {
    "baseline": {"local_gradient": False, "condition": False},
    "condition closure": {"local_gradient": False, "condition": True},
    "local gradient eq": {"local_gradient": True, "condition": False},
    "both": {"local_gradient": True, "condition": True},
}
SWEEP_ERRORS = (1, 2, 3, 5, 10, 20, 40)
OPERATOR_SWEEPS = (1, 2, 3, 5)


def to_tensor(h6: jnp.ndarray) -> jnp.ndarray:
    """Six independent components -> symmetric ``(..., 3, 3)``."""
    rows = []
    for i in range(3):
        row = [h6[..., SYM.index((min(i, j), max(i, j)))] for j in range(3)]
        rows.append(jnp.stack(row, axis=-1))
    return jnp.stack(rows, axis=-2)


def to_six(m: jnp.ndarray) -> jnp.ndarray:
    """The symmetric part of ``(..., 3, 3)`` as its six independent components."""
    sym = 0.5 * (m + jnp.swapaxes(m, -1, -2))
    return jnp.stack([sym[..., i, j] for i, j in SYM], axis=-1)


class BetchenPrototype:
    """Betchen's equations on one mesh, with the two reformulations as switches.

    Parameters
    ----------
    mesh, geometry
        The mesh and its metrics.
    local_gradient, condition : bool
        The two reformulations (module docstring).
    weight : float
        The neighbour-averaged boundary Hessian's weight, used where ``condition`` is off.
    reduction : str
        ``"weighted"`` (the library's) or ``"symmetric"``.
    neumann : ndarray of bool, optional
        Per face: this boundary face's datum is the PRESCRIBED NORMAL DERIVATIVE rather than a value
        (a zero-gradient or Neumann condition). Its face value is then Taylor-extrapolated from the
        owner -- exact for a quadratic -- and the Hessian equation's face gradient takes its normal
        component from the datum and its tangential component from the owner, which needs no
        neighbour average. Default: every boundary face carries a value.
    """

    def __init__(
        self, mesh, geometry, *, local_gradient, condition, weight, reduction, neumann=None
    ):
        self.mesh, self.geometry = mesh, geometry
        self.local_gradient, self.condition = local_gradient, condition
        self.weight, self.reduction = weight, reduction
        n = self.n = mesh.n_cells
        fc = self.fc = mesh.face_cells
        self.owner_np, self.neighbour_np = np.asarray(fc.owner), np.asarray(fc.neighbour)
        self.interior_np = self.neighbour_np >= 0
        self.at_wall = np.isin(np.arange(n), self.owner_np[~self.interior_np])
        x = self.x = geometry.cell.centroid
        self.x_ip = geometry.face.centroid
        self.s = fc.neighbour_centroid(x) - x[fc.owner]
        self.f = interpolation_factor(fc, geometry)
        self.skew = self.x_ip - (x[fc.owner] + self.f[:, None] * self.s)
        s, f = self.s, self.f
        self.curvature = self.skew[:, :, None] * self.skew[:, None, :] - (f * (1.0 - f))[
            :, None, None
        ] * (s[:, :, None] * s[:, None, :])
        self.normal = geometry.face.normal
        self.area = self.normal * geometry.face.area[:, None]
        self.d_own = self.x_ip - x[fc.owner]
        self.d_nb = self.x_ip - fc.neighbour_centroid(x)
        self.volume = geometry.cell.volume
        self.along = jnp.sum(self.d_own * self.normal, axis=-1)
        self.d_tangent = self.d_own - self.along[:, None] * self.normal
        self.inverse_distance = jnp.where(
            fc.interior,
            1.0 / jnp.linalg.norm(jnp.where(fc.interior[:, None], s, 1.0), axis=-1),
            0.0,
        )
        self.neumann = (
            jnp.zeros(fc.n_faces, dtype=bool) if neumann is None else jnp.asarray(neumann)
        )
        self.neumann_np = np.asarray(self.neumann)
        self.trash = jnp.where(fc.interior, fc.safe_neighbour, n)
        self.weight_sum = self.to_cells(self.inverse_distance, self.inverse_distance)
        self._colour()
        self.reducer = None
        if reduction == "weighted":
            self.reducer = jnp.asarray(self._own_hessian_block())
        elif reduction != "symmetric":
            raise ValueError(f"reduction must be 'weighted' or 'symmetric', not {reduction!r}")

    # -- face machinery ---------------------------------------------------------------------------
    def to_cells(self, own_part, neighbour_part):
        fc, n = self.fc, self.n
        shape = own_part.shape[1:]
        total = jax.ops.segment_sum(own_part, fc.owner, num_segments=n + 1)
        total = total + jax.ops.segment_sum(
            jnp.where(fc.interior.reshape((-1,) + (1,) * len(shape)), neighbour_part, 0.0),
            self.trash,
            num_segments=n + 1,
        )
        return total[:n]

    def neighbour_average(self, h):
        fc, w = self.fc, self.inverse_distance[:, None, None]
        gathered = self.to_cells(w * h[fc.safe_neighbour], w * h[fc.owner])
        return gathered / self.weight_sum[:, None, None]

    # -- the equations --------------------------------------------------------------------------------
    def unreduced(self, u, phi, bvals):
        """``(r_g (n, 3), r_H (n, 3, 3))`` -- both equations, the Hessian one before reduction."""
        fc = self.fc
        o, nb, interior = fc.owner, fc.safe_neighbour, fc.interior
        f, s, skew, curvature = self.f, self.s, self.skew, self.curvature
        g, h = u[:, :3], to_tensor(u[:, 3:])
        g_o, g_n, h_o, h_n = g[o], g[nb], h[o], h[nb]
        interp = (1.0 - f) * phi[o] + f * phi[nb]
        g_face = (1.0 - f)[:, None] * g_o + f[:, None] * g_n
        h_face = (1.0 - f)[:, None, None] * h_o + f[:, None, None] * h_n
        if self.local_gradient:
            to_line_own = f[:, None] * s
            to_line_nb = -(1.0 - f)[:, None] * s
            phi_own = (
                interp
                + jnp.sum(skew * (g_o + jnp.einsum("fij,fj->fi", h_o, to_line_own)), axis=-1)
                + 0.5 * jnp.sum(curvature * h_o, axis=(1, 2))
            )
            phi_nbr = (
                interp
                + jnp.sum(skew * (g_n + jnp.einsum("fij,fj->fi", h_n, to_line_nb)), axis=-1)
                + 0.5 * jnp.sum(curvature * h_n, axis=(1, 2))
            )
        else:
            phi_own = phi_nbr = (
                interp
                + jnp.sum(skew * g_face, axis=-1)
                + 0.5 * jnp.sum(curvature * h_face, axis=(1, 2))
            )
        # A boundary face with a prescribed value reads it; one with a prescribed normal derivative
        # gets its value from the owner, with the NORMAL part of the offset carried by the datum
        # rather than by the owner's own gradient: exact for a quadratic, and without the
        # self-coupling that taking `g_P . d` whole puts into the cell's own gradient equation --
        # which nearly cancels its `V g_P` term and leaves the operator near-singular (measured:
        # a quadratic reconstructed to 1e+07, 2170 nonzeros per row). This is the structure the
        # library's zero-gradient closure carries, which offsets by the TANGENTIAL vector alone.
        extrapolated = (
            phi[o]
            + jnp.sum(g_o * self.d_tangent, axis=-1)
            + (bvals - jnp.einsum("fi,fij,fj->f", self.normal, h_o, self.d_own)) * self.along
            + 0.5 * jnp.einsum("fi,fij,fj->f", self.d_own, h_o, self.d_own)
        )
        phi_own = jnp.where(interior, phi_own, jnp.where(self.neumann, extrapolated, bvals))
        moment_own = 0.5 * jnp.einsum("fi,fij,fj->f", self.d_own, h_o, self.d_own)
        moment_nb = 0.5 * jnp.einsum("fi,fij,fj->f", self.d_nb, h_n, self.d_nb)
        r_g = self.to_cells(
            self.area * (phi_own - moment_own)[:, None],
            -self.area * (phi_nbr - moment_nb)[:, None],
        )
        r_g = r_g - self.volume[:, None] * g

        g_ip = g_face + jnp.einsum("fij,fj->fi", h_face, skew)
        normal, d_own = self.normal, self.d_own
        carried = g_o + jnp.einsum("fij,fj->fi", h_o, d_own)
        tangential = carried - jnp.sum(carried * normal, axis=-1)[:, None] * normal
        prescribed = tangential + bvals[:, None] * normal  # the Neumann datum IS n . g at the face
        if self.condition:
            normal_derivative = (
                bvals
                - phi[o]
                - jnp.sum(g_o * self.d_tangent, axis=-1)
                - 0.5 * jnp.einsum("fi,fij,fj->f", d_own, h_o, d_own)
            ) / self.along + jnp.einsum("fi,fij,fj->f", normal, h_o, d_own)
            g_bnd = tangential + normal_derivative[:, None] * normal
        else:
            h_bnd = (1.0 - self.weight) * h + self.weight * self.neighbour_average(h)
            g_bnd = g_o + jnp.einsum("fij,fj->fi", h_bnd[o], self.d_own)
        g_bnd = jnp.where(self.neumann[:, None], prescribed, g_bnd)
        g_ip = jnp.where(interior[:, None], g_ip, g_bnd)
        outer = g_ip[:, :, None] * self.area[:, None, :]
        r_h = self.to_cells(outer, -outer) - self.volume[:, None, None] * h
        return r_g, r_h

    def residual(self, u, phi, bvals):
        """Both equations, the Hessian one reduced to six rows -- shape ``(n, 9)``."""
        r_g, r_h = self.unreduced(u, phi, bvals)
        if self.reducer is None:
            reduced = to_six(r_h)
        else:
            reduced = jnp.einsum("nka,nk->na", self.reducer, r_h.reshape(self.n, 9))
            reduced = reduced / self.volume[:, None]
        return jnp.concatenate([r_g, reduced], axis=-1)

    # -- assembly ---------------------------------------------------------------------------------------
    def _colour(self):
        """No two cells within two face hops share a colour: one evaluation per colour and component
        then recovers every column it seeds, since each residual row reads cells within one hop."""
        n = self.n
        adjacent = [set() for _ in range(n)]
        for a, b in zip(
            self.owner_np[self.interior_np], self.neighbour_np[self.interior_np], strict=True
        ):
            adjacent[a].add(b)
            adjacent[b].add(a)
        self.adjacent = adjacent
        closed = [adjacent[p] | {p} for p in range(n)]
        colour = np.full(n, -1)
        for p in range(n):
            taken = {colour[q] for r in closed[p] for q in closed[r]}
            c = 0
            while c in taken:
                c += 1
            colour[p] = c
        self.colour = colour
        self.n_colours = colour.max() + 1
        self.column_of = np.full((n, self.n_colours), -1)
        for p in range(n):
            for q in closed[p]:
                self.column_of[p, colour[q]] = q

    def _own_hessian_block(self):
        """Each cell's unreduced Hessian rows against its own six Hessian unknowns, ``(n, 9, 6)``."""
        n = self.n
        zero_phi, zero_b = jnp.zeros(n), jnp.zeros(self.fc.n_faces)
        apply = jax.jit(jax.vmap(lambda u: self.unreduced(u, zero_phi, zero_b)[1]))
        block = np.zeros((n, 9, 6))
        for c in range(self.n_colours):
            members = self.colour == c
            seeds = np.zeros((6, n, 9))
            for k in range(6):
                seeds[k, members, 3 + k] = 1.0
            out = np.asarray(apply(jnp.asarray(seeds))).reshape(6, n, 9)
            for k in range(6):
                block[members, :, k] = out[k][members]
        return block

    def assemble(self):
        """``(A, B, B_b)``: ``A u = B phi + B_b phi_b`` over all faces' values ``phi_b``."""
        n, fc = self.n, self.fc
        zero_phi, zero_b = jnp.zeros(n), jnp.zeros(fc.n_faces)
        apply_u = jax.jit(jax.vmap(lambda u: self.residual(u, zero_phi, zero_b)))
        apply_phi = jax.jit(jax.vmap(lambda p: self.residual(jnp.zeros((n, 9)), p, zero_b)))
        apply_b = jax.jit(jax.vmap(lambda b: self.residual(jnp.zeros((n, 9)), zero_phi, b)))
        a_parts, b_parts = [], []
        for c in range(self.n_colours):
            members = self.colour == c
            seeds = np.zeros((9, n, 9))
            for k in range(9):
                seeds[k, members, k] = 1.0
            out = np.asarray(apply_u(jnp.asarray(seeds)))
            q = self.column_of[:, c]
            rows_cells = np.flatnonzero(q >= 0)
            for k in range(9):
                a_parts.append(
                    (
                        (9 * rows_cells[:, None] + np.arange(9)).ravel(),
                        np.repeat(9 * q[rows_cells] + k, 9),
                        out[k][rows_cells].ravel(),
                    )
                )
            out_phi = -np.asarray(apply_phi(jnp.asarray(members[None].astype(float))))[0]
            b_parts.append(
                (
                    (9 * rows_cells[:, None] + np.arange(9)).ravel(),
                    np.repeat(q[rows_cells], 9),
                    out_phi[rows_cells].ravel(),
                )
            )
        # A boundary value reaches only its owner's rows, so the k-th boundary face of every cell
        # can share one seed: at most as many seeds as a cell has boundary faces.
        boundary_faces = np.flatnonzero(~self.interior_np)
        owners = self.owner_np[boundary_faces]
        rank = np.zeros(boundary_faces.size, dtype=int)
        seen: dict[int, int] = {}
        for i, cell in enumerate(owners):
            rank[i] = seen.get(cell, 0)
            seen[cell] = rank[i] + 1
        bb_parts = []
        for r in range(rank.max() + 1):
            chosen = boundary_faces[rank == r]
            seed = np.zeros(fc.n_faces)
            seed[chosen] = 1.0
            out_b = -np.asarray(apply_b(jnp.asarray(seed[None])))[0]
            cells = self.owner_np[chosen]
            bb_parts.append(
                (
                    (9 * cells[:, None] + np.arange(9)).ravel(),
                    np.repeat(chosen, 9),
                    out_b[cells].ravel(),
                )
            )

        def build(parts, shape):
            rows, cols, vals = (np.concatenate(p) for p in zip(*parts, strict=True))
            matrix = sp.csr_matrix((vals, (rows, cols)), shape=shape)
            matrix.eliminate_zeros()
            return matrix

        return (
            build(a_parts, (9 * n, 9 * n)),
            build(b_parts, (9 * n, n)),
            build(bb_parts, (9 * n, fc.n_faces)),
        )


def quadratic_case(prototype):
    """A fixed quadratic: its cell values, its per-face DATA (a value, or the normal derivative on a
    Neumann face) and the exact ``[g, H]`` per cell."""
    hessian = np.array([[1.3, 0.4, -0.2], [0.4, -0.7, 0.3], [-0.2, 0.3, 0.9]])
    slope = np.array([0.5, -1.1, 0.8])
    xc, xf = np.asarray(prototype.x), np.asarray(prototype.x_ip)

    def value(points):
        return 0.5 * np.einsum("...i,ij,...j->...", points, hessian, points) + points @ slope

    exact = np.concatenate(
        [
            xc @ hessian + slope,
            np.tile(np.array([hessian[i, j] for i, j in SYM]), (prototype.n, 1)),
        ],
        axis=1,
    )
    face_gradient = xf @ hessian + slope
    normal_derivative = np.sum(face_gradient * np.asarray(prototype.normal), axis=-1)
    data = np.where(prototype.neumann_np, normal_derivative, value(xf))
    return value(xc), data, exact


class Damping:
    """The Rhie–Chow damping operator of a gradient operator ``G`` of shape ``(n, 3, n)``: unit
    weights, Dirichlet-zero boundary values, as in ``rhie_chow_sign_probe.py``."""

    def __init__(self, prototype):
        p = prototype
        interior = p.interior_np
        s_normal = np.sum(np.asarray(p.s * p.normal), -1)
        weight = np.where(
            interior, np.asarray(p.geometry.face.area) / np.where(interior, s_normal, 1.0), 0.0
        )
        self.o, self.nb, self.w = p.owner_np[interior], p.neighbour_np[interior], weight[interior]
        self.f, self.s = np.asarray(p.f)[interior], np.asarray(p.s)[interior]
        n = self.n = p.n
        compact = np.zeros((n, n))
        np.add.at(compact, (self.o, self.nb), self.w)
        np.add.at(compact, (self.o, self.o), -self.w)
        np.add.at(compact, (self.nb, self.nb), -self.w)
        np.add.at(compact, (self.nb, self.o), self.w)
        self.compact = compact

    def __call__(self, gradient_operator):
        """``(flipped diagonals, largest eigenvalue of the symmetric part, mean nonzeros per row)``."""
        f = self.f[:, None, None]
        face_gradient = (1.0 - f) * gradient_operator[self.o] + f * gradient_operator[self.nb]
        through = np.einsum("fi,fij->fj", self.s, face_gradient) * self.w[:, None]
        operator = self.compact.copy()
        np.add.at(operator, self.o, -through)
        np.add.at(operator, self.nb, through)
        eigenvalues = np.linalg.eigvalsh(0.5 * (operator + operator.T))
        magnitude = np.abs(gradient_operator).sum(axis=1)
        reach = np.count_nonzero(magnitude > 1e-12 * magnitude.max()) / self.n
        return int((np.diag(operator) > 0).sum()), eigenvalues.max(), reach


def main() -> None:
    from aquaflux.io import read_openfoam
    from compare import POLYMESH

    weight = float(os.environ.get("BV_WEIGHT", "0.5"))
    reduction = os.environ.get("BV_REDUCTION", "weighted")
    chosen = os.environ.get("BV_VARIANTS", ",".join(VARIANTS)).split(",")
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    rng = np.random.default_rng(7)
    print(f"reduction {reduction}, weight {weight}", flush=True)

    for name in chosen:
        options = VARIANTS[name]
        print(f"\n=== {name} ===", flush=True)
        proto = BetchenPrototype(mesh, geometry, **options, weight=weight, reduction=reduction)
        n, at_wall = proto.n, proto.at_wall
        cells_q, faces_q, exact = quadratic_case(proto)
        check = proto.residual(jnp.asarray(exact), jnp.asarray(cells_q), jnp.asarray(faces_q))
        print(f"equation residual on an exact quadratic: {float(jnp.abs(check).max()):.2e}")
        a, b, b_b = proto.assemble()
        factor = splu(a.tocsc())
        solved = factor.solve(b @ cells_q + b_b @ faces_q).reshape(n, 9)
        error = np.linalg.norm(solved[:, :3] - exact[:, :3], axis=-1) / np.abs(exact[:, :3]).max()
        print(
            f"converged quadratic error: all {error.max():.2e}, interior {error[~at_wall].max():.2e}",
            flush=True,
        )

        blocks = np.stack([a[9 * p : 9 * p + 9, 9 * p : 9 * p + 9].toarray() for p in range(n)])
        preconditioner = sp.block_diag(np.linalg.inv(blocks), format="csr")
        iteration = LinearOperator(
            (9 * n, 9 * n), matvec=lambda v, pre=preconditioner, a=a: v - pre @ (a @ v), dtype=float
        )
        values, vectors = eigs(iteration, k=4, which="LM", tol=1e-8, maxiter=5000)
        slowest = np.abs(vectors[:, 0]).reshape(n, 9) ** 2
        print(
            f"sweep rate (spectral radius): {np.abs(values).max():.4f}   next: "
            + ", ".join(f"{abs(v):.4f}" for v in sorted(values, key=abs, reverse=True)[1:])
            + f"   slowest mode's share at the wall: {slowest[at_wall].sum() / slowest.sum():.3f}",
            flush=True,
        )

        unit = (np.asarray(proto.x) - np.asarray(proto.x).min(0)) / np.ptp(np.asarray(proto.x), 0)
        fields = {
            "smooth": np.sin(2.0 * unit[:, 0]) * np.cos(1.5 * unit[:, 1]) + unit[:, 2] ** 2,
            "rough": rng.standard_normal(n),
        }
        for label, field in fields.items():
            target = factor.solve(b @ field).reshape(n, 9)[:, :3]
            scale = np.abs(target).max()
            u, line = np.zeros(9 * n), []
            for k in range(1, max(SWEEP_ERRORS) + 1):
                u = u + preconditioner @ (b @ field - a @ u)
                if k in SWEEP_ERRORS:
                    line.append(f"{k}:{np.abs(u.reshape(n, 9)[:, :3] - target).max() / scale:.1e}")
            print(f"  [{label}] error after k sweeps  " + "  ".join(line), flush=True)

        damping = Damping(proto)
        dense_b = b.toarray()
        state = np.zeros((9 * n, n))
        for k in range(1, max(OPERATOR_SWEEPS) + 1):
            state = state + preconditioner @ (dense_b - a @ state)
            if k in OPERATOR_SWEEPS:
                flipped, largest, reach = damping(state.reshape(n, 9, n)[:, :3, :])
                print(
                    f"  {k} sweeps: flipped {flipped:5d}  largest eigenvalue {largest:+.3e}  "
                    f"nnz/row {reach:7.1f}",
                    flush=True,
                )
        del state
        converged = factor.solve(dense_b).reshape(n, 9, n)[:, :3, :]
        del dense_b
        flipped, largest, reach = damping(converged)
        print(
            f"  converged: flipped {flipped:5d}  largest eigenvalue {largest:+.3e}  "
            f"nnz/row {reach:7.1f}",
            flush=True,
        )
        if name == "baseline":
            library = HessianCorrectedGradient(
                boundary_closure=AveragedNeighbourHessian(weight=weight),
                hessian_solve=CoupledBlockSweep(sweeps=300),
            ).bind(mesh, geometry)
            zero_b = jnp.zeros(proto.fc.n_faces)
            for label, field in fields.items():
                ours = np.einsum("pin,n->pi", converged, field)
                theirs = np.asarray(library.gradients(jnp.asarray(field), mesh, geometry, zero_b))
                print(
                    f"  library check ({label}): max |ours - library| / max|g| = "
                    f"{np.abs(ours - theirs).max() / np.abs(theirs).max():.2e}",
                    flush=True,
                )
        del converged, factor


if __name__ == "__main__":
    main()
