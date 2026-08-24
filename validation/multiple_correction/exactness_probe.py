"""Is the multiple-correction reconstruction exact for quadratics on a bad mesh?

The Hessian-corrected (Betchen) reconstruction reaches exactness for quadratic fields by solving
a globally coupled gradient--Hessian system, which on a heavily warped mesh costs twelve to fifteen
sweeps of two face-kernel passes each. Pont et al. (J. Comput. Phys. 350, 2017) reach the same
exactness by a different route: a Green--Gauss sum corrected by per-cell matrices that depend only
on the mesh, applied in a fixed sequence with no system to solve at all.

This decides whether that route is worth building here, before any of it is built. It reproduces the
three operators on small meshes and asks the one question that matters -- does the reconstruction
return a quadratic field's gradient and Hessian to machine precision, on meshes bad enough to matter?
Everything else about the method is a cost argument, and a cost argument is worthless if the accuracy
contract does not hold.

The construction, in the paper's terms and ours
-----------------------------------------------
``R`` is the raw Green--Gauss sum with the distance-weighted face interpolation this package already
uses. It is **not** consistent on a distorted mesh: handed a linear field of gradient ``a`` it returns
``M1 a`` rather than ``a``. The whole method follows from that one observation.

* ``M1`` is recovered by applying ``R`` to the coordinate fields, and ``D1 = M1^-1 R`` is then exact
  for linear fields by construction.
* ``D1`` applied twice gives a Hessian that is inconsistent (an O(1) error), and the same trick
  repairs it: ``M2`` is what ``D1 D1`` returns for each quadratic basis field, so ``M2^-1 D1 D1`` is
  exact for quadratics.
* ``D1`` on a quadratic carries a first-order error, and that error is a fixed linear function of the
  Hessian -- ``H2``, again recovered by probing with the quadratic basis fields. Subtracting it lifts
  the gradient to second order.

**Every correction matrix is obtained by running the operators on coordinate monomials**, which is
the same probe-the-operator pattern the package already uses to recover per-cell diagonal blocks.
There are no bespoke geometric formulas to port and no volume moments to compute.

Scope, deliberately narrow
--------------------------
Boundary faces are given their **exact analytic values**. Pont's own boundary treatment simply drops
the missing neighbour and the paper says of it that "an in-depth analysis of the effect of the
boundary conditions is warranted as future research" -- so boundaries are a known weak point and a
separate question. This probe isolates the interior construction; a failure here cannot be blamed on
a boundary closure, and a success here says nothing about one.

Usage
-----
    python3 validation/multiple_correction/exactness_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "tests"))

import aquaflux  # noqa: E402,F401  (enables x64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from aquaflux.schemes.gradient import (  # noqa: E402
    contract_symmetric,
    expand_symmetric,
    symmetric_components,
)
from aquaflux.schemes.interpolation import interpolation_factor  # noqa: E402

from support.meshes import (  # noqa: E402
    perturbed_grid_2d,
    perturbed_grid_3d,
    tetrahedral_grid_3d,
)


class Operators:
    """The multiple-correction operators for one geometry, built once."""

    def __init__(self, mesh, geometry, *, corrected=True, boundary="exact", blend=0.5):
        self.mesh, self.geometry = mesh, geometry
        self.boundary, self.blend = boundary, blend
        self.dim = mesh.dim
        self.n_cells = mesh.n_cells
        self.n_sym = symmetric_components(self.dim)
        face_cells = mesh.face_cells
        self.owner = np.asarray(face_cells.owner)
        self.neighbour = np.asarray(face_cells.safe_neighbour)
        self.interior = np.asarray(face_cells.interior)
        # Outward-from-owner area vector, and the interpolation weight this package already uses.
        self.area = np.asarray(geometry.face.normal) * np.asarray(geometry.face.area)[:, None]
        self.beta = np.asarray(interpolation_factor(face_cells, geometry))
        self.volume = np.asarray(geometry.cell.volume)
        # Centre the coordinates. `R` annihilates constants, so this cannot change any operator --
        # it only keeps the monomial magnitudes comparable to the cell size.
        # Cells owning no boundary face. The distinction is load-bearing: a boundary closure that
        # loses one order on the boundary layer alone still shows up in a global L2 as a HALF-order
        # loss, because that layer is an O(h) fraction of the volume -- so a global norm cannot tell
        # "the method degraded" from "the closure degraded, on the cells a closure touches".
        touches_boundary = np.zeros(mesh.n_cells, dtype=bool)
        exterior = ~self.interior
        touches_boundary[self.owner[exterior]] = True
        self.interior_cells = ~touches_boundary

        centroid = np.asarray(geometry.cell.centroid)
        self.origin = centroid.mean(axis=0)
        self.x = centroid - self.origin
        self.face_x = np.asarray(geometry.face.centroid) - self.origin

        identity = np.broadcast_to(np.eye(self.dim), (self.n_cells, self.dim, self.dim))
        self.m1_inv = np.linalg.inv(self._probe_m1()) if corrected else identity.copy()
        self.h2, self.m2_inv = self._probe_quadratic()
        if not corrected:
            # The control: the bare Green--Gauss sum applied twice, with no correction at any
            # level. Pont reports this as inconsistent on distorted grids -- O(h) for the
            # gradient and O(1) for the Hessian -- so it MUST fail here. If it does not, the
            # exact boundary values are carrying the result and this probe measures nothing.
            self.h2 = np.zeros_like(self.h2)
            self.m2_inv = np.broadcast_to(
                np.eye(self.n_sym), (self.n_cells, self.n_sym, self.n_sym)
            ).copy()

    def neighbour_average(self, cell_values):
        """Mean of a cell field over each cell's face neighbours."""
        total = np.zeros_like(cell_values)
        count = np.zeros(self.n_cells)
        live = self.interior
        np.add.at(total, self.owner[live], cell_values[self.neighbour[live]])
        np.add.at(total, self.neighbour[live], cell_values[self.owner[live]])
        np.add.at(count, self.owner[live], 1.0)
        np.add.at(count, self.neighbour[live], 1.0)
        shape = (-1,) + (1,) * (cell_values.ndim - 1)
        return total / np.maximum(count, 1.0).reshape(shape)

    def boundary_gradient(self, gradient, exact):
        """Boundary-face values for the GRADIENT, which no boundary condition supplies.

        ``exact`` is analytic and is the reference arm only. ``owner`` takes the owner cell's own
        value -- the zeroth-order closure a solver can always form. ``averaged`` blends that with the
        owner's face-neighbour mean, which is the shape that rescued this scheme's own ``A_HH`` from
        singularity on tetrahedra.
        """
        if self.boundary == "exact":
            return exact
        owner_value = gradient[self.owner]
        if self.boundary == "owner":
            return owner_value
        if self.boundary == "dirichlet":
            # A Dirichlet patch fixes phi over the whole boundary surface, so the face gradient's
            # TANGENTIAL components are known there and only the normal one is not. The owner
            # closure discards that, which is what leaves a boundary tetrahedron short of the
            # independent directions its six Hessian components need. This arm supplies the
            # tangential part exactly and the normal part from the owner, to ask whether the
            # information a real boundary condition carries is enough.
            normal = np.asarray(self.geometry.face.normal)
            owner_normal = (owner_value * normal).sum(-1)[:, None] * normal
            exact_normal = (exact * normal).sum(-1)[:, None] * normal
            return exact - exact_normal + owner_normal
        if self.boundary == "averaged":
            averaged = self.neighbour_average(gradient)[self.owner]
            return (1.0 - self.blend) * owner_value + self.blend * averaged
        raise ValueError(f"unknown boundary closure {self.boundary!r}")

    def raw(self, cell_values, face_values=None):
        """The uncorrected Green--Gauss sum: ``(1/V) sum_f interp(u) A_f``.

        ``cell_values`` is ``(n_cells, ...)``. ``face_values`` supplies the boundary faces (interior
        entries are ignored, being interpolated from the cells); ``None`` means **close the boundary
        with the owner cell's own value**, which is the zeroth-order closure available to a solver
        that has no analytic data -- and the only option for the gradient field, since a boundary
        condition supplies the field there but not its derivative.
        """
        if face_values is None:
            face_values = cell_values[self.owner]
        trailing = cell_values.shape[1:]
        own = cell_values[self.owner]
        nb = cell_values[self.neighbour]
        weight = self.beta.reshape((-1,) + (1,) * len(trailing))
        interpolated = np.where(
            self.interior.reshape((-1,) + (1,) * len(trailing)),
            (1.0 - weight) * own + weight * nb,
            face_values,
        )
        # outer product of the interpolated value with the area vector
        contribution = interpolated[..., None] * self.area.reshape(
            (-1,) + (1,) * len(trailing) + (self.dim,)
        )
        out = np.zeros((self.n_cells, *trailing, self.dim))
        np.add.at(out, self.owner, contribution)
        np.add.at(
            out,
            self.neighbour,
            -np.where(self.interior.reshape((-1,) + (1,) * (len(trailing) + 1)), contribution, 0.0),
        )
        return out / self.volume.reshape((-1,) + (1,) * (len(trailing) + 1))

    def _probe_m1(self):
        """``M1[:, i] = R(x_i)`` -- what the raw operator returns for each coordinate field."""
        columns = [self.raw(self.x[:, i], self.face_x[:, i]) for i in range(self.dim)]
        return np.stack(columns, axis=-1)

    def d1(self, cell_values, face_values):
        """The 1-exact gradient operator ``M1^-1 R``."""
        return np.einsum("nij,nj...->ni...", self.m1_inv, self.raw(cell_values, face_values))

    def _probe_quadratic(self):
        """``H2`` (the gradient's first-order error per unit Hessian) and ``M2``, by probing."""
        h2_cols, m2_cols = [], []
        for basis in _sym_basis(self.dim):
            # psi(x) = 1/2 x . E . x, a global field; its exact gradient is E x.
            psi = 0.5 * np.einsum("ni,ij,nj->n", self.x, basis, self.x)
            psi_face = 0.5 * np.einsum("ni,ij,nj->n", self.face_x, basis, self.face_x)
            grad_exact_cell = self.x @ basis.T
            grad_exact_face = self.face_x @ basis.T

            g1 = self.d1(psi, psi_face)
            h2_cols.append(g1 - grad_exact_cell)  # the O(h) error, per unit Hessian

            # ⚠️ Probe through the SAME closure the operator will run with. Building the
            # correction with exact data and applying it with a real closure corrects an
            # operator nobody evaluates, and the mismatch reads as the closure destroying
            # exactness -- which is what it looked like until this line was checked.
            raw_hessian = self.d1(g1, self.boundary_gradient(g1, grad_exact_face))
            m2_cols.append(_to_symmetric(0.5 * (raw_hessian + np.swapaxes(raw_hessian, 1, 2))))
        h2 = np.stack(h2_cols, axis=-1)  # (n_cells, dim, n_sym)
        m2 = np.stack(m2_cols, axis=-1)  # (n_cells, n_sym, n_sym)
        # A singular M2 is a RESULT, not a crash: it is how a boundary closure fails here, and it
        # is the same failure this scheme's own A_HH shows on tetrahedra under the owner closure.
        # Record it and fall back to a pseudo-inverse so the run still reports the other meshes.
        self.singular = int((np.abs(np.linalg.det(m2)) < 1e-30).sum())
        self.m2_worst_cond = float(np.linalg.cond(m2).max())
        return h2, np.linalg.pinv(m2)

    def reconstruct(self, phi_cell, phi_face, grad_face):
        """Return the 2-exact gradient and the Hessian, in symmetric components."""
        g1 = self.d1(phi_cell, phi_face)
        raw_hessian = self.d1(g1, self.boundary_gradient(g1, grad_face))
        symmetric = _to_symmetric(0.5 * (raw_hessian + np.swapaxes(raw_hessian, 1, 2)))
        hessian = np.einsum("nab,nb->na", self.m2_inv, symmetric)
        gradient = g1 - np.einsum("nia,na->ni", self.h2, hessian)
        return gradient, hessian


def _sym_basis(dim):
    """The symmetric basis tensors in the package's component order."""
    basis = []
    for a in range(symmetric_components(dim)):
        unit = jnp.zeros((1, symmetric_components(dim))).at[0, a].set(1.0)
        basis.append(np.asarray(expand_symmetric(unit, dim))[0])
    return basis


def _to_symmetric(tensor):
    """Contract a symmetric ``(n, dim, dim)`` tensor to its independent components."""
    return np.asarray(contract_symmetric(jnp.asarray(tensor), tensor.shape[-1]))


class SmoothField:
    """A smooth non-polynomial field, with analytic gradient and Hessian.

    Exactness on a quadratic says the operators reproduce the polynomials they are built for; it
    says nothing about the ORDER at which they converge on a general field, because a 2-exact
    operator is machine-zero on a quadratic at every mesh size. Only a non-polynomial field reveals
    the rate, which is the quantity Pont measures (his Figs. 10 and 11) and the one that decides
    whether a fixed-cost reconstruction keeps its accuracy under refinement.
    """

    def __init__(self, dim, wavenumbers=(1.3, 1.7, 0.9)):
        self.k = np.asarray(wavenumbers[:dim])

    def value(self, points):
        return np.prod(np.sin(points * self.k), axis=-1)

    def gradient(self, points):
        sines = np.sin(points * self.k)
        cosines = np.cos(points * self.k)
        out = np.empty_like(points)
        for i in range(points.shape[-1]):
            factors = sines.copy()
            factors[:, i] = cosines[:, i] * self.k[i]
            out[:, i] = np.prod(factors, axis=-1)
        return out

    def hessian(self, points):
        dim = points.shape[-1]
        sines = np.sin(points * self.k)
        cosines = np.cos(points * self.k)
        out = np.empty((points.shape[0], dim, dim))
        for i in range(dim):
            for j in range(dim):
                factors = sines.copy()
                if i == j:
                    factors[:, i] = -sines[:, i] * self.k[i] ** 2
                else:
                    factors[:, i] = cosines[:, i] * self.k[i]
                    factors[:, j] = cosines[:, j] * self.k[j]
                out[:, i, j] = np.prod(factors, axis=-1)
        return out


def order_study(label, build, sizes, boundary):
    """L2 error of the reconstruction on a smooth field, against mesh size.

    Reported over all cells and over the interior alone, because the two answer different
    questions. A closure that loses one order on the boundary-adjacent cells shows up in a global
    L2 as a HALF-order loss -- that layer is an O(h) fraction of the volume, so its contribution
    scales as ``sqrt(h * h^2a)`` -- and a global norm therefore cannot distinguish "the method
    degraded" from "the closure degraded, on exactly the cells a closure touches".
    """
    print(f"\n{label}   (boundary closure: {boundary})")
    print(
        f"  {'cells':>7} {'h':>9} {'grad all':>11} {'ord':>5} {'hess all':>11} {'ord':>5}"
        f" {'grad interior':>14} {'ord':>5} {'hess interior':>14} {'ord':>5}"
    )
    previous = None
    for n in sizes:
        mesh = build(n)
        ops = Operators(mesh, mesh.geometry(), boundary=boundary)
        dim = mesh.dim
        field = SmoothField(dim)

        gradient, packed = ops.reconstruct(
            field.value(ops.x), field.value(ops.face_x), field.gradient(ops.face_x)
        )
        hessian = np.asarray(expand_symmetric(jnp.asarray(packed), dim))

        def weighted_l2(diff, axes, mask, ops=ops):
            weight = ops.volume[mask]
            return float(np.sqrt((weight * (diff[mask] ** 2).sum(axes)).sum() / weight.sum()))

        every = np.ones(mesh.n_cells, dtype=bool)
        inner = ops.interior_cells
        g_diff = gradient - field.gradient(ops.x)
        h_diff = hessian - field.hessian(ops.x)
        current = (
            float(ops.volume.mean() ** (1.0 / dim)),
            weighted_l2(g_diff, -1, every),
            weighted_l2(h_diff, (-1, -2), every),
            weighted_l2(g_diff, -1, inner),
            weighted_l2(h_diff, (-1, -2), inner),
        )
        h, g_all, h_all, g_in, h_in = current
        if previous is None:
            orders = ["--"] * 4
        else:
            ratio = np.log(previous[0] / h)
            orders = [f"{np.log(previous[i] / current[i]) / ratio:.2f}" for i in range(1, 5)]
        print(
            f"  {mesh.n_cells:>7} {h:>9.4f} {g_all:>11.3e} {orders[0]:>5} {h_all:>11.3e}"
            f" {orders[1]:>5} {g_in:>14.3e} {orders[2]:>5} {h_in:>14.3e} {orders[3]:>5}"
        )
        previous = current


def check(label, mesh, seed=0, boundary="exact", blend=0.5):
    geometry = mesh.geometry()
    ops = Operators(mesh, geometry, boundary=boundary, blend=blend)
    control = Operators(mesh, geometry, corrected=False, boundary=boundary, blend=blend)
    rng = np.random.default_rng(seed)
    dim = mesh.dim

    hessian = rng.standard_normal((dim, dim))
    hessian = hessian + hessian.T
    linear = rng.standard_normal(dim)

    def field(points):
        return 0.5 * np.einsum("ni,ij,nj->n", points, hessian, points) + points @ linear

    def gradient_of(points):
        return points @ hessian + linear

    phi_cell = field(ops.x)
    phi_face = field(ops.face_x)
    grad_face = gradient_of(ops.face_x)

    gradient, packed = ops.reconstruct(phi_cell, phi_face, grad_face)
    exact_gradient = gradient_of(ops.x)

    # ⚠️ Compare the Hessian as a TENSOR, not in packed components. `contract_symmetric` is the
    # ADJOINT of the expansion, not its inverse -- an off-diagonal entry receives the sum of both
    # halves -- so a packed-space comparison against a naively contracted reference is wrong by
    # exactly a factor of two on the off-diagonals, and reads as a 0.5 relative error that looks
    # like a broken method rather than a convention mismatch.
    hessian_tensor = np.asarray(expand_symmetric(jnp.asarray(packed), dim))
    exact_tensor = np.broadcast_to(hessian, (mesh.n_cells, dim, dim))

    raw_gradient, raw_packed = control.reconstruct(phi_cell, phi_face, grad_face)
    raw_tensor = np.asarray(expand_symmetric(jnp.asarray(raw_packed), dim))

    def worst(computed, exact):
        return np.abs(computed - exact).max() / np.abs(exact).max()

    m2_cond = np.array([ops.m2_worst_cond])
    m1_cond = np.linalg.cond(ops.m1_inv)
    print(
        f"{label:<26} {mesh.n_cells:>5} "
        f"{worst(gradient, exact_gradient):>9.2e} {worst(hessian_tensor, exact_tensor):>9.2e} "
        f"{worst(raw_gradient, exact_gradient):>9.2e} {worst(raw_tensor, exact_tensor):>9.2e} "
        f"{m1_cond.max():>8.1e} {m2_cond.max():>8.1e} {ops.singular:>5d}"
    )


def main():
    print("multiple-correction reconstruction, exactness on a quadratic field")
    print("(boundary faces given exact values -- this tests the INTERIOR construction only)")
    print("'uncorrected' is the same code with M1, M2 and H2 switched off: it MUST fail, or the")
    print("exact boundary values are carrying the result and this probe measures nothing.\n")
    print(
        f"{'mesh':<26} {'cells':>5} {'grad':>9} {'hess':>9} "
        f"{'grad raw':>9} {'hess raw':>9} {'cond M1':>8} {'cond M2':>8} {'sing':>5}"
    )
    check("2D perturbed 0.25", perturbed_grid_2d(8, 8, perturb=0.25, seed=1))
    check("2D perturbed 0.40", perturbed_grid_2d(8, 8, perturb=0.40, seed=2))
    check("3D perturbed 0.20", perturbed_grid_3d(5, 5, 5, perturb=0.20, seed=3))
    check("3D perturbed 0.35", perturbed_grid_3d(5, 5, 5, perturb=0.35, seed=4))
    check("3D tetrahedra 0.15", tetrahedral_grid_3d(3, perturb=0.15, seed=5))
    check("3D tetrahedra 0.25", tetrahedral_grid_3d(3, perturb=0.25, seed=6))

    # The correction matrices are probed THROUGH the closure, so they should absorb a crude one for
    # the very polynomials they are built from. If exactness survives here, the closure costs only
    # the higher-order terms -- a quite different problem from losing the accuracy contract.
    for boundary, blend in (("owner", 0.0), ("averaged", 0.5), ("dirichlet", 0.0)):
        tag = boundary if boundary == "owner" else f"{boundary} w={blend}"
        print(f"\ngradient boundary faces closed by: {tag}")
        check(
            "2D perturbed 0.40",
            perturbed_grid_2d(8, 8, perturb=0.40, seed=2),
            boundary=boundary,
            blend=blend,
        )
        check(
            "3D perturbed 0.35",
            perturbed_grid_3d(5, 5, 5, perturb=0.35, seed=4),
            boundary=boundary,
            blend=blend,
        )
        check(
            "3D tetrahedra 0.15",
            tetrahedral_grid_3d(3, perturb=0.15, seed=5),
            boundary=boundary,
            blend=blend,
        )
        check(
            "3D tetrahedra 0.25",
            tetrahedral_grid_3d(3, perturb=0.25, seed=6),
            boundary=boundary,
            blend=blend,
        )

    print("\n\nOrder of accuracy on a smooth NON-polynomial field")
    print(
        "(a 2-exact operator is machine-zero on a quadratic at every h, so only this shows the rate)"
    )
    for boundary in ("exact", "owner", "dirichlet"):
        order_study(
            "3D perturbed hex 0.30",
            lambda n: perturbed_grid_3d(n, n, n, perturb=0.30, seed=7),
            (6, 9, 13, 18),
            boundary=boundary,
        )


if __name__ == "__main__":
    main()
