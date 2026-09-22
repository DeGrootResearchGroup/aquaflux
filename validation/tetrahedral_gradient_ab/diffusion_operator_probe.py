"""Issue #435: why does the diffusion operator lose its positive diagonal under multiple correction?

A scalar Laplace operator with a unit coefficient and all-Dirichlet boundaries, so no boundary closure
and no turbulence is involved. Its Jacobian ``J`` is split into the two-point orthogonal part
``J_orth`` (assembled with no gradient scheme) and the non-orthogonal correction ``J_corr = J -
J_orth``, and compared across gradient schemes on three meshes:

* the tetrahedral duct of this case (``of_case``);
* the synthetic tetrahedral unit cube used by the unit tests;
* a perturbed hexahedral grid, as a control.

The multiple-correction scheme is also run with its first pass only (``M1^-1 R``, linear-exact, no
Hessian correction), which separates the two steps it adds to a Green--Gauss sum.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/diffusion_operator_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aquaflux  # noqa: F401  (enables x64)
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions, Dirichlet
from aquaflux.discretization import DiffusionFlux, ResidualAssembler
from aquaflux.io import read_openfoam
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    GradientScheme,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
    interpolation_factor,
    multiple_correction,
)
from aquaflux.vectors import dot, scale
from tests.support.meshes import perturbed_grid_3d, tetrahedral_grid_3d

HERE = Path(__file__).resolve().parent


class FirstPassOnly(GradientScheme):
    """The multiple-correction scheme's linear-exact first pass, ``M1^-1 R``, and nothing after it."""

    inner: MultipleCorrectionGradient

    def _reconstruct_gradient(
        self,
        field,
        mesh,
        geometry,
        boundary_values,
        *,
        operator_hook=None,
        imposed=None,
        boundary_values_at=None,
        boundary_gradient_weight=None,
    ):
        prepared = self.inner.prepared
        face_cells = mesh.face_cells
        m1_inverse = (
            prepared.m1_inverse
            if boundary_gradient_weight is None
            else multiple_correction._boundary_condition_first_pass(
                prepared.m1, boundary_gradient_weight, face_cells, geometry
            )
        )
        return multiple_correction._one_exact(
            field,
            boundary_values,
            m1_inverse,
            interpolation_factor(face_cells, geometry),
            face_cells,
            scale(geometry.face.normal, geometry.face.area),
            geometry,
        )


class FirstPassWhereIllConditioned(GradientScheme):
    """The full multiple-correction gradient, except the first pass on cells whose ``max|M2^-1|`` exceeds
    ``limit`` -- the causal test of whether the amplifying cells are the ill-conditioned Hessian ones."""

    inner: MultipleCorrectionGradient
    limit: float

    def _reconstruct_gradient(
        self,
        field,
        mesh,
        geometry,
        boundary_values,
        *,
        operator_hook=None,
        imposed=None,
        boundary_values_at=None,
        boundary_gradient_weight=None,
    ):
        given = {
            "imposed": imposed,
            "boundary_values_at": boundary_values_at,
            "boundary_gradient_weight": boundary_gradient_weight,
        }
        full = self.inner.gradients(field, mesh, geometry, boundary_values, **given)
        first = FirstPassOnly(self.inner).gradients(field, mesh, geometry, boundary_values, **given)
        worst = jnp.max(jnp.abs(self.inner.prepared.m2_inverse), axis=(1, 2))
        return jnp.where((~(worst <= self.limit))[:, None], first, full)


class FirstPassOnCells(GradientScheme):
    """The full multiple-correction gradient, except the first pass on the named cells."""

    inner: MultipleCorrectionGradient
    cells: jnp.ndarray

    def _reconstruct_gradient(
        self,
        field,
        mesh,
        geometry,
        boundary_values,
        *,
        operator_hook=None,
        imposed=None,
        boundary_values_at=None,
        boundary_gradient_weight=None,
    ):
        given = {
            "imposed": imposed,
            "boundary_values_at": boundary_values_at,
            "boundary_gradient_weight": boundary_gradient_weight,
        }
        full = self.inner.gradients(field, mesh, geometry, boundary_values, **given)
        first = FirstPassOnly(self.inner).gradients(field, mesh, geometry, boundary_values, **given)
        return full.at[self.cells].set(first[self.cells])


class CorrectionCapped(GradientScheme):
    """The full multiple-correction gradient with its second-pass correction smoothly capped.

    ``g = g_first + theta (g_full - g_first)`` with ``theta = 1 / sqrt(1 + |g_full - g_first|^2 /
    (kappa^2 rho^2))``, where ``rho^2`` is the sum over a cell's interior faces of the squared difference
    between the neighbour's first-pass gradient and its own -- the local spread of the field's gradient.
    The correction is therefore about ``kappa rho`` at most once it exceeds that, and untouched when it
    is much smaller. Smooth everywhere, so it keeps the Newton Jacobian well defined. It gives up the
    exactness for quadratics wherever the cap engages.
    """

    inner: MultipleCorrectionGradient
    kappa: float

    def _reconstruct_gradient(
        self,
        field,
        mesh,
        geometry,
        boundary_values,
        *,
        operator_hook=None,
        imposed=None,
        boundary_values_at=None,
        boundary_gradient_weight=None,
    ):
        given = {
            "imposed": imposed,
            "boundary_values_at": boundary_values_at,
            "boundary_gradient_weight": boundary_gradient_weight,
        }
        full = self.inner.gradients(field, mesh, geometry, boundary_values, **given)
        first = FirstPassOnly(self.inner).gradients(field, mesh, geometry, boundary_values, **given)
        face_cells = mesh.face_cells
        jump = first[face_cells.safe_neighbour] - first[face_cells.owner]
        squared = jnp.where(face_cells.interior, jnp.sum(jump * jump, axis=-1), 0.0)
        spread_squared = face_cells.scatter_symmetric(squared)
        correction = full - first
        size_squared = jnp.sum(correction * correction, axis=-1)
        # A floor relative to the field's own gradient size keeps the ratio (and its derivative) finite
        # where the local spread is exactly zero, as on a uniform field.
        floor = 1e-18 * jnp.mean(jnp.sum(first * first, axis=-1)) + 1e-100
        theta = 1.0 / jnp.sqrt(1.0 + size_squared / (self.kappa**2 * spread_squared + floor))
        return first + theta[:, None] * correction


def meshes():
    yield "tet duct (of_case)", read_openfoam(HERE / "of_case" / "constant" / "polyMesh")
    yield "tet unit cube n=4 p=0.25", tetrahedral_grid_3d(4, perturb=0.25, seed=6)
    yield "hex 6^3 p=0.35 (control)", perturbed_grid_3d(6, 6, 6, perturb=0.35, seed=4)


def jacobian(mesh, geometry, scheme):
    boundary = BoundaryConditions({name: Dirichlet(0.0) for name in mesh.face_patches.names})
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": Constant(1.0)}),
        (DiffusionFlux(),),
        boundary,
        gradient_scheme=scheme,
    )
    return np.asarray(jax.jacfwd(assembler.residual)(jnp.zeros(mesh.n_cells)))


def cell_non_orthogonality(mesh, geometry) -> np.ndarray:
    """Largest face angle (degrees) between the centroid connection and the face normal, per cell."""
    face_cells = mesh.face_cells
    x = geometry.cell.centroid
    d = face_cells.neighbour_centroid(x) - x[face_cells.owner]
    cosine = dot(d, geometry.face.normal) / jnp.linalg.norm(d, axis=-1)
    angle = np.degrees(np.arccos(np.clip(np.asarray(cosine), -1.0, 1.0)))
    angle = np.where(np.asarray(face_cells.interior), angle, 0.0)
    worst = np.zeros(mesh.n_cells)
    np.maximum.at(worst, np.asarray(face_cells.owner), angle)
    inner = np.asarray(face_cells.interior)
    np.maximum.at(worst, np.asarray(face_cells.neighbour)[inner], angle[inner])
    return worst


def gradient_self_sensitivity(mesh, geometry, scheme) -> np.ndarray:
    """``|d grad_P / d phi_P|`` per cell under all-zero Dirichlet boundary values, shape ``(n_cells,)``."""
    bound = scheme.bind(mesh, geometry)
    zeros = jnp.zeros(mesh.n_faces)

    def grad(phi):
        return bound.gradients(phi, mesh, geometry, zeros)

    jac = np.asarray(jax.jacfwd(grad)(jnp.zeros(mesh.n_cells)))  # (n, dim, n)
    cells = np.arange(mesh.n_cells)
    return np.linalg.norm(jac[cells, :, cells], axis=-1)


def _rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a))


def explain_amplifiers(mesh, geometry, scheme, corr_over_orth, label) -> None:
    """Which cells carry the largest correction, and what their correction matrices look like."""
    bound = scheme.bind(mesh, geometry)
    prepared = bound.prepared
    m2 = np.max(np.abs(np.asarray(prepared.m2_inverse)), axis=(1, 2))
    defect = np.max(np.abs(np.asarray(prepared.gradient_defect)), axis=(1, 2))
    h = np.cbrt(np.asarray(geometry.cell.volume))
    owner = np.asarray(mesh.face_cells.owner)
    boundary = ~np.asarray(mesh.face_cells.interior)
    owned = np.bincount(owner[boundary], minlength=mesh.n_cells)
    self_sens = gradient_self_sensitivity(mesh, geometry, scheme) * h
    first = gradient_self_sensitivity(mesh, geometry, FirstPassOnly(bound)) * h
    worst = np.argsort(np.abs(corr_over_orth))[::-1][:8]
    print(f"    [{label}] cells with the largest |corr diag| / orth diag:", flush=True)
    print(
        f"      {'cell':>6} {'corr/orth':>10} {'h|dg/dphi|':>11} {'(1st pass)':>11} {'max|M2^-1|':>11} "
        f"{'max|defect|/h':>14} {'bnd faces':>9}",
        flush=True,
    )
    for c in worst:
        print(
            f"      {c:>6} {corr_over_orth[c]:>10.2f} {self_sens[c]:>11.2f} {first[c]:>11.2f} "
            f"{m2[c]:>11.2e} {defect[c] / h[c]:>14.2f} {owned[c]:>9}",
            flush=True,
        )
    rho = np.corrcoef(_rank(np.abs(corr_over_orth)), _rank(m2))[0, 1]
    rho_s = np.corrcoef(_rank(np.abs(corr_over_orth)), _rank(self_sens))[0, 1]
    print(
        f"      rank correlation of |corr/orth| with max|M2^-1|: {rho:.2f}; with h|dg_P/dphi_P|: {rho_s:.2f}; "
        f"mesh median max|M2^-1| {np.median(m2):.2f}, cells > 100: {(m2 > 100).sum()}",
        flush=True,
    )


def main() -> None:
    for label, mesh in meshes():
        geometry = mesh.geometry()
        n = mesh.n_cells
        bound = MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None).bind(
            mesh, geometry
        )
        schemes = {
            "compact GG": CompactGreenGauss(),
            "corrected GG": CorrectedGreenGauss(),
            "multcorr first pass": FirstPassOnly(bound),
            "multcorr owner": MultipleCorrectionGradient(
                boundary_closure=OwnerGradient(), fallback=None
            ),
            "multcorr repaired": MultipleCorrectionGradient(
                boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
            ),
        }
        repaired = MultipleCorrectionGradient(
            boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
        ).bind(mesh, geometry)
        for limit in (1e3, 1e2, 1e1):
            schemes[f"owner, 1st pass M2>{limit:g}"] = FirstPassWhereIllConditioned(bound, limit)
            schemes[f"repaired, 1st M2>{limit:g}"] = FirstPassWhereIllConditioned(repaired, limit)
        orth = jacobian(mesh, geometry, None)
        d_orth = np.diag(orth)
        angle = cell_non_orthogonality(mesh, geometry)
        print(
            f"=== {label}: {n} cells, face non-orthogonality max {angle.max():.1f} deg, "
            f"median cell-max {np.median(angle):.1f} deg ===",
            flush=True,
        )
        print(
            f"  {'scheme':<22} {'diag<=0':>8} {'min diag/orth':>14} {'max |corr diag|/orth':>21} "
            f"{'not diag-dominant':>18} {'cond':>10}",
            flush=True,
        )
        for name, scheme in schemes.items():
            full = jacobian(mesh, geometry, scheme)
            d = np.diag(full)
            corr = d - d_orth
            off = np.abs(full).sum(axis=1) - np.abs(d)
            s = np.linalg.svd(full, compute_uv=False)
            negative = d <= 0
            print(
                f"  {name:<22} {int(negative.sum()):>8} {np.min(d / d_orth):>14.3f} "
                f"{np.max(np.abs(corr) / d_orth):>21.3f} {int((np.abs(d) < off).sum()):>18} "
                f"{s[0] / s[-1]:>10.2e}",
                flush=True,
            )
            if (
                name.startswith("multcorr ")
                and "first" not in name
                and label.startswith("tet duct")
            ):
                explain_amplifiers(mesh, geometry, scheme, corr / d_orth, name)
            if negative.any():
                print(
                    f"      non-positive cells: cell-max non-orthogonality median "
                    f"{np.median(angle[negative]):.1f} deg (mesh median {np.median(angle):.1f}), "
                    f"corr/orth there median {np.median(corr[negative] / d_orth[negative]):.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
