"""Gradient reconstruction schemes — reconstruct cell gradients from a cell field.

A ``GradientScheme`` is the swappable numerics object the flow terms consume for their
non-orthogonal corrections (and later Rhie–Chow). It is defined and verified *independently
of any physics*: the exact test is to reconstruct the gradient of a known analytic field and
compare to its analytic gradient (order-of-accuracy study).

:class:`CompactGreenGauss` is the base, one-shot Green–Gauss reconstruction:

    grad(phi)_P = (1 / V_P) * sum_faces  phi_ip * S_f          (S_f = A_f n_f, owner-outward)

with a linearly-interpolated interior face value ``phi_ip = (1-g) phi_P + g phi_N`` (``g`` the
projection factor of the face centroid onto the P–N line) and the supplied boundary value on
boundary faces. It is 2nd-order and linear-exact on orthogonal grids but **inconsistent**
(order ~0) on irregular grids — the known Green–Gauss deficiency. :class:`CorrectedGreenGauss`
adds the non-orthogonal correction (a coupled system): linear-exact on any mesh, consistent
on irregular grids, but capped near 1st order there (the accuracy ceiling the implicit
gradient later removes).
"""

from __future__ import annotations

import abc
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from aquaflux.vectors import dot, scale

from .interpolation import interpolate_owner_neighbour, interpolation_factor

if TYPE_CHECKING:
    from aquaflux.mesh import FaceCellConnectivity, Mesh, MeshGeometry


_GRADIENT_UNCONVERGED_WARNED = False


def _warn_gradient_unconverged(sweeps: int, tol: float) -> None:
    """Host-side diagnostic: warn (once per process) if the fixed-sweep gradient solve is under-resolved.

    Invoked from a ``jax.debug.callback`` inside :meth:`SweptGradientSolve.solve`, gated by a
    ``lax.cond`` so it fires only when the residual (which the sweep already computed) exceeds
    ``tol``. Because that callback runs on every under-resolved gradient solve (many per Newton
    step), a module-level flag guarantees a single emission — the mesh conditioning is fixed, so one
    warning is the whole message.
    """
    global _GRADIENT_UNCONVERGED_WARNED
    if _GRADIENT_UNCONVERGED_WARNED:
        return
    _GRADIENT_UNCONVERGED_WARNED = True
    warnings.warn(
        f"SweptGradientSolve: the corrected-gradient sweeps are under-resolved on this mesh "
        f"(relative residual exceeded {tol:.0e} after {sweeps} sweeps). Increase `sweeps` for this "
        f"non-orthogonality, or set `warn_tol=None` to silence.",
        stacklevel=1,
    )


class _CorrectedTerms(NamedTuple):
    """Geometry-only intermediates shared by the corrected-gradient operator ``A_g`` and RHS ``B``.

    Bundling them lets one face-geometry computation feed both the operator (which is
    field-independent) and the right-hand side (which carries the field), so both linear-solve
    strategies (:class:`GmresGradientSolve`, :class:`SweptGradientSolve`) build on the same system.
    """

    face_cells: FaceCellConnectivity  # face→cell gather/scatter operators (owner / neighbour)
    g: jnp.ndarray  # (n_faces,) projection factor of the face centroid onto the P–N line
    skew: jnp.ndarray  # (n_faces, dim) skewness offset D_g,ip from the P–N line to the face
    area_vector: jnp.ndarray  # (n_faces, dim) owner-outward S_f = A_f n_f
    volume: jnp.ndarray  # (n_cells,) cell volumes


class GradientScheme(eqx.Module):
    """Strategy interface: reconstruct cell gradients from a cell field."""

    @abc.abstractmethod
    def gradients(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
    ) -> jnp.ndarray:
        """Cell gradients of ``field``, shape ``(n_cells, dim)``.

        Parameters
        ----------
        field : jnp.ndarray
            Cell values, shape ``(n_cells,)``.
        mesh : Mesh
            Provides owner/neighbour connectivity.
        geometry : MeshGeometry
            Face and cell metrics (areas, owner-outward normals, centroids, volumes).
        boundary_values : jnp.ndarray
            Face values on boundary faces, shape ``(n_faces,)`` (interior entries ignored).
        """


class CompactGreenGauss(GradientScheme):
    """One-shot Green–Gauss with linearly-interpolated interior face values."""

    def gradients(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
    ) -> jnp.ndarray:
        face_geometry, cell_geometry = geometry.face, geometry.cell
        face_cells = mesh.face_cells
        g = interpolation_factor(face_cells, geometry)
        phi_interior = interpolate_owner_neighbour(field, g, face_cells)
        phi_face = face_cells.combine_face_values(phi_interior, boundary_values)

        area_vector = scale(face_geometry.normal, face_geometry.area)  # owner-outward S_f
        grad_sum = face_cells.scatter_conservative(scale(area_vector, phi_face))
        return scale(grad_sum, 1.0 / cell_geometry.volume)


class GradientSolve(eqx.Module):
    """Strategy: apply ``A_g⁻¹`` to solve the corrected-gradient system ``A_g·G = B·φ``.

    The corrected Green–Gauss reconstruction reduces to a sparse linear system whose operator
    ``A_g`` is geometry-only and volume-dominated (see :class:`CorrectedGreenGauss`). *How* that
    system is inverted — a Krylov solve, a fixed sweep — is orthogonal to the discretization, so it
    is an injected strategy rather than a separate scheme. A concrete strategy receives the shared
    geometry ``terms``, the matrix-free operator ``A_g``, and the right-hand side ``B·φ``.
    """

    @abc.abstractmethod
    def solve(
        self,
        terms: _CorrectedTerms,
        operator: Callable[[jnp.ndarray], jnp.ndarray],
        rhs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve ``A_g·G = rhs`` for the cell gradients ``G``, shape ``(n_cells, dim)``.

        Parameters
        ----------
        terms : _CorrectedTerms
            The geometry-only system intermediates (supplies ``volume`` for a preconditioner).
        operator : callable
            The matrix-free operator ``A_g`` from :meth:`CorrectedGreenGauss.operator`.
        rhs : jnp.ndarray
            The right-hand side ``B·φ``, shape ``(n_cells, dim)``.
        """


class GmresGradientSolve(GradientSolve):
    """Solve the corrected-gradient system with matrix-free GMRES, differentiated by implicit diff.

    Robust to any conditioning — GMRES converges to the requested tolerance regardless of skew —
    and exact to that tolerance. The default strategy: self-tuning where the fixed-sweep count of
    :class:`SweptGradientSolve` would have to be raised for a badly-skewed mesh.

    Attributes
    ----------
    rtol, atol : float
        GMRES relative / absolute tolerances (static).
    """

    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-10)

    def solve(
        self,
        terms: _CorrectedTerms,
        operator: Callable[[jnp.ndarray], jnp.ndarray],
        rhs: jnp.ndarray,
    ) -> jnp.ndarray:
        op = lx.FunctionLinearOperator(operator, jax.ShapeDtypeStruct(rhs.shape, rhs.dtype))
        return lx.linear_solve(op, rhs, solver=lx.GMRES(rtol=self.rtol, atol=self.atol)).value


class SweptGradientSolve(GradientSolve):
    """Solve the corrected-gradient system by a fixed number of matrix-free preconditioned-Richardson
    sweeps — a sparse, ``O(n)``, scalable way to apply the constant ``A_g⁻¹``.

    ``A_g = V ⊙ I − C`` is volume-dominated (``V`` dominates the skewness coupling ``C`` for mild
    non-orthogonality), so the inverse-volume-preconditioned Richardson iteration

        g_{k+1} = g_k + V⁻¹ (B·φ − A_g·g_k)

    converges geometrically with rate ``ρ(I − V⁻¹A_g) < 1``. A **fixed** ``sweeps`` count reaches
    machine precision for this well-conditioned operator with no dense matrix and no nested Krylov
    solve; each sweep is a single operator apply, so the cost is **linear in the mesh** — where a
    dense LU of ``A_g`` would be ``O((n·dim)²)`` per apply and cross over to a loss on finer meshes.
    Differentiated by simply unrolling the short, static-length loop, so the gradient's response to
    ``φ`` is carried implicitly into the flow Jacobian **without** an implicit-diff tangent solve.

    Because ``A_g`` is volume-dominated the iteration converges in very few sweeps — the default
    ``sweeps=4`` reproduces the exact solve to machine precision on mild-to-moderate skew and stays
    well within discretisation error even on aggressively irregular grids. A too-skewed mesh needs
    more; rather than pay for a data-dependent stop (which would defeat the cheap unrolled
    differentiation), the residual the last sweep already computed is checked against ``warn_tol`` and
    a **warning** is emitted once if the sweeps are under-resolved — a diagnostic, not a termination.

    Attributes
    ----------
    sweeps : int
        Number of preconditioned-Richardson sweeps (static).
    warn_tol : float or None
        Emit a one-time warning if the relative gradient residual after ``sweeps`` exceeds this
        (default ``5e-2``, i.e. the sweep is clearly stalling — the converged field stays accurate
        well below this, so it flags only a genuinely under-resolved mesh). ``None`` disables the
        check entirely.
    """

    sweeps: int = eqx.field(static=True, default=4)
    warn_tol: float | None = eqx.field(static=True, default=5e-2)

    def solve(
        self,
        terms: _CorrectedTerms,
        operator: Callable[[jnp.ndarray], jnp.ndarray],
        rhs: jnp.ndarray,
    ) -> jnp.ndarray:
        inv_volume = 1.0 / terms.volume
        grad = jnp.zeros_like(rhs)
        residual = rhs  # rhs - A_g·0; overwritten each sweep with the current residual
        for _ in range(self.sweeps):
            residual = rhs - operator(grad)
            grad = grad + scale(residual, inv_volume)
        if self.warn_tol is not None:
            # `residual` is rhs - A_g·grad from the last sweep (one apply already spent) — a free,
            # slightly conservative convergence indicator. The host-side warning is gated behind a
            # `lax.cond` on the tolerance so the (host-synchronising) callback fires *only* when the
            # sweeps are actually under-resolved; on a converged mesh no callback runs, so the check
            # is free in the common case.
            relative = jnp.linalg.norm(residual) / (jnp.linalg.norm(rhs) + jnp.finfo(rhs.dtype).tiny)
            jax.lax.cond(
                relative > self.warn_tol,
                lambda: jax.debug.callback(
                    _warn_gradient_unconverged, self.sweeps, self.warn_tol, ordered=False
                ),
                lambda: None,
            )
        return grad


class CorrectedGreenGauss(GradientScheme):
    """Green–Gauss with the non-orthogonal skewness correction — a coupled sparse system.

    The corrected face value adds a gradient-based extrapolation from the P–N line to the
    face centroid:

        phi_ip = (1-g) phi_P + g phi_N  +  [(1-g) grad(phi)_P + g grad(phi)_N] . D_g,ip

    where ``g`` is the projection factor of the face centroid onto the P–N line and
    ``D_g,ip = x_ip - x_g`` is the skewness offset. Because the correction depends on the
    *gradients* of the cell and its neighbours, substituting into Green–Gauss gives a
    nearest-neighbour-coupled linear system

        A_g . G = B . phi ,     A_g = V (.) I  -  (the correction coupling)

    with ``A_g`` **geometry-only** and well-conditioned (``V`` dominates for mild skew). *How* the
    system is solved is an injected :class:`GradientSolve` strategy — :class:`GmresGradientSolve`
    (default, exact via ``lineax`` + implicit diff) or :class:`SweptGradientSolve` (fixed sweeps,
    ``O(n)``, scalable); the discretization is identical either way. This is the standalone,
    physics-free form; coupling ``A_g``/``B`` into a flow Newton solve later is the Schur step (same
    ``A_g``/``B``). The correction makes the face value exact for linear fields, so the
    reconstruction is **linear-exact on any mesh** — the fix for :class:`CompactGreenGauss`'s
    inconsistency on irregular grids.

    Attributes
    ----------
    solver : GradientSolve
        The strategy applying ``A_g⁻¹`` to solve ``A_g·G = B·φ`` (default
        :class:`GmresGradientSolve`; use :class:`SweptGradientSolve` for the scalable sweep).
    """

    solver: GradientSolve = eqx.field(default_factory=GmresGradientSolve)

    @staticmethod
    def terms(mesh: Mesh, geometry: MeshGeometry) -> _CorrectedTerms:
        """Geometry-only intermediates of the corrected-gradient system (operator + RHS share them).

        Parameters
        ----------
        mesh : Mesh
            Owner/neighbour connectivity.
        geometry : MeshGeometry
            Face and cell metrics (centroids, owner-outward area vectors, volumes).

        Returns
        -------
        _CorrectedTerms
            The bundled per-face/per-cell geometry the operator and RHS both consume.
        """
        face_geometry, cell_geometry = geometry.face, geometry.cell
        face_cells = mesh.face_cells
        x_p = cell_geometry.centroid[face_cells.owner]
        d = cell_geometry.centroid[face_cells.safe_neighbour] - x_p
        g = interpolation_factor(face_cells, geometry)
        skew = face_geometry.centroid - (x_p + scale(d, g))  # D_g,ip: offset from P–N line to face
        area_vector = scale(face_geometry.normal, face_geometry.area)  # owner-outward S_f
        return _CorrectedTerms(face_cells, g, skew, area_vector, cell_geometry.volume)

    @classmethod
    def operator(cls, t: _CorrectedTerms) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The field-independent, geometry-only linear operator ``A_g`` (a matvec on the gradient).

        ``A_g = V ⊙ I − (correction coupling)``; it depends only on ``t``, never on the field, and is
        volume-dominated — which is exactly what lets :class:`SweptGradientSolve` invert it by a
        few fixed matrix-free sweeps.
        """
        fc = t.face_cells
        owner, nb = fc.owner, fc.safe_neighbour

        def matvec(grad: jnp.ndarray) -> jnp.ndarray:
            w = (1.0 - t.g) * dot(t.skew, grad[owner]) + t.g * dot(t.skew, grad[nb])
            # the correction vanishes on boundary faces (owner side too), so pre-mask before scatter
            correction = fc.scatter_conservative(
                fc.combine_face_values(scale(t.area_vector, w), 0.0)
            )
            return scale(grad, t.volume) - correction

        return matvec

    @classmethod
    def rhs(
        cls, t: _CorrectedTerms, field: jnp.ndarray, boundary_values: jnp.ndarray
    ) -> jnp.ndarray:
        """The right-hand side ``B·φ``: base (interpolated) Green–Gauss with exact boundary values."""
        fc = t.face_cells
        phi_base = fc.combine_face_values(
            interpolate_owner_neighbour(field, t.g, fc), boundary_values
        )
        return fc.scatter_conservative(scale(t.area_vector, phi_base))

    def gradients(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
    ) -> jnp.ndarray:
        t = self.terms(mesh, geometry)
        return self.solver.solve(t, self.operator(t), self.rhs(t, field, boundary_values))


class HessianCorrectedGradient(GradientScheme):
    """Second-order gradient via Betchen's coupled gradient + Hessian reconstruction, with
    the **Hessian Schur-eliminated** so only the gradient is the primary unknown.

    Betchen & Straatman (2010) reconstruct the gradient by a Green–Gauss sum with a
    face-curvature correction, and the Hessian by a Green–Gauss sum of the gradient
    components — a coupled linear system in ``[g, H]`` per cell. The Hessian is needed only
    to lift the gradient to 2nd order; it is not wanted as an output. So the coupled system

        [ A_gg  A_gH ] [ g ]   [ b_g ]
        [ A_Hg  A_HH ] [ H ] = [  0  ]

    is reduced by **Schur elimination of ``H``** to a gradient-only system
    ``S·g = b_g`` with ``S = A_gg − A_gH · A_HH⁻¹ · A_Hg`` (``A_HH`` geometry-only,
    well-conditioned). Every block comes from **AD** — the residual is the forward
    reconstruction (a few interpolations and Green–Gauss sums), never the paper's
    hand-derived coefficient matrices — and the ``A_HH⁻¹`` is applied matrix-free by an
    inner ``lineax`` solve. Set ``schur=False`` to solve the full ``[g, H]`` system instead
    (used to check the two agree).

    Exact for linear *and* quadratic fields on any mesh (the Hessian captures the exact
    second derivative), and 2nd-order for smooth fields — the reconstruction that removes
    :class:`CorrectedGreenGauss`'s ~1st-order cap on irregular grids. The same ``A_g``/``B``
    (with the Hessian pre-eliminated) is what later Schur-couples into the flow Newton solve.
    """

    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=1e-10)
    schur: bool = eqx.field(static=True, default=True)

    def gradients(
        self,
        field: jnp.ndarray,
        mesh: Mesh,
        geometry: MeshGeometry,
        boundary_values: jnp.ndarray,
    ) -> jnp.ndarray:
        dim = mesh.dim
        face_geometry, cell_geometry = geometry.face, geometry.cell
        face_cells = mesh.face_cells
        owner = face_cells.owner
        nb = face_cells.safe_neighbour
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces

        x_own = cell_geometry.centroid[owner]
        x_ip = face_geometry.centroid
        s = cell_geometry.centroid[nb] - x_own
        f = interpolation_factor(face_cells, geometry)
        skew = x_ip - (x_own + scale(s, f))  # D_f,ip
        nhat = face_geometry.normal  # owner-outward unit normal
        area_vector = scale(nhat, face_geometry.area)  # S_f
        d_own = x_ip - x_own  # owner centroid → face centroid
        d_nb = x_ip - cell_geometry.centroid[nb]  # neighbour centroid → face centroid
        vol = cell_geometry.volume

        def _hessian_moment(h, d):
            # ½ dᵀ H d — the Hessian's correction to the mean of φ over the face, so the Green–Gauss
            # face integral is exact for a quadratic. Written from each cell's centroid-to-face
            # vector d (not an explicit face second-moment tensor), so it is dimension-general.
            return 0.5 * jnp.einsum("fi,fij,fj->f", d, h, d)

        def assemble(g, hess, fld, bvals):
            """Green–Gauss RHS for the gradient and Hessian equations (linear in g, hess)."""
            g_o, h_o, h_n = g[owner], hess[owner], hess[nb]  # owner/nb gathers used at boundaries
            g_face = interpolate_owner_neighbour(g, f, face_cells)
            h_face = interpolate_owner_neighbour(hess, f, face_cells)

            # Gradient equation: phi_ip (2nd-order interp) + face-curvature correction.
            q = skew[:, :, None] * skew[:, None, :] - (f * (1.0 - f))[:, None, None] * (
                s[:, :, None] * s[:, None, :]
            )
            phi_int = (
                interpolate_owner_neighbour(fld, f, face_cells)
                + dot(skew, g_face)
                + 0.5 * jnp.sum(q * h_face, axis=(1, 2))
            )
            phi_ip = face_cells.combine_face_values(phi_int, bvals)
            grad_o = scale(area_vector, phi_ip - _hessian_moment(h_o, d_own))
            grad_n = -scale(area_vector, phi_ip - _hessian_moment(h_n, d_nb))
            rhs_g = face_cells.scatter(grad_o, grad_n)

            # Hessian equation: Green–Gauss of the gradient components. Interior faces use the
            # 2nd-order interpolation of grad; boundary faces extrapolate grad from the owner.
            gi_int = g_face + jnp.einsum("fij,fj->fi", h_face, skew)
            gi_bnd = g_o + jnp.einsum("fij,fj->fi", h_o, x_ip - x_own)
            gi = face_cells.combine_face_values(gi_int, gi_bnd)
            hess_contrib = gi[:, :, None] * area_vector[:, None, :]
            rhs_h = face_cells.scatter_conservative(hess_contrib)
            return rhs_g, rhs_h

        zero_g = jnp.zeros((n_cells, dim))
        zero_h = jnp.zeros((n_cells, dim, dim))
        zero_f = jnp.zeros(n_cells)
        zero_b = jnp.zeros(n_faces)
        b_g, b_h = assemble(zero_g, zero_h, field, boundary_values)  # φ-only RHS; b_h == 0
        solver = lx.GMRES(rtol=self.rtol, atol=self.atol)

        if not self.schur:
            struct = (
                jax.ShapeDtypeStruct((n_cells, dim), b_g.dtype),
                jax.ShapeDtypeStruct((n_cells, dim, dim), b_h.dtype),
            )

            def coupled(u):
                g, h = u
                rg, rh = assemble(g, h, zero_f, zero_b)
                return (scale(g, vol) - rg, vol[:, None, None] * h - rh)

            return lx.linear_solve(
                lx.FunctionLinearOperator(coupled, struct), (b_g, b_h), solver=solver
            ).value[0]

        # Schur elimination of the Hessian block: solve S·g = b_g − A_gH·A_HH⁻¹·b_h.
        hess_struct = jax.ShapeDtypeStruct((n_cells, dim, dim), b_h.dtype)

        def a_hg(v):
            return -assemble(v, zero_h, zero_f, zero_b)[1]

        def a_hh(h):
            return vol[:, None, None] * h - assemble(zero_g, h, zero_f, zero_b)[1]

        def a_hh_inv(rhs_h):
            return lx.linear_solve(
                lx.FunctionLinearOperator(a_hh, hess_struct), rhs_h, solver=solver
            ).value

        def a_gh(h):
            return -assemble(zero_g, h, zero_f, zero_b)[0]

        def a_gg(v):
            return scale(v, vol) - assemble(v, zero_h, zero_f, zero_b)[0]

        def schur(v):
            return a_gg(v) - a_gh(a_hh_inv(a_hg(v)))

        rhs = b_g - a_gh(a_hh_inv(b_h))
        return lx.linear_solve(
            lx.FunctionLinearOperator(schur, jax.ShapeDtypeStruct((n_cells, dim), b_g.dtype)),
            rhs,
            solver=solver,
        ).value
