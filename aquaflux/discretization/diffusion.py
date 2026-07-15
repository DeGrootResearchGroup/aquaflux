"""The diffusion face-flux operator: a flux-continuous non-orthogonal diffusion flux.

The diffusive flux through a face is ``Gamma (grad phi . n) A``. Its normal derivative is
built to be **flux-continuous across the face**, so that a jump in the diffusion coefficient
(different materials / zones sharing a face) is handled natively. Requiring

    Gamma_P grad phi|_P . n  =  Gamma_N grad phi|_N . n

and extrapolating a one-sided normal derivative from each cell centroid to the face
integration point,

    grad phi|_P . n  =  (phi_ip - phi_P - corr_P) / (D_P . n),   corr_P = grad phi_P . (D_P - (D_P.n) n),

(and likewise from ``N``), then eliminating the common face value ``phi_ip`` gives the
normal derivative in terms of cell-centred quantities:

    grad phi|_ip . n  =  [ (phi_N - phi_P) + corr_N - corr_P ] / denom,
    denom = (D_P . n) - (Gamma_P / Gamma_N)(D_N . n),

where ``D_P = x_ip - x_P`` and ``D_N = x_ip - x_N`` are the owner/neighbour centroid →
face-centroid displacements. The face flux is then ``Gamma_P (grad phi|_ip . n) A``, which —
by construction of ``denom`` from the continuity condition — is the single conservative flux
both cells share (owner ``+``, neighbour ``-``).

The two ``corr`` terms are the non-orthogonal correction: each is the owner/neighbour cell
gradient dotted with the tangential part of its centroid-to-face displacement, and both
vanish on an orthogonal grid, where ``denom -> (x_N - x_P) . n`` and the whole expression
reduces to the harmonic-mean flux of Patankar. Because the correction is written directly
into this residual term (not deferred to an explicit source), automatic differentiation
places it *in the Jacobian* — a consistently linearized non-orthogonal operator with no
hand-derived coefficients.

At a boundary face the neighbour side is replaced by the weak boundary value ``phi_ip``
supplied by a :class:`~aquaflux.boundary.conditions.BoundaryCondition`, giving the
one-sided flux ``Gamma_P (phi_ip - phi_P - corr_P) / (D_P . n) A``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from aquaflux.vectors import dot, scale

from .face_flux import FaceContext, FaceFluxOperator


class DiffusionFlux(FaceFluxOperator):
    """Flux-continuous non-orthogonal diffusion flux.

    The full physical flux (orthogonal part + non-orthogonal correction + coefficient-jump
    handling) is written as one residual term, so its linearization comes entirely from
    automatic differentiation. It consumes the injected per-cell gradient (from a
    :class:`~aquaflux.schemes.GradientScheme`, carried on the context) for the correction; on an
    orthogonal grid the correction is identically zero and the gradient is inert.

    Attributes
    ----------
    coefficient : str
        The name of the material property this operator uses as its diffusion coefficient
        ``Gamma`` (``"diffusivity"`` for a generic scalar, ``"conductivity"`` for heat,
        ``"viscosity"`` for momentum) — read from ``context.materials``. Static.
    """

    coefficient: str = eqx.field(static=True, default="diffusivity")

    def face_flux(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        fc = context.face_cells
        owner, neighbour = fc.owner, fc.safe_neighbour
        fg = context.geometry.face
        n, area, x_ip = fg.normal, fg.area, fg.centroid
        x_cell = context.geometry.cell.centroid

        phi_owner, phi_neighbour = field[owner], field[neighbour]
        grad_owner, grad_neighbour = context.gradient[owner], context.gradient[neighbour]
        gamma = context.materials[self.coefficient]
        gamma_owner, gamma_neighbour = gamma[owner], gamma[neighbour]

        d_p = x_ip - x_cell[owner]
        d_n = x_ip - x_cell[neighbour]
        dpn = dot(d_p, n)  # D_P . n  (> 0)
        dnn = dot(d_n, n)  # D_N . n  (< 0 on interior faces)

        corr_p = dot(grad_owner, d_p - scale(n, dpn))
        corr_n = dot(grad_neighbour, d_n - scale(n, dnn))

        # Interior: two-sided, flux-continuous normal derivative (Gamma-jump in denom).
        denom = dpn - (gamma_owner / gamma_neighbour) * dnn
        denom_safe = jnp.where(fc.interior, denom, 1.0)  # boundary branch unused; keep grad finite
        normal_grad_interior = ((phi_neighbour - phi_owner) + corr_n - corr_p) / denom_safe

        # Boundary: one-sided normal derivative to the weak face value.
        normal_grad_boundary = (context.boundary_values - phi_owner - corr_p) / dpn

        normal_grad = jnp.where(fc.interior, normal_grad_interior, normal_grad_boundary)
        # Owner-outward flux of phi is down-gradient (Fourier): -Gamma (grad phi . n) A.
        return -gamma_owner * normal_grad * area
