"""The k-omega SST turbulence source terms as volume-source operators.

The k and omega transport equations reuse the advection and diffusion flux operators; what is
particular to turbulence are the volumetric *source* terms, which this module supplies as
:class:`~aquaflux.discretization.source.VolumeSource` operators:

    k:      + P̃_k                        (limited production)
            − β* k ω                     (destruction)
    omega:  + α S²                       (production, α blended by F₁)
            − β ω²                       (destruction, β blended by F₁)
            + 2 (1 − F₁) σ_ω2 ∇k·∇ω / ω  (cross-diffusion)

Each term is written in full and left to automatic differentiation to linearize (no Patankar
implicit/explicit source split); the production limiter is a plain ``min`` in the residual. Each
operator carries the fields held fixed for the evaluation — the eddy viscosity, strain rate, the
other turbulence field, F₁, and the frozen gradients — as constructor state, the same way an
advection operator carries its prescribed mass flux, and reads the shared model constants from an
injected :class:`~aquaflux.turbulence.sst.SSTModel`.

Sign convention follows the volume-source contract: a source returns its volume-integrated value
(production positive), and the residual assembler subtracts it — so a destruction term returns a
negative value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquaflux.discretization import VolumeSource
from aquaflux.vectors import dot

if TYPE_CHECKING:
    from aquaflux.discretization import FaceContext

    from .sst import SSTModel

# The ratio capping production against destruction in the k-production limiter,
# P̃_k = min(P_k, ratio * β* k ω) — a standard SST safeguard against build-up in stagnation regions.
_PRODUCTION_LIMIT_RATIO = 10.0


class KProduction(VolumeSource):
    """Limited production of turbulent kinetic energy, ``P̃_k = min(ν_t S², 10 β* k ω)``.

    The unlimited production ``ν_t S²`` is capped at ``10 β* k ω`` (the destruction scale) to
    prevent unphysical build-up in stagnation regions. The cap depends on the solved ``k``; the eddy
    viscosity, strain rate, and ``ω`` are held fixed.

    Attributes
    ----------
    nu_t : jnp.ndarray
        Eddy viscosity per cell, shape ``(n_cells,)`` (frozen).
    strain_rate : jnp.ndarray
        Strain-rate magnitude ``S`` per cell, shape ``(n_cells,)`` (frozen).
    omega : jnp.ndarray
        Specific dissipation rate per cell, shape ``(n_cells,)`` (frozen).
    model : SSTModel
        The model constants (reads ``beta_star``).
    """

    nu_t: jnp.ndarray
    strain_rate: jnp.ndarray
    omega: jnp.ndarray
    model: SSTModel

    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        production = self.nu_t * self.strain_rate**2
        limit = _PRODUCTION_LIMIT_RATIO * self.model.beta_star * field * self.omega
        return jnp.minimum(production, limit) * context.geometry.cell.volume


class KDestruction(VolumeSource):
    """Destruction of turbulent kinetic energy, ``− β* k ω`` (a sink).

    Linear in the solved ``k`` with ``ω`` frozen, so its linearization is exact.

    Attributes
    ----------
    omega : jnp.ndarray
        Specific dissipation rate per cell, shape ``(n_cells,)`` (frozen).
    model : SSTModel
        The model constants (reads ``beta_star``).
    """

    omega: jnp.ndarray
    model: SSTModel

    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        return -self.model.beta_star * field * self.omega * context.geometry.cell.volume


class OmegaProduction(VolumeSource):
    """Production of the specific dissipation rate, ``α S²`` with ``α`` blended by ``F₁``.

    Independent of the solved ``ω`` (a per-cell source computed from the frozen strain rate and
    blending function).

    Attributes
    ----------
    strain_rate : jnp.ndarray
        Strain-rate magnitude ``S`` per cell, shape ``(n_cells,)`` (frozen).
    f1 : jnp.ndarray
        The ``F₁`` blending function per cell, shape ``(n_cells,)`` (frozen).
    model : SSTModel
        The model constants (blends ``alpha_1`` / ``alpha_2``).
    """

    strain_rate: jnp.ndarray
    f1: jnp.ndarray
    model: SSTModel

    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        alpha = self.model.blend(self.f1, self.model.alpha_1, self.model.alpha_2)
        return alpha * self.strain_rate**2 * context.geometry.cell.volume


class OmegaDestruction(VolumeSource):
    """Destruction of the specific dissipation rate, ``− β ω²`` with ``β`` blended by ``F₁`` (a sink).

    Quadratic in the solved ``ω``; automatic differentiation linearizes it to ``− 2 β ω``.

    Attributes
    ----------
    f1 : jnp.ndarray
        The ``F₁`` blending function per cell, shape ``(n_cells,)`` (frozen).
    model : SSTModel
        The model constants (blends ``beta_1`` / ``beta_2``).
    """

    f1: jnp.ndarray
    model: SSTModel

    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        beta = self.model.blend(self.f1, self.model.beta_1, self.model.beta_2)
        return -beta * field**2 * context.geometry.cell.volume


class OmegaCrossDiffusion(VolumeSource):
    """Cross-diffusion of the specific dissipation rate, ``2 (1 − F₁) σ_ω2 (∇k·∇ω) / ω``.

    The term that reintroduces the k-epsilon cross-term away from the wall. It is treated explicitly
    (a per-cell source from the frozen ``ω`` and frozen gradients), which keeps the ``1/ω`` bounded
    by the realizability floor the driver maintains rather than by the in-solve iterate.

    Attributes
    ----------
    omega : jnp.ndarray
        Specific dissipation rate per cell, shape ``(n_cells,)`` (frozen).
    grad_k, grad_omega : jnp.ndarray
        Cell gradients of ``k`` and ``ω``, shape ``(n_cells, dim)`` (frozen).
    f1 : jnp.ndarray
        The ``F₁`` blending function per cell, shape ``(n_cells,)`` (frozen).
    model : SSTModel
        The model constants (reads ``sigma_omega2``).
    """

    omega: jnp.ndarray
    grad_k: jnp.ndarray
    grad_omega: jnp.ndarray
    f1: jnp.ndarray
    model: SSTModel

    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        cross = (
            2.0
            * (1.0 - self.f1)
            * self.model.sigma_omega2
            * dot(self.grad_k, self.grad_omega)
            / self.omega
        )
        return cross * context.geometry.cell.volume
