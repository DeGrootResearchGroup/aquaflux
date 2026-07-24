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

The two k terms additionally accept a shared :class:`NearWallKClosure` collaborator: in a
wall-adjacent cell that has left the viscous sublayer, neither the near-wall velocity profile nor the
``omega`` peak is resolved, so both sides of that cell's k budget are modelled instead. It is one
object because the two substitutions only work together (see its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aquaflux.discretization import VolumeSource
from aquaflux.vectors import dot

if TYPE_CHECKING:
    from aquaflux.discretization import FaceContext

# Runtime imports: the adaptive near-wall k treatment is evaluated inside `NearWallKClosure`, so
# these must not be deferred into the type-checking block.
from .boundary import k_wall_production, omega_wall, wall_function_weight
from .sst import SSTModel

# The ratio capping production against destruction in the k-production limiter,
# P̃_k = min(P_k, ratio * β* k ω) — a standard SST safeguard against build-up in stagnation regions.
# The same ratio caps the ω production (see :class:`OmegaProduction`).
_PRODUCTION_LIMIT_RATIO = 10.0

# A tiny positive floor guarding the ``1 / ν_t`` in the ω-production cap where ν_t → 0 (k → 0). It never
# affects a physical field: ν_t is orders above it wherever the cap is active (k / ν_t stays finite as
# both vanish), so it only prevents a 0/0 at an all-zero cold-start cell.
_EDDY_VISCOSITY_FLOOR = 1e-30


class NearWallKClosure(eqx.Module):
    """The wall-adjacent cells' k budget: log-layer production, and the wall's own dissipation rate.

    A wall-adjacent cell that has left the viscous sublayer resolves neither the near-wall velocity
    profile nor the ``omega`` peak, so **both** sides of its k budget are modelled rather than
    computed from the mesh, and they must be modelled *together*:

    - :meth:`production` blends the resolved ``nu_t S**2`` toward the log-layer form
      (:func:`~aquaflux.turbulence.k_wall_production` -- the wall shear stress times the analytical
      log-law mean shear) over the smooth crossover
      :func:`~aquaflux.turbulence.wall_function_weight`.
    - :meth:`dissipation_rate` returns the ``omega`` the destruction ``beta_star k omega`` must read
      there: the analytical wall value :func:`~aquaflux.turbulence.omega_wall` **as a live function of
      the solved k**, since that is exactly the value the ``omega`` equation fixes in these cells.

    **Both, or neither.** Out in the log layer the modelled production grows very nearly *linearly*
    in ``k`` (the wall eddy viscosity scales like ``sqrt(k)`` and so does the friction velocity), so
    against a **frozen** ``omega`` the destruction ``beta_star k omega`` is linear in ``k`` too: the
    wall row degenerates into a homogeneous equation whose diagonal changes sign the moment production
    exceeds destruction, and the k solve runs away (measured -- it fails outright). Reading the wall
    ``omega`` live restores the physical ``k**1.5`` destruction, so the balance closes at the
    equilibrium ``k`` and the diagonal stays positive. On a wall-resolved mesh this is a no-op in all
    but name: the fixed ``omega`` there is the viscous ``6 nu/(beta_1 d**2)``, independent of ``k``.

    The wall-adjacent cell indices and the three per-cell quantities the closures read always travel
    together, so they are carried as one collaborator rather than threaded as loose arrays.

    Attributes
    ----------
    cells : jnp.ndarray
        Indices of the wall-adjacent cells, shape ``(n_wall,)``.
    distance : jnp.ndarray
        Wall distance of those cells, shape ``(n_wall,)``.
    viscosity : jnp.ndarray
        Molecular (kinematic) viscosity there, shape ``(n_wall,)``.
    shear_rate : jnp.ndarray
        Wall-face normal velocity gradient magnitude ``|U_P - U_wall| / d`` there, shape
        ``(n_wall,)`` — frozen for the sweep in the segregated path, live in the coupled residual.
    model : SSTModel
        The model constants (reads ``beta_star``, ``beta_1``, ``kappa``, ``e_wall``,
        ``wall_y_star_lam``).
    """

    cells: jnp.ndarray
    distance: jnp.ndarray
    viscosity: jnp.ndarray
    shear_rate: jnp.ndarray
    model: SSTModel

    def production(self, resolved: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
        """``resolved`` with the wall-adjacent rows blended toward the log-layer production.

        Parameters
        ----------
        resolved : jnp.ndarray
            The resolved (limited) production per cell, shape ``(n_cells,)``.
        k : jnp.ndarray
            The **solved** turbulent kinetic energy per cell, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            The production per cell with the near-wall blend applied, shape ``(n_cells,)``.
        """
        # The wall production reads the LIVE `k`, never a frozen copy. Freezing is right for the
        # production *cap* (whose exact linearization is a large positive diagonal that breaks the
        # M-matrix) but wrong here: a frozen source turns the k update into the Picard map
        # k <- P(k)/(beta* omega), whose slope is P'(k)/(beta* omega). Where the production's
        # k-sensitivity exceeds the destruction's, that slope passes 1 and the iteration *diverges*
        # (measured). Linearizing it exactly keeps Newton on a genuine root instead.
        k_wall = k[self.cells]
        log_layer = wall_function_weight(self.viscosity, self.distance, k_wall, self.model)
        wall_value = k_wall_production(
            self.viscosity, self.distance, k_wall, self.shear_rate, self.model
        )
        # Blend, do not switch: the two branches differ by several-fold at the crossover, and a jump
        # would make the residual non-differentiable (see `wall_function_weight`).
        return resolved.at[self.cells].set(
            (1.0 - log_layer) * resolved[self.cells] + log_layer * wall_value
        )

    def dissipation_rate(self, omega: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
        """``omega`` with the wall-adjacent rows replaced by the analytical wall value at ``k``.

        No blend and no crossover: the ``omega`` equation replaces these cells' balance by the
        fixation :func:`~aquaflux.turbulence.omega_wall` at *every* ``y+``, so this is simply the same
        value read at the solved ``k`` instead of the previous sweep's copy.

        Parameters
        ----------
        omega : jnp.ndarray
            The frozen specific dissipation rate per cell, shape ``(n_cells,)``.
        k : jnp.ndarray
            The **solved** turbulent kinetic energy per cell, shape ``(n_cells,)``.

        Returns
        -------
        jnp.ndarray
            The dissipation rate per cell with the wall fixation applied, shape ``(n_cells,)``.
        """
        return omega.at[self.cells].set(
            omega_wall(self.viscosity, self.distance, k[self.cells], self.model)
        )


class KProduction(VolumeSource):
    """Limited production of turbulent kinetic energy, ``P̃_k = min(ν_t S², 10 β* k ω)``.

    The unlimited production ``ν_t S²`` is capped at ``10 β* k ω`` (the destruction scale) to
    prevent unphysical build-up in stagnation regions. The cap depends on the solved ``k``; the eddy
    viscosity, strain rate, and ``ω`` are held fixed.

    Where the cap is active it makes production an *increasing* function of ``k``, whose exact
    linearization is a negative diagonal contribution (production feeds ``k`` back into itself) that
    can destroy the diagonal dominance of the k-equation's Jacobian and stall its linear solve at
    high Reynolds number. ``explicit_limiter`` freezes the cap's ``k`` (a Patankar / deferred-
    correction treatment: production is evaluated with the current ``k`` but not differentiated
    through it), keeping the residual value exact while removing that destabilizing term from the
    Jacobian, so the forward solve sees an M-matrix. It is a **forward-solve device only**: the
    default (``False``) is the exact operator, which the coupled sensitivity residual uses so the
    adjoint stays exact.

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
    explicit_limiter : bool
        Freeze the cap's ``k`` for the linearization (a forward-solve stabilization); static.
    near_wall : NearWallKClosure or None
        The adaptive near-wall treatment, blending the resolved production toward the log-layer wall
        form in the wall-adjacent cells. ``None`` (default) keeps the resolved production everywhere.
    """

    nu_t: jnp.ndarray
    strain_rate: jnp.ndarray
    omega: jnp.ndarray
    model: SSTModel
    explicit_limiter: bool = eqx.field(static=True, default=False)
    near_wall: NearWallKClosure | None = None

    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        production = self.nu_t * self.strain_rate**2
        cap_field = jax.lax.stop_gradient(field) if self.explicit_limiter else field
        limit = _PRODUCTION_LIMIT_RATIO * self.model.beta_star * cap_field * self.omega
        limited = jnp.minimum(production, limit)
        if self.near_wall is not None:
            limited = self.near_wall.production(limited, field)
        return limited * context.geometry.cell.volume


class KDestruction(VolumeSource):
    """Destruction of turbulent kinetic energy, ``− β* k ω`` (a sink).

    Linear in the solved ``k`` with ``ω`` frozen, so its linearization is exact — except in the
    wall-adjacent cells, where ``near_wall`` substitutes the analytical wall ``ω`` evaluated at the
    solved ``k`` (making the term ``k**1.5`` there, and its linearization still exact under automatic
    differentiation).

    Attributes
    ----------
    omega : jnp.ndarray
        Specific dissipation rate per cell, shape ``(n_cells,)`` (frozen).
    model : SSTModel
        The model constants (reads ``beta_star``).
    near_wall : NearWallKClosure or None
        The adaptive near-wall treatment, supplying the wall cells' own dissipation rate. Must be
        given whenever :class:`KProduction` is given one — see the "both, or neither" note there.
        ``None`` (default) uses the frozen ``omega`` everywhere.
    """

    omega: jnp.ndarray
    model: SSTModel
    near_wall: NearWallKClosure | None = None

    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        omega = self.omega
        if self.near_wall is not None:
            omega = self.near_wall.dissipation_rate(omega, field)
        return -self.model.beta_star * field * omega * context.geometry.cell.volume


class OmegaProduction(VolumeSource):
    """Limited production of the specific dissipation rate, ``α P̃_k / ν_t`` with ``α`` blended by ``F₁``.

    The ω production is the k production scaled by ``α / ν_t``; writing it that way and reusing the
    **limited** k production ``P̃_k = min(ν_t S², 10 β* k ω)`` (the same destruction-scale cap
    :class:`KProduction` applies) gives ``α min(S², 10 β* k ω / ν_t)`` — the standard SST ω-production
    limiter. Without it the ω source is the unlimited ``α S²``, which over-stiffens the ω equation
    where the strain is large (stagnation regions, and the transient over-shoots a segregated march can
    pass through).

    Independent of the *solved* ``ω`` (a per-cell source from the frozen closure fields), so it adds no
    diagonal term to the ω-equation Jacobian; in the coupled residual it differentiates exactly through
    the live closure.

    Attributes
    ----------
    strain_rate : jnp.ndarray
        Strain-rate magnitude ``S`` per cell, shape ``(n_cells,)`` (frozen).
    nu_t : jnp.ndarray
        Eddy viscosity per cell, shape ``(n_cells,)`` (frozen) — sets the ``α / ν_t`` scaling and the cap.
    k, omega : jnp.ndarray
        Turbulent kinetic energy and specific dissipation rate per cell, shape ``(n_cells,)`` (frozen)
        — the destruction-scale cap ``10 β* k ω``.
    f1 : jnp.ndarray
        The ``F₁`` blending function per cell, shape ``(n_cells,)`` (frozen).
    model : SSTModel
        The model constants (blends ``alpha_1`` / ``alpha_2``, reads ``beta_star``).
    """

    strain_rate: jnp.ndarray
    nu_t: jnp.ndarray
    k: jnp.ndarray
    omega: jnp.ndarray
    f1: jnp.ndarray
    model: SSTModel

    def source(self, field: jnp.ndarray, context: FaceContext) -> jnp.ndarray:
        alpha = self.model.blend(self.f1, self.model.alpha_1, self.model.alpha_2)
        # α min(S², 10 β* k ω / ν_t): the cap is α/ν_t times the destruction-scale k-production cap.
        # The ν_t floor only guards the k → 0 (ν_t → 0) edge; k/ν_t stays finite there (both vanish),
        # so where the cap actually bites it is unaffected.
        cap = (
            _PRODUCTION_LIMIT_RATIO
            * self.model.beta_star
            * self.k
            * self.omega
            / jnp.maximum(self.nu_t, _EDDY_VISCOSITY_FLOOR)
        )
        return alpha * jnp.minimum(self.strain_rate**2, cap) * context.geometry.cell.volume


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
