"""Boundary values for the k-omega SST fields.

The k and omega transport equations use the generic scalar boundary closures -- a
:class:`~aquaflux.boundary.conditions.Dirichlet` value at an inlet or wall, a
:class:`~aquaflux.boundary.conditions.ZeroGradient` at an outlet -- so what is turbulence-specific
is only *what value* to impose. These helpers compute those values:

- :func:`omega_wall` -- the adaptive (``y+``-insensitive) near-wall omega, fixed in the wall-adjacent
  cell (via :class:`~aquaflux.discretization.fixed_value.FixedValueCells`), since a resolved viscous
  wall has no finite face value for omega. It blends the viscous-sublayer :func:`omega_wall_value`
  with a log-layer branch so the same wall value holds across ``y+``.
- :func:`nut_wall` -- the adaptive (``y+``-insensitive) wall-face eddy viscosity, the momentum
  companion to :func:`omega_wall`, applied through the momentum diffusion's boundary coefficient so
  the wall shear follows the law of the wall across ``y+``.
- :func:`wall_shear_stress` -- the stress that wall face carries, and :func:`k_wall_production` --
  the log-layer k production it drives in the wall-adjacent cell, replacing the resolved
  ``nu_t S**2`` there as the cell leaves the sublayer.
- :func:`wall_k_diffusivity` -- the wall-face k diffusivity, faded to zero over the same crossover
  so a modelled (rather than resolved) sublayer carries no turbulent-energy flux into the wall.
- :func:`inlet_k` / :func:`inlet_omega` -- the free-stream k and omega from a turbulence intensity
  and a turbulent length scale (a Dirichlet value at the inlet). Wall k is simply zero.

Every crossover is a **smooth blend** on :func:`wall_function_weight`, never a ``y*`` switch: the
residual these closures enter is solved by an automatically-differentiated Newton method, which
cannot converge through a jump (see that function).

All are pure functions of their inputs and the model constants, differentiable in both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from .strain import safe_sqrt

if TYPE_CHECKING:
    from .sst import SSTModel


def omega_wall_value(nu: jnp.ndarray, d: jnp.ndarray, model: SSTModel) -> jnp.ndarray:
    """Near-wall omega ``6 nu / (beta_1 d**2)``, the value fixed in the wall-adjacent cell.

    This is the analytical viscous-sublayer solution evaluated **at the cell centroid**, which is
    where it is imposed. As ``k -> 0`` at a smooth wall the omega equation collapses to a balance of
    viscous diffusion against destruction, ``nu d2(omega)/dy2 = beta_1 omega**2``; substituting
    ``omega = A / y**2`` gives ``6 nu A = beta_1 A**2``, hence ``A = 6 nu / beta_1``. So
    ``omega(y) = 6 nu / (beta_1 y**2)`` solves the sublayer equation exactly (Wilcox).

    The larger ``60 nu / (beta_1 dy**2)`` seen in the literature is a different quantity: a wall
    *face* boundary value, ten times the asymptote, standing in for the singularity at ``y = 0``
    where the analytical solution diverges (Menter, 1994). It is not the value the solution takes at
    a finite distance, so imposing it at a cell centroid overshoots the near-wall omega by 10x --
    which suppresses the near-wall eddy viscosity and stiffens the omega equation.

    This is the **viscous-sublayer branch** of the near-wall omega. On a wall-resolved mesh
    (``y+ ~ 1``, the first cell inside the sublayer) it is the whole story; on a coarser mesh whose
    first cell sits in the log layer it is too low, and the adaptive :func:`omega_wall` blends it
    with a log-layer branch.

    Parameters
    ----------
    nu : jnp.ndarray
        Kinematic (molecular) viscosity at the wall-adjacent cells, shape ``(n_wall,)``.
    d : jnp.ndarray
        Wall distance of those cells (centroid to wall), shape ``(n_wall,)``.
    model : SSTModel
        The model constants (reads ``beta_1``).

    Returns
    -------
    jnp.ndarray
        The near-wall omega per wall-adjacent cell, shape ``(n_wall,)``.
    """
    return 6.0 * nu / (model.beta_1 * d**2)


def omega_wall(nu: jnp.ndarray, d: jnp.ndarray, k: jnp.ndarray, model: SSTModel) -> jnp.ndarray:
    r"""Adaptive (``y+``-insensitive) near-wall omega: ``sqrt(omega_vis**2 + omega_log**2)``.

    The automatic near-wall treatment (Menter, 2003): a smooth blend of the two limiting near-wall
    omega solutions, so the same wall value is correct whether the wall-adjacent cell sits in the
    viscous sublayer or the log layer, with no ``y+`` switch.

    - **Viscous branch** ``omega_vis = 6 nu / (beta_1 d**2)`` (:func:`omega_wall_value`) -- the
      analytical sublayer solution, correct as ``y+ -> 0``.
    - **Log branch** ``omega_log = sqrt(k) / (beta_star**0.25 kappa d)`` -- the equilibrium log-layer
      value, where ``omega = u_tau / (sqrt(beta_star) kappa d)`` and ``u_tau**2 = sqrt(beta_star) k``
      (so ``u_tau = beta_star**0.25 sqrt(k)``).

    The quadrature blend ``sqrt(omega_vis**2 + omega_log**2)`` recovers whichever branch dominates:
    ``omega_vis`` as ``d -> 0`` (it grows like ``1/d**2``, the log branch only ``1/d``), ``omega_log``
    once the first cell is out in the log layer. Because ``omega_vis > 0`` keeps the radicand away
    from zero and the log branch enters as ``k`` (not ``sqrt(k)``), the value is differentiable in
    ``k`` everywhere including ``k = 0`` -- where it reduces exactly to :func:`omega_wall_value`, so
    the wall-resolved limit is unchanged. ``k`` is clamped at zero for the log term (a negative ``k``
    is off-solution and has no log-layer omega); the clamp is inactive at a converged field, so it
    does not pollute the sensitivity.

    Parameters
    ----------
    nu : jnp.ndarray
        Kinematic (molecular) viscosity at the wall-adjacent cells, shape ``(n_wall,)``.
    d : jnp.ndarray
        Wall distance of those cells (centroid to wall), shape ``(n_wall,)``.
    k : jnp.ndarray
        Turbulent kinetic energy at those cells, shape ``(n_wall,)``.
    model : SSTModel
        The model constants (reads ``beta_1``, ``beta_star``, ``kappa``).

    Returns
    -------
    jnp.ndarray
        The adaptive near-wall omega per wall-adjacent cell, shape ``(n_wall,)``.
    """
    omega_vis = omega_wall_value(nu, d, model)
    omega_log_squared = jnp.maximum(k, 0.0) / (jnp.sqrt(model.beta_star) * model.kappa**2 * d**2)
    return jnp.sqrt(omega_vis**2 + omega_log_squared)


def nut_wall(nu: jnp.ndarray, d: jnp.ndarray, k: jnp.ndarray, model: SSTModel) -> jnp.ndarray:
    r"""Adaptive (``y+``-insensitive) wall-face eddy viscosity ``nu_t,wall``.

    The momentum companion to :func:`omega_wall`: the effective **kinematic** eddy viscosity to use
    at a wall face so the momentum wall shear matches the law of the wall, whether the wall-adjacent
    cell sits in the viscous sublayer or the log layer. It is the **k-based** form (the friction
    velocity taken from ``u_tau = beta_star**0.25 sqrt(k)`` rather than from the near-wall velocity),
    which is **velocity-independent** -- so, unlike a velocity-based law, it has no singularity where
    the near-wall velocity vanishes (separation / reattachment), and the momentum wall shear
    ``(mu + rho nu_t,wall) |U|/d`` passes through zero there on its own, the physically correct
    behaviour on a reattaching flow.

    With the k-based wall coordinate ``y* = beta_star**0.25 sqrt(k) d / nu`` (which the equilibrium
    ``u_tau = beta_star**0.25 sqrt(k)`` turns into the usual ``y+ = u_tau d / nu``):

    - **Viscous sublayer** (``y* <= y*_lam``): ``nu_t,wall = 0`` -- the wall is resolved, the momentum
      wall shear is the molecular ``mu |U|/d``, recovering the plain no-slip wall exactly.
    - **Log layer** (``y* > y*_lam``): ``nu_t,wall = nu (y* kappa / ln(E y*) - 1)``, the value that
      makes ``(mu + rho nu_t,wall) |U|/d`` reproduce the log-law wall stress.

    ``y*_lam`` (:attr:`~aquaflux.turbulence.SSTModel.wall_y_star_lam`) is the laminar/log crossover, so
    the branch is off in the sublayer and the ``ln`` is evaluated only where ``E y* > E y*_lam >> 1``
    (its argument floored at that value on the discarded sublayer branch, so the switch is finite and
    differentiable, no ``ln`` singularity). ``k`` is clamped ``>= 0`` (off-solution, inactive at
    convergence). Being the wall-face value it is applied through the momentum diffusion's boundary
    coefficient, not as a cell eddy viscosity -- the interior closure stays ``nu_t = k / omega``.

    Parameters
    ----------
    nu : jnp.ndarray
        Kinematic (molecular) viscosity at the wall-adjacent cells, shape ``(n_wall,)``.
    d : jnp.ndarray
        Wall distance of those cells (centroid to wall), shape ``(n_wall,)``.
    k : jnp.ndarray
        Turbulent kinetic energy at those cells, shape ``(n_wall,)``.
    model : SSTModel
        The model constants (reads ``beta_star``, ``kappa``, ``e_wall``, ``wall_y_star_lam``).

    Returns
    -------
    jnp.ndarray
        The wall-face eddy viscosity ``nu_t,wall`` per wall-adjacent cell, shape ``(n_wall,)``.
    """
    y_star = wall_y_star(nu, d, k, model)
    y_lam = model.wall_y_star_lam
    # Floor the log argument at its crossover value so the discarded sublayer branch stays finite
    # (E y* -> 0 would send ln -> -inf); in the log branch (y* > y_lam) the floor is inactive.
    log_arg = model.e_wall * jnp.maximum(y_star, y_lam)
    nut_over_nu = y_star * model.kappa / jnp.log(log_arg) - 1.0
    return nu * jnp.where(y_star > y_lam, nut_over_nu, 0.0)


def wall_y_star(nu: jnp.ndarray, d: jnp.ndarray, k: jnp.ndarray, model: SSTModel) -> jnp.ndarray:
    """The k-based near-wall coordinate ``y* = beta_star**0.25 sqrt(k) d / nu``.

    The wall coordinate every piece of the adaptive treatment switches on, defined from ``k`` rather
    than from the near-wall velocity (so it is velocity-independent, and has no singularity where the
    wall-parallel velocity vanishes). It is the usual ``y+ = u_tau d / nu`` with the equilibrium
    friction velocity ``u_tau = beta_star**0.25 sqrt(k)``. ``k`` is clamped ``>= 0`` (off-solution).

    Parameters
    ----------
    nu, d, k : jnp.ndarray
        Molecular viscosity, wall distance, and turbulent kinetic energy at the wall-adjacent cells.
    model : SSTModel
        The model constants (reads ``beta_star``).

    Returns
    -------
    jnp.ndarray
        The wall coordinate ``y*``, matching the shape of its inputs.
    """
    # Guarded sqrt: d/dk sqrt(k) -> inf at k = 0 would put a NaN into every Jacobian that reads a
    # wall closure (the k boundary diagonal is derived by differentiating the residual), which is
    # the same cone point the strain-rate magnitude guards.
    return model.beta_star**0.25 * safe_sqrt(jnp.maximum(k, 0.0)) * d / nu


def wall_function_weight(
    nu: jnp.ndarray, d: jnp.ndarray, k: jnp.ndarray, model: SSTModel
) -> jnp.ndarray:
    r"""Smooth log-layer weight ``f = tanh((y*/y*_lam)**4)`` in ``[0, 1)``.

    How far into the log layer a wall-adjacent cell sits: ``f -> 0`` well inside the viscous sublayer,
    ``f -> 1`` well outside it, crossing over around ``y*_lam``. The quartic makes the transition
    sharp without making it abrupt.

    **Why a weight and not a boolean switch (binding).** The near-wall closures do not meet at the
    crossover — the resolved production ``nu_t S**2`` and the log-layer :func:`k_wall_production` differ
    by ~5x there — so a hard ``y* > y*_lam`` switch makes the k residual **discontinuous in k**. A
    segregated, under-relaxed solver tolerates that (it is what OpenFOAM does), but this project solves
    the residual with an automatically-differentiated Newton method, which cannot converge through a
    jump: the iterate simply oscillates across the threshold with no root on either side (measured — the
    k solve fails outright). Blending restores a differentiable residual. This is the same constraint
    that forces the guarded ``sqrt`` on the strain rate and a smooth rather than clipped slope limiter.

    Parameters
    ----------
    nu, d, k : jnp.ndarray
        Molecular viscosity, wall distance, and turbulent kinetic energy at the wall-adjacent cells.
    model : SSTModel
        The model constants (reads ``beta_star``, ``wall_y_star_lam``).

    Returns
    -------
    jnp.ndarray
        The log-layer weight, matching the shape of its inputs.
    """
    return jnp.tanh((wall_y_star(nu, d, k, model) / model.wall_y_star_lam) ** 4)


def log_layer_shear_rate(d: jnp.ndarray, k: jnp.ndarray, model: SSTModel) -> jnp.ndarray:
    r"""Log-layer mean shear ``|dU/dy| = u_tau / (kappa d)``, with ``u_tau = beta_star**0.25 sqrt(k)``.

    The velocity gradient the law of the wall says holds at distance ``d`` from the wall, written in
    the k-based (velocity-independent) friction velocity. It is the mean shear a wall-adjacent cell
    *would* see if the mesh resolved it, and on a wall-function mesh it is the only defensible value:
    the reconstructed cell gradient there averages across a cell spanning most of the log layer, so it
    is neither the local shear nor the wall gradient (measured on a ``y+ ~ 26`` channel: reconstructed
    ``11.5``, log-layer ``2.8``, in ``1/s``).

    Two closures read it, and reading the *same* expression is what keeps them consistent: the
    near-wall k production (:func:`k_wall_production`) uses it as the mean shear the wall stress works
    against, and the strain rate the eddy-viscosity limiter sees in those cells is blended onto it.
    ``k`` is clamped ``>= 0`` under a guarded square root (off-solution; a plain ``sqrt`` would return
    a NaN derivative at ``k = 0``).

    Parameters
    ----------
    d : jnp.ndarray
        Wall distance of the wall-adjacent cells, shape ``(n_wall,)``.
    k : jnp.ndarray
        Turbulent kinetic energy there, shape ``(n_wall,)``.
    model : SSTModel
        The model constants (reads ``beta_star``, ``kappa``).

    Returns
    -------
    jnp.ndarray
        The log-layer mean shear rate, matching the shape of its inputs.
    """
    return model.beta_star**0.25 * safe_sqrt(jnp.maximum(k, 0.0)) / (model.kappa * d)


def wall_shear_stress(
    nu: jnp.ndarray, d: jnp.ndarray, k: jnp.ndarray, shear_rate: jnp.ndarray, model: SSTModel
) -> jnp.ndarray:
    r"""Kinematic wall shear stress ``tau_w / rho = (nu + nu_t,wall) |dU/dn|_wall``.

    The stress the momentum block applies at the wall: its wall-face effective viscosity
    (:func:`nut_wall`, added to the molecular ``nu``) times the wall-face normal velocity gradient
    ``shear_rate``. Written once here because two places need the *same* number -- momentum, through
    the diffusion boundary coefficient, and the k equation, whose near-wall production is this stress
    times the log-layer mean shear (:func:`k_wall_production`). If the k equation used any other
    estimate of the shear (the cell strain-rate magnitude, say), the energy it fed into the wall cell
    would not be the work the wall stress actually does, and the two would settle at different
    friction velocities.

    Substituting :func:`nut_wall` gives the standard hybrid form: with ``u_k = beta_star**0.25 sqrt(k)``
    (the k-based friction velocity) and ``u_log = kappa |dU/dn| d / ln(E y*)`` (the velocity-based one),
    ``tau_w = u_k u_log`` -- their geometric mean, which is a genuine equation for ``k`` and is
    satisfied only where the two agree.

    Parameters
    ----------
    nu : jnp.ndarray
        Molecular viscosity at the wall-adjacent cells, shape ``(n_wall,)``.
    d : jnp.ndarray
        Wall distance of those cells, shape ``(n_wall,)``.
    k : jnp.ndarray
        Turbulent kinetic energy there, shape ``(n_wall,)``.
    shear_rate : jnp.ndarray
        Wall-face normal velocity gradient magnitude ``|U_P - U_wall| / d``, shape ``(n_wall,)``.
    model : SSTModel
        The model constants (through :func:`nut_wall`: ``beta_star``, ``kappa``, ``e_wall``,
        ``wall_y_star_lam``).

    Returns
    -------
    jnp.ndarray
        The kinematic wall shear stress, matching the shape of its inputs.
    """
    return (nu + nut_wall(nu, d, k, model)) * shear_rate


def k_wall_production(
    nu: jnp.ndarray, d: jnp.ndarray, k: jnp.ndarray, shear_rate: jnp.ndarray, model: SSTModel
) -> jnp.ndarray:
    r"""Log-layer k production in a wall-adjacent cell (per unit volume).

    ``P_k = tau_w (beta_star**0.25 sqrt(k)) / (kappa d)`` — the wall shear stress
    (:func:`wall_shear_stress`) times the log-layer mean shear ``u_tau / (kappa d)``, with ``u_tau``
    in its k-based equilibrium form ``beta_star**0.25 sqrt(k)``. That product is the rate the mean
    flow does work against the wall stress across the wall-adjacent cell, which in the log layer is
    exactly what feeds the turbulence.

    **The shear must be the wall-face one, not the cell strain-rate magnitude.** On a wall-function
    mesh the wall-adjacent cell is far too coarse to represent the near-wall profile, so its
    reconstructed strain rate is neither the local log-layer shear nor the wall gradient (measured on a
    ``y+ ~ 26`` channel: cell strain ``11.5``, wall gradient ``17.9``, true log-layer shear ``2.8``, all
    in ``1/s``). Only the wall-face gradient has a defensible meaning here — multiplied by the effective
    wall viscosity it is the stress momentum actually applies — which is why the mean-shear factor is
    taken from the *analytical* log law and the stress from the *discrete* wall flux, rather than both
    from the same badly-resolved gradient. Using the cell strain instead left the production ~19% below
    the destruction at the fixed point, holding the wall-cell ``k`` at ~0.72 of its equilibrium and the
    predicted wall shear ~25% below the wall-resolved answer.

    **Why not the pure-k equilibrium form** ``beta_star**0.75 k**1.5/(kappa d)``: with the wall
    ``omega`` fixed at ``omega_log = sqrt(k)/(beta_star**0.25 kappa d)``, the destruction
    ``beta_star k omega_log`` equals that expression **identically, for every k**. Production and
    destruction would cancel exactly, leaving the wall cell's k equation degenerate — every ``k``
    satisfies it, there is no restoring force, and the scalar solve has nothing to converge to (it
    fails outright; measured). Carrying the actual wall shear instead makes the balance hold only *at*
    equilibrium ``k = u_tau**2/sqrt(beta_star)`` — the solution rather than an identity — so it
    genuinely drives ``k``. It also vanishes smoothly as the wall-parallel velocity does (a separation
    or reattachment point), since it is a product rather than a quotient.

    Blended in by :func:`wall_function_weight`, so a resolved wall keeps the ordinary ``nu_t S**2`` and
    the wall-resolved path is untouched.

    Parameters
    ----------
    nu : jnp.ndarray
        Molecular viscosity at the wall-adjacent cells, shape ``(n_wall,)``.
    d : jnp.ndarray
        Wall distance of those cells, shape ``(n_wall,)``.
    k : jnp.ndarray
        Turbulent kinetic energy there (clamped ``>= 0``; off-solution otherwise).
    shear_rate : jnp.ndarray
        Wall-face normal velocity gradient magnitude ``|U_P - U_wall| / d``, shape ``(n_wall,)``.
    model : SSTModel
        The model constants (reads ``beta_star``, ``kappa``, and, through :func:`nut_wall`,
        ``e_wall`` / ``wall_y_star_lam``).

    Returns
    -------
    jnp.ndarray
        The production per unit volume, matching the shape of its inputs.
    """
    return wall_shear_stress(nu, d, k, shear_rate, model) * log_layer_shear_rate(d, k, model)


def inlet_k(velocity_magnitude: jnp.ndarray, intensity: float) -> jnp.ndarray:
    """Inlet turbulent kinetic energy from a turbulence intensity: ``k = 1.5 (U I)**2``.

    Parameters
    ----------
    velocity_magnitude : jnp.ndarray
        The inlet velocity magnitude ``U``, shape ``(n_inlet,)`` (or a scalar).
    intensity : float
        The turbulence intensity ``I`` (a fraction, e.g. 0.05 for 5%).

    Returns
    -------
    jnp.ndarray
        The inlet ``k``, matching the shape of ``velocity_magnitude``.
    """
    return 1.5 * (velocity_magnitude * intensity) ** 2


def equilibrium_k(friction_velocity: jnp.ndarray, model: SSTModel) -> jnp.ndarray:
    """Turbulent kinetic energy of equilibrium wall turbulence: ``k = u_tau**2 / sqrt(beta_star)``.

    In the log layer production balances dissipation, the shear stress is the wall value
    ``-u'v' = u_tau**2``, and the eddy-viscosity closure ties that to ``k`` through
    ``-u'v' = sqrt(beta_star) k``. Rearranged, ``k = u_tau**2 / sqrt(beta_star)`` (~``3.3 u_tau**2``
    for the standard ``beta_star = 0.09``).

    This is the counterpart to :func:`inlet_k` for a flow with **no inlet to read a level from** — a
    streamwise-periodic channel driven by a body force, where the friction velocity instead follows
    from the global force balance (:func:`~aquaflux.flow.scales.friction_velocity`). Being an
    equilibrium relation it is a level for the core flow, not a wall value; ``k`` still goes to zero at
    the wall through its boundary condition.

    Parameters
    ----------
    friction_velocity : jnp.ndarray
        The wall friction velocity ``u_tau``, any shape (or a scalar).
    model : SSTModel
        The model constants (reads ``beta_star``).

    Returns
    -------
    jnp.ndarray
        The equilibrium ``k``, matching the shape of ``friction_velocity``.
    """
    return friction_velocity**2 / jnp.sqrt(model.beta_star)


def inlet_omega(k: jnp.ndarray, length_scale: float, model: SSTModel) -> jnp.ndarray:
    """Inlet omega from a turbulent length scale: ``omega = sqrt(k) / (beta_star**0.25 L)``.

    Parameters
    ----------
    k : jnp.ndarray
        The inlet turbulent kinetic energy (see :func:`inlet_k`), shape ``(n_inlet,)`` (or scalar).
    length_scale : float
        The turbulent length scale ``L`` (e.g. a fraction of a characteristic dimension).
    model : SSTModel
        The model constants (reads ``beta_star``).

    Returns
    -------
    jnp.ndarray
        The inlet ``omega``, matching the shape of ``k``.
    """
    return jnp.sqrt(k) / (model.beta_star**0.25 * length_scale)


def wall_k_diffusivity(
    gamma: jnp.ndarray, nu: jnp.ndarray, d: jnp.ndarray, k: jnp.ndarray, model: SSTModel
) -> jnp.ndarray:
    r"""Wall-face ``k`` diffusion coefficient, faded out as the cell enters the log layer.

    ``gamma_wall = (1 - f) gamma``, with ``f`` the log-layer weight :func:`wall_function_weight`.

    A wall-resolved mesh carries ``k = 0`` at the wall, and the diffusive flux that Dirichlet drives
    out of the wall-adjacent cell is real physics — the molecular transport of turbulent energy into
    the viscous sublayer, where it dissipates. On a wall-function mesh it is **not**: the sublayer is
    modelled rather than resolved, the wall-adjacent cell sits in the log layer where production
    balances dissipation locally, and the wall carries no turbulent-energy flux at all. Retaining the
    resolved condition there drains the wall cell below its log-layer equilibrium (measured: ~7.5% of
    the local destruction, holding ``k`` a few percent low and the predicted wall shear with it).

    Fading the wall-face **coefficient** rather than moving the face **value** toward zero gradient
    gives the identical flux — ``(1 - f) gamma (0 - k_P)/d`` either way — while keeping the residual's
    ``k``-linearization clean. A ``k``-dependent face value ``f k_P`` instead contributes
    ``d(phi_ip)/d(k_P) = f + k_P f'``, which exceeds one near the crossover: the wall face becomes a
    ``k``-amplifying source and the near-wall cells run away (measured — the solve does not converge).
    The coefficient carries ``f`` at the frozen closure ``k`` instead, exactly as the rest of the
    ``k`` diffusivity ``nu + sigma_k nu_t`` is already frozen per outer sweep, and live (so
    automatically differentiated) in the coupled residual.

    Parameters
    ----------
    gamma : jnp.ndarray
        The unmodified wall-face ``k`` diffusivity ``nu + sigma_k nu_t``, shape ``(n_wall_faces,)``.
    nu : jnp.ndarray
        Molecular viscosity at those faces' owner cells, shape ``(n_wall_faces,)``.
    d : jnp.ndarray
        Wall distance of those owner cells, shape ``(n_wall_faces,)``.
    k : jnp.ndarray
        Turbulent kinetic energy there, shape ``(n_wall_faces,)``.
    model : SSTModel
        The model constants (reads ``beta_star``, ``wall_y_star_lam``).

    Returns
    -------
    jnp.ndarray
        The faded wall-face diffusivity, matching the shape of its inputs.
    """
    return (1.0 - wall_function_weight(nu, d, k, model)) * gamma
