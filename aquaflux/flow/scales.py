"""The characteristic velocity scale of a flow problem, derived from what drives it.

Several parts of the solve need a *representative* flow speed before any flow has been computed: the
convection-aware velocity block freezes its convective linearization at one (so the frozen operator
carries the operating cell Peclet), and the field initializer needs a plug velocity to start a
body-force-driven domain from something other than rest. Both want the same number, so it is derived
once here.

A flow is driven in one of two ways, and the scale is read from whichever applies: a **prescribed
boundary velocity** (an inlet, a moving lid), or a **uniform body force** with no prescribed velocity
anywhere — the streamwise-periodic channel, where the drive is a mean pressure gradient carried as a
force. The second case has no velocity to read, so the speed comes from the global force balance
against the wall drag instead.

Nothing here enters the residual: these are estimates that shape the *path* a solve takes (the frozen
preconditioner, the initial condition), never the converged answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from .momentum import MomentumContinuity

# Bulk-to-friction velocity ratio ``U_b / u_tau`` used to size a body-force-driven flow whose laminar
# estimate is not credible. For wall-bounded turbulent flow the log law gives
# ``U_b / u_tau = (1/kappa) ln(Re_tau) + B``, a slowly-varying number in the high-teens to low-twenties
# across the Reynolds numbers a channel is run at; 20 is representative, and it only sizes an estimate.
BULK_TO_FRICTION_RATIO = 20.0


def hydraulic_length(assembler: MomentumContinuity) -> float:
    """The domain's hydraulic length ``V_total / A_wall`` — the half-height of a plane channel.

    ``A_wall`` is the wetted area of the patches that shear the flow (see
    :meth:`~aquaflux.flow.boundary.FlowBoundary.shears_flow`). Returns ``0.0`` when no patch does, so
    callers can treat "no wall to balance against" as "no scale derivable".

    Parameters
    ----------
    assembler : MomentumContinuity
        The flow assembler; reads its geometry and boundary patches.

    Returns
    -------
    float
        The hydraulic length, or ``0.0`` when the domain has no wetted wall.
    """
    geometry = assembler.geometry
    wall_area = sum(
        float(jnp.sum(geometry.face.area[assembler.mesh.face_patches.indices(name)]))
        for name, closure in assembler.boundary.conditions.items()
        if closure.shears_flow()
    )
    if wall_area <= 0.0:
        return 0.0
    return float(jnp.sum(geometry.cell.volume)) / wall_area


def friction_velocity(assembler: MomentumContinuity) -> jnp.ndarray:
    """Wall friction velocity ``u_tau = sqrt(tau_w / rho)`` of a body-force-driven flow.

    A uniform body force ``beta`` (per unit volume) is balanced at steady state by the drag on the
    solid surfaces, ``beta * V_total = tau_w * A_wall``, so the wall shear stress is ``tau_w = beta * h``
    with ``h`` the hydraulic length (:func:`hydraulic_length`) and

        u_tau = sqrt(beta * h / rho).

    This follows from the force balance alone — no viscous or turbulence assumption — which makes it
    the one velocity scale a body-force-driven wall-bounded flow always has, laminar or turbulent. It
    sets the turbulent branch of :func:`body_force_speed`, and the equilibrium turbulence levels a
    :func:`~aquaflux.turbulence.hybrid_initialize` starts such a domain from.

    Parameters
    ----------
    assembler : MomentumContinuity
        The flow assembler; reads its body force, density, geometry and boundary patches.

    Returns
    -------
    jnp.ndarray
        The scalar friction velocity (zero with no body force, or no wetted wall to balance it).
    """
    h = hydraulic_length(assembler)
    if h <= 0.0:
        return jnp.asarray(0.0)
    beta = jnp.linalg.norm(assembler.body_force)
    return jnp.sqrt(beta * h / jnp.mean(assembler.density))


def body_force_speed(
    assembler: MomentumContinuity, bulk_to_friction: float = BULK_TO_FRICTION_RATIO
) -> jnp.ndarray:
    """Characteristic speed of a flow driven by a uniform body force, from a global force balance.

    A streamwise-periodic channel prescribes velocity nowhere: the flow is driven by the body force
    ``beta`` (per unit volume) and resisted by the drag on the solid surfaces, so at steady state

        beta * V_total = tau_w * A_wall,

    giving the wall shear stress ``tau_w = beta * h`` in terms of the hydraulic length
    ``h = V_total / A_wall`` (:func:`hydraulic_length`). That length is what the boundaries fail to
    supply as a velocity, and it closes the estimate two ways:

    - **Laminar.** Fully-developed flow between plates has bulk velocity ``beta h^2 / (3 mu)``
      (equivalently ``beta H^2 / (12 mu)`` for full height ``H = 2h``) — exact in this regime.
    - **Turbulent.** The friction velocity ``u_tau = sqrt(tau_w / rho) = sqrt(beta h / rho)`` follows
      from the same balance with no viscous assumption, and the bulk velocity is
      ``bulk_to_friction * u_tau``.

    The laminar formula diverges as ``mu`` falls — it is the no-turbulence limit, which stops being
    physical once the flow trips — so the estimate takes the **smaller** of the two: the laminar value
    where it is credible, the friction-velocity scaling once it is not. This is the reverse of the
    prescribed-velocity rule in :func:`characteristic_velocity` (which takes the *fastest*), and
    deliberately so: there the candidates are all genuinely imposed speeds, whereas here the laminar
    branch is an extrapolation that overshoots without bound.

    Parameters
    ----------
    assembler : MomentumContinuity
        The flow assembler; reads its body force, properties, geometry and boundary patches.
    bulk_to_friction : float
        The ratio ``U_b / u_tau`` used by the turbulent branch.

    Returns
    -------
    jnp.ndarray
        The scalar characteristic speed (zero with no body force, or no wetted wall to balance it).
    """
    beta = jnp.linalg.norm(assembler.body_force)
    h = hydraulic_length(assembler)
    if h <= 0.0:
        return jnp.asarray(0.0)
    mu = jnp.mean(assembler.viscosity)
    laminar = beta * h**2 / (3.0 * mu)
    turbulent = bulk_to_friction * friction_velocity(assembler)
    return jnp.minimum(laminar, turbulent)


def body_force_velocity(assembler: MomentumContinuity) -> jnp.ndarray:
    """The uniform plug a body force sustains, shape ``(dim,)`` — :func:`body_force_speed` directed.

    Zero when the domain carries no body force (or has no wetted wall to balance one against), so a
    caller can use a non-zero result as the test for "this domain is body-force-driven".

    Parameters
    ----------
    assembler : MomentumContinuity
        The flow assembler; reads its body force, properties, geometry and boundary patches.

    Returns
    -------
    jnp.ndarray
        The plug velocity vector, shape ``(dim,)``.
    """
    force = assembler.body_force
    magnitude = jnp.linalg.norm(force)
    if float(magnitude) == 0.0:
        return jnp.zeros_like(force)
    return body_force_speed(assembler) * force / magnitude


def characteristic_velocity(assembler: MomentumContinuity) -> jnp.ndarray:
    """The representative flow velocity the domain is driven at, shape ``(dim,)``.

    Read from whichever mechanism drives the flow:

    - **A prescribed velocity.** Each patch declares the velocity it imposes (see
      :meth:`~aquaflux.flow.boundary.FlowBoundary.reference_velocity`), and the **fastest** of them is
      the characteristic speed a Reynolds number is formed from — the inlet speed of a channel, the lid
      speed of a driven cavity. Taking the fastest is the safe side for the frozen preconditioner:
      those hierarchies stay stable as the cell Peclet rises, whereas under-estimating the convection
      drifts the velocity block back toward the Peclet-blind viscous one.
    - **A body force.** A streamwise-periodic channel prescribes velocity nowhere and is driven by a
      uniform body force, so the speed comes from the force balance (:func:`body_force_speed`),
      directed along the force.

    Only a domain driven by *neither* returns zero — correctly, since nothing is making the fluid move.
    A caller that knows the operating speed (a mass-flow controller holds a bulk velocity target)
    should use that instead of this estimate.

    Parameters
    ----------
    assembler : MomentumContinuity
        The flow assembler; reads its boundary closures, body force, properties and geometry.

    Returns
    -------
    jnp.ndarray
        The characteristic velocity vector, shape ``(dim,)``.
    """
    face = assembler.geometry.face
    prescribed = assembler.boundary.apply(
        assembler.mesh.face_cells,
        jnp.zeros((assembler.mesh.n_faces, assembler.mesh.dim)),
        lambda bc, faces, owner: bc.reference_velocity(face.normal[faces], face.centroid[faces]),
    )
    fastest = prescribed[jnp.argmax(jnp.sum(prescribed**2, axis=1))]
    if float(jnp.linalg.norm(fastest)) > 0.0:
        return fastest
    return body_force_velocity(assembler)
