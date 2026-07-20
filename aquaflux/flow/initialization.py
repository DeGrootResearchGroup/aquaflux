"""Cheap field initializers -- a scalar Laplace solve and a potential-flow velocity.

A good initial condition is what lets the monolithic coupled Newton solve (and, less critically, the
segregated loop) start from nothing. The two building blocks here are both **single linear SPD
solves**, multigrid-preconditioned so they stay robust on the high-aspect-ratio cells of a
wall-resolved mesh, so they cost a fraction of one nonlinear iteration:

- :func:`laplace_field` solves ``div(Gamma grad phi) = 0`` for the boundary-value harmonic field --
  the smooth interpolant of a scalar's boundary data into the interior.
- :func:`potential_flow` uses it to build an irrotational velocity ``u = grad phi`` whose normal
  component matches the flow boundary conditions (inflow at inlets, no penetration at walls), i.e. the
  classic Fluent-style "hybrid" velocity initializer. It is divergence-free and respects the geometry,
  unlike a uniform plug guess -- and, being a real discrete gradient field, it carries the tiny
  asymmetry that lifts the coupled solve's degeneracy on a perfectly symmetric velocity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aquaflux.boundary import BoundaryConditions, Dirichlet, Neumann, ZeroGradient
from aquaflux.discretization import DiffusionFlux, FixedValueCells, ResidualAssembler
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import (
    build_smoothed_hierarchy,
    convection_diffusion_operator,
    decouple_dof,
    newton_step,
    smoothed_multigrid_solve,
)
from aquaflux.vectors import dot

from .boundary import PressureOutlet, VelocityInlet
from .scales import body_force_velocity

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.schemes import GradientScheme

    from .momentum import MomentumContinuity


def _laplace_preconditioner(
    mesh: Mesh,
    geometry: MeshGeometry,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    diffusivity: float,
    fixed_cells: jnp.ndarray | None,
) -> Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """A frozen smoothed-aggregation V-cycle preconditioner for the Laplace operator.

    The Laplacian of a graded, wall-resolved mesh is severely ill-conditioned: its condition number
    grows like the square of the near-wall cell aspect ratio, so an unpreconditioned Krylov solve
    stagnates and returns a non-finite iterate once the aspect ratio reaches a few hundred -- exactly
    the meshes a wall-resolved turbulent case needs. Diagonal scaling is not enough (it removes only
    one factor of the aspect ratio); the smooth error modes need a coarse space.

    The operator is the symmetric graph Laplacian of the diffusion face coefficients
    ``Gamma_face A / (d . n)``, plus the per-cell boundary stiffness the interior faces do not carry.
    That boundary diagonal comes from a single Jacobian-vector product ``J . 1``: a conservative
    pure-diffusion interior stencil has zero row sums, so whatever ``J . 1`` leaves on a row is the
    Dirichlet boundary-face stiffness (a ``ZeroGradient`` or ``Neumann`` patch contributes nothing,
    which is what makes an all-Neumann Laplacian singular and why a datum is required).
    """
    face_cells = mesh.face_cells
    owner_e, nb_e, interior_faces = face_cells.interior_edges()
    n = mesh.n_cells

    d = (
        face_cells.neighbour_centroid(geometry.cell.centroid)
        - geometry.cell.centroid[face_cells.owner]
    )
    coefficient = diffusivity * geometry.face.area / dot(d, geometry.face.normal)
    coefficient_int = np.asarray(jax.lax.stop_gradient(coefficient))[interior_faces]

    zero = jnp.zeros(n)
    _, j_dot_one = jax.jvp(residual_fn, (zero,), (jnp.ones_like(zero),))
    # Clamp non-negative to keep an M-matrix: a fixated row contributes its own unit diagonal (it is
    # decoupled below), and any negative entry would make the V-cycle diverge.
    boundary_diagonal = np.maximum(np.asarray(jax.lax.stop_gradient(j_dot_one)), 0.0)

    a = convection_diffusion_operator(
        owner_e, nb_e, coefficient_int, n, boundary_diagonal=boundary_diagonal
    )
    for cell in np.atleast_1d(np.asarray(fixed_cells)) if fixed_cells is not None else ():
        a = decouple_dof(a, int(cell))
    hierarchy = build_smoothed_hierarchy(a)

    def factory(_: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
        return lambda residual: smoothed_multigrid_solve(hierarchy, residual)

    return factory


def laplace_field(
    mesh: Mesh,
    geometry: MeshGeometry,
    boundary: BoundaryConditions,
    *,
    gradient_scheme: GradientScheme | None = None,
    fixed_cells: jnp.ndarray | None = None,
    fixed_values: jnp.ndarray | None = None,
    diffusivity: float = 1.0,
) -> tuple[jnp.ndarray, ResidualAssembler]:
    """Solve the scalar Laplace equation ``div(Gamma grad phi) = 0`` for the given boundary data.

    A pure-diffusion residual is linear, so a single Newton step is exact. The step is preconditioned
    by a smoothed-aggregation V-cycle (:func:`_laplace_preconditioner`), without which the solve
    stagnates to a non-finite iterate on the high-aspect-ratio cells of a wall-resolved mesh. Returns
    the solved field **and** the assembler (so a caller can reconstruct ``grad phi`` with the same
    boundary closures).

    Parameters
    ----------
    mesh, geometry : Mesh, MeshGeometry
        The mesh and its geometry.
    boundary : BoundaryConditions
        The scalar boundary conditions for ``phi`` (must pin at least one value -- a ``Dirichlet`` patch
        or ``fixed_cells`` -- or the all-Neumann Laplacian is singular).
    gradient_scheme : GradientScheme or None
        Injected so the returned assembler can reconstruct the cell gradient; ``None`` disables the
        non-orthogonal correction in the operator (fine for the initializer).
    fixed_cells, fixed_values : jnp.ndarray or None
        Optional cell-value fixation (e.g. a datum cell when there is no Dirichlet patch, or near-wall
        values), applied as a :class:`~aquaflux.discretization.FixedValueCells` row replacement.
    diffusivity : float
        The (constant) ``Gamma``; irrelevant to the harmonic solution but sets the operator scale.

    Returns
    -------
    tuple of (jnp.ndarray, ResidualAssembler)
        The harmonic field ``(n_cells,)`` and the assembler used to build it.
    """
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({"diffusivity": Constant(diffusivity)}),
        (DiffusionFlux(),),
        boundary,
        gradient_scheme=gradient_scheme,
    )
    residual = assembler.residual
    if fixed_cells is not None:
        fixation = FixedValueCells(fixed_cells, fixed_values)

        def residual(phi: jnp.ndarray) -> jnp.ndarray:
            return fixation.apply(assembler.residual(phi), phi)

    preconditioner = _laplace_preconditioner(mesh, geometry, residual, diffusivity, fixed_cells)
    # Pure diffusion is linear, so one Newton correction is the exact solve. Compiled here because
    # newton_step leaves the jit boundary to its caller; un-jitted, the preconditioned linear solve
    # would dispatch operation by operation.
    field = eqx.filter_jit(newton_step)(
        residual, jnp.zeros(mesh.n_cells), preconditioner=preconditioner
    )
    return field, assembler


def potential_flow(
    momentum: MomentumContinuity, *, gradient_scheme: GradientScheme | None = None
) -> jnp.ndarray:
    """Potential-flow velocity initializer: ``u = grad phi`` with the flow's normal-velocity BCs.

    Solves ``div(grad phi) = 0`` with a boundary condition per patch derived from the flow closures --
    a ``Neumann`` prescribing ``d(phi)/dn = u_in . n`` at a :class:`~aquaflux.flow.VelocityInlet`, a
    ``Dirichlet`` reference at a :class:`~aquaflux.flow.PressureOutlet`, and no penetration
    (``ZeroGradient``) at every wall -- then returns the flat flow state ``[grad phi, p=0]``. The result
    is irrotational, divergence-free, and matches the through-flow, so it is a far better start than a
    uniform plug for anything but a straight duct, at the cost of one linear solve.

    A domain with no through-flow boundary at all has no potential to solve for, but may still be
    driven: a streamwise-periodic channel is pushed by a uniform ``body_force``, whose potential is
    uniform. That case returns a plug at the characteristic speed the force sustains against the wall
    drag (:func:`~aquaflux.flow.scales.body_force_velocity`) — far closer to the developed flow than
    rest, which leaves a globalized solve to march the entire viscous spin-up. A domain driven only by
    a moving wall is not this case (it has no net through-flow, so its potential really is zero), and
    a domain driven by nothing returns the zero state. When there is no outlet but the
    momentum carries a ``pressure_pin``, that cell pins the otherwise-singular Laplacian.

    Parameters
    ----------
    momentum : MomentumContinuity
        The flow assembler; its boundary closures and mesh drive the potential solve.
    gradient_scheme : GradientScheme or None
        The scheme reconstructing ``grad phi`` (defaults to :class:`~aquaflux.schemes.CompactGreenGauss`).

    Returns
    -------
    jnp.ndarray
        The flat flow state ``[vel..., pressure]``, shape ``((dim + 1) n_cells,)``.
    """
    mesh, geometry = momentum.mesh, momentum.geometry
    gradient_scheme = gradient_scheme or CompactGreenGauss()

    conditions: dict[str, object] = {}
    has_reference = False
    for name, closure in momentum.boundary.conditions.items():
        if isinstance(closure, PressureOutlet):
            conditions[name] = Dirichlet(0.0)  # potential datum at the outflow
            has_reference = True
        elif isinstance(closure, VelocityInlet):
            faces = mesh.face_patches.indices(name)
            normal = geometry.face.normal[faces]
            centroid = geometry.face.centroid[faces]
            inlet_velocity = closure.reference_velocity(normal, centroid)
            normal_velocity = float(jnp.mean(jnp.sum(inlet_velocity * normal, axis=1)))
            # Neumann prescribes -Gamma d(phi)/dn; with Gamma = 1, d(phi)/dn = u_in . n.
            conditions[name] = Neumann(flux=-normal_velocity)
        else:  # walls (no-slip, moving): no penetration
            conditions[name] = ZeroGradient()

    fixed_cells = fixed_values = None
    if not has_reference:
        # No inflow/outflow to build a potential from. A body force still drives the domain, though
        # (a streamwise-periodic channel is exactly this: no through-flow boundary at all), and its
        # potential is uniform, so the harmonic solve has nothing to add — start from the plug the
        # force sustains instead of from rest. A domain driven only by a moving wall is *not* this
        # case: it has no net through-flow, so its potential really is zero and a plug would violate
        # the stationary walls.
        plug = body_force_velocity(momentum)
        if float(jnp.linalg.norm(plug)) > 0.0:
            velocity = jnp.broadcast_to(plug, (mesh.n_cells, mesh.dim))
            return momentum.pack(velocity, jnp.zeros(mesh.n_cells))
        if momentum.pressure_pin is None:
            return momentum.initial_state()  # closed domain: no potential through-flow
        fixed_cells = jnp.array([momentum.pressure_pin])
        fixed_values = jnp.array([0.0])

    phi, assembler = laplace_field(
        mesh,
        geometry,
        BoundaryConditions(conditions),
        gradient_scheme=gradient_scheme,
        fixed_cells=fixed_cells,
        fixed_values=fixed_values,
    )
    velocity = assembler.gradient(phi)
    return momentum.pack(velocity, jnp.zeros(mesh.n_cells))
