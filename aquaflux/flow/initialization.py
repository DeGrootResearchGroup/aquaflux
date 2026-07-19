"""Cheap field initializers -- a scalar Laplace solve and a potential-flow velocity.

A good initial condition is what lets the monolithic coupled Newton solve (and, less critically, the
segregated loop) start from nothing. The two building blocks here are both **single linear SPD
solves**, so they cost a fraction of one nonlinear iteration:

- :func:`laplace_field` solves ``div(Gamma grad phi) = 0`` for the boundary-value harmonic field --
  the smooth interpolant of a scalar's boundary data into the interior.
- :func:`potential_flow` uses it to build an irrotational velocity ``u = grad phi`` whose normal
  component matches the flow boundary conditions (inflow at inlets, no penetration at walls), i.e. the
  classic Fluent-style "hybrid" velocity initializer. It is divergence-free and respects the geometry,
  unlike a uniform plug guess -- and, being a real discrete gradient field, it carries the tiny
  asymmetry that lifts the coupled solve's degeneracy on a perfectly symmetric velocity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from aquaflux.boundary import BoundaryConditions, Dirichlet, Neumann, ZeroGradient
from aquaflux.discretization import DiffusionFlux, FixedValueCells, ResidualAssembler
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.solve import NewtonSolver

from .boundary import PressureOutlet, VelocityInlet

if TYPE_CHECKING:
    from aquaflux.mesh import Mesh, MeshGeometry
    from aquaflux.schemes import GradientScheme

    from .momentum import MomentumContinuity


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

    A pure-diffusion residual is linear, so a single Newton step is exact. Returns the solved field
    **and** the assembler (so a caller can reconstruct ``grad phi`` with the same boundary closures).

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

    field = NewtonSolver(iterations=1).solve(residual, jnp.zeros(mesh.n_cells))
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

    A closed domain (no outlet, no pressure datum) has no potential through-flow; this returns the zero
    state there. When there is no outlet but the momentum carries a ``pressure_pin``, that cell pins the
    otherwise-singular Laplacian.

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
