"""Analytical validation of the coupled flow on a skewed (non-orthogonal) mesh.

A linear Couette velocity ``u = (y, 0)`` with constant pressure is an exact Stokes solution
(``div u = 0``, ``grad^2 u = 0``, so ``grad p = 0``). Because the Rhie--Chow mass flux and the
momentum pressure force reconstruct face values to the integration point — carrying the
``grad·(x_ip − x_g)`` skewness correction rather than stopping at the projection foot ``x_g`` — a
linear velocity is represented exactly, so the discrete divergence and the pressure force vanish at
the exact field and the solver reproduces it to solver tolerance even on a non-orthogonal mesh.

A plain owner/neighbour blend (the pre-correction behaviour) leaves an ``O(skew)`` error on such a
mesh, so the tight tolerance here is what distinguishes the integration-point reconstruction. In
:func:`test_stokes_couette_is_exact_on_a_skewed_mesh` all boundaries are Dirichlet velocity, so no
boundary tangential correction enters at all.

:func:`test_a_zero_gradient_flow_patch_carries_its_tangential_correction` opens one side into a
pressure outlet, whose velocity closure is zero-gradient, and is the case that *does* exercise the
correction: the boundary face value is then ``u_P + grad u_P . (d - (d.n) n)``, and dropping that
term reports the owner cell's own velocity at a face displaced tangentially from it.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions
from aquaflux.flow import (
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CorrectedGreenGauss, MultipleCorrectionGradient, OwnerGradient
from aquaflux.solve import newton_step

from tests.support.meshes import perturbed_grid_2d


def _couette(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack([x[:, 1], jnp.zeros(x.shape[0])], axis=1)  # u = (y, 0)


def _solve_couette(n: int = 8, perturb: float = 0.2, seed: int = 2):
    mesh = perturbed_grid_2d(
        n, n, lx=1.0, ly=1.0, perturb=perturb, seed=seed, named_boundaries=True
    )
    geom = mesh.geometry()
    assembler = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        CorrectedGreenGauss(),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),  # u = (1, 0) at y = 1
                "bottom": NoSlipWall(),  # u = (0, 0) at y = 0
                "left": VelocityInlet(velocity=_couette),  # u = (y, 0)
                "right": VelocityInlet(velocity=_couette),  # u = (y, 0)
            }
        ),
        pressure_pin=0,  # closed domain (all velocity Dirichlet): fix the pressure level
    )
    # Stokes (no advection) so the residual is affine: one Newton step is exact.
    state = eqx.filter_jit(newton_step)(assembler.residual, assembler.initial_state())
    return geom, assembler, state


def test_stokes_couette_is_exact_on_a_skewed_mesh() -> None:
    """The linear Couette velocity is reproduced to solver tolerance on a non-orthogonal mesh."""
    geom, assembler, state = _solve_couette()
    velocity, _ = assembler.unpack(state)
    u_exact = np.asarray(_couette(geom.cell.centroid))
    error = np.max(np.abs(np.asarray(velocity) - u_exact))
    assert error < 1e-6


def _open_couette(n: int = 8, perturb: float = 0.2, seed: int = 2):
    """The same Couette domain with the right side opened into a pressure outlet.

    ``u = (y, 0)`` with ``p = 0`` is still the exact solution, and the outlet's closures are exact
    for it: the velocity is zero-gradient there and the exact field has ``grad u . n = 0`` on that
    patch (the perturbation leaves boundary nodes in place, so the right faces keep the normal
    ``(1, 0)`` exactly), while the prescribed ``p_b = 0`` is the exact pressure. What the
    perturbation *does* move is the owner centroid behind each of those faces, so ``d`` picks up a
    tangential component and the zero-gradient closure's correction becomes load-bearing.

    Reconstructed with :class:`~aquaflux.schemes.MultipleCorrectionGradient`, which is exact for a
    quadratic field on any mesh — so the reconstructed gradient carries no error of its own and the
    boundary values are the only thing under test. (Corrected Green--Gauss caps near first order on
    a skewed mesh: on this same 8x8 grid it reconstructs the linear field's unit gradient to 11 %,
    which would blur the comparison rather than sharpen it.)
    """
    mesh = perturbed_grid_2d(
        n, n, lx=1.0, ly=1.0, perturb=perturb, seed=seed, named_boundaries=True
    )
    geom = mesh.geometry()
    assembler = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),  # u = (1, 0) at y = 1
                "bottom": NoSlipWall(),  # u = (0, 0) at y = 0
                "left": VelocityInlet(velocity=_couette),  # u = (y, 0)
                "right": PressureOutlet(pressure=0.0),  # zero-gradient velocity, p = 0
            }
        ),
    )
    exact = assembler.pack(_couette(geom.cell.centroid), jnp.zeros(mesh.n_cells))
    return mesh, geom, assembler, exact


def test_a_zero_gradient_flow_patch_carries_its_tangential_correction() -> None:
    """The outlet's face velocity is the exact field there, not the owner cell's own velocity.

    A zero-gradient closure states that the *normal* derivative vanishes, not that the face value
    equals the owner value: on a mesh where the owner centroid sits off the face normal the two
    differ by the field's variation along the tangential offset. Evaluating the closure at the
    reconstructed gradient recovers the exact value to round-off here; evaluating it at a zero
    gradient — which is all the flow block could do before it carried one — returns the owner value,
    and the error is the tangential displacement itself.
    """
    mesh, geom, assembler, exact = _open_couette()
    fields = assembler.flow_fields(exact).velocity_fields
    faces = np.asarray(mesh.face_patches.indices("right"))
    owner = np.asarray(mesh.face_cells.owner)[faces]

    exact_face = np.asarray(_couette(geom.face.centroid[faces]))
    corrected = np.asarray(fields.boundary_velocity[faces])
    uncorrected = np.asarray(fields.velocity[owner])  # the closure at a zero gradient

    assert np.max(np.abs(corrected - exact_face)) < 1e-13
    # The correction is not merely present but load-bearing: without it the patch is wrong by the
    # tangential offset -- 1.2e-2 on this mesh, ten orders of magnitude above the round-off above.
    assert np.max(np.abs(uncorrected - exact_face)) > 1e-3


def test_the_flow_residual_vanishes_at_the_exact_field_through_an_outlet() -> None:
    """With the outlet's face velocity exact, the whole coupled residual is round-off.

    The consequence of the boundary value for the equations that read it: the viscous flux at the
    outlet is built from that face value, so a face value carrying no tangential correction leaves a
    spurious wall-normal difference divided by ``d.n``. The exact Stokes field then fails to satisfy
    the discrete momentum balance on a skewed mesh, which is the defect
    :func:`test_a_zero_gradient_flow_patch_carries_its_tangential_correction` pins for the boundary
    value itself.

    Both blocks are asserted. :meth:`~aquaflux.flow.PressureOutlet.mass_flux` forms its through-flow
    term from that same corrected face velocity (not the owner cell's), so the continuity residual
    vanishes here too -- before that fix it held a residue exactly equal to the outlet's mass-flux
    error, ``rho ((u_face - u_owner) . n) A``, which does not shrink under mesh refinement (it is a
    zeroth-order consistency error at the outlet-owning cells, not a discretization truncation term).
    """
    _, _, assembler, exact = _open_couette()
    momentum_residual, continuity_residual = assembler.unpack(assembler.residual(exact))
    assert np.max(np.abs(np.asarray(momentum_residual))) < 1e-13
    assert np.max(np.abs(np.asarray(continuity_residual))) < 1e-13


def test_stokes_couette_is_exact_on_a_skewed_mesh_through_an_outlet() -> None:
    """The open-domain (outlet) case reaches the same solver tolerance as the closed one.

    The control here is :func:`test_stokes_couette_is_exact_on_a_skewed_mesh`, which is this same
    case with every velocity boundary Dirichlet. Before the outlet's through-flow term read the
    corrected face velocity, the zeroth-order mass-flux error at the outlet capped this solve's
    accuracy at ``~1.2e-3`` regardless of mesh refinement (8x8 -> 32x32 gave ``4.0e-3 -> 1.2e-3``,
    not a converging sequence) -- the outlet closure, not the interior discretization, was the
    limit.
    """
    mesh = perturbed_grid_2d(8, 8, lx=1.0, ly=1.0, perturb=0.2, seed=2, named_boundaries=True)
    geom = mesh.geometry()
    assembler = MomentumContinuity.build(
        mesh,
        geom,
        PropertyModel({"viscosity": Constant(1.0), "density": Constant(1.0)}),
        MultipleCorrectionGradient(boundary_closure=OwnerGradient(), fallback=None),
        BoundaryConditions(
            {
                "top": MovingWall(velocity=(1.0, 0.0)),
                "bottom": NoSlipWall(),
                "left": VelocityInlet(velocity=_couette),
                "right": PressureOutlet(pressure=0.0),
            }
        ),
    )
    state = eqx.filter_jit(newton_step)(assembler.residual, assembler.initial_state())
    velocity, _ = assembler.unpack(state)
    u_exact = np.asarray(_couette(geom.cell.centroid))
    error = np.max(np.abs(np.asarray(velocity) - u_exact))
    assert error < 1e-6
