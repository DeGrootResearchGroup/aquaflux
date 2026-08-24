"""Why a one-sided boundary closure fails on a gradient-type patch: it corrects twice.

A residual assembler reconstructs its gradient from **leading-order** boundary values -- the field's
boundary closures evaluated at *zero* gradient -- so that the residual stays a single-pass function
of the field. That is deliberate and it is fine for the value itself. It is not fine for a closure
that then differences against that value.

``ZeroGradient.face_value`` is ``phi_owner + tangential_correction(grad_owner, d, n)``. Evaluated at
zero gradient the correction vanishes, so the boundary value handed to the reconstruction is
**exactly** ``phi_owner``. :class:`~aquaflux.schemes.SkewCorrectedGradient` then forms::

    rise = boundary_value - phi_owner - non_orthogonal_correction(owner_gradient, d, n)
         = -non_orthogonal_correction(owner_gradient, d, n)

and reports a face-normal derivative of ``rise / (d.n)``. It has subtracted a correction the
boundary value never added, and divided the residue by the wall-normal distance -- the smallest
length in the mesh. The whole normal derivative on such a patch is an artifact.

On a **Dirichlet** patch there is no such problem: the leading-order and full boundary values
coincide (a prescribed value does not depend on the gradient), so the subtraction is exactly the
term that makes the one-sided difference linear-exact on a skewed mesh, which is what it is for.

This probe measures both halves on the pitzDaily mesh: that ``boundary_value - phi_owner`` is
identically zero on every gradient-type patch, and how large the resulting spurious derivative is.

Run
---
``python3 validation/multiple_correction/boundary_closure_probe.py``
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "pitzdaily_openfoam"))

import compare  # noqa: E402  (the validated benchmark: mesh, physics, boundary conditions)
from aquaflux.discretization import ResidualAssembler  # noqa: E402
from aquaflux.properties import PropertyModel  # noqa: E402
from aquaflux.schemes import (  # noqa: E402
    MultipleCorrectionGradient,
    SkewCorrectedGradient,
)
from aquaflux.schemes.interpolation import non_orthogonal_correction  # noqa: E402
from aquaflux.turbulence import omega_wall  # noqa: E402
from aquaflux.vectors import dot  # noqa: E402


def main() -> None:
    """Measure the closure's boundary rise and normal derivative, per patch."""
    scheme = MultipleCorrectionGradient(boundary_closure=SkewCorrectedGradient())
    case = compare.build_case(gradient_scheme=scheme)
    mesh, geometry, turbulence = case["momentum"].mesh, case["geom"], case["turbulence"]
    face_cells = mesh.face_cells

    # A field whose boundary behaviour is the point: the analytical near-wall omega, which is what
    # the wall cells genuinely carry and which varies like 1/d**2.
    k = jnp.full(mesh.n_cells, 0.1)
    omega = omega_wall(
        turbulence.molecular_viscosity, turbulence.wall_distance, k, turbulence.model
    )

    bound = scheme.bind(mesh, geometry)
    assembler = ResidualAssembler.build(
        mesh,
        geometry,
        PropertyModel({}),
        (),
        turbulence.omega_boundary,
        gradient_scheme=bound,
    )
    # Leading order: the closures evaluated at zero gradient, exactly as `_gradient` does.
    boundary_values = assembler.boundary_values(omega, jnp.zeros((mesh.n_cells, mesh.dim)), {})

    gradient = bound.reconstruct(omega, mesh, geometry, boundary_values)[0]
    owner, normal = face_cells.owner, geometry.face.normal
    displacement = geometry.face.centroid - geometry.cell.centroid[owner]
    along = np.asarray(dot(displacement, normal))
    correction = np.asarray(non_orthogonal_correction(gradient[owner], displacement, normal))
    owner_value = np.asarray(omega)[np.asarray(owner)]
    difference = np.asarray(boundary_values) - owner_value
    rise = difference - correction

    print(f"pitzDaily: {mesh.n_cells} cells, {mesh.n_faces} faces")
    print(
        f"\n{'patch':12s} {'faces':>6s} {'max |bval - phi_P|':>19s} {'med |corr|':>12s} "
        f"{'med |d.n|':>11s} {'med |rise/d.n|':>15s} {'max':>11s}"
    )
    for name in mesh.face_patches.names:
        faces = np.asarray(mesh.face_patches.indices(name))
        if faces.size == 0:
            continue
        derivative = np.abs(rise[faces] / along[faces])
        print(
            f"{name:12s} {faces.size:6d} {np.abs(difference[faces]).max():19.3e} "
            f"{np.median(np.abs(correction[faces])):12.3e} "
            f"{np.median(np.abs(along[faces])):11.3e} "
            f"{np.median(derivative):15.3e} {derivative.max():11.3e}"
        )
    print(
        "\nEvery gradient-type patch reports max |bval - phi_P| of exactly zero, so its whole\n"
        "reported normal derivative is -correction/(d.n): an artifact of correcting twice.\n"
        "The Dirichlet inlet carries a real difference, which is what the correction is for."
    )


if __name__ == "__main__":
    main()
