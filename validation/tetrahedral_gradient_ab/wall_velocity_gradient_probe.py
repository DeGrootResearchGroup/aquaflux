"""Issue #435: impose a wall-model velocity gradient at the wall cells, the way omega's is imposed.

On a wall-function mesh a wall cell's reconstructed velocity gradient reads the no-slip wall value -- a
jump across the cell where the true profile is logarithmic -- and the multiple-correction scheme's
Hessian correction in the next ring of cells differentiates that jump into a large spurious strain rate.
Omega has the same problem at its wall cells and is handled by imposing its analytical gradient
(:class:`~aquaflux.schemes.ImposedGradient`) before the reconstruction's second pass. This prototype does
the same for velocity:

    grad u_i = t_i * s * n,   s = (1 - f) |U_t| / d + f u_tau / (kappa d)

at each wall cell: ``n`` the area-weighted inward wall normal, ``t`` the direction of the cell's
tangential velocity, ``u_tau = beta*^(1/4) sqrt(k)`` and ``f`` the log-layer weight -- the same
:func:`~aquaflux.turbulence.log_layer_shear_rate` and :func:`~aquaflux.turbulence.wall_function_weight`
the closure's strain rate is blended with. ``k`` is read live from the coupled state. Tangential
derivatives are not modelled and are set to zero there.

It reports, at a smooth state (the Re/10 anchor converged with corrected Green--Gauss) and optionally
from a uniform plug: the coupled residual, the strain rate over omega off the wall cells, where the omega
production cap binds, non-positive omega diagonals, and a short march.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/wall_velocity_gradient_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aquaflux.turbulence.coupled as coupled_module
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from aquaflux.flow import MomentumContinuity
from aquaflux.io import read_openfoam
from aquaflux.schemes import (
    CorrectedGreenGauss,
    ImposedGradient,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.turbulence import (
    CoupledRANS,
    SSTModel,
    inlet_k,
    inlet_omega,
    log_layer_shear_rate,
    wall_function_weight,
)
from aquaflux.turbulence.strain import safe_sqrt
from aquaflux.vectors import dot, norm_squared, scale
from compare import INTENSITY, LENGTH_SCALE, POLYMESH, U_IN, build_case
from diffusion_operator_probe import FirstPassOnly
from seed_state_probe import CAP, rings_from_wall
from warm_start_probe import RATIO, SEED_STEPS, march

STEPS = int(os.environ.get("TET_STEPS", "12"))
FROM_PLUG = os.environ.get("TET_FROM_PLUG", "0") == "1"


class WallModelVelocityMomentum(MomentumContinuity):
    """The momentum block with its wall cells' velocity gradient imposed from the wall model."""

    # Defaulted only because the parent's last field is; `with_wall_model` always sets all of them.
    wall_cells: jnp.ndarray = None
    wall_normal: jnp.ndarray = None  # (n_wall, dim), inward unit normal
    wall_distance: jnp.ndarray = None  # (n_wall,)
    wall_nu: jnp.ndarray = None  # (n_wall,) molecular kinematic viscosity
    wall_k: jnp.ndarray = None  # (n_wall,) set from the live state by the coupled residual
    model: SSTModel = None
    #: Cells whose velocity gradient skips the Hessian correction (first pass only); ``None`` for none.
    first_pass_cells: jnp.ndarray = None

    def _imposed_velocity_gradient(self, velocity: jnp.ndarray) -> jnp.ndarray:
        u = velocity[self.wall_cells]
        tangential = u - scale(self.wall_normal, dot(u, self.wall_normal))
        speed = safe_sqrt(norm_squared(tangential))
        direction = tangential / jnp.where(speed > 0.0, speed, 1.0)[:, None]
        weight = wall_function_weight(self.wall_nu, self.wall_distance, self.wall_k, self.model)
        shear = (1.0 - weight) * speed / self.wall_distance + weight * log_layer_shear_rate(
            self.wall_distance, self.wall_k, self.model
        )
        # (n_wall, dim, dim): row i is grad u_i
        return direction[:, :, None] * (shear[:, None] * self.wall_normal)[:, None, :]

    def _velocity_gradient(self, velocity: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        zero_gradient = jnp.zeros((self.mesh.n_cells, self.mesh.dim, self.mesh.dim))
        leading = self._boundary_velocity(velocity, zero_gradient)
        imposed = self._imposed_velocity_gradient(velocity)
        columns = [
            self.gradient_scheme.gradients(
                velocity[:, i],
                self.mesh,
                self.geometry,
                leading[:, i],
                imposed=ImposedGradient(self.wall_cells, imposed[:, i, :]),
                boundary_values_at=lambda g, i=i: self._boundary_velocity_component(velocity, i, g),
                boundary_gradient_weight=self._velocity_boundary_gradient_weight(
                    velocity, i, zero_gradient
                ),
            )
            for i in range(self.mesh.dim)
        ]
        gradient = jnp.stack(columns, axis=1)
        if self.first_pass_cells is not None:
            first_pass = FirstPassOnly(self.gradient_scheme)
            first = jnp.stack(
                [
                    first_pass.gradients(
                        velocity[:, i],
                        self.mesh,
                        self.geometry,
                        leading[:, i],
                        imposed=ImposedGradient(self.wall_cells, imposed[:, i, :]),
                        boundary_gradient_weight=self._velocity_boundary_gradient_weight(
                            velocity, i, zero_gradient
                        ),
                    )
                    for i in range(self.mesh.dim)
                ],
                axis=1,
            )
            cells = self.first_pass_cells
            gradient = gradient.at[cells].set(first[cells])
        return gradient, self._boundary_velocity(velocity, gradient)


_ORIGINAL_EFFECTIVE_MOMENTUM = coupled_module._effective_momentum


def _effective_momentum(momentum, turbulence, flow, k, omega):
    if isinstance(momentum, WallModelVelocityMomentum):
        momentum = eqx.tree_at(lambda m: m.wall_k, momentum, k[momentum.wall_cells])
    return _ORIGINAL_EFFECTIVE_MOMENTUM(momentum, turbulence, flow, k, omega)


coupled_module._effective_momentum = _effective_momentum


def with_wall_model(coupled: CoupledRANS, first_ring: bool = False) -> CoupledRANS:
    momentum, turbulence = coupled.momentum, coupled.turbulence
    mesh, geometry = momentum.mesh, momentum.geometry
    face_cells = mesh.face_cells
    wall = turbulence.wall_cells
    zeros = jnp.zeros((mesh.n_faces, mesh.dim))
    inward = zeros.at[turbulence.wall_faces].set(
        -scale(geometry.face.normal, geometry.face.area)[turbulence.wall_faces]
    )
    summed = face_cells.scatter(inward, zeros)[wall]
    normal = summed / jnp.linalg.norm(summed, axis=-1, keepdims=True)
    fields = {name: getattr(momentum, name) for name in momentum.__dataclass_fields__}
    wall_model = WallModelVelocityMomentum(
        **fields,
        wall_cells=wall,
        wall_normal=normal,
        wall_distance=turbulence.wall_distance[wall],
        wall_nu=turbulence.molecular_viscosity[wall],
        wall_k=jnp.zeros(wall.shape[0]),
        model=turbulence.model,
        first_pass_cells=(
            jnp.asarray(np.flatnonzero(rings_from_wall(mesh, wall) == 1)) if first_ring else None
        ),
    )
    return CoupledRANS.build(wall_model, turbulence)


def report(label, coupled, flow, k, omega, rings) -> None:
    n = coupled.momentum.mesh.n_cells
    closure, _ = coupled.effective_momentum(flow, k, omega)
    ratio = np.asarray(closure.strain_rate / omega)
    state = coupled.state_from_physical(flow, k, omega)

    def r_omega(w):
        return coupled.residual(state.at[5 * n :].set(w))[5 * n :]

    diag = np.diag(np.asarray(jax.jacfwd(r_omega)(state[5 * n :])))
    free = rings > 0
    bad = free & (diag <= 0)
    print(
        f"[{label:34s}] |R| {float(jnp.linalg.norm(coupled.residual(state))):.3e}; S/omega max "
        f"{ratio[free].max():.2e}; cap binds {(free & (ratio > CAP)).sum()}; omega diag<=0 {bad.sum()}"
        + (f" at rings {np.bincount(rings[bad], minlength=4)[1:].tolist()}" if bad.any() else ""),
        flush=True,
    )


def main() -> None:
    mesh = read_openfoam(POLYMESH)
    geometry = mesh.geometry()
    n = mesh.n_cells
    seed_case = build_case(CorrectedGreenGauss()).with_scaled_molecular_viscosity(RATIO)
    rings = rings_from_wall(mesh, seed_case.turbulence.wall_cells)
    flow0 = seed_case.momentum.pack(jnp.zeros((n, 3)).at[:, 0].set(U_IN), jnp.zeros(n))
    k0 = jnp.full(n, float(inlet_k(jnp.array(U_IN), INTENSITY)))
    omega0 = jnp.full(n, float(inlet_omega(jnp.array(k0[0]), LENGTH_SCALE, SSTModel())))
    seed = march("seed", seed_case, flow0, k0, omega0, 1e-4, SEED_STEPS)
    if seed is None:
        return
    multcorr = MultipleCorrectionGradient(
        boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
    ).bind(mesh, geometry)
    plain = build_case(multcorr).with_scaled_molecular_viscosity(RATIO)
    # Attach the wall model AFTER rescaling: it copies the molecular viscosity at build.
    walled = with_wall_model(build_case(multcorr).with_scaled_molecular_viscosity(RATIO))
    option1 = with_wall_model(
        build_case(multcorr).with_scaled_molecular_viscosity(RATIO), first_ring=True
    )
    report("multcorr, as shipped", plain, *seed, rings)
    report("multcorr + wall-model grad (b)", walled, *seed, rings)
    report("(b) + first pass in ring 1 (option 1)", option1, *seed, rings)
    if STEPS:
        march("option 1, from seed", option1, *seed, 1e-3, STEPS)
        if FROM_PLUG:
            march("option 1, from plug", option1, flow0, k0, omega0, 1e-3, STEPS)


if __name__ == "__main__":
    main()
