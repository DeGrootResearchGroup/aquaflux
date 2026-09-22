"""Issue #435: does the multiple-correction scheme march a LAMINAR duct on the tetrahedral mesh?

The scheme's second pass is exact for quadratic fields and, on tetrahedra, has to cancel a first-order
gradient error comparable to the whole within-cell variation (``exact_function_probe.py``, and the
``|gradient_defect| / h`` comparison in the case README). If that is what stops the coupled RANS march,
it should be harmless where the solution is resolved and close to quadratic: a laminar duct at low
Reynolds number, whose developed profile is a parabola.

This runs the flow residual alone -- no turbulence fields, wall model or production terms -- through
``solve_flow_march``, the same staged march ``compare.py`` runs the turbulent case on, and configured the
same way: the wall-tapered seed of ``compare._hybrid_start`` (potential flow's Laplace solve is unusable
on these meshes), a two-point ramp (anchor at ``mu * RATIO``, then the target), a complete-LU
materialized preconditioner, the same dual-time loop and retry policy, one line per step. The
difference from the turbulent march is only the residual and its measure (``FlowMeasures``); the
convergence test here is ``Convergence(rtol=TET_LAMINAR_RTOL, atol=0)`` on the row-scaled measure.

Meshes come from ``TET_MESHES`` (``label=polyMesh path`` pairs, comma separated; default the case mesh),
the Reynolds number from ``TET_LAMINAR_RE`` (default 50), the arms from ``TET_ARMS`` (comma
separated; default the first four) and the per-rung step cap from ``TET_STEPS``.

Prediction of the quadratic-exactness explanation: every scheme, the full multiple-correction scheme
included, converges. A failure of the full scheme here would say the explanation is wrong; a failure of
the schemes that march the turbulent case would say this control is not sound.

Run: validation/run_case.sh validation/tetrahedral_gradient_ab/laminar_duct_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from aquaflux.boundary import BoundaryConditions
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    MomentumContinuity,
    NoSlipWall,
    PressureOutlet,
    VelocityFields,
    VelocityInlet,
    solve_flow_march,
)
from aquaflux.flow.rhie_chow import interior_mass_flux
from aquaflux.io import read_openfoam
from aquaflux.mesh import distance_to_patches
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import (
    CompactGreenGauss,
    CorrectedGreenGauss,
    MultipleCorrectionGradient,
    OwnerGradient,
    SkewCorrectedGradient,
)
from aquaflux.solve import CompleteLu, Convergence, DualTimeLoop, MaterializedJacobian
from compare import BACKEND, INNER_STEPS, INNER_TOL, POLYMESH, RATIO, RETRY
from diffusion_operator_probe import (
    CorrectionCapped,
    FirstPassOnCells,
    FirstPassOnly,
    FirstPassWhereIllConditioned,
)
from seed_state_probe import rings_from_wall

RE = float(os.environ.get("TET_LAMINAR_RE", "50"))
RTOL = float(os.environ.get("TET_LAMINAR_RTOL", "1e-6"))
DH, U_IN, RHO = 0.025, 1.0, 1.0
MU = RHO * U_IN * DH / RE
MAX_STEPS = int(os.environ.get("TET_STEPS", "30"))


class _SplitMomentum(MomentumContinuity):
    """The flow residual with the second pass withheld from ONE field's gradient.

    ``first_pass_on`` is ``"pressure"`` or ``"velocity"``: that field's gradient is reconstructed by
    the multiple-correction first pass alone, the other's by the full scheme. The residual reads one
    ``gradient_scheme`` for both, so the split swaps it for the named field's call.
    """

    first_pass_on: str = eqx.field(static=True, default="pressure")

    def _first_pass(self) -> MomentumContinuity:
        return eqx.tree_at(lambda m: m.gradient_scheme, self, FirstPassOnly(self.gradient_scheme))

    def _pressure_gradient(self, pressure):
        target = self._first_pass() if self.first_pass_on == "pressure" else self
        return MomentumContinuity._pressure_gradient(target, pressure)

    def _velocity_gradient(self, velocity):
        target = self._first_pass() if self.first_pass_on == "velocity" else self
        return MomentumContinuity._velocity_gradient(target, velocity)


def split(momentum: MomentumContinuity, first_pass_on: str) -> _SplitMomentum:
    fields = {f.name: getattr(momentum, f.name) for f in dataclasses.fields(momentum) if f.init}
    return _SplitMomentum(**fields, first_pass_on=first_pass_on)


class _StabilizedMomentum(MomentumContinuity):
    """The flow residual with the first-pass gradient in the cancelling-difference terms only.

    Two terms of the residual are a compact two-point difference plus a gradient-reconstruction
    correction whose sign is load-bearing: the Rhie-Chow pressure damping ``(p_N - p_P) - interp(grad p)
    . d`` and the non-orthogonal diffusion correction ``grad phi . (tangential offset)``. With
    ``stabilize_damping`` the damping reads the multiple-correction FIRST-pass pressure gradient; with
    ``stabilize_diffusion`` the momentum viscous flux reads the first-pass velocity gradient. Every face
    VALUE (face pressure, the velocity interpolation, the boundary closures) keeps the full scheme.
    """

    stabilize_damping: bool = eqx.field(static=True, default=False)
    stabilize_diffusion: bool = eqx.field(static=True, default=False)

    def _first_pass(self) -> MomentumContinuity:
        return eqx.tree_at(lambda m: m.gradient_scheme, self, FirstPassOnly(self.gradient_scheme))

    def _mass_flux(
        self, velocity, grad_velocity, boundary_velocity, pressure, grad_pressure, d_coeff
    ):
        if not self.stabilize_damping:
            return MomentumContinuity._mass_flux(
                self, velocity, grad_velocity, boundary_velocity, pressure, grad_pressure, d_coeff
            )
        face_cells = self.mesh.face_cells
        damping_gradient, _ = MomentumContinuity._pressure_gradient(self._first_pass(), pressure)
        interior_flux = interior_mass_flux(
            velocity,
            grad_velocity,
            pressure,
            damping_gradient,
            d_coeff,
            face_cells,
            self.geometry,
            self.interp_factor,
            self.normal_distance,
            self.density,
        )
        mdot = face_cells.combine_face_values(interior_flux, 0.0)
        return self._boundary_mass_flux(boundary_velocity, pressure, grad_pressure, d_coeff, mdot)

    def _momentum_residual(self, kinematic, pressure_face, mdot):
        if self.stabilize_diffusion:
            gradient, boundary_velocity = MomentumContinuity._velocity_gradient(
                self._first_pass(), kinematic.velocity
            )
            kinematic = VelocityFields(kinematic.velocity, boundary_velocity, gradient)
        return MomentumContinuity._momentum_residual(self, kinematic, pressure_face, mdot)


def stabilized(momentum: MomentumContinuity, *, damping: bool, diffusion: bool):
    fields = {f.name: getattr(momentum, f.name) for f in dataclasses.fields(momentum) if f.init}
    return _StabilizedMomentum(**fields, stabilize_damping=damping, stabilize_diffusion=diffusion)


#: The consumers of the cell gradient in the flow residual that the bisect gives the full scheme to.
CONSUMERS = (
    "damping",
    "viscous flux",
    "face pressure",
    "momentum interpolation",
    "momentum diagonal",
    "boundary mass flux",
    "boundary velocity",
    "boundary pressure",
)


class _FullGradientFor(MomentumContinuity):
    """A first-pass flow residual in which ONE consumer of the gradient reads the full scheme.

    The base residual reconstructs every gradient by the multiple-correction first pass alone, which
    marches. ``consumer`` names the single place that reads the full scheme's gradient (from the twin
    ``full``): the Rhie-Chow pressure damping, the momentum viscous flux, the face pressure of the
    pressure force, the momentum interpolation inside the mass flux, the momentum diagonal, the boundary
    mass-flux closure, or the boundary velocity / pressure the closures re-evaluate at the gradient.
    """

    full: MomentumContinuity | None = None
    consumer: str = eqx.field(static=True, default="damping")

    def _mass_flux(
        self, velocity, grad_velocity, boundary_velocity, pressure, grad_pressure, d_coeff
    ):
        damping = grad_pressure
        interpolation = grad_velocity
        closure = grad_pressure
        if self.consumer == "damping":
            damping = MomentumContinuity._pressure_gradient(self.full, pressure)[0]
        if self.consumer == "momentum interpolation":
            interpolation = MomentumContinuity._velocity_gradient(self.full, velocity)[0]
        if self.consumer == "boundary mass flux":
            closure = MomentumContinuity._pressure_gradient(self.full, pressure)[0]
        face_cells = self.mesh.face_cells
        interior_flux = interior_mass_flux(
            velocity,
            interpolation,
            pressure,
            damping,
            d_coeff,
            face_cells,
            self.geometry,
            self.interp_factor,
            self.normal_distance,
            self.density,
        )
        mdot = face_cells.combine_face_values(interior_flux, 0.0)
        return self._boundary_mass_flux(boundary_velocity, pressure, closure, d_coeff, mdot)

    def _momentum_residual(self, kinematic, pressure_face, mdot):
        if self.consumer == "viscous flux":
            gradient = MomentumContinuity._velocity_gradient(self.full, kinematic.velocity)[0]
            kinematic = VelocityFields(kinematic.velocity, kinematic.boundary_velocity, gradient)
        return MomentumContinuity._momentum_residual(self, kinematic, pressure_face, mdot)

    def _face_pressure(self, pressure, grad_pressure, boundary_pressure):
        if self.consumer == "face pressure":
            grad_pressure = MomentumContinuity._pressure_gradient(self.full, pressure)[0]
        return MomentumContinuity._face_pressure(self, pressure, grad_pressure, boundary_pressure)

    def momentum_matrix_diagonal(self, velocity, grad_velocity=None):
        if self.consumer == "momentum diagonal":
            grad_velocity = MomentumContinuity._velocity_gradient(self.full, velocity)[0]
        return MomentumContinuity.momentum_matrix_diagonal(self, velocity, grad_velocity)

    def _velocity_gradient(self, velocity):
        gradient, boundary = MomentumContinuity._velocity_gradient(self, velocity)
        if self.consumer == "boundary velocity":
            boundary = MomentumContinuity._velocity_gradient(self.full, velocity)[1]
        return gradient, boundary

    def _pressure_gradient(self, pressure):
        gradient, boundary = MomentumContinuity._pressure_gradient(self, pressure)
        if self.consumer == "boundary pressure":
            boundary = MomentumContinuity._pressure_gradient(self.full, pressure)[1]
        return gradient, boundary


def full_gradient_for(first: MomentumContinuity, full: MomentumContinuity, consumer: str):
    fields = {f.name: getattr(first, f.name) for f in dataclasses.fields(first) if f.init}
    return _FullGradientFor(**fields, full=full, consumer=consumer)


def meshes() -> dict[str, Path]:
    spec = os.environ.get("TET_MESHES", "")
    if not spec:
        return {"case mesh": POLYMESH}
    return {pair.split("=", 1)[0]: Path(pair.split("=", 1)[1]) for pair in spec.split(",")}


def seed(momentum: MomentumContinuity) -> jnp.ndarray:
    """The wall-tapered velocity of ``compare._hybrid_start`` (its flow part), zero pressure."""
    mesh, geometry = momentum.mesh, momentum.geometry
    distance = distance_to_patches(mesh, geometry, ["walls"])
    fraction = jnp.clip(distance / jnp.max(distance), 0.0, 1.0)
    shape = 1.0 - (1.0 - fraction) ** 2
    velocity = jnp.zeros((mesh.n_cells, mesh.dim)).at[:, 0].set(U_IN * shape)
    return momentum.pack(velocity, jnp.zeros(mesh.n_cells))


def main() -> None:
    print(f"=== laminar duct, Re_Dh {RE:g} (mu {MU:.2e}), flow residual only ===", flush=True)
    preconditioner = MaterializedJacobian(CompleteLu(backend=BACKEND))
    dual_time = DualTimeLoop(inner_steps=INNER_STEPS, inner_tol=INNER_TOL)
    for label, path in meshes().items():
        mesh = read_openfoam(path)
        geometry = mesh.geometry()
        repaired = MultipleCorrectionGradient(
            boundary_closure=OwnerGradient(), fallback=SkewCorrectedGradient()
        ).bind(mesh, geometry)
        face_cells = mesh.face_cells
        boundary_owner = np.asarray(face_cells.owner)[~np.asarray(face_cells.interior)]
        wall_owner = np.asarray(face_cells.owner)[np.asarray(mesh.face_patches.mask("walls"))]
        rings = rings_from_wall(mesh, np.unique(wall_owner))

        def patch_cells(patch: str, mesh=mesh, face_cells=face_cells):
            return jnp.asarray(
                np.unique(np.asarray(face_cells.owner)[np.asarray(mesh.face_patches.mask(patch))])
            )

        corners = jnp.asarray(
            np.flatnonzero(np.bincount(boundary_owner, minlength=mesh.n_cells) >= 2)
        )
        schemes = {
            "corrected GG": CorrectedGreenGauss(),
            "compact GG": CompactGreenGauss(),
            "multcorr first pass": FirstPassOnly(repaired),
            "multcorr repaired": repaired,
            # The full scheme with its Hessian withheld (first pass only) in the wall cells and every
            # ring out to k, in the cells owning two or more boundary faces, or where max|M2^-1| > 10.
            "hessian off corners": FirstPassOnCells(repaired, corners),
            "multcorr M2<=10": FirstPassWhereIllConditioned(repaired, 10.0),
            # Full scheme, but one field's gradient by the first pass alone (see _SplitMomentum).
            **{f"capped kappa={k}": CorrectionCapped(repaired, k) for k in (0.25, 0.5, 1.0, 2.0)},
            "bisect: all first pass": repaired,
            **{f"bisect: full for {c}": repaired for c in CONSUMERS},
            "stabilized damping only": repaired,
            "stabilized diffusion only": repaired,
            "stabilized both": repaired,
            "second pass on velocity only": repaired,
            "second pass on pressure only": repaired,
            # The cells owning a face of the named open boundary patch: the outlet's mass-flux closure and
            # boundary pressure read the FULL pressure gradient there and nowhere else, and on this duct
            # they sit two to three hops from the walls, so the ring arms only withhold them at rings <= 3.
            "first pass on outlet cells": FirstPassOnCells(repaired, patch_cells("outlet")),
            "first pass on inlet+outlet cells": FirstPassOnCells(
                repaired, jnp.union1d(patch_cells("inlet"), patch_cells("outlet"))
            ),
            "hessian off rings<=2 + inlet/outlet": FirstPassOnCells(
                repaired,
                jnp.union1d(
                    jnp.asarray(np.flatnonzero(rings <= 2)),
                    jnp.union1d(patch_cells("inlet"), patch_cells("outlet")),
                ),
            ),
            "hessian off rings<=3 except outlet": FirstPassOnCells(
                repaired,
                jnp.setdiff1d(jnp.asarray(np.flatnonzero(rings <= 3)), patch_cells("outlet")),
            ),
            **{
                f"hessian off rings<={k}": FirstPassOnCells(
                    repaired, jnp.asarray(np.flatnonzero(rings <= k))
                )
                for k in range(4)
            },
        }
        chosen = os.environ.get("TET_ARMS")
        if chosen:
            schemes = {name: schemes[name] for name in chosen.split(",")}
        else:
            schemes = {name: schemes[name] for name in list(schemes)[:4]}
        for name, scheme in schemes.items():

            def build_momentum(gradient_scheme, mesh=mesh, geometry=geometry):
                return MomentumContinuity.build(
                    mesh,
                    geometry,
                    PropertyModel({"viscosity": Constant(MU), "density": Constant(RHO)}),
                    BoundaryConditions(
                        {
                            "inlet": VelocityInlet(velocity=(U_IN, 0.0, 0.0)),
                            "outlet": PressureOutlet(pressure=0.0),
                            "walls": NoSlipWall(),
                        }
                    ),
                    gradient_scheme=gradient_scheme,
                    advection_scheme=FirstOrderUpwind(),
                )

            if name.startswith("bisect:"):
                first = build_momentum(FirstPassOnly(repaired))
                momentum = first
                if name != "bisect: all first pass":
                    momentum = full_gradient_for(
                        first, build_momentum(repaired), name.removeprefix("bisect: full for ")
                    )
            else:
                momentum = build_momentum(scheme)

            def on_step(report, rung="?", tag=f"{label}, {name}"):
                print(
                    f"  [{tag}] {rung:<6} step {report.step:>3d}  |R| {report.residual_norm:.4e}  "
                    f"ratio {report.residual_ratio:.3e}  alpha {report.alpha:>8.4g}  "
                    f"cyc {report.cycles:>3d}  esc {report.escalations}",
                    flush=True,
                )

            if name.startswith("second pass on"):
                withheld = "pressure" if "velocity only" in name else "velocity"
                momentum = split(momentum, withheld)
            if name.startswith("stabilized"):
                momentum = stabilized(
                    momentum,
                    damping=name in ("stabilized damping only", "stabilized both"),
                    diffusion=name in ("stabilized diffusion only", "stabilized both"),
                )
            started = time.perf_counter()
            try:
                flow = solve_flow_march(
                    momentum.with_scaled_molecular_viscosity(RATIO),
                    seed(momentum),
                    preconditioner=preconditioner,
                    dual_time=dual_time,
                    max_steps=MAX_STEPS,
                    convergence=Convergence(rtol=0.01, atol=0.0),
                    retry=RETRY,
                    on_step=lambda r: on_step(r, "anchor"),
                )
                flow = solve_flow_march(
                    momentum,
                    flow,
                    preconditioner=preconditioner,
                    dual_time=dual_time,
                    max_steps=MAX_STEPS,
                    convergence=Convergence(rtol=RTOL, atol=0.0),
                    retry=RETRY,
                    on_step=lambda r: on_step(r, "target"),
                )
                outcome = f"converged, |R| {float(jnp.linalg.norm(momentum.residual(flow))):.2e}"
            except Exception as exc:  # the march's own guard raises on non-convergence
                outcome = f"FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
            print(
                f"RESULT [{label}, {mesh.n_cells} cells] {name:<20} {outcome}  "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
