"""Is the k-production cap ACTIVE at a converged root, and does freezing it corrupt the adjoint?

``KProduction`` caps production at the destruction scale, ``P_k = min(nu_t S^2, 10 beta* k omega)``.
The cap itself is a model term. ``explicit_limiter`` is a separate and much less innocent thing: it
wraps the cap's ``k`` in ``stop_gradient``, so the Jacobian omits the term where the cap is active.

That is a **forward-solve** device, and it is only free if the cap is **inactive at the converged
state** -- exactly the discipline the positivity floors are held to. Where the cap binds at the root,
the implicit-function-theorem adjoint linearizes a residual different from the one solved: the
converged fields are unchanged and the *sensitivity* is wrong, silently, with a perfectly finite
gradient coming back.

``KProduction``'s own docstring states the intended contract -- "a forward-solve device only: the
default (``False``) is the exact operator, which the coupled sensitivity residual uses so the adjoint
stays exact". This measures whether that contract holds as shipped.

Reports, at the converged root of a turbulent channel:

1. **cap activity** -- the fraction of cells where ``nu_t S^2 > 10 beta* k omega``, i.e. where the
   ``stop_gradient`` actually removes a Jacobian term. Zero would make the whole question moot.
2. **forward equivalence** -- whether the solve converges with the limiter OFF, and to the same root.
   If the forward path does not need the stabilization, the exact operator is free.
3. **adjoint error** -- ``jax.grad`` of an objective through the converged solve, limiter ON vs OFF,
   against a central finite difference. The finite difference is the arbiter: it knows nothing about
   either linearization.

Run: ``python3 validation/production_cap_activity.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run directly (`python3 validation/production_cap_activity.py`): Python puts THIS file's
# directory on the path, not the repository root, so put the root there ourselves -- the same
# thing the case harnesses under `validation/*/` do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aquaflux  # noqa: F401, E402  (enables x64)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.mesh import graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    CoupledRANS,
    SSTModel,
    SSTTurbulence,
    coupled_continuation,
    hybrid_initialize,
    inlet_k,
    inlet_omega,
    solve_coupled,
)

RHO, U_IN, H, L = 1.0, 1.0, 1.0, 4.0
NU = 4e-4  # Re = U H / nu = 2500 -- genuinely turbulent, so k stays clear of its floor
INTENSITY, LENGTH_SCALE = 0.05, 0.07 * H
PRECONDITIONER = {"velocity": "convection"}


def build_case(nu=NU, *, explicit_limiter: bool):
    """The graded turbulent channel, with the production limiter set either way."""
    y_nodes = graded_nodes(14, H, 1.2)
    mesh = structured_grid_2d(20, 14, lx=L, ly=H, named_boundaries=True, y_nodes=y_nodes)
    geometry = mesh.geometry()
    model = SSTModel()
    k_in = float(inlet_k(jnp.array(U_IN), INTENSITY))
    omega_in = float(inlet_omega(jnp.array(k_in), LENGTH_SCALE, model))
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * nu), "density": Constant(RHO)}),
        CompactGreenGauss(),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(U_IN, 0.0)),
                "right": PressureOutlet(pressure=0.0),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        CompactGreenGauss(),
        FirstOrderUpwind(),
        density=RHO,
        molecular_viscosity=jnp.full(mesh.n_cells, nu),
        wall_patches=["bottom", "top"],
        explicit_production_limiter=explicit_limiter,
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(k_in),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(omega_in),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return CoupledRANS.build(momentum, turbulence)


def cap_activity(coupled, flow, k, omega):
    """Where the cap binds at this state: ``nu_t S^2`` against the limit ``10 beta* k omega``.

    Reported as the fraction of cells over the cap, and how far over -- a cap that binds in a handful
    of cells by a hair is a different proposition from one that binds broadly.
    """
    closure = coupled.turbulence.closure_fields(coupled.momentum.velocity_fields(flow), k, omega)
    production = closure.nu_t * closure.strain_rate**2
    beta_star = coupled.turbulence.model.beta_star
    limit = 10.0 * beta_star * k * omega
    active = production > limit
    frac = float(jnp.mean(active))
    over = production / jnp.maximum(limit, 1e-300)
    # `S / omega` is the scale-free form of the same question, and the one that transfers between
    # cases. On the unlimited eddy-viscosity branch `nu_t = k / omega`, so
    #     production / limit = (k S^2 / omega) / (10 beta* k omega) = S^2 / (10 beta* omega^2),
    # i.e. the cap binds at `S / omega > sqrt(10 beta*) = 0.949` (beta* = 0.09) -- independent of k.
    # An equilibrium boundary layer sits at `S / omega ~ sqrt(beta*) = 0.3`, so the cap needs roughly
    # THREE TIMES the equilibrium shear-to-dissipation ratio before it binds at all. (Where Menter's
    # shear limiter is itself active, `nu_t = a1 k / (S F2)`, the threshold rises further, to
    # `S / omega > 10 beta* F2 / a1 ~ 2.9 F2`.) That is why a well-behaved attached flow never trips
    # it, and why a strongly non-equilibrium region -- impingement, a separating shear layer -- is
    # where to look.
    s_over_omega = closure.strain_rate / jnp.maximum(omega, 1e-300)
    threshold = float(jnp.sqrt(10.0 * beta_star))
    return {
        "fraction_active": frac,
        "cells_active": int(jnp.sum(active)),
        "n_cells": int(k.size),
        "max_production_over_limit": float(jnp.max(over)),
        "median_over_where_active": float(jnp.median(jnp.where(active, over, jnp.nan)))
        if frac > 0
        else 0.0,
        "max_s_over_omega": float(jnp.max(s_over_omega)),
        "p99_s_over_omega": float(jnp.percentile(s_over_omega, 99)),
        "binding_threshold_s_over_omega": threshold,
        "headroom": threshold / max(float(jnp.max(s_over_omega)), 1e-300),
    }


def solve(nu, *, explicit_limiter, state=None):
    """Converge the coupled channel; returns the physical fields."""
    coupled = build_case(nu, explicit_limiter=explicit_limiter)
    f0, k0, o0 = hybrid_initialize(coupled.momentum, coupled.turbulence) if state is None else state
    return coupled, solve_coupled(
        coupled, f0, k0, o0, max_steps=60, rtol=1e-10, atol=1e-12, **PRECONDITIONER
    )


def objective(nu, *, explicit_limiter, seed):
    """A scalar through the converged solve: mean turbulent kinetic energy.

    The continuation is built OUTSIDE the differentiated call, on concrete parameters, as the coupled
    adjoint requires.
    """
    coupled0 = build_case(NU, explicit_limiter=explicit_limiter)
    continuation = coupled_continuation(coupled0, coupled0.state_from_physical(*seed), **PRECONDITIONER)

    def scalar(viscosity):
        coupled = build_case(viscosity, explicit_limiter=explicit_limiter)
        _, k, _ = solve_coupled(
            coupled, *seed, continuation=continuation, max_steps=60, rtol=1e-10, atol=1e-12
        )
        return jnp.mean(k)

    return scalar


def main() -> None:
    print("=" * 78)
    print("k-production cap: activity at the root, and the adjoint cost of freezing it")
    print("=" * 78)

    print("\n[1] converging with the limiter ON (the shipped default) ...", flush=True)
    coupled_on, (f_on, k_on, o_on) = solve(NU, explicit_limiter=True)
    act = cap_activity(coupled_on, f_on, k_on, o_on)
    print(f"    cap ACTIVE in {act['cells_active']}/{act['n_cells']} cells "
          f"({100 * act['fraction_active']:.1f}%)")
    if act["cells_active"]:
        print(f"    production/limit: max {act['max_production_over_limit']:.3g}, "
              f"median where active {act['median_over_where_active']:.3g}")
        print("    -> the stop_gradient removes a Jacobian term in exactly those cells")
    else:
        print("    -> the stop_gradient removes NOTHING here: the cap binds nowhere at the root")
    print(f"    S/omega: max {act['max_s_over_omega']:.3f}, p99 {act['p99_s_over_omega']:.3f} "
          f"| binds above {act['binding_threshold_s_over_omega']:.3f} "
          f"({act['headroom']:.1f}x headroom)")

    print("\n[2] converging with the limiter OFF (the exact operator) ...", flush=True)
    coupled_off, (f_off, k_off, o_off) = solve(NU, explicit_limiter=False)
    same = max(
        float(jnp.max(jnp.abs(f_on - f_off))),
        float(jnp.max(jnp.abs(k_on - k_off))),
        float(jnp.max(jnp.abs(o_on - o_off))),
    )
    print(f"    converged; max |field difference| vs the limited solve = {same:.3e}")
    print("    -> the forward path does not need the stabilization on this case"
          if same < 1e-6 else "    -> the two solves reach DIFFERENT roots")

    print("\n[3] gradient through the converged solve, against central finite differences ...",
          flush=True)
    seed = hybrid_initialize(coupled_on.momentum, coupled_on.turbulence)
    h = NU * 1e-3
    exact_fn = objective(NU, explicit_limiter=False, seed=seed)
    fd = (float(exact_fn(NU + h)) - float(exact_fn(NU - h))) / (2 * h)
    print(f"    central finite difference      : {fd:+.8e}")
    for label, flag in (("limiter OFF (exact operator)", False), ("limiter ON  (shipped default)", True)):
        g = float(jax.grad(objective(NU, explicit_limiter=flag, seed=seed))(NU))
        rel = abs(g - fd) / max(abs(fd), 1e-300)
        print(f"    jax.grad, {label}: {g:+.8e}   rel. error vs FD = {rel:.3e}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
