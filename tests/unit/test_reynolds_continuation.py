"""Unit tests for Reynolds-number continuation (schedule + the viscosity rescale, no solve).

The end-to-end "reaches the same root" and "gradient matches a direct solve" gates live in
``tests/integration/test_reynolds_continuation.py`` -- these cover the pure pieces: the geometric
schedule (physics-free) and the molecular-viscosity rescale on a small built coupled system.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import MomentumContinuity, NoSlipWall, PressureOutlet, VelocityInlet
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import CompactGreenGauss
from aquaflux.turbulence import (
    CoupledRANS,
    GeometricReynoldsSchedule,
    SSTModel,
    SSTTurbulence,
    solve_reynolds_continuation,
)

RHO, NU, U_IN = 1.0, 1e-3, 1.0


# --- GeometricReynoldsSchedule (pure, no mesh) ------------------------------------------


def test_schedule_zero_points_is_the_target_alone() -> None:
    assert GeometricReynoldsSchedule().scales(0) == (1.0,)


def test_schedule_default_decade_per_step() -> None:
    assert GeometricReynoldsSchedule().scales(1) == (10.0, 1.0)
    assert GeometricReynoldsSchedule().scales(2) == (100.0, 10.0, 1.0)
    assert GeometricReynoldsSchedule().scales(3) == (1000.0, 100.0, 10.0, 1.0)


def test_schedule_descends_to_one_and_has_n_plus_one_points() -> None:
    scales = GeometricReynoldsSchedule().scales(4)
    assert len(scales) == 5
    assert scales[-1] == 1.0  # dissolves at the target
    assert list(scales) == sorted(scales, reverse=True)  # strictly descending anchor -> target


def test_schedule_ratio_is_configurable() -> None:
    assert GeometricReynoldsSchedule(ratio=4.0).scales(2) == (16.0, 4.0, 1.0)


def test_negative_points_raise() -> None:
    coupled = _tiny_coupled()
    with pytest.raises(ValueError, match="n_points must be >= 0"):
        solve_reynolds_continuation(coupled, -1)


# --- the molecular-viscosity rescale (small built coupled, no solve) --------------------


def _tiny_coupled(nx: int = 4, ny: int = 3) -> CoupledRANS:
    mesh = structured_grid_2d(nx, ny, lx=2.0, ly=1.0, named_boundaries=True)
    geometry = mesh.geometry()
    model = SSTModel()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(RHO * NU), "density": Constant(RHO)}),
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
        molecular_viscosity=jnp.full(mesh.n_cells, NU),
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(0.1),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(100.0),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
    )
    return CoupledRANS.build(momentum, turbulence)


def _dynamic_viscosity(coupled: CoupledRANS) -> float:
    return coupled.momentum.properties.properties["viscosity"].value


def test_coupled_rescale_scales_both_viscosity_leaves() -> None:
    coupled = _tiny_coupled()
    scaled = coupled.with_scaled_molecular_viscosity(10.0)
    # Momentum dynamic viscosity mu = rho * nu: scaled by the factor.
    assert _dynamic_viscosity(scaled) == 10.0 * (RHO * NU)
    # Turbulence kinematic viscosity nu: the whole per-cell field scaled by the factor.
    np.testing.assert_allclose(
        np.asarray(scaled.turbulence.molecular_viscosity),
        np.asarray(coupled.turbulence.molecular_viscosity) * 10.0,
    )


def test_coupled_rescale_leaves_density_untouched() -> None:
    coupled = _tiny_coupled()
    scaled = coupled.with_scaled_molecular_viscosity(10.0)
    assert scaled.momentum.properties.properties["density"].value == RHO
    assert float(scaled.turbulence.density) == RHO


def test_coupled_rescale_is_immutable() -> None:
    coupled = _tiny_coupled()
    coupled.with_scaled_molecular_viscosity(10.0)
    assert _dynamic_viscosity(coupled) == RHO * NU  # original unchanged


def test_coupled_rescale_preserves_the_scalar_transforms() -> None:
    """The rescale carries the omega log-transform (and everything else) through unchanged."""
    from aquaflux.turbulence import LogScalars

    momentum = _tiny_coupled().momentum
    turbulence = _tiny_coupled().turbulence
    coupled = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())
    scaled = coupled.with_scaled_molecular_viscosity(5.0)
    assert isinstance(scaled.omega_transform, LogScalars)


# --- the ramp structure: seeds, viscosity scales, and the intermediate tolerance ---------
#
# These stub out solve_coupled (via the name the wrapper calls) to record each per-Re solve's inputs,
# so the loop's structure is verified without any actual Newton solve.


def _record_solves(monkeypatch):
    """Patch the wrapper's ``solve_coupled`` to record ``(scale, seed_is_none, rtol)`` per call."""
    import aquaflux.turbulence.reynolds as reynolds

    calls = []
    result = (jnp.zeros(3), jnp.ones(1), jnp.ones(1))  # a stand-in converged (flow, k, omega)

    def fake_solve_coupled(coupled, flow=None, k=None, omega=None, **kwargs):
        # The momentum dynamic viscosity encodes the scale (mu = factor * RHO * NU).
        scale = float(coupled.momentum.properties.properties["viscosity"].value / (RHO * NU))
        calls.append({"scale": scale, "seed_is_none": flow is None, "rtol": kwargs.get("rtol")})
        return result

    monkeypatch.setattr(reynolds, "solve_coupled", fake_solve_coupled)
    return calls


def test_ramp_visits_every_scale_and_threads_seeds(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10)
    # Three solves at the schedule's scales, descending to the target (1.0).
    assert [round(c["scale"], 6) for c in calls] == [100.0, 10.0, 1.0]
    # The first point self-starts (no seed); every later point is warm-started.
    assert [c["seed_is_none"] for c in calls] == [True, False, False]


def test_intermediate_points_use_the_loose_tolerance_and_target_uses_rtol(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10, intermediate_rtol=1e-2)
    # Lower-Re points converge loosely; the target keeps the caller's tight rtol.
    assert [c["rtol"] for c in calls] == [1e-2, 1e-2, 1e-10]


def test_intermediate_rtol_none_converges_every_point_to_rtol(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10, intermediate_rtol=None)
    assert [c["rtol"] for c in calls] == [1e-10, 1e-10, 1e-10]


def test_zero_points_calls_solve_once_at_the_target(monkeypatch) -> None:
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=0, rtol=1e-8)
    assert len(calls) == 1
    assert calls[0]["scale"] == 1.0 and calls[0]["seed_is_none"] and calls[0]["rtol"] == 1e-8


def test_point_setup_builds_per_point_kwargs_and_materializes_the_first_seed(monkeypatch) -> None:
    """``point_setup`` is called for every Reynolds point with that point's companion; its returned
    kwargs are merged into that point's solve; and the lowest point's seed is materialized (so a
    per-point continuation can freeze at the same state the solve starts from).
    """
    import aquaflux.turbulence.reynolds as reynolds

    coupled = _tiny_coupled()
    n = coupled.momentum.mesh.n_cells
    dim = coupled.layout.dim
    # A correctly-shaped stand-in converged state, so each point's seed packs into the next cleanly.
    fields = (jnp.zeros((dim + 1) * n), jnp.full(n, 0.5), jnp.full(n, 100.0))

    calls = []

    def fake_solve_coupled(c, flow=None, k=None, omega=None, **kwargs):
        scale = float(c.momentum.properties.properties["viscosity"].value / (RHO * NU))
        calls.append({"scale": scale, "seed_is_none": flow is None, "tag": kwargs.get("tag")})
        return fields

    monkeypatch.setattr(reynolds, "solve_coupled", fake_solve_coupled)
    # Stub the hybrid start so the test stays structural (no real Laplace solve).
    monkeypatch.setattr(reynolds, "hybrid_initialize", lambda momentum, turbulence: fields)

    setups = []

    def point_setup(companion, state):
        scale = float(companion.momentum.properties.properties["viscosity"].value / (RHO * NU))
        setups.append(scale)
        return {"tag": scale}  # a marker kwarg proving the merge reaches solve_coupled

    solve_reynolds_continuation(coupled, n_points=2, rtol=1e-10, point_setup=point_setup)

    # Called once per point (lower-Re and target), at each companion's viscosity scale...
    assert setups == [100.0, 10.0, 1.0]
    # ...its kwargs are merged into every point's solve...
    assert [c["tag"] for c in calls] == [100.0, 10.0, 1.0]
    # ...and the lowest point is now warm-started from the materialized seed too (not solve_coupled's
    # internal hybrid start), so the built continuation and the solve agree on the starting state.
    assert [c["seed_is_none"] for c in calls] == [False, False, False]


def test_point_setup_none_is_byte_identical_to_the_plain_ramp(monkeypatch) -> None:
    """Default (``point_setup=None``): the lowest point self-starts inside solve_coupled and no
    per-point kwargs are added -- the ramp is exactly the pre-existing one."""
    calls = _record_solves(monkeypatch)
    solve_reynolds_continuation(_tiny_coupled(), n_points=2, rtol=1e-10)
    assert [c["seed_is_none"] for c in calls] == [True, False, False]  # first point self-starts
