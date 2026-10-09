"""Each equation's assembler sets a scaled advection scheme for the magnitude of the field it solves.

A softened slope limiter's softening is a fraction of the field's magnitude, which the limiter cannot
know: the momentum assembler supplies the flow's speed, the turbulence closure a ``k`` and an
``omega`` derived from that speed and the walls, and a transported scalar the range of the values its
boundary conditions prescribe. These tests pin what each builder hands its scheme, that a scale stated
on the limiter wins, that a builder whose problem states no magnitude refuses a scheme that needs one
and only such a scheme, and that ``k`` and ``omega`` each read their own.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from aquaflux.boundary import BoundaryConditions, Dirichlet, DirichletField, ZeroGradient
from aquaflux.discretization import AdvectionScheme, FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import (
    MassFlow,
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PinnedPoint,
    PressureOutlet,
    UniformBodyForce,
    VelocityFields,
    VelocityInlet,
    reference_speed,
    wetted_length,
)
from aquaflux.flow.scales import body_force_speed
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, FieldProperty, PropertyModel
from aquaflux.schemes import CompactGreenGauss, VenkatakrishnanLimiter
from aquaflux.transport import ScalarTransport, prescribed_range
from aquaflux.turbulence import SSTModel, SSTTurbulence, turbulence_scales

LIMITED = LimitedUpwind(limiter=VenkatakrishnanLimiter())
FLUID = PropertyModel({"viscosity": Constant(1e-3), "density": Constant(1.0)})


def _flow(boundary, *, scheme=LIMITED, sources=(), drive=None, datum=None, periodic=False):
    mesh = structured_grid_2d(
        8, 4, lx=2.0, ly=1.0, named_boundaries=True, **({"periodic": ("x",)} if periodic else {})
    )
    return MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        FLUID,
        BoundaryConditions(boundary),
        gradient_scheme=CompactGreenGauss(),
        advection_scheme=scheme,
        sources=sources,
        pressure_datum=datum,
        **({} if drive is None else {"drive": drive}),
    )


def _duct(scheme=LIMITED):
    return _flow(
        {
            "left": VelocityInlet((3.0, 4.0)),
            "right": PressureOutlet(0.0),
            "bottom": NoSlipWall(),
            "top": MovingWall((-7.0, 0.0)),
        },
        scheme=scheme,
    )


def _limiter_scale(momentum):
    return float(momentum.advection_scheme.limiter.scale)


# --- the flow's speed -------------------------------------------------------------------------


def test_the_momentum_limiter_takes_the_fastest_prescribed_speed() -> None:
    """An inlet at speed 5 and a lid at speed 7: the scale is the lid's, the faster of the two."""
    momentum = _duct()
    assert reference_speed(momentum) == pytest.approx(7.0)
    assert _limiter_scale(momentum) == pytest.approx(7.0)


def test_a_held_bulk_velocity_is_the_speed_whatever_the_force_starts_at() -> None:
    """A mass-flow channel prescribes no velocity; its target is the speed, not its seed force's."""
    momentum = _flow(
        {"bottom": NoSlipWall(), "top": NoSlipWall()},
        drive=MassFlow(target=-2.5, force=jnp.asarray(1e-6)),
        datum=PinnedPoint((0.0, 0.0)),
        periodic=True,
    )
    assert _limiter_scale(momentum) == pytest.approx(2.5)


def test_a_prescribed_body_force_is_sized_by_its_force_balance() -> None:
    """A body-force channel's scale is the force-balance speed the initializer also starts it at."""
    momentum = _flow(
        {"bottom": NoSlipWall(), "top": NoSlipWall()},
        sources=(UniformBodyForce(jnp.asarray((0.3, 0.0))),),
        datum=PinnedPoint((0.0, 0.0)),
        periodic=True,
    )
    expected = float(body_force_speed(momentum))
    assert expected > 0.0
    assert _limiter_scale(momentum) == pytest.approx(expected)


def test_a_scale_stated_on_the_limiter_is_kept() -> None:
    momentum = _duct(LimitedUpwind(limiter=VenkatakrishnanLimiter(scale=0.25)))
    assert _limiter_scale(momentum) == pytest.approx(0.25)


def test_a_domain_nothing_drives_refuses_a_limiter_that_needs_a_speed() -> None:
    walls = {name: NoSlipWall() for name in ("left", "right", "bottom", "top")}
    with pytest.raises(ValueError, match="nothing in this problem sets one"):
        _flow(walls, datum=PinnedPoint((0.5, 0.5)))


@pytest.mark.parametrize("scheme", [FirstOrderUpwind(), LimitedUpwind(), None])
def test_a_scheme_that_reads_no_scale_needs_no_speed(scheme) -> None:
    """First-order, unlimited and Stokes momentum build in a domain with no speed at all."""
    walls = {name: NoSlipWall() for name in ("left", "right", "bottom", "top")}
    momentum = _flow(walls, scheme=scheme, datum=PinnedPoint((0.5, 0.5)))
    assert momentum.advection_scheme is scheme


# --- k and omega ------------------------------------------------------------------------------


def test_the_turbulence_scales_are_an_intensity_and_an_outer_mixing_length() -> None:
    """``k = 1.5 (0.1 U)^2`` and ``omega = sqrt(k) / (beta_star^(1/4) 0.09 h)``, by hand."""
    scales = turbulence_scales(2.0, 0.5, SSTModel())
    assert scales.k == pytest.approx(1.5 * 0.2**2)
    assert scales.omega == pytest.approx(np.sqrt(0.06) / (0.09**0.25 * 0.09 * 0.5))


@pytest.mark.parametrize(
    ("speed", "length", "reason"),
    [(0.0, 1.0, "this flow has none"), (1.0, 0.0, "no wall to take one from")],
)
def test_the_turbulence_scales_refuse_a_flow_with_no_speed_or_no_wall(
    speed, length, reason
) -> None:
    with pytest.raises(ValueError, match=reason):
        turbulence_scales(speed, length, SSTModel())


def _turbulence(scheme=LIMITED, **options):
    mesh = structured_grid_2d(6, 4, lx=3.0, ly=1.0, named_boundaries=True)
    return SSTTurbulence.build(
        SSTModel(),
        mesh,
        mesh.geometry(),
        scheme,
        FLUID,
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(0.01),
                "right": ZeroGradient(),
                "bottom": Dirichlet(0.0),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(10.0),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
        **options,
    )


def test_k_and_omega_each_take_their_own_scale_from_the_flow_s_speed() -> None:
    turbulence = _turbulence(velocity_scale=2.0)
    length = wetted_length(turbulence.mesh, turbulence.geometry, ["bottom", "top"])
    assert length == pytest.approx(0.5)  # a 3 x 1 channel: V / A_wall = 3 / 6
    expected = turbulence_scales(2.0, length, turbulence.model)
    assert float(turbulence.k_advection_scheme.limiter.scale) == pytest.approx(expected.k)
    assert float(turbulence.omega_advection_scheme.limiter.scale) == pytest.approx(expected.omega)


def test_a_limited_turbulence_advection_needs_the_flow_s_speed() -> None:
    with pytest.raises(ValueError, match="pass velocity_scale"):
        _turbulence()


def test_a_first_order_turbulence_advection_needs_no_speed() -> None:
    turbulence = _turbulence(FirstOrderUpwind())
    assert isinstance(turbulence.k_advection_scheme, FirstOrderUpwind)
    assert isinstance(turbulence.omega_advection_scheme, FirstOrderUpwind)


class _ConstantFace(AdvectionScheme):
    """A stand-in reconstruction: every face carries ``value``, whatever the field."""

    value: float

    def face_value(self, field, context, mass_flux):
        return jnp.full(context.mesh.face_cells.owner.shape, self.value)


def test_each_turbulence_equation_advects_with_its_own_scheme() -> None:
    """Swapping one field's scheme moves that equation's residual and leaves the other's alone.

    Storing two schemes is half of it; a residual reading the other field's scheme (or the two
    reading one) would pass every check of what is stored.
    """
    turbulence = _turbulence(velocity_scale=2.0)
    mesh = turbulence.mesh
    centroid = turbulence.geometry.cell.centroid
    k = 0.02 + 0.01 * centroid[:, 0]
    omega = 80.0 - 5.0 * centroid[:, 0]
    closure = turbulence.closure_fields(
        VelocityFields(
            velocity=jnp.zeros((mesh.n_cells, 2)),
            boundary_velocity=jnp.zeros((mesh.n_faces, 2)),
            gradient=jnp.zeros((mesh.n_cells, 2, 2)),
        ),
        k,
        omega,
    )
    # A uniform rightward volume flux, so advection reaches every row.
    mdot = turbulence.geometry.face.area * turbulence.geometry.face.normal[:, 0]

    def equations(model):
        return model.k_residual(mdot, closure)(k), model.omega_residual(mdot, closure)(omega)

    base_k, base_omega = equations(turbulence)
    stub = _ConstantFace(3.0)
    swapped_k, omega_beside = equations(
        eqx.tree_at(lambda m: m.k_advection_scheme, turbulence, stub)
    )
    k_beside, swapped_omega = equations(
        eqx.tree_at(lambda m: m.omega_advection_scheme, turbulence, stub)
    )
    assert float(jnp.max(jnp.abs(swapped_k - base_k))) > 1e-6
    assert float(jnp.max(jnp.abs(swapped_omega - base_omega))) > 1e-6
    np.testing.assert_allclose(np.asarray(k_beside), np.asarray(base_k), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(omega_beside), np.asarray(base_omega), rtol=1e-13)


# --- a transported scalar -----------------------------------------------------------------------


def _scalar(boundary, scheme=LIMITED):
    mesh = structured_grid_2d(4, 3, named_boundaries=True)
    return ScalarTransport.build(
        mesh,
        mesh.geometry(),
        FieldProperty(values=jnp.full(mesh.n_cells, 1e-3)),
        BoundaryConditions(boundary),
        scheme,
    )


def test_a_scalar_s_scale_is_the_range_its_boundaries_prescribe() -> None:
    """Prescribed at 290 and 300 (a temperature in kelvin): the scale is the 10 K between them."""
    transport = _scalar(
        {
            "left": Dirichlet(290.0),
            "right": Dirichlet(300.0),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    assert float(transport.advection_scheme.limiter.scale) == pytest.approx(10.0)


def test_a_position_dependent_value_is_read_at_the_face_centroids() -> None:
    """An injector from 0 to 2 across the inlet: its own range, read where it is imposed."""

    def injector(x):
        return 2.0 * x[..., 1]

    transport = _scalar(
        {
            "left": DirichletField(field_fn=injector),
            "right": ZeroGradient(),
            "bottom": ZeroGradient(),
            "top": ZeroGradient(),
        }
    )
    centroids = transport.geometry.face.centroid[transport.boundary.faces["left"]]
    expected = float(jnp.max(injector(centroids)) - jnp.min(injector(centroids)))
    assert 1.0 < expected < 2.0
    assert float(transport.advection_scheme.limiter.scale) == pytest.approx(expected)


def test_a_single_prescribed_level_is_its_own_scale() -> None:
    """One inlet concentration and nothing else prescribed: the level is the magnitude."""
    boundary = {
        "left": Dirichlet(-0.4),
        "right": ZeroGradient(),
        "bottom": ZeroGradient(),
        "top": ZeroGradient(),
    }
    transport = _scalar(boundary)
    assert prescribed_range(transport.boundary, transport.geometry) == pytest.approx(0.4)
    assert float(transport.advection_scheme.limiter.scale) == pytest.approx(0.4)


def test_a_scalar_with_nothing_prescribed_refuses_a_scaled_scheme_but_not_others() -> None:
    boundary = {name: ZeroGradient() for name in ("left", "right", "bottom", "top")}
    with pytest.raises(ValueError, match="none prescribes a non-zero value"):
        _scalar(boundary)
    assert isinstance(_scalar(boundary, FirstOrderUpwind()).advection_scheme, FirstOrderUpwind)
