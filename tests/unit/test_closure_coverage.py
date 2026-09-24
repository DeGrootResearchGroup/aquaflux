"""Unit: a boundary closure declares which fields it closes, and the two blocks agree on walls.

Two families of closure meet at an assembler and nothing about their shapes distinguishes them. A
:class:`~aquaflux.boundary.BoundaryCondition` closes one field -- its host equation's, whose name it
does not know -- while a :class:`~aquaflux.flow.FlowBoundary` is a bundle closing the velocity, the
pressure and the mass flux. Handed the wrong one, an assembler used to fail on whichever method the
wrong family lacked (``AttributeError: 'Dirichlet' object has no attribute 'velocity_face'``), which
names an internal method rather than the mistake and never says which patch carries it.

The declaration is also what lets a coupled build reconcile the *other* thing stated twice: which
patches are walls. The flow block says it by giving a patch a closure that shears the flow; the
closure says it by naming the patch in ``wall_patches``, which drives the wall distance and hence the
``omega`` fixation. Neither is derivable from the other while an imported mesh drops the patch type
its file declared, so the pair is checked.
"""

from __future__ import annotations

import aquaflux  # noqa: F401  (enables x64)
import pytest
from aquaflux.boundary import (
    HOST_EQUATION_FIELD,
    BoundaryCondition,
    BoundaryConditions,
    Convective,
    Dirichlet,
    DirichletField,
    Neumann,
    ZeroGradient,
    refuse_a_closure_that_closes_other_fields,
)
from aquaflux.discretization import FirstOrderUpwind
from aquaflux.flow import (
    FlowBoundary,
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
)
from aquaflux.flow.boundary import FLOW_FIELDS
from aquaflux.mesh import structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.turbulence import SSTModel, SSTTurbulence
from aquaflux.turbulence.coupled import CoupledRANS

SCALAR_CLOSURES = (
    Dirichlet(1.0),
    DirichletField(lambda x: x[:, 0]),
    ZeroGradient(),
    Neumann(1.0),
    Convective(h=1.0, t_inf=0.0),
)
FLOW_CLOSURES = (
    NoSlipWall(),
    MovingWall(velocity=(1.0, 0.0)),
    VelocityInlet(velocity=(1.0, 0.0)),
    PressureOutlet(pressure=0.0),
)


def _concrete(base):
    """Every concrete subclass of ``base``, however deeply nested."""
    found, pending = set(), [base]
    while pending:
        for child in type.__subclasses__(pending.pop()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return {kind for kind in found if not getattr(kind, "__abstractmethods__", None)}


# --- the declaration -----------------------------------------------------------------------------


@pytest.mark.parametrize("closure", SCALAR_CLOSURES, ids=lambda c: type(c).__name__)
def test_a_scalar_closure_declares_one_field_it_does_not_name(closure) -> None:
    """Empty, because the field is the assembler's: writing a name here would be a second copy."""
    assert closure.closes() == HOST_EQUATION_FIELD
    assert closure.closes() == ()


@pytest.mark.parametrize("closure", FLOW_CLOSURES, ids=lambda c: type(c).__name__)
def test_a_flow_closure_declares_the_three_it_bundles(closure) -> None:
    assert closure.closes() == ("velocity", "pressure", "mdot")
    assert closure.closes() == FLOW_FIELDS


def test_no_closure_in_the_package_overrides_its_family_s_declaration() -> None:
    """A census over the hierarchy, because the per-class tests above enumerate a literal list.

    Written as "nobody overrides it" rather than "each returns the right tuple" because that is the
    invariant that makes the declaration trustworthy: a closure added later inherits its family's
    answer, and one that overrides it has changed what family it belongs to -- which is a decision to
    make deliberately, not to arrive at. Reading the method off the class also needs no instance, so
    a closure with required constructor arguments cannot quietly fall out of the census.
    """
    for kind in _concrete(BoundaryCondition):
        assert kind.closes is BoundaryCondition.closes, (
            f"{kind.__name__} overrides closes(); a single-field closure declares "
            "HOST_EQUATION_FIELD"
        )
    for kind in _concrete(FlowBoundary):
        assert kind.closes is FlowBoundary.closes, (
            f"{kind.__name__} overrides closes(); a flow bundle declares FLOW_FIELDS"
        )
    # The census is only worth reading if it saw the families at all.
    assert len(_concrete(BoundaryCondition)) >= len(SCALAR_CLOSURES)
    assert len(_concrete(FlowBoundary)) >= len(FLOW_CLOSURES)


# --- the refusal ---------------------------------------------------------------------------------


def test_a_scalar_closure_where_a_flow_bundle_belongs_is_refused_by_patch() -> None:
    """The message names the patch, what it has, and what the equation needs.

    Before this the same mistake raised ``AttributeError: 'Dirichlet' object has no attribute
    'velocity_face'`` from inside the build -- an internal method name, and no patch.
    """
    boundary = BoundaryConditions({"top": Dirichlet(0.0), "bottom": NoSlipWall()})
    with pytest.raises(ValueError) as raised:
        refuse_a_closure_that_closes_other_fields(boundary, FLOW_FIELDS, "Caller.build")
    message = str(raised.value)
    assert "'top'" in message and "Dirichlet" in message
    assert "velocity, pressure and mdot" in message
    assert "'bottom'" not in message  # the correct one is not named


def test_a_flow_bundle_where_a_scalar_closure_belongs_is_refused() -> None:
    boundary = BoundaryConditions({"wall": NoSlipWall()})
    with pytest.raises(ValueError, match="velocity, pressure and mdot"):
        refuse_a_closure_that_closes_other_fields(boundary, HOST_EQUATION_FIELD, "Caller.build")


def test_every_wrong_patch_is_named_not_only_the_first() -> None:
    """Fixing one per run would be the same defect one level up."""
    boundary = BoundaryConditions({"a": Dirichlet(0.0), "b": NoSlipWall(), "c": ZeroGradient()})
    with pytest.raises(ValueError) as raised:
        refuse_a_closure_that_closes_other_fields(boundary, FLOW_FIELDS, "Caller.build")
    assert "'a'" in str(raised.value) and "'c'" in str(raised.value)


def test_something_that_declares_nothing_at_all_is_refused() -> None:
    """The collection is generic over its closure type, so a bare value reaches here and must not pass.

    It is exactly what an assembler must not accept: ``apply`` would fold it in and the residual
    would read whatever it happens to be.
    """
    with pytest.raises(ValueError, match="nothing it declares"):
        refuse_a_closure_that_closes_other_fields(
            BoundaryConditions({"left": 1.0}), HOST_EQUATION_FIELD, "Caller.build"
        )


def test_the_right_family_passes_silently() -> None:
    assert (
        refuse_a_closure_that_closes_other_fields(
            BoundaryConditions({"a": NoSlipWall(), "b": PressureOutlet(pressure=0.0)}),
            FLOW_FIELDS,
            "Caller.build",
        )
        is None
    )
    assert (
        refuse_a_closure_that_closes_other_fields(
            BoundaryConditions({"a": Dirichlet(0.0), "b": ZeroGradient()}),
            HOST_EQUATION_FIELD,
            "Caller.build",
        )
        is None
    )


def test_the_flow_build_refuses_a_scalar_closure() -> None:
    """The real path, not just the helper: the refusal is reached before anything is assembled."""
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    properties = PropertyModel({"viscosity": Constant(0.1), "density": Constant(1.0)})
    boundary = BoundaryConditions(
        {n: (Dirichlet(0.0) if n == "top" else NoSlipWall()) for n in mesh.face_patches.names}
    )
    with pytest.raises(ValueError, match=r"MomentumContinuity.build.*'top'"):
        MomentumContinuity.build(mesh, mesh.geometry(), properties, boundary, pressure_pin=0)


# --- the two blocks agree about which patches are walls -------------------------------------------


def _cavity_blocks(wall_patches, *, lid=True):
    """A small driven cavity's two blocks, with the turbulence closure told ``wall_patches``."""
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    geometry = mesh.geometry()
    properties = PropertyModel({"viscosity": Constant(0.01), "density": Constant(1.0)})
    walls = {n: NoSlipWall() for n in ("bottom", "left", "right")}
    walls["top"] = MovingWall(velocity=(1.0, 0.0)) if lid else NoSlipWall()
    momentum = MomentumContinuity.build(
        mesh, geometry, properties, BoundaryConditions(walls), pressure_pin=0
    )
    every = tuple(mesh.face_patches.names)
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        wall_patches,
        BoundaryConditions({n: Dirichlet(0.0) for n in every if n != "interior"}),
        BoundaryConditions({n: ZeroGradient() for n in every if n != "interior"}),
    )
    return momentum, turbulence


WALLS = ("bottom", "left", "right", "top")


def test_the_two_blocks_agreeing_about_walls_builds() -> None:
    CoupledRANS.build(*_cavity_blocks(WALLS))


def test_a_moving_wall_counts_as_a_wall() -> None:
    """A lid shears the flow, so it is a wall for the closure too -- and omitting it is refused."""
    with pytest.raises(ValueError, match=r"\['top'\].*not in wall_patches"):
        CoupledRANS.build(*_cavity_blocks(("bottom", "left", "right")))


def test_a_wall_the_closure_does_not_list_is_refused_and_named() -> None:
    """It would get no wall distance, so omega is fixed nowhere near it -- and nothing else says so."""
    with pytest.raises(ValueError) as raised:
        CoupledRANS.build(*_cavity_blocks(("bottom", "left", "top"), lid=False))
    assert "'right'" in str(raised.value)
    assert "no wall distance" in str(raised.value)


def test_a_patch_listed_as_a_wall_that_passes_fluid_is_refused_and_named() -> None:
    """The other direction: a spurious zero-distance surface pulls the wall distance down around it."""
    mesh = structured_grid_2d(4, 4, named_boundaries=True)
    geometry = mesh.geometry()
    properties = PropertyModel({"viscosity": Constant(0.01), "density": Constant(1.0)})
    boundary = BoundaryConditions(
        {
            "left": VelocityInlet(velocity=(1.0, 0.0)),
            "right": PressureOutlet(pressure=0.0),
            "bottom": NoSlipWall(),
            "top": NoSlipWall(),
        }
    )
    momentum = MomentumContinuity.build(mesh, geometry, properties, boundary)
    every = tuple(n for n in mesh.face_patches.names if n != "interior")
    turbulence = SSTTurbulence.build(
        SSTModel(),
        mesh,
        geometry,
        FirstOrderUpwind(),
        properties,
        ("bottom", "top", "left"),  # 'left' is the inlet
        BoundaryConditions({n: Dirichlet(0.0) for n in every}),
        BoundaryConditions({n: ZeroGradient() for n in every}),
    )
    with pytest.raises(ValueError) as raised:
        CoupledRANS.build(momentum, turbulence)
    assert "'left'" in str(raised.value)
    assert "passes fluid" in str(raised.value)


def test_the_assembler_keeps_the_patch_names_not_only_their_faces() -> None:
    """Indices cannot be compared against another block's declaration, and do not survive a renumber."""
    _, turbulence = _cavity_blocks(WALLS)
    assert turbulence.wall_patches == WALLS
    assert len(turbulence.wall_faces) > 0
