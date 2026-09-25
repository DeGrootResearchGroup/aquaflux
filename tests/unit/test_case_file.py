"""A case file read into a case, refused where it is wrong, written back unchanged, and checked against its mesh."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from aquaflux.boundary import BoundaryConditions, Dirichlet, ZeroGradient
from aquaflux.case import (
    RANS,
    BodyForce,
    BulkVelocity,
    CaseFile,
    CaseSpec,
    FixedTurbulence,
    Fluid,
    GeometricGrading,
    Inlet,
    IntensityLength,
    Numerics,
    OpenFOAMMesh,
    Outlet,
    StructuredGrid,
    Wall,
    case_spec_from_mapping,
    case_spec_to_mapping,
    read_case,
    write_case,
)
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind
from aquaflux.flow import (
    MassFlow,
    MomentumContinuity,
    MovingWall,
    NoSlipWall,
    PinnedPoint,
    PressureOutlet,
    UniformBodyForce,
    VelocityInlet,
)
from aquaflux.io import read_openfoam
from aquaflux.io.openfoam.cyclic import DEFAULT_MATCH_TOLERANCE
from aquaflux.mesh import Mesh, MeshGeometry, graded_nodes, structured_grid_2d
from aquaflux.properties import Constant, PropertyModel
from aquaflux.schemes import (
    CorrectedGreenGauss,
    HessianCorrectedGradient,
    MultipleCorrectionGradient,
    VenkatakrishnanLimiter,
)
from aquaflux.turbulence import CoupledRANS, LogScalars, SSTModel, SSTTurbulence

REPO = Path(__file__).resolve().parents[2]
#: A one-cell-thick slab between `empty` front and back patches, read as 2D: left, right, bottom, top.
SLAB = REPO / "tests" / "fixtures" / "polymesh_2d_slab_frontandback"
PITZDAILY = REPO / "validation" / "pitzdaily_openfoam" / "case.yaml"

# The default gradient reconstruction warns that the fixture's two cells, each with three boundary
# faces, leave it underdetermined -- true, and beside the point of a test comparing two builds.
_DEFAULT_GRADIENT_ON_TWO_CELLS = pytest.mark.filterwarnings(
    "ignore:MultipleCorrectionGradient.*underdetermined:UserWarning"
)


def _sections(**overrides: object) -> dict[str, object]:
    """A laminar channel on the slab fixture, as a file would state it, with ``overrides`` replacing sections."""
    sections = {
        "mesh": {"kind": "OpenFOAMMesh", "path": str(SLAB)},
        "fluid": {"density": 1.0, "kinematic_viscosity": 1.0e-3},
        "physics": {"kind": "Laminar"},
        "boundaries": {
            "left": {"kind": "Inlet", "velocity": [1.0, 0.0]},
            "right": {"kind": "Outlet", "pressure": 0.0},
            "bottom": {"kind": "Wall"},
            "top": {"kind": "Wall"},
        },
        "numerics": {"momentum_advection": {"kind": "FirstOrderUpwind"}},
    }
    return {**sections, **overrides}


def _boundaries(**overrides: object) -> dict[str, object]:
    return {**_sections()["boundaries"], **overrides}


def _rans_case() -> CaseSpec:
    return CaseSpec(
        mesh=OpenFOAMMesh("runs/kwsst/polyMesh"),
        fluid=Fluid(density=1.0, kinematic_viscosity=1.0e-5),
        physics=RANS(
            advection=FirstOrderUpwind(),
            omega_variable=LogScalars(),
            explicit_production_limiter=True,
        ),
        boundaries={
            "inlet": Inlet((10.0, 0.0), turbulence=FixedTurbulence(k=0.375, omega=440.15)),
            "outlet": Outlet(pressure=0.0),
            "upperWall": Wall(k="zero_gradient"),
            "lowerWall": Wall(k="zero_gradient"),
        },
        numerics=Numerics(
            momentum_advection=LimitedUpwind(limiter=VenkatakrishnanLimiter()),
            gradient=MultipleCorrectionGradient(),
        ),
    )


# --- reading and writing -----------------------------------------------------------------------


def test_the_shipped_pitzdaily_file_reads_as_the_case_its_driver_builds_and_fits_its_mesh() -> None:
    """The one case file in the repository, read, compared against the same case built in code, and checked.

    The comparison is against a value written out independently here, so a file whose settings drift
    from the case (or a reader that misreads one) fails, not only a file that fails to parse.
    """
    case = read_case(PITZDAILY)
    # The solver section is compared against what its script used to pass, in test_case_solver.py.
    assert dataclasses.replace(case.spec, solver=None) == _rans_case()
    assert case.directory == PITZDAILY.parent
    checked = case.check()
    assert (checked.mesh.n_cells, checked.mesh.dim) == (12225, 2)


def test_a_case_is_written_as_plain_data_and_read_back_equal(tmp_path: Path) -> None:
    spec = _rans_case()
    mapping = case_spec_to_mapping(spec)
    assert "kind" not in mapping
    assert "kind" not in mapping["fluid"] and "kind" not in mapping["numerics"]
    assert mapping["boundaries"]["upperWall"] == {"kind": "Wall", "k": "zero_gradient"}
    assert case_spec_from_mapping(mapping) == spec

    path = tmp_path / "case.yaml"
    write_case(spec, path)
    assert read_case(path).spec == spec


def test_a_patch_order_is_kept_through_a_file(tmp_path: Path) -> None:
    spec = _rans_case()
    path = tmp_path / "case.yaml"
    write_case(spec, path)
    assert list(read_case(path).spec.boundaries) == ["inlet", "outlet", "upperWall", "lowerWall"]


def test_a_boundaries_table_given_in_code_is_frozen_and_copied() -> None:
    held = dict(_rans_case().boundaries)
    spec = dataclasses.replace(_rans_case(), boundaries=held)
    held["extra"] = Wall()
    assert "extra" not in spec.boundaries
    with pytest.raises(TypeError):
        spec.boundaries["extra"] = Wall()  # type: ignore[index]


# --- refused when the file is read -------------------------------------------------------------


@pytest.mark.parametrize(
    ("sections", "error", "match"),
    [
        (
            _sections(fluid={"density": 1.0}),
            ValueError,
            "exactly one of kinematic_viscosity and dynamic_viscosity, got neither",
        ),
        (
            _sections(
                fluid={"density": 1.0, "kinematic_viscosity": 1e-3, "dynamic_viscosity": 1e-3}
            ),
            ValueError,
            "exactly one of kinematic_viscosity and dynamic_viscosity, got both",
        ),
        (
            _sections(fluid={"density": 0.0, "dynamic_viscosity": 1e-3}),
            ValueError,
            "density must be a positive, finite number, got 0.0",
        ),
        (
            _sections(fluid={"density": 1.0, "kinematic_viscosity": -1.0}),
            ValueError,
            "kinematic_viscosity must be a positive, finite number",
        ),
        (
            _sections(boundaries=_boundaries(right={"kind": "Outlet"})),
            ValueError,
            r"Outlet at 'boundaries.right' needs 'pressure', which has no default",
        ),
        (
            _sections(
                boundaries=_boundaries(left={"kind": "Inlet", "velocity": [1.0, 0.0, 0.0, 0.0]})
            ),
            ValueError,
            r"Inlet at 'boundaries.left': an inlet velocity has two or three components, got 4",
        ),
        (
            _sections(boundaries=_boundaries(bottom={"kind": "Wal"})),
            ValueError,
            r"unknown kind 'Wal' at 'boundaries.bottom'",
        ),
        (
            _sections(boundaries=_boundaries(bottom={"kind": "Wall", "k": "zero-gradient"})),
            ValueError,
            r"'zero-gradient' at 'boundaries.bottom.k' is not accepted there; Wall.k takes one of "
            r"'zero_gradient', 'zero' or null",
        ),
        (
            _sections(
                boundaries=_boundaries(
                    bottom={"kind": "Outlet", "pressure": 0.0},
                    top={"kind": "FixedTurbulence", "k": 1.0, "omega": 1.0},
                )
            ),
            ValueError,
            r"at 'boundaries.top' is not accepted there; each entry of CaseSpec.boundaries takes one of "
            r"'Inlet', 'Outlet', 'Wall'",
        ),
        (
            _sections(numerics={"gradient": {"kind": "CompactGreenGauss"}}),
            ValueError,
            r"Numerics at 'numerics' needs 'momentum_advection'",
        ),
        (
            _sections(physics={"kind": "RANS"}),
            ValueError,
            r"RANS at 'physics' needs 'advection'",
        ),
        (
            _sections(solvers={"kind": "CoupledMarch"}),
            ValueError,
            r"CaseSpec has no field 'solvers'",
        ),
        ({k: v for k, v in _sections().items() if k != "fluid"}, ValueError, r"needs 'fluid'"),
    ],
    ids=[
        "no-viscosity",
        "both-viscosities",
        "zero-density",
        "negative-viscosity",
        "missing-required-setting",
        "a-value-refusing-itself-is-named-by-path",
        "misspelt-kind",
        "misspelt-choice",
        "a-kind-that-is-not-a-patch-condition",
        "numerics-missing-momentum-advection",
        "rans-missing-its-advection",
        "unknown-section",
        "missing-section",
    ],
)
def test_a_case_that_is_wrong_on_its_own_terms_is_refused_naming_where(
    sections, error, match
) -> None:
    with pytest.raises(error, match=match):
        case_spec_from_mapping(sections)


def test_either_viscosity_is_accepted_on_its_own() -> None:
    for fluid in (
        {"density": 2.0, "kinematic_viscosity": 1e-3},
        {"density": 2.0, "dynamic_viscosity": 2e-3},
    ):
        assert case_spec_from_mapping(_sections(fluid=fluid)).fluid == Fluid(**fluid)


def test_a_laminar_case_refuses_every_turbulence_setting_at_once() -> None:
    """Named by path, all of them -- a laminar case would otherwise ignore them without a word."""
    sections = _sections(
        boundaries=_boundaries(
            left={
                "kind": "Inlet",
                "velocity": [1.0, 0.0],
                "turbulence": {"kind": "FixedTurbulence", "k": 0.1, "omega": 10.0},
            },
            top={"kind": "Wall", "k": "zero"},
        )
    )
    with pytest.raises(
        ValueError, match=r"boundaries.left.turbulence, boundaries.top.k: a laminar case"
    ):
        case_spec_from_mapping(sections)


def test_a_rans_case_refuses_an_inlet_with_no_inflow_turbulence() -> None:
    sections = _sections(physics={"kind": "RANS", "advection": {"kind": "FirstOrderUpwind"}})
    with pytest.raises(ValueError, match=r"boundaries.left.turbulence: a RANS case needs"):
        case_spec_from_mapping(sections)


def test_a_rans_case_with_inflow_turbulence_and_no_wall_setting_loads() -> None:
    sections = _sections(
        physics={"kind": "RANS", "advection": {"kind": "FirstOrderUpwind"}},
        boundaries=_boundaries(
            left={
                "kind": "Inlet",
                "velocity": [1.0, 0.0],
                "turbulence": {"kind": "FixedTurbulence", "k": 0.1, "omega": 10.0},
            }
        ),
    )
    spec = case_spec_from_mapping(sections)
    assert spec.boundaries["left"].turbulence.inflow((1.0, 0.0), SSTModel()) == (0.1, 10.0)
    assert spec.boundaries["top"] == Wall()


# --- the pressure level: fixed exactly once -----------------------------------------------------

#: A lid-driven cavity on the slab fixture: every patch a wall, the top one moving.
_CAVITY_PATCHES = {
    "left": {"kind": "Wall"},
    "right": {"kind": "Wall"},
    "bottom": {"kind": "Wall"},
    "top": {"kind": "Wall", "velocity": [1.0, 0.0]},
}


def test_a_closed_domain_with_no_datum_is_refused_when_read() -> None:
    """Without one its pressure level is free and the system singular -- refused, not left to solve."""
    with pytest.raises(ValueError, match=r"the case: no boundary patch prescribes the pressure"):
        case_spec_from_mapping(_sections(boundaries=_CAVITY_PATCHES))


def test_a_datum_beside_an_outlet_is_refused_when_read() -> None:
    sections = _sections(pressure_datum={"kind": "PinnedPoint", "point": [0.5, 0.5]})
    with pytest.raises(ValueError, match=r"'right' already fixes the pressure level"):
        case_spec_from_mapping(sections)


def test_a_closed_case_with_a_datum_reads_and_round_trips(tmp_path: Path) -> None:
    sections = _sections(
        boundaries=_CAVITY_PATCHES,
        pressure_datum={"kind": "PinnedPoint", "point": [0.5, 0.25], "value": 1.5},
    )
    spec = case_spec_from_mapping(sections)
    assert spec.pressure_datum == PinnedPoint((0.5, 0.25), value=1.5)
    assert spec.boundaries["top"] == Wall(velocity=(1.0, 0.0))
    path = tmp_path / "case.yaml"
    write_case(spec, path)
    assert read_case(path).spec == spec


@_DEFAULT_GRADIENT_ON_TWO_CELLS
def test_a_closed_case_builds_the_pinned_cavity_written_by_hand() -> None:
    """The moving wall, the datum and the resolved pin, against the same cavity assembled by hand."""
    mesh, geometry = _slab()
    reference = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(jnp.asarray(4.0e-3)), "density": Constant(2.0)}),
        BoundaryConditions(
            {
                "left": NoSlipWall(),
                "right": NoSlipWall(),
                "bottom": NoSlipWall(),
                "top": MovingWall(velocity=(1.0, 0.0)),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
        pressure_datum=PinnedPoint((1.2, 0.1), value=1.5),
    )
    sections = _sections(
        fluid=_SLAB_FLUID,
        boundaries=_CAVITY_PATCHES,
        pressure_datum={"kind": "PinnedPoint", "point": [1.2, 0.1], "value": 1.5},
    )
    built = CaseFile(case_spec_from_mapping(sections), REPO).check().build()
    _same_problem(built, reference)
    assert built.pressure_pin == 1  # the slab's second cell, centred at x = 1.5


@pytest.mark.parametrize(
    ("datum", "match"),
    [
        ({"kind": "PinnedPoint", "point": [0.5, 0.5, 0.5]}, r"pressure_datum: the point .* has 3"),
        (
            {"kind": "PinnedPoint", "point": [5.0, 0.5]},
            r"pressure_datum: the point .* lies outside",
        ),
    ],
    ids=["wrong-dimension", "outside-the-mesh"],
)
def test_a_datum_that_does_not_fit_the_mesh_is_refused_when_checked(datum, match) -> None:
    case = CaseFile(
        case_spec_from_mapping(_sections(boundaries=_CAVITY_PATCHES, pressure_datum=datum)), REPO
    )
    with pytest.raises(ValueError, match=match):
        case.check()


def test_a_wall_velocity_of_the_wrong_dimension_is_refused_when_checked() -> None:
    patches = {**_CAVITY_PATCHES, "top": {"kind": "Wall", "velocity": [1.0, 0.0, 0.0]}}
    datum = {"kind": "PinnedPoint", "point": [0.5, 0.5]}
    case = CaseFile(
        case_spec_from_mapping(_sections(boundaries=patches, pressure_datum=datum)), REPO
    )
    with pytest.raises(ValueError, match=r"boundaries.top: the wall velocity .* has 3 components"):
        case.check()


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ([1, 2], "a case is a mapping of its sections"),
        (None, "a case is a mapping of its sections"),
        ({**_sections(), "kind": "Inlet"}, "names no kind, got kind 'Inlet'"),
    ],
    ids=["a-list", "an-empty-file", "a-top-level-kind"],
)
def test_a_document_that_is_not_a_case_is_refused(document, match) -> None:
    with pytest.raises(ValueError, match=match):
        case_spec_from_mapping(document)


def test_a_value_no_case_file_can_name_is_refused_on_writing() -> None:
    spec = dataclasses.replace(
        _rans_case(), numerics=Numerics(FirstOrderUpwind(), gradient=HessianCorrectedGradient())
    )
    with pytest.raises(
        TypeError, match=r"HessianCorrectedGradient at 'numerics.gradient' is not a kind"
    ):
        case_spec_to_mapping(spec)


# --- the YAML parse ----------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


_SLAB_CASE = f"""
mesh: {{kind: OpenFOAMMesh, path: {SLAB}}}
fluid: {{density: 1, kinematic_viscosity: VISCOSITY}}
physics: {{kind: Laminar}}
boundaries:
  left: {{kind: Inlet, velocity: [1, 0]}}
  right: {{kind: Outlet, pressure: PRESSURE}}
  bottom: {{kind: Wall}}
  top: {{kind: Wall}}
numerics:
  momentum_advection: {{kind: FirstOrderUpwind}}
"""


def _slab_file(tmp_path: Path, viscosity: str = "1.0e-3", pressure: str = "0") -> Path:
    text = _SLAB_CASE.replace("VISCOSITY", viscosity).replace("PRESSURE", pressure)
    return _write(tmp_path / "case.yaml", text)


@pytest.mark.parametrize("written", ["1e-5", "1E-5", "1.0e-5", "1.e-5", ".1e-4", "+1e-5"])
def test_an_exponent_is_read_as_a_number_however_it_is_written(
    tmp_path: Path, written: str
) -> None:
    """YAML 1.1 reads ``1e-5`` as a string; a case file reads it as the number it plainly is."""
    spec = read_case(_slab_file(tmp_path, viscosity=written)).spec
    assert spec.fluid.kinematic_viscosity == pytest.approx(1e-5)


def test_a_leading_zero_is_a_decimal_integer_not_octal(tmp_path: Path) -> None:
    assert read_case(_slab_file(tmp_path, pressure="010")).spec.boundaries["right"].pressure == 10


@pytest.mark.parametrize("written", ["no", "off", "yes", "on"])
def test_a_yaml_1_1_boolean_word_is_a_string_and_is_refused_where_a_boolean_belongs(
    tmp_path: Path, written: str
) -> None:
    text = _SLAB_CASE.replace("VISCOSITY", "1.0e-3").replace("PRESSURE", "0")
    text = text.replace(
        "physics: {kind: Laminar}",
        f"physics: {{kind: RANS, advection: {{kind: FirstOrderUpwind}}, explicit_production_limiter: {written}}}",
    )
    with pytest.raises(ValueError, match=rf"'{written}' at 'physics.explicit_production_limiter'"):
        read_case(_write(tmp_path / "case.yaml", text))


def test_a_patch_named_twice_is_refused_rather_than_the_first_being_dropped(tmp_path: Path) -> None:
    text = _SLAB_CASE.replace("VISCOSITY", "1.0e-3").replace("PRESSURE", "0")
    text = text.replace("  top: {kind: Wall}", "  top: {kind: Wall}\n  left: {kind: Wall}")
    with pytest.raises(yaml.YAMLError, match=r"the key 'left' appears twice"):
        read_case(_write(tmp_path / "case.yaml", text))


def test_a_refusal_names_the_file(tmp_path: Path) -> None:
    path = _slab_file(tmp_path, viscosity="-1.0")
    with pytest.raises(ValueError, match=rf"^{path}: .*kinematic_viscosity must be a positive"):
        read_case(path)


# --- checked against the mesh ------------------------------------------------------------------


def test_a_relative_mesh_path_is_taken_from_the_case_file_directory(tmp_path: Path) -> None:
    shutil.copytree(SLAB, tmp_path / "mesh")
    path = _write(tmp_path / "case.yaml", _SLAB_CASE.replace(str(SLAB), "mesh"))
    path = _write(path, path.read_text().replace("VISCOSITY", "1.0e-3").replace("PRESSURE", "0"))
    checked = read_case(path).check()
    assert (checked.mesh.n_cells, checked.mesh.dim) == (2, 2)


def test_the_slab_case_fits_its_mesh() -> None:
    checked = CaseFile(case_spec_from_mapping(_sections()), REPO).check()
    assert checked.mesh.dim == 2
    assert set(checked.spec.boundaries) == {"left", "right", "bottom", "top"}


@pytest.mark.parametrize(
    ("boundaries", "match"),
    [
        (
            {**_boundaries(), "inlet": {"kind": "Wall"}},
            r"the mesh has no patch 'inlet' \(its boundary patches are \['bottom', 'left', 'right', 'top'\]\)",
        ),
        (
            {k: v for k, v in _boundaries().items() if k != "top"},
            r"no condition is given for the boundary faces of 'top' \(2 faces\)",
        ),
        ({**_boundaries(), "interior": {"kind": "Wall"}}, r"'interior' is not a boundary patch"),
        (
            _boundaries(left={"kind": "Inlet", "velocity": [1.0, 0.0, 0.0]}),
            r"boundaries.left: the inlet velocity \(1.0, 0.0, 0.0\) has 3 components, but the mesh is "
            r"2-dimensional",
        ),
    ],
    ids=["unknown-patch", "uncovered-patch", "interior-is-not-a-boundary", "wrong-dimension"],
)
def test_a_case_that_does_not_fit_its_mesh_is_refused_when_checked(boundaries, match) -> None:
    """Reading succeeds -- the case is consistent on its own terms -- and the mesh check refuses it."""
    case = CaseFile(case_spec_from_mapping(_sections(boundaries=boundaries)), REPO)
    with pytest.raises(ValueError, match=match):
        case.check()


def test_every_misfit_is_reported_at_once() -> None:
    boundaries = {k: v for k, v in _boundaries().items() if k != "top"}
    boundaries["lid"] = {"kind": "Wall"}
    boundaries["left"] = {"kind": "Inlet", "velocity": [1.0, 0.0, 0.0]}
    case = CaseFile(case_spec_from_mapping(_sections(boundaries=boundaries)), REPO)
    with pytest.raises(ValueError) as refused:
        case.check()
    message = str(refused.value)
    assert "no patch 'lid'" in message
    assert "'top' (2 faces)" in message
    assert "3 components" in message


def test_the_mesh_tolerance_reaches_the_reader_and_is_left_to_it_when_unset() -> None:
    relative = str(SLAB.relative_to(REPO))
    assert OpenFOAMMesh(relative).reader(REPO).cyclic_match_tolerance == DEFAULT_MATCH_TOLERANCE
    tolerant = OpenFOAMMesh(relative, cyclic_match_tolerance=1e-3).reader(REPO)
    assert tolerant.cyclic_match_tolerance == 1e-3
    assert Path(tolerant.directory) == SLAB


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda: FixedTurbulence(k=-0.1, omega=1.0), ValueError, r"FixedTurbulence.k must be >= 0"),
        (
            lambda: FixedTurbulence(k=0.1, omega=0.0),
            ValueError,
            r"FixedTurbulence.omega must be > 0",
        ),
        (lambda: FixedTurbulence(k=float("nan"), omega=1.0), ValueError, r"must be finite"),
        (lambda: Inlet((1.0, float("inf"))), ValueError, r"Inlet.velocity must be finite"),
        (lambda: Inlet((1.0, 0.0), turbulence=(0.1, 1.0)), TypeError, r"Inlet.turbulence must be"),
        (lambda: Outlet(pressure=float("nan")), ValueError, r"Outlet.pressure must be finite"),
        (lambda: Wall(velocity=(1.0,)), ValueError, r"a wall velocity has two or three components"),
        (lambda: Wall(velocity=(1.0, float("nan"))), ValueError, r"Wall.velocity must be finite"),
        (lambda: OpenFOAMMesh(""), ValueError, r"needs the path"),
        (lambda: OpenFOAMMesh("m", cyclic_match_tolerance=0.0), ValueError, r"positive, finite"),
        (lambda: RANS(advection=None), TypeError, r"RANS.advection must be an AdvectionScheme"),
        (
            lambda: RANS(advection=FirstOrderUpwind(), omega_variable=FirstOrderUpwind()),
            TypeError,
            r"RANS.omega_variable must be a ScalarVariableTransform",
        ),
        (lambda: Numerics(momentum_advection=None), TypeError, r"momentum_advection must be"),
        (
            lambda: Numerics(FirstOrderUpwind(), gradient=FirstOrderUpwind()),
            TypeError,
            r"Numerics.gradient must be a GradientScheme",
        ),
        (
            lambda: dataclasses.replace(_rans_case(), boundaries={}),
            ValueError,
            r"at least one boundary patch",
        ),
        (
            lambda: dataclasses.replace(_rans_case(), boundaries={"inlet": Outlet(0.0), "w": 3}),
            TypeError,
            r"maps each patch name to an Inlet, Outlet or Wall",
        ),
        (lambda: dataclasses.replace(_rans_case(), physics="RANS"), TypeError, r"CaseSpec.physics"),
    ],
    ids=[
        "negative-k",
        "zero-omega",
        "non-finite-k",
        "non-finite-velocity",
        "turbulence-of-the-wrong-family",
        "non-finite-pressure",
        "a-wall-velocity-of-one-component",
        "a-non-finite-wall-velocity",
        "empty-mesh-path",
        "zero-tolerance",
        "rans-without-advection",
        "rans-variable-of-the-wrong-family",
        "numerics-without-advection",
        "gradient-of-the-wrong-family",
        "no-patches",
        "a-patch-that-is-not-a-condition",
        "physics-of-the-wrong-family",
    ],
)
def test_a_value_built_in_code_refuses_what_a_file_could_not_say(build, error, match) -> None:
    """The same refusals hold for a case assembled in code, which no mapping has checked first."""
    with pytest.raises(error, match=match):
        build()


# --- built into the problem it describes -------------------------------------------------------
#
# The reference for each build is the same problem assembled by hand from the library's own builders,
# and the comparison is the strongest one available: one pytree -- the same structure, static fields
# included, and every array leaf bit-for-bit equal. Two problems that pass it evaluate every residual,
# Jacobian and adjoint identically.


def _same_problem(built: object, reference: object) -> None:
    built_leaves, built_def = jax.tree.flatten(built)
    reference_leaves, reference_def = jax.tree.flatten(reference)
    assert built_def == reference_def
    for a, b in zip(built_leaves, reference_leaves, strict=True):
        # An array leaf must be matched by an array leaf: a Python number beside an equal array is a
        # different compiled program (a number is static to a jitted function, so a continuation
        # that rescales it recompiles), however equal their values.
        assert eqx.is_array(a) == eqx.is_array(b), (type(a), type(b))
        if eqx.is_array(a):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        else:
            assert a == b


def _slab() -> tuple[Mesh, MeshGeometry]:
    mesh = read_openfoam(SLAB)
    return mesh, mesh.geometry()


def _hand_built_momentum(mesh, geometry, *, rho=2.0, mu=4.0e-3) -> MomentumContinuity:
    return MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(jnp.asarray(mu)), "density": Constant(rho)}),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(1.0, 0.0)),
                "right": PressureOutlet(pressure=0.5),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        advection_scheme=FirstOrderUpwind(),
    )


_SLAB_FLUID = {"density": 2.0, "kinematic_viscosity": 2.0e-3}  # mu = 4e-3
_SLAB_PATCHES = {
    "left": {"kind": "Inlet", "velocity": [1.0, 0.0]},
    "right": {"kind": "Outlet", "pressure": 0.5},
    "bottom": {"kind": "Wall"},
    "top": {"kind": "Wall"},
}


@_DEFAULT_GRADIENT_ON_TWO_CELLS
def test_a_laminar_case_builds_the_flow_assembler_written_by_hand() -> None:
    mesh, geometry = _slab()
    spec = case_spec_from_mapping(_sections(fluid=_SLAB_FLUID, boundaries=_SLAB_PATCHES))
    built = CaseFile(spec, REPO).check().build()
    assert isinstance(built, MomentumContinuity)
    _same_problem(built, _hand_built_momentum(mesh, geometry))


@_DEFAULT_GRADIENT_ON_TWO_CELLS
def test_a_dynamic_viscosity_builds_the_same_fluid_as_the_kinematic_one_it_equals() -> None:
    mesh, geometry = _slab()
    fluid = {"density": 2.0, "dynamic_viscosity": 4.0e-3}
    spec = case_spec_from_mapping(_sections(fluid=fluid, boundaries=_SLAB_PATCHES))
    _same_problem(CaseFile(spec, REPO).check().build(), _hand_built_momentum(mesh, geometry))


def test_a_rans_case_builds_the_coupled_system_written_by_hand() -> None:
    """Every derived piece at once: the closures per patch, the wall set, one fluid, the settings.

    The walls differ on purpose -- one zero-gradient ``k`` (the default), one zero ``k`` -- so a build
    that applied one wall's setting to every wall, or ignored it, differs from the reference.
    """
    mesh, geometry = _slab()
    gradient = CorrectedGreenGauss()
    momentum = MomentumContinuity.build(
        mesh,
        geometry,
        PropertyModel({"viscosity": Constant(jnp.asarray(4.0e-3)), "density": Constant(2.0)}),
        BoundaryConditions(
            {
                "left": VelocityInlet(velocity=(1.0, 0.0)),
                "right": PressureOutlet(pressure=0.5),
                "bottom": NoSlipWall(),
                "top": NoSlipWall(),
            }
        ),
        gradient_scheme=gradient,
        advection_scheme=LimitedUpwind(limiter=VenkatakrishnanLimiter()),
    )
    model = SSTModel(wall_omega_exponent=3.0)
    turbulence = SSTTurbulence.build(
        model,
        mesh,
        geometry,
        FirstOrderUpwind(),
        momentum.properties,
        wall_patches=["bottom", "top"],
        k_boundary=BoundaryConditions(
            {
                "left": Dirichlet(0.02),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": Dirichlet(0.0),
            }
        ),
        omega_boundary=BoundaryConditions(
            {
                "left": Dirichlet(30.0),
                "right": ZeroGradient(),
                "bottom": ZeroGradient(),
                "top": ZeroGradient(),
            }
        ),
        gradient_scheme=gradient,
        explicit_production_limiter=True,
    )
    reference = CoupledRANS.build(momentum, turbulence, omega_transform=LogScalars())

    sections = _sections(
        fluid=_SLAB_FLUID,
        physics={
            "kind": "RANS",
            "advection": {"kind": "FirstOrderUpwind"},
            "model": {"kind": "SSTModel", "wall_omega_exponent": 3.0},
            "omega_variable": {"kind": "LogScalars"},
            "explicit_production_limiter": True,
        },
        boundaries={
            **_SLAB_PATCHES,
            "left": {
                "kind": "Inlet",
                "velocity": [1.0, 0.0],
                "turbulence": {"kind": "FixedTurbulence", "k": 0.02, "omega": 30.0},
            },
            "top": {"kind": "Wall", "k": "zero"},
        },
        numerics={
            "momentum_advection": {
                "kind": "LimitedUpwind",
                "limiter": {"kind": "VenkatakrishnanLimiter"},
            },
            "gradient": {"kind": "CorrectedGreenGauss"},
        },
    )
    built = CaseFile(case_spec_from_mapping(sections), REPO).check().build()
    assert isinstance(built, CoupledRANS)
    _same_problem(built, reference)
    # Both equations read one fluid, not two equal ones.
    assert built.turbulence.molecular_viscosity[0] == pytest.approx(2.0e-3)


@_DEFAULT_GRADIENT_ON_TWO_CELLS
def test_an_unset_setting_leaves_the_builders_default_in_force() -> None:
    """Nothing is passed for a setting the file leaves out, so the builder's own default applies."""
    mesh, geometry = _slab()
    sections = _sections(
        fluid=_SLAB_FLUID,
        physics={"kind": "RANS", "advection": {"kind": "FirstOrderUpwind"}},
        boundaries={
            **_SLAB_PATCHES,
            "left": {
                "kind": "Inlet",
                "velocity": [1.0, 0.0],
                "turbulence": {"kind": "FixedTurbulence", "k": 0.02, "omega": 30.0},
            },
        },
    )
    built = CaseFile(case_spec_from_mapping(sections), REPO).check().build()
    reference = CoupledRANS.build(
        (momentum := _hand_built_momentum(mesh, geometry)),
        SSTTurbulence.build(
            SSTModel(),
            mesh,
            geometry,
            FirstOrderUpwind(),
            momentum.properties,
            wall_patches=["bottom", "top"],
            k_boundary=BoundaryConditions(
                {
                    "left": Dirichlet(0.02),
                    "right": ZeroGradient(),
                    "bottom": ZeroGradient(),
                    "top": ZeroGradient(),
                }
            ),
            omega_boundary=BoundaryConditions(
                {
                    "left": Dirichlet(30.0),
                    "right": ZeroGradient(),
                    "bottom": ZeroGradient(),
                    "top": ZeroGradient(),
                }
            ),
        ),
    )
    _same_problem(built, reference)


# --- generated meshes, a held bulk velocity, a prescribed force -------------------------------


def _channel_sections(**overrides: object) -> dict[str, object]:
    """A laminar periodic channel, as a file would state it."""
    sections = {
        "mesh": {
            "kind": "StructuredGrid",
            "cells": [4, 6],
            "lengths": [1.0, 2.0],
            "periodic": ["x"],
            "grading": {"y": {"kind": "GeometricGrading", "growth": 1.3}},
        },
        "fluid": {"density": 1.0, "kinematic_viscosity": 0.1},
        "physics": {"kind": "Laminar"},
        "boundaries": {"bottom": {"kind": "Wall"}, "top": {"kind": "Wall"}},
        "numerics": {"momentum_advection": {"kind": "FirstOrderUpwind"}},
        "drive": {"kind": "BulkVelocity", "target": 1.0, "direction": "x", "initial_force": 0.01},
        "pressure_datum": {"kind": "PinnedPoint", "point": [0.0, 0.0]},
    }
    return {**sections, **overrides}


def test_a_structured_grid_generates_the_mesh_its_settings_describe() -> None:
    grid = StructuredGrid(
        cells=(4, 6),
        lengths=(1.0, 2.0),
        periodic=("x",),
        grading={"y": GeometricGrading(growth=1.3)},
    )
    mesh = grid.read(REPO)
    reference = structured_grid_2d(
        4,
        6,
        1.0,
        2.0,
        named_boundaries=True,
        periodic=("x",),
        y_nodes=graded_nodes(6, 2.0, 1.3),
    )
    _same_problem(mesh, reference)
    # A periodic axis's two sides are an interior seam, not patches.
    assert {"bottom", "top"} <= set(mesh.face_patches.names)
    assert not {"left", "right"} & set(mesh.face_patches.names)


def test_a_grading_toward_one_wall_reaches_the_generator() -> None:
    grid = StructuredGrid(
        cells=(2, 5), lengths=(1.0, 1.0), grading={"y": GeometricGrading(1.5, both_sides=False)}
    )
    y = np.unique(np.asarray(grid.read(REPO).node_coords)[:, 1])
    np.testing.assert_allclose(y, graded_nodes(5, 1.0, 1.5, both_sides=False))


def test_a_whole_number_cell_count_written_as_a_float_is_a_count() -> None:
    assert StructuredGrid(cells=(4.0, 6.0), lengths=(1.0, 1.0)).cells == (4, 6)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: StructuredGrid(cells=(2, 2, 2), lengths=(1.0, 1.0, 1.0)), "two-dimensional"),
        (lambda: StructuredGrid(cells=(0, 2), lengths=(1.0, 1.0)), "cell counts must be >= 1"),
        (lambda: StructuredGrid(cells=(2, 2), lengths=(1.0, -1.0)), "lengths must be positive"),
        (
            lambda: StructuredGrid(cells=(1, 2), lengths=(1.0, 1.0), periodic=("x",)),
            "needs at least two cells",
        ),
        (
            lambda: StructuredGrid(cells=(2, 2), lengths=(1.0, 1.0), periodic=("x", "x")),
            "named twice as periodic",
        ),
        (
            lambda: StructuredGrid(
                cells=(2, 2), lengths=(1.0, 1.0), grading={"z": GeometricGrading(1.1)}
            ),
            r"grading names axes \('x', 'y'\), got \['z'\]",
        ),
        (lambda: GeometricGrading(growth=0.0), "growth must be a positive"),
        (lambda: BulkVelocity(target=float("nan")), "BulkVelocity.target must be finite"),
        (lambda: BodyForce(force=(1.0,)), "two or three components, got 1"),
    ],
    ids=[
        "three-dimensional",
        "no-cells",
        "negative-length",
        "one-periodic-cell",
        "periodic-twice",
        "grading-an-axis-it-lacks",
        "zero-growth",
        "non-finite-target",
        "one-component-force",
    ],
)
def test_a_generated_mesh_or_forcing_that_cannot_exist_is_refused(build, match) -> None:
    with pytest.raises(ValueError, match=match):
        build()


def test_a_bulk_velocity_builds_the_mass_flow_drive_it_describes() -> None:
    drive = BulkVelocity(target=2.5, direction="y", initial_force=0.3).drive()
    assert isinstance(drive, MassFlow)
    assert (drive.target, drive.flow_direction, float(drive.force)) == (2.5, 1, 0.3)
    unset = BulkVelocity(target=1.0).drive()
    assert (unset.flow_direction, float(unset.force)) == (0, 0.0)  # MassFlow's own defaults


def test_a_body_force_builds_the_uniform_source_it_describes() -> None:
    source = BodyForce(force=(0.004, -0.5)).momentum_source()
    assert isinstance(source, UniformBodyForce)
    np.testing.assert_array_equal(np.asarray(source.force), [0.004, -0.5])


def test_a_periodic_channel_reads_round_trips_and_builds_as_written_by_hand(tmp_path: Path) -> None:
    sections = _channel_sections(sources=[{"kind": "BodyForce", "force": [0.002, 0.0]}])
    spec = case_spec_from_mapping(sections)
    path = tmp_path / "case.yaml"
    write_case(spec, path)
    assert read_case(path).spec == spec

    mesh = structured_grid_2d(
        4, 6, 1.0, 2.0, named_boundaries=True, periodic=("x",), y_nodes=graded_nodes(6, 2.0, 1.3)
    )
    reference = MomentumContinuity.build(
        mesh,
        mesh.geometry(),
        PropertyModel({"viscosity": Constant(jnp.asarray(0.1)), "density": Constant(1.0)}),
        BoundaryConditions({"bottom": NoSlipWall(), "top": NoSlipWall()}),
        advection_scheme=FirstOrderUpwind(),
        pressure_datum=PinnedPoint((0.0, 0.0)),
        drive=MassFlow(target=1.0, flow_direction=0, force=0.01),
        sources=(UniformBodyForce(jnp.asarray((0.002, 0.0))),),
    )
    _same_problem(CaseFile(spec, tmp_path).check().build(), reference)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"drive": {"kind": "BulkVelocity", "target": 1.0, "direction": "z"}},
            r"drive: the bulk velocity is held along z, but the mesh is 2-dimensional",
        ),
        (
            {"sources": [{"kind": "BodyForce", "force": [1.0, 0.0, 0.0]}]},
            r"sources\[0\]: the body force .* has 3 components",
        ),
    ],
    ids=["a-direction-the-mesh-lacks", "a-force-of-the-wrong-dimension"],
)
def test_forcing_that_does_not_fit_the_mesh_is_refused_when_checked(overrides, match) -> None:
    case = CaseFile(case_spec_from_mapping(_channel_sections(**overrides)), REPO)
    with pytest.raises(ValueError, match=match):
        case.check()


def test_the_default_drive_is_not_a_kind_a_file_names() -> None:
    """Unset means driven by the boundaries and sources; there is no second spelling of that default."""
    with pytest.raises(ValueError, match=r"unknown kind 'BoundaryDriven' at 'drive'"):
        case_spec_from_mapping(_sections(drive={"kind": "BoundaryDriven"}))


# --- inflow turbulence as an intensity and a length scale -----------------------------------------


def test_an_intensity_and_length_scale_give_the_step_cases_inflow() -> None:
    """The two backward-facing steps state 5% intensity with a length scale of 0.1 h and 0.07 H1.

    Their OpenFOAM cases give k = 0.375 and omega = 440.15 (pitzDaily) and ~1600 (bfs3d, whose
    ``0.orig/omega`` writes the same relation out), so those are the answers to reproduce.
    """
    model = SSTModel()
    k, omega = IntensityLength(intensity=0.05, length=0.1 * 0.0254).inflow((10.0, 0.0), model)
    assert k == 0.375
    assert omega == pytest.approx(440.15, rel=1e-4)
    _, omega = IntensityLength(intensity=0.05, length=7e-4).inflow((10.0, 0.0, 0.0), model)
    assert omega == pytest.approx(1600.0, rel=2e-3)


def test_the_intensity_is_of_the_inflow_speed_whatever_its_direction() -> None:
    model = SSTModel()
    along = IntensityLength(intensity=0.1, length=0.01).inflow((5.0, 0.0, 0.0), model)
    oblique = IntensityLength(intensity=0.1, length=0.01).inflow((3.0, 0.0, 4.0), model)
    assert along == pytest.approx(oblique, rel=1e-15)
    assert along[0] == pytest.approx(1.5 * 0.5**2, rel=1e-15)


def test_the_length_scale_is_read_against_the_cases_own_model_constant() -> None:
    turbulence = IntensityLength(intensity=0.05, length=0.01)
    _, default = turbulence.inflow((10.0, 0.0), SSTModel())
    _, stiffer = turbulence.inflow((10.0, 0.0), dataclasses.replace(SSTModel(), beta_star=0.16))
    assert stiffer / default == pytest.approx((0.09 / 0.16) ** 0.25, rel=1e-14)


@pytest.mark.parametrize(
    ("turbulence", "match"),
    [
        (
            {"kind": "IntensityLength", "intensity": 0.0, "length": 0.01},
            r"intensity must be a positive",
        ),
        (
            {"kind": "IntensityLength", "intensity": 0.05, "length": -1.0},
            r"length must be a positive",
        ),
        ({"kind": "IntensityLength", "intensity": 0.05, "length": ".inf"}, r"length"),
        ({"kind": "IntensityLength", "intensity": 0.05}, r"needs 'length'"),
    ],
    ids=["no-intensity", "negative-length", "infinite-length", "no-length"],
)
def test_an_intensity_or_length_that_gives_no_turbulence_is_refused(turbulence, match) -> None:
    if turbulence.get("length") == ".inf":
        turbulence = {**turbulence, "length": float("inf")}
    inlet = {"kind": "Inlet", "velocity": [1.0, 0.0], "turbulence": turbulence}
    rans = {"kind": "RANS", "advection": {"kind": "FirstOrderUpwind"}}
    with pytest.raises(ValueError, match=match):
        case_spec_from_mapping(_sections(physics=rans, boundaries=_boundaries(left=inlet)))


def test_an_intensity_of_a_still_inflow_is_refused() -> None:
    with pytest.raises(ValueError, match=r"this inlet's velocity is zero"):
        IntensityLength(intensity=0.05, length=0.01).inflow((0.0, 0.0), SSTModel())


def test_an_intensity_length_inlet_builds_the_problem_its_k_and_omega_would() -> None:
    """The whole build, against the same case stated with the k and omega the relation gives.

    A non-default model constant, so a build that read its own default instead of the case's model --
    or that fixed the constant at 0.09 -- builds a different omega and fails the comparison.
    """
    model = {"kind": "SSTModel", "beta_star": 0.1}
    rans = {"kind": "RANS", "advection": {"kind": "FirstOrderUpwind"}, "model": model}
    speed, intensity, length = 1.0, 0.04, 0.02
    k = 1.5 * (intensity * speed) ** 2
    omega = k**0.5 / (0.1**0.25 * length)

    def built(turbulence):
        inlet = {"kind": "Inlet", "velocity": [speed, 0.0], "turbulence": turbulence}
        spec = case_spec_from_mapping(
            _sections(physics=rans, fluid=_SLAB_FLUID, boundaries={**_SLAB_PATCHES, "left": inlet})
        )
        return CaseFile(spec, REPO).check().build()

    stated = {"kind": "IntensityLength", "intensity": intensity, "length": length}
    _same_problem(built(stated), built({"kind": "FixedTurbulence", "k": k, "omega": omega}))
