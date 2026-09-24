"""A case file read into a case, refused where it is wrong, written back unchanged, and checked against its mesh."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest
import yaml
from aquaflux.case import (
    RANS,
    CaseFile,
    CaseSpec,
    FixedTurbulence,
    Fluid,
    Inlet,
    Numerics,
    OpenFOAMMesh,
    Outlet,
    Wall,
    case_spec_from_mapping,
    case_spec_to_mapping,
    read_case,
    write_case,
)
from aquaflux.discretization import FirstOrderUpwind, LimitedUpwind
from aquaflux.io.openfoam.cyclic import DEFAULT_MATCH_TOLERANCE
from aquaflux.schemes import (
    HessianCorrectedGradient,
    MultipleCorrectionGradient,
    VenkatakrishnanLimiter,
)
from aquaflux.turbulence import LogScalars

REPO = Path(__file__).resolve().parents[2]
#: A one-cell-thick slab between `empty` front and back patches, read as 2D: left, right, bottom, top.
SLAB = REPO / "tests" / "fixtures" / "polymesh_2d_slab_frontandback"
PITZDAILY = REPO / "validation" / "pitzdaily_openfoam" / "case.yaml"


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
    assert case.spec == _rans_case()
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
        (_sections(solver={"kind": "Anything"}), ValueError, r"CaseSpec has no field 'solver'"),
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
    assert spec.boundaries["left"].turbulence.inflow((1.0, 0.0)) == (0.1, 10.0)
    assert spec.boundaries["top"] == Wall()


def test_a_domain_with_no_outlet_is_refused_for_want_of_a_pressure_datum() -> None:
    """A closed domain would solve a singular system: its pressure level is free and nothing fixes it."""
    sections = _sections(boundaries=_boundaries(right={"kind": "Wall"}))
    with pytest.raises(ValueError, match=r"no boundary patch fixes the pressure"):
        case_spec_from_mapping(sections)


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
