"""Tests for the case-file form: rows built from the solver's own schema, and the edits they make.

The schema here is the real one (``aquaflux.case.case_schema``), so these also pin that the form can
show every case file the repository holds, and that what an edit produces is a case the solver reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyvista")

from aquaflux.case import case_schema, case_spec_from_mapping, read_case_document
from aquaflux_ui import case_form
from aquaflux_ui.case_form import CaseSchema, form_sections, parse_input

REPO = Path(__file__).resolve().parents[2]
CASE_FILES = sorted(REPO.glob("validation/*/case.yaml")) + sorted(
    REPO.glob("validation/*/cases/*.yaml")
)
PITZDAILY = REPO / "validation" / "pitzdaily_openfoam" / "case.yaml"
SCHEMA = CaseSchema(case_schema())


def _rows(document):
    return {row.path: row for section in form_sections(SCHEMA, document) for row in section["rows"]}


def _leaves(value, path=()):
    """Every plain setting in a document, by path -- not kinds, which rows show as their own value."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key != "kind":
                yield from _leaves(item, (*path, key))
    elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path, value


def _boundary_kinds():
    """The kinds a ``boundaries`` entry may be, read straight from the schema the form is built from."""
    (boundaries,) = [f for f in SCHEMA.fields("CaseSpec") if f["name"] == "boundaries"]
    return boundaries["accepts"][0]["of"][0]["kinds"]


@pytest.mark.parametrize("case_file", CASE_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_every_case_file_in_the_repository_is_shown_whole(case_file):
    document = read_case_document(case_file)
    rows = _rows(document)
    assert not [row for row in rows.values() if row.unknown]
    for path, value in _leaves(document):
        assert path in rows, f"{path} has no row"
        row = rows[path]
        assert row.is_set
        if row.widget == "choice":
            assert value in [item["value"] for item in row.items]
        if row.widget == "choices":
            assert row.value == value and set(value) <= {item["value"] for item in row.items}
    # Every nested value's kind is one its row offers.
    for row in rows.values():
        if row.widget in ("kind", "entry") and row.value:
            assert row.value in [item["value"] for item in row.items]


def test_a_choice_offers_exactly_the_values_the_schema_gives():
    document = read_case_document(PITZDAILY)
    row = _rows(document)[("boundaries", "upperWall", "k")]
    (wall_k,) = [f for f in SCHEMA.fields("Wall") if f["name"] == "k"]
    assert row.widget == "choice"
    assert [item["value"] for item in row.items] == wall_k["accepts"][0]["values"]


def test_a_position_offers_the_kinds_the_schema_gives_and_an_optional_one_its_default():
    rows = _rows(read_case_document(PITZDAILY))
    entry = rows[("boundaries", "inlet")]
    assert [item["value"] for item in entry.items] == _boundary_kinds()
    assert "Inlet" in _boundary_kinds()  # the reference is not empty
    turbulence = rows[("boundaries", "inlet", "turbulence")]
    # Optional, with no default the schema can state: offered as its kinds, and shown as not set.
    assert [item["value"] for item in turbulence.items] == ["FixedTurbulence", "IntensityLength"]
    # Optional in general, but required wherever it is offered: a RANS case needs it at an inlet.
    assert turbulence.placeholder == "required"
    # Unset is no value at all, so the dropdown shows that placeholder rather than a blank choice.
    assert rows[("physics", "k_variable")].value is None
    assert rows[("physics", "k_variable")].placeholder == case_form.NOT_SET


def test_a_section_left_to_its_default_still_shows_its_fields_with_their_defaults():
    rows = _rows(read_case_document(PITZDAILY))
    log = rows[("outputs", "log")]
    assert not log.is_set and log.placeholder == "march.log (default)"


#: A schema written for the test, holding one setting of each way a default can be stated.
DEFAULTS = CaseSchema(
    {
        "root": "Case",
        "one_form_sections": [],
        "kinds": {
            "Case": {"summary": "", "fields": [{
                "name": "settings", "required": False, "doc": "",
                "accepts": [{"type": "nested", "kinds": ["Settings"]}], "default": {"kind": "Settings"},
            }]},
            "Settings": {"summary": "", "fields": [
                {"name": "scheme", "required": False, "doc": "", "default": "second",
                 "accepts": [{"type": "choice", "values": ["first", "second"]}]},
                {"name": "limiter", "required": False, "doc": "", "default": None,
                 "accepts": [{"type": "choice", "values": ["minmod", "vanleer"]}, {"type": "null"}]},
                {"name": "verbose", "required": False, "doc": "", "default": False,
                 "accepts": [{"type": "boolean"}]},
                {"name": "relaxation", "required": False, "doc": "", "default": 0.7,
                 "accepts": [{"type": "number"}]},
                {"name": "smoother", "required": False, "doc": "", "default": {"kind": "Jacobi"},
                 "accepts": [{"type": "nested", "kinds": ["Gauss", "Jacobi"]}]},
                {"name": "steps", "required": True, "doc": "", "accepts": [{"type": "integer"}]},
                # Stored as None, and resolved by the solver to the default it falls through to.
                {"name": "sweeps", "required": False, "doc": "", "default": None,
                 "resolved_default": 4, "accepts": [{"type": "integer"}, {"type": "null"}]},
                {"name": "coarse", "required": False, "doc": "", "default": None,
                 "resolved_default": {"kind": "Gauss"},
                 "accepts": [{"type": "nested", "kinds": ["Gauss", "Jacobi"]}, {"type": "null"}]},
                {"name": "budget", "required": False, "doc": "", "default": None, "off": True,
                 "accepts": [{"type": "integer"}, {"type": "null"}]},
            ]},
            "Gauss": {"summary": "", "fields": []},
            "Jacobi": {"summary": "", "fields": []},
        },
    }
)  # fmt: skip


def test_an_unset_setting_shows_its_default_and_the_default_option_says_it_is_one():
    rows = {row.path[-1]: row for section in form_sections(DEFAULTS, {}) for row in section["rows"]}
    scheme, limiter = rows["scheme"], rows["limiter"]
    assert [item["title"] for item in scheme.items] == ["first", "second (default)"]
    assert [item["value"] for item in scheme.items] == ["first", "second"]  # the values unmarked
    assert scheme.placeholder == "second (default)"
    # No default the schema can state: no option is marked, and none is a stand-in for "default".
    assert [item["title"] for item in limiter.items] == ["minmod", "vanleer"]
    assert limiter.placeholder == case_form.NOT_SET
    assert [item["title"] for item in rows["verbose"].items] == ["True", "False (default)"]
    assert rows["relaxation"].placeholder == "0.7 (default)"
    assert [item["title"] for item in rows["smoother"].items] == ["Gauss", "Jacobi (default)"]
    assert rows["steps"].placeholder == "required"
    # A default the solver resolves is shown exactly as a stated one.
    assert rows["sweeps"].placeholder == "4 (default)"
    assert [item["title"] for item in rows["coarse"].items] == ["Gauss (default)", "Jacobi"]
    assert rows["budget"].placeholder == case_form.OFF


def test_a_key_the_schema_does_not_know_is_shown_so_it_can_be_removed():
    document = read_case_document(PITZDAILY)
    document["fluid"]["colour"] = "blue"
    document["extra"] = 1
    rows = _rows(document)
    assert rows[("fluid", "colour")].unknown and rows[("fluid", "colour")].removable
    assert rows[("extra",)].unknown


def test_edits_made_through_the_form_give_a_case_the_solver_reads():
    document = read_case_document(PITZDAILY)
    # A typed number, a choice, a new table entry of a chosen kind, and a setting created inside a
    # section left to its default.
    document = case_form.set_value(
        document, ("boundaries", "outlet", "pressure"), parse_input("number", "101325")[1], SCHEMA
    )
    document = case_form.set_value(
        document, ("boundaries", "upperWall", "k"), "zero_gradient", SCHEMA
    )
    document = case_form.unset(document, ("boundaries", "upperWall"))
    document = case_form.add_entry(document, ("boundaries",), "upperWall", "Wall", SCHEMA)
    document = case_form.set_value(document, ("outputs", "log"), "steps.log", SCHEMA)
    assert document["outputs"]["kind"] == "Outputs"
    spec = case_spec_from_mapping(document)
    assert spec.outputs.log == "steps.log"
    assert spec.boundaries["outlet"].pressure == 101325.0


def test_an_edit_returns_a_new_document_and_leaves_the_old_one():
    document = read_case_document(PITZDAILY)
    edited = case_form.set_value(document, ("fluid", "density"), 2.0)
    assert document["fluid"]["density"] == 1.0 and edited["fluid"]["density"] == 2.0


def test_a_new_kind_keeps_the_settings_the_old_one_shares_with_it():
    document = {
        "boundaries": {"lid": {"kind": "Wall", "velocity": [1.0, 0.0], "k": "zero_gradient"}}
    }
    edited = case_form.set_kind(document, ("boundaries", "lid"), "Inlet", SCHEMA)
    # Inlet has a velocity and no k.
    assert edited["boundaries"]["lid"] == {"kind": "Inlet", "velocity": [1.0, 0.0]}
    assert case_form.set_kind(edited, ("boundaries", "lid"), "", SCHEMA)["boundaries"] == {}


def test_a_table_entry_needs_a_new_name_and_a_list_item_is_appended():
    document = read_case_document(PITZDAILY)
    with pytest.raises(ValueError, match="already an entry named 'inlet'"):
        case_form.add_entry(document, ("boundaries",), "inlet", "Wall", SCHEMA)
    with pytest.raises(ValueError, match="needs a name"):
        case_form.add_entry(document, ("boundaries",), "  ", "Wall", SCHEMA)
    edited = case_form.add_item(document, ("outputs", "fields"), "Vtk", SCHEMA)
    assert edited["outputs"]["fields"][-1] == {"kind": "Vtk"}
    removed = case_form.remove_at(edited, ("outputs", "fields", 0))
    assert len(removed["outputs"]["fields"]) == len(edited["outputs"]["fields"]) - 1


def test_the_field_at_a_path_follows_the_kinds_the_document_states():
    document = read_case_document(PITZDAILY)
    assert (
        case_form.field_at(SCHEMA, document, ("boundaries", "inlet", "turbulence", "k"))["name"]
        == "k"
    )
    assert case_form.entry_kinds(SCHEMA, document, ("boundaries",)) == _boundary_kinds()
    assert case_form.field_at(SCHEMA, document, ("fluid", "no_such_field")) is None


@pytest.mark.parametrize(
    ("widget", "text", "expected"),
    [
        ("number", " 1e-5 ", (True, 1e-5)),
        ("integer", "3", (True, 3)),
        ("integer", "3.0", (True, 3)),
        ("numbers", "10, 0", (True, [10.0, 0.0])),
        ("numbers", "1 2 3", (True, [1.0, 2.0, 3.0])),
        ("strings", "U, p", (True, ["U", "p"])),
        ("string", "results", (True, "results")),
        ("raw", '{"a": 1}', (True, {"a": 1})),
        ("number", "", (False, None)),
        ("boolean", True, (True, True)),
        ("choices", ["x"], (True, ["x"])),
        ("choices", [], (False, None)),
    ],
)
def test_typed_text_is_read_as_its_rows_kind_of_value(widget, text, expected):
    assert parse_input(widget, text) == expected


@pytest.mark.parametrize(
    ("widget", "text", "match"),
    [
        ("number", "fast", "could not convert"),
        ("integer", "2.5", "not a whole number"),
        ("raw", "{", "not JSON"),
    ],
)
def test_typed_text_that_is_not_its_rows_kind_of_value_is_refused(widget, text, match):
    with pytest.raises(ValueError, match=match):
        parse_input(widget, text)


def test_help_text_is_plain():
    assert case_form.plain_doc("a :class:`~aquaflux.flow.PinnedPoint`, ``None`` unset") == (
        "a PinnedPoint, None unset"
    )


def test_a_list_of_fixed_choices_is_picked_from_its_choices():
    document = read_case_document(
        REPO / "validation" / "turbulent_channel" / "cases" / "re20000.yaml"
    )
    row = _rows(document)[("mesh", "periodic")]
    assert row.widget == "choices" and row.value == ["x"]
    assert [item["value"] for item in row.items] == ["x"]


def test_a_group_says_what_is_set_inside_it_so_it_can_stay_folded():
    document = read_case_document(PITZDAILY)
    inlet = document["boundaries"]["inlet"]
    rows = _rows(document)
    group = rows[("boundaries", "inlet")]
    # Its own settings by name and value; a nested group by its kind.
    velocity = ", ".join(str(float(v)) for v in inlet["velocity"])
    assert group.contents == f"velocity {velocity} · turbulence {inlet['turbulence']['kind']}"
    # Every setting the file states anywhere inside: the velocity, and the turbulence's own.
    assert group.set_inside == 1 + len([key for key in inlet["turbulence"] if key != "kind"])
    assert group.kind_summary == SCHEMA.summary("Inlet")
    # A nested group counts only what is inside it; a plain setting carries none of this.
    turbulence = rows[("boundaries", "inlet", "turbulence")]
    assert turbulence.set_inside == len(inlet["turbulence"]) - 1
    assert rows[("boundaries", "inlet", "velocity")].contents == ""


def test_a_group_with_nothing_set_inside_it_has_no_summary_to_show():
    rows = {row.path[-1]: row for section in form_sections(DEFAULTS, {}) for row in section["rows"]}
    smoother = rows["smoother"]  # a nested kind left at its default, holding nothing set
    assert smoother.contents == "" and smoother.set_inside == 0
    assert smoother.kind_summary == DEFAULTS.summary("Jacobi")


def test_a_sections_kind_heads_its_settings_and_only_a_group_with_contents_folds():
    rows = _rows(read_case_document(PITZDAILY))
    physics, advection = rows[("physics",)], rows[("numerics", "turbulence_advection")]
    # The section's own kind row heads it, so the kind's settings sit one level in, beneath it.
    assert physics.widget == "kind" and physics.depth == 0 and advection.depth == 1
    assert rows[("physics", "omega_variable")].depth == 1 and physics.has_inside
    # FirstOrderUpwind has no settings: nothing to fold, so no fold arrow.
    assert advection.value == "FirstOrderUpwind" and not advection.has_inside
    # Likewise an optional section's: the fluid's kind row heads its settings.
    assert rows[("fluid",)].depth == 0 and rows[("fluid", "density")].depth == 1


def test_a_long_default_is_shown_to_six_significant_figures():
    rows = _rows(read_case_document(PITZDAILY))
    assert rows[("physics", "model", "alpha_1")].placeholder == "0.555556 (default)"


def test_a_laminar_case_is_not_offered_the_settings_only_a_turbulence_closure_reads():
    rans = read_case_document(PITZDAILY)
    laminar = case_form.set_kind(rans, ("physics",), "Laminar", SCHEMA)
    for document, turbulent in ((rans, True), (laminar, False)):
        rows = _rows(document)
        for path in (
            ("numerics", "turbulence_advection"),  # a scheme only k and omega need
            ("boundaries", "upperWall", "k"),  # a wall's k condition
            ("boundaries", "inlet", "turbulence"),  # the turbulence an inlet admits
        ):
            assert (path in rows) is turbulent, (path, turbulent)
        # Settings every flow reads are offered either way; light is offered to neither.
        assert ("numerics", "momentum_advection") in rows and (
            "boundaries",
            "inlet",
            "velocity",
        ) in rows
        assert ("boundaries", "upperWall", "reflectance") not in rows
    # Turning the case laminar drops what it would refuse, so the file stays one the solver reads.
    assert "turbulence_advection" not in laminar["numerics"]
    assert laminar["boundaries"]["upperWall"] == {"kind": "Wall"}
    assert "turbulence" not in laminar["boundaries"]["inlet"]
    assert laminar["numerics"]["momentum_advection"] == rans["numerics"]["momentum_advection"]


def test_a_setting_its_scope_requires_reads_as_required_where_it_is_offered():
    document = read_case_document(PITZDAILY)
    del document["numerics"]["turbulence_advection"]
    assert _rows(document)[("numerics", "turbulence_advection")].placeholder == "required"
