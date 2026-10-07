"""A case's ``initial`` section: the checkpoint file it reads, what it refuses, and what it hands a solve.

What a whole run does with a starting state -- resuming a stopped march to the same root -- is
``tests/integration/test_case_run.py``'s. These tests pin the pieces without a solve: the file format
(physical fields, so a log-variable case and a direct one agree), the mesh digest, each way a state is
refused, the order a run checks and clears in, and the records.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import aquaflux  # noqa: F401  (enables x64)
import equinox as eqx
import numpy as np
import pytest
import yaml
from aquaflux.case import (
    CheckedCase,
    Checkpoint,
    case_spec_from_mapping,
    case_spec_to_mapping,
    prepare_run,
)
from aquaflux.case.restart_file import (
    RestartHeader,
    checkpoint_writer,
    mesh_digest,
    read_restart,
)
from aquaflux.mesh import permute_cells
from aquaflux.solve import StateCheckpointer, StepReport


def _sections(physics: str = "Laminar", cells=(4, 4), **overrides: object) -> dict[str, object]:
    """A channel on a structured grid, as a file would state it."""
    rans = physics == "RANS"
    inlet = {"kind": "Inlet", "velocity": [1.0, 0.0]}
    if rans:
        inlet["turbulence"] = {"kind": "FixedTurbulence", "k": 1e-3, "omega": 2.0}
    sections = {
        "mesh": {"kind": "StructuredGrid", "cells": list(cells), "lengths": [2.0, 1.0]},
        "fluid": {"density": 1.0, "kinematic_viscosity": 1.0e-2},
        "physics": {
            "kind": physics,
            **({"omega_variable": {"kind": "LogScalars"}} if rans else {}),
        },
        "boundaries": {
            "left": inlet,
            "right": {"kind": "Outlet", "pressure": 0.0},
            "bottom": {"kind": "Wall"},
            "top": {"kind": "Wall"},
        },
        "numerics": {
            "momentum_advection": {"kind": "FirstOrderUpwind"},
            **({"turbulence_advection": {"kind": "FirstOrderUpwind"}} if rans else {}),
        },
    }
    return {**sections, **overrides}


def _checked(directory: Path, **kwargs: object) -> CheckedCase:
    """The case read and checked against its mesh, with the directory its paths are taken from."""
    spec = case_spec_from_mapping(_sections(**kwargs))
    mesh = spec.mesh.read(directory).validate()
    spec.check_against(mesh)
    return CheckedCase(spec=spec, mesh=mesh, directory=directory)


def _report(step: int = 0, residual: float = 1.0e-3) -> StepReport:
    return StepReport(step=step, cycles=3, residual_norm=residual, residual_ratio=0.5, alpha=1.0)


def _write_checkpoints(checked: CheckedCase, directory: Path, states) -> None:
    """Checkpoint ``states`` in order into ``directory``, as a run of this case would."""
    problem = checked.build()
    physics = checked.spec.physics
    header = RestartHeader.of(physics, checked.mesh, "digest-of-the-writer")
    checkpointer = StateCheckpointer(
        directory, keep=len(states), save=checkpoint_writer(physics, problem, header)
    )
    for step, state in enumerate(states):
        checkpointer.on_checkpoint(_report(step, residual=10.0 ** -(step + 1)), state)


# --- the mesh digest -------------------------------------------------------------------------------


def test_the_same_mesh_digests_the_same_wherever_and_however_often_it_is_read(tmp_path) -> None:
    spec = case_spec_from_mapping(_sections())
    first, second = (spec.mesh.read(tmp_path) for _ in range(2))
    assert mesh_digest(first) == mesh_digest(second)


def test_a_digest_ignores_rounding_in_the_last_bits_of_a_coordinate(tmp_path) -> None:
    """The same mesh read on another machine can differ in the last bit of a node: that is not a new mesh."""
    mesh = case_spec_from_mapping(_sections()).mesh.read(tmp_path)
    noisy = eqx.tree_at(lambda m: m.node_coords, mesh, mesh.node_coords + 1.0e-12)
    assert mesh_digest(noisy) == mesh_digest(mesh)


def test_a_digest_tells_a_moved_mesh_from_the_original(tmp_path) -> None:
    mesh = case_spec_from_mapping(_sections()).mesh.read(tmp_path)
    stretched = case_spec_from_mapping(
        _sections(mesh={"kind": "StructuredGrid", "cells": [4, 4], "lengths": [2.0, 1.01]})
    ).mesh.read(tmp_path)
    assert mesh_digest(stretched) != mesh_digest(mesh)


def test_a_digest_tells_a_renumbered_mesh_from_the_original(tmp_path) -> None:
    """Same nodes, same cell count, cells in another order: a field of the right length would load as nonsense."""
    mesh = case_spec_from_mapping(_sections()).mesh.read(tmp_path)
    renumbered = permute_cells(mesh, np.roll(np.arange(mesh.n_cells), 1))
    assert np.array_equal(np.asarray(renumbered.node_coords), np.asarray(mesh.node_coords))
    assert mesh_digest(renumbered) != mesh_digest(mesh)


# --- what a header refuses -------------------------------------------------------------------------

_HEADER = RestartHeader(physics="Laminar", n_cells=16, dim=2, mesh_digest="a", case_digest="x")


def _refusal(found: RestartHeader, expected: RestartHeader = _HEADER) -> str:
    with pytest.raises(ValueError) as error:
        found.refuse_unless_fits(expected, Path("state-00001.npz"))
    return str(error.value)


def test_a_header_that_fits_is_accepted_whatever_case_wrote_it() -> None:
    _HEADER.refuse_unless_fits(dataclasses.replace(_HEADER, case_digest="another"), Path("f.npz"))


def test_a_checkpoint_of_another_physics_is_refused_naming_both() -> None:
    message = _refusal(dataclasses.replace(_HEADER, physics="RANS"))
    assert "its fields are of a RANS case, but this case is Laminar" in message
    assert "cells" not in message and "same mesh" not in message


def test_a_checkpoint_of_another_cell_count_is_refused_naming_both_counts() -> None:
    message = _refusal(dataclasses.replace(_HEADER, n_cells=24, mesh_digest="b"))
    assert "it holds 24 cells, but this case's mesh has 16" in message
    # The digest differs whenever the count does, so reporting it as well would say nothing more.
    assert "same number of cells" not in message


def test_a_checkpoint_of_another_dimension_is_refused() -> None:
    message = _refusal(dataclasses.replace(_HEADER, dim=3, mesh_digest="b"))
    assert "3-dimensional mesh, but this case's is 2-dimensional" in message
    assert "same number of cells" not in message


def test_a_checkpoint_of_a_different_mesh_of_the_same_size_is_refused() -> None:
    message = _refusal(dataclasses.replace(_HEADER, mesh_digest="b"))
    assert "same number of cells as this case's but is not the same mesh" in message


def test_every_misfit_is_listed_at_once() -> None:
    message = _refusal(dataclasses.replace(_HEADER, physics="RANS", n_cells=24, mesh_digest="b"))
    assert "RANS" in message and "24 cells" in message


# --- the file ------------------------------------------------------------------------------------


def test_a_bare_solved_state_is_refused_as_not_a_case_checkpoint(tmp_path) -> None:
    """What a library checkpointer writes by default has no header, so it cannot say what mesh it is on."""
    StateCheckpointer(tmp_path).on_checkpoint(_report(), np.zeros(10))
    (file,) = tmp_path.iterdir()
    with pytest.raises(ValueError, match=r"is not a case checkpoint: it has no .*mesh_digest"):
        read_restart(file)


def test_a_laminar_checkpoint_holds_the_physical_fields_and_the_headers_record(tmp_path) -> None:
    checked = _checked(tmp_path)
    problem = checked.build()
    rng = np.random.default_rng(0)
    velocity = rng.normal(size=(16, 2))
    pressure = rng.normal(size=16)
    state = problem.pack(velocity, pressure)
    _write_checkpoints(checked, tmp_path / "ck", [state])

    (file,) = (tmp_path / "ck").iterdir()
    restart = read_restart(file)
    np.testing.assert_array_equal(restart.fields["U"], velocity)
    np.testing.assert_array_equal(restart.fields["p"], pressure)
    assert set(restart.fields) == {"U", "p"}
    assert restart.header == RestartHeader.of(
        checked.spec.physics, checked.mesh, "digest-of-the-writer"
    )
    assert restart.residual == 0.1


def test_a_rans_checkpoint_holds_omega_not_its_logarithm(tmp_path) -> None:
    """The march solves for log(omega) here; a file of that vector read by a direct case would be garbage."""
    checked = _checked(tmp_path, physics="RANS")
    problem = checked.build()
    physics = checked.spec.physics
    rng = np.random.default_rng(1)
    fields = {
        "U": rng.normal(size=(16, 2)),
        "p": rng.normal(size=16),
        "k": rng.uniform(1e-3, 1.0, size=16),
        "omega": rng.uniform(1.0, 500.0, size=16),
    }
    state = problem.state_from_physical(*physics.initial_fields(problem, fields))
    # The solved vector really is in the log variable, so this test is about the transform.
    solved_omega = np.asarray(problem.layout.unpack(state)[2])
    np.testing.assert_allclose(solved_omega, np.log(fields["omega"]), rtol=1e-12)

    _write_checkpoints(checked, tmp_path / "ck", [state])
    (file,) = (tmp_path / "ck").iterdir()
    restart = read_restart(file)
    assert set(restart.fields) == {"U", "p", "k", "omega"}
    for name, values in fields.items():
        np.testing.assert_allclose(restart.fields[name], values, rtol=1e-12, err_msg=name)

    # And back: the seed a solve takes maps to the state the march was in.
    seed = physics.initial_fields(problem, restart.fields)
    np.testing.assert_allclose(
        np.asarray(problem.state_from_physical(*seed)), np.asarray(state), rtol=1e-12
    )


def test_a_starting_state_missing_a_field_is_refused_naming_it(tmp_path) -> None:
    checked = _checked(tmp_path, physics="RANS")
    problem = checked.build()
    with pytest.raises(ValueError, match=r"\['omega'\] is missing from \['U', 'k', 'p'\]"):
        checked.spec.physics.initial_fields(
            problem, {"U": np.zeros((16, 2)), "p": np.zeros(16), "k": np.ones(16)}
        )


# --- reading a checkpoint for a case -------------------------------------------------------------


def test_a_checkpoint_reads_the_latest_step_or_the_one_named(tmp_path) -> None:
    checked = _checked(tmp_path)
    problem = checked.build()
    states = [problem.pack(np.full((16, 2), float(i)), np.full(16, float(i))) for i in (1, 2, 3)]
    _write_checkpoints(checked, tmp_path / "ck", states)
    physics = checked.spec.physics

    latest = Checkpoint(path="ck").read(tmp_path, physics, checked.mesh)
    assert float(latest.fields["p"][0]) == 3.0
    named = Checkpoint(path="ck", step=2).read(tmp_path, physics, checked.mesh)
    assert float(named.fields["p"][0]) == 2.0
    assert Path(named.source["file"]).name == "state-00002.npz"
    assert named.source["kind"] == "Checkpoint"
    assert named.source["case_digest"] == "digest-of-the-writer"
    assert named.source["residual"] == 0.01


def test_a_checkpoint_of_another_mesh_is_refused_before_anything_is_built(tmp_path) -> None:
    checked = _checked(tmp_path)
    _write_checkpoints(checked, tmp_path / "ck", [checked.build().initial_state()])
    other = _checked(tmp_path, cells=(8, 4))
    with pytest.raises(ValueError, match=r"it holds 16 cells, but this case's mesh has 32"):
        Checkpoint(path="ck").read(tmp_path, other.spec.physics, other.mesh)


def test_a_checkpoint_of_a_diverged_state_is_refused_naming_the_field(tmp_path) -> None:
    """The checkpointer also writes the step a march died on, so the newest file can be poison."""
    checked = _checked(tmp_path)
    problem = checked.build()
    good = problem.pack(np.ones((16, 2)), np.ones(16))
    bad = problem.pack(np.ones((16, 2)), np.where(np.arange(16) == 5, np.nan, 1.0))
    _write_checkpoints(checked, tmp_path / "ck", [good, bad])
    physics = checked.spec.physics

    with pytest.raises(ValueError, match=r"not finite in \['p'\]: the march had diverged"):
        Checkpoint(path="ck").read(tmp_path, physics, checked.mesh)
    # An earlier step that was kept still starts the case.
    assert (
        Checkpoint(path="ck", step=1).read(tmp_path, physics, checked.mesh).fields["p"].min() == 1.0
    )


def test_a_checkpoint_missing_the_step_asked_for_lists_those_it_has(tmp_path) -> None:
    checked = _checked(tmp_path)
    _write_checkpoints(checked, tmp_path / "ck", [checked.build().initial_state()] * 2)
    with pytest.raises(
        FileNotFoundError, match=r"no checkpoint of step 9; it holds steps \[1, 2\]"
    ):
        Checkpoint(path="ck", step=9).read(tmp_path, checked.spec.physics, checked.mesh)


# --- the section ---------------------------------------------------------------------------------


def test_an_initial_section_reads_and_writes_back_equal() -> None:
    section = {"kind": "Checkpoint", "path": "../first/checkpoints", "step": 4}
    spec = case_spec_from_mapping(_sections(initial=section))
    assert spec.initial == Checkpoint(path="../first/checkpoints", step=4)
    assert case_spec_to_mapping(spec)["initial"] == section
    assert case_spec_from_mapping(case_spec_to_mapping(spec)) == spec


def test_a_file_with_no_initial_section_starts_from_scratch() -> None:
    spec = case_spec_from_mapping(_sections())
    assert spec.initial is None
    assert "initial" not in case_spec_to_mapping(spec)
    assert Checkpoint(path="ck").step == "latest"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"path": ""}, r"Checkpoint.path names where the starting state is"),
        ({"path": "ck", "step": 0}, r"Checkpoint.step is 'latest' or a step >= 1, got 0"),
        ({"path": "ck", "step": "newest"}, r"Checkpoint.step is 'latest' or a step >= 1"),
        ({"path": "ck", "step": True}, r"Checkpoint.step is 'latest' or a step >= 1, got True"),
    ],
)
def test_a_checkpoint_refuses_what_names_no_state(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        Checkpoint(**kwargs)


def test_an_initial_section_of_the_wrong_kind_is_refused_where_it_stands() -> None:
    with pytest.raises(ValueError, match=r"initial"):
        case_spec_from_mapping(_sections(initial={"kind": "Vtk"}))


_RAMP = {
    "kind": "CoupledMarch",
    "continuation": {
        "kind": "ViscosityRamp",
        "anchor": 10.0,
        "stations": 3,
        "steps_per_station": 2,
    },
}


def test_a_ramp_with_a_starting_state_is_refused_when_the_file_is_read() -> None:
    with pytest.raises(ValueError, match=r"a viscosity ramp opens on a seed fitted to its anchor"):
        case_spec_from_mapping(
            _sections(
                physics="RANS",
                solver=_RAMP,
                initial={"kind": "Checkpoint", "path": "ck"},
            )
        )


@pytest.mark.parametrize(
    "solver",
    [
        {"kind": "CoupledMarch"},
        {"kind": "Segregated", "sweeps": 5},
        None,
    ],
    ids=["march", "segregated", "default"],
)
def test_every_other_solve_can_start_from_a_state(solver) -> None:
    extra = {} if solver is None else {"solver": solver}
    spec = case_spec_from_mapping(
        _sections(physics="RANS", initial={"kind": "Checkpoint", "path": "ck"}, **extra)
    )
    assert spec.initial == Checkpoint(path="ck")


def test_the_same_ramp_without_a_starting_state_is_allowed() -> None:
    assert case_spec_from_mapping(_sections(physics="RANS", solver=_RAMP)).initial is None


# --- what a run checks before it clears anything, and what it records ---------------------------------


_EARLIER = "an earlier run's checkpoint"


def _file(tmp_path: Path, name: str = "case.yaml", **overrides: object) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(_sections(**overrides)))
    return path


def test_a_restart_reading_its_own_output_directory_is_refused_even_when_told_to_overwrite(
    tmp_path,
) -> None:
    case = _file(tmp_path, initial={"kind": "Checkpoint", "path": "results/checkpoints"})
    for overwrite in (False, True):
        with pytest.raises(ValueError, match=r"lies inside the output directory .* replaces"):
            prepare_run(case, overwrite=overwrite)


def test_a_restart_that_does_not_fit_leaves_the_checkpoints_it_was_told_to_overwrite(
    tmp_path,
) -> None:
    """A refused state must not have cost an earlier run its checkpoints."""
    earlier = _checked(tmp_path)
    _write_checkpoints(earlier, tmp_path / "first", [earlier.build().initial_state()])
    results = tmp_path / "results"
    (results / "checkpoints").mkdir(parents=True)
    survivor = results / "checkpoints" / "state-00001.npz"
    survivor.write_text(_EARLIER)

    case = _file(
        tmp_path,
        mesh={"kind": "StructuredGrid", "cells": [8, 4], "lengths": [2.0, 1.0]},
        initial={"kind": "Checkpoint", "path": "first"},
    )
    with pytest.raises(ValueError, match=r"it holds 16 cells, but this case's mesh has 32"):
        prepare_run(case, overwrite=True)
    assert survivor.read_text() == _EARLIER


def test_a_prepared_restart_carries_the_state_it_was_read_from(tmp_path) -> None:
    earlier = _checked(tmp_path)
    _write_checkpoints(earlier, tmp_path / "first", [earlier.build().initial_state()])
    prepared = prepare_run(_file(tmp_path, initial={"kind": "Checkpoint", "path": "first"}))
    assert prepared.starting is not None
    assert (
        Path(prepared.starting.source["file"]) == (tmp_path / "first" / "state-00001.npz").resolve()
    )
    assert set(prepared.starting.fields) == {"U", "p"}


def test_a_case_with_no_initial_section_prepares_with_no_starting_state(tmp_path) -> None:
    assert prepare_run(_file(tmp_path)).starting is None


# --- a physics with nothing marched has no state to start from ------------------------------------


def test_a_radiation_case_refuses_a_starting_state_when_the_file_is_read() -> None:
    """Nothing is marched in a radiation case, so a state to start from would be read by nothing."""
    from aquaflux.case import case_spec_from_mapping as read

    radiation = {
        "mesh": {"kind": "StructuredGrid", "cells": [2, 2, 2], "lengths": [1.0, 1.0, 1.0]},
        "physics": {"kind": "Radiation"},
        "boundaries": {
            "left": {"kind": "Lamp", "profile": {"kind": "LambertianProfile"}, "power": 1.0},
            "right": {"kind": "Wall"},
            "bottom": {"kind": "Wall"},
            "top": {"kind": "Wall"},
            "back": {"kind": "Wall"},
            "front": {"kind": "Wall"},
        },
    }
    assert read(radiation).initial is None
    with pytest.raises(ValueError, match=r"initial: a radiation case has no flow"):
        read({**radiation, "initial": {"kind": "Checkpoint", "path": "ck"}})


def test_a_physics_that_marches_nothing_has_no_fields_to_save_or_start_from(tmp_path) -> None:
    from aquaflux.case import Radiation

    with pytest.raises(ValueError, match=r"a Radiation case has no march state to save"):
        Radiation().restart_fields(None, None)
    with pytest.raises(ValueError, match=r"a Radiation case has no state to start from"):
        Radiation().initial_fields(None, {})
