"""The ``aquaflux`` command: check a case file, or run it; make a mesh from a mesh file.

::

    aquaflux check case.yaml            # read the file and check it against its mesh, in seconds
    aquaflux run case.yaml              # read, check, build, solve and write the results
    aquaflux run case.yaml --overwrite  # replace the results of an earlier run
    aquaflux mesh mesh.yaml             # mesh the region a mesh file describes, and write it
    aquaflux mesh mesh.yaml --overwrite # replace a mesh made before

``python -m aquaflux`` is the same command. ``run`` exits with status 0 when the solve converges and 1
when it stops short (its log, checkpoints and run record are still written); ``check``, ``run`` and
``mesh`` exit with status 2 when the file is refused, with the reason on standard error.

Three more commands are for a program editing case files -- the browser interface -- and speak JSON
(JavaScript Object Notation) on their standard output:

    aquaflux schema                     # what a case file may hold: every kind, field and choice
    aquaflux show case.yaml             # the file as aquaflux reads it, and why it is refused if it is
    aquaflux plan case.yaml             # where a run would write, and which earlier results it would replace
    aquaflux write case.yaml < case.json  # write the case given on standard input, checked first
    aquaflux write copy.yaml --relative-to original/ < case.json  # re-base its relative paths
    aquaflux export-mesh out/ --relative-to case_dir/ < mesh.json  # a case's mesh and its patches, as VTK
    aquaflux export-mesh out/ --case case.yaml --openfoam  # a case file's mesh, also as out/constant/polyMesh
    aquaflux drawing out/ --relative-to mesh_dir/ < mesh_file.json  # a mesh file's drawing and its patches
    aquaflux serve                      # answer any of these, one per line, without restarting

``schema``, ``show`` and ``write`` take ``--mesh-file`` to read and write a mesh file in place of a
case file. ``show``, ``plan``, ``write``, ``export-mesh`` and ``drawing`` print ``{"error": null, ...}``
on success and ``{"error": "<reason>"}`` with exit status 2 when the file, or the case, is refused.

``serve`` is for a program that asks many of these in turn: starting Python and importing the solver
takes seconds, and the commands themselves take a fraction of that. It reads one request per line --
``{"arguments": ["show", "case.yaml"], "stdin": null}`` -- and answers each with one line,
``{"status": 0, "output": "...", "errors": "..."}``: what the command would have exited with and
printed. It stops at the end of its input.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from aquaflux.case import (
    case_schema,
    case_spec_from_mapping,
    drawing_sections_from_mapping,
    make_mesh,
    mesh_schema,
    mesh_source_from_mapping,
    mesh_spec_from_mapping,
    plan_run,
    prepare_run,
    read_case,
    read_case_document,
    relocated,
    solver_for,
    view_drawing,
    write_case,
    write_mesh_file,
)
from aquaflux.io import write_openfoam, write_patches, write_vtu

__all__ = ["main"]

#: The exit status of a refused file, distinct from a solve that did not converge.
_REFUSED = 2

#: The commands ``serve`` answers: the ones that read and write case files, each finished in seconds.
SERVED = ("schema", "show", "write", "export-mesh", "drawing", "check", "plan")


@dataclasses.dataclass(frozen=True)
class _Document:
    """One kind of file the editing commands read and write: what it may hold, how it reads, how
    it is written."""

    name: str
    schema: Callable[[], dict]
    read: Callable[[object], object]
    write: Callable[[object, str], None]


_CASE_FILE = _Document("case", case_schema, case_spec_from_mapping, write_case)
_MESH_FILE = _Document("mesh file", mesh_schema, mesh_spec_from_mapping, write_mesh_file)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``aquaflux`` command.

    Parameters
    ----------
    argv : sequence of str, optional
        The arguments after the program name; unset, the command line's.

    Returns
    -------
    int
        The exit status: 0 on success, 1 for a run that did not converge, 2 for a refused file.
    """
    parser = argparse.ArgumentParser(
        prog="aquaflux", description="Check or run an aquaflux case file."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="read a case file and check it against its mesh")
    check.add_argument("case", help="the case file")
    run = commands.add_parser("run", help="solve a case file and write its outputs")
    run.add_argument("case", help="the case file")
    run.add_argument(
        "--overwrite", action="store_true", help="replace the results of an earlier run"
    )
    plan = commands.add_parser(
        "plan",
        help="print where a run of a case file would write, and what it would replace, as JSON",
    )
    plan.add_argument("case", help="the case file")
    schema = commands.add_parser("schema", help="print what a case file may hold, as JSON")
    show = commands.add_parser("show", help="print a case file as aquaflux reads it, as JSON")
    show.add_argument("case", help="the case file")
    write = commands.add_parser(
        "write", help="write the case given as JSON on standard input to a case file"
    )
    write.add_argument("case", help="the case file to write; replaced if it exists")
    for editing in (schema, show, write):
        editing.add_argument(
            "--mesh-file", action="store_true", help="a mesh file in place of a case file"
        )
    write.add_argument(
        "--relative-to",
        metavar="DIR",
        help="the directory the case's relative paths are relative to now; they are re-based onto "
        "the written file's directory (default: that directory, so nothing moves)",
    )
    make = commands.add_parser("mesh", help="mesh the region a mesh file describes, and write it")
    make.add_argument("mesh_file", help="the mesh file (YAML)")
    make.add_argument(
        "--overwrite", action="store_true", help="replace a mesh already where this one goes"
    )
    mesh = commands.add_parser(
        "export-mesh", help="read the mesh section given as JSON on standard input, as VTK"
    )
    mesh.add_argument("directory", help="where to write mesh.vtu")
    mesh.add_argument(
        "--relative-to",
        metavar="DIR",
        default=None,
        help="the directory the mesh section's relative paths are relative to (default: here, or "
        "the case file's directory with --case)",
    )
    mesh.add_argument(
        "--case",
        metavar="FILE",
        default=None,
        help="take the mesh section from this case file rather than from standard input",
    )
    mesh.add_argument(
        "--openfoam",
        action="store_true",
        help="also write the mesh as an OpenFOAM polyMesh, in DIRECTORY/constant/polyMesh",
    )
    drawing = commands.add_parser(
        "drawing",
        help="write the drawing of the mesh file given as JSON on standard input as VTK, and print "
        "its faces and their patches",
    )
    drawing.add_argument("directory", help="where to write drawing.vtp")
    drawing.add_argument(
        "--relative-to",
        metavar="DIR",
        default=".",
        help="the directory the mesh file's relative paths are relative to (default: here)",
    )
    commands.add_parser(
        "serve", help=f"answer {', '.join(SERVED)} requests, one JSON line each, until input ends"
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "serve":
        return _serve_process()
    if arguments.command == "mesh":
        return _make_mesh(arguments.mesh_file, arguments.overwrite)
    if arguments.command == "export-mesh":
        return _mesh(
            arguments.directory,
            arguments.case,
            arguments.relative_to,
            arguments.openfoam,
            None if arguments.case is not None else sys.stdin.read(),
        )
    if arguments.command == "drawing":
        return _drawing(arguments.directory, arguments.relative_to, sys.stdin.read())
    if arguments.command in ("schema", "show", "write"):
        document = _MESH_FILE if arguments.mesh_file else _CASE_FILE
        if arguments.command == "schema":
            print(json.dumps(document.schema()))
            return 0
        if arguments.command == "show":
            return _show(arguments.case, document)
        return _write(arguments.case, sys.stdin.read(), arguments.relative_to, document)
    if arguments.command == "plan":
        return _plan(arguments.case)
    try:
        if arguments.command == "check":
            return _check(arguments.case)
        prepared = prepare_run(arguments.case, overwrite=arguments.overwrite)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError, yaml.YAMLError) as error:
        print(f"aquaflux: {error}", file=sys.stderr)
        return _REFUSED
    return 0 if prepared.run().converged else 1


#: What a case file, or a case, is refused with.
_REFUSALS = (ValueError, TypeError, FileNotFoundError, yaml.YAMLError)


def _show(path: str, kind: _Document = _CASE_FILE) -> int:
    """Print ``path``'s document as JSON, with the reason it is refused, if it is.

    The document is under ``case`` whichever ``kind`` of file it is: it is the file's content.
    """
    try:
        document = read_case_document(path)
    except (*_REFUSALS, OSError) as error:
        print(json.dumps({"error": f"{path}: {error}"}))
        return _REFUSED
    try:
        kind.read(document)
        error = None
    except (ValueError, TypeError) as refusal:
        error = str(refusal)
    print(json.dumps({"error": error, "case": document}))
    return 0


def _plan(path: str) -> int:
    """Print where a run of ``path`` would write, and what it would replace, as JSON."""
    try:
        plan = plan_run(path)
    except (*_REFUSALS, OSError) as error:
        print(json.dumps({"error": f"{path}: {error}"}))
        return _REFUSED
    print(
        json.dumps(
            {
                "error": None,
                "directory": str(plan.directory),
                "log": None if plan.log is None else str(plan.log),
                "history": None if plan.history is None else str(plan.history),
                "occupied": [str(target) for target in plan.occupied],
            }
        )
    )
    return 0


def _write(
    path: str, given: str, relative_to: str | None = None, kind: _Document = _CASE_FILE
) -> int:
    """Write the document in the JSON ``given`` (``{"case": {...}}``) to ``path``, if it reads as a
    file of ``kind``.

    Its relative paths are re-based from ``relative_to`` onto ``path``'s directory, so a copy saved
    elsewhere still finds its mesh, or its drawing.
    """
    try:
        spec = kind.read(json.loads(given)["case"])
        if relative_to is not None:
            spec = relocated(spec, Path(relative_to).resolve(), Path(path).resolve().parent)
        kind.write(spec, path)
    except (*_REFUSALS, KeyError, json.JSONDecodeError, OSError) as error:
        print(json.dumps({"error": str(error)}))
        return _REFUSED
    print(json.dumps({"error": None, "path": path}))
    return 0


def _make_mesh(path: str, overwrite: bool) -> int:
    """Mesh the region the mesh file ``path`` describes, write it, and say what was made.

    Each stage is printed as it starts, after the seconds since meshing began, so a long meshing
    can be followed; then the mesh made, and its quality.
    """
    started = time.perf_counter()

    def on_stage(stage: str) -> None:
        print(f"{time.perf_counter() - started:7.1f} s  {stage}", flush=True)

    try:
        record = make_mesh(path, overwrite=overwrite, on_stage=on_stage)
    except (*_REFUSALS, FileExistsError) as error:
        print(f"aquaflux: {error}", file=sys.stderr)
        return _REFUSED
    mesh = record.mesh
    patches = ", ".join(
        f"{name} ({mesh.face_patches.size(name)} faces)"
        for name in mesh.face_patches.names
        if mesh.face_patches.size(name)
        and mesh.face_patches.is_boundary_patch(name, mesh.face_cells)
    )
    written = str(record.aqmesh) + ("" if record.polymesh is None else f" and {record.polymesh}")
    quality = record.quality
    print(
        f"{path}: {mesh.n_cells} cells ({mesh.dim}D), patches {patches}; meshed in "
        f"{record.seconds:.1f} s; written to {written}.\n"
        f"quality: non-orthogonality {quality.non_orthogonality_p99:.1f} deg at the 99th "
        f"percentile, {quality.non_orthogonality_max:.1f} deg at most; neighbour volume ratio "
        f"{quality.volume_ratio_p99:.1f} at the 99th percentile; {quality.concave_cells} "
        f"concave cells; smallest cell {quality.smallest_volume:.3g} m^3."
    )
    return 0


def _drawing(directory: str, relative_to: str, given: str) -> int:
    """Write the drawing of the mesh file in the JSON ``given`` (``{"mesh_file": {...}}``) into
    ``directory`` as ``drawing.vtp``, and print its faces and their patches.

    Only the file's drawing, region, patches, ``others`` and feature angle are read, so a mesh file
    still being written can be looked at; relative paths are taken from ``relative_to``. Prints
    ``{"error": null, ...}`` with what :class:`~aquaflux.case.DrawingView` holds.
    """
    try:
        sections = drawing_sections_from_mapping(json.loads(given)["mesh_file"])
        view = view_drawing(sections, Path(relative_to), Path(directory))
    except (*_REFUSALS, KeyError, json.JSONDecodeError, OSError) as error:
        print(json.dumps({"error": str(error)}))
        return _REFUSED
    # A refusal of the selections alone still shows the drawing, so it is not the reply's error.
    print(json.dumps({"error": None, **view.to_mapping()}))
    return 0


def _mesh(
    directory: str, case: str | None, relative_to: str | None, openfoam: bool, given: str | None
) -> int:
    """Read a case's mesh section; write it as VTK, and as a polyMesh with ``openfoam``.

    The section is the case file ``case``'s, or the JSON ``given`` (``{"mesh": {...}}``). Writes the
    cells as ``mesh.vtu`` and the boundary patches as ``patches.vtm`` (one block per patch, named by
    it), and prints the mesh file written and the mesh's boundary: each patch holding faces with its
    face count, and each patch group with the patches in it -- the names a case's ``boundaries``
    section may use, the polyMesh directory written (``null`` without ``openfoam``), and how the mesh
    was made, as its file records it (``null`` for a polyMesh). Relative paths are taken from
    ``relative_to``: by default the case file's directory, or here.
    """
    try:
        if case is not None:
            section = read_case_document(case)["mesh"]
            base = Path(case).parent if relative_to is None else Path(relative_to)
        else:
            section = json.loads(given)["mesh"]
            base = Path("." if relative_to is None else relative_to)
        source = mesh_source_from_mapping(section)
        mesh = source.read(base).validate()
        made = source.provenance(base)
        written = write_vtu(mesh, None, Path(directory) / "mesh.vtu")
        write_patches(mesh, None, Path(directory) / "patches.vtm")
        polymesh = write_openfoam(mesh, directory, overwrite=True) if openfoam else None
    except (*_REFUSALS, KeyError, json.JSONDecodeError, OSError) as error:
        print(json.dumps({"error": str(error)}))
        return _REFUSED
    patches = mesh.face_patches
    boundary = [
        {"name": name, "faces": patches.size(name)}
        for name in patches.names
        if patches.size(name) and patches.is_boundary_patch(name, mesh.face_cells)
    ]
    groups = {group: list(patches.group_members(group)) for group in patches.group_names}
    print(
        json.dumps(
            {
                "error": None,
                "mesh": str(written),
                "polymesh": None if polymesh is None else str(polymesh),
                "cells": mesh.n_cells,
                "dim": mesh.dim,
                "patches": boundary,
                "groups": groups,
                "made": made,
            }
        )
    )
    return 0


def _check(path: str) -> int:
    """Read and check ``path``, and say what it describes."""
    case_file = read_case(path)
    checked = case_file.check()
    spec, mesh = checked.spec, checked.mesh
    solver = solver_for(spec)
    print(
        f"{path}: a {type(spec.physics).__name__} case on {mesh.n_cells} cells ({mesh.dim}D), "
        f"patches {', '.join(spec.patch_conditions(mesh))}; solved by {type(solver).__name__}"
        f"{'' if spec.solver is not None else ' (the default)'}; checked."
    )
    return 0


def serve(requests, replies) -> None:
    """Answer each request line from ``requests`` with one reply line on ``replies``.

    Parameters
    ----------
    requests : iterable of str
        One JSON object per line: ``{"arguments": [...], "stdin": str or null}``.
    replies : text stream
        Where each answer is written, and flushed: ``{"status", "output", "errors"}`` -- the exit
        status the command would have had, and what it printed on standard output and error.

    It returns quietly when interrupted (Ctrl-C, run by hand in a terminal) or when ``replies`` has
    been closed by whatever was reading them: either way, nobody is left to answer.
    """
    try:
        for line in requests:
            if line.strip():
                replies.write(json.dumps(_answer(line)) + "\n")
                replies.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        return


def _answer(line: str) -> dict:
    """Run one request as the command would have run, capturing what it prints."""
    try:
        request = json.loads(line)
        arguments = [str(argument) for argument in request["arguments"]]
        given = request.get("stdin")
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return {"status": _REFUSED, "output": "", "errors": f"aquaflux serve: bad request: {error}"}
    if not arguments or arguments[0] not in SERVED:
        return {
            "status": _REFUSED,
            "output": "",
            "errors": f"aquaflux serve: answers only {', '.join(SERVED)}, not {arguments[:1]}",
        }
    output, errors = io.StringIO(), io.StringIO()
    held = sys.stdin
    sys.stdin = io.StringIO(given or "")
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = main(arguments)
    except SystemExit as exit:  # the argument parser's refusal
        status = exit.code if isinstance(exit.code, int) else _REFUSED
    except Exception:  # a failure in one request must not end the others
        errors.write(traceback.format_exc())
        status = 1
    finally:
        sys.stdin = held
    return {"status": status, "output": output.getvalue(), "errors": errors.getvalue()}


def _serve_process() -> int:
    """Serve this process's standard input, replying on a private copy of its standard output.

    Anything else that writes to standard output -- a library printing, compiled code writing to the
    file descriptor directly -- is sent to standard error instead, so it cannot corrupt a reply.
    """
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    # Closing flushes; if the reader has gone, there is nothing to flush to.
    with contextlib.suppress(BrokenPipeError), replies:
        serve(sys.stdin, replies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
