"""The ``aquaflux`` command: check a case file, or run it.

::

    aquaflux check case.yaml            # read the file and check it against its mesh, in seconds
    aquaflux run case.yaml              # read, check, build, solve and write the results
    aquaflux run case.yaml --overwrite  # replace the results of an earlier run

``python -m aquaflux`` is the same command. ``run`` exits with status 0 when the solve converges and 1
when it stops short (its log, checkpoints and run record are still written); either command exits with
status 2 when the file is refused, with the reason on standard error.

Three more commands are for a program editing case files -- the browser interface -- and speak JSON
(JavaScript Object Notation) on their standard output:

    aquaflux schema                     # what a case file may hold: every kind, field and choice
    aquaflux show case.yaml             # the file as aquaflux reads it, and why it is refused if it is
    aquaflux write case.yaml < case.json  # write the case given on standard input, checked first
    aquaflux write copy.yaml --relative-to original/ < case.json  # re-base its relative paths
    aquaflux mesh out/ --relative-to case_dir/ < mesh.json  # read or generate a case's mesh and its patches, as VTK
    aquaflux serve                      # answer any of these, one per line, without restarting

``show``, ``write`` and ``mesh`` print ``{"error": null, ...}`` on success and ``{"error": "<reason>"}`` with exit
status 2 when the file, or the case, is refused.

``serve`` is for a program that asks many of these in turn: starting Python and importing the solver
takes seconds, and the commands themselves take a fraction of that. It reads one request per line --
``{"arguments": ["show", "case.yaml"], "stdin": null}`` -- and answers each with one line,
``{"status": 0, "output": "...", "errors": "..."}``: what the command would have exited with and
printed. It stops at the end of its input.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import yaml

from aquaflux.case import (
    case_schema,
    case_spec_from_mapping,
    mesh_source_from_mapping,
    prepare_run,
    read_case,
    read_case_document,
    solver_for,
    write_case,
)
from aquaflux.case.paths import relocated
from aquaflux.io import write_patches, write_vtu

__all__ = ["main"]

#: The exit status of a refused file, distinct from a solve that did not converge.
_REFUSED = 2

#: The commands ``serve`` answers: the ones that read and write case files, each finished in seconds.
SERVED = ("schema", "show", "write", "mesh", "check")


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
    commands.add_parser("schema", help="print what a case file may hold, as JSON")
    show = commands.add_parser("show", help="print a case file as aquaflux reads it, as JSON")
    show.add_argument("case", help="the case file")
    write = commands.add_parser(
        "write", help="write the case given as JSON on standard input to a case file"
    )
    write.add_argument("case", help="the case file to write; replaced if it exists")
    write.add_argument(
        "--relative-to",
        metavar="DIR",
        help="the directory the case's relative paths are relative to now; they are re-based onto "
        "the written file's directory (default: that directory, so nothing moves)",
    )
    mesh = commands.add_parser(
        "mesh", help="read or generate the mesh section given as JSON on standard input, as VTK"
    )
    mesh.add_argument("directory", help="where to write mesh.vtu")
    mesh.add_argument(
        "--relative-to",
        metavar="DIR",
        default=".",
        help="the directory the mesh section's relative paths are relative to (default: here)",
    )
    commands.add_parser(
        "serve", help=f"answer {', '.join(SERVED)} requests, one JSON line each, until input ends"
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "serve":
        return _serve_process()
    if arguments.command == "mesh":
        return _mesh(arguments.directory, sys.stdin.read(), arguments.relative_to)
    if arguments.command == "schema":
        print(json.dumps(case_schema()))
        return 0
    if arguments.command == "show":
        return _show(arguments.case)
    if arguments.command == "write":
        return _write(arguments.case, sys.stdin.read(), arguments.relative_to)
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


def _show(path: str) -> int:
    """Print ``path``'s document as JSON, with the reason its case is refused, if it is."""
    try:
        document = read_case_document(path)
    except (*_REFUSALS, OSError) as error:
        print(json.dumps({"error": f"{path}: {error}"}))
        return _REFUSED
    try:
        case_spec_from_mapping(document)
        error = None
    except (ValueError, TypeError) as refusal:
        error = str(refusal)
    print(json.dumps({"error": error, "case": document}))
    return 0


def _write(path: str, given: str, relative_to: str | None = None) -> int:
    """Write the case in the JSON ``given`` (``{"case": {...}}``) to ``path``, if it is a case.

    Its relative paths are re-based from ``relative_to`` onto ``path``'s directory, so a copy saved
    elsewhere still finds its mesh.
    """
    try:
        spec = case_spec_from_mapping(json.loads(given)["case"])
        if relative_to is not None:
            spec = relocated(spec, Path(relative_to).resolve(), Path(path).resolve().parent)
        write_case(spec, path)
    except (*_REFUSALS, KeyError, json.JSONDecodeError, OSError) as error:
        print(json.dumps({"error": str(error)}))
        return _REFUSED
    print(json.dumps({"error": None, "path": path}))
    return 0


def _mesh(directory: str, given: str, relative_to: str) -> int:
    """Read or generate the mesh section in the JSON ``given`` (``{"mesh": {...}}``); write it as VTK.

    Writes the cells as ``mesh.vtu`` and the boundary patches as ``patches.vtm`` (one block per patch,
    named by it), and prints the mesh file written and the mesh's boundary: each patch holding faces
    with its face count, and each patch group with the patches in it -- the names a case's
    ``boundaries`` section may use.
    """
    try:
        source = mesh_source_from_mapping(json.loads(given)["mesh"])
        mesh = source.read(Path(relative_to)).validate()
        written = write_vtu(mesh, None, Path(directory) / "mesh.vtu")
        write_patches(mesh, None, Path(directory) / "patches.vtm")
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
                "cells": mesh.n_cells,
                "dim": mesh.dim,
                "patches": boundary,
                "groups": groups,
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
