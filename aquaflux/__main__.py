"""The ``aquaflux`` command: check a case file, or run it.

::

    aquaflux check case.yaml            # read the file and check it against its mesh, in seconds
    aquaflux run case.yaml              # read, check, build, solve and write the results
    aquaflux run case.yaml --overwrite  # replace the results of an earlier run

``python -m aquaflux`` is the same command. ``run`` exits with status 0 when the solve converges and 1
when it stops short (its log, checkpoints and run record are still written); either command exits with
status 2 when the file is refused, with the reason on standard error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import yaml

from aquaflux.case import prepare_run, read_case, solver_for

__all__ = ["main"]

#: The exit status of a refused file, distinct from a solve that did not converge.
_REFUSED = 2


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
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "check":
            return _check(arguments.case)
        prepared = prepare_run(arguments.case, overwrite=arguments.overwrite)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError, yaml.YAMLError) as error:
        print(f"aquaflux: {error}", file=sys.stderr)
        return _REFUSED
    return 0 if prepared.run().converged else 1


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


if __name__ == "__main__":
    sys.exit(main())
