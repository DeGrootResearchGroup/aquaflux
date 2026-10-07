"""The ``aquaflux-ui`` command: the browser interface, opened on a run's results.

::

    aquaflux-ui results/              # a run's output directory, opened on Results
    aquaflux-ui fields.vtu            # or one VTK file (.vtu, .pvd, .vtp, .vtm)
    aquaflux-ui case.yaml             # a case file, opened on Setup
    aquaflux-ui                       # nothing open yet: Setup, to open a case file
    aquaflux-ui results/ --port 8080 --no-browser

It serves the page on this machine only (``127.0.0.1``) and opens it in the default browser. Stop it
with Ctrl-C. ``python -m aquaflux_ui`` is the same command.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``aquaflux-ui`` command.

    Parameters
    ----------
    argv : sequence of str, optional
        The arguments after the program name; unset, the command line's.

    Returns
    -------
    int
        The exit status: 0 when the server stops, 2 when what it was given cannot be opened.
    """
    parser = argparse.ArgumentParser(
        prog="aquaflux-ui",
        description="The aquaflux browser interface, served on this machine, opened on a run's results.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="a run's output directory, a .vtu/.pvd/.vtp/.vtm file, or a .yaml case file",
    )
    parser.add_argument(
        "--port", type=int, default=0, help="the port to serve on; 0 (the default) picks a free one"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="the address to serve on (default: this machine only)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="print the address instead of opening a browser"
    )
    arguments = parser.parse_args(argv)

    from .sources import open_source

    source, case = None, None
    if arguments.path is not None and Path(arguments.path).suffix.lower() in (".yaml", ".yml"):
        if not Path(arguments.path).is_file():
            print(f"aquaflux-ui: {arguments.path} does not exist.", file=sys.stderr)
            return 2
        case = arguments.path
    elif arguments.path is not None:
        try:
            source = open_source(arguments.path)
        except (FileNotFoundError, ValueError) as error:
            print(f"aquaflux-ui: {error}", file=sys.stderr)
            return 2

    # The page's dependencies are imported only once what it opens has been found.
    from .app import Workspace

    workspace = Workspace(source, case=case)
    workspace.server.start(
        port=arguments.port, host=arguments.host, open_browser=not arguments.no_browser
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
