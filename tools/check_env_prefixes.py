#!/usr/bin/env python3
"""Fail when a ``validation/`` script reads an env-var prefix ``run_case.sh`` does not capture.

``validation/run_case.sh`` writes a run-file recording, among other things, every environment
variable a case was configured with -- by grepping the process environment for a fixed list of
name prefixes. That list is an allow-list: a script reading a prefix nobody added to it produces a
run-file with no ``env:`` line for that setting, which looks exactly like a run that used every
default. Four scripts were found in this state (``PROFILE_``, ``CONSISTENCY_``, ``FLOW_``,
``TAPER_``) before the prefixes were added to the grep -- and an allow-list cannot cover a prefix
that does not exist yet, so every new probe script starts unrecorded and stays that way until
someone happens to notice.

This scans every ``.py`` file under ``validation/`` for an ``os.environ`` read (``.get(...)``,
``[...]``, or ``os.getenv(...)``) naming a literal string, and checks the variable name against
the capture line's own pattern -- read out of ``run_case.sh`` itself, never duplicated here, so
the check and the thing it checks cannot drift apart the way the original gap did.

A "prefix", for the report below, is the leading run of ``[A-Z0-9]`` characters up to and
including the first underscore (``BFS3D_MESH`` -> ``BFS3D_``, ``ILU0_SWEEP_ARMS`` -> ``ILU0_``).
That is only how an uncovered name is *grouped and displayed*; whether a name counts as covered is
decided by matching it against the actual capture pattern, not by comparing single-word prefixes --
one existing prefix (``ILU0_SWEEP``) is itself two words, and a name is covered whenever the real
pattern matches it, however many words that alternative spells out.

Usage
-----
    tools/check_env_prefixes.py [scan_dir] [--run-case PATH]

``scan_dir`` defaults to ``<repo top>/validation``; ``--run-case`` defaults to
``<repo top>/validation/run_case.sh``. Exits 0 when every prefix found is covered, 1 otherwise.
Like ``tools/check_rules.sh`` this is a gate, not a report: the condition is mechanical and cheap
to leave broken.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

#: A literal env-var name read via ``os.environ.get("NAME", ...)``, ``os.environ["NAME"]``, or
#: ``os.getenv("NAME", ...)``. Only a plain quoted literal is recognized -- a name built at runtime
#: (f-string, concatenation, a variable) cannot be attributed to a prefix by source inspection, and
#: no script under validation/ does that today.
_ENV_READ = re.compile(
    r"""
    os\.environ\.get\(\s*['"](?P<name_get>[A-Z_][A-Z0-9_]*)['"]
    | os\.environ\[\s*['"](?P<name_item>[A-Z_][A-Z0-9_]*)['"]\s*\]
    | os\.getenv\(\s*['"](?P<name_getenv>[A-Z_][A-Z0-9_]*)['"]
    """,
    re.VERBOSE,
)

#: The run-file's capture line, e.g. ``env | grep -E '^(BFS3D|PITZ|...)_'``. The pattern between
#: the quotes is read out of the live file rather than copied here -- copying it is exactly the
#: duplication that let the original four prefixes fall out of step.
_CAPTURE_LINE = re.compile(r"env \| grep -E '([^']+)'")

#: The leading word of an env-var name, for grouping an uncovered read in the report. Purely
#: cosmetic: coverage itself is decided by matching the *whole* name against the pattern above.
_LEADING_WORD = re.compile(r"^[A-Z0-9]+_")


def _repo_top() -> pathlib.Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return pathlib.Path(result.stdout.strip())


def _covered_pattern(run_case: pathlib.Path) -> re.Pattern[str] | None:
    match = _CAPTURE_LINE.search(run_case.read_text())
    return re.compile(match.group(1)) if match else None


def _env_reads(scan_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Every distinct literal env-var name read under ``scan_dir``, mapped to one file it appears in."""
    names: dict[str, pathlib.Path] = {}
    for path in sorted(scan_dir.rglob("*.py")):
        for match in _ENV_READ.finditer(path.read_text()):
            name = match.group("name_get") or match.group("name_item") or match.group("name_getenv")
            names.setdefault(name, path)
    return names


def _prefix_of(name: str) -> str:
    match = _LEADING_WORD.match(name)
    return match.group(0) if match else name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scan_dir",
        nargs="?",
        type=pathlib.Path,
        default=None,
        help="directory to scan for .py files (default: <repo top>/validation)",
    )
    parser.add_argument(
        "--run-case",
        type=pathlib.Path,
        default=None,
        help="path to run_case.sh whose capture line defines covered prefixes "
        "(default: <repo top>/validation/run_case.sh)",
    )
    args = parser.parse_args()

    scan_dir, run_case = args.scan_dir, args.run_case
    if scan_dir is None or run_case is None:
        top = _repo_top()
        if top is None:
            print("env-prefixes: not inside a work tree and no path given.", file=sys.stderr)
            return 1
        scan_dir = scan_dir if scan_dir is not None else top / "validation"
        run_case = run_case if run_case is not None else top / "validation" / "run_case.sh"

    if not scan_dir.is_dir():
        print(f"env-prefixes: no such directory: {scan_dir}", file=sys.stderr)
        return 1
    if not run_case.is_file():
        print(f"env-prefixes: no such file: {run_case}", file=sys.stderr)
        return 1

    covered = _covered_pattern(run_case)
    if covered is None:
        print(
            f"env-prefixes: could not find the env-prefix capture line in {run_case}.",
            file=sys.stderr,
        )
        return 1

    names = _env_reads(scan_dir)

    # Guard the guard: a broken glob or a moved directory would otherwise pass by scanning nothing,
    # which looks exactly like a tree where every read happens to be covered.
    if not names:
        print(
            f"env-prefixes: examined no env-var reads under {scan_dir} -- this check has stopped seeing anything.",
            file=sys.stderr,
        )
        return 1

    uncovered: dict[str, list[tuple[str, pathlib.Path]]] = {}
    for name, path in sorted(names.items()):
        if covered.match(name):
            continue
        uncovered.setdefault(_prefix_of(name), []).append((name, path))

    if not uncovered:
        return 0

    for prefix, entries in sorted(uncovered.items()):
        example_name, example_path = entries[0]
        print(
            f"env-prefixes: {prefix} is not covered by {run_case}'s capture line "
            f"(e.g. {example_name} in {example_path}).",
            file=sys.stderr,
        )
    print(
        "       Add the missing prefix(es) to the `grep -E` alternation in run_case.sh so the "
        "run-file records what these scripts were configured with.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
