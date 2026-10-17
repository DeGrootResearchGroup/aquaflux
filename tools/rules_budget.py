#!/usr/bin/env python3
"""Hold the auto-loaded guidance files to a size budget.

Every file under ``.claude/rules/`` loads into an agent's context whenever a file matching its
``paths:`` globs is read or edited, and ``CLAUDE.md`` loads into every session. Nothing reports what
they cost: a session simply starts with less room. Written rules to keep them small did not hold --
four files grew to 300-420 KB each while a rule to split at ~1,800 lines sat in one of them -- so the
budget is checked here instead, in the fast tier through ``tools/check_rules.sh``.

Three checks:

* **per file** -- ``CLAUDE.md`` and each rules file at most ``FILE_BUDGET`` bytes;
* **loaded together** -- for every tracked file, the rules whose globs match it at most
  ``LOADED_BUDGET`` bytes in total (that sum is what a session editing that file pays, on top of
  ``CLAUDE.md``);
* **scoped to something** -- a rules file whose globs match no tracked file never loads, so it is
  dead or mis-scoped.

``OVER_BUDGET`` lists files allowed above ``FILE_BUDGET`` for now, each with a ceiling it may not
exceed. The list may only shrink: a listed file that is back under budget, or whose ceiling sits more
than ``SLACK`` above its size, fails until the entry is removed or lowered.

Usage: ``tools/rules_budget.py [repo-root]``. Exits 1 on any failure, printing what to do.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

KB = 1024
FILE_BUDGET = 120 * KB
LOADED_BUDGET = 200 * KB
SLACK = 8 * KB

# path relative to the repository root -> ceiling in bytes. Lower a ceiling as the file shrinks.
OVER_BUDGET: dict[str, int] = {}

WHERE_IT_GOES = (
    "  Dated measurements, probe tables and investigation steps go to the matching\n"
    "  .claude/notes/<name>-log.md, leaving a current-status line; history of deleted code is\n"
    "  deleted (one line per trap that still applies); a closed direction gets one paragraph in\n"
    "  .claude/notes/solve-refuted-directions.md. See CLAUDE.md, 'Writing the design records'.\n"
)


def _globs(rule: Path) -> list[str]:
    lines = rule.read_text().splitlines()
    if not lines or lines[0] != "---":
        return []
    globs: list[str] = []
    in_paths = False
    for line in lines[1:]:
        if line == "---":
            break
        if line.startswith("paths:"):
            in_paths = True
        elif in_paths and line.lstrip().startswith("- "):
            globs.append(line.split("- ", 1)[1].strip().strip("\"'"))
        elif line and not line[0].isspace():
            in_paths = False
    return globs


def _regex(glob: str) -> re.Pattern[str]:
    out = ""
    i = 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif glob.startswith("**", i):
            out += ".*"
            i += 2
        elif glob[i] == "*":
            out += "[^/]*"
            i += 1
        elif glob[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(glob[i])
            i += 1
    return re.compile(out + r"\Z")


def check(root: Path) -> list[str]:
    """Return one message per failure; empty when the records are within budget."""
    rules_dir = root / ".claude" / "rules"
    rules = sorted(rules_dir.glob("*.md"))
    failures: list[str] = []

    sized = [root / "CLAUDE.md", *rules] if (root / "CLAUDE.md").exists() else rules
    for path in sized:
        name = str(path.relative_to(root))
        size = path.stat().st_size
        ceiling = OVER_BUDGET.get(name)
        if ceiling is None and size > FILE_BUDGET:
            failures.append(
                f"{name} is {size // KB} KB, over the {FILE_BUDGET // KB} KB budget for one file."
            )
        elif ceiling is not None and size > ceiling:
            failures.append(
                f"{name} is {size // KB} KB, over its {ceiling // KB} KB ceiling in OVER_BUDGET; it\n"
                "  may shrink, never grow."
            )
        elif ceiling is not None and size <= FILE_BUDGET:
            failures.append(f"{name} is back under budget: remove it from OVER_BUDGET.")
        elif ceiling is not None and ceiling - size > SLACK:
            failures.append(
                f"{name} is {size // KB} KB: lower its OVER_BUDGET ceiling to its size "
                f"(now {ceiling // KB} KB)."
            )
    for name in OVER_BUDGET:
        if not (root / name).exists():
            failures.append(f"OVER_BUDGET lists {name}, which does not exist: remove it.")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    matched: dict[Path, set[str]] = {}
    for rule in rules:
        patterns = [_regex(g) for g in _globs(rule)]
        matched[rule] = {f for f in tracked if any(p.match(f) for p in patterns)}
        if patterns and not matched[rule]:
            failures.append(
                f"{rule.relative_to(root)}: its paths: globs match no tracked file, so it never loads."
            )

    loads: dict[str, list[Path]] = {}
    for rule, files in matched.items():
        for f in files:
            loads.setdefault(f, []).append(rule)
    worst: dict[tuple[Path, ...], str] = {}
    for f, rs in loads.items():
        worst.setdefault(tuple(sorted(rs)), f)
    for combo, example in sorted(worst.items(), key=lambda kv: kv[1]):
        total = sum(r.stat().st_size for r in combo)
        if total > LOADED_BUDGET:
            names = ", ".join(f"{r.name} ({r.stat().st_size // KB} KB)" for r in combo)
            failures.append(
                f"editing {example} loads {total // KB} KB of rules, over the "
                f"{LOADED_BUDGET // KB} KB budget: {names}.\n"
                "  Shrink them, or narrow a paths: glob so they stop loading together."
            )
    return failures


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        root = Path(argv[1]).resolve()
    else:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
        )
        if top.returncode != 0:
            print("rules budget: not inside a work tree and no root given.", file=sys.stderr)
            return 1
        root = Path(top.stdout.strip())
    failures = check(root)
    for message in failures:
        print(f"rules budget: {message}", file=sys.stderr)
    if failures:
        print("rules budget: where content goes when a file is over budget:", file=sys.stderr)
        print(WHERE_IT_GOES, file=sys.stderr, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
