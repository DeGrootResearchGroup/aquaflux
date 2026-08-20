#!/usr/bin/env python3
"""Report sibling builders: functions that construct the SAME class with a heavily shared surface.

Two functions that build one class and take mostly the same parameters are one builder written twice.
The overlap is not a coincidence -- it is that object's configuration surface, duplicated -- and such
copies drift, because no single change ever looks wrong: each adds one keyword to one of them, so every
check scoped to "your change" is blind to it. What that costs is not tidiness. The coupled march's
k-positivity limit -- the fix for a solve that went non-finite from ``k < 0`` in two cells of 23040 --
was wired on one of *four* builders of the same step, and the other three marched without it.

A grep cannot do this: the package legitimately constructs several classes at more than one site
(``StepOutcome``, ``ShiftTerm``, ``SmoothedHierarchy``), and those are value objects built from local
data, not sibling builders. What separates the two is the *parameter overlap*, which needs the syntax
tree rather than a pattern.

Reports pairs in the same package that build a common class and share at least ``--shared`` parameters,
listing what each has that the other does not -- which is the drift, stated directly. Exits 0 always:
this is a report, and whether a pair is one builder or two genuinely different methods is a judgement.

Usage
-----
    tools/sibling_builders.py [path] [--shared N]
"""

from __future__ import annotations

import argparse
import ast
import itertools
import pathlib

#: Methods that build and return something, by convention, alongside module-level functions.
_FACTORY_METHODS = ("build", "create", "make")


def _constructed(fn: ast.FunctionDef) -> set[str]:
    """Class-like names this function returns a call to — its constructed types, by naming convention."""
    made = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Call):
                func = sub.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name and name[0].isupper():
                    made.add(name)
    return made


def _builders(root: pathlib.Path):
    """Every builder-like function under ``root``, with its parameter set and what it constructs."""
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))

        def record(fn: ast.FunctionDef, owner: str = "", path: pathlib.Path = path):
            params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} - {"self", "cls"}
            made = _constructed(fn)
            if made:
                label = f"{owner}.{fn.name}" if owner else fn.name
                yield path, label, fn.lineno, params, made

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                yield from record(node)
            elif isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, ast.FunctionDef) and (
                        member.name in _FACTORY_METHODS or member.name.startswith("from_")
                    ):
                        yield from record(member, node.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="aquaflux", type=pathlib.Path)
    parser.add_argument("--shared", type=int, default=5, help="minimum shared parameters to report")
    args = parser.parse_args()

    found = 0
    for a, b in itertools.combinations(list(_builders(args.path)), 2):
        if a[0].parent != b[0].parent:  # same package: siblings, not unrelated namesakes
            continue
        shared, common = a[3] & b[3], a[4] & b[4]
        if len(shared) < args.shared or not common:
            continue
        found += 1
        print(f"\n{len(shared)} shared parameters | both construct {sorted(common)}")
        print(f"  {a[0]}:{a[2]} {a[1]}\n      only here: {sorted(a[3] - b[3])}")
        print(f"  {b[0]}:{b[2]} {b[1]}\n      only here: {sorted(b[3] - a[3])}")
    print(
        f"\n{found} sibling-builder pair(s). Each 'only here' list is a capability one of them has and "
        f"the other does not — check that every one is a genuine property of that path."
        if found
        else "\nno sibling-builder pairs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
