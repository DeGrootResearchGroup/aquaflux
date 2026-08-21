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

Delegation through a shared private tail is followed, transitively, so extracting one does not hide the
drift above it. That case is the whole reason the extra pass exists: pulling the duplicated body into a
single private builder is the right repair, but it leaves the *public surfaces* hand-copied while
removing the only mechanical signal that they drift -- the builders no longer construct a common class
directly, so a naive check reports clean on exactly the code that was half-fixed.

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


def _returned_calls(fn: ast.FunctionDef) -> set[str]:
    """Every name this function returns a call to — classes by naming convention, and private helpers."""
    called = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Call):
                func = sub.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name and (name[0].isupper() or name.startswith("_")):
                    called.add(name)
    return called


def _resolve_tails(calls: set[str], tails: dict[str, set[str]]) -> set[str]:
    """What a function ultimately builds: its own classes, plus what each private tail it hands to builds.

    Without this the tool is blind to the drift it exists to catch, in exactly the case where the
    duplication was half-fixed. Extracting the shared tail into one private builder is the right repair
    for the *bodies*, but the public surfaces above it stay hand-copied — and the extraction removes the
    only mechanical signal that they drift, because the builders no longer construct a common class
    directly. That is not hypothetical: four coupled-march builders delegating to one private tail
    reported clean here while a forward-solve stopping measure sat on one of them and a shift source on
    two, both of which were properties of the march rather than of any preconditioner.

    Resolved transitively, since a builder may hand off through more than one tail (a public builder to a
    per-family seam to the shared step). ``tails`` holds only **module-level private functions**, which
    are the ones a call site names unambiguously; a method is reached through an attribute whose owning
    class is not recoverable from the syntax tree, so following one would be a guess.

    Parameters
    ----------
    calls : set of str
        Names this function returns a call to — classes by naming convention, private helpers by prefix.
    tails : dict
        Module-level private function name -> the set of names it returns a call to.

    Returns
    -------
    set of str
        The class-like names, after substitution. Bounded by ``len(tails)`` passes, so mutual delegation
        terminates rather than spinning.
    """
    seen, pending = set(), set(calls)
    for _ in range(len(tails) + 1):
        private = {c for c in pending if c in tails} - seen
        if not private:
            break
        seen |= private
        pending = (pending - private) | {c for name in private for c in tails[name]}
    return {c for c in pending if c[0].isupper()}


def _builders(root: pathlib.Path):
    """Every builder-like function under ``root``, with its parameter set and what it constructs.

    Private tails are resolved per module: a builder delegating to ``_helper`` in the same file is
    credited with what ``_helper`` builds, so extracting a shared tail does not hide the surfaces above
    it (see :func:`_resolve_tails`).
    """
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))

        def entries(tree: ast.Module = tree):
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    yield "", node
                elif isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, ast.FunctionDef) and (
                            member.name in _FACTORY_METHODS or member.name.startswith("from_")
                        ):
                            yield node.name, member

        tails = {
            node.name: _returned_calls(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_")
        }
        for owner, fn in entries():
            made = _resolve_tails(_returned_calls(fn), tails)
            if made:
                params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} - {"self", "cls"}
                label = f"{owner}.{fn.name}" if owner else fn.name
                yield path, label, fn.lineno, params, made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="aquaflux", type=pathlib.Path)
    parser.add_argument("--shared", type=int, default=5, help="minimum shared parameters to report")
    args = parser.parse_args()

    found = 0
    for a, b in itertools.combinations(list(_builders(args.path)), 2):
        if a[0].parent != b[0].parent:  # same package: siblings, not unrelated namesakes
            continue
        # Public surfaces only. A private tail necessarily shares most of its parameters with every
        # builder that delegates to it -- that is the extraction working, not drift -- and reporting
        # each builder against each tail buries the pairs a reader has to judge. What this looks for is
        # drift between the surfaces callers actually see. (The label carries the owning class, so a
        # factory on a private class is filtered too.)
        if a[1].startswith("_") or b[1].startswith("_"):
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
