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

#: Methods that build and return something, by convention, alongside module-level functions. A
#: ``@classmethod`` returning ``cls(...)`` is recognized structurally as well (see `_is_factory`), so
#: this list only has to cover factories whose construction the syntax tree cannot see.
_FACTORY_METHODS = ("build", "create", "make", "calibrated")


def _is_factory(member: ast.FunctionDef) -> bool:
    """Whether a class member builds and returns something, so its surface belongs in the report.

    Two ways to qualify, and the second is the one that matters. **By name** — ``build`` / ``create``
    / ``make`` / ``calibrated`` / ``from_*`` — covers a factory whose construction happens somewhere
    this tool cannot follow. **By shape** — a ``@classmethod`` whose body returns ``cls(...)`` — needs
    no naming convention at all, which is the point: a name list is blind to every factory nobody
    thought to add to it, and a pair this tool cannot see reports as a clean tree rather than as a gap.
    That blindness has twice let a builder surface drift unreported, so the structural test is the
    primary one and the name list is the fallback.
    """
    if member.name in _FACTORY_METHODS or member.name.startswith("from_"):
        return True
    decorated = any(
        isinstance(d, ast.Name) and d.id == "classmethod" for d in member.decorator_list
    )
    # Naming the owner "cls" makes `_returned_calls` report a `return cls(...)` as the literal name
    # "cls", which is the question being asked -- rather than repeating its walk over returns here.
    return decorated and "cls" in _returned_calls(member, owner="cls")


def _callee(func: ast.expr, owner: str, bound: dict[str, str]) -> str | None:
    """How a call's target is named for resolution, keeping whatever says which class owns it.

    Three spellings carry more than the bare name, and each was a blind spot until it was kept:

    * ``self.m(...)`` names ``Owner.m`` — the enclosing class's own method, whatever other classes
      also define an ``m``;
    * ``x.m(...)``, where ``x`` was bound by ``x = f(...)`` earlier in the function, names
      ``f().m``, which :func:`_reach` reads as "``m`` on whichever class ``f`` returns";
    * ``cls(...)`` inside a classmethod names the owning class.

    Every other attribute call keeps its bare name, as before, and resolves only if that name is
    defined once in the package.
    """
    if isinstance(func, ast.Name):
        return owner if func.id == "cls" and owner else func.id
    if not isinstance(func, ast.Attribute):
        return None
    receiver = func.value
    if isinstance(receiver, ast.Name):
        if receiver.id == "self" and owner:
            return f"{owner}.{func.attr}"
        if receiver.id in bound:
            return f"{bound[receiver.id]}().{func.attr}"
    return func.attr


def _call_bindings(fn: ast.FunctionDef, owner: str) -> dict[str, str]:
    """Locals bound by ``name = call(...)`` in this function, mapped to how that call is named."""
    bound: dict[str, str] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            name = _callee(node.value.func, owner, {})
            if name:
                bound[node.targets[0].id] = name
    return bound


def _returned_expressions(fn: ast.FunctionDef):
    """The expression of every ``return`` in the function that returns something."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            yield node.value


def _value_positions(expr: ast.expr):
    """The parts of a returned expression that are returned as they stand.

    The expression itself, either arm of a conditional expression, or an operand of ``and`` / ``or``.
    Anything else inside it -- a call's argument, an attribute read off it, an arithmetic operand -- is
    used to compute what is returned, and is not itself returned.
    """
    if isinstance(expr, ast.IfExp):
        yield from _value_positions(expr.body)
        yield from _value_positions(expr.orelse)
    elif isinstance(expr, ast.BoolOp):
        for operand in expr.values:
            yield from _value_positions(operand)
    else:
        yield expr


def _returned_locals(expr: ast.expr, bound: dict[str, str]) -> set[str]:
    """The calls behind the bound locals ``expr`` returns as they stand (see :func:`_value_positions`)."""
    return {
        bound[position.id]
        for position in _value_positions(expr)
        if isinstance(position, ast.Name) and position.id in bound
    }


def _returned_calls(fn: ast.FunctionDef, owner: str = "") -> set[str]:
    """Every name this function returns a call to, whatever it is spelled like.

    ``owner`` is the enclosing class, if any. A classmethod factory builds its own class by writing
    ``cls(...)``, which no naming convention can spot, so the class is credited by name instead —
    without it every ``@classmethod`` factory looks like it constructs nothing and drops out of the
    report entirely, which is the silent-blindness this tool is supposed to be the cure for.

    A local that is itself returned is followed to the call that bound it: ``step = build_it(...)``
    then ``return step if ... else None`` delegates to ``build_it`` exactly as ``return build_it(...)``
    does. ⚠️ **Only where the local is returned as it stands** — not where the return merely reads it.
    ``rate = measure(...)`` then ``return cls(sweeps=rate.value)`` builds ``cls``, not whatever
    ``measure`` built; crediting every mention once put six invented pairs in the package report, all
    of them sharing nothing but a measurement a calibration helper returns.

    **No filtering happens here**, deliberately: an attribute call such as ``globalization.step(...)``
    is a delegation exactly as a bare ``_helper(...)`` is, and it is spelled lowercase. Whether a name
    means anything is settled in :func:`_resolve_tails`, which resolves it against the definitions
    actually present and discards the rest — so a name that names nothing costs a dictionary lookup
    rather than a blind spot.
    """
    bound = _call_bindings(fn, owner)
    called = set()
    for expr in _returned_expressions(fn):
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Call):
                name = _callee(sub.func, owner, bound)
                if name:
                    called.add(name)
        called |= _returned_locals(expr, bound)
    return called


def _returned_values(fn: ast.FunctionDef, owner: str = "") -> set[str]:
    """The calls whose result this function returns as it stands: what its return value can *be*.

    Narrower than :func:`_returned_calls`, which also counts a call made only to compute an argument.
    ``return Session(Helper())`` delegates to both, but it returns a ``Session`` — so this is what a
    method receiver bound to the function's result is typed by (see :func:`_reach`).
    """
    bound = _call_bindings(fn, owner)
    values = set()
    for expr in _returned_expressions(fn):
        for position in _value_positions(expr):
            if isinstance(position, ast.Call):
                name = _callee(position.func, owner, bound)
                if name:
                    values.add(name)
        values |= _returned_locals(expr, bound)
    return values


def _key(name: str, tails: dict[str, set[str]]) -> str | None:
    """The ``tails`` entry a call name resolves to: itself, or a qualified method's unique bare name.

    The fallback covers a ``self.m(...)`` whose ``m`` is inherited, so it is not defined on the owning
    class under that name, but is defined exactly once in the package.
    """
    if name in tails:
        return name
    owner, dot, bare = name.rpartition(".")
    return bare if dot and owner and bare in tails else None


def _reach(
    calls: set[str],
    tails: dict[str, set[str]],
    classes: set[str],
    values: dict[str, set[str]],
    producers: frozenset = frozenset(),
) -> tuple[set[str], set[str]]:
    """Everything delegation from ``calls`` ends at, unfiltered, and the ``tails`` entries it followed.

    A receiver name ``f().m`` becomes ``C.m`` for each package class ``C`` that ``f`` returns as its
    value, so the method is resolved on the classes the receiver can actually be and on no other class
    that happens to define an ``m``. That typing follows ``values`` — what each function *returns* —
    rather than ``tails``, which also counts calls made only to build an argument: typed by ``tails``,
    ``return Session(Helper())`` would resolve ``session.m`` on ``Helper`` too, which is the union of
    definitions this resolution exists to avoid, arriving by another route. ``producers`` guards the
    recursion.

    ⚠️ **When ``f`` returns no package class, the receiver falls back to the bare ``m``** — resolved,
    as any attribute call is, only if ``m`` is defined once. Typing the receiver may add precision; it
    must never lose a delegation the bare name already followed. It did, on its first version: the
    coupled step returns ``globalization.step(...)`` after rebinding ``globalization`` from a helper
    that constructs nothing in the package, so the receiver resolved to no class, and the builder that
    had always reached its step classes through the unique ``Globalization.step`` stopped reaching them.
    """
    followed: set[str] = set()
    leaves: set[str] = set()
    pending = set(calls)
    for _ in range(len(tails) + 1):
        following: set[str] = set()
        for name in pending:
            producer, receiver, method = name.partition("().")
            if receiver:
                known = set()
                if producer not in producers:
                    made, _ = _reach({producer}, values, classes, values, producers | {producer})
                    known = made & classes
                following |= {f"{cls}.{method}" for cls in known} if known else {method}
                continue
            key = _key(name, tails)
            if key is None:
                leaves.add(name)
            elif key not in followed:
                followed.add(key)
                following |= tails[key]
        if not following:
            break
        pending = following
    return leaves, followed


def _resolve_tails(
    calls: set[str],
    tails: dict[str, set[str]],
    classes: set[str] = frozenset(),
    values: dict[str, set[str]] | None = None,
) -> tuple[set[str], set[str]]:
    """What a function ultimately builds, and the tails it passed through to get there.

    Without this the tool is blind to the drift it exists to catch, in exactly the case where the
    duplication was half-fixed. Extracting the shared tail into one private builder is the right repair
    for the *bodies*, but the public surfaces above it stay hand-copied — and the extraction removes the
    only mechanical signal that they drift, because the builders no longer construct a common class
    directly. That is not hypothetical: four coupled-march builders delegating to one private tail
    reported clean here while a forward-solve stopping measure sat on one of them and a shift source on
    two, both of which were properties of the march rather than of any preconditioner.

    Resolved transitively, since a builder may hand off through more than one tail (a public builder to a
    per-family seam to a shared step-builder on another object).

    ``tails`` holds the module's own functions first and then, as a fallback, every name defined
    **exactly once in the package** — module-level function or method alike. The uniqueness is what
    keeps this from being a guess: an attribute call names a method whose owning class the syntax tree
    cannot recover, so following it is sound only where one definition of that name exists. Ambiguous
    names are dropped rather than unioned, which under-credits instead of inventing a pair.

    ⚠️ **Following methods and other modules is not a refinement, it is the difference between seeing
    a family and not seeing it.** Resolution used to stop at same-module private functions, so any
    extraction that crossed a file — or that ended in ``config.build_the_thing(...)`` rather than in
    ``TheThing(...)`` — silently dropped every builder above it out of the report. That is the shape
    this tool exists to catch, arriving as a clean report.

    ⚠️ **An ambiguous method name is resolved through its receiver where the receiver's class is
    knowable, never by unioning its definitions.** A builder that opens a strategy object and builds
    through it (``session = open_session(...)``; ``return session._build(...)``) calls a name several
    classes define, and dropping it credited the builder with constructing nothing — so the coupled
    march's one remaining builder vanished from the report, which read as a clean tree. Unioning every
    ``_build`` would instead credit it with whatever an unrelated multigrid class builds. Resolving
    through what ``open_session`` returns is exact, and so is ``self.m(...)`` on its own class.

    Parameters
    ----------
    calls : set of str
        Every name this function returns a call to.
    tails : dict
        Function name, method name or ``Class.method`` -> the set of names it returns a call to.
    classes : set of str
        Every class defined in the package, which is what a receiver may resolve to.
    values : dict, optional
        The same keys -> the names each returns as its value (:func:`_returned_values`), which is what a
        receiver is typed by. Defaults to ``tails``.

    Returns
    -------
    tuple of set of str
        ``(made, followed)``: the class-like names delegation ends at, and the ``tails`` entries it
        passed through on the way -- which is what :func:`main` uses to tell a builder that *delegates
        to* another from one that duplicates it. Bounded by ``len(tails)`` passes, so mutual delegation
        terminates rather than spinning.
    """
    leaves, followed = _reach(calls, tails, set(classes), tails if values is None else values)
    return {c for c in leaves if c[0].isupper() and "." not in c}, followed


def _definitions(tree: ast.Module):
    """``(owning class or "", function)`` for every module-level function and every method."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield "", node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    yield node.name, member


def _package_tails(
    trees: dict[pathlib.Path, ast.Module],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Every name defined exactly once in the package, mapped to what it returns a call to.

    The cross-file half of :func:`_resolve_tails`. A name defined twice is dropped: two definitions
    make an attribute call ambiguous, and a report that invents a delegation is worse than one that
    misses it, since a reader cannot tell an invented pair from a real one.

    Each method is also entered as ``Class.method``, which is unambiguous however many classes define
    ``method`` — that is the key a receiver-typed call resolves through. A class name defined in two
    modules drops those entries too, for the same reason.

    Returns ``(tails, values)``: the same keys mapped to :func:`_returned_calls` and to
    :func:`_returned_values` respectively.
    """
    returns: dict[str, tuple[set[str], set[str]]] = {}
    duplicated: set[str] = set()
    for tree in trees.values():
        for owner, fn in _definitions(tree):
            entry = (_returned_calls(fn, owner), _returned_values(fn, owner))
            for name in (fn.name, f"{owner}.{fn.name}") if owner else (fn.name,):
                if name in returns:
                    duplicated.add(name)
                returns[name] = entry
    kept = {name: entry for name, entry in returns.items() if name not in duplicated}
    return (
        {name: calls for name, (calls, _) in kept.items()},
        {name: values for name, (_, values) in kept.items()},
    )


def _classes(trees: dict[pathlib.Path, ast.Module]) -> set[str]:
    """Every class defined at module level in the package: what a method receiver can resolve to."""
    return {
        node.name for tree in trees.values() for node in tree.body if isinstance(node, ast.ClassDef)
    }


def _builders(root: pathlib.Path):
    """Every builder-like function under ``root``, with its parameter set and what it constructs.

    Delegation is resolved against the module's own functions first and the package's unambiguous
    names second, so extracting a shared tail — into this file or another one, as a function or as a
    method — does not hide the surfaces above it (see :func:`_resolve_tails`).
    """
    trees = {
        path: ast.parse(path.read_text(), filename=str(path)) for path in sorted(root.rglob("*.py"))
    }
    package, package_values = _package_tails(trees)
    classes = _classes(trees)
    for path, tree in trees.items():
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        tails = {**package, **{node.name: _returned_calls(node) for node in functions}}
        values = {**package_values, **{node.name: _returned_values(node) for node in functions}}
        for owner, fn in _definitions(tree):
            if owner and not _is_factory(fn):
                continue
            made, followed = _resolve_tails(_returned_calls(fn, owner), tails, classes, values)
            if made:
                params = {a.arg for a in fn.args.args + fn.args.kwonlyargs} - {"self", "cls"}
                label = f"{owner}.{fn.name}" if owner else fn.name
                yield path, label, fn.lineno, params, made, fn.name, followed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="aquaflux", type=pathlib.Path)
    parser.add_argument("--shared", type=int, default=5, help="minimum shared parameters to report")
    args = parser.parse_args()

    found = 0
    for a, b in itertools.combinations(list(_builders(args.path)), 2):
        # No directory rule. One stood here ("same package: siblings, not unrelated namesakes") and it
        # discarded the flow-only and coupled builders of one march -- six shared parameters against a
        # threshold of five -- because `flow/` and `turbulence/` are different directories. Namesakes
        # are left to what remains below: a shared constructed class, a shared surface, and not
        # delegating to one another. The first is weaker than it reads -- a helper every builder calls
        # counts (`boundary.resolve()` builds a `BoundaryConditions` for all three assemblers) -- and a
        # delegation through an ambiguous name such as `build` is followed only where its receiver's
        # class is recoverable, so a few assembler pairs report that a reader has to judge.
        #
        # Public surfaces only. A private tail necessarily shares most of its parameters with every
        # builder that delegates to it -- that is the extraction working, not drift -- and reporting
        # each builder against each tail buries the pairs a reader has to judge. What this looks for is
        # drift between the surfaces callers actually see. (The label carries the owning class, so a
        # factory on a private class is filtered too.)
        if a[1].startswith("_") or b[1].startswith("_"):
            continue
        # Nor a builder beside a public builder it delegates to. A wrapper forwards its options to the
        # builder it calls, so the two share that surface by construction -- the delegation working, not
        # two copies drifting -- which is the reasoning that excludes a private tail above, for a public one.
        # A delegation followed through a typed receiver is recorded as `Class.method`, so the label is
        # compared as well as the bare name.
        if {a[1], a[5]} & b[6] or {b[1], b[5]} & a[6]:
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
