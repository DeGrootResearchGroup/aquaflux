#!/usr/bin/env python3
"""Select the test files a change can reach, from the import graph and a declared table.

A session in a small container cannot afford the whole fast tier before every push, and the cheaper
substitutes it reaches for -- pushing unvalidated, or picking "the tests that look related" by hand --
fail silently in the narrow direction. This selects instead, and it is built to fail WIDE:

* **By import.** Every test module that imports, directly or transitively, a module the change touched.
  An import ``from aquaflux.solve import X`` is followed to the submodule that *defines* ``X``, through
  the package ``__init__``'s own re-exports, rather than to every submodule the ``__init__`` loads --
  otherwise a test touching one name of a package depends on all of it, and the selection is most of
  the suite for any change at all. The importer still depends on the ``__init__``'s *own* code, and
  every module depends on its ancestor packages' ``__init__``, so a change to ``aquaflux/__init__.py``
  selects everything. Imports inside functions and under ``TYPE_CHECKING`` count: over-selecting is the
  safe direction. A shared leaf such as ``vectors.py`` widens to everything that imports it, which is
  most of the suite, and that is the correct answer.
* **By declaration.** A test that reaches a file *without* importing it -- reading the package source as
  text, running a shell script, loading a case file, putting ``validation/`` on ``sys.path``, starting
  ``python -m aquaflux`` -- is invisible to an import walk. Each one is listed in :data:`REACHES` with
  the paths it reaches, and ``tests/unit/test_scope_reaches.py`` fails if a test module shows such a
  reach without an entry, so the table cannot quietly fall behind the suite.
* **Widening.** A changed path neither reaches -- a configuration file, a script nothing declares --
  selects the WHOLE tier, and says which path did it. Only the paths in :data:`REACHES_NO_TEST`
  (prose read by no test) select nothing, and they say so too.

What it cannot see, recorded so its silence is not read as a clean selection: a module-level error in
a submodule that breaks *every* importer of its package (the package ``__init__`` loads every
submodule, so an import of any name fails), when nothing that imports the broken module's own names is
selected; and a reach a test makes in a way the census does not recognize. The full gate in continuous
integration is the backstop for both, which is why this is a pre-push check and not the gate.

Usage
-----
    tools/scope_tests.py [--base REF] [--root DIR] [--write FILE] [PATH ...]

With no ``PATH`` the change is everything that differs from ``git merge-base HEAD REF`` (default
``origin/main``): committed, staged, unstaged and untracked. ``--write`` records the selection for a
caller: the first line is ``all``, ``some`` or ``none``, and the selected test files follow, one per
line. The report goes to standard output. Exits 0 when it produced a selection, 2 when it could not
(an unknown base, a tree that is not a git checkout).
"""

from __future__ import annotations

import argparse
import ast
import collections
import fnmatch
import pathlib
import subprocess
import sys
import textwrap
from typing import NamedTuple

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The directories whose Python files form the import graph: the two packages and the test tree.
GRAPH_DIRECTORIES = ("aquaflux", "aquaflux_ui", "tests")

#: Each test module that reaches a file without importing it: module path -> (globs it reaches, how).
#: A ``tests/support`` helper may be listed too, and its reach is then charged to every test importing
#: it. A glob is matched against repository-relative paths with ``fnmatch`` (so ``*`` crosses ``/``).
#: ``tests/unit/test_scope_reaches.py`` requires an entry for every test module (or helper it imports)
#: that builds a path from ``__file__``, names a repository path, starts a subprocess, edits
#: ``sys.path`` or imports by name, and requires every glob here to match a file that exists, so an
#: entry that has gone stale fails rather than lingers.
REACHES: dict[str, tuple[tuple[str, ...], str]] = {
    # -- tests that read the package source as text, or import every module by name
    "tests/unit/test_layering.py": (
        ("aquaflux/*", "tools/scope_tests.py"),
        "parses every module's imports; loads its import walk from the selector",
    ),
    "tests/unit/test_vocabulary.py": (("aquaflux/*",), "parses every module's parameter names"),
    "tests/unit/test_declared_inputs.py": (("aquaflux/*",), "imports every module by name"),
    "tests/unit/test_preconditioner_spec_mapping.py": (
        ("aquaflux/*",),
        "imports every subpackage by name to enumerate its preconditioners",
    ),
    "tests/unit/test_validation_api.py": (
        ("aquaflux/*", "validation/*"),
        "checks every study harness's imports against the package surface",
    ),
    "tests/unit/test_solve_api.py": (
        ("aquaflux/*", "validation/*.py", "README.md", "docs/*.md"),
        "checks the package surface against the harnesses and the prose that names it",
    ),
    "tests/unit/test_sibling_builders.py": (
        ("tools/sibling_builders.py", "aquaflux/*"),
        "runs the report over the package",
    ),
    "tests/ui/test_independence.py": (
        ("aquaflux/*", "aquaflux_ui/*"),
        "parses both packages' imports to check neither reaches the other",
    ),
    # -- tests that read the test tree itself
    "tests/unit/test_optional_dependency_skips.py": (("tests/*test_*.py",), "the skip census"),
    "tests/unit/test_distributed_marker.py": (
        ("tests/*test_*.py",),
        "the distributed-marker census",
    ),
    "tests/unit/test_scope_reaches.py": (
        ("tests/*test_*.py", "tests/support/*.py", "tools/scope_tests.py"),
        "the census of this table",
    ),
    # -- tests of the repository's scripts
    "tests/unit/test_fastgate.py": (
        (
            "tools/fastgate.sh",
            "tools/check_hooks.sh",
            "tools/scope_tests.py",
            "validation/run_case.sh",
        ),
        "runs the test runner, which calls the other three",
    ),
    "tests/unit/test_check_hooks.py": (("tools/check_hooks.sh", ".githooks/*"), "runs the checker"),
    "tests/unit/test_check_rules.py": (
        ("tools/check_rules.sh", "tools/rules_budget.py", ".claude/rules/*", "CLAUDE.md"),
        "runs the checker and the size budget over this repository's guidance files",
    ),
    "tests/unit/test_check_env_prefixes.py": (
        ("tools/check_env_prefixes.py", "validation/*.py", "validation/run_case.sh"),
        "runs the checker over the study harnesses",
    ),
    "tests/unit/test_run_case.py": (("validation/run_case.sh",), "runs the case runner"),
    "tests/unit/test_scope_tests.py": (("tools/scope_tests.py",), "loads the selector"),
    # -- tests that read case files and data under validation/
    "tests/unit/test_case_file.py": (("validation/*.yaml",), "reads the shipped case files"),
    "tests/unit/test_shipped_case_paths.py": (
        ("validation/*",),
        "reads the shipped case files and checks the git index for every path they name",
    ),
    "tests/unit/test_case_solver.py": (
        ("validation/*",),
        "reads the shipped case files and runs harnesses in their case directories",
    ),
    "tests/unit/test_case_editing_commands.py": (
        ("validation/*.yaml",),
        "edits a shipped case file",
    ),
    "tests/unit/test_case_radiation.py": (
        ("validation/ray_effects_room/*",),
        "reads a lamp's IES file",
    ),
    "tests/unit/test_channel_case_files.py": (
        ("validation/*.yaml",),
        "reads the channel case files",
    ),
    "tests/unit/test_cad_step.py": (
        ("validation/uvreactor_openfoam/of_case/*",),
        "reads the reactor's STEP drawing",
    ),
    "tests/integration/test_bfs3d_species.py": (
        ("validation/bfs3d_species/*", "validation/bfs3d_openfoam/*"),
        "imports the species harness from validation/ and reads the OpenFOAM case",
    ),
    "tests/unit/test_compile_timeline.py": (
        ("validation/pitzdaily_openfoam/compile_timeline.py",),
        "loads the compile-timeline harness by path",
    ),
    "tests/ui/test_case_form.py": (
        ("validation/*.yaml",),
        "builds forms from the shipped case files",
    ),
    "tests/ui/test_run_monitor.py": (
        ("validation/pitzdaily_openfoam/*",),
        "reads the pitzDaily case",
    ),
    "tests/ui/test_run_section.py": (
        ("validation/pitzdaily_openfoam/*",),
        "reads the pitzDaily case",
    ),
    # -- tests that read the mesh fixtures
    "tests/unit/test_openfoam_fields.py": (("tests/fixtures/*",), "reads a polyMesh fixture"),
    "tests/unit/test_openfoam_reader.py": (("tests/fixtures/*",), "reads the polyMesh fixtures"),
    "tests/unit/test_openfoam_mesh_writer.py": (
        ("tests/fixtures/*",),
        "writes back a polyMesh fixture OpenFOAM generated",
    ),
    "tests/unit/test_patch_groups.py": (("tests/fixtures/*",), "reads a polyMesh fixture"),
    "tests/integration/test_case_run.py": (
        ("tests/fixtures/*",),
        "runs a case on a polyMesh fixture",
    ),
    # -- test helpers: a reach declared here is charged to every test that imports the helper
    "tests/support/polymesh.py": (("tests/fixtures/*",), "builds meshes from a polyMesh fixture"),
    "tests/support/devices.py": (
        (),
        "starts the child interpreter a test's program runs in; the program's own imports are read "
        "from the string that holds it",
    ),
    # -- reaches the import graph already covers, declared so the census does not ask again
    "tests/unit/test_x64_import.py": (
        ("aquaflux/__init__.py",),
        "imports the package in a subprocess, never in its own process",
    ),
    "tests/unit/test_compilation_cache.py": (
        (),
        "its subprocesses import only what it imports itself",
    ),
    "tests/unit/test_case_run.py": (
        (),
        "`python -m aquaflux` runs __main__, which it imports itself",
    ),
    "tests/ui/test_sources.py": ((), "names `aquaflux` only as a key of a run record"),
    "tests/unit/test_compiled_residual_census.py": (
        (),
        "uses the repository root only as the case directory of a structured grid",
    ),
}

#: Paths that no test reads. A change confined to these selects no tests and says so. A path a test
#: in :data:`REACHES` declares (``README.md`` and ``docs/*.md``, for one) still selects it.
REACHES_NO_TEST: tuple[str, ...] = (
    "docs/*",
    ".claude/*",
    "CLAUDE.md",
    "README.md",
    "LICENSE",
    ".github/*",
    ".readthedocs.yaml",
    ".gitignore",
    ".test_durations",
    "tools/build_docs.sh",
    "tools/canadian-spelling.dict",
)


class ImportStatement(NamedTuple):
    """One ``import`` or ``from ... import`` statement.

    Attributes
    ----------
    line : int
        Its line number.
    module : str or None
        The module named after ``from`` (``None`` for ``from . import x``), or the module of a plain
        ``import``.
    level : int
        The number of leading dots of a relative import; ``0`` for an absolute one.
    names : tuple of str or None
        The names a ``from`` import binds (``"*"`` for a star import); ``None`` for a plain ``import``.
    """

    line: int
    module: str | None
    level: int
    names: tuple[str, ...] | None


def imports(path: pathlib.Path, programs: bool = False) -> list[ImportStatement]:
    """Every import statement in ``path``, at any depth (inside functions too).

    With ``programs``, also the imports of every string in it that is itself a Python program: a test
    that runs its code in a fresh interpreter (``python -c <source>``) holds that code as a string, and
    what the child imports is what the test reaches. Each such import carries the string's line.
    """
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            found.append(ImportStatement(node.lineno, node.module, node.level, names))
        elif isinstance(node, ast.Import):
            found.extend(ImportStatement(node.lineno, a.name, 0, None) for a in node.names)
        elif programs and isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.extend(s._replace(line=node.lineno) for s in _program_imports(node.value))
    return found


def _program_imports(text: str) -> list[ImportStatement]:
    """The absolute imports of ``text`` if it parses as a Python program, else none."""
    if "import" not in text:
        return []
    try:
        tree = ast.parse(textwrap.dedent(text))
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            found.append(ImportStatement(0, node.module, 0, tuple(a.name for a in node.names)))
        elif isinstance(node, ast.Import):
            found.extend(ImportStatement(0, a.name, 0, None) for a in node.names)
    return found


def absolute(package: list[str], module: str | None, level: int) -> list[str]:
    """The dotted parts of the module an import names, a relative import resolved.

    Parameters
    ----------
    package : list of str
        The dotted parts of the package the importing file sits in (its directory, for an
        ``__init__.py`` as for any other module).
    module, level
        The import's module and relative level, as :class:`ImportStatement` holds them.
    """
    if level == 0:
        return (module or "").split(".")
    return [*package[: len(package) - (level - 1)], *(module.split(".") if module else [])]


def module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    """The dotted module name of ``path`` (relative to ``root``); a package is named by its directory."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


class ImportGraph:
    """Which module depends on which, with imported names followed to the module that defines them.

    Parameters
    ----------
    root : pathlib.Path
        The repository root.
    directories : sequence of str
        The top-level directories whose ``.py`` files are the graph's modules.
    """

    def __init__(
        self, root: pathlib.Path, directories: tuple[str, ...] = GRAPH_DIRECTORIES
    ) -> None:
        self.root = root
        self.paths: dict[str, pathlib.Path] = {}
        for directory in directories:
            if not (root / directory).is_dir():
                continue
            for path in sorted((root / directory).rglob("*.py")):
                if "__pycache__" not in path.parts:
                    self.paths[module_name(path, root)] = path
        self.packages = {m for m, p in self.paths.items() if p.name == "__init__.py"}
        self._statements = {
            m: imports(p, programs=m.startswith("tests.")) for m, p in self.paths.items()
        }
        self._exports = {p: self._exports_of(p) for p in self.packages}
        self.edges: dict[str, set[str]] = {m: self._edges_of(m) for m in self.paths}

    def _package_parts(self, module: str) -> list[str]:
        return module.split(".") if module in self.packages else module.split(".")[:-1]

    def _exports_of(self, package: str) -> dict[str, str]:
        """``name -> module`` for each name the package's ``__init__`` imports from inside itself."""
        exported = {}
        for statement in self._statements[package]:
            if statement.names is None:
                continue
            target = ".".join(absolute(self._package_parts(package), *statement[1:3]))
            if not target.startswith(package + ".") and target != package:
                continue
            for name in statement.names:
                if f"{target}.{name}" in self.paths:
                    exported[name] = f"{target}.{name}"
                elif target != package:
                    exported[name] = target
        return exported

    def _defining(self, package: str, name: str, seen: frozenset[str] = frozenset()) -> str | None:
        """The module ``name`` comes from when imported from ``package``, re-exports followed."""
        target = self._exports.get(package, {}).get(name)
        if target in self.packages and target not in seen and name in self._exports.get(target, {}):
            return self._defining(target, name, seen | {package})
        return target

    def _edges_of(self, module: str) -> set[str]:
        """The modules ``module`` depends on: one hop, imported names resolved."""
        parts = module.split(".")
        edges = {".".join(parts[:i]) for i in range(1, len(parts))} & set(self.paths)
        if module.startswith("tests.") and pathlib.Path(self.paths[module]).name.startswith(
            "test_"
        ):
            # pytest loads every conftest.py from the rootdir down to a test module's directory
            edges |= {
                ".".join([*parts[:i], "conftest"])
                for i in range(1, len(parts))
                if ".".join([*parts[:i], "conftest"]) in self.paths
            }
        for statement in self._statements[module]:
            target = ".".join(absolute(self._package_parts(module), *statement[1:3]))
            if (
                module in self.packages
                and statement.names is not None
                and (target == module or target.startswith(module + "."))
            ):
                continue  # a re-export: followed per name by whoever imports it from here
            edges.add(target)  # kept even when it is not a file: a deleted module's importers
            if statement.names is None:
                if target in self.packages:  # `import pkg` then `pkg.anything`: all of it
                    edges |= set(self._exports[target].values())
                continue
            for name in statement.names:
                if f"{target}.{name}" in self.paths:
                    edges.add(f"{target}.{name}")
                elif name == "*":
                    edges |= set(self._exports.get(target, {}).values())
                elif (defining := self._defining(target, name)) is not None:
                    edges.add(defining)
        edges.discard(module)
        return edges

    def dependents(self, changed: set[str]) -> dict[str, list[str]]:
        """Every module reaching one in ``changed``, each with a shortest chain back to it."""
        reverse = collections.defaultdict(set)
        for source, targets in self.edges.items():
            for target in targets:
                reverse[target].add(source)
        chains = {module: [module] for module in changed}
        queue = collections.deque(sorted(changed))
        while queue:
            module = queue.popleft()
            for importer in sorted(reverse[module]):
                if importer not in chains:
                    chains[importer] = [importer, *chains[module]]
                    queue.append(importer)
        return chains


def is_test_file(relative: str) -> bool:
    """Whether a repository-relative path is a test module pytest collects."""
    path = pathlib.PurePosixPath(relative)
    return path.parts[0] == "tests" and path.name.startswith("test_") and path.suffix == ".py"


class Selection(NamedTuple):
    """What a change selects.

    Attributes
    ----------
    everything : bool
        Whether the change widened the selection to the whole tier.
    files : dict
        Each selected test file (repository-relative) -> why, as a short human-readable string.
    widened_by : list of str
        The changed paths that widened it to the whole tier.
    unreached : list of str
        The changed paths that selected nothing because no test reads them (:data:`REACHES_NO_TEST`).
    """

    everything: bool
    files: dict[str, str]
    widened_by: list[str]
    unreached: list[str]


def matches(path: str, globs: tuple[str, ...]) -> bool:
    """Whether a repository-relative path matches any of ``globs`` (``fnmatch``: ``*`` crosses ``/``)."""
    return any(fnmatch.fnmatchcase(path, glob) for glob in globs)


def select(
    changed: list[str],
    graph: ImportGraph,
    reaches: dict[str, tuple[tuple[str, ...], str]] = REACHES,
    reaches_no_test: tuple[str, ...] = REACHES_NO_TEST,
) -> Selection:
    """The test files the repository-relative ``changed`` paths can reach.

    A changed ``.py`` file inside the import graph selects the test modules that depend on it; one
    outside it, and any other file, selects the test modules that declare reaching it. A path that
    selects nothing either way widens the selection to everything, unless it is one no test reads.
    """
    in_graph = {}
    for path in changed:
        pure = pathlib.PurePosixPath(path)
        if pure.suffix == ".py" and pure.parts[0] in GRAPH_DIRECTORIES:
            in_graph[path] = module_name(graph.root / path, graph.root)
    chains = graph.dependents(set(in_graph.values()))
    files: dict[str, str] = {}
    widened_by, unreached = [], []
    for path in changed:
        reached = []
        if path in in_graph:
            for module, chain in chains.items():
                if chain[-1] != in_graph[path]:
                    continue
                test = graph.paths.get(module)
                if test is not None and is_test_file(test.relative_to(graph.root).as_posix()):
                    relative = test.relative_to(graph.root).as_posix()
                    reached.append(relative)
                    hops = " <- ".join(chain[1:4]) + (" <- ..." if len(chain) > 4 else "")
                    files.setdefault(relative, f"imports {hops}" if hops else "changed")
        for declared, (globs, _) in reaches.items():
            if not matches(path, globs):
                continue
            # A declaring test is selected; a declaring helper (a tests/support module) selects every
            # test that imports it.
            tests = [declared] if is_test_file(declared) else _tests_importing(declared, graph)
            for test in tests:
                if (graph.root / test).is_file():  # not one this change deletes
                    reached.append(test)
                    files.setdefault(test, f"declares reaching {path}")
        if reached:
            continue
        if matches(path, reaches_no_test) or path in in_graph:
            # A module nothing imports is reached by no test: a true empty answer, not a hole in the
            # graph -- but it is reported, rather than left to read as covered.
            unreached.append(path)
        else:
            widened_by.append(path)
    return Selection(bool(widened_by), files, widened_by, unreached)


def _tests_importing(helper: str, graph: ImportGraph) -> list[str]:
    """The test files that import the test-tree module at ``helper`` (repository-relative)."""
    chains = graph.dependents({module_name(graph.root / helper, graph.root)})
    tests = [graph.paths[m].relative_to(graph.root).as_posix() for m in chains if m in graph.paths]
    return sorted(t for t in tests if is_test_file(t))


def changed_paths(root: pathlib.Path, base: str) -> list[str]:
    """Every path differing from the merge base with ``base``: committed, staged, unstaged, untracked."""

    def git(*args: str) -> list[str]:
        out = subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout
        return [line for line in out.splitlines() if line]

    merge_base = git("merge-base", "HEAD", base)[0]
    # --no-renames: a rename is its old path deleted and its new path added, and both matter --
    # whatever still imports the old name is exactly what the change can break.
    paths = git("diff", "--name-only", "--no-renames", merge_base)
    paths += git("ls-files", "--others", "--exclude-standard")
    return sorted(set(paths))


def report(selection: Selection, changed: list[str], universe: int) -> str:
    """The human-readable account of a selection, saying what selected each file."""
    if not changed:
        return "scope: nothing differs from the base -- NO test files selected."
    lines = [f"scope: {len(changed)} changed path(s)"]
    if selection.everything:
        lines.append(
            f"scope: the WHOLE tier -- {len(selection.widened_by)} path(s) no test declares:"
        )
        lines += [f"  widened by {path}" for path in selection.widened_by]
        return "\n".join(lines)
    if selection.unreached:
        lines.append("scope: changed but read by no test:")
        lines += [f"  {path}" for path in selection.unreached]
    if not selection.files:
        lines.append("scope: NO test files selected -- no test reads any changed path.")
        return "\n".join(lines)
    lines.append(f"scope: {len(selection.files)} of {universe} test files selected:")
    width = max(map(len, selection.files))
    lines += [f"  {path:<{width}}  {why}" for path, why in sorted(selection.files.items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="changed paths, relative to the repository root")
    parser.add_argument("--base", default="origin/main", help="the ref to diff against")
    parser.add_argument("--write", type=pathlib.Path, help="record the selection here for a caller")
    parser.add_argument(
        "--root", type=pathlib.Path, default=ROOT, help="the checkout to scope (default: this one)"
    )
    args = parser.parse_args(argv)
    try:
        changed = sorted(set(args.paths)) or changed_paths(args.root, args.base)
    except (subprocess.CalledProcessError, IndexError) as error:
        print(f"scope: cannot read the change against {args.base}: {error}", file=sys.stderr)
        return 2
    graph = ImportGraph(args.root)
    selection = select(changed, graph)
    universe = sum(is_test_file(p.relative_to(args.root).as_posix()) for p in graph.paths.values())
    print(report(selection, changed, universe))
    if args.write is not None:
        mode = "all" if selection.everything else ("some" if selection.files else "none")
        body = [] if selection.everything else sorted(selection.files)
        args.write.write_text("\n".join([mode, *body]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
