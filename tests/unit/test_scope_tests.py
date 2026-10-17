"""``tools/scope_tests.py`` -- the selector behind ``tools/fastgate.sh --changed``.

Tested for the same reason ``tools/sibling_builders.py`` and ``tools/check_hooks.sh`` are: a selector
that has stopped following some kind of import selects *fewer* tests, and a run of fewer tests is green
in exactly the way a run of the right tests is. So each way a test can reach a module -- through a
package's re-export, a nested re-export, a transitive import, an import inside a function, a plain
``import pkg``, a star import, a relative import, a ``conftest.py`` -- has a fixture of its own here,
built so that only that rule selects the test, and each rule was deleted in turn to watch its test fail.
The other direction is pinned too: a name re-exported from a package does not drag in the package's
other submodules, which is what keeps the selection from being the whole suite for every change.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.support.tools import load_tool

scope = load_tool("scope_tests")

REPOSITORY = Path(__file__).resolve().parents[2]


def _write(root: Path, files: dict[str, str]) -> Path:
    """Write each ``relative path -> source`` under ``root``, and return ``root``."""
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


#: A small package: ``pkg`` re-exports ``A`` from ``a`` and ``B`` from ``b``; ``a`` imports ``c``.
PACKAGE = {
    "aquaflux/__init__.py": "",
    "aquaflux/pkg/__init__.py": "from .a import A\nfrom .b import B\n",
    "aquaflux/pkg/a.py": "from .c import C\n\nclass A(C): ...\n",
    "aquaflux/pkg/b.py": "class B: ...\n",
    "aquaflux/pkg/c.py": "class C: ...\n",
    "tests/__init__.py": "",
    "tests/unit/__init__.py": "",
}


def _selected(root: Path, *changed: str, reaches=None) -> set[str]:
    # The repository's own declared table is kept out: these fixtures pin the import walk alone.
    graph = scope.ImportGraph(root)
    selection = scope.select(list(changed), graph, reaches=reaches or {}, reaches_no_test=())
    assert not selection.everything, f"widened by {selection.widened_by}"
    return set(selection.files)


def test_a_name_imported_from_a_package_reaches_the_submodule_that_DEFINES_it(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_a.py": "from aquaflux.pkg import A\n"})
    assert _selected(root, "aquaflux/pkg/a.py") == {"tests/unit/test_a.py"}


def test_it_does_NOT_reach_the_package_s_other_submodules(tmp_path: Path):
    # The package __init__ loads b as well, but nothing the test imports comes from it. Following the
    # __init__'s re-exports wholesale would select this test for every change to the package.
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_a.py": "from aquaflux.pkg import A\n"})
    assert _selected(root, "aquaflux/pkg/b.py") == set()


def test_it_follows_a_transitive_import_of_the_defining_module(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_a.py": "from aquaflux.pkg import A\n"})
    assert _selected(root, "aquaflux/pkg/c.py") == {"tests/unit/test_a.py"}


def test_it_follows_a_re_export_of_a_re_export(tmp_path: Path):
    files = {
        **PACKAGE,
        "aquaflux/pkg/__init__.py": "from .sub import D\n",
        "aquaflux/pkg/sub/__init__.py": "from .deep import D\nfrom .other import E\n",
        "aquaflux/pkg/sub/deep.py": "class D: ...\n",
        "aquaflux/pkg/sub/other.py": "class E: ...\n",
        "tests/unit/test_d.py": "from aquaflux.pkg import D\n",
    }
    root = _write(tmp_path, files)
    assert _selected(root, "aquaflux/pkg/sub/deep.py") == {"tests/unit/test_d.py"}
    assert _selected(root, "aquaflux/pkg/sub/other.py") == set()


def test_an_import_inside_a_function_counts(tmp_path: Path):
    source = "def test_it():\n    from aquaflux.pkg import B\n"
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_b.py": source})
    assert _selected(root, "aquaflux/pkg/b.py") == {"tests/unit/test_b.py"}


def test_a_package_s_own_init_reaches_everything_imported_from_it(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_a.py": "from aquaflux.pkg import A\n"})
    assert _selected(root, "aquaflux/pkg/__init__.py") == {"tests/unit/test_a.py"}


def test_the_top_level_init_reaches_every_module_below_it(tmp_path: Path):
    # A test importing a submodule directly still runs aquaflux/__init__.py first.
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_c.py": "from aquaflux.pkg.c import C\n"})
    assert _selected(root, "aquaflux/__init__.py") == {"tests/unit/test_c.py"}


def test_a_plain_import_of_a_package_reaches_all_of_it(tmp_path: Path):
    # `import aquaflux.pkg` then `aquaflux.pkg.B`: which names are used is not visible, so all are.
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_p.py": "import aquaflux.pkg\n"})
    assert _selected(root, "aquaflux/pkg/b.py") == {"tests/unit/test_p.py"}


def test_a_star_import_reaches_all_of_it(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_s.py": "from aquaflux.pkg import *\n"})
    assert _selected(root, "aquaflux/pkg/b.py") == {"tests/unit/test_s.py"}


def test_an_imported_submodule_is_reached_by_name(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_m.py": "from aquaflux.pkg import b\n"})
    assert _selected(root, "aquaflux/pkg/b.py") == {"tests/unit/test_m.py"}
    assert _selected(root, "aquaflux/pkg/a.py") == set()


def test_a_relative_import_in_the_test_tree_is_resolved(tmp_path: Path):
    files = {
        **PACKAGE,
        "tests/support/__init__.py": "",
        "tests/support/helper.py": "from aquaflux.pkg import B\n",
        "tests/unit/test_h.py": "from ..support import helper\n",
    }
    root = _write(tmp_path, files)
    assert _selected(root, "aquaflux/pkg/b.py") == {"tests/unit/test_h.py"}


def test_a_conftest_reaches_the_tests_below_it_and_no_others(tmp_path: Path):
    files = {
        **PACKAGE,
        "tests/unit/conftest.py": "from aquaflux.pkg import B\n",
        "tests/unit/test_u.py": "",
        "tests/other/__init__.py": "",
        "tests/other/test_o.py": "",
    }
    root = _write(tmp_path, files)
    assert _selected(root, "aquaflux/pkg/b.py") == {"tests/unit/test_u.py"}


def test_a_DELETED_module_still_reaches_what_imports_it(tmp_path: Path):
    # Whatever still imports the old name is exactly what deleting it can break.
    root = _write(
        tmp_path, {**PACKAGE, "tests/unit/test_g.py": "from aquaflux.pkg.gone import G\n"}
    )
    assert _selected(root, "aquaflux/pkg/gone.py") == {"tests/unit/test_g.py"}


def test_a_changed_test_file_selects_itself(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_a.py": "from aquaflux.pkg import A\n"})
    assert _selected(root, "tests/unit/test_a.py") == {"tests/unit/test_a.py"}


def test_a_DECLARED_reach_selects_its_test(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_r.py": ""})
    reaches = {"tests/unit/test_r.py": (("tools/run.sh",), "runs it")}
    assert _selected(root, "tools/run.sh", reaches=reaches) == {"tests/unit/test_r.py"}


def test_an_UNDECLARED_path_widens_to_the_whole_tier_and_names_itself(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_a.py": "from aquaflux.pkg import A\n"})
    selection = scope.select(
        ["aquaflux/pkg/a.py", "tools/new.sh"], scope.ImportGraph(root), reaches={}
    )
    assert selection.everything
    assert selection.widened_by == ["tools/new.sh"]
    assert "widened by tools/new.sh" in scope.report(selection, ["tools/new.sh"], 1)


def test_a_path_no_test_reads_selects_nothing_and_says_so(tmp_path: Path):
    root = _write(tmp_path, PACKAGE)
    selection = scope.select(
        ["docs/page.md"], scope.ImportGraph(root), reaches={}, reaches_no_test=("docs/*",)
    )
    assert (selection.everything, selection.files, selection.unreached) == (
        False,
        {},
        ["docs/page.md"],
    )
    assert "NO test files selected" in scope.report(selection, ["docs/page.md"], 1)


def test_a_package_module_nothing_imports_selects_nothing_rather_than_widening(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "aquaflux/pkg/orphan.py": ""})
    selection = scope.select(["aquaflux/pkg/orphan.py"], scope.ImportGraph(root), reaches={})
    assert (selection.everything, selection.unreached) == (False, ["aquaflux/pkg/orphan.py"])


def test_the_report_says_which_import_chain_selected_each_file(tmp_path: Path):
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_a.py": "from aquaflux.pkg import A\n"})
    selection = scope.select(["aquaflux/pkg/c.py"], scope.ImportGraph(root), reaches={})
    text = scope.report(selection, ["aquaflux/pkg/c.py"], 1)
    assert "tests/unit/test_a.py  imports aquaflux.pkg.a <- aquaflux.pkg.c" in text


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_the_change_is_everything_since_the_merge_base_committed_or_not(tmp_path: Path):
    root = _write(tmp_path, {"kept.py": "", "edited.py": "", "deleted.py": "", "staged.py": ""})
    _git(root, "init", "-q", "-b", "main")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    _git(root, "checkout", "-qb", "work")
    (root / "committed.py").write_text("")
    _git(root, "add", "committed.py")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "work")
    (root / "staged.py").write_text("x = 1\n")
    _git(root, "add", "staged.py")
    (root / "edited.py").write_text("x = 1\n")
    (root / "deleted.py").unlink()
    (root / "untracked.py").write_text("")
    assert scope.changed_paths(root, "main") == [
        "committed.py",
        "deleted.py",
        "edited.py",
        "staged.py",
        "untracked.py",
    ]


@pytest.mark.parametrize(
    ("changed", "mode"),
    [(["aquaflux/vectors.py"], "some"), (["pyproject.toml"], "all"), (["LICENSE"], "none")],
)
def test_the_written_selection_starts_with_its_mode(tmp_path: Path, changed: list[str], mode: str):
    out = tmp_path / "selection"
    assert scope.main([*changed, "--write", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0] == mode
    assert (len(lines) > 1) == (mode == "some")


def test_on_this_repository_a_shared_leaf_widens_and_a_narrow_module_does_not():
    # Pins that the graph sees the real package at all: a selector that had stopped following imports
    # would select little for both, and that reads exactly like a narrow change.
    graph = scope.ImportGraph(REPOSITORY)
    leaf = scope.select(["aquaflux/vectors.py"], graph)
    narrow = scope.select(["aquaflux/precondition/sparse_jacobian.py"], graph)
    assert not leaf.everything and not narrow.everything
    assert "tests/unit/test_vectors.py" in leaf.files
    assert "tests/unit/test_sparse_jacobian.py" in narrow.files
    assert "tests/unit/test_vectors.py" not in narrow.files
    assert len(leaf.files) > 2 * len(narrow.files) > 40


def test_a_program_a_test_runs_in_a_child_interpreter_is_followed(tmp_path: Path):
    # The distributed tests hold their child's code as a string; what the child imports is reached.
    source = 'CHILD = """\nfrom aquaflux.pkg import B\nprint("ok")\n"""\n'
    root = _write(tmp_path, {**PACKAGE, "tests/unit/test_child.py": source})
    assert _selected(root, "aquaflux/pkg/b.py") == {"tests/unit/test_child.py"}


def test_a_string_in_the_package_is_not_read_as_a_program(tmp_path: Path):
    # Only the test tree runs code from strings; the package's own strings are prose and messages.
    files = {
        **PACKAGE,
        "aquaflux/pkg/c.py": 'HELP = "from aquaflux.pkg import B"\n',
        "tests/unit/test_c.py": "from aquaflux.pkg.c import HELP\n",
    }
    root = _write(tmp_path, files)
    assert _selected(root, "aquaflux/pkg/b.py") == set()


def test_a_helper_s_declared_reach_selects_every_test_importing_it(tmp_path: Path):
    files = {
        **PACKAGE,
        "tests/support/__init__.py": "",
        "tests/support/meshes.py": "",
        "tests/unit/test_mesh.py": "from tests.support import meshes\n",
        "tests/unit/test_other.py": "",
    }
    root = _write(tmp_path, files)
    reaches = {"tests/support/meshes.py": (("tests/fixtures/*",), "reads a fixture")}
    assert _selected(root, "tests/fixtures/slab/points", reaches=reaches) == {
        "tests/unit/test_mesh.py"
    }
