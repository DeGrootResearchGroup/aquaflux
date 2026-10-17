"""``tools/check_rules.sh`` fails when a guidance file does not declare the paths it is scoped to.

Those files are loaded automatically for whoever is working in the repository, selected by the
``paths:`` glob in each file's frontmatter so that a file loads only while the code it describes is
being read or edited. A file with **no** ``paths:`` is not scoped to nothing -- it is scoped to
everything, and loads for every session whatever is being worked on. The script under test is what
stops that happening by omission; see its own header for what it cost when it was not there.

The inversion is silent from both ends, which is why the checker gets a test: the file looks inert,
and a session paying for it reports only that it began with less room than it should have. So does a
checker that has quietly stopped seeing anything -- hence the arms below assert both that a
violation is caught and that a clean directory passes, and that an empty or missing directory is
itself an error rather than a pass.

Each case points the script at a throwaway directory, which is also why nothing here hardcodes the
real one: the repository's own files are checked by the last case, through the script's default.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

CHECK_RULES = Path(__file__).resolve().parents[2] / "tools" / "check_rules.sh"

SCOPED = '---\npaths:\n  - "src/**"\n---\n\n# A scoped rule\n'


def _run(directory: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [str(CHECK_RULES)] + ([str(directory)] if directory is not None else [])
    return subprocess.run(command, capture_output=True, text=True)


def test_a_directory_of_scoped_files_passes(tmp_path: Path) -> None:
    (tmp_path / "one.md").write_text(SCOPED)
    (tmp_path / "two.md").write_text(SCOPED)
    assert _run(tmp_path).returncode == 0


def test_a_file_with_no_frontmatter_at_all_fails(tmp_path: Path) -> None:
    (tmp_path / "scoped.md").write_text(SCOPED)
    (tmp_path / "orphan.md").write_text("# No frontmatter\n\nprose\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "orphan.md" in result.stderr
    assert "scoped.md" not in result.stderr


def test_a_file_whose_frontmatter_omits_the_key_fails(tmp_path: Path) -> None:
    """Frontmatter alone is not a declaration -- the ``paths:`` key is what scopes the file."""
    (tmp_path / "orphan.md").write_text("---\nname: orphan\n---\n\n# Titled but unscoped\n")
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "orphan.md" in result.stderr


def test_the_key_counts_only_inside_the_frontmatter_block(tmp_path: Path) -> None:
    """Prose *about* the key is documentation, not declaration.

    Several of these files legitimately discuss ``paths:`` in their body, so a check that simply
    grepped the whole file would pass exactly the file it exists to catch.
    """
    (tmp_path / "orphan.md").write_text(
        "---\nname: orphan\n---\n\nThis file explains what paths: does elsewhere.\n"
    )
    assert _run(tmp_path).returncode == 1


@pytest.mark.parametrize("case", ["empty", "missing"])
def test_examining_nothing_is_an_error_not_a_pass(tmp_path: Path, case: str) -> None:
    """A clean tree and a check that has gone blind must not look the same from the outside."""
    directory = tmp_path if case == "empty" else tmp_path / "absent"
    result = _run(directory)
    assert result.returncode == 1
    assert result.stderr.strip()


def test_this_repository_passes_its_own_check() -> None:
    """The default target: every guidance file here declares its scope.

    This is the arm that fails when someone adds an unscoped file, which is the whole point; the
    arms above only establish that the script can still tell the difference.
    """
    result = _run()
    assert result.returncode == 0, result.stderr


# --- The size budget (tools/rules_budget.py, run by check_rules.sh on <repo>/.claude/rules) ---
#
# A file that is merely large still loads; the cost is the context it takes from every session that
# touches its paths. These arms build a throwaway repository, so the budget is exercised against
# sizes chosen here rather than against whatever the real files happen to weigh today.

import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "rules_budget", CHECK_RULES.parent / "rules_budget.py"
)
rules_budget = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rules_budget)

KB = rules_budget.KB


def _rule(glob: str, size: int = 1 * KB) -> str:
    head = f'---\npaths:\n  - "{glob}"\n---\n\n'
    return head + "x" * max(0, size - len(head))


def _repo(tmp_path: Path, rules: dict[str, str], tracked: list[str]) -> Path:
    (tmp_path / ".claude" / "rules").mkdir(parents=True)
    (tmp_path / "CLAUDE.md").write_text("# briefing\n")
    for name, text in rules.items():
        (tmp_path / ".claude" / "rules" / name).write_text(text)
    for f in tracked:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_records_within_budget_pass_through_check_rules(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.md": _rule("src/**")}, ["src/a.py"])
    assert _run(root / ".claude" / "rules").returncode == 0


def test_a_rules_file_over_the_per_file_budget_fails(tmp_path: Path) -> None:
    big = rules_budget.FILE_BUDGET + KB
    root = _repo(tmp_path, {"big.md": _rule("src/**", big)}, ["src/a.py"])
    result = _run(root / ".claude" / "rules")
    assert result.returncode == 1
    assert "big.md" in result.stderr and "budget for one file" in result.stderr


def test_claude_md_is_held_to_the_same_per_file_budget(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.md": _rule("src/**")}, ["src/a.py"])
    (root / "CLAUDE.md").write_text("x" * (rules_budget.FILE_BUDGET + KB))
    assert any("CLAUDE.md" in m for m in rules_budget.check(root))


def test_rules_that_load_together_are_held_to_a_combined_budget(tmp_path: Path) -> None:
    """Each file is under its own budget; one source file loads both, and that sum is over."""
    each = rules_budget.LOADED_BUDGET // 2 + KB
    assert each < rules_budget.FILE_BUDGET
    rules = {"a.md": _rule("src/**", each), "b.md": _rule("src/b.py", each)}
    root = _repo(tmp_path, rules, ["src/a.py", "src/b.py"])
    failures = rules_budget.check(root)
    assert len(failures) == 1
    assert "src/b.py" in failures[0] and "a.md" in failures[0] and "b.md" in failures[0]


def test_a_rules_file_whose_globs_match_nothing_fails(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.md": _rule("src/**"), "dead.md": _rule("gone/**")}, ["src/a.py"])
    failures = rules_budget.check(root)
    assert len(failures) == 1 and "dead.md" in failures[0]


@pytest.mark.parametrize(
    ("glob", "path", "matches"),
    [
        ("aquaflux/solve/**", "aquaflux/solve/x/y.py", True),
        ("aquaflux/solve/**", "aquaflux/solver.py", False),
        ("validation/*/case.yaml", "validation/a/case.yaml", True),
        ("validation/*/case.yaml", "validation/a/b/case.yaml", False),
        ("aquaflux/solve/march.py", "aquaflux/solve/march_log.py", False),
    ],
)
def test_globs_match_as_the_loader_reads_them(glob: str, path: str, matches: bool) -> None:
    assert bool(rules_budget._regex(glob).match(path)) is matches


def test_the_over_budget_list_is_a_ratchet(tmp_path: Path, monkeypatch) -> None:
    """A listed file may not grow past its ceiling, and the entry must follow it down."""
    size = rules_budget.FILE_BUDGET + 20 * KB
    root = _repo(tmp_path, {"big.md": _rule("src/**", size)}, ["src/a.py"])
    name = ".claude/rules/big.md"

    monkeypatch.setattr(rules_budget, "OVER_BUDGET", {name: size})
    assert rules_budget.check(root) == []

    monkeypatch.setattr(rules_budget, "OVER_BUDGET", {name: size - KB})
    assert "may shrink, never grow" in rules_budget.check(root)[0]

    monkeypatch.setattr(rules_budget, "OVER_BUDGET", {name: size + rules_budget.SLACK + KB})
    assert "lower its OVER_BUDGET ceiling" in rules_budget.check(root)[0]

    (root / ".claude" / "rules" / "big.md").write_text(_rule("src/**"))
    assert "back under budget" in rules_budget.check(root)[0]
