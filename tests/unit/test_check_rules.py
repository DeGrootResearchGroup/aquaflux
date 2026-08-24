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
