"""``tools/check_env_prefixes.py`` fails when a validation/ script reads an env-var prefix that
``run_case.sh``'s capture line does not cover.

That capture line is how a run-file answers "what was this case configured with" -- it greps the
process environment for a fixed list of prefixes and writes whatever matches. A script that reads a
prefix outside that list produces a run-file with no line for it, which is indistinguishable from a
run that used every default. Four scripts (``PROFILE_``, ``CONSISTENCY_``, ``FLOW_``, ``TAPER_``)
were in exactly that state; see the script under test's own header for the mechanism and why an
allow-list is otherwise self-perpetuating.

The checker reads the covered-prefix pattern out of ``run_case.sh`` itself rather than a second copy
kept here, so the cases below build a throwaway ``run_case.sh``-shaped file for each scenario --
never hardcoding the real repository's prefix list -- and the last case checks the real files through
the script's own defaults.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[2] / "tools" / "check_env_prefixes.py"

# A minimal stand-in for run_case.sh: only the capture line matters to the checker.
RUN_CASE = "env | grep -E '^(BFS3D|PITZ)_' | sort | sed 's/^/env: /' || true\n"


def _run(scan_dir: Path, run_case: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECK), str(scan_dir), "--run-case", str(run_case)],
        capture_output=True,
        text=True,
    )


def _make_run_case(tmp_path: Path, pattern: str = RUN_CASE) -> Path:
    run_case = tmp_path / "run_case.sh"
    run_case.write_text(pattern)
    return run_case


def test_a_covered_prefix_passes(tmp_path: Path) -> None:
    (tmp_path / "case.py").write_text('import os\nos.environ.get("BFS3D_MESH", "default")\n')
    result = _run(tmp_path, _make_run_case(tmp_path))
    assert result.returncode == 0, result.stderr


def test_an_uncovered_prefix_fails(tmp_path: Path) -> None:
    (tmp_path / "probe.py").write_text('import os\nos.environ.get("PROFILE_SWEEPS", "20")\n')
    result = _run(tmp_path, _make_run_case(tmp_path))
    assert result.returncode == 1
    assert "PROFILE_" in result.stderr
    assert "probe.py" in result.stderr


def test_adding_the_prefix_makes_it_pass_again(tmp_path: Path) -> None:
    """The same file, covered this time -- isolates the prefix as the cause of the failure above."""
    (tmp_path / "probe.py").write_text('import os\nos.environ.get("PROFILE_SWEEPS", "20")\n')
    run_case = _make_run_case(tmp_path, "env | grep -E '^(BFS3D|PITZ|PROFILE)_' | sort || true\n")
    result = _run(tmp_path, run_case)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "read",
    [
        'os.environ["FLOW_ORDER"]',
        'os.environ.get("FLOW_ORDER")',
        'os.getenv("FLOW_ORDER", "default")',
    ],
)
def test_all_three_read_forms_are_recognized(tmp_path: Path, read: str) -> None:
    (tmp_path / "case.py").write_text(f"import os\n{read}\n")
    result = _run(tmp_path, _make_run_case(tmp_path))
    assert result.returncode == 1
    assert "FLOW_" in result.stderr


def test_a_multiword_covered_alternative_covers_names_with_that_whole_prefix(
    tmp_path: Path,
) -> None:
    """``ILU0_SWEEP`` is a real prefix in run_case.sh's own list, and it is two words.

    Coverage has to be decided by matching the *whole* name against the capture pattern, not by
    comparing single-word prefixes -- a checker that only ever compared the leading word would
    wrongly flag every ``ILU0_SWEEP_*`` name, because "ILU0" alone is not one of the alternatives.
    """
    (tmp_path / "sweep.py").write_text('import os\nos.environ.get("ILU0_SWEEP_ARMS", "")\n')
    run_case = _make_run_case(tmp_path, "env | grep -E '^(BFS3D|ILU0_SWEEP)_' | sort || true\n")
    result = _run(tmp_path, run_case)
    assert result.returncode == 0, result.stderr


def test_a_non_literal_env_read_is_ignored_rather_than_misreported(tmp_path: Path) -> None:
    """A name built at runtime cannot be attributed to a prefix from source text alone."""
    (tmp_path / "case.py").write_text(
        'import os\nkey = "PROFILE_" + "SWEEPS"\nos.environ.get(key, "20")\n'
    )
    result = _run(tmp_path, _make_run_case(tmp_path))
    assert result.returncode == 1  # zero literal reads found anywhere -- guarded below, not a pass
    assert "stopped seeing anything" in result.stderr


def test_examining_no_python_files_is_an_error_not_a_pass(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("nothing to see here\n")
    result = _run(tmp_path, _make_run_case(tmp_path))
    assert result.returncode == 1
    assert "stopped seeing anything" in result.stderr


def test_a_run_case_file_with_no_capture_line_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "case.py").write_text('import os\nos.environ.get("BFS3D_MESH")\n')
    run_case = tmp_path / "run_case.sh"
    run_case.write_text("#!/usr/bin/env bash\necho hello\n")
    result = _run(tmp_path, run_case)
    assert result.returncode == 1
    assert "capture line" in result.stderr


def test_a_missing_run_case_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "case.py").write_text('import os\nos.environ.get("BFS3D_MESH")\n')
    result = _run(tmp_path, tmp_path / "does-not-exist.sh")
    assert result.returncode == 1


def test_this_repository_passes_its_own_check() -> None:
    """The default target: every prefix read under validation/ is covered by run_case.sh today.

    This is the arm that fails when a new script adds an uncovered prefix, which is the whole
    point; the arms above only establish that the checker can still tell the difference.
    """
    result = subprocess.run(
        [sys.executable, str(CHECK)],
        capture_output=True,
        text=True,
        cwd=CHECK.parents[1],
    )
    assert result.returncode == 0, result.stderr
