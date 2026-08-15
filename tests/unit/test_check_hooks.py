"""``tools/check_hooks.sh`` reports when git will not run the committed hooks.

The hooks in ``.githooks/`` are the local half of the gate CI enforces, but git runs
them only when ``core.hooksPath`` resolves to them -- and when it does not, git is
silent. The script under test is the only thing that reports that, so its own
branches are worth pinning: a warning that stops firing is indistinguishable, from
the outside, from a repository whose hooks are fine.

Each case builds a throwaway repository with a chosen configuration and asserts on
what the script writes to stderr. The script must never fail, whatever it finds --
callers rely on their own exit status surviving it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

CHECK_HOOKS = Path(__file__).resolve().parents[2] / "tools" / "check_hooks.sh"


def _run(cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the checker in ``cwd``, with ``CI`` cleared unless the case sets it."""
    environment = dict(os.environ)
    environment.pop("CI", None)
    environment.update(env or {})
    return subprocess.run(
        [str(CHECK_HOOKS)],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=environment,
    )


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    """A minimal git repository with no hooks configuration of its own."""
    root = tmp_path / name
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def _add_hooks(root: Path, directory: str = ".githooks") -> Path:
    """Give ``root`` an executable pre-push hook, as the repository ships."""
    hooks = root / directory
    hooks.mkdir(parents=True, exist_ok=True)
    pre_push = hooks / "pre-push"
    pre_push.write_text("#!/usr/bin/env bash\nexit 0\n")
    pre_push.chmod(0o755)
    return hooks


def _set_hooks_path(root: Path, value: str) -> None:
    subprocess.run(["git", "config", "core.hooksPath", value], cwd=root, check=True)


def test_silent_when_wired_relative(tmp_path: Path) -> None:
    """The shipped configuration -- a relative path to a populated directory."""
    root = _repo(tmp_path)
    _add_hooks(root)
    _set_hooks_path(root, ".githooks")

    result = _run(root)

    assert result.returncode == 0
    assert result.stderr == ""


def test_warns_when_not_enabled(tmp_path: Path) -> None:
    """No core.hooksPath at all: git installs nothing on its own."""
    root = _repo(tmp_path)
    _add_hooks(root)

    result = _run(root)

    assert result.returncode == 0
    assert "not enabled" in result.stderr
    assert "git config core.hooksPath .githooks" in result.stderr


def test_warns_when_path_holds_no_hooks(tmp_path: Path) -> None:
    """Configured, but pointing somewhere with no pre-push -- git stays silent here."""
    root = _repo(tmp_path)
    empty = root / "empty-hooks"
    empty.mkdir()
    _set_hooks_path(root, "empty-hooks")

    result = _run(root)

    assert result.returncode == 0
    assert "NOT wired up" in result.stderr
    assert "no executable pre-push" in result.stderr


def test_warns_when_hooks_are_not_executable(tmp_path: Path) -> None:
    """A pre-push without the executable bit is one git will not run."""
    root = _repo(tmp_path)
    hooks = _add_hooks(root)
    (hooks / "pre-push").chmod(0o644)
    _set_hooks_path(root, ".githooks")

    result = _run(root)

    assert result.returncode == 0
    assert "NOT wired up" in result.stderr


def test_warns_when_absolute_path_points_outside_the_checkout(tmp_path: Path) -> None:
    """Hooks that run today, but from another checkout -- sound only by coincidence.

    This is the case that reads as healthy: the hooks resolve and do run. What makes
    it fragile is that the directory belongs to a different checkout, so its contents
    follow that checkout's branch, and the hooks vanish without a word when it moves.
    """
    root = _repo(tmp_path)
    elsewhere = _repo(tmp_path, name="elsewhere")
    external_hooks = _add_hooks(elsewhere)
    _set_hooks_path(root, str(external_hooks))

    result = _run(root)

    assert result.returncode == 0
    assert "fragile" in result.stderr
    assert "vanish" in result.stderr
    # It must not be mistaken for the hooks-do-not-run case: they do run.
    assert "NOT wired up" not in result.stderr


def test_fragile_worktree_is_told_to_drop_the_override_not_to_set_the_repository(
    tmp_path: Path,
) -> None:
    """The remedy has to name the level the setting actually lives at.

    A worktree-level ``core.hooksPath`` shadows the repository one, so the general
    advice -- set ``core.hooksPath`` to ``.githooks`` -- writes a value git never
    consults. The hooks stay exactly as fragile and this warning repeats verbatim,
    which reads as the advice being wrong rather than aimed at the wrong level. This
    is the shape a worktree is routinely created in, so it is the shape most likely
    to be met.
    """
    root = _repo(tmp_path)
    _add_hooks(root)
    _set_hooks_path(root, ".githooks")  # the repository is already correct
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "base"], cwd=root, check=True)

    tree = tmp_path / "tree"
    subprocess.run(["git", "worktree", "add", "-q", "--detach", str(tree)], cwd=root, check=True)
    _add_hooks(tree)
    # A worktree-level value exists at all only under this extension, which is precisely
    # why the shadowing is easy to miss: without it, `git config --worktree` refuses.
    subprocess.run(["git", "config", "extensions.worktreeConfig", "true"], cwd=tree, check=True)
    # What the worktree tooling does, and what makes the repository setting inert.
    subprocess.run(
        ["git", "config", "--worktree", "core.hooksPath", str(root / ".githooks")],
        cwd=tree,
        check=True,
    )

    result = _run(tree)

    assert result.returncode == 0
    assert "fragile" in result.stderr
    assert "--worktree --unset core.hooksPath" in result.stderr
    # The repository level is already right, so it must not be suggested as the fix.
    assert "point it at this checkout" not in result.stderr


def test_silent_when_absolute_path_is_this_checkout(tmp_path: Path) -> None:
    """An absolute path naming this checkout's own .githooks is not fragile."""
    root = _repo(tmp_path)
    hooks = _add_hooks(root)
    _set_hooks_path(root, str(hooks))

    result = _run(root)

    assert result.returncode == 0
    assert result.stderr == ""


def test_silent_under_ci(tmp_path: Path) -> None:
    """CI checks out without wiring hooks; the workflow runs the gate directly."""
    root = _repo(tmp_path)
    _set_hooks_path(root, "empty-hooks")

    result = _run(root, env={"CI": "true"})

    assert result.returncode == 0
    assert result.stderr == ""


def test_silent_outside_a_work_tree(tmp_path: Path) -> None:
    """Nothing to check when there is no repository."""
    outside = tmp_path / "plain"
    outside.mkdir()

    result = _run(outside)

    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.parametrize("hooks_path", [".githooks", "empty-hooks", "/nonexistent/hooks"])
def test_never_fails(tmp_path: Path, hooks_path: str) -> None:
    """Whatever it finds, the checker exits 0: a caller's status must be its own."""
    root = _repo(tmp_path)
    _add_hooks(root)
    (root / "empty-hooks").mkdir()
    _set_hooks_path(root, hooks_path)

    assert _run(root).returncode == 0
