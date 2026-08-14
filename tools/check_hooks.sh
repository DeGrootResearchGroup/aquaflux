#!/usr/bin/env bash
#
# Report -- on stderr, without ever failing -- when git will not run the committed hooks in
# .githooks/, or will run them only by accident.
#
# Those hooks are the local half of the gate CI enforces, but git runs them only when
# core.hooksPath resolves to them, and when it does not, git says nothing whatsoever. The gate is
# simply gone while everyone assumes it is there, and the first sign is a red required check on a
# pull request. A safety net that is quietly absent is worse than no net at all, because it is
# trusted. Since the hooks cannot report their own absence, something that runs often has to.
#
# The path is resolved the way git resolves it: a RELATIVE core.hooksPath is taken from the top
# level of the working tree, so it follows each checkout -- including each worktree -- to its own
# .githooks/. An ABSOLUTE one instead names one fixed directory for every checkout that inherits
# it, so what runs depends on the branch THAT checkout happens to be sitting on. Both failures are
# reported: hooks that will not run, and hooks that run today but are one branch switch away from
# silently disappearing.
#
# Usage
#   tools/check_hooks.sh      silent when the wiring is sound; warns on stderr otherwise
#
# Always exits 0. This is a warning, never a gate: a caller's exit status must be its own.

set -uo pipefail

# A CI checkout deliberately does not wire the hooks; the workflow runs the same gate directly.
[ -n "${CI:-}" ] && exit 0

# Outside a work tree there is nothing to check.
top=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0

configured=$(git config --get core.hooksPath 2>/dev/null || true)

if [ -z "$configured" ]; then
  printf 'hooks: not enabled -- the local ruff/codespell gate will not run before a push.\n' >&2
  printf '       enable them once per clone:  git config core.hooksPath .githooks\n' >&2
  exit 0
fi

case "$configured" in
  /*) resolved="$configured"; absolute=1 ;;
  *)  resolved="$top/$configured"; absolute=0 ;;
esac
resolved="${resolved%/}"

if [ ! -x "$resolved/pre-push" ]; then
  printf 'hooks: NOT wired up -- core.hooksPath = %s\n' "$configured" >&2
  printf '       resolves to %s, which holds no executable pre-push,\n' "$resolved" >&2
  printf '       so the ruff/codespell gate will NOT run before a push.\n' >&2
  if [ -n "$(git config --worktree --get core.hooksPath 2>/dev/null || true)" ]; then
    printf '       this worktree overrides the repository setting; drop the override with:\n' >&2
    printf '         git config --worktree --unset core.hooksPath\n' >&2
  else
    printf '       point it at this checkout with:  git config core.hooksPath .githooks\n' >&2
  fi
  exit 0
fi

# The hooks do resolve -- but through a directory outside this checkout, which is sound only by
# coincidence. What runs is whatever that other directory holds right now, and it holds nothing at
# all while that checkout sits on a branch predating .githooks/. A relative path cannot drift this
# way, because it follows each checkout to its own copy.
if [ "$absolute" -eq 1 ] && [ "$resolved" != "$top/.githooks" ]; then
  printf 'hooks: fragile -- core.hooksPath = %s\n' "$configured" >&2
  printf '       they run right now, but from outside this checkout, so they will vanish\n' >&2
  printf '       silently whenever that directory changes -- its contents follow whichever\n' >&2
  printf '       branch that other checkout is on.\n' >&2
  printf '       make them follow this checkout instead:  git config core.hooksPath .githooks\n' >&2
fi

exit 0
