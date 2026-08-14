#!/usr/bin/env bash
#
# Run a test tier and report what it actually said.
#
# The failure this prevents is small, silent and expensive: `pytest ... | tail -n` reads as a way to
# see the summary, but a pipeline's exit status is the LAST stage's, and `tail` exits 0 whatever
# pytest did. Worse, this suite prints a block of solver-library shutdown chatter after the summary
# line, so a fixed-size tail shows only the chatter and buries the one line that matters. A run whose
# result was read that way reports success no matter what happened.
#
# So: redirect to a file, never pipe; report pytest's own exit status; and find the summary line by
# pattern rather than by position.
#
# Usage
#   tools/fastgate.sh                 the always-on gate: not slow, not validation
#   tools/fastgate.sh slow            the slow tier
#   tools/fastgate.sh validation      the validation tier
#   tools/fastgate.sh all             everything
#   tools/fastgate.sh <tier> -k name  remaining arguments are passed through to pytest
#
# Exits with pytest's status, so it composes in a shell `&&` chain and in a hook.
#
# It also warns, before running anything, when the committed hooks in .githooks/ are not wired up --
# see check_hook_wiring below. That warning never changes the exit status; only pytest does.

set -uo pipefail

# Warn when git will not run the committed hooks in .githooks/.
#
# Those hooks are the local half of the same gate CI enforces, but git runs them only when
# core.hooksPath points at them -- and when it does not, git says nothing whatsoever. A safety net
# that is quietly absent is worse than no net at all, because it is trusted: a lint slip then
# reaches CI as a red required check on the pull request, which is exactly what the hook exists to
# prevent. This script is the thing run most often, so it is the cheapest place to notice.
#
# The path is resolved the way git itself resolves it: a RELATIVE core.hooksPath is taken from the
# top level of the working tree, so it follows each checkout -- including each worktree -- to its
# own .githooks/. An ABSOLUTE one instead pins every worktree to the single directory it names,
# whose contents depend on whatever branch that other checkout happens to be sitting on.
check_hook_wiring() {
  # A CI checkout deliberately does not wire the hooks; the workflow runs the gate directly.
  [ -n "${CI:-}" ] && return 0

  local top configured resolved
  top=$(git rev-parse --show-toplevel 2>/dev/null) || return 0
  configured=$(git config --get core.hooksPath 2>/dev/null || true)

  if [ -z "$configured" ]; then
    printf 'fastgate: git hooks are NOT enabled -- the local ruff/codespell gate will not run.\n' >&2
    printf '          enable them once per clone:  git config core.hooksPath .githooks\n' >&2
    return 0
  fi

  case "$configured" in
    /*) resolved="$configured" ;;
    *)  resolved="$top/$configured" ;;
  esac

  [ -x "$resolved/pre-push" ] && return 0

  printf 'fastgate: git hooks are NOT wired up -- core.hooksPath = %s\n' "$configured" >&2
  printf '          resolves to %s, which holds no executable pre-push,\n' "$resolved" >&2
  printf '          so the ruff/codespell gate will NOT run before a push.\n' >&2
  if [ -n "$(git config --worktree --get core.hooksPath 2>/dev/null || true)" ]; then
    printf '          this worktree overrides the repository setting; drop the override with:\n' >&2
    printf '            git config --worktree --unset core.hooksPath\n' >&2
  else
    printf '          point it at this checkout with:  git config core.hooksPath .githooks\n' >&2
  fi
}

check_hook_wiring

TIER="${1:-fast}"
[ $# -gt 0 ] && shift

case "$TIER" in
  fast)       MARK='not slow and not validation' ;;
  slow)       MARK='slow' ;;
  validation) MARK='validation' ;;
  all)        MARK='' ;;
  -*)         set -- "$TIER" "$@"; MARK='not slow and not validation' ;;
  *)          printf 'fastgate: unknown tier %s (fast|slow|validation|all)\n' "$TIER" >&2; exit 2 ;;
esac

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="${TMPDIR:-/tmp}/aquaflux-tests-${TIER}-${STAMP}.log"

printf 'running the %s tier -> %s\n' "$TIER" "$LOG"

# Unbuffered and redirected. The point of the file is that a long run can be watched WHILE it runs
# (`tail -f` on the file is fine -- that is a reader, not the run's own stdout).
if [ -n "$MARK" ]; then
  python3 -u -m pytest -q -m "$MARK" "$@" > "$LOG" 2>&1
else
  python3 -u -m pytest -q "$@" > "$LOG" 2>&1
fi
STATUS=$?

echo
# The summary is matched by SHAPE, not by position: this suite emits library shutdown output after it,
# so `tail -n` is not a reliable way to find it.
SUMMARY=$(grep -aE '^[0-9]+ (passed|failed)|[0-9]+ (passed|failed|error)' "$LOG" | tail -1 || true)
if [ -n "$SUMMARY" ]; then
  printf 'result: %s\n' "$SUMMARY"
else
  printf 'result: no pytest summary line found -- the run did not reach one.\n'
  printf 'last lines of %s:\n' "$LOG"
  tail -15 "$LOG" | sed 's/^/  /'
fi

if [ "$STATUS" -ne 0 ]; then
  printf '\nFAILED (pytest exit %s). Failures:\n' "$STATUS"
  grep -aE '^(FAILED|ERROR)' "$LOG" | sed 's/^/  /' | head -40
  printf '\nfull log: %s\n' "$LOG"
else
  printf 'pytest exit 0; full log: %s\n' "$LOG"
fi

exit "$STATUS"
