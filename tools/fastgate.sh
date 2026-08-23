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
# The FAST tier runs across worker processes (pytest-xdist), because it is the tier that runs on
# every change and its wall clock is what makes or breaks the edit-test loop. Three details are
# load-bearing and are the reason this is not simply `-n auto`:
#
#   * Only the fast tier. The `slow` and `validation` tiers hold multi-minute solves whose live
#     JAX buffers run to gigabytes apiece -- a materialized 3D coupled Jacobian is ~2 GB per copy --
#     so running several at once is how this machine gets driven into swap, taking every other
#     application with it. Those tiers stay serial, which is also what CI does (it shards them
#     across jobs at `-n 1`, never within one).
#   * `--dist loadfile`, not the default per-test distribution. A file's tests stay together on one
#     worker, so a module-scoped fixture is built once rather than once per worker, and each file's
#     tests keep their recorded order. The heavy integration modules here are exactly the ones with
#     expensive module fixtures.
#   * One BLAS/XLA thread per worker. Left alone, every worker grabs every core, and N workers x M
#     cores thrashes rather than scales -- the same pinning the CI workflow sets for the same reason.
#     XLA_FLAGS is APPENDED to, never assigned: several distributed tests add their own device-count
#     flag to it, and assigning would silently drop whatever the caller already set.
#
# Override with FASTGATE_JOBS: a worker count (`FASTGATE_JOBS=4`), or `0` to run serially. Passing
# your own `-n`/`--numprocesses` also wins, so a one-off serial run is `tools/fastgate.sh fast -n0`.
#
# It also warns, before running anything, when the committed hooks in .githooks/ are not wired up
# (or are wired only by coincidence). That check lives in tools/check_hooks.sh, which always exits
# 0 -- the warning never changes this script's exit status; only pytest does.

set -uo pipefail

"$(dirname "$0")/check_hooks.sh" || true

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

# The log name carries the CHECKOUT it came from, not just the tier and the time. Several worktrees
# of this repository share one $TMPDIR, so without it two concurrent runs write indistinguishable
# names into the same directory and `ls -t ... | head -1` returns whichever wrote most recently --
# which is not necessarily yours. Reading another checkout's log as your own is not a hypothetical:
# it makes a run look like it restarted, stalled, or died, and every conclusion drawn from it is
# about someone else's tree. This is the question `validation/run_case.sh` answers with a run-file --
# "is this run mine, and what is it testing?" -- asked of the test tiers instead.
CHECKOUT=$(basename "$(cd "$(dirname "$0")/.." && pwd -P)")
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="${TMPDIR:-/tmp}/aquaflux-tests-${CHECKOUT}-${TIER}-${STAMP}.log"

printf 'running the %s tier in %s -> %s\n' "$TIER" "$CHECKOUT" "$LOG"

# Worker processes for the fast tier only -- see the header for why this is not a blanket `-n auto`.
# A caller who names their own `-n`/`--numprocesses` keeps it; FASTGATE_JOBS=0 opts out entirely.
PARALLEL=()
JOBS="${FASTGATE_JOBS:-auto}"
case " $* " in
  *" -n"*|*" --numprocesses"*) JOBS=0 ;;   # the caller has chosen; do not add a second -n
esac
# pytest-xdist ships in the `test` extra, but an environment without it must still be able to run
# the gate: an unrecognized `-n` would fail the whole run for a reason that has nothing to do with
# the tests. Fall back to a serial run instead, and say so rather than silently going slow.
if [ "$TIER" = "fast" ] && [ "$JOBS" != "0" ] && ! python3 -c "import xdist" 2>/dev/null; then
  printf 'fastgate: pytest-xdist is not installed -- running serially. `pip install -e ".[test]"` to parallelize.\n' >&2
  JOBS=0
fi
if [ "$TIER" = "fast" ] && [ "$JOBS" != "0" ]; then
  PARALLEL=(-n "$JOBS" --dist loadfile --max-worker-restart=0)
  # One thread per worker, so N workers do not each try to use the whole machine. Appended, so a
  # caller's existing XLA_FLAGS survives.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
fi

# Unbuffered and redirected. The point of the file is that a long run can be watched WHILE it runs
# (`tail -f` on the file is fine -- that is a reader, not the run's own stdout).
# `${PARALLEL[@]+...}` rather than a bare `"${PARALLEL[@]}"`: under `set -u`, bash 3.2 -- which is
# what /bin/bash is on macOS -- treats expanding an EMPTY array as an unbound variable and aborts.
# That would take out precisely the serial tiers, and only on some machines.
if [ -n "$MARK" ]; then
  python3 -u -m pytest -q -m "$MARK" ${PARALLEL[@]+"${PARALLEL[@]}"} "$@" > "$LOG" 2>&1
else
  python3 -u -m pytest -q ${PARALLEL[@]+"${PARALLEL[@]}"} "$@" > "$LOG" 2>&1
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
