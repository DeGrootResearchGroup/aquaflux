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
# Exits with pytest's status, so it composes in a shell `&&` chain and in a hook. Three statuses are
# the script's own rather than pytest's: 2 for a tier it does not recognize, 3 for refusing to start
# beside a running validation case, and 4 for refusing to start beside another running tier (both
# below).
#
# It REFUSES TO START WHILE A VALIDATION CASE IS RUNNING, because the mutual exclusion those cases rely
# on is over *cases* and a test tier is not one -- so nothing stopped a gate landing on top of a march,
# and on 2026-09-09 that happened three times in one evening between three sessions that all knew the
# one-heavy-job-at-a-time rule. It is not a knowledge problem: the rule is ENFORCED for case-vs-case and
# merely known for tier-vs-case, so this closes the half that was only known.
#
# The cost is symmetric, which is the part worth knowing -- it is not a trade of one job's latency for
# another's throughput. In the measured collision the fast tier alone took 17:25 against its usual
# 6:34-8:53 while the case it landed on ran 3.2x slow over the overlap. Both jobs lost; run end to end
# they are about 8 and 9 minutes.
#
# It also REFUSES TO START BESIDE ANOTHER RUNNING TIER, anywhere on the machine -- a separate mutual
# exclusion from the case guard above, because a tier can collide with either. On 2026-09-13 two fast
# tiers started five minutes apart, from two different worktrees, neither touching a case, and the
# machine had to be hard-reset: a fast tier's real memory footprint (mostly compressed pages, which
# RSS does not count) runs to tens of GB across its worker pool, so two at once is roughly double that
# on a machine that does not have it. A machine-wide lock file makes this the same kind of ENFORCED
# rule the case guard is, rather than a second thing to merely remember.
#
# Override with FASTGATE_FORCE=1 when you mean it (a quick `-k` on one test beside a long march, or
# beside another quick run, is usually harmless). Both checks are skipped under CI, which runs neither
# cases nor concurrent tiers.
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

# --- refuse to start beside a running validation case --------------------------------------------
# Asks `run_case.sh` rather than reading the run-file here: that script owns the file's format and the
# liveness rule, and a second copy of either is the duplication that made this class of bug possible in
# the first place. A missing or unrunnable script is not this gate's problem, so it degrades to running.
RUN_CASE="$(dirname "$0")/../validation/run_case.sh"
if [ -z "${CI:-}" ] && [ -z "${FASTGATE_FORCE:-}" ] && [ -x "$RUN_CASE" ]; then
  if CASE_PID=$("$RUN_CASE" --running 2>/dev/null); then
    printf 'fastgate: a validation case is running (pid %s) -- refusing to start a test tier.\n' \
      "$CASE_PID" >&2
    printf '\n' >&2
    "$RUN_CASE" --status 2>/dev/null | sed 's/^/  /' >&2
    printf '\n' >&2
    printf '  Both jobs lose when these overlap: the case slows and so does the tier.\n' >&2
    printf '  Wait for it (validation/run_case.sh --wait), or FASTGATE_FORCE=1 to run anyway.\n' >&2
    exit 3
  fi
fi

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

# --- refuse to start beside ANOTHER running tier, from any worktree or session -------------------
# A different mutual exclusion from the case guard above (that one is tier-vs-case; this one is
# tier-vs-tier), because #379 was two fast tiers colliding with each other, not with a case. The lock
# lives under a fixed, machine-global directory rather than $TMPDIR -- $TMPDIR's visibility across
# worktrees and sessions is exactly the ambiguity the CHECKOUT-qualified log name below routes around,
# and a lock a competing shell could miss is worse than no lock. `~/.cache/aquaflux` is already
# machine-global (it holds the compiled ILU(0) kernel and the JAX compilation cache), so every
# invocation of this script looks in the same place regardless of which worktree started it.
LOCK_DIR="${FASTGATE_LOCK_DIR:-$HOME/.cache/aquaflux}"
LOCK_FILE="$LOCK_DIR/fastgate.lock"
mkdir -p "$LOCK_DIR" 2>/dev/null || true

# A lock held by a pid that no longer exists is a crashed run's leftovers, not a live claim -- the
# same liveness rule `run_case.sh` uses for its own run-file (`kill -0`, never "the file exists").
tier_lock_holder() {
  [ -f "$LOCK_FILE" ] || return 0
  local pid
  pid=$(sed -n 's/^pid=//p' "$LOCK_FILE")
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then printf '%s' "$pid"; else rm -f "$LOCK_FILE"; fi
}

if [ -z "${CI:-}" ] && [ -z "${FASTGATE_FORCE:-}" ]; then
  HOLDER=$(tier_lock_holder)
  if [ -n "$HOLDER" ]; then
    printf 'fastgate: another test tier is already running (pid %s) -- refusing to start a second.\n' \
      "$HOLDER" >&2
    printf '\n' >&2
    sed 's/^/  /' "$LOCK_FILE" >&2
    printf '\n' >&2
    printf '  Two tiers at once is how this machine got hard-reset (#379): one alone already holds\n' >&2
    printf '  tens of GB of real, mostly-compressed footprint that a quick RSS check does not show.\n' >&2
    printf '  Wait for it, or FASTGATE_FORCE=1 to run anyway.\n' >&2
    exit 4
  fi
  # Atomic create (bash's noclobber refuses an existing target the same way O_EXCL does): a competing
  # fastgate racing to this exact line loses HERE, not later -- there is no window where both believe
  # they hold the lock, except the one closed by the recheck immediately below.
  if ( set -o noclobber
       { printf 'pid=%s\ncheckout=%s\ntier=%s\nstarted=%s\n' \
           "$$" "$CHECKOUT" "$TIER" "$(date '+%Y-%m-%d %H:%M:%S')" > "$LOCK_FILE"
       } ) 2>/dev/null; then
    trap 'rm -f "$LOCK_FILE"' EXIT
  else
    # The create failed either because a concurrent fastgate won the race between our check and our
    # attempt (recheck and refuse), or because the lock file/directory is not writable for some other
    # reason -- in which case there is no lock to enforce, and this degrades to running rather than
    # blocking every tier on a broken cache directory, the same choice the case guard makes when
    # run_case.sh itself is missing.
    HOLDER=$(tier_lock_holder)
    if [ -n "$HOLDER" ]; then
      printf 'fastgate: another test tier just started (pid %s) -- refusing to start a second.\n' \
        "$HOLDER" >&2
      exit 4
    fi
  fi
fi

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
