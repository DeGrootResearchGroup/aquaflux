#!/usr/bin/env bash
#
# Run a long validation case the way a long validation case has to be run.
#
# A validation march takes 30-60 minutes, saturates the machine, and leaves one artifact that matters:
# its log. Every one of the ways that goes wrong has happened here, and each cost real time:
#
#   * the output was piped through `tail`, which buffers to EOF -- so the run was invisible while it
#     ran AND its summary was lost when the pipeline's exit code (always 0, from `tail`) was read as
#     the result;
#   * the process was backgrounded in a way that did not survive its parent, and died seconds in;
#   * the machine slept mid-run, so the log's wall-clock column silently absorbed the sleep and the
#     numbers could not be compared against anything;
#   * two sessions started a case in the same working tree at once, each believing the other's process
#     was its own, on a machine with barely enough memory for one;
#   * a waiter looking for the run with `pgrep -f compare.py` matched ITSELF and waited forever.
#
# None of those are knowledge problems, so this script exists to make them unavailable rather than
# documented. It runs the case unbuffered, redirects (never pipes) to a timestamped log, holds the
# machine awake, refuses to start alongside another case, and records what it started in a run-file so
# that "is this run mine, and what is it testing?" is a question with a written answer.
#
# The run-file is deliberately MACHINE-GLOBAL rather than per-worktree: the resource being contended
# is the machine's memory, and the collision that actually happened was between two sessions sharing
# one tree. A per-tree lock would not have caught it, and would not catch two worktrees either.
#
# Usage
#   validation/run_case.sh <script.py>          launch, print how to watch, return immediately
#   validation/run_case.sh <script.py> --wait   launch and block until it exits
#   validation/run_case.sh --status             what is running, since when, under what settings
#   validation/run_case.sh --wait               block on whatever is already running
#   validation/run_case.sh <script.py> --force  start even if the health pre-flight objects
#
# Case settings are passed as environment, and are recorded in the run-file verbatim:
#   BFS3D_K_WALL=dirichlet validation/run_case.sh validation/bfs3d_openfoam/compare.py --wait

set -euo pipefail

RUN_FILE="${TMPDIR:-/tmp}/aquaflux-case-run"
MIN_FREE_GB="${AQUAFLUX_MIN_FREE_GB:-5}"
MAX_LOAD="${AQUAFLUX_MAX_LOAD:-8}"

die() { printf 'run_case: %s\n' "$1" >&2; exit 1; }

# Free-ish memory in whole GB. `Pages free` alone reads near zero on a warm machine and would block
# every launch; inactive pages are reclaimable, so the two together are the number a launch cares about.
free_gb() {
  python3 -c "
import re, subprocess
out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
size = int(re.search(r'page size of (\d+)', out).group(1))
pages = dict(re.findall(r'^(.*?):\s+(\d+)\.', out, re.M))
free = int(pages['Pages free']) + int(pages['Pages inactive'])
print(int(free * size / 1e9))
"
}

load_1min() { uptime | sed 's/.*load averages*: *//' | awk '{print $1}' | tr -d ','; }

# How long the machine slept since a given "YYYY-MM-DD HH:MM:SS", and how many times.
#
# `caffeinate` holds off IDLE sleep, but it does NOT stop a deliberate suspend -- closing the lid puts
# the machine out regardless, and sometimes a run has to survive a commute. The run keeps converging
# correctly across that, so nothing in the march log looks wrong; only its WALL CLOCK is silently
# wrong, having counted the sleep as compute. A run measured that way is void for cost and looks
# identical to one that is not. So the run records its own sleep, and a contaminated run declares
# itself rather than relying on someone thinking to check afterwards.
#
# Counts each Sleep..(Wake|DarkWake) pair. DarkWake resumes enough of the machine for a compute
# process to make progress, so treating it as the end of a sleep UNDERSTATES the loss slightly --
# which is the right direction for a warning that exists to make you distrust a number.
sleep_since() {
  python3 - "$1" <<'PYEOF' 2>/dev/null || echo "0 0"
import datetime as dt, re, subprocess, sys
start = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M:%S")
try:
    out = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True, timeout=30).stdout
except Exception:
    print("0 0"); raise SystemExit
# pmset states each sleep's DURATION on its own Sleep line ("... 780 secs"), which is its own
# accounting and is what to trust. Pairing Sleep with the following Wake looks equivalent and is not:
# the log interleaves DarkWake and maintenance events, so the pairing silently under-counts -- an
# earlier version of this returned 15 s for a window containing an 18-minute suspend.
total = count = 0
for line in out.splitlines():
    m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s[-+]\d+\s+Sleep\s", line)
    if not m or dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") < start:
        continue
    secs = re.search(r"(\d+)\s+secs\s*$", line)
    if secs:
        total += int(secs.group(1)); count += 1
print(f"{count} {total}")
PYEOF
}

# The PID of a live run, or empty. Reads the run-file rather than scanning the process table: a
# `pgrep -f <script>` matches any shell whose own command line mentions the script, including the
# waiter doing the matching, which is a deadlock that has happened here.
live_pid() {
  [ -f "$RUN_FILE" ] || return 0
  local pid
  pid=$(sed -n 's/^pid=//p' "$RUN_FILE")
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then printf '%s' "$pid"; else rm -f "$RUN_FILE"; fi
}

show_status() {
  local pid
  pid=$(live_pid)
  if [ -z "$pid" ]; then echo "no case is running"; return 0; fi
  echo "a case IS running:"
  sed 's/^/  /' "$RUN_FILE"
  echo "  elapsed: $(ps -o etime= -p "$pid" | tr -d ' ')"
}

# Block until the recorded run exits. Polls the recorded PID, so it cannot match itself.
wait_for_run() {
  local pid log
  pid=$(live_pid)
  if [ -z "$pid" ]; then echo "no case is running"; return 0; fi
  log=$(sed -n 's/^log=//p' "$RUN_FILE")
  echo "waiting on pid $pid; log: $log"
  while kill -0 "$pid" 2>/dev/null; do sleep 20; done
  echo "case exited"
  [ -n "$log" ] && [ -f "$log" ] && tail -5 "$log"
  return 0
}

WANT_WAIT=0
FORCE=0
SCRIPT=""
for arg in "$@"; do
  case "$arg" in
    --wait)   WANT_WAIT=1 ;;
    --force)  FORCE=1 ;;
    --status) show_status; exit 0 ;;
    -*)       die "unknown option $arg" ;;
    *)        SCRIPT="$arg" ;;
  esac
done

if [ -z "$SCRIPT" ]; then
  [ "$WANT_WAIT" -eq 1 ] && { wait_for_run; exit 0; }
  die "no script given (try --status)"
fi
[ -f "$SCRIPT" ] || die "no such script: $SCRIPT"

# --- refuse to run two at once -------------------------------------------------------------------
# This is the memory guard, not a tidiness rule: one materialized 3D coupled Jacobian is ~2 GB, and
# concurrent runs have exhausted this machine and suspended every application on it, including the
# session driving them -- a state that cannot be debugged from inside.
EXISTING=$(live_pid)
if [ -n "$EXISTING" ]; then
  echo "run_case: a case is ALREADY running -- refusing to start a second." >&2
  echo >&2
  sed 's/^/  /' "$RUN_FILE" >&2
  echo >&2
  echo "  Wait for it (validation/run_case.sh --wait), or stop it (kill $EXISTING)." >&2
  exit 1
fi

# --- health pre-flight ----------------------------------------------------------------------------
FREE=$(free_gb)
LOAD=$(load_1min)
if [ "$FORCE" -eq 0 ]; then
  if [ "$FREE" -lt "$MIN_FREE_GB" ]; then
    die "only ${FREE} GB free (want >= ${MIN_FREE_GB}). Wait, or --force."
  fi
  if [ "$(echo "$LOAD > $MAX_LOAD" | bc -l)" = "1" ]; then
    die "1-minute load is ${LOAD} (want <= ${MAX_LOAD}). Wait, or --force."
  fi
fi

# The compiled zero-fill incomplete factorization, which the coupled preconditioner's CPU path spends
# most of its time in. It is a gitignored artifact, so it belongs to a CHECKOUT and a fresh worktree
# starts without it -- and `aquaflux.solve.ilu0` then falls back to its pure-Python reference twin
# WITHOUT SAYING SO. Nothing breaks and no number changes; only the wall clock does, by orders of
# magnitude, which quietly turns a preconditioner comparison into a comparison of two languages. This
# warns rather than refuses, because a run that only wants step and cycle counts is unaffected -- those
# are identical either way. Never fails the launch: a warning is not a gate.
if ! python3 -c 'from aquaflux.solve.ilu0 import COMPILED; raise SystemExit(0 if COMPILED else 1)' \
     >/dev/null 2>&1; then
  printf 'run_case: WARNING -- the compiled incomplete factorization is NOT built in this checkout.\n' >&2
  printf '          Timings from this run are not comparable to any other; step and cycle counts are\n' >&2
  printf '          unaffected. Build it with:  tools/build_ext.sh\n' >&2
fi

# --- launch ----------------------------------------------------------------------------------------
STAMP=$(date +%Y%m%d-%H%M%S)
STARTED_AT=$(date "+%Y-%m-%d %H:%M:%S")
LOG="$(cd "$(dirname "$SCRIPT")" && pwd)/run-${STAMP}.log"

# `caffeinate -ims` holds the machine awake for the run's lifetime. A march that spans a sleep keeps
# converging correctly but its per-step wall-clock silently absorbs the sleep, which makes the log
# useless for the cost comparison it was run for -- and nothing in the log says so. Absent elsewhere
# than macOS, so it is optional rather than required.
HOLD_AWAKE=""
command -v caffeinate >/dev/null 2>&1 && HOLD_AWAKE="caffeinate -ims"

# Unbuffered, and REDIRECTED rather than piped. A pipe through `tail`/`head` buffers to EOF, so the
# run is invisible while it matters and the exit status read afterwards belongs to the pipe's last
# stage rather than to the case.
# shellcheck disable=SC2086
$HOLD_AWAKE python3 -u "$SCRIPT" > "$LOG" 2>&1 &
PID=$!

{
  echo "pid=$PID"
  echo "script=$SCRIPT"
  echo "log=$LOG"
  echo "started=$STARTED_AT"
  echo "worktree=$(pwd)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(not a git tree)')"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo '-')"
  # The case settings, verbatim. Two runs differing only in an environment variable have otherwise
  # produced logs identical to the character, leaving launch order as the only way to tell them apart.
  # Every prefix a case reads its settings from belongs here: one that is missing does not announce
  # itself, it just silently drops the one line that says what the run was testing.
  env | grep -E '^(BFS3D|AQUAFLUX|ILU0_SWEEP|PROBE)_' | sort | sed 's/^/env: /' || true
} > "$RUN_FILE"

# Appended to the run's OWN log, not only printed, so the warning travels with the artifact it
# invalidates -- a log read months later carries its own provenance.
report_sleep() {
  local started="$1" log="$2" n secs
  read -r n secs <<< "$(sleep_since "$started")"
  if [ "${n:-0}" -gt 0 ] 2>/dev/null && [ "${secs:-0}" -gt 60 ] 2>/dev/null; then
    {
      echo
      echo "[!] THIS RUN SPANNED $n MACHINE SLEEP(S), ~$((secs / 60)) MIN TOTAL."
      echo "[!] Its wall-clock columns counted that as compute and are VOID for cost comparison."
      echo "[!] Step counts, cycle counts and residuals are unaffected -- use those."
    } | tee -a "$log"
  fi
}

echo "launched pid $PID"
sed 's/^/  /' "$RUN_FILE"
echo
echo "  watch:  tail -f $LOG"
echo "  status: validation/run_case.sh --status"

if [ "$WANT_WAIT" -eq 1 ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 20; done
  echo "case exited"
  tail -5 "$LOG"
  report_sleep "$STARTED_AT" "$LOG"
fi
