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
#   * a waiter looking for the run with `pgrep -f compare.py` matched ITSELF and waited forever;
#   * `--wait` exited 0 for a case that had died -- of a `ModuleNotFoundError`, and of an out-of-memory
#     kill that left no traceback at all -- because the case ran detached and nothing collected its
#     status, so `run_case.sh ... --wait && next-step` went on to the next step after a crash.
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
#   validation/run_case.sh <script.py> --wait   launch, block until it exits, exit with ITS status
#   validation/run_case.sh --status             what is running, since when, under what settings
#   validation/run_case.sh --running            exit 0 and print the pid if a case is live, else 1
#   validation/run_case.sh --wait               block on whatever is already running, and exit
#                                               with its status
#   validation/run_case.sh <script.py> --force  start even if the health pre-flight objects
#
# Case settings are passed as environment, and are recorded in the run-file verbatim:
#   BFS3D_K_WALL=dirichlet validation/run_case.sh validation/bfs3d_openfoam/compare.py --wait

set -euo pipefail

RUN_FILE="${TMPDIR:-/tmp}/aquaflux-case-run"
MIN_FREE_GB="${AQUAFLUX_MIN_FREE_GB:-5}"
MAX_LOAD="${AQUAFLUX_MAX_LOAD:-8}"
# How often a waiter polls the case's pid. Twenty seconds is nothing against a 30-60 minute march; the
# override exists so a test of the waiter does not spend twenty seconds per case.
POLL_SECONDS="${AQUAFLUX_CASE_POLL_SECONDS:-20}"

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

# The same question as `--status`, answered by EXIT STATUS so another script can branch on it without
# parsing prose. It exists because `tools/fastgate.sh` has to ask it before starting a test tier, and a
# second implementation of "is a case live" would be one more copy of the run-file's format and of the
# `kill -0` liveness rule -- the two things this file exists to own. Prints the pid so a caller can name
# it in its own message.
is_running() {
  local pid
  pid=$(live_pid)
  [ -n "$pid" ] || return 1
  printf '%s\n' "$pid"
}

show_status() {
  local pid
  pid=$(live_pid)
  if [ -z "$pid" ]; then echo "no case is running"; return 0; fi
  echo "a case IS running:"
  sed 's/^/  /' "$RUN_FILE"
  echo "  elapsed: $(ps -o etime= -p "$pid" | tr -d ' ')"
}

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

# Where the case's exit status is written: beside its log, named after it, so a waiter that knows the
# log knows where to look. Not in the run-file, which `live_pid` deletes the moment the pid is gone --
# i.e. exactly when the status is wanted, and possibly by some other caller's `--status`.
status_file_for() { printf '%s' "${1%.log}.exit"; }

# Block until the case with this pid exits, then report how it went and RETURN ITS EXIT STATUS.
#
# Polls `kill -0` on the recorded pid, so it cannot match itself the way a `pgrep -f` does. The pid is
# the launch wrapper's, which writes the status file before it exits -- so once `kill -0` fails the
# status is already on disk, and there is no window in which a finished case reads as "no status".
#
# A missing status file is reported as a failure, not as success: it means the wrapper itself was
# killed (or the run was launched by a version of this script that did not record one), and "we do not
# know how it ended" is not a result anything should proceed on.
await_case() {
  local pid="$1" log="$2" started="$3" status_file status
  while kill -0 "$pid" 2>/dev/null; do sleep "$POLL_SECONDS"; done
  echo "case exited"
  [ -n "$log" ] && [ -f "$log" ] && tail -5 "$log"
  [ -n "$started" ] && [ -n "$log" ] && [ -f "$log" ] && report_sleep "$started" "$log"
  status_file=$(status_file_for "$log")
  status=$(cat "$status_file" 2>/dev/null || true)
  if ! [[ "$status" =~ ^[0-9]+$ ]]; then
    echo "run_case: NO exit status was recorded (expected $status_file) -- treating the run as failed." >&2
    return 1
  fi
  echo "exit status: $status"
  return "$status"
}

# `--wait` with no script: block on whatever the run-file names.
wait_for_run() {
  local pid log started
  pid=$(live_pid)
  if [ -z "$pid" ]; then echo "no case is running"; return 0; fi
  log=$(sed -n 's/^log=//p' "$RUN_FILE")
  started=$(sed -n 's/^started=//p' "$RUN_FILE")
  echo "waiting on pid $pid; log: $log"
  await_case "$pid" "$log" "$started"
}

WANT_WAIT=0
FORCE=0
SCRIPT=""
for arg in "$@"; do
  case "$arg" in
    --wait)   WANT_WAIT=1 ;;
    --force)  FORCE=1 ;;
    --status)  show_status; exit 0 ;;
    # `if`, not `is_running; exit $?`: under `set -e` a bare failing call exits the shell before
    # the `exit` runs -- with the right status here by luck, which is not a thing to rely on.
    --running) if is_running; then exit 0; else exit 1; fi ;;
    -*)       die "unknown option $arg" ;;
    *)        SCRIPT="$arg" ;;
  esac
done

if [ -z "$SCRIPT" ]; then
  if [ "$WANT_WAIT" -eq 1 ]; then status=0; wait_for_run || status=$?; exit "$status"; fi
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
# Measured only when it will be judged: `free_gb` reads `vm_stat`, which exists only on macOS, so taking
# the reading unconditionally made `--force` fail there too, on a number it was about to ignore.
if [ "$FORCE" -eq 0 ]; then
  FREE=$(free_gb)
  LOAD=$(load_1min)
  if [ "$FREE" -lt "$MIN_FREE_GB" ]; then
    die "only ${FREE} GB free (want >= ${MIN_FREE_GB}). Wait, or --force."
  fi
  if [ "$(echo "$LOAD > $MAX_LOAD" | bc -l)" = "1" ]; then
    die "1-minute load is ${LOAD} (want <= ${MAX_LOAD}). Wait, or --force."
  fi
fi

# --- launch ----------------------------------------------------------------------------------------
STAMP=$(date +%Y%m%d-%H%M%S)
STARTED_AT=$(date "+%Y-%m-%d %H:%M:%S")
LOG="$(cd "$(dirname "$SCRIPT")" && pwd)/run-${STAMP}.log"

STATUS_FILE=$(status_file_for "$LOG")

# The case runs inside a wrapper that outlives it by exactly one line: the one recording its exit
# status. The case is detached -- it must survive this script, and `--wait` on an already-running case
# is not its parent -- so no waiter can `wait` for it, and without the wrapper its status went nowhere
# and every `--wait` exited 0 however the case ended.
#
#   * `pid=` in the run-file is the WRAPPER's. It is alive exactly as long as the case plus the status
#     write, so `kill -0` on it stays the liveness test and a finished case always has its status on
#     disk by the time a waiter notices it has gone.
#   * The case runs as the wrapper's background child and the wrapper `wait`s on it, so a signal sent
#     to the recorded pid (`kill <pid>`, the advice this script gives) is FORWARDED rather than killing
#     the wrapper and orphaning a case that would then run on unseen. `wait` returns early when a
#     trapped signal lands, hence the loop: it waits again until the case itself is gone.
#   * A death by signal is recorded as 128+N, the shell's own convention -- so an out-of-memory SIGKILL,
#     which leaves no traceback in the log, still reads as 137 rather than as nothing.
#   * The status also goes at the foot of the log, so the artifact read months later says how it ended.
#   * The wrapper's own stdio is detached. It would otherwise hold the caller's stdout open for the
#     case's whole lifetime, so a caller capturing this script's output -- `$(run_case.sh x.py)`, or a
#     test -- would block until the case ended even without `--wait`.
#
# Unbuffered, and REDIRECTED rather than piped. A pipe through `tail`/`head` buffers to EOF, so the
# run is invisible while it matters and the exit status read afterwards belongs to the pipe's last
# stage rather than to the case.
(
  set +e
  python3 -u "$SCRIPT" > "$LOG" 2>&1 &
  case_pid=$!
  trap 'kill -TERM "$case_pid" 2>/dev/null' TERM
  trap 'kill -HUP "$case_pid" 2>/dev/null' HUP
  wait "$case_pid"
  status=$?
  while kill -0 "$case_pid" 2>/dev/null; do
    wait "$case_pid"
    status=$?
  done
  if [ "$status" -gt 128 ]; then how=" (killed by signal $((status - 128)))"; else how=""; fi
  printf '\n[run_case] the case exited with status %s%s\n' "$status" "$how" >> "$LOG"
  printf '%s\n' "$status" > "$STATUS_FILE"
) < /dev/null > /dev/null 2>&1 &
PID=$!

# `caffeinate` holds the machine awake for as long as the run's pid lives. A march that spans a sleep
# keeps converging correctly but its per-step wall-clock silently absorbs the sleep, which makes the log
# useless for the cost comparison it was run for -- and nothing in the log says so. It watches the pid
# (`-w`) rather than wrapping the case, so it stays out of the path the exit status travels. Absent
# elsewhere than macOS, so it is optional rather than required.
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -ims -w "$PID" < /dev/null > /dev/null 2>&1 &
fi

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
  # itself, it just silently drops the one line that says what the run was testing. UV_ was missing
  # until 2026-08-22, and UV_MESH is the path to a gitignored 1.6M-cell mesh that lives outside the
  # worktree -- so its run-files recorded a case whose input could no longer be identified, and the
  # mesh had to be hunted for across checkouts before the run could be repeated. PROFILE_, CONSISTENCY_,
  # FLOW_ and TAPER_ were missing the same way until 2026-09-15 -- tools/check_env_prefixes.py now
  # gates on this list falling behind the prefixes actually read under validation/ again.
  env | grep -E '^(BFS3D|PITZ|UV|AQUAFLUX|ILU0_SWEEP|PROBE|PROFILE|CONSISTENCY|FLOW|TAPER|LAM|TET|SOZZI|RC|REF|BV|BP|BL)_' | sort | sed 's/^/env: /' || true
} > "$RUN_FILE"

echo "launched pid $PID"
sed 's/^/  /' "$RUN_FILE"
echo
echo "  watch:  tail -f $LOG"
echo "  status: validation/run_case.sh --status"

if [ "$WANT_WAIT" -eq 1 ]; then
  status=0
  await_case "$PID" "$LOG" "$STARTED_AT" || status=$?
  exit "$status"
fi
