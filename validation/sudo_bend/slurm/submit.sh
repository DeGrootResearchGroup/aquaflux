#!/bin/bash
# Submit the bend's grid study to Slurm: one job per grid, then the GCI once all three succeed.
#
#   validation/sudo_bend/slurm/submit.sh <account> [grid ...]
#
# <account> is the allocation to charge (def-<pi>, rrg-<pi>, ...); the grids default to all three.
# Run from anywhere in the repository; every job runs from its root and writes into the grids'
# work/ directories, its own output to validation/sudo_bend/slurm/logs/<name>-<job id>.out.
#
# SUDO_WORK_ROOT puts the work/ directories on another file system (a cluster's scratch, say):
# each grid's work/ becomes a link to $SUDO_WORK_ROOT/<grid>, its contents (meshes, finished runs)
# moved there first. Everything a job writes but its log goes through work/. A grid job
# that runs out of time resumes when resubmitted: meshes and converged runs are kept, but a run
# that was cut off starts again from its beginning.
#
# Resources per grid (cores, memory, time). The memory is the march's estimate (about 200 KB a
# bend cell, from the coarse grid's 10.5 GB) with room to spare -- a job killed for memory costs a
# whole resubmission. The times are generous guesses from the coarse grid on 4 cores (precursor 10
# minutes, bend 40) and the fine precursor (3 hours on 4 cores); edit them here once a run has told
# you better.
set -euo pipefail

account="${1:?usage: submit.sh <account> [grid ...]}"
shift
grids=("$@")
[ ${#grids[@]} -gt 0 ] || grids=(round36 round48 round64)

declare -A cores=([round36]=8 [round48]=12 [round64]=16)
declare -A memory=([round36]=24G [round48]=48G [round64]=96G)
declare -A walltime=([round36]=0-06:00:00 [round48]=1-00:00:00 [round64]=3-00:00:00)

root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
logs="$root/validation/sudo_bend/slurm/logs"
mkdir -p "$logs"

# The jobs' environment must run aquaflux's command line (its own dependencies, not only the
# source): a broken install fails here, before any job queues, rather than minutes into each one.
python="${SUDO_VENV:-$HOME/venvs/aquaflux}/bin/python"
(cd "$root" && "$python" -m aquaflux --help >/dev/null) || {
  echo "$python cannot run aquaflux; in that environment: pip install -e \"$root\"" >&2; exit 2; }

ids=()
for grid in "${grids[@]}"; do
  [ -n "${cores[$grid]-}" ] || { echo "unknown grid $grid (round36, round48, round64)" >&2; exit 2; }
  if [ -n "${SUDO_WORK_ROOT-}" ]; then
    work="$root/validation/sudo_bend/grids/$grid/work" target="$SUDO_WORK_ROOT/$grid"
    mkdir -p "$target"
    if [ -d "$work" ] && [ ! -L "$work" ]; then
      # -n: never overwrite what the target already holds; anything left behind stops the move.
      [ -z "$(ls -A "$work")" ] || mv -n "$work"/* "$target"/
      rmdir "$work" || { echo "$work: not everything moved to $target; sort it out first" >&2; exit 2; }
    fi
    [ -L "$work" ] || ln -s "$target" "$work"
    [ "$(readlink "$work")" = "$target" ] || {
      echo "$work links to $(readlink "$work"), not $target" >&2; exit 2; }
    echo "$grid: work/ -> $target"
  fi
  id=$(sbatch --parsable --account="$account" --chdir="$root" \
    --job-name="sudo-$grid" --output="$logs/%x-%j.out" \
    --cpus-per-task="${cores[$grid]}" --mem="${memory[$grid]}" --time="${walltime[$grid]}" \
    --export=ALL,SUDO_GRIDS="$grid" \
    "$root/validation/sudo_bend/slurm/grid.sbatch")
  id="${id%%;*}"  # --parsable prints "<id>;<cluster>" on a federated cluster
  echo "$grid: job $id (${cores[$grid]} cores, ${memory[$grid]}, ${walltime[$grid]})"
  ids+=("$id")
done

dependency=$(IFS=:; echo "afterok:${ids[*]}")
id=$(sbatch --parsable --account="$account" --chdir="$root" \
  --job-name="sudo-gci" --output="$logs/%x-%j.out" \
  --cpus-per-task=1 --mem=16G --time=0-01:00:00 --dependency="$dependency" \
  "$root/validation/sudo_bend/slurm/gci.sbatch")
echo "gci: job ${id%%;*}, after ${ids[*]}"
echo "watch: squeue -u \$USER; tail -f $logs/sudo-*.out"
