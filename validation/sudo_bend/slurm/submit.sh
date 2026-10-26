#!/bin/bash
# Submit the bend's grid study to Slurm: one job per grid, then the GCI once all three succeed.
#
#   validation/sudo_bend/slurm/submit.sh <account> [grid ...]
#
# <account> is the allocation to charge (def-<pi>, rrg-<pi>, ...); the grids default to all three.
# Run from anywhere in the repository; every job runs from its root and writes into the grids'
# work/ directories, its own output to validation/sudo_bend/slurm/logs/<name>-<job id>.out. A grid job
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

ids=()
for grid in "${grids[@]}"; do
  [ -n "${cores[$grid]-}" ] || { echo "unknown grid $grid (round36, round48, round64)" >&2; exit 2; }
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
