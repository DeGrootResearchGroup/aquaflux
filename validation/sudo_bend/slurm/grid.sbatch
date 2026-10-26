#!/bin/bash
# One grid of the bend's grid study: its meshes, its precursor and its bend (run_study.py), resuming
# whatever an earlier job left done. Submitted by submit.sh, which sets the grid (SUDO_GRIDS), the
# cores, memory and time; run from the repository root.
#SBATCH --job-name=sudo-bend
set -euo pipefail
source validation/sudo_bend/slurm/environment.sh
export SUDO_GCI=0  # the GCI job waits for all three grids
exec python -u validation/sudo_bend/run_study.py
