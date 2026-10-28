#!/bin/bash
# The grid convergence index of the three converged bends. Submitted by submit.sh to start once the
# three grid jobs have succeeded; run from the repository root.
#SBATCH --job-name=sudo-gci
set -euo pipefail
source validation/sudo_bend/slurm/environment.sh
exec python -u validation/sudo_bend/gci.py
