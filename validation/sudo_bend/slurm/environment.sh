# Sourced by the study's batch jobs: the Python environment aquaflux is installed in, and one thread
# pool per job sized to the cores Slurm gave it.
#
#   SUDO_MODULES -- modules to load first (default "python/3.12"; empty for none)
#   SUDO_VENV    -- the virtual environment holding aquaflux (default ~/venvs/aquaflux)

if [ -n "${SUDO_MODULES-python/3.12}" ] && command -v module >/dev/null 2>&1; then
  module load ${SUDO_MODULES-python/3.12}
fi
# shellcheck disable=SC1091
source "${SUDO_VENV:-$HOME/venvs/aquaflux}/bin/activate"

# The jobs ask for no GPU; an environment with JAX's CUDA plugin would otherwise probe for one and
# print a traceback (CUDA_ERROR_NO_DEVICE) before falling back to the CPU.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Numba and the BLAS libraries size their pools by the node's cores, not the job's, unless told.
threads="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export NUMBA_NUM_THREADS="$threads"
echo "[$(date +%T)] job ${SLURM_JOB_ID:-?} on $(hostname): ${threads} cores, --mem ${SLURM_MEM_PER_NODE:-?} MB;" \
  "$(python --version 2>&1), aquaflux $(python -c 'import aquaflux; print(aquaflux.__version__)')," \
  "commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
